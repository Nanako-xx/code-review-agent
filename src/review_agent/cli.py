from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

from review_agent.agent_loop import agent_loop_run_to_dict, run_reviewer_agent_loop
from review_agent.brief import build_review_brief, review_brief_to_dict
from review_agent.checkpoint import CheckpointStore, _atomic_write_text
from review_agent.completion import check_completion, completion_to_dict
from review_agent.evidence import reconcile_evidence, reconciliation_to_dict
from review_agent.final_risk import final_risk_to_dict, reassess_final_risk
from review_agent.git_repo import ChangeSummary, collect_change_summary
from review_agent.intent import build_intent_packet
from review_agent.model_adapter_factory import (
    AdapterConfigError,
    ModelAdapterConfig,
    ModelAdapterFactory,
    build_model_adapter_factory_from_config,
)
from review_agent.models import IntentPacket, QualityGateResult, ReviewRequest, RiskAssessment
from review_agent.observations import ObservationStore
from review_agent.orchestrator import (
    MultiReviewerRun,
    ReviewerExecution,
    multi_reviewer_run_to_dict,
    run_multi_reviewer,
)
from review_agent.quality import detect_quality_gates, run_python_compile_gate
from review_agent.repository_intelligence import (
    build_repository_intelligence,
    repository_intelligence_raw_json,
    repository_intelligence_to_dict,
    summarize_repository_intelligence,
)
from review_agent.reporting import render_review_brief_markdown
from review_agent.revision import RevisionResolver
from review_agent.risk import LocalRiskAssessor, build_risk_packet
from review_agent.reviewer import reviewer_result_to_dict, run_single_reviewer
from review_agent.run_state import (
    RunPhase,
    RunState,
    RunStatus,
    advance_run_state,
    fail_run_state,
    initial_run_state,
)
from review_agent.runtime import build_assignments
from review_agent.session import (
    SESSION_PHASES,
    PhaseStatus,
    ReviewExecutionConfig,
    initial_session_manifest,
)
from review_agent.session_store import SessionStore
from review_agent.tool_gateway import ToolGateway


_ARTIFACT_SCHEMAS = {
    "request": "review_request_v1",
    "intent": "intent_packet_v1",
    "risk_packet": "risk_packet_v1",
    "risk": "risk_assessment_v1",
    "assignments": "reviewer_assignments_v1",
    "quality_gates": "quality_gate_results_v1",
    "repository_intelligence": "repository_intelligence_v1",
    "multi_reviewer": "multi_reviewer_result_v1",
    "reviewer_envelope": "model_request_envelope_v1",
    "reviewer_raw_response": "model_raw_response_v1",
    "reviewer": "reviewer_result_v1",
    "reviewer_agent_trace": "reviewer_agent_trace_v1",
    "reconciliation": "evidence_reconciliation_v1",
    "completion": "completion_check_v1",
    "final_risk": "final_risk_assessment_v1",
    "review_brief": "review_brief_v1",
    "report": "review_report_markdown_v1",
    "observations": "observation_log_jsonl_v1",
}

_PER_REVIEWER_SCHEMAS = {
    "_envelope": "model_request_envelope_v1",
    "_raw_response": "model_raw_response_v1",
    "_result": "reviewer_result_v1",
    "_agent_trace": "reviewer_agent_trace_v1",
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "review":
        return _run_review(args)
    if args.command == "resume":
        return _run_resume(args)
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="review-agent")
    subparsers = parser.add_subparsers(dest="command")

    review = subparsers.add_parser("review")
    review.add_argument("--repo", default=".")
    review.add_argument("--base", required=True)
    review.add_argument("--head", required=True)
    review.add_argument("--intent")
    review.add_argument("--focus")
    review.add_argument("--non-interactive", action="store_true")
    review.add_argument("--reviewer-provider", choices=["none", "fake", "openai-compatible"], default="none")
    review.add_argument("--reviewer-mode", choices=["single", "multi"], default="single")
    review.add_argument("--reviewer-loop", choices=["single-shot", "agent-loop"], default="single-shot")
    review.add_argument("--reviewer-model")
    review.add_argument("--reviewer-base-url")
    review.add_argument("--reviewer-api-key-env", default="REVIEW_AGENT_API_KEY")

    resume = subparsers.add_parser("resume", help="Inspect and resume from a local review checkpoint")
    resume.add_argument("review_id", help="Review id under .review-agent/runs")
    resume.add_argument("--repo", default=".", help="Repository path")

    return parser


def _run_review(args: argparse.Namespace) -> int:
    requested_repo = Path(args.repo).resolve()
    resolver = RevisionResolver()
    try:
        repository_identity = resolver.repository_identity(requested_repo)
        repo = Path(repository_identity.canonical_path)
        revisions = resolver.resolve_pair(repo, args.base, args.head)
    except Exception as error:
        print(f"Review failed: unable to resolve revisions: {error}", file=sys.stderr)
        return 1

    try:
        execution_config = ReviewExecutionConfig(
            reviewer_provider=args.reviewer_provider,
            reviewer_model=args.reviewer_model,
            reviewer_base_url=args.reviewer_base_url,
            reviewer_api_key_env=args.reviewer_api_key_env,
            reviewer_mode=args.reviewer_mode,
            reviewer_loop=args.reviewer_loop,
            non_interactive=args.non_interactive,
        )
    except ValueError as error:
        print(f"Reviewer session configuration error: {error}")
        return 2

    review_id = f"review-{uuid.uuid4().hex[:12]}"
    try:
        store = CheckpointStore(repo, review_id)
        session_store = SessionStore(store.run_dir)
        session_store.create(
            initial_session_manifest(
                review_id=review_id,
                repository=repository_identity,
                revisions=revisions,
                execution=execution_config,
                now=_utc_now(),
            )
        )
    except Exception as error:
        print(
            "Review failed: unable to create review Session: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    revision_binding = f"{revisions.resolved_base_sha}..{revisions.resolved_head_sha}"
    request = ReviewRequest(
        repository_path=str(repo),
        base_revision=revisions.requested_base,
        head_revision=revisions.requested_head,
        user_intent=args.intent,
        review_focus=args.focus,
    )
    state = initial_run_state(
        review_id=review_id,
        repository_path=str(repo),
        base_revision=revisions.requested_base,
        head_revision=revisions.requested_head,
        resolved_base_revision=revisions.resolved_base_sha,
        resolved_head_revision=revisions.resolved_head_sha,
    )
    try:
        store.write_state(state)
        store.write_json("request.json", asdict(request))
        session_store.register_existing_artifact(
            name="request",
            relative_path="request.json",
            schema=_artifact_schema("request"),
            phase=RunPhase.PREFLIGHT,
            revision_binding=None,
            now=_utc_now(),
        )
        state = advance_run_state(
            state,
            phase=RunPhase.CREATED,
            message="Request checkpointed",
            artifacts={"request": "request.json"},
        )
        store.write_state(state)

        change_summary = collect_change_summary(
            repo,
            revisions.resolved_base_sha,
            revisions.resolved_head_sha,
        )
        intent = build_intent_packet(request, change_summary)
        gates = detect_quality_gates(repo, revision=revisions.resolved_head_sha)
        quality_results = []
        if "python_compile" in gates:
            quality_results.append(
                run_python_compile_gate(
                    repo,
                    revision=revisions.resolved_head_sha,
                )
            )
        quality_status = {result.name: result.status for result in quality_results}
        risk_packet = build_risk_packet(change_summary, intent, quality_status)
        risk_assessment = LocalRiskAssessor().assess(risk_packet)
        assignments = [
            replace(
                assignment,
                initial_context=replace(
                    assignment.initial_context,
                    changed_files=list(change_summary.changed_files),
                ),
            )
            for assignment in build_assignments(risk_assessment)
        ]

        preflight_artifacts = {
            "request": "request.json",
            "intent": "intent.json",
            "risk_packet": "risk_packet.json",
            "risk": "risk.json",
            "assignments": "assignments.json",
            "quality_gates": "quality_gates.json",
        }
        store.write_json("intent.json", asdict(intent))
        store.write_json("risk_packet.json", asdict(risk_packet))
        store.write_json("risk.json", asdict(risk_assessment))
        store.write_json(
            "assignments.json",
            {"assignments": [asdict(item) for item in assignments]},
        )
        store.write_json(
            "quality_gates.json",
            {"results": [asdict(item) for item in quality_results]},
        )
        _print_preflight_summary(
            review_id=review_id,
            repo=repo,
            requested_base_revision=revisions.requested_base,
            requested_head_revision=revisions.requested_head,
            resolved_base_revision=revisions.resolved_base_sha,
            resolved_head_revision=revisions.resolved_head_sha,
            change_summary=change_summary,
            intent_packet=intent,
            quality_results=quality_results,
            risk_assessment=risk_assessment,
            run_dir=store.run_dir,
        )
        _complete_session_phase(
            session_store,
            RunPhase.PREFLIGHT,
            preflight_artifacts,
            revision_binding,
        )
        state = advance_run_state(
            state,
            phase=RunPhase.PREFLIGHT,
            message="Preflight completed",
            artifacts=preflight_artifacts,
        )
        store.write_state(state)

        observation_store = ObservationStore(store.run_dir)
        repository_intelligence = build_repository_intelligence(
            repo=repo,
            base_revision=revisions.resolved_base_sha,
            head_revision=revisions.resolved_head_sha,
            changed_files=change_summary.changed_files,
        )
        repository_intelligence_summary = summarize_repository_intelligence(
            repository_intelligence
        )
        store.write_json(
            "repository_intelligence.json",
            repository_intelligence_to_dict(repository_intelligence),
        )
        observation_store.record(
            source="repo_intelligence.snapshot",
            revision=revision_binding,
            path=None,
            line_start=None,
            line_end=None,
            raw_content=repository_intelligence_raw_json(repository_intelligence),
            context_view=repository_intelligence_summary,
        )
        repository_artifacts = {
            "repository_intelligence": "repository_intelligence.json"
        }
        _complete_session_phase(
            session_store,
            RunPhase.REPOSITORY_INTELLIGENCE,
            repository_artifacts,
            revision_binding,
        )
        state = advance_run_state(
            state,
            phase=RunPhase.REPOSITORY_INTELLIGENCE,
            message="Repository intelligence collected",
            artifacts=repository_artifacts,
        )
        store.write_state(state)

        try:
            adapter_factory: ModelAdapterFactory | None = (
                build_model_adapter_factory_from_config(
                    ModelAdapterConfig(
                        provider_name=args.reviewer_provider,
                        model=args.reviewer_model,
                        base_url=args.reviewer_base_url,
                        api_key_env=args.reviewer_api_key_env,
                    )
                )
            )
        except AdapterConfigError as error:
            error_text = f"{type(error).__name__}: {error}"
            _record_failed_review(
                session_store,
                store,
                state,
                message="Review failed before reviewer execution",
                error=error_text,
            )
            print(f"Reviewer adapter configuration error: {error}")
            return 2
        if args.reviewer_mode == "multi" and adapter_factory is None:
            error_text = (
                "--reviewer-mode multi requires --reviewer-provider "
                "fake or openai-compatible"
            )
            _record_failed_review(
                session_store,
                store,
                state,
                message="Review failed before reviewer execution",
                error=error_text,
            )
            print(error_text)
            return 2

        reviewer_result = None
        multi_reviewer_summary = None
        reconciliation_payload = None
        reconciliation_summary = None
        completion_payload = None
        completion_summary = None
        completion_executions: list[ReviewerExecution] | None = None
        completion_reconciliation = None
        multi_run: MultiReviewerRun | None = None
        reviewer_artifacts: dict[str, str] = {}
        loop_runs = []
        reviewer_invocation_model = _reviewer_invocation_model(args)

        if adapter_factory is not None and assignments:
            gateway = ToolGateway(
                repository_path=repo,
                base_revision=revisions.resolved_base_sha,
                head_revision=revisions.resolved_head_sha,
                observation_store=observation_store,
            )
            for changed_file in change_summary.changed_files:
                gateway.execute("compare_base_head", {"path": changed_file})
            reviewer_observations = observation_store.summaries_by_id()

            if args.reviewer_mode == "multi":
                if args.reviewer_loop == "single-shot":
                    multi_run = run_multi_reviewer(
                        adapter_factory=adapter_factory,
                        assignments=assignments,
                        intent=intent,
                        diff_excerpt=change_summary.diff_excerpt,
                        observations=reviewer_observations,
                        trace_id_prefix=review_id,
                        model=reviewer_invocation_model,
                    )
                else:
                    executions = []
                    for index, assignment in enumerate(assignments):
                        trace_id = f"{review_id}-reviewer-{index}"
                        loop_run = run_reviewer_agent_loop(
                            adapter=adapter_factory.create(),
                            gateway=gateway,
                            assignment=assignment,
                            intent=intent,
                            diff_excerpt=change_summary.diff_excerpt,
                            observations=reviewer_observations,
                            trace_id=trace_id,
                            model=reviewer_invocation_model,
                        )
                        loop_runs.append(loop_run)
                        executions.append(
                            ReviewerExecution(
                                reviewer_index=index,
                                trace_id=trace_id,
                                assignment=assignment,
                                envelope=loop_run.envelope,
                                response=loop_run.response,
                                result=loop_run.result,
                            )
                        )
                    multi_run = MultiReviewerRun(executions=executions)

                multi_payload = multi_reviewer_run_to_dict(multi_run)
                store.write_json("multi_reviewer_result.json", multi_payload)
                reviewer_artifacts["multi_reviewer"] = "multi_reviewer_result.json"
                for execution in multi_run.executions:
                    index = execution.reviewer_index
                    envelope_name = f"reviewer_{index}_envelope"
                    raw_name = f"reviewer_{index}_raw_response"
                    result_name = f"reviewer_{index}_result"
                    store.write_json(
                        f"{envelope_name}.json",
                        asdict(execution.envelope),
                    )
                    store.write_json(
                        f"{raw_name}.json",
                        {
                            "provider_name": execution.response.provider_name,
                            "model": execution.response.model,
                            "content": execution.response.content,
                            "raw": execution.response.raw,
                        },
                    )
                    store.write_json(
                        f"{result_name}.json",
                        reviewer_result_to_dict(execution.result),
                    )
                    reviewer_artifacts[envelope_name] = f"{envelope_name}.json"
                    reviewer_artifacts[raw_name] = f"{raw_name}.json"
                    reviewer_artifacts[result_name] = f"{result_name}.json"
                    if args.reviewer_loop == "agent-loop":
                        trace_name = f"reviewer_{index}_agent_trace"
                        store.write_json(
                            f"{trace_name}.json",
                            agent_loop_run_to_dict(loop_runs[index])["trace"],
                        )
                        reviewer_artifacts[trace_name] = f"{trace_name}.json"
                multi_reviewer_summary = {
                    "reviewer_count": multi_payload["reviewer_count"],
                    "status_counts": multi_payload["status_counts"],
                    "roles": [item["role"] for item in multi_payload["executions"]],
                }
            elif args.reviewer_loop == "single-shot":
                reviewer_run = run_single_reviewer(
                    adapter=adapter_factory.create(),
                    assignment=assignments[0],
                    intent=intent,
                    diff_excerpt=change_summary.diff_excerpt,
                    observations=reviewer_observations,
                    trace_id=f"{review_id}-reviewer-0",
                    model=reviewer_invocation_model,
                )
                reviewer_result = reviewer_run.result
                store.write_json("reviewer_envelope.json", asdict(reviewer_run.envelope))
                store.write_json(
                    "reviewer_raw_response.json",
                    {
                        "provider_name": reviewer_run.response.provider_name,
                        "model": reviewer_run.response.model,
                        "content": reviewer_run.response.content,
                        "raw": reviewer_run.response.raw,
                    },
                )
                store.write_json(
                    "reviewer_result.json",
                    reviewer_result_to_dict(reviewer_run.result),
                )
                reviewer_artifacts = {
                    "reviewer_envelope": "reviewer_envelope.json",
                    "reviewer_raw_response": "reviewer_raw_response.json",
                    "reviewer": "reviewer_result.json",
                }
            else:
                loop_run = run_reviewer_agent_loop(
                    adapter=adapter_factory.create(),
                    gateway=gateway,
                    assignment=assignments[0],
                    intent=intent,
                    diff_excerpt=change_summary.diff_excerpt,
                    observations=reviewer_observations,
                    trace_id=f"{review_id}-reviewer-0",
                    model=reviewer_invocation_model,
                )
                reviewer_result = loop_run.result
                store.write_json("reviewer_envelope.json", asdict(loop_run.envelope))
                store.write_json(
                    "reviewer_raw_response.json",
                    {
                        "provider_name": loop_run.response.provider_name,
                        "model": loop_run.response.model,
                        "content": loop_run.response.content,
                        "raw": loop_run.response.raw,
                    },
                )
                store.write_json(
                    "reviewer_result.json",
                    reviewer_result_to_dict(loop_run.result),
                )
                store.write_json(
                    "reviewer_agent_trace.json",
                    agent_loop_run_to_dict(loop_run)["trace"],
                )
                reviewer_artifacts = {
                    "reviewer_envelope": "reviewer_envelope.json",
                    "reviewer_raw_response": "reviewer_raw_response.json",
                    "reviewer": "reviewer_result.json",
                    "reviewer_agent_trace": "reviewer_agent_trace.json",
                }

        _complete_session_phase(
            session_store,
            RunPhase.REVIEWERS,
            reviewer_artifacts,
            revision_binding,
        )
        state = advance_run_state(
            state,
            phase=RunPhase.REVIEWERS,
            message="Reviewer execution completed",
            artifacts=reviewer_artifacts,
        )
        store.write_state(state)

        reconciliation_artifacts: dict[str, str] = {}
        if multi_run is not None:
            reconciliation = reconcile_evidence(
                executions=multi_run.executions,
                authorized_observation_ids=set(observation_store.summaries_by_id()),
            )
            reconciliation_payload = reconciliation_to_dict(reconciliation)
            store.write_json("reconciliation.json", reconciliation_payload)
            reconciliation_artifacts["reconciliation"] = "reconciliation.json"
            reconciliation_summary = {
                "canonical_count": len(reconciliation.canonical_findings),
                "rejected_count": len(reconciliation.rejected_findings),
                "evidence_quality": reconciliation.evidence_quality,
            }
            completion_executions = multi_run.executions
            completion_reconciliation = reconciliation
            initial_completion = check_completion(
                intent=intent,
                quality_results=quality_results,
                executions=multi_run.executions,
                reconciliation=reconciliation,
            )
            completion_payload = completion_to_dict(initial_completion)
        _complete_session_phase(
            session_store,
            RunPhase.RECONCILIATION,
            reconciliation_artifacts,
            revision_binding,
        )
        state = advance_run_state(
            state,
            phase=RunPhase.RECONCILIATION,
            message="Evidence reconciliation completed",
            artifacts=reconciliation_artifacts,
        )
        store.write_state(state)

        final_risk = reassess_final_risk(
            initial_risk=risk_assessment,
            intent_packet=intent,
            quality_results=quality_results,
            reviewer_result=reviewer_result,
            reconciliation_payload=reconciliation_payload,
            completion_summary=completion_payload,
        )
        final_risk_payload = final_risk_to_dict(final_risk)

        completion_artifacts: dict[str, str] = {}
        if completion_executions is not None and completion_reconciliation is not None:
            completion = check_completion(
                intent=intent,
                quality_results=quality_results,
                executions=completion_executions,
                reconciliation=completion_reconciliation,
                require_final_risk=True,
                final_risk_level=final_risk.level.value,
            )
            completion_payload = completion_to_dict(completion)
            completion_summary = completion_payload
            store.write_json("completion.json", completion_payload)
            completion_artifacts["completion"] = "completion.json"
        _complete_session_phase(
            session_store,
            RunPhase.COMPLETION,
            completion_artifacts,
            revision_binding,
        )
        state = advance_run_state(
            state,
            phase=RunPhase.COMPLETION,
            message="Completion check completed",
            artifacts=completion_artifacts,
        )
        store.write_state(state)

        store.write_json("final_risk.json", final_risk_payload)
        final_risk_artifacts = {"final_risk": "final_risk.json"}
        _complete_session_phase(
            session_store,
            RunPhase.FINAL_RISK,
            final_risk_artifacts,
            revision_binding,
        )
        state = advance_run_state(
            state,
            phase=RunPhase.FINAL_RISK,
            message="Final risk reassessment completed",
            artifacts=final_risk_artifacts,
        )
        store.write_state(state)

        brief = build_review_brief(
            review_id=review_id,
            base_revision=revisions.resolved_base_sha,
            head_revision=revisions.resolved_head_sha,
            intent_packet=intent,
            risk_assessment=risk_assessment,
            changed_files=change_summary.changed_files,
            quality_results=quality_results,
            reviewer_result=reviewer_result,
            observation_summaries=observation_store.summaries_by_id(),
            repository_intelligence_summary=repository_intelligence_summary,
            multi_reviewer_summary=multi_reviewer_summary,
            reconciliation_payload=reconciliation_payload,
            completion_summary=completion_summary,
            final_risk_assessment=final_risk_payload,
        )
        store.write_json("review_brief.json", review_brief_to_dict(brief))
        _atomic_write_text(
            store.run_dir / "report.md",
            render_review_brief_markdown(brief),
        )
        reporting_artifacts = {
            "review_brief": "review_brief.json",
            "report": "report.md",
            "observations": "observations.jsonl",
        }
        _complete_session_phase(
            session_store,
            RunPhase.REPORTING,
            reporting_artifacts,
            revision_binding,
        )
        session_store.mark_session_completed(_utc_now())
        try:
            completed_state = advance_run_state(
                state,
                phase=RunPhase.COMPLETED,
                message="Review completed",
                artifacts=reporting_artifacts,
            )
            store.write_state(completed_state)
            state = completed_state
        except Exception as state_error:
            _print_recording_warning(
                "unable to write completed legacy state summary",
                state_error,
            )
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
        _record_failed_review(
            session_store,
            store,
            state,
            message="Review failed",
            error=error_text,
        )
        print(f"Review failed: {error}", file=sys.stderr)
        return 1

    print(f"Review foundation completed: {store.run_dir}")
    print(f"Requested Base: {revisions.requested_base}")
    print(f"Requested Head: {revisions.requested_head}")
    print(f"Resolved Base: {revisions.resolved_base_sha}")
    print(f"Resolved Head: {revisions.resolved_head_sha}")
    print(f"Final risk: {final_risk.level.value}")
    print(f"Review brief: {store.run_dir / 'report.md'}")
    print(f"Review brief JSON: {store.run_dir / 'review_brief.json'}")
    print(f"Recommendation: {brief.non_binding_recommendation}")
    print(f"Remaining uncertainties: {len(brief.uncertainties)}")
    return 0


def _run_resume(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    store = CheckpointStore(repo, args.review_id, create=False)

    if not store.run_dir.exists():
        print(f"Review run not found: {store.run_dir}", file=sys.stderr)
        return 2

    session_path = store.run_dir / "session.json"
    if session_path.exists():
        session_store = SessionStore(store.run_dir)
        try:
            session = session_store.load()
        except Exception as error:
            print(f"Review run has an invalid session.json: {error}", file=sys.stderr)
            return 2

        print("Resume")
        print(f"  Review ID: {session.review_id}")
        print(f"  Status: {session.status.value}")
        print(f"  Phase: {session.current_phase.value}")
        print(f"  Repository: {session.repository.canonical_path}")
        print(f"  Requested Base: {session.revisions.requested_base}")
        print(f"  Requested Head: {session.revisions.requested_head}")
        print(f"  Resolved Base: {session.revisions.resolved_base_sha}")
        print(f"  Resolved Head: {session.revisions.resolved_head_sha}")
        print(f"  Run directory: {store.run_dir}")
        print("  Artifacts:")
        for name, descriptor in sorted(session.artifacts.items()):
            artifact_path = store.run_dir / descriptor.path
            if not artifact_path.exists():
                marker = "missing"
            elif session_store.validate_artifact(descriptor):
                marker = "present"
            else:
                marker = "invalid"
            print(f"    - {name}: {descriptor.path} ({marker})")
        if session.errors:
            print("  Errors:")
            for error in session.errors:
                print(f"    - {error}")
        if session.status is RunStatus.COMPLETED:
            try:
                session_store.mark_session_completed(_utc_now())
            except Exception as audit_error:
                print("  Audit: invalid")
                print(f"  Audit error: {audit_error}")
                return 2
            print("  Audit: valid")
        return 0

    state_path = store.run_dir / "state.json"
    request_path = store.run_dir / "request.json"
    if not state_path.exists():
        print(f"Review run has no state.json: {store.run_dir}", file=sys.stderr)
        return 2
    if not request_path.exists():
        print(f"Review run has no request.json: {store.run_dir}", file=sys.stderr)
        return 2

    state = store.read_state()
    request = store.read_json("request.json")

    print("Resume")
    print(f"  Review ID: {state.review_id}")
    print(f"  Status: {state.status.value}")
    print(f"  Phase: {state.phase.value}")
    print(f"  Repository: {state.repository_path}")
    print(f"  Base: {state.base_revision}")
    print(f"  Head: {state.head_revision}")
    print(f"  Message: {state.message}")
    print(f"  Run directory: {store.run_dir}")
    request_repository = request.get("repository_path")
    if request_repository and str(request_repository) != state.repository_path:
        print(f"  Request repository: {request_repository}")
    print("  Artifacts:")
    for name, relative_path in sorted(state.artifacts.items()):
        marker = "present" if (store.run_dir / relative_path).exists() else "missing"
        print(f"    - {name}: {relative_path} ({marker})")

    if state.errors:
        print("  Errors:")
        for error in state.errors:
            print(f"    - {error}")

    return 0


def _record_failed_review_state(store: CheckpointStore, state: RunState, *, message: str, error: str) -> None:
    store.write_state(fail_run_state(state, message=message, error=error))


def _record_failed_review(
    session_store: SessionStore,
    store: CheckpointStore,
    state: RunState,
    *,
    message: str,
    error: str,
) -> None:
    try:
        manifest = session_store.load()
    except Exception as session_error:
        _print_recording_warning(
            "unable to load Session while recording failure",
            session_error,
        )
    else:
        if manifest.status is RunStatus.COMPLETED:
            return
        phase = next(
            (
                candidate
                for candidate in SESSION_PHASES
                if manifest.phases[candidate.value].status
                is not PhaseStatus.COMPLETED
            ),
            None,
        )
        if phase is None:
            _print_recording_warning(
                "Session finalization remains retryable; no failed phase or "
                "legacy failed state was fabricated",
            )
            return
        try:
            session_store.mark_session_failed(phase, error, _utc_now())
        except Exception as session_error:
            _print_recording_warning(
                "unable to mark Session phase failed",
                session_error,
            )

    try:
        _record_failed_review_state(store, state, message=message, error=error)
    except Exception as state_error:
        _print_recording_warning(
            "unable to write legacy failed state",
            state_error,
        )


def _print_recording_warning(
    message: str,
    error: Exception | None = None,
) -> None:
    detail = f": {type(error).__name__}: {error}" if error is not None else ""
    print(f"Warning: {message}{detail}", file=sys.stderr)


def _complete_session_phase(
    session_store: SessionStore,
    phase: RunPhase,
    artifacts: dict[str, str],
    revision_binding: str,
) -> None:
    now = _utc_now()
    for name, relative_path in artifacts.items():
        session_store.register_existing_artifact(
            name=name,
            relative_path=relative_path,
            schema=_artifact_schema(name),
            phase=phase,
            revision_binding=None if name == "request" else revision_binding,
            now=now,
        )
    session_store.mark_phase_completed(phase, artifacts, now)


def _artifact_schema(name: str) -> str:
    schema = _ARTIFACT_SCHEMAS.get(name)
    if schema is not None:
        return schema
    if name.startswith("reviewer_"):
        for suffix, reviewer_schema in _PER_REVIEWER_SCHEMAS.items():
            reviewer_number = name[len("reviewer_") : -len(suffix)]
            if name.endswith(suffix) and reviewer_number.isdigit():
                return reviewer_schema
    raise ValueError(f"No stable artifact schema is defined for: {name}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _print_preflight_summary(
    *,
    review_id: str,
    repo: Path,
    requested_base_revision: str,
    requested_head_revision: str,
    resolved_base_revision: str,
    resolved_head_revision: str,
    change_summary: ChangeSummary,
    intent_packet: IntentPacket,
    quality_results: list[QualityGateResult],
    risk_assessment: RiskAssessment,
    run_dir: Path,
) -> None:
    print("Preflight")
    print(f"  Review ID: {review_id}")
    print(f"  Repository: {repo}")
    print(f"  Requested Base: {requested_base_revision}")
    print(f"  Requested Head: {requested_head_revision}")
    print(f"  Resolved Base: {resolved_base_revision}")
    print(f"  Resolved Head: {resolved_head_revision}")
    print(f"  Changed files: {len(change_summary.changed_files)}")
    print(f"  Intent status: {intent_packet.status.value}")
    print(f"  Risk level: {risk_assessment.level.value}")
    print(f"  Quality gates: {_format_quality_gate_summary(quality_results)}")
    print(f"  Run directory: {run_dir}")


def _format_quality_gate_summary(quality_results: list[QualityGateResult]) -> str:
    if not quality_results:
        return "none"
    return ", ".join(f"{result.name}={result.status}" for result in quality_results)


def _reviewer_invocation_model(args: argparse.Namespace) -> str:
    if args.reviewer_model:
        return str(args.reviewer_model)
    if args.reviewer_provider == "fake":
        return "fake-reviewer"
    if args.reviewer_provider == "none":
        return "none"
    return "configured-reviewer-model"

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
import uuid

from review_agent.agent_loop import agent_loop_run_to_dict, run_reviewer_agent_loop
from review_agent.checkpoint import CheckpointStore
from review_agent.completion import check_completion, completion_to_dict
from review_agent.evidence import reconcile_evidence, reconciliation_to_dict
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
from review_agent.reporting import render_markdown_report
from review_agent.risk import LocalRiskAssessor, build_risk_packet
from review_agent.reviewer import reviewer_result_to_dict, run_single_reviewer
from review_agent.run_state import RunPhase, RunState, advance_run_state, fail_run_state, initial_run_state
from review_agent.runtime import build_assignments
from review_agent.tool_gateway import ToolGateway


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
    repo = Path(args.repo).resolve()
    review_id = f"review-{uuid.uuid4().hex[:12]}"

    request = ReviewRequest(
        repository_path=str(repo),
        base_revision=args.base,
        head_revision=args.head,
        user_intent=args.intent,
        review_focus=args.focus,
    )
    store = CheckpointStore(repo, review_id)
    state = initial_run_state(
        review_id=review_id,
        repository_path=str(repo),
        base_revision=args.base,
        head_revision=args.head,
    )
    store.write_state(state)
    store.write_json("request.json", asdict(request))
    state = advance_run_state(
        state,
        phase=RunPhase.CREATED,
        message="Request checkpointed",
        artifacts={"request": "request.json"},
    )
    store.write_state(state)
    try:
        change_summary = collect_change_summary(repo, args.base, args.head)
    except Exception as error:
        _record_failed_review_state(store, state, message="Review failed", error=f"{type(error).__name__}: {error}")
        print(f"Review failed: {error}", file=sys.stderr)
        return 1
    intent = build_intent_packet(request, change_summary)

    gates = detect_quality_gates(repo)
    quality_results = []
    if "python_compile" in gates:
        quality_results.append(run_python_compile_gate(repo))
    quality_status = {result.name: result.status for result in quality_results}

    risk_packet = build_risk_packet(change_summary, intent, quality_status)
    risk_assessment = LocalRiskAssessor().assess(risk_packet)
    assignments = build_assignments(risk_assessment)
    assignments = [
        replace(
            assignment,
            initial_context=replace(
                assignment.initial_context,
                changed_files=list(change_summary.changed_files),
            ),
        )
        for assignment in assignments
    ]
    try:
        adapter_factory: ModelAdapterFactory | None = build_model_adapter_factory_from_config(
            ModelAdapterConfig(
                provider_name=args.reviewer_provider,
                model=args.reviewer_model,
                base_url=args.reviewer_base_url,
                api_key_env=args.reviewer_api_key_env,
            )
        )
    except AdapterConfigError as error:
        print(f"Reviewer adapter configuration error: {error}")
        _record_failed_review_state(
            store,
            state,
            message="Review failed before reviewer execution",
            error=f"{type(error).__name__}: {error}",
        )
        return 2
    if args.reviewer_mode == "multi" and adapter_factory is None:
        error_message = "--reviewer-mode multi requires --reviewer-provider fake or openai-compatible"
        print(error_message)
        _record_failed_review_state(
            store,
            state,
            message="Review failed before reviewer execution",
            error=error_message,
        )
        return 2

    observation_store = ObservationStore(store.run_dir)
    repository_intelligence = build_repository_intelligence(
        repo=repo,
        base_revision=args.base,
        head_revision=args.head,
        changed_files=change_summary.changed_files,
    )
    repository_intelligence_summary = summarize_repository_intelligence(repository_intelligence)
    store.write_json("intent.json", asdict(intent))
    store.write_json("risk_packet.json", asdict(risk_packet))
    store.write_json("risk.json", asdict(risk_assessment))
    store.write_json("assignments.json", {"assignments": [asdict(item) for item in assignments]})
    store.write_json("quality_gates.json", {"results": [asdict(item) for item in quality_results]})
    store.write_json("repository_intelligence.json", repository_intelligence_to_dict(repository_intelligence))
    _print_preflight_summary(
        review_id=review_id,
        repo=repo,
        base_revision=args.base,
        head_revision=args.head,
        change_summary=change_summary,
        intent_packet=intent,
        quality_results=quality_results,
        risk_assessment=risk_assessment,
        run_dir=store.run_dir,
    )
    state = advance_run_state(
        state,
        phase=RunPhase.PREFLIGHT,
        message="Preflight completed",
        artifacts={
            "request": "request.json",
            "intent": "intent.json",
            "risk_packet": "risk_packet.json",
            "risk": "risk.json",
            "assignments": "assignments.json",
            "quality_gates": "quality_gates.json",
        },
    )
    store.write_state(state)
    state = advance_run_state(
        state,
        phase=RunPhase.REPOSITORY_INTELLIGENCE,
        message="Repository intelligence collected",
        artifacts={"repository_intelligence": "repository_intelligence.json"},
    )
    store.write_state(state)
    observation_store.record(
        source="repo_intelligence.snapshot",
        revision=f"{args.base}..{args.head}",
        path=None,
        line_start=None,
        line_end=None,
        raw_content=repository_intelligence_raw_json(repository_intelligence),
        context_view=repository_intelligence_summary,
    )

    reviewer_result = None
    multi_reviewer_summary = None
    reconciliation_summary = None
    completion_summary = None
    if adapter_factory is not None and assignments:
        gateway = ToolGateway(
            repository_path=repo,
            base_revision=args.base,
            head_revision=args.head,
            observation_store=observation_store,
        )
        for changed_file in change_summary.changed_files:
            gateway.execute("compare_base_head", {"path": changed_file})
        reviewer_observations = observation_store.summaries_by_id()
        if args.reviewer_mode == "multi":
            loop_runs = []
            if args.reviewer_loop == "single-shot":
                multi_run = run_multi_reviewer(
                    adapter_factory=adapter_factory,
                    assignments=assignments,
                    intent=intent,
                    diff_excerpt=change_summary.diff_excerpt,
                    observations=reviewer_observations,
                    trace_id_prefix=review_id,
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
            state = advance_run_state(
                state,
                phase=RunPhase.REVIEWERS,
                message="Reviewer execution completed",
                artifacts={"multi_reviewer": "multi_reviewer_result.json"},
            )
            store.write_state(state)
            multi_reviewer_summary = {
                "reviewer_count": multi_payload["reviewer_count"],
                "status_counts": multi_payload["status_counts"],
                "roles": [item["role"] for item in multi_payload["executions"]],
            }
            reconciliation = reconcile_evidence(
                executions=multi_run.executions,
                authorized_observation_ids=set(observation_store.summaries_by_id()),
            )
            reconciliation_payload = reconciliation_to_dict(reconciliation)
            store.write_json("reconciliation.json", reconciliation_payload)
            state = advance_run_state(
                state,
                phase=RunPhase.RECONCILIATION,
                message="Evidence reconciliation completed",
                artifacts={"reconciliation": "reconciliation.json"},
            )
            store.write_state(state)
            reconciliation_summary = {
                "canonical_count": len(reconciliation.canonical_findings),
                "rejected_count": len(reconciliation.rejected_findings),
                "evidence_quality": reconciliation.evidence_quality,
            }
            completion = check_completion(
                intent=intent,
                quality_results=quality_results,
                executions=multi_run.executions,
                reconciliation=reconciliation,
            )
            completion_payload = completion_to_dict(completion)
            store.write_json("completion.json", completion_payload)
            state = advance_run_state(
                state,
                phase=RunPhase.COMPLETION,
                message="Completion check completed",
                artifacts={"completion": "completion.json"},
            )
            store.write_state(state)
            completion_summary = completion_payload
            for execution in multi_run.executions:
                index = execution.reviewer_index
                store.write_json(f"reviewer_{index}_envelope.json", asdict(execution.envelope))
                store.write_json(
                    f"reviewer_{index}_raw_response.json",
                    {
                        "provider_name": execution.response.provider_name,
                        "model": execution.response.model,
                        "content": execution.response.content,
                        "raw": execution.response.raw,
                    },
                )
                store.write_json(f"reviewer_{index}_result.json", reviewer_result_to_dict(execution.result))
                if args.reviewer_loop == "agent-loop":
                    store.write_json(
                        f"reviewer_{index}_agent_trace.json",
                        agent_loop_run_to_dict(loop_runs[index])["trace"],
                    )
        else:
            if args.reviewer_loop == "single-shot":
                reviewer_run = run_single_reviewer(
                    adapter=adapter_factory.create(),
                    assignment=assignments[0],
                    intent=intent,
                    diff_excerpt=change_summary.diff_excerpt,
                    observations=reviewer_observations,
                    trace_id=f"{review_id}-reviewer-0",
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
                store.write_json("reviewer_result.json", reviewer_result_to_dict(reviewer_run.result))
                state = advance_run_state(
                    state,
                    phase=RunPhase.REVIEWERS,
                    message="Reviewer execution completed",
                    artifacts={
                        "reviewer_envelope": "reviewer_envelope.json",
                        "reviewer_raw_response": "reviewer_raw_response.json",
                        "reviewer": "reviewer_result.json",
                    },
                )
                store.write_state(state)
            else:
                loop_run = run_reviewer_agent_loop(
                    adapter=adapter_factory.create(),
                    gateway=gateway,
                    assignment=assignments[0],
                    intent=intent,
                    diff_excerpt=change_summary.diff_excerpt,
                    observations=reviewer_observations,
                    trace_id=f"{review_id}-reviewer-0",
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
                store.write_json("reviewer_result.json", reviewer_result_to_dict(loop_run.result))
                store.write_json("reviewer_agent_trace.json", agent_loop_run_to_dict(loop_run)["trace"])
                state = advance_run_state(
                    state,
                    phase=RunPhase.REVIEWERS,
                    message="Reviewer execution completed",
                    artifacts={
                        "reviewer_envelope": "reviewer_envelope.json",
                        "reviewer_raw_response": "reviewer_raw_response.json",
                        "reviewer": "reviewer_result.json",
                        "reviewer_agent_trace": "reviewer_agent_trace.json",
                    },
                )
                store.write_state(state)

    report = render_markdown_report(
        review_id=review_id,
        base_revision=args.base,
        head_revision=args.head,
        risk_assessment=risk_assessment,
        changed_files=change_summary.changed_files,
        reviewer_result=reviewer_result,
        observation_summaries=observation_store.summaries_by_id(),
        repository_intelligence_summary=repository_intelligence_summary,
        multi_reviewer_summary=multi_reviewer_summary,
        reconciliation_summary=reconciliation_summary,
        completion_summary=completion_summary,
    )
    (store.run_dir / "report.md").write_text(report, encoding="utf-8")
    state = advance_run_state(
        state,
        phase=RunPhase.COMPLETED,
        message="Review completed",
        artifacts={"report": "report.md"},
    )
    store.write_state(state)

    print(f"Review foundation completed: {store.run_dir}")
    return 0


def _run_resume(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    store = CheckpointStore(repo, args.review_id, create=False)

    if not store.run_dir.exists():
        print(f"Review run not found: {store.run_dir}", file=sys.stderr)
        return 2

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


def _print_preflight_summary(
    *,
    review_id: str,
    repo: Path,
    base_revision: str,
    head_revision: str,
    change_summary: ChangeSummary,
    intent_packet: IntentPacket,
    quality_results: list[QualityGateResult],
    risk_assessment: RiskAssessment,
    run_dir: Path,
) -> None:
    print("Preflight")
    print(f"  Review ID: {review_id}")
    print(f"  Repository: {repo}")
    print(f"  Base: {base_revision}")
    print(f"  Head: {head_revision}")
    print(f"  Changed files: {len(change_summary.changed_files)}")
    print(f"  Intent status: {intent_packet.status.value}")
    print(f"  Risk level: {risk_assessment.level.value}")
    print(f"  Quality gates: {_format_quality_gate_summary(quality_results)}")
    print(f"  Run directory: {run_dir}")


def _format_quality_gate_summary(quality_results: list[QualityGateResult]) -> str:
    if not quality_results:
        return "none"
    return ", ".join(f"{result.name}={result.status}" for result in quality_results)

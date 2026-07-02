from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import uuid

from review_agent.agent_loop import agent_loop_run_to_dict, run_reviewer_agent_loop
from review_agent.checkpoint import CheckpointStore
from review_agent.completion import check_completion, completion_to_dict
from review_agent.evidence import reconcile_evidence, reconciliation_to_dict
from review_agent.git_repo import collect_change_summary
from review_agent.intent import build_intent_packet
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import ModelResponseKind, ModelToolCall, ModelTurnResponse
from review_agent.models import ReviewRequest
from review_agent.observations import ObservationStore
from review_agent.orchestrator import (
    MultiReviewerRun,
    ReviewerExecution,
    multi_reviewer_run_to_dict,
    run_multi_reviewer,
)
from review_agent.provider import ProviderConfigError, build_provider_from_config
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
from review_agent.runtime import build_assignments
from review_agent.tool_gateway import ToolGateway


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "review":
        return _run_review(args)
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

    return parser


def _fake_agent_loop_adapter(path: str) -> FakeToolCallingAdapter:
    def final_response(request):
        observation_id = request.tool_results[-1].observation_ids[0] if request.tool_results else ""
        return ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text=json.dumps(
                {
                    "contract_assessments": [
                        {
                            "contract": "regression_safety",
                            "status": "covered",
                            "summary": "Fake agent loop used a tool observation.",
                            "evidence_refs": [observation_id] if observation_id else [],
                        }
                    ],
                    "confirmed_findings": [],
                    "rejected_hypotheses": [],
                    "uncertainties": [],
                    "observation_refs": [observation_id] if observation_id else [],
                    "investigation_summary": "Fake agent loop reviewer executed.",
                    "status": "completed",
                }
            ),
        )

    return FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[ModelToolCall("call-1", "compare_base_head", {"path": path})],
            ),
            final_response,
        ]
    )


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
    change_summary = collect_change_summary(repo, args.base, args.head)
    intent = build_intent_packet(request, change_summary)

    gates = detect_quality_gates(repo)
    quality_results = []
    if "python_compile" in gates:
        quality_results.append(run_python_compile_gate(repo))
    quality_status = {result.name: result.status for result in quality_results}

    risk_packet = build_risk_packet(change_summary, intent, quality_status)
    risk_assessment = LocalRiskAssessor().assess(risk_packet)
    assignments = build_assignments(risk_assessment)
    try:
        provider = build_provider_from_config(
            provider_name=args.reviewer_provider,
            model=args.reviewer_model,
            base_url=args.reviewer_base_url,
            api_key_env=args.reviewer_api_key_env,
        )
    except ProviderConfigError as error:
        print(f"Reviewer provider configuration error: {error}")
        return 2
    if args.reviewer_mode == "multi" and provider is None:
        print("--reviewer-mode multi requires --reviewer-provider fake or openai-compatible")
        return 2
    if args.reviewer_loop == "agent-loop" and args.reviewer_provider != "fake":
        print("--reviewer-loop agent-loop currently requires --reviewer-provider fake")
        return 2

    store = CheckpointStore(repo, review_id)
    observation_store = ObservationStore(store.run_dir)
    repository_intelligence = build_repository_intelligence(
        repo=repo,
        base_revision=args.base,
        head_revision=args.head,
        changed_files=change_summary.changed_files,
    )
    repository_intelligence_summary = summarize_repository_intelligence(repository_intelligence)
    store.write_json("request.json", asdict(request))
    store.write_json("intent.json", asdict(intent))
    store.write_json("risk_packet.json", asdict(risk_packet))
    store.write_json("risk.json", asdict(risk_assessment))
    store.write_json("assignments.json", {"assignments": [asdict(item) for item in assignments]})
    store.write_json("quality_gates.json", {"results": [asdict(item) for item in quality_results]})
    store.write_json("repository_intelligence.json", repository_intelligence_to_dict(repository_intelligence))
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
    if provider is not None and assignments:
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
                    provider=provider,
                    assignments=assignments,
                    intent=intent,
                    diff_excerpt=change_summary.diff_excerpt,
                    observations=reviewer_observations,
                    trace_id_prefix=review_id,
                )
            else:
                executions = []
                agent_loop_path = change_summary.changed_files[0] if change_summary.changed_files else "."
                for index, assignment in enumerate(assignments):
                    trace_id = f"{review_id}-reviewer-{index}"
                    loop_run = run_reviewer_agent_loop(
                        adapter=_fake_agent_loop_adapter(agent_loop_path),
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
                    provider=provider,
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
            else:
                agent_loop_path = change_summary.changed_files[0] if change_summary.changed_files else "."
                loop_run = run_reviewer_agent_loop(
                    adapter=_fake_agent_loop_adapter(agent_loop_path),
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

    print(f"Review foundation completed: {store.run_dir}")
    return 0

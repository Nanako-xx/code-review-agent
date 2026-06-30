from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import uuid

from review_agent.checkpoint import CheckpointStore
from review_agent.git_repo import collect_change_summary
from review_agent.intent import build_intent_packet
from review_agent.models import ReviewRequest
from review_agent.observations import ObservationStore
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
    review.add_argument("--reviewer-model")
    review.add_argument("--reviewer-base-url")
    review.add_argument("--reviewer-api-key-env", default="REVIEW_AGENT_API_KEY")

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
    if provider is not None and assignments:
        gateway = ToolGateway(
            repository_path=repo,
            base_revision=args.base,
            head_revision=args.head,
            observation_store=observation_store,
        )
        for changed_file in change_summary.changed_files:
            gateway.execute("compare_base_head", {"path": changed_file})
        reviewer_run = run_single_reviewer(
            provider=provider,
            assignment=assignments[0],
            intent=intent,
            diff_excerpt=change_summary.diff_excerpt,
            observations=observation_store.summaries_by_id(),
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

    report = render_markdown_report(
        review_id=review_id,
        base_revision=args.base,
        head_revision=args.head,
        risk_assessment=risk_assessment,
        changed_files=change_summary.changed_files,
        reviewer_result=reviewer_result,
        observation_summaries=observation_store.summaries_by_id(),
        repository_intelligence_summary=repository_intelligence_summary,
    )
    (store.run_dir / "report.md").write_text(report, encoding="utf-8")

    print(f"Review foundation completed: {store.run_dir}")
    return 0

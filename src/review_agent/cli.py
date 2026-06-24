from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import uuid

from review_agent.checkpoint import CheckpointStore
from review_agent.git_repo import collect_change_summary
from review_agent.intent import build_intent_packet
from review_agent.models import ReviewRequest
from review_agent.quality import detect_quality_gates, run_python_compile_gate
from review_agent.reporting import render_markdown_report
from review_agent.risk import LocalRiskAssessor, build_risk_packet
from review_agent.runtime import build_assignments


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

    store = CheckpointStore(repo, review_id)
    store.write_json("request.json", asdict(request))
    store.write_json("intent.json", asdict(intent))
    store.write_json("risk_packet.json", asdict(risk_packet))
    store.write_json("risk.json", asdict(risk_assessment))
    store.write_json("assignments.json", {"assignments": [asdict(item) for item in assignments]})
    store.write_json("quality_gates.json", {"results": [asdict(item) for item in quality_results]})

    report = render_markdown_report(
        review_id=review_id,
        base_revision=args.base,
        head_revision=args.head,
        risk_assessment=risk_assessment,
        changed_files=change_summary.changed_files,
    )
    (store.run_dir / "report.md").write_text(report, encoding="utf-8")

    print(f"Review foundation completed: {store.run_dir}")
    return 0

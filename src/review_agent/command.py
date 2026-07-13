from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

from review_agent.checkpoint import CheckpointStore
from review_agent.git_repo import ChangeSummary, collect_change_summary
from review_agent.intent_clarification import ConsoleIntentClarifier
from review_agent.model_adapter_factory import build_model_adapter_factory_from_config
from review_agent.models import IntentPacket, QualityGateResult, ReviewRequest, RiskAssessment
from review_agent.pipeline import (
    PipelineConfigurationError,
    PipelineError,
    ReviewPipeline,
)
from review_agent.revision import RevisionResolver
from review_agent.resume import ResumeAction, ResumeBlockedError, ReviewSessionResumer
from review_agent.run_state import RunStatus
from review_agent.session import (
    DEFAULT_MODEL_STAGE_MAX_ELAPSED_SECONDS,
    DEFAULT_MODEL_STAGE_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_STAGE_MAX_PROVIDER_ATTEMPTS,
    ModelStageConfig,
    ReviewExecutionConfig,
    initial_session_manifest,
)
from review_agent.session_store import SessionStore


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
    review.add_argument("--title")
    review.add_argument("--description")
    review.add_argument("--requirement", action="append", default=[])
    review.add_argument("--project-rule", action="append", default=[])
    review.add_argument("--non-interactive", action="store_true")
    review.add_argument(
        "--reviewer-provider",
        choices=["none", "fake", "openai-compatible"],
        default="none",
    )
    review.add_argument("--reviewer-mode", choices=["single", "multi"], default="single")
    review.add_argument(
        "--reviewer-loop",
        choices=["single-shot", "agent-loop"],
        default="single-shot",
    )
    review.add_argument("--reviewer-model")
    review.add_argument("--reviewer-base-url")
    review.add_argument("--reviewer-api-key-env", default="REVIEW_AGENT_API_KEY")
    _add_model_stage_arguments(review, "risk-assessor")
    _add_model_stage_arguments(review, "portfolio-planner")
    _add_model_stage_arguments(review, "semantic-reconciler")

    resume = subparsers.add_parser(
        "resume",
        help="Inspect and resume from a local review checkpoint",
    )
    resume.add_argument("review_id", help="Review id under .review-agent/runs")
    resume.add_argument("--repo", default=".", help="Repository path")
    return parser


def _add_model_stage_arguments(
    parser: argparse.ArgumentParser,
    stage: str,
) -> None:
    parser.add_argument(
        f"--{stage}-mode",
        choices=["local", "model"],
        default="local",
    )
    parser.add_argument(
        f"--{stage}-provider",
        choices=["inherit", "none", "fake", "openai-compatible"],
        default="inherit",
    )
    parser.add_argument(f"--{stage}-model")
    parser.add_argument(f"--{stage}-base-url")
    parser.add_argument(f"--{stage}-api-key-env")
    parser.add_argument(
        f"--{stage}-max-output-tokens",
        type=int,
        default=DEFAULT_MODEL_STAGE_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument(
        f"--{stage}-max-provider-attempts",
        type=int,
        default=DEFAULT_MODEL_STAGE_MAX_PROVIDER_ATTEMPTS,
    )
    parser.add_argument(
        f"--{stage}-max-elapsed-seconds",
        type=float,
        default=DEFAULT_MODEL_STAGE_MAX_ELAPSED_SECONDS,
    )


def _resolve_model_stage_config(
    args: argparse.Namespace,
    stage: str,
    reviewer: ReviewExecutionConfig,
) -> ModelStageConfig:
    argument_prefix = stage.replace("-", "_")
    mode = getattr(args, f"{argument_prefix}_mode")
    api_key_env_override = getattr(args, f"{argument_prefix}_api_key_env")
    common = {
        "api_key_env": (
            reviewer.reviewer_api_key_env
            if api_key_env_override is None
            else api_key_env_override
        ),
        "max_output_tokens": getattr(
            args,
            f"{argument_prefix}_max_output_tokens",
        ),
        "max_provider_attempts": getattr(
            args,
            f"{argument_prefix}_max_provider_attempts",
        ),
        "max_elapsed_seconds": getattr(
            args,
            f"{argument_prefix}_max_elapsed_seconds",
        ),
    }
    if mode == "local":
        try:
            return ModelStageConfig(mode="local", provider="none", **common)
        except ValueError as error:
            raise ValueError(f"{stage}: {error}") from error

    requested_provider = getattr(args, f"{argument_prefix}_provider")
    provider = (
        reviewer.reviewer_provider
        if requested_provider == "inherit"
        else requested_provider
    )
    model_override = getattr(args, f"{argument_prefix}_model")
    base_url_override = getattr(args, f"{argument_prefix}_base_url")
    try:
        return ModelStageConfig(
            mode="model",
            provider=provider,
            model=(
                reviewer.reviewer_model
                if model_override is None
                else model_override
            ),
            base_url=(
                reviewer.reviewer_base_url
                if base_url_override is None
                else base_url_override
            ),
            **common,
        )
    except ValueError as error:
        raise ValueError(f"{stage}: {error}") from error


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

    try:
        execution_config = replace(
            execution_config,
            risk_assessor=_resolve_model_stage_config(
                args,
                "risk-assessor",
                execution_config,
            ),
            portfolio_planner=_resolve_model_stage_config(
                args,
                "portfolio-planner",
                execution_config,
            ),
            semantic_reconciler=_resolve_model_stage_config(
                args,
                "semantic-reconciler",
                execution_config,
            ),
        )
    except ValueError as error:
        print(f"Planning model stage configuration error: {error}")
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

    request = ReviewRequest(
        repository_path=str(repo),
        base_revision=revisions.requested_base,
        head_revision=revisions.requested_head,
        title=args.title,
        description=args.description,
        linked_requirements=tuple(args.requirement),
        user_intent=args.intent,
        review_focus=args.focus,
        project_rules=tuple(args.project_rule),
    )
    pipeline = ReviewPipeline(
        repository=repo,
        checkpoint_store=store,
        session_store=session_store,
        request=request,
        collect_change_summary_fn=collect_change_summary,
        adapter_factory_builder=build_model_adapter_factory_from_config,
        intent_clarifier=ConsoleIntentClarifier(),
    )
    try:
        result = pipeline.execute()
    except PipelineConfigurationError as error:
        print(f"Reviewer adapter configuration error: {error}")
        return 2
    except PipelineError as error:
        print(f"Review failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        manifest = session_store.load()
        retryable = all(
            checkpoint.status.value == "completed"
            for checkpoint in manifest.phases.values()
        )
        suffix = "; Session remains retryable" if retryable else ""
        print(f"Review failed: {error}{suffix}", file=sys.stderr)
        return 1

    if result.awaiting_user:
        print(f"Review awaiting intent clarification: {store.run_dir}")
        print(f"Review id: {review_id}")
        for question in result.open_questions:
            print(f"  [{question.field.value}] {question.question}")
            print(f"    Why: {question.rationale}")
        print(f"Resume with: review-agent resume {review_id} --repo {repo}")
        return 0

    context = result.context
    for warning in context.compatibility_warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    _print_preflight_summary(
        review_id=review_id,
        repo=repo,
        requested_base_revision=revisions.requested_base,
        requested_head_revision=revisions.requested_head,
        resolved_base_revision=revisions.resolved_base_sha,
        resolved_head_revision=revisions.resolved_head_sha,
        change_summary=_require(context.change_summary, "change summary"),
        intent_packet=_require(context.intent, "intent"),
        quality_results=context.quality_results,
        risk_assessment=_require(context.risk_assessment, "risk assessment"),
        run_dir=store.run_dir,
    )
    final_risk = _require(context.final_risk, "final risk")
    brief = _require(context.brief, "review brief")
    completion = _require(context.completion, "completion result")
    print(f"Review execution completed: {store.run_dir}")
    print(f"Review outcome: {completion.status}")
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
            session_store.load()
        except Exception as error:
            print(f"Review run has an invalid session.json: {error}", file=sys.stderr)
            return 2
        try:
            result = ReviewSessionResumer(
                repository=repo,
                checkpoint_store=store,
                session_store=session_store,
                intent_clarifier=ConsoleIntentClarifier(),
            ).resume()
        except ResumeBlockedError as error:
            print(f"Resume blocked: {error}", file=sys.stderr)
            return 2
        except PipelineConfigurationError as error:
            print(f"Resume failed: reviewer configuration: {error}", file=sys.stderr)
            return 1
        except PipelineError as error:
            print(f"Resume failed: {error}", file=sys.stderr)
            return 1
        except Exception as error:
            print(f"Resume failed: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        summary_store = store
        summary_session_store = session_store
        if (
            result.action is ResumeAction.CREATE_INCREMENTAL_SESSION
            and result.new_review_id is not None
        ):
            summary_store = CheckpointStore(
                repo,
                result.new_review_id,
                create=False,
            )
            summary_session_store = SessionStore(summary_store.run_dir)
        _print_session_summary(summary_session_store, summary_store)
        print(f"  Action: {result.action.value}")
        if result.action is ResumeAction.CREATE_INCREMENTAL_SESSION:
            print(f"  Parent review: {result.parent_review_id}")
            print(f"  New review: {result.new_review_id}")
            if result.change_kind is not None:
                print(f"  Change: {result.change_kind.value}")
            print(f"  Full range: {result.full_range}")
            if result.incremental_range is not None:
                print(f"  Incremental priority range: {result.incremental_range}")
        if result.starting_phase is not None:
            print(f"  Starting phase: {result.starting_phase.value}")
        if result.reused_phases:
            print(
                "  Reused phases: "
                + ", ".join(phase.value for phase in result.reused_phases)
            )
        if result.action is ResumeAction.AUDIT_COMPLETED:
            print("  No model or tool execution performed")
            print("  Audit: valid")
        if result.action is ResumeAction.AWAITING_USER:
            print("  Intent clarification is still awaiting user input")
        return 0

    return _run_legacy_resume(store)


def _print_session_summary(session_store: SessionStore, store: CheckpointStore) -> None:
    session = session_store.load()
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


def _run_legacy_resume(store: CheckpointStore) -> int:
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
    return ", ".join(
        f"{result.name}={result.status}" for result in quality_results
    )


def _require(value, label: str):
    if value is None:
        raise RuntimeError(f"Pipeline completed without {label}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

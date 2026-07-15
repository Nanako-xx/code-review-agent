from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence
import uuid

from review_agent.checkpoint import CheckpointStore
from review_agent.artifacts import artifact_schema
from review_agent.git_repo import ChangeSummary, collect_change_summary
from review_agent.intent_clarification import ConsoleIntentClarifier
from review_agent.memory_identity import (
    MemoryIdentityError,
    MemoryRootResolver,
    RepositoryMemoryNamespace,
    materialize_repository_memory_namespace,
    plan_repository_memory_namespace,
    repository_key as canonical_repository_key,
    repository_namespace_path,
)
from review_agent.memory_feedback import (
    FeedbackError,
    FeedbackErrorCode,
    FeedbackImportRequest,
    FeedbackImportService,
    MissedFindingInput,
)
from review_agent.memory_lifecycle import (
    ApprovalResult,
    CandidateDedupeKind,
    MemoryLifecycle,
    MemoryLifecycleError,
    MemoryLifecycleErrorCode,
)
from review_agent.memory_models import (
    CandidateStatus,
    CURRENT_MEMORY_STORE_SCHEMA_VERSION,
    DurableMemoryRecord,
    FeedbackDecision,
    FeedbackReasonCode,
    FeedbackStatus,
    FindingSeverity,
    GenerationMetadata,
    HumanDeclarationAuthority,
    HumanDeclarationOrigin,
    HumanDeclarationSourceRef,
    MAX_SNAPSHOT_RECORDS,
    MemoryCandidate,
    MemoryExecutionConfig,
    MemoryMode,
    ProducerType,
    RecordStatus,
    SourceRef,
    canonical_sha256,
    stable_request_id,
    validate_stable_id,
)
from review_agent.memory_relink import (
    RepositoryAuthorityResolution,
    RepositoryRelinkConflictError,
    RepositoryRelinkError,
    RepositoryRelinkIntegrityError,
    RepositoryRelinkRegistry,
    RepositoryRelinkValidationError,
    resolve_repository_authority,
)
from review_agent.memory_sources import (
    SourceValidationCode,
    SourceValidationError,
    SourceValidationReport,
    SourceValidator,
    TrustedCandidateProvenance,
)
from review_agent.memory_store import (
    BlobGCResult,
    ImportPlan,
    MemoryStore,
    MemoryStoreError,
    MemoryStoreErrorCode,
    WriteResult,
)
from review_agent.model_adapter_factory import build_model_adapter_factory_from_config
from review_agent.models import IntentPacket, QualityGateResult, ReviewRequest, RiskAssessment
from review_agent.pipeline import (
    PipelineConfigurationError,
    PipelineError,
    ReviewPipeline,
)
from review_agent.revision import RevisionResolver
from review_agent.resume import ResumeAction, ResumeBlockedError, ReviewSessionResumer
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import (
    DEFAULT_MODEL_STAGE_MAX_ELAPSED_SECONDS,
    DEFAULT_MODEL_STAGE_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_STAGE_MAX_PROVIDER_ATTEMPTS,
    ModelStageConfig,
    PhaseStatus,
    ReviewExecutionConfig,
    initial_session_manifest,
)
from review_agent.session_store import SessionStore


class _MemoryJSONParseError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = _parse_command_args(parser, argv)
    except _MemoryJSONParseError:
        print(
            json.dumps(
                {
                    "error": {
                        "code": "usage",
                        "message": "memory command arguments are invalid",
                    },
                    "schema": "memory_cli_v1",
                    "type": "error",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    if args.command == "review":
        return _run_review(args)
    if args.command == "resume":
        return _run_resume(args)
    if args.command == "memory":
        return _run_memory(args)
    parser.print_help()
    return 2


def _parse_command_args(
    parser: argparse.ArgumentParser,
    argv: list[str] | None,
) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    json_memory_error = (
        arguments[:1] == ["memory"]
        and _memory_json_output_requested(arguments[1:])
        and "--help" not in arguments
        and "-h" not in arguments
    )
    if not json_memory_error:
        return parser.parse_args(argv)
    with redirect_stderr(io.StringIO()):
        try:
            return parser.parse_args(arguments)
        except SystemExit as error:
            if error.code == 0:
                raise
            raise _MemoryJSONParseError() from None


def _memory_json_output_requested(arguments: Sequence[str]) -> bool:
    for index, argument in enumerate(arguments):
        if (
            len(argument) > 2
            and argument.startswith("--")
            and "--json".startswith(argument)
        ):
            return True
        option, separator, value = argument.partition("=")
        if (
            separator
            and len(option) > 2
            and option.startswith("--")
            and "--format".startswith(option)
            and value == "json"
        ):
            return True
        if (
            len(argument) > 2
            and argument.startswith("--")
            and "--format".startswith(argument)
            and index + 1 < len(arguments)
            and arguments[index + 1] == "json"
        ):
            return True
    return False


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
    _add_review_memory_arguments(review)
    _add_model_stage_arguments(review, "risk-assessor")
    _add_model_stage_arguments(review, "portfolio-planner")
    _add_model_stage_arguments(review, "semantic-reconciler")
    _add_model_stage_arguments(review, "memory-curator")

    resume = subparsers.add_parser(
        "resume",
        help="Inspect and resume from a local review checkpoint",
    )
    resume.add_argument("review_id", help="Review id under .review-agent/runs")
    resume.add_argument("--repo", default=".", help="Repository path")
    resume.add_argument(
        "--upgrade-to-v5",
        action="store_true",
        help=(
            "Explicitly upgrade a legacy revision-drift Session to the v5 "
            "Memory-aware execution protocol"
        ),
    )
    _add_review_memory_arguments(resume)
    _add_model_stage_arguments(resume, "memory-curator")
    _add_memory_parser(subparsers)
    return parser


def _add_review_memory_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--memory-mode",
        choices=[item.value for item in MemoryMode],
        default=MemoryMode.READ_WRITE.value,
        help="Durable Memory permission boundary (default: read-write)",
    )
    parser.add_argument(
        "--memory-root",
        help=(
            "Absolute Memory root override; otherwise "
            "REVIEW_AGENT_MEMORY_ROOT then the platform default is used"
        ),
    )
    parser.add_argument(
        "--memory-required",
        action="store_true",
        help="Block instead of degrading when required Memory cannot be validated",
    )
    parser.add_argument(
        "--memory-max-snapshot-records",
        type=int,
        default=MAX_SNAPSHOT_RECORDS,
    )
    parser.add_argument(
        "--memory-max-snapshot-bytes",
        type=int,
        default=8_388_608,
    )
    parser.add_argument(
        "--memory-max-context-records",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--memory-max-query-results",
        type=int,
        default=8,
    )


def _add_memory_parser(subparsers: Any) -> None:
    memory = subparsers.add_parser(
        "memory",
        help="Inspect and manage repository-scoped durable Memory",
    )
    _add_memory_common_arguments(memory)
    commands = memory.add_subparsers(dest="memory_command")

    status = commands.add_parser("status", help="Show Memory Store status")
    _add_memory_common_arguments(status, suppress_defaults=True)
    status.set_defaults(memory_action="status")

    records = commands.add_parser("list", help="List durable Memory records")
    _add_memory_common_arguments(records, suppress_defaults=True)
    records.add_argument(
        "--status",
        choices=[item.value for item in RecordStatus],
        dest="record_status",
    )
    records.set_defaults(memory_action="list")

    show = commands.add_parser("show", help="Show one durable Memory record")
    _add_memory_common_arguments(show, suppress_defaults=True)
    show.add_argument("memory_id", help="Stable MEM- identifier")
    show.set_defaults(memory_action="show")

    candidates = commands.add_parser("candidates", help="List Memory candidates")
    _add_memory_common_arguments(candidates, suppress_defaults=True)
    candidates.add_argument(
        "--status",
        choices=[item.value for item in CandidateStatus],
        dest="candidate_status",
    )
    candidates.set_defaults(memory_action="candidates")

    candidate = commands.add_parser("candidate", help="Inspect a Memory candidate")
    _add_memory_common_arguments(candidate, suppress_defaults=True)
    candidate_commands = candidate.add_subparsers(dest="candidate_command")
    candidate_show = candidate_commands.add_parser(
        "show",
        help="Show one Memory candidate",
    )
    _add_memory_common_arguments(candidate_show, suppress_defaults=True)
    candidate_show.add_argument("candidate_id", help="Stable MC- identifier")
    candidate_show.set_defaults(memory_action="candidate_show")

    approve = commands.add_parser("approve", help="Approve a pending candidate")
    _add_memory_common_arguments(approve, suppress_defaults=True)
    approve.add_argument("candidate_id", help="Stable MC- identifier")
    _add_memory_decision_arguments(approve)
    approve.set_defaults(memory_action="approve")

    reject = commands.add_parser("reject", help="Reject a validated or pending candidate")
    _add_memory_common_arguments(reject, suppress_defaults=True)
    reject.add_argument("candidate_id", help="Stable MC- identifier")
    _add_memory_decision_arguments(reject, reason_code=True)
    reject.set_defaults(memory_action="reject")

    revoke = commands.add_parser("revoke", help="Revoke an active Memory record")
    _add_memory_common_arguments(revoke, suppress_defaults=True)
    revoke.add_argument("memory_id", help="Stable MEM- identifier")
    _add_memory_decision_arguments(revoke)
    revoke.set_defaults(memory_action="revoke")

    revalidate = commands.add_parser(
        "revalidate",
        help="Replace and supersede a Memory record with a reviewed candidate",
    )
    _add_memory_common_arguments(revalidate, suppress_defaults=True)
    revalidate.add_argument("memory_id", help="Stable predecessor MEM- identifier")
    revalidate.add_argument(
        "--candidate",
        "--candidate-id",
        required=True,
        dest="candidate_id",
        help="Stable MC- identifier for the immutable replacement",
    )
    _add_memory_decision_arguments(revalidate)
    revalidate.set_defaults(memory_action="revalidate")

    feedback = commands.add_parser(
        "feedback",
        help="Record or inspect audited human review Feedback",
    )
    _add_memory_common_arguments(feedback, suppress_defaults=True)
    feedback_commands = feedback.add_subparsers(
        dest="feedback_command",
        required=True,
    )

    feedback_record = feedback_commands.add_parser(
        "record",
        help="Record one human decision about a canonical review Finding",
    )
    _add_memory_common_arguments(feedback_record, suppress_defaults=True)
    feedback_record.add_argument("review_id", help="Completed review Session id")
    feedback_record.add_argument("finding_id", help="Stable canonical F- identifier")
    feedback_record.add_argument("--head-sha", required=True)
    feedback_record.add_argument("--finding-hash", required=True)
    feedback_record.add_argument(
        "--evidence-ref",
        action="append",
        required=True,
        default=[],
        help="Canonical Observation ID; repeat for every Finding evidence ref",
    )
    feedback_record.add_argument(
        "--decision",
        choices=[item.value for item in FeedbackDecision],
        required=True,
    )
    feedback_record.add_argument(
        "--final-severity",
        choices=[item.value for item in FindingSeverity],
        required=True,
    )
    feedback_record.add_argument(
        "--reason-code",
        choices=[item.value for item in FeedbackReasonCode],
        required=True,
    )
    feedback_record.add_argument("--actor", required=True)
    feedback_record.add_argument("--reason", required=True)
    feedback_record.add_argument("--created-at")
    feedback_record.add_argument("--request-id")
    feedback_record.add_argument(
        "--missed-claim",
        "--claim",
        dest="missed_claim",
    )
    feedback_record.add_argument(
        "--missed-path",
        "--path",
        dest="missed_path",
    )
    feedback_record.add_argument(
        "--missed-line",
        "--line",
        dest="missed_line",
        type=int,
    )
    feedback_record.add_argument(
        "--missed-contract",
        "--contract",
        dest="missed_contracts",
        action="append",
        default=[],
    )
    feedback_record.add_argument(
        "--missed-original-severity",
        "--original-severity",
        dest="missed_original_severity",
        choices=[item.value for item in FindingSeverity],
    )
    feedback_record.add_argument(
        "--human-declaration",
        help="Explicit human declaration required for a missed Finding",
    )
    feedback_record.add_argument(
        "--source-ref-json",
        "--source-ref",
        dest="source_ref_json",
        action="append",
        default=[],
        help="Canonical SourceRef JSON (or @path); repeat for missed evidence",
    )
    feedback_record.add_argument(
        "--yes",
        action="store_true",
        help="Confirm a non-interactive Feedback write",
    )
    feedback_record.set_defaults(memory_action="feedback_record")

    feedback_list = feedback_commands.add_parser(
        "list",
        help="List audited human Feedback records",
    )
    _add_memory_common_arguments(feedback_list, suppress_defaults=True)
    feedback_list.add_argument("--review-id")
    feedback_list.add_argument(
        "--decision",
        choices=[item.value for item in FeedbackDecision],
        dest="feedback_decision",
    )
    feedback_list.add_argument(
        "--status",
        choices=[item.value for item in FeedbackStatus],
        dest="feedback_status",
    )
    feedback_list.set_defaults(memory_action="feedback_list")

    replay_outbox = commands.add_parser(
        "replay-outbox",
        help="Replay one hash-verified Session Memory outbox through its service",
    )
    _add_memory_common_arguments(replay_outbox, suppress_defaults=True)
    replay_outbox.add_argument("review_id", help="Review Session id")
    _add_memory_decision_arguments(replay_outbox)
    replay_outbox.set_defaults(memory_action="replay_outbox")

    export = commands.add_parser("export", help="Write a redacted Memory export")
    _add_memory_common_arguments(export, suppress_defaults=True)
    export.add_argument("path", help="Destination manifest file or directory")
    export.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing an existing export manifest",
    )
    export.add_argument(
        "--yes",
        action="store_true",
        help="Confirm a non-interactive overwrite",
    )
    export.set_defaults(memory_action="export")

    import_command = commands.add_parser(
        "import",
        help="Validate or explicitly restore a Memory export",
    )
    _add_memory_common_arguments(import_command, suppress_defaults=True)
    import_command.add_argument("path", help="Manifest file or export directory")
    import_mode = import_command.add_mutually_exclusive_group()
    import_mode.add_argument(
        "--dry-run",
        action="store_false",
        dest="apply",
        help="Validate without changing the target (default)",
    )
    import_mode.add_argument(
        "--apply",
        action="store_true",
        help="Apply a restorable manifest",
    )
    import_command.set_defaults(apply=False)
    import_command.add_argument(
        "--identity-match",
        nargs="?",
        const="current",
        metavar="REPOSITORY_KEY",
        help=(
            "Explicitly assert that the manifest targets the current repository; "
            "optionally name its exact repository key"
        ),
    )
    import_command.add_argument(
        "--yes",
        action="store_true",
        help="Confirm a non-interactive import apply",
    )
    import_command.set_defaults(memory_action="import")

    gc = commands.add_parser("gc", help="Scan or collect unreferenced Memory blobs")
    _add_memory_common_arguments(gc, suppress_defaults=True)
    gc_mode = gc.add_mutually_exclusive_group()
    gc_mode.add_argument(
        "--dry-run",
        action="store_false",
        dest="apply",
        help="Report eligible blobs without deleting them (default)",
    )
    gc_mode.add_argument(
        "--apply",
        action="store_true",
        help="Delete blobs still eligible after pin revalidation",
    )
    gc.add_argument(
        "--grace-seconds",
        type=float,
        default=0.0,
        help="Only consider blobs at least this old",
    )
    gc.add_argument(
        "--yes",
        action="store_true",
        help="Confirm non-interactive garbage collection",
    )
    gc.set_defaults(apply=False, memory_action="gc")

    relink = commands.add_parser(
        "relink",
        help="Explicitly bind this live repository to an existing Memory authority",
    )
    _add_memory_common_arguments(relink, suppress_defaults=True)
    relink.add_argument(
        "--from-key",
        required=True,
        dest="from_repository_key",
        metavar="REPOSITORY_KEY",
        help="Exact existing repository authority key; origin is never inferred",
    )
    _add_memory_decision_arguments(relink)
    relink.set_defaults(memory_action="relink")


def _add_memory_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
) -> None:
    default: Any = argparse.SUPPRESS if suppress_defaults else "."
    parser.add_argument("--repo", default=default, help="Git repository path")
    parser.add_argument(
        "--memory-root",
        default=argparse.SUPPRESS if suppress_defaults else None,
        help="Absolute Memory root override",
    )
    parser.add_argument(
        "--format",
        choices=("human", "json"),
        dest="output_format",
        default=argparse.SUPPRESS if suppress_defaults else "human",
        help="Output format",
    )
    parser.add_argument(
        "--json",
        action="store_const",
        const="json",
        dest="output_format",
        default=argparse.SUPPRESS if suppress_defaults else "human",
        help="Alias for --format json",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        default=argparse.SUPPRESS if suppress_defaults else False,
        help="Never prompt; writes require --yes",
    )


def _add_memory_decision_arguments(
    parser: argparse.ArgumentParser,
    *,
    reason_code: bool = False,
) -> None:
    parser.add_argument("--actor", required=True, help="Human actor identifier")
    if reason_code:
        parser.add_argument("--reason-code", required=True, help="Stable rejection reason code")
    parser.add_argument("--reason", required=True, help="Human decision rationale")
    parser.add_argument(
        "--request-id",
        help="Optional stable REQ- id for idempotent scripting",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm a non-interactive write",
    )


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


def _resolve_memory_execution_config(
    args: argparse.Namespace,
) -> MemoryExecutionConfig:
    resolution = MemoryRootResolver().resolve(
        getattr(args, "memory_root", None),
        create=False,
    )
    return MemoryExecutionConfig(
        mode=MemoryMode(args.memory_mode),
        root_path=resolution.path,
        required=args.memory_required,
        max_snapshot_records=args.memory_max_snapshot_records,
        max_snapshot_bytes=args.memory_max_snapshot_bytes,
        max_context_records=args.memory_max_context_records,
        max_query_results=args.memory_max_query_results,
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


_MEMORY_CLI_SCHEMA = "memory_cli_v1"
_MEMORY_EXIT_OPERATIONAL = 1
_MEMORY_EXIT_USAGE = 2
_MEMORY_EXIT_NOT_FOUND = 3
_MEMORY_EXIT_CONFLICT = 4
_MEMORY_OUTBOX_SCHEMA = "memory_candidate_outbox_v1"
_MEMORY_PERSISTENCE_RECEIPT_SCHEMA = "memory_persistence_receipt_v1"

_FEEDBACK_ERROR_EXIT_CODES = {
    FeedbackErrorCode.INVALID_INPUT: _MEMORY_EXIT_USAGE,
    FeedbackErrorCode.SESSION_NOT_FOUND: _MEMORY_EXIT_NOT_FOUND,
    FeedbackErrorCode.SESSION_UNTRUSTED: _MEMORY_EXIT_CONFLICT,
    FeedbackErrorCode.SESSION_NOT_COMPLETED: _MEMORY_EXIT_CONFLICT,
    FeedbackErrorCode.REPOSITORY_MISMATCH: _MEMORY_EXIT_CONFLICT,
    FeedbackErrorCode.HEAD_MISMATCH: _MEMORY_EXIT_CONFLICT,
    FeedbackErrorCode.FINDING_NOT_FOUND: _MEMORY_EXIT_NOT_FOUND,
    FeedbackErrorCode.FINDING_NOT_CANONICAL: _MEMORY_EXIT_CONFLICT,
    FeedbackErrorCode.FINDING_HASH_MISMATCH: _MEMORY_EXIT_CONFLICT,
    FeedbackErrorCode.EVIDENCE_MISMATCH: _MEMORY_EXIT_CONFLICT,
    FeedbackErrorCode.HUMAN_DECLARATION_REQUIRED: _MEMORY_EXIT_USAGE,
    FeedbackErrorCode.VERIFIABLE_SOURCE_REQUIRED: _MEMORY_EXIT_USAGE,
    FeedbackErrorCode.SOURCE_VALIDATION_FAILED: _MEMORY_EXIT_CONFLICT,
    FeedbackErrorCode.CONFLICTING_DECISION: _MEMORY_EXIT_CONFLICT,
    FeedbackErrorCode.AGGREGATION_LIMIT_EXCEEDED: _MEMORY_EXIT_OPERATIONAL,
    FeedbackErrorCode.DURABLE_MEMORY_CONVERSION_PROHIBITED: _MEMORY_EXIT_USAGE,
}

_SOURCE_ERROR_EXIT_CODES = {
    SourceValidationCode.INVALID_INPUT: _MEMORY_EXIT_USAGE,
    SourceValidationCode.INVALID_CONFIGURATION: _MEMORY_EXIT_OPERATIONAL,
    SourceValidationCode.UNTYPED_SOURCE: _MEMORY_EXIT_USAGE,
    SourceValidationCode.SOURCE_NOT_ALLOWLISTED: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.RUNTIME_PROVENANCE_REQUIRED: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.SENSITIVITY_BLOCKED: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.SENSITIVE_CONTENT: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.VALIDATION_SKIPPED: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.UNSAFE_PATH: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.REPOSITORY_UNAVAILABLE: _MEMORY_EXIT_OPERATIONAL,
    SourceValidationCode.REVISION_NOT_FOUND: _MEMORY_EXIT_NOT_FOUND,
    SourceValidationCode.REVISION_MISMATCH: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.REVISION_LINEAGE_UNAUTHORIZED: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.SOURCE_NOT_FOUND: _MEMORY_EXIT_NOT_FOUND,
    SourceValidationCode.SOURCE_NOT_REGULAR: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.SOURCE_TOO_LARGE: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.SOURCE_ENCODING_INVALID: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.RANGE_OUT_OF_BOUNDS: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.HASH_MISMATCH: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.SYMBOL_NOT_FOUND: _MEMORY_EXIT_NOT_FOUND,
    SourceValidationCode.SYMBOL_AMBIGUOUS: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.SYMBOL_UNSUPPORTED: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.SESSION_NOT_FOUND: _MEMORY_EXIT_NOT_FOUND,
    SourceValidationCode.SESSION_UNTRUSTED: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.REVIEW_ID_MISMATCH: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.REPOSITORY_MISMATCH: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.DESCRIPTOR_NOT_FOUND: _MEMORY_EXIT_NOT_FOUND,
    SourceValidationCode.DESCRIPTOR_SCHEMA_MISMATCH: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.SESSION_ARTIFACT_INELIGIBLE: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.REVISION_BINDING_MISMATCH: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.SESSION_ARTIFACT_INVALID: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.OBSERVATION_NOT_FOUND: _MEMORY_EXIT_NOT_FOUND,
    SourceValidationCode.OBSERVATION_UNTRUSTED: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.HUMAN_DECLARATION_UNAUTHORIZED: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.HUMAN_DECLARATION_MISMATCH: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.CANDIDATE_AUTHORITY_MISMATCH: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.VALIDATION_REPORT_MISMATCH: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.AUTHORITY_RECEIPT_INVALID: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.PRODUCER_POLICY_UNAUTHORIZED: _MEMORY_EXIT_CONFLICT,
    SourceValidationCode.INTERNAL_ERROR: _MEMORY_EXIT_OPERATIONAL,
}

_LIFECYCLE_ERROR_EXIT_CODES = {
    MemoryLifecycleErrorCode.INVALID_INPUT: _MEMORY_EXIT_USAGE,
    MemoryLifecycleErrorCode.INVALID_TRANSITION: _MEMORY_EXIT_CONFLICT,
    MemoryLifecycleErrorCode.HUMAN_ACTOR_REQUIRED: _MEMORY_EXIT_USAGE,
    MemoryLifecycleErrorCode.HUMAN_REASON_REQUIRED: _MEMORY_EXIT_USAGE,
    MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED: _MEMORY_EXIT_CONFLICT,
    MemoryLifecycleErrorCode.DUPLICATE_SUPPRESSED: _MEMORY_EXIT_CONFLICT,
    MemoryLifecycleErrorCode.INVALID_REPLACEMENT: _MEMORY_EXIT_CONFLICT,
}

if set(_FEEDBACK_ERROR_EXIT_CODES) != set(FeedbackErrorCode):
    raise RuntimeError("Feedback CLI error classification is incomplete")
if set(_SOURCE_ERROR_EXIT_CODES) != set(SourceValidationCode):
    raise RuntimeError("source CLI error classification is incomplete")
if set(_LIFECYCLE_ERROR_EXIT_CODES) != set(MemoryLifecycleErrorCode):
    raise RuntimeError("Memory lifecycle CLI error classification is incomplete")


class _MemoryCLIError(RuntimeError):
    def __init__(self, code: str, message: str, exit_code: int) -> None:
        self.code = code
        self.message = message
        self.exit_code = exit_code
        super().__init__(message)


@dataclass(frozen=True)
class _MemoryCommandContext:
    repository: Path
    repository_key: str
    locator_repository_key: str
    authority_resolution_hash: str
    binding_id: str | None
    registry_generation: int
    memory_root: Path
    root_source: str
    namespace: RepositoryMemoryNamespace
    store: MemoryStore | None


@dataclass(frozen=True)
class _CandidateRuntimeAuthority:
    provenance: TrustedCandidateProvenance
    validator: SourceValidator


@dataclass(frozen=True)
class _PreparedOutboxReplay:
    outbox_digest: str
    batch_digest: str
    entries: tuple[tuple[str, str], ...]


def _run_memory(args: argparse.Namespace) -> int:
    action = getattr(args, "memory_action", None)
    handlers: Mapping[str, Callable[[argparse.Namespace], int]] = {
        "status": _memory_status,
        "list": _memory_list,
        "show": _memory_show,
        "candidates": _memory_candidates,
        "candidate_show": _memory_candidate_show,
        "approve": _memory_approve,
        "reject": _memory_reject,
        "revoke": _memory_revoke,
        "revalidate": _memory_revalidate,
        "feedback_record": _memory_feedback_record,
        "feedback_list": _memory_feedback_list,
        "replay_outbox": _memory_replay_outbox,
        "export": _memory_export,
        "import": _memory_import,
        "gc": _memory_gc,
        "relink": _memory_relink,
    }
    if action not in handlers:
        return _emit_memory_error(
            args,
            code="usage",
            message="a complete memory command is required",
            exit_code=_MEMORY_EXIT_USAGE,
        )
    try:
        return handlers[action](args)
    except _MemoryCLIError as error:
        return _emit_memory_error(
            args,
            code=error.code,
            message=error.message,
            exit_code=error.exit_code,
        )
    except MemoryStoreError as error:
        return _emit_memory_store_error(args, error)
    except MemoryLifecycleError as error:
        return _emit_memory_error(
            args,
            code="lifecycle_%s" % error.code.value,
            message="the requested memory lifecycle transition was rejected",
            exit_code=_LIFECYCLE_ERROR_EXIT_CODES[error.code],
        )
    except FeedbackError as error:
        return _emit_memory_error(
            args,
            code="feedback_%s" % error.code.value,
            message="human Feedback failed strict authority validation",
            exit_code=_FEEDBACK_ERROR_EXIT_CODES[error.code],
        )
    except SourceValidationError as error:
        return _emit_memory_error(
            args,
            code="source_%s" % error.code.value,
            message="memory source validation failed",
            exit_code=_SOURCE_ERROR_EXIT_CODES[error.code],
        )
    except MemoryIdentityError:
        return _emit_memory_error(
            args,
            code="identity_invalid",
            message="the memory root or repository identity is invalid",
            exit_code=_MEMORY_EXIT_USAGE,
        )
    except RepositoryRelinkConflictError:
        return _emit_memory_error(
            args,
            code="relink_conflict",
            message="repository Memory relink preconditions changed or conflict",
            exit_code=_MEMORY_EXIT_CONFLICT,
        )
    except RepositoryRelinkIntegrityError:
        return _emit_memory_error(
            args,
            code="relink_integrity",
            message="repository Memory authority registry failed integrity checks",
            exit_code=_MEMORY_EXIT_OPERATIONAL,
        )
    except RepositoryRelinkValidationError:
        return _emit_memory_error(
            args,
            code="relink_invalid",
            message="repository Memory relink input could not be validated",
            exit_code=_MEMORY_EXIT_USAGE,
        )
    except RepositoryRelinkError:
        return _emit_memory_error(
            args,
            code="authority_resolution_failed",
            message="repository Memory authority could not be resolved safely",
            exit_code=_MEMORY_EXIT_CONFLICT,
        )
    except (TypeError, ValueError):
        return _emit_memory_error(
            args,
            code="input_invalid",
            message="the memory command input or repository could not be validated",
            exit_code=_MEMORY_EXIT_USAGE,
        )
    except (OSError, RuntimeError):
        return _emit_memory_error(
            args,
            code="operational_failure",
            message="the memory command failed safely during an operation",
            exit_code=_MEMORY_EXIT_OPERATIONAL,
        )
    except Exception:
        return _emit_memory_error(
            args,
            code="internal_error",
            message="the memory command failed safely",
            exit_code=_MEMORY_EXIT_OPERATIONAL,
        )


def _resolve_memory_context(
    args: argparse.Namespace,
    *,
    write: bool,
    require_store: bool = False,
) -> _MemoryCommandContext:
    try:
        requested_repository = Path(args.repo).resolve(strict=False)
        revision_resolver = RevisionResolver()
        repository_identity = revision_resolver.repository_identity(requested_repository)
        repository = Path(repository_identity.canonical_path)
    except Exception:
        raise _MemoryCLIError(
            "repository_invalid",
            "the requested path is not an accessible Git repository",
            _MEMORY_EXIT_USAGE,
        ) from None

    resolution = MemoryRootResolver().resolve(
        getattr(args, "memory_root", None),
        create=False,
    )
    namespace_plan = plan_repository_memory_namespace(
        repository_identity,
        resolution,
        revision_resolver=revision_resolver,
    )

    def build_context(
        authority: RepositoryAuthorityResolution,
        namespace: RepositoryMemoryNamespace,
        store: MemoryStore | None,
    ) -> _MemoryCommandContext:
        return _MemoryCommandContext(
            repository=repository,
            repository_key=authority.authority_repository_key,
            locator_repository_key=authority.locator_repository_key,
            authority_resolution_hash=authority.authority_resolution_hash,
            binding_id=authority.binding_id,
            registry_generation=authority.registry_generation,
            memory_root=Path(resolution.path),
            root_source=resolution.source.value,
            namespace=namespace,
            store=store,
        )

    if write:
        # Relink uses this same locator lock.  Re-resolving under the lock and
        # materializing a direct Store before releasing it closes the only
        # direct-vs-bound creation race.
        with MemoryStore.lock_namespaces(namespace_plan.namespace):
            authority = resolve_repository_authority(
                resolution,
                namespace_plan.locator.identity,
                revision_resolver=revision_resolver,
            )
            if authority.binding_id is None:
                database_path = (
                    Path(namespace_plan.namespace.namespace_path)
                    / "memory.sqlite3"
                )
                if require_store and not database_path.is_file():
                    raise _MemoryCLIError(
                        "store_not_found",
                        "no Memory Store exists for this repository authority",
                        _MEMORY_EXIT_NOT_FOUND,
                    )
                namespace = materialize_repository_memory_namespace(
                    namespace_plan,
                    revision_resolver=revision_resolver,
                )
                store = MemoryStore(namespace)
                return build_context(authority, namespace, store)
    else:
        authority = resolve_repository_authority(
            resolution,
            namespace_plan.locator.identity,
            revision_resolver=revision_resolver,
        )

    direct = authority.binding_id is None
    if direct:
        namespace = namespace_plan.namespace
    else:
        namespace = RepositoryMemoryNamespace(
            repository_key=authority.authority_repository_key,
            memory_root=str(resolution.path),
            namespace_path=str(
                repository_namespace_path(
                    resolution,
                    authority.authority_repository_key,
                )
            ),
            metadata=authority.authority_identity,
        )
    database_path = Path(namespace.namespace_path) / "memory.sqlite3"
    if (require_store or not direct) and not database_path.is_file():
        raise _MemoryCLIError(
            "store_not_found",
            "no Memory Store exists for this repository authority",
            _MEMORY_EXIT_NOT_FOUND,
        )
    store: MemoryStore | None
    if write:
        # The registry descriptor is an audited relink-time snapshot, not a
        # claim that its descriptive canonical path remains current. Verify
        # the bound authority core without re-registering that offline
        # snapshot over newer Store metadata.
        MemoryStore(namespace, read_only=True)
        store = MemoryStore(Path(namespace.namespace_path))
    else:
        store = (
            MemoryStore(namespace, read_only=True)
            if database_path.is_file()
            else None
        )
    if require_store and store is None:
        raise _MemoryCLIError(
            "store_not_found",
            "no Memory Store exists for this repository namespace",
            _MEMORY_EXIT_NOT_FOUND,
        )
    return build_context(authority, namespace, store)


def _memory_status(args: argparse.Namespace) -> int:
    context = _resolve_memory_context(args, write=False)
    generations = _context_generations(context)
    candidates: Sequence[MemoryCandidate] = ()
    records: Sequence[DurableMemoryRecord] = ()
    knowledge_count = 0
    feedback_count = 0
    event_count = 0
    integrity: Mapping[str, Any] | None = None
    if context.store is not None:
        candidates = context.store.list_candidates(context.repository_key)
        view = context.store.read_view(context.repository_key)
        records = view.records
        knowledge_count = len(view.knowledge_entries)
        feedback_count = len(view.feedback)
        event_count = context.store.verify_event_chain(context.repository_key)
        report = context.store.validate_integrity()
        integrity = {
            "repository_count": report.repository_count,
            "event_count": report.event_count,
            "blob_count": report.blob_count,
            "candidate_count": report.candidate_count,
            "record_count": report.record_count,
            "feedback_count": report.feedback_count,
            "knowledge_count": report.knowledge_count,
        }
    payload = _memory_envelope(context, "status", generations)
    payload.update(
        {
            "store_present": context.store is not None,
            "root_source": context.root_source,
            "counts": {
                "candidates": len(candidates),
                "records": len(records),
                "feedback": feedback_count,
                "knowledge": knowledge_count,
                "events": event_count,
            },
            "candidate_status_counts": _enum_status_counts(candidates),
            "record_status_counts": _enum_status_counts(records),
            "integrity": integrity,
        }
    )
    lines = [
        "Memory status",
        "Repository: %s" % context.repository_key,
        "Store: %s" % ("present" if context.store is not None else "absent"),
        "Generation: %d" % generations.memory_generation,
        "Feedback generation: %d" % generations.feedback_generation,
        "Knowledge generation: %d" % generations.knowledge_generation,
        "Candidates: %d" % len(candidates),
        "Records: %d" % len(records),
        "Events: %d" % event_count,
    ]
    _emit_memory_document(args, payload, lines)
    return 0


def _memory_list(args: argparse.Namespace) -> int:
    context = _resolve_memory_context(args, write=False)
    generations = _context_generations(context)
    requested_status = getattr(args, "record_status", None)
    status = None if requested_status is None else RecordStatus(requested_status)
    records = (
        ()
        if context.store is None
        else context.store.list_records(context.repository_key, status=status)
    )
    payload = _memory_envelope(context, "record_list", generations)
    payload.update(
        {
            "status_filter": requested_status,
            "count": len(records),
            "records": [record.to_dict() for record in records],
        }
    )
    lines = [
        "Durable Memory records",
        "Repository: %s" % context.repository_key,
        "Generation: %d" % generations.memory_generation,
        "Count: %d" % len(records),
    ]
    lines.extend(
        "%s  %s  %s  %s"
        % (
            record.memory_id,
            record.status.value,
            record.kind.value,
            _terminal_text(record.statement, limit=120),
        )
        for record in records
    )
    _emit_memory_document(args, payload, lines)
    return 0


def _memory_show(args: argparse.Namespace) -> int:
    context = _resolve_memory_context(args, write=False, require_store=True)
    assert context.store is not None
    record = context.store.get_record(args.memory_id)
    _require_repository_subject(record.repository_key, context.repository_key)
    generations = context.store.get_generations(context.repository_key)
    payload = _memory_envelope(context, "record", generations)
    payload["record"] = record.to_dict()
    lines = _record_human_lines(record, generations)
    _emit_memory_document(args, payload, lines)
    return 0


def _memory_candidates(args: argparse.Namespace) -> int:
    context = _resolve_memory_context(args, write=False)
    generations = _context_generations(context)
    requested_status = getattr(args, "candidate_status", None)
    status = None if requested_status is None else CandidateStatus(requested_status)
    candidates = (
        ()
        if context.store is None
        else context.store.list_candidates(context.repository_key, status=status)
    )
    payload = _memory_envelope(context, "candidate_list", generations)
    payload.update(
        {
            "status_filter": requested_status,
            "count": len(candidates),
            "candidates": [candidate.to_dict() for candidate in candidates],
        }
    )
    lines = [
        "Memory candidates",
        "Repository: %s" % context.repository_key,
        "Generation: %d" % generations.memory_generation,
        "Count: %d" % len(candidates),
    ]
    lines.extend(
        "%s  %s  %s  %s"
        % (
            candidate.candidate_id,
            candidate.status.value,
            candidate.kind.value,
            _terminal_text(candidate.statement, limit=120),
        )
        for candidate in candidates
    )
    _emit_memory_document(args, payload, lines)
    return 0


def _memory_candidate_show(args: argparse.Namespace) -> int:
    context = _resolve_memory_context(args, write=False, require_store=True)
    assert context.store is not None
    candidate = context.store.get_candidate(args.candidate_id)
    _require_repository_subject(candidate.repository_key, context.repository_key)
    generations = context.store.get_generations(context.repository_key)
    payload = _memory_envelope(context, "candidate", generations)
    payload["candidate"] = candidate.to_dict()
    lines = _candidate_human_lines(candidate, generations)
    _emit_memory_document(args, payload, lines)
    return 0


def _memory_approve(args: argparse.Namespace) -> int:
    actor, reason = _decision_fields(args)
    preview_context = _resolve_memory_context(
        args,
        write=False,
        require_store=True,
    )
    assert preview_context.store is not None
    candidate = preview_context.store.get_candidate(args.candidate_id)
    _require_repository_subject(
        candidate.repository_key,
        preview_context.repository_key,
    )
    generations = preview_context.store.get_generations(
        preview_context.repository_key
    )
    proposed_candidate = replace(candidate, status=CandidateStatus.PROPOSED)
    authority = _candidate_runtime_authority(preview_context, proposed_candidate)
    validation = authority.validator.validate_candidate(
        proposed_candidate,
        runtime_provenance=authority.provenance,
    )
    preview = _candidate_decision_preview(
        preview_context,
        action="approve",
        candidate=candidate,
        generations=generations,
        validation=validation,
        policy_before=None,
    )
    _emit_memory_document(
        args,
        preview,
        _candidate_preview_human_lines(preview),
    )
    _confirm_memory_write(args, "approve", candidate.candidate_id)

    context = _resolve_memory_context(args, write=True, require_store=True)
    assert context.store is not None
    current = context.store.get_candidate(candidate.candidate_id)
    _require_repository_subject(current.repository_key, context.repository_key)
    current_head = RevisionResolver().resolve_commit(context.repository, "HEAD")
    if (
        context.authority_resolution_hash
        != preview_context.authority_resolution_hash
        or current_head != authority.provenance.target_head_sha
    ):
        raise _MemoryCLIError(
            "repository_changed",
            "the repository authority changed after the approval preview",
            _MEMORY_EXIT_CONFLICT,
        )
    current_authority = _candidate_runtime_authority(
        context,
        replace(current, status=CandidateStatus.PROPOSED),
        target_head=current_head,
    )
    if current_authority.provenance != authority.provenance:
        raise _MemoryCLIError(
            "repository_changed",
            "the candidate authority changed after the approval preview",
            _MEMORY_EXIT_CONFLICT,
        )
    lifecycle = MemoryLifecycle(context.store, current_authority.validator)
    result = lifecycle.approve_candidate(
        current.candidate_id,
        runtime_provenance=current_authority.provenance,
        actor=actor,
        reason=reason,
        request_id=_decision_request_id(
            args,
            "approve",
            current.candidate_id,
            actor,
            "approved",
            reason,
        ),
        expected_generation=generations.memory_generation,
    )
    payload = _approval_result_payload(context, "approve", result)
    lines = [
        "Memory approval committed",
        "Repository: %s" % context.repository_key,
        "Candidate: %s" % result.record.candidate_id,
        "Memory: %s" % result.record.memory_id,
        "Event: %s" % result.write_result.event_id,
        "Generation: %d" % result.write_result.generations.memory_generation,
    ]
    _emit_memory_document(args, payload, lines)
    return 0


def _memory_reject(args: argparse.Namespace) -> int:
    actor, reason = _decision_fields(args)
    reason_code = _required_cli_text(args.reason_code, "reason code")
    preview_context = _resolve_memory_context(
        args,
        write=False,
        require_store=True,
    )
    assert preview_context.store is not None
    candidate = preview_context.store.get_candidate(args.candidate_id)
    _require_repository_subject(
        candidate.repository_key,
        preview_context.repository_key,
    )
    generations = preview_context.store.get_generations(
        preview_context.repository_key
    )
    preview = _status_decision_preview(
        preview_context,
        action="reject",
        subject_type="candidate",
        subject_id=candidate.candidate_id,
        current_status=candidate.status.value,
        requested_status=CandidateStatus.REJECTED.value,
        generations=generations,
        reason_code=reason_code,
    )
    _emit_memory_document(args, preview, _status_preview_human_lines(preview))
    _confirm_memory_write(args, "reject", candidate.candidate_id)

    context = _resolve_memory_context(args, write=True, require_store=True)
    assert context.store is not None
    _require_same_memory_authority(preview_context, context)
    lifecycle = MemoryLifecycle(context.store, SourceValidator(context.repository))
    result = lifecycle.reject_candidate(
        candidate.candidate_id,
        actor=actor,
        reason_code=reason_code,
        reason=reason,
        request_id=_decision_request_id(
            args,
            "reject",
            candidate.candidate_id,
            actor,
            reason_code,
            reason,
        ),
        expected_generation=generations.memory_generation,
    )
    payload = _write_result_payload(
        context,
        "reject_result",
        result,
        candidate_id=candidate.candidate_id,
    )
    lines = _write_result_human_lines(
        "Candidate rejection committed",
        context,
        result,
    )
    _emit_memory_document(args, payload, lines)
    return 0


def _memory_revoke(args: argparse.Namespace) -> int:
    actor, reason = _decision_fields(args)
    preview_context = _resolve_memory_context(
        args,
        write=False,
        require_store=True,
    )
    assert preview_context.store is not None
    record = preview_context.store.get_record(args.memory_id)
    _require_repository_subject(record.repository_key, preview_context.repository_key)
    generations = preview_context.store.get_generations(
        preview_context.repository_key
    )
    preview = _status_decision_preview(
        preview_context,
        action="revoke",
        subject_type="record",
        subject_id=record.memory_id,
        current_status=record.status.value,
        requested_status=RecordStatus.REVOKED.value,
        generations=generations,
        reason_code="revoked",
    )
    _emit_memory_document(args, preview, _status_preview_human_lines(preview))
    _confirm_memory_write(args, "revoke", record.memory_id)

    context = _resolve_memory_context(args, write=True, require_store=True)
    assert context.store is not None
    _require_same_memory_authority(preview_context, context)
    lifecycle = MemoryLifecycle(context.store, SourceValidator(context.repository))
    result = lifecycle.revoke_record(
        record.memory_id,
        actor=actor,
        reason=reason,
        request_id=_decision_request_id(
            args,
            "revoke",
            record.memory_id,
            actor,
            "revoked",
            reason,
        ),
        expected_generation=generations.memory_generation,
    )
    payload = _write_result_payload(
        context,
        "revoke_result",
        result,
        memory_id=record.memory_id,
    )
    lines = _write_result_human_lines(
        "Memory revocation committed",
        context,
        result,
    )
    _emit_memory_document(args, payload, lines)
    return 0


def _memory_revalidate(args: argparse.Namespace) -> int:
    actor, reason = _decision_fields(args)
    preview_context = _resolve_memory_context(
        args,
        write=False,
        require_store=True,
    )
    assert preview_context.store is not None
    predecessor = preview_context.store.get_record(args.memory_id)
    replacement = preview_context.store.get_candidate(args.candidate_id)
    _require_repository_subject(
        predecessor.repository_key,
        preview_context.repository_key,
    )
    _require_repository_subject(
        replacement.repository_key,
        preview_context.repository_key,
    )
    proposed_replacement = replace(
        replacement,
        status=CandidateStatus.PROPOSED,
    )
    generations = preview_context.store.get_generations(
        preview_context.repository_key
    )
    authority = _candidate_runtime_authority(
        preview_context,
        proposed_replacement,
    )
    validation = authority.validator.validate_candidate(
        proposed_replacement,
        runtime_provenance=authority.provenance,
    )
    preview = _candidate_decision_preview(
        preview_context,
        action="revalidate",
        candidate=proposed_replacement,
        generations=generations,
        validation=validation,
        policy_before=(
            None
            if predecessor.policy_effect is None
            else predecessor.policy_effect.to_dict()
        ),
    )
    preview["predecessor_memory_id"] = predecessor.memory_id
    preview["predecessor_candidate_id"] = predecessor.candidate_id
    preview["predecessor_status"] = predecessor.status.value
    _emit_memory_document(
        args,
        preview,
        _candidate_preview_human_lines(preview),
    )
    _confirm_memory_write(args, "revalidate", predecessor.memory_id)

    context = _resolve_memory_context(args, write=True, require_store=True)
    assert context.store is not None
    current_predecessor = context.store.get_record(predecessor.memory_id)
    current_replacement = replace(
        context.store.get_candidate(replacement.candidate_id),
        status=CandidateStatus.PROPOSED,
    )
    _require_repository_subject(
        current_predecessor.repository_key,
        context.repository_key,
    )
    _require_repository_subject(
        current_replacement.repository_key,
        context.repository_key,
    )
    current_head = RevisionResolver().resolve_commit(context.repository, "HEAD")
    if (
        context.authority_resolution_hash
        != preview_context.authority_resolution_hash
        or current_head != authority.provenance.target_head_sha
    ):
        raise _MemoryCLIError(
            "repository_changed",
            "the repository authority changed after the revalidation preview",
            _MEMORY_EXIT_CONFLICT,
        )
    current_authority = _candidate_runtime_authority(
        context,
        current_replacement,
        target_head=current_head,
    )
    if current_authority.provenance != authority.provenance:
        raise _MemoryCLIError(
            "repository_changed",
            "the candidate authority changed after the revalidation preview",
            _MEMORY_EXIT_CONFLICT,
        )
    lifecycle = MemoryLifecycle(context.store, current_authority.validator)
    result = lifecycle.revalidate_record(
        current_predecessor.memory_id,
        current_replacement,
        runtime_provenance=current_authority.provenance,
        actor=actor,
        reason=reason,
        request_id=_decision_request_id(
            args,
            "revalidate",
            current_predecessor.memory_id,
            actor,
            current_replacement.candidate_id,
            reason,
        ),
        expected_generation=generations.memory_generation,
    )
    payload = _approval_result_payload(context, "revalidate", result)
    payload["predecessor_memory_id"] = current_predecessor.memory_id
    lines = [
        "Memory revalidation committed",
        "Repository: %s" % context.repository_key,
        "Predecessor: %s" % current_predecessor.memory_id,
        "Replacement candidate: %s" % result.record.candidate_id,
        "Replacement memory: %s" % result.record.memory_id,
        "Event: %s" % result.write_result.event_id,
        "Generation: %d" % result.write_result.generations.memory_generation,
    ]
    _emit_memory_document(args, payload, lines)
    return 0


def _memory_feedback_list(args: argparse.Namespace) -> int:
    context = _resolve_memory_context(args, write=False)
    generations = _context_generations(context)
    requested_status = getattr(args, "feedback_status", None)
    status = None if requested_status is None else FeedbackStatus(requested_status)
    records = (
        ()
        if context.store is None
        else context.store.list_feedback(context.repository_key, status=status)
    )
    requested_review = getattr(args, "review_id", None)
    requested_decision = getattr(args, "feedback_decision", None)
    if requested_review is not None:
        requested_review = _required_cli_text(requested_review, "review_id")
        records = tuple(
            record for record in records if record.review_id == requested_review
        )
    if requested_decision is not None:
        decision = FeedbackDecision(requested_decision)
        records = tuple(record for record in records if record.decision is decision)

    payload = _memory_envelope(context, "feedback_list", generations)
    payload.update(
        {
            "review_id_filter": requested_review,
            "decision_filter": requested_decision,
            "status_filter": requested_status,
            "count": len(records),
            "feedback": [record.to_dict() for record in records],
        }
    )
    lines = [
        "Human Feedback",
        "Repository: %s" % context.repository_key,
        "Feedback generation: %d" % generations.feedback_generation,
        "Records: %d" % len(records),
    ]
    lines.extend(
        "  %s  %s  %s  %s"
        % (
            record.feedback_id,
            record.review_id,
            record.finding_id,
            record.decision.value,
        )
        for record in records
    )
    _emit_memory_document(args, payload, lines)
    return 0


def _memory_feedback_record(args: argparse.Namespace) -> int:
    preview_context = _resolve_memory_context(args, write=False)
    request, declaration = _feedback_request(preview_context, args)
    declarations = () if declaration is None else (declaration,)
    source_validator = SourceValidator(
        preview_context.repository,
        sessions_root=preview_context.repository / ".review-agent" / "runs",
        human_declarations=declarations,
    )
    preparation_service = FeedbackImportService(
        None,
        source_validator,
    )
    prepared = preparation_service.prepare_feedback(request)
    prepared_record = prepared.record
    generations = _context_generations(preview_context)
    preview = _memory_envelope(
        preview_context,
        "feedback_record_preview",
        generations,
    )
    preview.update(
        {
            "request_id": request.request_id,
            "review_id": request.review_id,
            "finding_id": request.finding_id,
            "head_sha": request.head_sha,
            "finding_hash": request.finding_hash,
            "evidence_refs": list(request.evidence_refs),
            "decision": request.decision.value,
            "final_severity": request.final_severity.value,
            "reason_code": request.reason_code.value,
            "created_at": request.created_at,
            "missed": request.missed_finding is not None,
            "source_ref_count": len(prepared_record.source_refs),
            "preparation_hash": prepared.preparation_hash,
            "source_validation_hash": prepared.validation.report_hash,
            "safe_text_validated": True,
        }
    )
    _emit_memory_document(
        args,
        preview,
        [
            "Human Feedback write preview",
            "Repository: %s" % preview_context.repository_key,
            "Review: %s" % request.review_id,
            "Finding: %s" % request.finding_id,
            "Decision: %s" % request.decision.value,
            "Final severity: %s" % request.final_severity.value,
            "Actor and free text: validated",
        ],
    )
    _confirm_memory_write(args, "record Feedback", request.finding_id)

    context = _resolve_memory_context(args, write=True)
    _require_same_memory_authority(preview_context, context)
    assert context.store is not None
    service = FeedbackImportService(
        context.store,
        source_validator,
    )
    result = service.record_prepared(prepared)
    payload = _write_result_payload(
        context,
        "feedback_record",
        result.write_result,
    )
    payload.update(
        {
            "feedback": result.record.to_dict(),
            "validation": (
                None if result.validation is None else result.validation.to_dict()
            ),
        }
    )
    lines = [
        "Human Feedback recorded",
        "Repository: %s" % context.repository_key,
        "Feedback: %s" % result.record.feedback_id,
        "Review: %s" % result.record.review_id,
        "Finding: %s" % result.record.finding_id,
        "Decision: %s" % result.record.decision.value,
        "Event: %s" % result.write_result.event_id,
        "Feedback generation: %d"
        % result.write_result.generations.feedback_generation,
        "Replayed: %s" % ("yes" if result.write_result.replayed else "no"),
    ]
    _emit_memory_document(args, payload, lines)
    return 0


def _feedback_request(
    context: _MemoryCommandContext,
    args: argparse.Namespace,
) -> tuple[FeedbackImportRequest, HumanDeclarationAuthority | None]:
    actor, reason = _decision_fields(args)
    review_id = _required_cli_text(args.review_id, "review_id")
    finding_id = _required_cli_text(args.finding_id, "finding_id")
    created_at = (
        _utc_now()
        if getattr(args, "created_at", None) is None
        else _required_cli_text(args.created_at, "created_at")
    )
    evidence_refs = tuple(args.evidence_ref)
    decision = FeedbackDecision(args.decision)
    final_severity = FindingSeverity(args.final_severity)
    reason_code = FeedbackReasonCode(args.reason_code)
    request_id = getattr(args, "request_id", None)
    if request_id is None:
        request_id = stable_request_id(
            _MEMORY_CLI_SCHEMA,
            "feedback_record",
            context.repository_key,
            review_id,
            finding_id,
            args.head_sha,
            args.finding_hash,
            *evidence_refs,
            decision.value,
            final_severity.value,
            reason_code.value,
            actor,
            reason,
            created_at,
        )

    source_refs = tuple(
        _source_ref_from_cli_json(value)
        for value in getattr(args, "source_ref_json", ())
    )
    declaration: HumanDeclarationAuthority | None = None
    missed: MissedFindingInput | None = None
    missed_values_present = any(
        value is not None
        for value in (
            getattr(args, "missed_claim", None),
            getattr(args, "missed_path", None),
            getattr(args, "missed_line", None),
            getattr(args, "missed_original_severity", None),
            getattr(args, "human_declaration", None),
        )
    ) or bool(getattr(args, "missed_contracts", ()))
    if decision is FeedbackDecision.MISSED:
        declaration_text = _required_cli_text(
            getattr(args, "human_declaration", None),
            "human_declaration",
        )
        declaration_ref = HumanDeclarationSourceRef(
            request_id=request_id,
            actor=actor,
            declaration_hash=hashlib.sha256(
                declaration_text.encode("utf-8")
            ).hexdigest(),
            created_at=created_at,
            review_id=review_id,
        )
        declaration = HumanDeclarationAuthority(
            source_ref=declaration_ref,
            origin=HumanDeclarationOrigin.CLI_REQUEST,
            declaration=declaration_text,
        )
        original_severity = getattr(args, "missed_original_severity", None)
        if original_severity is None:
            raise _MemoryCLIError(
                "feedback_missed_input_required",
                "missed Feedback requires the original Finding severity",
                _MEMORY_EXIT_USAGE,
            )
        line = getattr(args, "missed_line", None)
        if line is None:
            raise _MemoryCLIError(
                "feedback_missed_input_required",
                "missed Feedback requires a Finding line",
                _MEMORY_EXIT_USAGE,
            )
        missed = MissedFindingInput(
            finding_id=finding_id,
            claim=_required_cli_text(args.missed_claim, "missed_claim"),
            path=_required_cli_text(args.missed_path, "missed_path"),
            line=line,
            contracts=tuple(getattr(args, "missed_contracts", ())),
            original_severity=FindingSeverity(original_severity),
            evidence_refs=evidence_refs,
        )
        source_refs = (declaration.source_ref, *source_refs)
    elif missed_values_present or source_refs:
        raise _MemoryCLIError(
            "feedback_source_not_allowed",
            "typed missed-Finding sources are only valid for decision=missed",
            _MEMORY_EXIT_USAGE,
        )

    try:
        request = FeedbackImportRequest(
            request_id=request_id,
            repository_key=context.repository_key,
            review_id=review_id,
            finding_id=finding_id,
            head_sha=args.head_sha,
            finding_hash=args.finding_hash,
            evidence_refs=evidence_refs,
            decision=decision,
            final_severity=final_severity,
            reason_code=reason_code,
            reason=reason,
            actor=actor,
            created_at=created_at,
            missed_finding=missed,
            human_declaration=declaration,
            source_refs=source_refs,
        )
    except (TypeError, ValueError) as error:
        raise _MemoryCLIError(
            "feedback_input_invalid",
            "the Feedback request is not canonical",
            _MEMORY_EXIT_USAGE,
        ) from error
    return request, declaration


def _source_ref_from_cli_json(value: str) -> SourceRef:
    if not isinstance(value, str) or not value:
        raise _MemoryCLIError(
            "feedback_source_invalid",
            "a Feedback source ref must be canonical JSON",
            _MEMORY_EXIT_USAGE,
        )
    try:
        if value.startswith("@"):
            source_path = Path(value[1:])
            if source_path.stat().st_size > 1_048_576:
                raise ValueError("source ref input is too large")
            raw = source_path.read_text(encoding="utf-8")
        else:
            raw = value
        payload = json.loads(raw, object_pairs_hook=_strict_cli_json_object)
        return SourceRef.from_dict(payload)
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise _MemoryCLIError(
            "feedback_source_invalid",
            "a Feedback source ref failed strict hydration",
            _MEMORY_EXIT_USAGE,
        ) from error


def _strict_cli_json_object(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _memory_replay_outbox(args: argparse.Namespace) -> int:
    context = _resolve_memory_context(args, write=False)
    actor, reason = _decision_fields(args)
    review_id = _required_cli_text(args.review_id, "review_id")
    prepared_outbox = _prepare_outbox_replay(
        context,
        review_id,
    )
    request_id = _decision_request_id(
        args,
        "replay_outbox",
        review_id,
        actor,
        "memory_candidate_outbox_v1",
        reason,
    )
    try:
        request_id = validate_stable_id(
            request_id,
            "REQ",
            "outbox replay request_id",
        )
    except (TypeError, ValueError) as error:
        raise _MemoryCLIError(
            "outbox_request_invalid",
            "the outbox replay request ID is not canonical",
            _MEMORY_EXIT_USAGE,
        ) from error
    generations = _context_generations(context)
    preview = _memory_envelope(
        context,
        "replay_outbox_preview",
        generations,
    )
    preview.update(
        {
            "review_id": review_id,
            "request_id": request_id,
            "actor": actor,
            "required_artifact_schema": _MEMORY_OUTBOX_SCHEMA,
            "validation": "delegated_to_memory_outbox_persistence_service",
            "expected_outbox_digest": prepared_outbox.outbox_digest,
            "batch_digest": prepared_outbox.batch_digest,
            "entry_count": len(prepared_outbox.entries),
        }
    )
    _emit_memory_document(
        args,
        preview,
        [
            "Memory outbox replay preview",
            "Repository: %s" % context.repository_key,
            "Review: %s" % review_id,
            "Actor: %s" % actor,
            "Artifact validation: required",
        ],
    )
    _confirm_memory_write(args, "replay outbox", review_id)

    receipt = _replay_memory_outbox_via_service(
        repository=context.repository,
        review_id=review_id,
        memory_root=context.memory_root,
        actor=actor,
        reason=reason,
        request_id=request_id,
        expected_repository_key=context.repository_key,
        expected_locator_repository_key=context.locator_repository_key,
        expected_authority_resolution_hash=context.authority_resolution_hash,
        expected_binding_id=context.binding_id,
        expected_outbox_digest=prepared_outbox.outbox_digest,
        expected_batch_digest=prepared_outbox.batch_digest,
        expected_entries=prepared_outbox.entries,
    )
    result_context = _resolve_memory_context(args, write=False)
    _require_same_memory_authority(context, result_context)
    payload = _memory_envelope(
        result_context,
        "replay_outbox_result",
        _context_generations(result_context),
    )
    payload.update({"review_id": review_id, "receipt": receipt})
    _emit_memory_document(
        args,
        payload,
        [
            "Memory outbox replay completed",
            "Repository: %s" % result_context.repository_key,
            "Review: %s" % review_id,
            "Receipt: %s" % _terminal_json(receipt),
        ],
    )
    return 0


def _prepare_outbox_replay(
    context: _MemoryCommandContext,
    review_id: str,
) -> _PreparedOutboxReplay:
    """Read and bind one completed, hash-valid Session outbox for CAS."""

    if (
        not isinstance(review_id, str)
        or not 1 <= len(review_id) <= 128
        or not review_id[0].isalnum()
        or any(
            not (character.isalnum() or character in "._-")
            for character in review_id
        )
    ):
        raise _MemoryCLIError(
            "outbox_review_invalid",
            "the outbox review ID is not canonical",
            _MEMORY_EXIT_USAGE,
        )
    try:
        sessions_root = (
            context.repository / ".review-agent" / "runs"
        ).resolve(strict=True)
        run_dir = sessions_root / review_id
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise ValueError("unsafe Session run directory")
        resolved_run_dir = run_dir.resolve(strict=True)
        resolved_run_dir.relative_to(sessions_root)
    except FileNotFoundError as error:
        raise _MemoryCLIError(
            "outbox_session_not_found",
            "the requested outbox Session does not exist",
            _MEMORY_EXIT_NOT_FOUND,
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise _MemoryCLIError(
            "outbox_session_untrusted",
            "the requested outbox Session path is not trusted",
            _MEMORY_EXIT_CONFLICT,
        ) from error

    session_store = SessionStore(resolved_run_dir)
    try:
        manifest = session_store.load()
        session_repository_key = canonical_repository_key(manifest.repository)
    except FileNotFoundError as error:
        raise _MemoryCLIError(
            "outbox_session_not_found",
            "the requested outbox Session does not exist",
            _MEMORY_EXIT_NOT_FOUND,
        ) from error
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise _MemoryCLIError(
            "outbox_session_untrusted",
            "the requested outbox Session failed strict hydration",
            _MEMORY_EXIT_CONFLICT,
        ) from error
    if (
        manifest.review_id != review_id
        or manifest.status is not RunStatus.COMPLETED
        or session_repository_key != context.locator_repository_key
    ):
        raise _MemoryCLIError(
            "outbox_session_conflict",
            "the requested outbox Session state or repository authority conflicts",
            _MEMORY_EXIT_CONFLICT,
        )

    descriptor = manifest.artifacts.get("memory_outbox")
    if descriptor is None:
        raise _MemoryCLIError(
            "outbox_not_found",
            "the Session has no committed Memory outbox",
            _MEMORY_EXIT_NOT_FOUND,
        )
    checkpoint = manifest.phases.get(RunPhase.MEMORY_PROPOSAL.value)
    expected_revision = (
        manifest.revisions.resolved_base_sha
        + ".."
        + manifest.revisions.resolved_head_sha
    )
    try:
        valid_descriptor = (
            artifact_schema("memory_outbox") == _MEMORY_OUTBOX_SCHEMA
            and descriptor.schema == _MEMORY_OUTBOX_SCHEMA
            and descriptor.phase is RunPhase.MEMORY_PROPOSAL
            and descriptor.revision_binding is not None
            and descriptor.revision_binding.casefold()
            == expected_revision.casefold()
            and checkpoint is not None
            and checkpoint.status is PhaseStatus.COMPLETED
            and descriptor.name in checkpoint.artifacts
            and session_store.validate_artifact(descriptor)
            and _is_canonical_sha256(descriptor.sha256)
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        valid_descriptor = False
    if not valid_descriptor:
        raise _MemoryCLIError(
            "outbox_untrusted",
            "the Session Memory outbox failed descriptor, revision, or hash validation",
            _MEMORY_EXIT_CONFLICT,
        )
    try:
        payload = _read_bounded_session_json(
            session_store,
            descriptor,
            max_bytes=8 * 1024 * 1024,
        )
        return _hydrate_memory_outbox(
            payload,
            review_id=review_id,
            expected_repository_key=context.repository_key,
            expected_locator_repository_key=context.locator_repository_key,
            expected_authority_resolution_hash=context.authority_resolution_hash,
            expected_binding_id=context.binding_id,
            expected_head_sha=manifest.revisions.resolved_head_sha,
        )
    except _MemoryCLIError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise _MemoryCLIError(
            "outbox_untrusted",
            "the Session Memory outbox failed strict payload hydration",
            _MEMORY_EXIT_CONFLICT,
        ) from error


def _read_bounded_session_json(
    session_store: SessionStore,
    descriptor: Any,
    *,
    max_bytes: int,
) -> Mapping[str, Any]:
    root = session_store.run_dir.resolve(strict=True)
    relative_parts = PurePosixPath(descriptor.path).parts
    candidate = root.joinpath(*relative_parts)
    current = root
    for index, part in enumerate(relative_parts):
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("artifact path contains a symlink")
        if index < len(relative_parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("artifact parent is not a directory")
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root)
    metadata = resolved.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
        raise ValueError("artifact is not a bounded regular file")
    content = resolved.read_bytes()
    if len(content) > max_bytes:
        raise ValueError("artifact grew beyond its bound")
    if hashlib.sha256(content).hexdigest() != descriptor.sha256:
        raise ValueError("artifact content hash changed")
    payload = json.loads(
        content.decode("utf-8"),
        object_pairs_hook=_strict_cli_json_object,
    )
    if type(payload) is not dict or not session_store.validate_artifact(descriptor):
        raise ValueError("artifact changed during strict hydration")
    return payload


def _hydrate_memory_outbox(
    payload: Mapping[str, Any],
    *,
    review_id: str,
    expected_repository_key: str,
    expected_locator_repository_key: str,
    expected_authority_resolution_hash: str,
    expected_binding_id: str | None,
    expected_head_sha: str,
) -> _PreparedOutboxReplay:
    fields = {
        "schema",
        "review_id",
        "repository_key",
        "locator_repository_key",
        "authority_resolution_hash",
        "binding_id",
        "head_sha",
        "snapshot_id",
        "batch_digest",
        "actor_type",
        "actor_id",
        "reason_code",
        "entries",
        "outbox_digest",
    }
    if type(payload) is not dict or set(payload) != fields:
        raise _MemoryCLIError(
            "outbox_untrusted",
            "the Session Memory outbox schema is invalid",
            _MEMORY_EXIT_CONFLICT,
        )
    if (
        payload["schema"] != _MEMORY_OUTBOX_SCHEMA
        or payload["review_id"] != review_id
        or payload["repository_key"] != expected_repository_key
        or payload["locator_repository_key"]
        != expected_locator_repository_key
        or payload["authority_resolution_hash"]
        != expected_authority_resolution_hash
        or payload["binding_id"] != expected_binding_id
        or payload["head_sha"] != expected_head_sha
        or not _is_canonical_sha256(payload["batch_digest"])
        or not _is_canonical_sha256(payload["outbox_digest"])
        or not _canonical_outbox_text(payload["actor_type"])
        or not _canonical_outbox_text(payload["actor_id"])
        or not _canonical_outbox_text(payload["reason_code"])
    ):
        raise _MemoryCLIError(
            "outbox_untrusted",
            "the Session Memory outbox authority binding is invalid",
            _MEMORY_EXIT_CONFLICT,
        )
    try:
        validate_stable_id(payload["snapshot_id"], "MSNAP", "outbox snapshot_id")
    except (TypeError, ValueError) as error:
        raise _MemoryCLIError(
            "outbox_untrusted",
            "the Session Memory outbox snapshot binding is invalid",
            _MEMORY_EXIT_CONFLICT,
        ) from error

    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) > 64:
        raise _MemoryCLIError(
            "outbox_untrusted",
            "the Session Memory outbox entries are invalid",
            _MEMORY_EXIT_CONFLICT,
        )
    entries: list[tuple[str, str]] = []
    candidate_ids: list[str] = []
    request_ids: list[str] = []
    try:
        for entry in raw_entries:
            if type(entry) is not dict or set(entry) != {
                "candidate_id",
                "candidate_hash",
                "request_id",
                "origin",
                "allowed_source_refs",
            }:
                raise ValueError("outbox entry fields are invalid")
            candidate_id = validate_stable_id(
                entry["candidate_id"],
                "MC",
                "outbox candidate_id",
            )
            request_id = validate_stable_id(
                entry["request_id"],
                "REQ",
                "outbox request_id",
            )
            if not _is_canonical_sha256(entry["candidate_hash"]):
                raise ValueError("outbox candidate_hash is invalid")
            if entry["origin"] not in {
                ProducerType.LOCAL.value,
                ProducerType.MODEL.value,
            }:
                raise ValueError("outbox candidate origin is invalid")
            raw_source_refs = entry["allowed_source_refs"]
            if not isinstance(raw_source_refs, list) or not raw_source_refs:
                raise ValueError("outbox source allowlist is invalid")
            source_refs = tuple(
                SourceRef.from_dict(item) for item in raw_source_refs
            )
            if [item.to_dict() for item in source_refs] != raw_source_refs:
                raise ValueError("outbox source allowlist is not canonical")
            candidate_ids.append(candidate_id)
            request_ids.append(request_id)
            entries.append((candidate_id, request_id))
    except (TypeError, ValueError) as error:
        raise _MemoryCLIError(
            "outbox_untrusted",
            "the Session Memory outbox entries failed strict hydration",
            _MEMORY_EXIT_CONFLICT,
        ) from error
    if (
        candidate_ids != sorted(set(candidate_ids))
        or len(request_ids) != len(set(request_ids))
    ):
        raise _MemoryCLIError(
            "outbox_untrusted",
            "the Session Memory outbox entries are not canonical",
            _MEMORY_EXIT_CONFLICT,
        )

    body = {key: value for key, value in payload.items() if key != "outbox_digest"}
    if canonical_sha256(body) != payload["outbox_digest"]:
        raise _MemoryCLIError(
            "outbox_untrusted",
            "the Session Memory outbox digest is invalid",
            _MEMORY_EXIT_CONFLICT,
        )
    return _PreparedOutboxReplay(
        outbox_digest=payload["outbox_digest"],
        batch_digest=payload["batch_digest"],
        entries=tuple(entries),
    )


def _canonical_outbox_text(value: Any) -> bool:
    return (
        type(value) is str
        and value == value.strip()
        and 1 <= len(value) <= 512
        and all(character.isprintable() for character in value)
    )


def _replay_memory_outbox_via_service(
    *,
    repository: Path,
    review_id: str,
    memory_root: Path,
    actor: str,
    reason: str,
    request_id: str,
    expected_repository_key: str,
    expected_locator_repository_key: str,
    expected_authority_resolution_hash: str,
    expected_binding_id: str | None,
    expected_outbox_digest: str,
    expected_batch_digest: str,
    expected_entries: tuple[tuple[str, str], ...],
) -> Mapping[str, Any]:
    """Final CLI seam for Task 15's authoritative outbox service.

    Artifact descriptor/hash/revision checks and idempotent candidate writes
    belong to that service.  The CLI intentionally never rehydrates an outbox
    or writes candidates itself.
    """

    pipeline_module = importlib.import_module("review_agent.pipeline")
    replay = getattr(pipeline_module, "replay_memory_outbox", None)
    if not callable(replay):
        raise _MemoryCLIError(
            "outbox_service_unavailable",
            "the authoritative Memory outbox replay service is not available",
            _MEMORY_EXIT_OPERATIONAL,
        )
    try:
        result = replay(
            repository=repository,
            review_id=review_id,
            memory_root=memory_root,
            actor=actor,
            reason=reason,
            request_id=request_id,
            expected_repository_key=expected_repository_key,
            expected_authority_resolution_hash=(
                expected_authority_resolution_hash
            ),
            expected_outbox_digest=expected_outbox_digest,
        )
        if hasattr(result, "to_dict") and callable(result.to_dict):
            result = result.to_dict()
        if not isinstance(result, Mapping):
            raise _invalid_outbox_receipt()
        canonical = json.loads(
            json.dumps(
                dict(result),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            object_pairs_hook=_strict_cli_json_object,
        )
        return _validate_outbox_replay_receipt(
            canonical,
            review_id=review_id,
            expected_repository_key=expected_repository_key,
            expected_locator_repository_key=expected_locator_repository_key,
            expected_authority_resolution_hash=(
                expected_authority_resolution_hash
            ),
            expected_binding_id=expected_binding_id,
            expected_outbox_digest=expected_outbox_digest,
            expected_batch_digest=expected_batch_digest,
            expected_entries=expected_entries,
        )
    except (
        _MemoryCLIError,
        MemoryStoreError,
        RepositoryRelinkConflictError,
    ):
        raise
    except Exception as error:
        raise _MemoryCLIError(
            "outbox_service_failed",
            "the authoritative Memory outbox service failed safely",
            _MEMORY_EXIT_OPERATIONAL,
        ) from error


def _validate_outbox_replay_receipt(
    receipt: Mapping[str, Any],
    *,
    review_id: str,
    expected_repository_key: str,
    expected_locator_repository_key: str,
    expected_authority_resolution_hash: str,
    expected_binding_id: str | None,
    expected_outbox_digest: str,
    expected_batch_digest: str,
    expected_entries: tuple[tuple[str, str], ...],
) -> Mapping[str, Any]:
    required_fields = {
        "schema",
        "success",
        "review_id",
        "repository_key",
        "locator_repository_key",
        "authority_resolution_hash",
        "binding_id",
        "outbox_digest",
        "batch_digest",
        "persisted_candidate_ids",
        "replayed_candidate_ids",
        "results",
        "receipt_digest",
    }
    if type(receipt) is not dict or set(receipt) != required_fields:
        raise _invalid_outbox_receipt()
    if (
        receipt["schema"] != _MEMORY_PERSISTENCE_RECEIPT_SCHEMA
        or receipt["success"] is not True
        or receipt["review_id"] != review_id
        or receipt["repository_key"] != expected_repository_key
        or receipt["locator_repository_key"]
        != expected_locator_repository_key
        or receipt["authority_resolution_hash"]
        != expected_authority_resolution_hash
        or receipt["binding_id"] != expected_binding_id
        or receipt["outbox_digest"] != expected_outbox_digest
        or receipt["batch_digest"] != expected_batch_digest
        or not _is_canonical_sha256(receipt["receipt_digest"])
    ):
        raise _invalid_outbox_receipt()

    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_digest"
    }
    if canonical_sha256(receipt_body) != receipt["receipt_digest"]:
        raise _invalid_outbox_receipt()

    persisted = _canonical_candidate_id_list(
        receipt["persisted_candidate_ids"]
    )
    replayed = _canonical_candidate_id_list(
        receipt["replayed_candidate_ids"]
    )
    if persisted is None or replayed is None or set(persisted) & set(replayed):
        raise _invalid_outbox_receipt()
    expected_entry_map = dict(expected_entries)
    expected_results = set(expected_entry_map)
    if {*persisted, *replayed} != expected_results:
        raise _invalid_outbox_receipt()
    raw_results = receipt["results"]
    if not isinstance(raw_results, list) or len(raw_results) != len(
        expected_results
    ):
        raise _invalid_outbox_receipt()
    hydrated_results: list[
        tuple[
            str,
            str,
            bool,
            CandidateStatus,
            CandidateDedupeKind,
            tuple[WriteResult, ...],
        ]
    ] = []
    try:
        for row in raw_results:
            if type(row) is not dict or set(row) != {
                "candidate_id",
                "request_id",
                "replayed",
                "status",
                "dedupe",
                "validation_report_hash",
                "write_results",
            }:
                raise ValueError("receipt result fields are invalid")
            candidate_id = validate_stable_id(
                row["candidate_id"],
                "MC",
                "receipt candidate_id",
            )
            row_request_id = validate_stable_id(
                row["request_id"],
                "REQ",
                "receipt request_id",
            )
            if type(row["replayed"]) is not bool:
                raise ValueError("receipt replayed flag is invalid")
            status = CandidateStatus(row["status"])
            dedupe = CandidateDedupeKind(row["dedupe"])
            if not _is_canonical_sha256(row["validation_report_hash"]):
                raise ValueError("receipt validation report hash is invalid")
            raw_write_results = row["write_results"]
            if not isinstance(raw_write_results, list) or len(raw_write_results) > 4:
                raise ValueError("receipt write results are invalid")
            write_results = tuple(
                WriteResult.from_dict(item) for item in raw_write_results
            )
            if [item.to_dict() for item in write_results] != raw_write_results:
                raise ValueError("non-canonical write result")
            hydrated_results.append(
                (
                    candidate_id,
                    row_request_id,
                    row["replayed"],
                    status,
                    dedupe,
                    write_results,
                )
            )
    except (MemoryStoreError, TypeError, ValueError):
        raise _invalid_outbox_receipt() from None
    result_ids = [candidate_id for candidate_id, *_ in hydrated_results]
    if (
        len(result_ids) != len(set(result_ids))
        or result_ids != [candidate_id for candidate_id, _ in expected_entries]
        or set(result_ids) != expected_results
        or any(
            expected_entry_map[candidate_id] != row_request_id
            or replayed_flag != (candidate_id in set(replayed))
            or (not replayed_flag and not write_results)
            or any(
                result.operation not in {"put_candidate", "transition_candidate"}
                or result.subject_id != candidate_id
                for result in write_results
            )
            for (
                candidate_id,
                row_request_id,
                replayed_flag,
                _status,
                _dedupe,
                write_results,
            ) in hydrated_results
        )
    ):
        raise _invalid_outbox_receipt()
    return receipt


def _canonical_candidate_id_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    try:
        canonical = [
            validate_stable_id(item, "MC", "replay candidate_id")
            for item in value
        ]
    except (TypeError, ValueError):
        return None
    if canonical != sorted(set(canonical)):
        return None
    return canonical


def _is_canonical_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.casefold()
        and all(character in "0123456789abcdef" for character in value)
    )


def _invalid_outbox_receipt() -> _MemoryCLIError:
    return _MemoryCLIError(
        "outbox_service_invalid",
        "the authoritative Memory outbox service returned an invalid receipt",
        _MEMORY_EXIT_OPERATIONAL,
    )


def _memory_export(args: argparse.Namespace) -> int:
    context = _resolve_memory_context(args, write=False, require_store=True)
    assert context.store is not None
    generations = context.store.get_generations(context.repository_key)
    destination, destination_kind, manifest_path = _export_destination(
        context,
        args.path,
    )
    if manifest_path.exists():
        if not bool(getattr(args, "overwrite", False)):
            raise _MemoryCLIError(
                "export_exists",
                "the export manifest already exists; use --overwrite and confirm",
                _MEMORY_EXIT_CONFLICT,
            )
        preview = _memory_envelope(
            context,
            "export_overwrite_preview",
            generations,
        )
        preview.update(
            {
                "destination_kind": destination_kind,
                "existing_manifest": True,
                "redacted": True,
            }
        )
        _emit_memory_document(
            args,
            preview,
            [
                "Memory export overwrite preview",
                "Repository: %s" % context.repository_key,
                "Generation: %d" % generations.memory_generation,
                "Destination: existing %s" % destination_kind,
                "Redacted: yes",
            ],
        )
        _confirm_memory_write(args, "export overwrite", context.repository_key)
    if destination_kind == "directory":
        manifest = context.store.export_to_directory(
            destination,
            redact=True,
            include_blobs=False,
        )
    else:
        manifest = context.store.export_manifest(destination, redact=True)
    payload = _memory_envelope(context, "export_result", generations)
    payload.update(
        {
            "manifest_hash": manifest["manifest_hash"],
            "redacted": manifest["redacted"],
            "restorable": manifest["restorable"],
            "destination_kind": destination_kind,
            "counts": {
                "repositories": len(manifest["repositories"]),
                "candidates": len(manifest["candidates"]),
                "candidate_authority_receipts": len(
                    manifest["candidate_authority_receipts"]
                ),
                "records": len(manifest["records"]),
                "feedback": len(manifest["feedback"]),
                "knowledge": len(manifest["knowledge_entries"]),
                "events": len(manifest["events"]),
                "blobs": len(manifest["blobs"]),
            },
        }
    )
    lines = [
        "Redacted Memory export written",
        "Repository: %s" % context.repository_key,
        "Manifest: %s" % manifest["manifest_hash"],
        "Generation: %d" % generations.memory_generation,
        "Redacted: yes",
        "Restorable: no",
    ]
    _emit_memory_document(args, payload, lines)
    return 0


def _export_destination(
    context: _MemoryCommandContext,
    requested_path: str,
) -> tuple[Path, str, Path]:
    try:
        requested = Path(requested_path).expanduser()
        destination = requested.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise _MemoryCLIError(
            "export_destination_invalid",
            "the export destination could not be resolved safely",
            _MEMORY_EXIT_USAGE,
        ) from None
    destination_kind = (
        "directory"
        if destination.is_dir()
        or (not destination.exists() and destination.suffix == "")
        else "manifest"
    )
    manifest_path = (
        destination / "manifest.json"
        if destination_kind == "directory"
        else destination
    )
    protected_paths = (
        context.repository,
        Path(context.namespace.metadata.git_common_dir),
        context.memory_root,
        Path(context.namespace.namespace_path),
    )
    if any(_paths_overlap(manifest_path, protected) for protected in protected_paths):
        raise _MemoryCLIError(
            "export_destination_protected",
            "the export destination overlaps repository or Memory authority",
            _MEMORY_EXIT_USAGE,
        )
    if manifest_path.is_symlink() or (
        manifest_path.exists() and not manifest_path.is_file()
    ):
        raise _MemoryCLIError(
            "export_destination_invalid",
            "the export manifest destination is not a regular file",
            _MEMORY_EXIT_USAGE,
        )
    return destination, destination_kind, manifest_path


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        normalized_left = os.path.normcase(str(left.resolve(strict=False)))
        normalized_right = os.path.normcase(str(right.resolve(strict=False)))
        common = os.path.normcase(
            os.path.commonpath((normalized_left, normalized_right))
        )
    except (OSError, RuntimeError, ValueError):
        return True
    return common in {normalized_left, normalized_right}


def _memory_import(args: argparse.Namespace) -> int:
    context = _resolve_memory_context(args, write=False)
    generations = _context_generations(context)
    requested_path = Path(args.path).expanduser().resolve(strict=False)
    manifest_path = (
        requested_path / "manifest.json"
        if requested_path.is_dir()
        else requested_path
    )
    blob_source_root = (
        requested_path if requested_path.is_dir() else requested_path.parent
    )
    with tempfile.TemporaryDirectory(prefix="review-agent-memory-import-") as directory:
        validation_store = MemoryStore(Path(directory) / "namespace")
        prepared = validation_store.prepare_import_manifest(manifest_path)
        plan = prepared.plan

    identity_matches = plan.repository_keys == (context.repository_key,)
    asserted_identity = getattr(args, "identity_match", None)
    if asserted_identity is not None:
        asserted_key = (
            context.repository_key
            if asserted_identity == "current"
            else asserted_identity
        )
        if asserted_key != context.repository_key or not identity_matches:
            raise _MemoryCLIError(
                "identity_mismatch",
                "the import manifest does not match the explicitly selected repository identity",
                _MEMORY_EXIT_CONFLICT,
            )

    payload = _memory_envelope(context, "import_plan", generations)
    payload.update(_import_plan_payload(plan))
    payload.update(
        {
            "identity_matches_current_repository": identity_matches,
            "apply_requested": bool(args.apply),
        }
    )
    lines = _import_plan_human_lines(context, generations, plan, identity_matches)
    if not args.apply:
        _emit_memory_document(args, payload, lines)
        return 0

    payload["type"] = "import_preview"
    _emit_memory_document(args, payload, ["Import apply preview", *lines[1:]])
    if asserted_identity is None:
        raise _MemoryCLIError(
            "identity_confirmation_required",
            "import apply requires --identity-match for the current repository key",
            _MEMORY_EXIT_USAGE,
        )
    if not identity_matches:
        raise _MemoryCLIError(
            "identity_mismatch",
            "cross-identity import requires a state-safe explicit relink service",
            _MEMORY_EXIT_CONFLICT,
        )
    if not plan.restorable:
        raise _MemoryCLIError(
            "import_not_restorable",
            "a redacted Memory export cannot be applied",
            _MEMORY_EXIT_USAGE,
        )
    _confirm_memory_write(args, "import", context.repository_key)

    write_context = _resolve_memory_context(args, write=True)
    assert write_context.store is not None
    _require_same_memory_authority(context, write_context)
    applied = write_context.store.apply_prepared_import(
        prepared,
        blob_source_root=blob_source_root,
    )
    applied_generations = write_context.store.get_generations(
        write_context.repository_key
    )
    result_payload = _memory_envelope(
        write_context,
        "import_result",
        applied_generations,
    )
    result_payload.update(_import_plan_payload(applied))
    result_payload["identity_matches_current_repository"] = True
    result_lines = [
        "Memory import committed",
        "Repository: %s" % write_context.repository_key,
        "Generation: %d" % applied_generations.memory_generation,
        "Candidates: %d" % applied.candidate_count,
        "Records: %d" % applied.record_count,
        "Events: %d" % applied.event_count,
        "Applied: yes",
    ]
    _emit_memory_document(args, result_payload, result_lines)
    return 0


def _memory_gc(args: argparse.Namespace) -> int:
    context = _resolve_memory_context(args, write=False)
    generations = _context_generations(context)
    if context.store is None:
        scan = BlobGCResult(
            candidate_hashes=(),
            deleted_hashes=(),
            orphan_paths=(),
            deleted_orphan_paths=(),
            reclaimed_bytes=0,
            dry_run=True,
        )
    else:
        scan = context.store.gc_blobs(
            dry_run=True,
            grace_seconds=args.grace_seconds,
        )
    preview = _gc_payload(context, "gc_plan", generations, scan)
    lines = _gc_human_lines(context, generations, scan, heading="Memory GC dry run")
    if not args.apply:
        _emit_memory_document(args, preview, lines)
        return 0

    preview["type"] = "gc_preview"
    _emit_memory_document(args, preview, ["Memory GC apply preview", *lines[1:]])
    _confirm_memory_write(args, "gc", context.repository_key)
    if context.store is None:
        result = BlobGCResult(
            candidate_hashes=(),
            deleted_hashes=(),
            orphan_paths=(),
            deleted_orphan_paths=(),
            reclaimed_bytes=0,
            dry_run=False,
        )
        result_context = context
        result_generations = generations
    else:
        result_context = _resolve_memory_context(args, write=True, require_store=True)
        assert result_context.store is not None
        _require_same_memory_authority(context, result_context)
        result = result_context.store.apply_blob_gc(scan)
        result_generations = result_context.store.get_generations(
            result_context.repository_key
        )
    payload = _gc_payload(
        result_context,
        "gc_result",
        result_generations,
        result,
    )
    result_lines = _gc_human_lines(
        result_context,
        result_generations,
        result,
        heading="Memory GC committed",
    )
    _emit_memory_document(args, payload, result_lines)
    return 0


def _memory_relink(args: argparse.Namespace) -> int:
    actor, reason = _decision_fields(args)
    from_repository_key = args.from_repository_key
    try:
        requested_repository = Path(args.repo).resolve(strict=False)
        revision_resolver = RevisionResolver()
        repository_identity = revision_resolver.repository_identity(
            requested_repository
        )
        repository = Path(repository_identity.canonical_path)
    except Exception:
        raise _MemoryCLIError(
            "repository_invalid",
            "the requested path is not an accessible Git repository",
            _MEMORY_EXIT_USAGE,
        ) from None

    memory_root = MemoryRootResolver().resolve(
        getattr(args, "memory_root", None),
        create=False,
    )
    locator_plan = plan_repository_memory_namespace(
        repository_identity,
        memory_root,
        revision_resolver=revision_resolver,
    )
    locator_identity = locator_plan.metadata
    authority_namespace_path = repository_namespace_path(
        memory_root,
        from_repository_key,
    )
    authority_database_path = authority_namespace_path / "memory.sqlite3"
    if not authority_database_path.is_file():
        raise _MemoryCLIError(
            "authority_not_found",
            "the exact --from-key Memory authority does not exist",
            _MEMORY_EXIT_NOT_FOUND,
        )

    authority_probe = MemoryStore(authority_namespace_path, read_only=True)
    authority_snapshot = authority_probe.repository_authority_snapshot(
        from_repository_key
    )
    authority_identity = authority_snapshot.repository_identity
    request_id = getattr(args, "request_id", None) or stable_request_id(
        _MEMORY_CLI_SCHEMA,
        "relink",
        from_repository_key,
        locator_identity.repository_key,
        actor,
        reason,
    )
    registry = RepositoryRelinkRegistry(
        memory_root,
        revision_resolver=revision_resolver,
    )
    prepared = registry.prepare_relink(
        authority_identity,
        locator_identity,
        from_repository_key=from_repository_key,
        actor=actor,
        reason=reason,
        request_id=request_id,
        revision_resolver=revision_resolver,
    )
    preview_authority = authority_probe.repository_authority_snapshot(
        from_repository_key
    )
    if (
        preview_authority.repository_identity.to_payload()
        != prepared.authority_identity.to_payload()
        or preview_authority.state_token
        != prepared.old_authority_state_token
    ):
        raise RepositoryRelinkConflictError(
            "repository authority changed during relink preparation"
        )
    generations = preview_authority.generations
    preview = {
        "schema": _MEMORY_CLI_SCHEMA,
        "type": "relink_preview",
        "repository_key": prepared.authority_repository_key,
        "locator_repository_key": prepared.locator_repository_key,
        "authority_resolution_hash": prepared.authority_resolution_hash,
        "binding_id": prepared.binding_id,
        "generation": generations.memory_generation,
        "generations": generations.to_dict(),
        "registry_generation": prepared.registry_generation,
        "request_id": prepared.request_id,
        "prepared_hash": prepared.prepared_hash,
        "descriptor_hash": prepared.descriptor_hash,
        "actor": actor,
        "reason": reason,
        "new_namespace_empty": prepared.new_namespace_empty,
    }
    preview_lines = [
        "Repository Memory relink preview",
        "Locator: %s" % prepared.locator_repository_key,
        "Authority: %s" % prepared.authority_repository_key,
        "Binding: %s" % prepared.binding_id,
        "Registry generation: %d" % prepared.registry_generation,
        "Authority generation: %d" % generations.memory_generation,
        "Actor: %s" % _terminal_text(actor),
        "Reason: %s" % _terminal_text(reason),
        "New locator namespace empty: yes",
    ]
    _emit_memory_document(args, preview, preview_lines)
    _confirm_memory_write(args, "relink", prepared.binding_id)

    result = registry.apply_relink(
        prepared,
        revision_resolver=revision_resolver,
    )
    result_context = _resolve_memory_context(
        args,
        write=False,
        require_store=True,
    )
    assert result_context.store is not None
    result_generations = result_context.store.get_generations(
        result_context.repository_key
    )
    payload = _memory_envelope(
        result_context,
        "relink_result",
        result_generations,
    )
    payload.update(
        {
            "registry_generation": result.resolution.registry_generation,
            "request_id": result.request_id,
            "result_hash": result.result_hash,
            "event_id": result.event.event_id,
            "event_hash": result.event.event_hash,
            "outcome": result.outcome,
            "applied": result.applied,
        }
    )
    result_lines = [
        "Repository Memory relink committed",
        "Locator: %s" % result.resolution.locator_repository_key,
        "Authority: %s" % result.resolution.authority_repository_key,
        "Binding: %s" % result.resolution.binding_id,
        "Event: %s" % result.event.event_id,
        "Registry generation: %d" % result.resolution.registry_generation,
        "Outcome: %s" % result.outcome,
    ]
    _emit_memory_document(args, payload, result_lines)
    return 0


def _context_generations(context: _MemoryCommandContext) -> GenerationMetadata:
    if context.store is None:
        return GenerationMetadata(
            store_schema_version=CURRENT_MEMORY_STORE_SCHEMA_VERSION,
            memory_generation=0,
            feedback_generation=0,
            knowledge_generation=0,
        )
    return context.store.get_generations(context.repository_key)


def _memory_envelope(
    context: _MemoryCommandContext,
    document_type: str,
    generations: GenerationMetadata,
) -> dict[str, Any]:
    return {
        "schema": _MEMORY_CLI_SCHEMA,
        "type": document_type,
        "repository_key": context.repository_key,
        "locator_repository_key": context.locator_repository_key,
        "authority_resolution_hash": context.authority_resolution_hash,
        "binding_id": context.binding_id,
        "root_fingerprint": _memory_root_fingerprint(context.memory_root),
        "generation": generations.memory_generation,
        "generations": generations.to_dict(),
    }


def _memory_root_fingerprint(root_path: str | Path) -> str:
    canonical = Path(root_path).as_posix()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _enum_status_counts(items: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = item.status.value
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _require_repository_subject(subject_key: str, repository_key: str) -> None:
    if subject_key != repository_key:
        raise _MemoryCLIError(
            "subject_not_found",
            "the requested Memory subject is not in this repository namespace",
            _MEMORY_EXIT_NOT_FOUND,
        )


def _require_same_memory_authority(
    preview: _MemoryCommandContext,
    current: _MemoryCommandContext,
) -> None:
    if (
        preview.memory_root != current.memory_root
        or preview.locator_repository_key != current.locator_repository_key
        or preview.repository_key != current.repository_key
        or preview.binding_id != current.binding_id
        or preview.authority_resolution_hash != current.authority_resolution_hash
    ):
        raise _MemoryCLIError(
            "repository_changed",
            "the repository Memory authority changed after the preview",
            _MEMORY_EXIT_CONFLICT,
        )


def _record_human_lines(
    record: DurableMemoryRecord,
    generations: GenerationMetadata,
) -> list[str]:
    return [
        "Durable Memory record",
        "Memory: %s" % record.memory_id,
        "Candidate: %s" % record.candidate_id,
        "Generation: %d" % generations.memory_generation,
        "Status: %s" % record.status.value,
        "Kind: %s" % record.kind.value,
        "Statement: %s" % _terminal_text(record.statement),
        "Scope: %s" % _terminal_json(record.scope.to_dict()),
        "Sources: %s"
        % _terminal_json([item.to_dict() for item in record.source_refs]),
        "Valid from: %s" % record.valid_from_sha,
        "Validity: %s"
        % ", ".join(item.value for item in record.validity_policies),
        "Policy: %s"
        % _terminal_json(
            None if record.policy_effect is None else record.policy_effect.to_dict()
        ),
    ]


def _candidate_human_lines(
    candidate: MemoryCandidate,
    generations: GenerationMetadata,
) -> list[str]:
    return [
        "Memory candidate",
        "Candidate: %s" % candidate.candidate_id,
        "Generation: %d" % generations.memory_generation,
        "Status: %s" % candidate.status.value,
        "Kind: %s" % candidate.kind.value,
        "Statement: %s" % _terminal_text(candidate.statement),
        "Scope: %s" % _terminal_json(candidate.scope.to_dict()),
        "Sources: %s"
        % _terminal_json([item.to_dict() for item in candidate.source_refs]),
        "Valid from: %s" % candidate.valid_from_sha,
        "Validity: %s"
        % ", ".join(item.value for item in candidate.validity_policies),
        "Policy: %s"
        % _terminal_json(
            None
            if candidate.policy_effect is None
            else candidate.policy_effect.to_dict()
        ),
    ]


def _candidate_runtime_authority(
    context: _MemoryCommandContext,
    candidate: MemoryCandidate,
    *,
    target_head: str | None = None,
) -> _CandidateRuntimeAuthority:
    if context.store is None:
        raise _MemoryCLIError(
            "store_not_found",
            "candidate authority requires an existing Memory Store",
            _MEMORY_EXIT_NOT_FOUND,
        )
    if candidate.repository_key != context.repository_key:
        raise _MemoryCLIError(
            "candidate_repository_mismatch",
            "the candidate belongs to a different repository authority",
            _MEMORY_EXIT_USAGE,
        )
    current_target = (
        RevisionResolver().resolve_commit(context.repository, "HEAD")
        if target_head is None
        else target_head
    )
    receipt = context.store.select_candidate_authority_receipt(
        candidate.candidate_id,
        authority_resolution_hash=context.authority_resolution_hash,
    )
    if (
        receipt.locator_repository_key != context.locator_repository_key
        or receipt.authority_repository_key != context.repository_key
        or receipt.binding_id != context.binding_id
    ):
        raise _MemoryCLIError(
            "candidate_authority_mismatch",
            "the stored candidate authority does not match the live repository binding",
            _MEMORY_EXIT_CONFLICT,
        )
    provenance = TrustedCandidateProvenance(
        origin=receipt.origin,
        review_id=receipt.review_id,
        target_head_sha=current_target,
        locator_repository_key=context.locator_repository_key,
        authority_repository_key=context.repository_key,
        authority_resolution_hash=context.authority_resolution_hash,
        binding_id=context.binding_id,
        allowed_source_refs=receipt.authorized_source_refs,
    )
    validator = SourceValidator(
        context.repository,
        human_declarations=receipt.human_declarations,
    )
    restoration = validator.restore_candidate_authority(
        receipt,
        replace(candidate, status=CandidateStatus.PROPOSED),
        current_provenance=provenance,
        current_target_head_sha=current_target,
    )
    return _CandidateRuntimeAuthority(
        provenance=restoration.provenance,
        validator=validator,
    )


def _decision_fields(args: argparse.Namespace) -> tuple[str, str]:
    return (
        _required_cli_text(args.actor, "actor"),
        _required_cli_text(args.reason, "reason"),
    )


def _required_cli_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _MemoryCLIError(
            "decision_field_required",
            "a non-empty %s is required" % label,
            _MEMORY_EXIT_USAGE,
        )
    return value.strip()


def _decision_request_id(
    args: argparse.Namespace,
    action: str,
    subject_id: str,
    actor: str,
    reason_code: str,
    reason: str,
) -> str:
    supplied = getattr(args, "request_id", None)
    if supplied is not None:
        return supplied
    return stable_request_id(
        _MEMORY_CLI_SCHEMA,
        action,
        subject_id,
        actor,
        reason_code,
        reason,
    )


def _candidate_decision_preview(
    context: _MemoryCommandContext,
    *,
    action: str,
    candidate: MemoryCandidate,
    generations: GenerationMetadata,
    validation: SourceValidationReport,
    policy_before: Mapping[str, Any] | None,
) -> dict[str, Any]:
    policy_after = (
        None
        if candidate.policy_effect is None
        else candidate.policy_effect.to_dict()
    )
    payload = _memory_envelope(
        context,
        "%s_preview" % action,
        generations,
    )
    payload.update(
        {
            "candidate_id": candidate.candidate_id,
            "candidate_status": candidate.status.value,
            "statement": candidate.statement,
            "scope": candidate.scope.to_dict(),
            "sources": [item.to_dict() for item in candidate.source_refs],
            "validity": {
                "valid_from_sha": candidate.valid_from_sha,
                "policies": [item.value for item in candidate.validity_policies],
            },
            "policy_diff": {
                "before": policy_before,
                "after": policy_after,
                "changed": policy_before != policy_after,
            },
            "source_validation": {
                "valid": validation.valid,
                "report_hash": validation.report_hash,
                "issue_codes": [item.code.value for item in validation.issues],
            },
        }
    )
    return payload


def _candidate_preview_human_lines(preview: Mapping[str, Any]) -> list[str]:
    lines = [
        "%s preview" % str(preview["type"]).replace("_", " ").title(),
        "Repository: %s" % preview["repository_key"],
        "Generation: %s" % preview["generation"],
    ]
    if "predecessor_memory_id" in preview:
        lines.append("Predecessor: %s" % preview["predecessor_memory_id"])
    lines.extend(
        [
            "Candidate: %s" % preview["candidate_id"],
            "Statement: %s" % _terminal_text(preview["statement"]),
            "Scope: %s" % _terminal_json(preview["scope"]),
            "Sources: %s" % _terminal_json(preview["sources"]),
            "Validity: %s" % _terminal_json(preview["validity"]),
            "Policy diff: %s" % _terminal_json(preview["policy_diff"]),
            "Source validation: %s"
            % ("valid" if preview["source_validation"]["valid"] else "invalid"),
        ]
    )
    return lines


def _status_decision_preview(
    context: _MemoryCommandContext,
    *,
    action: str,
    subject_type: str,
    subject_id: str,
    current_status: str,
    requested_status: str,
    generations: GenerationMetadata,
    reason_code: str,
) -> dict[str, Any]:
    payload = _memory_envelope(context, "%s_preview" % action, generations)
    payload.update(
        {
            "subject_type": subject_type,
            "subject_id": subject_id,
            "current_status": current_status,
            "requested_status": requested_status,
            "reason_code": reason_code,
        }
    )
    return payload


def _status_preview_human_lines(preview: Mapping[str, Any]) -> list[str]:
    return [
        "%s" % str(preview["type"]).replace("_", " ").title(),
        "Repository: %s" % preview["repository_key"],
        "Generation: %s" % preview["generation"],
        "Subject: %s" % preview["subject_id"],
        "Status: %s -> %s"
        % (preview["current_status"], preview["requested_status"]),
        "Reason code: %s" % preview["reason_code"],
    ]


def _confirm_memory_write(
    args: argparse.Namespace,
    action: str,
    subject_id: str,
) -> None:
    if bool(getattr(args, "yes", False)):
        return
    if bool(getattr(args, "non_interactive", False)) or not sys.stdin.isatty():
        raise _MemoryCLIError(
            "confirmation_required",
            "non-interactive memory writes require --yes",
            _MEMORY_EXIT_USAGE,
        )
    try:
        response = input(
            "Confirm memory %s for %s? [y/N] " % (action, subject_id)
        )
    except (EOFError, OSError):
        raise _MemoryCLIError(
            "confirmation_required",
            "memory write confirmation was unavailable",
            _MEMORY_EXIT_USAGE,
        ) from None
    if response.strip().casefold() not in {"y", "yes"}:
        raise _MemoryCLIError(
            "cancelled",
            "memory write was not confirmed",
            _MEMORY_EXIT_USAGE,
        )


def _approval_result_payload(
    context: _MemoryCommandContext,
    action: str,
    result: ApprovalResult,
) -> dict[str, Any]:
    payload = _memory_envelope(
        context,
        "%s_result" % action,
        result.write_result.generations,
    )
    payload.update(
        {
            "candidate_id": result.record.candidate_id,
            "memory_id": result.record.memory_id,
            "source_bundle_id": result.bundle.bundle_hash,
            "event_id": result.write_result.event_id,
            "applied": result.write_result.applied,
            "replayed": result.write_result.replayed,
            "status": result.record.status.value,
        }
    )
    return payload


def _write_result_payload(
    context: _MemoryCommandContext,
    document_type: str,
    result: WriteResult,
    **identifiers: str,
) -> dict[str, Any]:
    payload = _memory_envelope(context, document_type, result.generations)
    payload.update(identifiers)
    payload.update(
        {
            "subject_id": result.subject_id,
            "event_id": result.event_id,
            "operation": result.operation,
            "applied": result.applied,
            "replayed": result.replayed,
        }
    )
    return payload


def _write_result_human_lines(
    heading: str,
    context: _MemoryCommandContext,
    result: WriteResult,
) -> list[str]:
    return [
        heading,
        "Repository: %s" % context.repository_key,
        "Subject: %s" % result.subject_id,
        "Event: %s" % result.event_id,
        "Generation: %d" % result.generations.memory_generation,
        "Applied: %s" % ("yes" if result.applied else "no"),
        "Replayed: %s" % ("yes" if result.replayed else "no"),
    ]


def _import_plan_payload(plan: ImportPlan) -> dict[str, Any]:
    return {
        "repository_keys": list(plan.repository_keys),
        "counts": {
            "candidates": plan.candidate_count,
            "candidate_authority_receipts": plan.authority_receipt_count,
            "records": plan.record_count,
            "feedback": plan.feedback_count,
            "knowledge": plan.knowledge_count,
            "source_bundles": plan.source_bundle_count,
            "events": plan.event_count,
            "blobs": plan.blob_count,
            "outbox_receipts": plan.outbox_receipt_count,
        },
        "redacted": plan.redacted,
        "restorable": plan.restorable,
        "applied": plan.applied,
    }


def _import_plan_human_lines(
    context: _MemoryCommandContext,
    generations: GenerationMetadata,
    plan: ImportPlan,
    identity_matches: bool,
) -> list[str]:
    return [
        "Memory import dry run",
        "Repository: %s" % context.repository_key,
        "Generation: %d" % generations.memory_generation,
        "Manifest repositories: %s" % ", ".join(plan.repository_keys),
        "Identity match: %s" % ("yes" if identity_matches else "no"),
        "Candidates: %d" % plan.candidate_count,
        "Candidate authority receipts: %d" % plan.authority_receipt_count,
        "Records: %d" % plan.record_count,
        "Events: %d" % plan.event_count,
        "Blobs: %d" % plan.blob_count,
        "Redacted: %s" % ("yes" if plan.redacted else "no"),
        "Restorable: %s" % ("yes" if plan.restorable else "no"),
        "Applied: no",
    ]


def _gc_payload(
    context: _MemoryCommandContext,
    document_type: str,
    generations: GenerationMetadata,
    result: BlobGCResult,
) -> dict[str, Any]:
    payload = _memory_envelope(context, document_type, generations)
    payload.update(
        {
            "dry_run": result.dry_run,
            "candidate_blob_ids": list(result.candidate_hashes),
            "deleted_blob_ids": list(result.deleted_hashes),
            "orphan_path_count": len(result.orphan_paths),
            "deleted_orphan_count": len(result.deleted_orphan_paths),
            "reclaimed_bytes": result.reclaimed_bytes,
        }
    )
    return payload


def _gc_human_lines(
    context: _MemoryCommandContext,
    generations: GenerationMetadata,
    result: BlobGCResult,
    *,
    heading: str,
) -> list[str]:
    return [
        heading,
        "Repository: %s" % context.repository_key,
        "Generation: %d" % generations.memory_generation,
        "Candidate blobs: %d" % len(result.candidate_hashes),
        "Candidate IDs: %s"
        % (", ".join(result.candidate_hashes) if result.candidate_hashes else "none"),
        "Deleted blobs: %d" % len(result.deleted_hashes),
        "Deleted IDs: %s"
        % (", ".join(result.deleted_hashes) if result.deleted_hashes else "none"),
        "Orphan paths: %d" % len(result.orphan_paths),
        "Reclaimed bytes: %d" % result.reclaimed_bytes,
        "Dry run: %s" % ("yes" if result.dry_run else "no"),
    ]


def _emit_memory_document(
    args: argparse.Namespace,
    payload: Mapping[str, Any],
    human_lines: Sequence[str],
) -> None:
    if getattr(args, "output_format", "human") == "json":
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return
    for line in human_lines:
        print(line)


def _emit_memory_error(
    args: argparse.Namespace,
    *,
    code: str,
    message: str,
    exit_code: int,
) -> int:
    if getattr(args, "output_format", "human") == "json":
        print(
            json.dumps(
                {
                    "schema": _MEMORY_CLI_SCHEMA,
                    "type": "error",
                    "error": {"code": code, "message": message},
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
    else:
        print("Memory error [%s]: %s" % (code, message), file=sys.stderr)
    return exit_code


def _emit_memory_store_error(
    args: argparse.Namespace,
    error: MemoryStoreError,
) -> int:
    if error.code is MemoryStoreErrorCode.NOT_FOUND:
        exit_code = _MEMORY_EXIT_NOT_FOUND
    elif error.code in {MemoryStoreErrorCode.CONFLICT, MemoryStoreErrorCode.BUSY}:
        exit_code = _MEMORY_EXIT_CONFLICT
    elif error.code is MemoryStoreErrorCode.VALIDATION:
        exit_code = _MEMORY_EXIT_USAGE
    else:
        exit_code = _MEMORY_EXIT_OPERATIONAL
    messages = {
        MemoryStoreErrorCode.UNAVAILABLE: "the Memory Store is unavailable",
        MemoryStoreErrorCode.UNSUPPORTED_SCHEMA: "the Memory Store schema is unsupported",
        MemoryStoreErrorCode.CORRUPTION: "Memory Store integrity validation failed",
        MemoryStoreErrorCode.CONFLICT: "the Memory Store changed or conflicts with this request",
        MemoryStoreErrorCode.BUSY: "the Memory Store is busy",
        MemoryStoreErrorCode.VALIDATION: "the Memory Store rejected invalid input",
        MemoryStoreErrorCode.NOT_FOUND: "the requested Memory subject was not found",
        MemoryStoreErrorCode.READ_ONLY: "the Memory Store is read-only",
        MemoryStoreErrorCode.MIGRATION: "the Memory Store migration failed safely",
    }
    return _emit_memory_error(
        args,
        code="store_%s" % error.code.value,
        message=messages[error.code],
        exit_code=exit_code,
    )


def _terminal_json(value: Any) -> str:
    return _terminal_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
    )


def _terminal_text(value: Any, *, limit: int | None = None) -> str:
    raw = str(value)
    safe = "".join(
        character if character.isprintable() else " " for character in raw
    )
    safe = " ".join(safe.split())
    if limit is not None and len(safe) > limit:
        return safe[: max(0, limit - 1)] + "…"
    return safe


def _print_memory_execution_summary(
    config: MemoryExecutionConfig | None,
    *,
    prefix: str = "",
) -> None:
    if config is None:
        return
    print(f"{prefix}Memory mode: {config.mode.value}")
    print(
        f"{prefix}Memory root fingerprint: "
        f"{_memory_root_fingerprint(config.root_path)}"
    )
    print(f"{prefix}Memory required: {'yes' if config.required else 'no'}")


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
        memory_config = _resolve_memory_execution_config(args)
    except (MemoryIdentityError, ValueError) as error:
        print(f"Memory session configuration error: {error}")
        return 2

    try:
        execution_config = ReviewExecutionConfig(
            reviewer_provider=args.reviewer_provider,
            reviewer_model=args.reviewer_model,
            reviewer_base_url=args.reviewer_base_url,
            reviewer_api_key_env=args.reviewer_api_key_env,
            reviewer_mode=args.reviewer_mode,
            reviewer_loop=args.reviewer_loop,
            non_interactive=args.non_interactive,
            memory=memory_config,
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
            memory_curator=_resolve_model_stage_config(
                args,
                "memory-curator",
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

    _print_memory_execution_summary(execution_config.memory)
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
            manifest = session_store.load()
        except Exception as error:
            print(f"Review run has an invalid session.json: {error}", file=sys.stderr)
            return 2
        upgrade_execution = None
        if args.upgrade_to_v5:
            try:
                upgrade_execution = replace(
                    manifest.execution,
                    memory=_resolve_memory_execution_config(args),
                )
                upgrade_execution = replace(
                    upgrade_execution,
                    memory_curator=_resolve_model_stage_config(
                        args,
                        "memory-curator",
                        upgrade_execution,
                    ),
                )
            except (MemoryIdentityError, ValueError) as error:
                print(
                    f"Resume v5 upgrade configuration error: {error}",
                    file=sys.stderr,
                )
                return 2
        try:
            result = ReviewSessionResumer(
                repository=repo,
                checkpoint_store=store,
                session_store=session_store,
                intent_clarifier=ConsoleIntentClarifier(),
                upgrade_execution=upgrade_execution,
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
    _print_memory_execution_summary(session.execution.memory, prefix="  ")
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

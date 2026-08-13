from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Mapping, Sequence

from review_agent.execution_profile import AgentExecutionProfile
from review_agent.model_adapter_factory import ModelAdapterConfig
from review_agent.product_runtime import (
    IntentTrustPolicyV6,
    ProductReviewInputV6,
    ProductReviewOutcomeV6,
    ProductRuntimeConfigV6,
    ProductRuntimeInfrastructureError,
    ProductRuntimeIntegrityError,
    ProductRuntimeUsageError,
    resume_product_review_v6,
    start_product_review_v6,
)
from review_agent.review_protocol import (
    ConversationMessage,
    ConversationSpeaker,
    ReviewRequest as ReviewRequestV6,
)

_CI_EVIDENCE_BUNDLE_SCHEMA_VERSION = "review_agent_ci_evidence_bundle_v1"
_CI_EVIDENCE_CONTROL_RELATIVE_PATH = PurePosixPath(".review-agent/eval-input")
_CI_EVIDENCE_BUNDLE_PREFIX = "existing-ci-evidence."
_CI_EVIDENCE_BUNDLE_SUFFIX = ".v1.json"
_MAX_CI_EVIDENCE_BUNDLE_BYTES = 2 * 1024 * 1024
_MAX_CI_EVIDENCE_ENTRIES = 256
_MAX_CI_SOURCE_ID_CHARS = 512
_MAX_CI_EVIDENCE_TEXT_CHARS = 32768
_MAX_CI_EVIDENCE_PATH_CHARS = 4096
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["memory"]:
        compatibility = importlib.import_module(
            "review_agent.legacy_memory_command"
        )
        return int(compatibility.main(arguments))
    parser = _build_parser()
    args = parser.parse_args(arguments)
    if args.command == "review":
        return _run_review(args)
    if args.command == "resume":
        return _run_resume(args)
    parser.print_help()
    return 2

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="review-agent", allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command")

    review = subparsers.add_parser("review", allow_abbrev=False)
    review.add_argument("--repo", default=".")
    review.add_argument("--base", required=True)
    review.add_argument("--head", required=True)
    review.add_argument("--external-review-id")
    review.add_argument("--workspace-root")
    review.add_argument(
        "--format",
        dest="output_format",
        choices=["json", "markdown"],
        default="json",
    )
    review.add_argument("--intent")
    review.add_argument("--focus")
    review.add_argument("--title")
    review.add_argument("--description")
    review.add_argument("--requirement", action="append", default=[])
    review.add_argument("--project-rule", action="append", default=[])
    ci_evidence = review.add_mutually_exclusive_group()
    ci_evidence.add_argument("--ci-evidence", action="append", default=[])
    ci_evidence.add_argument("--ci-evidence-file")
    review.add_argument("--non-interactive", action="store_true")
    review.add_argument(
        "--evaluation-trust-model-intent",
        action="store_true",
        help=(
            "Evaluation-only policy: promote one reliable Intent Agent goal "
            "to explicit when no human can confirm it"
        ),
    )
    review.add_argument(
        "--reviewer-provider",
        choices=["none", "fake", "openai-compatible"],
        default="none",
    )
    review.add_argument("--reviewer-model")
    review.add_argument("--reviewer-base-url")
    review.add_argument("--reviewer-api-key-env", default="REVIEW_AGENT_API_KEY")
    _add_risk_model_arguments_v6(review)

    resume = subparsers.add_parser(
        "resume",
        help="Resume a PRWorkspace Session v6 or inspect a legacy review",
        allow_abbrev=False,
    )
    resume.add_argument("session_id", help="SESSION- locator or legacy review id")
    resume.add_argument("--repo", default=".", help="Repository path")
    resume.add_argument("--workspace-root")
    resume.add_argument("--pr-id")
    resume.add_argument("--snapshot-id")
    resume.add_argument(
        "--format",
        dest="output_format",
        choices=["json", "markdown"],
        default="json",
    )
    resume.add_argument(
        "--reviewer-provider",
        choices=["none", "fake", "openai-compatible"],
        default="none",
    )
    resume.add_argument("--reviewer-model")
    resume.add_argument("--reviewer-base-url")
    resume.add_argument("--reviewer-api-key-env", default="REVIEW_AGENT_API_KEY")
    _add_risk_model_arguments_v6(resume)
    return parser

def _add_risk_model_arguments_v6(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--risk-assessor-mode",
        choices=["local", "model"],
        default="local",
    )
    parser.add_argument(
        "--risk-assessor-provider",
        choices=["inherit", "none", "fake", "openai-compatible"],
        default="inherit",
    )
    parser.add_argument("--risk-assessor-model")
    parser.add_argument("--risk-assessor-base-url")
    parser.add_argument("--risk-assessor-api-key-env")

def review_execution_profile_from_arguments(
    review_arguments: Sequence[str],
    *,
    memory_mode: str | None = None,
    memory_root: Path | None = None,
) -> AgentExecutionProfile:
    del memory_mode, memory_root
    parsed = _build_parser().parse_args(
        [
            "review",
            "--base=" + ("0" * 40),
            "--head=" + ("1" * 40),
            "--external-review-id=execution-profile",
            "--workspace-root=.",
            *review_arguments,
        ]
    )
    config = _product_runtime_config_v6(parsed)
    return AgentExecutionProfile.from_product_configuration(
        reviewer=config.reviewer,
        risk=config.risk,
    )

def _strict_cli_json_object(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result

def _ci_evidence_regular_file(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and not (getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)
        and metadata.st_nlink == 1
    )

def _ci_evidence_directory(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and not (getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)
    )

def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)

def _ci_evidence_bundle_path(repository: Path, supplied: str) -> Path:
    if (
        type(supplied) is not str
        or not supplied
        or len(supplied) > _MAX_CI_EVIDENCE_PATH_CHARS
        or "\x00" in supplied
    ):
        raise ValueError("CI evidence file path is invalid")
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = repository / candidate
    candidate = Path(os.path.abspath(candidate))
    expected_parent = repository.joinpath(*_CI_EVIDENCE_CONTROL_RELATIVE_PATH.parts)
    if os.path.normcase(str(candidate.parent)) != os.path.normcase(
        str(expected_parent)
    ):
        raise ValueError("CI evidence file must be inside the eval-input control root")
    name = candidate.name
    if not (
        name.startswith(_CI_EVIDENCE_BUNDLE_PREFIX)
        and name.endswith(_CI_EVIDENCE_BUNDLE_SUFFIX)
    ):
        raise ValueError("CI evidence bundle filename is not digest-bound")
    digest = name[
        len(_CI_EVIDENCE_BUNDLE_PREFIX) : -len(_CI_EVIDENCE_BUNDLE_SUFFIX)
    ]
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("CI evidence bundle filename digest is invalid")
    return candidate

def _read_ci_evidence_bundle_file(repository: Path, supplied: str) -> bytes:
    candidate = _ci_evidence_bundle_path(repository, supplied)
    control_root = candidate.parent
    review_root = control_root.parent
    directory_metadata: list[tuple[Path, os.stat_result]] = []
    try:
        for path in (review_root, control_root):
            metadata = os.lstat(path)
            if not _ci_evidence_directory(metadata):
                raise ValueError("CI evidence path contains an unsafe directory")
            directory_metadata.append((path, metadata))

        before = os.lstat(candidate)
        if (
            not _ci_evidence_regular_file(before)
            or before.st_size > _MAX_CI_EVIDENCE_BUNDLE_BYTES
        ):
            raise ValueError("CI evidence bundle is not a bounded regular file")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(candidate, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not _ci_evidence_regular_file(opened)
                or not _same_file_identity(before, opened)
                or opened.st_size > _MAX_CI_EVIDENCE_BUNDLE_BYTES
            ):
                raise ValueError("CI evidence bundle changed before it was opened")
            chunks: list[bytes] = []
            remaining = _MAX_CI_EVIDENCE_BUNDLE_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = os.lstat(candidate)
        if (
            len(data) > _MAX_CI_EVIDENCE_BUNDLE_BYTES
            or len(data) != after.st_size
            or not _ci_evidence_regular_file(after)
            or not _same_file_identity(opened, after)
            or not _same_file_identity(after, path_after)
        ):
            raise ValueError("CI evidence bundle changed while it was read")
        for path, metadata in directory_metadata:
            current = os.lstat(path)
            if (
                not _ci_evidence_directory(current)
                or not _same_file_identity(metadata, current)
            ):
                raise ValueError("CI evidence directory changed while it was read")
        return data
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("CI evidence bundle could not be read safely") from error

def _canonical_ci_evidence_entry(entry: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(entry),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )

def _load_ci_evidence_bundle(repository: Path, supplied: str) -> tuple[str, ...]:
    candidate = _ci_evidence_bundle_path(repository, supplied)
    data = _read_ci_evidence_bundle_file(repository, supplied)
    expected_digest = candidate.name[
        len(_CI_EVIDENCE_BUNDLE_PREFIX) : -len(_CI_EVIDENCE_BUNDLE_SUFFIX)
    ]
    if hashlib.sha256(data).hexdigest() != expected_digest:
        raise ValueError("CI evidence bundle does not match its filename digest")
    try:
        payload = json.loads(
            data.decode("utf-8", "strict"),
            object_pairs_hook=_strict_cli_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("CI evidence bundle contains a non-finite number")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("CI evidence bundle is invalid JSON") from error
    if type(payload) is not dict or set(payload) != {"schema_version", "entries"}:
        raise ValueError("CI evidence bundle has an invalid root schema")
    if payload["schema_version"] != _CI_EVIDENCE_BUNDLE_SCHEMA_VERSION:
        raise ValueError("CI evidence bundle has an unsupported schema version")
    entries = payload["entries"]
    if type(entries) is not list or not 1 <= len(entries) <= _MAX_CI_EVIDENCE_ENTRIES:
        raise ValueError("CI evidence bundle has an invalid entry count")
    encoded: list[str] = []
    source_ids: set[str] = set()
    for entry in entries:
        if type(entry) is not dict or set(entry) != {
            "source_id",
            "text",
            "content_hash",
        }:
            raise ValueError("CI evidence bundle entry has an invalid schema")
        source_id = entry["source_id"]
        text = entry["text"]
        content_hash = entry["content_hash"]
        if (
            type(source_id) is not str
            or not source_id.strip()
            or len(source_id) > _MAX_CI_SOURCE_ID_CHARS
            or source_id != source_id.strip()
            or any(
                character.isspace()
                or ord(character) < 32
                or ord(character) == 127
                for character in source_id
            )
        ):
            raise ValueError("CI evidence source_id is invalid")
        try:
            source_id.encode("utf-8", "strict")
        except UnicodeEncodeError as error:
            raise ValueError("CI evidence source_id contains invalid Unicode") from error
        if source_id in source_ids:
            raise ValueError("CI evidence source_id is duplicated")
        if type(text) is not str or len(text) > _MAX_CI_EVIDENCE_TEXT_CHARS:
            raise ValueError("CI evidence text is invalid")
        try:
            encoded_text = text.encode("utf-8", "strict")
        except UnicodeEncodeError as error:
            raise ValueError("CI evidence text contains invalid Unicode") from error
        if (
            type(content_hash) is not str
            or len(content_hash) != 64
            or any(character not in "0123456789abcdef" for character in content_hash)
            or hashlib.sha256(encoded_text).hexdigest() != content_hash
        ):
            raise ValueError("CI evidence content_hash is invalid")
        source_ids.add(source_id)
        encoded.append(_canonical_ci_evidence_entry(entry))

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if data != canonical:
        raise ValueError("CI evidence bundle is not canonical JSON")
    return tuple(encoded)

def _product_workspace_root(args: argparse.Namespace, repository: Path) -> Path:
    supplied = getattr(args, "workspace_root", None)
    if supplied is None:
        supplied = os.environ.get("REVIEW_AGENT_WORKSPACE_ROOT")
    if supplied is None:
        return repository / ".review-agent" / "workspaces-v6"
    if type(supplied) is not str or not supplied.strip() or "\x00" in supplied:
        raise ProductRuntimeUsageError("workspace root is invalid")
    return Path(supplied).expanduser().resolve()

def _product_review_input_v6(args: argparse.Namespace) -> ProductReviewInputV6:
    repository = Path(args.repo).resolve()
    if args.ci_evidence and args.ci_evidence_file is not None:
        raise ProductRuntimeUsageError(
            "direct and file CI evidence are mutually exclusive"
        )
    ci_evidence = tuple(args.ci_evidence)
    if args.ci_evidence_file is not None:
        try:
            ci_evidence = _load_ci_evidence_bundle(
                repository,
                args.ci_evidence_file,
            )
        except ValueError as error:
            raise ProductRuntimeUsageError(str(error)) from error

    lines = [
        (
            "Review the immutable code changes between the requested base and "
            "head revisions."
        )
    ]
    for label, value in (
        ("Title", args.title),
        ("Description", args.description),
        ("Declared intent", args.intent),
        ("Review focus", args.focus),
    ):
        if value is not None:
            lines.append(f"{label}: {value}")
    if args.requirement:
        lines.append("Linked requirements:")
        lines.extend(f"- {value}" for value in args.requirement)
    if args.project_rule:
        lines.append("User review rules:")
        lines.extend(f"- {value}" for value in args.project_rule)
    if ci_evidence:
        lines.append("Existing CI evidence:")
        lines.extend(f"- {value}" for value in ci_evidence)
    request = ReviewRequestV6(
        conversation=(
            ConversationMessage(
                speaker=ConversationSpeaker.USER,
                content="\n".join(lines),
            ),
        )
    )
    return ProductReviewInputV6(
        request=request,
        declared_goal=args.intent,
        title=args.title,
        description=args.description,
        intent_trust_policy=(
            IntentTrustPolicyV6.EVALUATION_TRUST_MODEL
            if args.evaluation_trust_model_intent
            else IntentTrustPolicyV6.NORMAL
        ),
    )

def _product_runtime_config_v6(
    args: argparse.Namespace,
) -> ProductRuntimeConfigV6:
    reviewer = ModelAdapterConfig(
        provider_name=args.reviewer_provider,
        model=args.reviewer_model,
        base_url=args.reviewer_base_url,
        api_key_env=args.reviewer_api_key_env,
        stage_label="reviewer",
    )
    risk = None
    if getattr(args, "risk_assessor_mode", "local") == "model":
        requested_provider = args.risk_assessor_provider
        provider = (
            reviewer.provider_name
            if requested_provider == "inherit"
            else requested_provider
        )
        risk = ModelAdapterConfig(
            provider_name=provider,
            model=args.risk_assessor_model or reviewer.model,
            base_url=args.risk_assessor_base_url or reviewer.base_url,
            api_key_env=(
                args.risk_assessor_api_key_env or reviewer.api_key_env
            ),
            stage_label="risk-assessor",
        )
    return ProductRuntimeConfigV6(reviewer=reviewer, risk=risk)

def _emit_product_outcome(
    outcome: ProductReviewOutcomeV6,
    output_format: str,
) -> None:
    if output_format == "json":
        rendered = outcome.review_result_json
    elif output_format == "markdown":
        rendered = outcome.review_markdown
    else:
        raise ProductRuntimeUsageError("output format is unsupported")
    sys.stdout.write(rendered)
    if not rendered.endswith("\n"):
        sys.stdout.write("\n")
    _emit_product_locator(outcome.locator())

def _emit_product_locator(locator: Mapping[str, str]) -> None:
    print(
        "Review locator: "
        + json.dumps(
            dict(locator),
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
    )

def _inspect_legacy_review(repository: Path, review_id: str) -> int:
    if (
        type(review_id) is not str
        or re.fullmatch(r"review-[A-Za-z0-9._-]{1,128}", review_id) is None
    ):
        print(
            "Resume configuration error: expected a Session v6 locator or a "
            "canonical legacy review id",
            file=sys.stderr,
        )
        return 2
    run_dir = repository / ".review-agent" / "runs" / review_id
    if not run_dir.is_dir():
        print(f"Legacy review not found: {run_dir}", file=sys.stderr)
        return 2
    session_path = run_dir / "session.json"
    if session_path.is_file():
        try:
            diagnostic = importlib.import_module(
                "review_agent.resume"
            ).diagnose_legacy_session(run_dir)
        except Exception as error:
            print(
                f"Legacy review has an invalid Session: {error}",
                file=sys.stderr,
            )
            return 2
        invalid = [
            artifact.name
            for artifact in diagnostic.artifacts
            if not artifact.valid
        ]
        print(
            "Legacy review is read-only and cannot be resumed by Session v6. "
            f"schema=v{diagnostic.schema_version} status={diagnostic.status} "
            f"phase={diagnostic.current_phase} invalid_artifacts={invalid}",
            file=sys.stderr,
        )
    else:
        print(
            "Legacy review has no Session manifest and is inspect-only under "
            f"Session v6: {run_dir}",
            file=sys.stderr,
        )
    print(
        "Start a new v6 review with --external-review-id; no legacy Phase was run.",
        file=sys.stderr,
    )
    return 2

def _run_review_v6(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    if not isinstance(args.external_review_id, str) or not args.external_review_id.strip():
        print("Review configuration error: --external-review-id is required", file=sys.stderr)
        return 2
    try:
        review_input = _product_review_input_v6(args)
        config = _product_runtime_config_v6(args)
        outcome = start_product_review_v6(
            repository=repo,
            workspace_root=_product_workspace_root(args, repo),
            base_revision=args.base,
            head_revision=args.head,
            external_review_id=args.external_review_id,
            review_input=review_input,
            config=config,
        )
    except (ProductRuntimeUsageError, ProductRuntimeIntegrityError) as error:
        print(f"Review configuration error: {error}", file=sys.stderr)
        return 2
    except ProductRuntimeInfrastructureError as error:
        print(f"Review failed: {error}", file=sys.stderr)
        if error.locator is not None:
            _emit_product_locator(error.locator)
        return 1
    except Exception as error:
        print(
            f"Review failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    _emit_product_outcome(outcome, args.output_format)
    return 0

def _run_review(args: argparse.Namespace) -> int:
    return _run_review_v6(args)

def _run_resume_v6(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    if not args.session_id.startswith("SESSION-"):
        return _inspect_legacy_review(repo, args.session_id)
    if not isinstance(args.pr_id, str) or not args.pr_id:
        print("Resume configuration error: --pr-id is required", file=sys.stderr)
        return 2
    if not isinstance(args.snapshot_id, str) or not args.snapshot_id:
        print(
            "Resume configuration error: --snapshot-id is required",
            file=sys.stderr,
        )
        return 2
    try:
        outcome = resume_product_review_v6(
            repository=repo,
            workspace_root=_product_workspace_root(args, repo),
            pr_id=args.pr_id,
            snapshot_id=args.snapshot_id,
            session_id=args.session_id,
            config=_product_runtime_config_v6(args),
        )
    except (ProductRuntimeUsageError, ProductRuntimeIntegrityError) as error:
        print(f"Resume configuration error: {error}", file=sys.stderr)
        return 2
    except ProductRuntimeInfrastructureError as error:
        print(f"Resume failed: {error}", file=sys.stderr)
        if error.locator is not None:
            _emit_product_locator(error.locator)
        return 1
    except Exception as error:
        print(
            f"Resume failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    _emit_product_outcome(outcome, args.output_format)
    return 0

def _run_resume(args: argparse.Namespace) -> int:
    return _run_resume_v6(args)

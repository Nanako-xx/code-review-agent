from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from review_agent_eval.models import (  # noqa: E402
    ClarificationScript,
    EvalCase,
    EvalInput,
    IntentTruth,
    RepositoryReviewTarget,
    ReviewTruth,
    canonical_json,
    canonical_sha256,
)
from review_agent_eval import repository as repository_models  # noqa: E402


ANNOTATION_PROTOCOL_VERSION = "core-annotation-v2"
PACKET_SCHEMA_VERSION = "core_blind_review_packet_v4"
BATCH_SCHEMA_VERSION = "core_blind_review_batch_v2"
RESPONSE_SCHEMA_VERSION = "core_independent_human_response_v3"
APPROVAL_SCHEMA_VERSION = "core_human_approval_decision_v2"
ADJUDICATION_SCHEMA_VERSION = "core_human_adjudication_v2"
LEDGER_SCHEMA_VERSION = "core_human_approval_record_v3"
PACKET_BINDING_SCHEMA_VERSION = "core_blind_review_binding_v2"
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_TEXT_CHARS = 32_768

REVIEWER_CHECKLIST_KEYS = frozenset(
    {
        "fixture_binding_reproduced",
        "intent_truth_completed_independently",
        "clarification_decision_reviewed",
        "findings_are_atomic_and_independently_fixable",
        "severity_category_context_reviewed",
        "locations_and_evidence_anchors_reviewed",
        "truth_completeness_reviewed",
        "known_invalid_findings_reviewed",
        "rationales_record_material_reasoning",
        "private_truth_and_agent_output_absence_rechecked",
    }
)
AUTHOR_CHECKLIST_KEYS = frozenset(
    {
        "binding_reverified",
        "author_truth_rechecked",
        "reviewer_independence_evidence_reviewed",
        "material_disagreements_handled",
        "fixture_and_packet_leakage_reviewed",
        "external_identity_evidence_recorded",
    }
)
ADJUDICATOR_CHECKLIST_KEYS = frozenset(
    {
        "all_material_differences_reviewed",
        "each_difference_has_one_resolution",
        "resolution_preserves_atomic_truth",
        "current_case_binding_rechecked",
        "external_identity_evidence_recorded",
    }
)
REVIEWER_ATTESTATION_KEYS = frozenset(
    {
        "reviewer_is_human",
        "review_was_blind",
        "no_author_truth_or_rationale_seen",
        "no_agent_or_judge_output_seen",
        "annotation_was_completed_independently",
    }
)
ADJUDICATOR_ATTESTATION_KEYS = frozenset(
    {
        "adjudicator_is_human",
        "independent_of_reviewer",
        "author_and_reviewer_truth_were_compared",
        "no_agent_or_judge_output_used",
    }
)
_IDENTITY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:@/+-]{2,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_OBVIOUS_MACHINE_SUBSTRINGS = (
    "anthropic",
    "chatgpt",
    "claude",
    "codex",
    "deepseek",
    "gemini",
    "openai",
)
_OBVIOUS_MACHINE_TOKENS = frozenset(
    {"agent", "assistant", "bot", "gpt", "llm", "model", "subagent"}
)


class HumanReviewError(ValueError):
    """The human-review artifact is invalid or not source-bound."""


@dataclass(frozen=True)
class VerifiedResponse:
    case: EvalCase
    packet: Mapping[str, Any]
    batch: Mapping[str, Any]
    response: Mapping[str, Any]
    raw_response_sha256: str
    response_digest: str
    independent_annotation_digest: str
    comparison: Mapping[str, Any]


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse)


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _capture_directory_chain(path: Path, context: str) -> tuple[tuple[Path, os.stat_result], ...]:
    absolute = _absolute(path)
    chain = tuple(reversed((absolute, *absolute.parents)))
    captured: list[tuple[Path, os.stat_result]] = []
    for directory in chain:
        try:
            metadata = directory.lstat()
        except FileNotFoundError as exc:
            raise HumanReviewError(f"{context} directory disappeared: {directory}") from exc
        if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise HumanReviewError(
                f"{context} contains a link, reparse point, or non-directory: {directory}"
            )
        captured.append((directory, metadata))
    return tuple(captured)


def _verify_directory_chain(
    captured: Sequence[tuple[Path, os.stat_result]], context: str
) -> None:
    for directory, before in captured:
        try:
            after = directory.lstat()
        except FileNotFoundError as exc:
            raise HumanReviewError(f"{context} directory changed during access") from exc
        if (
            _is_link_or_reparse(after)
            or not stat.S_ISDIR(after.st_mode)
            or not _same_file_identity(before, after)
        ):
            raise HumanReviewError(f"{context} directory changed during access")


def _assert_directory(path: Path, context: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise HumanReviewError(f"{context} does not exist: {path}") from exc
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise HumanReviewError(f"{context} is a link, reparse point, or non-directory")


def _assert_safe_existing_ancestors(path: Path, context: str) -> None:
    current = _absolute(path)
    existing: list[Path] = []
    while True:
        if os.path.lexists(current):
            existing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    for item in reversed(existing):
        metadata = item.lstat()
        if _is_link_or_reparse(metadata):
            raise HumanReviewError(f"{context} contains a symlink/junction/reparse point: {item}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        _absolute(path).relative_to(_absolute(root))
        return True
    except ValueError:
        return False


def _safe_relative_parts(value: str, context: str) -> tuple[str, ...]:
    if type(value) is not str or not value:
        raise HumanReviewError(f"{context} must be a non-empty relative POSIX path")
    if "\\" in value or "\x00" in value:
        raise HumanReviewError(f"{context} contains an unsafe separator or NUL")
    parts = value.split("/")
    if any(part in {"", ".", ".."} or ":" in part for part in parts):
        raise HumanReviewError(f"{context} contains an unsafe path component")
    pure = PurePosixPath(value)
    if pure.is_absolute() or tuple(pure.parts) != tuple(parts):
        raise HumanReviewError(f"{context} is not a canonical relative POSIX path")
    return tuple(parts)


def _safe_child(root: Path, relative: str, context: str) -> Path:
    root = _absolute(root)
    _assert_directory(root, context + " root")
    current = root
    for part in _safe_relative_parts(relative, context):
        current = current / part
        if os.path.lexists(current):
            metadata = current.lstat()
            if _is_link_or_reparse(metadata):
                raise HumanReviewError(f"{context} traverses a link or reparse point")
    if not _is_within(current, root):
        raise HumanReviewError(f"{context} escapes its root")
    return current


def _strict_object(value: Any, keys: Iterable[str], context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise HumanReviewError(f"{context} must be an object")
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise HumanReviewError(
            f"{context} fields differ; missing={missing!r}, unknown={unknown!r}"
        )
    return value


def _reject_constant(value: str) -> None:
    raise HumanReviewError(f"JSON contains a non-finite number: {value}")


def _pairs_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HumanReviewError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json_bytes(raw: bytes, context: str) -> dict[str, Any]:
    if len(raw) > MAX_JSON_BYTES:
        raise HumanReviewError(f"{context} exceeds {MAX_JSON_BYTES} bytes")
    try:
        text = raw.decode("utf-8", "strict")
        value = json.loads(
            text,
            object_pairs_hook=_pairs_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise HumanReviewError(f"{context} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise HumanReviewError(f"{context} must contain a JSON object")
    return value


def _read_regular(path: Path, context: str, *, outside_repository: bool = False) -> bytes:
    path = _absolute(path)
    _assert_safe_existing_ancestors(path, context)
    if outside_repository and _is_within(path, REPOSITORY_ROOT):
        raise HumanReviewError(f"{context} must remain outside the repository")
    parent_chain = _capture_directory_chain(path.parent, context)
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise HumanReviewError(f"{context} does not exist: {path}") from exc
    if (
        _is_link_or_reparse(before)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > MAX_JSON_BYTES
    ):
        raise HumanReviewError(f"{context} is a link, reparse point, or non-file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HumanReviewError(f"{context} could not be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            _is_link_or_reparse(opened)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size > MAX_JSON_BYTES
            or not _same_file_identity(before, opened)
        ):
            raise HumanReviewError(f"{context} changed before it was opened")
        chunks: list[bytes] = []
        remaining = MAX_JSON_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = path.lstat()
    except FileNotFoundError as exc:
        raise HumanReviewError(f"{context} changed while it was read") from exc
    if (
        len(raw) > MAX_JSON_BYTES
        or len(raw) != after.st_size
        or _is_link_or_reparse(after)
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or not _same_file_identity(opened, after)
        or not _same_file_identity(after, path_after)
    ):
        raise HumanReviewError(f"{context} changed while it was read")
    _verify_directory_chain(parent_chain, context)
    return raw


def _load_json_file(
    path: Path,
    context: str,
    *,
    canonical: bool = False,
    outside_repository: bool = False,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, context, outside_repository=outside_repository)
    value = _load_json_bytes(raw, context)
    if canonical and raw != canonical_json(value).encode("utf-8"):
        raise HumanReviewError(f"{context} is not canonical JSON")
    return value, raw


def _require_digest(value: Any, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise HumanReviewError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: Any, context: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value.strip()):
        raise HumanReviewError(f"{context} must be a non-empty string")
    if len(value) > MAX_TEXT_CHARS:
        raise HumanReviewError(f"{context} is too long")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise HumanReviewError(f"{context} contains invalid Unicode") from exc
    return value


def _human_identity(value: Any, context: str) -> str:
    value = _require_text(value, context)
    if _IDENTITY_RE.fullmatch(value) is None:
        raise HumanReviewError(f"{context} is not a stable opaque identity")
    folded = value.casefold()
    tokens = {item for item in re.split(r"[^a-z0-9]+", folded) if item}
    if any(marker in folded for marker in _OBVIOUS_MACHINE_SUBSTRINGS) or (
        tokens & _OBVIOUS_MACHINE_TOKENS
    ):
        raise HumanReviewError(f"{context} is an obvious Agent/LLM identity")
    return value


def _utc(value: Any, context: str) -> datetime:
    value = _require_text(value, context)
    if _UTC_RE.fullmatch(value) is None:
        raise HumanReviewError(f"{context} must be an RFC3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise HumanReviewError(f"{context} is not a valid timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise HumanReviewError(f"{context} must be UTC")
    return parsed


def _all_true(value: Any, keys: Iterable[str], context: str) -> dict[str, bool]:
    payload = _strict_object(value, keys, context)
    for key, item in payload.items():
        if item is not True:
            raise HumanReviewError(f"{context}.{key} must be true")
    return payload


def _snapshot(root: Path) -> list[dict[str, Any]]:
    _assert_directory(root, "fixture snapshot")
    entries: list[dict[str, Any]] = []

    def visit(directory: Path, prefix: tuple[str, ...]) -> None:
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda item: item.name)
        for child in children:
            metadata = child.stat(follow_symlinks=False)
            if _is_link_or_reparse(metadata):
                raise HumanReviewError("fixture contains a symlink/junction/reparse point")
            relative_parts = prefix + (child.name,)
            relative = "/".join(relative_parts)
            _safe_relative_parts(relative, "fixture path")
            if stat.S_ISDIR(metadata.st_mode):
                if child.name.casefold() in {".git", ".hg", ".svn"}:
                    raise HumanReviewError("fixture contains VCS metadata")
                visit(Path(child.path), relative_parts)
            elif stat.S_ISREG(metadata.st_mode):
                raw = _read_regular(Path(child.path), "fixture file")
                entries.append(
                    {
                        "path": relative,
                        "size_bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
            else:
                raise HumanReviewError("fixture contains a special file")

    visit(root, ())
    return entries


def fixture_manifest_from_mappings(
    base_files: Mapping[str, str], head_files: Mapping[str, str]
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for side, files in (("base", base_files), ("head", head_files)):
        for relative, text in sorted(files.items()):
            _safe_relative_parts(relative, "fixture source path")
            if type(text) is not str:
                raise HumanReviewError("fixture source values must be strings")
            raw = text.encode("utf-8")
            entries.append(
                {
                    "side": side,
                    "path": relative,
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    core = {"schema_version": "core_fixture_manifest_v1", "entries": entries}
    return {**core, "fixture_manifest_digest": canonical_sha256(core)}


def _fixture_manifest_from_root(repository_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for side in ("base", "head"):
        for item in _snapshot(repository_root / side):
            entries.append({"side": side, **item})
    core = {"schema_version": "core_fixture_manifest_v1", "entries": entries}
    return {**core, "fixture_manifest_digest": canonical_sha256(core)}


def annotation_protocol_bytes(eval_root: Path) -> bytes:
    return _read_regular(
        _safe_child(eval_root, "annotation-guidelines.md", "annotation protocol"),
        "annotation protocol",
    )


def annotation_protocol_binding(eval_root: Path) -> dict[str, str]:
    raw = annotation_protocol_bytes(eval_root)
    return {
        "version": ANNOTATION_PROTOCOL_VERSION,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def make_packet_binding(
    case: EvalCase,
    repository_binding: Mapping[str, Any],
    fixture_manifest: Mapping[str, Any],
    protocol_binding: Mapping[str, str],
) -> dict[str, Any]:
    repository = _strict_object(
        dict(repository_binding),
        {
            "base_revision",
            "head_revision",
            "base_tree",
            "head_tree",
            "source_digest",
        },
        "repository binding",
    )
    for key, value in repository.items():
        _require_text(value, f"repository binding.{key}")
    target_repository = _repository_review_target(case).repository
    if (
        repository["base_revision"] != target_repository.base_revision
        or repository["head_revision"] != target_repository.head_revision
    ):
        raise HumanReviewError("repository binding differs from EvalCase review target")
    _require_digest(fixture_manifest.get("fixture_manifest_digest"), "fixture manifest digest")
    protocol = _strict_object(
        dict(protocol_binding), {"version", "sha256"}, "annotation protocol binding"
    )
    if protocol["version"] != ANNOTATION_PROTOCOL_VERSION:
        raise HumanReviewError("annotation protocol version is stale")
    _require_digest(protocol["sha256"], "annotation protocol digest")
    return {
        "schema_version": PACKET_BINDING_SCHEMA_VERSION,
        "task_id": case.task_id,
        "case_version": case.case_version,
        "canonical_eval_input_digest": case.eval_input().digest(),
        "canonical_case_digest": canonical_sha256(case),
        "repository_binding": repository,
        "fixture_manifest_digest": fixture_manifest["fixture_manifest_digest"],
        "annotation_protocol": protocol,
    }


def make_packet(
    case: EvalCase,
    repository_binding: Mapping[str, Any],
    fixture_manifest: Mapping[str, Any],
    protocol_binding: Mapping[str, str],
) -> dict[str, Any]:
    core = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "binding": make_packet_binding(
            case, repository_binding, fixture_manifest, protocol_binding
        ),
        "eval_input": case.eval_input().to_dict(),
        "fixture_manifest": dict(fixture_manifest),
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
    }
    return {**core, "packet_digest": canonical_sha256(core)}


def _case(eval_root: Path, task_id: str) -> EvalCase:
    if _IDENTITY_RE.fullmatch(task_id) is None or "/" in task_id or "\\" in task_id:
        raise HumanReviewError("task_id is not a safe stable identifier")
    path = _safe_child(eval_root, f"cases/core/{task_id}/case.json", "Eval Case")
    raw = _read_regular(path, "Eval Case")
    try:
        return EvalCase.from_json(raw)
    except Exception as exc:
        raise HumanReviewError("Eval Case failed canonical schema hydration") from exc


def _annotation(eval_root: Path, task_id: str) -> dict[str, Any]:
    path = _safe_child(eval_root, f"cases/core/{task_id}/annotation.json", "annotation")
    payload, _ = _load_json_file(path, "annotation")
    return payload


def _repository_binding(annotation: Mapping[str, Any]) -> dict[str, Any]:
    value = annotation.get("repository_binding")
    return _strict_object(
        value,
        {"base_revision", "head_revision", "base_tree", "head_tree", "source_digest"},
        "annotation repository_binding",
    )


def _repository_review_target(case: EvalCase) -> RepositoryReviewTarget:
    target = case.input.review_target
    if type(target) is not RepositoryReviewTarget:
        raise HumanReviewError("Core human review requires a Repository review target")
    return target


def _fixture_root(eval_root: Path, case: EvalCase) -> Path:
    relative = _repository_review_target(case).repository.path
    if relative is None:
        raise HumanReviewError("Core Case fixture repository path is missing")
    root = _safe_child(eval_root, relative, "fixture repository")
    _assert_directory(root, "fixture repository")
    return root


def _replay_fixture(
    eval_root: Path, case: EvalCase, expected: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    root = _fixture_root(eval_root, case)
    manifest = _fixture_manifest_from_root(root)
    # Reuse the evaluator's deterministic Git-object algorithms without
    # materializing a throwaway loose-object repository.  This replays every
    # fixture byte while avoiding slow Windows cleanup of a 256-way object tree.
    base_files = repository_models._scan_fixture_snapshot(root / "base")
    head_files = repository_models._scan_fixture_snapshot(root / "head")
    objects: dict[str, Any] = {}
    base_tree = repository_models._build_fixture_tree(base_files, "sha1", objects)
    head_tree = repository_models._build_fixture_tree(head_files, "sha1", objects)
    base_revision = repository_models._fixture_commit(
        objects,
        "sha1",
        tree=base_tree,
        parent=None,
        message=b"review-agent-eval fixture base",
    )
    head_revision = repository_models._fixture_commit(
        objects,
        "sha1",
        tree=head_tree,
        parent=base_revision,
        message=b"review-agent-eval fixture head",
    )
    closure = repository_models._closure_from_objects(
        objects,
        object_format="sha1",
        base_revision=base_revision,
        head_revision=head_revision,
    )
    actual = {
        "base_revision": base_revision,
        "head_revision": head_revision,
        "base_tree": base_tree,
        "head_tree": head_tree,
        "source_digest": closure.source_digest,
    }
    if actual != dict(expected):
        raise HumanReviewError("fixture replay differs from the current repository binding")
    repository = _repository_review_target(case).repository
    if repository.base_revision != actual["base_revision"]:
        raise HumanReviewError("EvalInput base revision differs from fixture replay")
    if repository.head_revision != actual["head_revision"]:
        raise HumanReviewError("EvalInput head revision differs from fixture replay")
    return root, manifest


def _batch_core(batch_id: str, packets: Sequence[Mapping[str, Any]], protocol: Mapping[str, str]) -> dict[str, Any]:
    _require_text(batch_id, "batch_id")
    if _IDENTITY_RE.fullmatch(batch_id) is None:
        raise HumanReviewError("batch_id is not a stable opaque identifier")
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "annotation_protocol": dict(protocol),
        "packets": sorted((dict(item) for item in packets), key=lambda item: item["task_id"]),
    }


def _response_template(packet: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, Any]:
    binding = packet["binding"]
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "binding": {
            "task_id": binding["task_id"],
            "case_version": binding["case_version"],
            "packet_digest": packet["packet_digest"],
            "batch_id": batch["batch_id"],
            "batch_digest": batch["batch_digest"],
            "canonical_case_digest": binding["canonical_case_digest"],
            "canonical_eval_input_digest": binding["canonical_eval_input_digest"],
        },
        "reviewer": {
            "reviewer_id": "REPLACE_WITH_STABLE_HUMAN_ID",
            "started_at": "YYYY-MM-DDTHH:MM:SSZ",
            "completed_at": "YYYY-MM-DDTHH:MM:SSZ",
        },
        "attestations": {key: False for key in sorted(REVIEWER_ATTESTATION_KEYS)},
        "intent_truth": None,
        "clarification_decision": {
            "policy": None,
            "max_rounds": 1,
            "answers": [],
            "rationale": "",
            "exchanges": [],
        },
        "review_truth": None,
        "human_checklist": {key: False for key in sorted(REVIEWER_CHECKLIST_KEYS)},
        "reviewer_notes": "",
    }


def _write_new(path: Path, raw: bytes) -> None:
    path = _absolute(path)
    _assert_safe_existing_ancestors(path.parent, "output parent")
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_chain = _capture_directory_chain(path.parent, "output parent")
    _verify_directory_chain(parent_chain, "output parent")
    if os.path.lexists(path):
        raise HumanReviewError(f"refusing to overwrite {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    created_identity: os.stat_result | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        created_identity = os.fstat(descriptor)
        visible = path.lstat()
        if (
            _is_link_or_reparse(created_identity)
            or _is_link_or_reparse(visible)
            or not stat.S_ISREG(created_identity.st_mode)
            or not stat.S_ISREG(visible.st_mode)
            or created_identity.st_nlink != 1
            or visible.st_nlink != 1
            or not _same_file_identity(created_identity, visible)
        ):
            raise HumanReviewError("output file is a link, reparse point, hard link, or non-file")
        _verify_directory_chain(parent_chain, "output parent")
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise HumanReviewError("output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        visible_after = path.lstat()
        if (
            after.st_nlink != 1
            or visible_after.st_nlink != 1
            or not _same_file_identity(created_identity, after)
            or not _same_file_identity(after, visible_after)
            or _is_link_or_reparse(visible_after)
        ):
            raise HumanReviewError("output file identity changed during publication")
        _verify_directory_chain(parent_chain, "output parent")
    except FileExistsError as exc:
        raise HumanReviewError(f"refusing to overwrite {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _copy_fixture(source: Path, target: Path, manifest: Mapping[str, Any]) -> None:
    for entry in manifest["entries"]:
        relative = f"repository/{entry['side']}/{entry['path']}"
        source_path = _safe_child(source, f"{entry['side']}/{entry['path']}", "fixture file")
        raw = _read_regular(source_path, "fixture file")
        if len(raw) != entry["size_bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise HumanReviewError("fixture bytes changed while exporting")
        _write_new(target / Path(*relative.split("/")), raw)


def _private_case_strings(case: EvalCase) -> tuple[str, ...]:
    values = {case.source.suite}
    for item in case.intent_truth.expected_claims:
        values.update((item.truth_id, item.text))
    for item in case.intent_truth.forbidden_claims:
        values.update((item.truth_id, item.text, item.rationale))
    for item in case.review_truth.expected_findings:
        values.update((item.truth_id, item.claim, item.rationale))
        values.update(anchor.fact for anchor in item.evidence_anchors)
    for item in case.review_truth.known_invalid_findings:
        values.update((item.truth_id, item.claim, item.rationale))
    visible = canonical_json(case.eval_input().to_dict()).casefold()
    return tuple(
        sorted(
            value
            for value in values
            if len(value) >= 6 and value.casefold() not in visible
        )
    )


def _assert_packet_has_no_private_truth(case: EvalCase, packet: Mapping[str, Any], template: Mapping[str, Any]) -> None:
    visible_artifacts = (canonical_json(packet) + canonical_json(template)).casefold()
    for value in _private_case_strings(case):
        if value.casefold() in visible_artifacts:
            raise HumanReviewError("blind packet leaks Author-private truth or suite placement")


def export_blind_review_batch(
    eval_root: Path,
    output_directory: Path,
    task_ids: Sequence[str],
    batch_id: str,
) -> Mapping[str, Any]:
    eval_root = _absolute(eval_root)
    output_directory = _absolute(output_directory)
    _assert_directory(eval_root, "Eval root")
    _assert_safe_existing_ancestors(output_directory, "blind-review output")
    if _is_within(output_directory, REPOSITORY_ROOT):
        raise HumanReviewError("blind-review output must be outside the repository")
    if os.path.lexists(output_directory):
        raise HumanReviewError("blind-review output already exists; overwrite is forbidden")
    parent = output_directory.parent
    _assert_directory(parent, "blind-review output parent")
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise HumanReviewError("task_ids must be a non-empty unique list")

    protocol_raw = annotation_protocol_bytes(eval_root)
    protocol = annotation_protocol_binding(eval_root)
    prepared: list[tuple[EvalCase, Path, dict[str, Any], dict[str, Any]]] = []
    packet_refs: list[dict[str, Any]] = []
    for task_id in sorted(task_ids):
        case = _case(eval_root, task_id)
        annotation = _annotation(eval_root, task_id)
        repository = _repository_binding(annotation)
        fixture_root, fixture_manifest = _replay_fixture(eval_root, case, repository)
        packet = make_packet(case, repository, fixture_manifest, protocol)
        prepared.append((case, fixture_root, fixture_manifest, packet))
        packet_refs.append(
            {
                "task_id": case.task_id,
                "case_version": case.case_version,
                "packet_digest": packet["packet_digest"],
                "canonical_case_digest": packet["binding"]["canonical_case_digest"],
            }
        )
    batch_core = _batch_core(batch_id, packet_refs, protocol)
    batch = {**batch_core, "batch_digest": canonical_sha256(batch_core)}

    temporary: Path | None = parent / (
        ".core-blind-review-" + batch["batch_digest"]
    )
    if os.path.lexists(temporary):
        raise HumanReviewError(
            "deterministic blind-review staging directory already exists; refusing reuse"
        )
    temporary.mkdir()
    try:
        assert temporary is not None
        _write_new(temporary / "batch.json", canonical_json(batch).encode("utf-8"))
        _write_new(temporary / "protocol.md", protocol_raw)
        for case, fixture_root, fixture_manifest, packet in prepared:
            case_root = temporary / "cases" / case.task_id
            template = _response_template(packet, batch)
            _assert_packet_has_no_private_truth(case, packet, template)
            _write_new(case_root / "packet.json", canonical_json(packet).encode("utf-8"))
            _write_new(
                case_root / "response-template.json",
                (json.dumps(template, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
            _copy_fixture(fixture_root, case_root, fixture_manifest)
        verify_blind_review_batch(eval_root, temporary)
        repository_models._rename_directory_no_replace(
            temporary, output_directory
        )
        temporary = None
    finally:
        if temporary is not None and os.path.lexists(temporary):
            repository_models._remove_tree_safely(parent, temporary)
    return batch


def _validate_protocol(value: Any, context: str) -> dict[str, Any]:
    payload = _strict_object(value, {"version", "sha256"}, context)
    if payload["version"] != ANNOTATION_PROTOCOL_VERSION:
        raise HumanReviewError(f"{context} has a stale version")
    _require_digest(payload["sha256"], context + ".sha256")
    return payload


def _validate_batch(batch: Mapping[str, Any]) -> None:
    payload = _strict_object(
        batch,
        {"schema_version", "batch_id", "annotation_protocol", "packets", "batch_digest"},
        "blind-review batch",
    )
    if payload["schema_version"] != BATCH_SCHEMA_VERSION:
        raise HumanReviewError("blind-review batch schema is unsupported")
    _require_text(payload["batch_id"], "batch_id")
    _validate_protocol(payload["annotation_protocol"], "batch annotation_protocol")
    if type(payload["packets"]) is not list or not payload["packets"]:
        raise HumanReviewError("batch packets must be a non-empty list")
    task_ids: set[str] = set()
    for item in payload["packets"]:
        item = _strict_object(
            item,
            {"task_id", "case_version", "packet_digest", "canonical_case_digest"},
            "batch packet reference",
        )
        _require_text(item["task_id"], "batch packet task_id")
        if item["task_id"] in task_ids:
            raise HumanReviewError("batch contains duplicate task_id")
        task_ids.add(item["task_id"])
        if type(item["case_version"]) is not int or item["case_version"] < 1:
            raise HumanReviewError("batch packet case_version is invalid")
        _require_digest(item["packet_digest"], "batch packet digest")
        _require_digest(item["canonical_case_digest"], "batch case digest")
    core = {key: payload[key] for key in payload if key != "batch_digest"}
    if _require_digest(payload["batch_digest"], "batch digest") != canonical_sha256(core):
        raise HumanReviewError("blind-review batch digest does not match")


def _validate_fixture_manifest(value: Any) -> dict[str, Any]:
    payload = _strict_object(
        value,
        {"schema_version", "entries", "fixture_manifest_digest"},
        "fixture manifest",
    )
    if payload["schema_version"] != "core_fixture_manifest_v1":
        raise HumanReviewError("fixture manifest schema is unsupported")
    if type(payload["entries"]) is not list:
        raise HumanReviewError("fixture manifest entries must be a list")
    previous: tuple[str, str] | None = None
    for entry in payload["entries"]:
        entry = _strict_object(
            entry, {"side", "path", "size_bytes", "sha256"}, "fixture manifest entry"
        )
        if entry["side"] not in {"base", "head"}:
            raise HumanReviewError("fixture side must be base or head")
        _safe_relative_parts(entry["path"], "fixture manifest path")
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] < 0:
            raise HumanReviewError("fixture size is invalid")
        _require_digest(entry["sha256"], "fixture sha256")
        key = (entry["side"], entry["path"])
        if previous is not None and key <= previous:
            raise HumanReviewError("fixture manifest entries are not unique and sorted")
        previous = key
    core = {"schema_version": payload["schema_version"], "entries": payload["entries"]}
    if payload["fixture_manifest_digest"] != canonical_sha256(core):
        raise HumanReviewError("fixture manifest digest does not match")
    return payload


def _validate_packet_binding(
    value: Any,
    eval_input: EvalInput,
    fixture_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _strict_object(
        value,
        {
            "schema_version",
            "task_id",
            "case_version",
            "canonical_eval_input_digest",
            "canonical_case_digest",
            "repository_binding",
            "fixture_manifest_digest",
            "annotation_protocol",
        },
        "blind-review packet binding",
    )
    if payload["schema_version"] != PACKET_BINDING_SCHEMA_VERSION:
        raise HumanReviewError(
            "blind-review packet binding is stale and requires independent re-review"
        )
    if payload["task_id"] != eval_input.task_id:
        raise HumanReviewError("packet binding task differs from EvalInput")
    if type(payload["case_version"]) is not int or payload["case_version"] < 1:
        raise HumanReviewError("packet binding Case version is invalid")
    if payload["canonical_eval_input_digest"] != eval_input.digest():
        raise HumanReviewError("packet binding EvalInput digest does not match")
    _require_digest(payload["canonical_case_digest"], "packet binding Case digest")
    if payload["fixture_manifest_digest"] != fixture_manifest["fixture_manifest_digest"]:
        raise HumanReviewError("packet binding fixture digest does not match")
    protocol = _validate_protocol(
        payload["annotation_protocol"], "packet annotation_protocol"
    )
    repository_binding = _strict_object(
        payload["repository_binding"],
        {"base_revision", "head_revision", "base_tree", "head_tree", "source_digest"},
        "packet repository binding",
    )
    target = eval_input.review_target
    if type(target) is not RepositoryReviewTarget:
        raise HumanReviewError("Core blind-review packet Target must be Repository")
    if (
        repository_binding["base_revision"] != target.repository.base_revision
        or repository_binding["head_revision"] != target.repository.head_revision
    ):
        raise HumanReviewError("packet Repository binding differs from EvalInput")
    return {**payload, "annotation_protocol": protocol}


def _validate_packet(packet: Mapping[str, Any]) -> None:
    payload = _strict_object(
        packet,
        {
            "schema_version",
            "binding",
            "eval_input",
            "fixture_manifest",
            "response_schema_version",
            "packet_digest",
        },
        "blind-review packet",
    )
    if payload["schema_version"] != PACKET_SCHEMA_VERSION:
        raise HumanReviewError("blind-review packet schema is unsupported")
    if payload["response_schema_version"] != RESPONSE_SCHEMA_VERSION:
        raise HumanReviewError("blind-review response schema is stale")
    fixture_manifest = _validate_fixture_manifest(payload["fixture_manifest"])
    try:
        eval_input = EvalInput.from_dict(payload["eval_input"])
    except Exception as exc:
        raise HumanReviewError(
            "blind-review packet EvalInput is not the sole v2 projection"
        ) from exc
    _validate_packet_binding(payload["binding"], eval_input, fixture_manifest)
    core = {key: payload[key] for key in payload if key != "packet_digest"}
    if _require_digest(payload["packet_digest"], "packet digest") != canonical_sha256(core):
        raise HumanReviewError("blind-review packet digest does not match")


def _expected_packet_files(packet: Mapping[str, Any], task_id: str) -> set[str]:
    expected = {
        "batch.json",
        "protocol.md",
        f"cases/{task_id}/packet.json",
        f"cases/{task_id}/response-template.json",
    }
    for entry in packet["fixture_manifest"]["entries"]:
        expected.add(f"cases/{task_id}/repository/{entry['side']}/{entry['path']}")
    return expected


def _walk_regular_files(root: Path) -> set[str]:
    result: set[str] = set()

    def visit(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            entries = list(iterator)
        for entry in entries:
            path = Path(entry.path)
            metadata = entry.stat(follow_symlinks=False)
            if _is_link_or_reparse(metadata):
                raise HumanReviewError("packet tree contains a symlink/junction/reparse point")
            if stat.S_ISDIR(metadata.st_mode):
                visit(path)
            elif stat.S_ISREG(metadata.st_mode):
                result.add(path.relative_to(root).as_posix())
            else:
                raise HumanReviewError("packet tree contains a special file")

    visit(root)
    return result


def verify_blind_review_batch(eval_root: Path, batch_directory: Path) -> Mapping[str, Any]:
    eval_root = _absolute(eval_root)
    batch_directory = _absolute(batch_directory)
    _assert_directory(eval_root, "Eval root")
    _assert_safe_existing_ancestors(batch_directory, "blind-review batch")
    _assert_directory(batch_directory, "blind-review batch")
    if _is_within(batch_directory, REPOSITORY_ROOT):
        raise HumanReviewError("blind-review batch must remain outside the repository")
    batch, _ = _load_json_file(
        batch_directory / "batch.json", "batch manifest", canonical=True, outside_repository=True
    )
    _validate_batch(batch)
    protocol_raw = _read_regular(
        batch_directory / "protocol.md", "packet protocol", outside_repository=True
    )
    if hashlib.sha256(protocol_raw).hexdigest() != batch["annotation_protocol"]["sha256"]:
        raise HumanReviewError("packet protocol bytes differ from the batch binding")
    if protocol_raw != annotation_protocol_bytes(eval_root):
        raise HumanReviewError("packet protocol is stale relative to the current evaluator")

    expected_files = {"batch.json", "protocol.md"}
    for reference in batch["packets"]:
        task_id = reference["task_id"]
        case = _case(eval_root, task_id)
        annotation = _annotation(eval_root, task_id)
        repository = _repository_binding(annotation)
        fixture_root, manifest = _replay_fixture(eval_root, case, repository)
        packet_path = batch_directory / "cases" / task_id / "packet.json"
        packet, _ = _load_json_file(
            packet_path, "blind-review packet", canonical=True, outside_repository=True
        )
        _validate_packet(packet)
        expected_packet = make_packet(
            case, repository, manifest, batch["annotation_protocol"]
        )
        if packet != expected_packet:
            raise HumanReviewError("packet is stale or differs from the current Case binding")
        if reference != {
            "task_id": case.task_id,
            "case_version": case.case_version,
            "packet_digest": packet["packet_digest"],
            "canonical_case_digest": packet["binding"]["canonical_case_digest"],
        }:
            raise HumanReviewError("batch packet reference differs from its packet")
        template, _ = _load_json_file(
            batch_directory / "cases" / task_id / "response-template.json",
            "response template",
            outside_repository=True,
        )
        expected_template = _response_template(packet, batch)
        if template != expected_template:
            raise HumanReviewError("response template differs from its packet/batch binding")
        _assert_packet_has_no_private_truth(case, packet, template)
        expected_files.update(_expected_packet_files(packet, task_id))
        for entry in manifest["entries"]:
            source = _read_regular(
                _safe_child(fixture_root, f"{entry['side']}/{entry['path']}", "fixture source"),
                "fixture source",
            )
            packet_file = _read_regular(
                batch_directory
                / "cases"
                / task_id
                / "repository"
                / entry["side"]
                / Path(*entry["path"].split("/")),
                "packet fixture",
                outside_repository=True,
            )
            if packet_file != source:
                raise HumanReviewError("packet fixture bytes do not replay exactly")
    actual_files = _walk_regular_files(batch_directory)
    if actual_files != expected_files:
        raise HumanReviewError(
            "packet contains missing or forbidden files: "
            f"missing={sorted(expected_files - actual_files)!r}, "
            f"unexpected={sorted(actual_files - expected_files)!r}"
        )
    return batch


def _response_binding(packet: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, Any]:
    return _response_template(packet, batch)["binding"]


def _hydrate_clarification_decision(
    value: Any,
    *,
    started: datetime,
    completed: datetime,
) -> dict[str, Any]:
    payload = _strict_object(
        value,
        {"policy", "max_rounds", "answers", "rationale", "exchanges"},
        "clarification_decision",
    )
    if payload["policy"] not in {"required", "optional", "not_required"}:
        raise HumanReviewError("clarification_decision.policy is invalid")
    _require_text(payload["rationale"], "clarification_decision.rationale")
    try:
        script = ClarificationScript.from_dict(
            {"max_rounds": payload["max_rounds"], "answers": payload["answers"]}
        )
    except Exception as exc:
        raise HumanReviewError(
            "clarification_decision answers failed canonical ClarificationScript/ClarificationAnswer hydration"
        ) from exc

    if type(payload["exchanges"]) is not list:
        raise HumanReviewError("clarification_decision.exchanges must be a list")
    answer_ids = {item.answer_id for item in script.answers}
    exchange_ids: set[str] = set()
    exchanges: list[dict[str, Any]] = []
    for index, value in enumerate(payload["exchanges"]):
        exchange = _strict_object(
            value,
            {"answer_id", "question", "answer", "answered_at"},
            f"clarification exchange {index}",
        )
        answer_id = _require_text(exchange["answer_id"], "clarification exchange answer_id")
        if _IDENTITY_RE.fullmatch(answer_id) is None:
            raise HumanReviewError("clarification exchange answer_id is not an opaque identifier")
        if answer_id not in answer_ids:
            raise HumanReviewError(
                "clarification exchange answer_id does not bind a Reviewer answer"
            )
        if answer_id in exchange_ids:
            raise HumanReviewError("clarification answer has more than one exchange")
        exchange_ids.add(answer_id)
        _require_text(exchange["question"], "clarification question")
        _require_text(exchange["answer"], "clarification answer")
        answered_at = _utc(exchange["answered_at"], "clarification answered_at")
        if answered_at < started or answered_at > completed:
            raise HumanReviewError(
                "clarification exchange time must fall within the blind review interval"
            )
        exchanges.append(dict(exchange))

    policy = payload["policy"]
    if policy == "required":
        if not answer_ids:
            raise HumanReviewError("required clarification must include at least one answer")
        if exchange_ids != answer_ids:
            raise HumanReviewError(
                "required clarification must bind exactly one exchange to every answer"
            )
    elif policy == "not_required":
        if answer_ids or exchange_ids:
            raise HumanReviewError(
                "not_required clarification must have empty answers and exchanges"
            )
    elif exchange_ids != answer_ids:
        raise HumanReviewError(
            "optional clarification must bind exactly one exchange to every recorded answer"
        )

    return {
        "policy": policy,
        "max_rounds": script.max_rounds,
        "answers": [item.to_dict() for item in script.answers],
        "rationale": payload["rationale"],
        "exchanges": sorted(exchanges, key=lambda item: item["answer_id"]),
    }


def _validate_response(
    response: Mapping[str, Any], packet: Mapping[str, Any], batch: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = _strict_object(
        response,
        {
            "schema_version",
            "binding",
            "reviewer",
            "attestations",
            "intent_truth",
            "clarification_decision",
            "review_truth",
            "human_checklist",
            "reviewer_notes",
        },
        "independent human response",
    )
    if payload["schema_version"] != RESPONSE_SCHEMA_VERSION:
        raise HumanReviewError("independent human response schema is unsupported")
    if payload["binding"] != _response_binding(packet, batch):
        raise HumanReviewError("human response binding is stale or does not match the packet")
    reviewer = _strict_object(
        payload["reviewer"], {"reviewer_id", "started_at", "completed_at"}, "reviewer"
    )
    _human_identity(reviewer["reviewer_id"], "reviewer_id")
    started = _utc(reviewer["started_at"], "reviewer.started_at")
    completed = _utc(reviewer["completed_at"], "reviewer.completed_at")
    if completed <= started:
        raise HumanReviewError("review completion must be after review start")
    _all_true(payload["attestations"], REVIEWER_ATTESTATION_KEYS, "reviewer attestations")
    _all_true(payload["human_checklist"], REVIEWER_CHECKLIST_KEYS, "reviewer checklist")
    _require_text(payload["reviewer_notes"], "reviewer_notes", allow_empty=True)
    try:
        intent = IntentTruth.from_dict(payload["intent_truth"])
        review = ReviewTruth.from_dict(payload["review_truth"])
    except Exception as exc:
        raise HumanReviewError("independent truth failed canonical model hydration") from exc
    clarification = _hydrate_clarification_decision(
        payload["clarification_decision"], started=started, completed=completed
    )
    intent_policy = None if intent.clarification_policy is None else intent.clarification_policy.value
    if clarification["policy"] != intent_policy:
        raise HumanReviewError("clarification decision differs from independent IntentTruth")
    return intent.to_dict(), clarification, review.to_dict()


_MISSING = {"__missing__": True}


def _material_differences(author: Any, reviewer: Any, path: str = "") -> list[dict[str, str]]:
    differences: list[dict[str, str]] = []
    if type(author) is dict and type(reviewer) is dict:
        for key in sorted(set(author) | set(reviewer)):
            child = f"{path}/{key}"
            differences.extend(
                _material_differences(author.get(key, _MISSING), reviewer.get(key, _MISSING), child)
            )
        return differences
    if type(author) is list and type(reviewer) is list:
        for index in range(max(len(author), len(reviewer))):
            left = author[index] if index < len(author) else _MISSING
            right = reviewer[index] if index < len(reviewer) else _MISSING
            differences.extend(_material_differences(left, right, f"{path}/{index}"))
        return differences
    if type(author) is type(reviewer) and author == reviewer:
        return differences
    left_digest = canonical_sha256(author)
    right_digest = canonical_sha256(reviewer)
    core = {
        "path": path or "/",
        "author_value_digest": left_digest,
        "reviewer_value_digest": right_digest,
    }
    differences.append({"difference_id": canonical_sha256(core), **core})
    return differences


def compare_independent_truth(case: EvalCase, response: Mapping[str, Any]) -> dict[str, Any]:
    def semantics(policy: str | None, max_rounds: int, answers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        normalized_answers = []
        for answer in answers:
            normalized_answers.append(
                {
                    "dimension": answer["dimension"],
                    "material_claim": answer["material_claim"],
                    "action": answer["action"],
                    "response": answer["response"],
                    "corrected_values": answer["corrected_values"],
                }
            )
        normalized_answers.sort(key=canonical_json)
        return {
            "policy": policy,
            "max_rounds": max_rounds,
            "answers": normalized_answers,
        }

    author_script = case.clarification_script.to_dict()
    reviewer_decision = response["clarification_decision"]
    author_truth = {
        "intent_truth": case.intent_truth.to_dict(),
        "clarification_decision": semantics(
            None if case.intent_truth.clarification_policy is None else case.intent_truth.clarification_policy.value,
            author_script["max_rounds"],
            author_script["answers"],
        ),
        "review_truth": case.review_truth.to_dict(),
    }
    reviewer_truth = {
        "intent_truth": response["intent_truth"],
        "clarification_decision": semantics(
            reviewer_decision["policy"],
            reviewer_decision["max_rounds"],
            reviewer_decision["answers"],
        ),
        "review_truth": response["review_truth"],
    }
    differences = _material_differences(author_truth, reviewer_truth)
    core = {
        "schema_version": "core_human_truth_comparison_v2",
        "author_truth_digest": canonical_sha256(author_truth),
        "independent_truth_digest": canonical_sha256(reviewer_truth),
        "material_disagreement": bool(differences),
        "differences": differences,
    }
    return {**core, "comparison_digest": canonical_sha256(core)}


def _packet_for_task(batch_directory: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    batch, _ = _load_json_file(
        batch_directory / "batch.json", "batch manifest", canonical=True, outside_repository=True
    )
    packet, _ = _load_json_file(
        batch_directory / "cases" / task_id / "packet.json",
        "blind-review packet",
        canonical=True,
        outside_repository=True,
    )
    return batch, packet


def verify_completed_response(
    eval_root: Path, batch_directory: Path, response_path: Path
) -> VerifiedResponse:
    batch = verify_blind_review_batch(eval_root, batch_directory)
    response, raw = _load_json_file(
        response_path, "independent human response", outside_repository=True
    )
    binding = response.get("binding")
    if type(binding) is not dict or type(binding.get("task_id")) is not str:
        raise HumanReviewError("human response has no usable task binding")
    task_id = binding["task_id"]
    references = [item for item in batch["packets"] if item["task_id"] == task_id]
    if len(references) != 1:
        raise HumanReviewError("human response task is not uniquely present in the batch")
    _, packet = _packet_for_task(_absolute(batch_directory), task_id)
    intent, clarification, review = _validate_response(response, packet, batch)
    canonical_response = dict(response)
    canonical_response["intent_truth"] = intent
    canonical_response["clarification_decision"] = clarification
    canonical_response["review_truth"] = review
    independent = {
        "intent_truth": intent,
        "clarification_decision": clarification,
        "review_truth": review,
    }
    case = _case(_absolute(eval_root), task_id)
    comparison = compare_independent_truth(case, canonical_response)
    return VerifiedResponse(
        case=case,
        packet=packet,
        batch=batch,
        response=canonical_response,
        raw_response_sha256=hashlib.sha256(raw).hexdigest(),
        response_digest=canonical_sha256(canonical_response),
        independent_annotation_digest=canonical_sha256(independent),
        comparison=comparison,
    )


def _decision_binding(verified: VerifiedResponse) -> dict[str, Any]:
    return {
        "task_id": verified.case.task_id,
        "case_version": verified.case.case_version,
        "packet_digest": verified.packet["packet_digest"],
        "response_digest": verified.response_digest,
        "comparison_digest": verified.comparison["comparison_digest"],
    }


def _audit_evidence(value: Any, needs_adjudicator: bool) -> dict[str, Any]:
    payload = _strict_object(
        value,
        {"author_reference", "reviewer_reference", "adjudicator_reference"},
        "external_audit_evidence",
    )
    _require_text(payload["author_reference"], "author audit reference")
    _require_text(payload["reviewer_reference"], "reviewer audit reference")
    if needs_adjudicator:
        _require_text(payload["adjudicator_reference"], "adjudicator audit reference")
    elif payload["adjudicator_reference"] is not None:
        raise HumanReviewError("adjudicator audit reference must be null without adjudication")
    return payload


def _validate_approval(value: Mapping[str, Any], verified: VerifiedResponse, needs_adjudicator: bool) -> dict[str, Any]:
    payload = _strict_object(
        value,
        {
            "schema_version",
            "binding",
            "author_id",
            "signed_at",
            "final_decision",
            "leakage_review_completed",
            "author_checklist",
            "external_audit_evidence",
        },
        "Author approval",
    )
    if payload["schema_version"] != APPROVAL_SCHEMA_VERSION:
        raise HumanReviewError("Author approval schema is unsupported")
    if payload["binding"] != _decision_binding(verified):
        raise HumanReviewError("Author approval binding is stale")
    _human_identity(payload["author_id"], "author_id")
    signed_at = _utc(payload["signed_at"], "approval.signed_at")
    review_completed = _utc(
        verified.response["reviewer"]["completed_at"], "reviewer.completed_at"
    )
    if signed_at <= review_completed:
        raise HumanReviewError("Author signature must be after independent review completion")
    if payload["final_decision"] not in {"accepted", "rejected"}:
        raise HumanReviewError("final_decision must be accepted or rejected")
    if payload["leakage_review_completed"] is not True:
        raise HumanReviewError("leakage review must be completed")
    _all_true(payload["author_checklist"], AUTHOR_CHECKLIST_KEYS, "Author checklist")
    _audit_evidence(payload["external_audit_evidence"], needs_adjudicator)
    return payload


def _validate_adjudication(value: Mapping[str, Any], verified: VerifiedResponse) -> dict[str, Any]:
    payload = _strict_object(
        value,
        {
            "schema_version",
            "binding",
            "adjudicator_id",
            "started_at",
            "completed_at",
            "decision",
            "rationale",
            "resolutions",
            "attestations",
            "human_checklist",
        },
        "adjudication",
    )
    if payload["schema_version"] != ADJUDICATION_SCHEMA_VERSION:
        raise HumanReviewError("adjudication schema is unsupported")
    if payload["binding"] != _decision_binding(verified):
        raise HumanReviewError("adjudication binding is stale")
    _human_identity(payload["adjudicator_id"], "adjudicator_id")
    started = _utc(payload["started_at"], "adjudication.started_at")
    completed = _utc(payload["completed_at"], "adjudication.completed_at")
    if completed <= started:
        raise HumanReviewError("adjudication completion must be after start")
    review_completed = _utc(
        verified.response["reviewer"]["completed_at"], "reviewer.completed_at"
    )
    if started <= review_completed:
        raise HumanReviewError(
            "adjudication must start after independent review completion"
        )
    if payload["decision"] not in {"accepted", "rejected"}:
        raise HumanReviewError("adjudication decision must be accepted or rejected")
    _require_text(payload["rationale"], "adjudication rationale")
    _all_true(payload["attestations"], ADJUDICATOR_ATTESTATION_KEYS, "adjudicator attestations")
    _all_true(payload["human_checklist"], ADJUDICATOR_CHECKLIST_KEYS, "adjudicator checklist")
    if type(payload["resolutions"]) is not list:
        raise HumanReviewError("adjudication resolutions must be a list")
    expected = {item["difference_id"] for item in verified.comparison["differences"]}
    seen: set[str] = set()
    accepted_current_truth = True
    for item in payload["resolutions"]:
        item = _strict_object(
            item, {"difference_id", "resolution", "rationale"}, "adjudication resolution"
        )
        _require_digest(item["difference_id"], "resolution difference_id")
        if item["difference_id"] in seen:
            raise HumanReviewError("adjudication repeats a difference resolution")
        seen.add(item["difference_id"])
        if item["resolution"] not in {"author", "reviewer", "merged", "case_rework"}:
            raise HumanReviewError("adjudication resolution is invalid")
        if item["resolution"] != "author":
            accepted_current_truth = False
        _require_text(item["rationale"], "resolution rationale")
    if seen != expected:
        raise HumanReviewError("adjudication must resolve every material difference exactly once")
    if payload["decision"] == "accepted" and not accepted_current_truth:
        raise HumanReviewError(
            "current Author truth can only be accepted when every difference resolves to author; revise and rebind the Case otherwise"
        )
    return payload


def _record_core(
    verified: VerifiedResponse,
    approval: Mapping[str, Any],
    adjudication: Mapping[str, Any] | None,
) -> dict[str, Any]:
    reviewer_id = verified.response["reviewer"]["reviewer_id"]
    author_id = approval["author_id"]
    if author_id == reviewer_id:
        raise HumanReviewError("Author A and Reviewer B must have different identities")
    adjudicator_id = None
    if adjudication is not None:
        adjudicator_id = adjudication["adjudicator_id"]
        if adjudicator_id in {author_id, reviewer_id}:
            raise HumanReviewError("Adjudicator C must be independent of Author A and Reviewer B")
        if adjudication["decision"] != approval["final_decision"]:
            raise HumanReviewError("Author final decision differs from adjudication")
        if _utc(approval["signed_at"], "approval.signed_at") <= _utc(
            adjudication["completed_at"], "adjudication.completed_at"
        ):
            raise HumanReviewError("Author signature must be after adjudication completion")
    final_decision = approval["final_decision"]
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record_revision": 1,
        "previous_record_digest": None,
        "status": "approved" if final_decision == "accepted" else "rejected",
        "final_decision": final_decision,
        "binding": verified.packet["binding"],
        "batch_manifest": dict(verified.batch),
        "packet_digest": verified.packet["packet_digest"],
        "response": {
            "canonical_response": verified.response,
            "response_digest": verified.response_digest,
            "raw_response_sha256": verified.raw_response_sha256,
            "independent_annotation_digest": verified.independent_annotation_digest,
        },
        "comparison": verified.comparison,
        "approval": approval,
        "adjudication": adjudication,
        "identities": {
            "author_id": author_id,
            "reviewer_id": reviewer_id,
            "adjudicator_id": adjudicator_id,
        },
        "timestamps": {
            "blind_review_started_at": verified.response["reviewer"]["started_at"],
            "blind_review_completed_at": verified.response["reviewer"]["completed_at"],
            "author_signed_at": approval["signed_at"],
        },
        "leakage_review_completed": True,
        "checklists": {
            "reviewer": verified.response["human_checklist"],
            "author": approval["author_checklist"],
            "adjudicator": None if adjudication is None else adjudication["human_checklist"],
        },
        "external_audit_evidence": approval["external_audit_evidence"],
    }


def _record_with_digest(core: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(core), "record_digest": canonical_sha256(core)}


def _record_path(ledger_root: Path, task_id: str) -> Path:
    if _IDENTITY_RE.fullmatch(task_id) is None or "/" in task_id or "\\" in task_id:
        raise HumanReviewError("ledger task_id is unsafe")
    root = _absolute(ledger_root)
    _assert_safe_existing_ancestors(root, "human-review ledger")
    if not os.path.lexists(root):
        root.mkdir(parents=True)
    _assert_directory(root, "human-review ledger")
    records = root / "records"
    if not os.path.lexists(records):
        records.mkdir()
    _assert_directory(records, "human-review ledger records")
    return records / f"{task_id}.json"


def _atomic_write_new(path: Path, raw: bytes) -> None:
    path = _absolute(path)
    parent_chain = _capture_directory_chain(path.parent, "human-review ledger parent")
    _verify_directory_chain(parent_chain, "human-review ledger parent")
    if os.path.lexists(path):
        raise HumanReviewError("ledger record already exists; overwrite/rollback is forbidden")
    temporary: Path | None = None
    temporary_identity: os.stat_result | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".human-review-ledger-", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            temporary_identity = os.fstat(stream.fileno())
            visible_temporary = temporary.lstat()
            if (
                _is_link_or_reparse(temporary_identity)
                or _is_link_or_reparse(visible_temporary)
                or not stat.S_ISREG(temporary_identity.st_mode)
                or not stat.S_ISREG(visible_temporary.st_mode)
                or temporary_identity.st_nlink != 1
                or visible_temporary.st_nlink != 1
                or not _same_file_identity(temporary_identity, visible_temporary)
            ):
                raise HumanReviewError(
                    "ledger temporary is a link, reparse point, hard link, or non-file"
                )
            _verify_directory_chain(parent_chain, "human-review ledger parent")
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            after_write = os.fstat(stream.fileno())
            if (
                after_write.st_nlink != 1
                or not _same_file_identity(temporary_identity, after_write)
            ):
                raise HumanReviewError("ledger temporary identity changed during write")
        _verify_directory_chain(parent_chain, "human-review ledger parent")
        visible_temporary = temporary.lstat()
        if (
            _is_link_or_reparse(visible_temporary)
            or not stat.S_ISREG(visible_temporary.st_mode)
            or visible_temporary.st_nlink != 1
            or not _same_file_identity(temporary_identity, visible_temporary)
        ):
            raise HumanReviewError("ledger temporary changed before publication")
        if os.path.lexists(path):
            raise HumanReviewError(
                "ledger record appeared concurrently; overwrite is forbidden"
            )
        _verify_directory_chain(parent_chain, "human-review ledger parent")
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise HumanReviewError(
                "ledger record appeared concurrently; overwrite is forbidden"
            ) from exc
        except OSError as exc:
            raise HumanReviewError(
                "filesystem cannot atomically publish a no-overwrite ledger record"
            ) from exc
        published = path.lstat()
        temporary_after_link = temporary.lstat()
        if (
            _is_link_or_reparse(published)
            or not stat.S_ISREG(published.st_mode)
            or published.st_nlink != 2
            or temporary_after_link.st_nlink != 2
            or not _same_file_identity(temporary_identity, published)
            or not _same_file_identity(published, temporary_after_link)
        ):
            raise HumanReviewError("ledger publication did not preserve file identity")
        _verify_directory_chain(parent_chain, "human-review ledger parent")
        temporary.unlink()
        temporary = None
        final = path.lstat()
        if (
            _is_link_or_reparse(final)
            or not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or not _same_file_identity(temporary_identity, final)
        ):
            raise HumanReviewError("ledger record is not a unique regular file")
        _verify_directory_chain(parent_chain, "human-review ledger parent")
    finally:
        if temporary is not None:
            try:
                _verify_directory_chain(parent_chain, "human-review ledger parent")
                visible = temporary.lstat()
                if temporary_identity is not None and _same_file_identity(
                    temporary_identity, visible
                ):
                    temporary.unlink()
            except FileNotFoundError:
                pass


def import_approved_response(
    eval_root: Path,
    batch_directory: Path,
    response_path: Path,
    approval_path: Path,
    ledger_root: Path,
    adjudication_path: Path | None = None,
) -> Mapping[str, Any]:
    verified = verify_completed_response(eval_root, batch_directory, response_path)
    has_disagreement = bool(verified.comparison["material_disagreement"])
    if has_disagreement and adjudication_path is None:
        raise HumanReviewError(
            "material disagreement requires explicit Adjudicator C review before any approval/rejection"
        )
    if not has_disagreement and adjudication_path is not None:
        raise HumanReviewError("adjudication is forbidden when deterministic comparison has no disagreement")
    adjudication = None
    if adjudication_path is not None:
        value, _ = _load_json_file(
            adjudication_path, "adjudication", outside_repository=True
        )
        adjudication = _validate_adjudication(value, verified)
    approval_value, _ = _load_json_file(
        approval_path, "Author approval", outside_repository=True
    )
    approval = _validate_approval(approval_value, verified, adjudication is not None)
    core = _record_core(verified, approval, adjudication)
    record = _record_with_digest(core)
    path = _record_path(ledger_root, verified.case.task_id)
    _atomic_write_new(path, canonical_json(record).encode("utf-8"))
    return record


def _hydrate_ledger_record(
    record: Mapping[str, Any], case: EvalCase, expected_packet: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _strict_object(
        record,
        {
            "schema_version",
            "record_revision",
            "previous_record_digest",
            "status",
            "final_decision",
            "binding",
            "batch_manifest",
            "packet_digest",
            "response",
            "comparison",
            "approval",
            "adjudication",
            "identities",
            "timestamps",
            "leakage_review_completed",
            "checklists",
            "external_audit_evidence",
            "record_digest",
        },
        "human-review ledger record",
    )
    core = {key: payload[key] for key in payload if key != "record_digest"}
    if payload["schema_version"] != LEDGER_SCHEMA_VERSION:
        raise HumanReviewError(
            "ledger schema is stale and requires independent re-review"
        )
    if payload["record_revision"] != 1 or payload["previous_record_digest"] is not None:
        raise HumanReviewError("ledger record attempts an unsupported update/rollback")
    if _require_digest(payload["record_digest"], "record digest") != canonical_sha256(core):
        raise HumanReviewError("ledger record digest does not match")
    if payload["binding"] != expected_packet["binding"]:
        raise HumanReviewError(
            "ledger record has a stale Case/fixture/protocol binding and requires "
            "independent re-review"
        )
    if payload["packet_digest"] != expected_packet["packet_digest"]:
        raise HumanReviewError(
            "ledger record has a stale packet digest and requires independent re-review"
        )
    response_container = _strict_object(
        payload["response"],
        {
            "canonical_response",
            "response_digest",
            "raw_response_sha256",
            "independent_annotation_digest",
        },
        "ledger response",
    )
    response = response_container["canonical_response"]
    if response_container["response_digest"] != canonical_sha256(response):
        raise HumanReviewError("ledger response digest does not match")
    _require_digest(response_container["raw_response_sha256"], "raw response digest")
    independent = {
        "intent_truth": response.get("intent_truth"),
        "clarification_decision": response.get("clarification_decision"),
        "review_truth": response.get("review_truth"),
    }
    if response_container["independent_annotation_digest"] != canonical_sha256(independent):
        raise HumanReviewError("independent annotation digest does not match")
    try:
        IntentTruth.from_dict(response["intent_truth"])
        ReviewTruth.from_dict(response["review_truth"])
    except Exception as exc:
        raise HumanReviewError("ledger truth no longer hydrates") from exc
    batch_manifest = payload["batch_manifest"]
    _validate_batch(batch_manifest)
    if batch_manifest["annotation_protocol"] != expected_packet["binding"]["annotation_protocol"]:
        raise HumanReviewError("ledger batch protocol differs from the source-bound packet")
    expected_reference = {
        "task_id": case.task_id,
        "case_version": case.case_version,
        "packet_digest": expected_packet["packet_digest"],
        "canonical_case_digest": expected_packet["binding"]["canonical_case_digest"],
    }
    references = [
        item for item in batch_manifest["packets"] if item["task_id"] == case.task_id
    ]
    if references != [expected_reference]:
        raise HumanReviewError(
            "ledger batch does not contain the exact source-bound packet reference"
        )
    hydrated_intent, hydrated_clarification, hydrated_review = _validate_response(
        response, expected_packet, batch_manifest
    )
    if (
        hydrated_intent != response["intent_truth"]
        or hydrated_clarification != response["clarification_decision"]
        or hydrated_review != response["review_truth"]
    ):
        raise HumanReviewError("ledger response is not canonical after truth hydration")
    comparison = compare_independent_truth(case, response)
    if payload["comparison"] != comparison:
        raise HumanReviewError("ledger comparison differs from deterministic current comparison")
    identities = _strict_object(
        payload["identities"], {"author_id", "reviewer_id", "adjudicator_id"}, "ledger identities"
    )
    author_id = _human_identity(identities["author_id"], "ledger author_id")
    reviewer_id = _human_identity(identities["reviewer_id"], "ledger reviewer_id")
    if author_id == reviewer_id:
        raise HumanReviewError("ledger Author/Reviewer identity collision")
    if response["reviewer"]["reviewer_id"] != reviewer_id:
        raise HumanReviewError("ledger reviewer identity differs from response")
    has_disagreement = comparison["material_disagreement"]
    if has_disagreement:
        if payload["adjudication"] is None:
            raise HumanReviewError("ledger disagreement lacks adjudication")
        adjudicator_id = _human_identity(identities["adjudicator_id"], "ledger adjudicator_id")
        if adjudicator_id in {author_id, reviewer_id}:
            raise HumanReviewError("ledger adjudicator identity is not independent")
    elif payload["adjudication"] is not None or identities["adjudicator_id"] is not None:
        raise HumanReviewError("ledger has an unnecessary adjudication")
    expected_status = "approved" if payload["final_decision"] == "accepted" else "rejected"
    if payload["final_decision"] not in {"accepted", "rejected"} or payload["status"] != expected_status:
        raise HumanReviewError("ledger final decision/status is inconsistent")
    if payload["leakage_review_completed"] is not True:
        raise HumanReviewError("ledger leakage review is incomplete")
    verified = VerifiedResponse(
        case=case,
        packet=expected_packet,
        batch=batch_manifest,
        response=response,
        raw_response_sha256=response_container["raw_response_sha256"],
        response_digest=response_container["response_digest"],
        independent_annotation_digest=response_container["independent_annotation_digest"],
        comparison=comparison,
    )
    adjudication = None
    if payload["adjudication"] is not None:
        adjudication = _validate_adjudication(payload["adjudication"], verified)
    approval = _validate_approval(
        payload["approval"], verified, adjudication is not None
    )
    reconstructed = _record_core(verified, approval, adjudication)
    if reconstructed != core:
        raise HumanReviewError(
            "ledger receipt fields do not match the verified response/approval/adjudication"
        )
    return payload


def load_source_bound_ledger_record(
    ledger_root: Path,
    case: EvalCase,
    repository_binding: Mapping[str, Any],
    fixture_manifest: Mapping[str, Any],
    protocol_binding: Mapping[str, str],
) -> Mapping[str, Any] | None:
    root = _absolute(ledger_root)
    _assert_safe_existing_ancestors(root, "human-review ledger")
    path = root / "records" / f"{case.task_id}.json"
    if not os.path.lexists(path):
        return None
    value, _ = _load_json_file(path, "human-review ledger record", canonical=True)
    expected_packet = make_packet(
        case, repository_binding, fixture_manifest, protocol_binding
    )
    return _hydrate_ledger_record(value, case, expected_packet)


def verify_current_case_approval(
    eval_root: Path,
    task_id: str,
    ledger_root: Path | None = None,
) -> Mapping[str, Any]:
    """Open and verify the private receipt for the current source-bound Case.

    This is the release gate.  An annotation sidecar is only a projection and is
    deliberately insufficient to satisfy this function.
    """

    eval_root = _absolute(eval_root)
    case = _case(eval_root, task_id)
    annotation = _annotation(eval_root, task_id)
    repository = _repository_binding(annotation)
    _, fixture_manifest = _replay_fixture(eval_root, case, repository)
    root = eval_root / "human-reviews" if ledger_root is None else _absolute(ledger_root)
    record = load_source_bound_ledger_record(
        root,
        case,
        repository,
        fixture_manifest,
        annotation_protocol_binding(eval_root),
    )
    if record is None:
        raise HumanReviewError("no evaluator-private human approval ledger record exists")
    if record["status"] != "approved" or record["final_decision"] != "accepted":
        raise HumanReviewError("the source-bound human approval record is not approved")
    projected = project_ledger_record(annotation, record)
    if projected != annotation:
        raise HumanReviewError(
            "annotation sidecar is not the exact projection of its source-bound ledger record"
        )
    return record


def project_ledger_record(
    annotation: Mapping[str, Any], record: Mapping[str, Any] | None
) -> dict[str, Any]:
    result = json.loads(json.dumps(annotation))
    if record is None:
        return result
    response = record["response"]["canonical_response"]
    adjudication = record["adjudication"]
    identities = record["identities"]
    comparison = record["comparison"]
    result["human_review"].update(
        {
            "status": record["status"],
            "approval_identity_status": "current_source_bound",
            "prior_approval_carried_forward": False,
            "final_decision": record["final_decision"],
            "author_id": identities["author_id"],
            "reviewer_id": identities["reviewer_id"],
            "adjudicator_id": identities["adjudicator_id"],
            "review_batch_id": record["batch_manifest"]["batch_id"],
            "blind_review_started_at": response["reviewer"]["started_at"],
            "blind_review_completed_at": response["reviewer"]["completed_at"],
            "independent_annotation_digest": record["response"]["independent_annotation_digest"],
            "adjudication_digest": None if adjudication is None else canonical_sha256(adjudication),
            "leakage_review_completed": record["leakage_review_completed"],
        }
    )
    for key in (
        "atomic_findings_reviewed",
        "evidence_anchors_are_non_exclusive",
        "known_invalid_traps_reviewed",
        "semantic_truth_leakage_reviewed",
        "severity_category_context_reviewed",
        "truth_completeness_reviewed",
        "human_review_completed",
    ):
        result["checklist"][key] = True
    result["authoring"].update(
        {
            "author_id": identities["author_id"],
            "truth_frozen_at": record["approval"]["signed_at"],
            "status": "human_reviewed_" + record["status"],
        }
    )
    result["disagreements"] = {
        "status": "none" if not comparison["material_disagreement"] else "resolved",
        "items": comparison["differences"],
        "adjudicator_id": identities["adjudicator_id"],
        "adjudication_digest": None if adjudication is None else canonical_sha256(adjudication),
    }
    return result


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export and source-bind independent human review for Core Eval Cases"
    )
    parser.add_argument("--eval-root", type=Path, default=REPOSITORY_ROOT / "eval")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="export a new blind-review batch outside the repository")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--batch-id", required=True)
    export.add_argument("--task-id", action="append", required=True)
    verify = sub.add_parser("verify-packet", help="verify a blind-review batch and fixture replay")
    verify.add_argument("--batch", type=Path, required=True)
    response = sub.add_parser("verify-response", help="verify and compare a completed human response")
    response.add_argument("--batch", type=Path, required=True)
    response.add_argument("--response", type=Path, required=True)
    ingest = sub.add_parser("import", help="write a final immutable evaluator-private approval record")
    ingest.add_argument("--batch", type=Path, required=True)
    ingest.add_argument("--response", type=Path, required=True)
    ingest.add_argument("--approval", type=Path, required=True)
    ingest.add_argument("--adjudication", type=Path)
    ingest.add_argument("--ledger-root", type=Path, default=REPOSITORY_ROOT / "eval" / "human-reviews")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    if args.command == "export":
        result = export_blind_review_batch(
            args.eval_root, args.output, args.task_id, args.batch_id
        )
    elif args.command == "verify-packet":
        result = verify_blind_review_batch(args.eval_root, args.batch)
    elif args.command == "verify-response":
        verified = verify_completed_response(args.eval_root, args.batch, args.response)
        result = verified.comparison
    else:
        result = import_approved_response(
            args.eval_root,
            args.batch,
            args.response,
            args.approval,
            args.ledger_root,
            args.adjudication,
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

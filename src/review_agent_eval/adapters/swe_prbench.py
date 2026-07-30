"""Strict, offline SWE-PRBench preparation.

The pinned upstream dataset has two materially different execution protocols:

* ``native_repository`` is published as a runnable Repository Target Suite.
* ``official_frozen_context`` is published as a runnable Frozen Context Target
  Suite backed by the verified, hash-bound bundle format.

No function in this module performs network I/O.  Callers must acquire source
bytes before preparation and bind every consumed file in a
``PublicSourceManifest``.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
import tempfile
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..cases import (
    FROZEN_CONTEXT_MATERIALIZER_PROTOCOL,
    REPOSITORY_MATERIALIZER_PROTOCOL,
    CaseDimension,
    CaseSplit,
    WireContractV2,
)
from ..datasets import _coerce_suite_root
from ..models import (
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    MAX_CLAIM_CHARS,
    MAX_COUNTER,
    MAX_IDENTIFIER_CHARS,
    CaseOrigin,
    CaseSource,
    ClarificationScript,
    EvalCase,
    EvalCaseInput,
    EvaluatorContextProvenance,
    EvaluatorContextSource,
    EvaluatorContextSourceKind,
    EvaluatorContextTask,
    ExpectedFinding,
    FrozenContextReviewTarget,
    IntentTruth,
    MetricAuthority,
    NovelFindingPolicy,
    Repository,
    RepositoryReviewTarget,
    RepositorySource,
    RequiredContextLevel,
    ReviewRequest,
    ReviewEvaluatorContext,
    ReviewTruth,
    ReviewTargetKind,
    SchemaError,
    TruthCompleteness,
    TruthLocation,
    TruthEvaluatorContext,
    _JsonModel,
    _array,
    _boolean,
    _check_model_size,
    _digest,
    _exact_fields,
    _identifier,
    _integer,
    _object,
    _optional_integer,
    _safe_repo_path,
    _strict_json_loads,
    _string,
    canonical_json_bytes,
    canonical_sha256,
    stable_id,
)
from ._public import (
    PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION,
    PUBLIC_SOURCE_MANIFEST_SCHEMA_VERSION,
    PublicDatasetError,
    PublicConflictError,
    PublicFilterManifest,
    PublicFormatError,
    PublicFrozenBundlePublication,
    PublicOptionalDependencyError,
    PublicPreparationError,
    PublicPreparationResult,
    PublicPreparedCase,
    PublicRecordReceipt,
    PublicSourceIntegrityError,
    PublicSourceManifest,
    PublicStatistic,
    VerifiedPublicSource,
    _assert_no_portable_path_collisions,
    _assert_publication_parent,
    _cleanup_owned_staging,
    _file_identity,
    _publish_directory_create_only,
    _read_single_link_regular_file,
    _write_new,
    write_public_suite,
)


SWE_PRBENCH_DATASET_ID = "swe-prbench"
SWE_PRBENCH_DATASET_VERSION = "v0.4.1"
SWE_PRBENCH_DATASET_URI = (
    "https://huggingface.co/datasets/foundry-ai/swe-prbench"
)
SWE_PRBENCH_DATASET_REVISION = (
    "b87f5797aef3ed2c3153bb1304ea4d801d36ba6e"
)
SWE_PRBENCH_DATASET_LICENSE = "CC-BY-4.0"
SWE_PRBENCH_FIXTURE_DATASET_VERSION = "v0.4.1-fixture-v1"
SWE_PRBENCH_FIXTURE_SOURCE_URI = (
    "https://huggingface.co/datasets/foundry-ai/swe-prbench/tree/"
    "b87f5797aef3ed2c3153bb1304ea4d801d36ba6e"
)
SWE_PRBENCH_FIXTURE_SOURCE_REVISION = "fixture-dask-12221-b87f5797-v1"
SWE_PRBENCH_HARNESS_REVISION = (
    "379f0bfd8978a1734cd8399e115d04d4fdceeb89"
)
SWE_PRBENCH_HARNESS_LICENSE = "MIT"
SWE_PRBENCH_PIPELINE_VERSION = "v0.4.1"
SWE_PRBENCH_PARQUET_CONVERTER_REVISION = (
    "6ef82d99355479bb4c96f4c418e242fd3f22ec53"
)

SWE_PRBENCH_PROTOCOL_NATIVE = "native_repository"
SWE_PRBENCH_PROTOCOL_FROZEN = "official_frozen_context"
SWE_PRBENCH_SOURCE_RAW = "raw_jsonl"
SWE_PRBENCH_SOURCE_PARQUET = "parquet"
SWE_PRBENCH_SOURCE_PROFILE_OFFICIAL_RAW = "official_raw_v0.4.1"
SWE_PRBENCH_SOURCE_PROFILE_FIXTURE = "fixture_dask_12221_v1"
SWE_PRBENCH_SOURCE_PROFILE_EXPLICIT = "explicit_pinned_manifest"
SWE_PRBENCH_CONTEXT_CONFIGS = ("config_A", "config_B", "config_C")
SWE_PRBENCH_DIFFICULTIES = (
    "Type1_Direct",
    "Type2_Contextual",
    "Type3_Latent_Candidate",
)
SWE_PRBENCH_LANGUAGES = (
    "Go",
    "Java",
    "JavaScript",
    "Python",
    "TypeScript",
)

SWE_PRBENCH_ADAPTER_ID = "swe-prbench-adapter"
SWE_PRBENCH_ADAPTER_VERSION = "swe-prbench-adapter-v2"
SWE_PRBENCH_NATIVE_PROTOCOL_ID = "swe-prbench-native-repository-v2"
SWE_PRBENCH_FROZEN_PROTOCOL_ID = "swe-prbench-official-frozen-context-v2"
SWE_PRBENCH_UNDERLYING_REPOSITORY_LICENSE = "not_normalized_by_upstream"
SWE_PRBENCH_NATIVE_WIRE_CONTRACT = WireContractV2(
    case_schema_version=EVAL_CASE_SCHEMA_VERSION,
    input_schema_version=EVAL_INPUT_SCHEMA_VERSION,
    submission_schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
    review_target_kind=ReviewTargetKind.REPOSITORY,
    materializer_protocol=REPOSITORY_MATERIALIZER_PROTOCOL,
)
SWE_PRBENCH_FROZEN_WIRE_CONTRACT = WireContractV2(
    case_schema_version=EVAL_CASE_SCHEMA_VERSION,
    input_schema_version=EVAL_INPUT_SCHEMA_VERSION,
    submission_schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
    review_target_kind=ReviewTargetKind.FROZEN_CONTEXT,
    materializer_protocol=FROZEN_CONTEXT_MATERIALIZER_PROTOCOL,
)
SWE_PRBENCH_FROZEN_BUNDLE_SCHEMA_VERSION = (
    "swe_prbench_frozen_context_bundle_v1"
)
SWE_PRBENCH_FROZEN_RECORD_SCHEMA_VERSION = (
    "swe_prbench_frozen_context_record_v1"
)
SWE_PRBENCH_FROZEN_ENVELOPE_SCHEMA_VERSION = (
    "swe_prbench_frozen_context_envelope_v1"
)
SWE_PRBENCH_FROZEN_MANIFEST_PATH = "frozen_bundle_manifest.json"
SWE_PRBENCH_FROZEN_SUITE_RELATIVE_ROOT = "frozen_bundle"

SWE_PRBENCH_FULL_PR_COUNT = 350
SWE_PRBENCH_FULL_CONTEXT_COUNT = 1050
SWE_PRBENCH_RAW_PRS_SIZE = 28_764_421
SWE_PRBENCH_RAW_PRS_SHA256 = (
    "a58e1f713533f6bc260a93f6e234b85acd16a77f55a756893694b96495eb43cd"
)
SWE_PRBENCH_OFFICIAL_RAW_SOURCE_MANIFEST_DIGEST = (
    "75921f6ed330d7406ea966a8915443fc666117ea259ec8e609ae1f2724b907ab"
)
SWE_PRBENCH_FIXTURE_SOURCE_MANIFEST_DIGEST = (
    "830aece95199c1b9ff2e68c546b0a1381a23cc0d931a18d9231b6780c803981f"
)

_MAX_UPSTREAM_JSON_BYTES = 32 * 1024 * 1024
_MAX_FROZEN_RECORD_BYTES = 64 * 1024 * 1024
_MAX_FROZEN_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_UPSTREAM_TEXT_CHARS = 32 * 1024 * 1024
_MAX_PARQUET_ROWS = SWE_PRBENCH_FULL_PR_COUNT
_MAX_PARQUET_ROW_GROUPS = SWE_PRBENCH_FULL_PR_COUNT
_MAX_PARQUET_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
_MAX_PARQUET_CANONICAL_BYTES = 64 * 1024 * 1024
_PARQUET_BATCH_ROWS = 16

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_MERGED_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

_PR_FIELDS = (
    "task_id",
    "repo",
    "repo_name",
    "repo_clone_url",
    "repo_url",
    "pr_number",
    "pr_url",
    "title",
    "description",
    "language",
    "pr_type",
    "difficulty",
    "severity",
    "rvs_score",
    "rvs_breakdown",
    "lines_added",
    "lines_removed",
    "files_changed",
    "changed_files",
    "merged_at",
    "base_commit",
    "head_commit",
    "num_substantive_comments",
    "num_unique_reviewers",
    "has_requested_changes",
    "ai_comments_removed",
    "human_review_comments",
    "agent_input",
    "diff_patch",
)
_RVS_FIELDS = (
    "review_depth",
    "code_complexity",
    "discussion_signal",
    "test_change_signal",
    "bug_fix_signal",
)
_EMBEDDED_COMMENT_FIELDS = (
    "author",
    "body",
    "path",
    "line",
    "diffHunk",
    "replyTo",
)
_ANNOTATION_FIELDS = (
    "task_id",
    "pr_number",
    "repo",
    "has_severity_annotations",
    "has_requested_changes",
    "total_comment_count",
    "substantive_comment_count",
    "changes_required",
    "requested_change_count",
    "substantive_comment_ids",
    "requested_changes",
    "comments",
)
_ANNOTATION_COMMENT_FIELDS = (
    "comment_id",
    "body",
    "file",
    "line",
    "diff_hunk",
    "severity",
    "is_blocking",
    "reviewer",
    "is_initiating_comment",
    "is_reply",
    "reply_to",
    "requires_change",
    "is_in_diff",
    "thread_resolved",
)
_CONTEXT_FIELDS = (
    "pr_number",
    "repo",
    "task_id",
    "config_name",
    "pipeline_version",
    "total_tokens",
    "was_truncated",
    "rendered",
)


def _parquet_field(
    name: str, type_descriptor: Mapping[str, Any], *, nullable: bool = True
) -> Dict[str, Any]:
    return {
        "name": name,
        "nullable": nullable,
        "type": dict(type_descriptor),
    }


def _parquet_struct(
    fields: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {"kind": "struct", "fields": tuple(dict(item) for item in fields)}


def _parquet_list(value_type: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "kind": "list",
        "value_field": {
            "nullable": True,
            "type": dict(value_type),
        },
    }


_PARQUET_STRING = {"kind": "string"}
_PARQUET_INT64 = {"kind": "int64"}
_PARQUET_FLOAT64 = {"kind": "float64"}
_PARQUET_BOOLEAN = {"kind": "boolean"}
_PARQUET_NULL = {"kind": "null"}
_PARQUET_TIMESTAMP_S = {"kind": "timestamp", "unit": "s", "tz": None}

_SWE_PRBENCH_PARQUET_SCHEMA_DESCRIPTOR = (
    _parquet_field("task_id", _PARQUET_STRING),
    _parquet_field("repo", _PARQUET_STRING),
    _parquet_field("repo_name", _PARQUET_STRING),
    _parquet_field("repo_clone_url", _PARQUET_STRING),
    _parquet_field("repo_url", _PARQUET_STRING),
    _parquet_field("pr_number", _PARQUET_INT64),
    _parquet_field("pr_url", _PARQUET_STRING),
    _parquet_field("title", _PARQUET_STRING),
    _parquet_field("description", _PARQUET_STRING),
    _parquet_field("language", _PARQUET_STRING),
    _parquet_field("pr_type", _PARQUET_STRING),
    _parquet_field("difficulty", _PARQUET_STRING),
    _parquet_field("severity", _PARQUET_NULL),
    _parquet_field("rvs_score", _PARQUET_FLOAT64),
    _parquet_field(
        "rvs_breakdown",
        _parquet_struct(
            tuple(
                _parquet_field(name, _PARQUET_FLOAT64)
                for name in _RVS_FIELDS
            )
        ),
    ),
    _parquet_field("lines_added", _PARQUET_INT64),
    _parquet_field("lines_removed", _PARQUET_INT64),
    _parquet_field("files_changed", _PARQUET_INT64),
    _parquet_field("changed_files", _parquet_list(_PARQUET_STRING)),
    _parquet_field("merged_at", _PARQUET_TIMESTAMP_S),
    _parquet_field("base_commit", _PARQUET_STRING),
    _parquet_field("head_commit", _PARQUET_STRING),
    _parquet_field("num_substantive_comments", _PARQUET_INT64),
    _parquet_field("num_unique_reviewers", _PARQUET_INT64),
    _parquet_field("has_requested_changes", _PARQUET_BOOLEAN),
    _parquet_field("ai_comments_removed", _PARQUET_INT64),
    _parquet_field(
        "human_review_comments",
        _parquet_list(
            _parquet_struct(
                (
                    _parquet_field("author", _PARQUET_STRING),
                    _parquet_field("body", _PARQUET_STRING),
                    _parquet_field("path", _PARQUET_STRING),
                    _parquet_field("line", _PARQUET_INT64),
                    _parquet_field("diffHunk", _PARQUET_STRING),
                    _parquet_field(
                        "replyTo",
                        _parquet_struct(
                            (_parquet_field("id", _PARQUET_STRING),)
                        ),
                    ),
                )
            )
        ),
    ),
    _parquet_field("agent_input", _PARQUET_NULL),
    _parquet_field("diff_patch", _PARQUET_STRING),
)
_SWE_PRBENCH_PARQUET_SCHEMA_FINGERPRINT = canonical_sha256(
    _SWE_PRBENCH_PARQUET_SCHEMA_DESCRIPTOR
)
_EXPECTED_STATISTIC_NAMES = frozenset(
    {
        "pr_records",
        "annotations",
        "contexts",
        "context_config_a",
        "context_config_b",
        "context_config_c",
    }
)
_FILTER_NAMES = frozenset(
    {
        "source_scope",
        "source_profile",
        "source_format",
        "protocol",
        "context_config",
        "harness_revision",
        "harness_license",
        "pipeline_version",
        "parquet_converter_revision",
        "ground_truth_fallback",
        "task_id",
        "language",
        "difficulty",
    }
)


@dataclass(frozen=True)
class FrozenContextRecord(_JsonModel):
    """The exact eight-field official frozen-context record."""

    SCHEMA_VERSION: ClassVar[str] = SWE_PRBENCH_FROZEN_RECORD_SCHEMA_VERSION

    pr_number: int
    repo: str
    task_id: str
    config_name: str
    pipeline_version: str
    total_tokens: int
    was_truncated: bool
    rendered: str

    def __post_init__(self) -> None:
        _integer(self.pr_number, "frozen context.pr_number", minimum=1)
        _string(self.repo, "frozen context.repo", MAX_IDENTIFIER_CHARS)
        _identifier(self.task_id, "frozen context.task_id")
        if self.config_name not in SWE_PRBENCH_CONTEXT_CONFIGS:
            raise PublicFormatError("frozen context has an unknown config_name")
        if self.pipeline_version != SWE_PRBENCH_PIPELINE_VERSION:
            raise PublicFormatError("frozen context has an unknown pipeline_version")
        _integer(self.total_tokens, "frozen context.total_tokens", minimum=1)
        _boolean(self.was_truncated, "frozen context.was_truncated")
        _upstream_text(self.rendered, "frozen context.rendered")
        if not self.rendered:
            raise PublicFormatError("frozen context.rendered may not be empty")
        _check_model_size(self, _MAX_FROZEN_RECORD_BYTES, "frozen context record")

    @classmethod
    def from_upstream_dict(cls, value: Any, context: str) -> "FrozenContextRecord":
        payload = _format_object(value, context)
        _format_exact_fields(payload, _CONTEXT_FIELDS, context)
        return cls(
            pr_number=_format_integer(payload["pr_number"], context + ".pr_number", 1),
            repo=_upstream_text(payload["repo"], context + ".repo"),
            task_id=_format_identifier(payload["task_id"], context + ".task_id"),
            config_name=_upstream_text(
                payload["config_name"], context + ".config_name"
            ),
            pipeline_version=_upstream_text(
                payload["pipeline_version"], context + ".pipeline_version"
            ),
            total_tokens=_format_integer(
                payload["total_tokens"], context + ".total_tokens", 1
            ),
            was_truncated=_format_boolean(
                payload["was_truncated"], context + ".was_truncated"
            ),
            rendered=_upstream_text(payload["rendered"], context + ".rendered"),
        )

    @classmethod
    def from_dict(cls, value: Any) -> "FrozenContextRecord":
        payload = _format_object(value, "frozen context bundle record")
        _format_exact_fields(
            payload,
            ("schema_version",) + _CONTEXT_FIELDS,
            "frozen context bundle record",
        )
        if payload["schema_version"] != cls.SCHEMA_VERSION:
            raise PublicFormatError("frozen context record has an unknown schema_version")
        upstream = dict(payload)
        upstream.pop("schema_version")
        return cls.from_upstream_dict(upstream, "frozen context bundle record")

    @classmethod
    def from_json(cls, raw: Any) -> "FrozenContextRecord":
        return cls.from_dict(
            _format_json(raw, _MAX_FROZEN_RECORD_BYTES, "frozen context record JSON")
        )

    @property
    def rendered_sha256(self) -> str:
        return hashlib.sha256(self.rendered.encode("utf-8", "strict")).hexdigest()

    @property
    def rendered_utf8_bytes(self) -> int:
        return len(self.rendered.encode("utf-8", "strict"))

    def to_upstream_dict(self) -> Dict[str, Any]:
        return {
            "pr_number": self.pr_number,
            "repo": self.repo,
            "task_id": self.task_id,
            "config_name": self.config_name,
            "pipeline_version": self.pipeline_version,
            "total_tokens": self.total_tokens,
            "was_truncated": self.was_truncated,
            "rendered": self.rendered,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.SCHEMA_VERSION, **self.to_upstream_dict()}


@dataclass(frozen=True)
class FrozenContextEnvelope(_JsonModel):
    """One bundle record that points back to its immutable bundle identity."""

    SCHEMA_VERSION: ClassVar[str] = SWE_PRBENCH_FROZEN_ENVELOPE_SCHEMA_VERSION

    schema_version: str
    bundle_id: str
    record: FrozenContextRecord

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise PublicFormatError("frozen context envelope has an unknown schema_version")
        _identifier(self.bundle_id, "frozen context envelope.bundle_id")
        if not isinstance(self.record, FrozenContextRecord):
            raise PublicFormatError("frozen context envelope.record is invalid")
        _check_model_size(self, _MAX_FROZEN_RECORD_BYTES, "frozen context envelope")

    @classmethod
    def from_dict(cls, value: Any) -> "FrozenContextEnvelope":
        payload = _format_object(value, "frozen context envelope")
        _format_exact_fields(
            payload,
            ("schema_version", "bundle_id", "record"),
            "frozen context envelope",
        )
        return cls(
            schema_version=payload["schema_version"],
            bundle_id=_format_identifier(
                payload["bundle_id"], "frozen context envelope.bundle_id"
            ),
            record=FrozenContextRecord.from_dict(payload["record"]),
        )

    @classmethod
    def from_json(cls, raw: Any) -> "FrozenContextEnvelope":
        return cls.from_dict(
            _format_json(raw, _MAX_FROZEN_RECORD_BYTES, "frozen context envelope JSON")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "record": self.record.to_dict(),
        }


@dataclass(frozen=True)
class FrozenContextBinding(_JsonModel):
    task_id: str
    config_name: str
    path: str
    size_bytes: int
    sha256: str
    record_digest: str
    rendered_sha256: str
    rendered_utf8_bytes: int
    source_role: str
    source_file_sha256: str
    pr_record_sha256: str
    annotation_record_sha256: str
    review_truth_status: str
    review_truth_reason: Optional[str]
    offending_record_sha256: Optional[str]

    def __post_init__(self) -> None:
        _identifier(self.task_id, "frozen binding.task_id")
        if self.config_name not in SWE_PRBENCH_CONTEXT_CONFIGS:
            raise PublicFormatError("frozen binding has an unknown config_name")
        _safe_repo_path(self.path, "frozen binding.path")
        _integer(self.size_bytes, "frozen binding.size_bytes", minimum=1)
        _digest(self.sha256, "frozen binding.sha256")
        _digest(self.record_digest, "frozen binding.record_digest")
        _digest(self.rendered_sha256, "frozen binding.rendered_sha256")
        _integer(
            self.rendered_utf8_bytes,
            "frozen binding.rendered_utf8_bytes",
            minimum=1,
        )
        _identifier(self.source_role, "frozen binding.source_role")
        _digest(self.source_file_sha256, "frozen binding.source_file_sha256")
        _digest(self.pr_record_sha256, "frozen binding.pr_record_sha256")
        _digest(
            self.annotation_record_sha256,
            "frozen binding.annotation_record_sha256",
        )
        if self.review_truth_status not in {
            "representable",
            "empty_ground_truth",
            "claim_exceeds_claim_limit",
        }:
            raise PublicFormatError("frozen binding has an unknown review_truth_status")
        if self.review_truth_status == "representable":
            if self.review_truth_reason is not None or self.offending_record_sha256 is not None:
                raise PublicFormatError("representable frozen binding may not claim an isolation")
        else:
            _string(
                self.review_truth_reason,
                "frozen binding.review_truth_reason",
                8192,
            )
            _digest(
                self.offending_record_sha256,
                "frozen binding.offending_record_sha256",
            )

    @classmethod
    def from_dict(cls, value: Any) -> "FrozenContextBinding":
        payload = _format_object(value, "frozen context binding")
        fields = (
            "task_id",
            "config_name",
            "path",
            "size_bytes",
            "sha256",
            "record_digest",
            "rendered_sha256",
            "rendered_utf8_bytes",
            "source_role",
            "source_file_sha256",
            "pr_record_sha256",
            "annotation_record_sha256",
            "review_truth_status",
            "review_truth_reason",
            "offending_record_sha256",
        )
        _format_exact_fields(payload, fields, "frozen context binding")
        return cls(
            task_id=_format_identifier(payload["task_id"], "frozen binding.task_id"),
            config_name=_upstream_text(
                payload["config_name"], "frozen binding.config_name"
            ),
            path=_upstream_text(payload["path"], "frozen binding.path"),
            size_bytes=_format_integer(
                payload["size_bytes"], "frozen binding.size_bytes", 1
            ),
            sha256=_format_digest(payload["sha256"], "frozen binding.sha256"),
            record_digest=_format_digest(
                payload["record_digest"], "frozen binding.record_digest"
            ),
            rendered_sha256=_format_digest(
                payload["rendered_sha256"], "frozen binding.rendered_sha256"
            ),
            rendered_utf8_bytes=_format_integer(
                payload["rendered_utf8_bytes"],
                "frozen binding.rendered_utf8_bytes",
                1,
            ),
            source_role=_format_identifier(
                payload["source_role"], "frozen binding.source_role"
            ),
            source_file_sha256=_format_digest(
                payload["source_file_sha256"],
                "frozen binding.source_file_sha256",
            ),
            pr_record_sha256=_format_digest(
                payload["pr_record_sha256"],
                "frozen binding.pr_record_sha256",
            ),
            annotation_record_sha256=_format_digest(
                payload["annotation_record_sha256"],
                "frozen binding.annotation_record_sha256",
            ),
            review_truth_status=_upstream_text(
                payload["review_truth_status"],
                "frozen binding.review_truth_status",
            ),
            review_truth_reason=(
                None
                if payload["review_truth_reason"] is None
                else _upstream_text(
                    payload["review_truth_reason"],
                    "frozen binding.review_truth_reason",
                )
            ),
            offending_record_sha256=(
                None
                if payload["offending_record_sha256"] is None
                else _format_digest(
                    payload["offending_record_sha256"],
                    "frozen binding.offending_record_sha256",
                )
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "config_name": self.config_name,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "record_digest": self.record_digest,
            "rendered_sha256": self.rendered_sha256,
            "rendered_utf8_bytes": self.rendered_utf8_bytes,
            "source_role": self.source_role,
            "source_file_sha256": self.source_file_sha256,
            "pr_record_sha256": self.pr_record_sha256,
            "annotation_record_sha256": self.annotation_record_sha256,
            "review_truth_status": self.review_truth_status,
            "review_truth_reason": self.review_truth_reason,
            "offending_record_sha256": self.offending_record_sha256,
        }


def _frozen_bundle_identity_records(
    records: Sequence[FrozenContextBinding],
) -> List[Dict[str, Any]]:
    """Return the non-circular identity each envelope and manifest share."""

    return [
        {
            "task_id": item.task_id,
            "config_name": item.config_name,
            "path": item.path,
            "record_digest": item.record_digest,
            "rendered_sha256": item.rendered_sha256,
            "rendered_utf8_bytes": item.rendered_utf8_bytes,
            "source_role": item.source_role,
            "source_file_sha256": item.source_file_sha256,
            "pr_record_sha256": item.pr_record_sha256,
            "annotation_record_sha256": item.annotation_record_sha256,
            "review_truth_status": item.review_truth_status,
            "review_truth_reason": item.review_truth_reason,
            "offending_record_sha256": item.offending_record_sha256,
        }
        for item in sorted(records, key=lambda value: (value.task_id, value.config_name))
    ]


def _compute_frozen_bundle_id(
    *,
    adapter_id: str,
    adapter_version: str,
    harness_revision: str,
    harness_license: str,
    dataset_license: str,
    underlying_repository_license: str,
    source_manifest_digest: str,
    filter_manifest_digest: str,
    identity_records: Sequence[Mapping[str, Any]],
) -> str:
    """Bind every evaluator-relevant frozen-bundle identity component."""

    ordered_records = sorted(
        (dict(item) for item in identity_records),
        key=lambda item: (item["task_id"], item["config_name"]),
    )
    return stable_id(
        "swe-frozen-bundle",
        {
            "adapter_id": adapter_id,
            "adapter_version": adapter_version,
            "harness_revision": harness_revision,
            "harness_license": harness_license,
            "dataset_license": dataset_license,
            "underlying_repository_license": underlying_repository_license,
            "source_manifest_digest": source_manifest_digest,
            "filter_manifest_digest": filter_manifest_digest,
            "records": ordered_records,
        },
    )


@dataclass(frozen=True)
class FrozenContextBundleManifest(_JsonModel):
    SCHEMA_VERSION: ClassVar[str] = SWE_PRBENCH_FROZEN_BUNDLE_SCHEMA_VERSION

    schema_version: str
    bundle_id: str
    adapter_id: str
    adapter_version: str
    harness_revision: str
    harness_license: str
    dataset_license: str
    underlying_repository_license: str
    source_manifest: PublicSourceManifest
    source_manifest_digest: str
    filter_manifest: PublicFilterManifest
    filter_manifest_digest: str
    records: Tuple[FrozenContextBinding, ...]

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise PublicFormatError("frozen bundle has an unknown schema_version")
        _identifier(self.bundle_id, "frozen bundle.bundle_id")
        _identifier(self.adapter_id, "frozen bundle.adapter_id")
        _identifier(self.adapter_version, "frozen bundle.adapter_version")
        if self.adapter_id != SWE_PRBENCH_ADAPTER_ID:
            raise PublicFormatError("frozen bundle adapter ID drifted")
        if self.adapter_version != SWE_PRBENCH_ADAPTER_VERSION:
            raise PublicFormatError("frozen bundle adapter version drifted")
        if self.harness_revision != SWE_PRBENCH_HARNESS_REVISION:
            raise PublicFormatError("frozen bundle harness revision drifted")
        if self.harness_license != SWE_PRBENCH_HARNESS_LICENSE:
            raise PublicFormatError("frozen bundle harness license drifted")
        if self.dataset_license != SWE_PRBENCH_DATASET_LICENSE:
            raise PublicFormatError("frozen bundle dataset license drifted")
        if (
            self.underlying_repository_license
            != SWE_PRBENCH_UNDERLYING_REPOSITORY_LICENSE
        ):
            raise PublicFormatError("frozen bundle overstates repository licensing")
        if not isinstance(self.source_manifest, PublicSourceManifest):
            raise PublicFormatError("frozen bundle source manifest is invalid")
        if not isinstance(self.filter_manifest, PublicFilterManifest):
            raise PublicFormatError("frozen bundle filter manifest is invalid")
        if self.source_manifest.digest() != self.source_manifest_digest:
            raise PublicSourceIntegrityError("frozen bundle source digest drifted")
        if self.filter_manifest.digest() != self.filter_manifest_digest:
            raise PublicSourceIntegrityError("frozen bundle filter digest drifted")
        records = tuple(self.records)
        if not records or any(not isinstance(item, FrozenContextBinding) for item in records):
            raise PublicFormatError("frozen bundle records are invalid")
        identities = [(item.task_id, item.config_name) for item in records]
        paths = [item.path for item in records]
        if len(identities) != len(set(identities)) or len(paths) != len(set(paths)):
            raise PublicFormatError("frozen bundle contains duplicate records")
        try:
            _assert_no_portable_path_collisions(
                paths,
                "frozen bundle record paths",
                reserved=(SWE_PRBENCH_FROZEN_MANIFEST_PATH,),
            )
        except SchemaError as exc:
            raise PublicFormatError(str(exc)) from exc
        object.__setattr__(
            self,
            "records",
            tuple(sorted(records, key=lambda x: (x.task_id, x.config_name))),
        )
        source_files = {item.role: item.sha256 for item in self.source_manifest.files}
        for item in self.records:
            expected_role = "context.%s.%s" % (item.config_name, item.task_id)
            if (
                item.source_role != expected_role
                or source_files.get(item.source_role) != item.source_file_sha256
            ):
                raise PublicSourceIntegrityError(
                    "frozen binding source manifest role/hash back-reference drifted"
                )
        expected_id = _compute_frozen_bundle_id(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            harness_revision=self.harness_revision,
            harness_license=self.harness_license,
            dataset_license=self.dataset_license,
            underlying_repository_license=self.underlying_repository_license,
            source_manifest_digest=self.source_manifest_digest,
            filter_manifest_digest=self.filter_manifest_digest,
            identity_records=_frozen_bundle_identity_records(self.records),
        )
        if self.bundle_id != expected_id:
            raise PublicSourceIntegrityError("frozen bundle ID does not bind its records")
        _check_model_size(self, _MAX_FROZEN_MANIFEST_BYTES, "frozen context bundle")

    @classmethod
    def from_dict(cls, value: Any) -> "FrozenContextBundleManifest":
        payload = _format_object(value, "frozen bundle manifest")
        fields = (
            "schema_version",
            "bundle_id",
            "adapter_id",
            "adapter_version",
            "harness_revision",
            "harness_license",
            "dataset_license",
            "underlying_repository_license",
            "source_manifest",
            "source_manifest_digest",
            "filter_manifest",
            "filter_manifest_digest",
            "records",
        )
        _format_exact_fields(payload, fields, "frozen bundle manifest")
        raw_records = _format_array(payload["records"], "frozen bundle.records")
        return cls(
            schema_version=payload["schema_version"],
            bundle_id=_format_identifier(payload["bundle_id"], "frozen bundle.bundle_id"),
            adapter_id=_format_identifier(payload["adapter_id"], "frozen bundle.adapter_id"),
            adapter_version=_format_identifier(
                payload["adapter_version"], "frozen bundle.adapter_version"
            ),
            harness_revision=_format_identifier(
                payload["harness_revision"], "frozen bundle.harness_revision"
            ),
            harness_license=_upstream_text(
                payload["harness_license"], "frozen bundle.harness_license"
            ),
            dataset_license=_upstream_text(
                payload["dataset_license"], "frozen bundle.dataset_license"
            ),
            underlying_repository_license=_upstream_text(
                payload["underlying_repository_license"],
                "frozen bundle.underlying_repository_license",
            ),
            source_manifest=PublicSourceManifest.from_dict(payload["source_manifest"]),
            source_manifest_digest=_format_digest(
                payload["source_manifest_digest"],
                "frozen bundle.source_manifest_digest",
            ),
            filter_manifest=PublicFilterManifest.from_dict(payload["filter_manifest"]),
            filter_manifest_digest=_format_digest(
                payload["filter_manifest_digest"],
                "frozen bundle.filter_manifest_digest",
            ),
            records=tuple(FrozenContextBinding.from_dict(item) for item in raw_records),
        )

    @classmethod
    def from_json(cls, raw: Any) -> "FrozenContextBundleManifest":
        return cls.from_dict(
            _format_json(raw, _MAX_FROZEN_MANIFEST_BYTES, "frozen bundle manifest JSON")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "harness_revision": self.harness_revision,
            "harness_license": self.harness_license,
            "dataset_license": self.dataset_license,
            "underlying_repository_license": self.underlying_repository_license,
            "source_manifest": self.source_manifest.to_dict(),
            "source_manifest_digest": self.source_manifest_digest,
            "filter_manifest": self.filter_manifest.to_dict(),
            "filter_manifest_digest": self.filter_manifest_digest,
            "records": [item.to_dict() for item in self.records],
        }


@dataclass(frozen=True)
class PreparedFrozenContextBundle:
    root: Path
    manifest: FrozenContextBundleManifest


@dataclass(frozen=True)
class SWEPRBenchSourceValidation:
    source_manifest_digest: str
    pr_count: int
    annotation_count: int
    context_count: int
    task_ids: Tuple[str, ...]
    frozen_contexts: Tuple[FrozenContextRecord, ...]
    empty_ground_truth_task_ids: Tuple[str, ...]
    oversized_claim_task_ids: Tuple[str, ...]


@dataclass(frozen=True)
class _BoundRecord:
    value: Mapping[str, Any]
    source_role: str
    pointer: str
    source_sha256: str
    record_sha256: str
    record_size_bytes: int


@dataclass(frozen=True)
class _BoundContext:
    record: FrozenContextRecord
    source_role: str
    pointer: str
    source_sha256: str


@dataclass(frozen=True)
class _ParsedDataset:
    source: VerifiedPublicSource
    prs: Tuple[_BoundRecord, ...]
    annotations: Mapping[str, _BoundRecord]
    contexts: Mapping[Tuple[str, str], _BoundContext]


@dataclass(frozen=True)
class _Filter:
    source_scope: str
    source_profile: str
    source_format: str
    protocol: str
    context_config: str
    fallback: Optional[str]
    task_ids: Tuple[str, ...]
    languages: Tuple[str, ...]
    difficulties: Tuple[str, ...]


@dataclass(frozen=True)
class _TruthSelection:
    comments: Tuple[Mapping[str, Any], ...]
    fallback_used: bool
    status: str
    reason: Optional[str]
    offending_record_sha256: Optional[str]


def _raise_format(context: str, exc: BaseException) -> PublicFormatError:
    if isinstance(exc, PublicFormatError):
        return exc
    return PublicFormatError("%s: %s" % (context, exc))


def _format_object(value: Any, context: str) -> Dict[str, Any]:
    try:
        return _object(value, context)
    except SchemaError as exc:
        raise _raise_format(context, exc) from exc


def _format_array(value: Any, context: str, maximum: int = 262_144) -> List[Any]:
    try:
        return _array(value, context, maximum)
    except SchemaError as exc:
        raise _raise_format(context, exc) from exc


def _format_exact_fields(value: Mapping[str, Any], fields: Sequence[str], context: str) -> None:
    try:
        _exact_fields(value, fields, context)
    except SchemaError as exc:
        raise _raise_format(context, exc) from exc


def _format_json(raw: Any, maximum: int, context: str) -> Any:
    try:
        return _strict_json_loads(raw, maximum, context)
    except SchemaError as exc:
        raise _raise_format(context, exc) from exc


def _upstream_text(value: Any, context: str, *, allow_empty: bool = False) -> str:
    try:
        return _string(
            value,
            context,
            _MAX_UPSTREAM_TEXT_CHARS,
            allow_empty=allow_empty,
        )
    except SchemaError as exc:
        raise _raise_format(context, exc) from exc


def _format_identifier(value: Any, context: str) -> str:
    try:
        return _identifier(value, context)
    except SchemaError as exc:
        raise _raise_format(context, exc) from exc


def _format_integer(value: Any, context: str, minimum: int = 0) -> int:
    try:
        return _integer(value, context, minimum=minimum, maximum=MAX_COUNTER)
    except SchemaError as exc:
        raise _raise_format(context, exc) from exc


def _format_optional_line(value: Any, context: str) -> Optional[int]:
    try:
        return _optional_integer(value, context, minimum=1)
    except SchemaError as exc:
        raise _raise_format(context, exc) from exc


def _format_boolean(value: Any, context: str) -> bool:
    try:
        return _boolean(value, context)
    except SchemaError as exc:
        raise _raise_format(context, exc) from exc


def _format_digest(value: Any, context: str) -> str:
    try:
        return _digest(value, context)
    except SchemaError as exc:
        raise _raise_format(context, exc) from exc


def _number(value: Any, context: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise PublicFormatError(context + " must be a finite JSON number")
    return float(value)


def _safe_path(value: Any, context: str) -> str:
    try:
        return _safe_repo_path(value, context)
    except SchemaError as exc:
        raise _raise_format(context, exc) from exc


def _validate_source_manifest(
    manifest: PublicSourceManifest,
    parsed_filter: _Filter,
    expected_source_manifest_digest: Optional[str],
) -> None:
    if not isinstance(manifest, PublicSourceManifest):
        raise PublicFormatError("SWE-PRBench requires a PublicSourceManifest")
    if parsed_filter.source_scope == "fixture":
        expected = (
            PUBLIC_SOURCE_MANIFEST_SCHEMA_VERSION,
            SWE_PRBENCH_DATASET_ID,
            SWE_PRBENCH_FIXTURE_DATASET_VERSION,
            SWE_PRBENCH_FIXTURE_SOURCE_URI,
            SWE_PRBENCH_FIXTURE_SOURCE_REVISION,
            SWE_PRBENCH_DATASET_LICENSE,
        )
    else:
        expected = (
            PUBLIC_SOURCE_MANIFEST_SCHEMA_VERSION,
            SWE_PRBENCH_DATASET_ID,
            SWE_PRBENCH_DATASET_VERSION,
            SWE_PRBENCH_DATASET_URI,
            SWE_PRBENCH_DATASET_REVISION,
            SWE_PRBENCH_DATASET_LICENSE,
        )
    if parsed_filter.source_profile == SWE_PRBENCH_SOURCE_PROFILE_FIXTURE:
        trusted_digest = SWE_PRBENCH_FIXTURE_SOURCE_MANIFEST_DIGEST
    elif parsed_filter.source_profile == SWE_PRBENCH_SOURCE_PROFILE_OFFICIAL_RAW:
        trusted_digest = SWE_PRBENCH_OFFICIAL_RAW_SOURCE_MANIFEST_DIGEST
    else:
        trusted_digest = expected_source_manifest_digest
    actual = (
        manifest.schema_version,
        manifest.dataset_id,
        manifest.dataset_version,
        manifest.source_uri,
        manifest.source_revision,
        manifest.license,
    )
    if actual != expected:
        raise PublicSourceIntegrityError(
            "SWE-PRBench source must bind the pinned dataset revision and CC-BY-4.0 dataset license"
        )
    if trusted_digest is None:
        raise PublicSourceIntegrityError(
            "explicit SWE-PRBench source profile requires expected_source_manifest_digest from an independent trust anchor"
        )
    try:
        trusted_digest = _digest(
            trusted_digest, "expected SWE-PRBench source manifest digest"
        )
    except SchemaError as exc:
        raise PublicSourceIntegrityError(str(exc)) from exc
    if manifest.digest() != trusted_digest:
        raise PublicSourceIntegrityError(
            "SWE-PRBench source manifest does not match its trusted catalog/profile digest"
        )
    names = {item.name for item in manifest.expected_statistics}
    if names != _EXPECTED_STATISTIC_NAMES:
        raise PublicFormatError(
            "SWE-PRBench expected_statistics must declare the exact parser census"
        )
    roles = [item.role.casefold() for item in manifest.files]
    paths = [item.path.casefold() for item in manifest.files]
    if len(roles) != len(set(roles)) or len(paths) != len(set(paths)):
        raise PublicFormatError("SWE-PRBench source roles/paths are ambiguous by case")


def _one_selector(
    manifest: PublicFilterManifest,
    name: str,
    *,
    allowed: Optional[Iterable[str]] = None,
) -> str:
    values = manifest.values(name)
    if len(values) != 1:
        raise PublicFormatError("SWE-PRBench filter %s must contain exactly one value" % name)
    result = values[0]
    if allowed is not None and result not in frozenset(allowed):
        raise PublicFormatError("SWE-PRBench filter %s has an unsupported value" % name)
    return result


def _parse_filter(manifest: PublicFilterManifest, protocol: Optional[str] = None) -> _Filter:
    if not isinstance(manifest, PublicFilterManifest):
        raise PublicFormatError("SWE-PRBench requires a PublicFilterManifest")
    if (
        manifest.schema_version != PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION
        or manifest.dataset_id != SWE_PRBENCH_DATASET_ID
    ):
        raise PublicFormatError("SWE-PRBench filter manifest identity drifted")
    names = {item.name for item in manifest.selectors}
    unknown = names.difference(_FILTER_NAMES)
    if unknown:
        raise PublicFormatError(
            "SWE-PRBench filter contains unknown selectors: %s"
            % ", ".join(sorted(unknown))
        )
    scope = _one_selector(manifest, "source_scope", allowed=("full", "fixture"))
    source_profile = _one_selector(
        manifest,
        "source_profile",
        allowed=(
            SWE_PRBENCH_SOURCE_PROFILE_OFFICIAL_RAW,
            SWE_PRBENCH_SOURCE_PROFILE_FIXTURE,
            SWE_PRBENCH_SOURCE_PROFILE_EXPLICIT,
        ),
    )
    source_format = _one_selector(
        manifest,
        "source_format",
        allowed=(SWE_PRBENCH_SOURCE_RAW, SWE_PRBENCH_SOURCE_PARQUET),
    )
    if source_profile == SWE_PRBENCH_SOURCE_PROFILE_OFFICIAL_RAW and (
        scope != "full" or source_format != SWE_PRBENCH_SOURCE_RAW
    ):
        raise PublicFormatError(
            "official_raw_v0.4.1 requires source_scope=full and source_format=raw_jsonl"
        )
    if source_profile == SWE_PRBENCH_SOURCE_PROFILE_FIXTURE and (
        scope != "fixture" or source_format != SWE_PRBENCH_SOURCE_RAW
    ):
        raise PublicFormatError(
            "fixture_dask_12221_v1 requires source_scope=fixture and source_format=raw_jsonl"
        )
    selected_protocol = _one_selector(
        manifest,
        "protocol",
        allowed=(SWE_PRBENCH_PROTOCOL_NATIVE, SWE_PRBENCH_PROTOCOL_FROZEN),
    )
    if protocol is not None and protocol != selected_protocol:
        raise PublicFormatError(
            "SWE-PRBench protocol argument and filter manifest disagree; mixed protocols are forbidden"
        )
    context_config = _one_selector(
        manifest,
        "context_config",
        allowed=("none",) + SWE_PRBENCH_CONTEXT_CONFIGS,
    )
    if selected_protocol == SWE_PRBENCH_PROTOCOL_NATIVE and context_config != "none":
        raise PublicFormatError("native_repository must use context_config=none")
    if selected_protocol == SWE_PRBENCH_PROTOCOL_FROZEN and context_config == "none":
        raise PublicFormatError("official_frozen_context requires one A/B/C config")
    if _one_selector(manifest, "harness_revision") != SWE_PRBENCH_HARNESS_REVISION:
        raise PublicSourceIntegrityError("SWE-PRBench harness revision is not pinned")
    if _one_selector(manifest, "harness_license") != SWE_PRBENCH_HARNESS_LICENSE:
        raise PublicSourceIntegrityError("SWE-PRBench harness MIT license binding drifted")
    if _one_selector(manifest, "pipeline_version") != SWE_PRBENCH_PIPELINE_VERSION:
        raise PublicSourceIntegrityError("SWE-PRBench pipeline version is not pinned")
    converter = manifest.values("parquet_converter_revision")
    if source_format == SWE_PRBENCH_SOURCE_PARQUET:
        if tuple(converter) != (SWE_PRBENCH_PARQUET_CONVERTER_REVISION,):
            raise PublicSourceIntegrityError(
                "Parquet input requires the pinned converter source revision"
            )
    elif converter:
        raise PublicFormatError(
            "raw JSONL input may not declare a Parquet converter revision"
        )
    fallback_values = manifest.values("ground_truth_fallback")
    if len(fallback_values) > 1 or (
        fallback_values and fallback_values[0] != "initiating_comments"
    ):
        raise PublicFormatError("unknown SWE-PRBench ground_truth_fallback policy")
    task_ids = manifest.values("task_id")
    languages = manifest.values("language")
    difficulties = manifest.values("difficulty")
    if any(item not in SWE_PRBENCH_LANGUAGES for item in languages):
        raise PublicFormatError("SWE-PRBench language filter contains an unknown value")
    if any(item not in SWE_PRBENCH_DIFFICULTIES for item in difficulties):
        raise PublicFormatError("SWE-PRBench difficulty filter contains an unknown value")
    return _Filter(
        source_scope=scope,
        source_profile=source_profile,
        source_format=source_format,
        protocol=selected_protocol,
        context_config=context_config,
        fallback=(fallback_values[0] if fallback_values else None),
        task_ids=task_ids,
        languages=languages,
        difficulties=difficulties,
    )


def _validate_embedded_comment(value: Any, context: str) -> Dict[str, Any]:
    payload = _format_object(value, context)
    _format_exact_fields(payload, _EMBEDDED_COMMENT_FIELDS, context)
    result = {
        "author": _upstream_text(payload["author"], context + ".author"),
        "body": _upstream_text(payload["body"], context + ".body"),
        "path": _safe_path(payload["path"], context + ".path"),
        "line": _format_optional_line(payload["line"], context + ".line"),
        "diffHunk": _upstream_text(
            payload["diffHunk"], context + ".diffHunk", allow_empty=True
        ),
        "replyTo": payload["replyTo"],
    }
    if result["replyTo"] is not None:
        reply = _format_object(result["replyTo"], context + ".replyTo")
        _format_exact_fields(reply, ("id",), context + ".replyTo")
        result["replyTo"] = {
            "id": _upstream_text(reply["id"], context + ".replyTo.id")
        }
    return result


def _validate_pr(value: Any, context: str) -> Dict[str, Any]:
    payload = _format_object(value, context)
    _format_exact_fields(payload, _PR_FIELDS, context)
    result: Dict[str, Any] = {}
    for name in (
        "task_id",
        "repo",
        "repo_name",
        "repo_clone_url",
        "repo_url",
        "pr_url",
        "title",
        "language",
        "pr_type",
        "difficulty",
        "merged_at",
        "base_commit",
        "head_commit",
        "diff_patch",
    ):
        result[name] = _upstream_text(payload[name], context + "." + name)
    result["task_id"] = _format_identifier(result["task_id"], context + ".task_id")
    result["description"] = _upstream_text(
        payload["description"], context + ".description", allow_empty=True
    )
    for name in (
        "pr_number",
        "files_changed",
        "num_substantive_comments",
        "num_unique_reviewers",
        "ai_comments_removed",
    ):
        result[name] = _format_integer(
            payload[name], context + "." + name, 1 if name == "pr_number" else 0
        )
    for name in ("lines_added", "lines_removed"):
        try:
            result[name] = _optional_integer(
                payload[name], context + "." + name, minimum=0
            )
        except SchemaError as exc:
            raise _raise_format(context + "." + name, exc) from exc
    result["has_requested_changes"] = _format_boolean(
        payload["has_requested_changes"], context + ".has_requested_changes"
    )
    if payload["severity"] is not None or payload["agent_input"] is not None:
        raise PublicFormatError(
            context + " severity and agent_input must remain null at the pinned revision"
        )
    result["severity"] = None
    result["agent_input"] = None
    result["rvs_score"] = _number(payload["rvs_score"], context + ".rvs_score")
    breakdown = _format_object(payload["rvs_breakdown"], context + ".rvs_breakdown")
    _format_exact_fields(breakdown, _RVS_FIELDS, context + ".rvs_breakdown")
    result["rvs_breakdown"] = {
        name: _number(breakdown[name], context + ".rvs_breakdown." + name)
        for name in _RVS_FIELDS
    }
    changed = _format_array(payload["changed_files"], context + ".changed_files")
    result["changed_files"] = [
        _safe_path(item, "%s.changed_files[%d]" % (context, index))
        for index, item in enumerate(changed)
    ]
    if len(result["changed_files"]) != len(set(result["changed_files"])):
        raise PublicFormatError(context + ".changed_files contains duplicates")
    if result["files_changed"] != len(result["changed_files"]):
        raise PublicFormatError(context + ".files_changed does not bind changed_files")
    comments = _format_array(
        payload["human_review_comments"], context + ".human_review_comments"
    )
    result["human_review_comments"] = [
        _validate_embedded_comment(item, "%s.human_review_comments[%d]" % (context, index))
        for index, item in enumerate(comments)
    ]
    if result["language"] not in SWE_PRBENCH_LANGUAGES:
        raise PublicFormatError(context + " has an unknown language")
    if result["difficulty"] not in SWE_PRBENCH_DIFFICULTIES:
        raise PublicFormatError(context + " has an unknown difficulty")
    if not _MERGED_AT_RE.fullmatch(result["merged_at"]):
        raise PublicFormatError(context + ".merged_at is not pinned RFC3339 seconds")
    base = result["base_commit"]
    head = result["head_commit"]
    if not _GIT_SHA_RE.fullmatch(base) or not _GIT_SHA_RE.fullmatch(head) or base == head:
        raise PublicFormatError(context + " requires distinct 40-character Git revisions")
    expected_task = "%s__%d" % (result["repo_name"], result["pr_number"])
    expected_repo_url = "https://github.com/" + result["repo"]
    if (
        result["task_id"] != expected_task
        or result["repo_name"] != result["repo"].rsplit("/", 1)[-1]
        or result["repo_url"] != expected_repo_url
        or result["repo_clone_url"] != expected_repo_url + ".git"
        or result["pr_url"] != expected_repo_url + "/pull/%d" % result["pr_number"]
    ):
        raise PublicFormatError(context + " repository/PR identity is inconsistent")
    return result


def _validate_annotation_comment(value: Any, context: str) -> Dict[str, Any]:
    payload = _format_object(value, context)
    _format_exact_fields(payload, _ANNOTATION_COMMENT_FIELDS, context)
    result = {
        "comment_id": _format_identifier(payload["comment_id"], context + ".comment_id"),
        "body": _upstream_text(payload["body"], context + ".body"),
        "file": _safe_path(payload["file"], context + ".file"),
        "line": _format_optional_line(payload["line"], context + ".line"),
        "diff_hunk": _upstream_text(
            payload["diff_hunk"], context + ".diff_hunk", allow_empty=True
        ),
        "reviewer": _upstream_text(payload["reviewer"], context + ".reviewer"),
    }
    for name in (
        "is_initiating_comment",
        "is_reply",
        "requires_change",
        "is_in_diff",
    ):
        result[name] = _format_boolean(payload[name], context + "." + name)
    for name in ("severity", "is_blocking", "reply_to", "thread_resolved"):
        if payload[name] is not None:
            raise PublicFormatError(
                "%s.%s must remain null at the pinned revision" % (context, name)
            )
        result[name] = None
    return result


def _validate_annotation(value: Any, context: str) -> Dict[str, Any]:
    payload = _format_object(value, context)
    _format_exact_fields(payload, _ANNOTATION_FIELDS, context)
    result: Dict[str, Any] = {
        "task_id": _format_identifier(payload["task_id"], context + ".task_id"),
        "pr_number": _format_integer(payload["pr_number"], context + ".pr_number", 1),
        "repo": _upstream_text(payload["repo"], context + ".repo"),
    }
    for name in (
        "has_severity_annotations",
        "has_requested_changes",
        "changes_required",
    ):
        result[name] = _format_boolean(payload[name], context + "." + name)
    if result["has_severity_annotations"]:
        raise PublicFormatError(
            context + ".has_severity_annotations must be false at the pinned revision"
        )
    for name in (
        "total_comment_count",
        "substantive_comment_count",
        "requested_change_count",
    ):
        result[name] = _format_integer(payload[name], context + "." + name, 0)
    ids = _format_array(
        payload["substantive_comment_ids"], context + ".substantive_comment_ids"
    )
    result["substantive_comment_ids"] = [
        _format_identifier(item, "%s.substantive_comment_ids[%d]" % (context, index))
        for index, item in enumerate(ids)
    ]
    if len(result["substantive_comment_ids"]) != len(set(result["substantive_comment_ids"])):
        raise PublicFormatError(context + ".substantive_comment_ids contains duplicates")
    if result["substantive_comment_count"] != len(result["substantive_comment_ids"]):
        raise PublicFormatError(
            context + ".substantive_comment_count does not bind substantive_comment_ids"
        )
    comments = _format_array(payload["comments"], context + ".comments")
    result["comments"] = [
        _validate_annotation_comment(item, "%s.comments[%d]" % (context, index))
        for index, item in enumerate(comments)
    ]
    comment_ids = [item["comment_id"] for item in result["comments"]]
    if len(comment_ids) != len(set(comment_ids)):
        raise PublicFormatError(context + ".comments contains duplicate comment_id values")
    comment_map = {item["comment_id"]: item for item in result["comments"]}
    missing = set(result["substantive_comment_ids"]).difference(comment_map)
    if missing:
        raise PublicFormatError(context + " references missing substantive comments")
    requested = _format_array(
        payload["requested_changes"], context + ".requested_changes"
    )
    result["requested_changes"] = [
        _validate_annotation_comment(
            item, "%s.requested_changes[%d]" % (context, index)
        )
        for index, item in enumerate(requested)
    ]
    requested_ids = [item["comment_id"] for item in result["requested_changes"]]
    if len(requested_ids) != len(set(requested_ids)):
        raise PublicFormatError(context + ".requested_changes contains duplicate IDs")
    if result["requested_change_count"] != len(result["requested_changes"]):
        raise PublicFormatError(
            context + ".requested_change_count does not bind requested_changes"
        )
    for item in result["requested_changes"]:
        if comment_map.get(item["comment_id"]) != item:
            raise PublicFormatError(
                context + ".requested_changes is not an exact subset of comments"
            )
    if result["total_comment_count"] < len(result["comments"]):
        raise PublicFormatError(context + ".total_comment_count is smaller than comments")
    return result


def _parse_raw_prs(raw: bytes, role: str, pointer: str) -> Tuple[_BoundRecord, ...]:
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise PublicFormatError("SWE-PRBench prs.jsonl must use LF and end with a newline")
    lines = raw[:-1].split(b"\n")
    if not lines or any(not line for line in lines):
        raise PublicFormatError("SWE-PRBench prs.jsonl contains an empty record")
    source_sha = hashlib.sha256(raw).hexdigest()
    result = []
    for index, line in enumerate(lines, 1):
        record = _format_json(line, _MAX_UPSTREAM_JSON_BYTES, "prs.jsonl line %d" % index)
        validated = _validate_pr(record, "prs.jsonl line %d" % index)
        result.append(
            _BoundRecord(
                value=validated,
                source_role=role,
                pointer="%s#L%d" % (pointer, index),
                source_sha256=source_sha,
                record_sha256=hashlib.sha256(line).hexdigest(),
                record_size_bytes=len(line),
            )
        )
    return tuple(result)


def _normalize_parquet_record(value: Mapping[str, Any], context: str) -> Dict[str, Any]:
    result = dict(value)
    merged = result.get("merged_at")
    if not isinstance(merged, _datetime.datetime):
        raise PublicFormatError(context + ".merged_at must be an Arrow timestamp[s]")
    if merged.microsecond:
        raise PublicFormatError(context + ".merged_at exceeds timestamp[s] precision")
    if merged.tzinfo is not None and merged.utcoffset() != _datetime.timedelta(0):
        raise PublicFormatError(context + ".merged_at must be UTC")
    result["merged_at"] = merged.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%SZ")
    return result


def _arrow_field_descriptor(pa: Any, field: Any) -> Dict[str, Any]:
    return {
        "name": field.name,
        "nullable": field.nullable,
        "type": _arrow_type_descriptor(pa, field.type),
    }


def _arrow_type_descriptor(pa: Any, value: Any) -> Dict[str, Any]:
    types = pa.types
    if types.is_string(value):
        return {"kind": "string"}
    if types.is_int64(value):
        return {"kind": "int64"}
    if types.is_float64(value):
        return {"kind": "float64"}
    if types.is_boolean(value):
        return {"kind": "boolean"}
    if types.is_null(value):
        return {"kind": "null"}
    if types.is_timestamp(value):
        return {"kind": "timestamp", "unit": value.unit, "tz": value.tz}
    if types.is_struct(value):
        return {
            "kind": "struct",
            "fields": tuple(_arrow_field_descriptor(pa, field) for field in value),
        }
    if types.is_list(value):
        return {
            "kind": "list",
            "value_field": {
                "nullable": value.value_field.nullable,
                "type": _arrow_type_descriptor(pa, value.value_type),
            },
        }
    raise PublicFormatError(
        "SWE-PRBench Parquet contains an unsupported Arrow physical type"
    )


def _arrow_schema_descriptor(pa: Any, schema: Any) -> Tuple[Dict[str, Any], ...]:
    return tuple(_arrow_field_descriptor(pa, field) for field in schema)


def _validate_parquet_schema_descriptor(
    descriptor: Sequence[Mapping[str, Any]],
) -> None:
    try:
        actual = canonical_sha256(tuple(descriptor))
    except SchemaError as exc:
        raise PublicFormatError("SWE-PRBench Parquet Arrow schema is malformed") from exc
    if (
        actual != _SWE_PRBENCH_PARQUET_SCHEMA_FINGERPRINT
        or canonical_json_bytes(tuple(descriptor))
        != canonical_json_bytes(_SWE_PRBENCH_PARQUET_SCHEMA_DESCRIPTOR)
    ):
        raise PublicFormatError(
            "SWE-PRBench Parquet Arrow schema fingerprint drifted "
            "(expected %s, got %s)"
            % (_SWE_PRBENCH_PARQUET_SCHEMA_FINGERPRINT, actual)
        )


def _parquet_metadata_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PublicFormatError(context + " must be an integer")
    return value


def _validate_parquet_metadata(metadata: Any) -> int:
    if metadata is None:
        raise PublicFormatError("SWE-PRBench Parquet metadata is missing")
    rows = _parquet_metadata_integer(
        getattr(metadata, "num_rows", None), "SWE-PRBench Parquet metadata row count"
    )
    if rows < 1 or rows > _MAX_PARQUET_ROWS:
        raise PublicFormatError("SWE-PRBench Parquet metadata row count is out of bounds")
    groups = _parquet_metadata_integer(
        getattr(metadata, "num_row_groups", None),
        "SWE-PRBench Parquet metadata row-group count",
    )
    if groups < 1 or groups > _MAX_PARQUET_ROW_GROUPS:
        raise PublicFormatError(
            "SWE-PRBench Parquet metadata row-group count is out of bounds"
        )
    grouped_rows = 0
    uncompressed_bytes = 0
    for index in range(groups):
        try:
            group = metadata.row_group(index)
        except Exception as exc:
            raise PublicFormatError(
                "SWE-PRBench Parquet row-group metadata could not be read"
            ) from exc
        group_rows = _parquet_metadata_integer(
            getattr(group, "num_rows", None),
            "SWE-PRBench Parquet row-group row count",
        )
        group_bytes = _parquet_metadata_integer(
            getattr(group, "total_byte_size", None),
            "SWE-PRBench Parquet row-group uncompressed byte size",
        )
        if group_rows < 1 or group_bytes < 1:
            raise PublicFormatError(
                "SWE-PRBench Parquet row-group metadata contains an empty group"
            )
        grouped_rows += group_rows
        uncompressed_bytes += group_bytes
        if grouped_rows > rows:
            raise PublicFormatError(
                "SWE-PRBench Parquet row-group rows exceed the metadata row count"
            )
        if uncompressed_bytes > _MAX_PARQUET_UNCOMPRESSED_BYTES:
            raise PublicFormatError(
                "SWE-PRBench Parquet exceeds the uncompressed byte budget"
            )
    if grouped_rows != rows:
        raise PublicFormatError(
            "SWE-PRBench Parquet row-group rows do not match the metadata row count"
        )
    return rows


def _parse_parquet_prs(raw: bytes, role: str, pointer: str) -> Tuple[_BoundRecord, ...]:
    try:
        import pyarrow as pa  # type: ignore
        from pyarrow import parquet as pq  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise PublicOptionalDependencyError(
            "Parquet SWE-PRBench input requires pyarrow; install review-agent[eval-public]"
        ) from exc
    try:
        parquet = pq.ParquetFile(pa.BufferReader(raw))
    except Exception as exc:
        raise PublicFormatError("SWE-PRBench Parquet could not be read: %s" % exc) from exc
    try:
        expected_rows = _validate_parquet_metadata(parquet.metadata)
        _validate_parquet_schema_descriptor(
            _arrow_schema_descriptor(pa, parquet.schema_arrow)
        )
    except (PublicFormatError, PublicSourceIntegrityError):
        raise
    except Exception as exc:
        raise PublicFormatError(
            "SWE-PRBench Parquet metadata or Arrow schema could not be read"
        ) from exc
    source_sha = hashlib.sha256(raw).hexdigest()
    result = []
    canonical_bytes = 0
    row_index = 0
    try:
        batches = parquet.iter_batches(
            batch_size=_PARQUET_BATCH_ROWS,
            use_threads=False,
        )
        for batch in batches:
            batch_rows = _parquet_metadata_integer(
                getattr(batch, "num_rows", None),
                "SWE-PRBench Parquet batch row count",
            )
            if batch_rows < 1 or batch_rows > _PARQUET_BATCH_ROWS:
                raise PublicFormatError(
                    "SWE-PRBench Parquet batch row count is out of bounds"
                )
            _validate_parquet_schema_descriptor(
                _arrow_schema_descriptor(pa, batch.schema)
            )
            rows = batch.to_pylist()
            if not isinstance(rows, list) or len(rows) != batch_rows:
                raise PublicFormatError(
                    "SWE-PRBench Parquet batch materialization count drifted"
                )
            for row in rows:
                context = "Parquet row %d" % row_index
                if not isinstance(row, Mapping):
                    raise PublicFormatError(context + " must be an object")
                normalized = _normalize_parquet_record(row, context)
                validated = _validate_pr(normalized, context)
                record_bytes = canonical_json_bytes(normalized)
                if len(record_bytes) > _MAX_UPSTREAM_JSON_BYTES:
                    raise PublicFormatError(context + " exceeds the record byte budget")
                canonical_bytes += len(record_bytes)
                if canonical_bytes > _MAX_PARQUET_CANONICAL_BYTES:
                    raise PublicFormatError(
                        "SWE-PRBench Parquet exceeds the canonical byte budget"
                    )
                result.append(
                    _BoundRecord(
                        value=validated,
                        source_role=role,
                        pointer="%s#/rows/%d" % (pointer, row_index),
                        source_sha256=source_sha,
                        record_sha256=hashlib.sha256(record_bytes).hexdigest(),
                        record_size_bytes=len(record_bytes),
                    )
                )
                row_index += 1
                if row_index > expected_rows:
                    raise PublicFormatError(
                        "SWE-PRBench Parquet yielded more rows than its metadata"
                    )
    except (PublicFormatError, PublicSourceIntegrityError):
        raise
    except Exception as exc:
        raise PublicFormatError(
            "SWE-PRBench Parquet batch stream could not be read"
        ) from exc
    if row_index != expected_rows:
        raise PublicFormatError(
            "SWE-PRBench Parquet yielded fewer rows than its metadata"
        )
    return tuple(result)


def _parse_dataset(
    source_root: os.PathLike[str] | str,
    source_manifest: PublicSourceManifest,
    parsed_filter: _Filter,
    expected_source_manifest_digest: Optional[str],
) -> _ParsedDataset:
    _validate_source_manifest(
        source_manifest, parsed_filter, expected_source_manifest_digest
    )
    source = VerifiedPublicSource.open(source_root, source_manifest)
    expected_pr_role = (
        "prs_jsonl" if parsed_filter.source_format == SWE_PRBENCH_SOURCE_RAW else "prs_parquet"
    )
    known_pr_roles = {item.role for item in source_manifest.files}.intersection(
        {"prs_jsonl", "prs_parquet"}
    )
    if known_pr_roles != {expected_pr_role}:
        raise PublicFormatError(
            "SWE-PRBench source manifest must bind exactly one explicitly selected PR format"
        )
    pr_binding = source_manifest.file(expected_pr_role)
    expected_path = (
        "dataset/prs.jsonl"
        if parsed_filter.source_format == SWE_PRBENCH_SOURCE_RAW
        else "dataset/prs.parquet"
    )
    if pr_binding.path != expected_path:
        raise PublicFormatError("SWE-PRBench PR source path is not canonical")
    raw_prs = source.read(expected_pr_role)
    if parsed_filter.source_scope == "full" and parsed_filter.source_format == SWE_PRBENCH_SOURCE_RAW:
        if (
            len(raw_prs) != SWE_PRBENCH_RAW_PRS_SIZE
            or hashlib.sha256(raw_prs).hexdigest() != SWE_PRBENCH_RAW_PRS_SHA256
        ):
            raise PublicSourceIntegrityError(
                "full raw SWE-PRBench prs.jsonl does not match the pinned LFS object"
            )
    prs = (
        _parse_raw_prs(raw_prs, expected_pr_role, pr_binding.path)
        if parsed_filter.source_format == SWE_PRBENCH_SOURCE_RAW
        else _parse_parquet_prs(raw_prs, expected_pr_role, pr_binding.path)
    )
    task_ids = [item.value["task_id"] for item in prs]
    identities = [(item.value["repo"], item.value["pr_number"]) for item in prs]
    if len(task_ids) != len(set(task_ids)) or len(identities) != len(set(identities)):
        raise PublicFormatError("SWE-PRBench PR records contain duplicate identities")
    if len({item.casefold() for item in task_ids}) != len(task_ids):
        raise PublicFormatError("SWE-PRBench task IDs are ambiguous by case")
    if parsed_filter.source_scope == "full" and len(prs) != SWE_PRBENCH_FULL_PR_COUNT:
        raise PublicFormatError("full SWE-PRBench source must contain 350 PR records")

    expected_roles = {expected_pr_role}
    for task_id in task_ids:
        expected_roles.add("annotation.%s" % task_id)
        for config in SWE_PRBENCH_CONTEXT_CONFIGS:
            expected_roles.add("context.%s.%s" % (config, task_id))
    actual_roles = {item.role for item in source_manifest.files}
    if actual_roles != expected_roles:
        missing = expected_roles.difference(actual_roles)
        extra = actual_roles.difference(expected_roles)
        raise PublicFormatError(
            "SWE-PRBench source tree is incomplete or ambiguous (missing=%d, extra=%d)"
            % (len(missing), len(extra))
        )

    pr_map = {item.value["task_id"]: item for item in prs}
    annotations: Dict[str, _BoundRecord] = {}
    contexts: Dict[Tuple[str, str], _BoundContext] = {}
    for task_id in task_ids:
        pr = pr_map[task_id].value
        ann_role = "annotation.%s" % task_id
        ann_binding = source_manifest.file(ann_role)
        ann_path = "dataset/annotations/%s_human.json" % task_id
        if ann_binding.path != ann_path:
            raise PublicFormatError("SWE-PRBench annotation role/path binding drifted")
        ann_raw = source.read(ann_role)
        ann_value = _format_json(
            ann_raw, _MAX_UPSTREAM_JSON_BYTES, "annotation %s" % task_id
        )
        ann = _validate_annotation(ann_value, "annotation %s" % task_id)
        if (
            ann["task_id"] != task_id
            or ann["repo"] != pr["repo"]
            or ann["pr_number"] != pr["pr_number"]
            or ann["has_requested_changes"] != pr["has_requested_changes"]
        ):
            raise PublicFormatError("SWE-PRBench PR/Annotation binding drifted for " + task_id)
        annotations[task_id] = _BoundRecord(
            value=ann,
            source_role=ann_role,
            pointer=ann_path + "#",
            source_sha256=hashlib.sha256(ann_raw).hexdigest(),
            record_sha256=canonical_sha256(ann),
            record_size_bytes=len(ann_raw),
        )
        for config in SWE_PRBENCH_CONTEXT_CONFIGS:
            ctx_role = "context.%s.%s" % (config, task_id)
            ctx_binding = source_manifest.file(ctx_role)
            ctx_path = "dataset/contexts/%s/%s.json" % (config, task_id)
            if ctx_binding.path != ctx_path:
                raise PublicFormatError("SWE-PRBench Context role/path binding drifted")
            ctx_raw = source.read(ctx_role)
            ctx_value = _format_json(
                ctx_raw,
                _MAX_FROZEN_RECORD_BYTES,
                "context %s/%s" % (config, task_id),
            )
            ctx = FrozenContextRecord.from_upstream_dict(
                ctx_value, "context %s/%s" % (config, task_id)
            )
            if (
                ctx.task_id != task_id
                or ctx.repo != pr["repo"]
                or ctx.pr_number != pr["pr_number"]
                or ctx.config_name != config
            ):
                raise PublicFormatError("SWE-PRBench PR/Context binding drifted for " + task_id)
            contexts[(task_id, config)] = _BoundContext(
                record=ctx,
                source_role=ctx_role,
                pointer=ctx_path + "#",
                source_sha256=hashlib.sha256(ctx_raw).hexdigest(),
            )

    expected_statistics = {
        "pr_records": len(prs),
        "annotations": len(annotations),
        "contexts": len(contexts),
        "context_config_a": len(prs),
        "context_config_b": len(prs),
        "context_config_c": len(prs),
    }
    for name, value in expected_statistics.items():
        if source_manifest.statistic(name) != value:
            raise PublicSourceIntegrityError(
                "SWE-PRBench source statistic %s does not match parsed bytes" % name
            )
    if parsed_filter.source_scope == "full" and len(contexts) != SWE_PRBENCH_FULL_CONTEXT_COUNT:
        raise PublicFormatError("full SWE-PRBench source must contain 1050 Context records")
    return _ParsedDataset(source, prs, annotations, contexts)


def _select_truth(annotation: Mapping[str, Any], fallback: Optional[str]) -> _TruthSelection:
    comments = annotation["comments"]
    comment_map = {item["comment_id"]: item for item in comments}
    ids = tuple(annotation["substantive_comment_ids"])
    fallback_used = False
    if not ids:
        if fallback != "initiating_comments":
            digest = canonical_sha256(annotation)
            return _TruthSelection(
                (),
                False,
                "empty_ground_truth",
                "substantive_comment_ids is empty and no explicit fallback was selected",
                digest,
            )
        ids = tuple(
            item["comment_id"] for item in comments if item["is_initiating_comment"]
        )
        fallback_used = True
        if not ids:
            digest = canonical_sha256(annotation)
            return _TruthSelection(
                (),
                True,
                "empty_ground_truth",
                "explicit initiating_comments fallback produced no comments",
                digest,
            )
    selected = tuple(comment_map[item] for item in ids)
    for item in selected:
        if len(item["body"]) > MAX_CLAIM_CHARS:
            digest = canonical_sha256(item)
            return _TruthSelection(
                (),
                fallback_used,
                "claim_exceeds_claim_limit",
                "comment %s has %d chars; canonical claim limit is %d"
                % (item["comment_id"], len(item["body"]), MAX_CLAIM_CHARS),
                digest,
            )
    return _TruthSelection(
        selected,
        fallback_used,
        "representable",
        None,
        None,
    )


def _selected(pr: Mapping[str, Any], parsed_filter: _Filter) -> bool:
    if parsed_filter.task_ids and pr["task_id"] not in parsed_filter.task_ids:
        return False
    if parsed_filter.languages and pr["language"] not in parsed_filter.languages:
        return False
    if parsed_filter.difficulties and pr["difficulty"] not in parsed_filter.difficulties:
        return False
    return True


def _validate_filter_targets(dataset: _ParsedDataset, parsed_filter: _Filter) -> None:
    available = {item.value["task_id"] for item in dataset.prs}
    unknown = set(parsed_filter.task_ids).difference(available)
    if unknown:
        raise PublicFormatError(
            "SWE-PRBench filter references unknown task IDs: %s"
            % ", ".join(sorted(unknown))
        )
    if not any(_selected(item.value, parsed_filter) for item in dataset.prs):
        raise PublicPreparationError("SWE-PRBench filter selected no records")


def validate_swe_prbench_source(
    source_root: os.PathLike[str] | str,
    *,
    source_manifest: PublicSourceManifest,
    filter_manifest: PublicFilterManifest,
    expected_source_manifest_digest: Optional[str] = None,
) -> SWEPRBenchSourceValidation:
    """Strictly parse all PR, Annotation, and A/B/C Context source records."""

    parsed_filter = _parse_filter(filter_manifest)
    dataset = _parse_dataset(
        source_root,
        source_manifest,
        parsed_filter,
        expected_source_manifest_digest,
    )
    _validate_filter_targets(dataset, parsed_filter)
    empty = []
    oversized = []
    for item in dataset.prs:
        task_id = item.value["task_id"]
        truth = _select_truth(dataset.annotations[task_id].value, parsed_filter.fallback)
        if truth.status == "empty_ground_truth":
            empty.append(task_id)
        elif truth.status == "claim_exceeds_claim_limit":
            oversized.append(task_id)
    return SWEPRBenchSourceValidation(
        source_manifest_digest=source_manifest.digest(),
        pr_count=len(dataset.prs),
        annotation_count=len(dataset.annotations),
        context_count=len(dataset.contexts),
        task_ids=tuple(sorted(item.value["task_id"] for item in dataset.prs)),
        frozen_contexts=tuple(
            item.record
            for _key, item in sorted(dataset.contexts.items(), key=lambda pair: pair[0])
        ),
        empty_ground_truth_task_ids=tuple(sorted(empty)),
        oversized_claim_task_ids=tuple(sorted(oversized)),
    )


def _required_context(comment: Mapping[str, Any], changed_files: Sequence[str]) -> RequiredContextLevel:
    if comment["is_in_diff"]:
        return RequiredContextLevel.DIFF
    if comment["file"] in changed_files:
        return RequiredContextLevel.FILE
    return RequiredContextLevel.REPO


def _finding(task_id: str, comment: Mapping[str, Any], changed_files: Sequence[str]) -> ExpectedFinding:
    line = comment["line"]
    location = TruthLocation(
        path=comment["file"],
        side=None,
        from_line=line,
        to_line=line,
    )
    return ExpectedFinding(
        truth_id=stable_id("swe-truth", task_id, comment["comment_id"]),
        claim=comment["body"],
        severity=None,
        category="human_review_comment",
        required=True,
        metric_authority=MetricAuthority(
            severity_scorable=False,
            severity_authority=None,
            location_scorable=False,
            location_authority=None,
        ),
        locations=(location,),
        evidence_anchors=(),
        required_context_level=_required_context(comment, changed_files),
        rationale=(
            "SWE-PRBench human-observed substantive review comment. The upstream "
            "record has no severity or location metric authority; path and line "
            "are retained only as semantic diagnostic context."
        ),
    )


def _resolve_annotation_comment(
    annotation: _BoundRecord,
    comment: Mapping[str, Any],
) -> Tuple[int, Mapping[str, Any], str]:
    comment_id = comment["comment_id"]
    matches = [
        (source_index, source_comment)
        for source_index, source_comment in enumerate(annotation.value["comments"])
        if source_comment["comment_id"] == comment_id
    ]
    if len(matches) != 1:
        raise PublicFormatError(
            "SWE-PRBench selected comment_id %s must resolve to exactly one original annotation comment"
            % comment_id
        )
    source_index, source_comment = matches[0]
    if canonical_sha256(source_comment) != canonical_sha256(comment):
        raise PublicFormatError(
            "SWE-PRBench selected comment %s drifted from its original annotation record"
            % comment_id
        )
    return (
        source_index,
        source_comment,
        stable_id(
            "swe-truth",
            annotation.value["task_id"],
            source_comment["comment_id"],
        ),
    )


def _review_evaluator_contexts(
    *,
    annotation: _BoundRecord,
    findings: Sequence[ExpectedFinding],
    comments: Sequence[Mapping[str, Any]],
) -> ReviewEvaluatorContext:
    if len(findings) != len(comments):
        raise PublicFormatError(
            "SWE-PRBench findings and selected comments must have equal length"
        )
    contexts = []
    for finding, comment in zip(findings, comments):
        source_index, source_comment, truth_id = _resolve_annotation_comment(
            annotation, comment
        )
        if finding.truth_id != truth_id:
            raise PublicFormatError(
                "SWE-PRBench Finding truth_id does not bind its original annotation comment"
            )
        content = source_comment["diff_hunk"]
        source = EvaluatorContextSource(
            kind=EvaluatorContextSourceKind.DIFF_HUNK,
            content=content,
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            provenance=EvaluatorContextProvenance(
                source_role=annotation.source_role,
                source_file_sha256=annotation.source_sha256,
                record_pointer=annotation.pointer + "/comments/%d" % source_index,
                record_sha256=canonical_sha256(source_comment),
            ),
        )
        contexts.append(
            TruthEvaluatorContext(
                truth_id=truth_id,
                allowed_tasks=(EvaluatorContextTask.FINDING_EQUIVALENCE,),
                sources=(source,),
            )
        )
    return ReviewEvaluatorContext(truth_contexts=tuple(contexts))


def _case_dimensions(
    pr: Mapping[str, Any], truth_count: int, protocol: str
) -> Tuple[CaseDimension, ...]:
    frozen = protocol == SWE_PRBENCH_PROTOCOL_FROZEN
    values = {
        "benchmark": SWE_PRBENCH_DATASET_ID,
        "language": pr["language"],
        "repository": pr["repo"],
        "upstream_declared_difficulty": pr["difficulty"],
        "difficulty_policy": "upstream_declared_not_gate_truth",
        "protocol": protocol,
        "protocol_comparability": (
            "official_frozen_context" if frozen else "native_non_official_leaderboard"
        ),
        "truth_completeness": TruthCompleteness.HUMAN_OBSERVED.value,
        "truth_finding_count": str(truth_count),
        "severity_policy": "not_scorable_no_upstream_authority",
        "category_policy": "source_record_kind_human_review_comment",
        "line_policy": "semantic_only_upstream_nullable_side_unknown",
        "diff_hunk_policy": "truth_scoped_evaluator_context_finding_equivalence_only",
        "context_level_policy": "semantic_is_in_diff_and_changed_files",
        "dataset_license": SWE_PRBENCH_DATASET_LICENSE,
        "underlying_repository_license": "not_normalized_by_upstream",
        "harness_revision": SWE_PRBENCH_HARNESS_REVISION,
        "harness_license": SWE_PRBENCH_HARNESS_LICENSE,
    }
    return tuple(CaseDimension(name=name, value=value) for name, value in values.items())


def _mapping_record(
    pr: _BoundRecord,
    annotation: _BoundRecord,
    contexts: Mapping[str, _BoundContext],
    parsed_filter: _Filter,
    truth: _TruthSelection,
    disposition: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "swe_prbench_adapter_mapping_v2",
        "task_id": pr.value["task_id"],
        "protocol": parsed_filter.protocol,
        "source_format": parsed_filter.source_format,
        "source_profile": parsed_filter.source_profile,
        "harness_revision": SWE_PRBENCH_HARNESS_REVISION,
        "harness_license": SWE_PRBENCH_HARNESS_LICENSE,
        "dataset_license": SWE_PRBENCH_DATASET_LICENSE,
        "underlying_repository_license": "not_normalized_by_upstream",
        "upstream_pr_record_sha256": pr.record_sha256,
        "upstream_pr_record_size_bytes": pr.record_size_bytes,
        "upstream_annotation_record_sha256": annotation.record_sha256,
        "upstream_context_source_sha256": {
            name: contexts[name].source_sha256 for name in SWE_PRBENCH_CONTEXT_CONFIGS
        },
        "ground_truth": {
            "substantive_comment_ids": list(annotation.value["substantive_comment_ids"]),
            "fallback_policy": parsed_filter.fallback,
            "fallback_used": truth.fallback_used,
            "status": truth.status,
            "reason": truth.reason,
            "offending_record_sha256": truth.offending_record_sha256,
            "severity_policy": "not_scorable_no_upstream_authority",
            "category_policy": "source_record_kind_human_review_comment",
            "nullable_line_policy": "semantic_only_upstream_nullable_side_unknown",
        },
        "difficulty_policy": "upstream_declared_dimension_only_not_gate_truth",
        "disposition": disposition,
    }


def _record_partition(selected: bool, truth_status: str) -> str:
    """Return exactly one source-record partition for auditable statistics."""

    representable = truth_status == "representable"
    if not representable and truth_status not in {
        "empty_ground_truth",
        "claim_exceeds_claim_limit",
    }:
        raise PublicFormatError("unknown SWE-PRBench truth status")
    if selected:
        return "selected_included" if representable else "selected_isolated"
    return "unselected_filtered" if representable else "upstream_isolated"


def _frozen_bundle_extra_files(
    bundle: PreparedFrozenContextBundle,
    relative_root: str,
) -> Dict[str, bytes]:
    """Copy the verified Frozen bundle into a Suite-owned subdirectory."""

    try:
        extras: Dict[str, bytes] = {}
        manifest_raw = _read_single_link_regular_file(
            bundle.root,
            SWE_PRBENCH_FROZEN_MANIFEST_PATH,
            _MAX_FROZEN_MANIFEST_BYTES,
            "SWE-PRBench frozen context bundle manifest",
        )
        if manifest_raw != canonical_json_bytes(bundle.manifest.to_dict()):
            raise PublicPreparationError(
                "verified Frozen bundle manifest drifted before copy"
            )
        extras[
            "%s/%s" % (relative_root, SWE_PRBENCH_FROZEN_MANIFEST_PATH)
        ] = manifest_raw
        for binding in bundle.manifest.records:
            raw = _read_single_link_regular_file(
                bundle.root,
                binding.path,
                _MAX_FROZEN_RECORD_BYTES,
                "SWE-PRBench frozen context bundle record",
            )
            if (
                len(raw) != binding.size_bytes
                or hashlib.sha256(raw).hexdigest() != binding.sha256
            ):
                raise PublicPreparationError(
                    "verified Frozen bundle record size/hash drifted before copy"
                )
            envelope = FrozenContextEnvelope.from_json(raw)
            if canonical_json_bytes(envelope.to_dict()) != raw:
                raise PublicPreparationError(
                    "verified Frozen bundle record canonical bytes drifted before copy"
                )
            record = envelope.record
            if (
                envelope.bundle_id != bundle.manifest.bundle_id
                or record.digest() != binding.record_digest
                or record.task_id != binding.task_id
                or record.config_name != binding.config_name
                or record.rendered_sha256 != binding.rendered_sha256
                or record.rendered_utf8_bytes != binding.rendered_utf8_bytes
            ):
                raise PublicPreparationError(
                    "verified Frozen bundle record binding drifted before copy"
                )
            extras["%s/%s" % (relative_root, binding.path)] = raw
        return extras
    except PublicPreparationError:
        raise
    except (PublicDatasetError, OSError, SchemaError) as exc:
        raise PublicPreparationError(
            "verified Frozen bundle drifted before copy"
        ) from exc


def _prepare_swe_prbench_projection(
    source_root: os.PathLike[str] | str,
    output_root: os.PathLike[str] | str,
    *,
    source_manifest: PublicSourceManifest,
    filter_manifest: PublicFilterManifest,
    protocol: str,
    parsed_filter: _Filter,
    frozen_bundle: Optional[PreparedFrozenContextBundle],
    expected_source_manifest_digest: Optional[str] = None,
    suite_id: Optional[str] = None,
    suite_version: str = "v0.4.1-b87f5797",
) -> PublicPreparationResult:
    dataset = _parse_dataset(
        source_root,
        source_manifest,
        parsed_filter,
        expected_source_manifest_digest,
    )
    _validate_filter_targets(dataset, parsed_filter)
    resolved_suite_id = suite_id or (
        "swe-prbench-frozen"
        if protocol == SWE_PRBENCH_PROTOCOL_FROZEN
        else "swe-prbench-native"
    )
    wire_contract = (
        SWE_PRBENCH_FROZEN_WIRE_CONTRACT
        if protocol == SWE_PRBENCH_PROTOCOL_FROZEN
        else SWE_PRBENCH_NATIVE_WIRE_CONTRACT
    )
    prepared_cases: List[PublicPreparedCase] = []
    receipts: List[PublicRecordReceipt] = []
    statistics = {
        "source_pr_records": len(dataset.prs),
        "source_annotations": len(dataset.annotations),
        "source_contexts": len(dataset.contexts),
        "source_representable_records": 0,
        "source_isolated_records": 0,
        "selected_included": 0,
        "selected_isolated": 0,
        "unselected_filtered": 0,
        "upstream_isolated": 0,
        "included_cases": 0,
        "empty_ground_truth_isolations": 0,
        "oversized_claim_isolations": 0,
        "expected_findings": 0,
        "nullable_line_findings": 0,
        "severity_unscorable_findings": 0,
        "location_unscorable_findings": 0,
        "fallback_cases": 0,
        "capability_cases": 0,
    }
    for pr_record in dataset.prs:
        pr = pr_record.value
        task_id = pr["task_id"]
        annotation = dataset.annotations[task_id]
        contexts = {
            config: dataset.contexts[(task_id, config)]
            for config in SWE_PRBENCH_CONTEXT_CONFIGS
        }
        truth = _select_truth(annotation.value, parsed_filter.fallback)
        selected = _selected(pr, parsed_filter)
        partition = _record_partition(selected, truth.status)
        statistics[partition] += 1
        representable = truth.status == "representable"
        statistics[
            "source_representable_records"
            if representable
            else "source_isolated_records"
        ] += 1
        disposition = {
            "selected_included": "included",
            "selected_isolated": "isolated",
            "unselected_filtered": "filtered_out",
            "upstream_isolated": "isolated",
        }[partition]
        if truth.status == "empty_ground_truth":
            statistics["empty_ground_truth_isolations"] += 1
        elif truth.status == "claim_exceeds_claim_limit":
            statistics["oversized_claim_isolations"] += 1
        mapping = _mapping_record(
            pr_record, annotation, contexts, parsed_filter, truth, disposition
        )
        receipts.append(
            PublicRecordReceipt.from_record(
                task_id=task_id,
                truth_id=None,
                source_role=pr_record.source_role,
                record_pointer=pr_record.pointer,
                upstream_id=pr["pr_url"],
                record=mapping,
                disposition=disposition,
                reason=truth.reason if disposition == "isolated" else None,
            )
        )
        if disposition == "isolated" and truth.offending_record_sha256 is not None:
            offending = None
            pointer = annotation.pointer
            upstream_id = None
            for index, comment in enumerate(annotation.value["comments"]):
                if canonical_sha256(comment) == truth.offending_record_sha256:
                    offending = comment
                    pointer = annotation.pointer + "/comments/%d" % index
                    upstream_id = comment["comment_id"]
                    break
            if offending is None:
                offending = annotation.value
            receipts.append(
                PublicRecordReceipt.from_record(
                    task_id=task_id,
                    truth_id=None,
                    source_role=annotation.source_role,
                    record_pointer=pointer,
                    upstream_id=upstream_id,
                    record=offending,
                    disposition="isolated",
                    reason=truth.reason,
                )
            )
        if partition != "selected_included":
            continue
        findings = tuple(
            _finding(task_id, comment, pr["changed_files"])
            for comment in truth.comments
        )
        for index, comment in enumerate(truth.comments):
            source_index, source_comment, truth_id = _resolve_annotation_comment(
                annotation, comment
            )
            if findings[index].truth_id != truth_id:
                raise PublicFormatError(
                    "SWE-PRBench Finding truth_id does not bind its original annotation comment"
                )
            receipts.append(
                PublicRecordReceipt.from_record(
                    task_id=task_id,
                    truth_id=truth_id,
                    source_role=annotation.source_role,
                    record_pointer=annotation.pointer + "/comments/%d" % source_index,
                    upstream_id=source_comment["comment_id"],
                    record=source_comment,
                    disposition="included",
                )
            )
        review_evaluator_context = _review_evaluator_contexts(
            annotation=annotation,
            findings=findings,
            comments=truth.comments,
        )
        if protocol == SWE_PRBENCH_PROTOCOL_FROZEN:
            assert frozen_bundle is not None
            config = parsed_filter.context_config
            bindings = [
                item
                for item in frozen_bundle.manifest.records
                if item.task_id == task_id and item.config_name == config
            ]
            if len(bindings) != 1:
                raise PublicPreparationError(
                    "Frozen Suite has no unique context binding for %s" % task_id
                )
            binding = bindings[0]
            from ..frozen_context import frozen_context_source_binding_digest

            review_target = FrozenContextReviewTarget(
                kind=ReviewTargetKind.FROZEN_CONTEXT,
                bundle_id=frozen_bundle.manifest.bundle_id,
                record_id=binding.task_id,
                context_format="rendered_text",
                rendered_sha256=binding.rendered_sha256,
                rendered_utf8_bytes=binding.rendered_utf8_bytes,
                source_binding_digest=frozen_context_source_binding_digest(
                    frozen_bundle, binding
                ),
            )
            case_protocol = protocol
            case_hash = canonical_sha256(
                {
                    "adapter_version": SWE_PRBENCH_ADAPTER_VERSION,
                    "dataset_revision": source_manifest.source_revision,
                    "protocol": protocol,
                    "pr_record_sha256": pr_record.record_sha256,
                    "annotation_record_sha256": annotation.record_sha256,
                    "frozen_record_digest": binding.record_digest,
                }
            )
        else:
            review_target = RepositoryReviewTarget(
                kind=ReviewTargetKind.REPOSITORY,
                repository=Repository(
                    source=RepositorySource.GIT,
                    path=None,
                    url=pr["repo_clone_url"],
                    base_revision=pr["base_commit"],
                    head_revision=pr["head_commit"],
                ),
                review_request=ReviewRequest(
                    title=pr["title"],
                    description=pr["description"] or None,
                    user_intent=None,
                    review_focus=None,
                    linked_requirements=(),
                    project_rules=(),
                    existing_ci_evidence=(),
                ),
            )
            case_protocol = protocol
            case_hash = canonical_sha256(
                {
                    "adapter_version": SWE_PRBENCH_ADAPTER_VERSION,
                    "dataset_revision": source_manifest.source_revision,
                    "protocol": protocol,
                    "pr_record_sha256": pr_record.record_sha256,
                    "annotation_record_sha256": annotation.record_sha256,
                }
            )
        case = EvalCase(
            schema_version=EVAL_CASE_SCHEMA_VERSION,
            task_id=task_id,
            case_version=1,
            source=CaseSource(
                suite=resolved_suite_id,
                origin=CaseOrigin.SWE_PRBENCH,
                source_id=task_id,
                source_version=source_manifest.source_revision,
                source_uri=source_manifest.source_uri,
                license=SWE_PRBENCH_DATASET_LICENSE,
                content_hash=case_hash,
            ),
            input=EvalCaseInput(review_target=review_target),
            clarification_script=ClarificationScript(max_rounds=4, answers=()),
            intent_truth=IntentTruth(
                scorable=False,
                authority=None,
                expected_claims=(),
                forbidden_claims=(),
                clarification_policy=None,
            ),
            review_truth=ReviewTruth(
                completeness=TruthCompleteness.HUMAN_OBSERVED,
                novel_finding_policy=NovelFindingPolicy.VERIFY,
                expected_findings=findings,
                known_invalid_findings=(),
            ),
            review_evaluator_context=review_evaluator_context,
        )
        prepared_cases.append(
            PublicPreparedCase(
                case=case,
                split=CaseSplit.CAPABILITY,
                protocol_id=(
                    SWE_PRBENCH_FROZEN_PROTOCOL_ID
                    if protocol == SWE_PRBENCH_PROTOCOL_FROZEN
                    else SWE_PRBENCH_NATIVE_PROTOCOL_ID
                ),
                dimensions=_case_dimensions(pr, len(findings), case_protocol),
            )
        )
        statistics["included_cases"] += 1
        statistics["capability_cases"] += 1
        statistics["expected_findings"] += len(findings)
        statistics["severity_unscorable_findings"] += len(findings)
        statistics["location_unscorable_findings"] += len(findings)
        statistics["nullable_line_findings"] += sum(
            comment["line"] is None for comment in truth.comments
        )
        statistics["fallback_cases"] += int(truth.fallback_used)
    partition_total = sum(
        statistics[name]
        for name in (
            "selected_included",
            "selected_isolated",
            "unselected_filtered",
            "upstream_isolated",
        )
    )
    if (
        partition_total != statistics["source_pr_records"]
        or statistics["source_representable_records"]
        != statistics["selected_included"] + statistics["unselected_filtered"]
        or statistics["source_isolated_records"]
        != statistics["selected_isolated"] + statistics["upstream_isolated"]
        or statistics["source_isolated_records"]
        != statistics["empty_ground_truth_isolations"]
        + statistics["oversized_claim_isolations"]
        or statistics["included_cases"] != statistics["selected_included"]
    ):
        raise PublicPreparationError(
            "SWE-PRBench record partition statistics failed their total invariants"
        )
    if not prepared_cases:
        isolated = [
            item.value["task_id"]
            for item in dataset.prs
            if _selected(item.value, parsed_filter)
        ]
        raise PublicPreparationError(
            "SWE-PRBench selected records produced no runnable cases; fail-closed tasks: %s"
            % ", ".join(isolated[:20])
        )
    if parsed_filter.source_profile == SWE_PRBENCH_SOURCE_PROFILE_FIXTURE:
        trusted_source_digest = SWE_PRBENCH_FIXTURE_SOURCE_MANIFEST_DIGEST
    elif parsed_filter.source_profile == SWE_PRBENCH_SOURCE_PROFILE_OFFICIAL_RAW:
        trusted_source_digest = SWE_PRBENCH_OFFICIAL_RAW_SOURCE_MANIFEST_DIGEST
    else:
        assert expected_source_manifest_digest is not None
        trusted_source_digest = expected_source_manifest_digest
    frozen_publication = (
        None
        if frozen_bundle is None
        else PublicFrozenBundlePublication(
            bundle=frozen_bundle,
            relative_root=SWE_PRBENCH_FROZEN_SUITE_RELATIVE_ROOT,
        )
    )
    extras = (
        _frozen_bundle_extra_files(
            frozen_bundle,
            frozen_publication.relative_root,
        )
        if frozen_bundle is not None and frozen_publication is not None
        else {}
    )
    return write_public_suite(
        output_root,
        suite_id=resolved_suite_id,
        suite_version=suite_version,
        adapter_id=SWE_PRBENCH_ADAPTER_ID,
        adapter_version=SWE_PRBENCH_ADAPTER_VERSION,
        source_manifest=source_manifest,
        filter_manifest=filter_manifest,
        wire_contract=wire_contract,
        cases=prepared_cases,
        actual_statistics=tuple(
            PublicStatistic(name=name, value=value)
            for name, value in statistics.items()
        ),
        records=receipts,
        extra_files=extras,
        expected_source_manifest_digest=trusted_source_digest,
        frozen_publication=frozen_publication,
    )


def prepare_swe_prbench(
    source_root: os.PathLike[str] | str,
    output_root: os.PathLike[str] | str,
    *,
    source_manifest: PublicSourceManifest,
    filter_manifest: PublicFilterManifest,
    protocol: str,
    expected_source_manifest_digest: Optional[str] = None,
    suite_id: Optional[str] = None,
    suite_version: str = "v0.4.1-b87f5797",
) -> PublicPreparationResult:
    """Publish one target-specific, runnable SWE-PRBench v2 Suite."""

    parsed_filter = _parse_filter(filter_manifest, protocol)
    if protocol not in (SWE_PRBENCH_PROTOCOL_NATIVE, SWE_PRBENCH_PROTOCOL_FROZEN):
        raise PublicFormatError("unknown SWE-PRBench protocol")
    output = Path(os.path.abspath(os.fspath(output_root)))
    if os.path.lexists(output):
        raise PublicConflictError("public Suite output already exists")

    projection_kwargs = {
        "source_manifest": source_manifest,
        "filter_manifest": filter_manifest,
        "protocol": protocol,
        "parsed_filter": parsed_filter,
        "expected_source_manifest_digest": expected_source_manifest_digest,
        "suite_id": suite_id,
        "suite_version": suite_version,
    }
    if protocol == SWE_PRBENCH_PROTOCOL_NATIVE:
        return _prepare_swe_prbench_projection(
            source_root,
            output_root,
            frozen_bundle=None,
            **projection_kwargs,
        )

    parent = _assert_publication_parent(output.parent)
    bundle_container = Path(
        tempfile.mkdtemp(prefix=".%s.frozen." % output.name, dir=parent)
    )
    bundle_container_identity: Optional[Tuple[int, int, int]] = None
    try:
        bundle_container_identity = _file_identity(
            os.lstat(str(bundle_container))
        )
        frozen_bundle = prepare_swe_prbench_frozen_bundle(
            source_root,
            bundle_container / "bundle",
            source_manifest=source_manifest,
            filter_manifest=filter_manifest,
            expected_source_manifest_digest=expected_source_manifest_digest,
        )
        return _prepare_swe_prbench_projection(
            source_root,
            output_root,
            frozen_bundle=frozen_bundle,
            **projection_kwargs,
        )
    finally:
        if (
            bundle_container_identity is not None
            and os.path.lexists(bundle_container)
        ):
            _cleanup_owned_staging(
                bundle_container,
                parent,
                bundle_container_identity,
            )


def prepare_swe_prbench_frozen_bundle(
    source_root: os.PathLike[str] | str,
    output_root: os.PathLike[str] | str,
    *,
    source_manifest: PublicSourceManifest,
    filter_manifest: PublicFilterManifest,
    expected_source_manifest_digest: Optional[str] = None,
) -> PreparedFrozenContextBundle:
    """Publish exact frozen-context source records without claiming runnability."""

    parsed_filter = _parse_filter(filter_manifest, SWE_PRBENCH_PROTOCOL_FROZEN)
    dataset = _parse_dataset(
        source_root,
        source_manifest,
        parsed_filter,
        expected_source_manifest_digest,
    )
    _validate_filter_targets(dataset, parsed_filter)
    config = parsed_filter.context_config
    selected: List[
        Tuple[_BoundRecord, _BoundRecord, _BoundContext, _TruthSelection]
    ] = []
    for pr_record in dataset.prs:
        if not _selected(pr_record.value, parsed_filter):
            continue
        task_id = pr_record.value["task_id"]
        annotation = dataset.annotations[task_id]
        truth = _select_truth(annotation.value, parsed_filter.fallback)
        selected.append(
            (pr_record, annotation, dataset.contexts[(task_id, config)], truth)
        )
    if not selected:
        raise PublicPreparationError("frozen context filter selected no records")

    output = Path(os.path.abspath(os.fspath(output_root)))
    parent = _assert_publication_parent(output.parent)
    output = parent / output.name
    if os.path.lexists(output):
        raise PublicConflictError("frozen context bundle output already exists")
    source_digest = source_manifest.digest()
    filter_digest = filter_manifest.digest()
    identity_records: List[Dict[str, Any]] = []
    for pr_record, annotation, bound, truth in sorted(
        selected, key=lambda item: item[2].record.task_id
    ):
        record = bound.record
        identity_records.append(
            {
                "task_id": record.task_id,
                "config_name": record.config_name,
                "path": "records/%s/%s.json" % (
                    record.task_id,
                    record.config_name,
                ),
                "record_digest": record.digest(),
                "rendered_sha256": record.rendered_sha256,
                "rendered_utf8_bytes": record.rendered_utf8_bytes,
                "source_role": bound.source_role,
                "source_file_sha256": bound.source_sha256,
                "pr_record_sha256": pr_record.record_sha256,
                "annotation_record_sha256": annotation.record_sha256,
                "review_truth_status": truth.status,
                "review_truth_reason": truth.reason,
                "offending_record_sha256": truth.offending_record_sha256,
            }
        )
    try:
        _assert_no_portable_path_collisions(
            (item["path"] for item in identity_records),
            "SWE-PRBench frozen bundle record paths",
            reserved=(SWE_PRBENCH_FROZEN_MANIFEST_PATH,),
        )
    except SchemaError as exc:
        raise PublicPreparationError(str(exc)) from exc
    bundle_id = _compute_frozen_bundle_id(
        adapter_id=SWE_PRBENCH_ADAPTER_ID,
        adapter_version=SWE_PRBENCH_ADAPTER_VERSION,
        harness_revision=SWE_PRBENCH_HARNESS_REVISION,
        harness_license=SWE_PRBENCH_HARNESS_LICENSE,
        dataset_license=SWE_PRBENCH_DATASET_LICENSE,
        underlying_repository_license=(
            SWE_PRBENCH_UNDERLYING_REPOSITORY_LICENSE
        ),
        source_manifest_digest=source_digest,
        filter_manifest_digest=filter_digest,
        identity_records=identity_records,
    )
    staging = Path(
        tempfile.mkdtemp(prefix=".%s." % output.name, suffix=".staging", dir=parent)
    )
    staging_identity = _file_identity(os.lstat(str(staging)))
    published = False
    try:
        bindings: List[FrozenContextBinding] = []
        for (pr_record, annotation, bound, truth), identity in zip(
            sorted(selected, key=lambda item: item[2].record.task_id),
            identity_records,
        ):
            record = bound.record
            path = identity["path"]
            envelope = FrozenContextEnvelope(
                schema_version=SWE_PRBENCH_FROZEN_ENVELOPE_SCHEMA_VERSION,
                bundle_id=bundle_id,
                record=record,
            )
            raw = canonical_json_bytes(envelope.to_dict())
            _write_new(staging, path, raw)
            bindings.append(
                FrozenContextBinding(
                    task_id=record.task_id,
                    config_name=record.config_name,
                    path=path,
                    size_bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    record_digest=record.digest(),
                    rendered_sha256=record.rendered_sha256,
                    rendered_utf8_bytes=record.rendered_utf8_bytes,
                    source_role=bound.source_role,
                    source_file_sha256=bound.source_sha256,
                    pr_record_sha256=pr_record.record_sha256,
                    annotation_record_sha256=annotation.record_sha256,
                    review_truth_status=truth.status,
                    review_truth_reason=truth.reason,
                    offending_record_sha256=truth.offending_record_sha256,
                )
            )
        sorted_bindings = tuple(sorted(bindings, key=lambda item: (item.task_id, item.config_name)))
        manifest = FrozenContextBundleManifest(
            schema_version=SWE_PRBENCH_FROZEN_BUNDLE_SCHEMA_VERSION,
            bundle_id=bundle_id,
            adapter_id=SWE_PRBENCH_ADAPTER_ID,
            adapter_version=SWE_PRBENCH_ADAPTER_VERSION,
            harness_revision=SWE_PRBENCH_HARNESS_REVISION,
            harness_license=SWE_PRBENCH_HARNESS_LICENSE,
            dataset_license=SWE_PRBENCH_DATASET_LICENSE,
            underlying_repository_license=(
                SWE_PRBENCH_UNDERLYING_REPOSITORY_LICENSE
            ),
            source_manifest=source_manifest,
            source_manifest_digest=source_digest,
            filter_manifest=filter_manifest,
            filter_manifest_digest=filter_digest,
            records=sorted_bindings,
        )
        _write_new(
            staging,
            SWE_PRBENCH_FROZEN_MANIFEST_PATH,
            canonical_json_bytes(manifest.to_dict()),
        )
        loaded = read_swe_prbench_frozen_bundle(
            staging, expected_bundle_id=bundle_id
        )
        if loaded.manifest != manifest:
            raise PublicSourceIntegrityError("staged frozen context bundle drifted")
        prepared = PreparedFrozenContextBundle(root=output, manifest=manifest)
        _publish_directory_create_only(staging, output)
        published = True
        return prepared
    finally:
        if not published and os.path.lexists(staging):
            _cleanup_owned_staging(staging, parent, staging_identity)


def read_swe_prbench_frozen_bundle(
    root: os.PathLike[str] | str,
    *,
    expected_bundle_id: str,
) -> PreparedFrozenContextBundle:
    verified_root = _coerce_suite_root(root)
    raw_manifest = _read_single_link_regular_file(
        verified_root,
        SWE_PRBENCH_FROZEN_MANIFEST_PATH,
        _MAX_FROZEN_MANIFEST_BYTES,
        "SWE-PRBench frozen context bundle manifest",
    )
    try:
        manifest = FrozenContextBundleManifest.from_json(raw_manifest)
    except PublicSourceIntegrityError:
        raise
    except SchemaError as exc:
        raise PublicSourceIntegrityError(
            "frozen context bundle manifest is malformed"
        ) from exc
    if canonical_json_bytes(manifest.to_dict()) != raw_manifest:
        raise PublicSourceIntegrityError(
            "frozen context bundle manifest bytes are not canonical"
        )
    try:
        expected = _identifier(
            expected_bundle_id, "expected frozen context bundle ID"
        )
    except SchemaError as exc:
        raise PublicSourceIntegrityError(str(exc)) from exc
    if manifest.bundle_id != expected:
        raise PublicSourceIntegrityError(
            "frozen context bundle ID does not match the expected trust anchor"
        )
    parsed_filter = _parse_filter(
        manifest.filter_manifest, SWE_PRBENCH_PROTOCOL_FROZEN
    )
    _validate_source_manifest(
        manifest.source_manifest,
        parsed_filter,
        manifest.source_manifest_digest,
    )
    for binding in manifest.records:
        raw = _read_single_link_regular_file(
            verified_root,
            binding.path,
            _MAX_FROZEN_RECORD_BYTES,
            "SWE-PRBench frozen context bundle record",
        )
        if len(raw) != binding.size_bytes or hashlib.sha256(raw).hexdigest() != binding.sha256:
            raise PublicSourceIntegrityError("frozen context bundle record bytes drifted")
        try:
            envelope = FrozenContextEnvelope.from_json(raw)
        except SchemaError as exc:
            raise PublicSourceIntegrityError(
                "frozen context bundle record is malformed"
            ) from exc
        if canonical_json_bytes(envelope.to_dict()) != raw:
            raise PublicSourceIntegrityError(
                "frozen context bundle record bytes are not canonical"
            )
        if envelope.bundle_id != manifest.bundle_id:
            raise PublicSourceIntegrityError(
                "frozen context envelope does not point back to its manifest"
            )
        record = envelope.record
        if (
            record.digest() != binding.record_digest
            or record.task_id != binding.task_id
            or record.config_name != binding.config_name
            or record.rendered_sha256 != binding.rendered_sha256
            or record.rendered_utf8_bytes != binding.rendered_utf8_bytes
        ):
            raise PublicSourceIntegrityError("frozen context bundle record binding drifted")
    return PreparedFrozenContextBundle(root=verified_root, manifest=manifest)


__all__ = [
    "SWE_PRBENCH_DATASET_ID",
    "SWE_PRBENCH_DATASET_VERSION",
    "SWE_PRBENCH_DATASET_URI",
    "SWE_PRBENCH_DATASET_REVISION",
    "SWE_PRBENCH_DATASET_LICENSE",
    "SWE_PRBENCH_FIXTURE_DATASET_VERSION",
    "SWE_PRBENCH_FIXTURE_SOURCE_URI",
    "SWE_PRBENCH_FIXTURE_SOURCE_REVISION",
    "SWE_PRBENCH_HARNESS_REVISION",
    "SWE_PRBENCH_HARNESS_LICENSE",
    "SWE_PRBENCH_PIPELINE_VERSION",
    "SWE_PRBENCH_PARQUET_CONVERTER_REVISION",
    "SWE_PRBENCH_PROTOCOL_NATIVE",
    "SWE_PRBENCH_PROTOCOL_FROZEN",
    "SWE_PRBENCH_SOURCE_RAW",
    "SWE_PRBENCH_SOURCE_PARQUET",
    "SWE_PRBENCH_SOURCE_PROFILE_OFFICIAL_RAW",
    "SWE_PRBENCH_SOURCE_PROFILE_FIXTURE",
    "SWE_PRBENCH_SOURCE_PROFILE_EXPLICIT",
    "SWE_PRBENCH_CONTEXT_CONFIGS",
    "SWE_PRBENCH_DIFFICULTIES",
    "SWE_PRBENCH_LANGUAGES",
    "SWE_PRBENCH_NATIVE_PROTOCOL_ID",
    "SWE_PRBENCH_FROZEN_PROTOCOL_ID",
    "SWE_PRBENCH_NATIVE_WIRE_CONTRACT",
    "SWE_PRBENCH_FROZEN_WIRE_CONTRACT",
    "SWE_PRBENCH_OFFICIAL_RAW_SOURCE_MANIFEST_DIGEST",
    "SWE_PRBENCH_FIXTURE_SOURCE_MANIFEST_DIGEST",
    "SWE_PRBENCH_FROZEN_BUNDLE_SCHEMA_VERSION",
    "SWE_PRBENCH_FROZEN_RECORD_SCHEMA_VERSION",
    "SWE_PRBENCH_FROZEN_ENVELOPE_SCHEMA_VERSION",
    "FrozenContextRecord",
    "FrozenContextEnvelope",
    "FrozenContextBinding",
    "FrozenContextBundleManifest",
    "PreparedFrozenContextBundle",
    "SWEPRBenchSourceValidation",
    "validate_swe_prbench_source",
    "prepare_swe_prbench",
    "prepare_swe_prbench_frozen_bundle",
    "read_swe_prbench_frozen_bundle",
]

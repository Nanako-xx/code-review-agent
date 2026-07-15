"""Canonical, immutable models for the durable memory subsystem.

This module is deliberately independent from persistence, repository access, model
providers, and the review pipeline.  It defines the only values those layers may
exchange.  Every persisted model has strict hydration, bounded fields, canonical
JSON serialization, and content-derived full SHA-256 identities.

The implementation intentionally uses only Python 3.9 syntax and standard-library
types.  Boundary values are strings, integers, booleans, ``None``, and tuples; no
``Path``, ``datetime``, SQLite row, or provider object is retained by a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, cast


MODEL_SCHEMA_VERSION = 1
CURRENT_MEMORY_STORE_SCHEMA_VERSION = 2
SUPPORTED_MEMORY_STORE_SCHEMA_VERSIONS = frozenset({1, 2})
LEGACY_MEMORY_SELECTION_POLICY_VERSION = "memory_selection_v1"
MEMORY_SELECTION_POLICY_VERSION = "memory_selection_v2"
SUPPORTED_MEMORY_SELECTION_POLICY_VERSIONS = frozenset(
    {
        LEGACY_MEMORY_SELECTION_POLICY_VERSION,
        MEMORY_SELECTION_POLICY_VERSION,
    }
)
FEEDBACK_AGGREGATION_POLICY_VERSION = "feedback_aggregation_v1"

MAX_STATEMENT_LENGTH = 8_192
MAX_HUMAN_DECLARATION_LENGTH = 8_192
MAX_TEXT_LENGTH = 4_096
MAX_REASON_LENGTH = 2_048
MAX_IDENTIFIER_LENGTH = 512
MAX_PATH_LENGTH = 1_024
MAX_SCOPE_ITEMS = 128
MAX_SOURCE_REFS = 64
MAX_HUMAN_DECLARATIONS = 64
MAX_EVIDENCE_REFS = 256
MAX_DECISION_REASONS = 32
MAX_SNAPSHOT_RECORDS = 2_000
MAX_SNAPSHOT_DECISIONS = 4_000
MAX_KNOWLEDGE_REFS = 4_096
MAX_PINNED_REVIEWS = 4_096
MAX_FEEDBACK_SOURCES = 10_000
MAX_CALIBRATION_SIGNALS = 128

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_ID_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_STABLE_ID_PATTERN = re.compile(r"^(?P<prefix>[A-Z][A-Z0-9]*)-(?P<digest>[0-9a-f]{64})$")
_FINDING_ID_PATTERN = re.compile(r"^F-[0-9a-f]{32}(?:[0-9a-f]{32})?$")
_OBSERVATION_ID_PATTERN = re.compile(r"^O-[0-9a-f]{12,64}$")
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+#/@-]{0,511}$")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[/\\]")
_CONTENT_TYPE_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$"
)
_SENSITIVE_PATH_NAMES = frozenset(
    {
        ".env",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "credentials",
        "credentials.json",
        "secrets.json",
    }
)


class MemoryKind(str, Enum):
    ARCHITECTURE_BOUNDARY = "architecture_boundary"
    BUSINESS_INVARIANT = "business_invariant"
    REVIEW_RULE = "review_rule"
    COMPATIBILITY_REQUIREMENT = "compatibility_requirement"
    VERIFICATION_COMMAND = "verification_command"
    INCIDENT_LESSON = "incident_lesson"
    HIGH_RISK_MODULE = "high_risk_module"


class CandidateStatus(str, Enum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


class RecordStatus(str, Enum):
    ACTIVE = "active"
    REVALIDATION_REQUIRED = "revalidation_required"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"
    EXPIRED = "expired"


class FeedbackStatus(str, Enum):
    RECORDED = "recorded"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"


class FeedbackDecision(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SEVERITY_CHANGED = "severity_changed"
    MISSED = "missed"


class FeedbackReasonCode(str, Enum):
    DUPLICATE = "duplicate"
    EXPECTED_BEHAVIOR = "expected_behavior"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    WRONG_SCOPE = "wrong_scope"
    SEVERITY_MISMATCH = "severity_mismatch"
    OTHER = "other"


class Applicability(str, Enum):
    SELECTED = "selected"
    OUT_OF_SCOPE = "out_of_scope"
    NOT_YET_VALID = "not_yet_valid"
    LINEAGE_MISMATCH = "lineage_mismatch"
    SOURCE_MISSING = "source_missing"
    SOURCE_CHANGED = "source_changed"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUPERSEDED = "superseded"
    BUDGET_OMITTED = "budget_omitted"


class Sensitivity(str, Enum):
    NORMAL = "normal"
    LOCAL_ONLY = "local_only"
    BLOCKED = "blocked"


class ValidityPolicy(str, Enum):
    SOURCE_CONTENT_HASH = "source_content_hash"
    SYMBOL_SIGNATURE = "symbol_signature"
    SCOPE_CHANGE_TRIGGER = "scope_change_trigger"
    MANUAL_UNTIL_REVOKED = "manual_until_revoked"


class ExpiryConditionKind(str, Enum):
    AT_TIME = "at_time"
    AT_COMMIT = "at_commit"


class PolicyEffectKind(str, Enum):
    RISK_FLOOR = "risk_floor"
    REQUIRE_CONTRACT = "require_contract"
    REQUIRE_CHECK = "require_check"
    VERIFICATION_HINT = "verification_hint"


class ProducerType(str, Enum):
    LOCAL = "local"
    MODEL = "model"
    HUMAN = "human"


class HumanDeclarationOrigin(str, Enum):
    USER_REQUEST = "user_request"
    CLI_REQUEST = "cli_request"


class SourceRefType(str, Enum):
    REPOSITORY_RANGE = "repository_range"
    REPOSITORY_SYMBOL = "repository_symbol"
    GIT_COMMIT = "git_commit"
    OBSERVATION = "observation"
    SESSION_ARTIFACT = "session_artifact"
    HUMAN_DECLARATION = "human_declaration"


class SymbolHashKind(str, Enum):
    SIGNATURE = "signature"
    BODY = "body"


class MemoryConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MemoryMode(str, Enum):
    OFF = "off"
    READ = "read"
    READ_WRITE = "read-write"


class FindingSeverity(str, Enum):
    BLOCKER = "blocker"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RepositoryKnowledgeCapability(str, Enum):
    FILE_INDEX = "file_index"
    SYMBOL_INDEX = "symbol_index"
    DEFINITIONS = "definitions"
    REFERENCES = "references"
    CALLS = "calls"
    TESTS = "tests"
    PROJECT_CONFIG = "project_config"
    GIT_SUMMARY = "git_summary"


class FeedbackCalibrationSignalKind(str, Enum):
    INCREASE_CHECK_PRIORITY = "increase_check_priority"
    EVIDENCE_GAP_WARNING = "evidence_gap_warning"
    SEVERITY_UNCERTAINTY = "severity_uncertainty"


EnumT = TypeVar("EnumT", bound=Enum)


def _json_ready(value: Any, context: str = "value") -> Any:
    if value is None or type(value) in {str, int, bool}:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("%s contains a non-finite number" % context)
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("%s contains a non-string object key" % context)
            result[key] = _json_ready(item, "%s.%s" % (context, key))
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_ready(item, "%s[%d]" % (context, index))
            for index, item in enumerate(value)
        ]
    raise ValueError("%s contains a non-JSON value" % context)


def canonical_json(value: Any) -> str:
    """Return the sole canonical JSON representation used for memory identities."""

    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *identity: Any) -> str:
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Z][A-Z0-9]*", prefix):
        raise ValueError("stable ID prefix must contain uppercase ASCII letters or digits")
    digest = canonical_sha256({"namespace": prefix, "identity": identity})
    return "%s-%s" % (prefix, digest)


def stable_event_id(*identity: Any) -> str:
    return stable_id("EVT", *identity)


def stable_request_id(*identity: Any) -> str:
    return stable_id("REQ", *identity)


def stable_repository_binding_id(*identity: Any) -> str:
    return stable_id("RB", *identity)


def validate_stable_id(value: Any, prefix: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError("%s must be a string" % field_name)
    match = _STABLE_ID_PATTERN.fullmatch(value)
    if match is None or match.group("prefix") != prefix:
        raise ValueError(
            "%s must be %s- followed by a full SHA-256 digest" % (field_name, prefix)
        )
    return value


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be an object" % context)
    for key in value:
        if not isinstance(key, str):
            raise ValueError("%s field names must be strings" % context)
    return value


def _exact_fields(payload: Mapping[str, Any], expected: Iterable[str], context: str) -> None:
    expected_set = set(expected)
    actual = set(payload)
    missing = expected_set - actual
    if missing:
        raise ValueError(
            "%s is missing required field(s): %s"
            % (context, ", ".join(sorted(missing)))
        )
    unexpected = actual - expected_set
    if unexpected:
        raise ValueError(
            "%s contains unsupported field(s): %s"
            % (context, ", ".join(sorted(unexpected)))
        )


def _validate_schema(value: Any, context: str) -> int:
    if type(value) is not int or value != MODEL_SCHEMA_VERSION:
        raise ValueError(
            "%s.schema_version must be %d" % (context, MODEL_SCHEMA_VERSION)
        )
    return value


def _enum_value(enum_type: Type[EnumT], value: Any, context: str) -> EnumT:
    if not isinstance(value, str):
        raise ValueError("%s must be a string" % context)
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError("%s has unsupported value: %s" % (context, value)) from error


def _normalize_text(
    value: Any,
    context: str,
    *,
    max_length: int = MAX_TEXT_LENGTH,
    allow_empty: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    if not isinstance(value, str):
        raise ValueError("%s must be a string" % context)
    normalized = unicodedata.normalize("NFC", value)
    if collapse_whitespace:
        normalized = " ".join(normalized.split())
    else:
        normalized = normalized.strip()
    if not normalized and not allow_empty:
        raise ValueError("%s must be a non-empty string" % context)
    if len(normalized) > max_length:
        raise ValueError("%s exceeds the maximum length of %d" % (context, max_length))
    if any(ord(character) < 32 for character in normalized):
        raise ValueError("%s must not contain control characters" % context)
    return normalized


def _normalize_identifier(value: Any, context: str) -> str:
    return _normalize_text(
        value,
        context,
        max_length=MAX_IDENTIFIER_LENGTH,
        collapse_whitespace=False,
    )


def _normalize_token(value: Any, context: str, *, casefold: bool = False) -> str:
    normalized = _normalize_identifier(value, context)
    if not _TOKEN_PATTERN.fullmatch(normalized):
        raise ValueError("%s must be a bounded identifier" % context)
    return normalized.casefold() if casefold else normalized


def _sha256_digest(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError("%s must be a SHA-256 digest" % context)
    normalized = value.casefold()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("%s must be a SHA-256 digest" % context)
    return normalized


def _git_object_id(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError("%s must be a full Git object ID" % context)
    normalized = value.casefold()
    if not _GIT_OBJECT_ID_PATTERN.fullmatch(normalized):
        raise ValueError("%s must be a full Git object ID" % context)
    return normalized


def _revision_binding(value: Any, context: str) -> str:
    normalized = _normalize_identifier(value, context).casefold()
    if _GIT_OBJECT_ID_PATTERN.fullmatch(normalized):
        return normalized
    for prefix in ("base@", "head@"):
        if normalized.startswith(prefix):
            return prefix + _git_object_id(normalized[len(prefix) :], context)
    if ".." in normalized:
        parts = normalized.split("..")
        if len(parts) == 2:
            return "%s..%s" % (
                _git_object_id(parts[0], context),
                _git_object_id(parts[1], context),
            )
    raise ValueError(
        "%s must bind a full Git object ID as base@SHA, head@SHA, SHA..SHA, or SHA"
        % context
    )


def _utc_timestamp(value: Any, context: str) -> str:
    normalized = _normalize_identifier(value, context)
    if not _UTC_TIMESTAMP_PATTERN.fullmatch(normalized):
        raise ValueError("%s must be an RFC 3339 UTC timestamp ending in Z" % context)
    try:
        datetime.fromisoformat(normalized[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("%s must be a valid UTC timestamp" % context) from error
    return normalized


def _canonical_utc_timestamp(value: Any, context: str) -> str:
    """Validate the unique UTC ``Z`` representation used in identities."""

    if not isinstance(value, str) or not _UTC_TIMESTAMP_PATTERN.fullmatch(value):
        raise ValueError("%s must be a canonical RFC 3339 UTC timestamp ending in Z" % context)
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("%s must be a valid UTC timestamp" % context) from error
    timespec = "microseconds" if parsed.microsecond else "seconds"
    canonical = parsed.isoformat(timespec=timespec).replace("+00:00", "Z")
    if value != canonical:
        raise ValueError("%s must use canonical UTC Z form" % context)
    return value


def _positive_int(value: Any, context: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError("%s must be a %s integer" % (context, qualifier))
    return value


def _optional_text(value: Any, context: str, *, max_length: int) -> Optional[str]:
    if value is None:
        return None
    return _normalize_text(value, context, max_length=max_length)


def _normalize_repository_key(value: Any, context: str = "repository_key") -> str:
    return _sha256_digest(value, context)


def _normalize_repo_path(value: Any, context: str, *, allow_glob: bool) -> str:
    raw = _normalize_text(
        value,
        context,
        max_length=MAX_PATH_LENGTH,
        collapse_whitespace=False,
    )
    raw = raw.replace("\\", "/")
    if raw.startswith("//") or raw.startswith("/") or _WINDOWS_DRIVE_PATTERN.match(raw):
        raise ValueError("%s must be a repository-relative POSIX path" % context)
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw:
        raise ValueError("%s must be a repository-relative POSIX path" % context)
    if not allow_glob and any(character in raw for character in "*?[]"):
        raise ValueError("%s must not contain glob syntax" % context)
    components: List[str] = []
    for component in raw.split("/"):
        if component in {"", "."}:
            continue
        if component == "..":
            raise ValueError("%s must be a repository-relative POSIX path" % context)
        lowered = component.casefold()
        if lowered in {".git", ".hg", ".svn"}:
            raise ValueError("%s must be a repository-relative POSIX path" % context)
        if lowered in _SENSITIVE_PATH_NAMES or lowered.startswith(".env."):
            raise ValueError("%s references a sensitive path" % context)
        components.append(component)
    if not components:
        raise ValueError("%s must be a repository-relative POSIX path" % context)
    normalized = str(PurePosixPath(*components))
    if len(normalized) > MAX_PATH_LENGTH:
        raise ValueError("%s exceeds the maximum path length" % context)
    return normalized


def _canonical_string_tuple(
    values: Any,
    context: str,
    *,
    normalizer: Any,
    max_items: int,
    allow_empty: bool = True,
) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError("%s must be a list or tuple" % context)
    if len(values) > max_items:
        raise ValueError("%s exceeds the maximum item count of %d" % (context, max_items))
    canonical = tuple(sorted({normalizer(item, "%s item" % context) for item in values}))
    if not canonical and not allow_empty:
        raise ValueError("%s must not be empty" % context)
    return canonical


def _enum_tuple(
    values: Any,
    enum_type: Type[EnumT],
    context: str,
    *,
    max_items: int,
    allow_empty: bool = False,
) -> Tuple[EnumT, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError("%s must be a list or tuple" % context)
    if len(values) > max_items:
        raise ValueError("%s exceeds the maximum item count of %d" % (context, max_items))
    normalized: Dict[str, EnumT] = {}
    for item in values:
        if not isinstance(item, enum_type):
            raise ValueError("%s items must be %s values" % (context, enum_type.__name__))
        normalized[item.value] = item
    result = tuple(normalized[key] for key in sorted(normalized))
    if not result and not allow_empty:
        raise ValueError("%s must not be empty" % context)
    return result


def _validate_finding_id(value: Any, context: str = "finding_id") -> str:
    if not isinstance(value, str) or not _FINDING_ID_PATTERN.fullmatch(value):
        raise ValueError("%s must be F- followed by 32 or 64 lowercase hex characters" % context)
    return value


def _validate_observation_id(value: Any, context: str = "observation_id") -> str:
    if not isinstance(value, str) or not _OBSERVATION_ID_PATTERN.fullmatch(value):
        raise ValueError("%s must be a non-empty canonical Observation ID" % context)
    return value


class _JsonModel:
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


class SourceRef(_JsonModel):
    """Closed union root for the six source reference variants."""

    @classmethod
    def from_dict(cls, payload: Any) -> "SourceRef":
        root = _object(payload, "source_ref")
        if "schema_version" not in root:
            raise ValueError("source_ref is missing required field(s): schema_version")
        _validate_schema(root["schema_version"], "source_ref")
        if "type" not in root:
            raise ValueError("source_ref is missing required field(s): type")
        source_type = _enum_value(SourceRefType, root["type"], "source_ref.type")
        variants = {
            SourceRefType.REPOSITORY_RANGE: RepositoryRangeSourceRef,
            SourceRefType.REPOSITORY_SYMBOL: RepositorySymbolSourceRef,
            SourceRefType.GIT_COMMIT: GitCommitSourceRef,
            SourceRefType.OBSERVATION: ObservationSourceRef,
            SourceRefType.SESSION_ARTIFACT: SessionArtifactSourceRef,
            SourceRefType.HUMAN_DECLARATION: HumanDeclarationSourceRef,
        }
        return variants[source_type]._from_dict(root)


@dataclass(frozen=True)
class RepositoryRangeSourceRef(SourceRef):
    revision: str
    path: str
    line_start: int
    line_end: int
    content_hash: str
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "repository_range_source")
        object.__setattr__(self, "revision", _git_object_id(self.revision, "revision"))
        object.__setattr__(
            self, "path", _normalize_repo_path(self.path, "path", allow_glob=False)
        )
        start = _positive_int(self.line_start, "line_start")
        end = _positive_int(self.line_end, "line_end")
        if end < start:
            raise ValueError("line_end must be greater than or equal to line_start")
        object.__setattr__(self, "content_hash", _sha256_digest(self.content_hash, "content_hash"))

    @property
    def source_type(self) -> SourceRefType:
        return SourceRefType.REPOSITORY_RANGE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "type": self.source_type.value,
            "revision": self.revision,
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content_hash": self.content_hash,
        }

    @classmethod
    def _from_dict(cls, payload: Mapping[str, Any]) -> "RepositoryRangeSourceRef":
        _exact_fields(
            payload,
            {"schema_version", "type", "revision", "path", "line_start", "line_end", "content_hash"},
            "repository_range_source",
        )
        return cls(
            revision=payload["revision"],
            path=payload["path"],
            line_start=payload["line_start"],
            line_end=payload["line_end"],
            content_hash=payload["content_hash"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True)
class RepositorySymbolSourceRef(SourceRef):
    revision: str
    path: str
    qualified_name: str
    hash_kind: SymbolHashKind
    content_hash: str
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "repository_symbol_source")
        if not isinstance(self.hash_kind, SymbolHashKind):
            raise ValueError("hash_kind must be a SymbolHashKind")
        object.__setattr__(self, "revision", _git_object_id(self.revision, "revision"))
        object.__setattr__(
            self, "path", _normalize_repo_path(self.path, "path", allow_glob=False)
        )
        object.__setattr__(
            self,
            "qualified_name",
            _normalize_identifier(self.qualified_name, "qualified_name"),
        )
        object.__setattr__(self, "content_hash", _sha256_digest(self.content_hash, "content_hash"))

    @property
    def source_type(self) -> SourceRefType:
        return SourceRefType.REPOSITORY_SYMBOL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "type": self.source_type.value,
            "revision": self.revision,
            "path": self.path,
            "qualified_name": self.qualified_name,
            "hash_kind": self.hash_kind.value,
            "content_hash": self.content_hash,
        }

    @classmethod
    def _from_dict(cls, payload: Mapping[str, Any]) -> "RepositorySymbolSourceRef":
        _exact_fields(
            payload,
            {"schema_version", "type", "revision", "path", "qualified_name", "hash_kind", "content_hash"},
            "repository_symbol_source",
        )
        return cls(
            revision=payload["revision"],
            path=payload["path"],
            qualified_name=payload["qualified_name"],
            hash_kind=_enum_value(SymbolHashKind, payload["hash_kind"], "repository_symbol_source.hash_kind"),
            content_hash=payload["content_hash"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True)
class GitCommitSourceRef(SourceRef):
    commit_sha: str
    metadata_hash: Optional[str] = None
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "git_commit_source")
        object.__setattr__(self, "commit_sha", _git_object_id(self.commit_sha, "commit_sha"))
        if self.metadata_hash is not None:
            object.__setattr__(
                self,
                "metadata_hash",
                _sha256_digest(self.metadata_hash, "metadata_hash"),
            )

    @property
    def source_type(self) -> SourceRefType:
        return SourceRefType.GIT_COMMIT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "type": self.source_type.value,
            "commit_sha": self.commit_sha,
            "metadata_hash": self.metadata_hash,
        }

    @classmethod
    def _from_dict(cls, payload: Mapping[str, Any]) -> "GitCommitSourceRef":
        _exact_fields(
            payload,
            {"schema_version", "type", "commit_sha", "metadata_hash"},
            "git_commit_source",
        )
        return cls(
            commit_sha=payload["commit_sha"],
            metadata_hash=payload["metadata_hash"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True)
class ObservationSourceRef(SourceRef):
    review_id: str
    observation_id: str
    revision_binding: str
    content_hash: str
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "observation_source")
        object.__setattr__(self, "review_id", _normalize_identifier(self.review_id, "review_id"))
        object.__setattr__(
            self,
            "observation_id",
            _validate_observation_id(self.observation_id),
        )
        object.__setattr__(
            self,
            "revision_binding",
            _revision_binding(self.revision_binding, "revision_binding"),
        )
        object.__setattr__(self, "content_hash", _sha256_digest(self.content_hash, "content_hash"))

    @property
    def source_type(self) -> SourceRefType:
        return SourceRefType.OBSERVATION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "type": self.source_type.value,
            "review_id": self.review_id,
            "observation_id": self.observation_id,
            "revision_binding": self.revision_binding,
            "content_hash": self.content_hash,
        }

    @classmethod
    def _from_dict(cls, payload: Mapping[str, Any]) -> "ObservationSourceRef":
        _exact_fields(
            payload,
            {"schema_version", "type", "review_id", "observation_id", "revision_binding", "content_hash"},
            "observation_source",
        )
        return cls(
            review_id=payload["review_id"],
            observation_id=payload["observation_id"],
            revision_binding=payload["revision_binding"],
            content_hash=payload["content_hash"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True)
class SessionArtifactSourceRef(SourceRef):
    review_id: str
    artifact_name: str
    artifact_schema: str
    revision_binding: str
    artifact_hash: str
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "session_artifact_source")
        object.__setattr__(self, "review_id", _normalize_identifier(self.review_id, "review_id"))
        object.__setattr__(
            self,
            "artifact_name",
            _normalize_repo_path(self.artifact_name, "artifact_name", allow_glob=False),
        )
        object.__setattr__(
            self,
            "artifact_schema",
            _normalize_token(self.artifact_schema, "artifact_schema"),
        )
        object.__setattr__(
            self,
            "revision_binding",
            _revision_binding(self.revision_binding, "revision_binding"),
        )
        object.__setattr__(
            self,
            "artifact_hash",
            _sha256_digest(self.artifact_hash, "artifact_hash"),
        )

    @property
    def source_type(self) -> SourceRefType:
        return SourceRefType.SESSION_ARTIFACT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "type": self.source_type.value,
            "review_id": self.review_id,
            "artifact_name": self.artifact_name,
            "artifact_schema": self.artifact_schema,
            "revision_binding": self.revision_binding,
            "artifact_hash": self.artifact_hash,
        }

    @classmethod
    def _from_dict(cls, payload: Mapping[str, Any]) -> "SessionArtifactSourceRef":
        _exact_fields(
            payload,
            {"schema_version", "type", "review_id", "artifact_name", "artifact_schema", "revision_binding", "artifact_hash"},
            "session_artifact_source",
        )
        return cls(
            review_id=payload["review_id"],
            artifact_name=payload["artifact_name"],
            artifact_schema=payload["artifact_schema"],
            revision_binding=payload["revision_binding"],
            artifact_hash=payload["artifact_hash"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True)
class HumanDeclarationSourceRef(SourceRef):
    request_id: str
    actor: str
    declaration_hash: str
    created_at: str
    review_id: Optional[str] = None
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "human_declaration_source")
        object.__setattr__(
            self,
            "request_id",
            validate_stable_id(self.request_id, "REQ", "request_id"),
        )
        object.__setattr__(self, "actor", _normalize_identifier(self.actor, "actor"))
        object.__setattr__(
            self,
            "declaration_hash",
            _sha256_digest(self.declaration_hash, "declaration_hash"),
        )
        object.__setattr__(self, "created_at", _utc_timestamp(self.created_at, "created_at"))
        if self.review_id is not None:
            object.__setattr__(
                self,
                "review_id",
                _normalize_identifier(self.review_id, "review_id"),
            )

    @property
    def source_type(self) -> SourceRefType:
        return SourceRefType.HUMAN_DECLARATION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "type": self.source_type.value,
            "request_id": self.request_id,
            "actor": self.actor,
            "declaration_hash": self.declaration_hash,
            "created_at": self.created_at,
            "review_id": self.review_id,
        }

    @classmethod
    def _from_dict(cls, payload: Mapping[str, Any]) -> "HumanDeclarationSourceRef":
        _exact_fields(
            payload,
            {"schema_version", "type", "request_id", "actor", "declaration_hash", "created_at", "review_id"},
            "human_declaration_source",
        )
        return cls(
            request_id=payload["request_id"],
            actor=payload["actor"],
            declaration_hash=payload["declaration_hash"],
            created_at=payload["created_at"],
            review_id=payload["review_id"],
            schema_version=payload["schema_version"],
        )


def _human_declaration_text(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError("%s must be a string" % context)
    if not value.strip():
        raise ValueError("%s must be a non-empty string" % context)
    if "\x00" in value:
        raise ValueError("%s must not contain NUL characters" % context)
    if len(value) > MAX_HUMAN_DECLARATION_LENGTH:
        raise ValueError(
            "%s exceeds the maximum length of %d"
            % (context, MAX_HUMAN_DECLARATION_LENGTH)
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("%s must contain valid UTF-8 text" % context) from error
    return value


@dataclass(frozen=True)
class HumanDeclarationAuthority(_JsonModel):
    source_ref: HumanDeclarationSourceRef
    origin: HumanDeclarationOrigin
    declaration: str = field(repr=False)
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "human_declaration_authority")
        if type(self.source_ref) is not HumanDeclarationSourceRef:
            raise ValueError(
                "human_declaration_authority.source_ref must be a "
                "HumanDeclarationSourceRef"
            )
        if not isinstance(self.origin, HumanDeclarationOrigin):
            raise ValueError(
                "human_declaration_authority.origin must be a "
                "HumanDeclarationOrigin"
            )
        object.__setattr__(
            self,
            "declaration",
            _human_declaration_text(
                self.declaration,
                "human_declaration_authority.declaration",
            ),
        )
        declaration_hash = hashlib.sha256(
            self.declaration.encode("utf-8")
        ).hexdigest()
        if declaration_hash != self.source_ref.declaration_hash:
            raise ValueError(
                "human_declaration_authority.declaration_hash does not match "
                "the UTF-8 declaration"
            )
        if (
            self.origin is HumanDeclarationOrigin.USER_REQUEST
            and self.source_ref.review_id is None
        ):
            raise ValueError(
                "human_declaration_authority.source_ref.review_id is required "
                "for user_request declarations"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_ref": self.source_ref.to_dict(),
            "origin": self.origin.value,
            "declaration": self.declaration,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "HumanDeclarationAuthority":
        root = _object(payload, "human_declaration_authority")
        _exact_fields(
            root,
            {"schema_version", "source_ref", "origin", "declaration"},
            "human_declaration_authority",
        )
        _validate_schema(root["schema_version"], "human_declaration_authority")
        source_ref = SourceRef.from_dict(root["source_ref"])
        if type(source_ref) is not HumanDeclarationSourceRef:
            raise ValueError(
                "human_declaration_authority.source_ref must be a "
                "HumanDeclarationSourceRef"
            )
        return cls(
            source_ref=source_ref,
            origin=_enum_value(
                HumanDeclarationOrigin,
                root["origin"],
                "human_declaration_authority.origin",
            ),
            declaration=root["declaration"],
            schema_version=root["schema_version"],
        )


def _canonical_source_refs(
    values: Any,
    context: str,
    *,
    allow_empty: bool = False,
) -> Tuple[SourceRef, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError("%s must be a list or tuple of SourceRef values" % context)
    if len(values) > MAX_SOURCE_REFS:
        raise ValueError("%s exceeds the maximum item count of %d" % (context, MAX_SOURCE_REFS))
    allowed_types = {
        RepositoryRangeSourceRef,
        RepositorySymbolSourceRef,
        GitCommitSourceRef,
        ObservationSourceRef,
        SessionArtifactSourceRef,
        HumanDeclarationSourceRef,
    }
    by_json: Dict[str, SourceRef] = {}
    for value in values:
        if type(value) not in allowed_types:
            raise ValueError(
                "%s items must be an exact allowlisted SourceRef variant" % context
            )
        by_json[value.to_json()] = value
    result = tuple(by_json[key] for key in sorted(by_json))
    if not result and not allow_empty:
        raise ValueError("%s must not be empty" % context)
    return result


def _source_refs_from_payload(value: Any, context: str) -> Tuple[SourceRef, ...]:
    if not isinstance(value, list):
        raise ValueError("%s must be a list" % context)
    return _canonical_source_refs(
        tuple(SourceRef.from_dict(item) for item in value),
        context,
    )


def _canonical_human_declarations(
    values: Any,
    context: str,
) -> Tuple[HumanDeclarationAuthority, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError(
            "%s must be a list or tuple of HumanDeclarationAuthority values"
            % context
        )
    if len(values) > MAX_HUMAN_DECLARATIONS:
        raise ValueError(
            "%s exceeds the maximum item count of %d"
            % (context, MAX_HUMAN_DECLARATIONS)
        )
    by_json: Dict[str, HumanDeclarationAuthority] = {}
    by_source_ref: Dict[str, HumanDeclarationAuthority] = {}
    for value in values:
        if type(value) is not HumanDeclarationAuthority:
            raise ValueError(
                "%s items must be exact HumanDeclarationAuthority values" % context
            )
        item_json = value.to_json()
        source_ref_json = value.source_ref.to_json()
        existing = by_source_ref.get(source_ref_json)
        if existing is not None and existing.to_json() != item_json:
            raise ValueError(
                "%s contains a duplicate source_ref with conflicting authority "
                "semantics" % context
            )
        by_source_ref[source_ref_json] = value
        by_json[item_json] = value
    return tuple(by_json[key] for key in sorted(by_json))


def _human_declarations_from_payload(
    value: Any,
    context: str,
) -> Tuple[HumanDeclarationAuthority, ...]:
    if not isinstance(value, list):
        raise ValueError("%s must be a list" % context)
    return _canonical_human_declarations(
        tuple(HumanDeclarationAuthority.from_dict(item) for item in value),
        context,
    )


def _candidate_authority_receipt_id(identity_payload: Mapping[str, Any]) -> str:
    return "CAR-" + canonical_sha256(identity_payload)


@dataclass(frozen=True)
class CandidateAuthorityReceipt(_JsonModel):
    candidate_id: str
    authority_repository_key: str
    locator_repository_key: str
    origin: ProducerType
    review_id: str
    proposal_head_sha: str
    authorized_source_refs: Tuple[SourceRef, ...]
    human_declarations: Tuple[HumanDeclarationAuthority, ...]
    initial_validation_report_hash: str
    authority_resolution_hash: str
    binding_id: Optional[str]
    created_at: str
    schema_version: int = MODEL_SCHEMA_VERSION
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "candidate_authority_receipt")
        object.__setattr__(
            self,
            "candidate_id",
            validate_stable_id(self.candidate_id, "MC", "candidate_id"),
        )
        object.__setattr__(
            self,
            "authority_repository_key",
            _normalize_repository_key(
                self.authority_repository_key,
                "authority_repository_key",
            ),
        )
        object.__setattr__(
            self,
            "locator_repository_key",
            _normalize_repository_key(
                self.locator_repository_key,
                "locator_repository_key",
            ),
        )
        if not isinstance(self.origin, ProducerType):
            raise ValueError("origin must be a ProducerType")
        object.__setattr__(
            self,
            "review_id",
            _normalize_identifier(self.review_id, "review_id"),
        )
        object.__setattr__(
            self,
            "proposal_head_sha",
            _git_object_id(self.proposal_head_sha, "proposal_head_sha"),
        )
        object.__setattr__(
            self,
            "authorized_source_refs",
            _canonical_source_refs(
                self.authorized_source_refs,
                "authorized_source_refs",
            ),
        )
        object.__setattr__(
            self,
            "human_declarations",
            _canonical_human_declarations(
                self.human_declarations,
                "human_declarations",
            ),
        )
        object.__setattr__(
            self,
            "initial_validation_report_hash",
            _sha256_digest(
                self.initial_validation_report_hash,
                "initial_validation_report_hash",
            ),
        )
        object.__setattr__(
            self,
            "authority_resolution_hash",
            _sha256_digest(
                self.authority_resolution_hash,
                "authority_resolution_hash",
            ),
        )

        is_direct = self.locator_repository_key == self.authority_repository_key
        if is_direct:
            if self.binding_id is not None:
                raise ValueError(
                    "direct authority receipt requires binding_id to be None"
                )
        else:
            if self.binding_id is None:
                raise ValueError(
                    "bound authority receipt requires a canonical binding_id"
                )
            object.__setattr__(
                self,
                "binding_id",
                validate_stable_id(self.binding_id, "RB", "binding_id"),
            )

        if self.origin is ProducerType.HUMAN:
            if not self.human_declarations:
                raise ValueError(
                    "HUMAN authority receipt requires at least one "
                    "human_declarations item"
                )

        # ``origin`` identifies who proposed the Candidate; declaration
        # authority belongs to individual SourceRefs.  A local/model Curator
        # may therefore propose a Candidate grounded in an explicit human
        # declaration, and the receipt must be able to carry that independently
        # validated declaration for later approval-time restoration.

        authorized_source_ref_json = {
            source_ref.to_json() for source_ref in self.authorized_source_refs
        }
        for declaration in self.human_declarations:
            if declaration.source_ref.to_json() not in authorized_source_ref_json:
                raise ValueError(
                    "human declaration source_ref must be present in "
                    "authorized_source_refs with matching hash, actor, and review"
                )
            declaration_review_id = declaration.source_ref.review_id
            if (
                declaration_review_id is not None
                and declaration_review_id != self.review_id
            ):
                raise ValueError(
                    "human declaration source_ref.review_id must match receipt "
                    "review_id"
                )

        object.__setattr__(
            self,
            "created_at",
            _utc_timestamp(self.created_at, "created_at"),
        )
        object.__setattr__(
            self,
            "receipt_id",
            _candidate_authority_receipt_id(self._identity_payload()),
        )

    def _identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "authority_repository_key": self.authority_repository_key,
            "locator_repository_key": self.locator_repository_key,
            "origin": self.origin.value,
            "review_id": self.review_id,
            "proposal_head_sha": self.proposal_head_sha,
            "authorized_source_refs": [
                item.to_dict() for item in self.authorized_source_refs
            ],
            "human_declarations": [
                item.to_dict() for item in self.human_declarations
            ],
            "initial_validation_report_hash": self.initial_validation_report_hash,
            "authority_resolution_hash": self.authority_resolution_hash,
            "binding_id": self.binding_id,
            "created_at": self.created_at,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "candidate_id": self.candidate_id,
            "authority_repository_key": self.authority_repository_key,
            "locator_repository_key": self.locator_repository_key,
            "origin": self.origin.value,
            "review_id": self.review_id,
            "proposal_head_sha": self.proposal_head_sha,
            "authorized_source_refs": [
                item.to_dict() for item in self.authorized_source_refs
            ],
            "human_declarations": [
                item.to_dict() for item in self.human_declarations
            ],
            "initial_validation_report_hash": self.initial_validation_report_hash,
            "authority_resolution_hash": self.authority_resolution_hash,
            "binding_id": self.binding_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "CandidateAuthorityReceipt":
        root = _object(payload, "candidate_authority_receipt")
        _exact_fields(
            root,
            {
                "schema_version",
                "receipt_id",
                "candidate_id",
                "authority_repository_key",
                "locator_repository_key",
                "origin",
                "review_id",
                "proposal_head_sha",
                "authorized_source_refs",
                "human_declarations",
                "initial_validation_report_hash",
                "authority_resolution_hash",
                "binding_id",
                "created_at",
            },
            "candidate_authority_receipt",
        )
        _validate_schema(root["schema_version"], "candidate_authority_receipt")
        receipt = cls(
            candidate_id=root["candidate_id"],
            authority_repository_key=root["authority_repository_key"],
            locator_repository_key=root["locator_repository_key"],
            origin=_enum_value(
                ProducerType,
                root["origin"],
                "candidate_authority_receipt.origin",
            ),
            review_id=root["review_id"],
            proposal_head_sha=root["proposal_head_sha"],
            authorized_source_refs=_source_refs_from_payload(
                root["authorized_source_refs"],
                "candidate_authority_receipt.authorized_source_refs",
            ),
            human_declarations=_human_declarations_from_payload(
                root["human_declarations"],
                "candidate_authority_receipt.human_declarations",
            ),
            initial_validation_report_hash=root["initial_validation_report_hash"],
            authority_resolution_hash=root["authority_resolution_hash"],
            binding_id=root["binding_id"],
            created_at=root["created_at"],
            schema_version=root["schema_version"],
        )
        validate_stable_id(
            root["receipt_id"],
            "CAR",
            "candidate_authority_receipt.receipt_id",
        )
        if root["receipt_id"] != receipt.receipt_id:
            raise ValueError(
                "candidate_authority_receipt.receipt_id does not match "
                "canonical authority identity"
            )
        return receipt


@dataclass(frozen=True)
class MemoryScope(_JsonModel):
    paths: Tuple[str, ...] = field(default_factory=tuple)
    symbols: Tuple[str, ...] = field(default_factory=tuple)
    contracts: Tuple[str, ...] = field(default_factory=tuple)
    languages: Tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "memory_scope")
        object.__setattr__(
            self,
            "paths",
            _canonical_string_tuple(
                self.paths,
                "paths",
                normalizer=lambda item, name: _normalize_repo_path(item, name, allow_glob=True),
                max_items=MAX_SCOPE_ITEMS,
            ),
        )
        object.__setattr__(
            self,
            "symbols",
            _canonical_string_tuple(
                self.symbols,
                "symbols",
                normalizer=_normalize_identifier,
                max_items=MAX_SCOPE_ITEMS,
            ),
        )
        object.__setattr__(
            self,
            "contracts",
            _canonical_string_tuple(
                self.contracts,
                "contracts",
                normalizer=lambda item, name: _normalize_token(item, name, casefold=True),
                max_items=MAX_SCOPE_ITEMS,
            ),
        )
        object.__setattr__(
            self,
            "languages",
            _canonical_string_tuple(
                self.languages,
                "languages",
                normalizer=lambda item, name: _normalize_token(item, name, casefold=True),
                max_items=MAX_SCOPE_ITEMS,
            ),
        )

    @property
    def is_empty(self) -> bool:
        return not (self.paths or self.symbols or self.contracts or self.languages)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "paths": list(self.paths),
            "symbols": list(self.symbols),
            "contracts": list(self.contracts),
            "languages": list(self.languages),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "MemoryScope":
        root = _object(payload, "memory_scope")
        _exact_fields(
            root,
            {"schema_version", "paths", "symbols", "contracts", "languages"},
            "memory_scope",
        )
        _validate_schema(root["schema_version"], "memory_scope")
        for name in ("paths", "symbols", "contracts", "languages"):
            if not isinstance(root[name], list):
                raise ValueError("memory_scope.%s must be a list" % name)
        return cls(
            paths=tuple(root["paths"]),
            symbols=tuple(root["symbols"]),
            contracts=tuple(root["contracts"]),
            languages=tuple(root["languages"]),
            schema_version=root["schema_version"],
        )


def _validate_scope_for_kind(scope: MemoryScope, kind: MemoryKind) -> None:
    if not isinstance(scope, MemoryScope):
        raise ValueError("scope must be a MemoryScope")
    if scope.is_empty and kind not in {
        MemoryKind.REVIEW_RULE,
        MemoryKind.COMPATIBILITY_REQUIREMENT,
    }:
        raise ValueError(
            "scope must not be empty unless kind is review_rule or compatibility_requirement"
        )


@dataclass(frozen=True)
class PolicyEffect(_JsonModel):
    effect_kind: PolicyEffectKind
    value: str
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "policy_effect")
        if not isinstance(self.effect_kind, PolicyEffectKind):
            raise ValueError("effect_kind must be a PolicyEffectKind")
        if self.effect_kind is PolicyEffectKind.RISK_FLOOR:
            normalized = _normalize_token(self.value, "policy_effect.value", casefold=True)
            if normalized not in {"low", "medium", "high", "critical"}:
                raise ValueError("policy_effect.value must be a supported risk level")
        else:
            normalized = _normalize_token(self.value, "policy_effect.value")
        object.__setattr__(self, "value", normalized)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "type": self.effect_kind.value,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "PolicyEffect":
        root = _object(payload, "policy_effect")
        _exact_fields(root, {"schema_version", "type", "value"}, "policy_effect")
        _validate_schema(root["schema_version"], "policy_effect")
        return cls(
            effect_kind=_enum_value(PolicyEffectKind, root["type"], "policy_effect.type"),
            value=root["value"],
            schema_version=root["schema_version"],
        )


@dataclass(frozen=True)
class ExpiryCondition(_JsonModel):
    """A closed, immutable approval-time expiry predicate."""

    condition_kind: ExpiryConditionKind
    value: str
    schema_version: int = MODEL_SCHEMA_VERSION
    condition_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "expiry_condition")
        if not isinstance(self.condition_kind, ExpiryConditionKind):
            raise ValueError("condition_kind must be an ExpiryConditionKind")
        if self.condition_kind is ExpiryConditionKind.AT_TIME:
            normalized = _canonical_utc_timestamp(
                self.value,
                "expiry_condition.value",
            )
        else:
            normalized = _git_object_id(self.value, "expiry_condition.value")
        object.__setattr__(self, "value", normalized)
        object.__setattr__(
            self,
            "condition_fingerprint",
            canonical_sha256(self._identity_dict()),
        )

    @property
    def kind(self) -> ExpiryConditionKind:
        return self.condition_kind

    @property
    def fingerprint(self) -> str:
        return self.condition_fingerprint

    @property
    def expires_at(self) -> Optional[str]:
        if self.condition_kind is ExpiryConditionKind.AT_TIME:
            return self.value
        return None

    @property
    def commit_sha(self) -> Optional[str]:
        if self.condition_kind is ExpiryConditionKind.AT_COMMIT:
            return self.value
        return None

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "type": self.condition_kind.value,
            "value": self.value,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_fingerprint": self.condition_fingerprint,
            **self._identity_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "ExpiryCondition":
        root = _object(payload, "expiry_condition")
        _exact_fields(
            root,
            {"schema_version", "condition_fingerprint", "type", "value"},
            "expiry_condition",
        )
        _validate_schema(root["schema_version"], "expiry_condition")
        condition = cls(
            condition_kind=_enum_value(
                ExpiryConditionKind,
                root["type"],
                "expiry_condition.type",
            ),
            value=root["value"],
            schema_version=root["schema_version"],
        )
        if root["value"] != condition.value:
            raise ValueError("expiry_condition.value must use canonical form")
        if root["condition_fingerprint"] != condition.condition_fingerprint:
            raise ValueError(
                "expiry_condition.condition_fingerprint does not match canonical content"
            )
        return condition


@dataclass(frozen=True)
class Producer(_JsonModel):
    producer_type: ProducerType
    name: str
    version: str
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.producer_type, ProducerType):
            raise ValueError("producer_type must be a ProducerType")
        if type(self.schema_version) is not int or not (1 <= self.schema_version <= 1_000_000):
            raise ValueError("producer.schema_version must be a positive bounded integer")
        object.__setattr__(self, "name", _normalize_token(self.name, "producer.name"))
        object.__setattr__(self, "version", _normalize_identifier(self.version, "producer.version"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.producer_type.value,
            "name": self.name,
            "version": self.version,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "Producer":
        root = _object(payload, "producer")
        _exact_fields(root, {"type", "name", "version", "schema_version"}, "producer")
        return cls(
            producer_type=_enum_value(ProducerType, root["type"], "producer.type"),
            name=root["name"],
            version=root["version"],
            schema_version=root["schema_version"],
        )


def _validity_policy_tuple(value: Any, context: str) -> Tuple[ValidityPolicy, ...]:
    policies = _enum_tuple(
        value,
        ValidityPolicy,
        context,
        max_items=len(ValidityPolicy),
    )
    if ValidityPolicy.MANUAL_UNTIL_REVOKED in policies and len(policies) != 1:
        raise ValueError("manual_until_revoked cannot be combined with other validity policies")
    return policies


def _enum_values_from_payload(
    value: Any,
    enum_type: Type[EnumT],
    context: str,
) -> Tuple[EnumT, ...]:
    if not isinstance(value, list):
        raise ValueError("%s must be a list" % context)
    return tuple(
        _enum_value(enum_type, item, "%s item" % context) for item in value
    )


def _content_type(value: Any, context: str) -> str:
    normalized = _normalize_identifier(value, context).casefold()
    if not _CONTENT_TYPE_PATTERN.fullmatch(normalized):
        raise ValueError("%s must be a valid media type without parameters" % context)
    return normalized


def _canonical_stable_ids(
    values: Any,
    prefix: str,
    context: str,
    *,
    max_items: int,
    allow_empty: bool = True,
) -> Tuple[str, ...]:
    return _canonical_string_tuple(
        values,
        context,
        normalizer=lambda item, name: validate_stable_id(item, prefix, name),
        max_items=max_items,
        allow_empty=allow_empty,
    )


@dataclass(frozen=True)
class MemoryCandidate(_JsonModel):
    repository_key: str
    kind: MemoryKind
    statement: str
    scope: MemoryScope
    source_refs: Tuple[SourceRef, ...]
    valid_from_sha: str
    validity_policies: Tuple[ValidityPolicy, ...]
    confidence: MemoryConfidence
    sensitivity: Sensitivity
    policy_effect: Optional[PolicyEffect]
    producer: Producer
    origin_review_id: str
    status: CandidateStatus
    created_at: str
    schema_version: int = MODEL_SCHEMA_VERSION
    candidate_id: str = field(init=False)
    content_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "memory_candidate")
        object.__setattr__(
            self,
            "repository_key",
            _normalize_repository_key(self.repository_key),
        )
        if not isinstance(self.kind, MemoryKind):
            raise ValueError("kind must be a MemoryKind")
        object.__setattr__(
            self,
            "statement",
            _normalize_text(
                self.statement,
                "statement",
                max_length=MAX_STATEMENT_LENGTH,
            ),
        )
        _validate_scope_for_kind(self.scope, self.kind)
        object.__setattr__(
            self,
            "source_refs",
            _canonical_source_refs(self.source_refs, "source_refs"),
        )
        object.__setattr__(
            self,
            "valid_from_sha",
            _git_object_id(self.valid_from_sha, "valid_from_sha"),
        )
        object.__setattr__(
            self,
            "validity_policies",
            _validity_policy_tuple(self.validity_policies, "validity_policies"),
        )
        if not isinstance(self.confidence, MemoryConfidence):
            raise ValueError("confidence must be a MemoryConfidence")
        if not isinstance(self.sensitivity, Sensitivity):
            raise ValueError("sensitivity must be a Sensitivity")
        if self.policy_effect is not None and not isinstance(self.policy_effect, PolicyEffect):
            raise ValueError("policy_effect must be a PolicyEffect or None")
        if not isinstance(self.producer, Producer):
            raise ValueError("producer must be a Producer")
        object.__setattr__(
            self,
            "origin_review_id",
            _normalize_identifier(self.origin_review_id, "origin_review_id"),
        )
        if not isinstance(self.status, CandidateStatus):
            raise ValueError("status must be a CandidateStatus")
        object.__setattr__(self, "created_at", _utc_timestamp(self.created_at, "created_at"))

        identity = {
            "schema_version": self.schema_version,
            "repository_key": self.repository_key,
            "kind": self.kind.value,
            "statement": self.statement,
            "scope": self.scope.to_dict(),
            "source_refs": [item.to_dict() for item in self.source_refs],
            "valid_from_sha": self.valid_from_sha,
            "validity_policies": [item.value for item in self.validity_policies],
            "confidence": self.confidence.value,
            "sensitivity": self.sensitivity.value,
            "policy_effect": (
                None if self.policy_effect is None else self.policy_effect.to_dict()
            ),
            "producer_schema_version": self.producer.schema_version,
        }
        fingerprint = {
            "schema_version": self.schema_version,
            "repository_key": self.repository_key,
            "kind": self.kind.value,
            "statement": self.statement,
            "scope": self.scope.to_dict(),
            "validity_policies": [item.value for item in self.validity_policies],
            "sensitivity": self.sensitivity.value,
            "policy_effect": (
                None if self.policy_effect is None else self.policy_effect.to_dict()
            ),
        }
        object.__setattr__(self, "candidate_id", "MC-" + canonical_sha256(identity))
        object.__setattr__(self, "content_fingerprint", canonical_sha256(fingerprint))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "content_fingerprint": self.content_fingerprint,
            "repository_key": self.repository_key,
            "kind": self.kind.value,
            "statement": self.statement,
            "scope": self.scope.to_dict(),
            "source_refs": [item.to_dict() for item in self.source_refs],
            "valid_from_sha": self.valid_from_sha,
            "validity_policies": [item.value for item in self.validity_policies],
            "confidence": self.confidence.value,
            "sensitivity": self.sensitivity.value,
            "policy_effect": (
                None if self.policy_effect is None else self.policy_effect.to_dict()
            ),
            "producer": self.producer.to_dict(),
            "origin_review_id": self.origin_review_id,
            "status": self.status.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "MemoryCandidate":
        root = _object(payload, "memory_candidate")
        _exact_fields(
            root,
            {
                "schema_version",
                "candidate_id",
                "content_fingerprint",
                "repository_key",
                "kind",
                "statement",
                "scope",
                "source_refs",
                "valid_from_sha",
                "validity_policies",
                "confidence",
                "sensitivity",
                "policy_effect",
                "producer",
                "origin_review_id",
                "status",
                "created_at",
            },
            "memory_candidate",
        )
        _validate_schema(root["schema_version"], "memory_candidate")
        policy_effect = (
            None
            if root["policy_effect"] is None
            else PolicyEffect.from_dict(root["policy_effect"])
        )
        candidate = cls(
            repository_key=root["repository_key"],
            kind=_enum_value(MemoryKind, root["kind"], "memory_candidate.kind"),
            statement=root["statement"],
            scope=MemoryScope.from_dict(root["scope"]),
            source_refs=_source_refs_from_payload(
                root["source_refs"], "memory_candidate.source_refs"
            ),
            valid_from_sha=root["valid_from_sha"],
            validity_policies=_enum_values_from_payload(
                root["validity_policies"],
                ValidityPolicy,
                "memory_candidate.validity_policies",
            ),
            confidence=_enum_value(
                MemoryConfidence,
                root["confidence"],
                "memory_candidate.confidence",
            ),
            sensitivity=_enum_value(
                Sensitivity,
                root["sensitivity"],
                "memory_candidate.sensitivity",
            ),
            policy_effect=policy_effect,
            producer=Producer.from_dict(root["producer"]),
            origin_review_id=root["origin_review_id"],
            status=_enum_value(
                CandidateStatus,
                root["status"],
                "memory_candidate.status",
            ),
            created_at=root["created_at"],
            schema_version=root["schema_version"],
        )
        if root["candidate_id"] != candidate.candidate_id:
            raise ValueError("memory_candidate.candidate_id does not match canonical identity")
        if root["content_fingerprint"] != candidate.content_fingerprint:
            raise ValueError(
                "memory_candidate.content_fingerprint does not match canonical content"
            )
        return candidate


@dataclass(frozen=True)
class SourceBundleDescriptor(_JsonModel):
    repository_key: str
    candidate_id: str
    source_refs: Tuple[SourceRef, ...]
    blob_hash: str
    size_bytes: int
    media_type: str
    created_at: str
    schema_version: int = MODEL_SCHEMA_VERSION
    bundle_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "source_bundle")
        object.__setattr__(self, "repository_key", _normalize_repository_key(self.repository_key))
        object.__setattr__(
            self,
            "candidate_id",
            validate_stable_id(self.candidate_id, "MC", "candidate_id"),
        )
        object.__setattr__(
            self,
            "source_refs",
            _canonical_source_refs(self.source_refs, "source_refs"),
        )
        object.__setattr__(self, "blob_hash", _sha256_digest(self.blob_hash, "blob_hash"))
        _positive_int(self.size_bytes, "size_bytes", allow_zero=True)
        object.__setattr__(self, "media_type", _content_type(self.media_type, "media_type"))
        object.__setattr__(self, "created_at", _utc_timestamp(self.created_at, "created_at"))
        identity = {
            "schema_version": self.schema_version,
            "repository_key": self.repository_key,
            "candidate_id": self.candidate_id,
            "source_refs": [item.to_dict() for item in self.source_refs],
            "blob_hash": self.blob_hash,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "created_at": self.created_at,
        }
        object.__setattr__(self, "bundle_hash", canonical_sha256(identity))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_hash": self.bundle_hash,
            "repository_key": self.repository_key,
            "candidate_id": self.candidate_id,
            "source_refs": [item.to_dict() for item in self.source_refs],
            "blob_hash": self.blob_hash,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "SourceBundleDescriptor":
        root = _object(payload, "source_bundle")
        _exact_fields(
            root,
            {
                "schema_version",
                "bundle_hash",
                "repository_key",
                "candidate_id",
                "source_refs",
                "blob_hash",
                "size_bytes",
                "media_type",
                "created_at",
            },
            "source_bundle",
        )
        descriptor = cls(
            repository_key=root["repository_key"],
            candidate_id=root["candidate_id"],
            source_refs=_source_refs_from_payload(
                root["source_refs"], "source_bundle.source_refs"
            ),
            blob_hash=root["blob_hash"],
            size_bytes=root["size_bytes"],
            media_type=root["media_type"],
            created_at=root["created_at"],
            schema_version=root["schema_version"],
        )
        if root["bundle_hash"] != descriptor.bundle_hash:
            raise ValueError("source_bundle.bundle_hash does not match canonical content")
        return descriptor


_DURABLE_MEMORY_RECORD_SCHEMA_V1 = 1
_DURABLE_MEMORY_RECORD_SCHEMA_V2 = 2
_AUTO_DURABLE_MEMORY_RECORD_SCHEMA = object()


def _durable_memory_record_schema(value: Any) -> int:
    if type(value) is not int or value not in {
        _DURABLE_MEMORY_RECORD_SCHEMA_V1,
        _DURABLE_MEMORY_RECORD_SCHEMA_V2,
    }:
        raise ValueError("durable_memory_record.schema_version must be 1 or 2")
    return value


def _canonical_expiry_conditions(
    values: Any,
    context: str,
) -> Tuple[ExpiryCondition, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError("%s must be a list or tuple of ExpiryCondition values" % context)
    if len(values) > len(ExpiryConditionKind):
        raise ValueError("%s must contain at most two conditions" % context)
    by_kind: Dict[ExpiryConditionKind, ExpiryCondition] = {}
    for condition in values:
        if type(condition) is not ExpiryCondition:
            raise ValueError("%s items must be ExpiryCondition values" % context)
        if condition.condition_kind in by_kind:
            raise ValueError(
                "%s must contain at most one condition of each kind" % context
            )
        by_kind[condition.condition_kind] = condition
    return tuple(by_kind[kind] for kind in sorted(by_kind, key=lambda item: item.value))


def _expiry_conditions_from_payload(
    value: Any,
    context: str,
) -> Tuple[ExpiryCondition, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("%s must be a list or tuple" % context)
    return _canonical_expiry_conditions(
        tuple(ExpiryCondition.from_dict(item) for item in value),
        context,
    )


@dataclass(frozen=True)
class DurableMemoryRecord(_JsonModel):
    """An immutable approved record with disjunctive approval-time expiry.

    The model only preserves canonical predicates.  Runtime owns evaluation and
    expires the record when any one of them matches.
    """

    candidate_id: str
    repository_key: str
    kind: MemoryKind
    statement: str
    scope: MemoryScope
    source_refs: Tuple[SourceRef, ...]
    source_bundle_hash: str
    valid_from_sha: str
    validity_policies: Tuple[ValidityPolicy, ...]
    confidence: MemoryConfidence
    sensitivity: Sensitivity
    policy_effect: Optional[PolicyEffect]
    approved_by: str
    approval_event_id: str
    status: RecordStatus
    created_at: str
    schema_version: int = field(
        default=cast(int, _AUTO_DURABLE_MEMORY_RECORD_SCHEMA)
    )
    expiry_conditions: Tuple[ExpiryCondition, ...] = ()
    memory_id: str = field(init=False)

    def __post_init__(self) -> None:
        raw_schema_version = self.schema_version
        if raw_schema_version is _AUTO_DURABLE_MEMORY_RECORD_SCHEMA:
            expiry_conditions = _canonical_expiry_conditions(
                self.expiry_conditions,
                "expiry_conditions",
            )
            schema_version = (
                _DURABLE_MEMORY_RECORD_SCHEMA_V2
                if expiry_conditions
                else _DURABLE_MEMORY_RECORD_SCHEMA_V1
            )
        else:
            schema_version = _durable_memory_record_schema(raw_schema_version)
            expiry_conditions = _canonical_expiry_conditions(
                self.expiry_conditions,
                "expiry_conditions",
            )
        if schema_version == _DURABLE_MEMORY_RECORD_SCHEMA_V1 and expiry_conditions:
            raise ValueError(
                "durable_memory_record schema_version 1 must not carry expiry_conditions"
            )
        if schema_version == _DURABLE_MEMORY_RECORD_SCHEMA_V2 and not expiry_conditions:
            raise ValueError(
                "durable_memory_record schema_version 2 requires expiry_conditions"
            )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "expiry_conditions", expiry_conditions)
        object.__setattr__(
            self,
            "candidate_id",
            validate_stable_id(self.candidate_id, "MC", "candidate_id"),
        )
        object.__setattr__(self, "repository_key", _normalize_repository_key(self.repository_key))
        if not isinstance(self.kind, MemoryKind):
            raise ValueError("kind must be a MemoryKind")
        object.__setattr__(
            self,
            "statement",
            _normalize_text(self.statement, "statement", max_length=MAX_STATEMENT_LENGTH),
        )
        _validate_scope_for_kind(self.scope, self.kind)
        object.__setattr__(
            self,
            "source_refs",
            _canonical_source_refs(self.source_refs, "source_refs"),
        )
        object.__setattr__(
            self,
            "source_bundle_hash",
            _sha256_digest(self.source_bundle_hash, "source_bundle_hash"),
        )
        object.__setattr__(
            self,
            "valid_from_sha",
            _git_object_id(self.valid_from_sha, "valid_from_sha"),
        )
        object.__setattr__(
            self,
            "validity_policies",
            _validity_policy_tuple(self.validity_policies, "validity_policies"),
        )
        if not isinstance(self.confidence, MemoryConfidence):
            raise ValueError("confidence must be a MemoryConfidence")
        if not isinstance(self.sensitivity, Sensitivity):
            raise ValueError("sensitivity must be a Sensitivity")
        if self.policy_effect is not None and not isinstance(self.policy_effect, PolicyEffect):
            raise ValueError("policy_effect must be a PolicyEffect or None")
        object.__setattr__(self, "approved_by", _normalize_identifier(self.approved_by, "approved_by"))
        object.__setattr__(
            self,
            "approval_event_id",
            validate_stable_id(self.approval_event_id, "EVT", "approval_event_id"),
        )
        if not isinstance(self.status, RecordStatus):
            raise ValueError("status must be a RecordStatus")
        object.__setattr__(self, "created_at", _utc_timestamp(self.created_at, "created_at"))
        object.__setattr__(
            self,
            "memory_id",
            "MEM-" + hashlib.sha256(self.candidate_id.encode("utf-8")).hexdigest(),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "memory_id": self.memory_id,
            "candidate_id": self.candidate_id,
            "repository_key": self.repository_key,
            "kind": self.kind.value,
            "statement": self.statement,
            "scope": self.scope.to_dict(),
            "source_refs": [item.to_dict() for item in self.source_refs],
            "source_bundle_hash": self.source_bundle_hash,
            "valid_from_sha": self.valid_from_sha,
            "validity_policies": [item.value for item in self.validity_policies],
            "confidence": self.confidence.value,
            "sensitivity": self.sensitivity.value,
            "policy_effect": None if self.policy_effect is None else self.policy_effect.to_dict(),
            "approved_by": self.approved_by,
            "approval_event_id": self.approval_event_id,
            "status": self.status.value,
            "created_at": self.created_at,
        }
        if self.schema_version == _DURABLE_MEMORY_RECORD_SCHEMA_V2:
            payload["expiry_conditions"] = [
                item.to_dict() for item in self.expiry_conditions
            ]
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> "DurableMemoryRecord":
        root = _object(payload, "durable_memory_record")
        if "schema_version" not in root:
            raise ValueError(
                "durable_memory_record is missing required field(s): schema_version"
            )
        schema_version = _durable_memory_record_schema(root["schema_version"])
        expected_fields = {
            "schema_version",
            "memory_id",
            "candidate_id",
            "repository_key",
            "kind",
            "statement",
            "scope",
            "source_refs",
            "source_bundle_hash",
            "valid_from_sha",
            "validity_policies",
            "confidence",
            "sensitivity",
            "policy_effect",
            "approved_by",
            "approval_event_id",
            "status",
            "created_at",
        }
        if schema_version == _DURABLE_MEMORY_RECORD_SCHEMA_V2:
            expected_fields.add("expiry_conditions")
        _exact_fields(
            root,
            expected_fields,
            "durable_memory_record",
        )
        expiry_conditions = (
            ()
            if schema_version == _DURABLE_MEMORY_RECORD_SCHEMA_V1
            else _expiry_conditions_from_payload(
                root["expiry_conditions"],
                "durable_memory_record.expiry_conditions",
            )
        )
        record = cls(
            candidate_id=root["candidate_id"],
            repository_key=root["repository_key"],
            kind=_enum_value(MemoryKind, root["kind"], "durable_memory_record.kind"),
            statement=root["statement"],
            scope=MemoryScope.from_dict(root["scope"]),
            source_refs=_source_refs_from_payload(
                root["source_refs"], "durable_memory_record.source_refs"
            ),
            source_bundle_hash=root["source_bundle_hash"],
            valid_from_sha=root["valid_from_sha"],
            validity_policies=_enum_values_from_payload(
                root["validity_policies"],
                ValidityPolicy,
                "durable_memory_record.validity_policies",
            ),
            confidence=_enum_value(
                MemoryConfidence,
                root["confidence"],
                "durable_memory_record.confidence",
            ),
            sensitivity=_enum_value(
                Sensitivity,
                root["sensitivity"],
                "durable_memory_record.sensitivity",
            ),
            policy_effect=(
                None
                if root["policy_effect"] is None
                else PolicyEffect.from_dict(root["policy_effect"])
            ),
            approved_by=root["approved_by"],
            approval_event_id=root["approval_event_id"],
            status=_enum_value(
                RecordStatus,
                root["status"],
                "durable_memory_record.status",
            ),
            created_at=root["created_at"],
            schema_version=schema_version,
            expiry_conditions=expiry_conditions,
        )
        if root["memory_id"] != record.memory_id:
            raise ValueError("durable_memory_record.memory_id does not match candidate_id")
        return record


@dataclass(frozen=True)
class FindingSnapshot(_JsonModel):
    finding_id: str
    claim: str
    path: str
    line: int
    contracts: Tuple[str, ...]
    original_severity: FindingSeverity
    evidence_refs: Tuple[str, ...]
    schema_version: int = MODEL_SCHEMA_VERSION
    finding_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "finding_snapshot")
        object.__setattr__(self, "finding_id", _validate_finding_id(self.finding_id))
        object.__setattr__(
            self,
            "claim",
            _normalize_text(self.claim, "claim", max_length=MAX_STATEMENT_LENGTH),
        )
        object.__setattr__(self, "path", _normalize_repo_path(self.path, "path", allow_glob=False))
        _positive_int(self.line, "line")
        object.__setattr__(
            self,
            "contracts",
            _canonical_string_tuple(
                self.contracts,
                "contracts",
                normalizer=lambda item, name: _normalize_token(item, name, casefold=True),
                max_items=MAX_SCOPE_ITEMS,
            ),
        )
        if not isinstance(self.original_severity, FindingSeverity):
            raise ValueError("original_severity must be a FindingSeverity")
        object.__setattr__(
            self,
            "evidence_refs",
            _canonical_string_tuple(
                self.evidence_refs,
                "evidence_refs",
                normalizer=_validate_observation_id,
                max_items=MAX_EVIDENCE_REFS,
                allow_empty=False,
            ),
        )

        object.__setattr__(self, "finding_hash", canonical_sha256(self._identity_dict()))

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "finding_id": self.finding_id,
            "claim": self.claim,
            "path": self.path,
            "line": self.line,
            "contracts": list(self.contracts),
            "original_severity": self.original_severity.value,
            "evidence_refs": list(self.evidence_refs),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"finding_hash": self.finding_hash, **self._identity_dict()}

    @classmethod
    def from_dict(cls, payload: Any) -> "FindingSnapshot":
        root = _object(payload, "finding_snapshot")
        _exact_fields(
            root,
            {
                "schema_version",
                "finding_id",
                "claim",
                "path",
                "line",
                "contracts",
                "original_severity",
                "evidence_refs",
                "finding_hash",
            },
            "finding_snapshot",
        )
        for name in ("contracts", "evidence_refs"):
            if not isinstance(root[name], list):
                raise ValueError("finding_snapshot.%s must be a list" % name)
        finding = cls(
            finding_id=root["finding_id"],
            claim=root["claim"],
            path=root["path"],
            line=root["line"],
            contracts=tuple(root["contracts"]),
            original_severity=_enum_value(
                FindingSeverity,
                root["original_severity"],
                "finding_snapshot.original_severity",
            ),
            evidence_refs=tuple(root["evidence_refs"]),
            schema_version=root["schema_version"],
        )
        if root["finding_hash"] != finding.finding_hash:
            raise ValueError("finding_snapshot.finding_hash does not match canonical content")
        return finding


@dataclass(frozen=True)
class FeedbackRecord(_JsonModel):
    repository_key: str
    review_id: str
    finding_id: str
    head_sha: str
    finding_snapshot: FindingSnapshot
    decision: FeedbackDecision
    original_severity: FindingSeverity
    final_severity: FindingSeverity
    reason_code: FeedbackReasonCode
    reason: str
    actor: str
    source_refs: Tuple[SourceRef, ...]
    status: FeedbackStatus
    created_at: str
    schema_version: int = MODEL_SCHEMA_VERSION
    feedback_id: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "feedback_record")
        object.__setattr__(self, "repository_key", _normalize_repository_key(self.repository_key))
        object.__setattr__(self, "review_id", _normalize_identifier(self.review_id, "review_id"))
        object.__setattr__(self, "finding_id", _validate_finding_id(self.finding_id))
        object.__setattr__(self, "head_sha", _git_object_id(self.head_sha, "head_sha"))
        if not isinstance(self.finding_snapshot, FindingSnapshot):
            raise ValueError("finding_snapshot must be a FindingSnapshot")
        if self.finding_snapshot.finding_id != self.finding_id:
            raise ValueError("finding_snapshot.finding_id must match finding_id")
        if not isinstance(self.decision, FeedbackDecision):
            raise ValueError("decision must be a FeedbackDecision")
        if not isinstance(self.original_severity, FindingSeverity):
            raise ValueError("original_severity must be a FindingSeverity")
        if not isinstance(self.final_severity, FindingSeverity):
            raise ValueError("final_severity must be a FindingSeverity")
        if self.finding_snapshot.original_severity is not self.original_severity:
            raise ValueError(
                "original_severity must match finding_snapshot.original_severity"
            )
        if (
            self.decision is FeedbackDecision.SEVERITY_CHANGED
            and self.final_severity is self.original_severity
        ):
            raise ValueError(
                "final_severity must differ from original_severity for severity_changed"
            )
        if (
            self.decision is not FeedbackDecision.SEVERITY_CHANGED
            and self.final_severity is not self.original_severity
        ):
            raise ValueError(
                "final_severity may differ only for a severity_changed decision"
            )
        if not isinstance(self.reason_code, FeedbackReasonCode):
            raise ValueError("reason_code must be a FeedbackReasonCode")
        object.__setattr__(
            self,
            "reason",
            _normalize_text(self.reason, "reason", max_length=MAX_REASON_LENGTH),
        )
        object.__setattr__(self, "actor", _normalize_identifier(self.actor, "actor"))
        object.__setattr__(
            self,
            "source_refs",
            _canonical_source_refs(self.source_refs, "source_refs"),
        )
        if not isinstance(self.status, FeedbackStatus):
            raise ValueError("status must be a FeedbackStatus")
        object.__setattr__(self, "created_at", _utc_timestamp(self.created_at, "created_at"))
        identity = {
            "schema_version": self.schema_version,
            "repository_key": self.repository_key,
            "review_id": self.review_id,
            "finding_id": self.finding_id,
            "head_sha": self.head_sha,
            "finding_snapshot": self.finding_snapshot.to_dict(),
            "decision": self.decision.value,
            "original_severity": self.original_severity.value,
            "final_severity": self.final_severity.value,
            "reason_code": self.reason_code.value,
            "reason": self.reason,
            "actor": self.actor,
            "source_refs": [item.to_dict() for item in self.source_refs],
        }
        object.__setattr__(self, "feedback_id", "FB-" + canonical_sha256(identity))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feedback_id": self.feedback_id,
            "repository_key": self.repository_key,
            "review_id": self.review_id,
            "finding_id": self.finding_id,
            "head_sha": self.head_sha,
            "finding_snapshot": self.finding_snapshot.to_dict(),
            "decision": self.decision.value,
            "original_severity": self.original_severity.value,
            "final_severity": self.final_severity.value,
            "reason_code": self.reason_code.value,
            "reason": self.reason,
            "actor": self.actor,
            "source_refs": [item.to_dict() for item in self.source_refs],
            "status": self.status.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "FeedbackRecord":
        root = _object(payload, "feedback_record")
        _exact_fields(
            root,
            {
                "schema_version",
                "feedback_id",
                "repository_key",
                "review_id",
                "finding_id",
                "head_sha",
                "finding_snapshot",
                "decision",
                "original_severity",
                "final_severity",
                "reason_code",
                "reason",
                "actor",
                "source_refs",
                "status",
                "created_at",
            },
            "feedback_record",
        )
        record = cls(
            repository_key=root["repository_key"],
            review_id=root["review_id"],
            finding_id=root["finding_id"],
            head_sha=root["head_sha"],
            finding_snapshot=FindingSnapshot.from_dict(root["finding_snapshot"]),
            decision=_enum_value(
                FeedbackDecision,
                root["decision"],
                "feedback_record.decision",
            ),
            original_severity=_enum_value(
                FindingSeverity,
                root["original_severity"],
                "feedback_record.original_severity",
            ),
            final_severity=_enum_value(
                FindingSeverity,
                root["final_severity"],
                "feedback_record.final_severity",
            ),
            reason_code=_enum_value(
                FeedbackReasonCode,
                root["reason_code"],
                "feedback_record.reason_code",
            ),
            reason=root["reason"],
            actor=root["actor"],
            source_refs=_source_refs_from_payload(
                root["source_refs"], "feedback_record.source_refs"
            ),
            status=_enum_value(
                FeedbackStatus,
                root["status"],
                "feedback_record.status",
            ),
            created_at=root["created_at"],
            schema_version=root["schema_version"],
        )
        if root["feedback_id"] != record.feedback_id:
            raise ValueError("feedback_record.feedback_id does not match canonical identity")
        return record


@dataclass(frozen=True)
class GenerationMetadata(_JsonModel):
    store_schema_version: int
    memory_generation: int
    feedback_generation: int
    knowledge_generation: int
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "generation_metadata")
        if (
            type(self.store_schema_version) is not int
            or self.store_schema_version not in SUPPORTED_MEMORY_STORE_SCHEMA_VERSIONS
        ):
            raise ValueError("store_schema_version must be a supported version")
        for name in (
            "memory_generation",
            "feedback_generation",
            "knowledge_generation",
        ):
            _positive_int(getattr(self, name), name, allow_zero=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "store_schema_version": self.store_schema_version,
            "memory_generation": self.memory_generation,
            "feedback_generation": self.feedback_generation,
            "knowledge_generation": self.knowledge_generation,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "GenerationMetadata":
        root = _object(payload, "generation_metadata")
        _exact_fields(
            root,
            {
                "schema_version",
                "store_schema_version",
                "memory_generation",
                "feedback_generation",
                "knowledge_generation",
            },
            "generation_metadata",
        )
        return cls(
            store_schema_version=root["store_schema_version"],
            memory_generation=root["memory_generation"],
            feedback_generation=root["feedback_generation"],
            knowledge_generation=root["knowledge_generation"],
            schema_version=root["schema_version"],
        )


@dataclass(frozen=True)
class MemorySelectionInput(_JsonModel):
    review_id: str
    repository_key: str
    base_sha: str
    head_sha: str
    changed_paths: Tuple[str, ...]
    changed_symbols: Tuple[str, ...]
    contracts: Tuple[str, ...]
    languages: Tuple[str, ...]
    generations: GenerationMetadata
    selection_policy_version: str = MEMORY_SELECTION_POLICY_VERSION
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "memory_selection_input")
        object.__setattr__(self, "review_id", _normalize_identifier(self.review_id, "review_id"))
        object.__setattr__(self, "repository_key", _normalize_repository_key(self.repository_key))
        object.__setattr__(self, "base_sha", _git_object_id(self.base_sha, "base_sha"))
        object.__setattr__(self, "head_sha", _git_object_id(self.head_sha, "head_sha"))
        object.__setattr__(
            self,
            "changed_paths",
            _canonical_string_tuple(
                self.changed_paths,
                "changed_paths",
                normalizer=lambda item, name: _normalize_repo_path(item, name, allow_glob=False),
                max_items=MAX_KNOWLEDGE_REFS,
            ),
        )
        object.__setattr__(
            self,
            "changed_symbols",
            _canonical_string_tuple(
                self.changed_symbols,
                "changed_symbols",
                normalizer=_normalize_identifier,
                max_items=MAX_KNOWLEDGE_REFS,
            ),
        )
        object.__setattr__(
            self,
            "contracts",
            _canonical_string_tuple(
                self.contracts,
                "contracts",
                normalizer=lambda item, name: _normalize_token(item, name, casefold=True),
                max_items=MAX_SCOPE_ITEMS,
            ),
        )
        object.__setattr__(
            self,
            "languages",
            _canonical_string_tuple(
                self.languages,
                "languages",
                normalizer=lambda item, name: _normalize_token(item, name, casefold=True),
                max_items=MAX_SCOPE_ITEMS,
            ),
        )
        if not isinstance(self.generations, GenerationMetadata):
            raise ValueError("generations must be GenerationMetadata")
        if (
            self.selection_policy_version
            not in SUPPORTED_MEMORY_SELECTION_POLICY_VERSIONS
        ):
            raise ValueError(
                "selection_policy_version must be a supported Memory selection policy"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "review_id": self.review_id,
            "repository_key": self.repository_key,
            "base_sha": self.base_sha,
            "head_sha": self.head_sha,
            "changed_paths": list(self.changed_paths),
            "changed_symbols": list(self.changed_symbols),
            "contracts": list(self.contracts),
            "languages": list(self.languages),
            "generations": self.generations.to_dict(),
            "selection_policy_version": self.selection_policy_version,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "MemorySelectionInput":
        root = _object(payload, "memory_selection_input")
        _exact_fields(
            root,
            {
                "schema_version",
                "review_id",
                "repository_key",
                "base_sha",
                "head_sha",
                "changed_paths",
                "changed_symbols",
                "contracts",
                "languages",
                "generations",
                "selection_policy_version",
            },
            "memory_selection_input",
        )
        for name in ("changed_paths", "changed_symbols", "contracts", "languages"):
            if not isinstance(root[name], list):
                raise ValueError("memory_selection_input.%s must be a list" % name)
        return cls(
            review_id=root["review_id"],
            repository_key=root["repository_key"],
            base_sha=root["base_sha"],
            head_sha=root["head_sha"],
            changed_paths=tuple(root["changed_paths"]),
            changed_symbols=tuple(root["changed_symbols"]),
            contracts=tuple(root["contracts"]),
            languages=tuple(root["languages"]),
            generations=GenerationMetadata.from_dict(root["generations"]),
            selection_policy_version=root["selection_policy_version"],
            schema_version=root["schema_version"],
        )


@dataclass(frozen=True)
class MemorySelectionDecision(_JsonModel):
    memory_id: str
    applicability: Applicability
    matched_scope: MemoryScope
    reason_codes: Tuple[str, ...]
    rank: int
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "memory_selection_decision")
        object.__setattr__(
            self,
            "memory_id",
            validate_stable_id(self.memory_id, "MEM", "memory_id"),
        )
        if not isinstance(self.applicability, Applicability):
            raise ValueError("applicability must be an Applicability")
        if not isinstance(self.matched_scope, MemoryScope):
            raise ValueError("matched_scope must be a MemoryScope")
        object.__setattr__(
            self,
            "reason_codes",
            _canonical_string_tuple(
                self.reason_codes,
                "reason_codes",
                normalizer=lambda item, name: _normalize_token(item, name, casefold=True),
                max_items=MAX_DECISION_REASONS,
                allow_empty=False,
            ),
        )
        _positive_int(self.rank, "rank", allow_zero=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "memory_id": self.memory_id,
            "applicability": self.applicability.value,
            "matched_scope": self.matched_scope.to_dict(),
            "reason_codes": list(self.reason_codes),
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "MemorySelectionDecision":
        root = _object(payload, "memory_selection_decision")
        _exact_fields(
            root,
            {
                "schema_version",
                "memory_id",
                "applicability",
                "matched_scope",
                "reason_codes",
                "rank",
            },
            "memory_selection_decision",
        )
        if not isinstance(root["reason_codes"], list):
            raise ValueError("memory_selection_decision.reason_codes must be a list")
        return cls(
            memory_id=root["memory_id"],
            applicability=_enum_value(
                Applicability,
                root["applicability"],
                "memory_selection_decision.applicability",
            ),
            matched_scope=MemoryScope.from_dict(root["matched_scope"]),
            reason_codes=tuple(root["reason_codes"]),
            rank=root["rank"],
            schema_version=root["schema_version"],
        )


@dataclass(frozen=True)
class RepositoryKnowledgeKey(_JsonModel):
    repository_key: str
    revision_binding: str
    capability: RepositoryKnowledgeCapability
    analyzer_name: str
    analyzer_version: str
    configuration_digest: str
    input_digest: str
    schema_version: int = MODEL_SCHEMA_VERSION
    key_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "repository_knowledge_key")
        object.__setattr__(self, "repository_key", _normalize_repository_key(self.repository_key))
        object.__setattr__(
            self,
            "revision_binding",
            _revision_binding(self.revision_binding, "revision_binding"),
        )
        if not isinstance(self.capability, RepositoryKnowledgeCapability):
            raise ValueError("capability must be a RepositoryKnowledgeCapability")
        object.__setattr__(
            self,
            "analyzer_name",
            _normalize_token(self.analyzer_name, "analyzer_name", casefold=True),
        )
        object.__setattr__(
            self,
            "analyzer_version",
            _normalize_token(self.analyzer_version, "analyzer_version"),
        )
        object.__setattr__(
            self,
            "configuration_digest",
            _sha256_digest(self.configuration_digest, "configuration_digest"),
        )
        object.__setattr__(
            self,
            "input_digest",
            _sha256_digest(self.input_digest, "input_digest"),
        )
        object.__setattr__(self, "key_hash", canonical_sha256(self._identity_dict()))

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository_key": self.repository_key,
            "revision_binding": self.revision_binding,
            "capability": self.capability.value,
            "analyzer_name": self.analyzer_name,
            "analyzer_version": self.analyzer_version,
            "configuration_digest": self.configuration_digest,
            "input_digest": self.input_digest,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"key_hash": self.key_hash, **self._identity_dict()}

    @classmethod
    def from_dict(cls, payload: Any) -> "RepositoryKnowledgeKey":
        root = _object(payload, "repository_knowledge_key")
        _exact_fields(
            root,
            {
                "schema_version",
                "key_hash",
                "repository_key",
                "revision_binding",
                "capability",
                "analyzer_name",
                "analyzer_version",
                "configuration_digest",
                "input_digest",
            },
            "repository_knowledge_key",
        )
        key = cls(
            repository_key=root["repository_key"],
            revision_binding=root["revision_binding"],
            capability=_enum_value(
                RepositoryKnowledgeCapability,
                root["capability"],
                "repository_knowledge_key.capability",
            ),
            analyzer_name=root["analyzer_name"],
            analyzer_version=root["analyzer_version"],
            configuration_digest=root["configuration_digest"],
            input_digest=root["input_digest"],
            schema_version=root["schema_version"],
        )
        if root["key_hash"] != key.key_hash:
            raise ValueError("repository_knowledge_key.key_hash does not match canonical key")
        return key


@dataclass(frozen=True)
class RepositoryKnowledgeEntry(_JsonModel):
    key: RepositoryKnowledgeKey
    blob_hash: str
    size_bytes: int
    content_type: str
    artifact_schema: str
    created_at: str
    summary_hash: Optional[str] = None
    pinned_by_review_ids: Tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = MODEL_SCHEMA_VERSION
    entry_id: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "repository_knowledge_entry")
        if not isinstance(self.key, RepositoryKnowledgeKey):
            raise ValueError("key must be a RepositoryKnowledgeKey")
        object.__setattr__(self, "blob_hash", _sha256_digest(self.blob_hash, "blob_hash"))
        _positive_int(self.size_bytes, "size_bytes", allow_zero=True)
        object.__setattr__(self, "content_type", _content_type(self.content_type, "content_type"))
        object.__setattr__(
            self,
            "artifact_schema",
            _normalize_token(self.artifact_schema, "artifact_schema", casefold=True),
        )
        if self.summary_hash is not None:
            object.__setattr__(
                self,
                "summary_hash",
                _sha256_digest(self.summary_hash, "summary_hash"),
            )
        object.__setattr__(self, "created_at", _utc_timestamp(self.created_at, "created_at"))
        object.__setattr__(
            self,
            "pinned_by_review_ids",
            _canonical_string_tuple(
                self.pinned_by_review_ids,
                "pinned_by_review_ids",
                normalizer=_normalize_identifier,
                max_items=MAX_PINNED_REVIEWS,
            ),
        )
        identity = {
            "schema_version": self.schema_version,
            "key_hash": self.key.key_hash,
            "blob_hash": self.blob_hash,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "artifact_schema": self.artifact_schema,
            "summary_hash": self.summary_hash,
        }
        object.__setattr__(self, "entry_id", stable_id("RKE", identity))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entry_id": self.entry_id,
            "key": self.key.to_dict(),
            "blob_hash": self.blob_hash,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "artifact_schema": self.artifact_schema,
            "summary_hash": self.summary_hash,
            "created_at": self.created_at,
            "pinned_by_review_ids": list(self.pinned_by_review_ids),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "RepositoryKnowledgeEntry":
        root = _object(payload, "repository_knowledge_entry")
        _exact_fields(
            root,
            {
                "schema_version",
                "entry_id",
                "key",
                "blob_hash",
                "size_bytes",
                "content_type",
                "artifact_schema",
                "summary_hash",
                "created_at",
                "pinned_by_review_ids",
            },
            "repository_knowledge_entry",
        )
        if not isinstance(root["pinned_by_review_ids"], list):
            raise ValueError("repository_knowledge_entry.pinned_by_review_ids must be a list")
        entry = cls(
            key=RepositoryKnowledgeKey.from_dict(root["key"]),
            blob_hash=root["blob_hash"],
            size_bytes=root["size_bytes"],
            content_type=root["content_type"],
            artifact_schema=root["artifact_schema"],
            summary_hash=root["summary_hash"],
            created_at=root["created_at"],
            pinned_by_review_ids=tuple(root["pinned_by_review_ids"]),
            schema_version=root["schema_version"],
        )
        if root["entry_id"] != entry.entry_id:
            raise ValueError("repository_knowledge_entry.entry_id does not match canonical entry")
        return entry


@dataclass(frozen=True)
class FeedbackCalibrationSignal(_JsonModel):
    signal_kind: FeedbackCalibrationSignalKind
    scope: MemoryScope
    message: str
    sample_count: int
    review_count: int
    feedback_ids: Tuple[str, ...]
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "feedback_calibration_signal")
        if not isinstance(self.signal_kind, FeedbackCalibrationSignalKind):
            raise ValueError("signal_kind must be a FeedbackCalibrationSignalKind")
        if not isinstance(self.scope, MemoryScope):
            raise ValueError("scope must be a MemoryScope")
        object.__setattr__(
            self,
            "message",
            _normalize_text(self.message, "message", max_length=MAX_REASON_LENGTH),
        )
        _positive_int(self.sample_count, "sample_count")
        _positive_int(self.review_count, "review_count")
        if self.review_count > self.sample_count:
            raise ValueError("review_count must not exceed sample_count")
        object.__setattr__(
            self,
            "feedback_ids",
            _canonical_stable_ids(
                self.feedback_ids,
                "FB",
                "feedback_ids",
                max_items=MAX_FEEDBACK_SOURCES,
                allow_empty=False,
            ),
        )
        if self.sample_count != len(self.feedback_ids):
            raise ValueError("sample_count must equal the number of feedback_ids")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "signal_kind": self.signal_kind.value,
            "scope": self.scope.to_dict(),
            "message": self.message,
            "sample_count": self.sample_count,
            "review_count": self.review_count,
            "feedback_ids": list(self.feedback_ids),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "FeedbackCalibrationSignal":
        root = _object(payload, "feedback_calibration_signal")
        _exact_fields(
            root,
            {
                "schema_version",
                "signal_kind",
                "scope",
                "message",
                "sample_count",
                "review_count",
                "feedback_ids",
            },
            "feedback_calibration_signal",
        )
        if not isinstance(root["feedback_ids"], list):
            raise ValueError("feedback_calibration_signal.feedback_ids must be a list")
        return cls(
            signal_kind=_enum_value(
                FeedbackCalibrationSignalKind,
                root["signal_kind"],
                "feedback_calibration_signal.signal_kind",
            ),
            scope=MemoryScope.from_dict(root["scope"]),
            message=root["message"],
            sample_count=root["sample_count"],
            review_count=root["review_count"],
            feedback_ids=tuple(root["feedback_ids"]),
            schema_version=root["schema_version"],
        )


def _decision_count_tuple(value: Any) -> Tuple[Tuple[FeedbackDecision, int], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("decision_counts must be a list or tuple")
    if len(value) > len(FeedbackDecision):
        raise ValueError("decision_counts contains too many items")
    counts: Dict[str, Tuple[FeedbackDecision, int]] = {}
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("decision_counts items must be (FeedbackDecision, count) pairs")
        decision, count = item
        if not isinstance(decision, FeedbackDecision):
            raise ValueError("decision_counts decision must be a FeedbackDecision")
        _positive_int(count, "decision_counts count")
        if decision.value in counts:
            raise ValueError("decision_counts must not repeat a decision")
        counts[decision.value] = (decision, count)
    return tuple(counts[key] for key in sorted(counts))


@dataclass(frozen=True)
class FeedbackCalibrationSummary(_JsonModel):
    repository_key: str
    feedback_generation: int
    policy_version: str
    eligible: bool
    source_feedback_ids: Tuple[str, ...]
    source_review_ids: Tuple[str, ...]
    decision_counts: Tuple[Tuple[FeedbackDecision, int], ...]
    signals: Tuple[FeedbackCalibrationSignal, ...]
    created_at: str
    schema_version: int = MODEL_SCHEMA_VERSION
    summary_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "feedback_calibration_summary")
        object.__setattr__(self, "repository_key", _normalize_repository_key(self.repository_key))
        _positive_int(self.feedback_generation, "feedback_generation", allow_zero=True)
        if self.policy_version != FEEDBACK_AGGREGATION_POLICY_VERSION:
            raise ValueError(
                "policy_version must be %s" % FEEDBACK_AGGREGATION_POLICY_VERSION
            )
        if type(self.eligible) is not bool:
            raise ValueError("eligible must be a boolean")
        object.__setattr__(
            self,
            "source_feedback_ids",
            _canonical_stable_ids(
                self.source_feedback_ids,
                "FB",
                "source_feedback_ids",
                max_items=MAX_FEEDBACK_SOURCES,
            ),
        )
        object.__setattr__(
            self,
            "source_review_ids",
            _canonical_string_tuple(
                self.source_review_ids,
                "source_review_ids",
                normalizer=_normalize_identifier,
                max_items=MAX_FEEDBACK_SOURCES,
            ),
        )
        object.__setattr__(self, "decision_counts", _decision_count_tuple(self.decision_counts))
        if isinstance(self.signals, (str, bytes)) or not isinstance(self.signals, (list, tuple)):
            raise ValueError("signals must be a list or tuple")
        if len(self.signals) > MAX_CALIBRATION_SIGNALS:
            raise ValueError("signals exceeds the maximum item count")
        by_json: Dict[str, FeedbackCalibrationSignal] = {}
        for signal in self.signals:
            if not isinstance(signal, FeedbackCalibrationSignal):
                raise ValueError("signals items must be FeedbackCalibrationSignal values")
            by_json[signal.to_json()] = signal
        object.__setattr__(self, "signals", tuple(by_json[key] for key in sorted(by_json)))
        object.__setattr__(self, "created_at", _utc_timestamp(self.created_at, "created_at"))

        if sum(count for _, count in self.decision_counts) != len(self.source_feedback_ids):
            raise ValueError("decision_counts must account for every source_feedback_id")
        if self.eligible:
            if len(self.source_feedback_ids) < 5:
                raise ValueError("eligible feedback calibration requires at least 5 feedback records")
            if len(self.source_review_ids) < 3:
                raise ValueError("eligible feedback calibration requires at least 3 reviews")
        elif self.signals:
            raise ValueError("signals require an eligible feedback calibration summary")
        source_ids = set(self.source_feedback_ids)
        for signal in self.signals:
            if not set(signal.feedback_ids).issubset(source_ids):
                raise ValueError("signal feedback_ids must come from source_feedback_ids")
        object.__setattr__(self, "summary_hash", canonical_sha256(self._identity_dict()))

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository_key": self.repository_key,
            "feedback_generation": self.feedback_generation,
            "policy_version": self.policy_version,
            "eligible": self.eligible,
            "source_feedback_ids": list(self.source_feedback_ids),
            "source_review_ids": list(self.source_review_ids),
            "decision_counts": [
                {"decision": decision.value, "count": count}
                for decision, count in self.decision_counts
            ],
            "signals": [signal.to_dict() for signal in self.signals],
            "created_at": self.created_at,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"summary_hash": self.summary_hash, **self._identity_dict()}

    @classmethod
    def from_dict(cls, payload: Any) -> "FeedbackCalibrationSummary":
        root = _object(payload, "feedback_calibration_summary")
        _exact_fields(
            root,
            {
                "schema_version",
                "summary_hash",
                "repository_key",
                "feedback_generation",
                "policy_version",
                "eligible",
                "source_feedback_ids",
                "source_review_ids",
                "decision_counts",
                "signals",
                "created_at",
            },
            "feedback_calibration_summary",
        )
        for name in ("source_feedback_ids", "source_review_ids", "decision_counts", "signals"):
            if not isinstance(root[name], list):
                raise ValueError("feedback_calibration_summary.%s must be a list" % name)
        counts: List[Tuple[FeedbackDecision, int]] = []
        for index, item in enumerate(root["decision_counts"]):
            pair = _object(item, "feedback_calibration_summary.decision_counts[%d]" % index)
            _exact_fields(pair, {"decision", "count"}, "feedback_calibration_summary.decision_count")
            counts.append(
                (
                    _enum_value(
                        FeedbackDecision,
                        pair["decision"],
                        "feedback_calibration_summary.decision",
                    ),
                    pair["count"],
                )
            )
        summary = cls(
            repository_key=root["repository_key"],
            feedback_generation=root["feedback_generation"],
            policy_version=root["policy_version"],
            eligible=root["eligible"],
            source_feedback_ids=tuple(root["source_feedback_ids"]),
            source_review_ids=tuple(root["source_review_ids"]),
            decision_counts=tuple(counts),
            signals=tuple(FeedbackCalibrationSignal.from_dict(item) for item in root["signals"]),
            created_at=root["created_at"],
            schema_version=root["schema_version"],
        )
        if root["summary_hash"] != summary.summary_hash:
            raise ValueError("feedback_calibration_summary.summary_hash does not match canonical content")
        return summary


@dataclass(frozen=True)
class MemorySnapshot(_JsonModel):
    repository_key: str
    base_sha: str
    head_sha: str
    generations: GenerationMetadata
    selection_policy_version: str
    eligible_records: Tuple[DurableMemoryRecord, ...]
    applicability_decisions: Tuple[MemorySelectionDecision, ...]
    feedback_calibration_summary: Optional[FeedbackCalibrationSummary]
    repository_knowledge_refs: Tuple[str, ...]
    created_at: str
    schema_version: int = MODEL_SCHEMA_VERSION
    snapshot_id: str = field(init=False)
    snapshot_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "memory_snapshot")
        object.__setattr__(self, "repository_key", _normalize_repository_key(self.repository_key))
        object.__setattr__(self, "base_sha", _git_object_id(self.base_sha, "base_sha"))
        object.__setattr__(self, "head_sha", _git_object_id(self.head_sha, "head_sha"))
        if not isinstance(self.generations, GenerationMetadata):
            raise ValueError("generations must be GenerationMetadata")
        if (
            self.selection_policy_version
            not in SUPPORTED_MEMORY_SELECTION_POLICY_VERSIONS
        ):
            raise ValueError(
                "selection_policy_version must be a supported Memory selection policy"
            )

        if isinstance(self.eligible_records, (str, bytes)) or not isinstance(
            self.eligible_records, (list, tuple)
        ):
            raise ValueError("eligible_records must be a list or tuple")
        if len(self.eligible_records) > MAX_SNAPSHOT_RECORDS:
            raise ValueError("eligible_records exceeds the maximum item count")
        records: Dict[str, DurableMemoryRecord] = {}
        for record in self.eligible_records:
            if not isinstance(record, DurableMemoryRecord):
                raise ValueError("eligible_records items must be DurableMemoryRecord values")
            if record.repository_key != self.repository_key:
                raise ValueError("eligible record repository_key must match snapshot")
            if record.status is not RecordStatus.ACTIVE:
                raise ValueError("eligible_records must contain only active records")
            if record.memory_id in records:
                raise ValueError("eligible_records must not repeat a memory_id")
            records[record.memory_id] = record
        object.__setattr__(
            self,
            "eligible_records",
            tuple(records[key] for key in sorted(records)),
        )

        if isinstance(self.applicability_decisions, (str, bytes)) or not isinstance(
            self.applicability_decisions, (list, tuple)
        ):
            raise ValueError("applicability_decisions must be a list or tuple")
        if len(self.applicability_decisions) > MAX_SNAPSHOT_DECISIONS:
            raise ValueError("applicability_decisions exceeds the maximum item count")
        decisions: Dict[str, MemorySelectionDecision] = {}
        for decision in self.applicability_decisions:
            if not isinstance(decision, MemorySelectionDecision):
                raise ValueError(
                    "applicability_decisions items must be MemorySelectionDecision values"
                )
            if decision.memory_id in decisions:
                raise ValueError("applicability_decisions must not repeat a memory_id")
            decisions[decision.memory_id] = decision
        object.__setattr__(
            self,
            "applicability_decisions",
            tuple(
                sorted(
                    decisions.values(),
                    key=lambda item: (item.rank, item.memory_id),
                )
            ),
        )
        for memory_id in records:
            decision = decisions.get(memory_id)
            if decision is None or decision.applicability is not Applicability.SELECTED:
                raise ValueError(
                    "every eligible record requires a selected applicability decision"
                )
        selected_ids = {
            decision.memory_id
            for decision in decisions.values()
            if decision.applicability is Applicability.SELECTED
        }
        if selected_ids != set(records):
            raise ValueError(
                "selected applicability decisions must exactly match eligible records"
            )

        if self.feedback_calibration_summary is not None:
            if not isinstance(
                self.feedback_calibration_summary,
                FeedbackCalibrationSummary,
            ):
                raise ValueError(
                    "feedback_calibration_summary must be FeedbackCalibrationSummary or None"
                )
            if self.feedback_calibration_summary.repository_key != self.repository_key:
                raise ValueError("feedback calibration repository_key must match snapshot")
            if (
                self.feedback_calibration_summary.feedback_generation
                != self.generations.feedback_generation
            ):
                raise ValueError("feedback calibration generation must match snapshot")

        object.__setattr__(
            self,
            "repository_knowledge_refs",
            _canonical_stable_ids(
                self.repository_knowledge_refs,
                "RKE",
                "repository_knowledge_refs",
                max_items=MAX_KNOWLEDGE_REFS,
            ),
        )
        object.__setattr__(self, "created_at", _utc_timestamp(self.created_at, "created_at"))
        snapshot_hash = canonical_sha256(self._identity_dict())
        object.__setattr__(self, "snapshot_hash", snapshot_hash)
        object.__setattr__(self, "snapshot_id", "MSNAP-" + snapshot_hash)

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository_key": self.repository_key,
            "base_sha": self.base_sha,
            "head_sha": self.head_sha,
            "store_schema_version": self.generations.store_schema_version,
            "memory_generation": self.generations.memory_generation,
            "feedback_generation": self.generations.feedback_generation,
            "knowledge_generation": self.generations.knowledge_generation,
            "selection_policy_version": self.selection_policy_version,
            "eligible_records": [record.to_dict() for record in self.eligible_records],
            "applicability_decisions": [
                decision.to_dict() for decision in self.applicability_decisions
            ],
            "feedback_calibration_summary": (
                None
                if self.feedback_calibration_summary is None
                else self.feedback_calibration_summary.to_dict()
            ),
            "repository_knowledge_refs": list(self.repository_knowledge_refs),
            "created_at": self.created_at,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "snapshot_hash": self.snapshot_hash,
            **self._identity_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "MemorySnapshot":
        root = _object(payload, "memory_snapshot")
        _exact_fields(
            root,
            {
                "schema_version",
                "snapshot_id",
                "snapshot_hash",
                "repository_key",
                "base_sha",
                "head_sha",
                "store_schema_version",
                "memory_generation",
                "feedback_generation",
                "knowledge_generation",
                "selection_policy_version",
                "eligible_records",
                "applicability_decisions",
                "feedback_calibration_summary",
                "repository_knowledge_refs",
                "created_at",
            },
            "memory_snapshot",
        )
        for name in (
            "eligible_records",
            "applicability_decisions",
            "repository_knowledge_refs",
        ):
            if not isinstance(root[name], list):
                raise ValueError("memory_snapshot.%s must be a list" % name)
        snapshot = cls(
            repository_key=root["repository_key"],
            base_sha=root["base_sha"],
            head_sha=root["head_sha"],
            generations=GenerationMetadata(
                store_schema_version=root["store_schema_version"],
                memory_generation=root["memory_generation"],
                feedback_generation=root["feedback_generation"],
                knowledge_generation=root["knowledge_generation"],
            ),
            selection_policy_version=root["selection_policy_version"],
            eligible_records=tuple(
                DurableMemoryRecord.from_dict(item) for item in root["eligible_records"]
            ),
            applicability_decisions=tuple(
                MemorySelectionDecision.from_dict(item)
                for item in root["applicability_decisions"]
            ),
            feedback_calibration_summary=(
                None
                if root["feedback_calibration_summary"] is None
                else FeedbackCalibrationSummary.from_dict(
                    root["feedback_calibration_summary"]
                )
            ),
            repository_knowledge_refs=tuple(root["repository_knowledge_refs"]),
            created_at=root["created_at"],
            schema_version=root["schema_version"],
        )
        if root["snapshot_hash"] != snapshot.snapshot_hash:
            raise ValueError("memory_snapshot.snapshot_hash does not match canonical content")
        if root["snapshot_id"] != snapshot.snapshot_id:
            raise ValueError("memory_snapshot.snapshot_id does not match snapshot_hash")
        return snapshot


@dataclass(frozen=True)
class MemoryExecutionConfig(_JsonModel):
    mode: MemoryMode
    root_path: str
    required: bool = False
    selection_policy_version: str = MEMORY_SELECTION_POLICY_VERSION
    feedback_policy_version: str = FEEDBACK_AGGREGATION_POLICY_VERSION
    max_snapshot_records: int = MAX_SNAPSHOT_RECORDS
    max_snapshot_bytes: int = 8_388_608
    max_context_records: int = 12
    max_query_results: int = 8
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "memory_execution_config")
        if not isinstance(self.mode, MemoryMode):
            raise ValueError("mode must be a MemoryMode")
        raw_path = _normalize_text(
            self.root_path,
            "root_path",
            max_length=MAX_PATH_LENGTH,
            collapse_whitespace=False,
        )
        normalized_path = raw_path.replace("\\", "/")
        windows_path = PureWindowsPath(raw_path)
        posix_path = PurePosixPath(normalized_path)
        if not (windows_path.is_absolute() or posix_path.is_absolute()):
            raise ValueError("root_path must be an absolute path")
        if ".." in posix_path.parts:
            raise ValueError("root_path must be a canonical absolute path")
        if windows_path.is_absolute():
            canonical_root = windows_path.as_posix()
        else:
            canonical_root = posix_path.as_posix()
        if not re.fullmatch(r"[A-Za-z]:/", canonical_root):
            canonical_root = canonical_root.rstrip("/") or "/"
        object.__setattr__(self, "root_path", canonical_root)
        if type(self.required) is not bool:
            raise ValueError("required must be a boolean")
        if self.mode is MemoryMode.OFF and self.required:
            raise ValueError("required=true cannot be combined with mode=off")
        if (
            self.selection_policy_version
            not in SUPPORTED_MEMORY_SELECTION_POLICY_VERSIONS
        ):
            raise ValueError(
                "selection_policy_version must be a supported Memory selection policy"
            )
        if self.feedback_policy_version != FEEDBACK_AGGREGATION_POLICY_VERSION:
            raise ValueError(
                "feedback_policy_version must be %s"
                % FEEDBACK_AGGREGATION_POLICY_VERSION
            )
        bounds = {
            "max_snapshot_records": MAX_SNAPSHOT_RECORDS,
            "max_snapshot_bytes": 8_388_608,
            "max_context_records": 12,
            "max_query_results": 8,
        }
        for name, maximum in bounds.items():
            value = _positive_int(getattr(self, name), name)
            if value > maximum:
                raise ValueError("%s must not exceed %d" % (name, maximum))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "root_path": self.root_path,
            "required": self.required,
            "selection_policy_version": self.selection_policy_version,
            "feedback_policy_version": self.feedback_policy_version,
            "max_snapshot_records": self.max_snapshot_records,
            "max_snapshot_bytes": self.max_snapshot_bytes,
            "max_context_records": self.max_context_records,
            "max_query_results": self.max_query_results,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "MemoryExecutionConfig":
        root = _object(payload, "memory_execution_config")
        _exact_fields(
            root,
            {
                "schema_version",
                "mode",
                "root_path",
                "required",
                "selection_policy_version",
                "feedback_policy_version",
                "max_snapshot_records",
                "max_snapshot_bytes",
                "max_context_records",
                "max_query_results",
            },
            "memory_execution_config",
        )
        config = cls(
            mode=_enum_value(MemoryMode, root["mode"], "memory_execution_config.mode"),
            root_path=root["root_path"],
            required=root["required"],
            selection_policy_version=root["selection_policy_version"],
            feedback_policy_version=root["feedback_policy_version"],
            max_snapshot_records=root["max_snapshot_records"],
            max_snapshot_bytes=root["max_snapshot_bytes"],
            max_context_records=root["max_context_records"],
            max_query_results=root["max_query_results"],
            schema_version=root["schema_version"],
        )
        if root["root_path"] != config.root_path:
            raise ValueError(
                "memory_execution_config.root_path must be a canonical absolute path"
            )
        return config

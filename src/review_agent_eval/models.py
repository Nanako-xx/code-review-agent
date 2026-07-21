"""Canonical, immutable v2 protocol models for code-review evaluation.

This module deliberately depends only on Python's standard library.  The eval
wire protocols are an authority boundary: product Runtime, Session, Memory,
Provider, subprocess, ``Path`` and datetime objects never enter these models.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from urllib.parse import urlsplit


EVAL_INPUT_SCHEMA_VERSION = "eval_input_v2"
EVAL_SUBMISSION_SCHEMA_VERSION = "eval_submission_v2"
EVAL_CASE_SCHEMA_VERSION = "eval_case_v2"

MAX_EVAL_INPUT_BYTES = 2 * 1024 * 1024
MAX_EVAL_SUBMISSION_BYTES = 16 * 1024 * 1024
MAX_EVAL_CASE_BYTES = 16 * 1024 * 1024

MAX_IDENTIFIER_CHARS = 512
MAX_REPOSITORY_PATH_CHARS = 1024
MAX_URL_CHARS = 4096
MAX_TITLE_CHARS = 4096
MAX_DESCRIPTION_CHARS = 32768
MAX_CLAIM_CHARS = 8192
MAX_RATIONALE_CHARS = 8192
MAX_QUESTION_CHARS = 8192
MAX_ANSWER_CHARS = 8192
MAX_UNCERTAINTY_CHARS = 8192
MAX_EVIDENCE_EXCERPT_BYTES = 512 * 1024
MAX_EVALUATOR_CONTEXT_CONTENT_BYTES = 512 * 1024

MAX_REQUIREMENTS = 256
MAX_PROJECT_RULES = 256
MAX_EXISTING_CI_EVIDENCE = 256
MAX_TEXT_LIST_ITEMS = 256
MAX_CLARIFICATION_ANSWERS = 64
MAX_CLARIFICATION_QUESTIONS = 64
MAX_INTENT_CLAIMS = 1024
MAX_FINDINGS = 2048
MAX_EVIDENCE_ITEMS = 4096
MAX_EVIDENCE_REFS = 256
MAX_TRUTH_FINDINGS = 2048
MAX_TRUTH_LOCATIONS = 64
MAX_EVIDENCE_ANCHORS = 64
MAX_COMMAND_ARGUMENTS = 256

MAX_LINE_NUMBER = 2_147_483_647
MAX_COUNTER = 9_223_372_036_854_775_807
MAX_JSON_DEPTH = 128


class SchemaError(ValueError):
    """The value cannot be represented by a canonical eval v2 schema."""


class UnsupportedProtocolVersionError(SchemaError):
    """A root artifact declares a protocol version this implementation rejects."""

    code = "unsupported_protocol_version"

    def __init__(self, *, expected: str, actual: Any) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            "unsupported protocol version: expected %r, got %r"
            % (expected, actual)
        )


class RepositorySource(str, Enum):
    FIXTURE = "fixture"
    GIT = "git"


class TrialStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    INCOMPLETE = "incomplete"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    INVALID_OUTPUT = "invalid_output"


class SubmissionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    INVALID_OUTPUT = "invalid_output"


class JudgeStatus(str, Enum):
    GRADED = "graded"
    JUDGE_FAILED = "judge_failed"
    UNGRADED = "ungraded"


class FailureCode(str, Enum):
    TIMEOUT = "timeout"
    NON_ZERO_EXIT = "non_zero_exit"
    PROCESS_KILLED = "process_killed"
    OUTPUT_OVERFLOW = "output_overflow"
    INVALID_JSON = "invalid_json"
    SCHEMA_MISMATCH = "schema_mismatch"
    CLARIFICATION_REQUIRED = "clarification_required"
    AGENT_BLOCKED = "agent_blocked"
    ADAPTER_ERROR = "adapter_error"
    HARNESS_MATERIALIZATION_ERROR = "harness_materialization_error"
    UNKNOWN = "unknown"


class ClarificationAction(str, Enum):
    CONFIRM = "confirm"
    CORRECT = "correct"
    REJECT = "reject"
    SKIP = "skip"
    DEFER = "defer"


class IntentDimension(str, Enum):
    GOAL = "goal"
    ACCEPTANCE_CRITERION = "acceptance_criterion"
    SCOPE = "scope"
    CONSTRAINT = "constraint"


class IntentResult(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class IntentClaimSource(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class IntentClaimJudgement(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class FindingSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DiffSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class ReviewTargetKind(str, Enum):
    REPOSITORY = "repository"
    FROZEN_CONTEXT = "frozen_context"


class EvidenceKind(str, Enum):
    REPOSITORY_FILE = "repository_file"
    REPOSITORY_DIFF = "repository_diff"
    FROZEN_CONTEXT = "frozen_context"
    COMMAND_OUTPUT = "command_output"
    EXTERNAL_RECORD = "external_record"


class EvidenceStream(str, Enum):
    STDOUT = "stdout"
    STDERR = "stderr"
    COMBINED = "combined"


class TraceType(str, Enum):
    LOCAL_PATH = "local_path"
    URL = "url"
    OPAQUE_ID = "opaque_id"


class CaseOrigin(str, Enum):
    HAND_AUTHORED = "hand_authored"
    AACR_BENCH = "aacr_bench"
    SWE_PRBENCH = "swe_prbench"
    PRIVATE = "private"


class IntentAuthority(str, Enum):
    EXPLICIT_AUTHOR_METADATA = "explicit_author_metadata"
    LINKED_REQUIREMENT = "linked_requirement"
    EXPERT_RECONSTRUCTED = "expert_reconstructed"
    SYNTHETIC = "synthetic"


class ClarificationPolicy(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    NOT_REQUIRED = "not_required"


class TruthCompleteness(str, Enum):
    CLOSED_WORLD = "closed_world"
    EXPERT_AUGMENTED = "expert_augmented"
    HUMAN_OBSERVED = "human_observed"


class NovelFindingPolicy(str, Enum):
    VERIFY = "verify"
    FORBID = "forbid"


class RequiredContextLevel(str, Enum):
    DIFF = "diff"
    FILE = "file"
    REPO = "repo"


class MetricAuthoritySource(str, Enum):
    EXPERT_ANNOTATION = "expert_annotation"
    UPSTREAM_ANNOTATION = "upstream_annotation"


class EvaluatorContextTask(str, Enum):
    FINDING_EQUIVALENCE = "finding_equivalence"


class EvaluatorContextSourceKind(str, Enum):
    DIFF_HUNK = "diff_hunk"


class IssueJudgement(str, Enum):
    CONFIRMED = "confirmed"
    PLAUSIBLE = "plausible"
    FABRICATED = "fabricated"
    UNKNOWN = "unknown"


class EvidenceIntegrity(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    MISSING = "missing"


class EvidenceSupport(str, Enum):
    SUPPORTED = "supported"
    WEAK = "weak"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


EnumT = TypeVar("EnumT", bound=Enum)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_STABLE_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_VCS_METADATA = frozenset({".git", ".hg", ".svn"})


def _error(message: str) -> SchemaError:
    return SchemaError(message)


def _utf8_size(value: str, context: str) -> int:
    try:
        return len(value.encode("utf-8", "strict"))
    except UnicodeEncodeError as exc:
        raise _error("%s must contain valid Unicode scalar values" % context) from exc


def _string(
    value: Any,
    context: str,
    max_chars: int,
    *,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str:
        raise _error("%s must be a string" % context)
    _utf8_size(value, context)
    if not allow_empty and not value.strip():
        raise _error("%s must be a non-empty string" % context)
    if len(value) > max_chars:
        raise _error("%s exceeds the character limit of %d" % (context, max_chars))
    return value


def _optional_string(
    value: Any,
    context: str,
    max_chars: int,
    *,
    allow_empty: bool = False,
) -> Optional[str]:
    if value is None:
        return None
    return _string(value, context, max_chars, allow_empty=allow_empty)


def _identifier(value: Any, context: str) -> str:
    result = _string(value, context, MAX_IDENTIFIER_CHARS)
    if result != result.strip():
        raise _error("%s must not have leading or trailing whitespace" % context)
    for character in result:
        if character.isspace() or ord(character) < 32 or ord(character) == 127:
            raise _error("%s must be an opaque identifier without whitespace or controls" % context)
    return result


def _digest(value: Any, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _error("%s must be a full 64-character lowercase SHA-256 digest" % context)
    return value


def _git_object(value: Any, context: str) -> str:
    if type(value) is not str or _GIT_OBJECT_RE.fullmatch(value) is None:
        raise _error("%s must be a full 40- or 64-character lowercase Git object ID" % context)
    return value


def _integer(
    value: Any,
    context: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int:
        raise _error("%s must be an integer (bool is not accepted)" % context)
    if minimum is not None and value < minimum:
        raise _error("%s must be at least %d" % (context, minimum))
    if maximum is not None and value > maximum:
        raise _error("%s must be at most %d" % (context, maximum))
    return value


def _optional_integer(
    value: Any,
    context: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> Optional[int]:
    if value is None:
        return None
    return _integer(value, context, minimum=minimum, maximum=maximum)


def _number(value: Any, context: str) -> Any:
    if type(value) not in (int, float):
        raise _error("%s must be a JSON number (bool is not accepted)" % context)
    if isinstance(value, float) and not math.isfinite(value):
        raise _error("%s must be finite" % context)
    if value < 0:
        raise _error("%s must be non-negative" % context)
    return value


def _optional_number(value: Any, context: str) -> Any:
    if value is None:
        return None
    return _number(value, context)


def _boolean(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise _error("%s must be a boolean" % context)
    return value


def _enum_value(enum_type: Type[EnumT], value: Any, context: str) -> EnumT:
    if type(value) is not str:
        raise _error("%s must be a string enum value" % context)
    try:
        return enum_type(value)
    except ValueError as exc:
        raise _error("%s contains an unknown enum value: %r" % (context, value)) from exc


def _optional_enum(
    enum_type: Type[EnumT], value: Any, context: str
) -> Optional[EnumT]:
    if value is None:
        return None
    return _enum_value(enum_type, value, context)


def _require_enum(enum_type: Type[EnumT], value: Any, context: str) -> EnumT:
    if not isinstance(value, enum_type):
        raise _error("%s must be a %s" % (context, enum_type.__name__))
    return value


def _require_optional_enum(
    enum_type: Type[EnumT], value: Any, context: str
) -> Optional[EnumT]:
    if value is None:
        return None
    return _require_enum(enum_type, value, context)


def _object(value: Any, context: str) -> Dict[str, Any]:
    if type(value) is not dict:
        raise _error("%s must be a JSON object" % context)
    for key in value:
        if type(key) is not str:
            raise _error("%s must contain only string keys" % context)
    return value


def _exact_fields(payload: Dict[str, Any], expected: Iterable[str], context: str) -> None:
    expected_set = set(expected)
    actual_set = set(payload)
    missing = sorted(expected_set - actual_set)
    unknown = sorted(actual_set - expected_set)
    if missing or unknown:
        details: List[str] = []
        if missing:
            details.append("missing field(s): %s" % ", ".join(missing))
        if unknown:
            details.append("unknown field(s): %s" % ", ".join(unknown))
        raise _error("%s has %s" % (context, "; ".join(details)))


def _require_root_schema_version(
    payload: Dict[str, Any], expected: str, context: str
) -> str:
    if "schema_version" not in payload:
        raise _error("%s has missing field(s): schema_version" % context)
    actual = payload["schema_version"]
    if actual != expected:
        raise UnsupportedProtocolVersionError(expected=expected, actual=actual)
    return expected


def _array(value: Any, context: str, max_items: int) -> List[Any]:
    if type(value) is not list:
        raise _error("%s must be a JSON array" % context)
    if len(value) > max_items:
        raise _error("%s exceeds the item limit of %d" % (context, max_items))
    return value


def _sequence(value: Any, context: str, max_items: int) -> Tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise _error("%s must be a list or tuple" % context)
    if len(value) > max_items:
        raise _error("%s exceeds the item limit of %d" % (context, max_items))
    return tuple(value)


def _text_tuple(
    value: Any,
    context: str,
    max_items: int,
    max_chars: int,
) -> Tuple[str, ...]:
    raw = _sequence(value, context, max_items)
    return tuple(
        _string(item, "%s[%d]" % (context, index), max_chars)
        for index, item in enumerate(raw)
    )


def _model_tuple(
    value: Any,
    model_type: Type[Any],
    context: str,
    max_items: int,
) -> Tuple[Any, ...]:
    raw = _sequence(value, context, max_items)
    for index, item in enumerate(raw):
        if not isinstance(item, model_type):
            raise _error(
                "%s[%d] must be a %s" % (context, index, model_type.__name__)
            )
    return raw


def _unique_by(values: Iterable[Any], attribute: str, context: str) -> None:
    seen = set()
    for item in values:
        identity = getattr(item, attribute)
        if identity in seen:
            raise _error("%s contains duplicate %s %r" % (context, attribute, identity))
        seen.add(identity)


def _sorted_by(values: Iterable[Any], attribute: str) -> Tuple[Any, ...]:
    return tuple(sorted(values, key=lambda item: getattr(item, attribute)))


def _safe_repo_path(value: Any, context: str) -> str:
    path = _string(value, context, MAX_REPOSITORY_PATH_CHARS)
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        raise _error("%s may not contain control characters" % context)
    if path.startswith(("/", "\\")) or _WINDOWS_DRIVE_RE.match(path):
        raise _error("%s must be a relative POSIX path" % context)
    if "\\" in path:
        raise _error("%s must use POSIX separators" % context)
    components = path.split("/")
    if any(component in ("", ".", "..") for component in components):
        raise _error("%s contains an unsafe path component" % context)
    if any(component.casefold() in _VCS_METADATA for component in components):
        raise _error("%s may not traverse VCS metadata" % context)
    if any("\x00" in component for component in components):
        raise _error("%s may not contain NUL" % context)
    return path


def _loose_submission_path(value: Any, context: str) -> Optional[str]:
    if value is None:
        return None
    path = _string(value, context, MAX_REPOSITORY_PATH_CHARS)
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        raise _error("%s may not contain control characters" % context)
    return path


def _required_loose_submission_path(value: Any, context: str) -> str:
    path = _loose_submission_path(value, context)
    if path is None:
        raise _error("%s must be a non-null path string" % context)
    return path


def _repository_url(value: Any, context: str) -> str:
    url = _string(value, context, MAX_URL_CHARS)
    if any(ord(character) < 32 or character.isspace() for character in url):
        raise _error("%s may not contain whitespace or controls" % context)
    try:
        parsed = urlsplit(url)
        username = parsed.username
        password = parsed.password
        hostname = parsed.hostname
    except ValueError as exc:
        raise _error("%s is not a valid repository URL" % context) from exc
    if parsed.scheme.casefold() not in {"http", "https", "ssh", "git"} or not hostname:
        raise _error("%s must be an absolute repository URL" % context)
    if username is not None or password is not None or "@" in parsed.netloc:
        raise _error("%s may not contain userinfo or credentials" % context)
    return url


def _json_tree(value: Any, context: str = "JSON", depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise _error("%s exceeds the maximum nesting depth" % context)
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise _error("%s contains a non-finite number" % context)
        return
    if type(value) is str:
        _utf8_size(value, context)
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _json_tree(item, "%s[%d]" % (context, index), depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise _error("%s contains a non-string object key" % context)
            _utf8_size(key, "%s key" % context)
            _json_tree(item, "%s.%s" % (context, key), depth + 1)
        return
    raise _error("%s contains a non-JSON value" % context)


def _reject_duplicate_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _error("JSON contains duplicate object key %r" % key)
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise _error("JSON contains non-standard numeric constant %s" % value)


def _strict_json_loads(data: Any, max_bytes: int, context: str) -> Any:
    if type(data) is bytes:
        if len(data) > max_bytes:
            raise _error("%s exceeds the raw byte limit of %d" % (context, max_bytes))
        try:
            text = data.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise _error("%s must be strict UTF-8" % context) from exc
    elif type(data) is str:
        encoded_size = _utf8_size(data, context)
        if encoded_size > max_bytes:
            raise _error("%s exceeds the raw byte limit of %d" % (context, max_bytes))
        text = data
    else:
        raise _error("%s loader accepts only bytes or text" % context)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except SchemaError:
        raise
    except (
        json.JSONDecodeError,
        RecursionError,
        UnicodeError,
        OverflowError,
        ValueError,
    ) as exc:
        raise _error("%s is not valid strict JSON: %s" % (context, exc)) from exc
    _json_tree(value, context)
    return value


class _JsonModel:
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def to_json(self) -> str:
        return canonical_json(self)

    def digest(self) -> str:
        return canonical_sha256(self)


def _json_ready(value: Any, context: str = "value", depth: int = 0) -> Any:
    if depth > MAX_JSON_DEPTH:
        raise _error("%s exceeds the maximum nesting depth" % context)
    if isinstance(value, _JsonModel):
        return _json_ready(value.to_dict(), context, depth + 1)
    if isinstance(value, Enum):
        if type(value.value) is not str:
            raise _error("%s contains an enum with a non-string value" % context)
        return value.value
    if value is None or type(value) in (str, int, bool):
        if type(value) is str:
            _utf8_size(value, context)
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _error("%s contains a non-finite number" % context)
        return value
    if type(value) is dict:
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise _error("%s contains a non-string object key" % context)
            _utf8_size(key, "%s key" % context)
            result[key] = _json_ready(item, "%s.%s" % (context, key), depth + 1)
        return result
    if type(value) in (list, tuple):
        return [
            _json_ready(item, "%s[%d]" % (context, index), depth + 1)
            for index, item in enumerate(value)
        ]
    raise _error("%s contains a non-JSON-ready value" % context)


def canonical_json(value: Any) -> str:
    """Return the sole canonical v2 JSON text representation."""

    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def stable_id(prefix: str, *identity: Any) -> str:
    """Derive a namespaced ID using a complete, never-truncated SHA-256."""

    if type(prefix) is not str or _STABLE_PREFIX_RE.fullmatch(prefix) is None:
        raise _error("stable ID prefix is invalid")
    identity_payload = {
        "namespace": "review_agent_eval.identity_v2",
        "kind": prefix,
        "identity": list(identity),
    }
    return "%s-%s" % (prefix, canonical_sha256(identity_payload))


def validate_stable_id(value: Any, prefix: str, *identity: Any) -> str:
    expected = stable_id(prefix, *identity)
    if value != expected:
        raise _error("derived ID does not match its canonical identity payload")
    return expected


def _check_model_size(model: _JsonModel, maximum: int, context: str) -> None:
    size = len(canonical_json_bytes(model))
    if size > maximum:
        raise _error("%s exceeds the canonical byte limit of %d" % (context, maximum))


@dataclass(frozen=True)
class Repository(_JsonModel):
    source: RepositorySource
    path: Optional[str]
    url: Optional[str]
    base_revision: str
    head_revision: str

    def __post_init__(self) -> None:
        _require_enum(RepositorySource, self.source, "repository.source")
        base = _git_object(self.base_revision, "repository.base_revision")
        head = _git_object(self.head_revision, "repository.head_revision")
        if len(base) != len(head):
            raise _error("repository revisions must use the same object ID length")
        if base == head:
            raise _error("repository base_revision and head_revision must differ")
        path = None if self.path is None else _safe_repo_path(self.path, "repository.path")
        url = None if self.url is None else _repository_url(self.url, "repository.url")
        if self.source is RepositorySource.FIXTURE:
            if path is None or url is not None:
                raise _error("fixture repository requires path and requires url=null")
        elif (path is None) == (url is None):
            raise _error("git repository must provide exactly one of path or url")

    @classmethod
    def from_dict(cls, value: Any) -> "Repository":
        payload = _object(value, "repository")
        _exact_fields(
            payload,
            ("source", "path", "url", "base_revision", "head_revision"),
            "repository",
        )
        return cls(
            source=_enum_value(RepositorySource, payload["source"], "repository.source"),
            path=_optional_string(
                payload["path"], "repository.path", MAX_REPOSITORY_PATH_CHARS
            ),
            url=_optional_string(payload["url"], "repository.url", MAX_URL_CHARS),
            base_revision=_git_object(
                payload["base_revision"], "repository.base_revision"
            ),
            head_revision=_git_object(
                payload["head_revision"], "repository.head_revision"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "path": self.path,
            "url": self.url,
            "base_revision": self.base_revision,
            "head_revision": self.head_revision,
        }


@dataclass(frozen=True)
class ExistingCIEvidence(_JsonModel):
    source_id: str
    text: str
    content_hash: str

    def __post_init__(self) -> None:
        _identifier(self.source_id, "existing_ci_evidence.source_id")
        text = _string(
            self.text, "existing_ci_evidence.text", MAX_DESCRIPTION_CHARS, allow_empty=True
        )
        digest = _digest(self.content_hash, "existing_ci_evidence.content_hash")
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != expected:
            raise _error(
                "existing_ci_evidence.content_hash must hash the exact UTF-8 text"
            )

    @classmethod
    def from_dict(cls, value: Any) -> "ExistingCIEvidence":
        payload = _object(value, "existing_ci_evidence entry")
        _exact_fields(
            payload,
            ("source_id", "text", "content_hash"),
            "existing_ci_evidence entry",
        )
        return cls(
            source_id=_identifier(
                payload["source_id"], "existing_ci_evidence.source_id"
            ),
            text=_string(
                payload["text"],
                "existing_ci_evidence.text",
                MAX_DESCRIPTION_CHARS,
                allow_empty=True,
            ),
            content_hash=_digest(
                payload["content_hash"], "existing_ci_evidence.content_hash"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "text": self.text,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class ReviewRequest(_JsonModel):
    title: Optional[str]
    description: Optional[str]
    user_intent: Optional[str]
    review_focus: Optional[str]
    linked_requirements: Tuple[str, ...]
    project_rules: Tuple[str, ...]
    existing_ci_evidence: Tuple[ExistingCIEvidence, ...]

    def __post_init__(self) -> None:
        _optional_string(self.title, "review_request.title", MAX_TITLE_CHARS)
        _optional_string(
            self.description, "review_request.description", MAX_DESCRIPTION_CHARS
        )
        _optional_string(
            self.user_intent, "review_request.user_intent", MAX_DESCRIPTION_CHARS
        )
        _optional_string(
            self.review_focus, "review_request.review_focus", MAX_DESCRIPTION_CHARS
        )
        requirements = _text_tuple(
            self.linked_requirements,
            "review_request.linked_requirements",
            MAX_REQUIREMENTS,
            MAX_CLAIM_CHARS,
        )
        rules = _text_tuple(
            self.project_rules,
            "review_request.project_rules",
            MAX_PROJECT_RULES,
            MAX_CLAIM_CHARS,
        )
        ci = _model_tuple(
            self.existing_ci_evidence,
            ExistingCIEvidence,
            "review_request.existing_ci_evidence",
            MAX_EXISTING_CI_EVIDENCE,
        )
        _unique_by(ci, "source_id", "review_request.existing_ci_evidence")
        object.__setattr__(self, "linked_requirements", requirements)
        object.__setattr__(self, "project_rules", rules)
        object.__setattr__(self, "existing_ci_evidence", _sorted_by(ci, "source_id"))

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewRequest":
        payload = _object(value, "review_request")
        _exact_fields(
            payload,
            (
                "title",
                "description",
                "user_intent",
                "review_focus",
                "linked_requirements",
                "project_rules",
                "existing_ci_evidence",
            ),
            "review_request",
        )
        requirements = _array(
            payload["linked_requirements"],
            "review_request.linked_requirements",
            MAX_REQUIREMENTS,
        )
        rules = _array(
            payload["project_rules"], "review_request.project_rules", MAX_PROJECT_RULES
        )
        ci_payload = _array(
            payload["existing_ci_evidence"],
            "review_request.existing_ci_evidence",
            MAX_EXISTING_CI_EVIDENCE,
        )
        return cls(
            title=_optional_string(
                payload["title"], "review_request.title", MAX_TITLE_CHARS
            ),
            description=_optional_string(
                payload["description"],
                "review_request.description",
                MAX_DESCRIPTION_CHARS,
            ),
            user_intent=_optional_string(
                payload["user_intent"],
                "review_request.user_intent",
                MAX_DESCRIPTION_CHARS,
            ),
            review_focus=_optional_string(
                payload["review_focus"],
                "review_request.review_focus",
                MAX_DESCRIPTION_CHARS,
            ),
            linked_requirements=tuple(requirements),
            project_rules=tuple(rules),
            existing_ci_evidence=tuple(
                ExistingCIEvidence.from_dict(item) for item in ci_payload
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "user_intent": self.user_intent,
            "review_focus": self.review_focus,
            "linked_requirements": list(self.linked_requirements),
            "project_rules": list(self.project_rules),
            "existing_ci_evidence": [item.to_dict() for item in self.existing_ci_evidence],
        }


@dataclass(frozen=True)
class RepositoryReviewTarget(_JsonModel):
    kind: ReviewTargetKind
    repository: Repository
    review_request: ReviewRequest

    def __post_init__(self) -> None:
        if self.kind is not ReviewTargetKind.REPOSITORY:
            raise _error("repository review_target.kind must be repository")
        if not isinstance(self.repository, Repository):
            raise _error("review_target.repository must be a Repository")
        if not isinstance(self.review_request, ReviewRequest):
            raise _error("review_target.review_request must be a ReviewRequest")

    @classmethod
    def from_dict(cls, value: Any) -> "RepositoryReviewTarget":
        payload = _object(value, "review_target")
        _exact_fields(
            payload,
            ("kind", "repository", "review_request"),
            "review_target",
        )
        return cls(
            kind=_enum_value(
                ReviewTargetKind, payload["kind"], "review_target.kind"
            ),
            repository=Repository.from_dict(payload["repository"]),
            review_request=ReviewRequest.from_dict(payload["review_request"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "repository": self.repository.to_dict(),
            "review_request": self.review_request.to_dict(),
        }


@dataclass(frozen=True)
class FrozenContextReviewTarget(_JsonModel):
    kind: ReviewTargetKind
    bundle_id: str
    record_id: str
    context_format: str
    rendered_sha256: str
    rendered_utf8_bytes: int
    source_binding_digest: str

    def __post_init__(self) -> None:
        if self.kind is not ReviewTargetKind.FROZEN_CONTEXT:
            raise _error("frozen review_target.kind must be frozen_context")
        _identifier(self.bundle_id, "review_target.bundle_id")
        _identifier(self.record_id, "review_target.record_id")
        _string(
            self.context_format,
            "review_target.context_format",
            MAX_IDENTIFIER_CHARS,
        )
        if self.context_format != "rendered_text":
            raise _error("review_target.context_format must be rendered_text")
        _digest(self.rendered_sha256, "review_target.rendered_sha256")
        _integer(
            self.rendered_utf8_bytes,
            "review_target.rendered_utf8_bytes",
            minimum=0,
            maximum=MAX_COUNTER,
        )
        _digest(
            self.source_binding_digest,
            "review_target.source_binding_digest",
        )

    @classmethod
    def from_dict(cls, value: Any) -> "FrozenContextReviewTarget":
        payload = _object(value, "review_target")
        _exact_fields(
            payload,
            (
                "kind",
                "bundle_id",
                "record_id",
                "context_format",
                "rendered_sha256",
                "rendered_utf8_bytes",
                "source_binding_digest",
            ),
            "review_target",
        )
        context_format = _string(
            payload["context_format"],
            "review_target.context_format",
            MAX_IDENTIFIER_CHARS,
        )
        return cls(
            kind=_enum_value(
                ReviewTargetKind, payload["kind"], "review_target.kind"
            ),
            bundle_id=_identifier(payload["bundle_id"], "review_target.bundle_id"),
            record_id=_identifier(payload["record_id"], "review_target.record_id"),
            context_format=context_format,
            rendered_sha256=_digest(
                payload["rendered_sha256"], "review_target.rendered_sha256"
            ),
            rendered_utf8_bytes=_integer(
                payload["rendered_utf8_bytes"],
                "review_target.rendered_utf8_bytes",
                minimum=0,
                maximum=MAX_COUNTER,
            ),
            source_binding_digest=_digest(
                payload["source_binding_digest"],
                "review_target.source_binding_digest",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "bundle_id": self.bundle_id,
            "record_id": self.record_id,
            "context_format": self.context_format,
            "rendered_sha256": self.rendered_sha256,
            "rendered_utf8_bytes": self.rendered_utf8_bytes,
            "source_binding_digest": self.source_binding_digest,
        }


ReviewTargetV2 = Union[RepositoryReviewTarget, FrozenContextReviewTarget]


def _review_target_from_dict(value: Any) -> ReviewTargetV2:
    payload = _object(value, "review_target")
    if "kind" not in payload:
        raise _error("review_target has missing field(s): kind")
    kind = _enum_value(ReviewTargetKind, payload["kind"], "review_target.kind")
    if kind is ReviewTargetKind.REPOSITORY:
        return RepositoryReviewTarget.from_dict(payload)
    return FrozenContextReviewTarget.from_dict(payload)


@dataclass(frozen=True)
class EvalInput(_JsonModel):
    SCHEMA_VERSION: ClassVar[str] = EVAL_INPUT_SCHEMA_VERSION

    schema_version: str
    task_id: str
    review_target: ReviewTargetV2

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise UnsupportedProtocolVersionError(
                expected=self.SCHEMA_VERSION, actual=self.schema_version
            )
        _identifier(self.task_id, "eval_input.task_id")
        if not isinstance(
            self.review_target,
            (RepositoryReviewTarget, FrozenContextReviewTarget),
        ):
            raise _error("eval_input.review_target must be a ReviewTargetV2")
        _check_model_size(self, MAX_EVAL_INPUT_BYTES, "EvalInput")

    @classmethod
    def from_dict(cls, value: Any) -> "EvalInput":
        payload = _object(value, "EvalInput")
        _require_root_schema_version(payload, cls.SCHEMA_VERSION, "EvalInput")
        _exact_fields(
            payload,
            ("schema_version", "task_id", "review_target"),
            "EvalInput",
        )
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            task_id=_identifier(payload["task_id"], "eval_input.task_id"),
            review_target=_review_target_from_dict(payload["review_target"]),
        )

    @classmethod
    def from_json(cls, data: Any) -> "EvalInput":
        return cls.from_dict(
            _strict_json_loads(data, MAX_EVAL_INPUT_BYTES, "EvalInput JSON")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "review_target": self.review_target.to_dict(),
        }


@dataclass(frozen=True)
class EvalCaseInput(_JsonModel):
    review_target: ReviewTargetV2

    def __post_init__(self) -> None:
        if not isinstance(
            self.review_target,
            (RepositoryReviewTarget, FrozenContextReviewTarget),
        ):
            raise _error("case.input.review_target must be a ReviewTargetV2")

    @classmethod
    def from_dict(cls, value: Any) -> "EvalCaseInput":
        payload = _object(value, "case.input")
        _exact_fields(payload, ("review_target",), "case.input")
        return cls(review_target=_review_target_from_dict(payload["review_target"]))

    def to_eval_input(self, task_id: str) -> EvalInput:
        return EvalInput(
            schema_version=EVAL_INPUT_SCHEMA_VERSION,
            task_id=_identifier(task_id, "eval_case.task_id"),
            review_target=self.review_target,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"review_target": self.review_target.to_dict()}


@dataclass(frozen=True)
class SubmissionIntentClaim(_JsonModel):
    claim_id: str
    dimension: IntentDimension
    text: str
    source: IntentClaimSource

    def __post_init__(self) -> None:
        _identifier(self.claim_id, "intent claim.claim_id")
        _require_enum(IntentDimension, self.dimension, "intent claim.dimension")
        _string(self.text, "intent claim.text", MAX_CLAIM_CHARS)
        _require_enum(IntentClaimSource, self.source, "intent claim.source")

    @classmethod
    def from_dict(cls, value: Any) -> "SubmissionIntentClaim":
        payload = _object(value, "intent claim")
        _exact_fields(payload, ("claim_id", "dimension", "text", "source"), "intent claim")
        return cls(
            claim_id=_identifier(payload["claim_id"], "intent claim.claim_id"),
            dimension=_enum_value(
                IntentDimension, payload["dimension"], "intent claim.dimension"
            ),
            text=_string(payload["text"], "intent claim.text", MAX_CLAIM_CHARS),
            source=_enum_value(
                IntentClaimSource, payload["source"], "intent claim.source"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "dimension": self.dimension.value,
            "text": self.text,
            "source": self.source.value,
        }


@dataclass(frozen=True)
class SubmissionClarificationExchange(_JsonModel):
    turn_index: int
    question_id: str
    dimension: IntentDimension
    question: str
    material_claim: str
    matched_answer_id: Optional[str]
    action: Optional[ClarificationAction]
    response: Optional[str]
    resolved_values: Tuple[str, ...]

    def __post_init__(self) -> None:
        _integer(
            self.turn_index,
            "clarification exchange.turn_index",
            minimum=1,
            maximum=MAX_CLARIFICATION_QUESTIONS,
        )
        _identifier(self.question_id, "clarification exchange.question_id")
        _require_enum(
            IntentDimension, self.dimension, "clarification exchange.dimension"
        )
        _string(self.question, "clarification exchange.question", MAX_QUESTION_CHARS)
        _string(
            self.material_claim,
            "clarification exchange.material_claim",
            MAX_CLAIM_CHARS,
        )
        if self.matched_answer_id is not None:
            _identifier(
                self.matched_answer_id, "clarification exchange.matched_answer_id"
            )
        _require_optional_enum(
            ClarificationAction, self.action, "clarification exchange.action"
        )
        _optional_string(
            self.response, "clarification exchange.response", MAX_ANSWER_CHARS
        )
        resolved = _text_tuple(
            self.resolved_values,
            "clarification exchange.resolved_values",
            MAX_TEXT_LIST_ITEMS,
            MAX_ANSWER_CHARS,
        )
        if self.action is None:
            if self.matched_answer_id is not None or self.response is not None or resolved:
                raise _error(
                    "unanswered clarification must have null match/action/response and empty resolved_values"
                )
        else:
            if self.matched_answer_id is None:
                raise _error("answered clarification must have matched_answer_id")
            if self.action is ClarificationAction.CONFIRM:
                if not resolved:
                    raise _error("confirm clarification must have resolved_values")
            elif self.action is ClarificationAction.CORRECT:
                if self.response is None or not resolved:
                    raise _error(
                        "correct clarification must have response and resolved_values"
                    )
            elif resolved:
                raise _error(
                    "reject/skip/defer clarification must have empty resolved_values"
                )
        object.__setattr__(self, "resolved_values", resolved)

    @classmethod
    def from_dict(cls, value: Any) -> "SubmissionClarificationExchange":
        payload = _object(value, "clarification exchange")
        _exact_fields(
            payload,
            (
                "turn_index",
                "question_id",
                "dimension",
                "question",
                "material_claim",
                "matched_answer_id",
                "action",
                "response",
                "resolved_values",
            ),
            "clarification exchange",
        )
        resolved = _array(
            payload["resolved_values"],
            "clarification exchange.resolved_values",
            MAX_TEXT_LIST_ITEMS,
        )
        return cls(
            turn_index=_integer(
                payload["turn_index"],
                "clarification exchange.turn_index",
                minimum=1,
                maximum=MAX_CLARIFICATION_QUESTIONS,
            ),
            question_id=_identifier(
                payload["question_id"], "clarification exchange.question_id"
            ),
            dimension=_enum_value(
                IntentDimension,
                payload["dimension"],
                "clarification exchange.dimension",
            ),
            question=_string(
                payload["question"],
                "clarification exchange.question",
                MAX_QUESTION_CHARS,
            ),
            material_claim=_string(
                payload["material_claim"],
                "clarification exchange.material_claim",
                MAX_CLAIM_CHARS,
            ),
            matched_answer_id=(
                None
                if payload["matched_answer_id"] is None
                else _identifier(
                    payload["matched_answer_id"],
                    "clarification exchange.matched_answer_id",
                )
            ),
            action=_optional_enum(
                ClarificationAction,
                payload["action"],
                "clarification exchange.action",
            ),
            response=_optional_string(
                payload["response"],
                "clarification exchange.response",
                MAX_ANSWER_CHARS,
            ),
            resolved_values=tuple(resolved),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "question_id": self.question_id,
            "dimension": self.dimension.value,
            "question": self.question,
            "material_claim": self.material_claim,
            "matched_answer_id": self.matched_answer_id,
            "action": None if self.action is None else self.action.value,
            "response": self.response,
            "resolved_values": list(self.resolved_values),
        }


@dataclass(frozen=True)
class SubmissionIntent(_JsonModel):
    status: IntentResult
    goal: Optional[str]
    acceptance_criteria: Tuple[str, ...]
    scope: Tuple[str, ...]
    constraints: Tuple[str, ...]
    claims: Tuple[SubmissionIntentClaim, ...]
    clarification_questions: Tuple[SubmissionClarificationExchange, ...]
    uncertainties: Tuple[str, ...]

    def __post_init__(self) -> None:
        _require_enum(IntentResult, self.status, "submission intent.status")
        _optional_string(self.goal, "submission intent.goal", MAX_CLAIM_CHARS)
        acceptance = _text_tuple(
            self.acceptance_criteria,
            "submission intent.acceptance_criteria",
            MAX_TEXT_LIST_ITEMS,
            MAX_CLAIM_CHARS,
        )
        scope = _text_tuple(
            self.scope,
            "submission intent.scope",
            MAX_TEXT_LIST_ITEMS,
            MAX_CLAIM_CHARS,
        )
        constraints = _text_tuple(
            self.constraints,
            "submission intent.constraints",
            MAX_TEXT_LIST_ITEMS,
            MAX_CLAIM_CHARS,
        )
        claims = _model_tuple(
            self.claims,
            SubmissionIntentClaim,
            "submission intent.claims",
            MAX_INTENT_CLAIMS,
        )
        _unique_by(claims, "claim_id", "submission intent.claims")
        questions = _model_tuple(
            self.clarification_questions,
            SubmissionClarificationExchange,
            "submission intent.clarification_questions",
            MAX_CLARIFICATION_QUESTIONS,
        )
        _unique_by(
            questions,
            "question_id",
            "submission intent.clarification_questions",
        )
        for expected_turn, question in enumerate(questions, start=1):
            if question.turn_index != expected_turn:
                raise _error(
                    "clarification turn_index must be contiguous from 1 in transcript order"
                )
        uncertainties = _text_tuple(
            self.uncertainties,
            "submission intent.uncertainties",
            MAX_TEXT_LIST_ITEMS,
            MAX_UNCERTAINTY_CHARS,
        )
        object.__setattr__(self, "acceptance_criteria", acceptance)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "claims", _sorted_by(claims, "claim_id"))
        object.__setattr__(self, "clarification_questions", questions)
        object.__setattr__(self, "uncertainties", uncertainties)

    @classmethod
    def from_dict(cls, value: Any) -> "SubmissionIntent":
        payload = _object(value, "submission intent")
        _exact_fields(
            payload,
            (
                "status",
                "goal",
                "acceptance_criteria",
                "scope",
                "constraints",
                "claims",
                "clarification_questions",
                "uncertainties",
            ),
            "submission intent",
        )
        acceptance = _array(
            payload["acceptance_criteria"],
            "submission intent.acceptance_criteria",
            MAX_TEXT_LIST_ITEMS,
        )
        scope = _array(payload["scope"], "submission intent.scope", MAX_TEXT_LIST_ITEMS)
        constraints = _array(
            payload["constraints"],
            "submission intent.constraints",
            MAX_TEXT_LIST_ITEMS,
        )
        claims = _array(
            payload["claims"], "submission intent.claims", MAX_INTENT_CLAIMS
        )
        questions = _array(
            payload["clarification_questions"],
            "submission intent.clarification_questions",
            MAX_CLARIFICATION_QUESTIONS,
        )
        uncertainties = _array(
            payload["uncertainties"],
            "submission intent.uncertainties",
            MAX_TEXT_LIST_ITEMS,
        )
        return cls(
            status=_enum_value(
                IntentResult, payload["status"], "submission intent.status"
            ),
            goal=_optional_string(
                payload["goal"], "submission intent.goal", MAX_CLAIM_CHARS
            ),
            acceptance_criteria=tuple(acceptance),
            scope=tuple(scope),
            constraints=tuple(constraints),
            claims=tuple(SubmissionIntentClaim.from_dict(item) for item in claims),
            clarification_questions=tuple(
                SubmissionClarificationExchange.from_dict(item) for item in questions
            ),
            uncertainties=tuple(uncertainties),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "goal": self.goal,
            "acceptance_criteria": list(self.acceptance_criteria),
            "scope": list(self.scope),
            "constraints": list(self.constraints),
            "claims": [item.to_dict() for item in self.claims],
            "clarification_questions": [
                item.to_dict() for item in self.clarification_questions
            ],
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True)
class SubmissionFinding(_JsonModel):
    finding_id: str
    claim: str
    severity: FindingSeverity
    path: Optional[str]
    side: Optional[DiffSide]
    from_line: Optional[int]
    to_line: Optional[int]
    evidence_refs: Tuple[str, ...]
    suggested_action: Optional[str]

    def __post_init__(self) -> None:
        _identifier(self.finding_id, "finding.finding_id")
        _string(self.claim, "finding.claim", MAX_CLAIM_CHARS)
        _require_enum(FindingSeverity, self.severity, "finding.severity")
        _loose_submission_path(self.path, "finding.path")
        _require_optional_enum(DiffSide, self.side, "finding.side")
        _optional_integer(
            self.from_line,
            "finding.from_line",
            minimum=1,
            maximum=MAX_LINE_NUMBER,
        )
        _optional_integer(
            self.to_line,
            "finding.to_line",
            minimum=1,
            maximum=MAX_LINE_NUMBER,
        )
        refs = _sequence(self.evidence_refs, "finding.evidence_refs", MAX_EVIDENCE_REFS)
        refs = tuple(
            _identifier(item, "finding.evidence_refs[%d]" % index)
            for index, item in enumerate(refs)
        )
        _optional_string(
            self.suggested_action, "finding.suggested_action", MAX_CLAIM_CHARS
        )
        object.__setattr__(self, "evidence_refs", refs)

    @classmethod
    def from_dict(cls, value: Any) -> "SubmissionFinding":
        payload = _object(value, "finding")
        _exact_fields(
            payload,
            (
                "finding_id",
                "claim",
                "severity",
                "path",
                "side",
                "from_line",
                "to_line",
                "evidence_refs",
                "suggested_action",
            ),
            "finding",
        )
        refs = _array(payload["evidence_refs"], "finding.evidence_refs", MAX_EVIDENCE_REFS)
        return cls(
            finding_id=_identifier(payload["finding_id"], "finding.finding_id"),
            claim=_string(payload["claim"], "finding.claim", MAX_CLAIM_CHARS),
            severity=_enum_value(
                FindingSeverity, payload["severity"], "finding.severity"
            ),
            path=_loose_submission_path(payload["path"], "finding.path"),
            side=_optional_enum(DiffSide, payload["side"], "finding.side"),
            from_line=_optional_integer(
                payload["from_line"],
                "finding.from_line",
                minimum=1,
                maximum=MAX_LINE_NUMBER,
            ),
            to_line=_optional_integer(
                payload["to_line"],
                "finding.to_line",
                minimum=1,
                maximum=MAX_LINE_NUMBER,
            ),
            evidence_refs=tuple(refs),
            suggested_action=_optional_string(
                payload["suggested_action"],
                "finding.suggested_action",
                MAX_CLAIM_CHARS,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "claim": self.claim,
            "severity": self.severity.value,
            "path": self.path,
            "side": None if self.side is None else self.side.value,
            "from_line": self.from_line,
            "to_line": self.to_line,
            "evidence_refs": list(self.evidence_refs),
            "suggested_action": self.suggested_action,
        }


@dataclass(frozen=True)
class SubmissionReview(_JsonModel):
    findings: Tuple[SubmissionFinding, ...]
    uncertainties: Tuple[str, ...]

    def __post_init__(self) -> None:
        findings = _model_tuple(
            self.findings, SubmissionFinding, "submission review.findings", MAX_FINDINGS
        )
        _unique_by(findings, "finding_id", "submission review.findings")
        uncertainties = _text_tuple(
            self.uncertainties,
            "submission review.uncertainties",
            MAX_TEXT_LIST_ITEMS,
            MAX_UNCERTAINTY_CHARS,
        )
        object.__setattr__(self, "findings", _sorted_by(findings, "finding_id"))
        object.__setattr__(self, "uncertainties", uncertainties)

    @classmethod
    def from_dict(cls, value: Any) -> "SubmissionReview":
        payload = _object(value, "submission review")
        _exact_fields(payload, ("findings", "uncertainties"), "submission review")
        findings = _array(payload["findings"], "submission review.findings", MAX_FINDINGS)
        uncertainties = _array(
            payload["uncertainties"],
            "submission review.uncertainties",
            MAX_TEXT_LIST_ITEMS,
        )
        return cls(
            findings=tuple(SubmissionFinding.from_dict(item) for item in findings),
            uncertainties=tuple(uncertainties),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "findings": [item.to_dict() for item in self.findings],
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True)
class RepositoryFileEvidenceSource(_JsonModel):
    kind: EvidenceKind
    target_materialization_id: str
    revision: str
    path: str
    from_line: int
    to_line: int

    def __post_init__(self) -> None:
        if self.kind is not EvidenceKind.REPOSITORY_FILE:
            raise _error("repository_file evidence.source.kind is invalid")
        _identifier(
            self.target_materialization_id,
            "evidence.source.target_materialization_id",
        )
        _identifier(self.revision, "evidence.source.revision")
        _required_loose_submission_path(self.path, "evidence.source.path")
        _integer(
            self.from_line,
            "evidence.source.from_line",
            minimum=1,
            maximum=MAX_LINE_NUMBER,
        )
        _integer(
            self.to_line,
            "evidence.source.to_line",
            minimum=1,
            maximum=MAX_LINE_NUMBER,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "RepositoryFileEvidenceSource":
        payload = _object(value, "evidence.source")
        _exact_fields(
            payload,
            (
                "kind",
                "target_materialization_id",
                "revision",
                "path",
                "from_line",
                "to_line",
            ),
            "evidence.source",
        )
        return cls(
            kind=_enum_value(EvidenceKind, payload["kind"], "evidence.source.kind"),
            target_materialization_id=_identifier(
                payload["target_materialization_id"],
                "evidence.source.target_materialization_id",
            ),
            revision=_identifier(payload["revision"], "evidence.source.revision"),
            path=_required_loose_submission_path(
                payload["path"], "evidence.source.path"
            ),
            from_line=_integer(
                payload["from_line"],
                "evidence.source.from_line",
                minimum=1,
                maximum=MAX_LINE_NUMBER,
            ),
            to_line=_integer(
                payload["to_line"],
                "evidence.source.to_line",
                minimum=1,
                maximum=MAX_LINE_NUMBER,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "target_materialization_id": self.target_materialization_id,
            "revision": self.revision,
            "path": self.path,
            "from_line": self.from_line,
            "to_line": self.to_line,
        }


@dataclass(frozen=True)
class RepositoryDiffEvidenceSource(_JsonModel):
    kind: EvidenceKind
    target_materialization_id: str
    base_revision: str
    head_revision: str
    path: str

    def __post_init__(self) -> None:
        if self.kind is not EvidenceKind.REPOSITORY_DIFF:
            raise _error("repository_diff evidence.source.kind is invalid")
        _identifier(
            self.target_materialization_id,
            "evidence.source.target_materialization_id",
        )
        _identifier(self.base_revision, "evidence.source.base_revision")
        _identifier(self.head_revision, "evidence.source.head_revision")
        _required_loose_submission_path(self.path, "evidence.source.path")

    @classmethod
    def from_dict(cls, value: Any) -> "RepositoryDiffEvidenceSource":
        payload = _object(value, "evidence.source")
        _exact_fields(
            payload,
            (
                "kind",
                "target_materialization_id",
                "base_revision",
                "head_revision",
                "path",
            ),
            "evidence.source",
        )
        return cls(
            kind=_enum_value(EvidenceKind, payload["kind"], "evidence.source.kind"),
            target_materialization_id=_identifier(
                payload["target_materialization_id"],
                "evidence.source.target_materialization_id",
            ),
            base_revision=_identifier(
                payload["base_revision"], "evidence.source.base_revision"
            ),
            head_revision=_identifier(
                payload["head_revision"], "evidence.source.head_revision"
            ),
            path=_required_loose_submission_path(
                payload["path"], "evidence.source.path"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "target_materialization_id": self.target_materialization_id,
            "base_revision": self.base_revision,
            "head_revision": self.head_revision,
            "path": self.path,
        }


@dataclass(frozen=True)
class FrozenContextEvidenceSource(_JsonModel):
    kind: EvidenceKind
    target_materialization_id: str
    context_ref: str
    from_line: int
    to_line: int

    def __post_init__(self) -> None:
        if self.kind is not EvidenceKind.FROZEN_CONTEXT:
            raise _error("frozen_context evidence.source.kind is invalid")
        _identifier(
            self.target_materialization_id,
            "evidence.source.target_materialization_id",
        )
        _identifier(self.context_ref, "evidence.source.context_ref")
        _integer(
            self.from_line,
            "evidence.source.from_line",
            minimum=1,
            maximum=MAX_LINE_NUMBER,
        )
        _integer(
            self.to_line,
            "evidence.source.to_line",
            minimum=1,
            maximum=MAX_LINE_NUMBER,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "FrozenContextEvidenceSource":
        payload = _object(value, "evidence.source")
        _exact_fields(
            payload,
            (
                "kind",
                "target_materialization_id",
                "context_ref",
                "from_line",
                "to_line",
            ),
            "evidence.source",
        )
        return cls(
            kind=_enum_value(EvidenceKind, payload["kind"], "evidence.source.kind"),
            target_materialization_id=_identifier(
                payload["target_materialization_id"],
                "evidence.source.target_materialization_id",
            ),
            context_ref=_identifier(
                payload["context_ref"], "evidence.source.context_ref"
            ),
            from_line=_integer(
                payload["from_line"],
                "evidence.source.from_line",
                minimum=1,
                maximum=MAX_LINE_NUMBER,
            ),
            to_line=_integer(
                payload["to_line"],
                "evidence.source.to_line",
                minimum=1,
                maximum=MAX_LINE_NUMBER,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "target_materialization_id": self.target_materialization_id,
            "context_ref": self.context_ref,
            "from_line": self.from_line,
            "to_line": self.to_line,
        }


@dataclass(frozen=True)
class CommandOutputEvidenceSource(_JsonModel):
    kind: EvidenceKind
    target_materialization_id: str
    command: Tuple[str, ...]
    exit_code: int
    stream: EvidenceStream
    artifact_ref: str

    def __post_init__(self) -> None:
        if self.kind is not EvidenceKind.COMMAND_OUTPUT:
            raise _error("command_output evidence.source.kind is invalid")
        _identifier(
            self.target_materialization_id,
            "evidence.source.target_materialization_id",
        )
        raw_command = _sequence(
            self.command, "evidence.source.command", MAX_COMMAND_ARGUMENTS
        )
        if not raw_command:
            raise _error("evidence.source.command must be non-empty")
        command = tuple(
            _string(
                item,
                "evidence.source.command[%d]" % index,
                MAX_CLAIM_CHARS,
                allow_empty=True,
            )
            for index, item in enumerate(raw_command)
        )
        _integer(
            self.exit_code,
            "evidence.source.exit_code",
            minimum=-MAX_COUNTER,
            maximum=MAX_COUNTER,
        )
        _require_enum(EvidenceStream, self.stream, "evidence.source.stream")
        _identifier(self.artifact_ref, "evidence.source.artifact_ref")
        object.__setattr__(self, "command", command)

    @classmethod
    def from_dict(cls, value: Any) -> "CommandOutputEvidenceSource":
        payload = _object(value, "evidence.source")
        _exact_fields(
            payload,
            (
                "kind",
                "target_materialization_id",
                "command",
                "exit_code",
                "stream",
                "artifact_ref",
            ),
            "evidence.source",
        )
        command = _array(
            payload["command"], "evidence.source.command", MAX_COMMAND_ARGUMENTS
        )
        return cls(
            kind=_enum_value(EvidenceKind, payload["kind"], "evidence.source.kind"),
            target_materialization_id=_identifier(
                payload["target_materialization_id"],
                "evidence.source.target_materialization_id",
            ),
            command=tuple(command),
            exit_code=_integer(
                payload["exit_code"],
                "evidence.source.exit_code",
                minimum=-MAX_COUNTER,
                maximum=MAX_COUNTER,
            ),
            stream=_enum_value(
                EvidenceStream, payload["stream"], "evidence.source.stream"
            ),
            artifact_ref=_identifier(
                payload["artifact_ref"], "evidence.source.artifact_ref"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "target_materialization_id": self.target_materialization_id,
            "command": list(self.command),
            "exit_code": self.exit_code,
            "stream": self.stream.value,
            "artifact_ref": self.artifact_ref,
        }


@dataclass(frozen=True)
class ExternalRecordEvidenceSource(_JsonModel):
    kind: EvidenceKind
    target_materialization_id: str
    source_ref: str

    def __post_init__(self) -> None:
        if self.kind is not EvidenceKind.EXTERNAL_RECORD:
            raise _error("external_record evidence.source.kind is invalid")
        _identifier(
            self.target_materialization_id,
            "evidence.source.target_materialization_id",
        )
        _identifier(self.source_ref, "evidence.source.source_ref")

    @classmethod
    def from_dict(cls, value: Any) -> "ExternalRecordEvidenceSource":
        payload = _object(value, "evidence.source")
        _exact_fields(
            payload,
            ("kind", "target_materialization_id", "source_ref"),
            "evidence.source",
        )
        return cls(
            kind=_enum_value(EvidenceKind, payload["kind"], "evidence.source.kind"),
            target_materialization_id=_identifier(
                payload["target_materialization_id"],
                "evidence.source.target_materialization_id",
            ),
            source_ref=_identifier(
                payload["source_ref"], "evidence.source.source_ref"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "target_materialization_id": self.target_materialization_id,
            "source_ref": self.source_ref,
        }


EvidenceSourceV2 = Union[
    RepositoryFileEvidenceSource,
    RepositoryDiffEvidenceSource,
    FrozenContextEvidenceSource,
    CommandOutputEvidenceSource,
    ExternalRecordEvidenceSource,
]


_EVIDENCE_SOURCE_TYPES = (
    RepositoryFileEvidenceSource,
    RepositoryDiffEvidenceSource,
    FrozenContextEvidenceSource,
    CommandOutputEvidenceSource,
    ExternalRecordEvidenceSource,
)


def _evidence_source_from_dict(value: Any) -> EvidenceSourceV2:
    payload = _object(value, "evidence.source")
    if "kind" not in payload:
        raise _error("evidence.source has missing field(s): kind")
    kind = _enum_value(EvidenceKind, payload["kind"], "evidence.source.kind")
    source_type = {
        EvidenceKind.REPOSITORY_FILE: RepositoryFileEvidenceSource,
        EvidenceKind.REPOSITORY_DIFF: RepositoryDiffEvidenceSource,
        EvidenceKind.FROZEN_CONTEXT: FrozenContextEvidenceSource,
        EvidenceKind.COMMAND_OUTPUT: CommandOutputEvidenceSource,
        EvidenceKind.EXTERNAL_RECORD: ExternalRecordEvidenceSource,
    }[kind]
    return source_type.from_dict(payload)


@dataclass(frozen=True)
class SubmissionEvidence(_JsonModel):
    evidence_id: str
    source: EvidenceSourceV2
    content_hash: str
    excerpt: str

    def __post_init__(self) -> None:
        _identifier(self.evidence_id, "evidence.evidence_id")
        if not isinstance(self.source, _EVIDENCE_SOURCE_TYPES):
            raise _error("evidence.source must be an EvidenceSourceV2")
        _digest(self.content_hash, "evidence.content_hash")
        excerpt = _string(
            self.excerpt,
            "evidence.excerpt",
            MAX_EVIDENCE_EXCERPT_BYTES,
            allow_empty=True,
        )
        if _utf8_size(excerpt, "evidence.excerpt") > MAX_EVIDENCE_EXCERPT_BYTES:
            raise _error(
                "evidence.excerpt exceeds the UTF-8 byte limit of %d"
                % MAX_EVIDENCE_EXCERPT_BYTES
            )

    @classmethod
    def from_dict(cls, value: Any) -> "SubmissionEvidence":
        payload = _object(value, "evidence")
        _exact_fields(
            payload,
            ("evidence_id", "source", "content_hash", "excerpt"),
            "evidence",
        )
        return cls(
            evidence_id=_identifier(payload["evidence_id"], "evidence.evidence_id"),
            source=_evidence_source_from_dict(payload["source"]),
            content_hash=_digest(payload["content_hash"], "evidence.content_hash"),
            excerpt=_string(
                payload["excerpt"],
                "evidence.excerpt",
                MAX_EVIDENCE_EXCERPT_BYTES,
                allow_empty=True,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source.to_dict(),
            "content_hash": self.content_hash,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class SubmissionUsage(_JsonModel):
    elapsed_seconds: Any
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    tool_calls: Optional[int]
    cost_amount: Any
    cost_currency: Optional[str]

    def __post_init__(self) -> None:
        _optional_number(self.elapsed_seconds, "usage.elapsed_seconds")
        input_tokens = _optional_integer(
            self.input_tokens, "usage.input_tokens", minimum=0, maximum=MAX_COUNTER
        )
        output_tokens = _optional_integer(
            self.output_tokens, "usage.output_tokens", minimum=0, maximum=MAX_COUNTER
        )
        total_tokens = _optional_integer(
            self.total_tokens, "usage.total_tokens", minimum=0, maximum=MAX_COUNTER
        )
        _optional_integer(
            self.tool_calls, "usage.tool_calls", minimum=0, maximum=MAX_COUNTER
        )
        _optional_number(self.cost_amount, "usage.cost_amount")
        if self.cost_currency is not None:
            if type(self.cost_currency) is not str or _CURRENCY_RE.fullmatch(
                self.cost_currency
            ) is None:
                raise _error("usage.cost_currency must be an uppercase ISO-4217 token")
        if (self.cost_amount is None) != (self.cost_currency is None):
            raise _error("usage cost_amount and cost_currency must appear together")
        if (
            input_tokens is not None
            and output_tokens is not None
            and total_tokens is not None
            and total_tokens != input_tokens + output_tokens
        ):
            raise _error("usage.total_tokens must equal input_tokens + output_tokens")

    @classmethod
    def from_dict(cls, value: Any) -> "SubmissionUsage":
        payload = _object(value, "usage")
        _exact_fields(
            payload,
            (
                "elapsed_seconds",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "tool_calls",
                "cost_amount",
                "cost_currency",
            ),
            "usage",
        )
        return cls(
            elapsed_seconds=_optional_number(
                payload["elapsed_seconds"], "usage.elapsed_seconds"
            ),
            input_tokens=_optional_integer(
                payload["input_tokens"],
                "usage.input_tokens",
                minimum=0,
                maximum=MAX_COUNTER,
            ),
            output_tokens=_optional_integer(
                payload["output_tokens"],
                "usage.output_tokens",
                minimum=0,
                maximum=MAX_COUNTER,
            ),
            total_tokens=_optional_integer(
                payload["total_tokens"],
                "usage.total_tokens",
                minimum=0,
                maximum=MAX_COUNTER,
            ),
            tool_calls=_optional_integer(
                payload["tool_calls"],
                "usage.tool_calls",
                minimum=0,
                maximum=MAX_COUNTER,
            ),
            cost_amount=_optional_number(payload["cost_amount"], "usage.cost_amount"),
            cost_currency=(
                None
                if payload["cost_currency"] is None
                else _string(payload["cost_currency"], "usage.cost_currency", 3)
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "cost_amount": self.cost_amount,
            "cost_currency": self.cost_currency,
        }


@dataclass(frozen=True)
class TraceRef(_JsonModel):
    type: TraceType
    value: str

    def __post_init__(self) -> None:
        _require_enum(TraceType, self.type, "trace_ref.type")
        if self.type is TraceType.OPAQUE_ID:
            _identifier(self.value, "trace_ref.value")
        else:
            maximum = (
                MAX_URL_CHARS
                if self.type is TraceType.URL
                else MAX_REPOSITORY_PATH_CHARS
            )
            _string(self.value, "trace_ref.value", maximum)

    @classmethod
    def from_dict(cls, value: Any) -> "TraceRef":
        payload = _object(value, "trace_ref")
        _exact_fields(payload, ("type", "value"), "trace_ref")
        trace_type = _enum_value(TraceType, payload["type"], "trace_ref.type")
        if trace_type is TraceType.OPAQUE_ID:
            trace_value = _identifier(payload["value"], "trace_ref.value")
        else:
            maximum = (
                MAX_URL_CHARS
                if trace_type is TraceType.URL
                else MAX_REPOSITORY_PATH_CHARS
            )
            trace_value = _string(payload["value"], "trace_ref.value", maximum)
        return cls(
            type=trace_type,
            value=trace_value,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "value": self.value}


@dataclass(frozen=True)
class SubmissionFailure(_JsonModel):
    code: FailureCode
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        _require_enum(FailureCode, self.code, "failure.code")
        _string(self.message, "failure.message", MAX_DESCRIPTION_CHARS)
        _boolean(self.retryable, "failure.retryable")

    @classmethod
    def from_dict(cls, value: Any) -> "SubmissionFailure":
        payload = _object(value, "failure")
        _exact_fields(payload, ("code", "message", "retryable"), "failure")
        return cls(
            code=_enum_value(FailureCode, payload["code"], "failure.code"),
            message=_string(
                payload["message"], "failure.message", MAX_DESCRIPTION_CHARS
            ),
            retryable=_boolean(payload["retryable"], "failure.retryable"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
        }


_FAILURE_STATUS_BY_CODE = {
    FailureCode.TIMEOUT: SubmissionStatus.FAILED,
    FailureCode.NON_ZERO_EXIT: SubmissionStatus.FAILED,
    FailureCode.PROCESS_KILLED: SubmissionStatus.FAILED,
    FailureCode.ADAPTER_ERROR: SubmissionStatus.FAILED,
    FailureCode.HARNESS_MATERIALIZATION_ERROR: SubmissionStatus.FAILED,
    FailureCode.UNKNOWN: SubmissionStatus.FAILED,
    FailureCode.CLARIFICATION_REQUIRED: SubmissionStatus.BLOCKED,
    FailureCode.AGENT_BLOCKED: SubmissionStatus.BLOCKED,
    FailureCode.INVALID_JSON: SubmissionStatus.INVALID_OUTPUT,
    FailureCode.SCHEMA_MISMATCH: SubmissionStatus.INVALID_OUTPUT,
    FailureCode.OUTPUT_OVERFLOW: SubmissionStatus.INVALID_OUTPUT,
}


def submission_status_for_failure(code: FailureCode) -> SubmissionStatus:
    """Return the one canonical terminal Submission status for ``code``."""

    if not isinstance(code, FailureCode):
        raise TypeError("failure code must be FailureCode")
    try:
        return _FAILURE_STATUS_BY_CODE[code]
    except KeyError as exc:
        raise ValueError(
            "failure code has no canonical terminal Submission status"
        ) from exc


@dataclass(frozen=True)
class EvalSubmission(_JsonModel):
    SCHEMA_VERSION: ClassVar[str] = EVAL_SUBMISSION_SCHEMA_VERSION

    schema_version: str
    task_id: str
    agent_id: str
    trial_id: str
    eval_input_digest: str
    target_materialization_id: str
    status: SubmissionStatus
    intent: Optional[SubmissionIntent]
    review: Optional[SubmissionReview]
    evidence: Tuple[SubmissionEvidence, ...]
    usage: SubmissionUsage
    trace_ref: Optional[TraceRef]
    failure: Optional[SubmissionFailure]

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise UnsupportedProtocolVersionError(
                expected=self.SCHEMA_VERSION, actual=self.schema_version
            )
        _identifier(self.task_id, "submission.task_id")
        _identifier(self.agent_id, "submission.agent_id")
        _identifier(self.trial_id, "submission.trial_id")
        _digest(self.eval_input_digest, "submission.eval_input_digest")
        _identifier(
            self.target_materialization_id,
            "submission.target_materialization_id",
        )
        _require_enum(SubmissionStatus, self.status, "submission.status")
        if self.intent is not None and not isinstance(self.intent, SubmissionIntent):
            raise _error("submission.intent must be SubmissionIntent or null")
        if self.review is not None and not isinstance(self.review, SubmissionReview):
            raise _error("submission.review must be SubmissionReview or null")
        evidence = _model_tuple(
            self.evidence,
            SubmissionEvidence,
            "submission.evidence",
            MAX_EVIDENCE_ITEMS,
        )
        _unique_by(evidence, "evidence_id", "submission.evidence")
        if any(
            item.source.target_materialization_id
            != self.target_materialization_id
            for item in evidence
        ):
            raise _error(
                "evidence source target_materialization_id does not match submission binding"
            )
        if not isinstance(self.usage, SubmissionUsage):
            raise _error("submission.usage must be a SubmissionUsage")
        if self.trace_ref is not None and not isinstance(self.trace_ref, TraceRef):
            raise _error("submission.trace_ref must be TraceRef or null")
        if self.failure is not None and not isinstance(self.failure, SubmissionFailure):
            raise _error("submission.failure must be SubmissionFailure or null")

        if self.status is SubmissionStatus.COMPLETED:
            if self.failure is not None or self.intent is None or self.review is None:
                raise _error(
                    "completed submission requires intent/review and failure=null"
                )
        elif self.status is SubmissionStatus.FAILED:
            if self.failure is None or submission_status_for_failure(
                self.failure.code
            ) is not SubmissionStatus.FAILED:
                raise _error("failed submission has an invalid or missing failure code")
            if (
                self.failure.code is FailureCode.HARNESS_MATERIALIZATION_ERROR
                and (self.intent is not None or self.review is not None)
            ):
                raise _error(
                    "harness_materialization_error requires intent=null and review=null"
                )
        elif self.status is SubmissionStatus.BLOCKED:
            if self.failure is None or submission_status_for_failure(
                self.failure.code
            ) is not SubmissionStatus.BLOCKED:
                raise _error("blocked submission has an invalid or missing failure code")
            if self.failure.code is FailureCode.CLARIFICATION_REQUIRED:
                unresolved = (
                    self.intent is not None
                    and any(
                        exchange.action in (None, ClarificationAction.DEFER)
                        for exchange in self.intent.clarification_questions
                    )
                )
                if not unresolved:
                    raise _error(
                        "clarification_required needs Intent with an unresolved exchange"
                    )
        elif self.status is SubmissionStatus.INVALID_OUTPUT:
            if self.failure is None or submission_status_for_failure(
                self.failure.code
            ) is not SubmissionStatus.INVALID_OUTPUT:
                raise _error(
                    "invalid_output submission has an invalid or missing failure code"
                )
            if self.intent is not None or self.review is not None:
                raise _error("invalid_output must not fabricate partial intent or review")
        object.__setattr__(self, "evidence", _sorted_by(evidence, "evidence_id"))
        _check_model_size(self, MAX_EVAL_SUBMISSION_BYTES, "EvalSubmission")

    @classmethod
    def from_dict(cls, value: Any) -> "EvalSubmission":
        payload = _object(value, "EvalSubmission")
        _require_root_schema_version(payload, cls.SCHEMA_VERSION, "EvalSubmission")
        _exact_fields(
            payload,
            (
                "schema_version",
                "task_id",
                "agent_id",
                "trial_id",
                "eval_input_digest",
                "target_materialization_id",
                "status",
                "intent",
                "review",
                "evidence",
                "usage",
                "trace_ref",
                "failure",
            ),
            "EvalSubmission",
        )
        evidence = _array(
            payload["evidence"], "submission.evidence", MAX_EVIDENCE_ITEMS
        )
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            task_id=_identifier(payload["task_id"], "submission.task_id"),
            agent_id=_identifier(payload["agent_id"], "submission.agent_id"),
            trial_id=_identifier(payload["trial_id"], "submission.trial_id"),
            eval_input_digest=_digest(
                payload["eval_input_digest"], "submission.eval_input_digest"
            ),
            target_materialization_id=_identifier(
                payload["target_materialization_id"],
                "submission.target_materialization_id",
            ),
            status=_enum_value(
                SubmissionStatus, payload["status"], "submission.status"
            ),
            intent=(
                None
                if payload["intent"] is None
                else SubmissionIntent.from_dict(payload["intent"])
            ),
            review=(
                None
                if payload["review"] is None
                else SubmissionReview.from_dict(payload["review"])
            ),
            evidence=tuple(SubmissionEvidence.from_dict(item) for item in evidence),
            usage=SubmissionUsage.from_dict(payload["usage"]),
            trace_ref=(
                None
                if payload["trace_ref"] is None
                else TraceRef.from_dict(payload["trace_ref"])
            ),
            failure=(
                None
                if payload["failure"] is None
                else SubmissionFailure.from_dict(payload["failure"])
            ),
        )

    @classmethod
    def from_json(cls, data: Any) -> "EvalSubmission":
        return cls.from_dict(
            _strict_json_loads(data, MAX_EVAL_SUBMISSION_BYTES, "EvalSubmission JSON")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "trial_id": self.trial_id,
            "eval_input_digest": self.eval_input_digest,
            "target_materialization_id": self.target_materialization_id,
            "status": self.status.value,
            "intent": None if self.intent is None else self.intent.to_dict(),
            "review": None if self.review is None else self.review.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "usage": self.usage.to_dict(),
            "trace_ref": None if self.trace_ref is None else self.trace_ref.to_dict(),
            "failure": None if self.failure is None else self.failure.to_dict(),
        }


@dataclass(frozen=True)
class CaseSource(_JsonModel):
    suite: str
    origin: CaseOrigin
    source_id: str
    source_version: str
    source_uri: Optional[str]
    license: Optional[str]
    content_hash: str

    def __post_init__(self) -> None:
        _identifier(self.suite, "case.source.suite")
        _require_enum(CaseOrigin, self.origin, "case.source.origin")
        _identifier(self.source_id, "case.source.source_id")
        _identifier(self.source_version, "case.source.source_version")
        _optional_string(self.source_uri, "case.source.source_uri", MAX_URL_CHARS)
        _optional_string(self.license, "case.source.license", MAX_IDENTIFIER_CHARS)
        _digest(self.content_hash, "case.source.content_hash")

    @classmethod
    def from_dict(cls, value: Any) -> "CaseSource":
        payload = _object(value, "case.source")
        _exact_fields(
            payload,
            (
                "suite",
                "origin",
                "source_id",
                "source_version",
                "source_uri",
                "license",
                "content_hash",
            ),
            "case.source",
        )
        return cls(
            suite=_identifier(payload["suite"], "case.source.suite"),
            origin=_enum_value(CaseOrigin, payload["origin"], "case.source.origin"),
            source_id=_identifier(payload["source_id"], "case.source.source_id"),
            source_version=_identifier(
                payload["source_version"], "case.source.source_version"
            ),
            source_uri=_optional_string(
                payload["source_uri"], "case.source.source_uri", MAX_URL_CHARS
            ),
            license=_optional_string(
                payload["license"], "case.source.license", MAX_IDENTIFIER_CHARS
            ),
            content_hash=_digest(
                payload["content_hash"], "case.source.content_hash"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite,
            "origin": self.origin.value,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "source_uri": self.source_uri,
            "license": self.license,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class ClarificationAnswer(_JsonModel):
    answer_id: str
    dimension: IntentDimension
    material_claim: str
    action: ClarificationAction
    response: Optional[str]
    corrected_values: Tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.answer_id, "clarification answer.answer_id")
        _require_enum(
            IntentDimension, self.dimension, "clarification answer.dimension"
        )
        _string(
            self.material_claim,
            "clarification answer.material_claim",
            MAX_CLAIM_CHARS,
        )
        _require_enum(ClarificationAction, self.action, "clarification answer.action")
        _optional_string(
            self.response, "clarification answer.response", MAX_ANSWER_CHARS
        )
        corrected = _text_tuple(
            self.corrected_values,
            "clarification answer.corrected_values",
            MAX_TEXT_LIST_ITEMS,
            MAX_ANSWER_CHARS,
        )
        if self.action is ClarificationAction.CORRECT:
            if self.response is None or not corrected:
                raise _error(
                    "correct clarification answer requires response and corrected_values"
                )
        elif corrected:
            raise _error(
                "only correct clarification answers may contain corrected_values"
            )
        object.__setattr__(self, "corrected_values", corrected)

    @classmethod
    def from_dict(cls, value: Any) -> "ClarificationAnswer":
        payload = _object(value, "clarification answer")
        _exact_fields(
            payload,
            (
                "answer_id",
                "dimension",
                "material_claim",
                "action",
                "response",
                "corrected_values",
            ),
            "clarification answer",
        )
        corrected = _array(
            payload["corrected_values"],
            "clarification answer.corrected_values",
            MAX_TEXT_LIST_ITEMS,
        )
        return cls(
            answer_id=_identifier(
                payload["answer_id"], "clarification answer.answer_id"
            ),
            dimension=_enum_value(
                IntentDimension,
                payload["dimension"],
                "clarification answer.dimension",
            ),
            material_claim=_string(
                payload["material_claim"],
                "clarification answer.material_claim",
                MAX_CLAIM_CHARS,
            ),
            action=_enum_value(
                ClarificationAction,
                payload["action"],
                "clarification answer.action",
            ),
            response=_optional_string(
                payload["response"],
                "clarification answer.response",
                MAX_ANSWER_CHARS,
            ),
            corrected_values=tuple(corrected),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "dimension": self.dimension.value,
            "material_claim": self.material_claim,
            "action": self.action.value,
            "response": self.response,
            "corrected_values": list(self.corrected_values),
        }


@dataclass(frozen=True)
class ClarificationScript(_JsonModel):
    max_rounds: int
    answers: Tuple[ClarificationAnswer, ...]

    def __post_init__(self) -> None:
        _integer(
            self.max_rounds,
            "clarification_script.max_rounds",
            minimum=1,
            maximum=16,
        )
        answers = _model_tuple(
            self.answers,
            ClarificationAnswer,
            "clarification_script.answers",
            MAX_CLARIFICATION_ANSWERS,
        )
        _unique_by(answers, "answer_id", "clarification_script.answers")
        object.__setattr__(self, "answers", _sorted_by(answers, "answer_id"))

    @classmethod
    def from_dict(cls, value: Any) -> "ClarificationScript":
        payload = _object(value, "clarification_script")
        _exact_fields(payload, ("max_rounds", "answers"), "clarification_script")
        answers = _array(
            payload["answers"],
            "clarification_script.answers",
            MAX_CLARIFICATION_ANSWERS,
        )
        return cls(
            max_rounds=_integer(
                payload["max_rounds"],
                "clarification_script.max_rounds",
                minimum=1,
                maximum=16,
            ),
            answers=tuple(ClarificationAnswer.from_dict(item) for item in answers),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "answers": [item.to_dict() for item in self.answers],
        }


@dataclass(frozen=True)
class ExpectedIntentClaim(_JsonModel):
    truth_id: str
    dimension: IntentDimension
    text: str
    required: bool

    def __post_init__(self) -> None:
        _identifier(self.truth_id, "expected intent claim.truth_id")
        _require_enum(
            IntentDimension, self.dimension, "expected intent claim.dimension"
        )
        _string(self.text, "expected intent claim.text", MAX_CLAIM_CHARS)
        _boolean(self.required, "expected intent claim.required")

    @classmethod
    def from_dict(cls, value: Any) -> "ExpectedIntentClaim":
        payload = _object(value, "expected intent claim")
        _exact_fields(
            payload, ("truth_id", "dimension", "text", "required"), "expected intent claim"
        )
        return cls(
            truth_id=_identifier(
                payload["truth_id"], "expected intent claim.truth_id"
            ),
            dimension=_enum_value(
                IntentDimension,
                payload["dimension"],
                "expected intent claim.dimension",
            ),
            text=_string(
                payload["text"], "expected intent claim.text", MAX_CLAIM_CHARS
            ),
            required=_boolean(
                payload["required"], "expected intent claim.required"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "truth_id": self.truth_id,
            "dimension": self.dimension.value,
            "text": self.text,
            "required": self.required,
        }


@dataclass(frozen=True)
class ForbiddenIntentClaim(_JsonModel):
    truth_id: str
    dimension: IntentDimension
    text: str
    rationale: str

    def __post_init__(self) -> None:
        _identifier(self.truth_id, "forbidden intent claim.truth_id")
        _require_enum(
            IntentDimension, self.dimension, "forbidden intent claim.dimension"
        )
        _string(self.text, "forbidden intent claim.text", MAX_CLAIM_CHARS)
        _string(
            self.rationale, "forbidden intent claim.rationale", MAX_RATIONALE_CHARS
        )

    @classmethod
    def from_dict(cls, value: Any) -> "ForbiddenIntentClaim":
        payload = _object(value, "forbidden intent claim")
        _exact_fields(
            payload,
            ("truth_id", "dimension", "text", "rationale"),
            "forbidden intent claim",
        )
        return cls(
            truth_id=_identifier(
                payload["truth_id"], "forbidden intent claim.truth_id"
            ),
            dimension=_enum_value(
                IntentDimension,
                payload["dimension"],
                "forbidden intent claim.dimension",
            ),
            text=_string(
                payload["text"], "forbidden intent claim.text", MAX_CLAIM_CHARS
            ),
            rationale=_string(
                payload["rationale"],
                "forbidden intent claim.rationale",
                MAX_RATIONALE_CHARS,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "truth_id": self.truth_id,
            "dimension": self.dimension.value,
            "text": self.text,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class IntentTruth(_JsonModel):
    scorable: bool
    authority: Optional[IntentAuthority]
    expected_claims: Tuple[ExpectedIntentClaim, ...]
    forbidden_claims: Tuple[ForbiddenIntentClaim, ...]
    clarification_policy: Optional[ClarificationPolicy]

    def __post_init__(self) -> None:
        _boolean(self.scorable, "intent_truth.scorable")
        _require_optional_enum(IntentAuthority, self.authority, "intent_truth.authority")
        expected = _model_tuple(
            self.expected_claims,
            ExpectedIntentClaim,
            "intent_truth.expected_claims",
            MAX_INTENT_CLAIMS,
        )
        forbidden = _model_tuple(
            self.forbidden_claims,
            ForbiddenIntentClaim,
            "intent_truth.forbidden_claims",
            MAX_INTENT_CLAIMS,
        )
        if len(expected) + len(forbidden) > MAX_INTENT_CLAIMS:
            raise _error(
                "intent_truth claims exceed the item limit of %d"
                % MAX_INTENT_CLAIMS
            )
        _unique_by(expected, "truth_id", "intent_truth.expected_claims")
        _unique_by(forbidden, "truth_id", "intent_truth.forbidden_claims")
        if {item.truth_id for item in expected}.intersection(
            item.truth_id for item in forbidden
        ):
            raise _error(
                "intent_truth contains duplicate truth IDs across expected and forbidden claims"
            )
        _require_optional_enum(
            ClarificationPolicy,
            self.clarification_policy,
            "intent_truth.clarification_policy",
        )
        if self.scorable:
            if self.authority is None or self.clarification_policy is None:
                raise _error(
                    "scorable intent truth requires authority and clarification_policy"
                )
        elif (
            self.authority is not None
            or expected
            or forbidden
            or self.clarification_policy is not None
        ):
            raise _error(
                "unscorable intent truth requires null authority/policy and empty claims"
            )
        object.__setattr__(self, "expected_claims", _sorted_by(expected, "truth_id"))
        object.__setattr__(self, "forbidden_claims", _sorted_by(forbidden, "truth_id"))

    @classmethod
    def from_dict(cls, value: Any) -> "IntentTruth":
        payload = _object(value, "intent_truth")
        _exact_fields(
            payload,
            (
                "scorable",
                "authority",
                "expected_claims",
                "forbidden_claims",
                "clarification_policy",
            ),
            "intent_truth",
        )
        expected = _array(
            payload["expected_claims"],
            "intent_truth.expected_claims",
            MAX_INTENT_CLAIMS,
        )
        forbidden = _array(
            payload["forbidden_claims"],
            "intent_truth.forbidden_claims",
            MAX_INTENT_CLAIMS,
        )
        if len(expected) + len(forbidden) > MAX_INTENT_CLAIMS:
            raise _error(
                "intent_truth claims exceed the item limit of %d"
                % MAX_INTENT_CLAIMS
            )
        return cls(
            scorable=_boolean(payload["scorable"], "intent_truth.scorable"),
            authority=_optional_enum(
                IntentAuthority, payload["authority"], "intent_truth.authority"
            ),
            expected_claims=tuple(
                ExpectedIntentClaim.from_dict(item) for item in expected
            ),
            forbidden_claims=tuple(
                ForbiddenIntentClaim.from_dict(item) for item in forbidden
            ),
            clarification_policy=_optional_enum(
                ClarificationPolicy,
                payload["clarification_policy"],
                "intent_truth.clarification_policy",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scorable": self.scorable,
            "authority": None if self.authority is None else self.authority.value,
            "expected_claims": [item.to_dict() for item in self.expected_claims],
            "forbidden_claims": [item.to_dict() for item in self.forbidden_claims],
            "clarification_policy": (
                None
                if self.clarification_policy is None
                else self.clarification_policy.value
            ),
        }


@dataclass(frozen=True)
class TruthLocation(_JsonModel):
    path: str
    side: Optional[DiffSide]
    from_line: Optional[int]
    to_line: Optional[int]

    def __post_init__(self) -> None:
        _safe_repo_path(self.path, "truth location.path")
        _require_optional_enum(DiffSide, self.side, "truth location.side")
        from_line = _optional_integer(
            self.from_line,
            "truth location.from_line",
            minimum=1,
            maximum=MAX_LINE_NUMBER,
        )
        to_line = _optional_integer(
            self.to_line,
            "truth location.to_line",
            minimum=1,
            maximum=MAX_LINE_NUMBER,
        )
        if (from_line is None) != (to_line is None):
            raise _error("truth location lines must both be null or both be present")
        if from_line is not None and to_line is not None and to_line < from_line:
            raise _error("truth location.to_line must be >= from_line")

    @classmethod
    def from_dict(cls, value: Any) -> "TruthLocation":
        payload = _object(value, "truth location")
        _exact_fields(
            payload, ("path", "side", "from_line", "to_line"), "truth location"
        )
        return cls(
            path=_safe_repo_path(payload["path"], "truth location.path"),
            side=_optional_enum(DiffSide, payload["side"], "truth location.side"),
            from_line=_optional_integer(
                payload["from_line"],
                "truth location.from_line",
                minimum=1,
                maximum=MAX_LINE_NUMBER,
            ),
            to_line=_optional_integer(
                payload["to_line"],
                "truth location.to_line",
                minimum=1,
                maximum=MAX_LINE_NUMBER,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "side": None if self.side is None else self.side.value,
            "from_line": self.from_line,
            "to_line": self.to_line,
        }


def _canonical_model_sort(values: Iterable[_JsonModel]) -> Tuple[Any, ...]:
    return tuple(sorted(values, key=lambda item: canonical_json(item.to_dict())))


@dataclass(frozen=True)
class EvidenceAnchor(_JsonModel):
    fact: str
    locations: Tuple[TruthLocation, ...]

    def __post_init__(self) -> None:
        _string(self.fact, "evidence anchor.fact", MAX_CLAIM_CHARS)
        locations = _model_tuple(
            self.locations,
            TruthLocation,
            "evidence anchor.locations",
            MAX_TRUTH_LOCATIONS,
        )
        object.__setattr__(self, "locations", _canonical_model_sort(locations))

    @classmethod
    def from_dict(cls, value: Any) -> "EvidenceAnchor":
        payload = _object(value, "evidence anchor")
        _exact_fields(payload, ("fact", "locations"), "evidence anchor")
        locations = _array(
            payload["locations"], "evidence anchor.locations", MAX_TRUTH_LOCATIONS
        )
        return cls(
            fact=_string(payload["fact"], "evidence anchor.fact", MAX_CLAIM_CHARS),
            locations=tuple(TruthLocation.from_dict(item) for item in locations),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact": self.fact,
            "locations": [item.to_dict() for item in self.locations],
        }


@dataclass(frozen=True)
class MetricAuthority(_JsonModel):
    severity_scorable: bool
    severity_authority: Optional[MetricAuthoritySource]
    location_scorable: bool
    location_authority: Optional[MetricAuthoritySource]

    def __post_init__(self) -> None:
        _boolean(self.severity_scorable, "metric_authority.severity_scorable")
        _require_optional_enum(
            MetricAuthoritySource,
            self.severity_authority,
            "metric_authority.severity_authority",
        )
        _boolean(self.location_scorable, "metric_authority.location_scorable")
        _require_optional_enum(
            MetricAuthoritySource,
            self.location_authority,
            "metric_authority.location_authority",
        )
        if self.severity_scorable != (self.severity_authority is not None):
            raise _error(
                "severity_authority must be present exactly when severity is scorable"
            )
        if self.location_scorable != (self.location_authority is not None):
            raise _error(
                "location_authority must be present exactly when location is scorable"
            )

    @classmethod
    def from_dict(cls, value: Any) -> "MetricAuthority":
        payload = _object(value, "metric_authority")
        _exact_fields(
            payload,
            (
                "severity_scorable",
                "severity_authority",
                "location_scorable",
                "location_authority",
            ),
            "metric_authority",
        )
        return cls(
            severity_scorable=_boolean(
                payload["severity_scorable"],
                "metric_authority.severity_scorable",
            ),
            severity_authority=_optional_enum(
                MetricAuthoritySource,
                payload["severity_authority"],
                "metric_authority.severity_authority",
            ),
            location_scorable=_boolean(
                payload["location_scorable"],
                "metric_authority.location_scorable",
            ),
            location_authority=_optional_enum(
                MetricAuthoritySource,
                payload["location_authority"],
                "metric_authority.location_authority",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity_scorable": self.severity_scorable,
            "severity_authority": (
                None
                if self.severity_authority is None
                else self.severity_authority.value
            ),
            "location_scorable": self.location_scorable,
            "location_authority": (
                None
                if self.location_authority is None
                else self.location_authority.value
            ),
        }


@dataclass(frozen=True)
class ExpectedFinding(_JsonModel):
    truth_id: str
    claim: str
    severity: Optional[FindingSeverity]
    category: str
    required: bool
    metric_authority: MetricAuthority
    locations: Tuple[TruthLocation, ...]
    evidence_anchors: Tuple[EvidenceAnchor, ...]
    required_context_level: RequiredContextLevel
    rationale: str

    def __post_init__(self) -> None:
        _identifier(self.truth_id, "expected finding.truth_id")
        _string(self.claim, "expected finding.claim", MAX_CLAIM_CHARS)
        _require_optional_enum(
            FindingSeverity, self.severity, "expected finding.severity"
        )
        _string(self.category, "expected finding.category", MAX_IDENTIFIER_CHARS)
        _boolean(self.required, "expected finding.required")
        if not isinstance(self.metric_authority, MetricAuthority):
            raise _error(
                "expected finding.metric_authority must be a MetricAuthority"
            )
        locations = _model_tuple(
            self.locations,
            TruthLocation,
            "expected finding.locations",
            MAX_TRUTH_LOCATIONS,
        )
        anchors = _model_tuple(
            self.evidence_anchors,
            EvidenceAnchor,
            "expected finding.evidence_anchors",
            MAX_EVIDENCE_ANCHORS,
        )
        if self.metric_authority.severity_scorable != (self.severity is not None):
            raise _error(
                "expected finding severity must be present exactly when scorable"
            )
        if self.metric_authority.location_scorable and not any(
            location.from_line is not None and location.to_line is not None
            for location in locations
        ):
            raise _error(
                "location-scorable expected finding requires a complete truth location"
            )
        _require_enum(
            RequiredContextLevel,
            self.required_context_level,
            "expected finding.required_context_level",
        )
        _string(
            self.rationale, "expected finding.rationale", MAX_RATIONALE_CHARS
        )
        object.__setattr__(self, "locations", _canonical_model_sort(locations))
        object.__setattr__(self, "evidence_anchors", _canonical_model_sort(anchors))

    @classmethod
    def from_dict(cls, value: Any) -> "ExpectedFinding":
        payload = _object(value, "expected finding")
        _exact_fields(
            payload,
            (
                "truth_id",
                "claim",
                "severity",
                "category",
                "required",
                "metric_authority",
                "locations",
                "evidence_anchors",
                "required_context_level",
                "rationale",
            ),
            "expected finding",
        )
        locations = _array(
            payload["locations"], "expected finding.locations", MAX_TRUTH_LOCATIONS
        )
        anchors = _array(
            payload["evidence_anchors"],
            "expected finding.evidence_anchors",
            MAX_EVIDENCE_ANCHORS,
        )
        return cls(
            truth_id=_identifier(payload["truth_id"], "expected finding.truth_id"),
            claim=_string(
                payload["claim"], "expected finding.claim", MAX_CLAIM_CHARS
            ),
            severity=_optional_enum(
                FindingSeverity, payload["severity"], "expected finding.severity"
            ),
            category=_string(
                payload["category"],
                "expected finding.category",
                MAX_IDENTIFIER_CHARS,
            ),
            required=_boolean(payload["required"], "expected finding.required"),
            metric_authority=MetricAuthority.from_dict(
                payload["metric_authority"]
            ),
            locations=tuple(TruthLocation.from_dict(item) for item in locations),
            evidence_anchors=tuple(EvidenceAnchor.from_dict(item) for item in anchors),
            required_context_level=_enum_value(
                RequiredContextLevel,
                payload["required_context_level"],
                "expected finding.required_context_level",
            ),
            rationale=_string(
                payload["rationale"],
                "expected finding.rationale",
                MAX_RATIONALE_CHARS,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "truth_id": self.truth_id,
            "claim": self.claim,
            "severity": None if self.severity is None else self.severity.value,
            "category": self.category,
            "required": self.required,
            "metric_authority": self.metric_authority.to_dict(),
            "locations": [item.to_dict() for item in self.locations],
            "evidence_anchors": [item.to_dict() for item in self.evidence_anchors],
            "required_context_level": self.required_context_level.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class KnownInvalidFinding(_JsonModel):
    truth_id: str
    claim: str
    category: Optional[str]
    locations: Tuple[TruthLocation, ...]
    rationale: str

    def __post_init__(self) -> None:
        _identifier(self.truth_id, "known-invalid finding.truth_id")
        _string(self.claim, "known-invalid finding.claim", MAX_CLAIM_CHARS)
        _optional_string(
            self.category, "known-invalid finding.category", MAX_IDENTIFIER_CHARS
        )
        locations = _model_tuple(
            self.locations,
            TruthLocation,
            "known-invalid finding.locations",
            MAX_TRUTH_LOCATIONS,
        )
        _string(
            self.rationale,
            "known-invalid finding.rationale",
            MAX_RATIONALE_CHARS,
        )
        object.__setattr__(self, "locations", _canonical_model_sort(locations))

    @classmethod
    def from_dict(cls, value: Any) -> "KnownInvalidFinding":
        payload = _object(value, "known-invalid finding")
        _exact_fields(
            payload,
            ("truth_id", "claim", "category", "locations", "rationale"),
            "known-invalid finding",
        )
        locations = _array(
            payload["locations"],
            "known-invalid finding.locations",
            MAX_TRUTH_LOCATIONS,
        )
        return cls(
            truth_id=_identifier(
                payload["truth_id"], "known-invalid finding.truth_id"
            ),
            claim=_string(
                payload["claim"], "known-invalid finding.claim", MAX_CLAIM_CHARS
            ),
            category=_optional_string(
                payload["category"],
                "known-invalid finding.category",
                MAX_IDENTIFIER_CHARS,
            ),
            locations=tuple(TruthLocation.from_dict(item) for item in locations),
            rationale=_string(
                payload["rationale"],
                "known-invalid finding.rationale",
                MAX_RATIONALE_CHARS,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "truth_id": self.truth_id,
            "claim": self.claim,
            "category": self.category,
            "locations": [item.to_dict() for item in self.locations],
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ReviewTruth(_JsonModel):
    completeness: TruthCompleteness
    novel_finding_policy: NovelFindingPolicy
    expected_findings: Tuple[ExpectedFinding, ...]
    known_invalid_findings: Tuple[KnownInvalidFinding, ...]

    def __post_init__(self) -> None:
        _require_enum(TruthCompleteness, self.completeness, "review_truth.completeness")
        _require_enum(
            NovelFindingPolicy,
            self.novel_finding_policy,
            "review_truth.novel_finding_policy",
        )
        expected = _model_tuple(
            self.expected_findings,
            ExpectedFinding,
            "review_truth.expected_findings",
            MAX_TRUTH_FINDINGS,
        )
        invalid = _model_tuple(
            self.known_invalid_findings,
            KnownInvalidFinding,
            "review_truth.known_invalid_findings",
            MAX_TRUTH_FINDINGS,
        )
        if len(expected) + len(invalid) > MAX_TRUTH_FINDINGS:
            raise _error(
                "review_truth truth findings exceed the item limit of %d"
                % MAX_TRUTH_FINDINGS
            )
        _unique_by(expected, "truth_id", "review_truth.expected_findings")
        _unique_by(invalid, "truth_id", "review_truth.known_invalid_findings")
        if {item.truth_id for item in expected}.intersection(
            item.truth_id for item in invalid
        ):
            raise _error(
                "review_truth contains duplicate truth IDs across expected and known-invalid findings"
            )
        if (
            self.novel_finding_policy is NovelFindingPolicy.FORBID
            and self.completeness is not TruthCompleteness.CLOSED_WORLD
        ):
            raise _error("novel finding policy forbid is only valid for closed_world truth")
        object.__setattr__(self, "expected_findings", _sorted_by(expected, "truth_id"))
        object.__setattr__(
            self, "known_invalid_findings", _sorted_by(invalid, "truth_id")
        )

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewTruth":
        payload = _object(value, "review_truth")
        _exact_fields(
            payload,
            (
                "completeness",
                "novel_finding_policy",
                "expected_findings",
                "known_invalid_findings",
            ),
            "review_truth",
        )
        expected = _array(
            payload["expected_findings"],
            "review_truth.expected_findings",
            MAX_TRUTH_FINDINGS,
        )
        invalid = _array(
            payload["known_invalid_findings"],
            "review_truth.known_invalid_findings",
            MAX_TRUTH_FINDINGS,
        )
        if len(expected) + len(invalid) > MAX_TRUTH_FINDINGS:
            raise _error(
                "review_truth truth findings exceed the item limit of %d"
                % MAX_TRUTH_FINDINGS
            )
        return cls(
            completeness=_enum_value(
                TruthCompleteness,
                payload["completeness"],
                "review_truth.completeness",
            ),
            novel_finding_policy=_enum_value(
                NovelFindingPolicy,
                payload["novel_finding_policy"],
                "review_truth.novel_finding_policy",
            ),
            expected_findings=tuple(ExpectedFinding.from_dict(item) for item in expected),
            known_invalid_findings=tuple(
                KnownInvalidFinding.from_dict(item) for item in invalid
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completeness": self.completeness.value,
            "novel_finding_policy": self.novel_finding_policy.value,
            "expected_findings": [item.to_dict() for item in self.expected_findings],
            "known_invalid_findings": [
                item.to_dict() for item in self.known_invalid_findings
            ],
        }


@dataclass(frozen=True)
class EvaluatorContextProvenance(_JsonModel):
    source_role: str
    source_file_sha256: str
    record_pointer: str
    record_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.source_role, "evaluator context provenance.source_role")
        _digest(
            self.source_file_sha256,
            "evaluator context provenance.source_file_sha256",
        )
        _string(
            self.record_pointer,
            "evaluator context provenance.record_pointer",
            MAX_URL_CHARS,
        )
        _digest(
            self.record_sha256,
            "evaluator context provenance.record_sha256",
        )

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluatorContextProvenance":
        payload = _object(value, "evaluator context provenance")
        _exact_fields(
            payload,
            (
                "source_role",
                "source_file_sha256",
                "record_pointer",
                "record_sha256",
            ),
            "evaluator context provenance",
        )
        return cls(
            source_role=_identifier(
                payload["source_role"],
                "evaluator context provenance.source_role",
            ),
            source_file_sha256=_digest(
                payload["source_file_sha256"],
                "evaluator context provenance.source_file_sha256",
            ),
            record_pointer=_string(
                payload["record_pointer"],
                "evaluator context provenance.record_pointer",
                MAX_URL_CHARS,
            ),
            record_sha256=_digest(
                payload["record_sha256"],
                "evaluator context provenance.record_sha256",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_role": self.source_role,
            "source_file_sha256": self.source_file_sha256,
            "record_pointer": self.record_pointer,
            "record_sha256": self.record_sha256,
        }


@dataclass(frozen=True)
class EvaluatorContextSource(_JsonModel):
    kind: EvaluatorContextSourceKind
    content: str
    content_sha256: str
    provenance: EvaluatorContextProvenance

    def __post_init__(self) -> None:
        if self.kind is not EvaluatorContextSourceKind.DIFF_HUNK:
            raise _error("evaluator context source.kind must be diff_hunk")
        content = _string(
            self.content,
            "evaluator context source.content",
            MAX_EVALUATOR_CONTEXT_CONTENT_BYTES,
        )
        if (
            _utf8_size(content, "evaluator context source.content")
            > MAX_EVALUATOR_CONTEXT_CONTENT_BYTES
        ):
            raise _error(
                "evaluator context source.content exceeds the UTF-8 byte limit of %d"
                % MAX_EVALUATOR_CONTEXT_CONTENT_BYTES
            )
        content_sha256 = _digest(
            self.content_sha256,
            "evaluator context source.content_sha256",
        )
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != content_sha256:
            raise _error(
                "evaluator context source.content_sha256 must hash exact UTF-8 content"
            )
        if not isinstance(self.provenance, EvaluatorContextProvenance):
            raise _error(
                "evaluator context source.provenance must be EvaluatorContextProvenance"
            )

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluatorContextSource":
        payload = _object(value, "evaluator context source")
        _exact_fields(
            payload,
            ("kind", "content", "content_sha256", "provenance"),
            "evaluator context source",
        )
        return cls(
            kind=_enum_value(
                EvaluatorContextSourceKind,
                payload["kind"],
                "evaluator context source.kind",
            ),
            content=_string(
                payload["content"],
                "evaluator context source.content",
                MAX_EVALUATOR_CONTEXT_CONTENT_BYTES,
            ),
            content_sha256=_digest(
                payload["content_sha256"],
                "evaluator context source.content_sha256",
            ),
            provenance=EvaluatorContextProvenance.from_dict(
                payload["provenance"]
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "content": self.content,
            "content_sha256": self.content_sha256,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class TruthEvaluatorContext(_JsonModel):
    truth_id: str
    allowed_tasks: Tuple[EvaluatorContextTask, ...]
    sources: Tuple[EvaluatorContextSource, ...]

    def __post_init__(self) -> None:
        _identifier(self.truth_id, "truth evaluator context.truth_id")
        raw_tasks = _sequence(
            self.allowed_tasks,
            "truth evaluator context.allowed_tasks",
            MAX_TEXT_LIST_ITEMS,
        )
        if not raw_tasks:
            raise _error("truth evaluator context.allowed_tasks must be non-empty")
        tasks = tuple(
            _require_enum(
                EvaluatorContextTask,
                item,
                "truth evaluator context.allowed_tasks[%d]" % index,
            )
            for index, item in enumerate(raw_tasks)
        )
        if len(tasks) != len(set(tasks)):
            raise _error("truth evaluator context.allowed_tasks contains duplicates")
        sources = _model_tuple(
            self.sources,
            EvaluatorContextSource,
            "truth evaluator context.sources",
            MAX_EVIDENCE_ANCHORS,
        )
        object.__setattr__(self, "allowed_tasks", tuple(sorted(tasks, key=lambda item: item.value)))
        object.__setattr__(self, "sources", _canonical_model_sort(sources))

    @classmethod
    def from_dict(cls, value: Any) -> "TruthEvaluatorContext":
        payload = _object(value, "truth evaluator context")
        _exact_fields(
            payload,
            ("truth_id", "allowed_tasks", "sources"),
            "truth evaluator context",
        )
        raw_tasks = _array(
            payload["allowed_tasks"],
            "truth evaluator context.allowed_tasks",
            MAX_TEXT_LIST_ITEMS,
        )
        sources = _array(
            payload["sources"],
            "truth evaluator context.sources",
            MAX_EVIDENCE_ANCHORS,
        )
        return cls(
            truth_id=_identifier(
                payload["truth_id"], "truth evaluator context.truth_id"
            ),
            allowed_tasks=tuple(
                _enum_value(
                    EvaluatorContextTask,
                    item,
                    "truth evaluator context.allowed_tasks[%d]" % index,
                )
                for index, item in enumerate(raw_tasks)
            ),
            sources=tuple(EvaluatorContextSource.from_dict(item) for item in sources),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "truth_id": self.truth_id,
            "allowed_tasks": [item.value for item in self.allowed_tasks],
            "sources": [item.to_dict() for item in self.sources],
        }


@dataclass(frozen=True)
class ReviewEvaluatorContext(_JsonModel):
    truth_contexts: Tuple[TruthEvaluatorContext, ...]

    def __post_init__(self) -> None:
        contexts = _model_tuple(
            self.truth_contexts,
            TruthEvaluatorContext,
            "review evaluator context.truth_contexts",
            MAX_TRUTH_FINDINGS,
        )
        _unique_by(
            contexts,
            "truth_id",
            "review evaluator context.truth_contexts",
        )
        object.__setattr__(
            self, "truth_contexts", _sorted_by(contexts, "truth_id")
        )

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewEvaluatorContext":
        payload = _object(value, "review evaluator context")
        _exact_fields(payload, ("truth_contexts",), "review evaluator context")
        contexts = _array(
            payload["truth_contexts"],
            "review evaluator context.truth_contexts",
            MAX_TRUTH_FINDINGS,
        )
        return cls(
            truth_contexts=tuple(
                TruthEvaluatorContext.from_dict(item) for item in contexts
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "truth_contexts": [item.to_dict() for item in self.truth_contexts]
        }


@dataclass(frozen=True)
class EvalCase(_JsonModel):
    SCHEMA_VERSION: ClassVar[str] = EVAL_CASE_SCHEMA_VERSION

    schema_version: str
    task_id: str
    case_version: int
    source: CaseSource
    input: EvalCaseInput
    clarification_script: ClarificationScript
    intent_truth: IntentTruth
    review_truth: ReviewTruth
    review_evaluator_context: ReviewEvaluatorContext

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise UnsupportedProtocolVersionError(
                expected=self.SCHEMA_VERSION, actual=self.schema_version
            )
        _identifier(self.task_id, "eval_case.task_id")
        _integer(
            self.case_version,
            "eval_case.case_version",
            minimum=1,
            maximum=MAX_COUNTER,
        )
        if not isinstance(self.source, CaseSource):
            raise _error("eval_case.source must be a CaseSource")
        case_input = self.input
        if not isinstance(case_input, EvalCaseInput):
            raise _error("eval_case.input must be an EvalCaseInput")
        # A Case may carry more truth than an Agent-facing input, but its input
        # projection must still be a valid 2 MiB EvalInput in its own right.
        case_input.to_eval_input(self.task_id)
        if not isinstance(self.clarification_script, ClarificationScript):
            raise _error("eval_case.clarification_script must be ClarificationScript")
        if not isinstance(self.intent_truth, IntentTruth):
            raise _error("eval_case.intent_truth must be IntentTruth")
        if not isinstance(self.review_truth, ReviewTruth):
            raise _error("eval_case.review_truth must be ReviewTruth")
        if not isinstance(
            self.review_evaluator_context, ReviewEvaluatorContext
        ):
            raise _error(
                "eval_case.review_evaluator_context must be ReviewEvaluatorContext"
            )

        truth_ids = [item.truth_id for item in self.intent_truth.expected_claims]
        truth_ids.extend(item.truth_id for item in self.intent_truth.forbidden_claims)
        truth_ids.extend(item.truth_id for item in self.review_truth.expected_findings)
        truth_ids.extend(
            item.truth_id for item in self.review_truth.known_invalid_findings
        )
        if len(truth_ids) != len(set(truth_ids)):
            raise _error("EvalCase contains duplicate truth_id values")
        review_truth_ids = {
            item.truth_id for item in self.review_truth.expected_findings
        }
        review_truth_ids.update(
            item.truth_id for item in self.review_truth.known_invalid_findings
        )
        for context in self.review_evaluator_context.truth_contexts:
            if context.truth_id not in review_truth_ids:
                raise _error(
                    "review evaluator context references a truth outside this EvalCase"
                )
        _check_model_size(self, MAX_EVAL_CASE_BYTES, "EvalCase")

    @classmethod
    def from_dict(cls, value: Any) -> "EvalCase":
        payload = _object(value, "EvalCase")
        _require_root_schema_version(payload, cls.SCHEMA_VERSION, "EvalCase")
        _exact_fields(
            payload,
            (
                "schema_version",
                "task_id",
                "case_version",
                "source",
                "input",
                "clarification_script",
                "intent_truth",
                "review_truth",
                "review_evaluator_context",
            ),
            "EvalCase",
        )
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            task_id=_identifier(payload["task_id"], "eval_case.task_id"),
            case_version=_integer(
                payload["case_version"],
                "eval_case.case_version",
                minimum=1,
                maximum=MAX_COUNTER,
            ),
            source=CaseSource.from_dict(payload["source"]),
            input=EvalCaseInput.from_dict(payload["input"]),
            clarification_script=ClarificationScript.from_dict(
                payload["clarification_script"]
            ),
            intent_truth=IntentTruth.from_dict(payload["intent_truth"]),
            review_truth=ReviewTruth.from_dict(payload["review_truth"]),
            review_evaluator_context=ReviewEvaluatorContext.from_dict(
                payload["review_evaluator_context"]
            ),
        )

    @classmethod
    def from_json(cls, data: Any) -> "EvalCase":
        return cls.from_dict(
            _strict_json_loads(data, MAX_EVAL_CASE_BYTES, "EvalCase JSON")
        )

    def eval_input(self) -> EvalInput:
        return self.input.to_eval_input(self.task_id)

    def agent_input(self) -> EvalInput:
        return self.eval_input()

    def validate_submission(self, submission: EvalSubmission) -> None:
        validate_submission_for_case(submission, self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "case_version": self.case_version,
            "source": self.source.to_dict(),
            "input": self.input.to_dict(),
            "clarification_script": self.clarification_script.to_dict(),
            "intent_truth": self.intent_truth.to_dict(),
            "review_truth": self.review_truth.to_dict(),
            "review_evaluator_context": self.review_evaluator_context.to_dict(),
        }


def validate_submission_for_case(submission: EvalSubmission, case: EvalCase) -> None:
    """Validate the cross-protocol references available only with the private Case."""

    if not isinstance(submission, EvalSubmission):
        raise _error("submission must be an EvalSubmission")
    if not isinstance(case, EvalCase):
        raise _error("case must be an EvalCase")
    if submission.task_id != case.task_id:
        raise _error("submission task_id does not match EvalCase task_id")
    if submission.intent is None:
        return
    answers = {item.answer_id: item for item in case.clarification_script.answers}
    consumed = set()
    for exchange in submission.intent.clarification_questions:
        if exchange.turn_index > case.clarification_script.max_rounds:
            if exchange.action is not None:
                raise _error(
                    "clarification exchange beyond Case max_rounds must remain unresolved"
                )
            continue
        if exchange.matched_answer_id is None:
            continue
        answer = answers.get(exchange.matched_answer_id)
        if answer is None:
            raise _error("clarification exchange references an answer outside the Case script")
        if answer.answer_id in consumed:
            raise _error("clarification answer was consumed more than once")
        consumed.add(answer.answer_id)
        if (
            exchange.dimension is not answer.dimension
            or exchange.action is not answer.action
        ):
            raise _error("clarification exchange does not match its consumed Case answer")
        if exchange.response != answer.response:
            raise _error("clarification exchange response differs from the Case answer")
        if answer.action is ClarificationAction.CORRECT:
            if exchange.resolved_values != answer.corrected_values:
                raise _error("correct clarification exchange differs from the Case answer")


def load_eval_input(data: Any) -> EvalInput:
    return EvalInput.from_json(data)


def load_eval_submission(data: Any) -> EvalSubmission:
    return EvalSubmission.from_json(data)


def load_eval_case(data: Any) -> EvalCase:
    return EvalCase.from_json(data)


__all__ = [
    "EVAL_INPUT_SCHEMA_VERSION",
    "EVAL_SUBMISSION_SCHEMA_VERSION",
    "EVAL_CASE_SCHEMA_VERSION",
    "MAX_EVAL_INPUT_BYTES",
    "MAX_EVAL_SUBMISSION_BYTES",
    "MAX_EVAL_CASE_BYTES",
    "MAX_IDENTIFIER_CHARS",
    "MAX_REPOSITORY_PATH_CHARS",
    "MAX_URL_CHARS",
    "MAX_TITLE_CHARS",
    "MAX_DESCRIPTION_CHARS",
    "MAX_CLAIM_CHARS",
    "MAX_RATIONALE_CHARS",
    "MAX_QUESTION_CHARS",
    "MAX_ANSWER_CHARS",
    "MAX_UNCERTAINTY_CHARS",
    "MAX_EVIDENCE_EXCERPT_BYTES",
    "MAX_EVALUATOR_CONTEXT_CONTENT_BYTES",
    "MAX_REQUIREMENTS",
    "MAX_PROJECT_RULES",
    "MAX_EXISTING_CI_EVIDENCE",
    "MAX_TEXT_LIST_ITEMS",
    "MAX_CLARIFICATION_ANSWERS",
    "MAX_CLARIFICATION_QUESTIONS",
    "MAX_INTENT_CLAIMS",
    "MAX_FINDINGS",
    "MAX_EVIDENCE_ITEMS",
    "MAX_EVIDENCE_REFS",
    "MAX_TRUTH_FINDINGS",
    "MAX_TRUTH_LOCATIONS",
    "MAX_EVIDENCE_ANCHORS",
    "MAX_COMMAND_ARGUMENTS",
    "MAX_LINE_NUMBER",
    "MAX_COUNTER",
    "MAX_JSON_DEPTH",
    "SchemaError",
    "UnsupportedProtocolVersionError",
    "RepositorySource",
    "TrialStatus",
    "SubmissionStatus",
    "JudgeStatus",
    "FailureCode",
    "ClarificationAction",
    "IntentDimension",
    "IntentResult",
    "IntentClaimSource",
    "IntentClaimJudgement",
    "FindingSeverity",
    "DiffSide",
    "ReviewTargetKind",
    "EvidenceKind",
    "EvidenceStream",
    "TraceType",
    "CaseOrigin",
    "IntentAuthority",
    "ClarificationPolicy",
    "TruthCompleteness",
    "NovelFindingPolicy",
    "RequiredContextLevel",
    "MetricAuthoritySource",
    "EvaluatorContextTask",
    "EvaluatorContextSourceKind",
    "IssueJudgement",
    "EvidenceIntegrity",
    "EvidenceSupport",
    "Repository",
    "ExistingCIEvidence",
    "ReviewRequest",
    "RepositoryReviewTarget",
    "FrozenContextReviewTarget",
    "ReviewTargetV2",
    "EvalInput",
    "EvalCaseInput",
    "SubmissionIntentClaim",
    "SubmissionClarificationExchange",
    "SubmissionIntent",
    "SubmissionFinding",
    "SubmissionReview",
    "RepositoryFileEvidenceSource",
    "RepositoryDiffEvidenceSource",
    "FrozenContextEvidenceSource",
    "CommandOutputEvidenceSource",
    "ExternalRecordEvidenceSource",
    "EvidenceSourceV2",
    "SubmissionEvidence",
    "SubmissionUsage",
    "TraceRef",
    "SubmissionFailure",
    "EvalSubmission",
    "submission_status_for_failure",
    "CaseSource",
    "ClarificationAnswer",
    "ClarificationScript",
    "ExpectedIntentClaim",
    "ForbiddenIntentClaim",
    "IntentTruth",
    "TruthLocation",
    "EvidenceAnchor",
    "MetricAuthority",
    "ExpectedFinding",
    "KnownInvalidFinding",
    "ReviewTruth",
    "EvaluatorContextProvenance",
    "EvaluatorContextSource",
    "TruthEvaluatorContext",
    "ReviewEvaluatorContext",
    "EvalCase",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "stable_id",
    "validate_stable_id",
    "validate_submission_for_case",
    "load_eval_input",
    "load_eval_submission",
    "load_eval_case",
]

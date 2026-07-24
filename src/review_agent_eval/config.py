"""Immutable run configuration for the canonical code-review eval harness.

The configuration is an authority boundary.  It intentionally contains only
bounded JSON values needed to identify an evaluation and reproduce it; process
environments, credentials, provider responses, and hidden reasoning are not
representable here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from .cases import (
    MAX_SUITE_CASES,
    RunCaseSnapshot,
    SuiteCase,
    WireContractV2,
    validate_run_case_snapshot_id,
)
from .models import (
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    MAX_IDENTIFIER_CHARS,
    EvidenceKind,
    ReviewTargetKind,
    SchemaError,
    UnsupportedProtocolVersionError,
    _JsonModel,
    _array,
    _check_model_size,
    _digest,
    _enum_value,
    _exact_fields,
    _identifier,
    _integer,
    _number,
    _object,
    _strict_json_loads,
    _string,
    canonical_json_bytes,
    canonical_sha256,
    stable_id,
)


EVAL_RUN_CONFIG_SCHEMA_VERSION = "eval_run_config_v2"
EVALUATOR_EXECUTION_CONFIG_SCHEMA_VERSION = "eval_evaluator_execution_config_v2"
ADAPTER_CAPABILITIES_SCHEMA_VERSION = "eval_adapter_capabilities_v2"
JUDGE_PROFILE_SCHEMA_VERSION = "eval_judge_profile_v1"
JUDGE_EXECUTION_BUDGETS_SCHEMA_VERSION = "eval_judge_execution_budgets_v1"
DEFAULT_JUDGE_CACHE_POLICY_VERSION = "semantic-judge-cache-v1"
DEFAULT_REVIEW_EVALUATOR_CONTEXT_POLICY_VERSION = (
    "truth-scoped-evaluator-context-v2"
)
DEFAULT_METRIC_AUTHORITY_POLICY_VERSION = "metric-authority-v2"

# A Suite Manifest may be 16 MiB and SuiteRunConfig intentionally preserves
# the selected canonical SuiteCase bindings.  The Run Config therefore needs
# its own control-plane budget instead of inheriting Agent output limits.
MAX_EVAL_RUN_CONFIG_BYTES = 32 * 1024 * 1024
MAX_PARAMETER_BYTES = 256 * 1024
MAX_PARAMETER_NODES = 8_192
MAX_TRIAL_COUNT = 10_000
MAX_PLANNED_TRIALS = 100_000
MAX_VERSION_CHARS = 256
MAX_AGENT_NAME_CHARS = 512
MAX_ARTIFACT_BUDGET_BYTES = 1 << 40
MAX_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
MAX_JUDGE_ATTEMPTS_PER_REQUEST = 16
MAX_JUDGE_REQUESTS = 65_536
MAX_JUDGE_CONTEXT_BLOCKS_PER_REQUEST = 4_096
MAX_JUDGE_REASON_REFS = 32
MAX_JUDGE_TOKEN_BUDGET = 1 << 40


_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RUN_ID_RE = re.compile(r"^run-[0-9a-f]{64}$")
_TRIAL_ID_RE = re.compile(r"^trial-[0-9a-f]{64}$")
_EVALUATION_ID_RE = re.compile(r"^evaluation-[0-9a-f]{64}$")
_CASE_PATH_ID_RE = re.compile(r"^case-[0-9a-f]{64}$")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {"COM%d" % index for index in range(1, 10)}
    | {"LPT%d" % index for index in range(1, 10)}
)
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_COMMON_ENV_KEY_NAMES = frozenset(
    {
        "appdata",
        "comspec",
        "home",
        "localappdata",
        "path",
        "pathext",
        "programfiles",
        "pwd",
        "shell",
        "systemroot",
        "temp",
        "tmp",
        "user",
        "username",
        "userprofile",
        "windir",
    }
)

_FORBIDDEN_KEY_NAMES = frozenset(
    {
        "apikey",
        "apitoken",
        "token",
        "accesstoken",
        "refreshtoken",
        "authtoken",
        "bearertoken",
        "authorization",
        "authentication",
        "authorizationheader",
        "password",
        "passwd",
        "secret",
        "clientsecret",
        "credential",
        "credentials",
        "cookie",
        "setcookie",
        "env",
        "environ",
        "environment",
        "environmentvariables",
        "processenvironment",
        "rawreasoning",
        "hiddenreasoning",
        "chainofthought",
        "internalreasoning",
        "reasoningcontent",
        "reasoningtrace",
        "thinkingcontent",
        "intermediatesteps",
    }
)
_SECRET_VALUE_RE = re.compile(
    r"(?ix)(?:"
    r"\bAKIA[0-9A-Z]{16}\b|"
    r"\b(?:sk|rk|ghp|gho|ghu|ghs|ghr|xoxb|xoxp|xoxa|xoxr|ya29)[-_][A-Za-z0-9._-]{4,}\b|"
    r"-----BEGIN[ ](?:RSA[ ]|EC[ ]|OPENSSH[ ])?PRIVATE[ ]KEY-----|"
    r"\bBearer[ ]+[A-Za-z0-9._~+/=-]{8,}|"
    r"\bAuthorization\s*:\s*(?:Basic|Bearer)\s+[A-Za-z0-9._~+/=-]{4,}|"
    r"\b(?:AWS[_-]?SECRET[_-]?ACCESS[_-]?KEY|AWS[_-]?SESSION[_-]?TOKEN|"
    r"api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|"
    r"client[_-]?secret)\s*[:=]\s*[^\s,;]{4,}"
    r")"
)
_URL_AUTHORITY_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://([^\s/?#]+)")
_ENV_ASSIGNMENT_RE = re.compile(
    r"(?:^|[\s,{;])([A-Z_][A-Z0-9_]{1,63})\s*=",
    re.IGNORECASE | re.MULTILINE,
)
_ENV_MAPPING_RE = re.compile(
    r"['\"]([A-Z_][A-Z0-9_]{1,63})['\"]\s*:", re.IGNORECASE
)
_RAW_REASONING_TEXT_RE = re.compile(
    r"(?i)(?:\braw[ _-]?reasoning\b|\bhidden[ _-]?reasoning\b|"
    r"\bchain[ -]?of[ -]?thought\b|\breasoning[ _-]?content\b|"
    r"\b(?:private|hidden|raw)[ _-]?intermediate[ _-]?(?:steps|reasoning|thoughts)\b|"
    r"</?think(?:ing)?>)"
)


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _key_is_forbidden(key: str) -> bool:
    normalized = _normalized_key(key)
    if normalized in _FORBIDDEN_KEY_NAMES:
        return True
    return (
        "apikey" in normalized
        or "secret" in normalized
        or normalized.endswith("password")
        or normalized.endswith("passwd")
        or normalized.endswith("clientsecret")
        or normalized.endswith("credential")
        or normalized.endswith("credentials")
        or normalized in {"auth", "authentication", "token"}
        or (
            "reasoning" in normalized
            and any(
                marker in normalized
                for marker in (
                    "content",
                    "trace",
                    "raw",
                    "hidden",
                    "internal",
                    "intermediate",
                )
            )
        )
        or normalized in {"rawcot", "cot"}
    )


def _has_url_userinfo(value: str) -> bool:
    candidate = value.strip()
    if any("@" in match.group(1) for match in _URL_AUTHORITY_RE.finditer(candidate)):
        return True
    if "://" not in candidate and not candidate.startswith("//"):
        return False
    try:
        parsed = urlsplit(candidate)
        username = parsed.username
        password = parsed.password
    except ValueError:
        # A credential-looking authority that cannot be parsed is not safe to
        # persist merely because it is malformed.
        authority = candidate.split("//", 1)[-1].split("/", 1)[0]
        return "@" in authority
    return username is not None or password is not None or "@" in parsed.netloc


def validate_safe_text(value: Any, context: str = "value") -> str:
    """Reject credential material and authenticated URLs without echoing it."""

    text = _string(value, context, MAX_EVAL_RUN_CONFIG_BYTES, allow_empty=True)
    if _has_url_userinfo(text):
        raise SchemaError("%s may not contain URL userinfo or credentials" % context)
    if _SECRET_VALUE_RE.search(text) is not None:
        raise SchemaError("%s contains forbidden sensitive material" % context)
    if _RAW_REASONING_TEXT_RE.search(text) is not None:
        raise SchemaError("%s contains forbidden raw reasoning" % context)
    env_names = set(_ENV_ASSIGNMENT_RE.findall(text))
    env_names.update(_ENV_MAPPING_RE.findall(text))
    if len(env_names) >= 2:
        raise SchemaError("%s contains a forbidden full environment dump" % context)
    return text


_REVIEW_EVALUATION_SAFE_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "evaluator_revision",
        "evaluator_execution_digest",
        "submission_digest",
        "submission_review_digest",
        "submission_evidence_digest",
        "eval_input_digest",
        "review_truth_digest",
        "deterministic_context_digest",
        "review_policy_version",
        "assignment_policy_version",
        "location_policy_version",
        "evidence_integrity_policy_version",
        "truth_completeness",
        "novel_finding_policy",
        "status",
        "phase",
        "generated_findings",
        "expected_truth_findings",
        "known_invalid_truth_findings",
        "location_candidates",
        "known_invalid_candidates",
        "expected_candidates",
        "assignments",
        "finding_outcomes",
        "unmatched_expected_truth_ids",
        "judge_requests",
        "judge_decisions",
        "judge_failures",
        "judge_ungraded",
        "evidence_integrity_results",
        "coverage",
        "metrics",
        "reason_codes",
        "limit_failure",
    }
)
_EVALUATOR_CONTEXT_POLICIES = {
    "review_matches": (
        "eval_review_evaluation_v1",
        _REVIEW_EVALUATION_SAFE_ROOT_FIELDS,
        (
            (
                "judge_requests",
                int,
                "request",
                "contexts",
                int,
                "content",
            ),
        ),
    ),
    "judge_input": (
        "eval_judge_input_artifact_v1",
        frozenset({"schema_version", "evaluator_execution_digest", "requests"}),
        (("requests", int, "contexts", int, "content"),),
    ),
    "judge_output": (
        "eval_judge_output_artifact_v1",
        frozenset(
            {
                "schema_version",
                "evaluator_execution_digest",
                "input_artifact_digest",
                "intent_evaluation_digest",
                "results",
            }
        ),
        (
            ("results", int, "request", "contexts", int, "content"),
            (
                "results",
                int,
                "model_turn",
                "messages",
                int,
                "content",
                "context_blocks",
                int,
                "content",
            ),
        ),
    ),
}
_JUDGE_CONTEXT_BLOCK_FIELDS = frozenset(
    {"ref_id", "kind", "trust", "content", "metadata", "content_digest"}
)
_JUDGE_MODEL_CONTEXT_BLOCK_FIELDS = frozenset(
    {
        "ref_id",
        "kind",
        "data_boundary",
        "content",
        "metadata",
        "content_digest",
    }
)


def validate_safe_json(
    value: Any,
    context: str = "value",
    *,
    evaluator_context_policy: Optional[str] = None,
) -> None:
    """Validate a JSON tree before it crosses the persistent artifact boundary."""

    if (
        evaluator_context_policy is not None
        and evaluator_context_policy not in _EVALUATOR_CONTEXT_POLICIES
    ):
        raise SchemaError("%s has an unknown evaluator context policy" % context)

    policy_root = value.to_dict() if isinstance(value, _JsonModel) else value
    if evaluator_context_policy is not None:
        schema_version, root_fields, _ = _EVALUATOR_CONTEXT_POLICIES[
            evaluator_context_policy
        ]
        if not (
            type(policy_root) is dict
            or isinstance(policy_root, _MAPPING_PROXY_TYPE)
        ) or (
            frozenset(policy_root) != root_fields
            or policy_root.get("schema_version") != schema_version
        ):
            raise SchemaError(
                "%s evaluator context policy cannot bypass the full environment "
                "dump heuristic for this root artifact" % context
            )

    def path_matches(
        path: Tuple[Any, ...],
        expected: Tuple[Any, ...],
    ) -> bool:
        return len(path) == len(expected) and all(
            type(actual) is int if wanted is int else actual == wanted
            for actual, wanted in zip(path, expected)
        )

    def value_at_path(path: Tuple[Any, ...]) -> Any:
        item = policy_root
        for part in path:
            if type(part) is int:
                if type(item) not in (list, tuple) or not 0 <= part < len(item):
                    return None
                item = item[part]
            else:
                if not (
                    type(item) is dict
                    or isinstance(item, _MAPPING_PROXY_TYPE)
                ) or part not in item:
                    return None
                item = item[part]
        return item

    def allows_evaluator_context_content(
        path: Tuple[Any, ...],
        content: str,
    ) -> bool:
        if evaluator_context_policy is None:
            return False
        schema_version, root_fields, allowed_paths = _EVALUATOR_CONTEXT_POLICIES[
            evaluator_context_policy
        ]
        if not (
            type(policy_root) is dict
            or isinstance(policy_root, _MAPPING_PROXY_TYPE)
        ):
            return False
        if (
            frozenset(policy_root) != root_fields
            or policy_root.get("schema_version") != schema_version
            or not any(path_matches(path, expected) for expected in allowed_paths)
        ):
            return False
        block = value_at_path(path[:-1])
        if not (type(block) is dict or isinstance(block, _MAPPING_PROXY_TYPE)):
            return False
        expected_block_fields = (
            _JUDGE_MODEL_CONTEXT_BLOCK_FIELDS
            if "context_blocks" in path
            else _JUDGE_CONTEXT_BLOCK_FIELDS
        )
        return (
            frozenset(block) == expected_block_fields
            and block.get("content") == content
            and block.get("content_digest") == canonical_sha256(content)
        )

    def walk(
        item: Any,
        item_context: str,
        depth: int,
        path: Tuple[Any, ...],
    ) -> None:
        if depth > 128:
            raise SchemaError("%s exceeds the maximum nesting depth" % context)
        if item is None or type(item) in (bool, int, float):
            return
        if type(item) is str:
            evaluator_context_path = (
                evaluator_context_policy is not None
                and any(
                    path_matches(path, expected)
                    for expected in _EVALUATOR_CONTEXT_POLICIES[
                        evaluator_context_policy
                    ][2]
                )
            )
            evaluator_context_content_allowed = (
                allows_evaluator_context_content(path, item)
                if evaluator_context_path
                else False
            )
            try:
                validate_safe_text(item, item_context)
            except SchemaError as exc:
                if not (
                    evaluator_context_content_allowed
                    and str(exc)
                    == "%s contains a forbidden full environment dump"
                    % item_context
                ):
                    raise
            if evaluator_context_path and not evaluator_context_content_allowed:
                raise SchemaError(
                    "%s has an invalid evaluator context binding" % item_context
                )
            return
        if type(item) in (list, tuple):
            for index, child in enumerate(item):
                walk(
                    child,
                    "%s[%d]" % (item_context, index),
                    depth + 1,
                    (*path, index),
                )
            return
        if type(item) is dict or isinstance(item, _MAPPING_PROXY_TYPE):
            environment_keys = {
                _normalized_key(key)
                for key in item
                if type(key) is str
                and _normalized_key(key) in _COMMON_ENV_KEY_NAMES
            }
            if len(environment_keys) >= 3:
                raise SchemaError(
                    "%s contains a forbidden environment mapping" % item_context
                )
            for key, child in item.items():
                if type(key) is not str:
                    raise SchemaError("%s contains a non-string object key" % item_context)
                if _key_is_forbidden(key):
                    raise SchemaError(
                        "%s contains a forbidden secret, environment, or reasoning field"
                        % item_context
                    )
                validate_safe_text(key, "%s key" % item_context)
                walk(
                    child,
                    "%s.%s" % (item_context, key),
                    depth + 1,
                    (*path, key),
                )
            return
        if isinstance(item, _JsonModel):
            walk(item.to_dict(), item_context, depth + 1, path)
            return
        raise SchemaError("%s contains a non-JSON value" % item_context)

    walk(value, context, 0, ())


def validate_path_segment(value: Any, context: str = "identifier") -> str:
    """Validate an opaque ID that will also be used as one path component."""

    result = _identifier(value, context)
    if _PATH_SEGMENT_RE.fullmatch(result) is None:
        raise SchemaError("%s is not a traversal-safe path identifier" % context)
    if result in {".", ".."} or result.endswith((".", " ")):
        raise SchemaError("%s is not a traversal-safe path identifier" % context)
    if result.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise SchemaError("%s uses a reserved Windows path name" % context)
    return result


def _validate_derived_id(value: Any, pattern: re.Pattern[str], context: str) -> str:
    result = _identifier(value, context)
    if pattern.fullmatch(result) is None:
        raise SchemaError("%s is not a canonical derived ID" % context)
    validate_path_segment(result, context)
    return result


def validate_run_id(value: Any) -> str:
    return _validate_derived_id(value, _RUN_ID_RE, "run_id")


def validate_trial_id_shape(value: Any) -> str:
    return _validate_derived_id(value, _TRIAL_ID_RE, "trial_id")


def validate_evaluation_id_shape(value: Any) -> str:
    return _validate_derived_id(value, _EVALUATION_ID_RE, "evaluation_id")


def validate_case_path_id(value: Any) -> str:
    return _validate_derived_id(value, _CASE_PATH_ID_RE, "case_path_id")


def _thaw_json(value: Any) -> Any:
    if type(value) is tuple:
        return [_thaw_json(item) for item in value]
    if type(value) is dict or isinstance(value, _MAPPING_PROXY_TYPE):
        return {key: _thaw_json(item) for key, item in value.items()}
    return value


def _freeze_json(value: Any) -> Any:
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    if type(value) is dict:
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    return value


def _json_object(value: Any, context: str) -> Mapping[str, Any]:
    if type(value) is dict:
        candidate = value
    elif isinstance(value, _MAPPING_PROXY_TYPE):
        candidate = _thaw_json(value)
    else:
        raise SchemaError("%s must be a JSON object" % context)

    encoded = canonical_json_bytes(candidate)
    if len(encoded) > MAX_PARAMETER_BYTES:
        raise SchemaError(
            "%s exceeds the canonical byte limit of %d"
            % (context, MAX_PARAMETER_BYTES)
        )
    parsed = _strict_json_loads(encoded, MAX_PARAMETER_BYTES, context)
    nodes = 0

    def count(item: Any) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_PARAMETER_NODES:
            raise SchemaError(
                "%s exceeds the JSON node limit of %d"
                % (context, MAX_PARAMETER_NODES)
            )
        if type(item) is dict:
            for child in item.values():
                count(child)
        elif type(item) is list:
            for child in item:
                count(child)

    count(parsed)
    validate_safe_json(parsed, context)
    return _freeze_json(parsed)


def _positive_number(value: Any, context: str) -> Any:
    result = _number(value, context)
    if result <= 0 or result > MAX_TIMEOUT_SECONDS:
        raise SchemaError(
            "%s must be greater than zero and at most %d"
            % (context, MAX_TIMEOUT_SECONDS)
        )
    return result


def _version(value: Any, context: str) -> str:
    result = _string(value, context, MAX_VERSION_CHARS)
    validate_safe_text(result, context)
    return result


@dataclass(frozen=True)
class ClarificationMatcherSnapshot(_JsonModel):
    """Reproducible matcher implementation/config bound into Agent-side Run identity."""

    matcher_id: str
    matcher_version: str
    implementation_digest: str
    model_artifact_digest: Optional[str]
    rubric_digest: str
    normalization_version: str
    threshold: Any
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identifier(self.matcher_id, "clarification_matcher.matcher_id")
        _version(self.matcher_version, "clarification_matcher.matcher_version")
        _digest(
            self.implementation_digest,
            "clarification_matcher.implementation_digest",
        )
        if self.model_artifact_digest is not None:
            _digest(
                self.model_artifact_digest,
                "clarification_matcher.model_artifact_digest",
            )
        _digest(self.rubric_digest, "clarification_matcher.rubric_digest")
        _version(
            self.normalization_version,
            "clarification_matcher.normalization_version",
        )
        if self.threshold is not None:
            threshold = _number(
                self.threshold,
                "clarification_matcher.threshold",
            )
            if threshold > 1:
                raise SchemaError(
                    "clarification_matcher.threshold must be at most one"
                )
        object.__setattr__(
            self,
            "parameters",
            _json_object(
                self.parameters,
                "clarification_matcher.parameters",
            ),
        )

    @classmethod
    def from_dict(cls, value: Any) -> "ClarificationMatcherSnapshot":
        payload = _object(value, "clarification_matcher")
        _exact_fields(
            payload,
            (
                "matcher_id",
                "matcher_version",
                "implementation_digest",
                "model_artifact_digest",
                "rubric_digest",
                "normalization_version",
                "threshold",
                "parameters",
            ),
            "clarification_matcher",
        )
        model_artifact_digest = payload["model_artifact_digest"]
        if model_artifact_digest is not None:
            model_artifact_digest = _digest(
                model_artifact_digest,
                "clarification_matcher.model_artifact_digest",
            )
        threshold = payload["threshold"]
        if threshold is not None:
            threshold = _number(threshold, "clarification_matcher.threshold")
        return cls(
            matcher_id=_identifier(
                payload["matcher_id"],
                "clarification_matcher.matcher_id",
            ),
            matcher_version=_version(
                payload["matcher_version"],
                "clarification_matcher.matcher_version",
            ),
            implementation_digest=_digest(
                payload["implementation_digest"],
                "clarification_matcher.implementation_digest",
            ),
            model_artifact_digest=model_artifact_digest,
            rubric_digest=_digest(
                payload["rubric_digest"],
                "clarification_matcher.rubric_digest",
            ),
            normalization_version=_version(
                payload["normalization_version"],
                "clarification_matcher.normalization_version",
            ),
            threshold=threshold,
            parameters=_object(
                payload["parameters"],
                "clarification_matcher.parameters",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matcher_id": self.matcher_id,
            "matcher_version": self.matcher_version,
            "implementation_digest": self.implementation_digest,
            "model_artifact_digest": self.model_artifact_digest,
            "rubric_digest": self.rubric_digest,
            "normalization_version": self.normalization_version,
            "threshold": self.threshold,
            "parameters": _thaw_json(self.parameters),
        }


@dataclass(frozen=True)
class AgentConfigSnapshot(_JsonModel):
    agent_id: str
    agent_name: str
    agent_version: str
    commit: str
    model: str
    provider: str
    parameters: Mapping[str, Any]
    prompt_config_digest: str

    def __post_init__(self) -> None:
        _identifier(self.agent_id, "agent.agent_id")
        name = _string(self.agent_name, "agent.agent_name", MAX_AGENT_NAME_CHARS)
        validate_safe_text(name, "agent.agent_name")
        _version(self.agent_version, "agent.agent_version")
        _version(self.commit, "agent.commit")
        _version(self.model, "agent.model")
        _version(self.provider, "agent.provider")
        object.__setattr__(
            self,
            "parameters",
            _json_object(self.parameters, "agent.parameters"),
        )
        _digest(self.prompt_config_digest, "agent.prompt_config_digest")

    @classmethod
    def from_dict(cls, value: Any) -> "AgentConfigSnapshot":
        payload = _object(value, "agent")
        _exact_fields(
            payload,
            (
                "agent_id",
                "agent_name",
                "agent_version",
                "commit",
                "model",
                "provider",
                "parameters",
                "prompt_config_digest",
            ),
            "agent",
        )
        return cls(
            agent_id=_identifier(payload["agent_id"], "agent.agent_id"),
            agent_name=_string(
                payload["agent_name"], "agent.agent_name", MAX_AGENT_NAME_CHARS
            ),
            agent_version=_version(payload["agent_version"], "agent.agent_version"),
            commit=_version(payload["commit"], "agent.commit"),
            model=_version(payload["model"], "agent.model"),
            provider=_version(payload["provider"], "agent.provider"),
            parameters=_object(payload["parameters"], "agent.parameters"),
            prompt_config_digest=_digest(
                payload["prompt_config_digest"], "agent.prompt_config_digest"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "commit": self.commit,
            "model": self.model,
            "provider": self.provider,
            "parameters": _thaw_json(self.parameters),
            "prompt_config_digest": self.prompt_config_digest,
        }


class JudgeKind(str, Enum):
    INTENT_EQUIVALENCE = "intent_equivalence"
    FINDING_EQUIVALENCE = "finding_equivalence"
    NOVEL_FACTUALITY = "novel_factuality"
    EVIDENCE_SUPPORT = "evidence_support"


def _judge_kind(value: Any, context: str) -> JudgeKind:
    if type(value) is not str:
        raise SchemaError("%s must be a string enum value" % context)
    try:
        return JudgeKind(value)
    except ValueError as exc:
        raise SchemaError("%s has an unknown Judge kind" % context) from exc


@dataclass(frozen=True)
class JudgeProfileSnapshot(_JsonModel):
    """Complete output-affecting identity for one semantic Judge kind."""

    SCHEMA_VERSION: ClassVar[str] = JUDGE_PROFILE_SCHEMA_VERSION

    schema_version: str
    kind: JudgeKind
    judge_id: str
    judge_version: str
    adapter_id: str
    adapter_version: str
    adapter_config_digest: str
    provider: str
    model: str
    model_artifact_digest: Optional[str]
    parameters: Mapping[str, Any]
    system_prompt_version: str
    system_prompt_digest: str
    rubric_id: str
    rubric_version: str
    rubric_digest: str
    response_schema_version: str
    response_schema_digest: str
    context_builder_version: str
    parser_version: str

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise SchemaError("JudgeProfileSnapshot has an unknown schema_version")
        if type(self.kind) is not JudgeKind:
            raise SchemaError("judge profile.kind must be a JudgeKind")
        _identifier(self.judge_id, "judge profile.judge_id")
        _version(self.judge_version, "judge profile.judge_version")
        _identifier(self.adapter_id, "judge profile.adapter_id")
        _version(self.adapter_version, "judge profile.adapter_version")
        _digest(
            self.adapter_config_digest,
            "judge profile.adapter_config_digest",
        )
        _version(self.provider, "judge profile.provider")
        _version(self.model, "judge profile.model")
        if self.model_artifact_digest is not None:
            _digest(
                self.model_artifact_digest,
                "judge profile.model_artifact_digest",
            )
        object.__setattr__(
            self,
            "parameters",
            _json_object(self.parameters, "judge profile.parameters"),
        )
        _version(
            self.system_prompt_version,
            "judge profile.system_prompt_version",
        )
        _digest(
            self.system_prompt_digest,
            "judge profile.system_prompt_digest",
        )
        _identifier(self.rubric_id, "judge profile.rubric_id")
        _version(self.rubric_version, "judge profile.rubric_version")
        _digest(self.rubric_digest, "judge profile.rubric_digest")
        _version(
            self.response_schema_version,
            "judge profile.response_schema_version",
        )
        _digest(
            self.response_schema_digest,
            "judge profile.response_schema_digest",
        )
        _version(
            self.context_builder_version,
            "judge profile.context_builder_version",
        )
        _version(self.parser_version, "judge profile.parser_version")
        validate_safe_json(self.to_dict(), "judge profile")
        _check_model_size(
            self,
            MAX_EVAL_RUN_CONFIG_BYTES,
            "JudgeProfileSnapshot",
        )

    @classmethod
    def from_dict(cls, value: Any) -> "JudgeProfileSnapshot":
        payload = _object(value, "JudgeProfileSnapshot")
        fields = (
            "schema_version",
            "kind",
            "judge_id",
            "judge_version",
            "adapter_id",
            "adapter_version",
            "adapter_config_digest",
            "provider",
            "model",
            "model_artifact_digest",
            "parameters",
            "system_prompt_version",
            "system_prompt_digest",
            "rubric_id",
            "rubric_version",
            "rubric_digest",
            "response_schema_version",
            "response_schema_digest",
            "context_builder_version",
            "parser_version",
        )
        _exact_fields(payload, fields, "JudgeProfileSnapshot")
        model_artifact_digest = payload["model_artifact_digest"]
        if model_artifact_digest is not None:
            model_artifact_digest = _digest(
                model_artifact_digest,
                "judge profile.model_artifact_digest",
            )
        return cls(
            schema_version=payload["schema_version"],
            kind=_judge_kind(payload["kind"], "judge profile.kind"),
            judge_id=_identifier(payload["judge_id"], "judge profile.judge_id"),
            judge_version=_version(
                payload["judge_version"], "judge profile.judge_version"
            ),
            adapter_id=_identifier(
                payload["adapter_id"], "judge profile.adapter_id"
            ),
            adapter_version=_version(
                payload["adapter_version"], "judge profile.adapter_version"
            ),
            adapter_config_digest=_digest(
                payload["adapter_config_digest"],
                "judge profile.adapter_config_digest",
            ),
            provider=_version(payload["provider"], "judge profile.provider"),
            model=_version(payload["model"], "judge profile.model"),
            model_artifact_digest=model_artifact_digest,
            parameters=_object(payload["parameters"], "judge profile.parameters"),
            system_prompt_version=_version(
                payload["system_prompt_version"],
                "judge profile.system_prompt_version",
            ),
            system_prompt_digest=_digest(
                payload["system_prompt_digest"],
                "judge profile.system_prompt_digest",
            ),
            rubric_id=_identifier(
                payload["rubric_id"], "judge profile.rubric_id"
            ),
            rubric_version=_version(
                payload["rubric_version"], "judge profile.rubric_version"
            ),
            rubric_digest=_digest(
                payload["rubric_digest"], "judge profile.rubric_digest"
            ),
            response_schema_version=_version(
                payload["response_schema_version"],
                "judge profile.response_schema_version",
            ),
            response_schema_digest=_digest(
                payload["response_schema_digest"],
                "judge profile.response_schema_digest",
            ),
            context_builder_version=_version(
                payload["context_builder_version"],
                "judge profile.context_builder_version",
            ),
            parser_version=_version(
                payload["parser_version"], "judge profile.parser_version"
            ),
        )

    @classmethod
    def from_json(cls, data: Any) -> "JudgeProfileSnapshot":
        return cls.from_dict(
            _strict_json_loads(
                data,
                MAX_EVAL_RUN_CONFIG_BYTES,
                "JudgeProfileSnapshot JSON",
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "judge_id": self.judge_id,
            "judge_version": self.judge_version,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "adapter_config_digest": self.adapter_config_digest,
            "provider": self.provider,
            "model": self.model,
            "model_artifact_digest": self.model_artifact_digest,
            "parameters": _thaw_json(self.parameters),
            "system_prompt_version": self.system_prompt_version,
            "system_prompt_digest": self.system_prompt_digest,
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
            "rubric_digest": self.rubric_digest,
            "response_schema_version": self.response_schema_version,
            "response_schema_digest": self.response_schema_digest,
            "context_builder_version": self.context_builder_version,
            "parser_version": self.parser_version,
        }


@dataclass(frozen=True)
class EvaluatorRunConfig(_JsonModel):
    evaluator_id: str
    evaluator_version: str
    grader_version: str
    judge_profiles: Tuple[JudgeProfileSnapshot, ...]

    def __post_init__(self) -> None:
        _identifier(self.evaluator_id, "evaluator.evaluator_id")
        _version(self.evaluator_version, "evaluator.evaluator_version")
        _version(self.grader_version, "evaluator.grader_version")
        if type(self.judge_profiles) not in (list, tuple):
            raise SchemaError("evaluator.judge_profiles must be a list or tuple")
        profiles = tuple(self.judge_profiles)
        if len(profiles) != len(JudgeKind):
            raise SchemaError(
                "evaluator.judge_profiles must contain each JudgeKind exactly once"
            )
        if any(type(item) is not JudgeProfileSnapshot for item in profiles):
            raise SchemaError(
                "evaluator.judge_profiles must contain JudgeProfileSnapshot values"
            )
        kinds = [item.kind for item in profiles]
        if set(kinds) != set(JudgeKind) or len(kinds) != len(set(kinds)):
            raise SchemaError(
                "evaluator.judge_profiles must contain each JudgeKind exactly once"
            )
        for identity_field in (
            "rubric_id",
            "rubric_version",
            "rubric_digest",
            "response_schema_version",
            "response_schema_digest",
        ):
            identities = [getattr(item, identity_field) for item in profiles]
            if len(identities) != len(set(identities)):
                raise SchemaError(
                    "evaluator.judge_profiles must use a distinct %s for each "
                    "JudgeKind" % identity_field
                )
        object.__setattr__(
            self,
            "judge_profiles",
            tuple(sorted(profiles, key=lambda item: item.kind.value)),
        )
        validate_safe_json(self.to_dict(), "evaluator")

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluatorRunConfig":
        payload = _object(value, "evaluator")
        _exact_fields(
            payload,
            (
                "evaluator_id",
                "evaluator_version",
                "grader_version",
                "judge_profiles",
            ),
            "evaluator",
        )
        profiles = _array(
            payload["judge_profiles"],
            "evaluator.judge_profiles",
            len(JudgeKind),
        )
        return cls(
            evaluator_id=_identifier(
                payload["evaluator_id"], "evaluator.evaluator_id"
            ),
            evaluator_version=_version(
                payload["evaluator_version"], "evaluator.evaluator_version"
            ),
            grader_version=_version(
                payload["grader_version"], "evaluator.grader_version"
            ),
            judge_profiles=tuple(
                JudgeProfileSnapshot.from_dict(item) for item in profiles
            ),
        )

    def profile(self, kind: JudgeKind) -> JudgeProfileSnapshot:
        if type(kind) is not JudgeKind:
            raise SchemaError("kind must be a JudgeKind")
        for profile in self.judge_profiles:
            if profile.kind is kind:
                return profile
        raise SchemaError("evaluator has no profile for Judge kind")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "grader_version": self.grader_version,
            "judge_profiles": [item.to_dict() for item in self.judge_profiles],
        }


def _required_protocol_version(actual: Any, expected: str) -> str:
    if type(actual) is not str:
        raise SchemaError("protocol version must be a string")
    if actual != expected:
        raise UnsupportedProtocolVersionError(expected=expected, actual=actual)
    return actual


@dataclass(frozen=True)
class AdapterCapabilitiesV2(_JsonModel):
    """Immutable Adapter capability snapshot bound into one Run identity."""

    SCHEMA_VERSION: ClassVar[str] = ADAPTER_CAPABILITIES_SCHEMA_VERSION

    schema_version: str
    adapter_id: str
    adapter_version: str
    input_schema_version: str
    submission_schema_version: str
    target_kinds: Tuple[ReviewTargetKind, ...]
    evidence_kinds: Tuple[EvidenceKind, ...]
    clarification_protocol: str
    trace_protocol: str
    subprocess_wire_version: Optional[str]
    isolation_profile: str

    def __post_init__(self) -> None:
        _required_protocol_version(self.schema_version, self.SCHEMA_VERSION)
        _identifier(self.adapter_id, "adapter capabilities.adapter_id")
        _version(self.adapter_version, "adapter capabilities.adapter_version")
        _required_protocol_version(
            self.input_schema_version, EVAL_INPUT_SCHEMA_VERSION
        )
        _required_protocol_version(
            self.submission_schema_version, EVAL_SUBMISSION_SCHEMA_VERSION
        )
        if type(self.target_kinds) not in (tuple, list) or not self.target_kinds:
            raise SchemaError(
                "adapter capabilities.target_kinds must be a non-empty list"
            )
        targets = tuple(self.target_kinds)
        if any(not isinstance(item, ReviewTargetKind) for item in targets):
            raise SchemaError(
                "adapter capabilities.target_kinds contains an invalid target kind"
            )
        if len(targets) != len(set(targets)):
            raise SchemaError("adapter capabilities.target_kinds contains duplicates")
        if type(self.evidence_kinds) not in (tuple, list):
            raise SchemaError("adapter capabilities.evidence_kinds must be a list")
        evidence = tuple(self.evidence_kinds)
        if any(not isinstance(item, EvidenceKind) for item in evidence):
            raise SchemaError(
                "adapter capabilities.evidence_kinds contains an invalid evidence kind"
            )
        if len(evidence) != len(set(evidence)):
            raise SchemaError("adapter capabilities.evidence_kinds contains duplicates")
        _version(
            self.clarification_protocol,
            "adapter capabilities.clarification_protocol",
        )
        _version(self.trace_protocol, "adapter capabilities.trace_protocol")
        if self.subprocess_wire_version is not None:
            _version(
                self.subprocess_wire_version,
                "adapter capabilities.subprocess_wire_version",
            )
        _version(self.isolation_profile, "adapter capabilities.isolation_profile")
        object.__setattr__(
            self, "target_kinds", tuple(sorted(targets, key=lambda item: item.value))
        )
        object.__setattr__(
            self,
            "evidence_kinds",
            tuple(sorted(evidence, key=lambda item: item.value)),
        )
        validate_safe_json(self.to_dict(), "adapter capabilities")

    @classmethod
    def from_dict(cls, value: Any) -> "AdapterCapabilitiesV2":
        payload = _object(value, "AdapterCapabilitiesV2")
        if "schema_version" in payload:
            _required_protocol_version(payload["schema_version"], cls.SCHEMA_VERSION)
        fields = (
            "schema_version",
            "adapter_id",
            "adapter_version",
            "input_schema_version",
            "submission_schema_version",
            "target_kinds",
            "evidence_kinds",
            "clarification_protocol",
            "trace_protocol",
            "subprocess_wire_version",
            "isolation_profile",
        )
        _exact_fields(payload, fields, "AdapterCapabilitiesV2")
        targets = _array(
            payload["target_kinds"],
            "adapter capabilities.target_kinds",
            len(ReviewTargetKind),
        )
        evidence = _array(
            payload["evidence_kinds"],
            "adapter capabilities.evidence_kinds",
            len(EvidenceKind),
        )
        return cls(
            schema_version=_required_protocol_version(
                payload["schema_version"], cls.SCHEMA_VERSION
            ),
            adapter_id=_identifier(
                payload["adapter_id"], "adapter capabilities.adapter_id"
            ),
            adapter_version=_version(
                payload["adapter_version"],
                "adapter capabilities.adapter_version",
            ),
            input_schema_version=_required_protocol_version(
                payload["input_schema_version"], EVAL_INPUT_SCHEMA_VERSION
            ),
            submission_schema_version=_required_protocol_version(
                payload["submission_schema_version"], EVAL_SUBMISSION_SCHEMA_VERSION
            ),
            target_kinds=tuple(
                _enum_value(
                    ReviewTargetKind,
                    item,
                    "adapter capabilities.target_kinds[%d]" % index,
                )
                for index, item in enumerate(targets)
            ),
            evidence_kinds=tuple(
                _enum_value(
                    EvidenceKind,
                    item,
                    "adapter capabilities.evidence_kinds[%d]" % index,
                )
                for index, item in enumerate(evidence)
            ),
            clarification_protocol=_version(
                payload["clarification_protocol"],
                "adapter capabilities.clarification_protocol",
            ),
            trace_protocol=_version(
                payload["trace_protocol"],
                "adapter capabilities.trace_protocol",
            ),
            subprocess_wire_version=(
                None
                if payload["subprocess_wire_version"] is None
                else _version(
                    payload["subprocess_wire_version"],
                    "adapter capabilities.subprocess_wire_version",
                )
            ),
            isolation_profile=_version(
                payload["isolation_profile"],
                "adapter capabilities.isolation_profile",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "input_schema_version": self.input_schema_version,
            "submission_schema_version": self.submission_schema_version,
            "target_kinds": [item.value for item in self.target_kinds],
            "evidence_kinds": [item.value for item in self.evidence_kinds],
            "clarification_protocol": self.clarification_protocol,
            "trace_protocol": self.trace_protocol,
            "subprocess_wire_version": self.subprocess_wire_version,
            "isolation_profile": self.isolation_profile,
        }


@dataclass(frozen=True)
class SuiteRunConfig(_JsonModel):
    suite_id: str
    suite_version: str
    wire_contract: WireContractV2
    preparation_binding_digest: Optional[str]
    manifest_digest: str
    case_snapshot_id: str
    case_snapshot_digest: str
    cases: Tuple[SuiteCase, ...]

    def __post_init__(self) -> None:
        _identifier(self.suite_id, "suite.suite_id")
        _identifier(self.suite_version, "suite.suite_version")
        if not isinstance(self.wire_contract, WireContractV2):
            raise SchemaError("suite.wire_contract must be a WireContractV2")
        if self.preparation_binding_digest is not None:
            _digest(
                self.preparation_binding_digest,
                "suite.preparation_binding_digest",
            )
        _digest(self.manifest_digest, "suite.manifest_digest")
        validate_run_case_snapshot_id(
            self.case_snapshot_id, "suite.case_snapshot_id"
        )
        _digest(self.case_snapshot_digest, "suite.case_snapshot_digest")
        if type(self.cases) not in (list, tuple):
            raise SchemaError("suite.cases must be a list or tuple")
        if not self.cases or len(self.cases) > MAX_SUITE_CASES:
            raise SchemaError(
                "suite.cases must contain between 1 and %d entries" % MAX_SUITE_CASES
            )
        cases = tuple(self.cases)
        if any(not isinstance(item, SuiteCase) for item in cases):
            raise SchemaError("suite.cases must contain only SuiteCase values")
        task_ids = [item.task_id for item in cases]
        if len(task_ids) != len(set(task_ids)):
            raise SchemaError("suite.cases contains duplicate task_id values")
        object.__setattr__(self, "cases", tuple(sorted(cases, key=lambda item: item.task_id)))

    @classmethod
    def from_dict(cls, value: Any) -> "SuiteRunConfig":
        payload = _object(value, "suite")
        _exact_fields(
            payload,
            (
                "suite_id",
                "suite_version",
                "wire_contract",
                "preparation_binding_digest",
                "manifest_digest",
                "case_snapshot_id",
                "case_snapshot_digest",
                "cases",
            ),
            "suite",
        )
        cases = _array(payload["cases"], "suite.cases", MAX_SUITE_CASES)
        if not cases:
            raise SchemaError("suite.cases must not be empty")
        return cls(
            suite_id=_identifier(payload["suite_id"], "suite.suite_id"),
            suite_version=_identifier(
                payload["suite_version"], "suite.suite_version"
            ),
            wire_contract=WireContractV2.from_dict(payload["wire_contract"]),
            preparation_binding_digest=(
                None
                if payload["preparation_binding_digest"] is None
                else _digest(
                    payload["preparation_binding_digest"],
                    "suite.preparation_binding_digest",
                )
            ),
            manifest_digest=_digest(
                payload["manifest_digest"], "suite.manifest_digest"
            ),
            case_snapshot_id=validate_run_case_snapshot_id(
                payload["case_snapshot_id"], "suite.case_snapshot_id"
            ),
            case_snapshot_digest=_digest(
                payload["case_snapshot_digest"], "suite.case_snapshot_digest"
            ),
            cases=tuple(SuiteCase.from_dict(item) for item in cases),
        )

    @classmethod
    def from_case_snapshot(cls, snapshot: RunCaseSnapshot) -> "SuiteRunConfig":
        """Bind Task 2's verified, truth-free RunCaseSnapshot without copying truth."""

        if not isinstance(snapshot, RunCaseSnapshot):
            raise SchemaError("snapshot must be a verified RunCaseSnapshot")
        return cls(
            suite_id=snapshot.manifest.suite_id,
            suite_version=snapshot.manifest.suite_version,
            wire_contract=snapshot.wire_contract,
            preparation_binding_digest=(
                None
                if snapshot.manifest.source.preparation_binding is None
                else snapshot.manifest.source.preparation_binding.digest()
            ),
            manifest_digest=snapshot.manifest.digest(),
            case_snapshot_id=snapshot.snapshot_id,
            case_snapshot_digest=snapshot.snapshot_digest,
            cases=tuple(item.manifest_case for item in snapshot.cases),
        )

    def case(self, task_id: str) -> SuiteCase:
        for item in self.cases:
            if item.task_id == task_id:
                return item
        raise SchemaError("task_id is not bound by this run suite")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "wire_contract": self.wire_contract.to_dict(),
            "preparation_binding_digest": self.preparation_binding_digest,
            "manifest_digest": self.manifest_digest,
            "case_snapshot_id": self.case_snapshot_id,
            "case_snapshot_digest": self.case_snapshot_digest,
            "cases": [item.to_dict() for item in self.cases],
        }


@dataclass(frozen=True)
class ResourceBudgets(_JsonModel):
    agent_timeout_seconds: Any
    evaluator_timeout_seconds: Any
    max_agent_output_bytes: int
    max_trace_bytes: int
    max_execution_artifact_file_bytes: int
    max_execution_artifact_total_bytes: int
    max_parallel_trials: int

    def __post_init__(self) -> None:
        _positive_number(self.agent_timeout_seconds, "budgets.agent_timeout_seconds")
        _positive_number(
            self.evaluator_timeout_seconds, "budgets.evaluator_timeout_seconds"
        )
        values = {
            "max_agent_output_bytes": self.max_agent_output_bytes,
            "max_trace_bytes": self.max_trace_bytes,
            "max_execution_artifact_file_bytes": self.max_execution_artifact_file_bytes,
            "max_execution_artifact_total_bytes": self.max_execution_artifact_total_bytes,
        }
        for name, value in values.items():
            _integer(
                value,
                "budgets.%s" % name,
                minimum=1,
                maximum=MAX_ARTIFACT_BUDGET_BYTES,
            )
        _integer(
            self.max_parallel_trials,
            "budgets.max_parallel_trials",
            minimum=1,
            maximum=MAX_TRIAL_COUNT,
        )
        if (
            self.max_execution_artifact_file_bytes
            > self.max_execution_artifact_total_bytes
        ):
            raise SchemaError(
                "budgets.max_execution_artifact_file_bytes may not exceed total execution artifact bytes"
            )
        if (
            self.max_agent_output_bytes
            > self.max_execution_artifact_total_bytes
        ):
            raise SchemaError(
                "budgets.max_agent_output_bytes may not exceed total artifact bytes"
            )
        if self.max_trace_bytes > self.max_execution_artifact_total_bytes:
            raise SchemaError("budgets.max_trace_bytes may not exceed total artifact bytes")

    @classmethod
    def from_dict(cls, value: Any) -> "ResourceBudgets":
        payload = _object(value, "resource_budgets")
        fields = (
            "agent_timeout_seconds",
            "evaluator_timeout_seconds",
            "max_agent_output_bytes",
            "max_trace_bytes",
            "max_execution_artifact_file_bytes",
            "max_execution_artifact_total_bytes",
            "max_parallel_trials",
        )
        _exact_fields(payload, fields, "resource_budgets")
        return cls(
            agent_timeout_seconds=_positive_number(
                payload["agent_timeout_seconds"], "budgets.agent_timeout_seconds"
            ),
            evaluator_timeout_seconds=_positive_number(
                payload["evaluator_timeout_seconds"],
                "budgets.evaluator_timeout_seconds",
            ),
            max_agent_output_bytes=_integer(
                payload["max_agent_output_bytes"],
                "budgets.max_agent_output_bytes",
                minimum=1,
                maximum=MAX_ARTIFACT_BUDGET_BYTES,
            ),
            max_trace_bytes=_integer(
                payload["max_trace_bytes"],
                "budgets.max_trace_bytes",
                minimum=1,
                maximum=MAX_ARTIFACT_BUDGET_BYTES,
            ),
            max_execution_artifact_file_bytes=_integer(
                payload["max_execution_artifact_file_bytes"],
                "budgets.max_execution_artifact_file_bytes",
                minimum=1,
                maximum=MAX_ARTIFACT_BUDGET_BYTES,
            ),
            max_execution_artifact_total_bytes=_integer(
                payload["max_execution_artifact_total_bytes"],
                "budgets.max_execution_artifact_total_bytes",
                minimum=1,
                maximum=MAX_ARTIFACT_BUDGET_BYTES,
            ),
            max_parallel_trials=_integer(
                payload["max_parallel_trials"],
                "budgets.max_parallel_trials",
                minimum=1,
                maximum=MAX_TRIAL_COUNT,
            ),
        )

    @classmethod
    def defaults(cls) -> "ResourceBudgets":
        return cls(
            agent_timeout_seconds=1_800,
            evaluator_timeout_seconds=600,
            max_agent_output_bytes=16 * 1024 * 1024,
            max_trace_bytes=16 * 1024 * 1024,
            max_execution_artifact_file_bytes=32 * 1024 * 1024,
            max_execution_artifact_total_bytes=512 * 1024 * 1024,
            max_parallel_trials=4,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_timeout_seconds": self.agent_timeout_seconds,
            "evaluator_timeout_seconds": self.evaluator_timeout_seconds,
            "max_agent_output_bytes": self.max_agent_output_bytes,
            "max_trace_bytes": self.max_trace_bytes,
            "max_execution_artifact_file_bytes": self.max_execution_artifact_file_bytes,
            "max_execution_artifact_total_bytes": self.max_execution_artifact_total_bytes,
            "max_parallel_trials": self.max_parallel_trials,
        }


@dataclass(frozen=True)
class JudgeExecutionBudgets(_JsonModel):
    """Bound every semantic-Judge attempt, deadline, token, and byte surface."""

    SCHEMA_VERSION: ClassVar[str] = JUDGE_EXECUTION_BUDGETS_SCHEMA_VERSION

    schema_version: str
    max_attempts_per_request: int
    attempt_timeout_seconds: Any
    request_deadline_seconds: Any
    max_parallel_requests: int
    max_context_blocks_per_request: int
    max_context_block_bytes: int
    max_context_bytes_per_request: int
    max_model_request_bytes: int
    max_model_response_bytes: int
    max_model_request_tokens: int
    max_model_response_tokens: int
    max_reason_refs: int
    max_total_judge_request_bytes: int
    max_total_judge_response_bytes: int
    max_total_judge_request_tokens: int
    max_total_judge_response_tokens: int

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise SchemaError("JudgeExecutionBudgets has an unknown schema_version")
        _integer(
            self.max_attempts_per_request,
            "judge budgets.max_attempts_per_request",
            minimum=1,
            maximum=MAX_JUDGE_ATTEMPTS_PER_REQUEST,
        )
        _positive_number(
            self.attempt_timeout_seconds,
            "judge budgets.attempt_timeout_seconds",
        )
        _positive_number(
            self.request_deadline_seconds,
            "judge budgets.request_deadline_seconds",
        )
        if self.attempt_timeout_seconds > self.request_deadline_seconds:
            raise SchemaError(
                "Judge attempt timeout may not exceed request deadline"
            )
        if (
            self.attempt_timeout_seconds * self.max_attempts_per_request
            > self.request_deadline_seconds
        ):
            raise SchemaError(
                "Judge request deadline must cover every configured attempt timeout"
            )
        _integer(
            self.max_parallel_requests,
            "judge budgets.max_parallel_requests",
            minimum=1,
            maximum=MAX_JUDGE_REQUESTS,
        )
        _integer(
            self.max_context_blocks_per_request,
            "judge budgets.max_context_blocks_per_request",
            minimum=1,
            maximum=MAX_JUDGE_CONTEXT_BLOCKS_PER_REQUEST,
        )
        for name in (
            "max_context_block_bytes",
            "max_context_bytes_per_request",
            "max_model_request_bytes",
            "max_model_response_bytes",
            "max_total_judge_request_bytes",
            "max_total_judge_response_bytes",
        ):
            _integer(
                getattr(self, name),
                "judge budgets.%s" % name,
                minimum=1,
                maximum=MAX_ARTIFACT_BUDGET_BYTES,
            )
        for name in (
            "max_model_request_tokens",
            "max_model_response_tokens",
            "max_total_judge_request_tokens",
            "max_total_judge_response_tokens",
        ):
            _integer(
                getattr(self, name),
                "judge budgets.%s" % name,
                minimum=1,
                maximum=MAX_JUDGE_TOKEN_BUDGET,
            )
        _integer(
            self.max_reason_refs,
            "judge budgets.max_reason_refs",
            minimum=1,
            maximum=MAX_JUDGE_REASON_REFS,
        )
        if self.max_context_block_bytes > self.max_context_bytes_per_request:
            raise SchemaError(
                "judge context block bytes may not exceed per-request context bytes"
            )
        if self.max_context_bytes_per_request > self.max_model_request_bytes:
            raise SchemaError(
                "judge per-request context bytes may not exceed model request bytes"
            )
        if self.max_model_request_bytes > self.max_total_judge_request_bytes:
            raise SchemaError(
                "judge model request bytes may not exceed total Judge request bytes"
            )
        if self.max_model_response_bytes > self.max_total_judge_response_bytes:
            raise SchemaError(
                "judge model response bytes may not exceed total Judge response bytes"
            )
        if self.max_model_request_tokens > self.max_total_judge_request_tokens:
            raise SchemaError(
                "judge model request tokens may not exceed total Judge request tokens"
            )
        if self.max_model_response_tokens > self.max_total_judge_response_tokens:
            raise SchemaError(
                "judge model response tokens may not exceed total Judge response tokens"
            )
        if self.max_reason_refs > self.max_context_blocks_per_request + 2:
            raise SchemaError(
                "judge reason refs may not exceed context blocks plus two subject refs"
            )
        validate_safe_json(self.to_dict(), "judge budgets")
        _check_model_size(
            self,
            MAX_EVAL_RUN_CONFIG_BYTES,
            "JudgeExecutionBudgets",
        )

    @classmethod
    def defaults(
        cls,
        *,
        evaluator_timeout_seconds: Any = 60,
        max_execution_artifact_file_bytes: int = 32 * 1024 * 1024,
        max_execution_artifact_total_bytes: int = 512 * 1024 * 1024,
    ) -> "JudgeExecutionBudgets":
        timeout = _positive_number(
            evaluator_timeout_seconds,
            "evaluator_timeout_seconds",
        )
        file_bytes = _integer(
            max_execution_artifact_file_bytes,
            "max_execution_artifact_file_bytes",
            minimum=1,
            maximum=MAX_ARTIFACT_BUDGET_BYTES,
        )
        total_bytes = _integer(
            max_execution_artifact_total_bytes,
            "max_execution_artifact_total_bytes",
            minimum=1,
            maximum=MAX_ARTIFACT_BUDGET_BYTES,
        )
        if file_bytes > total_bytes:
            raise SchemaError(
                "max_execution_artifact_file_bytes may not exceed total bytes"
            )
        model_request_bytes = min(8 * 1024 * 1024, file_bytes)
        context_bytes = min(6 * 1024 * 1024, model_request_bytes)
        max_attempts = 2
        request_deadline = min(120, timeout)
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            max_attempts_per_request=max_attempts,
            attempt_timeout_seconds=min(60, request_deadline / max_attempts),
            request_deadline_seconds=request_deadline,
            max_parallel_requests=4,
            max_context_blocks_per_request=256,
            max_context_block_bytes=min(512 * 1024, context_bytes),
            max_context_bytes_per_request=context_bytes,
            max_model_request_bytes=model_request_bytes,
            max_model_response_bytes=min(1024 * 1024, file_bytes),
            max_model_request_tokens=131_072,
            max_model_response_tokens=8_192,
            max_reason_refs=MAX_JUDGE_REASON_REFS,
            max_total_judge_request_bytes=min(64 * 1024 * 1024, total_bytes),
            max_total_judge_response_bytes=min(16 * 1024 * 1024, total_bytes),
            max_total_judge_request_tokens=1_048_576,
            max_total_judge_response_tokens=65_536,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "JudgeExecutionBudgets":
        payload = _object(value, "JudgeExecutionBudgets")
        fields = (
            "schema_version",
            "max_attempts_per_request",
            "attempt_timeout_seconds",
            "request_deadline_seconds",
            "max_parallel_requests",
            "max_context_blocks_per_request",
            "max_context_block_bytes",
            "max_context_bytes_per_request",
            "max_model_request_bytes",
            "max_model_response_bytes",
            "max_model_request_tokens",
            "max_model_response_tokens",
            "max_reason_refs",
            "max_total_judge_request_bytes",
            "max_total_judge_response_bytes",
            "max_total_judge_request_tokens",
            "max_total_judge_response_tokens",
        )
        _exact_fields(payload, fields, "JudgeExecutionBudgets")
        return cls(
            schema_version=payload["schema_version"],
            max_attempts_per_request=_integer(
                payload["max_attempts_per_request"],
                "judge budgets.max_attempts_per_request",
                minimum=1,
                maximum=MAX_JUDGE_ATTEMPTS_PER_REQUEST,
            ),
            attempt_timeout_seconds=_positive_number(
                payload["attempt_timeout_seconds"],
                "judge budgets.attempt_timeout_seconds",
            ),
            request_deadline_seconds=_positive_number(
                payload["request_deadline_seconds"],
                "judge budgets.request_deadline_seconds",
            ),
            max_parallel_requests=_integer(
                payload["max_parallel_requests"],
                "judge budgets.max_parallel_requests",
                minimum=1,
                maximum=MAX_JUDGE_REQUESTS,
            ),
            max_context_blocks_per_request=_integer(
                payload["max_context_blocks_per_request"],
                "judge budgets.max_context_blocks_per_request",
                minimum=1,
                maximum=MAX_JUDGE_CONTEXT_BLOCKS_PER_REQUEST,
            ),
            max_context_block_bytes=_integer(
                payload["max_context_block_bytes"],
                "judge budgets.max_context_block_bytes",
                minimum=1,
                maximum=MAX_ARTIFACT_BUDGET_BYTES,
            ),
            max_context_bytes_per_request=_integer(
                payload["max_context_bytes_per_request"],
                "judge budgets.max_context_bytes_per_request",
                minimum=1,
                maximum=MAX_ARTIFACT_BUDGET_BYTES,
            ),
            max_model_request_bytes=_integer(
                payload["max_model_request_bytes"],
                "judge budgets.max_model_request_bytes",
                minimum=1,
                maximum=MAX_ARTIFACT_BUDGET_BYTES,
            ),
            max_model_response_bytes=_integer(
                payload["max_model_response_bytes"],
                "judge budgets.max_model_response_bytes",
                minimum=1,
                maximum=MAX_ARTIFACT_BUDGET_BYTES,
            ),
            max_model_request_tokens=_integer(
                payload["max_model_request_tokens"],
                "judge budgets.max_model_request_tokens",
                minimum=1,
                maximum=MAX_JUDGE_TOKEN_BUDGET,
            ),
            max_model_response_tokens=_integer(
                payload["max_model_response_tokens"],
                "judge budgets.max_model_response_tokens",
                minimum=1,
                maximum=MAX_JUDGE_TOKEN_BUDGET,
            ),
            max_reason_refs=_integer(
                payload["max_reason_refs"],
                "judge budgets.max_reason_refs",
                minimum=1,
                maximum=MAX_JUDGE_REASON_REFS,
            ),
            max_total_judge_request_bytes=_integer(
                payload["max_total_judge_request_bytes"],
                "judge budgets.max_total_judge_request_bytes",
                minimum=1,
                maximum=MAX_ARTIFACT_BUDGET_BYTES,
            ),
            max_total_judge_response_bytes=_integer(
                payload["max_total_judge_response_bytes"],
                "judge budgets.max_total_judge_response_bytes",
                minimum=1,
                maximum=MAX_ARTIFACT_BUDGET_BYTES,
            ),
            max_total_judge_request_tokens=_integer(
                payload["max_total_judge_request_tokens"],
                "judge budgets.max_total_judge_request_tokens",
                minimum=1,
                maximum=MAX_JUDGE_TOKEN_BUDGET,
            ),
            max_total_judge_response_tokens=_integer(
                payload["max_total_judge_response_tokens"],
                "judge budgets.max_total_judge_response_tokens",
                minimum=1,
                maximum=MAX_JUDGE_TOKEN_BUDGET,
            ),
        )

    @classmethod
    def from_json(cls, data: Any) -> "JudgeExecutionBudgets":
        return cls.from_dict(
            _strict_json_loads(
                data,
                MAX_EVAL_RUN_CONFIG_BYTES,
                "JudgeExecutionBudgets JSON",
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "max_attempts_per_request": self.max_attempts_per_request,
            "attempt_timeout_seconds": self.attempt_timeout_seconds,
            "request_deadline_seconds": self.request_deadline_seconds,
            "max_parallel_requests": self.max_parallel_requests,
            "max_context_blocks_per_request": self.max_context_blocks_per_request,
            "max_context_block_bytes": self.max_context_block_bytes,
            "max_context_bytes_per_request": self.max_context_bytes_per_request,
            "max_model_request_bytes": self.max_model_request_bytes,
            "max_model_response_bytes": self.max_model_response_bytes,
            "max_model_request_tokens": self.max_model_request_tokens,
            "max_model_response_tokens": self.max_model_response_tokens,
            "max_reason_refs": self.max_reason_refs,
            "max_total_judge_request_bytes": self.max_total_judge_request_bytes,
            "max_total_judge_response_bytes": self.max_total_judge_response_bytes,
            "max_total_judge_request_tokens": self.max_total_judge_request_tokens,
            "max_total_judge_response_tokens": self.max_total_judge_response_tokens,
        }


@dataclass(frozen=True)
class EvaluatorExecutionConfig(_JsonModel):
    SCHEMA_VERSION: ClassVar[str] = EVALUATOR_EXECUTION_CONFIG_SCHEMA_VERSION

    schema_version: str
    evaluator: EvaluatorRunConfig
    evaluator_config_digest: str
    judge_budgets: JudgeExecutionBudgets
    cache_policy_version: str
    review_evaluator_context_policy_version: str
    metric_authority_policy_version: str
    evaluator_timeout_seconds: Any
    max_execution_artifact_file_bytes: int
    max_execution_artifact_total_bytes: int

    def __post_init__(self) -> None:
        _required_protocol_version(self.schema_version, self.SCHEMA_VERSION)
        if not isinstance(self.evaluator, EvaluatorRunConfig):
            raise SchemaError(
                "evaluator execution.evaluator must be an EvaluatorRunConfig"
            )
        if self.evaluator_config_digest != self.evaluator.digest():
            raise SchemaError(
                "evaluator execution config digest does not match evaluator"
            )
        if not isinstance(self.judge_budgets, JudgeExecutionBudgets):
            raise SchemaError(
                "evaluator execution.judge_budgets must be JudgeExecutionBudgets"
            )
        _version(
            self.cache_policy_version,
            "evaluator execution.cache_policy_version",
        )
        _version(
            self.review_evaluator_context_policy_version,
            "evaluator execution.review_evaluator_context_policy_version",
        )
        _version(
            self.metric_authority_policy_version,
            "evaluator execution.metric_authority_policy_version",
        )
        evaluator_timeout = _positive_number(
            self.evaluator_timeout_seconds,
            "evaluator execution.evaluator_timeout_seconds",
        )
        _integer(
            self.max_execution_artifact_file_bytes,
            "evaluator execution.max_execution_artifact_file_bytes",
            minimum=1,
            maximum=MAX_ARTIFACT_BUDGET_BYTES,
        )
        _integer(
            self.max_execution_artifact_total_bytes,
            "evaluator execution.max_execution_artifact_total_bytes",
            minimum=1,
            maximum=MAX_ARTIFACT_BUDGET_BYTES,
        )
        if (
            self.max_execution_artifact_file_bytes
            > self.max_execution_artifact_total_bytes
        ):
            raise SchemaError(
                "evaluator execution file bytes may not exceed total bytes"
            )
        if self.judge_budgets.attempt_timeout_seconds > evaluator_timeout:
            raise SchemaError(
                "Judge attempt timeout may not exceed evaluator timeout"
            )
        if self.judge_budgets.request_deadline_seconds > evaluator_timeout:
            raise SchemaError(
                "Judge request deadline may not exceed evaluator timeout"
            )
        if (
            self.judge_budgets.max_model_request_bytes
            > self.max_execution_artifact_file_bytes
        ):
            raise SchemaError(
                "Judge model request bytes may not exceed execution artifact file bytes"
            )
        if (
            self.judge_budgets.max_model_response_bytes
            > self.max_execution_artifact_file_bytes
        ):
            raise SchemaError(
                "Judge model response bytes may not exceed execution artifact file bytes"
            )
        if (
            self.judge_budgets.max_total_judge_request_bytes
            > self.max_execution_artifact_total_bytes
        ):
            raise SchemaError(
                "total Judge request bytes may not exceed total execution artifact bytes"
            )
        if (
            self.judge_budgets.max_total_judge_response_bytes
            > self.max_execution_artifact_total_bytes
        ):
            raise SchemaError(
                "total Judge response bytes may not exceed total execution artifact bytes"
            )
        validate_safe_json(self.to_dict(), "evaluator execution")
        _check_model_size(
            self,
            MAX_EVAL_RUN_CONFIG_BYTES,
            "EvaluatorExecutionConfig",
        )

    def validate_runtime_policy_support(self) -> None:
        """Reject persisted policy identities this runtime cannot execute."""

        _required_protocol_version(
            self.review_evaluator_context_policy_version,
            DEFAULT_REVIEW_EVALUATOR_CONTEXT_POLICY_VERSION,
        )
        _required_protocol_version(
            self.metric_authority_policy_version,
            DEFAULT_METRIC_AUTHORITY_POLICY_VERSION,
        )

    @classmethod
    def create(
        cls,
        *,
        evaluator: EvaluatorRunConfig,
        evaluator_timeout_seconds: Any,
        max_execution_artifact_file_bytes: int,
        max_execution_artifact_total_bytes: int,
        judge_budgets: Optional[JudgeExecutionBudgets] = None,
        cache_policy_version: str = DEFAULT_JUDGE_CACHE_POLICY_VERSION,
        review_evaluator_context_policy_version: str = (
            DEFAULT_REVIEW_EVALUATOR_CONTEXT_POLICY_VERSION
        ),
        metric_authority_policy_version: str = (
            DEFAULT_METRIC_AUTHORITY_POLICY_VERSION
        ),
    ) -> "EvaluatorExecutionConfig":
        if not isinstance(evaluator, EvaluatorRunConfig):
            raise SchemaError("evaluator must be an EvaluatorRunConfig")
        if judge_budgets is None:
            judge_budgets = JudgeExecutionBudgets.defaults(
                evaluator_timeout_seconds=evaluator_timeout_seconds,
                max_execution_artifact_file_bytes=(
                    max_execution_artifact_file_bytes
                ),
                max_execution_artifact_total_bytes=(
                    max_execution_artifact_total_bytes
                ),
            )
        elif not isinstance(judge_budgets, JudgeExecutionBudgets):
            raise SchemaError("judge_budgets must be JudgeExecutionBudgets")
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            evaluator=evaluator,
            evaluator_config_digest=evaluator.digest(),
            judge_budgets=judge_budgets,
            cache_policy_version=cache_policy_version,
            review_evaluator_context_policy_version=(
                review_evaluator_context_policy_version
            ),
            metric_authority_policy_version=(
                metric_authority_policy_version
            ),
            evaluator_timeout_seconds=evaluator_timeout_seconds,
            max_execution_artifact_file_bytes=max_execution_artifact_file_bytes,
            max_execution_artifact_total_bytes=max_execution_artifact_total_bytes,
        )

    @classmethod
    def from_resource_budgets(
        cls,
        evaluator: EvaluatorRunConfig,
        resource_budgets: ResourceBudgets,
        *,
        judge_budgets: Optional[JudgeExecutionBudgets] = None,
        cache_policy_version: str = DEFAULT_JUDGE_CACHE_POLICY_VERSION,
        review_evaluator_context_policy_version: str = (
            DEFAULT_REVIEW_EVALUATOR_CONTEXT_POLICY_VERSION
        ),
        metric_authority_policy_version: str = (
            DEFAULT_METRIC_AUTHORITY_POLICY_VERSION
        ),
    ) -> "EvaluatorExecutionConfig":
        if not isinstance(resource_budgets, ResourceBudgets):
            raise SchemaError("resource_budgets must be a ResourceBudgets")
        return cls.create(
            evaluator=evaluator,
            evaluator_timeout_seconds=resource_budgets.evaluator_timeout_seconds,
            max_execution_artifact_file_bytes=(
                resource_budgets.max_execution_artifact_file_bytes
            ),
            max_execution_artifact_total_bytes=(
                resource_budgets.max_execution_artifact_total_bytes
            ),
            judge_budgets=judge_budgets,
            cache_policy_version=cache_policy_version,
            review_evaluator_context_policy_version=(
                review_evaluator_context_policy_version
            ),
            metric_authority_policy_version=(
                metric_authority_policy_version
            ),
        )

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluatorExecutionConfig":
        payload = _object(value, "EvaluatorExecutionConfig")
        if "schema_version" in payload:
            _required_protocol_version(
                payload["schema_version"], cls.SCHEMA_VERSION
            )
        _exact_fields(
            payload,
            (
                "schema_version",
                "evaluator",
                "evaluator_config_digest",
                "judge_budgets",
                "cache_policy_version",
                "review_evaluator_context_policy_version",
                "metric_authority_policy_version",
                "evaluator_timeout_seconds",
                "max_execution_artifact_file_bytes",
                "max_execution_artifact_total_bytes",
            ),
            "EvaluatorExecutionConfig",
        )
        return cls(
            schema_version=payload["schema_version"],
            evaluator=EvaluatorRunConfig.from_dict(payload["evaluator"]),
            evaluator_config_digest=_digest(
                payload["evaluator_config_digest"],
                "evaluator execution.evaluator_config_digest",
            ),
            judge_budgets=JudgeExecutionBudgets.from_dict(
                payload["judge_budgets"]
            ),
            cache_policy_version=_version(
                payload["cache_policy_version"],
                "evaluator execution.cache_policy_version",
            ),
            review_evaluator_context_policy_version=_version(
                payload["review_evaluator_context_policy_version"],
                "evaluator execution.review_evaluator_context_policy_version",
            ),
            metric_authority_policy_version=_version(
                payload["metric_authority_policy_version"],
                "evaluator execution.metric_authority_policy_version",
            ),
            evaluator_timeout_seconds=_positive_number(
                payload["evaluator_timeout_seconds"],
                "evaluator execution.evaluator_timeout_seconds",
            ),
            max_execution_artifact_file_bytes=_integer(
                payload["max_execution_artifact_file_bytes"],
                "evaluator execution.max_execution_artifact_file_bytes",
                minimum=1,
                maximum=MAX_ARTIFACT_BUDGET_BYTES,
            ),
            max_execution_artifact_total_bytes=_integer(
                payload["max_execution_artifact_total_bytes"],
                "evaluator execution.max_execution_artifact_total_bytes",
                minimum=1,
                maximum=MAX_ARTIFACT_BUDGET_BYTES,
            ),
        )

    @classmethod
    def from_json(cls, data: Any) -> "EvaluatorExecutionConfig":
        return cls.from_dict(
            _strict_json_loads(
                data,
                MAX_EVAL_RUN_CONFIG_BYTES,
                "EvaluatorExecutionConfig JSON",
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluator": self.evaluator.to_dict(),
            "evaluator_config_digest": self.evaluator_config_digest,
            "judge_budgets": self.judge_budgets.to_dict(),
            "cache_policy_version": self.cache_policy_version,
            "review_evaluator_context_policy_version": (
                self.review_evaluator_context_policy_version
            ),
            "metric_authority_policy_version": (
                self.metric_authority_policy_version
            ),
            "evaluator_timeout_seconds": self.evaluator_timeout_seconds,
            "max_execution_artifact_file_bytes": self.max_execution_artifact_file_bytes,
            "max_execution_artifact_total_bytes": self.max_execution_artifact_total_bytes,
        }


@dataclass(frozen=True)
class EvalRunConfig(_JsonModel):
    SCHEMA_VERSION: ClassVar[str] = EVAL_RUN_CONFIG_SCHEMA_VERSION

    schema_version: str
    run_id: str
    run_instance_key: str
    agent: AgentConfigSnapshot
    agent_config_digest: str
    clarification_matcher: ClarificationMatcherSnapshot
    clarification_matcher_config_digest: str
    evaluator: EvaluatorRunConfig
    evaluator_config_digest: str
    suite: SuiteRunConfig
    wire_contract: WireContractV2
    suite_preparation_binding_digest: Optional[str]
    adapter_capabilities: AdapterCapabilitiesV2
    adapter_capabilities_digest: str
    target_kinds: Tuple[ReviewTargetKind, ...]
    materializer_protocol: str
    trial_count: int
    resource_budgets: ResourceBudgets

    def __post_init__(self) -> None:
        _required_protocol_version(self.schema_version, self.SCHEMA_VERSION)
        validate_run_id(self.run_id)
        _identifier(self.run_instance_key, "run_config.run_instance_key")
        validate_safe_text(self.run_instance_key, "run_config.run_instance_key")
        if not isinstance(self.agent, AgentConfigSnapshot):
            raise SchemaError(
                "run_config.agent must be a persisted AgentConfigSnapshot"
            )
        if not isinstance(
            self.clarification_matcher,
            ClarificationMatcherSnapshot,
        ):
            raise SchemaError(
                "run_config.clarification_matcher must be a ClarificationMatcherSnapshot"
            )
        if not isinstance(self.evaluator, EvaluatorRunConfig):
            raise SchemaError("run_config.evaluator must be an EvaluatorRunConfig")
        if not isinstance(self.suite, SuiteRunConfig):
            raise SchemaError("run_config.suite must be a SuiteRunConfig")
        if self.wire_contract != self.suite.wire_contract:
            raise SchemaError("run_config.wire_contract does not match suite")
        if (
            self.suite_preparation_binding_digest
            != self.suite.preparation_binding_digest
        ):
            raise SchemaError(
                "run_config.suite_preparation_binding_digest does not match suite"
            )
        if not isinstance(self.adapter_capabilities, AdapterCapabilitiesV2):
            raise SchemaError(
                "run_config.adapter_capabilities must be an AdapterCapabilitiesV2"
            )
        if self.adapter_capabilities_digest != self.adapter_capabilities.digest():
            raise SchemaError(
                "adapter_capabilities_digest does not match adapter capabilities"
            )
        if type(self.target_kinds) not in (tuple, list):
            raise SchemaError("run_config.target_kinds must be a list")
        targets = tuple(self.target_kinds)
        if targets != (self.wire_contract.review_target_kind,):
            raise SchemaError(
                "run_config.target_kinds must contain the suite wire target kind"
            )
        object.__setattr__(self, "target_kinds", targets)
        if (
            self.adapter_capabilities.input_schema_version
            != self.wire_contract.input_schema_version
            or self.adapter_capabilities.submission_schema_version
            != self.wire_contract.submission_schema_version
        ):
            raise SchemaError("adapter capabilities do not match the wire contract")
        if self.materializer_protocol != self.wire_contract.materializer_protocol:
            raise SchemaError(
                "run_config.materializer_protocol does not match the wire contract"
            )
        if not isinstance(self.resource_budgets, ResourceBudgets):
            raise SchemaError(
                "run_config.resource_budgets must be a ResourceBudgets"
            )
        _integer(
            self.trial_count,
            "run_config.trial_count",
            minimum=1,
            maximum=MAX_TRIAL_COUNT,
        )
        planned_trials = len(self.suite.cases) * self.trial_count
        if planned_trials > MAX_PLANNED_TRIALS:
            raise SchemaError(
                "run_config planned Trials exceed the limit of %d"
                % MAX_PLANNED_TRIALS
            )
        if self.resource_budgets.max_parallel_trials > planned_trials:
            raise SchemaError(
                "max_parallel_trials may not exceed the number of planned Trials"
            )
        expected_agent_digest = self.agent.digest()
        expected_matcher_digest = self.clarification_matcher.digest()
        expected_evaluator_digest = self.evaluator.digest()
        if self.agent_config_digest != expected_agent_digest:
            raise SchemaError("agent_config_digest does not match agent config")
        if self.clarification_matcher_config_digest != expected_matcher_digest:
            raise SchemaError(
                "clarification_matcher_config_digest does not match matcher config"
            )
        if self.evaluator_config_digest != expected_evaluator_digest:
            raise SchemaError("evaluator_config_digest does not match evaluator config")
        expected_run_id = self.derive_run_id(
            run_instance_key=self.run_instance_key,
            agent_config_digest=self.agent_config_digest,
            clarification_matcher_config_digest=(
                self.clarification_matcher_config_digest
            ),
            suite=self.suite,
            adapter_capabilities_digest=self.adapter_capabilities_digest,
            trial_count=self.trial_count,
            resource_budgets=self.resource_budgets,
        )
        if self.run_id != expected_run_id:
            raise SchemaError("run_id does not match the canonical run identity")
        validate_safe_json(self.to_dict(), "run_config")
        _check_model_size(self, MAX_EVAL_RUN_CONFIG_BYTES, "EvalRunConfig")

    @staticmethod
    def _identity_payload(
        *,
        run_instance_key: str,
        agent_config_digest: str,
        clarification_matcher_config_digest: str,
        suite: SuiteRunConfig,
        adapter_capabilities_digest: str,
        trial_count: int,
        resource_budgets: ResourceBudgets,
    ) -> Dict[str, Any]:
        """Return agent-side run identity; Judge/evaluator is intentionally absent."""

        return {
            "schema_version": EVAL_RUN_CONFIG_SCHEMA_VERSION,
            "run_instance_key": run_instance_key,
            "agent_config_digest": agent_config_digest,
            "clarification_matcher_config_digest": (
                clarification_matcher_config_digest
            ),
            "agent_resource_budgets": {
                "agent_timeout_seconds": resource_budgets.agent_timeout_seconds,
                "max_agent_output_bytes": resource_budgets.max_agent_output_bytes,
                "max_trace_bytes": resource_budgets.max_trace_bytes,
                "max_execution_artifact_file_bytes": resource_budgets.max_execution_artifact_file_bytes,
                "max_execution_artifact_total_bytes": resource_budgets.max_execution_artifact_total_bytes,
                "max_parallel_trials": resource_budgets.max_parallel_trials,
            },
            "suite": suite.to_dict(),
            "adapter_capabilities_digest": adapter_capabilities_digest,
            "trial_count": trial_count,
        }

    @classmethod
    def derive_run_id(
        cls,
        *,
        run_instance_key: str,
        agent_config_digest: str,
        clarification_matcher_config_digest: str,
        suite: SuiteRunConfig,
        adapter_capabilities_digest: str,
        trial_count: int,
        resource_budgets: ResourceBudgets,
    ) -> str:
        return stable_id(
            "run",
            cls._identity_payload(
                run_instance_key=run_instance_key,
                agent_config_digest=agent_config_digest,
                clarification_matcher_config_digest=(
                    clarification_matcher_config_digest
                ),
                suite=suite,
                adapter_capabilities_digest=adapter_capabilities_digest,
                trial_count=trial_count,
                resource_budgets=resource_budgets,
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        run_instance_key: str,
        agent: AgentConfigSnapshot,
        clarification_matcher: ClarificationMatcherSnapshot,
        evaluator: EvaluatorRunConfig,
        suite: SuiteRunConfig,
        adapter_capabilities: AdapterCapabilitiesV2,
        trial_count: int,
        resource_budgets: ResourceBudgets,
    ) -> "EvalRunConfig":
        _identifier(run_instance_key, "run_instance_key")
        validate_safe_text(run_instance_key, "run_instance_key")
        if not isinstance(agent, AgentConfigSnapshot):
            raise SchemaError("agent must be a persisted AgentConfigSnapshot")
        if not isinstance(clarification_matcher, ClarificationMatcherSnapshot):
            raise SchemaError(
                "clarification_matcher must be a ClarificationMatcherSnapshot"
            )
        if not isinstance(evaluator, EvaluatorRunConfig):
            raise SchemaError("evaluator must be an EvaluatorRunConfig")
        if not isinstance(suite, SuiteRunConfig):
            raise SchemaError("suite must be a SuiteRunConfig")
        if not isinstance(adapter_capabilities, AdapterCapabilitiesV2):
            raise SchemaError("adapter_capabilities must be an AdapterCapabilitiesV2")
        if not isinstance(resource_budgets, ResourceBudgets):
            raise SchemaError("resource_budgets must be a ResourceBudgets")
        agent_digest = agent.digest()
        matcher_digest = clarification_matcher.digest()
        evaluator_digest = evaluator.digest()
        adapter_capabilities_digest = adapter_capabilities.digest()
        run_id = cls.derive_run_id(
            run_instance_key=run_instance_key,
            agent_config_digest=agent_digest,
            clarification_matcher_config_digest=matcher_digest,
            suite=suite,
            adapter_capabilities_digest=adapter_capabilities_digest,
            trial_count=trial_count,
            resource_budgets=resource_budgets,
        )
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            run_id=run_id,
            run_instance_key=run_instance_key,
            agent=agent,
            agent_config_digest=agent_digest,
            clarification_matcher=clarification_matcher,
            clarification_matcher_config_digest=matcher_digest,
            evaluator=evaluator,
            evaluator_config_digest=evaluator_digest,
            suite=suite,
            wire_contract=suite.wire_contract,
            suite_preparation_binding_digest=(
                suite.preparation_binding_digest
            ),
            adapter_capabilities=adapter_capabilities,
            adapter_capabilities_digest=adapter_capabilities_digest,
            target_kinds=(suite.wire_contract.review_target_kind,),
            materializer_protocol=suite.wire_contract.materializer_protocol,
            trial_count=trial_count,
            resource_budgets=resource_budgets,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "EvalRunConfig":
        payload = _object(value, "EvalRunConfig")
        if "schema_version" not in payload:
            raise SchemaError("EvalRunConfig has missing field(s): schema_version")
        _required_protocol_version(payload["schema_version"], cls.SCHEMA_VERSION)
        _exact_fields(
            payload,
            (
                "schema_version",
                "run_id",
                "run_instance_key",
                "agent",
                "agent_config_digest",
                "clarification_matcher",
                "clarification_matcher_config_digest",
                "evaluator",
                "evaluator_config_digest",
                "suite",
                "wire_contract",
                "suite_preparation_binding_digest",
                "adapter_capabilities",
                "adapter_capabilities_digest",
                "target_kinds",
                "materializer_protocol",
                "trial_count",
                "resource_budgets",
            ),
            "EvalRunConfig",
        )
        if "wire_contract" in payload:
            wire_payload = _object(
                payload["wire_contract"], "run_config.wire_contract"
            )
            for field_name, expected in (
                ("case_schema_version", EVAL_CASE_SCHEMA_VERSION),
                ("input_schema_version", EVAL_INPUT_SCHEMA_VERSION),
                ("submission_schema_version", EVAL_SUBMISSION_SCHEMA_VERSION),
            ):
                if field_name in wire_payload:
                    _required_protocol_version(wire_payload[field_name], expected)
        if "suite" in payload:
            suite_payload = _object(payload["suite"], "run_config.suite")
            if "wire_contract" in suite_payload:
                suite_wire = _object(
                    suite_payload["wire_contract"], "run_config.suite.wire_contract"
                )
                for field_name, expected in (
                    ("case_schema_version", EVAL_CASE_SCHEMA_VERSION),
                    ("input_schema_version", EVAL_INPUT_SCHEMA_VERSION),
                    ("submission_schema_version", EVAL_SUBMISSION_SCHEMA_VERSION),
                ):
                    if field_name in suite_wire:
                        _required_protocol_version(suite_wire[field_name], expected)
        if "adapter_capabilities" in payload:
            capability_payload = _object(
                payload["adapter_capabilities"],
                "run_config.adapter_capabilities",
            )
            if "schema_version" in capability_payload:
                _required_protocol_version(
                    capability_payload["schema_version"],
                    AdapterCapabilitiesV2.SCHEMA_VERSION,
                )
        return cls(
            schema_version=payload["schema_version"],
            run_id=validate_run_id(payload["run_id"]),
            run_instance_key=_identifier(
                payload["run_instance_key"], "run_config.run_instance_key"
            ),
            agent=AgentConfigSnapshot.from_dict(payload["agent"]),
            agent_config_digest=_digest(
                payload["agent_config_digest"], "run_config.agent_config_digest"
            ),
            clarification_matcher=ClarificationMatcherSnapshot.from_dict(
                payload["clarification_matcher"]
            ),
            clarification_matcher_config_digest=_digest(
                payload["clarification_matcher_config_digest"],
                "run_config.clarification_matcher_config_digest",
            ),
            evaluator=EvaluatorRunConfig.from_dict(payload["evaluator"]),
            evaluator_config_digest=_digest(
                payload["evaluator_config_digest"],
                "run_config.evaluator_config_digest",
            ),
            suite=SuiteRunConfig.from_dict(payload["suite"]),
            wire_contract=WireContractV2.from_dict(payload["wire_contract"]),
            suite_preparation_binding_digest=(
                None
                if payload["suite_preparation_binding_digest"] is None
                else _digest(
                    payload["suite_preparation_binding_digest"],
                    "run_config.suite_preparation_binding_digest",
                )
            ),
            adapter_capabilities=AdapterCapabilitiesV2.from_dict(
                payload["adapter_capabilities"]
            ),
            adapter_capabilities_digest=_digest(
                payload["adapter_capabilities_digest"],
                "run_config.adapter_capabilities_digest",
            ),
            target_kinds=tuple(
                _enum_value(
                    ReviewTargetKind,
                    item,
                    "run_config.target_kinds[%d]" % index,
                )
                for index, item in enumerate(
                    _array(
                        payload["target_kinds"],
                        "run_config.target_kinds",
                        len(ReviewTargetKind),
                    )
                )
            ),
            materializer_protocol=_version(
                payload["materializer_protocol"],
                "run_config.materializer_protocol",
            ),
            trial_count=_integer(
                payload["trial_count"],
                "run_config.trial_count",
                minimum=1,
                maximum=MAX_TRIAL_COUNT,
            ),
            resource_budgets=ResourceBudgets.from_dict(
                payload["resource_budgets"]
            ),
        )

    @classmethod
    def from_json(cls, data: Any) -> "EvalRunConfig":
        return cls.from_dict(
            _strict_json_loads(data, MAX_EVAL_RUN_CONFIG_BYTES, "EvalRunConfig JSON")
        )

    def trial_id(self, task_id: str, trial_index: int) -> str:
        self.suite.case(task_id)
        if trial_index > self.trial_count:
            raise SchemaError("trial_index exceeds run_config.trial_count")
        return derive_trial_id(self.run_id, task_id, trial_index)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_instance_key": self.run_instance_key,
            "agent": self.agent.to_dict(),
            "agent_config_digest": self.agent_config_digest,
            "clarification_matcher": self.clarification_matcher.to_dict(),
            "clarification_matcher_config_digest": (
                self.clarification_matcher_config_digest
            ),
            "evaluator": self.evaluator.to_dict(),
            "evaluator_config_digest": self.evaluator_config_digest,
            "suite": self.suite.to_dict(),
            "wire_contract": self.wire_contract.to_dict(),
            "suite_preparation_binding_digest": (
                self.suite_preparation_binding_digest
            ),
            "adapter_capabilities": self.adapter_capabilities.to_dict(),
            "adapter_capabilities_digest": self.adapter_capabilities_digest,
            "target_kinds": [item.value for item in self.target_kinds],
            "materializer_protocol": self.materializer_protocol,
            "trial_count": self.trial_count,
            "resource_budgets": self.resource_budgets.to_dict(),
        }


def derive_trial_id(run_id: str, task_id: str, trial_index: int) -> str:
    validate_run_id(run_id)
    _identifier(task_id, "task_id")
    _integer(
        trial_index,
        "trial_index",
        minimum=1,
        maximum=MAX_TRIAL_COUNT,
    )
    return stable_id("trial", run_id, task_id, trial_index)


def derive_case_path_id(task_id: str) -> str:
    """Map an opaque task ID to the only value permitted in artifact paths."""

    _identifier(task_id, "task_id")
    return stable_id("case", task_id)


def validate_trial_id(
    value: Any, run_id: str, task_id: str, trial_index: int
) -> str:
    result = validate_trial_id_shape(value)
    expected = derive_trial_id(run_id, task_id, trial_index)
    if result != expected:
        raise SchemaError("trial_id does not match the canonical trial identity")
    return result


def derive_trial_seed(run_id: str, task_id: str, trial_index: int) -> int:
    """Return a deterministic non-secret 63-bit seed for a Trial manifest."""

    trial_id = derive_trial_id(run_id, task_id, trial_index)
    digest = canonical_sha256(
        {
            "namespace": "review_agent_eval.trial_seed_v1",
            "trial_id": trial_id,
        }
    )
    return int(digest[:16], 16) & ((1 << 63) - 1)


def derive_evaluation_id(
    run_id: str, evaluator_execution_digest: str, revision: str
) -> str:
    validate_run_id(run_id)
    _digest(evaluator_execution_digest, "evaluator_execution_digest")
    validate_path_segment(revision, "evaluation revision")
    return stable_id("evaluation", run_id, evaluator_execution_digest, revision)


def validate_evaluation_id(
    value: Any,
    run_id: str,
    evaluator_execution_digest: str,
    revision: str,
) -> str:
    result = validate_evaluation_id_shape(value)
    expected = derive_evaluation_id(run_id, evaluator_execution_digest, revision)
    if result != expected:
        raise SchemaError("evaluation_id does not match its canonical identity")
    return result


def load_eval_run_config(data: Any) -> EvalRunConfig:
    return EvalRunConfig.from_json(data)


__all__ = [
    "EVAL_RUN_CONFIG_SCHEMA_VERSION",
    "EVALUATOR_EXECUTION_CONFIG_SCHEMA_VERSION",
    "ADAPTER_CAPABILITIES_SCHEMA_VERSION",
    "JUDGE_PROFILE_SCHEMA_VERSION",
    "JUDGE_EXECUTION_BUDGETS_SCHEMA_VERSION",
    "DEFAULT_JUDGE_CACHE_POLICY_VERSION",
    "DEFAULT_REVIEW_EVALUATOR_CONTEXT_POLICY_VERSION",
    "DEFAULT_METRIC_AUTHORITY_POLICY_VERSION",
    "MAX_EVAL_RUN_CONFIG_BYTES",
    "MAX_PARAMETER_BYTES",
    "MAX_SUITE_CASES",
    "MAX_TRIAL_COUNT",
    "MAX_PLANNED_TRIALS",
    "MAX_JUDGE_ATTEMPTS_PER_REQUEST",
    "MAX_JUDGE_REQUESTS",
    "MAX_JUDGE_CONTEXT_BLOCKS_PER_REQUEST",
    "MAX_JUDGE_REASON_REFS",
    "MAX_JUDGE_TOKEN_BUDGET",
    "ClarificationMatcherSnapshot",
    "AgentConfigSnapshot",
    "JudgeKind",
    "JudgeProfileSnapshot",
    "EvaluatorRunConfig",
    "AdapterCapabilitiesV2",
    "SuiteCase",
    "SuiteRunConfig",
    "ResourceBudgets",
    "JudgeExecutionBudgets",
    "EvaluatorExecutionConfig",
    "EvalRunConfig",
    "derive_trial_id",
    "derive_case_path_id",
    "validate_trial_id",
    "derive_trial_seed",
    "derive_evaluation_id",
    "validate_evaluation_id",
    "validate_run_id",
    "validate_trial_id_shape",
    "validate_evaluation_id_shape",
    "validate_case_path_id",
    "validate_path_segment",
    "validate_safe_json",
    "validate_safe_text",
    "load_eval_run_config",
]

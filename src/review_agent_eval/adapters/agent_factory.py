"""Strict construction boundary for Agent-under-test adapters.

The Eval composition root supplies only an immutable ``AgentConfigSnapshot``.
Judge provider/model/credential settings have no parameter in this API and are
therefore not accidentally forwarded to an Agent adapter.  The concrete
adapter modules are imported lazily after the snapshot has passed the static
schema and namespace checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any, Callable, Protocol

from ..config import AdapterCapabilitiesV2, AgentConfigSnapshot
from ..clarification import (
    ClarificationProtocolError,
    unanswered_clarification_action,
)
from ..models import EVAL_INPUT_SCHEMA_VERSION, EVAL_SUBMISSION_SCHEMA_VERSION
from .base import AgentUnderTestAdapter


CURRENT_AGENT_ADAPTER_KIND = "current-agent-cli-v2"
CURRENT_AGENT_ADAPTER_VERSION = "2"
SUBPROCESS_JSON_ADAPTER_KIND = "subprocess-json-v2"
SUBPROCESS_JSON_ADAPTER_VERSION = "2"
SUBPROCESS_WIRE_VERSION = "subprocess-json-v2"
SUBPROCESS_CLARIFICATION_PROTOCOL = "none-v2"
SUBPROCESS_TRACE_PROTOCOL = "local-trace-v2"
SUBPROCESS_ISOLATION_PROFILE = "target-workspace-v2"

_CURRENT_FIELDS = frozenset(
    {
        "kind",
        "command",
        "review_arguments",
        "environment_allowlist",
        "memory_mode",
    }
)
_SUBPROCESS_FIELDS = frozenset(
    {"kind", "command", "environment_allowlist", "capabilities"}
)
_ENVIRONMENT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_PLACEHOLDER_RE = re.compile(r"\{(agent_id|task_id|trial_id|workspace)\}")
_MAX_ARGUMENTS = 256
_MAX_ARGUMENT_CHARS = 8_192
_MAX_ENVIRONMENT_KEYS = 128
_JUDGE_OPTION_RE = re.compile(
    r"^--(?:eval-)?(?:judge|evaluator)(?:-|$)",
    re.IGNORECASE,
)


class AgentAdapterConfigError(ValueError):
    """The frozen Agent snapshot cannot be used by a supported adapter."""


class AgentAdapterFactory(Protocol):
    """Per-Trial constructor accepted by ``EvalRunner.adapter_factory``."""

    def __call__(self) -> AgentUnderTestAdapter:
        ...


@dataclass(frozen=True)
class SnapshotAgentAdapterFactory:
    """Create a fresh concrete adapter from one validated Agent snapshot."""

    snapshot: AgentConfigSnapshot
    kind: str
    process_runner: Callable[..., Any] | None = None

    def __call__(self) -> AgentUnderTestAdapter:
        # Recheck the immutable shape at the construction boundary.  This is
        # cheap, keeps the factory fail-closed if a test double mutates its
        # internals, and means every worker receives the same validated kind.
        kind = _validate_snapshot(self.snapshot)
        if kind != self.kind:
            raise AgentAdapterConfigError("Agent adapter kind changed")
        if kind == CURRENT_AGENT_ADAPTER_KIND:
            from .current_agent import CurrentAgentAdapter

            if self.process_runner is None:
                return CurrentAgentAdapter()
            return CurrentAgentAdapter(process_runner=self.process_runner)
        if self.process_runner is not None:
            raise AgentAdapterConfigError(
                "process_runner injection is only supported by the current Agent adapter"
            )
        from .subprocess_agent import SubprocessAgentAdapter

        return SubprocessAgentAdapter()


def build_agent_adapter_factory(
    snapshot: AgentConfigSnapshot,
    *,
    process_runner: Callable[..., Any] | None = None,
) -> AgentAdapterFactory:
    """Build a strict per-Trial Agent adapter factory.

    The only provider information accepted here is already frozen inside the
    Agent snapshot.  There are deliberately no ``judge_*`` arguments; Judge
    construction belongs to :mod:`review_agent_eval.adapters.model_adapter`.
    """

    kind = _validate_snapshot(snapshot)
    if process_runner is not None and not callable(process_runner):
        raise TypeError("process_runner must be callable or None")
    if kind != CURRENT_AGENT_ADAPTER_KIND and process_runner is not None:
        raise AgentAdapterConfigError(
            "process_runner injection is only supported by the current Agent adapter"
        )
    return SnapshotAgentAdapterFactory(
        snapshot=snapshot,
        kind=kind,
        process_runner=process_runner,
    )


def build_agent_adapter_from_snapshot(
    snapshot: AgentConfigSnapshot,
    *,
    process_runner: Callable[..., Any] | None = None,
) -> AgentUnderTestAdapter:
    """Construct one Agent adapter; use ``build_agent_adapter_factory`` for Trials."""

    return build_agent_adapter_factory(
        snapshot,
        process_runner=process_runner,
    )()


# Convenient short name for composition roots that do not need to emphasize
# the snapshot binding in their call site.
build_agent_adapter = build_agent_adapter_from_snapshot


def adapter_capabilities_from_snapshot(
    snapshot: AgentConfigSnapshot,
) -> AdapterCapabilitiesV2:
    """Return the capability declaration frozen into an Agent snapshot."""

    kind = _validate_snapshot(snapshot)
    if kind == CURRENT_AGENT_ADAPTER_KIND:
        from .current_agent import current_agent_capabilities

        return current_agent_capabilities()
    raw = snapshot.parameters["adapter"]
    if not isinstance(raw, Mapping):
        raise AgentAdapterConfigError("agent.parameters.adapter is required")
    return _validate_subprocess_capabilities(raw["capabilities"])


def _validate_snapshot(snapshot: AgentConfigSnapshot) -> str:
    if type(snapshot) is not AgentConfigSnapshot:
        raise TypeError("snapshot must be an AgentConfigSnapshot")
    parameters = snapshot.parameters
    if not isinstance(parameters, Mapping):
        raise AgentAdapterConfigError("agent parameters are not a mapping")
    if any(_is_judge_namespace_key(key) for key in parameters):
        raise AgentAdapterConfigError(
            "Judge configuration must not be present in Agent parameters"
        )
    try:
        unanswered_clarification_action(parameters)
    except ClarificationProtocolError as exc:
        raise AgentAdapterConfigError(
            "Agent clarification policy is invalid"
        ) from exc
    raw = parameters.get("adapter")
    if not isinstance(raw, Mapping):
        raise AgentAdapterConfigError("agent.parameters.adapter is required")
    kind = raw.get("kind")
    if kind == CURRENT_AGENT_ADAPTER_KIND:
        expected = _CURRENT_FIELDS
    elif kind == SUBPROCESS_JSON_ADAPTER_KIND:
        expected = _SUBPROCESS_FIELDS
    else:
        raise AgentAdapterConfigError("unsupported Agent adapter kind")
    if set(raw) != expected:
        raise AgentAdapterConfigError("Agent adapter fields do not match its v2 schema")

    _validate_command(raw["command"], kind=kind)
    _validate_environment_allowlist(raw["environment_allowlist"])
    if kind == CURRENT_AGENT_ADAPTER_KIND:
        review_arguments = _validate_argument_array(
            raw["review_arguments"],
            "adapter.review_arguments",
            allow_empty=True,
            require_absolute=False,
            reject_judge_options=True,
        )
        _validate_current_model_identity(snapshot, review_arguments)
        if raw["memory_mode"] not in {"off", "read", "read-write"}:
            raise AgentAdapterConfigError("current Agent memory_mode is invalid")
    else:
        _validate_subprocess_capabilities(raw["capabilities"])
    return kind


def _validate_subprocess_capabilities(value: Any) -> AdapterCapabilitiesV2:
    try:
        capabilities = AdapterCapabilitiesV2.from_dict(
            _plain_json_value(value)
        )
    except (TypeError, ValueError) as exc:
        raise AgentAdapterConfigError(
            "subprocess Agent capabilities are invalid"
        ) from exc
    if (
        capabilities.adapter_id != SUBPROCESS_JSON_ADAPTER_KIND
        or capabilities.adapter_version != SUBPROCESS_JSON_ADAPTER_VERSION
        or capabilities.subprocess_wire_version != SUBPROCESS_WIRE_VERSION
        or capabilities.input_schema_version != EVAL_INPUT_SCHEMA_VERSION
        or capabilities.submission_schema_version
        != EVAL_SUBMISSION_SCHEMA_VERSION
        or capabilities.clarification_protocol
        != SUBPROCESS_CLARIFICATION_PROTOCOL
        or capabilities.trace_protocol != SUBPROCESS_TRACE_PROTOCOL
        or capabilities.isolation_profile != SUBPROCESS_ISOLATION_PROFILE
    ):
        raise AgentAdapterConfigError(
            "subprocess Agent capabilities do not match subprocess-json-v2"
        )
    return capabilities


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_json_value(item) for item in value]
    return value


def _is_judge_namespace_key(value: Any) -> bool:
    if type(value) is not str:
        return True
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return (
        normalized in {"judge", "evaluator"}
        or normalized.startswith("judge_")
        or normalized.startswith("eval_judge_")
        or normalized.startswith("evaluator_")
    )


def _validate_current_model_identity(
    snapshot: AgentConfigSnapshot,
    arguments: tuple[str, ...],
) -> None:
    provider = _single_option_value(arguments, "--reviewer-provider")
    model = _single_option_value(arguments, "--reviewer-model")
    if provider is not None and provider != snapshot.provider:
        raise AgentAdapterConfigError(
            "current Agent provider argument differs from AgentConfigSnapshot"
        )
    if model is not None and model != snapshot.model:
        raise AgentAdapterConfigError(
            "current Agent model argument differs from AgentConfigSnapshot"
        )


def _single_option_value(
    arguments: tuple[str, ...],
    option: str,
) -> str | None:
    values: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == option:
            if index + 1 >= len(arguments):
                raise AgentAdapterConfigError("%s requires a value" % option)
            value = arguments[index + 1]
            if value.startswith("--"):
                raise AgentAdapterConfigError("%s requires a value" % option)
            values.append(value)
            index += 2
            continue
        prefix = option + "="
        if argument.startswith(prefix):
            value = argument[len(prefix) :]
            if not value:
                raise AgentAdapterConfigError("%s requires a value" % option)
            values.append(value)
        index += 1
    if len(values) > 1:
        raise AgentAdapterConfigError("%s may be configured only once" % option)
    return values[0] if values else None


def _validate_command(value: Any, *, kind: str) -> None:
    arguments = _validate_argument_array(
        value,
        "adapter.command",
        allow_empty=False,
        require_absolute=True,
        reject_judge_options=True,
    )
    if kind == SUBPROCESS_JSON_ADAPTER_KIND:
        executable = arguments[0]
        if _PLACEHOLDER_RE.search(executable) is not None:
            raise AgentAdapterConfigError(
                "subprocess Agent executable may not be templated"
            )
        remainder = _PLACEHOLDER_RE.sub("", executable)
        if "{" in remainder or "}" in remainder:
            raise AgentAdapterConfigError(
                "subprocess Agent executable contains an invalid placeholder"
            )
        for argument in arguments:
            remainder = _PLACEHOLDER_RE.sub("", argument)
            if "{" in remainder or "}" in remainder:
                raise AgentAdapterConfigError(
                    "subprocess Agent command contains an invalid placeholder"
                )


def _validate_argument_array(
    value: Any,
    context: str,
    *,
    allow_empty: bool,
    require_absolute: bool,
    reject_judge_options: bool,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AgentAdapterConfigError("%s must be an array" % context)
    if (not value and not allow_empty) or len(value) > _MAX_ARGUMENTS:
        raise AgentAdapterConfigError("%s has an invalid size" % context)
    result: list[str] = []
    for index, item in enumerate(value):
        if type(item) is not str or not item or len(item) > _MAX_ARGUMENT_CHARS:
            raise AgentAdapterConfigError("%s contains an invalid argument" % context)
        if "\x00" in item:
            raise AgentAdapterConfigError("%s contains a null byte" % context)
        if reject_judge_options and _JUDGE_OPTION_RE.match(item.split("=", 1)[0]):
            raise AgentAdapterConfigError(
                "Judge provider options cannot be passed to the Agent adapter"
            )
        if index == 0 and require_absolute:
            from pathlib import Path

            if not Path(item).is_absolute():
                raise AgentAdapterConfigError(
                    "Agent executable must be an absolute path"
                )
        result.append(item)
    return tuple(result)


def _validate_environment_allowlist(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AgentAdapterConfigError("adapter environment allowlist is invalid")
    if len(value) > _MAX_ENVIRONMENT_KEYS:
        raise AgentAdapterConfigError("adapter environment allowlist is too large")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if type(item) is not str or _ENVIRONMENT_KEY_RE.fullmatch(item) is None:
            raise AgentAdapterConfigError("adapter environment key is invalid")
        folded = item.casefold()
        if folded in seen:
            raise AgentAdapterConfigError("adapter environment key is duplicated")
        seen.add(folded)
        result.append(item)
    return tuple(result)


__all__ = [
    "AgentAdapterConfigError",
    "AgentAdapterFactory",
    "CURRENT_AGENT_ADAPTER_KIND",
    "CURRENT_AGENT_ADAPTER_VERSION",
    "SUBPROCESS_JSON_ADAPTER_KIND",
    "SUBPROCESS_JSON_ADAPTER_VERSION",
    "SUBPROCESS_WIRE_VERSION",
    "SnapshotAgentAdapterFactory",
    "adapter_capabilities_from_snapshot",
    "build_agent_adapter",
    "build_agent_adapter_factory",
    "build_agent_adapter_from_snapshot",
]

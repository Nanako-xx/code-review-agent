"""Black-box Agent adapters for the canonical evaluation harness."""

from typing import Any

from .base import (
    AdapterCompatibility,
    AdapterIncompatibilityReason,
    AgentAdapterError,
    AgentAdapterIncompatibleError,
    AgentInputCapability,
    AgentRunConfig,
    AgentUnderTestAdapter,
)

__all__ = [
    "AdapterCompatibility",
    "AdapterIncompatibilityReason",
    "AgentAdapterError",
    "AgentAdapterIncompatibleError",
    "AgentInputCapability",
    "AgentRunConfig",
    "AgentUnderTestAdapter",
    "AdapterConfigError",
    "EVAL_JUDGE_STAGE_LABEL",
    "EvalModelAdapterConfig",
    "ModelAdapter",
    "ModelAdapterCapabilities",
    "ModelAdapterConfig",
    "ModelAdapterFactory",
    "ModelResponseKind",
    "ModelTurnRequest",
    "ModelTurnResponse",
    "build_judge_model_adapter_factory",
    "build_model_adapter_factory_from_config",
    "AgentAdapterConfigError",
    "AgentAdapterFactory",
    "CURRENT_AGENT_ADAPTER_KIND",
    "SnapshotAgentAdapterFactory",
    "build_agent_adapter",
    "build_agent_adapter_factory",
    "build_agent_adapter_from_snapshot",
    "SUBPROCESS_JSON_ADAPTER_KIND",
    "SubprocessAgentAdapter",
]


def __getattr__(name: str) -> Any:
    if name in {"SUBPROCESS_JSON_ADAPTER_KIND", "SubprocessAgentAdapter"}:
        from .subprocess_agent import (
            SUBPROCESS_JSON_ADAPTER_KIND,
            SubprocessAgentAdapter,
        )

        value = {
            "SUBPROCESS_JSON_ADAPTER_KIND": SUBPROCESS_JSON_ADAPTER_KIND,
            "SubprocessAgentAdapter": SubprocessAgentAdapter,
        }[name]
        globals()[name] = value
        return value
    if name in {
        "AdapterConfigError",
        "EVAL_JUDGE_STAGE_LABEL",
        "EvalModelAdapterConfig",
        "ModelAdapter",
        "ModelAdapterCapabilities",
        "ModelAdapterConfig",
        "ModelAdapterFactory",
        "ModelResponseKind",
        "ModelTurnRequest",
        "ModelTurnResponse",
        "build_judge_model_adapter_factory",
        "build_model_adapter_factory_from_config",
    }:
        from . import model_adapter as boundary

        value = getattr(boundary, name)
        globals()[name] = value
        return value
    if name in {
        "AgentAdapterConfigError",
        "AgentAdapterFactory",
        "CURRENT_AGENT_ADAPTER_KIND",
        "SnapshotAgentAdapterFactory",
        "build_agent_adapter",
        "build_agent_adapter_factory",
        "build_agent_adapter_from_snapshot",
    }:
        from . import agent_factory as factory

        value = getattr(factory, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

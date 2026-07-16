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
    raise AttributeError(name)

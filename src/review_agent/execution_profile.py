from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any

from review_agent.completion import COMPLETION_POLICY_VERSION
from review_agent.context import reviewer_protocol_projection
from review_agent.intent_inference import (
    INTENT_INFERENCE_MAX_TOOL_CALLS,
    INTENT_INFERENCE_MAX_TURNS,
    intent_inference_protocol_projection,
)
from review_agent.memory_curator import MEMORY_CURATOR_SYSTEM_PROMPT
from review_agent.model_adapter import (
    DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
    provider_transport_projection,
)
from review_agent.model_risk import RISK_MODEL_SYSTEM_PROMPT
from review_agent.models import ReviewProfile, RiskLevel
from review_agent.portfolio import PORTFOLIO_PLANNER_SYSTEM_PROMPT
from review_agent.reconciler import (
    RECONCILIATION_POLICY_VERSION,
    SEMANTIC_RECONCILER_SYSTEM_PROMPT,
)
from review_agent.review_contract import REVIEW_CONTRACT_VALIDATION_VERSION
from review_agent.session import (
    ReviewExecutionConfig,
    review_execution_config_to_dict,
)
from review_agent.tool_gateway import (
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    tool_gateway_limits_projection,
)


AGENT_EXECUTION_PROFILE_SCHEMA_VERSION = "agent_execution_profile_v1"
PRODUCT_ORCHESTRATION_MARGIN_SECONDS = 300.0


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stage_prompt_digests() -> dict[str, str]:
    return {
        "risk_assessor": _text_sha256(RISK_MODEL_SYSTEM_PROMPT),
        "portfolio_planner": _text_sha256(
            PORTFOLIO_PLANNER_SYSTEM_PROMPT
        ),
        "semantic_reconciler": _text_sha256(
            SEMANTIC_RECONCILER_SYSTEM_PROMPT
        ),
        "memory_curator": _text_sha256(MEMORY_CURATOR_SYSTEM_PROMPT),
    }


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError("execution profile contains a non-JSON value")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _minimum_outer_timeout_seconds(
    execution: ReviewExecutionConfig,
    profiles: Mapping[str, Mapping[str, Any]],
) -> float:
    stage_seconds = sum(
        stage.max_elapsed_seconds
        for stage in (
            execution.risk_assessor,
            execution.portfolio_planner,
            execution.semantic_reconciler,
            execution.memory_curator,
        )
        if stage.mode == "model"
    )
    initial_review_seconds = (
        max(
            float(profile["max_elapsed_seconds"])
            * (
                int(profile["reviewer_count"])
                if execution.reviewer_mode == "single"
                else 1
            )
            for profile in profiles.values()
        )
        if execution.reviewer_provider != "none"
        else 0.0
    )
    intent_inference_seconds = (
        INTENT_INFERENCE_MAX_TURNS
        * DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
        if execution.reviewer_provider == "openai-compatible"
        else 0.0
    )
    intent_tool_seconds = (
        INTENT_INFERENCE_MAX_TOOL_CALLS * DEFAULT_TOOL_TIMEOUT_SECONDS
        if execution.reviewer_provider != "none"
        else 0.0
    )
    return float(
        stage_seconds
        + intent_inference_seconds
        + intent_tool_seconds
        + initial_review_seconds
        + execution.supplemental_policy.max_elapsed_seconds
        + PRODUCT_ORCHESTRATION_MARGIN_SECONDS
    )


@dataclass(frozen=True)
class AgentExecutionProfile:
    payload: Mapping[str, Any]

    @classmethod
    def from_execution(
        cls,
        execution: ReviewExecutionConfig,
    ) -> "AgentExecutionProfile":
        if not isinstance(execution, ReviewExecutionConfig):
            raise TypeError("execution must be ReviewExecutionConfig")
        execution_payload = review_execution_config_to_dict(execution)
        memory = execution_payload["memory"]
        if memory is not None:
            memory = dict(memory)
            memory.pop("root_path")
            memory["root_binding"] = "trial_private"
            execution_payload["memory"] = memory
        profiles = {
            risk.value: asdict(ReviewProfile.for_risk(risk))
            for risk in RiskLevel
        }
        return cls(
            _freeze_json(
                {
                    "schema_version": AGENT_EXECUTION_PROFILE_SCHEMA_VERSION,
                    "execution": execution_payload,
                    "risk_profiles": profiles,
                    "reviewer_protocol": reviewer_protocol_projection(),
                    "intent_protocol": intent_inference_protocol_projection(),
                    "tool_gateway_limits": tool_gateway_limits_projection(),
                    "provider_transport": provider_transport_projection(),
                    "stage_prompt_sha256": stage_prompt_digests(),
                    "review_contract_version": (
                        REVIEW_CONTRACT_VALIDATION_VERSION
                    ),
                    "reconciliation_policy_version": (
                        RECONCILIATION_POLICY_VERSION
                    ),
                    "completion_policy_version": COMPLETION_POLICY_VERSION,
                    "minimum_outer_timeout_seconds": (
                        _minimum_outer_timeout_seconds(execution, profiles)
                    ),
                    "capabilities": {
                        "shell": "unavailable",
                        "network": "provider_only",
                        "repository": "read_only",
                        "run_safe_check": "unavailable",
                    },
                }
            )
        )

    @classmethod
    def from_dict(cls, value: Any) -> "AgentExecutionProfile":
        if not isinstance(value, Mapping):
            raise ValueError("execution profile must be a JSON object")
        expected = {
            "schema_version",
            "execution",
            "risk_profiles",
            "reviewer_protocol",
            "intent_protocol",
            "tool_gateway_limits",
            "provider_transport",
            "stage_prompt_sha256",
            "review_contract_version",
            "reconciliation_policy_version",
            "completion_policy_version",
            "minimum_outer_timeout_seconds",
            "capabilities",
        }
        if set(value) != expected:
            raise ValueError("execution profile fields are not canonical")
        if value["schema_version"] != AGENT_EXECUTION_PROFILE_SCHEMA_VERSION:
            raise ValueError("execution profile schema is unsupported")
        return cls(_freeze_json(value))

    def to_dict(self) -> dict[str, Any]:
        return _thaw_json(self.payload)

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

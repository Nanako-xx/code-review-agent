from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any

from review_agent.aggregation import AGGREGATION_RECORD_SCHEMA
from review_agent.context import REVIEWER_TOOL_NAMES_V2, reviewer_tool_schemas_v2
from review_agent.context_window import (
    COMPACTION_SYSTEM_PROMPT,
    COMPACTION_USER_PROMPT,
    ContextWindowPolicy,
)
from review_agent.developer_rules import load_builtin_developer_rule_catalog
from review_agent.model_adapter import (
    DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
    provider_transport_projection,
)
from review_agent.model_adapter_factory import ModelAdapterConfig
from review_agent.model_risk import RISK_MODEL_SYSTEM_PROMPT_V2
from review_agent.review_context import DiffFitPolicy
from review_agent.review_pipeline import REVIEWER_EXECUTION_RECORD_SCHEMA
from review_agent.review_planning import fixed_reviewer_slots
from review_agent.review_policy import (
    DEFAULT_DEVELOPER_REVIEW_POLICY,
    DeveloperReviewPolicy,
    build_reviewer_system_prompt,
)
from review_agent.review_protocol import RiskLevel
from review_agent.reviewer_output import REVIEWER_OUTPUT_JSON_SCHEMA_V2
from review_agent.reviewer_runtime import ReviewerRuntimeLimitsV2
from review_agent.session import SESSION_V6_PHASES, SESSION_V6_SCHEMA_VERSION
from review_agent.tool_artifacts import (
    MAX_ARTIFACT_PAGE_CHARS,
    ToolResultLimits,
)


AGENT_EXECUTION_PROFILE_SCHEMA_VERSION = "agent_execution_profile_v2"
REVIEWER_EXECUTION_PROFILE_V2_SCHEMA = "reviewer_execution_profile_v2"
PRODUCT_ORCHESTRATION_MARGIN_SECONDS = 300.0


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def reviewer_execution_profile_v2(
    policy: DeveloperReviewPolicy,
    *,
    diff_fit_policy: DiffFitPolicy | None = None,
) -> dict[str, Any]:
    if not isinstance(policy, DeveloperReviewPolicy):
        raise TypeError("policy must be DeveloperReviewPolicy")
    fit = diff_fit_policy or DiffFitPolicy()
    if not isinstance(fit, DiffFitPolicy):
        raise TypeError("diff_fit_policy must be DiffFitPolicy")
    system = build_reviewer_system_prompt(policy)
    tools = reviewer_tool_schemas_v2(REVIEWER_TOOL_NAMES_V2)
    runtime = ReviewerRuntimeLimitsV2()
    rules = load_builtin_developer_rule_catalog()
    return {
        "schema_version": REVIEWER_EXECUTION_PROFILE_V2_SCHEMA,
        "invocation_inputs": ["system", "tools", "messages", "parameters"],
        "developer_policy_sha256": policy.digest(),
        "developer_rule_catalog_sha256": rules.digest,
        "developer_rule_resolver_version": rules.resolver_version,
        "reviewer_system_prompt_sha256": _text_sha256(system),
        "tool_catalog_sha256": _canonical_sha256(list(tools)),
        "tool_names": list(REVIEWER_TOOL_NAMES_V2),
        "diff_fit_policy": fit.to_dict(),
        "runtime_limits": {
            "max_elapsed_seconds": runtime.max_elapsed_seconds,
            "max_provider_attempts": runtime.max_provider_attempts,
            "tool_timeout_seconds": runtime.tool_timeout_seconds,
        },
        "invocation_defaults": {
            "reasoning_effort": "medium",
            "temperature": 0,
            "tool_choice_policy": "auto_if_tools_else_none",
            "response_schema": "reviewer_output_v2",
        },
    }


def _adapter_projection(config: ModelAdapterConfig) -> dict[str, Any]:
    if not isinstance(config, ModelAdapterConfig):
        raise TypeError("adapter configuration must be ModelAdapterConfig")
    return {
        "provider": config.provider_name or "none",
        "model": config.model,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "timeout_seconds": config.timeout_seconds,
        "max_response_bytes": config.max_response_bytes,
    }


def _slot_projection() -> dict[str, list[dict[str, str]]]:
    return {
        level.value: [
            {
                "slot_id": slot.slot_id,
                "role": slot.role,
                "role_kind": slot.role_kind.value,
            }
            for slot in fixed_reviewer_slots(level)
        ]
        for level in RiskLevel
    }


def _product_protocol_projection(
    policy: DeveloperReviewPolicy,
) -> dict[str, Any]:
    rules = load_builtin_developer_rule_catalog()
    return {
        "session": {
            "schema_version": SESSION_V6_SCHEMA_VERSION,
            "phases": [phase.value for phase in SESSION_V6_PHASES],
        },
        "diff_artifact": {
            "patch_schema": "diff_artifact_patch_v1",
            "index_schema": "diff_artifact_index_v1",
            "full_patch_persisted": True,
            "truncated_excerpt_fields": [],
        },
        "intent": {
            "schema": "intent_packet_v2_minimal",
            "fields": ["goal", "source", "uncertainties"],
            "sources": ["explicit", "inferred", None],
        },
        "risk": {
            "schema": "risk_decision_v2",
            "fields": ["level"],
            "levels": [level.value for level in RiskLevel],
            "model_prompt_sha256": _text_sha256(RISK_MODEL_SYSTEM_PROMPT_V2),
            "runtime_merge": "max_deterministic_and_model",
        },
        "review_planning": {
            "slot_policy": "fixed_by_final_risk_v1",
            "slots": _slot_projection(),
        },
        "reviewer_output": {
            "schema": "reviewer_output_v2",
            "top_level_fields": ["findings", "uncertainties"],
            "finding_fields": [
                "claim",
                "severity",
                "path",
                "line",
                "suggestion",
            ],
            "json_schema_sha256": _canonical_sha256(
                REVIEWER_OUTPUT_JSON_SCHEMA_V2
            ),
        },
        "aggregation": {
            "record_schema": AGGREGATION_RECORD_SCHEMA,
            "reviewer_record_schema": REVIEWER_EXECUTION_RECORD_SCHEMA,
            "review_result_schema": "review_result_v1",
            "review_result_fields": [
                "pr_id",
                "snapshot_id",
                "status",
                "risk_level",
                "findings",
                "uncertainties",
            ],
            "merge_policy": "exact_normalized_issue_identity_v1",
            "model_calls": 0,
        },
        "developer_policy_sha256": policy.digest(),
        "developer_rule_catalog_sha256": rules.digest,
        "developer_rule_resolver_version": rules.resolver_version,
    }


def _minimum_outer_timeout_seconds(
    reviewer: ModelAdapterConfig,
    risk: ModelAdapterConfig | None,
) -> float:
    reviewer_seconds = (
        0.0
        if (reviewer.provider_name or "none") == "none"
        else ReviewerRuntimeLimitsV2().max_elapsed_seconds
    )
    risk_seconds = 0.0
    if risk is not None and (risk.provider_name or "none") != "none":
        attempt_seconds = (
            DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
            if risk.timeout_seconds is None
            else float(risk.timeout_seconds)
        )
        risk_seconds = 3.0 * attempt_seconds
    return reviewer_seconds + risk_seconds + PRODUCT_ORCHESTRATION_MARGIN_SECONDS


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


@dataclass(frozen=True)
class AgentExecutionProfile:
    payload: Mapping[str, Any]

    @classmethod
    def from_product_configuration(
        cls,
        *,
        reviewer: ModelAdapterConfig,
        risk: ModelAdapterConfig | None,
        policy: DeveloperReviewPolicy = DEFAULT_DEVELOPER_REVIEW_POLICY,
    ) -> "AgentExecutionProfile":
        if not isinstance(reviewer, ModelAdapterConfig):
            raise TypeError("reviewer must be ModelAdapterConfig")
        if risk is not None and not isinstance(risk, ModelAdapterConfig):
            raise TypeError("risk must be ModelAdapterConfig or null")
        if not isinstance(policy, DeveloperReviewPolicy):
            raise TypeError("policy must be DeveloperReviewPolicy")
        tool_limits = ToolResultLimits()
        context_policy = ContextWindowPolicy()
        return cls(
            _freeze_json(
                {
                    "schema_version": AGENT_EXECUTION_PROFILE_SCHEMA_VERSION,
                    "configuration": {
                        "reviewer": _adapter_projection(reviewer),
                        "risk": (
                            None if risk is None else _adapter_projection(risk)
                        ),
                    },
                    "product_protocol": _product_protocol_projection(policy),
                    "reviewer_execution": reviewer_execution_profile_v2(policy),
                    "tool_result_policy": {
                        **asdict(tool_limits),
                        "max_artifact_page_chars": MAX_ARTIFACT_PAGE_CHARS,
                    },
                    "context_window_policy": {
                        **asdict(context_policy),
                        "compaction_system_prompt_sha256": _text_sha256(
                            COMPACTION_SYSTEM_PROMPT
                        ),
                        "compaction_user_prompt_sha256": _text_sha256(
                            COMPACTION_USER_PROMPT
                        ),
                    },
                    "provider_transport": provider_transport_projection(),
                    "minimum_outer_timeout_seconds": _minimum_outer_timeout_seconds(
                        reviewer,
                        risk,
                    ),
                    "capabilities": {
                        "shell": "unavailable",
                        "network": "provider_only",
                        "repository": "read_only",
                        "edit": "unavailable",
                        "write": "unavailable",
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
            "configuration",
            "product_protocol",
            "reviewer_execution",
            "tool_result_policy",
            "context_window_policy",
            "provider_transport",
            "minimum_outer_timeout_seconds",
            "capabilities",
        }
        if set(value) != expected:
            raise ValueError("execution profile fields are not canonical")
        if value["schema_version"] != AGENT_EXECUTION_PROFILE_SCHEMA_VERSION:
            raise ValueError("execution profile schema is unsupported")
        profile = cls(_freeze_json(value))
        if profile.to_dict() != _thaw_json(value):
            raise ValueError("execution profile is not canonical JSON data")
        return profile

    def to_dict(self) -> dict[str, Any]:
        return _thaw_json(self.payload)

    def digest(self) -> str:
        return _canonical_sha256(self.to_dict())


__all__ = [
    "AGENT_EXECUTION_PROFILE_SCHEMA_VERSION",
    "AgentExecutionProfile",
    "PRODUCT_ORCHESTRATION_MARGIN_SECONDS",
    "REVIEWER_EXECUTION_PROFILE_V2_SCHEMA",
    "reviewer_execution_profile_v2",
]

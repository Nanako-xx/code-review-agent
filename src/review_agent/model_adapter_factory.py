from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Protocol

from review_agent.model_adapter import (
    DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
    DEFAULT_MAX_RESPONSE_BYTES,
    MAX_ALLOWED_RESPONSE_BYTES,
    FakeToolCallingAdapter,
    ModelAdapter,
    OpenAICompatibleConfig,
    OpenAICompatibleToolAdapter,
)
from review_agent.model_protocol import ModelResponseKind, ModelTurnResponse
from review_agent.model_protocol import ModelToolCall, ModelTurnRequest


class AdapterConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelAdapterConfig:
    provider_name: str | None
    model: str | None
    base_url: str | None
    api_key_env: str
    stage_label: str = "reviewer"
    # These optional fields are runtime-only adapter budgets.  ``None`` keeps
    # the historical OpenAI-compatible defaults and therefore preserves the
    # product callers' existing behavior.  The Eval boundary supplies the
    # Judge budgets explicitly; the API key is intentionally not a field here.
    timeout_seconds: int | float | None = None
    max_response_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        if self.max_response_bytes is not None and (
            type(self.max_response_bytes) is not int
            or self.max_response_bytes < 1
            or self.max_response_bytes > MAX_ALLOWED_RESPONSE_BYTES
        ):
            raise ValueError(
                "max_response_bytes must be a positive integer no greater than "
                f"{MAX_ALLOWED_RESPONSE_BYTES}"
            )


class ModelAdapterFactory(Protocol):
    def create(self) -> ModelAdapter:
        raise NotImplementedError


@dataclass(frozen=True)
class FakeModelAdapterFactory:
    def create(self) -> ModelAdapter:
        return _factory_fake_adapter()


@dataclass(frozen=True)
class OpenAICompatibleModelAdapterFactory:
    config: OpenAICompatibleConfig

    def create(self) -> ModelAdapter:
        return OpenAICompatibleToolAdapter(self.config)


class _FactoryFakeToolCallingAdapter(FakeToolCallingAdapter):
    provider_name = "fake"


def build_model_adapter_factory_from_config(
    config: ModelAdapterConfig,
    *,
    stage_label: str | None = None,
    timeout_seconds: int | float | None = None,
    max_response_bytes: int | None = None,
) -> ModelAdapterFactory | None:
    if not isinstance(config, ModelAdapterConfig):
        raise TypeError("config must be a ModelAdapterConfig")
    stage_label = config.stage_label if stage_label is None else stage_label
    option_prefix = _option_prefix(stage_label)
    provider_name = config.provider_name or "none"
    if provider_name == "none":
        return None
    if provider_name == "fake":
        return FakeModelAdapterFactory()
    if provider_name == "openai-compatible":
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            stage_prefix = "" if option_prefix == "reviewer" else f"{option_prefix} "
            raise AdapterConfigError(
                f"missing {stage_prefix}API key environment variable: "
                f"{config.api_key_env}"
            )
        if not config.model:
            raise AdapterConfigError(
                f"--{option_prefix}-model is required for openai-compatible provider"
            )
        if not config.base_url:
            raise AdapterConfigError(
                f"--{option_prefix}-base-url is required for openai-compatible provider"
            )
        configured_timeout = (
            config.timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        configured_response_limit = (
            config.max_response_bytes
            if max_response_bytes is None
            else max_response_bytes
        )
        # Let OpenAICompatibleConfig perform the canonical finite/positive
        # validation, but use the product defaults when no Eval budget was
        # supplied.  This is deliberately resolved only after the provider
        # and credential checks, so old callers retain their error ordering.
        if configured_timeout is None:
            configured_timeout = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
        if configured_response_limit is None:
            configured_response_limit = DEFAULT_MAX_RESPONSE_BYTES
        return OpenAICompatibleModelAdapterFactory(
            OpenAICompatibleConfig(
                base_url=config.base_url,
                api_key=api_key,
                model=config.model,
                timeout_seconds=configured_timeout,
                max_response_bytes=configured_response_limit,
            )
        )
    raise AdapterConfigError(f"unsupported {option_prefix} provider: {provider_name}")


def _option_prefix(stage_label: str) -> str:
    if not isinstance(stage_label, str) or not stage_label:
        raise ValueError("stage_label must be a non-empty string")
    option_prefix = stage_label.replace("_", "-")
    if any(character.isspace() for character in option_prefix):
        raise ValueError("stage_label must not contain whitespace")
    return option_prefix


def _factory_fake_adapter() -> FakeToolCallingAdapter:
    return _FactoryFakeToolCallingAdapter(
        script=[
            _fake_response_for_request,
            _fake_response_for_request,
        ]
    )


def _fake_response_for_request(request: ModelTurnRequest) -> ModelTurnResponse:
    response_schema = request.parameters.get("response_schema")
    if response_schema == "intent_inference_result_v1":
        return _fake_intent_inference_response()
    if response_schema == "risk_proposal_v1":
        return _fake_risk_proposal_response()
    if response_schema == "risk_decision_v2":
        return ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text='{"level":"low"}',
            provider_name="fake",
            model="fake-risk-assessor-v2",
            raw={"fake": True, "response_schema": "risk_decision_v2"},
        )
    if response_schema == "portfolio_proposal_v1":
        return _fake_portfolio_proposal_response()
    if response_schema == "semantic_reconciliation_proposal_v1":
        return _fake_semantic_reconciliation_response(request)
    if response_schema == "memory_curator_proposal_v1":
        return _fake_memory_curator_response(request)
    if response_schema == "reviewer_output_v2":
        return ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text=json.dumps(
                {
                    "findings": [],
                    "uncertainties": [
                        "Fake provider does not perform semantic review."
                    ],
                },
                separators=(",", ":"),
            ),
            provider_name="fake",
            model="fake-reviewer-v2",
            raw={"fake": True, "response_schema": "reviewer_output_v2"},
        )
    if not request.tools or request.parameters.get("tool_choice") == "none":
        return _fake_single_shot_response()

    observation_id = _latest_observation_id(request)
    if observation_id:
        return _fake_completed_agent_loop_response(observation_id, request)

    changed_file = _first_changed_file(request)
    if changed_file:
        return ModelTurnResponse(
            kind=ModelResponseKind.TOOL_CALLS,
            tool_calls=[ModelToolCall("call-1", "compare_base_head", {"path": changed_file})],
            provider_name="fake",
            model="fake-reviewer",
        )

    return _fake_completed_agent_loop_response("", request)


def _fake_intent_inference_response() -> ModelTurnResponse:
    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=json.dumps(
            {
                "candidates": [
                    {
                        "field": "goal",
                        "value": "Review the behavior changed between the resolved base and head revisions.",
                        "origin": "llm_inference",
                        "confidence": "low",
                        "source_refs": [],
                        "evidence_refs": [],
                        "rationale": "The fake provider exercises intent inference without claiming repository evidence.",
                        "conclusion_impact": "material",
                    }
                ],
                "uncertainties": [
                    "Fake provider does not perform semantic intent analysis."
                ],
                "summary": "Fake intent inference executed.",
            }
        ),
        provider_name="fake",
        model="fake-intent-analyst",
        raw={"fake": True, "response_schema": "intent_inference_result_v1"},
    )


def _fake_risk_proposal_response() -> ModelTurnResponse:
    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=json.dumps(
            {
                "level": "medium",
                "dimensions": {
                    "impact": "Fake provider identified bounded behavioral impact.",
                    "blast_radius": "Fake provider assumes a localized blast radius.",
                    "reversibility": "The reviewed revision can be reverted.",
                    "uncertainty": "Fake provider does not perform semantic risk analysis.",
                    "verification_strength": "Runtime-owned checks remain authoritative.",
                },
                "reasons": [
                    "Fake provider exercises the model-assisted risk path."
                ],
                "signal_refs": [],
                "uncertainties": [
                    "Fake provider does not perform semantic risk analysis."
                ],
                "suggested_focus": [
                    "Verify changed behavior against the review contract."
                ],
            }
        ),
        provider_name="fake",
        model="fake-risk-assessor",
        raw={"fake": True, "response_schema": "risk_proposal_v1"},
    )


def _fake_portfolio_proposal_response() -> ModelTurnResponse:
    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "fake-core",
                        "role_kind": "core",
                        "role_name": "Fake Core Reviewer",
                        "perspective_key": "fake_core",
                        "mission": "Exercise the model-assisted portfolio path.",
                        "reason_refs": [],
                        "context_refs": [],
                        "extra_contract": [],
                        "required_checks": [
                            "Verify the Runtime-compiled review contract."
                        ],
                        "priority": 50,
                    }
                ],
                "summary": "Fake portfolio proposal executed.",
                "uncertainties": [
                    "Fake provider does not select repository-specific specialists."
                ],
            }
        ),
        provider_name="fake",
        model="fake-portfolio-planner",
        raw={"fake": True, "response_schema": "portfolio_proposal_v1"},
    )


def _fake_semantic_reconciliation_response(
    request: ModelTurnRequest,
) -> ModelTurnResponse:
    packet: dict[str, object] = {}
    for message in request.messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            candidate = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(
            candidate.get("candidate_catalog"),
            dict,
        ):
            packet = candidate
            break
    catalog = packet.get("candidate_catalog", {})
    catalog = catalog if isinstance(catalog, dict) else {}
    groups = []
    for candidate_id in sorted(catalog):
        row = catalog[candidate_id]
        if not isinstance(row, dict):
            continue
        refs = row.get("evidence_refs", [])
        refs = refs if isinstance(refs, list) else []
        groups.append(
            {
                "member_ids": [candidate_id],
                "representative_id": candidate_id,
                "canonical_claim": str(row.get("claim", "Verified finding")),
                "rationale": "Fake provider preserves each Runtime candidate independently.",
                "supporting_refs": refs,
                "proposed_confidence": str(row.get("confidence", "low")),
            }
        )
    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=json.dumps(
            {
                "canonical_groups": groups,
                "rejections": [],
                "disagreements": [],
                "supplemental_requests": [],
                "uncertainties": [],
                "summary": "Fake semantic reconciliation preserved Runtime candidates.",
            }
        ),
        provider_name="fake",
        model="fake-semantic-reconciler",
        raw={
            "fake": True,
            "response_schema": "semantic_reconciliation_proposal_v1",
        },
    )


def _fake_memory_curator_response(
    request: ModelTurnRequest,
) -> ModelTurnResponse:
    envelope: dict[str, object] = {}
    for message in request.messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            candidate = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(
            candidate.get("source_ref_allowlist"),
            list,
        ):
            envelope = candidate
            break

    source_ref_ids = []
    source_catalog = envelope.get("source_ref_allowlist", [])
    if isinstance(source_catalog, list):
        for item in source_catalog:
            if not isinstance(item, dict):
                continue
            source_ref_id = item.get("source_ref_id")
            if isinstance(source_ref_id, str) and source_ref_id:
                source_ref_ids.append(source_ref_id)
                break

    candidates = []
    if source_ref_ids:
        candidates.append(
            {
                "candidate_id": "fake-memory-candidate-1",
                "kind": "review_rule",
                "statement": (
                    "Preserve verified project behavior represented by the "
                    "authorized source."
                ),
                "scope": {
                    "schema_version": 1,
                    "paths": [],
                    "symbols": [],
                    "contracts": [],
                    "languages": [],
                },
                "source_ref_ids": source_ref_ids,
                "validity_policies": ["source_content_hash"],
                "confidence": "low",
                "sensitivity": "normal",
                "policy_effect_id": None,
            }
        )
    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=json.dumps(
            {
                "schema_version": 1,
                "candidates": candidates,
            }
        ),
        provider_name="fake",
        model="fake-memory-curator",
        raw={
            "fake": True,
            "response_schema": "memory_curator_proposal_v1",
        },
    )


def _fake_single_shot_response() -> ModelTurnResponse:
    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=json.dumps(
            {
                "contract_assessments": [],
                "confirmed_findings": [],
                "rejected_hypotheses": [],
                "uncertainties": ["Fake provider does not perform semantic review."],
                "observation_refs": [],
                "investigation_summary": "Fake reviewer executed.",
                "status": "partial",
            }
        ),
        provider_name="fake",
        model="fake-reviewer",
        raw={"fake": True},
    )


def _fake_completed_agent_loop_response(
    observation_id: str,
    request: ModelTurnRequest,
) -> ModelTurnResponse:
    evidence_refs = [observation_id] if observation_id else []
    contract_assessments = [
        {
            "contract": contract,
            "status": "covered",
            "summary": "Fake agent loop exercised the configured Runtime path.",
            "evidence_refs": evidence_refs,
        }
        for contract in _assigned_contracts(request)
    ]
    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=json.dumps(
            {
                "contract_assessments": contract_assessments,
                "confirmed_findings": [],
                "rejected_hypotheses": [],
                "uncertainties": [],
                "observation_refs": evidence_refs,
                "investigation_summary": "Fake agent loop reviewer executed.",
                "status": "completed",
            }
        ),
        provider_name="fake",
        model="fake-reviewer",
        raw={"fake": True},
    )


def _assigned_contracts(request: ModelTurnRequest) -> list[str]:
    prefix = "Assigned Contract:"
    for message in request.messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        for line in content.splitlines():
            if line.startswith(prefix):
                return [
                    item.strip()
                    for item in line.removeprefix(prefix).split(",")
                    if item.strip()
                ]
    return ["regression_safety"]


def _latest_observation_id(request: ModelTurnRequest) -> str:
    for result in reversed(request.tool_results):
        if result.observation_ids:
            return result.observation_ids[-1]
    return ""


def _first_changed_file(request: ModelTurnRequest) -> str:
    prefix = "Changed Files:"
    for message in request.messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        for line in content.splitlines():
            if not line.startswith(prefix):
                continue
            changed_files = line.removeprefix(prefix).strip()
            for changed_file in changed_files.split(","):
                changed_file = changed_file.strip()
                if changed_file:
                    return changed_file
    return ""

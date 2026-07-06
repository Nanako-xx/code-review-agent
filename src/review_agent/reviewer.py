from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

from review_agent.context import build_reviewer_envelope
from review_agent.model_adapter import ModelAdapter
from review_agent.model_protocol import ModelResponse, ModelResponseKind, ModelTurnRequest
from review_agent.models import (
    Assignment,
    ContractAssessment,
    ContractItemStatus,
    IntentPacket,
    ModelInvocationEnvelope,
    ReviewerFinding,
    ReviewerResult,
    ReviewerResultStatus,
)


class ReviewerResultParseError(ValueError):
    pass


@dataclass(frozen=True)
class ReviewerRun:
    envelope: ModelInvocationEnvelope
    response: ModelResponse
    result: ReviewerResult


REQUIRED_RESULT_KEYS = (
    "contract_assessments",
    "confirmed_findings",
    "rejected_hypotheses",
    "uncertainties",
    "observation_refs",
    "investigation_summary",
    "status",
)

LIST_RESULT_KEYS = (
    "contract_assessments",
    "confirmed_findings",
    "rejected_hypotheses",
    "uncertainties",
    "observation_refs",
)


def parse_reviewer_result(raw_text: str) -> ReviewerResult:
    payload = _loads_json_object(_strip_json_fence(raw_text))
    for key in REQUIRED_RESULT_KEYS:
        if key not in payload:
            raise ReviewerResultParseError(f"missing required key: {key}")
    for key in LIST_RESULT_KEYS:
        _require_list(payload, key)

    try:
        status = ReviewerResultStatus(payload["status"])
    except ValueError as error:
        raise ReviewerResultParseError(f"invalid reviewer status: {payload['status']}") from error

    return ReviewerResult(
        contract_assessments=[_parse_contract_assessment(item) for item in payload["contract_assessments"]],
        confirmed_findings=[_parse_finding(item) for item in payload["confirmed_findings"]],
        rejected_hypotheses=[str(item) for item in payload["rejected_hypotheses"]],
        uncertainties=[str(item) for item in payload["uncertainties"]],
        observation_refs=[str(item) for item in payload["observation_refs"]],
        investigation_summary=str(payload["investigation_summary"]),
        status=status,
    )


def reviewer_result_to_dict(result: ReviewerResult) -> dict[str, Any]:
    return asdict(result)


def run_single_reviewer(
    adapter: ModelAdapter,
    assignment: Assignment,
    intent: IntentPacket,
    diff_excerpt: list[str],
    observations: dict[str, str],
    trace_id: str,
) -> ReviewerRun:
    envelope = build_reviewer_envelope(
        assignment=assignment,
        intent=intent,
        code_snippets={"Diff Excerpt": "\n".join(diff_excerpt)},
        observations=observations,
        trace_id=trace_id,
    )
    request = ModelTurnRequest(
        system=envelope.system,
        tools=[],
        messages=[dict(message) for message in envelope.messages],
        tool_results=[],
        parameters={**dict(envelope.parameters), "tool_choice": "none"},
    )
    turn_response = adapter.complete_turn(request)
    response = ModelResponse(
        content=turn_response.final_text or turn_response.error or "",
        provider_name=turn_response.provider_name,
        model=turn_response.model,
        raw=turn_response.raw,
    )
    if turn_response.kind is ModelResponseKind.FINAL:
        try:
            result = parse_reviewer_result(turn_response.final_text or "")
        except ReviewerResultParseError as error:
            message = f"single-shot final response parse failed: {error}"
            result = ReviewerResult(
                uncertainties=[message],
                investigation_summary=message,
                status=ReviewerResultStatus.FAILED,
            )
        return ReviewerRun(envelope=envelope, response=response, result=result)
    if turn_response.kind is ModelResponseKind.TOOL_CALLS:
        message = "single-shot reviewer received tool calls; use --reviewer-loop agent-loop to enable tools"
    else:
        message = turn_response.error or f"single-shot reviewer received invalid response kind: {turn_response.kind.value}"
    result = ReviewerResult(
        uncertainties=[message],
        investigation_summary=message,
        status=ReviewerResultStatus.FAILED,
    )
    return ReviewerRun(envelope=envelope, response=response, result=result)


def _strip_json_fence(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()
    return text


def _loads_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ReviewerResultParseError(f"invalid reviewer JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ReviewerResultParseError("reviewer JSON must be an object")
    return payload


def _require_list(container: dict[str, Any], key: str, *, field_label: str | None = None) -> list[Any]:
    value = container[key]
    if not isinstance(value, list):
        raise ReviewerResultParseError(f"{field_label or key} must be a list")
    return value


def _optional_list(container: dict[str, Any], key: str, *, field_label: str | None = None) -> list[Any]:
    value = container.get(key, [])
    if not isinstance(value, list):
        raise ReviewerResultParseError(f"{field_label or key} must be a list")
    return value


def _parse_contract_assessment(item: Any) -> ContractAssessment:
    if not isinstance(item, dict):
        raise ReviewerResultParseError("contract assessment must be an object")
    try:
        status = ContractItemStatus(item["status"])
    except KeyError as error:
        raise ReviewerResultParseError("contract assessment missing required key: status") from error
    except ValueError as error:
        raise ReviewerResultParseError(f"invalid contract status: {item.get('status')}") from error
    evidence_refs = _optional_list(item, "evidence_refs", field_label="contract assessment evidence_refs")
    return ContractAssessment(
        contract=str(item.get("contract", "")),
        status=status,
        summary=str(item.get("summary", "")),
        evidence_refs=[str(ref) for ref in evidence_refs],
    )


def _parse_finding(item: Any) -> ReviewerFinding:
    if not isinstance(item, dict):
        raise ReviewerResultParseError("finding must be an object")
    suggested_action = item.get("suggested_action")
    evidence_refs = _optional_list(item, "evidence_refs", field_label="finding evidence_refs")
    return ReviewerFinding(
        claim=str(item.get("claim", "")),
        severity=str(item.get("severity", "")),
        confidence=str(item.get("confidence", "")),
        evidence_refs=[str(ref) for ref in evidence_refs],
        suggested_action=str(suggested_action) if suggested_action is not None else None,
    )

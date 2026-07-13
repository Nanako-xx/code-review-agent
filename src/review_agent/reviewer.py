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
    ReviewerRuntimeMetadata,
    ReviewerTerminationReason,
)
from review_agent.review_contract import (
    finding_path_error,
    result_with_validation_deficiencies,
    validate_reviewer_completion,
)
from review_agent.reviewer_runtime import (
    RuntimeTracker,
    budget_reason_after_call,
    budget_reason_before_call,
    request_parameters,
    termination_reason_for_result,
    termination_summary,
)


class ReviewerResultParseError(ValueError):
    pass


@dataclass(frozen=True)
class ReviewerRun:
    envelope: ModelInvocationEnvelope
    response: ModelResponse
    result: ReviewerResult
    runtime: ReviewerRuntimeMetadata


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
    _require_exact_keys(payload, set(REQUIRED_RESULT_KEYS), "reviewer result")
    for key in LIST_RESULT_KEYS:
        _require_list(payload, key)

    try:
        status = ReviewerResultStatus(payload["status"])
    except ValueError as error:
        raise ReviewerResultParseError(f"invalid reviewer status: {payload['status']}") from error

    return ReviewerResult(
        contract_assessments=[_parse_contract_assessment(item) for item in payload["contract_assessments"]],
        confirmed_findings=[_parse_finding(item) for item in payload["confirmed_findings"]],
        rejected_hypotheses=_string_list(
            payload["rejected_hypotheses"],
            "rejected_hypotheses",
        ),
        uncertainties=_string_list(payload["uncertainties"], "uncertainties"),
        observation_refs=_string_list(
            payload["observation_refs"],
            "observation_refs",
        ),
        investigation_summary=_non_empty_string(
            payload["investigation_summary"],
            "investigation_summary",
        ),
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
    *,
    model: str = "configured-reviewer-model",
) -> ReviewerRun:
    runtime = RuntimeTracker.start()
    envelope = build_reviewer_envelope(
        assignment=assignment,
        intent=intent,
        code_snippets={"Diff Excerpt": "\n".join(diff_excerpt)},
        observations=observations,
        trace_id=trace_id,
        model=model,
        max_output_tokens=assignment.max_output_tokens,
    )
    runtime.model_turns = 1
    turn_response = None
    attempt_failures: list[str] = []

    for attempt_index in range(1, assignment.max_provider_attempts + 1):
        budget_reason = budget_reason_before_call(assignment, runtime)
        if budget_reason is not None:
            return _single_shot_budget_run(
                envelope,
                turn_response,
                observations,
                runtime,
                budget_reason,
                attempt_failures,
            )

        request = ModelTurnRequest(
            system=envelope.system,
            tools=[],
            messages=[dict(message) for message in envelope.messages],
            tool_results=[],
            parameters={
                **request_parameters(envelope.parameters, assignment, runtime),
                "tool_choice": "none",
            },
        )
        try:
            candidate = adapter.complete_turn(request)
        except Exception as error:  # Provider adapters are an isolation boundary.
            runtime.record_attempt(None)
            attempt_failures.append(
                f"provider attempt {attempt_index} raised "
                f"{type(error).__name__}: {error}"
            )
            budget_reason = budget_reason_after_call(assignment, runtime)
            if budget_reason is not None:
                return _single_shot_budget_run(
                    envelope,
                    turn_response,
                    observations,
                    runtime,
                    budget_reason,
                    attempt_failures,
                )
            continue

        turn_response = candidate
        runtime.record_attempt(candidate.raw)
        budget_reason = budget_reason_after_call(assignment, runtime)
        if budget_reason is not None:
            return _single_shot_budget_run(
                envelope,
                turn_response,
                observations,
                runtime,
                budget_reason,
                attempt_failures,
            )
        if candidate.kind is not ModelResponseKind.INVALID:
            break
        attempt_failures.append(
            f"provider attempt {attempt_index} returned INVALID: "
            f"{candidate.error or 'unspecified invalid response'}"
        )

    if turn_response is None or turn_response.kind is ModelResponseKind.INVALID:
        message = "provider retry exhausted"
        failures = _dedupe([*attempt_failures, message])
        result = _runtime_result(
            status=ReviewerResultStatus.FAILED,
            reason=message,
            observation_refs=sorted(observations),
            uncertainties=failures,
        )
        response = _model_response(
            turn_response,
            adapter,
            model,
            fallback_error="; ".join(failures),
        )
        return ReviewerRun(
            envelope=envelope,
            response=response,
            result=result,
            runtime=runtime.snapshot(
                ReviewerTerminationReason.PROVIDER_RETRY_EXHAUSTED
            ),
        )

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
            result = _runtime_result(
                status=ReviewerResultStatus.FAILED,
                reason=message,
                observation_refs=sorted(observations),
                uncertainties=[message],
            )
        validation = validate_reviewer_completion(
            assignment,
            result,
            set(observations),
        )
        if not validation.accepted:
            result = result_with_validation_deficiencies(
                result,
                validation.deficiencies,
            )
        return ReviewerRun(
            envelope=envelope,
            response=response,
            result=result,
            runtime=runtime.snapshot(termination_reason_for_result(result)),
        )
    if turn_response.kind is ModelResponseKind.TOOL_CALLS:
        message = "single-shot reviewer received tool calls; use --reviewer-loop agent-loop to enable tools"
    else:
        message = turn_response.error or f"single-shot reviewer received invalid response kind: {turn_response.kind.value}"
    result = _runtime_result(
        status=ReviewerResultStatus.FAILED,
        reason=message,
        observation_refs=sorted(observations),
        uncertainties=[message],
    )
    return ReviewerRun(
        envelope=envelope,
        response=response,
        result=result,
        runtime=runtime.snapshot(ReviewerTerminationReason.RUNTIME_FAILURE),
    )


def _single_shot_budget_run(
    envelope: ModelInvocationEnvelope,
    turn_response: Any,
    observations: dict[str, str],
    runtime: RuntimeTracker,
    reason: ReviewerTerminationReason | None,
    failures: list[str],
) -> ReviewerRun:
    reason = reason or ReviewerTerminationReason.RUNTIME_FAILURE
    reason_text = termination_summary(reason)
    result = _runtime_result(
        status=ReviewerResultStatus.PARTIAL,
        reason=reason_text,
        observation_refs=sorted(observations),
        uncertainties=_dedupe([*failures, reason_text]),
    )
    response = _model_response(
        turn_response,
        None,
        str(envelope.parameters.get("model", "unavailable")),
        fallback_error=reason_text,
    )
    return ReviewerRun(
        envelope=envelope,
        response=response,
        result=result,
        runtime=runtime.snapshot(reason),
    )


def _model_response(
    turn_response: Any,
    adapter: ModelAdapter | None,
    model: str,
    *,
    fallback_error: str,
) -> ModelResponse:
    if turn_response is not None:
        return ModelResponse(
            content=turn_response.final_text or turn_response.error or fallback_error,
            provider_name=turn_response.provider_name,
            model=turn_response.model,
            raw=turn_response.raw,
        )
    return ModelResponse(
        content=fallback_error,
        provider_name=getattr(adapter, "provider_name", "review-agent"),
        model=model,
        raw={"error": fallback_error},
    )


def _runtime_result(
    *,
    status: ReviewerResultStatus,
    reason: str,
    observation_refs: list[str],
    uncertainties: list[str],
) -> ReviewerResult:
    retained = ", ".join(observation_refs) if observation_refs else "none"
    return ReviewerResult(
        uncertainties=uncertainties,
        observation_refs=observation_refs,
        investigation_summary=(
            f"Reviewer execution stopped because {reason}. "
            f"Authorized observations retained: {retained}."
        ),
        status=status,
    )


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


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


def _parse_contract_assessment(item: Any) -> ContractAssessment:
    if not isinstance(item, dict):
        raise ReviewerResultParseError("contract assessment must be an object")
    _require_exact_keys(
        item,
        {"contract", "status", "summary", "evidence_refs"},
        "contract assessment",
    )
    try:
        status = ContractItemStatus(item["status"])
    except KeyError as error:
        raise ReviewerResultParseError("contract assessment missing required key: status") from error
    except ValueError as error:
        raise ReviewerResultParseError(f"invalid contract status: {item.get('status')}") from error
    evidence_refs = _require_list(
        item,
        "evidence_refs",
        field_label="contract assessment evidence_refs",
    )
    return ContractAssessment(
        contract=_non_empty_string(item["contract"], "contract assessment contract"),
        status=status,
        summary=_non_empty_string(item["summary"], "contract assessment summary"),
        evidence_refs=_string_list(evidence_refs, "contract assessment evidence_refs"),
    )


def _parse_finding(item: Any) -> ReviewerFinding:
    if not isinstance(item, dict):
        raise ReviewerResultParseError("finding must be an object")
    _require_exact_keys(
        item,
        {
            "claim",
            "severity",
            "confidence",
            "path",
            "line",
            "evidence_refs",
            "impact",
            "suggested_action",
            "verification_performed",
        },
        "finding",
    )
    severity = _non_empty_string(item["severity"], "finding severity")
    if severity not in {"blocker", "high", "medium", "low"}:
        raise ReviewerResultParseError(f"invalid finding severity: {severity}")
    confidence = _non_empty_string(item["confidence"], "finding confidence")
    if confidence not in {"high", "medium", "low"}:
        raise ReviewerResultParseError(f"invalid finding confidence: {confidence}")
    path = _non_empty_string(item["path"], "finding path")
    if finding_path_error(path) is not None:
        raise ReviewerResultParseError(
            "finding path must be a safe repository-relative path"
        )
    line = item["line"]
    if type(line) is not int or line < 1:
        raise ReviewerResultParseError("finding line must be a positive integer")
    evidence_refs = _require_list(item, "evidence_refs", field_label="finding evidence_refs")
    parsed_evidence_refs = _string_list(evidence_refs, "finding evidence_refs")
    if not parsed_evidence_refs:
        raise ReviewerResultParseError("finding evidence_refs must not be empty")
    verification = _require_list(
        item,
        "verification_performed",
        field_label="finding verification_performed",
    )
    parsed_verification = _string_list(
        verification,
        "finding verification_performed",
    )
    if not parsed_verification:
        raise ReviewerResultParseError(
            "finding verification_performed must not be empty"
        )
    suggested_action = _non_empty_string(
        item["suggested_action"],
        "finding suggested_action",
    )
    return ReviewerFinding(
        claim=_non_empty_string(item["claim"], "finding claim"),
        severity=severity,
        confidence=confidence,
        evidence_refs=parsed_evidence_refs,
        suggested_action=suggested_action,
        path=path,
        line=line,
        impact=_non_empty_string(item["impact"], "finding impact"),
        verification_performed=parsed_verification,
    )


def _require_exact_keys(
    item: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    missing = expected - set(item)
    if missing:
        raise ReviewerResultParseError(
            f"{label} missing required key(s): {', '.join(sorted(missing))}"
        )
    unexpected = set(item) - expected
    if unexpected:
        raise ReviewerResultParseError(
            f"{label} contains unsupported key(s): "
            f"{', '.join(sorted(str(key) for key in unexpected))}"
        )


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewerResultParseError(f"{label} must be a non-empty string")
    return value.strip()


def _string_list(values: list[Any], label: str) -> list[str]:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ReviewerResultParseError(f"{label} must contain non-empty strings")
    return [value.strip() for value in values]

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from review_agent.context import (
    REVIEWER_RESULT_OUTPUT_INSTRUCTIONS,
    ReviewerMemoryContext,
    build_reviewer_envelope,
    current_reviewer_memory_context,
)
from review_agent.memory_retrieval import HardPolicyBudgetExceeded
from review_agent.model_adapter import (
    ModelAdapter,
    model_response_to_assistant_message,
    model_tool_result_to_message,
)
from review_agent.model_protocol import (
    ModelResponse,
    ModelResponseKind,
    ModelToolCall,
    ModelToolResult,
    ModelToolSpec,
    ModelTurnRequest,
    ModelTurnResponse,
)
from review_agent.models import (
    Assignment,
    IntentPacket,
    ModelInvocationEnvelope,
    ReviewerResult,
    ReviewerResultStatus,
    ReviewerRuntimeMetadata,
    ReviewerTerminationReason,
)
from review_agent.reviewer import (
    ReviewerResultParseError,
    parse_reviewer_result,
    reviewer_result_to_dict,
)
from review_agent.review_contract import (
    result_with_validation_deficiencies,
    validate_reviewer_completion,
)
from review_agent.reviewer_runtime import (
    RuntimeTracker,
    budget_reason_after_call,
    budget_reason_before_call,
    request_parameters,
    reviewer_runtime_to_dict,
    termination_reason_for_result,
    termination_summary,
)
from review_agent.tool_gateway import ToolGateway


@dataclass(frozen=True)
class AgentLoopProviderAttempt:
    provider_attempt: int
    response_kind: str
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    usage_available: bool = False


@dataclass(frozen=True)
class AgentLoopTurn:
    turn_index: int
    response_kind: str
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    tool_results: list[ModelToolResult] = field(default_factory=list)
    error: str | None = None
    provider_attempts: list[AgentLoopProviderAttempt] = field(default_factory=list)


@dataclass(frozen=True)
class AgentLoopTrace:
    trace_id: str
    turns: list[AgentLoopTurn]
    tool_call_count: int
    final_status: str
    provider_attempt_count: int = 0


@dataclass(frozen=True)
class AgentLoopRun:
    envelope: ModelInvocationEnvelope
    response: ModelResponse
    result: ReviewerResult
    trace: AgentLoopTrace
    runtime: ReviewerRuntimeMetadata


@dataclass(frozen=True)
class _JsonFinalizationOutcome:
    response: ModelTurnResponse
    result: ReviewerResult | None
    attempt: AgentLoopProviderAttempt
    error: str | None = None
    budget_reason: ReviewerTerminationReason | None = None


def run_reviewer_agent_loop(
    adapter: ModelAdapter,
    gateway: ToolGateway,
    assignment: Assignment,
    intent: IntentPacket,
    diff_excerpt: list[str],
    observations: dict[str, str],
    trace_id: str,
    *,
    model: str = "configured-reviewer-model",
) -> AgentLoopRun:
    runtime = RuntimeTracker.start()
    memory_context = current_reviewer_memory_context()
    if memory_context is None:
        gateway_snapshot = getattr(gateway, "memory_snapshot", None)
        gateway_service = getattr(gateway, "memory_query_service", None)
        if gateway_snapshot is not None and gateway_service is not None:
            memory_context = ReviewerMemoryContext(
                snapshot=gateway_snapshot,
                query_service=gateway_service,
            )
    envelope = build_reviewer_envelope(
        assignment=assignment,
        intent=intent,
        code_snippets={"Diff Excerpt": "\n".join(diff_excerpt)},
        observations=observations,
        trace_id=trace_id,
        model=model,
        max_output_tokens=assignment.max_output_tokens,
        memory_context=memory_context,
    )
    if memory_context is not None and gateway.memory_query_service is not None:
        context_metadata = envelope.parameters.get("context", {})
        gateway.bind_memory_context_ledger(
            limit_bytes=int(context_metadata["memory_ledger_limit_bytes"]),
            initial_bytes=int(context_metadata["memory_ledger_initial_bytes"]),
        )
    tools = [_tool_spec_from_envelope_tool(tool) for tool in envelope.tools]
    runtime_messages = list(envelope.messages)
    turns: list[AgentLoopTurn] = []
    tool_results: list[ModelToolResult] = []
    tool_call_count = 0
    last_response: ModelTurnResponse | None = None
    runtime_failures: list[str] = []
    json_finalization_attempted = False

    for turn_index in range(assignment.max_turns):
        budget_reason = budget_reason_before_call(assignment, runtime)
        if budget_reason is not None:
            return _budget_run(
                envelope,
                last_response,
                adapter,
                model,
                gateway,
                observations,
                trace_id,
                turns,
                runtime,
                budget_reason,
                runtime_failures,
            )

        runtime.model_turns += 1
        provider_attempts: list[AgentLoopProviderAttempt] = []
        response: ModelTurnResponse | None = None
        for provider_attempt in range(1, assignment.max_provider_attempts + 1):
            budget_reason = budget_reason_before_call(assignment, runtime)
            if budget_reason is not None:
                turns.append(
                    AgentLoopTurn(
                        turn_index=turn_index,
                        response_kind="budget_exhausted",
                        error=termination_summary(budget_reason),
                        provider_attempts=provider_attempts,
                    )
                )
                return _budget_run(
                    envelope,
                    last_response,
                    adapter,
                    model,
                    gateway,
                    observations,
                    trace_id,
                    turns,
                    runtime,
                    budget_reason,
                    runtime_failures,
                )

            request = ModelTurnRequest(
                system=envelope.system,
                tools=tools,
                messages=list(runtime_messages),
                tool_results=[],
                parameters=request_parameters(
                    envelope.parameters,
                    assignment,
                    runtime,
                ),
            )
            try:
                candidate = adapter.complete_turn(request)
            except Exception as error:  # Provider adapters are an isolation boundary.
                runtime.record_attempt(None)
                error_message = (
                    f"provider attempt {provider_attempt} raised "
                    f"{type(error).__name__}: {error}"
                )
                runtime_failures.append(error_message)
                provider_attempts.append(
                    AgentLoopProviderAttempt(
                        provider_attempt=provider_attempt,
                        response_kind="exception",
                        error=error_message,
                    )
                )
                budget_reason = budget_reason_after_call(assignment, runtime)
                if budget_reason is not None:
                    turns.append(
                        AgentLoopTurn(
                            turn_index=turn_index,
                            response_kind="exception",
                            error=error_message,
                            provider_attempts=provider_attempts,
                        )
                    )
                    return _budget_run(
                        envelope,
                        last_response,
                        adapter,
                        model,
                        gateway,
                        observations,
                        trace_id,
                        turns,
                        runtime,
                        budget_reason,
                        runtime_failures,
                    )
                continue

            response = candidate
            last_response = candidate
            usage = runtime.record_attempt(candidate.raw)
            provider_attempts.append(
                AgentLoopProviderAttempt(
                    provider_attempt=provider_attempt,
                    response_kind=candidate.kind.value,
                    error=candidate.error,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                    usage_available=usage.available,
                )
            )
            budget_reason = budget_reason_after_call(assignment, runtime)
            if budget_reason is not None:
                turns.append(
                    AgentLoopTurn(
                        turn_index=turn_index,
                        response_kind=candidate.kind.value,
                        error=termination_summary(budget_reason),
                        provider_attempts=provider_attempts,
                    )
                )
                return _budget_run(
                    envelope,
                    candidate,
                    adapter,
                    model,
                    gateway,
                    observations,
                    trace_id,
                    turns,
                    runtime,
                    budget_reason,
                    runtime_failures,
                )
            if candidate.kind is not ModelResponseKind.INVALID:
                break
            runtime_failures.append(
                f"provider attempt {provider_attempt} returned INVALID: "
                f"{candidate.error or 'unspecified invalid response'}"
            )

        if response is None or response.kind is ModelResponseKind.INVALID:
            error_message = "provider retry exhausted"
            failures = _dedupe([*runtime_failures, error_message])
            turns.append(
                AgentLoopTurn(
                    turn_index=turn_index,
                    response_kind=(response.kind.value if response else "exception"),
                    error=error_message,
                    provider_attempts=provider_attempts,
                )
            )
            result = _failed_result(
                error_message,
                _authorized_observation_ids(gateway, observations),
                failures,
            )
            return _run_from_parts(
                envelope,
                response or _synthetic_turn_response(adapter, model, error_message),
                result,
                trace_id,
                turns,
                runtime,
                ReviewerTerminationReason.PROVIDER_RETRY_EXHAUSTED,
            )

        if response.kind is ModelResponseKind.TOOL_CALLS:
            turn_tool_calls = list(response.tool_calls)
            if tool_call_count + len(turn_tool_calls) > assignment.max_tool_calls:
                turns.append(
                    AgentLoopTurn(
                        turn_index=turn_index,
                        response_kind=response.kind.value,
                        tool_calls=turn_tool_calls,
                        error="tool budget exhausted",
                        provider_attempts=provider_attempts,
                    )
                )
                result = _partial_result(
                    "tool budget exhausted",
                    _authorized_observation_ids(gateway, observations),
                )
                return _run_from_parts(
                    envelope,
                    response,
                    result,
                    trace_id,
                    turns,
                    runtime,
                    ReviewerTerminationReason.TOOL_BUDGET_EXHAUSTED,
                )

            turn_tool_results: list[ModelToolResult] = []
            attempted_in_turn = 0
            try:
                for call in turn_tool_calls:
                    attempted_in_turn += 1
                    turn_tool_results.append(_execute_tool_call(gateway, call))
            except HardPolicyBudgetExceeded as error:
                tool_call_count += attempted_in_turn
                runtime.tool_calls = tool_call_count
                error_message = f"blocking hard-policy budget failure: {error}"
                turns.append(
                    AgentLoopTurn(
                        turn_index=turn_index,
                        response_kind=response.kind.value,
                        tool_calls=turn_tool_calls,
                        tool_results=turn_tool_results,
                        error=error_message,
                        provider_attempts=provider_attempts,
                    )
                )
                result = _blocked_result(
                    error_message,
                    _authorized_observation_ids(gateway, observations),
                )
                return _run_from_parts(
                    envelope,
                    response,
                    result,
                    trace_id,
                    turns,
                    runtime,
                    ReviewerTerminationReason.REVIEWER_BLOCKED,
                )
            tool_call_count += attempted_in_turn
            runtime.tool_calls = tool_call_count
            tool_results.extend(turn_tool_results)
            runtime_messages.append(model_response_to_assistant_message(response))
            runtime_messages.extend(
                model_tool_result_to_message(result)
                for result in turn_tool_results
            )
            turns.append(
                AgentLoopTurn(
                    turn_index=turn_index,
                    response_kind=response.kind.value,
                    tool_calls=turn_tool_calls,
                    tool_results=turn_tool_results,
                    provider_attempts=provider_attempts,
                )
            )
            continue

        if response.kind is ModelResponseKind.FINAL:
            runtime_messages.append(model_response_to_assistant_message(response))
            repaired_parse_error: str | None = None
            try:
                result = parse_reviewer_result(response.final_text or "")
            except ReviewerResultParseError as error:
                error_message = f"final response parse failed: {error}"
                runtime_failures.append(error_message)
                repaired_parse_error = error_message
                if json_finalization_attempted:
                    finalization_error_message = (
                        "final response JSON finalization already attempted"
                    )
                    runtime_failures.append(finalization_error_message)
                    turns.append(
                        AgentLoopTurn(
                            turn_index=turn_index,
                            response_kind=response.kind.value,
                            error=finalization_error_message,
                            provider_attempts=provider_attempts,
                        )
                    )
                    result = _failed_result(
                        finalization_error_message,
                        _authorized_observation_ids(gateway, observations),
                        runtime_failures,
                    )
                    return _run_from_parts(
                        envelope,
                        response,
                        result,
                        trace_id,
                        turns,
                        runtime,
                        ReviewerTerminationReason.RUNTIME_FAILURE,
                    )
                if len(provider_attempts) >= assignment.max_provider_attempts:
                    finalization_error_message = (
                        "final response JSON finalization skipped: "
                        "provider attempt budget exhausted"
                    )
                    runtime_failures.append(finalization_error_message)
                    turns.append(
                        AgentLoopTurn(
                            turn_index=turn_index,
                            response_kind=response.kind.value,
                            error=error_message,
                            provider_attempts=provider_attempts,
                        )
                    )
                    result = _failed_result(
                        finalization_error_message,
                        _authorized_observation_ids(gateway, observations),
                        runtime_failures,
                    )
                    return _run_from_parts(
                        envelope,
                        response,
                        result,
                        trace_id,
                        turns,
                        runtime,
                        ReviewerTerminationReason.PROVIDER_RETRY_EXHAUSTED,
                    )
                budget_reason = budget_reason_before_call(assignment, runtime)
                if budget_reason is not None:
                    turns.append(
                        AgentLoopTurn(
                            turn_index=turn_index,
                            response_kind=response.kind.value,
                            error=error_message,
                            provider_attempts=provider_attempts,
                        )
                    )
                    return _budget_run(
                        envelope,
                        response,
                        adapter,
                        model,
                        gateway,
                        observations,
                        trace_id,
                        turns,
                        runtime,
                        budget_reason,
                        runtime_failures,
                    )

                json_finalization_attempted = True
                runtime_messages.append(
                    _runtime_json_finalization_message(error_message)
                )
                finalization = _finalize_reviewer_json(
                    adapter=adapter,
                    envelope=envelope,
                    assignment=assignment,
                    runtime=runtime,
                    runtime_messages=runtime_messages,
                    original_response=response,
                    attempt_number=len(provider_attempts) + 1,
                )
                response = finalization.response
                last_response = finalization.response
                provider_attempts.append(finalization.attempt)
                if (
                    finalization.attempt.response_kind != "exception"
                    and response.kind
                    in {ModelResponseKind.TOOL_CALLS, ModelResponseKind.FINAL}
                ):
                    runtime_messages.append(
                        model_response_to_assistant_message(response)
                    )
                if finalization.error is not None:
                    runtime_failures.append(finalization.error)
                if finalization.budget_reason is not None:
                    turns.append(
                        AgentLoopTurn(
                            turn_index=turn_index,
                            response_kind=response.kind.value,
                            error=termination_summary(finalization.budget_reason),
                            provider_attempts=provider_attempts,
                        )
                    )
                    return _budget_run(
                        envelope,
                        response,
                        adapter,
                        model,
                        gateway,
                        observations,
                        trace_id,
                        turns,
                        runtime,
                        finalization.budget_reason,
                        runtime_failures,
                    )
                if finalization.error is not None:
                    turns.append(
                        AgentLoopTurn(
                            turn_index=turn_index,
                            response_kind=response.kind.value,
                            error=finalization.error,
                            provider_attempts=provider_attempts,
                        )
                    )
                    result = _failed_result(
                        finalization.error,
                        _authorized_observation_ids(gateway, observations),
                        runtime_failures,
                    )
                    return _run_from_parts(
                        envelope,
                        response,
                        result,
                        trace_id,
                        turns,
                        runtime,
                        ReviewerTerminationReason.RUNTIME_FAILURE,
                    )
                if finalization.result is None:
                    raise AssertionError("successful JSON finalization has no result")
                result = finalization.result

            validation = validate_reviewer_completion(
                assignment,
                result,
                _authorized_observation_ids(gateway, observations),
            )
            if not validation.accepted:
                error_message = (
                    "Runtime rejected completion: "
                    + "; ".join(validation.deficiencies)
                )
                runtime_failures.append(error_message)
                turns.append(
                    AgentLoopTurn(
                        turn_index=turn_index,
                        response_kind=response.kind.value,
                        error=error_message,
                        provider_attempts=provider_attempts,
                    )
                )
                if turn_index + 1 < assignment.max_turns:
                    runtime_messages.append(
                        _runtime_rejection_message(error_message)
                    )
                    continue
                result = result_with_validation_deficiencies(
                    result,
                    validation.deficiencies,
                )
                result = _partial_result(
                    "turn budget exhausted",
                    _authorized_observation_ids(gateway, observations),
                    prior_result=result,
                )
                return _run_from_parts(
                    envelope,
                    response,
                    result,
                    trace_id,
                    turns,
                    runtime,
                    ReviewerTerminationReason.TURN_BUDGET_EXHAUSTED,
                )

            turns.append(
                AgentLoopTurn(
                    turn_index=turn_index,
                    response_kind=response.kind.value,
                    error=repaired_parse_error,
                    provider_attempts=provider_attempts,
                )
            )
            return _run_from_parts(
                envelope,
                response,
                result,
                trace_id,
                turns,
                runtime,
                termination_reason_for_result(result),
            )

        error_message = response.error or f"unexpected model response kind: {response.kind.value}"
        turns.append(
            AgentLoopTurn(
                turn_index=turn_index,
                response_kind=response.kind.value,
                error=error_message,
                provider_attempts=provider_attempts,
            )
        )
        failures = _dedupe([*runtime_failures, error_message])
        result = _failed_result(
            error_message,
            _authorized_observation_ids(gateway, observations),
            failures,
        )
        return _run_from_parts(
            envelope,
            response,
            result,
            trace_id,
            turns,
            runtime,
            ReviewerTerminationReason.RUNTIME_FAILURE,
        )

    failures = _dedupe([*runtime_failures, "turn budget exhausted"])
    result = _partial_result(
        "turn budget exhausted",
        _authorized_observation_ids(gateway, observations),
        extra_uncertainties=failures,
    )
    response = last_response or ModelTurnResponse(
        kind=ModelResponseKind.INVALID,
        error="turn budget exhausted",
        provider_name="review-agent",
        model="unavailable",
        raw={"error": "turn budget exhausted"},
    )
    return _run_from_parts(
        envelope,
        response,
        result,
        trace_id,
        turns,
        runtime,
        ReviewerTerminationReason.TURN_BUDGET_EXHAUSTED,
    )


def agent_loop_run_to_dict(run: AgentLoopRun) -> dict[str, Any]:
    return {
        "envelope": asdict(run.envelope),
        "response": asdict(run.response),
        "result": reviewer_result_to_dict(run.result),
        "trace": {
            "trace_id": run.trace.trace_id,
            "tool_call_count": run.trace.tool_call_count,
            "provider_attempt_count": run.trace.provider_attempt_count,
            "final_status": run.trace.final_status,
            "turns": [
                {
                    "turn_index": turn.turn_index,
                    "response_kind": turn.response_kind,
                    "tool_calls": [asdict(call) for call in turn.tool_calls],
                    "tool_results": [asdict(result) for result in turn.tool_results],
                    "error": turn.error,
                    "provider_attempts": [
                        asdict(attempt) for attempt in turn.provider_attempts
                    ],
                }
                for turn in run.trace.turns
            ],
        },
        "runtime": reviewer_runtime_to_dict(run.runtime),
    }


def _tool_spec_from_envelope_tool(tool: dict[str, object]) -> ModelToolSpec:
    return ModelToolSpec(
        name=str(tool["name"]),
        description=str(tool.get("description", "")),
        parameters_schema=dict(tool.get("parameters", {})),
    )


def _execute_tool_call(gateway: ToolGateway, call: ModelToolCall) -> ModelToolResult:
    try:
        result = gateway.execute(call.tool_name, call.arguments)
    except HardPolicyBudgetExceeded:
        raise
    except Exception as error:  # Tool execution is a Reviewer isolation boundary.
        return ModelToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            content=f"{type(error).__name__}: {error}",
            observation_ids=[],
            is_error=True,
        )
    return ModelToolResult(
        call_id=call.call_id,
        tool_name=call.tool_name,
        content=result.context_view,
        observation_ids=result.observation_ids,
    )


def _finalize_reviewer_json(
    *,
    adapter: ModelAdapter,
    envelope: ModelInvocationEnvelope,
    assignment: Assignment,
    runtime: RuntimeTracker,
    runtime_messages: list[dict[str, Any]],
    original_response: ModelTurnResponse,
    attempt_number: int,
) -> _JsonFinalizationOutcome:
    parameters = request_parameters(envelope.parameters, assignment, runtime)
    parameters.update(
        {
            "tool_choice": "none",
            "response_format": "json_object",
        }
    )
    request = ModelTurnRequest(
        system=envelope.system,
        tools=[],
        messages=list(runtime_messages),
        tool_results=[],
        parameters=parameters,
    )

    try:
        response = adapter.complete_turn(request)
    except Exception as error:
        runtime.record_attempt(None)
        error_message = (
            "final response JSON finalization raised "
            f"{type(error).__name__}: {error}"
        )
        return _JsonFinalizationOutcome(
            response=original_response,
            result=None,
            attempt=AgentLoopProviderAttempt(
                provider_attempt=attempt_number,
                response_kind="exception",
                error=error_message,
            ),
            error=error_message,
            budget_reason=budget_reason_after_call(assignment, runtime),
        )

    usage = runtime.record_attempt(response.raw)
    attempt = AgentLoopProviderAttempt(
        provider_attempt=attempt_number,
        response_kind=response.kind.value,
        error=response.error,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        usage_available=usage.available,
    )
    budget_reason = budget_reason_after_call(assignment, runtime)
    if budget_reason is not None:
        return _JsonFinalizationOutcome(
            response=response,
            result=None,
            attempt=attempt,
            budget_reason=budget_reason,
        )
    if response.kind is not ModelResponseKind.FINAL:
        return _JsonFinalizationOutcome(
            response=response,
            result=None,
            attempt=attempt,
            error=(
                "final response JSON finalization returned "
                f"{response.kind.value}"
            ),
        )
    try:
        result = parse_reviewer_result(response.final_text or "")
    except ReviewerResultParseError as error:
        return _JsonFinalizationOutcome(
            response=response,
            result=None,
            attempt=attempt,
            error=f"final response JSON finalization parse failed: {error}",
        )
    return _JsonFinalizationOutcome(
        response=response,
        result=result,
        attempt=attempt,
    )


def _blocked_result(
    reason: str,
    observation_refs: set[str],
) -> ReviewerResult:
    retained = sorted(observation_refs)
    retained_summary = ", ".join(retained) if retained else "none"
    return ReviewerResult(
        uncertainties=[reason],
        observation_refs=retained,
        investigation_summary=(
            f"Reviewer execution was blocked because {reason}. "
            f"Authorized observations retained: {retained_summary}."
        ),
        status=ReviewerResultStatus.BLOCKED,
    )


def _partial_result(
    uncertainty: str,
    observation_refs: set[str],
    *,
    prior_result: ReviewerResult | None = None,
    extra_uncertainties: list[str] | None = None,
) -> ReviewerResult:
    prior = prior_result or ReviewerResult()
    retained = _dedupe([*prior.observation_refs, *sorted(observation_refs)])
    uncertainties = _dedupe(
        [
            *prior.uncertainties,
            *(extra_uncertainties or []),
            uncertainty,
        ]
    )
    retained_summary = ", ".join(retained) if retained else "none"
    previous_summary = prior.investigation_summary.strip()
    stop_summary = (
        f"Reviewer execution stopped because {uncertainty}. "
        f"Authorized observations retained: {retained_summary}."
    )
    return replace(
        prior,
        uncertainties=uncertainties,
        observation_refs=retained,
        investigation_summary=(
            f"{previous_summary} {stop_summary}".strip()
            if previous_summary
            else stop_summary
        ),
        status=ReviewerResultStatus.PARTIAL,
    )


def _failed_result(
    reason: str,
    observation_refs: set[str],
    uncertainties: list[str] | None = None,
) -> ReviewerResult:
    retained = sorted(observation_refs)
    retained_summary = ", ".join(retained) if retained else "none"
    return ReviewerResult(
        uncertainties=_dedupe([*(uncertainties or []), reason]),
        observation_refs=retained,
        investigation_summary=(
            f"Reviewer execution failed because {reason}. "
            f"Authorized observations retained: {retained_summary}."
        ),
        status=ReviewerResultStatus.FAILED,
    )


def _authorized_observation_ids(
    gateway: ToolGateway,
    initial_observations: dict[str, str],
) -> set[str]:
    return {
        *initial_observations,
        *gateway.observation_store.summaries_by_id(),
    }


def _runtime_rejection_message(reason: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"{reason}. Continue the assigned investigation and submit a corrected "
            "structured result that satisfies every Runtime requirement."
        ),
    }


def _runtime_json_finalization_message(reason: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"{reason}. The investigation is complete. Do not call tools or add "
            "new analysis. Convert the completed analysis into exactly one JSON "
            "object that follows this Runtime-owned protocol.\n\n"
            f"{REVIEWER_RESULT_OUTPUT_INSTRUCTIONS}"
        ),
    }


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _run_from_parts(
    envelope: ModelInvocationEnvelope,
    turn_response: ModelTurnResponse,
    result: ReviewerResult,
    trace_id: str,
    turns: list[AgentLoopTurn],
    runtime: RuntimeTracker,
    termination_reason: ReviewerTerminationReason,
) -> AgentLoopRun:
    response = ModelResponse(
        content=turn_response.final_text or turn_response.error or "",
        provider_name=turn_response.provider_name,
        model=turn_response.model,
        raw=turn_response.raw,
    )
    trace = AgentLoopTrace(
        trace_id=trace_id,
        turns=turns,
        tool_call_count=runtime.tool_calls,
        final_status=result.status.value,
        provider_attempt_count=runtime.provider_attempts,
    )
    return AgentLoopRun(
        envelope=envelope,
        response=response,
        result=result,
        trace=trace,
        runtime=runtime.snapshot(termination_reason),
    )


def _budget_run(
    envelope: ModelInvocationEnvelope,
    last_response: ModelTurnResponse | None,
    adapter: ModelAdapter,
    model: str,
    gateway: ToolGateway,
    initial_observations: dict[str, str],
    trace_id: str,
    turns: list[AgentLoopTurn],
    runtime: RuntimeTracker,
    reason: ReviewerTerminationReason,
    runtime_failures: list[str],
) -> AgentLoopRun:
    reason_text = termination_summary(reason)
    result = _partial_result(
        reason_text,
        _authorized_observation_ids(gateway, initial_observations),
        extra_uncertainties=runtime_failures,
    )
    response = last_response or _synthetic_turn_response(
        adapter,
        model,
        reason_text,
    )
    return _run_from_parts(
        envelope,
        response,
        result,
        trace_id,
        turns,
        runtime,
        reason,
    )


def _synthetic_turn_response(
    adapter: ModelAdapter,
    model: str,
    error: str,
) -> ModelTurnResponse:
    return ModelTurnResponse(
        kind=ModelResponseKind.INVALID,
        error=error,
        provider_name=getattr(adapter, "provider_name", "review-agent"),
        model=model,
        raw={"error": error},
    )

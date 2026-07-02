from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from review_agent.context import build_reviewer_envelope
from review_agent.model_adapter import ModelAdapter
from review_agent.model_protocol import (
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
)
from review_agent.provider import ModelResponse
from review_agent.reviewer import ReviewerResultParseError, parse_reviewer_result, reviewer_result_to_dict
from review_agent.tool_gateway import ToolGateway, ToolGatewayError


@dataclass(frozen=True)
class AgentLoopTurn:
    turn_index: int
    response_kind: str
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    tool_results: list[ModelToolResult] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class AgentLoopTrace:
    trace_id: str
    turns: list[AgentLoopTurn]
    tool_call_count: int
    final_status: str


@dataclass(frozen=True)
class AgentLoopRun:
    envelope: ModelInvocationEnvelope
    response: ModelResponse
    result: ReviewerResult
    trace: AgentLoopTrace


def run_reviewer_agent_loop(
    adapter: ModelAdapter,
    gateway: ToolGateway,
    assignment: Assignment,
    intent: IntentPacket,
    diff_excerpt: list[str],
    observations: dict[str, str],
    trace_id: str,
) -> AgentLoopRun:
    envelope = build_reviewer_envelope(
        assignment=assignment,
        intent=intent,
        code_snippets={"Diff Excerpt": "\n".join(diff_excerpt)},
        observations=observations,
        trace_id=trace_id,
    )
    tools = [_tool_spec_from_envelope_tool(tool) for tool in envelope.tools]
    turns: list[AgentLoopTurn] = []
    tool_results: list[ModelToolResult] = []
    tool_call_count = 0
    last_response: ModelTurnResponse | None = None

    for turn_index in range(assignment.max_turns):
        request = ModelTurnRequest(
            system=envelope.system,
            tools=tools,
            messages=list(envelope.messages),
            tool_results=list(tool_results),
            parameters=dict(envelope.parameters),
        )
        response = adapter.complete_turn(request)
        last_response = response

        if response.kind is ModelResponseKind.TOOL_CALLS:
            turn_tool_calls = list(response.tool_calls)
            if tool_call_count + len(turn_tool_calls) > assignment.max_tool_calls:
                turns.append(
                    AgentLoopTurn(
                        turn_index=turn_index,
                        response_kind=response.kind.value,
                        tool_calls=turn_tool_calls,
                        error="tool budget exhausted",
                    )
                )
                result = _partial_result("tool budget exhausted")
                return _run_from_parts(envelope, response, result, trace_id, turns, tool_call_count)

            turn_tool_results = [_execute_tool_call(gateway, call) for call in turn_tool_calls]
            tool_call_count += len(turn_tool_calls)
            tool_results.extend(turn_tool_results)
            turns.append(
                AgentLoopTurn(
                    turn_index=turn_index,
                    response_kind=response.kind.value,
                    tool_calls=turn_tool_calls,
                    tool_results=turn_tool_results,
                )
            )
            continue

        if response.kind is ModelResponseKind.FINAL:
            try:
                result = parse_reviewer_result(response.final_text or "")
            except ReviewerResultParseError as error:
                error_message = f"final response parse failed: {error}"
                turns.append(
                    AgentLoopTurn(
                        turn_index=turn_index,
                        response_kind=response.kind.value,
                        error=error_message,
                    )
                )
                result = ReviewerResult(
                    uncertainties=[error_message],
                    investigation_summary=error_message,
                    status=ReviewerResultStatus.FAILED,
                )
                return _run_from_parts(envelope, response, result, trace_id, turns, tool_call_count)

            turns.append(AgentLoopTurn(turn_index=turn_index, response_kind=response.kind.value))
            return _run_from_parts(envelope, response, result, trace_id, turns, tool_call_count)

        error_message = response.error or f"unexpected model response kind: {response.kind.value}"
        turns.append(
            AgentLoopTurn(
                turn_index=turn_index,
                response_kind=response.kind.value,
                error=error_message,
            )
        )
        result = ReviewerResult(
            uncertainties=[error_message],
            investigation_summary=error_message,
            status=ReviewerResultStatus.FAILED,
        )
        return _run_from_parts(envelope, response, result, trace_id, turns, tool_call_count)

    result = _partial_result("turn budget exhausted")
    response = last_response or ModelTurnResponse(
        kind=ModelResponseKind.INVALID,
        error="turn budget exhausted",
        provider_name="review-agent",
        model="unavailable",
        raw={"error": "turn budget exhausted"},
    )
    return _run_from_parts(envelope, response, result, trace_id, turns, tool_call_count)


def agent_loop_run_to_dict(run: AgentLoopRun) -> dict[str, Any]:
    return {
        "envelope": asdict(run.envelope),
        "response": asdict(run.response),
        "result": reviewer_result_to_dict(run.result),
        "trace": {
            "trace_id": run.trace.trace_id,
            "tool_call_count": run.trace.tool_call_count,
            "final_status": run.trace.final_status,
            "turns": [
                {
                    "turn_index": turn.turn_index,
                    "response_kind": turn.response_kind,
                    "tool_calls": [asdict(call) for call in turn.tool_calls],
                    "tool_results": [asdict(result) for result in turn.tool_results],
                    "error": turn.error,
                }
                for turn in run.trace.turns
            ],
        },
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
    except (ToolGatewayError, KeyError, ValueError, TypeError) as error:
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


def _partial_result(uncertainty: str) -> ReviewerResult:
    return ReviewerResult(
        uncertainties=[uncertainty],
        investigation_summary=uncertainty,
        status=ReviewerResultStatus.PARTIAL,
    )


def _run_from_parts(
    envelope: ModelInvocationEnvelope,
    turn_response: ModelTurnResponse,
    result: ReviewerResult,
    trace_id: str,
    turns: list[AgentLoopTurn],
    tool_call_count: int,
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
        tool_call_count=tool_call_count,
        final_status=result.status.value,
    )
    return AgentLoopRun(envelope=envelope, response=response, result=result, trace=trace)

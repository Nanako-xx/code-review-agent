from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

from review_agent.execution_journal import (
    ExecutionJournal,
    JournalIntegrityError,
    PendingTurn,
    ToolCallIdentity,
)
from review_agent.model_adapter import model_response_to_assistant_message
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolSpec,
    ModelTurnRequest,
    ModelTurnResponse,
)
from review_agent.review_context import ReviewerInvocationV2
from review_agent.review_protocol import ReviewerAssignment
from review_agent.review_tool_gateway import ReviewToolGateway
from review_agent.reviewer_runtime import (
    ReviewerRuntimeLimitsV2,
    ReviewerRuntimeStateV2,
    request_parameters_v2,
)
from review_agent.tool_artifacts import ToolResultProjector
from review_agent.tool_result_protocol import (
    ReviewToolResult,
    ToolResultProjectionV2,
)


@dataclass(frozen=True)
class ReviewAgentRunV2:
    status: str
    final_text: str | None
    error_code: str | None
    messages: tuple[dict[str, Any], ...]
    runtime: ReviewerRuntimeStateV2


class ReviewAgentLoopError(ValueError):
    pass


class ReviewAgentLoopV2:
    def __init__(
        self,
        *,
        adapter: Any,
        gateway: ReviewToolGateway,
        projector: ToolResultProjector,
        journal: ExecutionJournal,
        assignment: ReviewerAssignment,
        invocation: ReviewerInvocationV2,
        limits: ReviewerRuntimeLimitsV2 | None = None,
        clock: Callable[[], float] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        if not hasattr(adapter, "complete_turn"):
            raise ReviewAgentLoopError("adapter must implement complete_turn")
        if not isinstance(gateway, ReviewToolGateway):
            raise ReviewAgentLoopError("gateway must be ReviewToolGateway")
        if not isinstance(projector, ToolResultProjector):
            raise ReviewAgentLoopError("projector must be ToolResultProjector")
        if not isinstance(journal, ExecutionJournal):
            raise ReviewAgentLoopError("journal must be ExecutionJournal")
        if type(assignment) is not ReviewerAssignment:
            raise ReviewAgentLoopError("assignment must be ReviewerAssignment")
        if assignment != journal.assignment:
            raise ReviewAgentLoopError("Journal Assignment binding does not match")
        if gateway.snapshot_id != assignment.snapshot_id:
            raise ReviewAgentLoopError("Gateway Snapshot binding does not match")
        if not isinstance(invocation, ReviewerInvocationV2):
            raise ReviewAgentLoopError("invocation must be ReviewerInvocationV2")
        self.adapter = adapter
        self.gateway = gateway
        self.projector = projector
        self.journal = journal
        self.assignment = assignment
        self.invocation = invocation
        self.limits = limits or ReviewerRuntimeLimitsV2()
        if abs(self.gateway.timeout_seconds - self.limits.tool_timeout_seconds) > 1e-9:
            raise ReviewAgentLoopError(
                "Gateway timeout must match Reviewer Runtime limits"
            )
        self.clock = clock or time.monotonic
        self.cancelled = cancelled or (lambda: False)

    def run(self) -> ReviewAgentRunV2:
        replay = self.journal.replay()
        runtime = ReviewerRuntimeStateV2(
            active_elapsed_seconds=replay.active_elapsed_seconds,
            provider_attempts=replay.provider_attempts,
            model_turns=replay.model_turns,
            tool_calls=replay.tool_calls,
            input_tokens=replay.input_tokens,
            output_tokens=replay.output_tokens,
            total_tokens=replay.total_tokens,
            all_usage_available=replay.all_usage_available,
        )
        messages = [dict(message) for message in self.invocation.messages]
        messages.extend(dict(message) for message in replay.committed_messages)

        if replay.final_text is not None:
            return self._result(
                "completed",
                replay.final_text,
                None,
                messages,
                runtime,
            )

        if replay.pending_turn is not None:
            try:
                projections = self._complete_pending_turn(
                    replay.pending_turn,
                    runtime,
                )
                messages.append(dict(replay.pending_turn.assistant_message))
                messages.extend(_tool_message(item) for item in projections)
            except TimeoutError:
                return self._result(
                    "timeout", None, "active_time_exhausted", messages, runtime
                )
            except (JournalIntegrityError, ValueError):
                return self._result(
                    "failed", None, "journal_integrity_error", messages, runtime
                )

        while True:
            if self.cancelled():
                return self._result(
                    "cancelled", None, "user_cancelled", messages, runtime
                )
            if runtime.remaining_seconds(self.limits) <= 0:
                return self._result(
                    "timeout", None, "active_time_exhausted", messages, runtime
                )

            response: ModelTurnResponse | None = None
            last_transport_error = False
            replay_before_request = self.journal.replay()
            request_turn_index = (
                replay_before_request.committed_turns[-1] + 1
                if replay_before_request.committed_turns
                else 0
            )
            for provider_attempt in range(1, self.limits.max_provider_attempts + 1):
                try:
                    parameters = request_parameters_v2(
                        self.invocation.parameters,
                        runtime,
                        self.limits,
                    )
                except TimeoutError:
                    return self._result(
                        "timeout",
                        None,
                        "active_time_exhausted",
                        messages,
                        runtime,
                    )
                request = ModelTurnRequest(
                    system=self.invocation.system,
                    tools=_tool_specs(self.invocation),
                    messages=[dict(message) for message in messages],
                    tool_results=[],
                    parameters=parameters,
                )
                started = self.clock()
                try:
                    candidate = self.adapter.complete_turn(request)
                except Exception:
                    runtime.consume_active(max(0.0, self.clock() - started))
                    usage = runtime.record_provider_attempt(None)
                    self.journal.record_provider_attempt(
                        turn_index=request_turn_index,
                        attempt=provider_attempt,
                        status="failed",
                        response_kind=None,
                        error_code="provider_transport_error",
                        usage={
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "total_tokens": usage.total_tokens,
                            "available": usage.available,
                        },
                        active_elapsed_seconds=runtime.active_elapsed_seconds,
                    )
                    last_transport_error = True
                    continue
                runtime.consume_active(max(0.0, self.clock() - started))
                usage = runtime.record_provider_attempt(
                    candidate.raw if isinstance(candidate, ModelTurnResponse) else None
                )
                self.journal.record_provider_attempt(
                    turn_index=request_turn_index,
                    attempt=provider_attempt,
                    status="succeeded",
                    response_kind=(
                        candidate.kind.value
                        if isinstance(candidate, ModelTurnResponse)
                        else None
                    ),
                    error_code=(
                        None
                        if isinstance(candidate, ModelTurnResponse)
                        else "invalid_provider_response"
                    ),
                    usage={
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                        "available": usage.available,
                    },
                    active_elapsed_seconds=runtime.active_elapsed_seconds,
                )
                if not isinstance(candidate, ModelTurnResponse):
                    return self._result(
                        "invalid_output",
                        None,
                        "invalid_provider_response",
                        messages,
                        runtime,
                    )
                response = candidate
                last_transport_error = False
                break
            if response is None:
                return self._result(
                    "failed",
                    None,
                    "provider_transport_failed" if last_transport_error else "provider_failed",
                    messages,
                    runtime,
                )

            runtime.model_turns += 1
            if response.kind is ModelResponseKind.INVALID:
                return self._result(
                    "invalid_output",
                    None,
                    "invalid_model_output",
                    messages,
                    runtime,
                )
            if response.kind is ModelResponseKind.FINAL:
                if not isinstance(response.final_text, str) or not response.final_text.strip():
                    return self._result(
                        "invalid_output",
                        None,
                        "missing_final_output",
                        messages,
                        runtime,
                    )
                self.journal.record_final_result(
                    final_text=response.final_text,
                    active_elapsed_seconds=runtime.active_elapsed_seconds,
                )
                return self._result(
                    "completed",
                    response.final_text,
                    None,
                    messages,
                    runtime,
                )
            if response.kind is not ModelResponseKind.TOOL_CALLS or not response.tool_calls:
                return self._result(
                    "invalid_output",
                    None,
                    "invalid_tool_call_batch",
                    messages,
                    runtime,
                )

            assistant_message = model_response_to_assistant_message(response)
            replay = self.journal.replay()
            repeated_completed_batch = True
            for call in response.tool_calls:
                completed = replay.completed_calls.get(call.call_id)
                if completed is None:
                    repeated_completed_batch = False
                    continue
                expected_identity = ToolCallIdentity.from_call(
                    session_id=self.journal.session.session_id,
                    assignment_id=self.assignment.assignment_id,
                    snapshot_id=self.assignment.snapshot_id,
                    call=call,
                )
                if completed.identity != expected_identity:
                    return self._result(
                        "failed",
                        None,
                        "journal_integrity_error",
                        messages,
                        runtime,
                    )
            if repeated_completed_batch:
                return self._result(
                    "failed",
                    None,
                    "repeated_tool_call_no_progress",
                    messages,
                    runtime,
                )
            turn_index = (
                replay.committed_turns[-1] + 1 if replay.committed_turns else 0
            )
            self.journal.record_model_response(
                turn_index=turn_index,
                assistant_message=assistant_message,
                tool_calls=tuple(response.tool_calls),
                active_elapsed_seconds=runtime.active_elapsed_seconds,
            )
            pending = self.journal.replay().pending_turn
            if pending is None:
                return self._result(
                    "failed", None, "journal_integrity_error", messages, runtime
                )
            try:
                projections = self._complete_pending_turn(pending, runtime)
            except TimeoutError:
                return self._result(
                    "timeout", None, "active_time_exhausted", messages, runtime
                )
            except (JournalIntegrityError, ValueError):
                return self._result(
                    "failed", None, "journal_integrity_error", messages, runtime
                )
            messages.append(dict(assistant_message))
            messages.extend(_tool_message(item) for item in projections)

    def _complete_pending_turn(
        self,
        pending: PendingTurn,
        runtime: ReviewerRuntimeStateV2,
    ) -> tuple[ToolResultProjectionV2, ...]:
        replay = self.journal.replay()
        projections_by_call: dict[str, ToolResultProjectionV2] = {
            call_id: completed.projection
            for call_id, completed in replay.completed_calls.items()
        }
        raw_results: list[ReviewToolResult] = []
        raw_identities: list[ToolCallIdentity] = []
        for call in pending.tool_calls:
            identity = ToolCallIdentity.from_call(
                session_id=self.journal.session.session_id,
                assignment_id=self.assignment.assignment_id,
                snapshot_id=self.assignment.snapshot_id,
                call=call,
            )
            completed = replay.completed_calls.get(call.call_id)
            if completed is not None:
                if completed.identity != identity:
                    raise JournalIntegrityError("completed Tool Call identity changed")
                continue
            self.journal.record_tool_started(
                identity,
                arguments=call.arguments,
                active_elapsed_seconds=runtime.active_elapsed_seconds,
            )
            if runtime.remaining_seconds(self.limits) <= 0:
                raise TimeoutError("Reviewer active time exhausted")
            started = self.clock()
            raw = self.gateway.execute(call.call_id, call.tool_name, call.arguments)
            runtime.consume_active(max(0.0, self.clock() - started))
            runtime.tool_calls += 1
            raw_results.append(raw)
            raw_identities.append(identity)

        if raw_results:
            started = self.clock()
            batch = self.projector.project_turn(tuple(raw_results))
            runtime.consume_active(max(0.0, self.clock() - started))
            for identity, projection in zip(raw_identities, batch.projections):
                self.journal.record_tool_completed(
                    identity,
                    projection,
                    active_elapsed_seconds=runtime.active_elapsed_seconds,
                )
                projections_by_call[identity.tool_call_id] = projection

        projections = tuple(
            projections_by_call[call.call_id] for call in pending.tool_calls
        )
        self.journal.record_turn_committed(
            turn_index=pending.turn_index,
            assistant_message=pending.assistant_message,
            projections=projections,
            active_elapsed_seconds=runtime.active_elapsed_seconds,
        )
        return projections

    @staticmethod
    def _result(
        status: str,
        final_text: str | None,
        error_code: str | None,
        messages: list[dict[str, Any]],
        runtime: ReviewerRuntimeStateV2,
    ) -> ReviewAgentRunV2:
        return ReviewAgentRunV2(
            status=status,
            final_text=final_text,
            error_code=error_code,
            messages=tuple(dict(message) for message in messages),
            runtime=runtime,
        )


def _tool_specs(invocation: ReviewerInvocationV2) -> list[ModelToolSpec]:
    return [
        ModelToolSpec(
            name=tool["name"],
            description=tool["description"],
            parameters_schema=dict(tool["parameters"]),
        )
        for tool in invocation.tools
    ]


def _tool_message(projection: ToolResultProjectionV2) -> dict[str, Any]:
    from review_agent.model_adapter import review_tool_projection_to_message

    return review_tool_projection_to_message(projection)


__all__ = [
    "ReviewAgentLoopError",
    "ReviewAgentLoopV2",
    "ReviewAgentRunV2",
]

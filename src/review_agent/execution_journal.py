from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable, Mapping

from review_agent.model_protocol import ModelToolCall
from review_agent.pr_workspace import (
    PRWorkspaceStore,
    SessionWorkspace,
)
from review_agent.review_protocol import ReviewerAssignment
from review_agent.safe_io import (
    SafeIOError,
    assert_regular_file,
    canonical_json_bytes,
    resolve_managed_path,
    strict_json_loads,
)
from review_agent.tool_result_protocol import (
    ToolErrorEnvelope,
    ToolResultProjectionV2,
    serialize_tool_result_projection_v2,
    validate_serialized_tool_result_projection_v2,
)


EXECUTION_JOURNAL_SCHEMA = "execution_journal_event_v1"
_EVENT_TYPES = frozenset(
    {
        "model_response",
        "provider_attempt",
        "tool_started",
        "tool_completed",
        "turn_committed",
        "context_idle_eviction",
        "context_compaction_started",
        "context_compaction_committed",
        "final_result",
    }
)
_SESSION_ID = re.compile(r"\ASESSION-[0-9a-f]{64}\Z")
_ASSIGNMENT_ID = re.compile(r"\AASG-[0-9a-f]{64}\Z")
_SNAPSHOT_ID = re.compile(r"\AS-[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


class JournalError(ValueError):
    pass


class JournalIntegrityError(JournalError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_object(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise JournalError(f"{field_name} must be an object")
    try:
        return json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError, UnicodeError) as error:
        raise JournalError(f"{field_name} must be canonical JSON") from error


def _arguments_hash(arguments: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(arguments))).hexdigest()


@dataclass(frozen=True)
class ToolCallIdentity:
    session_id: str
    assignment_id: str
    tool_call_id: str
    tool_name: str
    canonical_arguments_hash: str
    snapshot_id: str

    def __post_init__(self) -> None:
        if type(self.session_id) is not str or _SESSION_ID.fullmatch(
            self.session_id
        ) is None:
            raise JournalError("Tool Call session_id is invalid")
        if type(self.assignment_id) is not str or _ASSIGNMENT_ID.fullmatch(
            self.assignment_id
        ) is None:
            raise JournalError("Tool Call assignment_id is invalid")
        if type(self.tool_call_id) is not str or not self.tool_call_id.strip():
            raise JournalError("Tool Call ID must be non-empty")
        if type(self.tool_name) is not str or not self.tool_name.strip():
            raise JournalError("Tool name must be non-empty")
        if type(self.canonical_arguments_hash) is not str or _SHA256.fullmatch(
            self.canonical_arguments_hash
        ) is None:
            raise JournalError("Tool arguments hash is invalid")
        if type(self.snapshot_id) is not str or _SNAPSHOT_ID.fullmatch(
            self.snapshot_id
        ) is None:
            raise JournalError("Tool Call snapshot_id is invalid")

    @classmethod
    def from_call(
        cls,
        *,
        session_id: str,
        assignment_id: str,
        snapshot_id: str,
        call: ModelToolCall,
    ) -> "ToolCallIdentity":
        if not isinstance(call, ModelToolCall):
            raise JournalError("call must be ModelToolCall")
        arguments = _json_object(call.arguments, "Tool Call arguments")
        return cls(
            session_id=session_id,
            assignment_id=assignment_id,
            tool_call_id=call.call_id,
            tool_name=call.tool_name,
            canonical_arguments_hash=_arguments_hash(arguments),
            snapshot_id=snapshot_id,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "assignment_id": self.assignment_id,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "canonical_arguments_hash": self.canonical_arguments_hash,
            "snapshot_id": self.snapshot_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolCallIdentity":
        expected = {
            "session_id",
            "assignment_id",
            "tool_call_id",
            "tool_name",
            "canonical_arguments_hash",
            "snapshot_id",
        }
        if type(value) is not dict or set(value) != expected:
            raise JournalIntegrityError("Tool Call identity schema is invalid")
        return cls(**dict(value))


@dataclass(frozen=True)
class JournalEvent:
    sequence: int
    event_type: str
    session_id: str
    assignment_id: str
    snapshot_id: str
    occurred_at: str
    active_elapsed_seconds: float
    previous_hash: str | None
    payload: dict[str, Any]
    event_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXECUTION_JOURNAL_SCHEMA,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "session_id": self.session_id,
            "assignment_id": self.assignment_id,
            "snapshot_id": self.snapshot_id,
            "occurred_at": self.occurred_at,
            "active_elapsed_seconds": self.active_elapsed_seconds,
            "previous_hash": self.previous_hash,
            "payload": self.payload,
            "event_hash": self.event_hash,
        }


@dataclass(frozen=True)
class CompletedToolCall:
    identity: ToolCallIdentity
    projection: ToolResultProjectionV2


@dataclass(frozen=True)
class PendingTurn:
    turn_index: int
    assistant_message: dict[str, Any]
    tool_calls: tuple[ModelToolCall, ...]


@dataclass(frozen=True)
class JournalReplay:
    committed_messages: tuple[dict[str, Any], ...]
    pending_turn: PendingTurn | None
    completed_calls: dict[str, CompletedToolCall]
    started_without_terminal: dict[str, ToolCallIdentity]
    active_elapsed_seconds: float
    committed_turns: tuple[int, ...]
    final_text: str | None
    provider_attempts: int
    model_turns: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    all_usage_available: bool


def _projection_to_record(value: ToolResultProjectionV2) -> dict[str, Any]:
    return {
        "tool_call_id": value.tool_call_id,
        "tool_name": value.tool_name,
        "status": value.status,
        "original_size": value.original_size,
        "reacquirable": value.reacquirable,
        "content": value.content,
        "preview": value.preview,
        "artifact_id": value.artifact_id,
        "aggregate_entry": value.aggregate_entry,
        "reacquire_arguments": value.reacquire_arguments,
        "error": None if value.error is None else value.error.to_dict(),
    }


def _projection_from_record(value: Any) -> ToolResultProjectionV2:
    expected = {
        "tool_call_id",
        "tool_name",
        "status",
        "original_size",
        "reacquirable",
        "content",
        "preview",
        "artifact_id",
        "aggregate_entry",
        "reacquire_arguments",
        "error",
    }
    if type(value) is not dict or set(value) != expected:
        raise JournalIntegrityError("Tool Result projection record is invalid")
    error_payload = value["error"]
    error = None
    if error_payload is not None:
        if type(error_payload) is not dict:
            raise JournalIntegrityError("Tool Result error record is invalid")
        fields = {"is_error", "code", "retryable", "message"}
        if "exit_code" in error_payload:
            fields.add("exit_code")
        if set(error_payload) != fields or error_payload["is_error"] is not True:
            raise JournalIntegrityError("Tool Result error schema is invalid")
        error = ToolErrorEnvelope(
            code=error_payload["code"],
            retryable=error_payload["retryable"],
            message=error_payload["message"],
            exit_code=error_payload.get("exit_code"),
        )
    return ToolResultProjectionV2(
        tool_call_id=value["tool_call_id"],
        tool_name=value["tool_name"],
        status=value["status"],
        original_size=value["original_size"],
        reacquirable=value["reacquirable"],
        content=value["content"],
        preview=value["preview"],
        artifact_id=value["artifact_id"],
        aggregate_entry=value["aggregate_entry"],
        reacquire_arguments=value["reacquire_arguments"],
        error=error,
    )


class ExecutionJournal:
    def __init__(
        self,
        workspace_store: PRWorkspaceStore,
        session: SessionWorkspace,
        assignment: ReviewerAssignment,
        *,
        utc_now: Callable[[], str] | None = None,
        os_module: Any = os,
    ) -> None:
        if not isinstance(workspace_store, PRWorkspaceStore):
            raise JournalError("workspace_store must be PRWorkspaceStore")
        workspace_store.verify_session(session)
        if session.workspace != session.snapshot.workspace:
            raise JournalIntegrityError("Session workspace binding is invalid")
        if assignment.snapshot_id != session.snapshot.snapshot_id:
            raise JournalIntegrityError("Assignment Snapshot binding is invalid")
        try:
            path = resolve_managed_path(session.path, "execution-log.jsonl")
            assert_regular_file(path)
        except SafeIOError as error:
            raise JournalIntegrityError("Execution journal path is unavailable") from error
        self.workspace_store = workspace_store
        self.session = session
        self.assignment = assignment
        self.path = path
        self._utc_now = utc_now or _utc_now
        self._os = os_module
        key = str(path.resolve()).casefold()
        with _PATH_LOCKS_GUARD:
            self._lock = _PATH_LOCKS.setdefault(key, threading.Lock())
        self.read_events()

    def read_events(self) -> tuple[JournalEvent, ...]:
        try:
            raw = assert_regular_file(self.path).read_bytes()
        except (OSError, SafeIOError) as error:
            raise JournalIntegrityError("Execution journal is unavailable") from error
        events: list[JournalEvent] = []
        previous_hash: str | None = None
        elapsed_by_assignment: dict[str, float] = {}
        for sequence, line in enumerate(raw.splitlines(), start=1):
            try:
                payload = strict_json_loads(line)
            except SafeIOError as error:
                raise JournalIntegrityError("Execution journal JSON is invalid") from error
            expected = {
                "schema_version",
                "sequence",
                "event_type",
                "session_id",
                "assignment_id",
                "snapshot_id",
                "occurred_at",
                "active_elapsed_seconds",
                "previous_hash",
                "payload",
                "event_hash",
            }
            if type(payload) is not dict or set(payload) != expected:
                raise JournalIntegrityError("Execution journal event schema is invalid")
            if payload["schema_version"] != EXECUTION_JOURNAL_SCHEMA:
                raise JournalIntegrityError("Execution journal schema is unsupported")
            if payload["sequence"] != sequence:
                raise JournalIntegrityError("Execution journal sequence is invalid")
            if payload["event_type"] not in _EVENT_TYPES:
                raise JournalIntegrityError("Execution journal event type is invalid")
            if payload["session_id"] != self.session.session_id:
                raise JournalIntegrityError("Execution journal Session binding is invalid")
            if payload["snapshot_id"] != self.session.snapshot.snapshot_id:
                raise JournalIntegrityError("Execution journal Snapshot binding is invalid")
            if type(payload["assignment_id"]) is not str or _ASSIGNMENT_ID.fullmatch(
                payload["assignment_id"]
            ) is None:
                raise JournalIntegrityError("Execution journal Assignment ID is invalid")
            elapsed = payload["active_elapsed_seconds"]
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed)
                or elapsed < elapsed_by_assignment.get(payload["assignment_id"], 0.0)
            ):
                raise JournalIntegrityError("Execution journal active time is invalid")
            elapsed = float(elapsed)
            elapsed_by_assignment[payload["assignment_id"]] = elapsed
            if payload["previous_hash"] != previous_hash:
                raise JournalIntegrityError("Execution journal previous hash is invalid")
            event_base = dict(payload)
            event_hash = event_base.pop("event_hash")
            expected_hash = hashlib.sha256(canonical_json_bytes(event_base)).hexdigest()
            if event_hash != expected_hash:
                raise JournalIntegrityError("Execution journal event hash is invalid")
            event = JournalEvent(
                sequence=sequence,
                event_type=payload["event_type"],
                session_id=payload["session_id"],
                assignment_id=payload["assignment_id"],
                snapshot_id=payload["snapshot_id"],
                occurred_at=payload["occurred_at"],
                active_elapsed_seconds=elapsed,
                previous_hash=previous_hash,
                payload=_json_object(payload["payload"], "event payload"),
                event_hash=event_hash,
            )
            events.append(event)
            previous_hash = event_hash
        return tuple(events)

    def replay(self) -> JournalReplay:
        events = tuple(
            event
            for event in self.read_events()
            if event.assignment_id == self.assignment.assignment_id
        )
        committed_messages: list[dict[str, Any]] = []
        pending: PendingTurn | None = None
        identities: dict[str, ToolCallIdentity] = {}
        started: dict[str, ToolCallIdentity] = {}
        completed: dict[str, CompletedToolCall] = {}
        committed_turns: list[int] = []
        active_elapsed = 0.0
        final_text: str | None = None
        provider_attempts = 0
        model_turns = 0
        tool_calls = 0
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        all_usage_available = True
        for event in events:
            active_elapsed = event.active_elapsed_seconds
            if event.event_type == "provider_attempt":
                attempt = event.payload
                expected = {
                    "turn_index",
                    "attempt",
                    "status",
                    "response_kind",
                    "error_code",
                    "usage",
                }
                if set(attempt) != expected or attempt["status"] not in {
                    "succeeded",
                    "failed",
                }:
                    raise JournalIntegrityError(
                        "Provider attempt payload is invalid"
                    )
                usage = attempt["usage"]
                if type(usage) is not dict or set(usage) != {
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "available",
                }:
                    raise JournalIntegrityError("Provider usage payload is invalid")
                for name in ("input_tokens", "output_tokens", "total_tokens"):
                    if type(usage[name]) is not int or usage[name] < 0:
                        raise JournalIntegrityError("Provider usage value is invalid")
                if type(usage["available"]) is not bool:
                    raise JournalIntegrityError("Provider usage availability is invalid")
                provider_attempts += 1
                if attempt["status"] == "succeeded":
                    model_turns += 1
                input_tokens += usage["input_tokens"]
                output_tokens += usage["output_tokens"]
                total_tokens += usage["total_tokens"]
                all_usage_available = all_usage_available and usage["available"]
            elif event.event_type == "model_response":
                if pending is not None:
                    raise JournalIntegrityError(
                        "multiple uncommitted model responses are present"
                    )
                pending = _pending_from_payload(event.payload)
                for call in pending.tool_calls:
                    identity = ToolCallIdentity.from_call(
                        session_id=self.session.session_id,
                        assignment_id=self.assignment.assignment_id,
                        snapshot_id=self.session.snapshot.snapshot_id,
                        call=call,
                    )
                    _bind_identity(identities, identity)
            elif event.event_type == "tool_started":
                identity = ToolCallIdentity.from_dict(event.payload["identity"])
                _require_current_identity(self, identity)
                arguments = _json_object(event.payload["arguments"], "arguments")
                if _arguments_hash(arguments) != identity.canonical_arguments_hash:
                    raise JournalIntegrityError("Tool Call arguments hash is invalid")
                _bind_identity(identities, identity)
                if identity.tool_call_id not in completed:
                    started[identity.tool_call_id] = identity
            elif event.event_type == "tool_completed":
                identity = ToolCallIdentity.from_dict(event.payload["identity"])
                _require_current_identity(self, identity)
                _bind_identity(identities, identity)
                if identity.tool_call_id not in started:
                    raise JournalIntegrityError(
                        "Tool Call completed without a started event"
                    )
                projection = _projection_from_record(event.payload["projection"])
                if (
                    projection.tool_call_id != identity.tool_call_id
                    or projection.tool_name != identity.tool_name
                ):
                    raise JournalIntegrityError(
                        "Tool Result projection identity does not match"
                    )
                if identity.tool_call_id in completed:
                    raise JournalIntegrityError("Tool Call has multiple terminal results")
                completed[identity.tool_call_id] = CompletedToolCall(
                    identity=identity,
                    projection=projection,
                )
                tool_calls += 1
                started.pop(identity.tool_call_id, None)
            elif event.event_type == "turn_committed":
                if pending is None:
                    raise JournalIntegrityError("Turn committed without model response")
                turn_index = event.payload.get("turn_index")
                if turn_index != pending.turn_index:
                    raise JournalIntegrityError("Committed Turn index does not match")
                assistant = _json_object(
                    event.payload.get("assistant_message"), "assistant_message"
                )
                if assistant != pending.assistant_message:
                    raise JournalIntegrityError("Committed assistant message changed")
                tool_messages = event.payload.get("tool_messages")
                if type(tool_messages) is not list:
                    raise JournalIntegrityError("Committed tool messages are invalid")
                expected_ids = [call.call_id for call in pending.tool_calls]
                if [message.get("tool_call_id") for message in tool_messages] != expected_ids:
                    raise JournalIntegrityError(
                        "Committed Tool Results are not adjacent and ordered"
                    )
                for message in tool_messages:
                    if type(message) is not dict or set(message) != {
                        "role",
                        "tool_call_id",
                        "content",
                    }:
                        raise JournalIntegrityError("Committed Tool Result schema is invalid")
                    if message["role"] != "tool":
                        raise JournalIntegrityError("Committed Tool Result role is invalid")
                    try:
                        validate_serialized_tool_result_projection_v2(
                            message["content"]
                        )
                    except ValueError as error:
                        raise JournalIntegrityError(
                            "Committed Tool Result content is invalid"
                        ) from error
                    if message["tool_call_id"] not in completed:
                        raise JournalIntegrityError(
                            "Committed Tool Call has no terminal result"
                        )
                committed_messages.append(assistant)
                committed_messages.extend(dict(message) for message in tool_messages)
                committed_turns.append(turn_index)
                pending = None
            elif event.event_type == "final_result":
                if pending is not None:
                    raise JournalIntegrityError(
                        "Final result cannot follow an uncommitted Tool Turn"
                    )
                candidate = event.payload.get("final_text")
                if type(candidate) is not str or not candidate.strip():
                    raise JournalIntegrityError("Final result payload is invalid")
                if final_text is not None and final_text != candidate:
                    raise JournalIntegrityError("Final result changed")
                final_text = candidate
        return JournalReplay(
            committed_messages=tuple(committed_messages),
            pending_turn=pending,
            completed_calls=completed,
            started_without_terminal=started,
            active_elapsed_seconds=active_elapsed,
            committed_turns=tuple(committed_turns),
            final_text=final_text,
            provider_attempts=provider_attempts,
            model_turns=model_turns,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            all_usage_available=all_usage_available,
        )

    def record_provider_attempt(
        self,
        *,
        turn_index: int,
        attempt: int,
        status: str,
        response_kind: str | None,
        error_code: str | None,
        usage: Mapping[str, Any],
        active_elapsed_seconds: float,
    ) -> JournalEvent:
        if type(turn_index) is not int or turn_index < 0:
            raise JournalError("turn_index must be non-negative")
        if type(attempt) is not int or attempt <= 0:
            raise JournalError("attempt must be positive")
        if status not in {"succeeded", "failed"}:
            raise JournalError("Provider attempt status is invalid")
        if response_kind is not None and (
            type(response_kind) is not str or not response_kind
        ):
            raise JournalError("response_kind must be text or null")
        if error_code is not None and (
            type(error_code) is not str or not error_code
        ):
            raise JournalError("error_code must be text or null")
        normalized_usage = _json_object(usage, "usage")
        if set(normalized_usage) != {
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "available",
        }:
            raise JournalError("Provider usage schema is invalid")
        return self._append(
            "provider_attempt",
            {
                "turn_index": turn_index,
                "attempt": attempt,
                "status": status,
                "response_kind": response_kind,
                "error_code": error_code,
                "usage": normalized_usage,
            },
            active_elapsed_seconds,
        )

    def record_model_response(
        self,
        *,
        turn_index: int,
        assistant_message: Mapping[str, Any],
        tool_calls: tuple[ModelToolCall, ...],
        active_elapsed_seconds: float,
    ) -> JournalEvent:
        if type(turn_index) is not int or turn_index < 0:
            raise JournalError("turn_index must be non-negative")
        if type(tool_calls) is not tuple or not tool_calls or any(
            not isinstance(call, ModelToolCall) for call in tool_calls
        ):
            raise JournalError("tool_calls must be a non-empty ModelToolCall tuple")
        replay = self.replay()
        candidate = {
            "turn_index": turn_index,
            "assistant_message": _json_object(
                assistant_message, "assistant_message"
            ),
            "tool_calls": [
                {
                    "call_id": call.call_id,
                    "tool_name": call.tool_name,
                    "arguments": _json_object(call.arguments, "arguments"),
                }
                for call in tool_calls
            ],
        }
        if replay.pending_turn is not None:
            if _pending_from_payload(candidate) == replay.pending_turn:
                return next(
                    event
                    for event in reversed(self.read_events())
                    if event.assignment_id == self.assignment.assignment_id
                    and event.event_type == "model_response"
                )
            raise JournalIntegrityError("pending model response identity changed")
        return self._append(
            "model_response", candidate, active_elapsed_seconds
        )

    def record_tool_started(
        self,
        identity: ToolCallIdentity,
        *,
        arguments: Mapping[str, Any],
        active_elapsed_seconds: float,
    ) -> JournalEvent | CompletedToolCall:
        _require_current_identity(self, identity)
        normalized = _json_object(arguments, "arguments")
        if _arguments_hash(normalized) != identity.canonical_arguments_hash:
            raise JournalIntegrityError("Tool Call identity arguments changed")
        replay = self.replay()
        _check_replay_identity(replay, identity)
        if identity.tool_call_id in replay.completed_calls:
            return replay.completed_calls[identity.tool_call_id]
        if identity.tool_call_id in replay.started_without_terminal:
            return next(
                event
                for event in reversed(self.read_events())
                if event.assignment_id == self.assignment.assignment_id
                and event.event_type == "tool_started"
                and event.payload["identity"]["tool_call_id"]
                == identity.tool_call_id
            )
        return self._append(
            "tool_started",
            {"identity": identity.to_dict(), "arguments": normalized},
            active_elapsed_seconds,
        )

    def record_tool_completed(
        self,
        identity: ToolCallIdentity,
        projection: ToolResultProjectionV2,
        *,
        active_elapsed_seconds: float,
    ) -> ToolResultProjectionV2:
        _require_current_identity(self, identity)
        if not isinstance(projection, ToolResultProjectionV2):
            raise JournalError("projection must be ToolResultProjectionV2")
        if (
            projection.tool_call_id != identity.tool_call_id
            or projection.tool_name != identity.tool_name
        ):
            raise JournalIntegrityError("Tool Result projection identity changed")
        replay = self.replay()
        _check_replay_identity(replay, identity)
        existing = replay.completed_calls.get(identity.tool_call_id)
        if existing is not None:
            if existing.projection != projection:
                raise JournalIntegrityError("Tool Call terminal result changed")
            return existing.projection
        if identity.tool_call_id not in replay.started_without_terminal:
            raise JournalIntegrityError("Tool Call must be started before completion")
        self._append(
            "tool_completed",
            {
                "identity": identity.to_dict(),
                "projection": _projection_to_record(projection),
            },
            active_elapsed_seconds,
        )
        return projection

    def record_turn_committed(
        self,
        *,
        turn_index: int,
        assistant_message: Mapping[str, Any],
        projections: tuple[ToolResultProjectionV2, ...],
        active_elapsed_seconds: float,
    ) -> JournalEvent:
        replay = self.replay()
        if turn_index in replay.committed_turns and replay.pending_turn is None:
            return next(
                event
                for event in reversed(self.read_events())
                if event.assignment_id == self.assignment.assignment_id
                and event.event_type == "turn_committed"
                and event.payload["turn_index"] == turn_index
            )
        if replay.pending_turn is None or replay.pending_turn.turn_index != turn_index:
            raise JournalIntegrityError("Turn has no matching pending model response")
        calls = replay.pending_turn.tool_calls
        if type(projections) is not tuple or [
            projection.tool_call_id for projection in projections
        ] != [call.call_id for call in calls]:
            raise JournalIntegrityError("Turn projections do not match Tool Calls")
        for projection in projections:
            completed = replay.completed_calls.get(projection.tool_call_id)
            if completed is None or completed.projection != projection:
                raise JournalIntegrityError("Turn projection is not terminal")
        assistant = _json_object(assistant_message, "assistant_message")
        if assistant != replay.pending_turn.assistant_message:
            raise JournalIntegrityError("Turn assistant message changed")
        tool_messages = [
            {
                "role": "tool",
                "tool_call_id": projection.tool_call_id,
                "content": serialize_tool_result_projection_v2(projection),
            }
            for projection in projections
        ]
        return self._append(
            "turn_committed",
            {
                "turn_index": turn_index,
                "assistant_message": assistant,
                "tool_messages": tool_messages,
            },
            active_elapsed_seconds,
        )

    def record_final_result(
        self,
        *,
        final_text: str,
        active_elapsed_seconds: float,
    ) -> JournalEvent:
        if type(final_text) is not str or not final_text.strip():
            raise JournalError("final_text must be non-empty")
        replay = self.replay()
        if replay.final_text is not None:
            if replay.final_text != final_text:
                raise JournalIntegrityError("Final result changed")
            return next(
                event
                for event in reversed(self.read_events())
                if event.assignment_id == self.assignment.assignment_id
                and event.event_type == "final_result"
            )
        if replay.pending_turn is not None:
            raise JournalIntegrityError("cannot finalize an uncommitted Tool Turn")
        return self._append(
            "final_result",
            {"final_text": final_text},
            active_elapsed_seconds,
        )

    def _append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        active_elapsed_seconds: float,
    ) -> JournalEvent:
        if event_type not in _EVENT_TYPES:
            raise JournalError("event_type is unsupported")
        if (
            isinstance(active_elapsed_seconds, bool)
            or not isinstance(active_elapsed_seconds, (int, float))
            or not math.isfinite(active_elapsed_seconds)
            or active_elapsed_seconds < 0
        ):
            raise JournalError("active_elapsed_seconds must be non-negative")
        normalized_elapsed = round(float(active_elapsed_seconds), 6)
        with self._lock:
            events = self.read_events()
            assignment_events = [
                event
                for event in events
                if event.assignment_id == self.assignment.assignment_id
            ]
            if assignment_events and normalized_elapsed < (
                assignment_events[-1].active_elapsed_seconds
            ):
                raise JournalIntegrityError("active elapsed time moved backwards")
            base = {
                "schema_version": EXECUTION_JOURNAL_SCHEMA,
                "sequence": len(events) + 1,
                "event_type": event_type,
                "session_id": self.session.session_id,
                "assignment_id": self.assignment.assignment_id,
                "snapshot_id": self.session.snapshot.snapshot_id,
                "occurred_at": self._utc_now(),
                "active_elapsed_seconds": normalized_elapsed,
                "previous_hash": events[-1].event_hash if events else None,
                "payload": _json_object(payload, "event payload"),
            }
            event_hash = hashlib.sha256(canonical_json_bytes(base)).hexdigest()
            encoded = canonical_json_bytes({**base, "event_hash": event_hash}) + b"\n"
            try:
                descriptor = self._os.open(
                    self.path,
                    self._os.O_WRONLY | self._os.O_APPEND,
                )
                try:
                    offset = 0
                    while offset < len(encoded):
                        written = self._os.write(descriptor, encoded[offset:])
                        if written <= 0:
                            raise OSError("Execution journal append made no progress")
                        offset += written
                    self._os.fsync(descriptor)
                finally:
                    self._os.close(descriptor)
            except OSError as error:
                raise JournalIntegrityError("Execution journal append failed") from error
            return self.read_events()[-1]


def _pending_from_payload(payload: Mapping[str, Any]) -> PendingTurn:
    expected = {"turn_index", "assistant_message", "tool_calls"}
    if type(payload) is not dict or set(payload) != expected:
        raise JournalIntegrityError("model_response payload schema is invalid")
    turn_index = payload["turn_index"]
    if type(turn_index) is not int or turn_index < 0:
        raise JournalIntegrityError("model_response turn_index is invalid")
    assistant = _json_object(payload["assistant_message"], "assistant_message")
    calls_payload = payload["tool_calls"]
    if type(calls_payload) is not list or not calls_payload:
        raise JournalIntegrityError("model_response tool_calls are invalid")
    calls: list[ModelToolCall] = []
    for value in calls_payload:
        if type(value) is not dict or set(value) != {
            "call_id",
            "tool_name",
            "arguments",
        }:
            raise JournalIntegrityError("model_response Tool Call schema is invalid")
        calls.append(
            ModelToolCall(
                call_id=value["call_id"],
                tool_name=value["tool_name"],
                arguments=_json_object(value["arguments"], "arguments"),
            )
        )
    if len({call.call_id for call in calls}) != len(calls):
        raise JournalIntegrityError("model_response Tool Call IDs are duplicated")
    return PendingTurn(
        turn_index=turn_index,
        assistant_message=assistant,
        tool_calls=tuple(calls),
    )


def _bind_identity(
    identities: dict[str, ToolCallIdentity],
    identity: ToolCallIdentity,
) -> None:
    existing = identities.get(identity.tool_call_id)
    if existing is not None and existing != identity:
        raise JournalIntegrityError(
            "the same tool call_id has a different identity"
        )
    identities[identity.tool_call_id] = identity


def _require_current_identity(
    journal: ExecutionJournal,
    identity: ToolCallIdentity,
) -> None:
    if (
        identity.session_id != journal.session.session_id
        or identity.assignment_id != journal.assignment.assignment_id
        or identity.snapshot_id != journal.session.snapshot.snapshot_id
    ):
        raise JournalIntegrityError("Tool Call identity binding does not match")


def _check_replay_identity(
    replay: JournalReplay,
    identity: ToolCallIdentity,
) -> None:
    existing = replay.completed_calls.get(identity.tool_call_id)
    if existing is not None and existing.identity != identity:
        raise JournalIntegrityError("completed Tool Call identity changed")
    started = replay.started_without_terminal.get(identity.tool_call_id)
    if started is not None and started != identity:
        raise JournalIntegrityError("started Tool Call identity changed")
    if replay.pending_turn is not None:
        matching = [
            call
            for call in replay.pending_turn.tool_calls
            if call.call_id == identity.tool_call_id
        ]
        if matching:
            expected = ToolCallIdentity.from_call(
                session_id=identity.session_id,
                assignment_id=identity.assignment_id,
                snapshot_id=identity.snapshot_id,
                call=matching[0],
            )
            if expected != identity:
                raise JournalIntegrityError("pending Tool Call identity changed")


__all__ = [
    "CompletedToolCall",
    "EXECUTION_JOURNAL_SCHEMA",
    "ExecutionJournal",
    "JournalError",
    "JournalEvent",
    "JournalIntegrityError",
    "JournalReplay",
    "PendingTurn",
    "ToolCallIdentity",
]

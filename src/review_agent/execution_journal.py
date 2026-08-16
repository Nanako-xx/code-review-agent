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
from review_agent.reviewer_output import RejectedReviewerFinding
from review_agent.safe_io import (
    SafeIOError,
    assert_regular_file,
    canonical_json_bytes,
    ensure_secure_directory,
    publish_create_only_bytes,
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
REVIEWER_RUNTIME_BINDING_SCHEMA = "reviewer_runtime_binding_v1"
_REVIEWER_RUNTIME_PHYSICAL_DIGEST_CHARS = 8
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
_COMPACTION_PATH = re.compile(r"\Acontext-compaction-(?P<generation>[0-9]{8})\.txt\Z")
_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


class JournalError(ValueError):
    pass


class JournalIntegrityError(JournalError):
    pass


def _publish_binding_or_verify(path: Path, content: bytes, context: str) -> None:
    try:
        publish_create_only_bytes(path, content)
    except SafeIOError:
        try:
            existing = assert_regular_file(path).read_bytes()
        except (OSError, SafeIOError) as error:
            raise JournalIntegrityError(f"{context} is unavailable") from error
        if existing != content:
            raise JournalIntegrityError(f"{context} binding changed")


def _create_or_open_journal(path: Path) -> None:
    """Create an empty journal once, or reopen its existing append-only bytes."""

    try:
        publish_create_only_bytes(path, b"")
    except SafeIOError:
        try:
            assert_regular_file(path)
        except (OSError, SafeIOError) as error:
            raise JournalIntegrityError("Execution journal is unavailable") from error


def _reviewer_runtime_path(
    session: SessionWorkspace,
    assignment: ReviewerAssignment,
) -> Path:
    # Keep the complete Assignment ID in reviewer.json while using a short
    # physical namespace to preserve room for create-only staging names on
    # legacy Windows MAX_PATH. Any prefix collision fails closed on the full
    # immutable binding below.
    physical_id = (
        "r-"
        + assignment.assignment_id[
            4 : 4 + _REVIEWER_RUNTIME_PHYSICAL_DIGEST_CHARS
        ]
    )
    try:
        reviewers = resolve_managed_path(session.path, "Reviewers")
        ensure_secure_directory(reviewers)
        runtime = resolve_managed_path(reviewers, physical_id)
        ensure_secure_directory(runtime)
        binding = canonical_json_bytes(
            {
                "schema_version": REVIEWER_RUNTIME_BINDING_SCHEMA,
                "session_id": session.session_id,
                "pr_id": session.workspace.pr_id,
                "snapshot_id": session.snapshot.snapshot_id,
                "assignment_id": assignment.assignment_id,
            }
        )
        _publish_binding_or_verify(
            resolve_managed_path(runtime, "reviewer.json"),
            binding,
            "Reviewer Runtime binding",
        )
        return runtime
    except JournalIntegrityError:
        raise
    except (OSError, SafeIOError) as error:
        raise JournalIntegrityError("Reviewer Runtime path is unavailable") from error


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
class CommittedTurnRecord:
    turn_index: int
    messages: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ContextCompactionRecord:
    generation: int
    through_turn: int
    source_start_turn: int
    source_end_turn: int
    trigger: str
    summary_path: str
    summary_hash: str
    committed_sequence: int


@dataclass(frozen=True)
class JournalReplay:
    committed_messages: tuple[dict[str, Any], ...]
    committed_turn_messages: tuple[CommittedTurnRecord, ...]
    pending_turn: PendingTurn | None
    completed_calls: dict[str, CompletedToolCall]
    started_without_terminal: dict[str, ToolCallIdentity]
    active_elapsed_seconds: float
    committed_turns: tuple[int, ...]
    final_text: str | None
    final_rejections: tuple[dict[str, Any], ...]
    provider_attempts: int
    model_turns: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    all_usage_available: bool
    context_eviction_markers: dict[str, dict[str, Any]]
    context_compaction: ContextCompactionRecord | None
    max_compaction_generation: int


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


def _context_eviction_marker(value: object) -> dict[str, Any]:
    marker = _json_object(value, "Context eviction marker")
    expected = {
        "status",
        "reason",
        "tool_call_id",
        "tool_name",
        "arguments_hash",
        "reacquirable",
    }
    if set(marker) != expected:
        raise JournalIntegrityError("Context eviction marker schema is invalid")
    if marker["status"] != "context_evicted":
        raise JournalIntegrityError("Context eviction marker status is invalid")
    if marker["reason"] != "prompt_cache_idle_60m":
        raise JournalIntegrityError("Context eviction marker reason is invalid")
    if (
        type(marker["tool_call_id"]) is not str
        or not marker["tool_call_id"].strip()
        or type(marker["tool_name"]) is not str
        or not marker["tool_name"].strip()
    ):
        raise JournalIntegrityError("Context eviction marker identity is invalid")
    arguments_hash = marker["arguments_hash"]
    if (
        type(arguments_hash) is not str
        or not arguments_hash.startswith("sha256:")
        or _SHA256.fullmatch(arguments_hash[7:]) is None
    ):
        raise JournalIntegrityError("Context eviction arguments hash is invalid")
    if marker["reacquirable"] is not True:
        raise JournalIntegrityError("Context eviction must be reacquirable")
    return marker


def _context_compaction_started_payload(value: object) -> dict[str, Any]:
    payload = _json_object(value, "Context compaction start")
    expected = {
        "generation",
        "through_turn",
        "source_start_turn",
        "source_end_turn",
        "trigger",
    }
    if set(payload) != expected:
        raise JournalIntegrityError("Context compaction start schema is invalid")
    generation = payload["generation"]
    through_turn = payload["through_turn"]
    source_start_turn = payload["source_start_turn"]
    source_end_turn = payload["source_end_turn"]
    if type(generation) is not int or generation <= 0:
        raise JournalIntegrityError("Context compaction generation is invalid")
    if type(through_turn) is not int or through_turn < 0:
        raise JournalIntegrityError("Context compaction Turn is invalid")
    if (
        type(source_start_turn) is not int
        or source_start_turn < 0
        or type(source_end_turn) is not int
        or source_end_turn != through_turn
        or source_start_turn > source_end_turn
    ):
        raise JournalIntegrityError("Context compaction source range is invalid")
    if payload["trigger"] not in {"soft_threshold", "hard_input_limit"}:
        raise JournalIntegrityError("Context compaction trigger is invalid")
    return payload


def _context_compaction_committed_payload(value: object) -> dict[str, Any]:
    payload = _json_object(value, "Context compaction commit")
    expected = {
        "generation",
        "through_turn",
        "source_start_turn",
        "source_end_turn",
        "trigger",
        "summary_path",
        "summary_hash",
    }
    if set(payload) != expected:
        raise JournalIntegrityError("Context compaction commit schema is invalid")
    _context_compaction_started_payload(
        {key: payload[key] for key in expected if key not in {"summary_path", "summary_hash"}}
    )
    summary_path = payload["summary_path"]
    match = _COMPACTION_PATH.fullmatch(summary_path) if type(summary_path) is str else None
    if (
        match is None
        or int(match.group("generation")) != payload["generation"]
    ):
        raise JournalIntegrityError("Context compaction summary path is invalid")
    if type(payload["summary_hash"]) is not str or _SHA256.fullmatch(
        payload["summary_hash"]
    ) is None:
        raise JournalIntegrityError("Context compaction summary hash is invalid")
    return payload


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
        self.workspace_store = workspace_store
        self.session = session
        self.assignment = assignment
        self.runtime_path = _reviewer_runtime_path(session, assignment)
        try:
            path = resolve_managed_path(self.runtime_path, "execution-log.jsonl")
            _create_or_open_journal(path)
            assert_regular_file(path)
        except JournalIntegrityError:
            raise
        except SafeIOError as error:
            raise JournalIntegrityError(
                "Execution journal path is unavailable"
            ) from error
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
            if payload["assignment_id"] != self.assignment.assignment_id:
                raise JournalIntegrityError(
                    "Execution journal Assignment binding is invalid"
                )
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
        committed_turn_messages: list[CommittedTurnRecord] = []
        pending: PendingTurn | None = None
        identities: dict[str, ToolCallIdentity] = {}
        started: dict[str, ToolCallIdentity] = {}
        completed: dict[str, CompletedToolCall] = {}
        committed_turns: list[int] = []
        context_eviction_markers: dict[str, dict[str, Any]] = {}
        compactions_started: dict[int, dict[str, Any]] = {}
        context_compaction: ContextCompactionRecord | None = None
        max_compaction_generation = 0
        active_elapsed = 0.0
        final_text: str | None = None
        final_rejections: tuple[dict[str, Any], ...] = ()
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
                committed_turn_messages.append(
                    CommittedTurnRecord(
                        turn_index=turn_index,
                        messages=(
                            dict(assistant),
                            *(dict(message) for message in tool_messages),
                        ),
                    )
                )
                committed_turns.append(turn_index)
                pending = None
            elif event.event_type == "context_idle_eviction":
                marker_values = event.payload.get("markers")
                if set(event.payload) != {"markers"} or type(marker_values) is not list:
                    raise JournalIntegrityError(
                        "Context idle eviction payload is invalid"
                    )
                for marker_value in marker_values:
                    marker = _context_eviction_marker(marker_value)
                    call_id = marker["tool_call_id"]
                    terminal = completed.get(call_id)
                    if terminal is None:
                        raise JournalIntegrityError(
                            "Context eviction references an unknown Tool Call"
                        )
                    if (
                        terminal.identity.tool_name != marker["tool_name"]
                        or "sha256:"
                        + terminal.identity.canonical_arguments_hash
                        != marker["arguments_hash"]
                        or not terminal.projection.reacquirable
                    ):
                        raise JournalIntegrityError(
                            "Context eviction Tool Call binding changed"
                        )
                    existing_marker = context_eviction_markers.get(call_id)
                    if existing_marker is not None and existing_marker != marker:
                        raise JournalIntegrityError(
                            "Context eviction marker changed"
                        )
                    context_eviction_markers[call_id] = marker
            elif event.event_type == "context_compaction_started":
                candidate = _context_compaction_started_payload(event.payload)
                generation = candidate["generation"]
                if generation in compactions_started:
                    if compactions_started[generation] != candidate:
                        raise JournalIntegrityError(
                            "Context compaction start changed"
                        )
                    raise JournalIntegrityError(
                        "Context compaction generation started more than once"
                    )
                if candidate["through_turn"] not in committed_turns:
                    raise JournalIntegrityError(
                        "Context compaction does not end at a committed Turn"
                    )
                if generation <= max_compaction_generation:
                    raise JournalIntegrityError(
                        "Context compaction generation did not advance"
                    )
                compactions_started[generation] = candidate
                max_compaction_generation = generation
            elif event.event_type == "context_compaction_committed":
                candidate = _context_compaction_committed_payload(event.payload)
                generation = candidate["generation"]
                started_payload = compactions_started.get(generation)
                if started_payload is None:
                    raise JournalIntegrityError(
                        "Context compaction committed without a start"
                    )
                if {
                    key: candidate[key] for key in started_payload
                } != started_payload:
                    raise JournalIntegrityError(
                        "Context compaction commit changed its source"
                    )
                if (
                    context_compaction is not None
                    and generation <= context_compaction.generation
                ):
                    raise JournalIntegrityError(
                        "Context compaction commit did not advance"
                    )
                context_compaction = ContextCompactionRecord(
                    generation=generation,
                    through_turn=candidate["through_turn"],
                    source_start_turn=candidate["source_start_turn"],
                    source_end_turn=candidate["source_end_turn"],
                    trigger=candidate["trigger"],
                    summary_path=candidate["summary_path"],
                    summary_hash=candidate["summary_hash"],
                    committed_sequence=event.sequence,
                )
            elif event.event_type == "final_result":
                if pending is not None:
                    raise JournalIntegrityError(
                        "Final result cannot follow an uncommitted Tool Turn"
                    )
                candidate = event.payload.get("final_text")
                if type(candidate) is not str or not candidate.strip():
                    raise JournalIntegrityError("Final result payload is invalid")
                if set(event.payload) not in (
                    {"final_text"},
                    {"final_text", "rejected_findings"},
                ):
                    raise JournalIntegrityError("Final result schema is invalid")
                raw_rejections = event.payload.get("rejected_findings", [])
                if type(raw_rejections) is not list:
                    raise JournalIntegrityError(
                        "Final result rejection records are invalid"
                    )
                try:
                    normalized_rejections = tuple(
                        RejectedReviewerFinding.from_dict(item).to_dict()
                        for item in raw_rejections
                    )
                except ValueError as error:
                    raise JournalIntegrityError(
                        "Final result rejection record is invalid"
                    ) from error
                rejection_indices = [
                    item["candidate_index"] for item in normalized_rejections
                ]
                if rejection_indices != sorted(set(rejection_indices)):
                    raise JournalIntegrityError(
                        "Final result rejection records are not ordered and unique"
                    )
                if final_text is not None and final_text != candidate:
                    raise JournalIntegrityError("Final result changed")
                if final_text is not None and final_rejections != normalized_rejections:
                    raise JournalIntegrityError("Final result rejections changed")
                final_text = candidate
                final_rejections = normalized_rejections
        return JournalReplay(
            committed_messages=tuple(committed_messages),
            committed_turn_messages=tuple(committed_turn_messages),
            pending_turn=pending,
            completed_calls=completed,
            started_without_terminal=started,
            active_elapsed_seconds=active_elapsed,
            committed_turns=tuple(committed_turns),
            final_text=final_text,
            final_rejections=final_rejections,
            provider_attempts=provider_attempts,
            model_turns=model_turns,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            all_usage_available=all_usage_available,
            context_eviction_markers=context_eviction_markers,
            context_compaction=context_compaction,
            max_compaction_generation=max_compaction_generation,
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

    def record_context_idle_eviction(
        self,
        markers: tuple[Mapping[str, Any], ...],
        *,
        active_elapsed_seconds: float,
    ) -> JournalEvent:
        if type(markers) is not tuple or not markers:
            raise JournalError("Context eviction markers must be a non-empty tuple")
        normalized = tuple(_context_eviction_marker(marker) for marker in markers)
        if len({marker["tool_call_id"] for marker in normalized}) != len(normalized):
            raise JournalError("Context eviction marker call IDs must be unique")
        replay = self.replay()
        new_markers: list[dict[str, Any]] = []
        for marker in normalized:
            call_id = marker["tool_call_id"]
            terminal = replay.completed_calls.get(call_id)
            if terminal is None:
                raise JournalIntegrityError(
                    "Context eviction references an unknown Tool Call"
                )
            if (
                terminal.identity.tool_name != marker["tool_name"]
                or "sha256:" + terminal.identity.canonical_arguments_hash
                != marker["arguments_hash"]
                or not terminal.projection.reacquirable
            ):
                raise JournalIntegrityError(
                    "Context eviction Tool Call binding changed"
                )
            existing = replay.context_eviction_markers.get(call_id)
            if existing is not None:
                if existing != marker:
                    raise JournalIntegrityError("Context eviction marker changed")
                continue
            new_markers.append(marker)
        if not new_markers:
            return next(
                event
                for event in reversed(self.read_events())
                if event.assignment_id == self.assignment.assignment_id
                and event.event_type == "context_idle_eviction"
            )
        return self._append(
            "context_idle_eviction",
            {"markers": new_markers},
            active_elapsed_seconds,
        )

    def record_context_compaction_started(
        self,
        *,
        generation: int,
        through_turn: int,
        source_start_turn: int,
        source_end_turn: int,
        trigger: str,
        active_elapsed_seconds: float,
    ) -> JournalEvent:
        payload = _context_compaction_started_payload(
            {
                "generation": generation,
                "through_turn": through_turn,
                "source_start_turn": source_start_turn,
                "source_end_turn": source_end_turn,
                "trigger": trigger,
            }
        )
        replay = self.replay()
        if through_turn not in replay.committed_turns or replay.pending_turn is not None:
            raise JournalIntegrityError(
                "Context compaction requires a committed Turn boundary"
            )
        if generation <= replay.max_compaction_generation:
            raise JournalIntegrityError(
                "Context compaction generation must advance"
            )
        return self._append(
            "context_compaction_started",
            payload,
            active_elapsed_seconds,
        )

    def record_context_compaction_committed(
        self,
        *,
        generation: int,
        through_turn: int,
        source_start_turn: int,
        source_end_turn: int,
        trigger: str,
        summary_path: str,
        summary_hash: str,
        active_elapsed_seconds: float,
    ) -> JournalEvent:
        payload = _context_compaction_committed_payload(
            {
                "generation": generation,
                "through_turn": through_turn,
                "source_start_turn": source_start_turn,
                "source_end_turn": source_end_turn,
                "trigger": trigger,
                "summary_path": summary_path,
                "summary_hash": summary_hash,
            }
        )
        events = tuple(
            event
            for event in self.read_events()
            if event.assignment_id == self.assignment.assignment_id
        )
        matching_start = next(
            (
                event
                for event in reversed(events)
                if event.event_type == "context_compaction_started"
                and event.payload.get("generation") == generation
            ),
            None,
        )
        if matching_start is None:
            raise JournalIntegrityError(
                "Context compaction has no matching start"
            )
        started = _context_compaction_started_payload(matching_start.payload)
        if {key: payload[key] for key in started} != started:
            raise JournalIntegrityError("Context compaction source changed")
        replay = self.replay()
        if (
            replay.context_compaction is not None
            and generation <= replay.context_compaction.generation
        ):
            raise JournalIntegrityError(
                "Context compaction generation must advance"
            )
        return self._append(
            "context_compaction_committed",
            payload,
            active_elapsed_seconds,
        )

    def record_final_result(
        self,
        *,
        final_text: str,
        rejected_findings: tuple[Mapping[str, Any], ...] = (),
        active_elapsed_seconds: float,
    ) -> JournalEvent:
        if type(final_text) is not str or not final_text.strip():
            raise JournalError("final_text must be non-empty")
        if type(rejected_findings) is not tuple:
            raise JournalError("rejected_findings must be a tuple")
        try:
            normalized_rejections = tuple(
                RejectedReviewerFinding.from_dict(item).to_dict()
                for item in rejected_findings
            )
        except ValueError as error:
            raise JournalError("rejected_findings contains an invalid record") from error
        rejection_indices = [
            item["candidate_index"] for item in normalized_rejections
        ]
        if rejection_indices != sorted(set(rejection_indices)):
            raise JournalError(
                "rejected_findings must be ordered by unique candidate index"
            )
        replay = self.replay()
        if replay.final_text is not None:
            if (
                replay.final_text != final_text
                or replay.final_rejections != normalized_rejections
            ):
                raise JournalIntegrityError("Final result changed")
            return next(
                event
                for event in reversed(self.read_events())
                if event.assignment_id == self.assignment.assignment_id
                and event.event_type == "final_result"
            )
        if replay.pending_turn is not None:
            raise JournalIntegrityError("cannot finalize an uncommitted Tool Turn")
        payload: dict[str, Any] = {"final_text": final_text}
        if normalized_rejections:
            payload["rejected_findings"] = list(normalized_rejections)
        return self._append("final_result", payload, active_elapsed_seconds)

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
    "CommittedTurnRecord",
    "CompletedToolCall",
    "ContextCompactionRecord",
    "EXECUTION_JOURNAL_SCHEMA",
    "ExecutionJournal",
    "JournalError",
    "JournalEvent",
    "JournalIntegrityError",
    "JournalReplay",
    "PendingTurn",
    "REVIEWER_RUNTIME_BINDING_SCHEMA",
    "ToolCallIdentity",
]

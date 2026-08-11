from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping

from review_agent.context import reviewer_tool_schemas_v2
from review_agent.diff_artifact import DiffArtifactIndex
from review_agent.global_memory import GlobalMemorySnapshot
from review_agent.review_policy import (
    DeveloperReviewPolicy,
    build_reviewer_system_prompt,
    project_system_rule_block,
)
from review_agent.review_protocol import (
    IntentPacket,
    ReviewRequest,
    ReviewerAssignment,
)


_PR_ID = re.compile(r"\APR-[0-9a-f]{64}\Z")
_SNAPSHOT_ID = re.compile(r"\AS-[0-9a-f]{64}\Z")
_ARTIFACT_ID = re.compile(r"\AA-[0-9a-f]{64}\Z")
_ASSIGNMENT_ID = re.compile(r"\AASG-[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"\A[0-9a-f]{40,64}\Z")
_HUNK_REF = re.compile(r"\A(?P<path>.+)#hunk-(?P<index>[0-9]+)\Z")


class ReviewerContextError(ValueError):
    pass


def _text(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ReviewerContextError(
            f"{field_name} must be canonical non-empty text"
        )
    if "\x00" in value:
        raise ReviewerContextError(
            f"{field_name} contains an unsafe control character"
        )
    return value


@dataclass(frozen=True)
class AvailableArtifact:
    artifact_id: str
    kind: str
    description: str
    assignment_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.artifact_id) is not str or _ARTIFACT_ID.fullmatch(
            self.artifact_id
        ) is None:
            raise ReviewerContextError("artifact_id is invalid")
        _text(self.kind, "kind")
        _text(self.description, "description")
        if type(self.assignment_ids) is not tuple:
            raise ReviewerContextError("assignment_ids must be a tuple")
        for assignment_id in self.assignment_ids:
            if type(assignment_id) is not str or _ASSIGNMENT_ID.fullmatch(
                assignment_id
            ) is None:
                raise ReviewerContextError("assignment_ids contains an invalid ID")
        if len(self.assignment_ids) != len(set(self.assignment_ids)):
            raise ReviewerContextError("assignment_ids must not contain duplicates")

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "description": self.description,
        }


@dataclass(frozen=True)
class DiffFitPolicy:
    target_initial_tokens: int = 550_000
    estimated_chars_per_token: float = 4.0

    def __post_init__(self) -> None:
        if type(self.target_initial_tokens) is not int or self.target_initial_tokens <= 0:
            raise ReviewerContextError("target_initial_tokens must be positive")
        value = self.estimated_chars_per_token
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ReviewerContextError(
                "estimated_chars_per_token must be a positive finite number"
            )

    def estimate_tokens(self, character_count: int) -> int:
        if type(character_count) is not int or character_count < 0:
            raise ReviewerContextError("character_count must be non-negative")
        return int(math.ceil(character_count / self.estimated_chars_per_token))

    def to_dict(self) -> dict[str, object]:
        return {
            "target_initial_tokens": self.target_initial_tokens,
            "estimated_chars_per_token": float(self.estimated_chars_per_token),
        }


@dataclass(frozen=True)
class ReviewerContextInput:
    pr_id: str
    snapshot_id: str
    base_sha: str
    head_sha: str
    request: ReviewRequest
    developer_policy: DeveloperReviewPolicy
    global_memory: GlobalMemorySnapshot
    intent: IntentPacket
    assignment: ReviewerAssignment
    quality_summary: Mapping[str, Any]
    changed_symbols: tuple[Mapping[str, Any], ...]
    diff_bytes: bytes
    diff_index: DiffArtifactIndex
    diff_artifact_id: str
    available_artifacts: tuple[AvailableArtifact, ...]
    model: str
    diff_fit_policy: DiffFitPolicy = DiffFitPolicy()

    def __post_init__(self) -> None:
        if type(self.pr_id) is not str or _PR_ID.fullmatch(self.pr_id) is None:
            raise ReviewerContextError("pr_id is invalid")
        if type(self.snapshot_id) is not str or _SNAPSHOT_ID.fullmatch(
            self.snapshot_id
        ) is None:
            raise ReviewerContextError("snapshot_id is invalid")
        for field_name, value in (
            ("base_sha", self.base_sha),
            ("head_sha", self.head_sha),
        ):
            if type(value) is not str or _GIT_SHA.fullmatch(value) is None:
                raise ReviewerContextError(f"{field_name} is invalid")
        if type(self.request) is not ReviewRequest:
            raise ReviewerContextError("request must be a ReviewRequest")
        if not isinstance(self.developer_policy, DeveloperReviewPolicy):
            raise ReviewerContextError(
                "developer_policy must be DeveloperReviewPolicy"
            )
        if not isinstance(self.global_memory, GlobalMemorySnapshot):
            raise ReviewerContextError(
                "global_memory must be a frozen GlobalMemorySnapshot"
            )
        if type(self.intent) is not IntentPacket:
            raise ReviewerContextError("intent must be an IntentPacket")
        if type(self.assignment) is not ReviewerAssignment:
            raise ReviewerContextError(
                "assignment must be a ReviewerAssignment"
            )
        if self.assignment.snapshot_id != self.snapshot_id:
            raise ReviewerContextError(
                "Assignment Snapshot binding does not match context"
            )
        _json_mapping(self.quality_summary, "quality_summary")
        if type(self.changed_symbols) is not tuple:
            raise ReviewerContextError("changed_symbols must be a tuple")
        for index, symbol in enumerate(self.changed_symbols):
            _json_mapping(symbol, f"changed_symbols[{index}]")
        if type(self.diff_bytes) is not bytes:
            raise ReviewerContextError("diff_bytes must be bytes")
        try:
            self.diff_bytes.decode("utf-8", "strict")
        except UnicodeError as error:
            raise ReviewerContextError("Diff must be valid UTF-8") from error
        if not isinstance(self.diff_index, DiffArtifactIndex):
            raise ReviewerContextError(
                "diff_index must be a DiffArtifactIndex"
            )
        if (
            self.diff_index.snapshot_id != self.snapshot_id
            or self.diff_index.base_sha != self.base_sha
            or self.diff_index.head_sha != self.head_sha
        ):
            raise ReviewerContextError(
                "DiffArtifact Snapshot binding does not match context"
            )
        if type(self.diff_artifact_id) is not str or _ARTIFACT_ID.fullmatch(
            self.diff_artifact_id
        ) is None:
            raise ReviewerContextError("diff_artifact_id is invalid")
        if self.diff_index.patch_artifact_id != self.diff_artifact_id:
            raise ReviewerContextError("Diff Artifact ID binding does not match")
        if (
            self.diff_index.diff_size_bytes != len(self.diff_bytes)
            or self.diff_index.diff_sha256
            != hashlib.sha256(self.diff_bytes).hexdigest()
        ):
            raise ReviewerContextError("Diff content binding does not match index")
        if type(self.available_artifacts) is not tuple or any(
            type(artifact) is not AvailableArtifact
            for artifact in self.available_artifacts
        ):
            raise ReviewerContextError(
                "available_artifacts must be a tuple of AvailableArtifact values"
            )
        _text(self.model, "model")
        if not isinstance(self.diff_fit_policy, DiffFitPolicy):
            raise ReviewerContextError("diff_fit_policy must be DiffFitPolicy")


@dataclass(frozen=True)
class ReviewerInvocationV2:
    system: str
    tools: tuple[dict[str, Any], ...]
    messages: tuple[dict[str, str], ...]
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        _text(self.system, "system")
        if type(self.tools) is not tuple:
            raise ReviewerContextError("tools must be a tuple")
        if type(self.messages) is not tuple or not self.messages:
            raise ReviewerContextError("messages must be a non-empty tuple")
        for message in self.messages:
            if type(message) is not dict or set(message) != {"role", "content"}:
                raise ReviewerContextError("message schema is invalid")
            if message["role"] not in {"user", "assistant", "tool"}:
                raise ReviewerContextError("message role is invalid")
            _text(message["content"], "message.content")
        _json_mapping(self.parameters, "parameters")


def canonical_pinned_context_bytes_v2(
    invocation: ReviewerInvocationV2,
) -> bytes:
    """Return the immutable Reviewer input projection used across compactions."""

    if not isinstance(invocation, ReviewerInvocationV2):
        raise ReviewerContextError("invocation must be ReviewerInvocationV2")
    try:
        return json.dumps(
            {
                "system": invocation.system,
                "tools": list(invocation.tools),
                "messages": list(invocation.messages),
                "parameters": invocation.parameters,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ReviewerContextError(
            "Pinned Reviewer context must be canonical JSON"
        ) from error


def _json_mapping(value: object, field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise ReviewerContextError(f"{field_name} must be a mapping")
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ReviewerContextError(
            f"{field_name} must be JSON serializable"
        ) from error


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )


def _conversation(request: ReviewRequest) -> str:
    lines = ["<UserConversation>"]
    for message in request.conversation:
        lines.extend((f"[{message.speaker.value}]", message.content, ""))
    if lines[-1] == "":
        lines.pop()
    lines.append("</UserConversation>")
    return "\n".join(lines)


def _relevant_symbols(value: ReviewerContextInput) -> list[dict[str, Any]]:
    files = set(value.assignment.targets.files)
    symbols = set(value.assignment.targets.symbols)
    relevant: list[dict[str, Any]] = []
    for raw in value.changed_symbols:
        symbol = dict(raw)
        path = symbol.get("path")
        qualified = symbol.get("qualified_name")
        ref = (
            f"{path}::{qualified}"
            if type(path) is str and type(qualified) is str
            else None
        )
        if path in files or ref in symbols:
            if ref is not None:
                symbol["target_ref"] = ref
            relevant.append(symbol)
    return relevant


def _visible_artifacts(value: ReviewerContextInput) -> list[dict[str, str]]:
    assignment_id = value.assignment.assignment_id
    return [
        artifact.to_dict()
        for artifact in value.available_artifacts
        if not artifact.assignment_ids or assignment_id in artifact.assignment_ids
    ]


def _relevant_diff(value: ReviewerContextInput) -> str:
    pieces: list[tuple[str, bytes]] = []
    files = {entry.path: entry for entry in value.diff_index.files}
    for reference in value.assignment.targets.hunks:
        match = _HUNK_REF.fullmatch(reference)
        if match is None:
            continue
        file_entry = files.get(match.group("path"))
        if file_entry is None:
            continue
        hunk_index = int(match.group("index"))
        hunk = next(
            (item for item in file_entry.hunks if item.hunk_index == hunk_index),
            None,
        )
        if hunk is not None:
            pieces.append(
                (
                    reference,
                    value.diff_bytes[hunk.byte_start : hunk.byte_end],
                )
            )
    if not pieces:
        for path in value.assignment.targets.files:
            file_entry = files.get(path)
            if file_entry is not None:
                pieces.append(
                    (
                        f"file:{path}",
                        value.diff_bytes[
                            file_entry.byte_start : file_entry.byte_end
                        ],
                    )
                )
    rendered: list[str] = []
    for reference, content in pieces:
        rendered.extend(
            (
                f"<DiffFragment ref={json.dumps(reference)}>",
                content.decode("utf-8", "strict"),
                "</DiffFragment>",
            )
        )
    return "\n".join(rendered) if rendered else "(No inline fragment selected.)"


def _code_changes(
    value: ReviewerContextInput,
    *,
    fixed_character_count: int,
) -> tuple[str, str, int]:
    full_diff = value.diff_bytes.decode("utf-8", "strict")
    full_block = (
        f'<CodeChanges mode="full" artifact_id="{value.diff_artifact_id}">\n'
        + full_diff
        + "\n</CodeChanges>"
    )
    policy = value.diff_fit_policy
    full_tokens = policy.estimate_tokens(fixed_character_count + len(full_block))
    if full_tokens <= policy.target_initial_tokens:
        return "full", full_block, full_tokens

    indexed_block = (
        f'<CodeChanges mode="indexed" artifact_id="{value.diff_artifact_id}">\n'
        "<DiffIndex>\n"
        + _json(value.diff_index.to_dict())
        + "\n</DiffIndex>\n<RelevantDiff>\n"
        + _relevant_diff(value)
        + "\n</RelevantDiff>\n</CodeChanges>"
    )
    indexed_tokens = policy.estimate_tokens(
        fixed_character_count + len(indexed_block)
    )
    return "indexed", indexed_block, indexed_tokens


def build_reviewer_invocation_v2(
    value: ReviewerContextInput,
) -> ReviewerInvocationV2:
    if not isinstance(value, ReviewerContextInput):
        raise ReviewerContextError("value must be ReviewerContextInput")

    system = build_reviewer_system_prompt(value.developer_policy)
    tools = reviewer_tool_schemas_v2(value.assignment.permissions)
    tools = tuple(dict(tool) for tool in tools)
    for tool in tools:
        if tool["name"] == "query_project_memory":
            tool["parameters"]["properties"]["assignment_id"]["const"] = (
                value.assignment.assignment_id
            )

    identity = {
        "pr_id": value.pr_id,
        "snapshot_id": value.snapshot_id,
        "base_sha": value.base_sha,
        "head_sha": value.head_sha,
    }
    non_code_sections = (
        "<ReviewIdentity>\n"
        + _json(identity)
        + "\n</ReviewIdentity>\n\n"
        + _conversation(value.request)
        + "\n\n<{{system_rule}}>\n"
        + project_system_rule_block(
            value.global_memory, value.developer_policy
        )
        + "\n</{{system_rule}}>\n\n<IntentPacket>\n"
        + _json(value.intent.to_dict())
        + "\n</IntentPacket>\n\n<Assignment>\n"
        + _json(value.assignment.to_dict())
        + "\n</Assignment>\n\n<PreflightResults>\n"
        + _json(
            {
                "quality_gate": dict(value.quality_summary),
                "changed_symbols": _relevant_symbols(value),
            }
        )
        + "\n</PreflightResults>\n\n"
    )
    artifacts_section = (
        "\n\n<AvailableArtifacts>\n"
        + _json(_visible_artifacts(value))
        + "\n</AvailableArtifacts>"
    )
    base_parameters = {
        "model": value.model,
        "reasoning_effort": "medium",
        "temperature": 0,
        "tool_choice": "auto" if tools else "none",
        "response_schema": "reviewer_output_v2",
    }
    fixed_character_count = sum(
        len(part)
        for part in (
            system,
            _json(list(tools)),
            _json(base_parameters),
            non_code_sections,
            artifacts_section,
        )
    )
    diff_mode, code_changes, estimated_tokens = _code_changes(
        value,
        fixed_character_count=fixed_character_count,
    )
    message = non_code_sections + code_changes + artifacts_section
    parameters = {
        **base_parameters,
        "context_window": {
            "target_initial_tokens": value.diff_fit_policy.target_initial_tokens,
            "estimated_initial_tokens": estimated_tokens,
            "diff_mode": diff_mode,
        },
    }
    return ReviewerInvocationV2(
        system=system,
        tools=tools,
        messages=({"role": "user", "content": message},),
        parameters=parameters,
    )


__all__ = [
    "AvailableArtifact",
    "DiffFitPolicy",
    "ReviewerContextError",
    "ReviewerContextInput",
    "ReviewerInvocationV2",
    "build_reviewer_invocation_v2",
    "canonical_pinned_context_bytes_v2",
]

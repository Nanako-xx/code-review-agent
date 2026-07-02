from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ModelResponseKind(str, Enum):
    TOOL_CALLS = "tool_calls"
    FINAL = "final"
    INVALID = "invalid"


@dataclass(frozen=True)
class ModelToolSpec:
    name: str
    description: str
    parameters_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelToolCall:
    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelToolResult:
    call_id: str
    tool_name: str
    content: str
    observation_ids: list[str] = field(default_factory=list)
    is_error: bool = False


@dataclass(frozen=True)
class ModelTurnRequest:
    system: str
    tools: list[ModelToolSpec]
    messages: list[dict[str, Any]]
    tool_results: list[ModelToolResult]
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ModelTurnResponse:
    kind: ModelResponseKind
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    final_text: str | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    provider_name: str = "unknown"
    model: str = "unknown"


def model_turn_response_to_dict(response: ModelTurnResponse) -> dict[str, Any]:
    payload = asdict(response)
    payload["kind"] = response.kind.value
    return payload

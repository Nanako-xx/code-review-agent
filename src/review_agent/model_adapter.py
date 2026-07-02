from __future__ import annotations

from typing import Callable, Protocol, Union

from review_agent.model_protocol import ModelResponseKind, ModelTurnRequest, ModelTurnResponse


class ModelAdapter(Protocol):
    provider_name: str

    def complete_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        raise NotImplementedError


ScriptItem = Union[ModelTurnResponse, Callable[[ModelTurnRequest], ModelTurnResponse]]


class FakeToolCallingAdapter:
    provider_name = "fake-tool-calling"

    def __init__(self, script: list[ScriptItem]):
        self._script = list(script)
        self.requests: list[ModelTurnRequest] = []

    def complete_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        self.requests.append(request)
        if not self._script:
            return ModelTurnResponse(
                kind=ModelResponseKind.INVALID,
                error="fake adapter script exhausted",
                provider_name=self.provider_name,
                model="fake-tool-model",
            )
        item = self._script.pop(0)
        response = item(request) if callable(item) else item
        return ModelTurnResponse(
            kind=response.kind,
            tool_calls=response.tool_calls,
            final_text=response.final_text,
            error=response.error,
            raw=response.raw,
            provider_name=self.provider_name,
            model=response.model if response.model != "unknown" else "fake-tool-model",
        )

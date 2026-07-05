import json

from review_agent.model_adapter import (
    FakeToolCallingAdapter,
    OpenAICompatibleConfig,
    OpenAICompatibleToolAdapter,
)
from review_agent.model_protocol import (
    ModelResponse,
    ModelResponseKind,
    ModelToolCall,
    ModelToolSpec,
    ModelTurnRequest,
    ModelTurnResponse,
)


def make_request(tool_results=None):
    return ModelTurnRequest(
        system="system",
        tools=[],
        messages=[{"role": "user", "content": "Review change"}],
        tool_results=tool_results or [],
        parameters={"trace_id": "review-1-reviewer-0"},
    )


def test_model_response_lives_in_model_protocol():
    response = ModelResponse(
        content='{"status": "completed"}',
        provider_name="adapter",
        model="review-model",
        raw={"trace_id": "trace-1"},
    )

    assert response.provider_name == "adapter"
    assert response.raw["trace_id"] == "trace-1"


def test_openai_compatible_config_lives_in_model_adapter():
    assert OpenAICompatibleConfig.__module__ == "review_agent.model_adapter"


def test_fake_adapter_returns_scripted_tool_call():
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[ModelToolCall("call-1", "compare_base_head", {"path": "app.py"})],
            )
        ]
    )

    response = adapter.complete_turn(make_request())

    assert response.kind is ModelResponseKind.TOOL_CALLS
    assert response.tool_calls[0].tool_name == "compare_base_head"
    assert response.provider_name == "fake-tool-calling"


def test_fake_adapter_can_compute_final_response_from_request():
    def final_response(request):
        observation_id = request.tool_results[0].observation_ids[0]
        return ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text=json.dumps(
                {
                    "contract_assessments": [
                        {
                            "contract": "regression_safety",
                            "status": "covered",
                            "summary": "Checked diff observation.",
                            "evidence_refs": [observation_id],
                        }
                    ],
                    "confirmed_findings": [],
                    "rejected_hypotheses": [],
                    "uncertainties": [],
                    "observation_refs": [observation_id],
                    "investigation_summary": "Used tool observation.",
                    "status": "completed",
                }
            ),
        )

    adapter = FakeToolCallingAdapter(script=[final_response])
    response = adapter.complete_turn(
        make_request(
            tool_results=[
                type(
                    "Result",
                    (),
                    {
                        "observation_ids": ["O-abc"],
                    },
                )()
            ]
        )
    )

    assert response.kind is ModelResponseKind.FINAL
    assert "O-abc" in response.final_text


def test_openai_compatible_adapter_converts_tool_call_response():
    captured = {}

    def transport(url, headers, payload, timeout_seconds):
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "read_range",
                                    "arguments": (
                                        '{"path": "app.py", "revision": "head", '
                                        '"line_start": 1, "line_end": 10}'
                                    ),
                                },
                            }
                        ]
                    }
                }
            ]
        }

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    request = ModelTurnRequest(
        system="system",
        tools=[
            ModelToolSpec(
                name="read_range",
                description="Read range",
                parameters_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        ],
        messages=[{"role": "user", "content": "Review"}],
        tool_results=[],
        parameters={"max_output_tokens": 1000, "temperature": 0},
    )

    response = adapter.complete_turn(request)

    assert captured["payload"]["tools"][0]["function"]["name"] == "read_range"
    assert response.kind is ModelResponseKind.TOOL_CALLS
    assert response.tool_calls[0].tool_name == "read_range"
    assert response.tool_calls[0].arguments["path"] == "app.py"


def test_openai_compatible_adapter_converts_final_text_response():
    def transport(url, headers, payload, timeout_seconds):
        return {"choices": [{"message": {"content": '{"status": "partial"}'}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )

    response = adapter.complete_turn(make_request())

    assert response.kind is ModelResponseKind.FINAL
    assert response.final_text == '{"status": "partial"}'
    assert response.provider_name == "openai-compatible"

from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelToolResult,
    ModelToolSpec,
    ModelTurnRequest,
    ModelTurnResponse,
    model_turn_response_to_dict,
)


def test_model_protocol_serializes_tool_call_response():
    response = ModelTurnResponse(
        kind=ModelResponseKind.TOOL_CALLS,
        tool_calls=[
            ModelToolCall(
                call_id="call-1",
                tool_name="read_range",
                arguments={"path": "app.py", "revision": "head", "line_start": 1, "line_end": 20},
            )
        ],
        raw={"provider": "fake"},
    )

    payload = model_turn_response_to_dict(response)

    assert payload["kind"] == "tool_calls"
    assert payload["tool_calls"][0]["tool_name"] == "read_range"
    assert payload["tool_calls"][0]["arguments"]["path"] == "app.py"


def test_model_turn_request_carries_tools_and_tool_results():
    request = ModelTurnRequest(
        system="system",
        tools=[
            ModelToolSpec(
                name="search_code",
                description="Search code",
                parameters_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ],
        messages=[{"role": "user", "content": "Review this change"}],
        tool_results=[
            ModelToolResult(
                call_id="call-1",
                tool_name="search_code",
                content="O-123: result summary",
                observation_ids=["O-123"],
            )
        ],
        parameters={"trace_id": "review-1-reviewer-0"},
    )

    assert request.tools[0].name == "search_code"
    assert request.tool_results[0].observation_ids == ["O-123"]

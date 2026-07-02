import json

from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
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

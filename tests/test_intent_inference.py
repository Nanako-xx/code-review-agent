from __future__ import annotations

import json

import pytest

from review_agent.intent_inference import (
    INTENT_INFERENCE_SYSTEM_PROMPT,
    IntentInferenceParseError,
    intent_inference_run_to_dict,
    parse_intent_inference_result,
    run_intent_inference,
)
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelTurnResponse,
)
from review_agent.observations import ObservationStore
from review_agent.tool_gateway import ToolGateway
from tests.conftest import run_git


def test_intent_inference_runs_legal_tool_loop_with_bound_context(git_repo, tmp_path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "README.md").write_text(
        "# Requirement\nThe add operation must preserve integer addition.\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "README.md")
    run_git(git_repo, "commit", "-m", "document add behavior")
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path / "intent-observations")
    gateway = ToolGateway(git_repo, base, head, store)

    def final_response(request):
        observation_id = request.tool_results[-1].observation_ids[0]
        return ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text=_result_json(
                origin="repository_document",
                source_refs=["README.md:2"],
                evidence_refs=[observation_id],
            ),
        )

    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[
                    ModelToolCall(
                        "read-doc",
                        "read_range",
                        {
                            "path": "README.md",
                            "revision": "head",
                            "line_start": 1,
                            "line_end": 2,
                        },
                    )
                ],
            ),
            final_response,
        ]
    )

    run = _run(adapter, gateway, base=base, head=head)

    assert run.status == "completed"
    assert run.result.candidates[0].origin == "repository_document"
    assert run.result.candidates[0].source == "explicit"
    assert run.trace.tool_call_count == 1
    assert run.trace.turns[0].tool_results[0].observation_ids
    assert INTENT_INFERENCE_SYSTEM_PROMPT == adapter.requests[0].system
    assert "Intent Analyst" in adapter.requests[0].system
    assert "read-only" in adapter.requests[0].system
    assert "untrusted" in adapter.requests[0].system
    assert "Never report a Finding" in adapter.requests[0].system
    assert "Never claim that implementation code" in adapter.requests[0].system
    assert adapter.requests[0].parameters["response_schema"] == "intent_inference_result_v1"
    assert {tool.name for tool in adapter.requests[0].tools} == {
        "read_range",
        "compare_base_head",
        "search_code",
        "list_symbols",
        "inspect_symbol",
        "find_references",
        "read_commit_messages",
    }
    context = json.loads(adapter.requests[0].messages[0]["content"])
    assert context["resolved_revisions"] == {"base": base, "head": head}
    assert context["deterministic_request_summary"] == "Review request #42"
    assert context["change_summary"] == "README.md was added"
    assert context["existing_explicit_intent"] == {"goal": "Preserve behavior"}
    assert context["missing_fields"] == ["acceptance_criteria", "scope"]


def test_intent_inference_downgrades_false_explicit_document_claim(git_repo, tmp_path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path / "intent-false-explicit")
    gateway = ToolGateway(git_repo, base, base, store)
    implementation = gateway.execute(
        "read_range",
        {"path": "app.py", "revision": "head", "line_start": 1, "line_end": 2},
    )
    evidence_id = implementation.observation_ids[0]
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=_result_json(
                    origin="repository_document",
                    source_refs=["app.py:1"],
                    evidence_refs=[evidence_id],
                ),
            )
        ]
    )

    run = _run(
        adapter,
        gateway,
        base=base,
        head=base,
        initial_observation_summaries=store.summaries_by_id(),
    )

    candidate = run.result.candidates[0]
    assert run.status == "partial"
    assert candidate.origin == "llm_inference"
    assert candidate.source == "inferred"
    assert any("did not match its Observation source/path" in item for item in run.trace.deficiencies)


def test_intent_inference_strips_unauthorized_evidence_and_records_deficiency(
    git_repo,
    tmp_path,
):
    head = run_git(git_repo, "rev-parse", "HEAD")
    gateway = ToolGateway(
        git_repo,
        head,
        head,
        ObservationStore(tmp_path / "intent-unauthorized"),
    )
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=_result_json(
                    origin="llm_inference",
                    source_refs=["app.py"],
                    evidence_refs=["O-not-from-this-store"],
                ),
            )
        ]
    )

    run = _run(adapter, gateway, base=head, head=head)

    assert run.status == "partial"
    assert run.result.candidates[0].evidence_refs == []
    assert any("unauthorized evidence_refs" in item for item in run.trace.deficiencies)
    assert any("Runtime validation deficiency" in item for item in run.result.uncertainties)


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps(
            {
                "candidates": [],
                "uncertainties": [],
                "summary": "Unexpected field is rejected.",
                "status": "completed",
            }
        ),
        json.dumps(
            {
                "candidates": [
                    {
                        "field": "design_decision",
                        "value": "No",
                        "origin": "llm_inference",
                        "confidence": "high",
                        "source_refs": [],
                        "evidence_refs": [],
                        "rationale": "Invalid field.",
                        "conclusion_impact": "material",
                    }
                ],
                "uncertainties": [],
                "summary": "Invalid enum.",
            }
        ),
        json.dumps(
            {
                "candidates": [
                    {
                        "field": "goal",
                        "value": "Preserve behavior.",
                        "origin": "llm_inference",
                        "confidence": "high",
                        "source_refs": [],
                        "evidence_refs": [],
                        "rationale": "Candidate goal.",
                        "conclusion_impact": "important",
                    }
                ],
                "uncertainties": [],
                "summary": "Invalid conclusion impact.",
            }
        ),
    ],
)
def test_intent_inference_strict_parser_rejects_invalid_output(content):
    with pytest.raises(IntentInferenceParseError):
        parse_intent_inference_result(content)


def test_intent_inference_parse_error_returns_auditable_failed_run(git_repo, tmp_path):
    head = run_git(git_repo, "rev-parse", "HEAD")
    gateway = ToolGateway(
        git_repo,
        head,
        head,
        ObservationStore(tmp_path / "intent-parse-error"),
    )
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="not json",
                raw={"bad": True},
                model="bad-intent-model",
            )
        ]
    )

    run = _run(adapter, gateway, base=head, head=head, max_turns=1)
    payload = intent_inference_run_to_dict(run)

    assert run.status == "failed"
    assert run.result.candidates == []
    assert "final response parse failed" in run.result.uncertainties[0]
    assert run.trace.turns[0].error == run.result.uncertainties[0]
    assert payload["status"] == "failed"
    assert payload["response_text"] == "not json"
    assert payload["raw_response"] == {"bad": True}
    json.dumps(payload)


def test_intent_inference_tool_budget_exhaustion_returns_partial(git_repo, tmp_path):
    head = run_git(git_repo, "rev-parse", "HEAD")
    gateway = ToolGateway(
        git_repo,
        head,
        head,
        ObservationStore(tmp_path / "intent-tool-budget"),
    )
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[
                    ModelToolCall("read", "read_commit_messages", {})
                ],
            )
        ]
    )

    run = _run(adapter, gateway, base=head, head=head, max_tool_calls=0)

    assert run.status == "partial"
    assert run.trace.tool_call_count == 0
    assert "tool budget exhausted" in run.result.uncertainties


def test_intent_inference_turn_budget_exhaustion_returns_partial(git_repo, tmp_path):
    head = run_git(git_repo, "rev-parse", "HEAD")
    gateway = ToolGateway(
        git_repo,
        head,
        head,
        ObservationStore(tmp_path / "intent-turn-budget"),
    )
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[ModelToolCall("read", "read_commit_messages", {})],
            )
        ]
    )

    run = _run(adapter, gateway, base=head, head=head, max_turns=1)

    assert run.status == "partial"
    assert run.trace.tool_call_count == 1
    assert "turn budget exhausted" in run.result.uncertainties


def test_intent_inference_provider_invalid_returns_failed(git_repo, tmp_path):
    head = run_git(git_repo, "rev-parse", "HEAD")
    gateway = ToolGateway(
        git_repo,
        head,
        head,
        ObservationStore(tmp_path / "intent-provider-invalid"),
    )
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.INVALID,
                error="provider unavailable",
            )
        ]
    )

    run = _run(adapter, gateway, base=head, head=head)

    assert run.status == "failed"
    assert run.result.uncertainties == ["provider unavailable"]


def _run(
    adapter,
    gateway,
    *,
    base,
    head,
    initial_observation_summaries=None,
    max_turns=4,
    max_tool_calls=8,
):
    return run_intent_inference(
        adapter=adapter,
        gateway=gateway,
        resolved_base_revision=base,
        resolved_head_revision=head,
        deterministic_request_summary="Review request #42",
        change_summary="README.md was added",
        explicit_intent={"goal": "Preserve behavior"},
        missing_fields=["acceptance_criteria", "scope"],
        initial_observation_summaries=initial_observation_summaries or {},
        trace_id="intent-test-trace",
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
    )


def _result_json(
    *,
    origin,
    source_refs,
    evidence_refs,
):
    return json.dumps(
        {
            "candidates": [
                {
                    "field": "acceptance_criteria",
                    "value": "The add operation preserves integer addition.",
                    "origin": origin,
                    "confidence": "high",
                    "source_refs": source_refs,
                    "evidence_refs": evidence_refs,
                    "rationale": "This describes the expected behavior.",
                    "conclusion_impact": "material",
                }
            ],
            "uncertainties": [],
            "summary": "Intent candidate extracted.",
        }
    )

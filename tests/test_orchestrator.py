import json

from review_agent.model_adapter import ModelAdapter
from review_agent.model_adapter_factory import ModelAdapterFactory
from review_agent.model_protocol import ModelResponseKind, ModelTurnRequest, ModelTurnResponse
from review_agent.models import Assignment, InitialContext, IntentPacket, IntentSource, IntentStatus
from review_agent.orchestrator import multi_reviewer_run_to_dict, run_multi_reviewer


class RecordingAdapter:
    provider_name = "recording"

    def __init__(self, factory):
        self._factory = factory

    def complete_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        self._factory.trace_ids.append(request.parameters["trace_id"])
        content = request.messages[0]["content"]
        role_line = next(line for line in content.splitlines() if line.startswith("Role: "))
        role = role_line.removeprefix("Role: ")
        self._factory.roles.append(role)
        if self._factory.fail_on_call_number == len(self._factory.trace_ids):
            return ModelTurnResponse(
                kind=ModelResponseKind.INVALID,
                error="provider unavailable",
                provider_name="recording",
                model="recording-model",
            )
        return ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text=json.dumps(
                {
                    "contract_assessments": [],
                    "confirmed_findings": [],
                    "rejected_hypotheses": [],
                    "uncertainties": [f"{role} used fake semantic review."],
                    "observation_refs": ["O-shared"],
                    "investigation_summary": f"{role} finished.",
                    "status": "partial",
                }
            ),
            provider_name="recording",
            model="recording-model",
            raw={"role": role},
        )


class RecordingAdapterFactory:
    def __init__(self, fail_on_call_number=None):
        self.trace_ids = []
        self.roles = []
        self.fail_on_call_number = fail_on_call_number
        self.created_count = 0

    def create(self) -> ModelAdapter:
        self.created_count += 1
        return RecordingAdapter(self)


def make_assignment(role: str) -> Assignment:
    return Assignment(
        role=role,
        mission=f"{role} mission",
        assignment_reason=[f"{role} reason"],
        assigned_contract=["regression_safety"],
        required_checks=["inspect direct observations"],
        initial_context=InitialContext(observation_refs=["O-shared"]),
        max_turns=6,
        max_tool_calls=12,
    )


def make_intent() -> IntentPacket:
    return IntentPacket(
        goal="Review risky change",
        sources={"goal": IntentSource.EXPLICIT},
        status=IntentStatus.PARTIAL,
    )


def test_run_multi_reviewer_runs_every_assignment_with_isolated_traces():
    factory = RecordingAdapterFactory()
    assignments = [make_assignment("Core Reviewer"), make_assignment("Adversarial Reviewer")]

    run = run_multi_reviewer(
        adapter_factory=factory,
        assignments=assignments,
        intent=make_intent(),
        diff_excerpt=["+changed"],
        observations={"O-shared": "shared observation"},
        trace_id_prefix="review-123",
    )

    assert factory.created_count == 2
    assert factory.trace_ids == ["review-123-reviewer-0", "review-123-reviewer-1"]
    assert factory.roles == ["Core Reviewer", "Adversarial Reviewer"]
    assert [item.assignment.role for item in run.executions] == ["Core Reviewer", "Adversarial Reviewer"]
    assert [item.result.status.value for item in run.executions] == ["partial", "partial"]
    assert run.status_counts == {"partial": 2}


def test_multi_reviewer_run_to_dict_contains_artifact_summary():
    run = run_multi_reviewer(
        adapter_factory=RecordingAdapterFactory(),
        assignments=[make_assignment("Core Reviewer")],
        intent=make_intent(),
        diff_excerpt=[],
        observations={"O-shared": "shared observation"},
        trace_id_prefix="review-456",
    )

    payload = multi_reviewer_run_to_dict(run)

    assert payload["reviewer_count"] == 1
    assert payload["status_counts"] == {"partial": 1}
    assert payload["executions"][0]["role"] == "Core Reviewer"
    assert payload["executions"][0]["trace_id"] == "review-456-reviewer-0"
    assert payload["executions"][0]["result"]["investigation_summary"] == "Core Reviewer finished."


def test_run_multi_reviewer_records_failed_execution_without_aborting_remaining_artifacts():
    run = run_multi_reviewer(
        adapter_factory=RecordingAdapterFactory(fail_on_call_number=2),
        assignments=[make_assignment("Core Reviewer"), make_assignment("Adversarial Reviewer")],
        intent=make_intent(),
        diff_excerpt=[],
        observations={"O-shared": "shared observation"},
        trace_id_prefix="review-789",
    )

    payload = multi_reviewer_run_to_dict(run)

    assert payload["reviewer_count"] == 2
    assert payload["status_counts"] == {"partial": 1, "failed": 1}
    assert payload["executions"][0]["result"]["status"] == "partial"
    assert payload["executions"][1]["trace_id"] == "review-789-reviewer-1"
    assert payload["executions"][1]["result"]["status"] == "failed"
    assert "provider unavailable" in payload["executions"][1]["result"]["investigation_summary"]

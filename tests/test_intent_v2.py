from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

from review_agent.intent_inference import (
    IntentInferenceCandidate,
    IntentInferenceResult,
    IntentInferenceRun,
    IntentInferenceTrace,
    run_intent_inference,
)
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import ModelResponseKind, ModelTurnResponse
from review_agent.intent_runtime import IntentRuntime
from review_agent.artifacts import artifact_schema
from review_agent.pr_workspace import PRMetadata, PRWorkspaceStore
from review_agent.review_protocol import (
    ConversationMessage,
    ConversationSpeaker,
    IntentPacket,
    IntentSource,
    IntentVersionEnvelope,
    ReviewRequest,
)
from review_agent.revision import RepositoryIdentity


BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
NEXT_HEAD_SHA = "c" * 40


def _workspace(tmp_path: Path):
    repository = tmp_path / "repo"
    git_common = repository / ".git"
    git_common.mkdir(parents=True)
    identity = RepositoryIdentity(
        canonical_path=str(repository.resolve()),
        git_common_dir=str(git_common.resolve()),
        origin_url=None,
    )
    store = PRWorkspaceStore(tmp_path / "ra")
    workspace = store.create_or_load_workspace(
        store.resolve_pr(identity, "local", "intent-task"),
        PRMetadata(title="Improve retry safety"),
    )
    first = store.create_or_load_snapshot(workspace, BASE_SHA, HEAD_SHA)
    second = store.create_or_load_snapshot(workspace, BASE_SHA, NEXT_HEAD_SHA)
    return store, workspace, first, second


def _request() -> ReviewRequest:
    return ReviewRequest(
        conversation=(
            ConversationMessage(
                speaker=ConversationSpeaker.USER,
                content="Review the retry change.",
            ),
        )
    )


def _inference_run(goal: str) -> IntentInferenceRun:
    return IntentInferenceRun(
        result=IntentInferenceResult(
            candidates=[
                IntentInferenceCandidate(
                    field="goal",
                    value=goal,
                    origin="llm_inference",
                    confidence="medium",
                    source_refs=[],
                    evidence_refs=[],
                    rationale="The changed retry helper implies this objective.",
                    conclusion_impact="material",
                )
            ],
            uncertainties=["The PR description does not state the retry limit."],
            summary="Inferred a retry-safety objective.",
        ),
        trace=IntentInferenceTrace(
            trace_id="intent-trace-1",
            turns=[],
            tool_call_count=0,
            final_status="completed",
            deficiencies=[],
        ),
        provider_name="fake-provider",
        model="fake-intent-model",
        response_text='{"private":"model response"}',
        raw_response={"private_provider_payload": "must stay internal"},
    )


def test_declared_goal_projects_to_an_explicit_three_field_packet(
    tmp_path: Path,
) -> None:
    store, workspace, snapshot, _second = _workspace(tmp_path)
    runtime = IntentRuntime(store)

    version = runtime.resolve(
        workspace,
        snapshot,
        _request(),
        declared_goal="Prevent duplicate retry execution.",
    )

    assert version.version == 1
    assert version.source_snapshot_id == snapshot.snapshot_id
    assert version.packet == IntentPacket(
        goal="Prevent duplicate retry execution.",
        source=IntentSource.EXPLICIT,
        uncertainties=(),
    )
    assert set(version.packet.to_dict()) == {"goal", "source", "uncertainties"}
    assert [field.name for field in fields(IntentPacket)] == [
        "goal",
        "source",
        "uncertainties",
    ]
    assert runtime.load_current_packet(workspace) == version.packet
    assert artifact_schema("intent_version") == "intent_version_envelope_v1"
    assert artifact_schema("intent_analysis_record") == "intent_analysis_record_v2"


def test_model_analysis_projects_only_an_inferred_packet_and_keeps_trace_internal(
    tmp_path: Path,
) -> None:
    store, workspace, snapshot, _second = _workspace(tmp_path)
    runtime = IntentRuntime(store)
    run = _inference_run("Keep retries idempotent after provider failures.")

    version = runtime.resolve(
        workspace,
        snapshot,
        _request(),
        inference_run=run,
    )

    assert version.packet.goal == "Keep retries idempotent after provider failures."
    assert version.packet.source is IntentSource.INFERRED
    packet_json = version.packet.to_json()
    assert "private_provider_payload" not in packet_json
    assert "trace_id" not in packet_json
    assert "raw_response" not in packet_json
    for forbidden in (
        "rules",
        "acceptance_criteria",
        "scope",
        "constraints",
        "status",
        "provenance",
        "clarifications",
    ):
        assert forbidden not in version.packet.to_dict()

    analysis = runtime.load_analysis_record(workspace, version.analysis_record_ref)
    analysis_text = json.dumps(analysis, ensure_ascii=False)
    assert "private_provider_payload" in analysis_text
    assert "intent-trace-1" in analysis_text
    assert analysis["source_snapshot_id"] == snapshot.snapshot_id
    assert analysis["trust_policy"] == "normal"
    assert analysis["model_inference_promoted"] is False


def test_evaluation_policy_promotes_completed_unique_model_goal(
    tmp_path: Path,
) -> None:
    store, workspace, snapshot, _second = _workspace(tmp_path)
    runtime = IntentRuntime(store)

    version = runtime.resolve(
        workspace,
        snapshot,
        _request(),
        inference_run=_inference_run("Keep retries idempotent."),
        trust_policy="evaluation_trust_model",
    )

    assert version.packet.goal == "Keep retries idempotent."
    assert version.packet.source is IntentSource.EXPLICIT
    analysis = runtime.load_analysis_record(workspace, version.analysis_record_ref)
    assert analysis["trust_policy"] == "evaluation_trust_model"
    assert analysis["model_inference_promoted"] is True
    assert analysis["selection_reason"] == "evaluation_model_inference_promoted"


def test_evaluation_promotes_recovered_goal_and_ignores_legacy_candidates(
    tmp_path: Path,
) -> None:
    class GoalOnlyGateway:
        base_revision = BASE_SHA
        head_revision = HEAD_SHA

        @staticmethod
        def intent_evidence():
            return ()

        @staticmethod
        def execute(_tool_name, _arguments):
            raise AssertionError("This inference script does not call tools")

    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="not json",
            ),
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=json.dumps(
                    {
                        "candidates": [
                            {
                                "field": "goal",
                                "value": "Avoid exposing credentials in MongoDB logs.",
                                "origin": "llm_inference",
                                "confidence": "high",
                                "source_refs": [],
                                "evidence_refs": [],
                                "rationale": "The logging changes imply this objective.",
                                "conclusion_impact": "blocking",
                            },
                            {
                                "field": "acceptance_criteria",
                                "value": "Legacy output that IntentPacket v2 did not request.",
                                "origin": "repository_test",
                                "confidence": "high",
                                "source_refs": ["missing_test.go"],
                                "evidence_refs": ["O-not-authorized"],
                                "rationale": "This candidate has deliberately invalid provenance.",
                                "conclusion_impact": "material",
                            },
                        ],
                        "uncertainties": [],
                        "summary": "One reliable goal was inferred.",
                    }
                ),
            ),
        ]
    )
    run = run_intent_inference(
        adapter,
        GoalOnlyGateway(),
        deterministic_request_summary="Review the immutable changes.",
        change_summary="MongoDB logging changed.",
        explicit_intent={},
        missing_fields=("goal",),
        initial_observation_summaries={},
        trace_id="recovered-evaluation-intent",
        resolved_base_revision=BASE_SHA,
        resolved_head_revision=HEAD_SHA,
        goal_only=True,
    )
    store, workspace, snapshot, _second = _workspace(tmp_path)

    version = IntentRuntime(store).resolve(
        workspace,
        snapshot,
        _request(),
        inference_run=run,
        trust_policy="evaluation_trust_model",
    )

    assert run.status == "completed"
    assert run.trace.deficiencies == []
    assert run.trace.turns[0].error.startswith("final response parse failed")
    assert run.trace.turns[1].error.startswith(
        "Runtime ignored non-requested candidates"
    )
    assert version.packet.goal == "Avoid exposing credentials in MongoDB logs."
    assert version.packet.source is IntentSource.EXPLICIT
    analysis = IntentRuntime(store).load_analysis_record(
        workspace,
        version.analysis_record_ref,
    )
    assert analysis["model_inference_promoted"] is True
    assert analysis["selection_reason"] == "evaluation_model_inference_promoted"


def test_evaluation_policy_promotes_partial_goal_but_not_conflicting_analysis(
    tmp_path: Path,
) -> None:
    store, workspace, snapshot, _second = _workspace(tmp_path)
    runtime = IntentRuntime(store)
    base_run = _inference_run("Keep retries idempotent.")
    partial_run = IntentInferenceRun(
        result=base_run.result,
        trace=IntentInferenceTrace(
            trace_id="partial-evaluation-intent",
            turns=[],
            tool_call_count=0,
            final_status="partial",
            deficiencies=["provider response was incomplete"],
        ),
        provider_name="fake-provider",
        model="fake-intent-model",
    )

    partial = runtime.resolve(
        workspace,
        snapshot,
        _request(),
        inference_run=partial_run,
        trust_policy="evaluation_trust_model",
    )

    assert partial.packet.source is IntentSource.EXPLICIT
    assert runtime.load_analysis_record(
        workspace, partial.analysis_record_ref
    )["model_inference_promoted"] is True

    store2, workspace2, snapshot2, _second2 = _workspace(tmp_path / "conflict")
    conflicting = IntentInferenceRun(
        result=IntentInferenceResult(
            candidates=[
                *_inference_run("Keep retries idempotent.").result.candidates,
                IntentInferenceCandidate(
                    field="goal",
                    value="Remove retries entirely.",
                    origin="llm_inference",
                    confidence="medium",
                    source_refs=[],
                    evidence_refs=[],
                    rationale="A conflicting interpretation was produced.",
                    conclusion_impact="material",
                ),
            ],
            uncertainties=[],
            summary="Conflicting goals were produced.",
        ),
        trace=IntentInferenceTrace(
            trace_id="conflicting-evaluation-intent",
            turns=[],
            tool_call_count=0,
            final_status="completed",
        ),
        provider_name="fake-provider",
        model="fake-intent-model",
    )

    unresolved = IntentRuntime(store2).resolve(
        workspace2,
        snapshot2,
        _request(),
        inference_run=conflicting,
        trust_policy="evaluation_trust_model",
    )

    assert unresolved.packet.goal is None
    assert unresolved.packet.source is None
    assert any("conflicting" in item for item in unresolved.packet.uncertainties)


def test_evaluation_policy_does_not_invent_explicit_when_model_has_no_goal(
    tmp_path: Path,
) -> None:
    store, workspace, snapshot, _second = _workspace(tmp_path)
    run = IntentInferenceRun(
        result=IntentInferenceResult(
            candidates=[],
            uncertainties=["The changed behavior has no reliable objective."],
            summary="No reliable goal was found.",
        ),
        trace=IntentInferenceTrace(
            trace_id="no-goal-evaluation-intent",
            turns=[],
            tool_call_count=0,
            final_status="completed",
        ),
        provider_name="fake-provider",
        model="fake-intent-model",
    )

    version = IntentRuntime(store).resolve(
        workspace,
        snapshot,
        _request(),
        inference_run=run,
        trust_policy="evaluation_trust_model",
    )

    assert version.packet.goal is None
    assert version.packet.source is None
    assert any("no reliable goal" in item.casefold() for item in version.packet.uncertainties)


def test_missing_reliable_goal_uses_null_source_with_an_explicit_uncertainty(
    tmp_path: Path,
) -> None:
    store, workspace, snapshot, _second = _workspace(tmp_path)

    version = IntentRuntime(store).resolve(
        workspace,
        snapshot,
        _request(),
    )

    assert version.packet.goal is None
    assert version.packet.source is None
    assert version.packet.uncertainties
    assert "reliable" in version.packet.uncertainties[0]


def test_explicit_intent_continues_to_new_snapshot_as_a_new_create_only_version(
    tmp_path: Path,
) -> None:
    store, workspace, first, second = _workspace(tmp_path)
    runtime = IntentRuntime(store)
    first_version = runtime.resolve(
        workspace,
        first,
        _request(),
        declared_goal="Preserve public retry semantics.",
    )
    first_history = runtime.history_path(workspace, first_version.version)
    original_history = first_history.read_bytes()

    repeated = runtime.resolve(workspace, first, _request())
    continued = runtime.resolve(workspace, second, _request())

    assert repeated == first_version
    assert continued.version == 2
    assert continued.source_snapshot_id == second.snapshot_id
    assert continued.packet == first_version.packet
    assert first_history.read_bytes() == original_history
    assert runtime.history_path(workspace, 2).is_file()
    current = IntentVersionEnvelope.from_json(
        (workspace.path / "Intent" / "current.json").read_bytes()
    )
    assert current == continued
    manifest = json.loads((workspace.path / "manifest.json").read_text("utf-8"))
    assert manifest["current_intent_version"] == 2


def test_inferred_intent_requires_revalidation_on_a_new_snapshot(
    tmp_path: Path,
) -> None:
    store, workspace, first, second = _workspace(tmp_path)
    runtime = IntentRuntime(store)
    inferred = runtime.resolve(
        workspace,
        first,
        _request(),
        inference_run=_inference_run("Preserve retry idempotency."),
    )

    unvalidated = runtime.resolve(workspace, second, _request())

    assert inferred.packet.source is IntentSource.INFERRED
    assert unvalidated.version == 2
    assert unvalidated.packet.goal is None
    assert unvalidated.packet.source is None
    assert any("revalidation" in item for item in unvalidated.packet.uncertainties)


def test_revalidated_inference_creates_a_new_snapshot_bound_version(
    tmp_path: Path,
) -> None:
    store, workspace, first, second = _workspace(tmp_path)
    runtime = IntentRuntime(store)
    runtime.resolve(
        workspace,
        first,
        _request(),
        inferred_goal="Preserve retry idempotency.",
    )

    revalidated = runtime.resolve(
        workspace,
        second,
        _request(),
        inferred_goal="Preserve retry idempotency.",
    )

    assert revalidated.version == 2
    assert revalidated.source_snapshot_id == second.snapshot_id
    assert revalidated.packet.source is IntentSource.INFERRED


def test_evaluation_promoted_model_goal_requires_new_snapshot_revalidation(
    tmp_path: Path,
) -> None:
    store, workspace, first, second = _workspace(tmp_path)
    runtime = IntentRuntime(store)
    promoted = runtime.resolve(
        workspace,
        first,
        _request(),
        inference_run=_inference_run("Preserve retry idempotency."),
        trust_policy="evaluation_trust_model",
    )

    unvalidated = runtime.resolve(
        workspace,
        second,
        _request(),
        trust_policy="evaluation_trust_model",
    )

    assert promoted.packet.source is IntentSource.EXPLICIT
    assert unvalidated.packet.goal is None
    assert unvalidated.packet.source is None
    analysis = runtime.load_analysis_record(
        workspace,
        unvalidated.analysis_record_ref,
    )
    assert analysis["selection_reason"] == (
        "evaluation_model_inference_revalidation_missing"
    )

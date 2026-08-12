from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

from review_agent.intent_inference import (
    IntentInferenceCandidate,
    IntentInferenceResult,
    IntentInferenceRun,
    IntentInferenceTrace,
)
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
    assert artifact_schema("intent_analysis_record") == "intent_analysis_record_v1"


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

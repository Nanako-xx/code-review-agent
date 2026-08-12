from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import run_git

import review_agent.pipeline as pipeline_module
from review_agent.command import main as command_main
from review_agent.checkpoint import CheckpointStore
from review_agent.memory_models import (
    Applicability,
    DurableMemoryRecord,
    ExpiryCondition,
    ExpiryConditionKind,
    GitCommitSourceRef,
    HumanDeclarationSourceRef,
    MemoryCandidate,
    MemoryConfidence,
    MemoryExecutionConfig,
    MemoryKind,
    MemoryMode,
    MemoryScope,
    MemorySnapshot,
    CandidateStatus,
    PolicyEffect,
    PolicyEffectKind,
    Producer,
    ProducerType,
    RecordStatus,
    Sensitivity,
    ValidityPolicy,
    canonical_sha256,
    stable_request_id,
)
from review_agent.memory_identity import build_repository_memory_namespace
from review_agent.memory_lifecycle import (
    CandidateDedupeDecision,
    CandidateDedupeKind,
    CandidateLifecycleResult,
    ExpirySweepResult,
    MemoryLifecycle,
)
from review_agent.memory_store import MemoryStore, MemoryStoreCorruptionError
from review_agent.memory_sources import (
    SourceValidator,
    TrustedCandidateProvenance,
    candidate_authority_resolution_hash,
)
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelTurnResponse,
)
from review_agent.models import (
    IntentOrigin,
    IntentSource,
    QualityGateResult,
    ReviewRequest,
)
from review_agent.pipeline import (
    PHASE_MESSAGES,
    PipelineStageError,
    ReviewPipeline,
    replay_memory_outbox,
)
from review_agent.quality import QualityGateExecution
from review_agent.legacy_resume import ResumeAction, ReviewSessionResumer
from review_agent.revision import RevisionResolver
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import (
    ModelStageConfig,
    ReviewExecutionConfig,
    initial_session_manifest,
)
from review_agent.session_store import SessionStore


def _pipeline(
    git_repo: Path,
    tmp_path: Path,
    *,
    mode: MemoryMode,
    review_id: str,
    required: bool = False,
    project_rules: tuple[str, ...] = (),
    max_context_records: int = 12,
    max_query_results: int = 8,
    max_snapshot_records: int = 2_000,
    reviewer_loop: str = "single-shot",
    memory_curator: ModelStageConfig | None = None,
    adapter_factory_builder=None,
    change_marker: str | None = None,
    fixed_revisions: tuple[str, str] | None = None,
) -> tuple[ReviewPipeline, SessionStore, CheckpointStore, Path]:
    if fixed_revisions is None:
        base = run_git(git_repo, "rev-parse", "HEAD")
        marker = "" if change_marker is None else f"# {change_marker}\n"
        (git_repo / "app.py").write_text(
            "def add(a, b):\n    return a - b\n" + marker,
            encoding="utf-8",
        )
        run_git(git_repo, "add", "app.py")
        run_git(git_repo, "commit", "-m", "change app")
        head = run_git(git_repo, "rev-parse", "HEAD")
    else:
        base, head = fixed_revisions

    resolver = RevisionResolver()
    identity = resolver.repository_identity(git_repo)
    revisions = resolver.resolve_pair(git_repo, base, head)
    memory_root = (tmp_path / "durable-memory").resolve()
    checkpoint_store = CheckpointStore(git_repo, review_id)
    session_store = SessionStore(checkpoint_store.run_dir)
    session_store.create(
        initial_session_manifest(
            review_id=review_id,
            repository=identity,
            revisions=revisions,
            execution=ReviewExecutionConfig(
                reviewer_provider="fake",
                reviewer_model=None,
                reviewer_base_url=None,
                reviewer_api_key_env="REVIEW_AGENT_API_KEY",
                reviewer_mode="single",
                reviewer_loop=reviewer_loop,
                non_interactive=True,
                memory=MemoryExecutionConfig(
                    mode=mode,
                    root_path=str(memory_root),
                    required=required,
                    max_context_records=max_context_records,
                    max_query_results=max_query_results,
                    max_snapshot_records=max_snapshot_records,
                ),
                memory_curator=memory_curator or ModelStageConfig(),
            ),
            now="2026-07-15T00:00:00Z",
        )
    )
    pipeline_kwargs = {}
    if adapter_factory_builder is not None:
        pipeline_kwargs["adapter_factory_builder"] = adapter_factory_builder
    pipeline = ReviewPipeline(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
        request=ReviewRequest(
            repository_path=identity.canonical_path,
            base_revision=base,
            head_revision=head,
            user_intent="Preserve addition semantics",
            project_rules=project_rules,
        ),
        clock=lambda: "2026-07-15T00:00:00Z",
        **pipeline_kwargs,
    )
    return pipeline, session_store, checkpoint_store, memory_root


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


_RECEIPT_CANDIDATE_ID = "MC-" + "1" * 64
_RECEIPT_REQUEST_ID = "REQ-" + "2" * 64


def _persistence_write_result(
    *,
    applied: bool,
    subject_id: str = _RECEIPT_CANDIDATE_ID,
) -> dict[str, object]:
    return {
        "operation": "transition_candidate",
        "subject_id": subject_id,
        "event_id": None,
        "generations": {
            "schema_version": 1,
            "store_schema_version": 2,
            "memory_generation": 3,
            "feedback_generation": 0,
            "knowledge_generation": 0,
        },
        "applied": applied,
    }


def _persistence_receipt(
    *,
    replayed: bool,
    status: str,
    dedupe: str,
    write_results: list[dict[str, object]],
    classified_as_replayed: bool | None = None,
) -> dict[str, object]:
    listed_as_replayed = (
        replayed
        if classified_as_replayed is None
        else classified_as_replayed
    )
    body: dict[str, object] = {
        "schema": "memory_persistence_receipt_v1",
        "success": True,
        "review_id": "review-receipt-matrix",
        "repository_key": "3" * 64,
        "locator_repository_key": "3" * 64,
        "authority_resolution_hash": "4" * 64,
        "binding_id": None,
        "outbox_digest": "5" * 64,
        "batch_digest": "6" * 64,
        "persisted_candidate_ids": (
            [] if listed_as_replayed else [_RECEIPT_CANDIDATE_ID]
        ),
        "replayed_candidate_ids": (
            [_RECEIPT_CANDIDATE_ID] if listed_as_replayed else []
        ),
        "results": [
            {
                "candidate_id": _RECEIPT_CANDIDATE_ID,
                "request_id": _RECEIPT_REQUEST_ID,
                "replayed": replayed,
                "status": status,
                "dedupe": dedupe,
                "validation_report_hash": "7" * 64,
                "write_results": write_results,
            }
        ],
    }
    return {**body, "receipt_digest": canonical_sha256(body)}


def _memory_outbox() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": "memory_candidate_outbox_v1",
        "review_id": "review-outbox-parser",
        "repository_key": "3" * 64,
        "locator_repository_key": "3" * 64,
        "authority_resolution_hash": "4" * 64,
        "binding_id": None,
        "head_sha": "a" * 40,
        "snapshot_id": "MSNAP-" + "5" * 64,
        "batch_digest": "6" * 64,
        "actor_type": "runtime",
        "actor_id": "memory-curator",
        "reason_code": "candidate_submitted",
        "entries": [
            {
                "candidate_id": _RECEIPT_CANDIDATE_ID,
                "candidate_hash": "7" * 64,
                "request_id": _RECEIPT_REQUEST_ID,
                "origin": "local",
                "allowed_source_refs": [
                    GitCommitSourceRef("a" * 40).to_dict(),
                ],
            }
        ],
    }
    return {**body, "outbox_digest": canonical_sha256(body)}


def _active_record(
    *,
    repository_key: str,
    head_sha: str,
    index: int = 1,
    effect: PolicyEffect | None = None,
    scope: MemoryScope | None = None,
    status: RecordStatus = RecordStatus.ACTIVE,
    expiry_conditions: tuple[ExpiryCondition, ...] = (),
) -> DurableMemoryRecord:
    return DurableMemoryRecord(
        candidate_id="MC-" + format(index, "064x"),
        repository_key=repository_key,
        kind=MemoryKind.BUSINESS_INVARIANT,
        statement="The public add operation preserves addition semantics.",
        scope=scope or MemoryScope(paths=("app.py",)),
        source_refs=(
            GitCommitSourceRef(
                commit_sha=head_sha,
                metadata_hash="1" * 64,
            ),
        ),
        source_bundle_hash="2" * 64,
        valid_from_sha=head_sha,
        validity_policies=(ValidityPolicy.MANUAL_UNTIL_REVOKED,),
        confidence=MemoryConfidence.HIGH,
        sensitivity=Sensitivity.NORMAL,
        policy_effect=effect,
        approved_by="amy",
        approval_event_id="EVT-" + format(index + 1_000, "064x"),
        status=status,
        created_at="2026-07-14T12:00:00Z",
        expiry_conditions=expiry_conditions,
    )


def _persist_active_record(
    repo: Path,
    store: MemoryStore,
    *,
    repository_key: str,
    head_sha: str,
    index: int,
    expiry_conditions: tuple[ExpiryCondition, ...],
    effect: PolicyEffect | None = None,
) -> DurableMemoryRecord:
    candidate = MemoryCandidate(
        repository_key=repository_key,
        kind=MemoryKind.REVIEW_RULE,
        statement=f"Expiry integration record {index} must remain authoritative.",
        scope=MemoryScope(paths=("app.py",)),
        source_refs=(GitCommitSourceRef(head_sha),),
        valid_from_sha=head_sha,
        validity_policies=(ValidityPolicy.MANUAL_UNTIL_REVOKED,),
        confidence=MemoryConfidence.HIGH,
        sensitivity=Sensitivity.NORMAL,
        policy_effect=effect,
        producer=Producer(ProducerType.MODEL, "memory-curator", "1.0"),
        origin_review_id=f"review-expiry-fixture-{index}",
        status=CandidateStatus.PROPOSED,
        created_at="2026-07-14T10:00:00Z",
    )
    provenance = TrustedCandidateProvenance(
        origin=ProducerType.MODEL,
        review_id=candidate.origin_review_id,
        target_head_sha=head_sha,
        locator_repository_key=repository_key,
        authority_repository_key=repository_key,
        authority_resolution_hash=candidate_authority_resolution_hash(
            repository_key,
            repository_key,
        ),
        allowed_source_refs=candidate.source_refs,
    )
    lifecycle = MemoryLifecycle(store, SourceValidator(repo))
    submitted = lifecycle.submit_candidate(
        candidate,
        runtime_provenance=provenance,
        request_id=stable_request_id("pipeline-expiry-submit", str(index)),
    )
    assert submitted.status is CandidateStatus.PENDING_APPROVAL
    return lifecycle.approve_candidate(
        candidate.candidate_id,
        runtime_provenance=provenance,
        actor="amy",
        reason="Maintainer approved the bounded expiry integration fixture.",
        request_id=stable_request_id("pipeline-expiry-approve", str(index)),
        created_at="2026-07-14T12:00:00Z",
        expiry_conditions=expiry_conditions,
    ).record


def test_v5_dispatch_contains_both_memory_phases() -> None:
    assert PHASE_MESSAGES[RunPhase.MEMORY_SELECTION]
    assert PHASE_MESSAGES[RunPhase.MEMORY_PROPOSAL]


class _DynamicCuratorAdapter:
    provider_name = "fake-curator"

    def __init__(self) -> None:
        self.calls = 0

    def complete_turn(self, request):
        self.calls += 1
        envelope = json.loads(request.messages[0]["content"])
        source = envelope["source_ref_allowlist"][0]
        response = {
            "schema_version": 1,
            "candidates": [
                {
                    "candidate_id": "dynamic-curator-candidate",
                    "kind": "review_rule",
                    "statement": "The final reconciliation is authoritative evidence.",
                    "scope": {
                        "schema_version": 1,
                        "paths": ["app.py"],
                        "symbols": [],
                        "contracts": [],
                        "languages": [],
                    },
                    "source_ref_ids": [source["source_ref_id"]],
                    "validity_policies": ["source_content_hash"],
                    "confidence": "high",
                    "sensitivity": "normal",
                    "policy_effect_id": None,
                }
            ],
        }
        return ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text=json.dumps(response),
            provider_name=self.provider_name,
            model="dynamic-curator",
        )


class _DynamicCuratorFactory:
    def __init__(self) -> None:
        self.adapter = _DynamicCuratorAdapter()

    def create(self):
        return self.adapter


def test_model_curator_receives_final_reconciliation_source_without_project_rules(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    factory = _DynamicCuratorFactory()

    def builder(config):
        if config.stage_label == "memory-curator":
            return factory
        return pipeline_module.build_model_adapter_factory_from_config(config)

    pipeline, session_store, checkpoint_store, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ_WRITE,
        review_id="review-memory-model-curator-source",
        memory_curator=ModelStageConfig(
            mode="model",
            provider="fake",
            model="dynamic-curator",
        ),
        adapter_factory_builder=builder,
    )

    result = pipeline.execute()

    assert factory.adapter.calls == 1
    envelope = _json(checkpoint_store.run_dir / "memory_curator_envelope.json")
    source = envelope["source_ref_allowlist"][0]["source_ref"]
    assert source["type"] == "session_artifact"
    assert source["artifact_name"] == "reconciliation"
    assert result.context.memory_candidate_batch.candidates
    candidate_source = result.context.memory_candidate_batch.candidates[0].source_refs[0]
    assert candidate_source.to_dict() == source
    store = MemoryStore(
        build_repository_memory_namespace(
            session_store.load().repository,
            memory_root,
        ),
        read_only=True,
    )
    assert store.list_candidates(
        result.context.memory_repository_key
    )[0].status is CandidateStatus.PENDING_APPROVAL


def test_resume_discards_partial_model_curator_artifacts_before_reexecution(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _DynamicCuratorFactory()

    def builder(config):
        if config.stage_label == "memory-curator":
            return factory
        return pipeline_module.build_model_adapter_factory_from_config(config)

    pipeline, session_store, checkpoint_store, _memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ_WRITE,
        review_id="review-memory-curator-partial-commit",
        memory_curator=ModelStageConfig(
            mode="model",
            provider="fake",
            model="dynamic-curator",
        ),
        adapter_factory_builder=builder,
    )
    original_register = session_store.register_existing_artifact
    interrupted = False

    def fail_after_candidate_artifacts(*args, **kwargs):
        nonlocal interrupted
        if kwargs.get("name") == "memory_curator_envelope" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("synthetic curator artifact crash")
        return original_register(*args, **kwargs)

    monkeypatch.setattr(
        session_store,
        "register_existing_artifact",
        fail_after_candidate_artifacts,
    )
    with pytest.raises(KeyboardInterrupt, match="synthetic curator artifact crash"):
        pipeline.execute()
    assert factory.adapter.calls == 1
    interrupted_manifest = session_store.load()
    assert interrupted_manifest.phases[RunPhase.MEMORY_PROPOSAL.value].status.value == (
        "running"
    )
    assert {
        "memory_curator_decision",
        "memory_candidates",
    }.issubset(interrupted_manifest.artifacts)
    assert "memory_curator_envelope" not in interrupted_manifest.artifacts

    monkeypatch.setattr(
        session_store,
        "register_existing_artifact",
        original_register,
    )
    pipeline.execute(starting_phase=RunPhase.MEMORY_PROPOSAL, resuming=True)

    assert factory.adapter.calls == 2
    manifest = session_store.load()
    assert manifest.status is RunStatus.COMPLETED
    assert {
        "memory_curator_envelope",
        "memory_curator_raw_response",
    }.issubset(manifest.artifacts)
    assert _json(checkpoint_store.run_dir / "memory_curator_envelope.json")


def test_memory_off_commits_disabled_artifacts_without_touching_store(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    pipeline, session_store, checkpoint_store, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.OFF,
        review_id="review-memory-off",
    )

    result = pipeline.execute()

    manifest = session_store.load()
    assert manifest.status is RunStatus.COMPLETED
    assert set(manifest.phases[RunPhase.MEMORY_SELECTION.value].artifacts) == {
        "memory_selection_input",
        "memory_snapshot",
        "memory_selection_decision",
        "memory_feedback_summary",
    }
    assert set(manifest.phases[RunPhase.MEMORY_PROPOSAL.value].artifacts) == {
        "memory_curator_decision",
        "memory_candidates",
    }
    snapshot = MemorySnapshot.from_dict(
        _json(checkpoint_store.run_dir / "memory_snapshot.json")
    )
    assert snapshot.eligible_records == ()
    assert snapshot.applicability_decisions == ()
    assert _json(checkpoint_store.run_dir / "memory_selection_decision.json")[
        "status"
    ] == "disabled"
    assert _json(checkpoint_store.run_dir / "memory_curator_decision.json")[
        "outcome"
    ] == "skipped"
    assert result.context.memory_snapshot == snapshot
    assert not memory_root.exists()


def test_memory_read_with_empty_store_is_a_normal_session_only_selection(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    pipeline, session_store, checkpoint_store, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ,
        review_id="review-memory-read-empty",
    )

    result = pipeline.execute()

    assert session_store.load().status is RunStatus.COMPLETED
    decision = _json(checkpoint_store.run_dir / "memory_selection_decision.json")
    assert decision["status"] == "selected"
    assert decision["reason_codes"] == []
    assert _json(checkpoint_store.run_dir / "memory_curator_decision.json")[
        "reason_code"
    ] == "memory_read_only"
    assert result.context.memory_snapshot is not None
    assert result.context.memory_snapshot.eligible_records == ()
    assert result.context.repository_intelligence is not None
    provenance = result.context.repository_intelligence.cache_provenance
    assert provenance is not None
    assert provenance.persistent is False
    assert not memory_root.exists()


_LEGAL_PERSISTENCE_RESULT_OUTCOMES = frozenset(
    {
        (False, "pending_approval", "unique"),
        (False, "rejected", "unique"),
        (False, "pending_approval", "exact_replay"),
        (False, "approved", "exact_replay"),
        (False, "rejected", "active_duplicate"),
        (False, "rejected", "pending_duplicate"),
        (False, "pending_approval", "enhanced_provenance"),
        (True, "pending_approval", "exact_replay"),
        (True, "approved", "exact_replay"),
        (True, "rejected", "rejected_unchanged"),
    }
)


def test_memory_outbox_parser_rejects_digest_valid_empty_entries() -> None:
    outbox = _memory_outbox()
    outbox["entries"] = []
    body = {key: value for key, value in outbox.items() if key != "outbox_digest"}
    outbox["outbox_digest"] = canonical_sha256(body)

    with pytest.raises(ValueError, match="entries"):
        pipeline_module._memory_outbox_from_dict(outbox)


def test_memory_outbox_replay_preview_rejects_empty_entries() -> None:
    with pytest.raises(ValueError, match="entries.*empty"):
        pipeline_module.MemoryOutboxReplayPreview(
            outbox_digest="1" * 64,
            batch_digest="2" * 64,
            entries=(),
        )


def test_memory_persistence_receipt_rejects_digest_valid_empty_success() -> None:
    receipt = _persistence_receipt(
        replayed=False,
        status="pending_approval",
        dedupe="unique",
        write_results=[_persistence_write_result(applied=True)],
    )
    receipt["persisted_candidate_ids"] = []
    receipt["replayed_candidate_ids"] = []
    receipt["results"] = []
    body = {
        key: value for key, value in receipt.items() if key != "receipt_digest"
    }
    receipt["receipt_digest"] = canonical_sha256(body)

    with pytest.raises(ValueError, match="candidate IDs are empty"):
        pipeline_module._memory_persistence_receipt_from_dict(receipt)


@pytest.mark.parametrize(
    ("replayed", "status", "dedupe"),
    sorted(_LEGAL_PERSISTENCE_RESULT_OUTCOMES),
)
def test_memory_persistence_receipt_accepts_complete_legal_result_matrix(
    replayed: bool,
    status: str,
    dedupe: str,
) -> None:
    receipt = _persistence_receipt(
        replayed=replayed,
        status=status,
        dedupe=dedupe,
        write_results=(
            []
            if replayed
            else [_persistence_write_result(applied=True)]
        ),
    )

    assert pipeline_module._memory_persistence_receipt_from_dict(receipt) == receipt


@pytest.mark.parametrize(
    ("replayed", "status", "dedupe"),
    [
        (replayed, status, dedupe)
        for replayed in (False, True)
        for status in (
            "proposed",
            "validated",
            "pending_approval",
            "approved",
            "rejected",
        )
        for dedupe in (
            "unique",
            "exact_replay",
            "active_duplicate",
            "pending_duplicate",
            "rejected_unchanged",
            "enhanced_provenance",
        )
        if (replayed, status, dedupe)
        not in _LEGAL_PERSISTENCE_RESULT_OUTCOMES
    ],
)
def test_memory_persistence_receipt_rejects_all_other_result_outcomes(
    replayed: bool,
    status: str,
    dedupe: str,
) -> None:
    receipt = _persistence_receipt(
        replayed=replayed,
        status=status,
        dedupe=dedupe,
        write_results=(
            []
            if replayed
            else [_persistence_write_result(applied=True)]
        ),
    )

    with pytest.raises(ValueError, match="outcome"):
        pipeline_module._memory_persistence_receipt_from_dict(receipt)


@pytest.mark.parametrize(
    ("replayed", "write_results"),
    [
        (False, []),
        (False, [_persistence_write_result(applied=False)]),
        (True, [_persistence_write_result(applied=True)]),
        (
            False,
            [
                _persistence_write_result(
                    applied=True,
                    subject_id="MC-" + "9" * 64,
                )
            ],
        ),
    ],
)
def test_memory_persistence_receipt_rejects_unproven_write_classification(
    replayed: bool,
    write_results: list[dict[str, object]],
) -> None:
    receipt = _persistence_receipt(
        replayed=replayed,
        status="pending_approval",
        dedupe="exact_replay" if replayed else "unique",
        write_results=write_results,
    )

    with pytest.raises(ValueError, match="write|subject"):
        pipeline_module._memory_persistence_receipt_from_dict(receipt)


@pytest.mark.parametrize(
    ("replayed", "classified_as_replayed"),
    [(False, True), (True, False)],
)
def test_memory_persistence_receipt_rejects_result_id_classification_mismatch(
    replayed: bool,
    classified_as_replayed: bool,
) -> None:
    receipt = _persistence_receipt(
        replayed=replayed,
        status="pending_approval",
        dedupe="exact_replay" if replayed else "unique",
        write_results=(
            []
            if replayed
            else [_persistence_write_result(applied=True)]
        ),
        classified_as_replayed=classified_as_replayed,
    )

    with pytest.raises(ValueError, match="classification"):
        pipeline_module._memory_persistence_receipt_from_dict(receipt)


def test_rejected_unchanged_without_writes_is_classified_as_idempotent_replay(
) -> None:
    lifecycle_result = CandidateLifecycleResult(
        candidate_id=_RECEIPT_CANDIDATE_ID,
        status=CandidateStatus.REJECTED,
        validation=None,  # type: ignore[arg-type]
        dedupe=CandidateDedupeDecision(
            CandidateDedupeKind.REJECTED_UNCHANGED,
            related_candidate_id=_RECEIPT_CANDIDATE_ID,
        ),
        persisted=True,
    )

    assert pipeline_module._memory_lifecycle_result_was_replayed(lifecycle_result)


def test_memory_read_write_commits_outbox_before_idempotent_candidate_receipt(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    pipeline, session_store, checkpoint_store, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ_WRITE,
        review_id="review-memory-write",
        project_rules=("Preserve the public addition contract.",),
    )

    result = pipeline.execute()

    manifest = session_store.load()
    proposal_artifacts = set(
        manifest.phases[RunPhase.MEMORY_PROPOSAL.value].artifacts
    )
    assert {
        "memory_curator_decision",
        "memory_candidates",
        "memory_outbox",
        "memory_persistence_receipt",
    }.issubset(proposal_artifacts)
    outbox = _json(checkpoint_store.run_dir / "memory_outbox.json")
    receipt = _json(
        checkpoint_store.run_dir / "memory_persistence_receipt.json"
    )
    assert receipt["success"] is True
    assert receipt["outbox_digest"] == outbox["outbox_digest"]
    assert len(receipt["persisted_candidate_ids"]) == 1
    assert receipt["replayed_candidate_ids"] == []
    assert result.context.memory_persistence_receipt == receipt

    identity = RevisionResolver().repository_identity(git_repo)
    store = MemoryStore(
        build_repository_memory_namespace(identity, memory_root),
        read_only=True,
    )
    candidates = store.list_candidates(outbox["repository_key"])
    assert [item.candidate_id for item in candidates] == receipt[
        "persisted_candidate_ids"
    ]
    assert candidates[0].status is CandidateStatus.PENDING_APPROVAL
    assert isinstance(candidates[0].source_refs[0], HumanDeclarationSourceRef)
    assert receipt["results"][0]["status"] == CandidateStatus.PENDING_APPROVAL.value
    assert len(receipt["results"][0]["write_results"]) == 3


def test_candidate_requires_approval_across_reviews_and_remains_traceable(
    git_repo: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rule = "Preserve the public addition contract."
    review_a, session_a, checkpoint_a, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ_WRITE,
        review_id="review-memory-lifecycle-a",
        project_rules=(rule,),
        change_marker="memory lifecycle review A",
    )
    result_a = review_a.execute()
    receipt_a = _json(
        checkpoint_a.run_dir / "memory_persistence_receipt.json"
    )
    candidate_id = receipt_a["persisted_candidate_ids"][0]
    assert receipt_a["results"][0]["status"] == "pending_approval"
    assert [
        item["candidate_id"]
        for item in result_a.context.brief.memory_audit["pending_candidates"]
    ] == [candidate_id]

    replay_reason = "Maintainer requested an explicit outbox replay audit."
    for _attempt in range(2):
        assert (
            command_main(
                [
                    "memory",
                    "replay-outbox",
                    session_a.load().review_id,
                    "--actor",
                    "amy",
                    "--reason",
                    replay_reason,
                    "--non-interactive",
                    "--yes",
                    "--repo",
                    str(git_repo),
                    "--memory-root",
                    str(memory_root),
                    "--json",
                ]
            )
            == 0
        )
        replay_documents = [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if line.strip()
        ]
        replay_receipt = replay_documents[-1]["receipt"]
        assert replay_receipt["persisted_candidate_ids"] == []
        assert replay_receipt["replayed_candidate_ids"] == [candidate_id]

    # Re-proposing the same rule before approval remains non-authoritative. The
    # lifecycle records a deterministic duplicate outcome, and Reporting must
    # not turn that rejected Candidate into a phantom approval request.
    review_b, _session_b, checkpoint_b, _same_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ_WRITE,
        review_id="review-memory-lifecycle-b",
        project_rules=(rule,),
        fixed_revisions=(
            session_a.load().revisions.resolved_base_sha,
            session_a.load().revisions.resolved_head_sha,
        ),
    )
    result_b = review_b.execute()
    assert result_b.context.memory_snapshot.eligible_records == ()
    receipt_b = _json(
        checkpoint_b.run_dir / "memory_persistence_receipt.json"
    )
    assert receipt_b["results"][0]["status"] == "rejected"
    assert receipt_b["results"][0]["dedupe"] == "pending_duplicate"
    assert result_b.context.brief.memory_audit["pending_candidates"] == []
    outcomes = result_b.context.brief.memory_audit["status"]["outbox"][
        "candidate_outcomes"
    ]
    assert outcomes == [
        {
            "candidate_id": receipt_b["results"][0]["candidate_id"],
            "status": "rejected",
            "dedupe": "pending_duplicate",
            "replayed": False,
            "persistence": "persisted",
            "validation_report_hash": receipt_b["results"][0][
                "validation_report_hash"
            ],
        }
    ]

    exit_code = command_main(
        [
            "memory",
            "approve",
            candidate_id,
            "--actor",
            "amy",
            "--reason",
            "Maintainer confirmed the exact project rule and authority.",
            "--non-interactive",
            "--yes",
            "--repo",
            str(git_repo),
            "--memory-root",
            str(memory_root),
            "--json",
        ]
    )
    assert exit_code == 0
    cli_documents = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    approval = cli_documents[-1]
    assert approval["type"] == "approve_result"
    assert approval["candidate_id"] == candidate_id

    review_c, session_c, checkpoint_c, _same_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ,
        review_id="review-memory-lifecycle-c",
        change_marker="memory lifecycle review C",
    )
    result_c = review_c.execute()
    selected = result_c.context.memory_snapshot.eligible_records
    assert [item.candidate_id for item in selected] == [candidate_id]
    record = selected[0]
    assert record.memory_id == approval["memory_id"]
    assert record.approval_event_id == approval["event_id"]
    assert result_c.context.brief.memory_audit["pending_candidates"] == []
    assert [
        item["memory_id"]
        for item in result_c.context.brief.memory_audit["applied_memory"]
    ] == [record.memory_id]

    identity = RevisionResolver().repository_identity(git_repo)
    store = MemoryStore(
        build_repository_memory_namespace(identity, memory_root),
        read_only=True,
    )
    assert store.get_candidate(candidate_id).status is CandidateStatus.APPROVED
    assert store.get_record(record.memory_id) == record
    catalog = pipeline_module._memory_curator_fingerprint_catalog(
        store,
        record.repository_key,
    )
    assert [
        (item.content_fingerprint, item.state) for item in catalog
    ] == [
        (store.get_candidate(candidate_id).content_fingerprint, "active")
    ]
    approval_events = [
        event
        for event in store.list_events(
            record.repository_key,
            subject_type="candidate",
            subject_id=candidate_id,
        )
        if event.action == "approve"
    ]
    assert [event.event_id for event in approval_events] == [
        record.approval_event_id
    ]
    replay_events = store.list_events(
        record.repository_key,
        subject_type="outbox_replay",
        subject_id=session_a.load().review_id,
    )
    assert len(replay_events) == 1
    assert replay_events[0].actor_id == "amy"
    assert replay_events[0].reason == replay_reason

    brief_payload = _json(checkpoint_c.run_dir / "review_brief.json")
    assert brief_payload["memory_audit"]["applied_memory"][0][
        "memory_id"
    ] == record.memory_id
    markdown = (checkpoint_c.run_dir / "report.md").read_text(encoding="utf-8")
    assert record.memory_id in markdown
    assert candidate_id in markdown
    assert session_a.load().status is RunStatus.COMPLETED
    assert session_c.load().status is RunStatus.COMPLETED


def test_committed_outbox_can_be_replayed_after_store_write_failure(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, checkpoint_store, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ_WRITE,
        review_id="review-memory-outbox-replay",
        project_rules=("Preserve the public addition contract.",),
    )
    monkeypatch.setattr(
        pipeline_module,
        "replay_memory_outbox",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("synthetic store outage")),
    )

    result = pipeline.execute()

    manifest = session_store.load()
    assert manifest.status is RunStatus.COMPLETED
    assert "memory_outbox" in manifest.artifacts
    assert "memory_persistence_receipt" not in manifest.artifacts
    assert "outbox_pending" in result.context.memory_degradation_codes
    outbox = _json(checkpoint_store.run_dir / "memory_outbox.json")

    receipt = replay_memory_outbox(
        repository=git_repo,
        memory_root=memory_root,
        review_id=manifest.review_id,
        expected_repository_key=outbox["repository_key"],
        expected_authority_resolution_hash=outbox[
            "authority_resolution_hash"
        ],
        expected_outbox_digest=outbox["outbox_digest"],
    )

    assert receipt["success"] is True
    assert len(receipt["persisted_candidate_ids"]) == 1


def test_curator_failure_is_visible_and_cannot_change_review_conclusion(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, _checkpoint, _memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ_WRITE,
        review_id="review-memory-curator-fallback",
        project_rules=("Preserve the public addition contract.",),
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_local_memory_curator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic Curator failure")
        ),
    )

    result = pipeline.execute()
    context = result.context

    assert session_store.load().status is RunStatus.COMPLETED
    assert context.memory_candidate_batch.candidates == ()
    assert context.memory_curator_decision["outcome"] == "rejected"
    assert context.memory_curator_decision["review_conclusion_impact"] == "none"
    assert context.completion is not None
    assert context.brief.memory_audit["status"]["curator"]["status"] == "fallback"


def test_fixed_snapshot_drives_typed_stage_projections_and_memory_audit(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, checkpoint_store, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ,
        review_id="review-memory-projections",
    )
    manifest = session_store.load()
    namespace = build_repository_memory_namespace(
        manifest.repository,
        memory_root,
    )
    MemoryStore(namespace)
    record = _active_record(
        repository_key=namespace.repository_key,
        head_sha=manifest.revisions.resolved_head_sha,
        effect=PolicyEffect(
            effect_kind=PolicyEffectKind.REQUIRE_CONTRACT,
            value="behavioral_correctness",
        ),
    )
    original_read_view = MemoryStore.read_view

    def read_view(
        store: MemoryStore,
        repository_key: str,
    ):
        view = original_read_view(store, repository_key)
        if repository_key == record.repository_key:
            return replace(view, records=(record,))
        return view

    monkeypatch.setattr(MemoryStore, "read_view", read_view)

    result = pipeline.execute()
    context = result.context

    assert context.memory_policy_compilation is not None
    assert context.memory_policy_compilation.blocked is False
    assert any(
        item.origin is IntentOrigin.PROJECT_MEMORY
        and item.source is IntentSource.INFERRED
        for item in context.intent.provenance
    )
    assert context.planner_memory_projection is not None
    assert [
        item.requirement_id
        for item in context.planner_memory_projection.required_contracts
    ] == ["behavioral_correctness"]
    assert all(
        record.memory_id in assignment.initial_context.selected_memory_refs
        for assignment in context.assignments
    )
    assert context.reviewer_memory_selections
    assert context.brief is not None
    audit = context.brief.memory_audit
    assert audit["compiled_policy"]["blocked"] is False
    assert [item["memory_id"] for item in audit["applied_memory"]] == [
        record.memory_id
    ]
    assert "raw_response" not in json.dumps(audit, ensure_ascii=False)
    assert _json(
        checkpoint_store.run_dir / "memory_selection_decision.json"
    )["policy_compilation"] == context.memory_policy_compilation.to_dict()


def test_memory_required_check_uses_frozen_gate_registry_and_forces_execution(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (git_repo / "pyproject.toml").write_text(
        """
[[tool.review-agent.quality-gates]]
name = "bandit_security"
category = "security"
cost = "expensive"
command = ["python", "-m", "bandit", "-r", "."]
blocking = false
timeout_seconds = 30
trigger_risks = ["critical"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "pyproject.toml")
    run_git(git_repo, "commit", "-m", "configure frozen security gate")
    pipeline, session_store, _checkpoint, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ,
        review_id="review-memory-required-check",
        change_marker="memory required check",
    )
    manifest = session_store.load()
    namespace = build_repository_memory_namespace(
        manifest.repository,
        memory_root,
    )
    MemoryStore(namespace)
    record = _active_record(
        repository_key=namespace.repository_key,
        head_sha=manifest.revisions.resolved_head_sha,
        index=71,
        effect=PolicyEffect(
            effect_kind=PolicyEffectKind.REQUIRE_CHECK,
            value="bandit_security",
        ),
    )
    original_read_view = MemoryStore.read_view

    def read_view(store: MemoryStore, repository_key: str):
        view = original_read_view(store, repository_key)
        return replace(view, records=(record,))

    executed: list[str] = []

    def execute_gate(_repo: Path, _revision: str, gate):
        executed.append(gate.name)
        return QualityGateExecution(
            result=QualityGateResult(
                name=gate.name,
                status="passed",
                command=list(gate.command),
                summary="Memory-required security gate passed.",
                category=gate.category,
                cost=gate.cost,
                source=gate.source,
                blocking=gate.blocking,
                sandbox="test_snapshot",
            ),
            raw_output="Memory-required security gate passed.",
        )

    monkeypatch.setattr(MemoryStore, "read_view", read_view)
    monkeypatch.setattr(pipeline_module, "execute_quality_gate", execute_gate)
    monkeypatch.setattr(
        pipeline_module,
        "quality_gate_policy_decision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "an approved require_check must bypass the ordinary risk trigger"
            )
        ),
    )

    result = pipeline.execute()

    assert "bandit_security" in [
        gate.name for gate in result.context.quality_gate_plan.gates
    ]
    assert executed == ["bandit_security"]
    assert [
        item.requirement_id
        for item in result.context.planner_memory_projection.required_checks
    ] == ["bandit_security"]
    assert [
        item.status
        for item in result.context.quality_results
        if item.name == "bandit_security"
    ] == ["passed"]
    assert not any(
        "Memory-required check" in blocker
        for blocker in result.context.completion.blockers
    )


def test_selection_preserves_contract_scope_and_nonactive_record_audit_decisions(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, _checkpoint, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ,
        review_id="review-memory-contract-and-status-audit",
    )
    manifest = session_store.load()
    namespace = build_repository_memory_namespace(manifest.repository, memory_root)
    MemoryStore(namespace)
    active_contract = _active_record(
        repository_key=namespace.repository_key,
        head_sha=manifest.revisions.resolved_head_sha,
        index=11,
        scope=MemoryScope(contracts=("behavioral_correctness",)),
        effect=PolicyEffect(
            effect_kind=PolicyEffectKind.REQUIRE_CONTRACT,
            value="behavioral_correctness",
        ),
    )
    revoked = _active_record(
        repository_key=namespace.repository_key,
        head_sha=manifest.revisions.resolved_head_sha,
        index=12,
        status=RecordStatus.REVOKED,
    )
    stale = _active_record(
        repository_key=namespace.repository_key,
        head_sha=manifest.revisions.resolved_head_sha,
        index=13,
        status=RecordStatus.REVALIDATION_REQUIRED,
    )
    original_read_view = MemoryStore.read_view

    def read_view(store: MemoryStore, repository_key: str):
        view = original_read_view(store, repository_key)
        return replace(view, records=(active_contract, revoked, stale))

    monkeypatch.setattr(MemoryStore, "read_view", read_view)
    result = pipeline.execute()

    context = result.context
    assert "behavioral_correctness" in context.memory_selection_input.contracts
    assert [item.memory_id for item in context.memory_snapshot.eligible_records] == [
        active_contract.memory_id
    ]
    decisions = {
        item.memory_id: item
        for item in context.memory_snapshot.applicability_decisions
    }
    assert decisions[revoked.memory_id].applicability.value == "revoked"
    assert decisions[stale.memory_id].applicability.value == "source_changed"
    assert context.memory_policy_compilation is not None
    assert any(
        action.to_dict().get("contract_id") == "behavioral_correctness"
        for action in context.memory_policy_compilation.actions
    )


def test_same_revision_resume_loads_fixed_memory_without_opening_store(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, checkpoint_store, _memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ,
        review_id="review-memory-fixed-resume",
    )
    original = pipeline.execute()
    snapshot = original.context.memory_snapshot
    snapshot_id = snapshot.snapshot_id
    created_at = snapshot.created_at

    class ForbiddenStore:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("same-revision resume must not open MemoryStore")

    monkeypatch.setattr(pipeline_module, "MemoryStore", ForbiddenStore)
    monkeypatch.setattr(
        pipeline_module,
        "_utc_now",
        lambda: "2030-01-01T00:00:00Z",
    )
    resumed = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()

    assert resumed.action is ResumeAction.AUDIT_COMPLETED, (
        resumed.action,
        resumed.starting_phase,
        resumed.manifest.errors,
    )
    assert resumed.starting_phase is None
    assert RunPhase.MEMORY_SELECTION in resumed.reused_phases
    assert _json(checkpoint_store.run_dir / "memory_snapshot.json")[
        "snapshot_id"
    ] == snapshot_id
    assert _json(checkpoint_store.run_dir / "memory_snapshot.json")[
        "created_at"
    ] == created_at


def test_downstream_resume_uses_fixed_snapshot_after_store_becomes_unavailable(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, checkpoint_store, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ,
        review_id="review-memory-downstream-resume",
    )
    MemoryStore(
        build_repository_memory_namespace(
            session_store.load().repository,
            memory_root,
        )
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(
            pipeline,
            "_run_planning",
            lambda: (_ for _ in ()).throw(
                KeyboardInterrupt("interrupt after Memory Selection")
            ),
        )
        with pytest.raises(KeyboardInterrupt, match="after Memory Selection"):
            pipeline.execute()
    pinned = _json(checkpoint_store.run_dir / "memory_snapshot.json")

    class ForbiddenStore:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("downstream resume must not reopen MemoryStore")

    monkeypatch.setattr(pipeline_module, "MemoryStore", ForbiddenStore)
    resumed = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()

    assert resumed.action is ResumeAction.CONTINUE_SESSION
    assert resumed.starting_phase is RunPhase.PLANNING
    assert _json(checkpoint_store.run_dir / "memory_snapshot.json")[
        "snapshot_id"
    ] == pinned["snapshot_id"]
    assert session_store.load().status is RunStatus.COMPLETED


def test_snapshot_tamper_invalidates_from_memory_selection_only(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    pipeline, session_store, checkpoint_store, _memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.OFF,
        review_id="review-memory-tamper",
    )
    pipeline.execute()
    snapshot_path = checkpoint_store.run_dir / "memory_snapshot.json"
    snapshot_path.write_text(
        snapshot_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    resumed = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()

    assert resumed.action is ResumeAction.CONTINUE_SESSION
    assert resumed.starting_phase is RunPhase.MEMORY_SELECTION
    assert resumed.reused_phases == (
        RunPhase.PREFLIGHT,
        RunPhase.QUALITY_GATES,
        RunPhase.REPOSITORY_INTELLIGENCE,
    )
    assert session_store.load().status is RunStatus.COMPLETED


def test_second_review_of_same_revision_hits_persistent_repository_cache(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    first, first_session, _first_checkpoint, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ_WRITE,
        review_id="review-memory-cache-first",
    )
    first_result = first.execute()
    assert first_result.context.repository_intelligence.cache_provenance.status.value in {
        "miss",
        "rebuild",
    }
    first_manifest = first_session.load()

    second_checkpoint = CheckpointStore(
        git_repo,
        "review-memory-cache-second",
    )
    second_session = SessionStore(second_checkpoint.run_dir)
    second_session.create(
        initial_session_manifest(
            review_id="review-memory-cache-second",
            repository=first_manifest.repository,
            revisions=first_manifest.revisions,
            execution=first_manifest.execution,
            now="2026-07-15T00:00:01Z",
        )
    )
    second = ReviewPipeline(
        repository=git_repo,
        checkpoint_store=second_checkpoint,
        session_store=second_session,
        request=ReviewRequest(
            repository_path=first_manifest.repository.canonical_path,
            base_revision=first_manifest.revisions.requested_base,
            head_revision=first_manifest.revisions.requested_head,
            user_intent="Preserve addition semantics",
        ),
        clock=lambda: "2026-07-15T00:00:01Z",
    )

    second_result = second.execute()

    provenance = second_result.context.repository_intelligence.cache_provenance
    assert provenance is not None
    assert provenance.status.value == "hit"
    assert provenance.persistent is True
    assert Path(first_manifest.execution.memory.root_path) == memory_root

    (git_repo / "app.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "move cache target head")
    third_revisions = RevisionResolver().resolve_pair(
        git_repo,
        first_manifest.revisions.resolved_base_sha,
        "HEAD",
    )
    third_checkpoint = CheckpointStore(
        git_repo,
        "review-memory-cache-third",
    )
    third_session = SessionStore(third_checkpoint.run_dir)
    third_session.create(
        initial_session_manifest(
            review_id="review-memory-cache-third",
            repository=first_manifest.repository,
            revisions=third_revisions,
            execution=first_manifest.execution,
            now="2026-07-15T00:00:02Z",
        )
    )
    third = ReviewPipeline(
        repository=git_repo,
        checkpoint_store=third_checkpoint,
        session_store=third_session,
        request=ReviewRequest(
            repository_path=first_manifest.repository.canonical_path,
            base_revision=first_manifest.revisions.resolved_base_sha,
            head_revision="HEAD",
            user_intent="Preserve addition semantics",
        ),
        clock=lambda: "2026-07-15T00:00:02Z",
    ).execute()
    third_provenance = third.context.repository_intelligence.cache_provenance
    assert third_provenance.status.value in {"miss", "rebuild"}
    assert third_provenance.key_hash != provenance.key_hash
    assert provenance.entry_id not in (
        third.context.memory_snapshot.repository_knowledge_refs
    )


def test_required_unavailable_store_becomes_typed_completion_blocker(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, _session, _checkpoint, _memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ,
        required=True,
        review_id="review-memory-required-unavailable",
    )
    monkeypatch.setattr(
        pipeline_module,
        "plan_repository_memory_namespace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("synthetic Memory root outage")
        ),
    )

    result = pipeline.execute()
    context = result.context

    assert context.completion is not None
    assert context.completion.status == "blocked"
    assert len(context.completion.memory_diagnostics) == 1
    diagnostic = context.completion.memory_diagnostics[0]
    assert diagnostic.code.value == "memory_unavailable"
    assert diagnostic.blocking is True
    assert context.brief.memory_audit["status"]["memory_unavailable"] is True
    assert context.brief.memory_audit["status"]["hard_policy_blocked"] is False


def test_read_write_expiry_sweep_persists_before_the_pinned_read_view(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    pipeline, session_store, _checkpoint, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ_WRITE,
        review_id="review-memory-expiry-read-write",
    )
    manifest = session_store.load()
    namespace = build_repository_memory_namespace(
        manifest.repository,
        memory_root,
    )
    store = MemoryStore(namespace)
    record = _persist_active_record(
        git_repo,
        store,
        repository_key=namespace.repository_key,
        head_sha=manifest.revisions.resolved_head_sha,
        index=81,
        expiry_conditions=(
            ExpiryCondition(
                ExpiryConditionKind.AT_TIME,
                "2026-07-15T00:00:00Z",
            ),
        ),
        effect=PolicyEffect(
            PolicyEffectKind.REQUIRE_CHECK,
            "expired_security_check",
        ),
    )
    generation_before = store.get_generations(
        namespace.repository_key
    ).memory_generation

    result = pipeline.execute()

    context = result.context
    stored = store.get_record(record.memory_id)
    generation_after = store.get_generations(
        namespace.repository_key
    ).memory_generation
    decision = next(
        item
        for item in context.memory_snapshot.applicability_decisions
        if item.memory_id == record.memory_id
    )
    assert stored.status is RecordStatus.EXPIRED
    assert generation_after == generation_before + 1
    assert context.memory_snapshot.generations.memory_generation == generation_after
    assert context.memory_selection_input.generations.memory_generation == generation_after
    assert context.memory_snapshot.eligible_records == ()
    assert decision.applicability is Applicability.EXPIRED
    assert decision.reason_codes == ("record_expired",)
    assert context.memory_policy_compilation.actions == ()
    assert context.memory_policy_compilation.blocked is False
    assert context.memory_selection_input.selection_policy_version == "memory_selection_v2"
    assert context.memory_snapshot.selection_policy_version == "memory_selection_v2"
    assert context.memory_selection_decision["reason_codes"] == []


def test_read_mode_derives_due_expiry_without_mutating_store_or_compiling_policy(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    pipeline, session_store, _checkpoint, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ,
        review_id="review-memory-expiry-read-only",
    )
    manifest = session_store.load()
    namespace = build_repository_memory_namespace(
        manifest.repository,
        memory_root,
    )
    store = MemoryStore(namespace)
    record = _persist_active_record(
        git_repo,
        store,
        repository_key=namespace.repository_key,
        head_sha=manifest.revisions.resolved_head_sha,
        index=82,
        expiry_conditions=(
            ExpiryCondition(
                ExpiryConditionKind.AT_TIME,
                "2026-07-15T00:00:00Z",
            ),
        ),
        effect=PolicyEffect(
            PolicyEffectKind.REQUIRE_CHECK,
            "read_only_expired_check",
        ),
    )
    generations_before = store.get_generations(namespace.repository_key)
    events_before = store.list_events(namespace.repository_key)

    result = pipeline.execute()

    context = result.context
    decision = next(
        item
        for item in context.memory_snapshot.applicability_decisions
        if item.memory_id == record.memory_id
    )
    assert store.get_record(record.memory_id).status is RecordStatus.ACTIVE
    assert store.get_generations(namespace.repository_key) == generations_before
    assert store.list_events(namespace.repository_key) == events_before
    assert context.memory_snapshot.generations == generations_before
    assert context.memory_snapshot.eligible_records == ()
    assert decision.applicability is Applicability.EXPIRED
    assert decision.reason_codes == ("expiry_time_reached",)
    assert context.memory_policy_compilation.actions == ()
    assert context.memory_policy_compilation.blocked is False
    assert context.memory_selection_decision["status"] == "degraded"
    assert context.memory_selection_decision["reason_codes"] == [
        "expiry_persistence_deferred"
    ]


def test_failed_expiry_sweep_degrades_but_active_due_policy_stays_excluded(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, _checkpoint, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ_WRITE,
        required=True,
        review_id="review-memory-expiry-sweep-failed",
    )
    manifest = session_store.load()
    namespace = build_repository_memory_namespace(
        manifest.repository,
        memory_root,
    )
    MemoryStore(namespace)
    due = _active_record(
        repository_key=namespace.repository_key,
        head_sha=manifest.revisions.resolved_head_sha,
        index=83,
        effect=PolicyEffect(
            PolicyEffectKind.REQUIRE_CHECK,
            "failed_sweep_expired_check",
        ),
        expiry_conditions=(
            ExpiryCondition(
                ExpiryConditionKind.AT_TIME,
                "2026-07-15T00:00:00Z",
            ),
        ),
    )
    original_read_view = MemoryStore.read_view

    def failed_sweep(*_args, **_kwargs):
        raise OSError("synthetic expiry persistence failure")

    def read_view(store: MemoryStore, repository_key: str):
        return replace(
            original_read_view(store, repository_key),
            records=(due,),
        )

    monkeypatch.setattr(MemoryLifecycle, "expire_due_records", failed_sweep)
    monkeypatch.setattr(MemoryStore, "read_view", read_view)

    result = pipeline.execute()

    context = result.context
    applicability = next(
        item
        for item in context.memory_snapshot.applicability_decisions
        if item.memory_id == due.memory_id
    )
    assert context.memory_snapshot.eligible_records == ()
    assert applicability.applicability is Applicability.EXPIRED
    assert applicability.reason_codes == ("expiry_time_reached",)
    assert context.memory_policy_compilation.actions == ()
    assert context.memory_policy_compilation.blocked is False
    assert context.memory_selection_decision["status"] == "degraded"
    assert context.memory_selection_decision["reason_codes"] == [
        "expiry_sweep_failed",
        "expiry_persistence_deferred",
    ]
    assert "memory_required_unavailable" not in (
        context.memory_selection_decision["reason_codes"]
    )


def test_bounded_expiry_sweep_reports_truncation_unresolved_and_frozen_time(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, _checkpoint, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ_WRITE,
        required=True,
        max_snapshot_records=1,
        review_id="review-memory-expiry-sweep-bounded",
    )
    manifest = session_store.load()
    namespace = build_repository_memory_namespace(
        manifest.repository,
        memory_root,
    )
    MemoryStore(namespace)
    events: list[str] = []
    captured: dict[str, object] = {}
    original_read_view = MemoryStore.read_view

    def sweep(
        _lifecycle: MemoryLifecycle,
        repository_key: str,
        *,
        target_head: str,
        evaluated_at: str,
        max_records: int,
    ) -> ExpirySweepResult:
        events.append("sweep")
        captured["evaluated_at"] = evaluated_at
        captured["max_records"] = max_records
        return ExpirySweepResult(
            repository_key=repository_key,
            target_head=target_head,
            evaluated_at=evaluated_at,
            max_records=max_records,
            scanned_ids=("MEM-" + "e" * 64,),
            expired_ids=(),
            unresolved_ids=("MEM-" + "e" * 64,),
            write_results=(),
            truncated=True,
        )

    def read_view(store: MemoryStore, repository_key: str):
        events.append("read_view")
        return original_read_view(store, repository_key)

    monkeypatch.setattr(MemoryLifecycle, "expire_due_records", sweep)
    monkeypatch.setattr(MemoryStore, "read_view", read_view)

    result = pipeline.execute()

    context = result.context
    assert events == ["sweep", "read_view"]
    assert captured["max_records"] == 1
    assert captured["evaluated_at"] == context.memory_snapshot.created_at
    assert captured["evaluated_at"] == context.memory_feedback_summary.created_at
    assert context.memory_selection_decision["status"] == "degraded"
    assert context.memory_selection_decision["reason_codes"] == [
        "expiry_sweep_truncated",
        "expiry_condition_unresolved",
    ]


def test_selection_captures_generations_records_and_feedback_in_one_read_view(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, _checkpoint, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ,
        review_id="review-memory-atomic-view",
    )
    namespace = build_repository_memory_namespace(
        session_store.load().repository,
        memory_root,
    )
    MemoryStore(namespace)
    original_read_view = MemoryStore.read_view
    calls = 0

    def tracked_read_view(store: MemoryStore, repository_key: str):
        nonlocal calls
        calls += 1
        return original_read_view(store, repository_key)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Selection must use one atomic read_view")

    monkeypatch.setattr(MemoryStore, "read_view", tracked_read_view)
    monkeypatch.setattr(MemoryStore, "get_generations", forbidden)
    monkeypatch.setattr(MemoryStore, "list_records", forbidden)

    result = pipeline.execute()

    assert result.context.memory_snapshot is not None
    assert calls == 1


def test_post_initialization_store_corruption_degrades_without_live_fallback(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, _checkpoint, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ,
        review_id="review-memory-read-view-corrupt",
    )
    MemoryStore(
        build_repository_memory_namespace(
            session_store.load().repository,
            memory_root,
        )
    )
    monkeypatch.setattr(
        MemoryStore,
        "read_view",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MemoryStoreCorruptionError("synthetic read-view corruption")
        ),
    )

    result = pipeline.execute()
    context = result.context

    assert context.memory_snapshot.eligible_records == ()
    assert context.memory_snapshot.generations.memory_generation == 0
    assert context.completion.recommendation == "manual_review"
    assert context.completion.memory_diagnostics[0].blocking is False


def test_execution_context_limit_is_preserved_in_stage_and_reviewer_selection(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, checkpoint_store, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ,
        review_id="review-memory-context-limit",
        max_context_records=1,
        max_query_results=1,
    )
    manifest = session_store.load()
    namespace = build_repository_memory_namespace(
        manifest.repository,
        memory_root,
    )
    MemoryStore(namespace)
    records = (
        _active_record(
            repository_key=namespace.repository_key,
            head_sha=manifest.revisions.resolved_head_sha,
            index=1,
        ),
        _active_record(
            repository_key=namespace.repository_key,
            head_sha=manifest.revisions.resolved_head_sha,
            index=2,
        ),
    )
    original_read_view = MemoryStore.read_view

    def read_view(store: MemoryStore, repository_key: str):
        view = original_read_view(store, repository_key)
        return replace(view, records=records)

    monkeypatch.setattr(MemoryStore, "read_view", read_view)

    result = pipeline.execute()

    assert len(result.context.intent_memory_projection.claims) == 1
    assert result.context.reviewer_memory_selections
    assert all(
        len(selection.records) == 1
        for selection in result.context.reviewer_memory_selections.values()
    )
    envelope = _json(checkpoint_store.run_dir / "reviewer_envelope.json")
    assert len(envelope["parameters"]["context"]["selected_memory_ids"]) == 1


def test_agent_memory_query_reads_snapshot_and_records_independent_observation(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MemoryQueryFactory:
        def create(self):
            def query_turn(request):
                definition = next(
                    item
                    for item in request.tools
                    if item.name == "query_project_memory"
                )
                assignment_id = definition.parameters_schema["properties"][
                    "assignment_id"
                ]["const"]
                return ModelTurnResponse(
                    kind=ModelResponseKind.TOOL_CALLS,
                    tool_calls=[
                        ModelToolCall(
                            "call-memory",
                            "query_project_memory",
                            {
                                "assignment_id": assignment_id,
                                "path": "app.py",
                                "query": "addition contract",
                            },
                        )
                    ],
                )

            def final_turn(request):
                tool_result = request.tool_results[-1]
                if not tool_result.observation_ids:
                    raise RuntimeError(
                        "Memory tool returned no observation: " + tool_result.content
                    )
                observation_id = tool_result.observation_ids[0]
                return ModelTurnResponse(
                    kind=ModelResponseKind.FINAL,
                    final_text=json.dumps(
                        {
                            "contract_assessments": [
                                {
                                    "contract": contract,
                                    "status": "covered",
                                    "summary": "Verified through fixed Memory Snapshot.",
                                    "evidence_refs": [observation_id],
                                }
                                for contract in (
                                    "intent_alignment",
                                    "behavioral_correctness",
                                    "regression_safety",
                                    "test_adequacy",
                                    "unresolved_uncertainties",
                                )
                            ],
                            "confirmed_findings": [],
                            "rejected_hypotheses": [],
                            "uncertainties": [
                                "Only the Snapshot-backed project rule was inspected."
                            ],
                            "observation_refs": [observation_id],
                            "investigation_summary": "Queried fixed project Memory.",
                            "status": "partial",
                        }
                    ),
                )

            return FakeToolCallingAdapter(script=[query_turn, final_turn])

    pipeline, session_store, _checkpoint, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ,
        review_id="review-memory-agent-query",
        reviewer_loop="agent-loop",
        adapter_factory_builder=lambda _config: MemoryQueryFactory(),
    )
    manifest = session_store.load()
    namespace = build_repository_memory_namespace(
        manifest.repository,
        memory_root,
    )
    MemoryStore(namespace)
    record = _active_record(
        repository_key=namespace.repository_key,
        head_sha=manifest.revisions.resolved_head_sha,
    )
    original_read_view = MemoryStore.read_view
    calls = 0

    def one_read_view(store: MemoryStore, repository_key: str):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("Reviewer Memory query must not reread Store")
        return replace(
            original_read_view(store, repository_key),
            records=(record,),
        )

    monkeypatch.setattr(MemoryStore, "read_view", one_read_view)

    result = pipeline.execute()

    reviewer_store = result.context.reviewer_observations[0]
    memory_observations = [
        item
        for item in reviewer_store.list_observations()
        if item.source == "memory.query_project_memory"
    ]
    assert calls == 1
    assert not any(
        "provider attempt" in item
        for item in result.context.reviewer_executions[0].result.uncertainties
    ), " | ".join(result.context.reviewer_executions[0].result.uncertainties)
    assert len(memory_observations) == 1, (
        result.context.reviewer_executions[0].result,
        reviewer_store.summaries_by_id(),
    )
    summary = reviewer_store.summaries_by_id()[
        memory_observations[0].observation_id
    ]
    assert record.memory_id in summary
    assert memory_observations[0].revision == (
        f"head@{manifest.revisions.resolved_head_sha}"
    )


def test_unregistered_memory_policy_is_blocking_and_cannot_expand_registry(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, _checkpoint, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ,
        review_id="review-memory-policy-rejected",
    )
    manifest = session_store.load()
    namespace = build_repository_memory_namespace(
        manifest.repository,
        memory_root,
    )
    MemoryStore(namespace)
    record = _active_record(
        repository_key=namespace.repository_key,
        head_sha=manifest.revisions.resolved_head_sha,
        effect=PolicyEffect(
            effect_kind=PolicyEffectKind.REQUIRE_CONTRACT,
            value="memory_invented_contract",
        ),
    )
    original_read_view = MemoryStore.read_view

    def read_view(
        store: MemoryStore,
        repository_key: str,
    ):
        view = original_read_view(store, repository_key)
        if repository_key == record.repository_key:
            return replace(view, records=(record,))
        return view

    monkeypatch.setattr(MemoryStore, "read_view", read_view)

    result = pipeline.execute()
    context = result.context

    assert context.memory_policy_compilation.blocked is True
    assert context.memory_policy_compilation.actions == ()
    assert all(
        "memory_invented_contract" not in assignment.assigned_contract
        for assignment in context.assignments
    )
    assert context.completion.status == "blocked"
    assert any(
        item.code.value == "policy_rejected" and item.blocking
        for item in context.completion.memory_diagnostics
    )
    assert context.brief.memory_audit["status"]["hard_policy_blocked"] is True


def test_resume_after_receipt_before_phase_completion_reuses_committed_outbox(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, checkpoint_store, _memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ_WRITE,
        review_id="review-memory-receipt-crash",
        project_rules=("Preserve the public addition contract.",),
    )
    original_mark_completed = session_store.mark_phase_completed
    failed = False

    def fail_after_receipt(
        phase: RunPhase,
        artifacts: dict[str, str],
        now: str,
    ):
        nonlocal failed
        if phase is RunPhase.MEMORY_PROPOSAL and not failed:
            failed = True
            raise RuntimeError("synthetic crash after receipt commit")
        return original_mark_completed(phase, artifacts, now)

    with monkeypatch.context() as scoped:
        scoped.setattr(session_store, "mark_phase_completed", fail_after_receipt)
        with pytest.raises(PipelineStageError, match="after receipt commit"):
            pipeline.execute()

    interrupted = session_store.load()
    assert interrupted.phases[RunPhase.MEMORY_PROPOSAL.value].status.value == "failed"
    assert "memory_outbox" in interrupted.artifacts
    assert "memory_persistence_receipt" in interrupted.artifacts

    def forbidden(*_args, **_kwargs):
        raise AssertionError("resume must reuse committed proposal artifacts")

    with monkeypatch.context() as scoped:
        scoped.setattr(pipeline_module, "run_local_memory_curator", forbidden)
        scoped.setattr(pipeline_module, "replay_memory_outbox", forbidden)
        resumed = ReviewSessionResumer(
            repository=git_repo,
            checkpoint_store=checkpoint_store,
            session_store=session_store,
        ).resume()

    assert resumed.action is ResumeAction.CONTINUE_SESSION
    assert resumed.starting_phase is RunPhase.MEMORY_PROPOSAL
    assert session_store.load().status is RunStatus.COMPLETED


def test_resume_after_outbox_before_store_write_replays_without_recurating(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, checkpoint_store, _memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ_WRITE,
        review_id="review-memory-outbox-crash",
        project_rules=("Preserve the public addition contract.",),
    )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            pipeline_module,
            "replay_memory_outbox",
            lambda **_kwargs: (_ for _ in ()).throw(
                KeyboardInterrupt("synthetic crash before Store write")
            ),
        )
        with pytest.raises(KeyboardInterrupt, match="before Store write"):
            pipeline.execute()

    interrupted = session_store.load()
    assert interrupted.phases[RunPhase.MEMORY_PROPOSAL.value].status.value == "running"
    assert "memory_outbox" in interrupted.artifacts
    assert "memory_persistence_receipt" not in interrupted.artifacts

    with monkeypatch.context() as scoped:
        scoped.setattr(
            pipeline_module,
            "run_local_memory_curator",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("resume must not rerun Curator")
            ),
        )
        resumed = ReviewSessionResumer(
            repository=git_repo,
            checkpoint_store=checkpoint_store,
            session_store=session_store,
        ).resume()

    assert resumed.action is ResumeAction.CONTINUE_SESSION
    assert resumed.starting_phase is RunPhase.MEMORY_PROPOSAL
    completed = session_store.load()
    assert completed.status is RunStatus.COMPLETED
    assert "memory_persistence_receipt" in completed.artifacts


def test_replay_rejects_digest_valid_outbox_bound_to_another_snapshot(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    pipeline, session_store, checkpoint_store, memory_root = _pipeline(
        git_repo,
        tmp_path,
        mode=MemoryMode.READ_WRITE,
        review_id="review-memory-outbox-snapshot-tamper",
        project_rules=("Preserve the public addition contract.",),
    )
    pipeline.execute()
    outbox_path = checkpoint_store.run_dir / "memory_outbox.json"
    outbox = _json(outbox_path)
    outbox["snapshot_id"] = "MSNAP-" + "f" * 64
    body = {key: value for key, value in outbox.items() if key != "outbox_digest"}
    outbox["outbox_digest"] = canonical_sha256(body)
    encoded = json.dumps(
        outbox,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    outbox_path.write_text(encoded + "\n", encoding="utf-8")
    session_payload = _json(session_store.session_path)
    session_payload["artifacts"]["memory_outbox"]["sha256"] = hashlib.sha256(
        outbox_path.read_bytes()
    ).hexdigest()
    session_store.session_path.write_text(
        json.dumps(session_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected authority"):
        replay_memory_outbox(
            repository=git_repo,
            memory_root=memory_root,
            review_id=session_store.load().review_id,
            expected_repository_key=outbox["repository_key"],
            expected_authority_resolution_hash=outbox[
                "authority_resolution_hash"
            ],
            expected_outbox_digest=outbox["outbox_digest"],
        )

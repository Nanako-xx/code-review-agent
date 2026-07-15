from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from conftest import run_git
from review_agent.evidence import (
    CanonicalFinding,
    EvidenceReconciliation,
    reconciliation_to_dict,
)
from review_agent.memory_feedback import (
    AUTOMATIC_DURABLE_MEMORY_CONVERSION_ALLOWED,
    CalibrationAction,
    FeedbackAggregationDisposition,
    FeedbackError,
    FeedbackErrorCode,
    FeedbackImportRequest,
    FeedbackImportService,
    MissedFindingInput,
    aggregate_feedback,
    feedback_to_durable_memory,
    feedback_aggregation_v1,
    project_feedback_for_eval,
    project_feedback_for_reconciler,
    project_feedback_for_reviewer,
    project_feedback_for_scheduling,
)
from review_agent.memory_identity import repository_key
from review_agent.memory_policy import RuntimePolicyRegistry
from review_agent.memory_models import (
    FeedbackDecision,
    FeedbackReasonCode,
    FeedbackRecord,
    FeedbackStatus,
    FindingSeverity,
    FindingSnapshot,
    GitCommitSourceRef,
    HumanDeclarationOrigin,
    MemoryExecutionConfig,
    MemoryMode,
    ObservationSourceRef,
    RepositoryRangeSourceRef,
    Sensitivity,
    stable_request_id,
)
from review_agent.memory_sources import (
    SourceValidator,
    TrustedHumanDeclaration,
    repository_range_hash,
)
from review_agent.memory_store import MemoryStore
from review_agent.observations import ObservationStore
from review_agent.revision import ResolvedRevisions, RevisionResolver
from review_agent.reconciler import (
    SemanticModelSummary,
    SemanticReconciliation,
    SupplementalSemanticSummary,
    semantic_reconciliation_to_dict,
)
from review_agent.run_state import RunPhase
from review_agent.session import (
    ReviewExecutionConfig,
    initial_session_manifest,
    session_phases_for_schema,
)
from review_agent.session_store import SessionStore


NOW = "2026-07-14T08:00:00Z"
REVIEW_ID = "review-feedback-001"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finding_id(value: str) -> str:
    return "F-" + _digest(value)[:32]


def _head(repo: Path) -> str:
    return run_git(repo, "rev-parse", "HEAD")


def _repository_key(repo: Path) -> str:
    return repository_key(RevisionResolver().repository_identity(repo))


def _complete_session(
    store: SessionStore,
    artifacts_by_phase: dict[RunPhase, list[str]],
) -> None:
    current = store.load()
    for index, phase in enumerate(
        session_phases_for_schema(current.schema_version),
        start=1,
    ):
        store.mark_phase_completed(
            phase,
            artifacts_by_phase.get(phase, []),
            f"2026-07-14T09:{index:02d}:00Z",
        )
    store.mark_session_completed("2026-07-14T10:00:00Z")


def _session_authority(
    repo: Path,
    tmp_path: Path,
    *,
    review_id: str = REVIEW_ID,
    include_canonical: bool = True,
    semantic_matches_projection: bool = True,
    reconciliation_phase: RunPhase = RunPhase.RECONCILIATION,
) -> tuple[SessionStore, FindingSnapshot, ObservationSourceRef]:
    resolver = RevisionResolver()
    head = resolver.resolve_commit(repo, "HEAD")
    run_dir = repo / ".review-agent" / "runs" / review_id
    session_store = SessionStore(run_dir)
    session_store.create(
        initial_session_manifest(
            review_id=review_id,
            repository=resolver.repository_identity(repo),
            revisions=ResolvedRevisions("HEAD", "HEAD", head, head),
            execution=ReviewExecutionConfig(
                reviewer_provider="fake",
                reviewer_model=None,
                reviewer_base_url=None,
                reviewer_api_key_env="REVIEW_AGENT_API_KEY",
                reviewer_mode="single",
                reviewer_loop="single-shot",
                non_interactive=True,
                memory=MemoryExecutionConfig(
                    mode=MemoryMode.READ,
                    root_path=str((tmp_path / "memory-root").resolve()),
                ),
            ),
            now=NOW,
        )
    )

    observations = ObservationStore(run_dir)
    observation = observations.record(
        source="git.read_range",
        revision="head@" + head,
        path="app.py",
        line_start=1,
        line_end=2,
        raw_content="def add(a, b):\n    return a + b\n",
        context_view="Validated addition implementation.",
    )
    session_store.register_existing_artifact(
        name="observations",
        relative_path="observations.jsonl",
        schema="observation_log_jsonl_v1",
        phase=RunPhase.REPORTING,
        revision_binding=head + ".." + head,
        now="2026-07-14T08:01:00Z",
    )

    canonical = CanonicalFinding(
        finding_id=_finding_id(review_id),
        claim="Addition returns the wrong operand.",
        severity="high",
        confidence="high",
        path="app.py",
        line=2,
        impact="Incorrect arithmetic result.",
        suggested_action="Return a plus b.",
        verification_performed=["Inspected the exact committed range."],
        evidence_refs=[observation.observation_id],
        reviewer_indices=[],
        roles=["arithmetic specialist"],
    )
    reconciliation = EvidenceReconciliation(
        canonical_findings=[canonical] if include_canonical else [],
        rejected_findings=[],
        remaining_disagreements=[],
        contract_coverage=[],
        evidence_quality="verified",
    )
    semantic = SemanticReconciliation(
        status="local_only",
        canonical_findings=(
            tuple(reconciliation.canonical_findings)
            if semantic_matches_projection
            else ()
        ),
        rejected_findings=(),
        conflicts_resolved=(),
        remaining_disagreements=(),
        contract_coverage=(),
        evidence_quality="verified",
        supplemental=SupplementalSemanticSummary(),
        policy_actions=("deterministic_local_reconciliation",),
        uncertainties=(),
        model=SemanticModelSummary(status="disabled"),
    )
    (run_dir / "reconciliation.json").write_text(
        json.dumps(
            reconciliation_to_dict(reconciliation),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    session_store.register_existing_artifact(
        name="reconciliation",
        relative_path="reconciliation.json",
        schema="evidence_reconciliation_v1",
        phase=reconciliation_phase,
        revision_binding=head + ".." + head,
        now="2026-07-14T08:02:00Z",
    )
    (run_dir / "semantic_reconciliation.json").write_text(
        json.dumps(
            semantic_reconciliation_to_dict(semantic),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    session_store.register_existing_artifact(
        name="semantic_reconciliation",
        relative_path="semantic_reconciliation.json",
        schema="semantic_reconciliation_v1",
        phase=RunPhase.RECONCILIATION,
        revision_binding=head + ".." + head,
        now="2026-07-14T08:03:00Z",
    )
    artifacts_by_phase = {
        RunPhase.RECONCILIATION: ["semantic_reconciliation"],
        RunPhase.REPORTING: ["observations"],
    }
    artifacts_by_phase.setdefault(reconciliation_phase, []).append("reconciliation")
    _complete_session(session_store, artifacts_by_phase)

    snapshot = FindingSnapshot(
        finding_id=canonical.finding_id or "",
        claim=canonical.claim,
        path=canonical.path or "",
        line=canonical.line or 0,
        contracts=(),
        original_severity=FindingSeverity.HIGH,
        evidence_refs=tuple(canonical.evidence_refs),
    )
    observation_ref = ObservationSourceRef(
        review_id=review_id,
        observation_id=observation.observation_id,
        revision_binding=observation.revision,
        content_hash=observation.content_hash,
    )
    return session_store, snapshot, observation_ref


def _service(
    repo: Path,
    tmp_path: Path,
    *,
    declarations: tuple[TrustedHumanDeclaration, ...] = (),
    contract_ids: tuple[str, ...] = ("behavioral_correctness",),
) -> tuple[FeedbackImportService, MemoryStore]:
    store = MemoryStore(tmp_path / "memory")
    validator = SourceValidator(
        repo,
        sessions_root=repo / ".review-agent" / "runs",
        human_declarations=declarations,
    )
    return (
        FeedbackImportService(
            store,
            validator,
            RuntimePolicyRegistry(contract_ids=contract_ids),
        ),
        store,
    )


def _request(
    repo: Path,
    snapshot: FindingSnapshot,
    *,
    decision: FeedbackDecision = FeedbackDecision.ACCEPTED,
    final_severity: FindingSeverity = FindingSeverity.HIGH,
    request_id: str | None = None,
    reason: str = "Maintainer confirmed the reported behavior.",
) -> FeedbackImportRequest:
    return FeedbackImportRequest(
        request_id=request_id or stable_request_id("feedback", snapshot.finding_id),
        repository_key=_repository_key(repo),
        review_id=REVIEW_ID,
        finding_id=snapshot.finding_id,
        head_sha=_head(repo),
        finding_hash=snapshot.finding_hash,
        evidence_refs=snapshot.evidence_refs,
        decision=decision,
        final_severity=final_severity,
        reason_code=(
            FeedbackReasonCode.SEVERITY_MISMATCH
            if decision is FeedbackDecision.SEVERITY_CHANGED
            else FeedbackReasonCode.OTHER
        ),
        reason=reason,
        actor="amy",
        created_at=NOW,
    )


def _missed_request(
    repo: Path,
    snapshot: FindingSnapshot,
    observation_ref: ObservationSourceRef,
    *,
    finding_id: str,
    line: int,
    contracts: tuple[str, ...],
) -> tuple[FeedbackImportRequest, TrustedHumanDeclaration]:
    request_id = stable_request_id("feedback-missed-audit", finding_id)
    declaration = TrustedHumanDeclaration(
        request_id=request_id,
        actor="amy",
        created_at=NOW,
        declaration="Maintainer explicitly identified a missed Finding.",
        origin=HumanDeclarationOrigin.USER_REQUEST,
        review_id=REVIEW_ID,
    )
    missed = MissedFindingInput(
        finding_id=finding_id,
        claim=snapshot.claim,
        path=snapshot.path,
        line=line,
        contracts=contracts,
        original_severity=snapshot.original_severity,
        evidence_refs=snapshot.evidence_refs,
    )
    missed_snapshot = missed.materialize()
    authority = declaration.to_authority()
    return (
        FeedbackImportRequest(
            request_id=request_id,
            repository_key=_repository_key(repo),
            review_id=REVIEW_ID,
            finding_id=missed_snapshot.finding_id,
            head_sha=_head(repo),
            finding_hash=missed_snapshot.finding_hash,
            evidence_refs=missed_snapshot.evidence_refs,
            decision=FeedbackDecision.MISSED,
            final_severity=missed_snapshot.original_severity,
            reason_code=FeedbackReasonCode.OTHER,
            reason="Maintainer confirmed the review missed this Finding.",
            actor="amy",
            created_at=NOW,
            missed_finding=missed,
            human_declaration=authority,
            source_refs=(authority.source_ref, observation_ref),
        ),
        declaration,
    )


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"finding_id": _finding_id("unknown")}, FeedbackErrorCode.FINDING_NOT_FOUND),
        ({"head_sha": "a" * 40}, FeedbackErrorCode.HEAD_MISMATCH),
        ({"finding_hash": "b" * 64}, FeedbackErrorCode.FINDING_HASH_MISMATCH),
        ({"evidence_refs": ("O-" + "c" * 32,)}, FeedbackErrorCode.EVIDENCE_MISMATCH),
    ],
)
def test_session_bound_feedback_fails_closed_on_finding_authority_mismatch(
    git_repo: Path,
    tmp_path: Path,
    change: dict[str, object],
    code: FeedbackErrorCode,
) -> None:
    _session, snapshot, _observation = _session_authority(git_repo, tmp_path)
    service, store = _service(git_repo, tmp_path)

    with pytest.raises(FeedbackError) as raised:
        service.record_feedback(replace(_request(git_repo, snapshot), **change))

    assert raised.value.code is code
    assert store.list_feedback(_repository_key(git_repo)) == ()


def test_feedback_rejects_a_candidate_identity_absent_from_final_reconciliation(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    _session, snapshot, _observation = _session_authority(
        git_repo,
        tmp_path,
        include_canonical=False,
    )
    service, store = _service(git_repo, tmp_path)

    with pytest.raises(FeedbackError) as raised:
        service.record_feedback(_request(git_repo, snapshot))

    assert raised.value.code is FeedbackErrorCode.FINDING_NOT_FOUND
    assert store.list_feedback(_repository_key(git_repo)) == ()


@pytest.mark.parametrize(
    "session_options",
    [
        {"semantic_matches_projection": False},
        {"reconciliation_phase": RunPhase.REPORTING},
    ],
)
def test_feedback_requires_final_semantic_reconciliation_authority(
    git_repo: Path,
    tmp_path: Path,
    session_options: dict[str, object],
) -> None:
    _session, snapshot, _observation = _session_authority(
        git_repo,
        tmp_path,
        **session_options,
    )
    service, store = _service(git_repo, tmp_path)

    with pytest.raises(FeedbackError) as raised:
        service.record_feedback(_request(git_repo, snapshot))

    assert raised.value.code is FeedbackErrorCode.SESSION_UNTRUSTED
    assert store.list_feedback(_repository_key(git_repo)) == ()


def test_feedback_rejects_sensitive_actor_before_persistence(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    _session, snapshot, _observation = _session_authority(git_repo, tmp_path)
    service, store = _service(git_repo, tmp_path)
    request = replace(
        _request(git_repo, snapshot),
        actor="ghp_abcdefghijklmnopqrstuvwxyz0123456789",
    )

    with pytest.raises(FeedbackError) as raised:
        service.record_feedback(request)

    assert raised.value.code is FeedbackErrorCode.SOURCE_VALIDATION_FAILED
    assert store.list_feedback(_repository_key(git_repo)) == ()


@pytest.mark.parametrize(
    ("decision", "final_severity"),
    [
        (FeedbackDecision.ACCEPTED, FindingSeverity.HIGH),
        (FeedbackDecision.REJECTED, FindingSeverity.HIGH),
        (FeedbackDecision.SEVERITY_CHANGED, FindingSeverity.MEDIUM),
    ],
)
def test_session_bound_decisions_copy_the_minimal_immutable_snapshot(
    git_repo: Path,
    tmp_path: Path,
    decision: FeedbackDecision,
    final_severity: FindingSeverity,
) -> None:
    _session, snapshot, _observation = _session_authority(git_repo, tmp_path)
    service, _store = _service(git_repo, tmp_path)

    result = service.record_feedback(
        _request(
            git_repo,
            snapshot,
            decision=decision,
            final_severity=final_severity,
            request_id=stable_request_id("feedback-decision", decision.value),
        )
    )

    assert result.record.finding_snapshot == snapshot
    assert result.record.finding_snapshot is not snapshot
    assert result.record.finding_snapshot.finding_hash == snapshot.finding_hash
    assert result.record.source_refs
    assert result.validation.valid


def test_missed_requires_human_declaration_and_verifiable_typed_source(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    _session, canonical, observation_ref = _session_authority(git_repo, tmp_path)
    request_id = stable_request_id("feedback-missed", canonical.finding_id)
    declaration = TrustedHumanDeclaration(
        request_id=request_id,
        actor="amy",
        created_at=NOW,
        declaration="The review missed an incorrect boundary operation.",
        origin=HumanDeclarationOrigin.USER_REQUEST,
        review_id=REVIEW_ID,
    )
    authority = declaration.to_authority()
    missed = MissedFindingInput(
        finding_id=_finding_id("missed-boundary-operation"),
        claim="Boundary operation can return an incorrect result.",
        path="app.py",
        line=2,
        contracts=("behavioral_correctness",),
        original_severity=FindingSeverity.HIGH,
        evidence_refs=(observation_ref.observation_id,),
    )
    snapshot = missed.materialize()
    base = FeedbackImportRequest(
        request_id=request_id,
        repository_key=_repository_key(git_repo),
        review_id=REVIEW_ID,
        finding_id=snapshot.finding_id,
        head_sha=_head(git_repo),
        finding_hash=snapshot.finding_hash,
        evidence_refs=snapshot.evidence_refs,
        decision=FeedbackDecision.MISSED,
        final_severity=FindingSeverity.HIGH,
        reason_code=FeedbackReasonCode.OTHER,
        reason="Maintainer declared a missed finding with exact evidence.",
        actor="amy",
        created_at=NOW,
        missed_finding=missed,
    )
    service, store = _service(
        git_repo,
        tmp_path,
        declarations=(declaration,),
    )

    with pytest.raises(FeedbackError) as missing_declaration:
        service.record_feedback(base)
    assert missing_declaration.value.code is FeedbackErrorCode.HUMAN_DECLARATION_REQUIRED

    declaration_only = replace(
        base,
        human_declaration=authority,
        source_refs=(authority.source_ref,),
    )
    with pytest.raises(FeedbackError) as missing_evidence:
        service.record_feedback(declaration_only)
    assert missing_evidence.value.code is FeedbackErrorCode.VERIFIABLE_SOURCE_REQUIRED

    range_ref = RepositoryRangeSourceRef(
        revision=_head(git_repo),
        path="app.py",
        line_start=1,
        line_end=2,
        content_hash=repository_range_hash(git_repo, _head(git_repo), "app.py", 1, 2),
    )
    result = service.record_feedback(
        replace(
            base,
            human_declaration=authority,
            source_refs=(authority.source_ref, range_ref, observation_ref),
        )
    )

    assert result.record.decision is FeedbackDecision.MISSED
    assert result.record.finding_snapshot == snapshot
    assert store.get_feedback(result.record.feedback_id) == result.record


def test_missed_rejects_an_id_already_present_in_final_canonical_findings(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    _session, canonical, observation_ref = _session_authority(git_repo, tmp_path)
    request, declaration = _missed_request(
        git_repo,
        canonical,
        observation_ref,
        finding_id=canonical.finding_id,
        line=canonical.line,
        contracts=canonical.contracts,
    )
    service, store = _service(
        git_repo,
        tmp_path,
        declarations=(declaration,),
    )

    with pytest.raises(FeedbackError) as raised:
        service.record_feedback(request)

    assert raised.value.code is FeedbackErrorCode.FINDING_NOT_CANONICAL
    assert store.list_feedback(_repository_key(git_repo)) == ()


@pytest.mark.parametrize(
    ("line", "contracts", "expected_code"),
    [
        (999, ("behavioral_correctness",), FeedbackErrorCode.VERIFIABLE_SOURCE_REQUIRED),
        (2, ("invented_contract",), FeedbackErrorCode.INVALID_INPUT),
    ],
)
def test_missed_requires_location_coverage_and_registered_contracts(
    git_repo: Path,
    tmp_path: Path,
    line: int,
    contracts: tuple[str, ...],
    expected_code: FeedbackErrorCode,
) -> None:
    _session, canonical, observation_ref = _session_authority(git_repo, tmp_path)
    request, declaration = _missed_request(
        git_repo,
        canonical,
        observation_ref,
        finding_id=_finding_id("new-missed-finding"),
        line=line,
        contracts=contracts,
    )
    service, store = _service(
        git_repo,
        tmp_path,
        declarations=(declaration,),
    )

    with pytest.raises(FeedbackError) as raised:
        service.record_feedback(request)

    assert raised.value.code is expected_code
    assert store.list_feedback(_repository_key(git_repo)) == ()


def test_retry_is_idempotent_and_conflicting_decision_never_overwrites(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    _session, snapshot, _observation = _session_authority(git_repo, tmp_path)
    service, store = _service(git_repo, tmp_path)
    request = _request(git_repo, snapshot)

    first = service.record_feedback(request)
    replay = service.record_feedback(request)

    assert first.write_result.applied
    assert replay.write_result.replayed
    assert not replay.write_result.applied
    assert replay.record == first.record
    assert store.verify_event_chain(_repository_key(git_repo)) == 1

    conflict = replace(
        request,
        decision=FeedbackDecision.REJECTED,
        reason="A contradictory decision must not overwrite the first one.",
    )
    with pytest.raises(FeedbackError) as raised:
        service.record_feedback(conflict)
    assert raised.value.code is FeedbackErrorCode.CONFLICTING_DECISION
    assert store.list_feedback(_repository_key(git_repo)) == (first.record,)
    assert store.verify_event_chain(_repository_key(git_repo)) == 1


def _stored_feedback(
    store: MemoryStore,
    repository_key_value: str,
    *,
    index: int,
    review_id: str,
    decision: FeedbackDecision = FeedbackDecision.MISSED,
    reason_code: FeedbackReasonCode = FeedbackReasonCode.INSUFFICIENT_EVIDENCE,
    path: str = "app.py",
    contract: str = "behavioral_correctness",
    raw_reason: str = "private human explanation that must not be projected",
) -> FeedbackRecord:
    original = FindingSeverity.HIGH
    final = FindingSeverity.MEDIUM if decision is FeedbackDecision.SEVERITY_CHANGED else original
    snapshot = FindingSnapshot(
        finding_id=_finding_id(f"{review_id}-{index}"),
        claim=f"raw claim {index} that must not be projected",
        path=path,
        line=index + 1,
        contracts=(contract,),
        original_severity=original,
        evidence_refs=("O-" + _digest(f"observation-{index}")[:32],),
    )
    record = FeedbackRecord(
        repository_key=repository_key_value,
        review_id=review_id,
        finding_id=snapshot.finding_id,
        head_sha="a" * 40,
        finding_snapshot=snapshot,
        decision=decision,
        original_severity=original,
        final_severity=final,
        reason_code=reason_code,
        reason=raw_reason,
        actor="amy",
        source_refs=(GitCommitSourceRef(commit_sha="a" * 40),),
        status=FeedbackStatus.RECORDED,
        created_at=f"2026-07-{10 + index:02d}T08:00:00Z",
    )
    store.put_feedback(
        record,
        request_id=stable_request_id("aggregate-feedback", record.feedback_id),
    )
    return record


def test_aggregation_requires_five_comparable_records_and_three_reviews(
    tmp_path: Path,
) -> None:
    repository_key_value = _digest("aggregate-repository")
    store = MemoryStore(tmp_path / "memory")
    records = tuple(
        _stored_feedback(
            store,
            repository_key_value,
            index=index,
            review_id=f"review-{index % 3}",
        )
        for index in range(5)
    )

    result = aggregate_feedback(store, repository_key_value)
    reordered = feedback_aggregation_v1(
        tuple(reversed(records)),
        repository_key=repository_key_value,
        feedback_generation=store.get_generations(
            repository_key_value
        ).feedback_generation,
    )

    assert reordered == result
    assert result.summary.eligible
    assert result.summary.policy_version == "feedback_aggregation_v1"
    assert result.summary.source_feedback_ids == tuple(
        sorted(record.feedback_id for record in records)
    )
    assert result.first_created_at == records[0].created_at
    assert result.last_created_at == records[-1].created_at
    assert result.sample_count == 5
    assert result.review_count == 3
    assert len(result.groups) == 1
    group = result.groups[0]
    assert group.disposition is FeedbackAggregationDisposition.CALIBRATION
    assert group.sample_count == 5
    assert group.review_count == 3
    assert group.first_created_at == records[0].created_at
    assert group.last_created_at == records[-1].created_at
    assert group.feedback_ids == tuple(sorted(record.feedback_id for record in records))


def test_incomparable_or_insufficient_samples_are_eval_only(
    tmp_path: Path,
) -> None:
    repository_key_value = _digest("eval-only-repository")
    store = MemoryStore(tmp_path / "memory")
    for index in range(5):
        _stored_feedback(
            store,
            repository_key_value,
            index=index,
            review_id=f"review-{index % 3}",
            path="app.py" if index < 4 else "different.py",
        )

    result = aggregate_feedback(store, repository_key_value)

    assert not result.summary.eligible
    assert all(
        group.disposition is FeedbackAggregationDisposition.EVAL_ONLY
        for group in result.groups
    )
    assert result.context_summary is None
    assert project_feedback_for_reviewer(result) is None
    assert project_feedback_for_reconciler(result) is None
    assert project_feedback_for_scheduling(result) is None
    evaluation = project_feedback_for_eval(result)
    assert evaluation.sample_count == 5
    assert evaluation.groups


def test_safe_projection_contains_only_taxonomy_counts_and_monotonic_actions(
    tmp_path: Path,
) -> None:
    repository_key_value = _digest("safe-projection-repository")
    store = MemoryStore(tmp_path / "memory")
    forbidden_reason = "RAW-REASON-DO-NOT-PROJECT"
    for index in range(5):
        _stored_feedback(
            store,
            repository_key_value,
            index=index,
            review_id=f"review-{index % 3}",
            decision=FeedbackDecision.REJECTED,
            raw_reason=forbidden_reason,
        )

    result = aggregate_feedback(store, repository_key_value)
    reviewer = project_feedback_for_reviewer(result)
    reconciler = project_feedback_for_reconciler(result)
    assert reviewer is not None and reconciler is not None
    assert reviewer == reconciler
    assert {group.action for group in reviewer.groups} <= {
        CalibrationAction.RAISE_CHECK_PRIORITY,
        CalibrationAction.RAISE_PERSPECTIVE_PRIORITY,
        CalibrationAction.DEMAND_MORE_EVIDENCE,
    }

    payload = json.dumps(reviewer.to_dict(), sort_keys=True)
    assert forbidden_reason not in payload
    assert "raw claim" not in payload
    assert "finding_snapshot" not in payload
    assert "feedback_record" not in payload
    assert "suppress" not in payload.casefold()
    assert "lower_risk" not in payload.casefold()
    assert "lower_severity" not in payload.casefold()


def test_aggregation_survives_original_session_deletion(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    session, snapshot, _observation = _session_authority(git_repo, tmp_path)
    service, store = _service(git_repo, tmp_path)
    imported = service.record_feedback(_request(git_repo, snapshot))

    shutil.rmtree(session.run_dir)
    replay = service.record_feedback(_request(git_repo, snapshot))
    result = aggregate_feedback(store, _repository_key(git_repo))

    assert replay.write_result.replayed
    assert result.sample_count == 1
    assert result.summary.source_feedback_ids == (imported.record.feedback_id,)
    assert result.groups[0].disposition is FeedbackAggregationDisposition.EVAL_ONLY
    assert store.get_feedback(imported.record.feedback_id).finding_snapshot == snapshot


def test_feedback_can_never_be_automatically_converted_to_durable_memory(
    tmp_path: Path,
) -> None:
    repository_key_value = _digest("no-conversion-repository")
    store = MemoryStore(tmp_path / "memory")
    record = _stored_feedback(
        store,
        repository_key_value,
        index=0,
        review_id="review-0",
    )

    assert AUTOMATIC_DURABLE_MEMORY_CONVERSION_ALLOWED is False
    with pytest.raises(FeedbackError) as raised:
        feedback_to_durable_memory(record)
    assert raised.value.code is FeedbackErrorCode.DURABLE_MEMORY_CONVERSION_PROHIBITED

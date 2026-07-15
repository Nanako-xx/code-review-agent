from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from conftest import run_git
import review_agent.command as command_module
import review_agent.pipeline as pipeline_module
from review_agent.command import _build_parser, main
from review_agent.evidence import (
    CanonicalFinding,
    EvidenceReconciliation,
    reconciliation_to_dict,
)
from review_agent.memory_feedback import FeedbackErrorCode
from review_agent.memory_identity import (
    MemoryRootResolver,
    build_repository_memory_namespace,
    repository_key,
    repository_namespace_path,
)
from review_agent.memory_lifecycle import (
    MemoryLifecycle,
    MemoryLifecycleError,
    MemoryLifecycleErrorCode,
)
from review_agent.memory_models import (
    CandidateStatus,
    FeedbackDecision,
    FeedbackReasonCode,
    FindingSeverity,
    FindingSnapshot,
    MAX_SNAPSHOT_RECORDS,
    MemoryCandidate,
    MemoryConfidence,
    MemoryExecutionConfig,
    MemoryKind,
    MemoryMode,
    MemoryScope,
    PolicyEffect,
    PolicyEffectKind,
    Producer,
    ProducerType,
    RecordStatus,
    RepositoryRangeSourceRef,
    Sensitivity,
    ValidityPolicy,
    canonical_sha256,
    stable_request_id,
)
from review_agent.memory_sources import (
    SourceValidationCode,
    SourceValidator,
    TrustedCandidateProvenance,
    candidate_authority_resolution_hash,
    repository_range_hash,
)
from review_agent.memory_relink import (
    RepositoryRelinkConflictError,
    RepositoryRelinkRegistry,
)
from review_agent.memory_store import (
    MemoryStore,
    MemoryStoreValidationError,
)
from review_agent.models import RiskAssessment, RiskLevel
from review_agent.observations import ObservationStore
from review_agent.portfolio import (
    DEFAULT_CONTRACT_ALLOWLIST,
    build_portfolio_packet,
    portfolio_packet_to_dict,
)
from review_agent.reconciler import (
    SemanticModelSummary,
    SemanticReconciliation,
    SupplementalSemanticSummary,
    semantic_reconciliation_to_dict,
)
from review_agent.revision import ResolvedRevisions, RevisionResolver
from review_agent.run_state import RunPhase
from review_agent.session import (
    ReviewExecutionConfig,
    initial_session_manifest,
    session_phases_for_schema,
)
from review_agent.session_store import SessionStore


NOW = "2026-07-14T08:00:00Z"


def _namespace(repo: Path, memory_root: Path):
    resolution = MemoryRootResolver().resolve(memory_root.resolve(), create=True)
    identity = RevisionResolver().repository_identity(repo)
    return build_repository_memory_namespace(identity, resolution)


def _store(repo: Path, memory_root: Path) -> MemoryStore:
    return MemoryStore(_namespace(repo, memory_root))


def _head(repo: Path) -> str:
    return run_git(repo, "rev-parse", "HEAD")


def _finding_id(value: str) -> str:
    return "F-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _complete_session(
    store: SessionStore,
    artifacts_by_phase: dict[RunPhase, list[str]],
) -> None:
    manifest = store.load()
    for index, phase in enumerate(
        session_phases_for_schema(manifest.schema_version),
        start=1,
    ):
        store.mark_phase_completed(
            phase,
            artifacts_by_phase.get(phase, []),
            "2026-07-14T09:%02d:00Z" % index,
        )
    store.mark_session_completed("2026-07-14T10:00:00Z")


def _feedback_outbox_session(
    repo: Path,
    tmp_path: Path,
    *,
    review_id: str,
) -> tuple[SessionStore, FindingSnapshot, str]:
    resolver = RevisionResolver()
    head = resolver.resolve_commit(repo, "HEAD")
    identity = resolver.repository_identity(repo)
    run_dir = repo / ".review-agent" / "runs" / review_id
    session_store = SessionStore(run_dir)
    session_store.create(
        initial_session_manifest(
            review_id=review_id,
            repository=identity,
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
                    root_path=str((tmp_path / "session-memory-config").resolve()),
                ),
            ),
            now=NOW,
        )
    )

    portfolio_packet = build_portfolio_packet(
        RiskAssessment(
            level=RiskLevel.LOW,
            dimensions={
                "impact": "low",
                "blast_radius": "low",
                "reversibility": "high",
                "uncertainty": "low",
                "verification_strength": "high",
            },
            reasons=["Deterministic CLI fixture."],
            signal_refs=[],
            uncertainties=[],
            suggested_focus=["Behavioral correctness."],
        ),
        contract_allowlist=DEFAULT_CONTRACT_ALLOWLIST,
    )
    (run_dir / "portfolio_packet.json").write_text(
        json.dumps(
            portfolio_packet_to_dict(portfolio_packet),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    session_store.register_existing_artifact(
        name="portfolio_packet",
        relative_path="portfolio_packet.json",
        schema="portfolio_packet_v1",
        phase=RunPhase.PLANNING,
        revision_binding=head + ".." + head,
        now="2026-07-14T08:01:00Z",
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
        now="2026-07-14T08:02:00Z",
    )

    finding = CanonicalFinding(
        finding_id=_finding_id(review_id),
        claim="Addition returns the wrong operand.",
        severity="high",
        confidence="high",
        path="app.py",
        line=2,
        impact="Incorrect arithmetic result.",
        suggested_action="Return a plus b.",
        verification_performed=["Inspected the committed range."],
        evidence_refs=[observation.observation_id],
        reviewer_indices=[],
        roles=["arithmetic specialist"],
    )
    reconciliation = EvidenceReconciliation(
        canonical_findings=[finding],
        rejected_findings=[],
        remaining_disagreements=[],
        contract_coverage=[],
        evidence_quality="verified",
    )
    semantic = SemanticReconciliation(
        status="local_only",
        canonical_findings=(finding,),
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
        phase=RunPhase.RECONCILIATION,
        revision_binding=head + ".." + head,
        now="2026-07-14T08:03:00Z",
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
        now="2026-07-14T08:04:00Z",
    )

    batch_digest = hashlib.sha256(b"empty-candidate-batch").hexdigest()
    repository_key_value = repository_key(identity)
    outbox_body = {
        "schema": "memory_candidate_outbox_v1",
        "review_id": review_id,
        "repository_key": repository_key_value,
        "locator_repository_key": repository_key_value,
        "authority_resolution_hash": candidate_authority_resolution_hash(
            repository_key_value,
            repository_key_value,
        ),
        "binding_id": None,
        "head_sha": head,
        "snapshot_id": "MSNAP-" + hashlib.sha256(b"snapshot").hexdigest(),
        "batch_digest": batch_digest,
        "actor_type": "runtime",
        "actor_id": "memory-curator",
        "reason_code": "candidate_outbox",
        "entries": [],
    }
    outbox_payload = {
        **outbox_body,
        "outbox_digest": canonical_sha256(outbox_body),
    }
    (run_dir / "memory_outbox.json").write_text(
        json.dumps(
            outbox_payload,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    session_store.register_existing_artifact(
        name="memory_outbox",
        relative_path="memory_outbox.json",
        schema="memory_candidate_outbox_v1",
        phase=RunPhase.MEMORY_PROPOSAL,
        revision_binding=head + ".." + head,
        now="2026-07-14T08:05:00Z",
    )
    _complete_session(
        session_store,
        {
            RunPhase.PLANNING: ["portfolio_packet"],
            RunPhase.MEMORY_PROPOSAL: ["memory_outbox"],
            RunPhase.RECONCILIATION: [
                "reconciliation",
                "semantic_reconciliation",
            ],
            RunPhase.REPORTING: ["observations"],
        },
    )
    snapshot = FindingSnapshot(
        finding_id=finding.finding_id or "",
        claim=finding.claim,
        path=finding.path or "",
        line=finding.line or 0,
        contracts=(),
        original_severity=FindingSeverity.HIGH,
        evidence_refs=tuple(finding.evidence_refs),
    )
    return session_store, snapshot, str(outbox_payload["outbox_digest"])


def _candidate(
    repo: Path,
    repository_key: str,
    *,
    statement: str,
    review_id: str,
    policy: bool = False,
) -> MemoryCandidate:
    revision = _head(repo)
    source = RepositoryRangeSourceRef(
        revision=revision,
        path="app.py",
        line_start=1,
        line_end=2,
        content_hash=repository_range_hash(repo, revision, "app.py", 1, 2),
    )
    return MemoryCandidate(
        repository_key=repository_key,
        kind=MemoryKind.REVIEW_RULE,
        statement=statement,
        scope=MemoryScope(paths=("app.py",)),
        source_refs=(source,),
        valid_from_sha=revision,
        validity_policies=(ValidityPolicy.SOURCE_CONTENT_HASH,),
        confidence=MemoryConfidence.HIGH,
        sensitivity=Sensitivity.NORMAL,
        policy_effect=(
            PolicyEffect(PolicyEffectKind.RISK_FLOOR, "high") if policy else None
        ),
        producer=Producer(ProducerType.MODEL, "memory-curator", "1.0"),
        origin_review_id=review_id,
        status=CandidateStatus.PROPOSED,
        created_at=NOW,
    )


def _provenance(candidate: MemoryCandidate) -> TrustedCandidateProvenance:
    repository_key_value = candidate.repository_key
    return TrustedCandidateProvenance(
        origin=ProducerType.MODEL,
        review_id=candidate.origin_review_id,
        target_head_sha=candidate.valid_from_sha,
        locator_repository_key=repository_key_value,
        authority_repository_key=repository_key_value,
        authority_resolution_hash=candidate_authority_resolution_hash(
            repository_key_value,
            repository_key_value,
        ),
        allowed_source_refs=candidate.source_refs,
    )


def _submit(repo: Path, store: MemoryStore, candidate: MemoryCandidate) -> None:
    MemoryLifecycle(store, SourceValidator(repo)).submit_candidate(
        candidate,
        runtime_provenance=_provenance(candidate),
        request_id=stable_request_id("memory-cli-test-submit", candidate.candidate_id),
    )


def _approve_direct(repo: Path, store: MemoryStore, candidate: MemoryCandidate):
    return MemoryLifecycle(store, SourceValidator(repo)).approve_candidate(
        candidate.candidate_id,
        runtime_provenance=_provenance(candidate),
        actor="test-maintainer",
        reason="Test setup approval with reviewed evidence.",
        request_id=stable_request_id("memory-cli-test-approve", candidate.candidate_id),
    ).record


def _cli(repo: Path, memory_root: Path, *arguments: str) -> int:
    return main(
        [
            "memory",
            *arguments,
            "--repo",
            str(repo),
            "--memory-root",
            str(memory_root.resolve()),
            "--json",
        ]
    )


def _documents(output: str) -> list[dict]:
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def _valid_replay_receipt(
    call: dict[str, object],
    *,
    locator_repository_key: str | None = None,
    binding_id: str | None = None,
) -> dict[str, object]:
    repository_key_value = str(call["expected_repository_key"])
    body: dict[str, object] = {
        "schema": "memory_persistence_receipt_v1",
        "success": True,
        "review_id": str(call["review_id"]),
        "repository_key": repository_key_value,
        "locator_repository_key": (
            repository_key_value
            if locator_repository_key is None
            else locator_repository_key
        ),
        "authority_resolution_hash": str(
            call["expected_authority_resolution_hash"]
        ),
        "binding_id": binding_id,
        "outbox_digest": str(call["expected_outbox_digest"]),
        "batch_digest": hashlib.sha256(b"empty-candidate-batch").hexdigest(),
        "persisted_candidate_ids": [],
        "replayed_candidate_ids": [],
        "results": [],
    }
    return {**body, "receipt_digest": canonical_sha256(body)}


def test_parser_core_only() -> None:
    parser = _build_parser()
    parsed = parser.parse_args(
        [
            "memory",
            "revalidate",
            "MEM-" + "1" * 64,
            "--candidate",
            "MC-" + "2" * 64,
            "--actor",
            "amy",
            "--reason",
            "Evidence was refreshed.",
            "--yes",
        ]
    )
    assert parsed.memory_action == "revalidate"
    assert parsed.candidate_id.startswith("MC-")
    relink = parser.parse_args(
        [
            "memory",
            "relink",
            "--from-key",
            "4" * 64,
            "--actor",
            "amy",
            "--reason",
            "Restore the explicitly selected local authority.",
            "--yes",
        ]
    )
    assert relink.memory_action == "relink"
    assert relink.from_repository_key == "4" * 64
    assert main(["memory"]) == 2
    with pytest.raises(SystemExit) as missing_actor:
        parser.parse_args(
            ["memory", "approve", "MC-" + "3" * 64, "--reason", "ok"]
        )
    assert missing_actor.value.code == 2
    feedback = parser.parse_args(
        [
            "memory",
            "feedback",
            "record",
            "review-feedback",
            "F-" + "4" * 32,
            "--head-sha",
            "5" * 40,
            "--finding-hash",
            "6" * 64,
            "--evidence-ref",
            "O-" + "7" * 32,
            "--decision",
            "accepted",
            "--final-severity",
            "high",
            "--reason-code",
            "other",
            "--actor",
            "amy",
            "--reason",
            "Finding disposition was confirmed.",
            "--yes",
        ]
    )
    assert feedback.memory_action == "feedback_record"
    assert feedback.review_id == "review-feedback"
    assert feedback.finding_id == "F-" + "4" * 32

    feedback_list = parser.parse_args(["memory", "feedback", "list"])
    assert feedback_list.memory_action == "feedback_list"

    replay = parser.parse_args(
        [
            "memory",
            "replay-outbox",
            "review-feedback",
            "--actor",
            "amy",
            "--reason",
            "Replay the hash-verified Session outbox.",
            "--yes",
        ]
    )
    assert replay.memory_action == "replay_outbox"
    assert replay.review_id == "review-feedback"

    with pytest.raises(SystemExit):
        parser.parse_args(["memory", "feedback"])
    with pytest.raises(SystemExit):
        parser.parse_args(["memory", "replay-outbox"])


def _review_args(*arguments: str):
    return _build_parser().parse_args(
        ["review", "--base", "HEAD~1", "--head", "HEAD", *arguments]
    )


def test_review_memory_config_defaults_and_root_priority_are_fixed_without_creation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    environment_root = (tmp_path / "environment-memory").resolve()
    cli_root = (tmp_path / "cli-memory").resolve()
    monkeypatch.setenv("REVIEW_AGENT_MEMORY_ROOT", str(environment_root))

    environment_config = command_module._resolve_memory_execution_config(
        _review_args()
    )
    cli_config = command_module._resolve_memory_execution_config(
        _review_args("--memory-root", str(cli_root))
    )

    assert environment_config.mode is MemoryMode.READ_WRITE
    assert environment_config.root_path == environment_root.as_posix()
    assert cli_config.root_path == cli_root.as_posix()
    assert not environment_root.exists()
    assert not cli_root.exists()


def test_review_memory_config_validates_required_mode_and_snapshot_budgets(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "memory").resolve()

    with pytest.raises(ValueError, match="required=true cannot be combined"):
        command_module._resolve_memory_execution_config(
            _review_args(
                "--memory-root",
                str(root),
                "--memory-mode",
                "off",
                "--memory-required",
            )
        )

    with pytest.raises(ValueError, match="max_snapshot_records"):
        command_module._resolve_memory_execution_config(
            _review_args(
                "--memory-root",
                str(root),
                "--memory-max-snapshot-records",
                str(MAX_SNAPSHOT_RECORDS + 1),
            )
        )

    config = command_module._resolve_memory_execution_config(
        _review_args(
            "--memory-root",
            str(root),
            "--memory-mode",
            "read",
            "--memory-max-snapshot-records",
            "20",
            "--memory-max-snapshot-bytes",
            "4096",
            "--memory-max-context-records",
            "4",
            "--memory-max-query-results",
            "3",
        )
    )
    assert config.mode is MemoryMode.READ
    assert config.max_snapshot_records == 20
    assert config.max_snapshot_bytes == 4096
    assert config.max_context_records == 4
    assert config.max_query_results == 3


@pytest.mark.parametrize(
    "json_arguments",
    (
        ("--json",),
        ("--format", "json"),
        ("--format=json",),
        ("--form", "json"),
        ("--j",),
    ),
)
def test_memory_json_parser_errors_are_sanitized(
    capsys,
    json_arguments: tuple[str, ...],
) -> None:
    secret = "credential-that-must-not-be-echoed"

    assert (
        main(
            [
                "memory",
                "approve",
                "MC-" + "3" * 64,
                "--reason",
                secret,
                *json_arguments,
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    document = json.loads(captured.err)
    assert document == {
        "error": {
            "code": "usage",
            "message": "memory command arguments are invalid",
        },
        "schema": "memory_cli_v1",
        "type": "error",
    }
    assert secret not in captured.err


def test_lifecycle_state_conflicts_use_conflict_exit(monkeypatch, capsys) -> None:
    def fail_with_conflict(_args) -> int:
        raise MemoryLifecycleError(
            "sensitive transition detail",
            MemoryLifecycleErrorCode.INVALID_TRANSITION,
        )

    monkeypatch.setattr(command_module, "_memory_approve", fail_with_conflict)
    assert (
        main(
            [
                "memory",
                "approve",
                "MC-" + "3" * 64,
                "--actor",
                "amy",
                "--reason",
                "Reviewed evidence.",
                "--json",
            ]
        )
        == 4
    )
    captured = capsys.readouterr()
    document = json.loads(captured.err)
    assert document["error"]["code"] == "lifecycle_invalid_transition"
    assert "sensitive transition detail" not in captured.err


def test_absent_store_status_does_not_materialize_memory_root(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    memory_root = tmp_path / "absent-memory-root"
    assert not memory_root.exists()

    assert _cli(git_repo, memory_root, "status") == 0

    document = _documents(capsys.readouterr().out)[-1]
    assert document["store_present"] is False
    assert not memory_root.exists()


def test_feedback_list_is_read_only_and_absent_store_is_empty(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    memory_root = tmp_path / "absent-feedback-memory"

    assert _cli(git_repo, memory_root, "feedback", "list") == 0

    document = _documents(capsys.readouterr().out)[-1]
    assert document["type"] == "feedback_list"
    assert document["count"] == 0
    assert document["feedback"] == []
    assert document["generations"]["feedback_generation"] == 0
    assert not memory_root.exists()


def test_feedback_record_requires_explicit_confirmation_before_service_write(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    memory_root = tmp_path / "feedback-memory"
    review_id = "review-feedback-confirmation"
    _session, snapshot, _outbox_digest = _feedback_outbox_session(
        git_repo,
        tmp_path,
        review_id=review_id,
    )
    arguments = (
        "feedback",
        "record",
        review_id,
        snapshot.finding_id,
        "--head-sha",
        _head(git_repo),
        "--finding-hash",
        snapshot.finding_hash,
        "--evidence-ref",
        snapshot.evidence_refs[0],
        "--decision",
        FeedbackDecision.ACCEPTED.value,
        "--final-severity",
        FindingSeverity.HIGH.value,
        "--reason-code",
        FeedbackReasonCode.OTHER.value,
        "--actor",
        "amy",
        "--reason",
        "Maintainer confirmed the canonical Finding.",
        "--created-at",
        NOW,
        "--non-interactive",
    )

    assert _cli(git_repo, memory_root, *arguments) == 2

    captured = capsys.readouterr()
    preview = _documents(captured.out)[-1]
    error = _documents(captured.err)[-1]
    assert preview["type"] == "feedback_record_preview"
    assert preview["review_id"] == review_id
    assert preview["finding_id"] == snapshot.finding_id
    assert preview["safe_text_validated"] is True
    assert "actor" not in preview
    assert error["error"]["code"] == "confirmation_required"
    assert not memory_root.exists()


def test_replay_outbox_has_final_service_seam_and_fails_explicitly_when_absent(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    memory_root = tmp_path / "outbox-memory"
    review_id = "review-outbox"
    _session, _snapshot, outbox_digest = _feedback_outbox_session(
        git_repo,
        tmp_path,
        review_id=review_id,
    )
    monkeypatch.delattr(
        pipeline_module,
        "replay_memory_outbox",
        raising=False,
    )

    exit_code = _cli(
        git_repo,
        memory_root,
        "replay-outbox",
        review_id,
        "--actor",
        "amy",
        "--reason",
        "Replay the exact committed Session outbox.",
        "--yes",
        "--non-interactive",
    )

    captured = capsys.readouterr()
    preview = _documents(captured.out)[-1]
    error = _documents(captured.err)[-1]
    assert exit_code == 1
    assert preview["type"] == "replay_outbox_preview"
    assert preview["review_id"] == review_id
    assert preview["expected_outbox_digest"] == outbox_digest
    assert error["error"]["code"] == "outbox_service_unavailable"
    assert not memory_root.exists()


def test_feedback_tampered_session_wins_before_secret_scan_or_preview_and_creates_no_store(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    memory_root = tmp_path / "tampered-feedback-memory"
    review_id = "review-feedback-tampered"
    session, snapshot, _outbox_digest = _feedback_outbox_session(
        git_repo,
        tmp_path,
        review_id=review_id,
    )
    (session.run_dir / "reconciliation.json").write_text(
        "{}",
        encoding="utf-8",
    )
    secret_actor = "sk-" + "a" * 32

    exit_code = _cli(
        git_repo,
        memory_root,
        "feedback",
        "record",
        review_id,
        snapshot.finding_id,
        "--head-sha",
        _head(git_repo),
        "--finding-hash",
        snapshot.finding_hash,
        "--evidence-ref",
        snapshot.evidence_refs[0],
        "--decision",
        FeedbackDecision.ACCEPTED.value,
        "--final-severity",
        FindingSeverity.HIGH.value,
        "--reason-code",
        FeedbackReasonCode.OTHER.value,
        "--actor",
        secret_actor,
        "--reason",
        "This free text must never be previewed before Session validation.",
        "--created-at",
        NOW,
        "--yes",
        "--non-interactive",
    )

    captured = capsys.readouterr()
    assert exit_code == 4
    assert captured.out == ""
    assert "feedback_session_untrusted" in captured.err
    assert secret_actor not in captured.err
    assert "free text" not in captured.err
    assert not memory_root.exists()


def test_feedback_secret_scan_precedes_safe_preview_and_store_creation(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    memory_root = tmp_path / "secret-feedback-memory"
    review_id = "review-feedback-secret"
    _session, snapshot, _outbox_digest = _feedback_outbox_session(
        git_repo,
        tmp_path,
        review_id=review_id,
    )
    secret_actor = "sk-" + "b" * 32

    exit_code = _cli(
        git_repo,
        memory_root,
        "feedback",
        "record",
        review_id,
        snapshot.finding_id,
        "--head-sha",
        _head(git_repo),
        "--finding-hash",
        snapshot.finding_hash,
        "--evidence-ref",
        snapshot.evidence_refs[0],
        "--decision",
        FeedbackDecision.ACCEPTED.value,
        "--final-severity",
        FindingSeverity.HIGH.value,
        "--reason-code",
        FeedbackReasonCode.OTHER.value,
        "--actor",
        secret_actor,
        "--reason",
        "Maintainer confirmed the canonical Finding.",
        "--created-at",
        NOW,
        "--yes",
        "--non-interactive",
    )

    captured = capsys.readouterr()
    assert exit_code == 4
    assert captured.out == ""
    assert "feedback_source_validation_failed" in captured.err
    assert secret_actor not in captured.err
    assert not memory_root.exists()


def test_feedback_validated_preview_is_persisted_only_after_confirmation(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    memory_root = tmp_path / "validated-feedback-memory"
    review_id = "review-feedback-write"
    _session, snapshot, _outbox_digest = _feedback_outbox_session(
        git_repo,
        tmp_path,
        review_id=review_id,
    )

    exit_code = _cli(
        git_repo,
        memory_root,
        "feedback",
        "record",
        review_id,
        snapshot.finding_id,
        "--head-sha",
        _head(git_repo),
        "--finding-hash",
        snapshot.finding_hash,
        "--evidence-ref",
        snapshot.evidence_refs[0],
        "--decision",
        FeedbackDecision.ACCEPTED.value,
        "--final-severity",
        FindingSeverity.HIGH.value,
        "--reason-code",
        FeedbackReasonCode.OTHER.value,
        "--actor",
        "amy",
        "--reason",
        "Maintainer confirmed the canonical Finding.",
        "--created-at",
        NOW,
        "--yes",
        "--non-interactive",
    )

    documents = _documents(capsys.readouterr().out)
    assert exit_code == 0
    assert [document["type"] for document in documents] == [
        "feedback_record_preview",
        "feedback_record",
    ]
    store = _store(git_repo, memory_root)
    records = store.list_feedback(
        _namespace(git_repo, memory_root).repository_key
    )
    assert len(records) == 1
    assert records[0].finding_id == snapshot.finding_id


def test_replay_outbox_passes_only_final_cas_inputs_and_accepts_strict_receipt(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    memory_root = tmp_path / "strict-replay-memory"
    review_id = "review-outbox-strict"
    _session, _snapshot, outbox_digest = _feedback_outbox_session(
        git_repo,
        tmp_path,
        review_id=review_id,
    )
    observed: list[dict[str, object]] = []

    def replay_memory_outbox(**kwargs):
        observed.append(dict(kwargs))
        return _valid_replay_receipt(dict(kwargs))

    monkeypatch.setattr(
        pipeline_module,
        "replay_memory_outbox",
        replay_memory_outbox,
        raising=False,
    )

    exit_code = _cli(
        git_repo,
        memory_root,
        "replay-outbox",
        review_id,
        "--actor",
        "amy",
        "--reason",
        "Replay the exact committed Session outbox.",
        "--yes",
        "--non-interactive",
    )

    documents = _documents(capsys.readouterr().out)
    assert exit_code == 0
    assert [document["type"] for document in documents] == [
        "replay_outbox_preview",
        "replay_outbox_result",
    ]
    assert observed
    call = observed[0]
    assert {
        key for key in call if key.startswith("expected_")
    } == {
        "expected_repository_key",
        "expected_authority_resolution_hash",
        "expected_outbox_digest",
    }
    assert call["expected_outbox_digest"] == outbox_digest
    assert documents[-1]["receipt"]["success"] is True
    assert not memory_root.exists()


@pytest.mark.parametrize(
    "invalid_kind",
    (
        "empty",
        "false",
        "repository",
        "authority",
        "binding",
        "outbox",
        "digest",
    ),
)
def test_replay_outbox_rejects_empty_false_or_misbound_receipts_as_operational(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    invalid_kind: str,
) -> None:
    memory_root = tmp_path / ("invalid-receipt-" + invalid_kind)
    review_id = "review-outbox-invalid-" + invalid_kind
    _session, _snapshot, _outbox_digest = _feedback_outbox_session(
        git_repo,
        tmp_path,
        review_id=review_id,
    )

    def replay_memory_outbox(**kwargs):
        if invalid_kind == "empty":
            return {}
        receipt = _valid_replay_receipt(dict(kwargs))
        if invalid_kind == "false":
            receipt["success"] = False
        elif invalid_kind == "repository":
            receipt["repository_key"] = "f" * 64
        elif invalid_kind == "authority":
            receipt["authority_resolution_hash"] = "e" * 64
        elif invalid_kind == "binding":
            receipt["binding_id"] = "RB-" + "d" * 64
        elif invalid_kind == "outbox":
            receipt["outbox_digest"] = "c" * 64
        elif invalid_kind == "digest":
            receipt["receipt_digest"] = "b" * 64
        if invalid_kind != "digest":
            receipt["receipt_digest"] = canonical_sha256(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "receipt_digest"
                }
            )
        return receipt

    monkeypatch.setattr(
        pipeline_module,
        "replay_memory_outbox",
        replay_memory_outbox,
        raising=False,
    )

    exit_code = _cli(
        git_repo,
        memory_root,
        "replay-outbox",
        review_id,
        "--actor",
        "amy",
        "--reason",
        "Reject a forged replay receipt.",
        "--yes",
        "--non-interactive",
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert [document["type"] for document in _documents(captured.out)] == [
        "replay_outbox_preview"
    ]
    assert "outbox_service_invalid" in captured.err
    assert not memory_root.exists()


def test_replay_outbox_reports_relink_cas_race_as_conflict(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    memory_root = tmp_path / "replay-cas-conflict"
    review_id = "review-outbox-cas-conflict"
    _feedback_outbox_session(git_repo, tmp_path, review_id=review_id)

    def replay_memory_outbox(**_kwargs):
        raise RepositoryRelinkConflictError("simulated relink race")

    monkeypatch.setattr(
        pipeline_module,
        "replay_memory_outbox",
        replay_memory_outbox,
        raising=False,
    )

    exit_code = _cli(
        git_repo,
        memory_root,
        "replay-outbox",
        review_id,
        "--actor",
        "amy",
        "--reason",
        "Reject a concurrent authority relink.",
        "--yes",
        "--non-interactive",
    )

    captured = capsys.readouterr()
    assert exit_code == 4
    assert "relink_conflict" in captured.err
    assert not memory_root.exists()


def test_memory_error_exit_classification_is_exhaustive() -> None:
    assert set(command_module._FEEDBACK_ERROR_EXIT_CODES) == set(FeedbackErrorCode)
    assert set(command_module._SOURCE_ERROR_EXIT_CODES) == set(SourceValidationCode)
    assert set(command_module._LIFECYCLE_ERROR_EXIT_CODES) == set(
        MemoryLifecycleErrorCode
    )
    assert (
        command_module._FEEDBACK_ERROR_EXIT_CODES[
            FeedbackErrorCode.SESSION_NOT_COMPLETED
        ]
        == 4
    )
    assert (
        command_module._FEEDBACK_ERROR_EXIT_CODES[
            FeedbackErrorCode.AGGREGATION_LIMIT_EXCEEDED
        ]
        == 1
    )
    assert (
        command_module._SOURCE_ERROR_EXIT_CODES[SourceValidationCode.INTERNAL_ERROR]
        == 1
    )
    assert (
        command_module._LIFECYCLE_ERROR_EXIT_CODES[
            MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED
        ]
        == 4
    )


@pytest.mark.parametrize(
    "arguments,id_field",
    [
        (("status",), None),
        (("list",), "memory_id"),
        (("show", "{memory_id}"), "memory_id"),
        (("candidates",), "candidate_id"),
        (("candidate", "show", "{candidate_id}"), "candidate_id"),
    ],
)
def test_reads_are_generation_neutral(
    git_repo: Path,
    tmp_path: Path,
    capsys,
    arguments: tuple[str, ...],
    id_field: str | None,
) -> None:
    memory_root = tmp_path / "m"
    store = _store(git_repo, memory_root)
    candidate = _candidate(
        git_repo,
        _namespace(git_repo, memory_root).repository_key,
        statement="Approved arithmetic behavior must remain stable.",
        review_id="review-memory-cli-read",
    )
    _submit(git_repo, store, candidate)
    record = _approve_direct(git_repo, store, candidate)
    before = store.get_generations(candidate.repository_key)
    resolved = tuple(
        item.format(memory_id=record.memory_id, candidate_id=candidate.candidate_id)
        for item in arguments
    )

    assert _cli(git_repo, memory_root, *resolved) == 0
    document = _documents(capsys.readouterr().out)[-1]

    assert document["repository_key"] == candidate.repository_key
    assert document["generation"] == before.memory_generation
    if id_field == "memory_id":
        assert record.memory_id in json.dumps(document)
    if id_field == "candidate_id":
        assert candidate.candidate_id in json.dumps(document)
    assert store.get_generations(candidate.repository_key) == before


def test_approve_preview_and_confirmation(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    memory_root = tmp_path / "m"
    store = _store(git_repo, memory_root)
    candidate = _candidate(
        git_repo,
        _namespace(git_repo, memory_root).repository_key,
        statement="High-risk arithmetic changes require focused verification.",
        review_id="review-memory-cli-approve",
        policy=True,
    )
    _submit(git_repo, store, candidate)
    generation = store.get_generations(candidate.repository_key).memory_generation

    assert (
        _cli(
            git_repo,
            memory_root,
            "approve",
            candidate.candidate_id,
            "--actor",
            "amy",
            "--reason",
            "Maintainer reviewed the statement and exact source.",
            "--non-interactive",
        )
        == 2
    )
    refused = capsys.readouterr()
    assert _documents(refused.out)[0]["type"] == "approve_preview"
    assert "confirmation_required" in refused.err
    assert store.get_candidate(candidate.candidate_id).status is CandidateStatus.PENDING_APPROVAL
    assert store.get_generations(candidate.repository_key).memory_generation == generation

    assert (
        _cli(
            git_repo,
            memory_root,
            "approve",
            candidate.candidate_id,
            "--actor",
            "amy",
            "--reason",
            "Maintainer reviewed the statement and exact source.",
            "--non-interactive",
            "--yes",
        )
        == 0
    )
    documents = _documents(capsys.readouterr().out)
    preview, result = documents
    assert preview["type"] == "approve_preview"
    assert preview["statement"] == candidate.statement
    assert preview["scope"]["paths"] == ["app.py"]
    assert preview["sources"][0]["type"] == "repository_range"
    assert preview["validity"]["valid_from_sha"] == candidate.valid_from_sha
    assert preview["policy_diff"]["after"]["type"] == "risk_floor"
    assert preview["source_validation"]["valid"] is True
    assert result["type"] == "approve_result"
    assert result["candidate_id"] == candidate.candidate_id
    assert result["memory_id"].startswith("MEM-")
    assert result["generation"] > generation


def test_approve_restores_runtime_origin_from_authority_receipt(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    memory_root = tmp_path / "m"
    store = _store(git_repo, memory_root)
    candidate = _candidate(
        git_repo,
        _namespace(git_repo, memory_root).repository_key,
        statement="Persisted producer metadata is not approval authority.",
        review_id="review-memory-cli-runtime-origin",
    )
    trusted_local = replace(
        _provenance(candidate),
        origin=ProducerType.LOCAL,
        allowed_source_refs=(),
    )
    MemoryLifecycle(store, SourceValidator(git_repo)).submit_candidate(
        candidate,
        runtime_provenance=trusted_local,
        request_id=stable_request_id("memory-cli-local-submit", candidate.candidate_id),
    )
    observed_origins: list[ProducerType] = []
    original_validate = SourceValidator.validate_candidate

    def observe_validate(self, value, *, runtime_provenance=None):
        if value.candidate_id == candidate.candidate_id:
            observed_origins.append(runtime_provenance.origin)
        return original_validate(
            self,
            value,
            runtime_provenance=runtime_provenance,
        )

    monkeypatch.setattr(SourceValidator, "validate_candidate", observe_validate)

    assert (
        _cli(
            git_repo,
            memory_root,
            "approve",
            candidate.candidate_id,
            "--actor",
            "amy",
            "--reason",
            "Maintainer confirmed the stored Runtime authority.",
            "--non-interactive",
            "--yes",
        )
        == 0
    )
    capsys.readouterr()

    assert observed_origins
    assert set(observed_origins) == {ProducerType.LOCAL}
    assert candidate.producer.producer_type is ProducerType.MODEL
    assert store.get_candidate(candidate.candidate_id).status is CandidateStatus.APPROVED


def test_lifecycle_write_commands(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    memory_root = tmp_path / "m"
    store = _store(git_repo, memory_root)
    repository_key = _namespace(git_repo, memory_root).repository_key

    rejected = _candidate(
        git_repo,
        repository_key,
        statement="Candidate to reject after explicit review.",
        review_id="review-memory-cli-reject",
    )
    _submit(git_repo, store, rejected)
    assert (
        _cli(
            git_repo,
            memory_root,
            "reject",
            rejected.candidate_id,
            "--actor",
            "amy",
            "--reason-code",
            "unsupported_claim",
            "--reason",
            "The evidence does not support durable authority.",
            "--non-interactive",
            "--yes",
        )
        == 0
    )
    capsys.readouterr()
    assert store.get_candidate(rejected.candidate_id).status is CandidateStatus.REJECTED

    revoked_candidate = _candidate(
        git_repo,
        repository_key,
        statement="Candidate whose active record will be revoked.",
        review_id="review-memory-cli-revoke",
    )
    _submit(git_repo, store, revoked_candidate)
    revoked_record = _approve_direct(git_repo, store, revoked_candidate)
    assert (
        _cli(
            git_repo,
            memory_root,
            "revoke",
            revoked_record.memory_id,
            "--actor",
            "amy",
            "--reason",
            "The rule is no longer project policy.",
            "--non-interactive",
            "--yes",
        )
        == 0
    )
    capsys.readouterr()
    assert store.get_record(revoked_record.memory_id).status is RecordStatus.REVOKED

    old_candidate = _candidate(
        git_repo,
        repository_key,
        statement="Original wording tied to reviewed evidence.",
        review_id="review-memory-cli-old",
    )
    _submit(git_repo, store, old_candidate)
    old_record = _approve_direct(git_repo, store, old_candidate)
    replacement = _candidate(
        git_repo,
        repository_key,
        statement="Replacement wording tied to refreshed reviewed evidence.",
        review_id="review-memory-cli-replacement",
    )
    _submit(git_repo, store, replacement)
    assert (
        _cli(
            git_repo,
            memory_root,
            "revalidate",
            old_record.memory_id,
            "--candidate",
            replacement.candidate_id,
            "--actor",
            "amy",
            "--reason",
            "Maintainer approved the immutable replacement.",
            "--non-interactive",
            "--yes",
        )
        == 0
    )
    documents = _documents(capsys.readouterr().out)
    assert documents[0]["predecessor_memory_id"] == old_record.memory_id
    assert documents[-1]["predecessor_memory_id"] == old_record.memory_id
    assert store.get_record(old_record.memory_id).status is RecordStatus.SUPERSEDED
    replacement_record = store.get_record(documents[-1]["memory_id"])
    assert replacement_record.candidate_id == replacement.candidate_id
    assert replacement_record.status is RecordStatus.ACTIVE


def test_redacted_export_and_import_dry_run(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    memory_root = tmp_path / "m"
    store = _store(git_repo, memory_root)
    candidate = _candidate(
        git_repo,
        _namespace(git_repo, memory_root).repository_key,
        statement="Sensitive project wording must not appear in an export.",
        review_id="review-memory-cli-export",
    )
    _submit(git_repo, store, candidate)
    before = store.get_generations(candidate.repository_key)
    export_path = tmp_path / "memory-export.json"

    assert _cli(git_repo, memory_root, "export", str(export_path)) == 0
    export_document = _documents(capsys.readouterr().out)[-1]
    raw_export = export_path.read_text(encoding="utf-8")
    assert export_document["redacted"] is True
    assert export_document["restorable"] is False
    assert candidate.statement not in raw_export
    assert json.loads(raw_export)["redacted"] is True
    assert store.get_generations(candidate.repository_key) == before

    assert _cli(git_repo, memory_root, "import", str(export_path)) == 0
    plan = _documents(capsys.readouterr().out)[-1]
    assert plan["type"] == "import_plan"
    assert plan["applied"] is False
    assert plan["identity_matches_current_repository"] is True
    assert store.get_generations(candidate.repository_key) == before

    assert (
        _cli(
            git_repo,
            memory_root,
            "import",
            str(export_path),
            "--apply",
            "--identity-match",
            "--non-interactive",
            "--yes",
        )
        == 2
    )
    error = capsys.readouterr()
    assert "import_not_restorable" in error.err
    assert store.get_generations(candidate.repository_key) == before


def test_export_rejects_authority_paths_and_confirms_overwrite(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    memory_root = tmp_path / "m"
    store = _store(git_repo, memory_root)
    protected_targets = (
        Path(store.database_path),
        git_repo / ".git" / "config",
    )
    before = {path: path.read_bytes() for path in protected_targets}

    for target in protected_targets:
        assert _cli(git_repo, memory_root, "export", str(target)) == 2
        error = json.loads(capsys.readouterr().err)
        assert error["error"]["code"] == "export_destination_protected"
        assert target.read_bytes() == before[target]
    store.validate_integrity()

    destination = tmp_path / "existing-export.json"
    destination.write_text("existing user content", encoding="utf-8")
    assert _cli(git_repo, memory_root, "export", str(destination)) == 4
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "export_exists"
    assert destination.read_text(encoding="utf-8") == "existing user content"

    assert (
        _cli(
            git_repo,
            memory_root,
            "export",
            str(destination),
            "--overwrite",
            "--non-interactive",
        )
        == 2
    )
    captured = capsys.readouterr()
    assert _documents(captured.out)[-1]["type"] == "export_overwrite_preview"
    assert "confirmation_required" in captured.err
    assert destination.read_text(encoding="utf-8") == "existing user content"

    assert (
        _cli(
            git_repo,
            memory_root,
            "export",
            str(destination),
            "--overwrite",
            "--non-interactive",
            "--yes",
        )
        == 0
    )
    documents = _documents(capsys.readouterr().out)
    assert documents[0]["type"] == "export_overwrite_preview"
    assert documents[-1]["type"] == "export_result"
    assert json.loads(destination.read_text(encoding="utf-8"))["redacted"] is True


def test_import_apply_guards_and_restore(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    source_root = tmp_path / "s"
    source = _store(git_repo, source_root)
    candidate = _candidate(
        git_repo,
        _namespace(git_repo, source_root).repository_key,
        statement="Portable candidate restored only after explicit identity match.",
        review_id="review-memory-cli-import",
    )
    _submit(git_repo, source, candidate)
    export_directory = tmp_path / "portable-export"
    source.export_to_directory(
        export_directory,
        redact=False,
        include_blobs=True,
        created_at=NOW,
    )
    target_root = tmp_path / "t"

    assert (
        _cli(
            git_repo,
            target_root,
            "import",
            str(export_directory),
            "--apply",
            "--non-interactive",
            "--yes",
        )
        == 2
    )
    assert "identity_confirmation_required" in capsys.readouterr().err

    assert (
        _cli(
            git_repo,
            target_root,
            "import",
            str(export_directory),
            "--apply",
            "--identity-match",
            "--non-interactive",
        )
        == 2
    )
    assert "confirmation_required" in capsys.readouterr().err

    assert (
        _cli(
            git_repo,
            target_root,
            "import",
            str(export_directory),
            "--apply",
            "--identity-match",
            "--non-interactive",
            "--yes",
        )
        == 0
    )
    result = _documents(capsys.readouterr().out)[-1]
    assert result["type"] == "import_result"
    assert result["applied"] is True
    restored = _store(git_repo, target_root)
    assert restored.get_candidate(candidate.candidate_id).statement == candidate.statement


def test_import_apply_uses_the_exact_prepared_manifest_after_path_replacement(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    source_root = tmp_path / "source-memory"
    source = _store(git_repo, source_root)
    repository_key = _namespace(git_repo, source_root).repository_key
    prepared_candidate = _candidate(
        git_repo,
        repository_key,
        statement="The prepared manifest remains authoritative.",
        review_id="review-memory-cli-prepared-import",
    )
    _submit(git_repo, source, prepared_candidate)
    prepared_export = tmp_path / "prepared-export"
    source.export_to_directory(
        prepared_export,
        redact=False,
        include_blobs=True,
        created_at=NOW,
    )

    replacement_root = tmp_path / "replacement-memory"
    replacement_source = _store(git_repo, replacement_root)
    replacement_candidate = _candidate(
        git_repo,
        repository_key,
        statement="A later path replacement must not be imported.",
        review_id="review-memory-cli-replaced-import",
    )
    _submit(git_repo, replacement_source, replacement_candidate)
    replacement_export = tmp_path / "replacement-export"
    replacement_source.export_to_directory(
        replacement_export,
        redact=False,
        include_blobs=True,
        created_at=NOW,
    )

    manifest_path = prepared_export / "manifest.json"
    replacement_manifest = replacement_export / "manifest.json"
    original_prepare = MemoryStore.prepare_import_manifest
    replaced = False

    def prepare_then_replace(self, manifest_or_path):
        nonlocal replaced
        prepared = original_prepare(self, manifest_or_path)
        if Path(manifest_or_path).resolve() == manifest_path.resolve():
            manifest_path.write_bytes(replacement_manifest.read_bytes())
            replaced = True
        return prepared

    monkeypatch.setattr(
        MemoryStore,
        "prepare_import_manifest",
        prepare_then_replace,
    )
    target_root = tmp_path / "target-memory"

    assert (
        _cli(
            git_repo,
            target_root,
            "import",
            str(prepared_export),
            "--apply",
            "--identity-match",
            "--non-interactive",
            "--yes",
        )
        == 0
    )
    capsys.readouterr()
    restored = _store(git_repo, target_root)

    assert replaced
    assert restored.get_candidate(prepared_candidate.candidate_id) == replace(
        prepared_candidate,
        status=CandidateStatus.PENDING_APPROVAL,
    )
    assert restored.find_candidate(replacement_candidate.candidate_id) is None


def test_import_direct_store_creation_is_linearized_against_relink(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    export_root = tmp_path / "export-memory"
    export_store = _store(git_repo, export_root)
    candidate = _candidate(
        git_repo,
        _namespace(git_repo, export_root).repository_key,
        statement="Import must not create a direct Store after relink commits.",
        review_id="review-memory-cli-import-relink-race",
    )
    _submit(git_repo, export_store, candidate)
    export_directory = tmp_path / "restorable-export"
    export_store.export_to_directory(
        export_directory,
        redact=False,
        include_blobs=True,
        created_at=NOW,
    )

    old_repository = tmp_path / "old-authority-repository"
    run_git(tmp_path, "clone", str(git_repo), str(old_repository))
    target_root = tmp_path / "target-memory"
    old_namespace = _namespace(old_repository, target_root)
    old_store = MemoryStore(old_namespace)
    locator_namespace = _namespace(git_repo, target_root)

    def commit_relink_during_confirmation(_args, action, _subject_id) -> None:
        assert action == "import"
        old_snapshot = old_store.repository_authority_snapshot(
            old_namespace.repository_key
        )
        registry = RepositoryRelinkRegistry(target_root)
        prepared = registry.prepare_relink(
            old_snapshot.repository_identity,
            locator_namespace.metadata,
            from_repository_key=old_namespace.repository_key,
            actor="race-test",
            reason="Commit the explicit binding before import materializes.",
            request_id=stable_request_id(
                "memory-cli-import-relink-race",
                old_namespace.repository_key,
                locator_namespace.repository_key,
            ),
        )
        registry.apply_relink(prepared)

    monkeypatch.setattr(
        command_module,
        "_confirm_memory_write",
        commit_relink_during_confirmation,
    )

    assert (
        _cli(
            git_repo,
            target_root,
            "import",
            str(export_directory),
            "--apply",
            "--identity-match",
            "--non-interactive",
            "--yes",
        )
        == 4
    )
    captured = capsys.readouterr()
    assert "repository_changed" in captured.err
    assert not (
        Path(locator_namespace.namespace_path) / "memory.sqlite3"
    ).exists()
    assert old_store.find_candidate(candidate.candidate_id) is None


def test_clone_namespace_isolation(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    origin = "https://example.test/acme/review-agent.git"
    run_git(git_repo, "remote", "add", "origin", origin)
    clone = tmp_path / "clone"
    run_git(tmp_path, "clone", str(git_repo), str(clone))
    run_git(clone, "remote", "set-url", "origin", origin)
    memory_root = tmp_path / "m"
    source = _store(git_repo, memory_root)
    candidate = _candidate(
        git_repo,
        _namespace(git_repo, memory_root).repository_key,
        statement="Clone-local identity must not inherit by origin.",
        review_id="review-memory-cli-clone",
    )
    _submit(git_repo, source, candidate)
    export_directory = tmp_path / "identity-export"
    source.export_to_directory(
        export_directory,
        redact=False,
        include_blobs=True,
        created_at=NOW,
    )
    clone_namespace = _namespace(clone, memory_root)
    assert clone_namespace.repository_key != candidate.repository_key

    assert (
        _cli(
            clone,
            memory_root,
            "import",
            str(export_directory),
            "--apply",
            "--identity-match",
            "--non-interactive",
            "--yes",
        )
        == 4
    )
    assert "identity_mismatch" in capsys.readouterr().err
    assert not (Path(clone_namespace.namespace_path) / "memory.sqlite3").exists()
    assert source.get_candidate(candidate.candidate_id) == source.get_candidate(
        candidate.candidate_id
    )


def test_explicit_relink_binds_clone_without_copying_store(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    origin = "https://example.test/acme/review-agent.git"
    run_git(git_repo, "remote", "add", "origin", origin)
    clone = tmp_path / "replacement-clone"
    run_git(tmp_path, "clone", str(git_repo), str(clone))
    run_git(clone, "remote", "set-url", "origin", origin)
    memory_root = tmp_path / "m"
    source_namespace = _namespace(git_repo, memory_root)
    clone_namespace = _namespace(clone, memory_root)
    source = MemoryStore(source_namespace)
    candidate = _candidate(
        git_repo,
        source_namespace.repository_key,
        statement="A relink must preserve the explicitly selected authority.",
        review_id="review-memory-cli-relink",
    )
    _submit(git_repo, source, candidate)
    registry_path = memory_root / "repository-relinks.sqlite3"

    assert (
        _cli(
            clone,
            memory_root,
            "relink",
            "--from-key",
            source_namespace.repository_key,
            "--actor",
            "amy",
            "--reason",
            "Restore this exact local Memory authority.",
            "--non-interactive",
        )
        == 2
    )
    preview_capture = capsys.readouterr()
    assert _documents(preview_capture.out)[-1]["type"] == "relink_preview"
    assert "confirmation_required" in preview_capture.err
    assert not registry_path.exists()
    assert MemoryStore.namespace_has_no_store_state(clone_namespace)

    assert (
        _cli(
            clone,
            memory_root,
            "relink",
            "--from-key",
            "0" * 64,
            "--actor",
            "amy",
            "--reason",
            "Do not infer an authority from the shared origin.",
            "--non-interactive",
            "--yes",
        )
        == 3
    )
    wrong_key_capture = capsys.readouterr()
    assert "authority_not_found" in wrong_key_capture.err
    assert not registry_path.exists()

    assert (
        _cli(
            clone,
            memory_root,
            "relink",
            "--from-key",
            source_namespace.repository_key,
            "--actor",
            "amy",
            "--reason",
            "Restore this exact local Memory authority.",
            "--non-interactive",
            "--yes",
        )
        == 0
    )
    result_documents = _documents(capsys.readouterr().out)
    assert [document["type"] for document in result_documents] == [
        "relink_preview",
        "relink_result",
    ]
    result = result_documents[-1]
    assert result["repository_key"] == source_namespace.repository_key
    assert result["locator_repository_key"] == clone_namespace.repository_key
    assert result["binding_id"].startswith("RB-")
    assert result["event_id"].startswith("EVT-")
    assert result["outcome"] == "applied"
    assert registry_path.is_file()

    # The deterministic request ID replays the exact committed receipt; it
    # does not append another binding event or advance registry generation.
    assert (
        _cli(
            clone,
            memory_root,
            "relink",
            "--from-key",
            source_namespace.repository_key,
            "--actor",
            "amy",
            "--reason",
            "Restore this exact local Memory authority.",
            "--non-interactive",
            "--yes",
        )
        == 0
    )
    replay = _documents(capsys.readouterr().out)[-1]
    assert replay["event_id"] == result["event_id"]
    assert replay["result_hash"] == result["result_hash"]
    assert replay["registry_generation"] == result["registry_generation"]
    assert len(RepositoryRelinkRegistry(memory_root).verify_event_chain()) == 1

    moved_authority_path = tmp_path / "moved-authority"
    moved_authority_path.mkdir()
    moved_authority = replace(
        source_namespace.metadata,
        canonical_path=str(moved_authority_path.resolve()),
    )
    source.register_repository(moved_authority)
    assert (
        _cli(
            clone,
            memory_root,
            "gc",
            "--apply",
            "--non-interactive",
            "--yes",
        )
        == 0
    )
    capsys.readouterr()
    assert source.get_repository_descriptor(source_namespace.repository_key) == (
        moved_authority
    )

    assert _cli(clone, memory_root, "candidates") == 0
    candidates = _documents(capsys.readouterr().out)[-1]
    assert candidates["repository_key"] == source_namespace.repository_key
    assert candidates["locator_repository_key"] == clone_namespace.repository_key
    assert candidates["binding_id"] == result["binding_id"]
    assert [item["candidate_id"] for item in candidates["candidates"]] == [
        candidate.candidate_id
    ]
    assert not (
        repository_namespace_path(memory_root, clone_namespace.repository_key)
        / "memory.sqlite3"
    ).exists()

    # Relinking grants locator access to the old authority, but it does not
    # fabricate a new runtime-origin receipt for a pending candidate.
    assert (
        _cli(
            clone,
            memory_root,
            "approve",
            candidate.candidate_id,
            "--actor",
            "amy",
            "--reason",
            "Review through the replacement clone.",
            "--non-interactive",
            "--yes",
        )
        == 3
    )
    assert "store_not_found" in capsys.readouterr().err
    assert source.get_candidate(candidate.candidate_id).status is CandidateStatus.PENDING_APPROVAL


def test_relink_rejects_nonempty_locator_namespace(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    clone = tmp_path / "clone-with-memory"
    run_git(tmp_path, "clone", str(git_repo), str(clone))
    memory_root = tmp_path / "m"
    source_namespace = _namespace(git_repo, memory_root)
    clone_namespace = _namespace(clone, memory_root)
    MemoryStore(source_namespace)
    MemoryStore(clone_namespace)

    assert (
        _cli(
            clone,
            memory_root,
            "relink",
            "--from-key",
            source_namespace.repository_key,
            "--actor",
            "amy",
            "--reason",
            "This must not overwrite an existing locator namespace.",
            "--non-interactive",
            "--yes",
        )
        == 4
    )
    captured = capsys.readouterr()
    assert "relink_conflict" in captured.err
    assert not (memory_root / "repository-relinks.sqlite3").exists()
    assert (Path(clone_namespace.namespace_path) / "memory.sqlite3").is_file()


def test_relink_rechecks_authority_state_inside_apply_locks(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    clone = tmp_path / "replacement-clone"
    run_git(tmp_path, "clone", str(git_repo), str(clone))
    memory_root = tmp_path / "m"
    source_namespace = _namespace(git_repo, memory_root)
    clone_namespace = _namespace(clone, memory_root)
    source = MemoryStore(source_namespace)
    first = _candidate(
        git_repo,
        source_namespace.repository_key,
        statement="The prepare token must be compared again before binding.",
        review_id="review-memory-cli-relink-stale-1",
    )
    second = _candidate(
        git_repo,
        source_namespace.repository_key,
        statement="A concurrent authority write invalidates the prepared relink.",
        review_id="review-memory-cli-relink-stale-2",
    )
    _submit(git_repo, source, first)
    original_apply = RepositoryRelinkRegistry.apply_relink
    observed_apply_kwargs: list[dict[str, object]] = []

    def mutate_before_apply(self, prepared, **kwargs):
        observed_apply_kwargs.append(dict(kwargs))
        _submit(git_repo, source, second)
        return original_apply(self, prepared, **kwargs)

    monkeypatch.setattr(
        RepositoryRelinkRegistry,
        "apply_relink",
        mutate_before_apply,
    )

    assert (
        _cli(
            clone,
            memory_root,
            "relink",
            "--from-key",
            source_namespace.repository_key,
            "--actor",
            "amy",
            "--reason",
            "Bind only if the reviewed authority snapshot remains exact.",
            "--non-interactive",
            "--yes",
        )
        == 4
    )
    captured = capsys.readouterr()
    assert "relink_conflict" in captured.err
    assert observed_apply_kwargs
    assert "old_authority_state_verifier" not in observed_apply_kwargs[0]
    assert "new_namespace_empty_verifier" not in observed_apply_kwargs[0]
    assert not (memory_root / "repository-relinks.sqlite3").exists()
    assert MemoryStore.namespace_has_no_store_state(clone_namespace)


def test_gc_dry_run_and_pin_safety(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    memory_root = tmp_path / "m"
    store = _store(git_repo, memory_root)
    unpinned = store.put_blob(b"orphan", media_type="application/octet-stream")
    pinned = store.put_blob(b"pinned", media_type="application/octet-stream")
    store.pin_blob(pinned.blob_hash, pin_type="manual", pin_id="PIN-test")
    before = store.get_generations(_namespace(git_repo, memory_root).repository_key)

    assert _cli(git_repo, memory_root, "gc") == 0
    plan = _documents(capsys.readouterr().out)[-1]
    assert plan["dry_run"] is True
    assert unpinned.blob_hash in plan["candidate_blob_ids"]
    assert pinned.blob_hash not in plan["candidate_blob_ids"]
    assert Path(unpinned.path).is_file()
    assert Path(pinned.path).is_file()

    assert (
        _cli(
            git_repo,
            memory_root,
            "gc",
            "--apply",
            "--non-interactive",
            "--yes",
        )
        == 0
    )
    result = _documents(capsys.readouterr().out)[-1]
    assert result["dry_run"] is False
    assert unpinned.blob_hash in result["deleted_blob_ids"]
    assert pinned.blob_hash not in result["deleted_blob_ids"]
    assert not Path(unpinned.path).exists()
    assert Path(pinned.path).is_file()
    store.validate_blob(pinned.blob_hash)
    assert store.get_generations(_namespace(git_repo, memory_root).repository_key) == before


def test_gc_apply_consumes_the_exact_signed_preview(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    memory_root = tmp_path / "m"
    store = _store(git_repo, memory_root)
    blob = store.put_blob(b"collect exactly once", media_type="application/octet-stream")
    scans = []
    applied_previews = []
    original_scan = MemoryStore.gc_blobs
    original_apply = MemoryStore.apply_blob_gc

    def track_scan(self, *args, **kwargs):
        result = original_scan(self, *args, **kwargs)
        if kwargs.get("dry_run", True):
            scans.append(result)
        return result

    def track_apply(self, preview):
        applied_previews.append(preview)
        return original_apply(self, preview)

    monkeypatch.setattr(MemoryStore, "gc_blobs", track_scan)
    monkeypatch.setattr(MemoryStore, "apply_blob_gc", track_apply)

    assert (
        _cli(
            git_repo,
            memory_root,
            "gc",
            "--apply",
            "--non-interactive",
            "--yes",
        )
        == 0
    )
    capsys.readouterr()

    assert len(scans) == 1
    assert applied_previews == [scans[0]]
    assert scans[0].preview_token is not None
    assert not Path(blob.path).exists()


def test_errors_are_sanitized(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    memory_root = tmp_path / "m"
    _store(git_repo, memory_root)

    def fail_safely(self, candidate_id):
        raise MemoryStoreValidationError(
            "SELECT secret FROM credentials at https://user:api-key@example.test"
        )

    monkeypatch.setattr(MemoryStore, "get_candidate", fail_safely)
    exit_code = _cli(
        git_repo,
        memory_root,
        "candidate",
        "show",
        "MC-" + "0" * 64,
    )
    captured = capsys.readouterr()
    lowered = (captured.out + captured.err).casefold()
    assert exit_code == 2
    assert "store_validation" in captured.err
    assert "select" not in lowered
    assert "api-key" not in lowered
    assert "credentials" not in lowered
    assert "https://" not in lowered


@pytest.mark.parametrize(
    "unsafe_root",
    ("repository", "git", "review-agent"),
)
def test_memory_root_cannot_overlap_repository_authority_paths(
    git_repo: Path,
    capsys,
    unsafe_root: str,
) -> None:
    if unsafe_root == "repository":
        root = git_repo
    elif unsafe_root == "git":
        root = git_repo / ".git"
    else:
        root = git_repo / ".review-agent"

    exit_code = main(
        [
            "memory",
            "status",
            "--repo",
            str(git_repo),
            "--memory-root",
            str(root),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "identity_invalid" in captured.err
    assert not (root / "repositories").exists()

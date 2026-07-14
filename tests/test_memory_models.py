from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib

import pytest

from review_agent.memory_models import (
    Applicability,
    CandidateAuthorityReceipt,
    CandidateStatus,
    DurableMemoryRecord,
    FeedbackCalibrationSignal,
    FeedbackCalibrationSignalKind,
    FeedbackCalibrationSummary,
    FeedbackDecision,
    FeedbackReasonCode,
    FeedbackRecord,
    FeedbackStatus,
    FindingSeverity,
    FindingSnapshot,
    GenerationMetadata,
    GitCommitSourceRef,
    HumanDeclarationAuthority,
    HumanDeclarationOrigin,
    HumanDeclarationSourceRef,
    MemoryCandidate,
    MemoryConfidence,
    MemoryExecutionConfig,
    MemoryKind,
    MemoryMode,
    MemoryScope,
    MemorySelectionDecision,
    MemorySelectionInput,
    MemorySnapshot,
    ObservationSourceRef,
    PolicyEffect,
    PolicyEffectKind,
    Producer,
    ProducerType,
    RecordStatus,
    RepositoryKnowledgeCapability,
    RepositoryKnowledgeEntry,
    RepositoryKnowledgeKey,
    RepositoryRangeSourceRef,
    RepositorySymbolSourceRef,
    Sensitivity,
    SessionArtifactSourceRef,
    SourceBundleDescriptor,
    SourceRef,
    SymbolHashKind,
    ValidityPolicy,
    canonical_json,
    stable_event_id,
    stable_repository_binding_id,
    stable_request_id,
    validate_stable_id,
)


SHA_A = "a" * 40
SHA_B = "b" * 40
HASH_1 = "1" * 64
HASH_2 = "2" * 64
HASH_3 = "3" * 64
CREATED_AT = "2026-07-14T12:00:00Z"
REPOSITORY_KEY = "4" * 64
LOCATOR_REPOSITORY_KEY = "5" * 64
DECLARATION = "Approve this candidate for the current review."


def _human_source(actor: str = "amy") -> HumanDeclarationSourceRef:
    return HumanDeclarationSourceRef(
        request_id=stable_request_id("memory-approval", actor),
        actor=actor,
        declaration_hash=HASH_3,
        created_at=CREATED_AT,
        review_id="review-001",
    )


def _authority_declaration(
    declaration: str = DECLARATION,
    *,
    actor: str = "amy",
    review_id: str | None = "review-001",
    origin: HumanDeclarationOrigin = HumanDeclarationOrigin.USER_REQUEST,
) -> HumanDeclarationAuthority:
    source_ref = HumanDeclarationSourceRef(
        request_id=stable_request_id(
            "candidate-authority", actor, review_id, declaration
        ),
        actor=actor,
        declaration_hash=hashlib.sha256(declaration.encode("utf-8")).hexdigest(),
        created_at=CREATED_AT,
        review_id=review_id,
    )
    return HumanDeclarationAuthority(
        source_ref=source_ref,
        origin=origin,
        declaration=declaration,
    )


def _range_source(
    *, path: str = "payments/money.py", content_hash: str = HASH_1
) -> RepositoryRangeSourceRef:
    return RepositoryRangeSourceRef(
        revision=SHA_A,
        path=path,
        line_start=10,
        line_end=18,
        content_hash=content_hash,
    )


def _producer(*, schema_version: int = 1, version: str = "1.0.0") -> Producer:
    return Producer(
        producer_type=ProducerType.MODEL,
        name="memory-curator",
        version=version,
        schema_version=schema_version,
    )


def _scope() -> MemoryScope:
    return MemoryScope(
        paths=("payments/**",),
        symbols=("payments.money.calculate_total",),
        contracts=("numeric_correctness",),
        languages=("python",),
    )


def _candidate(**overrides: object) -> MemoryCandidate:
    values = {
        "repository_key": REPOSITORY_KEY,
        "kind": MemoryKind.BUSINESS_INVARIANT,
        "statement": "Amounts must use Decimal.",
        "scope": _scope(),
        "source_refs": (_range_source(), _human_source()),
        "valid_from_sha": SHA_A,
        "validity_policies": (ValidityPolicy.SOURCE_CONTENT_HASH,),
        "confidence": MemoryConfidence.HIGH,
        "sensitivity": Sensitivity.NORMAL,
        "policy_effect": None,
        "producer": _producer(),
        "origin_review_id": "review-001",
        "status": CandidateStatus.PROPOSED,
        "created_at": CREATED_AT,
    }
    values.update(overrides)
    return MemoryCandidate(**values)


def _authority_receipt(**overrides: object) -> CandidateAuthorityReceipt:
    declaration = _authority_declaration()
    values = {
        "candidate_id": _candidate().candidate_id,
        "authority_repository_key": REPOSITORY_KEY,
        "locator_repository_key": REPOSITORY_KEY,
        "origin": ProducerType.HUMAN,
        "review_id": "review-001",
        "proposal_head_sha": SHA_A,
        "authorized_source_refs": (_range_source(), declaration.source_ref),
        "human_declarations": (declaration,),
        "initial_validation_report_hash": HASH_1,
        "authority_resolution_hash": HASH_2,
        "binding_id": None,
        "created_at": CREATED_AT,
    }
    values.update(overrides)
    return CandidateAuthorityReceipt(**values)


def _bundle(candidate: MemoryCandidate) -> SourceBundleDescriptor:
    return SourceBundleDescriptor(
        repository_key=candidate.repository_key,
        candidate_id=candidate.candidate_id,
        source_refs=candidate.source_refs,
        blob_hash=HASH_2,
        size_bytes=512,
        media_type="application/vnd.review-agent.source-bundle+json",
        created_at=CREATED_AT,
    )


def _record(candidate: MemoryCandidate) -> DurableMemoryRecord:
    bundle = _bundle(candidate)
    return DurableMemoryRecord(
        candidate_id=candidate.candidate_id,
        repository_key=candidate.repository_key,
        kind=candidate.kind,
        statement=candidate.statement,
        scope=candidate.scope,
        source_refs=candidate.source_refs,
        source_bundle_hash=bundle.bundle_hash,
        valid_from_sha=candidate.valid_from_sha,
        validity_policies=candidate.validity_policies,
        confidence=candidate.confidence,
        sensitivity=candidate.sensitivity,
        policy_effect=candidate.policy_effect,
        approved_by="amy",
        approval_event_id=stable_event_id("approve", candidate.candidate_id),
        status=RecordStatus.ACTIVE,
        created_at=CREATED_AT,
    )


def _finding() -> FindingSnapshot:
    return FindingSnapshot(
        finding_id="F-" + "5" * 32,
        claim="Rounding can lose cents.",
        path="payments/money.py",
        line=42,
        contracts=("numeric_correctness",),
        original_severity=FindingSeverity.HIGH,
        evidence_refs=("O-" + "6" * 32,),
    )


def _feedback(index: int = 0, review_id: str = "review-001") -> FeedbackRecord:
    finding = _finding()
    return FeedbackRecord(
        repository_key=REPOSITORY_KEY,
        review_id=review_id,
        finding_id=finding.finding_id,
        head_sha=SHA_A,
        finding_snapshot=finding,
        decision=FeedbackDecision.ACCEPTED,
        original_severity=FindingSeverity.HIGH,
        final_severity=FindingSeverity.HIGH,
        reason_code=FeedbackReasonCode.OTHER,
        reason="Confirmed by maintainer %d." % index,
        actor="amy",
        source_refs=(_human_source("amy-%d" % index),),
        status=FeedbackStatus.RECORDED,
        created_at=CREATED_AT,
    )


def test_source_ref_round_trips_all_six_allowlisted_variants() -> None:
    refs = (
        _range_source(path="payments\\money.py"),
        RepositorySymbolSourceRef(
            revision=SHA_A.upper(),
            path="payments/money.py",
            qualified_name="payments.money.calculate_total",
            hash_kind=SymbolHashKind.SIGNATURE,
            content_hash=HASH_2,
        ),
        GitCommitSourceRef(commit_sha=SHA_A, metadata_hash=HASH_1),
        ObservationSourceRef(
            review_id="review-001",
            observation_id="O-" + "7" * 32,
            revision_binding="head@" + SHA_A,
            content_hash=HASH_1,
        ),
        SessionArtifactSourceRef(
            review_id="review-001",
            artifact_name="final_risk.json",
            artifact_schema="final_risk_v1",
            revision_binding=SHA_B + ".." + SHA_A,
            artifact_hash=HASH_2,
        ),
        _human_source(),
    )

    assert refs[0].path == "payments/money.py"
    assert refs[1].revision == SHA_A
    for source_ref in refs:
        assert SourceRef.from_dict(source_ref.to_dict()) == source_ref
        assert canonical_json(source_ref.to_dict()) == source_ref.to_json()


@pytest.mark.parametrize(
    "payload,match",
    [
        (
            {
                "schema_version": 1,
                "type": "web_page",
                "url": "https://example.test",
            },
            "unsupported",
        ),
        (
            {
                **_range_source().to_dict(),
                "surprise": True,
            },
            "unsupported field",
        ),
        (
            {
                **_range_source().to_dict(),
                "revision": "abc123",
            },
            "full Git object ID",
        ),
        (
            {
                **_range_source().to_dict(),
                "path": "../secret.txt",
            },
            "repository-relative",
        ),
        (
            {
                **_range_source().to_dict(),
                "line_start": 20,
                "line_end": 10,
            },
            "line_end",
        ),
        (
            {
                **_range_source().to_dict(),
                "content_hash": "f" * 32,
            },
            "SHA-256",
        ),
    ],
)
def test_source_ref_hydration_fails_closed(payload: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SourceRef.from_dict(payload)


def test_source_ref_rejects_empty_identifiers_and_incomplete_symbol_hash() -> None:
    with pytest.raises(ValueError, match="observation_id"):
        ObservationSourceRef(
            review_id="review-001",
            observation_id="",
            revision_binding="head@" + SHA_A,
            content_hash=HASH_1,
        )

    class UnapprovedSourceRef(SourceRef):
        def to_dict(self) -> dict:
            return _range_source().to_dict()

    with pytest.raises(ValueError, match="allowlisted SourceRef variant"):
        _candidate(source_refs=(UnapprovedSourceRef(),))
    with pytest.raises(ValueError, match="request_id"):
        HumanDeclarationSourceRef(
            request_id="",
            actor="amy",
            declaration_hash=HASH_1,
            created_at=CREATED_AT,
        )


def test_human_declaration_authority_is_strict_immutable_and_redacts_repr() -> None:
    authority = _authority_declaration()
    payload = authority.to_dict()

    assert set(payload) == {
        "schema_version",
        "source_ref",
        "origin",
        "declaration",
    }
    assert payload["declaration"] == DECLARATION
    assert HumanDeclarationAuthority.from_dict(payload) == authority
    assert authority.to_json() == canonical_json(payload)
    assert DECLARATION not in repr(authority)
    with pytest.raises(FrozenInstanceError):
        authority.origin = HumanDeclarationOrigin.CLI_REQUEST

    with pytest.raises(ValueError, match="declaration_hash"):
        HumanDeclarationAuthority(
            source_ref=HumanDeclarationSourceRef(
                request_id=stable_request_id("bad-declaration-hash"),
                actor="amy",
                declaration_hash=HASH_1,
                created_at=CREATED_AT,
                review_id="review-001",
            ),
            origin=HumanDeclarationOrigin.USER_REQUEST,
            declaration=DECLARATION,
        )
    with pytest.raises(ValueError, match="review_id"):
        _authority_declaration(
            review_id=None,
            origin=HumanDeclarationOrigin.USER_REQUEST,
        )

    cli_authority = _authority_declaration(
        review_id=None,
        origin=HumanDeclarationOrigin.CLI_REQUEST,
    )
    assert HumanDeclarationAuthority.from_dict(cli_authority.to_dict()) == cli_authority


@pytest.mark.parametrize("declaration", ["", "   ", "contains\x00nul", "x" * 8193])
def test_human_declaration_authority_rejects_invalid_or_unbounded_text(
    declaration: str,
) -> None:
    with pytest.raises(ValueError, match="declaration"):
        _authority_declaration(declaration)


def test_authority_models_reject_unknown_fields_and_versions() -> None:
    declaration = _authority_declaration()
    declaration_payload = declaration.to_dict()
    receipt = _authority_receipt()
    receipt_payload = receipt.to_dict()

    with pytest.raises(ValueError, match="unsupported field"):
        HumanDeclarationAuthority.from_dict({**declaration_payload, "actor": "amy"})
    with pytest.raises(ValueError, match="schema_version"):
        HumanDeclarationAuthority.from_dict(
            {**declaration_payload, "schema_version": 2}
        )
    with pytest.raises(ValueError, match="source_ref"):
        HumanDeclarationAuthority.from_dict(
            {**declaration_payload, "source_ref": _range_source().to_dict()}
        )

    with pytest.raises(ValueError, match="unsupported field"):
        CandidateAuthorityReceipt.from_dict({**receipt_payload, "trusted": True})
    with pytest.raises(ValueError, match="schema_version"):
        CandidateAuthorityReceipt.from_dict({**receipt_payload, "schema_version": 2})
    with pytest.raises(ValueError, match="receipt_id"):
        CandidateAuthorityReceipt.from_dict(
            {**receipt_payload, "receipt_id": "CAR-" + "0" * 64}
        )
    with pytest.raises(ValueError, match="must be a list"):
        CandidateAuthorityReceipt.from_dict(
            {**receipt_payload, "authorized_source_refs": tuple(receipt_payload["authorized_source_refs"])}
        )


def test_candidate_authority_receipt_round_trips_with_exact_fields() -> None:
    receipt = _authority_receipt()
    payload = receipt.to_dict()

    assert set(payload) == {
        "schema_version",
        "receipt_id",
        "candidate_id",
        "authority_repository_key",
        "locator_repository_key",
        "origin",
        "review_id",
        "proposal_head_sha",
        "authorized_source_refs",
        "human_declarations",
        "initial_validation_report_hash",
        "authority_resolution_hash",
        "binding_id",
        "created_at",
    }
    assert receipt.receipt_id.startswith("CAR-")
    assert len(receipt.receipt_id) == 68
    assert CandidateAuthorityReceipt.from_dict(payload) == receipt
    assert receipt.to_json() == canonical_json(payload)
    assert DECLARATION not in repr(receipt)
    validate_stable_id(receipt.receipt_id, "CAR", "receipt_id")
    with pytest.raises(FrozenInstanceError):
        receipt.binding_id = "RB-" + "0" * 64


def test_candidate_authority_receipt_canonicalizes_collection_order_and_duplicates() -> None:
    first = _authority_declaration("Approve the first authority statement.")
    second = _authority_declaration(
        "Approve the second authority statement.",
        actor="bob",
    )
    range_ref = _range_source()
    left = _authority_receipt(
        authorized_source_refs=(
            second.source_ref,
            range_ref,
            first.source_ref,
            range_ref,
        ),
        human_declarations=(second, first, first),
    )
    right = _authority_receipt(
        authorized_source_refs=(range_ref, first.source_ref, second.source_ref),
        human_declarations=(first, second),
    )

    assert left.authorized_source_refs == tuple(
        sorted(left.authorized_source_refs, key=lambda item: item.to_json())
    )
    assert left.human_declarations == tuple(
        sorted(left.human_declarations, key=lambda item: item.to_json())
    )
    assert left.authorized_source_refs == right.authorized_source_refs
    assert left.human_declarations == right.human_declarations
    assert left.receipt_id == right.receipt_id
    assert left.to_json() == right.to_json()


def test_candidate_authority_receipt_enforces_direct_and_bound_repository_modes() -> None:
    with pytest.raises(ValueError, match="direct.*binding_id"):
        _authority_receipt(binding_id=stable_repository_binding_id("unexpected"))
    with pytest.raises(ValueError, match="bound.*binding_id"):
        _authority_receipt(locator_repository_key=LOCATOR_REPOSITORY_KEY)
    with pytest.raises(ValueError, match="binding_id"):
        _authority_receipt(
            locator_repository_key=LOCATOR_REPOSITORY_KEY,
            binding_id="RB-short",
        )

    binding_id = stable_repository_binding_id(
        REPOSITORY_KEY,
        LOCATOR_REPOSITORY_KEY,
    )
    bound = _authority_receipt(
        locator_repository_key=LOCATOR_REPOSITORY_KEY,
        binding_id=binding_id,
    )
    assert bound.binding_id == binding_id
    validate_stable_id(bound.binding_id, "RB", "binding_id")
    assert CandidateAuthorityReceipt.from_dict(bound.to_dict()) == bound


def test_candidate_authority_receipt_matches_human_hash_actor_and_review() -> None:
    declaration = _authority_declaration()
    actor_mismatch = HumanDeclarationSourceRef(
        request_id=declaration.source_ref.request_id,
        actor="bob",
        declaration_hash=declaration.source_ref.declaration_hash,
        created_at=declaration.source_ref.created_at,
        review_id=declaration.source_ref.review_id,
    )

    with pytest.raises(ValueError, match="authorized_source_refs"):
        _authority_receipt(
            authorized_source_refs=(_range_source(), actor_mismatch),
            human_declarations=(declaration,),
        )
    with pytest.raises(ValueError, match="review_id"):
        _authority_receipt(review_id="review-002")
    with pytest.raises(ValueError, match="authorized_source_refs"):
        _authority_receipt(authorized_source_refs=(_range_source(),))

    cli_declaration = _authority_declaration(
        review_id=None,
        origin=HumanDeclarationOrigin.CLI_REQUEST,
    )
    cli_receipt = _authority_receipt(
        authorized_source_refs=(_range_source(), cli_declaration.source_ref),
        human_declarations=(cli_declaration,),
    )
    assert cli_receipt.human_declarations == (cli_declaration,)


@pytest.mark.parametrize("origin", [ProducerType.MODEL, ProducerType.LOCAL])
def test_non_human_authority_receipts_cannot_carry_declarations(
    origin: ProducerType,
) -> None:
    with pytest.raises(ValueError, match="non-human|MODEL|LOCAL|human_declarations"):
        _authority_receipt(origin=origin)


def test_human_authority_receipt_requires_a_declaration() -> None:
    with pytest.raises(ValueError, match="HUMAN|human_declarations"):
        _authority_receipt(
            origin=ProducerType.HUMAN,
            authorized_source_refs=(_range_source(),),
            human_declarations=(),
        )


def test_candidate_authority_receipt_identity_covers_every_authority_field() -> None:
    base_values = {
        "candidate_id": _candidate().candidate_id,
        "authority_repository_key": REPOSITORY_KEY,
        "locator_repository_key": LOCATOR_REPOSITORY_KEY,
        "origin": ProducerType.MODEL,
        "review_id": "review-001",
        "proposal_head_sha": SHA_A,
        "authorized_source_refs": (_range_source(),),
        "human_declarations": (),
        "initial_validation_report_hash": HASH_1,
        "authority_resolution_hash": HASH_2,
        "binding_id": stable_repository_binding_id("base-binding"),
        "created_at": CREATED_AT,
    }
    base = CandidateAuthorityReceipt(**base_values)
    variants = (
        {"candidate_id": _candidate(confidence=MemoryConfidence.MEDIUM).candidate_id},
        {"authority_repository_key": "6" * 64},
        {"locator_repository_key": "7" * 64},
        {"origin": ProducerType.LOCAL},
        {"review_id": "review-002"},
        {"proposal_head_sha": SHA_B},
        {"authorized_source_refs": (_range_source(content_hash=HASH_3),)},
        {"initial_validation_report_hash": HASH_3},
        {"authority_resolution_hash": HASH_3},
        {"binding_id": stable_repository_binding_id("changed-binding")},
        {"created_at": "2026-07-14T12:00:01Z"},
    )
    for override in variants:
        changed = CandidateAuthorityReceipt(**{**base_values, **override})
        assert changed.receipt_id != base.receipt_id

    changed_declaration = _authority_declaration("Approve changed authority text.")
    declaration_receipt = _authority_receipt(
        authorized_source_refs=(_range_source(), changed_declaration.source_ref),
        human_declarations=(changed_declaration,),
    )
    assert declaration_receipt.receipt_id != _authority_receipt().receipt_id


def test_candidate_authority_receipt_rejects_invalid_duplicate_and_unbounded_inputs() -> None:
    declaration = _authority_declaration()
    conflicting_duplicate = HumanDeclarationAuthority(
        source_ref=declaration.source_ref,
        origin=HumanDeclarationOrigin.CLI_REQUEST,
        declaration=declaration.declaration,
    )

    with pytest.raises(ValueError, match="duplicate.*source_ref"):
        _authority_receipt(
            human_declarations=(declaration, conflicting_duplicate),
        )
    with pytest.raises(ValueError, match="authorized_source_refs"):
        _authority_receipt(authorized_source_refs=())
    with pytest.raises(ValueError, match="maximum item count"):
        _authority_receipt(
            authorized_source_refs=tuple(_range_source() for _ in range(65)),
        )
    with pytest.raises(ValueError, match="maximum item count"):
        _authority_receipt(human_declarations=tuple(declaration for _ in range(65)))
    with pytest.raises(ValueError, match="candidate_id"):
        _authority_receipt(candidate_id="MC-short")
    with pytest.raises(ValueError, match="proposal_head_sha"):
        _authority_receipt(proposal_head_sha="abc123")
    with pytest.raises(ValueError, match="initial_validation_report_hash"):
        _authority_receipt(initial_validation_report_hash="1" * 32)
    with pytest.raises(ValueError, match="origin"):
        _authority_receipt(origin="human")


def test_memory_scope_normalizes_sorts_deduplicates_and_is_immutable() -> None:
    scope = MemoryScope(
        paths=("payments\\**", "payments/**", "./api/*.py"),
        symbols=("payments.money.total", "payments.money.total"),
        contracts=("NUMERIC_CORRECTNESS", "numeric_correctness"),
        languages=("Python", "python"),
    )

    assert scope.paths == ("api/*.py", "payments/**")
    assert scope.symbols == ("payments.money.total",)
    assert scope.contracts == ("numeric_correctness",)
    assert scope.languages == ("python",)
    assert isinstance(scope.paths, tuple)
    with pytest.raises(FrozenInstanceError):
        scope.paths = ()


@pytest.mark.parametrize(
    "path",
    ("/absolute.py", "C:/absolute.py", "../outside.py", ".git/config", ".env"),
)
def test_memory_scope_rejects_unsafe_path_globs(path: str) -> None:
    with pytest.raises(ValueError, match="repository-relative|sensitive"):
        MemoryScope(paths=(path,))


def test_empty_scope_is_only_valid_for_allowed_repository_wide_kinds() -> None:
    with pytest.raises(ValueError, match="scope must not be empty"):
        _candidate(scope=MemoryScope())

    candidate = _candidate(
        kind=MemoryKind.REVIEW_RULE,
        scope=MemoryScope(),
        statement="Always inspect authorization boundaries.",
    )
    assert candidate.scope.is_empty


def test_candidate_id_and_canonical_serialization_ignore_input_order() -> None:
    left = _candidate()
    right = _candidate(
        scope=MemoryScope(
            languages=("python",),
            contracts=("numeric_correctness",),
            symbols=("payments.money.calculate_total",),
            paths=("payments/**",),
        ),
        source_refs=tuple(reversed(left.source_refs)),
    )

    assert left.candidate_id == right.candidate_id
    assert left.content_fingerprint == right.content_fingerprint
    assert left.to_json() == right.to_json()
    assert left.candidate_id.startswith("MC-")
    assert len(left.candidate_id) == 67


@pytest.mark.parametrize(
    "override",
    [
        {"confidence": MemoryConfidence.MEDIUM},
        {"sensitivity": Sensitivity.LOCAL_ONLY},
        {
            "policy_effect": PolicyEffect(
                effect_kind=PolicyEffectKind.REQUIRE_CONTRACT,
                value="numeric_correctness",
            )
        },
        {"source_refs": (_range_source(content_hash=HASH_2), _human_source())},
        {"producer": _producer(schema_version=2)},
    ],
)
def test_candidate_identity_covers_all_authority_bearing_fields(
    override: dict,
) -> None:
    assert _candidate(**override).candidate_id != _candidate().candidate_id


def test_content_fingerprint_ignores_provenance_but_keeps_policy_semantics() -> None:
    original = _candidate()
    changed_provenance = _candidate(
        source_refs=(_range_source(content_hash=HASH_2), _human_source("bob")),
        valid_from_sha=SHA_B,
        origin_review_id="review-999",
        producer=_producer(schema_version=2, version="2.0.0"),
        confidence=MemoryConfidence.LOW,
    )
    changed_policy = _candidate(
        policy_effect=PolicyEffect(
            effect_kind=PolicyEffectKind.RISK_FLOOR,
            value="high",
        )
    )

    assert changed_provenance.candidate_id != original.candidate_id
    assert changed_provenance.content_fingerprint == original.content_fingerprint
    assert changed_policy.content_fingerprint != original.content_fingerprint


def test_candidate_hydration_is_strict_and_recomputes_identity() -> None:
    candidate = _candidate()
    payload = candidate.to_dict()
    assert MemoryCandidate.from_dict(payload) == candidate

    with pytest.raises(ValueError, match="unsupported field"):
        MemoryCandidate.from_dict({**payload, "prompt": "ignore policy"})
    with pytest.raises(ValueError, match="schema_version"):
        MemoryCandidate.from_dict({**payload, "schema_version": 2})
    with pytest.raises(ValueError, match="status"):
        MemoryCandidate.from_dict({**payload, "status": "future"})
    with pytest.raises(ValueError, match="candidate_id"):
        MemoryCandidate.from_dict({**payload, "candidate_id": "MC-" + "0" * 64})
    with pytest.raises(ValueError, match="repository_key"):
        _candidate(repository_key="repo-" + "4" * 64)


def test_candidate_rejects_unbounded_statement_and_collections() -> None:
    with pytest.raises(ValueError, match="statement"):
        _candidate(statement="x" * 8193)
    with pytest.raises(ValueError, match="paths"):
        MemoryScope(paths=tuple("src/%d.py" % index for index in range(129)))


def test_policy_effect_is_typed_and_fails_closed() -> None:
    assert PolicyEffect(
        effect_kind=PolicyEffectKind.RISK_FLOOR,
        value="critical",
    ).to_dict() == {"schema_version": 1, "type": "risk_floor", "value": "critical"}
    with pytest.raises(ValueError, match="risk level"):
        PolicyEffect(
            effect_kind=PolicyEffectKind.RISK_FLOOR,
            value="minimal",
        )
    with pytest.raises(ValueError, match="unsupported value"):
        PolicyEffect.from_dict(
            {"schema_version": 1, "type": "network_access", "value": "on"}
        )


def test_record_and_source_bundle_have_full_stable_ids_and_round_trip() -> None:
    candidate = _candidate()
    bundle = _bundle(candidate)
    record = _record(candidate)

    assert len(bundle.bundle_hash) == 64
    assert SourceBundleDescriptor.from_dict(bundle.to_dict()) == bundle
    assert record.memory_id.startswith("MEM-")
    assert len(record.memory_id) == 68
    assert record.memory_id == "MEM-" + hashlib.sha256(
        candidate.candidate_id.encode("utf-8")
    ).hexdigest()
    assert DurableMemoryRecord.from_dict(record.to_dict()) == record

    tampered = {**record.to_dict(), "memory_id": "MEM-" + "a" * 32}
    with pytest.raises(ValueError, match="memory_id"):
        DurableMemoryRecord.from_dict(tampered)


def test_finding_and_feedback_round_trip_and_enforce_decision_semantics() -> None:
    finding = _finding()
    feedback = _feedback()

    assert FindingSnapshot.from_dict(finding.to_dict()) == finding
    assert len(finding.finding_hash) == 64
    assert feedback.feedback_id.startswith("FB-")
    assert len(feedback.feedback_id) == 67
    assert FeedbackRecord.from_dict(feedback.to_dict()) == feedback

    payload = feedback.to_dict()
    payload["decision"] = FeedbackDecision.SEVERITY_CHANGED.value
    with pytest.raises(ValueError, match="final_severity"):
        FeedbackRecord.from_dict(payload)

    tampered_finding = {**finding.to_dict(), "claim": "Different claim."}
    with pytest.raises(ValueError, match="finding_hash"):
        FindingSnapshot.from_dict(tampered_finding)


def test_stable_event_and_request_ids_use_full_sha256_and_validate_prefix() -> None:
    event_id = stable_event_id("approve", _candidate().candidate_id)
    request_id = stable_request_id("cli", "approve", "amy")

    assert event_id.startswith("EVT-") and len(event_id) == 68
    assert request_id.startswith("REQ-") and len(request_id) == 68
    validate_stable_id(event_id, "EVT", "event_id")
    validate_stable_id(request_id, "REQ", "request_id")
    with pytest.raises(ValueError, match="full SHA-256"):
        validate_stable_id("EVT-" + "a" * 32, "EVT", "event_id")
    with pytest.raises(ValueError, match="full SHA-256"):
        validate_stable_id("REQ-" + "g" * 64, "REQ", "request_id")


def test_generation_selection_input_and_decision_are_strict_round_trips() -> None:
    generations = GenerationMetadata(
        store_schema_version=2,
        memory_generation=3,
        feedback_generation=4,
        knowledge_generation=5,
    )
    selection_input = MemorySelectionInput(
        review_id="review-001",
        repository_key=REPOSITORY_KEY,
        base_sha=SHA_B,
        head_sha=SHA_A,
        changed_paths=("payments\\money.py", "payments/money.py"),
        changed_symbols=("payments.money.calculate_total",),
        contracts=("NUMERIC_CORRECTNESS",),
        languages=("Python",),
        generations=generations,
        selection_policy_version="memory_selection_v1",
    )
    decision = MemorySelectionDecision(
        memory_id=_record(_candidate()).memory_id,
        applicability=Applicability.SELECTED,
        matched_scope=_scope(),
        reason_codes=("path_match", "contract_match"),
        rank=0,
    )

    assert GenerationMetadata.from_dict(generations.to_dict()) == generations
    assert MemorySelectionInput.from_dict(selection_input.to_dict()) == selection_input
    assert selection_input.changed_paths == ("payments/money.py",)
    assert MemorySelectionDecision.from_dict(decision.to_dict()) == decision


@pytest.mark.parametrize("store_schema_version", (1, 2))
def test_generation_metadata_accepts_current_and_legacy_store_versions(
    store_schema_version: int,
) -> None:
    generations = GenerationMetadata(
        store_schema_version=store_schema_version,
        memory_generation=0,
        feedback_generation=0,
        knowledge_generation=0,
    )

    assert GenerationMetadata.from_dict(generations.to_dict()) == generations

    for invalid in (0, 3, True):
        with pytest.raises(ValueError, match="supported version"):
            GenerationMetadata(
                store_schema_version=invalid,
                memory_generation=0,
                feedback_generation=0,
                knowledge_generation=0,
            )


def test_repository_knowledge_key_and_entry_bind_every_cache_dimension() -> None:
    key = RepositoryKnowledgeKey(
        repository_key=REPOSITORY_KEY,
        revision_binding="head@" + SHA_A,
        capability=RepositoryKnowledgeCapability.SYMBOL_INDEX,
        analyzer_name="python-ast",
        analyzer_version="3.12-v1",
        configuration_digest=HASH_1,
        input_digest=HASH_2,
    )
    entry = RepositoryKnowledgeEntry(
        key=key,
        blob_hash=HASH_3,
        size_bytes=1024,
        content_type="application/vnd.review-agent.symbol-index+json",
        artifact_schema="symbol_index_v1",
        summary_hash=HASH_1,
        created_at=CREATED_AT,
        pinned_by_review_ids=("review-002", "review-001", "review-001"),
    )

    assert RepositoryKnowledgeKey.from_dict(key.to_dict()) == key
    assert RepositoryKnowledgeEntry.from_dict(entry.to_dict()) == entry
    assert entry.pinned_by_review_ids == ("review-001", "review-002")
    assert entry.entry_id.startswith("RKE-") and len(entry.entry_id) == 68

    changed_key = RepositoryKnowledgeKey(
        repository_key=REPOSITORY_KEY,
        revision_binding="head@" + SHA_A,
        capability=RepositoryKnowledgeCapability.SYMBOL_INDEX,
        analyzer_name="python-ast",
        analyzer_version="3.12-v2",
        configuration_digest=HASH_1,
        input_digest=HASH_2,
    )
    assert changed_key.key_hash != key.key_hash


def test_feedback_calibration_summary_requires_thresholds_before_signals() -> None:
    feedback = tuple(_feedback(index, "review-%03d" % (index % 3)) for index in range(5))
    signal = FeedbackCalibrationSignal(
        signal_kind=FeedbackCalibrationSignalKind.EVIDENCE_GAP_WARNING,
        scope=MemoryScope(contracts=("numeric_correctness",)),
        message="Require stronger rounding evidence.",
        sample_count=5,
        review_count=3,
        feedback_ids=tuple(item.feedback_id for item in feedback),
    )
    summary = FeedbackCalibrationSummary(
        repository_key=REPOSITORY_KEY,
        feedback_generation=4,
        policy_version="feedback_aggregation_v1",
        eligible=True,
        source_feedback_ids=tuple(item.feedback_id for item in feedback),
        source_review_ids=("review-000", "review-001", "review-002"),
        decision_counts=((FeedbackDecision.ACCEPTED, 5),),
        signals=(signal,),
        created_at=CREATED_AT,
    )

    assert FeedbackCalibrationSummary.from_dict(summary.to_dict()) == summary
    assert len(summary.summary_hash) == 64
    with pytest.raises(ValueError, match="at least 5"):
        FeedbackCalibrationSummary(
            repository_key=REPOSITORY_KEY,
            feedback_generation=4,
            policy_version="feedback_aggregation_v1",
            eligible=True,
            source_feedback_ids=tuple(item.feedback_id for item in feedback[:4]),
            source_review_ids=("review-000", "review-001", "review-002"),
            decision_counts=((FeedbackDecision.ACCEPTED, 4),),
            signals=(signal,),
            created_at=CREATED_AT,
        )


def test_memory_snapshot_pins_canonical_records_decisions_feedback_and_knowledge() -> None:
    candidate = _candidate()
    record = _record(candidate)
    decision = MemorySelectionDecision(
        memory_id=record.memory_id,
        applicability=Applicability.SELECTED,
        matched_scope=record.scope,
        reason_codes=("path_match",),
        rank=0,
    )
    knowledge_key = RepositoryKnowledgeKey(
        repository_key=REPOSITORY_KEY,
        revision_binding="head@" + SHA_A,
        capability=RepositoryKnowledgeCapability.SYMBOL_INDEX,
        analyzer_name="python-ast",
        analyzer_version="3.12-v1",
        configuration_digest=HASH_1,
        input_digest=HASH_2,
    )
    knowledge = RepositoryKnowledgeEntry(
        key=knowledge_key,
        blob_hash=HASH_3,
        size_bytes=128,
        content_type="application/json",
        artifact_schema="symbol_index_v1",
        created_at=CREATED_AT,
    )
    snapshot = MemorySnapshot(
        repository_key=REPOSITORY_KEY,
        base_sha=SHA_B,
        head_sha=SHA_A,
        generations=GenerationMetadata(
            store_schema_version=1,
            memory_generation=3,
            feedback_generation=4,
            knowledge_generation=5,
        ),
        selection_policy_version="memory_selection_v1",
        eligible_records=(record,),
        applicability_decisions=(decision,),
        feedback_calibration_summary=None,
        repository_knowledge_refs=(knowledge.entry_id,),
        created_at=CREATED_AT,
    )

    assert snapshot.snapshot_id.startswith("MSNAP-")
    assert len(snapshot.snapshot_id) == 70
    assert len(snapshot.snapshot_hash) == 64
    payload = snapshot.to_dict()
    assert payload["store_schema_version"] == 1
    assert payload["memory_generation"] == 3
    assert payload["feedback_generation"] == 4
    assert payload["knowledge_generation"] == 5
    assert "generations" not in payload
    assert MemorySnapshot.from_dict(payload) == snapshot
    with pytest.raises(ValueError, match="snapshot_hash"):
        MemorySnapshot.from_dict({**snapshot.to_dict(), "snapshot_hash": HASH_1})
    with pytest.raises(ValueError, match="generations"):
        MemorySnapshot.from_dict(
            {
                **payload,
                "generations": snapshot.generations.to_dict(),
            }
        )


def test_memory_snapshot_requires_a_selected_decision_for_each_record() -> None:
    record = _record(_candidate())
    with pytest.raises(ValueError, match="selected applicability decision"):
        MemorySnapshot(
            repository_key=REPOSITORY_KEY,
            base_sha=SHA_B,
            head_sha=SHA_A,
            generations=GenerationMetadata(
                store_schema_version=1,
                memory_generation=1,
                feedback_generation=1,
                knowledge_generation=1,
            ),
            selection_policy_version="memory_selection_v1",
            eligible_records=(record,),
            applicability_decisions=(),
            feedback_calibration_summary=None,
            repository_knowledge_refs=(),
            created_at=CREATED_AT,
        )

    selected_without_record = MemorySelectionDecision(
        memory_id=record.memory_id,
        applicability=Applicability.SELECTED,
        matched_scope=record.scope,
        reason_codes=("path_match",),
        rank=0,
    )
    with pytest.raises(ValueError, match="selected applicability decisions"):
        MemorySnapshot(
            repository_key=REPOSITORY_KEY,
            base_sha=SHA_B,
            head_sha=SHA_A,
            generations=GenerationMetadata(
                store_schema_version=1,
                memory_generation=1,
                feedback_generation=1,
                knowledge_generation=1,
            ),
            selection_policy_version="memory_selection_v1",
            eligible_records=(),
            applicability_decisions=(selected_without_record,),
            feedback_calibration_summary=None,
            repository_knowledge_refs=(),
            created_at=CREATED_AT,
        )


def test_memory_execution_config_matches_final_session_v5_contract() -> None:
    config = MemoryExecutionConfig(
        mode=MemoryMode.READ_WRITE,
        root_path="C:/Users/Amy/AppData/Local/code-review-agent/memory",
        required=False,
    )

    assert MemoryExecutionConfig.from_dict(config.to_dict()) == config
    assert config.max_snapshot_records == 2000
    assert config.max_snapshot_bytes == 8_388_608
    assert config.max_context_records == 12
    assert config.max_query_results == 8
    drive_root = MemoryExecutionConfig(
        mode=MemoryMode.READ,
        root_path="C:/",
    )
    assert drive_root.root_path == "C:/"
    assert MemoryExecutionConfig.from_dict(drive_root.to_dict()) == drive_root
    with pytest.raises(ValueError, match="required=true"):
        MemoryExecutionConfig(
            mode=MemoryMode.OFF,
            root_path="C:/memory",
            required=True,
        )
    with pytest.raises(ValueError, match="absolute"):
        MemoryExecutionConfig(
            mode=MemoryMode.READ,
            root_path="relative/memory",
        )
    with pytest.raises(ValueError, match="selection_policy_version"):
        MemoryExecutionConfig.from_dict(
            {
                **config.to_dict(),
                "selection_policy_version": "memory_selection_v2",
            }
        )


def test_memory_execution_config_canonicalizes_construction_and_rejects_noncanonical_wire_paths() -> None:
    config = MemoryExecutionConfig(
        mode=MemoryMode.READ,
        root_path=r"C:\memory\.\nested",
    )

    assert config.root_path == "C:/memory/nested"
    for noncanonical in (
        "C:/memory/./nested",
        "C://memory//nested",
        r"C:\memory\nested",
    ):
        with pytest.raises(ValueError, match="canonical absolute path"):
            MemoryExecutionConfig.from_dict(
                {
                    **config.to_dict(),
                    "root_path": noncanonical,
                }
            )

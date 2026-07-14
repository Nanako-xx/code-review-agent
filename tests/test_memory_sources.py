from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from conftest import run_git
import review_agent.memory_sources as memory_sources_module
from review_agent.memory_models import (
    CandidateStatus,
    GitCommitSourceRef,
    HumanDeclarationSourceRef,
    MemoryCandidate,
    MemoryConfidence,
    MemoryKind,
    MemoryScope,
    ObservationSourceRef,
    Producer,
    ProducerType,
    RepositoryRangeSourceRef,
    RepositorySymbolSourceRef,
    Sensitivity,
    SessionArtifactSourceRef,
    SourceRef,
    SymbolHashKind,
    ValidityPolicy,
    stable_request_id,
)
from review_agent.memory_identity import repository_key as derive_repository_key
from review_agent.memory_sources import (
    HumanDeclarationOrigin,
    SourceValidationCode,
    SourceValidationError,
    SourceValidator,
    TrustedCandidateProvenance,
    TrustedHumanDeclaration,
    git_commit_metadata_hash,
    human_declaration_hash,
    repository_range_hash,
    repository_symbol_hash,
    scan_sensitive_text,
)
from review_agent.observations import ObservationStore
from review_agent.revision import ResolvedRevisions, RevisionResolver
from review_agent.run_state import RunPhase
from review_agent.session import (
    ReviewExecutionConfig,
    initial_session_manifest,
    session_phases_for_schema,
)
from review_agent.session_store import SessionStore


NOW = "2026-07-14T00:00:00Z"
REVIEW_ID = "review-memory-source"


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _revision_binding(sha: str) -> str:
    return f"{sha}..{sha}"


def _create_session(
    repo: Path,
    *,
    review_id: str = REVIEW_ID,
    artifact_content: str = '{"rule":"round totals once"}\n',
    observation_content: str = "def add(a, b):\n    return a + b\n",
    observation_path: str = "app.py",
    complete: bool = True,
) -> tuple[
    SessionStore,
    SessionArtifactSourceRef,
    ObservationSourceRef,
]:
    resolver = RevisionResolver()
    sha = resolver.resolve_commit(repo, "HEAD")
    identity = resolver.repository_identity(repo)
    run_dir = repo / ".review-agent" / "runs" / review_id
    store = SessionStore(run_dir)
    store.create(
        initial_session_manifest(
            review_id=review_id,
            repository=identity,
            revisions=ResolvedRevisions("HEAD", "HEAD", sha, sha),
            execution=ReviewExecutionConfig(
                reviewer_provider="fake",
                reviewer_model=None,
                reviewer_base_url=None,
                reviewer_api_key_env="REVIEW_AGENT_API_KEY",
                reviewer_mode="single",
                reviewer_loop="single-shot",
                non_interactive=True,
            ),
            now=NOW,
        )
    )

    artifact_path = run_dir / "evidence.json"
    artifact_path.write_text(artifact_content, encoding="utf-8")
    manifest = store.register_existing_artifact(
        name="review_brief",
        relative_path="evidence.json",
        schema="review_brief_v1",
        phase=RunPhase.REPORTING,
        revision_binding=_revision_binding(sha),
        now="2026-07-14T00:00:01Z",
    )
    artifact = manifest.artifacts["review_brief"]

    observation_store = ObservationStore(run_dir)
    observation = observation_store.record(
        source="git.read_range",
        revision=f"head@{sha}",
        path=observation_path,
        line_start=1,
        line_end=2,
        raw_content=observation_content,
        context_view="validated observation",
    )
    store.register_existing_artifact(
        name="observations",
        relative_path="observations.jsonl",
        schema="observation_log_jsonl_v1",
        phase=RunPhase.REPORTING,
        revision_binding=_revision_binding(sha),
        now="2026-07-14T00:00:02Z",
    )
    if complete:
        _complete_session(store, ["review_brief", "observations"])

    return (
        store,
        SessionArtifactSourceRef(
            review_id=review_id,
            artifact_name="review_brief",
            artifact_schema=artifact.schema,
            revision_binding=_revision_binding(sha),
            artifact_hash=artifact.sha256,
        ),
        ObservationSourceRef(
            review_id=review_id,
            observation_id=observation.observation_id,
            revision_binding=observation.revision,
            content_hash=observation.content_hash,
        ),
    )


def _complete_session(
    store: SessionStore,
    reporting_artifacts: list[str],
) -> None:
    current = store.load()
    for index, phase in enumerate(
        session_phases_for_schema(current.schema_version),
        start=1,
    ):
        current = store.mark_phase_completed(
            phase,
            reporting_artifacts if phase is RunPhase.REPORTING else [],
            f"2026-07-14T01:{index:02d}:00Z",
        )


def _declaration(
    text: str = "All monetary totals are rounded only at the boundary.",
) -> TrustedHumanDeclaration:
    return TrustedHumanDeclaration(
        request_id=stable_request_id("memory-source", "amy"),
        actor="amy",
        created_at=NOW,
        declaration=text,
        origin=HumanDeclarationOrigin.USER_REQUEST,
        review_id=REVIEW_ID,
    )


def _candidate(
    repo: Path,
    sha: str,
    source_ref: GitCommitSourceRef,
    *,
    statement: str = "Review arithmetic changes for boundary rounding.",
    sensitivity: Sensitivity = Sensitivity.NORMAL,
    producer_type: ProducerType = ProducerType.LOCAL,
) -> MemoryCandidate:
    return MemoryCandidate(
        repository_key=derive_repository_key(
            RevisionResolver().repository_identity(repo)
        ),
        kind=MemoryKind.REVIEW_RULE,
        statement=statement,
        scope=MemoryScope(paths=("app.py",)),
        source_refs=(source_ref,),
        valid_from_sha=sha,
        validity_policies=(ValidityPolicy.SOURCE_CONTENT_HASH,),
        confidence=MemoryConfidence.HIGH,
        sensitivity=sensitivity,
        policy_effect=None,
        producer=Producer(producer_type, "memory-curator", "1.0"),
        origin_review_id=REVIEW_ID,
        status=CandidateStatus.PROPOSED,
        created_at=NOW,
    )


def _runtime_provenance(
    sha: str,
    *,
    origin: ProducerType = ProducerType.LOCAL,
    review_id: str = REVIEW_ID,
    allowed_source_refs: tuple[SourceRef, ...] = (),
) -> TrustedCandidateProvenance:
    return TrustedCandidateProvenance(
        origin=origin,
        review_id=review_id,
        target_head_sha=sha,
        allowed_source_refs=allowed_source_refs,
    )


def _issue_codes(report: object) -> set[SourceValidationCode]:
    return {issue.code for issue in report.issues}  # type: ignore[attr-defined]


def test_validates_all_six_typed_sources_with_exact_authorities(
    git_repo: Path,
) -> None:
    sha = run_git(git_repo, "rev-parse", "HEAD")
    _, artifact_ref, observation_ref = _create_session(git_repo)
    declaration = _declaration()
    refs = (
        RepositoryRangeSourceRef(
            revision=sha,
            path="app.py",
            line_start=1,
            line_end=2,
            content_hash=repository_range_hash(git_repo, sha, "app.py", 1, 2),
        ),
        RepositorySymbolSourceRef(
            revision=sha,
            path="app.py",
            qualified_name="app.add",
            hash_kind=SymbolHashKind.BODY,
            content_hash=repository_symbol_hash(
                git_repo,
                sha,
                "app.py",
                "app.add",
                SymbolHashKind.BODY,
            ),
        ),
        GitCommitSourceRef(
            commit_sha=sha,
            metadata_hash=git_commit_metadata_hash(git_repo, sha),
        ),
        observation_ref,
        artifact_ref,
        declaration.to_source_ref(),
    )
    validator = SourceValidator(
        git_repo,
        human_declarations=(declaration,),
    )

    report = validator.validate_sources(
        refs,
        sensitivity=Sensitivity.NORMAL,
        statement="Round only at the payment boundary.",
    )

    assert report.valid
    assert report.persistable
    assert report.remote_sendable
    assert report.retain_content
    assert not report.issues
    assert len(report.source_results) == 6
    assert all(result.valid for result in report.source_results)
    assert {result.source_type for result in report.source_results} == {
        ref.source_type for ref in refs
    }
    assert all(result.source_ref_hash for result in report.source_results)
    assert "return a + b" not in report.to_json()


@pytest.mark.parametrize(
    ("source_factory", "expected_code"),
    [
        (
            lambda repo, sha: RepositoryRangeSourceRef(
                revision=sha,
                path="app.py",
                line_start=1,
                line_end=2,
                content_hash="0" * 64,
            ),
            SourceValidationCode.HASH_MISMATCH,
        ),
        (
            lambda repo, sha: RepositoryRangeSourceRef(
                revision=sha,
                path="app.py",
                line_start=99,
                line_end=99,
                content_hash=_sha256(""),
            ),
            SourceValidationCode.RANGE_OUT_OF_BOUNDS,
        ),
        (
            lambda repo, sha: RepositoryRangeSourceRef(
                revision="f" * 40,
                path="app.py",
                line_start=1,
                line_end=1,
                content_hash=_sha256("def add(a, b):\n"),
            ),
            SourceValidationCode.REVISION_NOT_FOUND,
        ),
        (
            lambda repo, sha: RepositorySymbolSourceRef(
                revision=sha,
                path="app.py",
                qualified_name="app.missing",
                hash_kind=SymbolHashKind.SIGNATURE,
                content_hash="0" * 64,
            ),
            SourceValidationCode.SYMBOL_NOT_FOUND,
        ),
    ],
)
def test_repository_sources_fail_closed_on_drift_or_missing_evidence(
    git_repo: Path,
    source_factory: object,
    expected_code: SourceValidationCode,
) -> None:
    sha = run_git(git_repo, "rev-parse", "HEAD")
    source_ref = source_factory(git_repo, sha)  # type: ignore[operator]

    report = SourceValidator(git_repo).validate_sources(
        (source_ref,),
        sensitivity=Sensitivity.NORMAL,
    )

    assert not report.valid
    assert not report.persistable
    assert not report.remote_sendable
    assert expected_code in _issue_codes(report)
    with pytest.raises(SourceValidationError) as captured:
        report.require_valid()
    assert captured.value.code is expected_code
    assert str(captured.value) == f"source validation failed: {expected_code.value}"


def test_symbol_signature_and_body_use_distinct_exact_hashes(git_repo: Path) -> None:
    sha = run_git(git_repo, "rev-parse", "HEAD")

    signature_hash = repository_symbol_hash(
        git_repo,
        sha,
        "app.py",
        "add",
        SymbolHashKind.SIGNATURE,
    )
    body_hash = repository_symbol_hash(
        git_repo,
        sha,
        "app.py",
        "app.add",
        SymbolHashKind.BODY,
    )

    assert signature_hash != body_hash
    refs = (
        RepositorySymbolSourceRef(
            revision=sha,
            path="app.py",
            qualified_name="add",
            hash_kind=SymbolHashKind.SIGNATURE,
            content_hash=signature_hash,
        ),
        RepositorySymbolSourceRef(
            revision=sha,
            path="app.py",
            qualified_name="app.add",
            hash_kind=SymbolHashKind.BODY,
            content_hash=body_hash,
        ),
    )
    assert SourceValidator(git_repo).validate_sources(
        refs,
        sensitivity=Sensitivity.NORMAL,
    ).valid


def test_repository_range_hashes_exact_committed_line_bytes_not_worktree(
    git_repo: Path,
) -> None:
    (git_repo / "crlf.txt").write_bytes(b"first\r\nsecond\r\n")
    blob = run_git(
        git_repo,
        "hash-object",
        "-w",
        "--no-filters",
        "crlf.txt",
    )
    run_git(
        git_repo,
        "update-index",
        "--add",
        "--cacheinfo",
        f"100644,{blob},crlf.txt",
    )
    run_git(git_repo, "commit", "-m", "add crlf source")
    sha = run_git(git_repo, "rev-parse", "HEAD")
    expected = hashlib.sha256(b"first\r\n").hexdigest()
    source_ref = RepositoryRangeSourceRef(
        revision=sha,
        path="crlf.txt",
        line_start=1,
        line_end=1,
        content_hash=expected,
    )
    (git_repo / "crlf.txt").write_bytes(b"uncommitted drift\n")

    assert repository_range_hash(
        git_repo,
        sha,
        "crlf.txt",
        1,
        1,
    ) == expected
    assert SourceValidator(git_repo).validate_sources(
        (source_ref,),
        sensitivity=Sensitivity.NORMAL,
    ).valid


def test_repository_tree_symlink_is_not_treated_as_source_file(
    git_repo: Path,
) -> None:
    target = git_repo / "link-target.txt"
    target.write_text("app.py", encoding="utf-8")
    blob = run_git(git_repo, "hash-object", "-w", "link-target.txt")
    run_git(
        git_repo,
        "update-index",
        "--add",
        "--cacheinfo",
        f"120000,{blob},linked.py",
    )
    run_git(git_repo, "commit", "-m", "add tree symlink")
    sha = run_git(git_repo, "rev-parse", "HEAD")
    source_ref = RepositoryRangeSourceRef(
        revision=sha,
        path="linked.py",
        line_start=1,
        line_end=1,
        content_hash=_sha256("app.py\n"),
    )

    report = SourceValidator(git_repo).validate_sources(
        (source_ref,),
        sensitivity=Sensitivity.NORMAL,
    )

    assert SourceValidationCode.SOURCE_NOT_REGULAR in _issue_codes(report)


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda ref: replace(ref, artifact_schema="other_schema_v1"),
            SourceValidationCode.DESCRIPTOR_SCHEMA_MISMATCH,
        ),
        (
            lambda ref: replace(ref, artifact_hash="0" * 64),
            SourceValidationCode.HASH_MISMATCH,
        ),
        (
            lambda ref: replace(
                ref,
                revision_binding=f"{'a' * 40}..{'b' * 40}",
            ),
            SourceValidationCode.REVISION_BINDING_MISMATCH,
        ),
    ],
)
def test_session_artifact_requires_exact_typed_descriptor_binding(
    git_repo: Path,
    mutate: object,
    expected_code: SourceValidationCode,
) -> None:
    _, artifact_ref, _ = _create_session(git_repo)
    invalid_ref = mutate(artifact_ref)  # type: ignore[operator]

    report = SourceValidator(git_repo).validate_sources(
        (invalid_ref,),
        sensitivity=Sensitivity.NORMAL,
    )

    assert expected_code in _issue_codes(report)


def test_session_artifact_rejects_symlink_escape(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, artifact_ref, _ = _create_session(git_repo)
    artifact_path = store.run_dir / "evidence.json"
    outside = git_repo / "outside-evidence.json"
    outside.write_bytes(artifact_path.read_bytes())
    artifact_path.unlink()
    try:
        artifact_path.symlink_to(outside)
    except OSError:
        artifact_path.write_bytes(outside.read_bytes())
        original_is_symlink = Path.is_symlink

        def report_artifact_as_symlink(path: Path) -> bool:
            return path == artifact_path or original_is_symlink(path)

        monkeypatch.setattr(Path, "is_symlink", report_artifact_as_symlink)

    report = SourceValidator(git_repo).validate_sources(
        (artifact_ref,),
        sensitivity=Sensitivity.NORMAL,
    )

    assert SourceValidationCode.SESSION_ARTIFACT_INVALID in _issue_codes(report)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.py",
        ".git/config",
        ".env",
        "C:/private/credentials.txt",
        "/private/credentials.txt",
    ],
)
def test_observation_authority_does_not_bypass_source_path_safety(
    git_repo: Path,
    unsafe_path: str,
) -> None:
    _, _, observation_ref = _create_session(
        git_repo,
        observation_path=unsafe_path,
    )

    report = SourceValidator(git_repo).validate_sources(
        (observation_ref,),
        sensitivity=Sensitivity.NORMAL,
    )

    assert SourceValidationCode.UNSAFE_PATH in _issue_codes(report)


def test_observation_requires_registered_session_authority_and_exact_hash(
    git_repo: Path,
) -> None:
    store, _, observation_ref = _create_session(git_repo)
    wrong_hash = replace(observation_ref, content_hash="0" * 64)

    mismatch = SourceValidator(git_repo).validate_sources(
        (wrong_hash,),
        sensitivity=Sensitivity.NORMAL,
    )
    assert SourceValidationCode.HASH_MISMATCH in _issue_codes(mismatch)

    observation = ObservationStore.load(
        store.run_dir,
        {observation_ref.revision_binding},
    ).list_observations()[0]
    (store.run_dir / observation.raw_artifact_ref).write_text(
        "tampered",
        encoding="utf-8",
    )
    tampered = SourceValidator(git_repo).validate_sources(
        (observation_ref,),
        sensitivity=Sensitivity.NORMAL,
    )
    assert SourceValidationCode.OBSERVATION_UNTRUSTED in _issue_codes(tampered)


def test_human_declaration_requires_explicit_trusted_request_binding(
    git_repo: Path,
) -> None:
    declaration = _declaration()
    source_ref = declaration.to_source_ref()
    validator = SourceValidator(
        git_repo,
        human_declarations=(declaration,),
    )

    assert validator.validate_sources(
        (source_ref,),
        sensitivity=Sensitivity.NORMAL,
    ).valid
    assert source_ref.declaration_hash == human_declaration_hash(
        declaration.declaration
    )

    mismatched = HumanDeclarationSourceRef(
        request_id=source_ref.request_id,
        actor="mallory",
        declaration_hash=source_ref.declaration_hash,
        created_at=source_ref.created_at,
        review_id=source_ref.review_id,
    )
    mismatch_report = validator.validate_sources(
        (mismatched,),
        sensitivity=Sensitivity.NORMAL,
    )
    assert SourceValidationCode.HUMAN_DECLARATION_MISMATCH in _issue_codes(
        mismatch_report
    )

    unauthorized = SourceValidator(git_repo).validate_sources(
        (source_ref,),
        sensitivity=Sensitivity.NORMAL,
    )
    assert SourceValidationCode.HUMAN_DECLARATION_UNAUTHORIZED in _issue_codes(
        unauthorized
    )


def test_untyped_json_and_non_allowlisted_refs_are_rejected_without_hydration(
    git_repo: Path,
) -> None:
    class RogueSourceRef(SourceRef):
        def to_dict(self) -> dict[str, object]:
            return {"type": "git_commit"}

    sha = run_git(git_repo, "rev-parse", "HEAD")
    allowed = GitCommitSourceRef(commit_sha=sha)
    other = RepositoryRangeSourceRef(
        revision=sha,
        path="app.py",
        line_start=1,
        line_end=1,
        content_hash=repository_range_hash(git_repo, sha, "app.py", 1, 1),
    )
    validator = SourceValidator(git_repo, allowed_source_refs=(allowed,))

    raw_report = validator.validate_sources(  # type: ignore[arg-type]
        ({"type": "git_commit", "commit_sha": sha},),
        sensitivity=Sensitivity.NORMAL,
    )
    assert SourceValidationCode.UNTYPED_SOURCE in _issue_codes(raw_report)

    subclass_report = validator.validate_sources(  # type: ignore[arg-type]
        (RogueSourceRef(),),
        sensitivity=Sensitivity.NORMAL,
    )
    assert SourceValidationCode.UNTYPED_SOURCE in _issue_codes(subclass_report)

    empty_report = validator.validate_sources(
        (),
        sensitivity=Sensitivity.NORMAL,
    )
    assert SourceValidationCode.INVALID_INPUT in _issue_codes(empty_report)

    allowlist_report = validator.validate_sources(
        (other,),
        sensitivity=Sensitivity.NORMAL,
        require_allowlisted=True,
    )
    assert SourceValidationCode.SOURCE_NOT_ALLOWLISTED in _issue_codes(
        allowlist_report
    )


@pytest.mark.parametrize(
    ("sensitivity", "valid", "persistable", "remote_sendable", "retain_content"),
    [
        (Sensitivity.NORMAL, True, True, True, True),
        (Sensitivity.LOCAL_ONLY, True, True, False, True),
        (Sensitivity.BLOCKED, False, False, False, False),
    ],
)
def test_sensitivity_has_explicit_persistence_and_remote_sendability_result(
    git_repo: Path,
    sensitivity: Sensitivity,
    valid: bool,
    persistable: bool,
    remote_sendable: bool,
    retain_content: bool,
) -> None:
    sha = run_git(git_repo, "rev-parse", "HEAD")
    candidate = _candidate(
        git_repo,
        sha,
        GitCommitSourceRef(commit_sha=sha),
        sensitivity=sensitivity,
    )

    report = SourceValidator(git_repo).validate_candidate(
        candidate,
        runtime_provenance=_runtime_provenance(sha),
    )

    assert report.valid is valid
    assert report.persistable is persistable
    assert report.remote_sendable is remote_sendable
    assert report.retain_content is retain_content
    assert report.sensitivity.declared is sensitivity
    if sensitivity is Sensitivity.BLOCKED:
        assert report.sensitivity.effective is Sensitivity.BLOCKED
        assert SourceValidationCode.SENSITIVITY_BLOCKED in _issue_codes(report)


@pytest.mark.parametrize(
    "secret_text",
    [
        "api_key = 'sk-abcdefghijklmnopqrstuv'",
        "-----BEGIN PRIVATE KEY-----\nnot-a-real-key\n-----END PRIVATE KEY-----",
        '{"credential":"super-secret-value"}',
        "https://user:password@example.test/private",
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
        "token: x",
        "OPENAI_API_KEY=short-secret",
        "tool --token short-secret",
    ],
)
def test_sensitive_candidate_content_is_blocked_without_echoing_body(
    git_repo: Path,
    secret_text: str,
) -> None:
    sha = run_git(git_repo, "rev-parse", "HEAD")
    candidate = _candidate(
        git_repo,
        sha,
        GitCommitSourceRef(commit_sha=sha),
        statement=secret_text,
    )

    report = SourceValidator(git_repo).validate_candidate(
        candidate,
        runtime_provenance=_runtime_provenance(sha),
    )

    assert not report.valid
    assert report.sensitivity.effective is Sensitivity.BLOCKED
    assert not report.persistable
    assert not report.remote_sendable
    assert not report.retain_content
    assert SourceValidationCode.SENSITIVE_CONTENT in _issue_codes(report)
    serialized = report.to_json()
    assert secret_text not in serialized
    assert "super-secret-value" not in serialized
    assert "abcdefghijklmnopqrstuv" not in serialized


def test_sensitive_source_artifact_is_blocked_even_when_descriptor_hash_is_valid(
    git_repo: Path,
) -> None:
    secret = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    _, artifact_ref, _ = _create_session(
        git_repo,
        artifact_content=json.dumps({"token": secret}),
    )

    report = SourceValidator(git_repo).validate_sources(
        (artifact_ref,),
        sensitivity=Sensitivity.NORMAL,
        statement="Safe candidate statement.",
    )

    assert SourceValidationCode.SENSITIVE_CONTENT in _issue_codes(report)
    assert report.sensitivity.effective is Sensitivity.BLOCKED
    assert not report.persistable
    assert not report.remote_sendable
    assert secret not in report.to_json()


def test_secret_scan_is_schema_aware_for_env_names_budgets_and_code_refs() -> None:
    safe = json.dumps(
        {
            "reviewer_api_key_env": "OPENAI_API_KEY",
            "token_budget": 4096,
            "implementation": "token = get_token()",
        }
    )

    assert scan_sensitive_text(safe, schema="model_envelope_v1").safe
    assert not scan_sensitive_text(
        '{"token":"actual-value"}',
        schema="model_response_v1",
    ).safe


def test_model_candidate_requires_exact_runtime_source_allowlist(
    git_repo: Path,
) -> None:
    sha = run_git(git_repo, "rev-parse", "HEAD")
    source_ref = GitCommitSourceRef(commit_sha=sha)
    candidate = _candidate(
        git_repo,
        sha,
        source_ref,
        producer_type=ProducerType.MODEL,
    )

    denied = SourceValidator(git_repo).validate_candidate(
        candidate,
        runtime_provenance=_runtime_provenance(
            sha,
            origin=ProducerType.MODEL,
        ),
    )
    allowed = SourceValidator(git_repo).validate_candidate(
        candidate,
        runtime_provenance=_runtime_provenance(
            sha,
            origin=ProducerType.MODEL,
            allowed_source_refs=(source_ref,),
        ),
    )

    assert SourceValidationCode.SOURCE_NOT_ALLOWLISTED in _issue_codes(denied)
    assert allowed.valid


def test_candidate_validation_requires_executor_owned_provenance_before_hydration(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sha = run_git(git_repo, "rev-parse", "HEAD")
    candidate = _candidate(
        git_repo,
        sha,
        GitCommitSourceRef(commit_sha=sha),
        producer_type=ProducerType.LOCAL,
    )
    validator = SourceValidator(git_repo)
    hydrated = False

    def unexpected_hydration(source_ref: SourceRef) -> object:
        nonlocal hydrated
        hydrated = True
        raise AssertionError(source_ref)

    monkeypatch.setattr(validator, "_validate_one", unexpected_hydration)

    report = validator.validate_candidate(candidate)

    assert SourceValidationCode.RUNTIME_PROVENANCE_REQUIRED in _issue_codes(report)
    assert not hydrated


def test_candidate_uses_derived_repository_key_and_trusted_commit_lineage(
    git_repo: Path,
) -> None:
    head_sha = run_git(git_repo, "rev-parse", "HEAD")
    source_ref = GitCommitSourceRef(commit_sha=head_sha)
    valid = _candidate(git_repo, head_sha, source_ref)
    provenance = _runtime_provenance(head_sha)

    assert SourceValidator(git_repo).validate_candidate(
        valid,
        runtime_provenance=provenance,
    ).valid

    wrong_repository = replace(valid, repository_key="0" * 64)
    wrong_repository_report = SourceValidator(git_repo).validate_candidate(
        wrong_repository,
        runtime_provenance=provenance,
    )
    assert SourceValidationCode.REPOSITORY_MISMATCH in _issue_codes(
        wrong_repository_report
    )
    assert all(
        not result.valid for result in wrong_repository_report.source_results
    )

    tree_sha = run_git(git_repo, "rev-parse", "HEAD^{tree}")
    unrelated_sha = run_git(
        git_repo,
        "commit-tree",
        tree_sha,
        "-m",
        "unrelated source lineage",
    )
    unrelated = replace(valid, valid_from_sha=unrelated_sha)
    lineage_report = SourceValidator(git_repo).validate_candidate(
        unrelated,
        runtime_provenance=provenance,
    )
    assert SourceValidationCode.REVISION_LINEAGE_UNAUTHORIZED in _issue_codes(
        lineage_report
    )

    missing = replace(valid, valid_from_sha="f" * 40)
    missing_report = SourceValidator(git_repo).validate_candidate(
        missing,
        runtime_provenance=provenance,
    )
    assert SourceValidationCode.REVISION_NOT_FOUND in _issue_codes(missing_report)


def test_runtime_origin_not_candidate_producer_controls_model_allowlisting(
    git_repo: Path,
) -> None:
    sha = run_git(git_repo, "rev-parse", "HEAD")
    source_ref = GitCommitSourceRef(commit_sha=sha)
    self_declared_local = _candidate(
        git_repo,
        sha,
        source_ref,
        producer_type=ProducerType.LOCAL,
    )

    denied = SourceValidator(git_repo).validate_candidate(
        self_declared_local,
        runtime_provenance=_runtime_provenance(
            sha,
            origin=ProducerType.MODEL,
        ),
    )
    allowed = SourceValidator(git_repo).validate_candidate(
        self_declared_local,
        runtime_provenance=_runtime_provenance(
            sha,
            origin=ProducerType.MODEL,
            allowed_source_refs=(source_ref,),
        ),
    )

    assert SourceValidationCode.SOURCE_NOT_ALLOWLISTED in _issue_codes(denied)
    assert allowed.valid


def test_secret_scanner_rejects_duplicate_json_keys_and_raw_dotted_assignments() -> None:
    duplicate = scan_sensitive_text(
        '{"implementation":"safe","implementation":"also-safe"}',
        schema="model_response_v1",
    )
    dotted_assignment = scan_sensitive_text(
        json.dumps({"implementation": "token = request.token"}),
        schema="model_response_v1",
    )

    assert not duplicate.safe
    assert {finding.kind for finding in duplicate.findings} == {
        memory_sources_module.SensitiveContentKind.DUPLICATE_JSON_KEY
    }
    assert not dotted_assignment.safe
    assert memory_sources_module.SensitiveContentKind.CREDENTIAL_FIELD in {
        finding.kind for finding in dotted_assignment.findings
    }


def test_secret_scanner_keeps_raw_assignment_scanning_after_json_parse() -> None:
    scan = scan_sensitive_text(
        json.dumps(
            {
                "metadata": "ordinary",
                "implementation": "password = concrete.value",
            }
        ),
        schema="model_response_v1",
    )

    assert not scan.safe


def test_all_six_source_ref_variants_scan_every_persisted_string_before_hydration(
    git_repo: Path,
) -> None:
    sha = run_git(git_repo, "rev-parse", "HEAD")
    secret = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    declaration = _declaration()
    git_ref = GitCommitSourceRef(commit_sha=sha)
    object.__setattr__(git_ref, "commit_sha", secret)
    refs: tuple[SourceRef, ...] = (
        RepositoryRangeSourceRef(
            revision=sha,
            path=f"{secret}.py",
            line_start=1,
            line_end=1,
            content_hash="0" * 64,
        ),
        RepositorySymbolSourceRef(
            revision=sha,
            path="app.py",
            qualified_name=secret,
            hash_kind=SymbolHashKind.BODY,
            content_hash="0" * 64,
        ),
        git_ref,
        ObservationSourceRef(
            review_id=secret,
            observation_id="O-" + "1" * 32,
            revision_binding=f"head@{sha}",
            content_hash="0" * 64,
        ),
        SessionArtifactSourceRef(
            review_id=REVIEW_ID,
            artifact_name=secret,
            artifact_schema="review_brief_v1",
            revision_binding=_revision_binding(sha),
            artifact_hash="0" * 64,
        ),
        replace(declaration.to_source_ref(), actor=secret),
    )

    for source_ref in refs:
        report = SourceValidator(git_repo).validate_sources(
            (source_ref,),
            sensitivity=Sensitivity.NORMAL,
        )
        assert SourceValidationCode.SENSITIVE_CONTENT in _issue_codes(report)
        assert report.sensitivity.effective is Sensitivity.BLOCKED
        assert secret not in report.to_json()


def test_source_git_commands_strip_inherited_routing_and_object_environment(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sha = run_git(git_repo, "rev-parse", "HEAD")
    source_ref = RepositoryRangeSourceRef(
        revision=sha,
        path="app.py",
        line_start=1,
        line_end=1,
        content_hash=repository_range_hash(git_repo, sha, "app.py", 1, 1),
    )
    inherited = {
        "GIT_DIR": str(git_repo / "attacker.git"),
        "GIT_WORK_TREE": str(git_repo / "attacker-worktree"),
        "GIT_COMMON_DIR": str(git_repo / "attacker-common"),
        "GIT_OBJECT_DIRECTORY": str(git_repo / "attacker-objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(git_repo / "alternate-objects"),
        "GIT_NAMESPACE": "attacker",
        "GIT_REPLACE_REF_BASE": "refs/attacker/",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.repositoryformatversion",
        "GIT_CONFIG_VALUE_0": "999",
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)

    calls: list[tuple[list[str], dict[str, str]]] = []
    real_run = subprocess.run

    def recording_run(
        command: list[str],
        *args: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[object]:
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        calls.append((command, environment))
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(memory_sources_module.subprocess, "run", recording_run)

    report = SourceValidator(git_repo).validate_sources(
        (source_ref,),
        sensitivity=Sensitivity.NORMAL,
    )

    assert report.valid
    assert calls
    for command, environment in calls:
        assert command[:2] == ["git", "--no-replace-objects"]
        assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
        assert environment["GIT_LITERAL_PATHSPECS"] == "1"
        for key in inherited:
            assert key not in environment


def test_exact_revision_sources_ignore_git_replacement_refs(git_repo: Path) -> None:
    original_sha = run_git(git_repo, "rev-parse", "HEAD")
    original_hash = _sha256("def add(a, b):\n")
    (git_repo / "app.py").write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "replacement content")
    replacement_sha = run_git(git_repo, "rev-parse", "HEAD")
    run_git(git_repo, "replace", original_sha, replacement_sha)
    source_ref = RepositoryRangeSourceRef(
        revision=original_sha,
        path="app.py",
        line_start=1,
        line_end=1,
        content_hash=original_hash,
    )

    report = SourceValidator(git_repo).validate_sources(
        (source_ref,),
        sensitivity=Sensitivity.NORMAL,
    )

    assert report.valid


def test_session_sources_require_completed_checkpoint_and_safe_schema_allowlist(
    git_repo: Path,
) -> None:
    sha = run_git(git_repo, "rev-parse", "HEAD")
    store, incomplete_ref, _ = _create_session(git_repo, complete=False)

    incomplete = SourceValidator(git_repo).validate_sources(
        (incomplete_ref,),
        sensitivity=Sensitivity.NORMAL,
    )
    assert SourceValidationCode.SESSION_ARTIFACT_INVALID in _issue_codes(incomplete)

    raw_path = store.run_dir / "reviewer-raw.json"
    raw_path.write_text('{"output":"untrusted model text"}', encoding="utf-8")
    manifest = store.register_existing_artifact(
        name="reviewer_raw_response",
        relative_path="reviewer-raw.json",
        schema="model_raw_response_v1",
        phase=RunPhase.REPORTING,
        revision_binding=_revision_binding(sha),
        now="2026-07-14T00:00:03Z",
    )
    descriptor = manifest.artifacts["reviewer_raw_response"]
    _complete_session(
        store,
        ["review_brief", "observations", "reviewer_raw_response"],
    )
    raw_ref = SessionArtifactSourceRef(
        review_id=REVIEW_ID,
        artifact_name=descriptor.name,
        artifact_schema=descriptor.schema,
        revision_binding=_revision_binding(sha),
        artifact_hash=descriptor.sha256,
    )

    raw_report = SourceValidator(git_repo).validate_sources(
        (raw_ref,),
        sensitivity=Sensitivity.NORMAL,
    )
    assert SourceValidationCode.SESSION_ARTIFACT_INELIGIBLE in _issue_codes(
        raw_report
    )


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "safe.py:secret",
        "CON.py",
        "aux.txt",
        "trailing.",
        "PROGRA~1/source.py",
    ],
)
def test_repository_sources_reject_cross_platform_path_aliases(
    git_repo: Path,
    unsafe_path: str,
) -> None:
    sha = run_git(git_repo, "rev-parse", "HEAD")
    source_ref = RepositoryRangeSourceRef(
        revision=sha,
        path=unsafe_path,
        line_start=1,
        line_end=1,
        content_hash="0" * 64,
    )

    report = SourceValidator(git_repo).validate_sources(
        (source_ref,),
        sensitivity=Sensitivity.NORMAL,
    )

    assert SourceValidationCode.UNSAFE_PATH in _issue_codes(report)

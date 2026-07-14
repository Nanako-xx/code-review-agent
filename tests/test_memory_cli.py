from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from conftest import run_git
import review_agent.command as command_module
from review_agent.command import _build_parser, main
from review_agent.memory_identity import (
    MemoryRootResolver,
    build_repository_memory_namespace,
    repository_namespace_path,
)
from review_agent.memory_lifecycle import (
    MemoryLifecycle,
    MemoryLifecycleError,
    MemoryLifecycleErrorCode,
)
from review_agent.memory_models import (
    CandidateStatus,
    MemoryCandidate,
    MemoryConfidence,
    MemoryKind,
    MemoryScope,
    PolicyEffect,
    PolicyEffectKind,
    Producer,
    ProducerType,
    RecordStatus,
    RepositoryRangeSourceRef,
    Sensitivity,
    ValidityPolicy,
    stable_request_id,
)
from review_agent.memory_sources import (
    SourceValidator,
    TrustedCandidateProvenance,
    candidate_authority_resolution_hash,
    repository_range_hash,
)
from review_agent.memory_relink import RepositoryRelinkRegistry
from review_agent.memory_store import (
    MemoryStore,
    MemoryStoreValidationError,
)
from review_agent.revision import RevisionResolver


NOW = "2026-07-14T08:00:00Z"


def _namespace(repo: Path, memory_root: Path):
    resolution = MemoryRootResolver().resolve(memory_root.resolve(), create=True)
    identity = RevisionResolver().repository_identity(repo)
    return build_repository_memory_namespace(identity, resolution)


def _store(repo: Path, memory_root: Path) -> MemoryStore:
    return MemoryStore(_namespace(repo, memory_root))


def _head(repo: Path) -> str:
    return run_git(repo, "rev-parse", "HEAD")


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
    with pytest.raises(SystemExit):
        parser.parse_args(["memory", "feedback"])
    with pytest.raises(SystemExit):
        parser.parse_args(["memory", "replay-outbox"])


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

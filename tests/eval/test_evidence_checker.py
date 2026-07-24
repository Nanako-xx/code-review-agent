from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
import hashlib
import inspect
from pathlib import Path
import shutil
import stat
from typing import Iterator

import pytest

from review_agent_eval.artifacts import TrialManifest
from review_agent_eval.cases import (
    REPOSITORY_MATERIALIZER_PROTOCOL,
    CaseSplit,
    SuiteCase,
    WireContractV2,
)
from review_agent_eval.config import (
    derive_case_path_id,
    derive_trial_id,
    derive_trial_seed,
)
from review_agent_eval.evidence_checker import (
    COMMAND_OUTPUT_ATTESTATION_SCHEMA_VERSION,
    MAX_REPLAY_LINES,
    CommandOutputAttestation,
    EvidenceDiagnostic,
    EvidenceIntegrityChecker,
    EvidenceIntegrityResult,
    EvidenceItemIntegrityResult,
    EvidenceReasonCode,
)
from review_agent_eval.models import (
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    DiffSide,
    CommandOutputEvidenceSource,
    EvalInput,
    EvidenceIntegrity,
    EvidenceKind,
    EvidenceStream,
    ExistingCIEvidence,
    ExternalRecordEvidenceSource,
    FindingSeverity,
    MAX_EVIDENCE_EXCERPT_BYTES,
    Repository,
    RepositoryDiffEvidenceSource,
    RepositoryFileEvidenceSource,
    RepositoryReviewTarget,
    ReviewRequest,
    ReviewTargetKind,
    SchemaError,
    SubmissionEvidence,
    SubmissionFinding,
    TrialStatus,
    TruthCompleteness,
    stable_id,
)
from review_agent_eval.repository import (
    FixtureRepositoryBuilder,
    PreparedRepository,
    PreparedRepositoryReplay,
    RepositoryIntegrityError,
    RepositoryPreparer,
    RepositorySecurityError,
    repository_from_eval_input,
)


TRIAL_ID = "trial-evidence-a"
TARGET_MATERIALIZATION_ID = "materialization-" + ("1" * 64)


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_text(value: str) -> str:
    return _digest_bytes(value.encode("utf-8"))


def _git_executable() -> Path:
    executable = shutil.which("git")
    assert executable is not None
    return Path(executable).absolute()


def _write_tree(root: Path, files: dict[str, bytes]) -> None:
    for relative, content in files.items():
        target = root / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _write_fixture(suite_root: Path, *, heavy: bool) -> Path:
    fixture = suite_root / "repositories" / "evidence"
    base = fixture / "base"
    head = fixture / "head"
    base.mkdir(parents=True)
    head.mkdir(parents=True)
    common = {
        "Case.py": b"case sensitive\n",
        "boundaries.txt": (
            b"a\vb\fc\x1cd\x1de\x1ef\xc2\x85g\xe2\x80\xa8h\xe2\x80\xa9last"
        ),
        "invalid.bin": b"valid-prefix\xffinvalid\n",
    }
    _write_tree(
        base,
        {
            **common,
            "src/app.py": b"alpha\r\nbase\rlast",
            "src/a[0].py": b"value = 0\n",
            "deleted.txt": b"deleted at head\n",
        },
    )
    _write_tree(
        head,
        {
            **common,
            "src/app.py": b"alpha\nhead\nlast\n",
            "src/a[0].py": b"value = 1\n",
            "added.txt": b"added at head\n",
        },
    )
    if heavy:
        oversized = b"x" * (MAX_EVIDENCE_EXCERPT_BYTES + 1)
        too_many_lines = b"\n" * (MAX_REPLAY_LINES + 1)
        for tree in (base, head):
            _write_tree(
                tree,
                {
                    "oversized.txt": oversized,
                    "many-lines.txt": too_many_lines,
                },
            )
    return fixture


@dataclass(frozen=True)
class Harness:
    preparer: RepositoryPreparer
    prepared: PreparedRepository
    replay: PreparedRepositoryReplay
    eval_input: EvalInput
    checker: EvidenceIntegrityChecker


def _build_harness(root: Path, *, heavy: bool = True) -> Harness:
    suite = root / "suite"
    suite.mkdir()
    fixture = _write_fixture(suite, heavy=heavy)
    built = FixtureRepositoryBuilder().build(fixture, root / "authored.git")
    descriptor = built.to_repository("repositories/evidence")
    ci_text = "linux: passed\nwindows: passed\n"
    eval_input = EvalInput(
        schema_version=EVAL_INPUT_SCHEMA_VERSION,
        task_id="case-evidence",
        review_target=RepositoryReviewTarget(
            kind=ReviewTargetKind.REPOSITORY,
            repository=descriptor,
            review_request=ReviewRequest(
                title="Review evidence integrity",
                description=None,
                user_intent=None,
                review_focus=None,
                linked_requirements=(),
                project_rules=(),
                existing_ci_evidence=(
                    ExistingCIEvidence(
                        source_id="ci-main",
                        text=ci_text,
                        content_hash=_digest_text(ci_text),
                    ),
                ),
            ),
        ),
    )
    preparer = RepositoryPreparer(
        suite_root=suite,
        data_root=root / ".eval-data",
        workspace_root=root / ".eval-workspaces",
        git_executable=_git_executable(),
    )
    preparer.__enter__()
    try:
        prepared = preparer.prepare(descriptor)
        replay = preparer.open_replay(prepared)
        checker = EvidenceIntegrityChecker(
            eval_input, replay, TRIAL_ID, TARGET_MATERIALIZATION_ID
        )
    except BaseException:
        preparer.__exit__(None, None, None)
        raise
    return Harness(preparer, prepared, replay, eval_input, checker)


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Harness]:
    value = _build_harness(tmp_path_factory.mktemp("evidence-checker"))
    try:
        yield value
    finally:
        value.preparer.__exit__(None, None, None)


def _reasons(result: object) -> set[EvidenceReasonCode]:
    return {item.reason_code for item in result.diagnostics}  # type: ignore[attr-defined]


def _file_evidence(
    harness: Harness,
    *,
    evidence_id: str = "ev-file",
    revision: str | None = None,
    path: str = "src/app.py",
    from_line: int = 1,
    to_line: int = 1,
    excerpt: str = "alpha\n",
    content_hash: str | None = None,
) -> SubmissionEvidence:
    repository = repository_from_eval_input(harness.eval_input)
    return SubmissionEvidence(
        evidence_id=evidence_id,
        source=RepositoryFileEvidenceSource(
            kind=EvidenceKind.REPOSITORY_FILE,
            target_materialization_id=TARGET_MATERIALIZATION_ID,
            revision=revision or repository.head_revision,
            path=path,
            from_line=from_line,
            to_line=to_line,
        ),
        content_hash=content_hash or _digest_text(excerpt),
        excerpt=excerpt,
    )


def _diff_evidence(
    harness: Harness,
    *,
    evidence_id: str = "ev-diff",
    path: str = "src/app.py",
    base_revision: str | None = None,
    head_revision: str | None = None,
    excerpt: str | None = None,
    content_hash: str | None = None,
) -> SubmissionEvidence:
    if excerpt is None:
        raw = harness.replay.diff(path, max_bytes=MAX_EVIDENCE_EXCERPT_BYTES)
        excerpt = raw.decode("utf-8")
    repository = repository_from_eval_input(harness.eval_input)
    return SubmissionEvidence(
        evidence_id=evidence_id,
        source=RepositoryDiffEvidenceSource(
            kind=EvidenceKind.REPOSITORY_DIFF,
            target_materialization_id=TARGET_MATERIALIZATION_ID,
            base_revision=base_revision or repository.base_revision,
            head_revision=head_revision or repository.head_revision,
            path=path,
        ),
        content_hash=content_hash or _digest_text(excerpt),
        excerpt=excerpt,
    )


def _attestation(
    harness: Harness,
    *,
    source_ref: str = "command-main",
    trial_id: str = TRIAL_ID,
    head_revision: str | None = None,
    argv: tuple[str, ...] = ("pytest", "-q"),
    exit_code: int = 0,
    stream: EvidenceStream = EvidenceStream.COMBINED,
    output: bytes = b"2 passed\n",
    byte_size: int | None = None,
    sha256: str | None = None,
) -> CommandOutputAttestation:
    return CommandOutputAttestation(
        schema_version=COMMAND_OUTPUT_ATTESTATION_SCHEMA_VERSION,
        source_ref=source_ref,
        trial_id=trial_id,
        head_revision=head_revision
        or repository_from_eval_input(harness.eval_input).head_revision,
        argv=argv,
        exit_code=exit_code,
        stream=stream,
        output_bytes=output,
        byte_size=len(output) if byte_size is None else byte_size,
        sha256=_digest_bytes(output) if sha256 is None else sha256,
    )


def _command_evidence(
    harness: Harness,
    *,
    evidence_id: str = "ev-command",
    command: tuple[str, ...] = ("pytest", "-q"),
    exit_code: int = 0,
    stream: EvidenceStream = EvidenceStream.COMBINED,
    source_ref: str = "command-main",
    excerpt: str = "2 passed\n",
    content_hash: str | None = None,
) -> SubmissionEvidence:
    return SubmissionEvidence(
        evidence_id=evidence_id,
        source=CommandOutputEvidenceSource(
            kind=EvidenceKind.COMMAND_OUTPUT,
            target_materialization_id=TARGET_MATERIALIZATION_ID,
            command=command,
            exit_code=exit_code,
            stream=stream,
            artifact_ref=source_ref,
        ),
        content_hash=content_hash or _digest_text(excerpt),
        excerpt=excerpt,
    )


def _external_evidence(
    harness: Harness,
    *,
    evidence_id: str = "ev-external",
    source_ref: str = "ci-main",
    excerpt: str = "linux: passed\nwindows: passed\n",
    content_hash: str | None = None,
) -> SubmissionEvidence:
    return SubmissionEvidence(
        evidence_id=evidence_id,
        source=ExternalRecordEvidenceSource(
            kind=EvidenceKind.EXTERNAL_RECORD,
            target_materialization_id=TARGET_MATERIALIZATION_ID,
            source_ref=source_ref,
        ),
        content_hash=content_hash or _digest_text(excerpt),
        excerpt=excerpt,
    )


def _finding(*refs: str, finding_id: str = "finding-a") -> SubmissionFinding:
    return SubmissionFinding(
        finding_id=finding_id,
        claim="Material issue",
        severity=FindingSeverity.HIGH,
        path=None,
        side=None,
        from_line=None,
        to_line=None,
        evidence_refs=tuple(refs),
        suggested_action=None,
    )


@pytest.mark.parametrize(
    ("revision_name", "from_line", "to_line", "excerpt"),
    [
        ("base", 1, 2, "alpha\nbase\n"),
        ("base", 3, 3, "last"),
        ("head", 1, 2, "alpha\nhead\n"),
        ("head", 3, 3, "last\n"),
    ],
)
def test_repository_file_replays_base_head_crlf_and_final_newline_exactly(
    harness: Harness,
    revision_name: str,
    from_line: int,
    to_line: int,
    excerpt: str,
) -> None:
    revision = getattr(
        repository_from_eval_input(harness.eval_input),
        "%s_revision" % revision_name,
    )
    evidence = _file_evidence(
        harness,
        revision=revision,
        from_line=from_line,
        to_line=to_line,
        excerpt=excerpt,
    )

    result = harness.checker.check_item(evidence)

    assert result.integrity is EvidenceIntegrity.VALID
    assert result.diagnostics == ()


def test_repository_file_canonicalizes_every_splitlines_boundary_to_lf(
    harness: Harness,
) -> None:
    excerpt = "a\nb\nc\nd\ne\nf\ng\nh\nlast"
    evidence = _file_evidence(
        harness,
        path="boundaries.txt",
        from_line=1,
        to_line=9,
        excerpt=excerpt,
        content_hash=_digest_text(excerpt),
    )

    result = harness.checker.check_item(evidence)

    assert result.integrity is EvidenceIntegrity.VALID


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"revision": "HEAD"}, EvidenceReasonCode.REVISION_MISMATCH),
        ({"path": "/src/app.py"}, EvidenceReasonCode.PATH_INVALID),
        ({"path": "../src/app.py"}, EvidenceReasonCode.PATH_INVALID),
        ({"path": "src\\app.py"}, EvidenceReasonCode.PATH_INVALID),
        ({"path": "a" * 256}, EvidenceReasonCode.PATH_INVALID),
        ({"path": "src/CON.py"}, EvidenceReasonCode.PATH_INVALID),
        ({"path": "src/cafe\u0301.py"}, EvidenceReasonCode.PATH_INVALID),
        ({"path": "case.py"}, EvidenceReasonCode.PATH_NOT_FOUND),
        ({"path": "missing.py"}, EvidenceReasonCode.PATH_NOT_FOUND),
        ({"from_line": 2, "to_line": 1}, EvidenceReasonCode.LINE_RANGE_REVERSED),
        ({"from_line": 4, "to_line": 4}, EvidenceReasonCode.LINE_RANGE_OUT_OF_BOUNDS),
    ],
)
def test_repository_file_rejects_symbolic_unsafe_case_wrong_and_bad_coordinates(
    harness: Harness,
    changes: dict[str, object],
    reason: EvidenceReasonCode,
) -> None:
    evidence = _file_evidence(harness, **changes)  # type: ignore[arg-type]

    result = harness.checker.check_item(evidence)

    assert result.integrity is EvidenceIntegrity.INVALID
    assert reason in _reasons(result)


@pytest.mark.parametrize(
    ("excerpt", "content_hash", "reasons"),
    [
        ("wrong\n", _digest_text("alpha\n"), {EvidenceReasonCode.EXCERPT_MISMATCH}),
        ("alpha\n", "0" * 64, {EvidenceReasonCode.CONTENT_HASH_MISMATCH}),
        (
            "wrong\n",
            _digest_text("wrong\n"),
            {
                EvidenceReasonCode.EXCERPT_MISMATCH,
                EvidenceReasonCode.CONTENT_HASH_MISMATCH,
            },
        ),
    ],
)
def test_repository_file_rejects_excerpt_and_hash_mismatches_independently(
    harness: Harness,
    excerpt: str,
    content_hash: str,
    reasons: set[EvidenceReasonCode],
) -> None:
    result = harness.checker.check_item(
        _file_evidence(harness, excerpt=excerpt, content_hash=content_hash)
    )

    assert result.integrity is EvidenceIntegrity.INVALID
    assert _reasons(result) == reasons


def test_repository_file_rejects_invalid_utf8(harness: Harness) -> None:
    evidence = _file_evidence(
        harness,
        path="invalid.bin",
        excerpt="",
        content_hash=_digest_bytes(b""),
    )

    result = harness.checker.check_item(evidence)

    assert result.integrity is EvidenceIntegrity.INVALID
    assert _reasons(result) == {EvidenceReasonCode.CONTENT_NOT_UTF8}


def test_repository_file_rejects_oversized_canonical_excerpt(harness: Harness) -> None:
    evidence = _file_evidence(
        harness,
        path="oversized.txt",
        excerpt="",
        content_hash=_digest_bytes(b""),
    )

    result = harness.checker.check_item(evidence)

    assert result.integrity is EvidenceIntegrity.INVALID
    assert _reasons(result) == {EvidenceReasonCode.EXCERPT_TOO_LARGE}


def test_repository_file_enforces_bounded_logical_line_replay(harness: Harness) -> None:
    evidence = _file_evidence(
        harness,
        path="many-lines.txt",
        excerpt="\n",
        content_hash=_digest_text("\n"),
    )

    result = harness.checker.check_item(evidence)

    assert result.integrity is EvidenceIntegrity.INVALID
    assert _reasons(result) == {EvidenceReasonCode.REPLAY_LINE_LIMIT_EXCEEDED}


def test_repository_diff_requires_the_complete_exact_replay(harness: Harness) -> None:
    evidence = _diff_evidence(harness)

    result = harness.checker.check_item(evidence)

    assert result.integrity is EvidenceIntegrity.VALID
    assert evidence.excerpt.encode("utf-8") == harness.replay.diff(
        "src/app.py", max_bytes=MAX_EVIDENCE_EXCERPT_BYTES
    )


def test_repository_diff_treats_metacharacter_path_as_literal(
    harness: Harness,
) -> None:
    path = "src/a[0].py"
    raw = harness.replay.diff(path, max_bytes=MAX_EVIDENCE_EXCERPT_BYTES)
    assert raw
    evidence = _diff_evidence(
        harness,
        path=path,
        excerpt=raw.decode("utf-8", "strict"),
        content_hash=_digest_bytes(raw),
    )

    result = harness.checker.check_item(evidence)

    assert result.integrity is EvidenceIntegrity.VALID


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"head_revision": "HEAD"}, EvidenceReasonCode.REVISION_MISMATCH),
        ({"base_revision": "c" * 40}, EvidenceReasonCode.REVISION_MISMATCH),
        ({"path": "missing.py", "excerpt": ""}, EvidenceReasonCode.PATH_NOT_FOUND),
        ({"path": "SRC/app.py", "excerpt": ""}, EvidenceReasonCode.PATH_NOT_FOUND),
        ({"path": "../src/app.py", "excerpt": ""}, EvidenceReasonCode.PATH_INVALID),
    ],
)
def test_repository_diff_rejects_wrong_range_path_and_extra_fields(
    harness: Harness,
    changes: dict[str, object],
    reason: EvidenceReasonCode,
) -> None:
    evidence = _diff_evidence(harness, **changes)  # type: ignore[arg-type]

    result = harness.checker.check_item(evidence)

    assert result.integrity is EvidenceIntegrity.INVALID
    assert reason in _reasons(result)


def _workspace_binding(harness: Harness) -> tuple[TrialManifest, SuiteCase]:
    run_id = stable_id("run", "evidence-workspace-test")
    trial_id = derive_trial_id(run_id, harness.eval_input.task_id, 1)
    wire_contract = WireContractV2(
        case_schema_version=EVAL_CASE_SCHEMA_VERSION,
        input_schema_version=EVAL_INPUT_SCHEMA_VERSION,
        submission_schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
        review_target_kind=ReviewTargetKind.REPOSITORY,
        materializer_protocol=REPOSITORY_MATERIALIZER_PROTOCOL,
    )
    suite_case = SuiteCase(
        task_id=harness.eval_input.task_id,
        case_version=1,
        path="cases/evidence.json",
        split=CaseSplit.REGRESSION,
        protocol_id="core-code-review-v1",
        dimensions=(),
        raw_file_size_bytes=1,
        raw_file_sha256="1" * 64,
        canonical_case_digest="c" * 64,
        eval_input_digest=harness.eval_input.digest(),
        truth_completeness=TruthCompleteness.CLOSED_WORLD,
    )
    manifest = TrialManifest(
        schema_version=TrialManifest.SCHEMA_VERSION,
        run_id=run_id,
        task_id=harness.eval_input.task_id,
        case_path_id=derive_case_path_id(harness.eval_input.task_id),
        canonical_case_digest=suite_case.canonical_case_digest,
        eval_input_digest=harness.eval_input.digest(),
        wire_contract=wire_contract,
        target_kind=ReviewTargetKind.REPOSITORY,
        materializer_protocol=REPOSITORY_MATERIALIZER_PROTOCOL,
        suite_preparation_binding_digest=None,
        adapter_capabilities_digest="d" * 64,
        trial_id=trial_id,
        trial_index=1,
        seed=derive_trial_seed(run_id, harness.eval_input.task_id, 1),
        agent_config_digest="a" * 64,
        initial_evaluator_execution_digest="b" * 64,
    )
    return manifest, suite_case


def test_repository_diff_does_not_accept_uncommitted_workspace_content(
    harness: Harness,
) -> None:
    canonical = _diff_evidence(harness)
    manifest, suite_case = _workspace_binding(harness)
    with harness.preparer.trial_workspace(
        harness.prepared,
        trial_manifest=manifest,
        suite_case=suite_case,
        eval_input=harness.eval_input,
        attempt=1,
    ) as workspace:
        (workspace.path / "src" / "app.py").write_text(
            "workspace-only change\n", encoding="utf-8"
        )
        workspace.record_terminal_status(TrialStatus.COMPLETED)
        claimed = replace(
            canonical,
            excerpt="workspace-only change\n",
            content_hash=_digest_text("workspace-only change\n"),
        )
        result = harness.checker.check_item(claimed)

    assert result.integrity is EvidenceIntegrity.INVALID
    assert {
        EvidenceReasonCode.EXCERPT_MISMATCH,
        EvidenceReasonCode.CONTENT_HASH_MISMATCH,
    }.issubset(_reasons(result))


def test_command_attestation_is_frozen_hash_bound_and_strictly_round_trips(
    harness: Harness,
) -> None:
    attestation = _attestation(harness)

    assert CommandOutputAttestation.from_dict(attestation.to_dict()) == attestation
    assert CommandOutputAttestation.from_json(attestation.to_json()) == attestation
    assert attestation.output is attestation.output_bytes
    with pytest.raises(FrozenInstanceError):
        attestation.exit_code = 1  # type: ignore[misc]
    with pytest.raises(SchemaError, match="byte_size"):
        _attestation(harness, byte_size=1)
    with pytest.raises(SchemaError, match="sha256"):
        _attestation(harness, sha256="0" * 64)
    with pytest.raises(SchemaError, match="bool"):
        _attestation(harness, exit_code=True)  # type: ignore[arg-type]

    duplicate_key_json = attestation.to_json().replace(
        '"schema_version":', '"schema_version":"duplicate","schema_version":', 1
    )
    with pytest.raises(SchemaError, match="duplicate"):
        CommandOutputAttestation.from_json(duplicate_key_json)


def test_command_output_accepts_only_complete_attested_bytes(harness: Harness) -> None:
    attestation = _attestation(harness)
    checker = EvidenceIntegrityChecker(
        harness.eval_input,
        harness.replay,
        TRIAL_ID,
        TARGET_MATERIALIZATION_ID,
        (attestation,),
    )

    result = checker.check_item(_command_evidence(harness))

    assert result.integrity is EvidenceIntegrity.VALID


@pytest.mark.parametrize(
    ("attestation_changes", "evidence_changes", "reason"),
    [
        ({"trial_id": "other-trial"}, {}, EvidenceReasonCode.ATTESTATION_TRIAL_MISMATCH),
        (
            {"head_revision": "USE_BASE"},
            {},
            EvidenceReasonCode.ATTESTATION_REVISION_MISMATCH,
        ),
        ({}, {"command": ("pytest",)}, EvidenceReasonCode.COMMAND_MISMATCH),
        ({}, {"exit_code": 1}, EvidenceReasonCode.EXIT_CODE_MISMATCH),
        ({}, {"stream": EvidenceStream.STDOUT}, EvidenceReasonCode.STREAM_MISMATCH),
        ({}, {"content_hash": "0" * 64}, EvidenceReasonCode.CONTENT_HASH_MISMATCH),
        (
            {},
            {"excerpt": "2 passed", "content_hash": _digest_text("2 passed")},
            EvidenceReasonCode.EXCERPT_MISMATCH,
        ),
    ],
)
def test_command_output_rejects_wrong_binding_metadata_hash_and_truncation(
    harness: Harness,
    attestation_changes: dict[str, object],
    evidence_changes: dict[str, object],
    reason: EvidenceReasonCode,
) -> None:
    changes = dict(attestation_changes)
    if changes.get("head_revision") == "USE_BASE":
        changes["head_revision"] = repository_from_eval_input(
            harness.eval_input
        ).base_revision
    attestation = _attestation(harness, **changes)  # type: ignore[arg-type]
    checker = EvidenceIntegrityChecker(
        harness.eval_input,
        harness.replay,
        TRIAL_ID,
        TARGET_MATERIALIZATION_ID,
        (attestation,),
    )
    evidence = _command_evidence(harness, **evidence_changes)  # type: ignore[arg-type]

    result = checker.check_item(evidence)

    assert result.integrity is EvidenceIntegrity.INVALID
    assert reason in _reasons(result)


@pytest.mark.parametrize(
    ("attestations", "source_ref", "reason"),
    [
        ((), "command-main", EvidenceReasonCode.ATTESTATION_NOT_FOUND),
    ],
)
def test_command_output_rejects_absent_source_or_attestation(
    harness: Harness,
    attestations: tuple[CommandOutputAttestation, ...],
    source_ref: str,
    reason: EvidenceReasonCode,
) -> None:
    checker = EvidenceIntegrityChecker(
        harness.eval_input,
        harness.replay,
        TRIAL_ID,
        TARGET_MATERIALIZATION_ID,
        attestations,
    )

    result = checker.check_item(
        _command_evidence(harness, source_ref=source_ref)
    )

    assert result.integrity is EvidenceIntegrity.INVALID
    assert reason in _reasons(result)


def test_command_output_rejects_ambiguous_bound_attestations(harness: Harness) -> None:
    first = _attestation(harness)
    second = _attestation(harness, output=b"different\n")
    checker = EvidenceIntegrityChecker(
        harness.eval_input,
        harness.replay,
        TRIAL_ID,
        TARGET_MATERIALIZATION_ID,
        (second, first),
    )

    result = checker.check_item(_command_evidence(harness))

    assert result.integrity is EvidenceIntegrity.INVALID
    assert _reasons(result) == {EvidenceReasonCode.ATTESTATION_AMBIGUOUS}


def test_external_record_accepts_exact_agent_visible_ci(harness: Harness) -> None:
    result = harness.checker.check_item(_external_evidence(harness))

    assert result.integrity is EvidenceIntegrity.VALID


@pytest.mark.parametrize(
    ("changes", "reasons"),
    [
        (
            {"source_ref": "missing-ci"},
            {EvidenceReasonCode.EXTERNAL_RECORD_NOT_FOUND},
        ),
        (
            {"excerpt": "linux: failed\n", "content_hash": "0" * 64},
            {
                EvidenceReasonCode.EXCERPT_MISMATCH,
                EvidenceReasonCode.CONTENT_HASH_MISMATCH,
            },
        ),
    ],
)
def test_external_record_rejects_missing_or_mismatched_ci(
    harness: Harness,
    changes: dict[str, object],
    reasons: set[EvidenceReasonCode],
) -> None:
    result = harness.checker.check_item(
        _external_evidence(harness, **changes)  # type: ignore[arg-type]
    )

    assert result.integrity is EvidenceIntegrity.INVALID
    assert _reasons(result) == reasons


def test_finding_without_refs_is_missing(harness: Harness) -> None:
    result = harness.checker.check_finding(_finding(), ())

    assert result.integrity is EvidenceIntegrity.MISSING
    assert _reasons(result) == {EvidenceReasonCode.NO_EVIDENCE_REFS}


def test_dangling_ref_has_missing_precedence_over_invalid_item(
    harness: Harness,
) -> None:
    invalid = _file_evidence(harness, content_hash="0" * 64)
    result = harness.checker.check_finding(
        _finding(invalid.evidence_id, "dangling"), (invalid,)
    )

    assert result.integrity is EvidenceIntegrity.MISSING
    assert EvidenceReasonCode.DANGLING_REF in _reasons(result)
    assert EvidenceReasonCode.CONTENT_HASH_MISMATCH in _reasons(result)


def test_duplicate_refs_remain_auditable_without_downgrading_valid_item(
    harness: Harness,
) -> None:
    evidence = _file_evidence(harness)
    result = harness.checker.check_finding(
        _finding(evidence.evidence_id, evidence.evidence_id), (evidence,)
    )

    assert result.integrity is EvidenceIntegrity.VALID
    assert result.referenced_evidence_ids == (evidence.evidence_id,) * 2
    assert len(result.item_results) == 2
    duplicate = [
        item
        for item in result.diagnostics
        if item.reason_code is EvidenceReasonCode.DUPLICATE_REF
    ]
    assert len(duplicate) == 1
    assert duplicate[0].ref_index == 1


def test_integrity_receipts_strictly_round_trip_and_preserve_ref_order(
    harness: Harness,
) -> None:
    valid = _file_evidence(harness, evidence_id="ev-z")
    invalid = _file_evidence(
        harness,
        evidence_id="ev-a",
        content_hash="0" * 64,
    )
    finding = _finding(
        valid.evidence_id,
        "ev-missing",
        valid.evidence_id,
        invalid.evidence_id,
        finding_id="finding-round-trip",
    )
    evidence = (invalid, valid)
    result = harness.checker.check_finding(finding, evidence)

    assert result.integrity is EvidenceIntegrity.MISSING
    assert result.referenced_evidence_ids == finding.evidence_refs
    assert tuple(item.evidence_id for item in result.item_results) == (
        "ev-z",
        "ev-z",
        "ev-a",
    )
    assert tuple(
        (item.reason_code, item.ref_index) for item in result.diagnostics
    ) == (
        (EvidenceReasonCode.DANGLING_REF, 1),
        (EvidenceReasonCode.DUPLICATE_REF, 2),
        (EvidenceReasonCode.CONTENT_HASH_MISMATCH, 3),
    )

    diagnostic = result.diagnostics[0]
    assert EvidenceDiagnostic.from_dict(diagnostic.to_dict()) == diagnostic
    assert EvidenceDiagnostic.from_json(diagnostic.to_json()) == diagnostic
    assert EvidenceDiagnostic.from_json(diagnostic.to_json().encode("utf-8")) == diagnostic

    item_result = result.item_results[-1]
    assert EvidenceItemIntegrityResult.from_dict(item_result.to_dict()) == item_result
    assert EvidenceItemIntegrityResult.from_json(item_result.to_json()) == item_result

    assert EvidenceIntegrityResult.from_dict(result.to_dict()) == result
    hydrated = EvidenceIntegrityResult.from_json(
        result.to_json(),
        finding=finding,
        evidence_items=evidence,
        checker=harness.checker,
    )
    assert hydrated == result
    assert hydrated.digest() == result.digest()


def test_diagnostic_hydration_rejects_unknown_duplicate_and_illegal_fields(
    harness: Harness,
) -> None:
    result = harness.checker.check_finding(
        _finding("ev-missing", finding_id="finding-diagnostic-tamper"),
        (),
    )
    diagnostic = result.diagnostics[0]

    unknown = diagnostic.to_dict()
    unknown["extra"] = None
    with pytest.raises(SchemaError, match="unknown or missing"):
        EvidenceDiagnostic.from_dict(unknown)

    illegal_enum = diagnostic.to_dict()
    illegal_enum["reason_code"] = "invented"
    with pytest.raises(SchemaError, match="unknown enum"):
        EvidenceDiagnostic.from_dict(illegal_enum)

    illegal_id = diagnostic.to_dict()
    illegal_id["evidence_id"] = "bad id"
    with pytest.raises(SchemaError, match="whitespace"):
        EvidenceDiagnostic.from_dict(illegal_id)

    duplicate_key_json = diagnostic.to_json().replace(
        '"reason_code":',
        '"reason_code":"dangling_ref","reason_code":',
        1,
    )
    with pytest.raises(SchemaError, match="duplicate"):
        EvidenceDiagnostic.from_json(duplicate_key_json)


def test_item_result_hydration_rejects_forged_state_scope_and_order(
    harness: Harness,
) -> None:
    evidence = _file_evidence(
        harness,
        excerpt="wrong\n",
        content_hash="0" * 64,
    )
    item_result = harness.checker.check_item(evidence)
    assert len(item_result.diagnostics) == 2

    forged_status = item_result.to_dict()
    forged_status["integrity"] = EvidenceIntegrity.VALID.value
    with pytest.raises(ValueError, match="valid item result"):
        EvidenceItemIntegrityResult.from_dict(forged_status)

    forged_scope = item_result.to_dict()
    forged_scope["diagnostics"][0]["evidence_id"] = "different-evidence"
    with pytest.raises(ValueError, match="outside its item scope"):
        EvidenceItemIntegrityResult.from_dict(forged_scope)

    noncanonical = item_result.to_dict()
    noncanonical["diagnostics"].reverse()
    with pytest.raises(ValueError, match="canonical order"):
        EvidenceItemIntegrityResult.from_dict(noncanonical)

    duplicate_key_json = item_result.to_json().replace(
        '"integrity":',
        '"integrity":"valid","integrity":',
        1,
    )
    with pytest.raises(SchemaError, match="duplicate"):
        EvidenceItemIntegrityResult.from_json(duplicate_key_json)

    coordinated = item_result.to_dict()
    coordinated["integrity"] = EvidenceIntegrity.VALID.value
    coordinated["diagnostics"] = []
    assert EvidenceItemIntegrityResult.from_dict(coordinated).integrity is EvidenceIntegrity.VALID
    with pytest.raises(SchemaError, match="deterministic Evidence replay"):
        EvidenceItemIntegrityResult.from_dict(
            coordinated,
            evidence=evidence,
            checker=harness.checker,
        )


def test_finding_result_hydration_rejects_forged_aggregate_and_diagnostics(
    harness: Harness,
) -> None:
    evidence = _file_evidence(
        harness,
        evidence_id="ev-invalid",
        excerpt="wrong\n",
        content_hash="0" * 64,
    )
    valid = _file_evidence(harness, evidence_id="ev-valid")
    finding = _finding(
        evidence.evidence_id,
        evidence.evidence_id,
        valid.evidence_id,
        "ev-missing",
        finding_id="finding-result-tamper",
    )
    result = harness.checker.check_finding(finding, (valid, evidence))

    forged_integrity = result.to_dict()
    forged_integrity["integrity"] = EvidenceIntegrity.VALID.value
    with pytest.raises(ValueError, match="canonically derived"):
        EvidenceIntegrityResult.from_dict(forged_integrity)

    missing_diagnostic = result.to_dict()
    missing_diagnostic["diagnostics"].pop()
    with pytest.raises(ValueError, match="canonical derived diagnostics"):
        EvidenceIntegrityResult.from_dict(missing_diagnostic)

    noncanonical_diagnostics = result.to_dict()
    noncanonical_diagnostics["diagnostics"].reverse()
    with pytest.raises(ValueError, match="canonical derived diagnostics"):
        EvidenceIntegrityResult.from_dict(noncanonical_diagnostics)

    noncanonical_items = result.to_dict()
    noncanonical_items["item_results"].reverse()
    with pytest.raises((SchemaError, ValueError), match="order|canonical"):
        EvidenceIntegrityResult.from_dict(noncanonical_items)

    duplicate_key_json = result.to_json().replace(
        '"integrity":',
        '"integrity":"valid","integrity":',
        1,
    )
    with pytest.raises(SchemaError, match="duplicate"):
        EvidenceIntegrityResult.from_json(duplicate_key_json)


def test_source_bound_hydration_rejects_coordinated_ref_and_replay_forgery(
    harness: Harness,
) -> None:
    first = _file_evidence(harness, evidence_id="ev-z")
    second = _file_evidence(harness, evidence_id="ev-a")
    finding = _finding(
        first.evidence_id,
        second.evidence_id,
        finding_id="finding-source-binding",
    )
    evidence = (second, first)
    result = harness.checker.check_finding(finding, evidence)

    reordered = result.to_dict()
    reordered["referenced_evidence_ids"].reverse()
    reordered["item_results"].reverse()
    structurally_valid = EvidenceIntegrityResult.from_dict(reordered)
    assert structurally_valid.referenced_evidence_ids != finding.evidence_refs
    with pytest.raises(SchemaError, match="evidence_refs order"):
        EvidenceIntegrityResult.from_dict(
            reordered,
            finding=finding,
            evidence_items=evidence,
        )

    invalid = _file_evidence(
        harness,
        evidence_id="ev-forged",
        content_hash="0" * 64,
    )
    invalid_finding = _finding(
        invalid.evidence_id,
        finding_id="finding-replay-binding",
    )
    invalid_result = harness.checker.check_finding(invalid_finding, (invalid,))
    replay_forgery = invalid_result.to_dict()
    replay_forgery["integrity"] = EvidenceIntegrity.VALID.value
    replay_forgery["item_results"][0]["integrity"] = EvidenceIntegrity.VALID.value
    replay_forgery["item_results"][0]["diagnostics"] = []
    replay_forgery["diagnostics"] = []
    assert EvidenceIntegrityResult.from_dict(replay_forgery).integrity is EvidenceIntegrity.VALID
    with pytest.raises(SchemaError, match="deterministic Evidence replay"):
        EvidenceIntegrityResult.from_dict(
            replay_forgery,
            finding=invalid_finding,
            evidence_items=(invalid,),
            checker=harness.checker,
        )


def test_resolved_invalid_item_aggregates_invalid(harness: Harness) -> None:
    evidence = _file_evidence(harness, excerpt="wrong\n")

    result = harness.checker.check_finding(
        _finding(evidence.evidence_id), (evidence,)
    )

    assert result.integrity is EvidenceIntegrity.INVALID
    assert EvidenceReasonCode.EXCERPT_MISMATCH in _reasons(result)


def test_evidence_input_and_attestation_order_do_not_change_results(
    harness: Harness,
) -> None:
    command_attestation = _attestation(harness)
    unrelated = _attestation(
        harness,
        source_ref="other-command",
        argv=("lint",),
        output=b"clean\n",
    )
    first_checker = EvidenceIntegrityChecker(
        harness.eval_input,
        harness.replay,
        TRIAL_ID,
        TARGET_MATERIALIZATION_ID,
        (command_attestation, unrelated),
    )
    second_checker = EvidenceIntegrityChecker(
        harness.eval_input,
        harness.replay,
        TRIAL_ID,
        TARGET_MATERIALIZATION_ID,
        (unrelated, command_attestation),
    )
    file_item = _file_evidence(harness, evidence_id="ev-a")
    command_item = _command_evidence(harness, evidence_id="ev-b")
    finding = _finding("ev-a", "ev-b")

    first = first_checker.check_finding(finding, (file_item, command_item))
    second = second_checker.check_finding(finding, (command_item, file_item))

    assert first == second
    assert first.integrity is EvidenceIntegrity.VALID


def test_check_all_replays_one_shared_item_once_across_findings(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _diff_evidence(harness)
    original_diff = PreparedRepositoryReplay.diff
    calls: list[str] = []

    def counted_diff(
        replay: PreparedRepositoryReplay,
        path: str,
        *,
        max_bytes: int = MAX_EVIDENCE_EXCERPT_BYTES,
    ) -> bytes:
        calls.append(path)
        return original_diff(replay, path, max_bytes=max_bytes)

    monkeypatch.setattr(PreparedRepositoryReplay, "diff", counted_diff)

    results = harness.checker.check_all(
        (
            _finding(evidence.evidence_id, finding_id="finding-b"),
            _finding(evidence.evidence_id, finding_id="finding-a"),
        ),
        (evidence,),
    )

    assert tuple(result.finding_id for result in results) == (
        "finding-a",
        "finding-b",
    )
    assert all(result.integrity is EvidenceIntegrity.VALID for result in results)
    assert calls == ["src/app.py"]


def test_check_all_caches_diff_source_across_distinct_evidence_ids(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = _diff_evidence(harness, evidence_id="ev-a")
    invalid = replace(
        valid,
        evidence_id="ev-b",
        excerpt="wrong\n",
        content_hash=_digest_text("wrong\n"),
    )
    original_diff = PreparedRepositoryReplay.diff
    calls: list[str] = []

    def counted_diff(
        replay: PreparedRepositoryReplay,
        path: str,
        *,
        max_bytes: int = MAX_EVIDENCE_EXCERPT_BYTES,
    ) -> bytes:
        calls.append(path)
        return original_diff(replay, path, max_bytes=max_bytes)

    monkeypatch.setattr(PreparedRepositoryReplay, "diff", counted_diff)
    results = harness.checker.check_all(
        (
            _finding(valid.evidence_id, finding_id="finding-a"),
            _finding(invalid.evidence_id, finding_id="finding-b"),
        ),
        (invalid, valid),
    )

    assert [item.integrity for item in results] == [
        EvidenceIntegrity.VALID,
        EvidenceIntegrity.INVALID,
    ]
    assert calls == ["src/app.py"]


def test_check_all_caches_file_source_across_distinct_evidence_ids(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _file_evidence(harness, evidence_id="ev-a")
    second = replace(first, evidence_id="ev-b")
    original_read = PreparedRepositoryReplay.read_file
    calls: list[tuple[str, str]] = []

    def counted_read(
        replay: PreparedRepositoryReplay,
        revision: str,
        path: str,
        *,
        max_bytes: int,
    ) -> bytes | None:
        calls.append((revision, path))
        return original_read(replay, revision, path, max_bytes=max_bytes)

    monkeypatch.setattr(PreparedRepositoryReplay, "read_file", counted_read)
    results = harness.checker.check_all(
        (
            _finding(first.evidence_id, finding_id="finding-a"),
            _finding(second.evidence_id, finding_id="finding-b"),
        ),
        (second, first),
    )

    assert all(item.integrity is EvidenceIntegrity.VALID for item in results)
    assert calls == [
        (repository_from_eval_input(harness.eval_input).head_revision, "src/app.py")
    ]


def test_check_all_does_not_replay_unreferenced_evidence(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    referenced = _file_evidence(harness, evidence_id="ev-referenced")
    unreferenced = _diff_evidence(harness, evidence_id="ev-unreferenced")

    def unexpected_diff(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("unreferenced Evidence must not be replayed")

    monkeypatch.setattr(PreparedRepositoryReplay, "diff", unexpected_diff)

    result = harness.checker.check_all(
        (_finding(referenced.evidence_id),),
        (unreferenced, referenced),
    )

    assert len(result) == 1
    assert result[0].integrity is EvidenceIntegrity.VALID


@pytest.mark.parametrize("field_name", ["repository_descriptor_digest", "base_revision", "head_revision"])
def test_checker_rejects_any_repository_binding_mismatch(
    harness: Harness, field_name: str
) -> None:
    replacement = "f" * (
        64 if field_name == "repository_descriptor_digest" else len(getattr(harness.replay, field_name))
    )
    mismatched = replace(harness.replay, **{field_name: replacement})

    with pytest.raises(ValueError, match="exactly bound"):
        EvidenceIntegrityChecker(
            harness.eval_input,
            mismatched,
            TRIAL_ID,
            TARGET_MATERIALIZATION_ID,
        )


def test_repository_integrity_failure_from_tampered_cache_propagates(
    tmp_path: Path,
) -> None:
    harness = _build_harness(tmp_path, heavy=False)
    try:
        evidence = _diff_evidence(harness)
        head = repository_from_eval_input(harness.eval_input).head_revision
        object_path = harness.prepared.cache_path / "objects" / head[:2] / head[2:]
        assert object_path.is_file()
        object_path.chmod(stat.S_IREAD | stat.S_IWRITE)
        object_path.write_bytes(b"tampered-cache-object")

        with pytest.raises(RepositoryIntegrityError):
            harness.checker.check_item(evidence)
    finally:
        harness.preparer.__exit__(None, None, None)


def test_repository_diff_revalidates_cache_control_files(
    tmp_path: Path,
) -> None:
    harness = _build_harness(tmp_path, heavy=False)
    try:
        evidence = _diff_evidence(harness)
        config_path = harness.prepared.cache_path / "config"
        config_path.chmod(stat.S_IREAD | stat.S_IWRITE)
        config_path.write_bytes(
            config_path.read_bytes() + b"\n[diff]\n\tnoprefix = true\n"
        )

        with pytest.raises(RepositoryIntegrityError):
            harness.checker.check_item(evidence)
    finally:
        harness.preparer.__exit__(None, None, None)


def test_repository_security_failure_from_replay_propagates(harness: Harness) -> None:
    def fail_closed() -> None:
        raise RepositorySecurityError("redacted replay boundary failure")

    replay = replace(harness.replay, _open_check=fail_closed)
    checker = EvidenceIntegrityChecker(
        harness.eval_input, replay, TRIAL_ID, TARGET_MATERIALIZATION_ID
    )

    with pytest.raises(RepositorySecurityError):
        checker.check_item(_file_evidence(harness))


def test_checker_api_has_no_ground_truth_anchor_or_support_input() -> None:
    constructor = inspect.signature(EvidenceIntegrityChecker)
    item_check = inspect.signature(EvidenceIntegrityChecker.check_item)
    finding_check = inspect.signature(EvidenceIntegrityChecker.check_finding)

    assert "anchor" not in " ".join(constructor.parameters).lower()
    assert "anchor" not in " ".join(item_check.parameters).lower()
    assert "anchor" not in " ".join(finding_check.parameters).lower()
    assert "support" not in " ".join(finding_check.parameters).lower()


def test_finding_location_and_claim_are_not_evidence_integrity_inputs(
    harness: Harness,
) -> None:
    evidence = _file_evidence(harness)
    first = _finding(evidence.evidence_id)
    second = replace(
        first,
        claim="Entirely different semantic claim",
        path="unrelated.py",
        side=DiffSide.LEFT,
        from_line=999,
        to_line=1000,
    )

    first_result = harness.checker.check_finding(first, (evidence,))
    second_result = harness.checker.check_finding(second, (evidence,))

    assert first_result.integrity is EvidenceIntegrity.VALID
    assert second_result.integrity is EvidenceIntegrity.VALID
    assert first_result.item_results == second_result.item_results

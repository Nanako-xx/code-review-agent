from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from review_agent_eval.artifacts import TrialManifest
from review_agent_eval.cases import (
    CaseSplit,
    REPOSITORY_MATERIALIZER_PROTOCOL,
    SuiteCase,
    WireContractV2,
)
from review_agent_eval.config import (
    ADAPTER_CAPABILITIES_SCHEMA_VERSION,
    AdapterCapabilitiesV2,
    derive_case_path_id,
    derive_trial_id,
    derive_trial_seed,
)
from review_agent_eval.materialization import (
    MaterializationError,
    MaterializationRequest,
    RepositoryTargetMaterializer,
)
from review_agent_eval.models import (
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    EvalInput,
    ReviewTargetKind,
    stable_id,
    TruthCompleteness,
)
from review_agent_eval.repository import (
    FixtureRepositoryBuilder,
    RepositoryMode,
    RepositoryPreparer,
    RepositoryPreparationError,
)


def _git_executable() -> Path:
    executable = shutil.which("git")
    assert executable is not None
    return Path(executable).absolute()


def _make_repository_fixture(root: Path):
    suite_root = root / "suite"
    fixture = suite_root / "repositories" / "demo"
    (fixture / "base").mkdir(parents=True)
    (fixture / "head").mkdir(parents=True)
    (fixture / "base" / "app.py").write_bytes(b"def allowed(user):\n    return False\n")
    (fixture / "head" / "app.py").write_bytes(
        b"def allowed(user):\n    return user.is_admin\n"
    )
    (fixture / "head" / "README.md").write_bytes(b"# Demo\n")
    built = FixtureRepositoryBuilder().build(fixture, root / "authored.git")
    return suite_root, built.to_repository("repositories/demo")


def _make_input(repository) -> EvalInput:
    return EvalInput.from_dict(
        {
            "schema_version": EVAL_INPUT_SCHEMA_VERSION,
            "task_id": "materialization-case",
            "review_target": {
                "kind": "repository",
                "repository": repository.to_dict(),
                "review_request": {
                    "title": "Review the change",
                    "description": None,
                    "user_intent": None,
                    "review_focus": None,
                    "linked_requirements": [],
                    "project_rules": [],
                    "existing_ci_evidence": [],
                },
            },
        }
    )


def _make_capabilities() -> AdapterCapabilitiesV2:
    return AdapterCapabilitiesV2.from_dict(
        {
            "schema_version": ADAPTER_CAPABILITIES_SCHEMA_VERSION,
            "adapter_id": "current-agent-cli-v2",
            "adapter_version": "2",
            "input_schema_version": EVAL_INPUT_SCHEMA_VERSION,
            "submission_schema_version": EVAL_SUBMISSION_SCHEMA_VERSION,
            "target_kinds": ["repository"],
            "evidence_kinds": ["repository_file", "repository_diff", "command_output"],
            "clarification_protocol": "canonical-clarification-v2",
            "trace_protocol": "local-trace-v2",
            "subprocess_wire_version": None,
            "isolation_profile": "repository-worktree-v2",
        }
    )


def _make_request(
    eval_input: EvalInput,
    capabilities: AdapterCapabilitiesV2,
) -> MaterializationRequest:
    wire = WireContractV2(
        case_schema_version="eval_case_v2",
        input_schema_version=EVAL_INPUT_SCHEMA_VERSION,
        submission_schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
        review_target_kind=ReviewTargetKind.REPOSITORY,
        materializer_protocol=REPOSITORY_MATERIALIZER_PROTOCOL,
    )
    run_id = stable_id("run", "materialization-tests")
    case_digest = "a" * 64
    trial_id = derive_trial_id(run_id, eval_input.task_id, 1)
    trial_manifest = TrialManifest(
        schema_version=TrialManifest.SCHEMA_VERSION,
        run_id=run_id,
        task_id=eval_input.task_id,
        case_path_id=derive_case_path_id(eval_input.task_id),
        canonical_case_digest=case_digest,
        eval_input_digest=eval_input.digest(),
        wire_contract=wire,
        target_kind=ReviewTargetKind.REPOSITORY,
        materializer_protocol=REPOSITORY_MATERIALIZER_PROTOCOL,
        suite_preparation_binding_digest=None,
        adapter_capabilities_digest=capabilities.digest(),
        trial_id=trial_id,
        trial_index=1,
        seed=derive_trial_seed(run_id, eval_input.task_id, 1),
        agent_config_digest="b" * 64,
        initial_evaluator_execution_digest="c" * 64,
    )
    suite_case = SuiteCase(
        task_id=eval_input.task_id,
        case_version=1,
        path="cases/materialization-case.json",
        split=CaseSplit.REGRESSION,
        protocol_id="native_repository",
        dimensions=(),
        raw_file_size_bytes=1,
        raw_file_sha256="d" * 64,
        canonical_case_digest=case_digest,
        eval_input_digest=eval_input.digest(),
        truth_completeness=TruthCompleteness.CLOSED_WORLD,
    )
    return MaterializationRequest(
        eval_input=eval_input,
        trial_manifest=trial_manifest,
        suite_case=suite_case,
        attempt=1,
        wire_contract=wire,
        suite_preparation_binding_digest=None,
        adapter_capabilities=capabilities,
    )


def _preparer(root: Path, suite_root: Path, mode: RepositoryMode) -> RepositoryPreparer:
    return RepositoryPreparer(
        suite_root=suite_root,
        data_root=root / ".eval-data",
        workspace_root=root / ".eval-workspaces",
        git_executable=_git_executable(),
        repository_mode=mode,
    )


def test_repository_materializer_uses_cache_only_and_round_trips(tmp_path: Path) -> None:
    suite_root, repository = _make_repository_fixture(tmp_path)
    eval_input = _make_input(repository)
    capabilities = _make_capabilities()
    request = _make_request(eval_input, capabilities)

    with _preparer(tmp_path, suite_root, RepositoryMode.ACQUIRE) as preparer:
        preparer.prepare(repository)

    with _preparer(tmp_path, suite_root, RepositoryMode.CACHE_ONLY) as preparer:
        materialized = RepositoryTargetMaterializer(preparer).materialize(request)
        with materialized:
            assert materialized.manifest.materialization_id == (
                materialized.target_access.target_materialization_id
            )
            assert {item.relative_path for item in materialized.manifest.files} == {
                "app.py",
                "README.md",
            }
            assert len(materialized.manifest.replay_binding_digest) == 64
            assert set(materialized.manifest.replay_binding_digest) <= set(
                "0123456789abcdef"
            )
            assert materialized.replay.repository_descriptor_digest == repository.digest()
            materialized.validate()
        assert materialized.closed


def test_repository_materializer_rejects_missing_cache_without_acquisition(
    tmp_path: Path,
) -> None:
    suite_root, repository = _make_repository_fixture(tmp_path)
    request = _make_request(_make_input(repository), _make_capabilities())
    with _preparer(tmp_path, suite_root, RepositoryMode.ACQUIRE):
        pass
    with _preparer(tmp_path, suite_root, RepositoryMode.CACHE_ONLY) as preparer:
        with pytest.raises(RepositoryPreparationError, match="not prepared"):
            RepositoryTargetMaterializer(preparer).materialize(request)


def test_repository_materialization_fails_when_agent_visible_file_drifts(
    tmp_path: Path,
) -> None:
    suite_root, repository = _make_repository_fixture(tmp_path)
    eval_input = _make_input(repository)
    request = _make_request(eval_input, _make_capabilities())
    with _preparer(tmp_path, suite_root, RepositoryMode.ACQUIRE) as preparer:
        preparer.prepare(repository)

    with _preparer(tmp_path, suite_root, RepositoryMode.CACHE_ONLY) as preparer:
        materialized = RepositoryTargetMaterializer(preparer).materialize(request)
        repository_lease = materialized._lease
        assert hasattr(repository_lease, "workspace")
        (repository_lease.workspace.path / "app.py").write_bytes(b"tampered\n")
        with pytest.raises(MaterializationError, match="drifted|differs"):
            with materialized:
                pass
        assert materialized.closed


def test_repository_materializer_rejects_frozen_target_before_cache_access(
    tmp_path: Path,
) -> None:
    suite_root, repository = _make_repository_fixture(tmp_path)
    request = _make_request(_make_input(repository), _make_capabilities())
    frozen = EvalInput.from_dict(
        {
            "schema_version": EVAL_INPUT_SCHEMA_VERSION,
            "task_id": request.eval_input.task_id,
            "review_target": {
                "kind": "frozen_context",
                "bundle_id": "bundle-1",
                "record_id": "record-1",
                "context_format": "rendered_text",
                "rendered_sha256": "e" * 64,
                "rendered_utf8_bytes": 4,
                "source_binding_digest": "f" * 64,
            },
        }
    )
    invalid_request = replace(request, eval_input=frozen)
    with _preparer(tmp_path, suite_root, RepositoryMode.ACQUIRE):
        pass
    with _preparer(tmp_path, suite_root, RepositoryMode.CACHE_ONLY) as preparer:
        with pytest.raises(MaterializationError, match="non-Repository|tagged"):
            RepositoryTargetMaterializer(preparer).materialize(invalid_request)


def test_repository_materializer_rejects_target_replacement_before_cache_access(
    tmp_path: Path,
) -> None:
    suite_root, repository = _make_repository_fixture(tmp_path)
    request = _make_request(_make_input(repository), _make_capabilities())
    replacement_repository = replace(repository, head_revision="0" * 40)
    replacement_target = replace(
        request.eval_input.review_target,
        repository=replacement_repository,
    )
    invalid_request = replace(
        request,
        eval_input=replace(
            request.eval_input,
            review_target=replacement_target,
        ),
    )
    with _preparer(tmp_path, suite_root, RepositoryMode.ACQUIRE):
        pass
    with _preparer(tmp_path, suite_root, RepositoryMode.CACHE_ONLY) as preparer:
        with pytest.raises(MaterializationError, match="input digest"):
            RepositoryTargetMaterializer(preparer).materialize(invalid_request)

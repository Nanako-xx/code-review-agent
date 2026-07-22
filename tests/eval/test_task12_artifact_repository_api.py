from __future__ import annotations

import json
from pathlib import Path

import pytest

import review_agent_eval.repository as repository_module
from review_agent_eval.artifacts import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactStateError,
    ArtifactStore,
    VerifiedTrialMaterialization,
)
from review_agent_eval.config import (
    EvaluatorExecutionConfig,
    derive_evaluation_id,
)
from review_agent_eval.models import (
    Repository,
    RepositorySource,
    UnsupportedProtocolVersionError,
    canonical_json_bytes,
)
from review_agent_eval.repository import (
    RepositoryCacheStatus,
    RepositoryMode,
    RepositoryPreparationError,
)

from .test_artifacts import (
    TASK_ID,
    complete_trial,
    make_store,
)
from .test_config import evaluator_config
from .test_repository import _author_fixture, _preparer


def _evaluation_values(score: int = 1):
    return {
        "intent_matches": {"matches": ["intent-1"]},
        "review_matches": {"matches": ["finding-1"]},
        "judge_input": {"requests": []},
        "judge_output": {"status": "graded", "results": []},
        "score": {"total": score},
        "report": "# Trial evaluation\n",
    }


def test_public_trial_materialization_is_prepare_receipt_bound(
    tmp_path: Path,
) -> None:
    store, config, _manifest, plan, trial = make_store(tmp_path)
    complete_trial(store, config, plan)

    verified = store.load_trial_materialization(
        config.run_id,
        TASK_ID,
        plan.trial_id,
    )

    assert type(verified) is VerifiedTrialMaterialization
    assert verified.eval_input.digest() == trial.eval_input_digest
    assert verified.manifest.materialization_id == (
        verified.prepare_receipt.materialization_id
    )
    assert verified.manifest.attempt == verified.active_attempt == 1
    assert verified.trial_manifest == trial


@pytest.mark.parametrize(
    "artifact,field,value",
    (
        ("input.json", "task_id", "tampered-task"),
        ("materialization_manifest.json", "replay_binding_digest", "0" * 64),
        ("prepare.json", "prepared_source_id", "tampered-source"),
        ("trial_manifest.json", "agent_config_digest", "0" * 64),
    ),
)
def test_public_trial_materialization_rejects_control_plane_drift(
    tmp_path: Path,
    artifact: str,
    field: str,
    value: str,
) -> None:
    store, config, _manifest, plan, trial = make_store(tmp_path)
    complete_trial(store, config, plan)
    trial_root = (
        store.root
        / config.run_id
        / "cases"
        / trial.case_path_id
        / "trials"
        / plan.trial_id
    )
    paths = {
        "input.json": trial_root / "input.json",
        "materialization_manifest.json": (
            trial_root
            / "materializations"
            / "attempt-0001"
            / "materialization_manifest.json"
        ),
        "prepare.json": (
            trial_root / "receipts" / "attempt-0001" / "prepare.json"
        ),
        "trial_manifest.json": trial_root / "trial_manifest.json",
    }
    path = paths[artifact]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises((ArtifactIntegrityError, UnsupportedProtocolVersionError)):
        store.load_trial_materialization(
            config.run_id,
            TASK_ID,
            plan.trial_id,
        )


def test_cache_only_mode_never_reacquires_and_opens_verified_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = tmp_path / "suite"
    descriptor, _built = _author_fixture(suite, tmp_path)
    with _preparer(tmp_path, suite) as preparer:
        prepared = preparer.prepare(descriptor)

    cache_before = {
        path.relative_to(tmp_path / ".eval-data").as_posix(): path.read_bytes()
        for root_name in ("repositories", "indexes")
        for path in (tmp_path / ".eval-data" / root_name).rglob("*")
        if path.is_file()
    }
    trash_sentinel = tmp_path / ".eval-workspaces" / ".trash" / "owned-by-agent-stage"
    trash_sentinel.mkdir()
    (trash_sentinel / "sentinel.txt").write_text(
        "cache-only must not prune unrelated workspace state",
        encoding="utf-8",
    )

    def forbidden_acquisition(*_args, **_kwargs):
        raise AssertionError("cache-only mode attempted repository acquisition")

    monkeypatch.setattr(
        repository_module.RepositoryPreparer,
        "_acquire_closure",
        forbidden_acquisition,
    )
    with _preparer(
        tmp_path,
        suite,
        repository_mode=RepositoryMode.CACHE_ONLY,
    ) as cache_only:
        check = cache_only.check_cached(descriptor)
        assert check.status is RepositoryCacheStatus.AVAILABLE
        assert check.available is True
        assert check.prepared_repository_id == prepared.manifest.prepared_repository_id
        assert "cache_path" not in check.to_dict()

        required = cache_only.require_cached(descriptor)
        assert cache_only.prepare(descriptor) == required
        replay = cache_only.open_replay_for(descriptor)
        assert replay.read_file(descriptor.head_revision, "app.py") == (
            b"def allowed(user):\n    return user.is_admin\n"
        )

        missing = Repository(
            source=RepositorySource.FIXTURE,
            path="repositories/missing",
            url=None,
            base_revision=descriptor.base_revision,
            head_revision=descriptor.head_revision,
        )
        missing_check = cache_only.check_cached(missing)
        assert missing_check.status is RepositoryCacheStatus.MISSING
        assert missing_check.available is False
        with pytest.raises(RepositoryPreparationError, match="run prepare first") as error:
            cache_only.prepare(missing)
        assert "repositories/missing" not in str(error.value)

    cache_after = {
        path.relative_to(tmp_path / ".eval-data").as_posix(): path.read_bytes()
        for root_name in ("repositories", "indexes")
        for path in (tmp_path / ".eval-data" / root_name).rglob("*")
        if path.is_file()
    }
    assert cache_after == cache_before
    assert (trash_sentinel / "sentinel.txt").read_text(encoding="utf-8") == (
        "cache-only must not prune unrelated workspace state"
    )


def test_committed_evaluation_bundle_is_strict_listable_and_resume_safe(
    tmp_path: Path,
) -> None:
    store, config, _manifest, plan, _trial = make_store(tmp_path)
    complete_trial(store, config, plan)
    execution = EvaluatorExecutionConfig.from_resource_budgets(
        evaluator_config(), config.resource_budgets
    )
    values = _evaluation_values()
    receipt = store.write_evaluation(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        evaluator_execution=execution,
        revision="task12-bundle-v1",
        **values,
    )
    assert receipt.evaluation_id is not None

    namespace = store.load_evaluation_namespace(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        receipt.evaluation_id,
    )
    loaded = store.load_evaluation_bundle(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        receipt.evaluation_id,
    )
    assert namespace == loaded.namespace
    assert store.list_evaluations(
        config.run_id, TASK_ID, plan.trial_id
    ) == (namespace,)
    assert store.list_evaluations(config.run_id) == (namespace,)
    assert loaded.evaluator_execution == execution
    assert loaded.score == {"total": 1}
    assert loaded.report == "# Trial evaluation\n"
    assert str(store.root) not in repr(loaded)
    assert all(not ref.relative_path.startswith(("/", "\\")) for ref in namespace.artifacts)

    detached_score = loaded.score
    detached_score["total"] = 999
    assert loaded.score == {"total": 1}

    resumed = store.write_evaluation(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        evaluator_execution=execution,
        revision="task12-bundle-v1",
        resume=True,
        **values,
    )
    assert resumed == receipt
    with pytest.raises(ArtifactConflictError, match="differs"):
        store.write_evaluation(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            evaluator_execution=execution,
            revision="task12-bundle-v1",
            resume=True,
            **_evaluation_values(score=2),
        )
    with pytest.raises(ArtifactConflictError, match="immutable"):
        store.write_evaluation(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            evaluator_execution=execution,
            revision="task12-bundle-v1",
            overwrite=True,
            **values,
        )


def test_evaluation_bundle_rejects_orphan_and_receipt_bound_tampering(
    tmp_path: Path,
) -> None:
    store, config, _manifest, plan, trial = make_store(tmp_path)
    complete_trial(store, config, plan)
    execution = EvaluatorExecutionConfig.from_resource_budgets(
        evaluator_config(), config.resource_budgets
    )
    orphan_id = derive_evaluation_id(
        config.run_id, execution.digest(), "task12-orphan-v1"
    )
    orphan_base = "cases/%s/trials/%s/evaluations/%s" % (
        trial.case_path_id,
        plan.trial_id,
        orphan_id,
    )
    store._write_json(
        config.run_id,
        orphan_base + "/score.json",
        {"orphan": True},
    )
    with pytest.raises(ArtifactStateError, match="commit receipt"):
        store.list_evaluations(config.run_id, TASK_ID, plan.trial_id)

    orphan_path = store.root / config.run_id / Path(*orphan_base.split("/"))
    (orphan_path / "score.json").unlink()
    orphan_path.rmdir()
    receipt = store.write_evaluation(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        evaluator_execution=execution,
        revision="task12-tamper-v1",
        **_evaluation_values(),
    )
    assert receipt.evaluation_id is not None
    score_path = (
        store.root
        / config.run_id
        / "cases"
        / trial.case_path_id
        / "trials"
        / plan.trial_id
        / "evaluations"
        / receipt.evaluation_id
        / "score.json"
    )
    score_path.write_bytes(canonical_json_bytes({"total": 9}))
    with pytest.raises(ArtifactIntegrityError, match="size|hash"):
        store.load_evaluation_bundle(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            receipt.evaluation_id,
        )


def test_v1_run_root_cannot_resume_load_or_rejudge(tmp_path: Path) -> None:
    store, config, _manifest, plan, _trial = make_store(tmp_path)
    complete_trial(store, config, plan)
    run_manifest_path = store.root / config.run_id / "run_manifest.json"
    payload = json.loads(
        run_manifest_path.read_text(encoding="utf-8")
    )
    payload["schema_version"] = "eval_run_manifest_v1"
    payload["legacy_unknown"] = True
    run_manifest_path.write_bytes(canonical_json_bytes(payload))
    evaluation_root = (
        store.root
        / config.run_id
        / "cases"
        / plan.case_path_id
        / "trials"
        / plan.trial_id
        / "evaluations"
    )
    before = tuple(evaluation_root.iterdir())

    for operation in (
        lambda: store.load_run_manifest(config.run_id),
        lambda: store.load_run_config(config.run_id),
        lambda: store.load_existing_submission(
            config.run_id, TASK_ID, plan.trial_id
        ),
        lambda: store.recover_trial(config.run_id, TASK_ID, plan.trial_id),
    ):
        with pytest.raises(UnsupportedProtocolVersionError):
            operation()

    assert tuple(evaluation_root.iterdir()) == before

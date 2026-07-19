from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import review_agent_eval.repository as repository_module
from review_agent_eval.artifacts import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactStateError,
    ArtifactStore,
)
from review_agent_eval.config import (
    EvaluatorExecutionConfig,
    derive_evaluation_id,
)
from review_agent_eval.models import Repository, RepositorySource, canonical_json_bytes
from review_agent_eval.report import ReportBuilder, render_run_markdown
from review_agent_eval.repository import (
    RepositoryCacheStatus,
    RepositoryMode,
    RepositoryPreparationError,
)

from .test_artifacts import (
    TASK_ID,
    complete_trial,
    make_store,
    required_runner_artifacts,
)
from .test_config import evaluator_config
from .test_metrics import _case_and_snapshot
from .test_report import _report_sources
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


def test_run_evaluation_summary_report_is_create_only_and_source_hydratable(
    tmp_path: Path,
) -> None:
    case, _replay, execution, config, sources = _report_sources()
    same_case, snapshot, _same_replay = _case_and_snapshot()
    assert same_case == case
    store = ArtifactStore(tmp_path / ".eval-runs")
    run_manifest = store.create_run(config, snapshot)

    bound_sources = []
    for source in sources:
        assert source.submission is not None
        plan = next(
            item
            for item in run_manifest.trials
            if item.trial_id == source.submission.trial_id
        )
        running = store.start_trial(config.run_id, case.task_id, plan.trial_id)
        assert running.active_attempt is not None
        store.write_prepare_stage(
            config.run_id,
            case.task_id,
            plan.trial_id,
            case.eval_input(),
            attempt=running.active_attempt,
        )
        store.finalize_submission(
            config.run_id,
            case.task_id,
            plan.trial_id,
            source.submission,
            attempt=running.active_attempt,
            runner_artifacts=required_runner_artifacts(source.submission),
        )
        store.write_evaluation(
            config.run_id,
            case.task_id,
            plan.trial_id,
            evaluator_execution=execution,
            revision="metrics-eval-v1",
            intent_matches=(
                {"status": "not_available"}
                if source.intent_result is None
                else source.intent_result.to_dict()
            ),
            review_matches=(
                {"status": "not_available"}
                if source.review_result is None
                else source.review_result.to_dict()
            ),
            judge_input={"requests": []},
            judge_output={"results": []},
            score=source.trial_score.to_dict(),
        )
        bound_sources.append(
            replace(
                source,
                trial_manifest=store.load_trial_manifest(
                    config.run_id,
                    case.task_id,
                    plan.trial_id,
                ),
            )
        )

    builder = ReportBuilder()
    summary = builder.build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=tuple(bound_sources),
        run_manifest=run_manifest,
    )
    written = store.write_run_evaluation(
        config.run_id,
        evaluator_execution=execution,
        revision="metrics-eval-v1",
        summary=summary,
    )
    assert written.summary == summary.to_dict()
    assert written.report == render_run_markdown(summary)
    assert written.namespace.summary.relative_path.endswith("/summary.json")
    assert written.namespace.report.relative_path.endswith("/report.md")
    assert str(store.root) not in repr(written)
    assert store.list_run_evaluations(config.run_id) == (written.namespace,)
    hydrated = written.hydrate_summary(
        builder=builder,
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=tuple(bound_sources),
        run_manifest=run_manifest,
    )
    assert hydrated.to_dict() == summary.to_dict()

    detached_summary = written.summary
    detached_summary["coverage"]["planned_trial_count"] = 999
    assert written.summary == summary.to_dict()
    with pytest.raises(ArtifactConflictError, match="use resume"):
        store.write_run_evaluation(
            config.run_id,
            evaluator_execution=execution,
            revision="metrics-eval-v1",
            summary=summary,
        )
    resumed = store.write_run_evaluation(
        config.run_id,
        evaluator_execution=execution,
        revision="metrics-eval-v1",
        summary=summary,
        resume=True,
    )
    assert resumed.namespace == written.namespace
    with pytest.raises(ArtifactConflictError, match="cannot be overwritten"):
        store.write_run_evaluation(
            config.run_id,
            evaluator_execution=execution,
            revision="metrics-eval-v1",
            summary=summary,
            overwrite=True,
        )

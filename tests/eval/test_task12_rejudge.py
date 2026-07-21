from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_agent_eval.config import EvaluatorExecutionConfig
from review_agent_eval.models import (
    UnsupportedProtocolVersionError,
    canonical_json_bytes,
)

from .test_artifacts import TASK_ID, complete_trial, make_store
from .test_config import evaluator_config


def _evaluation_values(score: int = 1) -> dict:
    return {
        "intent_matches": {"matches": ["intent-1"]},
        "review_matches": {"matches": ["finding-1"]},
        "judge_input": {"requests": []},
        "judge_output": {"status": "graded", "results": []},
        "score": {"total": score},
        "report": "# Trial evaluation %d\n" % score,
    }


def test_rejudge_changes_execution_namespace_and_reuses_submission(
    tmp_path: Path,
) -> None:
    store, config, _manifest, plan, _trial = make_store(tmp_path)
    submission, _state = complete_trial(store, config, plan)
    original_execution = EvaluatorExecutionConfig.from_resource_budgets(
        config.evaluator,
        config.resource_budgets,
    )
    revision = "task12-rejudge-v2"
    first = store.write_evaluation(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        evaluator_execution=original_execution,
        revision=revision,
        **_evaluation_values(1),
    )
    assert first.evaluation_id is not None
    first_bundle = store.load_evaluation_bundle(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        first.evaluation_id,
    )

    resumed = store.write_evaluation(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        evaluator_execution=original_execution,
        revision=revision,
        resume=True,
        **_evaluation_values(1),
    )
    assert resumed == first

    changed_judge_execution = EvaluatorExecutionConfig.from_resource_budgets(
        evaluator_config(judge_version="judge-v2", model="judge-model-v2"),
        config.resource_budgets,
    )
    changed_judge = store.write_evaluation(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        evaluator_execution=changed_judge_execution,
        revision=revision,
        **_evaluation_values(2),
    )
    changed_budget_execution = EvaluatorExecutionConfig.create(
        evaluator=config.evaluator,
        evaluator_timeout_seconds=(
            original_execution.evaluator_timeout_seconds + 1
        ),
        max_execution_artifact_file_bytes=(
            original_execution.max_execution_artifact_file_bytes
        ),
        max_execution_artifact_total_bytes=(
            original_execution.max_execution_artifact_total_bytes
        ),
    )
    changed_budget = store.write_evaluation(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        evaluator_execution=changed_budget_execution,
        revision=revision,
        **_evaluation_values(3),
    )
    changed_context_execution = EvaluatorExecutionConfig.from_resource_budgets(
        config.evaluator,
        config.resource_budgets,
        review_evaluator_context_policy_version="truth-scoped-context-v3",
    )
    changed_context = store.write_evaluation(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        evaluator_execution=changed_context_execution,
        revision=revision,
        **_evaluation_values(4),
    )
    changed_authority_execution = EvaluatorExecutionConfig.from_resource_budgets(
        config.evaluator,
        config.resource_budgets,
        metric_authority_policy_version="metric-authority-v3",
    )
    changed_authority = store.write_evaluation(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        evaluator_execution=changed_authority_execution,
        revision=revision,
        **_evaluation_values(5),
    )

    assert len(
        {
            first.evaluation_id,
            changed_judge.evaluation_id,
            changed_budget.evaluation_id,
            changed_context.evaluation_id,
            changed_authority.evaluation_id,
        }
    ) == 5
    assert store.load_run_config(config.run_id).run_id == config.run_id
    assert store.load_existing_submission(
        config.run_id,
        TASK_ID,
        plan.trial_id,
    ).digest() == submission.digest()
    assert first_bundle.submission_digest == submission.digest()
    assert len(store.list_evaluations(config.run_id)) == 5


def test_v1_evaluation_receipt_cannot_load_or_resume_rejudge(
    tmp_path: Path,
) -> None:
    store, config, _manifest, plan, trial = make_store(tmp_path)
    complete_trial(store, config, plan)
    execution = EvaluatorExecutionConfig.from_resource_budgets(
        config.evaluator,
        config.resource_budgets,
    )
    revision = "task12-v1-rejection"
    receipt = store.write_evaluation(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        evaluator_execution=execution,
        revision=revision,
        **_evaluation_values(),
    )
    assert receipt.evaluation_id is not None
    evaluation_root = (
        store.root
        / config.run_id
        / "cases"
        / trial.case_path_id
        / "trials"
        / plan.trial_id
        / "evaluations"
    )
    before_names = tuple(sorted(path.name for path in evaluation_root.iterdir()))
    receipt_path = (
        evaluation_root / receipt.evaluation_id / "receipt.json"
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "eval_stage_receipt_v1"
    payload["legacy_unknown"] = True
    receipt_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(UnsupportedProtocolVersionError):
        store.load_evaluation_bundle(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            receipt.evaluation_id,
        )
    with pytest.raises(UnsupportedProtocolVersionError):
        store.write_evaluation(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            evaluator_execution=execution,
            revision=revision,
            **_evaluation_values(),
        )
    with pytest.raises(UnsupportedProtocolVersionError):
        store.write_evaluation(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            evaluator_execution=execution,
            revision=revision,
            resume=True,
            **_evaluation_values(),
        )
    assert tuple(sorted(path.name for path in evaluation_root.iterdir())) == (
        before_names
    )

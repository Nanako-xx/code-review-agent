from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Iterator

import pytest

from review_agent_eval.intent_evaluator import IntentEvaluationStatus
from review_agent_eval.models import SubmissionStatus
from review_agent_eval.review_evaluator import ReviewEvaluationStatus


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROMOTION_SCRIPT = (
    REPOSITORY_ROOT / "eval" / "authoring" / "verify_core_regression.py"
)


@pytest.fixture(scope="module")
def promotion_module() -> Iterator[ModuleType]:
    module_name = "_review_agent_core_promotion_tests"
    spec = importlib.util.spec_from_file_location(module_name, PROMOTION_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize(
    "value",
    (
        {},
        {"run_id": "run-a", "evaluation_id": "eval-a"},
        {
            "run_id": "run-a",
            "evaluation_id": "eval-a",
            "summary_id": "summary-a",
            "passed": True,
        },
        {"run_id": "", "evaluation_id": "eval-a", "summary_id": "summary-a"},
    ),
)
def test_promotion_locator_rejects_redundant_or_incomplete_claims(
    promotion_module: ModuleType,
    value: dict[str, object],
) -> None:
    with pytest.raises(
        promotion_module.CoreRegressionPromotionError,
        match="promotion evidence",
    ):
        promotion_module._promotion_locator(value)


def _passing_trial() -> tuple[SimpleNamespace, SimpleNamespace]:
    intent = SimpleNamespace(
        status=IntentEvaluationStatus.GRADED,
        metrics=SimpleNamespace(intent_case_pass=True),
        judge_failures=(),
        judge_ungraded=(),
    )
    coverage = SimpleNamespace(
        judge_failed_count=0,
        judge_ungraded_count=0,
        judge_pending_count=0,
        semantic_unknown_count=0,
    )
    metrics = SimpleNamespace(
        scorable=True,
        generated_finding_count=1,
        expected_truth_count=1,
        required_expected_truth_count=1,
        matched_finding_count=1,
        matched_expected_truth_count=1,
        matched_required_truth_count=1,
        duplicate_finding_count=0,
        unmatched_required_truth_count=0,
        known_invalid_finding_count=0,
        plausible_novel_count=0,
        fabricated_finding_count=0,
        unknown_finding_count=0,
        unmatched_expected_truth_count=0,
        evidence_valid_count=1,
        evidence_invalid_count=0,
        evidence_missing_count=0,
        evidence_supported_count=1,
        evidence_weak_count=0,
        evidence_unsupported_count=0,
        evidence_support_unknown_count=0,
        strict_publishable_count=1,
    )
    review = SimpleNamespace(
        status=ReviewEvaluationStatus.GRADED,
        judge_failures=(),
        judge_ungraded=(),
        coverage=coverage,
        metrics=metrics,
        finding_outcomes=(
            SimpleNamespace(
                matched_expected_truth_id="issue-required",
                strict_publishable=True,
            ),
        ),
    )
    trial = SimpleNamespace(
        trial_id="trial-001",
        submission=SimpleNamespace(status=SubmissionStatus.COMPLETED),
        intent_result=intent,
        review_result=review,
    )
    case = SimpleNamespace(
        review_truth=SimpleNamespace(
            expected_findings=(
                SimpleNamespace(truth_id="issue-required", required=True),
            )
        )
    )
    return trial, case


def test_promotion_trial_requires_3_of_3_semantic_and_evidence_pass(
    promotion_module: ModuleType,
) -> None:
    trial, case = _passing_trial()

    promotion_module._assert_trial_pass(trial, case)

    trial.review_result.finding_outcomes[0].strict_publishable = False
    with pytest.raises(
        promotion_module.CoreRegressionPromotionError,
        match=r"valid\+supported Evidence",
    ):
        promotion_module._assert_trial_pass(trial, case)


@pytest.mark.parametrize(
    ("target", "field"),
    (
        ("intent", "intent_case_pass"),
        ("coverage", "semantic_unknown_count"),
        ("review_metrics", "unmatched_required_truth_count"),
        ("review_metrics", "duplicate_finding_count"),
        ("review_metrics", "known_invalid_finding_count"),
        ("review_metrics", "plausible_novel_count"),
        ("review_metrics", "fabricated_finding_count"),
        ("review_metrics", "unknown_finding_count"),
        ("review_metrics", "evidence_invalid_count"),
        ("review_metrics", "evidence_missing_count"),
        ("review_metrics", "evidence_weak_count"),
        ("review_metrics", "evidence_unsupported_count"),
        ("review_metrics", "evidence_support_unknown_count"),
    ),
)
def test_promotion_trial_rejects_each_core_failure_projection(
    promotion_module: ModuleType,
    target: str,
    field: str,
) -> None:
    trial, case = _passing_trial()
    if target == "intent":
        setattr(trial.intent_result.metrics, field, False)
    elif target == "coverage":
        setattr(trial.review_result.coverage, field, 1)
    else:
        setattr(trial.review_result.metrics, field, 1)

    with pytest.raises(promotion_module.CoreRegressionPromotionError):
        promotion_module._assert_trial_pass(trial, case)


@pytest.mark.parametrize(
    "field",
    (
        "evidence_valid_count",
        "evidence_supported_count",
        "strict_publishable_count",
    ),
)
def test_promotion_trial_requires_complete_publishable_metric_counts(
    promotion_module: ModuleType,
    field: str,
) -> None:
    trial, case = _passing_trial()
    setattr(trial.review_result.metrics, field, 0)

    with pytest.raises(
        promotion_module.CoreRegressionPromotionError,
        match="every generated Finding",
    ):
        promotion_module._assert_trial_pass(trial, case)


def test_promotion_trial_requires_one_strict_outcome_per_generated_finding(
    promotion_module: ModuleType,
) -> None:
    trial, case = _passing_trial()
    metrics = trial.review_result.metrics
    metrics.generated_finding_count = 2
    metrics.evidence_valid_count = 2
    metrics.evidence_supported_count = 2
    metrics.strict_publishable_count = 2

    with pytest.raises(
        promotion_module.CoreRegressionPromotionError,
        match="every generated Finding",
    ):
        promotion_module._assert_trial_pass(trial, case)


def test_promotion_trial_rejects_non_publishable_extra_finding_even_when_required_passes(
    promotion_module: ModuleType,
) -> None:
    trial, case = _passing_trial()
    metrics = trial.review_result.metrics
    metrics.generated_finding_count = 2
    metrics.matched_finding_count = 2
    metrics.duplicate_finding_count = 1
    metrics.evidence_valid_count = 1
    metrics.evidence_invalid_count = 1
    metrics.evidence_supported_count = 1
    metrics.evidence_unsupported_count = 1
    trial.review_result.finding_outcomes += (
        SimpleNamespace(
            matched_expected_truth_id="issue-required",
            strict_publishable=False,
        ),
    )

    with pytest.raises(promotion_module.CoreRegressionPromotionError):
        promotion_module._assert_trial_pass(trial, case)


def test_promotion_trial_allows_a_clean_case_with_no_findings(
    promotion_module: ModuleType,
) -> None:
    trial, case = _passing_trial()
    metrics = trial.review_result.metrics
    for field in (
        "generated_finding_count",
        "expected_truth_count",
        "required_expected_truth_count",
        "matched_finding_count",
        "matched_expected_truth_count",
        "matched_required_truth_count",
        "evidence_valid_count",
        "evidence_supported_count",
        "strict_publishable_count",
    ):
        setattr(metrics, field, 0)
    trial.review_result.finding_outcomes = ()
    case.review_truth.expected_findings = ()

    promotion_module._assert_trial_pass(trial, case)


def test_verify_case_promotion_checks_human_approval_before_artifact_access(
    promotion_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval_calls: list[tuple[Path, str]] = []
    eval_root = REPOSITORY_ROOT / "eval"

    class FakeCaseBank:
        root = eval_root

    class FakeArtifactStore:
        def load_run_config(self, _run_id: str) -> object:
            raise AssertionError("Run artifacts must not be read before approval")

    class FakeRepositoryPreparer:
        pass

    def reject_unapproved(eval_root: Path, task_id: str) -> None:
        approval_calls.append((eval_root, task_id))
        raise promotion_module.HumanReviewError("missing approval")

    monkeypatch.setattr(promotion_module, "CaseBank", FakeCaseBank)
    monkeypatch.setattr(promotion_module, "ArtifactStore", FakeArtifactStore)
    monkeypatch.setattr(
        promotion_module,
        "RepositoryPreparer",
        FakeRepositoryPreparer,
    )
    monkeypatch.setattr(
        promotion_module,
        "verify_current_case_approval",
        reject_unapproved,
    )

    with pytest.raises(
        promotion_module.CoreRegressionPromotionError,
        match="source-bound human approval",
    ):
        promotion_module.verify_case_promotion(
            case_bank=FakeCaseBank(),
            artifact_store=FakeArtifactStore(),
            repository_preparer=FakeRepositoryPreparer(),
            task_id="core-py-001",
            promotion_evidence={
                "run_id": "run-a",
                "evaluation_id": "eval-a",
                "summary_id": "summary-a",
            },
        )

    assert approval_calls == [(eval_root, "core-py-001")]

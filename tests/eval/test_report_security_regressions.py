from __future__ import annotations

from dataclasses import replace

import pytest

from review_agent_eval.artifacts import ArtifactStore
from review_agent_eval.intent_evaluator import IntentEvaluator
from review_agent_eval.judge import JudgeContextKind, repository_context
from review_agent_eval.metrics import MetricsAggregator, TrialScore
from review_agent_eval.models import TraceRef, TraceType
from review_agent_eval.report import (
    ReportBuilder,
    ReportError,
    RunReportSummary,
    TrialEvaluationSource,
    TrialInspection,
)
from review_agent_eval.review_evaluator import (
    ReviewContextBundle,
    ReviewEvaluator,
    ReviewFindingContextEntry,
)
from tests.eval.test_metrics import (
    TARGET_MATERIALIZATION_ID,
    _case_and_snapshot,
    _completed_submission,
    _score_completed,
    _score_sources,
)
from tests.eval.test_report import _report_sources


class _ForgedReplay:
    def __init__(self, payload) -> None:
        self._payload = payload

    def to_dict(self):
        return self._payload


def test_inspection_replaces_raw_judge_payloads_with_digest_refs() -> None:
    case, replay, execution, config = _score_sources()
    base_submission = _completed_submission(config, 1)
    finding_id = base_submission.review.findings[0].finding_id
    submission = replace(
        base_submission,
        review=replace(
            base_submission.review,
            findings=(
                replace(
                    base_submission.review.findings[0],
                    claim="A semantic claim that requires Judge work.",
                ),
            ),
        ),
    )
    context_content = (
        "FIRST_VALUE = compute_value()\n"
        "SECOND_VALUE = fallback_value()"
    )
    context = repository_context(
        source_id="inspection-env-like-code",
        kind=JudgeContextKind.CODE,
        content=context_content,
        revision="head",
        path="src/app.py",
    )
    context_bundle = ReviewContextBundle.create(
        finding_entries=(
            ReviewFindingContextEntry.create(finding_id, (context,)),
        )
    )
    intent_result = IntentEvaluator().evaluate(
        submission.intent,
        case.intent_truth,
        case.clarification_script,
    )
    review_result = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
        context_bundle=context_bundle,
    ).evaluate(submission, case.review_truth)
    assert review_result.judge_requests

    inspection = ReportBuilder().build_inspection(
        config,
        execution,
        "metrics-eval-v1",
        trial_source=TrialEvaluationSource(
            eval_case=case,
            submission=submission,
            intent_result=intent_result,
            review_result=review_result,
        ),
    )
    payload = inspection.to_dict()

    for projection_name in ("intent_evaluation", "review_evaluation"):
        projection = payload[projection_name]
        assert projection["source_digest"] == (
            intent_result.digest()
            if projection_name == "intent_evaluation"
            else review_result.digest()
        )
        assert "judge_payloads:judge_artifact_refs" in projection["redactions"]
        for field in (
            "judge_requests",
            "judge_decisions",
            "judge_failures",
            "judge_ungraded",
        ):
            assert field not in projection["payload"]

    request_refs = payload["judge_artifact_refs"]["review"]["requests"]
    assert len(request_refs) == 1
    assert request_refs[0]["request_id"] == review_result.judge_requests[0].request_id
    assert request_refs[0]["request_digest"]
    assert request_refs[0]["parent_result_digest"] == review_result.digest()
    assert context_content not in inspection.to_json()


def test_inspection_rejects_an_uninspected_trial_manifest_ref_tampering(
    tmp_path,
) -> None:
    """Inspection must validate every immutable Trial manifest ref in the Run."""

    case, _replay, execution, config, sources = _report_sources()
    _same_case, snapshot, _same_replay = _case_and_snapshot()
    store = ArtifactStore(tmp_path / ".eval-runs")
    run_manifest = store.create_run(config, snapshot)

    manifest_sources = []
    for source in sources:
        plan = next(
            item for item in run_manifest.trials if item.trial_id == source.trial_id
        )
        manifest = store.load_trial_manifest(
            config.run_id,
            case.task_id,
            plan.trial_id,
        )
        manifest_sources.append(replace(source, trial_manifest=manifest))

    inspected = manifest_sources[0]
    uninspected_trial_id = manifest_sources[1].trial_id
    forged_manifest = run_manifest.to_dict()
    forged_plan = next(
        item
        for item in forged_manifest["trials"]
        if item["trial_id"] == uninspected_trial_id
    )
    forged_plan["manifest"]["sha256"] = "1" * 64

    with pytest.raises(ReportError, match="Trial plan|manifest"):
        ReportBuilder().build_inspection(
            config,
            execution,
            "metrics-eval-v1",
            trial_source=inspected,
            run_manifest=forged_manifest,
        )


def test_trace_ref_projection_is_a_versioned_redacted_artifact_wrapper() -> None:
    case, replay, execution, config = _score_sources()
    submission, _intent, _review, _score = _score_completed(
        case,
        replay,
        execution,
        config,
        1,
    )
    trace_ref = TraceRef(
        TraceType.URL,
        "https://user:password@example.test/private-trace",
    )
    submission = replace(submission, trace_ref=trace_ref)
    intent_result = IntentEvaluator().evaluate(
        submission.intent,
        case.intent_truth,
        case.clarification_script,
    )
    review_result = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
    ).evaluate(submission, case.review_truth)
    source = TrialEvaluationSource(
        eval_case=case,
        submission=submission,
        intent_result=intent_result,
        review_result=review_result,
    )

    inspection = ReportBuilder().build_inspection(
        config,
        execution,
        "metrics-eval-v1",
        trial_source=source,
    )
    projected = inspection.to_dict()["trace"]["trace_ref"]

    assert projected["schema_version"] == "eval_redacted_artifact_projection_v1"
    assert projected["artifact_kind"] == "trace_ref"
    assert projected["source_digest"] == trace_ref.digest()
    assert projected["redactions"]
    assert "payload" in projected
    assert "type" not in projected
    assert "value" not in projected
    assert "password" not in inspection.to_json()
    assert "example.test" not in inspection.to_json()


def test_summary_from_dict_rejects_a_forged_builder() -> None:
    case, _replay, execution, config, sources = _report_sources(
        include_second=False
    )
    builder = ReportBuilder()
    summary = builder.build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=sources,
    )
    forged_payload = summary.to_dict()
    forged_payload["coverage"]["planned_trial_count"] += 100

    class ForgedBuilder:
        def build_summary(self, **_kwargs):
            return _ForgedReplay(forged_payload)

    with pytest.raises((TypeError, ValueError)):
        RunReportSummary.from_dict(
            forged_payload,
            builder=ForgedBuilder(),
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision="metrics-eval-v1",
            eval_cases=(case,),
            trial_sources=sources,
        )


def test_inspection_from_dict_rejects_a_forged_builder() -> None:
    case, _replay, execution, config, sources = _report_sources(
        include_second=False
    )
    builder = ReportBuilder()
    inspection = builder.build_inspection(
        config,
        execution,
        "metrics-eval-v1",
        trial_source=sources[0],
    )
    forged_payload = inspection.to_dict()
    forged_payload["source_bindings"]["task_id"] = "forged-task"

    class ForgedBuilder:
        def build_inspection(self, **_kwargs):
            return _ForgedReplay(forged_payload)

    with pytest.raises((TypeError, ValueError)):
        TrialInspection.from_dict(
            forged_payload,
            builder=ForgedBuilder(),
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision="metrics-eval-v1",
            trial_source=sources[0],
        )


def test_trial_score_from_dict_rejects_a_forged_scorer() -> None:
    case, replay, execution, config = _score_sources()
    submission, intent_result, review_result, score = _score_completed(
        case,
        replay,
        execution,
        config,
        1,
    )
    forged_payload = score.to_dict()
    forged_payload["score_id"] = "forged-score"

    class ForgedScorer:
        def score(self, **_kwargs):
            return _ForgedReplay(forged_payload)

    with pytest.raises((TypeError, ValueError)):
        TrialScore.from_dict(
            forged_payload,
            scorer=ForgedScorer(),
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision="metrics-eval-v1",
            eval_case=case,
            submission=submission,
            trial_index=1,
            intent_result=intent_result,
            review_result=review_result,
        )


def test_metrics_aggregator_rejects_instance_method_shadowing() -> None:
    aggregator = MetricsAggregator()

    with pytest.raises(AttributeError):
        aggregator.aggregate_case = lambda *_args, **_kwargs: _ForgedReplay({})


def test_builder_object_mutation_cannot_bypass_source_bound_hydration() -> None:
    case, _replay, execution, config, sources = _report_sources(
        include_second=False
    )
    trusted = ReportBuilder()
    summary = trusted.build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=sources,
    )
    inspection = trusted.build_inspection(
        config,
        execution,
        "metrics-eval-v1",
        trial_source=sources[0],
    )
    mutated = ReportBuilder()
    object.__setattr__(mutated, "scorer", _ForgedReplay({}))

    with pytest.raises(ReportError, match="TrialScorer was mutated"):
        RunReportSummary.from_dict(
            summary.to_dict(),
            builder=mutated,
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision="metrics-eval-v1",
            eval_cases=(case,),
            trial_sources=sources,
        )
    with pytest.raises(ReportError, match="TrialScorer was mutated"):
        TrialInspection.from_dict(
            inspection.to_dict(),
            builder=mutated,
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision="metrics-eval-v1",
            trial_source=sources[0],
        )


def test_pending_and_missing_evaluator_outputs_remain_distinct() -> None:
    case, replay, execution, config = _score_sources(trial_count=2)

    pending_base = _completed_submission(config, 1)
    pending_submission = replace(
        pending_base,
        review=replace(
            pending_base.review,
            findings=(
                replace(
                    pending_base.review.findings[0],
                    claim="A semantic claim that requires Judge work.",
                ),
            ),
        ),
    )
    pending_intent = IntentEvaluator().evaluate(
        pending_submission.intent,
        case.intent_truth,
        case.clarification_script,
    )
    pending_review = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=pending_submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
    ).evaluate(pending_submission, case.review_truth)

    missing_submission, missing_intent, _review, _full_score = _score_completed(
        case,
        replay,
        execution,
        config,
        2,
    )
    missing_score = ReportBuilder().scorer.score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=missing_submission,
        trial_index=2,
        intent_result=missing_intent,
        review_result=None,
    )
    summary = ReportBuilder().build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=(
            TrialEvaluationSource(
                eval_case=case,
                submission=pending_submission,
                intent_result=pending_intent,
                review_result=pending_review,
            ),
            TrialEvaluationSource(
                eval_case=case,
                submission=missing_submission,
                intent_result=missing_intent,
                trial_score=missing_score,
            ),
        ),
    )

    pending_ids = {
        item["trial_id"] for item in summary.diagnostics["pending_evaluations"]
    }
    missing_ids = {
        item["trial_id"] for item in summary.diagnostics["missing_evaluations"]
    }
    assert pending_submission.trial_id in pending_ids
    assert pending_submission.trial_id not in missing_ids
    assert missing_submission.trial_id in missing_ids
    assert missing_submission.trial_id not in pending_ids

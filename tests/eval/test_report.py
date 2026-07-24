from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

import review_agent_eval
import review_agent_eval.metrics as metrics_module
import review_agent_eval.report as report_module
from review_agent_eval.artifacts import (
    ArtifactRef,
    ArtifactStore,
    StageName,
    StageReceipt,
)
from review_agent_eval.cases import RunCaseSnapshot, SuiteManifest
from review_agent_eval.intent_evaluator import IntentEvaluator
from review_agent_eval.metrics import TrialScorer
from review_agent_eval.models import (
    NovelFindingPolicy,
    TraceRef,
    TraceType,
    TrialStatus,
    TruthCompleteness,
)
from review_agent_eval.review_evaluator import ReviewEvaluator
from review_agent_eval.report import (
    ReportBuilder,
    ReportError,
    RunReportSummary,
    TrialEvaluationSource,
    TrialInspection,
    render_run_markdown,
    render_trial_markdown,
)
from review_agent_eval.metrics_exports import METRICS_PUBLIC_NAMES
from review_agent_eval.report_exports import REPORT_PUBLIC_NAMES
from tests.eval.test_metrics import (
    TARGET_MATERIALIZATION_ID,
    _failed_submission,
    _case_and_snapshot,
    _completed_submission,
    _partial_failed_submission,
    _score_completed,
    _score_sources,
)
from tests.eval.test_config import run_config
from tests.eval.test_review_evaluator import _run_scripted_judge


def _report_sources(*, include_second: bool = True):
    case, replay, execution, config = _score_sources(trial_count=2)
    first_submission, first_intent, first_review, first_score = _score_completed(
        case,
        replay,
        execution,
        config,
        1,
    )
    sources = [
        TrialEvaluationSource(
            eval_case=case,
            submission=first_submission,
            intent_result=first_intent,
            review_result=first_review,
            trial_score=first_score,
        )
    ]
    if include_second:
        second_submission = _failed_submission(config, 2)
        second_score = TrialScorer().score(
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision="metrics-eval-v1",
            eval_case=case,
            submission=second_submission,
            trial_index=2,
            intent_result=None,
            review_result=None,
        )
        sources.append(
            TrialEvaluationSource(
                eval_case=case,
                submission=second_submission,
                trial_score=second_score,
            )
        )
    return case, replay, execution, config, tuple(sources)


def test_package_root_exports_match_metrics_and_report_public_contracts() -> None:
    assert set(METRICS_PUBLIC_NAMES) == set(metrics_module.__all__)
    assert set(REPORT_PUBLIC_NAMES) == set(report_module.__all__)
    assert review_agent_eval.TrialScorer is TrialScorer
    assert review_agent_eval.ReportBuilder is ReportBuilder


def test_summary_round_trip_is_sealed_and_source_bound() -> None:
    case, _replay, execution, config, sources = _report_sources()
    builder = ReportBuilder()
    summary = builder.build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=sources,
    )

    hydrated = RunReportSummary.from_json(
        summary.to_json(),
        builder=builder,
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=sources,
    )
    assert hydrated.to_json() == summary.to_json()
    with pytest.raises(TypeError, match="ReportBuilder"):
        RunReportSummary._seal(summary.to_dict())
    with pytest.raises(TypeError, match="ReportBuilder"):
        replace(summary, summary_id="forged")

    forged = summary.to_dict()
    forged["partitions"][0]["aggregate_score"]["metrics"][0]["numerator"] = 0
    with pytest.raises(ReportError, match="source-bound replay"):
        RunReportSummary.from_dict(
            forged,
            builder=builder,
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision="metrics-eval-v1",
            eval_cases=(case,),
            trial_sources=sources,
        )


def test_summary_keeps_planned_terminal_and_score_coverage_distinct() -> None:
    case, _replay, execution, config, sources = _report_sources(
        include_second=False
    )
    summary = ReportBuilder().build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=sources,
    )
    coverage = summary.coverage
    assert coverage["planned_case_count"] == 1
    assert coverage["planned_trial_count"] == 2
    assert coverage["represented_trial_count"] == 1
    assert coverage["terminal_submission_count"] == 1
    assert coverage["trial_score_count"] == 1
    assert len(coverage["nonterminal_trial_ids"]) == 1
    assert coverage["unevaluated_terminal_trial_ids"] == []
    assert summary.diagnostics["nonterminal_trials"][0]["reason"] == "trial_source_missing"
    assert summary.diagnostics["critical_high_misses"] == []


def test_nonterminal_trial_has_explicit_harness_diagnostic() -> None:
    case, _replay, execution, config, sources = _report_sources(
        include_second=False
    )
    nonterminal = TrialEvaluationSource(eval_case=case, trial_index=2)
    summary = ReportBuilder().build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=(*sources, nonterminal),
    )
    assert summary.coverage["nonterminal_trial_ids"] == [
        config.trial_id(case.task_id, 2)
    ]
    assert summary.coverage["represented_trial_count"] == 2
    assert summary.diagnostics["nonterminal_trials"][0]["reason"] == "terminal_submission_missing"
    assert summary.diagnostics["agent_failures"] == []
    assert summary.diagnostics["critical_high_misses"] == []


def test_trace_capture_metadata_is_bounded_and_self_consistent() -> None:
    case, _replay, execution, config, sources = _report_sources(
        include_second=False
    )
    with pytest.raises(ReportError, match="total_bytes"):
        replace(
            sources[0],
            trace_capture={
                "captured": True,
                "total_bytes": 11,
                "files": [
                    {
                        "path": "trace.json",
                        "size_bytes": 12,
                        "sha256": "d" * 64,
                        "content_truncated": False,
                    }
                ],
            },
        )
    with pytest.raises(ReportError, match="uncaptured"):
        replace(
            sources[0],
            trace_capture={
                "captured": False,
                "total_bytes": 1,
                "files": [
                    {
                        "path": "trace.json",
                        "size_bytes": 1,
                        "sha256": "d" * 64,
                        "content_truncated": False,
                    }
                ],
            },
        )
    file_url_source = replace(
        sources[0],
        trace_capture={
            "captured": True,
            "total_bytes": 1,
            "files": [
                {
                    "path": "file:///C:/private/trace.json",
                    "size_bytes": 1,
                    "sha256": "d" * 64,
                    "content_truncated": False,
                }
            ],
        },
    )
    inspection = ReportBuilder().build_inspection(
        config,
        execution,
        "metrics-eval-v1",
        trial_source=file_url_source,
    )
    assert inspection.trace["capture"]["files"][0]["path"] == "<absolute-path-redacted>"


def test_markdown_is_pure_deterministic_and_shows_metric_coverage() -> None:
    case, _replay, execution, config, sources = _report_sources()
    summary = ReportBuilder().build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=sources,
    )

    first = render_run_markdown(summary)
    second = render_run_markdown(summary)
    assert first == second
    assert first.endswith("\n")
    assert "issue_f1" in first
    assert "derived coverage" in first
    assert "Numerator" in first
    assert "Coverage" in first
    assert "failure_as_miss_count" in first
    assert "missing_count" in first
    assert "Target kind" in first
    assert "Wire contract digest" in first
    assert "Adapter capabilities digest" in first
    assert "Isolation profile" in first
    assert "Metric authority profile" in first
    assert "Metric authority policy digest" in first
    assert "Authority coverage" in first
    assert "location_precision_excluded_truth_count" in first
    partition = summary.partitions[0]
    assert partition["authority_coverage"] == partition["aggregate_score"][
        "authority_coverage"
    ]
    assert "Critical/high required misses" in first
    assert "overall_score" not in first.lower()
    assert "overall score" not in first.lower()


def test_failure_and_diagnostic_indexes_are_visible_in_summary() -> None:
    case, _replay, execution, config, sources = _report_sources()
    summary = ReportBuilder().build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=sources,
    )
    diagnostics = summary.diagnostics
    assert len(diagnostics["agent_failures"]) == 1
    assert diagnostics["agent_failures"][0]["failure_code"] == "timeout"
    assert len(diagnostics["critical_high_misses"]) == 1
    assert diagnostics["critical_high_misses"][0]["source_status"] == "failure_as_miss"
    assert diagnostics["usage_missing_trial_ids"]["cost_amount"]
    assert diagnostics["usage_missing_coverage"][0]["usage"]["cost"]["missing_count"] == 2


def test_failed_partial_evaluation_is_scored_per_phase_not_dropped() -> None:
    case, replay, execution, config, _sources = _report_sources(
        include_second=False
    )
    submission = _partial_failed_submission(config, 1)
    review_result = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
    ).evaluate(submission, case.review_truth)
    score = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=submission,
        trial_index=1,
        intent_result=None,
        review_result=review_result,
    )
    source = TrialEvaluationSource(
        eval_case=case,
        submission=submission,
        review_result=review_result,
        trial_score=score,
    )
    summary = ReportBuilder().build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=(source,),
    )
    assert summary.coverage["trial_score_count"] == 1
    assert summary.coverage["review_scored_trial_count"] == 1
    assert summary.coverage["intent_scored_trial_count"] == 0
    assert summary.coverage["unevaluated_terminal_trial_ids"] == []
    assert summary.diagnostics["missing_evaluations"] == [
        {
            "task_id": case.task_id,
            "trial_id": submission.trial_id,
            "phase": "intent",
            "status": "missing_evaluation",
        }
    ]


def test_completed_submission_with_missing_phase_is_reported_and_paths_are_redacted() -> None:
    case, _replay, execution, config = _score_sources()
    submission = _completed_submission(config, 1)
    submission = replace(
        submission,
        review=replace(
            submission.review,
            findings=(
                replace(
                    submission.review.findings[0],
                    path="C:\\private\\app.py",
                ),
            ),
        ),
    )
    intent_result = IntentEvaluator().evaluate(
        submission.intent,
        case.intent_truth,
        case.clarification_script,
    )
    score = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=submission,
        trial_index=1,
        intent_result=intent_result,
        review_result=None,
    )
    source = TrialEvaluationSource(
        eval_case=case,
        submission=submission,
        intent_result=intent_result,
        trial_score=score,
    )
    builder = ReportBuilder()
    summary = builder.build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=(source,),
    )
    inspection = builder.build_inspection(
        config,
        execution,
        "metrics-eval-v1",
        trial_source=source,
    )
    assert summary.coverage["trial_score_count"] == 1
    assert summary.diagnostics["missing_evaluations"][0]["phase"] == "review"
    projected_path = inspection.submission["payload"]["review"]["findings"][0]["path"]
    assert projected_path.startswith("path-ref-")
    assert "private" not in inspection.to_json()


def test_trial_inspection_reuses_canonical_evaluator_payload_and_is_sealed() -> None:
    case, _replay, execution, config, sources = _report_sources(
        include_second=False
    )
    source = sources[0]
    receipt = StageReceipt.create(
        run_id=config.run_id,
        task_id=case.task_id,
        trial_id=sources[0].trial_id,
        stage=StageName.AGENT,
        config_digest="a" * 64,
        attempt=1,
        artifacts=(
            ArtifactRef(
                relative_path="x/submission.json",
                sha256="c" * 64,
                size_bytes=1,
            ),
        ),
        terminal_status=TrialStatus.COMPLETED,
    )
    source = replace(
        source,
        timeline=(receipt,),
        trace_capture={
            "captured": True,
            "reason": "bounded",
            "total_bytes": 12,
            "files": [
                {
                    "path": "C:\\private\\trace.json",
                    "size_bytes": 12,
                    "sha256": "b" * 64,
                    "content_truncated": False,
                }
            ],
        },
    )
    builder = ReportBuilder()
    inspection = builder.build_inspection(
        config,
        execution,
        "metrics-eval-v1",
        trial_source=source,
    )
    payload = inspection.to_dict()
    assert payload["input"]["payload"]["task_id"] == case.task_id
    assert payload["intent_evaluation"]["source_digest"] == source.intent_result.digest()
    assert payload["intent_evaluation"]["payload"]["status"] == source.intent_result.status.value
    assert payload["review_evaluation"]["source_digest"] == source.review_result.digest()
    assert payload["review_evaluation"]["payload"]["assignments"] == [
        item.to_dict() for item in source.review_result.assignments
    ]
    assert payload["timeline"][0]["stage"] == "agent"
    assert payload["score"]["artifact_kind"] == "trial_score"
    assert payload["score"]["source_digest"] == source.trial_score.digest()
    assert "score_id" not in payload["score"]["payload"]
    assert payload["submission"]["artifact_kind"] == "eval_submission"
    assert payload["submission"]["source_digest"] == source.submission.digest()
    assert payload["trace"]["capture"]["files"][0]["path"] == "<absolute-path-redacted>"
    assert "private" not in inspection.to_json()
    markdown = render_trial_markdown(inspection)
    assert "Judge receipts" in markdown
    assert "Evidence diagnostics" in markdown
    assert "Trace metadata" in markdown
    assert "private" not in markdown

    hydrated = TrialInspection.from_json(
        inspection.to_json(),
        builder=builder,
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        trial_source=source,
    )
    assert hydrated.to_json() == inspection.to_json()
    with pytest.raises(TypeError, match="ReportBuilder"):
        TrialInspection._seal(inspection.to_dict())
    with pytest.raises(TypeError, match="ReportBuilder"):
        replace(inspection, inspection_id="forged")


@pytest.mark.parametrize(
    "trace_ref",
    [
        TraceRef(TraceType.OPAQUE_ID, "C:\\private\\trace.json"),
        TraceRef(TraceType.LOCAL_PATH, "C:\\private\\trace.json"),
        TraceRef(TraceType.URL, "https://user:password@example.test/trace"),
    ],
)
def test_inspection_redacts_non_opaque_trace_refs(trace_ref) -> None:
    case, replay, execution, config = _score_sources()
    submission, _intent_result, _review_result, _score = _score_completed(
        case, replay, execution, config, 1
    )
    submission = replace(submission, trace_ref=trace_ref)
    # Re-evaluate against the changed immutable Submission so the score binding
    # remains source-bound.
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
    score = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=submission,
        trial_index=1,
        intent_result=intent_result,
        review_result=review_result,
    )
    source = TrialEvaluationSource(
        eval_case=case,
        submission=submission,
        intent_result=intent_result,
        review_result=review_result,
        trial_score=score,
    )
    inspection = ReportBuilder().build_inspection(
        config,
        execution,
        "metrics-eval-v1",
        trial_source=source,
    )
    summary = ReportBuilder().build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=(source,),
    )
    projected = inspection.to_dict()["trace"]["trace_ref"]
    assert projected["schema_version"] == "eval_redacted_artifact_projection_v1"
    assert projected["artifact_kind"] == "trace_ref"
    assert projected["source_schema_version"] == "eval_trace_ref_v1"
    assert projected["source_digest"] == trace_ref.digest()
    assert projected["payload"]["type"] == "opaque_id"
    assert projected["payload"]["value"].startswith("trace-ref-")
    assert "type" not in projected
    assert "value" not in projected
    assert "private" not in inspection.to_json()
    assert "password" not in inspection.to_json()
    assert "private" not in summary.to_json()
    assert "password" not in summary.to_json()
    assert "private" not in render_trial_markdown(inspection)
    assert "password" not in render_run_markdown(summary)


def test_inspection_tampering_is_rejected_by_source_replay() -> None:
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
    forged = inspection.to_dict()
    forged["judge_artifact_refs"]["review"]["decisions"].append(
        {"request_id": "forged"}
    )
    with pytest.raises(ReportError, match="source-bound replay"):
        TrialInspection.from_dict(
            forged,
            builder=builder,
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision="metrics-eval-v1",
            trial_source=sources[0],
        )


def test_run_and_trial_manifests_are_cross_bound_to_report_sources(tmp_path) -> None:
    case, _replay, execution, config, sources = _report_sources()
    _same_case, snapshot, _same_replay = _case_and_snapshot()
    store = ArtifactStore(tmp_path / ".eval-runs")
    run_manifest = store.create_run(config, snapshot)
    manifest_sources = []
    for source_item in sources:
        source_plan = next(
            item
            for item in run_manifest.trials
            if item.trial_id == source_item.trial_id
        )
        source_manifest = store.load_trial_manifest(
            config.run_id,
            case.task_id,
            source_plan.trial_id,
        )
        manifest_sources.append(
            replace(source_item, trial_manifest=source_manifest)
        )
    source = manifest_sources[0]
    trial_manifest = source.trial_manifest
    builder = ReportBuilder()
    inspection = builder.build_inspection(
        config,
        execution,
        "metrics-eval-v1",
        trial_source=source,
        run_manifest=run_manifest,
    )
    assert inspection.trial_manifest["initial_evaluator_execution_digest"] == run_manifest.initial_evaluator_execution_digest
    summary = builder.build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=tuple(manifest_sources),
        run_manifest=run_manifest,
    )
    assert summary.source_bindings["run_manifest_digest"] == run_manifest.digest()

    forged_run = run_manifest.to_dict()
    forged_run["agent_config_digest"] = "0" * 64
    with pytest.raises(ReportError, match="RunManifest does not bind"):
        builder.build_summary(
            config,
            execution,
            "metrics-eval-v1",
            eval_cases=(case,),
            trial_sources=tuple(manifest_sources),
            run_manifest=forged_run,
        )

    forged_plan_run = run_manifest.to_dict()
    forged_plan_run["trials"][0]["manifest"]["sha256"] = "1" * 64
    with pytest.raises(ReportError, match="Trial plan differs"):
        builder.build_summary(
            config,
            execution,
            "metrics-eval-v1",
            eval_cases=(case,),
            trial_sources=tuple(manifest_sources),
            run_manifest=forged_plan_run,
        )

    forged_trial = trial_manifest.to_dict()
    forged_trial["initial_evaluator_execution_digest"] = "0" * 64
    forged_source = replace(sources[0], trial_manifest=forged_trial)
    with pytest.raises(ReportError, match="initial evaluator binding"):
        builder.build_inspection(
            config,
            execution,
            "metrics-eval-v1",
            trial_source=forged_source,
            run_manifest=run_manifest,
        )


def test_incompatible_case_scores_are_partitioned_without_cross_rollup() -> None:
    first_case, _first_snapshot, replay = _case_and_snapshot()
    second_case = replace(
        first_case,
        task_id="task-review-truth-2",
        source=replace(
            first_case.source,
            source_id="metrics-source-2",
            content_hash="7" * 64,
        ),
        review_truth=replace(
            first_case.review_truth,
            completeness=TruthCompleteness.EXPERT_AUGMENTED,
            novel_finding_policy=NovelFindingPolicy.VERIFY,
        ),
    )
    cases = (first_case, second_case)
    records = []
    for index, case in enumerate(cases, start=1):
        raw = case.to_json().encode("utf-8")
        records.append(
            {
                "task_id": case.task_id,
                "case_version": case.case_version,
                "path": f"cases/partition-{index}.json",
                "split": "regression",
                "protocol_id": "native_repository",
                "dimensions": [
                    {"name": "language", "value": "python"},
                    {"name": "pr_size", "value": "small"},
                ],
                "raw_file_size_bytes": len(raw),
                "raw_file_sha256": hashlib.sha256(raw).hexdigest(),
                "canonical_case_digest": case.digest(),
                "eval_input_digest": case.eval_input().digest(),
                "truth_completeness": case.review_truth.completeness.value,
            }
        )
    manifest = SuiteManifest.from_dict(
        {
            "schema_version": "suite_manifest_v2",
            "suite_id": "metrics-suite",
            "suite_version": "v1",
            "wire_contract": {
                "case_schema_version": "eval_case_v2",
                "input_schema_version": "eval_input_v2",
                "submission_schema_version": "eval_submission_v2",
                "review_target_kind": "repository",
                "materializer_protocol": "repository-materializer-v2",
            },
            "source": {
                "kind": "core",
                "source_id": "metrics-partition-source",
                "source_version": "v1",
                "source_uri": None,
                "license": None,
                "content_hash": "6" * 64,
                "preparation_binding": None,
            },
            "cases": records,
        }
    )
    by_task = {case.task_id: case for case in cases}
    snapshot = RunCaseSnapshot.build(
        manifest,
        tuple((item, by_task[item.task_id]) for item in manifest.cases),
    )
    _case, _replay, execution, _old_config, _sources = _report_sources(
        include_second=False
    )
    config = run_config(snapshot, evaluator=execution.evaluator, trial_count=1)
    base_submission = _score_completed(
        first_case,
        replay,
        execution,
        config,
        1,
    )[0]
    submissions = (
        base_submission,
        replace(
            base_submission,
            task_id=second_case.task_id,
            eval_input_digest=second_case.eval_input().digest(),
            trial_id=config.trial_id(second_case.task_id, 1),
        ),
    )
    sources = []
    for case, submission in zip(cases, submissions):
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
        score = TrialScorer().score(
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision="metrics-eval-v1",
            eval_case=case,
            submission=submission,
            trial_index=1,
            intent_result=intent_result,
            review_result=review_result,
        )
        sources.append(
            TrialEvaluationSource(
                eval_case=case,
                submission=submission,
                intent_result=intent_result,
                review_result=review_result,
                trial_score=score,
            )
        )

    summary = ReportBuilder().build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=cases,
        trial_sources=tuple(sources),
        group_dimension_names=("language", "pr_size"),
    )
    assert len(summary.partitions) == 2
    assert {
        item["compatibility"]["truth_completeness"]
        for item in summary.partitions
    } == {"closed_world", "expert_augmented"}
    assert all(item["aggregate_score"]["case_count"] == 1 for item in summary.partitions)
    assert all(
        item["groupings"][0]["dimension_names"] == ["language", "pr_size"]
        for item in summary.partitions
    )


def test_novel_disallowed_is_separate_from_fabricated_diagnostics() -> None:
    case, replay, execution, config = _score_sources()
    submission = _completed_submission(config, 1)
    submission = replace(
        submission,
        review=replace(
            submission.review,
            findings=(
                replace(
                    submission.review.findings[0],
                    claim="A distinct defect that is not the expected issue.",
                ),
            ),
        ),
    )
    intent_result = IntentEvaluator().evaluate(
        submission.intent,
        case.intent_truth,
        case.clarification_script,
    )
    evaluator = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
    )
    pending = evaluator.evaluate(submission, case.review_truth)
    decision = _run_scripted_judge(
        pending.judge_requests[0],
        execution,
        relation="different",
        score_ppm=0,
        severity_assessment="consistent",
        actionability="actionable",
    )
    review_result = evaluator.evaluate(
        submission,
        case.review_truth,
        judge_results=(decision,),
    )
    score = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=submission,
        trial_index=1,
        intent_result=intent_result,
        review_result=review_result,
    )
    source = TrialEvaluationSource(
        eval_case=case,
        submission=submission,
        intent_result=intent_result,
        review_result=review_result,
        trial_score=score,
    )
    summary = ReportBuilder().build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=(source,),
    )
    assert summary.diagnostics["fabricated_findings"] == []
    assert len(summary.diagnostics["novel_disallowed_findings"]) == 1
    assert "Novel Findings disallowed by policy" in render_run_markdown(summary)


def test_pending_evaluator_work_is_visible_and_not_agent_failure() -> None:
    case, replay, execution, config = _score_sources()
    submission = _completed_submission(config, 1)
    submission = replace(
        submission,
        review=replace(
            submission.review,
            findings=(
                replace(
                    submission.review.findings[0],
                    claim="A semantic claim that requires Judge work.",
                ),
            ),
        ),
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
    ).evaluate(submission, case.review_truth)
    source = TrialEvaluationSource(
        eval_case=case,
        submission=submission,
        intent_result=intent_result,
        review_result=review_result,
    )
    summary = ReportBuilder().build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=(source,),
    )
    assert summary.coverage["trial_score_count"] == 1
    assert summary.coverage["review_scored_trial_count"] == 0
    assert summary.coverage["intent_scored_trial_count"] == 1
    assert summary.coverage["unevaluated_terminal_trial_ids"] == []
    assert summary.diagnostics["pending_evaluations"][0]["phase"] == "review"
    assert summary.diagnostics["agent_failures"] == []

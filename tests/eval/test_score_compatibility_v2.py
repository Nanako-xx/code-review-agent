from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from review_agent_eval.cases import RunCaseSnapshot, SuiteManifest
from review_agent_eval.config import EvalRunConfig, SuiteRunConfig
from review_agent_eval.metrics import (
    CoreMetric,
    MetricSourceStatus,
    MetricsAggregator,
    TrialScorer,
)
from review_agent_eval.models import (
    DiffSide,
    EvalCase,
    EvalCaseInput,
    ExpectedFinding,
    FindingSeverity,
    FrozenContextReviewTarget,
    MetricAuthority,
    MetricAuthoritySource,
    RequiredContextLevel,
    ReviewTargetKind,
    TruthLocation,
)
from review_agent_eval.report import (
    ReportBuilder,
    TrialEvaluationSource,
    render_run_markdown,
)
from tests.eval.test_config import run_config
from tests.eval.test_metrics import (
    _case_and_snapshot,
    _failed_submission,
    _score_sources,
)
from tests.eval.test_review_truth_completeness import _execution


CORE_AUTHORITY = MetricAuthority(
    severity_scorable=True,
    severity_authority=MetricAuthoritySource.EXPERT_ANNOTATION,
    location_scorable=True,
    location_authority=MetricAuthoritySource.EXPERT_ANNOTATION,
)
AACR_AUTHORITY = MetricAuthority(
    severity_scorable=False,
    severity_authority=None,
    location_scorable=True,
    location_authority=MetricAuthoritySource.UPSTREAM_ANNOTATION,
)
SWE_AUTHORITY = MetricAuthority(
    severity_scorable=False,
    severity_authority=None,
    location_scorable=False,
    location_authority=None,
)
AUTHORITY_METRICS = (
    CoreMetric.SEVERITY_WEIGHTED_RECALL,
    CoreMetric.CRITICAL_HIGH_MISS_COUNT,
    CoreMetric.LINE_PRECISION,
    CoreMetric.LINE_RECALL,
)


def _truth(
    truth_id: str,
    authority: MetricAuthority,
    *,
    required: bool = True,
) -> ExpectedFinding:
    return ExpectedFinding(
        truth_id=truth_id,
        claim=f"Expected defect {truth_id}.",
        severity=(FindingSeverity.HIGH if authority.severity_scorable else None),
        category="correctness",
        required=required,
        metric_authority=authority,
        locations=(
            (
                TruthLocation(
                    path="src/app.py",
                    side=DiffSide.RIGHT,
                    from_line=1,
                    to_line=1,
                ),
            )
            if authority.location_scorable
            else ()
        ),
        evidence_anchors=(),
        required_context_level=RequiredContextLevel.DIFF,
        rationale="authority compatibility fixture",
    )


def _failed_source(
    case: EvalCase,
    execution,
    config: EvalRunConfig,
) -> TrialEvaluationSource:
    submission = _failed_submission(config, 1)
    suite_case = config.suite.case(case.task_id)
    if submission.task_id != case.task_id:
        submission = replace(
            submission,
            task_id=case.task_id,
            trial_id=config.trial_id(case.task_id, 1),
            eval_input_digest=suite_case.eval_input_digest,
        )
    score = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=submission,
        trial_index=1,
        intent_result=None,
        review_result=None,
    )
    return TrialEvaluationSource(
        eval_case=case,
        submission=submission,
        trial_score=score,
    )


def _snapshot_for_cases(cases: tuple[EvalCase, ...]) -> RunCaseSnapshot:
    target_kind = cases[0].input.review_target.kind
    assert all(case.input.review_target.kind is target_kind for case in cases)
    frozen = target_kind is ReviewTargetKind.FROZEN_CONTEXT
    records = []
    for index, case in enumerate(cases, start=1):
        raw = case.to_json().encode("utf-8")
        records.append(
            {
                "task_id": case.task_id,
                "case_version": case.case_version,
                "path": f"cases/compatibility-{index}.json",
                "split": "regression",
                "protocol_id": (
                    "frozen_context_protocol" if frozen else "native_repository"
                ),
                "dimensions": [{"name": "language", "value": "python"}],
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
            "suite_id": cases[0].source.suite,
            "suite_version": "v1",
            "wire_contract": {
                "case_schema_version": "eval_case_v2",
                "input_schema_version": "eval_input_v2",
                "submission_schema_version": "eval_submission_v2",
                "review_target_kind": target_kind.value,
                "materializer_protocol": (
                    "frozen-context-materializer-v2"
                    if frozen
                    else "repository-materializer-v2"
                ),
            },
            "source": {
                "kind": "core",
                "source_id": "compatibility-suite-source",
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
    return RunCaseSnapshot.build(
        manifest,
        tuple((item, by_task[item.task_id]) for item in manifest.cases),
    )


def test_authority_profile_limit_applies_after_canonical_deduplication() -> None:
    findings = tuple(
        _truth(f"duplicate-authority-{index}", SWE_AUTHORITY)
        for index in range(17)
    )
    case, _replay, execution, config = _score_sources(review_findings=findings)

    score = _failed_source(case, execution, config).trial_score

    assert len(case.review_truth.expected_findings) == 17
    assert len(score.compatibility.metric_authority_profile.authorities) == 1
    assert score.authority_coverage.expected_truth_count == 17


@pytest.mark.parametrize(
    ("findings", "expected", "not_scorable", "profile_size"),
    (
        (
            (_truth("core", CORE_AUTHORITY),),
            (1, 1, 1, 0, 1, 0, 1, 0),
            (),
            1,
        ),
        (
            (_truth("aacr", AACR_AUTHORITY),),
            (1, 1, 0, 1, 1, 0, 1, 0),
            (
                CoreMetric.SEVERITY_WEIGHTED_RECALL,
                CoreMetric.CRITICAL_HIGH_MISS_COUNT,
            ),
            1,
        ),
        (
            (_truth("swe", SWE_AUTHORITY),),
            (1, 1, 0, 1, 0, 1, 0, 1),
            AUTHORITY_METRICS,
            1,
        ),
        (
            (
                _truth("mixed-core", CORE_AUTHORITY),
                _truth("mixed-aacr", AACR_AUTHORITY),
                _truth("mixed-swe", SWE_AUTHORITY, required=False),
            ),
            (3, 2, 1, 1, 2, 1, 2, 0),
            (),
            3,
        ),
    ),
    ids=("core", "aacr", "swe", "mixed"),
)
def test_authority_coverage_matrix_is_case_based_and_not_scorable_is_metric_based(
    findings,
    expected,
    not_scorable,
    profile_size,
) -> None:
    case, _replay, execution, config = _score_sources(review_findings=findings)
    source = _failed_source(case, execution, config)
    trial = source.trial_score
    aggregator = MetricsAggregator()
    case_score = aggregator.aggregate_case((trial,))
    aggregate = aggregator.aggregate_cases((case_score,), source_trials=(trial,))
    fields = tuple(trial.authority_coverage.__dataclass_fields__)

    assert tuple(getattr(trial.authority_coverage, name) for name in fields) == expected
    assert case_score.authority_coverage == trial.authority_coverage
    assert aggregate.authority_coverage == case_score.authority_coverage
    assert len(trial.compatibility.metric_authority_profile.authorities) == profile_size
    for metric in AUTHORITY_METRICS:
        expected_count = int(metric in not_scorable)
        assert (
            trial.contribution(metric).source_status is MetricSourceStatus.NOT_SCORABLE
        ) is bool(expected_count)
        assert case_score.metric(metric).coverage.not_scorable_count == expected_count
        assert aggregate.metric(metric).coverage.not_scorable_count == expected_count


@pytest.mark.parametrize(
    ("different_profile", "expected_partition_count"),
    ((False, 1), (True, 2)),
)
def test_report_partitions_only_when_authority_profiles_differ(
    different_profile: bool,
    expected_partition_count: int,
) -> None:
    first_case, _snapshot, _replay = _case_and_snapshot(
        review_findings=(_truth("profile-core", CORE_AUTHORITY),)
    )
    second_truth = (
        (_truth("profile-swe", SWE_AUTHORITY),)
        if different_profile
        else first_case.review_truth.expected_findings
    )
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
            expected_findings=second_truth,
        ),
    )
    cases = (first_case, second_case)
    snapshot = _snapshot_for_cases(cases)
    execution = _execution()
    config = run_config(snapshot, evaluator=execution.evaluator, trial_count=1)
    sources = tuple(_failed_source(case, execution, config) for case in cases)

    summary = ReportBuilder().build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=cases,
        trial_sources=sources,
    )

    assert len(summary.partitions) == expected_partition_count
    assert {
        item["compatibility"]["truth_completeness"]
        for item in summary.partitions
    } == {first_case.review_truth.completeness.value}
    assert {
        item["compatibility"]["novel_finding_policy"]
        for item in summary.partitions
    } == {first_case.review_truth.novel_finding_policy.value}
    if different_profile:
        assert len(
            {
                item["compatibility"]["metric_authority_profile_digest"]
                for item in summary.partitions
            }
        ) == 2
        assert all(item["aggregate_score"]["case_count"] == 1 for item in summary.partitions)
    else:
        partition = summary.partitions[0]
        assert partition["aggregate_score"]["case_count"] == 2
        assert partition["authority_coverage"]["expected_truth_count"] == 2


def test_real_scores_with_different_isolation_profiles_cannot_roll_up() -> None:
    case, _replay, execution, config = _score_sources(
        review_findings=(_truth("isolation", SWE_AUTHORITY),)
    )
    isolated_capabilities = replace(
        config.adapter_capabilities,
        isolation_profile="repository-process-isolation-v9",
    )
    isolated_config = EvalRunConfig.create(
        run_instance_key=config.run_instance_key,
        agent=config.agent,
        clarification_matcher=config.clarification_matcher,
        evaluator=config.evaluator,
        suite=config.suite,
        adapter_capabilities=isolated_capabilities,
        trial_count=config.trial_count,
        resource_budgets=config.resource_budgets,
    )
    first = _failed_source(case, execution, config).trial_score
    second = _failed_source(case, execution, isolated_config).trial_score

    assert first.compatibility.isolation_profile != second.compatibility.isolation_profile
    assert (
        first.compatibility.adapter_capabilities_digest
        != second.compatibility.adapter_capabilities_digest
    )
    with pytest.raises(ValueError, match="incompatible"):
        MetricsAggregator().aggregate_case((first, second))


def test_real_repository_and_frozen_scores_bind_target_and_wire_and_cannot_roll_up() -> None:
    repository_case, _replay, execution, repository_config = _score_sources(
        review_findings=(_truth("wire", SWE_AUTHORITY),)
    )
    repository_score = _failed_source(
        repository_case, execution, repository_config
    ).trial_score
    frozen_target = FrozenContextReviewTarget(
        kind=ReviewTargetKind.FROZEN_CONTEXT,
        bundle_id="compatibility-bundle-v1",
        record_id="compatibility-record-v1",
        context_format="rendered_text",
        rendered_sha256="a" * 64,
        rendered_utf8_bytes=12,
        source_binding_digest="b" * 64,
    )
    frozen_case = replace(
        repository_case,
        input=EvalCaseInput(review_target=frozen_target),
        source=replace(
            repository_case.source,
            suite="frozen-compatibility-suite",
            source_id="frozen-compatibility-source",
            content_hash="5" * 64,
        ),
    )
    frozen_snapshot = _snapshot_for_cases((frozen_case,))
    frozen_capabilities = replace(
        repository_config.adapter_capabilities,
        adapter_id="frozen-context-test-adapter-v2",
        target_kinds=(ReviewTargetKind.FROZEN_CONTEXT,),
        isolation_profile="frozen-context-isolation-v2",
    )
    frozen_config = EvalRunConfig.create(
        run_instance_key="frozen-compatibility-instance-v1",
        agent=repository_config.agent,
        clarification_matcher=repository_config.clarification_matcher,
        evaluator=repository_config.evaluator,
        suite=SuiteRunConfig.from_case_snapshot(frozen_snapshot),
        adapter_capabilities=frozen_capabilities,
        trial_count=1,
        resource_budgets=repository_config.resource_budgets,
    )
    frozen_score = _failed_source(frozen_case, execution, frozen_config).trial_score

    repository_payload = repository_score.to_dict()["compatibility"]
    frozen_payload = frozen_score.to_dict()["compatibility"]
    assert repository_payload["target_kind"] == "repository"
    assert repository_payload["wire_contract"]["review_target_kind"] == "repository"
    assert frozen_payload["target_kind"] == "frozen_context"
    assert frozen_payload["wire_contract"]["review_target_kind"] == "frozen_context"
    assert repository_payload["wire_contract_digest"] != frozen_payload["wire_contract_digest"]
    with pytest.raises(ValueError, match="incompatible"):
        MetricsAggregator().aggregate_case((repository_score, frozen_score))


def test_swe_report_exposes_authority_compatibility_and_not_scorable_coverage() -> None:
    case, _snapshot, _replay = _case_and_snapshot(
        review_findings=(_truth("swe-report", SWE_AUTHORITY),)
    )
    snapshot = _snapshot_for_cases((case,))
    execution = _execution()
    config = run_config(snapshot, evaluator=execution.evaluator, trial_count=1)
    source = _failed_source(case, execution, config)
    summary = ReportBuilder().build_summary(
        config,
        execution,
        "metrics-eval-v1",
        eval_cases=(case,),
        trial_sources=(source,),
    )
    partition = summary.to_dict()["partitions"][0]
    compatibility = partition["compatibility"]
    metrics = {
        item["metric"]: item for item in partition["aggregate_score"]["metrics"]
    }

    assert partition["authority_coverage"] == {
        "expected_truth_count": 1,
        "required_expected_truth_count": 1,
        "severity_eligible_required_truth_count": 0,
        "severity_excluded_required_truth_count": 1,
        "location_precision_eligible_truth_count": 0,
        "location_precision_excluded_truth_count": 1,
        "location_recall_eligible_required_truth_count": 0,
        "location_recall_excluded_required_truth_count": 1,
    }
    for metric in AUTHORITY_METRICS:
        payload = metrics[metric.value]
        assert payload["null_reason"] == "not_scorable"
        assert payload["coverage"]["not_scorable_count"] == 1
    for field in (
        "target_kind",
        "wire_contract",
        "wire_contract_digest",
        "adapter_capabilities_digest",
        "isolation_profile",
        "metric_authority_profile",
        "metric_authority_profile_digest",
        "metric_authority_policy_version",
        "metric_authority_policy_digest",
    ):
        assert field in compatibility

    markdown = render_run_markdown(summary)
    for text in (
        "Target kind",
        "Wire contract digest",
        "Adapter capabilities digest",
        "Isolation profile",
        "Metric authority profile",
        "Metric authority policy digest",
        "Authority coverage",
        "not_scorable_count",
        *tuple(metric.value for metric in AUTHORITY_METRICS),
    ):
        assert text in markdown

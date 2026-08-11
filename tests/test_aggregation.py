from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from review_agent.aggregation import (
    AGGREGATION_RECORD_SCHEMA,
    DeterministicReviewAggregator,
    ReviewAggregationInput,
    finding_fingerprint,
)
from review_agent.pr_workspace import PRMetadata, PRWorkspaceStore
from review_agent.review_planning import compile_review_plan
from review_agent.review_protocol import (
    FindingSeverity,
    ReviewerFinding,
    ReviewerOutput,
    ReviewResultStatus,
    RiskLevel,
)
from review_agent.reviewer_executor import ReviewerExecutionResultV2
from review_agent.reviewer_output import RejectedReviewerFinding
from review_agent.revision import RepositoryIdentity


SNAPSHOT_ID = "S-" + "a" * 64
PR_ID = "PR-" + "b" * 64


def _plan(level: RiskLevel = RiskLevel.CRITICAL):
    return compile_review_plan(
        snapshot_id=SNAPSHOT_ID,
        risk_level=level,
        allowed_files=("src/cache.py",),
        allowed_symbols=(),
        allowed_hunks=("src/cache.py#hunk-0",),
    )


def _finding(
    *,
    claim: str = "When Foo is absent, dereferencing it raises and returns 500.",
    severity: FindingSeverity = FindingSeverity.HIGH,
    suggestion: str = "Guard the absent value and add a first-request test.",
    path: str = "src/cache.py",
    line: int = 10,
) -> ReviewerFinding:
    return ReviewerFinding(
        claim=claim,
        severity=severity,
        path=path,
        line=line,
        suggestion=suggestion,
    )


def _execution(
    assignment_id: str,
    *,
    status: str = "completed",
    findings: tuple[ReviewerFinding, ...] = (),
    uncertainties: tuple[str, ...] = (),
    error_code: str | None = None,
    rejected: tuple[RejectedReviewerFinding, ...] = (),
) -> ReviewerExecutionResultV2:
    output = (
        ReviewerOutput(findings=findings, uncertainties=uncertainties)
        if status == "completed"
        else None
    )
    return ReviewerExecutionResultV2(
        assignment_id=assignment_id,
        status=status,
        output=output.to_json() if output is not None else None,
        reviewer_output=output,
        rejected_findings=rejected,
        error_code=error_code,
        active_elapsed_seconds=123.0,
    )


def _inputs(plan, executions) -> tuple[ReviewAggregationInput, ...]:
    return tuple(
        ReviewAggregationInput(
            reviewer_id=f"reviewer-{index}",
            execution=execution,
        )
        for index, execution in enumerate(executions)
    )


def test_fingerprint_uses_only_snapshot_path_line_and_normalized_claim() -> None:
    first = _finding(claim="  When Foo is absent,\n it fails.  ")
    same_issue = _finding(
        claim="When Foo is absent, it fails.",
        severity=FindingSeverity.LOW,
        suggestion="Use a different safe correction.",
    )
    case_changed = _finding(claim="When foo is absent, it fails.")

    assert finding_fingerprint(SNAPSHOT_ID, first) == finding_fingerprint(
        SNAPSHOT_ID, same_issue
    )
    assert finding_fingerprint(SNAPSHOT_ID, first) != finding_fingerprint(
        SNAPSHOT_ID, case_changed
    )
    assert finding_fingerprint("S-" + "c" * 64, first) != finding_fingerprint(
        SNAPSHOT_ID, first
    )


def test_exact_identity_merges_with_highest_severity_then_role_order() -> None:
    plan = _plan()
    shared = _finding(severity=FindingSeverity.HIGH, suggestion="Core correction.")
    executions = (
        _execution(plan.assignments[0].assignment_id, findings=(shared,)),
        _execution(
            plan.assignments[1].assignment_id,
            findings=(
                replace(
                    shared,
                    severity=FindingSeverity.BLOCKER,
                    suggestion="Adversarial blocker correction.",
                ),
            ),
        ),
        _execution(
            plan.assignments[2].assignment_id,
            findings=(
                replace(
                    shared,
                    severity=FindingSeverity.BLOCKER,
                    suggestion="Dynamic blocker correction.",
                ),
            ),
        ),
        _execution(plan.assignments[3].assignment_id),
    )

    bundle = DeterministicReviewAggregator().aggregate(
        pr_id=PR_ID,
        plan=plan,
        reviewer_inputs=_inputs(plan, executions),
    )

    assert len(bundle.review_result.findings) == 1
    merged = bundle.review_result.findings[0]
    assert merged.severity is FindingSeverity.BLOCKER
    assert merged.suggestion == "Adversarial blocker correction."
    group = bundle.aggregation_record["merge_groups"][0]
    assert group["selected_reviewer_id"] == "reviewer-1"
    assert len(group["sources"]) == 3


def test_same_location_different_claim_and_case_remain_separate() -> None:
    plan = _plan(RiskLevel.LOW)
    findings = (
        _finding(claim="Foo can be null and crashes."),
        _finding(claim="foo can be null and crashes."),
        _finding(claim="The cache value can be stale."),
    )
    execution = _execution(plan.assignments[0].assignment_id, findings=findings)

    result = DeterministicReviewAggregator().aggregate(
        pr_id=PR_ID,
        plan=plan,
        reviewer_inputs=_inputs(plan, (execution,)),
    ).review_result

    assert len(result.findings) == 3
    assert len({finding.finding_id for finding in result.findings}) == 3


def test_equal_severity_prefers_core_over_adversarial() -> None:
    plan = _plan(RiskLevel.MEDIUM)
    shared = _finding(severity=FindingSeverity.HIGH)
    core = _execution(
        plan.assignments[0].assignment_id,
        findings=(replace(shared, suggestion="Use the Core correction."),),
    )
    adversarial = _execution(
        plan.assignments[1].assignment_id,
        findings=(replace(shared, suggestion="Use the adversarial correction."),),
    )

    result = DeterministicReviewAggregator().aggregate(
        PR_ID,
        plan,
        _inputs(plan, (core, adversarial)),
    ).review_result

    assert result.findings[0].suggestion == "Use the Core correction."


def test_input_order_and_runtime_time_do_not_change_canonical_outputs() -> None:
    plan = _plan(RiskLevel.MEDIUM)
    core = _execution(plan.assignments[0].assignment_id, findings=(_finding(),))
    adversarial = _execution(
        plan.assignments[1].assignment_id,
        findings=(_finding(severity=FindingSeverity.LOW),),
    )
    inputs = _inputs(plan, (core, adversarial))
    changed_time = tuple(
        replace(item, execution=replace(item.execution, active_elapsed_seconds=999.0))
        for item in inputs
    )
    aggregator = DeterministicReviewAggregator()

    first = aggregator.aggregate(PR_ID, plan, inputs)
    second = aggregator.aggregate(PR_ID, plan, tuple(reversed(changed_time)))

    assert first.review_result_bytes == second.review_result_bytes
    assert first.aggregation_bytes == second.aggregation_bytes


def test_partial_and_failed_status_preserve_valid_findings_and_safe_coverage() -> None:
    plan = _plan(RiskLevel.MEDIUM)
    core_failed = _execution(
        plan.assignments[0].assignment_id,
        status="failed",
        error_code="private/provider/path/SHOULD_NOT_LEAK",
    )
    adversarial = _execution(
        plan.assignments[1].assignment_id,
        findings=(_finding(),),
        uncertainties=("  Could not   exercise the optional path. ",),
        rejected=(RejectedReviewerFinding(1, "line_not_in_diff"),),
    )
    aggregator = DeterministicReviewAggregator()

    partial = aggregator.aggregate(
        PR_ID,
        plan,
        _inputs(plan, (core_failed, adversarial)),
    ).review_result
    failed = aggregator.aggregate(
        PR_ID,
        plan,
        _inputs(
            plan,
            (
                core_failed,
                _execution(
                    plan.assignments[1].assignment_id,
                    status="timeout",
                    error_code="active_time_exhausted",
                ),
            ),
        ),
    ).review_result

    assert partial.status is ReviewResultStatus.PARTIAL
    assert len(partial.findings) == 1
    joined = "\n".join(partial.uncertainties)
    assert "Core Reviewer coverage is incomplete" in joined
    assert "runtime_failure" in joined
    assert "SHOULD_NOT_LEAK" not in joined
    assert "Could not exercise the optional path." in partial.uncertainties
    assert "candidate 2" in joined
    assert failed.status is ReviewResultStatus.FAILED
    assert failed.findings == ()


def test_all_valid_zero_finding_outputs_are_completed() -> None:
    plan = _plan(RiskLevel.LOW)
    execution = _execution(plan.assignments[0].assignment_id)

    result = DeterministicReviewAggregator().aggregate(
        PR_ID,
        plan,
        _inputs(plan, (execution,)),
    ).review_result

    assert result.status is ReviewResultStatus.COMPLETED
    assert result.findings == ()


def test_uncertainties_are_nfkc_whitespace_normalized_ordered_and_deduplicated() -> None:
    plan = _plan(RiskLevel.MEDIUM)
    first = _execution(
        plan.assignments[0].assignment_id,
        uncertainties=("Ｂeta  gap", "Alpha\n gap"),
    )
    second = _execution(
        plan.assignments[1].assignment_id,
        uncertainties=("Beta gap", "Alpha gap"),
    )

    result = DeterministicReviewAggregator().aggregate(
        PR_ID,
        plan,
        _inputs(plan, (first, second)),
    ).review_result

    assert result.uncertainties == ("Alpha gap", "Beta gap")


def test_publish_is_create_only_and_resume_reuses_verified_bytes(tmp_path: Path) -> None:
    store, snapshot, pr_id = _workspace(tmp_path)
    plan = compile_review_plan(
        snapshot_id=snapshot.snapshot_id,
        risk_level=RiskLevel.LOW,
        allowed_files=("src/cache.py",),
        allowed_symbols=(),
        allowed_hunks=(),
    )
    execution = _execution(
        plan.assignments[0].assignment_id,
        findings=(_finding(),),
    )
    aggregator = DeterministicReviewAggregator()
    first = aggregator.publish_or_reuse(
        workspace_store=store,
        snapshot=snapshot,
        pr_id=pr_id,
        plan=plan,
        reviewer_inputs=_inputs(plan, (execution,)),
    )

    class ExplodingInputs:
        def __iter__(self):
            raise AssertionError("Resume must not aggregate again")

    resumed = aggregator.publish_or_reuse(
        workspace_store=store,
        snapshot=snapshot,
        pr_id=pr_id,
        plan=plan,
        reviewer_inputs=ExplodingInputs(),
    )

    assert resumed.reused is True
    assert resumed.review_result_bytes == first.review_result_bytes
    assert (snapshot.path / "Results" / "aggregation.json").read_bytes() == (
        first.aggregation_bytes
    )
    assert (snapshot.path / "Results" / "review-result.json").read_bytes() == (
        first.review_result_bytes
    )
    assert first.aggregation_record["schema_version"] == AGGREGATION_RECORD_SCHEMA


def test_resume_recovers_aggregation_only_crash_window(tmp_path: Path) -> None:
    store, snapshot, pr_id = _workspace(tmp_path)
    plan = compile_review_plan(
        snapshot_id=snapshot.snapshot_id,
        risk_level=RiskLevel.LOW,
        allowed_files=("src/cache.py",),
        allowed_symbols=(),
        allowed_hunks=(),
    )
    execution = _execution(plan.assignments[0].assignment_id, findings=(_finding(),))
    inputs = _inputs(plan, (execution,))
    aggregator = DeterministicReviewAggregator()
    expected = aggregator.aggregate(pr_id, plan, inputs)
    store.publish_create_only(
        snapshot,
        "Results/aggregation.json",
        expected.aggregation_bytes,
    )

    recovered = aggregator.publish_or_reuse(
        store,
        snapshot,
        pr_id,
        plan,
        inputs,
    )

    assert recovered.review_result_bytes == expected.review_result_bytes
    assert store.review_result_bundle_state(snapshot) == "complete"


def test_resume_rejects_a_different_review_plan(tmp_path: Path) -> None:
    store, snapshot, pr_id = _workspace(tmp_path)
    low = compile_review_plan(
        snapshot_id=snapshot.snapshot_id,
        risk_level=RiskLevel.LOW,
        allowed_files=("src/cache.py",),
        allowed_symbols=(),
        allowed_hunks=(),
    )
    aggregator = DeterministicReviewAggregator()
    aggregator.publish_or_reuse(
        store,
        snapshot,
        pr_id,
        low,
        _inputs(low, (_execution(low.assignments[0].assignment_id),)),
    )
    medium = compile_review_plan(
        snapshot_id=snapshot.snapshot_id,
        risk_level=RiskLevel.MEDIUM,
        allowed_files=("src/cache.py",),
        allowed_symbols=(),
        allowed_hunks=(),
    )

    with pytest.raises(ValueError, match="immutable ReviewPlan"):
        aggregator.publish_or_reuse(
            store,
            snapshot,
            pr_id,
            medium,
            (),
        )


def test_published_result_tampering_fails_closed(tmp_path: Path) -> None:
    store, snapshot, pr_id = _workspace(tmp_path)
    plan = compile_review_plan(
        snapshot_id=snapshot.snapshot_id,
        risk_level=RiskLevel.LOW,
        allowed_files=("src/cache.py",),
        allowed_symbols=(),
        allowed_hunks=(),
    )
    execution = _execution(plan.assignments[0].assignment_id)
    aggregator = DeterministicReviewAggregator()
    aggregator.publish_or_reuse(
        store,
        snapshot,
        pr_id,
        plan,
        _inputs(plan, (execution,)),
    )
    (snapshot.path / "Results" / "review-result.json").write_bytes(b"{}")

    with pytest.raises(ValueError):
        aggregator.load_published(store, snapshot)


def _workspace(tmp_path: Path):
    repository = tmp_path / "repo"
    git_common = repository / ".git"
    git_common.mkdir(parents=True)
    identity = RepositoryIdentity(
        canonical_path=str(repository.resolve()),
        git_common_dir=str(git_common.resolve()),
        origin_url=None,
    )
    store = PRWorkspaceStore(tmp_path / "ra")
    resolved = store.resolve_pr(identity, "local", "aggregation-task")
    workspace = store.create_or_load_workspace(
        resolved,
        PRMetadata(title="Aggregation task"),
    )
    snapshot = store.create_or_load_snapshot(workspace, "1" * 40, "2" * 40)
    return store, snapshot, resolved.pr_id

"""Small, in-memory truth-completeness smoke tests for Review Task 10.

These tests intentionally avoid repository preparation and model/network I/O.
They exercise the staged evaluator with a tiny verified replay and typed
``JudgeExecutionResult`` values.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

import review_agent_eval.review_evaluator as review_evaluator_module
import review_agent_eval.repository as repository_module
from review_agent.model_adapter import ModelAdapterCapabilities
from review_agent_eval.config import (
    EvaluatorExecutionConfig,
    EvaluatorRunConfig,
    JudgeKind,
    JudgeProfileSnapshot,
)
from review_agent_eval.judge import (
    DEFAULT_JUDGE_RUBRICS,
    GLOBAL_JUDGE_SYSTEM_PROMPT,
    JUDGE_CONTEXT_BUILDER_VERSION,
    JUDGE_PARSER_VERSION,
    JUDGE_SYSTEM_PROMPT_VERSION,
    JudgeExecutionResult,
    JudgeRubric,
    JudgeRubricCatalog,
    JudgeRunStatus,
    JudgeTask,
    SemanticJudge,
)
from review_agent_eval.models import (
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    DiffSide,
    EvalInput,
    EvalSubmission,
    ExpectedFinding,
    FindingSeverity,
    IntentResult,
    KnownInvalidFinding,
    NovelFindingPolicy,
    MetricAuthority,
    MetricAuthoritySource,
    Repository,
    RepositoryReviewTarget,
    RepositorySource,
    RequiredContextLevel,
    ReviewRequest,
    ReviewTargetKind,
    ReviewTruth,
    SubmissionFinding,
    SubmissionIntent,
    SubmissionReview,
    SubmissionStatus,
    SubmissionUsage,
    TruthCompleteness,
    TruthLocation,
    canonical_sha256,
)
from review_agent_eval.repository import PreparedRepositoryReplay
from review_agent_eval.review_evaluator import (
    FindingDisposition,
    FindingResolution,
    ReviewEvaluationStatus,
    ReviewEvaluator,
    ReviewEvaluationPhase,
)


BASE_REVISION = "a" * 40
HEAD_REVISION = "b" * 40
TARGET_MATERIALIZATION_ID = "materialization-" + ("2" * 64)


def _execution(
    rubrics: JudgeRubricCatalog = DEFAULT_JUDGE_RUBRICS,
) -> EvaluatorExecutionConfig:
    profiles = []
    for task in JudgeTask:
        rubric = rubrics.for_task(task)
        system_prompt = GLOBAL_JUDGE_SYSTEM_PROMPT + "\nTask rubric:\n" + rubric.instruction
        profiles.append(
            JudgeProfileSnapshot(
                schema_version="eval_judge_profile_v1",
                kind=JudgeKind(task.value),
                judge_id=f"{task.value}-judge",
                judge_version="judge-v1",
                adapter_id="scripted-adapter",
                adapter_version="adapter-v1",
                adapter_config_digest="0" * 64,
                provider="scripted-provider",
                model="scripted-model",
                model_artifact_digest=None,
                parameters={"temperature": 0},
                system_prompt_version=JUDGE_SYSTEM_PROMPT_VERSION,
                system_prompt_digest=canonical_sha256(system_prompt),
                rubric_id=rubric.rubric_id,
                rubric_version=rubric.rubric_version,
                rubric_digest=rubric.rubric_digest,
                response_schema_version=rubric.response_schema,
                response_schema_digest=canonical_sha256(rubric.response_schema),
                context_builder_version=JUDGE_CONTEXT_BUILDER_VERSION,
                parser_version=JUDGE_PARSER_VERSION,
            )
        )
    evaluator = EvaluatorRunConfig(
        evaluator_id="review-evaluator",
        evaluator_version="review-evaluator-v1",
        grader_version="review-grader-v1",
        judge_profiles=tuple(profiles),
    )
    return EvaluatorExecutionConfig.create(
        evaluator=evaluator,
        evaluator_timeout_seconds=60,
        max_execution_artifact_file_bytes=2 * 1024 * 1024,
        max_execution_artifact_total_bytes=8 * 1024 * 1024,
    )


def _fixture() -> tuple[EvalInput, PreparedRepositoryReplay]:
    repository = Repository(
        source=RepositorySource.FIXTURE,
        path="minimal-review-fixture",
        url=None,
        base_revision=BASE_REVISION,
        head_revision=HEAD_REVISION,
    )
    raw = b"return value\n"
    object_id = hashlib.sha1(raw).hexdigest()
    objects = MappingProxyType(
        {object_id: repository_module._GitObject(object_id, "blob", raw)}
    )
    files = MappingProxyType({"src/app.py": object_id})
    replay = PreparedRepositoryReplay(
        prepared_repository_id="minimal-prepared-replay",
        repository_descriptor_digest=repository.digest(),
        base_revision=BASE_REVISION,
        head_revision=HEAD_REVISION,
        _git_dir=Path("unused-replay.git"),
        _runner=None,
        _open_check=lambda: None,
        _verify_cache=lambda: None,
        _objects=objects,
        _files_by_revision=MappingProxyType(
            {BASE_REVISION: files, HEAD_REVISION: files}
        ),
    )
    eval_input = EvalInput(
        schema_version=EVAL_INPUT_SCHEMA_VERSION,
        task_id="task-review-truth",
        review_target=RepositoryReviewTarget(
            kind=ReviewTargetKind.REPOSITORY,
            repository=repository,
            review_request=ReviewRequest(
                title="minimal review",
                description=None,
                user_intent=None,
                review_focus=None,
                linked_requirements=(),
                project_rules=(),
                existing_ci_evidence=(),
            ),
        ),
    )
    return eval_input, replay


def _finding(claim: str, finding_id: str = "finding-1") -> SubmissionFinding:
    return SubmissionFinding(
        finding_id=finding_id,
        claim=claim,
        severity=FindingSeverity.HIGH,
        path="src/app.py",
        side=DiffSide.RIGHT,
        from_line=1,
        to_line=1,
        evidence_refs=(),
        suggested_action="fix it",
    )


def _expected(claim: str, truth_id: str = "truth-1") -> ExpectedFinding:
    return ExpectedFinding(
        truth_id=truth_id,
        claim=claim,
        severity=FindingSeverity.HIGH,
        category="correctness",
        required=True,
        metric_authority=MetricAuthority(
            severity_scorable=True,
            severity_authority=MetricAuthoritySource.EXPERT_ANNOTATION,
            location_scorable=True,
            location_authority=MetricAuthoritySource.EXPERT_ANNOTATION,
        ),
        locations=(
            TruthLocation(
                path="src/app.py",
                side=DiffSide.RIGHT,
                from_line=1,
                to_line=1,
            ),
        ),
        evidence_anchors=(),
        required_context_level=RequiredContextLevel.DIFF,
        rationale="minimal test truth",
    )


def _submission(finding: SubmissionFinding) -> EvalSubmission:
    return EvalSubmission(
        schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
        task_id="task-review-truth",
        agent_id="agent-under-test",
        trial_id="trial-review-truth",
        eval_input_digest=_fixture()[0].digest(),
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        status=SubmissionStatus.COMPLETED,
        intent=SubmissionIntent(
            status=IntentResult.SUFFICIENT,
            goal=None,
            acceptance_criteria=(),
            scope=(),
            constraints=(),
            claims=(),
            clarification_questions=(),
            uncertainties=(),
        ),
        review=SubmissionReview(findings=(finding,), uncertainties=()),
        evidence=(),
        usage=SubmissionUsage(
            elapsed_seconds=0,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            tool_calls=None,
            cost_amount=None,
            cost_currency=None,
        ),
        trace_ref=None,
        failure=None,
    )


def _evaluator() -> ReviewEvaluator:
    eval_input, replay = _fixture()
    return ReviewEvaluator(
        eval_input=eval_input,
        replay=replay,
        trial_id="trial-review-truth",
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=_execution(),
    )


class _CapabilityFailingFactory:
    def create(self):
        factory = self

        class Adapter:
            provider_name = "scripted-provider"
            capabilities = ModelAdapterCapabilities(
                supports_tool_choice_none=False,
                enforces_request_timeout=True,
                max_response_bytes=1024 * 1024,
            )

            def complete_turn(self, request):  # pragma: no cover - preflight stops first
                factory.called = True
                raise AssertionError("preflight should prevent model execution")

        return Adapter()

    called = False


def _typed_failure(request, execution) -> JudgeExecutionResult:
    return SemanticJudge(
        adapter_factory=_CapabilityFailingFactory(),
        evaluator_execution=execution,
    ).execute(request)


def test_closed_world_forbid_marks_unmatched_unknown_without_factuality_request() -> None:
    evaluator = _evaluator()
    truth = ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.FORBID,
        expected_findings=(),
        known_invalid_findings=(),
    )

    result = evaluator.evaluate(_submission(_finding("a novel defect")), truth)
    outcome = result.finding_outcomes[0]

    assert result.status is ReviewEvaluationStatus.UNGRADED
    assert result.phase is ReviewEvaluationPhase.COMPLETE
    assert outcome.issue_judgement.value == "unknown"
    assert outcome.issue_resolution is FindingResolution.UNGRADED
    assert outcome.disposition is FindingDisposition.NOVEL_DISALLOWED
    assert outcome.novel_request_id is None
    assert result.judge_requests == ()
    assert result.judge_ungraded == ()


def test_verify_unmatched_emits_one_pending_factuality_request() -> None:
    evaluator = _evaluator()
    truth = ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.VERIFY,
        expected_findings=(),
        known_invalid_findings=(),
    )

    result = evaluator.evaluate(_submission(_finding("a novel defect")), truth)
    outcome = result.finding_outcomes[0]

    assert result.status is ReviewEvaluationStatus.PENDING_JUDGE
    assert len(result.judge_requests) == 1
    request = result.judge_requests[0]
    assert request.phase is ReviewEvaluationPhase.NOVEL_FACTUALITY
    assert request.task is JudgeTask.NOVEL_FACTUALITY
    assert outcome.issue_judgement.value == "unknown"
    assert outcome.issue_resolution is FindingResolution.PENDING_JUDGE
    assert outcome.disposition is FindingDisposition.UNGRADED
    assert outcome.novel_request_id == request.request_id


def test_typed_judge_failure_cannot_become_different_or_confirmed() -> None:
    evaluator = _evaluator()
    truth = ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.VERIFY,
        expected_findings=(_expected("the expected defect"),),
        known_invalid_findings=(),
    )
    submission = _submission(_finding("a semantically unresolved defect"))
    pending = evaluator.evaluate(submission, truth)
    assert len(pending.judge_requests) == 1

    execution = evaluator.evaluator_execution
    typed = _typed_failure(pending.judge_requests[0].request, execution)
    assert type(typed) is JudgeExecutionResult
    assert typed.status is JudgeRunStatus.JUDGE_FAILED

    result = evaluator.evaluate(submission, truth, judge_results=(typed,))
    outcome = result.finding_outcomes[0]

    assert result.status is ReviewEvaluationStatus.UNGRADED
    assert result.phase is ReviewEvaluationPhase.EXPECTED_ASSIGNMENT
    assert outcome.issue_judgement.value == "unknown"
    assert outcome.issue_resolution in {
        FindingResolution.JUDGE_FAILED,
        FindingResolution.UNGRADED,
    }
    assert outcome.disposition is FindingDisposition.UNGRADED
    assert outcome.matched_expected_truth_id is None
    assert outcome.matched_known_invalid_truth_id is None
    assert result.assignments == ()
    assert len(result.judge_failures) == 1


@pytest.mark.parametrize(
    "completeness",
    [TruthCompleteness.EXPERT_AUGMENTED, TruthCompleteness.HUMAN_OBSERVED],
)
def test_incomplete_truth_cannot_use_forbid(completeness: TruthCompleteness) -> None:
    with pytest.raises(ValueError, match="forbid"):
        ReviewTruth(
            completeness=completeness,
            novel_finding_policy=NovelFindingPolicy.FORBID,
            expected_findings=(),
            known_invalid_findings=(),
        )


def test_known_invalid_failure_blocks_expected_assignment() -> None:
    evaluator = _evaluator()
    truth = ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.VERIFY,
        expected_findings=(_expected("the expected defect", "truth-expected"),),
        known_invalid_findings=(
            KnownInvalidFinding(
                truth_id="truth-invalid",
                claim="a different known invalid defect",
                category="correctness",
                locations=(),
                rationale="known invalid trap",
            ),
        ),
    )
    submission = _submission(_finding("a semantically unresolved defect"))
    pending = evaluator.evaluate(submission, truth)
    assert pending.phase is ReviewEvaluationPhase.KNOWN_INVALID
    assert len(pending.judge_requests) == 1

    typed = _typed_failure(pending.judge_requests[0].request, evaluator.evaluator_execution)
    result = evaluator.evaluate(submission, truth, judge_results=(typed,))

    assert result.phase is ReviewEvaluationPhase.KNOWN_INVALID
    assert result.assignments == ()
    assert result.finding_outcomes[0].disposition is FindingDisposition.UNGRADED


def test_candidate_limit_returns_harness_owned_ungraded(monkeypatch) -> None:
    evaluator = _evaluator()
    monkeypatch.setattr(review_evaluator_module, "MAX_REVIEW_CANDIDATES", 1)
    truth = ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.FORBID,
        expected_findings=(
            _expected("first expected", "truth-a"),
            _expected("second expected", "truth-b"),
        ),
        known_invalid_findings=(),
    )

    result = evaluator.evaluate(_submission(_finding("first expected", "finding-limit")), truth)

    assert result.status is ReviewEvaluationStatus.UNGRADED
    assert result.limit_failure is not None
    assert result.limit_failure.reason_code.value == "candidate_limit_exceeded"
    assert result.assignments == ()


def test_expected_and_known_invalid_canonical_claim_conflict_fails_closed() -> None:
    evaluator = _evaluator()
    truth = ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.FORBID,
        expected_findings=(_expected("same claim", "truth-expected"),),
        known_invalid_findings=(
            KnownInvalidFinding(
                truth_id="truth-invalid",
                claim="same claim",
                category="correctness",
                locations=(),
                rationale="conflicting annotation",
            ),
        ),
    )

    with pytest.raises(ValueError, match="same canonical claim"):
        evaluator.evaluate(_submission(_finding("same claim", "finding-conflict")), truth)


def _custom_rubric_catalog() -> JudgeRubricCatalog:
    original = DEFAULT_JUDGE_RUBRICS.for_task(JudgeTask.FINDING_EQUIVALENCE)
    custom = JudgeRubric.create(
        task=JudgeTask.FINDING_EQUIVALENCE,
        rubric_id="finding-equivalence-custom",
        rubric_version="finding-equivalence-custom-v1",
        response_schema=original.response_schema,
        instruction=original.instruction + " Custom root-cause calibration rule.",
    )
    return JudgeRubricCatalog.create(
        "custom-review-rubrics-v1",
        tuple(
            custom if item.task is JudgeTask.FINDING_EQUIVALENCE else item
            for item in DEFAULT_JUDGE_RUBRICS.rubrics
        ),
    )


def test_custom_review_rubric_catalog_is_bound_to_execution_profile() -> None:
    rubrics = _custom_rubric_catalog()
    eval_input, replay = _fixture()
    evaluator = ReviewEvaluator(
        eval_input=eval_input,
        replay=replay,
        trial_id="trial-review-truth",
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=_execution(rubrics),
        rubrics=rubrics,
    )
    truth = ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.VERIFY,
        expected_findings=(_expected("the expected defect"),),
        known_invalid_findings=(),
    )

    result = evaluator.evaluate(
        _submission(_finding("a semantic description")),
        truth,
    )

    request = result.judge_requests[0].request
    assert request.rubric == rubrics.for_task(JudgeTask.FINDING_EQUIVALENCE)
    assert "Custom root-cause calibration rule." in request.system_prompt

    with pytest.raises(ValueError, match="profile differs"):
        ReviewEvaluator(
            eval_input=eval_input,
            replay=replay,
            trial_id="trial-review-truth",
            target_materialization_id=TARGET_MATERIALIZATION_ID,
            evaluator_execution=_execution(),
            rubrics=rubrics,
        )


def test_custom_review_rubric_rejects_wrong_system_prompt_digest() -> None:
    rubrics = _custom_rubric_catalog()
    execution = _execution(rubrics)
    profiles = tuple(
        replace(profile, system_prompt_digest="0" * 64)
        if profile.kind is JudgeKind.FINDING_EQUIVALENCE
        else profile
        for profile in execution.evaluator.judge_profiles
    )
    evaluator_config = replace(execution.evaluator, judge_profiles=profiles)
    forged_execution = replace(
        execution,
        evaluator=evaluator_config,
        evaluator_config_digest=evaluator_config.digest(),
    )
    eval_input, replay = _fixture()

    with pytest.raises(ValueError, match="profile differs"):
        ReviewEvaluator(
            eval_input=eval_input,
            replay=replay,
            trial_id="trial-review-truth",
            target_materialization_id=TARGET_MATERIALIZATION_ID,
            evaluator_execution=forged_execution,
            rubrics=rubrics,
        )

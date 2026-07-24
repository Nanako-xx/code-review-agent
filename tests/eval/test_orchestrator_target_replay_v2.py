from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from review_agent_eval.artifacts import (
    ArtifactStateError,
    ArtifactStore,
    StageName,
)
from review_agent_eval.cases import RunCaseSnapshot, SuiteManifest
from review_agent_eval.config import EvaluatorExecutionConfig
from review_agent_eval.datasets import CaseBank
from review_agent_eval.intent_evaluator import IntentEvaluator
from review_agent_eval.judge import (
    BlindJudgeInput,
    JudgeTask,
    JudgeUngradedReason,
    SemanticJudge,
)
from review_agent_eval.materialization import MaterializationError
from review_agent_eval.models import (
    EvalCase,
    EvalInput,
    EvalSubmission,
    EvidenceIntegrity,
    EvidenceKind,
    ExpectedFinding,
    FailureCode,
    FindingSeverity,
    FrozenContextEvidenceSource,
    IntentTruth,
    MetricAuthority,
    MetricAuthoritySource,
    NovelFindingPolicy,
    RequiredContextLevel,
    ReviewEvaluatorContext,
    ReviewTargetKind,
    ReviewTruth,
    SubmissionEvidence,
    SubmissionFinding,
    SubmissionReview,
    SubmissionStatus,
    TruthCompleteness,
    UnsupportedProtocolVersionError,
    canonical_sha256,
)
import review_agent_eval.orchestrator as orchestrator_module
from review_agent_eval.orchestrator import (
    EvaluationOrchestrationError,
    EvaluationOrchestrator,
)
from review_agent_eval.repository import (
    RepositoryIntegrityError,
    RepositoryMode,
    RepositoryPreparer,
)
from review_agent_eval.runner import EvalRunner
import review_agent_eval.target_replay as target_replay_module
from review_agent_eval.target_replay import FrozenContextReplayResolver

from .test_cli import _prepare_and_run_cli_fixture
from .test_evaluator_context import HUNK_A, _provenance, _review_context, _truth_context
from .test_frozen_context import _eval_input, _preparation_binding, _prepared_bundle
from .test_judge import _Factory, _execution
from .test_target_runner import (
    _FrozenSuccessAdapter,
    _run_config,
    _runner as _frozen_runner,
)


def _expected(truth_id: str, claim: str) -> ExpectedFinding:
    return ExpectedFinding(
        truth_id=truth_id,
        claim=claim,
        severity=FindingSeverity.HIGH,
        category="correctness",
        required=True,
        metric_authority=MetricAuthority(
            severity_scorable=True,
            severity_authority=MetricAuthoritySource.EXPERT_ANNOTATION,
            location_scorable=False,
            location_authority=None,
        ),
        locations=(),
        evidence_anchors=(),
        required_context_level=RequiredContextLevel.DIFF,
        rationale="The frozen context demonstrates the defect.",
    )


def _frozen_snapshot_and_case(
    prepared: Any,
    *,
    intent_truth: IntentTruth | None = None,
    review_truth: ReviewTruth | None = None,
    review_evaluator_context: ReviewEvaluatorContext | None = None,
) -> tuple[RunCaseSnapshot, EvalCase]:
    eval_input = _eval_input(prepared)
    target = eval_input.review_target
    truth = review_truth or ReviewTruth(
        completeness=TruthCompleteness.HUMAN_OBSERVED,
        novel_finding_policy=NovelFindingPolicy.VERIFY,
        expected_findings=(),
        known_invalid_findings=(),
    )
    context = review_evaluator_context or ReviewEvaluatorContext(truth_contexts=())
    intent = intent_truth or IntentTruth.from_dict(
        {
            "scorable": False,
            "authority": None,
            "expected_claims": [],
            "forbidden_claims": [],
            "clarification_policy": None,
        }
    )
    case = EvalCase.from_dict(
        {
            "schema_version": "eval_case_v2",
            "task_id": eval_input.task_id,
            "case_version": 1,
            "source": {
                "suite": "swe-frozen-orchestrator-fixture",
                "origin": "swe_prbench",
                "source_id": eval_input.task_id,
                "source_version": "fixture-v1",
                "source_uri": "https://example.test/swe-prbench",
                "license": "CC-BY-4.0",
                "content_hash": target.source_binding_digest,
            },
            "input": {"review_target": target.to_dict()},
            "clarification_script": {"max_rounds": 1, "answers": []},
            "intent_truth": intent.to_dict(),
            "review_truth": truth.to_dict(),
            "review_evaluator_context": context.to_dict(),
        }
    )
    case_bytes = case.to_json().encode("utf-8")
    manifest = SuiteManifest.from_dict(
        {
            "schema_version": "suite_manifest_v2",
            "suite_id": "swe-frozen-orchestrator-fixture",
            "suite_version": "fixture-v1",
            "wire_contract": {
                "case_schema_version": "eval_case_v2",
                "input_schema_version": "eval_input_v2",
                "submission_schema_version": "eval_submission_v2",
                "review_target_kind": "frozen_context",
                "materializer_protocol": "frozen-context-materializer-v2",
            },
            "source": {
                "kind": "public",
                "source_id": "swe-prbench-fixture",
                "source_version": "fixture-v1",
                "source_uri": "https://example.test/swe-prbench",
                "license": "CC-BY-4.0",
                "content_hash": prepared.manifest.source_manifest_digest,
                "preparation_binding": _preparation_binding(prepared).to_dict(),
            },
            "cases": [
                {
                    "task_id": case.task_id,
                    "case_version": case.case_version,
                    "path": "cases/frozen-runtime.json",
                    "split": "capability",
                    "protocol_id": "official_frozen_context",
                    "dimensions": [],
                    "raw_file_size_bytes": len(case_bytes),
                    "raw_file_sha256": hashlib.sha256(case_bytes).hexdigest(),
                    "canonical_case_digest": case.digest(),
                    "eval_input_digest": eval_input.digest(),
                    "truth_completeness": truth.completeness.value,
                }
            ],
        }
    )
    return RunCaseSnapshot.build(manifest, ((manifest.cases[0], case),)), case


def _case_bank(root: Path, snapshot: RunCaseSnapshot, case: EvalCase) -> CaseBank:
    suite_root = root / "orchestrator-suite"
    case_path = suite_root / "cases" / "frozen-runtime.json"
    case_path.parent.mkdir(parents=True)
    case_path.write_text(case.to_json(), encoding="utf-8")
    (suite_root / "suite_manifest.json").write_text(
        snapshot.manifest.to_json(),
        encoding="utf-8",
    )
    return CaseBank.open(suite_root)


@dataclass
class _FrozenRun:
    prepared: Any
    snapshot: RunCaseSnapshot
    case: EvalCase
    config: Any
    runner: EvalRunner
    trial: Any
    bank: CaseBank

    @property
    def store(self) -> ArtifactStore:
        return self.runner.artifact_store


def _run_frozen(
    tmp_path: Path,
    adapter: Any,
    *,
    instance: str,
    intent_truth: IntentTruth | None = None,
    review_truth: ReviewTruth | None = None,
    review_evaluator_context: ReviewEvaluatorContext | None = None,
    materializer: Any = None,
) -> _FrozenRun:
    prepared = _prepared_bundle(tmp_path)
    snapshot, case = _frozen_snapshot_and_case(
        prepared,
        intent_truth=intent_truth,
        review_truth=review_truth,
        review_evaluator_context=review_evaluator_context,
    )
    config = _run_config(snapshot, current=False, instance=instance)
    runner = _frozen_runner(
        tmp_path,
        prepared,
        adapter,
        materializer=materializer,
    )
    result = runner.run(config, snapshot)
    return _FrozenRun(
        prepared=prepared,
        snapshot=snapshot,
        case=case,
        config=config,
        runner=runner,
        trial=result.trials[0],
        bank=_case_bank(tmp_path, snapshot, case),
    )


def _frozen_orchestrator(
    run: _FrozenRun,
    *,
    judge: Any = None,
    resolver: Any = None,
) -> EvaluationOrchestrator:
    selected = resolver or FrozenContextReplayResolver(
        bundle_root=run.prepared.root
    )
    return EvaluationOrchestrator(
        run.store,
        run.bank,
        target_replay_resolvers={ReviewTargetKind.FROZEN_CONTEXT: selected},
        judge=judge,
    )


def _evaluate(
    run: _FrozenRun,
    orchestrator: EvaluationOrchestrator,
    revision: str,
):
    return orchestrator.evaluate_trial(
        run.config.run_id,
        run.trial.task_id,
        run.trial.trial_id,
        evaluator_execution=_execution(),
        evaluation_revision=revision,
    )


class _CountingJudge:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, _request: BlindJudgeInput):
        self.calls += 1
        raise AssertionError("Judge must not be called")


class _RecordingJudge:
    def __init__(self, execution: EvaluatorExecutionConfig) -> None:
        self.execution = execution
        self.requests: list[BlindJudgeInput] = []

    def execute(self, request: BlindJudgeInput):
        self.requests.append(request)
        return SemanticJudge(
            adapter_factory=_Factory([]),
            evaluator_execution=self.execution,
        ).execute(
            request,
            ungraded_reason=JudgeUngradedReason.POLICY_SKIPPED,
        )


class _FrozenFindingAdapter(_FrozenSuccessAdapter):
    def __init__(self, claim: str, *, with_evidence: bool = False) -> None:
        super().__init__()
        self.claim = claim
        self.with_evidence = with_evidence

    def run(self, *args: Any, **kwargs: Any) -> EvalSubmission:
        submission = super().run(*args, **kwargs)
        eval_input: EvalInput = args[0]
        workspace: Path = args[1]
        target_materialization_id = kwargs["target_materialization_id"]
        evidence: tuple[SubmissionEvidence, ...] = ()
        refs: tuple[str, ...] = ()
        if self.with_evidence:
            data = (workspace / "target" / "context.txt").read_bytes()
            first = data.splitlines(keepends=True)[0].decode("utf-8", "strict")
            item = SubmissionEvidence(
                evidence_id="evidence-frozen-agent",
                source=FrozenContextEvidenceSource(
                    kind=EvidenceKind.FROZEN_CONTEXT,
                    target_materialization_id=target_materialization_id,
                    context_ref=eval_input.review_target.record_id,
                    from_line=1,
                    to_line=1,
                ),
                content_hash=hashlib.sha256(first.encode("utf-8")).hexdigest(),
                excerpt=first,
            )
            evidence = (item,)
            refs = (item.evidence_id,)
        finding = SubmissionFinding(
            finding_id="finding-frozen-agent",
            claim=self.claim,
            severity=FindingSeverity.HIGH,
            path=None,
            side=None,
            from_line=None,
            to_line=None,
            evidence_refs=refs,
            suggested_action="Correct the behavior described by the context.",
        )
        return replace(
            submission,
            review=SubmissionReview(findings=(finding,), uncertainties=()),
            evidence=evidence,
        )


class _FrozenTimeoutAdapter(_FrozenSuccessAdapter):
    def run(self, *args: Any, **kwargs: Any) -> EvalSubmission:
        del args, kwargs
        raise TimeoutError("fixture Agent timeout after PREPARE")


class _CountingFrozenResolver:
    def __init__(self, bundle_root: Path) -> None:
        self.inner = FrozenContextReplayResolver(bundle_root=bundle_root)
        self.calls = 0

    def resolve(self, source: Any):
        self.calls += 1
        return self.inner.resolve(source)


def test_frozen_runner_orchestrator_evaluate_and_hydrate_share_materialization(
    tmp_path: Path,
) -> None:
    run = _run_frozen(
        tmp_path,
        _FrozenSuccessAdapter(),
        instance="frozen-orchestrator-round-trip",
    )
    judge = _CountingJudge()
    orchestrator = _frozen_orchestrator(run, judge=judge)

    evaluated = _evaluate(run, orchestrator, "frozen-round-trip-v1")
    loaded = orchestrator.load_trial_evaluation(
        run.config.run_id,
        run.trial.task_id,
        run.trial.trial_id,
        evaluated.evaluation_id,
    )
    materialization = run.store.load_trial_materialization(
        run.config.run_id,
        run.trial.task_id,
        run.trial.trial_id,
    )

    assert run.trial.submission is not None
    assert evaluated.submission.target_materialization_id == (
        materialization.manifest.materialization_id
    )
    assert loaded.review_result == evaluated.review_result
    assert loaded.trial_score == evaluated.trial_score
    assert judge.calls == 0


def test_builtin_frozen_replay_is_validated_once_at_orchestrator_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run_frozen(
        tmp_path,
        _FrozenSuccessAdapter(),
        instance="frozen-single-replay-validation",
    )
    calls = 0
    original = target_replay_module.validate_target_replay

    def tracked_validate(source: Any, replay: Any) -> None:
        nonlocal calls
        calls += 1
        original(source, replay)

    monkeypatch.setattr(
        target_replay_module,
        "validate_target_replay",
        tracked_validate,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "validate_target_replay",
        tracked_validate,
    )

    _evaluate(
        run,
        _frozen_orchestrator(run, judge=_CountingJudge()),
        "single-replay-validation-v1",
    )

    assert calls == 1


def test_frozen_evidence_and_judge_use_receipt_bound_context_and_materialization(
    tmp_path: Path,
) -> None:
    claim = "The frozen context demonstrates an incorrect review behavior."
    truth = ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.FORBID,
        expected_findings=(_expected("truth-frozen", claim),),
        known_invalid_findings=(),
    )
    run = _run_frozen(
        tmp_path,
        _FrozenFindingAdapter(claim, with_evidence=True),
        instance="frozen-evidence-judge-binding",
        review_truth=truth,
    )
    execution = _execution()
    judge = _RecordingJudge(execution)
    orchestrator = _frozen_orchestrator(run, judge=judge)

    evaluated = orchestrator.evaluate_trial(
        run.config.run_id,
        run.trial.task_id,
        run.trial.trial_id,
        evaluator_execution=execution,
        evaluation_revision="frozen-evidence-v1",
    )
    materialization = run.store.load_trial_materialization(
        run.config.run_id,
        run.trial.task_id,
        run.trial.trial_id,
    )
    evidence = evaluated.submission.evidence[0]
    request = next(
        item for item in judge.requests if item.task is JudgeTask.EVIDENCE_SUPPORT
    )
    evidence_context = next(
        item for item in request.contexts if item.metadata["kind"] == "frozen_context"
    )
    evidence_binding = next(
        item
        for item in request.reference_bindings
        if item.source_id == evidence.evidence_id
    )

    assert evaluated.submission.target_materialization_id == (
        materialization.manifest.materialization_id
    )
    assert evidence.source.target_materialization_id == (
        materialization.manifest.materialization_id
    )
    assert evidence.source.context_ref == materialization.eval_input.review_target.record_id
    assert evidence_context.metadata["source_ref"] == evidence.source.context_ref
    assert evidence_binding.source_digest == canonical_sha256(evidence.to_dict())
    assert evaluated.review_result is not None
    assert evaluated.review_result.evidence_integrity_results[0].integrity is (
        EvidenceIntegrity.VALID
    )
    loaded = orchestrator.load_trial_evaluation(
        run.config.run_id,
        run.trial.task_id,
        run.trial.trial_id,
        evaluated.evaluation_id,
    )
    assert loaded.review_result == evaluated.review_result


def test_wrong_submission_materialization_fails_before_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent_truth = IntentTruth.from_dict(
        {
            "scorable": True,
            "authority": "synthetic",
            "expected_claims": [
                {
                    "truth_id": "intent-semantic",
                    "dimension": "goal",
                    "text": "Support a dry-run mode without changing persisted state.",
                    "required": True,
                }
            ],
            "forbidden_claims": [],
            "clarification_policy": "not_required",
        }
    )
    run = _run_frozen(
        tmp_path,
        _FrozenSuccessAdapter(),
        instance="wrong-submission-materialization",
        intent_truth=intent_truth,
    )
    submission = run.store.load_existing_submission(
        run.config.run_id,
        run.trial.task_id,
        run.trial.trial_id,
    )
    assert submission.intent is not None
    intent_probe = IntentEvaluator().evaluate(
        submission.intent,
        run.case.intent_truth,
        run.case.clarification_script,
        transcript=submission.intent.clarification_questions,
    )
    assert intent_probe.judge_requests
    wrong = replace(
        submission,
        target_materialization_id="materialization-" + "0" * 64,
    )
    monkeypatch.setattr(run.store, "load_existing_submission", lambda *_args: wrong)
    judge = _CountingJudge()

    with pytest.raises(EvaluationOrchestrationError, match="materialization_id"):
        _evaluate(run, _frozen_orchestrator(run, judge=judge), "wrong-submission-v1")

    assert judge.calls == 0


@pytest.mark.parametrize(
    ("policy_field", "unknown_version"),
    (
        (
            "review_evaluator_context_policy_version",
            "truth-scoped-context-v3",
        ),
        ("metric_authority_policy_version", "metric-authority-v3"),
    ),
)
def test_unknown_runtime_policy_fails_before_judge(
    tmp_path: Path,
    policy_field: str,
    unknown_version: str,
) -> None:
    intent_truth = IntentTruth.from_dict(
        {
            "scorable": True,
            "authority": "synthetic",
            "expected_claims": [
                {
                    "truth_id": "intent-runtime-policy",
                    "dimension": "goal",
                    "text": "Support a dry-run mode without changing persisted state.",
                    "required": True,
                }
            ],
            "forbidden_claims": [],
            "clarification_policy": "not_required",
        }
    )
    run = _run_frozen(
        tmp_path,
        _FrozenSuccessAdapter(),
        instance="unknown-runtime-" + policy_field,
        intent_truth=intent_truth,
    )
    submission = run.store.load_existing_submission(
        run.config.run_id,
        run.trial.task_id,
        run.trial.trial_id,
    )
    assert submission.intent is not None
    assert IntentEvaluator().evaluate(
        submission.intent,
        run.case.intent_truth,
        run.case.clarification_script,
    ).judge_requests

    judge = _CountingJudge()
    execution = replace(
        _execution(),
        **{policy_field: unknown_version},
    )
    with pytest.raises(UnsupportedProtocolVersionError):
        _frozen_orchestrator(run, judge=judge).evaluate_trial(
            run.config.run_id,
            run.trial.task_id,
            run.trial.trial_id,
            evaluator_execution=execution,
            evaluation_revision="unknown-runtime-policy-v1",
        )
    assert judge.calls == 0


@pytest.mark.parametrize(
    ("policy_field", "unknown_version"),
    (
        (
            "review_evaluator_context_policy_version",
            "truth-scoped-context-v3",
        ),
        ("metric_authority_policy_version", "metric-authority-v3"),
    ),
)
def test_load_rejects_unknown_runtime_policy_before_judge_hydration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy_field: str,
    unknown_version: str,
) -> None:
    run = _run_frozen(
        tmp_path,
        _FrozenSuccessAdapter(),
        instance="unknown-load-" + policy_field,
    )
    orchestrator = _frozen_orchestrator(run, judge=_CountingJudge())
    evaluated = _evaluate(run, orchestrator, "supported-runtime-policy-v1")
    assert evaluated.intent_result is not None
    assert evaluated.review_result is not None

    unsupported = replace(
        evaluated.evaluator_execution,
        **{policy_field: unknown_version},
    )
    receipt = run.store.write_evaluation(
        run.config.run_id,
        run.trial.task_id,
        run.trial.trial_id,
        evaluator_execution=unsupported,
        revision="persisted-unknown-" + policy_field,
        intent_matches=evaluated.intent_result.to_dict(),
        review_matches=evaluated.review_result.to_dict(),
        judge_input=evaluated.judge_input.to_dict(),
        judge_output=evaluated.judge_output.to_dict(),
        score=evaluated.trial_score.to_dict(),
        report=evaluated.report,
    )
    assert receipt.evaluation_id is not None

    def forbidden_hydration(*_args: Any, **_kwargs: Any):
        raise AssertionError("unknown runtime policy reached Judge hydration")

    monkeypatch.setattr(
        orchestrator_module.JudgeInputArtifact,
        "from_dict",
        forbidden_hydration,
    )
    with pytest.raises(UnsupportedProtocolVersionError):
        orchestrator.load_trial_evaluation(
            run.config.run_id,
            run.trial.task_id,
            run.trial.trial_id,
            receipt.evaluation_id,
        )


@pytest.mark.parametrize(
    "mode",
    ("wrong_type", "wrong_binding", "wrong_context"),
)
def test_bad_resolver_result_fails_before_judge(
    tmp_path: Path,
    mode: str,
) -> None:
    run = _run_frozen(
        tmp_path,
        _FrozenSuccessAdapter(),
        instance="bad-resolver-" + mode,
    )

    class Resolver:
        def resolve(self, source: Any):
            if mode == "wrong_type":
                return object()
            replay = FrozenContextReplayResolver(
                bundle_root=run.prepared.root
            ).resolve(source)
            if mode == "wrong_binding":
                return replace(replay, replay_binding_digest="0" * 64)
            return replace(replay, context_ref="wrong-context")

    judge = _CountingJudge()
    with pytest.raises(
        target_replay_module.TargetReplayIntegrityError,
        match="replay type|binding",
    ):
        _evaluate(
            run,
            _frozen_orchestrator(run, judge=judge, resolver=Resolver()),
            "bad-resolver-v1",
        )
    assert judge.calls == 0


def test_frozen_bundle_drift_after_evaluate_is_rejected_on_load(
    tmp_path: Path,
) -> None:
    run = _run_frozen(
        tmp_path,
        _FrozenSuccessAdapter(),
        instance="frozen-post-evaluate-drift",
    )
    orchestrator = _frozen_orchestrator(run, judge=_CountingJudge())
    evaluated = _evaluate(run, orchestrator, "frozen-drift-v1")
    binding = run.prepared.manifest.records[0]
    record_path = run.prepared.root / binding.path
    record_path.chmod(0o600)
    record_path.write_bytes(b"drifted after evaluation")

    with pytest.raises(MaterializationError, match="bundle trust"):
        orchestrator.load_trial_evaluation(
            run.config.run_id,
            run.trial.task_id,
            run.trial.trial_id,
            evaluated.evaluation_id,
        )


def test_post_prepare_failure_without_review_still_replays_and_detects_drift(
    tmp_path: Path,
) -> None:
    run = _run_frozen(
        tmp_path,
        _FrozenTimeoutAdapter(),
        instance="post-prepare-no-review",
    )
    assert run.trial.submission is not None
    assert run.trial.submission.status is SubmissionStatus.FAILED
    assert run.trial.submission.review is None
    assert run.trial.submission.failure is not None
    assert run.trial.submission.failure.code is FailureCode.TIMEOUT
    state = run.store.load_trial_state(
        run.config.run_id,
        run.trial.task_id,
        run.trial.trial_id,
    )
    assert StageName.PREPARE in state.completed_stages

    resolver = _CountingFrozenResolver(run.prepared.root)
    orchestrator = _frozen_orchestrator(
        run,
        judge=_CountingJudge(),
        resolver=resolver,
    )
    evaluated = _evaluate(run, orchestrator, "post-prepare-no-review-v1")
    assert evaluated.review_result is None
    assert resolver.calls == 1

    binding = run.prepared.manifest.records[0]
    record_path = run.prepared.root / binding.path
    record_path.chmod(0o600)
    record_path.write_bytes(b"drifted after failed Agent evaluation")

    with pytest.raises(MaterializationError, match="bundle trust"):
        orchestrator.load_trial_evaluation(
            run.config.run_id,
            run.trial.task_id,
            run.trial.trial_id,
            evaluated.evaluation_id,
        )
    assert resolver.calls == 2


def test_repository_resolver_is_cache_only_and_load_rejects_cache_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots, _common, run_id = _prepare_and_run_cli_fixture(tmp_path, capsys)
    store = ArtifactStore(roots["runs"], create_root=False)
    config = store.load_run_config(run_id)
    plan = store.load_run_manifest(run_id).trials[0]
    bank = CaseBank.open(roots["suite"])
    descriptor = bank.evaluator_case(plan.task_id).eval_input().review_target.repository
    executable = shutil.which("git")
    assert executable is not None
    require_calls = 0
    original_require = RepositoryPreparer.require_cached

    with RepositoryPreparer(
        suite_root=roots["suite"],
        data_root=roots["data"],
        workspace_root=roots["workspaces"],
        git_executable=Path(executable).absolute(),
        repository_mode=RepositoryMode.CACHE_ONLY,
    ) as preparer:
        prepared = original_require(preparer, descriptor)

        def tracked_require(self: RepositoryPreparer, value: Any):
            nonlocal require_calls
            require_calls += 1
            return original_require(self, value)

        def forbidden_prepare(*_args: Any, **_kwargs: Any):
            raise AssertionError("evaluation called RepositoryPreparer.prepare")

        monkeypatch.setattr(RepositoryPreparer, "require_cached", tracked_require)
        monkeypatch.setattr(RepositoryPreparer, "prepare", forbidden_prepare)
        orchestrator = EvaluationOrchestrator(
            store,
            bank,
            repository_preparer=preparer,
        )
        execution = EvaluatorExecutionConfig.from_resource_budgets(
            config.evaluator,
            config.resource_budgets,
        )
        evaluated = orchestrator.evaluate_trial(
            run_id,
            plan.task_id,
            plan.trial_id,
            evaluator_execution=execution,
            evaluation_revision="repository-cache-only-v1",
        )
        assert require_calls == 1

        manifest_path = prepared.cache_path.parent / "manifest.json"
        manifest_path.write_bytes(b"{}")
        with pytest.raises(RepositoryIntegrityError):
            orchestrator.load_trial_evaluation(
                run_id,
                plan.task_id,
                plan.trial_id,
                evaluated.evaluation_id,
            )


def test_truth_scoped_context_only_enters_its_equivalence_request_and_hydrates(
    tmp_path: Path,
) -> None:
    truth = ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.FORBID,
        expected_findings=(
            _expected("truth-a", "Canonical truth A."),
            _expected("truth-b", "Canonical truth B."),
        ),
        known_invalid_findings=(),
    )
    context = _review_context(
        _truth_context("truth-a", HUNK_A, provenance=_provenance("orchestrator"))
    )
    run = _run_frozen(
        tmp_path,
        _FrozenFindingAdapter("Generated semantic claim."),
        instance="orchestrator-truth-context",
        review_truth=truth,
        review_evaluator_context=context,
    )
    execution = _execution()
    judge = _RecordingJudge(execution)
    orchestrator = _frozen_orchestrator(run, judge=judge)

    evaluated = orchestrator.evaluate_trial(
        run.config.run_id,
        run.trial.task_id,
        run.trial.trial_id,
        evaluator_execution=execution,
        evaluation_revision="truth-context-v1",
    )
    requests = {
        next(
            binding.source_id
            for binding in request.reference_bindings
            if binding.source_kind == "expected_finding"
        ): request
        for request in judge.requests
        if request.task is JudgeTask.FINDING_EQUIVALENCE
    }
    assert [item.content for item in requests["truth-a"].contexts] == [HUNK_A]
    assert HUNK_A not in requests["truth-b"].to_json()

    loaded = orchestrator.load_trial_evaluation(
        run.config.run_id,
        run.trial.task_id,
        run.trial.trial_id,
        evaluated.evaluation_id,
    )
    assert loaded.review_result == evaluated.review_result


class _BrokenMaterializer:
    def materialize(self, _request: Any):
        raise RuntimeError("fixture materialization failure")


def test_pre_materialization_failure_without_review_needs_no_resolver(
    tmp_path: Path,
) -> None:
    run = _run_frozen(
        tmp_path,
        _FrozenSuccessAdapter(),
        instance="pre-materialization-no-review",
        materializer=_BrokenMaterializer(),
    )
    assert run.trial.submission is not None
    assert run.trial.submission.status is SubmissionStatus.FAILED
    assert run.trial.submission.review is None
    with pytest.raises(ArtifactStateError):
        run.store.load_trial_materialization(
            run.config.run_id,
            run.trial.task_id,
            run.trial.trial_id,
        )
    judge = _CountingJudge()
    orchestrator = EvaluationOrchestrator(run.store, run.bank, judge=judge)

    evaluated = _evaluate(run, orchestrator, "pre-materialization-v1")
    loaded = orchestrator.load_trial_evaluation(
        run.config.run_id,
        run.trial.task_id,
        run.trial.trial_id,
        evaluated.evaluation_id,
    )

    assert evaluated.review_result is None
    assert loaded.review_result is None
    assert loaded.trial_score == evaluated.trial_score
    assert judge.calls == 0


def test_target_replay_public_exports_are_canonical() -> None:
    assert target_replay_module.__all__ == [
        "FrozenContextReplayResolver",
        "RepositoryReplayResolver",
        "TargetReplay",
        "TargetReplayIntegrityError",
        "TargetReplayResolver",
        "validate_target_replay",
    ]

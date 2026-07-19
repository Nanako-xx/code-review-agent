"""Composition root for the separated code-review evaluation stages.

The domain modules in :mod:`review_agent_eval` deliberately do not know how a
Run is laid out on disk or how a model provider is constructed.  This module
is the small, explicit bridge used by the Eval CLI.  It loads immutable
Submissions, asks the existing evaluators for bounded Judge requests, executes
those requests through a supplied ``SemanticJudge`` and persists the typed
evaluation artifacts through ``ArtifactStore``.

There is intentionally no product Runtime, Session or Memory dependency here.
The object is also useful to embedding applications which want the same
prepare/run/evaluate separation without invoking the command line parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

from .artifacts import (
    ArtifactError,
    ArtifactStateError,
    ArtifactStore,
    RunManifest,
    StageName,
    TrialManifest,
)
from .cases import EvalCase
from .datasets import CaseBank
from .config import EvalRunConfig, EvaluatorExecutionConfig, derive_evaluation_id
from .intent_evaluator import IntentEvaluationResult, IntentEvaluator
from .judge import (
    BlindJudgeInput,
    JudgeExecutionResult,
    JudgeInputArtifact,
    JudgeOutputArtifact,
    SemanticJudge,
    build_intent_judge_input,
    intent_resolution_from_judge_result,
)
from .metrics import TrialScore, TrialScorer
from .models import EvalSubmission, SchemaError, SubmissionStatus, TrialStatus
from .report import (
    ReportBuilder,
    TrialEvaluationSource,
    TrialInspection,
    RunReportSummary,
    render_run_markdown,
    render_trial_markdown,
)
from .repository import PreparedRepositoryReplay, RepositoryPreparer
from .review_evaluator import ReviewEvaluationResult, ReviewEvaluator


class EvaluationOrchestrationError(RuntimeError):
    """A validly parsed Run cannot be evaluated with the supplied resources."""


class EvaluationPreconditionError(EvaluationOrchestrationError):
    """A required prepare artifact, terminal Submission, or Judge is absent."""


class EvaluationConflictError(EvaluationOrchestrationError):
    """An immutable evaluation namespace already has a different meaning."""


def _wire(value: Any) -> Any:
    """Convert a canonical domain object to JSON-ready values.

    ``ArtifactStore`` performs the final safe/canonical validation.  Keeping
    this conversion here avoids teaching the CLI about every evaluator leaf
    type while still ensuring no dataclass or provider response is persisted.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _wire(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_wire(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _wire(to_dict())
    raise TypeError("evaluation value is not JSON-ready: %s" % type(value).__name__)


def _terminal(status: TrialStatus) -> bool:
    return status in {
        TrialStatus.COMPLETED,
        TrialStatus.FAILED,
        TrialStatus.BLOCKED,
        TrialStatus.INVALID_OUTPUT,
    }


@dataclass(frozen=True)
class TrialEvaluationBundle:
    """All typed outputs produced for one immutable Trial evaluation."""

    run_id: str
    task_id: str
    trial_id: str
    trial_index: int
    evaluation_id: str
    evaluation_revision: str
    evaluator_execution: EvaluatorExecutionConfig
    eval_case: EvalCase
    submission: EvalSubmission
    intent_result: IntentEvaluationResult
    review_result: Optional[ReviewEvaluationResult]
    trial_score: TrialScore
    judge_input: JudgeInputArtifact
    judge_output: JudgeOutputArtifact
    inspection: TrialInspection
    report: str

    def to_dict(self) -> dict[str, Any]:
        """Return a metadata-safe result envelope for CLI JSON output."""

        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "trial_id": self.trial_id,
            "trial_index": self.trial_index,
            "evaluation_id": self.evaluation_id,
            "evaluation_revision": self.evaluation_revision,
            "submission_status": self.submission.status.value,
            "intent_status": self.intent_result.status.value,
            "review_status": (
                None if self.review_result is None else self.review_result.status.value
            ),
            "score_digest": self.trial_score.digest(),
            "report_digest": _sha256_text(self.report),
        }


@dataclass(frozen=True)
class RunEvaluationBundle:
    """The immutable, aggregate result of evaluating all terminal Trials."""

    run_id: str
    evaluation_id: str
    evaluation_revision: str
    evaluator_execution: EvaluatorExecutionConfig
    trials: Tuple[TrialEvaluationBundle, ...]
    summary: RunReportSummary
    report: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "evaluation_id": self.evaluation_id,
            "evaluation_revision": self.evaluation_revision,
            "trial_count": len(self.trials),
            "evaluated_trials": len(self.trials),
            "summary_id": self.summary.summary_id,
            "report_digest": _sha256_text(self.report),
            "trials": [item.to_dict() for item in self.trials],
        }


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class EvaluationOrchestrator:
    """Run evaluator stages without coupling them to Agent execution."""

    def __init__(
        self,
        artifact_store: ArtifactStore,
        case_bank: CaseBank,
        *,
        repository_preparer: Optional[RepositoryPreparer] = None,
        report_builder: Optional[ReportBuilder] = None,
        scorer: Optional[TrialScorer] = None,
        judge: Optional[SemanticJudge] = None,
        judge_factory: Optional[Callable[[], SemanticJudge]] = None,
        max_judge_rounds: int = 8,
    ) -> None:
        if not isinstance(artifact_store, ArtifactStore):
            raise TypeError("artifact_store must be an ArtifactStore")
        if not isinstance(case_bank, CaseBank):
            raise TypeError("case_bank must be a CaseBank")
        if repository_preparer is not None and not isinstance(
            repository_preparer, RepositoryPreparer
        ):
            raise TypeError("repository_preparer must be a RepositoryPreparer")
        if report_builder is not None and not isinstance(report_builder, ReportBuilder):
            raise TypeError("report_builder must be a ReportBuilder")
        if scorer is not None and not isinstance(scorer, TrialScorer):
            raise TypeError("scorer must be a TrialScorer")
        if judge is not None and not isinstance(judge, SemanticJudge):
            # A structural fake is useful in tests, but it must expose the
            # same execute boundary as SemanticJudge.
            if not callable(getattr(judge, "execute", None)):
                raise TypeError("judge must expose execute(request)")
        if judge_factory is not None and not callable(judge_factory):
            raise TypeError("judge_factory must be callable")
        if judge is not None and judge_factory is not None:
            raise ValueError("provide judge or judge_factory, not both")
        if type(max_judge_rounds) is not int or not 1 <= max_judge_rounds <= 64:
            raise ValueError("max_judge_rounds must be between 1 and 64")
        self.artifact_store = artifact_store
        self.case_bank = case_bank
        self.repository_preparer = repository_preparer
        self.report_builder = report_builder or ReportBuilder()
        self.scorer = scorer or TrialScorer()
        self.judge = judge
        self.judge_factory = judge_factory
        self.max_judge_rounds = max_judge_rounds

    def _new_judge(self) -> Any:
        if self.judge_factory is not None:
            value = self.judge_factory()
            if not callable(getattr(value, "execute", None)):
                raise EvaluationOrchestrationError(
                    "judge_factory returned an object without execute(request)"
                )
            return value
        return self.judge

    def _case(self, run_config: EvalRunConfig, task_id: str) -> EvalCase:
        case = self.case_bank.evaluator_case(task_id)
        suite_case = run_config.suite.case(task_id)
        if (
            case.digest() != suite_case.canonical_case_digest
            or case.eval_input().digest() != suite_case.eval_input_digest
            or case.source.suite != run_config.suite.suite_id
        ):
            raise EvaluationOrchestrationError(
                "Suite Case does not match the immutable Run binding"
            )
        return case

    def _replay(
        self,
        case: EvalCase,
    ) -> PreparedRepositoryReplay:
        preparer = self.repository_preparer
        if preparer is None:
            raise EvaluationPreconditionError(
                "evaluate requires a prepared repository cache"
            )
        # Task 12's cache-only method is preferred.  The fallback keeps the
        # orchestrator compatible with older embedders, but never silently
        # acquires a remote repository when the strict method is available.
        require_cached = getattr(preparer, "require_cached", None)
        if callable(require_cached):
            prepared = require_cached(case.input.repository)
        else:
            prepared = preparer.prepare(case.input.repository)
        return preparer.open_replay(prepared)

    @staticmethod
    def _receipt_payloads(
        store: ArtifactStore,
        run_id: str,
        receipt: Any,
    ) -> dict[str, Any]:
        if receipt is None:
            return {}
        refs = getattr(receipt, "artifacts", ())
        values = store.read_json_artifacts(run_id, refs)
        result: dict[str, Any] = {}
        for ref, value in zip(refs, values):
            name = str(getattr(ref, "relative_path", "")).rsplit("/", 1)[-1]
            result[name] = value
        return result

    def _execution_metadata(
        self,
        run_id: str,
        task_id: str,
        trial_id: str,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...], Optional[Any]]:
        """Load private runner metadata through ArtifactStore's verified refs."""

        from .clarification import MaterialClaimMatchReceipt
        from .evidence_checker import CommandOutputAttestation

        state = self.artifact_store.load_trial_state(run_id, task_id, trial_id)
        payloads = self._receipt_payloads(
            self.artifact_store,
            run_id,
            state.terminal_receipt,
        )
        clarification = payloads.get("clarification_match_receipts.json") or {}
        receipts = []
        # Older artifacts did not expose a hydration method.  The canonical
        # receipt wire format is intentionally reconstructed only after the
        # ArtifactStore has verified its hash/size binding.
        for item in clarification.get("receipts", ()):
            receipts.append(_hydrate_match_receipt(item))

        attestations = []
        raw_attestations = payloads.get("command_attestations.json") or {}
        if isinstance(raw_attestations, dict):
            raw_attestations = raw_attestations.get("attestations", ())
        for item in raw_attestations or ():
            attestations.append(CommandOutputAttestation.from_dict(item))
        trace_capture = payloads.get("trace_capture.json")
        return tuple(receipts), tuple(attestations), trace_capture

    def _judge_requests(
        self,
        requests: Sequence[BlindJudgeInput],
        judge: Any,
    ) -> tuple[JudgeExecutionResult, ...]:
        if not requests:
            return ()
        if judge is None:
            raise EvaluationPreconditionError(
                "semantic Judge requests exist; evaluate requires --judge-provider"
            )
        results: list[JudgeExecutionResult] = []
        seen: set[str] = set()
        for _round in range(self.max_judge_rounds):
            pending = [item for item in requests if item.request_id not in seen]
            if not pending:
                return tuple(results)
            for request in pending:
                result = judge.execute(request)
                if not isinstance(result, JudgeExecutionResult):
                    raise EvaluationOrchestrationError(
                        "Judge.execute returned a non-canonical result"
                    )
                if result.request.request_id != request.request_id:
                    raise EvaluationOrchestrationError(
                        "Judge result is bound to a different request"
                    )
                if request.request_id in seen:
                    raise EvaluationOrchestrationError(
                        "Judge returned a duplicate request result"
                    )
                seen.add(request.request_id)
                results.append(result)
        raise EvaluationOrchestrationError(
            "Judge rounds exhausted while requests remain pending"
        )

    def _evaluate_intent(
        self,
        submission: EvalSubmission,
        case: EvalCase,
        receipts: Sequence[Any],
        judge: Any,
    ) -> tuple[IntentEvaluationResult, tuple[BlindJudgeInput, ...], tuple[JudgeExecutionResult, ...]]:
        evaluator = IntentEvaluator()
        intent = submission.intent
        transcript = None if intent is None else intent.clarification_questions
        initial = evaluator.evaluate(
            intent,
            case.intent_truth,
            case.clarification_script,
            transcript=transcript,
            receipts=receipts,
        )
        requests = tuple(build_intent_judge_input(item) for item in initial.judge_requests)
        results = self._judge_requests(requests, judge)
        decisions = []
        failures = []
        ungraded = []
        for result in results:
            decision, failure, missing = intent_resolution_from_judge_result(result)
            if decision is not None:
                decisions.append(decision)
            if failure is not None:
                failures.append(failure)
            if missing is not None:
                ungraded.append(missing)
        final = evaluator.evaluate(
            intent,
            case.intent_truth,
            case.clarification_script,
            transcript=transcript,
            receipts=receipts,
            semantic_decisions=tuple(decisions),
            semantic_failures=tuple(failures),
            semantic_ungraded=tuple(ungraded),
        )
        return final, requests, results

    def _evaluate_review(
        self,
        submission: EvalSubmission,
        case: EvalCase,
        run_config: EvalRunConfig,
        execution: EvaluatorExecutionConfig,
        trial_id: str,
        replay: PreparedRepositoryReplay,
        attestations: Sequence[Any],
        judge: Any,
    ) -> tuple[Optional[ReviewEvaluationResult], tuple[BlindJudgeInput, ...], tuple[JudgeExecutionResult, ...]]:
        if submission.review is None:
            return None, (), ()
        evaluator = ReviewEvaluator(
            eval_input=case.eval_input(),
            replay=replay,
            trial_id=trial_id,
            evaluator_execution=execution,
            command_attestations=tuple(attestations),
        )
        initial = evaluator.evaluate(submission, case.review_truth)
        requests = tuple(item.request for item in initial.judge_requests)
        results = self._judge_requests(requests, judge)
        final = evaluator.evaluate(
            submission,
            case.review_truth,
            judge_results=results,
        )
        return final, requests, results

    def evaluate_trial(
        self,
        run_id: str,
        task_id: str,
        trial_id: str,
        *,
        evaluator_execution: EvaluatorExecutionConfig,
        evaluation_revision: str,
        judge: Any = None,
        resume: bool = False,
    ) -> TrialEvaluationBundle:
        """Evaluate one already terminal Trial and persist its namespace."""

        if not isinstance(evaluator_execution, EvaluatorExecutionConfig):
            raise TypeError("evaluator_execution must be EvaluatorExecutionConfig")
        if type(resume) is not bool:
            raise TypeError("resume must be a bool")
        evaluation_id = derive_evaluation_id(
            run_id,
            evaluator_execution.digest(),
            evaluation_revision,
        )
        if resume:
            try:
                return self.load_trial_evaluation(
                    run_id,
                    task_id,
                    trial_id,
                    evaluation_id,
                )
            except ArtifactStateError:
                # A missing/uncommitted namespace is resumed by the normal
                # create-only writer below.  Integrity/security failures are
                # deliberately not swallowed.
                pass
        config = self.artifact_store.load_run_config(run_id)
        if config.run_id != run_id:
            raise EvaluationOrchestrationError("Run config identity mismatch")
        manifest = self.artifact_store.load_trial_manifest(run_id, task_id, trial_id)
        submission = self.artifact_store.load_existing_submission(run_id, task_id, trial_id)
        case = self._case(config, task_id)
        if submission.trial_id != trial_id:
            raise EvaluationOrchestrationError("Submission is bound to another Trial")
        receipts, attestations, trace_capture = self._execution_metadata(
            run_id, task_id, trial_id
        )
        replay = self._replay(case)
        actual_judge = judge if judge is not None else self._new_judge()
        intent_result, intent_requests, intent_results = self._evaluate_intent(
            submission, case, receipts, actual_judge
        )
        review_result, review_requests, review_results = self._evaluate_review(
            submission,
            case,
            config,
            evaluator_execution,
            trial_id,
            replay,
            attestations,
            actual_judge,
        )
        all_requests = tuple(
            sorted(
                (*intent_requests, *review_requests),
                key=lambda item: item.request_id,
            )
        )
        all_results = tuple(
            sorted(
                (*intent_results, *review_results),
                key=lambda item: item.request.request_id,
            )
        )
        judge_input = JudgeInputArtifact.create(evaluator_execution, all_requests)
        judge_output = JudgeOutputArtifact.create(
            judge_input,
            evaluator_execution,
            all_results,
            intent_evaluation=intent_result if intent_requests else None,
        )
        score = self.scorer.score(
            run_config=config,
            evaluator_execution=evaluator_execution,
            evaluation_revision=evaluation_revision,
            eval_case=case,
            submission=submission,
            trial_index=manifest.trial_index,
            intent_result=intent_result,
            review_result=review_result,
        )
        source = TrialEvaluationSource(
            eval_case=case,
            submission=submission,
            intent_result=intent_result,
            review_result=review_result,
            trial_score=score,
            trial_index=manifest.trial_index,
            trial_id=trial_id,
            trial_manifest=manifest,
            trace_capture=trace_capture,
        )
        inspection = self.report_builder.build_inspection(
            run_config=config,
            evaluator_execution=evaluator_execution,
            evaluation_revision=evaluation_revision,
            source=source,
            run_manifest=self.artifact_store.load_run_manifest(run_id),
        )
        report = render_trial_markdown(inspection)
        receipt = self.artifact_store.write_evaluation(
            run_id,
            task_id,
            trial_id,
            evaluator_execution=evaluator_execution,
            revision=evaluation_revision,
            intent_matches=_wire(intent_result),
            review_matches=None if review_result is None else _wire(review_result),
            judge_input=_wire(judge_input),
            judge_output=_wire(judge_output),
            score=_wire(score),
            report=report,
        )
        committed_evaluation_id = receipt.evaluation_id
        if committed_evaluation_id is None:
            raise EvaluationOrchestrationError("Evaluator receipt has no evaluation ID")
        return TrialEvaluationBundle(
            run_id=run_id,
            task_id=task_id,
            trial_id=trial_id,
            trial_index=manifest.trial_index,
            evaluation_id=committed_evaluation_id,
            evaluation_revision=evaluation_revision,
            evaluator_execution=evaluator_execution,
            eval_case=case,
            submission=submission,
            intent_result=intent_result,
            review_result=review_result,
            trial_score=score,
            judge_input=judge_input,
            judge_output=judge_output,
            inspection=inspection,
            report=report,
        )

    def load_trial_evaluation(
        self,
        run_id: str,
        task_id: str,
        trial_id: str,
        evaluation_id: str,
    ) -> TrialEvaluationBundle:
        """Strictly hydrate one committed evaluation through public store APIs."""

        stored = self.artifact_store.load_evaluation_bundle(
            run_id,
            task_id,
            trial_id,
            evaluation_id,
        )
        config = self.artifact_store.load_run_config(run_id)
        manifest = self.artifact_store.load_trial_manifest(
            run_id, task_id, trial_id
        )
        submission = self.artifact_store.load_existing_submission(
            run_id, task_id, trial_id
        )
        case = self._case(config, task_id)
        execution = stored.evaluator_execution
        intent_result = IntentEvaluationResult.from_dict(stored.intent_matches)
        judge_input = JudgeInputArtifact.from_dict(
            stored.judge_input,
            evaluator_execution=execution,
        )
        raw_output = stored.judge_output
        bound_intent = (
            intent_result
            if isinstance(raw_output, Mapping)
            and raw_output.get("intent_evaluation_digest") is not None
            else None
        )
        judge_output = JudgeOutputArtifact.from_dict(
            raw_output,
            input_artifact=judge_input,
            evaluator_execution=execution,
            intent_evaluation=bound_intent,
        )
        receipts, attestations, trace_capture = self._execution_metadata(
            run_id, task_id, trial_id
        )
        del receipts
        review_result = None
        replay = self._replay(case)
        if stored.review_matches is not None:
            reviewer = ReviewEvaluator(
                eval_input=case.eval_input(),
                replay=replay,
                trial_id=trial_id,
                evaluator_execution=execution,
                command_attestations=tuple(attestations),
            )
            review_result = ReviewEvaluationResult.from_dict(
                stored.review_matches,
                submission=submission,
                review_truth=case.review_truth,
                evaluator=reviewer,
                judge_results=judge_output.results,
            )
        score = TrialScore.from_dict(
            stored.score,
            scorer=self.scorer,
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision=stored.evaluation_revision,
            eval_case=case,
            submission=submission,
            trial_index=manifest.trial_index,
            intent_result=intent_result,
            review_result=review_result,
        )
        source = TrialEvaluationSource(
            eval_case=case,
            submission=submission,
            intent_result=intent_result,
            review_result=review_result,
            trial_score=score,
            trial_index=manifest.trial_index,
            trial_id=trial_id,
            trial_manifest=manifest,
            trace_capture=trace_capture,
        )
        inspection = self.report_builder.build_inspection(
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision=stored.evaluation_revision,
            source=source,
            run_manifest=self.artifact_store.load_run_manifest(run_id),
        )
        rendered = render_trial_markdown(inspection)
        if stored.report is not None and stored.report != rendered:
            raise EvaluationOrchestrationError(
                "persisted Trial report differs from source-bound replay"
            )
        return TrialEvaluationBundle(
            run_id=run_id,
            task_id=task_id,
            trial_id=trial_id,
            trial_index=manifest.trial_index,
            evaluation_id=stored.evaluation_id,
            evaluation_revision=stored.evaluation_revision,
            evaluator_execution=execution,
            eval_case=case,
            submission=submission,
            intent_result=intent_result,
            review_result=review_result,
            trial_score=score,
            judge_input=judge_input,
            judge_output=judge_output,
            inspection=inspection,
            report=rendered,
        )

    def evaluate_run(
        self,
        run_id: str,
        *,
        evaluator_execution: EvaluatorExecutionConfig,
        evaluation_revision: str,
        judge: Any = None,
        task_ids: Optional[Iterable[str]] = None,
        resume: bool = False,
    ) -> RunEvaluationBundle:
        """Evaluate all selected terminal Trials and persist a Run report."""

        config = self.artifact_store.load_run_config(run_id)
        selected = None if task_ids is None else set(task_ids)
        manifest = self.artifact_store.load_run_manifest(run_id)
        planned_task_ids = {item.task_id for item in manifest.trials}
        if selected is not None and selected != planned_task_ids:
            raise EvaluationPreconditionError(
                "partial Run evaluation requires a separately filtered Run identity"
            )
        bundles = []
        for plan in manifest.trials:
            if selected is not None and plan.task_id not in selected:
                continue
            state = self.artifact_store.load_trial_state(run_id, plan.task_id, plan.trial_id)
            if not _terminal(state.status):
                raise EvaluationPreconditionError(
                    "evaluate requires every selected Trial to have a terminal Submission"
                )
            bundles.append(
                self.evaluate_trial(
                    run_id,
                    plan.task_id,
                    plan.trial_id,
                    evaluator_execution=evaluator_execution,
                    evaluation_revision=evaluation_revision,
                    judge=judge,
                    resume=resume,
                )
            )
        if not bundles:
            raise EvaluationPreconditionError("Run has no terminal Trials to evaluate")
        bundles = sorted(bundles, key=lambda item: (item.task_id, item.trial_index))
        sources = tuple(
            TrialEvaluationSource(
                eval_case=item.eval_case,
                submission=item.submission,
                intent_result=item.intent_result,
                review_result=item.review_result,
                trial_score=item.trial_score,
                trial_index=item.trial_index,
                trial_id=item.trial_id,
                trial_manifest=self.artifact_store.load_trial_manifest(
                    run_id,
                    item.task_id,
                    item.trial_id,
                ),
            )
            for item in bundles
        )
        cases_by_task = {item.task_id: item.eval_case for item in bundles}
        cases = tuple(cases_by_task[key] for key in sorted(cases_by_task))
        summary = self.report_builder.build_summary(
            run_config=config,
            evaluator_execution=evaluator_execution,
            evaluation_revision=evaluation_revision,
            cases=cases,
            sources=sources,
            run_manifest=manifest,
        )
        report = render_run_markdown(summary)
        write_run = getattr(self.artifact_store, "write_run_evaluation", None)
        if callable(write_run):
            write_run(
                run_id,
                evaluator_execution=evaluator_execution,
                revision=evaluation_revision,
                summary=summary,
                report=report,
                resume=resume,
                overwrite=False,
            )
        return RunEvaluationBundle(
            run_id=run_id,
            evaluation_id=bundles[0].evaluation_id,
            evaluation_revision=evaluation_revision,
            evaluator_execution=evaluator_execution,
            trials=tuple(bundles),
            summary=summary,
            report=report,
        )


def _hydrate_match_receipt(value: Any) -> Any:
    """Hydrate the Runner's private clarification receipt wire format."""

    from .clarification import (
        MaterialClaimCandidateDecision,
        MaterialClaimMatchOutcome,
        MaterialClaimMatchReceipt,
    )
    from .models import IntentDimension

    if not isinstance(value, Mapping):
        raise SchemaError("clarification receipt must be an object")
    expected = {
        "turn_index",
        "question_id",
        "dimension",
        "actual_claim_digest",
        "matcher_digest",
        "candidates",
        "outcome",
        "matched_answer_id",
    }
    if set(value) != expected or not isinstance(value["candidates"], (list, tuple)):
        raise SchemaError("clarification receipt has unexpected fields")
    candidates = []
    for item in value["candidates"]:
        if not isinstance(item, Mapping) or set(item) != {
            "answer_id",
            "request_digest",
            "equivalent",
            "action_eligible",
        }:
            raise SchemaError("clarification receipt candidate is invalid")
        candidates.append(MaterialClaimCandidateDecision(**dict(item)))
    return MaterialClaimMatchReceipt(
        turn_index=value["turn_index"],
        question_id=value["question_id"],
        dimension=IntentDimension(value["dimension"]),
        actual_claim_digest=value["actual_claim_digest"],
        matcher_digest=value["matcher_digest"],
        candidates=tuple(candidates),
        outcome=MaterialClaimMatchOutcome(value["outcome"]),
        matched_answer_id=value["matched_answer_id"],
    )


__all__ = [
    "EvaluationOrchestrationError",
    "EvaluationPreconditionError",
    "EvaluationConflictError",
    "TrialEvaluationBundle",
    "RunEvaluationBundle",
    "EvaluationOrchestrator",
]

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
AUTHORING_ROOT = Path(__file__).resolve().parent
if str(AUTHORING_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTHORING_ROOT))

from core_human_review import (  # noqa: E402
    HumanReviewError,
    verify_current_case_approval,
)
from review_agent_eval.adapters.current_agent import CurrentAgentAdapter  # noqa: E402
from review_agent_eval.artifacts import ArtifactStore  # noqa: E402
from review_agent_eval.cases import (  # noqa: E402
    REPOSITORY_MATERIALIZER_PROTOCOL,
    CaseSplit,
    SuiteKind,
)
from review_agent_eval.config import SuiteRunConfig  # noqa: E402
from review_agent_eval.datasets import CaseBank  # noqa: E402
from review_agent_eval.intent_evaluator import IntentEvaluationStatus  # noqa: E402
from review_agent_eval.models import (  # noqa: E402
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    ReviewTargetKind,
    SubmissionStatus,
)
from review_agent_eval.orchestrator import EvaluationOrchestrator  # noqa: E402
from review_agent_eval.repository import RepositoryPreparer  # noqa: E402
from review_agent_eval.review_evaluator import ReviewEvaluationStatus  # noqa: E402
from review_agent_eval.runner import (  # noqa: E402
    CapabilityPolicy,
    CapabilityPreflight,
)


PROMOTION_RUN_INSTANCE_KEY = "core-regression-promotion-v2"
PROMOTION_PROTOCOL_ID = "native_repository"
MINIMUM_PROMOTION_TRIALS = 3
_COMMIT_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


class CoreRegressionPromotionError(RuntimeError):
    """A source-bound Run does not satisfy the Core promotion contract."""


def _fail(message: str) -> None:
    raise CoreRegressionPromotionError(message)


def _is_non_real_model_identity(value: Any) -> bool:
    if type(value) is not str or not value.strip():
        return True
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", value.casefold())
        if token
    }
    return bool(
        tokens & {"none", "fake", "fixture", "mock", "scripted", "test", "unknown"}
    )


def _promotion_locator(value: Any) -> dict[str, str]:
    if type(value) is not dict or set(value) != {
        "run_id",
        "evaluation_id",
        "summary_id",
    }:
        _fail("promotion evidence must contain only run/evaluation/summary locators")
    result: dict[str, str] = {}
    for name in ("run_id", "evaluation_id", "summary_id"):
        item = value[name]
        if type(item) is not str or not item.strip():
            _fail("promotion evidence %s is invalid" % name)
        result[name] = item
    return result


def _assert_current_agent(config: Any, preflight: CapabilityPreflight) -> None:
    agent = config.agent
    parameters = agent.parameters
    if not isinstance(parameters, Mapping):
        _fail("Agent parameters do not contain a canonical adapter snapshot")
    adapter = parameters.get("adapter")
    if not isinstance(adapter, Mapping):
        _fail("Agent parameters do not contain an adapter object")
    if adapter.get("kind") != CurrentAgentAdapter.ADAPTER_KIND:
        _fail("promotion Run did not execute the current Agent adapter")
    if (
        preflight.adapter_id != CurrentAgentAdapter.ADAPTER_KIND
        or preflight.adapter_version != CurrentAgentAdapter.ADAPTER_VERSION
        or preflight.agent_id != agent.agent_id
    ):
        _fail("preflight adapter identity differs from the current Agent")
    if _COMMIT_RE.fullmatch(agent.commit) is None:
        _fail("promotion Agent commit must be a full immutable Git object ID")
    if agent.agent_version.casefold() in {"working-tree", "unknown"}:
        _fail("promotion Agent version must identify a release baseline")
    if _is_non_real_model_identity(agent.provider):
        _fail("promotion Run must use a real model provider")
    if _is_non_real_model_identity(agent.model):
        _fail("promotion Run must bind a real model")


def _assert_v2_repository_suite(handle: Any) -> None:
    manifest = handle.manifest
    contract = manifest.wire_contract
    if (
        manifest.source.kind is not SuiteKind.CORE
        or manifest.source.preparation_binding is not None
        or contract.case_schema_version != EVAL_CASE_SCHEMA_VERSION
        or contract.input_schema_version != EVAL_INPUT_SCHEMA_VERSION
        or contract.submission_schema_version != EVAL_SUBMISSION_SCHEMA_VERSION
        or contract.review_target_kind is not ReviewTargetKind.REPOSITORY
        or contract.materializer_protocol != REPOSITORY_MATERIALIZER_PROTOCOL
    ):
        _fail("promotion Case does not belong to the sole Core v2 Repository contract")


def _assert_preflight(config: Any, manifest: Any, raw: Any) -> None:
    try:
        preflight = CapabilityPreflight.from_dict(raw)
    except Exception as exc:
        raise CoreRegressionPromotionError(
            "promotion capability preflight is invalid"
        ) from exc
    expected_trials = {
        (item.task_id, item.trial_index) for item in manifest.trials
    }
    if (
        preflight.run_id != config.run_id
        or preflight.policy is not CapabilityPolicy.STRICT
        or preflight.filtered_from_run_id is not None
        or not preflight.compatible
        or preflight.issues
        or set(preflight.checked_trials) != expected_trials
        or set(preflight.compatible_task_ids)
        != {item.task_id for item in config.suite.cases}
    ):
        _fail("promotion Run requires complete, unfiltered strict preflight")
    _assert_current_agent(config, preflight)


def _assert_trial_pass(trial: Any, case: Any) -> None:
    if trial.submission.status is not SubmissionStatus.COMPLETED:
        _fail("promotion Trial did not complete: %s" % trial.trial_id)
    intent = trial.intent_result
    if (
        intent.status is not IntentEvaluationStatus.GRADED
        or intent.metrics.intent_case_pass is not True
        or intent.judge_failures
        or intent.judge_ungraded
    ):
        _fail("promotion Trial Intent did not pass: %s" % trial.trial_id)
    review = trial.review_result
    if review is None or review.status is not ReviewEvaluationStatus.GRADED:
        _fail("promotion Trial Review is absent or ungraded: %s" % trial.trial_id)
    coverage = review.coverage
    if (
        review.judge_failures
        or review.judge_ungraded
        or coverage.judge_failed_count
        or coverage.judge_ungraded_count
        or coverage.judge_pending_count
        or coverage.semantic_unknown_count
    ):
        _fail("promotion Trial has unresolved semantic grading: %s" % trial.trial_id)
    metrics = review.metrics
    if (
        metrics.unmatched_required_truth_count
        or metrics.duplicate_finding_count
        or metrics.known_invalid_finding_count
        or metrics.plausible_novel_count
        or metrics.fabricated_finding_count
        or metrics.unknown_finding_count
    ):
        _fail("promotion Trial Review violates required truth/precision gates: %s" % trial.trial_id)
    if (
        metrics.evidence_invalid_count
        or metrics.evidence_missing_count
        or metrics.evidence_weak_count
        or metrics.evidence_unsupported_count
        or metrics.evidence_support_unknown_count
        or metrics.evidence_valid_count != metrics.generated_finding_count
        or metrics.evidence_supported_count != metrics.generated_finding_count
        or metrics.strict_publishable_count != metrics.generated_finding_count
        or len(review.finding_outcomes) != metrics.generated_finding_count
        or any(not item.strict_publishable for item in review.finding_outcomes)
    ):
        _fail(
            "promotion Trial requires valid+supported Evidence and strict "
            "publishability for every generated Finding: %s" % trial.trial_id
        )
    matched_outcomes = {
        item.matched_expected_truth_id: item
        for item in review.finding_outcomes
        if item.matched_expected_truth_id is not None
    }
    required_ids = {
        item.truth_id
        for item in case.review_truth.expected_findings
        if item.required
    }
    if set(matched_outcomes).intersection(required_ids) != required_ids or any(
        not matched_outcomes[truth_id].strict_publishable for truth_id in required_ids
    ):
        _fail(
            "promotion Trial lacks valid+supported Evidence for required Findings: %s"
            % trial.trial_id
        )


def verify_case_promotion(
    *,
    case_bank: CaseBank,
    artifact_store: ArtifactStore,
    repository_preparer: RepositoryPreparer,
    task_id: str,
    promotion_evidence: Mapping[str, str],
) -> dict[str, str]:
    """Replay original artifacts and verify one Case's 3/3 promotion evidence."""

    if not isinstance(case_bank, CaseBank):
        raise TypeError("case_bank must be CaseBank")
    if not isinstance(artifact_store, ArtifactStore):
        raise TypeError("artifact_store must be ArtifactStore")
    if not isinstance(repository_preparer, RepositoryPreparer):
        raise TypeError("repository_preparer must be RepositoryPreparer")
    try:
        verify_current_case_approval(case_bank.root, task_id)
    except HumanReviewError as exc:
        raise CoreRegressionPromotionError(
            "promotion Case lacks source-bound human approval"
        ) from exc
    locator = _promotion_locator(dict(promotion_evidence))
    handle = case_bank.handle(task_id)
    _assert_v2_repository_suite(handle)
    if handle.split is not CaseSplit.REGRESSION:
        _fail("only a current Regression manifest Case may use promotion evidence")
    case = handle.load()
    if handle.entry.protocol_id != PROMOTION_PROTOCOL_ID:
        _fail("promotion Case does not use the native repository protocol")

    run_id = locator["run_id"]
    config = artifact_store.load_run_config(run_id)
    manifest = artifact_store.load_run_manifest(run_id)
    snapshot = case_bank.snapshot()
    expected_suite = SuiteRunConfig.from_case_snapshot(snapshot)
    if config.suite != expected_suite:
        _fail("promotion Run Suite/Case snapshot differs from the current manifest")
    if config.run_instance_key != PROMOTION_RUN_INSTANCE_KEY:
        _fail("promotion Run uses an unregistered run-instance key")
    if config.trial_count < MINIMUM_PROMOTION_TRIALS:
        _fail("promotion Run must plan at least three Trials per Case")
    _assert_preflight(
        config,
        manifest,
        artifact_store.load_run_preflight(run_id),
    )

    bundle = EvaluationOrchestrator(
        artifact_store,
        case_bank,
        repository_preparer=repository_preparer,
    ).load_run_evaluation(run_id, locator["evaluation_id"])
    if bundle.summary.summary_id != locator["summary_id"]:
        _fail("promotion summary locator differs from source-bound Run summary")
    if (
        bundle.evaluator_execution.digest()
        != manifest.initial_evaluator_execution_digest
    ):
        _fail("promotion selected a non-initial evaluator execution")
    trials = tuple(item for item in bundle.trials if item.task_id == task_id)
    if (
        len(trials) != config.trial_count
        or {item.trial_index for item in trials}
        != set(range(1, config.trial_count + 1))
    ):
        _fail("promotion evidence omits or selects target Case Trials")
    for trial in trials:
        if (
            trial.eval_case.digest() != handle.entry.canonical_case_digest
            or trial.eval_case.eval_input().digest() != handle.entry.eval_input_digest
        ):
            _fail("promotion Trial Case binding differs from the current manifest")
        _assert_trial_pass(trial, case)
    return locator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Source-bind and verify one Core Regression promotion",
    )
    parser.add_argument("--suite-root", type=Path, default=REPOSITORY_ROOT / "eval")
    parser.add_argument(
        "--manifest",
        default="suites/core-regression/manifest.json",
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--summary-id", required=True)
    args = parser.parse_args(argv)
    git = shutil.which("git")
    if git is None:
        parser.error("Git is required to replay Core repositories")
    bank = CaseBank.open(args.suite_root, args.manifest)
    store = ArtifactStore(args.runs_root, create_root=False)
    with RepositoryPreparer(
        suite_root=args.suite_root,
        data_root=args.data_root,
        workspace_root=args.workspace_root,
        git_executable=Path(git).absolute(),
    ) as preparer:
        locator = verify_case_promotion(
            case_bank=bank,
            artifact_store=store,
            repository_preparer=preparer,
            task_id=args.task_id,
            promotion_evidence={
                "run_id": args.run_id,
                "evaluation_id": args.evaluation_id,
                "summary_id": args.summary_id,
            },
        )
    print(json.dumps(locator, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from review_agent.model_protocol import ModelResponseKind, ModelTurnResponse
from review_agent_eval.analysis_artifacts import AnalysisArtifactStore
from review_agent_eval.artifacts import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactSecurityError,
)
from review_agent_eval.calibration import (
    CALIBRATION_ALGORITHM_VERSION,
    CALIBRATION_SELECTION_POLICY_SCHEMA_VERSION,
    CalibrationPackageV1,
    CalibrationResultV1,
    CalibrationSelectionPolicyV1,
    CalibrationStatus,
    HumanLabelSetV1,
    HumanLabelV1,
    export_calibration_package,
    score_calibration,
)
from review_agent_eval.comparison import VerifiedRunEvaluation
from review_agent_eval.intent_evaluator import IntentTruth
from review_agent_eval.judge import (
    BlindJudgeInput,
    JudgeTask,
    SemanticJudge,
)
from review_agent_eval.models import (
    FindingSeverity,
    MetricAuthority,
    MetricAuthoritySource,
    NovelFindingPolicy,
    RequiredContextLevel,
    ReviewTruth,
    SubmissionFinding,
    SubmissionReview,
    TruthCompleteness,
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    stable_id,
)

from .test_judge import _Factory, _execution
from .test_orchestrator_target_replay_v2 import (
    _FrozenFindingAdapter,
    _expected,
    _frozen_orchestrator,
    _run_frozen,
)
from .test_target_runner import _FrozenSuccessAdapter


POLICY = CalibrationSelectionPolicyV1(
    schema_version=CALIBRATION_SELECTION_POLICY_SCHEMA_VERSION,
    algorithm_version=CALIBRATION_ALGORITHM_VERSION,
    selection_seed=20260726,
    max_items_per_profile=32,
    max_normal_items_per_stratum=16,
    minimum_human_labels=1,
    minimum_human_coverage_ppm=1_000_000,
    minimum_labels_per_class=0,
    minimum_exact_agreement_ppm=900_000,
    minimum_cohen_kappa_ppm=800_000,
)


class _TwoFindingAdapter(_FrozenFindingAdapter):
    def run(self, *args: Any, **kwargs: Any):
        submission = super().run(*args, **kwargs)
        first = submission.review.findings[0]
        second = SubmissionFinding(
            finding_id="finding-frozen-paraphrase",
            claim=(
                "The frozen record retains an authorization path after the "
                "guard should reject it."
            ),
            severity=FindingSeverity.HIGH,
            path=None,
            side=None,
            from_line=None,
            to_line=None,
            evidence_refs=first.evidence_refs,
            suggested_action="Reject the unauthorized path before returning.",
        )
        return replace(
            submission,
            review=SubmissionReview(
                findings=(first, second),
                uncertainties=(),
            ),
        )


class _ProfileScriptJudge:
    def __init__(self, execution: Any) -> None:
        self.execution = execution
        self.calls = 0

    def execute(self, request: BlindJudgeInput):
        self.calls += 1
        fields: dict[str, Any]
        if request.task is JudgeTask.INTENT_EQUIVALENCE:
            fields = {"relation": "equivalent", "score_ppm": 930_000}
        elif request.task is JudgeTask.FINDING_EQUIVALENCE:
            fields = {
                "relation": (
                    "different"
                    if "incorrect review behavior" in request.items[0].text
                    else "equivalent"
                ),
                "score_ppm": 910_000,
                "severity_assessment": "consistent",
                "actionability": "actionable",
            }
        elif request.task is JudgeTask.NOVEL_FACTUALITY:
            fields = {
                "factuality": "plausible",
                "severity_assessment": "consistent",
                "actionability": "actionable",
            }
        else:
            fields = {"support": "supported"}
        output = {
            "schema_version": request.rubric.response_schema,
            "request_id": request.request_id,
            "reason_refs": ["item-a"],
            **fields,
        }
        output_text = json.dumps(output, separators=(",", ":"), sort_keys=True)
        for key in output:
            escaped = "\\u%04x%s" % (ord(key[0]), key[1:])
            output_text = output_text.replace(f'"{key}"', f'"{escaped}"')
        return SemanticJudge(
            adapter_factory=_Factory(
                [
                    ModelTurnResponse(
                        kind=ModelResponseKind.FINAL,
                        final_text=output_text,
                    )
                ]
            ),
            evaluator_execution=self.execution,
        ).execute(request)


@pytest.fixture(scope="module")
def calibration_source(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("calibration-source")
    intent_truth = IntentTruth.from_dict(
        {
            "scorable": True,
            "authority": "synthetic",
            "expected_claims": [
                {
                    "truth_id": "intent-authorization",
                    "dimension": "goal",
                    "text": "Support a dry-run mode without changing persisted state.",
                    "required": True,
                }
            ],
            "forbidden_claims": [],
            "clarification_policy": "not_required",
        }
    )
    execution = _execution()
    judges = []

    def evaluate(profile: JudgeTask, adapter: Any, **run_kwargs: Any):
        source_root = root / profile.value
        source_root.mkdir()
        run = _run_frozen(
            source_root,
            adapter,
            instance="blind-calibration-" + profile.value,
            **run_kwargs,
        )
        judge = _ProfileScriptJudge(execution)
        judges.append(judge)
        orchestrator = _frozen_orchestrator(run, judge=judge)
        evaluated = orchestrator.evaluate_run(
            run.config.run_id,
            evaluator_execution=execution,
            evaluation_revision="calibration-source-v1",
        )
        verified = VerifiedRunEvaluation.create(
            evaluated,
            run_config=run.config,
            case_snapshot=run.snapshot,
        )
        assert any(
            result.request.task is profile
            for trial in verified.trials
            for result in trial.judge_output.results
        )
        return verified, run

    verified_by_profile = {}
    runs = {}
    verified_by_profile[JudgeTask.INTENT_EQUIVALENCE], runs[JudgeTask.INTENT_EQUIVALENCE] = evaluate(
        JudgeTask.INTENT_EQUIVALENCE,
        _FrozenSuccessAdapter(),
        intent_truth=intent_truth,
    )
    finding_truth = ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.FORBID,
        expected_findings=(
            _expected(
                "truth-authorization",
                "An authorization guard permits a protected path to continue.",
            ),
        ),
        known_invalid_findings=(),
    )
    verified_by_profile[JudgeTask.FINDING_EQUIVALENCE], runs[JudgeTask.FINDING_EQUIVALENCE] = evaluate(
        JudgeTask.FINDING_EQUIVALENCE,
        _TwoFindingAdapter(
            "The frozen context demonstrates an incorrect review behavior.",
            with_evidence=False,
        ),
        review_truth=finding_truth,
    )
    novel_truth = ReviewTruth(
        completeness=TruthCompleteness.HUMAN_OBSERVED,
        novel_finding_policy=NovelFindingPolicy.VERIFY,
        expected_findings=(),
        known_invalid_findings=(),
    )
    verified_by_profile[JudgeTask.NOVEL_FACTUALITY], runs[JudgeTask.NOVEL_FACTUALITY] = evaluate(
        JudgeTask.NOVEL_FACTUALITY,
        _FrozenFindingAdapter(
            "The frozen context demonstrates an incorrect review behavior.",
            with_evidence=False,
        ),
        review_truth=novel_truth,
    )
    evidence_claim = "The frozen context demonstrates an evidence-backed defect."
    evidence_truth = ReviewTruth(
        completeness=TruthCompleteness.HUMAN_OBSERVED,
        novel_finding_policy=NovelFindingPolicy.VERIFY,
        expected_findings=(_expected("truth-evidence", evidence_claim),),
        known_invalid_findings=(),
    )
    verified_by_profile[JudgeTask.EVIDENCE_SUPPORT], runs[JudgeTask.EVIDENCE_SUPPORT] = evaluate(
        JudgeTask.EVIDENCE_SUPPORT,
        _FrozenFindingAdapter(evidence_claim, with_evidence=True),
        review_truth=evidence_truth,
    )
    return {
        "verified_by_profile": verified_by_profile,
        "judges": judges,
        "runs": runs,
    }


def _export(
    source: dict[str, Any],
    tmp_path: Path,
    profile: JudgeTask,
    *,
    policy: CalibrationSelectionPolicyV1 = POLICY,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return export_calibration_package(
        source["verified_by_profile"][profile],
        profile=profile,
        policy=policy,
        output_root=tmp_path / "external-calibration",
    )


def _human_label(
    package: Any,
    item: Any,
    label: str,
    *,
    disputed: bool = False,
    adjudication_ref: str | None = None,
) -> HumanLabelV1:
    return HumanLabelV1.create(
        package=package,
        item=item,
        label=label,
        severity_assessment=(
            "consistent"
            if package.profile
            in {JudgeTask.FINDING_EQUIVALENCE, JudgeTask.NOVEL_FACTUALITY}
            else None
        ),
        actionability=(
            "actionable"
            if package.profile
            in {JudgeTask.FINDING_EQUIVALENCE, JudgeTask.NOVEL_FACTUALITY}
            else None
        ),
        reviewer_id="independent-reviewer-1",
        reviewer_provenance="external-human-review-v1",
        blind_attestation=True,
        labeled_at="2026-07-26T10:00:00Z",
        disputed=disputed,
        adjudication_ref=adjudication_ref,
    )


def _matching_labels(package: Any) -> HumanLabelSetV1:
    fixed_profile_labels = {
        JudgeTask.INTENT_EQUIVALENCE: "equivalent",
        JudgeTask.NOVEL_FACTUALITY: "plausible",
        JudgeTask.EVIDENCE_SUPPORT: "supported",
    }
    labels = []
    for item in package.items:
        label = fixed_profile_labels.get(package.profile)
        if package.profile is JudgeTask.FINDING_EQUIVALENCE:
            submitted_claim = item.blinded_request_payload["items"][0]["text"]
            label = (
                "different"
                if "incorrect review behavior" in submitted_claim
                else "equivalent"
            )
        assert label is not None
        labels.append(
            _human_label(package, item, label)
        )
    return HumanLabelSetV1.create(package=package, labels=labels)


def _walk_keys(value: Any):
    if type(value) is dict:
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif type(value) is list:
        for child in value:
            yield from _walk_keys(child)


def _walk_strings(value: Any):
    if type(value) is str:
        yield value
    elif type(value) is dict:
        for child in value.values():
            yield from _walk_strings(child)
    elif type(value) is list:
        for child in value:
            yield from _walk_strings(child)


def test_export_hides_agent_baseline_candidate_and_judge_decision(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    package = _export(
        calibration_source,
        tmp_path,
        JudgeTask.FINDING_EQUIVALENCE,
    )
    exported = json.loads(
        (
            tmp_path
            / "external-calibration"
            / package.package_id
            / "calibration_package.json"
        ).read_text(encoding="utf-8")
    )

    forbidden_keys = {
        "agent",
        "agent_id",
        "agent_name",
        "baseline",
        "candidate",
        "expected_winner",
        "decision",
        "judge_result",
        "judge_result_digest",
        "failure",
        "provider",
        "model",
        "source_request_id",
        "request_id",
    }
    assert forbidden_keys.isdisjoint(set(_walk_keys(exported)))
    agent = calibration_source["verified_by_profile"][JudgeTask.FINDING_EQUIVALENCE].run_config.agent
    forbidden_values = {
        agent.agent_id,
        agent.agent_name,
        agent.agent_version,
        agent.model,
        agent.provider,
    }
    assert forbidden_values.isdisjoint(set(_walk_strings(exported)))
    assert exported["payload_digest"] == package.payload_digest
    assert exported["items"]


def test_selection_policy_is_seeded_and_recorded(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    first = _export(
        calibration_source,
        tmp_path / "first",
        JudgeTask.FINDING_EQUIVALENCE,
    )
    second = _export(
        calibration_source,
        tmp_path / "second",
        JudgeTask.FINDING_EQUIVALENCE,
    )

    assert first == second
    assert first.policy.selection_seed == 20260726
    assert tuple(record.selection_order for record in first.selection_records) == tuple(
        range(1, len(first.selection_records) + 1)
    )
    assert all(
        record.selection_seed == first.policy.selection_seed
        and record.selection_reasons
        and record.source_digest
        for record in first.selection_records
    )


def test_human_label_requires_package_and_item_digest(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    package = _export(
        calibration_source,
        tmp_path,
        JudgeTask.FINDING_EQUIVALENCE,
    )
    item = package.items[0]
    label = _human_label(package, item, "different")
    assert label.package_digest == package.digest()
    assert label.item_digest == item.digest()
    assert label.source_digest == item.source_digest

    with pytest.raises(ValueError, match="package|digest|binding"):
        HumanLabelSetV1.create(
            package=package,
            labels=(replace(label, package_digest="0" * 64),),
        )
    payload = label.to_dict()
    del payload["item_digest"]
    with pytest.raises(ValueError, match="missing|fields|keys"):
        HumanLabelV1.from_dict(payload)


def test_disputed_label_requires_adjudication_before_gate_eligibility(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    package = _export(
        calibration_source,
        tmp_path,
        JudgeTask.FINDING_EQUIVALENCE,
    )
    matching = _matching_labels(package)
    disputed = replace(
        matching.labels[0],
        disputed=True,
        adjudication_ref=None,
    )
    unresolved = HumanLabelSetV1.create(
        package=package,
        labels=(disputed, *matching.labels[1:]),
    )

    unresolved_result = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=unresolved,
    )
    assert unresolved_result.status is not CalibrationStatus.GATE_ELIGIBLE
    assert unresolved_result.profiles[0].unadjudicated_dispute_count == 1

    adjudicated = HumanLabelSetV1.create(
        package=package,
        labels=(
            replace(disputed, adjudication_ref="adjudication-record-1"),
            *matching.labels[1:],
        ),
    )
    adjudicated_result = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=adjudicated,
    )
    assert adjudicated_result.status is CalibrationStatus.GATE_ELIGIBLE


def test_calibration_profiles_are_scored_independently(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    results = {}
    for profile in JudgeTask:
        package = _export(
            calibration_source,
            tmp_path / profile.value,
            profile,
        )
        empty = HumanLabelSetV1.create(package=package, labels=())
        results[profile] = score_calibration(
            calibration_source["verified_by_profile"][profile],
            package=package,
            labels=empty,
        )

    assert {
        result.profiles[0].profile for result in results.values()
    } == set(JudgeTask)
    assert all(len(result.profiles) == 1 for result in results.values())
    assert len(
        {
            result.profiles[0].allowed_labels
            for result in results.values()
        }
    ) == len(JudgeTask)
    assert all(
        result.status is CalibrationStatus.PENDING_HUMAN_LABELS
        for result in results.values()
    )


def test_missing_labels_are_pending_not_fake_agreement(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    package = _export(
        calibration_source,
        tmp_path,
        JudgeTask.FINDING_EQUIVALENCE,
    )
    result = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=HumanLabelSetV1.create(package=package, labels=()),
    )
    profile = result.profiles[0]

    assert result.status is CalibrationStatus.PENDING_HUMAN_LABELS
    assert profile.pending_label_count == len(package.items)
    assert profile.exact_agreement_denominator == 0
    assert profile.exact_agreement_ppm is None
    assert profile.cohen_kappa_ppm is None
    assert profile.cohen_kappa_null_reason is not None


def test_kappa_and_confusion_matrix_are_reproducible(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    package = _export(
        calibration_source,
        tmp_path,
        JudgeTask.FINDING_EQUIVALENCE,
    )
    labels = _matching_labels(package)

    first = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=labels,
    )
    second = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=labels,
    )
    profile = first.profiles[0]

    assert canonical_json_bytes(first.to_dict()) == canonical_json_bytes(
        second.to_dict()
    )
    assert profile.exact_agreement_ppm == 1_000_000
    assert profile.cohen_kappa_ppm == 1_000_000
    assert sum(cell.count for cell in profile.confusion_matrix) == len(
        package.items
    )
    assert any(
        cell.human_label == "different"
        and cell.recorded_label == "different"
        and cell.count == 1
        for cell in profile.confusion_matrix
    )
    assert any(
        cell.human_label == "equivalent"
        and cell.recorded_label == "equivalent"
        and cell.count == 1
        for cell in profile.confusion_matrix
    )


def test_export_rejects_repository_paths_and_tampered_resume(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    with pytest.raises((ValueError, ArtifactSecurityError), match="repository|Analysis|Run Store"):
        export_calibration_package(
            calibration_source["verified_by_profile"][
                JudgeTask.INTENT_EQUIVALENCE
            ],
            profile=JudgeTask.INTENT_EQUIVALENCE,
            policy=POLICY,
            output_root=Path(__file__).resolve().parent / "forbidden-output",
        )

    package = _export(
        calibration_source,
        tmp_path,
        JudgeTask.INTENT_EQUIVALENCE,
    )
    payload_path = (
        tmp_path
        / "external-calibration"
        / package.package_id
        / "calibration_package.json"
    )
    payload_path.write_bytes(payload_path.read_bytes() + b" ")
    with pytest.raises((ArtifactConflictError, ArtifactIntegrityError)):
        _export(
            calibration_source,
            tmp_path,
            JudgeTask.INTENT_EQUIVALENCE,
        )


def test_recomputed_item_id_cannot_bypass_payload_binding(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    package = _export(
        calibration_source,
        tmp_path,
        JudgeTask.INTENT_EQUIVALENCE,
    )
    item = package.items[0]
    with pytest.raises(ValueError, match="canonical|calibration_item_id"):
        replace(
            item,
            calibration_item_id=stable_id(
                "calibration-item-v1",
                {"attacker": item.blinded_request_payload},
            ),
        )

    package_payload = package.to_dict()
    record = package_payload["selection_records"][0]
    record["recorded_outcome"] = "attacker_oov"
    record["recorded_primary_label"] = "attacker_oov"
    package_payload["selection_digest"] = canonical_sha256(
        package_payload["selection_records"]
    )
    package_payload["package_id"] = stable_id(
        "calibration-package-v1",
        package.profile.value,
        package.policy.digest(),
        package.source_digest,
        package.payload_digest,
        package_payload["selection_digest"],
    )
    with pytest.raises(ValueError, match="outcome|label|vocabulary"):
        CalibrationPackageV1.from_dict(package_payload)


def test_recomputed_result_ids_cannot_bypass_status_or_disagreement_metrics(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    package = _export(
        calibration_source,
        tmp_path,
        JudgeTask.FINDING_EQUIVALENCE,
    )
    pending = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=HumanLabelSetV1.create(package=package, labels=()),
    ).to_dict()
    pending_profile = pending["profiles"][0]
    pending_profile["status"] = "gate_eligible"
    profile_identity = dict(pending_profile)
    del profile_identity["profile_calibration_id"]
    pending_profile["profile_calibration_id"] = stable_id(
        "profile-calibration-v1",
        profile_identity,
    )
    pending["status"] = "gate_eligible"
    result_identity = dict(pending)
    del result_identity["calibration_result_id"]
    pending["calibration_result_id"] = stable_id(
        "calibration-result-v1",
        result_identity,
    )
    with pytest.raises(ValueError, match="status|coverage|threshold"):
        CalibrationResultV1.from_dict(pending)

    matching = _matching_labels(package)
    first_item = package.items[0]
    first_label = matching.labels[0]
    replacement = next(
        value for value in first_item.allowed_labels if value != first_label.label
    )
    disagreeing = HumanLabelSetV1.create(
        package=package,
        labels=(replace(first_label, label=replacement), *matching.labels[1:]),
    )
    disagreement = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=disagreeing,
    ).to_dict()
    disagreement_profile = disagreement["profiles"][0]
    assert disagreement_profile["disagreement_count"] == 1
    disagreement_profile["disagreement_item_refs"] = []
    profile_identity = dict(disagreement_profile)
    del profile_identity["profile_calibration_id"]
    disagreement_profile["profile_calibration_id"] = stable_id(
        "profile-calibration-v1",
        profile_identity,
    )
    result_identity = dict(disagreement)
    del result_identity["calibration_result_id"]
    disagreement["calibration_result_id"] = stable_id(
        "calibration-result-v1",
        result_identity,
    )
    with pytest.raises(ValueError, match="disagreement"):
        CalibrationResultV1.from_dict(disagreement)


def test_analysis_manifest_omits_raw_payload_and_verified_load_replays_sources(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    package = _export(
        calibration_source,
        tmp_path / "export",
        JudgeTask.FINDING_EQUIVALENCE,
    )
    labels = _matching_labels(package)
    result = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=labels,
    )
    store = AnalysisArtifactStore(tmp_path / ".eval-analyses")

    package_receipt = store.publish_calibration_package(
        package,
        evaluation=calibration_source["verified_by_profile"][package.profile],
        policy=POLICY,
    )
    manifest = store.load_calibration_package_manifest(
        package_receipt.artifact_id
    )
    manifest_bytes = canonical_json_bytes(manifest.to_dict())
    for item in package.items:
        assert item.blinded_request_payload_json.encode("utf-8") not in manifest_bytes
    assert b'"items"' not in manifest_bytes
    assert manifest.payload_digest == package.payload_digest
    assert store.load_verified_calibration_package_manifest(
        package_receipt.artifact_id,
        evaluation=calibration_source["verified_by_profile"][package.profile],
        policy=POLICY,
        package=package,
    ) == manifest

    label_receipt = store.publish_human_label_set(
        labels,
        evaluation=calibration_source["verified_by_profile"][package.profile],
        policy=POLICY,
        package=package,
    )
    assert store.load_human_label_set(
        label_receipt.artifact_id,
        package=package,
    ) == labels

    result_receipt = store.publish_calibration_result(
        result,
        evaluation=calibration_source["verified_by_profile"][package.profile],
        policy=POLICY,
        package=package,
        labels=labels,
    )
    assert store.load_verified_calibration_result(
        result_receipt.artifact_id,
        evaluation=calibration_source["verified_by_profile"][package.profile],
        policy=POLICY,
        package=package,
        labels=labels,
    ) == result


def test_score_and_export_do_not_call_judge(
    calibration_source: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls_before = sum(judge.calls for judge in calibration_source["judges"])

    def forbidden(*args: Any, **kwargs: Any):
        del args, kwargs
        raise AssertionError("calibration must not execute a Judge")

    monkeypatch.setattr(SemanticJudge, "execute", forbidden)
    package = _export(
        calibration_source,
        tmp_path,
        JudgeTask.FINDING_EQUIVALENCE,
    )
    score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=_matching_labels(package),
    )
    assert sum(judge.calls for judge in calibration_source["judges"]) == calls_before


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse behavior is platform-specific")
def test_export_rejects_symlink_or_junction_root(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    with pytest.raises(ArtifactSecurityError, match="link|reparse|unsafe"):
        export_calibration_package(
            calibration_source["verified_by_profile"][JudgeTask.INTENT_EQUIVALENCE],
            profile=JudgeTask.INTENT_EQUIVALENCE,
            policy=POLICY,
            output_root=linked,
        )

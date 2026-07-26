from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from review_agent.model_protocol import ModelResponseKind, ModelTurnResponse
import review_agent_eval.calibration as calibration_module
from review_agent_eval.analysis_artifacts import AnalysisArtifactStore
from review_agent_eval.artifacts import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactSecurityError,
)
from review_agent_eval.calibration import (
    AdjudicationV1,
    CALIBRATION_ALGORITHM_VERSION,
    CALIBRATION_SELECTION_POLICY_SCHEMA_VERSION,
    CalibrationPackageV1,
    CalibrationSelectionRecordV1,
    AuxiliaryCalibrationApplicability,
    CalibrationAuxiliaryDimension,
    CalibrationResultV1,
    CalibrationSelectionPolicyV1,
    CalibrationStatus,
    HumanAdjudicationV1,
    HumanLabelSetV1,
    HumanLabelV1,
    HumanReviewerProvenanceV1,
    ReviewerProvenanceKind,
    export_calibration_package,
    score_calibration,
)
from review_agent_eval.comparison import VerifiedRunEvaluation
from review_agent_eval.config import JudgeKind
from review_agent_eval.intent_evaluator import IntentMatchKind, IntentTruth
from review_agent_eval.judge import (
    ActionabilityAssessment,
    BlindJudgeInput,
    JudgeTask,
    SeverityAssessment,
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


TRUSTED_REVIEWER = HumanReviewerProvenanceV1.create(
    kind=ReviewerProvenanceKind.HUMAN,
    reviewer_id="independent-reviewer-1",
    provenance_ref="external-review-record-1",
    attestation_ref="human-attestation-record-1",
)
TRUSTED_ADJUDICATOR = HumanReviewerProvenanceV1.create(
    kind=ReviewerProvenanceKind.HUMAN,
    reviewer_id="independent-adjudicator-1",
    provenance_ref="external-adjudicator-record-1",
    attestation_ref="human-adjudicator-attestation-1",
)


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
    minimum_auxiliary_human_coverage_ppm=1_000_000,
    minimum_auxiliary_labels_per_class=0,
    minimum_auxiliary_exact_agreement_ppm=900_000,
    minimum_auxiliary_cohen_kappa_ppm=None,
    trusted_reviewer_provenance_digests=(TRUSTED_REVIEWER.digest(),),
    trusted_adjudicator_provenance_digests=(TRUSTED_ADJUDICATOR.digest(),),
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
                },
                {
                    "truth_id": "intent-deterministic-exact",
                    "dimension": "goal",
                    "text": "Preserve access control",
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
    reviewer_kind: ReviewerProvenanceKind = ReviewerProvenanceKind.HUMAN,
    reviewer_provenance: HumanReviewerProvenanceV1 | None = None,
    disputed: bool = False,
    adjudication: HumanAdjudicationV1 | None = None,
) -> HumanLabelV1:
    reviewer = reviewer_provenance or (
        TRUSTED_REVIEWER
        if reviewer_kind is ReviewerProvenanceKind.HUMAN
        else HumanReviewerProvenanceV1.create(
            kind=reviewer_kind,
            reviewer_id="independent-reviewer-1",
            provenance_ref="external-review-record-1",
            attestation_ref=None,
        )
    )
    return HumanLabelV1.create(
        package=package,
        item=item,
        label=label,
        severity_assessment=(
            SeverityAssessment.CONSISTENT
            if package.profile
            in {JudgeTask.FINDING_EQUIVALENCE, JudgeTask.NOVEL_FACTUALITY}
            else None
        ),
        actionability=(
            ActionabilityAssessment.ACTIONABLE
            if package.profile
            in {JudgeTask.FINDING_EQUIVALENCE, JudgeTask.NOVEL_FACTUALITY}
            else None
        ),
        reviewer_provenance=reviewer,
        blind_attestation=True,
        labeled_at="2026-07-26T10:00:00Z",
        disputed=disputed,
        adjudication=adjudication,
    )


def _matching_labels(
    package: Any,
    *,
    reviewer_kind: ReviewerProvenanceKind = ReviewerProvenanceKind.HUMAN,
) -> HumanLabelSetV1:
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
            _human_label(
                package,
                item,
                label,
                reviewer_kind=reviewer_kind,
            )
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


_FORBIDDEN_SELECTION_METADATA_KEYS = {
    "recorded_outcome",
    "recorded_primary_label",
    "recorded_severity_assessment",
    "recorded_actionability",
    "reason_digest",
    "seed_rank",
    "selection_category",
    "selection_rank",
    "selection_reasons",
    "selection_stratum_digest",
    "stratum",
    "stratum_digest",
}


def _assert_no_enumerable_selection_metadata(value: Any) -> None:
    assert _FORBIDDEN_SELECTION_METADATA_KEYS.isdisjoint(set(_walk_keys(value)))
    strings = tuple(item.casefold() for item in _walk_strings(value))
    assert "mandatory" not in strings
    assert "__judge_failed__" not in strings
    assert "__ungraded__" not in strings
    assert not any(item.startswith("normal:") for item in strings)


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
        "recorded_outcome",
        "recorded_primary_label",
        "recorded_severity_assessment",
        "recorded_actionability",
        "selection_reasons",
        "selection_category",
        "selection_rank",
        "selection_stratum_digest",
        "seed_rank",
        "stratum",
    }
    assert forbidden_keys.isdisjoint(set(_walk_keys(exported)))
    assert forbidden_keys.isdisjoint(set(_walk_keys(package.to_dict())))
    verified = calibration_source["verified_by_profile"][
        JudgeTask.FINDING_EQUIVALENCE
    ]
    agent = verified.run_config.agent
    judge_profile = verified.bundle.evaluator_execution.evaluator.profile(
        JudgeKind.FINDING_EQUIVALENCE
    )
    forbidden_values = {
        verified.run_id,
        verified.evaluation_id,
        verified.run_config.run_instance_key,
        agent.agent_id,
        agent.agent_name,
        agent.agent_version,
        agent.model,
        agent.provider,
        agent.prompt_config_digest,
        verified.run_config.agent_config_digest,
        judge_profile.judge_id,
        judge_profile.provider,
        judge_profile.model,
        judge_profile.adapter_config_digest,
        judge_profile.system_prompt_digest,
    }
    exported_strings = tuple(_walk_strings(exported))
    assert all(
        value.casefold() != forbidden.casefold()
        and (
            len(forbidden) < 4
            or forbidden.casefold() not in value.casefold()
        )
        for forbidden in forbidden_values
        for value in exported_strings
    )
    assert exported["payload_digest"] == package.payload_digest
    assert exported["items"]
    _assert_no_enumerable_selection_metadata(exported)
    _assert_no_enumerable_selection_metadata(package.to_dict())


@pytest.mark.parametrize(
    "injection",
    (
        "identity_key",
        "identity_value",
        "identity_key_unicode",
        "identity_value_unicode",
        "judge_result_key",
    ),
)
def test_export_rejects_deep_identity_and_judge_result_injection(
    calibration_source: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injection: str,
) -> None:
    original = calibration_module._blind_payload

    def injected(request: BlindJudgeInput):
        payload, dimension = original(request)
        if injection == "identity_key":
            payload["items"][0]["metadata"]["scripted-model"] = "nested"
        elif injection == "identity_value":
            payload["items"][0]["text"] += " nested scripted-model identity"
        elif injection == "identity_key_unicode":
            payload["items"][0]["metadata"]["scripted\uff0emodel"] = "nested"
        elif injection == "identity_value_unicode":
            payload["items"][0]["text"] += " nested scripted\uff0fmodel identity"
        else:
            payload["context_blocks"].append(
                {
                    "metadata": {
                        "nested": {
                            "judge_result": "equivalent",
                            "failure": None,
                        }
                    }
                }
            )
        return payload, dimension

    monkeypatch.setattr(calibration_module, "_blind_payload", injected)
    output_root = tmp_path / "external-calibration"
    with pytest.raises(ArtifactSecurityError, match="forbidden|identity|blind"):
        export_calibration_package(
            calibration_source["verified_by_profile"][
                JudgeTask.FINDING_EQUIVALENCE
            ],
            profile=JudgeTask.FINDING_EQUIVALENCE,
            policy=POLICY,
            output_root=output_root,
        )
    assert not output_root.exists()


@pytest.mark.parametrize(
    "forbidden_key",
    (
        "judge.result",
        "judge__result",
        "judge-result",
        "Judge Result",
        "judge\u2014result",
    ),
)
def test_export_rejects_collapsed_forbidden_key_markers(
    calibration_source: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forbidden_key: str,
) -> None:
    original = calibration_module._blind_payload

    def injected(request: BlindJudgeInput):
        payload, dimension = original(request)
        payload["items"][0]["metadata"][forbidden_key] = "nested"
        return payload, dimension

    monkeypatch.setattr(calibration_module, "_blind_payload", injected)
    output_root = tmp_path / "external-calibration"
    with pytest.raises(ArtifactSecurityError, match="forbidden|blind"):
        export_calibration_package(
            calibration_source["verified_by_profile"][
                JudgeTask.FINDING_EQUIVALENCE
            ],
            profile=JudgeTask.FINDING_EQUIVALENCE,
            policy=POLICY,
            output_root=output_root,
        )
    assert not output_root.exists()


@pytest.mark.parametrize(
    "forbidden_value",
    (
        "prefix::judge.result::suffix",
        "prefix[[judge__decision]]suffix",
        "prefix\u3010judge\u2014failure\u3011suffix",
        "encoded|expected/winner|marker",
        "role=baseline\uff0fcandidate",
    ),
)
def test_export_rejects_collapsed_forbidden_value_markers(
    calibration_source: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forbidden_value: str,
) -> None:
    original = calibration_module._blind_payload

    def injected(request: BlindJudgeInput):
        payload, dimension = original(request)
        payload["context_blocks"].append(
            {"metadata": {"nested": {"note": forbidden_value}}}
        )
        return payload, dimension

    monkeypatch.setattr(calibration_module, "_blind_payload", injected)
    output_root = tmp_path / "external-calibration"
    with pytest.raises(ArtifactSecurityError, match="forbidden|blind"):
        export_calibration_package(
            calibration_source["verified_by_profile"][
                JudgeTask.FINDING_EQUIVALENCE
            ],
            profile=JudgeTask.FINDING_EQUIVALENCE,
            policy=POLICY,
            output_root=output_root,
        )
    assert not output_root.exists()


@pytest.mark.parametrize(
    "forbidden_value",
    (
        "prefix decision: equivalent suffix",
        "prefix FAILURE: timeout suffix",
        "prefix\u3010DeCiSiOn\u3011\uff1aequivalent suffix",
        "prefix\uff1c\uff26\uff21\uff29\uff2c\uff35\uff32\uff25\uff0ftimeout\uff1esuffix",
    ),
)
def test_export_rejects_bare_decision_and_failure_substrings(
    calibration_source: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forbidden_value: str,
) -> None:
    original = calibration_module._blind_payload

    def injected(request: BlindJudgeInput):
        payload, dimension = original(request)
        payload["context_blocks"].append(
            {"metadata": {"nested": {"note": forbidden_value}}}
        )
        return payload, dimension

    monkeypatch.setattr(calibration_module, "_blind_payload", injected)
    output_root = tmp_path / "external-calibration"
    with pytest.raises(ArtifactSecurityError, match="forbidden|blind"):
        export_calibration_package(
            calibration_source["verified_by_profile"][
                JudgeTask.FINDING_EQUIVALENCE
            ],
            profile=JudgeTask.FINDING_EQUIVALENCE,
            policy=POLICY,
            output_root=output_root,
        )
    assert not output_root.exists()


def test_export_allows_unstructured_marker_words_in_source_context(
    calibration_source: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = calibration_module._blind_payload

    def injected(request: BlindJudgeInput):
        payload, dimension = original(request)
        payload["context_blocks"].append(
            {
                "metadata": {
                    "note": (
                        "The failure handling follows a decision tree while the "
                        "candidate collection retains a baseline snapshot."
                    )
                }
            }
        )
        return payload, dimension

    monkeypatch.setattr(calibration_module, "_blind_payload", injected)
    package = export_calibration_package(
        calibration_source["verified_by_profile"][JudgeTask.FINDING_EQUIVALENCE],
        profile=JudgeTask.FINDING_EQUIVALENCE,
        policy=POLICY,
        output_root=tmp_path / "external-calibration",
    )
    assert package.items


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
    expected_record_fields = {
        "schema_version",
        "selection_record_id",
        "calibration_item_id",
        "item_digest",
        "source_digest",
        "selection_order",
        "selection_seed",
    }
    assert all(
        record.selection_seed == first.policy.selection_seed
        and record.source_digest
        and set(record.to_dict()) == expected_record_fields
        and not hasattr(record, "selection_category")
        and not hasattr(record, "selection_rank")
        and not hasattr(record, "selection_stratum_digest")
        for record in first.selection_records
    )
    _assert_no_enumerable_selection_metadata(first.to_dict())
    _assert_no_enumerable_selection_metadata(first.to_blind_dict())
    assert CalibrationPackageV1.from_dict(first.to_dict()) == first


def test_selection_record_schema_rejects_enumerable_metadata_reforging(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    package = _export(
        calibration_source,
        tmp_path,
        JudgeTask.FINDING_EQUIVALENCE,
    )
    injected_fields = {
        "selection_category": "mandatory",
        "selection_stratum_digest": canonical_sha256("normal:equivalent"),
        "selection_rank": canonical_sha256("judge_failed"),
        "seed_rank": canonical_sha256("ungraded"),
        "stratum": "normal:unknown",
        "selection_reasons": ["mandatory_semantic_unknown"],
        "recorded_outcome": "__judge_failed__",
    }
    for field, value in injected_fields.items():
        payload = json.loads(canonical_json(package.to_dict()))
        payload["selection_records"][0][field] = value
        with pytest.raises(ValueError, match="unknown|fields|keys"):
            CalibrationPackageV1.from_dict(payload)


def test_real_deterministic_judge_conflict_is_mandatory(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    verified = calibration_source["verified_by_profile"][
        JudgeTask.INTENT_EQUIVALENCE
    ]
    trial = verified.trials[0]
    judge_result = next(
        item
        for item in trial.judge_output.results
        if item.request.task is JudgeTask.INTENT_EQUIVALENCE
    )
    semantic = next(
        item
        for item in trial.intent_result.candidates
        if item.request_id == judge_result.request.source_request_id
    )
    assert semantic.match_kind is IntentMatchKind.SEMANTIC
    assert semantic.selected is False
    assert any(
        item.selected
        and item.match_kind is not IntentMatchKind.SEMANTIC
        and item.generated_id == semantic.generated_id
        for item in trial.intent_result.candidates
    )
    assert calibration_module._deterministic_judge_conflict(trial, judge_result)

    package = _export(
        calibration_source,
        tmp_path,
        JudgeTask.INTENT_EQUIVALENCE,
    )
    assert len(package.selection_records) == 1
    _assert_no_enumerable_selection_metadata(package.to_dict())


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
        adjudication=None,
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

    untrusted_adjudicator = HumanReviewerProvenanceV1.create(
        kind=ReviewerProvenanceKind.HUMAN,
        reviewer_id="unregistered-adjudicator",
        provenance_ref="unregistered-adjudication-record",
        attestation_ref="unregistered-adjudication-attestation",
    )
    untrusted_adjudication = AdjudicationV1.create(
        package=package,
        item=package.items[0],
        adjudication_ref="untrusted-adjudication",
        original_label_refs=(disputed.original_label_digest,),
        final_primary_label=disputed.label,
        final_severity_assessment=disputed.severity_assessment,
        final_actionability=disputed.actionability,
        adjudicator_provenance=untrusted_adjudicator,
        blind_attestation=True,
    )
    untrusted_labels = HumanLabelSetV1.create(
        package=package,
        labels=(
            replace(disputed, adjudication=untrusted_adjudication),
            *matching.labels[1:],
        ),
    )
    untrusted_result = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=untrusted_labels,
    )
    assert untrusted_result.status is not CalibrationStatus.GATE_ELIGIBLE

    adjudication = AdjudicationV1.create(
        package=package,
        item=package.items[0],
        adjudication_ref="adjudication-record-1",
        original_label_refs=(disputed.original_label_digest,),
        final_primary_label=disputed.label,
        final_severity_assessment=disputed.severity_assessment,
        final_actionability=disputed.actionability,
        adjudicator_provenance=TRUSTED_ADJUDICATOR,
        blind_attestation=True,
    )
    adjudicated = HumanLabelSetV1.create(
        package=package,
        labels=(
            replace(disputed, adjudication=adjudication),
            *matching.labels[1:],
        ),
    )
    adjudicated_result = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=adjudicated,
    )
    assert adjudicated_result.status is CalibrationStatus.GATE_ELIGIBLE

    same_reviewer = AdjudicationV1.create(
        package=package,
        item=package.items[0],
        adjudication_ref="same-reviewer-adjudication",
        original_label_refs=(disputed.original_label_digest,),
        final_primary_label=disputed.label,
        final_severity_assessment=disputed.severity_assessment,
        final_actionability=disputed.actionability,
        adjudicator_provenance=disputed.reviewer_provenance,
        blind_attestation=True,
    )
    with pytest.raises(ValueError, match="independent|different|reviewer"):
        replace(disputed, adjudication=same_reviewer)


def test_unregistered_human_provenance_never_counts_as_eligible(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    package = _export(
        calibration_source,
        tmp_path,
        JudgeTask.FINDING_EQUIVALENCE,
    )
    unregistered = HumanReviewerProvenanceV1.create(
        kind=ReviewerProvenanceKind.HUMAN,
        reviewer_id="self-asserted-human",
        provenance_ref="self-asserted-record",
        attestation_ref="self-asserted-attestation",
    )
    labels = HumanLabelSetV1.create(
        package=package,
        labels=tuple(
            _human_label(
                package,
                item,
                (
                    "different"
                    if "incorrect review behavior"
                    in item.blinded_request_payload["items"][0]["text"]
                    else "equivalent"
                ),
                reviewer_provenance=unregistered,
            )
            for item in package.items
        ),
    )
    result = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=labels,
    )
    assert result.status is CalibrationStatus.PENDING_HUMAN_LABELS
    assert result.profiles[0].eligible_labeled_count == 0

    empty_trust_policy = replace(
        POLICY,
        trusted_reviewer_provenance_digests=(),
        trusted_adjudicator_provenance_digests=(),
    )
    empty_trust_package = _export(
        calibration_source,
        tmp_path / "empty-trust",
        JudgeTask.FINDING_EQUIVALENCE,
        policy=empty_trust_policy,
    )
    empty_trust_result = score_calibration(
        calibration_source["verified_by_profile"][empty_trust_package.profile],
        package=empty_trust_package,
        labels=_matching_labels(empty_trust_package),
    )
    assert empty_trust_result.status is CalibrationStatus.PENDING_HUMAN_LABELS
    assert empty_trust_result.profiles[0].eligible_labeled_count == 0


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


def test_authoritative_auxiliary_labels_are_scored_and_gate_independently(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    package = _export(
        calibration_source,
        tmp_path,
        JudgeTask.FINDING_EQUIVALENCE,
    )
    matching = _matching_labels(package)
    matching_result = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=matching,
    )
    auxiliary = {
        item.dimension: item
        for item in matching_result.profiles[0].auxiliary_calibrations
    }
    assert matching_result.status is CalibrationStatus.GATE_ELIGIBLE
    assert auxiliary[
        CalibrationAuxiliaryDimension.SEVERITY_ASSESSMENT
    ].applicability is AuxiliaryCalibrationApplicability.APPLICABLE
    assert auxiliary[
        CalibrationAuxiliaryDimension.ACTIONABILITY
    ].exact_agreement_ppm == 1_000_000

    missing = HumanLabelSetV1.create(
        package=package,
        labels=(
            replace(matching.labels[0], severity_assessment=None),
            *matching.labels[1:],
        ),
    )
    missing_result = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=missing,
    )
    missing_severity = next(
        item
        for item in missing_result.profiles[0].auxiliary_calibrations
        if item.dimension is CalibrationAuxiliaryDimension.SEVERITY_ASSESSMENT
    )
    assert missing_result.status is CalibrationStatus.INSUFFICIENT_COVERAGE
    assert missing_severity.eligible_labeled_item_count < (
        missing_severity.applicable_item_count
    )
    assert missing_result.profiles[0].exact_agreement_ppm == 1_000_000

    wrong = HumanLabelSetV1.create(
        package=package,
        labels=tuple(
            replace(label, severity_assessment=SeverityAssessment.OVERSTATED)
            for label in matching.labels
        ),
    )
    wrong_result = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=wrong,
    )
    wrong_severity = next(
        item
        for item in wrong_result.profiles[0].auxiliary_calibrations
        if item.dimension is CalibrationAuxiliaryDimension.SEVERITY_ASSESSMENT
    )
    assert wrong_result.status is CalibrationStatus.FAILED_THRESHOLDS
    assert wrong_result.profiles[0].exact_agreement_ppm == 1_000_000
    assert wrong_severity.exact_agreement_ppm != 1_000_000


def test_non_authoritative_auxiliary_dimensions_are_typed_not_applicable(
    calibration_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    package = _export(
        calibration_source,
        tmp_path,
        JudgeTask.INTENT_EQUIVALENCE,
    )
    result = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=_matching_labels(package),
    )
    assert all(
        item.applicability is AuxiliaryCalibrationApplicability.NOT_APPLICABLE
        and item.applicable_item_count == 0
        and item.applicable_occurrence_count == 0
        for item in result.profiles[0].auxiliary_calibrations
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


@pytest.mark.parametrize(
    "reviewer_kind",
    (ReviewerProvenanceKind.FIXTURE, ReviewerProvenanceKind.SYNTHETIC),
)
def test_non_human_matching_labels_never_become_gate_eligible(
    calibration_source: dict[str, Any],
    tmp_path: Path,
    reviewer_kind: ReviewerProvenanceKind,
) -> None:
    package = _export(
        calibration_source,
        tmp_path,
        JudgeTask.FINDING_EQUIVALENCE,
    )
    labels = _matching_labels(package, reviewer_kind=reviewer_kind)
    result = score_calibration(
        calibration_source["verified_by_profile"][package.profile],
        package=package,
        labels=labels,
    )
    profile = result.profiles[0]

    assert result.status in {
        CalibrationStatus.PENDING_HUMAN_LABELS,
        CalibrationStatus.INSUFFICIENT_COVERAGE,
    }
    assert result.status is not CalibrationStatus.GATE_ELIGIBLE
    assert profile.labeled_count == len(package.selection_records)
    assert profile.eligible_labeled_count == 0
    assert profile.exact_agreement_denominator == 0


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


def test_duplicate_item_coverage_is_unique_but_judge_metrics_are_per_occurrence(
    calibration_source: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = calibration_source["verified_by_profile"][
        JudgeTask.INTENT_EQUIVALENCE
    ]
    package, selected = calibration_module._build_package_with_selection(
        evaluation,
        profile=JudgeTask.INTENT_EQUIVALENCE,
        policy=POLICY,
    )
    assert len(package.items) == len(selected) == 1
    first_record = package.selection_records[0]
    duplicate_source_digest = canonical_sha256("duplicate-source-occurrence")
    second_selected = replace(
        selected[0],
        source_digest=duplicate_source_digest,
        recorded_outcome="different",
        primary_label="different",
    )
    second_record = CalibrationSelectionRecordV1.create(
        calibration_item_id=first_record.calibration_item_id,
        item_digest=first_record.item_digest,
        source_digest=duplicate_source_digest,
        selection_order=2,
        selection_seed=POLICY.selection_seed,
    )
    records = (first_record, second_record)
    selection_digest = canonical_sha256([item.to_dict() for item in records])
    duplicate_package = CalibrationPackageV1(
        schema_version=package.schema_version,
        package_id=stable_id(
            "calibration-package-v1",
            package.profile.value,
            package.policy.digest(),
            package.source_digest,
            package.payload_digest,
            selection_digest,
        ),
        profile=package.profile,
        policy=package.policy,
        source_digest=package.source_digest,
        payload_digest=package.payload_digest,
        selection_digest=selection_digest,
        items=package.items,
        selection_records=records,
        status=package.status,
    )
    labels = _matching_labels(duplicate_package)

    def duplicate_replay(*args: Any, **kwargs: Any):
        del args, kwargs
        return duplicate_package, (selected[0], second_selected)

    monkeypatch.setattr(
        calibration_module,
        "_build_package_with_selection",
        duplicate_replay,
    )
    result = score_calibration(
        evaluation,
        package=duplicate_package,
        labels=labels,
    )
    profile = result.profiles[0]
    assert profile.selected_count == 1
    assert profile.labeled_count == 1
    assert profile.eligible_labeled_count == 1
    assert profile.selected_occurrence_count == 2
    assert profile.eligible_labeled_occurrence_count == 2
    assert profile.exact_agreement_denominator == 2
    assert sum(cell.count for cell in profile.confusion_matrix) == 2
    assert profile.disagreement_count == 1
    assert profile.disagreement_item_refs == (second_record.selection_record_id,)


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


def test_export_entry_limit_stops_at_third_scandir_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "streaming-export"
    storage = calibration_module._CalibrationExportStorage(root)
    directory = root / "package"
    directory.mkdir()
    for name in ("one.json", "two.json", "three.json", "must-not-read.json"):
        (directory / name).write_text("{}", encoding="utf-8")

    original_scandir = os.scandir
    observed = {"count": 0}

    class CountingScandir:
        def __init__(self, path: Any) -> None:
            self._context = original_scandir(path)
            self._iterator: Any = None

        def __enter__(self):
            self._iterator = self._context.__enter__()
            return self

        def __exit__(self, *args: Any):
            return self._context.__exit__(*args)

        def __iter__(self):
            return self

        def __next__(self):
            observed["count"] += 1
            if observed["count"] > 3:
                raise AssertionError("export enumeration read beyond its hard limit")
            return next(self._iterator)

    monkeypatch.setattr(os, "scandir", CountingScandir)
    with pytest.raises(ArtifactIntegrityError, match="unknown|artifacts"):
        calibration_module._export_entries(storage, directory)
    assert observed["count"] == 3


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
    forbidden_result_fields = {
        "judge_result_digest",
        "recorded_outcome",
        "recorded_primary_label",
        "recorded_severity_assessment",
        "recorded_actionability",
        "seed_rank",
        "selection_category",
        "selection_rank",
        "selection_reasons",
        "selection_stratum_digest",
        "stratum",
    }
    assert forbidden_result_fields.isdisjoint(set(_walk_keys(package_payload)))
    record = package_payload["selection_records"][0]
    record["source_digest"] = "0" * 64
    with pytest.raises(ValueError, match="selection_record_id|canonical"):
        CalibrationPackageV1.from_dict(package_payload)

    record_identity = dict(record)
    del record_identity["selection_record_id"]
    record["selection_record_id"] = stable_id(
        "calibration-selection-v1",
        record_identity,
    )
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
    forged = CalibrationPackageV1.from_dict(package_payload)
    _assert_no_enumerable_selection_metadata(forged.to_dict())
    with pytest.raises(ArtifactIntegrityError, match="source replay|differs"):
        score_calibration(
            calibration_source["verified_by_profile"][package.profile],
            package=forged,
            labels=HumanLabelSetV1.create(package=forged, labels=()),
        )


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
    assert {
        "judge_result_digest",
        "recorded_outcome",
        "recorded_primary_label",
        "recorded_severity_assessment",
        "recorded_actionability",
        "seed_rank",
        "selection_category",
        "selection_rank",
        "selection_reasons",
        "selection_stratum_digest",
        "stratum",
        "decision",
        "failure",
    }.isdisjoint(set(_walk_keys(manifest.to_dict())))
    _assert_no_enumerable_selection_metadata(package.to_dict())
    _assert_no_enumerable_selection_metadata(package.to_blind_dict())
    _assert_no_enumerable_selection_metadata(manifest.to_dict())
    _assert_no_enumerable_selection_metadata(package_receipt.to_dict())
    assert manifest.payload_digest == package.payload_digest
    assert store.load_verified_calibration_package_manifest(
        package_receipt.artifact_id,
        evaluation=calibration_source["verified_by_profile"][package.profile],
        policy=POLICY,
        package=package,
    ) == manifest

    forged_payload = json.loads(canonical_json(package.to_dict()))
    forged_record = forged_payload["selection_records"][0]
    forged_record["source_digest"] = "0" * 64
    forged_record_identity = dict(forged_record)
    del forged_record_identity["selection_record_id"]
    forged_record["selection_record_id"] = stable_id(
        "calibration-selection-v1",
        forged_record_identity,
    )
    forged_payload["selection_digest"] = canonical_sha256(
        forged_payload["selection_records"]
    )
    forged_payload["package_id"] = stable_id(
        "calibration-package-v1",
        package.profile.value,
        package.policy.digest(),
        package.source_digest,
        package.payload_digest,
        forged_payload["selection_digest"],
    )
    forged_package = CalibrationPackageV1.from_dict(forged_payload)
    with pytest.raises(ArtifactIntegrityError, match="replay|source-bound"):
        store.publish_calibration_package(
            forged_package,
            evaluation=calibration_source["verified_by_profile"][package.profile],
            policy=POLICY,
        )

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
    assert store.load_verified_human_label_set(
        label_receipt.artifact_id,
        evaluation=calibration_source["verified_by_profile"][package.profile],
        policy=POLICY,
        package=package,
        labels=labels,
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
    except NotImplementedError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip(f"symlink privilege is unavailable: {exc}")
        raise
    with pytest.raises(ArtifactSecurityError, match="link|reparse|unsafe"):
        export_calibration_package(
            calibration_source["verified_by_profile"][JudgeTask.INTENT_EQUIVALENCE],
            profile=JudgeTask.INTENT_EQUIVALENCE,
            policy=POLICY,
            output_root=linked,
        )

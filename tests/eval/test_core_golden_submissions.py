from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from review_agent_eval.cases import REPOSITORY_MATERIALIZER_PROTOCOL
from review_agent_eval.intent_evaluator import IntentEvaluationStatus, IntentEvaluator
from review_agent_eval.models import (
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    EvalCase,
    EvalSubmission,
    FindingSeverity,
    IntentClaimJudgement,
    IntentClaimSource,
    IntentResult,
    NovelFindingPolicy,
    RepositoryFileEvidenceSource,
    RepositoryReviewTarget,
    SubmissionStatus,
    TruthCompleteness,
    canonical_sha256,
    stable_id,
    validate_submission_for_case,
)
from tests.eval.test_core_suite import EVAL_ROOT, _banks


GOLDEN_FILENAMES = {
    "perfect": "perfect.json",
    "empty": "empty.json",
    "duplicate": "duplicate.json",
    "fabricated": "fabricated.json",
    "bad_evidence_hash": "bad-evidence.json",
    "bad_evidence_path": "bad-evidence-path.json",
    "bad_evidence_line": "bad-evidence-line.json",
    "unsupported_evidence": "unsupported-evidence.json",
    "compound": "compound.json",
    "judge_unknown": "judge-unknown.json",
    "unsupported_intent": "unsupported-intent.json",
    "contradicted_intent": "contradicted-intent.json",
}

GOLDEN_SCENARIO_ORDER = (
    "perfect",
    "empty",
    "duplicate",
    "fabricated",
    "unsupported-evidence",
    "compound",
    "judge-unknown",
    "unsupported-intent",
    "contradicted-intent",
    "bad-evidence",
    "bad-evidence-path",
    "bad-evidence-line",
)
CORE_SOURCE_VERSION = "core-2026-07-21-v3"
GOLDEN_RUN_INSTANCE_KEY = "core-golden-authoring-v2"
GOLDEN_RUN_BINDING_SCHEMA_VERSION = "core_golden_run_binding_v2"
GOLDEN_REPLAY_BINDING_SCHEMA_VERSION = "core_golden_repository_replay_v2"
GOLDEN_MATERIALIZATION_BINDING_SCHEMA_VERSION = (
    "core_golden_materialization_binding_v2"
)
GOLDEN_ATTEMPT = 1
REPOSITORY_WIRE_CONTRACT = {
    "case_schema_version": EVAL_CASE_SCHEMA_VERSION,
    "input_schema_version": EVAL_INPUT_SCHEMA_VERSION,
    "submission_schema_version": EVAL_SUBMISSION_SCHEMA_VERSION,
    "review_target_kind": "repository",
    "materializer_protocol": REPOSITORY_MATERIALIZER_PROTOCOL,
}


def _golden_paths() -> dict[str, Path]:
    root = EVAL_ROOT / "cases" / "core"
    assert root.is_dir()
    result = {}
    for scenario, filename in GOLDEN_FILENAMES.items():
        matches = sorted(root.glob("*/golden/%s" % filename))
        assert len(matches) == 1, (
            "Core requires exactly one %s Golden Submission; found %d"
            % (scenario, len(matches))
        )
        result[scenario] = matches[0]
    assert set(root.glob("*/golden/*.json")) == set(result.values())
    return result


def _case_index() -> dict[str, EvalCase]:
    return {handle.task_id: handle.load() for bank in _banks() for handle in bank}


def _repository_target(case: EvalCase) -> RepositoryReviewTarget:
    target = case.input.review_target
    assert type(target) is RepositoryReviewTarget
    return target


def _goldens() -> dict[str, tuple[EvalCase, EvalSubmission]]:
    cases = _case_index()
    result = {}
    for scenario, path in _golden_paths().items():
        raw = path.read_bytes()
        submission = EvalSubmission.from_json(raw)
        assert raw == submission.to_json().encode("utf-8"), (
            "%s must be canonical EvalSubmission v2 JSON" % path
        )
        assert submission.task_id in cases
        assert path.parent.parent.name == submission.task_id
        validate_submission_for_case(submission, cases[submission.task_id])
        result[scenario] = (cases[submission.task_id], submission)
    return result


def _scenario_name(scenario: str) -> str:
    if scenario == "bad_evidence_hash":
        return "bad-evidence"
    return scenario.replace("_", "-")


def _golden_run_id() -> str:
    return stable_id(
        "run",
        {
            "schema_version": GOLDEN_RUN_BINDING_SCHEMA_VERSION,
            "run_instance_key": GOLDEN_RUN_INSTANCE_KEY,
            "source_version": CORE_SOURCE_VERSION,
            "scenario_order": list(GOLDEN_SCENARIO_ORDER),
        },
    )


def _expected_trial_id(case: EvalCase, scenario: str) -> str:
    canonical_scenario = _scenario_name(scenario)
    return stable_id(
        "trial",
        _golden_run_id(),
        case.task_id,
        GOLDEN_SCENARIO_ORDER.index(canonical_scenario) + 1,
    )


def _expected_materialization_id(case: EvalCase, scenario: str) -> str:
    trial_id = _expected_trial_id(case, scenario)
    repository = _repository_target(case).repository
    replay_binding_digest = canonical_sha256(
        {
            "schema_version": GOLDEN_REPLAY_BINDING_SCHEMA_VERSION,
            "task_id": case.task_id,
            "repository": repository.to_dict(),
        }
    )
    return stable_id(
        "materialization",
        {
            "schema_version": GOLDEN_MATERIALIZATION_BINDING_SCHEMA_VERSION,
            "run_id": _golden_run_id(),
            "task_id": case.task_id,
            "trial_id": trial_id,
            "attempt": GOLDEN_ATTEMPT,
            "eval_input_digest": case.eval_input().digest(),
            "review_target_digest": canonical_sha256(case.input.review_target),
            "wire_contract": REPOSITORY_WIRE_CONTRACT,
            "suite_preparation_binding_digest": None,
            "replay_binding_digest": replay_binding_digest,
        },
    )


def _assert_submission_evidence_is_closed(submission: EvalSubmission) -> None:
    assert submission.review is not None
    evidence_ids = {item.evidence_id for item in submission.evidence}
    refs = {
        ref
        for finding in submission.review.findings
        for ref in finding.evidence_refs
    }
    assert evidence_ids == refs
    assert all(finding.evidence_refs for finding in submission.review.findings)


def _evidence_faults(
    case: EvalCase, submission: EvalSubmission
) -> dict[str, set[str]]:
    repository = _repository_target(case).repository
    assert repository.path is not None
    fixture = EVAL_ROOT / repository.path
    result: dict[str, set[str]] = {}
    for evidence in submission.evidence:
        faults: set[str] = set()
        source = evidence.source
        assert type(source) is RepositoryFileEvidenceSource
        if source.target_materialization_id != submission.target_materialization_id:
            faults.add("materialization")
        if source.revision == repository.base_revision:
            side = "base"
        elif source.revision == repository.head_revision:
            side = "head"
        else:
            side = "head"
            faults.add("revision")
        path_parts = Path(source.path).parts
        if ".." in path_parts or Path(source.path).is_absolute():
            candidate = None
            faults.add("path")
        else:
            candidate = fixture / side / source.path
            if not candidate.is_file():
                candidate = None
                faults.add("path")
        if source.from_line > source.to_line:
            faults.add("line_range")
        if candidate is not None:
            lines = candidate.read_text(encoding="utf-8").splitlines(keepends=True)
            if source.to_line > len(lines):
                faults.add("line_range")
            elif "line_range" not in faults:
                expected_excerpt = "".join(
                    lines[source.from_line - 1 : source.to_line]
                )
                if evidence.excerpt != expected_excerpt:
                    faults.add("excerpt")
        if hashlib.sha256(evidence.excerpt.encode("utf-8")).hexdigest() != (
            evidence.content_hash
        ):
            faults.add("content_hash")
        result[evidence.evidence_id] = faults
    return result


def test_golden_submissions_are_exact_v2_case_and_materialization_bindings() -> None:
    goldens = _goldens()
    assert set(goldens) == set(GOLDEN_FILENAMES)
    assert len(goldens) == 12
    submissions = [submission for _case, submission in goldens.values()]
    assert len({item.trial_id for item in submissions}) == 12
    assert len({item.target_materialization_id for item in submissions}) == 12
    assert len({item.digest() for item in submissions}) == 12

    for scenario, (case, submission) in goldens.items():
        assert case.source.source_version == CORE_SOURCE_VERSION
        assert submission.schema_version == EVAL_SUBMISSION_SCHEMA_VERSION
        assert submission.status is SubmissionStatus.COMPLETED
        assert submission.failure is None
        assert submission.trace_ref is None
        assert submission.eval_input_digest == case.eval_input().digest()
        assert submission.trial_id == _expected_trial_id(case, scenario)
        assert submission.target_materialization_id == (
            _expected_materialization_id(case, scenario)
        )
        assert all(
            item.source.target_materialization_id
            == submission.target_materialization_id
            for item in submission.evidence
        )
        assert submission.intent is not None and submission.review is not None
        assert submission.intent.status in {IntentResult.SUFFICIENT, IntentResult.PARTIAL}
        _assert_submission_evidence_is_closed(submission)

        raw = json.loads(_golden_paths()[scenario].read_text(encoding="utf-8"))
        assert set(raw) == {
            "schema_version",
            "task_id",
            "agent_id",
            "trial_id",
            "eval_input_digest",
            "target_materialization_id",
            "status",
            "intent",
            "review",
            "evidence",
            "usage",
            "trace_ref",
            "failure",
        }
        for evidence in raw["evidence"]:
            assert set(evidence) == {
                "evidence_id",
                "source",
                "content_hash",
                "excerpt",
            }
            assert set(evidence["source"]) == {
                "kind",
                "target_materialization_id",
                "revision",
                "path",
                "from_line",
                "to_line",
            }

    assert {
        claim.source
        for submission in submissions
        if submission.intent is not None
        for claim in submission.intent.claims
    } == set(IntentClaimSource)
    assert {
        submission.intent.status
        for submission in submissions
        if submission.intent is not None
    } == {IntentResult.SUFFICIENT, IntentResult.PARTIAL}


def test_perfect_golden_matches_truth_and_replays_exact_fixture_bytes() -> None:
    case, submission = _goldens()["perfect"]
    assert submission.intent is not None and submission.review is not None
    assert Counter(claim.text for claim in submission.intent.claims) == Counter(
        claim.text for claim in case.intent_truth.expected_claims
    )
    assert Counter(finding.claim for finding in submission.review.findings) == Counter(
        finding.claim for finding in case.review_truth.expected_findings
    )
    repository = _repository_target(case).repository
    assert {
        item.source.revision for item in submission.evidence
    } == {repository.base_revision, repository.head_revision}
    assert all(
        item.source.path == "src/timeout.py"
        and item.source.from_line == 1
        and item.source.to_line == 4
        and "return value" in item.excerpt
        for item in submission.evidence
    )
    assert all(not faults for faults in _evidence_faults(case, submission).values())

    intent_result = IntentEvaluator().evaluate(
        submission.intent, case.intent_truth, case.clarification_script
    )
    assert intent_result.status is IntentEvaluationStatus.GRADED
    assert intent_result.judge_requests == ()
    assert intent_result.metrics.intent_case_pass is True

    anchor_locations = {
        (location.path, location.from_line, location.to_line)
        for finding in case.review_truth.expected_findings
        for anchor in finding.evidence_anchors
        for location in anchor.locations
    }
    evidence_locations = {
        (item.source.path, item.source.from_line, item.source.to_line)
        for item in submission.evidence
    }
    assert any(
        not anchor.locations
        for finding in case.review_truth.expected_findings
        for anchor in finding.evidence_anchors
    ) or bool(evidence_locations - anchor_locations)


def test_unsupported_and_contradicted_intent_goldens_are_distinct() -> None:
    goldens = _goldens()
    unsupported_case, unsupported = goldens["unsupported_intent"]
    assert unsupported.intent is not None
    unsupported_result = IntentEvaluator().evaluate(
        unsupported.intent,
        unsupported_case.intent_truth,
        unsupported_case.clarification_script,
    )
    assert unsupported_result.metrics.unsupported_claim_count == 1
    assert unsupported_result.metrics.contradicted_claim_count == 0
    assert any(
        item.judgement is IntentClaimJudgement.UNSUPPORTED
        for item in unsupported_result.claim_outcomes
    )

    contradicted_case, contradicted = goldens["contradicted_intent"]
    assert contradicted.intent is not None
    contradicted_result = IntentEvaluator().evaluate(
        contradicted.intent,
        contradicted_case.intent_truth,
        contradicted_case.clarification_script,
    )
    assert contradicted_result.metrics.contradicted_claim_count == 1
    assert contradicted_result.metrics.unsupported_claim_count == 0
    assert any(
        item.judgement is IntentClaimJudgement.CONTRADICTED
        for item in contradicted_result.claim_outcomes
    )
    assert contradicted.review is not None
    assert contradicted.review.findings[0].claim == (
        contradicted_case.review_truth.expected_findings[0].claim
    )
    assert all(
        not faults
        for faults in _evidence_faults(contradicted_case, contradicted).values()
    )


def test_valid_but_irrelevant_evidence_remains_a_support_trap() -> None:
    case, submission = _goldens()["unsupported_evidence"]
    assert submission.review is not None and len(submission.review.findings) == 1
    assert all(not faults for faults in _evidence_faults(case, submission).values())
    evidence = submission.evidence[0]
    assert evidence.source.from_line == evidence.source.to_line == 1
    truth_lines = {
        line
        for truth in case.review_truth.expected_findings
        for location in truth.locations
        if location.path == evidence.source.path
        and location.from_line is not None
        and location.to_line is not None
        for line in range(location.from_line, location.to_line + 1)
    }
    assert evidence.source.from_line not in truth_lines


def test_empty_duplicate_fabricated_and_compound_goldens_keep_their_semantics() -> None:
    goldens = _goldens()

    empty_case, empty = goldens["empty"]
    assert empty.review is not None
    assert empty.review.findings == () and empty.evidence == ()
    assert any(
        item.required and item.severity is FindingSeverity.CRITICAL
        for item in empty_case.review_truth.expected_findings
    )

    duplicate_case, duplicate = goldens["duplicate"]
    assert duplicate.review is not None
    duplicate_counts = Counter(item.claim for item in duplicate.review.findings)
    assert max(duplicate_counts.values(), default=0) == 2
    assert len({item.evidence_refs for item in duplicate.review.findings}) == 1
    assert all(
        not faults for faults in _evidence_faults(duplicate_case, duplicate).values()
    )

    fabricated_case, fabricated = goldens["fabricated"]
    assert fabricated.review is not None and len(fabricated.review.findings) == 1
    assert fabricated.review.findings[0].claim in {
        item.claim for item in fabricated_case.review_truth.known_invalid_findings
    }
    assert all(
        not faults
        for faults in _evidence_faults(fabricated_case, fabricated).values()
    )

    compound_case, compound = goldens["compound"]
    assert compound.review is not None and len(compound.review.findings) == 1
    compound_text = " ".join(compound.review.findings[0].claim.casefold().split())
    assert sum(
        " ".join(item.claim.casefold().split()) in compound_text
        for item in compound_case.review_truth.expected_findings
    ) == 2
    assert all(
        not faults for faults in _evidence_faults(compound_case, compound).values()
    )


@pytest.mark.parametrize(
    ("scenario", "expected_fault"),
    (
        ("bad_evidence_hash", "content_hash"),
        ("bad_evidence_path", "path"),
        ("bad_evidence_line", "line_range"),
    ),
)
def test_each_bad_evidence_golden_breaks_only_one_integrity_dimension(
    scenario: str, expected_fault: str
) -> None:
    case, submission = _goldens()[scenario]
    assert submission.review is not None and len(submission.review.findings) == 1
    assert submission.review.findings[0].claim == (
        case.review_truth.expected_findings[0].claim
    )
    faults = _evidence_faults(case, submission)
    assert Counter(fault for item in faults.values() for fault in item) == Counter(
        {expected_fault: 1}
    )
    assert sum(bool(item) for item in faults.values()) == 1


def test_judge_unknown_golden_remains_a_fail_closed_novel_candidate() -> None:
    case, submission = _goldens()["judge_unknown"]
    assert case.review_truth.novel_finding_policy is NovelFindingPolicy.VERIFY
    assert case.review_truth.completeness is TruthCompleteness.CLOSED_WORLD
    assert submission.review is not None and len(submission.review.findings) == 1
    claim = submission.review.findings[0].claim
    assert "repository-external consumers" in claim
    assert claim not in {
        item.claim
        for item in (
            *case.review_truth.expected_findings,
            *case.review_truth.known_invalid_findings,
        )
    }
    assert all(not faults for faults in _evidence_faults(case, submission).values())

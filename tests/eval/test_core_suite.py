from __future__ import annotations

import ast
import difflib
from functools import lru_cache
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import sys
import unicodedata

from review_agent_eval.cases import (
    REPOSITORY_MATERIALIZER_PROTOCOL,
    CaseSplit,
    SuiteCase,
    SuiteKind,
    SuiteManifest,
)
from review_agent_eval.datasets import CaseBank
from review_agent_eval.models import (
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    CaseOrigin,
    ClarificationAction,
    ClarificationPolicy,
    DiffSide,
    EvalCase,
    EvalSubmission,
    FindingSeverity,
    IntentAuthority,
    MetricAuthoritySource,
    NovelFindingPolicy,
    RequiredContextLevel,
    RepositoryReviewTarget,
    ReviewTargetKind,
    TruthLocation,
    TruthCompleteness,
    canonical_json,
    canonical_sha256,
)
from review_agent_eval.repository import FixtureRepositoryBuilder

AUTHORING_MODULE_ROOT = Path(__file__).resolve().parents[2] / "eval" / "authoring"
if str(AUTHORING_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTHORING_MODULE_ROOT))

from core_human_review import (  # noqa: E402
    HumanReviewError,
    annotation_protocol_binding,
    fixture_manifest_from_mappings,
    make_packet,
    verify_current_case_approval,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPOSITORY_ROOT / "eval"
MANIFEST_PATHS = {
    "core-regression": Path("suites/core-regression/manifest.json"),
    "core-capability": Path("suites/core-capability/manifest.json"),
}
ANNOTATION_GUIDELINES = EVAL_ROOT / "annotation-guidelines.md"
CORE_PROTOCOL_ID = "native_repository"
ANNOTATION_CHECKLIST_VERSION = "core-annotation-v2"
AUTHORING_SCRIPT = EVAL_ROOT / "authoring" / "build_core_suites.py"
PROMOTION_VERIFIER = EVAL_ROOT / "authoring" / "verify_core_regression.py"
GOLDEN_INDEX = EVAL_ROOT / "cases" / "core" / "golden-index.json"
GOLDEN_INDEX_SCHEMA_VERSION = "core_golden_index_v2"
GOLDEN_ENTRY_SCHEMA_VERSION = "core_golden_entry_v2"
SUITE_SOURCE_PACKET_SCHEMA_VERSION = "core_suite_source_v3"
CORE_SOURCE_VERSION = "core-2026-07-21-v3"

REQUIRED_INTENT_SCENARIOS = {
    "explicit_intent",
    "inferred_intent",
    "must_clarify",
    "must_not_clarify",
    "unsupported_intent",
    "contradicted_intent",
    "user_correction",
}
REQUIRED_REVIEW_SCENARIOS = {
    "security",
    "regression",
    "clean_pr",
    "preexisting_trap",
    "wrong_path_trap",
    "wrong_line_trap",
    "fabricated_finding_trap",
    "duplicate_finding_trap",
    "compound_finding_trap",
    "high_miss",
    "critical_miss",
}
REQUIRED_DIMENSIONS = {
    "language",
    "intent.behavior",
    "review.category",
    "review.context",
    "risk.level",
}
AUTOMATED_CHECKLIST_ITEMS = {
    "agent_input_contains_no_truth",
    "base_head_binding_reproduced",
    "fixture_contains_no_vcs_metadata",
}
HUMAN_CHECKLIST_ITEMS = {
    "atomic_findings_reviewed",
    "evidence_anchors_are_non_exclusive",
    "known_invalid_traps_reviewed",
    "semantic_truth_leakage_reviewed",
    "severity_category_context_reviewed",
    "truth_completeness_reviewed",
    "human_review_completed",
}


def _required_file(path: Path) -> Path:
    assert path.is_file(), "Task 13 artifact is missing: %s" % path.relative_to(
        REPOSITORY_ROOT
    ).as_posix()
    return path


@lru_cache(maxsize=1)
def _banks() -> tuple[CaseBank, ...]:
    banks = []
    for suite_id, relative in MANIFEST_PATHS.items():
        _required_file(EVAL_ROOT / relative)
        bank = CaseBank.open(EVAL_ROOT, relative.as_posix())
        assert bank.suite_id == suite_id
        banks.append(bank)
    return tuple(banks)


def _loaded_cases() -> tuple[tuple[CaseBank, SuiteCase, EvalCase], ...]:
    return tuple(
        (bank, handle.entry, handle.load())
        for bank in _banks()
        for handle in bank
    )


def _dimensions(entry: SuiteCase) -> dict[str, str]:
    return {item.name: item.value for item in entry.dimensions}


def _annotation(case: EvalCase) -> dict[str, object]:
    path = EVAL_ROOT / "cases" / "core" / case.task_id / "annotation.json"
    _required_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _coverage(case: EvalCase) -> set[str]:
    raw = _annotation(case)["coverage"]
    assert type(raw) is list and all(type(item) is str for item in raw)
    return set(raw)


def _repository_target(case: EvalCase) -> RepositoryReviewTarget:
    target = case.input.review_target
    assert type(target) is RepositoryReviewTarget
    return target


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _snapshot(root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        assert not path.is_symlink(), "Core fixtures may not contain links: %s" % path
        if path.is_dir():
            assert path.name not in {".git", ".hg", ".svn", "__pycache__"}
            continue
        assert path.is_file(), "Core fixtures may contain only regular files"
        relative = path.relative_to(root).as_posix()
        assert not relative.endswith((".pyc", ".pyo"))
        files[relative] = path.read_bytes()
    return files


def _location_file(case: EvalCase, location: TruthLocation) -> Path:
    repository_path = _repository_target(case).repository.path
    assert repository_path is not None
    fixture = EVAL_ROOT / repository_path
    if location.side is DiffSide.LEFT:
        candidates = (fixture / "base" / location.path,)
    elif location.side is DiffSide.RIGHT:
        candidates = (fixture / "head" / location.path,)
    else:
        candidates = (
            fixture / "head" / location.path,
            fixture / "base" / location.path,
        )
    existing = tuple(path for path in candidates if path.is_file())
    assert existing, "truth location does not exist in its declared side: %s" % (
        location.path,
    )
    return existing[0]


def _assert_location_is_stable(case: EvalCase, location: TruthLocation) -> None:
    path = _location_file(case, location)
    if location.from_line is None:
        assert location.to_line is None
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    assert location.to_line is not None
    assert 1 <= location.from_line <= location.to_line <= len(lines), (
        "truth location line range is outside the immutable fixture: %s" % path
    )


def _changed_lines(
    before: list[str], after: list[str]
) -> tuple[set[int], set[int]]:
    left: set[int] = set()
    right: set[int] = set()
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        left.update(range(before_start + 1, before_end + 1))
        right.update(range(after_start + 1, after_end + 1))
    return left, right


def _annotation_private_values(case: EvalCase) -> tuple[set[str], set[str]]:
    identifiers = {
        item.truth_id
        for item in (
            *case.intent_truth.expected_claims,
            *case.intent_truth.forbidden_claims,
            *case.review_truth.expected_findings,
            *case.review_truth.known_invalid_findings,
        )
    }
    identifiers.update(item.answer_id for item in case.clarification_script.answers)
    texts = {
        item.rationale
        for item in (
            *case.intent_truth.forbidden_claims,
            *case.review_truth.expected_findings,
            *case.review_truth.known_invalid_findings,
        )
    }
    texts.update(item.claim for item in case.review_truth.expected_findings)
    texts.update(item.claim for item in case.review_truth.known_invalid_findings)
    for answer in case.clarification_script.answers:
        if answer.response is not None:
            texts.add(answer.response)
        texts.update(answer.corrected_values)
    return identifiers, {item for item in texts if len(item.strip()) >= 8}


def test_task13_declares_two_canonical_core_manifests_and_a_guideline() -> None:
    for path in MANIFEST_PATHS.values():
        _required_file(EVAL_ROOT / path)
    _required_file(ANNOTATION_GUIDELINES)
    _required_file(GOLDEN_INDEX)

    banks = _banks()
    assert len(banks) == 2
    for bank in banks:
        assert type(bank.manifest) is SuiteManifest
        assert bank.manifest.schema_version == "suite_manifest_v2"
        assert bank.manifest.source.kind is SuiteKind.CORE
        assert bank.manifest.source.preparation_binding is None
        contract = bank.manifest.wire_contract
        assert contract.case_schema_version == EVAL_CASE_SCHEMA_VERSION
        assert contract.input_schema_version == EVAL_INPUT_SCHEMA_VERSION
        assert contract.submission_schema_version == EVAL_SUBMISSION_SCHEMA_VERSION
        assert contract.review_target_kind is ReviewTargetKind.REPOSITORY
        assert contract.materializer_protocol == REPOSITORY_MATERIALIZER_PROTOCOL
        raw = (EVAL_ROOT / MANIFEST_PATHS[bank.suite_id]).read_bytes()
        assert raw == bank.manifest.to_json().encode("utf-8"), (
            "%s must be canonical UTF-8 JSON" % bank.suite_id
        )
        assert bank.manifest.digest() == canonical_sha256(bank.manifest.to_dict())
        assert hashlib.sha256(raw + b"\n").hexdigest() != bank.manifest.digest()


def test_core_suites_share_the_final_case_schema_and_have_18_unique_cases() -> None:
    banks = _banks()
    regression = next(bank for bank in banks if bank.suite_id == "core-regression")
    capability = next(bank for bank in banks if bank.suite_id == "core-capability")
    assert len(regression.handles) == 10
    assert len(capability.handles) == 8
    assert all(handle.split is CaseSplit.REGRESSION for handle in regression)
    assert all(handle.split is CaseSplit.CAPABILITY for handle in capability)

    loaded = _loaded_cases()
    task_ids = [case.task_id for _bank, _entry, case in loaded]
    case_paths = [entry.path for _bank, entry, _case in loaded]
    digests = [entry.canonical_case_digest for _bank, entry, _case in loaded]
    assert len(task_ids) == 18
    assert len(task_ids) == len(set(task_ids))
    assert len(case_paths) == len(set(case_paths))
    assert len(digests) == len(set(digests))
    assert {entry.protocol_id for _bank, entry, _case in loaded} == {
        CORE_PROTOCOL_ID
    }

    for bank, entry, case in loaded:
        assert type(case) is EvalCase
        assert case.source.suite == bank.suite_id
        assert case.source.origin is CaseOrigin.HAND_AUTHORED
        assert case.source.source_id == case.task_id + "-source"
        assert case.schema_version == EVAL_CASE_SCHEMA_VERSION
        assert case.case_version == 3
        assert case.review_evaluator_context.truth_contexts == ()
        assert entry.path == "cases/core/%s/case.json" % case.task_id
        target = _repository_target(case)
        assert target.kind is ReviewTargetKind.REPOSITORY
        assert target.repository.path == (
            "cases/core/%s/repository" % case.task_id
        )
        assert set(case.input.to_dict()) == {"review_target"}
        assert set(case.eval_input().to_dict()) == {
            "schema_version",
            "task_id",
            "review_target",
        }
        raw = (EVAL_ROOT / entry.path).read_bytes()
        assert raw == case.to_json().encode("utf-8"), (
            "%s must be canonical EvalCase v2 JSON" % case.task_id
        )
        assert entry.raw_file_size_bytes == len(raw)
        assert entry.raw_file_sha256 == hashlib.sha256(raw).hexdigest()
        assert entry.canonical_case_digest == case.digest()
        assert entry.eval_input_digest == case.eval_input().digest()
        assert entry.truth_completeness is case.review_truth.completeness


def test_core_v3_private_identifiers_are_global_opaque_sequences() -> None:
    loaded = _loaded_cases()
    assert {case.case_version for _bank, _entry, case in loaded} == {3}
    assert {case.source.source_version for _bank, _entry, case in loaded} == {
        CORE_SOURCE_VERSION
    }

    intent_ids = [
        item.truth_id
        for _bank, _entry, case in loaded
        for item in case.intent_truth.expected_claims
    ]
    forbidden_ids = [
        item.truth_id
        for _bank, _entry, case in loaded
        for item in case.intent_truth.forbidden_claims
    ]
    issue_ids = [
        item.truth_id
        for _bank, _entry, case in loaded
        for item in case.review_truth.expected_findings
    ]
    invalid_ids = [
        item.truth_id
        for _bank, _entry, case in loaded
        for item in case.review_truth.known_invalid_findings
    ]
    answer_ids = [
        item.answer_id
        for _bank, _entry, case in loaded
        for item in case.clarification_script.answers
    ]

    assert intent_ids == ["intent-%04d" % index for index in range(1, 41)]
    assert forbidden_ids == [
        "forbidden-intent-%04d" % index for index in range(1, 3)
    ]
    assert issue_ids == ["issue-%04d" % index for index in range(1, 17)]
    assert invalid_ids == ["invalid-%04d" % index for index in range(1, 4)]
    assert answer_ids == ["answer-%04d" % index for index in range(1, 5)]
    private_ids = [*intent_ids, *forbidden_ids, *issue_ids, *invalid_ids, *answer_ids]
    assert len(private_ids) == len(set(private_ids))


def test_golden_index_binds_exact_bytes_into_each_suite_source_hash() -> None:
    raw_index = _required_file(GOLDEN_INDEX).read_bytes()
    index = json.loads(raw_index.decode("utf-8"))
    assert type(index) is dict
    assert raw_index == canonical_json(index).encode("utf-8")
    assert set(index) == {
        "entries",
        "run_binding",
        "schema_version",
        "source_version",
    }
    assert index["schema_version"] == GOLDEN_INDEX_SCHEMA_VERSION
    assert index["source_version"] == CORE_SOURCE_VERSION
    run_binding = index["run_binding"]
    assert set(run_binding) == {
        "schema_version",
        "run_instance_key",
        "run_id",
        "attempt",
        "scenario_order",
    }
    assert run_binding["schema_version"] == "core_golden_run_binding_v2"
    assert run_binding["run_instance_key"] == "core-golden-authoring-v2"
    assert re.fullmatch(r"run-[0-9a-f]{64}", run_binding["run_id"])
    assert run_binding["attempt"] == 1
    assert len(run_binding["scenario_order"]) == 12
    assert len(set(run_binding["scenario_order"])) == 12
    entries = index["entries"]
    assert type(entries) is list and len(entries) == 12
    assert entries == sorted(entries, key=lambda item: item["path"])

    cases = {case.task_id: case for _bank, _entry, case in _loaded_cases()}
    paths = []
    by_suite: dict[str, list[str]] = {suite_id: [] for suite_id in MANIFEST_PATHS}
    for entry in entries:
        assert type(entry) is dict
        assert set(entry) == {
            "entry_digest",
            "canonical_submission_digest",
            "eval_input_digest",
            "path",
            "raw_file_sha256",
            "raw_file_size_bytes",
            "scenario",
            "suite_id",
            "task_id",
            "target_materialization_id",
            "trial_id",
        }
        path = entry["path"]
        task_id = entry["task_id"]
        assert type(path) is str and type(task_id) is str
        assert path == "cases/core/%s/golden/%s.json" % (
            task_id,
            entry["scenario"],
        )
        assert task_id in cases
        assert entry["suite_id"] == cases[task_id].source.suite
        golden_bytes = _required_file(EVAL_ROOT / path).read_bytes()
        submission = EvalSubmission.from_json(golden_bytes)
        assert entry["raw_file_size_bytes"] == len(golden_bytes)
        assert entry["raw_file_sha256"] == hashlib.sha256(golden_bytes).hexdigest()
        assert entry["canonical_submission_digest"] == submission.digest()
        assert entry["eval_input_digest"] == submission.eval_input_digest
        assert entry["trial_id"] == submission.trial_id
        assert entry["target_materialization_id"] == (
            submission.target_materialization_id
        )
        assert submission.eval_input_digest == cases[task_id].eval_input().digest()
        core = {
            name: entry[name]
            for name in (
                "path",
                "task_id",
                "suite_id",
                "scenario",
                "raw_file_size_bytes",
                "raw_file_sha256",
                "canonical_submission_digest",
                "eval_input_digest",
                "trial_id",
                "target_materialization_id",
            )
        }
        assert entry["entry_digest"] == canonical_sha256(
            {"schema_version": GOLDEN_ENTRY_SCHEMA_VERSION, **core}
        )
        assert hashlib.sha256(golden_bytes + b"\n").hexdigest() != entry[
            "raw_file_sha256"
        ], "replacing a Golden byte must invalidate its indexed binding"
        paths.append(path)
        by_suite[entry["suite_id"]].append(entry["entry_digest"])
    assert len(paths) == len(set(paths))

    for bank in _banks():
        case_bindings = sorted(
            (
                {
                    "task_id": handle.task_id,
                    "case_source_content_hash": handle.load().source.content_hash,
                }
                for handle in bank
            ),
            key=lambda item: item["task_id"],
        )
        source_packet = {
            "schema_version": SUITE_SOURCE_PACKET_SCHEMA_VERSION,
            "suite_id": bank.suite_id,
            "source_version": bank.manifest.source.source_version,
            "cases": case_bindings,
            "golden_entry_digests": sorted(by_suite[bank.suite_id]),
        }
        assert bank.manifest.source.content_hash == canonical_sha256(source_packet)


def test_annotation_ledgers_bind_cases_repositories_and_suite_taxonomy() -> None:
    intent_scenarios: set[str] = set()
    review_scenarios: set[str] = set()

    for _bank, entry, case in _loaded_cases():
        dimensions = _dimensions(entry)
        assert REQUIRED_DIMENSIONS <= set(dimensions), (
            "%s lacks required Suite dimensions" % case.task_id
        )
        assert dimensions["language"] == "python"

        annotation = _annotation(case)
        assert set(annotation) == {
            "annotation_rationale",
            "authoring",
            "case_binding",
            "case_version",
            "checklist",
            "coverage",
            "disagreements",
            "human_review",
            "intent_expectation",
            "provenance_binding",
            "repository_binding",
            "schema_version",
            "suite_assignment",
            "suite_id",
            "task_id",
            "truth_summary",
        }
        assert annotation["schema_version"] == "core_annotation_record_v2"
        assert annotation["task_id"] == case.task_id
        assert annotation["case_version"] == case.case_version
        assert annotation["suite_id"] == case.source.suite

        case_binding = annotation["case_binding"]
        assert type(case_binding) is dict
        assert case_binding == {
            "canonical_case_digest": entry.canonical_case_digest,
            "eval_input_digest": entry.eval_input_digest,
            "case_source_content_hash": case.source.content_hash,
        }
        assignment = annotation["suite_assignment"]
        assert type(assignment) is dict
        assert set(assignment) == {
            "promotion_evidence",
            "split",
            "status",
        }
        assert assignment["split"] == entry.split.value
        assert assignment["status"] in {
            "capability",
            "pending_current_agent_baseline",
            "regression_baseline_confirmed",
        }
        if assignment["status"] != "regression_baseline_confirmed":
            assert assignment["promotion_evidence"] is None
        checklist = annotation["checklist"]
        assert type(checklist) is dict
        assert set(checklist) == AUTOMATED_CHECKLIST_ITEMS | HUMAN_CHECKLIST_ITEMS
        assert all(type(value) is bool for value in checklist.values())
        assert all(checklist[name] for name in AUTOMATED_CHECKLIST_ITEMS)
        human_review = annotation["human_review"]
        assert type(human_review) is dict
        assert set(human_review) == {
            "adjudication_digest",
            "adjudicator_id",
            "annotation_protocol_version",
            "approval_identity_status",
            "author_id",
            "base_revision",
            "base_tree",
            "blind_review_completed_at",
            "blind_review_packet_digest",
            "blind_review_started_at",
            "canonical_case_digest",
            "case_version",
            "eval_input_digest",
            "final_decision",
            "head_revision",
            "head_tree",
            "independent_annotation_digest",
            "leakage_review_completed",
            "prior_approval_carried_forward",
            "review_batch_id",
            "reviewer_id",
            "schema_version",
            "status",
            "task_id",
        }
        assert human_review["schema_version"] == "core_human_review_record_v2"
        assert human_review["status"] in {
            "requires_independent_re_review",
            "approved",
            "rejected",
        }
        assert human_review["approval_identity_status"] in {
            "requires_re_review",
            "current_source_bound",
        }
        assert human_review["prior_approval_carried_forward"] is False
        assert human_review["task_id"] == case.task_id
        assert human_review["case_version"] == case.case_version
        assert human_review["canonical_case_digest"] == entry.canonical_case_digest
        assert human_review["eval_input_digest"] == entry.eval_input_digest
        repository = _repository_target(case).repository
        assert human_review["base_revision"] == repository.base_revision
        assert human_review["head_revision"] == repository.head_revision
        assert human_review["base_tree"] == annotation["repository_binding"]["base_tree"]
        assert human_review["head_tree"] == annotation["repository_binding"]["head_tree"]
        assert human_review["annotation_protocol_version"] == (
            ANNOTATION_CHECKLIST_VERSION
        )
        assert repository.path is not None
        fixture_root = EVAL_ROOT / repository.path
        fixture_manifest = fixture_manifest_from_mappings(
            {
                path: raw.decode("utf-8")
                for path, raw in _snapshot(fixture_root / "base").items()
            },
            {
                path: raw.decode("utf-8")
                for path, raw in _snapshot(fixture_root / "head").items()
            },
        )
        expected_packet = make_packet(
            case,
            annotation["repository_binding"],
            fixture_manifest,
            annotation_protocol_binding(EVAL_ROOT),
        )
        assert human_review["blind_review_packet_digest"] == expected_packet["packet_digest"]

        disagreements = annotation["disagreements"]
        assert type(disagreements) is dict
        assert set(disagreements) == {
            "adjudication_digest",
            "adjudicator_id",
            "items",
            "status",
        }
        assert type(disagreements["items"]) is list
        assert disagreements["status"] in {
            "pending_blind_review",
            "requires_independent_re_review",
            "none",
            "resolved",
            "unresolved",
        }
        if human_review["status"] == "requires_independent_re_review":
            assert not any(checklist[name] for name in HUMAN_CHECKLIST_ITEMS)
            assert human_review["final_decision"] is None
            assert disagreements["status"] == "requires_independent_re_review"
            assert human_review["approval_identity_status"] == "requires_re_review"

        coverage = _coverage(case)
        intent_scenarios.update(coverage & REQUIRED_INTENT_SCENARIOS)
        review_scenarios.update(coverage & REQUIRED_REVIEW_SCENARIOS)

    assert REQUIRED_INTENT_SCENARIOS <= intent_scenarios
    assert REQUIRED_REVIEW_SCENARIOS <= review_scenarios


def test_independent_human_re_review_gate_is_explicitly_unmet() -> None:
    pending = []
    for _bank, _entry, case in _loaded_cases():
        try:
            verify_current_case_approval(EVAL_ROOT, case.task_id)
        except HumanReviewError as exc:
            pending.append(case.task_id)
            assert "no evaluator-private human approval" in str(exc)
        else:  # pragma: no cover - checked-in external evidence is intentionally absent
            raise AssertionError("Task 3 must not fabricate or carry forward approval")
        annotation = _annotation(case)
        assert annotation["human_review"]["status"] == (
            "requires_independent_re_review"
        )
        assert annotation["human_review"]["approval_identity_status"] == (
            "requires_re_review"
        )
        assert annotation["human_review"]["prior_approval_carried_forward"] is False
        assert not annotation["checklist"]["human_review_completed"]
    assert pending == sorted(case.task_id for _bank, _entry, case in _loaded_cases())


def test_real_three_trial_current_agent_baseline_gate_is_explicitly_unmet() -> None:
    pending = []
    for bank, entry, case in _loaded_cases():
        if entry.split is not CaseSplit.REGRESSION:
            continue
        assert bank.suite_id == "core-regression"
        assignment = _annotation(case)["suite_assignment"]
        assert type(assignment) is dict
        assert assignment["status"] == "pending_current_agent_baseline"
        assert assignment["promotion_evidence"] is None
        pending.append(case.task_id)
    assert pending == ["core-py-%03d" % index for index in range(1, 11)]
    _required_file(PROMOTION_VERIFIER)


def test_intent_truth_covers_authority_scorability_and_clarification_contracts() -> None:
    loaded = _loaded_cases()
    scorable = [case for _bank, _entry, case in loaded if case.intent_truth.scorable]
    assert len(scorable) == len(loaded)
    assert {case.intent_truth.authority for case in scorable} == {
        IntentAuthority.EXPLICIT_AUTHOR_METADATA,
        IntentAuthority.LINKED_REQUIREMENT,
        IntentAuthority.SYNTHETIC,
    }
    assert {
        case.intent_truth.clarification_policy for case in scorable
    } == {
        ClarificationPolicy.REQUIRED,
        ClarificationPolicy.NOT_REQUIRED,
    }

    by_scenario: dict[str, list[EvalCase]] = {
        scenario: [
            case
            for _bank, _entry, case in loaded
            if scenario in _coverage(case)
        ]
        for scenario in REQUIRED_INTENT_SCENARIOS
    }

    assert any(
        case.intent_truth.authority is IntentAuthority.EXPLICIT_AUTHOR_METADATA
        and _repository_target(case).review_request.user_intent is not None
        for case in by_scenario["explicit_intent"]
    )
    assert any(
        case.intent_truth.authority
        in {IntentAuthority.EXPERT_RECONSTRUCTED, IntentAuthority.SYNTHETIC}
        and _repository_target(case).review_request.user_intent is None
        and case.intent_truth.expected_claims
        and _annotation(case)["intent_expectation"]["initial_source"] == "inferred"
        for case in by_scenario["inferred_intent"]
    )
    assert all(
        case.intent_truth.clarification_policy is ClarificationPolicy.REQUIRED
        and case.clarification_script.answers
        for case in by_scenario["must_clarify"]
    )
    assert all(
        case.intent_truth.clarification_policy is ClarificationPolicy.NOT_REQUIRED
        for case in by_scenario["must_not_clarify"]
    )
    assert all(
        not case.intent_truth.forbidden_claims
        for case in by_scenario["unsupported_intent"]
    )
    assert all(
        case.intent_truth.forbidden_claims
        for case in by_scenario["contradicted_intent"]
    )
    assert any(
        answer.action is ClarificationAction.CORRECT
        for case in by_scenario["user_correction"]
        for answer in case.clarification_script.answers
    )
    for _bank, _entry, case in loaded:
        assert case.clarification_script.max_rounds == max(
            1, len(case.clarification_script.answers)
        )


def test_review_truth_is_atomic_audited_and_covers_core_review_risks() -> None:
    loaded = _loaded_cases()
    expected = [
        finding
        for _bank, _entry, case in loaded
        for finding in case.review_truth.expected_findings
    ]
    invalid = [
        finding
        for _bank, _entry, case in loaded
        for finding in case.review_truth.known_invalid_findings
    ]
    assert expected and invalid
    assert {case.review_truth.completeness for _bank, _entry, case in loaded} == {
        TruthCompleteness.CLOSED_WORLD
    }
    assert {case.review_truth.novel_finding_policy for _bank, _entry, case in loaded} == {
        NovelFindingPolicy.FORBID,
        NovelFindingPolicy.VERIFY,
    }
    assert {finding.severity for finding in expected} == set(FindingSeverity)
    assert all(
        finding.metric_authority.severity_scorable
        and finding.metric_authority.severity_authority
        is MetricAuthoritySource.EXPERT_ANNOTATION
        and finding.metric_authority.location_scorable
        and finding.metric_authority.location_authority
        is MetricAuthoritySource.EXPERT_ANNOTATION
        for finding in expected
    )
    assert all(finding.locations for finding in expected)
    assert all(
        case.review_evaluator_context.truth_contexts == ()
        for _bank, _entry, case in loaded
    )
    assert {finding.required_context_level for finding in expected} == set(
        RequiredContextLevel
    )
    categories = {_normalized(finding.category) for finding in expected}
    assert categories == {"correctness", "security", "regression"}
    assert any(not finding.evidence_anchors for finding in expected)
    assert any(finding.evidence_anchors for finding in expected)

    clean_cases = [
        case
        for _bank, _entry, case in loaded
        if "clean_pr" in _coverage(case)
    ]
    assert clean_cases
    assert all(not case.review_truth.expected_findings for case in clean_cases)

    trap_cases = [
        case
        for _bank, _entry, case in loaded
        if "preexisting_trap" in _coverage(case)
    ]
    assert any(
        location.side is DiffSide.LEFT
        for case in trap_cases
        for finding in case.review_truth.known_invalid_findings
        for location in finding.locations
    )
    required_severities = {
        finding.severity
        for _bank, _entry, case in loaded
        for finding in case.review_truth.expected_findings
        if finding.required
    }
    assert {FindingSeverity.HIGH, FindingSeverity.CRITICAL} <= required_severities

    for _bank, _entry, case in loaded:
        intent_claims = [
            _normalized(item.text)
            for item in (
                *case.intent_truth.expected_claims,
                *case.intent_truth.forbidden_claims,
            )
        ]
        review_claims = [
            _normalized(item.claim)
            for item in (
                *case.review_truth.expected_findings,
                *case.review_truth.known_invalid_findings,
            )
        ]
        assert len(intent_claims) == len(set(intent_claims))
        assert len(review_claims) == len(set(review_claims))
        for finding in (
            *case.review_truth.expected_findings,
            *case.review_truth.known_invalid_findings,
        ):
            assert "\n" not in finding.claim and "\r" not in finding.claim
            assert _normalized(finding.rationale) != _normalized(finding.claim)
            minimum_rationale_chars = (
                200 if finding in case.review_truth.expected_findings else 80
            )
            assert len(finding.rationale) >= minimum_rationale_chars
            for location in finding.locations:
                _assert_location_is_stable(case, location)
        for finding in case.review_truth.expected_findings:
            anchor_facts = [_normalized(anchor.fact) for anchor in finding.evidence_anchors]
            assert len(anchor_facts) == len(set(anchor_facts))
            for anchor in finding.evidence_anchors:
                assert _normalized(anchor.fact) != _normalized(finding.claim)
                for location in anchor.locations:
                    _assert_location_is_stable(case, location)
        if case.review_truth.completeness is not TruthCompleteness.CLOSED_WORLD:
            assert case.review_truth.novel_finding_policy is NovelFindingPolicy.VERIFY


def test_every_expected_finding_points_to_at_least_one_changed_line() -> None:
    for _bank, _entry, case in _loaded_cases():
        repository_path = _repository_target(case).repository.path
        assert repository_path is not None
        fixture = EVAL_ROOT / repository_path
        changed_by_path: dict[str, tuple[set[int], set[int]]] = {}
        relative_paths = {
            path.relative_to(fixture / side).as_posix()
            for side in ("base", "head")
            for path in (fixture / side).rglob("*")
            if path.is_file()
        }
        for relative in relative_paths:
            base_path = fixture / "base" / relative
            head_path = fixture / "head" / relative
            before = (
                base_path.read_text(encoding="utf-8").splitlines()
                if base_path.is_file()
                else []
            )
            after = (
                head_path.read_text(encoding="utf-8").splitlines()
                if head_path.is_file()
                else []
            )
            changed_by_path[relative] = _changed_lines(before, after)

        for finding in case.review_truth.expected_findings:
            overlaps_change = False
            for location in finding.locations:
                if location.from_line is None or location.to_line is None:
                    continue
                left, right = changed_by_path.get(location.path, (set(), set()))
                declared = set(range(location.from_line, location.to_line + 1))
                if location.side is DiffSide.LEFT:
                    overlaps_change = bool(declared & left)
                elif location.side is DiffSide.RIGHT:
                    overlaps_change = bool(declared & right)
                else:
                    overlaps_change = bool(declared & (left | right))
                if overlaps_change:
                    break
            assert overlaps_change, (
                "%s/%s has no TruthLocation on a changed line"
                % (case.task_id, finding.truth_id)
            )


def test_fixture_trees_are_small_python_and_bound_to_immutable_digests(
    tmp_path: Path,
) -> None:
    repository_bindings = []
    source_digests = []
    for _bank, _entry, case in _loaded_cases():
        target = _repository_target(case)
        repository = target.repository
        assert repository.path is not None
        fixture = EVAL_ROOT / repository.path
        assert fixture.is_dir()
        assert {path.name for path in fixture.iterdir()} == {"base", "head"}
        base_files = _snapshot(fixture / "base")
        head_files = _snapshot(fixture / "head")
        assert len(base_files) <= 64 and len(head_files) <= 64
        assert sum(map(len, base_files.values())) <= 256 * 1024
        assert sum(map(len, head_files.values())) <= 256 * 1024
        python_paths = {
            path for path in (*base_files, *head_files) if path.endswith(".py")
        }
        assert python_paths, "%s is not a Python Case" % case.task_id
        assert any(base_files.get(path) != head_files.get(path) for path in python_paths)
        for path in python_paths:
            for snapshot in (base_files, head_files):
                if path in snapshot:
                    ast.parse(snapshot[path].decode("utf-8"), filename=path)

        built = FixtureRepositoryBuilder().build(
            fixture, tmp_path / (case.task_id + ".git")
        )
        assert repository.base_revision == built.base_revision
        assert repository.head_revision == built.head_revision
        annotation = _annotation(case)
        binding = annotation["repository_binding"]
        assert type(binding) is dict
        assert binding == {
            "base_revision": built.base_revision,
            "head_revision": built.head_revision,
            "base_tree": built.base_tree,
            "head_tree": built.head_tree,
            "source_digest": built.source_digest,
        }
        expected_source_hash = canonical_sha256(
            {
                "schema_version": "core_case_provenance_v2",
                "task_id": case.task_id,
                "case_version": case.case_version,
                "source_version": case.source.source_version,
                "annotation_protocol_version": ANNOTATION_CHECKLIST_VERSION,
                "review_target_kind": "repository",
                "review_request": target.review_request.to_dict(),
                "fixture_source_digest": built.source_digest,
                "base_tree": built.base_tree,
                "head_tree": built.head_tree,
            }
        )
        assert case.source.content_hash == expected_source_hash
        provenance = annotation["provenance_binding"]
        assert type(provenance) is dict
        assert provenance["content_hash"] == expected_source_hash
        assert re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", built.base_tree)
        assert re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", built.head_tree)
        assert built.base_tree != built.head_tree
        repository_bindings.append((built.base_revision, built.head_revision))
        source_digests.append(built.source_digest)

    assert len(repository_bindings) == len(set(repository_bindings))
    assert len(source_digests) == len(set(source_digests))


def test_truth_never_leaks_into_agent_view_or_repository_fixtures() -> None:
    forbidden_wire_keys = {
        "intent_truth",
        "review_truth",
        "clarification_script",
        "expected_findings",
        "known_invalid_findings",
        "evidence_anchors",
    }
    for bank, _entry, case in _loaded_cases():
        agent_json = bank.agent_input(case.task_id).to_json()
        assert not any('"%s"' % key in agent_json for key in forbidden_wire_keys)
        identifiers, private_texts = _annotation_private_values(case)
        folded_agent = agent_json.casefold()
        for value in (*identifiers, *private_texts):
            assert value.casefold() not in folded_agent, (
                "private annotation leaked into EvalInput for %s" % case.task_id
            )

        repository_path = _repository_target(case).repository.path
        assert repository_path is not None
        fixture = EVAL_ROOT / repository_path
        for side in ("base", "head"):
            for relative, raw in _snapshot(fixture / side).items():
                folded_path = relative.casefold()
                assert not any(value.casefold() in folded_path for value in identifiers)
                assert not any(
                    marker in folded_path
                    for marker in ("ground_truth", "golden_answer", "solution_key")
                )
                text = raw.decode("utf-8", "ignore").casefold()
                for value in (*identifiers, *private_texts):
                    assert value.casefold() not in text, (
                        "private annotation leaked into %s/%s/%s"
                        % (case.task_id, side, relative)
                    )


def test_core_authoring_outputs_are_reproducible(tmp_path: Path) -> None:
    _required_file(AUTHORING_SCRIPT)
    module_name = "_review_agent_core_suite_authoring"
    spec = importlib.util.spec_from_file_location(module_name, AUTHORING_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        plan = module.build_plan(tmp_path)
        assert len(plan.writable_outputs) == 52
        assert len(plan.check_only_fixtures) == 54
        assert set(plan.writable_outputs).isdisjoint(plan.check_only_fixtures)
        assert not any(
            "/repository/base/" in relative
            or "/repository/head/" in relative
            for relative in plan.writable_outputs
        )
        assert all(
            "/repository/base/" in relative
            or "/repository/head/" in relative
            for relative in plan.check_only_fixtures
        )
        assert module._summary(plan) == {
            "schema_version": "core_suite_authoring_summary_v2",
            "case_count": 18,
            "golden_count": 12,
            "suite_count": 2,
            "regression_count": 10,
            "capability_count": 8,
            "writable_generated_file_count": 52,
            "checked_fixture_file_count": 54,
            "human_review_status": "requires_independent_re_review",
        }
        assert module.check_outputs(EVAL_ROOT, plan) == []
    finally:
        sys.modules.pop(module_name, None)


def test_core_authoring_write_outputs_round_trips_exact_bytes_and_check_is_clean(
    tmp_path: Path,
) -> None:
    module_name = "_review_agent_core_suite_authoring_integration"
    spec = importlib.util.spec_from_file_location(module_name, AUTHORING_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        plan = module.build_plan(tmp_path)
        isolated_eval = tmp_path / "isolated-eval"
        fixture_hashes: dict[str, str] = {}
        for relative, raw in plan.check_only_fixtures.items():
            target = isolated_eval.joinpath(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            fixture_hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()

        module.write_outputs(isolated_eval, plan)

        for relative, expected in plan.writable_outputs.items():
            assert isolated_eval.joinpath(*relative.split("/")).read_bytes() == expected
        for relative, expected_hash in fixture_hashes.items():
            target = isolated_eval.joinpath(*relative.split("/"))
            assert hashlib.sha256(target.read_bytes()).hexdigest() == expected_hash
        assert module.check_outputs(isolated_eval, plan) == []
    finally:
        sys.modules.pop(module_name, None)


def test_annotation_guidelines_define_the_human_audit_contract() -> None:
    text = _required_file(ANNOTATION_GUIDELINES).read_text(encoding="utf-8")
    folded = text.casefold()
    assert ANNOTATION_CHECKLIST_VERSION in folded
    required_tokens = {
        "Intent authority": "intentauthority",
        "clarification": "clarification",
        "atomic finding": "expectedfinding",
        "severity": "severity",
        "truth completeness": "truthcompleteness",
        "evidence anchor": "evidenceanchor",
        "metric authority": "metric_authority",
        "review evaluator context": "review_evaluator_context",
        "materialization binding": "target_materialization_id",
        "non-unique investigation path": "唯一调查路径",
        "known invalid": "knowninvalidfinding",
        "disagreement": "disagreement",
        "human review record": "人工审阅记录",
        "case version": "case_version",
        "external gates": "core-regression-promotion-v2",
    }
    for concept, token in required_tokens.items():
        assert token in folded, "annotation guidelines omit %s" % concept

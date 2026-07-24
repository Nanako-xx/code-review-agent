from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import review_agent_eval.models as eval_models
from review_agent_eval.models import (
    CaseOrigin,
    ClarificationAction,
    ClarificationPolicy,
    EvalCase,
    EvalInput,
    EvalSubmission,
    EvidenceIntegrity,
    EvidenceKind,
    EvidenceStream,
    EvidenceSupport,
    FailureCode,
    FindingSeverity,
    IntentAuthority,
    IntentClaimSource,
    IntentDimension,
    IntentResult,
    IssueJudgement,
    JudgeStatus,
    NovelFindingPolicy,
    RepositorySource,
    RequiredContextLevel,
    SchemaError,
    SubmissionClarificationExchange,
    SubmissionIntent,
    SubmissionStatus,
    TrialStatus,
    TruthCompleteness,
    canonical_json,
    canonical_sha256,
    stable_id,
    validate_submission_for_case,
)


BASE = "a" * 40
HEAD = "b" * 40
EMPTY_HASH = hashlib.sha256(b"").hexdigest()
CI_TEXT = "tests passed"
CI_HASH = hashlib.sha256(CI_TEXT.encode("utf-8")).hexdigest()
MATERIALIZATION_ID = "materialization-001"


def input_payload() -> dict:
    return {
        "schema_version": "eval_input_v2",
        "task_id": "task-001",
        "review_target": {
            "kind": "repository",
            "repository": {
                "source": "fixture",
                "path": "fixtures/auth-repo",
                "url": None,
                "base_revision": BASE,
                "head_revision": HEAD,
            },
            "review_request": {
                "title": "Review authorization change",
                "description": None,
                "user_intent": "Preserve admin-only access",
                "review_focus": None,
                "linked_requirements": ["REQ-7 must remain true"],
                "project_rules": ["Do not weaken authorization"],
                "existing_ci_evidence": [
                    {
                        "source_id": "ci-1",
                        "text": CI_TEXT,
                        "content_hash": CI_HASH,
                    }
                ],
            },
        },
    }


def frozen_input_payload() -> dict:
    return {
        "schema_version": "eval_input_v2",
        "task_id": "task-frozen",
        "review_target": {
            "kind": "frozen_context",
            "bundle_id": "bundle-001",
            "record_id": "record-001",
            "context_format": "rendered_text",
            "rendered_sha256": "e" * 64,
            "rendered_utf8_bytes": 128,
            "source_binding_digest": "f" * 64,
        },
    }


def usage_payload() -> dict:
    return {
        "elapsed_seconds": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "tool_calls": None,
        "cost_amount": None,
        "cost_currency": None,
    }


def intent_payload(*, clarification_questions: list | None = None) -> dict:
    return {
        "status": "sufficient",
        "goal": "Preserve admin-only access",
        "acceptance_criteria": ["Unauthenticated users remain denied"],
        "scope": ["app/auth.py"],
        "constraints": [],
        "claims": [
            {
                "claim_id": "claim-goal",
                "dimension": "goal",
                "text": "Preserve admin-only access",
                "source": "inferred",
            }
        ],
        "clarification_questions": clarification_questions or [],
        "uncertainties": [],
    }


def review_payload() -> dict:
    return {"findings": [], "uncertainties": []}


def submission_payload() -> dict:
    return {
        "schema_version": "eval_submission_v2",
        "task_id": "task-001",
        "agent_id": "agent-current",
        "trial_id": "trial-001",
        "eval_input_digest": EvalInput.from_dict(input_payload()).digest(),
        "target_materialization_id": MATERIALIZATION_ID,
        "status": "completed",
        "intent": intent_payload(),
        "review": review_payload(),
        "evidence": [],
        "usage": usage_payload(),
        "trace_ref": None,
        "failure": None,
    }


def clarification_answer_payload(
    *, answer_id: str = "answer-goal", action: str = "confirm"
) -> dict:
    return {
        "answer_id": answer_id,
        "dimension": "goal",
        "material_claim": "The endpoint remains admin-only",
        "action": action,
        "response": (
            "Yes"
            if action == "confirm"
            else "Use the corrected policy" if action == "correct" else None
        ),
        "corrected_values": ["Admin only"] if action == "correct" else [],
    }


def case_payload() -> dict:
    raw_input = input_payload()
    return {
        "schema_version": "eval_case_v2",
        "task_id": raw_input["task_id"],
        "case_version": 1,
        "source": {
            "suite": "core-regression",
            "origin": "hand_authored",
            "source_id": "auth-case-source",
            "source_version": "1",
            "source_uri": None,
            "license": None,
            "content_hash": "c" * 64,
        },
        "input": {"review_target": raw_input["review_target"]},
        "clarification_script": {
            "max_rounds": 2,
            "answers": [clarification_answer_payload()],
        },
        "intent_truth": {
            "scorable": True,
            "authority": "linked_requirement",
            "expected_claims": [
                {
                    "truth_id": "intent-expected-1",
                    "dimension": "goal",
                    "text": "Preserve admin-only access",
                    "required": True,
                }
            ],
            "forbidden_claims": [
                {
                    "truth_id": "intent-forbidden-1",
                    "dimension": "scope",
                    "text": "Rewrite unrelated billing code",
                    "rationale": "Billing is outside the requested change.",
                }
            ],
            "clarification_policy": "required",
        },
        "review_truth": {
            "completeness": "closed_world",
            "novel_finding_policy": "verify",
            "expected_findings": [
                {
                    "truth_id": "issue-1",
                    "claim": "The admin check was removed.",
                    "severity": "high",
                    "category": "security",
                    "required": True,
                    "metric_authority": {
                        "severity_scorable": True,
                        "severity_authority": "expert_annotation",
                        "location_scorable": True,
                        "location_authority": "expert_annotation",
                    },
                    "locations": [
                        {
                            "path": "app/auth.py",
                            "side": "right",
                            "from_line": 42,
                            "to_line": 45,
                        }
                    ],
                    "evidence_anchors": [
                        {
                            "fact": "The new path reaches the handler without an admin guard.",
                            "locations": [
                                {
                                    "path": "app/auth.py",
                                    "side": None,
                                    "from_line": None,
                                    "to_line": None,
                                }
                            ],
                        }
                    ],
                    "required_context_level": "file",
                    "rationale": "This exposes an admin-only endpoint.",
                }
            ],
            "known_invalid_findings": [
                {
                    "truth_id": "invalid-1",
                    "claim": "The endpoint was deleted.",
                    "category": None,
                    "locations": [],
                    "rationale": "The endpoint still exists in the head tree.",
                }
            ],
        },
        "review_evaluator_context": {"truth_contexts": []},
    }


def evidence_source_payload(kind: str) -> dict:
    common = {
        "kind": kind,
        "target_materialization_id": MATERIALIZATION_ID,
    }
    if kind == "repository_file":
        return dict(
            common,
            revision="HEAD",
            path="../not-authorized.py",
            from_line=9,
            to_line=3,
        )
    if kind == "repository_diff":
        return dict(
            common,
            base_revision="BASE",
            head_revision="HEAD",
            path="../not-authorized.py",
        )
    if kind == "frozen_context":
        return dict(common, context_ref="context-001", from_line=9, to_line=3)
    if kind == "command_output":
        return dict(
            common,
            command=["tool", "--flag"],
            exit_code=7,
            stream="stderr",
            artifact_ref="artifact-001",
        )
    if kind == "external_record":
        return dict(common, source_ref="dangling-attestation")
    raise AssertionError("unknown test evidence kind")


def evaluator_source_payload(content: str = "@@ -1 +1 @@\n-old\n+new") -> dict:
    return {
        "kind": "diff_hunk",
        "content": content,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "provenance": {
            "source_role": "annotations",
            "source_file_sha256": "1" * 64,
            "record_pointer": "/records/0/comments/0",
            "record_sha256": "2" * 64,
        },
    }


def exchange_payload(
    action: str | None,
    *,
    turn_index: int = 1,
    question_id: str = "question-1",
) -> dict:
    matched = action is not None
    return {
        "turn_index": turn_index,
        "question_id": question_id,
        "dimension": "goal",
        "question": "Should this remain admin-only?",
        "material_claim": "The endpoint remains admin-only",
        "matched_answer_id": "answer-goal" if matched else None,
        "action": action,
        "response": (
            "Yes"
            if action == "confirm"
            else "Use the corrected policy" if action == "correct" else None
        ),
        "resolved_values": (
            ["Admin only"] if action in {"confirm", "correct"} else []
        ),
    }


def test_eval_enums_are_owned_by_eval_and_have_frozen_v2_values() -> None:
    assert {item.value for item in TrialStatus} == {
        "pending",
        "running",
        "incomplete",
        "completed",
        "failed",
        "blocked",
        "invalid_output",
    }
    assert {item.value for item in SubmissionStatus} == {
        "completed",
        "failed",
        "blocked",
        "invalid_output",
    }
    assert {item.value for item in FailureCode} == {
        "timeout",
        "non_zero_exit",
        "process_killed",
        "output_overflow",
        "invalid_json",
        "schema_mismatch",
        "clarification_required",
        "agent_blocked",
        "adapter_error",
        "harness_materialization_error",
        "unknown",
    }
    assert {item.value for item in eval_models.ReviewTargetKind} == {
        "repository",
        "frozen_context",
    }
    assert {item.value for item in EvidenceKind} == {
        "repository_file",
        "repository_diff",
        "frozen_context",
        "command_output",
        "external_record",
    }
    assert {item.value for item in eval_models.MetricAuthoritySource} == {
        "expert_annotation",
        "upstream_annotation",
    }
    assert {item.value for item in eval_models.EvaluatorContextTask} == {
        "finding_equivalence"
    }
    assert {item.value for item in eval_models.EvaluatorContextSourceKind} == {
        "diff_hunk"
    }
    assert {item.value for item in IntentDimension} == {
        "goal",
        "acceptance_criterion",
        "scope",
        "constraint",
    }
    assert {item.value for item in IntentResult} == {
        "sufficient",
        "partial",
        "insufficient",
    }
    assert {item.value for item in IssueJudgement} == {
        "confirmed",
        "plausible",
        "fabricated",
        "unknown",
    }
    assert {item.value for item in EvidenceIntegrity} == {
        "valid",
        "invalid",
        "missing",
    }
    assert {item.value for item in EvidenceSupport} == {
        "supported",
        "weak",
        "unsupported",
        "unknown",
    }
    assert {item.value for item in JudgeStatus} == {
        "graded",
        "judge_failed",
        "ungraded",
    }
    assert all(item.__class__.__module__ == "review_agent_eval.models" for item in TrialStatus)


def test_models_are_python_39_source_compatible_and_do_not_import_product_runtime() -> None:
    source = Path(eval_models.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 9))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(name == "review_agent" or name.startswith("review_agent.") for name in imported_modules)


def test_eval_input_round_trips_as_deeply_immutable_tuples_without_private_script() -> None:
    model = EvalInput.from_dict(input_payload())

    assert model.to_dict() == input_payload()
    assert isinstance(model.review_target.review_request.linked_requirements, tuple)
    assert isinstance(model.review_target.review_request.existing_ci_evidence, tuple)
    assert not hasattr(model, "clarification_script")
    assert "clarification_script" not in model.to_dict()
    assert "clarification_policy" not in canonical_json(model)
    with pytest.raises(FrozenInstanceError):
        model.task_id = "changed"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        model.review_target.review_request.linked_requirements.append("mutate")  # type: ignore[attr-defined]


def test_repository_and_frozen_review_targets_round_trip_without_inline_content() -> None:
    repository_input = EvalInput.from_dict(input_payload())
    frozen_input = EvalInput.from_dict(frozen_input_payload())

    assert (
        repository_input.review_target.kind
        is eval_models.ReviewTargetKind.REPOSITORY
    )
    assert (
        frozen_input.review_target.kind
        is eval_models.ReviewTargetKind.FROZEN_CONTEXT
    )
    assert frozen_input.to_dict() == frozen_input_payload()
    assert "rendered" not in frozen_input.to_dict()["review_target"]


def test_metric_authority_accepts_only_authorized_severity_and_location_shapes() -> None:
    unscorable = case_payload()
    finding = unscorable["review_truth"]["expected_findings"][0]
    finding["severity"] = None
    finding["metric_authority"] = {
        "severity_scorable": False,
        "severity_authority": None,
        "location_scorable": False,
        "location_authority": None,
    }
    assert EvalCase.from_dict(unscorable).to_dict() == unscorable

    invalid_cases = []

    missing_severity = case_payload()
    missing_severity["review_truth"]["expected_findings"][0]["severity"] = None
    invalid_cases.append(missing_severity)

    unauthorized_severity = case_payload()
    unauthorized_severity["review_truth"]["expected_findings"][0][
        "metric_authority"
    ].update(severity_scorable=False, severity_authority=None)
    invalid_cases.append(unauthorized_severity)

    missing_severity_authority = case_payload()
    missing_severity_authority["review_truth"]["expected_findings"][0][
        "metric_authority"
    ]["severity_authority"] = None
    invalid_cases.append(missing_severity_authority)

    incomplete_location = case_payload()
    incomplete_location["review_truth"]["expected_findings"][0]["locations"][0].update(
        from_line=None, to_line=None
    )
    invalid_cases.append(incomplete_location)

    unauthorized_location = case_payload()
    unauthorized_location["review_truth"]["expected_findings"][0][
        "metric_authority"
    ]["location_scorable"] = False
    invalid_cases.append(unauthorized_location)

    for payload in invalid_cases:
        with pytest.raises(SchemaError):
            EvalCase.from_dict(payload)


def test_evaluator_context_hash_provenance_and_truth_references_are_strict() -> None:
    payload = case_payload()
    payload["review_evaluator_context"]["truth_contexts"] = [
        {
            "truth_id": "invalid-1",
            "allowed_tasks": ["finding_equivalence"],
            "sources": [evaluator_source_payload("known invalid context")],
        },
        {
            "truth_id": "issue-1",
            "allowed_tasks": ["finding_equivalence"],
            "sources": [evaluator_source_payload()],
        },
    ]

    case = EvalCase.from_dict(payload)

    assert [
        item.truth_id for item in case.review_evaluator_context.truth_contexts
    ] == ["invalid-1", "issue-1"]
    assert case.to_dict() == payload

    bad_hash = case_payload()
    bad_hash["review_evaluator_context"]["truth_contexts"] = [
        {
            "truth_id": "issue-1",
            "allowed_tasks": ["finding_equivalence"],
            "sources": [evaluator_source_payload()],
        }
    ]
    bad_hash["review_evaluator_context"]["truth_contexts"][0]["sources"][0][
        "content_sha256"
    ] = "0" * 64

    orphan = case_payload()
    orphan["review_evaluator_context"]["truth_contexts"] = [
        {
            "truth_id": "outside-case",
            "allowed_tasks": ["finding_equivalence"],
            "sources": [],
        }
    ]

    duplicate = case_payload()
    context = {
        "truth_id": "issue-1",
        "allowed_tasks": ["finding_equivalence"],
        "sources": [],
    }
    duplicate["review_evaluator_context"]["truth_contexts"] = [context, dict(context)]

    empty_tasks = case_payload()
    empty_tasks["review_evaluator_context"]["truth_contexts"] = [
        {"truth_id": "issue-1", "allowed_tasks": [], "sources": []}
    ]

    for invalid in (bad_hash, orphan, duplicate, empty_tasks):
        with pytest.raises(SchemaError):
            EvalCase.from_dict(invalid)


def test_eval_case_keeps_truth_and_clarification_private_from_agent_input() -> None:
    case = EvalCase.from_dict(case_payload())

    assert case.to_dict() == case_payload()
    agent_input = case.eval_input()
    assert isinstance(agent_input, EvalInput)
    assert agent_input.to_dict() == input_payload()
    rendered = canonical_json(agent_input)
    assert "answer-goal" not in rendered
    assert "intent_truth" not in rendered
    assert "review_truth" not in rendered
    assert isinstance(case.clarification_script.answers, tuple)
    assert isinstance(case.review_truth.expected_findings, tuple)


def test_eval_case_constructor_requires_the_canonical_private_input_type() -> None:
    case = EvalCase.from_dict(case_payload())

    with pytest.raises(SchemaError, match="EvalCaseInput"):
        EvalCase(
            schema_version=case.schema_version,
            task_id=case.task_id,
            case_version=case.case_version,
            source=case.source,
            input=case.eval_input(),  # type: ignore[arg-type]
            clarification_script=case.clarification_script,
            intent_truth=case.intent_truth,
            review_truth=case.review_truth,
            review_evaluator_context=case.review_evaluator_context,
        )


@pytest.mark.parametrize(
    ("status", "failure", "intent", "review"),
    [
        ("completed", None, intent_payload(), review_payload()),
        (
            "failed",
            {"code": "timeout", "message": "Timed out", "retryable": True},
            None,
            None,
        ),
        (
            "failed",
            {"code": "adapter_error", "message": "Adapter failed", "retryable": False},
            intent_payload(),
            review_payload(),
        ),
        (
            "failed",
            {
                "code": "harness_materialization_error",
                "message": "Target materialization failed",
                "retryable": False,
            },
            None,
            None,
        ),
        (
            "blocked",
            {"code": "agent_blocked", "message": "No credential", "retryable": False},
            None,
            None,
        ),
        (
            "blocked",
            {
                "code": "clarification_required",
                "message": "Waiting for clarification",
                "retryable": True,
            },
            intent_payload(clarification_questions=[exchange_payload(None)]),
            None,
        ),
        (
            "blocked",
            {
                "code": "clarification_required",
                "message": "Clarification was deferred",
                "retryable": True,
            },
            intent_payload(clarification_questions=[exchange_payload("defer")]),
            None,
        ),
        (
            "invalid_output",
            {"code": "invalid_json", "message": "Malformed output", "retryable": False},
            None,
            None,
        ),
    ],
)
def test_all_terminal_submission_forms_round_trip(
    status: str, failure: dict | None, intent: dict | None, review: dict | None
) -> None:
    payload = submission_payload()
    payload.update(status=status, failure=failure, intent=intent, review=review)

    submission = EvalSubmission.from_dict(payload)

    assert submission.to_dict() == payload
    assert isinstance(submission.evidence, tuple)
    if review is not None:
        assert submission.review is not None
        assert submission.review.findings == ()


@pytest.mark.parametrize(
    ("status", "failure", "intent", "review"),
    [
        (
            "completed",
            {"code": "unknown", "message": "Unexpected", "retryable": False},
            intent_payload(),
            review_payload(),
        ),
        ("completed", None, None, review_payload()),
        ("failed", None, None, None),
        (
            "failed",
            {"code": "invalid_json", "message": "Bad JSON", "retryable": False},
            None,
            None,
        ),
        (
            "failed",
            {
                "code": "harness_materialization_error",
                "message": "Target materialization failed",
                "retryable": False,
            },
            intent_payload(),
            None,
        ),
        (
            "blocked",
            {
                "code": "clarification_required",
                "message": "Need answer",
                "retryable": True,
            },
            None,
            None,
        ),
        (
            "blocked",
            {
                "code": "clarification_required",
                "message": "Need answer",
                "retryable": True,
            },
            intent_payload(clarification_questions=[exchange_payload("skip")]),
            None,
        ),
        (
            "invalid_output",
            {"code": "schema_mismatch", "message": "Bad schema", "retryable": False},
            intent_payload(),
            None,
        ),
        (
            "invalid_output",
            {"code": "timeout", "message": "Timed out", "retryable": True},
            None,
            None,
        ),
    ],
)
def test_terminal_submission_matrix_rejects_noncanonical_combinations(
    status: str, failure: dict | None, intent: dict | None, review: dict | None
) -> None:
    payload = submission_payload()
    payload.update(status=status, failure=failure, intent=intent, review=review)

    with pytest.raises(SchemaError):
        EvalSubmission.from_dict(payload)


@pytest.mark.parametrize("action", [None, "confirm", "correct", "reject", "skip", "defer"])
def test_submission_clarification_action_matrix_round_trips(action: str | None) -> None:
    exchange = SubmissionClarificationExchange.from_dict(exchange_payload(action))

    assert exchange.to_dict() == exchange_payload(action)
    assert isinstance(exchange.resolved_values, tuple)


@pytest.mark.parametrize(
    "mutation",
    [
        {"action": None, "matched_answer_id": "answer-goal"},
        {"action": None, "response": "guessed"},
        {"action": None, "resolved_values": ["guessed"]},
        {"action": "confirm", "matched_answer_id": None},
        {"action": "confirm", "resolved_values": []},
        {"action": "correct", "response": None},
        {"action": "correct", "resolved_values": []},
        {"action": "reject", "resolved_values": ["wrong"]},
        {"action": "skip", "resolved_values": ["wrong"]},
        {"action": "defer", "resolved_values": ["wrong"]},
    ],
)
def test_submission_clarification_action_matrix_rejects_guessed_or_impossible_values(
    mutation: dict,
) -> None:
    payload = exchange_payload(mutation.get("action"))
    payload.update(mutation)
    with pytest.raises(SchemaError):
        SubmissionClarificationExchange.from_dict(payload)


def test_clarification_transcript_requires_contiguous_turns_and_unique_question_ids() -> None:
    first = exchange_payload(None)
    gap = exchange_payload(None, turn_index=3, question_id="question-3")
    duplicate = exchange_payload(None, turn_index=2, question_id="question-1")

    for questions in ([first, gap], [first, duplicate]):
        payload = intent_payload(clarification_questions=questions)
        with pytest.raises(SchemaError):
            SubmissionIntent.from_dict(payload)


def test_typed_evidence_preserves_scorable_bad_revision_path_and_coordinates() -> None:
    evidence_items = []
    for index, kind in enumerate(EvidenceKind, start=1):
        evidence_items.append(
            {
                "evidence_id": "evidence-%d" % index,
                "source": evidence_source_payload(kind.value),
                "content_hash": EMPTY_HASH,
                "excerpt": "",
            }
        )
    payload = submission_payload()
    payload["evidence"] = evidence_items

    submission = EvalSubmission.from_dict(payload)

    assert {item.source.kind for item in submission.evidence} == set(EvidenceKind)
    repository_file = next(
        item.source
        for item in submission.evidence
        if item.source.kind is EvidenceKind.REPOSITORY_FILE
    )
    assert repository_file.revision == "HEAD"
    assert repository_file.path == "../not-authorized.py"
    assert repository_file.from_line == 9 and repository_file.to_line == 3
    assert submission.to_dict()["evidence"] == evidence_items


@pytest.mark.parametrize(
    "usage",
    [
        {
            "elapsed_seconds": 0,
            "input_tokens": 2,
            "output_tokens": 3,
            "total_tokens": 5,
            "tool_calls": 0,
            "cost_amount": 0.0,
            "cost_currency": "USD",
        },
        usage_payload(),
    ],
)
def test_usage_and_cost_valid_combinations_round_trip(usage: dict) -> None:
    payload = submission_payload()
    payload["usage"] = usage
    assert EvalSubmission.from_dict(payload).to_dict()["usage"] == usage


@pytest.mark.parametrize(
    "updates",
    [
        {"elapsed_seconds": -0.01},
        {"cost_amount": float("inf"), "cost_currency": "USD"},
        {"input_tokens": True},
        {"tool_calls": -1},
        {"input_tokens": 2, "output_tokens": 3, "total_tokens": 6},
        {"cost_amount": 1.2, "cost_currency": None},
        {"cost_amount": None, "cost_currency": "USD"},
        {"cost_amount": 1.2, "cost_currency": "usd"},
    ],
)
def test_usage_and_cost_cross_field_rules_fail_closed(updates: dict) -> None:
    payload = submission_payload()
    payload["usage"].update(updates)
    with pytest.raises(SchemaError):
        EvalSubmission.from_dict(payload)


def test_intent_claim_source_is_preserved_and_not_promoted_during_hydration() -> None:
    submission = EvalSubmission.from_dict(submission_payload())

    assert submission.intent is not None
    assert submission.intent.claims[0].source is IntentClaimSource.INFERRED
    assert submission.to_dict()["intent"]["claims"][0]["source"] == "inferred"


def test_unscorable_intent_has_one_canonical_null_and_empty_representation() -> None:
    payload = case_payload()
    payload["intent_truth"] = {
        "scorable": False,
        "authority": None,
        "expected_claims": [],
        "forbidden_claims": [],
        "clarification_policy": None,
    }
    case = EvalCase.from_dict(payload)
    assert case.intent_truth.scorable is False
    assert case.to_dict() == payload

    for field, value in [
        ("authority", "synthetic"),
        ("expected_claims", case_payload()["intent_truth"]["expected_claims"]),
        ("forbidden_claims", case_payload()["intent_truth"]["forbidden_claims"]),
        ("clarification_policy", "optional"),
    ]:
        invalid = case_payload()
        invalid["intent_truth"] = dict(payload["intent_truth"])
        invalid["intent_truth"][field] = value
        with pytest.raises(SchemaError):
            EvalCase.from_dict(invalid)


@pytest.mark.parametrize("completeness", ["expert_augmented", "human_observed"])
def test_non_closed_world_truth_rejects_forbid_novel_policy(completeness: str) -> None:
    payload = case_payload()
    payload["review_truth"]["completeness"] = completeness
    payload["review_truth"]["novel_finding_policy"] = "forbid"
    with pytest.raises(SchemaError):
        EvalCase.from_dict(payload)


def test_closed_world_truth_accepts_both_novel_policies() -> None:
    for policy in (NovelFindingPolicy.VERIFY.value, NovelFindingPolicy.FORBID.value):
        payload = case_payload()
        payload["review_truth"]["novel_finding_policy"] = policy
        assert EvalCase.from_dict(payload).review_truth.novel_finding_policy.value == policy


def test_case_validation_rejects_unconsumed_or_mismatched_answer_references() -> None:
    case = EvalCase.from_dict(case_payload())
    valid_payload = submission_payload()
    valid_payload["intent"] = intent_payload(
        clarification_questions=[exchange_payload("confirm")]
    )
    validate_submission_for_case(EvalSubmission.from_dict(valid_payload), case)

    dangling = submission_payload()
    exchange = exchange_payload("confirm")
    exchange["matched_answer_id"] = "answer-not-in-script"
    dangling["intent"] = intent_payload(clarification_questions=[exchange])
    with pytest.raises(SchemaError):
        validate_submission_for_case(EvalSubmission.from_dict(dangling), case)

    mismatch = submission_payload()
    exchange = exchange_payload("reject")
    mismatch["intent"] = intent_payload(clarification_questions=[exchange])
    with pytest.raises(SchemaError):
        validate_submission_for_case(EvalSubmission.from_dict(mismatch), case)


def test_case_validation_enforces_round_budget_and_recorded_answer_response() -> None:
    raw_case = case_payload()
    raw_case["clarification_script"]["max_rounds"] = 1
    case = EvalCase.from_dict(raw_case)

    preserved_unresolved = submission_payload()
    preserved_unresolved["intent"] = intent_payload(
        clarification_questions=[
            exchange_payload(None),
            exchange_payload(None, turn_index=2, question_id="question-2"),
        ]
    )
    validate_submission_for_case(
        EvalSubmission.from_dict(preserved_unresolved),
        case,
    )

    answered_over_budget = submission_payload()
    answered_over_budget["intent"] = intent_payload(
        clarification_questions=[
            exchange_payload(None),
            exchange_payload("confirm", turn_index=2, question_id="question-2"),
        ]
    )
    with pytest.raises(SchemaError, match="max_rounds"):
        validate_submission_for_case(
            EvalSubmission.from_dict(answered_over_budget),
            case,
        )

    mismatched_response = submission_payload()
    exchange = exchange_payload("confirm")
    exchange["response"] = "No"
    mismatched_response["intent"] = intent_payload(
        clarification_questions=[exchange]
    )
    with pytest.raises(SchemaError, match="response"):
        validate_submission_for_case(
            EvalSubmission.from_dict(mismatched_response),
            case,
        )


def test_case_validation_does_not_replace_semantic_material_claim_matching_with_text_equality() -> None:
    case = EvalCase.from_dict(case_payload())
    payload = submission_payload()
    exchange = exchange_payload("confirm")
    exchange["material_claim"] = "Administrative authorization must still be enforced"
    payload["intent"] = intent_payload(clarification_questions=[exchange])

    validate_submission_for_case(EvalSubmission.from_dict(payload), case)


def test_canonical_json_and_full_sha256_ids_are_stable_and_json_ready_only() -> None:
    value = {"z": "雪", "a": [1, True, None]}
    expected = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert canonical_json(value) == expected
    assert canonical_sha256(value) == hashlib.sha256(expected.encode("utf-8")).hexdigest()

    derived = stable_id("finding", "eval_submission_v2", {"claim": "x"})
    assert derived == (
        "finding-01f0a36e533b474486b4f7eab76f4db9"
        "10c7cb73be7599d69f1f4b182671fd9b"
    )
    v1_payload = {
        "namespace": "review_agent_eval.identity_v1",
        "kind": "finding",
        "identity": ["eval_submission_v2", {"claim": "x"}],
    }
    v1_vector = "finding-" + hashlib.sha256(
        json.dumps(
            v1_payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert v1_vector == (
        "finding-0e0ccba55dc4a108d45f4d944a20f3c0"
        "7df22fee0c45884e8206a52d929c609d"
    )
    assert re.fullmatch(r"finding-[0-9a-f]{64}", derived)
    assert derived != v1_vector
    assert derived != stable_id("finding", "eval_submission_v2", {"claim": "y"})

    with pytest.raises(SchemaError):
        canonical_json(Path("not-json"))
    with pytest.raises(SchemaError):
        canonical_json(b"not-json")
    with pytest.raises(SchemaError):
        canonical_json({"bad": float("nan")})


def test_enum_leaf_types_are_materialized_for_case_and_submission() -> None:
    case = EvalCase.from_dict(case_payload())
    submission = EvalSubmission.from_dict(submission_payload())

    assert case.source.origin is CaseOrigin.HAND_AUTHORED
    assert case.intent_truth.authority is IntentAuthority.LINKED_REQUIREMENT
    assert case.intent_truth.clarification_policy is ClarificationPolicy.REQUIRED
    assert case.review_truth.completeness is TruthCompleteness.CLOSED_WORLD
    assert case.review_truth.expected_findings[0].severity is FindingSeverity.HIGH
    assert (
        case.review_truth.expected_findings[0].required_context_level
        is RequiredContextLevel.FILE
    )
    assert submission.status is SubmissionStatus.COMPLETED
    assert submission.intent is not None
    assert submission.intent.status is IntentResult.SUFFICIENT
    assert submission.review is not None
    assert case.input.review_target.repository.source is RepositorySource.FIXTURE

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from review_agent_eval.models import (
    MAX_CLARIFICATION_QUESTIONS,
    MAX_EVAL_INPUT_BYTES,
    MAX_EVIDENCE_ITEMS,
    MAX_EVIDENCE_EXCERPT_BYTES,
    MAX_EVIDENCE_REFS,
    MAX_FINDINGS,
    MAX_INTENT_CLAIMS,
    MAX_TRUTH_FINDINGS,
    EvalCase,
    EvalInput,
    EvalSubmission,
    SchemaError,
    UnsupportedProtocolVersionError,
    canonical_json,
    load_eval_case,
    load_eval_input,
    load_eval_submission,
)


BASE = "a" * 40
HEAD = "b" * 40
TEXT = "ci ok"
TEXT_HASH = hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
INPUT_DIGEST = "c" * 64
MATERIALIZATION_ID = "materialization-001"


def input_payload() -> dict:
    return {
        "schema_version": "eval_input_v2",
        "task_id": "task-001",
        "review_target": {
            "kind": "repository",
            "repository": {
                "source": "fixture",
                "path": "fixtures/repo",
                "url": None,
                "base_revision": BASE,
                "head_revision": HEAD,
            },
            "review_request": {
                "title": "Review",
                "description": None,
                "user_intent": None,
                "review_focus": None,
                "linked_requirements": [],
                "project_rules": [],
                "existing_ci_evidence": [
                    {"source_id": "ci-1", "text": TEXT, "content_hash": TEXT_HASH}
                ],
            },
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


def intent_payload() -> dict:
    return {
        "status": "partial",
        "goal": None,
        "acceptance_criteria": [],
        "scope": [],
        "constraints": [],
        "claims": [],
        "clarification_questions": [],
        "uncertainties": [],
    }


def evidence_payload(evidence_id: str = "evidence-1") -> dict:
    return {
        "evidence_id": evidence_id,
        "source": {
            "kind": "repository_file",
            "target_materialization_id": MATERIALIZATION_ID,
            "revision": "HEAD",
            "path": "../unsafe-but-scorable.py",
            "from_line": 9,
            "to_line": 3,
        },
        "content_hash": "d" * 64,
        "excerpt": "evidence",
    }


def finding_payload(finding_id: str = "finding-1") -> dict:
    return {
        "finding_id": finding_id,
        "claim": "A material problem exists.",
        "severity": "medium",
        "path": None,
        "side": "right",
        "from_line": 7,
        "to_line": None,
        "evidence_refs": ["missing", "missing", "evidence-1"],
        "suggested_action": None,
    }


def submission_payload() -> dict:
    return {
        "schema_version": "eval_submission_v2",
        "task_id": "task-001",
        "agent_id": "agent-1",
        "trial_id": "trial-1",
        "eval_input_digest": INPUT_DIGEST,
        "target_materialization_id": MATERIALIZATION_ID,
        "status": "completed",
        "intent": intent_payload(),
        "review": {"findings": [], "uncertainties": []},
        "evidence": [],
        "usage": usage_payload(),
        "trace_ref": None,
        "failure": None,
    }


def answer_payload(answer_id: str = "answer-1") -> dict:
    return {
        "answer_id": answer_id,
        "dimension": "constraint",
        "material_claim": "Backward compatibility is required",
        "action": "confirm",
        "response": None,
        "corrected_values": [],
    }


def truth_location() -> dict:
    return {
        "path": "src/app.py",
        "side": None,
        "from_line": None,
        "to_line": None,
    }


def expected_finding(truth_id: str = "truth-1") -> dict:
    return {
        "truth_id": truth_id,
        "claim": "Authorization was weakened.",
        "severity": "high",
        "category": "security",
        "required": True,
        "metric_authority": {
            "severity_scorable": True,
            "severity_authority": "upstream_annotation",
            "location_scorable": False,
            "location_authority": None,
        },
        "locations": [truth_location()],
        "evidence_anchors": [
            {"fact": "The guard is absent on the head revision.", "locations": []}
        ],
        "required_context_level": "file",
        "rationale": "The endpoint is privileged.",
    }


def case_payload() -> dict:
    raw_input = input_payload()
    return {
        "schema_version": "eval_case_v2",
        "task_id": "task-001",
        "case_version": 1,
        "source": {
            "suite": "core",
            "origin": "private",
            "source_id": "source-1",
            "source_version": "v1",
            "source_uri": None,
            "license": None,
            "content_hash": "e" * 64,
        },
        "input": {"review_target": raw_input["review_target"]},
        "clarification_script": {"max_rounds": 1, "answers": [answer_payload()]},
        "intent_truth": {
            "scorable": True,
            "authority": "synthetic",
            "expected_claims": [],
            "forbidden_claims": [],
            "clarification_policy": "optional",
        },
        "review_truth": {
            "completeness": "closed_world",
            "novel_finding_policy": "verify",
            "expected_findings": [expected_finding()],
            "known_invalid_findings": [],
        },
        "review_evaluator_context": {"truth_contexts": []},
    }


@pytest.mark.parametrize(
    ("loader", "payload"),
    [
        (load_eval_input, input_payload()),
        (load_eval_submission, submission_payload()),
        (load_eval_case, case_payload()),
    ],
)
def test_bytes_and_text_loaders_are_strict_utf8_and_canonical_round_trip(
    loader, payload: dict
) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    from_text = loader(text)
    from_bytes = loader(text.encode("utf-8"))
    assert from_text == from_bytes
    assert from_text.to_dict() == payload
    assert from_text.to_json() == canonical_json(payload)
    assert len(from_text.digest()) == 64


@pytest.mark.parametrize("bad", [b"\xff", b"\x80{}", b"{\xc3(}"])
def test_bytes_loader_rejects_invalid_utf8(bad: bytes) -> None:
    with pytest.raises(SchemaError):
        load_eval_input(bad)


def test_input_loader_checks_raw_byte_limit_before_json_decode() -> None:
    oversized_invalid_json = b"[" + b" " * MAX_EVAL_INPUT_BYTES
    with pytest.raises(SchemaError, match="byte limit"):
        load_eval_input(oversized_invalid_json)


def test_recursive_duplicate_json_keys_are_rejected_before_hydration() -> None:
    duplicate_root = (
        '{"schema_version":"eval_input_v2","schema_version":"eval_input_v2"}'
    )
    duplicate_nested = json.dumps(input_payload(), separators=(",", ":")).replace(
        '"source":"fixture"', '"source":"fixture","source":"git"'
    )
    duplicate_deep = json.dumps(input_payload(), separators=(",", ":")).replace(
        '"source_id":"ci-1"', '"source_id":"ci-1","source_id":"ci-2"'
    )
    for raw in (duplicate_root, duplicate_nested, duplicate_deep):
        with pytest.raises(SchemaError, match="duplicate"):
            load_eval_input(raw)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonstandard_or_nonfinite_json_numbers_are_rejected(constant: str) -> None:
    raw = json.dumps(submission_payload(), separators=(",", ":"))
    raw = raw.replace('"elapsed_seconds":null', '"elapsed_seconds":%s' % constant)
    with pytest.raises(SchemaError):
        load_eval_submission(raw)


@pytest.mark.parametrize(
    ("loader", "version", "expected"),
    [
        (load_eval_submission, "eval_submission_v1", "eval_submission_v2"),
        (load_eval_case, "eval_case_v1", "eval_case_v2"),
    ],
)
def test_v1_submission_and_case_roots_are_stably_rejected_before_hydration(
    loader, version: str, expected: str
) -> None:
    with pytest.raises(UnsupportedProtocolVersionError) as exc_info:
        loader(json.dumps({"schema_version": version}))

    assert exc_info.value.code == "unsupported_protocol_version"
    assert exc_info.value.expected == expected
    assert exc_info.value.actual == version


def test_unknown_review_target_and_evidence_source_kinds_fail_closed() -> None:
    raw_input = input_payload()
    raw_input["review_target"]["kind"] = "workspace"
    with pytest.raises(SchemaError, match="unknown enum"):
        EvalInput.from_dict(raw_input)

    submission = submission_payload()
    evidence = evidence_payload()
    evidence["source"]["kind"] = "repository_search"
    submission["evidence"] = [evidence]
    with pytest.raises(SchemaError, match="unknown enum"):
        EvalSubmission.from_dict(submission)


def test_bool_cannot_impersonate_new_v2_target_or_source_integers() -> None:
    frozen = {
        "schema_version": "eval_input_v2",
        "task_id": "frozen-task",
        "review_target": {
            "kind": "frozen_context",
            "bundle_id": "bundle-1",
            "record_id": "record-1",
            "context_format": "rendered_text",
            "rendered_sha256": "1" * 64,
            "rendered_utf8_bytes": True,
            "source_binding_digest": "2" * 64,
        },
    }
    with pytest.raises(SchemaError, match="bool"):
        EvalInput.from_dict(frozen)

    submission = submission_payload()
    evidence = evidence_payload()
    evidence["source"]["from_line"] = True
    submission["evidence"] = [evidence]
    with pytest.raises(SchemaError, match="bool"):
        EvalSubmission.from_dict(submission)


@pytest.mark.parametrize(
    ("factory", "payload", "path", "value"),
    [
        (
            EvalInput.from_dict,
            input_payload(),
            ("review_target", "repository"),
            "extra",
        ),
        (EvalSubmission.from_dict, submission_payload(), ("usage",), "extra"),
        (EvalCase.from_dict, case_payload(), ("review_truth", "expected_findings", 0), "extra"),
    ],
)
def test_unknown_keys_at_every_layer_are_rejected(factory, payload, path, value) -> None:
    target = payload
    for item in path:
        target = target[item]
    target[value] = "unknown"
    with pytest.raises(SchemaError, match="unknown"):
        factory(payload)


@pytest.mark.parametrize(
    ("factory", "payload", "path", "field"),
    [
        (EvalInput.from_dict, input_payload(), (), "review_target"),
        (
            EvalInput.from_dict,
            input_payload(),
            ("review_target", "review_request"),
            "project_rules",
        ),
        (EvalSubmission.from_dict, submission_payload(), (), "usage"),
        (EvalSubmission.from_dict, submission_payload(), ("intent",), "uncertainties"),
        (EvalCase.from_dict, case_payload(), (), "clarification_script"),
        (EvalCase.from_dict, case_payload(), ("source",), "license"),
    ],
)
def test_missing_declared_fields_are_never_defaulted(factory, payload, path, field) -> None:
    target = payload
    for item in path:
        target = target[item]
    del target[field]
    with pytest.raises(SchemaError, match="missing"):
        factory(payload)


@pytest.mark.parametrize(
    ("factory", "payload", "path", "value"),
    [
        (EvalInput.from_dict, input_payload(), ("schema_version",), "eval_input_v3"),
        (EvalSubmission.from_dict, submission_payload(), ("status",), "running"),
        (EvalSubmission.from_dict, submission_payload(), ("intent", "status"), "unknown"),
        (EvalCase.from_dict, case_payload(), ("schema_version",), "eval_case_v3"),
        (EvalCase.from_dict, case_payload(), ("source", "origin"), "web"),
        (
            EvalCase.from_dict,
            case_payload(),
            ("review_truth", "completeness"),
            "partial",
        ),
    ],
)
def test_unknown_schema_versions_and_enums_are_rejected(factory, payload, path, value) -> None:
    target = payload
    for item in path[:-1]:
        target = target[item]
    target[path[-1]] = value
    with pytest.raises(SchemaError):
        factory(payload)


@pytest.mark.parametrize(
    ("payload", "path"),
    [
        (case_payload(), ("case_version",)),
        (case_payload(), ("clarification_script", "max_rounds")),
        (submission_payload(), ("usage", "tool_calls")),
    ],
)
def test_bool_cannot_impersonate_integer(payload, path) -> None:
    factory = EvalCase.from_dict if "case_version" in payload else EvalSubmission.from_dict
    target = payload
    for item in path[:-1]:
        target = target[item]
    target[path[-1]] = True
    with pytest.raises(SchemaError):
        factory(payload)


def test_bool_cannot_impersonate_line_or_turn_index() -> None:
    submission = submission_payload()
    submission["review"]["findings"] = [finding_payload()]
    submission["review"]["findings"][0]["from_line"] = True
    with pytest.raises(SchemaError):
        EvalSubmission.from_dict(submission)

    submission = submission_payload()
    question = {
        "turn_index": True,
        "question_id": "q-1",
        "dimension": "goal",
        "question": "What is required?",
        "material_claim": "Behavior is preserved",
        "matched_answer_id": None,
        "action": None,
        "response": None,
        "resolved_values": [],
    }
    submission["intent"]["clarification_questions"] = [question]
    with pytest.raises(SchemaError):
        EvalSubmission.from_dict(submission)


@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "g" * 64, "sha256:" + "a" * 64])
def test_digest_fields_require_full_lowercase_sha256_shape(digest: str) -> None:
    submission = submission_payload()
    submission["evidence"] = [evidence_payload()]
    submission["evidence"][0]["content_hash"] = digest
    with pytest.raises(SchemaError):
        EvalSubmission.from_dict(submission)

    case = case_payload()
    case["source"]["content_hash"] = digest
    with pytest.raises(SchemaError):
        EvalCase.from_dict(case)


def test_existing_ci_hash_must_match_exact_utf8_text() -> None:
    payload = input_payload()
    payload["review_target"]["review_request"]["existing_ci_evidence"][0][
        "content_hash"
    ] = "0" * 64
    with pytest.raises(SchemaError):
        EvalInput.from_dict(payload)


@pytest.mark.parametrize(
    ("source", "path", "url"),
    [
        ("fixture", None, None),
        ("fixture", "fixtures/repo", "https://example.test/repo.git"),
        ("git", None, None),
        ("git", "cache/repo", "https://example.test/repo.git"),
        ("git", None, "https://user:secret@example.test/repo.git"),
    ],
)
def test_repository_source_path_url_matrix_is_strict(source, path, url) -> None:
    payload = input_payload()
    payload["review_target"]["repository"].update(
        source=source, path=path, url=url
    )
    with pytest.raises(SchemaError):
        EvalInput.from_dict(payload)


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/repo",
        "C:/repo",
        "\\\\server\\share",
        "fixtures/../repo",
        "fixtures//repo",
        "fixtures/.git/objects",
        "fixtures\\repo",
    ],
)
def test_repository_paths_reject_escape_and_non_posix_forms(path: str) -> None:
    payload = input_payload()
    payload["review_target"]["repository"]["path"] = path
    with pytest.raises(SchemaError):
        EvalInput.from_dict(payload)


@pytest.mark.parametrize(
    ("base", "head"),
    [
        (BASE, BASE),
        ("a" * 39, HEAD),
        ("A" * 40, HEAD),
        ("a" * 64, "b" * 40),
        ("HEAD", HEAD),
    ],
)
def test_repository_revisions_are_distinct_same_length_full_object_ids(base, head) -> None:
    payload = input_payload()
    payload["review_target"]["repository"].update(
        base_revision=base, head_revision=head
    )
    with pytest.raises(SchemaError):
        EvalInput.from_dict(payload)


def test_duplicate_finding_and_evidence_object_ids_are_schema_errors() -> None:
    submission = submission_payload()
    submission["review"]["findings"] = [finding_payload(), finding_payload()]
    with pytest.raises(SchemaError, match="duplicate"):
        EvalSubmission.from_dict(submission)

    submission = submission_payload()
    submission["evidence"] = [evidence_payload(), evidence_payload()]
    with pytest.raises(SchemaError, match="duplicate"):
        EvalSubmission.from_dict(submission)


def test_duplicate_claim_question_ci_answer_and_truth_ids_are_schema_errors() -> None:
    submission = submission_payload()
    claim = {
        "claim_id": "claim-1",
        "dimension": "goal",
        "text": "Goal",
        "source": "explicit",
    }
    submission["intent"]["claims"] = [claim, copy.deepcopy(claim)]
    with pytest.raises(SchemaError, match="duplicate"):
        EvalSubmission.from_dict(submission)

    raw_input = input_payload()
    raw_input["review_target"]["review_request"]["existing_ci_evidence"].append(
        copy.deepcopy(
            raw_input["review_target"]["review_request"]["existing_ci_evidence"][0]
        )
    )
    with pytest.raises(SchemaError, match="duplicate"):
        EvalInput.from_dict(raw_input)

    case = case_payload()
    case["clarification_script"]["answers"].append(answer_payload())
    with pytest.raises(SchemaError, match="duplicate"):
        EvalCase.from_dict(case)

    case = case_payload()
    case["review_truth"]["known_invalid_findings"] = [
        {
            "truth_id": "truth-1",
            "claim": "Not true",
            "category": None,
            "locations": [],
            "rationale": "Known invalid",
        }
    ]
    with pytest.raises(SchemaError, match="duplicate"):
        EvalCase.from_dict(case)


def test_dangling_and_duplicate_evidence_refs_survive_hydration_in_order() -> None:
    submission = submission_payload()
    submission["review"]["findings"] = [finding_payload()]
    submission["evidence"] = [evidence_payload()]

    model = EvalSubmission.from_dict(submission)

    assert model.review is not None
    assert model.review.findings[0].evidence_refs == (
        "missing",
        "missing",
        "evidence-1",
    )
    assert model.to_dict()["review"]["findings"][0]["evidence_refs"] == [
        "missing",
        "missing",
        "evidence-1",
    ]


def test_semantically_duplicate_findings_are_not_set_deduplicated() -> None:
    submission = submission_payload()
    first = finding_payload("finding-b")
    second = finding_payload("finding-a")
    submission["review"]["findings"] = [first, second]

    model = EvalSubmission.from_dict(submission)

    assert model.review is not None
    assert [item.finding_id for item in model.review.findings] == ["finding-a", "finding-b"]
    assert len(model.review.findings) == 2


def test_id_addressed_collections_sort_stably_but_transcript_order_is_preserved() -> None:
    submission = submission_payload()
    submission["intent"]["claims"] = [
        {"claim_id": "z", "dimension": "goal", "text": "Same", "source": "explicit"},
        {"claim_id": "a", "dimension": "goal", "text": "Same", "source": "inferred"},
    ]
    submission["evidence"] = [evidence_payload("z"), evidence_payload("a")]
    model = EvalSubmission.from_dict(submission)
    assert model.intent is not None
    assert [item.claim_id for item in model.intent.claims] == ["a", "z"]
    assert [item.evidence_id for item in model.evidence] == ["a", "z"]

    case = case_payload()
    case["clarification_script"]["answers"] = [answer_payload("z"), answer_payload("a")]
    hydrated = EvalCase.from_dict(case)
    assert [item.answer_id for item in hydrated.clarification_script.answers] == ["a", "z"]


@pytest.mark.parametrize(
    ("path", "lines"),
    [
        ("../escape.py", (None, None)),
        ("/absolute.py", (None, None)),
        ("src\\app.py", (None, None)),
        ("src/.git/config", (None, None)),
        ("src/app.py", (1, None)),
        ("src/app.py", (None, 2)),
        ("src/app.py", (3, 2)),
        ("src/app.py", (0, 1)),
    ],
)
def test_truth_locations_are_strict_even_though_submission_locations_are_scorable(path, lines) -> None:
    case = case_payload()
    location = case["review_truth"]["expected_findings"][0]["locations"][0]
    location.update(path=path, from_line=lines[0], to_line=lines[1])
    with pytest.raises(SchemaError):
        EvalCase.from_dict(case)


def test_submission_locations_and_revisions_preserve_semantic_errors_but_reject_controls() -> None:
    submission = submission_payload()
    submission["evidence"] = [evidence_payload()]
    submission["evidence"][0]["source"]["revision"] = "HEAD\x00evil"
    with pytest.raises(SchemaError):
        EvalSubmission.from_dict(submission)

    submission = submission_payload()
    submission["review"]["findings"] = [finding_payload()]
    submission["review"]["findings"][0]["path"] = "src/app.py\nforged"
    with pytest.raises(SchemaError):
        EvalSubmission.from_dict(submission)


def test_command_evidence_preserves_platform_exit_codes_for_the_checker() -> None:
    submission = submission_payload()
    evidence = evidence_payload()
    evidence["source"] = {
        "kind": "command_output",
        "target_materialization_id": MATERIALIZATION_ID,
        "command": ["tool"],
        "exit_code": 0xC0000005,
        "stream": "combined",
        "artifact_ref": "artifact-001",
    }
    submission["evidence"] = [evidence]

    hydrated = EvalSubmission.from_dict(submission)

    assert hydrated.evidence[0].source.exit_code == 0xC0000005


def test_evidence_anchor_requires_nonempty_fact_and_strict_locations() -> None:
    case = case_payload()
    anchor = case["review_truth"]["expected_findings"][0]["evidence_anchors"][0]
    anchor["fact"] = "   "
    with pytest.raises(SchemaError):
        EvalCase.from_dict(case)

    case = case_payload()
    anchor = case["review_truth"]["expected_findings"][0]["evidence_anchors"][0]
    anchor["locations"] = [{"path": "../escape", "side": None, "from_line": None, "to_line": None}]
    with pytest.raises(SchemaError):
        EvalCase.from_dict(case)


@pytest.mark.parametrize("action", ["confirm", "reject", "skip", "defer"])
def test_case_script_non_correct_actions_reject_corrected_values(action: str) -> None:
    case = case_payload()
    answer = case["clarification_script"]["answers"][0]
    answer.update(action=action, corrected_values=["invented"])
    with pytest.raises(SchemaError):
        EvalCase.from_dict(case)


def test_case_script_correct_requires_response_and_nonempty_corrected_values() -> None:
    for response, values in [(None, ["fixed"]), ("fixed", [])]:
        case = case_payload()
        answer = case["clarification_script"]["answers"][0]
        answer.update(action="correct", response=response, corrected_values=values)
        with pytest.raises(SchemaError):
            EvalCase.from_dict(case)


@pytest.mark.parametrize("max_rounds", [0, 17, -1, 1.5, True])
def test_case_script_round_limit_is_one_through_sixteen(max_rounds) -> None:
    case = case_payload()
    case["clarification_script"]["max_rounds"] = max_rounds
    with pytest.raises(SchemaError):
        EvalCase.from_dict(case)


def test_resource_limits_fail_before_sort_or_deduplication() -> None:
    submission = submission_payload()
    claim = {
        "claim_id": "same-id",
        "dimension": "goal",
        "text": "same claim",
        "source": "explicit",
    }
    submission["intent"]["claims"] = [claim] * (MAX_INTENT_CLAIMS + 1)
    with pytest.raises(SchemaError, match="item limit"):
        EvalSubmission.from_dict(submission)

    submission = submission_payload()
    submission["evidence"] = [evidence_payload("same-id")] * (MAX_EVIDENCE_ITEMS + 1)
    with pytest.raises(SchemaError, match="item limit"):
        EvalSubmission.from_dict(submission)


def test_finding_and_ref_count_limits_fail_closed() -> None:
    submission = submission_payload()
    submission["review"]["findings"] = [
        finding_payload("finding-%d" % index) for index in range(MAX_FINDINGS + 1)
    ]
    with pytest.raises(SchemaError, match="item limit"):
        EvalSubmission.from_dict(submission)

    submission = submission_payload()
    finding = finding_payload()
    finding["evidence_refs"] = ["same"] * (MAX_EVIDENCE_REFS + 1)
    submission["review"]["findings"] = [finding]
    with pytest.raises(SchemaError, match="item limit"):
        EvalSubmission.from_dict(submission)


def test_question_and_truth_finding_count_limits_fail_closed() -> None:
    submission = submission_payload()
    question = {
        "turn_index": 1,
        "question_id": "same",
        "dimension": "goal",
        "question": "Question?",
        "material_claim": "Claim",
        "matched_answer_id": None,
        "action": None,
        "response": None,
        "resolved_values": [],
    }
    submission["intent"]["clarification_questions"] = [question] * (
        MAX_CLARIFICATION_QUESTIONS + 1
    )
    with pytest.raises(SchemaError, match="item limit"):
        EvalSubmission.from_dict(submission)

    case = case_payload()
    case["review_truth"]["expected_findings"] = [
        expected_finding("truth-%d" % index) for index in range(MAX_TRUTH_FINDINGS + 1)
    ]
    with pytest.raises(SchemaError, match="item limit"):
        EvalCase.from_dict(case)


def test_intent_truth_claim_limit_is_combined_across_expected_and_forbidden() -> None:
    case = case_payload()
    case["intent_truth"]["expected_claims"] = [
        {
            "truth_id": "intent-%d" % index,
            "dimension": "goal",
            "text": "Expected goal %d" % index,
            "required": True,
        }
        for index in range(MAX_INTENT_CLAIMS)
    ]
    case["intent_truth"]["forbidden_claims"] = [
        {
            "truth_id": "forbidden-over-limit",
            "dimension": "goal",
            "text": "Forbidden goal",
            "rationale": "This extra item crosses the combined protocol limit.",
        }
    ]
    with pytest.raises(SchemaError, match="item limit"):
        EvalCase.from_dict(case)


def test_case_input_projection_must_also_fit_the_eval_input_byte_limit() -> None:
    case = case_payload()
    text = "x" * 32768
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    case["input"]["review_target"]["review_request"]["existing_ci_evidence"] = [
        {"source_id": "ci-%d" % index, "text": text, "content_hash": digest}
        for index in range(65)
    ]
    with pytest.raises(SchemaError, match="EvalInput.*byte limit"):
        EvalCase.from_dict(case)


def test_excerpt_utf8_byte_limit_not_character_count() -> None:
    submission = submission_payload()
    evidence = evidence_payload()
    evidence["excerpt"] = "雪" * (MAX_EVIDENCE_EXCERPT_BYTES // 3 + 1)
    submission["evidence"] = [evidence]
    with pytest.raises(SchemaError, match="byte limit"):
        EvalSubmission.from_dict(submission)


@pytest.mark.parametrize(
    ("factory", "payload", "path"),
    [
        (EvalSubmission.from_dict, submission_payload(), ("intent", "goal")),
        (EvalSubmission.from_dict, submission_payload(), ("review", "uncertainties")),
        (EvalCase.from_dict, case_payload(), ("review_truth", "expected_findings", 0, "claim")),
    ],
)
def test_overlong_bounded_text_is_rejected(factory, payload, path) -> None:
    target = payload
    for item in path[:-1]:
        target = target[item]
    if path[-1] == "uncertainties":
        target[path[-1]] = ["x" * 8193]
    else:
        target[path[-1]] = "x" * 8193
    with pytest.raises(SchemaError):
        factory(payload)


def test_empty_string_does_not_replace_null_for_nullable_leaf_fields() -> None:
    payload = input_payload()
    payload["review_target"]["review_request"]["description"] = ""
    with pytest.raises(SchemaError):
        EvalInput.from_dict(payload)

    submission = submission_payload()
    submission["intent"]["goal"] = ""
    with pytest.raises(SchemaError):
        EvalSubmission.from_dict(submission)

    case = case_payload()
    case["source"]["license"] = ""
    with pytest.raises(SchemaError):
        EvalCase.from_dict(case)

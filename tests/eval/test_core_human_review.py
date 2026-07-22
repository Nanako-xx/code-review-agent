from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPOSITORY_ROOT / "eval"
AUTHORING_ROOT = EVAL_ROOT / "authoring"
if str(AUTHORING_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTHORING_ROOT))

from core_human_review import (  # noqa: E402
    ADJUDICATION_SCHEMA_VERSION,
    ADJUDICATOR_ATTESTATION_KEYS,
    ADJUDICATOR_CHECKLIST_KEYS,
    APPROVAL_SCHEMA_VERSION,
    AUTHOR_CHECKLIST_KEYS,
    HumanReviewError,
    REVIEWER_ATTESTATION_KEYS,
    REVIEWER_CHECKLIST_KEYS,
    annotation_protocol_binding,
    export_blind_review_batch,
    fixture_manifest_from_mappings,
    import_approved_response,
    load_source_bound_ledger_record,
    make_packet,
    project_ledger_record,
    verify_blind_review_batch,
    verify_completed_response,
    verify_current_case_approval,
)
import core_human_review as human_review_module  # noqa: E402
from review_agent_eval.models import (  # noqa: E402
    EvalCase,
    RepositoryReviewTarget,
    canonical_json,
)


TASK_ID = "core-py-001"
REQUIRED_CLARIFICATION_TASK_IDS = (
    "core-py-002",
    "core-py-003",
    "core-py-010",
    "core-py-011",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _rewrite_record_with_outer_digest(path: Path, record: dict) -> None:
    core = {key: value for key, value in record.items() if key != "record_digest"}
    record["record_digest"] = hashlib.sha256(
        canonical_json(core).encode("utf-8")
    ).hexdigest()
    path.write_text(canonical_json(record), encoding="utf-8")


def _case(task_id: str = TASK_ID) -> EvalCase:
    return EvalCase.from_json(
        (EVAL_ROOT / "cases" / "core" / task_id / "case.json").read_bytes()
    )


def _annotation() -> dict:
    return _read_json(EVAL_ROOT / "cases" / "core" / TASK_ID / "annotation.json")


def _repository_target(case: EvalCase) -> RepositoryReviewTarget:
    target = case.input.review_target
    assert type(target) is RepositoryReviewTarget
    return target


def _batch(tmp_path: Path, task_ids: tuple[str, ...] = (TASK_ID,)) -> Path:
    output = tmp_path / "blind-batch"
    export_blind_review_batch(
        EVAL_ROOT,
        output,
        list(task_ids),
        "human-review-test-batch",
    )
    return output


def _completed_response(
    batch: Path,
    path: Path,
    *,
    reviewer_id: str = "human:reviewer-b",
    task_id: str = TASK_ID,
) -> dict:
    template = _read_json(batch / "cases" / task_id / "response-template.json")
    case = _case(task_id)
    template["reviewer"] = {
        "reviewer_id": reviewer_id,
        "started_at": "2026-07-19T01:00:00Z",
        "completed_at": "2026-07-19T01:30:00Z",
    }
    template["attestations"] = {
        key: True for key in sorted(REVIEWER_ATTESTATION_KEYS)
    }
    template["intent_truth"] = case.intent_truth.to_dict()
    author_script = case.clarification_script.to_dict()
    reviewer_answers = []
    exchanges = []
    for index, author_answer in enumerate(author_script["answers"], start=1):
        answer = dict(author_answer)
        answer["answer_id"] = f"reviewer-answer-{index:04d}"
        reviewer_answers.append(answer)
        exchanges.append(
            {
                "answer_id": answer["answer_id"],
                "question": f"Independent clarification question {index}.",
                "answer": answer["response"] or "The requested claim is not accepted.",
                "answered_at": "2026-07-19T01:15:00Z",
            }
        )
    template["clarification_decision"] = {
        "policy": case.intent_truth.clarification_policy.value,
        "max_rounds": author_script["max_rounds"],
        "answers": reviewer_answers,
        "rationale": "The request already fixes every material acceptance boundary.",
        "exchanges": exchanges,
    }
    template["review_truth"] = case.review_truth.to_dict()
    template["human_checklist"] = {
        key: True for key in sorted(REVIEWER_CHECKLIST_KEYS)
    }
    template["reviewer_notes"] = "Independent review completed from the packet only."
    _write_json(path, template)
    return template


def _approval(verified, path: Path, *, author_id: str = "human:author-a", adjudicator: bool = False) -> dict:
    payload = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "binding": {
            "task_id": verified.case.task_id,
            "case_version": verified.case.case_version,
            "packet_digest": verified.packet["packet_digest"],
            "response_digest": verified.response_digest,
            "comparison_digest": verified.comparison["comparison_digest"],
        },
        "author_id": author_id,
        "signed_at": "2026-07-19T02:30:00Z" if adjudicator else "2026-07-19T02:00:00Z",
        "final_decision": "accepted",
        "leakage_review_completed": True,
        "author_checklist": {key: True for key in sorted(AUTHOR_CHECKLIST_KEYS)},
        "external_audit_evidence": {
            "author_reference": "audit://review-pr/author-a",
            "reviewer_reference": "audit://review-pr/reviewer-b",
            "adjudicator_reference": (
                "audit://review-pr/adjudicator-c" if adjudicator else None
            ),
        },
    }
    _write_json(path, payload)
    return payload


def _adjudication(verified, path: Path) -> dict:
    payload = {
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "binding": {
            "task_id": verified.case.task_id,
            "case_version": verified.case.case_version,
            "packet_digest": verified.packet["packet_digest"],
            "response_digest": verified.response_digest,
            "comparison_digest": verified.comparison["comparison_digest"],
        },
        "adjudicator_id": "human:adjudicator-c",
        "started_at": "2026-07-19T01:40:00Z",
        "completed_at": "2026-07-19T02:20:00Z",
        "decision": "accepted",
        "rationale": "Every strict field difference was reviewed against the fixed fixture; the current Author truth is retained.",
        "resolutions": [
            {
                "difference_id": item["difference_id"],
                "resolution": "author",
                "rationale": "The fixed acceptance contract supports the current Author field.",
            }
            for item in verified.comparison["differences"]
        ],
        "attestations": {
            key: True for key in sorted(ADJUDICATOR_ATTESTATION_KEYS)
        },
        "human_checklist": {
            key: True for key in sorted(ADJUDICATOR_CHECKLIST_KEYS)
        },
    }
    _write_json(path, payload)
    return payload


def _fixture_manifest() -> dict:
    case = _case()
    repository_path = _repository_target(case).repository.path
    assert repository_path is not None
    root = EVAL_ROOT / repository_path

    def files(side: str) -> dict[str, str]:
        return {
            path.relative_to(root / side).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted((root / side).rglob("*"))
            if path.is_file()
        }

    return fixture_manifest_from_mappings(files("base"), files("head"))


def test_exported_packet_is_blind_source_bound_and_replayable(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    manifest = verify_blind_review_batch(EVAL_ROOT, batch)
    assert manifest["batch_digest"]

    case = _case()
    packet_root = batch / "cases" / TASK_ID
    packet_text = (packet_root / "packet.json").read_text(encoding="utf-8")
    template_text = (packet_root / "response-template.json").read_text(encoding="utf-8")
    private_values = [
        case.source.suite,
        *(item.truth_id for item in case.intent_truth.expected_claims),
        *(item.truth_id for item in case.review_truth.expected_findings),
        *(item.rationale for item in case.review_truth.expected_findings),
    ]
    assert all(value not in packet_text + template_text for value in private_values)
    names = {path.name for path in packet_root.rglob("*") if path.is_file()}
    assert not ({"case.json", "annotation.json"} & names)
    assert not any("golden" in path.parts for path in packet_root.rglob("*"))

    packet = _read_json(packet_root / "packet.json")
    assert packet["eval_input"]["schema_version"] == "eval_input_v2"
    assert set(packet["eval_input"]) == {
        "schema_version",
        "task_id",
        "review_target",
    }
    assert set(packet["eval_input"]["review_target"]) == {
        "kind",
        "repository",
        "review_request",
    }
    expected = make_packet(
        case,
        _annotation()["repository_binding"],
        _fixture_manifest(),
        annotation_protocol_binding(EVAL_ROOT),
    )
    assert packet == expected


def test_packet_verifier_rejects_a_self_consistent_legacy_eval_input(
    tmp_path: Path,
) -> None:
    batch = _batch(tmp_path)
    packet_path = batch / "cases" / TASK_ID / "packet.json"
    packet = _read_json(packet_path)
    target = packet["eval_input"].pop("review_target")
    packet["eval_input"]["schema_version"] = "eval_input_v1"
    packet["eval_input"]["repository"] = target["repository"]
    packet["eval_input"]["review_request"] = target["review_request"]
    core = {key: value for key, value in packet.items() if key != "packet_digest"}
    packet["packet_digest"] = hashlib.sha256(
        canonical_json(core).encode("utf-8")
    ).hexdigest()
    packet_path.write_text(canonical_json(packet), encoding="utf-8")

    with pytest.raises(HumanReviewError, match="sole v2 projection"):
        verify_blind_review_batch(EVAL_ROOT, batch)


def test_export_rejects_traversal_overwrite_and_repository_output(tmp_path: Path) -> None:
    with pytest.raises(HumanReviewError):
        export_blind_review_batch(EVAL_ROOT, tmp_path / "traversal", ["../case"], "human-batch")
    output = _batch(tmp_path)
    with pytest.raises(HumanReviewError, match="overwrite"):
        export_blind_review_batch(EVAL_ROOT, output, [TASK_ID], "human-batch-two")
    with pytest.raises(HumanReviewError, match="outside"):
        export_blind_review_batch(
            EVAL_ROOT,
            REPOSITORY_ROOT / ".forbidden-human-packet",
            [TASK_ID],
            "human-batch-three",
        )


def test_export_rejects_symlink_or_reparse_ancestor(tmp_path: Path) -> None:
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "linked"
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this platform/account cannot create a directory symlink")
    with pytest.raises(HumanReviewError, match="symlink|reparse"):
        export_blind_review_batch(
            EVAL_ROOT, link / "packet", [TASK_ID], "human-symlink-batch"
        )


def test_containment_uses_canonical_existing_identity_for_short_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    alias_output = tmp_path / "REPOSI~1" / "new" / "packet"

    def canonical(path: Path) -> Path:
        if path == alias_output:
            return repository / "new" / "packet"
        if path == alias_output.parent:
            return repository / "new"
        return path

    monkeypatch.setattr(human_review_module, "_canonical_path_for_containment", canonical)

    assert human_review_module._is_within(alias_output, repository)


def test_outside_repository_read_rejects_injected_short_alias_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "packet.json"
    outside.write_bytes(b"{}")
    monkeypatch.setattr(human_review_module, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        human_review_module,
        "_canonical_path_for_containment",
        lambda path: repository / path.name if path == outside else path,
    )

    with pytest.raises(HumanReviewError, match="outside the repository"):
        human_review_module._read_regular(
            outside, "packet", outside_repository=True
        )


def test_export_rejects_injected_short_alias_before_any_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    alias_parent = tmp_path / "REPOSI~1"
    if not os.path.lexists(alias_parent):
        alias_parent.mkdir()
    sentinel = alias_parent / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    output = alias_parent / "packet"
    monkeypatch.setattr(human_review_module, "REPOSITORY_ROOT", repository)

    def canonical(path: Path) -> Path:
        if path == output:
            return repository / "packet"
        if path == alias_parent:
            return repository
        return path

    monkeypatch.setattr(human_review_module, "_canonical_path_for_containment", canonical)

    with pytest.raises(HumanReviewError, match="outside the repository"):
        export_blind_review_batch(
            EVAL_ROOT,
            output,
            [TASK_ID],
            "human-short-alias-batch",
        )

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert sorted(path.name for path in alias_parent.iterdir()) == ["sentinel.txt"]


@pytest.mark.parametrize(
    "component",
    (
        "COM\N{SUPERSCRIPT ONE}",
        "COM\N{SUPERSCRIPT TWO}.txt",
        "LPT\N{SUPERSCRIPT THREE}",
        "COM1 .txt",
        "COM2 .json",
        "NUL .json",
        "AUX .x",
        "LPT9 .data",
        "CONIN$",
        "conout$",
        "ConIn$.txt",
        "CONOUT$.log",
        "CONIN$ .txt",
        "cOnOuT$ .log",
    ),
)
def test_human_review_shared_path_policy_rejects_nfkc_device_names(
    component: str,
) -> None:
    with pytest.raises(HumanReviewError, match="Windows reserved device name"):
        human_review_module._safe_relative_parts(
            "repository/base/%s/input.py" % component,
            "fixture path",
        )


def test_human_review_path_policy_accepts_windows_reserved_near_miss_and_unicode() -> None:
    for component in ("COM10 .txt", "CONINX$.txt", "CONOUTER$.log", "普通话"):
        assert human_review_module._safe_relative_parts(
            "repository/base/%s/input.py" % component,
            "fixture path",
        )[-2] == component


@pytest.mark.skipif(os.name != "nt", reason="Windows GetShortPathNameW capability")
def test_windows_get_short_path_name_capability(tmp_path: Path) -> None:
    long_directory = tmp_path / "core-authoring-short-path-capability"
    long_directory.mkdir()

    short_path = human_review_module._windows_short_path_name(long_directory)

    assert short_path
    assert Path(short_path).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows DOS short-path containment")
def test_windows_short_path_alias_cannot_bypass_repository_containment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository-long-name"
    repository.mkdir()
    short_path = human_review_module._windows_short_path_name(repository)
    assert short_path is not None
    if os.path.normcase(short_path) == os.path.normcase(str(repository)):
        pytest.skip("this volume did not expose a distinct DOS short-path alias")
    output = Path(short_path) / "packet"
    monkeypatch.setattr(human_review_module, "REPOSITORY_ROOT", repository)

    with pytest.raises(HumanReviewError, match="outside the repository"):
        export_blind_review_batch(
            EVAL_ROOT,
            output,
            [TASK_ID],
            "human-real-short-alias-batch",
        )

    assert not os.path.lexists(output)


def test_packet_verifier_rejects_fixture_tampering_and_private_extra_file(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    fixture = next((batch / "cases" / TASK_ID / "repository" / "head").rglob("*.py"))
    fixture.write_bytes(fixture.read_bytes() + b"\n# tampered\n")
    with pytest.raises(HumanReviewError, match="fixture"):
        verify_blind_review_batch(EVAL_ROOT, batch)

    other = tmp_path / "other"
    shutil.copytree(batch, other)
    # Restore exact fixture, then inject a forbidden private artifact.
    source = EVAL_ROOT / "cases" / "core" / TASK_ID / "repository" / "head" / fixture.relative_to(
        batch / "cases" / TASK_ID / "repository" / "head"
    )
    copied_fixture = other / fixture.relative_to(batch)
    copied_fixture.write_bytes(source.read_bytes())
    (other / "cases" / TASK_ID / "case.json").write_bytes(
        (EVAL_ROOT / "cases" / "core" / TASK_ID / "case.json").read_bytes()
    )
    with pytest.raises(HumanReviewError, match="forbidden|unexpected"):
        verify_blind_review_batch(EVAL_ROOT, other)


def test_response_happy_path_hydrates_truth_and_imports_immutable_receipt(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    response_path = tmp_path / "response.json"
    _completed_response(batch, response_path)
    verified = verify_completed_response(EVAL_ROOT, batch, response_path)
    assert verified.comparison["material_disagreement"] is False

    approval_path = tmp_path / "approval.json"
    _approval(verified, approval_path)
    ledger = tmp_path / "private-ledger"
    record = import_approved_response(
        EVAL_ROOT, batch, response_path, approval_path, ledger
    )
    assert record["status"] == "approved"
    loaded = load_source_bound_ledger_record(
        ledger,
        _case(),
        _annotation()["repository_binding"],
        _fixture_manifest(),
        annotation_protocol_binding(EVAL_ROOT),
    )
    assert loaded == record
    projected = project_ledger_record(_annotation(), loaded)
    assert projected["human_review"]["status"] == "approved"
    assert projected["checklist"]["human_review_completed"] is True


@pytest.mark.parametrize("task_id", REQUIRED_CLARIFICATION_TASK_IDS)
def test_required_clarification_compares_full_semantics_with_independent_answer_ids(
    tmp_path: Path, task_id: str
) -> None:
    batch = _batch(tmp_path, (task_id,))
    response_path = tmp_path / f"{task_id}-response.json"
    response = _completed_response(batch, response_path, task_id=task_id)
    author_ids = {
        item.answer_id for item in _case(task_id).clarification_script.answers
    }
    reviewer_ids = {
        item["answer_id"] for item in response["clarification_decision"]["answers"]
    }
    assert reviewer_ids.isdisjoint(author_ids)
    verified = verify_completed_response(EVAL_ROOT, batch, response_path)
    assert verified.comparison["material_disagreement"] is False


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("response-14-to-99", "disagreement"),
        ("material-claim", "disagreement"),
        ("action", "schema"),
        ("max-rounds", "disagreement"),
    ),
)
def test_required_clarification_mutations_cannot_compare_equal(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    task_id = "core-py-003"
    batch = _batch(tmp_path, (task_id,))
    response_path = tmp_path / f"{mutation}.json"
    response = _completed_response(batch, response_path, task_id=task_id)
    answer = response["clarification_decision"]["answers"][0]
    if mutation == "response-14-to-99":
        answer["response"] = answer["response"].replace("14", "99")
        answer["corrected_values"] = [
            value.replace("14", "99") for value in answer["corrected_values"]
        ]
    elif mutation == "material-claim":
        answer["material_claim"] = "Set the default retention period to 99 days."
    elif mutation == "action":
        answer["action"] = "reject"
    elif mutation == "max-rounds":
        response["clarification_decision"]["max_rounds"] = 2
    _write_json(response_path, response)

    if expected == "schema":
        with pytest.raises(HumanReviewError, match="hydration|clarification"):
            verify_completed_response(EVAL_ROOT, batch, response_path)
    else:
        verified = verify_completed_response(EVAL_ROOT, batch, response_path)
        assert verified.comparison["material_disagreement"] is True


def test_clarification_exchange_must_bind_each_reviewer_answer_and_time_window(
    tmp_path: Path,
) -> None:
    task_id = "core-py-003"
    batch = _batch(tmp_path, (task_id,))
    response_path = tmp_path / "response.json"
    response = _completed_response(batch, response_path, task_id=task_id)
    response["clarification_decision"]["exchanges"][0]["answer_id"] = "reviewer-answer-9999"
    _write_json(response_path, response)
    with pytest.raises(HumanReviewError, match="does not bind"):
        verify_completed_response(EVAL_ROOT, batch, response_path)

    response = _completed_response(batch, response_path, task_id=task_id)
    response["clarification_decision"]["exchanges"][0]["answered_at"] = (
        "2026-07-19T00:59:59Z"
    )
    _write_json(response_path, response)
    with pytest.raises(HumanReviewError, match="review interval"):
        verify_completed_response(EVAL_ROOT, batch, response_path)


def test_not_required_clarification_rejects_answers_or_exchanges(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    response_path = tmp_path / "response.json"
    response = _completed_response(batch, response_path)
    response["clarification_decision"]["answers"] = [
        {
            "answer_id": "reviewer-answer-0001",
            "dimension": "goal",
            "material_claim": "A material claim.",
            "action": "reject",
            "response": "No.",
            "corrected_values": [],
        }
    ]
    _write_json(response_path, response)
    with pytest.raises(HumanReviewError, match="not_required"):
        verify_completed_response(EVAL_ROOT, batch, response_path)


def test_response_rejects_unknown_keys_stale_binding_and_fake_agent_identity(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    response_path = tmp_path / "response.json"
    response = _completed_response(batch, response_path)
    response["unknown"] = True
    _write_json(response_path, response)
    with pytest.raises(HumanReviewError, match="unknown"):
        verify_completed_response(EVAL_ROOT, batch, response_path)

    response.pop("unknown")
    response["binding"]["canonical_case_digest"] = "0" * 64
    _write_json(response_path, response)
    with pytest.raises(HumanReviewError, match="stale|binding"):
        verify_completed_response(EVAL_ROOT, batch, response_path)

    response = _completed_response(batch, response_path, reviewer_id="codex-agent-b")
    with pytest.raises(HumanReviewError, match="Agent/LLM"):
        verify_completed_response(EVAL_ROOT, batch, response_path)


def test_response_rejects_time_and_attestation_tampering(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    response_path = tmp_path / "response.json"
    response = _completed_response(batch, response_path)
    response["reviewer"]["completed_at"] = response["reviewer"]["started_at"]
    _write_json(response_path, response)
    with pytest.raises(HumanReviewError, match="after"):
        verify_completed_response(EVAL_ROOT, batch, response_path)
    response = _completed_response(batch, response_path)
    response["attestations"]["review_was_blind"] = False
    _write_json(response_path, response)
    with pytest.raises(HumanReviewError, match="must be true"):
        verify_completed_response(EVAL_ROOT, batch, response_path)

    response = _completed_response(batch, response_path)
    response["reviewer_notes"] = "\ud800"
    response_path.write_text(
        json.dumps(response, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(HumanReviewError, match="invalid Unicode"):
        verify_completed_response(EVAL_ROOT, batch, response_path)


def test_material_wording_disagreement_requires_complete_adjudication(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    response_path = tmp_path / "response.json"
    response = _completed_response(batch, response_path)
    response["review_truth"]["expected_findings"][0]["claim"] += " Different wording."
    _write_json(response_path, response)
    verified = verify_completed_response(EVAL_ROOT, batch, response_path)
    assert verified.comparison["material_disagreement"] is True
    approval_path = tmp_path / "approval.json"
    _approval(verified, approval_path, adjudicator=True)
    with pytest.raises(HumanReviewError, match="requires explicit"):
        import_approved_response(
            EVAL_ROOT,
            batch,
            response_path,
            approval_path,
            tmp_path / "ledger-without-c",
        )

    adjudication_path = tmp_path / "adjudication.json"
    _adjudication(verified, adjudication_path)
    record = import_approved_response(
        EVAL_ROOT,
        batch,
        response_path,
        approval_path,
        tmp_path / "ledger-with-c",
        adjudication_path,
    )
    assert record["adjudication"] is not None
    assert record["comparison"]["material_disagreement"] is True


def test_identity_collision_and_fake_adjudicator_are_rejected(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    response_path = tmp_path / "response.json"
    _completed_response(batch, response_path)
    verified = verify_completed_response(EVAL_ROOT, batch, response_path)
    approval_path = tmp_path / "approval.json"
    _approval(verified, approval_path, author_id="human:reviewer-b")
    with pytest.raises(HumanReviewError, match="different identities"):
        import_approved_response(
            EVAL_ROOT, batch, response_path, approval_path, tmp_path / "collision-ledger"
        )

    response = _completed_response(batch, response_path)
    response["review_truth"]["expected_findings"][0]["rationale"] += " disagreement"
    _write_json(response_path, response)
    verified = verify_completed_response(EVAL_ROOT, batch, response_path)
    _approval(verified, approval_path, adjudicator=True)
    adjudication_path = tmp_path / "adjudication.json"
    adjudication = _adjudication(verified, adjudication_path)
    adjudication["adjudicator_id"] = "gpt-adjudicator-agent"
    _write_json(adjudication_path, adjudication)
    with pytest.raises(HumanReviewError, match="Agent/LLM"):
        import_approved_response(
            EVAL_ROOT,
            batch,
            response_path,
            approval_path,
            tmp_path / "fake-c-ledger",
            adjudication_path,
        )


def test_ledger_rejects_overwrite_tampering_and_stale_case_digest(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    response_path = tmp_path / "response.json"
    _completed_response(batch, response_path)
    verified = verify_completed_response(EVAL_ROOT, batch, response_path)
    approval_path = tmp_path / "approval.json"
    _approval(verified, approval_path)
    ledger = tmp_path / "ledger"
    import_approved_response(EVAL_ROOT, batch, response_path, approval_path, ledger)
    with pytest.raises(HumanReviewError, match="overwrite"):
        import_approved_response(EVAL_ROOT, batch, response_path, approval_path, ledger)

    record_path = ledger / "records" / f"{TASK_ID}.json"
    raw = bytearray(record_path.read_bytes())
    raw[-10] = ord("0") if raw[-10] != ord("0") else ord("1")
    record_path.write_bytes(bytes(raw))
    with pytest.raises(HumanReviewError, match="digest|JSON"):
        load_source_bound_ledger_record(
            ledger,
            _case(),
            _annotation()["repository_binding"],
            _fixture_manifest(),
            annotation_protocol_binding(EVAL_ROOT),
        )

    # Recreate a syntactically valid but source-stale record with a self-consistent
    # outer digest; inner source binding must still reject it.
    clean_ledger = tmp_path / "clean-ledger"
    record = import_approved_response(
        EVAL_ROOT, batch, response_path, approval_path, clean_ledger
    )
    record["binding"]["canonical_case_digest"] = "0" * 64
    core = {key: value for key, value in record.items() if key != "record_digest"}
    record["record_digest"] = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()
    (clean_ledger / "records" / f"{TASK_ID}.json").write_text(
        canonical_json(record), encoding="utf-8"
    )
    with pytest.raises(HumanReviewError, match="requires independent re-review"):
        load_source_bound_ledger_record(
            clean_ledger,
            _case(),
            _annotation()["repository_binding"],
            _fixture_manifest(),
            annotation_protocol_binding(EVAL_ROOT),
        )


@pytest.mark.parametrize("tamper", ("zero-digest", "forged-id", "wrong-packet-reference"))
def test_ledger_replays_exact_batch_manifest_even_after_outer_digest_is_recomputed(
    tmp_path: Path, tamper: str
) -> None:
    batch = _batch(tmp_path)
    response_path = tmp_path / "response.json"
    _completed_response(batch, response_path)
    verified = verify_completed_response(EVAL_ROOT, batch, response_path)
    approval_path = tmp_path / "approval.json"
    _approval(verified, approval_path)
    ledger = tmp_path / f"ledger-{tamper}"
    record = import_approved_response(
        EVAL_ROOT, batch, response_path, approval_path, ledger
    )
    record_path = ledger / "records" / f"{TASK_ID}.json"
    if tamper == "zero-digest":
        record["batch_manifest"]["batch_digest"] = "0" * 64
    elif tamper == "forged-id":
        record["batch_manifest"]["batch_id"] = "human-forged-batch"
        batch_core = {
            key: value
            for key, value in record["batch_manifest"].items()
            if key != "batch_digest"
        }
        record["batch_manifest"]["batch_digest"] = hashlib.sha256(
            canonical_json(batch_core).encode("utf-8")
        ).hexdigest()
    else:
        record["batch_manifest"]["packets"][0]["canonical_case_digest"] = "0" * 64
        batch_core = {
            key: value
            for key, value in record["batch_manifest"].items()
            if key != "batch_digest"
        }
        record["batch_manifest"]["batch_digest"] = hashlib.sha256(
            canonical_json(batch_core).encode("utf-8")
        ).hexdigest()
    _rewrite_record_with_outer_digest(record_path, record)

    with pytest.raises(HumanReviewError, match="batch|binding|packet"):
        load_source_bound_ledger_record(
            ledger,
            _case(),
            _annotation()["repository_binding"],
            _fixture_manifest(),
            annotation_protocol_binding(EVAL_ROOT),
        )


def test_ledger_rejects_symlinked_records_root_before_publication(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    response_path = tmp_path / "response.json"
    _completed_response(batch, response_path)
    verified = verify_completed_response(EVAL_ROOT, batch, response_path)
    approval_path = tmp_path / "approval.json"
    _approval(verified, approval_path)
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    outside = tmp_path / "outside-records"
    outside.mkdir()
    try:
        (ledger / "records").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this platform/account cannot create a directory symlink")

    with pytest.raises(HumanReviewError, match="link|reparse"):
        import_approved_response(
            EVAL_ROOT,
            batch,
            response_path,
            approval_path,
            ledger,
        )
    assert list(outside.iterdir()) == []


def test_ledger_rejects_records_root_with_stable_reparse_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _batch(tmp_path)
    response_path = tmp_path / "response.json"
    _completed_response(batch, response_path)
    verified = verify_completed_response(EVAL_ROOT, batch, response_path)
    approval_path = tmp_path / "approval.json"
    _approval(verified, approval_path)
    ledger = tmp_path / "reparse-ledger"
    records = ledger / "records"
    records.mkdir(parents=True)
    real_lstat = Path.lstat

    def lstat_with_reparse(path: Path):
        metadata = real_lstat(path)
        if path == records:
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_dev=metadata.st_dev,
                st_ino=metadata.st_ino,
                st_nlink=metadata.st_nlink,
                st_size=metadata.st_size,
                st_file_attributes=0x400,
            )
        return metadata

    monkeypatch.setattr(Path, "lstat", lstat_with_reparse)
    with pytest.raises(HumanReviewError, match="link|reparse"):
        import_approved_response(
            EVAL_ROOT, batch, response_path, approval_path, ledger
        )


def test_ledger_rejects_preexisting_hardlinked_record_target(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    response_path = tmp_path / "response.json"
    _completed_response(batch, response_path)
    verified = verify_completed_response(EVAL_ROOT, batch, response_path)
    approval_path = tmp_path / "approval.json"
    _approval(verified, approval_path)
    ledger = tmp_path / "hardlinked-ledger"
    records = ledger / "records"
    records.mkdir(parents=True)
    outside = tmp_path / "outside-record.json"
    outside.write_text("outside", encoding="utf-8")
    try:
        os.link(outside, records / f"{TASK_ID}.json")
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem cannot create a hard link")

    with pytest.raises(HumanReviewError, match="overwrite|already exists"):
        import_approved_response(
            EVAL_ROOT, batch, response_path, approval_path, ledger
        )
    assert outside.read_text(encoding="utf-8") == "outside"


def test_response_reader_rejects_hardlinked_work_product(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    response_path = tmp_path / "response.json"
    _completed_response(batch, response_path)
    linked = tmp_path / "linked-response.json"
    try:
        os.link(response_path, linked)
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem cannot create a hard link")

    with pytest.raises(HumanReviewError, match="link|non-file"):
        verify_completed_response(EVAL_ROOT, batch, linked)


def test_direct_annotation_sidecar_edit_cannot_open_release_gate(tmp_path: Path) -> None:
    isolated_eval = tmp_path / "isolated-eval"
    (isolated_eval / "cases" / "core").mkdir(parents=True)
    shutil.copy2(EVAL_ROOT / "annotation-guidelines.md", isolated_eval / "annotation-guidelines.md")
    shutil.copytree(
        EVAL_ROOT / "cases" / "core" / TASK_ID,
        isolated_eval / "cases" / "core" / TASK_ID,
    )
    annotation_path = isolated_eval / "cases" / "core" / TASK_ID / "annotation.json"
    annotation = _read_json(annotation_path)
    annotation["human_review"].update(
        {
            "status": "approved",
            "final_decision": "accepted",
            "author_id": "human:author-a",
            "reviewer_id": "human:reviewer-b",
            "leakage_review_completed": True,
        }
    )
    annotation["checklist"] = {key: True for key in annotation["checklist"]}
    _write_json(annotation_path, annotation)
    with pytest.raises(HumanReviewError, match="no evaluator-private"):
        verify_current_case_approval(isolated_eval, TASK_ID)

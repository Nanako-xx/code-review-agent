from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, fields

import pytest

import review_agent_eval.cases as cases_module
from review_agent_eval.cases import (
    MAX_CASE_DIMENSIONS,
    MAX_SUITE_CASES,
    MAX_SUITE_TOTAL_CASE_BYTES,
    RUN_CASE_SNAPSHOT_SCHEMA_VERSION,
    SUITE_MANIFEST_SCHEMA_VERSION,
    AgentCaseView,
    CaseDimension,
    CaseSplit,
    RunCaseSnapshot,
    RunCaseSnapshotEntry,
    SuiteCase,
    SuiteKind,
    SuiteManifest,
    SuiteSource,
    is_windows_reserved_path_component,
    validate_case_for_manifest,
)
from review_agent_eval.models import (
    MAX_EVAL_CASE_BYTES,
    EvalCase,
    NovelFindingPolicy,
    ReviewTargetKind,
    SchemaError,
    UnsupportedProtocolVersionError,
    canonical_json,
    canonical_sha256,
)


BASE = "a" * 40
HEAD = "b" * 40
SOURCE_HASH = "c" * 64
FILE_HASH = "d" * 64


def case_payload(
    task_id: str = "task-001",
    *,
    source_origin: str = "hand_authored",
    source_uri: str | None = None,
    license_name: str | None = None,
    source_version: str = "source-v1",
    completeness: str = "closed_world",
    novel_policy: str = "forbid",
) -> dict:
    return {
        "schema_version": "eval_case_v2",
        "task_id": task_id,
        "case_version": 1,
        "source": {
            "suite": "core-suite",
            "origin": source_origin,
            "source_id": "source-%s" % task_id,
            "source_version": source_version,
            "source_uri": source_uri,
            "license": license_name,
            "content_hash": SOURCE_HASH,
        },
        "input": {
            "review_target": {
                "kind": "repository",
                "repository": {
                    "source": "fixture",
                    "path": "repositories/%s" % task_id,
                    "url": None,
                    "base_revision": BASE,
                    "head_revision": HEAD,
                },
                "review_request": {
                    "title": "Review authorization behavior",
                    "description": None,
                    "user_intent": "Keep authorization intact",
                    "review_focus": None,
                    "linked_requirements": ["REQ-1 keeps authorization intact"],
                    "project_rules": [],
                    "existing_ci_evidence": [],
                },
            },
        },
        "clarification_script": {
            "max_rounds": 2,
            "answers": [
                {
                    "answer_id": "answer-%s" % task_id,
                    "dimension": "constraint",
                    "material_claim": "Backward compatibility is required",
                    "action": "confirm",
                    "response": "Yes, preserve compatibility",
                    "corrected_values": [],
                }
            ],
        },
        "intent_truth": {
            "scorable": True,
            "authority": "linked_requirement",
            "expected_claims": [
                {
                    "truth_id": "intent-%s" % task_id,
                    "dimension": "goal",
                    "text": "Keep authorization intact",
                    "required": True,
                }
            ],
            "forbidden_claims": [
                {
                    "truth_id": "forbidden-%s" % task_id,
                    "dimension": "scope",
                    "text": "Rewrite unrelated billing code",
                    "rationale": "Billing is out of scope.",
                }
            ],
            "clarification_policy": "required",
        },
        "review_truth": {
            "completeness": completeness,
            "novel_finding_policy": novel_policy,
            "expected_findings": [],
            "known_invalid_findings": [],
        },
        "review_evaluator_context": {"truth_contexts": []},
    }


def source_payload(*, kind: str = "core") -> dict:
    public = kind == "public"
    return {
        "kind": kind,
        "source_id": "suite-source",
        "source_version": "source-v1",
        "source_uri": "https://example.test/dataset" if public else None,
        "license": "Apache-2.0" if public else None,
        "content_hash": "e" * 64,
        "preparation_binding": (
            {
                "schema_version": "public_suite_preparation_binding_v2",
                "source_catalog_digest": "3" * 64,
                "acquisition_receipt_digest": "4" * 64,
                "source_manifest_digest": "5" * 64,
                "filter_manifest_digest": "6" * 64,
                "preparation_packet_digest": "7" * 64,
                "repository_catalog_digest": "8" * 64,
                "frozen_bundle_trust_digest": None,
            }
            if public
            else None
        ),
    }


def repository_wire_contract_payload() -> dict:
    return {
        "case_schema_version": "eval_case_v2",
        "input_schema_version": "eval_input_v2",
        "submission_schema_version": "eval_submission_v2",
        "review_target_kind": "repository",
        "materializer_protocol": "repository-materializer-v2",
    }


def manifest_for_case(
    case: EvalCase,
    *,
    split: str = "regression",
    path: str | None = None,
    file_hash: str = FILE_HASH,
    kind: str = "core",
    protocol_id: str = "native_repository",
    dimensions: list[dict] | None = None,
) -> SuiteManifest:
    canonical_bytes = case.to_json().encode("utf-8")
    payload = {
        "schema_version": SUITE_MANIFEST_SCHEMA_VERSION,
        "suite_id": "core-suite",
        "suite_version": "suite-v1",
        "wire_contract": repository_wire_contract_payload(),
        "source": source_payload(kind=kind),
        "cases": [
            {
                "task_id": case.task_id,
                "case_version": case.case_version,
                "path": path or "cases/%s.json" % case.task_id,
                "split": split,
                "protocol_id": protocol_id,
                "dimensions": dimensions or [],
                "raw_file_size_bytes": len(canonical_bytes),
                "raw_file_sha256": file_hash,
                "canonical_case_digest": canonical_sha256(case),
                "eval_input_digest": case.eval_input().digest(),
                "truth_completeness": case.review_truth.completeness.value,
            }
        ],
    }
    return SuiteManifest.from_dict(payload)


def snapshot_for_case(case: EvalCase, **manifest_kwargs: object) -> RunCaseSnapshot:
    manifest = manifest_for_case(case, **manifest_kwargs)
    return RunCaseSnapshot.build(manifest, ((manifest.cases[0], case),))


def _manifest_payload_for_cases(cases: list[EvalCase]) -> dict:
    return {
        "schema_version": "suite_manifest_v2",
        "suite_id": "core-suite",
        "suite_version": "suite-v2",
        "wire_contract": repository_wire_contract_payload(),
        "source": source_payload(),
        "cases": [
            {
                "task_id": case.task_id,
                "case_version": case.case_version,
                "path": "cases/%s.json" % case.task_id,
                "split": "regression",
                "protocol_id": "report-partition-only",
                "dimensions": [],
                "raw_file_size_bytes": len(case.to_json().encode("utf-8")),
                "raw_file_sha256": hashlib.sha256(
                    case.to_json().encode("utf-8")
                ).hexdigest(),
                "canonical_case_digest": canonical_sha256(case),
                "eval_input_digest": case.eval_input().digest(),
                "truth_completeness": case.review_truth.completeness.value,
            }
            for case in cases
        ],
    }


def test_v2_manifest_rejects_v1_case_child() -> None:
    case = EvalCase.from_dict(case_payload())
    payload = _manifest_payload_for_cases([case])
    payload["wire_contract"]["case_schema_version"] = "eval_case_v1"
    payload["cases"][0]["raw_file_size_bytes"] = "malformed-but-deeper"

    with pytest.raises(UnsupportedProtocolVersionError):
        SuiteManifest.from_dict(payload)


def test_v2_snapshot_rejects_v1_children_before_nested_hydration() -> None:
    case = EvalCase.from_dict(case_payload())
    snapshot = snapshot_for_case(case)

    wire_child = snapshot.to_dict()
    wire_child["wire_contract"]["input_schema_version"] = "eval_input_v1"
    wire_child["manifest"]["cases"][0]["raw_file_size_bytes"] = "malformed"
    with pytest.raises(UnsupportedProtocolVersionError):
        RunCaseSnapshot.from_dict(wire_child)

    input_child = snapshot.to_dict()
    input_child["cases"][0]["input"]["schema_version"] = "eval_input_v1"
    input_child["cases"][0]["source"]["content_hash"] = "malformed"
    with pytest.raises(UnsupportedProtocolVersionError):
        RunCaseSnapshot.from_dict(input_child)


def test_snapshot_rejects_repository_and_frozen_case_mix() -> None:
    repository_case = EvalCase.from_dict(case_payload("task-repository"))
    frozen_payload = case_payload("task-frozen")
    frozen_payload["input"]["review_target"] = {
        "kind": "frozen_context",
        "bundle_id": "bundle-001",
        "record_id": "record-001",
        "context_format": "rendered_text",
        "rendered_sha256": "f" * 64,
        "rendered_utf8_bytes": 17,
        "source_binding_digest": "1" * 64,
    }
    frozen_case = EvalCase.from_dict(frozen_payload)

    with pytest.raises(SchemaError, match="single wire contract"):
        manifest = SuiteManifest.from_dict(
            _manifest_payload_for_cases([repository_case, frozen_case])
        )
        RunCaseSnapshot.build(
            manifest,
            tuple(zip(manifest.cases, (repository_case, frozen_case))),
        )


def test_suite_manifest_round_trips_canonical_metadata_and_is_immutable() -> None:
    first = EvalCase.from_dict(case_payload("task-b"))
    second = EvalCase.from_dict(case_payload("task-a"))
    raw = {
        "schema_version": SUITE_MANIFEST_SCHEMA_VERSION,
        "suite_id": "core-suite",
        "suite_version": "suite-v1",
        "wire_contract": repository_wire_contract_payload(),
        "source": source_payload(),
        "cases": [
            {
                "task_id": first.task_id,
                "case_version": 1,
                "path": "cases/task-b.json",
                "split": "capability",
                "protocol_id": "native_repository",
                "dimensions": [
                    {"name": "language", "value": "python"},
                    {"name": "benchmark.type", "value": "capability"},
                ],
                "raw_file_size_bytes": len(first.to_json().encode("utf-8")),
                "raw_file_sha256": "1" * 64,
                "canonical_case_digest": canonical_sha256(first),
                "eval_input_digest": first.eval_input().digest(),
                "truth_completeness": "closed_world",
            },
            {
                "task_id": second.task_id,
                "case_version": 1,
                "path": "cases/task-a.json",
                "split": "regression",
                "protocol_id": "native_repository",
                "dimensions": [{"name": "language", "value": "python"}],
                "raw_file_size_bytes": len(second.to_json().encode("utf-8")),
                "raw_file_sha256": "2" * 64,
                "canonical_case_digest": canonical_sha256(second),
                "eval_input_digest": second.eval_input().digest(),
                "truth_completeness": "closed_world",
            },
        ],
    }

    manifest = SuiteManifest.from_dict(raw)

    assert manifest.schema_version == "suite_manifest_v2"
    assert manifest.suite_id == "core-suite"
    assert manifest.suite_version == "suite-v1"
    assert manifest.source.source_id == "suite-source"
    assert manifest.source.content_hash == "e" * 64
    assert [item.task_id for item in manifest.cases] == ["task-a", "task-b"]
    assert [item.split for item in manifest.cases] == [
        CaseSplit.REGRESSION,
        CaseSplit.CAPABILITY,
    ]
    assert manifest.cases[0].protocol_id == "native_repository"
    assert manifest.cases[0].dimensions == (
        CaseDimension(name="language", value="python"),
    )
    assert SuiteManifest.from_json(manifest.to_json()) == manifest
    assert manifest.digest() == canonical_sha256(manifest.to_dict())
    with pytest.raises(FrozenInstanceError):
        manifest.suite_version = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.cases[0] = manifest.cases[1]  # type: ignore[index]


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda value: value.pop("suite_id"), "missing field"),
        (lambda value: value.update({"extra": True}), "unknown field"),
        (
            lambda value: value["cases"][0].update({"case_version": True}),
            "bool is not accepted",
        ),
        (
            lambda value: value["cases"][0].update(
                {"canonical_case_digest": "A" * 64}
            ),
            "lowercase SHA-256",
        ),
    ],
)
def test_suite_manifest_strictly_rejects_missing_unknown_and_invalid_fields(
    mutation, match: str
) -> None:
    case = EvalCase.from_dict(case_payload())
    raw = manifest_for_case(case).to_dict()
    mutation(raw)
    with pytest.raises(SchemaError, match=match):
        SuiteManifest.from_dict(raw)


def test_suite_manifest_json_reuses_strict_v2_json_rules() -> None:
    case = EvalCase.from_dict(case_payload())
    text = manifest_for_case(case).to_json()
    duplicated = text.replace(
        '"suite_id":"core-suite"',
        '"suite_id":"core-suite","suite_id":"other"',
    )
    with pytest.raises(SchemaError, match="duplicate object key"):
        SuiteManifest.from_json(duplicated)
    with pytest.raises(SchemaError, match="non-standard numeric constant"):
        SuiteManifest.from_json(text.replace('"suite-v1"', "NaN", 1))
    with pytest.raises(SchemaError):
        SuiteManifest.from_json(b"\xff")


def test_manifest_rejects_duplicate_ids_paths_and_portable_case_collisions() -> None:
    case = EvalCase.from_dict(case_payload())
    original = manifest_for_case(case).to_dict()

    for changed in (
        {"task_id": "task-001", "path": "cases/other.json"},
        {"task_id": "TASK-001", "path": "cases/other.json"},
        {"task_id": "task-002", "path": "cases/task-001.json"},
        {"task_id": "task-002", "path": "CASES/TASK-001.JSON"},
    ):
        raw = copy.deepcopy(original)
        duplicate = copy.deepcopy(raw["cases"][0])
        duplicate.update(changed)
        raw["cases"].append(duplicate)
        with pytest.raises(SchemaError, match="collision|duplicate"):
            SuiteManifest.from_dict(raw)


def test_manifest_collision_keys_use_unicode_normalization_before_casefold() -> None:
    case = EvalCase.from_dict(case_payload())
    original = manifest_for_case(case).to_dict()

    task_collision = copy.deepcopy(original)
    task_collision["cases"][0]["task_id"] = "caf\u00e9"
    second = copy.deepcopy(task_collision["cases"][0])
    second.update({"task_id": "CAFE\u0301", "path": "cases/second.json"})
    task_collision["cases"].append(second)
    with pytest.raises(SchemaError, match="task_id case collision"):
        SuiteManifest.from_dict(task_collision)

    path_collision = copy.deepcopy(original)
    path_collision["cases"][0]["path"] = "cases/caf\u00e9.json"
    second = copy.deepcopy(path_collision["cases"][0])
    second.update({"task_id": "task-002", "path": "CASES/CAFE\u0301.JSON"})
    path_collision["cases"].append(second)
    with pytest.raises(SchemaError, match="path case collision"):
        SuiteManifest.from_dict(path_collision)


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/case.json",
        "C:/absolute/case.json",
        "../escape.json",
        "cases/../escape.json",
        r"cases\escape.json",
        "cases//case.json",
        "cases/CON.json",
        "cases/COM1 .txt",
        "cases/COM2 .json",
        "cases/NUL .json",
        "cases/AUX .x",
        "cases/LPT9 .data",
        "cases/CONIN$.json",
        "cases/cOnOuT$ .log",
        "cases/COM¹.json",
        "cases/LPT³.txt",
        "cases/case.json.",
    ],
)
def test_manifest_rejects_unsafe_and_non_portable_case_paths(path: str) -> None:
    case = EvalCase.from_dict(case_payload())
    raw = manifest_for_case(case).to_dict()
    raw["cases"][0]["path"] = path
    with pytest.raises(SchemaError, match="path|unsafe|relative|portable"):
        SuiteManifest.from_dict(raw)


@pytest.mark.parametrize(
    ("component", "expected"),
    (
        ("COM1 .txt", True),
        ("COM2 .json", True),
        ("NUL .json", True),
        ("AUX .x", True),
        ("LPT9 .data", True),
        ("COM\N{SUPERSCRIPT ONE}", True),
        ("COM\N{SUPERSCRIPT TWO}", True),
        ("COM\N{SUPERSCRIPT THREE}", True),
        ("LPT\N{SUPERSCRIPT ONE}", True),
        ("LPT\N{SUPERSCRIPT TWO}", True),
        ("LPT\N{SUPERSCRIPT THREE}", True),
        ("CLOCK$ .txt", True),
        ("CONIN$", True),
        ("conout$", True),
        ("ConIn$.txt", True),
        ("CONOUT$.log", True),
        ("CONIN$ .txt", True),
        ("cOnOuT$ .log", True),
        ("COM10 .txt", False),
        ("CONINX$.txt", False),
        ("CONOUTER$.log", False),
        ("普通话.txt", False),
    ),
)
def test_windows_reserved_path_component_policy_handles_ignored_stem_suffixes(
    component: str,
    expected: bool,
) -> None:
    assert is_windows_reserved_path_component(component) is expected


def test_manifest_accepts_windows_reserved_near_miss_and_unicode_case_paths() -> None:
    case = EvalCase.from_dict(case_payload())
    for path in (
        "cases/COM10 .txt",
        "cases/CONINX$.txt",
        "cases/CONOUTER$.log",
        "cases/普通话.txt",
    ):
        manifest = manifest_for_case(case, path=path)
        assert manifest.cases[0].path == path


def test_case_dimensions_are_strict_generic_grouping_metadata() -> None:
    case = EvalCase.from_dict(case_payload())
    manifest = manifest_for_case(
        case,
        protocol_id="official_frozen_context",
        dimensions=[
            {"name": "language", "value": "python"},
            {"name": "difficulty", "value": "hard"},
            {"name": "benchmark.type", "value": "Type3 Latent"},
            {"name": "benchmark.config", "value": "C"},
        ],
    )

    assert manifest.cases[0].protocol_id == "official_frozen_context"
    assert [item.name for item in manifest.cases[0].dimensions] == [
        "benchmark.config",
        "benchmark.type",
        "difficulty",
        "language",
    ]

    duplicate = manifest.to_dict()
    duplicate["cases"][0]["dimensions"].append(
        {"name": "language", "value": "Python 3"}
    )
    with pytest.raises(SchemaError, match="duplicate grouping key"):
        SuiteManifest.from_dict(duplicate)

    for invalid_name in ("Language", "swe/type", ".difficulty", "语言"):
        invalid = manifest.to_dict()
        invalid["cases"][0]["dimensions"][0]["name"] = invalid_name
        with pytest.raises(SchemaError, match="dimension.name"):
            SuiteManifest.from_dict(invalid)


def test_agent_case_view_has_only_eval_input_in_its_type_and_serialization() -> None:
    case = EvalCase.from_dict(case_payload())
    view = AgentCaseView.from_case(case)

    assert [field.name for field in fields(AgentCaseView)] == ["input"]
    assert set(view.to_dict()) == {
        "schema_version",
        "task_id",
        "review_target",
    }
    assert view.to_dict() == case.eval_input().to_dict()
    assert AgentCaseView.from_json(view.to_json()) == view
    serialized = view.to_json()
    assert "truth" not in serialized
    assert "clarification" not in serialized
    assert "answer-" not in serialized
    assert not hasattr(view, "intent_truth")
    assert not hasattr(view, "review_truth")
    assert not hasattr(view, "clarification_script")
    with pytest.raises(FrozenInstanceError):
        view.input = case.eval_input()  # type: ignore[misc]


def test_case_binding_validates_intent_authority_claim_and_policy_combinations() -> None:
    linked = case_payload()
    linked["input"]["review_target"]["review_request"][
        "linked_requirements"
    ] = []
    linked_case = EvalCase.from_dict(linked)
    with pytest.raises(SchemaError, match="linked_requirement"):
        validate_case_for_manifest(linked_case, manifest_for_case(linked_case).cases[0], manifest_for_case(linked_case))

    unanswered = case_payload()
    unanswered["clarification_script"]["answers"] = []
    unanswered_case = EvalCase.from_dict(unanswered)
    with pytest.raises(SchemaError, match="required clarification"):
        validate_case_for_manifest(unanswered_case, manifest_for_case(unanswered_case).cases[0], manifest_for_case(unanswered_case))


def test_scorable_intent_allows_zero_required_claim_denominators() -> None:
    variants = []

    optional_only = case_payload()
    optional_only["intent_truth"]["expected_claims"][0]["required"] = False
    variants.append(optional_only)

    forbidden_only = case_payload()
    forbidden_only["intent_truth"]["expected_claims"] = []
    forbidden_only["intent_truth"]["clarification_policy"] = "not_required"
    variants.append(forbidden_only)

    clarification_only = case_payload()
    clarification_only["intent_truth"]["expected_claims"] = []
    clarification_only["intent_truth"]["forbidden_claims"] = []
    variants.append(clarification_only)

    for payload in variants:
        case = EvalCase.from_dict(payload)
        manifest = manifest_for_case(case)
        validate_case_for_manifest(case, manifest.cases[0], manifest)


def test_truth_completeness_and_novel_policy_are_pinned_and_fail_closed() -> None:
    human = case_payload(
        source_origin="private",
        completeness="human_observed",
        novel_policy="verify",
    )
    human["source"]["suite"] = "core-suite"
    case = EvalCase.from_dict(human)
    manifest = manifest_for_case(case, kind="private", split="held_out")
    validate_case_for_manifest(case, manifest.cases[0], manifest)
    assert case.review_truth.novel_finding_policy is NovelFindingPolicy.VERIFY

    forbidden = copy.deepcopy(human)
    forbidden["review_truth"]["novel_finding_policy"] = "forbid"
    with pytest.raises(SchemaError, match="only valid for closed_world"):
        EvalCase.from_dict(forbidden)

    mismatched = manifest.to_dict()
    mismatched["cases"][0]["truth_completeness"] = "closed_world"
    mismatched_manifest = SuiteManifest.from_dict(mismatched)
    with pytest.raises(SchemaError, match="truth completeness"):
        validate_case_for_manifest(case, mismatched_manifest.cases[0], mismatched_manifest)


def test_public_suite_source_metadata_is_mandatory_and_complete() -> None:
    for field_name in ("source_uri", "license"):
        raw = source_payload(kind="public")
        raw[field_name] = None
        with pytest.raises(SchemaError, match=field_name):
            SuiteSource.from_dict(raw)

    raw = source_payload(kind="public")
    raw.pop("content_hash")
    with pytest.raises(SchemaError, match="content_hash"):
        SuiteSource.from_dict(raw)


def test_public_preparation_binding_is_exact_and_target_specific() -> None:
    case = EvalCase.from_dict(case_payload())
    raw = manifest_for_case(case).to_dict()
    raw["source"] = source_payload(kind="public")
    manifest = SuiteManifest.from_dict(raw)
    assert manifest.source.preparation_binding is not None
    assert manifest.source.preparation_binding.repository_catalog_digest == "8" * 64

    missing = copy.deepcopy(raw)
    missing["source"]["preparation_binding"][
        "acquisition_receipt_digest"
    ] = None
    with pytest.raises(SchemaError, match="acquisition_receipt_digest"):
        SuiteManifest.from_dict(missing)

    unknown = copy.deepcopy(raw)
    unknown["source"]["preparation_binding"]["sidecar"] = "9" * 64
    with pytest.raises(SchemaError, match="unknown field"):
        SuiteManifest.from_dict(unknown)

    cross_kind = copy.deepcopy(raw)
    cross_kind["source"]["preparation_binding"][
        "frozen_bundle_trust_digest"
    ] = "9" * 64
    with pytest.raises(SchemaError, match="Repository Suite preparation"):
        SuiteManifest.from_dict(cross_kind)

    frozen = copy.deepcopy(raw)
    frozen["wire_contract"].update(
        {
            "review_target_kind": "frozen_context",
            "materializer_protocol": "frozen-context-materializer-v2",
        }
    )
    frozen["source"]["preparation_binding"]["repository_catalog_digest"] = None
    frozen["source"]["preparation_binding"][
        "frozen_bundle_trust_digest"
    ] = "9" * 64
    assert SuiteManifest.from_dict(frozen).wire_contract.review_target_kind is (
        ReviewTargetKind.FROZEN_CONTEXT
    )

    local = manifest_for_case(case).to_dict()
    local["source"]["preparation_binding"] = source_payload(kind="public")[
        "preparation_binding"
    ]
    with pytest.raises(SchemaError, match="preparation_binding=null"):
        SuiteManifest.from_dict(local)

    v1_binding = copy.deepcopy(raw)
    v1_binding["source"]["preparation_binding"]["schema_version"] = (
        "public_suite_preparation_binding_v1"
    )
    v1_binding["cases"][0]["raw_file_size_bytes"] = "malformed-but-deeper"
    with pytest.raises(UnsupportedProtocolVersionError):
        SuiteManifest.from_dict(v1_binding)


def test_run_case_snapshot_is_stable_self_contained_and_rejects_split_remap() -> None:
    case = EvalCase.from_dict(case_payload())
    snapshot = snapshot_for_case(case)

    assert snapshot.schema_version == RUN_CASE_SNAPSHOT_SCHEMA_VERSION
    assert snapshot.cases[0].input == case.eval_input()
    assert snapshot.cases[0].source == case.source
    assert snapshot.case(case.task_id) == snapshot.cases[0]
    assert snapshot.agent_view(case.task_id).to_dict() == case.eval_input().to_dict()
    assert RunCaseSnapshot.from_json(snapshot.to_json()) == snapshot
    assert snapshot.digest() == canonical_sha256(snapshot.to_dict())
    assert snapshot.snapshot_digest == snapshot.digest()
    assert snapshot.snapshot_id.startswith("run-case-snapshot-")
    serialized = snapshot.to_json()
    assert "intent_truth" not in serialized
    assert "review_truth" not in serialized
    assert "clarification_script" not in serialized
    assert '"answers"' not in serialized
    assert snapshot.cases[0].raw_file_sha256 == FILE_HASH
    assert snapshot.cases[0].canonical_case_digest == case.digest()
    assert snapshot.cases[0].case_source_provenance_hash == SOURCE_HASH
    assert set(snapshot.to_dict()) == {
        "schema_version",
        "snapshot_id",
        "manifest",
        "wire_contract",
        "cases",
    }
    assert set(snapshot.to_dict()["cases"][0]) == {
        "manifest_case",
        "source",
        "input",
    }
    with pytest.raises(FrozenInstanceError):
        snapshot.snapshot_id = "changed"  # type: ignore[misc]

    remapped = snapshot.to_dict()
    remapped["cases"][0]["manifest_case"]["split"] = "dev"
    with pytest.raises(SchemaError, match="manifest binding|split"):
        RunCaseSnapshot.from_dict(remapped)

    changed_id = snapshot.to_dict()
    changed_id["snapshot_id"] = "run-case-snapshot-" + "0" * 64
    with pytest.raises(SchemaError, match="canonical identity"):
        RunCaseSnapshot.from_dict(changed_id)

    changed_input = case.eval_input().to_dict()
    changed_input["review_target"]["review_request"][
        "title"
    ] = "Different but valid input"
    with pytest.raises(SchemaError, match="input digest"):
        RunCaseSnapshotEntry.from_dict(
            {
                "manifest_case": snapshot.cases[0].manifest_case.to_dict(),
                "source": snapshot.cases[0].source.to_dict(),
                "input": changed_input,
            }
        )


def test_snapshot_rejects_wire_and_source_cross_binding_drift() -> None:
    case = EvalCase.from_dict(case_payload())
    snapshot = snapshot_for_case(case)

    wire_drift = snapshot.to_dict()
    wire_drift["wire_contract"].update(
        {
            "review_target_kind": "frozen_context",
            "materializer_protocol": "frozen-context-materializer-v2",
        }
    )
    with pytest.raises(SchemaError, match="wire contract"):
        RunCaseSnapshot.from_dict(wire_drift)

    content_drift = snapshot.to_dict()
    content_drift["cases"][0]["source"]["content_hash"] = "0" * 64
    with pytest.raises(SchemaError, match="canonical identity"):
        RunCaseSnapshot.from_dict(content_drift)


def test_run_snapshot_order_and_id_are_independent_of_requested_case_order() -> None:
    case_a = EvalCase.from_dict(case_payload("task-a"))
    case_b = EvalCase.from_dict(case_payload("task-b"))
    manifest_dict = manifest_for_case(case_a).to_dict()
    manifest_dict["cases"].append(
        {
            "task_id": case_b.task_id,
            "case_version": 1,
            "path": "cases/task-b.json",
            "split": "dev",
            "protocol_id": "native_repository",
            "dimensions": [],
            "raw_file_size_bytes": len(case_b.to_json().encode("utf-8")),
            "raw_file_sha256": "f" * 64,
            "canonical_case_digest": canonical_sha256(case_b),
            "eval_input_digest": case_b.eval_input().digest(),
            "truth_completeness": "closed_world",
        }
    )
    manifest = SuiteManifest.from_dict(manifest_dict)
    entry_a = manifest.case("task-a")
    entry_b = manifest.case("task-b")

    left = RunCaseSnapshot.build(manifest, ((entry_b, case_b), (entry_a, case_a)))
    right = RunCaseSnapshot.build(manifest, ((entry_a, case_a), (entry_b, case_b)))

    assert left == right
    assert [item.task_id for item in left.cases] == ["task-a", "task-b"]
    assert left.snapshot_id == right.snapshot_id


def test_suite_and_snapshot_resource_limits_fail_before_unbounded_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = EvalCase.from_dict(case_payload())
    manifest = manifest_for_case(case)

    oversized_dimensions = manifest.to_dict()
    oversized_dimensions["cases"][0]["dimensions"] = [
        {"name": "group.%d" % index, "value": "x"}
        for index in range(MAX_CASE_DIMENSIONS + 1)
    ]
    with pytest.raises(SchemaError, match="item limit"):
        SuiteManifest.from_dict(oversized_dimensions)

    cumulative = manifest.to_dict()
    template = cumulative["cases"][0]
    cumulative["cases"] = []
    count = MAX_SUITE_TOTAL_CASE_BYTES // MAX_EVAL_CASE_BYTES + 1
    for index in range(count):
        item = copy.deepcopy(template)
        item["task_id"] = "task-%03d" % index
        item["path"] = "cases/task-%03d.json" % index
        item["raw_file_size_bytes"] = MAX_EVAL_CASE_BYTES
        cumulative["cases"].append(item)
    with pytest.raises(SchemaError, match="cumulative limit"):
        SuiteManifest.from_dict(cumulative)

    monkeypatch.setattr(cases_module, "MAX_SUITE_CASES", 3)
    consumed = []

    def bindings():
        for index in range(10):
            consumed.append(index)
            yield (manifest.cases[0], case)

    with pytest.raises(SchemaError, match="item limit"):
        RunCaseSnapshot.build(manifest, bindings())
    assert consumed == [0, 1, 2, 3]


def test_suite_manifest_byte_limit_precedes_child_and_manifest_sorting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = EvalCase.from_dict(case_payload())
    manifest = manifest_for_case(case)
    payload = manifest.to_dict()

    def forbidden_sort(*_args, **_kwargs):
        raise AssertionError("SuiteManifest sorted an oversized payload")

    monkeypatch.setattr(cases_module, "MAX_SUITE_MANIFEST_BYTES", 1)
    monkeypatch.setattr(cases_module, "sorted", forbidden_sort, raising=False)

    with pytest.raises(SchemaError, match="canonical byte limit"):
        SuiteManifest(
            schema_version=manifest.schema_version,
            suite_id=manifest.suite_id,
            suite_version=manifest.suite_version,
            wire_contract=manifest.wire_contract,
            source=manifest.source,
            cases=manifest.cases,
        )
    with pytest.raises(SchemaError, match="canonical byte limit"):
        SuiteManifest.from_dict(payload)


def test_snapshot_byte_limit_precedes_hydration_sort_hash_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = EvalCase.from_dict(case_payload())
    snapshot = snapshot_for_case(case)
    payload = snapshot.to_dict()

    def forbidden_sort(*_args, **_kwargs):
        raise AssertionError("RunCaseSnapshot sorted an oversized payload")

    def forbidden_hash(*_args, **_kwargs):
        raise AssertionError("RunCaseSnapshot hashed an oversized payload")

    def forbidden_identity(*_args, **_kwargs):
        raise AssertionError("RunCaseSnapshot derived an oversized identity")

    monkeypatch.setattr(cases_module, "MAX_RUN_CASE_SNAPSHOT_BYTES", 1)
    monkeypatch.setattr(cases_module, "sorted", forbidden_sort, raising=False)
    monkeypatch.setattr(cases_module, "canonical_sha256", forbidden_hash)
    monkeypatch.setattr(cases_module, "_snapshot_id", forbidden_identity)

    with pytest.raises(SchemaError, match="canonical byte limit"):
        RunCaseSnapshot(
            schema_version=snapshot.schema_version,
            snapshot_id=snapshot.snapshot_id,
            manifest=snapshot.manifest,
            wire_contract=snapshot.wire_contract,
            cases=snapshot.cases,
        )
    with pytest.raises(SchemaError, match="canonical byte limit"):
        RunCaseSnapshot.from_dict(payload)
    with pytest.raises(SchemaError, match="canonical byte limit"):
        RunCaseSnapshot.build(
            snapshot.manifest,
            ((snapshot.manifest.cases[0], case),),
        )


def test_manifest_canonical_json_matches_task1_json_configuration() -> None:
    case = EvalCase.from_dict(case_payload())
    manifest = manifest_for_case(case)

    expected = json.dumps(
        manifest.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert manifest.to_json() == expected == canonical_json(manifest)
    assert manifest.digest() == hashlib.sha256(expected.encode("utf-8")).hexdigest()

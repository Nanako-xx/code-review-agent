from __future__ import annotations

import copy
import hashlib
import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import review_agent_eval.datasets as datasets_module
from review_agent_eval.cases import (
    AgentCaseView,
    CaseHandle,
    CaseSplit,
    RunCaseSnapshot,
    SuiteManifest,
)
from review_agent_eval.datasets import CaseBank, CaseIntegrityError, DatasetError
from review_agent_eval.models import EvalCase, EvalInput, SchemaError, canonical_json_bytes


BASE = "a" * 40
HEAD = "b" * 40


def case_payload(
    task_id: str = "task-001",
    *,
    suite_id: str = "suite-1",
    origin: str = "hand_authored",
    source_version: str = "dataset-v1",
    source_uri: str | None = None,
    license_name: str | None = None,
    completeness: str = "closed_world",
    novel_policy: str = "forbid",
) -> dict:
    return {
        "schema_version": "eval_case_v1",
        "task_id": task_id,
        "case_version": 1,
        "source": {
            "suite": suite_id,
            "origin": origin,
            "source_id": "record-%s" % task_id,
            "source_version": source_version,
            "source_uri": source_uri,
            "license": license_name,
            "content_hash": "c" * 64,
        },
        "input": {
            "repository": {
                "source": "fixture",
                "path": "repositories/%s" % task_id,
                "url": None,
                "base_revision": BASE,
                "head_revision": HEAD,
            },
            "review_request": {
                "title": "Review %s" % task_id,
                "description": None,
                "user_intent": "Preserve behavior",
                "review_focus": None,
                "linked_requirements": ["REQ-1 preserve behavior"],
                "project_rules": [],
                "existing_ci_evidence": [],
            },
        },
        "clarification_script": {
            "max_rounds": 1,
            "answers": [
                {
                    "answer_id": "answer-%s" % task_id,
                    "dimension": "goal",
                    "material_claim": "Preserve behavior",
                    "action": "confirm",
                    "response": "Yes",
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
                    "text": "Preserve behavior",
                    "required": True,
                }
            ],
            "forbidden_claims": [],
            "clarification_policy": "required",
        },
        "review_truth": {
            "completeness": completeness,
            "novel_finding_policy": novel_policy,
            "expected_findings": [],
            "known_invalid_findings": [],
        },
    }


def write_suite(
    root: Path,
    case_payloads: list[dict] | None = None,
    *,
    kind: str = "core",
    splits: list[str] | None = None,
    case_paths: list[str] | None = None,
) -> tuple[Path, dict, list[EvalCase]]:
    payloads = case_payloads or [case_payload()]
    cases = [EvalCase.from_dict(payload) for payload in payloads]
    splits = splits or ["regression"] * len(cases)
    case_paths = case_paths or ["cases/%s.json" % case.task_id for case in cases]
    entries = []
    for case, split, relative in zip(cases, splits, case_paths):
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = canonical_json_bytes(case)
        path.write_bytes(raw)
        entries.append(
            {
                "task_id": case.task_id,
                "case_version": case.case_version,
                "path": relative,
                "split": split,
                "protocol_id": "native_repository",
                "dimensions": [{"name": "language", "value": "python"}],
                "raw_file_size_bytes": len(raw),
                "raw_file_sha256": hashlib.sha256(raw).hexdigest(),
                "canonical_case_digest": case.digest(),
                "eval_input_digest": case.eval_input().digest(),
                "truth_completeness": case.review_truth.completeness.value,
            }
        )

    public = kind == "public"
    manifest = {
        "schema_version": "suite_manifest_v1",
        "suite_id": "suite-1",
        "suite_version": "suite-v1",
        "source": {
            "kind": kind,
            "source_id": "dataset-source",
            "source_version": "dataset-v1",
            "source_uri": "https://example.test/dataset" if public else None,
            "license": "Apache-2.0" if public else None,
            "content_hash": "e" * 64,
        },
        "cases": entries,
    }
    manifest_path = root / "suite_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest_path, manifest, cases


def rewrite_manifest(path: Path, manifest: dict) -> None:
    path.write_bytes(canonical_json_bytes(manifest))


def test_case_bank_loads_verified_handles_and_keeps_agent_view_truth_free(
    tmp_path: Path,
) -> None:
    write_suite(tmp_path)

    bank = CaseBank.open(tmp_path)
    handle = bank.handle("task-001")

    assert isinstance(bank.manifest, SuiteManifest)
    assert isinstance(handle, CaseHandle)
    assert bank.suite_id == "suite-1"
    assert bank.suite_version == "suite-v1"
    assert bank.handles == (handle,)
    assert handle.split is CaseSplit.REGRESSION
    assert isinstance(bank.evaluator_case("task-001"), EvalCase)
    assert isinstance(bank.agent_input("task-001"), EvalInput)
    assert isinstance(bank.agent_view("task-001"), AgentCaseView)
    assert bank.agent_view("task-001").to_dict() == bank.agent_input(
        "task-001"
    ).to_dict()
    assert "truth" not in bank.agent_view("task-001").to_json()
    assert "clarification" not in bank.agent_view("task-001").to_json()
    assert "truth" not in handle.to_json()
    assert "clarification" not in handle.to_json()
    assert not hasattr(handle, "agent_input")
    assert not hasattr(handle, "agent_view")
    with pytest.raises(FrozenInstanceError):
        bank.manifest = bank.manifest  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        handle.entry = handle.entry  # type: ignore[misc]


def test_bank_open_fails_for_missing_case_and_loaded_task_id_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path, manifest, _ = write_suite(tmp_path)
    (tmp_path / "cases" / "task-001.json").unlink()
    with pytest.raises(DatasetError, match="missing|does not exist"):
        CaseBank.open(tmp_path)

    mismatch_root = tmp_path / "mismatch"
    manifest_path, manifest, _ = write_suite(mismatch_root)
    manifest["cases"][0]["task_id"] = "different-task"
    rewrite_manifest(manifest_path, manifest)
    with pytest.raises(CaseIntegrityError, match="task_id"):
        CaseBank.open(mismatch_root)


def test_bank_rejects_duplicate_or_case_colliding_task_ids_before_use(
    tmp_path: Path,
) -> None:
    manifest_path, manifest, _ = write_suite(tmp_path)
    duplicate = copy.deepcopy(manifest["cases"][0])
    duplicate["path"] = "cases/second.json"
    manifest["cases"].append(duplicate)
    rewrite_manifest(manifest_path, manifest)
    with pytest.raises(SchemaError, match="duplicate|collision"):
        CaseBank.open(tmp_path)

    collision_root = tmp_path / "collision"
    manifest_path, manifest, _ = write_suite(collision_root)
    duplicate = copy.deepcopy(manifest["cases"][0])
    duplicate.update({"task_id": "TASK-001", "path": "cases/second.json"})
    manifest["cases"].append(duplicate)
    rewrite_manifest(manifest_path, manifest)
    with pytest.raises(SchemaError, match="collision"):
        CaseBank.open(collision_root)


def test_each_case_load_revalidates_raw_file_hash_after_bank_open(tmp_path: Path) -> None:
    write_suite(tmp_path)
    bank = CaseBank.open(tmp_path)
    case_path = tmp_path / "cases" / "task-001.json"
    original = case_path.read_bytes()

    case_path.write_bytes(b"[" + original[1:])

    with pytest.raises(CaseIntegrityError, match="file hash"):
        bank.evaluator_case("task-001")
    with pytest.raises(CaseIntegrityError, match="file hash"):
        bank.agent_input("task-001")


def test_raw_file_sha_and_canonical_case_digest_are_distinct_bindings(
    tmp_path: Path,
) -> None:
    _manifest_path, manifest, cases = write_suite(tmp_path)
    canonical_raw = (tmp_path / "cases" / "task-001.json").read_bytes()
    assert manifest["cases"][0]["raw_file_sha256"] == cases[0].digest()
    assert hashlib.sha256(canonical_raw).hexdigest() == cases[0].digest()
    assert CaseBank.open(tmp_path).evaluator_case("task-001") == cases[0]

    noncanonical_root = tmp_path / "noncanonical"
    manifest_path, manifest, cases = write_suite(noncanonical_root)
    case_path = noncanonical_root / "cases" / "task-001.json"
    noncanonical_raw = b"\n  " + case_path.read_bytes() + b"\n"
    case_path.write_bytes(noncanonical_raw)
    manifest["cases"][0]["raw_file_size_bytes"] = len(noncanonical_raw)
    manifest["cases"][0]["raw_file_sha256"] = hashlib.sha256(
        noncanonical_raw
    ).hexdigest()
    rewrite_manifest(manifest_path, manifest)

    assert manifest["cases"][0]["raw_file_sha256"] != cases[0].digest()
    assert CaseBank.open(noncanonical_root).evaluator_case("task-001") == cases[0]


def test_each_case_load_revalidates_canonical_case_digest(tmp_path: Path) -> None:
    manifest_path, manifest, _ = write_suite(tmp_path)
    manifest["cases"][0]["canonical_case_digest"] = "0" * 64
    rewrite_manifest(manifest_path, manifest)

    with pytest.raises(CaseIntegrityError, match="canonical Case digest"):
        CaseBank.open(tmp_path)


def test_case_tamper_cannot_pass_by_updating_only_file_hash(tmp_path: Path) -> None:
    manifest_path, manifest, cases = write_suite(tmp_path)
    path = tmp_path / "cases" / "task-001.json"
    changed = cases[0].to_dict()
    changed["input"]["review_request"]["title"] = "Tampered title"
    raw = canonical_json_bytes(changed)
    path.write_bytes(raw)
    manifest["cases"][0]["raw_file_size_bytes"] = len(raw)
    manifest["cases"][0]["raw_file_sha256"] = hashlib.sha256(raw).hexdigest()
    rewrite_manifest(manifest_path, manifest)

    with pytest.raises(CaseIntegrityError, match="canonical Case digest"):
        CaseBank.open(tmp_path)


def test_case_size_binding_is_verified_before_case_acceptance(tmp_path: Path) -> None:
    manifest_path, manifest, _cases = write_suite(tmp_path)
    manifest["cases"][0]["raw_file_size_bytes"] += 1
    rewrite_manifest(manifest_path, manifest)

    with pytest.raises(CaseIntegrityError, match="file size"):
        CaseBank.open(tmp_path)


def test_existing_bank_revalidates_manifest_before_every_case_use(tmp_path: Path) -> None:
    manifest_path, manifest, _ = write_suite(tmp_path)
    bank = CaseBank.open(tmp_path)
    manifest["suite_version"] = "suite-v2"
    rewrite_manifest(manifest_path, manifest)

    with pytest.raises(CaseIntegrityError, match="manifest changed"):
        bank.evaluator_case("task-001")


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/outside.json",
        "C:/outside.json",
        "../outside.json",
        "cases/../../outside.json",
        r"cases\outside.json",
    ],
)
def test_case_bank_rejects_absolute_parent_and_non_posix_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    manifest_path, manifest, _ = write_suite(tmp_path)
    manifest["cases"][0]["path"] = unsafe_path
    rewrite_manifest(manifest_path, manifest)
    with pytest.raises(SchemaError, match="path|relative|unsafe"):
        CaseBank.open(tmp_path)


def test_case_bank_rejects_symlink_or_reparse_point_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path.parent / (tmp_path.name + "-outside-case.json")
    outside_case = EvalCase.from_dict(case_payload())
    raw = canonical_json_bytes(outside_case)
    outside.write_bytes(raw)
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True)
    linked = cases_dir / "linked.json"
    try:
        linked.symlink_to(outside)
    except (OSError, NotImplementedError):
        real_lstat = datasets_module.os.lstat
        linked_key = str(linked).casefold()

        class SymlinkMetadata:
            st_mode = datasets_module.stat.S_IFLNK
            st_dev = 0
            st_ino = 0
            st_file_attributes = 0

        def linked_lstat(path: str):
            if str(path).casefold() == linked_key:
                return SymlinkMetadata()
            return real_lstat(path)

        monkeypatch.setattr(datasets_module.os, "lstat", linked_lstat)

    manifest = {
        "schema_version": "suite_manifest_v1",
        "suite_id": "suite-1",
        "suite_version": "suite-v1",
        "source": {
            "kind": "core",
            "source_id": "dataset-source",
            "source_version": "dataset-v1",
            "source_uri": None,
            "license": None,
            "content_hash": "e" * 64,
        },
        "cases": [
            {
                "task_id": outside_case.task_id,
                "case_version": 1,
                "path": "cases/linked.json",
                "split": "regression",
                "protocol_id": "native_repository",
                "dimensions": [],
                "raw_file_size_bytes": len(raw),
                "raw_file_sha256": hashlib.sha256(raw).hexdigest(),
                "canonical_case_digest": outside_case.digest(),
                "eval_input_digest": outside_case.eval_input().digest(),
                "truth_completeness": "closed_world",
            }
        ],
    }
    (tmp_path / "suite_manifest.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(DatasetError, match="symlink|reparse"):
        CaseBank.open(tmp_path)


def test_case_bank_rejects_symlinked_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_root = tmp_path / "real"
    manifest_path, _, _ = write_suite(real_root)
    linked_root = tmp_path / "linked"
    linked_root.mkdir()
    linked = linked_root / "suite_manifest.json"
    try:
        linked.symlink_to(manifest_path)
    except (OSError, NotImplementedError):
        real_lstat = datasets_module.os.lstat
        linked_key = str(linked).casefold()

        class SymlinkMetadata:
            st_mode = datasets_module.stat.S_IFLNK
            st_dev = 0
            st_ino = 0
            st_file_attributes = 0

        def linked_lstat(path: str):
            if str(path).casefold() == linked_key:
                return SymlinkMetadata()
            return real_lstat(path)

        monkeypatch.setattr(datasets_module.os, "lstat", linked_lstat)

    with pytest.raises(DatasetError, match="symlink|reparse"):
        CaseBank.open(linked_root)


def test_windows_reparse_attribute_is_rejected_even_without_symlink_privilege(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_suite(tmp_path)
    real_lstat = datasets_module.os.lstat
    case_path = str(tmp_path / "cases" / "task-001.json").casefold()

    class ReparseMetadata:
        def __init__(self, original) -> None:
            self.st_mode = original.st_mode
            self.st_dev = original.st_dev
            self.st_ino = original.st_ino
            self.st_file_attributes = (
                getattr(original, "st_file_attributes", 0) | 0x0400
            )

    def marked_lstat(path: str):
        result = real_lstat(path)
        if str(path).casefold() == case_path:
            return ReparseMetadata(result)
        return result

    monkeypatch.setattr(datasets_module.os, "lstat", marked_lstat)
    with pytest.raises(DatasetError, match="reparse"):
        CaseBank.open(tmp_path)


def test_open_descriptor_identity_cannot_escape_suite_root_after_path_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_suite(tmp_path)
    calls = 0
    outside = tmp_path.parent / "outside-identical-case.json"
    outside.write_bytes((tmp_path / "cases" / "task-001.json").read_bytes())

    def escaped_after_manifest(_descriptor: int):
        nonlocal calls
        calls += 1
        return None if calls <= 2 else outside

    monkeypatch.setattr(
        datasets_module,
        "_windows_descriptor_path",
        escaped_after_manifest,
    )
    with pytest.raises(DatasetError, match="outside the Suite root"):
        CaseBank.open(tmp_path)


def test_public_case_source_metadata_fails_closed_even_with_valid_manifest(
    tmp_path: Path,
) -> None:
    public_case = case_payload(
        origin="aacr_bench",
        completeness="expert_augmented",
        novel_policy="verify",
    )
    write_suite(tmp_path, [public_case], kind="public")

    with pytest.raises(CaseIntegrityError, match="public.*source_uri|source_uri"):
        CaseBank.open(tmp_path)

    valid_root = tmp_path / "valid"
    public_case["source"]["source_uri"] = "https://example.test/dataset/task-001"
    public_case["source"]["license"] = "Apache-2.0"
    write_suite(valid_root, [public_case], kind="public")
    assert CaseBank.open(valid_root).evaluator_case("task-001").source.license == "Apache-2.0"


def test_public_manifest_missing_version_license_or_hash_fails_closed(
    tmp_path: Path,
) -> None:
    public_case = case_payload(
        origin="aacr_bench",
        source_uri="https://example.test/dataset/task-001",
        license_name="Apache-2.0",
        completeness="expert_augmented",
        novel_policy="verify",
    )
    manifest_path, manifest, _ = write_suite(tmp_path, [public_case], kind="public")

    for field_name, replacement in (
        ("source_version", None),
        ("license", None),
        ("content_hash", None),
    ):
        changed = copy.deepcopy(manifest)
        if replacement is None and field_name in ("source_version", "content_hash"):
            changed["source"].pop(field_name)
        else:
            changed["source"][field_name] = replacement
        rewrite_manifest(manifest_path, changed)
        with pytest.raises(SchemaError, match=field_name):
            CaseBank.open(tmp_path)


def test_fixed_splits_filter_without_remapping_and_snapshot_preserves_binding(
    tmp_path: Path,
) -> None:
    write_suite(
        tmp_path,
        [case_payload("task-a"), case_payload("task-b")],
        splits=["regression", "dev"],
    )
    bank = CaseBank.open(tmp_path)

    assert [item.task_id for item in bank.handles_for_split(CaseSplit.DEV)] == [
        "task-b"
    ]
    snapshot = bank.snapshot(split=CaseSplit.REGRESSION)
    assert isinstance(snapshot, RunCaseSnapshot)
    assert [item.task_id for item in snapshot.cases] == ["task-a"]
    assert snapshot.cases[0].manifest_case.split is CaseSplit.REGRESSION
    with pytest.raises(SchemaError, match="does not belong"):
        bank.snapshot(task_ids=("task-b",), split=CaseSplit.REGRESSION)


def test_run_snapshot_remains_immutable_after_source_case_changes(tmp_path: Path) -> None:
    write_suite(tmp_path)
    bank = CaseBank.open(tmp_path)
    snapshot = bank.snapshot()
    original_title = snapshot.case("task-001").input.review_request.title

    path = tmp_path / "cases" / "task-001.json"
    path.write_bytes(b"{}")

    assert snapshot.eval_input("task-001").review_request.title == original_title
    assert RunCaseSnapshot.from_json(snapshot.to_json()) == snapshot
    assert "intent_truth" not in snapshot.to_json()
    assert "review_truth" not in snapshot.to_json()
    assert "clarification_script" not in snapshot.to_json()
    assert '"answers"' not in snapshot.to_json()
    with pytest.raises(CaseIntegrityError, match="file (?:size|hash)"):
        bank.evaluator_case("task-001")


def test_core_dataset_loader_has_no_dataset_specific_branches() -> None:
    source = inspect.getsource(datasets_module).casefold()
    assert "aacr_bench" not in source
    assert "swe_prbench" not in source

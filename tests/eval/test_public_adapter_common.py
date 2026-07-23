from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

import review_agent_eval.adapters._public as public_module
import review_agent_eval.datasets as datasets_module
from review_agent_eval.adapters._public import (
    PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION,
    PUBLIC_PREPARATION_RECEIPT_SCHEMA_VERSION,
    PUBLIC_SOURCE_MANIFEST_SCHEMA_VERSION,
    PublicDatasetError,
    PublicFilterManifest,
    PublicPreparationError,
    PublicPreparationReceipt,
    PublicPreparedCase,
    PublicRecordReceipt,
    PublicSelector,
    PublicSourceFile,
    PublicSourceIntegrityError,
    PublicSourceManifest,
    PublicStatistic,
    VerifiedPublicSource,
    read_public_preparation_receipt,
    verify_public_source_manifest_digest,
    write_public_suite,
)
from review_agent_eval.cases import (
    REPOSITORY_MATERIALIZER_PROTOCOL,
    CaseDimension,
    CaseSplit,
    SuiteManifest,
    WireContractV2,
)
from review_agent_eval.datasets import CaseBank
from review_agent_eval.models import (
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    EvalCase,
    ReviewTargetKind,
    SchemaError,
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
)


BASE = "a" * 40
HEAD = "b" * 40
SOURCE_REVISION = "d" * 40
SOURCE_URI = "https://example.test/public-dataset"
LICENSE = "Apache-2.0"
SUITE_ID = "public-suite"


def _case_payload(task_id: str = "public-task-001") -> dict:
    return {
        "schema_version": EVAL_CASE_SCHEMA_VERSION,
        "task_id": task_id,
        "case_version": 1,
        "source": {
            "suite": SUITE_ID,
            "origin": "aacr_bench",
            "source_id": "record-%s" % task_id,
            "source_version": SOURCE_REVISION,
            "source_uri": SOURCE_URI,
            "license": LICENSE,
            "content_hash": "c" * 64,
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
                    "title": "Review %s" % task_id,
                    "description": None,
                    "user_intent": None,
                    "review_focus": None,
                    "linked_requirements": [],
                    "project_rules": [],
                    "existing_ci_evidence": [],
                },
            },
        },
        "clarification_script": {"max_rounds": 1, "answers": []},
        "intent_truth": {
            "scorable": False,
            "authority": None,
            "expected_claims": [],
            "forbidden_claims": [],
            "clarification_policy": None,
        },
        "review_truth": {
            "completeness": "expert_augmented",
            "novel_finding_policy": "verify",
            "expected_findings": [],
            "known_invalid_findings": [],
        },
        "review_evaluator_context": {"truth_contexts": []},
    }


def _repository_wire_contract() -> WireContractV2:
    return WireContractV2(
        case_schema_version=EVAL_CASE_SCHEMA_VERSION,
        input_schema_version=EVAL_INPUT_SCHEMA_VERSION,
        submission_schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
        review_target_kind=ReviewTargetKind.REPOSITORY,
        materializer_protocol=REPOSITORY_MATERIALIZER_PROTOCOL,
    )


def _source_manifest(source_root: Path) -> PublicSourceManifest:
    source_file = source_root / "upstream" / "records.json"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    raw = b'[{"record":1}]'
    source_file.write_bytes(raw)
    return PublicSourceManifest(
        schema_version=PUBLIC_SOURCE_MANIFEST_SCHEMA_VERSION,
        dataset_id="public-dataset",
        dataset_version="dataset-v1",
        source_uri=SOURCE_URI,
        source_revision=SOURCE_REVISION,
        license=LICENSE,
        files=(
            PublicSourceFile(
                role="records",
                path="upstream/records.json",
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            ),
        ),
        expected_statistics=(PublicStatistic("record_count", 1),),
    )


def _filter_manifest(*values: str) -> PublicFilterManifest:
    selectors = ()
    if values:
        selectors = (PublicSelector("language", tuple(values)),)
    return PublicFilterManifest(
        schema_version=PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION,
        dataset_id="public-dataset",
        selectors=selectors,
    )


def _prepared_case(task_id: str = "public-task-001") -> PublicPreparedCase:
    return PublicPreparedCase(
        case=EvalCase.from_dict(_case_payload(task_id)),
        split=CaseSplit.CAPABILITY,
        protocol_id="native_repository",
        dimensions=(CaseDimension("language", "Python"),),
    )


def _record(
    *,
    task_id: str = "public-task-001",
    value: int = 1,
) -> PublicRecordReceipt:
    return PublicRecordReceipt.from_record(
        task_id=task_id,
        truth_id=None,
        source_role="records",
        record_pointer="/0",
        upstream_id="upstream-1",
        record={"record": value},
        disposition="included",
    )


def _write_suite(
    tmp_path: Path,
    *,
    name: str = "suite",
    filter_manifest: PublicFilterManifest | None = None,
    statistics: tuple[PublicStatistic, ...] | None = None,
    records: tuple[PublicRecordReceipt, ...] | None = None,
    extra_files: dict[str, bytes] | None = None,
):
    source_root = tmp_path / (name + "-source")
    source_root.mkdir()
    source_manifest = _source_manifest(source_root)
    VerifiedPublicSource.open(source_root, source_manifest)
    result = write_public_suite(
        tmp_path / name,
        suite_id=SUITE_ID,
        suite_version="suite-v1",
        adapter_id="public-adapter",
        adapter_version="adapter-v1",
        source_manifest=source_manifest,
        filter_manifest=filter_manifest or _filter_manifest("Python"),
        wire_contract=_repository_wire_contract(),
        cases=(_prepared_case(),),
        actual_statistics=statistics
        or (PublicStatistic("case_count", 1),),
        records=records or (_record(),),
        extra_files=extra_files or {},
    )
    return source_root, source_manifest, result


@pytest.mark.parametrize("model", ["source", "filter"])
def test_manifests_reject_unknown_fields_and_duplicate_json_keys(
    tmp_path: Path, model: str
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = _source_manifest(source_root)
    manifest = source if model == "source" else _filter_manifest("Python")
    model_type = PublicSourceManifest if model == "source" else PublicFilterManifest

    payload = manifest.to_dict()
    payload["unexpected"] = True
    with pytest.raises(SchemaError, match="unknown field"):
        model_type.from_dict(payload)

    raw = manifest.to_json()
    duplicate_field = "dataset_id"
    duplicate = raw.replace(
        '"%s":"public-dataset"' % duplicate_field,
        '"%s":"public-dataset","%s":"shadow"'
        % (duplicate_field, duplicate_field),
        1,
    )
    with pytest.raises(SchemaError, match="duplicate object key"):
        model_type.from_json(duplicate)


def test_filter_rejects_explicit_empty_selector_values() -> None:
    with pytest.raises(PublicDatasetError, match="at least one|empty"):
        PublicSelector("language", ())

    with pytest.raises(PublicDatasetError, match="at least one|empty"):
        PublicFilterManifest.from_dict(
            {
                "schema_version": PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION,
                "dataset_id": "public-dataset",
                "selectors": [{"name": "language", "values": []}],
            }
        )


def test_source_manifest_rejects_portable_path_collisions() -> None:
    first = b"first"
    second = b"second"
    with pytest.raises(PublicDatasetError, match="portable|collision"):
        PublicSourceManifest(
            schema_version=PUBLIC_SOURCE_MANIFEST_SCHEMA_VERSION,
            dataset_id="public-dataset",
            dataset_version="dataset-v1",
            source_uri=SOURCE_URI,
            source_revision=SOURCE_REVISION,
            license=LICENSE,
            files=(
                PublicSourceFile(
                    "first",
                    "data/\u00e9.json",
                    len(first),
                    hashlib.sha256(first).hexdigest(),
                ),
                PublicSourceFile(
                    "second",
                    "DATA/e\u0301.json",
                    len(second),
                    hashlib.sha256(second).hexdigest(),
                ),
            ),
            expected_statistics=(),
        )


def test_verified_source_checks_every_file_size_and_sha256(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    manifest = _source_manifest(source_root)

    verified = VerifiedPublicSource.open(source_root, manifest)
    assert verified.read("records") == b'[{"record":1}]'

    source_path = source_root / "upstream" / "records.json"
    source_path.write_bytes(b'[{"record":2}]')
    with pytest.raises(PublicSourceIntegrityError, match="hash"):
        verified.read("records")

    source_path.write_bytes(b"short")
    with pytest.raises(PublicSourceIntegrityError, match="size"):
        verified.read("records")


def _create_hardlink_or_skip(source: Path, alias: Path) -> None:
    try:
        os.link(source, alias)
    except OSError as exc:  # pragma: no cover - filesystem capability
        pytest.skip("test filesystem does not support hard links: %s" % exc)
    assert source.stat().st_nlink > 1


def test_verified_source_rejects_hardlinked_source_file(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    manifest = _source_manifest(source_root)
    source_path = source_root / "upstream" / "records.json"
    _create_hardlink_or_skip(source_path, tmp_path / "source-alias.json")

    with pytest.raises(PublicSourceIntegrityError, match="hard link"):
        VerifiedPublicSource.open(source_root, manifest)


def test_control_plane_expected_source_manifest_digest_is_enforced(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = _source_manifest(source_root)
    expected = source.digest()
    mismatch = "0" * 64

    assert verify_public_source_manifest_digest(source, expected) == expected
    assert VerifiedPublicSource.open(
        source_root,
        source,
        expected_source_manifest_digest=expected,
    ).manifest == source
    with pytest.raises(PublicSourceIntegrityError, match="expected digest"):
        VerifiedPublicSource.open(
            source_root,
            source,
            expected_source_manifest_digest=mismatch,
        )

    with pytest.raises(PublicSourceIntegrityError, match="expected digest"):
        write_public_suite(
            tmp_path / "rejected-suite",
            suite_id=SUITE_ID,
            suite_version="suite-v1",
            adapter_id="public-adapter",
            adapter_version="adapter-v1",
            source_manifest=source,
            filter_manifest=_filter_manifest("Python"),
            wire_contract=_repository_wire_contract(),
            cases=(_prepared_case(),),
            actual_statistics=(PublicStatistic("case_count", 1),),
            records=(_record(),),
            expected_source_manifest_digest=mismatch,
        )
    assert not (tmp_path / "rejected-suite").exists()

    result = write_public_suite(
        tmp_path / "accepted-suite",
        suite_id=SUITE_ID,
        suite_version="suite-v1",
        adapter_id="public-adapter",
        adapter_version="adapter-v1",
        source_manifest=source,
        filter_manifest=_filter_manifest("Python"),
        wire_contract=_repository_wire_contract(),
        cases=(_prepared_case(),),
        actual_statistics=(PublicStatistic("case_count", 1),),
        records=(_record(),),
        expected_source_manifest_digest=expected,
    )
    assert read_public_preparation_receipt(
        result.suite_root,
        expected_source_manifest_digest=expected,
    ) == result.receipt
    with pytest.raises(PublicSourceIntegrityError, match="expected digest"):
        read_public_preparation_receipt(
            result.suite_root,
            expected_source_manifest_digest=mismatch,
        )


@pytest.mark.parametrize("unsafe_path", ["../outside.json", "/outside.json", "C:/x"])
def test_public_source_paths_fail_closed_on_escape(unsafe_path: str) -> None:
    with pytest.raises(SchemaError, match="path|relative|unsafe"):
        PublicSourceFile("records", unsafe_path, 1, "0" * 64)


@pytest.mark.parametrize("unsafe_kind", ["symlink", "reparse"])
def test_verified_source_rejects_symlink_and_reparse_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_kind: str,
) -> None:
    source_root = tmp_path / unsafe_kind
    source_root.mkdir()
    target = source_root / "records.json"
    raw = b"record"
    target.write_bytes(raw)
    manifest = PublicSourceManifest(
        schema_version=PUBLIC_SOURCE_MANIFEST_SCHEMA_VERSION,
        dataset_id="public-dataset",
        dataset_version="dataset-v1",
        source_uri=SOURCE_URI,
        source_revision=SOURCE_REVISION,
        license=LICENSE,
        files=(
            PublicSourceFile(
                "records", "records.json", len(raw), hashlib.sha256(raw).hexdigest()
            ),
        ),
        expected_statistics=(),
    )
    real_lstat = datasets_module.os.lstat
    target_key = str(target).casefold()

    class UnsafeMetadata:
        st_mode = (
            datasets_module.stat.S_IFLNK
            if unsafe_kind == "symlink"
            else datasets_module.stat.S_IFREG
        )
        st_dev = 0
        st_ino = 0
        st_size = len(raw)
        st_file_attributes = (
            0
            if unsafe_kind == "symlink"
            else datasets_module._REPARSE_POINT_FLAG
        )

    def unsafe_lstat(path: str):
        if str(path).casefold() == target_key:
            return UnsafeMetadata()
        return real_lstat(path)

    monkeypatch.setattr(datasets_module.os, "lstat", unsafe_lstat)
    with pytest.raises(PublicSourceIntegrityError, match="symlink|reparse"):
        VerifiedPublicSource.open(source_root, manifest)


def test_record_json_requires_canonical_bytes_and_matching_digest() -> None:
    record = _record()
    assert record.record_json == '{"record":1}'
    assert record.record_sha256 == hashlib.sha256(record.record_json.encode()).hexdigest()

    noncanonical = record.to_dict()
    noncanonical["record_json"] = '{ "record": 1 }'
    noncanonical["record_sha256"] = hashlib.sha256(
        noncanonical["record_json"].encode()
    ).hexdigest()
    with pytest.raises(PublicDatasetError, match="canonical JSON"):
        PublicRecordReceipt.from_dict(noncanonical)

    wrong_digest = record.to_dict()
    wrong_digest["record_sha256"] = "0" * 64
    with pytest.raises(PublicDatasetError, match="bind record_json"):
        PublicRecordReceipt.from_dict(wrong_digest)

    duplicate = record.to_dict()
    duplicate["record_json"] = '{"record":1,"record":2}'
    duplicate["record_sha256"] = hashlib.sha256(
        duplicate["record_json"].encode()
    ).hexdigest()
    with pytest.raises(SchemaError, match="duplicate object key"):
        PublicRecordReceipt.from_dict(duplicate)


def test_write_public_suite_round_trips_through_case_bank_and_packet_receipt(
    tmp_path: Path,
) -> None:
    _source_root, source, result = _write_suite(
        tmp_path,
        extra_files={"contexts/public-task-001.json": b"frozen context"},
    )

    bank = CaseBank.open(result.suite_root)
    loaded = read_public_preparation_receipt(result.suite_root)
    assert bank.evaluator_case("public-task-001") == _prepared_case().case
    assert loaded == result.receipt
    assert loaded.source_manifest_digest == source.digest()
    assert loaded.filter_manifest_digest == loaded.filter_manifest.digest()
    assert loaded.case_bindings_digest == canonical_sha256(
        [item.to_dict() for item in bank.manifest.cases]
    )
    assert loaded.preparation_packet_digest == bank.manifest.source.content_hash
    assert result.preparation_packet_digest == loaded.preparation_packet_digest
    assert result.suite_manifest_digest == result.manifest.digest()
    assert result.case_bindings_digest == loaded.case_bindings_digest
    assert bank.manifest.source.content_hash != source.digest()
    assert bank.manifest.wire_contract == _repository_wire_contract()
    preparation = bank.manifest.source.preparation_binding
    assert preparation is not None
    assert preparation.source_manifest_digest == source.digest()
    assert preparation.filter_manifest_digest == loaded.filter_manifest_digest
    assert preparation.preparation_packet_digest == loaded.preparation_packet_digest
    assert preparation.repository_catalog_digest is not None
    assert preparation.frozen_bundle_trust_digest is None
    assert loaded.extra_files[0].path == "contexts/public-task-001.json"
    assert loaded.extra_files[0].sha256 == hashlib.sha256(b"frozen context").hexdigest()


def _rewrite_receipt(root: Path, payload: dict) -> None:
    (root / "preparation_receipt.json").write_bytes(
        canonical_json(payload).encode("utf-8")
    )


@pytest.mark.parametrize(
    "component",
    ["adapter", "filter", "statistics", "records", "extra_files"],
)
def test_self_consistent_receipt_component_tamper_breaks_suite_packet_binding(
    tmp_path: Path, component: str
) -> None:
    _source_root, _source, result = _write_suite(
        tmp_path,
        name="suite-%s" % component,
        extra_files={"contexts/context.json": b"context"},
    )
    payload = result.receipt.to_dict()

    if component == "adapter":
        payload["adapter_version"] = "adapter-v2"
    elif component == "filter":
        payload["filter_manifest"]["selectors"][0]["values"] = ["Go"]
        payload["filter_manifest_digest"] = canonical_sha256(
            payload["filter_manifest"]
        )
    elif component == "statistics":
        payload["actual_statistics"][0]["value"] = 2
    elif component == "records":
        payload["records"][0]["record_json"] = '{"record":2}'
        payload["records"][0]["record_sha256"] = hashlib.sha256(
            payload["records"][0]["record_json"].encode()
        ).hexdigest()
        payload["records_digest"] = canonical_sha256(payload["records"])
    else:
        extra = result.suite_root / "contexts" / "context.json"
        extra.write_bytes(b"changed")
        payload["extra_files"][0]["size_bytes"] = len(b"changed")
        payload["extra_files"][0]["sha256"] = hashlib.sha256(b"changed").hexdigest()
        payload["extra_files_digest"] = canonical_sha256(payload["extra_files"])

    # The receipt remains internally valid after the attacker updates its local
    # component digest, but the immutable Suite packet hash must still reject it.
    assert PublicPreparationReceipt.from_dict(payload)
    _rewrite_receipt(result.suite_root, payload)
    with pytest.raises(
        PublicSourceIntegrityError, match="preparation (packet|binding)"
    ):
        read_public_preparation_receipt(result.suite_root)


def test_receipt_suite_manifest_digest_binds_manifest_in_reverse(tmp_path: Path) -> None:
    _source_root, _source, result = _write_suite(tmp_path)
    manifest_path = result.suite_root / "suite_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["suite_version"] = "suite-v2"
    manifest_path.write_bytes(canonical_json(payload).encode("utf-8"))

    with pytest.raises(PublicSourceIntegrityError, match="Suite manifest"):
        read_public_preparation_receipt(result.suite_root)


def test_case_bindings_digest_rejects_case_manifest_rewrite_without_packet_update(
    tmp_path: Path,
) -> None:
    _source_root, _source, result = _write_suite(tmp_path)
    entry = result.manifest.cases[0]
    case_path = result.suite_root / Path(*entry.path.split("/"))
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    case_payload["input"]["review_target"]["review_request"][
        "title"
    ] = "attacker-controlled"
    changed_case = EvalCase.from_dict(case_payload)
    changed_raw = canonical_json_bytes(changed_case.to_dict())
    case_path.write_bytes(changed_raw)

    manifest_path = result.suite_root / "suite_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = manifest_payload["cases"][0]
    binding["raw_file_size_bytes"] = len(changed_raw)
    binding["raw_file_sha256"] = hashlib.sha256(changed_raw).hexdigest()
    binding["canonical_case_digest"] = changed_case.digest()
    binding["eval_input_digest"] = changed_case.eval_input().digest()
    changed_manifest = SuiteManifest.from_dict(manifest_payload)
    manifest_path.write_bytes(canonical_json_bytes(changed_manifest.to_dict()))

    receipt_payload = result.receipt.to_dict()
    receipt_payload["suite_manifest_digest"] = changed_manifest.digest()
    assert PublicPreparationReceipt.from_dict(receipt_payload)
    _rewrite_receipt(result.suite_root, receipt_payload)

    with pytest.raises(PublicSourceIntegrityError, match="Case bindings"):
        read_public_preparation_receipt(result.suite_root)


def test_external_anchors_reject_fully_self_consistent_suite_rewrite(
    tmp_path: Path,
) -> None:
    _source_root, _source, result = _write_suite(tmp_path)
    old_packet_digest = result.preparation_packet_digest
    old_manifest_digest = result.suite_manifest_digest

    entry = result.manifest.cases[0]
    case_path = result.suite_root / Path(*entry.path.split("/"))
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    case_payload["input"]["review_target"]["review_request"][
        "title"
    ] = "attacker-controlled"
    changed_case = EvalCase.from_dict(case_payload)
    changed_raw = canonical_json_bytes(changed_case.to_dict())
    case_path.write_bytes(changed_raw)

    manifest_path = result.suite_root / "suite_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = manifest_payload["cases"][0]
    binding["raw_file_size_bytes"] = len(changed_raw)
    binding["raw_file_sha256"] = hashlib.sha256(changed_raw).hexdigest()
    binding["canonical_case_digest"] = changed_case.digest()
    binding["eval_input_digest"] = changed_case.eval_input().digest()

    receipt_payload = result.receipt.to_dict()
    receipt_payload["case_bindings_digest"] = canonical_sha256(
        manifest_payload["cases"]
    )
    provisional_receipt = PublicPreparationReceipt.from_dict(receipt_payload)
    new_packet_digest = provisional_receipt.preparation_packet_digest
    manifest_payload["source"]["content_hash"] = new_packet_digest
    manifest_payload["source"]["preparation_binding"][
        "preparation_packet_digest"
    ] = new_packet_digest
    changed_manifest = SuiteManifest.from_dict(manifest_payload)
    manifest_path.write_bytes(canonical_json_bytes(changed_manifest.to_dict()))

    receipt_payload["suite_manifest_digest"] = changed_manifest.digest()
    changed_receipt = PublicPreparationReceipt.from_dict(receipt_payload)
    _rewrite_receipt(result.suite_root, receipt_payload)

    assert new_packet_digest != old_packet_digest
    assert changed_manifest.digest() != old_manifest_digest
    assert read_public_preparation_receipt(result.suite_root) == changed_receipt
    with pytest.raises(PublicSourceIntegrityError, match="preparation packet.*expected"):
        read_public_preparation_receipt(
            result.suite_root,
            expected_preparation_packet_digest=old_packet_digest,
        )
    with pytest.raises(PublicSourceIntegrityError, match="Suite manifest.*expected"):
        read_public_preparation_receipt(
            result.suite_root,
            expected_suite_manifest_digest=old_manifest_digest,
        )
    assert read_public_preparation_receipt(
        result.suite_root,
        expected_preparation_packet_digest=new_packet_digest,
        expected_suite_manifest_digest=changed_manifest.digest(),
    ) == changed_receipt


@pytest.mark.parametrize(
    "anchor_name",
    ["expected_preparation_packet_digest", "expected_suite_manifest_digest"],
)
def test_receipt_external_anchors_require_valid_digests(
    tmp_path: Path, anchor_name: str
) -> None:
    _source_root, _source, result = _write_suite(tmp_path)

    with pytest.raises(PublicSourceIntegrityError, match="expected public.*digest"):
        read_public_preparation_receipt(
            result.suite_root,
            **{anchor_name: "not-a-digest"},
        )


def test_receipt_raw_bytes_must_remain_canonical(tmp_path: Path) -> None:
    _source_root, _source, result = _write_suite(tmp_path)
    receipt_path = result.suite_root / "preparation_receipt.json"
    receipt_path.write_bytes(b"\n" + receipt_path.read_bytes())

    with pytest.raises(PublicSourceIntegrityError, match="canonical"):
        read_public_preparation_receipt(result.suite_root)


def test_suite_manifest_raw_bytes_must_remain_canonical(tmp_path: Path) -> None:
    _source_root, _source, result = _write_suite(tmp_path)
    manifest_path = result.suite_root / "suite_manifest.json"
    manifest_path.write_bytes(b"\n" + manifest_path.read_bytes())

    with pytest.raises(PublicSourceIntegrityError, match="manifest.*canonical"):
        read_public_preparation_receipt(result.suite_root)


def test_bound_extra_file_bytes_are_reverified_on_hydration(tmp_path: Path) -> None:
    _source_root, _source, result = _write_suite(
        tmp_path, extra_files={"contexts/context.json": b"context"}
    )
    (result.suite_root / "contexts" / "context.json").write_bytes(b"tampered")

    with pytest.raises(PublicSourceIntegrityError, match="extra file.*size|extra file.*hash"):
        read_public_preparation_receipt(result.suite_root)


def test_published_extra_file_rejects_hardlink_alias(tmp_path: Path) -> None:
    _source_root, _source, result = _write_suite(
        tmp_path, extra_files={"contexts/context.json": b"context"}
    )
    extra = result.suite_root / "contexts" / "context.json"
    _create_hardlink_or_skip(extra, tmp_path / "published-extra-alias.json")

    with pytest.raises(PublicSourceIntegrityError, match="extra file.*hard link"):
        read_public_preparation_receipt(result.suite_root)


def test_published_case_rejects_hardlink_alias(tmp_path: Path) -> None:
    _source_root, _source, result = _write_suite(tmp_path)
    entry = result.manifest.cases[0]
    case_path = result.suite_root / Path(*entry.path.split("/"))
    _create_hardlink_or_skip(case_path, tmp_path / "published-case-alias.json")

    with pytest.raises(PublicSourceIntegrityError, match="Case.*hard link"):
        read_public_preparation_receipt(result.suite_root)


@pytest.mark.parametrize(
    "control_file", ["preparation_receipt.json", "suite_manifest.json"]
)
def test_receipt_hydration_rejects_hardlinked_control_files(
    tmp_path: Path, control_file: str
) -> None:
    _source_root, _source, result = _write_suite(
        tmp_path, name="suite-%s" % control_file.replace(".", "-")
    )
    target = result.suite_root / control_file
    alias = tmp_path / (control_file + ".alias")
    _create_hardlink_or_skip(target, alias)

    with pytest.raises(PublicSourceIntegrityError, match="hard link"):
        read_public_preparation_receipt(result.suite_root)


@pytest.mark.parametrize(
    "extra_files",
    [
        {"SUITE_MANIFEST.JSON": b"collision"},
        {"preparation_receipt.json/nested": b"collision"},
        {"cases/other.json": b"collision"},
        {"docs/\u00e9.txt": b"one", "DOCS/e\u0301.txt": b"two"},
        {"docs/readme": b"one", "DOCS/README/child": b"two"},
    ],
)
def test_extra_files_reject_control_plane_and_portable_path_collisions(
    tmp_path: Path, extra_files: dict[str, bytes]
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = _source_manifest(source_root)
    with pytest.raises(PublicPreparationError, match="collid|control-plane|portable"):
        write_public_suite(
            tmp_path / "suite",
            suite_id=SUITE_ID,
            suite_version="suite-v1",
            adapter_id="public-adapter",
            adapter_version="adapter-v1",
            source_manifest=source,
            filter_manifest=_filter_manifest("Python"),
            wire_contract=_repository_wire_contract(),
            cases=(_prepared_case(),),
            actual_statistics=(PublicStatistic("case_count", 1),),
            records=(_record(),),
            extra_files=extra_files,
        )
    assert not (tmp_path / "suite").exists()


def test_publication_is_create_only_and_preserves_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "suite"
    output.mkdir()
    sentinel = output / "owned.txt"
    sentinel.write_text("owner", encoding="utf-8")
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = _source_manifest(source_root)

    with pytest.raises(PublicPreparationError, match="already exists"):
        write_public_suite(
            output,
            suite_id=SUITE_ID,
            suite_version="suite-v1",
            adapter_id="public-adapter",
            adapter_version="adapter-v1",
            source_manifest=source,
            filter_manifest=_filter_manifest("Python"),
            wire_contract=_repository_wire_contract(),
            cases=(_prepared_case(),),
            actual_statistics=(PublicStatistic("case_count", 1),),
            records=(_record(),),
        )
    assert sentinel.read_text(encoding="utf-8") == "owner"


@pytest.mark.parametrize(
    ("budget_name", "budget_value", "cases", "message"),
    [
        (
            "MAX_SUITE_CASES",
            1,
            (_prepared_case("public-task-001"), _prepared_case("public-task-002")),
            "Case count",
        ),
        (
            "MAX_SUITE_TOTAL_CASE_BYTES",
            1,
            (_prepared_case(),),
            "Case bytes",
        ),
        (
            "MAX_SUITE_MANIFEST_BYTES",
            1,
            (_prepared_case(),),
            "manifest",
        ),
    ],
)
def test_suite_budgets_fail_before_staging_is_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget_name: str,
    budget_value: int,
    cases: tuple[PublicPreparedCase, ...],
    message: str,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = _source_manifest(source_root)

    def forbidden_mkdtemp(*args, **kwargs):  # pragma: no cover - assertion path
        raise AssertionError("staging must not be created before preflight succeeds")

    monkeypatch.setattr(public_module, budget_name, budget_value)
    monkeypatch.setattr(public_module.tempfile, "mkdtemp", forbidden_mkdtemp)
    with pytest.raises(PublicPreparationError, match=message):
        write_public_suite(
            tmp_path / "suite",
            suite_id=SUITE_ID,
            suite_version="suite-v1",
            adapter_id="public-adapter",
            adapter_version="adapter-v1",
            source_manifest=source,
            filter_manifest=_filter_manifest("Python"),
            wire_contract=_repository_wire_contract(),
            cases=cases,
            actual_statistics=(PublicStatistic("case_count", len(cases)),),
            records=tuple(_record(task_id=item.case.task_id) for item in cases),
        )
    assert not tuple(tmp_path.glob(".suite.*.staging"))


def _replace_with_directory_link_or_skip(
    path: Path, target: Path, kind: str
) -> None:
    os.rmdir(path)
    if kind == "symlink":
        try:
            os.symlink(target, path, target_is_directory=True)
        except OSError as exc:  # pragma: no cover - filesystem capability
            pytest.skip("directory symlink is unavailable: %s" % exc)
        return
    if os.name != "nt":
        pytest.skip("directory reparse-point race is Windows-specific")
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(path), str(target)],
        capture_output=True,
        text=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:  # pragma: no cover - filesystem capability
        pytest.skip("directory junction is unavailable: %s" % completed.stderr)


@pytest.mark.parametrize("unsafe_kind", ("symlink", "reparse"))
def test_write_new_rejects_directory_component_swap_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_kind: str,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    raced_component = staging / "cases"
    real_mkdir = public_module.os.mkdir
    injected = False

    def racing_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal injected
        if dir_fd is None:
            result = real_mkdir(path, mode)
        else:
            result = real_mkdir(path, mode, dir_fd=dir_fd)
        if not injected and Path(path).name == "cases":
            _replace_with_directory_link_or_skip(
                raced_component, outside, unsafe_kind
            )
            injected = True
        return result

    monkeypatch.setattr(public_module.os, "mkdir", racing_mkdir)
    try:
        with pytest.raises(PublicPreparationError, match="unsafe|symlink|reparse|safely"):
            public_module._write_new(staging, "cases/case.json", b"payload")
        assert not (outside / "case.json").exists()
    finally:
        if os.path.lexists(raced_component):
            if raced_component.is_symlink():
                raced_component.unlink()
            else:
                os.rmdir(raced_component)


def test_failed_publication_uses_owned_cleanup_not_recursive_rmtree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = _source_manifest(source_root)
    real_write = public_module._write_new

    def fail_manifest(root: Path, relative: str, data: bytes, **kwargs) -> None:
        if relative == "suite_manifest.json":
            raise PublicPreparationError("injected publication failure")
        real_write(root, relative, data, **kwargs)

    def forbidden_rmtree(*args, **kwargs):  # pragma: no cover - assertion path
        raise AssertionError("broad recursive cleanup is forbidden")

    monkeypatch.setattr(public_module, "_write_new", fail_manifest)
    monkeypatch.setattr(shutil, "rmtree", forbidden_rmtree)

    with pytest.raises(PublicPreparationError, match="injected"):
        write_public_suite(
            tmp_path / "suite",
            suite_id=SUITE_ID,
            suite_version="suite-v1",
            adapter_id="public-adapter",
            adapter_version="adapter-v1",
            source_manifest=source,
            filter_manifest=_filter_manifest("Python"),
            wire_contract=_repository_wire_contract(),
            cases=(_prepared_case(),),
            actual_statistics=(PublicStatistic("case_count", 1),),
            records=(_record(),),
        )
    assert not tuple(tmp_path.glob(".suite.*.staging"))


def test_receipt_rejects_unknown_fields_and_duplicate_keys(tmp_path: Path) -> None:
    _source_root, _source, result = _write_suite(tmp_path)
    payload = result.receipt.to_dict()
    payload["unexpected"] = True
    with pytest.raises(SchemaError, match="unknown field"):
        PublicPreparationReceipt.from_dict(payload)

    raw = result.receipt.to_json()
    duplicate = raw.replace(
        '"adapter_id":"public-adapter"',
        '"adapter_id":"public-adapter","adapter_id":"shadow"',
        1,
    )
    with pytest.raises(SchemaError, match="duplicate object key"):
        PublicPreparationReceipt.from_json(duplicate)


def test_receipt_schema_version_remains_explicit() -> None:
    assert PUBLIC_PREPARATION_RECEIPT_SCHEMA_VERSION.endswith("_v1")

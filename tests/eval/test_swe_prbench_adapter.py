from __future__ import annotations

import builtins
import copy
import datetime as _datetime
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

import review_agent_eval.adapters.swe_prbench as swe_adapter

from review_agent_eval.adapters._public import (
    PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION,
    PUBLIC_SOURCE_MANIFEST_SCHEMA_VERSION,
    PublicFilterManifest,
    PublicFormatError,
    PublicOptionalDependencyError,
    PublicPreparationError,
    PublicSelector,
    PublicSourceIntegrityError,
    PublicSourceManifest,
    PublicStatistic,
    read_public_preparation_receipt,
    source_file_from_path,
)
from review_agent_eval.adapters.swe_prbench import (
    SWE_PRBENCH_CONTEXT_CONFIGS,
    SWE_PRBENCH_DATASET_ID,
    SWE_PRBENCH_DATASET_LICENSE,
    SWE_PRBENCH_FIXTURE_DATASET_VERSION,
    SWE_PRBENCH_FIXTURE_SOURCE_MANIFEST_DIGEST,
    SWE_PRBENCH_FIXTURE_SOURCE_REVISION,
    SWE_PRBENCH_FIXTURE_SOURCE_URI,
    SWE_PRBENCH_FROZEN_PROTOCOL_ID,
    SWE_PRBENCH_HARNESS_LICENSE,
    SWE_PRBENCH_HARNESS_REVISION,
    SWE_PRBENCH_NATIVE_PROTOCOL_ID,
    SWE_PRBENCH_PARQUET_CONVERTER_REVISION,
    SWE_PRBENCH_PIPELINE_VERSION,
    SWE_PRBENCH_PROTOCOL_FROZEN,
    SWE_PRBENCH_PROTOCOL_NATIVE,
    SWE_PRBENCH_SOURCE_PARQUET,
    SWE_PRBENCH_SOURCE_PROFILE_EXPLICIT,
    SWE_PRBENCH_SOURCE_PROFILE_FIXTURE,
    SWE_PRBENCH_SOURCE_RAW,
    FrozenContextRecord,
    FrozenContextEnvelope,
    prepare_swe_prbench,
    prepare_swe_prbench_frozen_bundle,
    read_swe_prbench_frozen_bundle,
    validate_swe_prbench_source,
)
from review_agent_eval.cases import (
    FROZEN_CONTEXT_MATERIALIZER_PROTOCOL,
    REPOSITORY_MATERIALIZER_PROTOCOL,
)
from review_agent_eval.models import (
    CaseOrigin,
    EvaluatorContextSourceKind,
    EvaluatorContextTask,
    EvalCase,
    MetricAuthoritySource,
    NovelFindingPolicy,
    RepositorySource,
    ReviewTargetKind,
    TruthCompleteness,
    canonical_sha256,
    canonical_json_bytes,
    stable_id,
)


FIXTURE_ROOT = (
    Path(__file__).parent / "fixtures" / "public_datasets" / "swe_prbench"
).resolve()


def _task_ids(root: Path, source_format: str) -> list[str]:
    if source_format == SWE_PRBENCH_SOURCE_RAW:
        rows = [
            json.loads(line)
            for line in (root / "dataset" / "prs.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        return [row["task_id"] for row in rows]
    return sorted(
        path.name[: -len("_human.json")]
        for path in (root / "dataset" / "annotations").glob("*_human.json")
    )


def _source_manifest(
    root: Path,
    *,
    source_format: str = SWE_PRBENCH_SOURCE_RAW,
) -> PublicSourceManifest:
    tasks = _task_ids(root, source_format)
    if source_format == SWE_PRBENCH_SOURCE_RAW:
        pr_role = "prs_jsonl"
        pr_path = "dataset/prs.jsonl"
    else:
        pr_role = "prs_parquet"
        pr_path = "dataset/prs.parquet"
    files = [source_file_from_path(root, role=pr_role, path=pr_path)]
    context_counts = {config: 0 for config in SWE_PRBENCH_CONTEXT_CONFIGS}
    annotation_count = 0
    for task_id in tasks:
        annotation_path = "dataset/annotations/%s_human.json" % task_id
        if (root / Path(annotation_path)).exists():
            files.append(
                source_file_from_path(
                    root,
                    role="annotation.%s" % task_id,
                    path=annotation_path,
                )
            )
            annotation_count += 1
        for config in SWE_PRBENCH_CONTEXT_CONFIGS:
            context_path = "dataset/contexts/%s/%s.json" % (config, task_id)
            if (root / Path(context_path)).exists():
                files.append(
                    source_file_from_path(
                        root,
                        role="context.%s.%s" % (config, task_id),
                        path=context_path,
                    )
                )
                context_counts[config] += 1
    return PublicSourceManifest(
        schema_version=PUBLIC_SOURCE_MANIFEST_SCHEMA_VERSION,
        dataset_id=SWE_PRBENCH_DATASET_ID,
        dataset_version=SWE_PRBENCH_FIXTURE_DATASET_VERSION,
        source_uri=SWE_PRBENCH_FIXTURE_SOURCE_URI,
        source_revision=SWE_PRBENCH_FIXTURE_SOURCE_REVISION,
        license=SWE_PRBENCH_DATASET_LICENSE,
        files=tuple(files),
        expected_statistics=(
            PublicStatistic(name="pr_records", value=len(tasks)),
            PublicStatistic(name="annotations", value=annotation_count),
            PublicStatistic(
                name="contexts", value=sum(context_counts.values())
            ),
            PublicStatistic(
                name="context_config_a", value=context_counts["config_A"]
            ),
            PublicStatistic(
                name="context_config_b", value=context_counts["config_B"]
            ),
            PublicStatistic(
                name="context_config_c", value=context_counts["config_C"]
            ),
        ),
    )


def _filter_manifest(
    protocol: str,
    *,
    source_profile: str = SWE_PRBENCH_SOURCE_PROFILE_FIXTURE,
    source_format: str = SWE_PRBENCH_SOURCE_RAW,
    context_config: str | None = None,
    fallback: str | None = None,
) -> PublicFilterManifest:
    if context_config is None:
        context_config = (
            "none" if protocol == SWE_PRBENCH_PROTOCOL_NATIVE else "config_A"
        )
    selectors = [
        PublicSelector(name="source_scope", values=("fixture",)),
        PublicSelector(name="source_profile", values=(source_profile,)),
        PublicSelector(name="source_format", values=(source_format,)),
        PublicSelector(name="protocol", values=(protocol,)),
        PublicSelector(name="context_config", values=(context_config,)),
        PublicSelector(
            name="harness_revision", values=(SWE_PRBENCH_HARNESS_REVISION,)
        ),
        PublicSelector(
            name="harness_license", values=(SWE_PRBENCH_HARNESS_LICENSE,)
        ),
        PublicSelector(
            name="pipeline_version", values=(SWE_PRBENCH_PIPELINE_VERSION,)
        ),
    ]
    if source_format == SWE_PRBENCH_SOURCE_PARQUET:
        selectors.append(
            PublicSelector(
                name="parquet_converter_revision",
                values=(SWE_PRBENCH_PARQUET_CONVERTER_REVISION,),
            )
        )
    if fallback is not None:
        selectors.append(
            PublicSelector(name="ground_truth_fallback", values=(fallback,))
        )
    return PublicFilterManifest(
        schema_version=PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION,
        dataset_id=SWE_PRBENCH_DATASET_ID,
        selectors=tuple(selectors),
    )


def _copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "source"
    shutil.copytree(FIXTURE_ROOT, target)
    return target


def _explicit_digest(manifest: PublicSourceManifest) -> str:
    return manifest.digest()


def _load_case(result) -> EvalCase:
    entry = result.manifest.cases[0]
    return EvalCase.from_json((result.suite_root / entry.path).read_bytes())


def test_fixture_profile_is_independently_catalog_bound() -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    checked_in = PublicSourceManifest.from_json(
        (FIXTURE_ROOT / "source_manifest.json").read_bytes()
    )
    assert checked_in == manifest
    assert manifest.source_revision == SWE_PRBENCH_FIXTURE_SOURCE_REVISION
    assert manifest.digest() == SWE_PRBENCH_FIXTURE_SOURCE_MANIFEST_DIGEST


def test_strict_parser_preserves_all_context_fields_and_cross_file_binding() -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    validation = validate_swe_prbench_source(
        FIXTURE_ROOT,
        source_manifest=manifest,
        filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_NATIVE),
    )

    assert validation.pr_count == 1
    assert validation.annotation_count == 1
    assert validation.context_count == 3
    assert validation.task_ids == ("dask__12221",)
    assert validation.empty_ground_truth_task_ids == ()
    assert validation.oversized_claim_task_ids == ()
    assert {item.config_name for item in validation.frozen_contexts} == set(
        SWE_PRBENCH_CONTEXT_CONFIGS
    )

    raw = json.loads(
        (
            FIXTURE_ROOT
            / "dataset"
            / "contexts"
            / "config_A"
            / "dask__12221.json"
        ).read_text(encoding="utf-8")
    )
    record = next(
        item for item in validation.frozen_contexts if item.config_name == "config_A"
    )
    assert isinstance(record, FrozenContextRecord)
    assert record.to_upstream_dict() == raw
    assert record.rendered_sha256 == hashlib.sha256(
        raw["rendered"].encode("utf-8")
    ).hexdigest()


def test_native_repository_publishes_v2_repository_target_with_unscorable_authority(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    result = prepare_swe_prbench(
        FIXTURE_ROOT,
        tmp_path / "native-suite",
        source_manifest=manifest,
        filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_NATIVE),
        protocol=SWE_PRBENCH_PROTOCOL_NATIVE,
    )

    entry = result.manifest.cases[0]
    case = _load_case(result)
    assert entry.split.value == "capability"
    assert entry.protocol_id == SWE_PRBENCH_NATIVE_PROTOCOL_ID
    assert result.manifest.wire_contract.review_target_kind is ReviewTargetKind.REPOSITORY
    assert (
        result.manifest.wire_contract.materializer_protocol
        == REPOSITORY_MATERIALIZER_PROTOCOL
    )
    assert entry.truth_completeness is TruthCompleteness.HUMAN_OBSERVED
    assert case.source.origin is CaseOrigin.SWE_PRBENCH
    assert case.source.source_version == SWE_PRBENCH_FIXTURE_SOURCE_REVISION
    assert case.source.license == SWE_PRBENCH_DATASET_LICENSE
    assert case.input.review_target.kind is ReviewTargetKind.REPOSITORY
    assert case.input.review_target.repository.source is RepositorySource.GIT
    assert case.input.review_target.repository.url == "https://github.com/dask/dask.git"
    assert case.input.review_target.repository.base_revision == (
        "0a075534b29af7364b82fdf04a33838ab7189d77"
    )
    assert case.input.review_target.repository.head_revision == (
        "59dab320f45e409dec89df9e13f02cb049db6eb4"
    )
    assert case.input.review_target.review_request.user_intent is None
    assert case.input.review_target.review_request.existing_ci_evidence == ()
    assert case.intent_truth.scorable is False
    assert case.review_truth.completeness is TruthCompleteness.HUMAN_OBSERVED
    assert case.review_truth.novel_finding_policy is NovelFindingPolicy.VERIFY
    assert len(case.review_truth.expected_findings) == 3
    assert {item.severity for item in case.review_truth.expected_findings} == {None}
    assert all(
        not item.metric_authority.severity_scorable
        and item.metric_authority.severity_authority is None
        and not item.metric_authority.location_scorable
        and item.metric_authority.location_authority is None
        for item in case.review_truth.expected_findings
    )
    assert {item.category for item in case.review_truth.expected_findings} == {
        "human_review_comment"
    }
    assert all(item.locations[0].side is None for item in case.review_truth.expected_findings)
    assert all(
        len(item.locations) == 1
        and item.locations[0].path
        and item.locations[0].from_line is not None
        for item in case.review_truth.expected_findings
    )
    assert len(case.review_evaluator_context.truth_contexts) == 3
    assert all(
        context.allowed_tasks == (EvaluatorContextTask.FINDING_EQUIVALENCE,)
        and len(context.sources) == 1
        and context.sources[0].kind is EvaluatorContextSourceKind.DIFF_HUNK
        for context in case.review_evaluator_context.truth_contexts
    )
    assert "diff_hunk" not in json.dumps(case.eval_input().to_dict())
    annotation = json.loads(
        (
            FIXTURE_ROOT
            / "dataset"
            / "annotations"
            / "dask__12221_human.json"
        ).read_text(encoding="utf-8")
    )
    annotation_source = next(
        item
        for item in manifest.files
        if item.role == "annotation.dask__12221"
    )
    comments = {item["comment_id"]: item for item in annotation["comments"]}
    for finding in case.review_truth.expected_findings:
        comment_id = next(
            item["comment_id"]
            for item in annotation["comments"]
            if item["body"] == finding.claim
        )
        context = next(
            item
            for item in case.review_evaluator_context.truth_contexts
            if item.truth_id == finding.truth_id
        )
        source = context.sources[0]
        assert source.content == comments[comment_id]["diff_hunk"]
        assert source.provenance.source_file_sha256 == annotation_source.sha256
        assert source.provenance.record_pointer.endswith(
            "/comments/%d" % next(
                index
                for index, item in enumerate(annotation["comments"])
                if item["comment_id"] == comment_id
            )
        )
        assert source.provenance.record_sha256 == canonical_sha256(
            comments[comment_id]
        )

    dimensions = {item.name: item.value for item in entry.dimensions}
    assert dimensions["upstream_declared_difficulty"] == "Type1_Direct"
    assert dimensions["difficulty_policy"] == "upstream_declared_not_gate_truth"
    assert dimensions["underlying_repository_license"] == (
        "not_normalized_by_upstream"
    )
    assert dimensions["protocol_comparability"] == (
        "native_non_official_leaderboard"
    )
    assert dimensions["severity_policy"] == "not_scorable_no_upstream_authority"
    assert dimensions["line_policy"] == "semantic_only_upstream_nullable_side_unknown"
    assert dimensions["diff_hunk_policy"] == (
        "truth_scoped_evaluator_context_finding_equivalence_only"
    )

    receipt = read_public_preparation_receipt(result.suite_root)
    statistics = {item.name: item.value for item in receipt.actual_statistics}
    assert statistics["selected_included"] == 1
    assert statistics["selected_isolated"] == 0
    assert statistics["unselected_filtered"] == 0
    assert statistics["upstream_isolated"] == 0
    assert (
        statistics["selected_included"]
        + statistics["selected_isolated"]
        + statistics["unselected_filtered"]
        + statistics["upstream_isolated"]
        == statistics["source_pr_records"]
    )
    assert "selected_records" not in statistics
    assert "filtered_records" not in statistics
    assert "isolated_records" not in statistics
    mapping = next(item for item in receipt.records if item.source_role == "prs_jsonl")
    mapping_payload = json.loads(mapping.record_json)
    assert mapping_payload["source_profile"] == SWE_PRBENCH_SOURCE_PROFILE_FIXTURE
    assert mapping_payload["harness_revision"] == SWE_PRBENCH_HARNESS_REVISION
    assert mapping_payload["dataset_license"] == SWE_PRBENCH_DATASET_LICENSE
    assert mapping_payload["underlying_repository_license"] == (
        "not_normalized_by_upstream"
    )
    assert "eval_v1_limitations" not in mapping_payload
    assert "requires_eval_v2" not in mapping.record_json


def test_truth_context_uses_original_comment_index_when_truth_order_differs(
    tmp_path: Path,
) -> None:
    root = _copy_fixture(tmp_path)
    annotation_path = (
        root / "dataset" / "annotations" / "dask__12221_human.json"
    )
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation["substantive_comment_ids"] = ["c_3", "c_1", "c_2"]
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    manifest = _source_manifest(root)
    result = prepare_swe_prbench(
        root,
        tmp_path / "reordered-suite",
        source_manifest=manifest,
        filter_manifest=_filter_manifest(
            SWE_PRBENCH_PROTOCOL_NATIVE,
            source_profile=SWE_PRBENCH_SOURCE_PROFILE_EXPLICIT,
        ),
        protocol=SWE_PRBENCH_PROTOCOL_NATIVE,
        expected_source_manifest_digest=manifest.digest(),
    )

    case = _load_case(result)
    contexts = {
        item.truth_id: item
        for item in case.review_evaluator_context.truth_contexts
    }
    receipts = {
        item.truth_id: item
        for item in result.receipt.records
        if item.truth_id is not None
    }
    for source_index, comment in enumerate(annotation["comments"]):
        if comment["comment_id"] not in annotation["substantive_comment_ids"]:
            continue
        truth_id = stable_id(
            "swe-truth", annotation["task_id"], comment["comment_id"]
        )
        source = contexts[truth_id].sources[0]
        expected_pointer = (
            "dataset/annotations/dask__12221_human.json#/comments/%d"
            % source_index
        )
        assert source.provenance.record_pointer == expected_pointer
        assert source.provenance.record_sha256 == canonical_sha256(comment)
        assert source.content == comment["diff_hunk"]
        assert receipts[truth_id].record_pointer == expected_pointer


@pytest.mark.parametrize(
    ("selected", "truth_status", "expected"),
    (
        (True, "representable", "selected_included"),
        (True, "empty_ground_truth", "selected_isolated"),
        (False, "representable", "unselected_filtered"),
        (False, "claim_exceeds_claim_limit", "upstream_isolated"),
    ),
)
def test_record_partition_is_mutually_exclusive(
    selected: bool, truth_status: str, expected: str
) -> None:
    assert swe_adapter._record_partition(selected, truth_status) == expected


def test_frozen_protocol_publishes_runnable_frozen_target_suite(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    result = prepare_swe_prbench(
        FIXTURE_ROOT,
        tmp_path / "frozen-suite",
        source_manifest=manifest,
        filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_FROZEN),
        protocol=SWE_PRBENCH_PROTOCOL_FROZEN,
    )

    entry = result.manifest.cases[0]
    case = _load_case(result)
    assert result.manifest.suite_id == "swe-prbench-frozen"
    assert entry.protocol_id == SWE_PRBENCH_FROZEN_PROTOCOL_ID
    assert entry.protocol_id != SWE_PRBENCH_NATIVE_PROTOCOL_ID
    assert result.manifest.wire_contract.review_target_kind is ReviewTargetKind.FROZEN_CONTEXT
    assert (
        result.manifest.wire_contract.materializer_protocol
        == FROZEN_CONTEXT_MATERIALIZER_PROTOCOL
    )
    assert case.input.review_target.kind is ReviewTargetKind.FROZEN_CONTEXT
    target = case.input.review_target
    bundle_root = result.suite_root / "frozen_bundle"
    bundle = read_swe_prbench_frozen_bundle(
        bundle_root, expected_bundle_id=target.bundle_id
    )
    assert target.bundle_id == bundle.manifest.bundle_id
    assert bundle.manifest.records[0].task_id == case.task_id
    assert target.record_id == case.task_id
    assert case.review_truth.expected_findings
    assert all(
        item.severity is None
        and not item.metric_authority.severity_scorable
        and not item.metric_authority.location_scorable
        for item in case.review_truth.expected_findings
    )
    assert all(
        "requires_eval_v2" not in json.dumps(item.to_dict())
        for item in case.review_truth.expected_findings
    )

    from review_agent_eval.frozen_context import open_frozen_context_replay

    preparation = result.manifest.source.preparation_binding
    assert preparation is not None
    replay = open_frozen_context_replay(
        bundle_root=bundle_root,
        eval_input=case.eval_input(),
        suite_preparation_binding=preparation,
        suite_preparation_binding_digest=preparation.digest(),
    )
    assert replay.read_exact() == json.loads(
        (FIXTURE_ROOT / "dataset" / "contexts" / "config_A" / "dask__12221.json").read_text(
            encoding="utf-8"
        )
    )["rendered"].encode("utf-8")


def test_frozen_temp_container_is_cleaned_when_mapping_fails_after_bundle_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    output = tmp_path / "mapping-failure-suite"

    def fail_mapping(*args, **kwargs):
        raise PublicPreparationError("injected mapping failure")

    monkeypatch.setattr(swe_adapter, "_mapping_record", fail_mapping)
    with pytest.raises(PublicPreparationError, match="injected mapping failure"):
        prepare_swe_prbench(
            FIXTURE_ROOT,
            output,
            source_manifest=manifest,
            filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_FROZEN),
            protocol=SWE_PRBENCH_PROTOCOL_FROZEN,
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".mapping-failure-suite.frozen.*"))


@pytest.mark.parametrize(
    ("target_field", "wrong_value"),
    (("bundle_id", "unrelated-bundle"), ("record_id", "unrelated-record")),
)
def test_frozen_target_binding_is_rejected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_field: str,
    wrong_value: str,
) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    output = tmp_path / ("wrong-%s-suite" % target_field)
    real_write = swe_adapter.write_public_suite

    def write_with_wrong_target(*args, **kwargs):
        prepared = list(kwargs["cases"])
        case = prepared[0].case
        target = replace(
            case.input.review_target,
            **{target_field: wrong_value},
        )
        wrong_case = replace(
            case,
            input=replace(case.input, review_target=target),
        )
        prepared[0] = replace(prepared[0], case=wrong_case)
        kwargs["cases"] = tuple(prepared)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(swe_adapter, "write_public_suite", write_with_wrong_target)
    with pytest.raises(PublicPreparationError, match="Frozen.*Target|bundle"):
        prepare_swe_prbench(
            FIXTURE_ROOT,
            output,
            source_manifest=manifest,
            filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_FROZEN),
            protocol=SWE_PRBENCH_PROTOCOL_FROZEN,
        )
    assert not output.exists()


def test_frozen_record_drift_after_verify_is_rejected_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    output = tmp_path / "drift-suite"
    real_mapping = swe_adapter._mapping_record
    mutated = False

    def mutate_verified_record(*args, **kwargs):
        nonlocal mutated
        if not mutated:
            record_path = next(
                tmp_path.glob(
                    ".drift-suite.frozen.*/bundle/records/*/*.json"
                )
            )
            payload = json.loads(record_path.read_text(encoding="utf-8"))
            payload["record"]["rendered"] += "\npost-verify drift"
            record_path.write_bytes(canonical_json_bytes(payload))
            mutated = True
        return real_mapping(*args, **kwargs)

    def forbid_writer(*args, **kwargs):  # pragma: no cover - assertion path
        raise AssertionError("drifted bundle must fail before Suite writer")

    monkeypatch.setattr(swe_adapter, "_mapping_record", mutate_verified_record)
    monkeypatch.setattr(swe_adapter, "write_public_suite", forbid_writer)
    with pytest.raises(PublicPreparationError, match="drift|binding|hash|size"):
        prepare_swe_prbench(
            FIXTURE_ROOT,
            output,
            source_manifest=manifest,
            filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_FROZEN),
            protocol=SWE_PRBENCH_PROTOCOL_FROZEN,
        )
    assert mutated
    assert not output.exists()


def test_staging_frozen_bundle_verifier_blocks_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    output = tmp_path / "staging-rejected-suite"
    real_read = swe_adapter.read_swe_prbench_frozen_bundle
    staging_checks: list[Path] = []

    def reject_staging_bundle(root, *args, **kwargs):
        path = Path(root)
        if path.name == "frozen_bundle" and path.parent.name.endswith(".staging"):
            staging_checks.append(path)
            raise PublicSourceIntegrityError("injected staging bundle rejection")
        return real_read(root, *args, **kwargs)

    monkeypatch.setattr(
        swe_adapter,
        "read_swe_prbench_frozen_bundle",
        reject_staging_bundle,
    )
    with pytest.raises(PublicPreparationError, match="staging|Frozen bundle"):
        prepare_swe_prbench(
            FIXTURE_ROOT,
            output,
            source_manifest=manifest,
            filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_FROZEN),
            protocol=SWE_PRBENCH_PROTOCOL_FROZEN,
        )
    assert staging_checks
    assert not output.exists()


def test_frozen_bundle_is_exact_typed_create_only_and_not_a_runnable_suite(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    filter_manifest = _filter_manifest(
        SWE_PRBENCH_PROTOCOL_FROZEN, context_config="config_B"
    )
    output = tmp_path / "frozen-bundle"
    prepared = prepare_swe_prbench_frozen_bundle(
        FIXTURE_ROOT,
        output,
        source_manifest=manifest,
        filter_manifest=filter_manifest,
    )

    assert prepared.root == output.resolve()
    assert len(prepared.manifest.records) == 1
    binding = prepared.manifest.records[0]
    assert binding.config_name == "config_B"
    assert binding.review_truth_status == "representable"
    assert prepared.manifest.harness_revision == SWE_PRBENCH_HARNESS_REVISION
    assert prepared.manifest.harness_license == "MIT"
    assert prepared.manifest.dataset_license == "CC-BY-4.0"
    assert prepared.manifest.underlying_repository_license == (
        "not_normalized_by_upstream"
    )
    assert not (output / "suite_manifest.json").exists()
    assert not (output / "cases").exists()

    envelope = FrozenContextEnvelope.from_json((output / binding.path).read_bytes())
    assert envelope.bundle_id == prepared.manifest.bundle_id
    record = envelope.record
    upstream = json.loads(
        (
            FIXTURE_ROOT
            / "dataset"
            / "contexts"
            / "config_B"
            / "dask__12221.json"
        ).read_text(encoding="utf-8")
    )
    assert record.to_upstream_dict() == upstream
    assert binding.rendered_sha256 == hashlib.sha256(
        upstream["rendered"].encode("utf-8")
    ).hexdigest()
    assert read_swe_prbench_frozen_bundle(
        output, expected_bundle_id=prepared.manifest.bundle_id
    ).manifest == prepared.manifest

    with pytest.raises(PublicPreparationError, match="already exists"):
        prepare_swe_prbench_frozen_bundle(
            FIXTURE_ROOT,
            output,
            source_manifest=manifest,
            filter_manifest=filter_manifest,
        )


def test_frozen_bundle_detects_record_tampering(tmp_path: Path) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    prepared = prepare_swe_prbench_frozen_bundle(
        FIXTURE_ROOT,
        tmp_path / "bundle",
        source_manifest=manifest,
        filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_FROZEN),
    )
    path = prepared.root / prepared.manifest.records[0].path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["record"]["rendered"] += "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PublicSourceIntegrityError, match="bytes drifted"):
        read_swe_prbench_frozen_bundle(
            prepared.root, expected_bundle_id=prepared.manifest.bundle_id
        )


def test_frozen_bundle_read_requires_external_anchor_for_every_profile(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    prepared = prepare_swe_prbench_frozen_bundle(
        FIXTURE_ROOT,
        tmp_path / "bundle",
        source_manifest=manifest,
        filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_FROZEN),
    )
    with pytest.raises(TypeError, match="expected_bundle_id"):
        read_swe_prbench_frozen_bundle(prepared.root)  # type: ignore[call-arg]


def test_self_consistent_frozen_rewrite_is_rejected_by_original_trust_anchor(
    tmp_path: Path,
) -> None:
    original_manifest = _source_manifest(FIXTURE_ROOT)
    original = prepare_swe_prbench_frozen_bundle(
        FIXTURE_ROOT,
        tmp_path / "original",
        source_manifest=original_manifest,
        filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_FROZEN),
    )

    rewritten_root = _copy_fixture(tmp_path / "rewritten-source")
    context_path = (
        rewritten_root
        / "dataset"
        / "contexts"
        / "config_A"
        / "dask__12221.json"
    )
    payload = json.loads(context_path.read_text(encoding="utf-8"))
    payload["rendered"] += "\nself-consistent rewrite"
    context_path.write_text(json.dumps(payload), encoding="utf-8")
    rewritten_manifest = _source_manifest(rewritten_root)
    rewritten_filter = _filter_manifest(
        SWE_PRBENCH_PROTOCOL_FROZEN,
        source_profile=SWE_PRBENCH_SOURCE_PROFILE_EXPLICIT,
    )
    rewritten = prepare_swe_prbench_frozen_bundle(
        rewritten_root,
        tmp_path / "rewritten",
        source_manifest=rewritten_manifest,
        filter_manifest=rewritten_filter,
        expected_source_manifest_digest=rewritten_manifest.digest(),
    )

    assert rewritten.manifest.bundle_id != original.manifest.bundle_id
    with pytest.raises(PublicSourceIntegrityError, match="expected trust anchor"):
        read_swe_prbench_frozen_bundle(
            rewritten.root,
            expected_bundle_id=original.manifest.bundle_id,
        )


def test_frozen_bundle_identity_binds_adapter_and_license_metadata(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    prepared = prepare_swe_prbench_frozen_bundle(
        FIXTURE_ROOT,
        tmp_path / "bundle",
        source_manifest=manifest,
        filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_FROZEN),
    )
    bundle = prepared.manifest
    base = {
        "adapter_id": bundle.adapter_id,
        "adapter_version": bundle.adapter_version,
        "harness_revision": bundle.harness_revision,
        "harness_license": bundle.harness_license,
        "dataset_license": bundle.dataset_license,
        "underlying_repository_license": bundle.underlying_repository_license,
        "source_manifest_digest": bundle.source_manifest_digest,
        "filter_manifest_digest": bundle.filter_manifest_digest,
        "identity_records": swe_adapter._frozen_bundle_identity_records(
            bundle.records
        ),
    }
    assert swe_adapter._compute_frozen_bundle_id(**base) == bundle.bundle_id
    for name, value in (
        ("adapter_id", "different-adapter"),
        ("adapter_version", "different-version"),
        ("harness_revision", "0" * 40),
        ("harness_license", "different-license"),
        ("dataset_license", "different-license"),
        ("underlying_repository_license", "different-scope"),
    ):
        changed = dict(base)
        changed[name] = value
        assert swe_adapter._compute_frozen_bundle_id(**changed) != bundle.bundle_id


def test_frozen_binding_must_point_to_its_source_manifest_role_and_hash(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    prepared = prepare_swe_prbench_frozen_bundle(
        FIXTURE_ROOT,
        tmp_path / "bundle",
        source_manifest=manifest,
        filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_FROZEN),
    )
    path = prepared.root / "frozen_bundle_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["source_file_sha256"] = "0" * 64
    binding = swe_adapter.FrozenContextBinding.from_dict(payload["records"][0])
    payload["bundle_id"] = swe_adapter._compute_frozen_bundle_id(
        adapter_id=payload["adapter_id"],
        adapter_version=payload["adapter_version"],
        harness_revision=payload["harness_revision"],
        harness_license=payload["harness_license"],
        dataset_license=payload["dataset_license"],
        underlying_repository_license=payload["underlying_repository_license"],
        source_manifest_digest=payload["source_manifest_digest"],
        filter_manifest_digest=payload["filter_manifest_digest"],
        identity_records=swe_adapter._frozen_bundle_identity_records((binding,)),
    )
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(PublicSourceIntegrityError, match="source manifest role/hash"):
        read_swe_prbench_frozen_bundle(
            prepared.root, expected_bundle_id=payload["bundle_id"]
        )


def _hardlink_or_skip(source: Path, alias: Path) -> None:
    try:
        os.link(source, alias)
    except (OSError, NotImplementedError) as exc:
        pytest.skip("filesystem cannot create a hardlink for this test: %s" % exc)


@pytest.mark.parametrize("target_kind", ("manifest", "record"))
def test_frozen_bundle_reader_rejects_hardlinked_files(
    tmp_path: Path, target_kind: str
) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    prepared = prepare_swe_prbench_frozen_bundle(
        FIXTURE_ROOT,
        tmp_path / "bundle",
        source_manifest=manifest,
        filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_FROZEN),
    )
    target = (
        prepared.root / "frozen_bundle_manifest.json"
        if target_kind == "manifest"
        else prepared.root / prepared.manifest.records[0].path
    )
    _hardlink_or_skip(target, tmp_path / (target_kind + "-alias.json"))
    with pytest.raises(PublicSourceIntegrityError, match="hard links"):
        read_swe_prbench_frozen_bundle(
            prepared.root, expected_bundle_id=prepared.manifest.bundle_id
        )


def test_frozen_bundle_manifest_requires_canonical_bytes(tmp_path: Path) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    prepared = prepare_swe_prbench_frozen_bundle(
        FIXTURE_ROOT,
        tmp_path / "bundle",
        source_manifest=manifest,
        filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_FROZEN),
    )
    path = prepared.root / "frozen_bundle_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(PublicSourceIntegrityError, match="not canonical"):
        read_swe_prbench_frozen_bundle(
            prepared.root, expected_bundle_id=prepared.manifest.bundle_id
        )


def test_frozen_envelope_points_back_to_manifest_even_if_file_hash_is_rewritten(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    prepared = prepare_swe_prbench_frozen_bundle(
        FIXTURE_ROOT,
        tmp_path / "bundle",
        source_manifest=manifest,
        filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_FROZEN),
    )
    binding = prepared.manifest.records[0]
    record_path = prepared.root / binding.path
    envelope = json.loads(record_path.read_text(encoding="utf-8"))
    envelope["bundle_id"] = "swe-frozen-bundle-" + "0" * 64
    envelope_raw = canonical_json_bytes(envelope)
    record_path.write_bytes(envelope_raw)

    manifest_path = prepared.root / "frozen_bundle_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["records"][0]["size_bytes"] = len(envelope_raw)
    manifest_payload["records"][0]["sha256"] = hashlib.sha256(
        envelope_raw
    ).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest_payload))

    with pytest.raises(PublicSourceIntegrityError, match="point back"):
        read_swe_prbench_frozen_bundle(
            prepared.root, expected_bundle_id=prepared.manifest.bundle_id
        )


def test_frozen_bundle_publication_race_preserves_competing_output_and_cleans_owned_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    output = tmp_path / "bundle"
    real_publish = swe_adapter._publish_directory_create_only

    def publish_after_competitor(staging: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "competitor.txt").write_text("preserve", encoding="utf-8")
        real_publish(staging, destination)

    monkeypatch.setattr(
        swe_adapter,
        "_publish_directory_create_only",
        publish_after_competitor,
    )
    with pytest.raises(PublicPreparationError, match="already exists"):
        prepare_swe_prbench_frozen_bundle(
            FIXTURE_ROOT,
            output,
            source_manifest=manifest,
            filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_FROZEN),
        )
    assert (output / "competitor.txt").read_text(encoding="utf-8") == "preserve"
    assert not tuple(tmp_path.glob(".bundle.*.staging"))


def test_protocol_argument_and_filter_cannot_mix(tmp_path: Path) -> None:
    manifest = _source_manifest(FIXTURE_ROOT)
    with pytest.raises(PublicFormatError, match="mixed protocols are forbidden"):
        prepare_swe_prbench(
            FIXTURE_ROOT,
            tmp_path / "mixed",
            source_manifest=manifest,
            filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_FROZEN),
            protocol=SWE_PRBENCH_PROTOCOL_NATIVE,
        )


def test_self_signed_manifest_cannot_masquerade_as_catalog_fixture(
    tmp_path: Path,
) -> None:
    root = _copy_fixture(tmp_path)
    context_path = (
        root / "dataset" / "contexts" / "config_C" / "dask__12221.json"
    )
    payload = json.loads(context_path.read_text(encoding="utf-8"))
    payload["rendered"] += "\nmutated"
    context_path.write_text(json.dumps(payload), encoding="utf-8")
    self_signed = _source_manifest(root)

    with pytest.raises(PublicSourceIntegrityError, match="trusted catalog/profile"):
        validate_swe_prbench_source(
            root,
            source_manifest=self_signed,
            filter_manifest=_filter_manifest(SWE_PRBENCH_PROTOCOL_NATIVE),
        )


def test_independently_pinned_explicit_manifest_reaches_strict_schema_checks(
    tmp_path: Path,
) -> None:
    root = _copy_fixture(tmp_path)
    context_path = (
        root / "dataset" / "contexts" / "config_C" / "dask__12221.json"
    )
    payload = json.loads(context_path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    context_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = _source_manifest(root)

    with pytest.raises(PublicFormatError, match="unknown field.*unexpected"):
        validate_swe_prbench_source(
            root,
            source_manifest=manifest,
            filter_manifest=_filter_manifest(
                SWE_PRBENCH_PROTOCOL_NATIVE,
                source_profile=SWE_PRBENCH_SOURCE_PROFILE_EXPLICIT,
            ),
            expected_source_manifest_digest=_explicit_digest(manifest),
        )


def test_cross_file_task_binding_and_complete_a_b_c_tree_fail_closed(
    tmp_path: Path,
) -> None:
    root = _copy_fixture(tmp_path)
    path = root / "dataset" / "contexts" / "config_C" / "dask__12221.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["repo"] = "other/repository"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = _source_manifest(root)
    with pytest.raises(PublicFormatError, match="PR/Context binding drifted"):
        validate_swe_prbench_source(
            root,
            source_manifest=manifest,
            filter_manifest=_filter_manifest(
                SWE_PRBENCH_PROTOCOL_NATIVE,
                source_profile=SWE_PRBENCH_SOURCE_PROFILE_EXPLICIT,
            ),
            expected_source_manifest_digest=manifest.digest(),
        )

    missing_root = _copy_fixture(tmp_path / "missing")
    (missing_root / "dataset" / "contexts" / "config_C" / "dask__12221.json").unlink()
    missing_manifest = _source_manifest(missing_root)
    with pytest.raises(PublicFormatError, match="incomplete or ambiguous"):
        validate_swe_prbench_source(
            missing_root,
            source_manifest=missing_manifest,
            filter_manifest=_filter_manifest(
                SWE_PRBENCH_PROTOCOL_NATIVE,
                source_profile=SWE_PRBENCH_SOURCE_PROFILE_EXPLICIT,
            ),
            expected_source_manifest_digest=missing_manifest.digest(),
        )


def test_oversized_human_comment_is_not_truncated_or_split(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    annotation_path = (
        root / "dataset" / "annotations" / "dask__12221_human.json"
    )
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    body = "x" * 8193
    annotation["comments"][0]["body"] = body
    annotation["requested_changes"][0]["body"] = body
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    manifest = _source_manifest(root)
    filter_manifest = _filter_manifest(
        SWE_PRBENCH_PROTOCOL_FROZEN,
        source_profile=SWE_PRBENCH_SOURCE_PROFILE_EXPLICIT,
    )
    validation = validate_swe_prbench_source(
        root,
        source_manifest=manifest,
        filter_manifest=filter_manifest,
        expected_source_manifest_digest=manifest.digest(),
    )
    assert validation.oversized_claim_task_ids == ("dask__12221",)

    bundle = prepare_swe_prbench_frozen_bundle(
        root,
        tmp_path / "oversized-bundle",
        source_manifest=manifest,
        filter_manifest=filter_manifest,
        expected_source_manifest_digest=manifest.digest(),
    )
    binding = bundle.manifest.records[0]
    assert binding.review_truth_status == "claim_exceeds_claim_limit"
    assert binding.offending_record_sha256 == canonical_sha256(
        annotation["comments"][0]
    )
    assert "8193 chars" in binding.review_truth_reason

    native_filter = _filter_manifest(
        SWE_PRBENCH_PROTOCOL_NATIVE,
        source_profile=SWE_PRBENCH_SOURCE_PROFILE_EXPLICIT,
    )
    with pytest.raises(PublicPreparationError, match="no runnable cases"):
        prepare_swe_prbench(
            root,
            tmp_path / "oversized-native",
            source_manifest=manifest,
            filter_manifest=native_filter,
            protocol=SWE_PRBENCH_PROTOCOL_NATIVE,
            expected_source_manifest_digest=manifest.digest(),
        )
    assert not (tmp_path / "oversized-native").exists()


def test_empty_ground_truth_requires_explicit_fallback(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    annotation_path = (
        root / "dataset" / "annotations" / "dask__12221_human.json"
    )
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation["substantive_comment_ids"] = []
    annotation["substantive_comment_count"] = 0
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    manifest = _source_manifest(root)
    explicit_native = _filter_manifest(
        SWE_PRBENCH_PROTOCOL_NATIVE,
        source_profile=SWE_PRBENCH_SOURCE_PROFILE_EXPLICIT,
    )
    validation = validate_swe_prbench_source(
        root,
        source_manifest=manifest,
        filter_manifest=explicit_native,
        expected_source_manifest_digest=manifest.digest(),
    )
    assert validation.empty_ground_truth_task_ids == ("dask__12221",)

    fallback_filter = _filter_manifest(
        SWE_PRBENCH_PROTOCOL_NATIVE,
        source_profile=SWE_PRBENCH_SOURCE_PROFILE_EXPLICIT,
        fallback="initiating_comments",
    )
    result = prepare_swe_prbench(
        root,
        tmp_path / "fallback-native",
        source_manifest=manifest,
        filter_manifest=fallback_filter,
        protocol=SWE_PRBENCH_PROTOCOL_NATIVE,
        expected_source_manifest_digest=manifest.digest(),
    )
    assert len(_load_case(result).review_truth.expected_findings) == 3


def test_nullable_pr_diff_counts_and_comment_line_are_preserved_without_inference(
    tmp_path: Path,
) -> None:
    root = _copy_fixture(tmp_path)
    prs_path = root / "dataset" / "prs.jsonl"
    pr = json.loads(prs_path.read_text(encoding="utf-8"))
    pr["lines_added"] = None
    pr["lines_removed"] = None
    prs_path.write_bytes(json.dumps(pr).encode("utf-8") + b"\n")
    annotation_path = (
        root / "dataset" / "annotations" / "dask__12221_human.json"
    )
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation["comments"][1]["line"] = None
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    manifest = _source_manifest(root)
    filter_manifest = _filter_manifest(
        SWE_PRBENCH_PROTOCOL_NATIVE,
        source_profile=SWE_PRBENCH_SOURCE_PROFILE_EXPLICIT,
    )
    result = prepare_swe_prbench(
        root,
        tmp_path / "nullable-native",
        source_manifest=manifest,
        filter_manifest=filter_manifest,
        protocol=SWE_PRBENCH_PROTOCOL_NATIVE,
        expected_source_manifest_digest=manifest.digest(),
    )
    finding = next(
        item
        for item in _load_case(result).review_truth.expected_findings
        if item.claim == "Run most tests when psutil is not installed"
    )
    assert finding.locations[0].from_line is None
    assert finding.locations[0].to_line is None
    assert finding.locations[0].side is None


def test_explicit_parquet_never_falls_back_and_missing_pyarrow_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_fixture(tmp_path)
    raw_path = root / "dataset" / "prs.jsonl"
    parquet_path = root / "dataset" / "prs.parquet"
    raw_path.rename(parquet_path)
    manifest = _source_manifest(root, source_format=SWE_PRBENCH_SOURCE_PARQUET)
    filter_manifest = _filter_manifest(
        SWE_PRBENCH_PROTOCOL_NATIVE,
        source_profile=SWE_PRBENCH_SOURCE_PROFILE_EXPLICIT,
        source_format=SWE_PRBENCH_SOURCE_PARQUET,
    )
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ModuleNotFoundError("blocked by test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(
        PublicOptionalDependencyError,
        match=r"install review-agent\[eval-public\]",
    ):
        validate_swe_prbench_source(
            root,
            source_manifest=manifest,
            filter_manifest=filter_manifest,
            expected_source_manifest_digest=manifest.digest(),
        )


def test_parquet_schema_contract_pins_nullability_timestamp_and_nested_types() -> None:
    schema = copy.deepcopy(swe_adapter._SWE_PRBENCH_PARQUET_SCHEMA_DESCRIPTOR)
    swe_adapter._validate_parquet_schema_descriptor(schema)
    fields = {item["name"]: item for item in schema}
    assert fields["merged_at"]["type"] == {
        "kind": "timestamp",
        "unit": "s",
        "tz": None,
    }
    assert fields["rvs_breakdown"]["type"]["kind"] == "struct"
    comments = fields["human_review_comments"]["type"]
    assert comments["kind"] == "list"
    nested = {
        item["name"]: item
        for item in comments["value_field"]["type"]["fields"]
    }
    assert nested["line"]["nullable"] is True
    assert nested["line"]["type"] == {"kind": "int64"}
    assert nested["replyTo"]["type"]["fields"] == (
        {
            "name": "id",
            "nullable": True,
            "type": {"kind": "string"},
        },
    )

    drifted = copy.deepcopy(schema)
    next(item for item in drifted if item["name"] == "merged_at")["type"][
        "unit"
    ] = "ms"
    with pytest.raises(PublicFormatError, match="Arrow schema fingerprint"):
        swe_adapter._validate_parquet_schema_descriptor(drifted)

    drifted = copy.deepcopy(schema)
    comment_type = next(
        item for item in drifted if item["name"] == "human_review_comments"
    )["type"]
    next(
        item
        for item in comment_type["value_field"]["type"]["fields"]
        if item["name"] == "line"
    )["nullable"] = False
    with pytest.raises(PublicFormatError, match="Arrow schema fingerprint"):
        swe_adapter._validate_parquet_schema_descriptor(drifted)


class _FakeParquetMetadata:
    def __init__(
        self,
        *,
        num_rows: int,
        row_group_rows: tuple[int, ...],
        row_group_bytes: tuple[int, ...],
    ) -> None:
        self.num_rows = num_rows
        self.num_row_groups = len(row_group_rows)
        self._groups = tuple(
            SimpleNamespace(num_rows=rows, total_byte_size=size)
            for rows, size in zip(row_group_rows, row_group_bytes)
        )

    def row_group(self, index: int):
        return self._groups[index]


@pytest.mark.parametrize(
    ("metadata", "message"),
    (
        (
            _FakeParquetMetadata(
                num_rows=351, row_group_rows=(351,), row_group_bytes=(1024,)
            ),
            "row count",
        ),
        (
            _FakeParquetMetadata(
                num_rows=1,
                row_group_rows=tuple(0 for _ in range(351)),
                row_group_bytes=tuple(1 for _ in range(351)),
            ),
            "row-group count",
        ),
        (
            _FakeParquetMetadata(
                num_rows=1,
                row_group_rows=(1,),
                row_group_bytes=(swe_adapter._MAX_PARQUET_UNCOMPRESSED_BYTES + 1,),
            ),
            "uncompressed byte budget",
        ),
    ),
)
def test_parquet_metadata_is_bounded_before_row_reads(
    metadata: _FakeParquetMetadata, message: str
) -> None:
    with pytest.raises(PublicFormatError, match=message):
        swe_adapter._validate_parquet_metadata(metadata)


def test_parquet_parser_streams_small_batches_and_bounds_canonical_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = json.loads((FIXTURE_ROOT / "dataset" / "prs.jsonl").read_text("utf-8"))
    row["merged_at"] = _datetime.datetime.strptime(
        row["merged_at"], "%Y-%m-%dT%H:%M:%SZ"
    )

    class FakeBatch:
        num_rows = 1
        schema = object()

        def to_pylist(self):
            return [row]

    class FakeParquetFile:
        metadata = _FakeParquetMetadata(
            num_rows=1, row_group_rows=(1,), row_group_bytes=(1024,)
        )
        schema_arrow = object()

        def __init__(self, _reader) -> None:
            self.iteration_args = None

        def iter_batches(self, *, batch_size: int, use_threads: bool):
            self.iteration_args = (batch_size, use_threads)
            yield FakeBatch()

    reader_holder = {}

    def open_parquet(reader):
        opened = FakeParquetFile(reader)
        reader_holder["reader"] = opened
        return opened

    fake_pq = SimpleNamespace(ParquetFile=open_parquet)
    fake_pa = SimpleNamespace(BufferReader=lambda raw: raw, parquet=fake_pq)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyarrow":
            return fake_pa
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        swe_adapter,
        "_arrow_schema_descriptor",
        lambda _pa, _schema: copy.deepcopy(
            swe_adapter._SWE_PRBENCH_PARQUET_SCHEMA_DESCRIPTOR
        ),
    )
    records = swe_adapter._parse_parquet_prs(
        b"pinned-parquet-bytes", "prs_parquet", "dataset/prs.parquet"
    )
    assert len(records) == 1
    assert reader_holder["reader"].iteration_args == (
        swe_adapter._PARQUET_BATCH_ROWS,
        False,
    )

    monkeypatch.setattr(swe_adapter, "_MAX_PARQUET_CANONICAL_BYTES", 1)
    with pytest.raises(PublicFormatError, match="canonical byte budget"):
        swe_adapter._parse_parquet_prs(
            b"pinned-parquet-bytes", "prs_parquet", "dataset/prs.parquet"
        )

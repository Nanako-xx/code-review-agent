from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Mapping, Tuple
from unittest.mock import patch

import pytest

from review_agent_eval.adapters import aacr_bench as aacr_adapter
from review_agent_eval.adapters._public import (
    PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION,
    PUBLIC_SOURCE_MANIFEST_SCHEMA_VERSION,
    PublicDatasetError,
    PublicFilterManifest,
    PublicFormatError,
    PublicPreparationError,
    PublicSelector,
    PublicSourceIntegrityError,
    PublicSourceManifest,
    PublicStatistic,
    source_file_from_path,
)
from review_agent_eval.adapters.aacr_bench import (
    AACR_DATASET_ID,
    AACR_DATASET_VERSION,
    AACR_FIXTURE_DATASET_ID,
    AACR_FIXTURE_DATASET_VERSION,
    AACR_FIXTURE_PROFILE,
    AACR_FIXTURE_SOURCE_MANIFEST_DIGEST,
    AACR_FIXTURE_SOURCE_REVISION,
    AACR_LICENSE,
    AACR_NEGATIVE_ROLE,
    AACR_POSITIVE_ROLE,
    AACR_PROTOCOL_ID,
    AACR_OFFICIAL_REQUIRED_REJECTION_BINDINGS,
    AACR_REJECTION_SELECTOR,
    AACR_SOURCE_REVISION,
    AACR_SOURCE_URI,
    aacr_rejection_binding,
    prepare_aacr_bench,
)
from review_agent_eval.cases import CaseSplit
from review_agent_eval.cases import REPOSITORY_MATERIALIZER_PROTOCOL
from review_agent_eval.datasets import CaseBank
from review_agent_eval.models import (
    DiffSide,
    MetricAuthoritySource,
    NovelFindingPolicy,
    RequiredContextLevel,
    ReviewTargetKind,
    SchemaError,
    TruthCompleteness,
    canonical_sha256,
    stable_id,
)


FIXTURE_ROOT = (
    Path(__file__).parent / "fixtures" / "public_datasets" / "aacr"
)
VALID_ROOT = FIXTURE_ROOT / "valid"
FIXTURE_SOURCE_MANIFEST = VALID_ROOT / "source_manifest.json"
REVERSED_FIXTURE = FIXTURE_ROOT / "invalid" / "reversed-line-range.json"

_STATISTIC_NAMES = (
    "negative_comments",
    "negative_prs",
    "overlap_prs",
    "positive_comments",
    "positive_prs",
    "unique_prs",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _copy_source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    shutil.copytree(VALID_ROOT, root)
    return root


def _split_payloads(root: Path) -> Tuple[list, list]:
    return (
        _load(root / "dataset" / "positive_samples.json"),
        _load(root / "dataset" / "negative_samples.json"),
    )


def _write_splits(root: Path, positive: list, negative: list) -> None:
    _dump(root / "dataset" / "positive_samples.json", positive)
    _dump(root / "dataset" / "negative_samples.json", negative)


def _raw_statistics(positive: list, negative: list) -> Dict[str, int]:
    positive_urls = {item["githubPrUrl"] for item in positive}
    negative_urls = {item["githubPrUrl"] for item in negative}
    return {
        "positive_prs": len(positive),
        "negative_prs": len(negative),
        "positive_comments": sum(len(item["comments"]) for item in positive),
        "negative_comments": sum(len(item["comments"]) for item in negative),
        "unique_prs": len(positive_urls | negative_urls),
        "overlap_prs": len(positive_urls & negative_urls),
    }


def _fixed_source_manifest(root: Path = VALID_ROOT) -> PublicSourceManifest:
    return PublicSourceManifest.from_json((root / "source_manifest.json").read_bytes())


def _derived_source_manifest(
    root: Path,
    *,
    statistics: Mapping[str, int] | None = None,
) -> PublicSourceManifest:
    if statistics is None:
        statistics = _raw_statistics(*_split_payloads(root))
    assert set(statistics) == set(_STATISTIC_NAMES)
    return PublicSourceManifest(
        schema_version=PUBLIC_SOURCE_MANIFEST_SCHEMA_VERSION,
        dataset_id=AACR_FIXTURE_DATASET_ID,
        dataset_version=AACR_FIXTURE_DATASET_VERSION,
        source_uri=AACR_SOURCE_URI,
        source_revision=AACR_FIXTURE_SOURCE_REVISION,
        license=AACR_LICENSE,
        files=(
            source_file_from_path(
                root,
                role=AACR_POSITIVE_ROLE,
                path="dataset/positive_samples.json",
            ),
            source_file_from_path(
                root,
                role=AACR_NEGATIVE_ROLE,
                path="dataset/negative_samples.json",
            ),
        ),
        expected_statistics=tuple(
            PublicStatistic(name=name, value=statistics[name])
            for name in _STATISTIC_NAMES
        ),
    )


@contextmanager
def _trusted_test_manifest(manifest: PublicSourceManifest):
    """Install a test-only independent trust anchor for mutated parser inputs."""

    with patch.object(
        aacr_adapter,
        "AACR_FIXTURE_SOURCE_MANIFEST_DIGEST",
        manifest.digest(),
    ):
        yield


def _filter(
    *selectors: PublicSelector,
    dataset_id: str = AACR_FIXTURE_DATASET_ID,
) -> PublicFilterManifest:
    return PublicFilterManifest(
        schema_version=PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION,
        dataset_id=dataset_id,
        selectors=tuple(selectors),
    )


def _actual_statistics(result: Any) -> Dict[str, int]:
    return {item.name: item.value for item in result.receipt.actual_statistics}


def _dimensions(result: Any, task_id: str) -> Dict[str, str]:
    entry = result.manifest.case(task_id)
    return {item.name: item.value for item in entry.dimensions}


def _prepare_mutated_test_source(
    source: Path,
    manifest: PublicSourceManifest,
    filter_manifest: PublicFilterManifest,
    output: Path,
):
    with _trusted_test_manifest(manifest):
        return prepare_aacr_bench(source, manifest, filter_manifest, output)


def test_prepare_maps_official_records_and_publishes_case_bank(tmp_path: Path) -> None:
    source = _copy_source(tmp_path)
    source_manifest = _fixed_source_manifest(source)
    filter_manifest = _filter()

    first = prepare_aacr_bench(
        source, source_manifest, filter_manifest, tmp_path / "suite-one"
    )
    second = prepare_aacr_bench(
        source, source_manifest, filter_manifest, tmp_path / "suite-two"
    )

    assert first.manifest.digest() == second.manifest.digest()
    assert first.receipt == second.receipt
    assert len(first.manifest.cases) == 1
    entry = first.manifest.cases[0]
    assert entry.split is CaseSplit.CAPABILITY
    assert entry.protocol_id == AACR_PROTOCOL_ID
    assert first.manifest.wire_contract.review_target_kind is ReviewTargetKind.REPOSITORY
    assert (
        first.manifest.wire_contract.materializer_protocol
        == REPOSITORY_MATERIALIZER_PROTOCOL
    )

    bank = CaseBank.open(first.suite_root)
    case = bank.evaluator_case(entry.task_id)
    assert case.input.review_target.kind is ReviewTargetKind.REPOSITORY
    assert case.input.review_target.repository.base_revision == (
        "a050e422e23ce3eaee960b75ceff236b34f369b9"
    )
    assert case.input.review_target.repository.head_revision == (
        "0c65673a6fd2421be8fbe613116077120adea068"
    )
    assert case.input.review_target.repository.url == "https://github.com/FreeCAD/FreeCAD.git"
    assert case.input.review_target.review_request.title is None
    assert case.input.review_target.review_request.description is None
    assert case.intent_truth.scorable is False
    assert case.clarification_script.max_rounds == 4
    assert case.clarification_script.answers == ()
    assert case.review_truth.completeness is TruthCompleteness.EXPERT_AUGMENTED
    assert case.review_truth.novel_finding_policy is NovelFindingPolicy.VERIFY

    positive, negative = _split_payloads(source)
    expected = case.review_truth.expected_findings[0]
    assert expected.claim == positive[0]["comments"][0]["note"]
    assert expected.severity is None
    assert expected.metric_authority.severity_scorable is False
    assert expected.metric_authority.severity_authority is None
    assert expected.metric_authority.location_scorable is True
    assert (
        expected.metric_authority.location_authority
        is MetricAuthoritySource.UPSTREAM_ANNOTATION
    )
    assert expected.category == "Performance"
    assert expected.required is True
    assert expected.required_context_level is RequiredContextLevel.FILE
    assert expected.locations[0].path == (
        "src/Mod/TechDraw/App/DrawProjGroup.cpp"
    )
    assert expected.locations[0].side is DiffSide.RIGHT
    assert (expected.locations[0].from_line, expected.locations[0].to_line) == (
        1123,
        1126,
    )
    assert expected.truth_id == stable_id(
        "aacr-truth",
        AACR_POSITIVE_ROLE,
        "/0/comments/0",
        canonical_sha256(positive[0]["comments"][0]),
    )

    invalid = case.review_truth.known_invalid_findings[0]
    assert invalid.claim == negative[0]["comments"][0]["note"]
    assert invalid.category == "Code Defect"
    assert invalid.locations[0].side is DiffSide.RIGHT
    assert (invalid.locations[0].from_line, invalid.locations[0].to_line) == (
        1127,
        1132,
    )

    dimensions = _dimensions(first, case.task_id)
    assert dimensions["benchmark"] == "AACR-Bench"
    assert dimensions["benchmark_profile"] == AACR_FIXTURE_PROFILE
    assert dimensions["source_revision"] == AACR_FIXTURE_SOURCE_REVISION
    assert dimensions["completeness"] == "expert_augmented"
    assert dimensions["protocol"] == AACR_PROTOCOL_ID
    assert dimensions["language"] == "C++"
    assert dimensions["severity_source"] == "upstream_unavailable"
    assert dimensions["severity_metric_scope"] == "not_scorable"
    assert dimensions["location_metric_scope"] == "upstream_annotation"
    assert dimensions["isolated_truth_count"] == "0"
    assert dimensions["polarity_conflict_isolated_truth_count"] == "0"
    assert dimensions["reversed_line_range_isolated_truth_count"] == "0"
    assert dimensions["dataset_license_scope"] == "aacr_dataset_only"
    assert dimensions["underlying_repository_license"] == (
        "not_normalized_by_upstream"
    )

    assert len(first.receipt.records) == 4
    positive_receipt = next(
        item
        for item in first.receipt.records
        if item.source_role == AACR_POSITIVE_ROLE
        and item.truth_id is not None
    )
    assert json.loads(positive_receipt.record_json) == positive[0]["comments"][0]
    assert positive_receipt.truth_id == expected.truth_id
    assert positive_receipt.disposition == "expected_finding"
    assert "benchmark=AACR-Bench" in positive_receipt.reason
    assert "profile=fixture" in positive_receipt.reason
    assert "severity=not-scorable" in positive_receipt.reason
    assert "location=upstream-annotation" in positive_receipt.reason
    assert (
        "underlying_repository_license=not_normalized_by_upstream"
        in positive_receipt.reason
    )
    pr_receipts = [
        item for item in first.receipt.records if item.truth_id is None
    ]
    assert len(pr_receipts) == 2
    assert all(item.upstream_id == positive[0]["githubPrUrl"] for item in pr_receipts)
    assert all(json.loads(item.record_json)["comments"] for item in pr_receipts)

    statistics = _actual_statistics(first)
    assert statistics["selected_prs"] == 1
    assert statistics["expected_findings"] == 1
    assert statistics["known_invalid_findings"] == 1
    assert statistics["severity_unscorable_findings"] == 1
    assert statistics["pr_record_receipts"] == 2
    assert statistics["source_comments"] == 2
    assert statistics["source_scorable_comments"] == 2
    assert statistics["source_isolated_comments"] == 0
    assert statistics["selected_scorable_comments"] == 2
    assert all(
        "requires_eval_v2" not in item.record_json
        for item in first.receipt.records
    )

    with pytest.raises(PublicPreparationError, match="already exists"):
        prepare_aacr_bench(
            source, source_manifest, filter_manifest, first.suite_root
        )


def test_fixture_source_manifest_is_independently_pinned_and_not_self_signed(
    tmp_path: Path,
) -> None:
    manifest = _fixed_source_manifest()
    assert manifest.digest() == AACR_FIXTURE_SOURCE_MANIFEST_DIGEST
    assert FIXTURE_SOURCE_MANIFEST.read_bytes() == (
        manifest.to_json() + "\n"
    ).encode("utf-8")

    source = _copy_source(tmp_path)
    positive, negative = _split_payloads(source)
    positive[0]["comments"][0]["note"] += " changed"
    _write_splits(source, positive, negative)
    self_signed = _derived_source_manifest(source)
    assert self_signed.digest() != AACR_FIXTURE_SOURCE_MANIFEST_DIGEST

    with pytest.raises(PublicSourceIntegrityError, match="independently pinned"):
        prepare_aacr_bench(
            source,
            self_signed,
            _filter(),
            tmp_path / "self-signed-suite",
        )


def test_source_hash_drift_fails_before_parsing(tmp_path: Path) -> None:
    source = _copy_source(tmp_path)
    manifest = _fixed_source_manifest(source)
    path = source / "dataset" / "positive_samples.json"
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(PublicSourceIntegrityError, match="hash|size|byte limit"):
        prepare_aacr_bench(source, manifest, _filter(), tmp_path / "suite")


def test_source_manifest_requires_revision_and_pinned_identity(tmp_path: Path) -> None:
    source = _copy_source(tmp_path)
    manifest = _fixed_source_manifest(source)
    missing = manifest.to_dict()
    del missing["source_revision"]
    with pytest.raises(SchemaError, match="missing field"):
        PublicSourceManifest.from_dict(missing)

    mismatches = (
        replace(manifest, dataset_id="other-dataset"),
        replace(manifest, dataset_version="v2.0"),
        replace(manifest, source_uri="https://example.invalid/aacr"),
        replace(manifest, source_revision="main"),
        replace(manifest, license="MIT"),
    )
    for index, bad_manifest in enumerate(mismatches):
        with pytest.raises(PublicFormatError, match="pinned"):
            prepare_aacr_bench(
                source,
                bad_manifest,
                _filter(),
                tmp_path / ("suite-%d" % index),
            )


def test_fixture_hashes_cannot_masquerade_as_the_official_profile(
    tmp_path: Path,
) -> None:
    source = _copy_source(tmp_path)
    fixture_manifest = _fixed_source_manifest(source)
    masquerade = replace(
        fixture_manifest,
        dataset_id=AACR_DATASET_ID,
        dataset_version=AACR_DATASET_VERSION,
        source_revision=AACR_SOURCE_REVISION,
    )

    with pytest.raises(PublicFormatError, match="official profile.*size.*SHA-256"):
        prepare_aacr_bench(
            source,
            masquerade,
            _filter(dataset_id=AACR_DATASET_ID),
            tmp_path / "suite",
        )

    official_files = tuple(
        replace(
            item,
            size_bytes=(
                1_100_995
                if item.role == AACR_POSITIVE_ROLE
                else 496_162
            ),
            sha256=(
                "d8683cb240249bc4e0aff6428802bdffa7b7573ace600552cab1cd0cb7e905c9"
                if item.role == AACR_POSITIVE_ROLE
                else "c0601008ec5f444317143b0ee59d7f99a0bc2b45735710d25c2f1a305ee519d0"
            ),
        )
        for item in masquerade.files
    )
    count_masquerade = replace(masquerade, files=official_files)
    with pytest.raises(PublicFormatError, match="196/155/1505/640/200/151"):
        prepare_aacr_bench(
            source,
            count_masquerade,
            _filter(dataset_id=AACR_DATASET_ID),
            tmp_path / "count-suite",
        )


@pytest.mark.parametrize(
    "mutation, message",
    (
        ("missing_comments", "fields"),
        ("extra_comment_field", "fields"),
        ("unknown_language", "language"),
        ("unknown_pr_category", "PR category"),
        ("unknown_finding_category", "finding category"),
        ("unknown_context", "context"),
        ("unknown_side", "side"),
        ("null_path", "path"),
        ("ai_source_mismatch", "source_model"),
    ),
)
def test_upstream_schema_and_enums_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    source = _copy_source(tmp_path)
    positive, negative = _split_payloads(source)
    comment = positive[0]["comments"][0]
    if mutation == "missing_comments":
        del positive[0]["comments"]
    elif mutation == "extra_comment_field":
        comment["unexpected"] = "drift"
    elif mutation == "unknown_language":
        positive[0]["project_main_language"] = "Ruby"
    elif mutation == "unknown_pr_category":
        positive[0]["category"] = "Other"
    elif mutation == "unknown_finding_category":
        comment["category"] = "Style"
    elif mutation == "unknown_context":
        comment["context"] = "Function Level"
    elif mutation == "unknown_side":
        comment["side"] = "both"
    elif mutation == "null_path":
        comment["path"] = None
    elif mutation == "ai_source_mismatch":
        comment["source_model"] = ""
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(mutation)
    _write_splits(source, positive, negative)
    fallback_statistics = {
        "positive_prs": 1,
        "negative_prs": 1,
        "positive_comments": 1,
        "negative_comments": 1,
        "unique_prs": 1,
        "overlap_prs": 1,
    }
    manifest = _derived_source_manifest(
        source,
        statistics=(fallback_statistics if mutation == "missing_comments" else None),
    )

    with pytest.raises(PublicFormatError, match=message):
        _prepare_mutated_test_source(
            source, manifest, _filter(), tmp_path / "suite"
        )


def test_duplicate_pr_comment_and_overlap_metadata_conflict_fail(
    tmp_path: Path,
) -> None:
    source = _copy_source(tmp_path)
    positive, negative = _split_payloads(source)

    positive.append(deepcopy(positive[0]))
    _write_splits(source, positive, negative)
    manifest = _derived_source_manifest(source)
    with pytest.raises(PublicFormatError, match="duplicate PR"):
        _prepare_mutated_test_source(
            source, manifest, _filter(), tmp_path / "duplicate-pr"
        )

    source = _copy_source(tmp_path / "comment")
    positive, negative = _split_payloads(source)
    positive[0]["comments"].append(deepcopy(positive[0]["comments"][0]))
    _write_splits(source, positive, negative)
    manifest = _derived_source_manifest(source)
    with pytest.raises(PublicFormatError, match="duplicate comment"):
        _prepare_mutated_test_source(
            source,
            manifest,
            _filter(),
            tmp_path / "duplicate-comment",
        )

    source = _copy_source(tmp_path / "metadata")
    positive, negative = _split_payloads(source)
    negative[0]["category"] = "Bug Fix"
    _write_splits(source, positive, negative)
    manifest = _derived_source_manifest(source)
    with pytest.raises(PublicFormatError, match="metadata conflict"):
        _prepare_mutated_test_source(
            source,
            manifest,
            _filter(),
            tmp_path / "metadata-conflict",
        )


def test_nfc_polarity_conflict_requires_every_exact_binding_and_isolates_all(
    tmp_path: Path,
) -> None:
    source = _copy_source(tmp_path)
    positive, negative = _split_payloads(source)

    positive_one = deepcopy(positive[0]["comments"][0])
    positive_one.update(
        note="Polarity caf\u00e9 conflict",
        path="src/polarity-positive-one.cpp",
        from_line=201,
        to_line=202,
    )
    positive_two = deepcopy(positive_one)
    positive_two.update(
        note="Polarity cafe\u0301 conflict",
        path="src/polarity-positive-two.cpp",
        from_line=301,
        to_line=302,
    )
    negative_one = deepcopy(negative[0]["comments"][0])
    negative_one.update(
        note="Polarity caf\u00e9 conflict",
        path="src/polarity-negative.cpp",
        from_line=401,
        to_line=402,
    )
    positive[0]["comments"].extend((positive_one, positive_two))
    negative[0]["comments"].append(negative_one)
    _write_splits(source, positive, negative)
    manifest = _derived_source_manifest(source)

    with pytest.raises(PublicFormatError, match="NFC-equivalent.*polarity"):
        _prepare_mutated_test_source(
            source, manifest, _filter(), tmp_path / "unbound"
        )

    bindings = (
        aacr_rejection_binding(
            AACR_POSITIVE_ROLE, "/0/comments/1", positive_one
        ),
        aacr_rejection_binding(
            AACR_POSITIVE_ROLE, "/0/comments/2", positive_two
        ),
        aacr_rejection_binding(
            AACR_NEGATIVE_ROLE, "/0/comments/1", negative_one
        ),
    )
    with pytest.raises(PublicFormatError, match="every conflicting record"):
        _prepare_mutated_test_source(
            source,
            manifest,
            _filter(
                PublicSelector(
                    name=AACR_REJECTION_SELECTOR,
                    values=bindings[:-1],
                )
            ),
            tmp_path / "partially-bound",
        )

    accepted = _prepare_mutated_test_source(
        source,
        manifest,
        _filter(
            PublicSelector(name=AACR_REJECTION_SELECTOR, values=bindings)
        ),
        tmp_path / "bound",
    )
    case = CaseBank.open(accepted.suite_root).evaluator_case(
        accepted.manifest.cases[0].task_id
    )
    assert len(case.review_truth.expected_findings) == 1
    assert len(case.review_truth.known_invalid_findings) == 1
    assert case.review_truth.expected_findings[0].claim == (
        positive[0]["comments"][0]["note"]
    )
    assert case.review_truth.known_invalid_findings[0].claim == (
        negative[0]["comments"][0]["note"]
    )

    isolated = [
        item
        for item in accepted.receipt.records
        if item.disposition == "isolated_polarity_conflict"
    ]
    assert {item.record_pointer for item in isolated} == {
        "/0/comments/1",
        "/0/comments/2",
    }
    assert len(isolated) == 3
    assert {
        json.loads(item.record_json)["note"] for item in isolated
    } == {
        "Polarity caf\u00e9 conflict",
        "Polarity cafe\u0301 conflict",
    }
    assert all("no-polarity-selection" in item.reason for item in isolated)

    dimensions = _dimensions(accepted, case.task_id)
    assert dimensions["isolated_truth_count"] == "3"
    assert dimensions["polarity_conflict_isolated_truth_count"] == "3"
    assert dimensions["reversed_line_range_isolated_truth_count"] == "0"
    statistics = _actual_statistics(accepted)
    assert statistics["source_comments"] == 5
    assert statistics["source_scorable_comments"] == 2
    assert statistics["source_isolated_comments"] == 3
    assert statistics["polarity_conflict_isolated_comments"] == 3
    assert statistics["reversed_line_range_isolated_comments"] == 0
    assert statistics["selected_scorable_comments"] == 2
    assert statistics["expected_findings"] == 1
    assert statistics["known_invalid_findings"] == 1


def test_negative_only_and_empty_comment_prs_are_not_skipped_or_called_clean(
    tmp_path: Path,
) -> None:
    source = _copy_source(tmp_path)
    _positive, negative = _split_payloads(source)
    _write_splits(source, [], negative)
    manifest = _derived_source_manifest(source)
    result = _prepare_mutated_test_source(
        source, manifest, _filter(), tmp_path / "negative-only"
    )
    case = CaseBank.open(result.suite_root).evaluator_case(
        result.manifest.cases[0].task_id
    )
    assert case.review_truth.expected_findings == ()
    assert len(case.review_truth.known_invalid_findings) == 1
    assert _dimensions(result, case.task_id)["truth_profile"] == (
        "negative_only_not_clean"
    )

    source = _copy_source(tmp_path / "empty")
    positive, negative = _split_payloads(source)
    positive[0]["comments"] = []
    negative[0]["comments"] = []
    _write_splits(source, positive, negative)
    manifest = _derived_source_manifest(source)
    empty = _prepare_mutated_test_source(
        source, manifest, _filter(), tmp_path / "empty-comments"
    )
    empty_case = CaseBank.open(empty.suite_root).evaluator_case(
        empty.manifest.cases[0].task_id
    )
    assert empty_case.review_truth.expected_findings == ()
    assert empty_case.review_truth.known_invalid_findings == ()
    assert _dimensions(empty, empty_case.task_id)["truth_profile"] == (
        "zero_truth_not_clean"
    )
    empty_pr_receipts = [
        item
        for item in empty.receipt.records
        if item.task_id == empty_case.task_id and item.truth_id is None
    ]
    assert len(empty_pr_receipts) == 2
    assert all(json.loads(item.record_json)["comments"] == [] for item in empty_pr_receipts)


def _python_clone(record: Mapping[str, Any]) -> Dict[str, Any]:
    cloned = deepcopy(record)
    cloned["project_main_language"] = "Python"
    cloned["githubPrUrl"] = "https://github.com/example/python-project/pull/7"
    cloned["source_commit"] = "1" * 40
    cloned["target_commit"] = "2" * 40
    return cloned


def test_language_subset_is_explicit_and_all_exclusions_are_receipted(
    tmp_path: Path,
) -> None:
    source = _copy_source(tmp_path)
    positive, negative = _split_payloads(source)
    positive.append(_python_clone(positive[0]))
    negative.append(_python_clone(negative[0]))
    _write_splits(source, positive, negative)
    manifest = _derived_source_manifest(source)

    all_languages = _prepare_mutated_test_source(
        source, manifest, _filter(), tmp_path / "all-languages"
    )
    assert len(all_languages.manifest.cases) == 2

    python_only = _prepare_mutated_test_source(
        source,
        manifest,
        _filter(PublicSelector(name="language", values=("Python",))),
        tmp_path / "python-only",
    )
    assert len(python_only.manifest.cases) == 1
    task_id = python_only.manifest.cases[0].task_id
    assert _dimensions(python_only, task_id)["language"] == "Python"
    statistics = _actual_statistics(python_only)
    assert statistics["selected_prs"] == 1
    assert statistics["filtered_prs"] == 1
    assert statistics["source_comments"] == 4
    assert statistics["source_scorable_comments"] == 4
    assert statistics["selected_source_comments"] == 2
    assert statistics["selected_scorable_comments"] == 2
    assert statistics["filtered_source_comments"] == 2
    filtered = [
        item
        for item in python_only.receipt.records
        if item.disposition == "filtered_out"
    ]
    assert len(filtered) == 2
    assert all("language selector excluded C++" in item.reason for item in filtered)


@pytest.mark.parametrize(
    "selector",
    (
        PublicSelector(name="language", values=("Ruby",)),
        PublicSelector(name="implicit_product_eligibility", values=("Python",)),
    ),
)
def test_filter_manifest_rejects_unknown_or_implicit_selectors(
    tmp_path: Path, selector: PublicSelector
) -> None:
    source = _copy_source(tmp_path)
    with pytest.raises(PublicFormatError, match="selector|language"):
        prepare_aacr_bench(
            source,
            _fixed_source_manifest(source),
            _filter(selector),
            tmp_path / "suite",
        )


def test_filter_selector_cannot_hide_an_empty_language_selection() -> None:
    with pytest.raises(PublicDatasetError, match="explicit value"):
        PublicSelector(name="language", values=())


def test_reversed_range_requires_exact_pointer_and_digest_rejection(
    tmp_path: Path,
) -> None:
    source = _copy_source(tmp_path)
    positive, negative = _split_payloads(source)
    reversed_comment = _load(REVERSED_FIXTURE)["comments"][0]
    positive[0]["comments"].append(reversed_comment)
    _write_splits(source, positive, negative)
    manifest = _derived_source_manifest(source)

    with pytest.raises(PublicFormatError, match="reversed line range"):
        _prepare_mutated_test_source(
            source, manifest, _filter(), tmp_path / "unbound"
        )

    pointer = "/0/comments/1"
    binding = aacr_rejection_binding(
        AACR_POSITIVE_ROLE, pointer, reversed_comment
    )
    accepted = _prepare_mutated_test_source(
        source,
        manifest,
        _filter(
            PublicSelector(
                name=AACR_REJECTION_SELECTOR,
                values=(binding,),
            )
        ),
        tmp_path / "bound",
    )
    case = CaseBank.open(accepted.suite_root).evaluator_case(
        accepted.manifest.cases[0].task_id
    )
    assert len(case.review_truth.expected_findings) == 1
    rejected = [
        item
        for item in accepted.receipt.records
        if item.disposition == "isolated_reversed_line_range"
    ]
    assert len(rejected) == 1
    assert rejected[0].record_pointer == pointer
    assert json.loads(rejected[0].record_json) == reversed_comment
    assert "no-line-swapping" in rejected[0].reason
    dimensions = _dimensions(accepted, case.task_id)
    assert dimensions["isolated_truth_count"] == "1"
    assert dimensions["reversed_line_range_isolated_truth_count"] == "1"
    statistics = _actual_statistics(accepted)
    assert statistics["rejected_comments"] == 1
    assert statistics["source_comments"] == 3
    assert statistics["source_scorable_comments"] == 2
    assert statistics["source_isolated_comments"] == 1
    assert statistics["reversed_line_range_isolated_comments"] == 1
    assert statistics["polarity_conflict_isolated_comments"] == 0

    wrong_digest = (
        "%s#%s@%s" % (AACR_POSITIVE_ROLE, pointer, "0" * 64)
    )
    with pytest.raises(PublicFormatError, match="reversed line range"):
        _prepare_mutated_test_source(
            source,
            manifest,
            _filter(
                PublicSelector(
                    name=AACR_REJECTION_SELECTOR,
                    values=(wrong_digest,),
                )
            ),
            tmp_path / "wrong-digest",
        )

    valid_binding = aacr_rejection_binding(
        AACR_POSITIVE_ROLE, "/0/comments/0", positive[0]["comments"][0]
    )
    with pytest.raises(PublicFormatError, match="unused|reversed"):
        _prepare_mutated_test_source(
            source,
            manifest,
            _filter(
                PublicSelector(
                    name=AACR_REJECTION_SELECTOR,
                    values=(valid_binding,),
                )
            ),
            tmp_path / "valid-comment-rejection",
        )


def test_filter_dataset_must_match_aacr(tmp_path: Path) -> None:
    source = _copy_source(tmp_path)
    bad_filter = replace(_filter(), dataset_id="other-dataset")
    with pytest.raises(PublicFormatError, match="dataset_id"):
        prepare_aacr_bench(
            source,
            _fixed_source_manifest(source),
            bad_filter,
            tmp_path / "suite",
        )

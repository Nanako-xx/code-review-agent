from __future__ import annotations

import json
from pathlib import Path
import shutil
import socket
from typing import Any, Dict, Iterable

import pytest

from review_agent_eval.adapters import _public as public_module
from review_agent_eval.adapters._public import (
    PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION,
    PublicFilterManifest,
    PublicSelector,
    PublicSourceManifest,
    read_public_preparation_receipt,
)
from review_agent_eval.adapters.aacr_bench import (
    AACR_FIXTURE_DATASET_ID,
    AACR_PROTOCOL_ID,
)
from review_agent_eval.adapters.swe_prbench import (
    SWE_PRBENCH_DATASET_ID,
    SWE_PRBENCH_FROZEN_PROTOCOL_ID,
    SWE_PRBENCH_HARNESS_LICENSE,
    SWE_PRBENCH_HARNESS_REVISION,
    SWE_PRBENCH_PIPELINE_VERSION,
    SWE_PRBENCH_PROTOCOL_FROZEN,
    SWE_PRBENCH_PROTOCOL_NATIVE,
    SWE_PRBENCH_SOURCE_PROFILE_FIXTURE,
)
from review_agent_eval.cases import ReviewTargetKind
from review_agent_eval.cli import (
    EXIT_CONFLICT,
    EXIT_INTEGRITY,
    EXIT_OK,
    EXIT_OPERATIONAL,
    EXIT_PRECONDITION,
    EXIT_USAGE,
    main,
)
from review_agent_eval.datasets import CaseBank
from review_agent_eval.models import canonical_json_bytes, canonical_sha256


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "public_datasets"
AACR_ROOT = FIXTURE_ROOT / "aacr" / "valid"
SWE_ROOT = FIXTURE_ROOT / "swe_prbench"


def _copy_source(tmp_path: Path, root: Path, name: str = "source") -> Path:
    target = tmp_path / name
    shutil.copytree(root, target)
    # The checked-in adapter fixtures predate the CLI control-file boundary;
    # make the copied control file byte-for-byte canonical for this contract.
    manifest_path = target / "source_manifest.json"
    manifest = PublicSourceManifest.from_json(manifest_path.read_bytes())
    manifest_path.write_bytes(canonical_json_bytes(manifest.to_dict()))
    return target


def _load_source_manifest(root: Path) -> PublicSourceManifest:
    return PublicSourceManifest.from_json(
        (root / "source_manifest.json").read_bytes()
    )


def _write_filter(path: Path, *, dataset_id: str, selectors: Iterable[PublicSelector]) -> PublicFilterManifest:
    manifest = PublicFilterManifest(
        schema_version=PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION,
        dataset_id=dataset_id,
        selectors=tuple(selectors),
    )
    path.write_bytes(canonical_json_bytes(manifest.to_dict()))
    return manifest


def _swe_filter(path: Path, protocol: str) -> PublicFilterManifest:
    config = "none" if protocol == SWE_PRBENCH_PROTOCOL_NATIVE else "config_A"
    return _write_filter(
        path,
        dataset_id=SWE_PRBENCH_DATASET_ID,
        selectors=(
            PublicSelector("source_scope", ("fixture",)),
            PublicSelector("source_profile", (SWE_PRBENCH_SOURCE_PROFILE_FIXTURE,)),
            PublicSelector("source_format", ("raw_jsonl",)),
            PublicSelector("protocol", (protocol,)),
            PublicSelector("context_config", (config,)),
            PublicSelector("harness_revision", (SWE_PRBENCH_HARNESS_REVISION,)),
            PublicSelector("harness_license", (SWE_PRBENCH_HARNESS_LICENSE,)),
            PublicSelector("pipeline_version", (SWE_PRBENCH_PIPELINE_VERSION,)),
        ),
    )


def _aacr_filter(path: Path) -> PublicFilterManifest:
    return _write_filter(
        path,
        dataset_id=AACR_FIXTURE_DATASET_ID,
        selectors=(),
    )


def _args(
    *,
    dataset: str,
    source_root: Path,
    source_manifest: Path,
    source_digest: str,
    filter_manifest: Path,
    profile_digest: str,
    output_root: Path,
    json_output: bool = True,
) -> list[str]:
    result = [
        "prepare-public",
        "--mode",
        "local-import",
        "--dataset",
        dataset,
        "--source-root",
        str(source_root),
        "--source-manifest",
        str(source_manifest),
        "--expected-source-manifest-digest",
        source_digest,
        "--filter-manifest",
        str(filter_manifest),
        "--expected-profile-digest",
        profile_digest,
        "--output-root",
        str(output_root),
    ]
    if json_output:
        result.append("--json")
    return result


def _run(capsys: pytest.CaptureFixture[str], args: list[str]) -> Dict[str, Any]:
    code = main(args)
    captured = capsys.readouterr()
    assert captured.out
    return {"code": code, "payload": json.loads(captured.out), "stderr": captured.err}


def _all_output_bytes(root: Path) -> bytes:
    chunks = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            chunks.append(path.read_bytes())
    return b"\n".join(chunks)


def test_aacr_local_import_emits_complete_metadata_and_bound_digests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _copy_source(tmp_path, AACR_ROOT)
    source_manifest = _load_source_manifest(source)
    filter_path = tmp_path / "aacr-filter.json"
    filter_manifest = _aacr_filter(filter_path)
    output = tmp_path / "aacr-suite"

    result = _run(
        capsys,
        _args(
            dataset="aacr-bench",
            source_root=source,
            source_manifest=source / "source_manifest.json",
            source_digest=source_manifest.digest(),
            filter_manifest=filter_path,
            profile_digest=filter_manifest.digest(),
            output_root=output,
        ),
    )

    assert result["code"] == EXIT_OK, result
    payload = result["payload"]
    receipt = read_public_preparation_receipt(output)
    suite = CaseBank.open(output).manifest
    assert payload["dataset_id"] == source_manifest.dataset_id
    assert payload["dataset_version"] == source_manifest.dataset_version
    assert payload["source_revision"] == source_manifest.source_revision
    assert payload["source_uri"] == source_manifest.source_uri
    assert payload["license"] == source_manifest.license
    assert payload["source_manifest_digest"] == source_manifest.digest()
    assert payload["profile_digest"] == filter_manifest.digest()
    assert payload["filter_manifest_digest"] == filter_manifest.digest()
    assert payload["protocol"] == AACR_PROTOCOL_ID
    assert payload["target_kind"] == ReviewTargetKind.REPOSITORY.value
    assert payload["wire_contract_digest"] == suite.wire_contract.digest()
    assert payload["suite_id"] == suite.suite_id
    assert payload["suite_version"] == suite.suite_version
    assert payload["suite_manifest_digest"] == receipt.suite_manifest_digest
    assert payload["preparation_packet_digest"] == receipt.preparation_packet_digest
    assert payload["preparation_receipt_digest"] == receipt.digest()
    assert payload["preparation_receipt_relative_name"] == "preparation_receipt.json"


@pytest.mark.parametrize("protocol", (SWE_PRBENCH_PROTOCOL_NATIVE, SWE_PRBENCH_PROTOCOL_FROZEN))
def test_swe_local_import_derives_protocol_from_filter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], protocol: str
) -> None:
    source = _copy_source(tmp_path, SWE_ROOT)
    source_manifest = _load_source_manifest(source)
    filter_path = tmp_path / ("swe-%s-filter.json" % protocol)
    filter_manifest = _swe_filter(filter_path, protocol)
    output = tmp_path / ("swe-%s-suite" % protocol)

    result = _run(
        capsys,
        _args(
            dataset="swe-prbench",
            source_root=source,
            source_manifest=source / "source_manifest.json",
            source_digest=source_manifest.digest(),
            filter_manifest=filter_path,
            profile_digest=filter_manifest.digest(),
            output_root=output,
        ),
    )
    assert result["code"] == EXIT_OK
    payload = result["payload"]
    receipt = read_public_preparation_receipt(output)
    assert payload["protocol"] == protocol
    assert payload["source_manifest_digest"] == receipt.source_manifest_digest
    assert payload["profile_digest"] == receipt.filter_manifest_digest
    if protocol == SWE_PRBENCH_PROTOCOL_FROZEN:
        from review_agent_eval.adapters.swe_prbench import read_swe_prbench_frozen_bundle

        assert payload["bundle_id"]
        assert payload["target_kind"] == ReviewTargetKind.FROZEN_CONTEXT.value
        assert payload["protocol_id"] == SWE_PRBENCH_FROZEN_PROTOCOL_ID
        bundle = read_swe_prbench_frozen_bundle(
            output / "frozen_bundle",
            expected_bundle_id=payload["bundle_id"],
        )
        assert bundle.manifest.bundle_id == payload["bundle_id"]
        assert any(
            item.path.startswith("frozen_bundle/")
            for item in receipt.extra_files
        )
    else:
        assert payload["target_kind"] == ReviewTargetKind.REPOSITORY.value


@pytest.mark.parametrize("anchor", ("source", "profile"))
def test_digest_anchor_mismatch_fails_before_output_creation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], anchor: str
) -> None:
    source = _copy_source(tmp_path, AACR_ROOT)
    source_manifest = _load_source_manifest(source)
    filter_path = tmp_path / "filter.json"
    filter_manifest = _aacr_filter(filter_path)
    output = tmp_path / ("failed-%s" % anchor)
    args = _args(
        dataset="aacr-bench",
        source_root=source,
        source_manifest=source / "source_manifest.json",
        source_digest=("0" * 64 if anchor == "source" else source_manifest.digest()),
        filter_manifest=filter_path,
        profile_digest=("0" * 64 if anchor == "profile" else filter_manifest.digest()),
        output_root=output,
    )
    result = _run(capsys, args)
    assert result["code"] == EXIT_INTEGRITY
    assert result["payload"]["error_code"] == "integrity"
    assert not output.exists()


def test_manifest_filter_drift_fails_closed_without_reading_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _copy_source(tmp_path, AACR_ROOT)
    source_manifest = _load_source_manifest(source)
    filter_path = tmp_path / "filter.json"
    filter_manifest = _aacr_filter(filter_path)
    filter_path.write_bytes(
        canonical_json_bytes(
            PublicFilterManifest(
                schema_version=filter_manifest.schema_version,
                dataset_id="swe-prbench",
                selectors=filter_manifest.selectors,
            ).to_dict()
        )
    )
    output = tmp_path / "drifted"
    result = _run(
        capsys,
        _args(
            dataset="aacr-bench",
            source_root=source,
            source_manifest=source / "source_manifest.json",
            source_digest=source_manifest.digest(),
            filter_manifest=filter_path,
            profile_digest=filter_manifest.digest(),
            output_root=output,
        ),
    )
    assert result["code"] == EXIT_INTEGRITY
    assert not output.exists()


def test_create_only_rerun_preserves_original_output_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _copy_source(tmp_path, AACR_ROOT)
    source_manifest = _load_source_manifest(source)
    filter_path = tmp_path / "filter.json"
    filter_manifest = _aacr_filter(filter_path)
    output = tmp_path / "suite"
    args = _args(
        dataset="aacr-bench",
        source_root=source,
        source_manifest=source / "source_manifest.json",
        source_digest=source_manifest.digest(),
        filter_manifest=filter_path,
        profile_digest=filter_manifest.digest(),
        output_root=output,
    )
    assert _run(capsys, args)["code"] == EXIT_OK
    before = _all_output_bytes(output)
    rerun = _run(capsys, args)
    assert rerun["code"] == EXIT_CONFLICT
    assert _all_output_bytes(output) == before


def test_cli_publish_rename_race_is_conflict_and_preserves_competitor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _copy_source(tmp_path, AACR_ROOT)
    source_manifest = _load_source_manifest(source)
    filter_path = tmp_path / "filter.json"
    filter_manifest = _aacr_filter(filter_path)
    output = tmp_path / "suite"
    real_publish = public_module._publish_directory_create_only

    def race(staging: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "competitor.bin").write_bytes(b"competitor")
        real_publish(staging, destination)

    monkeypatch.setattr(public_module, "_publish_directory_create_only", race)
    result = _run(
        capsys,
        _args(
            dataset="aacr-bench",
            source_root=source,
            source_manifest=source / "source_manifest.json",
            source_digest=source_manifest.digest(),
            filter_manifest=filter_path,
            profile_digest=filter_manifest.digest(),
            output_root=output,
        ),
    )
    assert result["code"] == EXIT_CONFLICT
    assert (output / "competitor.bin").read_bytes() == b"competitor"


def test_cli_missing_output_parent_is_precondition(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _copy_source(tmp_path, AACR_ROOT)
    source_manifest = _load_source_manifest(source)
    filter_path = tmp_path / "filter.json"
    filter_manifest = _aacr_filter(filter_path)
    output = tmp_path / "missing" / "suite"
    result = _run(
        capsys,
        _args(
            dataset="aacr-bench",
            source_root=source,
            source_manifest=source / "source_manifest.json",
            source_digest=source_manifest.digest(),
            filter_manifest=filter_path,
            profile_digest=filter_manifest.digest(),
            output_root=output,
        ),
    )
    assert result["code"] == EXIT_PRECONDITION
    assert not output.exists()


def test_cli_publication_io_failure_is_operational(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _copy_source(tmp_path, AACR_ROOT)
    source_manifest = _load_source_manifest(source)
    filter_path = tmp_path / "filter.json"
    filter_manifest = _aacr_filter(filter_path)
    output = tmp_path / "suite"

    def fail_publish(_staging: Path, _output: Path) -> None:
        raise public_module.PublicOperationalError("publication I/O failed")

    monkeypatch.setattr(public_module, "_publish_directory_create_only", fail_publish)
    result = _run(
        capsys,
        _args(
            dataset="aacr-bench",
            source_root=source,
            source_manifest=source / "source_manifest.json",
            source_digest=source_manifest.digest(),
            filter_manifest=filter_path,
            profile_digest=filter_manifest.digest(),
            output_root=output,
        ),
    )
    assert result["code"] == EXIT_OPERATIONAL
    assert not output.exists()


def test_rename_is_final_fallible_publication_boundary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _copy_source(tmp_path, AACR_ROOT)
    source_manifest = _load_source_manifest(source)
    filter_path = tmp_path / "filter.json"
    filter_manifest = _aacr_filter(filter_path)
    output = tmp_path / "suite"
    real_open = public_module.CaseBank.open

    def staging_only(root: Path, *args: object, **kwargs: object):
        if Path(root) == output:
            raise AssertionError("post-commit CaseBank verification ran")
        return real_open(root, *args, **kwargs)

    monkeypatch.setattr(public_module.CaseBank, "open", staticmethod(staging_only))
    result = _run(
        capsys,
        _args(
            dataset="aacr-bench",
            source_root=source,
            source_manifest=source / "source_manifest.json",
            source_digest=source_manifest.digest(),
            filter_manifest=filter_path,
            profile_digest=filter_manifest.digest(),
            output_root=output,
        ),
    )
    assert result["code"] == EXIT_OK
    assert output.is_dir()


@pytest.mark.parametrize("kind", ("oversized", "noncanonical", "symlink"))
def test_control_reader_failures_leave_no_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    source = _copy_source(tmp_path, AACR_ROOT)
    source_manifest = _load_source_manifest(source)
    filter_path = tmp_path / "filter.json"
    filter_manifest = _aacr_filter(filter_path)
    supplied_filter = filter_path
    if kind == "oversized":
        filter_path.write_bytes(b"{" + b" " * public_module.MAX_PUBLIC_FILTER_MANIFEST_BYTES)
    elif kind == "noncanonical":
        filter_path.write_bytes(canonical_json_bytes(filter_manifest.to_dict()) + b"\n")
    else:
        supplied_filter = tmp_path / "filter-link.json"
        try:
            supplied_filter.symlink_to(filter_path)
        except OSError as exc:  # pragma: no cover - Windows privilege dependent
            pytest.skip("file symlink is unavailable: %s" % exc)
    output = tmp_path / "suite"
    result = _run(
        capsys,
        _args(
            dataset="aacr-bench",
            source_root=source,
            source_manifest=source / "source_manifest.json",
            source_digest=source_manifest.digest(),
            filter_manifest=supplied_filter,
            profile_digest=filter_manifest.digest(),
            output_root=output,
        ),
    )
    assert result["code"] == EXIT_INTEGRITY
    assert not output.exists()


def test_output_and_result_do_not_contain_host_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _copy_source(tmp_path, SWE_ROOT)
    source_manifest = _load_source_manifest(source)
    filter_path = tmp_path / "filter.json"
    filter_manifest = _swe_filter(filter_path, SWE_PRBENCH_PROTOCOL_NATIVE)
    output = tmp_path / "suite"
    result = _run(
        capsys,
        _args(
            dataset="swe-prbench",
            source_root=source,
            source_manifest=source / "source_manifest.json",
            source_digest=source_manifest.digest(),
            filter_manifest=filter_path,
            profile_digest=filter_manifest.digest(),
            output_root=output,
        ),
    )
    captured = json.dumps(result["payload"], sort_keys=True).encode("utf-8")
    artifacts = _all_output_bytes(output)
    for forbidden in (str(source), str(output), str(source_manifest_path := source / "source_manifest.json")):
        assert forbidden.lower().encode("utf-8") not in captured.lower()
        assert forbidden.lower().encode("utf-8") not in artifacts.lower()


def test_prepare_public_does_not_construct_network_or_trial_clients(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _copy_source(tmp_path, AACR_ROOT)
    source_manifest = _load_source_manifest(source)
    filter_path = tmp_path / "filter.json"
    filter_manifest = _aacr_filter(filter_path)
    output = tmp_path / "suite"

    def forbidden_socket(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("prepare-public opened a network socket")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    result = _run(
        capsys,
        _args(
            dataset="aacr-bench",
            source_root=source,
            source_manifest=source / "source_manifest.json",
            source_digest=source_manifest.digest(),
            filter_manifest=filter_path,
            profile_digest=filter_manifest.digest(),
            output_root=output,
        ),
    )
    assert result["code"] == EXIT_OK


def test_prepare_public_mode_and_required_arguments_are_usage_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["prepare-public", "--mode", "pinned-download"]) == EXIT_USAGE
    capsys.readouterr()
    assert main(["prepare-public", "--mode", "local-import"]) == EXIT_USAGE
    capsys.readouterr()

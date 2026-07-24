from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from review_agent_eval.adapters.swe_prbench import (
    SWE_PRBENCH_PROTOCOL_FROZEN,
    FrozenContextEnvelope,
    prepare_swe_prbench_frozen_bundle,
)
from review_agent_eval.adapters.subprocess_agent import (
    subprocess_adapter_capabilities,
)
from review_agent_eval.artifacts import TrialManifest
from review_agent_eval.cases import (
    CaseSplit,
    FROZEN_CONTEXT_MATERIALIZER_PROTOCOL,
    PublicSuitePreparationBindingV2,
    SuiteCase,
    WireContractV2,
)
from review_agent_eval.config import (
    derive_case_path_id,
    derive_trial_id,
    derive_trial_seed,
)
from review_agent_eval.frozen_context import (
    FROZEN_CONTEXT_TARGET_PATH,
    FrozenContextTargetMaterializer,
    frozen_bundle_trust_digest,
    frozen_context_record_id,
    frozen_context_source_binding_digest,
    frozen_materialization_workspace,
    open_frozen_context_replay,
)
from review_agent_eval.materialization import (
    MaterializationError,
    MaterializationRequest,
)
from review_agent_eval.models import (
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    EvalInput,
    ReviewTargetKind,
    TrialStatus,
    TruthCompleteness,
    canonical_sha256,
    stable_id,
)

from .test_swe_prbench_adapter import (
    FIXTURE_ROOT,
    _filter_manifest,
    _source_manifest,
)


def _prepared_bundle(root: Path):
    return prepare_swe_prbench_frozen_bundle(
        FIXTURE_ROOT,
        root / "bundle",
        source_manifest=_source_manifest(FIXTURE_ROOT),
        filter_manifest=_filter_manifest(
            SWE_PRBENCH_PROTOCOL_FROZEN,
            context_config="config_B",
        ),
    )


def _eval_input(prepared, **changes) -> EvalInput:
    binding = prepared.manifest.records[0]
    target = {
        "kind": "frozen_context",
        "bundle_id": prepared.manifest.bundle_id,
        "record_id": frozen_context_record_id(binding),
        "context_format": "rendered_text",
        "rendered_sha256": binding.rendered_sha256,
        "rendered_utf8_bytes": binding.rendered_utf8_bytes,
        "source_binding_digest": frozen_context_source_binding_digest(
            prepared,
            binding,
        ),
    }
    target.update(changes)
    return EvalInput.from_dict(
        {
            "schema_version": EVAL_INPUT_SCHEMA_VERSION,
            "task_id": binding.task_id,
            "review_target": target,
        }
    )


def _preparation_binding(prepared) -> PublicSuitePreparationBindingV2:
    provisional = PublicSuitePreparationBindingV2.from_dict(
        {
            "schema_version": "public_suite_preparation_binding_v2",
            "source_catalog_digest": canonical_sha256(
                {"fixture": "source-catalog"}
            ),
            "acquisition_receipt_digest": canonical_sha256(
                {"fixture": "acquisition-receipt"}
            ),
            "source_manifest_digest": prepared.manifest.source_manifest_digest,
            "filter_manifest_digest": prepared.manifest.filter_manifest_digest,
            "preparation_packet_digest": canonical_sha256(
                {"fixture": "preparation-packet"}
            ),
            "repository_catalog_digest": None,
            "frozen_bundle_trust_digest": "0" * 64,
        }
    )
    return replace(
        provisional,
        frozen_bundle_trust_digest=frozen_bundle_trust_digest(
            prepared,
            provisional,
        ),
    )


def _request(eval_input: EvalInput, prepared) -> MaterializationRequest:
    capabilities = subprocess_adapter_capabilities()
    wire = WireContractV2(
        case_schema_version=EVAL_CASE_SCHEMA_VERSION,
        input_schema_version=EVAL_INPUT_SCHEMA_VERSION,
        submission_schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
        review_target_kind=ReviewTargetKind.FROZEN_CONTEXT,
        materializer_protocol=FROZEN_CONTEXT_MATERIALIZER_PROTOCOL,
    )
    run_id = stable_id("run", "frozen-context-tests", eval_input.digest())
    trial_id = derive_trial_id(run_id, eval_input.task_id, 1)
    case_digest = canonical_sha256(
        {"task_id": eval_input.task_id, "input": eval_input.to_dict()}
    )
    preparation = _preparation_binding(prepared)
    preparation_digest = preparation.digest()
    trial = TrialManifest(
        schema_version=TrialManifest.SCHEMA_VERSION,
        run_id=run_id,
        task_id=eval_input.task_id,
        case_path_id=derive_case_path_id(eval_input.task_id),
        canonical_case_digest=case_digest,
        eval_input_digest=eval_input.digest(),
        wire_contract=wire,
        target_kind=ReviewTargetKind.FROZEN_CONTEXT,
        materializer_protocol=FROZEN_CONTEXT_MATERIALIZER_PROTOCOL,
        suite_preparation_binding_digest=preparation_digest,
        adapter_capabilities_digest=capabilities.digest(),
        trial_id=trial_id,
        trial_index=1,
        seed=derive_trial_seed(run_id, eval_input.task_id, 1),
        agent_config_digest="a" * 64,
        initial_evaluator_execution_digest="b" * 64,
    )
    suite_case = SuiteCase(
        task_id=eval_input.task_id,
        case_version=1,
        path="cases/frozen.json",
        split=CaseSplit.CAPABILITY,
        protocol_id="official_frozen_context",
        dimensions=(),
        raw_file_size_bytes=1,
        raw_file_sha256="c" * 64,
        canonical_case_digest=case_digest,
        eval_input_digest=eval_input.digest(),
        truth_completeness=TruthCompleteness.CLOSED_WORLD,
    )
    return MaterializationRequest(
        eval_input=eval_input,
        trial_manifest=trial,
        suite_case=suite_case,
        attempt=1,
        wire_contract=wire,
        suite_preparation_binding=preparation,
        suite_preparation_binding_digest=preparation_digest,
        adapter_capabilities=capabilities,
    )


def test_frozen_materializer_preserves_exact_rendered_bytes_and_relative_access(
    tmp_path: Path,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    eval_input = _eval_input(prepared)
    materialized = FrozenContextTargetMaterializer(
        bundle_root=prepared.root,
        workspace_root=tmp_path / ".eval-workspaces",
    ).materialize(_request(eval_input, prepared))
    workspace = frozen_materialization_workspace(materialized)
    binding = prepared.manifest.records[0]
    envelope = FrozenContextEnvelope.from_json(
        (prepared.root / binding.path).read_bytes()
    )
    exact = envelope.record.rendered.encode("utf-8", "strict")

    with materialized:
        assert (workspace / FROZEN_CONTEXT_TARGET_PATH).read_bytes() == exact
        assert materialized.replay.read_exact() == exact
        assert materialized.replay.context_ref == eval_input.review_target.record_id
        assert materialized.replay.read_lines(1, 1) == exact.split(b"\n", 1)[0] + (
            b"\n" if b"\n" in exact else b""
        )
        assert materialized.target_access.readable_relative_paths == (
            FROZEN_CONTEXT_TARGET_PATH,
        )
        assert all(
            not Path(path).is_absolute()
            for path in materialized.target_access.readable_relative_paths
        )
        assert "manifest" not in " ".join(
            materialized.target_access.readable_relative_paths
        )
        assert hashlib.sha256(exact).hexdigest() == (
            materialized.manifest.files[0].sha256
        )
    assert materialized.closed
    assert not workspace.exists()


def test_open_frozen_replay_reuses_verified_local_bundle_without_workspace(
    tmp_path: Path,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    eval_input = _eval_input(prepared)
    request = _request(eval_input, prepared)

    replay = open_frozen_context_replay(
        bundle_root=prepared.root,
        eval_input=eval_input,
        suite_preparation_binding=request.suite_preparation_binding,
        suite_preparation_binding_digest=(
            request.suite_preparation_binding_digest
        ),
    )

    assert replay.bundle_id == eval_input.review_target.bundle_id
    assert replay.source_binding_digest == (
        eval_input.review_target.source_binding_digest
    )
    assert hashlib.sha256(replay.read_exact()).hexdigest() == (
        eval_input.review_target.rendered_sha256
    )
    assert not (tmp_path / ".eval-workspaces").exists()


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"bundle_id": "wrong-bundle"}, "bundle trust"),
        ({"record_id": "wrong-record"}, "record identity"),
        ({"rendered_sha256": "0" * 64}, "record binding|rendered"),
        ({"rendered_utf8_bytes": 1}, "record binding|rendered"),
        ({"source_binding_digest": "1" * 64}, "source binding"),
    ],
)
def test_frozen_materializer_rejects_target_and_trust_drift(
    tmp_path: Path,
    changes: dict[str, object],
    match: str,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    materializer = FrozenContextTargetMaterializer(
        bundle_root=prepared.root,
        workspace_root=tmp_path / ".eval-workspaces",
    )

    with pytest.raises(MaterializationError, match=match):
        materializer.materialize(
            _request(_eval_input(prepared, **changes), prepared)
        )


def test_frozen_materialization_detects_agent_visible_target_replacement(
    tmp_path: Path,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    materialized = FrozenContextTargetMaterializer(
        bundle_root=prepared.root,
        workspace_root=tmp_path / ".eval-workspaces",
    ).materialize(_request(_eval_input(prepared), prepared))
    workspace = frozen_materialization_workspace(materialized)
    context = workspace / FROZEN_CONTEXT_TARGET_PATH
    context.chmod(0o600)
    context.write_bytes(b"tampered")

    with pytest.raises(MaterializationError, match="Target drifted"):
        with materialized:
            pass
    assert materialized.closed


def test_frozen_workspace_cleanup_never_follows_agent_created_links(
    tmp_path: Path,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    materialized = FrozenContextTargetMaterializer(
        bundle_root=prepared.root,
        workspace_root=tmp_path / ".eval-workspaces",
    ).materialize(_request(_eval_input(prepared), prepared))
    workspace = frozen_materialization_workspace(materialized)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("must survive cleanup", encoding="utf-8")
    link = workspace / "work" / "outside-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        materialized.close(TrialStatus.FAILED)
        pytest.skip("symlink creation is not permitted: %s" % exc)

    materialized.close(TrialStatus.COMPLETED)

    assert materialized.closed
    assert not workspace.exists()
    assert sentinel.read_text(encoding="utf-8") == "must survive cleanup"


def test_frozen_workspace_cleanup_rejects_replacement_directory_without_deleting_it(
    tmp_path: Path,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    materialized = FrozenContextTargetMaterializer(
        bundle_root=prepared.root,
        workspace_root=tmp_path / ".eval-workspaces",
    ).materialize(_request(_eval_input(prepared), prepared))
    workspace = frozen_materialization_workspace(materialized)
    held = workspace.with_name("held-owned-workspace")
    replacement = workspace.with_name("replacement-workspace")
    workspace.rename(held)
    workspace.mkdir()
    sentinel = workspace / "sentinel.txt"
    sentinel.write_text("replacement must survive", encoding="utf-8")

    try:
        with pytest.raises(MaterializationError, match="identity"):
            materialized.close(TrialStatus.FAILED)
        assert not materialized.closed
        assert sentinel.read_text(encoding="utf-8") == "replacement must survive"
    finally:
        workspace.rename(replacement)
        held.rename(workspace)
        materialized.close(TrialStatus.FAILED)

    assert (replacement / "sentinel.txt").read_text(encoding="utf-8") == (
        "replacement must survive"
    )


def test_frozen_workspace_close_failure_does_not_mark_lease_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import review_agent_eval.frozen_context as frozen_module

    prepared = _prepared_bundle(tmp_path)
    materialized = FrozenContextTargetMaterializer(
        bundle_root=prepared.root,
        workspace_root=tmp_path / ".eval-workspaces",
    ).materialize(_request(_eval_input(prepared), prepared))
    workspace = frozen_materialization_workspace(materialized)
    real_remove = frozen_module._remove_tree_safely

    def fail_remove(active_root: Path, path: Path) -> None:
        del active_root, path
        raise PermissionError("cleanup-secret-must-not-leak")

    monkeypatch.setattr(frozen_module, "_remove_tree_safely", fail_remove)
    try:
        with pytest.raises(PermissionError):
            materialized.close(TrialStatus.FAILED)
        assert not materialized.closed
        assert workspace.exists()
    finally:
        monkeypatch.setattr(frozen_module, "_remove_tree_safely", real_remove)
        materialized.close(TrialStatus.FAILED)


def test_frozen_replay_rejects_bundle_record_tampering_without_workspace_fallback(
    tmp_path: Path,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    materialized = FrozenContextTargetMaterializer(
        bundle_root=prepared.root,
        workspace_root=tmp_path / ".eval-workspaces",
    ).materialize(_request(_eval_input(prepared), prepared))
    binding = prepared.manifest.records[0]
    record_path = prepared.root / binding.path
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["record"]["rendered"] += "tampered"
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        with pytest.raises(MaterializationError, match="record (size|hash)"):
            materialized.replay.read_exact()
    finally:
        materialized.close(TrialStatus.FAILED)


def test_frozen_replay_normalizes_malformed_envelope_errors(
    tmp_path: Path,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    materialized = FrozenContextTargetMaterializer(
        bundle_root=prepared.root,
        workspace_root=tmp_path / ".eval-workspaces",
    ).materialize(_request(_eval_input(prepared), prepared))
    binding = prepared.manifest.records[0]
    raw = b"{not-json"
    (prepared.root / binding.path).write_bytes(raw)
    replay = replace(
        materialized.replay,
        _binding=replace(
            binding,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        ),
    )

    try:
        with pytest.raises(MaterializationError, match="record is malformed"):
            replay.read_exact()
    finally:
        materialized.close(TrialStatus.FAILED)

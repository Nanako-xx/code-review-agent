from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import review_agent_eval.artifacts as artifact_module
from review_agent_eval.analysis_artifacts import (
    ANALYSIS_RECEIPT_SCHEMA_VERSION,
    AnalysisArtifactRef,
    AnalysisArtifactStore,
    AnalysisReceipt,
    AnalysisSourceBinding,
    bind_analysis_source,
    derive_analysis_artifact_id,
)
from review_agent_eval.artifacts import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactSecurityError,
)
from review_agent_eval.config import derive_evaluation_id
from review_agent_eval.models import (
    SchemaError,
    canonical_json_bytes,
    canonical_sha256,
    stable_id,
)

from .test_artifacts import make_case_snapshot, make_config, make_store
from .test_orchestrator_target_replay_v2 import (
    _CountingJudge,
    _FrozenSuccessAdapter,
    _execution,
    _frozen_orchestrator,
    _run_frozen,
)


def _source_binding() -> AnalysisSourceBinding:
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot)
    evaluation_id = derive_evaluation_id(
        config.run_id,
        _execution().digest(),
        "analysis-fixture-v1",
    )
    return AnalysisSourceBinding(
        run_id=config.run_id,
        evaluation_id=evaluation_id,
        summary_id=stable_id("run-report-summary-v1", {"fixture": True}),
        summary_digest=canonical_sha256({"summary": "fixture"}),
        run_config_digest=config.digest(),
        case_snapshot_digest=snapshot.digest(),
        trial_score_digests=(canonical_sha256({"score": 1}),),
    )


def _receipt(
    files: dict[str, object],
    *,
    kind: str = "statistics",
) -> AnalysisReceipt:
    return AnalysisReceipt.create(
        kind=kind,
        source_bindings=(_source_binding(),),
        algorithm_digest=canonical_sha256(
            {"algorithm": "analysis-artifact-test-v1"}
        ),
        files=files,
    )


def _store(tmp_path: Path, **kwargs: object) -> AnalysisArtifactStore:
    return AnalysisArtifactStore(
        tmp_path / ".eval-analyses",
        create_root=True,
        max_file_bytes=1024 * 1024,
        max_total_read_bytes=4 * 1024 * 1024,
        **kwargs,
    )


def test_analysis_bundle_is_create_only_and_reloads_identically(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    files = {
        "statistics.json": {"metrics": [], "status": "not_scorable"},
        "coverage.json": {"planned_trials": 1},
    }
    receipt = _receipt(files)

    published = store.publish_json_bundle(
        receipt.kind,
        receipt.artifact_id,
        files,
        receipt,
    )
    assert published == receipt
    assert store.load_json_bundle(receipt.kind, receipt.artifact_id) == files

    resumed = store.publish_json_bundle(
        receipt.kind,
        receipt.artifact_id,
        files,
        receipt,
    )
    assert resumed == receipt

    changed = dict(files)
    changed["statistics.json"] = {"metrics": ["changed"]}
    with pytest.raises(ArtifactIntegrityError, match="receipt|digest"):
        store.publish_json_bundle(
            receipt.kind,
            receipt.artifact_id,
            changed,
            receipt,
        )

    alternate = AnalysisReceipt.create(
        kind=receipt.kind,
        source_bindings=receipt.source_bindings,
        algorithm_digest=receipt.algorithm_digest,
        files=changed,
    )
    assert alternate.artifact_id == receipt.artifact_id
    with pytest.raises(ArtifactConflictError, match="differs|conflict"):
        store.publish_json_bundle(
            alternate.kind,
            alternate.artifact_id,
            changed,
            alternate,
        )


def test_analysis_receipt_binds_run_evaluation_and_source_digests(
    tmp_path: Path,
) -> None:
    run = _run_frozen(
        tmp_path,
        _FrozenSuccessAdapter(),
        instance="analysis-source-binding",
    )
    orchestrator = _frozen_orchestrator(run, judge=_CountingJudge())
    evaluated = orchestrator.evaluate_run(
        run.config.run_id,
        evaluator_execution=_execution(),
        evaluation_revision="analysis-source-v1",
    )
    hydrated = orchestrator.load_run_evaluation(
        run.config.run_id,
        evaluated.evaluation_id,
    )

    binding = bind_analysis_source(
        hydrated,
        run_config=run.config,
        case_snapshot=run.snapshot,
    )
    assert binding.run_id == run.config.run_id
    assert binding.evaluation_id == evaluated.evaluation_id
    assert binding.summary_id == hydrated.summary.summary_id
    assert binding.summary_digest == hydrated.summary.digest()
    assert binding.run_config_digest == run.config.digest()
    assert binding.case_snapshot_digest == run.snapshot.digest()
    assert binding.trial_score_digests == tuple(
        sorted(item.trial_score.digest() for item in hydrated.trials)
    )

    receipt = AnalysisReceipt.create(
        kind="statistics",
        source_bindings=(binding,),
        algorithm_digest=canonical_sha256({"algorithm": "statistics-v1"}),
        files={"statistics.json": {"status": "ok"}},
    )
    assert receipt.schema_version == ANALYSIS_RECEIPT_SCHEMA_VERSION
    assert receipt.source_bindings == (binding,)
    assert AnalysisReceipt.from_dict(receipt.to_dict()) == receipt
    assert receipt.to_json().encode("utf-8") == canonical_json_bytes(receipt.to_dict())

    with pytest.raises(ArtifactIntegrityError, match="RunConfig|source"):
        bind_analysis_source(
            hydrated,
            run_config=make_config(
                instance="another-analysis-source",
                case_snapshot=run.snapshot,
            ),
            case_snapshot=run.snapshot,
        )
    with pytest.raises(ArtifactIntegrityError, match="identity"):
        bind_analysis_source(
            replace(hydrated, evaluation_id="../escape"),
            run_config=run.config,
            case_snapshot=run.snapshot,
        )

    forged_bundle = SimpleNamespace(
        **{
            name: getattr(hydrated, name)
            for name in (
                "run_id",
                "evaluation_id",
                "evaluation_revision",
                "evaluator_execution",
                "trials",
                "summary",
                "report",
            )
        }
    )
    with pytest.raises(TypeError, match="RunEvaluationBundle"):
        bind_analysis_source(
            forged_bundle,
            run_config=run.config,
            case_snapshot=run.snapshot,
        )

    trial = hydrated.trials[0]
    with pytest.raises(ArtifactIntegrityError, match="TrialEvaluationBundle"):
        bind_analysis_source(
            replace(hydrated, trials=(SimpleNamespace(**vars(trial)),)),
            run_config=run.config,
            case_snapshot=run.snapshot,
        )

    summary_duck = SimpleNamespace(
        source_bindings=hydrated.summary.source_bindings,
        summary_id=hydrated.summary.summary_id,
        cases=hydrated.summary.cases,
        digest=hydrated.summary.digest,
    )
    with pytest.raises(ArtifactIntegrityError, match="RunReportSummary|summary"):
        bind_analysis_source(
            replace(hydrated, summary=summary_duck),
            run_config=run.config,
            case_snapshot=run.snapshot,
        )

    with pytest.raises(ArtifactIntegrityError, match="Trial|Run"):
        bind_analysis_source(
            replace(
                hydrated,
                trials=(replace(trial, run_id="run-" + "0" * 64),),
            ),
            run_config=run.config,
            case_snapshot=run.snapshot,
        )

    score = trial.trial_score
    score_duck = SimpleNamespace(**vars(score))
    score_duck.digest = score.digest
    with pytest.raises(ArtifactIntegrityError, match="TrialScore|score"):
        bind_analysis_source(
            replace(
                hydrated,
                trials=(replace(trial, trial_score=score_duck),),
            ),
            run_config=run.config,
            case_snapshot=run.snapshot,
        )

    with pytest.raises(ArtifactIntegrityError, match="Case|case"):
        bind_analysis_source(
            replace(
                hydrated,
                trials=(
                    replace(
                        trial,
                        eval_case=replace(
                            trial.eval_case,
                            case_version=trial.eval_case.case_version + 1,
                        ),
                    ),
                ),
            ),
            run_config=run.config,
            case_snapshot=run.snapshot,
        )

    with pytest.raises(ArtifactIntegrityError, match="report|render"):
        bind_analysis_source(
            replace(hydrated, report=hydrated.report + "\n"),
            run_config=run.config,
            case_snapshot=run.snapshot,
        )

    assert trial.intent_result is not None
    assert trial.review_result is not None
    assert trial.submission.review is not None
    tampered_submission = replace(
        trial.submission,
        usage=replace(
            trial.submission.usage,
            tool_calls=(trial.submission.usage.tool_calls or 0) + 1,
        ),
    )
    tampered_intent = replace(
        trial.intent_result,
        submission_intent_digest="0" * 64,
    )
    tampered_review_submission = replace(
        trial.submission,
        review=replace(
            trial.submission.review,
            uncertainties=trial.submission.review.uncertainties
            + ("nested review tamper",),
        ),
    )
    tampered_trials = (
        replace(trial, submission=tampered_submission),
        replace(trial, intent_result=tampered_intent),
        replace(trial, review_result=None),
        replace(trial, submission=tampered_review_submission),
    )
    for tampered_trial in tampered_trials:
        with pytest.raises(
            ArtifactIntegrityError,
            match="Submission|Intent|Review|source",
        ):
            bind_analysis_source(
                replace(hydrated, trials=(tampered_trial,)),
                run_config=run.config,
                case_snapshot=run.snapshot,
            )


def test_analysis_source_order_uses_complete_canonical_identity() -> None:
    first = _source_binding()
    second = replace(
        first,
        run_config_digest="1" * 64,
        case_snapshot_digest="2" * 64,
        trial_score_digests=("3" * 64,),
    )
    files = {"statistics.json": {"status": "ok"}}
    algorithm_digest = canonical_sha256({"algorithm": "canonical-order-v1"})

    forward = AnalysisReceipt.create(
        kind="statistics",
        source_bindings=(first, second),
        algorithm_digest=algorithm_digest,
        files=files,
    )
    reverse = AnalysisReceipt.create(
        kind="statistics",
        source_bindings=(second, first),
        algorithm_digest=algorithm_digest,
        files=files,
    )

    assert forward.artifact_id == reverse.artifact_id
    assert canonical_json_bytes(forward.to_dict()) == canonical_json_bytes(
        reverse.to_dict()
    )
    assert derive_analysis_artifact_id(
        "statistics", (first, second), algorithm_digest
    ) == derive_analysis_artifact_id(
        "statistics", (second, first), algorithm_digest
    )


def test_analysis_source_binding_duplicates_are_rejected() -> None:
    binding = _source_binding()
    algorithm_digest = canonical_sha256({"algorithm": "duplicate-source-v1"})
    with pytest.raises(SchemaError, match="duplicate"):
        derive_analysis_artifact_id(
            "statistics",
            (binding, binding),
            algorithm_digest,
        )
    with pytest.raises(SchemaError, match="duplicate"):
        AnalysisReceipt.create(
            kind="statistics",
            source_bindings=(binding, binding),
            algorithm_digest=algorithm_digest,
            files={"statistics.json": {"status": "ok"}},
        )


def test_analysis_json_names_reject_portable_casefold_collisions() -> None:
    binding = _source_binding()
    algorithm_digest = canonical_sha256({"algorithm": "portable-name-v1"})
    with pytest.raises(SchemaError, match="portable|collision"):
        AnalysisReceipt.create(
            kind="statistics",
            source_bindings=(binding,),
            algorithm_digest=algorithm_digest,
            files={"A.json": {"value": 1}, "a.json": {"value": 2}},
        )

    receipt = AnalysisReceipt.create(
        kind="statistics",
        source_bindings=(binding,),
        algorithm_digest=algorithm_digest,
        files={"A.json": {"value": 1}},
    )
    colliding_ref = AnalysisArtifactRef.create(
        kind=receipt.kind,
        artifact_id=receipt.artifact_id,
        name="a.json",
        data=canonical_json_bytes({"value": 2}),
    )
    colliding_refs = tuple(
        sorted(
            (*receipt.artifacts, colliding_ref),
            key=lambda item: item.relative_path,
        )
    )
    with pytest.raises(SchemaError, match="portable|collision"):
        replace(receipt, artifacts=colliding_refs)


def test_analysis_tamper_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    files = {"statistics.json": {"metrics": [1]}}
    receipt = _receipt(files)
    store.publish_json_bundle(
        receipt.kind,
        receipt.artifact_id,
        files,
        receipt,
    )
    bundle = store.root / receipt.kind / receipt.artifact_id

    artifact = bundle / "statistics.json"
    artifact.write_bytes(canonical_json_bytes({"metrics": [2]}))
    with pytest.raises(ArtifactIntegrityError, match="size|hash|digest"):
        store.load_json_bundle(receipt.kind, receipt.artifact_id)

    artifact.write_bytes(canonical_json_bytes(files["statistics.json"]))
    receipt_path = bundle / "receipt.json"
    payload = receipt.to_dict()
    payload["algorithm_digest"] = "0" * 64
    receipt_path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ArtifactIntegrityError, match="ID|canonical|receipt"):
        store.load_json_bundle(receipt.kind, receipt.artifact_id)


def test_analysis_rejects_traversal_symlink_and_unknown_artifact_names(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    files = {"statistics.json": {"metrics": []}}
    receipt = _receipt(files)
    store.publish_json_bundle(
        receipt.kind,
        receipt.artifact_id,
        files,
        receipt,
    )

    with pytest.raises((ValueError, ArtifactIntegrityError)):
        store.load_json_bundle("../statistics", receipt.artifact_id)
    with pytest.raises((ValueError, ArtifactIntegrityError)):
        store.publish_json_bundle(
            receipt.kind,
            receipt.artifact_id,
            {"../escape.json": {}},
            receipt,
        )

    bundle = store.root / receipt.kind / receipt.artifact_id
    (bundle / "unknown.json").write_bytes(canonical_json_bytes({"unknown": True}))
    with pytest.raises(ArtifactIntegrityError, match="unknown"):
        store.load_json_bundle(receipt.kind, receipt.artifact_id)
    (bundle / "unknown.json").unlink()

    target = tmp_path / "outside.json"
    target.write_bytes(canonical_json_bytes(files["statistics.json"]))
    artifact = bundle / "statistics.json"
    artifact.unlink()
    try:
        artifact.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip("symlink creation is unavailable: %s" % exc)
    with pytest.raises(ArtifactSecurityError, match="symlink|reparse|unsafe"):
        store.load_json_bundle(receipt.kind, receipt.artifact_id)


def test_analysis_rejects_junction_reparse_and_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    files = {"statistics.json": {"metrics": []}}
    receipt = _receipt(files)
    store.publish_json_bundle(
        receipt.kind,
        receipt.artifact_id,
        files,
        receipt,
    )
    bundle = store.root / receipt.kind / receipt.artifact_id
    artifact = bundle / "statistics.json"

    alias = tmp_path / "artifact-hardlink.json"
    try:
        os.link(artifact, alias)
    except OSError as exc:
        pytest.skip("hardlink creation is unavailable: %s" % exc)
    with pytest.raises(ArtifactSecurityError, match="hardlink|link count"):
        store.load_json_bundle(receipt.kind, receipt.artifact_id)
    alias.unlink()

    target_key = os.path.normcase(os.path.abspath(artifact))
    real_lstat = artifact_module.os.lstat

    class ReparseStat:
        def __init__(self, wrapped: os.stat_result) -> None:
            self._wrapped = wrapped
            self.st_file_attributes = (
                getattr(wrapped, "st_file_attributes", 0) | 0x400
            )

        def __getattr__(self, name: str) -> object:
            return getattr(self._wrapped, name)

    def fake_lstat(path: os.PathLike[str] | str) -> os.stat_result:
        result = real_lstat(path)
        if os.path.normcase(os.path.abspath(os.fspath(path))) == target_key:
            return ReparseStat(result)  # type: ignore[return-value]
        return result

    monkeypatch.setattr(artifact_module.os, "lstat", fake_lstat)
    with pytest.raises(ArtifactSecurityError, match="reparse"):
        store.load_json_bundle(receipt.kind, receipt.artifact_id)


def test_run_artifact_store_hardlink_behavior_is_unchanged(tmp_path: Path) -> None:
    store, config, _manifest, _plan, _trial = make_store(tmp_path)
    ref = store._write_json(
        config.run_id,
        "auxiliary/historical-hardlink.json",
        {"historical": True},
    )
    source = store.root / config.run_id / Path(*ref.relative_path.split("/"))
    alias = tmp_path / "historical-hardlink-alias.json"
    try:
        os.link(source, alias)
    except OSError as exc:
        pytest.skip("hardlink creation is unavailable: %s" % exc)

    assert store.read_json_artifact(config.run_id, ref) == {"historical": True}


@pytest.mark.skipif(os.name != "nt", reason="junction behavior is Windows-specific")
def test_analysis_rejects_windows_junction_namespace(tmp_path: Path) -> None:
    root = tmp_path / ".eval-analyses"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    junction = root / "statistics"
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("junction creation is unavailable: %s" % completed.stderr)
    store = AnalysisArtifactStore(
        root,
        create_root=False,
        max_file_bytes=4096,
        max_total_read_bytes=16384,
    )
    receipt = _receipt({"statistics.json": {"metrics": []}})
    with pytest.raises(ArtifactSecurityError, match="reparse|unexpected|unsafe"):
        store.publish_json_bundle(
            receipt.kind,
            receipt.artifact_id,
            {"statistics.json": {"metrics": []}},
            receipt,
        )


@pytest.mark.skipif(os.name != "nt", reason="short paths are Windows-specific")
def test_analysis_rejects_windows_short_path_alias(tmp_path: Path) -> None:
    root = tmp_path / "analysis root with spaces" / ".eval-analyses"
    root.mkdir(parents=True)
    buffer = ctypes.create_unicode_buffer(32768)
    written = ctypes.windll.kernel32.GetShortPathNameW(str(root), buffer, len(buffer))
    if not written or os.path.normcase(buffer.value) == os.path.normcase(str(root)):
        pytest.skip("8.3 short path aliases are unavailable")
    with pytest.raises(ArtifactSecurityError, match="unexpected|identity|path"):
        AnalysisArtifactStore(
            buffer.value,
            create_root=False,
            max_file_bytes=4096,
            max_total_read_bytes=16384,
        ).publish_json_bundle(
            "statistics",
            _receipt({"statistics.json": {}}).artifact_id,
            {"statistics.json": {}},
            _receipt({"statistics.json": {}}),
        )


def test_analysis_write_does_not_create_missing_read_only_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "missing" / ".eval-analyses"
    assert not root.exists()
    with pytest.raises(ArtifactIntegrityError, match="does not exist"):
        AnalysisArtifactStore(
            root,
            create_root=False,
            max_file_bytes=1024,
            max_total_read_bytes=4096,
        )
    assert not root.exists()


def test_analysis_artifacts_import_does_not_load_product_runtime() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = (
        "import sys\n"
        "import review_agent_eval.analysis_artifacts\n"
        "forbidden = sorted(name for name in sys.modules "
        "if name == 'review_agent' or name.startswith('review_agent.'))\n"
        "if forbidden:\n"
        "    raise SystemExit('product Runtime modules loaded: ' + ','.join(forbidden))\n"
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(repository_root / "src") + (
        "" if not existing_pythonpath else os.pathsep + existing_pythonpath
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_analysis_receipt_exact_keys_and_canonical_artifact_order() -> None:
    files = {"z.json": {"z": 1}, "a.json": {"a": 1}}
    receipt = _receipt(files)
    assert tuple(ref.relative_path for ref in receipt.artifacts) == tuple(
        sorted(ref.relative_path for ref in receipt.artifacts)
    )
    payload = receipt.to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="keys|fields|schema"):
        AnalysisReceipt.from_dict(payload)


def test_analysis_store_enforces_file_and_total_read_budgets(
    tmp_path: Path,
) -> None:
    files = {
        "a.json": {"blob": "a" * 80},
        "b.json": {"blob": "b" * 80},
    }
    receipt = _receipt(files)
    writer = _store(tmp_path)
    writer.publish_json_bundle(receipt.kind, receipt.artifact_id, files, receipt)

    single = AnalysisArtifactStore(
        writer.root,
        create_root=False,
        max_file_bytes=64,
        max_total_read_bytes=256,
    )
    with pytest.raises(ArtifactIntegrityError, match="single-file"):
        single.load_json_bundle(receipt.kind, receipt.artifact_id)

    receipt_bytes = len(canonical_json_bytes(receipt.to_dict()))
    artifact_sizes = [len(canonical_json_bytes(item)) for item in files.values()]
    largest_file = max(receipt_bytes, *artifact_sizes)
    cumulative_size = receipt_bytes + sum(artifact_sizes)
    assert cumulative_size - 1 >= largest_file
    total = AnalysisArtifactStore(
        writer.root,
        create_root=False,
        max_file_bytes=largest_file,
        max_total_read_bytes=cumulative_size - 1,
    )
    with pytest.raises(ArtifactIntegrityError, match="cumulative"):
        total.load_json_bundle(receipt.kind, receipt.artifact_id)

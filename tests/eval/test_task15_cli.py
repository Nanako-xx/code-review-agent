from __future__ import annotations

import argparse
import json
import os
import stat
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import review_agent_eval.artifacts as artifact_module
import review_agent_eval.cli as cli_module
from review_agent_eval.cli import (
    EXIT_INTEGRITY,
    EXIT_OK,
    EXIT_POLICY,
    EXIT_PRECONDITION,
    EXIT_USAGE,
    _build_parser,
    main,
)
from review_agent_eval.comparison import ComparisonStatus
from review_agent_eval.gates import GateDecision
from review_agent_eval.artifacts import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactSecurityError,
    ArtifactStore,
)


def _strict_tree_snapshot(root: Path) -> tuple[tuple[str, str, bytes | None], ...]:
    """Inventory every reachable entry and fail instead of skipping scan errors."""

    scan_root = root
    if os.name == "nt":
        absolute = os.path.abspath(os.fspath(root))
        if not absolute.startswith("\\\\?\\"):
            scan_root = Path("\\\\?\\" + absolute)
    root_info = os.lstat(scan_root)
    if not stat.S_ISDIR(root_info.st_mode):
        raise AssertionError("snapshot root is not a directory")
    pending = [scan_root]
    values: list[tuple[str, str, bytes | None]] = []
    while pending:
        directory = pending.pop()
        entries = sorted(os.scandir(directory), key=lambda item: item.name)
        for entry in entries:
            metadata = entry.stat(follow_symlinks=False)
            relative = Path(entry.path).relative_to(scan_root).as_posix()
            is_reparse = bool(getattr(metadata, "st_file_attributes", 0) & 0x400)
            if is_reparse:
                raise AssertionError("snapshot tree contains a reparse point")
            if stat.S_ISDIR(metadata.st_mode):
                values.append((relative, "directory", None))
                pending.append(Path(entry.path))
            elif stat.S_ISREG(metadata.st_mode):
                values.append((relative, "file", Path(entry.path).read_bytes()))
            else:
                raise AssertionError("snapshot tree contains a special entry")
    return tuple(sorted(values))


def _analysis_loader_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        suite_root=str(tmp_path / "suite"),
        manifest="suite_manifest.json",
        expected_manifest_digest=None,
        runs_root=str(tmp_path / ".eval-runs"),
        data_root=str(tmp_path / ".eval-data"),
        workspace_root=str(tmp_path / ".eval-workspaces"),
        analysis_root=str(tmp_path / ".eval-analyses"),
    )


def test_real_analysis_loader_accepts_different_verified_suite_subsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from review_agent_eval.orchestrator import EvaluationOrchestrator

    from .test_artifacts import make_config
    from .test_datasets import case_payload, write_suite

    args = _analysis_loader_args(tmp_path)
    write_suite(
        Path(args.suite_root),
        (case_payload("task-kept"), case_payload("task-filtered")),
    )
    from review_agent_eval.datasets import CaseBank

    bank = CaseBank.open(Path(args.suite_root))
    subset = bank.snapshot(("task-kept",))
    other_subset = bank.snapshot(("task-filtered",))
    assert subset != other_subset
    config = make_config(
        instance="analysis-filtered-subset",
        case_snapshot=subset,
    )
    other_config = make_config(
        instance="analysis-other-filtered-subset",
        case_snapshot=other_subset,
    )
    store = ArtifactStore(Path(args.runs_root))
    store.create_run(config, subset)
    store.create_run(other_config, other_subset)

    class HydrationReached(RuntimeError):
        pass

    monkeypatch.setattr(
        cli_module,
        "_repository_preparer_context",
        lambda *_args, **_kwargs: nullcontext(None),
    )
    hydration_calls = []

    def hydration_reached(_self: Any, run_id: str, _evaluation_id: str):
        hydration_calls.append(run_id)
        raise HydrationReached()

    monkeypatch.setattr(
        EvaluationOrchestrator,
        "load_run_evaluation",
        hydration_reached,
    )

    with cli_module._analysis_evaluation_loader(args) as (load, _store, _root):
        for index, run in enumerate((config, other_config), start=1):
            with pytest.raises(HydrationReached):
                load(run.run_id, "evaluation-" + str(index) * 64)
    assert hydration_calls == [config.run_id, other_config.run_id]


@pytest.mark.parametrize("mutation", ("unknown", "rewritten"))
def test_real_analysis_loader_rejects_untrusted_snapshot_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    from review_agent_eval.orchestrator import EvaluationOrchestrator

    from .test_artifacts import make_config
    from .test_datasets import case_payload, write_suite

    args = _analysis_loader_args(tmp_path)
    trusted = case_payload("task-kept")
    write_suite(Path(args.suite_root), (trusted, case_payload("task-other")))
    foreign_root = tmp_path / "foreign-suite"
    foreign = case_payload("unknown-task" if mutation == "unknown" else "task-kept")
    if mutation == "rewritten":
        foreign["input"]["review_target"]["review_request"]["user_intent"] = (
            "rewritten persisted input"
        )
    write_suite(foreign_root, (foreign,))
    from review_agent_eval.datasets import CaseBank

    foreign_snapshot = CaseBank.open(foreign_root).snapshot()
    config = make_config(
        instance="analysis-untrusted-" + mutation,
        case_snapshot=foreign_snapshot,
    )
    store = ArtifactStore(Path(args.runs_root))
    store.create_run(config, foreign_snapshot)
    monkeypatch.setattr(
        cli_module,
        "_repository_preparer_context",
        lambda *_args, **_kwargs: nullcontext(None),
    )
    monkeypatch.setattr(
        EvaluationOrchestrator,
        "load_run_evaluation",
        lambda *_args, **_kwargs: pytest.fail("untrusted snapshot reached hydration"),
    )

    with cli_module._analysis_evaluation_loader(args) as (load, _store, _root):
        with pytest.raises(cli_module.CliIntegrityError, match="trusted Suite"):
            load(config.run_id, "evaluation-" + "2" * 64)


def _command_choices(parser: Any) -> dict[str, Any]:
    action = next(item for item in parser._actions if item.dest == "command")
    return action.choices


def _nested_choices(parser: Any, destination: str) -> set[str]:
    action = next(item for item in parser._actions if item.dest == destination)
    return set(action.choices)


def test_parser_exposes_complete_task15_analysis_lifecycle() -> None:
    commands = _command_choices(_build_parser())

    assert {"compare", "calibrate", "gate"} <= set(commands)
    assert _nested_choices(commands["calibrate"], "calibrate_command") == {
        "export",
        "import-labels",
        "score",
    }
    assert _nested_choices(commands["gate"], "gate_command") == {
        "prepare",
        "evaluate",
    }


@pytest.mark.parametrize(
    "arguments",
    (
        ["compare"],
        ["calibrate", "export"],
        ["calibrate", "import-labels"],
        ["calibrate", "score"],
        ["gate", "prepare"],
        ["gate", "evaluate"],
    ),
)
def test_analysis_commands_require_source_arguments_and_accept_analysis_root(
    arguments: list[str],
) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as missing:
        parser.parse_args(arguments)
    assert missing.value.code == EXIT_USAGE

    command = _command_choices(parser)[arguments[0]]
    if len(arguments) == 2:
        nested = next(
            item
            for item in command._actions
            if item.dest == arguments[0] + "_command"
        )
        command = nested.choices[arguments[1]]
    destinations = {item.dest for item in command._actions}
    assert "analysis_root" in destinations
    assert destinations.isdisjoint(
        {
            "agent_provider",
            "agent_command",
            "agent_config",
            "judge_provider",
            "judge_model",
            "source_root",
            "git_executable",
        }
    )


@pytest.mark.parametrize(
    "arguments",
    (
        ["compare"],
        ["calibrate", "export"],
        ["calibrate", "import-labels"],
        ["calibrate", "score"],
        ["gate", "prepare"],
        ["gate", "evaluate"],
    ),
)
def test_analysis_commands_reject_provider_and_acquisition_options(
    arguments: list[str],
    tmp_path: Path,
) -> None:
    for forbidden in (
        "--agent-provider",
        "--judge-provider",
        "--agent-command",
        "--source-root",
        "--git-executable",
    ):
        assert main([*arguments, forbidden, "fake"]) == EXIT_USAGE


def test_compare_missing_run_root_is_precondition_and_does_not_create_analysis(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    analysis = tmp_path / ".eval-analyses"
    policy = tmp_path / "comparison-policy.json"
    policy.write_text("{}", encoding="utf-8")

    code = main(
        [
            "compare",
            "--suite-root",
            str(suite),
            "--runs-root",
            str(tmp_path / ".eval-runs"),
            "--analysis-root",
            str(analysis),
            "--baseline-run-id",
            "run-" + "1" * 64,
            "--baseline-evaluation-id",
            "evaluation-" + "2" * 64,
            "--candidate-run-id",
            "run-" + "3" * 64,
            "--candidate-evaluation-id",
            "evaluation-" + "4" * 64,
            "--policy",
            str(policy),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_PRECONDITION
    assert payload["error_code"] == "precondition"
    assert not analysis.exists()


@pytest.mark.parametrize("malformed", ("../run", "evaluation/not-an-id"))
def test_compare_malformed_source_ids_are_integrity_errors_without_path_echo(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    malformed: str,
) -> None:
    @contextmanager
    def loader(_args: Any):
        def reject(_run: str, _evaluation: str):
            raise ValueError("private/path/" + malformed)

        yield reject, object(), Path("analysis")

    monkeypatch.setattr(cli_module, "_analysis_evaluation_loader", loader)
    code = main(
        [
            "compare",
            "--suite-root",
            "suite",
            "--baseline-run-id",
            malformed,
            "--baseline-evaluation-id",
            "evaluation-" + "2" * 64,
            "--candidate-run-id",
            "run-" + "3" * 64,
            "--candidate-evaluation-id",
            "evaluation-" + "4" * 64,
            "--policy",
            "comparison.json",
            "--json",
        ]
    )
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert code == EXIT_INTEGRITY
    assert payload["error_code"] == "integrity"
    assert "private/path" not in output


@pytest.mark.parametrize(
    ("decision", "ci", "expected"),
    (
        ("promote", False, EXIT_OK),
        ("promote", True, EXIT_OK),
        ("block", False, EXIT_OK),
        ("block", True, EXIT_POLICY),
        ("ineligible", False, EXIT_OK),
        ("ineligible", True, EXIT_POLICY),
    ),
)
def test_gate_policy_exit_mapping_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
    ci: bool,
    expected: int,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "_handle_gate_evaluate",
        lambda args: cli_module._gate_exit_code(decision, ci=args.ci),
    )
    arguments = [
        "gate",
        "evaluate",
        "--suite-root",
        "suite",
        "--baseline-run-id",
        "run-" + "1" * 64,
        "--baseline-evaluation-id",
        "evaluation-" + "2" * 64,
        "--candidate-run-id",
        "run-" + "3" * 64,
        "--candidate-evaluation-id",
        "evaluation-" + "4" * 64,
        "--comparison-id",
        "analysis-artifact-v1-" + "5" * 64,
        "--comparison-policy",
        "comparison.json",
        "--gate-policy-id",
        "analysis-artifact-v1-" + "6" * 64,
    ]
    if ci:
        arguments.append("--ci")

    assert main(arguments) == expected


class _Receipt:
    artifact_id = "analysis-artifact-v1-" + "a" * 64

    @staticmethod
    def digest() -> str:
        return "b" * 64


def _analysis_arguments(command: list[str]) -> list[str]:
    return [
        *command,
        "--suite-root",
        "suite",
        "--baseline-run-id",
        "run-" + "1" * 64,
        "--baseline-evaluation-id",
        "evaluation-" + "2" * 64,
        "--candidate-run-id",
        "run-" + "3" * 64,
        "--candidate-evaluation-id",
        "evaluation-" + "4" * 64,
    ]


def test_compare_publishes_not_comparable_as_normal_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = SimpleNamespace(run_id="run-" + "1" * 64)
    candidate = SimpleNamespace(run_id="run-" + "3" * 64)

    @contextmanager
    def loader(_args: Any):
        values = iter((baseline, candidate))
        yield lambda _run, _evaluation: next(values), object(), Path("analysis")

    result = SimpleNamespace(
        comparison_id="comparison-v1-" + "9" * 64,
        status=ComparisonStatus.NOT_COMPARABLE,
        incompatibilities=("trial.count",),
    )
    store = SimpleNamespace(
        publish_comparison=lambda _comparison, policy: _Receipt()
    )
    monkeypatch.setattr(cli_module, "_analysis_evaluation_loader", loader)
    monkeypatch.setattr(cli_module, "_analysis_store", lambda *_args, **_kwargs: store)
    monkeypatch.setattr(cli_module, "_comparison_policy", lambda _path: object())
    import review_agent_eval.comparison as comparison_module

    monkeypatch.setattr(
        comparison_module,
        "compare_runs",
        lambda _baseline, _candidate, _policy: result,
    )
    code = main(
        [
            *_analysis_arguments(["compare"]),
            "--policy",
            "comparison.json",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["comparison_status"] == "not_comparable"
    assert payload["incompatibilities"] == ["trial.count"]
    assert payload["artifact_id"] == _Receipt.artifact_id


@pytest.mark.parametrize(
    ("decision", "ci", "expected"),
    (
        (GateDecision.BLOCK, False, EXIT_OK),
        (GateDecision.BLOCK, True, EXIT_POLICY),
        (GateDecision.INELIGIBLE, False, EXIT_OK),
        (GateDecision.INELIGIBLE, True, EXIT_POLICY),
    ),
)
def test_gate_block_and_ineligible_publish_before_optional_ci_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    decision: GateDecision,
    ci: bool,
    expected: int,
) -> None:
    baseline = object()
    candidate = object()
    policy = SimpleNamespace(artifact_id="analysis-artifact-v1-" + "6" * 64)
    comparison = SimpleNamespace(comparison_id="comparison-v1-" + "5" * 64)
    result = SimpleNamespace(
        gate_result_id="gate-result-v1-" + "7" * 64,
        decision=decision,
        checks=(),
    )

    class Store:
        def load_verified_comparison(self, *_args: Any, **_kwargs: Any):
            return comparison

        def load_verified_gate_policy(self, *_args: Any, **_kwargs: Any):
            return policy

        def publish_gate_result(self, *_args: Any, **_kwargs: Any):
            return _Receipt()

    store = Store()
    run_store = SimpleNamespace(load_run_config=lambda _run: object())

    @contextmanager
    def loader(_args: Any):
        values = iter((baseline, candidate))
        yield lambda _run, _evaluation: next(values), run_store, Path("analysis")

    monkeypatch.setattr(cli_module, "_analysis_evaluation_loader", loader)
    monkeypatch.setattr(cli_module, "_analysis_store", lambda *_args, **_kwargs: store)
    monkeypatch.setattr(cli_module, "_comparison_policy", lambda _path: object())
    import review_agent_eval.gates as gates_module

    monkeypatch.setattr(
        gates_module,
        "evaluate_gate",
        lambda _store, _policy, _comparison, _calibrations: result,
    )
    arguments = [
        *_analysis_arguments(["gate", "evaluate"]),
        "--comparison-id",
        "analysis-artifact-v1-" + "5" * 64,
        "--comparison-policy",
        "comparison.json",
        "--gate-policy-id",
        policy.artifact_id,
        "--json",
    ]
    if ci:
        arguments.append("--ci")

    code = main(arguments)
    payload = json.loads(capsys.readouterr().out)

    assert code == expected
    assert payload["status"] == "ok"
    assert payload["decision"] == decision.value
    assert payload["artifact_id"] == _Receipt.artifact_id


def test_gate_prepare_holds_candidate_start_lock_through_policy_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from review_agent_eval.gates import GateEligibility, GatePolicyV1

    from .test_artifacts import make_store

    run_root = tmp_path / "run"
    run_root.mkdir()
    run_store, candidate, _manifest, plan, _trial = make_store(run_root)
    trial_lock = run_store._trial_lock_path(_trial)
    assert trial_lock.read_bytes() == b"\0"
    run_tree_before_reservation = _strict_tree_snapshot(run_store.root)
    snapshot = run_store.load_case_snapshot(candidate.run_id)
    baseline = SimpleNamespace(case_snapshot=snapshot)
    proposal = object()
    entered_publish = threading.Event()
    release_publish = threading.Event()
    policy_receipt = tmp_path / "analysis" / "gate-policy" / "policy" / "receipt.json"

    class Store:
        def publish_gate_policy(self, *_args: Any, **_kwargs: Any):
            entered_publish.set()
            assert release_publish.wait(timeout=10)
            policy_receipt.parent.mkdir(parents=True)
            policy_receipt.write_bytes(b"{}")
            return SimpleNamespace(
                policy=SimpleNamespace(
                    policy_id="gate-policy-v1-" + "5" * 64,
                    eligibility=GateEligibility.RELEASE_BLOCKING,
                ),
                artifact_id="analysis-artifact-v1-" + "6" * 64,
                receipt_digest="7" * 64,
            )

    @contextmanager
    def loader(_args: Any):
        yield lambda _run, _evaluation: baseline, run_store, tmp_path / "analysis"

    monkeypatch.setattr(cli_module, "_analysis_evaluation_loader", loader)
    monkeypatch.setattr(cli_module, "_analysis_store", lambda *_args, **_kwargs: Store())
    monkeypatch.setattr(cli_module, "_read_json", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        GatePolicyV1,
        "from_dict",
        classmethod(lambda _cls, _value: proposal),
    )
    args = argparse.Namespace(
        baseline_run_id="run-" + "1" * 64,
        baseline_evaluation_id="evaluation-" + "2" * 64,
        candidate_run_id=candidate.run_id,
        policy="policy.json",
        json=True,
    )
    outcome: list[Any] = []

    def prepare_gate() -> None:
        try:
            outcome.append(cli_module._handle_gate_prepare(args))
        except BaseException as exc:  # preserve the exact competing outcome
            outcome.append(exc)

    worker = threading.Thread(target=prepare_gate)
    worker.start()
    assert entered_publish.wait(timeout=10)
    try:
        try:
            run_store.start_trial(candidate.run_id, plan.task_id, plan.trial_id)
        except BaseException as exc:
            start_outcome: Any = exc
        else:
            start_outcome = "started"
    finally:
        release_publish.set()
        worker.join(timeout=10)

    assert not worker.is_alive()
    assert outcome == [EXIT_OK]
    assert isinstance(start_outcome, ArtifactConflictError)
    assert _strict_tree_snapshot(run_store.root) == run_tree_before_reservation

    # Once Policy publication releases the reservation, Trial execution may start.
    started = run_store.start_trial(candidate.run_id, plan.task_id, plan.trial_id)
    assert started.active_attempt == 1
    start_receipts = tuple(run_store.root.rglob("start.json"))
    assert len(start_receipts) == 1
    assert policy_receipt.stat().st_mtime_ns <= start_receipts[0].stat().st_mtime_ns
    with pytest.raises(ArtifactConflictError, match="already started"):
        with run_store.reserve_run_before_execution(candidate.run_id):
            raise AssertionError("started Run must never enter gate reservation")
    capsys.readouterr()


def _configure_gate_prepare_test(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_store: Any,
    candidate: Any,
    snapshot: Any,
) -> tuple[argparse.Namespace, Any]:
    from review_agent_eval.gates import GateEligibility, GatePolicyV1

    proposal = object()

    class Store:
        published = False

        def publish_gate_policy(self, *_args: Any, **_kwargs: Any):
            self.published = True
            return SimpleNamespace(
                policy=SimpleNamespace(
                    policy_id="gate-policy-v1-" + "5" * 64,
                    eligibility=GateEligibility.RELEASE_BLOCKING,
                ),
                artifact_id="analysis-artifact-v1-" + "6" * 64,
                receipt_digest="7" * 64,
            )

    store = Store()

    @contextmanager
    def loader(_args: Any):
        baseline = SimpleNamespace(case_snapshot=snapshot)
        yield lambda _run, _evaluation: baseline, run_store, Path("analysis")

    monkeypatch.setattr(cli_module, "_analysis_evaluation_loader", loader)
    monkeypatch.setattr(cli_module, "_analysis_store", lambda *_args, **_kwargs: store)
    monkeypatch.setattr(cli_module, "_read_json", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        GatePolicyV1,
        "from_dict",
        classmethod(lambda _cls, _value: proposal),
    )
    return (
        argparse.Namespace(
            baseline_run_id="run-" + "1" * 64,
            baseline_evaluation_id="evaluation-" + "2" * 64,
            candidate_run_id=candidate.run_id,
            policy="policy.json",
            json=True,
        ),
        store,
    )


def test_gate_prepare_missing_precreated_lock_fails_without_run_store_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .test_artifacts import make_store

    run_store, candidate, _manifest, _plan, trial = make_store(tmp_path)
    run_store._trial_lock_path(trial).unlink(missing_ok=True)
    before = _strict_tree_snapshot(run_store.root)
    args, analysis_store = _configure_gate_prepare_test(
        monkeypatch,
        run_store=run_store,
        candidate=candidate,
        snapshot=run_store.load_case_snapshot(candidate.run_id),
    )

    with pytest.raises(ArtifactIntegrityError, match="writer lock"):
        cli_module._handle_gate_prepare(args)

    assert analysis_store.published is False
    assert _strict_tree_snapshot(run_store.root) == before


@pytest.mark.parametrize(
    "relative_path",
    (
        "submission.json",
        "runner/orphan.json",
        "materializations/orphan.json",
        "trace.json",
        "evaluations/orphan.json",
        "results/orphan.json",
        "receipts/orphan.json",
        "unknown.bin",
    ),
)
def test_gate_prepare_rejects_every_orphan_execution_artifact_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    from .test_artifacts import make_store

    run_store, candidate, _manifest, _plan, trial = make_store(tmp_path)
    orphan = run_store._trial_dir(trial) / relative_path
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"orphan")
    before = _strict_tree_snapshot(run_store.root)
    args, analysis_store = _configure_gate_prepare_test(
        monkeypatch,
        run_store=run_store,
        candidate=candidate,
        snapshot=run_store.load_case_snapshot(candidate.run_id),
    )

    with pytest.raises(
        ArtifactIntegrityError,
        match="prepared Trial inventory|Trial receipts",
    ):
        cli_module._handle_gate_prepare(args)

    assert analysis_store.published is False
    assert _strict_tree_snapshot(run_store.root) == before


def test_prepared_inventory_stops_at_directory_limit_plus_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .test_artifacts import make_store

    run_store, candidate, _manifest, _plan, trial = make_store(tmp_path)
    trial_root = run_store._trial_dir(trial)
    original_scandir = os.scandir
    with original_scandir(trial_root) as entries:
        expected_entries = list(entries)
    assert len(expected_entries) == 5
    observed = {"count": 0}

    class BoundedEntries:
        def __enter__(self):
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def __iter__(self):
            return self

        def __next__(self):
            observed["count"] += 1
            index = observed["count"] - 1
            if index < len(expected_entries):
                return expected_entries[index]
            if index == len(expected_entries):
                return SimpleNamespace(name="limit-plus-one")
            raise AssertionError("prepared inventory read beyond limit+1")

    def bounded_scandir(path: Any):
        if os.path.normcase(os.fspath(path)) == os.path.normcase(
            os.fspath(trial_root)
        ):
            return BoundedEntries()
        return original_scandir(path)

    monkeypatch.setattr(artifact_module.os, "scandir", bounded_scandir)
    with pytest.raises(ArtifactIntegrityError, match="entry limit"):
        with run_store.reserve_run_before_execution(candidate.run_id):
            raise AssertionError("overflowed inventory must not reserve")
    assert observed["count"] == 6


def test_prepared_inventory_rejects_large_orphan_set(
    tmp_path: Path,
) -> None:
    from .test_artifacts import make_store

    run_store, candidate, _manifest, _plan, trial = make_store(tmp_path)
    trial_root = run_store._trial_dir(trial)
    for index in range(128):
        (trial_root / ("orphan-%04d.json" % index)).write_bytes(b"orphan")

    with pytest.raises(ArtifactIntegrityError, match="entry limit|unknown file"):
        with run_store.reserve_run_before_execution(candidate.run_id):
            raise AssertionError("orphan inventory must not reserve")


@pytest.mark.parametrize("artifact_name", ("trial.lock", "trial_manifest.json"))
def test_gate_prepare_rejects_external_hardlinks_independent_of_store_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    from .test_artifacts import make_store

    run_store, candidate, _manifest, _plan, trial = make_store(tmp_path)
    assert run_store._reject_hardlinks is False
    source = (
        run_store._trial_lock_path(trial)
        if artifact_name == "trial.lock"
        else run_store._trial_dir(trial) / artifact_name
    )
    external_alias = tmp_path / ("external-" + artifact_name)
    try:
        os.link(source, external_alias)
    except NotImplementedError as exc:
        pytest.skip(f"hardlink capability is unavailable: {exc}")
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip(f"hardlink capability is unavailable: {exc}")
        raise
    assert os.stat(source).st_nlink == 2

    before = _strict_tree_snapshot(run_store.root)
    args, analysis_store = _configure_gate_prepare_test(
        monkeypatch,
        run_store=run_store,
        candidate=candidate,
        snapshot=run_store.load_case_snapshot(candidate.run_id),
    )

    with pytest.raises(ArtifactSecurityError, match="hardlink"):
        cli_module._handle_gate_prepare(args)

    assert analysis_store.published is False
    assert _strict_tree_snapshot(run_store.root) == before


@pytest.mark.parametrize(
    "relative_path",
    (
        "run_config.json",
        "case_snapshot.json",
        "run_manifest.json",
        "receipts/capability_preflight.json",
    ),
)
def test_gate_prepare_rejects_hardlinked_run_control_files_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    from .test_artifacts import make_case_snapshot, make_config, make_store

    if relative_path == "receipts/capability_preflight.json":
        run_store = ArtifactStore(tmp_path / ".eval-runs")
        snapshot = make_case_snapshot()
        candidate = make_config(
            instance="hardlinked-run-preflight",
            case_snapshot=snapshot,
        )
        run_store.create_run(
            candidate,
            snapshot,
            run_preflight={"capability": "verified"},
        )
    else:
        run_store, candidate, _manifest, _plan, _trial = make_store(tmp_path)
    source = run_store._run_dir(candidate.run_id) / relative_path
    external_alias = tmp_path / (
        "external-" + relative_path.replace("/", "-")
    )
    try:
        os.link(source, external_alias)
    except NotImplementedError as exc:
        pytest.skip(f"hardlink capability is unavailable: {exc}")
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip(f"hardlink capability is unavailable: {exc}")
        raise
    assert os.stat(source).st_nlink == 2

    before = _strict_tree_snapshot(run_store.root)
    args, analysis_store = _configure_gate_prepare_test(
        monkeypatch,
        run_store=run_store,
        candidate=candidate,
        snapshot=run_store.load_case_snapshot(candidate.run_id),
    )

    with pytest.raises(ArtifactSecurityError, match="hardlink|link count"):
        cli_module._handle_gate_prepare(args)

    assert analysis_store.published is False
    assert _strict_tree_snapshot(run_store.root) == before

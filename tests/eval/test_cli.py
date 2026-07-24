from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path

import pytest

from review_agent_eval.cli import (
    EXIT_CONFLICT,
    EXIT_INTEGRITY,
    EXIT_OK,
    EXIT_PRECONDITION,
    _canonical_digest,
    _build_parser,
    _domain_error_category,
    main,
)
from review_agent_eval.repository import FixtureRepositoryBuilder
from review_agent_eval.target_replay import TargetReplayIntegrityError

from .test_agent_adapter import _AGENT_PROGRAM
from .test_datasets import case_payload, write_suite


def _roots(tmp_path: Path) -> dict[str, Path]:
    return {
        "suite": tmp_path / "suite",
        "runs": tmp_path / ".eval-runs",
        "data": tmp_path / ".eval-data",
        "workspaces": tmp_path / ".eval-workspaces",
    }


def _write_cli_suite(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    roots = _roots(tmp_path)
    fixture = roots["suite"] / "repositories" / "task-001"
    (fixture / "base").mkdir(parents=True)
    (fixture / "head").mkdir(parents=True)
    (fixture / "base" / "app.py").write_text(
        "def value():\n    return 1\n", encoding="utf-8"
    )
    (fixture / "head" / "app.py").write_text(
        "def value():\n    return 2\n", encoding="utf-8"
    )
    built = FixtureRepositoryBuilder().build(
        fixture,
        tmp_path / "authored.git",
    )
    payload = case_payload()
    payload["input"]["review_target"]["repository"][
        "base_revision"
    ] = built.base_revision
    payload["input"]["review_target"]["repository"][
        "head_revision"
    ] = built.head_revision
    payload["input"]["review_target"]["review_request"]["user_intent"] = (
        "Review the requested change"
    )
    payload["clarification_script"] = {"max_rounds": 1, "answers": []}
    payload["intent_truth"] = {
        "scorable": True,
        "authority": "explicit_author_metadata",
        "expected_claims": [
            {
                "truth_id": "intent-task-001",
                "dimension": "goal",
                "text": "Review the requested change",
                "required": True,
            }
        ],
        "forbidden_claims": [],
        "clarification_policy": "not_required",
    }
    write_suite(roots["suite"], [payload])
    program = tmp_path / "subprocess-agent.py"
    program.write_text(_AGENT_PROGRAM, encoding="utf-8")
    return roots, program


def _root_arguments(roots: dict[str, Path]) -> list[str]:
    return [
        "--suite-root",
        str(roots["suite"]),
        "--runs-root",
        str(roots["runs"]),
        "--data-root",
        str(roots["data"]),
        "--workspace-root",
        str(roots["workspaces"]),
    ]


def _json_stdout(capsys: pytest.CaptureFixture[str]) -> dict:
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def _without_message(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "message"}


def test_target_replay_integrity_error_has_stable_cli_category() -> None:
    assert _domain_error_category(
        TargetReplayIntegrityError("fixture replay drift")
    ) == ("integrity", EXIT_INTEGRITY)


def test_parser_exposes_the_four_trial_commands_and_public_prepare() -> None:
    parser = _build_parser()
    action = next(
        item for item in parser._actions if item.dest == "command"  # type: ignore[attr-defined]
    )

    assert set(action.choices) == {
        "prepare",
        "prepare-public",
        "run-agent",
        "evaluate",
        "inspect",
    }
    assert "compare" not in action.choices
    assert "calibrate" not in action.choices


def test_prepare_dry_run_is_write_free_and_emits_stable_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    roots, _program = _write_cli_suite(tmp_path)

    code = main(
        [
            "prepare",
            *_root_arguments(roots),
            "--dry-run",
            "--json",
        ]
    )
    payload = _json_stdout(capsys)

    assert code == EXIT_OK
    assert payload["schema_version"] == "review_agent_eval_cli_v1"
    assert payload["command"] == "prepare"
    assert payload["dry_run"] is True
    assert payload["case_count"] == 1
    assert not roots["runs"].exists()
    assert not roots["data"].exists()
    assert not roots["workspaces"].exists()


def test_prepare_and_run_agent_use_the_v2_subprocess_protocol_without_evaluation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    roots, program = _write_cli_suite(tmp_path)
    common = _root_arguments(roots)
    command = [
        str(Path(sys.executable).resolve()),
        str(program),
        "success",
        "{agent_id}",
        "{task_id}",
        "{trial_id}",
    ]
    prepare_args = [
        "prepare",
        *common,
        "--agent-adapter",
        "subprocess",
        "--json",
    ]
    for item in command:
        prepare_args.extend(("--agent-command", item))

    assert main(prepare_args) == EXIT_OK
    run_id = _json_stdout(capsys)["run_id"]

    assert main(["run-agent", run_id, *common, "--json"]) == EXIT_OK
    result = _json_stdout(capsys)
    assert result["run_status"] == "completed"
    assert result["trials"][0]["submission_status"] == "completed"


def test_filter_prepare_resume_reuses_canonical_planned_run_before_acquisition(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import review_agent_eval.cli as cli_module
    from review_agent_eval.artifacts import ArtifactStore

    from .test_runner import _SelectiveIncompatibleAdapter, _two_case_snapshot

    snapshot = _two_case_snapshot()

    class SnapshotBank:
        def snapshot(self):
            return snapshot

    monkeypatch.setattr(
        cli_module,
        "_load_case_bank",
        lambda _args, _suite_root: SnapshotBank(),
    )
    adapters = []
    original_configs = []

    def adapter_for(config):
        original_configs.append(config)
        adapter = _SelectiveIncompatibleAdapter()
        adapters.append(adapter)
        return _SelectiveIncompatibleAdapter, adapter

    monkeypatch.setattr(cli_module, "_agent_adapter", adapter_for)
    acquisitions = []
    expected_run_id = None

    def record_acquisition(_args, roots, selected_snapshot):
        if expected_run_id is not None:
            ArtifactStore(roots[1], create_root=False).load_run_config(
                expected_run_id
            )
        acquisitions.append(
            tuple(item.task_id for item in selected_snapshot.cases)
        )

    monkeypatch.setattr(
        cli_module, "_prepare_repository_targets", record_acquisition
    )
    roots = _roots(tmp_path)
    arguments = [
        "prepare",
        *_root_arguments(roots),
        "--capability-policy",
        "filter",
        "--run-instance-key",
        "filter-resume",
        "--json",
    ]

    assert main(arguments) == EXIT_OK
    first = _json_stdout(capsys)
    expected_run_id = first["run_id"]
    assert expected_run_id != original_configs[0].run_id
    run_root = roots["runs"] / expected_run_id
    immutable_paths = (
        "run_config.json",
        "case_snapshot.json",
        "receipts/capability_preflight.json",
        "run_manifest.json",
    )
    first_bytes = {
        relative: (run_root / relative).read_bytes()
        for relative in immutable_paths
    }
    assert acquisitions == [("task-kept",)]

    original_load_preflight = ArtifactStore.load_run_preflight
    monkeypatch.setattr(
        ArtifactStore,
        "load_run_preflight",
        lambda _self, _run_id: {},
    )
    assert main([*arguments, "--resume"]) == EXIT_INTEGRITY
    _json_stdout(capsys)
    assert acquisitions == [("task-kept",)]
    monkeypatch.setattr(
        ArtifactStore,
        "load_run_preflight",
        original_load_preflight,
    )

    assert main([*arguments, "--resume"]) == EXIT_OK
    resumed = _json_stdout(capsys)
    assert resumed["run_id"] == expected_run_id
    assert resumed["resumed"] is True
    assert acquisitions == [("task-kept",), ("task-kept",)]
    assert {
        relative: (run_root / relative).read_bytes()
        for relative in immutable_paths
    } == first_bytes
    assert len(
        [path for path in roots["runs"].iterdir() if path.name.startswith("run-")]
    ) == 1
    assert all(adapter.run_calls == 0 for adapter in adapters)


def test_full_prepare_run_evaluate_inspect_flow_keeps_stages_separate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    roots, program = _write_cli_suite(tmp_path)
    common = _root_arguments(roots)
    agent_command = [
        str(Path(sys.executable).resolve()),
        str(program),
        "success",
        "{agent_id}",
        "{task_id}",
        "{trial_id}",
    ]
    prepare_args = ["prepare", *common, "--agent-adapter", "subprocess", "--json"]
    for item in agent_command:
        prepare_args.extend(("--agent-command", item))

    assert main(prepare_args) == EXIT_OK
    prepared = _json_stdout(capsys)
    run_id = prepared["run_id"]
    assert prepared["status"] == "ok"

    assert main(["run-agent", run_id, *common, "--dry-run", "--json"]) == EXIT_OK
    run_dry = _json_stdout(capsys)
    assert run_dry["dry_run"] is True
    assert run_dry["agent_execution"] == "not_invoked"

    assert main(["run-agent", run_id, *common, "--json"]) == EXIT_OK
    agent_result = _json_stdout(capsys)
    assert agent_result["status"] == "ok"
    assert agent_result["run_status"] == "completed"
    assert agent_result["trials"][0]["submission_status"] == "completed"
    from review_agent_eval.artifacts import ArtifactStore

    store = ArtifactStore(roots["runs"], create_root=False)
    agent_trial_id = agent_result["trials"][0]["trial_id"]
    assert store.list_evaluations(run_id, "task-001", agent_trial_id) == ()
    assert store.list_run_evaluations(run_id) == ()

    assert (
        main(
            [
                "evaluate",
                run_id,
                *common,
                "--revision",
                "deterministic-v1",
                "--judge-provider",
                "none",
                "--dry-run",
                "--json",
            ]
        )
        == EXIT_OK
    )
    evaluate_dry = _json_stdout(capsys)
    assert evaluate_dry["dry_run"] is True
    assert evaluate_dry["judge_execution"] == "not_invoked"
    assert evaluate_dry["planned_trials"] == 1
    assert evaluate_dry["terminal_trials"] == 1

    assert (
        main(
            [
                "evaluate",
                run_id,
                *common,
                "--revision",
                "deterministic-v1",
                "--judge-provider",
                "none",
                "--json",
            ]
        )
        == EXIT_OK
    )
    evaluated = _json_stdout(capsys)
    assert evaluated["status"] == "ok"
    assert evaluated["trial_count"] == 1
    evaluation_id = evaluated["evaluation_id"]
    trial_id = evaluated["trials"][0]["trial_id"]

    assert (
        main(
            [
                "inspect",
                run_id,
                *common,
                "--task-id",
                "task-001",
                "--trial-id",
                trial_id,
                "--evaluation-id",
                evaluation_id,
                "--format",
                "json",
            ]
        )
        == EXIT_OK
    )
    inspected = _json_stdout(capsys)
    assert inspected["command"] == "inspect"
    assert inspected["inspection"]["submission"]["status"] == "completed"
    assert inspected["inspection"]["submission"]["trace"]["value"] is None
    assert "raw" not in inspected["inspection"]

    assert (
        main(
            [
                "inspect",
                run_id,
                *common,
                "--task-id",
                "task-001",
                "--trial-id",
                trial_id,
                "--evaluation-id",
                evaluation_id,
                "--dry-run",
                "--json",
            ]
        )
        == EXIT_OK
    )
    inspect_dry = _json_stdout(capsys)
    assert inspect_dry["dry_run"] is True
    assert "inspection" not in inspect_dry


def _prepare_and_run_cli_fixture(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> tuple[dict[str, Path], list[str], str]:
    """Create one terminal Run and return its roots, common args and ID."""

    roots, program = _write_cli_suite(tmp_path)
    common = _root_arguments(roots)
    agent_command = [
        str(Path(sys.executable).resolve()),
        str(program),
        "success",
        "{agent_id}",
        "{task_id}",
        "{trial_id}",
    ]
    prepare_args = [
        "prepare",
        *common,
        "--agent-adapter",
        "subprocess",
        "--json",
    ]
    for item in agent_command:
        prepare_args.extend(("--agent-command", item))
    assert main(prepare_args) == EXIT_OK
    run_id = _json_stdout(capsys)["run_id"]
    assert main(["run-agent", run_id, *common, "--json"]) == EXIT_OK
    run_result = _json_stdout(capsys)
    assert run_result["run_status"] == "completed"
    return roots, common, run_id


def test_evaluate_resume_reuses_trial_and_run_namespaces_without_reexecution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots, common, run_id = _prepare_and_run_cli_fixture(tmp_path, capsys)
    evaluate_args = [
        "evaluate",
        run_id,
        *common,
        "--revision",
        "resume-v1",
        "--judge-provider",
        "none",
        "--json",
    ]

    assert main(evaluate_args) == EXIT_OK
    first = _json_stdout(capsys)
    first_trial = first["trials"][0]
    from review_agent_eval.artifacts import ArtifactStore

    store = ArtifactStore(roots["runs"], create_root=False)
    before_trial_namespaces = store.list_evaluations(
        run_id, "task-001", first_trial["trial_id"]
    )
    before_run_namespaces = store.list_run_evaluations(run_id)

    # A true evaluator resume source-validates the committed namespace but
    # must not run either evaluator phase or invoke a Judge again.
    from review_agent_eval.orchestrator import EvaluationOrchestrator

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("resume re-executed an evaluator phase")

    monkeypatch.setattr(EvaluationOrchestrator, "_evaluate_intent", forbidden)
    monkeypatch.setattr(EvaluationOrchestrator, "_evaluate_review", forbidden)

    assert main(evaluate_args) == EXIT_OK
    resumed = _json_stdout(capsys)
    after_trial_namespaces = store.list_evaluations(
        run_id, "task-001", first_trial["trial_id"]
    )
    after_run_namespaces = store.list_run_evaluations(run_id)

    assert _without_message(resumed) == _without_message(first)
    assert resumed["evaluation_id"] == first["evaluation_id"]
    assert resumed["trials"][0]["evaluation_id"] == first_trial["evaluation_id"]
    assert after_trial_namespaces == before_trial_namespaces
    assert after_run_namespaces == before_run_namespaces


def test_evaluate_no_resume_rejects_existing_namespace_before_reexecution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _roots, common, run_id = _prepare_and_run_cli_fixture(tmp_path, capsys)
    evaluate_args = [
        "evaluate",
        run_id,
        *common,
        "--revision",
        "create-only-v1",
        "--judge-provider",
        "none",
        "--json",
    ]
    assert main(evaluate_args) == EXIT_OK
    _json_stdout(capsys)

    from review_agent_eval.orchestrator import EvaluationOrchestrator

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("create-only evaluation re-executed a Trial")

    monkeypatch.setattr(EvaluationOrchestrator, "_evaluate_intent", forbidden)
    monkeypatch.setattr(EvaluationOrchestrator, "_evaluate_review", forbidden)
    code = main([*evaluate_args, "--no-resume"])
    payload = _json_stdout(capsys)

    assert code == EXIT_CONFLICT
    assert payload["error_code"] == "conflict"


def test_evaluate_with_a_new_judge_identity_creates_a_new_namespace(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    roots, common, run_id = _prepare_and_run_cli_fixture(tmp_path, capsys)
    first_args = [
        "evaluate",
        run_id,
        *common,
        "--revision",
        "judge-v1",
        "--judge-provider",
        "none",
        "--json",
    ]
    assert main(first_args) == EXIT_OK
    first = _json_stdout(capsys)

    second_args = [
        "evaluate",
        run_id,
        *common,
        "--revision",
        "judge-v1",
        "--judge-provider",
        "fake",
        "--judge-model",
        "eval-judge",
        "--json",
    ]
    assert main(second_args) == EXIT_OK
    second = _json_stdout(capsys)

    assert second["evaluation_id"] != first["evaluation_id"]
    assert second["evaluation_revision"] == first["evaluation_revision"]
    assert second["trials"][0]["evaluation_id"] != first["trials"][0]["evaluation_id"]

    from review_agent_eval.artifacts import ArtifactStore

    store = ArtifactStore(roots["runs"], create_root=False)
    namespaces = store.list_evaluations(run_id, "task-001", second["trials"][0]["trial_id"])
    assert {item.evaluation_id for item in namespaces} == {
        first["trials"][0]["evaluation_id"],
        second["trials"][0]["evaluation_id"],
    }


def test_missing_judge_credential_is_a_redacted_precondition_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _roots, common, run_id = _prepare_and_run_cli_fixture(tmp_path, capsys)
    credential_name = "TASK12_PRIVATE_JUDGE_KEY"
    monkeypatch.delenv(credential_name, raising=False)

    code = main(
        [
            "evaluate",
            run_id,
            *common,
            "--revision",
            "missing-key-v1",
            "--judge-provider",
            "openai-compatible",
            "--judge-model",
            "judge-model-v1",
            "--judge-base-url",
            "https://judge.example.test/v1",
            "--judge-api-key-env",
            credential_name,
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == EXIT_PRECONDITION
    assert payload["error_code"] == "precondition"
    assert credential_name not in captured.out
    assert "judge.example.test" not in captured.out


def test_completed_evaluation_resume_requires_neither_judge_nor_repository(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _roots, common, run_id = _prepare_and_run_cli_fixture(tmp_path, capsys)
    credential_name = "TASK12_RESUME_JUDGE_KEY"
    monkeypatch.setenv(credential_name, "temporary-test-credential")
    evaluate_args = [
        "evaluate",
        run_id,
        *common,
        "--revision",
        "credential-free-resume-v1",
        "--judge-provider",
        "openai-compatible",
        "--judge-model",
        "judge-model-v1",
        "--judge-base-url",
        "https://judge.example.test/v1",
        "--judge-api-key-env",
        credential_name,
        "--json",
    ]

    assert main(evaluate_args) == EXIT_OK
    first = _json_stdout(capsys)
    monkeypatch.delenv(credential_name)

    import review_agent_eval.cli as cli_module

    def forbidden_repository(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("completed resume opened RepositoryPreparer")

    monkeypatch.setattr(cli_module, "_repository_preparer", forbidden_repository)
    monkeypatch.setattr(cli_module, "_load_case_bank", forbidden_repository)

    assert main([*evaluate_args, "--resume"]) == EXIT_OK
    resumed = _json_stdout(capsys)
    assert _without_message(resumed) == _without_message(first)


def test_explicit_execution_config_cannot_claim_a_different_judge_endpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    roots, common, run_id = _prepare_and_run_cli_fixture(tmp_path, capsys)
    from review_agent_eval.artifacts import ArtifactStore
    from review_agent_eval.config import EvaluatorExecutionConfig, EvaluatorRunConfig

    run_config = ArtifactStore(roots["runs"], create_root=False).load_run_config(
        run_id
    )
    bound_identity = {
        "provider": "openai-compatible",
        "model": "judge-model-v1",
        "base_url": "https://bound-judge.example.test/v1",
        "api_key_env": "TASK12_BOUND_JUDGE_KEY",
        "stage_label": "eval-judge",
    }
    evaluator = EvaluatorRunConfig(
        evaluator_id=run_config.evaluator.evaluator_id,
        evaluator_version=run_config.evaluator.evaluator_version,
        grader_version=run_config.evaluator.grader_version,
        judge_profiles=tuple(
            replace(
                profile,
                provider=bound_identity["provider"],
                model=bound_identity["model"],
                adapter_config_digest=_canonical_digest(bound_identity),
            )
            for profile in run_config.evaluator.judge_profiles
        ),
    )
    execution = EvaluatorExecutionConfig.from_resource_budgets(
        evaluator,
        run_config.resource_budgets,
    )
    execution_path = tmp_path / "explicit-evaluator-execution.json"
    execution_path.write_text(execution.to_json(), encoding="utf-8")

    code = main(
        [
            "evaluate",
            run_id,
            *common,
            "--evaluator-execution-config",
            str(execution_path),
            "--judge-provider",
            "openai-compatible",
            "--judge-model",
            "judge-model-v1",
            "--judge-base-url",
            "https://different-judge.example.test/v1",
            "--judge-api-key-env",
            "TASK12_BOUND_JUDGE_KEY",
            "--dry-run",
            "--json",
        ]
    )
    payload = _json_stdout(capsys)

    assert code == EXIT_INTEGRITY
    assert payload["error_code"] == "integrity"


def test_agent_and_judge_cli_namespaces_are_mutually_exclusive() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["run-agent", "run-x", "--suite-root", "suite", "--judge-provider", "fake"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["evaluate", "run-x", "--suite-root", "suite", "--agent-provider", "fake"]
        )

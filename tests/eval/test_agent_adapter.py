from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import pytest

from review_agent_eval.adapters.agent_factory import (
    AgentAdapterConfigError,
    build_agent_adapter_factory,
)
from review_agent_eval.adapters.base import (
    AgentAdapterError,
    AgentRunConfig,
    AgentUnderTestAdapter,
)
from review_agent_eval.adapters.subprocess_agent import (
    SubprocessAgentAdapter,
    subprocess_adapter_capabilities,
)
from review_agent_eval.artifacts import TargetAccess
from review_agent_eval.cases import (
    REPOSITORY_MATERIALIZER_PROTOCOL,
    WireContractV2,
)
from review_agent_eval.clarification import (
    ClarificationSession,
    canonical_material_claim_matcher_snapshot,
)
from review_agent_eval.config import (
    AgentConfigSnapshot,
    ResourceBudgets,
    derive_trial_id,
)
from review_agent_eval.models import (
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    EvalInput,
    EvalSubmission,
    FailureCode,
    ClarificationAction,
    ClarificationAnswer,
    ClarificationScript,
    IntentDimension,
    SchemaError,
    SubmissionStatus,
    stable_id,
    submission_status_for_failure,
)


_AGENT_PROGRAM = r'''
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def completed(agent_id, task_id, trial_id):
    return {
        "schema_version": "eval_submission_v2",
        "task_id": task_id,
        "agent_id": agent_id,
        "trial_id": trial_id,
        "eval_input_digest": eval_input_digest,
        "target_materialization_id": target_materialization_id,
        "status": "completed",
        "intent": {
            "status": "sufficient",
            "goal": "Review the requested change",
            "acceptance_criteria": [],
            "scope": [],
            "constraints": [],
            "claims": [],
            "clarification_questions": [],
            "uncertainties": [],
        },
        "review": {"findings": [], "uncertainties": []},
        "evidence": [],
        "usage": {
            "elapsed_seconds": 987.0,
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "tool_calls": 3,
            "cost_amount": 1.25,
            "cost_currency": "USD",
        },
        "trace_ref": None,
        "failure": None,
    }


def emit(payload):
    sys.stdout.buffer.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    sys.stdout.buffer.flush()


mode = sys.argv[1]
agent_id = sys.argv[2]
task_id = sys.argv[3]
trial_id = sys.argv[4]
extra = sys.argv[5:]

if mode == "block-stdin":
    time.sleep(60)
    raise SystemExit(0)

raw_input = sys.stdin.buffer.read()
invocation = json.loads(raw_input.decode("utf-8"))
trial_binding = invocation["trial_binding"]
eval_input_digest = trial_binding["eval_input_digest"]
target_materialization_id = invocation["materialization_id"]

if mode == "capture":
    capture_path = Path(extra[0])
    capture_path.write_text(
        json.dumps(
            {
                "argv": extra[1:],
                "cwd": os.getcwd(),
                "environment": dict(os.environ),
                "stdin": raw_input.decode("utf-8"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    emit(completed(agent_id, task_id, trial_id))
elif mode == "success":
    emit(completed(agent_id, task_id, trial_id))
elif mode == "stderr-then-success":
    sys.stderr.buffer.write(b"e" * int(extra[0]))
    sys.stderr.buffer.flush()
    emit(completed(agent_id, task_id, trial_id))
elif mode == "stdout-overflow":
    sys.stdout.buffer.write(b"o" * int(extra[0]))
    sys.stdout.buffer.flush()
elif mode == "stderr-overflow":
    sys.stderr.buffer.write(b"e" * int(extra[0]))
    sys.stderr.buffer.flush()
elif mode == "dual-overflow":
    amount = int(extra[0])
    def write_stdout():
        sys.stdout.buffer.write(b"o" * amount)
        sys.stdout.buffer.flush()
    def write_stderr():
        sys.stderr.buffer.write(b"e" * amount)
        sys.stderr.buffer.flush()
    first = threading.Thread(target=write_stdout)
    second = threading.Thread(target=write_stderr)
    first.start()
    second.start()
    first.join()
    second.join()
elif mode == "nonzero":
    sys.stderr.write(" ".join(extra))
    sys.stderr.flush()
    raise SystemExit(7)
elif mode == "invalid-utf8":
    sys.stdout.buffer.write(b"\xff\xfe")
    sys.stdout.buffer.flush()
elif mode == "invalid-json":
    sys.stdout.write("{not-json")
    sys.stdout.flush()
elif mode == "multiple-json":
    sys.stdout.write("{}{}")
    sys.stdout.flush()
elif mode == "schema-mismatch":
    value = completed(agent_id, task_id, trial_id)
    value["unexpected"] = True
    emit(value)
elif mode == "wrong-task":
    emit(completed(agent_id, "task-substituted", trial_id))
elif mode == "wrong-agent":
    emit(completed("agent-substituted", task_id, trial_id))
elif mode == "wrong-trial":
    emit(completed(agent_id, task_id, "trial-" + "f" * 64))
elif mode == "wrong-materialization":
    value = completed(agent_id, task_id, trial_id)
    value["target_materialization_id"] = "materialization-" + "f" * 64
    emit(value)
elif mode == "forged-clarification":
    value = completed(agent_id, task_id, trial_id)
    value["intent"]["clarification_questions"] = [
        {
            "turn_index": 1,
            "question_id": "question-forged",
            "dimension": "goal",
            "question": "Fabricated question?",
            "material_claim": "Fabricated material claim",
            "matched_answer_id": None,
            "action": None,
            "response": None,
            "resolved_values": [],
        }
    ]
    emit(value)
elif mode == "trace":
    trace_path = Path(extra[0])
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_bytes(b"t" * int(extra[2]))
    value = completed(agent_id, task_id, trial_id)
    value["trace_ref"] = {"type": "local_path", "value": extra[1]}
    emit(value)
elif mode == "trace-ref":
    value = completed(agent_id, task_id, trial_id)
    value["trace_ref"] = {"type": "local_path", "value": extra[0]}
    emit(value)
elif mode == "killed":
    if os.name == "nt":
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.argtypes = ()
        get_current_process.restype = wintypes.HANDLE
        terminate_process = kernel32.TerminateProcess
        terminate_process.argtypes = (wintypes.HANDLE, wintypes.UINT)
        terminate_process.restype = wintypes.BOOL
        terminate_process(get_current_process(), 0xC000013A)
    else:
        os.kill(os.getpid(), signal.SIGKILL)
elif mode == "spawn-descendant":
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        close_fds=True,
    )
    Path(extra[0]).write_text(str(child.pid), encoding="ascii")
    time.sleep(60)
elif mode == "spawn-detached-descendants":
    detached_program = (
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "grandchild = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(60)'],\n"
        "    close_fds=True,\n"
        "    start_new_session=True,\n"
        ")\n"
        "Path(sys.argv[1]).write_text(\n"
        "    '%d\\n%d' % (os.getpid(), grandchild.pid), encoding='ascii'\n"
        ")\n"
        "time.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", detached_program, extra[0]],
        close_fds=True,
        start_new_session=True,
    )
    time.sleep(60)
else:
    raise RuntimeError("unknown test mode")
'''


def test_failure_code_to_submission_status_mapping_is_exhaustive() -> None:
    expected = {
        FailureCode.TIMEOUT: SubmissionStatus.FAILED,
        FailureCode.NON_ZERO_EXIT: SubmissionStatus.FAILED,
        FailureCode.PROCESS_KILLED: SubmissionStatus.FAILED,
        FailureCode.ADAPTER_ERROR: SubmissionStatus.FAILED,
        FailureCode.HARNESS_MATERIALIZATION_ERROR: SubmissionStatus.FAILED,
        FailureCode.UNKNOWN: SubmissionStatus.FAILED,
        FailureCode.CLARIFICATION_REQUIRED: SubmissionStatus.BLOCKED,
        FailureCode.AGENT_BLOCKED: SubmissionStatus.BLOCKED,
        FailureCode.INVALID_JSON: SubmissionStatus.INVALID_OUTPUT,
        FailureCode.SCHEMA_MISMATCH: SubmissionStatus.INVALID_OUTPUT,
        FailureCode.OUTPUT_OVERFLOW: SubmissionStatus.INVALID_OUTPUT,
    }

    assert set(expected) == set(FailureCode)
    assert {
        code: submission_status_for_failure(code) for code in FailureCode
    } == expected


class _ForbiddenClarificationChannel:
    def ask(self, question: object) -> object:
        raise AssertionError("generic subprocess adapter must not use the channel")


@pytest.fixture
def agent_program(tmp_path: Path) -> Path:
    path = tmp_path / "subprocess_agent_fixture.py"
    path.write_text(_AGENT_PROGRAM, encoding="utf-8")
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "trial workspace"
    path.mkdir()
    return path


@pytest.fixture
def eval_input() -> EvalInput:
    return EvalInput.from_dict(
        {
            "schema_version": EVAL_INPUT_SCHEMA_VERSION,
            "task_id": "task-adapter",
            "review_target": {
                "kind": "repository",
                "repository": {
                    "source": "fixture",
                    "path": "fixtures/adapter-repository",
                    "url": None,
                    "base_revision": "a" * 40,
                    "head_revision": "b" * 40,
                },
                "review_request": {
                    "title": "Review adapter behavior",
                    "description": "Only public input belongs on stdin.",
                    "user_intent": "Return a canonical review.",
                    "review_focus": None,
                    "linked_requirements": ["REQ-ADAPTER"],
                    "project_rules": ["Do not expose private truth."],
                    "existing_ci_evidence": [],
                },
            },
        }
    )


def _command(
    program: Path,
    mode: str,
    *extra: str,
) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str(program),
        mode,
        "{agent_id}",
        "{task_id}",
        "{trial_id}",
        *extra,
    ]


def _snapshot(
    command: Sequence[object],
    *,
    environment_allowlist: Sequence[object] = (),
    agent_id: str = "agent-generic",
    adapter_override: object = ...,
) -> AgentConfigSnapshot:
    adapter: object
    if adapter_override is ...:
        adapter = {
            "kind": "subprocess-json-v2",
            "command": list(command),
            "environment_allowlist": list(environment_allowlist),
            "capabilities": subprocess_adapter_capabilities().to_dict(),
        }
    else:
        adapter = adapter_override
    return AgentConfigSnapshot(
        agent_id=agent_id,
        agent_name="Generic subprocess fixture",
        agent_version="1.0.0",
        commit="c" * 40,
        model="none",
        provider="subprocess",
        parameters={"temperature": 0, "adapter": adapter},
        prompt_config_digest="d" * 64,
    )


def _capabilities_dict() -> dict[str, object]:
    return subprocess_adapter_capabilities().to_dict()


@pytest.mark.parametrize(
    ("field", "unsupported"),
    [
        ("clarification_protocol", "canonical-clarification-v2"),
        ("trace_protocol", "none-v2"),
        ("isolation_profile", "repository-worktree-v2"),
    ],
)
def test_subprocess_factory_rejects_unimplemented_capability_protocols(
    field: str,
    unsupported: str,
) -> None:
    capabilities = _capabilities_dict()
    capabilities[field] = unsupported
    snapshot = _snapshot(
        (str(Path(sys.executable).resolve()),),
        adapter_override={
            "kind": "subprocess-json-v2",
            "command": [str(Path(sys.executable).resolve())],
            "environment_allowlist": [],
            "capabilities": capabilities,
        },
    )

    with pytest.raises(AgentAdapterConfigError, match="capabilities"):
        build_agent_adapter_factory(snapshot)


def _config(
    eval_input: EvalInput,
    command: Sequence[object],
    *,
    environment_allowlist: Sequence[object] = (),
    timeout_seconds: float = 3.0,
    max_output_bytes: int = 256 * 1024,
    agent_id: str = "agent-generic",
    adapter_override: object = ...,
) -> AgentRunConfig:
    agent = _snapshot(
        command,
        environment_allowlist=environment_allowlist,
        agent_id=agent_id,
        adapter_override=adapter_override,
    )
    run_id = stable_id("run", "subprocess-adapter-tests", agent.digest())
    trial_id = derive_trial_id(run_id, eval_input.task_id, 1)
    total = max(max_output_bytes, 16 * 1024)
    matcher = canonical_material_claim_matcher_snapshot()
    capabilities = subprocess_adapter_capabilities()
    wire = WireContractV2(
        case_schema_version=EVAL_CASE_SCHEMA_VERSION,
        input_schema_version=EVAL_INPUT_SCHEMA_VERSION,
        submission_schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
        review_target_kind=eval_input.review_target.kind,
        materializer_protocol=REPOSITORY_MATERIALIZER_PROTOCOL,
    )
    return AgentRunConfig._from_verified_binding(
        run_id=run_id,
        task_id=eval_input.task_id,
        eval_input_digest=eval_input.digest(),
        wire_contract=wire,
        adapter_capabilities=capabilities,
        adapter_capabilities_digest=capabilities.digest(),
        clarification_matcher=matcher,
        clarification_matcher_config_digest=matcher.digest(),
        trial_index=1,
        trial_id=trial_id,
        agent=agent,
        budgets=ResourceBudgets(
            agent_timeout_seconds=timeout_seconds,
            evaluator_timeout_seconds=3,
            max_agent_output_bytes=max_output_bytes,
            max_trace_bytes=4 * 1024,
            max_execution_artifact_file_bytes=total,
            max_execution_artifact_total_bytes=total,
            max_parallel_trials=1,
        ),
    )


def _target_binding(
    workspace: Path,
    eval_input: EvalInput,
) -> tuple[TargetAccess, str]:
    target = workspace / "target" / "input.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(eval_input.to_json().encode("utf-8"))
    materialization_id = stable_id(
        "materialization",
        "subprocess-adapter-test",
    )
    return (
        TargetAccess(
            target_materialization_id=materialization_id,
            readable_relative_paths=("target/input.json",),
        ),
        materialization_id,
    )


def _run(
    adapter: SubprocessAgentAdapter,
    eval_input: EvalInput,
    workspace: Path,
    config: AgentRunConfig,
) -> EvalSubmission:
    target_access, materialization_id = _target_binding(workspace, eval_input)
    return adapter.run(
        eval_input,
        workspace,
        config,
        _ForbiddenClarificationChannel(),  # type: ignore[arg-type]
        target_access=target_access,
        target_materialization_id=materialization_id,
    )


def _assert_honest_usage(submission: EvalSubmission) -> None:
    assert submission.usage.elapsed_seconds is not None
    assert submission.usage.elapsed_seconds >= 0
    assert submission.usage.input_tokens is None
    assert submission.usage.output_tokens is None
    assert submission.usage.total_tokens is None
    assert submission.usage.tool_calls is None
    assert submission.usage.cost_amount is None
    assert submission.usage.cost_currency is None


def _assert_failure(
    submission: EvalSubmission,
    status: SubmissionStatus,
    code: FailureCode,
) -> None:
    assert submission.status is status
    assert submission.failure is not None
    assert submission.failure.code is code
    assert submission.trace_ref is None
    _assert_honest_usage(submission)
    assert EvalSubmission.from_json(submission.to_json()) == submission


def test_argv_is_not_a_shell_cwd_is_workspace_and_stdin_is_only_canonical_input(
    monkeypatch: pytest.MonkeyPatch,
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    import review_agent_eval.adapters.subprocess_agent as adapter_module

    capture_one = workspace / "capture-one.json"
    capture_two = workspace / "capture-two.json"
    metachar_argument = "literal ; & | > $(not-a-shell) ^ %PATH%"
    marker = workspace / "not-a-shell"
    command_one = _command(
        agent_program,
        "capture",
        str(capture_one),
        "{workspace}",
        metachar_argument,
    )
    config_one = _config(eval_input, command_one)
    seen: list[tuple[object, Dict[str, Any]]] = []
    real_popen = adapter_module.subprocess.Popen

    def recording_popen(argv: object, *args: object, **kwargs: Any):
        seen.append((argv, dict(kwargs)))
        return real_popen(argv, *args, **kwargs)

    monkeypatch.setattr(adapter_module.subprocess, "Popen", recording_popen)
    adapter = SubprocessAgentAdapter()

    assert adapter.compatibility(eval_input, config_one).compatible
    first = _run(adapter, eval_input, workspace, config_one)
    command_two = _command(
        agent_program,
        "capture",
        str(capture_two),
        "{workspace}",
        metachar_argument,
    )
    second = _run(adapter, eval_input, workspace, _config(eval_input, command_two))

    assert isinstance(adapter, AgentUnderTestAdapter)
    assert first.status is SubmissionStatus.COMPLETED
    assert second.status is SubmissionStatus.COMPLETED
    first_capture = json.loads(capture_one.read_text(encoding="utf-8"))
    second_capture = json.loads(capture_two.read_text(encoding="utf-8"))
    first_invocation = json.loads(first_capture["stdin"])
    second_invocation = json.loads(second_capture["stdin"])
    assert second_invocation["eval_input"] == eval_input.to_dict()
    assert set(first_invocation) == {
        "schema_version",
        "eval_input",
        "trial_binding",
        "target_access",
        "materialization_id",
    }
    assert first_invocation["schema_version"] == "eval_subprocess_invocation_v2"
    assert first_invocation["eval_input"] == eval_input.to_dict()
    assert first_invocation["trial_binding"]["trial_id"] == config_one.trial_id
    assert first_invocation["trial_binding"]["eval_input_digest"] == (
        eval_input.digest()
    )
    assert first_invocation["target_access"]["readable_relative_paths"] == [
        "target/input.json"
    ]
    assert first_invocation["materialization_id"] == (
        first_invocation["target_access"]["target_materialization_id"]
    )
    assert str(workspace.resolve()) not in first_capture["stdin"]
    assert "private-case-truth-sentinel" not in first_capture["stdin"]
    assert first_capture["argv"] == [
        str(workspace.resolve()),
        metachar_argument,
    ]
    assert first_capture["cwd"] == str(workspace.resolve())
    assert not marker.exists()
    assert len(seen) == 2
    assert seen[0][0] == [
        str(Path(sys.executable).resolve()),
        str(agent_program),
        "capture",
        config_one.agent_id,
        eval_input.task_id,
        config_one.trial_id,
        str(capture_one),
        str(workspace.resolve()),
        metachar_argument,
    ]
    assert seen[0][1]["shell"] is False
    assert Path(seen[0][1]["cwd"]).resolve() == workspace.resolve()
    _assert_honest_usage(first)


def test_environment_inherits_only_fixed_runtime_keys_and_explicit_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    capture_path = workspace / "environment.json"
    monkeypatch.setenv("ADAPTER_ALLOWED_VALUE", "allowed-value")
    monkeypatch.setenv("ADAPTER_HIDDEN_SECRET", "hidden-secret")
    monkeypatch.setenv("PYTHONPATH", "python-path-must-not-leak")
    config = _config(
        eval_input,
        _command(agent_program, "capture", str(capture_path)),
        environment_allowlist=("ADAPTER_ALLOWED_VALUE",),
    )

    result = _run(SubprocessAgentAdapter(), eval_input, workspace, config)

    assert result.status is SubmissionStatus.COMPLETED
    environment = json.loads(capture_path.read_text(encoding="utf-8"))["environment"]
    folded = {key.casefold(): value for key, value in environment.items()}
    assert folded["adapter_allowed_value"] == "allowed-value"
    assert "adapter_hidden_secret" not in folded
    assert "pythonpath" not in folded
    for fixed_key in ("SystemRoot", "WINDIR", "COMSPEC", "PATHEXT"):
        if fixed_key in os.environ:
            assert folded[fixed_key.casefold()] == os.environ[fixed_key]


@pytest.mark.parametrize(
    "adapter_config",
    [
        None,
        {},
        {
            "kind": "wrong-version",
            "command": [str(Path(sys.executable).resolve())],
            "environment_allowlist": [],
        },
        {
            "kind": "subprocess-json-v2",
            "command": [str(Path(sys.executable).resolve())],
            "environment_allowlist": [],
            "capabilities": _capabilities_dict(),
            "unknown": True,
        },
        {
            "kind": "subprocess-json-v2",
            "command": "not-an-array",
            "environment_allowlist": [],
            "capabilities": _capabilities_dict(),
        },
        {
            "kind": "subprocess-json-v2",
            "command": [],
            "environment_allowlist": [],
            "capabilities": _capabilities_dict(),
        },
        {
            "kind": "subprocess-json-v2",
            "command": [str(Path(sys.executable).resolve()), 7],
            "environment_allowlist": [],
            "capabilities": _capabilities_dict(),
        },
        {
            "kind": "subprocess-json-v2",
            "command": ["relative-python"],
            "environment_allowlist": [],
            "capabilities": _capabilities_dict(),
        },
        {
            "kind": "subprocess-json-v2",
            "command": ["{workspace}"],
            "environment_allowlist": [],
            "capabilities": _capabilities_dict(),
        },
        {
            "kind": "subprocess-json-v2",
            "command": [str(Path(sys.executable).resolve()), "{unknown}"],
            "environment_allowlist": [],
            "capabilities": _capabilities_dict(),
        },
        {
            "kind": "subprocess-json-v2",
            "command": [str(Path(sys.executable).resolve()), "bad-{trial_id"],
            "environment_allowlist": [],
            "capabilities": _capabilities_dict(),
        },
        {
            "kind": "subprocess-json-v2",
            "command": [str(Path(sys.executable).resolve())],
            "environment_allowlist": "PATH",
            "capabilities": _capabilities_dict(),
        },
        {
            "kind": "subprocess-json-v2",
            "command": [str(Path(sys.executable).resolve())],
            "environment_allowlist": ["BAD-KEY"],
            "capabilities": _capabilities_dict(),
        },
        {
            "kind": "subprocess-json-v2",
            "command": [str(Path(sys.executable).resolve())],
            "environment_allowlist": ["Path", "PATH"],
            "capabilities": _capabilities_dict(),
        },
    ],
)
def test_adapter_snapshot_configuration_is_strict_and_fails_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    adapter_config: object,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    import review_agent_eval.adapters.subprocess_agent as adapter_module

    def forbidden_launch(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid adapter config reached subprocess launch")

    monkeypatch.setattr(adapter_module.subprocess, "Popen", forbidden_launch)
    config = _config(eval_input, (), adapter_override=adapter_config)

    result = _run(SubprocessAgentAdapter(), eval_input, workspace, config)

    _assert_failure(result, SubmissionStatus.FAILED, FailureCode.ADAPTER_ERROR)


def test_run_config_rejects_same_task_id_with_substituted_input_content(
    monkeypatch: pytest.MonkeyPatch,
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    import review_agent_eval.adapters.subprocess_agent as adapter_module

    config = _config(eval_input, _command(agent_program, "success"))
    substituted = EvalInput(
        schema_version=eval_input.schema_version,
        task_id=eval_input.task_id,
        review_target=replace(
            eval_input.review_target,
            review_request=replace(
                eval_input.review_target.review_request,
                description="Substituted after AgentRunConfig binding",
            ),
        ),
    )

    def forbidden_launch(*args: object, **kwargs: object) -> object:
        raise AssertionError("substituted input reached subprocess launch")

    monkeypatch.setattr(adapter_module.subprocess, "Popen", forbidden_launch)
    with pytest.raises(AgentAdapterError) as raised:
        _run(SubprocessAgentAdapter(), substituted, workspace, config)
    assert raised.value.code is FailureCode.SCHEMA_MISMATCH


def test_agent_run_config_rejects_matcher_digest_not_bound_to_snapshot(
    eval_input: EvalInput,
) -> None:
    config = _config(eval_input, (str(Path(sys.executable).resolve()),))

    with pytest.raises(
        SchemaError,
        match="matcher digest does not match its snapshot",
    ):
        AgentRunConfig._from_verified_binding(
            run_id=config.run_id,
            task_id=config.task_id,
            eval_input_digest=config.eval_input_digest,
            wire_contract=config.wire_contract,
            adapter_capabilities=config.adapter_capabilities,
            adapter_capabilities_digest=config.adapter_capabilities_digest,
            clarification_matcher=config.clarification_matcher,
            clarification_matcher_config_digest="f" * 64,
            trial_index=config.trial_index,
            trial_id=config.trial_id,
            agent=config.agent,
            budgets=config.budgets,
        )


def test_clarification_session_binds_directly_to_agent_run_config(
    eval_input: EvalInput,
) -> None:
    config = _config(eval_input, (str(Path(sys.executable).resolve()),))
    session = ClarificationSession(
        ClarificationScript(
            max_rounds=1,
            answers=(
                ClarificationAnswer(
                    answer_id="answer-run-bound",
                    dimension=IntentDimension.GOAL,
                    material_claim="Preserve deterministic behavior",
                    action=ClarificationAction.CONFIRM,
                    response="Yes",
                    corrected_values=(),
                ),
            ),
        ),
        run_binding=config,
    )

    exchange = session.channel.ask(
        question_id="question-run-bound",
        dimension=IntentDimension.GOAL,
        question="Should deterministic behavior be preserved?",
        material_claim="Preserve deterministic behavior",
        proposed_values=("Preserve deterministic behavior",),
    )

    assert exchange.matched_answer_id == "answer-run-bound"
    assert session.match_receipts[0].matcher_digest == (
        config.clarification_matcher_config_digest
    )


def test_one_stateless_instance_uses_each_runtime_agent_snapshot(
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    adapter = SubprocessAgentAdapter()
    first_path = workspace / "first-snapshot.json"
    second_path = workspace / "second-snapshot.json"
    first_config = _config(
        eval_input,
        _command(agent_program, "capture", str(first_path), "first"),
        agent_id="agent-first",
    )
    second_config = _config(
        eval_input,
        _command(agent_program, "capture", str(second_path), "second"),
        agent_id="agent-second",
    )

    first = _run(adapter, eval_input, workspace, first_config)
    second = _run(adapter, eval_input, workspace, second_config)

    assert first.agent_id == "agent-first"
    assert second.agent_id == "agent-second"
    assert json.loads(first_path.read_text(encoding="utf-8"))["argv"] == ["first"]
    assert json.loads(second_path.read_text(encoding="utf-8"))["argv"] == ["second"]
    assert first_config.agent.digest() != second_config.agent.digest()


def test_zero_finding_completed_submission_is_legal_and_usage_is_measured(
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    config = _config(eval_input, _command(agent_program, "success"))

    result = _run(SubprocessAgentAdapter(), eval_input, workspace, config)

    assert result.status is SubmissionStatus.COMPLETED
    assert result.review is not None
    assert result.review.findings == ()
    assert result.failure is None
    _assert_honest_usage(result)
    assert result.usage.elapsed_seconds != 987.0
    assert EvalSubmission.from_json(result.to_json()) == result


def test_stdout_and_stderr_are_drained_concurrently_with_one_combined_budget(
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    config = _config(
        eval_input,
        _command(agent_program, "stderr-then-success", "100000"),
        max_output_bytes=160000,
    )

    result = _run(SubprocessAgentAdapter(), eval_input, workspace, config)

    assert result.status is SubmissionStatus.COMPLETED


@pytest.mark.parametrize(
    ("mode", "amount"),
    [
        ("stdout-overflow", 4096),
        ("stderr-overflow", 4096),
        ("dual-overflow", 700),
    ],
)
def test_output_overflow_is_bounded_across_both_streams(
    mode: str,
    amount: int,
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    config = _config(
        eval_input,
        _command(agent_program, mode, str(amount)),
        max_output_bytes=1024,
    )

    result = _run(SubprocessAgentAdapter(), eval_input, workspace, config)

    _assert_failure(
        result,
        SubmissionStatus.INVALID_OUTPUT,
        FailureCode.OUTPUT_OVERFLOW,
    )


def test_timeout_cannot_be_defeated_by_a_child_that_never_reads_large_stdin(
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    large_input = replace(
        eval_input,
        review_target=replace(
            eval_input.review_target,
            review_request=replace(
                eval_input.review_target.review_request,
                project_rules=tuple(
                    "rule-%03d-%s" % (index, "x" * 7000)
                    for index in range(200)
                ),
            ),
        ),
    )
    config = _config(
        large_input,
        _command(agent_program, "block-stdin"),
        timeout_seconds=0.2,
    )
    started = time.monotonic()

    result = _run(SubprocessAgentAdapter(), large_input, workspace, config)

    assert time.monotonic() - started < 5
    _assert_failure(result, SubmissionStatus.FAILED, FailureCode.TIMEOUT)


def _write_fake_proc_stat(
    proc_root: Path,
    pid: int,
    parent_pid: int,
    start_time: int,
    *,
    name: str = "worker",
) -> None:
    process_dir = proc_root / str(pid)
    process_dir.mkdir()
    fields_after_state = [
        str(parent_pid),
        *("0" for _ in range(17)),
        str(start_time),
    ]
    (process_dir / "stat").write_text(
        "%d (%s) S %s\n" % (pid, name, " ".join(fields_after_state)),
        encoding="ascii",
    )


def test_proc_snapshot_collects_nested_descendants_independent_of_session(
    tmp_path: Path,
) -> None:
    import review_agent_eval.adapters.subprocess_agent as adapter_module

    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_fake_proc_stat(proc_root, 100, 1, 10, name="root")
    _write_fake_proc_stat(proc_root, 101, 100, 11, name="detached ) child")
    _write_fake_proc_stat(proc_root, 102, 101, 12, name="nested detached")
    _write_fake_proc_stat(proc_root, 200, 1, 13, name="unrelated")

    snapshot = adapter_module._read_linux_process_snapshot(proc_root)

    assert snapshot is not None
    processes, complete = snapshot
    descendants, tree_complete = adapter_module._collect_posix_descendants(
        100, processes
    )
    assert complete
    assert tree_complete
    assert set(descendants) == {101, 102}
    assert descendants[101].birth_marker == "proc:11"


def test_ps_snapshot_fallback_preserves_parent_and_process_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import review_agent_eval.adapters.subprocess_agent as adapter_module

    monkeypatch.setattr(adapter_module.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(adapter_module.os, "access", lambda _path, _mode: True)
    monkeypatch.setattr(
        adapter_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                b"100 1 Thu Jul 16 09:00:00 2026 S\n"
                b"101 100 Thu Jul 16 09:00:01 2026 Ss\n"
            ),
        ),
    )

    snapshot = adapter_module._read_ps_process_snapshot()

    assert snapshot is not None
    processes, complete = snapshot
    assert complete
    assert processes[101].parent_pid == 100
    assert processes[101].birth_marker == "ps:Thu Jul 16 09:00:01 2026"
    assert processes[101].state == "Ss"


def test_descendant_tracker_retains_reparented_process_but_rejects_pid_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import review_agent_eval.adapters.subprocess_agent as adapter_module

    identity = adapter_module._PosixProcessIdentity
    snapshots = iter(
        (
            (
                {
                    100: identity(100, 1, "proc:10", "S"),
                    101: identity(101, 100, "proc:11", "S"),
                },
                True,
            ),
            (
                {
                    101: identity(101, 1, "proc:11", "S"),
                    102: identity(102, 101, "proc:12", "S"),
                },
                True,
            ),
            ({101: identity(101, 1, "proc:99", "S")}, True),
        )
    )
    monkeypatch.setattr(
        adapter_module,
        "_read_posix_process_snapshot",
        lambda: next(snapshots),
    )
    tracker = adapter_module._PosixDescendantTracker(100)

    assert [item.pid for item in tracker.observe()] == [101]
    assert [item.pid for item in tracker.observe()] == [101, 102]
    assert tracker.observe() == ()


def _process_is_running(pid: int) -> bool:
    if os.name != "nt":
        stat_path = Path("/proc") / str(pid) / "stat"
        try:
            data = stat_path.read_bytes()
            closing = data.rfind(b")")
            state = data[closing + 2 :].split(maxsplit=1)[0]
            if state in {b"X", b"Z"}:
                return False
        except (FileNotFoundError, IndexError, OSError):
            pass
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_process.restype = wintypes.HANDLE
    handle = open_process(0x00100000, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == 259
    finally:
        kernel32.CloseHandle(handle)


def _wait_until_processes_stop(pids: Sequence[int]) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not any(_process_is_running(pid) for pid in pids):
            return True
        time.sleep(0.02)
    return not any(_process_is_running(pid) for pid in pids)


def _force_stop_processes(pids: Sequence[int]) -> None:
    if os.name == "nt":
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_timeout_terminates_descendant_process_tree(
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    pid_path = workspace / "descendant.pid"
    config = _config(
        eval_input,
        _command(agent_program, "spawn-descendant", str(pid_path)),
        timeout_seconds=0.75,
    )

    result = _run(SubprocessAgentAdapter(), eval_input, workspace, config)

    _assert_failure(result, SubmissionStatus.FAILED, FailureCode.TIMEOUT)
    assert pid_path.exists()
    descendant_pid = int(pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 5
    while _process_is_running(descendant_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _process_is_running(descendant_pid)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX setsid")
def test_timeout_terminates_nested_detached_setsid_descendants(
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    pid_path = workspace / "detached-timeout-descendants.pid"
    config = _config(
        eval_input,
        _command(
            agent_program,
            "spawn-detached-descendants",
            str(pid_path),
        ),
        timeout_seconds=0.75,
    )
    pids: tuple[int, ...] = ()

    result = _run(SubprocessAgentAdapter(), eval_input, workspace, config)
    assert pid_path.exists()
    pids = tuple(int(value) for value in pid_path.read_text().splitlines())
    try:
        _assert_failure(result, SubmissionStatus.FAILED, FailureCode.TIMEOUT)
        assert len(pids) == 2
        assert _wait_until_processes_stop(pids)
    finally:
        _force_stop_processes(pids)


def test_cancellation_terminates_descendant_process_tree(
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    pid_path = workspace / "cancelled-descendant.pid"
    config = _config(
        eval_input,
        _command(agent_program, "spawn-descendant", str(pid_path)),
        timeout_seconds=30,
    )
    cancelled = threading.Event()
    result_holder: Dict[str, EvalSubmission] = {}
    target_access, materialization_id = _target_binding(workspace, eval_input)

    worker = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result",
            SubprocessAgentAdapter().run(
                eval_input,
                workspace,
                config,
                _ForbiddenClarificationChannel(),  # type: ignore[arg-type]
                target_access=target_access,
                target_materialization_id=materialization_id,
                cancel_event=cancelled,
            ),
        ),
        daemon=True,
    )
    worker.start()
    deadline = time.monotonic() + 5
    while not pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pid_path.exists()
    cancelled.set()
    worker.join(5)

    assert not worker.is_alive()
    result = result_holder["result"]
    _assert_failure(result, SubmissionStatus.FAILED, FailureCode.PROCESS_KILLED)
    descendant_pid = int(pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 5
    while _process_is_running(descendant_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _process_is_running(descendant_pid)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX setsid")
def test_cancellation_terminates_nested_detached_setsid_descendants(
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    pid_path = workspace / "detached-cancelled-descendants.pid"
    config = _config(
        eval_input,
        _command(
            agent_program,
            "spawn-detached-descendants",
            str(pid_path),
        ),
        timeout_seconds=30,
    )
    cancelled = threading.Event()
    result_holder: Dict[str, EvalSubmission] = {}
    pids: tuple[int, ...] = ()
    target_access, materialization_id = _target_binding(workspace, eval_input)
    worker = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result",
            SubprocessAgentAdapter().run(
                eval_input,
                workspace,
                config,
                _ForbiddenClarificationChannel(),  # type: ignore[arg-type]
                target_access=target_access,
                target_materialization_id=materialization_id,
                cancel_event=cancelled,
            ),
        ),
        daemon=True,
    )

    worker.start()
    try:
        deadline = time.monotonic() + 5
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pid_path.exists()
        pids = tuple(int(value) for value in pid_path.read_text().splitlines())
        cancelled.set()
        worker.join(5)
        assert not worker.is_alive()
        _assert_failure(
            result_holder["result"],
            SubmissionStatus.FAILED,
            FailureCode.PROCESS_KILLED,
        )
        assert len(pids) == 2
        assert _wait_until_processes_stop(pids)
    finally:
        cancelled.set()
        worker.join(5)
        _force_stop_processes(pids)


def test_nonzero_exit_and_raw_stderr_do_not_leak_secrets(
    monkeypatch: pytest.MonkeyPatch,
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    argv_secret = "argv-secret-4f57"
    environment_secret = "environment-secret-91ac"
    monkeypatch.setenv("ADAPTER_ALLOWED_SECRET", environment_secret)
    config = _config(
        eval_input,
        _command(
            agent_program,
            "nonzero",
            argv_secret,
            environment_secret,
        ),
        environment_allowlist=("ADAPTER_ALLOWED_SECRET",),
    )

    result = _run(SubprocessAgentAdapter(), eval_input, workspace, config)

    _assert_failure(result, SubmissionStatus.FAILED, FailureCode.NON_ZERO_EXIT)
    rendered = result.to_json()
    assert argv_secret not in rendered
    assert environment_secret not in rendered


def test_signal_or_control_kill_is_process_killed(
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    config = _config(eval_input, _command(agent_program, "killed"))

    result = _run(SubprocessAgentAdapter(), eval_input, workspace, config)

    _assert_failure(result, SubmissionStatus.FAILED, FailureCode.PROCESS_KILLED)


@pytest.mark.parametrize(
    ("mode", "expected_code"),
    [
        ("invalid-utf8", FailureCode.INVALID_JSON),
        ("invalid-json", FailureCode.INVALID_JSON),
        ("multiple-json", FailureCode.INVALID_JSON),
        ("schema-mismatch", FailureCode.SCHEMA_MISMATCH),
        ("wrong-task", FailureCode.SCHEMA_MISMATCH),
        ("wrong-agent", FailureCode.SCHEMA_MISMATCH),
        ("wrong-trial", FailureCode.SCHEMA_MISMATCH),
        ("wrong-materialization", FailureCode.SCHEMA_MISMATCH),
        ("forged-clarification", FailureCode.SCHEMA_MISMATCH),
    ],
)
def test_invalid_json_schema_binding_and_forged_transcript_fail_closed(
    mode: str,
    expected_code: FailureCode,
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    config = _config(eval_input, _command(agent_program, mode))

    result = _run(SubprocessAgentAdapter(), eval_input, workspace, config)

    _assert_failure(result, SubmissionStatus.INVALID_OUTPUT, expected_code)


def test_launch_failure_is_sanitized_adapter_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    executable_secret = "missing-argv-secret-6201"
    environment_secret = "launch-environment-secret-e831"
    missing_executable = workspace / executable_secret
    monkeypatch.setenv("ADAPTER_LAUNCH_SECRET", environment_secret)
    config = _config(
        eval_input,
        [str(missing_executable), environment_secret],
        environment_allowlist=("ADAPTER_LAUNCH_SECRET",),
    )

    result = _run(SubprocessAgentAdapter(), eval_input, workspace, config)

    _assert_failure(result, SubmissionStatus.FAILED, FailureCode.ADAPTER_ERROR)
    rendered = result.to_json()
    assert executable_secret not in rendered
    assert environment_secret not in rendered


def test_internal_boundary_failure_is_sanitized_adapter_error(
    monkeypatch: pytest.MonkeyPatch,
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    import review_agent_eval.adapters.subprocess_agent as adapter_module

    internal_secret = "internal-boundary-secret-72be"

    def fail_boundary(*args: object, **kwargs: object) -> object:
        raise RuntimeError(internal_secret)

    monkeypatch.setattr(adapter_module, "run_bounded_process", fail_boundary)
    config = _config(eval_input, _command(agent_program, "success"))

    result = _run(SubprocessAgentAdapter(), eval_input, workspace, config)

    _assert_failure(result, SubmissionStatus.FAILED, FailureCode.ADAPTER_ERROR)
    assert internal_secret not in result.to_json()


def test_local_trace_must_be_relative_bounded_and_inside_workspace(
    agent_program: Path,
    workspace: Path,
    eval_input: EvalInput,
) -> None:
    trace_path = workspace / "agent-trace" / "trace.jsonl"
    relative_trace = trace_path.relative_to(workspace).as_posix()
    valid = _run(
        SubprocessAgentAdapter(),
        eval_input,
        workspace,
        _config(
            eval_input,
            _command(
                agent_program,
                "trace",
                str(trace_path),
                relative_trace,
                "128",
            ),
        ),
    )
    assert valid.status is SubmissionStatus.COMPLETED
    assert valid.trace_ref is not None
    assert valid.trace_ref.value == relative_trace

    oversized_path = workspace / "oversized-trace.bin"
    oversized = _run(
        SubprocessAgentAdapter(),
        eval_input,
        workspace,
        _config(
            eval_input,
            _command(
                agent_program,
                "trace",
                str(oversized_path),
                oversized_path.name,
                "8192",
            ),
        ),
    )
    _assert_failure(
        oversized,
        SubmissionStatus.INVALID_OUTPUT,
        FailureCode.OUTPUT_OVERFLOW,
    )

    absolute = _run(
        SubprocessAgentAdapter(),
        eval_input,
        workspace,
        _config(
            eval_input,
            _command(agent_program, "trace-ref", str(trace_path)),
        ),
    )
    _assert_failure(
        absolute,
        SubmissionStatus.INVALID_OUTPUT,
        FailureCode.SCHEMA_MISMATCH,
    )

    escaped = _run(
        SubprocessAgentAdapter(),
        eval_input,
        workspace,
        _config(
            eval_input,
            _command(agent_program, "trace-ref", "../outside-trace"),
        ),
    )
    _assert_failure(
        escaped,
        SubmissionStatus.INVALID_OUTPUT,
        FailureCode.SCHEMA_MISMATCH,
    )

    linked_source = workspace / "linked-source.txt"
    linked_trace = workspace / "linked-trace.txt"
    linked_source.write_bytes(b"trace")
    os.link(linked_source, linked_trace)
    hardlinked = _run(
        SubprocessAgentAdapter(),
        eval_input,
        workspace,
        _config(
            eval_input,
            _command(agent_program, "trace-ref", linked_trace.name),
        ),
    )
    _assert_failure(
        hardlinked,
        SubmissionStatus.INVALID_OUTPUT,
        FailureCode.SCHEMA_MISMATCH,
    )

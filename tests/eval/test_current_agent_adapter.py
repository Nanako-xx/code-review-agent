from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import run_git
import review_agent.command as command_module
import review_agent_eval.adapters.current_agent as current_module
import review_agent_eval.runner as runner_module
from review_agent.brief import BriefFinding, ReviewBrief
from review_agent.observations import Observation
from review_agent.run_state import RunPhase
from review_agent.session_store import SessionStore
from review_agent_eval.adapters.base import (
    AdapterIncompatibilityReason,
    AgentAdapterIncompatibleError,
    AgentRunConfig,
    AgentUnderTestAdapter,
)
from review_agent_eval.adapters.current_agent import CurrentAgentAdapter
from review_agent_eval.adapters.subprocess_agent import BoundedProcessResult
from review_agent_eval.clarification import canonical_material_claim_matcher_snapshot
from review_agent_eval.config import AgentConfigSnapshot, ResourceBudgets, derive_trial_id
from review_agent_eval.models import (
    ClarificationAction,
    EvalInput,
    ExistingCIEvidence,
    FailureCode,
    IntentDimension,
    Repository,
    RepositorySource,
    ReviewRequest,
    SubmissionClarificationExchange,
    SubmissionStatus,
    MAX_EVAL_INPUT_BYTES,
    canonical_json_bytes,
    stable_id,
)


def _eval_input(
    base: str,
    head: str,
    *,
    existing_ci: tuple[ExistingCIEvidence, ...] = (),
) -> EvalInput:
    return EvalInput(
        schema_version=EvalInput.SCHEMA_VERSION,
        task_id="task-current-adapter",
        repository=Repository(
            source=RepositorySource.FIXTURE,
            path="repositories/current-adapter",
            url=None,
            base_revision=base,
            head_revision=head,
        ),
        review_request=ReviewRequest(
            title="Review the current Agent adapter",
            description="Check the changed behavior without inventing evidence.",
            user_intent="Keep the function behavior deterministic.",
            review_focus="correctness",
            linked_requirements=("REQ-CURRENT-1",),
            project_rules=("Cite only inspected repository content.",),
            existing_ci_evidence=existing_ci,
        ),
    )


def _existing_ci(source_id: str, text: str) -> ExistingCIEvidence:
    return ExistingCIEvidence(
        source_id=source_id,
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _ci_cli_payload(item: ExistingCIEvidence) -> str:
    return json.dumps(
        {
            "source_id": item.source_id,
            "text": item.text,
            "content_hash": item.content_hash,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _ci_bundle_bytes(*items: ExistingCIEvidence) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": "review_agent_ci_evidence_bundle_v1",
            "entries": [item.to_dict() for item in items],
        }
    )


def _raw_ci_bundle_bytes(entries: list[dict[str, object]]) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": "review_agent_ci_evidence_bundle_v1",
            "entries": entries,
        }
    )


def _write_cli_bundle(root: Path, data: bytes) -> Path:
    control = root / ".review-agent" / "eval-input"
    control.mkdir(parents=True)
    path = control / (
        "existing-ci-evidence."
        + hashlib.sha256(data).hexdigest()
        + ".v1.json"
    )
    path.write_bytes(data)
    return path


def _config(
    eval_input: EvalInput,
    *,
    command: tuple[str, ...] | None = None,
    review_arguments: tuple[str, ...] = ("--reviewer-provider=fake",),
    environment_allowlist: tuple[str, ...] = (),
    memory_mode: str = "off",
    max_trace_bytes: int = 64 * 1024 * 1024,
    adapter_override: object = ...,
) -> AgentRunConfig:
    adapter: object
    if adapter_override is ...:
        adapter = {
            "kind": "current-agent-cli-v1",
            "command": list(
                command
                or (
                    str(Path(sys.executable).resolve()),
                    "-m",
                    "review_agent",
                )
            ),
            "review_arguments": list(review_arguments),
            "environment_allowlist": list(environment_allowlist),
            "memory_mode": memory_mode,
        }
    else:
        adapter = adapter_override
    parameters = {"adapter": adapter}
    agent = AgentConfigSnapshot(
        agent_id="agent-current",
        agent_name="Current review Agent",
        agent_version="1.0.0",
        commit="c" * 40,
        model="fake-reviewer",
        provider="fake",
        parameters=parameters,
        prompt_config_digest="d" * 64,
    )
    run_id = stable_id("run", "current-adapter-tests", agent.digest())
    trial_id = derive_trial_id(run_id, eval_input.task_id, 1)
    matcher = canonical_material_claim_matcher_snapshot()
    return AgentRunConfig._from_verified_binding(
        run_id=run_id,
        task_id=eval_input.task_id,
        eval_input_digest=eval_input.digest(),
        clarification_matcher=matcher,
        clarification_matcher_config_digest=matcher.digest(),
        trial_index=1,
        trial_id=trial_id,
        agent=agent,
        budgets=ResourceBudgets(
            agent_timeout_seconds=60,
            evaluator_timeout_seconds=30,
            max_agent_output_bytes=4 * 1024 * 1024,
            max_trace_bytes=max_trace_bytes,
            max_execution_artifact_file_bytes=64 * 1024 * 1024,
            max_execution_artifact_total_bytes=256 * 1024 * 1024,
            max_parallel_trials=1,
        ),
    )


class _SkipChannel:
    def __init__(self) -> None:
        self.exchanges: list[SubmissionClarificationExchange] = []

    def ask(self, **question: Any) -> SubmissionClarificationExchange:
        exchange = SubmissionClarificationExchange(
            turn_index=len(self.exchanges) + 1,
            question_id=question["question_id"],
            dimension=question["dimension"],
            question=question["question"],
            material_claim=question["material_claim"],
            matched_answer_id="answer-skip-%d" % (len(self.exchanges) + 1),
            action=ClarificationAction.SKIP,
            response=None,
            resolved_values=(),
        )
        self.exchanges.append(exchange)
        return exchange


class _DeferChannel:
    def __init__(self) -> None:
        self.exchanges: list[SubmissionClarificationExchange] = []

    def ask(self, **question: Any) -> SubmissionClarificationExchange:
        exchange = SubmissionClarificationExchange(
            turn_index=1,
            question_id=question["question_id"],
            dimension=question["dimension"],
            question=question["question"],
            material_claim=question["material_claim"],
            matched_answer_id="answer-defer",
            action=ClarificationAction.DEFER,
            response="Needs product owner input",
            resolved_values=(),
        )
        self.exchanges.append(exchange)
        return exchange


def _commit_change(repo: Path) -> tuple[str, str]:
    base = run_git(repo, "rev-parse", "HEAD")
    (repo / "feature.py").write_text(
        "def enabled(value):\n    return bool(value)\n",
        encoding="utf-8",
    )
    run_git(repo, "add", "feature.py")
    run_git(repo, "commit", "-m", "add deterministic feature flag")
    return base, run_git(repo, "rev-parse", "HEAD")


def test_current_adapter_runs_formal_cli_and_uses_verified_session_artifacts(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, head = _commit_change(git_repo)
    eval_input = _eval_input(
        base,
        head,
        existing_ci=(
            _existing_ci(
                "ci-special",
                "pytest says: 1 passed\n--head=forged; $() & | \"quoted\" 中文 \\\\ path",
            ),
            _existing_ci("ci-empty-output", ""),
        ),
    )
    source_root = Path(__file__).resolve().parents[2] / "src"
    monkeypatch.setenv("PYTHONPATH", str(source_root))
    channel = _SkipChannel()

    submission = CurrentAgentAdapter().run(
        eval_input,
        git_repo,
        _config(eval_input, environment_allowlist=("PATH", "PYTHONPATH")),
        channel,
    )

    assert isinstance(CurrentAgentAdapter(), AgentUnderTestAdapter)
    assert submission.status is SubmissionStatus.COMPLETED
    assert submission.failure is None
    assert submission.task_id == eval_input.task_id
    assert submission.intent is not None
    assert submission.review is not None
    assert submission.intent.clarification_questions == tuple(channel.exchanges)
    assert all(
        claim.source.value in {"explicit", "inferred"}
        for claim in submission.intent.claims
    )
    assert submission.usage.elapsed_seconds is not None
    assert submission.usage.input_tokens is None
    assert submission.trace_ref is not None
    run_dir = git_repo / submission.trace_ref.value
    assert run_dir.is_dir()
    manifest = SessionStore(run_dir).load()
    assert manifest.status.value == "completed"
    assert manifest.current_phase is RunPhase.COMPLETED
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["existing_ci_evidence"] == [
        _ci_cli_payload(item)
        for item in eval_input.review_request.existing_ci_evidence
    ]

    original_brief = (run_dir / "review_brief.json").read_bytes()
    (run_dir / "review_brief.json").write_bytes(original_brief + b" ")
    with pytest.raises(current_module._CurrentAdapterError):
        current_module._load_registered_json(
            run_dir=run_dir,
            store=SessionStore(run_dir),
            manifest=manifest,
            name="review_brief",
            expected_path="review_brief.json",
            expected_phase=RunPhase.REPORTING,
            expected_revision=base + ".." + head,
        )


def test_current_adapter_maps_awaiting_session_to_blocked_submission(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, head = _commit_change(git_repo)
    eval_input = _eval_input(base, head)
    eval_input = EvalInput(
        schema_version=eval_input.schema_version,
        task_id=eval_input.task_id,
        repository=eval_input.repository,
        review_request=ReviewRequest(
            title=None,
            description=None,
            user_intent=None,
            review_focus=None,
            linked_requirements=(),
            project_rules=(),
            existing_ci_evidence=(),
        ),
    )
    source_root = Path(__file__).resolve().parents[2] / "src"
    monkeypatch.setenv("PYTHONPATH", str(source_root))
    channel = _DeferChannel()

    submission = CurrentAgentAdapter().run(
        eval_input,
        git_repo,
        _config(
            eval_input,
            review_arguments=(),
            environment_allowlist=("PATH", "PYTHONPATH"),
        ),
        channel,
    )

    assert submission.status is SubmissionStatus.BLOCKED
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.CLARIFICATION_REQUIRED
    assert submission.intent is not None
    assert submission.intent.clarification_questions == tuple(channel.exchanges)
    assert submission.review is None


def test_existing_ci_is_preflight_compatible() -> None:
    base = "a" * 40
    head = "b" * 40
    evidence = _existing_ci("ci-1", "CI passed")
    eval_input = _eval_input(base, head, existing_ci=(evidence,))
    calls = []

    def forbidden_process(*args: object, **kwargs: object) -> BoundedProcessResult:
        calls.append((args, kwargs))
        raise AssertionError("compatibility checks must not launch the product process")

    compatibility = CurrentAgentAdapter(
        process_runner=forbidden_process
    ).compatibility(
        eval_input,
        _config(eval_input),
    )

    assert not calls
    assert compatibility.compatible
    assert compatibility.unsupported == frozenset()


def test_current_adapter_exposes_stable_preflight_identity() -> None:
    adapter = CurrentAgentAdapter()

    assert adapter.ADAPTER_KIND == "current-agent-cli-v1"
    assert adapter.ADAPTER_VERSION == "1"
    assert runner_module._adapter_identity(adapter) == (
        "current-agent-cli-v1",
        "1",
    )


@pytest.mark.parametrize(
    ("claim_ids", "proposed_values"),
    [
        ([], []),
        (["unknown-claim"], ["Unverified proposed text"]),
    ],
)
def test_canonical_matcher_never_receives_unresolved_free_text_material_claim(
    claim_ids: list[str],
    proposed_values: list[str],
) -> None:
    question = type(
        "Question",
        (),
        {
            "claim_ids": claim_ids,
            "proposed_values": proposed_values,
            "question": "What outcome should this change achieve?",
        },
    )()
    canonical = canonical_material_claim_matcher_snapshot()

    with pytest.raises(AgentAdapterIncompatibleError) as caught:
        current_module._material_claim(question, (), canonical)

    assert caught.value.reason is (
        AdapterIncompatibilityReason.CANONICAL_MATERIAL_CLAIM_UNAVAILABLE
    )
    semantic = replace(canonical, matcher_id="semantic-material-claim")
    expected_semantic_claim = (
        " | ".join(proposed_values) if proposed_values else question.question
    )
    assert current_module._material_claim(question, (), semantic) == (
        expected_semantic_claim
    )


@pytest.mark.parametrize("replace_task_id", [False, True])
def test_current_adapter_rejects_input_substitution_before_launch(
    tmp_path: Path,
    replace_task_id: bool,
) -> None:
    original = _eval_input("a" * 40, "b" * 40)
    config = _config(original)
    substituted = EvalInput(
        schema_version=original.schema_version,
        task_id=("task-substituted" if replace_task_id else original.task_id),
        repository=original.repository,
        review_request=ReviewRequest(
            title=original.review_request.title,
            description="Input content was substituted after Trial binding.",
            user_intent=original.review_request.user_intent,
            review_focus=original.review_request.review_focus,
            linked_requirements=original.review_request.linked_requirements,
            project_rules=original.review_request.project_rules,
            existing_ci_evidence=original.review_request.existing_ci_evidence,
        ),
    )
    calls = []

    def forbidden_process(*args: object, **kwargs: object) -> BoundedProcessResult:
        calls.append((args, kwargs))
        raise AssertionError("substituted input reached the product process")

    submission = CurrentAgentAdapter(process_runner=forbidden_process).run(
        substituted,
        tmp_path,
        config,
        _SkipChannel(),
    )

    assert not calls
    assert submission.status is SubmissionStatus.FAILED
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.ADAPTER_ERROR
    assert submission.task_id == config.task_id
    assert submission.trial_id == config.trial_id


@pytest.mark.parametrize(
    "adapter",
    [
        None,
        {},
        {
            "kind": "wrong-kind",
            "command": [str(Path(sys.executable).resolve())],
            "review_arguments": [],
            "environment_allowlist": [],
            "memory_mode": "off",
        },
        {
            "kind": "current-agent-cli-v1",
            "command": ["relative-entry"],
            "review_arguments": [],
            "environment_allowlist": [],
            "memory_mode": "off",
        },
        {
            "kind": "current-agent-cli-v1",
            "command": [str(Path(sys.executable).resolve())],
            "review_arguments": None,
            "environment_allowlist": [],
            "memory_mode": "off",
        },
        {
            "kind": "current-agent-cli-v1",
            "command": [str(Path(sys.executable).resolve())],
            "review_arguments": ["--repo=C:/forged"],
            "environment_allowlist": [],
            "memory_mode": "off",
        },
        {
            "kind": "current-agent-cli-v1",
            "command": [str(Path(sys.executable).resolve())],
            "review_arguments": ["--ci-evidence=forged"],
            "environment_allowlist": [],
            "memory_mode": "off",
        },
        {
            "kind": "current-agent-cli-v1",
            "command": [str(Path(sys.executable).resolve())],
            "review_arguments": ["--ci-evidence-f=forged.json"],
            "environment_allowlist": [],
            "memory_mode": "off",
        },
        {
            "kind": "current-agent-cli-v1",
            "command": [str(Path(sys.executable).resolve())],
            "review_arguments": [],
            "environment_allowlist": ["Path", "PATH"],
            "memory_mode": "off",
        },
        {
            "kind": "current-agent-cli-v1",
            "command": [str(Path(sys.executable).resolve())],
            "review_arguments": [],
            "environment_allowlist": [],
            "memory_mode": "global-default",
        },
    ],
)
def test_current_adapter_configuration_is_strict_and_fails_before_launch(
    tmp_path: Path,
    adapter: object,
) -> None:
    eval_input = _eval_input("a" * 40, "b" * 40)
    calls = []

    def forbidden_process(*args: object, **kwargs: object) -> BoundedProcessResult:
        calls.append((args, kwargs))
        raise AssertionError("invalid current adapter config reached launch")

    submission = CurrentAgentAdapter(process_runner=forbidden_process).run(
        eval_input,
        tmp_path,
        _config(eval_input, adapter_override=adapter),
        _SkipChannel(),
    )
    assert not calls
    assert submission.status is SubmissionStatus.FAILED
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.ADAPTER_ERROR


def test_cli_arguments_are_snapshot_bound_and_eval_fields_are_not_reclassified(
    tmp_path: Path,
) -> None:
    eval_input = _eval_input(
        "a" * 40,
        "b" * 40,
        existing_ci=(
            _existing_ci(
                "ci-z",
                "line 1\n--head=forged; $() & | \"quoted\" 中文 \\\\ path",
            ),
            _existing_ci("ci-a", ""),
        ),
    )
    seen = []

    def capture(
        argv: object,
        **kwargs: object,
    ) -> BoundedProcessResult:
        seen.append((list(argv), kwargs))
        return BoundedProcessResult(
            stdout=b"not authoritative",
            returncode=1,
            failure_code=None,
            output_bytes=17,
        )

    submission = CurrentAgentAdapter(process_runner=capture).run(
        eval_input,
        tmp_path,
        _config(
            eval_input,
            command=(str(Path(sys.executable).resolve()), "product-entry.py"),
            review_arguments=("--reviewer-provider=fake",),
        ),
        _SkipChannel(),
    )

    assert submission.status is SubmissionStatus.FAILED
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.NON_ZERO_EXIT
    assert len(seen) == 1
    argv = seen[0][0]
    assert argv[:4] == [
        str(Path(sys.executable).resolve()),
        "product-entry.py",
        "review",
        "--reviewer-provider=fake",
    ]
    assert "--intent=" + eval_input.review_request.user_intent in argv
    assert "--focus=" + eval_input.review_request.review_focus in argv
    assert "--requirement=REQ-CURRENT-1" in argv
    assert "--project-rule=Cite only inspected repository content." in argv
    assert not any(item.startswith("--ci-evidence=") for item in argv)
    ci_file_arguments = [
        item for item in argv if item.startswith("--ci-evidence-file=")
    ]
    assert ci_file_arguments == [
        "--ci-evidence-file=.review-agent/eval-input/"
        "existing-ci-evidence."
        + hashlib.sha256(
            _ci_bundle_bytes(*eval_input.review_request.existing_ci_evidence)
        ).hexdigest()
        + ".v1.json"
    ]
    assert len(ci_file_arguments[0]) < 256
    bundle_path = tmp_path.joinpath(
        *ci_file_arguments[0].split("=", 1)[1].split("/")
    )
    assert bundle_path.read_bytes() == _ci_bundle_bytes(
        *eval_input.review_request.existing_ci_evidence
    )
    assert command_module._load_ci_evidence_bundle(
        tmp_path.resolve(),
        ci_file_arguments[0].split("=", 1)[1],
    ) == tuple(
        _ci_cli_payload(item)
        for item in eval_input.review_request.existing_ci_evidence
    )

    parsed = command_module._build_parser().parse_args(argv[2:])
    assert parsed.ci_evidence == []
    assert parsed.ci_evidence_file == (
        ci_file_arguments[0].split("=", 1)[1]
    )
    assert parsed.head == "b" * 40
    assert not any(item == "--non-interactive" for item in argv)
    assert seen[0][1]["stdin_bytes"] == b""
    assert Path(seen[0][1]["workspace"]).resolve() == tmp_path.resolve()


def test_empty_ci_evidence_adds_no_cli_argument(tmp_path: Path) -> None:
    eval_input = _eval_input("a" * 40, "b" * 40)
    config = _config(
        eval_input,
        command=(str(Path(sys.executable).resolve()), "product-entry.py"),
    )
    argv = current_module._initial_argv(
        current_module._configuration(config),
        eval_input,
        tmp_path,
        tmp_path / "memory",
    )

    assert not any(item.startswith("--ci-evidence=") for item in argv)
    assert not any(item.startswith("--ci-evidence-file=") for item in argv)
    parsed = command_module._build_parser().parse_args(argv[2:])
    assert parsed.ci_evidence == []
    assert parsed.ci_evidence_file is None

    calls: list[object] = []

    def capture(argv: object, **kwargs: object) -> BoundedProcessResult:
        calls.append((argv, kwargs))
        return BoundedProcessResult(
            stdout=b"",
            returncode=1,
            failure_code=None,
            output_bytes=0,
        )

    CurrentAgentAdapter(process_runner=capture).run(
        eval_input,
        tmp_path,
        config,
        _SkipChannel(),
    )
    assert len(calls) == 1
    assert not (tmp_path / ".review-agent" / "eval-input").exists()


def test_large_ci_payload_with_nul_and_unicode_stays_out_of_argv(
    tmp_path: Path,
) -> None:
    evidence = tuple(
        _existing_ci(
            "ci-large-%02d" % index,
            ("\x00 Unicode 中文 🚦 line %02d\n" % index) + "x" * 31_000,
        )
        for index in range(64)
    )
    eval_input = _eval_input("a" * 40, "b" * 40, existing_ci=evidence)
    input_size = len(canonical_json_bytes(eval_input))
    assert 1_900_000 < input_size <= MAX_EVAL_INPUT_BYTES
    seen: list[list[str]] = []

    def capture(argv: object, **kwargs: object) -> BoundedProcessResult:
        del kwargs
        seen.append(list(argv))
        return BoundedProcessResult(
            stdout=b"",
            returncode=1,
            failure_code=None,
            output_bytes=0,
        )

    CurrentAgentAdapter(process_runner=capture).run(
        eval_input,
        tmp_path,
        _config(
            eval_input,
            command=(str(Path(sys.executable).resolve()), "product-entry.py"),
        ),
        _SkipChannel(),
    )

    assert len(seen) == 1
    assert sum(len(item) for item in seen[0]) < 16_384
    assert not any("\x00" in item or "ci-large-" in item for item in seen[0])
    file_argument = next(
        item for item in seen[0] if item.startswith("--ci-evidence-file=")
    )
    bundle_path = tmp_path.joinpath(*file_argument.split("=", 1)[1].split("/"))
    assert bundle_path.stat().st_size > 1_900_000
    encoded = command_module._load_ci_evidence_bundle(
        tmp_path.resolve(), file_argument.split("=", 1)[1]
    )
    assert tuple(json.loads(item) for item in encoded) == tuple(
        item.to_dict() for item in evidence
    )


def test_ci_bundle_publication_is_create_only_and_fails_before_launch(
    tmp_path: Path,
) -> None:
    evidence = _existing_ci("ci-1", "passed")
    eval_input = _eval_input("a" * 40, "b" * 40, existing_ci=(evidence,))
    path = _write_cli_bundle(tmp_path, b"preexisting bytes")
    calls: list[object] = []

    def forbidden(argv: object, **kwargs: object) -> BoundedProcessResult:
        calls.append((argv, kwargs))
        raise AssertionError("preexisting control input reached process launch")

    submission = CurrentAgentAdapter(process_runner=forbidden).run(
        eval_input,
        tmp_path,
        _config(eval_input),
        _SkipChannel(),
    )

    assert not calls
    assert path.read_bytes() == b"preexisting bytes"
    assert submission.status is SubmissionStatus.FAILED
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.ADAPTER_ERROR


def test_ci_bundle_rejects_symlinked_control_root(tmp_path: Path) -> None:
    evidence = _existing_ci("ci-1", "passed")
    eval_input = _eval_input("a" * 40, "b" * 40, existing_ci=(evidence,))
    target = tmp_path / "outside"
    target.mkdir()
    try:
        (tmp_path / ".review-agent").symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip("symlink creation is not permitted: %s" % error)
    calls: list[object] = []

    def forbidden(argv: object, **kwargs: object) -> BoundedProcessResult:
        calls.append((argv, kwargs))
        raise AssertionError("symlinked control root reached process launch")

    submission = CurrentAgentAdapter(process_runner=forbidden).run(
        eval_input,
        tmp_path,
        _config(eval_input),
        _SkipChannel(),
    )

    assert not calls
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.ADAPTER_ERROR


def test_ci_bundle_helpers_reject_windows_reparse_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = SimpleNamespace(
        st_mode=stat.S_IFDIR | 0o700,
        st_file_attributes=current_module._REPARSE_POINT,
        st_dev=1,
        st_ino=2,
        st_nlink=1,
        st_size=0,
    )
    monkeypatch.setattr(current_module.os, "lstat", lambda path: metadata)

    with pytest.raises(current_module._CurrentAdapterError, match="safe directory"):
        current_module._safe_control_directory(tmp_path, "test control root")
    assert not command_module._ci_evidence_directory(metadata)
    metadata.st_mode = stat.S_IFREG | 0o600
    assert not command_module._ci_evidence_regular_file(metadata)


def test_cli_ci_bundle_preserves_exact_values_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    evidence = _existing_ci("ci-unicode", "\x00中文🚦\n")
    path = _write_cli_bundle(tmp_path, _ci_bundle_bytes(evidence))
    supplied = path.relative_to(tmp_path).as_posix()

    assert command_module._load_ci_evidence_bundle(
        tmp_path.resolve(), supplied
    ) == (_ci_cli_payload(evidence),)

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["entries"][0]["text"] = "substituted source"
    path.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="filename digest"):
        command_module._load_ci_evidence_bundle(tmp_path.resolve(), supplied)


def test_cli_ci_bundle_rejects_valid_source_substitution(tmp_path: Path) -> None:
    original = _existing_ci("ci-original", "passed")
    path = _write_cli_bundle(tmp_path, _ci_bundle_bytes(original))
    supplied = path.relative_to(tmp_path).as_posix()
    substituted = _existing_ci("ci-substituted", "also passed")
    path.write_bytes(_ci_bundle_bytes(substituted))

    with pytest.raises(ValueError, match="filename digest"):
        command_module._load_ci_evidence_bundle(tmp_path.resolve(), supplied)


@pytest.mark.parametrize(
    "source_id",
    [
        "",
        " ",
        " ci-edge",
        "ci-edge ",
        "ci\tid",
        "ci\nid",
        "ci\rid",
        "ci\x00id",
        "ci\x1fid",
        "ci\x7fid",
        "ci\u00a0id",
        "x" * 513,
    ],
)
def test_cli_ci_bundle_cannot_bypass_canonical_source_id_rules(
    tmp_path: Path,
    source_id: str,
) -> None:
    text = "passed"
    data = _raw_ci_bundle_bytes(
        [
            {
                "source_id": source_id,
                "text": text,
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        ]
    )
    path = _write_cli_bundle(tmp_path, data)

    with pytest.raises(ValueError, match="source_id"):
        command_module._load_ci_evidence_bundle(
            tmp_path.resolve(), path.relative_to(tmp_path).as_posix()
        )


def test_cli_ci_bundle_cannot_bypass_canonical_text_character_limit(
    tmp_path: Path,
) -> None:
    text = "界" * 32_769
    data = _raw_ci_bundle_bytes(
        [
            {
                "source_id": "ci-oversized-entry",
                "text": text,
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        ]
    )
    path = _write_cli_bundle(tmp_path, data)

    with pytest.raises(ValueError, match="text"):
        command_module._load_ci_evidence_bundle(
            tmp_path.resolve(), path.relative_to(tmp_path).as_posix()
        )


def test_cli_ci_bundle_accepts_exact_canonical_text_character_limit(
    tmp_path: Path,
) -> None:
    text = "界" * 32_768
    evidence = _existing_ci("ci-max-entry", text)
    path = _write_cli_bundle(tmp_path, _ci_bundle_bytes(evidence))

    assert command_module._load_ci_evidence_bundle(
        tmp_path.resolve(), path.relative_to(tmp_path).as_posix()
    ) == (_ci_cli_payload(evidence),)


def test_product_cli_rejects_eval_invalid_bundle_before_review_request(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = "x" * 32_769
    data = _raw_ci_bundle_bytes(
        [
            {
                "source_id": "ci-invalid-product-boundary",
                "text": text,
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        ]
    )
    path = _write_cli_bundle(tmp_path, data)
    args = command_module._build_parser().parse_args(
        [
            "review",
            "--repo=" + str(tmp_path),
            "--base=" + "a" * 40,
            "--head=" + "b" * 40,
            "--ci-evidence-file=" + path.relative_to(tmp_path).as_posix(),
        ]
    )

    assert command_module._run_review(args) == 2
    assert "CI evidence input error" in capsys.readouterr().err
    assert not (tmp_path / ".review-agent" / "runs").exists()


@pytest.mark.parametrize(
    "data",
    [
        b'{"entries":[],"schema_version":"review_agent_ci_evidence_bundle_v1",'
        b'"unknown":true}',
        b'{"entries":[],"entries":[],"schema_version":'
        b'"review_agent_ci_evidence_bundle_v1"}',
        canonical_json_bytes(
            {
                "schema_version": "wrong_version",
                "entries": [_existing_ci("ci-1", "passed").to_dict()],
            }
        ),
        canonical_json_bytes(
            {
                "schema_version": "review_agent_ci_evidence_bundle_v1",
                "entries": [
                    _existing_ci("ci-1", "passed").to_dict(),
                    _existing_ci("ci-1", "passed").to_dict(),
                ],
            }
        ),
        b'{"entries":[{"content_hash":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
        b'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA","source_id":"ci-1","text":""}],'
        b'"schema_version":"review_agent_ci_evidence_bundle_v1"}',
    ],
)
def test_cli_ci_bundle_rejects_invalid_schema_and_duplicates(
    tmp_path: Path,
    data: bytes,
) -> None:
    path = _write_cli_bundle(tmp_path, data)
    with pytest.raises(ValueError):
        command_module._load_ci_evidence_bundle(
            tmp_path.resolve(), path.relative_to(tmp_path).as_posix()
        )


def test_cli_ci_bundle_rejects_oversize_and_symlink_file(tmp_path: Path) -> None:
    path = _write_cli_bundle(tmp_path, b"x" * (2 * 1024 * 1024 + 1))
    supplied = path.relative_to(tmp_path).as_posix()
    with pytest.raises(ValueError, match="bounded regular file"):
        command_module._load_ci_evidence_bundle(tmp_path.resolve(), supplied)

    path.unlink()
    target = tmp_path / "outside-bundle.json"
    target.write_bytes(_ci_bundle_bytes(_existing_ci("ci-1", "passed")))
    try:
        path.symlink_to(target)
    except OSError as error:
        pytest.skip("symlink creation is not permitted: %s" % error)
    with pytest.raises(ValueError, match="bounded regular file"):
        command_module._load_ci_evidence_bundle(tmp_path.resolve(), supplied)


def test_cli_parser_binds_file_transport_and_makes_sources_mutually_exclusive() -> None:
    parser = command_module._build_parser()
    parsed = parser.parse_args(
        [
            "review",
            "--base=" + "a" * 40,
            "--head=" + "b" * 40,
            "--ci-evidence-file=.review-agent/eval-input/existing-ci-evidence."
            + "0" * 64
            + ".v1.json",
        ]
    )
    assert parsed.ci_evidence == []
    assert parsed.ci_evidence_file.endswith(".v1.json")

    direct = parser.parse_args(
        [
            "review",
            "--base=" + "a" * 40,
            "--head=" + "b" * 40,
            "--ci-evidence={}",
        ]
    )
    assert direct.ci_evidence == ["{}"]
    assert direct.ci_evidence_file is None

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "review",
                "--base=" + "a" * 40,
                "--head=" + "b" * 40,
                "--ci-evidence={}",
                "--ci-evidence-file=bundle.json",
            ]
        )


def test_invalid_completed_artifact_state_is_invalid_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_input = _eval_input("a" * 40, "b" * 40)

    def create_run(
        argv: object,
        **kwargs: object,
    ) -> BoundedProcessResult:
        del argv
        workspace = Path(kwargs["workspace"])
        (workspace / ".review-agent" / "runs" / "review-123456789abc").mkdir(
            parents=True
        )
        return BoundedProcessResult(
            stdout=b"ignored",
            returncode=0,
            failure_code=None,
            output_bytes=7,
        )

    def reject_session(*args: object, **kwargs: object) -> object:
        raise current_module._CurrentArtifactError("tampered Session")

    monkeypatch.setattr(current_module, "_load_session", reject_session)
    submission = CurrentAgentAdapter(process_runner=create_run).run(
        eval_input,
        tmp_path,
        _config(eval_input),
        _SkipChannel(),
    )

    assert submission.status is SubmissionStatus.INVALID_OUTPUT
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.SCHEMA_MISMATCH
    assert "tampered Session" not in submission.to_json()


def _brief(
    *,
    findings: list[BriefFinding],
    quality_gates: list[dict[str, Any]] | None = None,
    clarification_history: list[dict[str, Any]] | None = None,
) -> ReviewBrief:
    return ReviewBrief(
        review_id="review-123456789abc",
        base_revision="a" * 40,
        head_revision="b" * 40,
        change_intent={
            "goal": "Keep behavior deterministic",
            "acceptance_criteria": ["Return a stable boolean"],
            "scope": ["feature.py"],
            "constraints": [],
            "sources": {"goal": "inferred"},
            "provenance": [
                {
                    "claim_id": "claim_inferred",
                    "field": "goal",
                    "value": "Keep behavior deterministic",
                    "source": "inferred",
                    "origin": "llm_inference",
                    "confidence": "medium",
                    "source_refs": [],
                    "evidence_refs": [],
                    "claim_state": "active",
                    "conclusion_impact": "material",
                }
            ],
        },
        intent_assessment={
            "status": "sufficient",
            "uncertainties": [],
            "source_counts": {"inferred": 1},
            "clarification_history": clarification_history or [],
            "unresolved_questions": [],
            "unconfirmed_inferred_claims": [],
        },
        initial_and_final_risk_assessment={},
        quality_gates=quality_gates or [],
        change_map_and_repository_impact={},
        verified_findings=findings,
        rejected_hypotheses=[],
        uncertainties=[],
        reviewer_disagreements=[],
        review_contract_coverage=[],
        verification_evidence=[],
        human_review_checklist_and_reading_order=[],
        non_binding_recommendation="manual_review",
    )


def test_brief_mapping_preserves_inferred_provenance_missing_location_and_zero_findings() -> None:
    intent = current_module._intent_from_brief(_brief(findings=[]), ())
    assert intent.claims[0].source.value == "inferred"
    assert current_module._findings_from_brief(_brief(findings=[])) == ()

    finding = BriefFinding(
        claim="The fallback can return the wrong value",
        severity="high",
        confidence="high",
        evidence_refs=["O-read"],
        path=None,
        line=None,
    )
    mapped = current_module._findings_from_brief(_brief(findings=[finding]))[0]
    assert mapped.path is None
    assert mapped.side is None
    assert mapped.from_line is None
    assert mapped.to_line is None

    blocker = BriefFinding(
        claim="This change can corrupt persisted authorization state",
        severity="blocker",
        confidence="high",
        evidence_refs=[],
        path="feature.py",
        line=2,
    )
    mapped_blocker = current_module._findings_from_brief(
        _brief(findings=[blocker])
    )[0]
    assert mapped_blocker.severity.value == "critical"


def _ordered_clarification_fixture() -> tuple[
    tuple[SubmissionClarificationExchange, ...],
    list[dict[str, Any]],
]:
    transcript = (
        SubmissionClarificationExchange(
            turn_index=1,
            question_id="question-goal",
            dimension=IntentDimension.GOAL,
            question="Confirm the requested goal?",
            material_claim="Preserve deterministic behavior",
            matched_answer_id="answer-goal",
            action=ClarificationAction.SKIP,
            response=None,
            resolved_values=(),
        ),
        SubmissionClarificationExchange(
            turn_index=2,
            question_id="question-scope",
            dimension=IntentDimension.SCOPE,
            question="Confirm the requested scope?",
            material_claim="Only change feature.py",
            matched_answer_id="answer-scope",
            action=ClarificationAction.SKIP,
            response=None,
            resolved_values=(),
        ),
    )
    history = [
        {
            "question_id": "question-goal",
            "field": "goal",
            "question": "Confirm the requested goal?",
            "status": "skipped",
            "user_response": None,
            "resolved_values": [],
        },
        {
            "question_id": "question-scope",
            "field": "scope",
            "question": "Confirm the requested scope?",
            "status": "skipped",
            "user_response": None,
            "resolved_values": [],
        },
    ]
    return transcript, history


def test_product_clarification_history_matches_channel_in_strict_order() -> None:
    transcript, history = _ordered_clarification_fixture()

    intent = current_module._intent_from_brief(
        _brief(findings=[], clarification_history=history),
        transcript,
    )

    assert intent.clarification_questions == transcript


@pytest.mark.parametrize("tamper", ["reordered", "duplicate"])
def test_product_clarification_history_rejects_reorder_and_duplicate_ids(
    tamper: str,
) -> None:
    transcript, history = _ordered_clarification_fixture()
    if tamper == "reordered":
        history = list(reversed(history))
    else:
        history[1] = dict(history[0])

    with pytest.raises(
        current_module._CurrentArtifactError,
        match="clarification history does not match",
    ):
        current_module._intent_from_brief(
            _brief(findings=[], clarification_history=history),
            transcript,
        )


def test_evidence_conversion_uses_only_referenced_replayable_observations(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "review-123456789abc"
    raw_root = run_dir / "observations"
    raw_root.mkdir(parents=True)
    base = "a" * 40
    head = "b" * 40

    def observation(
        observation_id: str,
        source: str,
        revision: str,
        raw: str,
        *,
        path: str | None,
        line_start: int | None,
        line_end: int | None,
    ) -> Observation:
        raw_ref = "observations/%s.txt" % observation_id
        (run_dir / raw_ref).write_bytes(raw.encode("utf-8"))
        return Observation(
            observation_id=observation_id,
            source=source,
            revision=revision,
            path=path,
            line_start=line_start,
            line_end=line_end,
            content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            raw_artifact_ref=raw_ref,
            context_view="summary",
        )

    observations = {
        item.observation_id: item
        for item in (
            observation(
                "O-read",
                "git.read_range",
                "head@" + head,
                "return bool(value)\n",
                path="feature.py",
                line_start=2,
                line_end=2,
            ),
            observation(
                "O-search",
                "git.search_code",
                "head@" + head,
                "feature.py:2:return bool(value)\n",
                path=None,
                line_start=None,
                line_end=None,
            ),
            observation(
                "O-unused",
                "git.read_range",
                "head@" + head,
                "unused\n",
                path="feature.py",
                line_start=1,
                line_end=1,
            ),
        )
    }
    brief_finding = BriefFinding(
        claim="Check the return path",
        severity="medium",
        confidence="high",
        evidence_refs=["O-read", "O-search", "O-missing"],
        path="feature.py",
        line=2,
    )
    brief = _brief(findings=[brief_finding])
    findings = current_module._findings_from_brief(brief)
    eval_input = _eval_input(base, head)

    evidence = current_module._evidence_from_observations(
        run_dir=run_dir,
        brief=brief,
        findings=findings,
        observations=observations,
        eval_input=eval_input,
    )

    assert [item.evidence_id for item in evidence] == ["O-read"]
    assert evidence[0].revision == head
    assert evidence[0].path == "feature.py"
    assert findings[0].evidence_refs == ("O-read", "O-search", "O-missing")


@pytest.mark.parametrize("output_truncated", [False, True])
def test_command_output_requires_runner_attestation_before_becoming_evidence(
    tmp_path: Path,
    output_truncated: bool,
) -> None:
    run_dir = tmp_path / "review-123456789abc"
    raw_path = run_dir / "observations" / "O-gate.txt"
    raw_path.parent.mkdir(parents=True)
    raw = "1 passed\n"
    raw_path.write_bytes(raw.encode("utf-8"))
    base = "a" * 40
    head = "b" * 40
    observation = Observation(
        observation_id="O-gate",
        source="quality_gate.pytest",
        revision="head@" + head,
        path=None,
        line_start=None,
        line_end=None,
        # Self-reported command artifacts are ignored before their bytes/hash
        # can poison the canonical Submission; only Runner attestation may use it.
        content_hash="0" * 64,
        raw_artifact_ref="observations/O-gate.txt",
        context_view="summary",
    )
    brief = _brief(
        findings=[
            BriefFinding(
                claim="The test run verifies the changed behavior",
                severity="medium",
                confidence="high",
                evidence_refs=["O-gate"],
                path=None,
                line=None,
            )
        ],
        quality_gates=[
            {
                "observation_ref": "O-gate",
                "command": ["pytest", "-q"],
                "exit_code": 0,
                "output_truncated": output_truncated,
            }
        ],
    )
    findings = current_module._findings_from_brief(brief)

    evidence = current_module._evidence_from_observations(
        run_dir=run_dir,
        brief=brief,
        findings=findings,
        observations={"O-gate": observation},
        eval_input=_eval_input(base, head),
    )

    assert evidence == ()
    assert findings[0].evidence_refs == ("O-gate",)


def test_correct_answer_must_round_trip_through_product_semicolon_protocol() -> None:
    question = type(
        "Question",
        (),
        {"claim_ids": ["claim-1"]},
    )()
    valid = SubmissionClarificationExchange(
        turn_index=1,
        question_id="question-1",
        dimension=IntentDimension.GOAL,
        question="What is the goal?",
        material_claim="Goal",
        matched_answer_id="answer-1",
        action=ClarificationAction.CORRECT,
        response="first; second",
        resolved_values=("first", "second"),
    )
    assert current_module._answer_input(valid, question) == b"correct\nfirst; second\n"

    invalid = SubmissionClarificationExchange(
        turn_index=1,
        question_id="question-1",
        dimension=IntentDimension.GOAL,
        question="What is the goal?",
        material_claim="Goal",
        matched_answer_id="answer-1",
        action=ClarificationAction.CORRECT,
        response="first and second",
        resolved_values=("first", "second"),
    )
    with pytest.raises(current_module._CurrentAdapterError):
        current_module._answer_input(invalid, question)


def test_only_explicit_product_adapter_boundaries_may_import_product_modules() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "review_agent_eval"
    violations = []
    allowed_imports = {
        "adapters/current_agent.py": [],
        "adapters/model_adapter.py": [],
    }
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
            for module in modules:
                if module == "review_agent" or module.startswith("review_agent."):
                    relative = path.relative_to(package_root).as_posix()
                    if relative in allowed_imports:
                        allowed_imports[relative].append(module)
                    else:
                        violations.append((relative, module))
    assert all(allowed_imports.values())
    assert not violations


def test_generic_adapter_import_does_not_load_product_runtime() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    code = (
        "import sys; import review_agent_eval.adapters; "
        "assert not any(name == 'review_agent' or "
        "name.startswith('review_agent.') for name in sys.modules)"
    )
    result = subprocess.run(
        [str(Path(sys.executable).resolve()), "-c", code],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")

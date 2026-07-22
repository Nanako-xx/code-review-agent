"""Command line composition root for the independent Eval harness.

The product command remains ``review-agent``.  This module exposes a separate
``review-agent-eval`` executable with four deliberately narrow stages:

``prepare``
    verify a Suite, prepare repository caches and create an immutable Run plan;
``run-agent``
    execute only the frozen Agent plan and publish terminal Submissions;
``evaluate``
    read those Submissions and publish a versioned evaluator namespace;
``inspect``
    print a redacted, source-bound inspection without invoking an Agent or a
    model.

The parser is intentionally boring.  All domain behavior lives in the
existing CaseBank, Runner, Evaluators, Metrics, ReportBuilder and ArtifactStore
components (or in :mod:`orchestrator`), so changing a CLI option cannot create
a second scoring implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


CLI_SCHEMA_VERSION = "review_agent_eval_cli_v1"

# Stable process-level categories.  A valid failed Agent Submission or a
# Judge-failed result is data, not a CLI infrastructure failure.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_PRECONDITION = 10
EXIT_CONFLICT = 11
EXIT_INTEGRITY = 12
EXIT_OPERATIONAL = 13


class CliUsageError(ValueError):
    pass


class CliPreconditionError(RuntimeError):
    pass


class CliConflictError(RuntimeError):
    pass


class CliIntegrityError(RuntimeError):
    pass


def _canonical_digest(value: Any) -> str:
    from .models import canonical_json_bytes

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_json(path: Path, *, context: str) -> Any:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CliPreconditionError("cannot read %s" % context) from exc
    try:
        # Domain hydration performs duplicate-key and numeric validation.  For
        # generic JSON config files still reject duplicate keys here.
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result

        return json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise CliIntegrityError("%s is not valid JSON" % context) from exc


def _path(value: str, *, name: str, required_name: Optional[str] = None) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CliUsageError("%s must be a non-empty path" % name)
    result = Path(value).expanduser()
    if required_name is not None and result.name != required_name:
        raise CliUsageError("%s must name an explicit %s directory" % (name, required_name))
    return result


def _positive_int(value: str, *, name: str, maximum: int = 100_000) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CliUsageError("%s must be an integer" % name) from exc
    if result < 1 or result > maximum:
        raise CliUsageError("%s is outside its allowed range" % name)
    return result


def _positive_float(
    value: str,
    *,
    name: str,
    maximum: float = 7 * 24 * 60 * 60,
) -> float:
    import math

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CliUsageError("%s must be a number" % name) from exc
    if not math.isfinite(result) or result <= 0 or result > maximum:
        raise CliUsageError("%s is outside its allowed range" % name)
    return result


def _json_output(command: str, status: str, **payload: Any) -> None:
    from .models import canonical_json

    envelope = {
        "schema_version": CLI_SCHEMA_VERSION,
        "command": command,
        "status": status,
        **payload,
    }
    sys.stdout.write(canonical_json(envelope) + "\n")


def _text_output(command: str, status: str, **payload: Any) -> None:
    # Text mode intentionally remains a compact human-readable summary.  JSON
    # is the stable machine interface and is used by all tests/automation.
    if status == "ok":
        headline = payload.get("message", command + " completed")
    else:
        headline = payload.get("message", command + " failed")
    sys.stdout.write(str(headline) + "\n")
    if "run_id" in payload:
        sys.stdout.write("run_id: %s\n" % payload["run_id"])


def _emit(args: argparse.Namespace, command: str, status: str, **payload: Any) -> None:
    if getattr(args, "json", False) or getattr(args, "format", None) == "json":
        _json_output(command, status, **payload)
    else:
        _text_output(command, status, **payload)


def _common_paths(parser: argparse.ArgumentParser, *, suite: bool = False) -> None:
    if suite:
        parser.add_argument("--suite-root", required=True)
    parser.add_argument("--runs-root", default=".eval-runs")
    parser.add_argument("--data-root", default=".eval-data")
    parser.add_argument("--workspace-root", default=".eval-workspaces")
    parser.add_argument("--git-executable", default="git")
    parser.add_argument("--json", action="store_true", help="emit stable JSON")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-agent-eval",
        description="Separated, evidence-driven code-review evaluation harness",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser(
        "prepare",
        help="verify a Suite, prepare repository caches and create a Run plan",
    )
    _common_paths(prepare, suite=True)
    prepare.add_argument("--manifest", default="suite_manifest.json")
    prepare.add_argument("--expected-manifest-digest")
    prepare.add_argument("--run-config", "--config", dest="run_config")
    prepare.add_argument("--agent-config")
    prepare.add_argument("--evaluator-config")
    prepare.add_argument("--matcher-config")
    prepare.add_argument("--run-instance-key", default="local")
    prepare.add_argument("--trial-count", default="1")
    prepare.add_argument(
        "--capability-policy", choices=("strict", "filter"), default="strict"
    )
    prepare.add_argument("--agent-adapter", choices=("current", "subprocess"), default="current")
    prepare.add_argument("--agent-command", action="append", default=[])
    prepare.add_argument("--agent-argument", action="append", default=[])
    prepare.add_argument("--agent-provider", choices=("none", "fake", "openai-compatible"), default="none")
    prepare.add_argument("--agent-model", default="none")
    prepare.add_argument("--agent-base-url")
    prepare.add_argument("--agent-api-key-env", default="REVIEW_AGENT_API_KEY")
    prepare.add_argument("--agent-id", default="current-agent")
    prepare.add_argument("--agent-name", default="Current code review agent")
    prepare.add_argument("--agent-version", default="working-tree")
    prepare.add_argument("--agent-commit", default="unknown")
    prepare.add_argument("--memory-mode", choices=("off", "read", "read-write"), default="off")
    prepare.add_argument(
        "--agent-timeout-seconds",
        type=lambda value: _positive_float(value, name="agent_timeout_seconds"),
        default=900.0,
    )
    prepare.add_argument(
        "--evaluator-timeout-seconds",
        type=lambda value: _positive_float(value, name="evaluator_timeout_seconds"),
        default=300.0,
    )
    prepare.add_argument("--max-agent-output-bytes", default=str(2 * 1024 * 1024))
    prepare.add_argument("--max-trace-bytes", default=str(4 * 1024 * 1024))
    prepare.add_argument("--max-execution-artifact-file-bytes", default=str(16 * 1024 * 1024))
    prepare.add_argument("--max-execution-artifact-total-bytes", default=str(128 * 1024 * 1024))
    prepare.add_argument("--max-parallel-trials", default="1")
    prepare.add_argument("--dry-run", action="store_true")
    prepare.add_argument("--resume", action="store_true")
    prepare.add_argument(
        "--overwrite",
        action="store_true",
        help="rejected for immutable Run artifacts; use a new run-instance-key",
    )

    run_agent = sub.add_parser(
        "run-agent",
        help="run the frozen Agent plan and write only Submission/trace artifacts",
    )
    _common_paths(run_agent, suite=True)
    run_agent.add_argument("--manifest", default="suite_manifest.json")
    run_agent.add_argument("--expected-manifest-digest")
    run_agent.add_argument("run_id_positional", nargs="?")
    run_agent.add_argument("--run-id")
    run_agent.add_argument("--resume", dest="resume", action="store_true", default=True)
    run_agent.add_argument("--no-resume", dest="resume", action="store_false")
    run_agent.add_argument(
        "--max-workers",
        type=lambda value: _positive_int(value, name="max_workers", maximum=1024),
    )
    run_agent.add_argument("--dry-run", action="store_true")
    run_agent.add_argument("--overwrite", action="store_true")

    evaluate = sub.add_parser(
        "evaluate",
        help="evaluate immutable Submissions and write a versioned Judge/score namespace",
    )
    _common_paths(evaluate, suite=True)
    evaluate.add_argument("--manifest", default="suite_manifest.json")
    evaluate.add_argument("--expected-manifest-digest")
    evaluate.add_argument("run_id_positional", nargs="?")
    evaluate.add_argument("--run-id")
    evaluate.add_argument("--evaluator-execution-config", "--execution-config", dest="execution_config")
    evaluate.add_argument("--revision", default="local")
    evaluate.add_argument("--judge-provider", choices=("fake", "openai-compatible", "none"), default="none")
    evaluate.add_argument("--judge-model", default="eval-judge")
    evaluate.add_argument("--judge-base-url")
    evaluate.add_argument("--judge-api-key-env", default="REVIEW_AGENT_EVAL_API_KEY")
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.add_argument("--resume", action="store_true", default=True)
    evaluate.add_argument("--no-resume", dest="resume", action="store_false")
    evaluate.add_argument("--overwrite", action="store_true")

    inspect = sub.add_parser(
        "inspect",
        help="show one redacted Case/Trial inspection without Agent or Judge execution",
    )
    _common_paths(inspect, suite=True)
    inspect.add_argument("run_id_positional", nargs="?")
    inspect.add_argument("--run-id")
    inspect.add_argument("--task-id", required=True)
    inspect.add_argument("--trial-id", required=True)
    inspect.add_argument("--evaluation-id", required=True)
    inspect.add_argument("--format", choices=("json", "markdown"), default="json")
    inspect.add_argument("--dry-run", action="store_true")

    return parser


def _run_id(args: argparse.Namespace) -> str:
    value = getattr(args, "run_id", None) or getattr(args, "run_id_positional", None)
    if not isinstance(value, str) or not value:
        raise CliUsageError("run_id is required")
    return value


def _roots(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    return (
        _path(args.suite_root, name="suite_root"),
        _path(args.runs_root, name="runs_root", required_name=".eval-runs"),
        _path(args.data_root, name="data_root", required_name=".eval-data"),
        _path(
            args.workspace_root,
            name="workspace_root",
            required_name=".eval-workspaces",
        ),
    )


def _load_case_bank(args: argparse.Namespace, suite_root: Path):
    from .datasets import CaseBank

    if not suite_root.exists():
        raise CliPreconditionError("Suite root is unavailable")
    bank = CaseBank.open(
        suite_root,
        args.manifest if hasattr(args, "manifest") else "suite_manifest.json",
        expected_manifest_digest=getattr(args, "expected_manifest_digest", None),
    )
    bank.verify()
    return bank


def _default_matcher():
    from .clarification import canonical_material_claim_matcher_snapshot

    return canonical_material_claim_matcher_snapshot()


def _default_evaluator_config(
    *,
    provider: str = "fake",
    model: str = "eval-judge",
    base_url: Optional[str] = None,
    api_key_env: str = "REVIEW_AGENT_EVAL_API_KEY",
):
    from .config import (
        EvaluatorRunConfig,
        JudgeKind,
        JUDGE_PROFILE_SCHEMA_VERSION,
        JudgeProfileSnapshot,
    )
    from .judge import (
        DEFAULT_JUDGE_RUBRICS,
        GLOBAL_JUDGE_SYSTEM_PROMPT,
        JUDGE_CONTEXT_BUILDER_VERSION,
        JUDGE_PARSER_VERSION,
        JUDGE_SYSTEM_PROMPT_VERSION,
    )
    from .judge import JudgeTask

    profiles = []
    for kind in JudgeKind:
        rubric = DEFAULT_JUDGE_RUBRICS.for_task(JudgeTask(kind.value))
        system_prompt = GLOBAL_JUDGE_SYSTEM_PROMPT + "\nTask rubric:\n" + rubric.instruction
        adapter_identity = {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "stage_label": "eval-judge",
        }
        profiles.append(
            JudgeProfileSnapshot(
                schema_version=JUDGE_PROFILE_SCHEMA_VERSION,
                kind=kind,
                judge_id=kind.value + "-judge",
                judge_version="cli-v1",
                adapter_id="unified-model-adapter",
                adapter_version="v1",
                adapter_config_digest=_canonical_digest(adapter_identity),
                provider=provider,
                model=model,
                model_artifact_digest=None,
                parameters={"temperature": 0, "reasoning_effort": "medium"},
                system_prompt_version=JUDGE_SYSTEM_PROMPT_VERSION,
                system_prompt_digest=_canonical_digest(system_prompt),
                rubric_id=rubric.rubric_id,
                rubric_version=rubric.rubric_version,
                rubric_digest=rubric.rubric_digest,
                response_schema_version=rubric.response_schema,
                response_schema_digest=_canonical_digest(rubric.response_schema),
                context_builder_version=JUDGE_CONTEXT_BUILDER_VERSION,
                parser_version=JUDGE_PARSER_VERSION,
            )
        )
    return EvaluatorRunConfig(
        evaluator_id="core-code-review-evaluator",
        evaluator_version="cli-v1",
        grader_version="core-grader-v1",
        judge_profiles=tuple(profiles),
    )


def _default_agent_snapshot(args: argparse.Namespace):
    from .adapters.current_agent import CURRENT_AGENT_ADAPTER_KIND
    from .adapters.subprocess_agent import (
        SUBPROCESS_JSON_ADAPTER_KIND,
        subprocess_adapter_capabilities,
    )
    from .config import AgentConfigSnapshot
    from .models import ReviewTargetKind

    command = list(args.agent_command)
    if not command:
        command = [sys.executable, "-m", "review_agent"]
    if not Path(command[0]).is_absolute():
        command[0] = str(Path(command[0]).resolve())
    if args.agent_adapter == "current":
        review_arguments = list(args.agent_argument)
        if args.agent_provider:
            review_arguments.append("--reviewer-provider=" + args.agent_provider)
        if args.agent_model and args.agent_model != "none":
            review_arguments.append("--reviewer-model=" + args.agent_model)
        if args.agent_base_url:
            review_arguments.append("--reviewer-base-url=" + args.agent_base_url)
        adapter = {
            "kind": CURRENT_AGENT_ADAPTER_KIND,
            "command": command,
            "review_arguments": review_arguments,
            "environment_allowlist": [args.agent_api_key_env]
            if args.agent_provider == "openai-compatible"
            else [],
            "memory_mode": args.memory_mode,
        }
    else:
        adapter = {
            "kind": SUBPROCESS_JSON_ADAPTER_KIND,
            "command": command,
            "environment_allowlist": [],
            "capabilities": subprocess_adapter_capabilities(
                target_kinds=(ReviewTargetKind.REPOSITORY,),
            ).to_dict(),
        }
    parameters = {"adapter": adapter}
    return AgentConfigSnapshot(
        agent_id=args.agent_id,
        agent_name=args.agent_name,
        agent_version=args.agent_version,
        commit=args.agent_commit,
        model=args.agent_model,
        provider=args.agent_provider,
        parameters=parameters,
        prompt_config_digest=_canonical_digest(parameters),
    )


def _load_agent_snapshot(args: argparse.Namespace):
    from .config import AgentConfigSnapshot

    if args.agent_config:
        payload = _read_json(_path(args.agent_config, name="agent_config"), context="Agent config")
        return AgentConfigSnapshot.from_dict(payload)
    return _default_agent_snapshot(args)


def _load_evaluator_config(args: argparse.Namespace):
    from .config import EvaluatorRunConfig

    if args.evaluator_config:
        payload = _read_json(
            _path(args.evaluator_config, name="evaluator_config"),
            context="Evaluator config",
        )
        return EvaluatorRunConfig.from_dict(payload)
    return _default_evaluator_config()


def _load_matcher(args: argparse.Namespace):
    from .config import ClarificationMatcherSnapshot

    if args.matcher_config:
        payload = _read_json(
            _path(args.matcher_config, name="matcher_config"),
            context="Matcher config",
        )
        return ClarificationMatcherSnapshot.from_dict(payload)
    return _default_matcher()


def _budgets(args: argparse.Namespace):
    from .config import ResourceBudgets

    return ResourceBudgets(
        agent_timeout_seconds=args.agent_timeout_seconds,
        evaluator_timeout_seconds=args.evaluator_timeout_seconds,
        max_agent_output_bytes=_positive_int(
            args.max_agent_output_bytes, name="max_agent_output_bytes", maximum=2**31
        ),
        max_trace_bytes=_positive_int(args.max_trace_bytes, name="max_trace_bytes", maximum=2**31),
        max_execution_artifact_file_bytes=_positive_int(
            args.max_execution_artifact_file_bytes,
            name="max_execution_artifact_file_bytes",
            maximum=2**31,
        ),
        max_execution_artifact_total_bytes=_positive_int(
            args.max_execution_artifact_total_bytes,
            name="max_execution_artifact_total_bytes",
            maximum=2**31,
        ),
        max_parallel_trials=_positive_int(
            args.max_parallel_trials, name="max_parallel_trials", maximum=1024
        ),
    )


def _load_run_config_for_prepare(args: argparse.Namespace, bank: Any):
    from .adapters.agent_factory import adapter_capabilities_from_snapshot
    from .config import EvalRunConfig, SuiteRunConfig

    snapshot = bank.snapshot()
    suite = SuiteRunConfig.from_case_snapshot(snapshot)
    if args.run_config:
        payload = _read_json(_path(args.run_config, name="run_config"), context="Run config")
        config = EvalRunConfig.from_dict(payload)
        if config.suite != suite:
            raise CliIntegrityError("Run config does not match the verified Suite snapshot")
        return config, snapshot
    agent = _load_agent_snapshot(args)
    capabilities = adapter_capabilities_from_snapshot(agent)
    config = EvalRunConfig.create(
        run_instance_key=args.run_instance_key,
        agent=agent,
        clarification_matcher=_load_matcher(args),
        evaluator=_load_evaluator_config(args),
        suite=suite,
        adapter_capabilities=capabilities,
        trial_count=_positive_int(args.trial_count, name="trial_count", maximum=10_000),
        resource_budgets=_budgets(args),
    )
    return config, snapshot


def _artifact_store(path: Path, *, create: bool):
    from .artifacts import ArtifactStore

    if not create and not path.exists():
        raise CliPreconditionError("Eval artifact root is unavailable")
    try:
        return ArtifactStore(path, create_root=create)
    except OSError as exc:
        raise CliPreconditionError("Eval artifact root is unavailable") from exc


def _repository_preparer(
    args: argparse.Namespace,
    roots: tuple[Path, Path, Path, Path],
    *,
    cache_only: bool = False,
):
    import shutil

    from .repository import RepositoryMode, RepositoryPreparer

    suite_root, _runs_root, data_root, workspace_root = roots
    git_executable = args.git_executable
    if not Path(git_executable).is_absolute():
        resolved = shutil.which(git_executable)
        if resolved is None:
            raise CliPreconditionError("Git executable is unavailable")
        git_executable = resolved
    return RepositoryPreparer(
        suite_root=suite_root,
        data_root=data_root,
        workspace_root=workspace_root,
        git_executable=git_executable,
        allow_remote=False,
        repository_mode=(RepositoryMode.CACHE_ONLY if cache_only else RepositoryMode.ACQUIRE),
    )


def _agent_adapter(config: Any):
    from .adapters.agent_factory import (
        build_agent_adapter,
        build_agent_adapter_factory,
    )

    try:
        return build_agent_adapter_factory(config.agent), build_agent_adapter(config.agent)
    except Exception as exc:
        raise CliIntegrityError("Run config contains an invalid Agent adapter binding") from exc


def _judge_runtime_identity(args: argparse.Namespace) -> dict[str, Any]:
    provider = args.judge_provider
    model = args.judge_model
    base_url = args.judge_base_url
    api_key_env = args.judge_api_key_env
    if provider == "none":
        model = "none"
        base_url = None
        api_key_env = "none"
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "stage_label": "eval-judge",
    }


def _execution_config(args: argparse.Namespace, run_config: Any):
    from .config import EvaluatorExecutionConfig

    if args.execution_config:
        payload = _read_json(
            _path(args.execution_config, name="execution_config"),
            context="Evaluator execution config",
        )
        return EvaluatorExecutionConfig.from_dict(payload)
    evaluator = run_config.evaluator
    # Re-evaluation is explicitly allowed to change Judge identity.  The
    # command-line ``none`` provider is itself an identity (no semantic model
    # execution), not an instruction to silently inherit the provider that was
    # present in the immutable Run config.  Otherwise a no-Judge evaluation
    # could be incorrectly resumed after switching to a real/fake Judge with
    # the same model name.
    from dataclasses import replace

    adapter_identity = _judge_runtime_identity(args)
    effective_provider = adapter_identity["provider"]
    effective_model = adapter_identity["model"]
    profiles = []
    for profile in evaluator.judge_profiles:
        profiles.append(
            replace(
                profile,
                provider=effective_provider,
                model=effective_model,
                adapter_config_digest=_canonical_digest(adapter_identity),
            )
        )
    from .config import EvaluatorRunConfig

    evaluator = EvaluatorRunConfig(
        evaluator_id=evaluator.evaluator_id,
        evaluator_version=evaluator.evaluator_version,
        grader_version=evaluator.grader_version,
        judge_profiles=tuple(profiles),
    )
    return EvaluatorExecutionConfig.from_resource_budgets(
        evaluator,
        run_config.resource_budgets,
    )


def _validate_judge_boundary(args: argparse.Namespace, execution: Any) -> None:
    """Validate Judge transport identity without constructing or calling it.

    A dry-run must be able to check the shape of an OpenAI-compatible endpoint
    without reading a credential or creating a provider client.  The actual
    factory remains in ``_judge_for`` and is only reached by a real evaluate.
    """

    identity = _judge_runtime_identity(args)
    identity_digest = _canonical_digest(identity)
    profiles = execution.evaluator.judge_profiles
    if any(
        profile.provider != identity["provider"]
        or profile.model != identity["model"]
        or profile.adapter_config_digest != identity_digest
        for profile in profiles
    ):
        raise CliIntegrityError(
            "Judge runtime identity differs from EvaluatorExecutionConfig"
        )
    if args.judge_provider == "none":
        return
    from .adapters.model_adapter import ModelAdapterConfig

    try:
        ModelAdapterConfig(
            provider_name=args.judge_provider,
            model=args.judge_model,
            base_url=args.judge_base_url,
            api_key_env=args.judge_api_key_env,
            stage_label="eval-judge",
            timeout_seconds=execution.judge_budgets.attempt_timeout_seconds,
            max_response_bytes=execution.judge_budgets.max_model_response_bytes,
        )
    except (TypeError, ValueError) as exc:
        raise CliUsageError("Judge provider configuration is invalid") from exc


def _validate_revision(value: Any) -> str:
    from .config import validate_path_segment

    try:
        return validate_path_segment(value, "evaluation revision")
    except (TypeError, ValueError) as exc:
        raise CliUsageError("evaluation revision is invalid") from exc


def _trial_plans(store: Any, run_id: str) -> tuple[Any, ...]:
    """Return the complete immutable Run plan used by run-level metrics.

    Case selection belongs to the versioned Suite/Run created by ``prepare``.
    ``evaluate`` deliberately has no partial task filter because a subset
    cannot reuse the complete Run identity or its metric denominators.
    """

    return tuple(store.load_run_manifest(run_id).trials)


def _validate_evaluation_inputs(
    store: Any,
    run_id: str,
    run_config: Any,
    bank: Any,
) -> tuple[int, int]:
    """Validate selected terminal Submissions without invoking a Judge."""

    plans = _trial_plans(store, run_id)
    terminal = 0
    for plan in plans:
        state = store.load_trial_state(run_id, plan.task_id, plan.trial_id)
        if state.terminal_receipt is None or state.status.value not in {
            "completed",
            "failed",
            "blocked",
            "invalid_output",
        }:
            raise CliPreconditionError(
                "evaluate requires every selected Trial to have a terminal Submission"
            )
        submission = store.load_existing_submission(
            run_id, plan.task_id, plan.trial_id
        )
        case = bank.evaluator_case(plan.task_id)
        suite_case = run_config.suite.case(plan.task_id)
        if (
            case.digest() != suite_case.canonical_case_digest
            or case.eval_input().digest() != suite_case.eval_input_digest
            or case.source.suite != run_config.suite.suite_id
        ):
            raise CliIntegrityError("Suite Case does not match the immutable Run binding")
        if submission.task_id != plan.task_id or submission.trial_id != plan.trial_id:
            raise CliIntegrityError("Submission does not match the immutable Trial binding")
        terminal += 1
    return len(plans), terminal


def _reject_existing_evaluation_namespaces(
    store: Any,
    run_id: str,
    plans: Sequence[Any],
    execution: Any,
    revision: str,
) -> None:
    """Enforce create-only evaluation before any evaluator/model work."""

    from .artifacts import ArtifactStateError
    from .config import derive_evaluation_id

    evaluation_id = derive_evaluation_id(run_id, execution.digest(), revision)
    for plan in plans:
        try:
            namespaces = store.list_evaluations(
                run_id, plan.task_id, plan.trial_id
            )
        except ArtifactStateError as exc:
            raise CliConflictError(
                "uncommitted evaluation namespace exists; use --resume or a new revision"
            ) from exc
        if any(item.evaluation_id == evaluation_id for item in namespaces):
            raise CliConflictError(
                "evaluation namespace already exists; use --resume or a new revision"
            )
    if any(
        item.evaluation_id == evaluation_id
        for item in store.list_run_evaluations(run_id)
    ):
        raise CliConflictError(
            "Run evaluation namespace already exists; use --resume or a new revision"
        )


def _completed_evaluation_metadata(
    store: Any,
    run_id: str,
    execution: Any,
    revision: str,
) -> Optional[dict[str, Any]]:
    """Load a fully committed evaluation without semantic/repository replay.

    This fast path is deliberately all-or-nothing.  The Run summary/report and
    every planned Trial evaluation must already have their immutable commit
    markers and pass ArtifactStore hash/size/source-identity validation.  An
    incomplete evaluation returns ``None`` only when the Run-level commit is
    absent; once that commit exists, any missing or inconsistent Trial fails
    closed instead of falling back to model/repository execution.
    """

    from .artifacts import ArtifactStateError
    from .config import derive_evaluation_id

    evaluation_id = derive_evaluation_id(run_id, execution.digest(), revision)
    try:
        run_bundle = store.load_run_evaluation(run_id, evaluation_id)
    except ArtifactStateError:
        return None
    if (
        run_bundle.namespace.evaluator_execution_digest != execution.digest()
        or run_bundle.namespace.evaluation_revision != revision
    ):
        raise CliIntegrityError("Run evaluation identity differs from requested sources")

    trial_values = []
    for plan in _trial_plans(store, run_id):
        try:
            trial_bundle = store.load_evaluation_bundle(
                run_id,
                plan.task_id,
                plan.trial_id,
                evaluation_id,
            )
        except ArtifactStateError as exc:
            raise CliIntegrityError(
                "committed Run evaluation is missing a Trial evaluation"
            ) from exc
        submission = store.load_existing_submission(
            run_id, plan.task_id, plan.trial_id
        )
        if submission.digest() != trial_bundle.submission_digest:
            raise CliIntegrityError(
                "Trial evaluation is bound to a different Submission"
            )
        intent = trial_bundle.intent_matches
        review = trial_bundle.review_matches
        score = trial_bundle.score
        if not isinstance(intent, Mapping) or not isinstance(score, Mapping):
            raise CliIntegrityError("committed Trial evaluation payload is invalid")
        if review is not None and not isinstance(review, Mapping):
            raise CliIntegrityError("committed Review evaluation payload is invalid")
        if trial_bundle.report is None:
            raise CliIntegrityError("committed Trial evaluation has no report")
        trial_values.append(
            {
                "run_id": run_id,
                "task_id": plan.task_id,
                "trial_id": plan.trial_id,
                "trial_index": plan.trial_index,
                "evaluation_id": evaluation_id,
                "evaluation_revision": revision,
                "submission_status": submission.status.value,
                "intent_status": intent.get("status"),
                "review_status": None if review is None else review.get("status"),
                "score_digest": _canonical_digest(score),
                "report_digest": hashlib.sha256(
                    trial_bundle.report.encode("utf-8")
                ).hexdigest(),
            }
        )
    trial_values.sort(key=lambda item: (item["task_id"], item["trial_index"]))
    return {
        "run_id": run_id,
        "evaluation_id": evaluation_id,
        "evaluation_revision": revision,
        "trial_count": len(trial_values),
        "evaluated_trials": len(trial_values),
        "summary_id": run_bundle.namespace.summary_id,
        "report_digest": hashlib.sha256(
            run_bundle.report.encode("utf-8")
        ).hexdigest(),
        "trials": trial_values,
    }


def _judge_for(args: argparse.Namespace, execution: Any):
    _validate_judge_boundary(args, execution)
    if args.judge_provider == "none":
        return None
    from .adapters.model_adapter import (
        AdapterConfigError,
        ModelAdapterConfig,
        build_judge_model_adapter_factory,
    )
    from .judge import SemanticJudge

    try:
        factory = build_judge_model_adapter_factory(
            ModelAdapterConfig(
                provider_name=args.judge_provider,
                model=args.judge_model,
                base_url=args.judge_base_url,
                api_key_env=args.judge_api_key_env,
                stage_label="eval-judge",
            ),
            budgets=execution.judge_budgets,
        )
    except AdapterConfigError as exc:
        raise CliPreconditionError("Judge provider is unavailable") from exc
    if factory is None:
        raise CliPreconditionError("Judge provider did not produce a factory")
    return SemanticJudge(adapter_factory=factory, evaluator_execution=execution)


def _handle_prepare(args: argparse.Namespace) -> int:
    from .repository import repository_from_eval_input

    roots = _roots(args)
    suite_root, runs_root, _data_root, _workspace_root = roots
    if args.overwrite:
        raise CliConflictError(
            "immutable Run artifacts cannot be overwritten; choose a new --run-instance-key"
        )
    bank = _load_case_bank(args, suite_root)
    config, snapshot = _load_run_config_for_prepare(args, bank)
    if args.dry_run:
        _agent_adapter(config)
        _emit(
            args,
            "prepare",
            "ok",
            message="prepare dry-run statically validated Suite, Run config and Agent binding",
            dry_run=True,
            run_id=config.run_id,
            suite_id=config.suite.suite_id,
            case_count=len(config.suite.cases),
            trial_count=config.trial_count,
            repository_cache="not_modified",
            repository_preparation="deferred",
            capability_preflight="deferred",
        )
        return EXIT_OK

    store = _artifact_store(runs_root, create=True)
    if args.resume:
        from .artifacts import ArtifactStateError

        try:
            existing = store.load_run_config(config.run_id)
        except (ArtifactStateError, FileNotFoundError):
            existing = None
        if existing is not None:
            if existing != config:
                raise CliConflictError("existing Run ID is bound to different config bytes")
            # ``prepare`` is the only phase allowed to acquire repository
            # data.  Resuming an existing immutable plan therefore verifies
            # (and, if necessary, repairs) its cache here instead of deferring
            # a missing-cache failure to run-agent/evaluate.
            with _repository_preparer(args, roots, cache_only=False) as preparer:
                for case_entry in snapshot.cases:
                    preparer.prepare(
                        repository_from_eval_input(case_entry.input)
                    )
            _emit(args, "prepare", "ok", message="existing immutable Run reused", run_id=config.run_id, resumed=True)
            return EXIT_OK
    adapter_factory, adapter = _agent_adapter(config)
    from .runner import CapabilityPolicy, EvalRunner

    policy = CapabilityPolicy(args.capability_policy)
    with _repository_preparer(args, roots, cache_only=False) as preparer:
        # Preparation is the only stage allowed to acquire repository data.
        for case_entry in snapshot.cases:
            preparer.prepare(repository_from_eval_input(case_entry.input))
    runner = EvalRunner(
        store,
        None,
        adapter,
        case_provider=bank,
        adapter_factory=adapter_factory,
        capability_policy=policy,
        max_workers=config.resource_budgets.max_parallel_trials,
    )
    setup = runner.create_run(config, snapshot, policy=policy)
    _emit(
        args,
        "prepare",
        "ok",
        message="Suite and repository cache prepared; immutable Run plan created",
        run_id=setup.config.run_id,
        suite_id=setup.config.suite.suite_id,
        case_count=len(setup.config.suite.cases),
        trial_count=setup.config.trial_count,
        preflight=setup.preflight.to_dict(),
    )
    return EXIT_OK


def _handle_run_agent(args: argparse.Namespace) -> int:
    roots = _roots(args)
    suite_root, runs_root, _data_root, _workspace_root = roots
    run_id = _run_id(args)
    if args.overwrite:
        raise CliConflictError("run-agent never overwrites immutable Submissions")
    store = _artifact_store(runs_root, create=False)
    config = store.load_run_config(run_id)
    bank = _load_case_bank(args, suite_root)
    if args.max_workers is not None and (
        type(args.max_workers) is not int or not 1 <= args.max_workers <= 1024
    ):
        raise CliUsageError("max-workers is outside its allowed range")
    if args.dry_run:
        # Constructing the frozen adapter is a pure binding check; it does not
        # start an Agent process or read Judge configuration.
        _agent_adapter(config)
        state = store.load_run_state(run_id)
        pending = sum(1 for item in state.trials if item.status.value in {"pending", "incomplete", "running"})
        terminal = sum(1 for item in state.trials if item.terminal_receipt is not None)
        _emit(args, "run-agent", "ok", message="run-agent dry-run inspected immutable Trial plan", dry_run=True, run_id=run_id, pending=pending, terminal=terminal, agent_execution="not_invoked", repository_cache_check="deferred")
        return EXIT_OK
    adapter_factory, adapter = _agent_adapter(config)
    from .runner import EvalRunner

    with _repository_preparer(args, roots, cache_only=True) as preparer:
        runner = EvalRunner(
            store,
            preparer,
            adapter,
            case_provider=bank,
            adapter_factory=adapter_factory,
            max_workers=args.max_workers,
            retry_incomplete=args.resume,
        )
        result = runner.run_agent(run_id, resume=args.resume, max_workers=args.max_workers)
    trials = []
    for item in result.trials:
        trials.append({
            "task_id": item.task_id,
            "trial_id": item.trial_id,
            "trial_index": item.trial_index,
            "status": item.status.value,
            "submission_status": None if item.submission is None else item.submission.status.value,
            "skipped": item.skipped,
        })
    _emit(
        args,
        "run-agent",
        "ok",
        message="Agent Trials completed; no evaluation was run",
        run_id=run_id,
        run_status=result.status.value,
        trials=trials,
    )
    return EXIT_OK


def _handle_evaluate(args: argparse.Namespace) -> int:
    roots = _roots(args)
    suite_root, runs_root, _data_root, _workspace_root = roots
    run_id = _run_id(args)
    revision = _validate_revision(args.revision)
    if args.overwrite:
        raise CliConflictError(
            "evaluate never overwrites an immutable evaluation namespace"
        )
    store = _artifact_store(runs_root, create=False)
    run_config = store.load_run_config(run_id)
    execution = _execution_config(args, run_config)
    _validate_judge_boundary(args, execution)
    if args.resume and not args.dry_run:
        completed = _completed_evaluation_metadata(
            store,
            run_id,
            execution,
            revision,
        )
        if completed is not None:
            _emit(
                args,
                "evaluate",
                "ok",
                message="existing immutable evaluation reused",
                **completed,
            )
            return EXIT_OK
    bank = _load_case_bank(args, suite_root)
    planned_count, terminal = _validate_evaluation_inputs(
        store,
        run_id,
        run_config,
        bank,
    )
    if not args.resume:
        _reject_existing_evaluation_namespaces(
            store,
            run_id,
            _trial_plans(store, run_id),
            execution,
            revision,
        )
    if args.dry_run:
        _emit(args, "evaluate", "ok", message="evaluate dry-run validated terminal Submissions and static Judge configuration", dry_run=True, run_id=run_id, evaluation_revision=revision, planned_trials=planned_count, terminal_trials=terminal, judge_execution="not_invoked", credential_check="deferred" if args.judge_provider == "openai-compatible" else "not_required", repository_replay_check="deferred", evaluator_execution_digest=execution.digest())
        return EXIT_OK
    from .orchestrator import EvaluationOrchestrator

    judge_factory = None
    if args.judge_provider != "none":
        # Keep provider construction lazy.  A fully committed resume can be
        # source-validated and reused without reading credentials or opening a
        # network-capable adapter.  Cache the result so all newly evaluated
        # Trials still share the same run-scoped Judge budget state.
        judge_holder: list[Any] = []

        def build_judge() -> Any:
            if not judge_holder:
                judge_holder.append(_judge_for(args, execution))
            return judge_holder[0]

        judge_factory = build_judge
    with _repository_preparer(args, roots, cache_only=True) as preparer:
        orchestrator = EvaluationOrchestrator(
            store,
            bank,
            repository_preparer=preparer,
            judge_factory=judge_factory,
        )
        bundle = orchestrator.evaluate_run(
            run_id,
            evaluator_execution=execution,
            evaluation_revision=revision,
            resume=args.resume,
        )
    _emit(args, "evaluate", "ok", message="Intent/Review evaluation and report committed", **bundle.to_dict())
    return EXIT_OK


def _handle_inspect(args: argparse.Namespace) -> int:
    roots = _roots(args)
    suite_root, runs_root, _data_root, _workspace_root = roots
    run_id = _run_id(args)
    if args.json and args.format != "json":
        raise CliUsageError("--json cannot be combined with --format markdown")
    store = _artifact_store(runs_root, create=False)
    # Read only through receipt-bound ArtifactStore APIs.  In particular this
    # command never concatenates an internal path, opens a repository, invokes
    # an Agent, or constructs a Judge provider.
    bundle = store.load_evaluation_bundle(
        run_id,
        args.task_id,
        args.trial_id,
        args.evaluation_id,
    )
    if args.dry_run:
        _emit(
            args,
            "inspect",
            "ok",
            message="inspect dry-run validated receipt-bound evaluation",
            dry_run=True,
            run_id=run_id,
            task_id=args.task_id,
            trial_id=args.trial_id,
            evaluation_id=args.evaluation_id,
        )
        return EXIT_OK
    submission = store.load_existing_submission(
        run_id, args.task_id, args.trial_id
    )
    trial_manifest = store.load_trial_manifest(
        run_id, args.task_id, args.trial_id
    )
    intent = bundle.intent_matches
    review = bundle.review_matches
    score = bundle.score
    judge_output = bundle.judge_output
    # JSON inspect exposes evaluator outcomes and stable source bindings, not
    # raw repository/Judge context blocks or TraceRef values.  The committed
    # Markdown report remains the rich human view and was produced by
    # ReportBuilder's redacted projection.
    inspection = {
        "source_bindings": {
            "run_id": run_id,
            "task_id": args.task_id,
            "trial_id": args.trial_id,
            "evaluation_id": args.evaluation_id,
            "evaluation_revision": bundle.evaluation_revision,
            "evaluator_execution_digest": bundle.evaluator_execution.digest(),
            "submission_digest": bundle.submission_digest,
            "canonical_case_digest": bundle.canonical_case_digest,
            "eval_input_digest": trial_manifest.eval_input_digest,
            "trial_manifest_digest": bundle.trial_manifest_digest,
        },
        "submission": {
            "status": submission.status.value,
            "failure_code": (
                None if submission.failure is None else submission.failure.code.value
            ),
            "intent_present": submission.intent is not None,
            "review_present": submission.review is not None,
            "finding_count": (
                0 if submission.review is None else len(submission.review.findings)
            ),
            "evidence_count": len(submission.evidence),
            "trace": {
                "present": submission.trace_ref is not None,
                "type": (
                    None
                    if submission.trace_ref is None
                    else submission.trace_ref.type.value
                ),
                "value": "redacted" if submission.trace_ref is not None else None,
            },
        },
        "intent_evaluation": _inspection_status(intent),
        "review_evaluation": _inspection_status(review),
        "score": _inspection_score(score),
        "judge": _inspection_judge(judge_output),
        "report": {
            "available": bundle.report is not None,
            "sha256": None if bundle.report is None else hashlib.sha256(bundle.report.encode("utf-8")).hexdigest(),
        },
    }
    if args.format == "markdown":
        sys.stdout.write(_render_inspection_markdown(inspection))
        return EXIT_OK
    _json_output(
        "inspect",
        "ok",
        run_id=run_id,
        task_id=args.task_id,
        trial_id=args.trial_id,
        evaluation_id=args.evaluation_id,
        inspection=inspection,
    )
    return EXIT_OK


def _inspection_status(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CliIntegrityError("evaluation result is not an object")
    allowed_statuses = {
        "graded",
        "pending_judge",
        "ungraded",
        "not_scorable",
        "not_available",
    }
    raw_status = value.get("status")
    status = raw_status if raw_status in allowed_statuses else "unknown"
    reason_codes = value.get("reason_codes", [])
    if not isinstance(reason_codes, (list, tuple)):
        raise CliIntegrityError("evaluation reason_codes is not a list")
    result: dict[str, Any] = {
        "status": status,
        "reason_count": len(reason_codes),
        "metric_count": _collection_size(value.get("metrics")),
        "assignment_count": _collection_size(value.get("assignments")),
        "finding_outcome_count": _collection_size(value.get("finding_outcomes")),
        "evidence_integrity_result_count": _collection_size(
            value.get("evidence_integrity_results")
        ),
    }
    coverage = value.get("coverage")
    if coverage is not None:
        result["coverage"] = _numeric_projection(
            coverage,
            {
                "judge_request_count",
                "judge_graded_count",
                "judge_failed_count",
                "judge_ungraded_count",
                "judge_pending_count",
                "semantic_unknown_count",
                "finding_count",
                "finding_resolved_count",
                "evidence_result_count",
            },
        )
    return result


def _inspection_score(value: Any) -> Any:
    if not isinstance(value, Mapping):
        raise CliIntegrityError("evaluation score is not an object")
    allowed_submission_statuses = {
        "completed",
        "failed",
        "blocked",
        "invalid_output",
    }
    status = value.get("submission_status")
    reason_codes = value.get("reason_codes", [])
    if not isinstance(reason_codes, (list, tuple)):
        raise CliIntegrityError("score reason_codes is not a list")
    return {
        "artifact_digest": _canonical_digest(value),
        "trial_index": value.get("trial_index")
        if type(value.get("trial_index")) is int
        else None,
        "submission_status": (
            status if status in allowed_submission_statuses else "unknown"
        ),
        "metric_count": _collection_size(value.get("metrics")),
        "reason_count": len(reason_codes),
        "usage": _numeric_projection(
            value.get("usage", {}),
            {
                "elapsed_seconds",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "tool_calls",
                "cost_amount",
            },
        ),
    }


def _inspection_judge(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CliIntegrityError("Judge output artifact is not an object")
    results = value.get("results", [])
    if not isinstance(results, list):
        raise CliIntegrityError("Judge output results are not a list")
    allowed_statuses = {"graded", "judge_failed", "ungraded"}
    allowed_sources = {"live", "cache", "not_run", "fake"}
    allowed_tasks = {
        "intent_equivalence",
        "finding_equivalence",
        "review_finding_equivalence",
        "novel_factuality",
        "evidence_support",
    }
    status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    failure_count = 0
    ungraded_count = 0
    for item in results:
        if not isinstance(item, Mapping):
            raise CliIntegrityError("Judge output contains an invalid result")
        request = item.get("request")
        status = item.get("status")
        status_key = status if status in allowed_statuses else "unknown"
        status_counts[status_key] = status_counts.get(status_key, 0) + 1
        source = item.get("source")
        source_key = source if source in allowed_sources else "unknown"
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        task = request.get("task") if isinstance(request, Mapping) else None
        task_key = task if task in allowed_tasks else "unknown"
        task_counts[task_key] = task_counts.get(task_key, 0) + 1
        if isinstance(item.get("failure"), Mapping):
            failure_count += 1
        if item.get("ungraded_reason") is not None:
            ungraded_count += 1
    return {
        "evaluator_execution_digest": value.get("evaluator_execution_digest"),
        "request_count": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "failure_count": failure_count,
        "ungraded_count": ungraded_count,
    }


def _collection_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, Mapping)):
        return len(value)
    raise CliIntegrityError("evaluation collection has an invalid shape")


def _numeric_projection(
    value: Any,
    allowed_keys: set[str],
) -> dict[str, int | float | bool | None]:
    if not isinstance(value, Mapping):
        raise CliIntegrityError("numeric evaluation projection is not an object")
    result: dict[str, int | float | bool | None] = {}
    for key, item in value.items():
        if key not in allowed_keys:
            continue
        if item is None or type(item) in {bool, int, float}:
            result[key] = item
    return dict(sorted(result.items()))


def _render_inspection_markdown(inspection: Mapping[str, Any]) -> str:
    bindings = inspection["source_bindings"]
    submission = inspection["submission"]
    intent = inspection.get("intent_evaluation")
    review = inspection.get("review_evaluation")
    score = inspection["score"]
    judge = inspection["judge"]
    report = inspection["report"]
    lines = [
        "# Evaluation inspection",
        "",
        "## Source bindings",
        "",
        "- Run: `%s`" % bindings["run_id"],
        "- Task: `%s`" % bindings["task_id"],
        "- Trial: `%s`" % bindings["trial_id"],
        "- Evaluation: `%s`" % bindings["evaluation_id"],
        "- Revision: `%s`" % bindings["evaluation_revision"],
        "- Evaluator execution: `%s`" % bindings["evaluator_execution_digest"],
        "- Eval input digest: `%s`" % bindings["eval_input_digest"],
        "- Submission digest: `%s`" % bindings["submission_digest"],
        "",
        "## Submission",
        "",
        "- Status: `%s`" % submission["status"],
        "- Findings: %s" % submission["finding_count"],
        "- Evidence records: %s" % submission["evidence_count"],
        "- Trace present: %s" % str(submission["trace"]["present"]).lower(),
        "",
        "## Evaluation status",
        "",
        "- Intent: `%s`" % ("not_available" if intent is None else intent["status"]),
        "- Review: `%s`" % ("not_available" if review is None else review["status"]),
        "- Score artifact: `%s`" % score["artifact_digest"],
        "",
        "## Judge",
        "",
        "- Requests: %s" % judge["request_count"],
        "- Failures: %s" % judge["failure_count"],
        "- Ungraded: %s" % judge["ungraded_count"],
        "",
        "## Persisted report",
        "",
        "- Available: %s" % str(report["available"]).lower(),
        "- SHA-256: `%s`" % report["sha256"],
        "",
    ]
    return "\n".join(lines)


def _dispatch(args: argparse.Namespace) -> int:
    handlers = {
        "prepare": _handle_prepare,
        "run-agent": _handle_run_agent,
        "evaluate": _handle_evaluate,
        "inspect": _handle_inspect,
    }
    return handlers[args.command](args)


def _domain_error_category(exc: BaseException) -> tuple[str, int]:
    """Map domain failures to stable CLI categories without exposing details."""

    # Imports stay lazy so importing the parser does not eagerly construct the
    # repository, evaluator, or runner subsystems.
    from .artifacts import (
        ArtifactConflictError,
        ArtifactIntegrityError,
        ArtifactStateError,
    )
    from .orchestrator import EvaluationConflictError, EvaluationPreconditionError
    from .repository import (
        RepositoryIntegrityError,
        RepositoryPreparationError,
        RepositorySecurityError,
    )
    from .runner import RunIncompatibilityError
    from .target_replay import TargetReplayIntegrityError

    if isinstance(exc, (CliConflictError, ArtifactConflictError, EvaluationConflictError)):
        return "conflict", EXIT_CONFLICT
    if isinstance(
        exc,
        (
            CliIntegrityError,
            ArtifactIntegrityError,
            RepositoryIntegrityError,
            RepositorySecurityError,
            TargetReplayIntegrityError,
        ),
    ):
        return "integrity", EXIT_INTEGRITY
    if isinstance(
        exc,
        (
            CliPreconditionError,
            ArtifactStateError,
            EvaluationPreconditionError,
            RepositoryPreparationError,
            RunIncompatibilityError,
        ),
    ):
        return "precondition", EXIT_PRECONDITION
    return "operational", EXIT_OPERATIONAL


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the Eval CLI and return its stable process exit code."""

    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        return _dispatch(args)
    except SystemExit as exc:
        # argparse already printed help/usage.  Preserve its standard code.
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    except CliUsageError as exc:
        _json_output("cli", "error", error_code="usage", message=str(exc))
        return EXIT_USAGE
    except CliConflictError as exc:
        _json_output("cli", "error", error_code="conflict", message=str(exc))
        return EXIT_CONFLICT
    except (CliIntegrityError, ValueError, KeyError) as exc:
        # Do not echo provider exceptions, URLs or filesystem paths.  The
        # stable class name is enough for automation and avoids credential
        # leakage in a failed command's stdout.
        _json_output("cli", "error", error_code="integrity", message=type(exc).__name__)
        return EXIT_INTEGRITY
    except (CliPreconditionError, FileNotFoundError) as exc:
        _json_output("cli", "error", error_code="precondition", message=type(exc).__name__)
        return EXIT_PRECONDITION
    except Exception as exc:
        error_code, exit_code = _domain_error_category(exc)
        _json_output("cli", "error", error_code=error_code, message=type(exc).__name__)
        return exit_code


__all__ = [
    "CLI_SCHEMA_VERSION",
    "EXIT_OK",
    "EXIT_USAGE",
    "EXIT_PRECONDITION",
    "EXIT_CONFLICT",
    "EXIT_INTEGRITY",
    "EXIT_OPERATIONAL",
    "main",
]

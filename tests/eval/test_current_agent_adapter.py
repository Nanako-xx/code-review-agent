from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
from typing import Any, Callable

import pytest

from conftest import run_git
from review_agent.command import main as product_main
from review_agent.execution_profile import AgentExecutionProfile
from review_agent.model_protocol import ModelResponseKind, ModelTurnResponse
from review_agent_eval.adapters.base import (
    AdapterIncompatibilityReason,
    AgentAdapterIncompatibleError,
    AgentRunConfig,
)
from review_agent_eval.adapters.current_agent import (
    CURRENT_AGENT_ADAPTER_KIND,
    CURRENT_AGENT_ADAPTER_VERSION,
    CurrentAgentAdapter,
    current_agent_capabilities,
)
from review_agent_eval.adapters.subprocess_agent import BoundedProcessResult
from review_agent_eval.artifacts import TargetAccess
from review_agent_eval.cases import REPOSITORY_MATERIALIZER_PROTOCOL, WireContractV2
from review_agent_eval.clarification import canonical_material_claim_matcher_snapshot
from review_agent_eval.config import AgentConfigSnapshot, ResourceBudgets, derive_trial_id
from review_agent_eval.models import (
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    DiffSide,
    EvalInput,
    FailureCode,
    IntentResult,
    Repository,
    RepositoryReviewTarget,
    RepositorySource,
    ReviewRequest,
    ReviewTargetKind,
    SubmissionStatus,
    stable_id,
)


def _eval_input(base: str, head: str) -> EvalInput:
    return EvalInput(
        schema_version=EvalInput.SCHEMA_VERSION,
        task_id="task-current-adapter",
        review_target=RepositoryReviewTarget(
            kind=ReviewTargetKind.REPOSITORY,
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
                existing_ci_evidence=(),
            ),
        ),
    )


def _config(
    eval_input: EvalInput,
    *,
    review_arguments: tuple[str, ...] = ("--reviewer-provider=fake",),
    profile_arguments: tuple[str, ...] | None = None,
    adapter_extra: dict[str, object] | None = None,
    max_trace_bytes: int = 64 * 1024 * 1024,
) -> AgentRunConfig:
    from review_agent.command import review_execution_profile_from_arguments

    profile = review_execution_profile_from_arguments(
        review_arguments if profile_arguments is None else profile_arguments
    )
    adapter: dict[str, object] = {
        "kind": CURRENT_AGENT_ADAPTER_KIND,
        "command": [str(Path(sys.executable).resolve()), "-m", "review_agent"],
        "review_arguments": list(review_arguments),
        "environment_allowlist": [],
    }
    if adapter_extra:
        adapter.update(adapter_extra)
    agent = AgentConfigSnapshot(
        agent_id="agent-current",
        agent_name="Current review Agent",
        agent_version="2.0.0",
        commit="c" * 40,
        model="fake-reviewer",
        provider="fake",
        parameters={
            "adapter": adapter,
            "agent_execution_profile": {
                "profile": profile.to_dict(),
                "digest": profile.digest(),
            },
        },
        prompt_config_digest="d" * 64,
    )
    run_id = stable_id("run", "current-adapter-v3-tests", agent.digest())
    trial_id = derive_trial_id(run_id, eval_input.task_id, 1)
    matcher = canonical_material_claim_matcher_snapshot()
    capabilities = current_agent_capabilities()
    return AgentRunConfig._from_verified_binding(
        run_id=run_id,
        task_id=eval_input.task_id,
        eval_input_digest=eval_input.digest(),
        wire_contract=WireContractV2(
            case_schema_version=EVAL_CASE_SCHEMA_VERSION,
            input_schema_version=EVAL_INPUT_SCHEMA_VERSION,
            submission_schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
            review_target_kind=ReviewTargetKind.REPOSITORY,
            materializer_protocol=REPOSITORY_MATERIALIZER_PROTOCOL,
        ),
        adapter_capabilities=capabilities,
        adapter_capabilities_digest=capabilities.digest(),
        clarification_matcher=matcher,
        clarification_matcher_config_digest=matcher.digest(),
        trial_index=1,
        trial_id=trial_id,
        agent=agent,
        budgets=ResourceBudgets(
            agent_timeout_seconds=180,
            evaluator_timeout_seconds=30,
            max_agent_output_bytes=4 * 1024 * 1024,
            max_trace_bytes=max_trace_bytes,
            max_execution_artifact_file_bytes=64 * 1024 * 1024,
            max_execution_artifact_total_bytes=256 * 1024 * 1024,
            max_parallel_trials=1,
        ),
    )


def _commit_change(repo: Path) -> tuple[str, str]:
    base = run_git(repo, "rev-parse", "HEAD")
    (repo / "feature.py").write_text(
        "def enabled(value):\n    return bool(value)\n",
        encoding="utf-8",
    )
    run_git(repo, "add", "feature.py")
    run_git(repo, "commit", "-m", "add deterministic feature flag")
    return base, run_git(repo, "rev-parse", "HEAD")


def _run(
    adapter: CurrentAgentAdapter,
    eval_input: EvalInput,
    workspace: Path,
    config: AgentRunConfig,
):
    materialization_id = stable_id(
        "materialization",
        config.trial_id,
        config.eval_input_digest,
    )
    return adapter.run(
        eval_input,
        workspace,
        config,
        object(),
        target_access=TargetAccess(
            target_materialization_id=materialization_id,
            readable_relative_paths=("target/repository",),
        ),
        target_materialization_id=materialization_id,
    )


def _in_process_runner(
    after: Callable[[Path, bytes], bytes] | None = None,
):
    def run(argv, **kwargs):
        stdout = io.StringIO()
        stderr = io.StringIO()
        command_index = list(argv).index("review")
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                returncode = product_main(list(argv)[command_index:])
            except SystemExit as error:
                returncode = int(error.code)
        output = stdout.getvalue().encode("utf-8")
        if after is not None:
            output = after(Path(kwargs["workspace"]), output)
        return BoundedProcessResult(
            stdout=output,
            returncode=returncode,
            failure_code=None,
            output_bytes=len(output) + len(stderr.getvalue().encode("utf-8")),
        )

    return run


def test_current_adapter_v3_runs_product_v6_and_returns_measurable_submission(
    git_repo: Path,
) -> None:
    base, head = _commit_change(git_repo)
    eval_input = _eval_input(base, head)
    config = _config(eval_input)

    submission = _run(
        CurrentAgentAdapter(process_runner=_in_process_runner()),
        eval_input,
        git_repo,
        config,
    )

    assert CURRENT_AGENT_ADAPTER_VERSION == "3"
    assert submission.status is SubmissionStatus.COMPLETED
    assert submission.failure is None
    assert submission.intent.status is IntentResult.SUFFICIENT
    assert submission.intent.goal == "Keep the function behavior deterministic."
    assert submission.review.findings == ()
    assert submission.review.uncertainties
    assert submission.trace_ref is not None
    assert submission.trace_ref.value == ".ra-v6"
    assert (git_repo / ".ra-v6").is_dir()
    assert not (git_repo / ".review-agent" / "runs").exists()
    pr = next((git_repo / ".ra-v6" / "pr").glob("p-*"))
    metadata = json.loads((pr / "PR" / "pr.json").read_text("utf-8"))
    assert metadata["pr_number_or_external_review_id"] == eval_input.task_id


def test_product_failed_review_result_is_still_scored_as_completed_submission(
    git_repo: Path,
) -> None:
    base, head = _commit_change(git_repo)
    eval_input = _eval_input(base, head)
    config = _config(
        eval_input,
        review_arguments=("--reviewer-provider=none",),
    )

    submission = _run(
        CurrentAgentAdapter(process_runner=_in_process_runner()),
        eval_input,
        git_repo,
        config,
    )

    product_result_path = next(
        (git_repo / ".ra-v6").glob("pr/p-*/Snapshots/s-*/Results/review-result.json")
    )
    assert json.loads(product_result_path.read_text("utf-8"))["status"] == "failed"
    assert submission.status is SubmissionStatus.COMPLETED
    assert submission.failure is None
    assert submission.review.findings == ()
    assert submission.review.uncertainties


def test_adapter_projects_product_finding_to_line_and_diff_evidence(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, head = _commit_change(git_repo)
    eval_input = _eval_input(base, head)
    config = _config(eval_input)

    class FindingAdapter:
        def complete_turn(self, request):
            assert request.parameters["response_schema"] == "reviewer_output_v2"
            return ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=json.dumps(
                    {
                        "findings": [
                            {
                                "claim": (
                                    "When value defines __bool__ with side effects, "
                                    "coercing it here executes those effects."
                                ),
                                "severity": "medium",
                                "path": "feature.py",
                                "line": 2,
                                "suggestion": (
                                    "Require a bool input or add a side-effecting "
                                    "__bool__ regression test."
                                ),
                            }
                        ],
                        "uncertainties": [],
                    },
                    separators=(",", ":"),
                ),
                provider_name="fake",
                model="finding-reviewer",
            )

    class Factory:
        def create(self):
            return FindingAdapter()

    monkeypatch.setattr(
        "review_agent.product_runtime.build_model_adapter_factory_from_config",
        lambda *_args, **_kwargs: Factory(),
    )
    submission = _run(
        CurrentAgentAdapter(process_runner=_in_process_runner()),
        eval_input,
        git_repo,
        config,
    )

    assert submission.status is SubmissionStatus.COMPLETED
    assert len(submission.review.findings) == 1
    finding = submission.review.findings[0]
    assert finding.path == "feature.py"
    assert finding.side is DiffSide.RIGHT
    assert finding.from_line == finding.to_line == 2
    assert len(finding.evidence_refs) == 1
    assert len(submission.evidence) == 1
    evidence = submission.evidence[0]
    assert evidence.evidence_id == finding.evidence_refs[0]
    assert evidence.source.path == "feature.py"
    assert "+    return bool(value)" in evidence.excerpt


def test_adapter_rejects_tampered_review_result_artifact(
    git_repo: Path,
) -> None:
    base, head = _commit_change(git_repo)
    eval_input = _eval_input(base, head)
    config = _config(eval_input)

    def tamper(workspace: Path, output: bytes) -> bytes:
        result = next(
            (workspace / ".ra-v6").glob(
                "pr/p-*/Snapshots/s-*/Results/review-result.json"
            )
        )
        result.write_bytes(result.read_bytes() + b" ")
        return output

    submission = _run(
        CurrentAgentAdapter(process_runner=_in_process_runner(tamper)),
        eval_input,
        git_repo,
        config,
    )

    assert submission.status is SubmissionStatus.INVALID_OUTPUT
    assert submission.failure.code is FailureCode.SCHEMA_MISMATCH


def test_adapter_rejects_stdout_that_differs_from_authoritative_review_result(
    git_repo: Path,
) -> None:
    base, head = _commit_change(git_repo)
    eval_input = _eval_input(base, head)
    config = _config(eval_input)

    submission = _run(
        CurrentAgentAdapter(
            process_runner=_in_process_runner(
                lambda _workspace, output: output + b"unexpected\n"
            )
        ),
        eval_input,
        git_repo,
        config,
    )

    assert submission.status is SubmissionStatus.INVALID_OUTPUT
    assert submission.failure.code is FailureCode.SCHEMA_MISMATCH


def test_adapter_passes_stable_identity_and_private_workspace_arguments(
    git_repo: Path,
) -> None:
    base, head = _commit_change(git_repo)
    eval_input = _eval_input(base, head)
    config = _config(eval_input)
    seen: list[tuple[str, ...]] = []

    def process(argv, **_kwargs):
        seen.append(tuple(argv))
        return BoundedProcessResult(
            stdout=b"",
            returncode=1,
            failure_code=None,
            output_bytes=0,
        )

    submission = _run(
        CurrentAgentAdapter(process_runner=process),
        eval_input,
        git_repo,
        config,
    )

    assert submission.status is SubmissionStatus.FAILED
    assert submission.failure.code is FailureCode.NON_ZERO_EXIT
    argv = seen[0]
    assert "--external-review-id=" + eval_input.task_id in argv
    assert "--workspace-root=" + str(git_repo / ".ra-v6") in argv
    assert "--format=json" in argv
    assert not any("memory" in value for value in argv)
    assert not any("reviewer-loop" in value for value in argv)


@pytest.mark.parametrize(
    "forged",
    (
        "--external-review-id=forged",
        "--workspace-root=forged",
        "--format=markdown",
        "--memory-mode=read",
        "--reviewer-loop=agent-loop",
    ),
)
def test_adapter_rejects_arguments_that_override_v6_invocation_authority(
    git_repo: Path,
    forged: str,
) -> None:
    base, head = _commit_change(git_repo)
    eval_input = _eval_input(base, head)
    config = _config(
        eval_input,
        review_arguments=(forged,),
        profile_arguments=("--reviewer-provider=fake",),
    )

    submission = _run(CurrentAgentAdapter(), eval_input, git_repo, config)

    assert submission.status is SubmissionStatus.FAILED
    assert submission.failure.code is FailureCode.ADAPTER_ERROR
    assert not (git_repo / ".ra-v6").exists()


def test_adapter_rejects_execution_profile_drift(
    git_repo: Path,
) -> None:
    base, head = _commit_change(git_repo)
    eval_input = _eval_input(base, head)
    config = _config(
        eval_input,
        review_arguments=("--reviewer-provider=fake",),
        profile_arguments=("--reviewer-provider=none",),
    )

    with pytest.raises(AgentAdapterIncompatibleError) as raised:
        _run(CurrentAgentAdapter(), eval_input, git_repo, config)

    assert raised.value.reason is AdapterIncompatibilityReason.EXECUTION_PROFILE_MISMATCH
    assert not (git_repo / ".ra-v6").exists()


def test_adapter_rejects_legacy_memory_configuration_field(
    git_repo: Path,
) -> None:
    base, head = _commit_change(git_repo)
    eval_input = _eval_input(base, head)
    config = _config(eval_input, adapter_extra={"memory_mode": "off"})

    submission = _run(CurrentAgentAdapter(), eval_input, git_repo, config)

    assert submission.status is SubmissionStatus.FAILED
    assert submission.failure.code is FailureCode.ADAPTER_ERROR


def test_capabilities_are_bound_to_adapter_v3() -> None:
    capabilities = current_agent_capabilities()

    assert capabilities.adapter_id == CURRENT_AGENT_ADAPTER_KIND
    assert capabilities.adapter_version == "3"

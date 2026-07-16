from __future__ import annotations

import hashlib
import re
from dataclasses import FrozenInstanceError
from typing import Any, Dict, Optional

import pytest

import review_agent_eval.config as config_module
from review_agent_eval.cases import RunCaseSnapshot, SuiteCase, SuiteManifest
from review_agent_eval.config import (
    AgentConfigSnapshot,
    ClarificationMatcherSnapshot,
    EvalRunConfig,
    EvaluatorExecutionConfig,
    EvaluatorRunConfig,
    ResourceBudgets,
    SuiteRunConfig,
    derive_case_path_id,
    derive_evaluation_id,
    derive_trial_id,
    load_eval_run_config,
    validate_trial_id,
)
from review_agent_eval.models import (
    EvalCase,
    SchemaError,
    canonical_json,
    canonical_sha256,
)


def _build_case_snapshot(task_id: str = "task-001") -> RunCaseSnapshot:
    case = EvalCase.from_dict(
        {
            "schema_version": "eval_case_v1",
            "task_id": task_id,
            "case_version": 2,
            "source": {
                "suite": "core-capability",
                "origin": "hand_authored",
                "source_id": "source-case-001",
                "source_version": "source-v1",
                "source_uri": None,
                "license": None,
                "content_hash": "9" * 64,
            },
            "input": {
                "repository": {
                    "source": "fixture",
                    "path": "repositories/case-001",
                    "url": None,
                    "base_revision": "a" * 40,
                    "head_revision": "b" * 40,
                },
                "review_request": {
                    "title": "Review authorization behavior",
                    "description": None,
                    "user_intent": "Keep authorization intact",
                    "review_focus": None,
                    "linked_requirements": [],
                    "project_rules": [],
                    "existing_ci_evidence": [],
                },
            },
            "clarification_script": {"max_rounds": 1, "answers": []},
            "intent_truth": {
                "scorable": False,
                "authority": None,
                "expected_claims": [],
                "forbidden_claims": [],
                "clarification_policy": None,
            },
            "review_truth": {
                "completeness": "closed_world",
                "novel_finding_policy": "forbid",
                "expected_findings": [],
                "known_invalid_findings": [],
            },
        }
    )
    case_bytes = case.to_json().encode("utf-8")
    manifest = SuiteManifest.from_dict(
        {
            "schema_version": "suite_manifest_v1",
            "suite_id": "core-capability",
            "suite_version": "3",
            "source": {
                "kind": "core",
                "source_id": "suite-source",
                "source_version": "source-v1",
                "source_uri": None,
                "license": None,
                "content_hash": "8" * 64,
            },
            "cases": [
                {
                    "task_id": task_id,
                    "case_version": case.case_version,
                    "path": "cases/case-001.json",
                    "split": "regression",
                    "protocol_id": "native_repository",
                    "dimensions": [{"name": "language", "value": "python"}],
                    "raw_file_size_bytes": len(case_bytes),
                    "raw_file_sha256": hashlib.sha256(case_bytes).hexdigest(),
                    "canonical_case_digest": canonical_sha256(case),
                    "eval_input_digest": case.eval_input().digest(),
                    "truth_completeness": "closed_world",
                }
            ],
        }
    )
    return RunCaseSnapshot.build(manifest, ((manifest.cases[0], case),))


@pytest.fixture
def case_snapshot() -> RunCaseSnapshot:
    return _build_case_snapshot()


def agent_config(
    *,
    model: str = "agent-model",
    parameters: Optional[Dict[str, Any]] = None,
) -> AgentConfigSnapshot:
    return AgentConfigSnapshot(
        agent_id="agent-current",
        agent_name="Current review agent",
        agent_version="3.4.1",
        commit="a" * 40,
        model=model,
        provider="provider-a",
        parameters={"temperature": 0, "max_tokens": 4096}
        if parameters is None
        else parameters,
        prompt_config_digest="b" * 64,
    )


def evaluator_config(
    *, judge_version: str = "judge-v1", model: str = "judge-model"
) -> EvaluatorRunConfig:
    return EvaluatorRunConfig(
        evaluator_id="core-evaluator",
        evaluator_version="1.2.0",
        grader_version="grader-v4",
        judge_version=judge_version,
        model=model,
        provider="provider-j",
        parameters={"temperature": 0, "reasoning_effort": "medium"},
        rubric_version="rubric-v7",
        rubric_digest="c" * 64,
    )


def matcher_config(
    *,
    matcher_version: str = "matcher-v1",
    threshold: float | None = None,
) -> ClarificationMatcherSnapshot:
    return ClarificationMatcherSnapshot(
        matcher_id="canonical-material-claim",
        matcher_version=matcher_version,
        implementation_digest="d" * 64,
        model_artifact_digest=None,
        rubric_digest="e" * 64,
        normalization_version="unicode-whitespace-casefold-v1",
        threshold=threshold,
        parameters={"unicode_version": "14.0.0"},
    )


def suite_config(snapshot: RunCaseSnapshot) -> SuiteRunConfig:
    return SuiteRunConfig.from_case_snapshot(snapshot)


def budgets(*, parallel: int = 2) -> ResourceBudgets:
    return ResourceBudgets(
        agent_timeout_seconds=900,
        evaluator_timeout_seconds=300,
        max_agent_output_bytes=2 * 1024 * 1024,
        max_trace_bytes=4 * 1024 * 1024,
        max_execution_artifact_file_bytes=16 * 1024 * 1024,
        max_execution_artifact_total_bytes=128 * 1024 * 1024,
        max_parallel_trials=parallel,
    )


def run_config(
    snapshot: RunCaseSnapshot,
    *,
    run_instance_key: str = "instance-20260716-001",
    agent: Optional[AgentConfigSnapshot] = None,
    matcher: Optional[ClarificationMatcherSnapshot] = None,
    evaluator: Optional[EvaluatorRunConfig] = None,
    suite: Optional[SuiteRunConfig] = None,
    resource_budgets: Optional[ResourceBudgets] = None,
    trial_count: int = 3,
) -> EvalRunConfig:
    return EvalRunConfig.create(
        run_instance_key=run_instance_key,
        agent=agent or agent_config(),
        clarification_matcher=matcher or matcher_config(),
        evaluator=evaluator or evaluator_config(),
        suite=suite or suite_config(snapshot),
        trial_count=trial_count,
        resource_budgets=resource_budgets
        or budgets(parallel=min(2, trial_count)),
    )


def test_suite_run_config_reuses_verified_snapshot_protocol_and_round_trips(
    case_snapshot: RunCaseSnapshot,
) -> None:
    suite = SuiteRunConfig.from_case_snapshot(case_snapshot)

    assert config_module.SuiteCase is SuiteCase
    assert suite.suite_id == case_snapshot.manifest.suite_id
    assert suite.suite_version == case_snapshot.manifest.suite_version
    assert suite.manifest_digest == case_snapshot.manifest.digest()
    assert suite.case_snapshot_id == case_snapshot.snapshot_id
    assert suite.case_snapshot_digest == case_snapshot.snapshot_digest
    assert suite.cases == tuple(
        entry.manifest_case for entry in case_snapshot.cases
    )
    assert all(
        suite_case is entry.manifest_case
        for suite_case, entry in zip(suite.cases, case_snapshot.cases)
    )
    assert suite.cases[0].canonical_case_digest == (
        case_snapshot.cases[0].canonical_case_digest
    )
    assert "canonical_case_digest" in suite.cases[0].to_dict()

    assert SuiteRunConfig.from_dict(suite.to_dict()) == suite


def test_final_run_config_records_complete_reproduction_identity_and_round_trips(
    case_snapshot: RunCaseSnapshot,
) -> None:
    config = run_config(case_snapshot)

    assert config.schema_version == "eval_run_config_v1"
    assert config.run_instance_key == "instance-20260716-001"
    assert config.agent.agent_id == "agent-current"
    assert config.agent.commit == "a" * 40
    assert config.agent.model == "agent-model"
    assert config.agent.provider == "provider-a"
    assert dict(config.agent.parameters)["temperature"] == 0
    assert config.agent.prompt_config_digest == "b" * 64
    assert config.agent_config_digest == canonical_sha256(config.agent)
    assert config.clarification_matcher.matcher_id == "canonical-material-claim"
    assert config.clarification_matcher_config_digest == canonical_sha256(
        config.clarification_matcher
    )
    assert config.evaluator_config_digest == canonical_sha256(config.evaluator)
    assert config.evaluator.judge_version == "judge-v1"
    assert config.evaluator.rubric_version == "rubric-v7"
    assert config.suite.manifest_digest == case_snapshot.manifest.digest()
    assert config.suite.case_snapshot_id == case_snapshot.snapshot_id
    assert config.suite.case_snapshot_digest == case_snapshot.snapshot_digest
    assert config.suite.cases[0].canonical_case_digest == (
        case_snapshot.cases[0].manifest_case.canonical_case_digest
    )
    assert config.suite.cases[0].eval_input_digest == (
        case_snapshot.cases[0].manifest_case.eval_input_digest
    )
    assert config.trial_count == 3
    assert config.resource_budgets.agent_timeout_seconds == 900

    encoded = config.to_json().encode("utf-8")
    assert encoded == canonical_json(config).encode("utf-8")
    assert load_eval_run_config(encoded) == config
    assert load_eval_run_config(encoded).digest() == config.digest()


def test_config_exports_only_final_protocol_names() -> None:
    assert AgentConfigSnapshot.__name__ == "AgentConfigSnapshot"
    assert config_module.SuiteCase is SuiteCase
    assert not hasattr(config_module, "AgentRunConfig")
    for alias in (
        "RunConfig",
        "CaseDigest",
        "ResourceBudget",
        "PersistedAgentConfig",
    ):
        assert not hasattr(config_module, alias)
        assert alias not in config_module.__all__


def test_run_id_excludes_evaluator_but_includes_snapshot_binding(
    case_snapshot: RunCaseSnapshot,
) -> None:
    original = run_config(case_snapshot)
    same = run_config(case_snapshot)
    changed_judge = run_config(
        case_snapshot,
        evaluator=evaluator_config(judge_version="judge-v99", model="other-judge"),
    )
    next_instance = run_config(
        case_snapshot,
        run_instance_key="instance-20260716-002",
    )
    changed_agent = run_config(
        case_snapshot,
        agent=agent_config(model="agent-model-v2"),
    )
    changed_matcher = run_config(
        case_snapshot,
        matcher=matcher_config(matcher_version="matcher-v2"),
    )
    changed_agent_budgets_payload = budgets(parallel=2).to_dict()
    changed_agent_budgets_payload["agent_timeout_seconds"] = 901
    changed_agent_budgets = run_config(
        case_snapshot,
        resource_budgets=ResourceBudgets.from_dict(
            changed_agent_budgets_payload
        ),
    )
    changed_evaluator_budgets_payload = budgets(parallel=2).to_dict()
    changed_evaluator_budgets_payload["evaluator_timeout_seconds"] = 301
    changed_evaluator_budgets = run_config(
        case_snapshot,
        resource_budgets=ResourceBudgets.from_dict(
            changed_evaluator_budgets_payload
        ),
    )

    changed_binding_payload = suite_config(case_snapshot).to_dict()
    changed_binding_payload["case_snapshot_digest"] = "0" * 64
    changed_binding = SuiteRunConfig.from_dict(changed_binding_payload)
    changed_snapshot = run_config(case_snapshot, suite=changed_binding)

    assert original.run_id == same.run_id
    assert original.run_id == changed_judge.run_id
    assert original.evaluator_config_digest != changed_judge.evaluator_config_digest
    assert original.run_id != changed_snapshot.run_id
    assert original.run_id != next_instance.run_id
    assert original.run_id != changed_agent.run_id
    assert original.run_id != changed_matcher.run_id
    assert original.run_id != changed_agent_budgets.run_id
    assert original.run_id == changed_evaluator_budgets.run_id
    assert EvaluatorExecutionConfig.from_resource_budgets(
        original.evaluator, original.resource_budgets
    ).digest() != EvaluatorExecutionConfig.from_resource_budgets(
        changed_evaluator_budgets.evaluator,
        changed_evaluator_budgets.resource_budgets,
    ).digest()
    assert re.fullmatch(r"run-[0-9a-f]{64}", original.run_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("matcher_version", "matcher-v2"),
        ("implementation_digest", "1" * 64),
        ("model_artifact_digest", "2" * 64),
        ("rubric_digest", "3" * 64),
        ("normalization_version", "normalization-v2"),
        ("threshold", 0.75),
        ("parameters", {"unicode_version": "15.0.0"}),
    ],
)
def test_run_id_binds_every_matcher_reproduction_dimension(
    case_snapshot: RunCaseSnapshot,
    field: str,
    value: object,
) -> None:
    original = run_config(case_snapshot)
    changed_snapshot = matcher_config().to_dict()
    changed_snapshot[field] = value
    changed = run_config(
        case_snapshot,
        matcher=ClarificationMatcherSnapshot.from_dict(changed_snapshot),
    )

    assert original.run_id != changed.run_id


@pytest.mark.parametrize("threshold", [-0.1, 1.1, True, float("nan")])
def test_matcher_snapshot_rejects_invalid_thresholds(threshold: object) -> None:
    with pytest.raises(SchemaError, match="threshold"):
        ClarificationMatcherSnapshot(
            matcher_id="semantic-material-claim",
            matcher_version="1",
            implementation_digest="1" * 64,
            model_artifact_digest="2" * 64,
            rubric_digest="3" * 64,
            normalization_version="normalization-v1",
            threshold=threshold,
            parameters={},
        )


def test_trial_and_case_path_ids_are_stable_and_never_embed_opaque_task_id() -> None:
    opaque_task_id = "../../private/case:with/slashes"
    snapshot = _build_case_snapshot(opaque_task_id)
    config = run_config(snapshot)
    trial_id = derive_trial_id(config.run_id, opaque_task_id, 1)
    case_path_id = derive_case_path_id(opaque_task_id)

    assert trial_id == config.trial_id(opaque_task_id, 1)
    assert validate_trial_id(trial_id, config.run_id, opaque_task_id, 1) == trial_id
    assert re.fullmatch(r"trial-[0-9a-f]{64}", trial_id)
    assert re.fullmatch(r"case-[0-9a-f]{64}", case_path_id)
    assert opaque_task_id not in trial_id
    assert opaque_task_id not in case_path_id
    assert "/" not in trial_id and "\\" not in trial_id

    with pytest.raises(SchemaError):
        validate_trial_id("../trial", config.run_id, opaque_task_id, 1)


def test_evaluation_id_is_versioned_by_evaluator_digest_and_revision(
    case_snapshot: RunCaseSnapshot,
) -> None:
    config = run_config(case_snapshot)
    execution = EvaluatorExecutionConfig.from_resource_budgets(
        config.evaluator, config.resource_budgets
    )
    changed = EvaluatorExecutionConfig.from_resource_budgets(
        evaluator_config(judge_version="judge-v2"), config.resource_budgets
    )
    first = derive_evaluation_id(config.run_id, execution.digest(), "v1")
    second = derive_evaluation_id(config.run_id, execution.digest(), "v2")

    assert first != second
    assert first != derive_evaluation_id(config.run_id, changed.digest(), "v1")
    assert re.fullmatch(r"evaluation-[0-9a-f]{64}", first)
    with pytest.raises(SchemaError):
        derive_evaluation_id(config.run_id, execution.digest(), "../v1")


def test_evaluator_execution_config_binds_timeout_limits_and_judge() -> None:
    evaluator = evaluator_config()
    resources = budgets(parallel=1)
    execution = EvaluatorExecutionConfig.from_resource_budgets(
        evaluator, resources
    )
    changed_timeout = EvaluatorExecutionConfig.create(
        evaluator=evaluator,
        evaluator_timeout_seconds=resources.evaluator_timeout_seconds + 1,
        max_execution_artifact_file_bytes=(
            resources.max_execution_artifact_file_bytes
        ),
        max_execution_artifact_total_bytes=(
            resources.max_execution_artifact_total_bytes
        ),
    )

    assert EvaluatorExecutionConfig.from_json(execution.to_json()) == execution
    assert execution.digest() != changed_timeout.digest()
    assert execution.evaluator_config_digest == evaluator.digest()

    payload = execution.to_dict()
    payload["evaluator_config_digest"] = "0" * 64
    with pytest.raises(SchemaError, match="digest"):
        EvaluatorExecutionConfig.from_dict(payload)


@pytest.mark.parametrize(
    "parameters",
    [
        {"api_key": "ordinary-secret"},
        {"environment": {"HOME": "/private"}},
        {"raw_reasoning": "hidden chain"},
        {"endpoint": "https://user:password@example.test/v1"},
        {"endpoint": "sk-test-secret-value"},
        {"nested": [{"client_secret": "ordinary-secret"}]},
        {"token": "ordinary-secret-value"},
        {"credential_note": "AWS_SECRET_ACCESS_KEY=ordinary-secret"},
        {"headers": "Authorization: Basic dXNlcjpwYXNz"},
        {"reasoning_content": "private intermediate steps"},
        {"provider_output": "<think>private intermediate steps</think>"},
        {
            "diagnostic": (
                "Path=C:\\Windows\n"
                "UserProfile=C:\\Users\\private\n"
                "ComSpec=C:\\Windows\\System32\\cmd.exe"
            )
        },
        {
            "diagnostic": {
                "HOME": "/private",
                "PATH": "/bin",
                "USER": "private-user",
            }
        },
    ],
)
def test_config_rejects_secrets_full_env_url_userinfo_and_raw_reasoning(
    parameters: Dict[str, Any],
) -> None:
    with pytest.raises(SchemaError) as caught:
        agent_config(parameters=parameters)

    # Validation errors identify the unsafe class, never echo the value.
    assert "ordinary-secret" not in str(caught.value)
    assert "sk-test-secret-value" not in str(caught.value)
    assert "user:password" not in str(caught.value)


def test_strict_hydration_rejects_unknown_duplicate_and_digest_mismatch(
    case_snapshot: RunCaseSnapshot,
) -> None:
    config = run_config(case_snapshot)
    payload = config.to_dict()
    payload["unknown"] = True
    with pytest.raises(SchemaError, match="unknown field"):
        EvalRunConfig.from_dict(payload)

    duplicate = config.to_json().replace(
        '"schema_version":"eval_run_config_v1"',
        '"schema_version":"eval_run_config_v1","schema_version":"eval_run_config_v1"',
        1,
    )
    with pytest.raises(SchemaError, match="duplicate"):
        EvalRunConfig.from_json(duplicate)

    payload = config.to_dict()
    payload["agent_config_digest"] = "0" * 64
    with pytest.raises(SchemaError, match="agent_config_digest"):
        EvalRunConfig.from_dict(payload)

    payload = config.to_dict()
    payload["clarification_matcher_config_digest"] = "0" * 64
    with pytest.raises(SchemaError, match="clarification_matcher_config_digest"):
        EvalRunConfig.from_dict(payload)

    payload = config.to_dict()
    payload["run_id"] = "run-" + "0" * 64
    with pytest.raises(SchemaError, match="run_id"):
        EvalRunConfig.from_dict(payload)


def test_snapshot_and_case_tampering_invalidates_run_identity(
    case_snapshot: RunCaseSnapshot,
) -> None:
    config = run_config(case_snapshot)
    mutations = (
        ("manifest_digest", "0" * 64),
        ("case_snapshot_id", "run-case-snapshot-" + "0" * 64),
        ("case_snapshot_digest", "0" * 64),
    )
    for field, value in mutations:
        payload = config.to_dict()
        payload["suite"][field] = value
        with pytest.raises(SchemaError, match="run_id"):
            EvalRunConfig.from_dict(payload)

    payload = config.to_dict()
    payload["suite"]["cases"][0]["canonical_case_digest"] = "0" * 64
    with pytest.raises(SchemaError, match="run_id"):
        EvalRunConfig.from_dict(payload)


def test_suite_snapshot_identity_and_case_selection_fail_closed(
    case_snapshot: RunCaseSnapshot,
) -> None:
    suite = suite_config(case_snapshot)

    payload = suite.to_dict()
    payload["case_snapshot_id"] = "../snapshot"
    with pytest.raises(SchemaError, match="case_snapshot_id"):
        SuiteRunConfig.from_dict(payload)

    payload = suite.to_dict()
    payload["cases"] = []
    with pytest.raises(SchemaError, match="must not be empty"):
        SuiteRunConfig.from_dict(payload)

    payload = suite.to_dict()
    payload["cases"].append(dict(payload["cases"][0]))
    with pytest.raises(SchemaError, match="duplicate task_id"):
        SuiteRunConfig.from_dict(payload)


def test_config_and_nested_parameter_tree_are_deeply_immutable(
    case_snapshot: RunCaseSnapshot,
) -> None:
    config = run_config(
        case_snapshot,
        agent=agent_config(parameters={"nested": {"items": [1, 2, 3]}}),
    )

    with pytest.raises(FrozenInstanceError):
        config.run_id = "run-" + "0" * 64
    with pytest.raises(TypeError):
        config.agent.parameters["new"] = True
    with pytest.raises(TypeError):
        config.agent.parameters["nested"]["new"] = True
    assert config.agent.parameters["nested"]["items"] == (1, 2, 3)


def test_resource_and_trial_budgets_fail_closed(
    case_snapshot: RunCaseSnapshot,
) -> None:
    with pytest.raises(SchemaError):
        ResourceBudgets(
            agent_timeout_seconds=0,
            evaluator_timeout_seconds=1,
            max_agent_output_bytes=1,
            max_trace_bytes=1,
            max_execution_artifact_file_bytes=1,
            max_execution_artifact_total_bytes=1,
            max_parallel_trials=1,
        )

    with pytest.raises(SchemaError, match="max_parallel_trials"):
        EvalRunConfig.create(
            run_instance_key="instance",
            agent=agent_config(),
            clarification_matcher=matcher_config(),
            evaluator=evaluator_config(),
            suite=suite_config(case_snapshot),
            trial_count=1,
            resource_budgets=budgets(parallel=2),
        )


def test_json_parameters_reject_non_json_and_non_finite_values() -> None:
    with pytest.raises(SchemaError):
        agent_config(parameters={"bad": object()})
    with pytest.raises(SchemaError):
        agent_config(parameters={"bad": float("nan")})


def test_safe_model_token_count_parameters_are_not_mistaken_for_credentials() -> None:
    config = agent_config(
        parameters={
            "max_tokens": 4096,
            "max_completion_tokens": 2048,
            "reasoning_effort": "medium",
        }
    )
    assert config.parameters["max_tokens"] == 4096

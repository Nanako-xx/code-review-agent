from __future__ import annotations

import hashlib
import re
from dataclasses import FrozenInstanceError
from typing import Any, Dict, Optional, Sequence

import pytest

import review_agent_eval.config as config_module
from review_agent_eval.cases import RunCaseSnapshot, SuiteCase, SuiteManifest
from review_agent_eval.config import (
    AgentConfigSnapshot,
    ClarificationMatcherSnapshot,
    DEFAULT_JUDGE_CACHE_POLICY_VERSION,
    EvalRunConfig,
    EvaluatorExecutionConfig,
    EvaluatorRunConfig,
    JudgeExecutionBudgets,
    JudgeKind,
    JudgeProfileSnapshot,
    ResourceBudgets,
    SuiteRunConfig,
    derive_case_path_id,
    derive_evaluation_id,
    derive_trial_id,
    load_eval_run_config,
    validate_safe_json,
    validate_trial_id,
)
from review_agent_eval.models import (
    EvidenceKind,
    EvalCase,
    ReviewTargetKind,
    SchemaError,
    UnsupportedProtocolVersionError,
    canonical_json,
    canonical_sha256,
)


def _build_case_snapshot(task_id: str = "task-001") -> RunCaseSnapshot:
    case = EvalCase.from_dict(
        {
            "schema_version": "eval_case_v2",
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
                "review_target": {
                    "kind": "repository",
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
            "review_evaluator_context": {"truth_contexts": []},
        }
    )
    case_bytes = case.to_json().encode("utf-8")
    manifest = SuiteManifest.from_dict(
        {
            "schema_version": "suite_manifest_v2",
            "suite_id": "core-capability",
            "suite_version": "3",
            "wire_contract": {
                "case_schema_version": "eval_case_v2",
                "input_schema_version": "eval_input_v2",
                "submission_schema_version": "eval_submission_v2",
                "review_target_kind": "repository",
                "materializer_protocol": "repository-materializer-v2",
            },
            "source": {
                "kind": "core",
                "source_id": "suite-source",
                "source_version": "source-v1",
                "source_uri": None,
                "license": None,
                "content_hash": "8" * 64,
                "preparation_binding": None,
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


def _identity_digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def judge_profile(
    kind: JudgeKind,
    *,
    judge_version: str = "judge-v1",
    model: str = "judge-model",
) -> JudgeProfileSnapshot:
    slug = kind.value
    return JudgeProfileSnapshot(
        schema_version="eval_judge_profile_v1",
        kind=kind,
        judge_id=f"{slug}-judge",
        judge_version=judge_version,
        adapter_id="unified-model-adapter",
        adapter_version="adapter-v1",
        adapter_config_digest=_identity_digest(f"{slug}:adapter-config"),
        provider="provider-j",
        model=model,
        model_artifact_digest=None,
        parameters={"temperature": 0, "reasoning_effort": "medium"},
        system_prompt_version=f"{slug}-system-v1",
        system_prompt_digest=_identity_digest(f"{slug}:system-prompt"),
        rubric_id=f"{slug}-rubric",
        rubric_version=f"{slug}-rubric-v7",
        rubric_digest=_identity_digest(f"{slug}:rubric"),
        response_schema_version=f"eval_{slug}_judge_response_v1",
        response_schema_digest=_identity_digest(f"{slug}:response-schema"),
        context_builder_version=f"{slug}-context-v1",
        parser_version=f"{slug}-parser-v1",
    )


def evaluator_config(
    *,
    judge_version: str = "judge-v1",
    model: str = "judge-model",
    judge_profiles: Optional[Sequence[JudgeProfileSnapshot]] = None,
) -> EvaluatorRunConfig:
    profiles = (
        tuple(judge_profiles)
        if judge_profiles is not None
        else tuple(
            judge_profile(kind, judge_version=judge_version, model=model)
            for kind in JudgeKind
        )
    )
    return EvaluatorRunConfig(
        evaluator_id="core-evaluator",
        evaluator_version="1.2.0",
        grader_version="grader-v4",
        judge_profiles=profiles,
    )


def judge_budgets(**overrides: Any) -> JudgeExecutionBudgets:
    payload = JudgeExecutionBudgets.defaults(
        evaluator_timeout_seconds=300,
        max_execution_artifact_file_bytes=16 * 1024 * 1024,
        max_execution_artifact_total_bytes=128 * 1024 * 1024,
    ).to_dict()
    payload.update(overrides)
    return JudgeExecutionBudgets.from_dict(payload)


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


def adapter_capabilities():
    capability_type = getattr(config_module, "AdapterCapabilitiesV2")
    return capability_type.from_dict(
        {
            "schema_version": "eval_adapter_capabilities_v2",
            "adapter_id": "current-agent-cli-v2",
            "adapter_version": "2",
            "input_schema_version": "eval_input_v2",
            "submission_schema_version": "eval_submission_v2",
            "target_kinds": ["repository"],
            "evidence_kinds": [
                "repository_file",
                "repository_diff",
                "command_output",
            ],
            "clarification_protocol": "canonical-clarification-v2",
            "trace_protocol": "local-trace-v2",
            "subprocess_wire_version": None,
            "isolation_profile": "repository-worktree-v2",
        }
    )


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
        adapter_capabilities=adapter_capabilities(),
        trial_count=trial_count,
        resource_budgets=resource_budgets
        or budgets(parallel=min(2, trial_count)),
    )


def test_run_config_binds_wire_preparation_and_adapter_capabilities(
    case_snapshot: RunCaseSnapshot,
) -> None:
    config = run_config(case_snapshot)

    assert config.schema_version == "eval_run_config_v2"
    assert config.wire_contract == case_snapshot.wire_contract
    assert config.suite_preparation_binding_digest is None
    assert config.adapter_capabilities == adapter_capabilities()
    assert config.adapter_capabilities_digest == adapter_capabilities().digest()
    assert config.target_kinds == (ReviewTargetKind.REPOSITORY,)
    assert config.materializer_protocol == "repository-materializer-v2"

    payload = config.to_dict()
    payload["adapter_capabilities_digest"] = "0" * 64
    with pytest.raises(SchemaError, match="adapter_capabilities_digest"):
        EvalRunConfig.from_dict(payload)

    payload = config.to_dict()
    payload["wire_contract"]["materializer_protocol"] = (
        "frozen-context-materializer-v2"
    )
    with pytest.raises(SchemaError, match="materializer|wire contract"):
        EvalRunConfig.from_dict(payload)

    payload = config.to_dict()
    payload["target_kinds"] = ["frozen_context"]
    with pytest.raises(SchemaError, match="target kind|target_kinds"):
        EvalRunConfig.from_dict(payload)


def test_run_config_direct_construction_freezes_target_kinds_alias(
    case_snapshot: RunCaseSnapshot,
) -> None:
    baseline = run_config(case_snapshot)
    caller_targets = [ReviewTargetKind.REPOSITORY]
    config = EvalRunConfig(
        **{
            **vars(baseline),
            "target_kinds": caller_targets,
        }
    )
    encoded = config.to_json()
    digest = config.digest()
    run_id = config.run_id

    caller_targets.clear()

    assert config.target_kinds == (ReviewTargetKind.REPOSITORY,)
    assert type(config.target_kinds) is tuple
    assert config.to_json() == encoded
    assert config.digest() == digest
    assert config.run_id == run_id


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

    assert config.schema_version == "eval_run_config_v2"
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
    assert tuple(item.kind for item in config.evaluator.judge_profiles) == tuple(
        sorted(JudgeKind, key=lambda item: item.value)
    )
    assert all(
        item.judge_version == "judge-v1"
        for item in config.evaluator.judge_profiles
    )
    assert config.evaluator.profile(
        JudgeKind.INTENT_EQUIVALENCE
    ).rubric_version == "intent_equivalence-rubric-v7"
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


def test_judge_profiles_round_trip_sort_and_use_distinct_typed_contracts() -> None:
    reversed_profiles = tuple(reversed(tuple(judge_profile(kind) for kind in JudgeKind)))
    evaluator = evaluator_config(judge_profiles=reversed_profiles)
    canonical = evaluator_config()

    assert tuple(item.kind for item in evaluator.judge_profiles) == tuple(
        sorted(JudgeKind, key=lambda item: item.value)
    )
    assert len({item.rubric_digest for item in evaluator.judge_profiles}) == 4
    assert len({item.response_schema_version for item in evaluator.judge_profiles}) == 4
    assert len({item.response_schema_digest for item in evaluator.judge_profiles}) == 4
    assert evaluator.digest() == canonical.digest()
    for profile in evaluator.judge_profiles:
        assert JudgeProfileSnapshot.from_json(profile.to_json()) == profile
        assert profile.digest() == canonical_sha256(profile)
        assert profile.parameters["temperature"] == 0


def test_evaluator_requires_exactly_one_profile_per_judge_kind() -> None:
    profiles = list(evaluator_config().judge_profiles)

    with pytest.raises(SchemaError, match="each JudgeKind exactly once"):
        evaluator_config(judge_profiles=profiles[:-1])

    duplicate = profiles[:-1] + [profiles[0]]
    with pytest.raises(SchemaError, match="each JudgeKind exactly once"):
        evaluator_config(judge_profiles=duplicate)

    payload = evaluator_config().to_dict()
    payload["judge_profiles"][0]["kind"] = "unsupported_judge"
    with pytest.raises(SchemaError, match="unknown Judge kind"):
        EvaluatorRunConfig.from_dict(payload)


@pytest.mark.parametrize(
    "identity_field",
    [
        "rubric_id",
        "rubric_version",
        "rubric_digest",
        "response_schema_version",
        "response_schema_digest",
    ],
)
def test_evaluator_rejects_duplicate_rubric_or_schema_identity(
    identity_field: str,
) -> None:
    payload = evaluator_config().to_dict()
    payload["judge_profiles"][1][identity_field] = payload["judge_profiles"][0][
        identity_field
    ]

    with pytest.raises(SchemaError, match=f"distinct {identity_field}"):
        EvaluatorRunConfig.from_dict(payload)


def test_evaluator_v1_rejects_legacy_scalar_judge_schema() -> None:
    legacy = {
        "evaluator_id": "core-evaluator",
        "evaluator_version": "1.2.0",
        "grader_version": "grader-v4",
        "judge_version": "judge-v1",
        "model": "judge-model",
        "provider": "provider-j",
        "parameters": {},
        "rubric_version": "rubric-v7",
        "rubric_digest": "c" * 64,
    }
    with pytest.raises(SchemaError, match="unknown field"):
        EvaluatorRunConfig.from_dict(legacy)

    assert set(evaluator_config().to_dict()) == {
        "evaluator_id",
        "evaluator_version",
        "grader_version",
        "judge_profiles",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("judge_id", "changed-judge"),
        ("judge_version", "judge-v2"),
        ("adapter_id", "changed-adapter"),
        ("adapter_version", "adapter-v2"),
        ("adapter_config_digest", "1" * 64),
        ("provider", "provider-k"),
        ("model", "judge-model-v2"),
        ("model_artifact_digest", "2" * 64),
        ("parameters", {"temperature": 0.25}),
        ("system_prompt_version", "system-v2"),
        ("system_prompt_digest", "3" * 64),
        ("rubric_id", "changed-rubric"),
        ("rubric_version", "rubric-v8"),
        ("rubric_digest", "4" * 64),
        ("response_schema_version", "changed-response-v2"),
        ("response_schema_digest", "5" * 64),
        ("context_builder_version", "context-v2"),
        ("parser_version", "parser-v2"),
    ],
)
def test_every_judge_profile_identity_dimension_changes_evaluator_identity(
    field: str,
    value: object,
) -> None:
    original = evaluator_config()
    target = original.profile(JudgeKind.INTENT_EQUIVALENCE)
    changed_payload = target.to_dict()
    changed_payload[field] = value
    changed_profile = JudgeProfileSnapshot.from_dict(changed_payload)
    changed = evaluator_config(
        judge_profiles=tuple(
            changed_profile if item.kind is target.kind else item
            for item in original.judge_profiles
        )
    )

    assert changed.digest() != original.digest()


def test_judge_profile_strict_hydration_safe_json_and_deep_freeze() -> None:
    profile = judge_profile(JudgeKind.EVIDENCE_SUPPORT)
    payload = profile.to_dict()
    payload["unknown"] = True
    with pytest.raises(SchemaError, match="unknown field"):
        JudgeProfileSnapshot.from_dict(payload)

    payload = profile.to_dict()
    payload["parameters"] = {"client_secret": "ordinary-secret"}
    with pytest.raises(SchemaError) as caught:
        JudgeProfileSnapshot.from_dict(payload)
    assert "ordinary-secret" not in str(caught.value)

    nested_payload = profile.to_dict()
    nested_payload["parameters"] = {"nested": {"values": [1, 2]}}
    nested = JudgeProfileSnapshot.from_dict(nested_payload)
    with pytest.raises(TypeError):
        nested.parameters["nested"]["new"] = True
    assert nested.parameters["nested"]["values"] == (1, 2)

    payload = profile.to_dict()
    payload.pop("response_schema_digest")
    with pytest.raises(SchemaError, match="missing field"):
        JudgeProfileSnapshot.from_dict(payload)


def test_config_exports_only_final_protocol_names() -> None:
    assert AgentConfigSnapshot.__name__ == "AgentConfigSnapshot"
    assert JudgeProfileSnapshot.__name__ == "JudgeProfileSnapshot"
    assert JudgeExecutionBudgets.__name__ == "JudgeExecutionBudgets"
    assert config_module.SuiteCase is SuiteCase
    assert "JudgeKind" in config_module.__all__
    assert "JudgeProfileSnapshot" in config_module.__all__
    assert "JudgeExecutionBudgets" in config_module.__all__
    assert "AdapterCapabilitiesV2" in config_module.__all__
    assert "MAX_JUDGE_TOKEN_BUDGET" in config_module.__all__
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
    assert execution.cache_policy_version == DEFAULT_JUDGE_CACHE_POLICY_VERSION
    assert execution.judge_budgets.attempt_timeout_seconds <= (
        execution.evaluator_timeout_seconds
    )
    assert execution.judge_budgets.request_deadline_seconds <= (
        execution.evaluator_timeout_seconds
    )
    assert execution.judge_budgets.attempt_timeout_seconds <= (
        execution.judge_budgets.request_deadline_seconds
    )

    payload = execution.to_dict()
    payload["evaluator_config_digest"] = "0" * 64
    with pytest.raises(SchemaError, match="digest"):
        EvaluatorExecutionConfig.from_dict(payload)

    payload = execution.to_dict()
    payload.pop("judge_budgets")
    with pytest.raises(SchemaError, match="missing field"):
        EvaluatorExecutionConfig.from_dict(payload)

    with pytest.raises(SchemaError) as caught:
        EvaluatorExecutionConfig.from_resource_budgets(
            evaluator,
            resources,
            cache_policy_version="api_key=ordinary-secret",
        )
    assert "ordinary-secret" not in str(caught.value)


def test_judge_execution_budgets_round_trip_and_enforce_hierarchy() -> None:
    original = judge_budgets()
    assert original.max_reason_refs == 32
    assert JudgeExecutionBudgets.from_json(original.to_json()) == original

    invalid_mutations = (
        ("max_attempts_per_request", 0),
        ("attempt_timeout_seconds", 0),
        ("request_deadline_seconds", 0),
        (
            "attempt_timeout_seconds",
            original.request_deadline_seconds + 1,
        ),
        (
            "request_deadline_seconds",
            original.attempt_timeout_seconds
            * original.max_attempts_per_request
            - 1,
        ),
        ("max_parallel_requests", 0),
        ("max_context_blocks_per_request", 0),
        (
            "max_context_block_bytes",
            original.max_context_bytes_per_request + 1,
        ),
        (
            "max_context_bytes_per_request",
            original.max_model_request_bytes + 1,
        ),
        (
            "max_model_request_bytes",
            original.max_total_judge_request_bytes + 1,
        ),
        (
            "max_model_response_bytes",
            original.max_total_judge_response_bytes + 1,
        ),
        ("max_model_request_tokens", 0),
        ("max_model_response_tokens", True),
        (
            "max_model_request_tokens",
            original.max_total_judge_request_tokens + 1,
        ),
        (
            "max_model_response_tokens",
            original.max_total_judge_response_tokens + 1,
        ),
        (
            "max_reason_refs",
            original.max_context_blocks_per_request + 3,
        ),
        ("max_reason_refs", 33),
    )
    for field, value in invalid_mutations:
        payload = original.to_dict()
        payload[field] = value
        with pytest.raises(SchemaError):
            JudgeExecutionBudgets.from_dict(payload)

    payload = original.to_dict()
    payload["unknown"] = 1
    with pytest.raises(SchemaError, match="unknown field"):
        JudgeExecutionBudgets.from_dict(payload)

    duplicate = original.to_json().replace(
        '"max_attempts_per_request":2',
        '"max_attempts_per_request":2,"max_attempts_per_request":2',
        1,
    )
    with pytest.raises(SchemaError, match="duplicate"):
        JudgeExecutionBudgets.from_json(duplicate)

    with pytest.raises(FrozenInstanceError):
        original.max_model_response_tokens = 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_attempts_per_request", 1),
        ("attempt_timeout_seconds", 59),
        ("request_deadline_seconds", 121),
        ("max_parallel_requests", 3),
        ("max_context_blocks_per_request", 255),
        ("max_context_block_bytes", 500 * 1024),
        ("max_context_bytes_per_request", 5 * 1024 * 1024),
        ("max_model_request_bytes", 7 * 1024 * 1024),
        ("max_model_response_bytes", 900 * 1024),
        ("max_model_request_tokens", 120_000),
        ("max_model_response_tokens", 7_000),
        ("max_reason_refs", 31),
        ("max_total_judge_request_bytes", 63 * 1024 * 1024),
        ("max_total_judge_response_bytes", 15 * 1024 * 1024),
        ("max_total_judge_request_tokens", 1_000_000),
        ("max_total_judge_response_tokens", 60_000),
    ],
)
def test_every_judge_budget_dimension_changes_execution_identity(
    field: str,
    value: object,
) -> None:
    original_budgets = judge_budgets()
    changed_payload = original_budgets.to_dict()
    changed_payload[field] = value
    changed_budgets = JudgeExecutionBudgets.from_dict(changed_payload)
    evaluator = evaluator_config()
    resources = budgets()

    original = EvaluatorExecutionConfig.from_resource_budgets(
        evaluator,
        resources,
        judge_budgets=original_budgets,
    )
    changed = EvaluatorExecutionConfig.from_resource_budgets(
        evaluator,
        resources,
        judge_budgets=changed_budgets,
    )

    assert changed_budgets.digest() != original_budgets.digest()
    assert changed.digest() != original.digest()


def test_execution_cross_validates_judge_and_artifact_budgets() -> None:
    evaluator = evaluator_config()
    base = judge_budgets()

    with pytest.raises(SchemaError, match="attempt timeout"):
        EvaluatorExecutionConfig.create(
            evaluator=evaluator,
            evaluator_timeout_seconds=base.attempt_timeout_seconds - 1,
            max_execution_artifact_file_bytes=16 * 1024 * 1024,
            max_execution_artifact_total_bytes=128 * 1024 * 1024,
            judge_budgets=base,
        )

    deadline_payload = base.to_dict()
    deadline_payload["request_deadline_seconds"] = 301
    with pytest.raises(SchemaError, match="request deadline"):
        EvaluatorExecutionConfig.create(
            evaluator=evaluator,
            evaluator_timeout_seconds=300,
            max_execution_artifact_file_bytes=16 * 1024 * 1024,
            max_execution_artifact_total_bytes=128 * 1024 * 1024,
            judge_budgets=JudgeExecutionBudgets.from_dict(deadline_payload),
        )

    with pytest.raises(SchemaError, match="model request bytes"):
        EvaluatorExecutionConfig.create(
            evaluator=evaluator,
            evaluator_timeout_seconds=300,
            max_execution_artifact_file_bytes=base.max_model_request_bytes - 1,
            max_execution_artifact_total_bytes=128 * 1024 * 1024,
            judge_budgets=base,
        )

    with pytest.raises(SchemaError, match="total Judge request bytes"):
        EvaluatorExecutionConfig.create(
            evaluator=evaluator,
            evaluator_timeout_seconds=300,
            max_execution_artifact_file_bytes=16 * 1024 * 1024,
            max_execution_artifact_total_bytes=(
                base.max_total_judge_request_bytes - 1
            ),
            judge_budgets=base,
        )


def test_budget_and_cache_policy_change_execution_digest_and_evaluation_id(
    case_snapshot: RunCaseSnapshot,
) -> None:
    config = run_config(case_snapshot)
    original = EvaluatorExecutionConfig.from_resource_budgets(
        config.evaluator,
        config.resource_budgets,
    )
    changed_budget_payload = original.judge_budgets.to_dict()
    changed_budget_payload["max_attempts_per_request"] -= 1
    changed_budget = EvaluatorExecutionConfig.from_resource_budgets(
        config.evaluator,
        config.resource_budgets,
        judge_budgets=JudgeExecutionBudgets.from_dict(changed_budget_payload),
    )
    changed_cache = EvaluatorExecutionConfig.from_resource_budgets(
        config.evaluator,
        config.resource_budgets,
        cache_policy_version="semantic-judge-cache-v2",
    )
    changed_context = EvaluatorExecutionConfig.from_resource_budgets(
        config.evaluator,
        config.resource_budgets,
        review_evaluator_context_policy_version="truth-scoped-context-v3",
    )
    changed_authority = EvaluatorExecutionConfig.from_resource_budgets(
        config.evaluator,
        config.resource_budgets,
        metric_authority_policy_version="metric-authority-v3",
    )

    assert original.digest() != changed_budget.digest()
    assert original.digest() != changed_cache.digest()
    assert original.digest() != changed_context.digest()
    assert original.digest() != changed_authority.digest()
    original_id = derive_evaluation_id(config.run_id, original.digest(), "v1")
    assert original_id != derive_evaluation_id(
        config.run_id, changed_budget.digest(), "v1"
    )
    assert original_id != derive_evaluation_id(
        config.run_id, changed_cache.digest(), "v1"
    )
    assert original_id != derive_evaluation_id(
        config.run_id, changed_context.digest(), "v1"
    )
    assert original_id != derive_evaluation_id(
        config.run_id, changed_authority.digest(), "v1"
    )


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


_ENV_LIKE_CODE_CONTEXT = (
    "count = len(items)\n"
    "affinity = score(item)"
)


_REVIEW_EVALUATION_ROOT_FIELDS = (
    "schema_version",
    "evaluator_revision",
    "evaluator_execution_digest",
    "submission_digest",
    "submission_review_digest",
    "submission_evidence_digest",
    "eval_input_digest",
    "review_truth_digest",
    "deterministic_context_digest",
    "review_policy_version",
    "assignment_policy_version",
    "location_policy_version",
    "evidence_integrity_policy_version",
    "truth_completeness",
    "novel_finding_policy",
    "status",
    "phase",
    "generated_findings",
    "expected_truth_findings",
    "known_invalid_truth_findings",
    "location_candidates",
    "known_invalid_candidates",
    "expected_candidates",
    "assignments",
    "finding_outcomes",
    "unmatched_expected_truth_ids",
    "judge_requests",
    "judge_decisions",
    "judge_failures",
    "judge_ungraded",
    "evidence_integrity_results",
    "coverage",
    "metrics",
    "reason_codes",
    "limit_failure",
)


def _context_block(
    content: Any,
    *,
    model_payload: bool = False,
    content_digest: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "ref_id": "context-1",
        "kind": "code",
        ("data_boundary" if model_payload else "trust"): (
            "untrusted_repository_data"
        ),
        "content": content,
        "metadata": {},
        "content_digest": content_digest or canonical_sha256(content),
    }


def _review_context_payload(content: Any) -> Dict[str, Any]:
    payload = dict.fromkeys(_REVIEW_EVALUATION_ROOT_FIELDS)
    payload["schema_version"] = "eval_review_evaluation_v1"
    payload["judge_requests"] = [
        {"request": {"contexts": [_context_block(content)]}}
    ]
    return payload


def _judge_input_context_payload(content: Any) -> Dict[str, Any]:
    return {
        "schema_version": "eval_judge_input_artifact_v1",
        "evaluator_execution_digest": "0" * 64,
        "requests": [{"contexts": [_context_block(content)]}],
    }


def _judge_output_request_context_payload(content: Any) -> Dict[str, Any]:
    return {
        "schema_version": "eval_judge_output_artifact_v1",
        "evaluator_execution_digest": "0" * 64,
        "input_artifact_digest": "1" * 64,
        "intent_evaluation_digest": None,
        "results": [{"request": {"contexts": [_context_block(content)]}}],
    }


def _model_turn_context_block_payload(content: Any) -> Dict[str, Any]:
    return {
        "schema_version": "eval_judge_output_artifact_v1",
        "evaluator_execution_digest": "0" * 64,
        "input_artifact_digest": "1" * 64,
        "intent_evaluation_digest": None,
        "results": [
            {},
            {
                "model_turn": {
                    "messages": [
                        {
                            "content": {
                                "context_blocks": [
                                    _context_block(content, model_payload=True)
                                ],
                            }
                        }
                    ]
                }
            },
        ],
    }


@pytest.mark.parametrize(
    ("policy", "payload_factory"),
    [
        ("review_matches", _review_context_payload),
        ("judge_input", _judge_input_context_payload),
        ("judge_output", _judge_output_request_context_payload),
        ("judge_output", _model_turn_context_block_payload),
    ],
)
def test_safe_json_typed_evaluator_context_is_artifact_and_path_scoped(
    policy: str,
    payload_factory: Any,
) -> None:
    payload = payload_factory(_ENV_LIKE_CODE_CONTEXT)

    with pytest.raises(SchemaError, match="full environment dump"):
        validate_safe_json(payload)

    validate_safe_json(payload, evaluator_context_policy=policy)


def test_evaluator_context_policy_rejects_review_bypass_and_invalid_bindings() -> None:
    with pytest.raises(SchemaError):
        validate_safe_json(
            {"safe": "ordinary"},
            evaluator_context_policy="judge_input",
        )

    with pytest.raises(SchemaError, match="full environment dump"):
        validate_safe_json(
            {
                "not_a_judge_schema": {
                    "contexts": [{"content": _ENV_LIKE_CODE_CONTEXT}]
                }
            },
            evaluator_context_policy="review_matches",
        )

    wrong_schema = _judge_input_context_payload(_ENV_LIKE_CODE_CONTEXT)
    wrong_schema["schema_version"] = "unknown"
    with pytest.raises(SchemaError, match="full environment dump"):
        validate_safe_json(
            wrong_schema,
            evaluator_context_policy="judge_input",
        )

    wrong_root = _judge_input_context_payload(_ENV_LIKE_CODE_CONTEXT)
    wrong_root["unexpected"] = True
    with pytest.raises(SchemaError, match="full environment dump"):
        validate_safe_json(
            wrong_root,
            evaluator_context_policy="judge_input",
        )

    wrong_path = _judge_input_context_payload("ordinary")
    wrong_path["requests"] = [
        {"not_contexts": [_context_block(_ENV_LIKE_CODE_CONTEXT)]}
    ]
    with pytest.raises(SchemaError, match="full environment dump"):
        validate_safe_json(
            wrong_path,
            evaluator_context_policy="judge_input",
        )

    wrong_digest = _judge_input_context_payload("ordinary context")
    wrong_digest["requests"][0]["contexts"][0]["content_digest"] = "f" * 64
    with pytest.raises(SchemaError):
        validate_safe_json(
            wrong_digest,
            evaluator_context_policy="judge_input",
        )


@pytest.mark.parametrize(
    "content",
    [
        "count = len(items)\napi_key=ordinary-secret",
        "<think>private intermediate steps</think>",
    ],
)
def test_typed_context_block_content_still_rejects_sensitive_text(
    content: str,
) -> None:
    with pytest.raises(SchemaError):
        validate_safe_json(
            _model_turn_context_block_payload(content),
            evaluator_context_policy="judge_output",
        )


@pytest.mark.parametrize(
    "content",
    [
        "FIRST_VALUE = compute_value()\napi_key=ordinary-secret",
        "<think>private intermediate steps</think>",
        "https://user:password@example.test/context",
        {
            "HOME": "/private",
            "PATH": "/bin",
            "USER": "private-user",
        },
        {"client_secret": "ordinary-secret"},
    ],
)
def test_typed_evaluator_context_never_allows_other_unsafe_classes(
    content: Any,
) -> None:
    with pytest.raises(SchemaError):
        validate_safe_json(
            _judge_input_context_payload(content),
            evaluator_context_policy="judge_input",
        )


def test_strict_hydration_rejects_unknown_duplicate_and_digest_mismatch(
    case_snapshot: RunCaseSnapshot,
) -> None:
    config = run_config(case_snapshot)
    payload = config.to_dict()
    payload["unknown"] = True
    with pytest.raises(SchemaError, match="unknown field"):
        EvalRunConfig.from_dict(payload)

    duplicate = config.to_json().replace(
        '"schema_version":"eval_run_config_v2"',
        '"schema_version":"eval_run_config_v2","schema_version":"eval_run_config_v2"',
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
            adapter_capabilities=adapter_capabilities(),
            trial_count=1,
            resource_budgets=budgets(parallel=2),
        )


def test_v2_run_config_rejects_v1_roots_and_children_before_deeper_hydration(
    case_snapshot: RunCaseSnapshot,
) -> None:
    config = run_config(case_snapshot)

    v1_root = config.to_dict()
    v1_root["schema_version"] = "eval_run_config_v1"
    v1_root["legacy_unknown"] = True
    with pytest.raises(UnsupportedProtocolVersionError):
        EvalRunConfig.from_dict(v1_root)

    v1_capability = config.to_dict()
    v1_capability["adapter_capabilities"]["schema_version"] = (
        "eval_adapter_capabilities_v1"
    )
    v1_capability["agent_config_digest"] = "malformed"
    with pytest.raises(UnsupportedProtocolVersionError):
        EvalRunConfig.from_dict(v1_capability)

    execution = EvaluatorExecutionConfig.from_resource_budgets(
        config.evaluator,
        config.resource_budgets,
    )
    v1_execution = execution.to_dict()
    v1_execution["schema_version"] = "eval_evaluator_execution_config_v1"
    v1_execution["legacy_unknown"] = True
    v1_execution["evaluator"] = "malformed-but-deeper"
    with pytest.raises(UnsupportedProtocolVersionError):
        EvaluatorExecutionConfig.from_dict(v1_execution)


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

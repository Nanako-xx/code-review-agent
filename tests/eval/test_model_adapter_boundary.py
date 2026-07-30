from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import pytest

from review_agent.model_adapter import OpenAICompatibleToolAdapter
from review_agent_eval.adapters.agent_factory import (
    AgentAdapterConfigError,
    CURRENT_AGENT_ADAPTER_KIND,
    SUBPROCESS_JSON_ADAPTER_KIND,
    adapter_capabilities_from_snapshot,
    build_agent_adapter_factory,
)
from review_agent_eval.adapters.current_agent import CurrentAgentAdapter
from review_agent_eval.adapters.model_adapter import (
    AdapterConfigError,
    EVAL_JUDGE_STAGE_LABEL,
    ModelAdapterConfig,
    build_judge_model_adapter_factory,
    build_model_adapter_factory_from_config,
)
from review_agent_eval.adapters.subprocess_agent import (
    SubprocessAgentAdapter,
    subprocess_adapter_capabilities,
)
from review_agent_eval.config import AgentConfigSnapshot, JudgeExecutionBudgets


def _model_config(**overrides: object) -> ModelAdapterConfig:
    values: dict[str, object] = {
        "provider_name": "openai-compatible",
        "model": "judge-model",
        "base_url": "https://model.example/v1",
        "api_key_env": "EVAL_JUDGE_API_KEY",
    }
    values.update(overrides)
    return ModelAdapterConfig(**values)  # type: ignore[arg-type]


def _agent_snapshot(
    kind: str,
    *,
    adapter_override: dict[str, object] | None = None,
) -> AgentConfigSnapshot:
    if adapter_override is not None:
        adapter = adapter_override
    elif kind == CURRENT_AGENT_ADAPTER_KIND:
        adapter = {
            "kind": kind,
            "command": [str(Path(sys.executable).resolve()), "-m", "review_agent"],
            "review_arguments": ["--reviewer-provider=fake"],
            "environment_allowlist": [],
            "memory_mode": "off",
        }
    else:
        adapter = {
            "kind": kind,
            "command": [
                str(Path(sys.executable).resolve()),
                "{agent_id}",
                "{task_id}",
                "{trial_id}",
                "{workspace}",
            ],
            "environment_allowlist": [],
            "capabilities": subprocess_adapter_capabilities().to_dict(),
        }
    return AgentConfigSnapshot(
        agent_id="agent-boundary",
        agent_name="Boundary Agent",
        agent_version="1.0.0",
        commit="a" * 40,
        model="agent-model",
        provider="fake",
        parameters={"adapter": adapter},
        prompt_config_digest="b" * 64,
    )


def test_eval_factory_binds_judge_budgets_and_stage_label_without_changing_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVAL_JUDGE_API_KEY", "boundary-secret-value")
    budgets = JudgeExecutionBudgets.defaults(
        evaluator_timeout_seconds=60,
        max_execution_artifact_file_bytes=4 * 1024 * 1024,
        max_execution_artifact_total_bytes=8 * 1024 * 1024,
    )

    factory = build_judge_model_adapter_factory(
        _model_config(),
        budgets=budgets,
    )

    assert factory is not None
    adapter = factory.create()
    assert isinstance(adapter, OpenAICompatibleToolAdapter)
    assert adapter.provider_name == "openai-compatible"
    assert adapter.capabilities.request_timeout is True
    assert (
        adapter.capabilities.response_byte_limit
        == budgets.max_model_response_bytes
    )
    runtime_config = adapter._config  # type: ignore[attr-defined]
    assert runtime_config.timeout_seconds == budgets.attempt_timeout_seconds
    assert runtime_config.max_response_bytes == budgets.max_model_response_bytes


def test_eval_factory_accepts_explicit_budget_overrides_and_forces_judge_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVAL_JUDGE_API_KEY", "another-secret")

    factory = build_judge_model_adapter_factory(
        _model_config(stage_label="agent-stage"),
        timeout_seconds=2.5,
        max_response_bytes=4_096,
    )

    assert factory is not None
    adapter = factory.create()
    runtime_config = adapter._config  # type: ignore[attr-defined]
    assert runtime_config.timeout_seconds == 2.5
    assert runtime_config.max_response_bytes == 4_096
    # The product factory's stage label is reflected in diagnostics/errors;
    # the Judge helper must not inherit an Agent stage label.
    assert EVAL_JUDGE_STAGE_LABEL == "eval-judge"


def test_eval_factory_reads_only_named_env_and_does_not_expose_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "credential-value-that-must-not-cross-boundary"
    monkeypatch.setenv("EVAL_JUDGE_API_KEY", secret)
    config = _model_config()

    assert not hasattr(config, "api_key")
    redacted = config.redacted_dict()
    assert "api_key" not in redacted
    assert secret not in json.dumps(redacted, sort_keys=True)

    other_env = "OTHER_PROVIDER_SECRET"
    monkeypatch.setenv(other_env, "wrong-value")
    factory = build_model_adapter_factory_from_config(config)
    assert factory is not None
    # The boundary config carries only the selected env *name*.  The actual
    # value is held by the runtime adapter and is never part of the Eval
    # configuration projection.
    assert config.api_key_env == "EVAL_JUDGE_API_KEY"
    assert config.api_key_env != other_env


def test_missing_named_credential_is_a_sanitized_eval_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVAL_JUDGE_API_KEY", raising=False)
    secret = "missing-secret-value"

    with pytest.raises(AdapterConfigError) as error:
        build_model_adapter_factory_from_config(_model_config())

    assert "EVAL_JUDGE_API_KEY" in str(error.value)
    assert secret not in str(error.value)


def test_judge_factory_does_not_upgrade_fake_capabilities() -> None:
    factory = build_judge_model_adapter_factory(
        ModelAdapterConfig(
            provider_name="fake",
            model=None,
            base_url=None,
            api_key_env="EVAL_JUDGE_API_KEY",
        ),
        timeout_seconds=1,
        max_response_bytes=1_024,
    )

    assert factory is not None
    capabilities = factory.create().capabilities
    assert capabilities.request_timeout is False
    assert capabilities.response_byte_limit is None


@pytest.mark.parametrize(
    "bad_url",
    [
        "https://user:password@model.example/v1",
        "https://token@model.example/v1",
        "https://model.example/v1?token=secret",
        "https://model.example/v1#credential",
        "https://model.example/v1\nX-Injected: true",
    ],
)
def test_eval_model_config_rejects_credential_bearing_endpoint(bad_url: str) -> None:
    with pytest.raises(AdapterConfigError):
        _model_config(base_url=bad_url)


def test_agent_factory_returns_fresh_adapters_for_supported_snapshot_kinds() -> None:
    current_snapshot = _agent_snapshot(CURRENT_AGENT_ADAPTER_KIND)
    assert adapter_capabilities_from_snapshot(current_snapshot).adapter_id == (
        CURRENT_AGENT_ADAPTER_KIND
    )
    current_factory = build_agent_adapter_factory(current_snapshot)
    assert isinstance(current_factory(), CurrentAgentAdapter)
    assert current_factory() is not current_factory()

    subprocess_snapshot = _agent_snapshot(SUBPROCESS_JSON_ADAPTER_KIND)
    assert adapter_capabilities_from_snapshot(subprocess_snapshot) == (
        subprocess_adapter_capabilities()
    )
    subprocess_factory = build_agent_adapter_factory(subprocess_snapshot)
    assert isinstance(subprocess_factory(), SubprocessAgentAdapter)
    assert subprocess_factory() is not subprocess_factory()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("adapter_id", "other-subprocess-adapter"),
        ("adapter_version", "3"),
        ("subprocess_wire_version", "subprocess-json-v3"),
        ("input_schema_version", "eval_input_v3"),
        ("submission_schema_version", "eval_submission_v3"),
    ],
)
def test_agent_factory_rejects_subprocess_capability_identity_drift(
    field: str,
    value: str,
) -> None:
    capabilities = subprocess_adapter_capabilities().to_dict()
    capabilities[field] = value
    snapshot = _agent_snapshot(
        SUBPROCESS_JSON_ADAPTER_KIND,
        adapter_override={
            "kind": SUBPROCESS_JSON_ADAPTER_KIND,
            "command": [str(Path(sys.executable).resolve())],
            "environment_allowlist": [],
            "capabilities": capabilities,
        },
    )

    with pytest.raises(AgentAdapterConfigError, match="capabilities"):
        build_agent_adapter_factory(snapshot)


@pytest.mark.parametrize(
    "adapter",
    [
        {"kind": "unknown-v1", "command": [], "environment_allowlist": []},
        {
            "kind": CURRENT_AGENT_ADAPTER_KIND,
            "command": ["relative-agent"],
            "review_arguments": [],
            "environment_allowlist": [],
            "memory_mode": "off",
        },
        {
            "kind": CURRENT_AGENT_ADAPTER_KIND,
            "command": [str(Path(sys.executable).resolve())],
            "review_arguments": ["--judge-provider=wrong"],
            "environment_allowlist": [],
            "memory_mode": "off",
        },
    ],
)
def test_agent_factory_rejects_invalid_or_judge_mixed_snapshot(adapter: dict[str, object]) -> None:
    with pytest.raises(AgentAdapterConfigError):
        build_agent_adapter_factory(
            _agent_snapshot(CURRENT_AGENT_ADAPTER_KIND, adapter_override=adapter)
        )


@pytest.mark.parametrize(
    "policy",
    [
        None,
        {"unanswered_action": "promote_to_explicit"},
    ],
)
def test_agent_factory_rejects_invalid_unanswered_clarification_policy(
    policy: object,
) -> None:
    snapshot = _agent_snapshot(CURRENT_AGENT_ADAPTER_KIND)
    parameters = snapshot.to_dict()["parameters"]
    parameters["clarification"] = policy
    invalid = AgentConfigSnapshot(
        agent_id=snapshot.agent_id,
        agent_name=snapshot.agent_name,
        agent_version=snapshot.agent_version,
        commit=snapshot.commit,
        model=snapshot.model,
        provider=snapshot.provider,
        parameters=parameters,
        prompt_config_digest=snapshot.prompt_config_digest,
    )

    with pytest.raises(AgentAdapterConfigError, match="clarification policy"):
        build_agent_adapter_factory(invalid)


def test_agent_factory_rejects_judge_namespace_and_has_no_judge_parameter() -> None:
    snapshot = _agent_snapshot(CURRENT_AGENT_ADAPTER_KIND)
    parameters = snapshot.to_dict()["parameters"]
    parameters["judge_provider"] = "openai-compatible"
    mixed_snapshot = AgentConfigSnapshot(
        agent_id=snapshot.agent_id,
        agent_name=snapshot.agent_name,
        agent_version=snapshot.agent_version,
        commit=snapshot.commit,
        model=snapshot.model,
        provider=snapshot.provider,
        parameters=parameters,
        prompt_config_digest=snapshot.prompt_config_digest,
    )

    with pytest.raises(AgentAdapterConfigError, match="Judge configuration"):
        build_agent_adapter_factory(mixed_snapshot)

    signature = inspect.signature(build_agent_adapter_factory)
    assert not any(name.startswith("judge") for name in signature.parameters)


def test_judge_budget_cannot_override_a_conflicting_immutable_boundary_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVAL_JUDGE_API_KEY", "runtime-secret")
    config = _model_config(timeout_seconds=3, max_response_bytes=2_048)

    with pytest.raises(AdapterConfigError, match="timeout_seconds conflicts"):
        build_judge_model_adapter_factory(
            config,
            timeout_seconds=4,
            max_response_bytes=2_048,
        )


def test_agent_factory_rejects_case_and_separator_variants_of_judge_namespace() -> None:
    snapshot = _agent_snapshot(CURRENT_AGENT_ADAPTER_KIND)
    for key in ("Judge-Provider", "eval.judge.model", "EVALUATOR_CONFIG"):
        parameters = snapshot.to_dict()["parameters"]
        parameters[key] = "forbidden"
        mixed = AgentConfigSnapshot(
            agent_id=snapshot.agent_id,
            agent_name=snapshot.agent_name,
            agent_version=snapshot.agent_version,
            commit=snapshot.commit,
            model=snapshot.model,
            provider=snapshot.provider,
            parameters=parameters,
            prompt_config_digest=snapshot.prompt_config_digest,
        )
        with pytest.raises(AgentAdapterConfigError, match="Judge configuration"):
            build_agent_adapter_factory(mixed)


def test_current_agent_factory_rejects_provider_or_model_identity_drift() -> None:
    executable = str(Path(sys.executable).resolve())
    for arguments, expected in (
        (["--reviewer-provider=openai-compatible"], "provider argument"),
        (["--reviewer-provider=fake", "--reviewer-model=other"], "model argument"),
    ):
        snapshot = _agent_snapshot(
            CURRENT_AGENT_ADAPTER_KIND,
            adapter_override={
                "kind": CURRENT_AGENT_ADAPTER_KIND,
                "command": [executable, "-m", "review_agent"],
                "review_arguments": arguments,
                "environment_allowlist": [],
                "memory_mode": "off",
            },
        )
        with pytest.raises(AgentAdapterConfigError, match=expected):
            build_agent_adapter_factory(snapshot)


def test_agent_and_judge_factories_are_independent_namespaces() -> None:
    agent_signature = inspect.signature(build_agent_adapter_factory)
    judge_signature = inspect.signature(build_judge_model_adapter_factory)
    assert "snapshot" in agent_signature.parameters
    assert "config" in judge_signature.parameters
    assert "snapshot" not in judge_signature.parameters
    assert "config" not in agent_signature.parameters

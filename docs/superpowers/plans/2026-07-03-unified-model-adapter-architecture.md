# Unified Model Adapter Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ModelAdapter` the single reviewer model interface for both single-shot and agent-loop reviewer execution.

**Architecture:** Move shared model response/config ownership out of the legacy provider path, introduce a `ModelAdapterFactory`, then migrate reviewer, orchestrator, and CLI paths so business logic depends only on `ModelAdapter`. Keep `provider.py` as a compatibility shim until a later removal pass, but do not let the reviewer main path call `ModelProvider`.

**Tech Stack:** Python dataclasses, protocols, stdlib `json`/`os`/`urllib`, existing `ModelTurnRequest` / `ModelTurnResponse`, existing `ToolGateway`, `pytest`.

---

## Scope

This plan implements the final B architecture from:

`docs/superpowers/specs/2026-07-03-unified-model-adapter-architecture-design.md`

The result must satisfy:

- `single-shot` reviewer execution uses `ModelAdapter`.
- `agent-loop` reviewer execution uses `ModelAdapter`.
- multi-reviewer orchestration receives an adapter factory, not a provider instance.
- CLI reviewer main path builds a `ModelAdapterFactory`, not a `ModelProvider`.
- `--reviewer-provider openai-compatible --reviewer-loop agent-loop` is allowed.
- `provider.py` can remain for legacy compatibility, but reviewer business modules should not import `ModelProvider`.

## File structure

- `src/review_agent/model_protocol.py`
  - Add shared `ModelResponse`.

- `src/review_agent/model_adapter.py`
  - Own `OpenAICompatibleConfig`, `Transport`, and `_urllib_transport`.
  - Keep `FakeToolCallingAdapter` and `OpenAICompatibleToolAdapter`.

- `src/review_agent/model_adapter_factory.py`
  - New factory module.
  - Creates fresh adapters for fake and openai-compatible providers.

- `src/review_agent/reviewer.py`
  - Make `run_single_reviewer()` consume `ModelAdapter`.
  - Convert envelope to `ModelTurnRequest`.
  - Convert adapter response to `ModelResponse`.

- `src/review_agent/orchestrator.py`
  - Make `run_multi_reviewer()` consume `ModelAdapterFactory`.
  - Create a fresh adapter for each reviewer.

- `src/review_agent/cli.py`
  - Build adapter factory once.
  - Use factory for single-shot and agent-loop paths.
  - Remove the fake-only agent-loop guard.

- `src/review_agent/provider.py`
  - Legacy compatibility only.
  - Re-export shared config/response types as needed.

- Tests:
  - `tests/test_model_adapter.py`
  - `tests/test_model_adapter_factory.py`
  - `tests/test_reviewer.py`
  - `tests/test_orchestrator.py`
  - `tests/test_cli_smoke.py`
  - `tests/test_provider.py`

---

## Task 1: Move shared response and OpenAI-compatible config ownership into adapter/protocol modules

**Files:**

- Modify: `src/review_agent/model_protocol.py`
- Modify: `src/review_agent/model_adapter.py`
- Modify: `src/review_agent/provider.py`
- Modify: `tests/test_model_adapter.py`
- Modify: `tests/test_provider.py`

- [ ] **Step 1: Write failing import ownership tests**

Update imports in `tests/test_model_adapter.py`:

```python
from review_agent.model_adapter import (
    FakeToolCallingAdapter,
    OpenAICompatibleConfig,
    OpenAICompatibleToolAdapter,
)
from review_agent.model_protocol import (
    ModelResponse,
    ModelResponseKind,
    ModelToolCall,
    ModelToolSpec,
    ModelTurnRequest,
    ModelTurnResponse,
)
```

Add this test to `tests/test_model_adapter.py`:

```python
def test_model_response_lives_in_model_protocol():
    response = ModelResponse(
        content='{"status": "completed"}',
        provider_name="adapter",
        model="review-model",
        raw={"trace_id": "trace-1"},
    )

    assert response.provider_name == "adapter"
    assert response.raw["trace_id"] == "trace-1"
```

Update `tests/test_provider.py` imports so `OpenAICompatibleConfig` and `ModelResponse` come from the new owner modules:

```python
from review_agent.model_adapter import OpenAICompatibleConfig
from review_agent.model_protocol import ModelResponse
from review_agent.provider import (
    FakeProvider,
    ModelProviderError,
    OpenAICompatibleProvider,
    ProviderConfigError,
    build_provider_from_config,
)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -B -m pytest tests/test_model_adapter.py tests/test_provider.py -q -p no:cacheprovider
```

Expected: fail because `ModelResponse` is not exported by `model_protocol.py`, and `OpenAICompatibleConfig` is not exported by `model_adapter.py`.

- [ ] **Step 3: Add `ModelResponse` to `model_protocol.py`**

In `src/review_agent/model_protocol.py`, add:

```python
@dataclass(frozen=True)
class ModelResponse:
    content: str
    provider_name: str
    model: str
    raw: dict[str, Any] = field(default_factory=dict)
```

Place it after `ModelTurnResponse`. `Any` and `field` are already imported in that module.

- [ ] **Step 4: Move OpenAI-compatible config and transport into `model_adapter.py`**

In `src/review_agent/model_adapter.py`, remove this import:

```python
from review_agent.provider import OpenAICompatibleConfig, Transport, _urllib_transport
```

Add these definitions above `OpenAICompatibleToolAdapter`:

```python
@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 60


Transport = Callable[[str, dict[str, str], dict[str, Any], int], dict[str, Any]]


def _urllib_transport(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))
```

Also update imports at the top of `model_adapter.py`:

```python
from dataclasses import dataclass
import urllib.request
```

- [ ] **Step 5: Make `provider.py` a compatibility consumer of adapter-owned types**

In `src/review_agent/provider.py`, remove local definitions of:

```python
@dataclass(frozen=True)
class ModelResponse: ...

@dataclass(frozen=True)
class OpenAICompatibleConfig: ...

Transport = ...

def _urllib_transport(...): ...
```

Import them instead:

```python
from review_agent.model_adapter import OpenAICompatibleConfig, Transport, _urllib_transport
from review_agent.model_protocol import ModelResponse
```

Keep `ModelProvider`, `FakeProvider`, `OpenAICompatibleProvider`, `ProviderConfigError`, and `build_provider_from_config()` in `provider.py` for legacy tests.

- [ ] **Step 6: Run tests and verify pass**

Run:

```powershell
python -B -m pytest tests/test_model_adapter.py tests/test_provider.py -q -p no:cacheprovider
```

Expected: tests pass.

- [ ] **Step 7: Commit Task 1**

Run:

```powershell
git add src/review_agent/model_protocol.py src/review_agent/model_adapter.py src/review_agent/provider.py tests/test_model_adapter.py tests/test_provider.py
git commit -m "refactor: move model response and adapter config ownership"
```

---

## Task 2: Add ModelAdapterFactory

**Files:**

- Create: `src/review_agent/model_adapter_factory.py`
- Create: `tests/test_model_adapter_factory.py`

- [ ] **Step 1: Write failing factory tests**

Create `tests/test_model_adapter_factory.py`:

```python
import pytest

from review_agent.model_adapter import FakeToolCallingAdapter, OpenAICompatibleToolAdapter
from review_agent.model_adapter_factory import (
    AdapterConfigError,
    ModelAdapterConfig,
    build_model_adapter_factory_from_config,
)


def test_factory_returns_none_for_provider_none():
    factory = build_model_adapter_factory_from_config(
        ModelAdapterConfig(
            provider_name="none",
            model=None,
            base_url=None,
            api_key_env="REVIEW_AGENT_API_KEY",
        )
    )

    assert factory is None


def test_factory_creates_fresh_fake_adapters():
    factory = build_model_adapter_factory_from_config(
        ModelAdapterConfig(
            provider_name="fake",
            model=None,
            base_url=None,
            api_key_env="REVIEW_AGENT_API_KEY",
        )
    )

    first = factory.create()
    second = factory.create()

    assert isinstance(first, FakeToolCallingAdapter)
    assert isinstance(second, FakeToolCallingAdapter)
    assert first is not second


def test_factory_creates_openai_compatible_adapter(monkeypatch):
    monkeypatch.setenv("REVIEW_AGENT_API_KEY", "secret-key")

    factory = build_model_adapter_factory_from_config(
        ModelAdapterConfig(
            provider_name="openai-compatible",
            model="review-model",
            base_url="https://example.test/v1",
            api_key_env="REVIEW_AGENT_API_KEY",
        )
    )

    adapter = factory.create()

    assert isinstance(adapter, OpenAICompatibleToolAdapter)
    assert adapter.provider_name == "openai-compatible"


def test_factory_rejects_missing_openai_api_key(monkeypatch):
    monkeypatch.delenv("REVIEW_AGENT_API_KEY", raising=False)

    with pytest.raises(AdapterConfigError, match="REVIEW_AGENT_API_KEY"):
        build_model_adapter_factory_from_config(
            ModelAdapterConfig(
                provider_name="openai-compatible",
                model="review-model",
                base_url="https://example.test/v1",
                api_key_env="REVIEW_AGENT_API_KEY",
            )
        )


def test_factory_rejects_missing_openai_model(monkeypatch):
    monkeypatch.setenv("REVIEW_AGENT_API_KEY", "secret-key")

    with pytest.raises(AdapterConfigError, match="--reviewer-model"):
        build_model_adapter_factory_from_config(
            ModelAdapterConfig(
                provider_name="openai-compatible",
                model=None,
                base_url="https://example.test/v1",
                api_key_env="REVIEW_AGENT_API_KEY",
            )
        )


def test_factory_rejects_missing_openai_base_url(monkeypatch):
    monkeypatch.setenv("REVIEW_AGENT_API_KEY", "secret-key")

    with pytest.raises(AdapterConfigError, match="--reviewer-base-url"):
        build_model_adapter_factory_from_config(
            ModelAdapterConfig(
                provider_name="openai-compatible",
                model="review-model",
                base_url=None,
                api_key_env="REVIEW_AGENT_API_KEY",
            )
        )
```

- [ ] **Step 2: Run factory tests and verify failure**

Run:

```powershell
python -B -m pytest tests/test_model_adapter_factory.py -q -p no:cacheprovider
```

Expected: fail because `review_agent.model_adapter_factory` does not exist.

- [ ] **Step 3: Implement factory module**

Create `src/review_agent/model_adapter_factory.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Protocol

from review_agent.model_adapter import (
    FakeToolCallingAdapter,
    ModelAdapter,
    OpenAICompatibleConfig,
    OpenAICompatibleToolAdapter,
)
from review_agent.model_protocol import ModelResponseKind, ModelToolCall, ModelTurnRequest, ModelTurnResponse


class AdapterConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelAdapterConfig:
    provider_name: str | None
    model: str | None
    base_url: str | None
    api_key_env: str


class ModelAdapterFactory(Protocol):
    def create(self) -> ModelAdapter:
        raise NotImplementedError


@dataclass(frozen=True)
class FakeModelAdapterFactory:
    def create(self) -> ModelAdapter:
        return _fake_single_shot_adapter()


@dataclass(frozen=True)
class OpenAICompatibleModelAdapterFactory:
    config: OpenAICompatibleConfig

    def create(self) -> ModelAdapter:
        return OpenAICompatibleToolAdapter(self.config)


def build_model_adapter_factory_from_config(
    config: ModelAdapterConfig,
) -> ModelAdapterFactory | None:
    provider_name = config.provider_name or "none"
    if provider_name == "none":
        return None
    if provider_name == "fake":
        return FakeModelAdapterFactory()
    if provider_name == "openai-compatible":
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise AdapterConfigError(f"missing API key environment variable: {config.api_key_env}")
        if not config.model:
            raise AdapterConfigError("--reviewer-model is required for openai-compatible provider")
        if not config.base_url:
            raise AdapterConfigError("--reviewer-base-url is required for openai-compatible provider")
        return OpenAICompatibleModelAdapterFactory(
            OpenAICompatibleConfig(
                base_url=config.base_url,
                api_key=api_key,
                model=config.model,
            )
        )
    raise AdapterConfigError(f"unsupported reviewer provider: {provider_name}")


def _fake_single_shot_adapter() -> FakeToolCallingAdapter:
    return FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=json.dumps(
                    {
                        "contract_assessments": [],
                        "confirmed_findings": [],
                        "rejected_hypotheses": [],
                        "uncertainties": ["Fake provider does not perform semantic review."],
                        "observation_refs": [],
                        "investigation_summary": "Fake reviewer executed.",
                        "status": "partial",
                    }
                ),
                provider_name="fake",
                model="fake-reviewer",
                raw={"fake": True},
            )
        ]
    )
```

- [ ] **Step 4: Run factory tests and selected adapter tests**

Run:

```powershell
python -B -m pytest tests/test_model_adapter_factory.py tests/test_model_adapter.py -q -p no:cacheprovider
```

Expected: tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src/review_agent/model_adapter_factory.py tests/test_model_adapter_factory.py
git commit -m "feat: add model adapter factory"
```

---

## Task 3: Migrate single-shot reviewer to ModelAdapter

**Files:**

- Modify: `src/review_agent/reviewer.py`
- Modify: `tests/test_reviewer.py`

- [ ] **Step 1: Replace provider-based reviewer test with adapter-based test**

In `tests/test_reviewer.py`, replace:

```python
from review_agent.provider import FakeProvider
```

with:

```python
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import ModelResponseKind, ModelToolCall, ModelTurnResponse
```

Replace `test_run_single_reviewer_calls_provider_and_parses_result()` with:

```python
def test_run_single_reviewer_calls_adapter_and_parses_result():
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="""
                {
                  "contract_assessments": [],
                  "confirmed_findings": [],
                  "rejected_hypotheses": [],
                  "uncertainties": ["No tool gateway available."],
                  "observation_refs": ["O-diff-auth"],
                  "investigation_summary": "Reviewed diff excerpt.",
                  "status": "partial"
                }
                """,
                provider_name="fake",
                model="fake-reviewer",
            )
        ]
    )

    run = run_single_reviewer(
        adapter=adapter,
        assignment=make_assignment(),
        intent=make_intent(),
        diff_excerpt=["-    return user.role == 'admin'", "+    return True"],
        observations={"O-diff-auth": "auth.py changed between base and head"},
        trace_id="trace-reviewer-1",
    )

    assert run.result.status is ReviewerResultStatus.PARTIAL
    assert run.result.observation_refs == ["O-diff-auth"]
    assert run.response.provider_name == "fake-tool-calling"
    assert run.envelope.parameters["trace_id"] == "trace-reviewer-1"
    assert "return True" in run.envelope.messages[0]["content"]
    assert adapter.requests[0].tools == []
    assert adapter.requests[0].parameters["tool_choice"] == "none"
```

- [ ] **Step 2: Add single-shot rejection tests**

Add to `tests/test_reviewer.py`:

```python
def test_run_single_reviewer_rejects_tool_calls_without_executing_tools():
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[ModelToolCall("call-1", "compare_base_head", {"path": "auth.py"})],
                provider_name="fake",
                model="fake-reviewer",
            )
        ]
    )

    run = run_single_reviewer(
        adapter=adapter,
        assignment=make_assignment(),
        intent=make_intent(),
        diff_excerpt=["+changed"],
        observations={},
        trace_id="trace-reviewer-tool-call",
    )

    assert run.result.status is ReviewerResultStatus.FAILED
    assert "single-shot reviewer received tool calls" in run.result.investigation_summary
    assert run.response.provider_name == "fake-tool-calling"


def test_run_single_reviewer_handles_invalid_adapter_response():
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.INVALID,
                error="bad response shape",
                provider_name="fake",
                model="fake-reviewer",
            )
        ]
    )

    run = run_single_reviewer(
        adapter=adapter,
        assignment=make_assignment(),
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="trace-reviewer-invalid",
    )

    assert run.result.status is ReviewerResultStatus.FAILED
    assert run.result.uncertainties == ["bad response shape"]
    assert run.response.content == "bad response shape"
```

- [ ] **Step 3: Run reviewer tests and verify failure**

Run:

```powershell
python -B -m pytest tests/test_reviewer.py -q -p no:cacheprovider
```

Expected: fail because `run_single_reviewer()` still accepts `provider`, not `adapter`.

- [ ] **Step 4: Implement adapter-based single-shot reviewer**

In `src/review_agent/reviewer.py`, remove:

```python
from review_agent.provider import ModelProvider, ModelResponse
```

Add:

```python
from review_agent.model_adapter import ModelAdapter
from review_agent.model_protocol import ModelResponse, ModelResponseKind, ModelTurnRequest
```

Replace `run_single_reviewer()` with:

```python
def run_single_reviewer(
    adapter: ModelAdapter,
    assignment: Assignment,
    intent: IntentPacket,
    diff_excerpt: list[str],
    observations: dict[str, str],
    trace_id: str,
) -> ReviewerRun:
    envelope = build_reviewer_envelope(
        assignment=assignment,
        intent=intent,
        code_snippets={"Diff Excerpt": "\n".join(diff_excerpt)},
        observations=observations,
        trace_id=trace_id,
    )
    request = ModelTurnRequest(
        system=envelope.system,
        tools=[],
        messages=list(envelope.messages),
        tool_results=[],
        parameters={**dict(envelope.parameters), "tool_choice": "none"},
    )
    turn_response = adapter.complete_turn(request)
    response = ModelResponse(
        content=turn_response.final_text or turn_response.error or "",
        provider_name=turn_response.provider_name,
        model=turn_response.model,
        raw=turn_response.raw,
    )
    if turn_response.kind is ModelResponseKind.FINAL:
        try:
            result = parse_reviewer_result(turn_response.final_text or "")
        except ReviewerResultParseError as error:
            message = f"single-shot final response parse failed: {error}"
            result = ReviewerResult(
                uncertainties=[message],
                investigation_summary=message,
                status=ReviewerResultStatus.FAILED,
            )
        return ReviewerRun(envelope=envelope, response=response, result=result)
    if turn_response.kind is ModelResponseKind.TOOL_CALLS:
        message = "single-shot reviewer received tool calls; use --reviewer-loop agent-loop to enable tools"
    else:
        message = turn_response.error or f"single-shot reviewer received invalid response kind: {turn_response.kind.value}"
    result = ReviewerResult(
        uncertainties=[message],
        investigation_summary=message,
        status=ReviewerResultStatus.FAILED,
    )
    return ReviewerRun(envelope=envelope, response=response, result=result)
```

- [ ] **Step 5: Update second reviewer test constructor**

In `test_run_single_reviewer_uses_diff_excerpt_as_code_snippet()`, replace `FakeProvider(...)` with:

```python
adapter = FakeToolCallingAdapter(
    script=[
        ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text="""
            {
              "contract_assessments": [],
              "confirmed_findings": [],
              "rejected_hypotheses": [],
              "uncertainties": [],
              "observation_refs": [],
              "investigation_summary": "Reviewed diff.",
              "status": "completed"
            }
            """,
            provider_name="fake",
            model="fake-reviewer",
        )
    ]
)
```

Then call:

```python
run = run_single_reviewer(
    adapter=adapter,
    assignment=make_assignment(),
    intent=make_intent(),
    diff_excerpt=["+changed"],
    observations={},
    trace_id="trace-reviewer-2",
)
```

- [ ] **Step 6: Run reviewer tests and selected suites**

Run:

```powershell
python -B -m pytest tests/test_reviewer.py tests/test_model_adapter.py tests/test_model_protocol.py -q -p no:cacheprovider
```

Expected: tests pass.

- [ ] **Step 7: Commit Task 3**

Run:

```powershell
git add src/review_agent/reviewer.py tests/test_reviewer.py
git commit -m "refactor: run single reviewer through model adapter"
```

---

## Task 4: Migrate multi-reviewer orchestrator to adapter factory

**Files:**

- Modify: `src/review_agent/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

- [ ] **Step 1: Rewrite orchestrator tests around adapter factory**

In `tests/test_orchestrator.py`, remove provider imports:

```python
from review_agent.provider import ModelProviderError, ModelResponse
```

Add:

```python
from review_agent.model_adapter import ModelAdapter
from review_agent.model_adapter_factory import ModelAdapterFactory
from review_agent.model_protocol import ModelResponseKind, ModelTurnRequest, ModelTurnResponse
```

Replace `RecordingProvider` and `FailingSecondProvider` with:

```python
class RecordingAdapter:
    provider_name = "recording"

    def __init__(self, factory):
        self._factory = factory

    def complete_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        self._factory.trace_ids.append(request.parameters["trace_id"])
        content = request.messages[0]["content"]
        role_line = next(line for line in content.splitlines() if line.startswith("Role: "))
        role = role_line.removeprefix("Role: ")
        self._factory.roles.append(role)
        if self._factory.fail_on_call_number == len(self._factory.trace_ids):
            return ModelTurnResponse(
                kind=ModelResponseKind.INVALID,
                error="provider unavailable",
                provider_name="recording",
                model="recording-model",
            )
        return ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text=json.dumps(
                {
                    "contract_assessments": [],
                    "confirmed_findings": [],
                    "rejected_hypotheses": [],
                    "uncertainties": [f"{role} used fake semantic review."],
                    "observation_refs": ["O-shared"],
                    "investigation_summary": f"{role} finished.",
                    "status": "partial",
                }
            ),
            provider_name="recording",
            model="recording-model",
            raw={"role": role},
        )


class RecordingAdapterFactory:
    def __init__(self, fail_on_call_number=None):
        self.trace_ids = []
        self.roles = []
        self.fail_on_call_number = fail_on_call_number
        self.created_count = 0

    def create(self) -> ModelAdapter:
        self.created_count += 1
        return RecordingAdapter(self)
```

Update the three tests:

```python
def test_run_multi_reviewer_runs_every_assignment_with_isolated_traces():
    factory = RecordingAdapterFactory()
    assignments = [make_assignment("Core Reviewer"), make_assignment("Adversarial Reviewer")]

    run = run_multi_reviewer(
        adapter_factory=factory,
        assignments=assignments,
        intent=make_intent(),
        diff_excerpt=["+changed"],
        observations={"O-shared": "shared observation"},
        trace_id_prefix="review-123",
    )

    assert factory.created_count == 2
    assert factory.trace_ids == ["review-123-reviewer-0", "review-123-reviewer-1"]
    assert factory.roles == ["Core Reviewer", "Adversarial Reviewer"]
    assert [item.assignment.role for item in run.executions] == ["Core Reviewer", "Adversarial Reviewer"]
    assert [item.result.status.value for item in run.executions] == ["partial", "partial"]
    assert run.status_counts == {"partial": 2}
```

```python
def test_multi_reviewer_run_to_dict_contains_artifact_summary():
    run = run_multi_reviewer(
        adapter_factory=RecordingAdapterFactory(),
        assignments=[make_assignment("Core Reviewer")],
        intent=make_intent(),
        diff_excerpt=[],
        observations={"O-shared": "shared observation"},
        trace_id_prefix="review-456",
    )

    payload = multi_reviewer_run_to_dict(run)

    assert payload["reviewer_count"] == 1
    assert payload["status_counts"] == {"partial": 1}
    assert payload["executions"][0]["role"] == "Core Reviewer"
    assert payload["executions"][0]["trace_id"] == "review-456-reviewer-0"
    assert payload["executions"][0]["result"]["investigation_summary"] == "Core Reviewer finished."
```

```python
def test_run_multi_reviewer_records_failed_execution_without_aborting_remaining_artifacts():
    run = run_multi_reviewer(
        adapter_factory=RecordingAdapterFactory(fail_on_call_number=2),
        assignments=[make_assignment("Core Reviewer"), make_assignment("Adversarial Reviewer")],
        intent=make_intent(),
        diff_excerpt=[],
        observations={"O-shared": "shared observation"},
        trace_id_prefix="review-789",
    )

    payload = multi_reviewer_run_to_dict(run)

    assert payload["reviewer_count"] == 2
    assert payload["status_counts"] == {"partial": 1, "failed": 1}
    assert payload["executions"][0]["result"]["status"] == "partial"
    assert payload["executions"][1]["trace_id"] == "review-789-reviewer-1"
    assert payload["executions"][1]["result"]["status"] == "failed"
    assert "provider unavailable" in payload["executions"][1]["result"]["investigation_summary"]
```

- [ ] **Step 2: Run orchestrator tests and verify failure**

Run:

```powershell
python -B -m pytest tests/test_orchestrator.py -q -p no:cacheprovider
```

Expected: fail because `run_multi_reviewer()` still accepts `provider`.

- [ ] **Step 3: Update orchestrator implementation**

In `src/review_agent/orchestrator.py`, remove:

```python
from review_agent.provider import ModelProvider, ModelResponse
```

Add:

```python
from review_agent.model_adapter_factory import ModelAdapterFactory
from review_agent.model_protocol import ModelResponse
```

Change `run_multi_reviewer()` signature:

```python
def run_multi_reviewer(
    adapter_factory: ModelAdapterFactory,
    assignments: list[Assignment],
    intent: IntentPacket,
    diff_excerpt: list[str],
    observations: dict[str, str],
    trace_id_prefix: str,
) -> MultiReviewerRun:
```

Inside the loop, replace `provider=provider` with:

```python
adapter = adapter_factory.create()
run = run_single_reviewer(
    adapter=adapter,
    assignment=assignment,
    intent=intent,
    diff_excerpt=diff_excerpt,
    observations=observations,
    trace_id=trace_id,
)
```

Keep `_failed_execution()` unchanged except for importing `ModelResponse` from `model_protocol.py`.

- [ ] **Step 4: Run orchestrator and reviewer tests**

Run:

```powershell
python -B -m pytest tests/test_orchestrator.py tests/test_reviewer.py -q -p no:cacheprovider
```

Expected: tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add src/review_agent/orchestrator.py tests/test_orchestrator.py
git commit -m "refactor: run multi reviewer through adapter factory"
```

---

## Task 5: Migrate CLI reviewer path to ModelAdapterFactory

**Files:**

- Modify: `src/review_agent/cli.py`
- Modify: `tests/test_cli_smoke.py`

- [ ] **Step 1: Update CLI smoke tests for final provider behavior**

In `tests/test_cli_smoke.py`, rename:

```python
test_cli_agent_loop_rejects_unsupported_provider_before_provider_config
```

to:

```python
test_cli_agent_loop_openai_compatible_uses_adapter_factory
```

Replace the body with a monkeypatch that avoids network:

```python
def test_cli_agent_loop_openai_compatible_uses_adapter_factory(git_repo: Path, monkeypatch):
    from review_agent.model_adapter import FakeToolCallingAdapter
    from review_agent.model_adapter_factory import FakeModelAdapterFactory
    from review_agent.model_protocol import ModelResponseKind, ModelToolCall, ModelTurnResponse

    class ToolThenFinalFactory:
        def create(self):
            def final_response(request):
                observation_id = request.tool_results[-1].observation_ids[0]
                return ModelTurnResponse(
                    kind=ModelResponseKind.FINAL,
                    final_text=json.dumps(
                        {
                            "contract_assessments": [
                                {
                                    "contract": "regression_safety",
                                    "status": "covered",
                                    "summary": "OpenAI-compatible adapter path used tools.",
                                    "evidence_refs": [observation_id],
                                }
                            ],
                            "confirmed_findings": [],
                            "rejected_hypotheses": [],
                            "uncertainties": [],
                            "observation_refs": [observation_id],
                            "investigation_summary": "OpenAI-compatible agent loop executed.",
                            "status": "completed",
                        }
                    ),
                    provider_name="openai-compatible",
                    model="review-model",
                )

            return FakeToolCallingAdapter(
                script=[
                    ModelTurnResponse(
                        kind=ModelResponseKind.TOOL_CALLS,
                        tool_calls=[ModelToolCall("call-1", "compare_base_head", {"path": "app.py"})],
                    ),
                    final_response,
                ]
            )

    def fake_build_factory(config):
        assert config.provider_name == "openai-compatible"
        assert config.model == "review-model"
        assert config.base_url == "https://example.test/v1"
        return ToolThenFinalFactory()

    monkeypatch.setattr("review_agent.cli.build_model_adapter_factory_from_config", fake_build_factory)
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change app")
    head = run_git(git_repo, "rev-parse", "HEAD")

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            base,
            "--head",
            head,
            "--reviewer-loop",
            "agent-loop",
            "--reviewer-provider",
            "openai-compatible",
            "--reviewer-model",
            "review-model",
            "--reviewer-base-url",
            "https://example.test/v1",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    trace = json.loads((run_dir / "reviewer_agent_trace.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "reviewer_result.json").read_text(encoding="utf-8"))
    raw = json.loads((run_dir / "reviewer_raw_response.json").read_text(encoding="utf-8"))

    assert trace["tool_call_count"] == 1
    assert result["status"] == "completed"
    assert raw["provider_name"] == "openai-compatible"
```

- [ ] **Step 2: Run the updated CLI test and verify failure**

Run:

```powershell
python -B -m pytest tests/test_cli_smoke.py::test_cli_agent_loop_openai_compatible_uses_adapter_factory -q -p no:cacheprovider
```

Expected: fail because CLI still rejects openai-compatible agent-loop.

- [ ] **Step 3: Update CLI imports**

In `src/review_agent/cli.py`, remove:

```python
import json
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import ModelResponseKind, ModelToolCall, ModelTurnResponse
from review_agent.provider import ProviderConfigError, build_provider_from_config
```

Add:

```python
from review_agent.model_adapter_factory import (
    AdapterConfigError,
    ModelAdapterConfig,
    ModelAdapterFactory,
    build_model_adapter_factory_from_config,
)
```

- [ ] **Step 4: Remove CLI-owned fake adapter builder**

Delete `_fake_agent_loop_adapter()` from `src/review_agent/cli.py`.

The fake script now belongs to `FakeModelAdapterFactory` in `model_adapter_factory.py`.

- [ ] **Step 5: Build adapter factory before reviewer execution**

Replace the provider construction block in `_run_review()`:

```python
if args.reviewer_loop == "agent-loop" and args.reviewer_provider != "fake":
    print("--reviewer-loop agent-loop currently requires --reviewer-provider fake")
    return 2
try:
    provider = build_provider_from_config(...)
except ProviderConfigError as error:
    ...
if args.reviewer_mode == "multi" and provider is None:
    ...
```

with:

```python
try:
    adapter_factory = build_model_adapter_factory_from_config(
        ModelAdapterConfig(
            provider_name=args.reviewer_provider,
            model=args.reviewer_model,
            base_url=args.reviewer_base_url,
            api_key_env=args.reviewer_api_key_env,
        )
    )
except AdapterConfigError as error:
    print(f"Reviewer adapter configuration error: {error}")
    return 2
if args.reviewer_mode == "multi" and adapter_factory is None:
    print("--reviewer-mode multi requires --reviewer-provider fake or openai-compatible")
    return 2
```

- [ ] **Step 6: Route reviewer execution through adapter factory**

Replace:

```python
if provider is not None and assignments:
```

with:

```python
if adapter_factory is not None and assignments:
```

Single-shot single reviewer branch:

```python
reviewer_run = run_single_reviewer(
    adapter=adapter_factory.create(),
    assignment=assignments[0],
    intent=intent,
    diff_excerpt=change_summary.diff_excerpt,
    observations=reviewer_observations,
    trace_id=f"{review_id}-reviewer-0",
)
```

Single agent-loop branch:

```python
loop_run = run_reviewer_agent_loop(
    adapter=adapter_factory.create(),
    gateway=gateway,
    assignment=assignments[0],
    intent=intent,
    diff_excerpt=change_summary.diff_excerpt,
    observations=reviewer_observations,
    trace_id=f"{review_id}-reviewer-0",
)
```

Multi single-shot branch:

```python
multi_run = run_multi_reviewer(
    adapter_factory=adapter_factory,
    assignments=assignments,
    intent=intent,
    diff_excerpt=change_summary.diff_excerpt,
    observations=reviewer_observations,
    trace_id_prefix=review_id,
)
```

Multi agent-loop branch:

```python
loop_run = run_reviewer_agent_loop(
    adapter=adapter_factory.create(),
    gateway=gateway,
    assignment=assignment,
    intent=intent,
    diff_excerpt=change_summary.diff_excerpt,
    observations=reviewer_observations,
    trace_id=trace_id,
)
```

- [ ] **Step 7: Update existing CLI assertions for config message**

In `test_cli_openai_compatible_provider_requires_api_key()`, update expected output:

```python
assert "Reviewer adapter configuration error" in capsys.readouterr().out
```

The test name can remain or be renamed to:

```python
test_cli_openai_compatible_adapter_requires_api_key
```

- [ ] **Step 8: Run CLI smoke tests**

Run:

```powershell
python -B -m pytest tests/test_cli_smoke.py -q -p no:cacheprovider
```

Expected: tests pass.

- [ ] **Step 9: Commit Task 5**

Run:

```powershell
git add src/review_agent/cli.py tests/test_cli_smoke.py
git commit -m "refactor: route cli reviewers through model adapter factory"
```

---

## Task 6: Provider dependency cleanup and regression guard

**Files:**

- Modify: `tests/test_provider.py`
- Create: `tests/test_architecture_boundaries.py`

- [ ] **Step 1: Add architecture boundary tests**

Create `tests/test_architecture_boundaries.py`:

```python
from pathlib import Path


def test_reviewer_business_modules_do_not_import_model_provider():
    modules = [
        Path("src/review_agent/reviewer.py"),
        Path("src/review_agent/orchestrator.py"),
        Path("src/review_agent/cli.py"),
        Path("src/review_agent/agent_loop.py"),
    ]

    for module in modules:
        text = module.read_text(encoding="utf-8")
        assert "ModelProvider" not in text
        assert "build_provider_from_config" not in text
```

- [ ] **Step 2: Run architecture boundary test and verify failure if any old dependency remains**

Run:

```powershell
python -B -m pytest tests/test_architecture_boundaries.py -q -p no:cacheprovider
```

Expected: pass after Tasks 3 through 5. If it fails, remove the provider import from the named reviewer business module.

- [ ] **Step 3: Mark provider tests as legacy compatibility**

At the top of `tests/test_provider.py`, add a module-level comment:

```python
# Legacy compatibility tests. Reviewer business paths use ModelAdapterFactory.
```

Do not remove provider tests in this plan. They prove the compatibility shim still behaves until a later deletion pass.

- [ ] **Step 4: Run boundary and provider tests**

Run:

```powershell
python -B -m pytest tests/test_architecture_boundaries.py tests/test_provider.py -q -p no:cacheprovider
```

Expected: tests pass.

- [ ] **Step 5: Commit Task 6**

Run:

```powershell
git add tests/test_architecture_boundaries.py tests/test_provider.py
git commit -m "test: guard reviewer model adapter boundary"
```

---

## Task 7: Final verification

**Files:**

- Modify only files required by failing tests.

- [ ] **Step 1: Run plan placeholder scan**

Run:

```powershell
$patterns = @('TB'+'D','TO'+'DO','PLACE'+'HOLDER','x'+'xx','implement '+'later','fill in '+'details','Add '+'appropriate','Write tests '+'for the above','Similar '+'to Task') -join '|'
rg -n $patterns docs/superpowers/plans/2026-07-03-unified-model-adapter-architecture.md
```

Expected: no matches.

- [ ] **Step 2: Run dependency scan**

Run:

```powershell
rg -n "ModelProvider|build_provider_from_config" src/review_agent/reviewer.py src/review_agent/orchestrator.py src/review_agent/cli.py src/review_agent/agent_loop.py
```

Expected: no matches.

- [ ] **Step 3: Run full test suite**

Run:

```powershell
python -B -m pytest -q -p no:cacheprovider
```

Expected: all tests pass. If Windows emits the known `pytest-105` temp cleanup warning with exit code 0, record it as environment noise and continue.

- [ ] **Step 4: Run local fake smoke**

Run a temporary git repo smoke equivalent to:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
python -B -m review_agent review --repo <temp-repo> --base <base> --head <head> --intent "Review local change" --reviewer-provider fake --reviewer-loop agent-loop --non-interactive
```

Verify:

```powershell
Test-Path <run-dir>\reviewer_agent_trace.json
Test-Path <run-dir>\reviewer_result.json
Select-String -Path <run-dir>\observations.jsonl -Pattern "git.compare_base_head"
```

Expected:

- trace file exists.
- reviewer result status is `completed`.
- observations contain `git.compare_base_head`.

- [ ] **Step 5: Commit plan if not already committed**

If this plan file has not been committed before implementation starts, commit it:

```powershell
git add docs/superpowers/plans/2026-07-03-unified-model-adapter-architecture.md
git commit -m "docs: plan unified model adapter architecture"
```

---

## Self-review checklist

- Every requirement in `2026-07-03-unified-model-adapter-architecture-design.md` maps to at least one task above.
- `ModelAdapterFactory` is the only new factory.
- reviewer business modules do not import `ModelProvider`.
- single-shot keeps its product meaning: one model call and no tool execution.
- agent-loop keeps Runtime isolated from provider-specific schemas.
- multi-reviewer creates a fresh adapter per reviewer execution.
- CLI no longer rejects openai-compatible agent-loop.
- legacy provider compatibility remains tested without being the reviewer main path.

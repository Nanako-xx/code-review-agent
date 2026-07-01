# Model Adapter Agent Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a provider-agnostic reviewer agent loop where Runtime speaks a project-owned model protocol, model adapters translate that protocol to concrete APIs, and reviewers can request Tool Gateway observations before returning final review results.

**Architecture:** Create a small internal model protocol first, then implement adapters against it. The reviewer agent loop consumes only the internal protocol, executes requested tools through `ToolGateway`, records observations, and returns existing `ReviewerResult` objects plus an audit trace. CLI exposes the loop behind `--reviewer-loop single-shot|agent-loop`, preserving current single-shot behavior by default.

**Tech Stack:** Python 3.11+, stdlib dataclasses/enums/json/urllib, existing `ToolGateway`, `ObservationStore`, `ReviewerResult` parser, `pytest`.

---

## Scope

In scope:

- Project-owned model protocol types for tool specs, tool calls, tool results, turn requests, and turn responses.
- `FakeToolCallingAdapter` for deterministic tests and local smoke runs.
- `OpenAICompatibleToolAdapter` for OpenAI-compatible chat completions tool calling.
- Reviewer agent loop that:
  - sends model turn requests,
  - executes model-requested tools through `ToolGateway`,
  - records observation IDs,
  - feeds tool results back into later turns,
  - parses final reviewer JSON using the existing parser,
  - records an auditable trace,
  - respects assignment `max_turns` and `max_tool_calls`.
- CLI flag: `--reviewer-loop single-shot|agent-loop`, default `single-shot`.
- Agent loop artifacts:
  - `reviewer_agent_trace.json` for single reviewer mode.
  - `reviewer_<index>_agent_trace.json` for multi reviewer mode.

Out of scope:

- GitHub / PR platform integration.
- Eval harness.
- Claude-native adapter.
- JSON-text fallback adapter.
- True parallel reviewer execution.
- Dynamic follow-up assignments.
- LLM Reconciler Agent.
- Long-term memory.

## File structure

- Create `src/review_agent/model_protocol.py`
  - Owns internal model protocol dataclasses and serialization helpers.
- Create `src/review_agent/model_adapter.py`
  - Owns adapter protocol, fake adapter, and OpenAI-compatible tool adapter.
- Create `src/review_agent/agent_loop.py`
  - Owns reviewer multi-turn runtime and trace dataclasses.
- Modify `src/review_agent/cli.py`
  - Adds `--reviewer-loop`.
  - Routes to single-shot or agent-loop reviewer execution.
  - Writes agent trace artifacts.
- Keep `src/review_agent/reviewer.py`
  - Keeps existing single-shot path and parsing logic.
- Keep `src/review_agent/orchestrator.py`
  - No large rewrite required for this slice; CLI can loop assignments for agent-loop multi mode while preserving existing `run_multi_reviewer()` for single-shot multi mode.
- Create `tests/test_model_protocol.py`
- Create `tests/test_model_adapter.py`
- Create `tests/test_agent_loop.py`
- Modify `tests/test_cli_smoke.py`

## Task 1: Internal Model Protocol And Fake Adapter

**Files:**

- Create: `src/review_agent/model_protocol.py`
- Create: `src/review_agent/model_adapter.py`
- Create: `tests/test_model_protocol.py`
- Create: `tests/test_model_adapter.py`

- [ ] **Step 1: Write failing model protocol tests**

Create `tests/test_model_protocol.py`:

```python
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelToolResult,
    ModelToolSpec,
    ModelTurnRequest,
    ModelTurnResponse,
    model_turn_response_to_dict,
)


def test_model_protocol_serializes_tool_call_response():
    response = ModelTurnResponse(
        kind=ModelResponseKind.TOOL_CALLS,
        tool_calls=[
            ModelToolCall(
                call_id="call-1",
                tool_name="read_range",
                arguments={"path": "app.py", "revision": "head", "line_start": 1, "line_end": 20},
            )
        ],
        raw={"provider": "fake"},
    )

    payload = model_turn_response_to_dict(response)

    assert payload["kind"] == "tool_calls"
    assert payload["tool_calls"][0]["tool_name"] == "read_range"
    assert payload["tool_calls"][0]["arguments"]["path"] == "app.py"


def test_model_turn_request_carries_tools_and_tool_results():
    request = ModelTurnRequest(
        system="system",
        tools=[
            ModelToolSpec(
                name="search_code",
                description="Search code",
                parameters_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ],
        messages=[{"role": "user", "content": "Review this change"}],
        tool_results=[
            ModelToolResult(
                call_id="call-1",
                tool_name="search_code",
                content="O-123: result summary",
                observation_ids=["O-123"],
            )
        ],
        parameters={"trace_id": "review-1-reviewer-0"},
    )

    assert request.tools[0].name == "search_code"
    assert request.tool_results[0].observation_ids == ["O-123"]
```

- [ ] **Step 2: Run model protocol tests and verify failure**

Run:

```powershell
python -B -m pytest tests/test_model_protocol.py -q -p no:cacheprovider
```

Expected: fail because `review_agent.model_protocol` does not exist.

- [ ] **Step 3: Implement `src/review_agent/model_protocol.py`**

Implement:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ModelResponseKind(str, Enum):
    TOOL_CALLS = "tool_calls"
    FINAL = "final"
    INVALID = "invalid"


@dataclass(frozen=True)
class ModelToolSpec:
    name: str
    description: str
    parameters_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelToolCall:
    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelToolResult:
    call_id: str
    tool_name: str
    content: str
    observation_ids: list[str] = field(default_factory=list)
    is_error: bool = False


@dataclass(frozen=True)
class ModelTurnRequest:
    system: str
    tools: list[ModelToolSpec]
    messages: list[dict[str, Any]]
    tool_results: list[ModelToolResult]
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ModelTurnResponse:
    kind: ModelResponseKind
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    final_text: str | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    provider_name: str = "unknown"
    model: str = "unknown"


def model_turn_response_to_dict(response: ModelTurnResponse) -> dict[str, Any]:
    payload = asdict(response)
    payload["kind"] = response.kind.value
    return payload
```

- [ ] **Step 4: Run model protocol tests and verify pass**

Run:

```powershell
python -B -m pytest tests/test_model_protocol.py -q -p no:cacheprovider
```

Expected: `2 passed`.

- [ ] **Step 5: Write failing fake adapter tests**

Create `tests/test_model_adapter.py`:

```python
import json

from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelTurnRequest,
    ModelTurnResponse,
)


def make_request(tool_results=None):
    return ModelTurnRequest(
        system="system",
        tools=[],
        messages=[{"role": "user", "content": "Review change"}],
        tool_results=tool_results or [],
        parameters={"trace_id": "review-1-reviewer-0"},
    )


def test_fake_adapter_returns_scripted_tool_call():
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[ModelToolCall("call-1", "compare_base_head", {"path": "app.py"})],
            )
        ]
    )

    response = adapter.complete_turn(make_request())

    assert response.kind is ModelResponseKind.TOOL_CALLS
    assert response.tool_calls[0].tool_name == "compare_base_head"
    assert response.provider_name == "fake-tool-calling"


def test_fake_adapter_can_compute_final_response_from_request():
    def final_response(request):
        observation_id = request.tool_results[0].observation_ids[0]
        return ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text=json.dumps(
                {
                    "contract_assessments": [
                        {
                            "contract": "regression_safety",
                            "status": "covered",
                            "summary": "Checked diff observation.",
                            "evidence_refs": [observation_id],
                        }
                    ],
                    "confirmed_findings": [],
                    "rejected_hypotheses": [],
                    "uncertainties": [],
                    "observation_refs": [observation_id],
                    "investigation_summary": "Used tool observation.",
                    "status": "completed",
                }
            ),
        )

    adapter = FakeToolCallingAdapter(script=[final_response])
    response = adapter.complete_turn(
        make_request(
            tool_results=[
                type(
                    "Result",
                    (),
                    {
                        "observation_ids": ["O-abc"],
                    },
                )()
            ]
        )
    )

    assert response.kind is ModelResponseKind.FINAL
    assert "O-abc" in response.final_text
```

- [ ] **Step 6: Run fake adapter tests and verify failure**

Run:

```powershell
python -B -m pytest tests/test_model_adapter.py -q -p no:cacheprovider
```

Expected: fail because `review_agent.model_adapter` does not exist.

- [ ] **Step 7: Implement fake adapter**

In `src/review_agent/model_adapter.py`, implement:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from review_agent.model_protocol import ModelTurnRequest, ModelTurnResponse


class ModelAdapter(Protocol):
    provider_name: str

    def complete_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        raise NotImplementedError


ScriptItem = ModelTurnResponse | Callable[[ModelTurnRequest], ModelTurnResponse]


class FakeToolCallingAdapter:
    provider_name = "fake-tool-calling"

    def __init__(self, script: list[ScriptItem]):
        self._script = list(script)
        self.requests: list[ModelTurnRequest] = []

    def complete_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        self.requests.append(request)
        if not self._script:
            return ModelTurnResponse(
                kind=ModelResponseKind.INVALID,
                error="fake adapter script exhausted",
                provider_name=self.provider_name,
                model="fake-tool-model",
            )
        item = self._script.pop(0)
        response = item(request) if callable(item) else item
        return ModelTurnResponse(
            kind=response.kind,
            tool_calls=response.tool_calls,
            final_text=response.final_text,
            error=response.error,
            raw=response.raw,
            provider_name=self.provider_name,
            model=response.model if response.model != "unknown" else "fake-tool-model",
        )
```

Import `ModelResponseKind` in the file.

- [ ] **Step 8: Run Task 1 tests and commit**

Run:

```powershell
python -B -m pytest tests/test_model_protocol.py tests/test_model_adapter.py -q -p no:cacheprovider
```

Expected: all Task 1 tests pass.

Commit:

```powershell
git add src/review_agent/model_protocol.py src/review_agent/model_adapter.py tests/test_model_protocol.py tests/test_model_adapter.py
git commit -m "feat: add model protocol and fake adapter"
```

## Task 2: Reviewer Agent Loop Runtime

**Files:**

- Create: `src/review_agent/agent_loop.py`
- Create: `tests/test_agent_loop.py`
- Modify: `src/review_agent/model_protocol.py` if trace serialization needs small helpers.

- [ ] **Step 1: Write failing agent loop tests**

Create `tests/test_agent_loop.py`:

```python
import json

from review_agent.agent_loop import agent_loop_run_to_dict, run_reviewer_agent_loop
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import ModelResponseKind, ModelToolCall, ModelTurnResponse
from review_agent.models import IntentPacket, IntentSource, IntentStatus
from review_agent.observations import ObservationStore
from review_agent.tool_gateway import ToolGateway
from tests.conftest import run_git
from tests.test_orchestrator import make_assignment


def make_intent():
    return IntentPacket(
        goal="Review risky change",
        sources={"goal": IntentSource.EXPLICIT},
        status=IntentStatus.SUFFICIENT,
    )


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
                        "summary": "Compared base and head.",
                        "evidence_refs": [observation_id],
                    }
                ],
                "confirmed_findings": [],
                "rejected_hypotheses": [],
                "uncertainties": [],
                "observation_refs": [observation_id],
                "investigation_summary": "Reviewed with a tool observation.",
                "status": "completed",
            }
        ),
    )


def test_agent_loop_executes_tool_call_and_returns_final_result(git_repo):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change app")
    head = run_git(git_repo, "rev-parse", "HEAD")
    run_dir = git_repo / ".review-agent" / "runs" / "review-1"
    observation_store = ObservationStore(run_dir)
    gateway = ToolGateway(git_repo, base, head, observation_store)
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[ModelToolCall("call-1", "compare_base_head", {"path": "app.py"})],
            ),
            final_response,
        ]
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=make_assignment("Core Reviewer"),
        intent=make_intent(),
        diff_excerpt=["diff excerpt"],
        observations={},
        trace_id="review-1-reviewer-0",
    )

    assert run.result.status.value == "completed"
    assert run.result.investigation_summary == "Reviewed with a tool observation."
    assert run.trace.tool_call_count == 1
    assert run.trace.turns[0].tool_calls[0].tool_name == "compare_base_head"
    assert run.trace.turns[0].tool_results[0].observation_ids
    assert adapter.requests[1].tool_results[0].content
    assert list(observation_store.summaries_by_id())


def test_agent_loop_returns_partial_when_tool_budget_is_exhausted(git_repo):
    base = run_git(git_repo, "rev-parse", "HEAD")
    head = base
    observation_store = ObservationStore(git_repo / ".review-agent" / "runs" / "review-budget")
    gateway = ToolGateway(git_repo, base, head, observation_store)
    assignment = make_assignment("Core Reviewer")
    assignment = type(assignment)(
        role=assignment.role,
        mission=assignment.mission,
        assignment_reason=assignment.assignment_reason,
        assigned_contract=assignment.assigned_contract,
        required_checks=assignment.required_checks,
        initial_context=assignment.initial_context,
        max_turns=assignment.max_turns,
        max_tool_calls=0,
    )
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[ModelToolCall("call-1", "compare_base_head", {"path": "app.py"})],
            )
        ]
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=assignment,
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="review-budget-reviewer-0",
    )

    assert run.result.status.value == "partial"
    assert "tool budget exhausted" in run.result.uncertainties
    assert run.trace.final_status == "partial"
```

- [ ] **Step 2: Run agent loop tests and verify failure**

Run:

```powershell
python -B -m pytest tests/test_agent_loop.py -q -p no:cacheprovider
```

Expected: fail because `review_agent.agent_loop` does not exist.

- [ ] **Step 3: Implement `src/review_agent/agent_loop.py`**

Implement these dataclasses:

```python
@dataclass(frozen=True)
class AgentLoopTurn:
    turn_index: int
    response_kind: str
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    tool_results: list[ModelToolResult] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class AgentLoopTrace:
    trace_id: str
    turns: list[AgentLoopTurn]
    tool_call_count: int
    final_status: str


@dataclass(frozen=True)
class AgentLoopRun:
    envelope: ModelInvocationEnvelope
    response: ModelResponse
    result: ReviewerResult
    trace: AgentLoopTrace
```

Implement:

```python
def run_reviewer_agent_loop(
    adapter: ModelAdapter,
    gateway: ToolGateway,
    assignment: Assignment,
    intent: IntentPacket,
    diff_excerpt: list[str],
    observations: dict[str, str],
    trace_id: str,
) -> AgentLoopRun:
    ...


def agent_loop_run_to_dict(run: AgentLoopRun) -> dict[str, Any]:
    ...
```

Implementation rules:

- Build the first envelope using existing `build_reviewer_envelope()`.
- Convert envelope tools into `ModelToolSpec` using:

```python
ModelToolSpec(
    name=str(tool["name"]),
    description=str(tool.get("description", "")),
    parameters_schema=dict(tool.get("parameters", {})),
)
```

- Each turn creates a `ModelTurnRequest`.
- If adapter returns `TOOL_CALLS`, check `tool_call_count + len(tool_calls) <= assignment.max_tool_calls`.
- Execute each call through `gateway.execute(call.tool_name, call.arguments)`.
- Convert each `ToolExecutionResult` into:

```python
ModelToolResult(
    call_id=call.call_id,
    tool_name=call.tool_name,
    content=result.context_view,
    observation_ids=result.observation_ids,
)
```

- If `ToolGatewayError` occurs, return a `ModelToolResult` with `is_error=True` and content containing the error.
- If adapter returns `FINAL`, parse with `parse_reviewer_result(response.final_text or "")`.
- If final parsing fails, return failed `ReviewerResult` with uncertainty.
- If loop exhausts turns, return partial `ReviewerResult`.
- Use `ModelResponse` with provider/model/raw from the final turn for compatibility with existing `ReviewerExecution`.

- [ ] **Step 4: Run agent loop tests and verify pass**

Run:

```powershell
python -B -m pytest tests/test_agent_loop.py tests/test_model_protocol.py tests/test_model_adapter.py -q -p no:cacheprovider
```

Expected: all Task 1 and Task 2 tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src/review_agent/agent_loop.py tests/test_agent_loop.py
git commit -m "feat: add reviewer agent loop runtime"
```

## Task 3: CLI Integration For Agent Loop

**Files:**

- Modify: `src/review_agent/cli.py`
- Modify: `tests/test_cli_smoke.py`

- [ ] **Step 1: Write failing CLI tests**

Add to `tests/test_cli_smoke.py`:

```python
def test_cli_agent_loop_fake_reviewer_writes_trace_artifact(git_repo: Path):
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
            "--intent",
            "Review arithmetic change",
            "--reviewer-provider",
            "fake",
            "--reviewer-loop",
            "agent-loop",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    trace = json.loads((run_dir / "reviewer_agent_trace.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "reviewer_result.json").read_text(encoding="utf-8"))

    assert trace["tool_call_count"] == 1
    assert trace["turns"][0]["tool_calls"][0]["tool_name"] == "compare_base_head"
    assert result["status"] == "completed"
```

Add a multi reviewer test:

```python
def test_cli_multi_agent_loop_writes_per_reviewer_trace_artifacts(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text("def is_admin(user):\n    return True\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "change auth")
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
            "--intent",
            "Change authorization behavior",
            "--reviewer-provider",
            "fake",
            "--reviewer-mode",
            "multi",
            "--reviewer-loop",
            "agent-loop",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]

    assert (run_dir / "reviewer_0_agent_trace.json").exists()
    assert (run_dir / "reviewer_1_agent_trace.json").exists()
    assert (run_dir / "multi_reviewer_result.json").exists()
    assert (run_dir / "reconciliation.json").exists()
    assert (run_dir / "completion.json").exists()
```

- [ ] **Step 2: Run focused CLI tests and verify failure**

Run:

```powershell
python -B -m pytest tests/test_cli_smoke.py::test_cli_agent_loop_fake_reviewer_writes_trace_artifact tests/test_cli_smoke.py::test_cli_multi_agent_loop_writes_per_reviewer_trace_artifacts -q -p no:cacheprovider
```

Expected: fail because `--reviewer-loop` does not exist.

- [ ] **Step 3: Add CLI flag and fake adapter builder**

In `src/review_agent/cli.py`:

- Add parser arg:

```python
review.add_argument("--reviewer-loop", choices=["single-shot", "agent-loop"], default="single-shot")
```

- Import:

```python
from review_agent.agent_loop import agent_loop_run_to_dict, run_reviewer_agent_loop
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import ModelResponseKind, ModelToolCall, ModelTurnResponse
```

- Add helper `_fake_agent_loop_adapter(path: str) -> FakeToolCallingAdapter`:

```python
def _fake_agent_loop_adapter(path: str) -> FakeToolCallingAdapter:
    def final_response(request):
        observation_id = request.tool_results[-1].observation_ids[0] if request.tool_results else ""
        return ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text=json.dumps(
                {
                    "contract_assessments": [
                        {
                            "contract": "regression_safety",
                            "status": "covered",
                            "summary": "Fake agent loop used a tool observation.",
                            "evidence_refs": [observation_id] if observation_id else [],
                        }
                    ],
                    "confirmed_findings": [],
                    "rejected_hypotheses": [],
                    "uncertainties": [],
                    "observation_refs": [observation_id] if observation_id else [],
                    "investigation_summary": "Fake agent loop reviewer executed.",
                    "status": "completed",
                }
            ),
        )

    return FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[ModelToolCall("call-1", "compare_base_head", {"path": path})],
            ),
            final_response,
        ]
    )
```

Use the first changed file as `path`, falling back to `"."` only when there are no changed files.

- [ ] **Step 4: Route single reviewer agent-loop mode**

In the existing single reviewer branch:

- If `args.reviewer_loop == "single-shot"`, keep current `run_single_reviewer()` path unchanged.
- If `args.reviewer_loop == "agent-loop"`:
  - build adapter with `_fake_agent_loop_adapter(change_summary.changed_files[0])` when provider is fake,
  - create `ToolGateway`,
  - call `run_reviewer_agent_loop(...)`,
  - set `reviewer_result = loop_run.result`,
  - write:
    - `reviewer_envelope.json`
    - `reviewer_raw_response.json`
    - `reviewer_result.json`
    - `reviewer_agent_trace.json`

- [ ] **Step 5: Route multi reviewer agent-loop mode**

In the existing multi reviewer branch:

- If `args.reviewer_loop == "single-shot"`, keep current `run_multi_reviewer()` path unchanged.
- If `args.reviewer_loop == "agent-loop"`:
  - loop through assignments,
  - call `run_reviewer_agent_loop(...)` per assignment with trace IDs `f"{review_id}-reviewer-{index}"`,
  - convert each loop run into `ReviewerExecution`,
  - build `MultiReviewerRun(executions=executions)`,
  - write current per-reviewer envelope/raw/result artifacts,
  - additionally write `reviewer_<index>_agent_trace.json`,
  - run existing reconciliation and completion code.

- [ ] **Step 6: Run CLI tests and selected suites**

Run:

```powershell
python -B -m pytest tests/test_cli_smoke.py tests/test_agent_loop.py tests/test_model_adapter.py tests/test_model_protocol.py -q -p no:cacheprovider
```

Expected: selected tests pass and previous CLI smoke tests still pass.

- [ ] **Step 7: Commit Task 3**

Run:

```powershell
git add src/review_agent/cli.py tests/test_cli_smoke.py
git commit -m "feat: expose reviewer agent loop in cli"
```

## Task 4: OpenAI-Compatible Tool Adapter

**Files:**

- Modify: `src/review_agent/model_adapter.py`
- Modify: `tests/test_model_adapter.py`

- [ ] **Step 1: Write failing OpenAI-compatible adapter tests**

Append to `tests/test_model_adapter.py`:

```python
from review_agent.model_adapter import OpenAICompatibleToolAdapter
from review_agent.model_protocol import ModelToolSpec
from review_agent.provider import OpenAICompatibleConfig


def test_openai_compatible_adapter_converts_tool_call_response():
    captured = {}

    def transport(url, headers, payload, timeout_seconds):
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "read_range",
                                    "arguments": '{"path": "app.py", "revision": "head", "line_start": 1, "line_end": 10}',
                                },
                            }
                        ]
                    }
                }
            ]
        }

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    request = ModelTurnRequest(
        system="system",
        tools=[
            ModelToolSpec(
                name="read_range",
                description="Read range",
                parameters_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        ],
        messages=[{"role": "user", "content": "Review"}],
        tool_results=[],
        parameters={"max_output_tokens": 1000, "temperature": 0},
    )

    response = adapter.complete_turn(request)

    assert captured["payload"]["tools"][0]["function"]["name"] == "read_range"
    assert response.kind is ModelResponseKind.TOOL_CALLS
    assert response.tool_calls[0].tool_name == "read_range"
    assert response.tool_calls[0].arguments["path"] == "app.py"


def test_openai_compatible_adapter_converts_final_text_response():
    def transport(url, headers, payload, timeout_seconds):
        return {"choices": [{"message": {"content": '{"status": "partial"}'}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )

    response = adapter.complete_turn(make_request())

    assert response.kind is ModelResponseKind.FINAL
    assert response.final_text == '{"status": "partial"}'
    assert response.provider_name == "openai-compatible"
```

- [ ] **Step 2: Run adapter tests and verify failure**

Run:

```powershell
python -B -m pytest tests/test_model_adapter.py -q -p no:cacheprovider
```

Expected: fail because `OpenAICompatibleToolAdapter` does not exist.

- [ ] **Step 3: Implement `OpenAICompatibleToolAdapter`**

In `src/review_agent/model_adapter.py`:

- Reuse `OpenAICompatibleConfig` and the same transport concept from `provider.py`.
- Build payload:

```python
{
    "model": config.model,
    "messages": [{"role": "system", "content": request.system}, *request.messages, *tool_result_messages],
    "tools": [_tool_spec_to_openai(tool) for tool in request.tools],
    "tool_choice": request.parameters.get("tool_choice", "auto"),
    "max_tokens": request.parameters.get("max_output_tokens", 4096),
    "temperature": request.parameters.get("temperature", 0),
}
```

- Convert `ModelToolResult` objects to OpenAI-compatible tool messages:

```python
{
    "role": "tool",
    "tool_call_id": result.call_id,
    "content": result.content,
}
```

- Parse response:
  - If `message.tool_calls` exists, return `ModelResponseKind.TOOL_CALLS`.
  - Else return `ModelResponseKind.FINAL` with `message.content`.
  - If JSON arguments cannot be parsed, return `ModelResponseKind.INVALID`.

- [ ] **Step 4: Run adapter tests and selected suites**

Run:

```powershell
python -B -m pytest tests/test_model_adapter.py tests/test_agent_loop.py tests/test_provider.py -q -p no:cacheprovider
```

Expected: selected tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add src/review_agent/model_adapter.py tests/test_model_adapter.py
git commit -m "feat: add openai compatible tool adapter"
```

## Task 5: Final Verification And Plan Commit

**Files:**

- Modify only files with failing tests.
- Add plan file after implementation commits.

- [ ] **Step 1: Run plan placeholder scan**

Run:

```powershell
$patterns = @('TB'+'D','TO'+'DO','PLACE'+'HOLDER','x'+'xx','implement '+'later','fill in '+'details','Add '+'appropriate','Write tests '+'for the above','Similar '+'to Task') -join '|'
rg -n $patterns docs/superpowers/plans/2026-07-01-model-adapter-agent-loop.md
```

Expected: no matches.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
python -B -m pytest -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 3: Run local fake agent-loop smoke manually**

Create a temporary git repo under the test temp directory or `C:\tmp`, then run:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
python -B -m review_agent review --repo <temp-repo> --base <base> --head <head> --intent "Review local change" --reviewer-provider fake --reviewer-loop agent-loop --non-interactive
```

Verify:

- `reviewer_agent_trace.json` exists.
- `reviewer_result.json` status is `completed`.
- `observations.jsonl` contains a `git.compare_base_head` record.

- [ ] **Step 4: Commit plan**

Run:

```powershell
git add docs/superpowers/plans/2026-07-01-model-adapter-agent-loop.md
git commit -m "docs: plan model adapter agent loop"
```

## Self-review checklist

- Runtime depends only on internal protocol types.
- Adapter executes no tools and validates no evidence.
- Tool execution always goes through `ToolGateway`.
- Observation IDs flow back into model turns as `ModelToolResult`.
- Trace artifacts include tool calls and observation IDs.
- `single-shot` remains default and backward compatible.
- Fake adapter gives deterministic tests without network.
- OpenAI-compatible adapter is tested through injected transport, not live network.

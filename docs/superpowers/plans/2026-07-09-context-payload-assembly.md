# Context Payload Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the spec section 16 Context & Model Invocation System slice so reviewer model calls use a bounded, observable `messages` payload instead of unbounded raw string assembly.

**Architecture:** Keep the public model invocation envelope as the four standard inputs: `system`, `tools`, `messages`, and `parameters`. Add a focused context payload assembler inside `review_agent.context` that builds reviewer `messages`, applies a message-only character budget, compacts variable sections, and records context metadata in `parameters`. Thread the configured reviewer model into envelopes from CLI/reviewer/orchestrator paths.

**Tech Stack:** Python dataclasses, existing `ModelInvocationEnvelope`, pytest, local CLI smoke tests.

---

## Scope

Implement a local M1 slice of spec sections 16.1 through 16.5:

- `messages` are built by an explicit Context Payload Assembly step.
- Context budget applies only to `messages`, not `system`, `tools`, or model parameters.
- Large code snippets and observation summaries are compacted before they enter `messages`.
- Context assembly records metadata about included, compacted, and omitted sections.
- Reviewer envelopes receive the configured reviewer model name instead of the placeholder `configured-reviewer-model` in CLI paths.
- Risk level remains absent from reviewer `messages`; reviewers only see the Runtime-expanded assignment, checks, budget, and completion rules.

This plan does not add evals, GitHub/PR integration, durable memory, interactive intent questioning, full LSP, or a separate LLM-powered context planner.

## File structure

- Modify `src/review_agent/context.py`
  - Add `ContextBudget`, `ContextAssemblyResult`, and internal `ContextSection`.
  - Add `build_reviewer_context_payload(...)`.
  - Make `build_reviewer_envelope(...)` delegate `messages` construction to the assembler.
  - Add model/parameter overrides to `build_reviewer_envelope(...)`.
- Modify `src/review_agent/reviewer.py`
  - Add a `model` parameter to `run_single_reviewer(...)` and pass it into the envelope.
- Modify `src/review_agent/agent_loop.py`
  - Add a `model` parameter to `run_reviewer_agent_loop(...)` and pass it into the envelope.
- Modify `src/review_agent/orchestrator.py`
  - Add a `model` parameter to `run_multi_reviewer(...)` and `_failed_execution(...)`.
- Modify `src/review_agent/cli.py`
  - Compute reviewer invocation model from CLI args/provider.
  - Pass that model through all reviewer execution paths.
- Modify `tests/test_context.py`
  - Cover context budget, compaction metadata, four input categories, and explicit model parameters.
- Modify `tests/test_cli_smoke.py`
  - Cover CLI envelopes using real reviewer model values for fake and openai-compatible paths.

---

## Task 1: Add context budget and payload assembly

**Files:**
- Modify: `src/review_agent/context.py`
- Modify: `tests/test_context.py`

- [ ] **Step 1: Write failing context assembly tests**

Append these helpers and tests to `tests/test_context.py`:

```python
from review_agent.context import ContextBudget, build_reviewer_context_payload


def _context_assignment() -> Assignment:
    return Assignment(
        role="Core Reviewer",
        mission="Check intent alignment",
        assignment_reason=["runtime expanded low risk into one core reviewer"],
        assigned_contract=["intent_alignment"],
        required_checks=["map changed behavior to intent"],
        initial_context=InitialContext(
            changed_files=["app.py"],
            diff_ranges=["app.py:1-200"],
            code_ranges=["app.py:1-200"],
            quality_gate_summary={"python_compile": "passed"},
            observation_refs=["O-diff-app"],
        ),
        max_turns=6,
        max_tool_calls=12,
    )


def _context_intent() -> IntentPacket:
    return IntentPacket(
        goal="Review changes touching app.py",
        sources={"goal": IntentSource.INFERRED},
        status=IntentStatus.PARTIAL,
        uncertainties=["user did not provide user intent"],
    )


def test_context_payload_compacts_variable_sections_to_message_budget():
    huge_snippet = "\n".join(f"line {index}: return value_{index}" for index in range(300))
    huge_observation = "\n".join(f"observation {index}" for index in range(300)) + "\ntail-marker"

    result = build_reviewer_context_payload(
        assignment=_context_assignment(),
        intent=_context_intent(),
        code_snippets={"app.py:1-200": huge_snippet},
        observations={"O-huge": huge_observation},
        context_budget=ContextBudget(max_message_chars=1800),
    )

    content = result.messages[0]["content"]

    assert result.metadata["max_message_chars"] == 1800
    assert result.metadata["message_chars"] <= 1800
    assert "Assignment" in content
    assert "Intent Packet" in content
    assert "Completion Rules" in content
    assert "[compacted" in content
    assert "tail-marker" not in content
    assert "Code Snippets" in result.metadata["compressed_sections"]
    assert "Observation Summary" in result.metadata["compressed_sections"]


def test_context_budget_applies_to_messages_only_not_tools_or_parameters():
    result = build_reviewer_context_payload(
        assignment=_context_assignment(),
        intent=_context_intent(),
        code_snippets={"app.py:1-20": "x = 1\n" * 200},
        observations={},
        context_budget=ContextBudget(max_message_chars=1200),
    )

    assert result.metadata["message_chars"] <= 1200
    assert result.metadata["budget_scope"] == "messages_only"
    assert result.metadata["excluded_from_budget"] == ["system", "tools", "parameters"]
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/test_context.py::test_context_payload_compacts_variable_sections_to_message_budget tests/test_context.py::test_context_budget_applies_to_messages_only_not_tools_or_parameters -q -p no:cacheprovider
```

Expected: FAIL during import with `ImportError` or `AttributeError` because `ContextBudget` and `build_reviewer_context_payload` do not exist.

- [ ] **Step 3: Implement context budget and assembly result**

In `src/review_agent/context.py`, add imports and dataclasses near the top:

```python
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextBudget:
    max_message_chars: int = 16000
    compacted_section_min_chars: int = 180


@dataclass(frozen=True)
class ContextAssemblyResult:
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ContextSection:
    name: str
    content: str
    required: bool
```

Add this function below `build_reviewer_envelope(...)` or before the private block helpers:

```python
def build_reviewer_context_payload(
    *,
    assignment: Assignment,
    intent: IntentPacket,
    code_snippets: dict[str, str],
    observations: dict[str, str],
    context_budget: ContextBudget | None = None,
) -> ContextAssemblyResult:
    budget = context_budget or ContextBudget()
    sections = [
        ContextSection("Assignment", _assignment_block(assignment), True),
        ContextSection("Intent Packet", _intent_block(intent), True),
        ContextSection("Initial Context", _initial_context_block(assignment), True),
        ContextSection("Code Snippets", _code_block(code_snippets), False),
        ContextSection("Observation Summary", _observation_block(observations), False),
        ContextSection("Completion Rules", _completion_block(assignment), True),
    ]
    content, metadata = _assemble_sections(sections, budget)
    return ContextAssemblyResult(messages=[{"role": "user", "content": content}], metadata=metadata)
```

Add these helpers near the bottom of `context.py`:

```python
def _assemble_sections(sections: list[ContextSection], budget: ContextBudget) -> tuple[str, dict[str, Any]]:
    included: list[str] = []
    compressed: list[str] = []
    omitted: list[str] = []
    rendered: list[str] = []

    for section in sections:
        candidate = section.content
        next_content = "\n\n".join([*rendered, candidate]) if rendered else candidate
        if len(next_content) <= budget.max_message_chars:
            rendered.append(candidate)
            included.append(section.name)
            continue

        remaining = _remaining_chars(rendered, budget.max_message_chars)
        if section.required:
            compacted = _compact_text(candidate, max(remaining, budget.compacted_section_min_chars), section.name)
            rendered.append(compacted)
            included.append(section.name)
            compressed.append(section.name)
            continue

        if remaining >= budget.compacted_section_min_chars:
            compacted = _compact_text(candidate, remaining, section.name)
            rendered.append(compacted)
            included.append(section.name)
            compressed.append(section.name)
            continue

        omitted.append(section.name)

    content = "\n\n".join(rendered)
    if len(content) > budget.max_message_chars:
        content = _compact_text(content, budget.max_message_chars, "Context Payload")
        if "Context Payload" not in compressed:
            compressed.append("Context Payload")

    metadata = {
        "budget_scope": "messages_only",
        "excluded_from_budget": ["system", "tools", "parameters"],
        "max_message_chars": budget.max_message_chars,
        "message_chars": len(content),
        "included_sections": included,
        "compressed_sections": compressed,
        "omitted_sections": omitted,
    }
    return content, metadata


def _remaining_chars(rendered: list[str], max_chars: int) -> int:
    if not rendered:
        return max_chars
    used = len("\n\n".join(rendered)) + 2
    return max(0, max_chars - used)


def _compact_text(text: str, max_chars: int, section_name: str) -> str:
    marker = f"\n[compacted {section_name}; full content retained in Session/Observation Store]"
    if max_chars <= len(marker):
        return marker[-max_chars:]
    if len(text) <= max_chars:
        return text
    head = text[: max_chars - len(marker)].rstrip()
    return f"{head}{marker}"
```

- [ ] **Step 4: Run context assembly tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_context.py::test_context_payload_compacts_variable_sections_to_message_budget tests/test_context.py::test_context_budget_applies_to_messages_only_not_tools_or_parameters -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Run all context tests**

Run:

```powershell
python -m pytest tests/test_context.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

Run:

```powershell
git add src/review_agent/context.py tests/test_context.py
git commit -m "feat: assemble bounded reviewer context"
```

---

## Task 2: Make reviewer envelopes use assembled context metadata

**Files:**
- Modify: `src/review_agent/context.py`
- Modify: `tests/test_context.py`

- [ ] **Step 1: Write failing envelope metadata test**

Append this test to `tests/test_context.py`:

```python
def test_reviewer_envelope_records_context_metadata():
    envelope = build_reviewer_envelope(
        assignment=_context_assignment(),
        intent=_context_intent(),
        code_snippets={"app.py:1-20": "x = 1\n" * 200},
        observations={"O-1": "app.py changed\n" * 200},
        trace_id="trace-context-metadata",
        context_budget=ContextBudget(max_message_chars=1500),
    )

    metadata = envelope.parameters["context"]

    assert metadata["budget_scope"] == "messages_only"
    assert metadata["message_chars"] <= 1500
    assert "system" in metadata["excluded_from_budget"]
    assert "tools" in metadata["excluded_from_budget"]
    assert "parameters" in metadata["excluded_from_budget"]
    assert envelope.messages[0]["role"] == "user"
    assert len(envelope.messages[0]["content"]) == metadata["message_chars"]
```

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
python -m pytest tests/test_context.py::test_reviewer_envelope_records_context_metadata -q -p no:cacheprovider
```

Expected: FAIL with `TypeError: build_reviewer_envelope() got an unexpected keyword argument 'context_budget'`.

- [ ] **Step 3: Update `build_reviewer_envelope(...)` to delegate message assembly**

In `src/review_agent/context.py`, change the function signature:

```python
def build_reviewer_envelope(
    assignment: Assignment,
    intent: IntentPacket,
    code_snippets: dict[str, str],
    observations: dict[str, str],
    trace_id: str,
    *,
    context_budget: ContextBudget | None = None,
    model: str = "configured-reviewer-model",
    max_output_tokens: int = 4096,
    reasoning_effort: str = "medium",
) -> ModelInvocationEnvelope:
```

Replace the current `content = "\n\n".join(...)` block with:

```python
    context_payload = build_reviewer_context_payload(
        assignment=assignment,
        intent=intent,
        code_snippets=code_snippets,
        observations=observations,
        context_budget=context_budget,
    )
```

Change the envelope fields:

```python
        messages=context_payload.messages,
        parameters={
            "model": model,
            "max_output_tokens": max_output_tokens,
            "reasoning_effort": reasoning_effort,
            "temperature": 0,
            "tool_choice": "auto",
            "response_schema": "reviewer_assignment_result_v1",
            "trace_id": trace_id,
            "context": context_payload.metadata,
        },
```

- [ ] **Step 4: Run envelope metadata test to verify GREEN**

Run:

```powershell
python -m pytest tests/test_context.py::test_reviewer_envelope_records_context_metadata -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Run all context tests**

Run:

```powershell
python -m pytest tests/test_context.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

Run:

```powershell
git add src/review_agent/context.py tests/test_context.py
git commit -m "feat: record reviewer context metadata"
```

---

## Task 3: Thread reviewer model parameters through runtime paths

**Files:**
- Modify: `src/review_agent/context.py`
- Modify: `src/review_agent/reviewer.py`
- Modify: `src/review_agent/agent_loop.py`
- Modify: `src/review_agent/orchestrator.py`
- Modify: `src/review_agent/cli.py`
- Modify: `tests/test_context.py`
- Modify: `tests/test_cli_smoke.py`

- [ ] **Step 1: Write failing explicit model test**

Append this test to `tests/test_context.py`:

```python
def test_reviewer_envelope_uses_explicit_model_parameters():
    envelope = build_reviewer_envelope(
        assignment=_context_assignment(),
        intent=_context_intent(),
        code_snippets={},
        observations={},
        trace_id="trace-model-params",
        model="deepseek-chat",
        max_output_tokens=2048,
        reasoning_effort="high",
    )

    assert envelope.parameters["model"] == "deepseek-chat"
    assert envelope.parameters["max_output_tokens"] == 2048
    assert envelope.parameters["reasoning_effort"] == "high"
```

- [ ] **Step 2: Write failing CLI envelope model tests**

In `tests/test_cli_smoke.py::test_cli_review_with_fake_reviewer_writes_reviewer_artifacts`, after loading `raw`, load the envelope and assert the fake model:

```python
    envelope = json.loads((run_dir / "reviewer_envelope.json").read_text(encoding="utf-8"))

    assert envelope["parameters"]["model"] == "fake-reviewer"
    assert envelope["parameters"]["context"]["budget_scope"] == "messages_only"
```

In `tests/test_cli_smoke.py::test_cli_agent_loop_openai_compatible_uses_adapter_factory`, after loading `raw`, load the envelope and assert the configured model:

```python
    envelope = json.loads((run_dir / "reviewer_envelope.json").read_text(encoding="utf-8"))

    assert envelope["parameters"]["model"] == "review-model"
    assert envelope["parameters"]["context"]["budget_scope"] == "messages_only"
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/test_context.py::test_reviewer_envelope_uses_explicit_model_parameters tests/test_cli_smoke.py::test_cli_review_with_fake_reviewer_writes_reviewer_artifacts tests/test_cli_smoke.py::test_cli_agent_loop_openai_compatible_uses_adapter_factory -q -p no:cacheprovider
```

Expected: FAIL because CLI envelopes still use the placeholder model or because `build_reviewer_envelope(...)` does not accept the model overrides before Task 2 is complete.

- [ ] **Step 4: Add model parameters to reviewer runtime functions**

In `src/review_agent/reviewer.py`, change `run_single_reviewer(...)`:

```python
def run_single_reviewer(
    adapter: ModelAdapter,
    assignment: Assignment,
    intent: IntentPacket,
    diff_excerpt: list[str],
    observations: dict[str, str],
    trace_id: str,
    *,
    model: str = "configured-reviewer-model",
) -> ReviewerRun:
```

Pass the model into `build_reviewer_envelope(...)`:

```python
        model=model,
```

In `src/review_agent/agent_loop.py`, change `run_reviewer_agent_loop(...)`:

```python
def run_reviewer_agent_loop(
    adapter: ModelAdapter,
    gateway: ToolGateway,
    assignment: Assignment,
    intent: IntentPacket,
    diff_excerpt: list[str],
    observations: dict[str, str],
    trace_id: str,
    *,
    model: str = "configured-reviewer-model",
) -> AgentLoopRun:
```

Pass the model into `build_reviewer_envelope(...)`:

```python
        model=model,
```

In `src/review_agent/orchestrator.py`, change `run_multi_reviewer(...)`:

```python
def run_multi_reviewer(
    adapter_factory: ModelAdapterFactory,
    assignments: list[Assignment],
    intent: IntentPacket,
    diff_excerpt: list[str],
    observations: dict[str, str],
    trace_id_prefix: str,
    *,
    model: str = "configured-reviewer-model",
) -> MultiReviewerRun:
```

Pass `model=model` into `run_single_reviewer(...)` and `_failed_execution(...)`.

Change `_failed_execution(...)`:

```python
def _failed_execution(
    index: int,
    trace_id: str,
    assignment: Assignment,
    intent: IntentPacket,
    diff_excerpt: list[str],
    observations: dict[str, str],
    error: Exception,
    *,
    model: str,
) -> ReviewerExecution:
```

Pass `model=model` into `build_reviewer_envelope(...)`.

- [ ] **Step 5: Thread reviewer model from CLI**

In `src/review_agent/cli.py`, add a helper near `_format_quality_gate_summary(...)`:

```python
def _reviewer_invocation_model(args: argparse.Namespace) -> str:
    if args.reviewer_model:
        return str(args.reviewer_model)
    if args.reviewer_provider == "fake":
        return "fake-reviewer"
    if args.reviewer_provider == "none":
        return "none"
    return "configured-reviewer-model"
```

After `adapter_factory` is created in `_run_review(...)`, add:

```python
    reviewer_invocation_model = _reviewer_invocation_model(args)
```

Pass `model=reviewer_invocation_model` into:

```python
run_multi_reviewer(...)
run_reviewer_agent_loop(...)
run_single_reviewer(...)
```

Every direct reviewer execution path in `_run_review(...)` should now pass the same `reviewer_invocation_model`.

- [ ] **Step 6: Run targeted tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_context.py::test_reviewer_envelope_uses_explicit_model_parameters tests/test_cli_smoke.py::test_cli_review_with_fake_reviewer_writes_reviewer_artifacts tests/test_cli_smoke.py::test_cli_agent_loop_openai_compatible_uses_adapter_factory -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 7: Run broader reviewer path tests**

Run:

```powershell
python -m pytest tests/test_context.py tests/test_reviewer.py tests/test_agent_loop.py tests/test_orchestrator.py tests/test_cli_smoke.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

Run:

```powershell
git add src/review_agent/context.py src/review_agent/reviewer.py src/review_agent/agent_loop.py src/review_agent/orchestrator.py src/review_agent/cli.py tests/test_context.py tests/test_cli_smoke.py
git commit -m "feat: thread reviewer model into context envelopes"
```

---

## Task 4: Final verification

**Files:**
- Modify tests only if verification reveals a real mismatch.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/test_context.py tests/test_reviewer.py tests/test_agent_loop.py tests/test_orchestrator.py tests/test_cli_smoke.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 2: Run full suite**

Run:

```powershell
python -m pytest -q -p no:cacheprovider
```

Expected: PASS. If Windows emits the known pytest temporary-directory cleanup warning after the pass summary, record it and do not change feature code for that warning.

- [ ] **Step 3: Manual CLI smoke**

Run:

```powershell
$env:PYTHONPATH='src'; python -c "from review_agent.cli import main; raise SystemExit(main(['review','--repo','.','--base','HEAD~1','--head','HEAD','--reviewer-provider','fake','--non-interactive']))"
```

Expected output includes:

```text
Preflight
Final risk:
Review brief:
Review brief JSON:
```

Then inspect the generated fake reviewer envelope:

```powershell
$run = Get-ChildItem -LiteralPath '.review-agent/runs' | Sort-Object LastWriteTime | Select-Object -Last 1; Select-String -LiteralPath ($run.FullName + '\reviewer_envelope.json') -Pattern '"context"|"model"'
```

Expected output includes:

```text
"model": "fake-reviewer"
"context":
```

- [ ] **Step 4: Clean generated artifacts**

Run:

```powershell
$root = (Resolve-Path -LiteralPath 'D:\Agent\code review agent').Path; $paths = @('D:\Agent\code review agent\.review-agent', 'D:\Agent\code review agent\src\review_agent\__pycache__', 'D:\Agent\code review agent\tests\__pycache__'); foreach ($path in $paths) { $resolved = Resolve-Path -LiteralPath $path -ErrorAction SilentlyContinue; if ($resolved -and $resolved.Path.StartsWith($root)) { Remove-Item -LiteralPath $resolved.Path -Recurse -Force } }
```

---

## Completion checklist

- Reviewer envelopes still contain only `system`, `tools`, `messages`, and `parameters`.
- `messages` are assembled through `build_reviewer_context_payload(...)`.
- Context metadata records `budget_scope = messages_only`.
- Context metadata records included, compressed, and omitted sections.
- Long code snippets and observation summaries are compacted before entering `messages`.
- `system`, `tools`, and `parameters` are explicitly excluded from context budget metadata.
- CLI fake reviewer envelopes use `model = fake-reviewer`.
- CLI openai-compatible reviewer envelopes use the configured reviewer model.
- Reviewers still do not receive abstract `risk_level` in `messages`.
- Full tests pass.

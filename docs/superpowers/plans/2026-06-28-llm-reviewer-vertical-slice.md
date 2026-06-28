# LLM Reviewer Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first real model-backed reviewer path: a provider interface, fake provider tests, OpenAI-compatible HTTP provider, single reviewer runner, CLI flags, artifacts, and report integration.

**Architecture:** The Runtime still builds deterministic request, intent, risk, assignment, and context objects. A new provider layer accepts a `ModelInvocationEnvelope` and returns raw model text. A new single reviewer runner calls one reviewer assignment, parses a strict JSON reviewer result, stores it as an artifact, and renders it in the Review Brief without implementing tool calls, repository intelligence, multi-agent scheduling, or reconciliation.

**Tech Stack:** Python 3.11+, stdlib `dataclasses`, `json`, `os`, `urllib.request`, `urllib.error`, `argparse`, `pathlib`, `pytest`, Git CLI. No new third-party dependencies.

---

## Scope and sequencing

This is the next vertical slice after the M1 schema alignment. It proves the project can call a reviewer model and persist a structured reviewer result.

In scope:

- Provider protocol: `ModelProvider.complete(envelope) -> ModelResponse`.
- `FakeProvider` for deterministic tests and local smoke.
- `OpenAICompatibleProvider` using a Chat Completions-compatible HTTP endpoint.
- Provider config loaded from CLI flags and environment variables.
- Reviewer result models and JSON parser.
- Single reviewer runner that uses the first Runtime assignment.
- CLI flags to enable reviewer execution.
- Artifacts:
  - `reviewer_envelope.json`
  - `reviewer_raw_response.json`
  - `reviewer_result.json`
- Markdown report section for single reviewer output.

Out of scope:

- Tool Gateway and tool-call execution.
- Observation Store beyond the existing observation summaries passed into context.
- Repository Intelligence, AST, ripgrep, or LSP expansion.
- Multi-Agent Orchestrator.
- Evidence Reconciler and Completion Checker.
- Eval Harness and SWE-PRBench adapter.

## Current code context

Existing modules:

- `src/review_agent/models.py`: core dataclasses and enums.
- `src/review_agent/context.py`: builds `ModelInvocationEnvelope` with `system`, `tools`, `messages`, `parameters`.
- `src/review_agent/runtime.py`: builds reviewer assignments from risk.
- `src/review_agent/cli.py`: orchestrates local foundation review and writes artifacts.
- `src/review_agent/reporting.py`: renders Markdown Review Brief.

New modules:

- `src/review_agent/provider.py`: provider protocol, fake provider, OpenAI-compatible provider, provider config, and provider errors.
- `src/review_agent/reviewer.py`: reviewer result dataclasses, parser, and single reviewer runner.

New tests:

- `tests/test_provider.py`
- `tests/test_reviewer.py`

Modified tests:

- `tests/test_cli_smoke.py`
- `tests/test_checkpoint_reporting.py`

---

## Task 1: Reviewer result models and parser

**Files:**
- Modify: `src/review_agent/models.py`
- Create: `src/review_agent/reviewer.py`
- Create: `tests/test_reviewer.py`

- [ ] **Step 1: Write failing reviewer parser tests**

Create `tests/test_reviewer.py`:

```python
import pytest

from review_agent.models import ReviewerResultStatus
from review_agent.reviewer import ReviewerResultParseError, parse_reviewer_result


def test_parse_reviewer_result_accepts_valid_json():
    result = parse_reviewer_result(
        """
        {
          "contract_assessments": [
            {
              "contract": "intent_alignment",
              "status": "covered",
              "summary": "The change matches the stated intent.",
              "evidence_refs": ["O-diff-auth"]
            }
          ],
          "confirmed_findings": [
            {
              "claim": "The admin check now always returns true.",
              "severity": "high",
              "confidence": "high",
              "evidence_refs": ["O-diff-auth"],
              "suggested_action": "Restore the role check."
            }
          ],
          "rejected_hypotheses": ["No caller compatibility issue found in the provided context."],
          "uncertainties": ["No repository-wide caller search was available in this slice."],
          "observation_refs": ["O-diff-auth"],
          "investigation_summary": "Reviewed the assignment, intent, diff excerpt, and observations.",
          "status": "completed"
        }
        """
    )

    assert result.status is ReviewerResultStatus.COMPLETED
    assert result.confirmed_findings[0].claim == "The admin check now always returns true."
    assert result.contract_assessments[0].evidence_refs == ["O-diff-auth"]
    assert result.observation_refs == ["O-diff-auth"]


def test_parse_reviewer_result_strips_markdown_json_fence():
    result = parse_reviewer_result(
        """```json
        {
          "contract_assessments": [],
          "confirmed_findings": [],
          "rejected_hypotheses": [],
          "uncertainties": ["needs broader repository context"],
          "observation_refs": [],
          "investigation_summary": "No finding.",
          "status": "partial"
        }
        ```"""
    )

    assert result.status is ReviewerResultStatus.PARTIAL
    assert result.uncertainties == ["needs broader repository context"]


def test_parse_reviewer_result_rejects_missing_required_keys():
    with pytest.raises(ReviewerResultParseError, match="missing required key: status"):
        parse_reviewer_result('{"confirmed_findings": []}')


def test_parse_reviewer_result_rejects_invalid_status():
    with pytest.raises(ReviewerResultParseError, match="invalid reviewer status"):
        parse_reviewer_result(
            """
            {
              "contract_assessments": [],
              "confirmed_findings": [],
              "rejected_hypotheses": [],
              "uncertainties": [],
              "observation_refs": [],
              "investigation_summary": "Bad status.",
              "status": "done"
            }
            """
        )
```

- [ ] **Step 2: Run reviewer tests and verify they fail**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_reviewer.py -q -p no:cacheprovider
```

Expected: fail with `ModuleNotFoundError` or missing `ReviewerResultStatus`.

- [ ] **Step 3: Add reviewer result models**

In `src/review_agent/models.py`, add:

```python
class ReviewerResultStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class ContractAssessment:
    contract: str
    status: ContractItemStatus
    summary: str
    evidence_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewerFinding:
    claim: str
    severity: str
    confidence: str
    evidence_refs: list[str] = field(default_factory=list)
    suggested_action: str | None = None


@dataclass(frozen=True)
class ReviewerResult:
    contract_assessments: list[ContractAssessment] = field(default_factory=list)
    confirmed_findings: list[ReviewerFinding] = field(default_factory=list)
    rejected_hypotheses: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    observation_refs: list[str] = field(default_factory=list)
    investigation_summary: str = ""
    status: ReviewerResultStatus = ReviewerResultStatus.PARTIAL
```

- [ ] **Step 4: Implement reviewer parser**

Create `src/review_agent/reviewer.py` with:

```python
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from review_agent.models import (
    Assignment,
    ContractAssessment,
    ContractItemStatus,
    IntentPacket,
    ModelInvocationEnvelope,
    ReviewerFinding,
    ReviewerResult,
    ReviewerResultStatus,
)


class ReviewerResultParseError(ValueError):
    pass


REQUIRED_RESULT_KEYS = (
    "contract_assessments",
    "confirmed_findings",
    "rejected_hypotheses",
    "uncertainties",
    "observation_refs",
    "investigation_summary",
    "status",
)


def parse_reviewer_result(raw_text: str) -> ReviewerResult:
    payload = _loads_json_object(_strip_json_fence(raw_text))
    for key in REQUIRED_RESULT_KEYS:
        if key not in payload:
            raise ReviewerResultParseError(f"missing required key: {key}")

    try:
        status = ReviewerResultStatus(payload["status"])
    except ValueError as error:
        raise ReviewerResultParseError(f"invalid reviewer status: {payload['status']}") from error

    return ReviewerResult(
        contract_assessments=[_parse_contract_assessment(item) for item in payload["contract_assessments"]],
        confirmed_findings=[_parse_finding(item) for item in payload["confirmed_findings"]],
        rejected_hypotheses=[str(item) for item in payload["rejected_hypotheses"]],
        uncertainties=[str(item) for item in payload["uncertainties"]],
        observation_refs=[str(item) for item in payload["observation_refs"]],
        investigation_summary=str(payload["investigation_summary"]),
        status=status,
    )


def reviewer_result_to_dict(result: ReviewerResult) -> dict[str, Any]:
    return asdict(result)


def _strip_json_fence(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()
    return text


def _loads_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ReviewerResultParseError(f"invalid reviewer JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ReviewerResultParseError("reviewer JSON must be an object")
    return payload


def _parse_contract_assessment(item: Any) -> ContractAssessment:
    if not isinstance(item, dict):
        raise ReviewerResultParseError("contract assessment must be an object")
    try:
        status = ContractItemStatus(item["status"])
    except KeyError as error:
        raise ReviewerResultParseError("contract assessment missing required key: status") from error
    except ValueError as error:
        raise ReviewerResultParseError(f"invalid contract status: {item.get('status')}") from error
    return ContractAssessment(
        contract=str(item.get("contract", "")),
        status=status,
        summary=str(item.get("summary", "")),
        evidence_refs=[str(ref) for ref in item.get("evidence_refs", [])],
    )


def _parse_finding(item: Any) -> ReviewerFinding:
    if not isinstance(item, dict):
        raise ReviewerResultParseError("finding must be an object")
    return ReviewerFinding(
        claim=str(item.get("claim", "")),
        severity=str(item.get("severity", "")),
        confidence=str(item.get("confidence", "")),
        evidence_refs=[str(ref) for ref in item.get("evidence_refs", [])],
        suggested_action=item.get("suggested_action"),
    )
```

Unused imports are allowed temporarily in this task because Task 3 will add the single reviewer runner to the same file. If the linter complains, remove the unused names and re-add them in Task 3.

- [ ] **Step 5: Run reviewer tests and verify they pass**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_reviewer.py -q -p no:cacheprovider
```

Expected: `4 passed`.

- [ ] **Step 6: Commit Task 1**

Run:

```powershell
git add src/review_agent/models.py src/review_agent/reviewer.py tests/test_reviewer.py
git commit -m "feat: parse structured reviewer results"
```

---

## Task 2: Model provider interface and providers

**Files:**
- Create: `src/review_agent/provider.py`
- Create: `tests/test_provider.py`

- [ ] **Step 1: Write failing provider tests**

Create `tests/test_provider.py`:

```python
import json
import os
import urllib.error

import pytest

from review_agent.models import ModelInvocationEnvelope
from review_agent.provider import (
    FakeProvider,
    ModelProviderError,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    ProviderConfigError,
    build_provider_from_config,
)


def make_envelope() -> ModelInvocationEnvelope:
    return ModelInvocationEnvelope(
        system="system rules",
        tools=[],
        messages=[{"role": "user", "content": "review this"}],
        parameters={
            "model": "test-model",
            "max_output_tokens": 256,
            "temperature": 0,
            "trace_id": "trace-1",
        },
    )


def test_fake_provider_returns_configured_text():
    provider = FakeProvider('{"status":"completed"}')

    response = provider.complete(make_envelope())

    assert response.content == '{"status":"completed"}'
    assert response.provider_name == "fake"
    assert response.model == "fake-reviewer"


def test_build_provider_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("REVIEW_AGENT_API_KEY", raising=False)

    with pytest.raises(ProviderConfigError, match="REVIEW_AGENT_API_KEY"):
        build_provider_from_config(
            provider_name="openai-compatible",
            model="review-model",
            base_url="https://example.test/v1",
            api_key_env="REVIEW_AGENT_API_KEY",
        )


def test_openai_compatible_provider_builds_expected_payload():
    captured = {}

    def fake_transport(url, headers, payload, timeout_seconds):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        captured["timeout_seconds"] = timeout_seconds
        return {
            "choices": [
                {
                    "message": {
                        "content": "{\"status\":\"completed\"}"
                    }
                }
            ],
            "usage": {"total_tokens": 12},
        }

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret-key",
            model="review-model",
            timeout_seconds=7,
        ),
        transport=fake_transport,
    )

    response = provider.complete(make_envelope())

    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["payload"]["model"] == "review-model"
    assert captured["payload"]["messages"][0] == {"role": "system", "content": "system rules"}
    assert captured["payload"]["messages"][1] == {"role": "user", "content": "review this"}
    assert captured["payload"]["max_tokens"] == 256
    assert captured["payload"]["temperature"] == 0
    assert captured["timeout_seconds"] == 7
    assert response.content == "{\"status\":\"completed\"}"
    assert response.raw["usage"]["total_tokens"] == 12


def test_openai_compatible_provider_wraps_transport_errors():
    def failing_transport(url, headers, payload, timeout_seconds):
        raise urllib.error.URLError("connection refused")

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret-key",
            model="review-model",
        ),
        transport=failing_transport,
    )

    with pytest.raises(ModelProviderError, match="provider request failed"):
        provider.complete(make_envelope())
```

- [ ] **Step 2: Run provider tests and verify they fail**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_provider.py -q -p no:cacheprovider
```

Expected: fail because `review_agent.provider` does not exist.

- [ ] **Step 3: Implement provider module**

Create `src/review_agent/provider.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
import json
import os
import urllib.error
import urllib.request

from review_agent.models import ModelInvocationEnvelope


class ModelProviderError(RuntimeError):
    pass


class ProviderConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelResponse:
    content: str
    provider_name: str
    model: str
    raw: dict[str, Any] = field(default_factory=dict)


class ModelProvider(Protocol):
    def complete(self, envelope: ModelInvocationEnvelope) -> ModelResponse:
        raise NotImplementedError


class FakeProvider:
    def __init__(self, content: str, model: str = "fake-reviewer") -> None:
        self._content = content
        self._model = model

    def complete(self, envelope: ModelInvocationEnvelope) -> ModelResponse:
        return ModelResponse(
            content=self._content,
            provider_name="fake",
            model=self._model,
            raw={"trace_id": envelope.parameters.get("trace_id")},
        )


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 60


Transport = Callable[[str, dict[str, str], dict[str, Any], int], dict[str, Any]]


class OpenAICompatibleProvider:
    def __init__(self, config: OpenAICompatibleConfig, transport: Transport | None = None) -> None:
        self._config = config
        self._transport = transport or _urllib_transport

    def complete(self, envelope: ModelInvocationEnvelope) -> ModelResponse:
        payload = _build_chat_payload(self._config.model, envelope)
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        try:
            raw = self._transport(url, headers, payload, self._config.timeout_seconds)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ModelProviderError(f"provider request failed: {error}") from error

        content = _extract_chat_content(raw)
        return ModelResponse(
            content=content,
            provider_name="openai-compatible",
            model=self._config.model,
            raw=raw,
        )


def build_provider_from_config(
    provider_name: str | None,
    model: str | None,
    base_url: str | None,
    api_key_env: str,
) -> ModelProvider | None:
    if provider_name in (None, "none"):
        return None
    if provider_name == "fake":
        return FakeProvider(_fake_reviewer_result_json())
    if provider_name == "openai-compatible":
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ProviderConfigError(f"missing API key environment variable: {api_key_env}")
        if not model:
            raise ProviderConfigError("--reviewer-model is required for openai-compatible provider")
        if not base_url:
            raise ProviderConfigError("--reviewer-base-url is required for openai-compatible provider")
        return OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
            )
        )
    raise ProviderConfigError(f"unsupported reviewer provider: {provider_name}")


def _build_chat_payload(model: str, envelope: ModelInvocationEnvelope) -> dict[str, Any]:
    messages = [{"role": "system", "content": envelope.system}]
    messages.extend(envelope.messages)
    return {
        "model": model,
        "messages": messages,
        "max_tokens": envelope.parameters.get("max_output_tokens", 4096),
        "temperature": envelope.parameters.get("temperature", 0),
    }


def _extract_chat_content(raw: dict[str, Any]) -> str:
    try:
        return str(raw["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as error:
        raise ModelProviderError("provider response did not contain choices[0].message.content") from error


def _urllib_transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _fake_reviewer_result_json() -> str:
    return json.dumps(
        {
            "contract_assessments": [],
            "confirmed_findings": [],
            "rejected_hypotheses": [],
            "uncertainties": ["Fake provider does not perform semantic review."],
            "observation_refs": [],
            "investigation_summary": "Fake reviewer executed.",
            "status": "partial",
        }
    )
```

- [ ] **Step 4: Run provider tests and verify they pass**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_provider.py -q -p no:cacheprovider
```

Expected: `4 passed`.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src/review_agent/provider.py tests/test_provider.py
git commit -m "feat: add model provider interface"
```

---

## Task 3: Single reviewer runner

**Files:**
- Modify: `src/review_agent/reviewer.py`
- Modify: `tests/test_reviewer.py`

- [ ] **Step 1: Add failing single reviewer tests**

Append to `tests/test_reviewer.py`:

```python
from review_agent.models import Assignment, InitialContext, IntentPacket, IntentSource, IntentStatus
from review_agent.provider import FakeProvider
from review_agent.reviewer import run_single_reviewer


def make_assignment() -> Assignment:
    return Assignment(
        role="Core Reviewer",
        mission="Check intent alignment",
        assignment_reason=["sensitive path changed: auth.py"],
        assigned_contract=["intent_alignment"],
        required_checks=["map changed behavior to intent"],
        initial_context=InitialContext(
            changed_files=["auth.py"],
            diff_ranges=["auth.py"],
            observation_refs=["O-diff-auth"],
        ),
        max_turns=6,
        max_tool_calls=12,
    )


def make_intent() -> IntentPacket:
    return IntentPacket(
        goal="Refactor admin check",
        sources={"goal": IntentSource.EXPLICIT},
        status=IntentStatus.PARTIAL,
        uncertainties=["acceptance criteria are not explicitly declared"],
    )


def test_run_single_reviewer_calls_provider_and_parses_result():
    provider = FakeProvider(
        """
        {
          "contract_assessments": [],
          "confirmed_findings": [],
          "rejected_hypotheses": [],
          "uncertainties": ["No tool gateway available."],
          "observation_refs": ["O-diff-auth"],
          "investigation_summary": "Reviewed diff excerpt.",
          "status": "partial"
        }
        """
    )

    run = run_single_reviewer(
        provider=provider,
        assignment=make_assignment(),
        intent=make_intent(),
        diff_excerpt=["-    return user.role == 'admin'", "+    return True"],
        observations={"O-diff-auth": "auth.py changed between base and head"},
        trace_id="trace-reviewer-1",
    )

    assert run.result.status is ReviewerResultStatus.PARTIAL
    assert run.result.observation_refs == ["O-diff-auth"]
    assert run.response.provider_name == "fake"
    assert run.envelope.parameters["trace_id"] == "trace-reviewer-1"
    assert "return True" in run.envelope.messages[0]["content"]


def test_run_single_reviewer_uses_diff_excerpt_as_code_snippet():
    provider = FakeProvider(
        """
        {
          "contract_assessments": [],
          "confirmed_findings": [],
          "rejected_hypotheses": [],
          "uncertainties": [],
          "observation_refs": [],
          "investigation_summary": "Reviewed diff.",
          "status": "completed"
        }
        """
    )

    run = run_single_reviewer(
        provider=provider,
        assignment=make_assignment(),
        intent=make_intent(),
        diff_excerpt=["+changed"],
        observations={},
        trace_id="trace-reviewer-2",
    )

    assert "Diff Excerpt" in run.envelope.messages[0]["content"]
    assert "+changed" in run.envelope.messages[0]["content"]
```

- [ ] **Step 2: Run reviewer tests and verify new failures**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_reviewer.py -q -p no:cacheprovider
```

Expected: fail because `run_single_reviewer` does not exist.

- [ ] **Step 3: Add reviewer run dataclass and function**

In `src/review_agent/reviewer.py`, add imports:

```python
from dataclasses import dataclass

from review_agent.context import build_reviewer_envelope
from review_agent.provider import ModelProvider, ModelResponse
```

Add:

```python
@dataclass(frozen=True)
class ReviewerRun:
    envelope: ModelInvocationEnvelope
    response: ModelResponse
    result: ReviewerResult


def run_single_reviewer(
    provider: ModelProvider,
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
    response = provider.complete(envelope)
    result = parse_reviewer_result(response.content)
    return ReviewerRun(envelope=envelope, response=response, result=result)
```

- [ ] **Step 4: Run reviewer tests and verify they pass**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_reviewer.py -q -p no:cacheprovider
```

Expected: all reviewer tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add src/review_agent/reviewer.py tests/test_reviewer.py
git commit -m "feat: run single reviewer with provider"
```

---

## Task 4: CLI reviewer execution and artifacts

**Files:**
- Modify: `src/review_agent/cli.py`
- Modify: `tests/test_cli_smoke.py`

- [ ] **Step 1: Add failing CLI fake reviewer smoke test**

Append to `tests/test_cli_smoke.py`:

```python
def test_cli_review_with_fake_reviewer_writes_reviewer_artifacts(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text("def check(token):\n    return token == 'ok'\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
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
            "Add auth token check",
            "--reviewer-provider",
            "fake",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_root = git_repo / ".review-agent" / "runs"
    run_dirs = sorted(run_root.iterdir())
    run_dir = run_dirs[-1]

    assert (run_dir / "reviewer_envelope.json").exists()
    assert (run_dir / "reviewer_raw_response.json").exists()
    assert (run_dir / "reviewer_result.json").exists()

    result = json.loads((run_dir / "reviewer_result.json").read_text(encoding="utf-8"))
    raw = json.loads((run_dir / "reviewer_raw_response.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")

    assert result["status"] == "partial"
    assert raw["provider_name"] == "fake"
    assert "## Single Reviewer Result" in report
    assert "Fake reviewer executed." in report
```

- [ ] **Step 2: Run CLI smoke tests and verify failure**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_cli_smoke.py -q -p no:cacheprovider
```

Expected: fail because `--reviewer-provider` is not a known argument.

- [ ] **Step 3: Add CLI flags**

In `src/review_agent/cli.py`, add parser arguments:

```python
review.add_argument("--reviewer-provider", choices=["none", "fake", "openai-compatible"], default="none")
review.add_argument("--reviewer-model")
review.add_argument("--reviewer-base-url")
review.add_argument("--reviewer-api-key-env", default="REVIEW_AGENT_API_KEY")
```

- [ ] **Step 4: Run single CLI smoke test and verify provider config failure changes**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_cli_smoke.py::test_cli_review_with_fake_reviewer_writes_reviewer_artifacts -q -p no:cacheprovider
```

Expected: fail later in `_run_review` because reviewer artifacts are not written yet.

- [ ] **Step 5: Wire fake reviewer into CLI**

In `src/review_agent/cli.py`, add imports:

```python
from review_agent.provider import ProviderConfigError, build_provider_from_config
from review_agent.reviewer import reviewer_result_to_dict, run_single_reviewer
```

After `assignments = build_assignments(risk_assessment)`, add:

```python
    provider = build_provider_from_config(
        provider_name=args.reviewer_provider,
        model=args.reviewer_model,
        base_url=args.reviewer_base_url,
        api_key_env=args.reviewer_api_key_env,
    )
```

After writing the existing artifacts, add:

```python
    reviewer_result = None
    if provider is not None and assignments:
        reviewer_run = run_single_reviewer(
            provider=provider,
            assignment=assignments[0],
            intent=intent,
            diff_excerpt=change_summary.diff_excerpt,
            observations={
                ref: ref for ref in assignments[0].initial_context.observation_refs
            },
            trace_id=f"{review_id}-reviewer-0",
        )
        reviewer_result = reviewer_run.result
        store.write_json("reviewer_envelope.json", asdict(reviewer_run.envelope))
        store.write_json(
            "reviewer_raw_response.json",
            {
                "provider_name": reviewer_run.response.provider_name,
                "model": reviewer_run.response.model,
                "content": reviewer_run.response.content,
                "raw": reviewer_run.response.raw,
            },
        )
        store.write_json("reviewer_result.json", reviewer_result_to_dict(reviewer_run.result))
```

Catch provider config errors near the provider construction:

```python
    try:
        provider = build_provider_from_config(...)
    except ProviderConfigError as error:
        print(f"Reviewer provider configuration error: {error}")
        return 2
```

Pass `reviewer_result` into report rendering in Task 5. For this task, write reviewer artifacts first and keep report integration failing until Task 5 if the test is not yet checking report content. If the test already checks the report section, implement Task 5 changes in the same red/green loop and commit both tasks together.

- [ ] **Step 6: Run CLI smoke tests and verify artifact creation**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_cli_smoke.py -q -p no:cacheprovider
```

Expected: all CLI smoke tests pass after Task 5 report integration is added. If report integration is still missing, proceed directly to Task 5 before committing.

---

## Task 5: Report integration for single reviewer result

**Files:**
- Modify: `src/review_agent/reporting.py`
- Modify: `src/review_agent/cli.py`
- Modify: `tests/test_checkpoint_reporting.py`
- Modify: `tests/test_cli_smoke.py`

- [ ] **Step 1: Add failing report unit test**

Append to `tests/test_checkpoint_reporting.py`:

```python
from review_agent.models import ReviewerResult, ReviewerResultStatus


def test_markdown_report_includes_single_reviewer_result():
    assessment = RiskAssessment(
        level=RiskLevel.LOW,
        dimensions={"impact": "local"},
        reasons=["small change"],
        signal_refs=[],
        uncertainties=[],
        suggested_focus=["intent alignment"],
    )
    reviewer_result = ReviewerResult(
        investigation_summary="Fake reviewer executed.",
        status=ReviewerResultStatus.PARTIAL,
        uncertainties=["Fake provider does not perform semantic review."],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        risk_assessment=assessment,
        changed_files=["auth.py"],
        reviewer_result=reviewer_result,
    )

    assert "## Single Reviewer Result" in report
    assert "Status: partial" in report
    assert "Fake reviewer executed." in report
    assert "- Fake provider does not perform semantic review." in report
```

- [ ] **Step 2: Run reporting tests and verify failure**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_checkpoint_reporting.py -q -p no:cacheprovider
```

Expected: fail because `render_markdown_report` does not accept `reviewer_result`.

- [ ] **Step 3: Update report rendering**

In `src/review_agent/reporting.py`, update signature:

```python
from review_agent.models import ReviewerResult, RiskAssessment


def render_markdown_report(
    review_id: str,
    base_revision: str,
    head_revision: str,
    risk_assessment: RiskAssessment,
    changed_files: list[str],
    reviewer_result: ReviewerResult | None = None,
) -> str:
```

Before the final non-binding recommendation section, insert:

```python
            *_reviewer_result_section(reviewer_result),
```

Add helper:

```python
def _reviewer_result_section(reviewer_result: ReviewerResult | None) -> list[str]:
    if reviewer_result is None:
        return []
    findings = (
        "\n".join(f"- {finding.claim}" for finding in reviewer_result.confirmed_findings)
        or "- No confirmed findings reported by the single reviewer"
    )
    uncertainties = (
        "\n".join(f"- {uncertainty}" for uncertainty in reviewer_result.uncertainties)
        or "- No reviewer uncertainties recorded"
    )
    return [
        "## Single Reviewer Result",
        "",
        f"Status: {reviewer_result.status.value}",
        f"Summary: {reviewer_result.investigation_summary}",
        "",
        "### Reviewer Findings",
        "",
        findings,
        "",
        "### Reviewer Uncertainties",
        "",
        uncertainties,
        "",
    ]
```

- [ ] **Step 4: Pass reviewer result from CLI to report**

In `src/review_agent/cli.py`, initialize before provider execution:

```python
    reviewer_result = None
```

Pass it into the report call:

```python
    report = render_markdown_report(
        review_id=review_id,
        base_revision=args.base,
        head_revision=args.head,
        risk_assessment=risk_assessment,
        changed_files=change_summary.changed_files,
        reviewer_result=reviewer_result,
    )
```

- [ ] **Step 5: Run reporting and CLI tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_checkpoint_reporting.py tests/test_cli_smoke.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Tasks 4 and 5**

Run:

```powershell
git add src/review_agent/cli.py src/review_agent/reporting.py tests/test_cli_smoke.py tests/test_checkpoint_reporting.py
git commit -m "feat: add single reviewer CLI path"
```

---

## Task 6: Provider configuration failure test and final verification

**Files:**
- Modify: `tests/test_cli_smoke.py`
- Modify only failing implementation files if tests reveal a bug.

- [ ] **Step 1: Add CLI provider config failure test**

Append to `tests/test_cli_smoke.py`:

```python
def test_cli_openai_compatible_provider_requires_api_key(git_repo: Path, monkeypatch, capsys):
    monkeypatch.delenv("REVIEW_AGENT_API_KEY", raising=False)
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
            "--reviewer-provider",
            "openai-compatible",
            "--reviewer-model",
            "review-model",
            "--reviewer-base-url",
            "https://example.test/v1",
            "--non-interactive",
        ]
    )

    assert exit_code == 2
    assert "Reviewer provider configuration error" in capsys.readouterr().out
```

- [ ] **Step 2: Run CLI smoke tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_cli_smoke.py -q -p no:cacheprovider
```

Expected: pass if Task 4 handled provider config errors correctly. If it fails, adjust `src/review_agent/cli.py` to catch `ProviderConfigError` and return `2`.

- [ ] **Step 3: Search for real-network calls in tests**

Run:

```powershell
rg -n "urlopen|openai-compatible|REVIEW_AGENT_API_KEY" tests src
```

Expected:

- `urllib.request.urlopen` appears only in `src/review_agent/provider.py`.
- Tests use injected transport or missing-key config errors.
- No test requires a real API key.

- [ ] **Step 4: Run full test suite**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 5: Run CLI fake reviewer manual smoke**

Run a local fake-provider smoke:

```powershell
$sample = Join-Path 'C:\tmp' ('review-agent-llm-slice-smoke-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $sample | Out-Null
Push-Location $sample
git init | Out-Null
git config user.email 'test@example.com'
git config user.name 'Test User'
Set-Content -Encoding UTF8 -Path 'auth.py' -Value "def is_admin(user):`n    return user.role == 'admin'`n"
git add .
git commit -m 'base' | Out-Null
$base = git rev-parse HEAD
Set-Content -Encoding UTF8 -Path 'auth.py' -Value "def is_admin(user):`n    return True`n"
git add .
git commit -m 'head' | Out-Null
$head = git rev-parse HEAD
Pop-Location
$env:PYTHONPATH=(Resolve-Path .\src).Path
python -m review_agent review --repo $sample --base $base --head $head --intent 'Refactor admin check' --focus 'authorization regression' --reviewer-provider fake --non-interactive
```

Expected:

- Command prints `Review foundation completed: ...`.
- Run directory contains `reviewer_envelope.json`, `reviewer_raw_response.json`, and `reviewer_result.json`.
- Report contains `## Single Reviewer Result`.

- [ ] **Step 6: Commit Task 6 if changes were needed**

If Step 2 required a fix, run:

```powershell
git add src/review_agent/cli.py tests/test_cli_smoke.py
git commit -m "test: cover reviewer provider configuration errors"
```

If Step 2 passed without code changes, commit only the new CLI test:

```powershell
git add tests/test_cli_smoke.py
git commit -m "test: cover reviewer provider configuration errors"
```

---

## Manual real-provider smoke

This smoke is optional and must not run in CI. It is only for a developer who has an API key and a Chat Completions-compatible endpoint.

```powershell
$env:REVIEW_AGENT_API_KEY = "<real key>"
$env:PYTHONPATH=(Resolve-Path .\src).Path
python -m review_agent review `
  --repo "C:\path\to\sample\repo" `
  --base "<base-sha>" `
  --head "<head-sha>" `
  --intent "Describe the change intent" `
  --focus "correctness and regression safety" `
  --reviewer-provider openai-compatible `
  --reviewer-model "<model-name>" `
  --reviewer-base-url "https://api.example.com/v1" `
  --non-interactive
```

Expected:

- Run directory contains reviewer artifacts.
- `reviewer_raw_response.json` contains provider metadata and raw response.
- `reviewer_result.json` contains parsed structured result.

If the provider returns non-JSON reviewer text, the command should fail with a clear parse error. A later plan can add retry/repair behavior.

## Self-review checklist

- Spec coverage:
  - Context envelope reused: Tasks 3 and 4.
  - Provider configuration and model parameters: Task 2.
  - Single reviewer invocation: Task 3.
  - Structured reviewer result: Task 1.
  - Artifact persistence: Task 4.
  - Review Brief integration: Task 5.
  - No real network in tests: Task 2 and Task 6.
- Type consistency:
  - `ReviewerResultStatus` values: `completed`, `partial`, `blocked`, `failed`.
  - `ReviewerResult` parser and report renderer both use the same dataclasses.
  - `ModelProvider.complete()` always returns `ModelResponse`.
  - CLI passes a `ReviewerResult | None` into `render_markdown_report`.
- Scope check:
  - Tool Gateway, Observation Store, Repository Intelligence, Multi-Agent Orchestrator, Reconciler, Completion Checker, and Eval Harness are intentionally outside this plan.

## Execution handoff

Plan complete when this file is saved. Recommended execution is Subagent-Driven if quota is available. Inline execution is acceptable because the plan is a contained vertical slice with clear TDD checkpoints.


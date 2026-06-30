# Tool Gateway And Observation Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only Tool Gateway and persistent Observation Store so deterministic tool reads create stable observation IDs, raw artifacts, and context summaries for the single reviewer path.

**Architecture:** Keep this as a vertical slice, not a full agent loop. `ObservationStore` owns stable IDs, JSONL persistence, raw artifacts, and summary lookup. `ToolGateway` exposes read-only `read_range`, `compare_base_head`, and simple `search_code` calls scoped to the request `base_revision` and `head_revision`; CLI uses it to pre-record changed-file diff observations before invoking the single reviewer.

**Tech Stack:** Python 3.11+, stdlib `dataclasses`, `hashlib`, `json`, `pathlib`, `subprocess`, `pytest`, Git CLI. No new third-party dependencies.

---

## Scope

In scope:

- Persistent `observations.jsonl`.
- Raw observation artifacts under `observations/<observation_id>.txt`.
- Stable observation IDs derived from source, revision, path, line range, and content hash.
- Context-view summaries passed to the reviewer.
- Read-only Tool Gateway with:
  - `read_range`
  - `compare_base_head`
  - `search_code`
- Path and revision safety checks.
- CLI fake-reviewer path records changed-file diff observations and writes observation artifacts.

Out of scope:

- Model-driven tool-call loop.
- OpenAI/Claude function-call protocol mapping.
- AST, ripgrep, LSP, symbol intelligence, call graph, or test mapping.
- Multi-agent orchestration.
- Evidence Reconciler and Completion Checker.
- Eval harness.

## File Structure

- Create `src/review_agent/observations.py`
  - `Observation` dataclass
  - `ObservationStore`
  - stable ID and content hash helpers
- Create `src/review_agent/tool_gateway.py`
  - `ToolGateway`
  - `ToolGatewayError`
  - `ToolExecutionResult`
  - safe Git read/search helpers
- Modify `src/review_agent/cli.py`
  - create `ObservationStore`
  - create `ToolGateway`
  - record compare observations for changed files in fake/real reviewer mode
  - pass stored observation summaries into `run_single_reviewer`
  - write `observations.jsonl` and raw artifact files through the store
- Modify `src/review_agent/reporting.py`
  - optional observation section with recorded observation IDs and summaries
- Create `tests/test_observations.py`
- Create `tests/test_tool_gateway.py`
- Modify `tests/test_cli_smoke.py`
- Modify `tests/test_checkpoint_reporting.py`

---

## Task 1: Observation Store

**Files:**
- Create: `src/review_agent/observations.py`
- Create: `tests/test_observations.py`

- [ ] **Step 1: Write failing observation store tests**

Create `tests/test_observations.py`:

```python
from pathlib import Path
import json

from review_agent.observations import ObservationStore


def test_observation_store_records_stable_id_and_raw_artifact(tmp_path: Path):
    store = ObservationStore(tmp_path)

    first = store.record(
        source="git.read_range",
        revision="head@abc",
        path="src/auth.py",
        line_start=1,
        line_end=2,
        raw_content="def check():\n    return True\n",
        context_view="src/auth.py:1-2 changed",
    )
    second = store.record(
        source="git.read_range",
        revision="head@abc",
        path="src/auth.py",
        line_start=1,
        line_end=2,
        raw_content="def check():\n    return True\n",
        context_view="src/auth.py:1-2 changed",
    )

    assert first.observation_id == second.observation_id
    assert first.content_hash == second.content_hash
    assert (tmp_path / "observations" / f"{first.observation_id}.txt").read_text(encoding="utf-8") == (
        "def check():\n    return True\n"
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["observation_id"] == first.observation_id
    assert records[-1]["raw_artifact_ref"] == f"observations/{first.observation_id}.txt"


def test_observation_store_returns_summary_map(tmp_path: Path):
    store = ObservationStore(tmp_path)
    observation = store.record(
        source="git.compare_base_head",
        revision="base..head",
        path="auth.py",
        line_start=None,
        line_end=None,
        raw_content="diff --git a/auth.py b/auth.py",
        context_view="auth.py changed between base and head",
    )

    assert store.summaries_by_id() == {
        observation.observation_id: "auth.py changed between base and head"
    }
    assert store.list_observations()[0].observation_id == observation.observation_id
```

- [ ] **Step 2: Run observation tests and verify failure**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_observations.py -q -p no:cacheprovider
```

Expected: fail because `review_agent.observations` does not exist.

- [ ] **Step 3: Implement observation store**

Create `src/review_agent/observations.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json


@dataclass(frozen=True)
class Observation:
    observation_id: str
    source: str
    revision: str
    path: str | None
    line_start: int | None
    line_end: int | None
    content_hash: str
    raw_artifact_ref: str
    context_view: str


class ObservationStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.observations_dir = run_dir / "observations"
        self.observations_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = run_dir / "observations.jsonl"
        self._observations: list[Observation] = []

    def record(
        self,
        source: str,
        revision: str,
        path: str | None,
        line_start: int | None,
        line_end: int | None,
        raw_content: str,
        context_view: str,
    ) -> Observation:
        content_hash = _sha256(raw_content)
        observation_id = _observation_id(
            source=source,
            revision=revision,
            path=path,
            line_start=line_start,
            line_end=line_end,
            content_hash=content_hash,
        )
        artifact_ref = f"observations/{observation_id}.txt"
        (self.run_dir / artifact_ref).write_text(raw_content, encoding="utf-8")
        observation = Observation(
            observation_id=observation_id,
            source=source,
            revision=revision,
            path=path,
            line_start=line_start,
            line_end=line_end,
            content_hash=content_hash,
            raw_artifact_ref=artifact_ref,
            context_view=context_view,
        )
        self._observations.append(observation)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(observation), ensure_ascii=False))
            handle.write("\n")
        return observation

    def list_observations(self) -> list[Observation]:
        return list(self._observations)

    def summaries_by_id(self) -> dict[str, str]:
        return {observation.observation_id: observation.context_view for observation in self._observations}


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _observation_id(
    source: str,
    revision: str,
    path: str | None,
    line_start: int | None,
    line_end: int | None,
    content_hash: str,
) -> str:
    seed = "|".join(
        [
            source,
            revision,
            path or "",
            "" if line_start is None else str(line_start),
            "" if line_end is None else str(line_end),
            content_hash,
        ]
    )
    return f"O-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:12]}"
```

- [ ] **Step 4: Run observation tests and verify pass**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_observations.py -q -p no:cacheprovider
```

Expected: `2 passed`.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src/review_agent/observations.py tests/test_observations.py
git commit -m "feat: add observation store"
```

---

## Task 2: Read-only Tool Gateway

**Files:**
- Create: `src/review_agent/tool_gateway.py`
- Create: `tests/test_tool_gateway.py`

- [ ] **Step 1: Write failing tool gateway tests**

Create `tests/test_tool_gateway.py`:

```python
from pathlib import Path

import pytest

from conftest import run_git
from review_agent.observations import ObservationStore
from review_agent.tool_gateway import ToolGateway, ToolGatewayError


def test_tool_gateway_read_range_records_observation(git_repo: Path, tmp_path: Path):
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path)
    gateway = ToolGateway(git_repo, base_revision=head, head_revision=head, observation_store=store)

    result = gateway.execute(
        "read_range",
        {"path": "app.py", "revision": "head", "line_start": 1, "line_end": 2},
    )

    assert len(result.observation_ids) == 1
    assert "def add" in result.context_view
    observation = store.list_observations()[0]
    assert observation.source == "git.read_range"
    assert observation.revision == f"head@{head}"
    assert observation.path == "app.py"
    assert observation.line_start == 1
    assert observation.line_end == 2


def test_tool_gateway_compare_base_head_records_diff(git_repo: Path, tmp_path: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change behavior")
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path)
    gateway = ToolGateway(git_repo, base_revision=base, head_revision=head, observation_store=store)

    result = gateway.execute("compare_base_head", {"path": "app.py"})

    assert len(result.observation_ids) == 1
    assert "-    return a + b" in result.context_view
    assert "+    return a - b" in result.context_view
    assert store.list_observations()[0].revision == f"{base}..{head}"


def test_tool_gateway_search_code_records_matches(git_repo: Path, tmp_path: Path):
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path)
    gateway = ToolGateway(git_repo, base_revision=head, head_revision=head, observation_store=store)

    result = gateway.execute("search_code", {"query": "def add", "revision": "head", "max_results": 5})

    assert len(result.observation_ids) == 1
    assert "app.py:1:def add" in result.context_view
    assert store.list_observations()[0].source == "git.search_code"


def test_tool_gateway_rejects_unsafe_paths(git_repo: Path, tmp_path: Path):
    head = run_git(git_repo, "rev-parse", "HEAD")
    gateway = ToolGateway(git_repo, base_revision=head, head_revision=head, observation_store=ObservationStore(tmp_path))

    with pytest.raises(ToolGatewayError, match="unsafe repository path"):
        gateway.execute("read_range", {"path": "../secret.txt", "revision": "head", "line_start": 1, "line_end": 1})

    with pytest.raises(ToolGatewayError, match="unsafe repository path"):
        gateway.execute("read_range", {"path": ".git/config", "revision": "head", "line_start": 1, "line_end": 1})


def test_tool_gateway_rejects_unauthorized_revision(git_repo: Path, tmp_path: Path):
    head = run_git(git_repo, "rev-parse", "HEAD")
    gateway = ToolGateway(git_repo, base_revision=head, head_revision=head, observation_store=ObservationStore(tmp_path))

    with pytest.raises(ToolGatewayError, match="unauthorized revision"):
        gateway.execute("read_range", {"path": "app.py", "revision": "main", "line_start": 1, "line_end": 1})
```

- [ ] **Step 2: Run tool gateway tests and verify failure**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_tool_gateway.py -q -p no:cacheprovider
```

Expected: fail because `review_agent.tool_gateway` does not exist.

- [ ] **Step 3: Implement tool gateway**

Create `src/review_agent/tool_gateway.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath, Path
import subprocess
from typing import Any

from review_agent.observations import ObservationStore


class ToolGatewayError(ValueError):
    pass


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    observation_ids: list[str]
    context_view: str
    truncated: bool = False


class ToolGateway:
    def __init__(
        self,
        repository_path: Path,
        base_revision: str,
        head_revision: str,
        observation_store: ObservationStore,
        max_context_chars: int = 4000,
        timeout_seconds: int = 10,
    ) -> None:
        self.repository_path = repository_path
        self.base_revision = base_revision
        self.head_revision = head_revision
        self.observation_store = observation_store
        self.max_context_chars = max_context_chars
        self.timeout_seconds = timeout_seconds

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        if tool_name == "read_range":
            return self._read_range(arguments)
        if tool_name == "compare_base_head":
            return self._compare_base_head(arguments)
        if tool_name == "search_code":
            return self._search_code(arguments)
        raise ToolGatewayError(f"unsupported tool: {tool_name}")

    def _read_range(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        path = _safe_repo_path(str(arguments["path"]))
        revision_label, revision = self._resolve_revision(str(arguments.get("revision", "head")))
        line_start = int(arguments["line_start"])
        line_end = int(arguments["line_end"])
        if line_start < 1 or line_end < line_start:
            raise ToolGatewayError("invalid line range")
        content = _git_show(self.repository_path, revision, path, self.timeout_seconds)
        lines = content.splitlines()
        selected = "\n".join(lines[line_start - 1 : line_end])
        if selected:
            selected += "\n"
        context_view, truncated = _context_view(selected, self.max_context_chars)
        observation = self.observation_store.record(
            source="git.read_range",
            revision=f"{revision_label}@{revision}",
            path=path,
            line_start=line_start,
            line_end=line_end,
            raw_content=selected,
            context_view=context_view,
        )
        return ToolExecutionResult("read_range", [observation.observation_id], context_view, truncated)

    def _compare_base_head(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        path = _safe_repo_path(str(arguments["path"]))
        raw = _run_git(
            self.repository_path,
            ["diff", "--unified=3", f"{self.base_revision}..{self.head_revision}", "--", path],
            self.timeout_seconds,
            allow_exit_codes={0},
        )
        context_view, truncated = _context_view(raw, self.max_context_chars)
        observation = self.observation_store.record(
            source="git.compare_base_head",
            revision=f"{self.base_revision}..{self.head_revision}",
            path=path,
            line_start=None,
            line_end=None,
            raw_content=raw,
            context_view=context_view,
        )
        return ToolExecutionResult("compare_base_head", [observation.observation_id], context_view, truncated)

    def _search_code(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        query = str(arguments["query"])
        if not query:
            raise ToolGatewayError("search query must not be empty")
        revision_label, revision = self._resolve_revision(str(arguments.get("revision", "head")))
        max_results = int(arguments.get("max_results", 20))
        if max_results < 1:
            raise ToolGatewayError("max_results must be positive")
        raw = _run_git(
            self.repository_path,
            ["grep", "-n", "--fixed-strings", query, revision, "--", "."],
            self.timeout_seconds,
            allow_exit_codes={0, 1},
        )
        lines = raw.splitlines()[:max_results]
        limited = "\n".join(lines)
        if limited:
            limited += "\n"
        context_view, truncated = _context_view(limited, self.max_context_chars)
        observation = self.observation_store.record(
            source="git.search_code",
            revision=f"{revision_label}@{revision}",
            path=None,
            line_start=None,
            line_end=None,
            raw_content=limited,
            context_view=context_view,
        )
        return ToolExecutionResult("search_code", [observation.observation_id], context_view, truncated)

    def _resolve_revision(self, revision_label: str) -> tuple[str, str]:
        if revision_label == "base":
            return "base", self.base_revision
        if revision_label == "head":
            return "head", self.head_revision
        raise ToolGatewayError(f"unauthorized revision: {revision_label}")


def _safe_repo_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or not normalized or pure.parts[0] in {".git", ".env"}:
        raise ToolGatewayError(f"unsafe repository path: {path}")
    return normalized


def _git_show(repo: Path, revision: str, path: str, timeout_seconds: int) -> str:
    return _run_git(repo, ["show", f"{revision}:{path}"], timeout_seconds, allow_exit_codes={0})


def _run_git(repo: Path, args: list[str], timeout_seconds: int, allow_exit_codes: set[int]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
    )
    if result.returncode not in allow_exit_codes:
        raise ToolGatewayError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def _context_view(content: str, max_chars: int) -> tuple[str, bool]:
    if len(content) <= max_chars:
        return content, False
    return content[:max_chars] + "\n[truncated]\n", True
```

- [ ] **Step 4: Run tool gateway tests and verify pass**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_tool_gateway.py -q -p no:cacheprovider
```

Expected: `5 passed`.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src/review_agent/tool_gateway.py tests/test_tool_gateway.py
git commit -m "feat: add read-only tool gateway"
```

---

## Task 3: CLI Observation Integration

**Files:**
- Modify: `src/review_agent/cli.py`
- Modify: `src/review_agent/reporting.py`
- Modify: `tests/test_cli_smoke.py`
- Modify: `tests/test_checkpoint_reporting.py`

- [ ] **Step 1: Write failing CLI and report tests**

Append to `tests/test_cli_smoke.py`:

```python
def test_cli_fake_reviewer_writes_observation_store_artifacts(git_repo: Path):
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
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    observation_records = [
        json.loads(line)
        for line in (run_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert observation_records
    assert observation_records[0]["source"] == "git.compare_base_head"
    assert observation_records[0]["path"] == "auth.py"
    assert (run_dir / observation_records[0]["raw_artifact_ref"]).exists()

    envelope = json.loads((run_dir / "reviewer_envelope.json").read_text(encoding="utf-8"))
    assert observation_records[0]["observation_id"] in envelope["messages"][0]["content"]
    assert "## Observations" in (run_dir / "report.md").read_text(encoding="utf-8")
```

Append to `tests/test_checkpoint_reporting.py`:

```python
def test_markdown_report_includes_observation_summaries():
    assessment = RiskAssessment(
        level=RiskLevel.LOW,
        dimensions={"impact": "local"},
        reasons=[],
        signal_refs=[],
        uncertainties=[],
        suggested_focus=[],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        risk_assessment=assessment,
        changed_files=["auth.py"],
        observation_summaries={"O-abc": "auth.py changed between base and head"},
    )

    assert "## Observations" in report
    assert "- O-abc: auth.py changed between base and head" in report
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_cli_smoke.py::test_cli_fake_reviewer_writes_observation_store_artifacts tests/test_checkpoint_reporting.py::test_markdown_report_includes_observation_summaries -q -p no:cacheprovider
```

Expected: fail because CLI does not create observation artifacts and report does not accept `observation_summaries`.

- [ ] **Step 3: Update report rendering**

In `src/review_agent/reporting.py`, add an optional `observation_summaries` parameter:

```python
def render_markdown_report(
    review_id: str,
    base_revision: str,
    head_revision: str,
    risk_assessment: RiskAssessment,
    changed_files: list[str],
    reviewer_result: ReviewerResult | None = None,
    observation_summaries: dict[str, str] | None = None,
) -> str:
```

Insert before the single reviewer section:

```python
            *_observation_section(observation_summaries or {}),
```

Add:

```python
def _observation_section(observation_summaries: dict[str, str]) -> list[str]:
    if not observation_summaries:
        return []
    items = "\n".join(
        f"- {observation_id}: {summary}"
        for observation_id, summary in observation_summaries.items()
    )
    return [
        "## Observations",
        "",
        items,
        "",
    ]
```

- [ ] **Step 4: Update CLI integration**

In `src/review_agent/cli.py`, import:

```python
from review_agent.observations import ObservationStore
from review_agent.tool_gateway import ToolGateway
```

After `store = CheckpointStore(repo, review_id)`, create:

```python
    observation_store = ObservationStore(store.run_dir)
```

Before `run_single_reviewer`, record changed-file diff observations:

```python
        gateway = ToolGateway(
            repository_path=repo,
            base_revision=args.base,
            head_revision=args.head,
            observation_store=observation_store,
        )
        for changed_file in change_summary.changed_files:
            gateway.execute("compare_base_head", {"path": changed_file})
```

Pass summaries into reviewer and report:

```python
            observations=observation_store.summaries_by_id(),
```

```python
        observation_summaries=observation_store.summaries_by_id(),
```

- [ ] **Step 5: Run focused tests and verify pass**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_cli_smoke.py::test_cli_fake_reviewer_writes_observation_store_artifacts tests/test_checkpoint_reporting.py::test_markdown_report_includes_observation_summaries -q -p no:cacheprovider
```

Expected: `2 passed`.

- [ ] **Step 6: Run CLI and reporting suites**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_cli_smoke.py tests/test_checkpoint_reporting.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 3**

Run:

```powershell
git add src/review_agent/cli.py src/review_agent/reporting.py tests/test_cli_smoke.py tests/test_checkpoint_reporting.py
git commit -m "feat: persist reviewer observations from tool gateway"
```

---

## Task 4: Final Verification

**Files:**
- Modify only files with failing implementation bugs.

- [ ] **Step 1: Run no-placeholder scan on the plan**

Run:

```powershell
$patterns = @('TB'+'D','TO'+'DO','PLACE'+'HOLDER','x'+'xx','implement '+'later','fill in '+'details') -join '|'
rg -n $patterns docs/superpowers/plans/2026-06-30-tool-gateway-observation-store.md
```

Expected: no matches.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 3: Run fake reviewer manual smoke**

Run:

```powershell
$ErrorActionPreference='Stop'
$sample = Join-Path 'C:\tmp' ('review-agent-tool-gateway-smoke-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $sample -Force | Out-Null
Push-Location $sample
try {
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
}
finally {
    Pop-Location
}
$env:PYTHONPATH=(Resolve-Path .\src).Path
$env:PYTHONDONTWRITEBYTECODE='1'
python -m review_agent review --repo $sample --base $base --head $head --intent 'Refactor admin check' --focus 'authorization regression' --reviewer-provider fake --non-interactive
```

Expected:

- Command prints `Review foundation completed: ...`.
- Run directory contains `observations.jsonl`.
- Run directory contains at least one `observations/O-*.txt` raw artifact.
- `reviewer_envelope.json` contains an observation ID.
- `report.md` contains `## Observations`.

- [ ] **Step 4: Commit the plan**

Run:

```powershell
git add docs/superpowers/plans/2026-06-30-tool-gateway-observation-store.md
git commit -m "docs: plan tool gateway observation store"
```

## Self-review checklist

- Spec coverage:
  - Observation Store records stable IDs, raw artifacts, and context views.
  - Tool Gateway performs read-only path/revision checks.
  - CLI path records observations before single reviewer invocation.
  - Report and envelope expose context summaries, not the full store.
- Type consistency:
  - `ObservationStore.summaries_by_id()` returns `dict[str, str]`.
  - `ToolGateway.execute()` returns `ToolExecutionResult`.
  - CLI passes summary maps to both reviewer and report.
- Scope check:
  - No AST, ripgrep/LSP, multi-agent loop, evidence reconciliation, completion checking, or eval harness is included.

## Execution handoff

This plan is intended for inline execution in this session. Subagent-driven execution remains acceptable if quota is available, but this slice is small enough to execute with direct TDD checkpoints.

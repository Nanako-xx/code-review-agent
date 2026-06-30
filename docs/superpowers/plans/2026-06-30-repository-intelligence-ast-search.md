# Repository Intelligence AST And Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first Repository Intelligence vertical slice: revision-bound Python AST symbols, changed-symbol detection, repository text references, LSP-unavailable fallback metadata, Tool Gateway navigation tools, and CLI/report context integration.

**Architecture:** A new `repository_intelligence.py` module reads Git snapshots, parses Python AST symbols, detects changed symbols between `base_revision` and `head_revision`, and performs revision-bound text reference search through Git. `ToolGateway` exposes `list_symbols`, `inspect_symbol`, and `find_references`, recording every result as an Observation. CLI records a repository-intelligence snapshot artifact and passes the resulting observation summary to the reviewer/report without passing the full repository.

**Tech Stack:** Python 3.11+, stdlib `ast`, `dataclasses`, `hashlib`, `json`, `subprocess`, `pathlib`, `pytest`, Git CLI. No new third-party dependencies.

---

## Scope

In scope:

- Python AST symbol index for committed Git revisions.
- Function, async function, class, and method symbols with path and line ranges.
- Per-symbol body hash and simple call-name extraction.
- Changed-symbol detection for changed Python files.
- Revision-bound text reference search using Git grep as the safe fallback backend.
- Explicit LSP status metadata: unavailable, fallback to Python AST plus revision-bound text search.
- Tool Gateway navigation tools:
  - `list_symbols`
  - `inspect_symbol`
  - `find_references`
- CLI artifact: `repository_intelligence.json`.
- Observation entry for the repository-intelligence snapshot.
- Report section: `## Repository Intelligence`.

Out of scope:

- A real LSP client process.
- TypeScript or multi-language semantic indexing.
- Full call graph or test mapping.
- Model-driven tool-call loop.
- Evidence reconciliation, completion checking, multi-agent orchestration, and eval harness.

## File Structure

- Create `src/review_agent/repository_intelligence.py`
  - `PythonSymbol`
  - `ChangedSymbol`
  - `TextSearchMatch`
  - `RepositoryIntelligenceSnapshot`
  - symbol parsing, changed-symbol detection, text search, snapshot summary, and artifact serialization
- Modify `src/review_agent/tool_gateway.py`
  - add `list_symbols`, `inspect_symbol`, and `find_references`
- Modify `src/review_agent/context.py`
  - expose the new repository navigation tools in the model envelope
- Modify `src/review_agent/cli.py`
  - build repository-intelligence snapshot
  - record snapshot observation
  - write `repository_intelligence.json`
  - pass report summary
- Modify `src/review_agent/reporting.py`
  - render repository intelligence summary
- Create `tests/test_repository_intelligence.py`
- Modify `tests/test_tool_gateway.py`
- Modify `tests/test_context.py`
- Modify `tests/test_cli_smoke.py`
- Modify `tests/test_checkpoint_reporting.py`

---

## Task 1: Python AST Symbol Index And Changed Symbols

**Files:**
- Create: `src/review_agent/repository_intelligence.py`
- Create: `tests/test_repository_intelligence.py`

- [ ] **Step 1: Write failing repository intelligence tests**

Create `tests/test_repository_intelligence.py`:

```python
from pathlib import Path

from conftest import run_git
from review_agent.repository_intelligence import (
    build_repository_intelligence,
    collect_python_symbols,
    repository_intelligence_to_dict,
    search_repository_text,
    summarize_repository_intelligence,
)


def test_collect_python_symbols_reads_committed_head(git_repo: Path):
    (git_repo / "auth.py").write_text(
        "import os\n\n"
        "class User:\n"
        "    def is_admin(self):\n"
        "        return check_role(self.role)\n\n"
        "def check_role(role):\n"
        "    return role == 'admin'\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth symbols")
    head = run_git(git_repo, "rev-parse", "HEAD")

    symbols = collect_python_symbols(git_repo, head, paths=["auth.py"])
    by_name = {symbol.qualified_name: symbol for symbol in symbols}

    assert by_name["User"].kind == "class"
    assert by_name["User.is_admin"].kind == "method"
    assert by_name["User.is_admin"].calls == ["check_role"]
    assert by_name["check_role"].line_start == 7
    assert by_name["check_role"].line_end == 8


def test_build_repository_intelligence_detects_changed_symbols(git_repo: Path):
    (git_repo / "auth.py").write_text(
        "def check_role(role):\n"
        "    return role == 'admin'\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "base auth")
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text(
        "def check_role(role):\n"
        "    return role in {'admin', 'owner'}\n\n"
        "def is_owner(role):\n"
        "    return role == 'owner'\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "head auth")
    head = run_git(git_repo, "rev-parse", "HEAD")

    snapshot = build_repository_intelligence(git_repo, base, head, changed_files=["auth.py"])

    changed = {(item.qualified_name, item.change_type) for item in snapshot.changed_symbols}
    assert ("check_role", "modified") in changed
    assert ("is_owner", "added") in changed
    assert snapshot.lsp_status == "unavailable"
    assert snapshot.fallback_strategy == "python_ast+git_grep"
    assert snapshot.text_search_backend == "git-grep"


def test_search_repository_text_is_revision_bound(git_repo: Path):
    (git_repo / "auth.py").write_text("def check_role(role):\n    return role == 'admin'\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth")
    head = run_git(git_repo, "rev-parse", "HEAD")

    matches = search_repository_text(git_repo, head, "check_role", max_results=5)

    assert matches[0].path == "auth.py"
    assert matches[0].line_number == 1
    assert "def check_role" in matches[0].line


def test_repository_intelligence_summary_and_dict(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change add")
    head = run_git(git_repo, "rev-parse", "HEAD")

    snapshot = build_repository_intelligence(git_repo, base, head, changed_files=["app.py"])
    payload = repository_intelligence_to_dict(snapshot)
    summary = summarize_repository_intelligence(snapshot)

    assert payload["revision"] == head
    assert payload["base_revision"] == base
    assert "LSP unavailable; using python_ast+git_grep" in summary
    assert "modified function add app.py:1-2" in summary
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_repository_intelligence.py -q -p no:cacheprovider
```

Expected: fail because `review_agent.repository_intelligence` does not exist.

- [ ] **Step 3: Implement repository intelligence module**

Create `src/review_agent/repository_intelligence.py` with:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import ast
import hashlib
import json
import subprocess
from typing import Iterable


@dataclass(frozen=True)
class PythonSymbol:
    path: str
    name: str
    qualified_name: str
    kind: str
    line_start: int
    line_end: int
    body_hash: str
    calls: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChangedSymbol:
    path: str
    qualified_name: str
    kind: str
    change_type: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class TextSearchMatch:
    path: str
    line_number: int
    line: str


@dataclass(frozen=True)
class RepositoryIntelligenceSnapshot:
    base_revision: str
    revision: str
    changed_symbols: list[ChangedSymbol]
    lsp_status: str = "unavailable"
    fallback_strategy: str = "python_ast+git_grep"
    text_search_backend: str = "git-grep"


def collect_python_symbols(repo: Path, revision: str, paths: list[str] | None = None) -> list[PythonSymbol]:
    candidate_paths = paths if paths is not None else _list_python_files(repo, revision)
    symbols: list[PythonSymbol] = []
    for path in candidate_paths:
        if not path.endswith(".py"):
            continue
        content = _git_show(repo, revision, path, allow_missing=True)
        if content is None:
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        symbols.extend(_symbols_from_tree(path, content, tree))
    return symbols


def build_repository_intelligence(
    repo: Path,
    base_revision: str,
    head_revision: str,
    changed_files: list[str],
) -> RepositoryIntelligenceSnapshot:
    changed_symbols = detect_changed_symbols(repo, base_revision, head_revision, changed_files)
    return RepositoryIntelligenceSnapshot(
        base_revision=base_revision,
        revision=head_revision,
        changed_symbols=changed_symbols,
    )


def detect_changed_symbols(
    repo: Path,
    base_revision: str,
    head_revision: str,
    changed_files: list[str],
) -> list[ChangedSymbol]:
    python_files = [path for path in changed_files if path.endswith(".py")]
    base_symbols = {
        (symbol.path, symbol.qualified_name): symbol
        for symbol in collect_python_symbols(repo, base_revision, python_files)
    }
    head_symbols = {
        (symbol.path, symbol.qualified_name): symbol
        for symbol in collect_python_symbols(repo, head_revision, python_files)
    }
    changes: list[ChangedSymbol] = []
    for key, symbol in head_symbols.items():
        previous = base_symbols.get(key)
        if previous is None:
            changes.append(_changed(symbol, "added"))
        elif previous.body_hash != symbol.body_hash:
            changes.append(_changed(symbol, "modified"))
    for key, symbol in base_symbols.items():
        if key not in head_symbols:
            changes.append(_changed(symbol, "deleted"))
    return sorted(changes, key=lambda item: (item.path, item.line_start, item.qualified_name, item.change_type))


def search_repository_text(repo: Path, revision: str, query: str, max_results: int = 20) -> list[TextSearchMatch]:
    if not query:
        return []
    raw = _run_git(repo, ["grep", "-n", "--fixed-strings", query, revision, "--", "."], allow_exit_codes={0, 1})
    matches: list[TextSearchMatch] = []
    for line in raw.splitlines()[:max_results]:
        parsed = _parse_git_grep_line(line)
        if parsed is not None:
            matches.append(parsed)
    return matches


def repository_intelligence_to_dict(snapshot: RepositoryIntelligenceSnapshot) -> dict[str, object]:
    return asdict(snapshot)


def repository_intelligence_raw_json(snapshot: RepositoryIntelligenceSnapshot) -> str:
    return json.dumps(repository_intelligence_to_dict(snapshot), ensure_ascii=False, indent=2)


def summarize_repository_intelligence(snapshot: RepositoryIntelligenceSnapshot) -> str:
    lines = [
        "Repository Intelligence",
        f"Revision: {snapshot.revision}",
        f"LSP unavailable; using {snapshot.fallback_strategy}",
        f"Text search backend: {snapshot.text_search_backend}",
        "Changed Symbols:",
    ]
    if not snapshot.changed_symbols:
        lines.append("- No changed Python symbols detected")
    for symbol in snapshot.changed_symbols:
        lines.append(
            f"- {symbol.change_type} {symbol.kind} {symbol.qualified_name} "
            f"{symbol.path}:{symbol.line_start}-{symbol.line_end}"
        )
    return "\n".join(lines)


def _symbols_from_tree(path: str, content: str, tree: ast.AST) -> list[PythonSymbol]:
    lines = content.splitlines()
    collector = _SymbolCollector(path, lines)
    collector.visit(tree)
    return collector.symbols


class _SymbolCollector(ast.NodeVisitor):
    def __init__(self, path: str, lines: list[str]) -> None:
        self.path = path
        self.lines = lines
        self.stack: list[tuple[str, str]] = []
        self.symbols: list[PythonSymbol] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add_symbol(node, "class")
        self.stack.append((node.name, "class"))
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_symbol(node, "method" if self._inside_class() else "function")
        self.stack.append((node.name, "function"))
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_symbol(node, "method" if self._inside_class() else "async_function")
        self.stack.append((node.name, "function"))
        self.generic_visit(node)
        self.stack.pop()

    def _add_symbol(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> None:
        line_start = int(node.lineno)
        line_end = int(getattr(node, "end_lineno", node.lineno))
        qualified_name = ".".join([name for name, _kind in self.stack] + [node.name])
        body = "\n".join(self.lines[line_start - 1 : line_end])
        self.symbols.append(
            PythonSymbol(
                path=self.path,
                name=node.name,
                qualified_name=qualified_name,
                kind=kind,
                line_start=line_start,
                line_end=line_end,
                body_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                calls=_call_names(node),
            )
        )

    def _inside_class(self) -> bool:
        return any(kind == "class" for _name, kind in self.stack)


def _call_names(node: ast.AST) -> list[str]:
    calls: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _call_name(child.func)
            if name:
                calls.add(name)
    return sorted(calls)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _changed(symbol: PythonSymbol, change_type: str) -> ChangedSymbol:
    return ChangedSymbol(
        path=symbol.path,
        qualified_name=symbol.qualified_name,
        kind=symbol.kind,
        change_type=change_type,
        line_start=symbol.line_start,
        line_end=symbol.line_end,
    )


def _list_python_files(repo: Path, revision: str) -> list[str]:
    raw = _run_git(repo, ["ls-tree", "-r", "--name-only", revision], allow_exit_codes={0})
    return [line for line in raw.splitlines() if line.endswith(".py")]


def _git_show(repo: Path, revision: str, path: str, allow_missing: bool) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        return result.stdout
    if allow_missing:
        return None
    raise RuntimeError(result.stderr.strip() or f"git show failed for {path}")


def _run_git(repo: Path, args: list[str], allow_exit_codes: set[int]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode not in allow_exit_codes:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def _parse_git_grep_line(line: str) -> TextSearchMatch | None:
    parts = line.split(":", 2)
    if len(parts) != 3:
        return None
    path, line_number, content = parts
    if line_number.isdigit():
        return TextSearchMatch(path=path, line_number=int(line_number), line=content)
    parts = content.split(":", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return TextSearchMatch(path=line_number, line_number=int(parts[0]), line=parts[1])
    return None
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_repository_intelligence.py -q -p no:cacheprovider
```

Expected: `4 passed`.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src/review_agent/repository_intelligence.py tests/test_repository_intelligence.py
git commit -m "feat: add repository intelligence snapshot"
```

---

## Task 2: Repository Intelligence Tool Gateway Tools

**Files:**
- Modify: `src/review_agent/tool_gateway.py`
- Modify: `src/review_agent/context.py`
- Modify: `tests/test_tool_gateway.py`
- Modify: `tests/test_context.py`

- [ ] **Step 1: Add failing Tool Gateway tests**

Append to `tests/test_tool_gateway.py`:

```python
def test_tool_gateway_list_symbols_records_ast_observation(git_repo: Path, tmp_path: Path):
    (git_repo / "auth.py").write_text("def check_role(role):\n    return role == 'admin'\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add symbol")
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path)
    gateway = ToolGateway(git_repo, base_revision=head, head_revision=head, observation_store=store)

    result = gateway.execute("list_symbols", {"path": "auth.py", "revision": "head"})

    assert len(result.observation_ids) == 1
    assert "function check_role auth.py:1-2" in result.context_view
    assert store.list_observations()[0].source == "repo_intelligence.list_symbols"


def test_tool_gateway_inspect_symbol_records_calls(git_repo: Path, tmp_path: Path):
    (git_repo / "auth.py").write_text(
        "def is_admin(user):\n"
        "    return check_role(user.role)\n\n"
        "def check_role(role):\n"
        "    return role == 'admin'\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add symbol calls")
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path)
    gateway = ToolGateway(git_repo, base_revision=head, head_revision=head, observation_store=store)

    result = gateway.execute("inspect_symbol", {"name": "is_admin", "revision": "head"})

    assert "is_admin" in result.context_view
    assert "calls: check_role" in result.context_view
    assert store.list_observations()[0].source == "repo_intelligence.inspect_symbol"


def test_tool_gateway_find_references_uses_revision_bound_text_search(git_repo: Path, tmp_path: Path):
    (git_repo / "auth.py").write_text("def check_role(role):\n    return role == 'admin'\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add references")
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path)
    gateway = ToolGateway(git_repo, base_revision=head, head_revision=head, observation_store=store)

    result = gateway.execute("find_references", {"name": "check_role", "revision": "head", "max_results": 5})

    assert "auth.py:1:def check_role" in result.context_view
    assert store.list_observations()[0].source == "repo_intelligence.find_references"
```

Append to `tests/test_context.py`:

```python
def test_reviewer_envelope_includes_repository_intelligence_tools():
    assignment = Assignment(
        role="Core Reviewer",
        mission="Check intent alignment",
        assignment_reason=[],
        assigned_contract=[],
        required_checks=[],
        initial_context=InitialContext(),
        max_turns=6,
        max_tool_calls=12,
    )
    intent = IntentPacket(goal="Review changes", status=IntentStatus.PARTIAL)

    envelope = build_reviewer_envelope(
        assignment=assignment,
        intent=intent,
        code_snippets={},
        observations={},
        trace_id="trace-ri-tools",
    )

    tool_names = {tool["name"] for tool in envelope.tools}
    assert {"list_symbols", "inspect_symbol", "find_references"}.issubset(tool_names)
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_tool_gateway.py::test_tool_gateway_list_symbols_records_ast_observation tests/test_tool_gateway.py::test_tool_gateway_inspect_symbol_records_calls tests/test_tool_gateway.py::test_tool_gateway_find_references_uses_revision_bound_text_search tests/test_context.py::test_reviewer_envelope_includes_repository_intelligence_tools -q -p no:cacheprovider
```

Expected: fail because the new tools are not supported or exposed.

- [ ] **Step 3: Implement Tool Gateway tools**

Modify `src/review_agent/tool_gateway.py`:

- Import:

```python
from dataclasses import asdict
import json
from review_agent.repository_intelligence import collect_python_symbols, search_repository_text
```

- Add execute cases for `list_symbols`, `inspect_symbol`, and `find_references`.
- Add helper methods that:
  - resolve only `base` or `head`
  - use `_safe_repo_path` for path-scoped calls
  - record raw JSON in Observation Store
  - return short context views

- [ ] **Step 4: Update context tools**

Modify `src/review_agent/context.py` tool list to include:

```python
{
    "name": "list_symbols",
    "description": "List Python AST symbols for a repository file at an authorized revision.",
},
{
    "name": "inspect_symbol",
    "description": "Inspect a Python AST symbol, including path, line range, and simple call names.",
},
{
    "name": "find_references",
    "description": "Find textual references to a symbol name within the authorized repository revision.",
},
```

- [ ] **Step 5: Run focused tests and verify pass**

Run the same focused command from Step 2.

Expected: `4 passed`.

- [ ] **Step 6: Commit Task 2**

Run:

```powershell
git add src/review_agent/tool_gateway.py src/review_agent/context.py tests/test_tool_gateway.py tests/test_context.py
git commit -m "feat: expose repository intelligence tools"
```

---

## Task 3: CLI Snapshot, Observation, And Report Integration

**Files:**
- Modify: `src/review_agent/cli.py`
- Modify: `src/review_agent/reporting.py`
- Modify: `tests/test_cli_smoke.py`
- Modify: `tests/test_checkpoint_reporting.py`

- [ ] **Step 1: Add failing CLI/report tests**

Append to `tests/test_cli_smoke.py`:

```python
def test_cli_writes_repository_intelligence_artifacts(git_repo: Path):
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
            "fake",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    payload = json.loads((run_dir / "repository_intelligence.json").read_text(encoding="utf-8"))
    observation_records = [
        json.loads(line) for line in (run_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    envelope = json.loads((run_dir / "reviewer_envelope.json").read_text(encoding="utf-8"))

    assert payload["changed_symbols"][0]["qualified_name"] == "add"
    assert any(record["source"] == "repo_intelligence.snapshot" for record in observation_records)
    assert "## Repository Intelligence" in report
    assert "modified function add app.py:1-2" in report
    assert "Repository Intelligence" in envelope["messages"][0]["content"]
```

Append to `tests/test_checkpoint_reporting.py`:

```python
def test_markdown_report_includes_repository_intelligence_summary():
    assessment = RiskAssessment(
        level=RiskLevel.LOW,
        dimensions={},
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
        changed_files=["app.py"],
        repository_intelligence_summary="Repository Intelligence\n- modified function add app.py:1-2",
    )

    assert "## Repository Intelligence" in report
    assert "modified function add app.py:1-2" in report
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_cli_smoke.py::test_cli_writes_repository_intelligence_artifacts tests/test_checkpoint_reporting.py::test_markdown_report_includes_repository_intelligence_summary -q -p no:cacheprovider
```

Expected: fail because CLI/report do not yet write repository intelligence.

- [ ] **Step 3: Update report rendering**

Modify `render_markdown_report` to accept `repository_intelligence_summary: str | None = None` and render:

```markdown
## Repository Intelligence

<summary>
```

- [ ] **Step 4: Update CLI integration**

In `src/review_agent/cli.py`:

- Import repository intelligence helpers.
- Build snapshot after `ObservationStore`.
- Write `repository_intelligence.json`.
- Record source `repo_intelligence.snapshot` into `ObservationStore`.
- Pass the summary to `render_markdown_report`.

- [ ] **Step 5: Run focused tests and verify pass**

Run the focused command from Step 2.

Expected: `2 passed`.

- [ ] **Step 6: Run selected suites**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_repository_intelligence.py tests/test_tool_gateway.py tests/test_context.py tests/test_cli_smoke.py tests/test_checkpoint_reporting.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 3**

Run:

```powershell
git add src/review_agent/cli.py src/review_agent/reporting.py tests/test_cli_smoke.py tests/test_checkpoint_reporting.py
git commit -m "feat: include repository intelligence in review context"
```

---

## Task 4: Final Verification

**Files:**
- Modify only files with failing implementation bugs.

- [ ] **Step 1: Scan plan for unfinished markers**

Run:

```powershell
$patterns = @('TB'+'D','TO'+'DO','PLACE'+'HOLDER','x'+'xx','implement '+'later','fill in '+'details') -join '|'
rg -n $patterns docs/superpowers/plans/2026-06-30-repository-intelligence-ast-search.md
```

Expected: no matches.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 3: Run fake reviewer smoke**

Run a local fake-reviewer smoke with a changed Python function and verify:

- `repository_intelligence.json` exists.
- `observations.jsonl` contains `repo_intelligence.snapshot`.
- `report.md` contains `## Repository Intelligence`.
- `reviewer_envelope.json` contains the repository intelligence observation summary.

- [ ] **Step 4: Commit the plan**

Run:

```powershell
git add docs/superpowers/plans/2026-06-30-repository-intelligence-ast-search.md
git commit -m "docs: plan repository intelligence ast search"
```

## Self-review checklist

- Spec coverage:
  - Repository Intelligence facts are revision-bound.
  - Python AST symbol index and changed symbols are implemented.
  - Text references are revision-bound through Git grep, with LSP marked unavailable.
  - Context receives summaries/observation refs, not the full repository.
  - Tool Gateway exposes navigation tools and records Observations.
- Type consistency:
  - `RepositoryIntelligenceSnapshot.revision` is the head revision.
  - Tool Gateway records `repo_intelligence.*` observations.
  - Report accepts `repository_intelligence_summary`.
- Scope check:
  - No real LSP process, TypeScript, multi-agent loop, reconciler, completion checker, or eval harness is included.

## Execution handoff

This plan is intended for inline execution in this session. Subagent-driven execution is acceptable if quota is available, but direct TDD checkpoints are sufficient for this vertical slice.

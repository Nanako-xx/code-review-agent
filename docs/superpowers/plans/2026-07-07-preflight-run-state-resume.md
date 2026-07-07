# Preflight + Run State + Resume Implementation Plan

> Scope: local-only review execution observability and recovery foundation. This does not add evals or GitHub/PR integration.

## Goal

Make a local review run understandable and recoverable:

- Before review work starts, show a concrete preflight summary: review id, repo, base/head, changed files, intent status, quality gate summary, risk level, and run directory.
- Persist run state to `.review-agent/runs/<review_id>/state.json` as the review moves through phases.
- Add a `resume` command that reads an existing local run checkpoint and reports where the run is, what artifacts exist, and whether it completed or failed.
- Preserve the final architecture direction: this is not a throwaway CLI shortcut; it is the local state layer the later runtime/orchestrator can keep using.

## Current code touchpoints

- `src/review_agent/cli.py`
  - Owns the `review` command and writes artifacts.
  - Currently has no `resume` command and no state lifecycle.
- `src/review_agent/checkpoint.py`
  - Owns run directory writes.
  - Currently writes JSON/JSONL only; no read helpers and no state-specific API.
- `src/review_agent/reporting.py`
  - Generates the markdown report. This plan does not require report format changes.
- Existing tests around CLI/checkpoint/reporting should be extended, not replaced.

## Phase model

Use explicit phase/status names so future runtime, orchestrator, and UI can depend on them.

```python
class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunPhase(str, Enum):
    CREATED = "created"
    PREFLIGHT = "preflight"
    QUALITY_GATES = "quality_gates"
    REPOSITORY_INTELLIGENCE = "repository_intelligence"
    REVIEWERS = "reviewers"
    RECONCILIATION = "reconciliation"
    COMPLETION = "completion"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"
```

The persisted state should be stable JSON:

```json
{
  "review_id": "review-20260707-120000",
  "status": "completed",
  "phase": "completed",
  "repository_path": "D:/Agent/code review agent",
  "base_revision": "master",
  "head_revision": "HEAD",
  "message": "Review completed",
  "artifacts": {
    "request": "request.json",
    "intent": "intent.json",
    "quality_gates": "quality_gates.json",
    "repository_intelligence": "repository_intelligence.json",
    "report": "report.md"
  },
  "errors": []
}
```

Artifact paths are relative to the run directory so the state remains portable if the repository moves.

## Task 1: Add run state model and checkpoint read/write support

Files:

- Create `src/review_agent/run_state.py`
- Update `src/review_agent/checkpoint.py`
- Add `tests/test_run_state.py`
- Extend `tests/test_checkpoint_reporting.py`

Implementation details:

Create `src/review_agent/run_state.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunPhase(str, Enum):
    CREATED = "created"
    PREFLIGHT = "preflight"
    QUALITY_GATES = "quality_gates"
    REPOSITORY_INTELLIGENCE = "repository_intelligence"
    REVIEWERS = "reviewers"
    RECONCILIATION = "reconciliation"
    COMPLETION = "completion"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RunState:
    review_id: str
    status: RunStatus
    phase: RunPhase
    repository_path: str
    base_revision: str
    head_revision: str
    message: str
    artifacts: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def initial_run_state(
    *,
    review_id: str,
    repository_path: str,
    base_revision: str,
    head_revision: str,
) -> RunState:
    return RunState(
        review_id=review_id,
        status=RunStatus.CREATED,
        phase=RunPhase.CREATED,
        repository_path=repository_path,
        base_revision=base_revision,
        head_revision=head_revision,
        message="Run created",
    )


def advance_run_state(
    state: RunState,
    *,
    phase: RunPhase,
    message: str,
    artifacts: dict[str, str] | None = None,
) -> RunState:
    next_artifacts = dict(state.artifacts)
    if artifacts:
        next_artifacts.update(artifacts)
    status = RunStatus.COMPLETED if phase is RunPhase.COMPLETED else RunStatus.RUNNING
    return RunState(
        review_id=state.review_id,
        status=status,
        phase=phase,
        repository_path=state.repository_path,
        base_revision=state.base_revision,
        head_revision=state.head_revision,
        message=message,
        artifacts=next_artifacts,
        errors=list(state.errors),
    )


def fail_run_state(state: RunState, *, message: str, error: str) -> RunState:
    return RunState(
        review_id=state.review_id,
        status=RunStatus.FAILED,
        phase=RunPhase.FAILED,
        repository_path=state.repository_path,
        base_revision=state.base_revision,
        head_revision=state.head_revision,
        message=message,
        artifacts=dict(state.artifacts),
        errors=[*state.errors, error],
    )


def run_state_to_dict(state: RunState) -> dict[str, Any]:
    return {
        "review_id": state.review_id,
        "status": state.status.value,
        "phase": state.phase.value,
        "repository_path": state.repository_path,
        "base_revision": state.base_revision,
        "head_revision": state.head_revision,
        "message": state.message,
        "artifacts": dict(state.artifacts),
        "errors": list(state.errors),
    }


def run_state_from_dict(payload: dict[str, Any]) -> RunState:
    return RunState(
        review_id=str(payload["review_id"]),
        status=RunStatus(str(payload["status"])),
        phase=RunPhase(str(payload["phase"])),
        repository_path=str(payload["repository_path"]),
        base_revision=str(payload["base_revision"]),
        head_revision=str(payload["head_revision"]),
        message=str(payload["message"]),
        artifacts={str(key): str(value) for key, value in dict(payload.get("artifacts", {})).items()},
        errors=[str(item) for item in list(payload.get("errors", []))],
    )
```

Update `CheckpointStore`:

```python
from review_agent.run_state import RunState, run_state_from_dict, run_state_to_dict


class CheckpointStore:
    def __init__(self, repository_path: Path, review_id: str, *, create: bool = True) -> None:
        self.repository_path = Path(repository_path)
        self.review_id = review_id
        self.run_dir = self.repository_path / ".review-agent" / "runs" / review_id
        if create:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def read_json(self, filename: str) -> dict[str, object]:
        return json.loads((self.run_dir / filename).read_text(encoding="utf-8"))

    def write_state(self, state: RunState) -> Path:
        return self.write_json("state.json", run_state_to_dict(state))

    def read_state(self) -> RunState:
        return run_state_from_dict(self.read_json("state.json"))
```

Tests:

```python
def test_run_state_advances_and_serializes_artifacts() -> None:
    state = initial_run_state(
        review_id="review-1",
        repository_path="repo",
        base_revision="main",
        head_revision="HEAD",
    )

    state = advance_run_state(
        state,
        phase=RunPhase.PREFLIGHT,
        message="Preflight completed",
        artifacts={"request": "request.json"},
    )

    payload = run_state_to_dict(state)
    loaded = run_state_from_dict(payload)

    assert loaded.status is RunStatus.RUNNING
    assert loaded.phase is RunPhase.PREFLIGHT
    assert loaded.artifacts == {"request": "request.json"}
```

```python
def test_checkpoint_store_writes_and_reads_run_state(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "review-1")
    state = initial_run_state(
        review_id="review-1",
        repository_path=str(tmp_path),
        base_revision="main",
        head_revision="HEAD",
    )

    store.write_state(state)

    assert store.read_state() == state
```

Verification:

```powershell
python -m pytest tests/test_run_state.py tests/test_checkpoint_reporting.py -q -p no:cacheprovider
```

## Task 2: Add review preflight output and state lifecycle writes

Files:

- Update `src/review_agent/cli.py`
- Extend `tests/test_cli_smoke.py`

Implementation details:

In `_run_review`, create the checkpoint store immediately after resolving repo/base/head/review id. Write initial state before expensive work:

```python
store = CheckpointStore(repo, review_id)
state = initial_run_state(
    review_id=review_id,
    repository_path=str(repo),
    base_revision=args.base,
    head_revision=args.head,
)
store.write_state(state)
```

Add a small helper in `cli.py`:

```python
def _print_preflight_summary(
    *,
    review_id: str,
    repo: Path,
    base_revision: str,
    head_revision: str,
    change_summary: ChangeSummary,
    intent_packet: IntentPacket,
    quality_gates: QualityGateResult,
    risk_assessment: RiskAssessment,
    run_dir: Path,
) -> None:
    print("Preflight")
    print(f"  Review ID: {review_id}")
    print(f"  Repository: {repo}")
    print(f"  Base: {base_revision}")
    print(f"  Head: {head_revision}")
    print(f"  Changed files: {len(change_summary.changed_files)}")
    print(f"  Intent status: {intent_packet.status.value}")
    print(f"  Risk level: {risk_assessment.level.value}")
    print(f"  Quality gates: {quality_gates.overall.value}")
    print(f"  Run directory: {run_dir}")
```

After request, intent, quality gates, risk assessment, and assignments are written, print the preflight summary and advance state:

```python
_print_preflight_summary(
    review_id=review_id,
    repo=repo,
    base_revision=args.base,
    head_revision=args.head,
    change_summary=change_summary,
    intent_packet=intent_packet,
    quality_gates=quality_gates,
    risk_assessment=risk_assessment,
    run_dir=store.run_dir,
)

state = advance_run_state(
    state,
    phase=RunPhase.PREFLIGHT,
    message="Preflight completed",
    artifacts={
        "request": "request.json",
        "intent": "intent.json",
        "quality_gates": "quality_gates.json",
        "risk": "risk.json",
        "assignments": "assignments.json",
    },
)
store.write_state(state)
```

Advance state at important existing artifact boundaries:

```python
state = advance_run_state(
    state,
    phase=RunPhase.REPOSITORY_INTELLIGENCE,
    message="Repository intelligence collected",
    artifacts={"repository_intelligence": "repository_intelligence.json"},
)
store.write_state(state)
```

```python
state = advance_run_state(
    state,
    phase=RunPhase.REVIEWERS,
    message="Reviewer execution completed",
    artifacts={"reviewer": "reviewer.json"},
)
store.write_state(state)
```

Use `"multi_reviewer": "multi_reviewer.json"` instead when the multi-agent path runs.

At report write:

```python
state = advance_run_state(
    state,
    phase=RunPhase.COMPLETED,
    message="Review completed",
    artifacts={"report": "report.md"},
)
store.write_state(state)
```

Test expectations:

```python
def test_cli_review_writes_state_and_preflight_summary(git_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([
        "review",
        "--repo",
        str(git_repo),
        "--base",
        "HEAD~1",
        "--head",
        "HEAD",
        "--non-interactive",
    ])

    output = capsys.readouterr().out
    run_dirs = sorted((git_repo / ".review-agent" / "runs").iterdir())
    state = json.loads((run_dirs[-1] / "state.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "Preflight" in output
    assert "Changed files:" in output
    assert state["status"] == "completed"
    assert state["phase"] == "completed"
    assert state["artifacts"]["report"] == "report.md"
```

Verification:

```powershell
python -m pytest tests/test_cli_smoke.py tests/test_run_state.py tests/test_checkpoint_reporting.py -q -p no:cacheprovider
```

## Task 3: Add local `resume` command

Files:

- Update `src/review_agent/cli.py`
- Add `tests/test_cli_resume.py`

CLI shape:

```powershell
review-agent resume <review_id> --repo .
```

Parser changes:

```python
resume_parser = subparsers.add_parser("resume", help="Inspect and resume from a local review checkpoint")
resume_parser.add_argument("review_id", help="Review id under .review-agent/runs")
resume_parser.add_argument("--repo", default=".", help="Repository path")
```

Dispatch:

```python
if args.command == "resume":
    return _run_resume(args)
```

Implementation:

```python
def _run_resume(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    store = CheckpointStore(repo, args.review_id, create=False)

    if not store.run_dir.exists():
        print(f"Review run not found: {store.run_dir}", file=sys.stderr)
        return 2

    state_path = store.run_dir / "state.json"
    request_path = store.run_dir / "request.json"
    if not state_path.exists():
        print(f"Review run has no state.json: {store.run_dir}", file=sys.stderr)
        return 2
    if not request_path.exists():
        print(f"Review run has no request.json: {store.run_dir}", file=sys.stderr)
        return 2

    state = store.read_state()
    request = store.read_json("request.json")

    print("Resume")
    print(f"  Review ID: {state.review_id}")
    print(f"  Status: {state.status.value}")
    print(f"  Phase: {state.phase.value}")
    print(f"  Repository: {state.repository_path}")
    print(f"  Base: {state.base_revision}")
    print(f"  Head: {state.head_revision}")
    print(f"  Message: {state.message}")
    print(f"  Run directory: {store.run_dir}")
    print("  Artifacts:")
    for name, relative_path in sorted(state.artifacts.items()):
        marker = "present" if (store.run_dir / relative_path).exists() else "missing"
        print(f"    - {name}: {relative_path} ({marker})")

    if state.errors:
        print("  Errors:")
        for error in state.errors:
            print(f"    - {error}")

    return 0
```

The `request` read is intentional even if only state fields are printed: it validates that resume has enough checkpoint data to reproduce or continue later.

Tests:

```python
def test_cli_resume_prints_completed_run_summary(git_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([
        "review",
        "--repo",
        str(git_repo),
        "--base",
        "HEAD~1",
        "--head",
        "HEAD",
        "--non-interactive",
    ]) == 0
    run_id = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1].name

    assert main(["resume", run_id, "--repo", str(git_repo)]) == 0

    output = capsys.readouterr().out
    assert "Resume" in output
    assert f"Review ID: {run_id}" in output
    assert "Status: completed" in output
    assert "report.md (present)" in output
```

```python
def test_cli_resume_missing_run_returns_usage_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["resume", "missing-review", "--repo", str(tmp_path)])

    assert exit_code == 2
    assert "Review run not found" in capsys.readouterr().err
```

Verification:

```powershell
python -m pytest tests/test_cli_resume.py tests/test_cli_smoke.py -q -p no:cacheprovider
```

## Task 4: Persist failed run state for review errors

Files:

- Update `src/review_agent/cli.py`
- Add focused failure test in `tests/test_cli_smoke.py` or `tests/test_cli_resume.py`

Implementation details:

Keep `_run_review` returning numeric exit codes. Wrap the review body after initial state creation:

```python
try:
    ...
except Exception as error:
    failed_state = fail_run_state(
        state,
        message="Review failed",
        error=f"{type(error).__name__}: {error}",
    )
    store.write_state(failed_state)
    print(f"Review failed: {error}", file=sys.stderr)
    return 1
```

Do not catch `BaseException`, so user interrupts still behave normally.

If the implementation has early validation returns after state creation, write failed state before returning:

```python
state = fail_run_state(
    state,
    message="Review failed before reviewer execution",
    error=error_message,
)
store.write_state(state)
return 2
```

Test with monkeypatch:

```python
def test_cli_review_records_failed_state_when_collection_fails(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def raise_error(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr("review_agent.cli.collect_change_summary", raise_error)

    exit_code = main([
        "review",
        "--repo",
        str(git_repo),
        "--base",
        "HEAD~1",
        "--head",
        "HEAD",
        "--non-interactive",
    ])

    run_dirs = sorted((git_repo / ".review-agent" / "runs").iterdir())
    state = json.loads((run_dirs[-1] / "state.json").read_text(encoding="utf-8"))

    assert exit_code == 1
    assert state["status"] == "failed"
    assert state["phase"] == "failed"
    assert "RuntimeError: boom" in state["errors"]
    assert "Review failed" in capsys.readouterr().err
```

Verification:

```powershell
python -m pytest tests/test_cli_smoke.py tests/test_cli_resume.py tests/test_run_state.py tests/test_checkpoint_reporting.py -q -p no:cacheprovider
```

## Final verification

Run the full test suite:

```powershell
python -m pytest -q -p no:cacheprovider
```

Run one manual local smoke command in the repository:

```powershell
python -m review_agent.cli review --repo . --base HEAD~1 --head HEAD --non-interactive
```

Then inspect the latest run:

```powershell
python -m review_agent.cli resume <latest-review-id> --repo .
```

Expected outcomes:

- `state.json` exists in the run directory.
- Review output includes `Preflight`.
- Resume output includes status, phase, base/head, run directory, and artifact presence.
- Full tests pass.


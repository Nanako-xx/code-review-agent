# Multi-Agent Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first multi-reviewer orchestration slice that can run every Runtime assignment with isolated reviewer context, persist per-reviewer artifacts, and summarize aggregate reviewer status.

**Architecture:** Keep this version as deterministic orchestration, not a full ReAct runtime. `runtime.py` already expands risk into multiple `Assignment` objects; a new `orchestrator.py` loops through those assignments, calls the existing single-reviewer runner once per assignment, assigns isolated trace IDs, and returns a structured aggregate result. CLI adds `--reviewer-mode single|multi`; `single` preserves current behavior, while `multi` runs all assignments and writes both per-reviewer and aggregate artifacts.

**Tech Stack:** Python 3.11+, stdlib `dataclasses`, `json`, `argparse`, `pathlib`, `pytest`, Git CLI. No new dependencies.

---

## Scope

In scope:

- Multi-reviewer orchestration over existing `Assignment` objects.
- Isolated trace IDs: `<review_id>-reviewer-<index>`.
- Per-reviewer envelope/raw/result artifacts.
- Aggregate artifact: `multi_reviewer_result.json`.
- Report section: `## Multi-Reviewer Summary`.
- CLI flag: `--reviewer-mode single|multi`, default `single`.
- Fake-provider smoke path for multi mode.

Out of scope:

- True parallel execution.
- Model-driven multi-turn tool loop.
- Reconciler, conflict resolution, and completion checking.
- Dynamic follow-up assignment dispatch.
- Cross-reviewer memory sharing.
- Remote PR comments or GitHub integration.

## Task 1: Orchestrator Core

**Files:**
- Create: `src/review_agent/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing orchestrator tests**

Create `tests/test_orchestrator.py`:

```python
import json

from review_agent.models import Assignment, InitialContext, IntentPacket, IntentSource, IntentStatus
from review_agent.orchestrator import multi_reviewer_run_to_dict, run_multi_reviewer
from review_agent.provider import ModelResponse


class RecordingProvider:
    def __init__(self):
        self.trace_ids = []
        self.roles = []

    def complete(self, envelope):
        self.trace_ids.append(envelope.parameters["trace_id"])
        content = envelope.messages[0]["content"]
        role_line = next(line for line in content.splitlines() if line.startswith("Role: "))
        role = role_line.removeprefix("Role: ")
        self.roles.append(role)
        return ModelResponse(
            content=json.dumps(
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


def make_assignment(role: str) -> Assignment:
    return Assignment(
        role=role,
        mission=f"{role} mission",
        assignment_reason=[f"{role} reason"],
        assigned_contract=["regression_safety"],
        required_checks=["inspect direct observations"],
        initial_context=InitialContext(observation_refs=["O-shared"]),
        max_turns=6,
        max_tool_calls=12,
    )


def make_intent() -> IntentPacket:
    return IntentPacket(
        goal="Review risky change",
        sources={"goal": IntentSource.EXPLICIT},
        status=IntentStatus.PARTIAL,
    )


def test_run_multi_reviewer_runs_every_assignment_with_isolated_traces():
    provider = RecordingProvider()
    assignments = [make_assignment("Core Reviewer"), make_assignment("Adversarial Reviewer")]

    run = run_multi_reviewer(
        provider=provider,
        assignments=assignments,
        intent=make_intent(),
        diff_excerpt=["+changed"],
        observations={"O-shared": "shared observation"},
        trace_id_prefix="review-123",
    )

    assert provider.trace_ids == ["review-123-reviewer-0", "review-123-reviewer-1"]
    assert provider.roles == ["Core Reviewer", "Adversarial Reviewer"]
    assert [item.assignment.role for item in run.executions] == ["Core Reviewer", "Adversarial Reviewer"]
    assert [item.result.status.value for item in run.executions] == ["partial", "partial"]
    assert run.status_counts == {"partial": 2}


def test_multi_reviewer_run_to_dict_contains_artifact_summary():
    run = run_multi_reviewer(
        provider=RecordingProvider(),
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

- [ ] **Step 2: Run orchestrator tests and verify failure**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_orchestrator.py -q -p no:cacheprovider
```

Expected: fail because `review_agent.orchestrator` does not exist.

- [ ] **Step 3: Implement orchestrator core**

Create `src/review_agent/orchestrator.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from review_agent.models import Assignment, IntentPacket, ReviewerResult
from review_agent.provider import ModelProvider, ModelResponse
from review_agent.reviewer import ReviewerRun, reviewer_result_to_dict, run_single_reviewer


@dataclass(frozen=True)
class ReviewerExecution:
    reviewer_index: int
    trace_id: str
    assignment: Assignment
    envelope: object
    response: ModelResponse
    result: ReviewerResult


@dataclass(frozen=True)
class MultiReviewerRun:
    executions: list[ReviewerExecution]

    @property
    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for execution in self.executions:
            status = execution.result.status.value
            counts[status] = counts.get(status, 0) + 1
        return counts


def run_multi_reviewer(
    provider: ModelProvider,
    assignments: list[Assignment],
    intent: IntentPacket,
    diff_excerpt: list[str],
    observations: dict[str, str],
    trace_id_prefix: str,
) -> MultiReviewerRun:
    executions: list[ReviewerExecution] = []
    for index, assignment in enumerate(assignments):
        trace_id = f"{trace_id_prefix}-reviewer-{index}"
        run = run_single_reviewer(
            provider=provider,
            assignment=assignment,
            intent=intent,
            diff_excerpt=diff_excerpt,
            observations=observations,
            trace_id=trace_id,
        )
        executions.append(_execution_from_run(index, trace_id, assignment, run))
    return MultiReviewerRun(executions=executions)


def multi_reviewer_run_to_dict(run: MultiReviewerRun) -> dict[str, Any]:
    return {
        "reviewer_count": len(run.executions),
        "status_counts": run.status_counts,
        "executions": [
            {
                "reviewer_index": execution.reviewer_index,
                "trace_id": execution.trace_id,
                "role": execution.assignment.role,
                "result": reviewer_result_to_dict(execution.result),
                "provider_name": execution.response.provider_name,
                "model": execution.response.model,
            }
            for execution in run.executions
        ],
    }


def _execution_from_run(
    index: int,
    trace_id: str,
    assignment: Assignment,
    run: ReviewerRun,
) -> ReviewerExecution:
    return ReviewerExecution(
        reviewer_index=index,
        trace_id=trace_id,
        assignment=assignment,
        envelope=run.envelope,
        response=run.response,
        result=run.result,
    )
```

- [ ] **Step 4: Run orchestrator tests and verify pass**

Run the Task 1 test command again.

Expected: `2 passed`.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src/review_agent/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add multi reviewer orchestrator"
```

---

## Task 2: CLI Multi-Reviewer Mode And Artifacts

**Files:**
- Modify: `src/review_agent/cli.py`
- Modify: `src/review_agent/reporting.py`
- Modify: `tests/test_cli_smoke.py`
- Modify: `tests/test_checkpoint_reporting.py`

- [ ] **Step 1: Write failing CLI/report tests**

Append to `tests/test_cli_smoke.py`:

```python
def test_cli_multi_reviewer_mode_writes_per_reviewer_artifacts(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text(
        "def is_admin(user):\n"
        "    return True\n",
        encoding="utf-8",
    )
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
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    multi = json.loads((run_dir / "multi_reviewer_result.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")

    assert multi["reviewer_count"] >= 2
    assert {item["role"] for item in multi["executions"]} >= {"Core Reviewer", "Adversarial Reviewer"}
    assert (run_dir / "reviewer_0_envelope.json").exists()
    assert (run_dir / "reviewer_1_envelope.json").exists()
    assert (run_dir / "reviewer_0_result.json").exists()
    assert (run_dir / "reviewer_1_result.json").exists()
    assert "## Multi-Reviewer Summary" in report
```

Append to `tests/test_checkpoint_reporting.py`:

```python
def test_markdown_report_includes_multi_reviewer_summary():
    assessment = RiskAssessment(
        level=RiskLevel.MEDIUM,
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
        changed_files=["auth.py"],
        multi_reviewer_summary={
            "reviewer_count": 2,
            "status_counts": {"partial": 2},
            "roles": ["Core Reviewer", "Adversarial Reviewer"],
        },
    )

    assert "## Multi-Reviewer Summary" in report
    assert "Reviewers: 2" in report
    assert "- Core Reviewer" in report
    assert "- Adversarial Reviewer" in report
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_cli_smoke.py::test_cli_multi_reviewer_mode_writes_per_reviewer_artifacts tests/test_checkpoint_reporting.py::test_markdown_report_includes_multi_reviewer_summary -q -p no:cacheprovider
```

Expected: fail because `--reviewer-mode` and report summary are not implemented.

- [ ] **Step 3: Update report rendering**

Modify `render_markdown_report` to accept:

```python
multi_reviewer_summary: dict[str, object] | None = None
```

Render a `## Multi-Reviewer Summary` section with reviewer count, status counts, and roles.

- [ ] **Step 4: Update CLI multi mode**

In `src/review_agent/cli.py`:

- Add parser arg:

```python
review.add_argument("--reviewer-mode", choices=["single", "multi"], default="single")
```

- Import:

```python
from review_agent.orchestrator import multi_reviewer_run_to_dict, run_multi_reviewer
```

- If provider is present and `args.reviewer_mode == "multi"`, call `run_multi_reviewer(...)` with all assignments.
- Write:
  - `multi_reviewer_result.json`
  - `reviewer_<index>_envelope.json`
  - `reviewer_<index>_raw_response.json`
  - `reviewer_<index>_result.json`
- Keep existing single-mode artifact names unchanged for compatibility.
- For tests to get at least two assignments, authentication/security-sensitive changes already make risk medium or above.

- [ ] **Step 5: Run focused tests and verify pass**

Run the focused command from Step 2.

Expected: `2 passed`.

- [ ] **Step 6: Run selected suites**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_orchestrator.py tests/test_cli_smoke.py tests/test_checkpoint_reporting.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 2**

Run:

```powershell
git add src/review_agent/cli.py src/review_agent/reporting.py tests/test_cli_smoke.py tests/test_checkpoint_reporting.py
git commit -m "feat: add multi reviewer cli mode"
```

---

## Task 3: Final Verification

**Files:**
- Modify only files with failing implementation bugs.

- [ ] **Step 1: Scan plan for unfinished markers**

Run:

```powershell
$patterns = @('TB'+'D','TO'+'DO','PLACE'+'HOLDER','x'+'xx','implement '+'later','fill in '+'details') -join '|'
rg -n $patterns docs/superpowers/plans/2026-06-30-multi-agent-orchestrator.md
```

Expected: no matches.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 3: Run fake multi-reviewer smoke**

Run a local fake-provider review with `--reviewer-mode multi` and verify:

- `multi_reviewer_result.json` exists.
- At least two per-reviewer artifact sets exist.
- `report.md` contains `## Multi-Reviewer Summary`.

- [ ] **Step 4: Commit the plan**

Run:

```powershell
git add docs/superpowers/plans/2026-06-30-multi-agent-orchestrator.md
git commit -m "docs: plan multi agent orchestrator"
```

## Self-review checklist

- Spec coverage:
  - Multiple independent assignments run through isolated reviewer envelopes.
  - Each reviewer gets its own trace ID and artifacts.
  - CLI preserves single-reviewer mode and adds explicit multi mode.
  - No cross-reviewer hidden reasoning is shared.
- Type consistency:
  - `run_multi_reviewer()` returns `MultiReviewerRun`.
  - `multi_reviewer_run_to_dict()` contains reviewer count, status counts, roles, trace IDs, and results.
  - Report consumes a compact multi-reviewer summary, not full envelopes.
- Scope check:
  - No real parallelism, model tool loop, reconciler, completion checker, dynamic follow-up dispatch, or eval harness is included.

## Execution handoff

This plan is intended for inline execution in this session. Subagent-driven execution remains acceptable if quota is available, but direct TDD checkpoints are enough for this vertical slice.

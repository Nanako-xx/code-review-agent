# M1 Spec Schema Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the existing M1 local review foundation with the latest main design spec: `explicit/inferred` intent sources, `uncertainties`, risk `signal_refs`, Observation-first terminology, base/head read boundaries, and reviewer assignments that receive initial context rather than prebuilt evidence.

**Architecture:** This plan keeps the current small Python package structure and performs a schema migration in place. The Runtime remains the deterministic control layer; model-facing context is still assembled into the four standard invocation inputs (`system`, `tools`, `messages`, `parameters`), but the payload vocabulary changes from old evidence/unknown names to current spec terms. This plan does not implement the full LLM reviewer loop, repository intelligence layer, or SWE-PRBench eval harness.

**Tech Stack:** Python 3.11+, stdlib `dataclasses`, `argparse`, `subprocess`, `json`, `pathlib`, `pytest`, Git CLI, Markdown/JSON artifacts.

---

## Scope and sequencing

This plan is a migration slice after the first M1 foundation. It should be executed before building the real LLM reviewer loop, because later code generation should target the current spec vocabulary.

In scope:

- Rename `IntentSource.DECLARED` / `LINKED_SOURCE` to `IntentSource.EXPLICIT`.
- Keep `IntentSource.INFERRED` as the source for LLM/system-inferred intent before user confirmation.
- Keep `IntentStatus` as `sufficient | partial | insufficient`.
- Rename `unknowns` to `uncertainties`.
- Rename risk-stage `evidence_refs` to `signal_refs`.
- Rename quality gate `evidence_ref` to `observation_ref`.
- Replace `Assignment.provided_evidence_refs` and loose `code_ranges` with a structured `InitialContext`.
- Update reviewer context assembly to use Observation terminology and avoid pre-seeded evidence.
- Update CLI artifacts and reports to use current schema names.
- Add regression tests that `--focus` is a review preference, not intent.
- Add regression tests for base/head authorized context wording in model-facing tools.

Out of scope:

- Real LLM provider integration.
- Multi-turn Reviewer Agent Loop.
- Repository Intelligence with AST/LSP.
- Observation Store implementation beyond artifact/schema naming.
- Evidence Reconciler and Completion Checker.
- Eval runner or SWE-PRBench adapter.

Current workspace note: `docs/superpowers/specs/2026-06-22-evidence-driven-multi-agent-code-review-design.md` may contain uncommitted design edits. During implementation, do not stage that spec file unless the user explicitly asks. Each commit command below stages only implementation files and tests.

## File structure

- Modify `src/review_agent/models.py`: migrate dataclass and enum names to the current spec vocabulary.
- Modify `src/review_agent/intent.py`: build Intent Packet with `EXPLICIT` / `INFERRED` sources and `uncertainties`; keep `review_focus` out of Intent Packet.
- Modify `src/review_agent/risk.py`: consume `intent_uncertainties`; emit `signal_refs`; make local fallback risk signals less dependent on file count alone.
- Modify `src/review_agent/runtime.py`: build assignments with structured `InitialContext` instead of provided evidence refs.
- Modify `src/review_agent/context.py`: assemble reviewer messages with Observation and initial-context terminology.
- Modify `src/review_agent/reporting.py`: render uncertainties and risk signals.
- Modify `src/review_agent/cli.py`: persist current schema artifacts and keep `--focus` as review preference.
- Modify tests under `tests/`: update old schema assertions and add regressions for the spec clarifications.

---

## Task 1: Core model schema migration

**Files:**
- Modify: `src/review_agent/models.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for current schema vocabulary**

Replace the relevant old assertions in `tests/test_models.py` with tests that use the new model names:

```python
from review_agent.models import (
    Assignment,
    ContractItemStatus,
    InitialContext,
    IntentPacket,
    IntentSource,
    IntentStatus,
    QualityGateResult,
    ReviewProfile,
    ReviewRequest,
    RiskAssessment,
    RiskLevel,
)


def test_review_request_requires_base_and_head():
    request = ReviewRequest(
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
        user_intent="tighten auth checks",
        review_focus="backward compatibility",
    )

    assert request.base_revision == "main"
    assert request.head_revision == "HEAD"
    assert request.user_intent == "tighten auth checks"
    assert request.review_focus == "backward compatibility"


def test_intent_packet_tracks_source_status_and_uncertainties():
    packet = IntentPacket(
        goal="Add idempotency to payment callback",
        acceptance_criteria=["duplicate callbacks are safe"],
        scope=["payment callback"],
        constraints=["do not double charge"],
        sources={"goal": IntentSource.INFERRED},
        status=IntentStatus.PARTIAL,
        uncertainties=["whether duplicate callback should return 200 or 409"],
    )

    assert IntentSource.EXPLICIT.value == "explicit"
    assert IntentSource.INFERRED.value == "inferred"
    assert packet.sources["goal"] is IntentSource.INFERRED
    assert packet.status is IntentStatus.PARTIAL
    assert packet.uncertainties == ["whether duplicate callback should return 200 or 409"]


def test_quality_gate_uses_observation_ref_name():
    result = QualityGateResult(
        name="python_compile",
        status="passed",
        command=["python", "-m", "compileall"],
        summary="compiled 2 files",
        observation_ref="O-quality-python-compile",
    )

    assert result.observation_ref == "O-quality-python-compile"


def test_risk_assessment_uses_signal_refs_and_uncertainties():
    assessment = RiskAssessment(
        level=RiskLevel.HIGH,
        dimensions={"impact": "sensitive path"},
        reasons=["sensitive path changed: auth.py"],
        signal_refs=["diff:auth.py", "quality_gate:python_compile"],
        uncertainties=["acceptance criteria are not explicitly declared"],
        suggested_focus=["caller compatibility"],
    )

    assert assessment.signal_refs == ["diff:auth.py", "quality_gate:python_compile"]
    assert assessment.uncertainties == ["acceptance criteria are not explicitly declared"]


def test_assignment_has_structured_initial_context():
    assignment = Assignment(
        role="Caller Compatibility Reviewer",
        mission="Inspect callers affected by changed public API",
        assignment_reason=["public API changed", "legacy callers exist"],
        assigned_contract=["regression_safety"],
        required_checks=["inspect direct callers or record why unavailable"],
        initial_context=InitialContext(
            changed_files=["src/api.py"],
            diff_ranges=["src/api.py:10-30"],
            code_ranges=["src/api.py:10-30"],
            quality_gate_summary={"python_compile": "passed"},
            observation_refs=["O-diff-api"],
        ),
        max_turns=8,
        max_tool_calls=20,
    )

    assert assignment.initial_context.changed_files == ["src/api.py"]
    assert assignment.initial_context.observation_refs == ["O-diff-api"]


def test_review_profile_maps_risk_to_depth():
    profile = ReviewProfile.for_risk(RiskLevel.HIGH)

    assert profile.reviewer_count == 3
    assert profile.max_turns_per_reviewer == 16
    assert "dynamic_specialist" in profile.reviewer_roles


def test_contract_status_values_are_stable():
    assert ContractItemStatus.COVERED.value == "covered"
    assert ContractItemStatus.NOT_APPLICABLE.value == "not_applicable"
```

- [ ] **Step 2: Run model tests and verify they fail**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_models.py -q -p no:cacheprovider
```

Expected: fail because `InitialContext`, `IntentSource.EXPLICIT`, `uncertainties`, `observation_ref`, and `signal_refs` do not exist yet.

- [ ] **Step 3: Update model definitions**

In `src/review_agent/models.py`, replace the affected definitions with:

```python
class IntentSource(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class IntentStatus(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
```

Update the dataclasses:

```python
@dataclass(frozen=True)
class IntentPacket:
    goal: str | None
    acceptance_criteria: list[str] = field(default_factory=list)
    scope: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    sources: dict[str, IntentSource] = field(default_factory=dict)
    status: IntentStatus = IntentStatus.INSUFFICIENT
    uncertainties: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QualityGateResult:
    name: str
    status: str
    command: list[str]
    summary: str
    observation_ref: str | None = None


@dataclass(frozen=True)
class RiskAssessmentPacket:
    change_summary: dict[str, object]
    deterministic_signals: dict[str, object]
    intent_status: IntentStatus
    intent_uncertainties: list[str]
    diff_excerpt: list[str]


@dataclass(frozen=True)
class RiskAssessment:
    level: RiskLevel
    dimensions: dict[str, str]
    reasons: list[str]
    signal_refs: list[str]
    uncertainties: list[str]
    suggested_focus: list[str]


@dataclass(frozen=True)
class InitialContext:
    changed_files: list[str] = field(default_factory=list)
    diff_ranges: list[str] = field(default_factory=list)
    code_ranges: list[str] = field(default_factory=list)
    quality_gate_summary: dict[str, str] = field(default_factory=dict)
    observation_refs: list[str] = field(default_factory=list)
```

Update `Assignment` to use `InitialContext`:

```python
@dataclass(frozen=True)
class Assignment:
    role: str
    mission: str
    assignment_reason: list[str]
    assigned_contract: list[str]
    required_checks: list[str]
    initial_context: InitialContext
    max_turns: int
    max_tool_calls: int
    repository_permission: str = "read_only"
    command_permission: str = "safe_checks_only"
```

- [ ] **Step 4: Run model tests and verify they pass**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_models.py -q -p no:cacheprovider
```

Expected: `6 passed`.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src/review_agent/models.py tests/test_models.py
git commit -m "refactor: align core review models with spec schema"
```

---

## Task 2: Intent Packet builder migration

**Files:**
- Modify: `src/review_agent/intent.py`
- Modify: `tests/test_intent.py`

- [ ] **Step 1: Write failing tests for explicit/inferred source behavior**

Replace `tests/test_intent.py` with:

```python
from review_agent.intent import build_intent_packet
from review_agent.models import IntentSource, IntentStatus, ReviewRequest


def test_user_intent_is_explicit_and_focus_is_not_intent(git_change_summary):
    request = ReviewRequest(
        repository_path=git_change_summary.repository_path,
        base_revision=git_change_summary.base_revision,
        head_revision=git_change_summary.head_revision,
        user_intent="Add idempotency to payment callback",
        review_focus="duplicate execution and retry safety",
    )

    packet = build_intent_packet(request, git_change_summary)

    assert packet.goal == "Add idempotency to payment callback"
    assert packet.sources["goal"] is IntentSource.EXPLICIT
    assert "review_focus" not in packet.sources
    assert "duplicate execution and retry safety" not in packet.acceptance_criteria
    assert "acceptance criteria are not explicitly declared" in packet.uncertainties


def test_missing_user_intent_creates_inferred_goal(git_change_summary):
    request = ReviewRequest(
        repository_path=git_change_summary.repository_path,
        base_revision=git_change_summary.base_revision,
        head_revision=git_change_summary.head_revision,
    )

    packet = build_intent_packet(request, git_change_summary)

    assert packet.goal.startswith("Review changes touching")
    assert packet.sources["goal"] is IntentSource.INFERRED
    assert "user did not provide explicit intent" in packet.uncertainties
    assert packet.status is IntentStatus.PARTIAL


def test_empty_change_set_is_insufficient(empty_change_summary):
    request = ReviewRequest(
        repository_path=empty_change_summary.repository_path,
        base_revision=empty_change_summary.base_revision,
        head_revision=empty_change_summary.head_revision,
    )

    packet = build_intent_packet(request, empty_change_summary)

    assert packet.goal is None
    assert packet.status is IntentStatus.INSUFFICIENT
    assert "no changed files were detected" in packet.uncertainties


def test_project_rules_are_explicit_constraints(git_change_summary):
    request = ReviewRequest(
        repository_path=git_change_summary.repository_path,
        base_revision=git_change_summary.base_revision,
        head_revision=git_change_summary.head_revision,
        project_rules=("preserve public API compatibility",),
    )

    packet = build_intent_packet(request, git_change_summary)

    assert packet.constraints == ["preserve public API compatibility"]
    assert packet.sources["constraints"] is IntentSource.EXPLICIT
```

- [ ] **Step 2: Run intent tests and verify they fail**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_intent.py -q -p no:cacheprovider
```

Expected: fail because the code still uses `DECLARED` and `unknowns`.

- [ ] **Step 3: Update `build_intent_packet`**

In `src/review_agent/intent.py`, replace the function body with:

```python
def build_intent_packet(request: ReviewRequest, change_summary: ChangeSummary) -> IntentPacket:
    uncertainties: list[str] = []
    sources: dict[str, IntentSource] = {}

    if request.user_intent:
        goal = request.user_intent
        sources["goal"] = IntentSource.EXPLICIT
    elif change_summary.changed_files:
        files = ", ".join(change_summary.changed_files[:3])
        goal = f"Review changes touching {files}"
        sources["goal"] = IntentSource.INFERRED
        uncertainties.append("user did not provide explicit intent")
    else:
        goal = None
        uncertainties.append("no changed files were detected")

    acceptance_criteria: list[str] = []
    uncertainties.append("acceptance criteria are not explicitly declared")

    scope = list(change_summary.changed_files)
    if scope:
        sources["scope"] = IntentSource.INFERRED

    constraints: list[str] = []
    if request.project_rules:
        constraints.extend(request.project_rules)
        sources["constraints"] = IntentSource.EXPLICIT
    else:
        uncertainties.append("project constraints are not explicitly declared")

    if goal is None:
        status = IntentStatus.INSUFFICIENT
    elif acceptance_criteria and constraints and sources.get("goal") is IntentSource.EXPLICIT:
        status = IntentStatus.SUFFICIENT
    else:
        status = IntentStatus.PARTIAL

    return IntentPacket(
        goal=goal,
        acceptance_criteria=acceptance_criteria,
        scope=scope,
        constraints=constraints,
        sources=sources,
        status=status,
        uncertainties=uncertainties,
    )
```

- [ ] **Step 4: Run intent tests and verify they pass**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_intent.py -q -p no:cacheprovider
```

Expected: all intent tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src/review_agent/intent.py tests/test_intent.py
git commit -m "refactor: distinguish explicit and inferred intent sources"
```

---

## Task 3: Risk packet and local assessor migration

**Files:**
- Modify: `src/review_agent/risk.py`
- Modify: `tests/test_risk.py`

- [ ] **Step 1: Write failing risk tests**

Update `tests/test_risk.py` so the assertions use `intent_uncertainties`, `signal_refs`, and `uncertainties`:

```python
from review_agent.models import ReviewRequest, RiskAssessment, RiskLevel
from review_agent.intent import build_intent_packet
from review_agent.risk import LocalRiskAssessor, build_risk_packet
from review_agent.runtime import build_assignments


def test_risk_packet_carries_intent_uncertainties(git_change_summary):
    request = ReviewRequest(
        repository_path=git_change_summary.repository_path,
        base_revision=git_change_summary.base_revision,
        head_revision=git_change_summary.head_revision,
    )
    intent = build_intent_packet(request, git_change_summary)

    packet = build_risk_packet(git_change_summary, intent, {"python_compile": "passed"})

    assert packet.intent_status == intent.status
    assert packet.intent_uncertainties == intent.uncertainties
    assert packet.deterministic_signals["quality_gates"] == {"python_compile": "passed"}


def test_failed_quality_gate_produces_signal_ref(git_change_summary):
    request = ReviewRequest(
        repository_path=git_change_summary.repository_path,
        base_revision=git_change_summary.base_revision,
        head_revision=git_change_summary.head_revision,
    )
    intent = build_intent_packet(request, git_change_summary)
    packet = build_risk_packet(git_change_summary, intent, {"python_compile": "failed"})

    assessment = LocalRiskAssessor().assess(packet)

    assert assessment.level is RiskLevel.HIGH
    assert "quality_gate:python_compile" in assessment.signal_refs
    assert assessment.uncertainties == intent.uncertainties


def test_many_doc_files_do_not_become_medium_risk_by_count_only(git_many_doc_change_summary):
    request = ReviewRequest(
        repository_path=git_many_doc_change_summary.repository_path,
        base_revision=git_many_doc_change_summary.base_revision,
        head_revision=git_many_doc_change_summary.head_revision,
    )
    intent = build_intent_packet(request, git_many_doc_change_summary)
    packet = build_risk_packet(git_many_doc_change_summary, intent, {"python_compile": "passed"})

    assessment = LocalRiskAssessor().assess(packet)

    assert assessment.level is RiskLevel.LOW
    assert "many files changed" not in " ".join(assessment.reasons)


def test_sensitive_path_still_high_risk(git_sensitive_change_summary):
    request = ReviewRequest(
        repository_path=git_sensitive_change_summary.repository_path,
        base_revision=git_sensitive_change_summary.base_revision,
        head_revision=git_sensitive_change_summary.head_revision,
    )
    intent = build_intent_packet(request, git_sensitive_change_summary)
    packet = build_risk_packet(git_sensitive_change_summary, intent, {"python_compile": "passed"})

    assessment = LocalRiskAssessor().assess(packet)

    assert assessment.level is RiskLevel.HIGH
    assert any(ref.startswith("changed_file:") for ref in assessment.signal_refs)


def test_runtime_assignments_use_initial_context():
    assessment = RiskAssessment(
        level=RiskLevel.MEDIUM,
        dimensions={"impact": "derived from changed paths"},
        reasons=["public behavior may change"],
        signal_refs=["diff:src/app.py"],
        uncertainties=["project constraints are not explicitly declared"],
        suggested_focus=["test adequacy"],
    )

    assignments = build_assignments(assessment)

    assert len(assignments) == 2
    assert assignments[0].initial_context.observation_refs == ["diff:src/app.py"]
```

- [ ] **Step 2: Add or update fixtures needed by the tests**

If `git_many_doc_change_summary` and `git_sensitive_change_summary` do not exist in `tests/conftest.py`, add fixtures that create temporary Git repos and return `collect_change_summary(...)`.

Use this pattern:

```python
@pytest.fixture
def git_sensitive_change_summary(tmp_path):
    repo = tmp_path / "sensitive-repo"
    repo.mkdir()
    run_git(repo, "init")
    (repo / "auth.py").write_text("def check(user):\n    return user.is_admin\n", encoding="utf-8")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-m", "base")
    base = run_git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "auth.py").write_text("def check(user):\n    return True\n", encoding="utf-8")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-m", "head")
    head = run_git(repo, "rev-parse", "HEAD").stdout.strip()
    return collect_change_summary(repo, base, head)


@pytest.fixture
def git_many_doc_change_summary(tmp_path):
    repo = tmp_path / "docs-repo"
    repo.mkdir()
    run_git(repo, "init")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-m", "base")
    base = run_git(repo, "rev-parse", "HEAD").stdout.strip()
    docs = repo / "docs"
    docs.mkdir()
    for index in range(10):
        (docs / f"note-{index}.md").write_text(f"note {index}\n", encoding="utf-8")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-m", "head")
    head = run_git(repo, "rev-parse", "HEAD").stdout.strip()
    return collect_change_summary(repo, base, head)
```

- [ ] **Step 3: Run risk tests and verify they fail**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_risk.py -q -p no:cacheprovider
```

Expected: fail because risk code still emits `evidence_refs` and `unknowns`.

- [ ] **Step 4: Update risk packet construction and assessor output**

In `src/review_agent/risk.py`, update `build_risk_packet`:

```python
def build_risk_packet(
    change_summary: ChangeSummary,
    intent_packet: IntentPacket,
    quality_gate_status: dict[str, str],
) -> RiskAssessmentPacket:
    return RiskAssessmentPacket(
        change_summary={
            "repository_path": change_summary.repository_path,
            "base_revision": change_summary.base_revision,
            "head_revision": change_summary.head_revision,
            "changed_files": change_summary.changed_files,
            "diff_stat": change_summary.diff_stat,
        },
        deterministic_signals={
            "quality_gates": quality_gate_status,
            "changed_file_count": len(change_summary.changed_files),
        },
        intent_status=intent_packet.status,
        intent_uncertainties=list(intent_packet.uncertainties),
        diff_excerpt=list(change_summary.diff_excerpt[:80]),
    )
```

Update `LocalRiskAssessor.assess` so it builds `signal_refs`:

```python
failed_gates = [name for name, status in quality_gates.items() if status == "failed"]
signal_refs: list[str] = []

if failed_gates:
    level = RiskLevel.HIGH
    reasons = [f"quality gate failed: {name}" for name in failed_gates]
    signal_refs.extend(f"quality_gate:{name}" for name in failed_gates)
    focus = ["failed quality gate", "regression safety"]
elif sensitive_files:
    level = RiskLevel.HIGH
    reasons = [f"sensitive path changed: {path}" for path in sensitive_files]
    signal_refs.extend(f"changed_file:{path}" for path in sensitive_files)
    focus = ["caller compatibility", "regression safety", "test adequacy"]
elif len(changed_files) > 8 and not _all_doc_like(changed_files):
    level = RiskLevel.MEDIUM
    reasons = [f"many non-documentation files changed: {len(changed_files)}"]
    signal_refs.append("changed_file_count")
    focus = ["blast radius", "test adequacy"]
else:
    level = RiskLevel.LOW
    reasons = ["small or documentation-only non-sensitive change set"]
    focus = ["intent alignment", "changed file sanity"]

return RiskAssessment(
    level=level,
    dimensions={
        "impact": "derived from changed paths and quality gates",
        "blast_radius": "derived from changed file semantics and count",
        "reversibility": "not assessed by local fallback",
        "uncertainty": "derived from intent uncertainties",
        "verification_strength": "derived from quality gates",
    },
    reasons=reasons,
    signal_refs=signal_refs,
    uncertainties=list(packet.intent_uncertainties),
    suggested_focus=focus,
)
```

Add the helper:

```python
def _all_doc_like(paths: list[str]) -> bool:
    doc_suffixes = (".md", ".rst", ".txt", ".adoc")
    doc_prefixes = ("docs/", "doc/")
    return bool(paths) and all(
        path.lower().endswith(doc_suffixes) or path.lower().startswith(doc_prefixes)
        for path in paths
    )
```

- [ ] **Step 5: Run risk tests and verify they pass**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_risk.py -q -p no:cacheprovider
```

Expected: all risk tests pass.

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
git add src/review_agent/risk.py src/review_agent/runtime.py tests/test_risk.py tests/conftest.py
git commit -m "refactor: use risk signals and uncertainties"
```

---

## Task 4: Runtime assignment initial context

**Files:**
- Modify: `src/review_agent/runtime.py`
- Modify: `tests/test_risk.py`

- [ ] **Step 1: Write a focused assignment test**

Add this test to `tests/test_risk.py` if not already covered by Task 3:

```python
def test_assignments_receive_initial_context_not_raw_evidence():
    assessment = RiskAssessment(
        level=RiskLevel.LOW,
        dimensions={"impact": "local"},
        reasons=["small or documentation-only non-sensitive change set"],
        signal_refs=["diff:README.md"],
        uncertainties=["acceptance criteria are not explicitly declared"],
        suggested_focus=["intent alignment"],
    )

    assignment = build_assignments(assessment)[0]

    assert assignment.initial_context.observation_refs == ["diff:README.md"]
    assert assignment.initial_context.quality_gate_summary == {}
    assert not hasattr(assignment, "provided_evidence_refs")
    assert not hasattr(assignment, "code_ranges")
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_risk.py::test_assignments_receive_initial_context_not_raw_evidence -q -p no:cacheprovider
```

Expected: fail because assignments still use `provided_evidence_refs`.

- [ ] **Step 3: Update `build_assignments`**

In `src/review_agent/runtime.py`, import `InitialContext`:

```python
from review_agent.models import Assignment, InitialContext, RiskAssessment, ReviewProfile
```

Update the `Assignment(...)` construction:

```python
Assignment(
    role=role,
    mission=_mission_for_role(role),
    assignment_reason=list(risk_assessment.reasons),
    assigned_contract=_contract_for_role(role),
    required_checks=[
        "map changed behavior to intent",
        "inspect direct observations for assigned contract items",
        "record unavailable observations as uncertainty",
    ],
    initial_context=InitialContext(
        observation_refs=list(risk_assessment.signal_refs),
    ),
    max_turns=profile.max_turns_per_reviewer,
    max_tool_calls=profile.max_tool_calls_per_reviewer,
)
```

- [ ] **Step 4: Run runtime-related tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_models.py tests/test_risk.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add src/review_agent/runtime.py tests/test_risk.py
git commit -m "refactor: pass initial context to reviewer assignments"
```

---

## Task 5: Reviewer envelope Observation terminology

**Files:**
- Modify: `src/review_agent/context.py`
- Modify: `tests/test_context.py`

- [ ] **Step 1: Write failing context tests**

Replace `tests/test_context.py` with:

```python
from review_agent.context import build_reviewer_envelope
from review_agent.models import Assignment, InitialContext, IntentPacket, IntentSource, IntentStatus


def test_reviewer_envelope_uses_standard_four_inputs():
    assignment = Assignment(
        role="Core Reviewer",
        mission="Check intent alignment",
        assignment_reason=["small non-sensitive change set"],
        assigned_contract=["intent_alignment"],
        required_checks=["map changed behavior to intent"],
        initial_context=InitialContext(
            changed_files=["app.py"],
            diff_ranges=["app.py:1-5"],
            code_ranges=["app.py:1-5"],
            quality_gate_summary={"python_compile": "passed"},
            observation_refs=["O-diff-app"],
        ),
        max_turns=6,
        max_tool_calls=12,
    )
    intent = IntentPacket(
        goal="Review changes touching app.py",
        sources={"goal": IntentSource.INFERRED},
        status=IntentStatus.PARTIAL,
        uncertainties=["user did not provide explicit intent"],
    )

    envelope = build_reviewer_envelope(
        assignment=assignment,
        intent=intent,
        code_snippets={"app.py:1-5": "def add(a, b):\n    return a + b\n"},
        observations={"O-diff-app": "app.py changed between base and head"},
        trace_id="trace-1",
    )

    assert set(envelope.__dict__.keys()) == {"system", "tools", "messages", "parameters"}
    assert "tools" not in envelope.messages[0]
    assert envelope.parameters["trace_id"] == "trace-1"
    assert "Review Contract" in envelope.system
    assert "risk_level" not in str(envelope.messages)
    assert "Assignment" in envelope.messages[0]["content"]
    assert "Observation Summary" in envelope.messages[0]["content"]
    assert "Initial Context" in envelope.messages[0]["content"]
    assert "Evidence" not in envelope.messages[0]["content"]
    assert "explicit" not in envelope.messages[0]["content"]
    assert "inferred" in envelope.messages[0]["content"]


def test_reviewer_tools_describe_head_default_and_base_head_comparison():
    assignment = Assignment(
        role="Core Reviewer",
        mission="Check intent alignment",
        assignment_reason=["small non-sensitive change set"],
        assigned_contract=["intent_alignment"],
        required_checks=["map changed behavior to intent"],
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
        trace_id="trace-2",
    )

    tool_text = " ".join(str(tool) for tool in envelope.tools)
    assert "head revision" in tool_text
    assert "base and head" in tool_text
```

- [ ] **Step 2: Run context tests and verify they fail**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_context.py -q -p no:cacheprovider
```

Expected: fail because `build_reviewer_envelope` still takes `evidence` and renders an `Evidence` block.

- [ ] **Step 3: Update `build_reviewer_envelope` signature and message blocks**

In `src/review_agent/context.py`, update the function signature:

```python
def build_reviewer_envelope(
    assignment: Assignment,
    intent: IntentPacket,
    code_snippets: dict[str, str],
    observations: dict[str, str],
    trace_id: str,
) -> ModelInvocationEnvelope:
```

Update the content assembly:

```python
content = "\n\n".join(
    [
        _assignment_block(assignment),
        _intent_block(intent),
        _initial_context_block(assignment),
        _code_block(code_snippets),
        _observation_block(observations),
        _completion_block(assignment),
    ]
)
```

Update tool descriptions:

```python
tools=[
    {
        "name": "search_code",
        "description": "Search repository text using a read-only index of the reviewed head revision.",
    },
    {
        "name": "read_range",
        "description": "Read a bounded range from a repository file at the reviewed head revision.",
    },
    {
        "name": "compare_base_head",
        "description": "Read Runtime-authorized base and head file ranges or diff hunks for comparison.",
    },
]
```

Update `_intent_block`:

```python
def _intent_block(intent: IntentPacket) -> str:
    sources = ", ".join(f"{key}={value.value}" for key, value in intent.sources.items())
    return "\n".join(
        [
            "Intent Packet",
            f"Goal: {intent.goal}",
            f"Status: {intent.status.value}",
            f"Sources: {sources}",
            f"Uncertainties: {'; '.join(intent.uncertainties)}",
        ]
    )
```

Add `_initial_context_block` and replace `_evidence_block`:

```python
def _initial_context_block(assignment: Assignment) -> str:
    context = assignment.initial_context
    return "\n".join(
        [
            "Initial Context",
            f"Changed Files: {', '.join(context.changed_files)}",
            f"Diff Ranges: {', '.join(context.diff_ranges)}",
            f"Code Ranges: {', '.join(context.code_ranges)}",
            f"Quality Gates: {context.quality_gate_summary}",
            f"Observation Refs: {', '.join(context.observation_refs)}",
        ]
    )


def _observation_block(observations: dict[str, str]) -> str:
    parts = ["Observation Summary"]
    for observation_id, summary in observations.items():
        parts.append(f"{observation_id}: {summary}")
    return "\n".join(parts)
```

Update `_completion_block`:

```python
def _completion_block(assignment: Assignment) -> str:
    return "\n".join(
        [
            "Completion Rules",
            "You may request completion only after addressing every assigned contract item.",
            "If a required check cannot be performed, record the reason as an uncertainty.",
            "Findings must cite observation IDs as evidence_refs in the final structured output.",
        ]
    )
```

- [ ] **Step 4: Run context tests and verify they pass**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_context.py -q -p no:cacheprovider
```

Expected: `2 passed`.

- [ ] **Step 5: Commit Task 5**

Run:

```powershell
git add src/review_agent/context.py tests/test_context.py
git commit -m "refactor: use observations in reviewer context"
```

---

## Task 6: CLI artifacts and report vocabulary

**Files:**
- Modify: `src/review_agent/cli.py`
- Modify: `src/review_agent/reporting.py`
- Modify: `tests/test_checkpoint_reporting.py`
- Modify: `tests/test_cli_smoke.py`

- [ ] **Step 1: Write failing report tests**

Update report assertions in `tests/test_checkpoint_reporting.py`:

```python
from review_agent.models import RiskAssessment, RiskLevel
from review_agent.reporting import render_markdown_report


def test_markdown_report_contains_risk_signals_and_uncertainties():
    assessment = RiskAssessment(
        level=RiskLevel.HIGH,
        dimensions={"impact": "sensitive path"},
        reasons=["sensitive path changed: auth.py"],
        signal_refs=["changed_file:auth.py"],
        uncertainties=["user did not provide explicit intent"],
        suggested_focus=["regression safety"],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        risk_assessment=assessment,
        changed_files=["auth.py"],
    )

    assert "Risk level: high" in report
    assert "## Risk Signals" in report
    assert "- changed_file:auth.py" in report
    assert "## Uncertainties" in report
    assert "- user did not provide explicit intent" in report
```

- [ ] **Step 2: Write failing CLI smoke assertions**

Update `tests/test_cli_smoke.py` to assert current artifact names and fields:

```python
def test_cli_review_writes_current_schema_artifacts(sample_git_repo, monkeypatch, capsys):
    repo, base, head = sample_git_repo
    monkeypatch.setenv("PYTHONPATH", "src")

    exit_code = main(
        [
            "review",
            "--repo",
            str(repo),
            "--base",
            base,
            "--head",
            head,
            "--focus",
            "regression safety",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    run_dir = Path(output.strip().split(": ", 1)[1])

    intent = json.loads((run_dir / "intent.json").read_text(encoding="utf-8"))
    risk = json.loads((run_dir / "risk.json").read_text(encoding="utf-8"))
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))

    assert "uncertainties" in intent
    assert "unknowns" not in intent
    assert "signal_refs" in risk
    assert "evidence_refs" not in risk
    assert "initial_context" in assignments["assignments"][0]
    assert "provided_evidence_refs" not in assignments["assignments"][0]
    assert (run_dir / "report.md").exists()
```

Keep fixture names aligned with the existing `tests/test_cli_smoke.py`; if it uses a different fixture name, update only the test function body.

- [ ] **Step 3: Run reporting and CLI tests and verify they fail**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_checkpoint_reporting.py tests/test_cli_smoke.py -q -p no:cacheprovider
```

Expected: fail because reports and CLI artifacts still use `unknowns` / `evidence_refs`.

- [ ] **Step 4: Update report rendering**

In `src/review_agent/reporting.py`, update field access:

```python
signals = "\n".join(f"- {ref}" for ref in risk_assessment.signal_refs) or "- No risk signals recorded"
uncertainties = "\n".join(
    f"- {uncertainty}" for uncertainty in risk_assessment.uncertainties
) or "- No unresolved uncertainties recorded"
```

Add a `Risk Signals` section before `Uncertainties`:

```python
"## Risk Signals",
"",
signals,
"",
"## Uncertainties",
"",
uncertainties,
```

- [ ] **Step 5: Update CLI serialization call sites**

In `src/review_agent/cli.py`, no custom conversion should reference old names after model migration. Keep these artifact writes:

```python
store.write_json("request.json", asdict(request))
store.write_json("intent.json", asdict(intent))
store.write_json("risk_packet.json", asdict(risk_packet))
store.write_json("risk.json", asdict(risk_assessment))
store.write_json("assignments.json", {"assignments": [asdict(item) for item in assignments]})
store.write_json("quality_gates.json", {"results": [asdict(item) for item in quality_results]})
```

Do not add `review_focus` to `intent.json` except through `request.json`.

- [ ] **Step 6: Run reporting and CLI tests and verify they pass**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_checkpoint_reporting.py tests/test_cli_smoke.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 6**

Run:

```powershell
git add src/review_agent/cli.py src/review_agent/reporting.py tests/test_checkpoint_reporting.py tests/test_cli_smoke.py
git commit -m "refactor: persist current review schema artifacts"
```

---

## Task 7: Full regression and manual smoke

**Files:**
- Modify only if failures reveal missed references.

- [ ] **Step 1: Search for old schema names**

Run:

```powershell
rg -n -e "DECLARED" -e "LINKED_SOURCE" -e "unknowns" -e "provided_evidence_refs" -e "evidence_ref" -e "evidence_refs" src tests
```

Expected:

- No `DECLARED`, `LINKED_SOURCE`, `unknowns`, `provided_evidence_refs`, or singular `evidence_ref`.
- `evidence_refs` may appear only in reviewer-output wording, not as a runtime input field.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 3: Run CLI help smoke**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path; python -m review_agent --help
```

Expected: output contains `review`.

- [ ] **Step 4: Run local review smoke**

Create a small temporary repo:

```powershell
$sample = "C:\tmp\review-agent-schema-smoke"
if (Test-Path $sample) { Remove-Item -LiteralPath $sample -Recurse -Force }
New-Item -ItemType Directory -Path $sample | Out-Null
Set-Location $sample
git init
git config user.email "test@example.com"
git config user.name "Test User"
Set-Content -Encoding UTF8 -Path "auth.py" -Value "def is_admin(user):`n    return user.role == 'admin'`n"
git add .
git commit -m "base"
$base = git rev-parse HEAD
Set-Content -Encoding UTF8 -Path "auth.py" -Value "def is_admin(user):`n    return True`n"
git add .
git commit -m "head"
$head = git rev-parse HEAD
Set-Location "D:\Agent\code review agent"
$env:PYTHONPATH=(Resolve-Path .\src).Path
python -m review_agent review --repo $sample --base $base --head $head --intent "Refactor admin check" --focus "authorization regression" --non-interactive
```

Expected: command prints `Review foundation completed: ...`.

- [ ] **Step 5: Inspect smoke artifacts**

Open the newest run directory under `C:\tmp\review-agent-schema-smoke\.review-agent\runs\` and check:

```powershell
$run = Get-ChildItem "C:\tmp\review-agent-schema-smoke\.review-agent\runs" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content -Raw -LiteralPath (Join-Path $run.FullName "intent.json")
Get-Content -Raw -LiteralPath (Join-Path $run.FullName "risk.json")
Get-Content -Raw -LiteralPath (Join-Path $run.FullName "assignments.json")
```

Expected:

- `intent.json` contains `uncertainties`.
- `risk.json` contains `signal_refs`.
- `assignments.json` contains `initial_context`.
- None of these artifacts contain `unknowns`, `evidence_refs`, or `provided_evidence_refs` as runtime input fields.

- [ ] **Step 6: Commit Task 7 if fixes were needed**

If Step 1 through Step 5 required code or test fixes, run:

```powershell
git add src tests
git commit -m "test: verify current schema alignment"
```

If no files changed after Task 6, skip this commit.

---

## Self-review checklist

- Spec coverage:
  - `--focus` as Review Preference: Task 2.
  - `inferred` as `IntentSource`: Tasks 1 and 2.
  - `IntentStatus` unchanged: Task 1.
  - `uncertainties`: Tasks 1, 2, 3, 6, 7.
  - risk `signal_refs`: Tasks 1, 3, 6, 7.
  - Observation terminology: Tasks 1, 5, 6, 7.
  - base/head authorized context: Task 5 and Task 7.
  - implementation source-of-truth migration: all tasks.
- Placeholder scan:
  - This plan contains no unresolved placeholder markers and no open-ended implementation instruction without a concrete test or command.
- Type consistency:
  - `IntentSource.EXPLICIT` / `IntentSource.INFERRED`.
  - `IntentPacket.uncertainties`.
  - `RiskAssessment.signal_refs`.
  - `RiskAssessment.uncertainties`.
  - `Assignment.initial_context`.
  - `InitialContext.observation_refs`.
  - `QualityGateResult.observation_ref`.

## Execution handoff

Plan complete when this file is saved. Recommended execution is Subagent-Driven, one task per worker, with a review checkpoint after each commit. Inline execution is also viable because this is a contained schema migration.

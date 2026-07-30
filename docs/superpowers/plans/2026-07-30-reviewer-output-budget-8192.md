# Reviewer 8192 Output Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise every Runtime-owned Reviewer per-call output ceiling to 8192 tokens without changing any other Reviewer limit or validation rule.

**Architecture:** Keep the existing `ReviewProfile`, `Assignment`, and supplemental budget models. Change the shared Reviewer output default and the low/medium profile literals to 8192; high/critical profiles already use the required value. Make supplemental Reviewer caps inherit that shared default, and keep legacy hydration bound to the same constant.

**Tech Stack:** Python 3.11, dataclasses, pytest

---

### Task 1: Raise and lock the Reviewer output budget

**Files:**
- Modify: `src/review_agent/models.py:1177`
- Modify: `src/review_agent/models.py:1372-1390`
- Modify: `src/review_agent/supplemental.py:143-148`
- Modify: `tests/test_models.py:216-229`
- Modify: `tests/test_supplemental.py`
- Verify: `tests/test_hydration.py:497-514`

- [ ] **Step 1: Tighten the profile regression before changing production code**

Replace the loose positive-value assertion in
`test_review_profiles_expand_every_runtime_budget_by_risk` with the exact policy:

```python
def test_review_profiles_expand_every_runtime_budget_by_risk():
    profiles = [ReviewProfile.for_risk(level) for level in RiskLevel]

    assert [profile.max_output_tokens for profile in profiles] == [8192] * len(RiskLevel)
    assert [profile.max_total_tokens for profile in profiles] == sorted(
        profile.max_total_tokens for profile in profiles
    )
    assert [profile.max_elapsed_seconds for profile in profiles] == sorted(
        profile.max_elapsed_seconds for profile in profiles
    )
    for profile in profiles:
        assert profile.max_total_tokens > 0
        assert profile.max_elapsed_seconds > 0
        assert profile.max_provider_attempts > 0
```

Also add an exact literal regression proving that the shared default and a default
`Assignment` resolve to `8192`, rather than comparing one mutable default to another.
Add a supplemental regression that compiles a task with default `ReviewerBudgetCaps`
and asserts its Assignment receives `8192`.

```python
def test_reviewer_output_default_is_exactly_8192():
    assert DEFAULT_REVIEWER_MAX_OUTPUT_TOKENS == 8192
    assignment = Assignment(
        role="core",
        mission="review",
        assignment_reason=[],
        assigned_contract=[],
        required_checks=[],
        initial_context=InitialContext(),
        max_turns=1,
        max_tool_calls=0,
    )
    assert assignment.max_output_tokens == 8192


def test_default_supplemental_reviewer_uses_shared_8192_output_budget():
    plan = compile_supplemental_plan(
        review_id="review-default-budget",
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        risk_level=RiskLevel.MEDIUM,
        wave_index=1,
        trigger_digest="trigger-default-budget",
        requests=[_request()],
    )

    assert plan.tasks[0].assignment.max_output_tokens == 8192
```

- [ ] **Step 2: Run the focused regression and observe the policy mismatch**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_models.py::test_review_profiles_expand_every_runtime_budget_by_risk `
  tests/test_models.py::test_reviewer_output_default_is_exactly_8192 `
  tests/test_supplemental.py::test_default_supplemental_reviewer_uses_shared_8192_output_budget `
  -q -p no:cacheprovider --basetemp 'D:\tmp\reviewer-output-8192-red'
```

Expected before the production changes: FAIL because low/medium profiles, the shared
default, and the default supplemental Reviewer still expose 4096.

- [ ] **Step 3: Raise the shared default and low/medium profile values**

In `src/review_agent/models.py`, make the following exact changes:

```python
DEFAULT_REVIEWER_MAX_OUTPUT_TOKENS = 8192
```

Set both `RiskLevel.LOW` and `RiskLevel.MEDIUM` profile constructors to:

```python
max_output_tokens=8192,
```

In `src/review_agent/supplemental.py`, import
`DEFAULT_REVIEWER_MAX_OUTPUT_TOKENS` from `review_agent.models` and use it as the
`ReviewerBudgetCaps.max_output_tokens` default:

```python
max_output_tokens: int = DEFAULT_REVIEWER_MAX_OUTPUT_TOKENS
```

Do not change high/critical values, total-token budgets, elapsed-time budgets,
provider-attempt budgets, parsing, Review Contract validation, or Evidence authorization.

- [ ] **Step 4: Run focused model and hydration verification**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_models.py `
  tests/test_supplemental.py `
  tests/test_hydration.py::test_assignment_hydration_adds_runtime_defaults_to_legacy_artifact `
  -q -p no:cacheprovider --basetemp 'D:\tmp\reviewer-output-8192-focused'
```

Expected: PASS.

- [ ] **Step 5: Run Product and Eval suites**

Run Product:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests --ignore=tests/eval `
  -q -p no:cacheprovider --basetemp 'D:\tmp\reviewer-output-8192-product'
```

Run Eval:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval `
  -q -p no:cacheprovider --basetemp 'D:\tmp\reviewer-output-8192-eval'
```

Expected: both commands exit 0.

- [ ] **Step 6: Commit only the budget implementation and regression**

```powershell
git add -- src/review_agent/models.py src/review_agent/supplemental.py tests/test_models.py tests/test_supplemental.py
git commit -m "fix: raise reviewer output budget to 8192"
```

Do not stage `.pytest-tmp/`, `.test-tmp/`, `__pycache__/`, `.eval-data/`, or smoke
artifacts.

### Final verification outside the implementation commit

After Task 1 passes specification and code-quality review, rerun the frozen current-HEAD
DeepSeek smoke from the approved Observation-aware Tool Result Envelope plan. Inspect only
safe structured metadata. Every scheduled Reviewer must be `completed` or `partial`, no
Reviewer may fail because of malformed/truncated JSON or unauthorized Observation IDs,
Semantic Reconciliation must remain accepted, all Session phases must complete without
errors, Memory must remain off, and Completion may be blocked only for insufficient Intent.

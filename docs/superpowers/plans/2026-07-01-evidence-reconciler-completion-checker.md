# Evidence Reconciler Completion Checker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic evidence reconciliation and completion checking so multi-reviewer runs cannot silently approve unsupported findings or failed critical reviewer coverage.

**Architecture:** Keep this slice local and deterministic. A new `evidence.py` validates reviewer finding evidence refs against the Observation Store and produces canonical/rejected findings. A new `completion.py` consumes intent, assignments, reviewer executions, quality results, and reconciliation output to produce a global completion decision. CLI writes `reconciliation.json` and `completion.json`, then report rendering summarizes both.

**Tech Stack:** Python 3.11+, stdlib dataclasses/enums, existing ObservationStore, existing MultiReviewerRun, pytest.

---

## Scope

In scope:

- Validate `ReviewerFinding.evidence_refs` against authorized Observation IDs.
- Reject unsupported findings with reason `unsupported_claim`.
- Preserve supported findings as `canonical_findings`.
- Aggregate exact duplicate supported claims by normalized claim and evidence refs.
- Build deterministic contract coverage rows from reviewer contract assessments.
- Produce completion status and recommendation for multi-reviewer runs.
- Enforce failure policy:
  - Core Reviewer failure prevents `completed`.
  - Specialist/adversarial reviewer failure can be `completed_with_uncertainties`.
  - Unsupported findings force `manual_review`.
- Write `reconciliation.json` and `completion.json` in CLI multi mode.
- Add report sections for evidence reconciliation and completion status.

Out of scope:

- LLM Reconciler Agent.
- Semantic duplicate detection.
- Dynamic follow-up assignment dispatch.
- Final risk reassessment with model judgment.
- GitHub PR comments.

## Files

- Create: `src/review_agent/evidence.py`
- Create: `src/review_agent/completion.py`
- Create: `tests/test_evidence.py`
- Create: `tests/test_completion.py`
- Modify: `src/review_agent/cli.py`
- Modify: `src/review_agent/reporting.py`
- Modify: `tests/test_cli_smoke.py`
- Modify: `tests/test_checkpoint_reporting.py`

## Task 1: Deterministic Evidence Reconciliation

- [ ] **Step 1: Write failing evidence tests**

Create `tests/test_evidence.py` with tests that build reviewer executions directly:

```python
from review_agent.evidence import reconcile_evidence, reconciliation_to_dict
from review_agent.models import ReviewerFinding, ReviewerResult, ReviewerResultStatus
from review_agent.orchestrator import ReviewerExecution
from review_agent.provider import ModelResponse
from tests.test_orchestrator import make_assignment


def execution(index, role, findings):
    assignment = make_assignment(role)
    return ReviewerExecution(
        reviewer_index=index,
        trace_id=f"review-1-reviewer-{index}",
        assignment=assignment,
        envelope=None,
        response=ModelResponse(content="{}", provider_name="fake", model="fake"),
        result=ReviewerResult(
            confirmed_findings=findings,
            investigation_summary=f"{role} done",
            status=ReviewerResultStatus.COMPLETED,
        ),
    )


def finding(claim, refs):
    return ReviewerFinding(
        claim=claim,
        severity="high",
        confidence="high",
        evidence_refs=refs,
        suggested_action="fix it",
    )


def test_reconcile_evidence_rejects_findings_with_missing_evidence_refs():
    reconciliation = reconcile_evidence(
        executions=[execution(0, "Core Reviewer", [finding("Auth bypass", ["O-known", "O-missing"])])],
        authorized_observation_ids={"O-known"},
    )

    assert reconciliation.canonical_findings == []
    assert len(reconciliation.rejected_findings) == 1
    rejected = reconciliation.rejected_findings[0]
    assert rejected.reason == "unsupported_claim"
    assert rejected.missing_evidence_refs == ["O-missing"]
    assert reconciliation.evidence_quality == "unsupported_claims"


def test_reconcile_evidence_keeps_and_deduplicates_supported_findings():
    reconciliation = reconcile_evidence(
        executions=[
            execution(0, "Core Reviewer", [finding("Auth bypass", ["O-auth"])]),
            execution(1, "Adversarial Reviewer", [finding(" auth bypass ", ["O-auth"])]),
        ],
        authorized_observation_ids={"O-auth"},
    )

    payload = reconciliation_to_dict(reconciliation)

    assert payload["evidence_quality"] == "verified"
    assert len(payload["canonical_findings"]) == 1
    assert payload["canonical_findings"][0]["claim"] == "Auth bypass"
    assert payload["canonical_findings"][0]["reviewer_indices"] == [0, 1]
    assert payload["canonical_findings"][0]["roles"] == ["Core Reviewer", "Adversarial Reviewer"]
```

- [ ] **Step 2: Run failing evidence tests**

Run:

```powershell
python -B -m pytest tests/test_evidence.py -q -p no:cacheprovider
```

Expected: fail because `review_agent.evidence` does not exist.

- [ ] **Step 3: Implement `src/review_agent/evidence.py`**

Implement dataclasses:

- `CanonicalFinding`
- `RejectedFinding`
- `ContractCoverage`
- `EvidenceReconciliation`

Implement:

- `reconcile_evidence(executions, authorized_observation_ids)`
- `reconciliation_to_dict(reconciliation)`

Rules:

- Missing or empty finding evidence refs reject the finding as `unsupported_claim`.
- Supported findings are canonical.
- Exact duplicate supported findings use normalized claim plus sorted evidence refs.
- Evidence quality is `verified` when no rejected findings exist, otherwise `unsupported_claims`.

- [ ] **Step 4: Run evidence tests and commit**

Run:

```powershell
python -B -m pytest tests/test_evidence.py -q -p no:cacheprovider
```

Expected: `2 passed`.

Commit:

```powershell
git add src/review_agent/evidence.py tests/test_evidence.py
git commit -m "feat: reconcile reviewer evidence"
```

## Task 2: Global Completion Checker

- [ ] **Step 1: Write failing completion tests**

Create `tests/test_completion.py`:

```python
from review_agent.completion import check_completion, completion_to_dict
from review_agent.evidence import EvidenceReconciliation
from review_agent.models import IntentPacket, IntentSource, IntentStatus, ReviewerResult, ReviewerResultStatus
from review_agent.orchestrator import ReviewerExecution
from review_agent.provider import ModelResponse
from tests.test_orchestrator import make_assignment


def execution(index, role, status):
    return ReviewerExecution(
        reviewer_index=index,
        trace_id=f"review-1-reviewer-{index}",
        assignment=make_assignment(role),
        envelope=None,
        response=ModelResponse(content="{}", provider_name="fake", model="fake"),
        result=ReviewerResult(status=status, investigation_summary=f"{role} {status.value}"),
    )


def intent(status=IntentStatus.SUFFICIENT):
    return IntentPacket(goal="Review change", sources={"goal": IntentSource.EXPLICIT}, status=status)


def reconciliation(canonical=0, rejected=0):
    return EvidenceReconciliation(
        canonical_findings=[object()] * canonical,
        rejected_findings=[object()] * rejected,
        remaining_disagreements=[],
        contract_coverage=[],
        evidence_quality="verified" if rejected == 0 else "unsupported_claims",
    )


def test_completion_blocks_when_core_reviewer_failed():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[
            execution(0, "Core Reviewer", ReviewerResultStatus.FAILED),
            execution(1, "Adversarial Reviewer", ReviewerResultStatus.COMPLETED),
        ],
        reconciliation=reconciliation(),
    )

    assert result.status == "blocked"
    assert result.recommendation == "manual_review"
    assert "Core Reviewer failed" in result.blockers


def test_completion_with_uncertainties_when_specialist_failed():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[
            execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED),
            execution(1, "Adversarial Reviewer", ReviewerResultStatus.FAILED),
        ],
        reconciliation=reconciliation(),
    )

    assert result.status == "completed_with_uncertainties"
    assert result.recommendation == "manual_review"
    assert result.missing_perspectives == ["Adversarial Reviewer"]


def test_completion_requires_manual_review_for_unsupported_findings():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)],
        reconciliation=reconciliation(rejected=1),
    )

    payload = completion_to_dict(result)

    assert payload["status"] == "completed_with_uncertainties"
    assert payload["recommendation"] == "manual_review"
    assert "unsupported findings rejected" in payload["uncertainties"]
```

- [ ] **Step 2: Run failing completion tests**

Run:

```powershell
python -B -m pytest tests/test_completion.py -q -p no:cacheprovider
```

Expected: fail because `review_agent.completion` does not exist.

- [ ] **Step 3: Implement `src/review_agent/completion.py`**

Implement:

- `CompletionResult`
- `check_completion(intent, quality_results, executions, reconciliation)`
- `completion_to_dict(result)`

Rules:

- Intent `INSUFFICIENT` blocks completion.
- Intent `PARTIAL` and intent uncertainties force `completed_with_uncertainties` plus `manual_review`.
- Core Reviewer failed => `blocked`, `manual_review`.
- Non-core failed reviewer => `completed_with_uncertainties`, `manual_review`, missing perspective recorded.
- Missing Core Reviewer contract coverage => `blocked`, `manual_review`.
- Missing non-core contract coverage => `completed_with_uncertainties`, `manual_review`.
- Blocked non-core reviewer is recorded as a missing perspective.
- Rejected findings => `completed_with_uncertainties`, `manual_review`.
- Supported blocking/high finding => `completed`, `needs_work`.
- Clean supported completion => `completed`, `approve`.

- [ ] **Step 4: Run completion tests and commit**

Run:

```powershell
python -B -m pytest tests/test_completion.py -q -p no:cacheprovider
```

Expected: `3 passed`.

Commit:

```powershell
git add src/review_agent/completion.py tests/test_completion.py
git commit -m "feat: check global review completion"
```

## Task 3: CLI And Report Integration

- [ ] **Step 1: Write failing CLI/report tests**

Modify `tests/test_cli_smoke.py` multi-reviewer smoke to assert:

```python
reconciliation = json.loads((run_dir / "reconciliation.json").read_text(encoding="utf-8"))
completion = json.loads((run_dir / "completion.json").read_text(encoding="utf-8"))
assert "canonical_findings" in reconciliation
assert completion["status"] in {"completed", "completed_with_uncertainties", "blocked"}
assert "## Evidence Reconciliation" in report
assert "## Completion Status" in report
```

Add to `tests/test_checkpoint_reporting.py`:

```python
def test_markdown_report_includes_reconciliation_and_completion_sections():
    ...
```

The report test should call `render_markdown_report(..., reconciliation_summary={...}, completion_summary={...})`.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```powershell
python -B -m pytest tests/test_cli_smoke.py::test_cli_multi_reviewer_mode_writes_per_reviewer_artifacts tests/test_checkpoint_reporting.py::test_markdown_report_includes_reconciliation_and_completion_sections -q -p no:cacheprovider
```

Expected: fail because CLI/report integration does not exist.

- [ ] **Step 3: Integrate CLI**

In `src/review_agent/cli.py`, after multi reviewer execution:

- call `reconcile_evidence(...)` with `observation_store.summaries_by_id().keys()`
- write `reconciliation.json`
- call `check_completion(...)`
- write `completion.json`
- pass compact summaries into `render_markdown_report(...)`

- [ ] **Step 4: Integrate report sections**

In `src/review_agent/reporting.py`, add optional:

- `reconciliation_summary`
- `completion_summary`

Render:

- `## Evidence Reconciliation`
- `## Completion Status`

- [ ] **Step 5: Verify and commit**

Run:

```powershell
python -B -m pytest tests/test_cli_smoke.py tests/test_checkpoint_reporting.py tests/test_evidence.py tests/test_completion.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

Commit:

```powershell
git add src/review_agent/cli.py src/review_agent/reporting.py tests/test_cli_smoke.py tests/test_checkpoint_reporting.py
git commit -m "feat: report reconciliation completion"
```

## Task 4: Final Verification

- [ ] **Step 1: Scan plan for unfinished markers**

Run:

```powershell
$patterns = @('TB'+'D','TO'+'DO','PLACE'+'HOLDER','x'+'xx','implement '+'later','fill in '+'details','Add '+'appropriate','Write tests '+'for the above','Similar '+'to Task') -join '|'
rg -n $patterns docs/superpowers/plans/2026-07-01-evidence-reconciler-completion-checker.md
```

Expected: no matches.

- [ ] **Step 2: Run full suite**

Run:

```powershell
python -B -m pytest -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 3: Commit plan**

Run:

```powershell
git add docs/superpowers/plans/2026-07-01-evidence-reconciler-completion-checker.md
git commit -m "docs: plan evidence reconciler completion checker"
```

## Self-review checklist

- Unsupported findings cannot become canonical.
- Completion status cannot be `completed` when Core Reviewer failed.
- Specialist reviewer failure remains visible as missing perspective.
- CLI writes both deterministic artifacts in multi mode.
- Report separates evidence reconciliation from completion status.

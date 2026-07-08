# Final Risk Reassessment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic final risk reassessment pass after reviewer/evidence/completion data exists, and make completion/brief/CLI consume it.

**Architecture:** Add a focused `final_risk.py` module that produces `FinalRiskAssessment` from initial risk plus post-review signals. Harden `check_completion(...)` with optional final-risk enforcement. Thread `final_risk.json` through CLI, run state, `review_brief.json`, and Markdown.

**Tech Stack:** Python dataclasses, existing risk/completion/brief modules, pytest, local CLI artifacts.

---

## Scope

Implement spec sections 9.4 and 14 for local M1 runs:

- Final risk reassessment runs after available reviewer/evidence/completion signals.
- `final_risk.json` is written for successful local review runs.
- Completion can require final risk to be present.
- High/critical final risk makes the final recommendation conservative.
- Review Brief no longer reports `final.status = not_reassessed` when CLI has run final risk.

This plan does not add LLM-based reassessment, evals, GitHub review comments, durable memory, or PR ingestion.

## File structure

- Create `src/review_agent/final_risk.py`
  - Owns `FinalRiskAssessment`, `reassess_final_risk(...)`, `final_risk_to_dict(...)`.
  - Encodes deterministic post-review escalation rules.
- Create `tests/test_final_risk.py`
  - Unit coverage for verified findings, rejected findings, quality gates, and completion blockers.
- Modify `src/review_agent/completion.py`
  - Add optional `final_risk_level` and `require_final_risk` inputs to `check_completion(...)`.
- Modify `tests/test_completion.py`
  - Cover required final risk and high final risk recommendation.
- Modify `src/review_agent/brief.py`
  - Accept `final_risk_assessment` and render it into the final risk slot.
- Modify `tests/test_brief.py`
  - Cover reassessed final risk in JSON and Markdown.
- Modify `src/review_agent/cli.py`
  - Write `final_risk.json`, update `completion.json` after final risk for multi-reviewer runs, include state artifact.
- Modify `src/review_agent/run_state.py`
  - Add `RunPhase.FINAL_RISK`.
- Modify `tests/test_cli_smoke.py` and `tests/test_cli_resume.py`
  - Cover `final_risk.json` artifact and resume presence.

---

## Task 1: Add deterministic final risk reassessment

**Files:**
- Create: `src/review_agent/final_risk.py`
- Create: `tests/test_final_risk.py`

- [ ] **Step 1: Write failing final risk tests**

Create `tests/test_final_risk.py`:

```python
from review_agent.final_risk import reassess_final_risk, final_risk_to_dict
from review_agent.models import IntentPacket, IntentStatus, QualityGateResult, ReviewerFinding, ReviewerResult, RiskAssessment, RiskLevel


def initial(level=RiskLevel.LOW) -> RiskAssessment:
    return RiskAssessment(
        level=level,
        dimensions={"impact": "local"},
        reasons=["initial local change"],
        signal_refs=[],
        uncertainties=[],
        suggested_focus=["intent alignment"],
    )


def intent() -> IntentPacket:
    return IntentPacket(goal="Review change", status=IntentStatus.SUFFICIENT)


def test_final_risk_escalates_for_verified_high_finding() -> None:
    result = reassess_final_risk(
        initial_risk=initial(RiskLevel.LOW),
        intent_packet=intent(),
        quality_results=[],
        reviewer_result=None,
        reconciliation_payload={
            "canonical_findings": [
                {
                    "claim": "Authorization bypass remains possible",
                    "severity": "high",
                    "confidence": "medium",
                    "evidence_refs": ["O-1"],
                }
            ],
            "rejected_findings": [],
            "remaining_disagreements": [],
        },
        completion_summary={"status": "completed", "recommendation": "approve"},
    )

    payload = final_risk_to_dict(result)

    assert payload["status"] == "reassessed"
    assert payload["initial_level"] == "low"
    assert payload["level"] == "high"
    assert "verified high finding: Authorization bypass remains possible" in payload["reasons"]


def test_final_risk_escalates_for_failed_quality_gate() -> None:
    result = reassess_final_risk(
        initial_risk=initial(RiskLevel.LOW),
        intent_packet=intent(),
        quality_results=[
            QualityGateResult(
                name="python_compile",
                status="failed",
                command=["python", "-m", "compileall"],
                summary="SyntaxError",
            )
        ],
        reviewer_result=None,
        reconciliation_payload={},
        completion_summary={},
    )

    assert result.level is RiskLevel.HIGH
    assert "quality gate failed after review: python_compile" in result.reasons


def test_final_risk_does_not_escalate_for_rejected_findings_only() -> None:
    result = reassess_final_risk(
        initial_risk=initial(RiskLevel.LOW),
        intent_packet=intent(),
        quality_results=[],
        reviewer_result=None,
        reconciliation_payload={
            "canonical_findings": [],
            "rejected_findings": [{"claim": "Unsupported critical issue", "reason": "unsupported_claim"}],
            "remaining_disagreements": [],
        },
        completion_summary={"status": "completed", "recommendation": "approve"},
    )

    assert result.level is RiskLevel.LOW
    assert "rejected unsupported findings were not used for escalation" in result.reasons


def test_final_risk_uses_single_reviewer_findings_when_no_reconciliation_exists() -> None:
    result = reassess_final_risk(
        initial_risk=initial(RiskLevel.LOW),
        intent_packet=intent(),
        quality_results=[],
        reviewer_result=ReviewerResult(
            confirmed_findings=[
                ReviewerFinding(
                    claim="Missing rollback path",
                    severity="medium",
                    confidence="medium",
                    evidence_refs=[],
                )
            ]
        ),
        reconciliation_payload=None,
        completion_summary=None,
    )

    assert result.level is RiskLevel.MEDIUM
    assert "single reviewer medium finding: Missing rollback path" in result.reasons
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/test_final_risk.py -q -p no:cacheprovider
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'review_agent.final_risk'`.

- [ ] **Step 3: Implement `src/review_agent/final_risk.py`**

Create `src/review_agent/final_risk.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from review_agent.models import IntentPacket, IntentStatus, QualityGateResult, ReviewerResult, RiskAssessment, RiskLevel


RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


@dataclass(frozen=True)
class FinalRiskAssessment:
    status: str
    initial_level: RiskLevel
    level: RiskLevel
    reasons: list[str]
    escalations: list[str]
    deescalations: list[str]
    uncertainties: list[str]
    signal_refs: list[str]


def reassess_final_risk(
    *,
    initial_risk: RiskAssessment,
    intent_packet: IntentPacket,
    quality_results: list[QualityGateResult],
    reviewer_result: ReviewerResult | None,
    reconciliation_payload: dict[str, Any] | None,
    completion_summary: dict[str, Any] | None,
) -> FinalRiskAssessment:
    level = initial_risk.level
    reasons = list(initial_risk.reasons)
    escalations: list[str] = []
    deescalations: list[str] = []
    uncertainties = list(initial_risk.uncertainties)
    signal_refs = list(initial_risk.signal_refs)
    reconciliation = reconciliation_payload or {}
    completion = completion_summary or {}

    if intent_packet.status is IntentStatus.INSUFFICIENT:
        level = _raise_to(level, RiskLevel.HIGH)
        escalations.append("intent insufficient at final reassessment")
        reasons.append("intent insufficient at final reassessment")
    elif intent_packet.status is IntentStatus.PARTIAL:
        level = _raise_to(level, RiskLevel.MEDIUM)
        uncertainties.append("Intent Packet partial at final reassessment")

    for result in quality_results:
        if result.status == "failed":
            level = _raise_to(level, RiskLevel.HIGH)
            message = f"quality gate failed after review: {result.name}"
            escalations.append(message)
            reasons.append(message)
            signal_refs.append(f"quality_gate:{result.name}")

    for item in reconciliation.get("canonical_findings", []):
        row = dict(item)
        claim = str(row.get("claim", ""))
        severity = str(row.get("severity", "")).casefold()
        target = _risk_for_finding_severity(severity)
        if target is not None:
            level = _raise_to(level, target)
            label = "critical" if target is RiskLevel.CRITICAL else target.value
            message = f"verified {label} finding: {claim}"
            escalations.append(message)
            reasons.append(message)

    if not reconciliation.get("canonical_findings") and reviewer_result is not None:
        for finding in reviewer_result.confirmed_findings:
            target = _risk_for_finding_severity(finding.severity.casefold())
            if target is not None:
                level = _raise_to(level, target)
                message = f"single reviewer {target.value} finding: {finding.claim}"
                escalations.append(message)
                reasons.append(message)

    if reconciliation.get("rejected_findings") and not reconciliation.get("canonical_findings"):
        reasons.append("rejected unsupported findings were not used for escalation")

    if reconciliation.get("remaining_disagreements"):
        level = _raise_to(level, RiskLevel.MEDIUM)
        uncertainties.append("reviewer disagreements remain unresolved")

    blockers = [str(item) for item in completion.get("blockers", [])]
    if blockers:
        level = _raise_to(level, RiskLevel.HIGH)
        uncertainties.extend(blockers)
        reasons.append("completion blockers remain at final reassessment")

    if completion.get("status") == "completed_with_uncertainties":
        level = _raise_to(level, RiskLevel.MEDIUM)

    return FinalRiskAssessment(
        status="reassessed",
        initial_level=initial_risk.level,
        level=level,
        reasons=_dedupe(reasons),
        escalations=_dedupe(escalations),
        deescalations=deescalations,
        uncertainties=_dedupe(uncertainties),
        signal_refs=_dedupe(signal_refs),
    )


def final_risk_to_dict(result: FinalRiskAssessment) -> dict[str, Any]:
    payload = asdict(result)
    payload["initial_level"] = result.initial_level.value
    payload["level"] = result.level.value
    return payload


def _risk_for_finding_severity(severity: str) -> RiskLevel | None:
    if severity in {"critical", "blocker"}:
        return RiskLevel.CRITICAL
    if severity == "high":
        return RiskLevel.HIGH
    if severity == "medium":
        return RiskLevel.MEDIUM
    return None


def _raise_to(current: RiskLevel, target: RiskLevel) -> RiskLevel:
    if RISK_ORDER[target] > RISK_ORDER[current]:
        return target
    return current


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_final_risk.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src/review_agent/final_risk.py tests/test_final_risk.py
git commit -m "feat: reassess final review risk"
```

---

## Task 2: Harden completion with final risk awareness

**Files:**
- Modify: `src/review_agent/completion.py`
- Modify: `tests/test_completion.py`

- [ ] **Step 1: Write failing completion tests**

Append to `tests/test_completion.py`:

```python
def test_completion_blocks_when_final_risk_is_required_but_missing():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)],
        reconciliation=reconciliation_with_coverage(coverage(0, "Core Reviewer")),
        require_final_risk=True,
    )

    assert result.status == "blocked"
    assert result.recommendation == "manual_review"
    assert "Final risk reassessment not completed" in result.blockers


def test_completion_requires_manual_review_when_final_risk_is_high():
    result = check_completion(
        intent=intent(),
        quality_results=[],
        executions=[execution(0, "Core Reviewer", ReviewerResultStatus.COMPLETED)],
        reconciliation=reconciliation_with_coverage(coverage(0, "Core Reviewer")),
        require_final_risk=True,
        final_risk_level="high",
    )

    assert result.status == "completed_with_uncertainties"
    assert result.recommendation == "manual_review"
    assert "Final risk is high" in result.uncertainties
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/test_completion.py::test_completion_blocks_when_final_risk_is_required_but_missing tests/test_completion.py::test_completion_requires_manual_review_when_final_risk_is_high -q -p no:cacheprovider
```

Expected: FAIL with `TypeError: check_completion() got an unexpected keyword argument 'require_final_risk'`.

- [ ] **Step 3: Update `check_completion(...)`**

In `src/review_agent/completion.py`, change the signature:

```python
def check_completion(
    intent: IntentPacket,
    quality_results: list[QualityGateResult],
    executions: list[ReviewerExecution],
    reconciliation: EvidenceReconciliation,
    *,
    require_final_risk: bool = False,
    final_risk_level: str | None = None,
) -> CompletionResult:
```

Before the blockers return:

```python
    if require_final_risk and final_risk_level is None:
        blockers.append("Final risk reassessment not completed")

    if final_risk_level in {"high", "critical"}:
        uncertainties.append(f"Final risk is {final_risk_level}")
```

Leave `_recommendation(...)` behavior unchanged because uncertainties already force `manual_review`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_completion.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src/review_agent/completion.py tests/test_completion.py
git commit -m "feat: require final risk for completion"
```

---

## Task 3: Thread final risk through Brief and CLI artifacts

**Files:**
- Modify: `src/review_agent/brief.py`
- Modify: `src/review_agent/cli.py`
- Modify: `src/review_agent/run_state.py`
- Modify: `tests/test_brief.py`
- Modify: `tests/test_cli_smoke.py`
- Modify: `tests/test_cli_resume.py`

- [ ] **Step 1: Write failing Brief and CLI tests**

In `tests/test_brief.py`, change the final risk expectation in the first test:

```python
    final_risk = {
        "status": "reassessed",
        "initial_level": "high",
        "level": "critical",
        "reasons": ["verified critical finding: data loss"],
        "escalations": ["verified critical finding: data loss"],
        "deescalations": [],
        "uncertainties": [],
        "signal_refs": ["finding:data-loss"],
    }
```

Pass it to `build_review_brief(..., final_risk_assessment=final_risk)` and assert:

```python
assert payload["initial_and_final_risk_assessment"]["final"]["status"] == "reassessed"
assert payload["initial_and_final_risk_assessment"]["final"]["level"] == "critical"
```

In `tests/test_cli_smoke.py::test_cli_review_writes_state_and_preflight_summary`, add:

```python
final_risk = json.loads((run_dirs[-1] / "final_risk.json").read_text(encoding="utf-8"))
assert "Final risk:" in output
assert state["artifacts"]["final_risk"] == "final_risk.json"
assert final_risk["status"] == "reassessed"
assert brief["initial_and_final_risk_assessment"]["final"]["status"] == "reassessed"
```

In `tests/test_cli_resume.py::test_cli_resume_prints_completed_run_summary`, add:

```python
assert "final_risk.json (present)" in output
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/test_brief.py::test_review_brief_to_dict_contains_spec_sections_and_recommendation tests/test_cli_smoke.py::test_cli_review_writes_state_and_preflight_summary tests/test_cli_resume.py::test_cli_resume_prints_completed_run_summary -q -p no:cacheprovider
```

Expected: FAIL because `build_review_brief` does not accept `final_risk_assessment` and CLI does not write `final_risk.json`.

- [ ] **Step 3: Update Brief**

In `src/review_agent/brief.py`, add a parameter:

```python
    final_risk_assessment: dict[str, Any] | None = None,
```

Replace the current final risk block:

```python
            "final": final_risk_assessment
            or {
                "status": "not_reassessed",
                "level": risk_assessment.level.value,
                "reasons": ["Final risk reassessment has not run in the local M1 path."],
            },
```

- [ ] **Step 4: Update run state phase**

In `src/review_agent/run_state.py`, add:

```python
    FINAL_RISK = "final_risk"
```

- [ ] **Step 5: Update CLI final risk flow**

In `src/review_agent/cli.py`, import:

```python
from review_agent.final_risk import final_risk_to_dict, reassess_final_risk
```

After reviewer/completion work and before `build_review_brief(...)`, add:

```python
    final_risk = reassess_final_risk(
        initial_risk=risk_assessment,
        intent_packet=intent,
        quality_results=quality_results,
        reviewer_result=reviewer_result,
        reconciliation_payload=reconciliation_payload,
        completion_summary=completion_payload,
    )
    final_risk_payload = final_risk_to_dict(final_risk)
    store.write_json("final_risk.json", final_risk_payload)
    state = advance_run_state(
        state,
        phase=RunPhase.FINAL_RISK,
        message="Final risk reassessment completed",
        artifacts={"final_risk": "final_risk.json"},
    )
    store.write_state(state)
```

If multi-reviewer completion ran, recompute completion with final risk:

```python
            completion = check_completion(
                intent=intent,
                quality_results=quality_results,
                executions=multi_run.executions,
                reconciliation=reconciliation,
                require_final_risk=True,
                final_risk_level=final_risk.level.value,
            )
            completion_payload = completion_to_dict(completion)
            store.write_json("completion.json", completion_payload)
            completion_summary = completion_payload
```

Pass final risk into brief:

```python
        final_risk_assessment=final_risk_payload,
```

Update final completion state artifacts:

```python
        artifacts={"report": "report.md", "review_brief": "review_brief.json", "final_risk": "final_risk.json"},
```

Print final risk:

```python
    print(f"Final risk: {final_risk.level.value}")
```

- [ ] **Step 6: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_brief.py tests/test_cli_smoke.py::test_cli_review_writes_state_and_preflight_summary tests/test_cli_resume.py::test_cli_resume_prints_completed_run_summary -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

Run:

```powershell
git add src/review_agent/brief.py src/review_agent/cli.py src/review_agent/run_state.py tests/test_brief.py tests/test_cli_smoke.py tests/test_cli_resume.py
git commit -m "feat: write final risk artifact"
```

---

## Task 4: Final verification

**Files:**
- Modify tests only if verification reveals a real mismatch.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/test_final_risk.py tests/test_completion.py tests/test_brief.py tests/test_cli_smoke.py tests/test_cli_resume.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 2: Run full suite**

Run:

```powershell
python -m pytest -q -p no:cacheprovider
```

Expected: PASS. If Windows emits the known pytest temporary-directory cleanup warning after the pass summary, record it and do not change feature code for that warning.

- [ ] **Step 3: Manual review/resume smoke**

Run:

```powershell
$env:PYTHONPATH='src'; python -c "from review_agent.cli import main; raise SystemExit(main(['review','--repo','.','--base','HEAD~1','--head','HEAD','--non-interactive']))"
```

Expected output includes:

```text
Preflight
Final risk:
Review brief:
Review brief JSON:
```

Then run resume for the generated id:

```powershell
$env:PYTHONPATH='src'; python -c "from review_agent.cli import main; raise SystemExit(main(['resume','<generated-review-id>','--repo','.']))"
```

Expected output includes:

```text
final_risk.json (present)
review_brief.json (present)
```

- [ ] **Step 4: Clean generated artifacts**

Run:

```powershell
$root = (Resolve-Path -LiteralPath 'D:\Agent\code review agent').Path; $paths = @('D:\Agent\code review agent\.review-agent', 'D:\Agent\code review agent\src\review_agent\__pycache__', 'D:\Agent\code review agent\tests\__pycache__'); foreach ($path in $paths) { $resolved = Resolve-Path -LiteralPath $path -ErrorAction SilentlyContinue; if ($resolved -and $resolved.Path.StartsWith($root)) { Remove-Item -LiteralPath $resolved.Path -Recurse -Force } }
```

---

## Completion checklist

- `final_risk.json` exists for successful local review runs.
- `state.json` includes `"final_risk": "final_risk.json"` after completion.
- `review_brief.json` final risk status is `reassessed` when CLI runs.
- Completion can block when final risk is required but missing.
- High/critical final risk keeps recommendation conservative.
- Rejected unsupported findings do not directly escalate final risk.
- Full tests pass.


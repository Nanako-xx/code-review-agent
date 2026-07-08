# Final Review Brief + JSON Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a spec-aligned final Review Brief in both human-readable Markdown and stable machine-readable JSON.

**Architecture:** Add a focused `brief.py` module that assembles a `ReviewBrief` from existing review state, reviewer output, evidence reconciliation, completion status, and quality gates. Keep Markdown rendering in `reporting.py`, and keep CLI responsibility limited to collecting existing objects, writing `review_brief.json`, writing `report.md`, updating run state artifacts, and printing paths/recommendation.

**Tech Stack:** Python dataclasses, existing `CheckpointStore`, pytest, current local CLI review flow.

---

## Scope

Implement spec section 19 for local runs:

1. `report.md` should use the section structure from the main design spec.
2. `.review-agent/runs/<review_id>/review_brief.json` should contain the same review conclusion in stable JSON.
3. CLI completion output should make the useful outputs obvious: report path, JSON brief path, recommendation, and remaining uncertainty count.

This plan does not add evals, GitHub review comments, PR ingestion, final risk reassessment logic, or durable memory.

## File structure

- Create `src/review_agent/brief.py`
  - Owns `ReviewBrief` and small row dataclasses.
  - Owns `build_review_brief(...)` and `review_brief_to_dict(...)`.
  - Converts existing dataclasses and dict payloads into stable, spec-shaped data.
- Modify `src/review_agent/reporting.py`
  - Add `render_review_brief_markdown(brief: ReviewBrief) -> str`.
  - Keep `render_markdown_report(...)` as a backwards-compatible wrapper for existing tests and callers.
- Modify `src/review_agent/cli.py`
  - Build a `ReviewBrief`.
  - Write `review_brief.json`.
  - Render `report.md` from the same `ReviewBrief`.
  - Add `review_brief` to `state.json` artifacts.
  - Print output paths and recommendation.
- Modify `tests/test_checkpoint_reporting.py`
  - Update Markdown assertions to the new section names while preserving backwards compatibility coverage.
- Create `tests/test_brief.py`
  - Unit tests for JSON shape and derived sections.
- Modify `tests/test_cli_smoke.py`
  - Integration test for `review_brief.json`, state artifact, and CLI final output.
- Modify `tests/test_cli_resume.py`
  - Confirm `resume` lists `review_brief.json` as a present artifact after a completed run.

---

## Task 1: Add structured ReviewBrief builder and JSON serialization

**Files:**
- Create: `src/review_agent/brief.py`
- Create: `tests/test_brief.py`

- [ ] **Step 1: Write failing tests for structured JSON**

Create `tests/test_brief.py`:

```python
from review_agent.brief import build_review_brief, review_brief_to_dict
from review_agent.models import (
    IntentPacket,
    IntentSource,
    IntentStatus,
    QualityGateResult,
    RiskAssessment,
    RiskLevel,
)


def test_review_brief_to_dict_contains_spec_sections_and_recommendation() -> None:
    intent = IntentPacket(
        goal="Add auth token check",
        acceptance_criteria=["reject bad token"],
        scope=["auth.py"],
        constraints=["read-only review"],
        sources={"goal": IntentSource.EXPLICIT},
        status=IntentStatus.PARTIAL,
        uncertainties=["acceptance criteria inferred from code"],
    )
    risk = RiskAssessment(
        level=RiskLevel.HIGH,
        dimensions={"impact": "auth path"},
        reasons=["auth.py changed"],
        signal_refs=["changed_file:auth.py"],
        uncertainties=["missing integration tests"],
        suggested_focus=["regression safety"],
    )
    reconciliation_payload = {
        "canonical_findings": [
            {
                "claim": "Bad token path is not covered",
                "severity": "high",
                "confidence": "medium",
                "evidence_refs": ["O-1"],
                "reviewer_indices": [0],
                "roles": ["core"],
                "suggested_action": "Add a negative-token test",
            }
        ],
        "rejected_findings": [
            {
                "reviewer_index": 1,
                "role": "adversarial",
                "claim": "Session storage changed",
                "reason": "unsupported_claim",
                "evidence_refs": [],
                "missing_evidence_refs": [],
            }
        ],
        "remaining_disagreements": ["core and adversarial disagree on token expiry"],
        "contract_coverage": [
            {
                "reviewer_index": 0,
                "role": "core",
                "contract": "Behavioral Correctness",
                "status": "covered",
                "summary": "Token happy path reviewed",
                "evidence_refs": ["O-1"],
                "unsupported_evidence_refs": [],
            }
        ],
        "evidence_quality": "verified",
    }
    completion_summary = {
        "status": "completed_with_uncertainties",
        "recommendation": "manual_review",
        "blockers": [],
        "uncertainties": ["Intent Packet partial"],
        "missing_perspectives": [],
    }

    brief = build_review_brief(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        intent_packet=intent,
        risk_assessment=risk,
        changed_files=["auth.py"],
        quality_results=[
            QualityGateResult(
                name="python_compile",
                status="passed",
                command=["python", "-m", "compileall"],
                summary="Compiled 1 Python file",
            )
        ],
        observation_summaries={"O-1": "auth.py changed between base and head"},
        repository_intelligence_summary="Repository Intelligence\n- modified function check auth.py:1-2",
        reconciliation_payload=reconciliation_payload,
        completion_summary=completion_summary,
    )

    payload = review_brief_to_dict(brief)

    assert payload["review_id"] == "review-1"
    assert payload["change_intent"]["goal"] == "Add auth token check"
    assert payload["intent_assessment"]["status"] == "partial"
    assert payload["initial_and_final_risk_assessment"]["initial"]["level"] == "high"
    assert payload["initial_and_final_risk_assessment"]["final"]["status"] == "not_reassessed"
    assert payload["quality_gates"][0]["name"] == "python_compile"
    assert payload["change_map_and_repository_impact"]["changed_files"] == ["auth.py"]
    assert payload["verified_findings"][0]["claim"] == "Bad token path is not covered"
    assert payload["rejected_hypotheses"][0]["claim"] == "Session storage changed"
    assert payload["reviewer_disagreements"] == ["core and adversarial disagree on token expiry"]
    assert payload["review_contract_coverage"][0]["contract"] == "Behavioral Correctness"
    assert payload["non_binding_recommendation"] == "manual_review"
    assert "auth.py" in payload["human_review_checklist_and_reading_order"][0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m pytest tests/test_brief.py -q -p no:cacheprovider
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'review_agent.brief'`.

- [ ] **Step 3: Implement `src/review_agent/brief.py`**

Create `src/review_agent/brief.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any

from review_agent.models import (
    IntentPacket,
    QualityGateResult,
    ReviewerResult,
    RiskAssessment,
)


@dataclass(frozen=True)
class BriefFinding:
    claim: str
    severity: str
    confidence: str
    evidence_refs: list[str]
    reviewer_indices: list[int] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    suggested_action: str | None = None


@dataclass(frozen=True)
class RejectedHypothesis:
    claim: str
    reason: str
    evidence_refs: list[str] = field(default_factory=list)
    reviewer_index: int | None = None
    role: str | None = None


@dataclass(frozen=True)
class ReviewBrief:
    review_id: str
    base_revision: str
    head_revision: str
    change_intent: dict[str, Any]
    intent_assessment: dict[str, Any]
    initial_and_final_risk_assessment: dict[str, Any]
    quality_gates: list[dict[str, Any]]
    change_map_and_repository_impact: dict[str, Any]
    verified_findings: list[BriefFinding]
    rejected_hypotheses: list[RejectedHypothesis]
    uncertainties: list[str]
    reviewer_disagreements: list[str]
    review_contract_coverage: list[dict[str, Any]]
    verification_evidence: list[dict[str, Any]]
    human_review_checklist_and_reading_order: list[str]
    non_binding_recommendation: str


def build_review_brief(
    *,
    review_id: str,
    base_revision: str,
    head_revision: str,
    intent_packet: IntentPacket,
    risk_assessment: RiskAssessment,
    changed_files: list[str],
    quality_results: list[QualityGateResult],
    observation_summaries: dict[str, str] | None = None,
    repository_intelligence_summary: str | None = None,
    reviewer_result: ReviewerResult | None = None,
    multi_reviewer_summary: dict[str, object] | None = None,
    reconciliation_payload: dict[str, Any] | None = None,
    completion_summary: dict[str, Any] | None = None,
) -> ReviewBrief:
    observations = observation_summaries or {}
    reconciliation = reconciliation_payload or {}
    completion = completion_summary or {}
    verified_findings = _verified_findings(reconciliation)
    rejected_hypotheses = _rejected_hypotheses(reconciliation, reviewer_result)
    uncertainties = _uncertainties(intent_packet, risk_assessment, reviewer_result, completion)
    contract_coverage = _contract_coverage(reconciliation, reviewer_result)
    verification_evidence = _verification_evidence(quality_results, observations)

    return ReviewBrief(
        review_id=review_id,
        base_revision=base_revision,
        head_revision=head_revision,
        change_intent={
            "goal": intent_packet.goal,
            "acceptance_criteria": list(intent_packet.acceptance_criteria),
            "scope": list(intent_packet.scope),
            "constraints": list(intent_packet.constraints),
            "sources": {key: value.value for key, value in intent_packet.sources.items()},
        },
        intent_assessment={
            "status": intent_packet.status.value,
            "uncertainties": list(intent_packet.uncertainties),
            "source_counts": _source_counts(intent_packet),
        },
        initial_and_final_risk_assessment={
            "initial": _risk_to_dict(risk_assessment),
            "final": {
                "status": "not_reassessed",
                "level": risk_assessment.level.value,
                "reasons": ["Final risk reassessment has not run in the local M1 path."],
            },
        },
        quality_gates=[_quality_result_to_dict(result) for result in quality_results],
        change_map_and_repository_impact={
            "changed_files": list(changed_files),
            "repository_intelligence_summary": repository_intelligence_summary or "",
            "observation_count": len(observations),
            "reviewer_summary": dict(multi_reviewer_summary or {}),
        },
        verified_findings=verified_findings,
        rejected_hypotheses=rejected_hypotheses,
        uncertainties=uncertainties,
        reviewer_disagreements=[str(item) for item in reconciliation.get("remaining_disagreements", [])],
        review_contract_coverage=contract_coverage,
        verification_evidence=verification_evidence,
        human_review_checklist_and_reading_order=_human_review_checklist(
            changed_files=changed_files,
            risk_assessment=risk_assessment,
            verified_findings=verified_findings,
            uncertainties=uncertainties,
        ),
        non_binding_recommendation=str(completion.get("recommendation", "manual_review")),
    )


def review_brief_to_dict(brief: ReviewBrief) -> dict[str, Any]:
    return _json_ready(asdict(brief))


def _verified_findings(reconciliation: dict[str, Any]) -> list[BriefFinding]:
    findings: list[BriefFinding] = []
    for item in reconciliation.get("canonical_findings", []):
        row = dict(item)
        findings.append(
            BriefFinding(
                claim=str(row.get("claim", "")),
                severity=str(row.get("severity", "")),
                confidence=str(row.get("confidence", "")),
                evidence_refs=[str(ref) for ref in row.get("evidence_refs", [])],
                reviewer_indices=[int(index) for index in row.get("reviewer_indices", [])],
                roles=[str(role) for role in row.get("roles", [])],
                suggested_action=str(row["suggested_action"]) if row.get("suggested_action") is not None else None,
            )
        )
    return findings


def _rejected_hypotheses(
    reconciliation: dict[str, Any],
    reviewer_result: ReviewerResult | None,
) -> list[RejectedHypothesis]:
    rejected: list[RejectedHypothesis] = []
    for item in reconciliation.get("rejected_findings", []):
        row = dict(item)
        rejected.append(
            RejectedHypothesis(
                claim=str(row.get("claim", "")),
                reason=str(row.get("reason", "unsupported_claim")),
                evidence_refs=[str(ref) for ref in row.get("evidence_refs", [])],
                reviewer_index=int(row["reviewer_index"]) if row.get("reviewer_index") is not None else None,
                role=str(row["role"]) if row.get("role") is not None else None,
            )
        )
    if reviewer_result is not None:
        rejected.extend(
            RejectedHypothesis(claim=str(item), reason="reviewer_rejected_hypothesis")
            for item in reviewer_result.rejected_hypotheses
        )
    return rejected


def _uncertainties(
    intent_packet: IntentPacket,
    risk_assessment: RiskAssessment,
    reviewer_result: ReviewerResult | None,
    completion: dict[str, Any],
) -> list[str]:
    items: list[str] = []
    items.extend(intent_packet.uncertainties)
    items.extend(risk_assessment.uncertainties)
    if reviewer_result is not None:
        items.extend(reviewer_result.uncertainties)
    items.extend(str(item) for item in completion.get("uncertainties", []))
    items.extend(str(item) for item in completion.get("blockers", []))
    items.extend(f"Missing perspective: {item}" for item in completion.get("missing_perspectives", []))
    return _dedupe(items)


def _contract_coverage(
    reconciliation: dict[str, Any],
    reviewer_result: ReviewerResult | None,
) -> list[dict[str, Any]]:
    if reconciliation.get("contract_coverage"):
        return [dict(item) for item in reconciliation["contract_coverage"]]
    if reviewer_result is None:
        return []
    return [
        {
            "contract": assessment.contract,
            "status": assessment.status.value,
            "summary": assessment.summary,
            "evidence_refs": list(assessment.evidence_refs),
        }
        for assessment in reviewer_result.contract_assessments
    ]


def _verification_evidence(
    quality_results: list[QualityGateResult],
    observations: dict[str, str],
) -> list[dict[str, Any]]:
    evidence = [
        {
            "kind": "quality_gate",
            "name": result.name,
            "status": result.status,
            "summary": result.summary,
            "command": list(result.command),
            "observation_ref": result.observation_ref,
        }
        for result in quality_results
    ]
    evidence.extend(
        {
            "kind": "observation",
            "id": observation_id,
            "summary": summary,
        }
        for observation_id, summary in observations.items()
    )
    return evidence


def _human_review_checklist(
    *,
    changed_files: list[str],
    risk_assessment: RiskAssessment,
    verified_findings: list[BriefFinding],
    uncertainties: list[str],
) -> list[str]:
    checklist: list[str] = []
    checklist.extend(f"Read changed file: {path}" for path in changed_files)
    checklist.extend(f"Check review focus: {focus}" for focus in risk_assessment.suggested_focus)
    checklist.extend(f"Verify finding: {finding.claim}" for finding in verified_findings)
    checklist.extend(f"Resolve uncertainty: {uncertainty}" for uncertainty in uncertainties)
    return checklist or ["No prioritized human review items were generated."]


def _source_counts(intent_packet: IntentPacket) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in intent_packet.sources.values():
        counts[source.value] = counts.get(source.value, 0) + 1
    return counts


def _risk_to_dict(risk_assessment: RiskAssessment) -> dict[str, Any]:
    return {
        "level": risk_assessment.level.value,
        "dimensions": dict(risk_assessment.dimensions),
        "reasons": list(risk_assessment.reasons),
        "signal_refs": list(risk_assessment.signal_refs),
        "uncertainties": list(risk_assessment.uncertainties),
        "suggested_focus": list(risk_assessment.suggested_focus),
    }


def _quality_result_to_dict(result: QualityGateResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "status": result.status,
        "command": list(result.command),
        "summary": result.summary,
        "observation_ref": result.observation_ref,
    }


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
```

- [ ] **Step 4: Run the tests to verify Task 1 passes**

Run:

```powershell
python -m pytest tests/test_brief.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src/review_agent/brief.py tests/test_brief.py
git commit -m "feat: build structured review brief"
```

---

## Task 2: Render spec-aligned Markdown from ReviewBrief

**Files:**
- Modify: `src/review_agent/reporting.py`
- Modify: `tests/test_checkpoint_reporting.py`
- Modify: `tests/test_brief.py`

- [ ] **Step 1: Write failing Markdown tests**

Append to `tests/test_brief.py`:

```python
from review_agent.reporting import render_review_brief_markdown


def test_render_review_brief_markdown_uses_spec_section_order() -> None:
    intent = IntentPacket(goal="Add auth token check", status=IntentStatus.SUFFICIENT)
    risk = RiskAssessment(
        level=RiskLevel.MEDIUM,
        dimensions={"impact": "auth path"},
        reasons=["auth.py changed"],
        signal_refs=["changed_file:auth.py"],
        uncertainties=[],
        suggested_focus=["test adequacy"],
    )
    brief = build_review_brief(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        intent_packet=intent,
        risk_assessment=risk,
        changed_files=["auth.py"],
        quality_results=[],
        completion_summary={"recommendation": "manual_review"},
    )

    markdown = render_review_brief_markdown(brief)

    expected_sections = [
        "## Change Intent",
        "## Intent Assessment",
        "## Initial And Final Risk Assessment",
        "## Quality Gates",
        "## Change Map And Repository Impact",
        "## Verified Findings",
        "## Rejected Hypotheses",
        "## Uncertainties",
        "## Reviewer Disagreements",
        "## Review Contract Coverage",
        "## Verification Evidence",
        "## Human Review Checklist And Reading Order",
        "## Non-Binding Recommendation",
    ]
    positions = [markdown.index(section) for section in expected_sections]
    assert positions == sorted(positions)
    assert "Risk level: medium" in markdown
    assert "Manual review required before merge." in markdown
```

Update `tests/test_checkpoint_reporting.py` so the existing report tests assert the new section names:

```python
assert "## Initial And Final Risk Assessment" in report
assert "- changed_file:auth.py" in report
assert "## Uncertainties" in report
```

Keep existing assertions for content that should still appear, such as reviewer status and repository intelligence summary.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_brief.py tests/test_checkpoint_reporting.py -q -p no:cacheprovider
```

Expected: FAIL because `render_review_brief_markdown` does not exist and `render_markdown_report` still emits the older section layout.

- [ ] **Step 3: Implement Markdown rendering**

Modify `src/review_agent/reporting.py` to import the new brief builder and expose both the new renderer and the existing wrapper.

The top-level API should become:

```python
from review_agent.brief import ReviewBrief, build_review_brief
from review_agent.models import IntentPacket, QualityGateResult, ReviewerResult, RiskAssessment


def render_review_brief_markdown(brief: ReviewBrief) -> str:
    return "\n".join(
        [
            "# Review Brief",
            "",
            f"Review ID: {brief.review_id}",
            f"Base: {brief.base_revision}",
            f"Head: {brief.head_revision}",
            "",
            "## Change Intent",
            "",
            _change_intent_section(brief),
            "",
            "## Intent Assessment",
            "",
            _intent_assessment_section(brief),
            "",
            "## Initial And Final Risk Assessment",
            "",
            _risk_section(brief),
            "",
            "## Quality Gates",
            "",
            _quality_gates_section(brief),
            "",
            "## Change Map And Repository Impact",
            "",
            _change_map_section(brief),
            "",
            "## Verified Findings",
            "",
            _verified_findings_section(brief),
            "",
            "## Rejected Hypotheses",
            "",
            _rejected_hypotheses_section(brief),
            "",
            "## Uncertainties",
            "",
            _string_list(brief.uncertainties, "No unresolved uncertainties recorded"),
            "",
            "## Reviewer Disagreements",
            "",
            _string_list(brief.reviewer_disagreements, "No reviewer disagreements recorded"),
            "",
            "## Review Contract Coverage",
            "",
            _contract_coverage_section(brief),
            "",
            "## Verification Evidence",
            "",
            _verification_evidence_section(brief),
            "",
            "## Human Review Checklist And Reading Order",
            "",
            _string_list(brief.human_review_checklist_and_reading_order, "No prioritized human review items generated"),
            "",
            "## Non-Binding Recommendation",
            "",
            _recommendation_text(brief.non_binding_recommendation),
            "",
        ]
    )
```

Implement helper functions in the same file:

```python
def _change_intent_section(brief: ReviewBrief) -> str:
    intent = brief.change_intent
    return "\n".join(
        [
            f"Goal: {intent.get('goal') or 'No goal recorded'}",
            "",
            "Acceptance criteria:",
            _string_list(intent.get("acceptance_criteria", []), "No acceptance criteria recorded"),
            "",
            "Scope:",
            _string_list(intent.get("scope", []), "No scope recorded"),
            "",
            "Constraints:",
            _string_list(intent.get("constraints", []), "No constraints recorded"),
        ]
    )


def _intent_assessment_section(brief: ReviewBrief) -> str:
    assessment = brief.intent_assessment
    source_counts = assessment.get("source_counts", {})
    source_lines = [f"{key}: {value}" for key, value in dict(source_counts).items()]
    return "\n".join(
        [
            f"Status: {assessment.get('status', 'unknown')}",
            "",
            "Source counts:",
            _string_list(source_lines, "No intent sources recorded"),
            "",
            "Intent uncertainties:",
            _string_list(assessment.get("uncertainties", []), "No intent uncertainties recorded"),
        ]
    )


def _risk_section(brief: ReviewBrief) -> str:
    initial = brief.initial_and_final_risk_assessment["initial"]
    final = brief.initial_and_final_risk_assessment["final"]
    return "\n".join(
        [
            f"Risk level: {initial.get('level', 'unknown')}",
            "",
            "Initial risk reasons:",
            _string_list(initial.get("reasons", []), "No risk reasons recorded"),
            "",
            "Risk signals:",
            _string_list(initial.get("signal_refs", []), "No risk signals recorded"),
            "",
            f"Final risk status: {final.get('status', 'unknown')}",
            f"Final risk level: {final.get('level', 'unknown')}",
            "",
            "Final risk notes:",
            _string_list(final.get("reasons", []), "No final risk notes recorded"),
        ]
    )
```

Add equivalent compact helpers for quality gates, changed files/repository intelligence, findings, rejected hypotheses, contract coverage, verification evidence, and recommendation. Use deterministic fallback strings when lists are empty.

Update `render_markdown_report(...)` to build a brief and render it:

```python
def render_markdown_report(
    review_id: str,
    base_revision: str,
    head_revision: str,
    risk_assessment: RiskAssessment,
    changed_files: list[str],
    reviewer_result: ReviewerResult | None = None,
    observation_summaries: dict[str, str] | None = None,
    repository_intelligence_summary: str | None = None,
    multi_reviewer_summary: dict[str, object] | None = None,
    reconciliation_summary: dict[str, object] | None = None,
    completion_summary: dict[str, object] | None = None,
    intent_packet: IntentPacket | None = None,
    quality_results: list[QualityGateResult] | None = None,
) -> str:
    brief = build_review_brief(
        review_id=review_id,
        base_revision=base_revision,
        head_revision=head_revision,
        intent_packet=intent_packet or IntentPacket(goal=None),
        risk_assessment=risk_assessment,
        changed_files=changed_files,
        quality_results=quality_results or [],
        observation_summaries=observation_summaries,
        repository_intelligence_summary=repository_intelligence_summary,
        reviewer_result=reviewer_result,
        multi_reviewer_summary=multi_reviewer_summary,
        reconciliation_payload=reconciliation_summary,
        completion_summary=completion_summary,
    )
    return render_review_brief_markdown(brief)
```

- [ ] **Step 4: Run tests to verify Task 2 passes**

Run:

```powershell
python -m pytest tests/test_brief.py tests/test_checkpoint_reporting.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src/review_agent/reporting.py tests/test_brief.py tests/test_checkpoint_reporting.py
git commit -m "feat: render spec aligned review brief"
```

---

## Task 3: Wire JSON Brief into CLI artifacts and run state

**Files:**
- Modify: `src/review_agent/cli.py`
- Modify: `tests/test_cli_smoke.py`
- Modify: `tests/test_cli_resume.py`

- [ ] **Step 1: Write failing CLI integration tests**

Append assertions to `tests/test_cli_smoke.py::test_cli_review_writes_state_and_preflight_summary`:

```python
brief = json.loads((run_dirs[-1] / "review_brief.json").read_text(encoding="utf-8"))

assert "Review brief:" in output
assert "Review brief JSON:" in output
assert "Recommendation:" in output
assert state["artifacts"]["review_brief"] == "review_brief.json"
assert brief["review_id"] == run_dirs[-1].name
assert brief["change_map_and_repository_impact"]["changed_files"] == ["auth.py"]
assert brief["non_binding_recommendation"] == "manual_review"
```

Append to `tests/test_cli_resume.py::test_cli_resume_prints_completed_run_summary`:

```python
assert "review_brief.json (present)" in output
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_cli_smoke.py::test_cli_review_writes_state_and_preflight_summary tests/test_cli_resume.py::test_cli_resume_prints_completed_run_summary -q -p no:cacheprovider
```

Expected: FAIL because `review_brief.json` is not written and final CLI output does not print the new paths.

- [ ] **Step 3: Modify CLI imports and state variables**

In `src/review_agent/cli.py`, import:

```python
from review_agent.brief import build_review_brief, review_brief_to_dict
from review_agent.reporting import render_review_brief_markdown
```

Replace the old `render_markdown_report` import.

Near the existing summary variables, initialize raw payloads:

```python
multi_payload = None
reconciliation_payload = None
completion_payload = None
```

Keep assigning the existing dicts where they are currently created:

```python
multi_payload = multi_reviewer_run_to_dict(multi_run)
...
reconciliation_payload = reconciliation_to_dict(reconciliation)
...
completion_payload = completion_to_dict(completion)
```

- [ ] **Step 4: Build and write the final brief**

Replace the final report block with:

```python
brief = build_review_brief(
    review_id=review_id,
    base_revision=args.base,
    head_revision=args.head,
    intent_packet=intent,
    risk_assessment=risk_assessment,
    changed_files=change_summary.changed_files,
    quality_results=quality_results,
    observation_summaries=observation_store.summaries_by_id(),
    repository_intelligence_summary=repository_intelligence_summary,
    reviewer_result=reviewer_result,
    multi_reviewer_summary=multi_reviewer_summary,
    reconciliation_payload=reconciliation_payload,
    completion_summary=completion_payload,
)
store.write_json("review_brief.json", review_brief_to_dict(brief))
report = render_review_brief_markdown(brief)
(store.run_dir / "report.md").write_text(report, encoding="utf-8")
```

Update final state artifacts:

```python
artifacts={"report": "report.md", "review_brief": "review_brief.json"}
```

Update CLI final output:

```python
print(f"Review foundation completed: {store.run_dir}")
print(f"Review brief: {store.run_dir / 'report.md'}")
print(f"Review brief JSON: {store.run_dir / 'review_brief.json'}")
print(f"Recommendation: {brief.non_binding_recommendation}")
print(f"Remaining uncertainties: {len(brief.uncertainties)}")
```

- [ ] **Step 5: Run tests to verify Task 3 passes**

Run:

```powershell
python -m pytest tests/test_cli_smoke.py::test_cli_review_writes_state_and_preflight_summary tests/test_cli_resume.py::test_cli_resume_prints_completed_run_summary -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
git add src/review_agent/cli.py tests/test_cli_smoke.py tests/test_cli_resume.py
git commit -m "feat: write final review brief artifact"
```

---

## Task 4: Final integration verification and compatibility coverage

**Files:**
- Modify tests only if a regression appears during verification.

- [ ] **Step 1: Run reporting and CLI focused tests**

Run:

```powershell
python -m pytest tests/test_brief.py tests/test_checkpoint_reporting.py tests/test_cli_smoke.py tests/test_cli_resume.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
python -m pytest -q -p no:cacheprovider
```

Expected: PASS. If Windows emits the known pytest temporary-directory cleanup warning after the pass summary, record it in the final handoff and do not change feature code for that warning.

- [ ] **Step 3: Run manual local smoke**

Run:

```powershell
$env:PYTHONPATH='src'; python -c "from review_agent.cli import main; raise SystemExit(main(['review','--repo','.','--base','HEAD~1','--head','HEAD','--non-interactive']))"
```

Expected output includes:

```text
Preflight
Review foundation completed:
Review brief:
Review brief JSON:
Recommendation:
Remaining uncertainties:
```

Capture the generated review id from the run directory path, then run:

```powershell
$env:PYTHONPATH='src'; python -c "from review_agent.cli import main; raise SystemExit(main(['resume','<generated-review-id>','--repo','.']))"
```

Expected output includes:

```text
Resume
Status: completed
report.md (present)
review_brief.json (present)
```

- [ ] **Step 4: Clean manual smoke artifacts**

Run this from the repository root after manual smoke:

```powershell
$root = (Resolve-Path -LiteralPath 'D:\Agent\code review agent').Path; $paths = @('D:\Agent\code review agent\.review-agent', 'D:\Agent\code review agent\src\review_agent\__pycache__', 'D:\Agent\code review agent\tests\__pycache__'); foreach ($path in $paths) { $resolved = Resolve-Path -LiteralPath $path -ErrorAction SilentlyContinue; if ($resolved -and $resolved.Path.StartsWith($root)) { Remove-Item -LiteralPath $resolved.Path -Recurse -Force } }
```

- [ ] **Step 5: Commit verification-only adjustments if any**

If Step 1 or Step 2 required test-only compatibility edits, commit them:

```powershell
git add tests src
git commit -m "test: cover final review brief integration"
```

If no files changed, do not create a commit.

---

## Completion checklist

- `review_brief.json` exists for successful local review runs.
- `state.json` includes `"review_brief": "review_brief.json"` after completion.
- `report.md` uses all section names from spec section 19.
- `render_markdown_report(...)` remains callable for existing tests.
- CLI final output names the Markdown report, JSON brief, recommendation, and uncertainty count.
- `resume` shows `review_brief.json` as a present artifact.
- Full tests pass.


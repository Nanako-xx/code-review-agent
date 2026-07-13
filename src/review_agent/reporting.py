from __future__ import annotations

from typing import Any

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
            _string_list(
                brief.human_review_checklist_and_reading_order,
                "No prioritized human review items generated",
            ),
            "",
            "## Non-Binding Recommendation",
            "",
            _recommendation_text(brief.non_binding_recommendation),
            "",
        ]
    )


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
            "",
            "Claim-level provenance:",
            _intent_provenance_section(intent.get("provenance", [])),
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
            "Clarification and decision history:",
            _clarification_history_section(
                assessment.get("clarification_history", [])
            ),
            "",
            "Unresolved clarification questions:",
            _unresolved_questions_section(
                assessment.get("unresolved_questions", [])
            ),
            "",
            "Unconfirmed inferred claims:",
            _unconfirmed_claims_section(
                assessment.get("unconfirmed_inferred_claims", [])
            ),
            "",
            "Intent uncertainties:",
            _string_list(assessment.get("uncertainties", []), "No intent uncertainties recorded"),
        ]
    )


def _intent_provenance_section(value: object) -> str:
    rows = _mapping_rows(value)
    if not rows:
        return "- No claim-level provenance recorded"

    lines: list[str] = []
    for row in rows:
        field = row.get("field", "unknown")
        claim_value = row.get("value", "No value recorded")
        lines.append(f"- `{field}`: {claim_value}")
        lines.append(
            "  - Provenance: "
            f"{row.get('source', 'unknown')} via {row.get('origin', 'unknown')}; "
            f"confidence {row.get('confidence', 'unknown')}; "
            f"state {row.get('claim_state', 'unknown')}; "
            f"conclusion impact {row.get('conclusion_impact', 'unknown')}"
        )
        lines.append(f"  - Claim ID: {row.get('claim_id', 'unknown')}")
        source_refs = row.get("source_refs", [])
        if source_refs:
            lines.append(f"  - Source refs: {_inline_value(source_refs)}")
        evidence_refs = row.get("evidence_refs", [])
        if evidence_refs:
            lines.append(f"  - Evidence refs: {_inline_value(evidence_refs)}")
    return "\n".join(lines)


def _clarification_history_section(value: object) -> str:
    rows = _mapping_rows(value)
    if not rows:
        return "- No clarification decisions recorded"

    lines: list[str] = []
    for row in rows:
        lines.append(
            f"- [{row.get('status', 'unknown')}] "
            f"`{row.get('field', 'unknown')}`: {row.get('question', '')}"
        )
        lines.append(f"  - Question ID: {row.get('question_id', 'unknown')}")
        lines.append(f"  - Rationale: {row.get('rationale', 'Not recorded')}")
        claim_ids = row.get("claim_ids", [])
        if claim_ids:
            lines.append(f"  - Related claim IDs: {_inline_value(claim_ids)}")
        proposed_values = row.get("proposed_values", [])
        if proposed_values:
            lines.append(f"  - Proposed values: {_inline_value(proposed_values)}")
        resolved_values = row.get("resolved_values", [])
        if resolved_values:
            lines.append(f"  - Resolved values: {_inline_value(resolved_values)}")
        user_response = row.get("user_response")
        if user_response:
            lines.append(f"  - User response: {user_response}")
        continuation_basis = row.get("continuation_basis")
        if continuation_basis:
            lines.append(f"  - Continuation basis: {continuation_basis}")
        decision_id = row.get("decision_id")
        if decision_id:
            lines.append(f"  - Decision ID: {decision_id}")
    return "\n".join(lines)


def _unresolved_questions_section(value: object) -> str:
    rows = _mapping_rows(value)
    if not rows:
        return "- No unresolved clarification questions"

    lines: list[str] = []
    for row in rows:
        lines.append(
            f"- `{row.get('field', 'unknown')}`: {row.get('question', '')} "
            f"(status: {row.get('status', 'unknown')}; "
            f"question ID: {row.get('question_id', 'unknown')})"
        )
        lines.append(f"  - Rationale: {row.get('rationale', 'Not recorded')}")
        claim_ids = row.get("claim_ids", [])
        if claim_ids:
            lines.append(f"  - Related claim IDs: {_inline_value(claim_ids)}")
        proposed_values = row.get("proposed_values", [])
        if proposed_values:
            lines.append(f"  - Proposed values: {_inline_value(proposed_values)}")
    return "\n".join(lines)


def _unconfirmed_claims_section(value: object) -> str:
    rows = _mapping_rows(value)
    if not rows:
        return "- No unconfirmed inferred claims"
    return "\n".join(
        f"- `{row.get('field', 'unknown')}`: {row.get('value', '')} "
        f"(confidence: {row.get('confidence', 'unknown')}; "
        f"origin: {row.get('origin', 'unknown')}; "
        f"claim ID: {row.get('claim_id', 'unknown')})"
        for row in rows
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


def _quality_gates_section(brief: ReviewBrief) -> str:
    if not brief.quality_gates:
        return "- No quality gates recorded"
    return "\n".join(
        f"- {gate.get('name', 'unknown')}: {gate.get('status', 'unknown')} - {gate.get('summary', '')}".rstrip()
        for gate in brief.quality_gates
    )


def _change_map_section(brief: ReviewBrief) -> str:
    change_map = brief.change_map_and_repository_impact
    lines = [
        "Changed files:",
        _string_list(change_map.get("changed_files", []), "No changed files detected"),
        "",
        f"Observation count: {change_map.get('observation_count', 0)}",
    ]
    summary = str(change_map.get("repository_intelligence_summary", "")).strip()
    if summary:
        lines.extend(["", "Repository intelligence:", summary])
    incremental = change_map.get("incremental_priority")
    if isinstance(incremental, dict):
        lines.extend(
            [
                "",
                "Incremental priority map:",
                f"- Range: {incremental.get('from_revision', '')}..{incremental.get('to_revision', '')}",
                _string_list(
                    incremental.get("changed_files", []),
                    "No incremental files detected",
                ),
            ]
        )
    reviewer_summary = change_map.get("reviewer_summary", {})
    if reviewer_summary:
        lines.extend(["", "Reviewer summary:", _dict_lines(dict(reviewer_summary))])
    return "\n".join(lines)


def _verified_findings_section(brief: ReviewBrief) -> str:
    if not brief.verified_findings:
        return "- No verified findings recorded"
    lines = []
    for finding in brief.verified_findings:
        line = f"- [{finding.severity}/{finding.confidence}] {finding.claim}"
        if finding.path:
            location = finding.path
            if finding.line is not None:
                location += f":{finding.line}"
            line += f" Location: {location}"
        if finding.evidence_refs:
            line += f" Evidence: {', '.join(finding.evidence_refs)}"
        if finding.impact:
            line += f" Impact: {finding.impact}"
        if finding.verification_performed:
            line += (
                " Verification: "
                + "; ".join(finding.verification_performed)
            )
        if finding.suggested_action:
            line += f" Suggested action: {finding.suggested_action}"
        lines.append(line)
    return "\n".join(lines)


def _rejected_hypotheses_section(brief: ReviewBrief) -> str:
    if not brief.rejected_hypotheses:
        return "- No rejected hypotheses recorded"
    lines = []
    for item in brief.rejected_hypotheses:
        role = f" ({item.role})" if item.role else ""
        lines.append(f"-{role} {item.claim}: {item.reason}")
    return "\n".join(lines)


def _contract_coverage_section(brief: ReviewBrief) -> str:
    if not brief.review_contract_coverage:
        return "- No review contract coverage recorded"
    rows = []
    for row in brief.review_contract_coverage:
        contract = row.get("contract", "unknown")
        status = row.get("status", "unknown")
        summary = row.get("summary", "")
        rows.append(f"- {contract}: {status} - {summary}".rstrip())
    return "\n".join(rows)


def _verification_evidence_section(brief: ReviewBrief) -> str:
    if not brief.verification_evidence:
        return "- No verification evidence recorded"
    rows = []
    for row in brief.verification_evidence:
        kind = row.get("kind", "evidence")
        if kind == "quality_gate":
            rows.append(
                f"- quality_gate:{row.get('name', 'unknown')} {row.get('status', 'unknown')} - {row.get('summary', '')}".rstrip()
            )
        elif kind == "observation":
            rows.append(f"- observation:{row.get('id', 'unknown')} - {row.get('summary', '')}".rstrip())
        else:
            rows.append(f"- {kind}: {_inline_value(row)}")
    return "\n".join(rows)


def _recommendation_text(recommendation: str) -> str:
    if recommendation == "approve":
        return "Approve is non-binding; human review is still required before merge."
    if recommendation == "needs_work":
        return "Needs work before merge."
    return "Manual review required before merge."


def _string_list(items: object, fallback: str) -> str:
    if not items:
        return f"- {fallback}"
    return "\n".join(f"- {item}" for item in items)


def _mapping_rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, dict)]


def _dict_lines(payload: dict[str, Any]) -> str:
    if not payload:
        return "- No reviewer summary recorded"
    return "\n".join(f"- {key}: {_inline_value(value)}" for key, value in payload.items())


def _inline_value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={item}" for key, item in value.items())
    return str(value)

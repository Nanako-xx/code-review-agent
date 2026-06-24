from __future__ import annotations

from typing import Protocol

from review_agent.git_repo import ChangeSummary
from review_agent.models import IntentPacket, RiskAssessment, RiskAssessmentPacket, RiskLevel


SENSITIVE_PATH_MARKERS = ("auth", "payment", "billing", "security", "migration", "permissions")


class RiskAssessor(Protocol):
    def assess(self, packet: RiskAssessmentPacket) -> RiskAssessment:
        raise NotImplementedError


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
        intent_unknowns=list(intent_packet.unknowns),
        diff_excerpt=list(change_summary.diff_excerpt[:80]),
    )


class LocalRiskAssessor:
    """Deterministic offline assessor for tests and provider-free smoke runs.

    A later LLM provider plan will add a model-backed assessor that consumes the
    same RiskAssessmentPacket shape.
    """

    def assess(self, packet: RiskAssessmentPacket) -> RiskAssessment:
        changed_files = [str(path) for path in packet.change_summary["changed_files"]]
        sensitive_files = [
            path
            for path in changed_files
            if any(marker in path.lower() for marker in SENSITIVE_PATH_MARKERS)
        ]
        quality_gates = packet.deterministic_signals.get("quality_gates", {})
        failed_gates = [name for name, status in quality_gates.items() if status == "failed"]

        if failed_gates:
            level = RiskLevel.HIGH
            reasons = [f"quality gate failed: {name}" for name in failed_gates]
            focus = ["failed quality gate", "regression safety"]
        elif sensitive_files:
            level = RiskLevel.HIGH
            reasons = [f"sensitive path changed: {path}" for path in sensitive_files]
            focus = ["caller compatibility", "regression safety", "test adequacy"]
        elif len(changed_files) > 8:
            level = RiskLevel.MEDIUM
            reasons = [f"many files changed: {len(changed_files)}"]
            focus = ["blast radius", "test adequacy"]
        else:
            level = RiskLevel.LOW
            reasons = ["small non-sensitive change set"]
            focus = ["intent alignment", "changed file sanity"]

        return RiskAssessment(
            level=level,
            dimensions={
                "impact": "derived from changed paths and quality gates",
                "blast_radius": "derived from changed file count",
                "reversibility": "not assessed by local fallback",
                "uncertainty": "derived from intent unknowns",
                "verification_strength": "derived from quality gates",
            },
            reasons=reasons,
            evidence_refs=[],
            unknowns=list(packet.intent_unknowns),
            suggested_focus=focus,
        )

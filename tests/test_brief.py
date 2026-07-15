from collections.abc import Iterator, Mapping
from dataclasses import replace
import json

import pytest

import review_agent.brief as brief_module
from review_agent.brief import (
    build_memory_audit_projection,
    build_review_brief,
    review_brief_to_dict,
)
from review_agent.memory_models import (
    Applicability,
    DurableMemoryRecord,
    GenerationMetadata,
    GitCommitSourceRef,
    MemoryConfidence,
    MemoryKind,
    MemoryScope,
    MemorySelectionDecision,
    MemorySnapshot,
    RecordStatus,
    Sensitivity,
    ValidityPolicy,
)
from review_agent.memory_policy import (
    PolicyCompilation,
    PolicyDisposition,
    PolicyProvenance,
)
from review_agent.models import (
    ClarificationQuestion,
    ClarificationStatus,
    IntentClaim,
    IntentClaimState,
    IntentConfidence,
    IntentField,
    IntentOrigin,
    IntentPacket,
    IntentSource,
    IntentStatus,
    QualityGateResult,
    RiskAssessment,
    RiskLevel,
)
from review_agent.reporting import render_review_brief_markdown


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
                "finding_id": "F-" + "a" * 32,
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
                status="failed",
                command=["python", "-m", "compileall"],
                summary="Syntax error in auth.py",
                observation_ref="O-QG",
                category="compile",
                source="builtin",
                reason="invalid syntax",
                exit_code=1,
                duration_seconds=0.25,
                output_truncated=True,
                sandbox="git_blob_compile",
            )
        ],
        observation_summaries={"O-1": "auth.py changed between base and head"},
        repository_intelligence_summary="Repository Intelligence\n- modified function check auth.py:1-2",
        reconciliation_payload=reconciliation_payload,
        completion_summary=completion_summary,
        final_risk_assessment=final_risk,
        incremental_priority={
            "from_revision": "b" * 40,
            "to_revision": "c" * 40,
            "changed_files": ["auth.py"],
            "diff_stat": "1 file changed",
            "diff_excerpt": ["+reject bad token"],
        },
        planning_summary={
            "risk": {
                "status": "accepted",
                "local_floor": "high",
                "proposed_level": "critical",
                "final_level": "critical",
            },
            "portfolio": {
                "status": "accepted",
                "reviewer_count": 4,
                "policy_actions": ["runtime injected Core Reviewer"],
            },
        },
    )

    payload = review_brief_to_dict(brief)

    assert payload["review_id"] == "review-1"
    assert payload["change_intent"]["goal"] == "Add auth token check"
    assert payload["intent_assessment"]["status"] == "partial"
    assert payload["initial_and_final_risk_assessment"]["initial"]["level"] == "high"
    assert payload["initial_and_final_risk_assessment"]["final"]["status"] == "reassessed"
    assert payload["initial_and_final_risk_assessment"]["final"]["level"] == "critical"
    assert payload["quality_gates"][0]["name"] == "python_compile"
    assert payload["change_map_and_repository_impact"]["changed_files"] == ["auth.py"]
    assert (
        payload["change_map_and_repository_impact"]["incremental_priority"][
            "changed_files"
        ]
        == ["auth.py"]
    )
    assert payload["verified_findings"][0]["claim"] == "Bad token path is not covered"
    assert payload["verified_findings"][0]["finding_id"] == "F-" + "a" * 32
    assert payload["rejected_hypotheses"][0]["claim"] == "Session storage changed"
    assert payload["reviewer_disagreements"] == ["core and adversarial disagree on token expiry"]
    assert payload["review_contract_coverage"][0]["contract"] == "Behavioral Correctness"
    assert payload["non_binding_recommendation"] == "manual_review"
    assert payload["orchestration"]["risk"]["local_floor"] == "high"
    assert "auth.py" in payload["human_review_checklist_and_reading_order"][0]
    markdown = render_review_brief_markdown(brief)
    assert "Incremental priority map:" in markdown
    assert f"{'b' * 40}..{'c' * 40}" in markdown
    assert "[compile/cheap; builtin; non-blocking]: failed (0.25s)" in markdown
    assert "  - Reason: invalid syntax" in markdown
    assert "  - Output truncated: True" in markdown
    assert "  - Sandbox: git_blob_compile" in markdown
    assert "  - Observation: O-QG" in markdown
    assert "## Risk And Portfolio Orchestration" in markdown
    assert "local_floor=high" in markdown
    assert "`F-" + "a" * 32 + "`" in markdown


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
        "## Risk And Portfolio Orchestration",
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


def test_review_brief_discloses_intent_provenance_and_clarification_history() -> None:
    superseded_goal = IntentClaim(
        field=IntentField.GOAL,
        value="Infer the auth behavior",
        source=IntentSource.INFERRED,
        origin=IntentOrigin.LLM_INFERENCE,
        confidence=IntentConfidence.MEDIUM,
        source_refs=["request:description"],
        evidence_refs=["O-intent-1"],
        claim_state=IntentClaimState.SUPERSEDED,
    )
    corrected_goal = IntentClaim(
        field=IntentField.GOAL,
        value="Reject expired auth tokens",
        source=IntentSource.EXPLICIT,
        origin=IntentOrigin.USER_CORRECTION,
        confidence=IntentConfidence.HIGH,
        source_refs=["clarification:goal"],
    )
    inferred_scope = IntentClaim(
        field=IntentField.SCOPE,
        value="auth.py",
        source=IntentSource.INFERRED,
        origin=IntentOrigin.CHANGED_FILES,
        confidence=IntentConfidence.LOW,
        source_refs=["changed_file:auth.py"],
    )
    corrected_question = ClarificationQuestion(
        field=IntentField.GOAL,
        question="Is the inferred goal correct?",
        rationale="The goal changes the behavioral correctness conclusion.",
        proposed_values=[superseded_goal.value],
        claim_ids=[superseded_goal.claim_id],
        status=ClarificationStatus.CORRECTED,
        user_response="The review should cover expired tokens.",
        resolved_values=[corrected_goal.value],
        decision_id="decision-goal-correction",
    )
    open_question = ClarificationQuestion(
        field=IntentField.SCOPE,
        question="Is auth.py the complete intended scope?",
        rationale="The changed file is the only available scope signal.",
        proposed_values=[inferred_scope.value],
        claim_ids=[inferred_scope.claim_id],
        status=ClarificationStatus.OPEN,
    )
    intent = IntentPacket(
        goal=corrected_goal.value,
        scope=[inferred_scope.value],
        sources={
            IntentField.GOAL.value: IntentSource.EXPLICIT,
            IntentField.SCOPE.value: IntentSource.INFERRED,
        },
        status=IntentStatus.PARTIAL,
        uncertainties=["intended scope contains unconfirmed inferred values"],
        provenance=[superseded_goal, corrected_goal, inferred_scope],
        clarifications=[corrected_question, open_question],
    )
    brief = build_review_brief(
        review_id="review-intent-history",
        base_revision="base",
        head_revision="head",
        intent_packet=intent,
        risk_assessment=RiskAssessment(
            level=RiskLevel.MEDIUM,
            dimensions={},
            reasons=[],
            signal_refs=[],
            uncertainties=[],
            suggested_focus=[],
        ),
        changed_files=["auth.py"],
        quality_results=[],
    )

    payload = review_brief_to_dict(brief)

    provenance = payload["change_intent"]["provenance"]
    assert provenance[0] == {
        "claim_id": superseded_goal.claim_id,
        "field": "goal",
        "value": "Infer the auth behavior",
        "source": "inferred",
        "origin": "llm_inference",
        "confidence": "medium",
        "source_refs": ["request:description"],
        "evidence_refs": ["O-intent-1"],
        "claim_state": "superseded",
        "conclusion_impact": "material",
    }
    assessment = payload["intent_assessment"]
    assert assessment["clarification_history"][0]["status"] == "corrected"
    assert (
        assessment["clarification_history"][0]["decision_id"]
        == "decision-goal-correction"
    )
    assert assessment["unresolved_questions"] == [
        assessment["clarification_history"][1]
    ]
    assert assessment["unconfirmed_inferred_claims"] == [provenance[2]]

    markdown = render_review_brief_markdown(brief)
    assert "Claim-level provenance:" in markdown
    assert "inferred via changed_files" in markdown
    assert "Clarification and decision history:" in markdown
    assert "Decision ID: decision-goal-correction" in markdown
    assert "Unresolved clarification questions:" in markdown
    assert "Is auth.py the complete intended scope?" in markdown
    assert "Unconfirmed inferred claims:" in markdown
    assert "intended scope contains unconfirmed inferred values" in markdown


def test_review_brief_discloses_semantic_and_supplemental_audit_details() -> None:
    semantic_payload = {
        "schema_version": "semantic_reconciliation_v1",
        "status": "partial",
        "canonical_findings": [],
        "rejected_findings": [
            {
                "candidate_id": "F-1",
                "reviewer_index": 1,
                "role": "Adversarial Reviewer",
                "claim": "Session storage changed",
                "reason": "unsupported_claim",
                "rationale": "No authorized Observation supports the claim.",
                "evidence_refs": ["O-1"],
                "missing_evidence_refs": ["O-missing"],
                "decision_refs": ["O-2"],
                "decision_source": "semantic_reconciler",
            }
        ],
        "conflicts_resolved": [
            {
                "conflict_id": "C-resolved",
                "candidate_ids": ["F-1"],
                "status": "resolved",
                "issue": "Two claims used different wording.",
                "resolution": "They describe the same behavior.",
                "decision_refs": ["O-1"],
                "decision_source": "semantic_reconciler",
            }
        ],
        "remaining_disagreements": [
            {
                "conflict_id": "C-open",
                "candidate_ids": ["F-2"],
                "status": "unresolved",
                "issue": "Runtime behavior remains unverified.",
                "resolution": "",
                "decision_refs": [],
                "decision_source": "runtime_policy",
            }
        ],
        "contract_coverage": [],
        "evidence_quality": "mixed",
        "supplemental": {
            "status": "budget_exhausted",
            "waves": 2,
            "tasks": 3,
            "completed": 1,
            "partial": 1,
            "failed": 0,
            "unavailable": 1,
            "budget": {
                "limits": {"tasks": 3, "tokens": 1000},
                "charged": {"tasks": 2, "tokens": 600},
                "unknown_consumed": {"tasks": 1, "tokens": 300},
                "reserved": {"tasks": 0, "tokens": 0},
                "remaining": {"tasks": 0, "tokens": 100},
            },
            "stop_reason": "max_waves",
        },
        "policy_actions": ["preserved_severe_finding:F-2"],
        "uncertainties": ["A targeted runtime check did not complete."],
        "model": {
            "status": "fallback",
            "invocation_ids": ["I-semantic"],
            "input_digests": ["a" * 64],
        },
    }
    brief = build_review_brief(
        review_id="review-semantic-brief",
        base_revision="base",
        head_revision="head",
        intent_packet=IntentPacket(
            goal="Preserve behavior",
            status=IntentStatus.SUFFICIENT,
        ),
        risk_assessment=RiskAssessment(
            level=RiskLevel.HIGH,
            dimensions={},
            reasons=[],
            signal_refs=[],
            uncertainties=[],
            suggested_focus=[],
        ),
        changed_files=["app.py"],
        quality_results=[],
        completion_summary={
            "recommendation": "manual_review",
            "uncertainties": ["Semantic reconciliation used deterministic fallback"],
        },
        semantic_reconciliation_payload=semantic_payload,
    )

    payload = review_brief_to_dict(brief)
    markdown = render_review_brief_markdown(brief)

    assert payload["semantic_reconciliation"] == semantic_payload
    assert "## Semantic Reconciliation And Supplemental Investigation" in markdown
    assert "Status: partial" in markdown
    assert "They describe the same behavior." in markdown
    assert "Runtime behavior remains unverified." in markdown
    assert "No authorized Observation supports the claim." in markdown
    assert "decision_source=semantic_reconciler" in markdown
    assert "status=budget_exhausted, stop_reason=max_waves" in markdown
    assert "unknown_consumed: tasks=1, tokens=300" in markdown
    assert "preserved_severe_finding:F-2" in markdown
    assert "Semantic reconciliation used fallback or is partial" in markdown


def test_legacy_review_brief_omits_empty_semantic_sidecar_and_section() -> None:
    brief = build_review_brief(
        review_id="review-legacy-brief",
        base_revision="base",
        head_revision="head",
        intent_packet=IntentPacket(goal="Preserve behavior"),
        risk_assessment=RiskAssessment(
            level=RiskLevel.LOW,
            dimensions={},
            reasons=[],
            signal_refs=[],
            uncertainties=[],
            suggested_focus=[],
        ),
        changed_files=[],
        quality_results=[],
    )

    assert "semantic_reconciliation" not in review_brief_to_dict(brief)
    assert (
        "## Semantic Reconciliation And Supplemental Investigation"
        not in render_review_brief_markdown(brief)
    )
    assert "memory_audit" not in review_brief_to_dict(brief)
    assert "## Memory Audit" not in render_review_brief_markdown(brief)


def test_review_brief_projects_bounded_memory_audit_with_markdown_parity() -> None:
    memory_id = "MEM-" + "1" * 64
    stale_memory_id = "MEM-" + "2" * 64
    candidate_id = "MC-" + "3" * 64
    entry_id = "RKE-" + "4" * 64
    secret = "raw-secret-that-must-not-be-reported"
    memory_audit_payload = {
        "snapshot": {
            "snapshot_id": "MSNAP-" + "5" * 64,
            "snapshot_hash": "5" * 64,
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "selection_policy_version": "memory_selection_v1",
            "memory_generation": 7,
            "feedback_generation": 8,
            "knowledge_generation": 9,
            "eligible_records": [
                {
                    "memory_id": memory_id,
                    "candidate_id": "MC-" + "6" * 64,
                    "kind": "review_rule",
                    "statement": "Require the registered auth regression check.",
                    "scope": {
                        "schema_version": 1,
                        "paths": ["auth/**"],
                        "symbols": [],
                        "contracts": ["behavioral-correctness"],
                        "languages": ["python"],
                    },
                    "source_refs": [
                        {
                            "schema_version": 1,
                            "type": "repository_range",
                            "revision": "a" * 40,
                            "path": "auth/check.py",
                            "line_start": 10,
                            "line_end": 20,
                            "content_hash": "7" * 64,
                            "raw_excerpt": secret,
                        }
                    ],
                    "source_bundle_hash": "8" * 64,
                    "source_bundle": {"raw": secret},
                    "valid_from_sha": "a" * 40,
                    "validity_policies": ["source_content_hash"],
                    "policy_effect": {
                        "schema_version": 1,
                        "type": "require_check",
                        "value": "auth-regression",
                    },
                    "approved_by": "review-owner",
                    "approval_event_id": "EVT-" + "9" * 64,
                    "status": "active",
                }
            ],
            "applicability_decisions": [
                {
                    "memory_id": memory_id,
                    "applicability": "selected",
                    "reason_codes": ["target_revision_valid", "selected"],
                    "rank": 0,
                },
                {
                    "memory_id": stale_memory_id,
                    "applicability": "lineage_mismatch",
                    "reason_codes": ["diverged_lineage"],
                    "rank": 1,
                },
                {
                    "memory_id": "MEM-" + "a" * 64,
                    "applicability": "source_changed",
                    "reason_codes": ["record_revalidation_required"],
                    "rank": 2,
                },
            ],
            "feedback_calibration_summary": {
                "policy_version": "feedback_aggregation_v1",
                "eligible": True,
                "source_feedback_ids": ["FB-" + str(index) * 64 for index in range(1, 6)],
                "source_review_ids": ["review-a", "review-b", "review-c"],
                "raw_feedback": [{"reason": secret}],
            },
            "repository_knowledge_refs": [entry_id],
            "database": {"dump": secret},
        },
        "compiled_policy": {
            "policy_version": "memory_policy_v1",
            "initial_risk_floor": "medium",
            "effective_risk_floor": "high",
            "blocked": True,
            "actions": [
                {
                    "type": "require_check",
                    "check_id": "auth-regression",
                    "memory_ids": [memory_id],
                }
            ],
            "diagnostics": [
                {
                    "code": "unknown_check",
                    "severity": "blocking",
                    "message": "Required check is unavailable.",
                    "memory_id": memory_id,
                }
            ],
            "provenance": [
                {
                    "memory_id": memory_id,
                    "candidate_id": "MC-" + "6" * 64,
                    "approved_by": "review-owner",
                    "approval_event_id": "EVT-" + "9" * 64,
                    "effect_kind": "require_check",
                    "effect_value": "auth-regression",
                    "disposition": "applied",
                    "runtime_action_kind": "require_check",
                    "diagnostic_codes": ["unknown_check"],
                }
            ],
            "hidden_reasoning": secret,
        },
        "cache_provenance": [
            {
                "status": "hit",
                "key_hash": "b" * 64,
                "revision_binding": "head@" + "b" * 40,
                "capability": "symbol_index",
                "analyzer": {"name": "ast", "version": "1"},
                "entry_id": entry_id,
                "blob_hash": "c" * 64,
                "persistent": True,
                "session_pinned": True,
                "content": secret,
            }
        ],
        "pending_candidates": [
            {
                "candidate_id": candidate_id,
                "kind": "business_invariant",
                "statement": "Token expiry must remain backward compatible.",
                "scope": {
                    "paths": ["auth/**"],
                    "symbols": [],
                    "contracts": [],
                    "languages": [],
                },
                "status": "pending_approval",
                "raw_curator_response": secret,
            }
        ],
        "status": {
            "mode": "read-write",
            "available": False,
            "unavailable_reason": "store_corrupt",
            "hard_policy_blocked": True,
            "outbox_pending": True,
            "outbox": {
                "request_id": "REQ-" + "d" * 64,
                "candidate_ids": [candidate_id],
                "raw_payload": secret,
            },
            "curator": {
                "mode": "model",
                "status": "fallback",
                "warning_codes": ["provider_failure"],
                "raw_response": secret,
            },
            "database": secret,
        },
        "raw_feedback": secret,
        "raw_curator_response": secret,
        "hidden_reasoning": secret,
        "blob": secret,
    }
    brief = build_review_brief(
        review_id="review-memory-audit",
        base_revision="a" * 40,
        head_revision="b" * 40,
        intent_packet=IntentPacket(goal="Preserve auth behavior"),
        risk_assessment=RiskAssessment(
            level=RiskLevel.HIGH,
            dimensions={},
            reasons=[],
            signal_refs=[],
            uncertainties=[],
            suggested_focus=[],
        ),
        changed_files=["auth/check.py"],
        quality_results=[],
        memory_audit_payload=memory_audit_payload,
    )

    payload = review_brief_to_dict(brief)
    audit = payload["memory_audit"]
    markdown = render_review_brief_markdown(brief)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    assert audit["schema_version"] == "memory_audit_v1"
    assert audit["applied_memory"][0]["memory_id"] == memory_id
    assert audit["applied_memory"][0]["authority"] == "runtime_compiled_policy"
    assert audit["applied_memory"][0]["source_refs"][0]["path"] == "auth/check.py"
    assert audit["applied_memory"][0]["validity"]["applicability"] == "selected"
    assert audit["compiled_policy"]["blocked"] is True
    assert audit["cache_provenance"][0]["entry_id"] == entry_id
    assert {warning["category"] for warning in audit["warnings"]} == {
        "lineage",
        "revalidation",
    }
    assert audit["feedback_summary"] == {
        "policy_version": "feedback_aggregation_v1",
        "sample_count": 5,
        "review_count": 3,
        "eligible": True,
    }
    assert audit["pending_candidates"] == [
        {
            "candidate_id": candidate_id,
            "kind": "business_invariant",
            "statement": "Token expiry must remain backward compatible.",
            "scope": {
                "paths": ["auth/**"],
                "symbols": [],
                "contracts": [],
                "languages": [],
            },
            "status": "pending_approval",
            "active": False,
            "approval_hint": f"memory approve {candidate_id} --actor <actor> --reason <reason>",
        }
    ]
    assert audit["status"]["memory_unavailable"] is True
    assert audit["status"]["hard_policy_blocked"] is True
    assert audit["status"]["outbox_pending"] is True
    assert audit["status"]["curator"]["status"] == "fallback"
    assert candidate_id not in {
        item["memory_id"] for item in audit["applied_memory"]
    }
    assert secret not in encoded
    assert secret not in markdown

    for expected in (
        "## Memory Audit",
        "### Applied Memory",
        memory_id,
        "runtime_compiled_policy",
        "### Runtime-Compiled Policy",
        "hard_policy_blocked: True",
        "### Repository Knowledge Cache Provenance",
        entry_id,
        "### Memory Validity Warnings",
        "diverged_lineage",
        "record_revalidation_required",
        "### Feedback Calibration Summary",
        "feedback_aggregation_v1",
        "sample_count: 5",
        "### Pending Memory Candidates",
        candidate_id,
        f"memory approve {candidate_id}",
        "memory_unavailable: True",
        "outbox_pending: True",
        "curator status: fallback",
    ):
        assert expected in markdown


class _StoreBackedDuck:
    """A hostile object whose conversion would be a Store query."""

    calls = 0

    def to_dict(self):  # pragma: no cover - the test asserts this is unreachable
        type(self).calls += 1
        raise AssertionError("unknown conversion method was called")

    @property
    def eligible_records(self):  # pragma: no cover - also must be unreachable
        type(self).calls += 1
        raise AssertionError("unknown property was queried")

    def __deepcopy__(self, memo):  # pragma: no cover - catches the old asdict path
        type(self).calls += 1
        raise AssertionError("memory sidecar was deep-copied")


class _StoreBackedMapping(Mapping[str, object]):
    calls = 0

    def __getitem__(self, key: str) -> object:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("custom Mapping was queried")

    def __iter__(self) -> Iterator[str]:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("custom Mapping was iterated")

    def __len__(self) -> int:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("custom Mapping was sized")


def _memory_test_brief(memory_audit: object) -> object:
    return build_review_brief(
        review_id="review-memory-boundary",
        base_revision="base",
        head_revision="head",
        intent_packet=IntentPacket(goal="Preserve behavior"),
        risk_assessment=RiskAssessment(
            level=RiskLevel.MEDIUM,
            dimensions={},
            reasons=[],
            signal_refs=[],
            uncertainties=[],
            suggested_focus=[],
        ),
        changed_files=["app.py"],
        quality_results=[],
        memory_audit_payload=memory_audit,
    )


def test_memory_audit_never_calls_duck_conversion_or_deepcopy() -> None:
    _StoreBackedDuck.calls = 0
    _StoreBackedMapping.calls = 0

    assert build_memory_audit_projection(_StoreBackedDuck())["status"]["degraded"]
    assert build_memory_audit_projection(_StoreBackedMapping())["status"]["degraded"]

    brief = _memory_test_brief({})
    unsafe_brief = replace(brief, memory_audit=_StoreBackedDuck())
    payload = review_brief_to_dict(unsafe_brief)

    assert payload["memory_audit"]["status"]["degraded"] is True
    assert _StoreBackedDuck.calls == 0
    assert _StoreBackedMapping.calls == 0


@pytest.mark.parametrize(
    ("payload_factory", "expected_code"),
    [
        (
            lambda: {
                "status": {
                    "nested": {
                        "again": {
                            "again": {
                                "again": {
                                    "again": {
                                        "again": {
                                            "again": {"leaf": "value"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
            },
            "memory_audit_depth_exceeded",
        ),
        (
            lambda: {"status": {f"key_{index}": "value" for index in range(513)}},
            "memory_audit_keys_exceeded",
        ),
        (
            lambda: {"pending_candidates": ["item"] * 257},
            "memory_audit_items_exceeded",
        ),
        (
            lambda: {"status": {"reason": "S" * (brief_module._AUDIT_MAX_TEXT + 1)}},
            "memory_audit_string_exceeded",
        ),
        (
            lambda: {
                "status": {
                    "blob": ["SECRET-" + ("x" * 6990) for _ in range(20)]
                }
            },
            "memory_audit_bytes_exceeded",
        ),
    ],
)
def test_memory_audit_limits_fail_closed_without_source_text(
    payload_factory, expected_code: str
) -> None:
    payload = payload_factory()
    audit = build_memory_audit_projection(payload)
    encoded = json.dumps(audit, sort_keys=True)

    assert audit["applied_memory"] == []
    assert audit["status"]["degraded"] is True
    assert expected_code in audit["status"]["degradation_reasons"]
    assert "SECRET-" not in encoded


def test_memory_audit_validates_sensitive_scalars_and_fixed_taxonomies() -> None:
    secret = "raw-feedback-hidden-reasoning-secret"
    memory_id = "MEM-" + "a" * 64
    candidate_id = "MC-" + "b" * 64
    entry_id = "RKE-" + "c" * 64
    audit = build_memory_audit_projection(
        {
            "compiled_policy": {
                "policy_version": "memory_policy_v1",
                "blocked": "true",
                "diagnostics": [
                    {
                        "code": "unknown_check",
                        "severity": "blocking",
                        "message": secret,
                    }
                ],
            },
            "cache_provenance": [
                {
                    "status": "rebuild",
                    "entry_id": entry_id,
                    "blob_hash": secret,
                    "corruption_reason": secret,
                    "persistent": "true",
                    "fallback": {"strategy": secret, "reason": secret},
                }
            ],
            "status": {
                "hard_policy_blocked": "true",
                "degradation_reasons": [secret, "outbox_pending"],
                "curator": {
                    "mode": "model",
                    "status": "fallback",
                    "outcome": "rejected",
                    "attempt_count": secret,
                    "candidate_ids": [candidate_id, secret],
                    "warning_codes": ["provider_failure", secret],
                    "review_conclusion_impact": secret,
                },
            },
            "raw_feedback": secret,
            "hidden_reasoning": secret,
        }
    )
    encoded = json.dumps(audit, sort_keys=True)

    assert secret not in encoded
    assert audit["compiled_policy"]["blocked"] is True
    assert audit["compiled_policy"]["diagnostics"] == [
        {"code": "unknown_check", "severity": "blocking"}
    ]
    cache = audit["cache_provenance"][0]
    assert cache["entry_id"] == entry_id
    assert "blob_hash" not in cache
    assert cache["corruption_reason"] == "unknown"
    assert "fallback" not in cache
    curator = audit["status"]["curator"]
    assert curator["candidate_ids"] == [candidate_id]
    assert "attempt_count" not in curator
    assert "review_conclusion_impact" not in curator
    assert set(audit["status"]["degradation_reasons"]) == {
        "outbox_pending",
        "hard_policy_blocked",
        "curator_fallback",
    }


def test_applied_memory_requires_active_record_selected_decision_approval_and_provenance() -> None:
    def record(index: int, **changes: object) -> dict[str, object]:
        value: dict[str, object] = {
            "memory_id": "MEM-" + (str(index) * 64),
            "candidate_id": "MC-" + (str(index + 1) * 64),
            "kind": "review_rule",
            "statement": "The registered check remains required.",
            "scope": {"paths": ["auth/**"], "symbols": [], "contracts": [], "languages": []},
            "source_refs": [],
            "source_bundle_hash": "f" * 64,
            "valid_from_sha": "a" * 40,
            "validity_policies": ["manual_until_revoked"],
            "approved_by": "review-owner",
            "approval_event_id": "EVT-" + (str(index + 2) * 64),
            "status": "active",
        }
        value.update(changes)
        return value

    runtime_id = "MEM-" + "1" * 64
    human_id = "MEM-" + "2" * 64
    missing_status_id = "MEM-" + "3" * 64
    missing_approval_id = "MEM-" + "4" * 64
    missing_decision_id = "MEM-" + "5" * 64
    pending_id = "MEM-" + "6" * 64
    records = [
        record(1, memory_id=runtime_id),
        record(2, memory_id=human_id, authority="runtime_compiled_policy"),
        record(3, memory_id=missing_status_id),
        record(4, memory_id=missing_approval_id),
        record(5, memory_id=missing_decision_id),
        record(6, memory_id=pending_id, status="pending"),
    ]
    records[2].pop("status")
    records[3].pop("approved_by")
    decisions = [
        {
            "memory_id": memory_id,
            "applicability": "selected",
            "reason_codes": ["selected"],
        }
        for memory_id in (
            runtime_id,
            human_id,
            missing_status_id,
            missing_approval_id,
            pending_id,
        )
    ]
    policy_provenance = [
        {
            "memory_id": runtime_id,
            "disposition": "applied",
            "effect_kind": "require_check",
            "effect_value": "auth-regression",
            "runtime_action_kind": "require_check",
        },
        {
            "memory_id": human_id,
            "disposition": "informational",
        },
        *[
            {
                "memory_id": memory_id,
                "disposition": "applied",
                "effect_kind": "require_check",
                "effect_value": "auth-regression",
                "runtime_action_kind": "require_check",
            }
            for memory_id in (
                missing_status_id,
                missing_approval_id,
                missing_decision_id,
                pending_id,
            )
        ],
    ]
    audit = build_memory_audit_projection(
        {
            "snapshot": {
                "eligible_records": records,
                "applicability_decisions": decisions,
            },
            "compiled_policy": {"provenance": policy_provenance},
        }
    )

    applied = {row["memory_id"]: row for row in audit["applied_memory"]}
    not_applied = {
        row["memory_id"]: row for row in audit["not_applied_memory"]
    }
    assert set(applied) == {runtime_id, human_id}
    assert applied[runtime_id]["authority"] == "runtime_compiled_policy"
    assert applied[human_id]["authority"] == "human_approved_context"
    assert set(not_applied) == {
        missing_status_id,
        missing_approval_id,
        missing_decision_id,
        pending_id,
    }
    assert not_applied[missing_status_id]["reason_code"] == "record_status_missing"
    assert not_applied[missing_approval_id]["reason_code"] == "approval_missing"
    assert not_applied[missing_decision_id]["reason_code"] == "selection_missing"
    assert not_applied[pending_id]["reason_code"] == "invalid_record"


def test_memory_audit_accepts_exact_canonical_snapshot_and_policy_types() -> None:
    candidate_id = "MC-" + "1" * 64
    repository_key = "2" * 64
    scope = MemoryScope(paths=("auth/**",))
    record = DurableMemoryRecord(
        candidate_id=candidate_id,
        repository_key=repository_key,
        kind=MemoryKind.REVIEW_RULE,
        statement="The registered auth check remains required.",
        scope=scope,
        source_refs=(GitCommitSourceRef(commit_sha="3" * 40),),
        source_bundle_hash="4" * 64,
        valid_from_sha="3" * 40,
        validity_policies=(ValidityPolicy.MANUAL_UNTIL_REVOKED,),
        confidence=MemoryConfidence.HIGH,
        sensitivity=Sensitivity.NORMAL,
        policy_effect=None,
        approved_by="review-owner",
        approval_event_id="EVT-" + "5" * 64,
        status=RecordStatus.ACTIVE,
        created_at="2026-01-01T00:00:00Z",
    )
    decision = MemorySelectionDecision(
        memory_id=record.memory_id,
        applicability=Applicability.SELECTED,
        matched_scope=scope,
        reason_codes=("selected",),
        rank=0,
    )
    snapshot = MemorySnapshot(
        repository_key=repository_key,
        base_sha="3" * 40,
        head_sha="6" * 40,
        generations=GenerationMetadata(
            store_schema_version=1,
            memory_generation=1,
            feedback_generation=0,
            knowledge_generation=0,
        ),
        selection_policy_version="memory_selection_v1",
        eligible_records=(record,),
        applicability_decisions=(decision,),
        feedback_calibration_summary=None,
        repository_knowledge_refs=(),
        created_at="2026-01-01T00:00:00Z",
    )
    policy = PolicyCompilation(
        initial_risk_floor=RiskLevel.MEDIUM,
        effective_risk_floor=RiskLevel.MEDIUM,
        actions=(),
        diagnostics=(),
        provenance=(
            PolicyProvenance(
                memory_id=record.memory_id,
                disposition=PolicyDisposition.INFORMATIONAL,
                candidate_id=record.candidate_id,
                approved_by=record.approved_by,
                approval_event_id=record.approval_event_id,
            ),
        ),
    )

    audit = build_memory_audit_projection(
        snapshot=snapshot,
        compiled_policy=policy,
    )

    assert [row["memory_id"] for row in audit["applied_memory"]] == [record.memory_id]
    assert audit["applied_memory"][0]["authority"] == "human_approved_context"
    assert audit["applied_memory"][0]["source_refs"][0]["type"] == "git_commit"
    assert audit["status"].get("degraded") is not True

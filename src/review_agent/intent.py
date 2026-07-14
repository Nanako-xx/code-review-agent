from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import replace

from review_agent.git_repo import ChangeSummary
from review_agent.models import (
    ClarificationQuestion,
    ClarificationStatus,
    ConclusionImpact,
    IntentClaim,
    IntentClaimState,
    IntentConfidence,
    IntentDecision,
    IntentDecisionAction,
    IntentField,
    IntentOrigin,
    IntentPacket,
    IntentSource,
    IntentStatus,
    ReviewRequest,
)


_FIELD_ORDER = (
    IntentField.GOAL,
    IntentField.ACCEPTANCE_CRITERIA,
    IntentField.SCOPE,
    IntentField.CONSTRAINTS,
)
_EXTRACTED_EXPLICIT_ORIGINS = {
    IntentOrigin.REPOSITORY_DOCUMENT,
    IntentOrigin.REPOSITORY_TEST,
    IntentOrigin.COMMIT_MESSAGE,
}
_SENSITIVE_PATH_MARKERS = (
    "api",
    "auth",
    "billing",
    "credential",
    "migration",
    "payment",
    "permission",
    "schema",
    "security",
    "token",
)


def collect_deterministic_claims(
    request: ReviewRequest,
    change_summary: ChangeSummary,
) -> list[IntentClaim]:
    """Collect explicit request metadata and deterministic changed-file baselines."""
    claims: list[IntentClaim] = []

    user_intent = _clean_optional(request.user_intent)
    if user_intent:
        claims.append(
            _claim(
                IntentField.GOAL,
                user_intent,
                IntentSource.EXPLICIT,
                IntentOrigin.USER_INPUT,
                IntentConfidence.HIGH,
                ["request.user_intent"],
            )
        )

    title = _clean_optional(request.title)
    if title:
        claims.append(
            _claim(
                IntentField.GOAL,
                title,
                IntentSource.EXPLICIT,
                IntentOrigin.REQUEST_METADATA,
                IntentConfidence.HIGH,
                ["request.title"],
            )
        )

    description = _clean_optional(request.description)
    if description:
        claims.append(
            _claim(
                IntentField.ACCEPTANCE_CRITERIA,
                description,
                IntentSource.EXPLICIT,
                IntentOrigin.REQUEST_METADATA,
                IntentConfidence.HIGH,
                ["request.description"],
            )
        )

    for index, requirement in enumerate(request.linked_requirements):
        value = _clean_optional(requirement)
        if value:
            claims.append(
                _claim(
                    IntentField.ACCEPTANCE_CRITERIA,
                    value,
                    IntentSource.EXPLICIT,
                    IntentOrigin.REQUEST_METADATA,
                    IntentConfidence.HIGH,
                    [f"request.linked_requirements:{index}"],
                )
            )

    changed_files = _unique_clean_values(change_summary.changed_files)
    if not any(claim.field is IntentField.GOAL for claim in claims) and changed_files:
        files = ", ".join(changed_files[:3])
        claims.append(
            _claim(
                IntentField.GOAL,
                f"Review changes touching {files}",
                IntentSource.INFERRED,
                IntentOrigin.CHANGED_FILES,
                IntentConfidence.MEDIUM,
                [f"diff:{path}" for path in changed_files[:3]],
            )
        )

    for path in changed_files:
        claims.append(
            _claim(
                IntentField.SCOPE,
                path,
                IntentSource.INFERRED,
                IntentOrigin.CHANGED_FILES,
                IntentConfidence.MEDIUM,
                [f"diff:{path}"],
            )
        )

    for index, rule in enumerate(request.project_rules):
        value = _clean_optional(rule)
        if value:
            claims.append(
                _claim(
                    IntentField.CONSTRAINTS,
                    value,
                    IntentSource.EXPLICIT,
                    IntentOrigin.PROJECT_RULE,
                    IntentConfidence.HIGH,
                    [f"request.project_rules:{index}"],
                )
            )

    return _deduplicate_claims(claims)


def merge_inference_claims(
    baseline_claims: Sequence[IntentClaim],
    inference_claims: Sequence[IntentClaim],
    *,
    authorized_evidence_refs: Collection[str] = (),
) -> list[IntentClaim]:
    """Validate, normalize, and deterministically merge model-proposed claims."""
    _validate_claim_sequence(baseline_claims, "baseline_claims")
    _validate_claim_sequence(inference_claims, "inference_claims")
    authorized = set(authorized_evidence_refs)
    if any(not isinstance(ref, str) or not ref.strip() for ref in authorized):
        raise ValueError("authorized_evidence_refs must contain non-empty strings")

    normalized: list[IntentClaim] = list(baseline_claims)
    for claim in inference_claims:
        candidate = claim
        if any(ref not in authorized for ref in candidate.evidence_refs):
            candidate = replace(candidate, claim_state=IntentClaimState.INVALID)
        elif candidate.origin in {
            IntentOrigin.USER_CONFIRMATION,
            IntentOrigin.USER_CORRECTION,
        }:
            candidate = replace(candidate, claim_state=IntentClaimState.INVALID)
        elif candidate.source is IntentSource.EXPLICIT:
            has_authorized_raw_source = (
                candidate.origin in _EXTRACTED_EXPLICIT_ORIGINS
                and bool(candidate.source_refs)
                and bool(candidate.evidence_refs)
            )
            if not has_authorized_raw_source:
                candidate = replace(candidate, source=IntentSource.INFERRED)
        normalized.append(candidate)

    return _deduplicate_claims(normalized)


def generate_material_questions(
    claims: Sequence[IntentClaim],
    *,
    sensitive_change: bool = False,
) -> list[ClarificationQuestion]:
    """Create one deterministic clarification question per material field."""
    _validate_claim_sequence(claims, "claims")
    active = _active_claims(_deduplicate_claims(claims))
    questions: list[ClarificationQuestion] = []

    for intent_field in _FIELD_ORDER:
        field_claims = [claim for claim in active if claim.field is intent_field]
        material_claims = [
            claim
            for claim in field_claims
            if claim.conclusion_impact
            in {ConclusionImpact.BLOCKING, ConclusionImpact.MATERIAL}
        ]
        inferred_claims = [
            claim for claim in material_claims if claim.source is IntentSource.INFERRED
        ]
        has_goal_conflict = (
            intent_field is IntentField.GOAL
            and len({claim.claim_id for claim in field_claims}) > 1
        )
        missing_is_material = intent_field in {
            IntentField.GOAL,
            IntentField.ACCEPTANCE_CRITERIA,
            IntentField.SCOPE,
        } or (intent_field is IntentField.CONSTRAINTS and sensitive_change)

        if has_goal_conflict:
            target_claims = field_claims
            reason = "conflict"
        elif inferred_claims:
            target_claims = inferred_claims
            reason = "inferred"
        elif not field_claims and missing_is_material:
            target_claims = []
            reason = "missing"
        else:
            continue

        questions.append(
            ClarificationQuestion(
                field=intent_field,
                question=_question_text(intent_field, reason, target_claims),
                rationale=_question_rationale(intent_field, reason),
                proposed_values=[claim.value for claim in target_claims],
                claim_ids=[claim.claim_id for claim in target_claims],
            )
        )

    return questions


def apply_user_decision(
    claims: Sequence[IntentClaim],
    questions: Sequence[ClarificationQuestion],
    decision: IntentDecision,
) -> tuple[list[IntentClaim], list[ClarificationQuestion]]:
    """Apply one idempotent clarification decision without doing any I/O."""
    _validate_claim_sequence(claims, "claims")
    _validate_question_sequence(questions)
    if not isinstance(decision, IntentDecision):
        raise ValueError("decision must be an IntentDecision")

    matching = [
        question for question in questions if question.question_id == decision.question_id
    ]
    if len(matching) != 1:
        raise ValueError("decision must reference exactly one clarification question")
    question = matching[0]
    if question.status not in {ClarificationStatus.PENDING, ClarificationStatus.OPEN}:
        if question.decision_id == decision.decision_id:
            return list(claims), list(questions)
        raise ValueError("clarification question already has a different decision")

    target_ids = set(question.claim_ids)
    updated_claims = list(claims)
    resolved_values: list[str] = []

    if decision.action is IntentDecisionAction.CONFIRMED:
        active_targets = [
            claim
            for claim in updated_claims
            if claim.claim_id in target_ids
            and claim.claim_state is IntentClaimState.ACTIVE
        ]
        if question.field is IntentField.GOAL and len(active_targets) > 1:
            raise ValueError(
                "conflicting goal candidates require correction or rejection"
            )
        confirmable = [
            claim
            for claim in updated_claims
            if claim.claim_id in target_ids
            and claim.claim_state is IntentClaimState.ACTIVE
            and claim.source is IntentSource.INFERRED
        ]
        if not confirmable:
            raise ValueError("confirmed decision requires an active inferred candidate")
        resolved_values = [claim.value for claim in confirmable]
        updated_claims = [
            replace(
                claim,
                source=IntentSource.EXPLICIT,
                origin=IntentOrigin.USER_CONFIRMATION,
                confidence=IntentConfidence.HIGH,
            )
            if claim in confirmable
            else claim
            for claim in updated_claims
        ]
    elif decision.action is IntentDecisionAction.CORRECTED:
        updated_claims = [
            replace(claim, claim_state=IntentClaimState.SUPERSEDED)
            if claim.claim_id in target_ids
            and claim.claim_state is IntentClaimState.ACTIVE
            else claim
            for claim in updated_claims
        ]
        resolved_values = list(decision.corrected_values)
        updated_claims.extend(
            _claim(
                question.field,
                value,
                IntentSource.EXPLICIT,
                IntentOrigin.USER_CORRECTION,
                IntentConfidence.HIGH,
                [f"clarification:{question.question_id}"],
            )
            for value in decision.corrected_values
        )
    elif decision.action is IntentDecisionAction.REJECTED:
        rejectable = [
            claim
            for claim in updated_claims
            if claim.claim_id in target_ids
            and claim.claim_state is IntentClaimState.ACTIVE
        ]
        if not rejectable:
            raise ValueError("rejected decision requires an active candidate")
        updated_claims = [
            replace(claim, claim_state=IntentClaimState.REJECTED)
            if claim in rejectable
            else claim
            for claim in updated_claims
        ]
    elif decision.action in {
        IntentDecisionAction.SKIPPED,
        IntentDecisionAction.SKIPPED_NON_INTERACTIVE,
    }:
        pass
    else:  # pragma: no cover - enum validation makes this defensive only
        raise ValueError(f"unsupported intent decision action: {decision.action}")

    terminal_status = ClarificationStatus(decision.action.value)
    updated_question = replace(
        question,
        status=terminal_status,
        user_response=decision.user_response,
        continuation_basis=decision.continuation_basis,
        resolved_values=resolved_values,
        decision_id=decision.decision_id,
    )
    updated_questions = [
        updated_question if item.question_id == question.question_id else item
        for item in questions
    ]
    return _deduplicate_claims(updated_claims), updated_questions


def compute_intent_status(
    claims: Sequence[IntentClaim],
    clarifications: Sequence[ClarificationQuestion] = (),
    *,
    sensitive_change: bool = False,
) -> IntentStatus:
    """Compute sufficiency solely from validated Runtime state."""
    _validate_claim_sequence(claims, "claims")
    _validate_question_sequence(clarifications)
    active = _active_claims(_deduplicate_claims(claims))
    by_field = {
        intent_field: [claim for claim in active if claim.field is intent_field]
        for intent_field in _FIELD_ORDER
    }

    goal_claims = by_field[IntentField.GOAL]
    if not goal_claims or len({claim.claim_id for claim in goal_claims}) > 1:
        return IntentStatus.INSUFFICIENT

    required_fields = [
        IntentField.GOAL,
        IntentField.ACCEPTANCE_CRITERIA,
        IntentField.SCOPE,
    ]
    if sensitive_change:
        required_fields.append(IntentField.CONSTRAINTS)

    all_required_explicit = all(
        by_field[intent_field]
        and all(
            claim.source is IntentSource.EXPLICIT
            for claim in by_field[intent_field]
        )
        for intent_field in required_fields
    )
    has_unresolved_question = any(
        question.status in {ClarificationStatus.PENDING, ClarificationStatus.OPEN}
        for question in clarifications
    )
    has_skipped_question = any(
        question.status
        in {
            ClarificationStatus.SKIPPED,
            ClarificationStatus.SKIPPED_NON_INTERACTIVE,
        }
        for question in clarifications
    )
    if all_required_explicit and not has_unresolved_question and not has_skipped_question:
        return IntentStatus.SUFFICIENT
    return IntentStatus.PARTIAL


def finalize_intent_packet(
    claims: Sequence[IntentClaim],
    clarifications: Sequence[ClarificationQuestion] = (),
    *,
    sensitive_change: bool = False,
    base_uncertainties: Sequence[str] = (),
) -> IntentPacket:
    """Reduce claims and clarification history into the compatible IntentPacket view."""
    _validate_claim_sequence(claims, "claims")
    _validate_question_sequence(clarifications)
    if any(not isinstance(item, str) or not item.strip() for item in base_uncertainties):
        raise ValueError("base_uncertainties must contain non-empty strings")

    provenance = _deduplicate_claims(claims)
    active = _active_claims(provenance)
    by_field = {
        intent_field: [claim for claim in active if claim.field is intent_field]
        for intent_field in _FIELD_ORDER
    }
    goal_claims = by_field[IntentField.GOAL]
    goal = goal_claims[0].value if goal_claims else None
    acceptance_criteria = [
        claim.value for claim in by_field[IntentField.ACCEPTANCE_CRITERIA]
    ]
    scope = [claim.value for claim in by_field[IntentField.SCOPE]]
    constraints = [claim.value for claim in by_field[IntentField.CONSTRAINTS]]

    sources: dict[str, IntentSource] = {}
    for intent_field in _FIELD_ORDER:
        field_claims = by_field[intent_field]
        if field_claims:
            sources[intent_field.value] = (
                IntentSource.EXPLICIT
                if all(claim.source is IntentSource.EXPLICIT for claim in field_claims)
                else IntentSource.INFERRED
            )

    uncertainties = list(base_uncertainties)
    if not goal_claims:
        uncertainties.append("goal is not declared")
    elif len({claim.claim_id for claim in goal_claims}) > 1:
        uncertainties.append(
            "conflicting goal claims remain unresolved: "
            + "; ".join(claim.value for claim in goal_claims)
        )
    elif goal_claims[0].source is IntentSource.INFERRED:
        if goal_claims[0].origin is IntentOrigin.CHANGED_FILES:
            uncertainties.append("user did not provide explicit intent")
        else:
            uncertainties.append("goal remains inferred")

    if not acceptance_criteria:
        uncertainties.append("acceptance criteria are not explicitly declared")
    elif sources[IntentField.ACCEPTANCE_CRITERIA.value] is IntentSource.INFERRED:
        uncertainties.append("acceptance criteria contain unconfirmed inferred values")

    if not scope:
        uncertainties.append("intended scope is not explicitly declared")
    elif sources[IntentField.SCOPE.value] is IntentSource.INFERRED:
        uncertainties.append("intended scope contains unconfirmed inferred values")

    if not constraints:
        uncertainties.append("project constraints are not explicitly declared")
    elif sources[IntentField.CONSTRAINTS.value] is IntentSource.INFERRED:
        uncertainties.append("constraints contain unconfirmed inferred values")

    for claim in provenance:
        if claim.claim_state is IntentClaimState.REJECTED:
            uncertainties.append(
                f"{claim.field.value} candidate was rejected: {claim.value}"
            )
        elif claim.claim_state is IntentClaimState.INVALID:
            uncertainties.append(
                f"{claim.field.value} candidate failed Runtime validation: {claim.value}"
            )

    for question in clarifications:
        if question.status in {ClarificationStatus.PENDING, ClarificationStatus.OPEN}:
            uncertainties.append(
                f"clarification remains open for {question.field.value}: "
                f"{question.rationale}"
            )
        elif question.status in {
            ClarificationStatus.SKIPPED,
            ClarificationStatus.SKIPPED_NON_INTERACTIVE,
        }:
            uncertainties.append(
                f"clarification skipped for {question.field.value}: "
                f"{question.continuation_basis}"
            )

    return IntentPacket(
        goal=goal,
        acceptance_criteria=acceptance_criteria,
        scope=scope,
        constraints=constraints,
        sources=sources,
        status=compute_intent_status(
            provenance,
            clarifications,
            sensitive_change=sensitive_change,
        ),
        uncertainties=_unique_values(uncertainties),
        provenance=provenance,
        clarifications=list(clarifications),
    )


def build_intent_packet(request: ReviewRequest, change_summary: ChangeSummary) -> IntentPacket:
    """Backward-compatible deterministic entry point used by the current pipeline."""
    claims = collect_deterministic_claims(request, change_summary)
    sensitive_change = is_sensitive_change(change_summary.changed_files)
    questions = generate_material_questions(
        claims,
        sensitive_change=sensitive_change,
    )
    base_uncertainties = (
        ["no changed files were detected"] if not change_summary.changed_files else []
    )
    return finalize_intent_packet(
        claims,
        questions,
        sensitive_change=sensitive_change,
        base_uncertainties=base_uncertainties,
    )


def is_sensitive_change(changed_files: Sequence[str]) -> bool:
    """Return whether changed paths require an explicit material constraint."""
    return any(
        marker in path.casefold()
        for path in changed_files
        for marker in _SENSITIVE_PATH_MARKERS
    )


def _claim(
    intent_field: IntentField,
    value: str,
    source: IntentSource,
    origin: IntentOrigin,
    confidence: IntentConfidence,
    source_refs: list[str],
) -> IntentClaim:
    return IntentClaim(
        field=intent_field,
        value=value.strip(),
        source=source,
        origin=origin,
        confidence=confidence,
        source_refs=source_refs,
    )


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _unique_clean_values(values: Sequence[str]) -> list[str]:
    return _unique_values(value.strip() for value in values if value.strip())


def _unique_values(values: Sequence[str] | Collection[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _validate_claim_sequence(claims: Sequence[IntentClaim], name: str) -> None:
    if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of IntentClaim")
    if any(not isinstance(claim, IntentClaim) for claim in claims):
        raise ValueError(f"{name} must contain only IntentClaim values")


def _validate_question_sequence(questions: Sequence[ClarificationQuestion]) -> None:
    if not isinstance(questions, Sequence) or isinstance(questions, (str, bytes)):
        raise ValueError("clarifications must be a sequence of ClarificationQuestion")
    if any(not isinstance(question, ClarificationQuestion) for question in questions):
        raise ValueError("clarifications must contain only ClarificationQuestion values")
    question_ids = [question.question_id for question in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("clarifications must not contain duplicate question IDs")


def _active_claims(claims: Sequence[IntentClaim]) -> list[IntentClaim]:
    return [
        claim for claim in claims if claim.claim_state is IntentClaimState.ACTIVE
    ]


def _deduplicate_claims(claims: Sequence[IntentClaim]) -> list[IntentClaim]:
    result: list[IntentClaim] = []
    positions: dict[str, int] = {}
    for claim in claims:
        position = positions.get(claim.claim_id)
        if position is None:
            positions[claim.claim_id] = len(result)
            result.append(claim)
            continue
        result[position] = _merge_duplicate_claim(result[position], claim)
    return result


def _merge_duplicate_claim(current: IntentClaim, candidate: IntentClaim) -> IntentClaim:
    state_rank = {
        IntentClaimState.ACTIVE: 3,
        IntentClaimState.REJECTED: 2,
        IntentClaimState.SUPERSEDED: 1,
        IntentClaimState.INVALID: 0,
    }
    source_rank = {IntentSource.EXPLICIT: 1, IntentSource.INFERRED: 0}
    confidence_rank = {
        IntentConfidence.HIGH: 2,
        IntentConfidence.MEDIUM: 1,
        IntentConfidence.LOW: 0,
    }
    current_rank = (state_rank[current.claim_state], source_rank[current.source])
    candidate_rank = (state_rank[candidate.claim_state], source_rank[candidate.source])
    base = candidate if candidate_rank > current_rank else current
    confidence = max(
        (current.confidence, candidate.confidence),
        key=confidence_rank.__getitem__,
    )
    impact_rank = {
        ConclusionImpact.BLOCKING: 2,
        ConclusionImpact.MATERIAL: 1,
        ConclusionImpact.SUPPLEMENTAL: 0,
    }
    impact = max(
        (current.conclusion_impact, candidate.conclusion_impact),
        key=impact_rank.__getitem__,
    )
    return replace(
        base,
        confidence=confidence,
        conclusion_impact=impact,
        source_refs=_unique_values([*current.source_refs, *candidate.source_refs]),
        evidence_refs=_unique_values([*current.evidence_refs, *candidate.evidence_refs]),
    )


def _question_text(
    intent_field: IntentField,
    reason: str,
    claims: Sequence[IntentClaim],
) -> str:
    labels = {
        IntentField.GOAL: "goal",
        IntentField.ACCEPTANCE_CRITERIA: "acceptance criteria",
        IntentField.SCOPE: "intended scope",
        IntentField.CONSTRAINTS: "required constraints",
    }
    label = labels[intent_field]
    if reason == "missing":
        return f"What {label} should this review use?"
    values = "; ".join(claim.value for claim in claims)
    if reason == "conflict":
        return f"Which {label} is authoritative: {values}?"
    return f"Should this review use the inferred {label}: {values}?"


def _question_rationale(intent_field: IntentField, reason: str) -> str:
    effects = {
        IntentField.GOAL: "It determines which behavior counts as correct.",
        IntentField.ACCEPTANCE_CRITERIA: "It determines which outcomes must be verified.",
        IntentField.SCOPE: "It determines which changed areas are intentional.",
        IntentField.CONSTRAINTS: "It can change compatibility, security, or data-safety conclusions.",
    }
    prefixes = {
        "missing": "This material intent field is missing. ",
        "inferred": "This material intent field is not yet user-confirmed. ",
        "conflict": "Multiple active values conflict. ",
    }
    return prefixes[reason] + effects[intent_field]

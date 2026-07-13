from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import PurePosixPath
from typing import Any

from review_agent.models import ContractAssessment, ReviewerFinding
from review_agent.observations import Observation
from review_agent.orchestrator import ReviewerExecution


FINDING_SEVERITIES = ("blocker", "high", "medium", "low")
FINDING_CONFIDENCES = ("high", "medium", "low")
FINDING_ORIGINS = ("initial", "supplemental")
FINDING_VALIDATION_STATUSES = ("supported", "rejected")
DETERMINISTIC_REJECTION_REASONS = ("unsupported_claim", "stale_evidence")
CONFLICT_HINT_KINDS = (
    "exact_duplicate",
    "same_location",
    "shared_evidence",
    "severity_mismatch",
    "location_mismatch",
)


@dataclass(frozen=True)
class FindingCandidate:
    """One immutable reviewer Finding before any semantic aggregation.

    ``finding_id`` deliberately excludes execution order and wall-clock state.  A
    candidate remains addressable even when deterministic validation rejects it;
    only supported candidates are eligible for semantic disposition.
    """

    finding_id: str
    origin: str
    reviewer_task_id: str
    reviewer_index: int | None
    role: str
    role_kind: str
    claim: str
    severity: str
    confidence: str
    path: str | None
    line: int | None
    impact: str
    suggested_action: str | None
    verification_performed: list[str]
    evidence_refs: list[str]
    validation_status: str
    deterministic_rejection_reason: str | None = None

    def __post_init__(self) -> None:
        for name, value in {
            "finding_id": self.finding_id,
            "origin": self.origin,
            "reviewer_task_id": self.reviewer_task_id,
            "role": self.role,
            "role_kind": self.role_kind,
            "claim": self.claim,
            "severity": self.severity,
            "confidence": self.confidence,
            "validation_status": self.validation_status,
        }.items():
            _require_non_empty_text(value, f"candidate.{name}")
        if not self.finding_id.startswith("F-"):
            raise ValueError("candidate.finding_id must start with F-")
        _require_enum(self.origin, set(FINDING_ORIGINS), "candidate.origin")
        _require_enum(self.severity, set(FINDING_SEVERITIES), "candidate.severity")
        _require_enum(
            self.confidence,
            set(FINDING_CONFIDENCES),
            "candidate.confidence",
        )
        _require_enum(
            self.validation_status,
            set(FINDING_VALIDATION_STATUSES),
            "candidate.validation_status",
        )
        if self.reviewer_index is not None and (
            type(self.reviewer_index) is not int or self.reviewer_index < 0
        ):
            raise ValueError("candidate.reviewer_index must be a non-negative integer or null")
        if self.path is not None:
            _require_non_empty_text(self.path, "candidate.path")
        if self.line is not None and (type(self.line) is not int or self.line < 1):
            raise ValueError("candidate.line must be a positive integer or null")
        if not isinstance(self.impact, str):
            raise ValueError("candidate.impact must be a string")
        if self.suggested_action is not None:
            _require_non_empty_text(
                self.suggested_action,
                "candidate.suggested_action",
            )
        _require_unique_text_list(
            self.verification_performed,
            "candidate.verification_performed",
            allow_empty=True,
        )
        _require_unique_text_list(
            self.evidence_refs,
            "candidate.evidence_refs",
            allow_empty=True,
        )
        if self.validation_status == "supported":
            if self.deterministic_rejection_reason is not None:
                raise ValueError("supported candidate cannot have a rejection reason")
            if not self.evidence_refs:
                raise ValueError("supported candidate must have evidence refs")
        else:
            _require_enum(
                self.deterministic_rejection_reason,
                set(DETERMINISTIC_REJECTION_REASONS),
                "candidate.deterministic_rejection_reason",
            )
        object.__setattr__(self, "verification_performed", list(self.verification_performed))
        object.__setattr__(self, "evidence_refs", list(self.evidence_refs))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConflictHint:
    """A deterministic relationship hint, never a semantic truth decision."""

    conflict_id: str
    candidate_ids: list[str]
    kind: str
    summary: str

    def __post_init__(self) -> None:
        _require_non_empty_text(self.conflict_id, "conflict_hint.conflict_id")
        if not self.conflict_id.startswith("C-"):
            raise ValueError("conflict_hint.conflict_id must start with C-")
        _require_enum(self.kind, set(CONFLICT_HINT_KINDS), "conflict_hint.kind")
        _require_unique_text_list(
            self.candidate_ids,
            "conflict_hint.candidate_ids",
            minimum=2,
        )
        if self.candidate_ids != sorted(self.candidate_ids):
            raise ValueError("conflict_hint.candidate_ids must use stable sorted order")
        _require_non_empty_text(self.summary, "conflict_hint.summary")
        object.__setattr__(self, "candidate_ids", list(self.candidate_ids))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalFinding:
    claim: str
    severity: str
    confidence: str
    evidence_refs: list[str]
    reviewer_indices: list[int]
    roles: list[str]
    suggested_action: str | None = None
    path: str | None = None
    line: int | None = None
    impact: str = ""
    verification_performed: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RejectedFinding:
    reviewer_index: int
    role: str
    claim: str
    reason: str
    evidence_refs: list[str]
    missing_evidence_refs: list[str]


@dataclass(frozen=True)
class ContractCoverage:
    reviewer_index: int
    role: str
    contract: str
    status: str
    summary: str
    evidence_refs: list[str]
    unsupported_evidence_refs: list[str]


@dataclass(frozen=True)
class EvidenceReconciliation:
    canonical_findings: list[CanonicalFinding]
    rejected_findings: list[RejectedFinding]
    remaining_disagreements: list[str]
    contract_coverage: list[ContractCoverage]
    evidence_quality: str


@dataclass(frozen=True)
class ReconciliationPrepass:
    review_id: str
    base_sha: str
    head_sha: str
    candidate_catalog: dict[str, FindingCandidate]
    conflict_hints: list[ConflictHint]
    rejected_findings: list[RejectedFinding]
    contract_coverage: list[ContractCoverage]
    evidence_quality: str
    schema_version: str = "reconciliation_prepass_v1"

    def __post_init__(self) -> None:
        for name, value in {
            "review_id": self.review_id,
            "base_sha": self.base_sha,
            "head_sha": self.head_sha,
            "schema_version": self.schema_version,
        }.items():
            _require_non_empty_text(value, f"prepass.{name}")
        if not isinstance(self.candidate_catalog, dict):
            raise ValueError("prepass.candidate_catalog must be an object")
        stable_catalog: dict[str, FindingCandidate] = {}
        for finding_id in sorted(self.candidate_catalog):
            candidate = self.candidate_catalog[finding_id]
            if not isinstance(candidate, FindingCandidate):
                raise ValueError("prepass.candidate_catalog values must be FindingCandidate")
            if finding_id != candidate.finding_id:
                raise ValueError("prepass candidate key must equal candidate.finding_id")
            stable_catalog[finding_id] = candidate
        if not isinstance(self.conflict_hints, list) or any(
            not isinstance(item, ConflictHint) for item in self.conflict_hints
        ):
            raise ValueError("prepass.conflict_hints must contain ConflictHint values")
        if not isinstance(self.rejected_findings, list) or any(
            not isinstance(item, RejectedFinding) for item in self.rejected_findings
        ):
            raise ValueError("prepass.rejected_findings must contain RejectedFinding values")
        if not isinstance(self.contract_coverage, list) or any(
            not isinstance(item, ContractCoverage) for item in self.contract_coverage
        ):
            raise ValueError("prepass.contract_coverage must contain ContractCoverage values")
        _require_enum(
            self.evidence_quality,
            {"verified", "mixed", "degraded"},
            "prepass.evidence_quality",
        )
        candidate_ids = set(stable_catalog)
        for hint in self.conflict_hints:
            if not set(hint.candidate_ids) <= candidate_ids:
                raise ValueError("conflict hint references an unknown candidate")
        object.__setattr__(self, "candidate_catalog", stable_catalog)
        object.__setattr__(self, "conflict_hints", list(self.conflict_hints))
        object.__setattr__(self, "rejected_findings", list(self.rejected_findings))
        object.__setattr__(self, "contract_coverage", list(self.contract_coverage))

    @property
    def revision_binding(self) -> dict[str, str]:
        return {"base_sha": self.base_sha, "head_sha": self.head_sha}

    @property
    def supported_candidates(self) -> list[FindingCandidate]:
        return [
            candidate
            for candidate in self.candidate_catalog.values()
            if candidate.validation_status == "supported"
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "review_id": self.review_id,
            "revision_binding": self.revision_binding,
            "candidate_catalog": {
                finding_id: candidate.to_dict()
                for finding_id, candidate in self.candidate_catalog.items()
            },
            "conflict_hints": [hint.to_dict() for hint in self.conflict_hints],
            "rejected_findings": [asdict(item) for item in self.rejected_findings],
            "contract_coverage": [asdict(item) for item in self.contract_coverage],
            "evidence_quality": self.evidence_quality,
        }


def reconcile_evidence(
    executions: list[ReviewerExecution],
    authorized_observation_ids: set[str],
) -> EvidenceReconciliation:
    canonical_by_key: dict[tuple[str, tuple[str, ...]], CanonicalFinding] = {}
    rejected: list[RejectedFinding] = []
    contract_coverage: list[ContractCoverage] = []

    for execution in executions:
        for finding in execution.result.confirmed_findings:
            missing_refs = _missing_refs(finding.evidence_refs, authorized_observation_ids)
            if not finding.evidence_refs or missing_refs:
                rejected.append(_rejected_finding(execution, finding, missing_refs))
                continue
            key = (_normalize_claim(finding.claim), tuple(sorted(finding.evidence_refs)))
            canonical_by_key[key] = _merge_canonical_finding(
                existing=canonical_by_key.get(key),
                execution=execution,
                finding=finding,
            )

        for assessment in execution.result.contract_assessments:
            contract_coverage.append(_contract_coverage(execution, assessment, authorized_observation_ids))

    return EvidenceReconciliation(
        canonical_findings=list(canonical_by_key.values()),
        rejected_findings=rejected,
        remaining_disagreements=[],
        contract_coverage=contract_coverage,
        evidence_quality="verified" if not rejected else "unsupported_claims",
    )


def reconciliation_to_dict(reconciliation: EvidenceReconciliation) -> dict[str, Any]:
    return asdict(reconciliation)


def build_reconciliation_prepass(
    executions: Sequence[ReviewerExecution],
    observations: Mapping[str, Observation | Mapping[str, Any]] | Iterable[Observation] | Collection[str],
    *,
    review_id: str,
    base_sha: str | None = None,
    head_sha: str | None = None,
    revision_binding: Mapping[str, str] | None = None,
    authorized_observation_ids: Collection[str] | None = None,
    origin: str = "initial",
    execution_metadata_by_trace_id: Mapping[str, Mapping[str, str]] | None = None,
) -> ReconciliationPrepass:
    """Build stable candidates and conservative deterministic conflict hints.

    Observation metadata is authoritative when supplied.  A plain collection of
    IDs remains supported for callers that have not yet hydrated Observation
    records; such callers receive ID validation but cannot claim revision or
    location validation.
    """

    _require_non_empty_text(review_id, "review_id")
    _require_enum(origin, set(FINDING_ORIGINS), "origin")
    execution_metadata = _normalize_execution_metadata(
        executions,
        execution_metadata_by_trace_id,
    )
    base_sha, head_sha = _resolved_revision_binding(
        base_sha,
        head_sha,
        revision_binding,
    )
    observation_catalog = _normalize_observation_catalog(observations)
    authorized_ids = (
        set(observation_catalog)
        if authorized_observation_ids is None
        else _normalized_ref_allowlist(authorized_observation_ids)
    )
    observation_catalog = {
        observation_id: observation_catalog[observation_id]
        for observation_id in sorted(observation_catalog)
        if observation_id in authorized_ids
    }
    for observation_id in sorted(authorized_ids):
        observation_catalog.setdefault(observation_id, None)

    candidate_catalog: dict[str, FindingCandidate] = {}
    rejected: list[RejectedFinding] = []
    contract_coverage: list[ContractCoverage] = []
    allowed_revisions = _allowed_observation_revisions(base_sha, head_sha)

    for execution in executions:
        if not isinstance(execution, ReviewerExecution):
            raise ValueError("executions must contain ReviewerExecution values")
        metadata = execution_metadata.get(execution.trace_id)
        execution_origin = metadata["origin"] if metadata is not None else origin
        task_id = (
            metadata["task_id"]
            if metadata is not None
            else _reviewer_task_id(execution)
        )
        for finding in execution.result.confirmed_findings:
            candidate, missing_refs = _finding_candidate(
                review_id=review_id,
                base_sha=base_sha,
                head_sha=head_sha,
                origin=execution_origin,
                task_id=task_id,
                execution=execution,
                finding=finding,
                authorized_ids=authorized_ids,
                observation_catalog=observation_catalog,
                allowed_revisions=allowed_revisions,
            )
            existing = candidate_catalog.get(candidate.finding_id)
            if existing is not None:
                if existing != candidate:
                    raise ValueError(
                        "stable finding ID collision with different candidate content: "
                        f"{candidate.finding_id}"
                    )
                continue
            candidate_catalog[candidate.finding_id] = candidate
            if candidate.validation_status == "rejected":
                rejected.append(
                    RejectedFinding(
                        reviewer_index=execution.reviewer_index,
                        role=execution.assignment.role,
                        claim=candidate.claim,
                        reason=candidate.deterministic_rejection_reason
                        or "unsupported_claim",
                        evidence_refs=list(candidate.evidence_refs),
                        missing_evidence_refs=(
                            missing_refs
                            if missing_refs
                            else list(candidate.evidence_refs)
                        ),
                    )
                )

        for assessment in execution.result.contract_assessments:
            contract_coverage.append(
                _contract_coverage(execution, assessment, authorized_ids)
            )

    candidate_catalog = {
        finding_id: candidate_catalog[finding_id]
        for finding_id in sorted(candidate_catalog)
    }
    supported = [
        candidate
        for candidate in candidate_catalog.values()
        if candidate.validation_status == "supported"
    ]
    conflict_hints = _build_conflict_hints(supported)
    if not rejected:
        evidence_quality = "verified"
    elif supported:
        evidence_quality = "mixed"
    else:
        evidence_quality = "degraded"
    return ReconciliationPrepass(
        review_id=review_id,
        base_sha=base_sha,
        head_sha=head_sha,
        candidate_catalog=candidate_catalog,
        conflict_hints=conflict_hints,
        rejected_findings=rejected,
        contract_coverage=contract_coverage,
        evidence_quality=evidence_quality,
    )


def deterministic_reconciliation_prepass(
    executions: Sequence[ReviewerExecution],
    observations: Mapping[str, Observation | Mapping[str, Any]] | Iterable[Observation] | Collection[str],
    **kwargs: Any,
) -> ReconciliationPrepass:
    """Compatibility spelling for the deterministic pre-pass entry point."""

    return build_reconciliation_prepass(executions, observations, **kwargs)


def reconciliation_prepass_to_dict(prepass: ReconciliationPrepass) -> dict[str, Any]:
    if not isinstance(prepass, ReconciliationPrepass):
        raise ValueError("prepass must be a ReconciliationPrepass")
    return prepass.to_dict()


def _missing_refs(evidence_refs: list[str], authorized_observation_ids: set[str]) -> list[str]:
    return [ref for ref in evidence_refs if ref not in authorized_observation_ids]


def _rejected_finding(
    execution: ReviewerExecution,
    finding: ReviewerFinding,
    missing_refs: list[str],
) -> RejectedFinding:
    return RejectedFinding(
        reviewer_index=execution.reviewer_index,
        role=execution.assignment.role,
        claim=finding.claim,
        reason="unsupported_claim",
        evidence_refs=list(finding.evidence_refs),
        missing_evidence_refs=missing_refs or list(finding.evidence_refs),
    )


def _merge_canonical_finding(
    existing: CanonicalFinding | None,
    execution: ReviewerExecution,
    finding: ReviewerFinding,
) -> CanonicalFinding:
    if existing is None:
        return CanonicalFinding(
            claim=finding.claim.strip(),
            severity=finding.severity,
            confidence=finding.confidence,
            evidence_refs=list(finding.evidence_refs),
            reviewer_indices=[execution.reviewer_index],
            roles=[execution.assignment.role],
            suggested_action=finding.suggested_action,
            path=finding.path,
            line=finding.line,
            impact=finding.impact,
            verification_performed=list(finding.verification_performed),
        )
    return CanonicalFinding(
        claim=existing.claim,
        severity=existing.severity,
        confidence=existing.confidence,
        evidence_refs=existing.evidence_refs,
        reviewer_indices=[*existing.reviewer_indices, execution.reviewer_index],
        roles=[*existing.roles, execution.assignment.role],
        suggested_action=existing.suggested_action,
        path=existing.path,
        line=existing.line,
        impact=existing.impact,
        verification_performed=list(existing.verification_performed or []),
    )


def _contract_coverage(
    execution: ReviewerExecution,
    assessment: ContractAssessment,
    authorized_observation_ids: set[str],
) -> ContractCoverage:
    return ContractCoverage(
        reviewer_index=execution.reviewer_index,
        role=execution.assignment.role,
        contract=assessment.contract,
        status=assessment.status.value,
        summary=assessment.summary,
        evidence_refs=list(assessment.evidence_refs),
        unsupported_evidence_refs=_missing_refs(assessment.evidence_refs, authorized_observation_ids),
    )


def _normalize_claim(claim: str) -> str:
    return " ".join(claim.casefold().split())


def _resolved_revision_binding(
    base_sha: str | None,
    head_sha: str | None,
    revision_binding: Mapping[str, str] | None,
) -> tuple[str, str]:
    if revision_binding is not None:
        if not isinstance(revision_binding, Mapping):
            raise ValueError("revision_binding must be an object")
        allowed_fields = {"base_sha", "head_sha"}
        if set(revision_binding) != allowed_fields:
            raise ValueError("revision_binding must contain exactly base_sha and head_sha")
        bound_base = revision_binding["base_sha"]
        bound_head = revision_binding["head_sha"]
        if base_sha is not None and base_sha != bound_base:
            raise ValueError("base_sha conflicts with revision_binding")
        if head_sha is not None and head_sha != bound_head:
            raise ValueError("head_sha conflicts with revision_binding")
        base_sha = bound_base
        head_sha = bound_head
    _require_non_empty_text(base_sha, "base_sha")
    _require_non_empty_text(head_sha, "head_sha")
    return base_sha, head_sha


def _normalize_observation_catalog(
    observations: Mapping[str, Observation | Mapping[str, Any]] | Iterable[Observation] | Collection[str],
) -> dict[str, Observation | Mapping[str, Any] | None]:
    if isinstance(observations, (str, bytes)):
        raise ValueError("observations must not be a string")
    if isinstance(observations, Mapping):
        catalog: dict[str, Observation | Mapping[str, Any] | None] = {}
        for raw_id, value in observations.items():
            observation_id = _require_non_empty_text(raw_id, "observation ID")
            if value is not None and not isinstance(value, (Observation, Mapping)):
                raise ValueError("observation catalog values must be Observation, object, or null")
            embedded_id = _observation_value(value, "observation_id")
            if embedded_id is not None and embedded_id != observation_id:
                raise ValueError("observation catalog key does not match observation_id")
            catalog[observation_id] = value
        return catalog
    try:
        values = list(observations)
    except TypeError as error:
        raise ValueError("observations must be a mapping or iterable") from error
    catalog = {}
    for value in values:
        if isinstance(value, str):
            observation_id = _require_non_empty_text(value, "observation ID")
            catalog[observation_id] = None
            continue
        if not isinstance(value, Observation):
            raise ValueError("observation iterable must contain Observation values or IDs")
        observation_id = _require_non_empty_text(
            value.observation_id,
            "observation.observation_id",
        )
        existing = catalog.get(observation_id)
        if existing is not None and existing != value:
            raise ValueError(f"duplicate observation ID has different metadata: {observation_id}")
        catalog[observation_id] = value
    return catalog


def _normalized_ref_allowlist(values: Collection[str]) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError("authorized_observation_ids must not be a string")
    return {
        _require_non_empty_text(value, "authorized observation ID")
        for value in values
    }


def _reviewer_task_id(execution: ReviewerExecution) -> str:
    assignment_id = getattr(execution.assignment, "assignment_id", "")
    task_id = assignment_id or execution.trace_id
    return _require_non_empty_text(task_id, "reviewer task ID")


def _normalize_execution_metadata(
    executions: Sequence[ReviewerExecution],
    value: Mapping[str, Mapping[str, str]] | None,
) -> dict[str, dict[str, str]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("execution_metadata_by_trace_id must be an object")
    trace_ids = {
        execution.trace_id
        for execution in executions
        if isinstance(execution, ReviewerExecution)
    }
    unknown = set(value) - trace_ids
    if unknown:
        raise ValueError(
            "execution metadata references unknown trace IDs: "
            + ", ".join(sorted(unknown))
        )
    normalized: dict[str, dict[str, str]] = {}
    for trace_id, raw in value.items():
        _require_non_empty_text(trace_id, "execution metadata trace ID")
        if not isinstance(raw, Mapping) or set(raw) != {"origin", "task_id"}:
            raise ValueError(
                "execution metadata must contain exactly origin and task_id"
            )
        normalized[trace_id] = {
            "origin": _require_enum(
                raw["origin"],
                set(FINDING_ORIGINS),
                f"execution metadata {trace_id}.origin",
            ),
            "task_id": _require_non_empty_text(
                raw["task_id"],
                f"execution metadata {trace_id}.task_id",
            ),
        }
    return normalized


def _finding_candidate(
    *,
    review_id: str,
    base_sha: str,
    head_sha: str,
    origin: str,
    task_id: str,
    execution: ReviewerExecution,
    finding: ReviewerFinding,
    authorized_ids: set[str],
    observation_catalog: Mapping[str, Observation | Mapping[str, Any] | None],
    allowed_revisions: set[str],
) -> tuple[FindingCandidate, list[str]]:
    if not isinstance(finding, ReviewerFinding):
        raise ValueError("confirmed_findings must contain ReviewerFinding values")
    claim = _require_non_empty_text(finding.claim, "finding.claim").strip()
    severity = _normalized_enum(finding.severity, set(FINDING_SEVERITIES), "finding.severity")
    confidence = _normalized_enum(
        finding.confidence,
        set(FINDING_CONFIDENCES),
        "finding.confidence",
    )
    path = _normalize_candidate_path(finding.path)
    line = finding.line
    if line is not None and (type(line) is not int or line < 1):
        path_is_valid = False
    else:
        path_is_valid = not (line is not None and path is None)
    impact = _require_string(finding.impact, "finding.impact").strip()
    suggested_action = (
        _require_non_empty_text(
            finding.suggested_action,
            "finding.suggested_action",
        ).strip()
        if finding.suggested_action is not None
        else None
    )
    verification_performed = _normalized_text_values(
        finding.verification_performed,
        "finding.verification_performed",
    )
    evidence_refs = _normalized_text_values(
        finding.evidence_refs,
        "finding.evidence_refs",
    )
    validation_status, rejection_reason, missing_refs = _validate_finding_authority(
        evidence_refs=evidence_refs,
        path=path,
        line=line,
        path_is_valid=path_is_valid,
        authorized_ids=authorized_ids,
        observation_catalog=observation_catalog,
        allowed_revisions=allowed_revisions,
    )
    role = _require_non_empty_text(execution.assignment.role, "assignment.role").strip()
    role_kind = _require_non_empty_text(
        getattr(execution.assignment, "role_kind", "legacy"),
        "assignment.role_kind",
    ).strip()
    finding_id = _stable_finding_id(
        review_id=review_id,
        base_sha=base_sha,
        head_sha=head_sha,
        origin=origin,
        task_id=task_id,
        role=role,
        role_kind=role_kind,
        claim=claim,
        severity=severity,
        confidence=confidence,
        path=path,
        line=line,
        impact=impact,
        suggested_action=suggested_action,
        verification_performed=verification_performed,
        evidence_refs=evidence_refs,
    )
    return (
        FindingCandidate(
            finding_id=finding_id,
            origin=origin,
            reviewer_task_id=task_id,
            reviewer_index=execution.reviewer_index,
            role=role,
            role_kind=role_kind,
            claim=claim,
            severity=severity,
            confidence=confidence,
            path=path,
            line=line if path_is_valid else None,
            impact=impact,
            suggested_action=suggested_action,
            verification_performed=verification_performed,
            evidence_refs=evidence_refs,
            validation_status=validation_status,
            deterministic_rejection_reason=rejection_reason,
        ),
        missing_refs,
    )


def _validate_finding_authority(
    *,
    evidence_refs: list[str],
    path: str | None,
    line: int | None,
    path_is_valid: bool,
    authorized_ids: set[str],
    observation_catalog: Mapping[str, Observation | Mapping[str, Any] | None],
    allowed_revisions: set[str],
) -> tuple[str, str | None, list[str]]:
    if not evidence_refs:
        return "rejected", "unsupported_claim", []
    missing_refs = [ref for ref in evidence_refs if ref not in authorized_ids]
    if missing_refs:
        return "rejected", "unsupported_claim", missing_refs
    if not path_is_valid:
        return "rejected", "unsupported_claim", []

    hydrated = [observation_catalog.get(ref) for ref in evidence_refs]
    for observation in hydrated:
        if observation is None:
            continue
        revision = _observation_value(observation, "revision")
        if not isinstance(revision, str) or revision.casefold() not in allowed_revisions:
            return "rejected", "stale_evidence", []

    if path is None:
        return "supported", None, []
    metadata = [item for item in hydrated if item is not None]
    if not metadata:
        # ID-only compatibility callers cannot make a metadata authority claim.
        return "supported", None, []
    same_path = [
        item
        for item in metadata
        if _normalize_observation_path(_observation_value(item, "path")) == path
    ]
    if not same_path:
        return "rejected", "unsupported_claim", []
    if line is None:
        return "supported", None, []
    for observation in same_path:
        line_start = _observation_value(observation, "line_start")
        line_end = _observation_value(observation, "line_end")
        if line_start is None and line_end is None:
            return "supported", None, []
        if (
            type(line_start) is int
            and type(line_end) is int
            and line_start <= line <= line_end
        ):
            return "supported", None, []
    return "rejected", "unsupported_claim", []


def _stable_finding_id(**identity: Any) -> str:
    normalized = {
        "review_id": identity["review_id"],
        "revision_binding": {
            "base_sha": identity["base_sha"],
            "head_sha": identity["head_sha"],
        },
        "origin": identity["origin"],
        "reviewer_task_id": identity["task_id"],
        "role": _normalize_text(identity["role"]),
        "role_kind": _normalize_text(identity["role_kind"]),
        "finding": {
            "claim": _normalize_claim(identity["claim"]),
            "severity": identity["severity"],
            "confidence": identity["confidence"],
            "path": identity["path"],
            "line": identity["line"],
            "impact": _normalize_text(identity["impact"]),
            "suggested_action": (
                _normalize_text(identity["suggested_action"])
                if identity["suggested_action"] is not None
                else None
            ),
            "verification_performed": sorted(
                _normalize_text(item) for item in identity["verification_performed"]
            ),
            "evidence_refs": sorted(identity["evidence_refs"]),
        },
    }
    digest = hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()
    return f"F-{digest[:32]}"


def _allowed_observation_revisions(base_sha: str, head_sha: str) -> set[str]:
    values = {
        base_sha,
        head_sha,
        f"base@{base_sha}",
        f"head@{head_sha}",
        f"{base_sha}..{head_sha}",
    }
    return {value.casefold() for value in values}


def _build_conflict_hints(candidates: Sequence[FindingCandidate]) -> list[ConflictHint]:
    candidate_by_id = {candidate.finding_id: candidate for candidate in candidates}
    hints: dict[tuple[str, tuple[str, ...]], ConflictHint] = {}

    def add(kind: str, candidate_ids: Iterable[str], summary: str) -> None:
        stable_ids = tuple(sorted(set(candidate_ids)))
        if len(stable_ids) < 2:
            return
        key = (kind, stable_ids)
        if key in hints:
            return
        digest = hashlib.sha256(
            _canonical_json({"kind": kind, "candidate_ids": stable_ids}).encode("utf-8")
        ).hexdigest()
        hints[key] = ConflictHint(
            conflict_id=f"C-{digest[:32]}",
            candidate_ids=list(stable_ids),
            kind=kind,
            summary=summary,
        )

    exact_groups: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    location_groups: dict[tuple[str, int | None], list[str]] = {}
    evidence_groups: dict[str, list[str]] = {}
    claim_groups: dict[str, list[str]] = {}
    for candidate in candidates:
        exact_groups.setdefault(
            (_normalize_claim(candidate.claim), tuple(sorted(candidate.evidence_refs))),
            [],
        ).append(candidate.finding_id)
        if candidate.path is not None:
            location_groups.setdefault((candidate.path, candidate.line), []).append(
                candidate.finding_id
            )
        for ref in candidate.evidence_refs:
            evidence_groups.setdefault(ref, []).append(candidate.finding_id)
        claim_groups.setdefault(_normalize_claim(candidate.claim), []).append(
            candidate.finding_id
        )

    for ids in exact_groups.values():
        add("exact_duplicate", ids, "Candidates have the same normalized claim and evidence refs.")
    for location, ids in location_groups.items():
        claims = {_normalize_claim(candidate_by_id[item].claim) for item in ids}
        if len(claims) > 1:
            add(
                "same_location",
                ids,
                f"Candidates report different claims at {location[0]}:{location[1]}.",
            )
    for ref, ids in evidence_groups.items():
        if len(set(ids)) > 1:
            add("shared_evidence", ids, f"Candidates cite the same Observation {ref}.")
    for normalized_claim, ids in claim_groups.items():
        if len(set(ids)) < 2:
            continue
        severities = {candidate_by_id[item].severity for item in ids}
        if len(severities) > 1:
            add(
                "severity_mismatch",
                ids,
                f"Candidates disagree on severity for normalized claim {normalized_claim!r}.",
            )
        locations = {
            (candidate_by_id[item].path, candidate_by_id[item].line)
            for item in ids
        }
        if len(locations) > 1:
            add(
                "location_mismatch",
                ids,
                f"Candidates disagree on location for normalized claim {normalized_claim!r}.",
            )

    kind_order = {kind: index for index, kind in enumerate(CONFLICT_HINT_KINDS)}
    return sorted(
        hints.values(),
        key=lambda item: (kind_order[item.kind], item.candidate_ids, item.conflict_id),
    )


def _normalize_candidate_path(value: object) -> str | None:
    if value is None:
        return None
    path = _require_non_empty_text(value, "finding.path").strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or ":" in pure.parts[0]:
        # Invalid reviewer-provided locations are retained as unsupported candidates.
        return None
    normalized = pure.as_posix()
    return normalized if normalized not in {"", "."} else None


def _normalize_observation_path(value: object) -> str | None:
    if value is None or not isinstance(value, str) or not value.strip():
        return None
    path = value.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return PurePosixPath(path).as_posix()


def _observation_value(
    observation: Observation | Mapping[str, Any] | None,
    field_name: str,
) -> Any:
    if observation is None:
        return None
    if isinstance(observation, Mapping):
        return observation.get(field_name)
    return getattr(observation, field_name, None)


def _normalized_text_values(values: object, context: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{context} must be a list")
    normalized = {
        _require_non_empty_text(value, f"{context} item").strip()
        for value in values
    }
    return sorted(normalized)


def _normalized_enum(value: object, choices: set[str], context: str) -> str:
    text = _require_non_empty_text(value, context).strip().casefold()
    _require_enum(text, choices, context)
    return text


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _require_non_empty_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    return value


def _require_enum(value: object, choices: set[str], context: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{context} must be one of: {', '.join(sorted(choices))}")
    return value


def _require_unique_text_list(
    value: object,
    context: str,
    *,
    minimum: int = 0,
    allow_empty: bool = False,
) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    if len(value) < minimum:
        raise ValueError(f"{context} must contain at least {minimum} items")
    seen: set[str] = set()
    for index, item in enumerate(value):
        if allow_empty and isinstance(item, str) and not item.strip():
            raise ValueError(f"{context}[{index}] must be a non-empty string")
        text = _require_non_empty_text(item, f"{context}[{index}]")
        if text in seen:
            raise ValueError(f"{context} must not contain duplicate values")
        seen.add(text)

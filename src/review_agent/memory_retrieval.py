"""Deterministic durable-memory retrieval and pinned Snapshot queries.

Target-revision authority is deliberately delegated to
``TargetHeadApplicabilityEvaluator``.  This module owns only the ordered
repository/status/stage/scope/ranking/budget gates and immutable Snapshot
projection.  Query services re-hydrate and retain a Snapshot copy; they never
accept a live persistence object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
import fnmatch
import re
import unicodedata
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from review_agent.memory_lifecycle import (
    ApplicabilityDecision,
    TargetHeadApplicabilityEvaluator,
)
from review_agent.memory_models import (
    MAX_SNAPSHOT_DECISIONS,
    MAX_SNAPSHOT_RECORDS,
    Applicability,
    DurableMemoryRecord,
    FeedbackCalibrationSummary,
    MemoryExecutionConfig,
    MemoryKind,
    MemoryScope,
    MemorySelectionDecision,
    MemorySelectionInput,
    MemorySnapshot,
    PolicyEffectKind,
    RecordStatus,
    canonical_json,
    validate_stable_id,
)


MAX_RETRIEVAL_TEXT_CHARS = 4_096
MAX_RETRIEVAL_BYTES = 8_388_608
MAX_QUERY_CALLS = 128
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@#-]{0,511}$")


class MemoryRetrievalError(RuntimeError):
    """Base class for deterministic retrieval failures."""


class RetrievalInputError(MemoryRetrievalError):
    """The caller supplied a non-canonical or inconsistent input."""


class SemanticRankerViolation(MemoryRetrievalError):
    """A semantic ranker attempted to escape the eligible record set."""


class SnapshotBudgetExceeded(MemoryRetrievalError):
    """Required Snapshot metadata cannot fit its configured byte budget."""


class ProjectionBudgetExceeded(MemoryRetrievalError):
    """A context or query projection cannot fit its configured byte budget."""

    def __init__(self, *, boundary: str, limit: int, required: int) -> None:
        self.boundary = boundary
        self.limit = limit
        self.required = required
        super().__init__(
            "%s projection byte budget exceeded: required %d, limit %d"
            % (boundary, required, limit)
        )


class HardPolicyBudgetExceeded(MemoryRetrievalError):
    """Applicable typed policy records cannot be silently compacted."""

    blocking = True

    def __init__(
        self,
        *,
        boundary: str,
        budget: str,
        limit: int,
        required: int,
        memory_ids: Sequence[str],
    ) -> None:
        self.boundary = boundary
        self.budget = budget
        self.limit = limit
        self.required = required
        self.memory_ids = tuple(sorted(memory_ids))
        super().__init__(
            "hard-policy %s %s budget exceeded: required %d, limit %d"
            % (boundary, budget, required, limit)
        )


class QueryLimitExceeded(MemoryRetrievalError):
    """A bounded Snapshot query limit was exceeded."""


class QueryScopeViolation(MemoryRetrievalError):
    """A query escaped its bound Assignment scope."""


class RetrievalStage(str, Enum):
    MEMORY_SELECTION = "memory_selection"
    INTENT_DISCOVERY = "intent_discovery"
    INITIAL_RISK = "initial_risk"
    PORTFOLIO_PLANNING = "portfolio_planning"
    REVIEWER = "reviewer"
    RECONCILER = "reconciler"
    COMPLETION = "completion"
    FINAL_RISK = "final_risk"


_ALL_KINDS = frozenset(MemoryKind)
_STAGE_KINDS = {
    RetrievalStage.MEMORY_SELECTION: _ALL_KINDS,
    RetrievalStage.INTENT_DISCOVERY: frozenset(
        {
            MemoryKind.ARCHITECTURE_BOUNDARY,
            MemoryKind.BUSINESS_INVARIANT,
            MemoryKind.COMPATIBILITY_REQUIREMENT,
        }
    ),
    RetrievalStage.INITIAL_RISK: frozenset(
        {MemoryKind.HIGH_RISK_MODULE, MemoryKind.INCIDENT_LESSON}
    ),
    RetrievalStage.PORTFOLIO_PLANNING: frozenset(
        {MemoryKind.REVIEW_RULE, MemoryKind.VERIFICATION_COMMAND}
    ),
    RetrievalStage.REVIEWER: _ALL_KINDS,
    RetrievalStage.RECONCILER: frozenset(
        {
            MemoryKind.ARCHITECTURE_BOUNDARY,
            MemoryKind.BUSINESS_INVARIANT,
            MemoryKind.COMPATIBILITY_REQUIREMENT,
        }
    ),
    RetrievalStage.COMPLETION: frozenset(),
    RetrievalStage.FINAL_RISK: frozenset(
        {MemoryKind.HIGH_RISK_MODULE, MemoryKind.INCIDENT_LESSON}
    ),
}
_STAGE_EFFECTS = {
    RetrievalStage.INITIAL_RISK: frozenset({PolicyEffectKind.RISK_FLOOR}),
    RetrievalStage.PORTFOLIO_PLANNING: frozenset(
        {
            PolicyEffectKind.REQUIRE_CONTRACT,
            PolicyEffectKind.REQUIRE_CHECK,
            PolicyEffectKind.VERIFICATION_HINT,
        }
    ),
    RetrievalStage.COMPLETION: frozenset(
        {PolicyEffectKind.REQUIRE_CONTRACT, PolicyEffectKind.REQUIRE_CHECK}
    ),
    RetrievalStage.FINAL_RISK: frozenset({PolicyEffectKind.RISK_FLOOR}),
}
_EFFECT_PRIORITY = {
    PolicyEffectKind.RISK_FLOOR: 0,
    PolicyEffectKind.REQUIRE_CONTRACT: 1,
    PolicyEffectKind.REQUIRE_CHECK: 2,
    PolicyEffectKind.VERIFICATION_HINT: 3,
}


@dataclass(frozen=True)
class RetrievalLimits:
    max_snapshot_records: int = MAX_SNAPSHOT_RECORDS
    max_snapshot_bytes: int = MAX_RETRIEVAL_BYTES
    max_context_records: int = 12
    max_context_bytes: int = 262_144
    max_query_results: int = 8
    max_query_bytes: int = 131_072
    max_query_calls: int = 16
    max_query_text_chars: int = 2_048
    per_kind_limits: Tuple[Tuple[MemoryKind, int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        bounds = {
            "max_snapshot_records": (1, MAX_SNAPSHOT_RECORDS),
            "max_snapshot_bytes": (1, MAX_RETRIEVAL_BYTES),
            "max_context_records": (1, 12),
            "max_context_bytes": (1, MAX_RETRIEVAL_BYTES),
            "max_query_results": (1, 8),
            "max_query_bytes": (1, MAX_RETRIEVAL_BYTES),
            "max_query_calls": (1, MAX_QUERY_CALLS),
            "max_query_text_chars": (1, MAX_RETRIEVAL_TEXT_CHARS),
        }
        for name, (minimum, maximum) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError("%s is outside its supported range" % name)
        raw_limits: Any = self.per_kind_limits
        if isinstance(raw_limits, Mapping):
            raw_limits = tuple(raw_limits.items())
        if not isinstance(raw_limits, (list, tuple)):
            raise ValueError("per_kind_limits must be a mapping or tuple")
        canonical: Dict[MemoryKind, int] = {}
        for item in raw_limits:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("per_kind_limits entries must be pairs")
            kind, limit = item
            if not isinstance(kind, MemoryKind) or type(limit) is not int or limit < 1:
                raise ValueError("per_kind_limits entries are invalid")
            canonical[kind] = limit
        object.__setattr__(
            self,
            "per_kind_limits",
            tuple((kind, canonical[kind]) for kind in sorted(canonical, key=lambda x: x.value)),
        )

    @classmethod
    def from_execution_config(
        cls,
        config: MemoryExecutionConfig,
        *,
        max_context_bytes: int = 262_144,
        max_query_bytes: int = 131_072,
        max_query_calls: int = 16,
        max_query_text_chars: int = 2_048,
        per_kind_limits: Mapping[MemoryKind, int] | Sequence[Tuple[MemoryKind, int]] = (),
    ) -> "RetrievalLimits":
        if type(config) is not MemoryExecutionConfig:
            raise ValueError("config must be a canonical MemoryExecutionConfig")
        return cls(
            max_snapshot_records=config.max_snapshot_records,
            max_snapshot_bytes=config.max_snapshot_bytes,
            max_context_records=config.max_context_records,
            max_context_bytes=max_context_bytes,
            max_query_results=config.max_query_results,
            max_query_bytes=max_query_bytes,
            max_query_calls=max_query_calls,
            max_query_text_chars=max_query_text_chars,
            per_kind_limits=tuple(per_kind_limits.items())
            if isinstance(per_kind_limits, Mapping)
            else tuple(per_kind_limits),
        )

    def kind_limit(self, kind: MemoryKind) -> Optional[int]:
        return dict(self.per_kind_limits).get(kind)


@dataclass(frozen=True)
class RetrievalRequest:
    stage: RetrievalStage
    paths: Tuple[str, ...] = field(default_factory=tuple)
    symbols: Tuple[str, ...] = field(default_factory=tuple)
    contracts: Tuple[str, ...] = field(default_factory=tuple)
    languages: Tuple[str, ...] = field(default_factory=tuple)
    query_text: str = ""
    graph_relevance: Tuple[Tuple[str, int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        stage = self.stage
        if isinstance(stage, str) and not isinstance(stage, RetrievalStage):
            aliases = {
                "planning": RetrievalStage.PORTFOLIO_PLANNING,
                "reviewers": RetrievalStage.REVIEWER,
                "reconciliation": RetrievalStage.RECONCILER,
            }
            try:
                stage = aliases[stage] if stage in aliases else RetrievalStage(stage)
            except ValueError:
                raise ValueError("stage is unsupported") from None
            object.__setattr__(self, "stage", stage)
        if not isinstance(stage, RetrievalStage):
            raise ValueError("stage must be a RetrievalStage")
        scope = MemoryScope(
            paths=tuple(self.paths),
            symbols=tuple(self.symbols),
            contracts=tuple(self.contracts),
            languages=tuple(self.languages),
        )
        object.__setattr__(self, "paths", scope.paths)
        object.__setattr__(self, "symbols", scope.symbols)
        object.__setattr__(self, "contracts", scope.contracts)
        object.__setattr__(self, "languages", scope.languages)
        text = _bounded_text(self.query_text, "query_text", MAX_RETRIEVAL_TEXT_CHARS, empty=True)
        object.__setattr__(self, "query_text", text)
        raw_graph: Any = self.graph_relevance
        if isinstance(raw_graph, Mapping):
            raw_graph = tuple(raw_graph.items())
        if not isinstance(raw_graph, (list, tuple)) or len(raw_graph) > MAX_SNAPSHOT_RECORDS:
            raise ValueError("graph_relevance is invalid")
        graph: Dict[str, int] = {}
        for item in raw_graph:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("graph_relevance entries must be pairs")
            memory_id, score = item
            try:
                canonical_memory_id = validate_stable_id(
                    memory_id,
                    "MEM",
                    "graph_relevance memory_id",
                )
            except (TypeError, ValueError):
                raise ValueError("graph_relevance entry is invalid") from None
            if type(score) is not int or not 0 <= score <= 1_000_000:
                raise ValueError("graph_relevance entry is invalid")
            graph[canonical_memory_id] = score
        object.__setattr__(self, "graph_relevance", tuple(sorted(graph.items())))

    @property
    def scope(self) -> MemoryScope:
        return MemoryScope(
            paths=self.paths,
            symbols=self.symbols,
            contracts=self.contracts,
            languages=self.languages,
        )

    @classmethod
    def from_selection_input(
        cls,
        selection_input: MemorySelectionInput,
        *,
        stage: RetrievalStage,
        query_text: str = "",
        graph_relevance: Mapping[str, int] | Sequence[Tuple[str, int]] = (),
    ) -> "RetrievalRequest":
        return cls(
            stage=stage,
            paths=selection_input.changed_paths,
            symbols=selection_input.changed_symbols,
            contracts=selection_input.contracts,
            languages=selection_input.languages,
            query_text=query_text,
            graph_relevance=tuple(graph_relevance.items())
            if isinstance(graph_relevance, Mapping)
            else tuple(graph_relevance),
        )


SemanticRanker = Callable[
    [Tuple[DurableMemoryRecord, ...], RetrievalRequest], Mapping[str, Any]
]


@dataclass(frozen=True)
class RecordSelection:
    snapshot_id: str
    stage: RetrievalStage
    records: Tuple[DurableMemoryRecord, ...]
    omitted_memory_ids: Tuple[str, ...]
    byte_size: int

    @property
    def selected_memory_ids(self) -> Tuple[str, ...]:
        return tuple(record.memory_id for record in self.records)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "stage": self.stage.value,
            "records": [record.to_dict() for record in self.records],
            "omitted_memory_ids": list(self.omitted_memory_ids),
            "byte_size": self.byte_size,
        }


@dataclass(frozen=True)
class MemoryQuery:
    assignment_id: str
    path: Optional[str] = None
    symbol: Optional[str] = None
    contract: Optional[str] = None
    query_text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignment_id", _bounded_id(self.assignment_id, "assignment_id"))
        if self.path is not None:
            scope = MemoryScope(paths=(self.path,))
            path = scope.paths[0]
            if any(character in path for character in "*?[]"):
                raise ValueError("query path must be concrete")
            object.__setattr__(self, "path", path)
        if self.symbol is not None:
            object.__setattr__(self, "symbol", _bounded_text(self.symbol, "symbol", 512))
        if self.contract is not None:
            value = _bounded_id(self.contract, "contract").casefold()
            object.__setattr__(self, "contract", value)
        object.__setattr__(
            self,
            "query_text",
            _bounded_text(self.query_text, "query_text", MAX_RETRIEVAL_TEXT_CHARS, empty=True),
        )
        if self.path is None and self.symbol is None and self.contract is None and not self.query_text:
            raise ValueError("query requires a selector or query_text")


@dataclass(frozen=True)
class MemoryQueryResult:
    assignment_id: str
    snapshot_id: str
    call_index: int
    records: Tuple[DurableMemoryRecord, ...]
    omitted_memory_ids: Tuple[str, ...]
    byte_size: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "snapshot_id": self.snapshot_id,
            "call_index": self.call_index,
            "records": [record.to_dict() for record in self.records],
            "omitted_memory_ids": list(self.omitted_memory_ids),
            "byte_size": self.byte_size,
        }


@dataclass
class _DecisionDraft:
    applicability: Applicability
    matched_scope: MemoryScope
    reasons: Tuple[str, ...]
    rank: int = 0


@dataclass(frozen=True)
class _RankedRecord:
    record: DurableMemoryRecord
    matched_scope: MemoryScope
    reasons: Tuple[str, ...]
    lexical_score: int
    graph_score: int
    semantic_score: Decimal
    rank: int


class MemorySnapshotBuilder:
    """Build an immutable, generation-pinned Snapshot from canonical values."""

    def __init__(
        self,
        applicability_evaluator: TargetHeadApplicabilityEvaluator,
        *,
        limits: Optional[RetrievalLimits] = None,
    ) -> None:
        if not callable(getattr(applicability_evaluator, "evaluate", None)):
            raise ValueError("applicability_evaluator must provide evaluate")
        self._applicability_evaluator = applicability_evaluator
        self.limits = limits or RetrievalLimits()

    def build(
        self,
        selection_input: MemorySelectionInput,
        records: Iterable[DurableMemoryRecord],
        *,
        created_at: str,
        stage: RetrievalStage = RetrievalStage.MEMORY_SELECTION,
        query_text: str = "",
        graph_relevance: Mapping[str, int] | Sequence[Tuple[str, int]] = (),
        semantic_ranker: Optional[SemanticRanker] = None,
        feedback_calibration_summary: Optional[FeedbackCalibrationSummary] = None,
        repository_knowledge_refs: Sequence[str] = (),
    ) -> MemorySnapshot:
        selection = _copy_selection_input(selection_input)
        _bounded_text(created_at, "created_at", 64)
        request = RetrievalRequest.from_selection_input(
            selection,
            stage=stage,
            query_text=query_text,
            graph_relevance=graph_relevance,
        )
        repository_records = _canonical_record_input(
            records,
            repository_key=selection.repository_key,
        )

        drafts: Dict[str, _DecisionDraft] = {}
        eligible: list[Tuple[DurableMemoryRecord, MemoryScope, Tuple[str, ...]]] = []
        for record in sorted(repository_records, key=lambda item: item.memory_id):
            status = _status_decision(record)
            if status is not None:
                drafts[record.memory_id] = status
                continue
            try:
                target = self._applicability_evaluator.evaluate(
                    record,
                    target_head=selection.head_sha,
                    changed_paths=selection.changed_paths,
                    changed_symbols=selection.changed_symbols,
                    changed_contracts=selection.contracts,
                    changed_languages=selection.languages,
                )
            except Exception:
                target = ApplicabilityDecision(
                    memory_id=record.memory_id,
                    target_head=selection.head_sha,
                    applicability=Applicability.SOURCE_MISSING,
                    reason_code="target_validity_unavailable",
                    requires_revalidation=True,
                )
            _validate_target_decision(target, record, selection.head_sha)
            if target.applicability not in {Applicability.SELECTED, Applicability.OUT_OF_SCOPE}:
                drafts[record.memory_id] = _DecisionDraft(
                    target.applicability,
                    MemoryScope(),
                    (_reason(target.reason_code),),
                )
                continue
            if not _stage_allowed(record, request.stage):
                drafts[record.memory_id] = _DecisionDraft(
                    Applicability.OUT_OF_SCOPE,
                    MemoryScope(),
                    ("stage_kind_not_allowed",),
                )
                continue
            matched, scope_reasons = _match_scope(record.scope, request.scope)
            if target.applicability is Applicability.OUT_OF_SCOPE or matched is None:
                drafts[record.memory_id] = _DecisionDraft(
                    Applicability.OUT_OF_SCOPE,
                    MemoryScope(),
                    ("target_scope_does_not_match",),
                )
                continue
            eligible.append(
                (
                    record,
                    matched,
                    tuple(scope_reasons)
                    + ("target_revision_valid", "stage_kind_allowed"),
                )
            )

        ranked = _rank_eligible(eligible, request, semantic_ranker)
        selected_ids, omitted_reasons = _count_budget(
            ranked,
            max_records=self.limits.max_snapshot_records,
            limits=self.limits,
            boundary="snapshot",
        )
        for item in ranked:
            reasons = item.reasons + (
                ("selected",) if item.record.memory_id in selected_ids else omitted_reasons[item.record.memory_id]
            )
            drafts[item.record.memory_id] = _DecisionDraft(
                Applicability.SELECTED
                if item.record.memory_id in selected_ids
                else Applicability.BUDGET_OMITTED,
                item.matched_scope,
                reasons,
                rank=item.rank,
            )
        _rank_ineligible_drafts(drafts, len(ranked))

        snapshot = self._fit_snapshot_bytes(
            selection,
            ranked,
            drafts,
            selected_ids,
            created_at=created_at,
            feedback_calibration_summary=feedback_calibration_summary,
            repository_knowledge_refs=repository_knowledge_refs,
        )
        return MemorySnapshot.from_dict(snapshot.to_dict())

    def _fit_snapshot_bytes(
        self,
        selection: MemorySelectionInput,
        ranked: Sequence[_RankedRecord],
        drafts: Dict[str, _DecisionDraft],
        selected_ids: set[str],
        *,
        created_at: str,
        feedback_calibration_summary: Optional[FeedbackCalibrationSummary],
        repository_knowledge_refs: Sequence[str],
    ) -> MemorySnapshot:
        records_by_id = {item.record.memory_id: item.record for item in ranked}
        ordinary = [
            item.record.memory_id
            for item in reversed(ranked)
            if item.record.policy_effect is None and item.record.memory_id in selected_ids
        ]
        while True:
            snapshot = _make_snapshot(
                selection,
                records_by_id,
                drafts,
                selected_ids,
                created_at=created_at,
                feedback_calibration_summary=feedback_calibration_summary,
                repository_knowledge_refs=repository_knowledge_refs,
            )
            size = _json_bytes(snapshot.to_dict())
            if size <= self.limits.max_snapshot_bytes:
                return snapshot
            if ordinary:
                memory_id = ordinary.pop(0)
                selected_ids.remove(memory_id)
                draft = drafts[memory_id]
                drafts[memory_id] = _DecisionDraft(
                    Applicability.BUDGET_OMITTED,
                    draft.matched_scope,
                    tuple(set(draft.reasons) | {"snapshot_byte_budget"}),
                    draft.rank,
                )
                continue
            hard_ids = sorted(
                memory_id
                for memory_id in selected_ids
                if records_by_id[memory_id].policy_effect is not None
            )
            minimal = _make_snapshot(
                selection,
                records_by_id,
                {memory_id: drafts[memory_id] for memory_id in hard_ids},
                set(hard_ids),
                created_at=created_at,
                feedback_calibration_summary=None,
                repository_knowledge_refs=(),
            )
            required = _json_bytes(minimal.to_dict())
            if hard_ids and required > self.limits.max_snapshot_bytes:
                raise HardPolicyBudgetExceeded(
                    boundary="snapshot",
                    budget="bytes",
                    limit=self.limits.max_snapshot_bytes,
                    required=required,
                    memory_ids=hard_ids,
                )
            raise SnapshotBudgetExceeded(
                "Snapshot decision/metadata bytes exceed configured limit"
            )


def build_disabled_snapshot(
    selection_input: MemorySelectionInput,
    *,
    created_at: str,
    max_snapshot_bytes: int = MAX_RETRIEVAL_BYTES,
) -> MemorySnapshot:
    """Return the deterministic empty Snapshot used by disabled Memory mode."""

    selection = _copy_selection_input(selection_input)
    snapshot = _make_snapshot(
        selection,
        {},
        {},
        set(),
        created_at=created_at,
        feedback_calibration_summary=None,
        repository_knowledge_refs=(),
    )
    if type(max_snapshot_bytes) is not int or max_snapshot_bytes < 1:
        raise ValueError("max_snapshot_bytes must be positive")
    if _json_bytes(snapshot.to_dict()) > max_snapshot_bytes:
        raise SnapshotBudgetExceeded("disabled Snapshot exceeds configured byte limit")
    return MemorySnapshot.from_dict(snapshot.to_dict())


build_empty_snapshot = build_disabled_snapshot


class SnapshotMemorySelector:
    """Select bounded projections from one immutable Snapshot copy."""

    def __init__(self, snapshot: MemorySnapshot, *, limits: Optional[RetrievalLimits] = None) -> None:
        if type(snapshot) is not MemorySnapshot:
            raise ValueError("snapshot must be a canonical MemorySnapshot")
        self._snapshot = MemorySnapshot.from_dict(snapshot.to_dict())
        self.limits = limits or RetrievalLimits()

    @property
    def snapshot_id(self) -> str:
        return self._snapshot.snapshot_id

    def select(
        self,
        request: RetrievalRequest,
        *,
        semantic_ranker: Optional[SemanticRanker] = None,
        max_records: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ) -> RecordSelection:
        if type(request) is not RetrievalRequest:
            raise ValueError("request must be a RetrievalRequest")
        record_limit = self.limits.max_context_records if max_records is None else max_records
        byte_limit = self.limits.max_context_bytes if max_bytes is None else max_bytes
        if type(record_limit) is not int or not 1 <= record_limit <= self.limits.max_context_records:
            raise RetrievalInputError("per-call record limit exceeds configured bound")
        if type(byte_limit) is not int or not 1 <= byte_limit <= self.limits.max_context_bytes:
            raise RetrievalInputError("per-call byte limit exceeds configured bound")
        eligible = []
        for record in self._snapshot.eligible_records:
            if not _stage_allowed(record, request.stage):
                continue
            matched, reasons = _match_scope(record.scope, request.scope)
            if matched is not None:
                eligible.append((record, matched, tuple(reasons)))
        ranked = _rank_eligible(eligible, request, semantic_ranker)
        records, omitted, size = _bounded_records(
            ranked,
            max_records=record_limit,
            max_bytes=byte_limit,
            limits=self.limits,
            boundary="context",
            envelope={"snapshot_id": self._snapshot.snapshot_id, "stage": request.stage.value},
        )
        return RecordSelection(
            snapshot_id=self._snapshot.snapshot_id,
            stage=request.stage,
            records=records,
            omitted_memory_ids=omitted,
            byte_size=size,
        )


class SnapshotMemoryQueryService:
    """Assignment-bound query service backed only by a copied Snapshot."""

    def __init__(
        self,
        snapshot: MemorySnapshot,
        *,
        assignment_id: str,
        assignment_scope: MemoryScope,
        limits: Optional[RetrievalLimits] = None,
    ) -> None:
        if type(snapshot) is not MemorySnapshot:
            raise ValueError("snapshot must be a canonical MemorySnapshot")
        if type(assignment_scope) is not MemoryScope:
            raise ValueError("assignment_scope must be a MemoryScope")
        self._snapshot = MemorySnapshot.from_dict(snapshot.to_dict())
        self._assignment_id = _bounded_id(assignment_id, "assignment_id")
        self._assignment_scope = MemoryScope.from_dict(assignment_scope.to_dict())
        self.limits = limits or RetrievalLimits()
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def query(
        self,
        request: MemoryQuery,
        *,
        semantic_ranker: Optional[SemanticRanker] = None,
    ) -> MemoryQueryResult:
        if type(request) is not MemoryQuery:
            raise ValueError("request must be a MemoryQuery")
        if self._call_count >= self.limits.max_query_calls:
            raise QueryLimitExceeded("Snapshot query call limit exceeded")
        self._call_count += 1
        if request.assignment_id != self._assignment_id:
            raise QueryScopeViolation("query Assignment does not match bound Assignment")
        if len(request.query_text) > self.limits.max_query_text_chars:
            raise QueryLimitExceeded("query text exceeds configured limit")
        _authorize_query(request, self._assignment_scope)
        explicit_scope = MemoryScope(
            paths=() if request.path is None else (request.path,),
            symbols=() if request.symbol is None else (request.symbol,),
            contracts=() if request.contract is None else (request.contract,),
        )
        query_scope = explicit_scope if not explicit_scope.is_empty else self._assignment_scope
        retrieval = RetrievalRequest(
            stage=RetrievalStage.REVIEWER,
            paths=query_scope.paths,
            symbols=query_scope.symbols,
            contracts=query_scope.contracts,
            languages=query_scope.languages,
            query_text=request.query_text,
        )
        eligible = []
        for record in self._snapshot.eligible_records:
            assignment_match = MemoryScope() if self._assignment_scope.is_empty else _match_scope(
                record.scope, self._assignment_scope
            )[0]
            query_match, reasons = _match_scope(record.scope, query_scope)
            if assignment_match is not None and query_match is not None:
                eligible.append((record, query_match, tuple(reasons)))
        ranked = _rank_eligible(eligible, retrieval, semantic_ranker)
        records, omitted, size = _bounded_records(
            ranked,
            max_records=self.limits.max_query_results,
            max_bytes=self.limits.max_query_bytes,
            limits=self.limits,
            boundary="query",
            envelope={
                "assignment_id": self._assignment_id,
                "snapshot_id": self._snapshot.snapshot_id,
                "call_index": self._call_count,
            },
        )
        return MemoryQueryResult(
            assignment_id=self._assignment_id,
            snapshot_id=self._snapshot.snapshot_id,
            call_index=self._call_count,
            records=records,
            omitted_memory_ids=omitted,
            byte_size=size,
        )


MemoryQueryService = SnapshotMemoryQueryService
SnapshotQueryService = SnapshotMemoryQueryService


def build_memory_snapshot(
    applicability_evaluator: TargetHeadApplicabilityEvaluator,
    selection_input: MemorySelectionInput,
    records: Iterable[DurableMemoryRecord],
    *,
    created_at: str,
    limits: Optional[RetrievalLimits] = None,
    **kwargs: Any,
) -> MemorySnapshot:
    return MemorySnapshotBuilder(applicability_evaluator, limits=limits).build(
        selection_input,
        records,
        created_at=created_at,
        **kwargs,
    )


def _copy_selection_input(value: MemorySelectionInput) -> MemorySelectionInput:
    if type(value) is not MemorySelectionInput:
        raise ValueError("selection_input must be a canonical MemorySelectionInput")
    return MemorySelectionInput.from_dict(value.to_dict())


def _canonical_record_input(
    records: Iterable[DurableMemoryRecord],
    *,
    repository_key: str,
) -> Tuple[DurableMemoryRecord, ...]:
    if isinstance(records, (str, bytes, Mapping)):
        raise ValueError("records must contain canonical DurableMemoryRecord values")
    try:
        iterator = iter(records)
    except TypeError:
        raise ValueError("records must be iterable") from None
    canonical: Dict[str, DurableMemoryRecord] = {}
    inspected = 0
    for value in iterator:
        inspected += 1
        if inspected > MAX_SNAPSHOT_DECISIONS:
            raise SnapshotBudgetExceeded(
                "retrieval input catalog exceeds the bounded decision limit"
            )
        if type(value) is not DurableMemoryRecord:
            raise ValueError("records must not contain rows or non-canonical values")
        # Repository is the first authority gate.  A foreign record cannot
        # collide with, invalidate, or otherwise influence this repository's
        # canonical Memory ID catalog.
        if value.repository_key != repository_key:
            continue
        copy = DurableMemoryRecord.from_dict(value.to_dict())
        if copy.memory_id in canonical:
            raise ValueError("records must not repeat a memory_id")
        canonical[copy.memory_id] = copy
    return tuple(canonical[key] for key in sorted(canonical))


def _status_decision(record: DurableMemoryRecord) -> Optional[_DecisionDraft]:
    mapping = {
        RecordStatus.REVALIDATION_REQUIRED: (
            Applicability.SOURCE_CHANGED,
            "record_revalidation_required",
        ),
        RecordStatus.REVOKED: (Applicability.REVOKED, "record_revoked"),
        RecordStatus.SUPERSEDED: (Applicability.SUPERSEDED, "record_superseded"),
        RecordStatus.EXPIRED: (Applicability.EXPIRED, "record_expired"),
    }
    if record.status is RecordStatus.ACTIVE:
        return None
    applicability, reason = mapping.get(
        record.status,
        (Applicability.SOURCE_CHANGED, "record_status_not_authoritative"),
    )
    return _DecisionDraft(applicability, MemoryScope(), (reason,))


def _validate_target_decision(
    decision: ApplicabilityDecision,
    record: DurableMemoryRecord,
    head_sha: str,
) -> None:
    if (
        not isinstance(decision, ApplicabilityDecision)
        or decision.memory_id != record.memory_id
        or decision.target_head.casefold() != head_sha.casefold()
        or not isinstance(decision.applicability, Applicability)
    ):
        raise RetrievalInputError("applicability evaluator returned an inconsistent decision")


def _stage_allowed(record: DurableMemoryRecord, stage: RetrievalStage) -> bool:
    if record.kind in _STAGE_KINDS[stage]:
        return True
    effect = record.policy_effect
    return effect is not None and effect.effect_kind in _STAGE_EFFECTS.get(stage, frozenset())


def _match_scope(record_scope: MemoryScope, request_scope: MemoryScope) -> Tuple[Optional[MemoryScope], Tuple[str, ...]]:
    if record_scope.is_empty:
        return MemoryScope(), ("global_scope",)
    paths = tuple(
        sorted(
            path
            for path in request_scope.paths
            if any(_path_patterns_overlap(path, pattern) for pattern in record_scope.paths)
        )
    )
    symbols = tuple(sorted(set(record_scope.symbols).intersection(request_scope.symbols)))
    contracts = tuple(sorted(set(record_scope.contracts).intersection(request_scope.contracts)))
    languages = tuple(sorted(set(record_scope.languages).intersection(request_scope.languages)))
    if not (paths or symbols or contracts or languages):
        return None, ()
    reasons = []
    if paths:
        reasons.append("path_match")
    if symbols:
        reasons.append("symbol_match")
    if contracts:
        reasons.append("contract_match")
    if languages:
        reasons.append("language_match")
    return MemoryScope(paths=paths, symbols=symbols, contracts=contracts, languages=languages), tuple(reasons)


def _path_patterns_overlap(left: str, right: str) -> bool:
    left_glob = any(character in left for character in "*?[]")
    right_glob = any(character in right for character in "*?[]")
    if not left_glob:
        return fnmatch.fnmatchcase(left, right)
    if not right_glob:
        return fnmatch.fnmatchcase(right, left)
    if left == right:
        return True
    # The common repository-scope form ``prefix/**`` has an exact, safe
    # intersection rule: two such trees overlap iff either root contains the
    # other.  More complex glob/glob pairs fail closed instead of guessing and
    # potentially exposing an out-of-assignment record.
    left_root = _recursive_tree_root(left)
    right_root = _recursive_tree_root(right)
    if left_root is None or right_root is None:
        return False
    return (
        not left_root
        or not right_root
        or left_root == right_root
        or left_root.startswith(right_root + "/")
        or right_root.startswith(left_root + "/")
    )


def _recursive_tree_root(pattern: str) -> Optional[str]:
    if pattern == "**":
        return ""
    if not pattern.endswith("/**"):
        return None
    root = pattern[:-3].rstrip("/")
    if any(character in root for character in "*?[]"):
        return None
    return root


def _rank_eligible(
    eligible: Sequence[Tuple[DurableMemoryRecord, MemoryScope, Tuple[str, ...]]],
    request: RetrievalRequest,
    semantic_ranker: Optional[SemanticRanker],
) -> Tuple[_RankedRecord, ...]:
    records = tuple(item[0] for item in eligible)
    semantic = _semantic_scores(records, request, semantic_ranker)
    graph = dict(request.graph_relevance)
    provisional = []
    for record, matched, reasons in eligible:
        lexical = _lexical_score(record, request)
        provisional.append(
            (
                _policy_priority(record),
                -semantic[record.memory_id],
                -lexical,
                -graph.get(record.memory_id, 0),
                record.memory_id,
                record,
                matched,
                reasons,
                lexical,
                graph.get(record.memory_id, 0),
            )
        )
    provisional.sort(key=lambda item: item[:5])
    return tuple(
        _RankedRecord(
            record=item[5],
            matched_scope=item[6],
            reasons=item[7],
            lexical_score=item[8],
            graph_score=item[9],
            semantic_score=-item[1],
            rank=rank,
        )
        for rank, item in enumerate(provisional)
    )


def _semantic_scores(
    records: Tuple[DurableMemoryRecord, ...],
    request: RetrievalRequest,
    ranker: Optional[SemanticRanker],
) -> Dict[str, Decimal]:
    ids = {record.memory_id for record in records}
    if ranker is None or not records:
        return {memory_id: Decimal(0) for memory_id in ids}
    try:
        values = ranker(records, request)
    except Exception as error:
        raise SemanticRankerViolation("semantic ranker failed") from error
    if not isinstance(values, Mapping) or set(values) != ids:
        raise SemanticRankerViolation("semantic ranker must score exactly the eligible set")
    result: Dict[str, Decimal] = {}
    for memory_id, raw in values.items():
        if isinstance(raw, bool):
            raise SemanticRankerViolation("semantic score must be numeric")
        try:
            score = Decimal(str(raw))
        except (InvalidOperation, ValueError):
            raise SemanticRankerViolation("semantic score must be finite") from None
        if not score.is_finite() or abs(score) > Decimal("1000000000"):
            raise SemanticRankerViolation("semantic score is outside the supported range")
        result[memory_id] = score
    return result


def _policy_priority(record: DurableMemoryRecord) -> Tuple[int, int]:
    if record.policy_effect is None:
        return (1, 0)
    return (0, _EFFECT_PRIORITY[record.policy_effect.effect_kind])


def _lexical_score(record: DurableMemoryRecord, request: RetrievalRequest) -> int:
    query_tokens = _tokens(
        " ".join(
            (
                request.query_text,
                *request.paths,
                *request.symbols,
                *request.contracts,
                *request.languages,
            )
        )
    )
    if not query_tokens:
        return 0
    statement = _tokens(record.statement)
    scope = _tokens(
        " ".join(
            (
                record.kind.value,
                *record.scope.paths,
                *record.scope.symbols,
                *record.scope.contracts,
                *record.scope.languages,
            )
        )
    )
    if record.policy_effect is not None:
        scope |= _tokens(record.policy_effect.effect_kind.value + " " + record.policy_effect.value)
    return 4 * len(query_tokens.intersection(statement)) + 2 * len(query_tokens.intersection(scope))


def _tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return {token for token in _TOKEN_RE.findall(normalized) if token}


def _count_budget(
    ranked: Sequence[_RankedRecord],
    *,
    max_records: int,
    limits: RetrievalLimits,
    boundary: str,
) -> Tuple[set[str], Dict[str, Tuple[str, ...]]]:
    hard = [item for item in ranked if item.record.policy_effect is not None]
    if len(hard) > max_records:
        raise HardPolicyBudgetExceeded(
            boundary=boundary,
            budget="records",
            limit=max_records,
            required=len(hard),
            memory_ids=[item.record.memory_id for item in hard],
        )
    hard_kind_counts: Dict[MemoryKind, int] = {}
    for item in hard:
        hard_kind_counts[item.record.kind] = hard_kind_counts.get(item.record.kind, 0) + 1
    for kind, count in hard_kind_counts.items():
        limit = limits.kind_limit(kind)
        if limit is not None and count > limit:
            raise HardPolicyBudgetExceeded(
                boundary=boundary,
                budget="per_kind",
                limit=limit,
                required=count,
                memory_ids=[item.record.memory_id for item in hard if item.record.kind is kind],
            )
    selected: set[str] = set()
    omitted: Dict[str, Tuple[str, ...]] = {}
    kind_counts: Dict[MemoryKind, int] = {}
    for item in ranked:
        record = item.record
        kind_limit = limits.kind_limit(record.kind)
        if len(selected) >= max_records:
            if record.policy_effect is not None:
                raise AssertionError("hard policy count preflight was inconsistent")
            omitted[record.memory_id] = ("record_budget",)
            continue
        if kind_limit is not None and kind_counts.get(record.kind, 0) >= kind_limit:
            if record.policy_effect is not None:
                raise AssertionError("hard policy per-kind preflight was inconsistent")
            omitted[record.memory_id] = ("per_kind_budget",)
            continue
        selected.add(record.memory_id)
        kind_counts[record.kind] = kind_counts.get(record.kind, 0) + 1
    return selected, omitted


def _bounded_records(
    ranked: Sequence[_RankedRecord],
    *,
    max_records: int,
    max_bytes: int,
    limits: RetrievalLimits,
    boundary: str,
    envelope: Mapping[str, Any],
) -> Tuple[Tuple[DurableMemoryRecord, ...], Tuple[str, ...], int]:
    selected_ids, omitted_reasons = _count_budget(
        ranked,
        max_records=max_records,
        limits=limits,
        boundary=boundary,
    )
    records = [item.record for item in ranked if item.record.memory_id in selected_ids]
    ordinary = [record for record in reversed(records) if record.policy_effect is None]
    while True:
        payload = dict(envelope)
        payload["records"] = [record.to_dict() for record in records]
        size = _json_bytes(payload)
        if size <= max_bytes:
            omitted = tuple(
                item.record.memory_id
                for item in ranked
                if item.record.memory_id not in {record.memory_id for record in records}
            )
            return tuple(records), omitted, size
        if ordinary:
            drop = ordinary.pop(0)
            records.remove(drop)
            omitted_reasons[drop.memory_id] = ("byte_budget",)
            continue
        hard_ids = [record.memory_id for record in records if record.policy_effect is not None]
        if hard_ids:
            raise HardPolicyBudgetExceeded(
                boundary=boundary,
                budget="bytes",
                limit=max_bytes,
                required=size,
                memory_ids=hard_ids,
            )
        empty_size = _json_bytes({**envelope, "records": []})
        if empty_size > max_bytes:
            raise ProjectionBudgetExceeded(
                boundary=boundary,
                limit=max_bytes,
                required=empty_size,
            )
        return tuple(), tuple(item.record.memory_id for item in ranked), empty_size


def _rank_ineligible_drafts(drafts: Dict[str, _DecisionDraft], start: int) -> None:
    unranked = sorted(
        (memory_id for memory_id, draft in drafts.items() if draft.applicability not in {Applicability.SELECTED, Applicability.BUDGET_OMITTED}),
        key=lambda memory_id: (drafts[memory_id].applicability.value, memory_id),
    )
    for offset, memory_id in enumerate(unranked):
        draft = drafts[memory_id]
        drafts[memory_id] = _DecisionDraft(draft.applicability, draft.matched_scope, draft.reasons, start + offset)


def _make_snapshot(
    selection: MemorySelectionInput,
    records_by_id: Mapping[str, DurableMemoryRecord],
    drafts: Mapping[str, _DecisionDraft],
    selected_ids: set[str],
    *,
    created_at: str,
    feedback_calibration_summary: Optional[FeedbackCalibrationSummary],
    repository_knowledge_refs: Sequence[str],
) -> MemorySnapshot:
    decisions = tuple(
        MemorySelectionDecision(
            memory_id=memory_id,
            applicability=draft.applicability,
            matched_scope=draft.matched_scope,
            reason_codes=tuple(sorted(set(draft.reasons))),
            rank=draft.rank,
        )
        for memory_id, draft in sorted(drafts.items())
    )
    feedback = None
    if feedback_calibration_summary is not None:
        if type(feedback_calibration_summary) is not FeedbackCalibrationSummary:
            raise ValueError("feedback_calibration_summary must be canonical")
        feedback = FeedbackCalibrationSummary.from_dict(feedback_calibration_summary.to_dict())
    return MemorySnapshot(
        repository_key=selection.repository_key,
        base_sha=selection.base_sha,
        head_sha=selection.head_sha,
        generations=type(selection.generations).from_dict(selection.generations.to_dict()),
        selection_policy_version=selection.selection_policy_version,
        eligible_records=tuple(records_by_id[memory_id] for memory_id in sorted(selected_ids)),
        applicability_decisions=decisions,
        feedback_calibration_summary=feedback,
        repository_knowledge_refs=tuple(repository_knowledge_refs),
        created_at=created_at,
    )


def _authorize_query(request: MemoryQuery, assignment: MemoryScope) -> None:
    if assignment.is_empty:
        return
    if request.path is not None and not any(
        fnmatch.fnmatchcase(request.path, pattern) for pattern in assignment.paths
    ):
        raise QueryScopeViolation("query path is outside the Assignment scope")
    if request.symbol is not None and request.symbol not in assignment.symbols:
        raise QueryScopeViolation("query symbol is outside the Assignment scope")
    if request.contract is not None and request.contract not in assignment.contracts:
        raise QueryScopeViolation("query contract is outside the Assignment scope")


def _json_bytes(value: Any) -> int:
    return len(canonical_json(value).encode("utf-8"))


def _bounded_text(value: Any, name: str, maximum: int, *, empty: bool = False) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("%s must be valid text" % name)
    normalized = " ".join(value.split())
    if (not empty and not normalized) or len(normalized) > maximum:
        raise ValueError("%s is outside its supported length" % name)
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("%s must be valid UTF-8" % name) from None
    return normalized


def _bounded_id(value: Any, name: str) -> str:
    text = _bounded_text(value, name, 512)
    if not _SAFE_ID_RE.fullmatch(text):
        raise ValueError("%s must be a canonical identifier" % name)
    return text


def _reason(value: Any) -> str:
    if not isinstance(value, str):
        return "target_validity_unavailable"
    normalized = re.sub(r"[^a-z0-9_.:+#/@-]+", "_", value.strip().casefold()).strip("_")
    return normalized[:512] or "target_validity_unavailable"


__all__ = [
    "HardPolicyBudgetExceeded",
    "MemoryQuery",
    "MemoryQueryResult",
    "MemoryQueryService",
    "MemoryRetrievalError",
    "MemorySnapshotBuilder",
    "ProjectionBudgetExceeded",
    "QueryLimitExceeded",
    "QueryScopeViolation",
    "RecordSelection",
    "RetrievalInputError",
    "RetrievalLimits",
    "RetrievalRequest",
    "RetrievalStage",
    "SemanticRankerViolation",
    "SnapshotBudgetExceeded",
    "SnapshotMemoryQueryService",
    "SnapshotMemorySelector",
    "SnapshotQueryService",
    "build_disabled_snapshot",
    "build_empty_snapshot",
    "build_memory_snapshot",
]

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

from review_agent.aggregation import ReviewAggregationInput
from review_agent.diff_artifact import DiffArtifact, DiffArtifactStore
from review_agent.developer_rules import DeveloperRuleResolver
from review_agent.execution_journal import ExecutionJournal
from review_agent.global_memory import GlobalMemoryFacade
from review_agent.intent_runtime import IntentRuntime
from review_agent.intent_inference import (
    IntentInferenceEvidence,
    IntentInferenceResult,
    IntentInferenceRun,
    IntentInferenceTrace,
    run_intent_inference,
)
from review_agent.local_quality import LocalQualityPlan, LocalQualityRunner
from review_agent.model_adapter_factory import (
    ModelAdapterConfig,
    build_model_adapter_factory_from_config,
)
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelToolResult,
    ModelTurnRequest,
)
from review_agent.model_risk import (
    RISK_MODEL_SYSTEM_PROMPT_V2,
    parse_risk_decision_v2,
)
from review_agent.preflight import DeterministicPreflight
from review_agent.pr_workspace import (
    ArtifactDescriptor,
    PRMetadata,
    PRWorkspaceStore,
    SessionWorkspace,
    SnapshotWorkspace,
)
from review_agent.repository_intelligence import (
    ChangedSymbolsV2,
    changed_symbols_v2_from_dict,
    collect_python_symbols,
    search_repository_text,
)
from review_agent.review_agent_loop import ReviewAgentLoopV2
from review_agent.review_context import (
    AvailableArtifact,
    ReviewerContextInput,
    build_reviewer_invocation_v2,
)
from review_agent.review_pipeline import (
    PipelineContextV6,
    ReviewPipelineServicesV6,
    ReviewPipelineV6,
    aggregate_and_render_v6,
    load_reviewer_results_v6,
    publish_reviewer_result_v6,
)
from review_agent.review_planning import (
    ReviewPlanningRuntime,
    fixed_reviewer_slots,
)
from review_agent.review_protocol import (
    IntentVersionEnvelope,
    ReviewPlan,
    ReviewRequest,
    ReviewResult,
    ReviewerAssignment,
)
from review_agent.review_renderer import render_review_result_markdown
from review_agent.review_tool_gateway import (
    ReviewToolFailure,
    ReviewToolGateway,
    ToolBackendResult,
)
from review_agent.reviewer_executor import (
    ReviewerExecutionResultV2,
    ReviewerExecutorV2,
)
from review_agent.reviewer_output import ReviewerOutputParser
from review_agent.reviewer_runtime import ReviewerRuntimeLimitsV2
from review_agent.revision import (
    RevisionResolver,
    canonical_repository_identity,
    sanitized_git_environment,
)
from review_agent.risk_runtime import RiskRuntime
from review_agent.safe_io import canonical_json_bytes
from review_agent.session import SessionV6ArtifactRef, SessionV6Manifest
from review_agent.session_store import SessionV6Store
from review_agent.tool_artifacts import (
    ToolResultArtifactStore,
    ToolResultProjector,
)
from review_agent.tool_result_protocol import serialize_tool_result_projection_v2


PRODUCT_REVIEW_REQUEST_SCHEMA = "product_review_request_v7"
_LEGACY_PRODUCT_REVIEW_REQUEST_SCHEMA = "product_review_request_v6"
_REQUEST_PREFIX = "Requests/request-"
_INTENT_PREFIX = "Intent/intent-"


class ProductRuntimeError(RuntimeError):
    pass


class ProductRuntimeUsageError(ProductRuntimeError):
    pass


class ProductRuntimeIntegrityError(ProductRuntimeError):
    pass


class ProductRuntimeInfrastructureError(ProductRuntimeError):
    def __init__(
        self,
        message: str,
        *,
        locator: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.locator = None if locator is None else dict(locator)


class IntentTrustPolicyV6(str, Enum):
    NORMAL = "normal"
    EVALUATION_TRUST_MODEL = "evaluation_trust_model"


@dataclass(frozen=True)
class ProductReviewInputV6:
    request: ReviewRequest
    declared_goal: str | None = None
    title: str | None = None
    description: str | None = None
    intent_trust_policy: IntentTrustPolicyV6 = IntentTrustPolicyV6.NORMAL

    def __post_init__(self) -> None:
        if type(self.request) is not ReviewRequest:
            raise ProductRuntimeUsageError("request must be a v6 ReviewRequest")
        for name in ("declared_goal", "title", "description"):
            value = getattr(self, name)
            if value is not None and (
                type(value) is not str
                or not value.strip()
                or "\x00" in value
            ):
                raise ProductRuntimeUsageError(
                    f"{name} must be non-empty text or null"
                )
        if type(self.intent_trust_policy) is not IntentTrustPolicyV6:
            raise ProductRuntimeUsageError("Intent trust policy is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PRODUCT_REVIEW_REQUEST_SCHEMA,
            "request": self.request.to_dict(),
            "declared_goal": self.declared_goal,
            "title": self.title,
            "description": self.description,
            "intent_trust_policy": self.intent_trust_policy.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProductReviewInputV6":
        legacy_expected = {
            "schema_version",
            "request",
            "declared_goal",
            "title",
            "description",
        }
        expected = {
            *legacy_expected,
            "intent_trust_policy",
        }
        if type(payload) is not dict:
            raise ProductRuntimeIntegrityError(
                "persisted product request schema is invalid"
            )
        schema = payload.get("schema_version")
        if schema == PRODUCT_REVIEW_REQUEST_SCHEMA:
            if set(payload) != expected:
                raise ProductRuntimeIntegrityError(
                    "persisted product request schema is invalid"
                )
            trust_policy = payload["intent_trust_policy"]
        elif schema == _LEGACY_PRODUCT_REVIEW_REQUEST_SCHEMA:
            if set(payload) != legacy_expected:
                raise ProductRuntimeIntegrityError(
                    "persisted product request schema is invalid"
                )
            trust_policy = IntentTrustPolicyV6.NORMAL.value
        else:
            raise ProductRuntimeIntegrityError(
                "persisted product request version is unsupported"
            )
        try:
            return cls(
                request=ReviewRequest.from_dict(payload["request"]),
                declared_goal=payload["declared_goal"],
                title=payload["title"],
                description=payload["description"],
                intent_trust_policy=IntentTrustPolicyV6(trust_policy),
            )
        except (TypeError, ValueError) as error:
            raise ProductRuntimeIntegrityError(
                "persisted product request is invalid"
            ) from error


@dataclass(frozen=True)
class ProductRuntimeConfigV6:
    reviewer: ModelAdapterConfig
    risk: ModelAdapterConfig | None = None
    quality_plan: LocalQualityPlan = LocalQualityPlan(commands=())
    reviewer_limits: ReviewerRuntimeLimitsV2 = ReviewerRuntimeLimitsV2()

    def __post_init__(self) -> None:
        if not isinstance(self.reviewer, ModelAdapterConfig):
            raise ProductRuntimeUsageError(
                "reviewer adapter configuration is invalid"
            )
        if self.risk is not None and not isinstance(self.risk, ModelAdapterConfig):
            raise ProductRuntimeUsageError("risk adapter configuration is invalid")
        if not isinstance(self.quality_plan, LocalQualityPlan):
            raise ProductRuntimeUsageError("quality plan is invalid")
        if not isinstance(self.reviewer_limits, ReviewerRuntimeLimitsV2):
            raise ProductRuntimeUsageError("reviewer limits are invalid")


@dataclass(frozen=True)
class ProductReviewOutcomeV6:
    pr_id: str
    snapshot_id: str
    session_id: str
    review_result: ReviewResult
    review_result_json: str
    review_markdown: str
    manifest: SessionV6Manifest | None
    reused: bool

    def locator(self) -> dict[str, str]:
        return {
            "pr_id": self.pr_id,
            "snapshot_id": self.snapshot_id,
            "session_id": self.session_id,
        }


def start_product_review_v6(
    *,
    repository: Path,
    workspace_root: Path,
    base_revision: str,
    head_revision: str,
    external_review_id: str,
    review_input: ProductReviewInputV6,
    config: ProductRuntimeConfigV6,
) -> ProductReviewOutcomeV6:
    if type(external_review_id) is not str or not external_review_id.strip():
        raise ProductRuntimeUsageError("external review ID is required")
    resolver = RevisionResolver()
    try:
        repository_identity = resolver.repository_identity(Path(repository))
        repo = Path(repository_identity.canonical_path)
        revisions = resolver.resolve_pair(repo, base_revision, head_revision)
        workspace_store = PRWorkspaceStore(Path(workspace_root))
        resolved_pr = workspace_store.resolve_pr(
            repository_identity,
            "cli",
            external_review_id,
        )
        workspace = workspace_store.create_or_load_workspace(
            resolved_pr,
            PRMetadata(
                title=review_input.title,
                description=review_input.description,
                base_ref=base_revision,
                head_ref=head_revision,
            ),
        )
        snapshot = workspace_store.create_or_load_snapshot(
            workspace,
            revisions.resolved_base_sha,
            revisions.resolved_head_sha,
        )
        session = workspace_store.create_session(workspace, snapshot)
    except ProductRuntimeError:
        raise
    except ValueError as error:
        raise ProductRuntimeUsageError(str(error)) from error
    except Exception as error:
        raise ProductRuntimeInfrastructureError(
            f"unable to initialize PRWorkspace: {type(error).__name__}"
        ) from error

    request_descriptor = workspace_store.publish_create_only(
        snapshot,
        _request_path(session),
        canonical_json_bytes(review_input.to_dict()),
    )
    runtime = _BoundProductRuntimeV6(
        repository=repo,
        workspace_store=workspace_store,
        session=session,
        config=config,
        request_descriptor=request_descriptor,
    )
    return runtime.run()


def resume_product_review_v6(
    *,
    repository: Path,
    workspace_root: Path,
    pr_id: str,
    snapshot_id: str,
    session_id: str,
    config: ProductRuntimeConfigV6,
) -> ProductReviewOutcomeV6:
    try:
        workspace_store = PRWorkspaceStore(Path(workspace_root))
        session = workspace_store.open_session(
            pr_id=pr_id,
            snapshot_id=snapshot_id,
            session_id=session_id,
        )
        resolver = RevisionResolver()
        identity = canonical_repository_identity(
            resolver.repository_identity(Path(repository))
        )
    except ValueError as error:
        raise ProductRuntimeIntegrityError(str(error)) from error
    except Exception as error:
        raise ProductRuntimeIntegrityError(
            f"unable to open Session v6: {type(error).__name__}"
        ) from error
    if identity != session.workspace.resolved_pr.repository:
        raise ProductRuntimeIntegrityError(
            "repository identity does not match the resumed PRWorkspace"
        )

    existing = _load_outcome(
        workspace_store,
        session,
        manifest=None,
        reused=True,
    )
    if existing is not None:
        return existing
    try:
        request_descriptor = workspace_store.find_snapshot_artifact(
            session.snapshot,
            _request_path(session),
        )
    except Exception as error:
        raise ProductRuntimeIntegrityError(
            "Session v6 product request is unavailable"
        ) from error
    runtime = _BoundProductRuntimeV6(
        repository=Path(repository).resolve(),
        workspace_store=workspace_store,
        session=session,
        config=config,
        request_descriptor=request_descriptor,
    )
    return runtime.run()


class _BoundProductRuntimeV6:
    def __init__(
        self,
        *,
        repository: Path,
        workspace_store: PRWorkspaceStore,
        session: SessionWorkspace,
        config: ProductRuntimeConfigV6,
        request_descriptor: ArtifactDescriptor,
    ) -> None:
        self.repository = Path(repository)
        self.workspace_store = workspace_store
        self.session = session
        self.snapshot = session.snapshot
        self.config = config
        self.request_descriptor = request_descriptor
        self.diff_store = DiffArtifactStore(workspace_store)
        self.intent_runtime = IntentRuntime(workspace_store)
        self.risk_runtime = RiskRuntime(workspace_store)
        self.planning_runtime = ReviewPlanningRuntime(workspace_store)
        self.tool_artifacts = ToolResultArtifactStore(
            workspace_store,
            self.snapshot,
        )
        self.developer_rules = DeveloperRuleResolver()
        self.session_store = SessionV6Store(workspace_store, session)
        self.context = PipelineContextV6(
            workspace_store=workspace_store,
            snapshot=self.snapshot,
            session=session,
            session_store=self.session_store,
        )
        self._reviewer_factory = None
        self._risk_factory = None

    def run(self) -> ProductReviewOutcomeV6:
        existing = _load_outcome(
            self.workspace_store,
            self.session,
            manifest=None,
            reused=True,
        )
        if existing is not None:
            return existing
        try:
            self._reviewer_factory = build_model_adapter_factory_from_config(
                self.config.reviewer,
                stage_label="reviewer",
            )
            self._risk_factory = (
                None
                if self.config.risk is None
                else build_model_adapter_factory_from_config(
                    self.config.risk,
                    stage_label="risk-assessor",
                )
            )
        except (TypeError, ValueError) as error:
            raise ProductRuntimeUsageError(str(error)) from error
        pipeline = ReviewPipelineV6(self.context, self._services())
        try:
            manifest = pipeline.run()
        except Exception as error:
            raise ProductRuntimeInfrastructureError(
                f"Session v6 execution failed: {type(error).__name__}",
                locator=self._locator(),
            ) from error
        outcome = _load_outcome(
            self.workspace_store,
            self.session,
            manifest=manifest,
            reused=False,
        )
        if outcome is None:
            phase = manifest.current_phase.value
            checkpoint = manifest.phases.get(phase)
            error_code = checkpoint.error_code if checkpoint is not None else None
            suffix = f" ({error_code})" if error_code else ""
            raise ProductRuntimeInfrastructureError(
                f"Session v6 stopped before ReviewResult at {phase}{suffix}",
                locator=self._locator(),
            )
        return outcome

    def _locator(self) -> dict[str, str]:
        return {
            "pr_id": self.session.workspace.pr_id,
            "snapshot_id": self.snapshot.snapshot_id,
            "session_id": self.session.session_id,
        }

    def _services(self) -> ReviewPipelineServicesV6:
        return ReviewPipelineServicesV6(
            preflight=self._preflight,
            intent=self._intent,
            planning=self._planning,
            load_review_plan=self._load_plan,
            assemble_and_run_reviewer=self._run_reviewer,
            persist_reviewer_result=publish_reviewer_result_v6,
            load_reviewer_results=load_reviewer_results_v6,
            aggregate_and_render=aggregate_and_render_v6,
        )

    def _preflight(
        self,
        _context: PipelineContextV6,
    ) -> tuple[SessionV6ArtifactRef, ...]:
        result = DeterministicPreflight(
            workspace_store=self.workspace_store,
            diff_store=self.diff_store,
            quality_runner=LocalQualityRunner(),
        ).run(
            self.repository,
            self.snapshot,
            self.config.quality_plan,
            sink=self.tool_artifacts.preflight_sink(),
        )
        diff = self.diff_store.load(self.snapshot)
        quality = self.workspace_store.find_snapshot_artifact(
            self.snapshot,
            "QualityGate/quality-gate.json",
        )
        symbols = self.workspace_store.find_snapshot_artifact(
            self.snapshot,
            "ChangedSymbols/changed-symbols.json",
        )
        if (
            result.diff_artifact_id != diff.patch.artifact_id
            or result.diff_index_artifact_id != diff.index_artifact.artifact_id
            or result.quality_artifact_id != quality.artifact_id
            or result.changed_symbols_artifact_id != symbols.artifact_id
        ):
            raise ProductRuntimeIntegrityError(
                "Preflight Artifact binding changed"
            )
        return (
            _artifact_ref("preflight.request", self.request_descriptor),
            _artifact_ref("preflight.diff_patch", diff.patch),
            _artifact_ref("preflight.diff_index", diff.index_artifact),
            _artifact_ref("preflight.quality_gate", quality),
            _artifact_ref("preflight.changed_symbols", symbols),
        )

    def _intent(
        self,
        _context: PipelineContextV6,
    ) -> tuple[SessionV6ArtifactRef, ...]:
        intent_path = _intent_path(self.session)
        existing = _find_optional_artifact(
            self.workspace_store,
            self.snapshot,
            intent_path,
        )
        if existing is not None:
            self._load_intent_envelope()
            return (_artifact_ref("intent.packet", existing),)
        review_input = self._load_review_input()
        explicit = (
            review_input.declared_goal
            or review_input.title
            or review_input.description
        )
        inference_run = None
        if explicit is None:
            inference_run = self._run_intent_agent(review_input)
        envelope = self.intent_runtime.resolve(
            self.session.workspace,
            self.snapshot,
            review_input.request,
            declared_goal=review_input.declared_goal,
            pr_title=review_input.title,
            pr_description=review_input.description,
            inference_run=inference_run,
            trust_policy=review_input.intent_trust_policy.value,
        )
        descriptor = self.workspace_store.publish_create_only(
            self.snapshot,
            intent_path,
            envelope.to_json_bytes(),
        )
        return (_artifact_ref("intent.packet", descriptor),)

    def _run_intent_agent(
        self,
        review_input: ProductReviewInputV6,
    ) -> IntentInferenceRun:
        factory = self._reviewer_factory
        if factory is None:
            return _failed_intent_inference_run(
                trace_id=self._intent_trace_id(),
                model=self.config.reviewer.model or "unavailable",
                reason="Intent Agent model provider is not configured.",
            )
        diff = self.diff_store.load(self.snapshot)
        gateway = _V6IntentInferenceGateway(
            snapshot_id=self.snapshot.snapshot_id,
            session_id=self.session.session_id,
            base_revision=self.snapshot.base_sha,
            head_revision=self.snapshot.head_sha,
            backend=_GitReviewToolBackend(
                self.repository,
                self.snapshot.base_sha,
                self.snapshot.head_sha,
            ),
            timeout_seconds=self.config.reviewer_limits.tool_timeout_seconds,
            artifact_store=self.tool_artifacts,
        )
        return run_intent_inference(
            adapter=factory.create(),
            gateway=gateway,
            deterministic_request_summary=json.dumps(
                review_input.request.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            change_summary=_intent_change_summary(diff),
            explicit_intent={},
            missing_fields=("goal",),
            initial_observation_summaries={},
            trace_id=self._intent_trace_id(),
            resolved_base_revision=self.snapshot.base_sha,
            resolved_head_revision=self.snapshot.head_sha,
            model=self.config.reviewer.model or "fake-intent-analyst",
            goal_only=True,
            enforce_candidate_provenance=(
                review_input.intent_trust_policy
                is not IntentTrustPolicyV6.EVALUATION_TRUST_MODEL
            ),
        )

    def _intent_trace_id(self) -> str:
        return "intent-" + hashlib.sha256(
            (
                self.session.session_id
                + "\0"
                + self.snapshot.snapshot_id
            ).encode("utf-8", "strict")
        ).hexdigest()

    def _planning(
        self,
        _context: PipelineContextV6,
    ) -> tuple[SessionV6ArtifactRef, ...]:
        diff = self.diff_store.load(self.snapshot)
        intent = self._load_intent_envelope().packet
        changed_symbols = self._load_changed_symbols()
        risk_descriptor = _find_optional_artifact(
            self.workspace_store,
            self.snapshot,
            "Risk/risk.json",
        )
        if risk_descriptor is None:
            risk = self.risk_runtime.finalize(
                self.snapshot,
                diff.index,
                intent,
                model_decision=self._model_risk_decision(
                    intent=intent.to_dict(),
                    diff_index=diff.index.to_dict(),
                    quality=self._load_quality(),
                    changed_symbols=changed_symbols,
                ),
            )
            risk_descriptor = self.workspace_store.find_snapshot_artifact(
                self.snapshot,
                "Risk/risk.json",
            )
        else:
            risk = self.risk_runtime.load(self.snapshot)

        plan_descriptor = _find_optional_artifact(
            self.workspace_store,
            self.snapshot,
            "ReviewPlan/plan.json",
        )
        if plan_descriptor is None:
            plan = self.planning_runtime.plan(
                self.snapshot,
                risk,
                diff.index,
                changed_symbols,
            )
            plan_descriptor = self.workspace_store.find_snapshot_artifact(
                self.snapshot,
                "ReviewPlan/plan.json",
            )
        else:
            plan = self._load_plan(self.context)

        refs = [
            _artifact_ref("planning.risk", risk_descriptor),
            _artifact_ref("planning.review_plan", plan_descriptor),
        ]
        for slot, assignment in zip(
            fixed_reviewer_slots(plan.risk_level),
            plan.assignments,
        ):
            descriptor = self.workspace_store.find_snapshot_artifact(
                self.snapshot,
                f"ReviewPlan/Assignments/{slot.slot_id}.json",
            )
            refs.append(
                _artifact_ref(
                    f"planning.assignment:{assignment.assignment_id}",
                    descriptor,
                )
            )
        return tuple(refs)

    def _load_plan(self, _context: PipelineContextV6) -> ReviewPlan:
        descriptor = self.workspace_store.find_snapshot_artifact(
            self.snapshot,
            "ReviewPlan/plan.json",
        )
        raw = self.workspace_store.read_verified_artifact(
            self.snapshot,
            descriptor.artifact_id,
        )
        try:
            plan = ReviewPlan.from_json(raw)
        except ValueError as error:
            raise ProductRuntimeIntegrityError(
                "ReviewPlan artifact is invalid"
            ) from error
        if plan.snapshot_id != self.snapshot.snapshot_id:
            raise ProductRuntimeIntegrityError(
                "ReviewPlan Snapshot binding changed"
            )
        return plan

    def _run_reviewer(
        self,
        _context: PipelineContextV6,
        assignment: ReviewerAssignment,
    ) -> ReviewAggregationInput:
        factory = self._reviewer_factory
        if factory is None:
            execution = ReviewerExecutionResultV2(
                assignment_id=assignment.assignment_id,
                status="failed",
                output=None,
                reviewer_output=None,
                rejected_findings=(),
                error_code="reviewer_not_configured",
                active_elapsed_seconds=0.0,
            )
            return _aggregation_input(assignment, execution)

        diff = self.diff_store.load(self.snapshot)
        review_input = self._load_review_input()
        intent = self._load_intent_envelope().packet
        changed_symbols = self._load_changed_symbols()
        invocation = build_reviewer_invocation_v2(
            ReviewerContextInput(
                pr_id=self.session.workspace.pr_id,
                snapshot_id=self.snapshot.snapshot_id,
                base_sha=self.snapshot.base_sha,
                head_sha=self.snapshot.head_sha,
                request=review_input.request,
                developer_policy=self.developer_rules.policy_for_assignment(
                    assignment
                ),
                global_memory=GlobalMemoryFacade().freeze(()),
                intent=intent,
                assignment=assignment,
                quality_summary=self._load_quality(),
                changed_symbols=tuple(
                    symbol.to_dict() for symbol in changed_symbols.symbols
                ),
                diff_bytes=self.workspace_store.read_verified_artifact(
                    self.snapshot,
                    diff.patch.artifact_id,
                ),
                diff_index=diff.index,
                diff_artifact_id=diff.patch.artifact_id,
                available_artifacts=self._available_artifacts(diff, assignment),
                model=self.config.reviewer.model or "fake-reviewer-v2",
            )
        )
        gateway = ReviewToolGateway(
            snapshot_id=self.snapshot.snapshot_id,
            session_id=self.session.session_id,
            allowed_tools=assignment.permissions,
            backend=_GitReviewToolBackend(
                self.repository,
                self.snapshot.base_sha,
                self.snapshot.head_sha,
            ),
            timeout_seconds=self.config.reviewer_limits.tool_timeout_seconds,
            artifact_store=self.tool_artifacts,
        )
        loop = ReviewAgentLoopV2(
            adapter=factory.create(),
            gateway=gateway,
            projector=ToolResultProjector(self.tool_artifacts),
            journal=ExecutionJournal(
                self.workspace_store,
                self.session,
                assignment,
            ),
            assignment=assignment,
            invocation=invocation,
            limits=self.config.reviewer_limits,
            output_parser=ReviewerOutputParser(
                diff_index=diff.index,
                assignment=assignment,
            ),
        )
        execution = ReviewerExecutorV2().execute(
            assignment.assignment_id,
            loop,
        )
        return _aggregation_input(assignment, execution)

    def _load_review_input(self) -> ProductReviewInputV6:
        payload = self.workspace_store.read_verified_json(
            self.snapshot,
            self.request_descriptor.artifact_id,
        )
        return ProductReviewInputV6.from_dict(payload)

    def _load_intent_envelope(self) -> IntentVersionEnvelope:
        descriptor = self.workspace_store.find_snapshot_artifact(
            self.snapshot,
            _intent_path(self.session),
        )
        raw = self.workspace_store.read_verified_artifact(
            self.snapshot,
            descriptor.artifact_id,
        )
        try:
            envelope = IntentVersionEnvelope.from_json(raw)
        except ValueError as error:
            raise ProductRuntimeIntegrityError(
                "Intent artifact is invalid"
            ) from error
        if envelope.source_snapshot_id != self.snapshot.snapshot_id:
            raise ProductRuntimeIntegrityError(
                "Intent Snapshot binding changed"
            )
        return envelope

    def _load_changed_symbols(self) -> ChangedSymbolsV2:
        descriptor = self.workspace_store.find_snapshot_artifact(
            self.snapshot,
            "ChangedSymbols/changed-symbols.json",
        )
        payload = self.workspace_store.read_verified_json(
            self.snapshot,
            descriptor.artifact_id,
        )
        try:
            changed = changed_symbols_v2_from_dict(payload)
        except ValueError as error:
            raise ProductRuntimeIntegrityError(
                "ChangedSymbols artifact is invalid"
            ) from error
        if changed.snapshot_id != self.snapshot.snapshot_id:
            raise ProductRuntimeIntegrityError(
                "ChangedSymbols Snapshot binding changed"
            )
        return changed

    def _load_quality(self) -> dict[str, Any]:
        descriptor = self.workspace_store.find_snapshot_artifact(
            self.snapshot,
            "QualityGate/quality-gate.json",
        )
        payload = self.workspace_store.read_verified_json(
            self.snapshot,
            descriptor.artifact_id,
        )
        if type(payload) is not dict:
            raise ProductRuntimeIntegrityError(
                "Quality Gate artifact is invalid"
            )
        return payload

    def _model_risk_decision(
        self,
        *,
        intent: Mapping[str, Any],
        diff_index: Mapping[str, Any],
        quality: Mapping[str, Any],
        changed_symbols: ChangedSymbolsV2,
    ):
        factory = self._risk_factory
        if factory is None:
            return None
        request = ModelTurnRequest(
            system=RISK_MODEL_SYSTEM_PROMPT_V2,
            tools=[],
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "intent": dict(intent),
                            "diff_index": dict(diff_index),
                            "quality_gate": dict(quality),
                            "changed_symbols": [
                                symbol.to_dict()
                                for symbol in changed_symbols.symbols
                            ],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            ],
            tool_results=[],
            parameters={
                "model": self.config.risk.model or "risk-assessor-v2",
                "temperature": 0,
                "tool_choice": "none",
                "response_schema": "risk_decision_v2",
            },
        )
        adapter = factory.create()
        for _attempt in range(3):
            try:
                response = adapter.complete_turn(request)
            except Exception:
                continue
            if (
                response.kind is ModelResponseKind.FINAL
                and type(response.final_text) is str
            ):
                try:
                    return parse_risk_decision_v2(response.final_text)
                except ValueError:
                    continue
        return None

    def _available_artifacts(
        self,
        diff: DiffArtifact,
        assignment: ReviewerAssignment,
    ) -> tuple[AvailableArtifact, ...]:
        plan = self._load_plan(self.context)
        assignment_descriptor: ArtifactDescriptor | None = None
        for slot, candidate in zip(
            fixed_reviewer_slots(plan.risk_level),
            plan.assignments,
        ):
            if candidate.assignment_id != assignment.assignment_id:
                continue
            if candidate != assignment:
                raise ProductRuntimeIntegrityError(
                    "Reviewer Assignment binding changed"
                )
            assignment_descriptor = self.workspace_store.find_snapshot_artifact(
                self.snapshot,
                f"ReviewPlan/Assignments/{slot.slot_id}.json",
            )
            persisted = self.workspace_store.read_verified_artifact(
                self.snapshot,
                assignment_descriptor.artifact_id,
            )
            try:
                loaded = ReviewerAssignment.from_json(persisted)
            except ValueError as error:
                raise ProductRuntimeIntegrityError(
                    "Reviewer Assignment artifact is invalid"
                ) from error
            if loaded != assignment:
                raise ProductRuntimeIntegrityError(
                    "Reviewer Assignment artifact binding changed"
                )
            break
        if assignment_descriptor is None:
            raise ProductRuntimeIntegrityError(
                "Reviewer Assignment is not present in the immutable ReviewPlan"
            )
        quality = self.workspace_store.find_snapshot_artifact(
            self.snapshot,
            "QualityGate/quality-gate.json",
        )
        symbols = self.workspace_store.find_snapshot_artifact(
            self.snapshot,
            "ChangedSymbols/changed-symbols.json",
        )
        return (
            AvailableArtifact(
                artifact_id=diff.patch.artifact_id,
                kind="complete_diff",
                description="Complete immutable base-to-head Git diff.",
            ),
            AvailableArtifact(
                artifact_id=quality.artifact_id,
                kind="quality_gate",
                description="Deterministic local preflight quality result.",
            ),
            AvailableArtifact(
                artifact_id=symbols.artifact_id,
                kind="changed_symbols",
                description="Changed-symbol analysis for the immutable Snapshot.",
            ),
            AvailableArtifact(
                artifact_id=assignment_descriptor.artifact_id,
                kind="reviewer_assignment",
                description=(
                    "Complete immutable Assignment for this Reviewer, including "
                    "its authorized files, symbols, and hunks."
                ),
                assignment_ids=(assignment.assignment_id,),
            ),
        )


class _GitReviewToolBackend:
    def __init__(self, repository: Path, base_sha: str, head_sha: str) -> None:
        self.repository = Path(repository)
        self.base_sha = base_sha
        self.head_sha = head_sha

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        timeout_seconds: float,
    ) -> ToolBackendResult:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("tool timeout is invalid")
        args = dict(arguments)
        if tool_name == "read_range":
            return self._read_range(args, timeout_seconds)
        if tool_name == "compare_base_head":
            path = _required_text(args, "path")
            output = self._git(
                [
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--unified=80",
                    self.base_sha,
                    self.head_sha,
                    "--",
                    path,
                ],
                timeout_seconds,
                {0},
            )
            return ToolBackendResult(output, reacquirable=True)
        if tool_name == "search_code":
            query = _required_text(args, "query")
            revision = self._revision(args.get("revision", "head"))
            limit = _bounded_int(args.get("max_results", 20), 1, 200)
            matches = search_repository_text(
                self.repository,
                revision,
                query,
                max_results=limit,
            )
            return _json_tool_result(
                {"matches": [asdict(match) for match in matches]}
            )
        if tool_name == "list_symbols":
            path = _required_text(args, "path")
            revision = self._revision(args.get("revision", "head"))
            symbols = collect_python_symbols(
                self.repository,
                revision,
                [path],
            )
            return _json_tool_result(
                {"symbols": [asdict(symbol) for symbol in symbols]}
            )
        if tool_name == "inspect_symbol":
            name = _required_text(args, "name")
            revision = self._revision(args.get("revision", "head"))
            symbols = collect_python_symbols(self.repository, revision)
            matches = [
                symbol
                for symbol in symbols
                if symbol.qualified_name == name or symbol.name == name
            ]
            return _json_tool_result(
                {"symbols": [asdict(symbol) for symbol in matches]}
            )
        if tool_name == "find_references":
            name = _required_text(args, "name")
            revision = self._revision(args.get("revision", "head"))
            limit = _bounded_int(args.get("max_results", 20), 1, 200)
            matches = search_repository_text(
                self.repository,
                revision,
                name,
                max_results=limit,
            )
            return _json_tool_result(
                {"matches": [asdict(match) for match in matches]}
            )
        if tool_name == "read_commit_messages":
            requested_count = args.get(
                "max_commits",
                args.get("max_count", 20),
            )
            count = _bounded_int(requested_count, 1, 50)
            output = self._git(
                [
                    "log",
                    f"--max-count={count}",
                    "--format=%H%n%s%n%b%n%x1e",
                    f"{self.base_sha}..{self.head_sha}",
                ],
                timeout_seconds,
                {0},
            )
            return ToolBackendResult(output, reacquirable=True)
        raise ReviewToolFailure(
            code="unsupported_operation",
            retryable=False,
            message="The requested read-only Tool is unsupported",
        )

    def _read_range(
        self,
        arguments: Mapping[str, Any],
        timeout_seconds: float,
    ) -> ToolBackendResult:
        path = _required_text(arguments, "path")
        revision_label = arguments.get("revision", "head")
        revision = self._revision(revision_label)
        line_start = _bounded_int(arguments.get("line_start"), 1, 10_000_000)
        line_end = _bounded_int(arguments.get("line_end"), 1, 10_000_000)
        if line_end < line_start:
            raise ValueError("line_end precedes line_start")
        content = self._git(
            ["show", f"{revision}:{path}"],
            timeout_seconds,
            {0},
        )
        lines = content.splitlines()
        selected = "\n".join(lines[line_start - 1 : line_end])
        if selected:
            selected += "\n"
        return _json_tool_result(
            {
                "path": path,
                "revision": revision_label,
                "line_start": line_start,
                "line_end": line_end,
                "content": selected,
            }
        )

    def _revision(self, value: Any) -> str:
        if value == "base":
            return self.base_sha
        if value == "head":
            return self.head_sha
        raise ValueError("revision must be base or head")

    def _git(
        self,
        arguments: list[str],
        timeout_seconds: float,
        allowed_exit_codes: set[int],
    ) -> str:
        try:
            completed = subprocess.run(
                ["git", "--no-replace-objects", *arguments],
                cwd=self.repository,
                env=sanitized_git_environment(),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("Git Tool timed out") from error
        except OSError as error:
            raise ReviewToolFailure(
                code="tool_unavailable",
                retryable=True,
                message="Git Tool is unavailable",
            ) from error
        if completed.returncode not in allowed_exit_codes:
            raise ReviewToolFailure(
                code="git_command_failed",
                retryable=False,
                message="Git could not read the requested Snapshot data",
                exit_code=completed.returncode,
            )
        return completed.stdout


_INTENT_TOOL_NAMES = (
    "read_range",
    "compare_base_head",
    "search_code",
    "list_symbols",
    "inspect_symbol",
    "find_references",
    "read_commit_messages",
)


@dataclass(frozen=True)
class _IntentToolExecution:
    context_view: str
    observation_ids: list[str]


class _V6IntentInferenceGateway:
    """Adapt the v6 Tool gateway to the Intent Agent's narrow protocol."""

    def __init__(
        self,
        *,
        snapshot_id: str,
        session_id: str,
        base_revision: str,
        head_revision: str,
        backend: _GitReviewToolBackend,
        timeout_seconds: float,
        artifact_store: ToolResultArtifactStore,
    ) -> None:
        self.base_revision = base_revision
        self.head_revision = head_revision
        self._gateway = ReviewToolGateway(
            snapshot_id=snapshot_id,
            session_id=session_id,
            allowed_tools=_INTENT_TOOL_NAMES,
            backend=backend,
            timeout_seconds=timeout_seconds,
            artifact_store=artifact_store,
        )
        self._projector = ToolResultProjector(artifact_store)
        self._evidence: dict[str, IntentInferenceEvidence] = {}

    def execute_intent_calls(
        self,
        calls: tuple[ModelToolCall, ...],
    ) -> tuple[ModelToolResult, ...]:
        raw_results = tuple(
            self._gateway.execute(
                call.call_id,
                call.tool_name,
                call.arguments,
            )
            for call in calls
        )
        batch = self._projector.project_turn(raw_results)
        projected: list[ModelToolResult] = []
        for call, raw_result, projection in zip(
            calls,
            raw_results,
            batch.projections,
        ):
            content = serialize_tool_result_projection_v2(projection)
            observation_ids: list[str] = []
            if not projection.is_error:
                evidence = _intent_tool_evidence(
                    raw_result,
                    content,
                    base_revision=self.base_revision,
                    head_revision=self.head_revision,
                )
                self._evidence[evidence.observation_id] = evidence
                observation_ids.append(evidence.observation_id)
            projected.append(
                ModelToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    content=content,
                    observation_ids=observation_ids,
                    is_error=projection.is_error,
                )
            )
        return tuple(projected)

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> _IntentToolExecution:
        call_id = "intent-call-" + hashlib.sha256(
            canonical_json_bytes(
                {"tool_name": tool_name, "arguments": arguments}
            )
        ).hexdigest()
        result = self.execute_intent_calls(
            (ModelToolCall(call_id, tool_name, arguments),)
        )[0]
        if result.is_error:
            raise ValueError(result.content)
        return _IntentToolExecution(result.content, result.observation_ids)

    def intent_evidence(self) -> tuple[IntentInferenceEvidence, ...]:
        return tuple(self._evidence[key] for key in sorted(self._evidence))

    def intent_evidence_summaries(self) -> dict[str, str]:
        return {
            key: self._evidence[key].context_view
            for key in sorted(self._evidence)
        }


def _intent_tool_evidence(
    result,
    context_view: str,
    *,
    base_revision: str,
    head_revision: str,
) -> IntentInferenceEvidence:
    arguments = result.arguments
    path = arguments.get("path")
    if type(path) is not str:
        path = None
    if result.tool_name == "read_commit_messages":
        source = "git.read_commit_messages"
        revision = f"{base_revision}..{head_revision}"
    else:
        source = {
            "read_range": "git.read_range",
            "compare_base_head": "git.compare_base_head",
            "search_code": "git.search_code",
            "list_symbols": "repo_intelligence.list_symbols",
            "inspect_symbol": "repo_intelligence.inspect_symbol",
            "find_references": "repo_intelligence.find_references",
        }.get(result.tool_name, "review_tool")
        revision = str(arguments.get("revision", "head"))
    line_start = arguments.get("line_start")
    line_end = arguments.get("line_end")
    if type(line_start) is not int:
        line_start = None
    if type(line_end) is not int:
        line_end = None
    identity = {
        "snapshot_id": result.snapshot_id,
        "tool_name": result.tool_name,
        "arguments": result.arguments,
        "content_hash": hashlib.sha256(
            context_view.encode("utf-8", "strict")
        ).hexdigest(),
    }
    return IntentInferenceEvidence(
        observation_id="O-" + hashlib.sha256(
            canonical_json_bytes(identity)
        ).hexdigest(),
        source=source,
        revision=revision,
        path=path,
        line_start=line_start,
        line_end=line_end,
        context_view=context_view,
    )


def _load_outcome(
    workspace_store: PRWorkspaceStore,
    session: SessionWorkspace,
    *,
    manifest: SessionV6Manifest | None,
    reused: bool,
) -> ProductReviewOutcomeV6 | None:
    try:
        bundle = workspace_store.load_review_result_bundle(session.snapshot)
    except Exception as error:
        raise ProductRuntimeIntegrityError(
            "ReviewResult bundle failed integrity validation"
        ) from error
    if bundle is None:
        return None
    try:
        result = ReviewResult.from_json(bundle.review_result_bytes)
    except ValueError as error:
        raise ProductRuntimeIntegrityError(
            "ReviewResult wire artifact is invalid"
        ) from error
    if (
        result.pr_id != session.workspace.pr_id
        or result.snapshot_id != session.snapshot.snapshot_id
    ):
        raise ProductRuntimeIntegrityError("ReviewResult locator binding changed")
    markdown = render_review_result_markdown(result)
    expected_markdown = markdown.encode("utf-8")
    markdown_descriptor = _find_optional_artifact(
        workspace_store,
        session.snapshot,
        "Results/review.md",
    )
    if markdown_descriptor is None:
        workspace_store.publish_create_only(
            session.snapshot,
            "Results/review.md",
            expected_markdown,
        )
    else:
        persisted_markdown = workspace_store.read_verified_artifact(
            session.snapshot,
            markdown_descriptor.artifact_id,
        )
        if persisted_markdown != expected_markdown:
            raise ProductRuntimeIntegrityError(
                "Review Markdown is not a pure ReviewResult render"
            )
    return ProductReviewOutcomeV6(
        pr_id=result.pr_id,
        snapshot_id=result.snapshot_id,
        session_id=session.session_id,
        review_result=result,
        review_result_json=bundle.review_result_bytes.decode("utf-8", "strict"),
        review_markdown=markdown,
        manifest=manifest,
        reused=reused,
    )


def _intent_change_summary(diff: DiffArtifact) -> str:
    payload = {
        "base_sha": diff.index.base_sha,
        "head_sha": diff.index.head_sha,
        "changed_file_count": len(diff.index.files),
        "files": [
            {
                "path": item.path,
                "previous_path": item.previous_path,
                "status": item.status,
                "additions": item.additions,
                "deletions": item.deletions,
                "binary": item.binary,
            }
            for item in diff.index.files
        ],
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _failed_intent_inference_run(
    *,
    trace_id: str,
    model: str,
    reason: str,
) -> IntentInferenceRun:
    return IntentInferenceRun(
        result=IntentInferenceResult(
            candidates=[],
            uncertainties=[reason],
            summary="Intent inference failed: " + reason,
        ),
        trace=IntentInferenceTrace(
            trace_id=trace_id,
            turns=[],
            tool_call_count=0,
            final_status="failed",
            deficiencies=[reason],
        ),
        provider_name="review-agent",
        model=model,
        response_error=reason,
    )


def _request_path(session: SessionWorkspace) -> str:
    return _REQUEST_PREFIX + session.session_id[8:40] + ".json"


def _intent_path(session: SessionWorkspace) -> str:
    return _INTENT_PREFIX + session.session_id[8:40] + ".json"


def _artifact_ref(
    logical_name: str,
    descriptor: ArtifactDescriptor,
) -> SessionV6ArtifactRef:
    return SessionV6ArtifactRef(
        logical_name=logical_name,
        artifact_id=descriptor.artifact_id,
        relative_path=descriptor.relative_path,
        sha256=descriptor.sha256,
    )


def _find_optional_artifact(
    store: PRWorkspaceStore,
    snapshot: SnapshotWorkspace,
    relative_path: str,
) -> ArtifactDescriptor | None:
    try:
        return store.find_snapshot_artifact(snapshot, relative_path)
    except Exception as error:
        if not (snapshot.path / Path(relative_path)).exists():
            return None
        raise ProductRuntimeIntegrityError(
            f"Snapshot Artifact is invalid: {relative_path}"
        ) from error


def _aggregation_input(
    assignment: ReviewerAssignment,
    execution: ReviewerExecutionResultV2,
) -> ReviewAggregationInput:
    return ReviewAggregationInput(
        reviewer_id="reviewer-" + assignment.assignment_id[4:20],
        execution=execution,
    )


def _json_tool_result(payload: Mapping[str, Any]) -> ToolBackendResult:
    return ToolBackendResult(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        reacquirable=True,
    )


def _required_text(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} is invalid")
    return value


def _bounded_int(value: Any, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("integer Tool argument is out of range")
    return value


__all__ = [
    "IntentTrustPolicyV6",
    "PRODUCT_REVIEW_REQUEST_SCHEMA",
    "ProductReviewInputV6",
    "ProductReviewOutcomeV6",
    "ProductRuntimeConfigV6",
    "ProductRuntimeError",
    "ProductRuntimeInfrastructureError",
    "ProductRuntimeIntegrityError",
    "ProductRuntimeUsageError",
    "resume_product_review_v6",
    "start_product_review_v6",
]

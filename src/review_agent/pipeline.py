from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from review_agent.agent_loop import AgentLoopRun, agent_loop_run_to_dict, run_reviewer_agent_loop
from review_agent.artifacts import artifact_schema
from review_agent.attempts import AttemptWorkspace
from review_agent.brief import ReviewBrief, build_review_brief, review_brief_to_dict
from review_agent.checkpoint import CheckpointStore
from review_agent.completion import CompletionResult, check_completion, completion_to_dict
from review_agent.evidence import EvidenceReconciliation, reconcile_evidence, reconciliation_to_dict
from review_agent.final_risk import FinalRiskAssessment, final_risk_to_dict, reassess_final_risk
from review_agent.git_repo import ChangeSummary, collect_change_summary
from review_agent.hydration import (
    assignments_from_dict,
    completion_from_dict,
    final_risk_from_dict,
    intent_from_dict,
    quality_results_from_dict,
    reconciliation_from_dict,
    repository_intelligence_from_dict,
    review_brief_from_dict,
    review_request_from_dict,
    reviewer_execution_from_artifacts,
    risk_assessment_from_dict,
    risk_packet_from_dict,
)
from review_agent.intent import build_intent_packet
from review_agent.incremental import (
    IncrementalPriorityMap,
    build_incremental_priority_map_from_summary,
    incremental_priority_from_dict,
    incremental_priority_to_dict,
)
from review_agent.model_adapter_factory import (
    AdapterConfigError,
    ModelAdapterConfig,
    ModelAdapterFactory,
    build_model_adapter_factory_from_config,
)
from review_agent.models import (
    Assignment,
    IntentPacket,
    QualityGateResult,
    ReviewRequest,
    ReviewerResult,
    RiskAssessment,
    RiskAssessmentPacket,
)
from review_agent.observations import ObservationStore
from review_agent.orchestrator import MultiReviewerRun, ReviewerExecution, multi_reviewer_run_to_dict
from review_agent.quality import detect_quality_gates, run_python_compile_gate
from review_agent.repository_intelligence import (
    RepositoryIntelligenceSnapshot,
    build_repository_intelligence,
    repository_intelligence_raw_json,
    repository_intelligence_to_dict,
    summarize_repository_intelligence,
)
from review_agent.reporting import render_review_brief_markdown
from review_agent.reviewer import reviewer_result_to_dict, run_single_reviewer
from review_agent.review_contract import validate_reviewer_completion
from review_agent.risk import LocalRiskAssessor, build_risk_packet
from review_agent.run_state import RunPhase, RunState, RunStatus
from review_agent.runtime import build_assignments
from review_agent.session import PhaseStatus, ReviewExecutionConfig, SessionManifest
from review_agent.session_store import SessionStore
from review_agent.tool_gateway import ToolGateway


PHASE_MESSAGES = {
    RunPhase.PREFLIGHT: "Preflight completed",
    RunPhase.REPOSITORY_INTELLIGENCE: "Repository intelligence collected",
    RunPhase.REVIEWERS: "Reviewer execution completed",
    RunPhase.RECONCILIATION: "Evidence reconciliation completed",
    RunPhase.COMPLETION: "Completion check completed",
    RunPhase.FINAL_RISK: "Final risk reassessment completed",
    RunPhase.REPORTING: "Reporting completed",
}


class PipelineError(RuntimeError):
    pass


class PipelineConfigurationError(PipelineError):
    pass


class PipelineHydrationError(PipelineError):
    def __init__(self, phase: RunPhase, message: str) -> None:
        super().__init__(f"unable to hydrate {phase.value}: {message}")
        self.phase = phase


class PipelineStageError(PipelineError):
    def __init__(self, phase: RunPhase, error: Exception) -> None:
        super().__init__(f"{phase.value} failed: {type(error).__name__}: {error}")
        self.phase = phase
        self.cause = error


@dataclass
class PipelineContext:
    repository: Path
    checkpoint_store: CheckpointStore
    session_store: SessionStore
    request: ReviewRequest | None = None
    change_summary: ChangeSummary | None = None
    intent: IntentPacket | None = None
    risk_packet: RiskAssessmentPacket | None = None
    risk_assessment: RiskAssessment | None = None
    incremental_priority: IncrementalPriorityMap | None = None
    assignments: list[Assignment] = field(default_factory=list)
    quality_results: list[QualityGateResult] = field(default_factory=list)
    repository_intelligence: RepositoryIntelligenceSnapshot | None = None
    repository_observations: ObservationStore | None = None
    reviewer_observations: dict[int, ObservationStore] = field(default_factory=dict)
    reviewer_executions: list[ReviewerExecution] = field(default_factory=list)
    reviewer_result: ReviewerResult | None = None
    multi_run: MultiReviewerRun | None = None
    reconciliation: EvidenceReconciliation | None = None
    completion: CompletionResult | None = None
    final_risk: FinalRiskAssessment | None = None
    brief: ReviewBrief | None = None
    compatibility_warnings: list[str] = field(default_factory=list)

    @property
    def manifest(self) -> SessionManifest:
        return self.session_store.load()

    @property
    def revision_binding(self) -> str:
        revisions = self.manifest.revisions
        return f"{revisions.resolved_base_sha}..{revisions.resolved_head_sha}"

    @property
    def observation_revision_bindings(self) -> set[str]:
        revisions = self.manifest.revisions
        return {
            self.revision_binding,
            f"base@{revisions.resolved_base_sha}",
            f"head@{revisions.resolved_head_sha}",
        }


@dataclass(frozen=True)
class PipelineResult:
    context: PipelineContext
    starting_phase: RunPhase
    reused_phases: tuple[RunPhase, ...]


class ReviewPipeline:
    def __init__(
        self,
        *,
        repository: Path,
        checkpoint_store: CheckpointStore,
        session_store: SessionStore,
        request: ReviewRequest | None = None,
        collect_change_summary_fn: Callable[..., ChangeSummary] = collect_change_summary,
        adapter_factory_builder: Callable[[ModelAdapterConfig], ModelAdapterFactory | None] = build_model_adapter_factory_from_config,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.context = PipelineContext(
            repository=Path(repository),
            checkpoint_store=checkpoint_store,
            session_store=session_store,
            request=request,
        )
        self._collect_change_summary = collect_change_summary_fn
        self._build_adapter_factory = adapter_factory_builder
        self._clock = clock or _utc_now

    def execute(
        self,
        *,
        starting_phase: RunPhase = RunPhase.PREFLIGHT,
        resuming: bool = False,
    ) -> PipelineResult:
        phases = tuple(PHASE_MESSAGES)
        if starting_phase not in phases:
            raise ValueError("starting_phase must be a persisted Pipeline phase")
        starting_index = phases.index(starting_phase)
        reused: list[RunPhase] = []

        for phase in phases[:starting_index]:
            self.load_phase(phase)
            reused.append(phase)

        for phase in phases[starting_index:]:
            try:
                self._prepare_phase_attempt(phase, resuming=resuming)
                artifacts = self._run_phase(phase)
                self.context.session_store.mark_phase_completed(
                    phase,
                    artifacts,
                    self._clock(),
                )
                self._write_compatibility_state(PHASE_MESSAGES[phase])
            except PipelineConfigurationError as error:
                self._record_failure(phase, error)
                raise
            except Exception as error:
                self._record_failure(phase, error)
                if isinstance(error, PipelineError):
                    raise
                raise PipelineStageError(phase, error) from error

        self.context.session_store.mark_session_completed(self._clock())
        self._write_compatibility_state("Review completed")
        return PipelineResult(
            context=self.context,
            starting_phase=starting_phase,
            reused_phases=tuple(reused),
        )

    def load_phase(self, phase: RunPhase) -> None:
        manifest = self.context.manifest
        checkpoint = manifest.phases[phase.value]
        expected_schemas = {
            name: artifact_schema(name) for name in checkpoint.artifacts
        }
        validation = self.context.session_store.validate_phase(
            phase,
            expected_schemas,
        )
        if not validation.valid:
            raise PipelineHydrationError(phase, validation.reason or "invalid phase")
        try:
            self._load_phase(phase)
        except Exception as error:
            raise PipelineHydrationError(phase, str(error)) from error

    def validate_completed_reviewer_tasks(self) -> dict[str, str]:
        manifest = self.context.manifest
        checkpoint = manifest.phases[RunPhase.REVIEWERS.value]
        failures: dict[str, str] = {}
        for task_name, task in checkpoint.tasks.items():
            if task.status is not PhaseStatus.COMPLETED:
                continue
            try:
                index = int(task_name.removeprefix("reviewer-"))
                self._load_reviewer_task(index)
            except Exception as error:
                failures[task_name] = f"{type(error).__name__}: {error}"
        return failures

    def _prepare_phase_attempt(self, phase: RunPhase, *, resuming: bool) -> None:
        manifest = self.context.manifest
        checkpoint = manifest.phases[phase.value]
        if resuming and checkpoint.status is PhaseStatus.RUNNING:
            manifest = self.context.session_store.restart_running_phase(
                phase,
                self._clock(),
            )
        else:
            manifest = self.context.session_store.mark_phase_running(
                phase,
                self._clock(),
            )
        checkpoint = manifest.phases[phase.value]
        preserve: tuple[str, ...] = ()
        if phase is RunPhase.REVIEWERS:
            preserve = tuple(
                artifact_name
                for task in checkpoint.tasks.values()
                if task.status is PhaseStatus.COMPLETED
                for artifact_name in task.artifacts
            )
        self.context.session_store.discard_uncommitted_phase_artifacts(
            phase,
            preserve,
            self._clock(),
        )

    def _run_phase(self, phase: RunPhase) -> dict[str, str]:
        dispatch = {
            RunPhase.PREFLIGHT: self._run_preflight,
            RunPhase.REPOSITORY_INTELLIGENCE: self._run_repository_intelligence,
            RunPhase.REVIEWERS: self._run_reviewers,
            RunPhase.RECONCILIATION: self._run_reconciliation,
            RunPhase.COMPLETION: self._run_completion,
            RunPhase.FINAL_RISK: self._run_final_risk,
            RunPhase.REPORTING: self._run_reporting,
        }
        return dispatch[phase]()

    def _load_phase(self, phase: RunPhase) -> None:
        dispatch = {
            RunPhase.PREFLIGHT: self._load_preflight,
            RunPhase.REPOSITORY_INTELLIGENCE: self._load_repository_intelligence,
            RunPhase.REVIEWERS: self._load_reviewers,
            RunPhase.RECONCILIATION: self._load_reconciliation,
            RunPhase.COMPLETION: self._load_completion,
            RunPhase.FINAL_RISK: self._load_final_risk,
            RunPhase.REPORTING: self._load_reporting,
        }
        dispatch[phase]()

    def _run_preflight(self) -> dict[str, str]:
        request = _required(self.context.request, "review request")
        manifest = self.context.manifest
        revisions = manifest.revisions
        summary = self._collect_change_summary(
            self.context.repository,
            revisions.resolved_base_sha,
            revisions.resolved_head_sha,
        )
        intent = build_intent_packet(request, summary)
        gates = detect_quality_gates(
            self.context.repository,
            revision=revisions.resolved_head_sha,
        )
        quality_results: list[QualityGateResult] = []
        if "python_compile" in gates:
            quality_results.append(
                run_python_compile_gate(
                    self.context.repository,
                    revision=revisions.resolved_head_sha,
                )
            )
        risk_packet = build_risk_packet(
            summary,
            intent,
            {result.name: result.status for result in quality_results},
        )
        risk = LocalRiskAssessor().assess(risk_packet)
        incremental_priority: IncrementalPriorityMap | None = None
        if manifest.incremental_from_sha is not None:
            incremental_summary = self._collect_change_summary(
                self.context.repository,
                manifest.incremental_from_sha,
                revisions.resolved_head_sha,
            )
            incremental_priority = build_incremental_priority_map_from_summary(
                incremental_summary,
                from_revision=manifest.incremental_from_sha,
                to_revision=revisions.resolved_head_sha,
            )
        assignments = [
            replace(
                assignment,
                initial_context=replace(
                    assignment.initial_context,
                    changed_files=list(summary.changed_files),
                    diff_ranges=(
                        [
                            f"incremental:{path}"
                            for path in incremental_priority.changed_files
                        ]
                        + [f"full:{path}" for path in summary.changed_files]
                        if incremental_priority is not None
                        else list(assignment.initial_context.diff_ranges)
                    ),
                ),
            )
            for assignment in build_assignments(risk)
        ]

        workspace = self._phase_workspace(RunPhase.PREFLIGHT)
        workspace.write_json("request.json", asdict(request))
        workspace.write_json("intent.json", asdict(intent))
        workspace.write_json("risk_packet.json", asdict(risk_packet))
        workspace.write_json("risk.json", asdict(risk))
        workspace.write_json(
            "assignments.json",
            {"assignments": [asdict(item) for item in assignments]},
        )
        workspace.write_json(
            "quality_gates.json",
            {"results": [asdict(item) for item in quality_results]},
        )
        if incremental_priority is not None:
            workspace.write_json(
                "incremental_priority.json",
                incremental_priority_to_dict(incremental_priority),
            )
        file_specs: dict[str, tuple[str, str]] = {
            "request": ("request.json", "request.json"),
            "intent": ("intent.json", "intent.json"),
            "risk_packet": ("risk_packet.json", "risk_packet.json"),
            "risk": ("risk.json", "risk.json"),
            "assignments": ("assignments.json", "assignments.json"),
            "quality_gates": ("quality_gates.json", "quality_gates.json"),
        }
        if incremental_priority is not None:
            file_specs["incremental_priority"] = (
                "incremental_priority.json",
                "incremental_priority.json",
            )
        artifacts = self._commit_files(
            RunPhase.PREFLIGHT,
            workspace,
            file_specs,
        )
        self.context.change_summary = summary
        self.context.intent = intent
        self.context.risk_packet = risk_packet
        self.context.risk_assessment = risk
        self.context.incremental_priority = incremental_priority
        self.context.assignments = assignments
        self.context.quality_results = quality_results
        return artifacts

    def _load_preflight(self) -> None:
        manifest = self.context.manifest
        request = review_request_from_dict(self._read_json_artifact("request"))
        intent = intent_from_dict(self._read_json_artifact("intent"))
        risk_packet = risk_packet_from_dict(self._read_json_artifact("risk_packet"))
        risk = risk_assessment_from_dict(self._read_json_artifact("risk"))
        assignments = assignments_from_dict(self._read_json_artifact("assignments"))
        quality = quality_results_from_dict(self._read_json_artifact("quality_gates"))
        has_incremental_priority = "incremental_priority" in manifest.artifacts
        if manifest.incremental_from_sha is not None and not has_incremental_priority:
            raise ValueError(
                "HEAD_MOVED preflight is missing its incremental priority map"
            )
        if manifest.incremental_from_sha is None and has_incremental_priority:
            raise ValueError(
                "non-HEAD drift preflight must not contain an incremental priority map"
            )
        incremental_priority = (
            incremental_priority_from_dict(
                self._read_json_artifact("incremental_priority")
            )
            if has_incremental_priority
            else None
        )
        if incremental_priority is not None and (
            incremental_priority.from_revision.casefold()
            != manifest.incremental_from_sha.casefold()
            or incremental_priority.to_revision.casefold()
            != manifest.revisions.resolved_head_sha.casefold()
        ):
            raise ValueError(
                "incremental priority map does not match the Session lineage"
            )
        self.context.request = request
        self.context.intent = intent
        self.context.risk_packet = risk_packet
        self.context.risk_assessment = risk
        self.context.assignments = assignments
        self.context.quality_results = quality
        self.context.incremental_priority = incremental_priority
        self.context.change_summary = _change_summary_from_packet(risk_packet)

    def _run_repository_intelligence(self) -> dict[str, str]:
        summary = _required(self.context.change_summary, "change summary")
        manifest = self.context.manifest
        snapshot = build_repository_intelligence(
            repo=self.context.repository,
            base_revision=manifest.revisions.resolved_base_sha,
            head_revision=manifest.revisions.resolved_head_sha,
            changed_files=summary.changed_files,
        )
        workspace = self._phase_workspace(RunPhase.REPOSITORY_INTELLIGENCE)
        workspace.write_json(
            "repository_intelligence.json",
            repository_intelligence_to_dict(snapshot),
        )
        observation_root = workspace.path / "obs"
        observations = ObservationStore(observation_root)
        observations.record(
            source="repo_intelligence.snapshot",
            revision=self.context.revision_binding,
            path=None,
            line_start=None,
            line_end=None,
            raw_content=repository_intelligence_raw_json(snapshot),
            context_view=summarize_repository_intelligence(snapshot),
        )
        artifacts = self._commit_files(
            RunPhase.REPOSITORY_INTELLIGENCE,
            workspace,
            {
                "repository_intelligence": (
                    "repository_intelligence.json",
                    "repository_intelligence.json",
                )
            },
        )
        observation_path = self._commit_observation_store(
            phase=RunPhase.REPOSITORY_INTELLIGENCE,
            workspace=workspace,
            source=observations,
            destination_root="observation_stores/repository",
            artifact_name="repository_observations",
        )
        artifacts["repository_observations"] = observation_path
        self.context.repository_intelligence = snapshot
        self.context.repository_observations = ObservationStore.load(
            self.context.checkpoint_store.run_dir / "observation_stores" / "repository",
            {self.context.revision_binding},
        )
        return artifacts

    def _load_repository_intelligence(self) -> None:
        self.context.repository_intelligence = repository_intelligence_from_dict(
            self._read_json_artifact("repository_intelligence")
        )
        if "repository_observations" in self.context.manifest.artifacts:
            self.context.repository_observations = self._load_observation_artifact(
                "repository_observations"
            )
        elif "observations" in self.context.manifest.artifacts:
            # Batch A stored only the final aggregate ObservationStore.
            self.context.repository_observations = self._load_observation_artifact(
                "observations"
            )
        else:
            raise ValueError("repository observation artifact is unavailable")

    def _run_reviewers(self) -> dict[str, str]:
        manifest = self.context.manifest
        config = manifest.execution
        try:
            adapter_factory = self._build_adapter_factory(
                ModelAdapterConfig(
                    provider_name=config.reviewer_provider,
                    model=config.reviewer_model,
                    base_url=config.reviewer_base_url,
                    api_key_env=config.reviewer_api_key_env,
                )
            )
        except AdapterConfigError as error:
            raise PipelineConfigurationError(str(error)) from error
        if config.reviewer_mode == "multi" and adapter_factory is None:
            raise PipelineConfigurationError(
                "--reviewer-mode multi requires --reviewer-provider "
                "fake or openai-compatible"
            )
        if adapter_factory is None or not self.context.assignments:
            self.context.reviewer_executions = []
            self.context.reviewer_result = None
            self.context.multi_run = None
            return {}

        assignments = (
            self.context.assignments
            if config.reviewer_mode == "multi"
            else self.context.assignments[:1]
        )
        task_names = [f"reviewer-{index}" for index in range(len(assignments))]
        self.context.session_store.initialize_reviewer_tasks(
            task_names,
            self._clock(),
        )
        executions: list[ReviewerExecution] = []
        for index, assignment in enumerate(assignments):
            task_name = f"reviewer-{index}"
            task = self.context.manifest.phases[RunPhase.REVIEWERS.value].tasks[
                task_name
            ]
            if task.status is PhaseStatus.COMPLETED:
                execution, observation_store = self._load_reviewer_task(index)
                executions.append(execution)
                self.context.reviewer_observations[index] = observation_store
                continue
            if task.status is PhaseStatus.RUNNING:
                self.context.session_store.restart_running_reviewer_task(
                    task_name,
                    self._clock(),
                )
            else:
                self.context.session_store.mark_reviewer_task_running(
                    task_name,
                    self._clock(),
                )
            try:
                execution, observation_store, artifact_names = self._execute_reviewer(
                    index=index,
                    assignment=assignment,
                    adapter_factory=adapter_factory,
                )
                self.context.session_store.mark_reviewer_task_completed(
                    task_name,
                    artifact_names,
                    self._clock(),
                )
            except Exception as error:
                self.context.session_store.mark_reviewer_task_failed(
                    task_name,
                    f"{type(error).__name__}: {error}",
                    self._clock(),
                )
                raise
            executions.append(execution)
            self.context.reviewer_observations[index] = observation_store

        self.context.reviewer_executions = list(executions)
        if config.reviewer_mode == "multi":
            multi_run = MultiReviewerRun(executions=executions)
            workspace = self._phase_workspace(RunPhase.REVIEWERS)
            workspace.write_json(
                "multi_reviewer_result.json",
                multi_reviewer_run_to_dict(multi_run),
            )
            self._commit_files(
                RunPhase.REVIEWERS,
                workspace,
                {
                    "multi_reviewer": (
                        "multi_reviewer_result.json",
                        "multi_reviewer_result.json",
                    )
                },
            )
            self.context.multi_run = multi_run
            self.context.reviewer_result = None
        else:
            self.context.multi_run = None
            self.context.reviewer_result = executions[0].result

        current = self.context.manifest
        return {
            name: descriptor.path
            for name, descriptor in current.artifacts.items()
            if descriptor.phase is RunPhase.REVIEWERS
        }

    def _execute_reviewer(
        self,
        *,
        index: int,
        assignment: Assignment,
        adapter_factory: ModelAdapterFactory,
    ) -> tuple[ReviewerExecution, ObservationStore, tuple[str, ...]]:
        manifest = self.context.manifest
        phase_attempt = manifest.phases[RunPhase.REVIEWERS.value].attempts
        workspace = AttemptWorkspace(
            self.context.checkpoint_store.run_dir,
            RunPhase.REVIEWERS,
            phase_attempt,
            reviewer_index=index,
        )
        workspace.prepare()
        observations = ObservationStore(workspace.path)
        gateway = ToolGateway(
            repository_path=self.context.repository,
            base_revision=manifest.revisions.resolved_base_sha,
            head_revision=manifest.revisions.resolved_head_sha,
            observation_store=observations,
        )
        summary = _required(self.context.change_summary, "change summary")
        for changed_file in summary.changed_files:
            gateway.execute("compare_base_head", {"path": changed_file})
        reviewer_observations = self._authorized_observation_summaries(observations)
        intent = _required(self.context.intent, "intent")
        trace_id = f"{manifest.review_id}-reviewer-{index}"
        model = _reviewer_invocation_model(manifest.execution)
        loop_run: AgentLoopRun | None = None
        if manifest.execution.reviewer_loop == "agent-loop":
            loop_run = run_reviewer_agent_loop(
                adapter=adapter_factory.create(),
                gateway=gateway,
                assignment=assignment,
                intent=intent,
                diff_excerpt=self._review_diff_excerpt(summary),
                observations=reviewer_observations,
                trace_id=trace_id,
                model=model,
            )
            envelope = loop_run.envelope
            response = loop_run.response
            result = loop_run.result
        else:
            run = run_single_reviewer(
                adapter=adapter_factory.create(),
                assignment=assignment,
                intent=intent,
                diff_excerpt=self._review_diff_excerpt(summary),
                observations=reviewer_observations,
                trace_id=trace_id,
                model=model,
            )
            envelope = run.envelope
            response = run.response
            result = run.result

        names = _reviewer_artifact_names(
            index,
            single=manifest.execution.reviewer_mode == "single",
            include_trace=loop_run is not None,
        )
        workspace.write_json(f"{names.envelope}.json", asdict(envelope))
        workspace.write_json(
            f"{names.raw_response}.json",
            {
                "provider_name": response.provider_name,
                "model": response.model,
                "content": response.content,
                "raw": response.raw,
            },
        )
        result_filename = _reviewer_artifact_filename(names.result)
        workspace.write_json(result_filename, reviewer_result_to_dict(result))
        file_specs: dict[str, tuple[str, str]] = {
            names.envelope: (f"{names.envelope}.json", f"{names.envelope}.json"),
            names.raw_response: (
                f"{names.raw_response}.json",
                f"{names.raw_response}.json",
            ),
            names.result: (result_filename, result_filename),
        }
        if loop_run is not None and names.trace is not None:
            workspace.write_json(
                f"{names.trace}.json",
                agent_loop_run_to_dict(loop_run)["trace"],
            )
            file_specs[names.trace] = (f"{names.trace}.json", f"{names.trace}.json")
        committed = self._commit_files(RunPhase.REVIEWERS, workspace, file_specs)
        observation_name = f"reviewer_{index}_observations"
        observation_path = self._commit_observation_store(
            phase=RunPhase.REVIEWERS,
            workspace=workspace,
            source=observations,
            destination_root=f"observation_stores/reviewer-{index}",
            artifact_name=observation_name,
        )
        committed[observation_name] = observation_path
        authoritative_store = self._load_observation_artifact(observation_name)
        execution = ReviewerExecution(
            reviewer_index=index,
            trace_id=trace_id,
            assignment=assignment,
            envelope=envelope,
            response=response,
            result=result,
        )
        return execution, authoritative_store, tuple(committed)

    def _load_reviewers(self) -> None:
        manifest = self.context.manifest
        if manifest.execution.reviewer_provider == "none" or not self.context.assignments:
            self.context.reviewer_executions = []
            self.context.reviewer_result = None
            self.context.multi_run = None
            return
        assignment_count = (
            len(self.context.assignments)
            if manifest.execution.reviewer_mode == "multi"
            else min(1, len(self.context.assignments))
        )
        executions: list[ReviewerExecution] = []
        checkpoint = manifest.phases[RunPhase.REVIEWERS.value]
        if checkpoint.tasks:
            for index in range(assignment_count):
                execution, observations = self._load_reviewer_task(index)
                executions.append(execution)
                self.context.reviewer_observations[index] = observations
        else:
            legacy_observations = (
                self._load_observation_artifact("observations")
                if "observations" in manifest.artifacts
                else _required(
                    self.context.repository_observations,
                    "legacy observations",
                )
            )
            for index in range(assignment_count):
                execution = self._load_reviewer_execution(index)
                executions.append(execution)
                self.context.reviewer_observations[index] = legacy_observations
        if manifest.execution.reviewer_mode == "multi":
            self.context.multi_run = MultiReviewerRun(executions=executions)
            self.context.reviewer_result = None
        else:
            self.context.multi_run = None
            self.context.reviewer_result = executions[0].result
        self.context.reviewer_executions = list(executions)

    def _load_reviewer_task(
        self,
        index: int,
    ) -> tuple[ReviewerExecution, ObservationStore]:
        manifest = self.context.manifest
        checkpoint = manifest.phases[RunPhase.REVIEWERS.value]
        task_name = f"reviewer-{index}"
        task = checkpoint.tasks.get(task_name)
        if task is None or task.status is not PhaseStatus.COMPLETED:
            raise ValueError(f"reviewer task is not reusable: {task_name}")
        for artifact_name in task.artifacts:
            descriptor = manifest.artifacts.get(artifact_name)
            if descriptor is None or not self.context.session_store.validate_artifact(
                descriptor
            ):
                raise ValueError(f"reviewer task artifact is invalid: {artifact_name}")
            if descriptor.schema != artifact_schema(artifact_name):
                raise ValueError(f"reviewer task artifact schema is invalid: {artifact_name}")
        names = _reviewer_artifact_names(
            index,
            single=manifest.execution.reviewer_mode == "single",
            include_trace=manifest.execution.reviewer_loop == "agent-loop",
        )
        observations = self._load_observation_artifact(
            f"reviewer_{index}_observations"
        )
        execution = self._load_reviewer_execution(index)
        validation = validate_reviewer_completion(
            execution.assignment,
            execution.result,
            set(self._authorized_observation_summaries(observations)),
        )
        if not validation.accepted:
            raise ValueError(
                f"reviewer task completion is invalid: {task_name}: "
                + "; ".join(validation.deficiencies)
            )
        return execution, observations

    def _load_reviewer_execution(self, index: int) -> ReviewerExecution:
        manifest = self.context.manifest
        names = _reviewer_artifact_names(
            index,
            single=manifest.execution.reviewer_mode == "single",
            include_trace=manifest.execution.reviewer_loop == "agent-loop",
        )
        envelope_payload = self._read_json_artifact(names.envelope)
        response_payload = self._read_json_artifact(names.raw_response)
        result_payload = self._read_json_artifact(names.result)
        parameters = envelope_payload.get("parameters")
        if not isinstance(parameters, dict) or not isinstance(
            parameters.get("trace_id"), str
        ):
            raise ValueError("reviewer envelope is missing trace_id")
        return reviewer_execution_from_artifacts(
            reviewer_index=index,
            trace_id=parameters["trace_id"],
            assignment=self.context.assignments[index],
            envelope_payload=envelope_payload,
            response_payload=response_payload,
            result_payload=result_payload,
        )

    def _run_reconciliation(self) -> dict[str, str]:
        reconciliation = reconcile_evidence(
            executions=self.context.reviewer_executions,
            authorized_observation_ids=set(self._authorized_observation_summaries()),
        )
        workspace = self._phase_workspace(RunPhase.RECONCILIATION)
        workspace.write_json(
            "reconciliation.json",
            reconciliation_to_dict(reconciliation),
        )
        artifacts = self._commit_files(
            RunPhase.RECONCILIATION,
            workspace,
            {
                "reconciliation": (
                    "reconciliation.json",
                    "reconciliation.json",
                )
            },
        )
        self.context.reconciliation = reconciliation
        return artifacts

    def _load_reconciliation(self) -> None:
        checkpoint = self.context.manifest.phases[RunPhase.RECONCILIATION.value]
        if not checkpoint.artifacts:
            self.context.reconciliation = None
            return
        self.context.reconciliation = reconciliation_from_dict(
            self._read_json_artifact("reconciliation")
        )

    def _run_completion(self) -> dict[str, str]:
        completion = check_completion(
            intent=_required(self.context.intent, "intent"),
            quality_results=self.context.quality_results,
            executions=self.context.reviewer_executions,
            reconciliation=_required(
                self.context.reconciliation,
                "evidence reconciliation",
            ),
        )
        workspace = self._phase_workspace(RunPhase.COMPLETION)
        workspace.write_json("completion.json", completion_to_dict(completion))
        artifacts = self._commit_files(
            RunPhase.COMPLETION,
            workspace,
            {"completion": ("completion.json", "completion.json")},
        )
        self.context.completion = completion
        return artifacts

    def _load_completion(self) -> None:
        checkpoint = self.context.manifest.phases[RunPhase.COMPLETION.value]
        if not checkpoint.artifacts:
            self.context.completion = None
            return
        self.context.completion = completion_from_dict(
            self._read_json_artifact("completion")
        )

    def _run_final_risk(self) -> dict[str, str]:
        reconciliation_payload = (
            reconciliation_to_dict(self.context.reconciliation)
            if self.context.reconciliation is not None
            else None
        )
        completion_payload = (
            completion_to_dict(self.context.completion)
            if self.context.completion is not None
            else None
        )
        final_risk = reassess_final_risk(
            initial_risk=_required(self.context.risk_assessment, "risk assessment"),
            intent_packet=_required(self.context.intent, "intent"),
            quality_results=self.context.quality_results,
            reviewer_result=self.context.reviewer_result,
            reconciliation_payload=reconciliation_payload,
            completion_summary=completion_payload,
        )
        workspace = self._phase_workspace(RunPhase.FINAL_RISK)
        workspace.write_json("final_risk.json", final_risk_to_dict(final_risk))
        artifacts = self._commit_files(
            RunPhase.FINAL_RISK,
            workspace,
            {"final_risk": ("final_risk.json", "final_risk.json")},
        )
        self.context.final_risk = final_risk
        return artifacts

    def _load_final_risk(self) -> None:
        self.context.final_risk = final_risk_from_dict(
            self._read_json_artifact("final_risk")
        )

    def _run_reporting(self) -> dict[str, str]:
        manifest = self.context.manifest
        summary = _required(self.context.change_summary, "change summary")
        repository_intelligence = _required(
            self.context.repository_intelligence,
            "repository intelligence",
        )
        reconciliation_payload = (
            reconciliation_to_dict(self.context.reconciliation)
            if self.context.reconciliation is not None
            else None
        )
        completion_payload = (
            completion_to_dict(self.context.completion)
            if self.context.completion is not None
            else None
        )
        final_risk_payload = final_risk_to_dict(
            _required(self.context.final_risk, "final risk")
        )
        multi_summary: dict[str, object] | None = None
        if self.context.multi_run is not None:
            payload = multi_reviewer_run_to_dict(self.context.multi_run)
            multi_summary = {
                "reviewer_count": payload["reviewer_count"],
                "status_counts": payload["status_counts"],
                "roles": [item["role"] for item in payload["executions"]],
            }

        workspace = self._phase_workspace(RunPhase.REPORTING)
        aggregate_root = workspace.path / "agg"
        aggregate = ObservationStore(aggregate_root)
        for source in self._observation_stores():
            _copy_observations(source, aggregate)
        brief = build_review_brief(
            review_id=manifest.review_id,
            base_revision=manifest.revisions.resolved_base_sha,
            head_revision=manifest.revisions.resolved_head_sha,
            intent_packet=_required(self.context.intent, "intent"),
            risk_assessment=_required(
                self.context.risk_assessment,
                "risk assessment",
            ),
            changed_files=summary.changed_files,
            quality_results=self.context.quality_results,
            reviewer_result=self.context.reviewer_result,
            observation_summaries=aggregate.summaries_by_id(),
            repository_intelligence_summary=summarize_repository_intelligence(
                repository_intelligence
            ),
            multi_reviewer_summary=multi_summary,
            reconciliation_payload=reconciliation_payload,
            completion_summary=completion_payload,
            final_risk_assessment=final_risk_payload,
            incremental_priority=(
                incremental_priority_to_dict(self.context.incremental_priority)
                if self.context.incremental_priority is not None
                else None
            ),
        )
        workspace.write_json("review_brief.json", review_brief_to_dict(brief))
        workspace.write_text("report.md", render_review_brief_markdown(brief))
        artifacts = self._commit_files(
            RunPhase.REPORTING,
            workspace,
            {
                "review_brief": ("review_brief.json", "review_brief.json"),
                "report": ("report.md", "report.md"),
            },
        )
        observation_path = self._commit_observation_store(
            phase=RunPhase.REPORTING,
            workspace=workspace,
            source=aggregate,
            destination_root="",
            artifact_name="observations",
        )
        artifacts["observations"] = observation_path
        self.context.brief = brief
        return artifacts

    def _load_reporting(self) -> None:
        self.context.brief = review_brief_from_dict(
            self._read_json_artifact("review_brief")
        )
        self._load_observation_artifact("observations")

    def _phase_workspace(self, phase: RunPhase) -> AttemptWorkspace:
        attempt = self.context.manifest.phases[phase.value].attempts
        workspace = AttemptWorkspace(
            self.context.checkpoint_store.run_dir,
            phase,
            attempt,
        )
        workspace.prepare()
        return workspace

    def _commit_files(
        self,
        phase: RunPhase,
        workspace: AttemptWorkspace,
        files: Mapping[str, tuple[str, str]],
    ) -> dict[str, str]:
        committed: dict[str, str] = {}
        for name, (source_path, destination_path) in files.items():
            workspace.promote_file(source_path, destination_path)
            self.context.session_store.register_existing_artifact(
                name=name,
                relative_path=destination_path,
                schema=artifact_schema(name),
                phase=phase,
                revision_binding=(
                    None if name == "request" else self.context.revision_binding
                ),
                now=self._clock(),
            )
            committed[name] = destination_path
        return committed

    def _commit_observation_store(
        self,
        *,
        phase: RunPhase,
        workspace: AttemptWorkspace,
        source: ObservationStore,
        destination_root: str,
        artifact_name: str,
    ) -> str:
        source_root = source.run_dir.relative_to(workspace.path).as_posix()
        prefix = "" if source_root == "." else f"{source_root}/"
        destination_prefix = f"{destination_root}/" if destination_root else ""
        destination_store_root = self.context.checkpoint_store.run_dir
        if destination_root:
            destination_store_root = destination_store_root.joinpath(
                *PurePosixPath(destination_root).parts
            )
        (destination_store_root / "observations").mkdir(
            parents=True,
            exist_ok=True,
        )
        for observation in source.list_observations():
            workspace.promote_file(
                f"{prefix}{observation.raw_artifact_ref}",
                f"{destination_prefix}{observation.raw_artifact_ref}",
            )
        destination_path = f"{destination_prefix}observations.jsonl"
        workspace.promote_file(
            f"{prefix}observations.jsonl",
            destination_path,
        )
        self.context.session_store.register_existing_artifact(
            name=artifact_name,
            relative_path=destination_path,
            schema=artifact_schema(artifact_name),
            phase=phase,
            revision_binding=self.context.revision_binding,
            now=self._clock(),
        )
        return destination_path

    def _read_json_artifact(self, name: str) -> dict[str, Any]:
        descriptor = self.context.manifest.artifacts.get(name)
        if descriptor is None:
            raise ValueError(f"artifact is not registered: {name}")
        path = self.context.checkpoint_store.run_dir.joinpath(
            *PurePosixPath(descriptor.path).parts
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"artifact {name} must contain a JSON object")
        return payload

    def _load_observation_artifact(self, name: str) -> ObservationStore:
        descriptor = self.context.manifest.artifacts.get(name)
        if descriptor is None:
            raise ValueError(f"observation artifact is not registered: {name}")
        if descriptor.schema != artifact_schema(name):
            raise ValueError(f"observation artifact schema is invalid: {name}")
        if not self.context.session_store.validate_artifact(descriptor):
            raise ValueError(f"observation artifact hash is invalid: {name}")
        root = self.context.checkpoint_store.run_dir.joinpath(
            *PurePosixPath(descriptor.path).parent.parts
        )
        return ObservationStore.load(
            root,
            self.context.observation_revision_bindings,
        )

    def _observation_stores(self) -> list[ObservationStore]:
        stores: list[ObservationStore] = []
        if self.context.repository_observations is not None:
            stores.append(self.context.repository_observations)
        stores.extend(
            self.context.reviewer_observations[index]
            for index in sorted(self.context.reviewer_observations)
        )
        return stores

    def _authorized_observation_summaries(
        self,
        current: ObservationStore | None = None,
    ) -> dict[str, str]:
        summaries: dict[str, str] = {}
        for store in self._observation_stores():
            summaries.update(store.summaries_by_id())
        if current is not None:
            summaries.update(current.summaries_by_id())
        return summaries

    def _review_diff_excerpt(self, full_summary: ChangeSummary) -> list[str]:
        priority = self.context.incremental_priority
        if priority is None:
            return list(full_summary.diff_excerpt)
        return [
            "Incremental priority diff (new changes since parent Head):",
            *priority.diff_excerpt,
            "Full review diff (child Base..Head):",
            *full_summary.diff_excerpt,
        ]

    def _write_compatibility_state(self, message: str) -> bool:
        manifest = self.context.manifest
        state = RunState(
            review_id=manifest.review_id,
            status=manifest.status,
            phase=manifest.current_phase,
            repository_path=manifest.repository.canonical_path,
            base_revision=manifest.revisions.requested_base,
            head_revision=manifest.revisions.requested_head,
            resolved_base_revision=manifest.revisions.resolved_base_sha,
            resolved_head_revision=manifest.revisions.resolved_head_sha,
            message=message,
            artifacts={
                name: descriptor.path
                for name, descriptor in manifest.artifacts.items()
            },
            errors=list(manifest.errors),
        )
        try:
            self.context.checkpoint_store.write_state(state)
        except Exception as error:
            self.context.compatibility_warnings.append(
                f"unable to write legacy state summary: {type(error).__name__}: {error}"
            )
            return False
        return True

    def _record_failure(self, phase: RunPhase, error: Exception) -> None:
        message = f"{type(error).__name__}: {error}"
        try:
            manifest = self.context.manifest
            checkpoint = manifest.phases[phase.value]
            if (
                manifest.status is not RunStatus.COMPLETED
                and checkpoint.status is not PhaseStatus.COMPLETED
                and checkpoint.status is not PhaseStatus.FAILED
            ):
                self.context.session_store.mark_session_failed(
                    phase,
                    message,
                    self._clock(),
                )
        finally:
            try:
                self._write_compatibility_state("Review failed")
            except Exception:
                pass


@dataclass(frozen=True)
class _ReviewerArtifactNames:
    envelope: str
    raw_response: str
    result: str
    trace: str | None


def _reviewer_artifact_names(
    index: int,
    *,
    single: bool,
    include_trace: bool,
) -> _ReviewerArtifactNames:
    if single:
        return _ReviewerArtifactNames(
            envelope="reviewer_envelope",
            raw_response="reviewer_raw_response",
            result="reviewer",
            trace="reviewer_agent_trace" if include_trace else None,
        )
    prefix = f"reviewer_{index}"
    return _ReviewerArtifactNames(
        envelope=f"{prefix}_envelope",
        raw_response=f"{prefix}_raw_response",
        result=f"{prefix}_result",
        trace=f"{prefix}_agent_trace" if include_trace else None,
    )


def _reviewer_artifact_filename(name: str) -> str:
    if name == "reviewer":
        return "reviewer_result.json"
    return f"{name}.json"


def _required(value: Any, label: str) -> Any:
    if value is None:
        raise ValueError(f"Pipeline context is missing {label}")
    return value


def _change_summary_from_packet(packet: RiskAssessmentPacket) -> ChangeSummary:
    payload = packet.change_summary
    required = {
        "repository_path",
        "base_revision",
        "head_revision",
        "changed_files",
        "diff_stat",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            "risk packet change_summary is missing: " + ", ".join(sorted(missing))
        )
    changed_files = payload["changed_files"]
    if not isinstance(changed_files, list) or any(
        not isinstance(item, str) for item in changed_files
    ):
        raise ValueError("risk packet changed_files must be a list of strings")
    return ChangeSummary(
        repository_path=str(payload["repository_path"]),
        base_revision=str(payload["base_revision"]),
        head_revision=str(payload["head_revision"]),
        changed_files=list(changed_files),
        diff_stat=str(payload["diff_stat"]),
        diff_excerpt=list(packet.diff_excerpt),
    )


def _copy_observations(source: ObservationStore, target: ObservationStore) -> None:
    for observation in source.list_observations():
        raw_path = source.run_dir.joinpath(
            *PurePosixPath(observation.raw_artifact_ref).parts
        )
        raw_content = raw_path.read_bytes().decode("utf-8")
        target.record(
            source=observation.source,
            revision=observation.revision,
            path=observation.path,
            line_start=observation.line_start,
            line_end=observation.line_end,
            raw_content=raw_content,
            context_view=observation.context_view,
        )


def _reviewer_invocation_model(config: ReviewExecutionConfig) -> str:
    if config.reviewer_model:
        return config.reviewer_model
    if config.reviewer_provider == "fake":
        return "fake-reviewer"
    if config.reviewer_provider == "none":
        return "none"
    return "configured-reviewer-model"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

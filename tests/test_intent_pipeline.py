from __future__ import annotations

import hashlib
import json
from pathlib import Path

from conftest import run_git

from review_agent.checkpoint import CheckpointStore
from review_agent.memory_models import MemoryExecutionConfig, MemoryMode
from review_agent.model_adapter_factory import FakeModelAdapterFactory
from review_agent.model_protocol import ModelResponseKind, ModelTurnResponse
from review_agent.models import (
    ClarificationQuestion,
    ClarificationStatus,
    IntentDecision,
    IntentDecisionAction,
    IntentField,
    IntentOrigin,
    IntentSource,
    IntentStatus,
    ReviewRequest,
)
from review_agent.pipeline import ReviewPipeline
from review_agent.resume import ResumeAction, ReviewSessionResumer
from review_agent.revision import RevisionResolver
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import ReviewExecutionConfig, initial_session_manifest
from review_agent.session_store import SessionStore


class ResolveMaterialIntent:
    def __init__(self) -> None:
        self.fields: list[IntentField] = []

    def decide(self, question: ClarificationQuestion) -> IntentDecision:
        self.fields.append(question.field)
        if question.field is IntentField.GOAL:
            return IntentDecision(
                question_id=question.question_id,
                action=IntentDecisionAction.CORRECTED,
                corrected_values=["Preserve addition semantics"],
                user_response="Preserve addition semantics",
            )
        if question.field is IntentField.ACCEPTANCE_CRITERIA:
            return IntentDecision(
                question_id=question.question_id,
                action=IntentDecisionAction.CORRECTED,
                corrected_values=["add(a, b) returns a + b"],
                user_response="add(a, b) returns a + b",
            )
        if question.field is IntentField.SCOPE:
            return IntentDecision(
                question_id=question.question_id,
                action=IntentDecisionAction.CONFIRMED,
            )
        return IntentDecision(
            question_id=question.question_id,
            action=IntentDecisionAction.SKIPPED,
            continuation_basis="test_policy",
        )


class DeferIntent:
    def decide(self, _question: ClarificationQuestion) -> None:
        return None


class ResolveFirstThenDefer:
    def __init__(self) -> None:
        self.resolver = ResolveMaterialIntent()
        self.calls = 0

    def decide(
        self,
        question: ClarificationQuestion,
    ) -> IntentDecision | None:
        self.calls += 1
        if self.calls == 1:
            return self.resolver.decide(question)
        return None


class MustNotBeCalled:
    def decide(self, _question: ClarificationQuestion) -> IntentDecision:
        raise AssertionError("non-interactive review must not call the injected clarifier")


def _pipeline(
    git_repo: Path,
    *,
    review_id: str,
    non_interactive: bool,
    clarifier=None,
    symbolic_head: bool = False,
    existing_ci_evidence: tuple[str, ...] = (),
    adapter_factory_builder=None,
) -> tuple[ReviewPipeline, SessionStore, CheckpointStore]:
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change implementation")
    head = run_git(git_repo, "rev-parse", "HEAD")

    resolver = RevisionResolver()
    identity = resolver.repository_identity(git_repo)
    requested_head = "HEAD" if symbolic_head else head
    revisions = resolver.resolve_pair(git_repo, base, requested_head)
    checkpoint_store = CheckpointStore(git_repo, review_id)
    session_store = SessionStore(checkpoint_store.run_dir)
    session_store.create(
        initial_session_manifest(
            review_id=review_id,
            repository=identity,
            revisions=revisions,
            execution=ReviewExecutionConfig(
                reviewer_provider="fake",
                reviewer_model=None,
                reviewer_base_url=None,
                reviewer_api_key_env="REVIEW_AGENT_API_KEY",
                reviewer_mode="single",
                reviewer_loop="agent-loop",
                non_interactive=non_interactive,
                memory=MemoryExecutionConfig(
                    mode=MemoryMode.OFF,
                    root_path=str((git_repo / ".memory-test").resolve()),
                ),
            ),
            now="2026-07-13T00:00:00Z",
        )
    )
    request = ReviewRequest(
        repository_path=identity.canonical_path,
        base_revision=base,
        head_revision=requested_head,
        user_intent="Preserve addition semantics",
        existing_ci_evidence=existing_ci_evidence,
    )
    pipeline_kwargs = {}
    if adapter_factory_builder is not None:
        pipeline_kwargs["adapter_factory_builder"] = adapter_factory_builder
    return (
        ReviewPipeline(
            repository=git_repo,
            checkpoint_store=checkpoint_store,
            session_store=session_store,
            request=request,
            intent_clarifier=clarifier,
            **pipeline_kwargs,
        ),
        session_store,
        checkpoint_store,
    )


def _ci_evidence_payload(source_id: str, text: str) -> str:
    return json.dumps(
        {
            "source_id": source_id,
            "text": text,
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def test_ci_evidence_is_visible_to_intent_inference_without_explicit_authority(
    git_repo: Path,
) -> None:
    encoded_evidence = (
        _ci_evidence_payload("ci-pytest", "1 passed; regression job is green"),
        _ci_evidence_payload("ci-empty", ""),
    )
    captured_contexts: list[dict[str, object]] = []
    inferred_value = "Treat the supplied CI report as contextual input."

    class IntentAdapter:
        provider_name = "ci-intent-test"

        def complete_turn(self, request):
            context = json.loads(request.messages[0]["content"])
            captured_contexts.append(context)
            ci_observation_id = next(
                observation_id
                for observation_id, summary in context[
                    "initial_observation_summaries"
                ].items()
                if "1 passed; regression job is green" in summary
            )
            return ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                provider_name=self.provider_name,
                final_text=json.dumps(
                    {
                        "candidates": [
                            {
                                "field": "constraints",
                                "value": inferred_value,
                                "origin": "request_metadata",
                                "confidence": "high",
                                "source_refs": ["request.existing_ci_evidence"],
                                "evidence_refs": [ci_observation_id],
                                "rationale": "The CI signal is useful context but not authority.",
                                "conclusion_impact": "supplemental",
                            }
                        ],
                        "uncertainties": [],
                        "summary": "CI evidence was considered as non-authoritative context.",
                    }
                ),
            )

    class Factory:
        def __init__(self) -> None:
            self.created = 0

        def create(self):
            self.created += 1
            if self.created == 1:
                return IntentAdapter()
            return FakeModelAdapterFactory().create()

    factory = Factory()
    pipeline, session_store, checkpoint_store = _pipeline(
        git_repo,
        review_id="review-intent-ci-evidence",
        non_interactive=True,
        existing_ci_evidence=encoded_evidence,
        adapter_factory_builder=lambda _config: factory,
    )

    result = pipeline.execute()

    assert session_store.load().status is RunStatus.COMPLETED
    assert len(captured_contexts) == 1
    inference_context = captured_contexts[0]
    request_summary = json.loads(inference_context["deterministic_request_summary"])
    assert request_summary["existing_ci_evidence"] == list(encoded_evidence)

    intent_store = result.context.intent_observations
    assert intent_store is not None
    ci_observations = [
        item
        for item in intent_store.list_observations()
        if item.source.startswith("review_request.existing_ci_evidence:")
    ]
    assert ci_observations[0].source.startswith(
        "review_request.existing_ci_evidence:0:"
    )
    assert ci_observations[1].source.startswith(
        "review_request.existing_ci_evidence:1:"
    )
    head = result.context.manifest.revisions.resolved_head_sha
    assert all(
        item.revision == f"head@{head}"
        and item.path is None
        and item.line_start is None
        and item.line_end is None
        for item in ci_observations
    )
    summaries = inference_context["initial_observation_summaries"]
    assert all(item.observation_id in summaries for item in ci_observations)
    assert "1 passed; regression job is green" in summaries[
        ci_observations[0].observation_id
    ]
    assert "(empty text)" in summaries[ci_observations[1].observation_id]
    assert [
        (checkpoint_store.run_dir / "observation_stores" / "intent").joinpath(
            item.raw_artifact_ref
        ).read_text(encoding="utf-8")
        for item in ci_observations
    ] == ["1 passed; regression job is green", ""]

    inferred_claim = next(
        claim for claim in result.context.intent_claims if claim.value == inferred_value
    )
    assert inferred_claim.source is IntentSource.INFERRED
    assert inferred_claim.origin is IntentOrigin.LLM_INFERENCE
    assert not any(
        claim.source is IntentSource.EXPLICIT and claim.value == inferred_value
        for claim in result.context.intent_claims
    )


def test_interactive_resolution_produces_sufficient_confirmed_intent(
    git_repo: Path,
) -> None:
    clarifier = ResolveMaterialIntent()
    pipeline, session_store, _ = _pipeline(
        git_repo,
        review_id="review-intent-confirmed",
        non_interactive=False,
        clarifier=clarifier,
    )

    result = pipeline.execute()

    assert result.awaiting_user is False
    assert session_store.load().status is RunStatus.COMPLETED
    assert result.context.intent is not None
    assert result.context.intent.status is IntentStatus.SUFFICIENT
    assert result.context.intent.goal == "Preserve addition semantics"
    assert result.context.intent.acceptance_criteria == ["add(a, b) returns a + b"]
    assert result.context.intent.scope == ["app.py"]
    assert set(clarifier.fields) == {
        IntentField.GOAL,
        IntentField.ACCEPTANCE_CRITERIA,
        IntentField.SCOPE,
    }
    assert all(
        question.status
        in {ClarificationStatus.CONFIRMED, ClarificationStatus.CORRECTED}
        for question in result.context.intent.clarifications
    )


def test_awaiting_user_resume_reuses_intent_discovery_without_reinference(
    git_repo: Path,
) -> None:
    encoded_evidence = (
        _ci_evidence_payload("ci-resume", "resume check passed"),
        _ci_evidence_payload("ci-resume-empty", ""),
    )
    pipeline, session_store, checkpoint_store = _pipeline(
        git_repo,
        review_id="review-intent-resume",
        non_interactive=False,
        clarifier=DeferIntent(),
        existing_ci_evidence=encoded_evidence,
    )

    first = pipeline.execute()

    awaiting = session_store.load()
    assert first.awaiting_user is True
    assert first.open_questions
    assert awaiting.status is RunStatus.AWAITING_USER
    discovery_before = awaiting.phases[RunPhase.INTENT_DISCOVERY.value]
    inference_before = awaiting.artifacts["intent_inference"]
    assert first.context.intent_observations is not None
    ci_observations_before = tuple(
        item
        for item in first.context.intent_observations.list_observations()
        if item.source.startswith("review_request.existing_ci_evidence:")
    )
    ci_summaries_before = {
        item.observation_id: first.context.intent_observations.summaries_by_id()[
            item.observation_id
        ]
        for item in ci_observations_before
    }

    clarifier = ResolveMaterialIntent()
    resumed = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
        intent_clarifier=clarifier,
    ).resume()

    completed = session_store.load()
    assert resumed.action is ResumeAction.CONTINUE_SESSION
    assert resumed.starting_phase is RunPhase.INTENT_RESOLUTION
    assert completed.status is RunStatus.COMPLETED
    assert completed.phases[RunPhase.INTENT_DISCOVERY.value].attempts == discovery_before.attempts
    assert completed.artifacts["intent_inference"] == inference_before
    assert resumed.pipeline_result is not None
    assert resumed.pipeline_result.context.intent is not None
    assert resumed.pipeline_result.context.intent.status is IntentStatus.SUFFICIENT
    assert resumed.pipeline_result.context.intent_observations is not None
    ci_observations_after = tuple(
        item
        for item in resumed.pipeline_result.context.intent_observations.list_observations()
        if item.source.startswith("review_request.existing_ci_evidence:")
    )
    assert ci_observations_after == ci_observations_before
    assert {
        item.observation_id: resumed.pipeline_result.context.intent_observations.summaries_by_id()[
            item.observation_id
        ]
        for item in ci_observations_after
    } == ci_summaries_before


def test_resume_does_not_repeat_questions_with_committed_decisions(
    git_repo: Path,
) -> None:
    first_clarifier = ResolveFirstThenDefer()
    pipeline, session_store, checkpoint_store = _pipeline(
        git_repo,
        review_id="review-intent-partial-decisions",
        non_interactive=False,
        clarifier=first_clarifier,
    )

    first = pipeline.execute()

    awaiting = session_store.load()
    assert first.awaiting_user is True
    assert first_clarifier.resolver.fields == [IntentField.GOAL]
    assert len(
        awaiting.phases[RunPhase.INTENT_RESOLUTION.value].user_decisions
    ) == 1

    resumed_clarifier = ResolveMaterialIntent()
    resumed = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
        intent_clarifier=resumed_clarifier,
    ).resume()

    assert resumed.action is ResumeAction.CONTINUE_SESSION
    assert session_store.load().status is RunStatus.COMPLETED
    assert IntentField.GOAL not in resumed_clarifier.fields
    assert set(resumed_clarifier.fields) == {
        IntentField.ACCEPTANCE_CRITERIA,
        IntentField.SCOPE,
    }


def test_non_interactive_pipeline_skips_questions_without_calling_clarifier(
    git_repo: Path,
) -> None:
    pipeline, session_store, _ = _pipeline(
        git_repo,
        review_id="review-intent-non-interactive",
        non_interactive=True,
        clarifier=MustNotBeCalled(),
    )

    result = pipeline.execute()

    assert session_store.load().status is RunStatus.COMPLETED
    assert result.context.intent is not None
    assert result.context.intent.status is IntentStatus.INSUFFICIENT
    assert result.context.intent.clarifications
    assert all(
        question.status is ClarificationStatus.SKIPPED_NON_INTERACTIVE
        for question in result.context.intent.clarifications
    )


def test_incremental_child_receives_interactive_intent_clarifier(
    git_repo: Path,
) -> None:
    parent_clarifier = ResolveMaterialIntent()
    pipeline, session_store, checkpoint_store = _pipeline(
        git_repo,
        review_id="review-intent-incremental-parent",
        non_interactive=False,
        clarifier=parent_clarifier,
        symbolic_head=True,
    )
    pipeline.execute()
    (git_repo / "later.py").write_text("value = 1\n", encoding="utf-8")
    run_git(git_repo, "add", "later.py")
    run_git(git_repo, "commit", "-m", "extend reviewed change")

    child_clarifier = ResolveMaterialIntent()
    result = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
        intent_clarifier=child_clarifier,
    ).resume()

    assert result.action is ResumeAction.CREATE_INCREMENTAL_SESSION
    assert result.manifest.status is RunStatus.COMPLETED
    assert child_clarifier.fields

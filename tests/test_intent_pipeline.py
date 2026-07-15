from __future__ import annotations

from pathlib import Path

from conftest import run_git

from review_agent.checkpoint import CheckpointStore
from review_agent.memory_models import MemoryExecutionConfig, MemoryMode
from review_agent.models import (
    ClarificationQuestion,
    ClarificationStatus,
    IntentDecision,
    IntentDecisionAction,
    IntentField,
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
    )
    return (
        ReviewPipeline(
            repository=git_repo,
            checkpoint_store=checkpoint_store,
            session_store=session_store,
            request=request,
            intent_clarifier=clarifier,
        ),
        session_store,
        checkpoint_store,
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
    pipeline, session_store, checkpoint_store = _pipeline(
        git_repo,
        review_id="review-intent-resume",
        non_interactive=False,
        clarifier=DeferIntent(),
    )

    first = pipeline.execute()

    awaiting = session_store.load()
    assert first.awaiting_user is True
    assert first.open_questions
    assert awaiting.status is RunStatus.AWAITING_USER
    discovery_before = awaiting.phases[RunPhase.INTENT_DISCOVERY.value]
    inference_before = awaiting.artifacts["intent_inference"]

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

from review_agent.git_repo import ChangeSummary
from review_agent.intent import build_intent_packet
from review_agent.models import IntentSource, IntentStatus, ReviewRequest


def test_user_intent_is_explicit_and_focus_is_not_intent():
    request = ReviewRequest(
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
        user_intent="Add idempotency to payment callback",
        review_focus="duplicate execution and retry safety",
    )
    summary = ChangeSummary("C:/repo", "main", "HEAD", ["payments/callback.py"], "", [])

    packet = build_intent_packet(request, summary)

    assert packet.goal == "Add idempotency to payment callback"
    assert packet.sources["goal"] is IntentSource.EXPLICIT
    assert "review_focus" not in packet.sources
    assert "duplicate execution and retry safety" not in packet.acceptance_criteria
    assert "acceptance criteria are not explicitly declared" in packet.uncertainties


def test_missing_user_intent_creates_inferred_goal():
    request = ReviewRequest(repository_path="C:/repo", base_revision="main", head_revision="HEAD")
    summary = ChangeSummary("C:/repo", "main", "HEAD", ["auth/session.py"], "", ["+def validate_session(token):"])

    packet = build_intent_packet(request, summary)

    assert packet.goal == "Review changes touching auth/session.py"
    assert packet.sources["goal"] is IntentSource.INFERRED
    assert "user did not provide explicit intent" in packet.uncertainties
    assert packet.status is IntentStatus.PARTIAL


def test_empty_change_set_is_insufficient():
    request = ReviewRequest(repository_path="C:/repo", base_revision="main", head_revision="HEAD")
    summary = ChangeSummary("C:/repo", "main", "HEAD", [], "", [])

    packet = build_intent_packet(request, summary)

    assert packet.goal is None
    assert packet.status is IntentStatus.INSUFFICIENT
    assert "no changed files were detected" in packet.uncertainties


def test_project_rules_are_explicit_constraints():
    request = ReviewRequest(
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
        project_rules=("preserve public API compatibility",),
    )
    summary = ChangeSummary("C:/repo", "main", "HEAD", ["api.py"], "", [])

    packet = build_intent_packet(request, summary)

    assert packet.constraints == ["preserve public API compatibility"]
    assert packet.sources["constraints"] is IntentSource.EXPLICIT

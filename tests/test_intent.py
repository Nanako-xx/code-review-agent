from review_agent.git_repo import ChangeSummary
from review_agent.intent import build_intent_packet
from review_agent.models import IntentSource, IntentStatus, ReviewRequest


def test_declared_intent_becomes_partial_packet():
    request = ReviewRequest(
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
        user_intent="Add idempotency to payment callback",
        review_focus="duplicate callbacks and retry behavior",
    )
    summary = ChangeSummary("C:/repo", "main", "HEAD", ["payments/callback.py"], "", [])

    packet = build_intent_packet(request, summary)

    assert packet.goal == "Add idempotency to payment callback"
    assert packet.sources["goal"] is IntentSource.DECLARED
    assert packet.status is IntentStatus.PARTIAL
    assert "acceptance criteria are not explicitly declared" in packet.unknowns


def test_missing_user_intent_is_inferred_from_changed_files():
    request = ReviewRequest(repository_path="C:/repo", base_revision="main", head_revision="HEAD")
    summary = ChangeSummary("C:/repo", "main", "HEAD", ["auth/session.py"], "", ["+def validate_session(token):"])

    packet = build_intent_packet(request, summary)

    assert packet.goal == "Review changes touching auth/session.py"
    assert packet.sources["goal"] is IntentSource.INFERRED
    assert packet.status is IntentStatus.PARTIAL
    assert "user did not provide declared intent" in packet.unknowns


def test_no_changed_files_makes_intent_insufficient():
    request = ReviewRequest(repository_path="C:/repo", base_revision="main", head_revision="HEAD")
    summary = ChangeSummary("C:/repo", "main", "HEAD", [], "", [])

    packet = build_intent_packet(request, summary)

    assert packet.status is IntentStatus.INSUFFICIENT
    assert "no changed files were detected" in packet.unknowns

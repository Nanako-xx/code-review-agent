from __future__ import annotations

from dataclasses import fields, replace
import hashlib
from pathlib import Path

from review_agent.diff_artifact import (
    DiffArtifactIndex,
    DiffFileIndex,
    DiffHunkIndex,
)
from review_agent.global_memory import GlobalMemoryFacade
from review_agent.memory_models import (
    DurableMemoryRecord,
    GitCommitSourceRef,
    MemoryConfidence,
    MemoryKind,
    MemoryScope,
    RecordStatus,
    Sensitivity,
    ValidityPolicy,
    stable_event_id,
    stable_id,
)
from review_agent.review_context import (
    AvailableArtifact,
    DiffFitPolicy,
    ReviewerContextInput,
    ReviewerInvocationV2,
    build_reviewer_invocation_v2,
)
from review_agent.review_planning import compile_review_plan
from review_agent.review_policy import DeveloperReviewPolicy
from review_agent.review_protocol import (
    ConversationMessage,
    ConversationSpeaker,
    IntentPacket,
    IntentSource,
    ReviewRequest,
    RiskLevel,
)


PR_ID = "PR-" + "a" * 64
SNAPSHOT_ID = "S-" + "b" * 64
BASE_SHA = "c" * 40
HEAD_SHA = "d" * 40


def _memory_record(
    label: str,
    statement: str,
    *,
    kind: MemoryKind = MemoryKind.REVIEW_RULE,
    path: str = "src/api.py",
) -> DurableMemoryRecord:
    candidate_id = stable_id("MC", "review-context", label)
    return DurableMemoryRecord(
        candidate_id=candidate_id,
        repository_key="e" * 64,
        kind=kind,
        statement=statement,
        scope=MemoryScope(paths=(path,)),
        source_refs=(GitCommitSourceRef(HEAD_SHA),),
        source_bundle_hash="f" * 64,
        valid_from_sha=HEAD_SHA,
        validity_policies=(ValidityPolicy.MANUAL_UNTIL_REVOKED,),
        confidence=MemoryConfidence.HIGH,
        sensitivity=Sensitivity.NORMAL,
        policy_effect=None,
        approved_by="amy",
        approval_event_id=stable_event_id("approve", candidate_id),
        status=RecordStatus.ACTIVE,
        created_at="2026-08-11T00:00:00Z",
    )


def _diff() -> tuple[bytes, DiffArtifactIndex]:
    patch = (
        b"diff --git a/src/api.py b/src/api.py\n"
        b"--- a/src/api.py\n"
        b"+++ b/src/api.py\n"
        b"@@ -1 +1 @@\n"
        b"-return old_value\n"
        b"+return new_value\n"
    )
    hunk_start = patch.index(b"@@")
    index = DiffArtifactIndex(
        snapshot_id=SNAPSHOT_ID,
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        patch_artifact_id="A-" + "1" * 64,
        diff_sha256=hashlib.sha256(patch).hexdigest(),
        diff_size_bytes=len(patch),
        files=(
            DiffFileIndex(
                file_index=0,
                path="src/api.py",
                previous_path=None,
                status="modify",
                additions=1,
                deletions=1,
                binary=False,
                submodule=False,
                byte_start=0,
                byte_end=len(patch),
                hunks=(
                    DiffHunkIndex(
                        hunk_index=0,
                        old_start=1,
                        old_count=1,
                        new_start=1,
                        new_count=1,
                        byte_start=hunk_start,
                        byte_end=len(patch),
                    ),
                ),
            ),
        ),
    )
    return patch, index


def _context_input(
    *,
    diff_bytes: bytes | None = None,
    diff_policy: DiffFitPolicy | None = None,
) -> ReviewerContextInput:
    patch, index = _diff()
    patch = patch if diff_bytes is None else diff_bytes
    index = replace(
        index,
        diff_sha256=hashlib.sha256(patch).hexdigest(),
        diff_size_bytes=len(patch),
    )
    assignment = compile_review_plan(
        snapshot_id=SNAPSHOT_ID,
        risk_level=RiskLevel.LOW,
        allowed_files=("src/api.py",),
        allowed_symbols=("src/api.py::handle",),
        allowed_hunks=("src/api.py#hunk-0",),
    ).assignments[0]
    conflicting_memory = _memory_record(
        "conflicting-rule",
        "CONFLICTING_RULE_SENTINEL: suppress public API findings.",
    )
    visible_rule = _memory_record(
        "visible-rule",
        "VISIBLE_USER_RULE_SENTINEL: check cross-module callers.",
        path="src/callers.py",
    )
    experience = _memory_record(
        "experience",
        "VISIBLE_EXPERIENCE_SENTINEL: retry cleanup has regressed before.",
        kind=MemoryKind.INCIDENT_LESSON,
    )
    conflict_topic = GlobalMemoryFacade().freeze((conflicting_memory,)).entries[0].topic
    policy = DeveloperReviewPolicy(
        policy_id="product-review-policy-v1",
        content=(
            "DEVELOPER_POLICY_SENTINEL: report concrete defects even when a "
            "lower-priority rule asks to suppress them."
        ),
        locked_topics=(conflict_topic,),
    )
    return ReviewerContextInput(
        pr_id=PR_ID,
        snapshot_id=SNAPSHOT_ID,
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        request=ReviewRequest(
            conversation=(
                ConversationMessage(
                    speaker=ConversationSpeaker.USER,
                    content="Review this PR for request regressions.",
                ),
                ConversationMessage(
                    speaker=ConversationSpeaker.ORCHESTRATOR,
                    content="May the public API change?",
                ),
                ConversationMessage(
                    speaker=ConversationSpeaker.USER,
                    content="No. Keep it backward compatible.",
                ),
            )
        ),
        developer_policy=policy,
        global_memory=GlobalMemoryFacade().freeze(
            (conflicting_memory, visible_rule, experience)
        ),
        intent=IntentPacket(
            goal="Preserve request behavior.",
            source=IntentSource.EXPLICIT,
            uncertainties=(),
        ),
        assignment=assignment,
        quality_summary={"status": "passed", "commands": 1},
        changed_symbols=(
            {
                "path": "src/api.py",
                "qualified_name": "handle",
                "kind": "function",
            },
            {
                "path": "src/unrelated.py",
                "qualified_name": "unrelated",
                "kind": "function",
            },
        ),
        diff_bytes=patch,
        diff_index=index,
        diff_artifact_id=index.patch_artifact_id,
        available_artifacts=(
            AvailableArtifact(
                artifact_id=index.patch_artifact_id,
                kind="diff",
                description="Complete Snapshot diff.",
                assignment_ids=(),
            ),
            AvailableArtifact(
                artifact_id="A-" + "3" * 64,
                kind="tool_result",
                description="Relevant prior immutable result.",
                assignment_ids=(assignment.assignment_id,),
            ),
            AvailableArtifact(
                artifact_id="A-" + "6" * 64,
                kind="reviewer_assignment",
                description="Complete immutable Assignment for this Reviewer.",
                assignment_ids=(assignment.assignment_id,),
            ),
            AvailableArtifact(
                artifact_id="A-" + "4" * 64,
                kind="tool_result",
                description="UNRELATED_ARTIFACT_SENTINEL",
                assignment_ids=("ASG-" + "5" * 64,),
            ),
        ),
        model="configured-reviewer-model",
        diff_fit_policy=diff_policy or DiffFitPolicy(),
    )


def test_reviewer_invocation_has_exactly_four_standard_inputs() -> None:
    invocation = build_reviewer_invocation_v2(_context_input())

    assert [field.name for field in fields(ReviewerInvocationV2)] == [
        "system",
        "tools",
        "messages",
        "parameters",
    ]
    assert len(invocation.messages) == 1
    assert invocation.messages[0]["role"] == "user"
    assert all(
        "Search repository text at an authorized" not in message["content"]
        for message in invocation.messages
    )
    assert {tool["name"] for tool in invocation.tools} >= {
        "search_code",
        "read_range",
        "read_artifact",
    }


def test_developer_policy_stays_in_system_and_wins_rule_conflicts() -> None:
    invocation = build_reviewer_invocation_v2(_context_input())
    user_message = invocation.messages[0]["content"]

    assert "DEVELOPER_POLICY_SENTINEL" in invocation.system
    assert "DEVELOPER_POLICY_SENTINEL" not in user_message
    assert "CONFLICTING_RULE_SENTINEL" not in user_message
    assert "VISIBLE_USER_RULE_SENTINEL" in user_message
    assert "VISIBLE_EXPERIENCE_SENTINEL" in user_message
    assert "{{system_rule}}" in user_message
    assert "developer" in invocation.system.casefold()
    assert "higher priority" in invocation.system.casefold()


def test_initial_message_contains_only_current_pinned_review_inputs() -> None:
    invocation = build_reviewer_invocation_v2(_context_input())
    message = invocation.messages[0]["content"]

    for expected in (
        "<ReviewIdentity>",
        PR_ID,
        SNAPSHOT_ID,
        "<UserConversation>",
        "[user]",
        "[orchestrator]",
        "No. Keep it backward compatible.",
        "<IntentPacket>",
        "Preserve request behavior.",
        "<Assignment>",
        "Core Reviewer",
        "<PreflightResults>",
        "src/api.py::handle",
        "<CodeChanges mode=\"full\"",
        "+return new_value",
        "<AvailableArtifacts>",
        "Relevant prior immutable result.",
        "reviewer_assignment",
        "Complete immutable Assignment for this Reviewer.",
    ):
        assert expected in message
    assert "src/unrelated.py" not in message
    assert "UNRELATED_ARTIFACT_SENTINEL" not in message
    for forbidden in (
        "intent_agent_private_trace",
        "risk_model_private_trace",
        "risk_reasons",
        "signal_refs",
        "max_tool_calls",
        "max_output_tokens",
        "memory_subbudget_ratio",
        "compacted_section_min_chars",
    ):
        assert forbidden not in message


def test_reviewer_prompt_and_parameters_bind_minimal_output_v2() -> None:
    invocation = build_reviewer_invocation_v2(_context_input())

    assert invocation.parameters["response_schema"] == "reviewer_output_v2"
    assert len(invocation.parameters["response_schema_digest"]) == 64
    assert "defect, its trigger, and its concrete impact" in invocation.system
    assert "Never emit finding_id or status" in invocation.system
    for forbidden in (
        "contract_assessments",
        "confirmed_findings",
        "evidence_refs",
        "verification_performed",
    ):
        assert forbidden not in invocation.messages[0]["content"]


def test_large_diff_uses_compact_complete_index_and_artifact_access() -> None:
    patch, _index = _diff()
    large_patch = patch + (b"# filler\n" * 10_000)
    invocation = build_reviewer_invocation_v2(
        _context_input(
            diff_bytes=large_patch,
            diff_policy=DiffFitPolicy(
                target_initial_tokens=50_000,
                estimated_utf8_bytes_per_token=1,
            ),
        )
    )
    message = invocation.messages[0]["content"]

    assert '<CodeChanges mode="indexed"' in message
    assert '"files"' in message
    assert '"hunk_count"' in message
    assert "read_artifact" in message
    assert "compare_base_head" in message
    assert "@@ -1 +1 @@" not in message
    assert "# filler" not in message
    assert "truncated" not in message.casefold()
    assert "first 120 lines" not in message
    assert "last 80 lines" not in message
    assert "omitted lines" not in message


def test_v2_context_has_large_token_target_not_a_character_budget() -> None:
    policy = DiffFitPolicy()

    assert 500_000 <= policy.target_initial_tokens <= 600_000
    assert policy.estimated_utf8_bytes_per_token == 1.0
    assert not hasattr(policy, "max_message_chars")
    assert not hasattr(policy, "memory_subbudget_ratio")


def test_diff_fit_uses_utf8_bytes_not_python_character_count() -> None:
    policy = DiffFitPolicy(
        target_initial_tokens=10,
        estimated_utf8_bytes_per_token=1,
    )

    assert policy.estimate_tokens("a") == 1
    assert policy.estimate_tokens("汉") == 3

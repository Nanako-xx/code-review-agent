from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from review_agent.artifacts import artifact_schema
from review_agent.review_protocol import (
    ConversationMessage,
    ConversationSpeaker,
    FinalFinding,
    FindingSeverity,
    IntentPacket,
    IntentSource,
    ReviewRequest,
    ReviewerFinding,
    ReviewerOutput,
    ReviewerRoleKind,
    ReviewResult,
    ReviewResultStatus,
    RiskDecision,
    RiskLevel,
    WireProtocolError,
    canonical_json_bytes,
)


PR_ID = "PR-" + "a" * 64
SNAPSHOT_ID = "S-" + "b" * 64
FINDING_ID = "F-" + "c" * 64


def _reviewer_finding() -> ReviewerFinding:
    return ReviewerFinding(
        claim="A missing cache entry is dereferenced and returns HTTP 500.",
        severity=FindingSeverity.HIGH,
        path="src/cache.py",
        line=87,
        suggestion="Handle the missing entry and add a first-request regression test.",
    )


def _final_finding() -> FinalFinding:
    return FinalFinding(
        finding_id=FINDING_ID,
        claim="A missing cache entry is dereferenced and returns HTTP 500.",
        severity=FindingSeverity.HIGH,
        path="src/cache.py",
        line=87,
        suggestion="Handle the missing entry and add a first-request regression test.",
    )


def test_minimal_wire_models_have_only_the_approved_fields() -> None:
    assert [field.name for field in fields(ConversationMessage)] == [
        "speaker",
        "content",
    ]
    assert [field.name for field in fields(ReviewRequest)] == ["conversation"]
    assert [field.name for field in fields(IntentPacket)] == [
        "goal",
        "source",
        "uncertainties",
    ]
    assert [field.name for field in fields(RiskDecision)] == ["level"]
    assert [field.name for field in fields(ReviewerFinding)] == [
        "claim",
        "severity",
        "path",
        "line",
        "suggestion",
    ]
    assert [field.name for field in fields(ReviewerOutput)] == [
        "findings",
        "uncertainties",
    ]
    assert [field.name for field in fields(FinalFinding)] == [
        "finding_id",
        "claim",
        "severity",
        "path",
        "line",
        "suggestion",
    ]
    assert [field.name for field in fields(ReviewResult)] == [
        "pr_id",
        "snapshot_id",
        "status",
        "risk_level",
        "findings",
        "uncertainties",
    ]
    assert tuple(role.value for role in ReviewerRoleKind) == (
        "core",
        "adversarial",
        "dynamic",
    )


def test_wire_models_are_immutable() -> None:
    decision = RiskDecision(level=RiskLevel.LOW)

    with pytest.raises(FrozenInstanceError):
        decision.level = RiskLevel.HIGH  # type: ignore[misc]


@pytest.mark.parametrize("source", [IntentSource.EXPLICIT, IntentSource.INFERRED])
def test_intent_packet_accepts_a_goal_only_with_a_real_source(
    source: IntentSource,
) -> None:
    packet = IntentPacket(
        goal="Preserve retry idempotency.",
        source=source,
        uncertainties=(),
    )

    assert packet.to_dict() == {
        "goal": "Preserve retry idempotency.",
        "source": source.value,
        "uncertainties": [],
    }
    assert IntentPacket.from_json(packet.to_json_bytes()) == packet


def test_intent_packet_requires_uncertainty_when_goal_is_unknown() -> None:
    packet = IntentPacket(
        goal=None,
        source=None,
        uncertainties=("The PR has no reliable goal statement.",),
    )

    assert IntentPacket.from_dict(packet.to_dict()) == packet

    with pytest.raises(WireProtocolError, match="uncertainties"):
        IntentPacket(goal=None, source=None, uncertainties=())


@pytest.mark.parametrize(
    ("goal", "source"),
    [
        ("Preserve retries.", None),
        (None, IntentSource.EXPLICIT),
        (None, IntentSource.INFERRED),
    ],
)
def test_intent_packet_rejects_invalid_goal_source_pairs(
    goal: str | None,
    source: IntentSource | None,
) -> None:
    with pytest.raises(WireProtocolError, match="goal.*source|source.*goal"):
        IntentPacket(
            goal=goal,
            source=source,
            uncertainties=("Intent is uncertain.",),
        )


def test_risk_reviewer_and_result_models_round_trip_strictly() -> None:
    risk = RiskDecision(level=RiskLevel.MEDIUM)
    reviewer = ReviewerOutput(
        findings=(_reviewer_finding(),),
        uncertainties=("The retry race could not be reproduced locally.",),
    )
    result = ReviewResult(
        pr_id=PR_ID,
        snapshot_id=SNAPSHOT_ID,
        status=ReviewResultStatus.COMPLETED,
        risk_level=RiskLevel.MEDIUM,
        findings=(_final_finding(),),
        uncertainties=(),
    )

    assert RiskDecision.from_json(risk.to_json()) == risk
    assert ReviewerOutput.from_json(reviewer.to_json()) == reviewer
    assert ReviewResult.from_json(result.to_json_bytes()) == result
    assert reviewer.to_dict() == {
        "findings": [_reviewer_finding().to_dict()],
        "uncertainties": ["The retry race could not be reproduced locally."],
    }


def test_review_request_preserves_only_speaker_labeled_public_conversation() -> None:
    request = ReviewRequest(
        conversation=(
            ConversationMessage(
                speaker=ConversationSpeaker.USER,
                content="Review this PR for concurrency regressions.",
            ),
            ConversationMessage(
                speaker=ConversationSpeaker.ORCHESTRATOR,
                content="Can the public API change?",
            ),
            ConversationMessage(
                speaker=ConversationSpeaker.USER,
                content="No, keep the public API stable.",
            ),
        )
    )

    assert request.to_dict() == {
        "conversation": [
            {
                "speaker": "user",
                "content": "Review this PR for concurrency regressions.",
            },
            {
                "speaker": "orchestrator",
                "content": "Can the public API change?",
            },
            {
                "speaker": "user",
                "content": "No, keep the public API stable.",
            },
        ]
    }
    assert ReviewRequest.from_json(request.to_json()) == request

    with pytest.raises(WireProtocolError, match="speaker"):
        ReviewRequest.from_dict(
            {
                "conversation": [
                    {"speaker": "intent_agent", "content": "private inference"}
                ]
            }
        )


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            IntentPacket,
            {
                "goal": "Preserve retries.",
                "source": "explicit",
                "uncertainties": [],
                "rules": [],
            },
        ),
        (RiskDecision, {"level": "low", "reasons": []}),
        (
            ReviewerFinding,
            {
                "claim": "A request fails.",
                "severity": "high",
                "path": "src/api.py",
                "line": 10,
                "suggestion": "Handle the missing value.",
                "evidence_refs": [],
            },
        ),
        (ReviewerOutput, {"findings": [], "uncertainties": [], "status": "ok"}),
        (
            ReviewResult,
            {
                "pr_id": PR_ID,
                "snapshot_id": SNAPSHOT_ID,
                "status": "completed",
                "risk_level": "low",
                "findings": [],
                "uncertainties": [],
                "generated_at": "2026-08-11T00:00:00Z",
            },
        ),
    ],
)
def test_wire_hydration_rejects_unknown_keys(model: type, payload: dict) -> None:
    with pytest.raises(WireProtocolError, match="unknown field"):
        model.from_dict(payload)


@pytest.mark.parametrize(
    ("model", "raw"),
    [
        (RiskDecision, '{"level":"low","level":"high"}'),
        (
            ReviewerOutput,
            '{"findings":[{"claim":"x","claim":"y",'
            '"severity":"low","path":"x.py","line":1,'
            '"suggestion":"Fix x."}],"uncertainties":[]}',
        ),
    ],
)
def test_wire_hydration_rejects_duplicate_keys_recursively(
    model: type,
    raw: str,
) -> None:
    with pytest.raises(WireProtocolError, match="duplicate JSON key"):
        model.from_json(raw)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "../cache.py",
        "src/../cache.py",
        "src/./cache.py",
        "/src/cache.py",
        "C:/src/cache.py",
        "src\\cache.py",
        "src//cache.py",
        "src/cache.py:secret",
        "src/cache.py/",
        "src/\x00cache.py",
    ],
)
def test_finding_rejects_non_canonical_or_unsafe_paths(path: str) -> None:
    with pytest.raises(WireProtocolError, match="path"):
        ReviewerFinding(
            claim="A request fails.",
            severity=FindingSeverity.LOW,
            path=path,
            line=1,
            suggestion="Handle the failure.",
        )


@pytest.mark.parametrize("line", [True, 0, -1, 1.5, "1"])
def test_finding_rejects_non_positive_integer_lines(line: object) -> None:
    with pytest.raises(WireProtocolError, match="line"):
        ReviewerFinding(
            claim="A request fails.",
            severity=FindingSeverity.LOW,
            path="src/api.py",
            line=line,  # type: ignore[arg-type]
            suggestion="Handle the failure.",
        )


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (RiskDecision, {"level": "urgent"}),
        (
            ReviewerFinding,
            {
                "claim": "A request fails.",
                "severity": "urgent",
                "path": "src/api.py",
                "line": 1,
                "suggestion": "Handle the failure.",
            },
        ),
        (
            ReviewResult,
            {
                "pr_id": PR_ID,
                "snapshot_id": SNAPSHOT_ID,
                "status": "approved",
                "risk_level": "low",
                "findings": [],
                "uncertainties": [],
            },
        ),
    ],
)
def test_wire_hydration_rejects_unknown_enum_values(model: type, payload: dict) -> None:
    with pytest.raises(WireProtocolError, match="must be one of"):
        model.from_dict(payload)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: IntentPacket(
            goal="   ", source=IntentSource.EXPLICIT, uncertainties=()
        ),
        lambda: IntentPacket(
            goal=None, source=None, uncertainties=("\t",)
        ),
        lambda: ConversationMessage(
            speaker=ConversationSpeaker.USER,
            content="\n",
        ),
        lambda: ReviewerFinding(
            claim="",
            severity=FindingSeverity.LOW,
            path="src/api.py",
            line=1,
            suggestion="Handle the failure.",
        ),
        lambda: ReviewerFinding(
            claim="A request fails.",
            severity=FindingSeverity.LOW,
            path="src/api.py",
            line=1,
            suggestion="  ",
        ),
        lambda: ReviewerOutput(findings=(), uncertainties=("",)),
    ],
)
def test_wire_models_reject_empty_text(factory) -> None:
    with pytest.raises(WireProtocolError, match="must not be empty"):
        factory()


def test_review_result_uses_fixed_utf8_json_without_wall_clock_fields() -> None:
    result = ReviewResult(
        pr_id=PR_ID,
        snapshot_id=SNAPSHOT_ID,
        status=ReviewResultStatus.COMPLETED,
        risk_level=RiskLevel.LOW,
        findings=(
            FinalFinding(
                finding_id=FINDING_ID,
                claim="空缓存项会导致请求失败。",
                severity=FindingSeverity.HIGH,
                path="src/cache.py",
                line=87,
                suggestion="先处理缺失缓存项。",
            ),
        ),
        uncertainties=(),
    )

    expected = (
        '{"pr_id":"'
        + PR_ID
        + '","snapshot_id":"'
        + SNAPSHOT_ID
        + '","status":"completed","risk_level":"low","findings":['
        '{"finding_id":"'
        + FINDING_ID
        + '","claim":"空缓存项会导致请求失败。","severity":"high",'
        '"path":"src/cache.py","line":87,"suggestion":"先处理缺失缓存项。"}'
        '],"uncertainties":[]}'
    ).encode("utf-8")

    assert canonical_json_bytes(result) == expected
    assert result.to_json_bytes() == expected
    assert b"generated_at" not in expected
    assert b" " not in expected


def test_v6_artifact_schema_names_are_registered_without_rebinding_v5() -> None:
    expected = {
        "pr_workspace_manifest": "pr_workspace_manifest_v1",
        "snapshot_manifest": "snapshot_manifest_v1",
        "diff_artifact_index": "diff_artifact_index_v1",
        "preflight_result": "preflight_result_v1",
        "intent_packet": "intent_packet_v2_minimal",
        "risk_decision": "risk_decision_v2",
        "review_plan": "review_plan_v2",
        "reviewer_assignment": "reviewer_assignment_v2",
        "reviewer_output": "reviewer_output_v2",
        "aggregation_record": "aggregation_record_v1",
        "review_result": "review_result_v1",
        "context_manifest": "context_manifest_v2",
        "execution_journal_event": "execution_journal_event_v1",
    }

    assert {name: artifact_schema(name) for name in expected} == expected
    assert artifact_schema("intent") == "intent_packet_v2"
    assert artifact_schema("risk_model_decision") == "risk_model_decision_v1"

from __future__ import annotations

import hashlib
import json

import pytest

from review_agent.diff_artifact import (
    DiffArtifactIndex,
    DiffFileIndex,
    DiffHunkIndex,
)
from review_agent.hydration import (
    rejected_reviewer_finding_v2_from_dict,
    reviewer_output_v2_from_dict,
)
from review_agent.review_planning import compile_review_plan
from review_agent.review_protocol import FindingSeverity, RiskLevel
from review_agent.reviewer_output import (
    REVIEWER_OUTPUT_JSON_SCHEMA_V2,
    RejectedReviewerFinding,
    ReviewerOutputEnvelopeError,
    ReviewerOutputParser,
)


SNAPSHOT_ID = "S-" + "a" * 64


def _index(snapshot_id: str = SNAPSHOT_ID) -> DiffArtifactIndex:
    patch = b"diff --git a/src/cache.py b/src/cache.py\n@@ -8,3 +10,3 @@\n"
    return DiffArtifactIndex(
        snapshot_id=snapshot_id,
        base_sha="b" * 40,
        head_sha="c" * 40,
        patch_artifact_id="A-" + "d" * 64,
        diff_sha256=hashlib.sha256(patch).hexdigest(),
        diff_size_bytes=len(patch),
        files=(
            DiffFileIndex(
                file_index=0,
                path="src/cache.py",
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
                        old_start=8,
                        old_count=3,
                        new_start=10,
                        new_count=3,
                        byte_start=40,
                        byte_end=len(patch),
                    ),
                ),
            ),
        ),
    )


def _parser() -> ReviewerOutputParser:
    assignment = compile_review_plan(
        snapshot_id=SNAPSHOT_ID,
        risk_level=RiskLevel.LOW,
        allowed_files=("src/cache.py",),
        allowed_symbols=(),
        allowed_hunks=("src/cache.py#hunk-0",),
    ).assignments[0]
    return ReviewerOutputParser(diff_index=_index(), assignment=assignment)


def _finding(**overrides):
    value = {
        "claim": (
            "When the cache entry is absent, get is called on null and the "
            "first request returns 500."
        ),
        "severity": "high",
        "path": "src/cache.py",
        "line": 10,
        "suggestion": (
            "Handle the missing entry before get and add a first-request test."
        ),
    }
    value.update(overrides)
    return value


def test_exact_zero_finding_output_is_valid() -> None:
    parsed = _parser().parse('{"findings":[],"uncertainties":[]}')

    assert parsed.output.findings == ()
    assert parsed.output.uncertainties == ()
    assert parsed.rejected_findings == ()
    assert parsed.output.to_json() == '{"findings":[],"uncertainties":[]}'


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("not-json", "invalid_json"),
        ("[]", "top_level_not_object"),
        (
            '{"findings":[],"uncertainties":[],"status":"completed"}',
            "top_level_fields_invalid",
        ),
        ('{"findings":{ },"uncertainties":[]}', "top_level_arrays_invalid"),
        ('{"findings":[],"uncertainties":[1]}', "uncertainties_invalid"),
        ('{"findings":[],"findings":[],"uncertainties":[]}', "invalid_json"),
    ],
)
def test_top_level_protocol_failure_invalidates_the_reviewer(raw: str, code: str) -> None:
    with pytest.raises(ReviewerOutputEnvelopeError) as caught:
        _parser().parse(raw)

    assert caught.value.code == code


def test_one_bad_finding_does_not_remove_a_good_finding() -> None:
    with_extra = _finding(finding_id="F-" + "e" * 64)
    payload = {
        "findings": [
            _finding(),
            with_extra,
            _finding(path="src/other.py"),
            _finding(line=999),
            _finding(line=11, suggestion="Please fix."),
        ],
        "uncertainties": ["The fallback path could not be exercised locally."],
    }

    parsed = _parser().parse(json.dumps(payload))

    assert len(parsed.output.findings) == 1
    accepted = parsed.output.findings[0]
    assert accepted.severity is FindingSeverity.HIGH
    assert accepted.path == "src/cache.py"
    assert accepted.line == 10
    assert parsed.output.uncertainties == (
        "The fallback path could not be exercised locally.",
    )
    assert [item.candidate_index for item in parsed.rejected_findings] == [1, 2, 3, 4]
    assert [item.reason for item in parsed.rejected_findings] == [
        "finding_fields_invalid",
        "path_outside_assignment",
        "line_not_in_diff",
        "suggestion_not_actionable",
    ]


@pytest.mark.parametrize(
    ("finding", "reason"),
    [
        (None, "candidate_not_object"),
        (_finding(claim=""), "claim_invalid"),
        (_finding(severity="urgent"), "severity_invalid"),
        (_finding(path="../cache.py"), "path_invalid"),
        (_finding(line=0), "line_invalid"),
        (_finding(suggestion=""), "suggestion_invalid"),
    ],
)
def test_candidate_rejection_reasons_are_stable(finding, reason: str) -> None:
    parsed = _parser().parse(
        json.dumps({"findings": [finding], "uncertainties": []})
    )

    assert parsed.output.findings == ()
    assert parsed.rejected_findings == (
        RejectedReviewerFinding(candidate_index=0, reason=reason),
    )


def test_missing_diff_binding_rejects_only_the_finding() -> None:
    parsed = ReviewerOutputParser(diff_index=None, assignment=None).parse(
        json.dumps({"findings": [_finding()], "uncertainties": []})
    )

    assert parsed.output.findings == ()
    assert parsed.rejected_findings[0].reason == "diff_index_unavailable"


def test_concise_concrete_chinese_suggestion_is_not_rejected_by_length() -> None:
    parsed = _parser().parse(
        json.dumps(
            {
                "findings": [_finding(suggestion="增加空值检查")],
                "uncertainties": [],
            },
            ensure_ascii=False,
        )
    )

    assert len(parsed.output.findings) == 1


def test_parser_rejects_mismatched_snapshot_binding_at_construction() -> None:
    assignment = compile_review_plan(
        snapshot_id=SNAPSHOT_ID,
        risk_level=RiskLevel.LOW,
        allowed_files=("src/cache.py",),
        allowed_symbols=(),
        allowed_hunks=(),
    ).assignments[0]

    with pytest.raises(ValueError, match="Snapshot binding"):
        ReviewerOutputParser(
            diff_index=_index("S-" + "f" * 64),
            assignment=assignment,
        )


def test_json_schema_forbids_model_status_ids_and_legacy_finding_fields() -> None:
    schema = REVIEWER_OUTPUT_JSON_SCHEMA_V2

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["findings", "uncertainties"]
    finding = schema["properties"]["findings"]["items"]
    assert finding["additionalProperties"] is False
    assert finding["required"] == [
        "claim",
        "severity",
        "path",
        "line",
        "suggestion",
    ]
    encoded = json.dumps(schema)
    for forbidden in (
        "finding_id",
        "status",
        "confidence",
        "impact",
        "evidence_refs",
        "verification_performed",
        "contract_assessments",
    ):
        assert forbidden not in encoded


def test_hydration_helpers_round_trip_valid_output_and_rejection() -> None:
    parsed = _parser().parse(
        json.dumps(
            {
                "findings": [_finding()],
                "uncertainties": ["Could not run the optional integration test."],
            }
        )
    )
    hydrated = reviewer_output_v2_from_dict(parsed.output.to_dict())
    rejection = rejected_reviewer_finding_v2_from_dict(
        {"candidate_index": 3, "reason": "line_not_in_diff"}
    )

    assert hydrated == parsed.output
    assert rejection == RejectedReviewerFinding(3, "line_not_in_diff")

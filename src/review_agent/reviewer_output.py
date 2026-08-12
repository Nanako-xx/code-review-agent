from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping
import unicodedata

from review_agent.diff_artifact import DiffArtifactIndex
from review_agent.review_protocol import (
    FindingSeverity,
    ReviewerAssignment,
    ReviewerFinding,
    ReviewerOutput,
    WireProtocolError,
)
from review_agent.safe_io import SafeIOError, strict_json_loads


REVIEWER_OUTPUT_JSON_SCHEMA_V2: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["findings", "uncertainties"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "claim",
                    "severity",
                    "path",
                    "line",
                    "suggestion",
                ],
                "properties": {
                    "claim": {"type": "string", "minLength": 1},
                    "severity": {
                        "type": "string",
                        "enum": ["blocker", "high", "medium", "low"],
                    },
                    "path": {"type": "string", "minLength": 1},
                    "line": {"type": "integer", "minimum": 1},
                    "suggestion": {"type": "string", "minLength": 1},
                },
            },
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}


_FINDING_FIELDS = {"claim", "severity", "path", "line", "suggestion"}
_REJECTION_REASONS = frozenset(
    {
        "candidate_not_object",
        "finding_fields_invalid",
        "claim_invalid",
        "severity_invalid",
        "path_invalid",
        "line_invalid",
        "suggestion_invalid",
        "suggestion_not_actionable",
        "diff_index_unavailable",
        "path_outside_assignment",
        "path_not_in_diff",
        "line_not_in_diff",
    }
)
_GENERIC_SUGGESTIONS = frozenset(
    {
        "fix",
        "fix it",
        "fix this",
        "please fix",
        "please fix it",
        "please address",
        "address this",
        "resolve this",
        "修复",
        "请修复",
        "请处理",
        "处理一下",
    }
)


class ReviewerOutputError(ValueError):
    pass


class ReviewerOutputEnvelopeError(ReviewerOutputError):
    def __init__(self, code: str) -> None:
        if type(code) is not str or not code:
            raise ValueError("ReviewerOutput error code must be non-empty")
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class RejectedReviewerFinding:
    candidate_index: int
    reason: str

    def __post_init__(self) -> None:
        if type(self.candidate_index) is not int or self.candidate_index < 0:
            raise ReviewerOutputError("candidate_index must be non-negative")
        if self.reason not in _REJECTION_REASONS:
            raise ReviewerOutputError("Finding rejection reason is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RejectedReviewerFinding":
        if type(value) is not dict or set(value) != {"candidate_index", "reason"}:
            raise ReviewerOutputError("Finding rejection schema is invalid")
        return cls(
            candidate_index=value["candidate_index"],
            reason=value["reason"],
        )


@dataclass(frozen=True)
class ReviewerOutputParseResult:
    output: ReviewerOutput
    rejected_findings: tuple[RejectedReviewerFinding, ...] = ()

    def __post_init__(self) -> None:
        if type(self.output) is not ReviewerOutput:
            raise ReviewerOutputError("output must be ReviewerOutput")
        if type(self.rejected_findings) is not tuple or any(
            type(item) is not RejectedReviewerFinding
            for item in self.rejected_findings
        ):
            raise ReviewerOutputError(
                "rejected_findings must contain RejectedReviewerFinding values"
            )


class ReviewerOutputParser:
    def __init__(
        self,
        *,
        diff_index: DiffArtifactIndex | None,
        assignment: ReviewerAssignment | None,
    ) -> None:
        if diff_index is not None and not isinstance(diff_index, DiffArtifactIndex):
            raise ValueError("diff_index must be DiffArtifactIndex or None")
        if assignment is not None and type(assignment) is not ReviewerAssignment:
            raise ValueError("assignment must be ReviewerAssignment or None")
        if (
            diff_index is not None
            and assignment is not None
            and diff_index.snapshot_id != assignment.snapshot_id
        ):
            raise ValueError("Reviewer output Snapshot binding does not match")
        self.diff_index = diff_index
        self.assignment = assignment
        self._allowed_paths = _assignment_paths(assignment)

    def parse(self, raw: str | bytes | bytearray) -> ReviewerOutputParseResult:
        try:
            value = strict_json_loads(raw)
        except SafeIOError as error:
            raise ReviewerOutputEnvelopeError("invalid_json") from error
        if type(value) is not dict:
            raise ReviewerOutputEnvelopeError("top_level_not_object")
        if set(value) != {"findings", "uncertainties"}:
            raise ReviewerOutputEnvelopeError("top_level_fields_invalid")
        findings = value["findings"]
        uncertainties = value["uncertainties"]
        if type(findings) is not list or type(uncertainties) is not list:
            raise ReviewerOutputEnvelopeError("top_level_arrays_invalid")
        if any(not _valid_text(item) for item in uncertainties):
            raise ReviewerOutputEnvelopeError("uncertainties_invalid")

        accepted: list[ReviewerFinding] = []
        rejected: list[RejectedReviewerFinding] = []
        for index, candidate in enumerate(findings):
            finding, reason = self._parse_candidate(candidate)
            if finding is not None:
                accepted.append(finding)
            else:
                rejected.append(
                    RejectedReviewerFinding(
                        candidate_index=index,
                        reason=reason or "finding_fields_invalid",
                    )
                )
        return ReviewerOutputParseResult(
            output=ReviewerOutput(
                findings=tuple(accepted),
                uncertainties=tuple(uncertainties),
            ),
            rejected_findings=tuple(rejected),
        )

    def _parse_candidate(
        self,
        candidate: object,
    ) -> tuple[ReviewerFinding | None, str | None]:
        if type(candidate) is not dict:
            return None, "candidate_not_object"
        if set(candidate) != _FINDING_FIELDS:
            return None, "finding_fields_invalid"
        if not _valid_text(candidate["claim"]):
            return None, "claim_invalid"
        if (
            type(candidate["severity"]) is not str
            or candidate["severity"] not in {item.value for item in FindingSeverity}
        ):
            return None, "severity_invalid"
        if not _valid_path(candidate["path"]):
            return None, "path_invalid"
        if type(candidate["line"]) is not int or candidate["line"] <= 0:
            return None, "line_invalid"
        if not _valid_text(candidate["suggestion"]):
            return None, "suggestion_invalid"
        if not _actionable_suggestion(candidate["suggestion"]):
            return None, "suggestion_not_actionable"

        try:
            finding = ReviewerFinding.from_dict(candidate)
        except WireProtocolError:
            return None, "finding_fields_invalid"
        if self.diff_index is None:
            return None, "diff_index_unavailable"
        if self._allowed_paths and finding.path not in self._allowed_paths:
            return None, "path_outside_assignment"
        file_entry = next(
            (entry for entry in self.diff_index.files if entry.path == finding.path),
            None,
        )
        if file_entry is None:
            return None, "path_not_in_diff"
        if not any(
            hunk.new_count > 0
            and hunk.new_start <= finding.line < hunk.new_start + hunk.new_count
            for hunk in file_entry.hunks
        ):
            return None, "line_not_in_diff"
        return finding, None


def _valid_text(value: object) -> bool:
    return type(value) is str and bool(value.strip()) and "\x00" not in value


def _valid_path(value: object) -> bool:
    if not _valid_text(value):
        return False
    assert isinstance(value, str)
    if value.startswith("/") or value.endswith("/") or "\\" in value or ":" in value:
        return False
    return all(
        part not in {"", ".", ".."} and part == part.strip()
        for part in value.split("/")
    )


def _actionable_suggestion(value: str) -> bool:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", normalized)
    normalized = " ".join(normalized.split())
    return normalized not in _GENERIC_SUGGESTIONS


def _assignment_paths(
    assignment: ReviewerAssignment | None,
) -> frozenset[str]:
    if assignment is None:
        return frozenset()
    paths = set(assignment.targets.files)
    paths.update(
        reference.rsplit("#hunk-", 1)[0]
        for reference in assignment.targets.hunks
        if "#hunk-" in reference
    )
    paths.update(
        reference.split("::", 1)[0]
        for reference in assignment.targets.symbols
        if "::" in reference
    )
    return frozenset(paths)


__all__ = [
    "REVIEWER_OUTPUT_JSON_SCHEMA_V2",
    "RejectedReviewerFinding",
    "ReviewerOutputEnvelopeError",
    "ReviewerOutputError",
    "ReviewerOutputParseResult",
    "ReviewerOutputParser",
]

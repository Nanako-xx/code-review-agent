from __future__ import annotations

from review_agent.reviewer_executor import ReviewerExecutorV2
from review_agent.review_protocol import ReviewerOutput
from review_agent.reviewer_output import RejectedReviewerFinding


class _Loop:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def run(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def test_executor_preserves_completed_loop_result() -> None:
    expected = type(
        "Run",
        (),
        {
            "status": "completed",
            "final_text": '{"findings":[],"uncertainties":[]}',
            "error_code": None,
            "runtime": type("Runtime", (), {"active_elapsed_seconds": 12.0})(),
        },
    )()
    loop = _Loop(result=expected)

    result = ReviewerExecutorV2().execute("ASG-" + "a" * 64, loop)

    assert result.status == "completed"
    assert result.output == expected.final_text
    assert result.reviewer_output == ReviewerOutput(findings=(), uncertainties=())
    assert result.rejected_findings == ()
    assert result.active_elapsed_seconds == 12.0
    assert loop.calls == 1


def test_executor_isolates_runtime_failure_as_one_reviewer_result() -> None:
    loop = _Loop(error=RuntimeError("private provider details"))

    result = ReviewerExecutorV2().execute("ASG-" + "b" * 64, loop)

    assert result.status == "failed"
    assert result.error_code == "reviewer_runtime_error"
    assert result.output is None
    assert "private provider details" not in str(result)


def test_executor_preserves_parsed_output_and_candidate_rejections() -> None:
    parsed = ReviewerOutput(findings=(), uncertainties=("Needs integration test.",))
    rejected = (RejectedReviewerFinding(2, "line_not_in_diff"),)
    expected = type(
        "Run",
        (),
        {
            "status": "completed",
            "final_text": parsed.to_json(),
            "reviewer_output": parsed,
            "rejected_findings": rejected,
            "error_code": None,
            "runtime": type("Runtime", (), {"active_elapsed_seconds": 8.0})(),
        },
    )()

    result = ReviewerExecutorV2().execute(
        "ASG-" + "c" * 64,
        _Loop(result=expected),
    )

    assert result.reviewer_output == parsed
    assert result.rejected_findings == rejected

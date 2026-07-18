"""Product-runtime-free list of lazy Review evaluator exports.

The package root intentionally does not import this module eagerly.  A caller
that wants Review APIs can use this name list to install the same lazy
attribute boundary used by the structured Judge layer without importing the
product Runtime.
"""

REVIEW_PUBLIC_NAMES = (
    "REVIEW_EVALUATION_SCHEMA_VERSION",
    "REVIEW_EVALUATOR_REVISION",
    "REVIEW_MATCH_POLICY_VERSION",
    "REVIEW_LOCATION_POLICY_VERSION",
    "MAX_REVIEW_EVALUATION_BYTES",
    "MAX_REVIEW_FINDINGS",
    "MAX_REVIEW_TRUTH_FINDINGS",
    "MAX_REVIEW_CANDIDATES",
    "MAX_REVIEW_LOCATION_AUDITS",
    "MAX_REVIEW_JUDGE_REQUESTS",
    "ReviewEvaluationError",
    "ReviewEvaluationStatus",
    "ReviewEvaluationPhase",
    "ReviewTruthKind",
    "FindingMatchKind",
    "FindingDisposition",
    "FindingResolution",
    "EvidenceSupportResolution",
    "ReviewReasonCode",
    "ReviewLimitScope",
    "ReviewContextBundle",
    "ReviewFindingContextEntry",
    "ReviewPairContextEntry",
    "ReviewLimitFailure",
    "LocationAuditRecord",
    "ReviewCandidateRecord",
    "ReviewAssignmentRecord",
    "FindingOutcome",
    "ReviewJudgeRequestRecord",
    "ReviewJudgeDecisionReceipt",
    "ReviewJudgeFailureReceipt",
    "ReviewJudgeUngradedReceipt",
    "ReviewCoverage",
    "ReviewMetricInputs",
    "ReviewEvaluationResult",
    "ReviewEvaluator",
)

__all__ = ["REVIEW_PUBLIC_NAMES"]

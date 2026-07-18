"""Product-runtime-free list of lazy public Report exports."""

REPORT_PUBLIC_NAMES = (
    "RUN_REPORT_SUMMARY_SCHEMA_VERSION",
    "TRIAL_INSPECTION_SCHEMA_VERSION",
    "REPORT_REVISION",
    "REDACTED_ARTIFACT_PROJECTION_VERSION",
    "MAX_REPORT_BYTES",
    "MAX_INSPECTION_BYTES",
    "ReportError",
    "TrialEvaluationSource",
    "RunReportSummary",
    "TrialInspection",
    "ReportBuilder",
    "render_run_markdown",
    "render_trial_markdown",
)

__all__ = ["REPORT_PUBLIC_NAMES"]

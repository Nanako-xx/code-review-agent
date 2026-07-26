"""Product-runtime-free list of lazy public Task 15 artifact exports."""

ANALYSIS_ARTIFACT_PUBLIC_NAMES = (
    "ANALYSIS_ARTIFACT_KINDS",
    "ANALYSIS_RECEIPT_SCHEMA_VERSION",
    "MAX_ANALYSIS_ARTIFACTS",
    "AnalysisSourceBinding",
    "AnalysisArtifactRef",
    "AnalysisReceipt",
    "AnalysisArtifactStore",
    "derive_analysis_artifact_id",
    "bind_analysis_source",
)

STATISTICS_PUBLIC_NAMES = (
    "RUN_STATISTICS_SCHEMA_VERSION",
    "STATISTICS_ALGORITHM_VERSION",
    "MAX_BOOTSTRAP_SEED",
    "MAX_BOOTSTRAP_ITERATIONS",
    "MAX_BOOTSTRAP_CASES",
    "MAX_RUN_BOOTSTRAP_DRAWS",
    "StatisticsError",
    "MetricUnit",
    "MetricDirection",
    "StatisticsMetricStatus",
    "DispersionNullReason",
    "ConfidenceIntervalStatus",
    "StatisticsPolicyV1",
    "MetricSourceCoverageV1",
    "StatisticsCoverageV1",
    "DerivedCaseContributionV1",
    "CaseContributionV1",
    "BootstrapCoverageV1",
    "ConfidenceIntervalV1",
    "DispersionCoverageV1",
    "DispersionV1",
    "TrialMetricProjectionV1",
    "StatisticsMetricV1",
    "RunStatisticsV1",
    "paired_bootstrap_interval",
    "compute_run_statistics",
)

ANALYSIS_PUBLIC_NAMES = ANALYSIS_ARTIFACT_PUBLIC_NAMES + STATISTICS_PUBLIC_NAMES

__all__ = [
    "ANALYSIS_ARTIFACT_PUBLIC_NAMES",
    "STATISTICS_PUBLIC_NAMES",
    "ANALYSIS_PUBLIC_NAMES",
]

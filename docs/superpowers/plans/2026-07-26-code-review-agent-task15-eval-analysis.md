
# Task 15 Evaluation Analysis Lifecycle Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

Goal: Add source-bound repeated-trial statistics, strict baseline/candidate comparison, independent Judge calibration, and pre-registered regression gates to the existing Eval v2 harness.

Architecture: Keep prepare -> run-agent -> evaluate unchanged. Add an analysis layer that consumes verified RunEvaluationBundle objects and publishes create-only Analysis Artifacts in a separate analysis root. Four bounded services own statistics, comparison, calibration, and gates; CLI code only composes them and renders stable JSON/text output.

Tech Stack: Python 3.11+, stdlib dataclasses/enums/json/hashlib/statistics, existing ArtifactStore, EvaluationOrchestrator, MetricsAggregator, CoreMetric, RunEvaluationBundle, pytest, canonical JSON/digest helpers, and the Windows-safe path and atomic-publication rules already used by Eval v2.

---

## Starting context and constraints

Work from a clean worktree based on merged master commit d33775f. Do not develop in the user's dirty codex/public-benchmark-adapters checkout. Use a new implementation worktree created with superpowers:using-git-worktrees.

Reuse these existing contracts:

- src/review_agent_eval/orchestrator.py: EvaluationOrchestrator.load_run_evaluation and RunEvaluationBundle source-bound hydration.
- src/review_agent_eval/artifacts.py: ArtifactStore, immutable Run Evaluation namespaces, digest/size/path security, and existing CLI artifact roots.
- src/review_agent_eval/metrics.py: CoreMetric, MetricContribution, MetricAggregate, CaseScore, AggregateScore, MetricsAggregator, MetricSourceStatus, MetricNullReason, and MetricsPolicy.
- src/review_agent_eval/report.py: RunReportSummary, ReportBuilder.hydrate_summary, and pure Markdown rendering.
- src/review_agent_eval/judge.py and review_evaluator.py: typed Judge profiles, request/output artifacts, FindingMatchRelation, NovelFactuality, EvidenceSupport, and Judge failure taxonomy.
- src/review_agent_eval/cli.py: artifact-store construction, root validation, repository/frozen replay composition, stable output envelopes, and existing error categories.

Preserve these decisions from the approved design:

- Truth remains the pre-authored Intent/Finding/known-invalid/Evidence data. Task 15 does not invent a new Case Pass or Overall Score.
- Core Regression Cases and Private Held-out data may block release. AACR-Bench, SWE-PRBench, and unpromoted Synthetic data are diagnostic-only.
- Formal baseline/candidate comparisons use at least 3 Trials per Case and equal Trial counts.
- Baseline/candidate share truth, Target, Evaluator/Judge/Rubric, Authority, and isolation inputs; only Agent-side identity/configuration may differ.
- Judge Calibration is separate for Intent, Finding, novel factuality, and Evidence support.
- Gate Policy is frozen before candidate execution and contains no code-default numeric thresholds.
- Missing authority is not_scorable, never zero or pass. Failed, unknown, and ungraded coverage remains visible.
- Analysis commands never run Agent/Judge, acquire datasets/repositories, or modify .eval-runs.

## File and API map

| Path | Responsibility |
| --- | --- |
| src/review_agent_eval/analysis_artifacts.py | Task 15 schemas, source bindings, create-only analysis publication/loading |
| src/review_agent_eval/statistics.py | Trial-index statistics and deterministic Case-clustered bootstrap |
| src/review_agent_eval/comparison.py | Compatibility projection, one-to-one pairing, metric/case deltas |
| src/review_agent_eval/calibration.py | Blind package selection, human labels, Judge-vs-human calibration |
| src/review_agent_eval/gates.py | Pre-registered policy, typed constraints, gate evaluation |
| src/review_agent_eval/analysis_exports.py | Lazy public export list for Task 15 symbols |
| src/review_agent_eval/artifacts.py | Reuse or extract safe create-only publication primitives for the separate analysis root; do not change the Run artifact layout |
| src/review_agent_eval/cli.py | Add compare, calibrate, and gate parsing and composition only |
| src/review_agent_eval/__init__.py | Expose new public symbols lazily without importing product Runtime |
| tests/eval/test_analysis_artifacts.py | Analysis schema, binding, tamper, atomic publication, and path security |
| tests/eval/test_statistics.py | Repeated-trial aggregation, coverage, standard deviation, and bootstrap |
| tests/eval/test_comparison.py | Strict compatibility and paired comparison |
| tests/eval/test_calibration.py | Blind package, label, and calibration |
| tests/eval/test_regression_gates.py | Policy freeze and per-condition gate |
| tests/eval/test_task15_cli.py | CLI parser, JSON envelope, read/write boundary, and CI status |
| tests/eval/test_task15_e2e.py | Scripted Core 3-Trial lifecycle through compare/calibrate/gate |
| tests/eval/test_architecture_boundaries.py | Analysis-layer import and product-runtime boundaries |
| docs/eval-system.md | Current Task 15 command and artifact documentation |

Do not modify product Runtime, Session, Memory, Risk, GitHub/PR integration, or public dataset acquisition code for this plan.

## Task 1: Analysis Artifacts and source-bound loading

Files:

- Create: src/review_agent_eval/analysis_artifacts.py
- Create: src/review_agent_eval/analysis_exports.py
- Modify: src/review_agent_eval/artifacts.py
- Modify: src/review_agent_eval/__init__.py
- Create: tests/eval/test_analysis_artifacts.py

- [ ] Step 1: Add contract tests for immutable Analysis Artifacts.

Add tests with these names:

    test_analysis_bundle_is_create_only_and_reloads_identically
    test_analysis_receipt_binds_run_evaluation_and_source_digests
    test_analysis_tamper_fails_closed
    test_analysis_rejects_traversal_symlink_and_unknown_artifact_names
    test_analysis_write_does_not_create_missing_read_only_root

Use existing ArtifactStore and fixture helpers to create a small committed Run Evaluation. Assert that a second publish with different bytes raises the existing conflict/integrity category, while a second publish with identical bytes returns the original receipt. Use short tmp_path roots and do not use C:/tmp.

- [ ] Step 2: Define typed analysis schemas.

Create strict canonical dataclasses:

    @dataclass(frozen=True)
    class AnalysisSourceBinding:
        run_id: str
        evaluation_id: str
        summary_id: str
        summary_digest: str
        run_config_digest: str
        case_snapshot_digest: str
        trial_score_digests: tuple[str, ...]

    @dataclass(frozen=True)
    class AnalysisArtifactRef:
        kind: str
        artifact_id: str
        relative_path: str
        sha256: str
        size_bytes: int

    @dataclass(frozen=True)
    class AnalysisReceipt:
        schema_version: str
        kind: str
        artifact_id: str
        source_bindings: tuple[AnalysisSourceBinding, ...]
        artifacts: tuple[AnalysisArtifactRef, ...]
        algorithm_digest: str

Enforce exact keys, canonical ordering, size limits, safe path segments, digest consistency, and stable IDs derived from complete identity payloads. Distinguish source digests from opaque display projections.

- [ ] Step 3: Add a separate create-only analysis publisher.

Extend low-level safe publication support in artifacts.py only as needed to share directory-chain verification, no-follow path checks, atomic write, fsync, and no-overwrite behavior. Keep ArtifactStore's Run API and paths unchanged.

Implement AnalysisArtifactStore with this surface:

    AnalysisArtifactStore(root, create_root, max_file_bytes, max_total_read_bytes)
    AnalysisArtifactStore.publish_json_bundle(kind, artifact_id, files, receipt)
    AnalysisArtifactStore.load_json_bundle(kind, artifact_id)

The publish method returns AnalysisReceipt and the load method returns a decoded JSON mapping.

publish_json_bundle must publish child files before the receipt commit marker, refuse unknown or duplicate names, validate all planned bytes before writing, and return an existing identical bundle on resume. load_json_bundle is read-only and must not create the root. Use the same Windows reparse/junction/hardlink/short-path protections as Eval v2.

- [ ] Step 4: Add verified source loading helpers and exports.

Add a helper that accepts an already hydrated RunEvaluationBundle, its EvalRunConfig, and RunCaseSnapshot, verifies all source digests, and returns an immutable source binding. The analysis store must not construct an Agent adapter or Judge.

Add Task 15 names to analysis_exports.py and lazy-load them from __init__.py. Importing analysis_artifacts must not import review_agent product Runtime modules.

- [ ] Step 5: Run focused tests and commit.

Run:

    $env:PYTHONDONTWRITEBYTECODE='1'; & 'D:/Anaconda/envs/MINIST/python.exe' -m pytest tests/eval/test_analysis_artifacts.py -q -p no:cacheprovider --basetemp 'D:/tmp/t15a'

Expected: all analysis artifact contract tests pass.

Commit only Task 1 files:

    git add src/review_agent_eval/analysis_artifacts.py src/review_agent_eval/analysis_exports.py src/review_agent_eval/artifacts.py src/review_agent_eval/__init__.py tests/eval/test_analysis_artifacts.py
    git commit -m "feat(eval): add task 15 analysis artifacts"

## Task 2: Repeated-trial Statistics

Files:

- Create: src/review_agent_eval/statistics.py
- Modify: src/review_agent_eval/analysis_exports.py
- Create: tests/eval/test_statistics.py

- [ ] Step 1: Add tests for numerator/denominator and coverage semantics.

Build three-Trial fixtures using existing test_metrics.py helpers. Cover a completed Trial, an Agent failure, a Judge unknown, a Judge failure, and an authority-excluded location metric. Add tests with these names:

    test_statistics_reaggregates_rates_from_numerators_and_denominators
    test_statistics_keeps_failed_and_ungraded_trials_in_coverage
    test_statistics_reports_each_trial_index_without_best_trial_selection
    test_statistics_marks_missing_authority_not_scorable_not_zero
    test_statistics_bootstrap_is_deterministic_for_fixed_policy

Prove that the all-Trial rate is summed numerator divided by summed denominator, not the arithmetic mean of Trial percentages.

- [ ] Step 2: Define Statistics Policy and result types.

Add:

    @dataclass(frozen=True)
    class StatisticsPolicyV1:
        algorithm_version: str
        bootstrap_seed: int
        bootstrap_iterations: int
        confidence_level_ppm: int

    @dataclass(frozen=True)
    class RunStatisticsV1:
        schema_version: str
        source_binding: AnalysisSourceBinding
        trial_count: int
        metrics: tuple[StatisticsMetricV1, ...]
        trial_metrics: tuple[TrialMetricProjectionV1, ...]
        bootstrap_policy: StatisticsPolicyV1

Validate seed, iteration, and confidence bounds, metric kind, status, and unit. Do not add a binary Case Pass or Overall Score field.

- [ ] Step 3: Implement source-bound statistics.

Implement:

    def compute_run_statistics(
        bundle: RunEvaluationBundle,
        *,
        run_config: EvalRunConfig,
        case_snapshot: RunCaseSnapshot,
        policy: StatisticsPolicyV1,
    ) -> RunStatisticsV1

Use existing TrialScore and MetricsAggregator as the only metric source. For every CoreMetric, retain numerator, denominator, value, source status, coverage, and direction. Compute Trial-index projections with the same aggregation policy. Report standard deviation only over available, scorable replicate values and retain counts excluded for each reason.

Implement Case-clustered paired-bootstrap support as a pure stdlib function:

    def paired_bootstrap_interval(
        case_contributions: Sequence[CaseContributionV1],
        *,
        seed: int,
        iterations: int,
        confidence_level_ppm: int,
    ) -> ConfidenceIntervalV1

Resample Case records with replacement and recompute metric numerator/denominator for every replicate. Return a stable null/coverage reason for zero denominators or insufficient Case populations.

- [ ] Step 4: Bind Statistics to Comparison Artifacts.

RunStatisticsV1 is embedded as baseline_statistics or candidate_statistics in RunComparisonV1, and is source-bound and algorithm-versioned. Do not add a separate top-level directory that duplicates the approved Analysis Store layout.

- [ ] Step 5: Run tests and commit.

Run:

    $env:PYTHONDONTWRITEBYTECODE='1'; & 'D:/Anaconda/envs/MINIST/python.exe' -m pytest tests/eval/test_statistics.py tests/eval/test_metrics.py -q -p no:cacheprovider --basetemp 'D:/tmp/t15b'

Expected: new statistics tests and all existing metrics tests pass.

Commit:

    git add src/review_agent_eval/statistics.py src/review_agent_eval/analysis_exports.py tests/eval/test_statistics.py
    git commit -m "feat(eval): add repeated trial statistics"

## Task 3: Strict Paired Comparison

Files:

- Create: src/review_agent_eval/comparison.py
- Modify: src/review_agent_eval/analysis_exports.py
- Modify: src/review_agent_eval/analysis_artifacts.py
- Create: tests/eval/test_comparison.py

- [ ] Step 1: Add compatibility and pairing tests.

Use two source-bound scripted Run Evaluations with the same fixture Case Snapshot. Add:

    test_compare_pairs_by_case_digest_and_trial_index
    test_compare_does_not_drop_failed_or_ungraded_trial_pairs
    test_compare_reports_metric_and_case_deltas_without_overall_score
    test_compare_rejects_case_digest_trial_count_and_evaluator_mismatch
    test_compare_allows_agent_identity_change_and_records_agent_delta
    test_compare_reuses_fixed_bootstrap_policy

Mutate one source binding at a time and assert not_comparable includes the exact field name. Assert that Judge coverage change is separate from metric delta.

- [ ] Step 2: Define comparison types and compatibility projection.

Add:

    @dataclass(frozen=True)
    class ComparisonPolicyV1:
        schema_version: str
        statistics_policy: StatisticsPolicyV1
        required_case_fields: tuple[str, ...]
        required_evaluator_fields: tuple[str, ...]

    @dataclass(frozen=True)
    class RunComparisonV1:
        schema_version: str
        comparison_id: str
        status: str
        baseline_binding: AnalysisSourceBinding
        candidate_binding: AnalysisSourceBinding
        compatibility: ComparisonCompatibilityV1
        baseline_statistics: RunStatisticsV1
        candidate_statistics: RunStatisticsV1
        metric_deltas: tuple[MetricDeltaV1, ...]
        case_deltas: tuple[CaseDeltaV1, ...]
        incompatibilities: tuple[str, ...]
        algorithm_digest: str

The compatibility projection excludes Run ID, Evaluation ID, Agent config digest, Agent provider/model, Prompt/config identity, and other intended Agent-side differences. It includes Suite/Case digests, Trial count, Target kind, Wire Contract, Materialization/isolation, Evaluator/Judge/Rubric, Metrics Policy, Truth Completeness, Novel Finding Policy, and Metric Authority.

- [ ] Step 3: Implement pairing and delta calculation.

Implement:

    def compare_runs(
        baseline: VerifiedRunEvaluation,
        candidate: VerifiedRunEvaluation,
        policy: ComparisonPolicyV1,
    ) -> RunComparisonV1

VerifiedRunEvaluation is a composition object containing the hydrated RunEvaluationBundle, EvalRunConfig, RunCaseSnapshot, and source binding.

The function must:

1. compare the compatibility projection;
2. construct the exact key (task_id, case_version, canonical_case_digest, trial_index);
3. compute each metric's baseline value, candidate value, and absolute delta;
4. classify metric and Case contributions as improved, regressed, or unchanged according to metric direction;
5. preserve missing, failed, ungraded, and not_scorable coverage;
6. compute the fixed Case-clustered bootstrap interval;
7. return not_comparable without deltas when compatibility fails.

No comparison function may call an Agent adapter, Judge, acquisition client, or product Runtime.

- [ ] Step 4: Publish and reload the Comparison Artifact.

Add AnalysisArtifactStore.publish_comparison() and load_comparison() wrappers. Validate RunComparisonV1, nested statistics, source bindings, Policy digest, and receipt. Hydration must reload both source Evaluations and byte-compare the recomputed canonical result.

- [ ] Step 5: Run tests and commit.

Run:

    $env:PYTHONDONTWRITEBYTECODE='1'; & 'D:/Anaconda/envs/MINIST/python.exe' -m pytest tests/eval/test_comparison.py tests/eval/test_statistics.py tests/eval/test_score_compatibility_v2.py -q -p no:cacheprovider --basetemp 'D:/tmp/t15c'

Expected: all comparison, statistics, and existing compatibility tests pass.

Commit:

    git add src/review_agent_eval/comparison.py src/review_agent_eval/analysis_artifacts.py src/review_agent_eval/analysis_exports.py tests/eval/test_comparison.py
    git commit -m "feat(eval): compare compatible repeated evaluations"

## Task 4: Blind Judge Calibration

Files:

- Create: src/review_agent_eval/calibration.py
- Modify: src/review_agent_eval/analysis_artifacts.py
- Modify: src/review_agent_eval/analysis_exports.py
- Create: tests/eval/test_calibration.py

- [ ] Step 1: Add blind-package and label-binding tests.

Use existing TrialEvaluationBundle.judge_input and judge_output fixtures. Add:

    test_export_hides_agent_baseline_candidate_and_judge_decision
    test_selection_policy_is_seeded_and_recorded
    test_human_label_requires_package_and_item_digest
    test_disputed_label_requires_adjudication_before_gate_eligibility
    test_calibration_profiles_are_scored_independently
    test_missing_labels_are_pending_not_fake_agreement
    test_kappa_and_confusion_matrix_are_reproducible

Inspect the exported blind payload and assert forbidden identity keys and persisted Judge decision fields are absent. Do not write the full package into the repository.

- [ ] Step 2: Define Profile, Package, and Label schemas.

Use the existing typed Judge vocabulary:

- Intent: IntentJudgeRelation plus dimension binding;
- Finding: FindingMatchRelation with severity/actionability when present;
- Novel factuality: NovelFactuality with severity/actionability when present;
- Evidence support: EvidenceSupport.

Add strict CalibrationSelectionPolicyV1, CalibrationPackageV1, HumanLabelSetV1, and CalibrationResultV1 dataclasses.

CalibrationPackageV1 stores Profile, Rubric/Context versions, stable calibration_item_id, blinded request payload, allowed labels, selection seed, and source digest. It does not store Agent identity, baseline/candidate label, expected winner, or Judge decision. Full code-bearing payload is written only to explicit external package output; Analysis Store keeps manifest, payload digest, and receipt.

HumanLabelSetV1 binds exactly one label per item, reviewer provenance, blind attestation, timestamps, and optional adjudication ref. Duplicate, missing, unknown, or out-of-vocabulary labels fail closed.

- [ ] Step 3: Implement selection and scoring.

Implement:

    def export_calibration_package(
        evaluation: VerifiedRunEvaluation,
        *,
        profile: JudgeTask,
        policy: CalibrationSelectionPolicyV1,
        output_root: Path,
    ) -> CalibrationPackageV1

    def score_calibration(
        evaluation: VerifiedRunEvaluation,
        *,
        package: CalibrationPackageV1,
        labels: HumanLabelSetV1,
    ) -> CalibrationResultV1

Selection includes unknowns, high/critical fabricated outcomes, deterministic/Judge conflicts, and fixed-seed stratified samples of normal categories. calibration_item_id is derived from Profile, Rubric/Context version, and canonical blinded item payload, not Run ID or Agent identity.

Compute independently per Profile:

- labeled and eligible coverage;
- graded, semantic unknown, Judge failure, and ungraded counts;
- confusion matrix;
- exact agreement;
- per-class precision/recall;
- Cohen's kappa;
- disagreement item refs.

Status is exact: no labels -> pending_human_labels; labels below configured minimum -> insufficient_coverage; configured quality failure -> failed_thresholds; only a fully labeled threshold-passing Profile -> gate_eligible. No fixture shortcut may emit gate_eligible without human-label provenance.

calibrate reads existing Judge output and never invokes Judge. A new Judge version is evaluated first through existing evaluate/re-evaluate, then passed to score_calibration.

- [ ] Step 4: Run tests and commit.

Run:

    $env:PYTHONDONTWRITEBYTECODE='1'; & 'D:/Anaconda/envs/MINIST/python.exe' -m pytest tests/eval/test_calibration.py tests/eval/test_judge_rubrics.py tests/eval/test_core_human_review.py -q -p no:cacheprovider --basetemp 'D:/tmp/t15d'

Expected: new calibration tests and existing Judge/human-review contract tests pass.

Commit:

    git add src/review_agent_eval/calibration.py src/review_agent_eval/analysis_artifacts.py src/review_agent_eval/analysis_exports.py tests/eval/test_calibration.py
    git commit -m "feat(eval): calibrate semantic judge profiles"

## Task 5: Pre-registered Regression Gates

Files:

- Create: src/review_agent_eval/gates.py
- Modify: src/review_agent_eval/analysis_artifacts.py
- Modify: src/review_agent_eval/analysis_exports.py
- Create: tests/eval/test_regression_gates.py

- [ ] Step 1: Add Policy and decision tests.

Add:

    test_gate_policy_binds_candidate_run_plan_before_results_exist
    test_gate_policy_cannot_be_overwritten_or_rebound
    test_gate_checks_absolute_and_baseline_delta_constraints
    test_gate_reports_case_and_trial_refs_for_each_failure
    test_gate_requires_calibration_for_semantic_metrics
    test_gate_marks_public_or_unscorable_data_diagnostic_only
    test_gate_returns_promote_block_and_ineligible_without_overall_score

- [ ] Step 2: Define typed Gate schemas.

Add:

    @dataclass(frozen=True)
    class MetricConstraintV1:
        metric: CoreMetric
        scope: str
        operator: str
        threshold: int | str
        unit: str
        required: bool
        min_coverage_ppm: int | None

    @dataclass(frozen=True)
    class GatePolicyV1:
        schema_version: str
        policy_id: str
        baseline_binding: AnalysisSourceBinding
        candidate_run_id: str
        candidate_run_config_digest: str
        case_snapshot_digest: str
        trial_count: int
        comparison_policy_digest: str
        calibration_result_digests: tuple[str, ...]
        eligibility: str
        constraints: tuple[MetricConstraintV1, ...]

    @dataclass(frozen=True)
    class GateResultV1:
        schema_version: str
        gate_result_id: str
        policy_digest: str
        comparison_id: str
        decision: str
        checks: tuple[GateCheckV1, ...]

Use metric metadata to reject a percentage threshold for a count/mean metric, require units for usage/cost, and reject negative, NaN, or infinite values. Leave threshold collections empty only when Policy explicitly intends diagnostic-only analysis; do not inject defaults.

- [ ] Step 3: Implement policy freeze and evaluation.

Implement:

    def prepare_gate_policy(
        baseline: VerifiedRunEvaluation,
        candidate_run_config: EvalRunConfig,
        *,
        policy: GatePolicyV1,
    ) -> GatePolicyV1

    def evaluate_gate(
        policy: GatePolicyV1,
        comparison: RunComparisonV1,
        calibrations: Mapping[str, CalibrationResultV1],
    ) -> GateResultV1

prepare_gate_policy verifies candidate Run Plan and source bindings without requiring candidate Submission/Score. evaluate_gate refuses a different candidate Run ID, Comparison Policy, Case Snapshot, Trial count, or Calibration digest.

Evaluate every configured constraint. A missing required semantic Calibration Profile returns insufficient_coverage/ineligible; a metric with missing Authority returns not_scorable; an unconfigured metric is not a hidden pass. Hard Case/Trial failures include opaque Case/Trial references and actual/threshold values.

Decisions are promote, block, or ineligible. A decision is not an Overall Score. The persisted Gate Result remains canonical; CLI ci mode only changes process exit behavior.

- [ ] Step 4: Run tests and commit.

Run:

    $env:PYTHONDONTWRITEBYTECODE='1'; & 'D:/Anaconda/envs/MINIST/python.exe' -m pytest tests/eval/test_regression_gates.py tests/eval/test_metric_authority.py tests/eval/test_score_compatibility_v2.py -q -p no:cacheprovider --basetemp 'D:/tmp/t15e'

Expected: Policy, authority, compatibility, and existing score tests pass.

Commit:

    git add src/review_agent_eval/gates.py src/review_agent_eval/analysis_artifacts.py src/review_agent_eval/analysis_exports.py tests/eval/test_regression_gates.py
    git commit -m "feat(eval): add preregistered regression gates"

## Task 6: CLI, composition, E2E, and documentation

Files:

- Modify: src/review_agent_eval/cli.py
- Modify: src/review_agent_eval/__init__.py
- Create: tests/eval/test_task15_cli.py
- Create: tests/eval/test_task15_e2e.py
- Modify: tests/eval/test_architecture_boundaries.py
- Modify: docs/eval-system.md

- [ ] Step 1: Add CLI parser and JSON contract tests.

Extend the parser test to require:

    compare
    calibrate export
    calibrate import-labels
    calibrate score
    gate prepare
    gate evaluate

Assert that all new commands accept analysis-root, emit the existing canonical CLI envelope, and do not add Agent/Judge/acquisition arguments that could invoke a provider. Add tests for missing roots, malformed IDs, not_comparable, pending_human_labels, promote, block, ineligible, and ci exit behavior.

- [ ] Step 2: Compose source-bound analysis commands.

Add analysis-store construction and verified-evaluation loading helpers that reuse:

- the existing read-only artifact-store loader for .eval-runs;
- EvaluationOrchestrator.load_run_evaluation for source-bound hydration;
- CaseBank/Suite loading for the same Case Snapshot;
- AnalysisArtifactStore for analysis publication;
- no Agent adapter factory, Judge factory, Repository acquisition client, or network provider.

Handlers have only these responsibilities:

- compare: load two verified Evaluations, load Comparison Policy, call compare_runs, publish and emit.
- calibrate export: select and write the external blind Package plus manifest/receipt.
- calibrate import-labels: validate external labels and publish the Label Set.
- calibrate score: load Judge Evaluation and Label Set, call score_calibration, publish the Result.
- gate prepare: load candidate Run Config/Plan and publish immutable Policy.
- gate evaluate: load Policy, Comparison, and Calibration Results, call evaluate_gate, publish and emit.

- [ ] Step 3: Add the local scripted E2E.

Use existing Core CLI fixture and scripted Agent/Judge helpers to run:

    prepare baseline with trial-count=3
    run-agent baseline
    evaluate baseline
    prepare candidate with trial-count=3
    create Gate Policy before candidate execution
    run-agent candidate
    evaluate candidate with the same Evaluator/Judge config
    compare baseline and candidate
    export/import a calibration fixture
    score calibration
    evaluate gate
    inspect resulting Analysis Artifacts

Assert that three Trial scores exist per Case, failed/unknown coverage remains visible, pairing uses Case digest and Trial index, fixture labels cannot become real-human gate_eligible without explicit test provenance, reloading recomputes identical canonical bytes, and compare/calibrate/gate do not change .eval-runs.

Use D:/tmp/t15e2e as the base directory.

- [ ] Step 4: Add architecture and security regressions.

Assert that Task 15 modules do not import review_agent.runtime, Session, Memory, Risk, product Reviewer orchestration, acquisition clients, or network adapters. Add path tests for external Package output, analysis-root traversal, symlink/junction/reparse/hardlink, credential-bearing labels/notes, and raw private context entering ordinary reports.

- [ ] Step 5: Update docs/eval-system.md.

Remove the statement that compare/calibrate/gate are unavailable. Document the Analysis Store, external Calibration Package boundary, baseline -> Policy -> candidate -> compare -> gate order, the truth-to-metric flow, Core/Private release_blocking versus public/Synthetic diagnostic_only, not_scorable, pending_human_labels, block and ineligible semantics, and the remaining real-model/Reviewer B/Private Held-out external gates.

- [ ] Step 6: Run targeted and complete Eval regression.

Run the Task 15 set:

    $env:PYTHONDONTWRITEBYTECODE='1'; & 'D:/Anaconda/envs/MINIST/python.exe' -m pytest tests/eval/test_analysis_artifacts.py tests/eval/test_statistics.py tests/eval/test_comparison.py tests/eval/test_calibration.py tests/eval/test_regression_gates.py tests/eval/test_task15_cli.py tests/eval/test_task15_e2e.py tests/eval/test_architecture_boundaries.py -q -p no:cacheprovider --basetemp 'D:/tmp/t15final'

Expected: all new and directly affected tests pass with no unexpected skips.

Then run the complete Eval suite:

    $env:PYTHONDONTWRITEBYTECODE='1'; & 'D:/Anaconda/envs/MINIST/python.exe' -m pytest tests/eval -q -p no:cacheprovider --basetemp 'D:/tmp/t15full'

Expected: all existing Eval v2 tests remain green. Any skip must be an already documented environment/capability skip. Do not use C:/tmp.

- [ ] Step 7: Commit the integration boundary.

Before committing, inspect the staged file list and ensure no fixture outputs, __pycache__, .eval-runs, .eval-analyses, or external Calibration Package is staged:

    git status --short
    git diff --check
    git add src/review_agent_eval/cli.py src/review_agent_eval/__init__.py tests/eval/test_task15_cli.py tests/eval/test_task15_e2e.py tests/eval/test_architecture_boundaries.py docs/eval-system.md
    git diff --cached --name-status
    git commit -m "feat(eval): expose task 15 analysis lifecycle"

## Final verification checklist

- [ ] Existing prepare -> run-agent -> evaluate -> inspect behavior and artifact paths are unchanged.
- [ ] Statistics aggregate existing metric numerators/denominators and expose all Trial/Coverage states.
- [ ] Comparison pairs exact Case/Trial identities and refuses incompatible truth/evaluator/target inputs.
- [ ] Comparison contains no Overall Score or Case Pass.
- [ ] Calibration hides Judge identity/results from human labels and has four independent Profiles.
- [ ] No fixture or fake label can be represented as real-human gate_eligible.
- [ ] Gate Policy is immutable and created before candidate results exist.
- [ ] Gate threshold values are explicit Policy inputs, not code defaults.
- [ ] Public/Synthetic data cannot be marked release-blocking without an explicit trusted eligibility binding.
- [ ] not_scorable and missing Coverage never silently become zero or pass.
- [ ] Analysis commands are Agent/Judge/acquisition-free and source-bound.
- [ ] All Analysis Artifacts are canonical, create-only, tamper-evident, and safe on Windows.
- [ ] Targeted Task 15 tests and complete tests/eval pass in D-drive short temporary roots.

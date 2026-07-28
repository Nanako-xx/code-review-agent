# Evaluation system

`review-agent-eval` is the local evaluation harness for the code review agent. It evaluates four related outputs:

- reconstructed Intent and clarification behavior;
- Review findings, including expected, novel, and invalid findings;
- Evidence integrity and support;
- authority-aware quality, failure, Judge, usage, and coverage metrics.

The harness deliberately does **not** calculate an Overall Score. Read the individual metrics together with their eligibility, coverage, and compatibility partition. A null or `not_scorable` metric is not a zero.

## Install

The core package requires Python 3.11 or newer:

```console
python -m pip install .
```

AACR-Bench and SWE-PRBench import support uses PyArrow. Install the optional `eval-public` extra when preparing those datasets:

```console
python -m pip install ".[eval-public]"
```

The installed evaluation entry point is separate from the product command:

```console
review-agent-eval --help
```

## Targets and replay identity

Every v2 Case has one discriminated `ReviewTarget`.

### Repository Target

A Repository Target binds a repository descriptor, immutable base and head revisions, and the review request. `prepare` may acquire or verify the repository cache. `run-agent` and `evaluate` then operate cache-only. Each Trial receives a private materialization and a manifest of the Agent-visible files.

Repository Evidence refers to the bound revision and repository file or diff. The Evidence checker and Judge context read the same verified repository replay that was materialized for the Trial.

### Frozen Context Target

A Frozen Context Target binds a prepared bundle, record ID, rendered-content hash and size, and source-binding digest. It does not pretend that the original repository is available. The Trial materializes the verified rendered context, and Frozen Evidence addresses that context by `context_ref` and rendered line range.

The built-in current-Agent adapter supports Repository Targets only. A Frozen Context Suite therefore fails current-Agent capability preflight. A subprocess adapter using the v2 wire contract can declare Frozen Context support and run it.

### One identity chain

For both target kinds, the Run and Trial plans bind the Case digest, EvalInput digest, Target digest, wire contract, Adapter capabilities, and Trial ID. Materialization produces a `target_materialization_id`, which is carried by Target access, the Agent invocation, the Submission, Evidence, evaluator replay, and Judge context.

Replacing a Target after binding, returning the wrong input/Trial/materialization identity, changing a replay, or mixing Repository and Frozen evidence fails closed. Resume and re-evaluation reuse the receipt-bound Submission and Target replay; they do not select a new Target.

## Preparing public data

`prepare-public` has exactly one mode: `local-import`. It verifies bytes that have already been acquired, checks the supplied source and filter manifests and expected digests, and publishes an immutable Suite.

The harness does not download AACR-Bench or SWE-PRBench. External acquisition, licensing checks, storage, and transport are the operator's responsibility. The harness makes no pinned-download claim and does not turn a source URL or revision label into proof of downloaded bytes.

The required arguments differ by dataset profile. Use the command's real help rather than copying a generic example:

```console
review-agent-eval prepare-public --help
```

The only accepted dataset names are `aacr-bench` and `swe-prbench`; the only accepted mode is `--mode local-import`.

## Lifecycle

The lifecycle is deliberately separated so that Agent execution cannot silently perform evaluation and re-evaluation cannot rerun the Agent:

1. `prepare` verifies the Suite and configuration, performs Adapter capability preflight, prepares compatible Repository caches, and creates an immutable Run plan.
2. `run-agent` materializes each planned Target and writes a terminal Submission and bounded trace artifacts. It does not run the evaluator or Judge.
3. `evaluate` verifies terminal Submissions and their Target replay, evaluates Intent, Review/Findings, and Evidence, optionally invokes a Judge, and writes a versioned evaluation namespace and report.
4. Running `evaluate` with a different evaluation revision or evaluator/Judge execution identity performs a re-evaluation against the same Submission and Target materialization. The previous namespace remains unchanged.
5. `inspect` reads one committed evaluation through receipt-bound artifact APIs and emits a redacted JSON or Markdown projection. It invokes neither Agent nor Judge.
6. `compare`, `calibrate`, and `gate` replay completed evaluations through the same source-bound loader and publish create-only artifacts in a separate Analysis Store. They invoke neither Agent nor Judge and never acquire a repository or modify `.eval-runs`.

The following is command synopsis, not a copy-and-run script: uppercase words are values returned by an earlier command or selected by the operator. Every flag shown below exists in the current CLI.

```text
review-agent-eval prepare --suite-root SUITE_ROOT --manifest MANIFEST --agent-adapter {current,subprocess} --json
review-agent-eval run-agent RUN_ID --suite-root SUITE_ROOT --manifest MANIFEST --json
review-agent-eval evaluate RUN_ID --suite-root SUITE_ROOT --manifest MANIFEST --revision REVISION --judge-provider {none,fake,openai-compatible} --json
review-agent-eval evaluate RUN_ID --suite-root SUITE_ROOT --manifest MANIFEST --revision NEW_REVISION --evaluator-execution-config EXECUTION_CONFIG --json
review-agent-eval inspect RUN_ID --suite-root SUITE_ROOT --task-id TASK_ID --trial-id TRIAL_ID --evaluation-id EVALUATION_ID --format {json,markdown}
review-agent-eval compare --suite-root SUITE_ROOT --baseline-run-id RUN_ID --baseline-evaluation-id EVALUATION_ID --candidate-run-id RUN_ID --candidate-evaluation-id EVALUATION_ID --policy COMPARISON_POLICY --analysis-root ANALYSIS_ROOT --json
review-agent-eval calibrate export --suite-root SUITE_ROOT --run-id RUN_ID --evaluation-id EVALUATION_ID --profile PROFILE --selection-policy POLICY --output-root CONTROLLED_EXTERNAL_ROOT --analysis-root ANALYSIS_ROOT --json
review-agent-eval calibrate import-labels --suite-root SUITE_ROOT --run-id RUN_ID --evaluation-id EVALUATION_ID --profile PROFILE --selection-policy POLICY --package-id PACKAGE_ARTIFACT_ID --labels LABELS_JSON --analysis-root ANALYSIS_ROOT --json
review-agent-eval calibrate score --suite-root SUITE_ROOT --run-id RUN_ID --evaluation-id EVALUATION_ID --profile PROFILE --selection-policy POLICY --package-id PACKAGE_ARTIFACT_ID --label-set-id LABEL_SET_ARTIFACT_ID --analysis-root ANALYSIS_ROOT --json
review-agent-eval gate prepare --suite-root SUITE_ROOT --baseline-run-id RUN_ID --baseline-evaluation-id EVALUATION_ID --candidate-run-id RUN_ID --policy GATE_POLICY --analysis-root ANALYSIS_ROOT --json
review-agent-eval gate evaluate --suite-root SUITE_ROOT --baseline-run-id RUN_ID --baseline-evaluation-id EVALUATION_ID --candidate-run-id RUN_ID --candidate-evaluation-id EVALUATION_ID --comparison-id COMPARISON_ARTIFACT_ID --comparison-policy COMPARISON_POLICY --gate-policy-id GATE_POLICY_ARTIFACT_ID --analysis-root ANALYSIS_ROOT --json
```

For complete and current arguments, including subprocess command construction, resource budgets, roots, provider settings, and dry runs, use:

```console
review-agent-eval prepare --help
review-agent-eval run-agent --help
review-agent-eval evaluate --help
review-agent-eval inspect --help
review-agent-eval compare --help
review-agent-eval calibrate --help
review-agent-eval gate --help
```

## Analysis lifecycle and store

The default Analysis Store is `.eval-analyses`; use `--analysis-root` to place it in another controlled directory. It is separate from `.eval-runs` and uses canonical JSON, content digests, receipt-last commits, and create-only namespaces for comparisons, calibration package manifests, human label sets, calibration results, frozen Gate policies, and Gate results. Loaders rehydrate the original Run evaluation through `EvaluationOrchestrator.load_run_evaluation`, the immutable Run configuration and Case Snapshot, and the same verified Suite before an analysis artifact can be published. A summary JSON file or Markdown report is never accepted as an analysis trust root.

Repeated-Trial statistics preserve the existing metric numerators and denominators. They expose per-Trial projections, dispersion, confidence intervals, Agent failure coverage, Judge graded/failed/unknown/ungraded coverage, and authority exclusions. Pairing is exact on `(task_id, case_version, canonical_case_digest, trial_index)`. Incompatible Case, Target, evaluator, Judge, rubric, truth, metric-authority, isolation, or Trial-count bindings produce a published `not_comparable` result; they are not coerced into a delta. Analysis adds no Overall Score, Case Pass, `pass@1`, or `pass^k`.

The release order is deliberately preregistered:

```text
baseline prepare → baseline run-agent/evaluate
                 → candidate prepare
                 → gate prepare (frozen Policy; no candidate results yet)
                 → candidate run-agent/evaluate
                 → compare
                 → gate evaluate
```

`gate prepare` refuses a candidate with started or completed Trial work. The frozen Policy binds the baseline Evaluation, candidate immutable Run plan, Case Snapshot, Trial count, Comparison Policy, explicit metric thresholds, eligibility, and any expected Calibration Result digests. `gate evaluate` reloads the frozen Policy and verified Comparison and Calibration Results before publishing a decision. A valid `block` or `ineligible` is a normal business result and exits successfully by default. `--ci` maps either decision to stable policy exit 20 without changing or rewriting the Gate artifact; `promote` remains exit 0.

## External calibration boundary

Calibration follows `verified Judge Evaluation → blind Calibration Package → independent labels → Calibration Result`. The external package contains only the minimum blinded request material and selection bindings needed by a reviewer. Judge identity, model configuration, recorded Judge decisions, expected winner, private truth, and hidden Judge output are excluded. The Analysis Store keeps only the package manifest and receipt; the complete package is written only beneath the operator's explicit `--output-root`, outside the Run Store, Analysis Store, and Git repository. Traversal, absolute child escape, unsafe links/reparse points, hardlinks, portable filename aliases, and Windows 8.3 aliases fail closed where the platform can detect them.

Imported labels are checked against the package, item, selection, source, reviewer, attestation, dispute, and adjudication bindings before publication. The four Judge profiles—intent equivalence, finding equivalence, novel factuality, and evidence support—are scored independently. `pending_human_labels`, `insufficient_coverage`, `failed_thresholds`, and `gate_eligible` remain distinct. Fixture, fake, or synthetic provenance can exercise the protocol but cannot claim real-human `gate_eligible`; credentials and free-form private notes do not belong in label files or ordinary CLI/report envelopes.

The truth-to-metric path remains unchanged: private evaluator truth is matched to the immutable Submission and Evidence, producing typed metric contributions; statistics organize those contributions; comparison computes paired changes; calibration qualifies semantic Judge outputs; and Gate applies only preregistered conditions. Analysis never feeds private truth back to the Agent and never runs a product Runtime, Session, Memory, Risk system, Agent, Judge, acquisition client, or network provider.

### Capability behavior

`prepare` defaults to `--capability-policy strict`. Any Case whose Target or required wire behavior is unsupported rejects Run creation. `--capability-policy filter` removes incompatible Cases, records the preflight result, and still refuses to create an empty Run. Capability declarations are frozen into the Run and checked again before Trial work; changing the Adapter identity on resume is an incompatibility, not an Agent quality result.

The current-Agent adapter declares Repository Target support. The subprocess v2 protocol can declare Repository and/or Frozen Context support, but the CLI-generated default subprocess Agent snapshot is Repository-only. To run a Frozen Context Suite, pass an explicit `--agent-config` whose canonical capabilities include Frozen Context and whose executable implements the declared v2 input, Submission, Evidence, clarification, and trace contracts. Capability, identity, or protocol violations fail before scoring and are not counted as Agent quality.

## Agent and Judge separation

The Agent identity and provider configuration are fixed during `prepare` and used only by `run-agent`. Judge configuration is supplied to `evaluate` and is part of the evaluator execution identity. Agent and Judge have separate provider/model/base-URL/API-key settings; the default environment variables are `REVIEW_AGENT_API_KEY` for the Agent and `REVIEW_AGENT_EVAL_API_KEY` for the Judge.

`--judge-provider fake` proves that request generation, receipts, hydration, namespaces, and reporting obey the protocol. It does not prove semantic grading accuracy. A real model-backed Judge still requires calibration against independent human decisions before its semantic metrics can support release decisions. `--judge-provider none` leaves work that requires semantic judgment ungraded instead of guessing.

## Authority-aware metrics

Metric authority is attached to each expected finding. It controls whether severity- and location-sensitive metrics are eligible; it does not change issue matching itself.

| Truth source | Severity metrics | Location metrics |
| --- | --- | --- |
| Core expert truth | eligible | eligible |
| AACR-Bench | excluded / `not_scorable` | eligible |
| SWE-PRBench | excluded / `not_scorable` | excluded / `not_scorable` |

Severity-sensitive metrics include severity-weighted recall and critical/high miss count. Location-sensitive metrics include line precision and line recall. When authority is absent, the harness stores null numerator/denominator values, reports exclusion and `not_scorable` coverage, and omits that truth from the affected ratio. It never converts missing authority into a zero. Issue, Evidence, failure, Judge, and coverage metrics remain separately visible when their own inputs are available.

## Compatibility partitions

Scores aggregate only when their compatibility profile is identical. Protocol and wire contract, Target kind, metric-authority profile, and Adapter isolation profile are quality boundaries. Repository and Frozen scores, Core/AACR/SWE authority profiles, or different isolation profiles remain separate report partitions and cannot be rolled into one quality number.

Other source-bound dimensions, such as evaluator revision and truth-completeness policy, are also preserved in the score and report artifacts. Compare the metrics within a compatible partition; do not average partition summaries by hand and call the result an Overall Score.

## Artifacts, create-only behavior, and recovery

Unless overridden, the CLI uses `.eval-runs`, `.eval-data`, and `.eval-workspaces`. Important persisted artifacts include:

- immutable Run configuration, Case snapshot, Run manifest, and Trial manifests;
- canonical EvalInput and per-attempt Target materialization manifests;
- prepare/Agent/evaluator stage receipts and terminal Submission artifacts;
- bounded trace-capture metadata and optional local trace artifacts;
- evaluator execution configuration, Intent and Review results, Evidence results, Judge input/output, Trial score, run summary, and Markdown reports.

Plans, receipts, Submissions, public prepared Suites, and evaluation namespaces are create-only. `--overwrite` is parsed for a stable interface but rejected by `prepare`, `run-agent`, and `evaluate`. Use a new `--run-instance-key` for a different Run, and a new evaluation identity for a re-evaluation.

`prepare --resume` reuses an existing Run only when its canonical plan is identical. `run-agent` resumes by default, skips terminal Trials, and may retry only incomplete work; `--no-resume` disables that recovery behavior. `evaluate` also resumes by default: an identical committed evaluation is returned without constructing a Judge or replaying the Agent. `--no-resume` rejects an existing namespace. A changed revision, evaluator execution configuration, or Judge identity creates a new re-evaluation namespace while preserving the old one and reusing the same Submission and Target materialization.

## Privacy and redaction boundary

The Run root is a local evidence store, not a public report directory. Suite inputs, canonical EvalInputs, Submissions, Evidence excerpts, and evaluation artifacts can contain source code or review content and must be protected accordingly.

`inspect` exposes stable source bindings, statuses, counts, coverage, usage, and digests. It does not emit raw repository context, raw Judge context, or a TraceRef value; trace references and unsafe paths are redacted. Generated reports use redacted artifact projections, and CLI failures avoid echoing provider exceptions, credentials, and arbitrary absolute paths.

Redaction at the inspection/report boundary does not erase sensitive bytes from the underlying local Suite, Submission, trace, or evaluator artifacts. Control access to `--suite-root`, `--runs-root`, `--data-root`, and `--workspace-root`, and review provider data-handling terms before using a network-backed Agent or Judge.

## Release eligibility and external gates

Core Regression Cases with authoritative reviewed truth, and future Private Held-out Cases with equivalent trust bindings, may support `release_blocking` policies. Public AACR-Bench/SWE-PRBench and unpromoted Synthetic Cases are `diagnostic_only`: they can expose statistics, comparisons, calibration problems, and Gate checks, but cannot independently block a formal release. Missing severity or location authority remains `not_scorable`, never zero or pass. A diagnostic policy or an unavailable required input produces `ineligible`; a fully eligible policy with a failed required threshold produces `block`.

The following release gates remain external and must not be replaced by synthetic or fake-provider evidence:

- an independent human Reviewer B completes blind review of the Core Cases;
- each Regression Case has at least three Trials with the configured real model;
- private held-out data is evaluated without leaking it into authoring or calibration.

The local compare/calibrate/gate lifecycle enforces artifact and policy mechanics, but it does not satisfy those external conditions. Formal promotion still requires the configured real model, independent human Reviewer B, calibrated real-human semantic labels, and Private Held-out evidence.

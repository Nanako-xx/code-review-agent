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

The following is command synopsis, not a copy-and-run script: uppercase words are values returned by an earlier command or selected by the operator. Every flag shown below exists in the current CLI.

```text
review-agent-eval prepare --suite-root SUITE_ROOT --manifest MANIFEST --agent-adapter {current,subprocess} --json
review-agent-eval run-agent RUN_ID --suite-root SUITE_ROOT --manifest MANIFEST --json
review-agent-eval evaluate RUN_ID --suite-root SUITE_ROOT --manifest MANIFEST --revision REVISION --judge-provider {none,fake,openai-compatible} --json
review-agent-eval evaluate RUN_ID --suite-root SUITE_ROOT --manifest MANIFEST --revision NEW_REVISION --evaluator-execution-config EXECUTION_CONFIG --json
review-agent-eval inspect RUN_ID --suite-root SUITE_ROOT --task-id TASK_ID --trial-id TRIAL_ID --evaluation-id EVALUATION_ID --format {json,markdown}
```

For complete and current arguments, including subprocess command construction, resource budgets, roots, provider settings, and dry runs, use:

```console
review-agent-eval prepare --help
review-agent-eval run-agent --help
review-agent-eval evaluate --help
review-agent-eval inspect --help
```

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

## External gates and next work

The following release gates remain external and must not be replaced by synthetic or fake-provider evidence:

- an independent human Reviewer B completes blind review of the Core Cases;
- each Regression Case has at least three Trials with the configured real model;
- private held-out data is evaluated without leaking it into authoring or calibration.

Paired comparison, Judge calibration, and automated regression gates are **not currently available**. They are the next planned work in Task 15. The CLI can plan multiple Trials with `--trial-count`, but it does not yet provide Task 15's paired compare, calibration workflow, or regression promotion gate.

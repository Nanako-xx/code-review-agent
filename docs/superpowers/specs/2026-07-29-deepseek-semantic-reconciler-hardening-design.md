# DeepSeek Semantic Reconciler Hardening Design

**Date:** 2026-07-29
**Status:** Approved for written-spec review
**Scope:** `core-py-001` real-model smoke and the model-backed Semantic Reconciler boundary

## 1. Problem statement

The first complete `core-py-001` DeepSeek run proved that all four Reviewers can return parseable, contract-valid `partial` results. The run still used deterministic Semantic Reconciler fallback for two independent reasons found by bounded replay:

1. The configured Semantic Reconciler stage allowed only 60 elapsed seconds. DeepSeek required approximately 65-84 seconds per response, so the original request was terminated at 60 seconds before a second attempt could start.
2. A 4096-token response was truncated. At 8192 tokens the response completed, but it invented field names such as `groups` because the System Prompt named `semantic_reconciliation_proposal_v1` without defining its exact wire shape.

The Runtime correctly failed closed in both cases. This design makes the provider-facing contract sufficiently explicit while preserving Runtime authority.

## 2. Goals and success criteria

The change must:

- give the model the exact Semantic Reconciler response fields, nested fields, enums, and accounting invariants;
- request JSON object mode only for this no-tool model turn;
- retain strict Runtime parsing, candidate accounting, Observation-reference validation, conservative retry, and deterministic fallback;
- record the JSON-mode request in the immutable invocation envelope;
- configure the DeepSeek baseline with an 8192-token Semantic Reconciler output limit and a 240-second stage elapsed limit;
- leave shared `ModelStageConfig` defaults unchanged, because baseline budgets are part of the evaluated Agent identity rather than universal provider defaults.

The live smoke succeeds when every Reviewer result remains structurally valid and Semantic Reconciliation no longer falls back because of timeout, output truncation, invalid JSON, or an undocumented response shape. An overall `blocked` result remains legitimate when the only blocker is an insufficient Intent Packet.

## 3. Provider-facing response contract

`SEMANTIC_RECONCILER_SYSTEM_PROMPT` will define one top-level JSON object with exactly these fields:

- `canonical_groups`
  - array of objects with exactly `member_ids`, `representative_id`, `canonical_claim`, `rationale`, `supporting_refs`, and `proposed_confidence`;
  - `proposed_confidence` is one of `high`, `medium`, or `low`.
- `rejections`
  - array of objects with exactly `candidate_id`, `reason`, `rationale`, and `decision_refs`;
  - `reason` is one of `unsupported_claim`, `contradicted_by_test`, or `outside_review_scope`.
- `disagreements`
  - array of objects with exactly `disagreement_id`, `candidate_ids`, `status`, `issue`, `resolution`, and `decision_refs`;
  - `status` is one of `resolved`, `needs_investigation`, or `unresolved`.
- `supplemental_requests`
  - array of objects with exactly `disagreement_id`, `question`, `required_evidence`, `preferred_perspective`, `related_candidate_ids`, and `reason_refs`.
- `uncertainties`
  - array of strings.
- `summary`
  - non-empty string.

The prompt will also state the existing Runtime invariants in operational terms:

- every candidate in the current batch is disposed exactly once by either one canonical group or one rejection;
- IDs and Observation references must come from the supplied allowlists;
- a representative belongs to its group;
- supporting references come from group members;
- every `needs_investigation` disagreement has exactly one matching supplemental request, and other disagreements have none;
- the model must not wrap the JSON in Markdown or add fields.

These instructions improve model compliance but do not replace parser or compiler checks.

## 4. JSON mode and adapter boundary

`run_semantic_reconciler_batch` will include `response_format: "json_object"` in both the persisted envelope parameters and each `ModelTurnRequest`. The request has no tools, so it satisfies the existing OpenAI-compatible adapter rule that JSON mode cannot be combined with tools.

No provider-specific API handling enters Reconciler business logic. The unified model adapter remains responsible for translating the project parameter into the OpenAI-compatible payload. Fake adapters may ignore the transport hint while still returning the same project protocol.

If a provider does not support the configured JSON mode or returns an invalid proposal, the existing bounded retry and deterministic fallback behavior remains authoritative.

## 5. Baseline budgets

The DeepSeek baseline Agent snapshot will include these current-Agent review arguments:

```text
--semantic-reconciler-max-output-tokens=8192
--semantic-reconciler-max-elapsed-seconds=240
```

`max_provider_attempts` remains 2. The OpenAI-compatible adapter retains its 180-second per-request ceiling, while the 240-second stage budget leaves bounded time for a second attempt after an early parse or provider failure.

The global `ModelStageConfig` defaults remain 4096 tokens and 60 seconds. Changing them would silently alter Risk Assessor, Portfolio Planner, Memory Curator, existing Sessions, and non-DeepSeek evaluations without evidence that those stages need larger budgets.

## 6. Error handling and privacy

- Parser failures continue to append a bounded rejection message for the next attempt.
- Provider errors, budget exhaustion, and invalid proposals remain explicit in local decision artifacts.
- Deterministic fallback preserves all Runtime-supported Findings and forces manual review where required.
- API keys remain environment-variable references and never enter artifacts or command output.
- Diagnostic and user-facing summaries expose status, counts, timings, and sanitized parser errors, not hidden reasoning or complete private model responses.

## 7. Verification

Implementation follows a minimal red-green cycle:

1. Add a failing test proving the Reconciler request contains `response_format=json_object` and that its System Prompt exposes the exact field contract and accounting rules.
2. Make the smallest production change that satisfies the test.
3. Run focused Reconciler and adapter tests, then the complete local regression suite.
4. Replay the existing `core-py-001` Reconciler packet with 8192 tokens and a 240-second stage budget.
5. Run one complete `core-py-001` smoke with the same frozen Agent arguments and inspect Reviewer, Reconciler, Completion, and Final Risk artifacts.

Formal 10-case x 3-trial baseline execution is outside this hardening change and starts only after the single-case smoke is clean.

## 8. Non-goals

- adding a generic second-model finalization layer for every one-turn model stage;
- weakening exact-field parsing or Runtime candidate accounting;
- changing Intent, Risk, Finding severity, Evidence, Completion, or supplemental-investigation policy;
- changing shared model-stage defaults to match one provider;
- starting release-gate evaluation or the formal repeated-trial baseline.

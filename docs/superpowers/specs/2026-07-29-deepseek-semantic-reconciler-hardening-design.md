# DeepSeek Semantic Reconciler Hardening Design

**Date:** 2026-07-29
**Status:** Approved for written-spec review
**Scope:** `core-py-001` real-model smoke and the model-backed Semantic Reconciler boundary

## 1. Problem statement

The first complete `core-py-001` DeepSeek run proved that all Runtime-scheduled Reviewers can return parseable, contract-valid `partial` results. The run still used deterministic Semantic Reconciler fallback for two independent reasons found by bounded replay:

1. The configured Semantic Reconciler stage allowed only 60 elapsed seconds. DeepSeek required approximately 65-84 seconds per response, so the original request was terminated at 60 seconds before a second attempt could start.
2. A 4096-token response was truncated. At 8192 tokens the response completed, but it invented field names such as `groups` because the System Prompt named `semantic_reconciliation_proposal_v1` without defining its exact wire shape.

The Runtime correctly failed closed in both cases. This design makes the provider-facing contract sufficiently explicit while preserving Runtime authority.

## 2. Goals and success criteria

The change must:

- give the model the exact Semantic Reconciler response fields, nested fields, enums, and accounting invariants;
- request JSON object mode only for this no-tool model turn;
- retain strict Runtime parsing, candidate accounting, Observation-reference validation, conservative retry, and deterministic fallback;
- record the JSON-mode request in the immutable invocation envelope;
- configure the DeepSeek baseline with an 8192-token Semantic Reconciler output limit, exactly two provider attempts, and a 240-second stage elapsed limit;
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

The direct-review DeepSeek baseline will include all four frozen arguments explicitly:

```text
--memory-mode=off
--semantic-reconciler-max-output-tokens=8192
--semantic-reconciler-max-provider-attempts=2
--semantic-reconciler-max-elapsed-seconds=240
```

Evaluation `prepare` will encode the same evaluated Agent identity explicitly in the
current-Agent snapshot:

```text
--memory-mode=off
--agent-argument=--semantic-reconciler-max-output-tokens=8192
--agent-argument=--semantic-reconciler-max-provider-attempts=2
--agent-argument=--semantic-reconciler-max-elapsed-seconds=240
```

Memory off prevents prior local project Memory from changing diagnostic or baseline
inputs, and prevents smoke or evaluation runs from proposing or writing Memory.
Reviewer count is not frozen: accepted model Risk feeds the local Runtime `ReviewProfile`,
so actual reviewer depth remains risk-dependent. Do not change global `ModelStageConfig`
defaults.

`--semantic-reconciler-max-provider-attempts=2` is frozen evaluated-Agent identity
configuration, not a value inherited from the current shared default. The
OpenAI-compatible adapter retains its 180-second per-request ceiling, while the 240-second
stage budget leaves bounded time for a second attempt after an early parse or provider
failure.

The global `ModelStageConfig` defaults remain unchanged at 4096 tokens, 2 provider
attempts, and 60 seconds. Changing them would silently alter Risk Assessor, Portfolio
Planner, Memory Curator, existing Sessions, and non-DeepSeek evaluations without evidence
that those stages need different budgets.

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
   The local direct malformed-finalization regression must prove this exact terminal path:
   the initial Reviewer `FINAL` is non-JSON; the sole JSON finalization is also non-JSON;
   total provider requests equal 2; the terminal Reviewer status is `failed`; the
   finalization parse diagnostic is preserved; and no third provider attempt occurs.
4. Replay the existing `core-py-001` Reconciler packet with 8192 tokens and a 240-second stage budget.
5. Run one complete `core-py-001` smoke with the same frozen Agent arguments and inspect
   every Runtime-scheduled Reviewer, Reconciler, Completion, and Final Risk artifact. The
   smoke validates the normal Reviewer protocol and need not deliberately trigger the
   malformed-finalization path proved by the local direct regression.
   Exactly one bounded, audited JSON finalization is acceptable when the terminal
   structured Reviewer result is `completed` or `partial` and provider-attempt, token,
   and elapsed budgets are respected. Unrecovered parser, provider, or budget errors
   remain failures.

Formal 10-case x 3-trial baseline execution is outside this hardening change and starts only after the single-case smoke is clean.

## 8. Non-goals

- adding a generic second-model finalization layer for every one-turn model stage;
- weakening exact-field parsing or Runtime candidate accounting;
- changing Intent, Risk, Finding severity, Evidence, Completion, or supplemental-investigation policy;
- changing shared model-stage defaults to match one provider;
- starting release-gate evaluation or the formal repeated-trial baseline.

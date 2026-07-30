# Reviewer 8192 Output Budget Design

**Date:** 2026-07-30

**Status:** Approved by user

**Scope:** Runtime-owned per-call output budgets for Reviewer assignments

## Problem

The current DeepSeek `core-py-001` smoke scheduled two medium-risk Reviewers. One
Reviewer completed its investigation but its sole JSON-finalization response stopped
with provider `finish_reason=length` at the configured 4096-token ceiling. The resulting
unterminated JSON was correctly rejected by Runtime and the Reviewer ended as `failed`.
No unknown or unauthorized Observation ID caused this failure.

## Decision

Raise the Reviewer per-call output ceiling from 4096 to 8192 tokens everywhere:

- `DEFAULT_REVIEWER_MAX_OUTPUT_TOKENS` becomes `8192`;
- low-risk and medium-risk `ReviewProfile` values become `8192`;
- high-risk and critical-risk profiles remain `8192`;
- legacy Assignment hydration continues to use the default constant and therefore also
  resolves missing `max_output_tokens` to `8192`.

This is a budget change only. Runtime JSON parsing, Review Contract validation,
Observation authorization, total-token limits, elapsed-time limits, tool limits, and
provider-attempt limits remain unchanged.

## Trade-off

Low- and medium-risk Reviewer calls may consume more output tokens when the model needs
them. Existing total-token and time budgets still bound an entire Reviewer execution.
The user selected this direct global increase instead of adding a separate finalization
budget or provider-specific thinking-mode control.

## Verification

The change is acceptable when:

- the default Reviewer output budget is exactly `8192`;
- every risk profile exposes an `8192` per-call Reviewer output budget;
- legacy Assignment hydration resolves an omitted value to `8192`;
- Product and Eval test suites pass;
- a fresh current-HEAD DeepSeek smoke has no `failed` Reviewer and preserves strict
  Observation authorization;
- Semantic Reconciliation remains model-accepted, Session phases have no errors, Memory
  remains off, and Completion may be blocked only by the insufficient Intent Packet.

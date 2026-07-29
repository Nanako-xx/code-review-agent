# Observation-Aware Tool Result Envelope Design

**Date:** 2026-07-30
**Status:** Approved
**Scope:** The project-level tool-result protocol used by Reviewer and Intent Inference model turns

## 1. Problem statement

The current Runtime creates a trusted `ModelToolResult` containing `tool_name`,
`observation_ids`, `is_error`, and the untrusted tool `content`. The OpenAI-compatible
adapter currently sends only `content` in the provider `role=tool` message. The model
therefore sees the repository excerpt but not the Runtime-issued Observation IDs that it
must cite in `evidence_refs`.

A real `core-py-001` DeepSeek smoke exposed the consequence. One Reviewer attempted to
return evidence, invented an Observation ID that was not among its 15 authorized tool
Observations, and was correctly rejected by Runtime. Relaxing the authorization check
would convert fabricated evidence into apparently valid evidence and would invalidate the
project's evidence-driven Review Contract.

The defect is missing model input, not excessive Runtime validation.

## 2. Decision

Keep the existing hard rule:

```text
result evidence_refs must be a subset of the Observation IDs authorized for that Reviewer
```

Send each tool result to the model as one versioned project envelope. The provider's
`tool_call_id` remains on the outer `role=tool` message and is not duplicated inside the
envelope.

The exact envelope is:

```json
{
  "schema_version": "review_agent_tool_result_v1",
  "tool_name": "read_range",
  "observation_ids": ["O-example"],
  "is_error": false,
  "content": "untrusted repository or tool output"
}
```

Every field is always present. `observation_ids` may be empty, especially for an errored
tool call. No new database, persistence layer, or evidence-resolution subsystem is added.

## 3. Ownership and trust boundary

Runtime owns and supplies:

- `schema_version`;
- `tool_name`;
- `observation_ids`;
- `is_error`.

The `content` field remains untrusted data. JSON encoding prevents repository text that
looks like metadata from changing the actual metadata fields, but it does not make the
repository text an instruction. Existing System Prompt rules continue to require the model
to treat tool content as untrusted.

The model must:

- cite Observation IDs exactly as shown in `observation_ids`;
- never invent, transform, shorten, or infer an Observation ID;
- cite an ID only when the associated `content` supports the claim;
- treat an empty `observation_ids` array as providing no citable Evidence.

Runtime remains authoritative and continues rejecting unknown or unauthorized references.

## 4. Serialization contract

The envelope is canonical project protocol, not an OpenAI- or DeepSeek-specific object.
It is serialized as one JSON string with:

```python
json.dumps(
    envelope,
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
)
```

The resulting string must be UTF-8 encodable before it enters a model request. The
existing provider message remains:

```json
{
  "role": "tool",
  "tool_call_id": "call-1",
  "content": "{...canonical envelope JSON...}"
}
```

The envelope adds only bounded metadata. It does not duplicate the repository excerpt or
increase the Tool Gateway's content limit.

## 5. Runtime and Adapter data flow

1. Tool Gateway executes an authorized tool and records zero or more Observations.
2. Runtime creates the existing typed `ModelToolResult`.
3. The project protocol serializer converts that result into
   `review_agent_tool_result_v1`.
4. Reviewer or Intent Inference appends the assistant tool-call message followed
   immediately by the envelope-bearing tool message to its ordered transcript.
5. The model adapter maps that transcript to the provider payload without adding hidden
   per-session history or moving messages.
6. Runtime retains typed `ModelToolResult` values as local audit metadata. When both the
   transcript and typed metadata are present, the adapter compares the transcript's
   canonical envelope string with the envelope derived from the typed value. A mismatch
   fails before network transport; the adapter never inserts a duplicate tool result.
7. Completion validation checks model-produced Observation references against the
   authorized Runtime set exactly as it does today.

Legacy stateless requests that provide an assistant tool call plus a separated typed
`tool_results` list use the same serializer when the adapter inserts the result. They do
not use adapter instance state.

## 6. Prompt contract

The Reviewer System Prompt will state that each tool message contains a
`review_agent_tool_result_v1` object and will explain the four Runtime metadata fields and
the untrusted `content` field. The existing requirement that Findings cite Observation IDs
remains unchanged.

The Intent Inference prompt will receive the same short protocol instruction because it
uses the same ordered tool loop and may cite repository Observations while inferring
intent. Provider-specific prompt branches are not added.

## 7. Error handling

- A non-UTF-8-safe or non-JSON-safe envelope fails locally before provider transport.
- Empty, duplicate, orphaned, non-adjacent, or partially paired tool-call IDs continue to
  fail locally under the ordered-transcript checks.
- Typed audit metadata that does not serialize to the exact transcript envelope fails
  locally with a fixed diagnostic that does not include tool content.
- A tool error is sent with `is_error=true`; its `observation_ids` array contains only IDs
  actually issued by Runtime and may be empty.
- Unknown model-produced Observation IDs remain a Runtime completion deficiency. They are
  never silently dropped, rewritten, or auto-bound to a different Observation.

## 8. Alternatives rejected

### Plain-text Observation ID prefix

Prepending `Observation IDs: ...` is shorter but mixes trusted metadata with untrusted
text, is easier for repository content to imitate, and creates an informal parsing
contract. It is not selected.

### Relax authorization validation

Allowing unknown IDs would make fabricated Evidence indistinguishable from observed
Evidence and would corrupt both review results and evaluation metrics. It is not selected.

### Runtime auto-attachment of Evidence

Automatically assigning every tool Observation to a Finding would remove the model's
responsibility to decide which evidence supports which claim and would create false
support. It is not selected.

## 9. Verification and acceptance

Required automated coverage:

- exact canonical envelope fields and schema version;
- multiple and empty `observation_ids` arrays;
- tool content containing JSON-like or instruction-like text cannot alter metadata;
- envelope output is JSON-safe and UTF-8-safe;
- captured OpenAI-compatible payload preserves ordered assistant/tool messages and exposes
  the exact authorized Observation IDs;
- complete transcript plus typed audit metadata produces one provider tool message, while
  any mismatch fails before transport;
- Reviewer and Intent Inference tool-loop regressions remain green;
- a Reviewer can cite an ID supplied by its tool envelope and pass completion validation;
- a fabricated or cross-Reviewer ID is still rejected;
- malformed-finalization, provider-attempt, elapsed, and turn budgets do not regress.

Final verification uses the current Product and Eval partitions plus one authorized
`core-py-001` DeepSeek smoke. The smoke is acceptable when:

- every Runtime-scheduled Reviewer result is `completed` or `partial`, not `failed`;
- no Reviewer failure is caused by an unknown or unauthorized Observation reference;
- Semantic Reconciler and its model status are `accepted`;
- all Session phases complete without errors;
- an overall `blocked` result remains allowed when its blocker is the insufficient Intent
  Packet.

This smoke is a protocol/integration gate, not a score for review quality and not the
formal 10 Case x 3 Trial baseline.

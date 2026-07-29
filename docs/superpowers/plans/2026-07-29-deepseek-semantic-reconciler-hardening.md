# DeepSeek Semantic Reconciler Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the model-backed Semantic Reconciler reliably return Runtime-valid structured output for the `core-py-001` DeepSeek smoke without weakening strict reconciliation or changing shared model-stage defaults.

**Architecture:** Keep the Semantic Reconciler and unified model adapter boundary unchanged. Strengthen the Reconciler System Prompt with the exact project response protocol, send the existing provider-neutral `response_format=json_object` hint on its no-tool turn, and freeze DeepSeek-specific 8192-token/240-second limits in the evaluated Agent arguments rather than global defaults. Runtime parsing, candidate accounting, bounded retry, deterministic fallback, and Completion remain authoritative.

**Tech Stack:** Python 3.12, dataclasses, pytest, the existing `ModelAdapter` protocol, OpenAI-compatible HTTP transport, PowerShell, Git.

---

## Working-state constraints

- Work only in `D:\Agent\code review agent\.worktrees\real-model-baseline` on branch `codex/real-model-baseline`.
- The eight existing modified Reviewer/adapter files are recovered work from the same real-model investigation. Verify and commit them before changing Reconciler files.
- Do not modify the user's main working tree.
- Do not stage `.pytest-tmp/`, `.test-tmp/`, `.eval-data/`, `__pycache__/`, raw provider responses, or API credentials.
- Never print `REVIEW_AGENT_API_KEY`, provider hidden reasoning, or complete private Case/model text.

### Task 1: Preserve and commit the recovered Reviewer protocol hardening

**Files:**
- Modify: `src/review_agent/agent_loop.py`
- Modify: `src/review_agent/context.py`
- Modify: `src/review_agent/model_adapter.py`
- Modify: `src/review_agent/model_adapter_factory.py`
- Test: `tests/test_agent_loop.py`
- Test: `tests/test_context.py`
- Test: `tests/test_model_adapter.py`
- Test: `tests/test_model_adapter_factory.py`

- [ ] **Step 1: Re-run the focused recovered-work regression**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_agent_loop.py `
  tests/test_context.py `
  tests/test_model_adapter.py `
  tests/test_model_adapter_factory.py `
  -q -p no:cacheprovider `
  --basetemp 'D:\Agent\code review agent\.worktrees\real-model-baseline\.test-tmp\reviewer-protocol'
```

Expected: all tests pass. A Windows temporary-directory cleanup warning may be reported separately, but no test may fail.

- [ ] **Step 2: Verify the recovered diff is limited to the tested boundary**

Run:

```powershell
git diff --check -- `
  src/review_agent/agent_loop.py `
  src/review_agent/context.py `
  src/review_agent/model_adapter.py `
  src/review_agent/model_adapter_factory.py `
  tests/test_agent_loop.py `
  tests/test_context.py `
  tests/test_model_adapter.py `
  tests/test_model_adapter_factory.py
git diff --stat -- `
  src/review_agent/agent_loop.py `
  src/review_agent/context.py `
  src/review_agent/model_adapter.py `
  src/review_agent/model_adapter_factory.py `
  tests/test_agent_loop.py `
  tests/test_context.py `
  tests/test_model_adapter.py `
  tests/test_model_adapter_factory.py
```

Expected: `git diff --check` is silent and the stat lists exactly the eight files above.

- [ ] **Step 3: Commit only the recovered Reviewer hardening**

Run:

```powershell
git add -- `
  src/review_agent/agent_loop.py `
  src/review_agent/context.py `
  src/review_agent/model_adapter.py `
  src/review_agent/model_adapter_factory.py `
  tests/test_agent_loop.py `
  tests/test_context.py `
  tests/test_model_adapter.py `
  tests/test_model_adapter_factory.py
git commit -m "fix: harden deepseek reviewer protocol"
```

Expected: one commit containing only these eight files.

### Task 2: Declare the exact Semantic Reconciler response protocol

**Files:**
- Modify: `tests/test_reconciler.py`
- Modify: `src/review_agent/reconciler.py:40-63`

- [ ] **Step 1: Write the failing System Prompt contract test**

Add to `tests/test_reconciler.py` after `test_reconciler_boundary_marks_memory_feedback_sources_and_evidence_as_untrusted`:

```python
def test_reconciler_system_prompt_declares_the_exact_proposal_contract():
    candidate = _candidate("protocol", claim="Candidate must be preserved")
    prepass, observations, _ = _packet([candidate])
    adapter = FakeToolCallingAdapter(
        [
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=json.dumps(_proposal_payload([candidate])),
            )
        ]
    )

    reconcile_semantically(
        prepass,
        observations,
        adapter=adapter,
        max_provider_attempts=1,
    )

    system = " ".join(adapter.requests[0].system.split()).casefold()
    for field_name in (
        "canonical_groups",
        "member_ids",
        "representative_id",
        "canonical_claim",
        "rationale",
        "supporting_refs",
        "proposed_confidence",
        "rejections",
        "candidate_id",
        "reason",
        "decision_refs",
        "disagreements",
        "disagreement_id",
        "candidate_ids",
        "status",
        "issue",
        "resolution",
        "supplemental_requests",
        "question",
        "required_evidence",
        "preferred_perspective",
        "related_candidate_ids",
        "reason_refs",
        "uncertainties",
        "summary",
    ):
        assert field_name in system
    for rule in (
        "disposed exactly once",
        "representative_id must belong to member_ids",
        "do not wrap the json in markdown",
        "needs_investigation",
        "exactly one matching supplemental request",
    ):
        assert rule in system
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_reconciler.py::test_reconciler_system_prompt_declares_the_exact_proposal_contract `
  -q -p no:cacheprovider `
  --basetemp 'D:\Agent\code review agent\.worktrees\real-model-baseline\.test-tmp\reconciler-prompt-red'
```

Expected: FAIL because the current System Prompt names the schema but omits fields such as `canonical_groups`.

- [ ] **Step 3: Append the exact protocol to `SEMANTIC_RECONCILER_SYSTEM_PROMPT`**

Append this text before the closing triple quote in `src/review_agent/reconciler.py`:

```python

Return exactly one JSON object and no Markdown or commentary. Do not wrap the JSON in Markdown
fences. The top-level object has exactly these fields:
- canonical_groups: an array of objects. Each object has exactly member_ids,
  representative_id, canonical_claim, rationale, supporting_refs, and proposed_confidence.
  proposed_confidence is high, medium, or low.
- rejections: an array of objects. Each object has exactly candidate_id, reason, rationale,
  and decision_refs. reason is unsupported_claim, contradicted_by_test, or
  outside_review_scope.
- disagreements: an array of objects. Each object has exactly disagreement_id,
  candidate_ids, status, issue, resolution, and decision_refs. status is resolved,
  needs_investigation, or unresolved.
- supplemental_requests: an array of objects. Each object has exactly disagreement_id,
  question, required_evidence, preferred_perspective, related_candidate_ids, and reason_refs.
- uncertainties: an array of strings.
- summary: a non-empty string.

Dispose every candidate in this batch exactly once: each candidate_id must appear in exactly
one canonical_groups member_ids array or exactly one rejection, never both. representative_id
must belong to member_ids. supporting_refs must come from the grouped candidates. Candidate IDs,
decision_refs, supporting_refs, and reason_refs must come from the packet allowlists. Every
needs_investigation disagreement requires exactly one matching supplemental request; resolved
and unresolved disagreements require none. Do not add fields or invent IDs, Observations,
Findings, or facts.
```

- [ ] **Step 4: Run the prompt test and complete Reconciler unit suite**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_reconciler.py `
  -q -p no:cacheprovider `
  --basetemp 'D:\Agent\code review agent\.worktrees\real-model-baseline\.test-tmp\reconciler-prompt-green'
```

Expected: all `tests/test_reconciler.py` tests pass.

- [ ] **Step 5: Commit the prompt contract**

Run:

```powershell
git add -- src/review_agent/reconciler.py tests/test_reconciler.py
git commit -m "fix: declare semantic reconciler protocol"
```

Expected: one commit containing only the prompt contract and its test.

### Task 3: Request auditable JSON object mode for the no-tool Reconciler turn

**Files:**
- Modify: `tests/test_reconciler.py`
- Modify: `src/review_agent/reconciler.py:870-919`

- [ ] **Step 1: Write the failing JSON-mode propagation test**

Add to `tests/test_reconciler.py`:

```python
def test_reconciler_records_and_requests_json_object_mode():
    candidate = _candidate("json-mode", claim="Candidate must be preserved")
    prepass, observations, _ = _packet([candidate])
    adapter = FakeToolCallingAdapter(
        [
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=json.dumps(_proposal_payload([candidate])),
            )
        ]
    )

    run = reconcile_semantically(
        prepass,
        observations,
        adapter=adapter,
        max_provider_attempts=1,
    )

    request = adapter.requests[0]
    assert request.tools == []
    assert request.parameters["response_format"] == "json_object"
    assert (
        run.batches[0].envelope["parameters"]["response_format"]
        == "json_object"
    )
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_reconciler.py::test_reconciler_records_and_requests_json_object_mode `
  -q -p no:cacheprovider `
  --basetemp 'D:\Agent\code review agent\.worktrees\real-model-baseline\.test-tmp\reconciler-json-red'
```

Expected: FAIL with a missing `response_format` parameter.

- [ ] **Step 3: Add the provider-neutral response hint to the immutable envelope**

In the `parameters` object built by `run_semantic_reconciler_batch`, add exactly:

```python
            "response_format": "json_object",
```

The request already copies `envelope["parameters"]`, so do not duplicate provider translation or add provider-specific branches in `reconciler.py`.

- [ ] **Step 4: Run Reconciler and adapter boundary tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_reconciler.py `
  tests/test_model_adapter.py `
  tests/test_model_adapter_factory.py `
  -q -p no:cacheprovider `
  --basetemp 'D:\Agent\code review agent\.worktrees\real-model-baseline\.test-tmp\reconciler-json-green'
```

Expected: all tests pass, including the adapter rule that JSON mode is accepted only without tools.

- [ ] **Step 5: Commit JSON-mode propagation**

Run:

```powershell
git add -- src/review_agent/reconciler.py tests/test_reconciler.py
git commit -m "fix: request reconciler json responses"
```

Expected: one commit containing only JSON-mode propagation and its test.

### Task 4: Run the complete local regression before using the network

**Files:**
- Verify: `src/review_agent/**/*.py`
- Verify: `src/review_agent_eval/**/*.py`
- Verify: `tests/**/*.py`

- [ ] **Step 1: Run the focused model and orchestration regression**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_agent_loop.py `
  tests/test_context.py `
  tests/test_model_adapter.py `
  tests/test_model_adapter_factory.py `
  tests/test_reconciler.py `
  tests/test_pipeline.py `
  -q -p no:cacheprovider `
  --basetemp 'D:\Agent\code review agent\.worktrees\real-model-baseline\.test-tmp\focused-full'
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete local test suite**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  -q -p no:cacheprovider `
  --basetemp 'D:\Agent\code review agent\.worktrees\real-model-baseline\.test-tmp\all-tests'
```

Expected: all tests pass. If a Windows cleanup warning occurs after pytest has reported all tests passed, record it separately and do not misclassify it as a product failure.

- [ ] **Step 3: Confirm tracked files are clean**

Run:

```powershell
git status --short
git log --oneline -4
```

Expected: no tracked implementation files are modified; only ignored or known temporary directories may remain untracked. The recent log contains the design, Reviewer hardening, prompt contract, and JSON-mode commits.

### Task 5: Replay only the existing `core-py-001` Semantic Reconciler packet

**Files:**
- Read: `.eval-data/deepseek-v4-pro-smoke/repo/.review-agent/runs/review-91fd9034aa94/reconciliation_prepass.json`
- Read: `.eval-data/deepseek-v4-pro-smoke/repo/.review-agent/runs/review-91fd9034aa94/reconciliation_packet.json`
- Read: `.eval-data/deepseek-v4-pro-smoke/repo/.review-agent/runs/review-91fd9034aa94/observations.jsonl`

- [ ] **Step 1: Confirm the credential exists without printing it**

Run:

```powershell
if ([string]::IsNullOrWhiteSpace($env:REVIEW_AGENT_API_KEY)) {
    throw 'REVIEW_AGENT_API_KEY is not configured'
}
'REVIEW_AGENT_API_KEY is configured'
```

Expected: `REVIEW_AGENT_API_KEY is configured`; the value itself is never printed.

- [ ] **Step 2: Run one bounded Reconciler replay with the frozen DeepSeek budget**

Run the following from the worktree. It reads the already-authorized local synthetic Case and prints only sanitized status/count metadata:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
@'
import json
from pathlib import Path

from review_agent.evidence import (
    ConflictHint,
    ContractCoverage,
    FindingCandidate,
    ReconciliationPrepass,
    RejectedFinding,
)
from review_agent.model_adapter_factory import (
    ModelAdapterConfig,
    build_model_adapter_factory_from_config,
)
from review_agent.observations import ObservationStore
from review_agent.reconciler import reconcile_semantically

run_dir = Path(
    r"D:\Agent\code review agent\.worktrees\real-model-baseline"
    r"\.eval-data\deepseek-v4-pro-smoke\repo\.review-agent\runs"
    r"\review-91fd9034aa94"
)
prepass_payload = json.loads(
    (run_dir / "reconciliation_prepass.json").read_text(encoding="utf-8")
)
packet_payload = json.loads(
    (run_dir / "reconciliation_packet.json").read_text(encoding="utf-8")
)
base_sha = prepass_payload["revision_binding"]["base_sha"]
head_sha = prepass_payload["revision_binding"]["head_sha"]
prepass = ReconciliationPrepass(
    review_id=prepass_payload["review_id"],
    base_sha=base_sha,
    head_sha=head_sha,
    candidate_catalog={
        key: FindingCandidate(**value)
        for key, value in prepass_payload["candidate_catalog"].items()
    },
    conflict_hints=[
        ConflictHint(**value) for value in prepass_payload["conflict_hints"]
    ],
    rejected_findings=[
        RejectedFinding(**value)
        for value in prepass_payload["rejected_findings"]
    ],
    contract_coverage=[
        ContractCoverage(**value)
        for value in prepass_payload["contract_coverage"]
    ],
    evidence_quality=prepass_payload["evidence_quality"],
    schema_version=prepass_payload["schema_version"],
)
observations = ObservationStore.load(
    run_dir,
    {
        base_sha + ".." + head_sha,
        "base@" + base_sha,
        "head@" + head_sha,
    },
).list_observations()
adapter = build_model_adapter_factory_from_config(
    ModelAdapterConfig(
        provider_name="openai-compatible",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        api_key_env="REVIEW_AGENT_API_KEY",
    )
).create()
result = reconcile_semantically(
    prepass,
    observations,
    intent_summary=packet_payload.get("intent_summary", {}),
    code_snippets=packet_payload.get("code_snippets", {}),
    policy_summary=packet_payload.get("policy_summary", {}),
    adapter=adapter,
    model="deepseek-v4-pro",
    max_output_tokens=8192,
    max_provider_attempts=2,
    max_elapsed_seconds=240.0,
)
summary = {
    "status": result.status,
    "model_status": result.reconciliation.model.status,
    "batch_statuses": [batch.status for batch in result.batches],
    "batch_elapsed_seconds": [
        round(batch.elapsed_seconds, 3) for batch in result.batches
    ],
    "attempt_statuses": [
        [attempt.status for attempt in batch.attempts]
        for batch in result.batches
    ],
    "canonical_finding_count": len(result.reconciliation.canonical_findings),
    "rejected_finding_count": len(result.reconciliation.rejected_findings),
    "resolved_conflict_count": len(result.reconciliation.conflicts_resolved),
    "remaining_disagreement_count": len(
        result.reconciliation.remaining_disagreements
    ),
    "supplemental_request_count": len(result.supplemental_requests),
    "uncertainty_count": len(result.reconciliation.uncertainties),
}
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
'@ | & 'D:\Anaconda\envs\MINIST\python.exe' -
```

Expected: `status` and `model_status` are `accepted`; every batch status is `accepted`; the accepted attempt has no timeout, truncation, invalid-JSON, or exact-field parse error. Supplemental requests and remaining disagreements may be non-zero because they are semantic business results.

### Task 6: Run one complete `core-py-001` DeepSeek smoke

**Files:**
- Read: `.eval-data/deepseek-v4-pro-smoke/repo/.review-agent/runs/review-91fd9034aa94/request.json`
- Create locally and keep untracked: `.eval-data/deepseek-v4-pro-smoke/repo/.review-agent/runs/review-*/`

- [ ] **Step 1: Run the complete Agent with the frozen Reconciler arguments**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
$oldRun = 'D:\Agent\code review agent\.worktrees\real-model-baseline\.eval-data\deepseek-v4-pro-smoke\repo\.review-agent\runs\review-91fd9034aa94'
$request = Get-Content -LiteralPath (Join-Path $oldRun 'request.json') -Raw | ConvertFrom-Json
& 'D:\Anaconda\envs\MINIST\python.exe' -B -m review_agent review `
  --repo $request.repository_path `
  --base $request.base_revision `
  --head $request.head_revision `
  --title $request.title `
  --intent $request.user_intent `
  --focus $request.review_focus `
  --reviewer-provider openai-compatible `
  --reviewer-model deepseek-v4-pro `
  --reviewer-base-url https://api.deepseek.com `
  --reviewer-mode multi `
  --reviewer-loop agent-loop `
  --risk-assessor-mode model `
  --portfolio-planner-mode model `
  --semantic-reconciler-mode model `
  --semantic-reconciler-max-output-tokens 8192 `
  --semantic-reconciler-max-elapsed-seconds 240 `
  --non-interactive
```

Expected: the command reaches a normal review outcome and prints a new Run directory. `blocked` is allowed; an uncaught provider, parser, timeout, or token-budget exception is not.

- [ ] **Step 2: Inspect only safe structured outcome metadata**

Run:

```powershell
$runsRoot = 'D:\Agent\code review agent\.worktrees\real-model-baseline\.eval-data\deepseek-v4-pro-smoke\repo\.review-agent\runs'
$run = Get-ChildItem -LiteralPath $runsRoot -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
$reviewers = foreach ($i in 0..3) {
    $result = Get-Content -LiteralPath (Join-Path $run.FullName "reviewer_${i}_result.json") -Raw | ConvertFrom-Json
    $trace = Get-Content -LiteralPath (Join-Path $run.FullName "reviewer_${i}_agent_trace.json") -Raw | ConvertFrom-Json
    [pscustomobject]@{
        reviewer = $i
        status = $result.status
        findings = @($result.confirmed_findings).Count
        provider_attempts = $trace.provider_attempt_count
        tool_calls = $trace.tool_call_count
        turns = @($trace.turns).Count
    }
}
$semantic = Get-Content -LiteralPath (Join-Path $run.FullName 'semantic_reconciliation.json') -Raw | ConvertFrom-Json
$completion = Get-Content -LiteralPath (Join-Path $run.FullName 'completion.json') -Raw | ConvertFrom-Json
$risk = Get-Content -LiteralPath (Join-Path $run.FullName 'final_risk.json') -Raw | ConvertFrom-Json
$reviewers | Format-Table -AutoSize
[pscustomobject]@{
    run_id = $run.Name
    semantic_status = $semantic.status
    semantic_model_status = $semantic.model.status
    canonical_findings = @($semantic.canonical_findings).Count
    remaining_disagreements = @($semantic.remaining_disagreements).Count
    supplemental_status = $semantic.supplemental.status
    completion_status = $completion.status
    recommendation = $completion.recommendation
    blockers = @($completion.blockers) -join ' | '
    final_risk = $risk.level
} | Format-List
```

Expected:

- all four Reviewer results are parseable `completed` or `partial`, not `failed`;
- `semantic_status` and `semantic_model_status` are not `fallback`;
- no artifact reports provider timeout, response truncation, invalid JSON, or an undocumented response shape;
- if Completion remains `blocked`, `Intent Packet insufficient` is its blocker rather than a model protocol failure.

### Task 7: Freeze the later evaluation snapshot arguments and complete verification

**Files:**
- Verify: `docs/superpowers/specs/2026-07-29-deepseek-semantic-reconciler-hardening-design.md`
- Verify: `docs/superpowers/plans/2026-07-29-deepseek-semantic-reconciler-hardening.md`

- [ ] **Step 1: Record the exact arguments for the later diagnostic/baseline `prepare` command**

When the formal evaluation Run is prepared, include both arguments in the immutable current-Agent snapshot:

```powershell
--agent-argument=--semantic-reconciler-max-output-tokens=8192
--agent-argument=--semantic-reconciler-max-elapsed-seconds=240
```

Expected: inspection of the prepared Agent snapshot shows both exact strings. Do not start the formal 10-case x 3-trial baseline as part of this hardening plan.

- [ ] **Step 2: Run final repository checks**

Run:

```powershell
git diff master...HEAD --check
git status --short --branch
git log --oneline --decorate -8
```

Expected: committed source and test changes have no whitespace errors; the branch is ahead of `master`; only known local runtime/test directories may be untracked.

- [ ] **Step 3: Request a final code review before integration**

Review the complete `master...HEAD` diff for:

- adherence to the approved design;
- no weakening of Runtime reconciliation validation;
- JSON mode restricted to no-tool requests;
- no API key, private Case text, raw provider response, or hidden reasoning in committed files;
- tests covering the exact Prompt contract and immutable envelope parameter.

Expected: review passes or produces concrete findings that are resolved and re-verified before the branch is offered for merge or PR creation.

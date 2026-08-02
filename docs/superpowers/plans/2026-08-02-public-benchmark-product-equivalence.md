# Public Benchmark Product Equivalence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 AACR-Bench、SWE-PRBench 等无权威 Intent 的公共数据集运行真实产品 `agent-loop`，使用可扩展的 Repository Review Closure，并依据产品 `completion.json` 产出可信、可评分的终态。

**Architecture:** Eval 只负责 Case 权威策略、隔离、身份冻结和测量；Reviewer 的模型、Prompt、工具、预算及执行循环继续来自产品配置。Repository Materializer 保存 Base/Head 完整快照与保持原生 `git log base..head` 所需的 commit 对象，但不保存中间或祖先提交的 tree/blob；公共无权威 Intent 通过现有 Clarification/Resume 边界自动采纳，并留下 `benchmark_auto_accept` 来源。

**Tech Stack:** Python 3.11+, pytest, Git CLI, existing `review_agent`/`review_agent_eval` protocols

---

## Implementation constraints

- 本版不增加 Bash、Reviewer 网络、`run_safe_check` 或 GitHub/PR 集成。
- 不修改 AACR/SWE 原始 Case、Ground Truth、Finding Matcher 或 Judge 阈值。
- 不把 `single-shot` 保留成公共评测回退路径；配置漂移使 Trial 不兼容，不产生能力分数。
- 不把 `100_000` 政名或调大；对象数只作为观测值，物理字节、文件系统节点和总时间仍是 Harness capacity。
- 不使用浅克隆、graft、replace ref 或 Eval 专属 commit-log 语义。为保持产品现有 `git log base..head` 完全一致，Review Closure 保存 Base/Head 可达的 commit metadata；只有 Base/Head 的 tree/blob 进入闭包。
- Agent `blocked`/`failed` 按现有 `failure_as_miss` 计入 Recall miss；Harness materialization failure 使评测无效，不能伪装成 Agent Recall=0。
- Bash 与 Reviewer 网络能力只在独立 VNext Spec 中同步设计产品和 Eval；本计划不预埋 Eval 单边能力开关。
- 按用户的效率要求，只在高风险协议边界添加针对性回归测试；不为常量移动或纯机械重排执行仪式化 red/green 循环，也不在每个 Task 后重跑全套测试。

## File map

**Create**

- `src/review_agent/execution_profile.py` — 从产品 `ReviewExecutionConfig`、Prompt/工具目录、Context/输出上限和风险预算生成唯一 canonical Agent Execution Profile。
- `tests/test_execution_profile.py` — 验证 Profile 对模型、Loop、预算和能力敏感，但不绑定 Trial 私有路径。

**Modify**

- `src/review_agent_eval/repository.py` — Review Closure 枚举、验证、digest、cache/schema version 和对象数语义。
- `tests/eval/test_repository.py` — 历史 tree/blob 排除、commit range、cache 版本和 Base/Head replay 回归。
- `src/review_agent/models.py` — 增加 `benchmark_auto_accept` Intent origin/basis。
- `src/review_agent/intent.py` — 按确认 basis 把 inferred claim 升级成对应的 explicit origin。
- `src/review_agent/intent_clarification.py` — 通过 Resume stdin 表示 benchmark auto-accept，而不伪装成人类确认。
- `tests/test_intent.py`, `tests/test_intent_clarification.py` — 产品侧普通确认与 benchmark 确认分离。
- `src/review_agent_eval/clarification.py` — Case Authority 驱动的自动确认模式、Receipt 和策略版本。
- `src/review_agent_eval/models.py`, `src/review_agent_eval/intent_evaluator.py` — 只在私有 Case Authority 与 benchmark Receipt 同时证明时接受 unmatched Confirm，并保持其不可伪造性。
- `src/review_agent_eval/datasets.py` — 只向 Runner 暴露派生后的 Intent continuation mode，不向 Adapter 暴露 Truth。
- `src/review_agent_eval/runner.py` — 把每个 Case 的 continuation mode 绑定到 Clarification Session。
- `tests/eval/test_clarification_script.py`, `tests/eval/test_runner.py` — 公共 auto-accept 与 Core scripted clarification 回归。
- `src/review_agent/context.py`, `src/review_agent/intent_inference.py`, `src/review_agent/review_contract.py` — 产品 Prompt、Result Contract、Intent loop 和 Reviewer Context 上限的 canonical projection。
- `src/review_agent/tool_gateway.py` — 把产品实际使用的 Tool Gateway 输出、提交消息和超时上限变成有名称的产品常量并投影到 Profile。
- `src/review_agent/session.py`, `src/review_agent/command.py` — 产品执行配置序列化和 CLI 参数解析复用。
- `tests/test_context.py`, `tests/test_intent_inference.py`, `tests/test_tool_gateway.py` — Prompt/工具目录及 Context/Tool/Intent 上限的产品侧身份回归。
- `src/review_agent/completion.py`, `src/review_agent/reconciler.py` — 暴露产品算法 identity 常量，不新增可调配置。
- `src/review_agent_eval/cli.py` — 冻结公共 Agent Profile，显式加入 `--reviewer-loop=agent-loop`，支持精确单 Case smoke 选择。
- `src/review_agent_eval/adapters/base.py` — 增加 execution-profile mismatch 不兼容原因。
- `src/review_agent_eval/adapters/current_agent.py` — Profile 校验、auto-accept Resume token、Completion authority 映射。
- `tests/eval/test_cli.py`, `tests/eval/test_current_agent_adapter.py` — argv/Profile/Completion 端到端映射。
- `src/review_agent_eval/orchestrator.py` — 在评分前拒绝 Harness infrastructure failure。
- `tests/eval/test_metrics.py`, `tests/eval/test_orchestrator_target_replay_v2.py` — failure-as-miss 与 infrastructure separation 回归。

## Task 1: Version the Review Closure policy and remove object count from the v2 manifest

**Files:**

- Modify: `src/review_agent_eval/repository.py:100-410`
- Test: `tests/eval/test_repository.py:1235-1458`

- [ ] **Step 1: Change all identities whose meaning depends on the Git closure**

Use these exact versions; do not accept v1 manifests or indices through a compatibility branch:

```python
PREPARED_REPOSITORY_MANIFEST_SCHEMA_VERSION = "prepared_repository_manifest_v2"
REPOSITORY_ACQUISITION_BINDING_SCHEMA_VERSION = (
    "repository_acquisition_binding_v2"
)
REPOSITORY_BUDGET_POLICY_VERSION = "repository_budget_policy_v2"
LOGICAL_GIT_SOURCE_VERSION = "logical_git_review_closure_v2"
CACHE_INDEX_SCHEMA_VERSION = "repository_cache_index_v2"
```

Keep `REPOSITORY_ISOLATION_POLICY_VERSION = "repository_isolation_v1"`: this plan deliberately avoids shallow/graft/replace behavior, so the isolation contract itself does not change.

- [ ] **Step 2: Replace the fixed object maximum with observational fields**

Keep `actual_objects` and `actual_blobs`, but remove `max_objects` from the v2 budget payload and validation:

```python
def _budget_policy(
    *,
    object_count: int,
    blob_count: int,
    raw_object_bytes: int,
    materialized_files: int,
    materialized_bytes: int,
) -> Dict[str, Any]:
    return {
        "schema_version": REPOSITORY_BUDGET_POLICY_VERSION,
        "object_count_policy": "observed_only",
        "max_blob_bytes": MAX_GIT_BLOB_BYTES,
        "max_materialized_files": MAX_MATERIALIZED_FILES,
        "max_materialized_bytes": MAX_MATERIALIZED_BYTES,
        "max_logical_tree_entries": MAX_LOGICAL_TREE_ENTRIES,
        "max_cache_bytes": MAX_CACHE_BYTES,
        "actual_objects": object_count,
        "actual_blobs": blob_count,
        "actual_raw_object_bytes": raw_object_bytes,
        "actual_materialized_files": materialized_files,
        "actual_materialized_bytes": materialized_bytes,
    }
```

Validate the two counts with `MAX_COUNTER`, which is a serialization bound, not a Case selection policy. Keep `actual_blobs <= actual_objects`, byte limits, path limits, materialized file limits and filesystem-node capacity unchanged.

Do not delete the Python constant in this Task: the old extraction/read paths still reference it until Tasks 2–3 replace those paths. It is no longer serialized as a v2 Case eligibility policy here; Task 3 removes the final internal references and the constant in one runnable commit.

- [ ] **Step 3: Add cache identity regressions**

Add assertions to `test_prepared_repository_budget_policy_is_immutable` and `test_prepared_repository_id_binds_every_serialized_content_field`:

```python
assert manifest.logical_source_version == "logical_git_review_closure_v2"
assert manifest.budget_policy["object_count_policy"] == "observed_only"
assert "max_objects" not in manifest.budget_policy
assert manifest.budget_policy["actual_objects"] > 0
```

Add this separate regression; use `from_dict` because `PreparedRepositoryManifest` deliberately has no permissive JSON convenience loader:

```python
def test_v1_manifest_and_cache_index_cannot_replay_as_v2(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    descriptor, _built = _author_fixture(suite, tmp_path)
    with _preparer(tmp_path, suite) as preparer:
        prepared = preparer.prepare(descriptor)
        old_manifest = prepared.manifest.to_dict()
        old_manifest["schema_version"] = "prepared_repository_manifest_v1"
        with pytest.raises(ValueError, match="schema_version"):
            PreparedRepositoryManifest.from_dict(old_manifest)

        request_id = repository_module._request_id(
            descriptor.digest(),
            prepared.acquisition_binding_digest,
            prepared.git_version,
            prepared.git_executable_sha256,
        )
        index_path = preparer.index_root / f"{request_id}.json"
        old_index = json.loads(index_path.read_text(encoding="utf-8"))
        old_index["schema_version"] = "repository_cache_index_v1"
        index_path.write_text(
            json.dumps(old_index, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        with pytest.raises(RepositoryIntegrityError, match="unknown schema"):
            repository_module._load_cache_index(index_path)
```

This proves old full-history manifests and request indices cannot be reused.

- [ ] **Step 4: Run the focused schema/cache tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_repository.py -k 'budget_policy or prepared_repository_id or cache_manifest or cache_is_content_addressed' -q -p no:cacheprovider --basetemp '.eval-data/pytest-public-equivalence-cache-v2'
```

Expected: all selected tests pass; every v2 manifest omits `max_objects` and reports `object_count_policy=observed_only`.

- [ ] **Step 5: Commit**

```powershell
git add src/review_agent_eval/repository.py tests/eval/test_repository.py
git commit -m "refactor(eval): version repository review closure policy"
```

## Task 2: Enumerate Base/Head snapshots plus commit-only history

**Files:**

- Modify: `src/review_agent_eval/repository.py:2901-3240`
- Modify: `src/review_agent_eval/repository.py:3685-3790`
- Test: `tests/eval/test_repository.py:100-175`
- Test: `tests/eval/test_repository.py:485-768`

- [ ] **Step 1: Add a history fixture that contains objects the Reviewer must not receive**

Create `_init_repository_with_review_history` in `tests/eval/test_repository.py`. Its graph must be `root -> base -> middle -> head`; `legacy.txt` exists only at root and `scratch.txt` exists only at middle. Return all four commit IDs plus the root and middle tree IDs.

The test must assert:

```python
with _preparer(tmp_path, suite) as preparer:
    prepared = preparer.prepare(descriptor)
    with _trial_workspace(preparer, prepared, descriptor) as workspace:
        assert _git(workspace.path, "cat-file", "-t", root) == "commit"
        assert _git(workspace.path, "cat-file", "-t", middle) == "commit"
        assert _git(workspace.path, "log", "--format=%s", f"{base}..{head}") == (
            "head\nmiddle"
        )
        assert _git(workspace.path, "show", f"{base}:app.py") == "value = 1"
        assert _git(workspace.path, "show", f"{head}:app.py") == "value = 3"
        assert _git_exit_code(workspace.path, "cat-file", "-e", root_tree) != 0
        assert _git_exit_code(workspace.path, "cat-file", "-e", middle_tree) != 0
```

The root commit remains because native Git range traversal needs commit ancestry. Its tree and blob do not remain because neither is part of Base or Head.

- [ ] **Step 2: Split object enumeration into two bounded Git walks**

Add `_read_object_id_file` and `_enumerate_review_object_ids`:

```python
def _read_object_id_file(path: Path, object_format: str) -> Set[str]:
    expected_length = 40 if object_format == "sha1" else 64
    result: Set[str] = set()
    with open(path, "rb", buffering=0) as handle:
        for raw_line in handle:
            line = raw_line.rstrip(b"\r\n")
            try:
                oid = line.decode("ascii", "strict")
            except UnicodeDecodeError as exc:
                raise RepositoryIntegrityError(
                    "Git closure enumeration was not ASCII"
                ) from exc
            if len(oid) != expected_length or _GIT_OID_RE.fullmatch(oid) is None:
                raise RepositoryIntegrityError(
                    "Git closure enumeration returned a non-canonical ID"
                )
            result.add(oid)
    return result


def _enumerate_review_object_ids(
    runner: _GitRunner,
    quarantine: Path,
    *,
    object_format: str,
    base_revision: str,
    head_revision: str,
) -> Set[str]:
    commits_path = runner.tmp_root / ("review-commits-" + uuid.uuid4().hex)
    snapshots_path = runner.tmp_root / ("review-snapshots-" + uuid.uuid4().hex)
    try:
        runner.run_to_file(
            [
                "--git-dir",
                str(quarantine),
                "rev-list",
                "--no-object-names",
                base_revision,
                head_revision,
            ],
            commits_path,
            stdout_limit=MAX_CACHE_BYTES,
        )
        runner.run_to_file(
            [
                "--git-dir",
                str(quarantine),
                "rev-list",
                "--objects",
                "--no-object-names",
                "--no-walk",
                base_revision,
                head_revision,
            ],
            snapshots_path,
            stdout_limit=MAX_CACHE_BYTES,
        )
        object_ids = _read_object_id_file(commits_path, object_format)
        object_ids.update(_read_object_id_file(snapshots_path, object_format))
    finally:
        for path in (commits_path, snapshots_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
    if base_revision not in object_ids or head_revision not in object_ids:
        raise RepositoryIntegrityError("review closure omitted a declared revision")
    return object_ids
```

Do not use `git rev-list --objects base head`: that is the full-history behavior being removed.

- [ ] **Step 3: Batch object reads without a count-derived stdout limit**

Use a fixed processing chunk, not a Case gate:

```python
GIT_BATCH_OBJECT_CHUNK = 2_048


def _object_id_chunks(object_ids: Iterable[str]) -> Iterator[Tuple[str, ...]]:
    chunk: List[str] = []
    for oid in sorted(object_ids):
        chunk.append(oid)
        if len(chunk) == GIT_BATCH_OBJECT_CHUNK:
            yield tuple(chunk)
            chunk.clear()
    if chunk:
        yield tuple(chunk)
```

Refactor `_extract_quarantine_closure` to call `cat-file --batch` once per chunk. Continue enforcing `MAX_GIT_BLOB_BYTES`, `MAX_GIT_METADATA_OBJECT_BYTES`, `MAX_CACHE_BYTES`, canonical object hash and exact union equality. Never compute an output limit from object count.

- [ ] **Step 4: Validate commit ancestry separately from endpoint snapshots**

Replace `reachable_from_commit`, which currently requires every commit tree, with two traversals:

```python
commit_ids = {
    oid for oid, obj in canonical.items() if obj.object_type == "commit"
}


def reachable_commits(starts: Iterable[str]) -> Set[str]:
    reachable: Set[str] = set()
    pending = list(starts)
    while pending:
        oid = pending.pop()
        if oid in reachable:
            continue
        if oid not in commit_ids:
            raise RepositoryIntegrityError(
                "commit closure references a missing commit object"
            )
        reachable.add(oid)
        _tree, parents = commit_data(oid)
        pending.extend(parents)
    return reachable


def reachable_tree_objects(tree_oid: str) -> Set[str]:
    reachable: Set[str] = set()
    pending = [(tree_oid, "tree")]
    while pending:
        oid, expected_type = pending.pop()
        if oid in reachable:
            continue
        obj = canonical.get(oid)
        if obj is None or obj.object_type != expected_type:
            raise RepositoryIntegrityError(
                "endpoint snapshot references a missing or wrong-type object"
            )
        reachable.add(oid)
        if expected_type == "tree":
            pending.extend(
                (entry.oid, entry.object_type) for entry in tree_data(oid)
            )
    return reachable
```

Then require:

```python
all_commits = reachable_commits((base_revision, head_revision))
if commit_ids != all_commits:
    raise RepositoryIntegrityError("cache contains commits outside review ancestry")

base_tree, _base_parents = commit_data(base_revision)
head_tree, _head_parents = commit_data(head_revision)
base_snapshot = reachable_tree_objects(base_tree)
head_snapshot = reachable_tree_objects(head_tree)
allowed_objects = all_commits | base_snapshot | head_snapshot
if set(canonical) != allowed_objects:
    raise RepositoryIntegrityError("cache contains objects outside review closure")
```

Run path/collision/blob policy and logical-entry bounds only over `base_tree` and `head_tree`. Intermediate commit tree IDs remain authenticated inside commit bytes but are intentionally absent.

- [ ] **Step 5: Recompute digests over the new exact sets**

Use these sets:

```python
base_reachable = reachable_commits((base_revision,)) | base_snapshot
head_reachable = reachable_commits((head_revision,)) | head_snapshot
union_reachable = base_reachable | head_reachable
```

`LOGICAL_GIT_SOURCE_VERSION` is already mixed into `_logical_source_digest`, so old and new digests cannot compare equal even for a tiny repository whose selected objects happen to match.

- [ ] **Step 6: Run repository closure tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_repository.py -k 'review_history or local_git_materialization or trials_are_independent or sha256 or tree_dag' -q -p no:cacheprovider --basetemp '.eval-data/pytest-public-equivalence-review-closure'
```

Expected: native `git log`, Base/Head reads, replay, SHA-1 and SHA-256 tests pass; root/middle tree objects are absent.

- [ ] **Step 7: Commit**

```powershell
git add src/review_agent_eval/repository.py tests/eval/test_repository.py
git commit -m "feat(eval): materialize hardened repository review closure"
```

## Task 3: Verify cache replay, capacity semantics and product commit tooling

**Files:**

- Modify: `src/review_agent_eval/repository.py:3287-3449`
- Modify: `src/review_agent_eval/repository.py:6280-6505`
- Test: `tests/eval/test_repository.py`
- Test: `tests/test_tool_gateway.py:249-326`

- [ ] **Step 1: Remove the final count-based read/write failures and constant**

Delete the `len(objects) > MAX_GIT_OBJECTS` checks from `_read_loose_repository`, `_closure_from_objects`, any remaining extraction branch and fixture object insertion, then delete `MAX_GIT_OBJECTS` itself. Keep the physical cache-byte and metadata-node checks. `actual_objects` remains in `PreparedRepositoryManifest.budget_policy` and must be compared during cache reload.

- [ ] **Step 2: Extend the prepared-workspace test through `ToolGateway`**

In the history fixture test, instantiate `ToolGateway` inside the Trial workspace:

```python
store = ObservationStore(workspace.path / ".review-agent" / "test-observations")
gateway = ToolGateway(
    workspace.path,
    base,
    head,
    store,
    allowed_tools=("read_commit_messages",),
)
result = gateway.execute("read_commit_messages", {"max_commits": 10})
assert '"subject": "head"' in result.context_view
assert '"subject": "middle"' in result.context_view
assert '"subject": "root"' not in result.context_view
```

This is the acceptance proof for the unresolved design question around `ToolGateway._read_commit_messages`: no shallow metadata and no alternate Eval implementation are needed because commit ancestry remains present.

- [ ] **Step 3: Verify immutable cache reload uses v2 semantics**

Prepare the same descriptor twice, close the first `RepositoryPreparer`, reopen in `RepositoryMode.CACHE_ONLY`, and assert:

```python
assert replayed.cache_id == prepared.cache_id
assert replayed.manifest.source_digest == prepared.manifest.source_digest
assert replayed.manifest.budget_policy["actual_objects"] == (
    prepared.manifest.budget_policy["actual_objects"]
)
```

Also assert a forged v1 request index cannot point at the v2 cache entry.

- [ ] **Step 4: Run the complete repository module tests once**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_repository.py tests/test_tool_gateway.py -q -p no:cacheprovider --basetemp '.eval-data/pytest-public-equivalence-repository-complete'
```

Expected: all tests pass. A physical cache/node overflow still raises `RepositoryLimitError`; no error message describes an object-count budget.

- [ ] **Step 5: Commit**

```powershell
git add src/review_agent_eval/repository.py tests/eval/test_repository.py tests/test_tool_gateway.py
git commit -m "test(eval): verify review closure replay and commit tools"
```

## Task 4: Add truthful benchmark auto-accept provenance to the product

**Files:**

- Modify: `src/review_agent/models.py:11-335`
- Modify: `src/review_agent/intent.py:245-360`
- Modify: `src/review_agent/intent_clarification.py:31-112`
- Test: `tests/test_intent.py:250-327`
- Test: `tests/test_intent_clarification.py`

- [ ] **Step 1: Add the canonical origin and basis**

```python
BENCHMARK_AUTO_ACCEPT_BASIS = "benchmark_auto_accept"


class IntentOrigin(str, Enum):
    USER_INPUT = "user_input"
    REQUEST_METADATA = "request_metadata"
    PROJECT_RULE = "project_rule"
    REPOSITORY_DOCUMENT = "repository_document"
    REPOSITORY_TEST = "repository_test"
    COMMIT_MESSAGE = "commit_message"
    LLM_INFERENCE = "llm_inference"
    USER_CONFIRMATION = "user_confirmation"
    BENCHMARK_AUTO_ACCEPT = "benchmark_auto_accept"
    USER_CORRECTION = "user_correction"
    CHANGED_FILES = "changed_files"
    PROJECT_MEMORY = "project_memory"
```

Allow `IntentDecision(action=CONFIRMED)` to carry either no `continuation_basis` or exactly `BENCHMARK_AUTO_ACCEPT_BASIS`. Other non-skip arbitrary confirmation bases must fail validation.

- [ ] **Step 2: Preserve ordinary user confirmation behavior**

In `apply_user_decision` select the origin from the basis:

```python
confirmation_origin = (
    IntentOrigin.BENCHMARK_AUTO_ACCEPT
    if decision.continuation_basis == BENCHMARK_AUTO_ACCEPT_BASIS
    else IntentOrigin.USER_CONFIRMATION
)
updated_claims = [
    replace(
        claim,
        source=IntentSource.EXPLICIT,
        origin=confirmation_origin,
        confidence=IntentConfidence.HIGH,
    )
    if claim in confirmable
    else claim
    for claim in updated_claims
]
```

The final active claim has `source=explicit` in both paths. Only its audit origin differs.

- [ ] **Step 3: Add an exact non-human Resume token**

Before the ordinary `confirm` branch in `ConsoleIntentClarifier.decide`, recognize only:

```python
if raw == "confirm:benchmark-auto-accept":
    if not question.proposed_values:
        self._output("There is no proposed value to auto-accept.")
        continue
    if question.field is IntentField.GOAL and len(question.proposed_values) > 1:
        self._output("Conflicting goal candidates cannot be auto-accepted.")
        continue
    return IntentDecision(
        question_id=question.question_id,
        action=IntentDecisionAction.CONFIRMED,
        continuation_basis=BENCHMARK_AUTO_ACCEPT_BASIS,
    )
```

Do not reinterpret `confirm`, `yes` or `y`; those remain `USER_CONFIRMATION`.

- [ ] **Step 4: Add product regressions**

Keep `test_console_confirm_uses_proposed_value` and add `assert decision.continuation_basis is None`. Add the exact token regression in `tests/test_intent_clarification.py`:

```python
def test_console_benchmark_auto_accept_uses_non_human_basis():
    decision = ConsoleIntentClarifier(
        input_fn=lambda _prompt: "confirm:benchmark-auto-accept",
        output_fn=lambda _message: None,
    ).decide(question())

    assert decision is not None
    assert decision.action is IntentDecisionAction.CONFIRMED
    assert decision.continuation_basis == "benchmark_auto_accept"
```

Add `test_benchmark_auto_accept_promotes_with_truthful_origin` beside the existing normal-confirm test in `tests/test_intent.py`:

```python
inferred = _claim(IntentField.GOAL, "Bound job retries")
questions = generate_material_questions([inferred])
updated_claims, updated_questions = apply_user_decision(
    [inferred],
    questions,
    IntentDecision(
        question_id=questions[0].question_id,
        action=IntentDecisionAction.CONFIRMED,
        continuation_basis=BENCHMARK_AUTO_ACCEPT_BASIS,
    ),
)
accepted = updated_claims[0]
updated_question = updated_questions[0]
assert accepted.source is IntentSource.EXPLICIT
assert accepted.origin is IntentOrigin.BENCHMARK_AUTO_ACCEPT
assert updated_question.continuation_basis == "benchmark_auto_accept"
assert updated_question.status is ClarificationStatus.CONFIRMED
```

The existing `test_confirm_upgrades_only_the_inferred_values_in_a_mixed_list_field` remains the assertion that ordinary confirmation uses `USER_CONFIRMATION`.

- [ ] **Step 5: Run product Intent tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_intent.py tests/test_intent_clarification.py -q -p no:cacheprovider --basetemp '.eval-data/pytest-public-equivalence-product-intent'
```

Expected: normal user behavior is unchanged and benchmark acceptance has truthful provenance.

- [ ] **Step 6: Commit**

```powershell
git add src/review_agent/models.py src/review_agent/intent.py src/review_agent/intent_clarification.py tests/test_intent.py tests/test_intent_clarification.py
git commit -m "feat(intent): record benchmark auto-accept provenance"
```

## Task 5: Drive auto-accept from Case Authority without exposing Truth to the Adapter

**Files:**

- Modify: `src/review_agent_eval/clarification.py`
- Modify: `src/review_agent_eval/models.py:1209-1275`
- Modify: `src/review_agent_eval/models.py:3785-3820`
- Modify: `src/review_agent_eval/intent_evaluator.py:3320-3630`
- Modify: `src/review_agent_eval/datasets.py:486-492`
- Modify: `src/review_agent_eval/runner.py:473-615`
- Modify: `src/review_agent_eval/cli.py:566-614`
- Modify: `src/review_agent_eval/adapters/current_agent.py:578-645`
- Test: `tests/eval/test_clarification_script.py`
- Test: `tests/eval/test_models.py`
- Test: `tests/eval/test_intent_evaluator.py`
- Test: `tests/eval/test_runner.py`
- Test: `tests/eval/test_current_agent_adapter.py`

- [ ] **Step 1: Define one versioned authority rule**

```python
BENCHMARK_AUTO_ACCEPT_POLICY_VERSION = "case-authority-auto-accept-v1"


class IntentContinuationMode(str, Enum):
    SCRIPTED = "scripted"
    BENCHMARK_AUTO_ACCEPT = "benchmark_auto_accept"


def intent_continuation_mode_for_case(case: EvalCase) -> IntentContinuationMode:
    if not isinstance(case, EvalCase):
        raise TypeError("intent continuation mode requires EvalCase")
    if not case.intent_truth.scorable and not case.clarification_script.answers:
        return IntentContinuationMode.BENCHMARK_AUTO_ACCEPT
    return IntentContinuationMode.SCRIPTED
```

Do not branch on `suite_id`, adapter name, AACR or SWE strings.

- [ ] **Step 2: Expose only the derived mode from `CaseBank`**

```python
def intent_continuation_mode(self, task_id: str) -> IntentContinuationMode:
    case = _load_case_from_handle(self.handle(task_id))
    return intent_continuation_mode_for_case(case)
```

Update Runner’s provider protocol with `intent_continuation_mode(task_id)`. `_LazyClarificationSession` receives the enum, never `IntentTruth`, and passes it to `ClarificationSession`.

- [ ] **Step 3: Produce an auditable unmatched confirm**

Add `BENCHMARK_AUTO_ACCEPTED` to `MaterialClaimMatchOutcome`.

The existing canonical Exchange rejects every answered action without `matched_answer_id`. Extend only its exact benchmark shape:

```python
benchmark_auto_accept = (
    self.action is ClarificationAction.CONFIRM
    and self.matched_answer_id is None
    and self.response is None
    and bool(resolved)
)
if (
    self.matched_answer_id is None
    and not policy_skip
    and not benchmark_auto_accept
):
    raise _error("answered clarification must have matched_answer_id")
```

In `validate_submission_for_case`, reject that shape unless the private Case proves the same authority rule:

```python
if (
    exchange.action is ClarificationAction.CONFIRM
    and exchange.matched_answer_id is None
    and (
        case.intent_truth.scorable
        or bool(case.clarification_script.answers)
    )
):
    raise _error(
        "benchmark auto-accept is not authorized by this Case"
    )
```

The JSON field set does not change, so keep `eval_submission_v2`; the newly legal combination is bound by `case-authority-auto-accept-v1` in Agent/Run identity and still fails private Case validation everywhere else.

In `ClarificationSession.__ask`, when mode is `BENCHMARK_AUTO_ACCEPT` and the question is inside `max_rounds`, require non-empty proposed values and return:

```python
SubmissionClarificationExchange(
    turn_index=turn_index,
    question_id=question_id,
    dimension=dimension,
    question=asked_text,
    material_claim=asked_claim,
    matched_answer_id=None,
    action=ClarificationAction.CONFIRM,
    response=None,
    resolved_values=proposed,
)
```

The Receipt outcome is `benchmark_auto_accepted`, with no candidates and no consumed scripted answer. Scripted Core answers continue through the matcher unchanged.

Pass this exact private authorization into `_validate_receipt`:

```python
allow_benchmark_auto_accept = (
    policy is None and not script.answers
)
```

In `_clarification`, require the Receipt whenever the transcript uses the benchmark shape:

```python
benchmark_exchange = (
    exchange.action is ClarificationAction.CONFIRM
    and exchange.matched_answer_id is None
)
if benchmark_exchange and receipt is None:
    raise _error("benchmark auto-accept requires its Harness receipt")
```

Before the existing MATCHED/AMBIGUOUS/UNMATCHED derivation, validate the new outcome:

```python
if receipt.outcome is MaterialClaimMatchOutcome.BENCHMARK_AUTO_ACCEPTED:
    if (
        not allow_benchmark_auto_accept
        or round_limited
        or decisions
        or receipt.matched_answer_id is not None
        or exchange.matched_answer_id is not None
        or exchange.action is not ClarificationAction.CONFIRM
        or exchange.response is not None
        or not exchange.resolved_values
    ):
        raise _error("benchmark auto-accept receipt is unauthorized")
elif (
    exchange.action is ClarificationAction.CONFIRM
    and exchange.matched_answer_id is None
):
    raise _error("unmatched Confirm requires benchmark auto-accept receipt")
```

Run the existing MATCHED/AMBIGUOUS/UNMATCHED expected-outcome calculation only when the outcome is not `BENCHMARK_AUTO_ACCEPTED`. Every other unmatched Confirm or forged auto-accept Receipt raises `IntentEvaluationError`.

In `_clarification`, treat `BENCHMARK_AUTO_ACCEPTED` as material but not as a consumed scripted answer:

```python
auto_accepted = (
    receipt.outcome
    is MaterialClaimMatchOutcome.BENCHMARK_AUTO_ACCEPTED
)
material = (
    receipt.outcome is MaterialClaimMatchOutcome.MATCHED
    or auto_accepted
)
consumed = True if receipt.outcome is MaterialClaimMatchOutcome.MATCHED else (
    None if auto_accepted else False
)
if auto_accepted:
    final_claims = {
        item.normalized_text
        for item in projected
        if item.dimension is exchange.dimension
    }
    resolved = {
        normalize_intent_text(item) for item in exchange.resolved_values
    }
    update = bool(resolved) and resolved.issubset(final_claims)
else:
    update = self._clarification_update(
        projected,
        exchange,
        answer if material else None,
    )
```

Do not synthesize a `ClarificationAnswer` or add `CLARIFICATION_ANSWER_NOT_CONSUMED` for the auto-accepted branch.

- [ ] **Step 4: Send the product token only for unmatched confirms**

Update `_answer_input`:

```python
if action is ClarificationAction.CONFIRM:
    if exchange.matched_answer_id is None:
        return b"confirm:benchmark-auto-accept\n"
    return b"confirm\n"
```

An unmatched confirm can only be created by the versioned Case Authority policy. Do not infer it from dataset name or empty text.

- [ ] **Step 5: Bind the policy version into Agent identity**

Change the `clarification` parameter shape to:

```python
"clarification": {
    "unanswered_action": args.unanswered_clarification.replace("-", "_"),
    "intent_continuation_policy_version": (
        BENCHMARK_AUTO_ACCEPT_POLICY_VERSION
    ),
}
```

Update `unanswered_clarification_action` to require exactly those two fields and reject any policy version other than `case-authority-auto-accept-v1`.

- [ ] **Step 6: Add four policy regressions**

Cover exactly:

1. `intent_truth.scorable=false` + no answers -> unmatched Confirm -> final explicit claim with `benchmark_auto_accept` origin.
2. `intent_truth.scorable=true` -> no automatic confirm.
3. A Case with a scripted answer -> matcher consumes that answer and preserves normal user/script semantics.
4. A directly constructed unmatched Confirm or forged `benchmark_auto_accepted` Receipt fails unless the private Case authority rule is satisfied.

Also assert the final `review_brief.json` provenance has no active `source=inferred` for the public path.

- [ ] **Step 7: Run focused Eval clarification tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_models.py tests/eval/test_clarification_script.py tests/eval/test_intent_evaluator.py tests/eval/test_runner.py tests/eval/test_current_agent_adapter.py -k 'clarification or intent or auto_accept or policy_skip' -q -p no:cacheprovider --basetemp '.eval-data/pytest-public-equivalence-eval-intent'
```

Expected: public authority auto-accepts; Core scripted behavior remains unchanged.

- [ ] **Step 8: Commit**

```powershell
git add src/review_agent_eval/clarification.py src/review_agent_eval/models.py src/review_agent_eval/intent_evaluator.py src/review_agent_eval/datasets.py src/review_agent_eval/runner.py src/review_agent_eval/cli.py src/review_agent_eval/adapters/current_agent.py tests/eval/test_models.py tests/eval/test_clarification_script.py tests/eval/test_intent_evaluator.py tests/eval/test_runner.py tests/eval/test_current_agent_adapter.py
git commit -m "feat(eval): auto-accept unscorable benchmark intent"
```

## Task 6: Create the product-owned canonical Agent Execution Profile

**Files:**

- Create: `src/review_agent/execution_profile.py`
- Modify: `src/review_agent/context.py:69-260`
- Modify: `src/review_agent/intent_inference.py:77-120`
- Modify: `src/review_agent/tool_gateway.py:45-100`
- Modify: `src/review_agent/review_contract.py`
- Modify: `src/review_agent/session.py:359-410`
- Modify: `src/review_agent/session.py:1355-1500`
- Modify: `src/review_agent/command.py:724-820`
- Modify: `src/review_agent/command.py:4183-4228`
- Modify: `src/review_agent/completion.py`
- Modify: `src/review_agent/reconciler.py`
- Create: `tests/test_execution_profile.py`
- Modify: `tests/test_context.py`
- Modify: `tests/test_intent_inference.py`
- Modify: `tests/test_tool_gateway.py`

- [ ] **Step 1: Extract one serializer for `ReviewExecutionConfig`**

Move the nested model-stage and supplemental-policy payload builders out of `session_manifest_to_dict` into:

```python
def model_stage_config_to_dict(config: ModelStageConfig) -> dict[str, Any]:
    if not isinstance(config, ModelStageConfig):
        raise TypeError("config must be ModelStageConfig")
    return {
        "mode": config.mode,
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "max_output_tokens": config.max_output_tokens,
        "max_provider_attempts": config.max_provider_attempts,
        "max_elapsed_seconds": config.max_elapsed_seconds,
    }


def supplemental_policy_to_dict(config: SupplementalPolicy) -> dict[str, Any]:
    if not isinstance(config, SupplementalPolicy):
        raise TypeError("config must be SupplementalPolicy")
    return {
        "version": config.version,
        "risk_level": config.risk_level,
        "max_waves": config.max_waves,
        "max_tasks": config.max_tasks,
        "max_tasks_per_wave": config.max_tasks_per_wave,
        "max_concurrency": config.max_concurrency,
        "max_turns_per_task": config.max_turns_per_task,
        "max_tool_calls_per_task": config.max_tool_calls_per_task,
        "max_tokens_per_task": config.max_tokens_per_task,
        "max_total_tokens": config.max_total_tokens,
        "max_elapsed_seconds": config.max_elapsed_seconds,
    }


def review_execution_config_to_dict(
    execution: ReviewExecutionConfig,
) -> dict[str, Any]:
    if not isinstance(execution, ReviewExecutionConfig):
        raise TypeError("execution must be ReviewExecutionConfig")
    return {
        "reviewer_provider": execution.reviewer_provider,
        "reviewer_model": execution.reviewer_model,
        "reviewer_base_url": execution.reviewer_base_url,
        "reviewer_api_key_env": execution.reviewer_api_key_env,
        "reviewer_mode": execution.reviewer_mode,
        "reviewer_loop": execution.reviewer_loop,
        "non_interactive": execution.non_interactive,
        "risk_assessor": model_stage_config_to_dict(execution.risk_assessor),
        "portfolio_planner": model_stage_config_to_dict(
            execution.portfolio_planner
        ),
        "semantic_reconciler": model_stage_config_to_dict(
            execution.semantic_reconciler
        ),
        "supplemental_policy": supplemental_policy_to_dict(
            execution.supplemental_policy
        ),
        "memory": None if execution.memory is None else execution.memory.to_dict(),
        "memory_curator": model_stage_config_to_dict(execution.memory_curator),
    }
```

`session_manifest_to_dict` must call this function, so Profile and persisted Session cannot drift through duplicated serialization logic.

- [ ] **Step 2: Expose protocol identities from their owning modules**

Add the constants to their owners: `REVIEWER_PROTOCOL_VERSION` in `context.py`, `REVIEW_CONTRACT_VALIDATION_VERSION` in `review_contract.py`, `COMPLETION_POLICY_VERSION` in `completion.py`, and `RECONCILIATION_POLICY_VERSION` in `reconciler.py`:

```python
REVIEWER_PROTOCOL_VERSION = "reviewer-protocol-v1"
REVIEW_CONTRACT_VALIDATION_VERSION = "review-contract-validation-v1"
COMPLETION_POLICY_VERSION = "completion-policy-v1"
RECONCILIATION_POLICY_VERSION = "evidence-reconciliation-v1"
```

In `context.py`, replace the remaining Reviewer invocation literals with product constants and use them in `build_reviewer_envelope`:

```python
REVIEWER_REASONING_EFFORT = "medium"
REVIEWER_TEMPERATURE = 0
REVIEWER_TOOL_CHOICE_POLICY = "auto_if_tools_else_none"
REVIEWER_RESPONSE_SCHEMA = "reviewer_assignment_result_v2"
```

Expose `reviewer_protocol_projection()` with exact keys `version`, `system_prompt_sha256`, `result_contract_sha256`, `tool_result_protocol_sha256`, `tool_catalog_sha256`, `tool_names`, `context_budget`, and `invocation_defaults`. The catalog digest is over canonical JSON for the complete `_REVIEWER_TOOL_DEFINITIONS` payload. `context_budget` must be `asdict(ContextBudget())`, so the Profile binds `max_message_chars`, compaction reserve and the Memory sub-budget from the same product class that builds model messages. `invocation_defaults` contains the four constants above; model and output/time budgets remain in `ReviewExecutionConfig` and expanded Risk Profiles.

In `tool_gateway.py`, replace signature literals with named product constants and use them as the constructor defaults:

```python
DEFAULT_TOOL_CONTEXT_CHARS = 4_000
DEFAULT_TOOL_TIMEOUT_SECONDS = 10
DEFAULT_MAX_COMMIT_MESSAGES = 50
DEFAULT_MAX_COMMIT_BODY_CHARS = 4_000


def tool_gateway_limits_projection() -> dict[str, int]:
    return {
        "max_context_chars": DEFAULT_TOOL_CONTEXT_CHARS,
        "timeout_seconds": DEFAULT_TOOL_TIMEOUT_SECONDS,
        "max_commit_messages": DEFAULT_MAX_COMMIT_MESSAGES,
        "max_commit_body_chars": DEFAULT_MAX_COMMIT_BODY_CHARS,
    }
```

In `intent_inference.py`, name the existing defaults instead of copying them into Eval:

```python
INTENT_INFERENCE_PROTOCOL_VERSION = "intent-inference-protocol-v1"
INTENT_INFERENCE_MAX_TURNS = 4
INTENT_INFERENCE_MAX_TOOL_CALLS = 8
INTENT_INFERENCE_MAX_OUTPUT_TOKENS = 4_096
INTENT_INFERENCE_REASONING_EFFORT = "low"
INTENT_INFERENCE_TEMPERATURE = 0
INTENT_INFERENCE_TOOL_CHOICE = "auto"
INTENT_INFERENCE_RESPONSE_SCHEMA = "intent_inference_result_v1"
```

Use those constants in `run_intent_inference`. Expose `intent_inference_protocol_projection()` with exact keys `version`, `system_prompt_sha256`, `tool_catalog_sha256`, `tool_names`, `runtime_limits`, and `invocation_defaults`; `runtime_limits` contains turns/tool calls/output tokens, while `invocation_defaults` contains reasoning effort, temperature, tool choice and response schema. Build the canonical tool payload as `[asdict(item) for item in _intent_tool_specs()]`, hash its sorted compact JSON, and derive `tool_names` from the same list. It includes `read_commit_messages`.

In `execution_profile.py`, also hash `RISK_MODEL_SYSTEM_PROMPT`, `PORTFOLIO_PLANNER_SYSTEM_PROMPT`, `SEMANTIC_RECONCILER_SYSTEM_PROMPT`, and `MEMORY_CURATOR_SYSTEM_PROMPT` under a `stage_prompt_sha256` mapping. Do not hash Python source files or absolute paths.

Project the already product-owned adapter constants from `model_adapter.py`; do not copy their numeric values into Eval:

```python
def provider_transport_projection() -> dict[str, Any]:
    return {
        "openai_compatible": {
            "request_timeout_seconds": (
                DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
            ),
            "max_response_bytes": DEFAULT_MAX_RESPONSE_BYTES,
        },
    }
```

This binds the provider response-byte ceiling and the per-request timeout used by Intent inference. Reviewer/model-stage requests may request a lower timeout from their own persisted elapsed budget; the transport never exceeds this product ceiling.

- [ ] **Step 3: Implement a projection-only Profile**

`AgentExecutionProfile` must be constructible from `ReviewExecutionConfig`; it must not carry independent editable defaults:

```python
AGENT_EXECUTION_PROFILE_SCHEMA_VERSION = "agent_execution_profile_v1"
PRODUCT_ORCHESTRATION_MARGIN_SECONDS = 300.0


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stage_prompt_digests() -> dict[str, str]:
    return {
        "risk_assessor": _text_sha256(RISK_MODEL_SYSTEM_PROMPT),
        "portfolio_planner": _text_sha256(PORTFOLIO_PLANNER_SYSTEM_PROMPT),
        "semantic_reconciler": _text_sha256(
            SEMANTIC_RECONCILER_SYSTEM_PROMPT
        ),
        "memory_curator": _text_sha256(MEMORY_CURATOR_SYSTEM_PROMPT),
    }


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError("execution profile contains a non-JSON value")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _minimum_outer_timeout_seconds(
    execution: ReviewExecutionConfig,
    profiles: Mapping[str, Mapping[str, Any]],
) -> float:
    stage_seconds = sum(
        stage.max_elapsed_seconds
        for stage in (
            execution.risk_assessor,
            execution.portfolio_planner,
            execution.semantic_reconciler,
            execution.memory_curator,
        )
        if stage.mode == "model"
    )
    initial_review_seconds = (
        max(
            float(profile["max_elapsed_seconds"])
            * (
                int(profile["reviewer_count"])
                if execution.reviewer_mode == "single"
                else 1
            )
            for profile in profiles.values()
        )
        if execution.reviewer_provider != "none"
        else 0.0
    )
    intent_inference_seconds = (
        INTENT_INFERENCE_MAX_TURNS
        * DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
        if execution.reviewer_provider == "openai-compatible"
        else 0.0
    )
    intent_tool_seconds = (
        INTENT_INFERENCE_MAX_TOOL_CALLS * DEFAULT_TOOL_TIMEOUT_SECONDS
        if execution.reviewer_provider != "none"
        else 0.0
    )
    return (
        stage_seconds
        + intent_inference_seconds
        + intent_tool_seconds
        + initial_review_seconds
        + execution.supplemental_policy.max_elapsed_seconds
        + PRODUCT_ORCHESTRATION_MARGIN_SECONDS
    )


@dataclass(frozen=True)
class AgentExecutionProfile:
    payload: Mapping[str, Any]

    @classmethod
    def from_execution(
        cls,
        execution: ReviewExecutionConfig,
    ) -> "AgentExecutionProfile":
        execution_payload = review_execution_config_to_dict(execution)
        memory = execution_payload["memory"]
        if memory is not None:
            memory = dict(memory)
            memory.pop("root_path")
            memory["root_binding"] = "trial_private"
            execution_payload["memory"] = memory
        profiles = {
            risk.value: asdict(ReviewProfile.for_risk(risk))
            for risk in RiskLevel
        }
        return cls(
            _freeze_json(
                {
                    "schema_version": AGENT_EXECUTION_PROFILE_SCHEMA_VERSION,
                    "execution": execution_payload,
                    "risk_profiles": profiles,
                    "reviewer_protocol": reviewer_protocol_projection(),
                    "intent_protocol": intent_inference_protocol_projection(),
                    "tool_gateway_limits": tool_gateway_limits_projection(),
                    "provider_transport": provider_transport_projection(),
                    "stage_prompt_sha256": stage_prompt_digests(),
                    "review_contract_version": REVIEW_CONTRACT_VALIDATION_VERSION,
                    "reconciliation_policy_version": RECONCILIATION_POLICY_VERSION,
                    "completion_policy_version": COMPLETION_POLICY_VERSION,
                    "minimum_outer_timeout_seconds": (
                        _minimum_outer_timeout_seconds(execution, profiles)
                    ),
                    "capabilities": {
                        "shell": "unavailable",
                        "network": "provider_only",
                        "repository": "read_only",
                        "run_safe_check": "unavailable",
                    },
                }
            )
        )

    @classmethod
    def from_dict(cls, value: Any) -> "AgentExecutionProfile":
        if not isinstance(value, Mapping):
            raise ValueError("execution profile must be a JSON object")
        expected = {
            "schema_version",
            "execution",
            "risk_profiles",
            "reviewer_protocol",
            "intent_protocol",
            "tool_gateway_limits",
            "provider_transport",
            "stage_prompt_sha256",
            "review_contract_version",
            "reconciliation_policy_version",
            "completion_policy_version",
            "minimum_outer_timeout_seconds",
            "capabilities",
        }
        if set(value) != expected:
            raise ValueError("execution profile fields are not canonical")
        if value["schema_version"] != AGENT_EXECUTION_PROFILE_SCHEMA_VERSION:
            raise ValueError("execution profile schema is unsupported")
        return cls(_freeze_json(value))

    def to_dict(self) -> dict[str, Any]:
        return _thaw_json(self.payload)

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
```

The Base URL and API-key environment variable name are identity; the API key value and Trial-specific Memory root are not representable.

- [ ] **Step 4: Reuse product argument resolution**

Extract the block currently constructing/replacing `ReviewExecutionConfig` in `_run_review` into `resolve_review_execution_config(args)`. Add:

```python
def review_execution_profile_from_arguments(
    review_arguments: Sequence[str],
    *,
    memory_mode: str,
    memory_root: Path,
) -> AgentExecutionProfile:
    parsed = _build_parser().parse_args(
        [
            "review",
            "--base=" + ("0" * 40),
            "--head=" + ("1" * 40),
            *review_arguments,
            "--memory-mode=" + memory_mode,
            "--memory-root=" + str(memory_root),
        ]
    )
    return AgentExecutionProfile.from_execution(
        resolve_review_execution_config(parsed)
    )
```

The helper parses the product parser with fixed dummy Base/Head IDs plus the exact `--memory-mode` and absolute Trial-private `--memory-root`, calls `resolve_review_execution_config`, then calls `AgentExecutionProfile.from_execution`. The root path is removed by the canonical projection, but Memory permission and limits remain. Eval must call this product helper instead of recreating product or Memory defaults.

- [ ] **Step 5: Add Profile tests**

Assert:

```python
assert first.digest() == same_config_different_memory_root.digest()
single_shot = AgentExecutionProfile.from_execution(
    replace(config, reviewer_loop="single-shot")
)
assert first.digest() != single_shot.digest()
assert first.payload["capabilities"] == {
    "shell": "unavailable",
    "network": "provider_only",
    "repository": "read_only",
    "run_safe_check": "unavailable",
}
assert "search_code" in first.payload["reviewer_protocol"]["tool_names"]
assert "bash" not in first.payload["reviewer_protocol"]["tool_names"]
assert "read_commit_messages" in first.payload["intent_protocol"]["tool_names"]
assert first.payload["reviewer_protocol"]["context_budget"] == asdict(
    ContextBudget()
)
assert first.payload["reviewer_protocol"]["invocation_defaults"] == {
    "reasoning_effort": "medium",
    "temperature": 0,
    "tool_choice_policy": "auto_if_tools_else_none",
    "response_schema": "reviewer_assignment_result_v2",
}
assert first.payload["tool_gateway_limits"] == {
    "max_context_chars": 4_000,
    "timeout_seconds": 10,
    "max_commit_messages": 50,
    "max_commit_body_chars": 4_000,
}
assert first.payload["intent_protocol"]["runtime_limits"] == {
    "max_turns": 4,
    "max_tool_calls": 8,
    "max_output_tokens": 4_096,
}
assert first.payload["intent_protocol"]["invocation_defaults"] == {
    "reasoning_effort": "low",
    "temperature": 0,
    "tool_choice": "auto",
    "response_schema": "intent_inference_result_v1",
}
assert first.payload["provider_transport"]["openai_compatible"] == {
    "request_timeout_seconds": 180,
    "max_response_bytes": 16 * 1024 * 1024,
}
```

Also verify changing a risk budget, Prompt/tool digest, Context limit, Tool Gateway limit, Intent output limit or provider transport limit changes the Profile digest.

Assert `minimum_outer_timeout_seconds` is at least the sum of bounded Intent model/tool work, configured model-stage deadlines, supplemental deadline and the fixed persistence/retry margin. For `reviewer_mode=single`, the Review component is the maximum Risk Profile’s `reviewer_count * max_elapsed_seconds` because the full Portfolio runs sequentially; for `multi`, it is the maximum single Reviewer deadline because the Portfolio runs concurrently. This value is a lower bound for Eval’s outer process timeout, not a second review-depth policy.

- [ ] **Step 6: Run Profile and Session tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_execution_profile.py tests/test_session.py tests/test_context.py tests/test_intent_inference.py tests/test_tool_gateway.py -q -p no:cacheprovider --basetemp '.eval-data/pytest-public-equivalence-profile'
```

Expected: all tests pass; Profile construction imports no Eval module.

- [ ] **Step 7: Commit**

```powershell
git add src/review_agent/execution_profile.py src/review_agent/context.py src/review_agent/intent_inference.py src/review_agent/tool_gateway.py src/review_agent/review_contract.py src/review_agent/session.py src/review_agent/command.py src/review_agent/completion.py src/review_agent/reconciler.py tests/test_execution_profile.py tests/test_session.py tests/test_context.py tests/test_intent_inference.py tests/test_tool_gateway.py
git commit -m "feat(runtime): expose canonical agent execution profile"
```

## Task 7: Freeze `agent-loop` and reject actual Session profile drift

**Files:**

- Modify: `src/review_agent_eval/cli.py:271-323`
- Modify: `src/review_agent_eval/cli.py:566-614`
- Modify: `src/review_agent_eval/adapters/base.py:287-322`
- Modify: `src/review_agent_eval/adapters/current_agent.py:100-260`
- Modify: `src/review_agent_eval/adapters/current_agent.py:833-917`
- Modify: `src/review_agent_eval/adapters/current_agent.py:918-1228`
- Test: `tests/eval/test_cli.py`
- Test: `tests/eval/test_current_agent_adapter.py:678-839`

- [ ] **Step 1: Make the public current-Agent loop explicit and non-overridable**

Do not add `--reviewer-loop` to `_FORBIDDEN_REVIEW_ARGUMENTS`, because that set is applied to the final frozen argv and would reject Eval’s own canonical argument.

In `cli.py`, reject user-provided `--agent-argument` options controlled by the public Profile before appending canonical values:

```python
_PROFILE_CONTROLLED_REVIEW_ARGUMENTS = frozenset(
    {
        "--reviewer-provider",
        "--reviewer-model",
        "--reviewer-base-url",
        "--reviewer-api-key-env",
        "--reviewer-loop",
        "--memory-mode",
        "--memory-root",
    }
)


def _validate_user_review_arguments(arguments: Sequence[str]) -> None:
    for argument in arguments:
        option = argument.split("=", 1)[0]
        controlled_abbreviation = (
            option.startswith("--")
            and len(option) > 2
            and any(
                value.startswith(option)
                for value in _PROFILE_CONTROLLED_REVIEW_ARGUMENTS
            )
        )
        if option in _PROFILE_CONTROLLED_REVIEW_ARGUMENTS or controlled_abbreviation:
            raise CliUsageError(
                "--agent-argument cannot override the product Profile"
            )
```

In `_default_agent_snapshot`, call that helper on the raw user arguments, then append these product arguments:

```python
review_arguments.extend(
    [
        "--reviewer-loop=agent-loop",
        "--reviewer-api-key-env=" + args.agent_api_key_env,
    ]
)
```

No code path may append `single-shot`. Keep the product CLI’s general default unchanged for real non-Eval users; public Eval never relies on that default.

In `CurrentAgentAdapter._configuration`, require the final frozen `review_arguments` to contain exactly one literal `--reviewer-loop=agent-loop`; zero loops, duplicates, split-form values or `single-shot` are `_CurrentAdapterError`. This also protects a supplied `--agent-config` that bypasses CLI construction.

- [ ] **Step 2: Freeze the expected Profile into `AgentConfigSnapshot.parameters`**

Compute the Profile with the product helper from Task 6. Use an absolute non-created path only to satisfy the product Memory parser; the projection removes only that path:

```python
profile = review_execution_profile_from_arguments(
    review_arguments,
    memory_mode=args.memory_mode,
    memory_root=(Path.cwd() / ".review-agent" / "eval-profile-memory").resolve(),
)
```

Then persist:

```python
parameters = {
    "adapter": adapter,
    "clarification": clarification_policy,
    "agent_execution_profile": {
        "profile": profile.to_dict(),
        "digest": profile.digest(),
    },
}
```

`prompt_config_digest` remains the canonical digest of all parameters, so loop, tools, prompt, model, budget and Intent policy all enter Run identity.

- [ ] **Step 3: Derive the Eval outer timeout from the product Profile**

Change `prepare --agent-timeout-seconds` to default to `None`. For the current product adapter, omission means “use the frozen Profile floor,” not a second Eval default. Resolve the timeout only after loading the Agent snapshot:

```python
minimum_timeout = float(
    profile.payload["minimum_outer_timeout_seconds"]
)
if args.agent_timeout_seconds is None:
    agent_timeout_seconds = minimum_timeout
elif args.agent_timeout_seconds < minimum_timeout:
    raise CliPreconditionError(
        "agent_timeout_seconds is below the product execution profile minimum"
    )
else:
    agent_timeout_seconds = args.agent_timeout_seconds
```

Pass the resolved value into `ResourceBudgets`. Eval may use a larger infrastructure timeout, but may not terminate a legal product execution sooner. For a supplied `--run-config`, hydrate its Agent Profile and perform the same lower-bound check before accepting the frozen config. Non-product subprocess adapters retain the existing Harness default and do not claim product equivalence.

Add CLI tests showing: an omitted current-Agent timeout equals the Profile floor; an explicit larger value is preserved; and a too-small CLI or supplied Run-config value fails during prepare before any Run/Trial artifact is created.

- [ ] **Step 4: Add exact single-Case selection for smoke Runs**

Add repeatable `prepare --task-id`. `_load_run_config_for_prepare` must use:

```python
task_ids = None if not args.task_id else tuple(args.task_id)
snapshot = bank.snapshot(task_ids=task_ids)
```

Reject duplicate IDs and reject `--task-id` when a supplied `--run-config` contains a different Suite projection. This selects immutable Cases; it does not rewrite AACR/SWE data.

- [ ] **Step 5: Validate the frozen binding and every actual persisted Session**

Extend `_CurrentAdapterConfiguration` with `execution_profile` and `execution_profile_digest`. In `_configuration`, require an exact binding instead of trusting arbitrary `AgentConfigSnapshot.parameters` content:

```python
profile_binding = snapshot.parameters.get("agent_execution_profile")
if not isinstance(profile_binding, Mapping) or set(profile_binding) != {
    "profile",
    "digest",
}:
    raise _CurrentAdapterError(
        "current Agent execution profile binding is invalid"
    )
try:
    execution_profile = AgentExecutionProfile.from_dict(
        profile_binding["profile"]
    )
except (TypeError, ValueError) as exc:
    raise _CurrentAdapterError(
        "current Agent execution profile is invalid"
    ) from exc
execution_profile_digest = profile_binding["digest"]
if (
    type(execution_profile_digest) is not str
    or execution_profile.digest() != execution_profile_digest
):
    raise _CurrentAdapterError(
        "current Agent execution profile digest is invalid"
    )
```

Return those canonical values in `_CurrentAdapterConfiguration`. Immediately after every `_load_session` call, rebuild the product projection from the actual persisted `manifest.execution` and compare both bytes and digest:

```python
actual_profile = AgentExecutionProfile.from_execution(manifest.execution)
if (
    actual_profile.digest() != adapter.execution_profile_digest
    or actual_profile.to_dict() != adapter.execution_profile.to_dict()
):
    raise AgentAdapterIncompatibleError(
        AdapterIncompatibilityReason.EXECUTION_PROFILE_MISMATCH
    )
```

Add `EXECUTION_PROFILE_MISMATCH` to `AdapterIncompatibilityReason`. Runner already marks a dynamic incompatibility `INCOMPLETE`; it does not publish an Agent failure or capability score.

- [ ] **Step 6: Add CLI/Adapter regressions**

Assert generated argv contains exactly one `--reviewer-loop=agent-loop`, and a fake Session persisted with `reviewer_loop="single-shot"` produces `adapter_incompatible.execution_profile_mismatch`. Also assert the expected Profile says shell/network/run-safe-check are unavailable and a timeout below the Profile floor is rejected before Run creation.

- [ ] **Step 7: Run CLI/Profile/Adapter tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_cli.py tests/eval/test_current_agent_adapter.py tests/test_execution_profile.py -q -p no:cacheprovider --basetemp '.eval-data/pytest-public-equivalence-agent-loop'
```

Expected: explicit agent-loop is in frozen argv and Session drift is unscored incompatibility.

- [ ] **Step 8: Commit**

```powershell
git add src/review_agent_eval/cli.py src/review_agent_eval/adapters/base.py src/review_agent_eval/adapters/current_agent.py tests/eval/test_cli.py tests/eval/test_current_agent_adapter.py tests/test_execution_profile.py
git commit -m "feat(eval): bind public runs to product agent loop"
```

## Task 8: Make `completion.json` authoritative for Submission status

**Files:**

- Modify: `src/review_agent_eval/adapters/current_agent.py:1273-1345`
- Test: `tests/eval/test_current_agent_adapter.py:1266-1438`

- [ ] **Step 1: Load and hydrate the registered Completion artifact first**

Import `completion_from_dict` and load:

```python
completion_payload = _load_registered_json(
    run_dir=run_dir,
    store=store,
    manifest=manifest,
    name="completion",
    expected_path="completion.json",
    expected_phase=RunPhase.COMPLETION,
    expected_revision=revision,
)
completion = completion_from_dict(completion_payload)
```

Do this before constructing a completed Submission. A missing, unregistered, tampered or malformed artifact must raise `_CurrentArtifactError`, which maps to `invalid_output / schema_mismatch`. After hydrating Completion, load/validate `review_brief.json` only far enough to construct the auditable Intent; do not load Finding observations or Evidence until the Completion status has passed the mapping below.

- [ ] **Step 2: Map the four product terminal states**

Use one exact mapping:

```python
if completion.status in {"blocked", "budget_exhausted"}:
    return failure_submission(
        eval_input=eval_input,
        config=config,
        target_materialization_id=target_materialization_id,
        code=FailureCode.AGENT_BLOCKED,
        message=(
            "Current Agent completion was " + completion.status
        ),
        retryable=False,
        intent=intent,
        review=None,
        evidence=(),
        usage=empty_usage(elapsed_seconds=elapsed),
        trace_ref=trace_ref,
    )

if completion.status not in {
    "completed",
    "completed_with_uncertainties",
}:
    raise _CurrentArtifactError("current Agent completion status is invalid")
```

Only after the second branch accepts a completed status should the Adapter call `_findings_from_brief`, `_load_final_observations` and `_evidence_from_observations`. Thus a legitimate blocked Completion does not become `schema_mismatch` merely because no completed-review Evidence projection exists.

For `completed_with_uncertainties`, merge `brief.uncertainties`, `completion.uncertainties` and `completion.missing_perspectives` with this local stable helper:

```python
def _stable_unique_text(*groups: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for value in group:
            if value not in seen:
                seen.add(value)
                result.append(value)
    return tuple(result)
```

Use the result as `SubmissionReview.uncertainties`. `blocked` and `budget_exhausted` carry Intent for audit but no Review/Finding denominator.

- [ ] **Step 3: Add a four-state parametrized test**

The expected pairs are:

```python
(
    ("completed", SubmissionStatus.COMPLETED, None),
    ("completed_with_uncertainties", SubmissionStatus.COMPLETED, None),
    ("blocked", SubmissionStatus.BLOCKED, FailureCode.AGENT_BLOCKED),
    ("budget_exhausted", SubmissionStatus.BLOCKED, FailureCode.AGENT_BLOCKED),
)
```

Add separate missing, malformed, unregistered and wrong-revision Completion tests; each must produce `SubmissionStatus.INVALID_OUTPUT` with `FailureCode.SCHEMA_MISMATCH`.

Add one regression where semantic reconciliation is `local_only` but Completion is `completed_with_uncertainties`; the Submission must remain completed. Eval must not inspect an empty Candidate Catalog or Reconciler mode to override the authoritative Completion, and it must not create a separate Reconciler capability score.

- [ ] **Step 4: Run Completion adapter tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_current_agent_adapter.py -k 'completion or completed_artifact or blocked_submission' -q -p no:cacheprovider --basetemp '.eval-data/pytest-public-equivalence-completion'
```

Expected: lifecycle `SessionManifest.status=completed` no longer forces a completed Submission.

- [ ] **Step 5: Commit**

```powershell
git add src/review_agent_eval/adapters/current_agent.py tests/eval/test_current_agent_adapter.py
git commit -m "fix(eval): map submissions from authoritative completion"
```

## Task 9: Preserve failure-as-miss and exclude infrastructure failures from capability scoring

**Files:**

- Modify: `src/review_agent_eval/orchestrator.py:312-381`
- Modify: `src/review_agent_eval/orchestrator.py:580-650`
- Test: `tests/eval/test_metrics.py:712-786`
- Test: `tests/eval/test_orchestrator_target_replay_v2.py`

- [ ] **Step 1: Add blocked coverage to the existing metric regression**

Parameterize the current failed-Trial test over `SubmissionStatus.FAILED` and `SubmissionStatus.BLOCKED`/`FailureCode.AGENT_BLOCKED`. For both, assert:

```python
assert aggregate.metric(CoreMetric.AGENT_FAILURE_RATE).numerator == 1
assert aggregate.metric(CoreMetric.ISSUE_RECALL).coverage.failure_as_miss_count == 1
assert aggregate.metric(CoreMetric.ISSUE_PRECISION).coverage.failure_excluded_count == 1
```

Do not alter `TrialScorer` ratio logic.

- [ ] **Step 2: Refuse to score Harness materialization failures**

In the public `evaluate_trial` method, immediately after `load_existing_submission` and before `_case`, `_evaluation_source` or any Intent/Review/Judge construction, add:

```python
if self._is_pre_materialization_failure(submission):
    raise EvaluationPreconditionError(
        "Run contains a Harness repository materialization failure; "
        "capability scoring is invalid"
    )
```

Also preflight every selected Submission in `evaluate_run` after terminal-state validation and before the first `evaluate_trial` call:

```python
selected_plans = tuple(
    plan
    for plan in manifest.trials
    if selected is None or plan.task_id in selected
)
for plan in selected_plans:
    submission = self.artifact_store.load_existing_submission(
        run_id,
        plan.task_id,
        plan.trial_id,
    )
    if self._is_pre_materialization_failure(submission):
        raise EvaluationPreconditionError(
            "Run contains a Harness repository materialization failure; "
            "capability scoring is invalid"
        )
```

Only after the full preflight passes may the loop create a Judge or write Trial evaluation artifacts. This makes the whole selected evaluation unpublishable without partially spending Judge quota. Keep the immutable failure Submission and diagnostic; do not convert it to `AGENT_BLOCKED`, delete the Case or emit Recall=0.

- [ ] **Step 3: Add an infrastructure separation regression**

Build a Trial with `FailureCode.HARNESS_MATERIALIZATION_ERROR`, call `EvaluationOrchestrator.evaluate_trial`, and assert `EvaluationPreconditionError`. Then build a two-Trial Run with one normal terminal Submission followed by one infrastructure failure and call `evaluate_run`; assert the Judge factory was never called and neither Trial received a score/Judge/report namespace.

- [ ] **Step 4: Run metrics/orchestrator tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_metrics.py tests/eval/test_orchestrator_target_replay_v2.py -q -p no:cacheprovider --basetemp '.eval-data/pytest-public-equivalence-scoring'
```

Expected: Agent failure remains a measurable miss; infrastructure failure cannot enter aggregation.

- [ ] **Step 5: Commit**

```powershell
git add src/review_agent_eval/orchestrator.py tests/eval/test_metrics.py tests/eval/test_orchestrator_target_replay_v2.py
git commit -m "fix(eval): separate agent and infrastructure failures"
```

## Task 10: Run the focused local regression gate

**Files:**

- Verify only; no code changes expected.

- [ ] **Step 1: Run all changed boundaries together from the D: worktree**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_repository.py tests/test_tool_gateway.py tests/test_intent.py tests/test_intent_clarification.py tests/eval/test_models.py tests/eval/test_clarification_script.py tests/eval/test_intent_evaluator.py tests/eval/test_runner.py tests/test_execution_profile.py tests/test_session.py tests/test_context.py tests/eval/test_cli.py tests/eval/test_current_agent_adapter.py tests/eval/test_metrics.py tests/eval/test_orchestrator_target_replay_v2.py -q -p no:cacheprovider --basetemp '.eval-data/pytest-public-equivalence-focused-gate'
```

Expected: all selected tests pass. Pytest temporary state stays under `D:\Agent\code review agent\.worktrees\public-intent-continuation\.eval-data`, not `C:\tmp`.

- [ ] **Step 2: Verify forbidden capabilities are absent**

Run:

```powershell
rg -n 'run_safe_check|"bash"|shell.*available|network.*unrestricted' src/review_agent/execution_profile.py src/review_agent/context.py src/review_agent_eval/adapters/current_agent.py
```

Expected: only explicit `unavailable`/`provider_only` declarations or test assertions appear; no Bash/network tool definition exists.

- [ ] **Step 3: Verify the worktree diff is scoped**

Run:

```powershell
git status --short
git diff --stat 0cf72a8..HEAD
```

`0cf72a8` is the approved design baseline for this implementation. Expected: the implementation delta contains this plan plus only files listed here; no `.eval-data`, benchmark Case, Ground Truth or prior Run artifact is staged. The older branch delta against `origin/main` is audited separately and is not falsely attributed to this implementation batch.

## Task 11: Prepare all ten AACR repositories without model calls

**Files:**

- Modify locally but do not stage: `.eval-data/live-public-20260730/configs/prepare_attested_public_run.py`.
- Operational artifacts only under ignored `.eval-data/live-public-20260730/product-equivalence-v1`.

- [ ] **Step 1: Make the ignored driver use Profile-derived timeout and report closure identity**

Remove the driver’s hard-coded pair:

```python
"--agent-timeout-seconds",
"3600",
```

After Task 7, omitting this option makes `prepare` persist the product Profile’s computed outer floor. This is required because the default `single` Reviewer mode executes the complete Portfolio sequentially and can legally exceed 3,600 seconds at Critical risk.

Extend only the existing `repository-cache` JSON record in `_prepare_cache`; do not change acquisition behavior:

```python
{
    "stage": "repository-cache",
    "repository_descriptor_digest": repository.digest(),
    "cache_id": prepared.cache_id,
    "source_digest": prepared.manifest.source_digest,
    "manifest_schema_version": prepared.manifest.schema_version,
    "logical_source_version": prepared.manifest.logical_source_version,
    "object_count_policy": prepared.manifest.budget_policy[
        "object_count_policy"
    ],
    "actual_objects": prepared.manifest.budget_policy["actual_objects"],
}
```

This local-only output is the suite-wide prepare receipt used below. It contains no repository content or credential.

- [ ] **Step 2: Use one fresh AACR execution root and v2 binding output**

From the feature worktree, run the existing attested public preparation driver with the new code:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
$agentCommit = git rev-parse HEAD
$aacrRoot = '.eval-data\live-public-20260730\product-equivalence-v1\aacr'
$aacrPrepareOutput = & 'D:\Anaconda\envs\MINIST\python.exe' '.eval-data\live-public-20260730\configs\prepare_attested_public_run.py' --suite-root '.eval-data\live-public-20260730\aacr-csharp' --execution-root $aacrRoot --run-instance-key 'aacr-review-closure-v2-prepare' --agent-commit $agentCommit --binding-output '.eval-data\live-public-20260730\product-equivalence-v1\aacr-bindings-v2.json' --proxy 'http://127.0.0.1:7897' --agent-python 'D:\Anaconda\envs\MINIST\python.exe' --agent-source-root 'src'
if ($LASTEXITCODE -ne 0) {
    throw "AACR repository preparation failed with exit code $LASTEXITCODE"
}
```

This stage fetches/attests repositories and creates caches/Run plans; it does not invoke DeepSeek.

- [ ] **Step 3: Check suite-wide preparation, not a filtered success subset**

Parse every driver line and require all ten descriptors:

```powershell
$aacrPrepareRecords = @(
    $aacrPrepareOutput | ForEach-Object { $_ | ConvertFrom-Json }
)
$aacrCaches = @(
    $aacrPrepareRecords | Where-Object { $_.stage -eq 'repository-cache' }
)
if ($aacrCaches.Count -ne 10) {
    throw "Expected 10 AACR repository caches, got $($aacrCaches.Count)"
}
foreach ($record in $aacrCaches) {
    if (
        $record.manifest_schema_version -ne 'prepared_repository_manifest_v2' -or
        $record.logical_source_version -ne 'logical_git_review_closure_v2' -or
        $record.object_count_policy -ne 'observed_only' -or
        [int64]$record.actual_objects -le 0
    ) {
        throw "AACR cache does not use the v2 Review Closure"
    }
}
$aacrPrepareResult = $aacrPrepareRecords | Select-Object -Last 1
if (
    [string]::IsNullOrWhiteSpace($aacrPrepareResult.run_id) -or
    [int]$aacrPrepareResult.case_count -ne 10
) {
    throw "AACR full-suite prepare did not bind all 10 Cases"
}
```

No Case may fail because of a `100,000` or fixed object-count message. Any byte/node/disk/time failure makes this command fail as infrastructure; it is not converted to Recall=0 or removed from the suite.

If any Case fails, stop here and diagnose that preparation failure before spending model quota.

## Task 12: Run one AACR and one SWE product-model smoke

**Files:**

- Modify locally but do not stage: `.eval-data/live-public-20260730/configs/prepare_attested_public_run.py`.
- Operational artifacts only under ignored `.eval-data/live-public-20260730/product-equivalence-v1`.

- [ ] **Step 1: Create a one-Case AACR Run using immutable task selection**

The existing ignored attestation driver predates `prepare --task-id`. Add `parser.add_argument("--task-id", action="append", default=[])`, move its current inline prepare argv into `prepare_cli_argv`, and append:

```python
for task_id in args.task_id:
    prepare_cli_argv.extend(("--task-id", task_id))
prepare_args = eval_cli._build_parser().parse_args(prepare_cli_argv)
```

Then use this existing Case ID with `prepare --task-id`:

```text
aacr-pr-05c538297b1ae7555119645c0398e0889b11d7dd300dfa1d7f5b20de10fdd36a
```

The driver may still attest/cache the full source set, but the Run Manifest must contain exactly this Trial. Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
$agentCommit = git rev-parse HEAD
$aacrRoot = '.eval-data\live-public-20260730\product-equivalence-v1\aacr'
$aacrTaskId = 'aacr-pr-05c538297b1ae7555119645c0398e0889b11d7dd300dfa1d7f5b20de10fdd36a'
$aacrOutput = & 'D:\Anaconda\envs\MINIST\python.exe' '.eval-data\live-public-20260730\configs\prepare_attested_public_run.py' --suite-root '.eval-data\live-public-20260730\aacr-csharp' --execution-root $aacrRoot --run-instance-key 'aacr-agent-loop-smoke-v1' --task-id $aacrTaskId --agent-commit $agentCommit --binding-output '.eval-data\live-public-20260730\product-equivalence-v1\aacr-bindings-v2.json' --proxy 'http://127.0.0.1:7897' --agent-python 'D:\Anaconda\envs\MINIST\python.exe' --agent-source-root 'src'
if ($LASTEXITCODE -ne 0) {
    throw "AACR smoke prepare failed with exit code $LASTEXITCODE"
}
$aacrPrepare = ($aacrOutput | Select-Object -Last 1) | ConvertFrom-Json
$aacrRunId = $aacrPrepare.run_id
if (
    [string]::IsNullOrWhiteSpace($aacrRunId) -or
    [int]$aacrPrepare.case_count -ne 1 -or
    [int]$aacrPrepare.trial_count -ne 1
) {
    throw 'AACR smoke Run must contain exactly one Trial'
}
```

The shared AACR data root reuses the v2 content-addressed caches verified in Task 11; the new immutable Run ID is distinct because its task selection and run-instance key differ.

- [ ] **Step 2: Prepare the existing one-Case SWE suite**

Run:

```powershell
$sweRoot = '.eval-data\live-public-20260730\product-equivalence-v1\swe'
$sweTaskId = 'webpack-bundle-analyzer__698'
$sweOutput = & 'D:\Anaconda\envs\MINIST\python.exe' '.eval-data\live-public-20260730\configs\prepare_attested_public_run.py' --suite-root '.eval-data\live-public-20260730\swe-webpack-698' --execution-root $sweRoot --run-instance-key 'swe-agent-loop-smoke-v1' --task-id $sweTaskId --agent-commit $agentCommit --binding-output '.eval-data\live-public-20260730\product-equivalence-v1\swe-bindings-v2.json' --proxy 'http://127.0.0.1:7897' --agent-python 'D:\Anaconda\envs\MINIST\python.exe' --agent-source-root 'src'
if ($LASTEXITCODE -ne 0) {
    throw "SWE smoke prepare failed with exit code $LASTEXITCODE"
}
$swePrepare = ($sweOutput | Select-Object -Last 1) | ConvertFrom-Json
$sweRunId = $swePrepare.run_id
if (
    [string]::IsNullOrWhiteSpace($sweRunId) -or
    [int]$swePrepare.case_count -ne 1 -or
    [int]$swePrepare.trial_count -ne 1
) {
    throw 'SWE smoke Run must contain exactly one Trial'
}
```

Use the same DeepSeek provider identity and `REVIEW_AGENT_API_KEY`; do not add Reviewer network or shell tools.

- [ ] **Step 3: Run the Agent stage for both one-Case Runs**

Invoke `review-agent-eval run-agent` with `--max-workers 1` and the exact `run_id` emitted by each prepare:

```powershell
if ([string]::IsNullOrWhiteSpace($env:REVIEW_AGENT_API_KEY)) {
    throw 'REVIEW_AGENT_API_KEY is not configured'
}
$aacrAgentOutput = & 'D:\Anaconda\envs\MINIST\python.exe' -m review_agent_eval run-agent --suite-root '.eval-data\live-public-20260730\aacr-csharp' --runs-root "$aacrRoot\.eval-runs" --data-root "$aacrRoot\.eval-data" --workspace-root "$aacrRoot\.eval-workspaces" --run-id $aacrRunId --max-workers 1 --json
if ($LASTEXITCODE -ne 0) {
    throw "AACR Agent smoke failed with exit code $LASTEXITCODE"
}
$aacrAgent = ($aacrAgentOutput | Select-Object -Last 1) | ConvertFrom-Json
$sweAgentOutput = & 'D:\Anaconda\envs\MINIST\python.exe' -m review_agent_eval run-agent --suite-root '.eval-data\live-public-20260730\swe-webpack-698' --runs-root "$sweRoot\.eval-runs" --data-root "$sweRoot\.eval-data" --workspace-root "$sweRoot\.eval-workspaces" --run-id $sweRunId --max-workers 1 --json
if ($LASTEXITCODE -ne 0) {
    throw "SWE Agent smoke failed with exit code $LASTEXITCODE"
}
$sweAgent = ($sweAgentOutput | Select-Object -Last 1) | ConvertFrom-Json
if (@($aacrAgent.trials).Count -ne 1 -or @($sweAgent.trials).Count -ne 1) {
    throw 'Each smoke Run must have exactly one terminal Trial'
}
$smokeStatuses = @('completed', 'blocked')
if (
    $aacrAgent.trials[0].status -notin $smokeStatuses -or
    $sweAgent.trials[0].status -notin $smokeStatuses -or
    $aacrAgent.trials[0].submission_status -notin $smokeStatuses -or
    $sweAgent.trials[0].submission_status -notin $smokeStatuses
) {
    throw 'Smoke Run did not produce a valid completed/blocked product terminal'
}
if (
    $aacrAgent.trials[0].submission_status -ne 'completed' -and
    $sweAgent.trials[0].submission_status -ne 'completed'
) {
    throw 'At least one smoke must complete to validate Finding/Evidence replay'
}
$aacrTrialId = $aacrAgent.trials[0].trial_id
$sweTrialId = $sweAgent.trials[0].trial_id
```

Expected for each Trial:

- persisted Session Profile digest equals the frozen Agent Profile digest;
- `reviewer_loop` is `agent-loop`;
- Trace contains model turns and tool calls, rather than parsing a tool call as final Reviewer JSON;
- public inferred Intent is finally stored as `source=explicit`, `origin=benchmark_auto_accept`;
- registered `completion.json` and Eval Submission status agree;
- a real product blocker appears as `blocked / agent_blocked`, not a false completed Submission;
- no Bash, Reviewer HTTP/browser or `run_safe_check` call exists.

- [ ] **Step 4: Evaluate and inspect both Trials**

Run the Judge stage:

```powershell
$aacrEvaluationOutput = & 'D:\Anaconda\envs\MINIST\python.exe' -m review_agent_eval evaluate --suite-root '.eval-data\live-public-20260730\aacr-csharp' --runs-root "$aacrRoot\.eval-runs" --data-root "$aacrRoot\.eval-data" --workspace-root "$aacrRoot\.eval-workspaces" --run-id $aacrRunId --revision 'product-equivalence-smoke-v1' --judge-provider 'openai-compatible' --judge-model 'deepseek-v4-pro' --judge-base-url 'https://api.deepseek.com' --judge-api-key-env 'REVIEW_AGENT_API_KEY' --json
if ($LASTEXITCODE -ne 0) {
    throw "AACR smoke evaluation failed with exit code $LASTEXITCODE"
}
$aacrEvaluation = ($aacrEvaluationOutput | Select-Object -Last 1) | ConvertFrom-Json
$sweEvaluationOutput = & 'D:\Anaconda\envs\MINIST\python.exe' -m review_agent_eval evaluate --suite-root '.eval-data\live-public-20260730\swe-webpack-698' --runs-root "$sweRoot\.eval-runs" --data-root "$sweRoot\.eval-data" --workspace-root "$sweRoot\.eval-workspaces" --run-id $sweRunId --revision 'product-equivalence-smoke-v1' --judge-provider 'openai-compatible' --judge-model 'deepseek-v4-pro' --judge-base-url 'https://api.deepseek.com' --judge-api-key-env 'REVIEW_AGENT_API_KEY' --json
if ($LASTEXITCODE -ne 0) {
    throw "SWE smoke evaluation failed with exit code $LASTEXITCODE"
}
$sweEvaluation = ($sweEvaluationOutput | Select-Object -Last 1) | ConvertFrom-Json
$aacrEvaluationId = if ($null -ne $aacrEvaluation.namespace) {
    $aacrEvaluation.namespace.evaluation_id
} else {
    $aacrEvaluation.evaluation_id
}
$sweEvaluationId = if ($null -ne $sweEvaluation.namespace) {
    $sweEvaluation.namespace.evaluation_id
} else {
    $sweEvaluation.evaluation_id
}
if (
    [string]::IsNullOrWhiteSpace($aacrEvaluationId) -or
    [string]::IsNullOrWhiteSpace($sweEvaluationId)
) {
    throw 'Smoke evaluation identity is missing'
}
```

Inspect the sole Trial in each Run without invoking Agent or Judge again:

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m review_agent_eval inspect --suite-root '.eval-data\live-public-20260730\aacr-csharp' --runs-root "$aacrRoot\.eval-runs" --data-root "$aacrRoot\.eval-data" --workspace-root "$aacrRoot\.eval-workspaces" --run-id $aacrRunId --task-id $aacrTaskId --trial-id $aacrTrialId --evaluation-id $aacrEvaluationId --format markdown
& 'D:\Anaconda\envs\MINIST\python.exe' -m review_agent_eval inspect --suite-root '.eval-data\live-public-20260730\swe-webpack-698' --runs-root "$sweRoot\.eval-runs" --data-root "$sweRoot\.eval-data" --workspace-root "$sweRoot\.eval-workspaces" --run-id $sweRunId --task-id $sweTaskId --trial-id $sweTrialId --evaluation-id $sweEvaluationId --format markdown
```

Confirm the report and immutable Submission distinguish:

1. completed review with zero matched Findings;
2. completed review with matched/missed Findings;
3. Agent blocked/failed with `failure_as_miss` and non-zero Failure Rate.

Infrastructure failure must abort evaluation instead of producing the third state.

If either smoke has protocol/profile/artifact failure, stop before the ten-Case baseline.

## Task 13: Run the first trustworthy ten-Case AACR baseline

**Files:**

- Operational artifacts only under the shared ignored AACR root; no source commit.

- [ ] **Step 1: Create a fresh full-suite Run after both smoke Runs pass**

Use the already verified ten-Case suite, v2 repository caches, the same model/provider/Judge identity, and run-instance key `aacr-agent-loop-baseline-v1`. Do not reuse either invalid historical Run:

```text
run-735b6d6948646b3312df09daf46a95c9753cc2aed24b81a80a56853af816a224
run-6204c8a9723f48bb95f361c0edb5d4e5593fe65557109a8f32a28ad0fdf5e2c1
```

Create the full-suite Run in the same content-addressed AACR data root used by Task 11:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
$agentCommit = git rev-parse HEAD
$aacrRoot = '.eval-data\live-public-20260730\product-equivalence-v1\aacr'
$baselinePrepareOutput = & 'D:\Anaconda\envs\MINIST\python.exe' '.eval-data\live-public-20260730\configs\prepare_attested_public_run.py' --suite-root '.eval-data\live-public-20260730\aacr-csharp' --execution-root $aacrRoot --run-instance-key 'aacr-agent-loop-baseline-v1' --agent-commit $agentCommit --binding-output '.eval-data\live-public-20260730\product-equivalence-v1\aacr-bindings-v2.json' --proxy 'http://127.0.0.1:7897' --agent-python 'D:\Anaconda\envs\MINIST\python.exe' --agent-source-root 'src'
if ($LASTEXITCODE -ne 0) {
    throw "AACR baseline prepare failed with exit code $LASTEXITCODE"
}
$baselinePrepare = ($baselinePrepareOutput | Select-Object -Last 1) | ConvertFrom-Json
$baselineRunId = $baselinePrepare.run_id
if (
    [string]::IsNullOrWhiteSpace($baselineRunId) -or
    [int]$baselinePrepare.case_count -ne 10 -or
    [int]$baselinePrepare.trial_count -ne 1
) {
    throw 'AACR baseline Run must bind 10 Cases and 10 total Trials'
}
```

- [ ] **Step 2: Run all ten Agent Trials with deterministic concurrency**

Run with `--max-workers 1` first to keep provider behavior and failure diagnosis deterministic:

```powershell
if ([string]::IsNullOrWhiteSpace($env:REVIEW_AGENT_API_KEY)) {
    throw 'REVIEW_AGENT_API_KEY is not configured'
}
$baselineAgentOutput = & 'D:\Anaconda\envs\MINIST\python.exe' -m review_agent_eval run-agent --suite-root '.eval-data\live-public-20260730\aacr-csharp' --runs-root "$aacrRoot\.eval-runs" --data-root "$aacrRoot\.eval-data" --workspace-root "$aacrRoot\.eval-workspaces" --run-id $baselineRunId --max-workers 1 --json
if ($LASTEXITCODE -ne 0) {
    throw "AACR baseline Agent stage failed with exit code $LASTEXITCODE"
}
$baselineAgent = ($baselineAgentOutput | Select-Object -Last 1) | ConvertFrom-Json
if (@($baselineAgent.trials).Count -ne 10) {
    throw "Expected 10 AACR Agent Trials, got $(@($baselineAgent.trials).Count)"
}
$terminalStatuses = @('completed', 'failed', 'blocked', 'invalid_output')
$nonterminal = @(
    $baselineAgent.trials |
        Where-Object { $_.status -notin $terminalStatuses }
)
if ($nonterminal.Count -ne 0) {
    throw 'AACR baseline contains a nonterminal or incompatible Trial'
}
```

An execution-profile mismatch is an incompatible/nonterminal Trial and must stop this baseline before Judge spending. Ordinary product `blocked`/`failed` terminal Submissions remain valid capability outcomes.

- [ ] **Step 3: Evaluate the complete Run and verify aggregate coverage**

```powershell
$baselineEvaluationOutput = & 'D:\Anaconda\envs\MINIST\python.exe' -m review_agent_eval evaluate --suite-root '.eval-data\live-public-20260730\aacr-csharp' --runs-root "$aacrRoot\.eval-runs" --data-root "$aacrRoot\.eval-data" --workspace-root "$aacrRoot\.eval-workspaces" --run-id $baselineRunId --revision 'product-equivalence-baseline-v1' --judge-provider 'openai-compatible' --judge-model 'deepseek-v4-pro' --judge-base-url 'https://api.deepseek.com' --judge-api-key-env 'REVIEW_AGENT_API_KEY' --json
if ($LASTEXITCODE -ne 0) {
    throw "AACR baseline evaluation failed with exit code $LASTEXITCODE"
}
$baselineEvaluation = (
    $baselineEvaluationOutput | Select-Object -Last 1
) | ConvertFrom-Json
$baselineEvaluationId = if ($null -ne $baselineEvaluation.namespace) {
    $baselineEvaluation.namespace.evaluation_id
} else {
    $baselineEvaluation.evaluation_id
}
$summary = $baselineEvaluation.summary
$coverage = $summary.coverage
if (
    [int]$coverage.planned_case_count -ne 10 -or
    [int]$coverage.planned_trial_count -ne 10 -or
    [int]$coverage.terminal_submission_count -ne 10 -or
    [int]$coverage.trial_score_count -ne 10 -or
    @($coverage.nonterminal_trial_ids).Count -ne 0 -or
    @($coverage.unevaluated_terminal_trial_ids).Count -ne 0
) {
    throw 'AACR baseline does not have complete 10/10 terminal coverage'
}
if (@(
    $coverage.failure_code_breakdown |
        Where-Object { $_.key -eq 'harness_materialization_error' }
).Count -ne 0) {
    throw 'Infrastructure failure cannot be published as Agent capability'
}
$metricNames = @(
    $summary.partitions[0].aggregate_score.metrics |
        ForEach-Object { $_.metric }
)
foreach ($requiredMetric in @(
    'agent_failure_rate',
    'issue_precision',
    'issue_recall',
    'evidence_validity',
    'evidence_support_rate',
    'judge_failure_rate',
    'judge_ungraded_rate'
)) {
    if ($requiredMetric -notin $metricNames) {
        throw "AACR baseline is missing metric $requiredMetric"
    }
}
if ($null -eq $summary.partitions[0].aggregate_score.usage) {
    throw 'AACR baseline is missing token/tool/time usage'
}
```

The values may legitimately be null where AACR lacks authority, but their coverage/null reason must be present. A successful completed review with no matches, a completed review with matches/misses, and an Agent failure remain distinguishable through Submission status, Failure Rate and `failure_as_miss`.

- [ ] **Step 4: Inspect every Trial and record the immutable baseline identity**

```powershell
foreach ($trial in $baselineAgent.trials) {
    & 'D:\Anaconda\envs\MINIST\python.exe' -m review_agent_eval inspect --suite-root '.eval-data\live-public-20260730\aacr-csharp' --runs-root "$aacrRoot\.eval-runs" --data-root "$aacrRoot\.eval-data" --workspace-root "$aacrRoot\.eval-workspaces" --run-id $baselineRunId --task-id $trial.task_id --trial-id $trial.trial_id --evaluation-id $baselineEvaluationId --format markdown
    if ($LASTEXITCODE -ne 0) {
        throw "Inspection failed for $($trial.task_id)"
    }
}
```

Keep prior invalid Runs immutable. Record the new Run ID, evaluation ID, Agent Profile digest, repository closure version, Intent policy version and the two smoke Run IDs in the final handoff/PR description.

## Final verification checklist

- [ ] `git diff --check` passes.
- [ ] Focused local regression gate in Task 10 passes from the D: worktree.
- [ ] All ten AACR repositories prepare or the entire validation explicitly fails as infrastructure.
- [ ] AACR and SWE one-Case smoke Runs use `agent-loop` and Completion authority.
- [ ] Full AACR baseline starts only after both smoke Runs pass.
- [ ] Core Regression Intent Truth and scripted clarification tests remain unchanged.
- [ ] No benchmark Case, Ground Truth, Matcher threshold, Bash, Reviewer network or `run_safe_check` change is present.

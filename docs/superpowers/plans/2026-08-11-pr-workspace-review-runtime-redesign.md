# PR Workspace 与 Reviewer Runtime 重构实施计划

> **For agentic workers:** 按 Task 顺序实施，使用 checkbox 跟踪进度；每个 Task 在定向测试通过后形成一个独立提交。不要把协议、存储、Agent Loop 和 Eval 切换压成一次大提交。

**状态：** 实施中（Task 1–5 已完成）

**设计来源：** `docs/superpowers/specs/2026-08-10-pr-workspace-preflight-reviewer-runtime-redesign.md`

**Goal:** 将当前以 Session phase、ChangeSummary、ObservationStore、Semantic Reconciliation 和 ReviewBrief 为中心的产品链路，迁移为 PRWorkspace/Snapshot、完整 DiffArtifact、精简 Intent/Risk/Assignment、可恢复 Reviewer Agent Loop、三层上下文压缩以及确定性 ReviewResult。

**Architecture:** 采用 strangler cutover。先增加不依赖旧 Pipeline 的新协议、PRWorkspace、DiffArtifact、Context Window、Journal 和 Aggregator；旧 v5 Pipeline 在这些模块建设期间继续通过回归。所有新边界独立通过测试后，引入 Session v6 五阶段 Pipeline，并让 CLI/Eval 一次切换到 v6。旧 v1-v5 Session 只保留只读诊断，不允许静默升级或继续执行旧后处理链。

**Tech Stack:** Python 3.11+ dataclasses、Git CLI、现有 OpenAI-compatible Model Adapter、pytest、SHA-256 内容寻址、UTF-8 canonical JSON、Windows/PowerShell。

---

## 1. 实施约束

1. 产品代码不得导入 `review_agent_eval`；Eval Adapter 可以只读产品公开协议。
2. 新路径不得双写旧 `ChangeSummary/Observation/ReviewBrief` 和新 PRWorkspace Artifact。切换前走旧路径，切换后只走新路径。
3. Session v6 是 breaking schema。v1-v5 可以读取和诊断，但不能自动 Resume 成 v6，也不能通过字段猜测伪装成 v6。
4. 单元和集成测试使用 Fake Adapter，不调用真实模型 API。DeepSeek/AACR/SWE smoke 必须单独取得用户授权。
5. 不修改 AACR/SWE Ground Truth、Judge 规则或基准数据来适配产品输出。
6. 不新增通用 Shell、网络、Edit 或 Write Reviewer 工具；Reviewer 继续只读。
7. 所有新持久化路径必须拒绝 traversal、symlink/junction、路径别名和越界 Artifact ID；发布使用 create-only 或原子替换语义。
8. 新模块保持小而专一。不得继续向已经过大的 `pipeline.py`、`session_store.py`、`context.py` 填入完整新子系统。
9. 每个 Task 先写 RED 测试，再实现最小 GREEN，再运行列出的邻接回归。
10. 每个提交前运行 `git diff --check`；不得提交 `.pytest_cache`、`__pycache__`、真实 API key 或临时评测目录。

## 2. 本计划固定的实施假设

- 本地和 Benchmark 调用新增稳定 `external_review_id`。Benchmark 使用 task ID；本地 CLI 未提供平台 PR 编号时要求显式 `--external-review-id`。
- `pr_id = stable_id(repository_identity, provider, pr_number_or_external_review_id)`；`session_id` 继续是一次执行身份，不能代替 PR 身份。
- Windows 默认工作区根由 `REVIEW_AGENT_WORKSPACE_ROOT` 或 CLI `--workspace-root` 注入；Eval 使用自己的短目录。平台默认值由 resolver 选择真实短路径，不创建 Junction。
- 小/大 PR 不使用固定文件数阈值：Pinned 内容组装后，如果完整 Diff 能进入初始 500K～600K Token 目标就内联，否则使用完整 Index 和 Artifact 读取。
- Assignment Target Selector 第一版使用确定性保守策略：Core/Adversarial 覆盖全部 changed files/hunks；Dynamic 使用 Planner perspective，但不能减少 Runtime 授权范围。以后可以优化相关性，不阻塞本次架构切换。
- Provider 提供 tokenizer 时使用精确计数；没有 tokenizer 时使用 UTF-8 byte upper bound，不能把字符数直接当 Token 数。
- 同一 Snapshot 已有通过 hash 验证的最终 ReviewResult 时直接复用。强制重新审查相同 Snapshot 不属于本次实现。
- DeveloperReviewPolicy 由产品代码/部署配置加载且没有用户 CLI override；默认策略可以为空，但其 digest 始终进入 Execution Profile。

## 3. 目标 Pipeline 与 Session v6

```text
PREFLIGHT
  DiffArtifact -> QualityGate -> ChangedSymbols

INTENT
  IntentAnalysisRecord -> IntentPacket(goal/source/uncertainties)

PLANNING
  deterministic risk floors + model level -> fixed slots -> Assignments

REVIEWERS
  ContextAssembly -> Agent Loop -> ReviewerOutput(findings/uncertainties)

AGGREGATION
  deterministic merge -> review-result.json -> optional review.md
```

v6 不再拥有以下 Phase：

```text
MEMORY_SELECTION
INTENT_DISCOVERY / INTENT_RESOLUTION
RECONCILIATION_ANALYSIS
SUPPLEMENTAL_INVESTIGATION
RECONCILIATION
COMPLETION
FINAL_RISK
MEMORY_PROPOSAL
REPORTING
```

Intent 内部分析、Global Memory 投影和 Markdown rendering 是阶段内部服务，不再各自占一个阻塞 Phase。

## 4. 文件地图

### 新增产品模块

- `src/review_agent/review_protocol.py`
  - 新 IntentPacket、RiskDecision、ReviewerSlot、ReviewerAssignment、ReviewerFinding、ReviewerOutput、FinalFinding、ReviewResult 及严格 wire codec。
- `src/review_agent/safe_io.py`
  - 统一 canonical JSON、create-only/atomic write、hash、路径和普通文件校验。
- `src/review_agent/pr_workspace.py`
  - PR/Snapshot identity、Workspace layout、manifest、Session/Artifact 授权和 Resume locator。
- `src/review_agent/diff_artifact.py`
  - 完整 `diff.patch` 生成、完整 index、按文件/hunk 分页读取和校验。
- `src/review_agent/preflight.py`
  - DiffArtifact、简单 QualityGate、ChangedSymbols 的确定性编排。
- `src/review_agent/local_quality.py`
  - v6 简单本地质量门；旧 `quality.py/quality_runner.py` 在切换前不做破坏性修改。
- `src/review_agent/intent_runtime.py`
  - Intent 内部分析到三字段 IntentPacket 的 v6 投影和版本化。
- `src/review_agent/risk_runtime.py`
  - deterministic floors、level-only model call 和最终 max；不复用旧多字段 Risk wire schema。
- `src/review_agent/review_planning.py`
  - 风险 floor、固定 Reviewer slots、Dynamic perspective 和 Assignment 生成。
- `src/review_agent/review_policy.py`
  - DeveloperReviewPolicy 与 user-level Global Memory/规则投影边界。
- `src/review_agent/global_memory.py`
  - 现有 Durable Memory 的精简全局规则/经验只读 facade；Reviewer 只能拿不可变投影，不能持有 live store。
- `src/review_agent/tool_artifacts.py`
  - 50K/200K Tool Result 分类、Artifact Store、2KB preview、分页读取和 index。
- `src/review_agent/review_tool_gateway.py`
  - v6 只读工具执行与统一 Tool Result envelope；不依赖 ObservationStore。
- `src/review_agent/execution_journal.py`
  - append-only Reviewer event journal、Tool Call 幂等账本、active elapsed 和 Resume 投影。
- `src/review_agent/context_window.py`
  - Token estimator、Pinned/动态分区、60 分钟清理、700K full compaction 和 Context Manifest。
- `src/review_agent/review_context.py`
  - v6 Reviewer System/tools/messages/parameters 与 Pinned Context 组装。
- `src/review_agent/review_agent_loop.py`
  - v6 unlimited Reviewer loop、Provider retry、Tool 调用、Journal 和 Compaction 集成。
- `src/review_agent/reviewer_output.py`
  - ReviewerOutput v2 Prompt/Parser 和候选级校验。
- `src/review_agent/reviewer_executor.py`
  - v6 单 Reviewer 执行隔离与 Runtime status。
- `src/review_agent/aggregation.py`
  - 确定性 Finding merge、状态、uncertainties、ReviewResult persistence。
- `src/review_agent/review_renderer.py`
  - ReviewResult 的 JSON/Markdown 纯渲染。
- `src/review_agent/review_pipeline.py`
  - Session v6 五阶段 Pipeline。

### 主要修改模块

- `src/review_agent/artifacts.py`
- `src/review_agent/git_repo.py`
- `src/review_agent/quality.py`
- `src/review_agent/quality_runner.py`
- `src/review_agent/repository_intelligence.py`
- `src/review_agent/intent.py`
- `src/review_agent/intent_inference.py`
- `src/review_agent/model_risk.py`
- `src/review_agent/context.py`
- `src/review_agent/tool_gateway.py`
- `src/review_agent/tool_result_protocol.py`
- `src/review_agent/model_adapter.py`
- `src/review_agent/agent_loop.py`
- `src/review_agent/reviewer_runtime.py`
- `src/review_agent/reviewer.py`
- `src/review_agent/reviewer_task_executor.py`
- `src/review_agent/session.py`
- `src/review_agent/session_store.py`
- `src/review_agent/run_state.py`
- `src/review_agent/resume.py`
- `src/review_agent/hydration.py`
- `src/review_agent/execution_profile.py`
- `src/review_agent/command.py`
- `src/review_agent_eval/adapters/current_agent.py`

### 旧主链模块

切换后新 Pipeline 不得导入：

- `src/review_agent/reconciler.py`
- `src/review_agent/supplemental.py`
- `src/review_agent/evidence.py`
- `src/review_agent/completion.py`
- `src/review_agent/final_risk.py`
- `src/review_agent/brief.py`
- 旧 `src/review_agent/reporting.py` 业务构造逻辑

这些文件只有在无剩余兼容读取依赖时才删除；否则移动到明确的 legacy-only 边界并由架构测试禁止新 Pipeline 导入。

## 5. 依赖顺序

```text
Task 1  Review Protocol
  -> Task 2  Safe I/O + PRWorkspace
  -> Task 3  DiffArtifact
  -> Task 4  Deterministic Preflight
  -> Task 5  Intent v2
  -> Task 6  Risk + Slots + Assignments
  -> Task 7  Reviewer Context + Rules
  -> Task 8  Tool Result Storage
  -> Task 9  Journal + Runtime
  -> Task 10 Context Window + Compaction
  -> Task 11 ReviewerOutput Integration
  -> Task 12 Aggregation + ReviewResult
  -> Task 13 Session v6 + Pipeline Cutover
  -> Task 14 CLI + Resume
  -> Task 15 Execution Profile + Eval Adapter
  -> Task 16 Legacy Disconnection
  -> Task 17 Full Regression + Authorized Smoke
```

Tasks 3～6 可以在独立分支并行开发，但合并和集成必须遵守上述顺序。Task 1～12 对旧 v5 共享模块的修改必须是 additive；破坏性删除统一留到 Task 16。

---

## Task 1：建立新 Review Protocol 与严格 Wire Codec

**Files:**

- Create: `src/review_agent/review_protocol.py`
- Modify: `src/review_agent/artifacts.py`
- Create: `tests/test_review_protocol_v2.py`
- Modify: `tests/test_architecture_boundaries.py`

- [x] **Step 1：为所有新 wire model 写 RED 测试**

覆盖：

- IntentPacket 严格只有 `goal/source/uncertainties`，并验证 explicit/inferred/null 不变量；
- RiskDecision 只有 `level`；
- ReviewerFinding 只有 `claim/severity/path/line/suggestion`；
- ReviewerOutput 顶层只有 `findings/uncertainties`；
- FinalFinding 只比 ReviewerFinding 多 `finding_id`；
- ReviewResult 顶层只有 `pr_id/snapshot_id/status/risk_level/findings/uncertainties`；
- ReviewRequest 保存 speaker-labeled 公开 conversation，不能混入 Intent/Risk 内部消息；
- unknown key、duplicate JSON key、空文本、非安全 path、非正 line、非法 enum 全部 fail closed；
- canonical serializer 固定 UTF-8、键顺序和 separators，不写时间字段。

- [x] **Step 2：实现不可变 dataclass、codec 和 canonical JSON**

不要复用旧 `models.py` 中同名但字段不同的模型。新代码只从 `review_protocol.py` 导入新协议；旧模型留给 v5 兼容读取。

- [x] **Step 3：注册新 Artifact schema version**

至少增加：

```text
pr_workspace_manifest_v1
snapshot_manifest_v1
diff_artifact_index_v1
preflight_result_v1
intent_packet_v2_minimal
risk_decision_v2
review_plan_v2
reviewer_assignment_v2
reviewer_output_v2
aggregation_record_v1
review_result_v1
context_manifest_v1
execution_journal_event_v1
```

- [x] **Step 4：运行协议测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_review_protocol_v2.py tests/test_architecture_boundaries.py `
  -q -p no:cacheprovider
```

Expected: PASS；新协议不导入 Pipeline、MemoryStore、Model Adapter 或 Eval。

- [x] **Step 5：提交**

```text
feat: add minimal review protocol v2
```

---

## Task 2：实现 Safe I/O、PRWorkspace 与 Snapshot 身份

**Files:**

- Create: `src/review_agent/safe_io.py`
- Create: `src/review_agent/pr_workspace.py`
- Modify: `src/review_agent/revision.py`
- Modify: `src/review_agent/checkpoint.py`
- Create: `tests/test_safe_io.py`
- Create: `tests/test_pr_workspace.py`

- [x] **Step 1：写 Workspace 安全与身份 RED 测试**

覆盖：

- 同 repository identity + external review ID 产生稳定 pr_id；
- 相同 base/head 产生相同 snapshot_id，新 head 产生新 Snapshot；
- 不同 PR 不能解析对方 Artifact ID；
- manifest、PR metadata、Intent history、Snapshot、Session 和 Results 使用规范目录；
- create-only 文件不能覆盖；atomic replacement 不暴露半文件；
- symlink、junction、alternate data stream、`..`、绝对路径、大小写/尾随点空格别名 fail closed；
- hash 验证、普通文件验证和 interrupted staging cleanup；
- Windows 路径使用真实短根和短 physical ID，不创建 `\\?\` 通用包装层或 Junction。

- [x] **Step 2：提取安全写入原语**

把 `checkpoint.py`/`attempts.py` 中可复用的原子写、fsync、canonical relative path 和 regular-file 检查移入 `safe_io.py`，旧调用暂时通过兼容 import 使用同一实现。

- [x] **Step 3：实现 PRWorkspaceStore**

提供：

```text
resolve_pr(...)
create_or_load_workspace(...)
create_or_load_snapshot(...)
create_session(...)
resolve_snapshot_artifact(...)
publish_create_only(...)
read_verified_json(...)
```

Workspace manifest 和 Snapshot manifest 均绑定 repository identity、pr_id、base/head SHA 和 schema version。

- [x] **Step 4：运行测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_safe_io.py tests/test_pr_workspace.py tests/test_checkpoint_reporting.py `
  -q -p no:cacheprovider
```

Expected: PASS；旧 CheckpointStore 原子写回归不变，新 PRWorkspace 隔离测试通过。

- [x] **Step 5：提交**

```text
feat: add isolated PR workspace and snapshot store
```

---

## Task 3：用完整 DiffArtifact 取代截断 ChangeSummary

**Files:**

- Create: `src/review_agent/diff_artifact.py`
- Modify: `src/review_agent/git_repo.py`
- Modify: `src/review_agent/pr_workspace.py`
- Create: `tests/test_diff_artifact.py`
- Modify: `tests/test_git_repo.py`

- [x] **Step 1：写完整性和索引 RED 测试**

覆盖普通修改、新增、删除、rename/copy、binary、mode change、无换行文件、Unicode path、CRLF、SHA-1 和 SHA-256 仓库：

- `diff.patch` 字节与权威 Git 命令输出完全一致且不截断；
- index 覆盖每个 file section 与 hunk，保存 byte offsets、旧/新 line range、status 和 path；
- 按 file/hunk 读取与原 patch 对应字节一致；
- content hash 或 index offset 被篡改时拒绝读取；
- 50K 分页读取返回 cursor/has_more；
- Git 调用显式使用 `-c core.longpaths=true`、`--no-ext-diff` 和固定 diff 配置。

- [x] **Step 2：实现两阶段发布**

先在 Snapshot staging 写完整 patch，再从已经写入的 patch 机械生成 index；二者 hash 校验成功后一起 create-only 发布。不得分别运行逐文件 Git diff 形成不一致视图。

- [x] **Step 3：保留旧 ChangeSummary 只供 v5**

新模块不得导出 `diff_excerpt/file_diff_excerpts/diff_truncated`。Task 13 切换前，旧 Pipeline 仍可调用 `collect_change_summary()`。

- [x] **Step 4：运行测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_diff_artifact.py tests/test_git_repo.py `
  -q -p no:cacheprovider
```

Expected: PASS；完整 Diff 与所有索引切片字节一致。

- [x] **Step 5：提交**

```text
feat: persist complete indexed diff artifacts
```

---

## Task 4：收敛 Deterministic Preflight

**Files:**

- Create: `src/review_agent/preflight.py`
- Create: `src/review_agent/local_quality.py`
- Modify: `src/review_agent/repository_intelligence.py`
- Modify: `src/review_agent/pr_workspace.py`
- Create: `tests/test_preflight_v2.py`
- Create: `tests/test_local_quality.py`
- Modify: `tests/test_repository_intelligence.py`

- [x] **Step 1：写唯一 Preflight 顺序 RED 测试**

断言顺序严格为 `DiffArtifact -> QualityGate -> ChangedSymbols`，三类产物都绑定同一 snapshot_id；Diff 或 Snapshot 不可建立时阻断，普通 QualityGate failed/unavailable/error 不阻断 Planning。

- [x] **Step 2：简化 QualityGate 状态和发现规则**

只保留：

```text
passed
failed
unavailable
error
```

新 `local_quality.py` 只运行项目已配置、无网络、只读、无需安装依赖的 syntax/compile/type/lint/build 静态检查。v6 不实现 cheap/deep、风险触发二次执行、test/security 扩展计划或 `skipped/timed_out` 产品状态；timeout 归一为 `error` 并保留稳定 reason code。

旧 `quality.py/quality_runner.py` 暂时保持 v5 行为，直到 Task 16 移除旧 Pipeline 引用。

QualityGate 单命令和阶段总 watchdog 均不得超过 1,800 秒；更短的项目级超时可以保留。

- [x] **Step 3：把大 gate stdout/stderr 接到通用 Artifact 接口占位**

本 Task 先定义 `PreflightArtifactSink` protocol；Task 8 提供统一实现。测试使用内存 sink，避免 Preflight 依赖 ToolGateway。

- [x] **Step 4：扩展 ChangedSymbols provenance**

每条保存 analyzer、version、configuration 和 language coverage；缓存 key 必须覆盖 repository/base/head、配置和 analyzer version。Python AST 成功不得宣称其他语言已覆盖。

- [x] **Step 5：运行测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_preflight_v2.py tests/test_local_quality.py `
  tests/test_quality.py tests/test_repository_intelligence.py `
  -q -p no:cacheprovider
```

Expected: PASS；没有 deep gate 调度，failed gate 仍允许后续阶段读取结果。

- [x] **Step 6：提交**

```text
refactor: collapse preflight to deterministic checks
```

---

## Task 5：实现 IntentPacket v2 与 PR 级版本历史

**Files:**

- Create: `src/review_agent/intent_runtime.py`
- Modify additively: `src/review_agent/intent_inference.py`
- Modify: `src/review_agent/pr_workspace.py`
- Modify: `src/review_agent/review_protocol.py`
- Create: `tests/test_intent_v2.py`
- Modify: `tests/test_intent_inference.py`

- [x] **Step 1：写 explicit/inferred/null RED 测试**

覆盖：

- 明确用户目标 -> `source=explicit`；
- 根据 PR metadata/Diff 推断 -> `source=inferred`；
- 无法形成可信目标 -> `goal=null, source=null`；
- goal/source nullability 不变量；
- 下游 Packet 不包含 rules、acceptance criteria、scope、constraints、status、provenance 或 clarifications；
- Intent Agent 的 envelope/raw response/tool trace 仅进入内部 IntentAnalysisRecord；
- 新版本引用 source_snapshot_id，旧版本不可覆盖。

- [x] **Step 2：把现有 Intent 复杂模型隔离为内部分析**

可以通过纯函数复用当前 inference 候选和证据收集，但旧 Intent wire model 与 v5 行为保持不变；`intent_runtime.py` 最终只投影新 IntentPacket。v6 不进入交互式 `AWAITING_USER`；缺失 Intent 直接形成 null source，后续 Risk floor 升为 high。

- [x] **Step 3：实现跨 Snapshot 延续**

显式 Intent 可延续到新 Snapshot；inferred Intent 必须重新校验并产生新版本。历史文件 create-only，`current.json` 只原子更新版本指针。

- [x] **Step 4：运行测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_intent_v2.py tests/test_intent.py tests/test_intent_inference.py `
  -q -p no:cacheprovider
```

Expected: PASS；Reviewer/Risk 只能取得三个字段的 Packet。

- [x] **Step 5：提交**

```text
refactor: project minimal intent packet v2
```

---

## Task 6：简化 Risk，并固定 Reviewer Slots 与 Assignments

**Files:**

- Create: `src/review_agent/risk_runtime.py`
- Create: `src/review_agent/review_planning.py`
- Modify additively: `src/review_agent/model_risk.py`
- Modify additively: `src/review_agent/portfolio.py`
- Modify: `src/review_agent/review_protocol.py`
- Create: `tests/test_risk_v2.py`
- Create: `tests/test_review_planning.py`
- Modify: `tests/test_model_risk.py`
- Modify: `tests/test_portfolio.py`

- [x] **Step 1：写 Risk floor RED 测试**

固定：

```text
50 files         no file-count floor
51 files         medium floor
source=explicit  low intent floor
source=inferred  medium intent floor
source=null      high intent floor
```

模型只返回 `{"level":"..."}`；最终等级为所有 deterministic floors 与模型等级的 max。业务敏感度、影响范围和可撤销性由模型判断，但非敏感普通改动的 Prompt 不应无依据升级。

- [x] **Step 2：删除 Risk 下游冗余字段**

新路径不产生 dimensions、reasons、signal_refs、uncertainties 或 suggested_focus。内部 Risk analysis 可以保留审计，但 `risk.json` 只保存 snapshot binding、最终 level、floor/model level 的最小审计字段。旧多字段 Risk/Portfolio API 在 Task 16 前继续只服务 v5。

- [x] **Step 3：实现固定 Slot mapping**

```text
low       core
medium    core + adversarial
high      core + adversarial + dynamic
critical  core + adversarial + dynamic + dynamic
```

Critical 两个 Dynamic perspective 必须不同。Planner 只能填 perspective、mission、targets 和 checks，不能改变 Slot 数、基础角色、权限、Provider 或 Runtime 限制。

- [x] **Step 4：精简 Assignment**

删除 risk reasons/signals、旧 Contract、max_turns/max_tool_calls/max_output_tokens/max_total_tokens。保留 assignment_id、role、role_kind、perspective、mission、targets、checks、snapshot_id 和只读权限。

- [x] **Step 5：运行测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_risk_v2.py tests/test_review_planning.py `
  tests/test_model_risk.py tests/test_portfolio.py `
  -q -p no:cacheprovider
```

Expected: PASS；每个风险等级得到固定 1/2/3/4 Slots，Planner 无法增删。

- [x] **Step 6：提交**

```text
refactor: simplify risk and fix reviewer slots
```

---

## Task 7：重建 Reviewer System、规则权威和初始上下文

**Files:**

- Create: `src/review_agent/review_policy.py`
- Create: `src/review_agent/global_memory.py`
- Create: `src/review_agent/review_context.py`
- Modify additively: `src/review_agent/context.py`
- Modify additively: `src/review_agent/execution_profile.py`
- Create: `tests/test_reviewer_context_v2.py`
- Create: `tests/test_global_memory.py`
- Modify: `tests/test_context.py`
- Modify: `tests/test_execution_profile.py`
- Modify: `tests/test_architecture_boundaries.py`

- [x] **Step 1：写四输入边界和规则优先级 RED 测试**

断言每次模型调用仍严格为 system/tools/messages/parameters；DeveloperReviewPolicy 位于 System 且没有 CLI/user override；用户规则与 LLM 经验只进入 user message 的 `{{system_rule}}`；冲突时低优先级规则不能覆盖开发者规则。

- [x] **Step 2：实现静态 Pinned Context**

初始 user message 固定包含：

```text
Review Identity
完整公开 User Conversation
{{system_rule}}
IntentPacket
当前 Assignment
QualityGate + relevant ChangedSymbols
完整 Diff 或完整 Diff Index/相关片段
Available Artifacts
```

不得包含 Intent/Risk 内部对话、其他 Reviewer 历史、整个 Review Plan、旧 Contract、Risk reasons 或无关 Artifact。

当前 CLI 的一条用户请求也必须先规范化为 speaker-labeled conversation；以后追加的公开澄清沿用同一结构，不能把 Orchestrator 私有推断伪装成用户消息。

- [x] **Step 3：实现 Global Memory facade 和规则投影**

复用现有 Durable Memory 的身份、完整性和 CLI 管理能力，但 Review Pipeline 不再拥有 MEMORY_SELECTION/MEMORY_PROPOSAL Phase。Facade 只选择用户规则和已批准经验，生成当前 Session 的不可变 snapshot；不得保存 PR Diff、Intent、Assignment 或临时 Tool Result，也不得把 live MemoryStore 传入 Reviewer execution modules。

- [x] **Step 4：实现 Diff fit policy**

先估算除 Diff 外的 Pinned Token；完整 Diff 能保持初始目标不超过 500K～600K 时内联，否则放 Index、相关 hunk 和 diff artifact_id。不得按 120/80 行截断。

- [x] **Step 5：移除旧字符预算和 Memory 10% 配额**

`ContextBudget(max_message_chars=16_000)`、`compacted_section_min_chars` 和固定 `memory_subbudget_ratio` 不进入新 Reviewer protocol projection。Task 16 前旧类可以留给 v5；Global Memory 通过 immutable snapshot 投影，不把 live MemoryStore 传给 Reviewer。

- [x] **Step 6：运行测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_reviewer_context_v2.py tests/test_global_memory.py tests/test_context.py `
  tests/test_execution_profile.py tests/test_architecture_boundaries.py `
  -q -p no:cacheprovider
```

Expected: PASS；旧 16K/10% 预算不出现在 v2 projection，规则权威和 Pinned 区块稳定。

- [x] **Step 7：提交**

```text
refactor: rebuild reviewer context and policy boundary
```

---

## Task 8：实现 Tool Result Artifact Store 与 50K/200K 协议

**Files:**

- Create: `src/review_agent/tool_artifacts.py`
- Create: `src/review_agent/review_tool_gateway.py`
- Modify additively: `src/review_agent/tool_gateway.py`
- Modify additively: `src/review_agent/tool_result_protocol.py`
- Modify additively: `src/review_agent/model_adapter.py`
- Modify: `src/review_agent/preflight.py`
- Create: `tests/test_tool_artifacts.py`
- Modify: `tests/test_tool_gateway.py`
- Modify: `tests/test_tool_result_protocol.py`
- Modify: `tests/test_model_adapter.py`

- [x] **Step 1：写边界值 RED 测试**

覆盖 49,999、50,000、50,001、199,999、200,000、200,001 字符，以及 Unicode 序列化后字符数：

- 不可重新获取 >50K 立即 Artifact + <=2K preview；
- 不可重新获取 <=50K 随 transcript 保留，不创建独立 Artifact；
- 可重新获取结果默认不创建 Artifact；
- 单轮 >200K 时先淘汰最旧/最大可重新获取结果，再对不可重新获取小结果做聚合 Artifact；
- `read_artifact` 每页 <=50K，cursor/has_more 正确；
- Artifact write 失败形成显式 Tool Error，不能丢内容。

- [x] **Step 2：增加调用级 reacquirable metadata**

当前 Snapshot-bound `read_range/compare_base_head/search_code/list_symbols/inspect_symbol/find_references/read_commit_messages/query_project_memory` 都可重新获取。分类记录在 Tool Result index，不依赖名称硬编码的上下文清理逻辑。

新 `ReviewToolGateway` 不要求 ObservationStore，也不返回 observation IDs；它返回统一 ToolExecutionResult/Artifact envelope。旧 `ToolGateway` 只留给 v5 compatibility，不能进入新 Reviewer path。

- [x] **Step 3：统一 Tool Error envelope**

只保留 `is_error/code/retryable/message` 和必要 exit metadata。timeout、临时锁、transient I/O 可重试；非法参数、越权路径、缺 Artifact、path_too_long 和确定性 parse error 不可重试。

- [x] **Step 4：接入 Preflight 大日志**

QualityGate stdout/stderr 使用同一 Artifact Store，但不通过 Reviewer ToolGateway 执行。

- [x] **Step 5：运行测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_tool_artifacts.py tests/test_tool_gateway.py `
  tests/test_tool_result_protocol.py tests/test_model_adapter.py `
  -q -p no:cacheprovider
```

Expected: PASS；所有边界值和分页完整性通过。

- [x] **Step 6：提交**

```text
feat: add bounded tool result artifact protocol
```

---

## Task 9：实现 Execution Journal、幂等 Tool Call 与新 Runtime 限制

**Files:**

- Create: `src/review_agent/execution_journal.py`
- Create: `src/review_agent/review_agent_loop.py`
- Create: `src/review_agent/reviewer_executor.py`
- Modify additively: `src/review_agent/reviewer_runtime.py`
- Modify additively: `src/review_agent/session.py`
- Create: `tests/test_execution_journal.py`
- Create: `tests/test_review_agent_loop_v2.py`
- Create: `tests/test_reviewer_executor_v2.py`
- Modify: `tests/test_agent_loop.py`
- Modify: `tests/test_resume.py`

- [ ] **Step 1：写每个 crash window 的 RED 测试**

在以下事件后模拟中断并 Resume：

```text
model_response
tool_started
tool_completed
turn_committed
```

断言已提交 Tool Call 不重跑；started 无终态的只读调用可恢复；相同 call_id 不同参数 fail closed；每个 assistant tool_call 恰好有匹配且相邻的 tool result。

- [ ] **Step 2：实现 append-only journal 和 Tool Call ledger**

事件先完整单行写入、flush/fsync，再推进状态。Tool Call identity 绑定 session/assignment/call/tool/args hash/snapshot。Resume 只重放到最后一个 `turn_committed`。

- [ ] **Step 3：移除人工 Turn/Tool/Token 停止条件**

新 ReviewAgentLoop 使用 active-time 控制的 while loop，不接收 max_turns、max_tool_calls、max_total_tokens 或 8,192 max_output 停止参数。保留 Usage 统计、用户取消、重复无进展调用检测和 Provider/Tool 安全超时。旧 `agent_loop.py` 在 Task 16 前保持 v5 行为。

- [ ] **Step 4：固定运行保护**

```text
active elapsed        1,800 seconds
provider attempts     3 per model turn
tool timeout          300 seconds
```

离线/暂停时间不计 active elapsed，Resume 不重置已消费时间。Provider transport retry 不执行工具；协议无效结果不伪装成 transport retry。

- [ ] **Step 5：省略 Reviewer max_output_tokens**

Provider 支持省略时不发送；强制要求时使用配置的模型能力。内部 Compaction output 在 Task 10 使用独立 50K 上限，不影响 Reviewer 最终输出。

- [ ] **Step 6：运行测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_execution_journal.py tests/test_review_agent_loop_v2.py `
  tests/test_reviewer_executor_v2.py tests/test_agent_loop.py tests/test_resume.py `
  -q -p no:cacheprovider
```

Expected: PASS；无 budget_exhausted 来自 turn/tool/token，所有 crash point 可恢复。

- [ ] **Step 7：提交**

```text
feat: make reviewer turns durable and resumable
```

---

## Task 10：实现三层 Context Window 与全量压缩

**Files:**

- Create: `src/review_agent/context_window.py`
- Modify: `src/review_agent/review_agent_loop.py`
- Modify: `src/review_agent/review_context.py`
- Modify additively: `src/review_agent/model_adapter.py`
- Modify: `src/review_agent/execution_journal.py`
- Create: `tests/test_context_window.py`
- Modify: `tests/test_review_agent_loop_v2.py`
- Modify: `tests/test_model_adapter.py`

- [ ] **Step 1：写 Token Window 和固定顺序 RED 测试**

断言每次 Provider 调用前顺序为 assemble -> Layer 1 -> Layer 2 -> token estimate -> Layer 3 -> re-estimate -> hard check。Layer 1 每轮检查，Layer 2/3 只在触发时改变投影。

- [ ] **Step 2：实现 TokenEstimator protocol**

精确 provider estimator 优先；fallback 使用 UTF-8 bytes 的保守 upper bound。完整请求计入 System、tools、messages、tool results、output reserve 和 50K safety reserve。

- [ ] **Step 3：实现 60 分钟清理**

3,599 秒不触发，3,600 秒触发；按完成顺序保留最近 5 个 context-evictable Tool Result 完整正文，旧正文替换为包含 call ID/tool/args hash/reason/reacquirable 的 marker。不得留下孤立 Tool Call。

`last_api_request_at` 使用真实 UTC，在请求交给 Provider Adapter 前持久化；Layer 2 清理后的第一轮请求立即建立新的时间基线。

- [ ] **Step 4：实现 700K full compaction**

用当前 Reviewer model 生成一个 <=50K Token 的 Summary，覆盖已完成调查、关键事实、候选 Findings、uncertainties 和下一步。压缩所有 committed 动态历史，不保留最近尾巴；System、规则、用户请求、Intent、Assignment、Preflight、ChangedSymbols、Diff/Index 字节不变。

Compaction Summary 作为 user-level 不可信数据块重新注入，不得提升为 System/Developer 指令。Compaction 调用计入 Reviewer active elapsed 和该 Turn 的 Provider attempt 保护。

- [ ] **Step 5：实现 Compaction 原子提交与 Resume**

写 `context_compaction_started`，Summary/hash/manifest 安全发布后写 committed。孤立 started 忽略；失败不改变活动投影。压缩后完整请求必须 <700K，否则返回 `context_compaction_failed`。

- [ ] **Step 6：运行测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_context_window.py tests/test_review_agent_loop_v2.py tests/test_model_adapter.py `
  -q -p no:cacheprovider
```

Expected: PASS；阈值、Pinned 不变、marker 配对、Compaction rollback 和 Resume 全部通过。

- [ ] **Step 7：提交**

```text
feat: add three-layer reviewer context compaction
```

---

## Task 11：切换 Reviewer Prompt、Parser 与 ReviewerOutput v2

**Files:**

- Create: `src/review_agent/reviewer_output.py`
- Modify: `src/review_agent/review_context.py`
- Modify: `src/review_agent/review_agent_loop.py`
- Modify: `src/review_agent/reviewer_executor.py`
- Modify additively: `src/review_agent/hydration.py`
- Create: `tests/test_reviewer_output_v2.py`
- Modify: `tests/test_review_agent_loop_v2.py`
- Modify: `tests/test_reviewer_executor_v2.py`
- Modify: `tests/test_reviewer.py`

- [ ] **Step 1：写严格输出 RED 测试**

Reviewer 最终响应只接受：

```json
{"findings": [], "uncertainties": []}
```

每个 Finding 只接受 claim/severity/path/line/suggestion。删除 confidence、impact、evidence_refs、verification_performed、contract_assessments、rejected_hypotheses、observation_refs、investigation_summary 和模型声明 status。

- [ ] **Step 2：更新 System Prompt 和 JSON Schema**

Finding claim 覆盖缺陷、触发条件、影响；suggestion 必须具体。模型不生成 finding_id。Runtime safety、DeveloperReviewPolicy、数据不可信边界和工具协议继续位于 System。

- [ ] **Step 3：实现 envelope 与候选级校验**

顶层 JSON/字段非法 -> Reviewer `invalid_output`。单个 Finding 非法 -> 只拒绝该候选并记录稳定 reason，其他合法 Finding 保留。path/line 必须能在当前 Snapshot/Diff index 解析。

- [ ] **Step 4：移除 Review Contract completion 校验**

新 Reviewer 路径不调用 `validate_reviewer_completion()` 或构造 ContractAssessment。旧 `reviewer.py/review_contract.py` 只供 v5 tests，Task 16 决定删除范围。

- [ ] **Step 5：运行测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_reviewer_output_v2.py tests/test_review_agent_loop_v2.py `
  tests/test_reviewer_executor_v2.py tests/test_reviewer.py `
  -q -p no:cacheprovider
```

Expected: PASS；模型无法伪造 Runtime status，单个坏 Finding 不丢掉同结果的好 Finding。

- [ ] **Step 6：提交**

```text
refactor: adopt minimal reviewer output v2
```

---

## Task 12：实现 Deterministic Aggregator、ReviewResult 与纯 Renderer

**Files:**

- Create: `src/review_agent/aggregation.py`
- Create: `src/review_agent/review_renderer.py`
- Modify: `src/review_agent/pr_workspace.py`
- Create: `tests/test_aggregation.py`
- Create: `tests/test_review_renderer.py`

- [ ] **Step 1：写 fingerprint 与 merge RED 测试**

问题身份严格由 snapshot_id + canonical path + line + NFKC/trim/whitespace-collapse claim 组成并保留大小写。severity、suggestion、Reviewer role/order/time 不进入 identity。

覆盖：

- exact identity 合并，severity 取最高；
- 平局按 core -> adversarial -> dynamic、Assignment order、Reviewer ID；
- 同位置不同 claim 不合并；`Foo`/`foo` 不合并；
- 相同输入重复聚合得到相同 Finding ID、排序和 canonical JSON bytes。

- [ ] **Step 2：实现状态和 uncertainty 规则**

```text
all valid     completed
some valid    partial
none valid    failed
```

Core 失败但其他有效时返回 partial 和已有 Findings。Runtime error 生成稳定、清洗后的 coverage uncertainty；Reviewer uncertainties 只做确定性规范化和去重。

- [ ] **Step 3：实现 create-only ReviewResult 发布**

写 `aggregation.json` 后原子发布 `review-result.json`；Result 严格六字段且无时间戳。Resume 验证 Snapshot/hash 后复用，不重新聚合。

- [ ] **Step 4：实现纯 Markdown renderer**

只显示 status、risk、severity、path:line、claim、suggestion 和 uncertainties。不得生成 summary、recommendation 或 JSON 中不存在的事实。

- [ ] **Step 5：运行测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_aggregation.py tests/test_review_renderer.py `
  -q -p no:cacheprovider
```

Expected: PASS；重复运行字节一致，Markdown 可删除后重建。

- [ ] **Step 6：提交**

```text
feat: add deterministic review result aggregation
```

---

## Task 13：引入 Session v6 五阶段 Pipeline 并切断旧后处理主链

**Files:**

- Create: `src/review_agent/review_pipeline.py`
- Modify: `src/review_agent/run_state.py`
- Modify: `src/review_agent/session.py`
- Modify: `src/review_agent/session_store.py`
- Modify: `src/review_agent/resume.py`
- Modify: `src/review_agent/hydration.py`
- Modify: `src/review_agent/artifacts.py`
- Create: `tests/test_pipeline_v6.py`
- Create: `tests/test_session_v6.py`
- Modify: `tests/test_session_store.py`
- Modify: `tests/test_resume.py`

- [ ] **Step 1：写 v6 phase/state RED 测试**

新 Session 只包含 PREFLIGHT/INTENT/PLANNING/REVIEWERS/AGGREGATION。每个 Phase 的 predecessor、artifact ownership、running restart、completed reuse 和 invalidation 规则单独测试。

- [ ] **Step 2：实现新 PipelineContext 和 Phase dispatcher**

使用 Tasks 2～12 的服务，不导入 reconciler/supplemental/evidence/completion/final_risk/brief。Context Assembly 在每个 Reviewer 启动时执行；Markdown rendering 属于 AGGREGATION 的确定性尾部，不成为独立 Phase。

- [ ] **Step 3：实现并行 Reviewer failure isolation**

所有计划 Reviewer 终态后才聚合；单个 failed/timeout/invalid_output 不终止其他 Reviewer。主线程按 Assignment order 提交结果，保证确定性。

- [ ] **Step 4：定义 legacy Session 边界**

v1-v5 manifest 仍可读并显示 schema/status/artifact diagnostics，但 Resume 返回明确 unsupported。删除 `--upgrade-to-v5` 对新路径的影响，不提供 v5 -> v6 自动字段迁移。

- [ ] **Step 5：添加架构禁止项**

`review_pipeline.py` 和新 Reviewer execution modules 不得导入旧后处理模块、live MemoryStore、Eval、SQLite 或 subprocess QualityGate 实现。

- [ ] **Step 6：运行测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_pipeline_v6.py tests/test_session_v6.py `
  tests/test_session_store.py tests/test_resume.py tests/test_architecture_boundaries.py `
  -q -p no:cacheprovider
```

Expected: PASS；新 Pipeline 无任何旧后处理 import，legacy fixture 只读诊断通过。

- [ ] **Step 7：提交**

```text
feat: cut product pipeline over to session v6
```

---

## Task 14：切换 CLI、PR 身份和 Resume 输出

**Files:**

- Modify: `src/review_agent/command.py`
- Modify: `src/review_agent/cli.py`
- Modify: `src/review_agent/__main__.py`
- Modify: `src/review_agent/resume.py`
- Create: `tests/test_cli_v6.py`
- Modify: `tests/test_cli_smoke.py`
- Modify: `tests/test_cli_resume.py`

- [ ] **Step 1：更新 review 参数**

增加 `--external-review-id`、`--workspace-root` 和 `--format json|markdown`。移除 review 主命令的 semantic-reconciler、memory-curator 和 supplemental 配置；保留 Risk model 与 Reviewer provider 配置。

- [ ] **Step 2：切换执行入口**

新 review 创建/复用 PRWorkspace 和 Snapshot，创建 Session v6，调用 `ReviewPipelineV6`，最终直接输出 ReviewResult 或纯 Markdown。CLI 不再读取 completion/final_risk/review_brief，也不打印 Recommendation。

- [ ] **Step 3：更新 Resume**

CLI 打印 pr_id、snapshot_id、session_id。Resume 使用稳定 locator 定位 PRWorkspace Session；已完成 ReviewResult 直接复用。Legacy review ID 只提供 inspect 指引，不启动旧 Pipeline。

- [ ] **Step 4：定义退出码**

建议固定：

```text
0 completed or partial ReviewResult successfully returned
1 runtime/infrastructure failure before authoritative ReviewResult
2 CLI/config/session schema error
```

ReviewResult `status=failed` 已是权威产品结果时仍返回 0，调用方读取 JSON status；不能把可测产品失败伪装成 CLI infrastructure crash。

- [ ] **Step 5：运行 fake CLI E2E**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_cli_v6.py tests/test_cli_smoke.py tests/test_cli_resume.py `
  -q -p no:cacheprovider
```

Expected: PASS；输出只来自 ReviewResult，resume 不运行旧 phases。

- [ ] **Step 6：提交**

```text
refactor: switch cli and resume to pr workspace v6
```

---

## Task 15：更新 Execution Profile 与 Eval Product-equivalence Adapter

**Files:**

- Modify: `src/review_agent/execution_profile.py`
- Modify: `src/review_agent_eval/adapters/current_agent.py`
- Modify: `src/review_agent_eval/models.py` only if adapter-neutral binding requires it
- Modify: `tests/test_execution_profile.py`
- Modify: `tests/eval/test_current_agent_adapter.py`
- Modify: `tests/eval/test_agent_adapter.py`
- Modify: `tests/eval/test_submission_boundary_v2.py`
- Modify: `tests/eval/test_protocol_v2_cutover.py`

- [ ] **Step 1：bump Product Execution Profile**

新 profile digest 必须绑定：

- IntentPacket v2；
- level-only Risk；
- fixed Slot mapping；
- ReviewerOutput/Finding schema；
- DeveloperReviewPolicy digest；
- Tool 50K/200K/300s；
- active 1,800s/provider attempts 3；
- 1M/60m/700K/50K Summary Context policy；
- deterministic aggregation/review-result schema。

删除 semantic reconciler、completion、final risk、ReviewBrief、16K/8K/turn/tool/token budget digests。

- [ ] **Step 2：让 Eval 注入稳定 PR 身份和短 workspace root**

Adapter 使用 benchmark task_id 作为 external_review_id；Trial 私有短根作为 workspace root。不得把 Ground Truth、Judge prompt 或 expected findings 暴露给产品。

- [ ] **Step 3：从 ReviewResult 构造 Submission**

不再读取 `completion.json/review_brief.json`。将 FinalFinding 投影为 SubmissionFinding：

```text
finding_id       direct
claim            direct
severity         direct mapping
path             direct
side/from/to     from DiffArtifact index + line
suggested_action suggestion
evidence_refs    Eval-only scoped Diff context generated by Adapter
```

产品 Finding 不回填 Eval 字段。无法解析 line anchor 是产品 artifact/schema failure，不允许查 Ground Truth 修复。

- [ ] **Step 4：保持 failure-as-miss 与 infrastructure isolation**

ReviewResult failed/partial 仍是可测 Submission；PRWorkspace、Diff、hash 或 adapter materialization 错误继续归 infrastructure，不进入能力分数。

- [ ] **Step 5：运行 Eval 定向回归**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_execution_profile.py `
  tests/eval/test_current_agent_adapter.py `
  tests/eval/test_agent_adapter.py `
  tests/eval/test_submission_boundary_v2.py `
  tests/eval/test_protocol_v2_cutover.py `
  -q -p no:cacheprovider
```

Expected: PASS；Eval 不再依赖 Completion/ReviewBrief，profile drift 会 fail closed。

- [ ] **Step 6：提交**

```text
refactor: bind eval adapter to review result v1
```

---

## Task 16：断开旧主链、收缩兼容面并更新文档

**Files:**

- Modify: `src/review_agent/pipeline.py`
- Modify: `src/review_agent/reporting.py`
- Modify: `src/review_agent/artifacts.py`
- Modify: `src/review_agent/session.py`
- Modify: `src/review_agent/session_store.py`
- Modify/Delete when unreferenced: `src/review_agent/reconciler.py`
- Modify/Delete when unreferenced: `src/review_agent/supplemental.py`
- Modify/Delete when unreferenced: `src/review_agent/evidence.py`
- Modify/Delete when unreferenced: `src/review_agent/completion.py`
- Modify/Delete when unreferenced: `src/review_agent/final_risk.py`
- Modify/Delete when unreferenced: `src/review_agent/brief.py`
- Modify: `tests/test_architecture_boundaries.py`
- Modify: affected legacy unit tests
- Modify: `docs/superpowers/specs/2026-08-10-pr-workspace-preflight-reviewer-runtime-redesign.md`

- [ ] **Step 1：用 rg 和 AST tests 证明新产品入口无旧 import**

检查 command/review_pipeline/resume/execution_profile/current Eval Adapter。新路径不得出现 Reconciler、Supplemental、Completion、FinalRisk、ReviewBrief 或 deep quality gate。

- [ ] **Step 2：删除旧调度和配置面**

删除 RunPhase dispatch、Session supplemental waves/budgets、CLI stage args、artifact ownership 和 profile digests。只读 legacy hydration 如仍需要，移动到明确命名的 compatibility module。

- [ ] **Step 3：删除旧测试预期，保留安全回归**

删除“旧功能仍在主链”的测试，不删除 path traversal、hash、prompt injection、provider timeout、artifact integrity 和 Eval isolation 等安全测试；将其改写到新边界。

- [ ] **Step 4：更新主 Spec 实施状态和迁移说明**

逐项勾选实现状态，记录 v6 breaking schema、legacy read-only 和外部 review ID 要求。不要把本实施计划中的临时步骤写成长期产品承诺。

- [ ] **Step 5：运行架构与邻接测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_architecture_boundaries.py `
  tests/test_session.py tests/test_session_store.py tests/test_resume.py `
  tests/test_pipeline_v6.py tests/test_cli_v6.py `
  -q -p no:cacheprovider
```

Expected: PASS；新主链对旧后处理模块零依赖。

- [ ] **Step 6：提交**

```text
refactor: remove legacy post-review product pipeline
```

---

## Task 17：完整回归、Fake E2E 与授权后的真实 Smoke

**Files:**

- Modify only if failures reveal an in-scope regression
- Update: this plan checkboxes and final verification record

- [ ] **Step 1：运行产品测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests `
  --ignore=tests/eval `
  -q -p no:cacheprovider
```

Expected: all product tests pass。

- [ ] **Step 2：运行 Eval 测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval `
  -q -p no:cacheprovider
```

Expected: all Eval tests pass；无 Ground Truth/Judge 泄漏。

- [ ] **Step 3：运行全套测试与静态检查**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  -q -p no:cacheprovider

git diff --check
git status --short
```

Expected: full suite pass；diff check 无输出；只有预期源代码、测试和文档变更。

- [ ] **Step 4：运行 Fake Adapter CLI E2E**

至少覆盖：

- explicit/inferred/null Intent；
- low/medium/high/critical Slot 数；
- 0 Finding completed；
- partial Reviewer failure；
- malformed Finding candidate isolation；
- 60m Resume cleanup；
- 700K Compaction；
- 已完成 ReviewResult Resume reuse；
- SHA-1/SHA-256 和 Windows long-path fixture。

- [ ] **Step 5：请求真实模型 smoke 授权**

在没有当次明确授权时停止，不读取或输出 API key。授权后只运行：

1. 一个小 AACR PR；
2. 一个大 Diff/Index PR；
3. 一个 SWE PR；
4. inspect ReviewResult、coverage、required metrics、无 harness materialization error。

真实 smoke 失败时保留 Run/Session/Artifact，不覆盖旧结果，不修改 Ground Truth。

- [ ] **Step 6：最终提交**

```text
feat: complete pr workspace reviewer runtime redesign
```

---

## 6. 每个 Task 的通用完成条件

- [ ] RED 测试先失败，且失败原因确实是待实现行为缺失。
- [ ] 最小实现后定向测试 GREEN。
- [ ] 邻接回归 GREEN。
- [ ] 新 Artifact/Schema 有严格 hydration 和 unknown-field rejection。
- [ ] 新持久化写入有 hash、Snapshot binding 和 crash-safe publish。
- [ ] 没有 API key、绝对私有路径或未清洗异常进入用户 JSON/Markdown。
- [ ] `git diff --check` 无输出。
- [ ] 一个 Task 一个逻辑提交，提交信息与本计划一致。

## 7. 最终验收映射

| Spec 能力 | 实施 Task |
|---|---:|
| PRWorkspace / Snapshot | 2, 13, 14 |
| 完整 DiffArtifact / Index | 3 |
| 简单 QualityGate / ChangedSymbols | 4 |
| IntentPacket v2 | 5 |
| Risk floor / fixed slots / Assignment | 6 |
| Developer/User rules / Reviewer Context | 7 |
| 50K/200K Tool Result | 8 |
| Unlimited turns/tools/tokens + 1800/3/300 | 9 |
| 60m/700K/1M Context Compaction | 10 |
| Finding / ReviewerOutput v2 | 11 |
| Deterministic ReviewResult | 12 |
| Five-phase Session v6 | 13 |
| CLI / Resume | 14 |
| Product-equivalent Eval Adapter | 15 |
| Legacy post-review removal | 16 |
| Full regression / real smoke | 17 |

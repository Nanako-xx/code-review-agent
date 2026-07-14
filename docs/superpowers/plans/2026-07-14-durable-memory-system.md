# Durable Memory System 实施计划

**状态：** 待执行（2026-07-14）

**设计来源：** `docs/superpowers/specs/2026-07-14-durable-memory-system-design.md`

**目标：** 按确认后的最终架构实现 revision-bound Repository Knowledge Cache、人工批准的 Durable Project Memory、非抑制性的 Review Feedback Memory，以及 Session v5、MemorySnapshot、Memory Curator、Context、Runtime、CLI、恢复与报告集成。

**执行方式：** 建议 Subagent-Driven。按下述 Wave 和文件所有权并行；主线程负责依赖接口、跨模块集成、兼容审计、全量回归和最终提交。这里的 Batch/Wave 是最终架构的依赖拆分，不允许引入之后会被替换的临时 schema、JSON store 或简化状态机。

**技术栈：** Python dataclasses/enums、stdlib `sqlite3`、Git、SHA-256 内容寻址 blob、现有统一 Model Adapter、pytest、现有 Session/Artifact/Observation 基础设施。

---

## 1. 全局不变量

- 现有 `.review-agent/runs/<review-id>/` 继续是单次 Review 的权威 Session Memory；长期数据库不复制完整 Session。
- Repository Knowledge 必须精确绑定 repository、revision、capability、分析器版本、配置摘要和输入摘要；旧 revision 结果不能冒充当前事实。
- Agent/模型只能提出 Candidate；只有人工审批事件能创建 `active` Durable Memory Record。
- `validated` 只表示结构和来源有效，不表示内容获得项目权威性。
- 普通自然语言 Memory 是信息性上下文；只有人工明确批准的白名单 `policy_effect` 可由 Runtime 编译为硬约束。
- `risk_floor` 只能提高风险；Memory 不能减少 Reviewer、删除 Contract、降低 severity、扩大工具/文件/网络/命令权限或增加预算。
- 每次 v5 Review 固定一个 Session 内不可变 MemorySnapshot；同 revision resume 不读取未来 generation，revision drift child 必须重新选择。
- raw Feedback 不进入 Reviewer/Reconciler Context，不自动形成 suppression rule，不得隐藏新的证据充分 Finding。
- Memory Store、Cache、Candidate outbox、Feedback 和 SourceBundle 都必须可审计、幂等、可恢复并显式暴露失败。
- 敏感内容、`.env`、密钥、认证 URL、隐藏 reasoning 不得进入长期记忆或报告。
- Business/Memory modules 不直接依赖 Provider；Memory Curator 只通过统一 Model Adapter 调用模型。
- v1 继续只读审计；v2/v3/v4 使用原阶段布局和原恢复语义，不进入 v5 Memory phase。
- 本批不实现 Eval Harness、GitHub/PR 产品集成、云同步、自动修复、自动评论、自动 Approve 或自动 Merge。

## 2. 执行与提交规则

- 每个 Task 先写失败测试，再实现最小完整的最终行为，再运行该 Task 定向测试。
- 并行 Task 的写入文件必须互斥；共享接口由主线程在 Wave 开始前固定，子 Agent 不自行改写其他所有权文件。
- 每个 Task 形成独立提交候选；主线程完成 spec compliance 和 code quality review 后才集成。
- 一个 Wave 的接口和定向测试全部通过后，才能开始依赖它的下一 Wave。
- Task 7 和 Task 13 会切换新 Session/CLI 默认协议，必须在隔离 worktree/任务分支开发；它们只能与 Task 15 的完整 Pipeline dispatch/load 一起提升到可交付集成分支。不得把“Session 已是 v5、Pipeline 尚不认识 Memory phase”的红色中间状态作为阶段成果交付。
- 不清理或暂存既有 `.intent-*`、`.p-*`、`.pytest-*`、`.tmp` 等临时目录。
- 不把主 Spec 的 Windows 行尾状态当作内容改动；最终同步时只暂存实际 diff。
- 未经用户明确要求，不 push、不创建 PR、不 merge 到 master。

### 实施开始门禁

- [ ] 精确提交已确认设计和本计划，排除既有临时目录及主 Spec 的无内容行尾状态。
- [ ] 从包含该文档提交的干净基线创建 `codex/durable-memory-system` 实现分支。
- [ ] 记录 `git status --short --untracked-files=no`、HEAD 和 Python 版本。
- [ ] 使用独立 `C:\tmp` basetemp 运行一次全量 pytest；任何真实失败先处理或向用户报告，不能把既有 cleanup warning 当成测试通过依据。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-baseline'
```

## 3. 依赖图

```text
Wave A1
  Task 1 Memory models ─────┐
  Task 2 Identity/root ─────┤
                            v
Wave A2                 Task 3 SQLite/event/blob store
                            |
             ┌──────────────┴──────────────┐
Wave A3      v                             v
         Task 4 Sources/lifecycle      Task 5 Repository cache
             └──────────────┬──────────────┘
                            v
Wave A4                 Task 6 Core memory CLI
                            |
        ┌───────────────────┼───────────────────┬───────────────────┐
Wave B1 v                   v                   v                   v
     Task 7 Session v5   Task 8 Retrieval    Task 9 Curator     Task 10 Feedback
                            |                   |                   |
              ┌─────────────┴──────────┐        |                   |
Wave B2       v                        v        |                   |
          Task 11 Context/tool     Task 12 Stage policy projections
              └─────────────┬──────────┘
                            v
Wave B3          Task 13 Review CLI/config     Task 14 Brief/report
              └─────────────┬──────────────────┘
                            v
Wave C1                 Task 15 Pipeline integration
                            |
Wave C2                 Task 16 E2E/hardening/docs
```

---

# Batch A：持久化内核与 Repository Cache

## Wave A1：并行基础模型

### Task 1：Canonical Memory Models、schema 与 stable IDs

**依赖：** 无。

**所有权：**

- 新建 `src/review_agent/memory_models.py`
- 新建 `tests/test_memory_models.py`

**RED 测试：**

- [ ] `SourceRef` 只接受设计允许的六种 typed source，拒绝任意字段、空 ID、非法 revision/path/range/hash。
- [ ] `MemoryScope` 规范化 POSIX path glob、symbol、contract、language，并拒绝空 scope 的非法 kind。
- [ ] Candidate canonical serialization 与输入顺序无关。
- [ ] 相同 canonical candidate 产生相同 `MC-<sha256>`；confidence、sensitivity、policy effect、source 或 producer schema 改变时 ID 改变。
- [ ] `content_fingerprint` 忽略 source/review/producer，但保留 kind、statement、scope 和 policy semantics。
- [ ] `MEM-`、`FB-`、`MSNAP-`、event/request ID 格式稳定，拒绝截断或非十六进制 ID。
- [ ] Candidate、Record、Feedback、Snapshot、SourceBundle、generation metadata 严格 round-trip。
- [ ] 所有 collections canonical、去重、不可变；未知 enum/schema/version fail closed。

**实现：**

- [ ] 定义 `MemoryKind`、Candidate/Record/Feedback status、decision、applicability、sensitivity、validity-policy 和 policy-effect enums。
- [ ] 定义 immutable `MemoryScope`、typed SourceRef variants、Producer、Candidate、Record、SourceBundle descriptor、FindingSnapshot、FeedbackRecord。
- [ ] 定义 `MemoryExecutionConfig`、SelectionInput/Decision、MemorySnapshot、RepositoryKnowledgeKey/Entry、FeedbackCalibrationSummary。
- [ ] 实现 canonical JSON-ready serialization、strict hydration、SHA-256 stable ID/fingerprint helpers。
- [ ] 模型中不保存 Path 对象、SQLite row、datetime 对象或 Provider 类型；边界统一使用规范化 string/int/bool/tuple。
- [ ] 所有正文和 collection 设置明确长度/数量上限，防止数据库或 Context 无界增长。

**验证：**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_memory_models.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-models'
```

**提交边界：** `feat(memory): add canonical memory models and stable identities`

### Task 2：Memory Root 与 Repository Identity

**依赖：** 无；可与 Task 1 并行。

**所有权：**

- 新建 `src/review_agent/memory_identity.py`
- 修改 `src/review_agent/revision.py`（仅增加 Memory identity 所需的规范化 helper；不改 revision resolution 语义）
- 新建 `tests/test_memory_identity.py`
- 修改 `tests/test_revision.py`（仅 identity 兼容测试）

**RED 测试：**

- [ ] CLI override、`REVIEW_AGENT_MEMORY_ROOT` 和 Windows/Linux/macOS 默认路径优先级。
- [ ] root 必须 canonical absolute path；相对路径、文件路径、不可安全创建的父目录形成明确错误。
- [ ] repository key 使用 normalized git common dir + sanitized origin。
- [ ] 同一 common dir 的 worktree 得到相同 key。
- [ ] 相同 origin 的独立 clone 得到不同 key。
- [ ] 无 origin 仓库仍有稳定 key。
- [ ] origin 中 userinfo/token/query 不进入 key metadata、Session 或错误消息。
- [ ] 仓库移动/重新 clone 不根据 origin 静默继承旧 namespace。

**实现：**

- [ ] 实现平台默认 root resolver 与 override/env 解析。
- [ ] 实现 repository key、namespace path 和 metadata payload。
- [ ] 路径解析拒绝 traversal/symlink escape；namespace 只允许固定十六进制目录名。
- [ ] 预留显式 relink/export/import 所需 old/new identity descriptor，不实现隐式匹配。
- [ ] 不写仓库工作树、`.git` 或 `.review-agent`。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_memory_identity.py tests/test_revision.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-identity'
```

**提交边界：** `feat(memory): isolate repository memory namespaces outside worktrees`

## Wave A2：权威 Store

### Task 3：SQLite、append-only events、generations 与 blob store

**依赖：** Task 1、Task 2。

**所有权：**

- 新建 `src/review_agent/memory_store.py`
- 新建 `tests/test_memory_store.py`
- 修改 `tests/test_architecture_boundaries.py`（仅 Store 依赖边界）

**RED 测试：**

- [ ] 首次打开创建最终 `memory_store_schema_v1`，再次打开不重复 migration。
- [ ] `PRAGMA foreign_keys=ON`、WAL、busy timeout 生效；schema version 未知时只读/失败，不猜测迁移。
- [ ] candidate/event/record/feedback/knowledge write 在同一事务内更新对应 generation。
- [ ] request ID 重放幂等；同一 subject 的冲突状态 compare-and-swap 失败且不产生第二 Record。
- [ ] event previous/current hash 连续；删除、换序、篡改可检测。
- [ ] blob temp write、hash/size 校验、原子提升、DB commit 和 orphan GC crash window。
- [ ] DB 引用缺失/错误 hash blob 时 fail closed。
- [ ] pinned Session/SourceBundle blob 不被普通 GC 删除。
- [ ] migration 在 staging copy 失败时保留原 DB，成功后原子替换。
- [ ] 并行 reader、幂等 writer、审批 writer 的锁和 busy timeout 行为。
- [ ] export manifest canonical、脱敏、有总 hash；import dry-run 不写状态。

**实现：**

- [ ] 建立 metadata/repositories/generations/blobs/knowledge entries/candidates/records/events/feedback/source bundles/outbox receipts 表与索引。
- [ ] 所有 authority writes 使用 `BEGIN IMMEDIATE` 和显式 commit/rollback。
- [ ] Store 只接收 canonical memory models，不接收 PipelineContext、ReviewerResult 或 Provider response。
- [ ] 实现 atomic blob writer、reference validation、pin、GC scan。
- [ ] 实现 event append + current projection 原子 API。
- [ ] 实现 validated read views、generation snapshot、backup/migration/export/import primitives。
- [ ] SQLite 异常转换为稳定的 MemoryStoreError taxonomy，不泄露 SQL、凭证或原始敏感内容。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_memory_store.py tests/test_architecture_boundaries.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-store'
```

**提交边界：** `feat(memory): add transactional sqlite event and blob store`

## Wave A3：并行来源与 Cache

### Task 4：Source Validator、SourceBundle、审批与生命周期

**依赖：** Task 1-3。

**所有权：**

- 新建 `src/review_agent/memory_sources.py`
- 修改 `src/review_agent/memory_store.py`（仅 lifecycle transaction API）
- 新建 `tests/test_memory_sources.py`
- 新建 `tests/test_memory_lifecycle.py`

**RED 测试：**

- [ ] repository range/symbol/commit source 在精确 revision 重读并校验 hash。
- [ ] Observation/session artifact source 校验 review ID、descriptor schema、revision binding、artifact hash 和 Observation authority。
- [ ] human declaration 只接受显式用户/CLI request、actor、时间和声明 hash。
- [ ] absolute path、`..`、symlink escape、`.git`、`.env`、secret/key/token/credential 内容被拒绝。
- [ ] `local_only` 可落盘但不能标记为 remote-sendable；`blocked` 不保存敏感正文。
- [ ] approval 前再次校验来源并原子生成最小 SourceBundle。
- [ ] SourceBundle 缺失/篡改使 Record 不可审计，但不能把 bundle 当成目标 HEAD 仍有效的证明。
- [ ] proposed → validated → pending → approved；invalid 自动 rejected；人工 reject/revoke/revalidate/supersede 合法转移。
- [ ] revalidate 创建新 Candidate/Record 并 supersede 旧 Record，不原地改正文。
- [ ] exact duplicate 幂等；content duplicate 无增强来源时不重复审批；rejected 内容无变化时不可反复提案。
- [ ] ancestor、not-yet-valid、diverged lineage、source changed/missing、scope trigger、manual-until-revoked applicability。

**实现：**

- [ ] SourceRef allowlist validator 复用现有 RevisionResolver、SessionStore、Observation hydration，不自行信任 JSON。
- [ ] schema-aware secret scan/redaction 和安全错误摘要。
- [ ] Candidate validation report、dedupe decision 和 rejection taxonomy。
- [ ] approval/reject/revoke/revalidate/supersede 的 actor/reason/request-ID 事务。
- [ ] SourceBundle materialization、pin 和审计读取。
- [ ] target-HEAD applicability evaluator 与全局 lifecycle projection 分离，历史分支检查不得错误改写主线状态。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_memory_sources.py tests/test_memory_lifecycle.py tests/test_session_store.py tests/test_observations.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-sources'
```

**提交边界：** `feat(memory): validate sources and enforce human-owned lifecycle`

### Task 5：Immutable Repository Knowledge Cache

**依赖：** Task 1-3；可与 Task 4 并行。

**所有权：**

- 新建 `src/review_agent/repository_cache.py`
- 修改 `src/review_agent/repository_intelligence.py`
- 新建 `tests/test_repository_cache.py`
- 修改 `tests/test_repository_intelligence.py`

**RED 测试：**

- [ ] cache key 完整包含 repository、revision binding、capability、analyzer name/version、config digest、input digest。
- [ ] 相同 exact key 命中；HEAD、LSP 状态、AST/ripgrep/version/config 任一变化均 miss。
- [ ] same-content file blob 可复用，但目标 revision manifest 始终不同且精确绑定。
- [ ] `off` 不读写跨运行 cache；`read` 命中可读、miss 只产 Session 结果；`read-write` 可持久写。
- [ ] cache corruption/缺失 blob 触发确定性重建，不返回 stale 数据。
- [ ] Session pin 防止使用中的 entry 被 GC。
- [ ] Repository Intelligence 输出和现有 summary/hydration 保持兼容。

**实现：**

- [ ] RepositoryKnowledgeKey/manifest/blob writer/lookup/validation API。
- [ ] file/symbol/definitions/references/calls/tests/config/git-summary capability metadata。
- [ ] 把现有 AST/ripgrep/LSP fallback 状态纳入 config digest。
- [ ] Repository Intelligence 支持可选 cache backend，但权威输出仍是该次 Session 的 `RepositoryIntelligenceSnapshot`。
- [ ] 记录 hit/miss/rebuild、entry ID、blob hash、analyzer provenance；不把 cache 当作越权读取通道。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_repository_cache.py tests/test_repository_intelligence.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-repository-cache'
```

**提交边界：** `feat(memory): add exact-revision repository knowledge cache`

## Wave A4：本地人工管理面

### Task 6：Core Memory CLI

**依赖：** Task 1-5。

**所有权：**

- 修改 `src/review_agent/command.py`
- 修改 `src/review_agent/__main__.py`（仅需要时）
- 新建 `tests/test_memory_cli.py`
- 修改 `tests/test_cli_smoke.py`（仅顶层 parser/exit code）

**RED 测试：**

- [ ] `memory status/list/show/candidates/candidate show` 只读命令不改变 generation。
- [ ] approve/reject/revoke/revalidate 要求明确 ID、actor、reason；非交互写入缺少 `--yes` 时拒绝。
- [ ] approve 输出 statement/scope/source/validity/policy diff 后才提交。
- [ ] import 默认 dry-run；apply 要求显式 identity match/relink 和确认。
- [ ] relink 不根据 origin 自动选择旧 namespace。
- [ ] export 脱敏；GC 默认 dry-run，不能删除 pinned blob。
- [ ] 所有错误返回稳定 exit code，不打印 API key、认证 URL、SQL 或敏感正文。

**实现：**

- [ ] 增加 `memory` subparser 和 core management commands。
- [ ] 统一 `--repo`、`--memory-root`、actor、reason、interactive/non-interactive 处理。
- [ ] CLI 只调用 Store/Validator service，不复制状态机或 SQL。
- [ ] JSON 与 human-readable 输出都包含 stable IDs 和 generation。
- [ ] 本 Task 只注册已经实现的 core 命令；`feedback`、`replay-outbox` 在 Task 13 具备完整依赖后一次性加入。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_memory_cli.py tests/test_cli_smoke.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-cli-a'
```

**提交边界：** `feat(memory): add explicit local memory approval workflow`

## Batch A Gate

- [ ] Task 1-6 spec compliance review。
- [ ] Store/models/sources/cache 不 import Pipeline、Provider 或 CLI。
- [ ] worktree/clone/secret/concurrency/crash-window 测试通过。
- [ ] SQLite schema、artifact-independent IDs 和 export schema 固定，不留临时字段。
- [ ] 运行 Batch A 合集：

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_memory_models.py tests/test_memory_identity.py tests/test_memory_store.py tests/test_memory_sources.py tests/test_memory_lifecycle.py tests/test_repository_cache.py tests/test_memory_cli.py tests/test_architecture_boundaries.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-batch-a'
```

---

# Batch B：Session v5、Selection 与 Runtime 集成

## Wave B1：并行协议能力

### Task 7：Session schema v5、Memory phases 与 artifact/hydration

**依赖：** Batch A Gate。

**所有权：**

- 修改 `src/review_agent/run_state.py`
- 修改 `src/review_agent/session.py`
- 修改 `src/review_agent/session_store.py`
- 修改 `src/review_agent/artifacts.py`
- 修改 `src/review_agent/hydration.py`
- 修改 `src/review_agent/attempts.py`
- 修改 `tests/test_run_state.py`
- 修改 `tests/test_session.py`
- 修改 `tests/test_session_store.py`
- 修改 `tests/test_artifacts.py`
- 修改 `tests/test_hydration.py`
- 修改 `tests/test_attempts.py`

**RED 测试：**

- [ ] v5 phase 顺序精确包含 Repository Intelligence 后的 `MEMORY_SELECTION` 和 Final Risk 后的 `MEMORY_PROPOSAL`。
- [ ] v1/v2/v3/v4 phase list 与 resume/validation 保持原样。
- [ ] `MemoryExecutionConfig` 全字段严格验证，`required=true + off` 非法。
- [ ] resolved root、mode、required、policy versions、budgets 和 `memory_curator: ModelStageConfig` Session round-trip。
- [ ] v1-v4 hydrate 为 legacy memory-off/no-curator 语义，不隐式升级 manifest。
- [ ] v5 artifacts 使用设计中的十个固定 schema 和正确 phase/revision binding。
- [ ] Memory artifact tamper、错误 phase、错误 schema、错误 revision 触发验证失败。
- [ ] invalidation 从 Memory Selection 或 Proposal 的最小正确下游范围开始。

**实现：**

- [ ] `SESSION_SCHEMA_VERSION = 5`，保留 v1-v4 constants、layouts、resumable rules。
- [ ] 扩展 RunPhase、phase messages、manifest validation、SessionStore predecessor/invalidation logic。
- [ ] execution config 新增固定 Memory config 和独立 Curator model stage。
- [ ] 注册 `memory_selection_input_v1`、`memory_snapshot_v1`、`memory_selection_decision_v1`、`feedback_calibration_summary_v1`、Curator/candidate/outbox/receipt schemas。
- [ ] typed hydration 不从缺失字段猜测 v5；legacy artifacts 继续按旧 defaults。
- [ ] AttemptWorkspace 和 artifact path 继续使用 canonical Session-relative 路径。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_session.py tests/test_session_store.py tests/test_artifacts.py tests/test_hydration.py tests/test_attempts.py tests/test_run_state.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-session-v5'
```

**提交边界：** `feat(memory): add schema v5 memory phases and artifacts`

### Task 8：Applicability、deterministic retrieval、Snapshot 与 policy compiler

**依赖：** Batch A Gate；可与 Task 7、9、10 并行。

**所有权：**

- 新建 `src/review_agent/memory_retrieval.py`
- 新建 `src/review_agent/memory_policy.py`
- 新建 `tests/test_memory_retrieval.py`
- 新建 `tests/test_memory_policy.py`

**RED 测试：**

- [ ] selection 顺序严格为 repository/status/revision/stage/scope/policy/relevance/stable-ID/budget。
- [ ] active 但 target HEAD not-yet-valid/diverged/source-changed/missing/scope-trigger 的 Record 不进入 authoritative set。
- [ ] stage-kind allowlist 与 path/symbol/contract/language scope 正确。
- [ ] lexical/graph score 稳定；相同 score 使用 memory ID tie-break。
- [ ] 可选 semantic ranker 只能重排 eligible 集合，不能加入 stale/revoked/pending 或超预算记录。
- [ ] Snapshot 复制 canonical record，不依赖实时 DB row；相同 input/generation 产生相同 hash/ID。
- [ ] snapshot record/byte 上限、每调用 record 上限和 query result 上限。
- [ ] ordinary record 可 budget-omitted 并记录原因；hard-policy 不能静默省略，超限明确阻塞。
- [ ] `risk_floor` 只提高；required contract/check 必须来自 registry；verification hint 只接受模板 ID。
- [ ] 未知/非法 effect fail closed；任何 effect 都不能扩大权限、工具、网络、shell 或预算。

**实现：**

- [ ] target-revision applicability、stage projection、scope matcher、stable ranker 和 budget ledger。
- [ ] MemorySnapshot builder、canonical decision catalog、generation capture、disabled/empty snapshot。
- [ ] Snapshot-only query service；不接受 live Store 作为 Reviewer tool 查询源。
- [ ] typed policy compiler 与 Runtime action/diagnostic 输出。
- [ ] feedback summary 只作为安全聚合输入，不读取 raw records。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_memory_retrieval.py tests/test_memory_policy.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-retrieval'
```

**提交边界：** `feat(memory): add deterministic snapshots and policy compiler`

### Task 9：Local/Model Memory Curator

**依赖：** Batch A Gate；可与 Task 7、8、10 并行。

**所有权：**

- 新建 `src/review_agent/memory_curator.py`
- 修改 `src/review_agent/model_adapter_factory.py`（仅 Curator fake/model response registration）
- 新建 `tests/test_memory_curator.py`
- 修改 `tests/test_model_adapter_factory.py`（仅 Curator coverage）

**RED 测试：**

- [ ] local Curator 只从显式 project rule/human declaration/validated typed source 产生 Candidate；无来源返回空 batch。
- [ ] model envelope 无工具、单轮、最小 final verified context、source-ref allowlist 和 existing fingerprint catalog。
- [ ] strict parser 拒绝 unknown/duplicate/missing keys、非法 enum、过长文本、未授权 source ref、重复 Candidate ID。
- [ ] 模型不能返回 approved/active 状态、actor decision 或任意 policy type。
- [ ] Provider/parse/timeout/attempt exhaustion 产生 deterministic warning/empty decision，不改变 Review conclusion。
- [ ] raw response 在持久化前执行 secret scan/redaction；无法安全脱敏时只保留 hash/metadata 并拒绝 batch。
- [ ] 同一 request digest/invocation 重试稳定。
- [ ] 模块只依赖 Model Adapter protocol/factory，不 import provider implementation。

**实现：**

- [ ] Curator input/output/decision schema 与 canonical candidate compiler。
- [ ] local deterministic candidate producer。
- [ ] model envelope、strict parse、finite retry、stable invocation metadata。
- [ ] Candidate Validator 前置 allowlist；Curator 不执行 approval/store transaction。
- [ ] sanitized raw response helper 与 failure taxonomy。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_memory_curator.py tests/test_model_adapter_factory.py tests/test_architecture_boundaries.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-curator'
```

**提交边界：** `feat(memory): add proposal-only local and model curator`

### Task 10：Review Feedback、FindingSnapshot 与安全聚合

**依赖：** Batch A Gate；可与 Task 7-9 并行。

**所有权：**

- 新建 `src/review_agent/memory_feedback.py`
- 新建 `tests/test_memory_feedback.py`

**RED 测试：**

- [ ] accepted/rejected/severity_changed 根据 Session canonical Finding ID、head SHA、Finding hash、evidence refs 校验。
- [ ] missed 要求人工声明和可验证 repository/Observation source。
- [ ] Feedback 写入复制最小 immutable FindingSnapshot；原 Session 删除后仍能聚合。
- [ ] 同一 feedback request 幂等，冲突 decision 不静默覆盖。
- [ ] 聚合至少 5 条、至少 3 个 Review，scope/contract/reason 可比，保留 record IDs/时间范围/样本数。
- [ ] 样本不足只可供 Eval，不进入 Context/调度。
- [ ] 聚合 API 只能提高检查/perspective 优先级或要求更多证据，不能形成 suppression/risk/severity lowering action。
- [ ] raw reason、claim 和 Feedback record 不出现在 Reviewer/Reconciler projection。

**实现：**

- [ ] Feedback validation/import service、FindingSnapshot materializer、event/store transaction。
- [ ] `feedback_aggregation_v1` deterministic group/threshold/provenance。
- [ ] calibration output 使用安全 taxonomy 和计数，不复制 raw Finding/人类自由文本。
- [ ] 明确禁止 Feedback → Durable Memory 自动转换。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_memory_feedback.py tests/test_evidence.py tests/test_reconciler.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-feedback'
```

**提交边界：** `feat(memory): add non-suppressive review feedback memory`

## Wave B2：并行 Context 与阶段语义

### Task 11：Reviewer Context、Memory subbudget 与 Snapshot query tool

**依赖：** Task 7、8。

**所有权：**

- 修改 `src/review_agent/context.py`
- 修改 `src/review_agent/tool_gateway.py`
- 修改 `src/review_agent/reviewer_task_executor.py`
- 修改 `tests/test_context.py`
- 修改 `tests/test_tool_gateway.py`
- 修改 `tests/test_reviewer_task_executor.py`
- 修改 `tests/test_agent_loop.py`（仅 Memory tool turn）

**RED 测试：**

- [ ] 新 section 为 Approved Project Memory、Repository Knowledge、Feedback Calibration Summary。
- [ ] 每条 Memory 带 ID/kind/scope/authority/source/target validity；不同 authority 不混淆。
- [ ] Memory + feedback 只占 messages budget 的 10%，不挤掉 Assignment、Intent、Initial Context、Completion Rules。
- [ ] hard-policy overflow 不能被普通 compaction 静默截断。
- [ ] envelope metadata 记录 snapshot ID、selected/omitted IDs、hash、policy version 和原因。
- [ ] `local_only` 不进入 remote model messages。
- [ ] `query_project_memory` 只查询 Assignment 绑定 Snapshot，只接受 bounded path/symbol/contract/query。
- [ ] 工具不能读 live DB、pending/revoked/stale Record，也不能修改状态。
- [ ] 工具结果形成当前 Reviewer Observation，受 tool/output/context budget 限制。
- [ ] 未提供 MemorySnapshot 的 legacy 调用输出保持现有结构。

**实现：**

- [ ] 扩展 Context assembler 的 optional memory inputs 和独立 subbudget。
- [ ] authority-aware renderer、compactor 和 metadata。
- [ ] 注册 `query_project_memory` tool definition、Runtime request validation 和 Snapshot service。
- [ ] Reviewer task executor/agent loop 传递固定 Snapshot handle，不传 Store。
- [ ] Repository Knowledge 只传相关 summary/ref，仍由既有代码工具读取源码。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_context.py tests/test_tool_gateway.py tests/test_reviewer_task_executor.py tests/test_agent_loop.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-context'
```

**提交边界：** `feat(memory): inject bounded approved memory into reviewer context`

### Task 12：Intent/Risk/Planner/Completion/Final Risk policy projections

**依赖：** Task 7、8、10；可与 Task 11 并行。

**所有权：**

- 修改 `src/review_agent/models.py`
- 修改 `src/review_agent/intent_inference.py`
- 修改 `src/review_agent/risk.py`
- 修改 `src/review_agent/model_risk.py`
- 修改 `src/review_agent/portfolio.py`
- 修改 `src/review_agent/review_contract.py`
- 修改 `src/review_agent/completion.py`
- 修改 `src/review_agent/final_risk.py`
- 修改 `tests/test_models.py`
- 修改 `tests/test_intent_inference.py`
- 修改 `tests/test_risk.py`
- 修改 `tests/test_model_risk.py`
- 修改 `tests/test_portfolio.py`
- 修改 `tests/test_review_contract.py`
- 修改 `tests/test_completion.py`
- 修改 `tests/test_final_risk.py`

**RED 测试：**

- [ ] architecture/business/compatibility Memory 只能生成 sourced inferred intent claim，不能直接标成 explicit。
- [ ] 用户确认/纠正后沿用现有 IntentSource/IntentStatus 语义。
- [ ] high-risk/incident Memory 形成 typed signal；只有 compiled risk-floor effect 能提高本地 floor。
- [ ] Reviewer 不接收抽象 `risk_level` 来自行决定深度，仍接收 Runtime 展开的 Assignment。
- [ ] Planner 只能增加 registry 中的 Contract/check/perspective，不删除 Core 或减少风险预算。
- [ ] verification command 只变成已注册 template hint，不能执行任意 shell。
- [ ] Completion 只消费 compiled required contract/check，不解析 Memory statement。
- [ ] memory unavailable/stale/hard-policy overflow 进入 uncertainty/blocker/manual-review policy。
- [ ] Final Risk 解释已应用 Memory 和 residual risk，不读取 pending Candidate/raw Feedback。
- [ ] legacy 无 Memory 输入时所有既有输出保持兼容。

**实现：**

- [ ] 为各阶段定义最小、typed Memory projection，而不是传完整 Snapshot。
- [ ] Intent inference 记录 memory source provenance，必要时继续询问用户。
- [ ] Risk signal catalog/Runtime floor compiler 接入 typed effect。
- [ ] Portfolio 编译 approved contract/check/perspective actions 与 aggregation hint。
- [ ] Completion/Final Risk 增加 memory diagnostic，但保持本地权威。
- [ ] Assignment/InitialContext 只保存 selected memory refs 和已展开要求，不复制整个 Record。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_intent_inference.py tests/test_risk.py tests/test_model_risk.py tests/test_portfolio.py tests/test_review_contract.py tests/test_completion.py tests/test_final_risk.py tests/test_models.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-stage-policy'
```

**提交边界：** `feat(memory): compile approved memory into bounded review policy`

## Wave B3：并行外层协议

### Task 13：Review CLI、MemoryExecutionConfig 与人工 Feedback/outbox 命令

**依赖：** Task 6-10、Task 7。

**所有权：**

- 修改 `src/review_agent/command.py`
- 修改 `tests/test_memory_cli.py`
- 修改 `tests/test_cli_smoke.py`
- 修改 `tests/test_cli_resume.py`

**RED 测试：**

- [ ] `review --memory-mode off|read|read-write` 默认 read-write。
- [ ] root 优先级、canonical path、required/off 冲突和 snapshot budget validation。
- [ ] 独立 `--memory-curator-*` 支持 local/model/inherit provider 规则，Session 只保存 API key env 名。
- [ ] resume 读取已固定 config，不因环境变量变化切换 root/mode/model。
- [ ] `memory feedback record/list` 校验 actor、Finding、source 和确认。
- [ ] `memory replay-outbox <review-id>` 校验 Session artifact/hash/revision 并幂等写入。
- [ ] `off` 不创建跨运行 DB；`read` 不写 Candidate/cache；`read-write` 不自动批准。

**实现：**

- [ ] review parser/config resolver 和 Session v5 construction。
- [ ] Curator ModelStageConfig 继承/覆盖与错误归类。
- [ ] 扩展 memory management CLI 的 Feedback/outbox 命令。
- [ ] human/JSON 输出展示 mode、root fingerprint、generation、pending/disabled/failure 状态。
- [ ] CLI 只组装依赖，不把 Store、selection、feedback 或 curator 业务逻辑写进 parser handler。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_memory_cli.py tests/test_cli_smoke.py tests/test_cli_resume.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-cli-b'
```

**提交边界：** `feat(memory): expose final memory and curator cli configuration`

### Task 14：Brief/Reporting、Finding ID 与 Memory audit projection

**依赖：** Task 7-10；可与 Task 13 并行。

**所有权：**

- 修改 `src/review_agent/brief.py`
- 修改 `src/review_agent/reporting.py`
- 修改 `src/review_agent/evidence.py`（仅 canonical Finding ID projection；不改 Reconciler authority）
- 修改 `src/review_agent/reconciler.py`（仅输出 ID 传递）
- 修改 `tests/test_brief.py`
- 修改 `tests/test_checkpoint_reporting.py`
- 修改 `tests/test_evidence.py`
- 修改 `tests/test_reconciler.py`

**RED 测试：**

- [ ] JSON/Markdown canonical Finding 均保留现有稳定 `finding_id`。
- [ ] Applied Memory 展示 ID/kind/scope/authority/source/validity，不泄漏 SourceBundle 敏感原文。
- [ ] compiled policy、cache provenance、stale/lineage/revalidation warning、feedback summary version/sample count。
- [ ] pending Candidate 显示审批提示但不列为 active rule。
- [ ] memory unavailable、hard-policy block、outbox pending、curator fallback/disabled 可见。
- [ ] JSON/Markdown 语义一致；无 Memory 的 legacy input 保持旧字段兼容。
- [ ] Brief 不内嵌完整 DB、raw Feedback、raw Curator response、hidden reasoning 或大 blob。

**实现：**

- [ ] ReviewBrief 增加 backward-compatible optional memory sections。
- [ ] Reconciler canonical finding ID 一路投影到 Brief/Markdown。
- [ ] reporting renderer 使用结构化 memory payload，不自行查询 Store。
- [ ] hydration/serialization 所需 helper 保持 exact schema 和旧 artifact 默认。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_evidence.py tests/test_reconciler.py tests/test_brief.py tests/test_checkpoint_reporting.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-brief'
```

**提交边界：** `feat(memory): expose auditable memory use in review briefs`

---

# Batch C：Pipeline、Curator Proposal、Feedback 与端到端加固

## Wave C1：单一主线程 Pipeline 集成

### Task 15：Memory Selection、Proposal、outbox、resume 与 revision drift

**依赖：** Task 7-14 全部完成并集成。

**执行限制：** 本 Task 只由主线程或一个专用集成 Agent 修改 Pipeline/Resume；不得与其他写 `pipeline.py` 的任务并行。

**所有权：**

- 修改 `src/review_agent/pipeline.py`
- 修改 `src/review_agent/resume.py`
- 修改 `src/review_agent/incremental.py`（仅需要的 memory lineage metadata）
- 修改 `tests/test_pipeline.py`
- 修改 `tests/test_resume.py`
- 修改 `tests/test_incremental.py`
- 修改 `tests/test_intent_pipeline.py`
- 新建 `tests/test_memory_pipeline.py`

**RED 测试：**

- [ ] v5 完整 phase dispatch/load 顺序与 Session list 一致。
- [ ] Repository Intelligence 在 read/read-write 使用 exact cache；off 完全不接触跨运行 Store。
- [ ] Memory Selection 在 changed files/symbols 后构造 input、applicability、compiled effects、feedback summary 和固定 Snapshot artifacts。
- [ ] off 提交 disabled empty Snapshot；read 正常选择但 Proposal skipped；read-write 正常选择并生成 pending Candidate。
- [ ] Store empty 不是错误；unavailable/corrupt 默认 no-memory + uncertainty/manual-review，required 时 blocked。
- [ ] 同 revision resume 验证并加载原 Snapshot，不读取新 generation。
- [ ] Snapshot tamper 从 Memory Selection 最小失效并重跑下游。
- [ ] DB 在 Selection 后改变/损坏不影响固定 Snapshot。
- [ ] Reviewer query 只读 Snapshot，结果进入独立 Observation。
- [ ] Intent/Risk/Planner/Reviewer/Reconciler/Completion/Final Risk 接收各自最小 projection。
- [ ] Memory Proposal 只在 read-write 执行 local/model Curator，使用最终 verified artifacts 和 allowlisted sources。
- [ ] 先提交 candidate batch/outbox Session artifacts，再幂等写 DB，再提交 receipt。
- [ ] Session 写完、DB 前崩溃和 DB 写完、phase completion 前崩溃均可安全 resume/replay。
- [ ] Curator/validator/store persistence 失败不改变 Review finding/completion，只产生 visible warning/outbox pending。
- [ ] revision drift child 重新 Repository Intelligence/Selection/Proposal，不继承 parent Snapshot、Candidate 或 generation。
- [ ] v1-v4 resume 不打开 Memory Store、不执行新 phase、不调用 Curator。

**实现：**

- [ ] PipelineContext 增加 typed Memory config/store/cache/snapshot/selection/compiled policy/feedback/candidate state。
- [ ] 实现 `_run/_load_memory_selection` 和 `_run/_load_memory_proposal`。
- [ ] Repository Intelligence cache read/build provenance 接入。
- [ ] MemorySnapshot projection 接入现有 Intent、Risk、Portfolio、Reviewer task、Reconciler、Completion、Final Risk 调用。
- [ ] outbox-first persistence 和 receipt/replay digest。
- [ ] resume preservation/discard/minimal invalidation 与 revision drift child 规则。
- [ ] Reporting 构造传递完整结构化 memory audit payload。
- [ ] PHASE_MESSAGES、dispatch、load、artifact commit 和 Session phase 完全一致。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_memory_pipeline.py tests/test_pipeline.py tests/test_resume.py tests/test_incremental.py tests/test_intent_pipeline.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-pipeline'
```

**提交边界：** `feat(memory): integrate durable memory into resumable review pipeline`

## Wave C2：端到端、安全和文档同步

### Task 16：E2E、兼容、故障、并发与最终文档

**依赖：** Task 15。

**所有权：**

- 新建或扩展 Memory integration/E2E 测试文件。
- 修改 `tests/test_architecture_boundaries.py`。
- 修改已确认设计文档状态。
- 修改本实施计划状态与 checkbox。
- 最后才修改 `docs/superpowers/specs/2026-06-22-evidence-driven-multi-agent-code-review-design.md` 的实现状态。

**E2E 场景：**

- [ ] 第一次 Review 无历史 Memory，产生 pending Candidate；人工批准；第二次 Review 选择并引用 active Record。
- [ ] Candidate 在批准前绝不进入第二次 Review 的 authoritative Snapshot。
- [ ] 同 commit Repository Knowledge 命中；新 commit 使用新 manifest，只复用内容相同 blob。
- [ ] source 文件/符号变化使 Memory stale，不再作为权威上下文。
- [ ] diverged/historical HEAD 只产生 per-review applicability，不错误撤销全局 Record。
- [ ] human-approved typed risk/contract/check effect 被 Runtime 执行；natural-language rule 只作信息上下文。
- [ ] Memory 尝试扩大工具/网络/shell/预算被 compiler 拒绝。
- [ ] raw rejected Feedback 不压制新的 blocker/high Finding；聚合只能增加验证要求。
- [ ] local-only Record 不发送给 fake remote adapter。
- [ ] secret/prompt-injection Candidate、Feedback、Curator response 被拒绝/脱敏。
- [ ] Store corruption、missing blob、migration failure、busy writer、read-only root、outbox crash 可见且可恢复。
- [ ] completed Selection resume 固定旧 generation；revision drift child 使用新 generation。
- [ ] v1 fixture 只读；v2/v3/v4 fixture 可恢复且零 Memory/Curator 调用。
- [ ] JSON/Markdown Brief、Session artifacts、SQLite audit events 和 CLI 输出可互相追溯。

**架构检查：**

- [ ] `memory_models/identity/store/sources/cache/retrieval/policy/feedback` 不 import Pipeline、command 或 Provider。
- [ ] `memory_curator` 不 import provider implementation。
- [ ] Pipeline 不包含 SQL、secret scanner、approval state machine 或 context ranking 实现。
- [ ] CLI 不复制 lifecycle/policy/retrieval 逻辑。
- [ ] Reviewer tool 只能持有 Snapshot query service，不持有 live MemoryStore。

**定向回归：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_memory_models.py tests/test_memory_identity.py tests/test_memory_store.py tests/test_memory_sources.py tests/test_memory_lifecycle.py tests/test_repository_cache.py tests/test_memory_retrieval.py tests/test_memory_policy.py tests/test_memory_curator.py tests/test_memory_feedback.py tests/test_memory_cli.py tests/test_memory_pipeline.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-final-targeted'

& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_session.py tests/test_session_store.py tests/test_hydration.py tests/test_context.py tests/test_tool_gateway.py tests/test_pipeline.py tests/test_resume.py tests/test_completion.py tests/test_final_risk.py tests/test_brief.py tests/test_checkpoint_reporting.py tests/test_architecture_boundaries.py -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-final-integration'
```

**全量回归：**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest -q -p no:cacheprovider --basetemp 'C:\tmp\review-agent-memory-full'
git diff --check
```

**文档与提交：**

- [ ] 设计文档状态改为“已实现并通过全量回归”，记录测试结果和 commit。
- [ ] 本计划所有完成项勾选，状态改为“已完成”。
- [ ] 主 Spec 第 15 节/23.1 更新为 Memory 已落地，剩余 Eval Harness 与 GitHub/PR 集成。
- [ ] 精确暂存本批源码、测试、设计和计划；排除所有既有临时目录和无内容行尾状态。
- [ ] 在功能分支创建本地提交；不自动 push/PR/merge。

**提交边界：** `feat(memory): complete durable memory system`

---

## 4. 最终验收映射

| 设计验收 | 主要 Task |
|---|---|
| exact-revision Repository Knowledge、不使用 stale facts | 3、5、15、16 |
| Candidate 人工批准前无 authority | 1、4、6、15、16 |
| source + approval 全链路可审计 | 3、4、6、14、16 |
| HEAD/lineage/source 失效处理 | 4、8、15、16 |
| same-revision resume 固定 Snapshot | 7、8、15、16 |
| revision drift child 重新选择 | 7、15、16 |
| natural language 不扩大 Runtime 权限 | 8、11、12、16 |
| typed policy 只经白名单 compiler | 8、12、15、16 |
| Feedback 不形成 suppression | 10、12、16 |
| outbox crash-safe、幂等 replay | 3、9、13、15、16 |
| Store failure 可配置且可见 | 3、13、15、16 |
| v1-v4 Session 兼容 | 7、15、16 |
| JSON/Markdown 可解释 Applied/Pending/Stale Memory | 14、15、16 |
| 全量单元/集成/安全/并发通过 | 16 |

## 5. 完成定义

本实施计划只有在以下条件全部成立时才算完成：

1. Task 1-16 的最终功能和测试全部完成，不存在临时存储或待替换 schema。
2. 全量 pytest 通过，且没有依赖 pytest cleanup PermissionError 伪装成功。
3. v1-v4 legacy Session 行为经 fixture/集成测试证明不变。
4. Memory Candidate、Record、Feedback、Snapshot、SourceBundle、Repository Cache 都可从 Session/SQLite/Brief 追溯。
5. 所有模型建议均经过 Runtime parser/validator/compiler，人工审批仍是 Durable Memory authority 的唯一入口。
6. 主 Spec、设计、计划与实现状态一致。
7. 用户决定后再执行 push、PR 和 merge。

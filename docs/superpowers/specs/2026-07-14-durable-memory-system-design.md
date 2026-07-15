# Durable Memory System 设计

**日期：** 2026-07-14

**状态：** 已实现并通过全量回归（2026-07-15）

**实现提交：** 本文档与实现同一提交（`feat(memory): complete durable memory system`）

**设计来源：** `2026-06-22-evidence-driven-multi-agent-code-review-design.md` 第 15、16、17、18、23 节

**范围：** Repository Knowledge Cache、Durable Project Memory、Review Feedback Memory，以及它们与现有 Session、Pipeline、Context、Runtime 的集成

## 1. 结论

本设计采用最终形态，不建设之后需要推翻的临时记忆层。实现可以按依赖关系分批完成，但所有批次共享同一套最终数据模型、持久化边界、审批语义和 Session v5 协议。

核心结论如下：

1. 现有 Review Session、Checkpoint、Observation Store 已经承担一次审查的运行记忆，不重新实现。
2. 新增三种生命周期完全不同的持久状态：
   - 绑定精确 revision 的 Repository Knowledge Cache。
   - 经过来源校验和人工批准的 Durable Project Memory。
   - 只用于校准与评测、不能直接压制 Finding 的 Review Feedback Memory。
3. 长期记忆保存在仓库工作区之外的本地应用状态目录，不写入源码目录或 `.git`。
4. SQLite 是权威元数据与审计事件存储；大型 Repository Knowledge 使用内容寻址的不可变 blob。
5. Agent 或模型只能提出 `Memory Candidate`。只有人工审批能创建可用的 Durable Memory Record。
6. 普通自然语言记忆只能作为有来源的上下文。只有人工明确批准、类型受 Runtime 白名单约束的 `policy_effect` 才能形成硬约束。
7. 每次 Review 在开始时固定一个 `MemorySnapshot`。同 revision 恢复时复用该快照，长期数据库后续变化不能污染正在运行的审查。
8. Session 升级为 v5，新增 `MEMORY_SELECTION` 和 `MEMORY_PROPOSAL`。v1-v4 继续使用原阶段布局，不会隐式读取记忆、调用 Curator 或改写历史结果。
9. 记忆读取默认启用；候选持久化、Repository Knowledge 跨运行写缓存和 Feedback 写入受明确的 memory mode 控制。任何候选写入都不会自动激活记忆。
10. Eval Harness 与 GitHub/PR 产品集成仍不在本设计范围内。

## 2. 目标与非目标

### 2.1 目标

- 跨 Review 复用可验证的项目知识，避免反复重建稳定事实。
- 让架构边界、业务不变量、兼容性要求和人工 Review 规则具有明确来源、适用范围和失效机制。
- 复用 revision-bound 的符号、引用、调用、测试映射和项目配置分析，同时杜绝把旧 revision 事实当成当前事实。
- 保存人工对 Finding 的处理结果，用于后续校准和正式评测。
- 保证每次 Review 的记忆输入可重放、可审计、可解释。
- 保持 Runtime 对权限、预算、Review Contract、工具和完成条件的最终控制权。

### 2.2 非目标

- 不保存模型隐藏思维链。
- 不把向量数据库设为必需依赖。
- 不允许模型自动批准、撤销或修改长期记忆。
- 不通过历史 rejected Finding 自动生成“以后不要报告”的规则。
- 不同步云端或团队共享记忆。
- 不在本阶段读取 GitHub 审批、评论或 Merge 结果。
- 不执行自动修复、自动评论、自动 Approve 或自动 Merge。

## 3. 与现有系统的边界

### 3.1 Review Session Memory 保持不变

现有 `.review-agent/runs/<review-id>/` 继续是单次 Review 的权威运行记录，保存：

- ReviewRequest 与 Base/Head。
- Intent、Risk、Assignment 和 Quality Gates。
- Reviewer 调用、Observation、Findings 与 Reconciliation。
- Completion、Final Risk、Brief、失败和恢复状态。

新 Memory System 不替代 Session Store，也不把完整 Session 搬入长期数据库。Session 负责“这次审查发生了什么”，Durable Memory 负责“哪些经过批准的知识可以跨审查继续使用”。

### 3.2 三种持久状态必须隔离

| 状态 | 产生方式 | 权威条件 | 生命周期 | 是否直接进入模型上下文 |
|---|---|---|---|---|
| Repository Knowledge | Runtime 分析 Git revision | 精确 revision、分析器版本和输入摘要匹配 | 可重建、可 GC | 只选相关摘要或片段 |
| Durable Project Memory | Agent/人提出，Runtime 校验，人工批准 | `active` 且对目标 HEAD 有效 | 跨 Review，需失效与撤销 | 按阶段、范围和预算选择 |
| Review Feedback Memory | 人工记录最终处理结果 | 来源 Review/Finding 可验证 | 跨 Review，保留审计历史 | 原始记录不直接传；只传安全聚合结果 |

三者不能共用“是否可信”这一布尔字段。Repository Knowledge 的可信度来自精确 revision 和确定性分析；Project Memory 的权威性来自来源校验与人工批准；Feedback 的权威性只表示“某次人工决定真实发生过”，不表示该决定可以成为项目规则。

## 4. 总体架构

```text
Review Pipeline
    |
    +-- Repository Intelligence
    |      +-- exact-revision cache lookup/build
    |
    +-- Memory Selection
    |      +-- source/revision validity check
    |      +-- deterministic scope filtering
    |      +-- immutable MemorySnapshot in Session
    |
    +-- Intent / Risk / Planning / Reviewers / Reconciler
    |      +-- stage-specific projections from MemorySnapshot
    |      +-- typed policy effects compiled by Runtime
    |
    +-- Final Risk
    |
    +-- Memory Proposal
    |      +-- local or model-assisted candidate proposal
    |      +-- strict source validation and deduplication
    |      +-- pending candidate only
    |
    +-- Reporting
           +-- applied memory, stale memory, pending candidates

Local Memory Root
    +-- repositories/<repository-key>/memory.sqlite3
    +-- repositories/<repository-key>/blobs/sha256/...
    +-- repositories/<repository-key>/exports/...
```

模型不直接访问 Memory Database。Runtime 负责查询、校验、裁剪、固定快照，再把最小充分材料放入标准 `messages`。Memory 不是第五类模型输入；模型调用仍然只有 `system`、`tools`、`messages`、`parameters`。

## 5. 存储位置与 Repository Identity

### 5.1 默认存储位置

Memory Root 按以下优先级解析：

1. CLI `--memory-root`。
2. 环境变量 `REVIEW_AGENT_MEMORY_ROOT`。
3. 平台本地应用状态目录：
   - Windows：`%LOCALAPPDATA%/code-review-agent/memory`
   - Linux：`$XDG_STATE_HOME/code-review-agent/memory`，未设置时使用 `~/.local/state/code-review-agent/memory`
   - macOS：`~/Library/Application Support/code-review-agent/memory`

不默认写入仓库工作树、`.review-agent` 或 `.git`。这样可以避免污染用户代码、被误提交、被仓库内容控制，也能让多个 worktree 共享同一份本地项目记忆。

### 5.2 Repository Key

Runtime 使用现有 `RepositoryIdentity` 生成本地命名空间：

```text
repository_key = sha256(
  normalized(git_common_dir)
  + "\0"
  + normalized(sanitized_origin_url_or_empty)
)
```

语义如下：

- 同一个 Git common dir 下的多个 worktree 共享记忆。
- 即使 origin 相同，不同 clone 也不会静默共享记忆。
- 没有 origin 时仍可使用本地路径身份。
- 仓库移动或重新 clone 后不会仅凭 origin 自动继承旧记忆；必须通过显式 `memory relink` 或受校验的 export/import 操作迁移。
- Repository metadata 保存 canonical path、git common dir、脱敏 origin 和首次/最后访问时间，但不保存认证信息。

## 6. 持久化格式

### 6.1 SQLite 权威存储

每个 repository namespace 使用独立的 `memory.sqlite3`。SQLite 保存：

- schema 与 migration metadata。
- Repository Knowledge manifest 和 cache metadata。
- Memory Candidate、来源校验结果和审批状态。
- Durable Memory Record 与当前状态投影。
- append-only 决策与生命周期事件。
- Review Feedback Record 与聚合版本。
- 单调递增的 `memory_generation`、`feedback_generation` 和 `knowledge_generation`。
- 内容 blob 的摘要、大小、类型和引用计数信息。

数据库启用 foreign keys、WAL 和 busy timeout。审批、撤销、revalidate、candidate import 等权威写操作使用 `BEGIN IMMEDIATE` 串行化，并通过当前状态与 generation 做 compare-and-swap 校验。

当前落地协议版本固定如下；这些版本属于不同边界，不能混成一个“Memory v2”：

| 边界 | 当前版本 | 兼容语义 |
|---|---|---|
| canonical model 与 Session artifact | v1 | 严格字段、canonical JSON 和 hash 校验 |
| Durable Memory Record | v1 / v2 | v1 不含 expiry；v2 必须包含 typed expiry conditions，v1 继续原样读取和序列化 |
| SQLite Store / export manifest | v2 | 可审计或迁移 v1 Store；未知版本 fail closed |
| Review Session | v5 | v1 只读审计；v2-v4 保留原阶段与恢复语义 |
| Memory selection policy | `memory_selection_v2` | 继续读取 `memory_selection_v1` 的固定 Snapshot，不把旧策略静默升级 |

Store v2 用于 authority receipt、严格 persistence receipt、Record v2 与生命周期投影；它不把所有 canonical model/artifact 的 `schema_version` 一并提升。Record schema v2 同样不要求 SQLite Store 再升到 v3。

### 6.2 内容寻址 blob

较大的文件索引、符号图、调用图、测试映射和 Git 摘要写入：

```text
blobs/sha256/<first-two-hex>/<full-sha256>
```

blob 先写临时文件，完成 hash、长度与格式校验后原子提升，再提交引用它的数据库事务。数据库提交失败产生的孤儿 blob 由 GC 清理；数据库引用但缺失或 hash 不匹配的 blob 一律视为 cache corruption，不得降级为“可能仍可用”。

人工批准 Durable Memory 时，Runtime 还要生成最小 `SourceBundle`：保存被批准 statement 所依赖的规范化来源摘录、revision/path/range、原始 content hash 和 bundle hash。它不复制完整 Session 或完整源码文件，也不包含敏感内容。Record 引用的 SourceBundle 永久 pin，不参与普通 cache GC。这样即使原 Review Session 被清理，审批时实际看过的来源仍可审计；如果来源无法安全保存，Candidate 不能被批准。

### 6.3 审计事件与投影

Candidate 和 Memory Record 的正文是不可变版本。状态变化写入 append-only event，再在同一事务内更新当前状态投影。事件至少包含：

- event ID 与 schema version。
- repository key。
- subject ID。
- action。
- actor 类型与 actor ID。
- reason code 和可选说明。
- previous/new status。
- request ID，用于重试幂等。
- UTC timestamp。
- 前一事件 hash 和当前事件 hash。

本设计不声称该 hash chain 能抵御拥有本机写权限的攻击者；它用于检测意外修改、缺失事件和导出不一致。

## 7. Repository Knowledge Cache

### 7.1 Cache Key

每个 cache entry 的唯一键包括：

```yaml
repository_key: ...
revision_binding: head@<full-sha> | base@<full-sha> | <base-sha>..<head-sha>
capability: file_index | symbol_index | definitions | references | calls | tests | project_config | git_summary
analyzer_name: python-ast | ripgrep | lsp-<server> | git | ...
analyzer_version: ...
configuration_digest: ...
input_digest: ...
```

只有所有字段完全匹配时才可命中。LSP 可用性、语言版本、忽略规则和分析参数都进入 configuration digest，不能仅凭 commit SHA 复用不同分析器产生的结果。

### 7.2 不可变与复用

- Cache entry 一经提交不可修改；重建产生新 entry。
- HEAD 变化时必须构建或命中新 revision manifest，不允许把旧 manifest 作为当前事实。
- 相同文件内容可以通过 blob hash 复用底层分析结果，但最终 Repository Knowledge manifest 仍绑定目标 revision。
- 每次 Review 把使用过的 cache entry ID、blob hash 和必要摘要写入 Session artifact。
- Session pin 的 cache entry 不参与普通 GC。
- Cache miss、分析器升级或 corruption 时重新计算；如果当前 memory mode 不允许持久写，则结果只保存在 Session。

### 7.3 与 Repository Intelligence 的关系

现有 `REPOSITORY_INTELLIGENCE` 阶段仍负责生成该次 Review 的权威 `RepositoryIntelligenceSnapshot`。Cache 只是该阶段的可验证加速层，不成为绕过工具权限的第二读取通道。

Repository Intelligence 的输出仍绑定 Base/Head，并继续向 Session、Risk、Planner、Reviewer 和 Brief 提供精简视图。

## 8. Durable Project Memory 数据模型

### 8.1 Memory Kind

`MemoryKind` 固定为：

```text
architecture_boundary
business_invariant
review_rule
compatibility_requirement
verification_command
incident_lesson
high_risk_module
```

新增 kind 需要 schema 迁移和 Runtime 版本升级，模型不能通过自由文本创建新 kind。

### 8.2 Scope

每条 Candidate/Record 至少具有一个结构化 scope：

```yaml
paths: ["payments/**"]
symbols: ["payments.money.calculate_total"]
contracts: ["numeric_correctness"]
languages: ["python"]
```

Scope 字段可为空数组，但整个 scope 不能完全为空，除非 kind 是经过人工批准的 repository-wide `review_rule` 或 `compatibility_requirement`。Path glob 在写入时规范化为仓库相对 POSIX 路径，禁止绝对路径、`..`、`.git` 和敏感路径。

### 8.3 Typed Source Reference

允许的 `SourceRef` 类型为：

- `repository_range`：revision、path、line range、content hash。
- `repository_symbol`：revision、path、qualified name、signature/body hash。
- `git_commit`：完整 commit SHA 和可选受限 metadata hash。
- `observation`：review ID、Observation ID、revision binding、content hash。
- `session_artifact`：review ID、artifact name、artifact schema、artifact hash。
- `human_declaration`：当前 Review/CLI request ID、actor、声明 hash 和时间。

模型只能引用 Runtime 提供的 source-ref allowlist，不能自行构造来源。Repository source 必须能在授权 revision 中重新读取并验证 hash；Observation 和 artifact 必须能在对应 Session 中通过 descriptor、revision binding 与 hash 校验；human declaration 必须来自明确的用户输入或 memory 管理命令。

外部 URL、未读取的文档名、自由文本“据说”和模型自身回答都不是有效 source。

Candidate 校验阶段验证原始 SourceRef；人工审批事务再次验证，并把最小证据固化为 `SourceBundle`。之后的 applicability 仍以目标 HEAD 和原 SourceRef 为准，SourceBundle 只用于审计，不能把已经变化的代码“证明”为仍然有效。

### 8.4 Memory Candidate

```yaml
candidate_id: MC-<sha256>
content_fingerprint: <sha256>
repository_key: ...
kind: business_invariant
statement: "金额计算必须使用 Decimal"
scope:
  paths: ["payments/**"]
  symbols: []
  contracts: ["numeric_correctness"]
  languages: ["python"]
source_refs: [...]
valid_from_sha: <full-sha>
validity_policy: source_content_hash
confidence: high
sensitivity: normal
policy_effect: null
producer:
  type: local | model | human
  name: memory-curator
  version: ...
origin_review_id: ...
status: proposed | validated | pending_approval | approved | rejected
created_at: ...
```

`candidate_id` 根据 repository、kind、规范化 statement、scope、source refs、valid-from、validity policy、confidence、sensitivity、policy effect、producer schema version 和 candidate schema 计算。相同输入重试必须产生相同 ID。

`content_fingerprint` 不包含 source refs、review ID 和 producer，用于发现“同一规则换了一份来源”或“不同 Review 重复提出”的语义重复。它不直接决定合并，最终由确定性规则和人工审批决定。

Candidate 不携带 expiry。`producer.type` 只说明由 human/local/model 中的谁提出 Candidate，不代表其 source authority；过期条件只能由人工在 approve/revalidate 时设置，不能由 Curator 或模型预埋。

### 8.5 Durable Memory Record

人工批准 Candidate 后，在同一事务内创建不可变 Record：

```yaml
memory_id: MEM-<sha256(candidate_id)>
candidate_id: MC-...
schema_version: 1 | 2
kind: ...
statement: ...
scope: ...
source_refs: [...]
source_bundle_hash: <sha256>
valid_from_sha: ...
validity_policy: ...
policy_effect: ...
approved_by: ...
approval_event_id: ...
status: active | revalidation_required | superseded | revoked | expired
expiry_conditions: [...] # 仅 v2；at_time / at_commit
created_at: ...
```

Record v1 不含 `expiry_conditions`，继续保持原 canonical bytes；Record v2 必须至少包含一个 typed expiry condition。无 expiry 的新 Record 仍使用 v1，有 expiry 才使用 v2。Record 正文不会原地编辑。修改 statement、scope、来源、validity policy 或 policy effect 必须产生新 Candidate、新 Record，并把旧 Record 标记为 `superseded`。

## 9. Candidate 校验、去重与审批

### 9.1 状态机

```text
proposed
  -> validated
  -> pending_approval
  -> approved
  -> Durable Memory Record(active)

proposed -> rejected(validation_failed)
validated/pending_approval -> rejected(human_decision)
```

`validated` 表示结构、来源、revision、敏感信息与 policy effect 通过 Runtime 校验，不表示内容已经获得项目权威性。

### 9.2 自动校验

Runtime 必须完成：

- schema 和枚举白名单校验。
- statement 规范化与长度限制。
- scope 路径安全校验。
- source ref allowlist 与 hash 校验。
- valid-from commit 存在性和 lineage 校验。
- secret、credential、`.env`、private key 与高风险隐私扫描。
- 与 active、pending、rejected Candidate 的 exact ID 和 content fingerprint 去重。
- policy effect 类型与参数校验。
- 只属于当前 PR 的临时结论检测；无法证明跨 Review 有效时拒绝持久化。

校验失败的 Candidate 保留最小审计信息和拒绝原因，但敏感正文不得为了审计继续保存。

### 9.3 人工决策

只有本地 memory 管理命令可以执行以下动作：

- approve Candidate。
- reject 已验证 Candidate。
- revoke active Memory。
- revalidate stale Memory。
- supersede Memory。

Review Pipeline、Reviewer、Reconciler 和 Memory Curator 都没有这些权限。人工命令必须提供明确 subject ID、actor 和 reason；非交互模式还必须显式传入确认参数。审批界面必须展示 statement、scope、所有 source refs、validity policy、policy effect、重复项和敏感信息结果。

### 9.4 重复 Candidate

- exact candidate ID 已存在时，import/outbox replay 幂等成功，不新增记录。
- content fingerprint 与 active Record 相同且来源未增强时，Candidate 标记为 duplicate，不再次要求审批。
- 与 rejected Candidate 相同且来源、producer schema 和正文均未变化时，不允许模型反复重新提出。
- 来源增强、scope 改变或 policy effect 改变时允许新 Candidate，但审批界面必须显示与旧记录的差异。

### 9.5 Authority receipt 与审批恢复

Candidate 只有在验证成功后才生成 `CandidateAuthorityReceipt`，固定 candidate、authority/locator repository、origin review、proposal HEAD、授权 source refs、校验报告 hash、authority-resolution hash，以及与其中 `HumanDeclarationSourceRef` 子集一一匹配的 `HumanDeclarationAuthority`。Receipt 中的 `origin` 是 Candidate producer，不是来源 authority；因此 local/model Curator 可以提出基于显式人类声明的 Candidate，但只有 receipt 内独立验证过的 human declaration 才能在审批时恢复 authority。

早期 local/model receipt 可能包含 `human_declaration` SourceRef、但其必需的 `human_declarations` 字段为空。兼容恢复不得修改旧 receipt，也不得从当前 CLI 参数或相似文本猜测：Runtime 只能读取 receipt 精确绑定的、已完成且 artifact hash 校验通过的 Session `request.json`，逐项验证 review ID、request ID、actor 和 declaration hash。Session 缺失、未完成、descriptor/hash 不符或声明集合不完整时一律 fail closed。

Store v1 迁移到 v2 后，历史 pending Candidate 还可能完全没有 authority receipt。CLI 只有在该 Candidate 没有任何 receipt、当前 HEAD 仍等于原 proposal HEAD，且原 Session 的 `request.json` 与 `memory_outbox.json` 均已完成、hash/revision/repository/authority 绑定有效时，才在用户确认后通过 Lifecycle 幂等补写 receipt，再执行审批；预览阶段保持只读。Store 在同一 `BEGIN IMMEDIATE` 中重新断言整个 receipt 集合为空并插入，关闭并发 authority context 的检查/写入窗口；补写后 CLI 再次读取 HEAD，漂移则不创建 active Record。无法满足任一条件时明确拒绝，不能从 Candidate producer 推导 authority。

审批和 outbox persistence receipt 也必须通过严格状态矩阵校验。空写入、`proposed/validated` Candidate、冲突 dedupe 结果或不合法的 replay/write 组合不能伪装成持久化成功。

## 10. 有效性、失效与重新验证

### 10.1 Validity Policy

支持以下类型：

| policy | 判断方式 | 典型用途 |
|---|---|---|
| `source_content_hash` | 在目标 HEAD 重读来源并比较 hash | ADR、配置、代码中的不变量 |
| `symbol_signature` | 比较目标 HEAD 的符号签名/body 摘要 | API、兼容性和关键实现约束 |
| `scope_change_trigger` | `valid_from..HEAD` 触及 scope 即要求复核 | 高风险模块、事故经验 |
| `manual_until_revoked` | 只受人工撤销或显式 expiry 影响 | 人工声明的组织级 Review 规则 |

一条 Memory 可以组合多个 predicate；所有强制 predicate 都通过后才对目标 HEAD 有效。

### 10.2 对目标 HEAD 的 applicability

Selection 对每条 Record 产生独立决定：

```text
selected
out_of_scope
not_yet_valid
lineage_mismatch
source_missing
source_changed
expired
revoked
superseded
budget_omitted
```

规则如下：

- `valid_from_sha == target_head`：按来源 hash 校验后可用。
- `valid_from_sha` 是 target HEAD 祖先：按 validity policy 重新验证。
- target HEAD 早于 valid-from：`not_yet_valid`。
- 两者不在同一 ancestry：`lineage_mismatch`，不使用。
- 来源缺失或改变：不作为权威记忆注入，并生成 stale/revalidation warning。
- 审查历史分支不会因为与最新主线不同而全局改写 Record 状态；全局 `revalidation_required` 只在明确针对当前项目线执行 revalidation scan 时更新。

### 10.3 Record 生命周期

```text
active
  -> revalidation_required
  -> superseded
  -> revoked
  -> expired
```

- `revalidation_required` 记录只能作为“旧知识可能已失效”的不确定性展示，不能继续作为指令或硬约束。
- revalidate 不原地更新正文，而是生成带新来源与 valid-from 的 Candidate；批准新 Record 后旧 Record 变为 superseded。
- revoke 只由人工执行。
- expiry 可以来自审批时设置的时间/commit 条件，命中后由 Runtime 确定性标记。

### 10.4 自动过期协议

审批时只允许两种 typed condition：

- `at_time`：canonical UTC timestamp；冻结的 Runtime clock 到达该时间即到期。
- `at_commit`：完整 commit SHA；目标 HEAD 等于该 commit 或是其后代即到期。边界 commit 必须存在，并且不能早于 Record 的 `valid_from_sha`。

同一 Record 每种类型最多一个条件；多个条件按 OR 计算，任一命中即到期。`memory approve` 和 `memory revalidate` 使用 `--expires-at`、`--expires-at-commit` 与 `--no-expiry`。Revalidate 默认继承 predecessor 的条件，只有显式 `--no-expiry` 才清除；`--no-expiry` 不能与其他 expiry 参数并用。

Runtime 对 due 的 `active` Record 使用状态 compare-and-swap 写入 `active -> expired` 事件，并在同一事务内递增 generation；重试幂等，并发扫描最终收敛到同一事件/状态，不生成第二个生命周期事实。`read-write` 在 Selection 前先尝试有界持久化 expiry，再冻结 Snapshot generation；扫描失败、超过 512 条或受 snapshot 上限截断时，未落盘的 due Record 仍在内存中确定性排除，并形成 `expiry_sweep_failed`、`expiry_sweep_truncated` 或 `expiry_persistence_deferred` 诊断。`read` 不修改 Store，但同样在固定时钟和目标 HEAD 下排除 due Record；`off` 不读取或评估 Store。Expiry 无法解析时不会继续使用该 Record，并形成可见诊断。

同 revision resume 复用已完成的 Snapshot，不因墙钟推进或 Store 后续 expiry 事件改变输入。新 Review 或 revision-drift child 使用新的冻结时钟、目标 revision 和 generation 重新计算。

## 11. MemorySnapshot 与确定性检索

### 11.1 Snapshot 时机

Session v5 在 `REPOSITORY_INTELLIGENCE` 后运行 `MEMORY_SELECTION`。此时 Runtime 已拥有：

- 精确 Base/Head。
- changed files、diff ranges 和 changed symbols。
- Repository Intelligence 与 Quality Gate 摘要。
- repository identity。

该阶段从持久存储读取一个 generation，完成有效性和初始 scope 判断，并把结果复制为 Session 内不可变 `MemorySnapshot`。

### 11.2 Snapshot 内容

```yaml
snapshot_id: MSNAP-<sha256>
repository_key: ...
base_sha: ...
head_sha: ...
store_schema_version: 2
memory_generation: ...
feedback_generation: ...
knowledge_generation: ...
selection_policy_version: memory_selection_v2
eligible_records: [...canonical record copies...]
applicability_decisions: [...]
feedback_calibration_summary: ...
repository_knowledge_refs: [...]
created_at: ...
snapshot_hash: ...
```

Snapshot 保存足以重放的 Record 正文、scope、source refs 摘要、policy effect 和 applicability 决定，而不是只保存数据库行号。持久数据库在 Review 过程中被批准、撤销或损坏，都不能改变已经固定的 Review 输入。

`memory_selection_v2` 增加 typed expiry 评估与相关诊断。Runtime 仍接受由 `memory_selection_v1` 固定的 legacy selection input/Snapshot，并按其原策略重放；恢复时不会把 v1 artifact 重写成 v2。

### 11.3 检索流程

检索顺序固定为：

1. repository key。
2. Record 状态与 target revision validity。
3. 当前 pipeline stage 允许的 MemoryKind。
4. path/symbol/contract/language scope。
5. typed policy effect 优先级。
6. 确定性 lexical/graph relevance score。
7. stable memory ID tie-break。
8. count、character/token 和 per-kind budget。

可选语义排序器只能重排第 1-5 步已经允许的集合，不能扩展 allowlist，不能把 stale/revoked Record 重新加入，也不能覆盖 stable tie-break 和预算。

### 11.4 快照与动态探索

Snapshot 包含本次 Review 对目标 HEAD 有效的 bounded record pool，而模型消息只收到其中少量 projection。Reviewer 探索到新路径时，可以通过受控 `query_project_memory` 工具查询 Snapshot 内记录；该工具不能回读实时数据库。

工具结果进入 Reviewer 自己的 Observation Store，受调用次数、返回大小和 Context Budget 限制。这样既支持按需探索，也保证恢复时不会看到未来批准的记忆。

### 11.5 Budget

Memory 有独立子预算，不能挤掉 Assignment、Intent、必要代码证据和 Completion Rules。默认 Runtime policy v1：

- Review-level snapshot metadata 有独立磁盘上限。
- 每次模型调用最多选择固定数量的 project memory records。
- message 中 project memory 与 feedback calibration 合计不超过 Context Budget 的 10%。
- typed hard-policy records 不因普通相关性排序被省略；如果 hard-policy 内容自身超过上限，Runtime 必须产生可见错误或阻塞，不能静默截断。
- 所有 omitted record 都保留原因和 ID，便于审计。

## 12. 分阶段使用规则

| 阶段 | 允许使用的记忆 | 用途 | 禁止行为 |
|---|---|---|---|
| Intent Discovery | architecture、business invariant、compatibility | 帮助识别目标、范围和约束，并保留来源 | 不把记忆当成用户本轮 explicit intent |
| Initial Risk | high-risk module、incident lesson、typed risk effect | 形成风险 signal 或本地 risk floor | 不把自然语言风险描述直接变成等级 |
| Portfolio Planning | review rule、verification command、聚合 feedback | 展开 Assignment、required checks 与角色覆盖 | 不让 raw feedback 删除 Reviewer |
| Reviewer | 与 Assignment scope 相关的 active memory、Repository Knowledge | 调查方向、业务约束和验证提示 | 不发送全部项目记忆或其他 Reviewer 推理 |
| Reconciler | business invariant、compatibility、architecture | 判断 Finding 冲突与影响 | 不用 rejected feedback 压制证据充分的 Finding |
| Completion | 仅编译后的 typed required contract/check | 本地确定性覆盖检查 | 自然语言不能修改完成条件 |
| Final Risk | 已应用的高风险与事故知识、已验证 Findings | 解释 residual risk | 不读取未批准 Candidate |
| Memory Curator | 最终已验证结果、显式规则、allowlisted sources | 提出 Candidate | 不批准、不激活、不写 policy authority |

Intent 中由记忆支持的 claim 仍标记为 `inferred` 或相应 repository-derived source。只有用户明确确认后才变成 `explicit`，不能因为 Memory 已获批准就伪装成本轮用户意图。

## 13. Hard Policy 与信息性记忆

### 13.1 默认均为信息性

人工批准只说明该项目知识可以跨 Review 使用，不自动获得修改 Runtime 的权限。普通 statement 即使 kind 为 `review_rule`，也只是有来源的审查上下文。

### 13.2 Typed Policy Effect

需要硬约束时，Candidate 必须包含单独展示并单独批准的结构化 `policy_effect`：

```text
risk_floor(level)
require_contract(contract_id)
require_check(check_id)
verification_hint(command_template_id)
```

Runtime Compiler 规则：

- `contract_id` 必须存在于本地 Review Contract registry。
- `check_id` 必须存在于本地 deterministic check registry。
- `risk_floor` 只能提高本地风险下限，不能降低风险。
- `verification_hint` 只选择本地预注册模板，不能授权任意 shell 字符串。
- 未识别或参数非法的 effect 整条 fail closed，不作为硬策略应用。
- effect 不能增加文件系统权限、网络、工具、命令执行能力、模型预算或外部副作用。
- 模型可以提出 effect，但 Runtime 只接受白名单 schema，最终仍要求人工明确批准。

硬约束的真正执行者是 Runtime、Risk Compiler、Planner 和 Completion Checker，而不是 prompt。

## 14. Review Feedback Memory

### 14.1 Feedback Record

```yaml
feedback_id: FB-<sha256>
repository_key: ...
review_id: ...
finding_id: F-...
head_sha: <full-sha>
finding_snapshot:
  claim: ...
  path: ...
  line: ...
  contracts: [...]
  original_severity: high
  evidence_refs: [...]
  finding_hash: <sha256>
decision: accepted | rejected | severity_changed | missed
original_severity: high
final_severity: medium
reason_code: duplicate | expected_behavior | insufficient_evidence | wrong_scope | severity_mismatch | other
reason: ...
actor: ...
source_refs: [...]
created_at: ...
```

现有 Reconciler 已产生稳定 `finding_id`。Session v5 的 JSON/Markdown Brief 必须继续投影该 ID，Feedback 命令据此校验 Review、revision、canonical Finding 和 evidence refs。

写入 Feedback 时，Runtime 复制最小 canonical Finding 为不可变 `finding_snapshot`。这样 Session 被清理后仍能做聚合和审计；snapshot 不是新的证据来源，必须保留原 review/head、Finding hash 和 evidence refs。对于 `missed`，snapshot 来自人工提交并通过同样的来源校验。

`missed` 记录必须提供人工声明和可验证的代码/Observation 来源，不能只写“模型漏了”。

### 14.2 Feedback 的权限边界

- Feedback 不自动创建 Durable Project Memory。
- raw accepted/rejected 内容不直接放入 Reviewer 或 Reconciler messages。
- rejected Finding 不自动变成 suppression rule。
- Feedback 不能降低 risk floor、移除 required contract、降低 Finding severity 或隐藏新的高危 Finding。
- Feedback 可以用于增加检查视角、暴露 recurring miss、评估角色组合和生成 Eval 样本。

### 14.3 聚合校准

Runtime 只生成带版本和来源计数的聚合摘要。默认 `feedback_aggregation_v1` 至少要求：

- 5 条人工决定。
- 来自至少 3 个不同 Review。
- scope、contract 或 reason taxonomy 可比较。
- 每项聚合保留 record IDs、时间范围和样本数。

聚合结果只能：

- 提高某类检查或 Reviewer perspective 的优先级。
- 提示常见误判原因，要求补充证据。
- 标记 severity calibration uncertainty。

样本不足时只用于 Eval 数据，不进入模型上下文或调度决策。

## 15. Pipeline 与 Session v5

### 15.1 新阶段布局

```text
PREFLIGHT
QUALITY_GATES
REPOSITORY_INTELLIGENCE
MEMORY_SELECTION
INTENT_DISCOVERY
INTENT_RESOLUTION
PLANNING
REVIEWERS
RECONCILIATION_ANALYSIS
SUPPLEMENTAL_INVESTIGATION
RECONCILIATION
COMPLETION
FINAL_RISK
MEMORY_PROPOSAL
REPORTING
```

`MEMORY_SELECTION` 放在 Repository Intelligence 后，是因为 scope 检索需要 changed files/symbols；放在 Intent 前，是因为已批准的架构、业务和兼容性知识可以帮助 Intent inference。

`MEMORY_PROPOSAL` 放在 Final Risk 后、Reporting 前，是因为只有这时才具备最终 verified findings、uncertainties、contract coverage 和 risk；Reporting 又需要展示本轮 pending candidates。

### 15.2 Session schema

- `SESSION_SCHEMA_VERSION = 5`。
- `ReviewExecutionConfig` 新增 `memory` 配置和 `memory_curator: ModelStageConfig`。
- 新 Review 和 revision-drift child Session 使用 v5。
- v1 仍只读审计；v2-v4 保持既有可恢复语义和原阶段列表。
- 恢复 v1-v4 时不创建 MemorySnapshot、不查长期数据库、不调用 Curator，也不改变历史 Brief。
- v5 execution config 与 MemorySnapshot 均是 Session 可审计输入。

`MemoryExecutionConfig` 使用最终字段：

```yaml
mode: off | read | read-write
root_path: <canonical absolute path>
required: false
selection_policy_version: memory_selection_v2
feedback_policy_version: feedback_aggregation_v1
max_snapshot_records: 2000
max_snapshot_bytes: 8388608
max_context_records: 12
max_query_results: 8
```

配置在创建 Session 时解析并固定，resume 不重新读取环境变量来改变它。`root_path` 只保存在本地 Session，不发送给模型；API key 仍只保存环境变量名。`required=true` 与 `mode=off` 是非法组合。

### 15.3 Memory mode

```text
off        不读长期记忆，不读写跨运行 cache，不生成 Candidate
read       读取 approved memory/cache；cache miss 只写 Session；不持久化 Candidate
read-write 读取并写 revision cache，生成并持久化 pending Candidate
```

默认是 `read-write`。该模式只会产生 cache、pending Candidate 和审计事件，永远不会自动产生 active Memory。需要 hermetic 或隐私隔离的运行使用 `read` 或 `off`。Feedback 只由独立人工命令写入，不会因为 Review 使用 `read-write` 而自动推断人工决定。

v5 的两个 Memory phase 始终存在，以保持单一阶段布局。`off` 模式的 Selection 提交带 `disabled` 原因的空 Snapshot，Proposal 提交 `skipped` decision 和空 candidate batch；`read` 模式正常 Selection，确定性排除 due Record 但不把 expiry 写回 Store，Proposal 同样不调用 Curator、不创建 outbox；`read-write` 先尝试有界提交 due expiry，再冻结新的 generation 和 Snapshot，未落盘项仍被排除并产生诊断。

### 15.4 Memory Curator

`memory_curator` 复用统一 Model Adapter 和 `ModelStageConfig`：

- local mode 是默认模式。
- local curator 只从本轮显式 `--project-rule`、人工声明和 Runtime 已验证的 typed source 中构造确定性 Candidate；没有合格来源时返回空集合。
- model mode 使用无工具、单轮、严格结构化输出；输入仅包含最终 verified artifacts、最小必要 source excerpts、现有 active/pending fingerprints 和候选 schema。
- 模型返回值只是 proposal，必须经过同一 Candidate Validator。
- Provider、解析或预算失败不影响 Review 结论，只形成可见的 memory proposal warning。

### 15.5 Proposal outbox

`MEMORY_PROPOSAL` 先把 canonical candidates 和 `memory_outbox.json` 提交为 Session artifact，再幂等写入长期数据库。这样可以关闭以下 crash window：

- Session 已产生 Candidate，但数据库尚未写入。
- 数据库已写入，但 phase 尚未标记 completed。
- Provider 重试或 resume 重复提交同一 Candidate。

数据库写失败时 Review 继续，Reporting 显示“Candidate pending local replay”。`memory replay-outbox <review-id>` 可以在之后重放；candidate ID 和 request ID 保证幂等。`memory_persistence_receipt.json` 只有在每项 Candidate 的终态、dedupe decision 与 replay/write 标志组成合法成功矩阵时才代表持久化成功；空结果或中间状态不能冒充成功。

## 16. Context 与模型调用集成

### 16.1 新 Context Sections

Reviewer Context 增加三个按需 section：

- `Approved Project Memory`
- `Repository Knowledge`
- `Feedback Calibration Summary`

每条 Project Memory 必须带：

- memory ID。
- kind 与 scope。
- authority label：`human_approved_context` 或 `runtime_compiled_policy`。
- source ref 摘要。
- target revision validity。

模型必须能区分：

```text
Runtime policy / compiled hard rule
human-approved informational memory
repository-derived fact
current user explicit intent
model-inferred claim
```

### 16.2 Budget 与 Compaction

- `system`、`tools`、`parameters` 仍不计入 messages budget。
- Memory 使用 messages 中独立的 10% 子预算。
- required core sections 不得被 Memory 挤出。
- Compactor 优先保留 memory ID、statement、scope、authority 和 source refs，压缩重复说明。
- 被省略或压缩的 ID 写入 envelope metadata。
- `ModelInvocationEnvelope.parameters.context` 记录 snapshot ID、selected memory IDs、record hashes、selection policy 和 omitted reasons。

### 16.3 Memory 工具

Reviewer 可选工具 `query_project_memory` 的输入只能是：

- path/symbol/contract。
- 当前 Assignment ID。
- bounded query text。

Runtime 只查询 Session 固定的 Snapshot，返回结构化记录，并将调用写为 Observation。工具不能列出 revoked/stale/pending Candidate，不能读 Memory DB 文件，不能修改任何状态。

## 17. CLI 与人工工作流

### 17.1 Review 参数

```text
review --memory-mode off|read|read-write
       --memory-root <path>
       --memory-required
       --memory-curator-mode local|model
       --memory-curator-provider ...
       --memory-curator-model ...
       --memory-curator-base-url ...
       --memory-curator-api-key-env ...
       --memory-curator-max-output-tokens ...
       --memory-curator-max-provider-attempts ...
       --memory-curator-max-elapsed-seconds ...
```

`--memory-required` 表示 Memory Store、有效性检查或 hard-policy snapshot 无法完成时阻塞 Review；默认行为是继续并形成明确 uncertainty/manual-review 建议。

### 17.2 管理命令

```text
memory status
memory list
memory show <memory-id>
memory candidates
memory candidate show <candidate-id>
memory approve <candidate-id> --actor ... --reason ...
memory reject <candidate-id> --actor ... --reason-code ... --reason ...
memory revoke <memory-id> --actor ... --reason ...
memory revalidate <memory-id> --actor ...
memory feedback record ...
memory feedback list
memory export <path>
memory import <path> --dry-run
memory replay-outbox <review-id>
memory gc --dry-run
memory relink ...
```

`approve` 与 `revalidate` 还接受：

```text
--expires-at <canonical-utc-timestamp>
--expires-at-commit <revision>
--no-expiry
```

commit 参数在预览时固定为完整 object ID，提交前再次验证 repository authority 与 expiry 条件没有漂移。Revalidate 默认继承 predecessor expiry；显式 `--no-expiry` 才清除。

所有命令接收 `--repo` 和 `--memory-root`。修改类命令必须使用 actor；交互模式显示 diff 并询问确认，脚本模式必须额外提供 `--yes`。任何管理命令都不能通过自由文本改变工具权限。

## 18. Session Artifacts 与 Brief

### 18.1 Memory Selection artifacts

```text
memory_selection_input.json
memory_snapshot.json
memory_selection_decision.json
memory_feedback_summary.json
```

### 18.2 Memory Proposal artifacts

```text
memory_curator_envelope.json       # model mode only
memory_curator_raw_response.json   # model mode only
memory_curator_decision.json
memory_candidates.json
memory_outbox.json
memory_persistence_receipt.json    # write succeeds when present
```

Artifact schema 固定为：

| artifact | schema |
|---|---|
| `memory_selection_input.json` | `memory_selection_input_v1` |
| `memory_snapshot.json` | `memory_snapshot_v1` |
| `memory_selection_decision.json` | `memory_selection_decision_v1` |
| `memory_feedback_summary.json` | `feedback_calibration_summary_v1` |
| `memory_curator_envelope.json` | `memory_curator_envelope_v1` |
| `memory_curator_raw_response.json` | `memory_curator_raw_response_v1` |
| `memory_curator_decision.json` | `memory_curator_decision_v1` |
| `memory_candidates.json` | `memory_candidate_batch_v1` |
| `memory_outbox.json` | `memory_candidate_outbox_v1` |
| `memory_persistence_receipt.json` | `memory_persistence_receipt_v1` |

所有 artifacts 绑定 Session Base/Head revision、具有 schema、hash 和 phase descriptor。它们不保存 API key、隐藏 reasoning 或敏感原文。

`memory_curator_raw_response.json` 是经过 schema-aware secret scan 和 redaction 后的 provider response，不是未处理的网络字节。若安全扫描无法生成可审计的脱敏结果，Runtime 只保存响应 hash、拒绝原因和调用 metadata，并拒绝该批 Candidate。

### 18.3 Review Brief

JSON 与 Markdown Brief 新增：

- Applied Memory：ID、kind、scope、authority、source refs。
- Runtime-compiled policy effects。
- Repository Knowledge cache provenance。
- stale/revalidation/lineage warnings。
- Feedback calibration summary 的版本和样本数。
- Pending Memory Candidates 及审批命令提示。
- memory unavailable、outbox pending 和降级原因。
- 每个 canonical Finding 的稳定 `finding_id`。

Brief 不内嵌完整 Memory DB、原始 Feedback 或大型 cache blob。

## 19. 失败、恢复与并发

### 19.1 Store 不可用或损坏

- `off`：按配置运行，不产生 warning。
- 空 store：等同没有历史记忆，不是错误。
- store 无法打开、schema 不支持或 hash 校验失败：不应用任何实时记录，形成 `memory_unavailable` uncertainty。
- 默认继续完成 Review，但非约束性建议至少为 `manual_review`；`--memory-required` 时阻塞。
- 已提交到 Session 的有效 MemorySnapshot 在数据库随后不可用时仍可使用。
- 不允许使用“上一次可能有效”的内存对象或未校验 JSON 作为静默 fallback。

### 19.2 Cache failure

- Cache miss/corruption 时重新运行 Repository Intelligence。
- `read` 模式只把新结果写入 Session。
- 必需分析器也失败时沿用现有 Repository Intelligence 降级规则并降低 confidence。
- Cache 本身从不决定 Review 是否 completed。

### 19.3 Resume

- `MEMORY_SELECTION` 已 completed：先验证 artifact hash，再加载原 Snapshot，不访问最新 generation。
- 已完成 Snapshot 的冻结时钟与 expiry 决定一并复用；同 revision resume 不重新按当前时间过期记录。
- Selection artifact 损坏：从 `MEMORY_SELECTION` 起失效并重跑下游阶段。
- `MEMORY_PROPOSAL` 中断：保留已提交 outbox，重启 attempt 后幂等重放。
- 已 completed Session 不追加 Candidate 或 Feedback；这些外部动作通过 memory 管理命令形成独立事件。

### 19.4 Revision drift

Revision drift 创建 v5 child Session，从 Preflight 重新执行 Repository Intelligence 和 Memory Selection。父 Session 的 Snapshot 只作为 lineage 审计引用，不能复制为新 HEAD 的权威输入。

### 19.5 并发

- 多个 Review 可以并行读同一数据库。
- Cache 和 Candidate 写入按 stable key 幂等。
- approve/reject/revoke/revalidate 使用事务与 generation compare-and-swap。
- 同一 Candidate 的并发审批最多一个成功；actor、reason、reason code、expiry、predecessor 等完全相同的同决策 loser 保存独立、可重放的 no-op request receipt，返回 winner 的最终状态，不产生第二条审批事件、第二个 Record 或覆盖 attribution。不同决策、不同 expiry 或相同 request ID 的冲突 payload 仍拒绝。
- 并发 expiry 扫描使用同样的 CAS/幂等语义，最终只保留一个 `active -> expired` 生命周期事实。
- migration 与 relink 获取 repository-level exclusive lock。

## 20. 安全与隐私

### 20.1 信任顺序

```text
Runtime Policy / permissions / Review Contract
Runtime-compiled, human-approved typed policy effect
human-approved informational project memory
current user explicit intent
repository-derived facts and repository content
model proposals and external/generated content
```

即使已人工批准，Memory statement 也不能覆盖 system prompt、工具 allowlist、文件边界、网络策略、预算、response schema 或 Completion Checker。

### 20.2 敏感信息

- `.git`、`.env`、credential、private key、token、cookie、认证 URL 和匹配 secret scanner 的内容禁止进入 Candidate/Record/Feedback 正文。
- source ref 尽量保存 hash 和定位，不复制完整敏感内容。
- `sensitivity=local_only` 的记录可以供本地确定性 Runtime 使用，但不得发送给远端模型 Provider。
- `sensitivity=blocked` 的 Candidate 立即拒绝且正文不落盘。
- export 默认脱敏，并附 manifest hash。
- Memory Root 使用当前用户可访问权限；权限无法收紧时产生 warning。

### 20.3 Prompt injection

Repository Knowledge、Memory statement、Feedback reason 和 source excerpt 都作为不可信数据呈现。Context 明确标注 authority，Runtime 不解析其中的自然语言命令来扩大权限。

### 20.4 Migration safety

升级数据库时：

1. 获取独占锁。
2. 创建校验过的 backup/staging copy。
3. 在 staging 上执行 migration transaction。
4. 运行 foreign-key、schema、event-chain 和 blob reference 检查。
5. 原子替换权威数据库。

失败时保留旧数据库，不允许半迁移继续写入。新代码可以在明确的只读审计模式查看旧版本，但不能猜测字段语义。

## 21. 模块设计

新增模块：

```text
memory_models.py       enums、Candidate、Record、Feedback、Snapshot、SourceRef
memory_identity.py     Memory Root 与 repository key
memory_store.py        SQLite、migration、events、transactions、export/import
memory_sources.py      source allowlist、revision/hash/sensitive validation
memory_retrieval.py    applicability、scope、ranking、budget、snapshot
memory_policy.py       typed policy effect compiler
memory_curator.py      local/model proposal、strict parser、dedupe
memory_feedback.py     feedback validation 与安全聚合
repository_cache.py    immutable revision cache 与 blob lifecycle
```

需要修改：

```text
run_state.py
session.py / session_store.py / hydration.py / resume.py
pipeline.py
repository_intelligence.py
context.py
models.py / evidence.py / reconciler.py
brief.py / reporting.py
command.py
artifacts.py / attempts.py
```

模块依赖方向固定为：

```text
models/identity
    -> store/sources/cache
    -> retrieval/policy/feedback/curator
    -> pipeline/context/brief/CLI
```

Store 和 Validator 不依赖 Pipeline 或模型 Provider，避免业务逻辑直接绑定模型 API。

## 22. 实现批次

这些批次只是最终架构的依赖拆分，不是“先做一个以后丢掉的简单版”。

### Batch A：持久化内核与 Repository Cache

- canonical model/artifact v1、Record v1/v2、Store/export v2、repository identity 和 storage root。
- SQLite migrations、event log、blob store、并发与 corruption handling。
- Candidate/Record/SourceRef/Feedback/Snapshot 完整模型。
- source validator、审批状态机、dedupe、revoke/revalidate。
- Repository Knowledge immutable cache。
- memory 管理 CLI 的读写、审批、export、GC 基础能力。

### Batch B：Session v5、Selection 与 Runtime 集成

- 新 RunPhase 和 schema-specific phase layout。
- MemoryExecutionConfig 与 Snapshot artifacts。
- deterministic selection、validity、scope 和 budget。
- typed policy compiler。
- Intent/Risk/Planner/Reviewer/Reconciler/Completion/Final Risk projections。
- `query_project_memory`、Context metadata、resume 与 revision drift。
- Brief 的 Applied Memory 与降级信息。

### Batch C：Curator、Feedback 与端到端加固

- local/model Memory Curator 和统一 Model Adapter 配置。
- strict candidate parser、proposal phase、outbox/replay。
- Feedback CLI、stable Finding ID projection 和 aggregation policy。
- Pending Candidate/Feedback calibration Brief 输出。
- migration、并发、corruption、安全、legacy Session 和端到端测试。

每个 Batch 合并时都必须保持全量测试通过；尚未接入的最终字段可以处于未使用状态，但不得使用与最终 schema 不兼容的临时字段或存储格式。

## 23. 测试策略

### 23.1 数据模型与存储

- stable ID、canonical serialization 和 content fingerprint。
- schema migration、rollback、event hash、foreign keys。
- blob 原子写、dedupe、缺失、tamper 和 GC pin。
- concurrent read/write、重复 outbox、并发审批。
- repository key、worktree 共享和 clone 隔离。

### 23.2 来源与安全

- Base/Head range、symbol、Observation、artifact 和 human declaration 校验。
- wrong revision、line drift、content hash mismatch、missing commit。
- absolute path、traversal、symlink、`.git`、`.env` 和 secret rejection。
- prompt injection 不能创建 policy effect 或改变权限。
- `local_only` 不发送给远端 Provider。

### 23.3 生命周期

- proposed/validated/pending/approved/rejected 转移。
- active/revalidation-required/superseded/revoked/expired 转移。
- at-time/at-commit OR expiry、commit boundary、READ/READ_WRITE、幂等与并发收敛。
- duplicate、re-proposal 和 changed evidence。
- ancestor、historical HEAD、diverged branch 和 revision drift applicability。
- revalidate 创建新 Record，不修改旧正文。

### 23.4 检索与 Runtime

- stage/kind/scope allowlist。
- deterministic ranking 和 stable tie-break。
- memory budget 不挤掉 required Context sections。
- hard policy 不被普通 compaction 省略。
- semantic ranker 不能扩大候选集合。
- policy effect 只能提高风险或增加已注册 contract/check。
- query tool 只访问 Snapshot。

### 23.5 Feedback

- Finding ID 与 Session artifact 校验。
- accepted/rejected/severity/missed schema。
- 样本阈值与 distinct-review 计数。
- raw feedback 不进入 Reviewer Context。
- rejected feedback 不能抑制新 Finding 或降低 severity/risk。

### 23.6 Pipeline、恢复与兼容

- 新 v5 happy path、local/model curator、empty memory、read/off 模式。
- store unavailable、corrupt cache、candidate write failure 和 outbox replay。
- Memory Selection resume 固定旧 generation。
- selection policy v2 与 v1 Snapshot 兼容；同 revision resume 固定旧 expiry 决定。
- Snapshot tamper 从正确 phase 失效。
- revision drift child 重新 selection。
- v1 只读；v2-v4 按原阶段恢复且不调用 Memory。
- JSON/Markdown Brief parity。

## 24. 验收标准

完成本设计的全部实现后，必须满足：

1. 两次针对同一 repository identity 的 Review 可以复用精确 revision 的 Repository Knowledge；HEAD 改变时旧事实不会被当成新事实。
2. 模型提出的 Candidate 在人工批准前，不能进入任何后续 Review 的权威上下文或 Runtime policy。
3. 每条 active Project Memory 都能追溯到通过 hash/revision 校验的来源和人工审批事件。
4. Memory 对目标 HEAD 失效、来源改变或 lineage 不匹配时不会静默使用。
5. 同 revision resume 使用原 MemorySnapshot；数据库中途变化不会改变结果。
6. revision drift child Session 重新计算 Snapshot。
7. 普通自然语言记忆不能扩大权限、增加工具、执行任意命令或绕过 Completion Checker。
8. typed policy effect 只能调用本地白名单 compiler，并由 Runtime 执行。
9. Feedback 不会自动形成 suppression rule，不能删除证据充分的新 Finding。
10. Candidate persistence 失败不会丢失本轮 proposal，也不会伪装成功；outbox 可幂等重放。
11. Memory Store 不可用时行为可配置、降级可见，不会使用未验证 stale 数据。
12. v1-v4 Session 的审计和恢复语义保持不变。
13. JSON/Markdown Brief 能解释用了哪些 Memory、为何适用、哪些被排除，以及有哪些 Candidate 待审批。
14. 全量单元、集成、安全、恢复和并发测试通过。
15. 时间/commit expiry 在 read-write 中先尝试有界落盘再冻结 Snapshot；失败或截断时仍确定性排除并报告诊断。在 read 中不写 Store 但同样排除，并在并发和 resume 下保持单一、可重放结果。

## 25. 后续关系

本设计已经实现；主 Spec 第 15 节对应的 Durable Project Memory、Repository Knowledge Cache 和 Review Feedback Memory 已落地，并通过 Memory 定向、跨模块集成和项目全量回归。

之后剩余的独立工作仍是：

1. Eval Harness：构建 case、grader、回归与模型/架构对比。
2. GitHub/PR 产品集成：拉取 PR、发布 Review 结果、接收人工反馈和远端状态。

GitHub/PR 集成未来可以把远端人工决定转换为本设计中的 Feedback/Candidate 输入，但仍必须经过身份校验、source validation 和本地审批边界，不能绕过本设计的状态机。

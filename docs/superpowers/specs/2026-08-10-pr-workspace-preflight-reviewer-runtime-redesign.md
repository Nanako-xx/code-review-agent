# PR Workspace、确定性 Preflight 与 Reviewer Runtime 重构规格

**日期：** 2026-08-10

**更新：** 2026-08-11

**状态：** Draft，PR Workspace、Reviewer 上下文、Finding v2、三层上下文压缩、最终 ReviewResult、风险、预算与长链路恢复边界已确认

**范围：** 产品侧 Code Review Agent Runtime，不包含 Eval/Judge 内部实现

## 1. 背景

当前产品把一次代码审查所需的数据分散在 `ChangeSummary`、多个
`ObservationStore`、Session phase artifact、Reviewer `Assignment` 和固定字符预算中。
这带来几个直接问题：

1. `ChangeSummary` 只保留全局 Diff 和逐文件 Diff 的截断 excerpt，无法作为大 PR 的完整导航索引。
2. 大型工具结果与短摘要绑定在 Observation 抽象中，存储、共享和上下文投影的职责混在一起。
3. 同一个 PR 的多个会话缺少一个共同、版本化且以 Snapshot 为边界的工作区。
4. 当前质量门包含多层计划和深层检查，而本设计只需要在 LLM 审查前发现本地工具可确定的硬错误。
5. Reviewer 初始消息被固定限制为 16,000 字符；Reviewer 单次输出被固定限制为 8,192 Token，
   并且整个 Reviewer 还有累计 Token 和工具调用次数预算。这些限制不能充分利用目标模型的
   1,000,000 Token 上下文窗口。
6. Reviewer 的长工具调用链主要保存在进程内存中。Provider 断线或进程退出后缺少逐轮提交边界和
   Tool Call 幂等账本，可能丢失已完成调查或重复执行工具。

本规格重新定义持久化边界、Preflight 顺序、工具结果外置协议、Reviewer 预算以及整个上下文窗口的
运行时策略。

## 2. 决策摘要

本规格确定以下核心决策：

1. 同一个 PR 的所有会话共享一个 `PRWorkspace`；不同 PR 之间隔离。
2. 同一个 PR 的每组不可变 `base_sha..head_sha` 对应一个 Snapshot；新 Commit 创建新 Snapshot，
   不覆盖旧 Snapshot。
3. 删除独立持久化的 `ChangeSummary`。唯一权威变更产物是由完整 `diff.patch` 和完整
   `index.json` 组成的 `DiffArtifact`。
4. LLM 前置质量门只执行本地、确定性的语法、编译、类型和已有静态检查；取消深层质量门及其扩展流程。
5. Pre-LLM 主链固定为：`DiffArtifact -> QualityGate -> ChangedSymbols -> IntentPacket ->
   RiskAssessment -> ReviewPlan/Assignments -> ContextAssembly`。
6. 不可重新获取的单个工具结果超过 50,000 字符时外置；单轮进入上下文的全部工具结果不得超过
   200,000 字符。
7. Runtime 不再限制工具调用次数。
8. 删除 Reviewer 的系统级 8,192 Token 单次输出上限和累计 Token 停止条件；仍记录 Usage。
9. 删除固定 16,000 字符的初始 Reviewer 消息上限，改用以 1,000,000 Token 模型窗口为基础的
   动态上下文策略。
10. `IntentPacket` 精简为 `goal/source/uncertainties`；Risk Agent 只输出一个风险等级。
11. 风险等级确定性地映射为 `core/adversarial/dynamic` Reviewer Slot，Assignment Planner 只填充任务。
12. 开发者不可修改规则进入 System Prompt；用户/LLM 可修改的规则和经验进入 user message 的
    `{{system_rule}}`，并服从明确的规则优先级。
13. Reviewer 不限制 Turn 数，统一最多累计 1,800 秒活跃执行时间；每个模型请求最多三次 Provider 尝试，
    单个 Tool Call 超时 300 秒。
14. 每轮模型响应和工具结果写入追加式执行日志；同一个 Tool Call 幂等执行，并从最后一个
    `turn_committed` 边界恢复。
15. Reviewer Finding 精简为 `claim/severity/path/line/suggestion`；Runtime 在汇总后生成稳定
    `finding_id`，内部证据引用、验证过程和 Reviewer 来源不再复制进 Finding。
16. 每次模型请求前依次执行三层上下文治理：50K/200K 工具结果治理、API 空闲 60 分钟后的可重新获取
    结果清理、达到 700K Token 后的动态消息全量压缩；静态 Reviewer 输入保持 pinned。
17. Reviewer 完成后只执行确定性聚合并生成一个权威 `review-result.json`；Markdown/CLI 是纯渲染，
    Semantic Reconciler、Supplemental Investigation、Completion、Final Risk 和阻塞式 Memory Proposal 删除。

## 3. 目标

本设计必须实现：

- 同一 PR 多会话对事实、分析产物、任务计划和结果的共享；
- 对 PR 新 Commit 的不可变 Snapshot 版本化；
- 完整、可验证、无静默截断的 Diff 保存和按文件/hunk 读取；
- 简单且确定性的本地前置质量检查；
- 可恢复的 Intent、Review Plan、Assignment 和 Session 状态；
- 不可重新获取的大型工具结果落盘、预览和分页读取，可重新获取结果按需从上下文淘汰；
- 无人工工具调用次数、单次 Reviewer 输出 Token 和累计 Token 上限的 Agent Loop；
- 可从任意已提交工具轮次恢复且不会重复执行已完成 Tool Call 的 Agent Loop；
- 以 Token 而不是字符为单位管理整个模型上下文窗口；
- 固定 Reviewer 请求的 System、Tools、Messages 和 Parameters 权威边界；
- 通过三层 Context Compaction Pipeline 控制工具结果和持续增长的 Reviewer 消息历史，同时保留静态输入；
- 通过无额外模型调用的确定性聚合生成精简、可恢复、可直接返回的最终 ReviewResult。

## 4. 非目标与延期决策

本规格暂不定义：

- Compaction Summary 的最终 Prompt 文案和 Provider 专用 Prompt Cache TTL 探测；第一版统一使用 60 分钟启发式；
- 长期审查经验的自动晋升、降权和淘汰策略；
- 语义型或模型驱动的 Finding 模糊合并；第一版只合并规范化身份完全相同的重复项；
- 小 PR 与大 PR 切换完整 Diff 和 Diff Index 的最终阈值；
- Assignment Planner 为每个 Slot 选择文件、Symbol 和 hunk 的具体算法；
- 新的通用 Shell、网络或仓库写入能力；
- Eval/Judge 的上下文和资源预算重构。

## 5. 核心术语与边界

### 5.1 PR Workspace

`PRWorkspace` 是同一个 PR 的所有会话共享的持久化工作区。它不是整个仓库的共享状态，也不是全局长期记忆。

PR 身份必须至少绑定：

```text
provider/repository identity + PR number
```

没有平台 PR 编号的本地或 Benchmark 场景必须使用稳定的外部 Review/Task ID 与规范化仓库身份生成
等价 `pr_id`。

### 5.2 Snapshot

Snapshot 由解析后的不可变 Commit SHA 对定义：

```text
snapshot_id = stable_id(repository_identity, pr_id, base_sha, head_sha)
```

分支名只用于解析，不得作为 Snapshot 的最终身份。所有 Diff、ChangedSymbols、质量门结果、Assignment
和 Findings 都必须声明自己的 `snapshot_id`。

### 5.3 规则权威与 Global Memory

开发人员配置且用户不可修改的 `DeveloperReviewPolicy` 直接进入 System Prompt。用户或 LLM 可修改的
全局审查规则与经验保存在 `GlobalMemoryStore`，并投影到初始 user message 的 `{{system_rule}}`。

规则优先级固定为：

```text
Runtime safety and output contract
> DeveloperReviewPolicy
> 当前用户请求与澄清
> 用户编写的全局审查规则
> LLM 生成的审查经验
> Repository/Diff/Tool Result 数据
```

发生冲突时只忽略冲突的低优先级规则，保留其余不冲突部分。开发者规则、用户规则和 LLM 经验不得
混入同一个 System 权威区块；`{{system_rule}}` 虽然名称包含 system，但其实际权限始终是 user 级。

`GlobalMemoryStore` 位于 `PRWorkspace` 之外，可以带全局、语言、框架或仓库等作用域元数据，但不得
直接保存某个 PR 的 Diff、Intent、Assignment 或临时工具输出。

### 5.4 Session Context

Session Context 是当前 Reviewer 调用实际发送给模型的短期内容。它是 `PRWorkspace` 和
`GlobalMemoryStore` 的动态投影，不是权威数据源。

上下文压缩只改变投影，不得删除或篡改权威 Artifact。

## 6. 持久化结构

规范结构如下：

```text
PRWorkspace
├─ manifest.json
│  ├─ pr_id
│  ├─ current_snapshot_id
│  ├─ current_intent_version
│  └─ workspace_schema_version
│
├─ PR
│  └─ pr.json
│     ├─ repository_identity
│     ├─ provider
│     ├─ pr_number_or_external_review_id
│     ├─ title
│     ├─ description
│     ├─ base/head refs
│     ├─ author
│     └─ status
│
├─ Intent
│  ├─ current.json
│  └─ history
│     └─ <intent-version>.json
│
├─ Snapshots
│  └─ <snapshot-id>
│     ├─ snapshot.json
│     ├─ DiffArtifact
│     │  ├─ diff.patch
│     │  └─ index.json
│     ├─ QualityGate
│     │  └─ quality-gate.json
│     ├─ ChangedSymbols
│     │  └─ changed-symbols.json
│     ├─ Risk
│     │  └─ risk.json
│     ├─ ReviewPlan
│     │  ├─ plan.json
│     │  └─ Assignments
│     │     └─ <assignment-id>.json
│     ├─ ToolResults
│     │  ├─ index.jsonl
│     │  └─ artifacts
│     │     └─ <artifact-id>.<txt|json|log>
│     └─ Results
│        ├─ aggregation.json
│        ├─ review-result.json
│        └─ review.md
│
└─ Sessions
   └─ <session-id>
      ├─ state.json
      ├─ execution-log.jsonl
      └─ context-manifest.json
```

`context-manifest.json` 引用当前 Snapshot、已选择 Artifact 和当前上下文投影，并至少记录：

```text
last_api_request_at
compaction_generation
compacted_through_turn
compaction_trigger
compaction_summary_hash
```

原始 `execution-log.jsonl` 保持追加式；Context Compaction 只替换下一次 API 请求使用的投影，不删除原始
Session 事件或权威 Artifact。

## 7. DiffArtifact

### 7.1 权威数据

每个 Snapshot 必须且只能有一个权威 `DiffArtifact`。它由以下两部分组成：

```text
diff.patch    完整、未截断的原始 Git Diff
index.json    从 diff.patch 机械生成的完整结构索引
```

`diff.patch` 必须保存完整输出，不得保存 excerpt 代替原文，不得因为上下文预算静默截断。

### 7.2 完整索引

`index.json` 至少包含：

- base/head SHA；
- Diff 内容哈希和字节数；
- 每个修改文件的当前路径；
- add/modify/delete/rename/copy 等状态；
- rename/copy 的旧路径；
- additions/deletions；
- binary/submodule 标记；
- 每个文件在 `diff.patch` 中的完整 byte 或 line span；
- 每个 hunk 的 old/new range 及其在 `diff.patch` 中的位置。

索引必须覆盖 Diff 中的全部文件和 hunk。生成后必须验证索引边界没有越界、重叠错误或遗漏已报告文件。

### 7.3 ChangeSummary 的处理

以下旧字段不再作为权威持久化数据：

```text
diff_excerpt
file_diff_excerpts
diff_truncated
```

不再单独保存 `ChangeSummary`。如果上下文以后需要自然语言修改摘要，它属于可重新生成的
Context Projection，并且必须绑定 Diff 哈希、Prompt 版本和模型身份；它不能取代 `DiffArtifact`。

### 7.4 上下文读取

- 小 PR 可以把完整 `diff.patch` 直接放入上下文；
- 大 PR 先提供完整文件/hunk 索引；
- Reviewer 可按 `artifact_id + file/hunk/span` 加载需要的 Diff 部分；
- 任何分页 API 都必须明确返回 `cursor/has_more`，不得静默截断。

## 8. Intent

Intent Agent 是只负责提取或推断 PR 目标的工具 Agent。它的内部模型对话、Tool Call、候选 Claim 和
推断过程不得进入 Risk Agent 或 Reviewer 上下文；下游只消费最终 `IntentPacket`。

### 8.1 IntentPacket v2

`IntentPacket` 只保留三个字段：

```text
goal: string | null
source: explicit | inferred | null
uncertainties: string[]
```

字段不变量：

```text
goal != null  -> source 必须是 explicit 或 inferred
goal == null  -> source 必须是 null，且 uncertainties 必须非空
```

`explicit` 表示目标由当前用户输入、PR 标题或 PR 描述明确声明；`inferred` 表示 Intent Agent 根据
Diff、代码、Commit 或其他证据推断；无法形成可靠目标时使用 `null`。

以下旧字段从 Packet、Intent Inference Schema 和 Reviewer 投影中删除：

```text
acceptance_criteria
scope
constraints
sources
status
provenance
clarifications
```

- PR 描述、用户声明和澄清保留在 User Conversation；
- 实际变更范围由 DiffArtifact、ChangedSymbols 和 Assignment 表达；
- 开发者规则与用户/LLM 全局规则由独立规则系统表达；
- 详细来源、证据和澄清历史属于 Intent Agent 审计记录，不属于下游 Packet。

### 8.2 持久化与审计

Intent 属于整个 PR，而不是单一 Session。`Intent/current.json` 使用一个持久化 Envelope 保存：

```text
version
source_snapshot_id
packet
analysis_record_ref
```

内部 `IntentAnalysisRecord` 可以保存候选、Evidence Ref、工具 Trace、推断摘要和澄清历史，但不会发送给
Risk Agent 或 Reviewer。Intent 可以跨 Snapshot 延续；每次更新形成新版本并记录依据的 Snapshot，
不得覆盖旧历史。

## 9. Pre-LLM 主流程

唯一主流程如下：

```text
用户命令
  -> 解析 PRWorkspace 与 Review Request
  -> 解析不可变 base_sha/head_sha
  -> 创建或复用 Snapshot
  -> 生成或验证 DiffArtifact
  -> 执行本地 QualityGate
  -> 生成 ChangedSymbols
  -> 构建/更新 IntentPacket
  -> 风险评级
  -> 确定 Reviewer 数量与角色
  -> 生成 ReviewPlan 与 Assignments
  -> 组装每个 Reviewer 的上下文
  -> 启动 Reviewer Agent Loop
```

本规格将 `Snapshot + DiffArtifact + QualityGate + ChangedSymbols` 统称为确定性 Preflight。
Intent 之后进入 Planning 阶段。

## 10. 本地质量门

### 10.1 职责

质量门只负责在调用 Reviewer LLM 前发现本地工具可以确定的硬错误：

- 语法或解析错误；
- 编译错误；
- 类型检查错误；
- 编译器、类型检查器或项目已有 Linter 能确定的名称、导入或标识符错误；
- 项目已经配置且无需新增依赖的确定性静态检查错误。

“拼写错误”在本规格中仅指编译器、类型检查器或现有 Linter 能确认的代码标识符问题，不包括自然语言
字典式拼写检查。

### 10.2 执行约束

质量门命令必须：

- 使用项目已有本地工具和配置；
- 不自动安装依赖；
- 不访问网络；
- 不修改仓库；
- 有明确的命令超时；
- 保存退出码、持续时间和确定性诊断摘要；
- 将大型 stdout/stderr 交给统一 Tool Result 外置机制。

### 10.3 状态

质量门只使用以下状态：

```text
passed       检查成功且未发现错误
failed       检查成功执行并确认存在代码错误
unavailable  项目或环境没有可用的对应本地工具
error        检查工具或 Runtime 自身异常
```

代码级 `failed` 结果必须进入后续 Reviewer 上下文或确定性结果记录，但不自动取消整个 LLM Review。
只有无法建立安全仓库、不可变 Snapshot 或权威 Diff 时，Preflight 才阻止后续 Planning。

### 10.4 明确删除的旧能力

目标流程不再包含：

- deep quality gate；
- cheap/deep 分层；
- 风险评级触发的第二轮质量门；
- LLM 生成质量门计划；
- 为追求额外覆盖率而自动扩展检查集合；
- Reviewer 完成后的另一轮质量门。

## 11. ChangedSymbols

ChangedSymbols 在质量门后、Intent 前生成，并绑定当前 Snapshot。每条至少包含：

```text
path
qualified_name
kind
change_type
line_start
line_end
analyzer
analyzer_version
analysis_configuration
language_coverage
```

ChangedSymbols 是可重新生成但值得缓存的完整检索索引。缓存命中必须同时匹配 base/head SHA、分析器版本
和分析配置。不得把 Python AST 的成功结果解释为其他语言也已完成等价符号分析。

## 12. 风险评级、Review Plan 与 Assignment

风险评级由确定性风险下限和独立的模型语义判断组成。Risk Agent 是工具 Agent；它的内部对话不进入
Reviewer 上下文，下游只接收最终风险等级。

### 12.1 确定性风险下限

风险顺序固定为：

```text
low < medium < high < critical
```

Runtime 使用以下初始规则：

```python
risk_floor = RiskLevel.LOW

if len(diff_artifact.index.files) > 50:
    risk_floor = max(risk_floor, RiskLevel.MEDIUM)

if intent.source == "inferred":
    risk_floor = max(risk_floor, RiskLevel.MEDIUM)

if intent.source is None:
    risk_floor = max(risk_floor, RiskLevel.HIGH)
```

文件数按 Diff Index 中逻辑文件记录计数；一次 rename/copy 算一个文件修改。正好 50 个文件不触发升级。
Intent 来源映射为：

```text
explicit -> low
inferred -> medium
null     -> high
```

其他确定性风险规则以后可以作为独立 floor 加入，但不得降低已有 floor。

### 12.2 模型风险判断

模型输入为：

```text
IntentPacket
+ DiffArtifact index 和必要 Diff 内容
+ ChangedSymbols
+ QualityGate summary
+ applicable review rules
```

模型只判断业务敏感性、影响范围和可撤销性。它的输出必须严格只有一个字段：

```json
{"level":"low"}
```

Schema：

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["level"],
  "properties": {
    "level": {
      "type": "string",
      "enum": ["low", "medium", "high", "critical"]
    }
  }
}
```

旧 Risk 输出中的以下字段全部删除：

```text
dimensions
reasons
signal_refs
uncertainties
suggested_focus
```

风险 Prompt 必须要求模型默认选择 `low`，不能因为代码复杂、Diff 较长或文件名看起来可疑就升级。
非敏感、局部且容易撤销的修改应尽量为 `low`；`medium` 需要至少一个明确升级因素；`high` 通常要求
敏感业务与广泛影响或难以撤销同时成立；`critical` 只用于有具体证据的大范围、严重且难以恢复的安全、
数据或财务后果。

### 12.3 风险 Prompt 示例

```text
Example 1
Intent: Improve formatting of generated log messages.
Diff: Three files in one helper; no public API change; easy to revert.
Output: {"level":"low"}

Example 2
Intent: Apply mechanical formatting to generated files.
Diff: Eighty generated files; no runtime behavior change; easy to revert.
Output: {"level":"low"}
Runtime final result: medium because changed file count is greater than 50.

Example 3
Intent: Fix token expiration validation.
Diff: Two authentication files; localized; no permission-model change.
Output: {"level":"medium"}

Example 4
Intent: Change a shared public API used by multiple modules.
Diff: Public request/response contract changes; coordinated rollback is required.
Output: {"level":"high"}

Example 5
Intent: Replace authorization checks and migrate existing permission data.
Diff: Global enforcement changes; migration is destructive and difficult to reverse.
Output: {"level":"critical"}
```

### 12.4 最终风险

```python
final_risk = max(
    deterministic_risk_floor,
    model_risk_level,
    *other_deterministic_risk_floors,
)
```

模型不能降低确定性下限。`Risk/risk.json` 最少保存：

```text
deterministic_floor
model_level
final_level
```

发送给 Assignment Planner 和 Reviewer 的业务投影只需要 `final_level`。模型在三次 Provider 尝试后仍
不可用时，记录模型失败并使用确定性 floor，不伪造模型理由。

### 12.5 Reviewer Slot

Reviewer 数量与基础类型由最终风险确定，不由 Assignment Planner 提议：

| 最终风险 | Reviewer Slot |
|---|---|
| `low` | `core` |
| `medium` | `core + adversarial` |
| `high` | `core + adversarial + dynamic` |
| `critical` | `core + adversarial + dynamic + dynamic` |

三种角色职责：

- `core`：每个 PR 必有，覆盖 Intent 对齐、主要业务正确性、调用方兼容、回归和主要测试；
- `adversarial`：只聚焦错误路径、边界值、并发/时序、重试、幂等、部分失败、资源清理和恢复，
  不重复完整 Core Review；
- `dynamic`：只聚焦 Planner 分配的专业视角，例如 security、database migration、concurrency、
  public API compatibility、performance、payment integrity 或 domain invariants。

旧 `specialist` role kind 统一改名为 `dynamic`。Security/Domain 不再是静态 role kind，而是 Dynamic 的
`perspective`。Critical 的两个 Dynamic Assignment 必须使用不同 perspective。

### 12.6 Assignment Planner

Runtime 先按风险创建固定 Slot，Assignment Planner 再直接读取 Intent、DiffArtifact、ChangedSymbols、
QualityGate 和规则，为每个 Slot 填充任务。Planner 不得改变 Reviewer 数量、角色类型、权限或预算，也不再
依赖已删除的 Risk `reasons/signal_refs/suggested_focus`。

每个 Assignment 至少保存：

```text
assignment_id
snapshot_id
role_kind: core | adversarial | dynamic
perspective: string | null
mission
assigned_files
assigned_symbols
assigned_hunks
required_checks
owner_session_id
status
findings_refs
```

Core 和 Adversarial 使用固定角色使命；Dynamic 的 perspective 和具体目标由 Planner 生成。PR 新增 Commit
后创建新 Snapshot，旧 Assignment 不得自动视为对新 Snapshot 已完成，只能保留为历史或显式标记
`needs_revalidation`。

## 13. Reviewer 上下文

### 13.1 模型调用边界

Intent Agent、Risk Agent 和每个 Reviewer 使用独立的模型对话。Intent/Risk Agent 的内部 assistant 消息、
Tool Call 和 Tool Result 不进入 Reviewer 上下文；只传递最终 IntentPacket 和最终 Risk Level。

每个 Reviewer 请求由四个标准部分组成：

```text
system
tools
messages
parameters
```

### 13.2 System

System Prompt 必须保持静态、权威并适合 Prompt Cache，包含：

```text
Reviewer identity and responsibility
Runtime safety and data trust boundary
DeveloperReviewPolicy
Rule priority
Tool-use and Artifact-read protocol
Reviewer output contract
Completion conditions
```

`DeveloperReviewPolicy` 由开发人员配置，用户和 LLM 不可修改；其内容哈希绑定产品 Execution Profile。
它高于 user message 中的所有规则。可变 Global Memory 不得进入 System Prompt。

### 13.3 Tools 与 Parameters

Tool Schema 通过 API `tools` 字段传递，不复制进 message。`parameters` 保存模型、reasoning、响应
JSON Schema、动态上下文窗口信息和 Provider 所需参数；这些内容虽然可能占用 Provider Token，
但不属于 Reviewer 初始 user message 文本。

### 13.4 初始 user message

初始 Reviewer message 固定包含以下逻辑区块：

```text
Review Identity
  pr_id, snapshot_id, base_sha, head_sha

User Conversation
  完整用户可见对话与澄清

{{system_rule}}
  用户编写的全局审查规则
  LLM 生成的审查经验

IntentPacket
  goal, source, uncertainties

Assignment
  当前 Reviewer 自己的角色、perspective、mission、targets 和 checks

Preflight Results
  QualityGate summary
  与 Assignment 相关的 ChangedSymbols

Code Changes
  小 PR 的完整 Diff，或大 PR 的完整 Diff Index、相关片段与 artifact_id

Available Artifacts
  可读取的 Diff、质量门日志和相关历史工具结果引用
```

用户可见对话通常很短，因此默认完整加入且不设置独立字符上限。当前用户请求和明确澄清属于静态 pinned
输入，不参与 Reviewer 动态消息压缩，也不得静默丢失。第三层所称“全部对话”专指 Reviewer 启动后不断
增长的 `assistant/tool` 执行链以及 Runtime 插入的动态续跑消息。

Reviewer 是新的独立 Agent，不能把 Orchestrator 的旧回复伪装成它自己的 API `assistant` 历史。
公开对话必须作为带 speaker 标签的数据嵌入初始 user message，例如：

```text
<UserConversation>
[user]
请审查这个 PR，重点检查并发问题。

[orchestrator]
需要确认是否允许修改公共接口。

[user]
不允许修改公共接口。
</UserConversation>
```

### 13.5 Reviewer 自己的消息链

Reviewer 启动后的真实历史严格为：

```text
user       initial reviewer context
assistant  tool_call batch
tool       matching tool_result messages
assistant  next tool_call or final result
...
```

恢复同一个 Reviewer Session 时重放其已提交的原始链路。新 Reviewer 使用其他 Session 的结果时，
只能通过 Artifact 或 prior-evidence 引用，不能伪造成自己以前执行过的 `assistant/tool` 消息。

### 13.6 明确排除

Reviewer 初始上下文不包含：

- Intent Agent 或 Risk Agent 的内部消息和推断过程；
- 风险模型已经删除的 reasons/dimensions/suggested focus；
- 其他 Reviewer 的完整消息历史；
- 整个 Review Plan 或无关 Assignment；
- 与当前 Assignment 无关的所有历史工具结果。

### 13.7 Finding v2

Reviewer 最终响应中的单个 Finding 严格只有五个字段：

```json
{
  "claim": "当缓存条目不存在时，这里会对 null 调用 get，首次请求将抛出异常并返回 500。",
  "severity": "high",
  "path": "src/cache.py",
  "line": 87,
  "suggestion": "在调用 get 前处理缺失的缓存条目，并增加首次请求的回归测试。"
}
```

字段语义：

- `claim`：一段简洁、自足的问题说明，同时覆盖缺陷、触发条件和实际影响，不在这里重复修复建议；
- `severity`：`blocker | high | medium | low`；
- `path`：Snapshot 内规范化的仓库相对路径；
- `line`：最能代表问题的单个正整数行锚点；第一版不增加 `start/end/range` 结构；
- `suggestion`：简洁、可执行且与该问题直接对应的修复建议；不要求生成完整补丁，但不得只是“请修复”一类
  空泛文本。

Reviewer 不生成 `finding_id`。Runtime 完成格式验证和确定性汇总后，为最终 Finding 增加稳定 ID：

```json
{
  "finding_id": "F-...",
  "claim": "...",
  "severity": "high",
  "path": "src/cache.py",
  "line": 87,
  "suggestion": "在调用 get 前处理缺失的缓存条目。"
}
```

`finding_id` 绑定 Snapshot 和规范化问题身份，不得包含 Reviewer 序号、角色、执行顺序、时间、置信度或
其他会因重试和汇总而变化的字段。相同问题由不同 Reviewer 报告时，应能合并到同一个最终 ID。

从 Finding 删除以下字段：

- `confidence`：只有证据充分的确认问题才进入 Findings；证据不足进入 Reviewer Result 的
  `uncertainties`，不再使用高/中/低置信度制造第二套严重度；
- `impact`：并入 `claim`，避免重复描述同一个后果；
- `suggested_action`：旧名称和旧可空语义删除，由必填且语义更直接的 `suggestion` 取代；
- `verification_performed`：验证过程已经存在于 Tool Results、Artifact 和 execution journal；
- `evidence_refs`：Diff、Tool Results 和 execution journal 已在 Session 中持久化，Finding 使用
  `path + line` 作为用户可理解的位置锚点，不再暴露内部 Artifact/Observation ID；
- `reviewer_indices`、`roles`、`origin`、`reviewer_task_id`、`role_kind`：属于汇总溯源元数据，保存在
  Aggregation Record，不属于用户问题本身；
- `validation_status`、`deterministic_rejection_reason`、`missing_evidence_refs`：属于被接受或拒绝候选项的
  决策日志，不进入最终 Finding。

底层 Diff、不可重新获取的 Tool Results 和 execution journal 仍按各自协议保存，只是不再逐项复制到 Finding。
Eval Adapter 可以根据 `snapshot_id + path + line` 读取对应 Diff 上下文，并生成评测协议专用的
`evidence_refs/side/from_line/to_line`；这些兼容字段不反向污染产品 Finding。

## 14. 工具结果外置与每轮预算

### 14.1 固定阈值

```text
non_reacquirable_artifact_threshold_chars = 50_000
tool_results_per_turn_budget_chars = 200_000
tool_result_preview_chars = 2_000
```

阈值按 Runtime 将要发送给模型的序列化文本字符数计算，不按 Token 计算。

### 14.2 单结果规则

- Tool Gateway 必须按具体调用标记结果是否 `reacquirable`，不能仅按工具名称推断；
- 不可重新获取的单个结果 `> 50,000` 字符时，在工具返回后立即完整写入 `ToolResults/artifacts`；
- 对于上述大结果，模型只接收不透明 `artifact_id`、工具名、原始大小、状态和最多约 2,000 字符预览；
- 不可重新获取但 `<= 50,000` 字符的结果不创建独立 Artifact，完整保留在当前上下文和 Session transcript；
- 可重新获取的结果默认不创建独立 Artifact，在当前轮预算允许时完整保留；超过当前轮预算时用最多 2,000
  字符预览和原 Tool Call 的重取信息代替，模型应改用更窄查询或分页读取；
- 模型不得接收受信任本地绝对路径作为读取授权；
- 外置失败时必须形成显式工具错误或采用明确的安全回退，不得悄悄丢失内容。

这里的“不落盘”是指不创建独立 Tool Result Artifact。为了支持断线恢复，已经进入模型消息链的小结果仍随
`execution-log.jsonl`/Session transcript 持久化；否则无法从 `turn_committed` 重建同一会话。

第一版默认分类：

- Snapshot 绑定的 `Read/Grep/Glob` 可重新获取；
- `Bash` 只有在 Tool Gateway 已声明命令只读、无网络且可确定性重跑时才可重新获取；
- `WebSearch` 可以在上下文中淘汰并重新查询，但不承诺返回完全相同内容；
- 当前 Reviewer 不提供 `Edit/Write`。以后如引入修改型 Agent，这类结果只有在副作用已提交且可从工作区
  重新观察时才能从上下文淘汰，绝不能通过自动重跑写操作来“重新获取”。

### 14.3 单轮聚合规则

一轮中最终渲染后的工具结果总量不得超过 200,000 字符。

处理算法：

1. 先外置所有超过 50,000 字符的不可重新获取单项；
2. 计算当前轮所有内联结果、预览和引用的最终渲染字符数；
3. 如果仍超过 200,000 字符，先把最旧、最大的可重新获取结果替换为预览和重取信息；
4. 如果多个不可重新获取的小结果合计仍造成溢出，把溢出部分组成一个不可变聚合 Artifact；
5. 重复计算，直到最终渲染消息不超过 200,000 字符；
6. 如果大量引用和预览本身超过 200,000 字符，继续缩短预览并生成可分页清单，仍不得越过硬上限。

不可重新获取的完整工具结果不能因为 200,000 字符预算被删除；预算只决定内联与 Artifact 引用方式。
可重新获取结果可以从上下文投影中淘汰，但必须保留重取信息。

### 14.4 索引与恢复

每次工具调用都必须记录到 `ToolResults/index.jsonl`，至少包括：

```text
tool_call_id
session_id
snapshot_id
tool_name
canonical_arguments_hash
status: started | completed | failed
is_error
error_code when failed
retryable when failed
exit_code when applicable
created_at
content_hash
rendered_size
reacquirable
artifact_id when externalized
context_evicted_at when evicted
```

小结果可以随 Session transcript 持久化；大结果必须能通过 Artifact Store 在恢复后读取。相同 Artifact 内容
可以内容寻址去重，但是否允许语义复用仍必须验证 Snapshot、工具参数和环境有效性。

### 14.5 Artifact 读取

统一读取接口至少支持：

```text
read_artifact(artifact_id, cursor, max_chars)
```

单次读取不得超过 50,000 字符，并必须返回下一页 cursor 和 `has_more`。这样读取大 Artifact 不会再次触发
大结果外置循环。

### 14.6 工具错误协议

工具错误只使用一个通用可重试标志，不建立复杂熔断状态机：

```json
{
  "is_error": true,
  "code": "path_too_long",
  "retryable": false,
  "message": "The requested path exceeds the supported path length"
}
```

典型分类：

```text
retryable=true
  tool timeout, temporary file lock, transient Git lock, transient I/O

retryable=false
  invalid arguments, unauthorized path, missing artifact, path too long,
  unsupported operation, deterministic parse error
```

Tool Result 只报告结构化事实，不携带“立即结束审查”等控制指令。System Prompt 统一要求模型不得重复
`retryable=false` 的同一调用指纹，但可以更换路径、参数、工具或使用已有 Artifact。熔断范围只覆盖
`tool_name + canonical_arguments_hash + snapshot_id`，不能因为一个路径失败而禁用整类读取工具。

## 15. Reviewer Runtime 限制

### 15.1 工具调用次数

删除 Runtime 的工具调用次数停止条件：

```text
max_tool_calls = unlimited / None
```

Runtime 不限制单轮并行工具调用数，也不限制一个 Reviewer 生命周期内的累计工具调用数。实现必须使用
“无上限”语义，而不是设置一个任意大的整数。

以下机制属于运行安全，不属于工具预算，继续保留：

- 用户取消；
- 单个工具执行超时；
- 完全相同工具与参数反复执行且没有新状态的死循环检测；
- Provider/transport 连续失败的有限重试保护。

### 15.2 单次 Reviewer 输出

删除系统拥有的 `8,192` Token 单次 Reviewer 输出上限：

```text
max_reviewer_output_tokens = unlimited / None
```

如果 Provider 允许省略 `max_output_tokens`，Adapter 应省略它；如果 Provider 强制要求该字段，Adapter 使用
该模型声明的最大输出能力。Runtime 不再自行截断到 8,192 Token。

“无限制”表示产品不增加人工上限，不表示可以越过 Provider 的最大输出能力或模型物理上下文窗口。

### 15.3 累计 Token

删除 Reviewer 的累计 Token 停止条件：

```text
max_total_tokens = unlimited / None
```

Runtime 仍必须记录 input/output/cached/reasoning 等可用 Usage 数据，用于日志、成本和评测，但不得因为历史
累计 Token 达到某个值而返回 `budget_exhausted`。

### 15.4 时间、Provider 尝试与 Tool Call 超时

所有风险等级和 Reviewer 角色统一使用：

```text
max_turns = unlimited / None
max_elapsed_seconds = 1_800
max_provider_attempts = 3
tool_timeout_seconds = 300
```

`max_elapsed_seconds` 是单个 Reviewer 的累计活跃执行时间上限，包含模型请求、Provider 重试等待、工具执行
和结果处理，不包含 Preflight、调度排队、进程关闭、显式暂停或等待恢复的离线时间。Runtime 必须持久化
已消费活跃时长，不能在 Resume 时重置为零。

这个定义使 60 分钟 Prompt Cache 空闲清理在长时间暂停后仍有意义：暂停一小时不会消耗 Reviewer 的
1,800 秒活跃期限，但 `last_api_request_at` 使用真实 UTC 时间，因此 Resume 后会触发 Layer 2。

`max_provider_attempts = 3` 表示每个模型 Turn 最多三次总尝试：首次请求加最多两次重试。429、可重试的
5xx、网络断开和 Provider request timeout 计入；Tool Error 和模型返回的协议无效结果不计为 Provider
transport 重试。第三次仍失败则当前 Reviewer 失败。

单个 Tool Call 最长运行 300 秒。超时后 Runtime 终止或取消该工具，并向模型返回
`code=tool_timeout, retryable=true`；它不直接终止 Reviewer。Preflight QualityGate 不属于 Reviewer Tool
Call，可以单独使用 `quality_gate_timeout_seconds = 1_800`。

这些时间和重试保护用于避免永久卡死，不是按风险分配的审查预算。

## 16. 1M Token 窗口与三层 Context Compaction

### 16.1 删除旧 16K 字符预算

当前 `ContextBudget.max_message_chars = 16_000` 只限制初始 Reviewer user message，并在超限时压缩或省略
Code Snippets、Observations 等区块。它不是模型上下文窗口，并且还把 Memory 默认限制为该字符预算的
10%。本设计删除这套固定字符上限和固定 10% Memory 字符预算。

### 16.2 固定阈值

目标模型的 1,000,000 Token 是物理上限，不是取消窗口管理的理由。每次请求必须满足：

```text
system tokens
+ tool definition tokens
+ current message history tokens
+ current tool result tokens
+ reserved output tokens
+ safety reserve tokens
<= 1_000_000 tokens
```

```text
context_window_tokens = 1_000_000
initial_context_target_tokens = 500_000..600_000
soft_compaction_trigger_tokens = 700_000
compaction_summary_max_tokens = 50_000
prompt_cache_idle_eviction_seconds = 3_600
recent_reacquirable_tool_results_to_keep = 5
output_reserve_tokens = provider_model_max_output_tokens
safety_reserve_tokens = 50_000
hard_input_limit_tokens =
    context_window_tokens
    - output_reserve_tokens
    - safety_reserve_tokens
```

初始上下文只装入有价值内容，目标不超过窗口的约 50%～60%。`output_reserve_tokens` 不是新的 Reviewer
输出预算，只是为 Provider 可能生成的最大输出留出物理空间。

`compaction_summary_max_tokens` 只约束内部压缩产物，不是 Reviewer 最终输出预算。没有这个边界，压缩模型
可能再次生成接近原历史大小的 Summary，使 Layer 3 失去意义。

Token 计数应优先使用模型/Provider 对应 tokenizer。无法取得精确 tokenizer 时必须使用保守估算并保留
安全余量，不得把字符数直接当作 Token 数。

### 16.3 每次 API 调用前的固定顺序

每一轮先构造候选上下文，再按以下顺序处理，最后才调用模型 API：

```text
assemble candidate context
  -> Layer 1: Tool Result size governance
  -> Layer 2: Prompt Cache idle eviction when applicable
  -> estimate complete request tokens
  -> Layer 3: Full dynamic-history compaction when applicable
  -> rebuild and re-estimate request
  -> hard window check
  -> Provider API call
```

三层按顺序运行，但不是每轮都会发生内容变化。Layer 1 每轮检查；Layer 2 和 Layer 3 只有触发条件满足时
才修改上下文投影。

### 16.4 Layer 1：50K/200K 工具结果治理

Layer 1 使用第 14 节的规则。不可重新获取且超过 50,000 字符的结果不等待下一轮，在工具返回时立即外置，
上下文只保留 2,000 字符预览和 Artifact ID。可重新获取结果及不可重新获取的小结果先以内联形式参与候选
上下文；单轮总量超过 200,000 字符时，再按 14.3 的顺序淘汰、预览或聚合外置。

这一层是尺寸治理，不调用模型，不生成语义摘要。

### 16.5 Layer 2：Prompt Cache 空闲清理

Runtime 在每次 Provider 请求前读取当前 Reviewer Session 的 `last_api_request_at`。首次请求不触发；如果
距离上一次 API 请求已经达到或超过 3,600 秒，则认为 Provider 侧 Prompt Cache 大概率已经过期，执行一次
确定性清理：

1. 按完成时间排列当前动态历史中可重新获取或可从当前状态重新观察的 Tool Result；
2. 最新五个结果保留完整内容；
3. 更旧结果的正文替换为极小的 eviction marker，保留 `tool_call_id`、工具名、参数哈希、清理原因和重取
   提示；
4. 不清理不可重新获取的 Tool Result，不修改任何静态 pinned 输入；
5. 保留原 assistant `tool_call` 与匹配的 tool-result marker，不能制造孤立 Tool Call 或破坏 Provider
   消息协议。

示意 marker：

```json
{
  "status": "context_evicted",
  "reason": "prompt_cache_idle_60m",
  "tool_call_id": "call-...",
  "tool_name": "grep",
  "arguments_hash": "sha256:...",
  "reacquirable": true
}
```

这是基于 60 分钟的简单启发式，不尝试猜测各 Provider 的实际 Cache TTL。`last_api_request_at` 在请求交给
Provider Adapter 时更新并持久化；因此清理后的第一轮调用会建立新的时间基线，而不会在后续紧邻轮次
重复触发。

### 16.6 Layer 3：达到窗口阈值后的全量压缩

Layer 1 和 Layer 2 完成后，Runtime 对整个待发送请求估算 Token。满足任一条件时触发全量压缩：

```text
estimated_input_tokens >= 700_000
or
estimated_input_tokens > hard_input_limit_tokens
```

第一版不建立独立 Compactor Agent，直接使用当前 Reviewer 模型执行一次 Compaction 请求。它把截至最近
`turn_committed` 的全部可压缩动态消息合并为一个新的 `ReviewerCompactionSummary`。这里的“全部”包括：

- Reviewer 启动后增长的 assistant 文本和动态续跑消息；
- 已完成的 tool_call/tool_result 链，包括 Layer 2 仍保留的最近五个完整结果；
- 不可重新获取但较小、此前以内联形式保留的 Tool Result；
- 上一代 Compaction Summary 以及它之后新增的动态历史。

成功提交新 Summary 后，上述旧动态消息不再以原文进入下一次 API 请求，不额外保留“最近几轮”尾巴。
不可重新获取的大结果 Artifact 继续存在于磁盘；Summary 只概括其结论和 Artifact 可用性，不复制大正文。

### 16.7 Pinned 与可压缩边界

以下内容在 Reviewer 启动时装入后不随工具轮次增长，保持 pinned，不进入 Layer 2 或 Layer 3：

```text
System Prompt and DeveloperReviewPolicy
Tool schemas and Runtime parameters
完整用户请求、公开对话与明确澄清
{{system_rule}}
IntentPacket
Risk level and current Assignment
QualityGate and ChangedSymbols
Diff or Diff Index and initial Artifact catalog
Review identity and Snapshot binding
```

如果 pinned 内容自身超过初始目标或硬输入上限，必须回到初始 Context Assembly，通过完整 Diff 与 Diff Index
的选择、分页 Artifact 引用等方式解决；不得通过压缩用户要求、规则、Intent 或 Assignment 掩盖问题。

### 16.8 Compaction Summary 与提交

Summary 使用一个简洁文本块，不建立新的复杂领域模型，但必须覆盖：

```text
已完成的调查
必须保留的事实与工具结论
当前候选 Findings 和 uncertainties
仍未完成的任务与下一步
```

Compaction 只在 Summary 生成成功、不超过 50,000 Token、通过非空和大小校验、写入 Session 并更新
`context-manifest.json` 后提交。压缩后的完整候选请求必须低于 700,000 Token；否则本次 Summary 无效，
不得提交。
至少记录 generation、压缩到的最后 Turn、触发原因、源消息范围和 Summary hash。原始 execution journal 和
Artifact 不删除；它们用于审计和恢复，但默认不重新注入模型上下文。

Compaction 请求失败时不得半写状态或静默截断历史。它遵循每个模型 Turn 最多三次 Provider 尝试；若仍
失败且原上下文无法通过硬窗口检查，当前 Reviewer 以 `context_compaction_failed` 明确失败。

### 16.9 最终硬检查

无论是否触发压缩，发送请求前都必须满足：

```text
system tokens
+ tool definition tokens
+ current message history tokens
+ current tool result tokens
+ reserved output tokens
+ safety reserve tokens
<= 1_000_000 tokens
```

三类阈值用途不同：

```text
不可重新获取大结果阈值     50,000 chars
单轮工具结果上下文预算     200,000 chars
Prompt Cache 空闲清理       3,600 seconds / keep latest 5
全量压缩软阈值             700,000 tokens
压缩 Summary 上限           50,000 tokens
整个 Reviewer 上下文窗口   1,000,000 tokens
```

任何层都不能用字符阈值代替整个请求的 Token 硬检查。

## 17. 长工具调用链的持久化与恢复

### 17.1 追加式执行日志

每个 Reviewer 使用 `Sessions/<session-id>/execution-log.jsonl` 保存逐轮执行事件。最小事件集合为：

```text
model_response
tool_started
tool_completed
turn_committed
context_idle_eviction
context_compaction_started
context_compaction_committed
final_result
```

执行顺序固定为：

```text
模型返回 tool_call batch
  -> 持久化 model_response
  -> 每个工具执行前持久化 tool_started
  -> 执行工具并应用 300 秒 timeout
  -> 完整结果/Artifact 创建成功后持久化 tool_completed
  -> 全部 call 都有匹配结果后持久化 turn_committed
  -> 发起下一轮模型请求
```

JSONL 使用追加写和完整单行事件，不在每轮覆盖一个不断增长的大 JSON。Artifact 必须先 create-only 成功，
再提交引用它的 `tool_completed`。

Layer 2 清理提交 `context_idle_eviction`。Layer 3 先写 `context_compaction_started`，只有 Summary 与更新后的
Context Manifest 均安全持久化后才写 `context_compaction_committed`；未提交的 Compaction 不改变活动投影。

### 17.2 Tool Call 幂等身份

每个工具执行由以下元组唯一标识：

```text
session_id
reviewer/assignment_id
tool_call_id
tool_name
canonical_arguments_hash
snapshot_id
```

同一身份只能有一个终态结果。重复收到或恢复到已完成调用时直接复用已提交的 Tool Result 投影或 Artifact，
不得再次执行。上下文淘汰后如果模型确实需要重新获取，必须发出新的 Tool Call ID。
相同 `tool_call_id` 对应不同工具或参数属于完整性错误。

### 17.3 Resume

恢复时先验证并重放到最后一个 `turn_committed`：

- `completed`：复用原始 Tool Result 或 Artifact；
- `failed, retryable=false`：复用原错误，禁止相同指纹再次执行；
- `tool_started` 但没有终态：当前产品工具均为只读，可以重新执行并提交一个终态；
- 已提交 assistant tool_call 必须恰好有一个相邻且 call ID 匹配的 tool result；
- 只有 `context_compaction_committed` 对应且 Summary hash 验证成功的投影可以恢复；孤立的
  `context_compaction_started` 必须忽略；
- 已清理的 Tool Result 恢复为相同 eviction marker，不因 Resume 自动重新执行；
- 新 Reviewer 不得继承另一个 Reviewer 的 tool_call 身份，只能读取其授权 Artifact。

如果以后引入非幂等写工具，`tool_started` 后状态未知的调用不得自动重放，必须另行设计恢复协议。

### 17.4 Provider 与工具重试分离

- Provider 429/5xx/网络失败只重试同一个模型请求，最多三次；
- Provider 重试不能触发任何已完成 Tool Call 再执行；
- 工具 `retryable` 只描述该 Tool Call，模型决定是否使用不同调用继续；
- 工具调用次数不限，但无进展的完全相同不可重试调用会由幂等账本直接返回原错误；
- 不为 Finalization 预留 Turn 或 Token，也不因单个工具失败强制 Reviewer 立即结束；
- Reviewer 仍受 1,800 秒累计活跃执行时间约束。

## 18. Reviewer 结果汇总与最终返回

### 18.1 ReviewerOutput v2

Reviewer 的最终模型响应只允许两个顶层字段：

```json
{
  "findings": [
    {
      "claim": "当缓存条目不存在时会抛出异常并返回 500。",
      "severity": "high",
      "path": "src/cache.py",
      "line": 87,
      "suggestion": "在调用 get 前处理缺失缓存项，并增加首次请求测试。"
    }
  ],
  "uncertainties": []
}
```

模型不再输出 `status`、`contract_assessments`、`rejected_hypotheses`、`observation_refs` 或
`investigation_summary`。执行状态由 Runtime 根据模型调用、超时和协议校验记录，不能让模型自行声明成功。
零 Finding 是合法结果，不表示 Reviewer 失败。

顶层 JSON 无法解析、顶层字段不精确或 `findings/uncertainties` 不是数组时，该 Reviewer 标记为
`invalid_output`。单个 Finding 无法通过 Finding v2、路径或行锚点校验时，只拒绝该候选项并在内部
Aggregation Record 记录原因；其他合法 Finding 不受影响。

### 18.2 确定性汇总

所有计划内 Reviewer 到达终态后，Runtime 执行一次本地确定性汇总，不调用 LLM、不运行工具，也不启动
补充 Reviewer：

1. 按 Review Plan 顺序收集协议有效 Reviewer 的合法 Finding 和 uncertainties；
2. 为每个 Finding 计算规范化问题身份；
3. 只合并问题身份完全相同的重复项；
4. 合并并去重 uncertainties，再补充失败、超时或无效 Reviewer 造成的覆盖缺口；
5. 生成最终状态、稳定 Finding ID 和固定排序；
6. 原子写入 Aggregation Record 与最终 ReviewResult。

第一版问题身份为：

```text
snapshot_id
+ normalized repository-relative path
+ positive line anchor
+ NFKC / trimmed / whitespace-collapsed claim with case preserved
```

`severity`、`suggestion`、Reviewer 角色、执行顺序和时间不进入问题身份。相同位置但 claim 不同的 Finding
继续分别保留；第一版不使用文本相似度、Embedding 或额外模型猜测它们是否等价。

Claim 规范化保留大小写，因为代码标识符可能区分大小写；不能为了多合并一个重复项而把 `Foo` 和 `foo`
相关缺陷误认为同一问题。

### 18.3 重复项与排序

同一问题身份存在多个候选项时：

- `severity` 取 `blocker > high > medium > low` 中最高值；
- claim 和 suggestion 从最高严重度候选中选择；仍并列时按 `core -> adversarial -> dynamic`、Assignment
  顺序和 Reviewer ID 依次打破平局；
- `finding_id = F-<sha256(normalized issue identity)>`；问题身份本身已经包含 `snapshot_id`；
- Aggregation Record 保存全部来源候选和选择过程，Final Finding 不保存 Reviewer 来源字段。

最终 Findings 按 severity 降序、规范化 path、line、finding_id 排序。相同输入必须逐字节产生相同顺序和
相同 Finding ID。

### 18.4 ReviewResult v1

权威最终结果严格只有六个顶层字段：

```json
{
  "pr_id": "PR-...",
  "snapshot_id": "S-...",
  "status": "completed",
  "risk_level": "medium",
  "findings": [
    {
      "finding_id": "F-...",
      "claim": "当缓存条目不存在时会抛出异常并返回 500。",
      "severity": "high",
      "path": "src/cache.py",
      "line": 87,
      "suggestion": "在调用 get 前处理缺失缓存项，并增加首次请求测试。"
    }
  ],
  "uncertainties": []
}
```

不增加 summary、recommendation、reviewer statuses、quality gates、intent、risk reasons、contract coverage、
rejected hypotheses、verification narrative、memory audit 或统计计数。需要这些内部事实时读取 Snapshot 或
Aggregation Record，不复制到用户结果。

`risk_level` 是生成 Reviewer Slot 时已经确定的风险等级，不在 Reviewer 完成后重新评估。

### 18.5 最终状态与 uncertainties

状态只由 Runtime 计算：

```text
completed  所有计划内 Reviewer 都产出协议有效的 ReviewerOutput
partial    至少一个 ReviewerOutput 有效，但至少一个计划内 Reviewer 没有有效输出
failed     没有任何协议有效的 ReviewerOutput
```

“没有有效输出”覆盖 `failed`、`timeout`、`invalid_output`、`context_compaction_failed` 及其他 Runtime
终态错误；这些细分状态只保存在 Aggregation Record。

`completed` 只表示计划内 Reviewer 执行与输出协议完整，不代表“批准合并”、没有 Finding 或没有
uncertainty；最终结果不生成 approve/reject recommendation。

Core 失败但其他 Reviewer 有有效结果时返回 `partial` 和已有 Findings，不把有用结果全部丢弃。所有 Reviewer
均有效且 Findings 为空时仍是 `completed`。

最终 uncertainties 由两类文本组成：

- 有效 ReviewerOutput 提交的 uncertainty；
- Runtime 根据 Reviewer role、终态和稳定 error code 生成的覆盖缺口，例如 Core timeout。

文本只做 NFKC、首尾空白和连续空白规范化后精确去重；不调用模型改写。顺序先按 Review Plan 来源，再按
规范化文本。被拒绝的单个候选 Finding 也生成一条不暴露内部路径或异常堆栈的 uncertainty。

### 18.6 持久化与返回

```text
Results/aggregation.json     内部审计：Reviewer 状态、候选项、重复组、拒绝原因和选择过程
Results/review-result.json   唯一权威用户结果
Results/review.md            review-result.json 的可选纯渲染
```

`review-result.json` 使用临时文件、flush/fsync 和原子 rename 发布，成功后不可原地覆盖。Resume 发现已有结果
时验证 Snapshot binding 和内容 hash 后直接复用，不重新聚合或调用模型。

JSON 使用 UTF-8、固定键顺序和规范化序列化，不写入 `generated_at` 或其他墙钟字段，确保相同输入得到
逐字节相同的 ReviewResult。

CLI JSON 直接返回 `review-result.json`。Markdown 只显示状态、风险、按序 Findings 及 uncertainties；每个
Finding 只渲染 severity、`path:line`、claim 和 suggestion。Markdown 不拥有新事实，删除后可以随时从 JSON
重建。

长期经验提取如以后启用，只能在 ReviewResult 已持久化并返回后异步执行；其失败不得修改 ReviewResult 或
把已完成审查降级为 partial。

### 18.7 删除的旧后处理阶段

Reviewer 终态之后的主链只剩：

```text
deterministic aggregation
  -> review-result.json
  -> optional pure rendering
```

以下阶段从阻塞式产品主链删除：

- Reconciliation Analysis / Semantic Reconciler；
- Supplemental Investigation 和追加 Reviewer Wave；
- 独立 Evidence Reconciliation；
- Completion Phase；
- Final Risk reassessment；
- 阻塞式 Memory Proposal；
- 构造多章节 `ReviewBrief` 的 Reporting Pipeline。

## 19. Windows 路径精简策略

Windows 路径问题与长 Tool Chain 恢复是两个独立问题。第一版只采用两项简单措施：

1. Runtime/Eval 使用较短的真实存储根目录和短内容 ID，避免深层重复嵌套；
2. 所有 GitRunner 调用在进程级显式使用 `git -c core.longpaths=true ...`。

示例目标路径：

```text
D:\ra\pr\<pr-id>\<snapshot-id>\...
```

不得使用 Junction 或符号链接伪造短路径。第一版不引入全项目通用 `\\?\` 包装层：它不能保证第三方
工具兼容，并会增加 UNC、规范化和安全校验复杂度。Runtime 自己的敏感读写继续通过统一安全文件接口；
出现确定性路径过长时返回 `code=path_too_long, retryable=false`。

## 20. ObservationStore 的新定位

`ObservationStore` 不再作为 PR 数据、工具大结果、上下文摘要和长期记忆的统一根抽象。目标职责拆分为：

```text
PRWorkspaceStore
    PR、Snapshot、Intent、Plan、Assignment、Result 和 Session 状态

ArtifactStore
    Diff、大型工具结果及其他不可内联大文件

GlobalMemoryStore
    审查规则和经过筛选的审查经验

ContextAssembler / ContextWindowManager
    从以上数据生成当前模型短期上下文并管理窗口
```

迁移期间可以保留 `ObservationStore` 作为旧接口兼容层或特定证据日志，但新核心流程不得继续要求每类
数据都通过 `raw_content + context_view` 形式写入 Observation。

## 21. 数据完整性与安全不变量

实现必须满足：

- Snapshot 和 DiffArtifact 创建后不可原地覆盖；
- 所有 Artifact 使用内容哈希并在读取时验证；
- 模型只通过不透明 Artifact ID 读取，不依赖可伪造绝对路径；
- Repository 内容、Diff、工具输出和长期经验都属于不受信任数据，不能成为 System 指令；
- 所有截断都必须显式提供 `has_more/cursor` 或清晰 compaction marker；
- 不允许静默丢弃不可重新获取的工具结果、Diff hunk、Intent 来源或 Assignment 状态；可重新获取结果从
  上下文淘汰时必须留下显式 marker 和重取信息；
- Session 只能访问当前 PRWorkspace 授权的 Artifact；
- Snapshot 缓存复用必须验证完整 repository/base/head/analyzer/config binding；
- `aggregation.json` 和 `review-result.json` 发布后不可原地覆盖，Resume 必须验证 Snapshot binding 和 hash；
- `review.md` 和 CLI 输出不得产生 `review-result.json` 中不存在的新事实。

## 22. 迁移影响

后续实现至少需要处理以下现有接口：

1. `ChangeSummary` 和 `collect_change_summary()` 改为 `DiffArtifact` 构建与验证接口。
2. 删除 `diff_excerpt`、`file_diff_excerpts` 和 `diff_truncated` 对 Context 的权威依赖。
3. 质量门 Pipeline 收敛为一个本地确定性阶段，删除 deep gate 相关计划、Observation 和恢复分支。
4. `RepositoryIntelligenceSnapshot.changed_symbols` 绑定新的 Snapshot/Artifact provenance。
5. `IntentPacket` 升级为只含 `goal/source/uncertainties` 的 v2；删除 acceptance criteria、scope、
   constraints、sources、status、provenance 和 clarifications 下游依赖。
6. Intent 持久化 Envelope 增加 PR 级版本、`source_snapshot_id` 和内部 analysis record 引用。
7. Risk Model Schema 改为仅输出 `level`；删除 dimensions、reasons、signal refs、uncertainties 和
   suggested focus，并加入确定性 file-count/Intent-source floor。
8. `RiskAssessmentPacket`、Portfolio 和 Reconciler 不再读取已删除的 Intent/Risk 字段。
9. `ReviewProfile` 只决定固定 Reviewer Slot；`specialist` role kind 改为 `dynamic`，Security/Domain
   变为 perspective。
10. Portfolio Candidate 流程收敛为固定 Slot 的 Assignment Planner，不再允许模型提议数量、基础角色、
    权限或预算。
11. Assignment 删除 risk reason/signal、旧 Contract 和工具/Token 预算字段，增加明确的 file/symbol/hunk
    targets 和 dynamic perspective。
12. Review Plan/Assignment 持久化迁入 Snapshot，并支持跨 Session owner/status 协作。
13. Reviewer System Prompt 接入不可变 DeveloperReviewPolicy 与规则优先级；可变 Global Memory 只进入
    user message 的 `{{system_rule}}`。
14. Context Assembler 接入完整 User Conversation、精简 Intent、Assignment、Preflight、Diff/Index 和
    Artifact Catalog，并排除 Intent/Risk 内部对话。
15. Reviewer 输出协议迁移到 Finding v2；删除 `confidence/impact/evidence_refs/verification_performed`，
    将旧的可空 `suggested_action` 改为必填 `suggestion`；`finding_id` 改由 Runtime 汇总后生成，Reviewer
    来源与候选拒绝原因迁入 Aggregation Record。
16. 旧 `ReviewerFinding -> FindingCandidate -> CanonicalFinding -> BriefFinding` 的重复字段复制链应收敛为
    Reviewer Finding、内部 Aggregation Record 和 Final Finding 三个边界清晰的结构。
17. Tool Gateway 接入调用级 `reacquirable` 分类、不可重新获取结果的 50K 外置阈值、200K 单轮治理、
    分页 Artifact 读取和 `retryable` 错误 Envelope。
18. `ReviewProfile`/`Assignment` 不再用 `max_turns`、`max_tool_calls`、`max_output_tokens`、
    `max_total_tokens` 作为停止条件；统一 1,800 秒累计活跃执行时间、三次 Provider 尝试和 300 秒工具超时。
19. Agent Loop 增加追加式 execution journal、Tool Call 幂等账本和 `turn_committed` Resume；旧 Session
    hydration 需要明确兼容策略。
20. Runtime/Eval 生成路径迁移到短真实根目录，GitRunner 统一显式启用 `core.longpaths=true`。
21. `ContextBudget(max_message_chars=16_000)` 被 Token-aware `ContextWindowPolicy` 和固定三层
    Compaction Pipeline 取代。
22. Session state 增加累计 `active_elapsed_seconds`；Context Manifest 增加 `last_api_request_at`、Compaction
    generation、source Turn 范围和 Summary hash；旧 Session hydration 需要明确默认值和迁移策略。
23. Agent Loop 增加 60 分钟 Prompt Cache 空闲清理，并保证被清理 tool result 仍与原 tool_call 合法配对。
24. Agent Loop 增加 700K Token 全量动态历史压缩、原子提交、Resume 校验和失败回滚。
25. Reviewer 最终响应协议改为严格的 `findings/uncertainties` 两字段；Reviewer 状态完全迁入 Runtime。
26. Semantic Reconciler、Supplemental Investigation、Evidence Reconciliation、Completion、Final Risk、
    阻塞式 Memory Proposal 和旧 ReviewBrief Reporting 从 Reviewer 后主链删除。
27. 新 Deterministic Aggregator 生成 `aggregation.json` 和六字段 `review-result.json`，并实现稳定 fingerprint、
    Finding ID、状态、uncertainty 合并和排序。
28. Reporting 改为 ReviewResult 的纯 JSON/Markdown Renderer；长期经验提取移到返回后的非阻塞流程。
29. Reviewer protocol projection、配置 digest、Eval product-equivalence binding 和相关测试必须随产品
    协议版本更新，不能让旧 Intent/Risk/ReviewerOutput/ReviewBrief Schema、角色和预算继续伪装成当前产品配置。

## 23. 验收标准

### 23.1 PR Workspace 与 Snapshot

- 同一 PR 的两个 Session 能读取相同 Snapshot、Intent 和 Assignment 状态；
- 不同 PR 无法互相解析 Artifact ID；
- 同一 PR 新 head SHA 创建新 Snapshot，旧 Snapshot 字节不变；
- 旧 Assignment 不会自动标记为新 Snapshot 已完成。

### 23.2 DiffArtifact

- 大 PR 的完整 Diff 字节可恢复且哈希一致；
- index 覆盖全部文件和 hunk；
- 按文件/hunk 读取与完整 Diff 对应片段一致；
- 不存在 120 行/80 行 excerpt 截断依赖。

### 23.3 QualityGate

- 只运行已有、本地、无网络、只读且有超时的确定性检查；
- `passed/failed/unavailable/error` 状态可恢复；
- 大 stdout/stderr 通过统一 Artifact 机制保存；
- 产品 Pipeline 不再调度 deep quality gate。

### 23.4 Intent 与 Risk

- Intent Agent 下游 Packet 严格只有 `goal/source/uncertainties`；
- `goal/source` 的 nullability 不变量被严格验证；
- Intent/Risk 内部对话不会进入 Reviewer Context；
- `explicit/inferred/null` 分别产生 low/medium/high Intent risk floor；
- 50 个文件不升级，51 个文件产生 medium floor；
- Risk Model 只能返回 `{"level":"..."}`，未知字段被拒绝；
- 最终 Risk 等于全部确定性 floor 与模型 level 的最大值。

### 23.5 Reviewer Slot 与 Assignment

- low/medium/high/critical 分别生成 1/2/3/4 个固定 Reviewer Slot；
- Core 每次必有，Adversarial 只在非 low 出现，Dynamic 只在 high/critical 出现；
- Critical 的两个 Dynamic perspective 不重复；
- Assignment Planner 不能改变 Slot 数量、基础角色、权限或 Runtime 限制；
- Assignment 只引用当前 Snapshot 的授权文件、Symbol、hunk 和 Artifact。

### 23.6 规则与 Reviewer 上下文

- DeveloperReviewPolicy 位于 System 且高于 user `{{system_rule}}`；
- 冲突时只忽略冲突的低优先级规则；
- 完整用户可见对话以 speaker-labeled 数据进入初始 user message；
- Orchestrator 历史不会伪装成新 Reviewer 的 assistant 消息；
- Tool Schema 位于 API tools 字段，Intent/Risk 内部历史和无关 Assignment 不进入上下文。

### 23.7 Finding v2

- Reviewer Finding 严格只有 `claim/severity/path/line/suggestion`，未知字段被拒绝；
- `claim` 自足地表达缺陷、触发条件和影响，不再重复生成 `impact`；
- `suggestion` 必须具体、可执行并直接对应 Finding；
- 证据不足的候选进入 `uncertainties`，不通过 `confidence` 留在 Findings；
- `finding_id` 只由 Runtime 在汇总后生成，不因 Reviewer 角色、重试或执行顺序变化；
- 产品 Finding 不包含内部 Evidence/Artifact ID；Reviewer 来源和候选拒绝原因只存在于 Aggregation Record；
- 产品 Finding 可以无损投影为 Eval SubmissionFinding，而无需在产品模型中保存 Eval 专用定位字段。

### 23.8 工具结果与长链路恢复

- 不可重新获取结果在 50,000 字符时可内联，50,001 字符时立即外置并只投影 2,000 字符预览；
- 可重新获取结果不因单项超过 50,000 字符而创建独立 Artifact，超出当前轮预算时改为预览和重取信息；
- 多个不可重新获取的小结果合计造成 200,000 字符溢出时形成聚合 Artifact；
- 最终单轮工具结果渲染始终不超过 200,000 字符；
- 大结果可分页完整读回；
- 工具调用次数不会触发 Runtime budget exhaustion；
- `retryable=false` 的相同调用指纹不会再次执行；
- 每个已提交 assistant tool_call 恰好有一个匹配且相邻的 Tool Result；
- 进程在任意工具轮次中断后可从最后一个 `turn_committed` 恢复；
- 已完成 Tool Call 在 Provider 重试或 Resume 后不会重复执行；
- 短真实路径和进程级 `core.longpaths=true` 覆盖 Windows 基础长路径场景。

### 23.9 Reviewer Runtime 与上下文窗口

- 产品不再传递或强制 8,192 Token Reviewer 输出上限；
- 历史累计 Token 不触发 `budget_exhausted`；
- Turn 和工具调用次数不触发 budget exhaustion；
- 每个 Reviewer 最长 1,800 秒，每个模型 Turn 最多三次 Provider 尝试；
- 1,800 秒按累计活跃执行时间计算，Resume 不重置，离线暂停时间不计入；
- 单个 Reviewer Tool Call 300 秒超时，超时作为可重试 Tool Error 返回；
- Usage 仍完整记录；
- 初始 Reviewer message 不再受 16,000 字符硬截断；
- 每次模型请求都在 Token Window Policy 下为输出和安全余量留出空间；
- API 空闲 3,599 秒不触发 Layer 2，达到 3,600 秒时只保留最新五个可重新获取 Tool Result 的完整正文；
- Layer 2 清理后每个历史 tool_call 仍有一个匹配的 tool-result marker；
- 达到 700,000 Token 时，所有已提交动态消息压缩为一个 Summary，不保留未压缩的最近历史尾巴；
- System、规则、用户请求、Intent、Assignment、Preflight、ChangedSymbols 和 Diff/Index 在压缩前后字节不变；
- Compaction Summary 至少保留调查进度、关键事实、候选 Findings、uncertainties 和下一步；
- Compaction Summary 不超过 50,000 Token，提交后完整候选请求低于 700,000 Token；
- 只有已提交且 hash 验证成功的 Compaction 可以 Resume，孤立 started 事件不会改变活动投影；
- Compaction 失败不会半写或静默截断，无法满足硬窗口时返回 `context_compaction_failed`；
- 所有请求最终都满足 1,000,000 Token 窗口、输出预留和安全余量约束。

### 23.10 Reviewer 汇总与最终结果

- ReviewerOutput 顶层严格只有 `findings/uncertainties`，零 Finding 合法；
- 一个非法候选 Finding 不会删除同一 Reviewer 的其他合法 Finding；
- 相同 Snapshot、path、line 和规范化 claim 产生相同 Finding ID，与 Reviewer、severity 和 suggestion 无关；
- 同一位置但不同 claim 的 Findings 不会被模糊合并；
- 重复项 severity 取最高，选择过程与全部来源只保存在 Aggregation Record；
- 所有 Reviewer 有效、部分有效、全部无效分别产生 `completed/partial/failed`；
- Core 失败但其他 Reviewer 有效时仍返回 `partial` 和已有 Findings；
- 最终 JSON 严格只有 `pr_id/snapshot_id/status/risk_level/findings/uncertainties`；
- 相同输入重复聚合产生逐字节相同的 Finding ID、排序和 ReviewResult；
- Markdown 删除后能从 JSON 重建，且不能增加 summary、recommendation 或其他模型生成事实；
- 已完成 ReviewResult 的 Resume 不重新调用模型，异步 Memory 失败不改变最终状态。

## 24. 后续讨论入口

下一阶段从以下问题继续，不回退本规格已经确定的存储和预算边界：

1. 小 PR 与大 PR 的完整 Diff/索引切换标准；
2. Assignment Planner 如何选择相关文件、Symbol、hunk 和历史工具 Artifact；
3. Compaction Summary 的最终 Prompt 文案、质量评测和 Provider Cache TTL 调优；
4. 是否在确定性精确去重之外增加非阻塞的语义重复提示；
5. 长期审查经验的自动晋升、降权和淘汰策略。

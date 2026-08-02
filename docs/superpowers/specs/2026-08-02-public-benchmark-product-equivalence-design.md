# Public Benchmark 产品等价执行与有效终态设计

- 日期：2026-08-02
- 状态：已确认，待实施计划
- 目标分支：`codex/public-intent-continuation`
- 适用范围：AACR-Bench、SWE-PRBench 与其他无权威 Intent 的公共无人值守能力评测
- 相关设计：`2026-07-16-core-code-review-eval-system-design.md`、`2026-07-22-focused-eval-v2-completion-design.md`

## 1. 背景

当前 Eval Harness 已能导入 AACR-Bench 和 SWE-PRBench、准备 Repository Target、调用当前产品 Agent、保存 Submission 与 Trace，并使用标注 Finding 计算指标。但是首批真实公共测评暴露了三类会使能力成绩失真的问题。

第一，当前产品 CLI 的 `--reviewer-loop` 默认值仍是 `single-shot`。公共评测没有在冻结的 Agent 配置中显式绑定 `agent-loop`，导致 DeepSeek 返回工具调用后，Single-shot 路径把工具请求当作最终 Reviewer JSON 解析。Core Reviewer 因而失败，Finding 从未产生。

第二，产品 Session 的 `completed` 只表示执行生命周期已经结束。Current Agent Adapter 仅依据该状态生成 `SubmissionStatus.COMPLETED`，没有读取权威 `completion.json`。因此产品内部的 `blocked` 被错误包装为正常完成，Evaluator 随后计算出 Finding Recall 为零，同时错误地报告 Agent Failure Rate 为零。

第三，公共仓库准备使用固定的 `100,000` Git object 上限，并通过 `git rev-list --objects base head` 枚举 Base 与 Head 可达的完整祖先历史。首批 10 个 AACR C# Case 中，只有 `microsoft/semantic-kernel` 进入 Agent 阶段；另外 9 个来自 aspnetcore、osu、PowerToys 和 PowerShell 的 Case 在仓库准备阶段超过该上限。它们不是 Agent 失败，而是 Eval 专属资源策略淘汰。

最近完成的 AACR 与 SWE 两个 Run 因 Reviewer 协议失败而无效，不能作为能力基线。该结论由 Reviewer Trace、空 Candidate Catalog 和产品 `completion.json` 中的 Core Reviewer blocker 共同支持。

## 2. 目标

本设计实现以下目标：

1. 公共 Benchmark 运行真实产品的完整 `agent-loop`，不使用 Single-shot 替代路径。
2. Agent 面向模型的工具、模型、Prompt、预算和执行模式只有产品配置这一个权威来源。
3. 无权威 Intent、无脚本化用户回答的公共 Case 可以无人值守完成 Intent 构建，并把最终有效 Intent 存为 `explicit`。
4. Eval 使用产品 `completion.json` 判断审查是否真正完成。
5. Agent failed/blocked 继续按现有 `failure_as_miss` 贡献 Recall miss，同时保留准确的 Failure Rate 与失败原因。
6. Repository Materializer 不再扫描和复制与 Base/Head Review 无关的完整历史 tree/blob，也不再用固定 Git object 数量淘汰公共 Case。
7. 保留每 Trial 隔离、精确 Base/Head、来源校验、哈希、Evidence Replay 和不可变产物。
8. 以最小必要改动恢复有效公共能力测评，不增加新的 Bash、网络或 `run_safe_check` 能力，不重写整个 Evaluator。

## 3. 非目标

本版本不包含：

- 通用 Bash 或任意 Shell 工具；
- Reviewer 任意互联网访问；
- `run_safe_check` Reviewer 工具；
- GitHub/PR 平台集成；
- 修改 AACR/SWE 原始输入或 Ground Truth；
- 把 AACR/SWE 缺失的 Intent、Severity 或 Location Authority 人工补成权威真值；
- 新建一套与产品不同的 Eval Reviewer Runtime；
- 对现有所有 Repository 安全策略进行无关重构；
- 在获得有效 Agent 输出前重写 Finding Matcher 或 Judge；
- 本版本内实现下一阶段 Bash/网络能力扩展。

## 4. 核心原则：产品等价，Eval 只负责隔离与测量

Eval 可以增加隔离、不可变身份、Trace、来源证明和结果验证，但不能减少 Agent 能看到的审查内容、工具能力或产品预算。

### 4.1 唯一 Agent 能力来源

以下配置必须来自产品 Agent Execution Profile，并作为一个整体冻结：

- Provider、模型和 Base URL 身份；
- System Prompt、Review Contract 与工具定义摘要；
- Reviewer Loop；
- Reviewer Mode 与 Portfolio；
- 每个 Risk Profile 展开的 Reviewer 数、轮数、工具调用数、token 和时间预算；
- Context 与输出上限；
- Reconciler 与 Completion 配置；
- Shell、网络、仓库写权限和 Memory 权限。

Eval Run Config 只引用、持久化并校验该 Profile。Eval 不得为上述字段维护一套独立默认值，也不得在 Adapter 内静默降级。

Agent Execution Profile 不是另一套新配置系统。它是现有产品 CLI 参数、模型阶段配置、Risk Profile、工具目录与 Session 持久化配置的 canonical projection。Eval 的 `AgentConfigSnapshot` 保存该 projection 及 digest；Current Agent Adapter 在产品 Session 建立后用实际持久化值重建同一 projection 并比较 digest。实现不得为了“统一配置”复制一套产品默认值到 Eval 模块。

当前公共能力 Profile 明确绑定：

```text
reviewer_loop = agent-loop
shell = unavailable
network = provider_only
repository = read_only
run_safe_check = unavailable
```

`provider_only` 表示 Runtime 可以连接配置的模型服务；Reviewer 没有浏览器、HTTP 或任意网络工具。Repository acquisition 可以在 `prepare` 阶段联网，Trial 中的 Reviewer 不能借此访问互联网。

### 4.2 Eval 外层预算

Eval 仍可持有进程总超时、Artifact 总字节数和数据根容量等基础设施保险，但这些限制不得先于产品合法预算终止 Agent。

外层总超时必须由冻结产品 Profile 的阶段预算计算，并包含编排、持久化和有限 Provider retry 的余量。它不是另一套 Reviewer 深度策略。

基础设施容量不足时，Run 标记为无效或 `infrastructure_failure`；不得只删除较大的 Case 后发布一个存在选择偏差的分数。

### 4.3 等价校验

Current Agent Adapter 在评分前必须验证实际 Session 中记录的模型、工具、Loop、预算和 Prompt/Contract 身份与冻结 Profile 一致。任何漂移都属于兼容性或协议失败，不进入能力评分。

## 5. Repository Review Closure

### 5.1 当前问题

`repository-materializer-v2` 当前把 Base 与 Head 的完整 Git 可达闭包作为缓存权威。对于历史很长但本次 Diff 很小的仓库，这会读取、验证并写出大量与审查无关的祖先 tree/blob。固定 `MAX_GIT_OBJECTS = 100_000` 只是最先暴露的症状；简单提高常量会继续增加准备时间、内存、D 盘空间和 loose object 数量。

### 5.2 Review Closure 内容

新的 Hardened Review Closure 只保存产品现有审查工具需要的 Git 内容：

1. 精确 Base commit 与 Head commit；
2. Base 完整 tree/blob 快照；
3. Head 完整 tree/blob 快照；
4. Base 与 Head 之间为提交说明、关系验证和 `base..head` 语义所需的 commit metadata；
5. Base/Head Diff、对象格式和来源绑定所需的身份信息。

Base 之前且不属于审查边界的历史 commit 可以作为已验证边界引用；它们的历史 tree/blob 不进入 Review Closure。中间提交如果只用于提交说明和关系验证，不要求复制其完整历史快照。Closure 验证仍校验所有已保存对象的 canonical object hash、类型和引用关系，并完整验证 Base/Head 两棵树。

该 Closure 必须继续支持产品当前实际使用的操作：

- `git show <base|head>:<path>`；
- Base/Head 文件对比和 Diff；
- Base/Head `git grep`；
- Base/Head AST/Repository Intelligence；
- Base 到 Head 的相关提交信息；
- Head snapshot Quality Gates；
- Repository Evidence 与 Location Replay。

### 5.3 资源语义

Git object 数量成为观测统计，不再成为公共 Case 的固定淘汰条件。实现使用流式枚举和流式对象处理，不能因为取消数量门槛而把全部对象一次性加载到无界内存。

数据根磁盘、总字节数、准备总时间和文件系统节点数属于 Harness capacity。若完整选定 Suite 无法在声明的 capacity 中准备，整个 Run 不具备发布资格；不得把超限 Case 当作 Agent Recall=0，也不得静默跳过。

现有 Hooks 禁用、外部 Filter 禁止执行、凭据隔离、Base/Head 固定、每 Trial 独立工作区、路径逃逸防护与来源证明继续保留。安全策略如果拒绝了产品本可读取的 Case，必须作为显式 `unsupported/infrastructure_failure` 报告，并使完整 Run 无效，而不是改变 Agent 能力分数。

### 5.4 版本与缓存

Review Closure 改变了 Repository 内容身份的计算范围。实现必须提升 Logical Git Source/Repository Cache Policy 版本，使旧 Full-history cache、旧 acquisition binding 和新 Review Closure 不会互相复用。

Agent 可见的 Base/Head Materialization 与 Evidence Replay 合同保持不变，因此除非实现证明 Wire Contract 的可观察语义发生变化，不要求仅因内部 Closure 缩小就重建全部 Core Suite Case。任何未提升版本却复用旧缓存的实现均不合格。

## 6. 无人值守 Intent 策略

### 6.1 单一判定规则

Intent 策略不按数据集名称硬编码，而按 Case Authority 决定：

```text
intent_truth.scorable = false
and no scripted clarification answer
    -> benchmark_auto_accept

intent_truth.scorable = true
or scripted clarification answer exists
    -> normal clarification protocol
```

AACR/SWE 当前均进入第一条路径。Core Regression 的 18 个 Case 均保留可评分 Intent Truth，其中需要澄清的 Case 继续使用脚本答案。

### 6.2 Auto-accept 行为

公共 Case 进入 `awaiting_user` 时，Harness 使用现有 Clarification/Resume 边界自动确认每个 material question 绑定的有效 inferred claim。确认后：

- 最终 active claim 的 `source` 为 `explicit`；
- 最终 active claim 的 canonical origin/basis 为 `benchmark_auto_accept`，而不是 `user_confirmation`；
- IntentStatus 按确认后的 claim 重新计算；
- Reviewer、Review Brief 和 Eval Submission 看到同一份确定 Intent；
- 最终有效 Intent 中不保留 `source=inferred`；
- Run/Intent decision metadata 记录 `confirmation_basis=benchmark_auto_accept`，不得伪称真人回答；
- 策略版本进入 Agent/Run 身份，使用不同策略的结果不能直接比较。

真实产品交互行为不变：非 Eval 用户场景仍然是 inferred、询问、用户确认后 explicit。

## 7. Agent-loop 产品等价执行

公共能力 Profile 必须显式向产品 CLI 传入 `--reviewer-loop=agent-loop`。Adapter 不依赖 CLI 默认值，也不允许 Single-shot 作为兼容回退。

Agent-loop 使用产品现有 Tool Gateway。当前 Reviewer 工具保持：

```text
search_code
read_range
compare_base_head
list_symbols
inspect_symbol
find_references
query_project_memory
```

本版本不补充 `run_safe_check`，也不增加 Bash 或网络工具。现有 Runtime 自动 Quality Gates 继续运行并把结果作为 Observation 提供给产品流程。

Trace 必须记录实际 Loop、模型轮次、工具调用、终止原因和 Runtime budget。Profile 声明 `agent-loop` 但实际 Session 使用 Single-shot 时，Trial 为协议失败。

## 8. 权威 Completion 与 Submission 映射

### 8.1 两类 completed

`SessionManifest.status=completed` 表示生命周期终止，不表示 Review Contract 已完成。`completion.json` 是审查结论的权威来源。

Adapter 只有在 Session artifact 已验证并且 Completion 允许时，才生成 Completed Submission：

| 产品 Completion | Eval Submission | Finding 是否评分 |
| --- | --- | --- |
| `completed` | `completed` | 是 |
| `completed_with_uncertainties` | `completed`，保留 uncertainties | 是 |
| `blocked` | `blocked / agent_blocked` | 否；Recall 按 failure-as-miss |
| `budget_exhausted` | `blocked / agent_blocked` | 否；Recall 按 failure-as-miss |
| Completion 缺失、未注册、漂移或格式错误 | `invalid_output / schema_mismatch` | 否 |
| Session 或产品进程失败 | `failed` | 否；Recall 按 failure-as-miss |

只有 Core Reviewer 完成、必需 Contract coverage 合格且 Completion 为 `completed` 或 `completed_with_uncertainties` 时，零 Finding 才是有效能力输出。

### 8.2 失败计分

本设计不改变现有 Metrics Policy：

- Agent failed/blocked 对 Issue Recall 和 Intent Recall 使用 `failure_as_miss`；
- 对 Precision 使用 `failure_excluded`，因为失败执行没有可解释的 generated denominator；
- Agent Failure Rate 单独增加；
- 基础设施失败不进入 Agent 能力分数；
- 成功运行但漏掉全部 Finding 与执行失败都可能得到 Recall=0，但 Failure Rate 和状态必须能够区分两者；
- 无 expected finding 的负样本 Recall 分母为零，因此不能用 Recall 单独表达执行失败。

### 8.3 Reconciler 边界

Eval 不给 Risk、Runtime、Reconciler、Session 或 Memory 单独打能力分。它们是产品内部机制。Reconciler 的降级、失败或剩余分歧通过产品 Completion 影响 Submission 终态。

`local_only` 本身不是失败；Core Reviewer failed 导致 Completion blocked 才是 Agent 未完成。Eval 不得把空 Candidate Catalog 误解为 Reconciler 删除了 Finding。

## 9. Evaluator 外部边界

公共 Benchmark Evaluator 继续主要观察：

- 最终 Finding 内容和位置；
- Finding 与 Ground Truth 的确定性/语义匹配；
- Evidence 是否真实、可重放并支持 Finding；
- Agent/Judge Failure；
- token、工具调用、时间和成本。

本版本不因为此前无效的零 Finding 结果而降低 Finding 质量要求。首要问题是 Agent 协议和终态包装错误，而不是已经证明 Matcher 过严。

修复后若有效 Agent Finding 被当前 Matcher 系统性错误拒绝，再以 Alibaba AACR-Bench 的位置匹配加 LLM/Embedding 语义匹配作为参考，单独提出有证据的 Matcher 修改。不得在没有有效输出前同时放宽多个规则。

## 10. 旧 Run 处理

以下已有 Run 不得作为能力基线：

- AACR `run-735b6d6948646b3312df09daf46a95c9753cc2aed24b81a80a56853af816a224`；
- SWE `run-6204c8a9723f48bb95f361c0edb5d4e5593fe65557109a8f32a28ad0fdf5e2c1`。

旧 Artifact 保持不可变，不回写或伪造新状态。新的报告或分析层可以把它们标记为 `invalid_for_capability_baseline`，原因是 Reviewer 协议失败与 Completion 状态误映射。

## 11. 实现边界

本次实现分为四个紧密关联的工作单元：

1. Repository Review Closure 与 cache/version identity；
2. 公共无 Authority Intent 的 benchmark auto-accept；
3. 产品 Agent Profile/agent-loop 显式绑定与等价校验；
4. Completion authority、Submission status 与 failure-as-miss 验证。

这四项共同恢复一个有效的端到端公共测评。不得只改 Submission 映射后继续使用 Single-shot，也不得只提高 Git object 常量后继续复制完整历史。

## 12. 验证策略

### 12.1 本地聚焦验证

使用小型本地 Git fixture 验证：

- Base/Head tree/blob 完整可读；
- 历史中只存在于审查边界外的旧 tree/blob 不进入 Review Closure；
- Base/Head Diff、grep、file replay 和提交范围仍可用；
- 新旧 cache policy 不互相复用；
- `intent_truth.scorable=false` 自动确认并最终保存 explicit；
- Core 可评分 Intent 与脚本澄清保持原行为；
- Profile 显式绑定 agent-loop，配置漂移被拒绝；
- Completion 四种终态正确映射；
- failed/blocked 的 Recall 为 failure-as-miss，Failure Rate 同时增加；
- infrastructure failure 不进入 Agent 能力分数。

不新建测试框架，不为了形式覆盖重跑无关大型测试集合。

### 12.2 公共仓库 prepare 验证

先只运行 Repository prepare，不调用 DeepSeek。原首批 10 个 AACR Case 必须全部完成准备，或者整个 Run 以明确 infrastructure failure 结束；不允许 1 个成功、9 个静默排除后继续发布分数。

### 12.3 模型 Smoke

prepare 验证通过后，分别运行：

- 1 个 AACR Case；
- 1 个 SWE-PRBench Case。

两个 Trial 都必须证明：

- 实际 Loop 为 agent-loop；
- Reviewer 没有协议解析失败；
- Completion 与 Submission 状态一致；
- Trace、Intent、Finding/Evidence 和失败状态可重放；
- 指标能够区分真实完成、漏报与执行失败。

### 12.4 第一批能力基线

Smoke 通过后再运行完整首批 10 个 AACR Case，避免在协议仍无效时浪费模型额度。该批结果作为无 Bash、无 Reviewer 网络能力的第一个可信产品基线。

## 13. VNext：Bash 与网络能力同步扩展

当前基线有效后，下一版本可以为产品 Agent 增加 Bash/命令与网络能力，但必须满足：

1. 能力首先属于产品 Runtime/Tool Gateway，不在 Eval 中单独仿造；
2. 产品与 Eval 使用同一个版本化 `CapabilityProfile`；
3. Shell、网络、文件系统、预算和沙箱策略完全写入 Profile identity；
4. Eval Session 与冻结 Profile 不一致时拒绝评分；
5. 使用相同 Case、模型、Prompt 和非能力预算与当前基线做一对一比较；
6. 报告 Recall、Precision、Evidence、Failure Rate、成本和时延变化。

公共 Benchmark 的标注和原始 PR 评论可能在线公开。VNext 不能直接开放可搜索 Ground Truth 的任意网络，否则成绩失效。网络能力需要单独设计反泄漏边界，例如允许经过批准的依赖文档或注册表，同时阻断数据集、PR 评论和其他答案来源，并记录网络访问。Bash 必须限制在 Agent 可见工作区，不能读取 Eval 隐藏标注或控制目录。

VNext 需要独立 Spec；本设计只冻结同步扩展原则，不提前实现其权限模型。

## 14. 验收标准

本设计实现完成的必要条件为：

1. 公共 Agent Profile 只有产品侧一个能力来源，实际 Session 身份可验证；
2. AACR/SWE 不再运行 Single-shot；
3. 公共无 Authority Intent 最终为 explicit，并记录 benchmark auto-accept basis；
4. Core Intent Truth 与脚本澄清不受影响；
5. 原 10 个 AACR Case 不再因固定 `100,000` Git object 门槛被逐 Case 淘汰；
6. Completion blocked/failed 不再包装为 Completed Submission；
7. Agent failure 对 Recall 贡献 miss，同时正确增加 Failure Rate；
8. 基础设施失败不会伪装为 Agent 低分或被静默排除；
9. AACR 与 SWE 各至少一个真实模型 Trial 形成有效、可重放终态；
10. 完整 10 Case AACR 基线只在 Smoke 有效后启动；
11. 当前版本没有新增 Bash、Reviewer 网络或 `run_safe_check`；
12. Bash/网络同步扩展明确留在独立 VNext 设计中。

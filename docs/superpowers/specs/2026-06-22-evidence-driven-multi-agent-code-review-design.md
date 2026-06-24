# Evidence-Driven Multi-Agent Code Review Agent 设计

- 日期：2026-06-22
- 状态：已确认，待实施计划
- 项目根目录：`D:\Agent\code review agent`
- M1 形态：面向 Python Git 仓库的本地交互式 CLI

## 1. 背景与问题

AI Coding Agent 显著提高了代码生产速度，但人类理解、验证和审查代码的速度没有同步提高。传统的单轮
Diff Review 容易产生以下问题：

- 只看到改动文本，没有调查完整仓库中的调用方、测试和行为影响。
- 能发现局部代码异味，却难以验证实现是否符合真实需求。
- PR 标题、描述和 Issue 可能缺失关键目标、约束与设计理由，形成 Intent Debt。
- 同一个模型、同一条推理路径容易遗漏问题或产生相关性盲点。
- AI 输出容易被误当成最终裁决，导致人类没有形成独立理解。
- Review 结果缺少可追溯证据，无法区分事实、推断和未知信息。

本项目不做另一个“读取 Diff 后自动评论”的机器人，也不优先复制完整 PR 工作台。项目核心是构建一个
能够自主调查代码变更、验证意图和影响范围，并向人类提供可追溯证据的多 Agent Review 系统。

## 2. 产品定义

> 一个 Repository-Aware、Evidence-Driven 的多 Agent Code Review 系统。它重建并校验变更意图，
> 根据整体风险分配审查资源，组织多个相互独立的 Reviewer Agent 调查完整仓库，协调证据与分歧，
> 最终为人类 Reviewer 生成可审计的 Review Brief。

系统的目标不是替人点击 Approve 或 Merge，而是帮助人类回答：

1. 这次修改想实现什么？
2. 实际修改是否符合声明意图与验收条件？
3. 变更会影响哪些代码路径、调用方和既有行为？
4. 测试和其他验证是否真正证明了预期行为？
5. 已确认的问题、被排除的疑点和剩余未知项分别是什么？
6. 人类应该优先阅读和判断哪些部分？

## 3. 核心原则

### 3.1 AI 是Sensor，不是Judge

- Agent 可以给出非约束性的 `approve`、`needs_work`、`manual_review` 建议。
- Agent 不自动发布评论、Approve、Reject 或 Merge。
- 最终 Merge 决策和责任始终属于人类。

### 3.2 系统定义审查责任，模型决定调查路径

- `Review Contract` 固定每次合格审查必须回答的高层问题。
- LLM 根据当前 PR 动态提出风险假设，不能通过规则穷举所有风险类型。
- Runtime 负责预算、权限、状态、完成条件和证据校验。
- Reviewer Agent 自主选择代码搜索、符号查询、文件阅读和安全检查等工具。

### 3.3 多 Reviewer 共享事实，不共享第一轮推理

- Reviewer 共用仓库索引、代码工具和工具结果缓存。
- 每个 Reviewer 拥有独立上下文、假设、证据集、调查轨迹和模型配置。
- 第一轮独立调查期间，Reviewer 看不到其他 Reviewer 的结论，避免锚定。
- 独立调查完成后才进入 Evidence Reconciliation。

### 3.4 风险等级控制调查强度，不写死调查内容

- 风险等级决定 Reviewer 数量、轮次、工具预算、证据门槛和人工要求。
- 具体调查方向由 Reviewer 根据意图、变更和仓库证据动态形成。
- 不建立试图枚举所有风险种类的固定触发器规则库。

### 3.5 完整信息保存在系统中，相关信息才进入模型窗口

- 原始工具结果和完整轨迹进入 Evidence Store 与 Session Store。
- Context Assembler 只向当前 Reviewer 提供与使命、假设和当前步骤相关的内容。
- 上下文压缩不能删除系统状态或证据，只能减少模型当前看到的历史。

## 4. M1 范围

### 4.1 包含

- 本地 Git 仓库输入：`base revision` 与 `head revision`。
- Python 项目优先。
- 交互式 CLI，支持非交互模式。
- Intent Packet 构建、充分性判断和用户澄清。
- Deterministic Quality Gates。
- Initial Risk Assessor 与 Final Risk Reassessment。
- Portfolio Planner。
- 多个独立 Reviewer Agent。
- Repository Intelligence Layer。
- Evidence Store、Session Store 与分层记忆。
- Evidence Reconciler 与全局 Completion Checker。
- Markdown、JSON 和 JSONL 运行产物。
- 暂停、恢复、预算限制、失败降级。
- 完整可观测数据，为后续 Agent Eval Harness 留出基础。

### 4.2 不包含

- GitHub、GitLab 或 Bitbucket 平台接入。
- 自动发布 PR 评论、Approve 或 Merge。
- Web 或桌面 GUI。
- 自动修改被审查代码。
- 完整的 Agent 能力评测体系。
- 多语言代码库的完整 LSP 支持。
- 长期记忆的全自动写入。

后续里程碑：

- M2：GitHub PR 输入、Issue/CI 采集、Inline Finding 草稿。
- M3：TypeScript/LSP 扩展、多模型异质 Reviewer、团队规则与项目知识增强。
- 独立 Eval 里程碑：在系统学习 Agent Evaluation 方法后单独设计。

## 5. 输入模型

用户入口：

```bash
review-agent review --base main --head HEAD
```

可选补充：

```bash
review-agent review \
  --base main \
  --head HEAD \
  --intent "为支付回调增加幂等保护" \
  --focus "重复执行、事务和失败重试"
```

`--intent` 和 `--focus` 只是用户提供的自然语言线索，不要求用户提交结构化 Intent Packet。用户可以完全不传意图；系统仍应根据 Diff、提交信息、代码上下文、测试、文档和历史记录构建内部意图模型。

内部统一为 `ReviewRequest`：

```yaml
repository_path: /path/to/repo
base_revision: <commit>
head_revision: <commit>
title: optional
description: optional
linked_requirements: []
user_intent: optional
review_focus: optional
project_rules: []
existing_ci_evidence: []
provider_configuration: {}
```

真正的审查对象不是一段孤立 Diff，而是：

> 一个仓库从 Base 状态变化到 Head 状态后，是否正确实现了声明或系统推断并明确标记的意图，并保持既有系统行为可接受。

Diff 提供调查起点，完整仓库提供影响与正确性的证据。

## 6. Intent Packet

### 6.1 定位

Intent Packet 是系统内部维护的意图模型，不是用户必须提交的输入格式，也不是必须填满的表单。

系统应先根据用户可选输入、Diff、提交信息、代码上下文、测试、文档、Issue/ADR 和历史记录自动构建 Intent Packet，并持续维护其充分性状态。用户提供的 `--intent`、PR 描述或 Issue 只是意图来源之一。

Intent Packet 的主要作用是让系统判断“当前意图理解是否足够支持可靠审查”。只有当缺失信息可能改变审查结论时，系统才向用户提出具体问题。

最小核心内容：

```yaml
goal: 为什么要改
acceptance_criteria: 改完后哪些行为必须成立
scope: 哪些内容属于本次修改范围
constraints: 哪些行为、接口或边界不能破坏
```

可选内容：

```yaml
design_decision: 采用的设计
alternatives_rejected: 放弃的方案与原因
linked_issue: 需求来源
author_evidence: 作者提供的验证
rollback_expectation: 回滚要求
```

### 6.2 来源与可信度

每条意图信息必须区分：

```text
declared：用户或作者明确声明
linked_source：来自 Issue、ADR 或验收文档
inferred：Agent 根据代码和提交历史推测
```

`inferred` 不能冒充真实需求，只能帮助提出问题和规划调查。

### 6.3 充分性状态

- `sufficient`：足以验证主要行为，剩余未知项不会改变核心审查判断。
- `partial`：可以继续审查，但部分结论受未知信息限制。
- `insufficient`：缺失信息会阻止可靠的语义性审查。

补齐意图的完成标志不是字段全部填写，而是：

- 主要目标与预期行为可验证。
- Scope 与关键约束明确。
- 没有阻塞性的内部矛盾。
- 高风险行为具有明确预期，或未知项已被显式标记。
- 剩余未知项已经记录影响和责任人。

### 6.4 用户询问

Agent 只提出可能改变审查结论的具体问题，例如：

- 重复支付回调应该返回成功并跳过，还是返回错误？
- 公共 API 是否必须兼容旧调用方？
- 历史数据是否需要被新版本继续读取？

用户可以回答，也可以选择 `continue with uncertainty`。未知答案本身是有效信息，必须进入报告。

## 7. Deterministic Quality Gates

确定性问题不交给 LLM 猜测：

```text
语法错误 -> parser/compiler
格式问题 -> formatter --check
类型错误 -> type checker
代码规范 -> linter
构建错误 -> build command
测试失败 -> test runner
依赖漏洞 -> dependency scanner
```

系统优先发现并复用仓库已有命令。M1 针对 Python 支持识别和运行适用的：

```text
python compile
ruff
mypy/pyright
pytest
仓库显式配置的安全命令
```

廉价检查在风险评估前运行。昂贵的完整测试、安全扫描或集成检查由风险等级与 Reviewer 调查按需触发。

Quality Gates 的完整输出进入 Evidence Store；模型先看到结构化摘要，需要时再按范围读取原始日志。

## 8. Review Contract

Review Contract 由系统预先定义并版本化。它定义每次合格 Review 必须回答的问题和最低证据要求，
不规定具体 Bug 类型或调查步骤。

M1 默认 Contract：

### 8.1 Intent Alignment

代码变更是否符合声明意图和验收条件？

最低证据：

- 意图或验收条件与代码变更的映射。
- 对应测试或其他验证证据。

### 8.2 Behavioral Correctness

修改后的核心行为是否正确，包括重要边界与失败路径？

最低证据：

- 相关实现路径。
- 适当的静态或动态验证。

### 8.3 Regression Safety

变更是否破坏未修改的调用方、接口或既有行为？

最低证据：

- 影响范围、定义/引用或调用方调查。
- 相关回归验证。

### 8.4 Test Adequacy

测试是否真实证明预期行为，而不是只适配当前实现？

最低证据：

- 测试与验收条件的对应关系。
- 缺失、删除、弱化或被修改断言的说明。

### 8.5 Unresolved Uncertainties

是否仍有可能改变审查结论的重要未知信息？

最低证据：

- 未知项及其来源。
- 对结论的影响。
- 建议的确认人或下一步。

每项输出：

```yaml
status: satisfied | concern | unverified | not_applicable
summary: ...
evidence_refs: [...]
confidence: high | medium | low
```

`not_applicable` 必须提供理由。模型不能删除 Contract 项。

## 9. 整体风险评估

### 9.1 通用维度

风险评估不穷举领域风险类型，而使用五个通用维度：

- `impact`：失败造成的后果。
- `blast_radius`：影响的模块、调用方、数据或用户范围。
- `reversibility`：错误是否容易发现、隔离和回滚。
- `uncertainty`：意图、关系或运行条件中存在多少重要未知。
- `verification_strength`：现有测试和证据的可靠程度。

### 9.2 Initial Risk Assessor

Initial Risk Assessment 是轻量分诊阶段，不是正式审查。它的目标是用较小上下文判断审查深度，而不是提前完成 Review。

Runtime 先采集确定性特征：

- Diff 规模与文件分布。
- changed symbols。
- 调用方与依赖关系概览。
- 公共接口变化。
- 测试是否同步修改。
- Quality Gates 和 CI 状态。
- Intent Packet 充分性。

Runtime 将这些特征整理成轻量 `Risk Assessment Packet` 传给 Risk Assessor。这个 Packet 可以包含高信号 Diff 片段、关键符号、调用方概览、测试/质量门状态和意图未知项，但不能包含完整仓库、完整 Session Store 或正式 Reviewer 调查历史。

Risk Assessor Agent 只基于该轻量 Packet 判断风险等级、风险理由和建议审查画像。它允许提出少量定向查询请求，但不负责完整 Review。

输出：

```yaml
risk_level: low | medium | high | critical
dimensions:
  impact: ...
  blast_radius: ...
  reversibility: ...
  uncertainty: ...
  verification_strength: ...
reasoning: [...]
evidence_refs: [...]
unknowns: [...]
suggested_review_profile:
  reviewer_count: ...
  required_focus: [...]
  budget_hint: ...
```

Risk Assessor 不能提交正式 Finding、Approve/Reject 或自动合并。

### 9.3 风险等级的影响

| 等级 | Reviewer 配置 | 调查预算 | 人工要求 |
|---|---|---|---|
| LOW | Core Reviewer | 较小 | 快速人工确认 |
| MEDIUM | Core + Adversarial | 中等 | 正常人工 Review |
| HIGH | Core + Adversarial + 动态专项 Reviewer | 较高 | 人工重点 Review，必要时领域负责人 |
| CRITICAL | 多个独立专项 Reviewer | 最高且证据门槛最严 | 强制领域负责人或安全人员参与 |

初始默认上限可配置，例如：

```yaml
low:
  reviewer_count: 1
  max_turns_per_reviewer: 6
medium:
  reviewer_count: 2
  max_turns_per_reviewer: 10
high:
  reviewer_count: 3-4
  max_turns_per_reviewer: 16
critical:
  min_reviewer_count: 4
  max_turns_per_reviewer: 24
```

这些数字是资源上限，不是固定调查清单。

风险等级首先是 Runtime 调度参数。Runtime 根据风险等级决定 Reviewer 数量、专项任务、预算、上下文范围、工具权限和 Completion 门槛。Reviewer 不依赖抽象的 `risk_level` 自行决定审查深度；它们接收的是 Runtime 已经展开后的具体 Assignment。

### 9.4 Final Risk Reassessment

多 Agent 调查后重新评估，并区分：

```text
Inherent Risk：变更本身潜在影响。
Review Confidence：证据支持审查结论的程度。
Residual Concerns：完成调查后仍然存在的风险。
```

高风险变更不会因为测试充分而被改写成低风险，只能提高 Review Confidence、降低 Residual Concerns。

## 10. 多 Agent 架构

```text
Review Request
  -> Intent Manager
  -> Quality Gate Runner
  -> Initial Risk Assessor
  -> Portfolio Planner
  -> Reviewer Agents in parallel
  -> Evidence Reconciler
  -> Final Risk Reassessment
  -> Global Completion Checker
  -> Review Brief
```

### 10.1 Review Orchestrator

Orchestrator 是“确定性 Runtime + LLM Portfolio Planner”，不是拥有最终裁决权的总指挥 Agent。

Runtime 是系统代码中的确定性控制层。它不靠模型自觉遵守规则，而是在代码层面控制状态流转、工具网关、权限、预算、上下文装配、证据校验和完成判定。

Runtime 固定负责：

- 阶段顺序和状态机。
- Intent Packet 完整性。
- Quality Gates。
- 构造轻量 Risk Assessment Packet。
- 根据风险等级生成审查深度、最低预算和完成门槛。
- 至少启动 Core Reviewer。
- 并发、token、工具和时间预算。
- Reviewer 上下文隔离与模型调用装配。
- 权限与证据校验。
- 暂停、恢复和 Completion Checker。

Portfolio Planner 动态负责：

- 根据意图、变更和风险理由选择审查视角。
- 为每个 Reviewer 生成明确的使命。
- 在风险等级允许的预算内提议专项 Reviewer。

Planner 不能跳过 Core Reviewer、降低风险等级、删除 Review Contract、扩大预算、授予写权限或自动合并。

### 10.2 Reviewer Portfolio

系统固定提供：

- `Core Reviewer`：意图一致性、行为正确性、调用方、回归与测试。
- `Adversarial Reviewer`：异常路径、边界条件、错误假设和生产失败模式。

高风险 PR 可以动态生成专项角色，例如：

```yaml
role: Async Lifecycle Reviewer
mission: 调查后台任务的取消、重试、资源释放和状态一致性
```

角色不是固定风险枚举。M1 可使用同一模型，但从第一天支持每个 Agent 独立配置 Provider、模型和预算。

### 10.3 Reviewer Assignment

```yaml
role: Adversarial Reviewer
mission: 寻找异常路径、边界条件和错误假设
intent_packet: ...
assignment_reason:
  - changed public API behavior
  - weak regression coverage
assigned_contract:
  - behavioral_correctness
  - regression_safety
required_checks:
  - inspect direct callers or record why unavailable
  - inspect related regression tests
provided_context:
  evidence_refs: [...]
  code_ranges: [...]
budget:
  max_turns: 12
  max_tool_calls: 30
permissions:
  repository: read_only
  commands: safe_checks_only
```

Reviewer 不能自行改变使命、Contract、预算和权限。风险等级可以保留在 Runtime state 中；Reviewer 看到的是由风险等级展开后的任务原因、检查清单、证据材料和完成条件，而不是依靠 `risk_level` 自主决定调查深度。

### 10.4 Reviewer Agent Loop

每个 Reviewer 运行相同的通用 ReAct Runtime：

```text
理解 Assignment
-> 提出风险假设
-> 选择工具调查
-> 收集结构化证据
-> 确认、否定或更新假设
-> 发现新线索后继续
-> 申请结束
```

动作包括：

```text
search_code
read_file/read_range
inspect_symbol
find_references
find_callers/find_callees
find_related_tests
inspect_git_history
run_safe_check
record/update/reject_hypothesis
submit_finding
record_uncertainty
request_human_context
finish_assignment
```

### 10.5 单个 Reviewer 结束条件

模型只能申请结束，Runtime 决定是否接受。

Runtime 检查：

- Assigned Contract 每项都有结论。
- 所有风险假设进入 `confirmed`、`rejected` 或 `unresolved`。
- Finding 引用有效证据。
- Reviewer 回答了自己的 Mission。
- 关键主张使用了适当的仓库级工具，或解释了无法执行的原因。
- 未知项被显式记录。

结束状态：

- `completed`：任务得到充分回答。
- `partial`：预算耗尽或部分信息不可获得，但结果仍可使用。
- `blocked`：必须依赖用户、环境或外部信息才能继续。
- `failed`：模型或 Runtime 错误导致任务失败。

### 10.6 Reviewer 输出

```yaml
contract_assessments: [...]
confirmed_findings: [...]
rejected_hypotheses: [...]
uncertainties: [...]
investigation_summary: ...
status: completed | partial | blocked | failed
```

## 11. Repository Intelligence Layer

### 11.1 设计原则

系统不启动时读取整个仓库并塞入模型，也不以向量数据库作为第一核心。采用 Claude Code 式按需探索：

```text
定位文件
-> 搜索符号
-> 读取相关范围
-> 查询定义、引用和调用层级
-> 阅读测试与历史
-> 根据线索继续外扩
```

### 11.2 工具层

M1 通用导航工具：

```text
list_files/glob
search_code
read_file/read_range
get_diff
git_log
git_blame
```

Python 语义工具：

```text
list_symbols
go_to_definition
find_references
find_callers/find_callees
find_related_tests
```

实现优先级：

```text
ripgrep：快速精确文本搜索
Python AST：符号、导入、调用和测试结构
LSP：可用时提供更准确的定义、引用和调用层级
Git：变更、历史、归属和 Base/Head 证据
```

LSP 不可用时降级到 AST + ripgrep，并降低对应结论的 confidence。

### 11.3 共享事实层

所有 Reviewer 共享：

- Base/Head 快照。
- Diff 与 changed symbols。
- 文件树和符号索引。
- LSP/AST 查询能力。
- Git 历史与 Quality Gates。
- 只读工具结果缓存。

不共享第一轮中的：

- Reviewer 假设。
- Reviewer Findings。
- Reviewer 自由文本结论。
- Reviewer 调查摘要。

### 11.4 Review Knowledge Graph

系统维护本次 Review 的局部关系图：

```text
Changed Symbol
  -> defined_in
  -> referenced_by
  -> calls/called_by
  -> implemented_by
  -> tested_by
  -> constrained_by
  -> satisfies_acceptance_criterion
```

Knowledge Graph 保存结构化事实和证据引用，不保存模型隐藏思维链。

## 12. Evidence System

每条证据必须具有稳定 ID 和来源：

```yaml
evidence_id: E-42
source: lsp.find_references
revision: head@abc123
path: src/auth/service.py
line_start: 81
line_end: 93
content_hash: ...
raw_artifact_ref: optional
```

工具返回两种视图：

- Raw Evidence：完整保存，供恢复、审计和重新读取。
- Context View：结构化、截断或摘要后提供给模型。

Finding 必须包含：

```yaml
severity: blocker | high | medium | low
confidence: high | medium | low
claim: ...
path: ...
line: ...
evidence_refs: [...]
impact: ...
suggested_action: ...
verification_performed: [...]
```

证据不足的内容只能进入 `uncertainties`，不能伪装成 Finding。

## 13. Evidence Reconciliation

### 13.1 确定性处理

- 校验证据 ID、revision、路径和行号。
- 检查 Finding 字段完整性。
- 按路径、行号和符号进行初步聚类。
- 检查 Contract 覆盖缺口。
- 检查证据是否过期或不属于当前 Head。

### 13.2 Reconciler Agent

- 判断 Findings 是否语义重复。
- 区分同一位置上的不同问题。
- 比较 Reviewer 对同一事实的冲突解释。
- 判断证据是否真正支持主张。
- 生成有界的补充调查任务。

不使用多数投票。一个 Reviewer 独立发现的严重问题，只要证据可靠，就必须保留；多个 Reviewer 重复同一个
无证据猜测，也不能提高其真实性。

### 13.3 补充调查

冲突不能靠 Reconciler 猜测解决，而应派发定向 Assignment：

```yaml
question: tenant_id 是否进入所有缓存键的读写路径
required_evidence:
  - key construction
  - read/write callers
  - cross-tenant tests
assigned_reviewer: Security Reviewer
```

补充调查有严格轮次和成本上限。

### 13.4 合并结果

```yaml
canonical_findings: [...]
rejected_findings: [...]
conflicts_resolved: [...]
remaining_disagreements: [...]
contract_coverage: [...]
evidence_quality: ...
```

被拒绝的 Finding 记录原因：

```text
duplicate
unsupported_claim
stale_evidence
contradicted_by_test
outside_review_scope
```

## 14. 全局 Completion Checker

整个 PR 审查完成前必须满足：

1. Intent Packet 为 `sufficient` 或明确的 `partial`。
2. Quality Gates 已运行，或记录无法执行的原因。
3. 风险等级要求的 Reviewer 已完成或显式失败。
4. Review Contract 每项都有合并结论。
5. Findings 完成证据校验和去重。
6. Reviewer 冲突已解决或明确保留。
7. 阻塞性未知项已询问用户或标记责任人。
8. Final Risk Reassessment 已完成。

整体状态：

- `completed`
- `completed_with_uncertainties`
- `blocked`
- `budget_exhausted`
- `failed`

非约束性建议：

- `needs_work`：存在证据充分的阻塞 Finding。
- `manual_review`：存在重要未知项、冲突、高风险人工判断或关键 Reviewer 失败。
- `approve`：没有阻塞 Finding，Contract 覆盖和验证达到要求。

## 15. 记忆系统

### 15.1 Review Session Memory

保存一次审查的完整运行状态：

```text
ReviewRequest
Intent Packet
风险评估
Reviewer Assignments
假设、工具调用和证据
Findings 与冲突
Completion 状态
最终报告
```

支持暂停、恢复、审计和增量重审。

### 15.2 Repository Knowledge

按 commit hash 缓存：

```text
文件与符号索引
定义、引用和调用关系
测试映射
项目配置
Git 历史摘要
```

所有事实绑定 revision。HEAD 变化后，相关事实必须重新验证。

### 15.3 Durable Project Memory

保存跨 PR 仍然有效的、带来源的项目知识：

```text
架构边界
业务不变量
人工批准的 Review 规则
兼容性要求
常用验证命令
历史事故经验
高风险模块说明
```

示例：

```yaml
statement: 金额计算必须使用 Decimal
scope: payments/**
source: ADR-017
confidence: verified
approved_by: human
valid_from: <commit>
```

Agent 只能提出 `Memory Candidate`。经过来源校验、去重和人工确认后才能进入长期记忆。

### 15.4 Review Feedback Memory

记录：

```text
Finding accepted/rejected
误报原因
severity 调整
Reviewer 遗漏
人工最终决定
```

这些记录用于未来校准和正式评测，不会简单转化为“以后不要报告类似问题”的自动规则。

### 15.5 不持久化

- 模型隐藏思维链。
- 无来源的项目事实。
- 未验证猜测。
- 只属于当前 PR 的临时结论。
- 已失效的代码片段。

## 16. Context & Model Invocation System

本系统不改变标准模型调用结构。一次模型调用仍然只由四类输入组成：

```text
system：系统提示词
tools：工具定义
messages：对话历史、任务材料、工具结果、附件和证据摘要
parameters：模型、输出上限、reasoning effort、temperature、response schema 等参数
```

Intent Packet、风险评估、证据、项目记忆和 Reviewer Mission 不是额外输入类别。它们只是 Runtime 按需选择后，放入 `messages` 的任务材料。

### 16.1 Model Invocation Envelope

每次模型调用由 Runtime 构造 `ModelInvocationEnvelope`：

```yaml
system: ...
tools: [...]
messages: [...]
parameters:
  model: ...
  max_output_tokens: ...
  reasoning_effort: ...
  temperature: ...
  tool_choice: ...
  response_schema: ...
  trace_id: ...
```

`system` 包含不可被仓库内容覆盖的规则：角色边界、权限、安全边界、工具使用规则、Review Contract、证据要求和输出格式。Review Contract 在 prompt 中可见，但硬约束由 Runtime 和 Completion Checker 执行。

`tools` 是本次调用允许使用的工具定义。Runtime 按 Agent 类型和阶段裁剪工具列表，并在工具网关中校验参数、权限、超时、输出大小和证据记录。

`messages` 承载当前任务材料和必要历史。Context System 的主要工作发生在这里：从系统状态中挑选最小充分材料，压缩后传给模型。

`parameters` 控制模型与调用行为，例如模型、输出上限、思考强度、结构化输出 schema 和 tracing metadata。

### 16.2 Context Payload Assembly

Context Payload 指 Runtime 放进 `messages` 的任务材料。它不是完整仓库，也不是完整 Session Store。

Runtime 从这些后端状态中选择材料：

- ReviewRequest。
- Intent Packet。
- Risk Assessment Packet 或风险评估结果。
- Assignment。
- Evidence Store 中的相关证据片段。
- Repository Knowledge 中的相关文件、符号、调用关系和测试映射。
- 当前 Agent 的调查摘要和最近工具结果。
- 已批准的项目记忆。
- 未解决未知项。

默认原则：

- 不把完整仓库放进模型窗口。
- 不把完整 Session Store 或 Evidence Store 直接传给模型。
- 不把其他 Reviewer 第一轮自由文本推理传给模型。
- 不传与当前阶段无关的风险、记忆、日志或代码。
- 能用 evidence/ref 表示的内容优先传引用；只有需要语义判断时才传片段。

### 16.3 Stage-Specific Minimal Context

不同阶段使用不同的最小上下文包：

| 阶段 | `messages` 中主要传入 | 明确不传 |
|---|---|---|
| Initial Risk Assessor | 轻量 Risk Assessment Packet：变更摘要、关键 Diff 片段、changed symbols、确定性信号、Intent 未知项 | 完整仓库、完整审查上下文、正式 Reviewer 调查历史 |
| Portfolio Planner | 风险等级、风险理由、变更地图、可用 Reviewer 类型、预算策略 | 大量代码细节、完整工具日志 |
| Reviewer Agent | Runtime 生成的具体 Assignment、相关代码片段、证据引用、必要工具结果、完成条件 | 抽象 `risk_level` 作为自主加深审查的提示、其他 Reviewer 第一轮推理、完整 Session Store |
| Evidence Reconciler | Findings、证据引用、冲突摘要、必要代码片段 | Reviewer 隐藏推理过程、无关原始日志 |
| Final Risk Reassessment | 已验证 Findings、Uncertainties、Quality Gates、Contract Coverage | 原始所有工具输出 |
| Completion Checker | 结构化覆盖摘要、失败/阻塞项、未决问题；多数检查由本地确定性逻辑完成 | 完整调查轨迹 |

风险等级首先留在 Runtime state 中，用来决定审查深度。Reviewer 接收的是风险等级展开后的 Assignment，例如任务原因、检查清单、证据材料、预算和完成条件；它不依赖 `risk_level: high` 这类抽象标签自行决定调查深度。

### 16.4 Reviewer Context Payload

Reviewer 每轮 `messages` 中的任务材料通常包括：

必需：

- 当前 Assignment：角色、Mission、任务原因、assigned Contract、required checks。
- 当前可用工具说明和工具调用规则摘要。
- Intent Packet 摘要，包括 declared / linked_source / inferred / unknown 的区分。
- 相关代码片段、Diff 片段、调用关系或测试映射。
- Evidence refs 与必要证据摘录。
- 当前预算、完成条件和不能完成的条件。
- 结构化输出 schema。

按需：

- 最近工具结果。
- 当前 Agent 自己的调查摘要。
- 未解决未知项。
- 已批准的项目记忆。
- 已确认 Finding 或需要复核的冲突。

不默认传：

- 完整文件树。
- 完整仓库内容。
- 完整历史对话。
- 完整工具日志。
- 其他 Reviewer 的第一轮假设和自由文本结论。
- 与当前 Assignment 无关的长期记忆。

### 16.5 Context Budget 与 Compaction

Context 预算用于 `messages` 中的任务材料，不包含 `system`、`tools` 和模型参数。

初始指导：

```text
Assignment 与意图：15%
当前代码证据：45%
调查状态摘要：20%
项目记忆与规则：10%
输出格式与工具结果预留：10%
```

Compactor 保留任务、未解决假设、结论、证据引用和下一步方向；压缩重复代码、过长日志、已完成搜索过程和已否定假设的细节。完整内容始终保存在 Session/Evidence Store，压缩只影响下一次模型调用可见的 `messages`。

## 17. 信任与安全

信任顺序：

```text
Runtime Policy / 权限 / Review Contract
人工批准的项目规则和记忆
用户本轮明确提供的意图
PR 描述、Issue、代码、注释、文档、日志、测试输出
外部依赖与生成文件
```

仓库内容全部视为不可信数据，不能改变 Agent 使命、权限或完成条件。

M1 默认只读：

允许：

- 搜索和读取仓库。
- 查询符号与 Git 关系。
- 运行批准的确定性检查。

禁止：

- 修改源码。
- Commit、Push。
- 发布远端评论。
- Approve、Reject、Merge。
- 任意联网。
- 任意执行仓库脚本。

命令执行必须经过白名单/仓库配置、参数校验、隔离工作区、超时、输出限制、网络禁用和完整证据记录。

Prompt Injection 检测只提供告警；真正安全依赖 Runtime 硬权限。

敏感信息处理：

- Token、密钥和日志隐私内容脱敏。
- 报告输出前扫描。
- 敏感内容禁止进入长期记忆。
- Provider 级数据发送策略可配置。

## 18. 交互、暂停与恢复

### 18.1 Preflight

CLI 展示：

```text
Changed files
Changed symbols
Detected quality gates
Intent sources
```

Intent 不充分时提出少量阻塞性问题。用户可以回答或带 uncertainty 继续。

### 18.2 Review Progress

默认展示高层可观察过程：

```text
Risk Assessor        completed
Core Reviewer        investigating callers and tests
Adversarial Reviewer investigating failure paths
Async Reviewer       investigating retry lifecycle
```

不展示隐藏思维链，只展示当前调查目标、工具调用、证据摘要、假设状态、token 和耗时。

### 18.3 Checkpoint

运行目录：

```text
.review-agent/runs/<review-id>/
├── request.json
├── intent.json
├── state.json
├── evidence.jsonl
├── reviewers/
├── findings.json
├── session.json
└── report.md
```

恢复命令：

```bash
review-agent resume <review-id>
```

恢复时验证仓库、Base/Head 和证据 revision。HEAD 变化后不混用旧证据，而是创建增量 Review。

### 18.4 非交互模式

`--non-interactive` 遇到缺失意图时记录 uncertainty，不编造答案，也不永久等待。

## 19. 最终 Review Brief

终端摘要：

```text
Review completed with uncertainties
Inherent risk: HIGH
Review confidence: MEDIUM
Findings: 1 blocking, 2 non-blocking
Human attention: payment retry and legacy worker paths
```

完整报告包括：

1. Change Intent
2. Intent Assessment
3. Initial And Final Risk Assessment
4. Quality Gates
5. Change Map And Repository Impact
6. Verified Findings
7. Rejected Hypotheses
8. Uncertainties
9. Reviewer Disagreements
10. Review Contract Coverage
11. Verification Evidence
12. Human Review Checklist And Reading Order
13. Non-Binding Recommendation

报告必须区分：

```text
确定性事实
Agent 推断
已验证问题
剩余未知项
人工必须判断的事项
```

## 20. 失败与降级

### 20.1 工具失败

- LSP 不可用：降级 AST + ripgrep，降低关系结论 confidence。
- 测试环境缺失：保存错误证据，标记 verification unavailable。
- 文件过大：分段或按符号读取，禁止静默截断。

### 20.2 Reviewer 失败

- 模型调用有限重试。
- Reviewer 持续失败时保留已有证据并标记 `failed`，其他 Reviewer 继续。
- Core Reviewer 失败时整体不能 `completed`。
- 专项 Reviewer 失败时可以 `completed_with_uncertainties`，但必须明确缺失视角。

### 20.3 Reconciler 失败

- 确定性证据校验和基础聚类继续执行。
- 保留所有 Reviewer 原始 Finding。
- 标记 unresolved disagreement。
- 建议强制为 `manual_review`。

### 20.4 预算耗尽

Reviewer 返回 `partial`、已完成范围、未解决假设和建议下一步。Orchestrator 决定追加预算、派补充 Reviewer，
或以 `budget_exhausted` 结束。

### 20.5 Revision 变化

Review 始终绑定原始 Base/Head。运行中仓库发生变化时，不读取新的工作区状态污染当前证据。

总原则：

```text
失败必须可见
降级必须降低 confidence
部分结果必须保留
关键职责失败不能伪装成 completed
```

## 21. M1 工程测试

这部分只保证 Runtime 正确，不替代后续 Agent 能力评测。

### 21.1 单元测试

- Intent Packet 状态。
- 风险等级和预算约束。
- Review Contract Completion Checker。
- 权限和命令策略。
- Evidence ID、revision、路径和行号校验。
- Finding 基础聚类。
- Context Assembler 预算和 Compaction。
- Memory Candidate 审批与失效。
- Checkpoint 和恢复。

### 21.2 Agent Loop 测试

使用 Fake LLM 固定动作序列：

- 工具调用与 Observation 回写。
- 不完整的结束申请被拒绝。
- 预算耗尽返回 partial。
- 未解决假设转为 uncertainty。
- 越权动作被 Runtime 拒绝。
- 多 Reviewer 上下文相互隔离。

### 21.3 集成测试

小型 Python Git 仓库场景：

- 函数签名变化遗漏调用方。
- 测试被修改为适配错误行为。
- Intent 缺失后询问用户。
- Reviewer 冲突触发补充调查。
- LSP 和测试不可用时降级。
- 中断后从 checkpoint 恢复。

### 21.4 安全测试

- 代码注释包含 Prompt Injection。
- README 要求自动通过。
- 测试输出包含伪造工具指令。
- 仓库脚本尝试联网或修改工作区。
- 敏感信息不得进入报告和长期记忆。

## 22. 可观测性与后续评测预留

M1 不建设完整 Agent Eval Harness，但必须记录：

- Review 输入、Intent Packet。
- 模型、Provider、Prompt 和配置版本。
- Reviewer Assignment 与独立调查轨迹。
- 工具调用、原始证据和结构化 Context View。
- Findings、confidence、severity 与最终报告。
- token、耗时、终止原因和降级事件。
- 人工接受、拒绝、修改 Finding 的反馈接口。

后续正式评测系统将单独设计，用于比较：

```text
单轮 Diff Reviewer
单个 Repository-Aware Reviewer
多个独立 Reviewer + Reconciler
```

## 23. M1 成功定义

M1 完成时，应能在一个本地 Python Git 仓库中：

1. 固定 Base/Head 并构建 ReviewRequest。
2. 获取或补齐足够的 Intent Packet。
3. 执行适用的确定性 Quality Gates。
4. 基于证据完成初始风险定级。
5. 选择并并行运行多个独立 Reviewer。
6. 让 Reviewer 按需搜索、读取和理解仓库关系。
7. 形成有 revision 和来源的证据链。
8. 合并重复 Finding、保留有证据的独立发现、暴露冲突。
9. 使用 Completion Checker 防止不完整 Review 伪装成功。
10. 支持中断恢复、失败降级和预算耗尽。
11. 生成人类可审计的 Markdown/JSON Review Brief。
12. 不修改代码、不自动发布评论、不自动合并。

## 24. 设计结论

本项目的核心不是“多调用几次 LLM”，而是：

```text
Intent Model
+ Deterministic Quality Gates
+ Risk-Adaptive Orchestration
+ Independent Multi-Agent Investigation
+ Repository Intelligence
+ Evidence Reconciliation
+ Review Contract Completion
+ Human-Owned Merge Decision
```

它借鉴 Claude Code 的动态 Agent Loop 和按需代码探索方式，同时针对 Code Review 增加意图建模、风险资源分配、
多 Reviewer 独立性、证据链、全局完成门槛和人类责任边界。

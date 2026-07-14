# Model-Assisted Risk Assessor 与 Portfolio Planner 完整设计

**状态：** 已实现并通过全量回归（2026-07-13）

**设计来源：** `2026-06-22-evidence-driven-multi-agent-code-review-design.md` 第 9、10、16、17、20 节。

## 1. 目标

本批把 Planning 从“纯本地风险判断 + 固定角色表”升级为两层架构：LLM 负责语义判断和动态审查视角，Runtime 负责最终风险、审查深度、Review Contract、预算、权限、恢复和审计。

```text
Risk Assessment Packet
  -> Local Risk Baseline
  -> LLM Risk Proposal
  -> Runtime Risk Compiler
  -> Authoritative Risk Assessment
  -> LLM Portfolio Proposal
  -> Runtime Portfolio Compiler
  -> Authoritative Assignment Portfolio
```

模型输出始终是不可信 proposal，不直接成为调度指令。

## 2. 权威边界

### 2.1 Risk Assessor

本地 `LocalRiskAssessor` 先从敏感路径、变更规模、Intent 未知项和 Quality Gate 结果计算风险下限。LLM 随后读取最小 Risk Packet，补充五个语义维度、理由、信号引用、未知项和建议关注点。

Runtime 合并时必须满足：

- 最终风险不得低于本地风险下限；
- 模型只能引用 Packet 中登记的 `signal_refs`；
- 本地理由、信号和未知项不得被模型删除；
- 模型输出非法、超时或不可用时，确定性降级为本地结果；
- 风险结果不是 Finding，不接受 `evidence_refs`、判决或合并建议。

### 2.2 Portfolio Planner

Planner 只提出 Reviewer candidate：角色类型、角色名、审查视角、使命、理由引用、上下文引用、附加 Contract 和检查建议。它不能输出最终预算、权限、风险等级、Provider、模型或可执行命令。

Runtime 编译时必须：

- LOW 至少 Core；MEDIUM 至少 Core + Adversarial；HIGH 为 3–4 个 Reviewer；CRITICAL 为 4–6 个 Reviewer；
- 缺失必选角色时确定性注入，重复视角确定性去重；
- Core 覆盖完整五项 Review Contract；其他角色只能增加、不能删除 Runtime 基线 Contract；
- 预算完全由风险档位生成，不采用模型自报预算；
- 仓库权限固定为 `read_only`，命令权限固定为 `safe_checks_only`；
- 所有 ref 必须来自 Portfolio Packet 的 allowlist；
- 模型失败时使用同一 Runtime compiler 的确定性 fallback candidates。

## 3. Stage-Specific Minimal Context

### 3.1 Risk Assessment Packet

只传：

- Base/Head、changed files、diff stat 和有界关键 Diff 片段；
- changed symbols 的结构化摘要；
- cheap Quality Gate 终态；
- Intent status 与 uncertainties；
- Runtime 生成的稳定 `signal_catalog`。

不传完整仓库、完整 Observation Store、Reviewer 历史或正式 Finding。

### 3.2 Portfolio Packet

只传：

- Runtime 编译后的风险等级、五维结果、理由和关注点；
- change map 与 changed symbols 摘要；
- Intent 摘要与未知项；
- 可选角色类型、Reviewer 数量边界、Contract allowlist 和 ref allowlist；
- Runtime 预算策略的摘要，不传可修改的执行权限。

不传大量代码、原始工具日志或其他 Agent 推理。

## 4. 严格模型协议

Risk proposal 必须是单个 JSON object，且只包含：

```yaml
level: low | medium | high | critical
dimensions:
  impact: string
  blast_radius: string
  reversibility: string
  uncertainty: string
  verification_strength: string
reasons: [string]
signal_refs: [authorized ref]
uncertainties: [string]
suggested_focus: [string]
```

Portfolio proposal 必须是单个 JSON object：

```yaml
candidates:
  - candidate_id: string
    role_kind: core | adversarial | specialist
    role_name: string
    perspective_key: string
    mission: string
    reason_refs: [authorized ref]
    context_refs: [authorized ref]
    extra_contract: [authorized contract item]
    required_checks: [string]
    priority: 0..100
summary: string
uncertainties: [string]
```

两种 parser 均拒绝多余字段、类型漂移、空文本、重复 ID、越界数量、未知 ref 和未知 Contract。解析失败可在同一逻辑 invocation 内有限重试；最终失败必须形成可见 fallback 决策。

## 5. 模型配置与统一适配器

Risk Assessor、Portfolio Planner 和 Reviewer 都只依赖项目统一的 `ModelAdapter`。Risk 与 Planner 各有独立的持久化配置：

```yaml
mode: local | model
provider: none | fake | openai-compatible
model: string | null
base_url: string | null
api_key_env: environment variable name
max_output_tokens: positive integer
max_provider_attempts: positive integer
max_elapsed_seconds: positive finite number
```

CLI 可选择继承 Reviewer 的 Provider 参数，也可分别覆盖；写入 Session 时必须解析为具体有效配置，不持久化 `inherit`。API key 值永不进入 Session。

新 Session schema 显式保存两个 stage config。旧 schema v1/v2 hydrate 为 `mode=local`，因此升级和 resume 不会意外新增模型调用。

## 6. 风险与 Portfolio 编译

### 6.1 Risk compiler

风险顺序固定为 `low < medium < high < critical`。最终等级取本地基线与合法模型提案中的较高者；五维描述优先采用合法模型语义分析，本地基线理由、信号、未知项和关注点与模型结果稳定去重合并。

决策 artifact 明确记录：local floor、model proposed level、final level、是否应用 floor、模型状态、失败原因和 fallback 状态。

### 6.2 Portfolio compiler

Runtime 先选取合法且不重复的模型 candidates，再按风险策略补足必选角色和最低数量，最后按稳定顺序生成 Assignment ID。HIGH/CRITICAL 可在最大槽位内采用高优先级专项视角；超出槽位的提案被拒绝并写入 policy actions。

所有 Assignment 使用强类型身份：

- `assignment_id`
- `role_kind`
- `perspective_key`
- `planner_source`

旧 Assignment 缺少这些字段时按 legacy defaults hydrate。Completion 以 `role_kind=core` 判断新产物，旧产物才回退到角色名匹配。

## 7. Reviewer 调度语义

风险档位决定的是完整 portfolio，而不是“只有 parallel 模式才执行完整 portfolio”。新 Session 中：

- `reviewer_mode=single` 表示单 worker 顺序执行完整 portfolio；
- `reviewer_mode=multi` 表示并行执行完整 portfolio。

旧 schema v2 的 `single` resume 保持历史的单 Core 行为，避免改变已存在 Session 的执行语义。无 Reviewer Provider 时仍保留规划结果，但 Completion 会如实显示缺失 Reviewer 视角。

## 8. Artifact 与恢复

Planning 保留现有 canonical artifacts：

- `risk_packet.json`
- `risk.json`
- `assignments.json`

并新增：

- `risk_model_envelope.json`
- `risk_model_raw_response.json`
- `risk_model_decision.json`
- `portfolio_packet.json`
- `portfolio_model_envelope.json`
- `portfolio_model_raw_response.json`
- `portfolio_model_decision.json`
- `portfolio_plan.json`
- `planning_summary.json`

禁用模型时不伪造 envelope/raw response，但仍写 decision 和 planning summary。每次模型调用使用由 review ID、stage 和输入 digest 派生的稳定 invocation ID；已提交的 Planning checkpoint 只 hydrate，不重复调用。Provider 不支持幂等键时，进程在“远端响应已返回、首个本地 artifact 尚未提交”的极窄窗口内仍可能发生 at-least-once 重放，必须在设计和 artifact 中明确，不能宣称外部 exactly-once。

## 9. Brief 与可观测性

JSON/Markdown Brief 披露：

- 风险来自 local 还是 model-assisted；
- local floor、模型提议等级和最终等级；
- Risk/Planner 的 accepted、disabled 或 fallback 状态；
- Runtime 注入、去重、截断和 Contract/权限/预算约束动作；
- 最终 Reviewer portfolio、角色来源和规划未知项。

raw provider response 只保存在 artifact，不进入 Reviewer Context 或 Brief。

## 10. 失败与安全

- 配置错误在模型调用前失败并返回 CLI 配置错误；
- Provider/解析错误是 stage fallback，不中断本地 Planning；
- artifact promotion、hash、Session registry 和 canonical hydration 错误仍是控制层失败；
- 仓库内容始终是不可信数据，不能改变 system prompt、schema、权限、Contract 或预算；
- Risk/Planner 无工具权限，只消费 Runtime 已构造的最小 packet。

## 11. 完成标准

- 模型可合法上调风险，不能降低本地下限或伪造 ref；
- 模型可提出动态专项视角，不能跳过 Core、删 Contract、改预算或扩权；
- 每个风险等级的 Reviewer 数量和预算由 Runtime 硬约束；
- model/local/fallback 三条路径均产生可审计结果；
- Risk/Planner 可独立配置 Provider、模型和调用预算；
- 新 Session 顺序/并行都执行完整 portfolio，旧 v2 resume 不行为漂移；
- canonical artifacts、Brief、resume、revision drift 和 legacy hydration 均通过测试；
- 定向测试与全量 pytest 无失败。

## 12. 非本批范围

- 语义 Evidence Reconciler。
- 有界 supplemental investigation。
- Durable Memory 自动写入。
- Eval Harness。
- GitHub/PR 集成。

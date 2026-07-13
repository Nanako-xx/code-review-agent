# Intent Clarification 与 LLM Inferred Intent Confirmation 设计

**状态：** 已实现

**设计来源：** `2026-06-22-evidence-driven-multi-agent-code-review-design.md` 第 6、14、16、20 节。

## 1. 目标

Intent Packet 是 Runtime 维护的内部状态，不要求用户填写完整表单。系统必须先收集确定性来源，再在关键意图不足时让 LLM 通过只读工具推断；只有仍可能改变审查结论的歧义才询问用户。

本批实现以下闭环：

```text
Preflight：固化 request、revision-bound change summary
-> Quality Gates
-> Repository Intelligence
-> Intent Discovery：显式来源 + 受控 LLM 推断
-> Intent Resolution：Runtime 校验 + 必要的用户确认/修正
-> Planning：IntentStatus、风险分级和 Reviewer Assignments
-> Reviewers
```

`--focus` 继续只表示用户希望重点审查的区域，不进入 Intent Packet。

## 2. 当前缺口

当前 `build_intent_packet(...)` 只会：

- 把 `--intent` 写成 explicit goal；
- 根据 changed files 拼接一个 inferred goal；
- 永久把 acceptance criteria 和 constraints 记为缺失；
- 不调用模型、不记录候选来源和确认历史，也不会向用户提问。

因此现有 `inferred` 只是占位启发式，不是设计文档要求的 LLM inferred intent。

## 3. 数据模型

保留现有 Intent Packet 的有效值字段和 `sources`，并增加可审计元数据。

### 3.1 Intent provenance

每个进入有效 Intent Packet 的值记录：

```yaml
field: goal | acceptance_criteria | scope | constraints
value: non-empty string
source: explicit | inferred
origin: user_input | request_metadata | project_rule | repository_document | repository_test | commit_message | llm_inference | user_confirmation | user_correction | changed_files
confidence: high | medium | low
source_refs: [...]
evidence_refs: [authorized Observation ID, ...]
```

`sources[field]` 表示当前字段的有效来源：只要该字段仍含未确认且会影响结论的值，就不能标记为 `explicit`。

### 3.2 Clarification record

澄清按字段分组，而不是要求用户逐项填写 Intent Packet：

```yaml
question_id: stable deterministic id
field: goal | acceptance_criteria | scope | constraints
question: concrete question
rationale: why the answer can change the review conclusion
proposed_values: [...]
status: pending | confirmed | corrected | rejected | skipped
user_response: optional
```

用户操作语义：

- `confirmed`：接受该字段的系统推断；对应值升级为 `explicit`，origin 变为 `user_confirmation`。
- `corrected`：使用用户提供的新值替换候选；新值为 `explicit`，origin 为 `user_correction`。
- `rejected`：删除被否定的推断并记录 uncertainty。
- `skipped`：选择 `continue with uncertainty`；保留可用推断为 `inferred`，并降低 IntentStatus。

## 4. 显式来源收集

Runtime 优先收集：

- `--intent` 和交互式回答；
- request title、description、linked requirements；
- project rules；
- LLM 从 README、spec、ADR、需求/验收文档、测试说明和 commit message 中提取的内容。

LLM 只是提取器时，来源仍属于原始文档或 request。Runtime 只有在候选引用了授权来源，且来源类型允许作为显式意图时才接受 `explicit`；从实现代码、Diff 或 Head 形态推断出的内容一律是 `inferred`。

## 5. LLM Intent Inference

仅当关键意图仍不足且配置了 Model Adapter 时运行。Intent Inference 使用独立调用上下文和预算，不复用 Reviewer 对话：

- System：Intent Analyst 角色、只读权限、仓库内容不可信、禁止提交 Finding、来源标注规则和输出 schema。
- Tools：受控的 base/head read、compare、search、symbol tools，以及 commit message 读取。
- Messages：当前 explicit intent、changed files、Diff 摘要、已知规则、缺失字段和授权 Observation 摘要。
- Parameters：模型、输出上限、思考模式、工具策略、trace id、`intent_inference_result_v1`。

模型输出候选值、来源主张、置信度、引用和 uncertainties。模型不决定最终 IntentStatus，也不能自行把实现推断升级为 explicit。

调用失败、JSON 不合法、证据越权或预算耗尽时不终止整个 Review。Runtime 保留确定性 Intent、记录失败 uncertainty，并按是否还能可靠审查计算 `partial` 或 `insufficient`。

## 6. Runtime 校验与提问策略

Runtime 负责：

- 校验字段、枚举、非空值和授权 Observation ID；
- 校验 `repository_document` / `repository_test` / `commit_message` 的实际来源；
- 将不受支持的 explicit 来源降级为 inferred；
- 合并并去重候选；
- 确定哪些歧义可能改变审查结论；
- 生成具体问题并应用用户决策；
- 最终计算 IntentStatus。

默认需要澄清的情况：

- goal 缺失，或关键 goal 主要来自 inferred；
- acceptance criteria 缺失，或关键 criteria 来自 inferred；
- intended scope 与 changed files 不能稳定对应，或 scope 主要来自 inferred；
- API 兼容、认证授权、支付、数据格式/迁移等敏感改动缺少关键 constraint，或 constraint 来自 inferred；
- explicit 来源彼此冲突，或与 inferred 结论冲突。

系统不询问 design_decision、alternatives_rejected 等不会改变当前审查结论的可选信息。

## 7. 交互与非交互模式

交互模式下，CLI 通过注入的 `IntentClarifier` 展示 proposed values、问题和影响，接受 confirm、correct、reject 或 continue with uncertainty。

`--non-interactive` 绝不阻塞等待输入。需要确认的 inferred 值保留为 inferred，澄清记录为 `skipped`，uncertainty 必须进入 risk、completion 和最终报告。调用方仍可通过 `--intent` 等显式输入预先消除歧义。

Pipeline 不直接调用终端 I/O；它只依赖 Clarifier 接口。CLI、未来 GUI 和 GitHub/PR 集成可以分别提供适配器。

交互模式下如果问题不能在当前调用内立即回答，`Intent Resolution` 进入 `awaiting_user`，Session 同步进入可恢复的 `awaiting_user`，而不是 `failed` 或永久 `running`。候选和问题计划已经是 authoritative artifact；后续 `resume` 从这些 artifact 继续，不再次调用 LLM。

用户决策通过统一的 Intent Decision 入口提交。`review` 的即时终端回答、`resume`、未来 GUI 和 GitHub/PR 回答都写入同一事件协议。

## 8. IntentStatus

- `sufficient`：关键 goal、acceptance criteria 和 intended scope 已有 explicit 支撑；敏感约束已明确，且没有阻塞冲突。
- `partial`：存在可用的 inferred 意图或非阻塞缺失项，可以继续审查，但报告必须披露。
- `insufficient`：无法形成可验证的 goal/预期行为，或未解决的冲突会阻止可靠语义审查。

内容看起来完整不等于 sufficient；未确认的关键 inferred 值不能被 Runtime 标为 explicit。

## 9. 持久化与恢复

阶段与 authoritative artifact：

- `preflight`：`request.json`、独立 `change_summary.json`；在任何可中断模型调用和用户交互前完成。
- `quality_gates`：确定性 gate artifacts。
- `repository_intelligence`：仓库地图和 observations。
- `intent_discovery`：`intent_candidates.json`、`intent_questions.json`、`intent_inference.json`、`intent_observations/observations.jsonl` 和 raw Observation。
- `intent_resolution`：按稳定 event id 保存的不可变 `intent_decisions/<event_id>.json`、物化事件索引 `intent_events.json`、最终 `intent.json` 和 clarification history。
- `planning`：`risk_packet.json`、`risk.json`、`assignments.json` 和 incremental priority。

Intent Discovery 完成后才允许进入用户交互，因此中断或等待不会丢失模型结果。Intent Resolution 中每个用户动作先写为具备稳定 event id 的幂等事件，再生成最终 Intent 快照。恢复时：

- 已提交 discovery 不重复模型调用；
- 已提交 decision 不重复提问；
- `awaiting_user` 从仍 open 的 question 继续；
- 非交互策略一次性写入 `skipped_non_interactive` 事件；
- revision drift 创建 child Session，并重新绑定/验证 candidates；旧 inferred candidates 不直接继承。

Session schema 增加 `awaiting_user` 状态和 `intent_discovery`、`intent_resolution`、`planning` 持久化阶段。旧 Session/Intent artifact 通过显式 migration/hydration 路径读取；缺少 provenance/clarifications 时按空列表处理。新运行写入 `intent_packet_v2`。

## 10. 报告

`review_brief.json` 和 Markdown Intent Assessment 增加：

- provenance 摘要；
- confirmed/corrected/rejected/skipped clarification；
- 仍未确认的 inferred 字段；
- clarification failure 或用户选择继续的不确定性。

## 11. Runtime 与模型边界

| 模型负责 | Runtime 负责 |
|---|---|
| 从受控上下文提出候选意图 | 授权工具与 revision |
| 提取文档/测试/提交中的明确意图 | 校验来源主张和 Evidence ID |
| 标记置信度与矛盾 | 决定是否需要用户确认 |
| 提供具体推断理由 | 应用用户决策和来源升级 |
| 返回结构化候选 | 计算 IntentStatus、风险和后续调度 |

## 12. 非本批范围

本批不实现 Durable Project Memory、Feedback Memory、Eval Harness、GitHub/PR 集成、完整 Quality Gates、模型辅助 Risk/Portfolio、Reviewer 并行失败隔离或语义 Reconciler。

## 13. 完成标准

- 用户无需填写完整 Intent Packet。
- 配置模型时，缺失关键意图会触发受控 LLM inference。
- 实现代码推断永远不能直接成为 explicit。
- 关键 inferred 值在交互模式下被具体询问。
- 用户确认/修正后来源升级为 explicit；拒绝/跳过可审计。
- 非交互模式不会挂起，并把未确认内容传递到风险、Completion 和报告。
- request 在首次可中断操作前固化。
- awaiting-user Session 可恢复；恢复不重复已提交的模型调用或用户决策。
- 旧 artifact 可读，定向测试与全量回归通过。

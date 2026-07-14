# Reviewer 并行、失败隔离与预算加固设计

**状态：** 已完成（2026-07-13）

**设计来源：** `2026-06-22-evidence-driven-multi-agent-code-review-design.md` 第 10、12、20 节。

## 1. 目标

本批补齐 Reviewer 执行层的最终 Runtime 语义：多个 Reviewer 真正独立并行；单个 Reviewer 的 Provider、解析或 Runtime 失败不会阻止其他 Reviewer；风险等级展开为可执行预算；预算耗尽、重试耗尽和失败结果都进入 authoritative artifact、Completion、Final Risk 与 Review Brief。

Revision Drift child Session、增量优先地图和恢复语义已经完成，本批只验证其与新执行层兼容，不重新设计。

## 2. 当前缺口

- Pipeline 按 Reviewer 顺序串行执行。
- `_execute_reviewer` 抛出异常时 reviewers phase 立即失败，后续 Reviewer 不再运行。
- Agent Loop 只硬限制 turn/tool；没有输出 token、累计 token、elapsed time 和 Provider retry policy。
- 预算耗尽结果没有完整保留已授权 Observation 和明确 termination metadata。
- `multi_reviewer_result.json` 只记录结果状态，不记录预算、调用次数、耗时和终止原因。

## 3. Runtime Budget

每个 Assignment 除现有 `max_turns`、`max_tool_calls` 外，增加：

```yaml
max_output_tokens: 每次模型调用最大输出
max_total_tokens: Reviewer 累计 token 上限；Provider 未返回 usage 时显式记录 unavailable
max_elapsed_seconds: Reviewer 墙钟时间上限
max_provider_attempts: 单次逻辑模型 turn 的最大 Provider 尝试次数
```

这些值由 `ReviewProfile.for_risk(...)` 在本地展开，Reviewer 只看到具体预算，不依赖抽象 `risk_level` 自行决定深度。旧 Assignment artifact 缺少新增字段时使用当前 Runtime 默认值读取；新 artifact 始终写完整预算。

## 4. Provider 重试

- Runtime 只对 Provider 异常和 `INVALID` 响应做有限重试。
- 重试使用同一个 Reviewer adapter 和同一个逻辑请求，保留工具对话状态。
- 结构化结果已经成功返回后，不因 Reviewer 自报失败而重试。
- Agent Loop 的 completion/schema 修正继续受 turn budget 约束，不与 Provider retry 混淆。
- 每次 Provider 尝试、错误和最终耗尽原因进入 trace/runtime metadata。

## 5. 并行与提交边界

Reviewer 分为两个阶段：

1. **并行调查：** 每个 Reviewer 在独立 AttemptWorkspace、ObservationStore、adapter 和预算内运行，只写自己的 attempt 目录。
2. **串行提交：** 主 Runtime 按 reviewer index 确定性提升完成 artifact、注册 Session descriptor、更新 task checkpoint。

因此并行模型调用不会并发写 `session.json` 或 authoritative artifact registry。输出顺序始终按 reviewer index 稳定，与完成先后无关。

## 6. 失败隔离

- Provider/create/parse/Reviewer Runtime 异常被转换为该 Reviewer 的结构化 `failed` ReviewerResult。
- 失败 Reviewer 已产生且通过 Tool Gateway 授权的 Observation 可以保留，但必须与失败状态和 uncertainty 一起披露。
- task checkpoint 的 `completed` 表示该执行任务已形成终态 artifact；审查结果是否成功由 `ReviewerResult.status` 表示。
- 文件提升、Session 写入、artifact hash 校验等控制层失败仍然失败整个 phase，不能伪装成 Reviewer 失败。
- Core Reviewer `failed/blocked` 使 Completion 为 `blocked`；专项 Reviewer 失败进入 missing perspective，并允许 `completed_with_uncertainties`。

## 7. 预算终止

- turn/tool/time/total-token 任一预算耗尽，ReviewerResult 为 `partial`，termination reason 精确记录。
- 已产生的合法 Observation ID 写入 `observation_refs`，调查摘要说明已完成范围和未完成原因。
- Provider usage 缺失不伪造 token 数；记录 `usage_available=false`，仍执行输出 token、turn、tool 和 time 硬限制。
- OpenAI-compatible adapter 使用 Runtime 传入的剩余时间作为单次 HTTP timeout 上限。

## 8. Artifact 与恢复

每个 Reviewer 的 raw-response artifact 增加 additive `runtime`：

```yaml
provider_attempts: int
model_turns: int
tool_calls: int
input_tokens: int
output_tokens: int
total_tokens: int
usage_available: bool
elapsed_seconds: number
termination_reason: completed | reviewer_partial | reviewer_blocked | provider_retry_exhausted | turn_budget_exhausted | tool_budget_exhausted | token_budget_exhausted | time_budget_exhausted | runtime_failure
```

Agent Loop trace 同步记录每次 Provider 尝试。旧 raw-response/trace artifact 继续可读，缺失 runtime metadata 时按 legacy unknown 处理。

中断恢复继续复用已提交的 completed task；running/failed/invalidated task 使用新 attempt 重新执行。Revision Drift child 不继承 parent Reviewer artifact 或预算消费记录。

## 9. 非本批范围

- 完整 Quality Gates。
- 模型辅助 Risk Assessor / Portfolio Planner。
- 语义 Reconciler 和有界补充调查。
- Durable Memory、Eval Harness、GitHub/PR 集成。

## 10. 完成标准

- 两个以上 Reviewer 的模型调用可证明发生重叠。
- 一个 Reviewer 重试耗尽或抛异常时，其他 Reviewer 仍完成并提交。
- Core 与专项 Reviewer 失败分别正确影响 Completion。
- turn/tool/token/time 终止均为可审计 partial，并保留已授权 Observation。
- 新预算随风险展开并进入 Assignment/Reviewer Context。
- resume 不重复已完成 Reviewer，Revision Drift child 使用独立预算与 artifact。
- 旧 artifact 可 hydrate，定向与全量测试通过。

## 11. 实现结果

- `ReviewProfile` 在本地将风险展开为完整 Reviewer budget，`Assignment` 和 Reviewer Context 只携带具体预算，不把抽象风险等级交给模型决定深度。
- single-shot 与 Agent Loop 共享 `reviewer_runtime`：Provider retry、usage 统计、token/time 检查、剩余输出上限和终止原因使用同一实现。
- 生产 Pipeline 使用独立 Reviewer attempt 并行调查；worker 不写 `session.json`，主线程按 reviewer index 串行提交 artifact 和 task checkpoint。
- Provider/create/parse/Reviewer Runtime 失败形成结构化 `failed` 结果；artifact promotion、Session、hash 等控制层失败继续使 phase 失败并由 resume 重跑未提交 task。
- raw response、Agent Loop trace、multi reviewer artifact 和 Review Brief 均披露 runtime/termination metadata；legacy Assignment、raw response 和 Brief 继续可恢复。

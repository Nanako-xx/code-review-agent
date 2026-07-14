# Model-Assisted Risk Assessor 与 Portfolio Planner 实施计划

**状态：** 已完成（2026-07-13）

**设计来源：** `docs/superpowers/specs/2026-07-13-model-assisted-risk-portfolio-design.md`

## Task 1：Session 与配置协议

- [x] 增加独立 Risk/Planner model stage config 与 Session schema v3。
- [x] v1/v2 严格 hydrate 为 local mode，保持旧 Session 行为。
- [x] CLI 支持继承或独立覆盖 Provider、模型和调用预算。

## Task 2：Model Risk Assessor

- [x] 扩展 Risk Packet 的 changed symbols 与稳定 signal catalog。
- [x] 实现最小上下文 envelope、严格 JSON parser 和有限重试。
- [x] 实现 local floor compiler、非法 ref 拒绝和确定性 fallback。
- [x] 增加 Risk envelope/raw/decision artifacts。

## Task 3：Model Portfolio Planner

- [x] 定义 Portfolio Packet、严格 candidate proposal 和 parser。
- [x] 实现角色/数量/Contract/权限/预算 Runtime compiler。
- [x] 增加强类型 Assignment 身份与 legacy hydration。
- [x] 增加 Portfolio envelope/raw/decision/plan artifacts。

## Task 4：Pipeline、调度与恢复

- [x] Planning 接入两阶段模型建议和 Runtime 编译。
- [x] 新 Session 的 single/multi 分别作为顺序/并行调度完整 portfolio。
- [x] 已完成 Planning hydrate 不重复调用；revision drift 使用新 Head 重算。
- [x] planning summary 记录稳定 invocation、fallback 和 policy actions。

## Task 5：Completion 与 Brief

- [x] Completion 使用强类型 Core 身份并保留 legacy fallback。
- [x] JSON/Markdown Brief 披露风险来源、floor、Planner 状态和 Runtime policy actions。
- [x] 模型失败和规划未知项降低可见 confidence，不伪装成功。

## Task 6：验证与同步

- [x] 单元覆盖 strict parser、risk floor、ref allowlist 和 portfolio invariants。
- [x] 集成覆盖 fake model、fallback、resume、顺序/并行和旧 Session。
- [x] 运行 CLI/Session/Brief 定向测试与全量 pytest。
- [x] 更新主 Spec 实现状态并本地提交功能分支。

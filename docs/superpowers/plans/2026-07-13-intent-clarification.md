# Intent Clarification 与 LLM Intent Inference 实施计划

**状态：** 已完成

**设计来源：** `docs/superpowers/specs/2026-07-13-intent-clarification-design.md`

## Task 1：Intent 状态与来源模型

- 增加 provenance、clarification action/status 和稳定 question id。
- 保留现有 Intent Packet 有效值与 `sources` 兼容接口。
- 定义 Runtime 充分性与来源升级规则。

## Task 2：LLM Intent Inference

- 增加独立 system prompt、context、输出 schema 和严格 parser。
- 使用统一 Model Adapter 和只读 ToolGateway。
- 持久化 inference result、trace 与授权 observations。
- 失败时降级为 uncertainty，不伪造 explicit intent。

## Task 3：Session 阶段与恢复协议

- Preflight 先独立提交 request 与 change summary。
- 增加 persisted Quality Gates、Intent Discovery、Intent Resolution、Planning 阶段。
- 增加 `awaiting_user` Session/phase 状态及幂等 resume 语义。
- 提交 candidate、question、event、final intent 为不同 authoritative artifacts。
- 增加旧 Session layout 的显式迁移/审计路径。

## Task 4：Runtime Intent Manager

- 合并 user/request/project/repository/LLM candidates。
- 校验来源和 Observation authority。
- 生成仅可能改变审查结论的问题。
- 应用 confirmed/corrected/rejected/skipped，并重新计算 IntentStatus。

## Task 5：CLI 与 Pipeline

- CLI 注入交互式 Clarifier，支持 confirm、correct、reject 和 continue with uncertainty。
- `--non-interactive` 自动记录 skipped，不读取 stdin。
- Intent Discovery 在 risk/assignment 前完成 inference。
- Intent Resolution 支持即时回答或进入 awaiting-user。
- Resume hydrate 已提交 candidates/events，不重复模型调用和已回答问题。

## Task 6：Artifacts、Brief 与兼容

- 写入 `intent_packet_v2`、intent inference 和 intent observations artifacts。
- 旧 Intent artifact additive hydrate。
- JSON/Markdown Brief 披露来源、确认历史和未确认 inferred 字段。

## Task 7：验证与提交

- 覆盖 explicit-only、LLM inferred、文档提取、非法来源、越权 Evidence。
- 覆盖 confirm/correct/reject/skip、非交互、模型失败和预算耗尽。
- 覆盖 preflight resume、旧 artifact hydration、risk/completion/report 传播。
- 运行定向与全量测试，更新主 spec 实现状态并提交。

# Runtime Review Contract Enforcement 实施计划

**状态：** 已完成

**设计来源：** `docs/superpowers/specs/2026-07-12-runtime-review-contract-enforcement-design.md`

## Task 1：Evidence authority

- Pipeline 统一允许 full range、base@SHA、head@SHA。
- 增加真实 agent-loop `read_range` 持久化回读测试。

## Task 2：Finding 与结果 schema

- 扩展 ReviewerFinding、CanonicalFinding、BriefFinding。
- 新模型输出严格校验必需字段、枚举、字符串和列表。
- Hydration 对历史 artifact 保持 additive 兼容。

## Task 3：Runtime completion validator

- 校验 Contract 唯一覆盖、Evidence authority、Finding 完整性。
- Agent Loop 拒绝过早 completion 并反馈 deficiencies。
- single-shot 无重试能力时降级为 partial。

## Task 4：统一后处理

- single/multi/no-provider 都生成 reconciliation 与 completion。
- Core Reviewer 未运行时 blocked。
- single Finding 进入 verified findings 和 Markdown/JSON 报告。

## Task 5：验证与同步

- 覆盖正常完成、缺失/重复/partial Contract、非法 Evidence、Finding 缺字段。
- 覆盖重试修正、预算耗尽、无 Provider、single Finding、resume hydration。
- 更新主设计实现状态，运行全量 pytest 并提交干净分支。

## 完成记录（2026-07-12）

- Runtime 已接管 Reviewer completion 的最终判定；模型自报 `completed` 不再直接生效。
- single、multi、no-provider 三条路径统一产出 reconciliation 与 completion。
- 新模型 Finding 使用严格 schema；历史 artifact 继续按 additive 规则读取。
- base/head/full-range Observation 均可按授权 revision 安全回读。
- 定向测试与全量测试通过；仅保留 2 个既有平台条件跳过。

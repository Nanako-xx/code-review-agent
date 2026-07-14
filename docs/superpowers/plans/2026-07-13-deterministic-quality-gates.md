# Deterministic Quality Gates 实施计划

**状态：** 已完成（2026-07-13）

**设计来源：** `docs/superpowers/specs/2026-07-13-deterministic-quality-gates-design.md`

## Task 1：Gate Model 与发现

- [x] 增加严格 Gate Plan、Definition、Result metadata 和 legacy hydration。
- [x] 从固定 Head revision 发现 compile、ruff、mypy/pyright、pytest。
- [x] 解析并验证 `pyproject.toml` 显式安全门禁。

## Task 2：隔离 Runner 与 Observation

- [x] 安全物化 Head snapshot，禁止 shell、路径逃逸和敏感环境继承。
- [x] 实现 timeout、输出上限、网络 guard、进程终止和日志脱敏。
- [x] 所有 gate 终态写入 raw Observation 与结构化摘要。

## Task 3：两阶段 Pipeline 集成

- [x] Quality checkpoint 执行 cheap gates 并向初始 Risk 提供信号。
- [x] Planning checkpoint 根据风险与 Reviewer portfolio 执行/跳过 expensive gates。
- [x] Deep results 和 Observation refs 传播到 Assignment/Reviewer Context。

## Task 4：Completion、Final Risk 与 Brief

- [x] Completion 核对 plan coverage 和 blocking policy。
- [x] Final Risk 处理全部失败、不可用、超时与验证缺口。
- [x] JSON/Markdown Brief 披露完整 gate metadata。

## Task 5：恢复、兼容与验证

- [x] Resume 复用已完成 gate artifacts，revision drift child 重新执行。
- [x] 旧 quality/Brief artifact additive hydrate。
- [x] 定向测试、CLI/Session 回归和全量 pytest。
- [x] 更新主 Spec 并提交功能分支。

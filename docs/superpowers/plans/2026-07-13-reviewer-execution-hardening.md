# Reviewer 并行、失败隔离与预算加固实施计划

**状态：** 已完成（2026-07-13）

**设计来源：** `docs/superpowers/specs/2026-07-13-reviewer-execution-hardening-design.md`

## Task 1：Budget 与 Runtime Metadata

- [x] 扩展 ReviewProfile、Assignment、Context 和 additive hydration。
- [x] 增加 output/total token、elapsed time、Provider attempts 与 termination reason。
- [x] OpenAI-compatible timeout 接受 Runtime 剩余预算。

## Task 2：Agent Loop 与 Single-shot 重试

- [x] Provider exception/INVALID 有限重试。
- [x] turn/tool/token/time exhaustion 统一返回 partial。
- [x] 保留已授权 Observation 和调用/usage 轨迹。

## Task 3：Pipeline 并行与失败隔离

- [x] Reviewer 在独立 attempt workspace 并行运行。
- [x] authoritative artifact 与 Session checkpoint 由主线程确定性提交。
- [x] Reviewer 运行失败结构化落盘，不中止其他 Reviewer。
- [x] 控制层提交失败继续阻断 phase。

## Task 4：Completion、Brief 与恢复

- [x] Core/专项失败按 Contract 传播。
- [x] Multi summary 与 Brief 披露 runtime/termination metadata。
- [x] Resume 复用 completed task，重跑未提交 task；Revision Drift child 保持隔离。

## Task 5：验证与提交

- [x] 并发重叠、Provider retry、异常隔离、稳定输出顺序。
- [x] turn/tool/token/time budget 与 usage unavailable。
- [x] Completion/Brief/hydration/resume/revision drift 回归。
- [x] 更新主 Spec 实现状态，运行全量测试并提交功能分支。

# Resumable Pipeline 与 Typed Hydration 实施计划

**状态：** 执行中  
**设计来源：** `docs/superpowers/specs/2026-07-10-review-session-memory-resume-design.md` 批次 B  
**目标：** 在 revision 未变化时，让未完成 Session 从最早不可信阶段真实续跑，并复用所有经过 hash、schema、revision binding 与 typed hydration 验证的上游结果。

## 架构原则

- `review` 与 `resume` 共用同一个 `ReviewPipeline`，CLI 不保留第二套业务流程。
- `session.json` 是唯一权威恢复状态；`state.json` 只由 Session/流水线结果派生为兼容摘要。
- completed 不是 manifest 中的一个布尔猜测。阶段必须同时通过 checkpoint、artifact registry、hash、schema、revision binding 和 typed loader 才可复用。
- 第一个无效、pending、running 或 failed 阶段及其全部下游阶段失效；有效上游只 hydrate，不重跑。
- attempt 输出先进入隔离目录，只有完整校验后才提升为权威 artifact。
- Reviewer 以单个 Assignment 为恢复原子；不恢复半个模型 turn。

## Task 1：Typed artifact hydration

**文件：**

- 新增 `src/review_agent/hydration.py`
- 新增 `tests/test_hydration.py`

**工作：**

- 为 request、intent、risk packet、risk assessment、assignments、quality gates、repository intelligence、reviewer execution、reconciliation、completion、final risk、review brief 提供严格 typed loader。
- 校验必需字段和字段类型，恢复 Enum/dataclass，拒绝未知值及会改变语义的缺失字段。
- 保持现有 artifact payload 兼容；schema 名由 artifact registry 验证。

**验收：** round-trip 成功；缺字段、错类型、错 Enum 和不支持 payload 明确失败。

## Task 2：ObservationStore 恢复与校验

**文件：**

- 修改 `src/review_agent/observations.py`
- 修改 `tests/test_observations.py`

**工作：**

- 实现 `ObservationStore.load(run_dir, expected_revisions)`。
- 校验 JSONL、重算 ID、raw artifact 普通文件与目录边界、内容 hash、revision 授权及重复 ID 一致性。
- 任一记录失败时拒绝整个恢复结果，防止部分 evidence 被误授权。

**验收：** 有效日志可恢复；篡改、缺失、越界 revision、路径穿越与冲突重复均失败。

## Task 3：Phase validation、invalidation 与 Reviewer 子 checkpoint

**文件：**

- 修改 `src/review_agent/session.py`
- 修改 `src/review_agent/session_store.py`
- 修改 `tests/test_session.py`
- 修改 `tests/test_session_store.py`

**工作：**

- 扩展 reviewer task checkpoint，保存 status、attempts、timestamps、artifact names 和 error。
- 提供 phase artifact validation 结果和从指定 phase 开始的下游 invalidation。
- 提供 phase/task running、completed、failed 转换，保证重试计数单调、幂等更新不重复计数。
- invalidation 删除失效 artifact 的 registry 引用，但保留 attempt 审计目录。

**验收：** 最早损坏阶段及下游失效；上游不变；单 reviewer 完成结果可独立复用。

## Task 4：Attempt workspace 与原子提升

**文件：**

- 新增 `src/review_agent/attempts.py`
- 新增 `tests/test_attempts.py`

**工作：**

- 实现 `attempts/<phase>/<attempt>/` 与 reviewer 子目录。
- 阶段只在 attempt 目录写半成品；成功后按固定 artifact 名原子提升到 run 根目录并注册。
- 清理遗留临时文件时只操作当前 run_dir 内已验证路径。

**验收：** 失败 attempt 不覆盖权威 artifact；重跑不向旧 observations JSONL 追加；提升后 hash 与 descriptor 一致。

## Task 5：可重入 ReviewPipeline

**文件：**

- 新增 `src/review_agent/pipeline.py`
- 修改 `src/review_agent/cli.py`
- 新增 `tests/test_pipeline.py`
- 修改 `tests/test_cli_smoke.py`

**工作：**

- 把 `_run_review()` 拆成 preflight、repository intelligence、reviewers、reconciliation、completion、final risk、reporting 七个阶段。
- `PipelineContext` 保存 typed stage outputs；每个阶段统一提供 required artifacts、load 和 run。
- 新 review 从 preflight 运行；resume 从首个无效/未完成阶段运行，有效上游通过 typed loader hydrate。
- provider 只在 reviewer 阶段真正需要执行时重建；复用下游时不触发模型或质量门。
- 每阶段提交 Session 后再更新兼容 `state.json`。

**验收：** 现有 single/multi、single-shot/agent-loop、fake/none 路径行为保持一致，CLI 不再承载阶段业务实现。

## Task 6：真实 resume

**文件：**

- 新增 `src/review_agent/resume.py`
- 修改 `src/review_agent/cli.py`
- 修改 `tests/test_cli_resume.py`
- 新增 `tests/integration/test_pipeline_resume.py`

**工作：**

- 验证 repository identity、已存 resolved commits 与当前 requested revision 解析结果。
- Batch B 对 revision 不变执行 `continue_session`；发现 drift 时安全阻断并留给 Batch C 创建 child Session。
- 校验所有 completed phase；损坏时 invalidation；确定 earliest phase；hydrate 上游并继续 Pipeline。
- completed 且全部 artifact 有效时 audit only，不执行模型、工具或质量门。

**验收：** preflight 后中断、reviewer 中断、reconciliation 前中断、artifact 篡改与 API key 补齐后恢复均有测试；第二次 resume 只审计。

## Task 7：集成审查与完成门

- 运行 Batch B 定向测试、架构边界测试与全量测试。
- 手工构造中断 Session，验证输出包含 starting phase 与 reused phases。
- 检查 secret、绝对未授权路径、旧 evidence 和失败 attempt 不进入权威产物。
- 更新本计划状态和设计实现记录，提交干净分支。

## 非本批范围

- Base/Head drift 的 child Session、lineage 去重与增量优先地图属于批次 C。
- Batch B 遇到 revision drift 必须阻断，不能在原 Session 上混用旧 artifact 或 evidence。

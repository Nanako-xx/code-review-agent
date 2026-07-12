# Revision Drift 与 Incremental Lineage 实施计划

**状态：** 已完成

**设计来源：** `docs/superpowers/specs/2026-07-10-review-session-memory-resume-design.md` 批次 C

**目标：** requested Base/Head 漂移时不修改 parent Session、不复用 parent evidence，幂等创建或恢复绑定新 SHA 的 child Session，并同时保留完整审查地图与 Head-only 增量优先地图。

## Task 1：Drift 分类与 child manifest

- 根据当前解析 SHA 与 parent SHA 分类 `HEAD_MOVED`、`BASE_MOVED`、`BASE_AND_HEAD_MOVED`。
- child 继承 root lineage、执行配置和原始请求，但拥有独立 run directory、artifact registry、ObservationStore 与 phase 状态。
- `original_base_sha` 始终保留 root Base；实际审查范围始终使用 child `resolved_base_sha..resolved_head_sha`。

## Task 2：幂等 child 创建与查询

- child ID 由 repository identity、parent ID、resolved Base/Head 确定性派生。
- 相同 drift 重复 resume 返回同一 child；已存在但 metadata 不匹配或损坏时阻断，不创建第二个 child。
- parent manifest 与 artifact 永不原地修改。

## Task 3：Full map 与 incremental priority map

- 所有 child 都按新的 resolved Base/Head 生成完整 ChangeSummary、Repository Intelligence、Observation 与最终报告。
- 仅 `HEAD_MOVED` 额外生成 `parent_head_sha..new_head_sha` 的 typed incremental priority artifact。
- incremental 文件和 diff 只调整调查优先级，不替代完整审查范围。

## Task 4：Resume/CLI 集成

- drift 时返回 `create_incremental_session`，显示 parent/new review ID、change kind、full range 与可选 incremental range。
- 新 child 从 Preflight 运行；已有 child 根据自身 checkpoint audit 或继续。
- child 不加载 parent ReviewerResult、Observation、Finding 或 quality artifact。

## Task 5：验证

- Head、Base、Base+Head drift。
- detached SHA 不产生 drift。
- 相同 drift 去重，已有 failed/running child 可继续。
- parent bytes 不变，child evidence ID 集合来自 child 自身。
- 全量测试与 CLI 端到端 smoke 通过。

## 完成记录（2026-07-12）

- Resume 会比较当前解析 SHA 与 parent 绑定 SHA，区分 Head、Base、Base+Head 漂移；detached SHA 保持不变时仍进入 audit。
- child review ID 由 canonical Git common directory、parent review ID 与新 Base/Head SHA 确定性派生；重复 resume 复用同一 child，损坏或 lineage 不匹配时安全阻断。
- child 继承 root lineage、原始 Base 和非敏感执行配置，但从空 artifact registry、空 evidence store 与全新阶段状态开始，parent manifest 和 evidence 不会被修改或复制。
- 所有 child 均重新运行完整 `Base..Head` 审查；仅 Head 漂移额外生成 typed `incremental_priority_map_v1`，Reviewer 同时收到增量优先 diff 与完整 diff。
- running/failed child 可以按自身 checkpoint 继续；CLI 输出 parent/new review、change kind、full range 与可选 incremental range。
- 定向测试与全量 pytest 均通过；仅保留既有的 2 个平台相关 skip。

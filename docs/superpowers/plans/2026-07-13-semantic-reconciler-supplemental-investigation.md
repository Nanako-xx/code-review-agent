# Semantic Evidence Reconciler 与有界补充调查实施计划

**状态：** 已完成（2026-07-14）

**设计来源：** `docs/superpowers/specs/2026-07-13-semantic-reconciler-supplemental-investigation-design.md`

**执行方式：** Subagent-Driven。按依赖分两轮并行，子 Agent 写入范围必须互斥；主线程负责跨模块集成、兼容审计、全量回归和最终提交。

## 总体不变量

- 模型输出只是 proposal，Runtime 保持证据、预算、权限、调度、终止和 Completion 权威；
- deterministic pre-pass 与 fallback 不得被绕过；
- 合法严重 Finding 不得被多数投票或无确定性依据的语义拒绝删除；
- 补充调查不得冒充初始 Core Reviewer 或补齐原始 Portfolio Contract coverage；
- 旧 v1/v2/v3 Session、旧 artifact schema 和历史 `single` 语义保持兼容；
- 失败 attempt Observation 不得进入最终授权集合；
- 所有模型调用、wave、task、预算和 fallback 必须可审计、可恢复；
- 本批不实现 Memory、Eval、GitHub/PR 或自动修复。

## Wave 1：并行基础能力

### Task 1：Session schema v4、阶段与 checkpoint

**所有权：**

- `src/review_agent/run_state.py`
- `src/review_agent/session.py`
- `src/review_agent/session_store.py`
- `src/review_agent/attempts.py`
- `src/review_agent/command.py`
- `tests/test_session.py`
- `tests/test_session_store.py`
- `tests/test_attempts.py`（如存在或新建）
- `tests/test_cli_smoke.py` 中只修改配置/session 相关测试

**交付：**

- [x] Session schema v4 与 `semantic_reconciler: ModelStageConfig`；
- [x] `RECONCILIATION_ANALYSIS`、`SUPPLEMENTAL_INVESTIGATION` phase；
- [x] immutable `SupplementalPolicy` 与风险上限配置承载；
- [x] `ReviewWaveCheckpoint`、`SupplementalTaskCheckpoint`、budget reservation/charge/unknown state；
- [x] 原子 wave/task/budget SessionStore API 与最小范围失效；
- [x] task-ID AttemptWorkspace namespace；
- [x] CLI `--semantic-reconciler-*` 继承规则；
- [x] v1/v2/v3 保持原 layout/resume，v4 strict round-trip。

### Task 2：Deterministic pre-pass 与 Semantic Reconciler

**所有权：**

- `src/review_agent/evidence.py`
- `src/review_agent/reconciler.py`（新建）
- `src/review_agent/model_adapter_factory.py`
- `tests/test_evidence.py`
- `tests/test_reconciler.py`（新建）
- `tests/test_model_adapter_factory.py` 中 Reconciler fake response 测试

**交付：**

- [x] stable Finding Candidate 与 Conflict Hint；
- [x] observation/revision/path/line authority pre-pass；
- [x] 最小 Reconciliation Packet 与 cluster-aligned context batch；
- [x] strict proposal parser：重复 key、exact fields、enum、candidate/ref allowlist；
- [x] finite retry、stable invocation/digest、raw/decision metadata；
- [x] Runtime compiler：candidate 完整处置、severity floor、保守严重 Finding policy、合法 rejection；
- [x] deterministic exact-dedupe fallback 与 unresolved disagreements；
- [x] fake model response 覆盖 accepted/no-supplemental 路径。

### Task 3：Supplemental Runtime、预算、工具与通用任务执行器

**所有权：**

- `src/review_agent/supplemental.py`（新建）
- `src/review_agent/reviewer_task_executor.py`（新建）
- `src/review_agent/models.py`
- `src/review_agent/context.py`
- `src/review_agent/tool_gateway.py`
- `tests/test_supplemental.py`（新建）
- `tests/test_context.py`
- `tests/test_tool_gateway.py`
- `tests/test_models.py`
- `tests/test_reviewer_task_executor.py`（如需要，新建）

**交付：**

- [x] `SupplementalInvestigationRequest/Plan/TaskSpec/BudgetLedger`；
- [x] request/wave/task/invocation stable IDs；
- [x] risk-to-policy 默认表、请求去重、角色/Contract/权限/预算编译；
- [x] durable reservation/charged/unknown consumption 规则；
- [x] single/limited-multi 调度所需稳定 task spec；
- [x] 抽取初始/补充 Reviewer 共用 task executor，不使用内存-only orchestrator；
- [x] envelope 与 Tool Gateway 双层 allowlist；非法调用计费且不产 Observation；
- [x] Supplemental targeted bootstrap，不做所有 changed files 的无界 diff 预取；
- [x] `planner_source=semantic_reconciler` 与补充任务身份验证。

## Wave 1 集成检查

- [x] 主线程审查三组改动的类型与 import 边界；
- [x] 解决 Session Policy 与 Supplemental Runtime 的单一来源；
- [x] 运行三组定向单元测试；
- [x] 确认没有修改历史临时目录或覆盖其他 Agent 改动；
- [x] 确认 architecture boundary 无循环依赖。

## Wave 2：持久化、下游与 Pipeline

### Task 4：Artifact schema 与 typed hydration

**所有权：**

- `src/review_agent/artifacts.py`
- `src/review_agent/hydration.py`
- `tests/test_artifacts.py`
- `tests/test_hydration.py`

**交付：**

- [x] analysis/model/plan/wave/task/final semantic artifact schemas；
- [x] 安全动态 artifact 名称正则；
- [x] `semantic_reconciliation_v1` typed hydration；
- [x] `reconciliation.json` 继续使用 `evidence_reconciliation_v1`；
- [x] v1/v2/v3 缺少新 sidecar 时严格 legacy defaults；
- [x] assignment/task/wave digest 与 revision binding 校验。

### Task 5：Completion、Final Risk、Brief 与报告

**所有权：**

- `src/review_agent/completion.py`
- `src/review_agent/final_risk.py`
- `src/review_agent/brief.py`
- `src/review_agent/reporting.py`
- `tests/test_completion.py`
- `tests/test_final_risk.py`
- `tests/test_brief.py`
- `tests/test_checkpoint_reporting.py`

**交付：**

- [x] accepted/local/fallback/partial semantic 状态传播；
- [x] `budget_exhausted` Completion 状态与 blocker 优先级；
- [x] remaining disagreements、supplemental failure/unavailable 强制 manual review；
- [x] Supplemental execution 不计入初始 Core/Contract coverage；
- [x] Final Risk 读取 semantic fallback、conflict 与 stop reason；
- [x] JSON/Markdown Brief 披露 resolved/rejected/unresolved、wave、budget、policy action；
- [x] 旧 Brief 输入没有新 sidecar 时保持旧输出。

### Task 6：Pipeline、wave coordinator 与 resume

**所有权：**

- `src/review_agent/pipeline.py`
- `src/review_agent/resume.py`
- `tests/test_pipeline.py`
- `tests/test_resume.py`
- `tests/test_cli_resume.py`
- 必要的 integration tests

**交付：**

- [x] 新三阶段 dispatch/load 与 schema-aware phase 顺序；
- [x] initial semantic analysis、首个 plan、补充 wave loop、final deterministic compile；
- [x] ReviewerTaskExecutor 接入初始和补充任务；
- [x] worker 有限并行、主线程稳定提交、全局 reservation-before-submit；
- [x] 补充 Observation 仅在成功 promotion/hash/revision 校验后授权；
- [x] completed batch/wave/task hydrate 复用，损坏时最小失效；
- [x] Provider fallback/unavailable、task failure、max wave、budget exhausted 终止；
- [x] revision drift child 不继承旧 wave/task/Observation/budget；
- [x] v2/v3 resume 不进入 v4 phase 或新增调用。

## Wave 2 集成检查

- [x] 主线程统一 PipelineContext、artifact 名称和 hydration API；
- [x] 核对 Session phase、`PHASE_MESSAGES`、dispatch/load 顺序完全一致；
- [x] 核对 Completion 只使用初始 executions 做 Core/Contract 判断；
- [x] 核对 Final Risk/Brief 消费完整 semantic artifact 而非丢失字段的兼容投影；
- [x] 核对失败 attempt store 不进入 `_authorized_observation_summaries()`。

## Task 7：端到端与恢复测试

- [x] 模型成功且不需补充：零 supplemental task；
- [x] 两个 Reviewer 冲突触发一个定向补充任务并解决；
- [x] 同位置不同问题不被误合并；
- [x] 模型试图删除唯一 blocker/high Finding 被 Runtime 保留；
- [x] 模型非法 ref/候选遗漏/重复处置触发 deterministic fallback；
- [x] Supplemental provider unavailable、partial、failed 形成 disagreement/manual review；
- [x] 全局 token/tool/time/max-wave exhaustion 形成 `budget_exhausted`；
- [x] single 顺序、multi 不超过 policy concurrency 且稳定提交；
- [x] 部分 task 完成后中断只重跑未完成 task；
- [x] invocation 返回后未提交记 unknown consumption；
- [x] artifact/Observation 篡改只失效最小 wave 及下游；
- [x] v1/v2/v3 fixture 可读/可恢复且不调用 Semantic Reconciler；
- [x] revision drift child 全部 ID/Observation 绑定新 revision；
- [x] JSON/Markdown Brief 与 Completion/Final Risk 完整传播。

## Task 8：验证、文档与本地提交

- [x] 运行 Reconciler/Supplemental/Session/Pipeline/Resume/Brief 定向测试；
- [x] 运行 architecture boundary tests；
- [x] 运行全量 pytest；
- [x] `git diff --check`；
- [x] 将设计状态改为“已实现并通过全量回归”；
- [x] 将本计划全部勾选并改为“已完成”；
- [x] 更新主 Spec 23.1 实现状态与剩余工作；
- [x] 精确暂存本批文件，排除 `.intent-*`、`.pytest-intent-*`、`.tmp`；
- [x] 在 `codex/semantic-reconciler` 创建本地提交，不自动 push 或 merge。

## 建议定向测试命令

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_evidence.py tests/test_reconciler.py tests/test_supplemental.py tests/test_session.py tests/test_session_store.py -q -p no:cacheprovider

& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/test_pipeline.py tests/test_resume.py tests/test_completion.py tests/test_final_risk.py tests/test_brief.py -q -p no:cacheprovider

& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest -q -p no:cacheprovider
```

# Focused Eval v2 Completion Design

- 日期：2026-07-22
- 状态：已确认并完成本地实现；PR #11 已合并
- 基线分支：`codex/eval-protocol-v2`
- 基线提交：`0000d76`
- 前置完成：Eval Protocol v2 Task 1–3

## 1. 文档关系与优先级

本文定义 Eval v2 剩余实现的最终范围。

它保留 `2026-07-16-core-code-review-eval-system-design.md` 中已经确认的核心评测语义，包括：

- 黑盒 `EvalInput -> Agent Adapter -> EvalSubmission -> Evaluators -> Scores/Report` 主干；
- Repository 与 Frozen Context 两种 Review Target；
- Intent Eval、Review Eval、Evidence、Metric Authority 和可解释报告；
- Agent 执行与 Evaluator 解耦；
- 重复 Trial、比较、校准和 Regression Gate 作为后续原 Task 15。

本文取代以下剩余实施范围：

- `2026-07-21-eval-protocol-v2.md` 的 Task 4–15 原拆分；
- `2026-07-16-core-code-review-eval-system-design.md` 中要求内置 pinned download、通用公共 Catalog 平台、HTTP/IPC Adapter、全量 v1 清理平台和大范围安全故障注入的实现要求。

若旧 Plan 与本文冲突，以本文为准。Task 1–3 已经完成的数据模型、artifact binding 和 Core 制品继续保留，不回退到 v1。

## 2. 问题与审计结论

V1 Task 1–12 已经提供完整的 Repository 评测主干：

- Repository prepare/cache/workspace/replay；
- per-attempt Trial、resume、stale worker 防护和 Always-Submission；
- current/subprocess Agent Adapter 与 capability preflight；
- Clarification、Intent Evaluator；
- Location Matcher、Evidence Integrity/Support；
- Finding semantic Judge、全局一对一 Assignment、fabricated/novel 判断；
- Metrics coverage、compatibility partition、Report、rejudge 和 CLI。

V1 Task 14 已经提供 AACR/SWE 的严格 Parser、source manifest、record receipt、create-only Suite publication 和 hash-bound Frozen Bundle，但保留四个明确缺口：

1. official Frozen Context 只能封存，不能运行；
2. AACR/SWE 缺权威 severity 时仍使用 `MEDIUM` 占位；
3. SWE 缺 side、部分缺 line 时仍可能进入 Line 指标；
4. SWE diff hunk 仅保存 provenance，未进入 truth-scoped Finding-equivalence context。

因此，剩余工作不是重写 Eval Harness，而是在现有主干上完成 Target Runtime、新 Target 的 Evaluator 适配、公共数据最终接入和两条 E2E。

## 3. 目标

完成一个本地可运行、可重放、不会制造假分数的 Eval v2：

```text
Repository Suite ---------------------> Repository Target Runtime --+
                                                                    |
Frozen public Suite -> verified bundle -> Frozen Target Runtime -----+-> Agent Adapter
                                                                      -> EvalSubmission
                                                                      -> Intent/Review Evaluator
                                                                      -> authority-aware Scores/Report
```

完成后必须满足：

- Core、AACR native、SWE native 使用 Repository Target；
- SWE official frozen 使用 Frozen Context Target；
- current Agent 只运行 Repository；
- 声明 frozen capability 的 subprocess Agent 可以运行 Frozen；
- Agent、Evidence Checker 和 Judge 使用同一份 immutable Target replay；
- 缺 severity/location authority 时对应指标为 `not_scorable`，不是零分或占位分；
- Repository/Frozen、不同 authority profile 不做质量 roll-up；
- 公共数据经显式本地导入和 hash 校验后，正式 Trial 全程离线。

## 4. 明确保留的现有实现

以下模块只做必要接口适配，不重写算法：

- `assignment.py`：最大权重一对一 Assignment；
- `clarification.py`：Clarification Script 与 matcher；
- `intent_evaluator.py`：Intent claim matching 与 clarification judgement；
- `review_evaluator.py`：Finding candidate、known-invalid、Assignment、novel/fabricated 主流程；
- `judge.py`：现有 structured blind Judge profiles；
- `metrics.py`：现有 Trial/Case/Aggregate 和 coverage 框架；
- `report.py`：现有 compatibility partition 与 inspect/report 框架；
- `repository.py`：现有 Repository prepare/cache/workspace/replay；
- `artifacts.py`：现有 create-only receipt、attempt lease、resume/rejudge integrity。

Task 1–3 已完成的以下协议继续作为活动协议：

- `ReviewTargetV2`；
- tagged `EvidenceSourceV2`；
- `MetricAuthority`；
- `ReviewEvaluatorContext`；
- `WireContractV2`；
- `AdapterCapabilitiesV2`；
- `TrialMaterializationManifest` 和 `target_materialization_id`；
- 已重新生成的 18 个 Core Case、Golden 和两个 Suite。

## 5. 工作包 A：Target Runtime

### 5.1 Materialization 边界

建立一个行为层 `materialization.py`，只负责：

- 根据 `review_target.kind` 分派；
- 产生和验证 `TrialMaterializationManifest`、`TargetAccess` 和 replay；
- 保证 Agent-visible Target、Evaluator replay 与 Trial/attempt binding 一致；
- 在调用 Agent 前检测 Target 漂移。

现有 Materialization DTO 只能有一份 canonical 定义。若需要从 `artifacts.py` 移至 `materialization.py`，作为本工作包的一次机械移动完成，不改变已通过的 schema、字段、digest 或 ID 语义，也不保留兼容别名和重复 parser。

### 5.2 Repository Target

Repository 路径必须复用现有 `RepositoryPreparer`：

- `prepare` 可以按现有受控流程准备 Repository cache；
- `run-agent` 和 `evaluate` 只允许 `require_cached/open_replay`；
- 不重写 Git acquisition、cache、workspace 或 replay；
- Repository materializer 只把现有 `PreparedRepository`、workspace manifest 和 replay 投影为通用 Materialization。

### 5.3 Frozen Context Target

Frozen 路径复用 V1 Task 14 已验证的 SWE Frozen Bundle：

- 打开时要求外部 expected bundle/trust digest；
- 验证 exact rendered UTF-8 bytes、record digest、bundle binding 和 source binding；
- 每个 Trial 在独立只读 Target 区暴露 exact rendered file；
- Agent 只得到相对 `TargetAccess`，看不到 bundle manifest、truth、annotation、receipt 或宿主路径；
- replay 直接读取 verified content object，不读取 Agent workspace 副本。

### 5.4 Adapter 与 Runner

- `current-agent` capabilities 固定为 Repository-only；
- `subprocess-json-v2` 显式声明支持的 Target kinds 和 Evidence kinds；
- Runner 只根据 Target tag 分派，不从 `protocol_id` 猜测；
- capability 不匹配属于 preflight incompatibility，不计 Agent failure；
- Materialization 缺失或漂移产生 Harness-owned materialization failure，不调用 Agent；
- Adapter 输出必须回指同一 task/trial/input/materialization identity。

不新增 HTTP/IPC Adapter，也不建设通用 sandbox 平台。`isolation_profile` 作为可复现和报告字段保留，不宣称平台无法强制的隔离能力。

## 6. 工作包 B：Evaluator 对新 Target 与 Authority 的适配

### 6.1 Evidence replay

`EvidenceIntegrityChecker` 接受统一 replay union：

- Repository file/diff 沿用现有 exact Git replay；
- Frozen Evidence 使用 `target_materialization_id + context_ref + line range`；
- command output 和 existing-CI 沿用现有 attestation/source binding；
- Evidence 的 materialization ID 必须匹配 Submission、Trial plan 和 Prepare receipt；
- Judge 只接收 Checker 成功重放的 canonical bytes。

### 6.2 Metric Authority

公共数据映射固定为：

| Cohort | Severity | Location |
|---|---|---|
| Core | expert authority，可评分 | expert authority，可评分 |
| AACR | 不可评分，`severity=null` | upstream authority，可评分 |
| SWE native/frozen | 不可评分，`severity=null` | 不可评分 |

规则：

- 不生成 neutral `MEDIUM` 占位；
- location 不可评分时仍可作为语义/诊断上下文，但不创建可评分 LocationAudit；
- severity-weighted recall、critical/high miss、line precision/recall 只读取 eligible truth；
- 没有任何 eligible truth 时保存 `not_scorable` 和 authority coverage，不能表示成真实 0；
- Metric Authority 不参与 Finding equivalence edge weight，不改变一对一 Assignment。

### 6.3 SWE truth-scoped context

- SWE diff hunk 以 evaluator-only source 保存；
- 每个 source 绑定唯一 truth ID 和 provenance digest；
- 只允许进入对应 Finding-equivalence request；
- 不进入 Agent Input、Submission Evidence、Location、Severity 或其他 candidate；
- repository 内容和 diff hunk 始终作为不可信数据，不得覆盖 Judge policy。

### 6.4 Scores 与 Report

复用现有 Score/Report 结构，只补充：

- authority profile/policy digest；
- eligible/excluded truth coverage；
- Target/wire/isolation compatibility；
- Repository/Frozen 和不同 authority profile 的独立 partition。

不重写现有 Intent、Finding matching、Judge、Assignment 或 Markdown renderer。

## 7. 工作包 C：公共数据最终接入

### 7.1 Adapter 输出迁移

- AACR Adapter 输出 Repository Target，并将 severity 标记为不可评分；
- SWE native 输出 Repository Target；
- SWE frozen 输出 Frozen Context Target 和可运行 Suite；
- 移除 `requires_eval_v2`、placeholder severity/category/location 等临时 limitation；
- 保留 raw/scorable/isolated statistics、source record receipt 和 protocol-specific Suite identity；
- native/frozen 使用不同 protocol 与 wire contract。

### 7.2 `prepare-public local-import`

新增唯一公共数据入口：

```text
review-agent-eval prepare-public --mode local-import ...
```

它负责：

- 接收用户在 Harness 外取得的本地 dataset 目录；
- 要求调用方提供 expected source manifest/profile digest；
- 复用现有 `VerifiedPublicSource`、PublicPreparationReceipt 和 Frozen Bundle verifier；
- 发布 create-only canonical Suite/bundle 到 D 盘用户指定目录或 `.eval-data`；
- 输出 source/version/license/hash、filter、protocol 和 preparation receipt；
- 不修改 Case 中 canonical remote Repository URL/base/head。

本阶段不实现内置 `pinned-download`、archive extraction、DNS/redirect 防御、通用 Catalog 服务或任意网络 client。下载发生在 Harness 外；Eval Harness 的职责从不可信本地传输源开始验证。

Repository cache acquisition 继续使用现有受控 `RepositoryAcquisitionBinding` 和 `prepare` 流程；`run-agent/evaluate` 保持 cache-only。

## 8. 工作包 D：聚焦 E2E 与切换验收

必须提供两条完整 E2E：

1. Repository：Core fixture `prepare -> run-agent -> evaluate -> re-evaluate -> inspect`；
2. Frozen：SWE fixture `prepare-public -> prepare -> run-agent -> evaluate -> inspect`。

故障验收只覆盖会改变评测真实性的边界：

- Target 在 manifest 后被替换；
- wrong materialization/input/trial identity；
- Adapter capability 与 Suite target 不兼容；
- Frozen Evidence 引用错误 context/line/hash；
- authority 缺失却尝试产生 severity/line 分数；
- Repository/Frozen 或不同 authority partition 被错误聚合；
- resume/rejudge 使用不同 Target replay。

不新增大范围 DOS 设备名、8.3 alias、下载器、archive、DNS 或通用多租户攻击矩阵。现有路径和 artifact 安全回归继续运行，不再扩展为独立平台项目。

## 9. 后续工作

完成 A–D 后，回到原 Eval 计划：

- Task 15：Repeated Trials、paired compare、Judge calibration、Regression Gate；
- Task 16：最终文档、全量回归和本地发布验收。

以下外部门禁不由代码伪造：

- 独立真人 Reviewer B 的 Core blind review；
- 10 个 Regression Case 每个真实模型至少三次 baseline；
- Private Held-out 数据。

## 10. 测试与执行约束

- 信任边界、Target 替换、错误计分和真实 bug 使用 RED/GREEN；
- 机械 schema 迁移、DTO 移动、生成制品和文档使用集中 GREEN；
- 每个工作包只修改与其验收直接相关的文件；
- 不把既有 V1 主干行为重新实现为新类；
- 不为纯目录整洁单独创建重构 Task；
- 不使用 C 盘临时目录；
- pytest basetemp 位于当前 worktree 外的短 D 盘路径；
- 不清理或提交用户 scratch；
- 不伪造真人、真实模型或外部数据门禁。

## 11. 成功定义

- Repository 与 Frozen 两种 Target 均可完成本地全链评测；
- current Agent 对 Frozen 稳定 preflight incompatible，frozen-capable subprocess 可运行；
- Agent、Evidence 与 Judge 的 Target 来源一致且可重放；
- AACR/SWE 不再用 placeholder severity/location 制造分数；
- SWE diff hunk 只在对应 truth 的语义匹配中可见；
- authority-sensitive 指标有明确 eligible/excluded/not-scorable coverage；
- public local-import 可复现，正式 Trial 无 dataset acquisition；
- 原有 Intent、Finding matching、Evidence、Metrics、Report、resume/rejudge 主干继续通过；
- 完成后可以直接进入原 Task 15，而不再启动另一轮协议重写。

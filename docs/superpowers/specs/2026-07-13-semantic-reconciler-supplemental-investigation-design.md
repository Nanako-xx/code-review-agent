# Semantic Evidence Reconciler 与有界补充调查完整设计

**状态：** 已实现并通过全量回归（2026-07-14）

**设计来源：** `2026-06-22-evidence-driven-multi-agent-code-review-design.md` 第 3、10、12、13、14、16、17、20、21、22、23 节。

## 1. 目标

本批补齐 Reviewer 调查结束到 Completion 之间缺失的语义层：

1. Runtime 先完成确定性证据校验、Finding 身份化和候选聚类；
2. Semantic Reconciler 模型只对已登记候选提出语义合并、拒绝、冲突解释和补充调查 proposal；
3. Runtime 编译 proposal，保留证据权威、严重性下限、预算、权限和终止权；
4. 对无法靠现有证据解决的冲突，执行可恢复、可计费、最多有限波次的定向补充调查；
5. 形成完整语义结果、兼容旧消费者的 reconciliation 投影、Completion/Final Risk/Brief 可见的降级与预算状态。

```text
Reviewer Results + Authorized Observations
  -> Deterministic Reconciliation Pre-pass
  -> Minimal Reconciliation Packet
  -> Semantic Reconciler Proposal
  -> Runtime Proposal Compiler
  -> Supplemental Plan (optional)
  -> Bounded Supplemental Waves
  -> Final Runtime Reconciliation
  -> Completion -> Final Risk -> Review Brief
```

本批直接实现最终架构，不把“一次不可恢复的模型调用”或“无全局预算的临时循环”作为交付目标。

## 2. 当前缺口

现有 `reconcile_evidence()` 已经能：

- 拒绝无 evidence refs 或引用未知 Observation 的 Finding；
- 按规范化 claim 与完全相同的 evidence refs 做精确去重；
- 汇总 Reviewer Contract assessment；
- 为 Completion 提供兼容 reconciliation 结构。

但它还不能：

- 给每个原始 Finding 建立稳定身份；
- 检测同位置不同主张、同主张不同证据、severity/location/impact 冲突；
- 判断语义重复或证据与 claim 的语义支持关系；
- 记录 resolved conflicts；
- 生成并执行补充调查；
- 对 Semantic Reconciler 失败、补充失败和全局预算耗尽提供明确终态；
- 在中断后复用已完成的语义调用和补充任务。

现有确定性逻辑继续作为不可绕过的 pre-pass 与 fallback，但不能在聚合前用“首项元数据获胜”丢失候选差异。

## 3. 权威边界

| 能力 | 模型 | Runtime |
|---|---|---|
| 判断语义重复、同位置不同问题 | 提出 proposal | 校验成员完整性并形成权威 group |
| 判断 evidence 是否语义支持 claim | 提出有理由的判断 | 校验证据存在、revision/path/line/hash 与 allowlist |
| 拒绝 Finding | 只能提出允许原因与依据 | 执行保守拒绝规则，保留审计记录 |
| 冲突处理 | 提出 resolved / needs investigation / unresolved | 决定是否接受、派任务或保留 disagreement |
| 补充调查 | 提出 question、required evidence、preferred perspective | 决定任务数量、角色、Contract、工具、预算、并发和波次 |
| 严重性和 confidence | 可建议 canonical wording/confidence | severity 不得低于成员最高值；confidence 不因投票自动提高 |
| Review 完成与建议 | 无权决定 | Completion Checker 和人类最终决定 |

硬约束：

- 模型输出始终是 untrusted proposal，不是调度命令或证据；
- 不使用多数投票；
- 单个 Reviewer 的可靠严重 Finding 不能因“只有一个人发现”被删除；
- 多个 Reviewer 重复无证据主张不能提升真实性；
- `assigned_contract` 是最低覆盖要求，不是 `outside_review_scope` 的边界；
- Reconciler 不直接使用工具，不能自行读取仓库或扩大上下文；
- 补充 Reviewer 仍经过标准 Tool Gateway、Observation Store 和 Review Contract validator；
- 模型不能创建 Finding、Observation、预算、角色、工具或 Completion 状态。

## 4. Pipeline 与 Session schema v4

新 Session 使用 schema v4，并增加两个显式阶段：

```text
REVIEWERS
  -> RECONCILIATION_ANALYSIS
  -> SUPPLEMENTAL_INVESTIGATION
  -> RECONCILIATION
  -> COMPLETION
  -> FINAL_RISK
  -> REPORTING
```

### 4.1 `RECONCILIATION_ANALYSIS`

- 生成确定性候选、拒绝项、Contract coverage 和 conflict hints；
- 构建最小 Reconciliation Packet；
- 执行零个或多个 cluster-aligned Semantic Reconciler batch；
- Runtime 编译 proposal；
- 生成首个 Supplemental Plan 或明确 `not_needed / fallback / unresolved`。

### 4.2 `SUPPLEMENTAL_INVESTIGATION`

- 按持久化 policy 执行有限 wave；
- 每个 wave 的任务可串行或有限并行，但由主线程稳定顺序提交；
- wave 完成后用已提交的新 Observation/Finding 再做语义判断；
- 达到无任务、已解决、模型 fallback、任务失败、预算耗尽或最大波次时终止；
- 阶段即使没有任务，也写入显式 `not_needed` 终态 artifact。

### 4.3 `RECONCILIATION`

- 只消费前两个阶段已经提交并通过 hash/revision 校验的 artifact；
- 生成 v4 完整权威 `semantic_reconciliation.json`；
- 同时生成保持旧结构的 `reconciliation.json` 投影视图；
- 不再调用模型或工具，因此可以确定性重建。

### 4.4 旧 Session

- v1 继续只读审计；
- v2/v3 按原 phase 布局和原 reconciliation 语义恢复；
- 不在 resume 时把 v2/v3 静默迁移成 v4，也不追加新的模型调用；
- 新 review 才创建 v4；revision drift child 使用当时当前 schema，并绑定新 revision 重新运行；
- v2 `single` 截断 portfolio 与 v3 `single` 顺序执行完整 portfolio 的历史语义保持不变。

## 5. 类型模型

### 5.1 Finding Candidate

Runtime 在任何语义合并前建立不可变候选：

```yaml
finding_id: F-<stable-digest>
origin: initial | supplemental
reviewer_task_id: string
reviewer_index: int | null       # legacy/report compatibility
role: string
role_kind: string
claim: string
severity: blocker | high | medium | low
confidence: high | medium | low
path: string
line: int
impact: string
suggested_action: string
verification_performed: [...]
evidence_refs: [ObservationID]
validation_status: supported | rejected
deterministic_rejection_reason: null | unsupported_claim | stale_evidence
```

`finding_id` 由 review ID、Base/Head、任务身份、规范化 Finding 字段和排序后的 evidence refs 派生，不含时间、随机数、列表顺序或完成顺序。

### 5.2 Conflict Hint

确定性层只生成 hint，不猜测语义真伪：

```yaml
conflict_id: C-<stable-digest>
candidate_ids: [...]
kind: exact_duplicate | same_location | shared_evidence | severity_mismatch | location_mismatch
summary: string
```

### 5.3 Reconciliation Packet

```yaml
schema_version: reconciliation_packet_v1
review_id: string
revision_binding: {base_sha, head_sha}
candidate_catalog: {F-...: FindingCandidate}
conflict_hints: [...]
observation_catalog:
  O-...:
    source: string
    revision: string
    path: string | null
    line_start: int | null
    line_end: int | null
    context_view: string
contract_coverage: [...]
intent_summary: {...}
code_snippets: {...}
allowed_rejection_reasons: [...]
policy_summary: {...}
```

Packet 不包含完整 Session、完整 Observation Store、原始工具日志或 Reviewer 隐藏推理。

### 5.4 Semantic Proposal

模型必须返回严格 JSON：

```yaml
canonical_groups:
  - member_ids: [F-...]
    representative_id: F-...
    canonical_claim: string
    rationale: string
    supporting_refs: [O-...]
    proposed_confidence: high | medium | low
rejections:
  - candidate_id: F-...
    reason: unsupported_claim | contradicted_by_test | outside_review_scope
    rationale: string
    decision_refs: [O-...]
disagreements:
  - disagreement_id: D-...
    candidate_ids: [F-...]
    status: resolved | needs_investigation | unresolved
    issue: string
    resolution: string
    decision_refs: [O-...]
supplemental_requests:
  - disagreement_id: D-...
    question: string
    required_evidence: [string]
    preferred_perspective: string
    related_candidate_ids: [F-...]
    reason_refs: [O-...]
uncertainties: [string]
summary: string
```

严格 parser 拒绝重复 JSON key、未知/缺失字段、非标准常量、非法枚举、空文本、重复 ID 和超出 Packet allowlist 的引用。

## 6. Runtime Proposal Compiler

Compiler 执行以下不变量：

1. 每个 supported candidate 必须且只能进入一个 canonical group 或一个 rejection；不能消失或重复处置；
2. semantic proposal 不能新增 Finding；canonical group 只能引用 Packet candidate；
3. representative 必须属于 group；group 的 evidence refs 必须是成员 evidence 的子集或已登记 decision refs；
4. canonical severity 取成员最高 severity，模型不能降低；
5. canonical confidence 不得高于成员最高 confidence，重复数量不能自动提高 confidence；
6. exact duplicate 可由 Runtime 直接合并；其他语义合并必须保留 model decision provenance；
7. `stale_evidence` 只能由确定性层判定；模型不能提出；
8. `contradicted_by_test` 必须引用已授权 test/quality Observation；
9. `outside_review_scope` 不能仅以“超出 assigned Contract”为理由；
10. blocker/high supported candidate 若只有语义拒绝、没有确定性 contradiction，Runtime 将其保留为 canonical Finding，并记录 unresolved disagreement；
11. 所有 rejection 都保留原 candidate、原因、rationale、decision refs 和 decision source；
12. 补充请求只能关联 `needs_investigation` disagreement，不能提出任意新审查范围；
13. 同一语义请求按规范化 question、required evidence、perspective 和 candidate IDs 去重；
14. 超限请求被裁剪时必须形成 policy action 和 remaining disagreement，不能静默丢弃。

模型 proposal 无效时，在同一逻辑 invocation 内有限重试；仍无效则使用确定性 fallback：

- 所有 supported 原始 Finding 都被保留；
- 只做确定性 exact dedupe；
- conflict hints 进入 `remaining_disagreements`；
- 不派发模型生成的补充任务；
- semantic status 为 `fallback`，Completion 强制 `manual_review`。

## 7. 有界补充调查

### 7.1 Request 与权威任务

模型输出 `SupplementalInvestigationRequest`，Runtime 编译成：

```yaml
SupplementalTaskSpec:
  request_id: SREQ-...
  wave_id: W-...
  task_id: STASK-...
  source_candidate_ids: [...]
  source_disagreement_id: D-...
  assignment: Assignment
  allowed_tools: [...]
  bootstrap_policy: targeted_only
  budget_reservation: {...}
```

补充 Assignment：

- 使用标准 Assignment/Reviewer Result 协议，但由 SupplementalTaskSpec 包装；
- `planner_source=semantic_reconciler`；
- 不加入原 Portfolio，不冒充原 Core Reviewer 或补齐原始 Contract coverage；
- Mission 只回答一个 material question；
- required checks 来自 Runtime 清洗后的 required evidence；
- initial context 只含关联 Finding、Observation、代码范围和已有 Quality Gate 摘要；
- 不传第一轮 Reviewer 自由文本推理；
- repository permission 固定 `read_only`，command permission 固定 `safe_checks_only`；
- 不允许新增或重跑 Quality Gate，只能引用现有已提交结果。

### 7.2 默认 Runtime Policy

风险只在本地展开为有效 policy；Reconciler/Reviewer 模型不靠抽象 risk label 自行决定深度。

| Risk | 最大补充 wave | 总任务 | 每 wave 任务 | 并发 | 每任务 turn/tool | 每任务 total token | 全局 token | 全局 wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| low | 1 | 1 | 1 | 1 | 4 / 8 | 16,384 | 16,384 | 120s |
| medium | 1 | 2 | 2 | 2 | 6 / 12 | 32,768 | 65,536 | 240s |
| high | 2 | 3 | 2 | 2 | 8 / 16 | 49,152 | 147,456 | 480s |
| critical | 2 | 4 | 2 | 2 | 10 / 24 | 65,536 | 262,144 | 600s |

每任务 output token、Provider attempt 和单任务 wall-clock 仍受 Assignment 与 Provider config 的更小值约束。项目 policy 可以进一步压低这些上限，但不能由模型提高。

### 7.3 Durable Budget Ledger

全局 ledger 持久化：

```yaml
reserved: {tasks, tool_calls, tokens, elapsed_seconds}
charged: {tasks, tool_calls, tokens, elapsed_seconds}
unknown_consumed: {tokens, elapsed_seconds, invocation_ids}
remaining: {...}
stop_reason: null | resolved | no_requests | model_fallback | task_failure | budget_exhausted | max_waves
```

规则：

- 主线程先原子 reservation，再提交并发 worker，防止 oversubscription；
- Provider 有 usage 时按实际值计费；没有 usage 时按保守 reservation 计费；
- 调用返回后、artifact 提交前崩溃时记为 `unknown_consumed`，不能在 resume 后免费退款；
- retry 与新 logical turn 分开计数；
- 达到任一全局上限立即停止创建新任务；
- 尚有 material disagreement 时，Completion 使用 `budget_exhausted`，建议 `manual_review`。

### 7.4 工具边界

工具采用双层 allowlist：

1. Context/envelope 只向模型公开 SupplementalTaskSpec 允许的工具；
2. Tool Gateway 再校验 allowed tools、参数、revision、路径、超时和输出上限。

非法工具调用返回结构化错误、计入 tool budget，但不创建 Observation。补充任务不执行当前第一轮 Reviewer 的“为所有 changed files 自动 compare”预取；只有目标范围内的读取由模型在预算内显式请求。

## 8. Context 与模型调用

Semantic Reconciler 使用独立 `ModelStageConfig`：

```yaml
semantic_reconciler:
  mode: local | model
  provider: none | fake | openai-compatible
  model: string | null
  base_url: string | null
  api_key_env: string
  max_output_tokens: int
  max_provider_attempts: int
  max_elapsed_seconds: float
```

CLI 使用与 Risk/Portfolio 相同的继承规则：`--semantic-reconciler-*`。补充 Reviewer 继续使用 Reviewer adapter，因为它执行的是标准 Reviewer Assignment。

Reconciler envelope：

- `system`：角色、只读语义使命、authority boundary、安全边界、拒绝原因、完整覆盖规则和输出 schema；
- `tools: []`，`tool_choice: none`；
- `messages`：最小 Packet batch；
- `parameters`：模型、输出上限、reasoning mode、temperature 0、schema、trace/invocation metadata；
- 仓库文本和 Observation content 都标记为 untrusted data；
- 不持久化隐藏思维链，只保存结构化 rationale、raw response、usage、耗时和终止原因。

Context 使用现有 `ContextBudget` 思路，按 deterministic cluster 分 batch，不在单个字符串尾部静默截掉 Finding。若模型调用批次数达到 policy 上限：未处理候选仍逐项保留，相关 cluster 进入 disagreement，并强制 `manual_review`。

## 9. 稳定身份与恢复

```text
candidate_id = SHA256(review + revisions + source task + canonical finding)
request_id = SHA256(question + required evidence + perspective + candidate IDs)
wave_id = SHA256(review + revisions + wave index + trigger digest + policy version)
task_id = SHA256(wave_id + request_id + compiled assignment digest)
invocation_id = SHA256(task_id-or-batch-id + logical turn + request digest)
```

ID 不含时间、随机数、数组下标或并发完成顺序。同一 logical turn 的 Provider retry 使用同一 invocation ID，attempt index 单独记录。稳定 ID 不等于外部 exactly-once；Provider 不支持幂等键时仍明确记录极窄的 at-least-once 窗口。

Session v4 新增：

- `SupplementalPolicy` immutable config；
- `ReviewWaveCheckpoint`，并持久化本次风险等级编译后的 effective policy；
- `SupplementalTaskCheckpoint`；
- task assignment digest、reservation、charged/unknown usage；
- `initialize_wave`、`reserve_task_budget`、`mark_task_*`、`mark_wave_completed`、`invalidate_wave_from` 原子操作。

恢复原则：

- 已完成且 hash/schema/revision/assignment digest 有效的 task 不重复调用模型；
- running task 按整个 task attempt 重试，旧 reservation 进入 unknown consumption；
- 单 task 损坏只失效该 task、所属 wave 聚合和后续 wave；
- 早期有效 wave 与 Observation 保留；
- 未提交 attempt Observation 永不进入最终授权集合；
- `AttemptWorkspace` 使用安全 task ID namespace，不与初始 `reviewer_<index>` artifact 冲突。

## 10. Artifact Contract

### 10.1 Reconciliation Analysis

- `reconciliation_prepass.json`
- `reconciliation_packet.json` 或按 batch 的 packet artifacts
- `reconciler_<batch>_envelope.json`
- `reconciler_<batch>_raw_response.json`
- `reconciler_<batch>_decision.json`
- `supplemental_initial_plan.json`
- `reconciliation_analysis_summary.json`

### 10.2 Supplemental Wave

- `supplemental_wave_<wave>_plan.json`
- `supplemental_wave_<wave>_budget.json`
- 每 task 的 spec/assignment/envelope/raw/result/trace/observations
- `supplemental_wave_<wave>_reconciler_decision.json`
- `supplemental_wave_<wave>_summary.json`

### 10.3 Final

- `semantic_reconciliation.json`：v4 完整权威结果；
- `reconciliation.json`：保持 `evidence_reconciliation_v1` 字段的兼容投影；
- `supplemental_summary.json`：波次、任务、预算、失败和 stop reason 汇总。

动态 artifact 名称必须通过严格正则和稳定 schema 映射；旧 artifact 不改名、不改 schema。Session descriptor、task checkpoint 与 artifact hash 三者必须一致。

## 11. 最终语义结果

```yaml
schema_version: semantic_reconciliation_v1
status: accepted | local_only | fallback | partial
canonical_findings: [...]
rejected_findings: [...]
conflicts_resolved: [...]
remaining_disagreements: [...]
contract_coverage: [...]
evidence_quality: verified | mixed | degraded
supplemental:
  waves: int
  tasks: int
  completed: int
  partial: int
  failed: int
  budget: {...}
  stop_reason: string
policy_actions: [...]
uncertainties: [...]
model:
  status: accepted | disabled | fallback
  invocation_ids: [...]
  input_digests: [...]
```

`reconciliation.json` 将上述结果投影为旧字段：`canonical_findings`、`rejected_findings`、字符串形式的 `remaining_disagreements`、`contract_coverage` 和 `evidence_quality`。详细 resolved conflicts、模型状态与补充信息只从新权威 artifact/summary 读取，避免破坏旧 hydration。

## 12. Completion、Final Risk 与 Brief

Completion 继续只把初始 Portfolio executions 用于 Core/专项 Reviewer presence 和原 assigned Contract coverage。Supplemental executions 只能解决特定 disagreement，不能冒充初始 Core coverage。

精确映射：

| 条件 | Completion status | recommendation |
|---|---|---|
| 无 blocker、semantic accepted、无剩余冲突 | 现有 completed 规则 | approve 或 needs_work |
| semantic disabled/local 且 deterministic hints 无 material conflict | 现有 completed 规则 | 现有规则 |
| semantic fallback/provider/parser failure | completed_with_uncertainties | manual_review |
| remaining disagreement 或补充 task partial/failed/unavailable | completed_with_uncertainties | manual_review |
| 补充全局预算耗尽且仍有 material disagreement | budget_exhausted | manual_review |
| 原 Core/Intent/blocking gate/Contract blocker | blocked | manual_review |
| artifact/hash/Session 控制层失败 | 不产出伪 Completion，Session failed | manual intervention |

若同时存在 blocker 与 budget exhaustion，`blocked` 优先，budget exhaustion 保留在 uncertainty/termination metadata。

Final Risk 必须读取 canonical Findings、remaining disagreements、semantic fallback 和 supplemental stop reason。JSON/Markdown Brief 新增：

- Semantic Reconciliation status；
- resolved conflicts 与 remaining disagreements；
- rejected Finding 的来源、原因、rationale 和 refs；
- supplemental wave/task/预算/失败摘要；
- model fallback 与 Runtime policy actions；
- `budget_exhausted`、`unknown_consumed` 和 manual-review 原因。

`Session.status=completed` 仍只表示生命周期结束，审查结论以 `completion.json` 为准。

## 13. 失败语义

- Reconciler adapter/config/Provider/parser/compiler 失败：有限重试后 deterministic fallback；保留原始 supported Finding；强制 manual review；
- 模型提出非法 candidate/ref/rejection/任务：整份 batch proposal 拒绝，不部分接受不完整 candidate accounting；
- Supplemental Reviewer 失败：其他任务继续；失败任务关联 disagreement 保留；
- Observation hash/revision 不合法：该任务结果不能进入授权集合；
- 单 wave plan/summary 损坏：失效该 wave 聚合与后续 wave，不重跑已验证的早期任务；
- 全局预算耗尽：停止新任务，保留已提交结果，Completion 标记 `budget_exhausted`；
- 无 Reviewer provider：计划写为 `unavailable`，不伪造 task 执行，冲突保留；
- Runtime artifact promotion、hash 或 Session 原子提交失败：阻断阶段并按 checkpoint 恢复，不降级成普通语义 uncertainty；
- revision drift：当前 review 仍只读原 Base/Head；child review 使用新 revision 重算所有 candidate/wave/task ID。

## 14. 实现模块

建议新增：

- `reconciler.py`：Packet、proposal、strict parser、model runner、Runtime compiler；
- `supplemental.py`：policy、request/plan、stable IDs、budget ledger、wave coordinator；
- `reviewer_task_executor.py`：从 Pipeline 抽取可复用的初始/补充 Reviewer task executor；

建议修改：

- `run_state.py`、`session.py`、`session_store.py`、`attempts.py`：schema v4、阶段、wave/task checkpoint；
- `models.py`、`evidence.py`、`hydration.py`：candidate 与新 typed artifact；
- `context.py`、`tool_gateway.py`：最小 Reconciler context 与双层 tool allowlist；
- `model_adapter_factory.py`、`command.py`：Semantic Reconciler stage config 与 fake response；
- `pipeline.py`：新阶段、task executor、resume、Observation authority；
- `artifacts.py`：静态和动态 schema；
- `completion.py`、`final_risk.py`、`brief.py`、`reporting.py`：终态传播。

## 15. 测试矩阵

必须覆盖：

1. deterministic pre-pass：未知/过期 evidence、精确重复、同位置不同问题、severity/location 冲突；
2. strict proposal：重复 JSON key、未知 candidate/ref、候选消失/重复、模型新增 Finding、非法 rejection；
3. authority：严重 Finding 不能被无确定性依据删除、模型预算/工具/角色不能越权、不得多数投票；
4. supplemental compiler：请求去重、角色映射、Contract 隔离、0/1/最大 wave 和 task off-by-one；
5. tool boundary：envelope/Gateway 双重 allowlist、非法调用计费且不产 Observation、无界 diff bootstrap 被禁用；
6. global budget：并发 reservation、usage 可用/缺失、unknown consumption、token/tool/time exhaustion；
7. scheduling：single 顺序、multi 有限并行、主线程稳定提交、一个 task 失败不阻断其他 task；
8. resume：plan 前/后中断、部分 task 完成、artifact 已提升未 checkpoint、wave summary 损坏、幂等重复 resume；
9. compatibility：固定 v1/v2/v3 fixture、历史 phase/`single`/artifact/hydration 语义不变，v4 round-trip；
10. integration：Reviewer 冲突触发补充、补充证据解决冲突、无冲突零任务、fallback、provider unavailable、budget exhausted；
11. downstream：Completion、Final Risk、JSON/Markdown Brief 完整披露；Supplemental 不补造 Core coverage；
12. revision drift：child 不继承旧 task/Observation/budget，ID 绑定新 revision；
13. 全量 pytest 与 architecture boundary tests。

## 16. 本批不实现

- Durable Project Memory 与 Review Feedback Memory；
- Eval Harness；
- GitHub/PR 集成、评论发布和自动合并；
- 自动修复代码；
- Reconciler 直接工具循环；
- 未批准的网络、写工作区或仓库脚本执行；
- 用补充 Reviewer 替代失败的初始 Core Reviewer；
- 隐藏思维链持久化；
- 把旧 completed Session 迁移后重新调用模型。

## 17. 完成标准

- Semantic Reconciler 的所有输出都经过 strict parser 与 Runtime compiler；
- deterministic pre-pass/fallback 在模型禁用或失败时保留所有合法 Finding；
- 补充调查有持久化全局预算、最大波次、最大任务和双层工具边界；
- 已完成 task/wave 可恢复复用，失败 attempt 证据不泄漏；
- v1/v2/v3 Session 与 `evidence_reconciliation_v1` 保持兼容；
- Completion 精确区分 blocked、completed_with_uncertainties 和 budget_exhausted；
- Final Risk、JSON/Markdown Brief、artifact 与 tracing 可审计；
- 定向与全量测试通过；
- 主 Spec 实现状态同步；
- 本批只提交本地功能分支，不自动 push 或 merge。

# Code Review Agent Task 15：重复试验、配对比较、Judge 校准与回归门禁设计

日期：2026-07-26  
状态：设计已确认并完成本地实现；PR #12 已合并，正式发布仍等待真实模型、独立真人 Reviewer B、真人 Judge calibration 与 Private Held-out 外部证据

基线：Eval v2 `d33775f`（PR #11 合并后的 `master`）

## 1. 设计结论

Task 15 建设在现有 Eval v2 生命周期之上，不改变 Code Review 的核心评测含义：

```text
提前标注的 Case / Suite
        ↓
Agent 产生 Intent、Finding、Evidence 和失败状态
        ↓
Evaluator 将输出与标注真值进行匹配
        ↓
计算 Intent、Review、Finding、Evidence、Failure 和 Usage 指标
        ↓
Task 15 对已完成的评测结果做重复试验统计、版本比较、Judge 校准和发布门禁
```

Task 15 不新增 Case Pass、`pass@1`、`pass^k` 或 Overall Score。重复运行的结果直接体现在已有各项指标的 Trial 分布、Coverage、Case 级变化和置信区间中。

推荐采用独立、不可变的评测分析制品链。`prepare`、`run-agent` 和 `evaluate` 继续拥有原来的职责；`compare`、`calibrate` 和 `gate` 只读取经过 source-bound 重放验证的 Evaluation，并产生新的 Analysis Artifact。

## 2. 背景与问题边界

现有 Eval v2 已经提供：

- Repository/Frozen Target 的 canonical materialization；
- 每个 Case 多 Trial 的 immutable Run/Trial plan；
- 独立 Submission、Evaluator、Judge、Evidence 和 Score artifact；
- failure、unknown、ungraded、not_scorable 和 Metric Authority coverage；
- `RunEvaluationBundle` 的 source-bound hydration 与报告重放；
- `MetricsAggregator` 的 numerator/denominator 聚合；
- Agent 与 Judge 的隔离。

因此 Task 15 解决的不是“如何定义一次评测”，而是以下四个后续问题：

1. 同一批标注数据运行多次后，指标是否稳定；
2. 新 Agent 与 baseline 在相同 Case 上改善或退化了什么；
3. 语义 Judge 的判断是否与独立真人一致；
4. candidate 是否满足预先冻结的发布条件。

## 3. 评测数据范围与权限

所有数据都可以用于计算指标、观察趋势和发现失败模式，但是否具有阻止发布的资格由 Suite/Case 的 release eligibility 决定。

### 3.1 Release-blocking 数据

- 经过人工审查和权威标注的 Core Regression Cases；
- 未来的 Private Held-out Suite。

它们可以进入 `release_blocking` Gate Policy。

### 3.2 Diagnostic-only 数据

- AACR-Bench；
- SWE-PRBench；
- 尚未经过人工审查晋升的 AI Synthetic 数据。

它们仍然可以运行、评分、比较和展示，但不能单独阻止正式发布。若 Synthetic Case 后续完成独立人工审查、真值补充和晋升登记，才可以成为 Core Regression Case。

Severity/location 缺少权威标注的数据，其对应指标为 `not_scorable`，不是 0，也不是 pass。其他有足够输入的指标仍然可以计算。

## 4. 整体架构

### 4.1 生命周期

```text
prepare → run-agent → evaluate
                         ↓
              Verified RunEvaluationBundle
                         ↓
          Statistics / Compare / Calibrate / Gate
                         ↓
              Immutable Analysis Artifacts
```

Task 15 的分析服务：

- 不调用 Agent；
- 不调用 Judge；
- 不执行数据集或 Repository acquisition；
- 不访问 Agent workspace；
- 不修改 `.eval-runs`；
- 不从 Markdown 反向恢复 canonical 数据；
- 不将不同 compatibility partition 合并成 Overall Score；
- 不自动发布评论、Approve 或 Merge PR。

输入必须经过 `RunEvaluationBundle` 的 source-bound hydration。孤立的 `summary.json`、Markdown 报告或用户手工拼接的分数不能作为比较或门禁的可信根。

### 4.2 Analysis Store

新增独立的 `EvaluationAnalysisStore`，默认根目录为 `.eval-analyses/`，支持 CLI 覆盖到 D 盘或其他受控目录：

```text
.eval-analyses/
├── comparisons/<comparison-id>/
│   ├── comparison_plan.json
│   ├── comparison_result.json
│   ├── report.md
│   └── receipt.json
├── calibration-packages/<package-id>/
│   ├── package_manifest.json
│   └── receipt.json
├── calibration-results/<calibration-id>/
│   ├── human_labels.json
│   ├── calibration_result.json
│   ├── report.md
│   └── receipt.json
├── gate-policies/<policy-id>/
│   ├── policy.json
│   └── receipt.json
└── gate-results/<gate-result-id>/
    ├── gate_result.json
    ├── report.md
    └── receipt.json
```

Analysis Artifact 使用 canonical JSON、内容 digest、create-only 发布和 receipt 绑定。不能覆盖已有制品。更换 Policy、Statistics 算法、Judge Calibration 或输入 Evaluation 时，必须生成新的 ID。

每个跨 Run 制品至少绑定：

- baseline/candidate Run ID 和 Evaluation ID（适用时）；
- Run Config、Case Snapshot、Summary 和 Trial Score digest；
- Suite、Case、Target 和 Wire Contract 版本；
- Statistics/Comparison/Calibration/Gate 算法版本；
- 输入覆盖范围和 `not_scorable` 原因；
- 任何外部 Label、Policy 或 Calibration Result 的 digest。

### 4.3 模块边界

| 模块 | 职责 |
| --- | --- |
| `statistics.py` | 多 Trial 的指标分布、Coverage 和确定性置信区间 |
| `comparison.py` | 严格兼容性验证、baseline/candidate 配对、Case 级差异 |
| `calibration.py` | 盲审包、人工标签导入、Judge 一致性和校准状态 |
| `gates.py` | Policy 预注册、逐指标条件判断和发布决策 |
| `analysis_artifacts.py` | Analysis schema、内容寻址、receipt、重放和安全路径 |
| CLI 层 | 参数解析、服务编排、JSON/文本输出和退出状态 |

分析模块不把 Runtime、Session、Memory、Risk 或产品 Reviewer 内部机制当作评测输入，也不为这些内部机制单独打分。

## 5. Repeated Trials 设计

### 5.1 运行规则

现有 Runner 已能为每个 Case 创建多个独立 Trial。Task 15 不重写 Runner，只消费每个 Trial 的 Submission、Score、Failure 和 Judge Coverage。

正式 Core Regression Gate 要求 baseline 和 candidate 对每个 Case 至少运行 3 个 Trial，且两边 Trial 数量相同。AACR、SWE 和 Synthetic 的探索性运行可以使用 1 个或其他数量的 Trial，但 Trial 数量不足正式要求时不能作为 release-blocking Gate 的依据。

每个 Trial 都保留：

- Submission 和 Score；
- Agent failure、timeout、invalid output 等终态；
- Judge failed、unknown、ungraded 和 Coverage；
- token、时间、工具调用和成本（如果 Provider 提供）；
- 原始 Trial 的 source bindings。

不挑选最好一次，也不把失败 Trial 从分母删除。

### 5.2 权威指标与稳定性信息

每个指标同时提供两种信息：

1. **权威汇总值**：沿用现有 `MetricsAggregator` 的 numerator/denominator 语义。不能先计算每个 Trial 的百分比，再简单平均百分比。
2. **重复运行分布**：按 `trial_index` 分别汇总每一轮的指标，报告每轮值、最小值、最大值和标准差。

示例：

```text
Issue Recall
Trial 1: 82%
Trial 2: 76%
Trial 3: 80%
All Trials: 79.3%
```

`failure_as_miss`、`failure_excluded`、`unknown`、`ungraded`、`not_scorable` 和 `zero_denominator` 继续使用既有分类，不允许在 Statistics 层互相转换。

### 5.3 Statistics Artifact

`RunStatisticsV1` 绑定一个已经完成的 Run Evaluation，并保存：

- source Run/Evaluation/Summary/Trial Score refs；
- Trial count、terminal count 和各类 Coverage；
- 每个 Core Metric 的 numerator、denominator、value、status；
- 每个 Trial Index 的同样投影；
- dispersion 投影；
- 统计算法和版本；
- 指标方向（higher-is-better 或 lower-is-better）。

Statistics 不新增产品指标；它只是把现有 Trial 分数以可比较、可重放的方式组织起来。

## 6. Paired Comparison 设计

### 6.1 一对一配对

baseline 与 candidate 依据以下键严格配对：

```text
(task_id, case_version, canonical_case_digest, trial_index)
```

两个 Run 的 Trial ID 可以不同，因为 Trial ID 包含各自的 Run ID。不能按输出顺序、Finding 数量或报告行号配对。

### 6.2 可比性

以下条件必须相同：

- Suite、Case 集合、Case 版本和 canonical Case digest；
- Trial 数量；
- Target kind、Wire Contract、Materialization Protocol 和隔离配置；
- Evaluator、Judge、Rubric、Metrics Policy；
- Truth Completeness、Metric Authority 和 Novel Finding Policy；
- 评测所需的 Context/Replay 身份。

允许不同的部分是被测 Agent 一侧，例如模型、Provider、Prompt、Agent 版本和内部策略。这些差异完整写入 `agent_delta`，不隐藏为“纯模型差异”。

如果两边的 Submission 已存在但 Evaluator/Judge/Rubric 不同，不能直接比较。可以先使用现有 re-evaluate 对两边 Submission 使用同一个 Evaluator Execution Config，再对新 Evaluation 比较；不需要重新运行 Agent。

### 6.3 比较输出

`RunComparisonV1` 对每个指标独立保存：

- baseline 值和 coverage；
- candidate 值和 coverage；
- absolute delta；
- 指标方向；
- 每个 Case、每个 Trial 的变化；
- `improved`、`regressed`、`unchanged`；
- Judge coverage delta；
- 置信区间；
- `not_scorable`、`insufficient_coverage` 和其他排除原因。

不生成单一 candidate 总分，也不把一个 Case 强行压缩成单一胜负。一个 Case 可以同时出现 Review Recall 改善和 Fabricated Rate 退化。

### 6.4 Deterministic Paired Bootstrap

比较的置信区间使用版本化、stdlib-only 的 paired bootstrap：

- 以 Case 为重采样单位，避免同一 Case 的多 Trial 被误当作完全独立的样本；
- 每次重采样重新聚合 numerator/denominator；
- 固定 seed、迭代次数、百分位数算法和实现版本；
- 默认报告 95% 区间；
- 对零分母、`not_scorable` 和 `ungraded` 保持显式 Coverage；
- 不因为缺少 numpy 改变核心结果。

不兼容时仍可发布 `RunComparisonV1`，但状态为 `not_comparable`，并列出所有不兼容字段；不能强行计算差值。

## 7. Judge Calibration 设计

### 7.1 目的

数据集真值可以直接支持确定性匹配，但下面的语义判断需要 Judge：

- Intent 改写是否表达同一意图；
- Finding 是否指向同一个实质性缺陷；
- 新 Finding 是 confirmed、plausible 还是 fabricated；
- Evidence 是否支持对应 Finding。

Calibration 验证的是 Judge 的这些判断是否与独立人工一致，不是重新计算产品指标。

### 7.2 三类制品

```text
Calibration Package
        ↓ 人工盲审
Human Label Set
        ↓ 与指定 Judge 输出比较
Calibration Result
```

#### Calibration Package

从已验证的 Evaluation 中按版本化 Selection Policy 抽取样本。Package 暴露 Judge 判断所需的最小上下文，隐藏：

- baseline/candidate 身份；
- Agent 模型、Prompt 和版本；
- Judge 原来的判断结果；
- 预期哪一方获胜。

每个样本有稳定的 `calibration_item_id`。该 ID 由 Judge Profile、Rubric/Context 版本和盲化 item payload 派生，不依赖单次 Run 的身份，因此在输入内容和 Rubric 不变时可以复用人工标签评估不同 Judge model。

#### Human Label Set

人工按 Profile 的标注协议作出判断。标签集合必须经过严格 schema、Package digest、item digest 和 source binding 校验。人工身份、时间、盲审声明和待裁决状态都保存为 provenance，但不进入 Judge 输入。

争议样本不能由系统自动选择一方；如果需要，可由独立 adjudicator 产生裁决记录。

#### Calibration Result

将一个明确 Judge Execution/Result 与 Human Label Set 对齐，输出 Judge 与人工的可重放比较。更换 Judge model 时，如果 Rubric、Context Builder 和 Calibration item 内容不变，可以复用同一 Human Label Set。

### 7.3 四类 Profile 独立计算

- `intent_equivalence`；
- `finding_equivalence`；
- `novel_factuality`；
- `evidence_support`。

每类独立报告：

- 有效人工标签数量和覆盖率；
- Judge graded、semantic unknown、judge failed 和 ungraded 数量；
- confusion matrix；
- exact agreement；
- 每个类别的 precision/recall；
- Cohen’s kappa；
- 人工与 Judge 不一致的样本引用。

不把四类 Profile 混成一个 Judge 总准确率。

### 7.4 Selection Policy 与状态

Selection Policy 至少支持：

- Judge 输出 `unknown` 的样本；
- high/critical fabricated 判断；
- Judge 与确定性检查冲突的样本；
- 从正常类别进行固定 seed 的分层随机抽样。

Policy、seed、样本数量和 source digest 进入 Package identity，不能看完结果后改变样本。

每个 Profile 独立拥有以下状态：

- `pending_human_labels`；
- `insufficient_coverage`；
- `failed_thresholds`；
- `gate_eligible`。

当前没有独立真人时，只能生成 Package 并停在 `pending_human_labels`。Fixture 可以验证协议和统计代码，但不能伪造 `gate_eligible`。

Calibration Package 可能包含源码和评审内容。完整 Package payload 只能导出到用户明确指定的受控目录，不进入 Git 或普通公开报告；Analysis Store 只保存 `package_manifest.json`、payload digest、选择记录和状态，不复制完整敏感上下文。

## 8. Regression Gate 设计

### 8.1 流程

```text
prepare baseline
prepare candidate
evaluate baseline
创建 Gate Policy
run-agent candidate
evaluate candidate
compare
执行 gate
```

Candidate 在运行前已经有不可变 Run Plan 和 Run ID，因此 Policy 可以绑定 candidate 身份而不需要读取 candidate 结果。

### 8.2 Gate Policy

`GatePolicyV1` 至少绑定：

- baseline Run/Evaluation digest；
- candidate Run Plan、Run ID 和 Agent config digest；
- Suite/Case Snapshot digest；
- Trial count；
- Comparison Policy digest；
- Evaluator/Judge/Metrics Policy digest；
- 必须使用的 Calibration Result digest；
- release eligibility 和允许参与门禁的 Case 集合；
- 逐指标约束；
- Policy schema/version 和创建时间。

Policy 支持以下约束类型：

- 绝对下限，例如 Recall 不得低于某值；
- 绝对上限，例如 Fabricated Rate 不得高于某值；
- 相对 baseline 的最大允许退化；
- Critical/High 新增漏报上限；
- Evidence Validity 下限；
- Agent Failure Rate 上限；
- Judge Coverage 最低要求；
- Token、时间和成本预算；
- 指定 Case 或 Case 集合的硬约束。

代码不内置假定的数值及格线。具体数字由真实 baseline、独立人工审查和产品要求产生，写入 Policy 后冻结。未配置的指标明确是 `not_configured`，不能默认 pass。

### 8.3 Gate Result

每项检查返回：

- `pass`；
- `fail`；
- `not_comparable`；
- `not_scorable`；
- `insufficient_coverage`；
- `not_configured`；
- `pending`。

Gate 是发布决策，不是质量分数：

- `promote`：所有必需条件通过；
- `block`：至少一个必需条件失败；
- `ineligible`：缺少可比较结果、校准、Coverage 或必要真值。

每个结果必须列出对应 Case/Trial、实际值、阈值、方向、Coverage、Comparison 引用和 Calibration 引用。

有效的 `block` 或 `ineligible` 是正常业务结果，不是制品崩溃。结构性错误仍使用现有稳定错误分类。为 CI 提供可选的严格模式：默认返回机器可读决策；`--ci` 将 `block` 和 `ineligible` 映射为稳定的非零 Policy 状态码。

### 8.4 Gate 数据权限

只有 Core Regression Cases 和 Private Held-out 可以通过 Policy 标记为 `release_blocking`。AACR、SWE 和未晋升 Synthetic 只能是 `diagnostic_only`，即使它们有可计算指标，也不能单独阻止发布。

Gate 不自动向 GitHub PR 写评论，不执行 Approve、Merge 或 Release。

## 9. CLI 设计

现有命令保持不变：

```text
review-agent-eval prepare
review-agent-eval run-agent
review-agent-eval evaluate
review-agent-eval inspect
```

新增命令：

```text
review-agent-eval compare
review-agent-eval calibrate export
review-agent-eval calibrate import-labels
review-agent-eval calibrate score
review-agent-eval gate prepare
review-agent-eval gate evaluate
```

所有命令支持 `--analysis-root` 和 JSON 输出。

### 9.1 compare

`compare` 接受 baseline/candidate Run ID、Evaluation ID 和 Comparison Policy，读取两个 source-bound Evaluation，完成严格兼容性校验，发布 Comparison Plan、Result、Report 和 Receipt。

### 9.2 calibrate

- `calibrate export` 创建盲审 Package；
- `calibrate import-labels` 校验并发布人工 Label Set；
- `calibrate score` 将指定 Judge Evaluation 与 Label Set 对齐并发布 Calibration Result。

`calibrate` 本身不调用 Judge；如果需要评估新 Judge，先使用现有 `evaluate`/re-evaluate 生成新的 Judge Evaluation。

### 9.3 gate

- `gate prepare` 在 candidate 运行前冻结 Gate Policy，并绑定 candidate Run Plan；
- `gate evaluate` 读取 Gate Policy、Comparison Result 和所需 Calibration Result，发布 Gate Result。

## 10. 安全与失败边界

### 10.1 正常分析状态

以下是有效分析结果，不应被当作程序异常：

- `not_comparable`；
- `not_scorable`；
- `insufficient_coverage`；
- `pending`；
- `block`。

### 10.2 制品/协议错误

以下情况必须 fail closed，不发布半成品：

- Run/Evaluation 不存在或 source binding 不一致；
- Summary、Trial Score 或原始 Receipt digest 不一致；
- Analysis Artifact 被篡改；
- 已存在的 Policy、Package、Comparison 或 Result 内容不同；
- Human Label 与 Package/item digest 不匹配；
- Path escape、symlink/junction/reparse/hardlink 或绝对路径越界；
- 私有代码、真值、Judge hidden output 或凭据进入普通报告；
- 比较服务意外实例化 Agent、Judge 或 acquisition client。

现有 CLI 错误分类 `2 usage`、`10 precondition`、`11 conflict`、`12 integrity`、`13 operational` 继续适用。有效的 Gate `block`/`ineligible` 通过结果字段表达；`--ci` 只额外提供稳定的 Policy 非零状态，不改变制品语义。

## 11. 测试与验收

采用风险驱动测试，不为无逻辑的 CLI 样板强制执行完整 TDD。重点增加：

- `tests/eval/test_statistics.py`：多 Trial 汇总、Coverage、分母语义、确定性 bootstrap；
- `tests/eval/test_comparison.py`：兼容性、严格配对、Case/Trial delta、not_comparable；
- `tests/eval/test_calibration.py`：盲化、Label binding、四类 Profile、confusion matrix、Cohen’s kappa 和状态机；
- `tests/eval/test_regression_gates.py`：Policy 冻结、阈值、Calibration prerequisite、逐 Case failure 和 release eligibility；
- `tests/eval/test_analysis_artifacts.py`：canonical JSON、create-only、receipt、重放和路径安全；
- CLI/API 回归：JSON 输出、`--analysis-root`、`--ci` 和错误分类。

本地 scripted E2E：

```text
Core fixture
  → 3 Trials
  → evaluate
  → calibration fixture
  → compare
  → gate
```

该 E2E 只证明协议、制品、统计和 Gate 逻辑。它不证明真实模型的 Code Review 质量，也不能替代：

- 三次真实模型 baseline；
- 独立真人 Reviewer B；
- Private Held-out 评测。

## 12. 实施分波

Task 15 的实现可拆为四个相互依赖但边界清晰的工作包：

1. `AnalysisArtifact`、RunStatistics 和确定性统计算法；
2. Strict Paired Comparison 和报告；
3. Calibration Package、Human Label Set 和 Calibration Result；
4. Gate Policy、Gate Result、CLI 与端到端/安全回归。

每个工作包都在现有 Eval v2 上增量接入，不修改已经稳定的 Target、Submission、Evidence 和 Evaluator source protocol。完成设计文档审阅后，另行生成实现 plan；实现 plan 再决定每个工作包的提交边界和风险驱动测试顺序。

## 13. 成功定义

Task 15 完成后，系统能够稳定回答：

1. 同一批提前标注的 Code Review 数据运行多次后，各项指标如何波动；
2. candidate 相比 baseline 在哪些 Case、哪些核心指标上改善或退化；
3. Judge 的语义判断是否经过独立人工校准；
4. 哪些指标因真值或 Judge Coverage 不足而不能作为门禁依据；
5. candidate 是否满足预先冻结的 release Policy；
6. 所有结论是否都能从不可变、source-bound 的原始 Evaluation 和 Analysis Artifact 重放。

## 14. 明确不做

- 不新增 Case Pass、`pass@1`、`pass^k`；
- 不创建 Overall Score；
- 不把 public benchmark 的缺失 Authority 当作零分；
- 不在 `compare` 时重新运行 Agent 或 Judge；
- 不让 Calibration 用 fixture 结果冒充真人一致率；
- 不把不同 Evaluator、Rubric、Truth、Authority 或 Target 条件的结果强行比较；
- 不在 candidate 结果产生后修改 Gate Policy；
- 不将 Runtime、Session、Memory、Risk 或内部 Reviewer 编排作为独立评测对象；
- 不下载公共数据或为 Analysis 服务开放网络；
- 不自动向 GitHub 发布、Approve 或 Merge。

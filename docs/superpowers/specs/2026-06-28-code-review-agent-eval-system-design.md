# Code Review Agent Eval System 设计

- 日期：2026-06-28
- 状态：已草拟，待用户审阅
- 项目根目录：`D:\Agent\code review agent`
- 关联主设计：`docs/superpowers/specs/2026-06-22-evidence-driven-multi-agent-code-review-design.md`
- 主要外部参考：
  - Anthropic Engineering: [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
  - SWE-PRBench: [paper](https://arxiv.org/abs/2603.26130), [GitHub harness](https://github.com/FoundryHQ-AI/swe-prbench), [Hugging Face dataset](https://huggingface.co/datasets/foundry-ai/swe-prbench)

## 1. 背景与目标

本项目的 Review Agent 不是单轮 Diff 评论器，而是一个由 Runtime、工具、上下文装配、风险分级、Reviewer
Agent、Observation Store、Evidence Reconciler 和报告器组成的系统。因此 Eval 不能只测试“某个模型会不会写一段评论”，
而要测试整个 Agent Harness 在可复现环境中是否能稳定产出正确 Review Outcome。

Eval System 的目标是回答：

1. Agent 是否发现了应该发现的问题？
2. Agent 是否给出了可追溯、有效、定位准确的证据？
3. Agent 是否避免了没有依据的 fabricated finding？
4. 风险分级和 Runtime 深度调度是否真的提升审查质量？
5. 更大的上下文、更多 Reviewer、更多工具调用是否带来真实收益，还是只增加成本和噪声？
6. 每次 Agent、Prompt、Runtime、工具或模型变更后，系统能力是否退化？

## 2. 核心原则

### 2.1 测的是 Agent System，不是裸模型

一次 eval run 必须记录：

- Agent 版本、Git commit 和配置。
- 模型、输出上限、reasoning effort、temperature、tool_choice 等调用参数。
- System prompt、工具定义、messages 构造策略和上下文预算策略。
- Runtime 决策：风险等级、review profile、assignments、工具权限和预算。
- 工具 observation、final findings、uncertainties 和最终 report。

模型结果不能脱离 Agent Harness 单独解释。一个分数对应的是“模型 + prompt + runtime + tools +
context policy + grader”的组合。

### 2.2 Outcome 优先，Transcript 用于诊断

正式评分以最终 outcome 为主：

- finding 是否命中真实问题。
- evidence_refs 是否真实存在并支撑 claim。
- fabricated finding 数量。
- Review Contract 覆盖情况。
- 运行成本、耗时和失败率。

transcript、messages、工具调用轨迹和 Observation Store 用于失败分析、debug 和定性审查，不作为唯一得分依据。

### 2.3 Deterministic Oracle 优先，LLM Judge 谨慎使用

只要可以用测试、静态检查、结构化匹配或确定性规则验证，就不交给 LLM Judge。LLM Judge 主要用于 code review
中不可避免的语义匹配，例如“Agent 的 finding 是否等价于某条人工 review comment”。

LLM Judge 的输出必须保存完整 judge input、judge output、rubric 版本和模型参数，方便复核。

### 2.4 公共 Benchmark 不能成为唯一目标

SWE-PRBench 很适合作为真实 PR review benchmark，但不能成为唯一测试集。原因：

- 公开 benchmark 可能被模型或系统间接污染。
- 人类 review comment 不是完整真理；没有被人类提到的问题不一定是错的。
- 如果只对公开 benchmark 调参，系统容易过拟合数据集风格，而不是真正提升审查能力。

因此 Eval System 必须同时包含：

1. 本地 Synthetic Deterministic Regression Suite。
2. SWE-PRBench External Benchmark Suite。
3. 后续 Private Held-out Calibrated Suite。

## 3. Eval Suite 分层

### 3.1 Suite A：Synthetic Deterministic Regression

这是第一层，也是开发期默认回归测试集。它由项目自己构建，包含小型可复现仓库或 patch，每个 case 都有明确
oracle。

用途：

- 快速验证 agent pipeline。
- 测试风险分级、审查深度、证据引用和 false positive 控制。
- 作为 CI 中可稳定运行的 regression gate。

第一版规模：

- 20 个核心 case。
- 其中 10 个必须发现真实问题。
- 5 个 false-positive trap，要求不要乱报。
- 5 个风险分级和审查深度 case，测试 Runtime 是否按风险展开不同 Assignment。

典型类型：

- 空值、边界条件、off-by-one。
- 权限绕过、路径穿越、敏感信息泄露。
- 删除校验、改变默认行为、兼容性破坏。
- 测试断言变弱、skip/xfail 滥用。
- 看似危险但实际安全的变更，用于误报控制。

### 3.2 Suite B：SWE-PRBench External Benchmark

SWE-PRBench 作为第二层，用来衡量真实 PR 场景下的 review issue detection 能力。

使用原则：

- 不把数据集内容 vendor 到仓库。
- 通过下载或用户提供路径放到本地 `.eval-data/swe-prbench`。
- Adapter 将 SWE-PRBench case 转换为本项目的 `ReviewRequest`、diff summary、context policy 和
  expected review comments。
- 保留 SWE-PRBench 的官方 rubric：`CONFIRMED`、`PLAUSIBLE`、`FABRICATED`。
- 按 Type 1 / Type 2 / Type 3 或数据集提供的难度维度分组统计。
- 按上下文配置 A / B / C，以及本项目的 Runtime Minimal Context 策略分别跑实验。

SWE-PRBench 在本项目中的定位不是“训练集”，而是外部真实世界 benchmark。

### 3.3 Suite C：Private Held-out Calibrated Suite

第三层在项目有真实使用数据后建立。来源可以是：

- 用户自己的 PR。
- Agent 实际漏报或误报的案例。
- AI 生成后经过人工校准的 case。
- 从开源项目抽取后人工确认的 review scenario。

Private Suite 不公开、不参与日常 prompt 调参，用于防止对公开 benchmark 过拟合。

## 4. Case Schema

每个 case 必须表达“代码变更 + 预期 review outcome + 评分方式”，而不只是一个代码样例。

```yaml
id: py-auth-missing-admin-check-001
suite: synthetic
origin: hand_authored
language: python
category: security
tags:
  - auth
  - permission
  - regression

repository:
  fixture_path: evals/fixtures/py-auth-missing-admin-check
  base_revision: base
  head_revision: buggy-change

task:
  intent: "Refactor admin user update endpoint"
  focus: "permission correctness and regression safety"
  context_policy: runtime_minimal

risk:
  expected_level: high
  expected_reasons:
    - "permission-sensitive endpoint changed"

expected_findings:
  - id: expected-1
    title: "Admin permission check was removed"
    severity: high
    file: app/auth.py
    evidence_hint: "update_admin_user no longer checks is_admin"
    required: true

must_not_report:
  - "style-only complaints"
  - "missing type hints"

oracle:
  type: pytest
  command: "pytest tests/security/test_admin_permissions.py"
  expected_result_on_base: pass
  expected_result_on_head: fail

grader:
  type: hybrid
  deterministic_checks:
    - pytest_oracle
    - evidence_ref_exists
  semantic_checks:
    - finding_matches_expected_issue
```

SWE-PRBench adapter 生成的 case 使用同一 schema，但 `origin` 为 `swe_prbench`，`expected_findings` 来自人工
review comments，`oracle` 以 semantic judge 为主。

## 5. SWE-PRBench Adapter 设计

### 5.1 输入

Adapter 接收：

```text
swe_prbench_dataset_path
local_repo_cache_path
case_filter
context_config
```

`case_filter` 支持：

- 按 repository。
- 按 language。
- 按 Type / difficulty。
- 按 changed files 数量。
- 按是否有可 materialize 的 base/head commit。
- 固定 smoke subset，例如 `swe-prbench-smoke-20`。

### 5.2 转换输出

每个 SWE-PRBench 样本转换为：

```yaml
ReviewRequest:
  repository_path: <materialized repo path or frozen context path>
  base_revision: <base commit>
  head_revision: <head commit>
  user_intent: <PR title/body if available>
  review_focus: "Find correctness, security, regression, and test-quality issues introduced by this PR."

EvalExpected:
  human_review_comments: [...]
  difficulty_type: ...
  official_context_config: A | B | C
```

如果样本不能稳定 materialize 成 Git repo，则进入 `frozen_context` 模式。该模式只能测试 review reasoning，不能测试完整
Repository Intelligence 工具链。

### 5.3 上下文配置

本项目对 SWE-PRBench 至少跑四种配置：

| 配置 | 含义 | 用途 |
|---|---|---|
| `swe_a_diff_only` | 只给 PR metadata 和 diff | 最小 baseline |
| `swe_b_changed_files` | diff + changed file content | 测试文件级上下文收益 |
| `swe_c_broader_context` | 使用数据集定义的更大上下文 | 对齐官方设置 |
| `runtime_minimal` | 由本项目 Runtime 根据 Intent、Risk、Assignment 动态装配上下文 | 测试本项目核心假设 |

实验不能假设“上下文越多越好”。更大上下文可能提升某些 case 的 recall，也可能稀释注意力并提高 fabricated finding。

## 6. Eval Run 数据流

```text
Eval Case
  -> Materializer
  -> Agent Invocation
  -> Artifact Capture
  -> Grader
  -> Metrics Aggregator
  -> Report
```

### 6.1 Materializer

负责准备 case 环境：

- 创建隔离临时工作区。
- 检出 base/head。
- 安装必要但安全的测试依赖，或跳过需要外部服务的 oracle。
- 记录 repo commit、数据集版本和 fixture hash。

Materializer 不允许修改原始 fixture 或数据集。

### 6.2 Agent Invocation

调用本项目的 review pipeline，并固定：

- agent commit。
- model provider 和模型名。
- parameters。
- context policy。
- runtime budget。
- reviewer_count 和是否启用多 Reviewer。

一次 eval 不应该混入临时人工干预。需要人工澄清的 case 进入 `blocked_by_missing_intent` 或使用 case 预置回答。

### 6.3 Artifact Capture

每次 run 写入：

```text
eval-runs/<run-id>/
├── run_config.json
├── cases/<case-id>/
│   ├── request.json
│   ├── materialization.json
│   ├── agent_artifacts/
│   ├── normalized_findings.json
│   ├── grader_input.json
│   ├── grader_output.json
│   └── score.json
└── summary.json
```

这些产物是调试评测结果的主要依据。

## 7. Grader 设计

### 7.1 Finding Normalizer

先把 Agent 输出统一为结构化 finding：

```yaml
finding_id: F-1
title: ...
claim: ...
severity: low | medium | high | critical
file: ...
line_range: ...
evidence_refs: [...]
confidence: ...
suggested_action: ...
```

Normalizer 必须保留原始文本和结构化字段之间的映射。

### 7.2 Deterministic Grader

用于 Synthetic Suite：

- oracle command 是否符合预期。
- expected finding 是否被命中。
- evidence_refs 是否存在。
- evidence path/line 是否在允许范围内。
- must_not_report 是否被违反。
- risk level 是否符合预期区间。

### 7.3 Semantic LLM Judge

用于 SWE-PRBench 和部分语义 case。Judge 输入只包含：

- PR metadata 和必要 diff/context。
- 人类 review comment。
- Agent finding。
- rubric。

Judge 输出：

```yaml
classification: CONFIRMED | PLAUSIBLE | FABRICATED
matched_human_comment_id: optional
reasoning_summary: brief
evidence_quality: valid | weak | missing | invalid
```

`CONFIRMED` 表示 Agent finding 与某条人工 review comment 指向同一实质问题。
`PLAUSIBLE` 表示 finding 看起来合理但未被 ground truth 明确覆盖。
`FABRICATED` 表示 finding 缺少依据、误解代码、引用无效或明显不成立。

### 7.4 Human Calibration

每次调整 Judge prompt、模型或 rubric 后，抽样复核：

- 所有 `PLAUSIBLE` finding 的一部分。
- 高严重度 `FABRICATED`。
- Agent 漏掉但系统认为容易命中的 human comments。
- Judge 与 deterministic oracle 冲突的 case。

Human Calibration 的结果进入 judge examples 和 rubric 修订记录，但不能悄悄覆盖历史分数。

## 8. Metrics

### 8.1 Finding Quality

- `confirmed_recall`: 命中的人工 expected finding / 总人工 expected finding。
- `confirmed_findings_per_pr`: 每个 PR 的 confirmed finding 数。
- `fabricated_findings_per_pr`: 每个 PR 的 fabricated finding 数。
- `plausible_rate`: plausible finding 占所有 finding 的比例。
- `severity_weighted_recall`: 按 severity 加权的 recall。

### 8.2 Evidence Quality

- `evidence_ref_validity`: evidence_refs 是否存在、revision 正确、路径和行号有效。
- `evidence_support_rate`: evidence 是否支撑 claim。
- `unsupported_claim_rate`: 无证据或证据无效的 claim 比例。

### 8.3 Runtime And Context

- `risk_calibration`: 初始风险等级与实际 confirmed/fabricated outcome 的关系。
- `contract_coverage`: Review Contract 覆盖率。
- `tool_calls_per_confirmed_finding`。
- `tokens_per_confirmed_finding`。
- `wall_time_per_case`。
- `blocked_or_failed_case_rate`。

### 8.4 Experiment Metrics

用于回答系统设计问题：

- dynamic risk depth 是否优于固定深度。
- multi-reviewer 是否优于 single-reviewer。
- `runtime_minimal` 是否优于 `swe_c_broader_context`。
- Intent Packet 充分性检查是否降低 fabricated finding。
- Evidence Reconciliation 是否降低重复报告和矛盾结论。

## 9. Report 格式

一次 eval report 至少包含：

```text
Run identity:
  agent commit, model, parameters, suite, case filter, context policy

Headline metrics:
  confirmed recall, fabricated per PR, evidence validity, failure rate, cost/time

Breakdowns:
  by suite, category, risk level, language, repository, SWE-PRBench type, context config

Regressions:
  cases improved, cases regressed, new fabricated findings, newly unsupported claims

Artifacts:
  links to case-level reports and grader traces
```

报告不只给总分。总分容易掩盖“召回提高但 fabricated finding 暴涨”的情况。

## 10. 数据管理与安全

- 外部数据集默认放在 `.eval-data/`，不提交到 Git。
- Synthetic fixtures 可以提交，但必须足够小且不包含真实密钥、私有代码或受限数据。
- SWE-PRBench 的下载、缓存、许可和版本信息写入 run artifact。
- Eval 运行默认禁网，除非 materializer 显式处于 download step。
- 仓库内容仍按主设计中的 prompt injection 边界处理，不能覆盖 Runtime Policy。
- Judge 不能看到被评测模型身份，除非该信息是实验变量。

## 11. 里程碑

### Eval-1：Local Synthetic Foundation

目标：

- 定义 case schema。
- 创建 20 个 synthetic deterministic cases。
- 实现 local eval runner。
- 实现 deterministic grader。
- 输出 run summary。

通过标准：

- 所有 case 可离线 materialize。
- grader 不依赖 LLM。
- 每个 finding 都能追踪到 case、agent run 和 evidence refs。

### Eval-2：SWE-PRBench Adapter

目标：

- 支持从 SWE-PRBench 数据集加载样本。
- 生成本项目统一 case schema。
- 支持 smoke subset 和 filter。
- 支持至少 `swe_a_diff_only` 与 `runtime_minimal` 两种 context policy。

通过标准：

- 可以稳定运行小规模 subset。
- 每个样本的 dataset version、case id、context policy 和 artifacts 可追踪。
- 无法 materialize 的样本被明确标记，而不是静默失败。

### Eval-3：Semantic Judge And Calibration

目标：

- 实现 CONFIRMED / PLAUSIBLE / FABRICATED judge。
- 保存 judge traces。
- 建立人工抽样校准流程。

通过标准：

- judge rubric 版本化。
- 对同一 run 重跑 judge 时结果可比较。
- 人工校准样本可以回写成 judge examples 或 rubric 修订，而不是修改历史事实。

### Eval-4：Regression Dashboard

目标：

- 对比两个 agent commits 或两个模型配置。
- 标出新增漏报、新增 fabricated finding、成本变化和失败率变化。
- 支持发布前 regression gate。

通过标准：

- CI 或本地命令能跑 synthetic suite。
- SWE-PRBench 作为较慢的 nightly/manual benchmark。
- 报告能直接指导下一轮修复。

## 12. 默认路线

下一步不直接从完整 SWE-PRBench 开始。默认路线是：

1. 先实现 Synthetic Deterministic Regression Suite，建立可控、快速、可调试的评测骨架。
2. 接入 SWE-PRBench smoke subset，验证真实 PR benchmark adapter。
3. 加入 Semantic Judge 和人工校准。
4. 扩大 SWE-PRBench subset，并建立 private held-out suite。

这条路线能避免一开始就陷入大型公开 benchmark 的环境、下载、repo materialization 和语义 judge 复杂度，同时不牺牲最终对真实 PR 场景的覆盖。


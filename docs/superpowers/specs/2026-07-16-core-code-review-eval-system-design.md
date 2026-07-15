# Core Code Review Eval System 设计

- 日期：2026-07-16
- 状态：已确认（含 v1 叶子协议补充，2026-07-16）
- 项目根目录：`D:\Agent\code review agent`
- 关联主设计：`docs/superpowers/specs/2026-06-22-evidence-driven-multi-agent-code-review-design.md`
- 历史参考：`docs/superpowers/specs/2026-06-28-code-review-agent-eval-system-design.md`
- 当前产品范围：面向 Python Git 仓库的本地 Code Review Agent

主要外部参考：

- Anthropic Engineering：[`Demystifying evals for AI agents`](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Alibaba AACR-Bench：[`alibaba/aacr-bench`](https://github.com/alibaba/aacr-bench)
- AACR-Bench 论文：[`arXiv:2601.19494`](https://arxiv.org/abs/2601.19494)
- SWE-PRBench Harness：[`FoundryHQ-AI/swe-prbench`](https://github.com/FoundryHQ-AI/swe-prbench)
- SWE-PRBench Dataset：[`foundry-ai/swe-prbench`](https://huggingface.co/datasets/foundry-ai/swe-prbench)

## 1. 设计结论

本项目的 Eval System 采用：

```text
AACR-Bench 的工业实践骨架
+
Anthropic Agent Eval 的方法论
+
本项目额外需要的 Intent Eval
```

AACR-Bench 提供“形”：

```text
Dataset
  -> Agent Runner
  -> Structured Review Output
  -> Location Matcher
  -> Semantic Matcher
  -> Metrics And Report
```

Anthropic 提供“神”：

- 测 Agent 最终 Outcome，不限制唯一执行路径；
- 区分 task、trial、grader、transcript 和 harness；
- Capability Eval 与 Regression Eval 分开；
- 确定性、模型和人工 Grader 各司其职；
- Trial 环境隔离；
- 用多 Trial 处理非确定性；
- Case 必须明确、可解、平衡并持续维护；
- 必须阅读失败 transcript 并校准 Judge。

本项目只评两个核心能力：

```text
1. Intent 是否理解正确
2. Review 是否正确
```

其他内部实现不进入 Eval 协议，也不形成独立产品得分。

## 2. 文档关系

旧 Eval 设计稿保留不动，继续作为历史讨论记录。本文不是对旧稿的增量修改，而是新的聚焦设计。

本文一旦确认，将成为后续 Eval Harness 设计与实现的最新来源。

## 3. 黑盒评测边界

### 3.1 Eval 只看输入和输出

被测 Agent 对 Eval System 是一个黑盒：

```text
EvalInput ------> AgentUnderTest ------> EvalSubmission
                       |
                       v
              ClarificationChannel
              (one asked turn only)
```

Eval System 不需要知道 Agent 内部：

- 如何规划；
- 如何调用工具；
- 有多少 Reviewer；
- 如何管理上下文；
- 如何保存运行状态；
- 如何恢复；
- 如何实现内部策略。

这些内容可以作为可选 trace 帮助诊断，但不是 Eval Harness 的依赖。

### 3.2 Agent Adapter

不同 Agent 通过统一 Adapter 接入：

```python
class AgentUnderTestAdapter:
    def run(
        self,
        eval_input: EvalInput,
        workspace: Path,
        config: AgentRunConfig,
        clarification_channel: ClarificationChannel,
    ) -> EvalSubmission:
        ...
```

`ClarificationChannel` 只暴露“提交一个实际问题并取得至多一个匹配回答”的方法，不暴露 Case policy、答案列表或剩余答案。Harness 持有并消费 Clarification Script；Adapter 和 Agent 永远拿不到完整脚本。

Adapter 可以：

- 调用本地 CLI；
- 调用 Python API；
- 调用 HTTP 服务；
- 调用 Claude Code、OpenCodeReview 或其他 Agent。

Eval 核心不因被测 Agent 的内部架构不同而改变。

### 3.3 当前项目 Adapter

当前项目的 Adapter 负责：

1. 把 `EvalInput` 转为正式 Review 请求；
2. 执行当前产品，并在产品实际提问时通过受控 `ClarificationChannel` 交换单个问答；
3. 从产品最终输出中提取 Intent、Findings、Evidence 和 Uncertainties；
4. 写出统一的 `EvalSubmission`；
5. 可选保存一个不透明 `trace_ref`。

产品内部 artifact 不直接成为 Eval 的 canonical schema。

## 4. 总体架构

```text
Dataset / Case Bank
  -> Case Loader
  -> Repository Preparer
  -> Agent Runner
  -> EvalSubmission
  -> Intent Evaluator
  -> Review Evaluator
       -> Location Matcher
       -> Semantic Matcher
       -> Evidence Checker
  -> Metrics Aggregator
  -> Run Report / Comparison
```

对应模块：

```text
review_agent_eval/
├── cases.py
├── datasets.py
├── repository.py
├── runner.py
├── submission.py
├── intent_evaluator.py
├── review_evaluator.py
├── match_location.py
├── match_semantic.py
├── evidence_checker.py
├── judge.py
├── metrics.py
├── report.py
└── adapters/
```

这套结构与 AACR-Bench 一样，把 Agent 执行和结果评判分开，但增加 Intent、Evidence 和更严格的 Judge 规则。

## 5. 核心术语

### 5.1 Case / Task

一个具有固定输入和成功条件的 Code Review 问题。

### 5.2 Trial

某个 Agent 配置对一个 Case 的一次独立执行。

### 5.3 Eval Run

固定一个 Agent 配置，在一组 Case 上运行一个或多个 Trial。

### 5.4 Grader

对 Intent 或 Review Outcome 的某一方面进行评分的逻辑。

### 5.5 Transcript

Agent 一次 Trial 的完整或部分运行记录。Transcript 用于诊断，不是主要得分来源。

### 5.6 Outcome

本项目的 Outcome 是：

```text
Intent Outcome
Review Outcome
```

### 5.7 执行状态与判定状态

Trial 生命周期和 Submission 终态不是同一个 enum：

```text
trial_status:
  pending | running | incomplete | completed | failed | blocked | invalid_output

submission_status:
  completed | failed | blocked | invalid_output

judge_status:
  graded | judge_failed | ungraded
```

`unknown` 是 Intent、Finding 或 Evidence 的判定结果，不是 Judge 执行成功状态。Judge 超时、Provider 失败或结构化输出非法时使用 `judge_failed`；因上游缺失而未进入 Judge 时使用 `ungraded`。

## 6. EvalInput v1

```yaml
schema_version: eval_input_v1
task_id: py-auth-admin-check-001

repository:
  source: fixture | git
  path: optional-local-path | null
  url: optional-repository-url | null
  base_revision: ...
  head_revision: ...

review_request:
  title: ...
  description: ...
  user_intent: ...
  review_focus: ...
  linked_requirements: []
  project_rules: []
  existing_ci_evidence:
    - source_id: ci-1
      text: ...
      content_hash: ...
```

EvalInput 只包含真实产品可获得的输入。

Ground truth 不得混入 EvalInput，也不得通过文件名、环境变量或工作区内容泄漏给被测 Agent。

### 6.1 Repository 不变量

- Repository exact keys 是 `source/path/url/base_revision/head_revision`；source/base/head 非 null，path/url 显式使用 string 或 null；
- `base_revision` 和 `head_revision` 必须是不同的、同长度的 40 或 64 位小写完整 Git object ID；
- `fixture` 必须提供安全的 suite-relative POSIX path，且 `url=null`；
- `git` 必须在 local path 和 URL 中恰好提供一个；URL 不得包含 userinfo/credential；
- repository path 拒绝绝对路径、盘符、UNC、`..`、空组件和 VCS metadata 逃逸。

`linked_requirements` 和 `project_rules` 是有界文本列表。每条 existing-CI evidence 是 Agent 可见的 typed snapshot：source ID 唯一，`content_hash` 是 exact UTF-8 text SHA-256；Trial 不根据 URL 重新下载内容。`external_record.source_ref` 只能引用这里的 source ID。

Review request exact keys 是 `title/description/user_intent/review_focus/linked_requirements/project_rules/existing_ci_evidence`。前四项是非空 string 或 null；后三项始终是 list。EvalInput 根 exact keys 是 `schema_version/task_id/repository/review_request`，全部非 null。

## 7. EvalSubmission v1

### 7.1 结构

```yaml
schema_version: eval_submission_v1
task_id: py-auth-admin-check-001
agent_id: current-review-agent
trial_id: trial-001
status: completed | failed | blocked | invalid_output

intent:
  status: sufficient | partial | insufficient
  goal: ... | null
  acceptance_criteria: []
  scope: []
  constraints: []
  claims:
    - claim_id: ...
      dimension: goal | acceptance_criterion | scope | constraint
      text: ...
      source: explicit | inferred
  clarification_questions:
    - turn_index: 1
      question_id: ...
      dimension: goal | acceptance_criterion | scope | constraint
      question: ...
      material_claim: ...
      matched_answer_id: ... | null
      action: confirm | correct | reject | skip | defer | null
      response: ... | null
      resolved_values: []
  uncertainties: []

review:
  findings:
    - finding_id: ...
      claim: ...
      severity: low | medium | high | critical
      path: ...
      side: left | right | null
      from_line: ...
      to_line: ...
      evidence_refs: []
      suggested_action: ... | null
  uncertainties: []

evidence:
  - evidence_id: ...
    kind: repository_file | repository_diff | command_output | external_record
    revision: ...
    path: ... | null
    from_line: ... | null
    to_line: ... | null
    command: [] | null
    exit_code: ... | null
    stream: stdout | stderr | combined | null
    source_ref: ... | null
    content_hash: ...
    excerpt: ...

usage:
  elapsed_seconds: ... | null
  input_tokens: ... | null
  output_tokens: ... | null
  total_tokens: ... | null
  tool_calls: ... | null
  cost_amount: ... | null
  cost_currency: ... | null

trace_ref: null

failure: null
```

非 completed Submission 使用结构化失败对象：

```yaml
failure:
  code: timeout | non_zero_exit | process_killed | output_overflow | invalid_json | schema_mismatch | clarification_required | agent_blocked | adapter_error | unknown
  message: ...
  retryable: true | false
```

### 7.2 Submission 叶子字段与 Clarification 组合

所有列表字段必须存在。Intent 的 `acceptance_criteria/scope/constraints/uncertainties` 和 Review 的 `uncertainties` 都是有界非空 UTF-8 文本列表；空 list 表示 Agent 明确没有该项。列表输入顺序保留，不由 hydration 根据语义去重。

Leaf exact-key/nullability 固定如下：

- Intent claim：`claim_id/dimension/text/source` 全部非 null；claim ID 唯一；
- Submission Intent：`status` 非 null，`goal` 可 null，其余六个 list 必须存在；status 与内容是否真实充分由 Intent Evaluator 评分，不由 hydration 猜测；
- Finding：`finding_id/claim/severity/evidence_refs` 非 null，Finding ID 唯一；`path/side/from_line/to_line/suggested_action` 可各自为 null，以保留可评分的坏位置；
- Evidence：YAML 中十二个字段全部存在；`evidence_id/kind/revision/content_hash/excerpt` 非 null，其他字段按 kind 在 Integrity Checker 中验证；Evidence object ID 唯一；
- Failure：`code/message/retryable` 全部非 null；message 是有界、脱敏、非空诊断文本；
- TraceRef：存在时 `type/value` 都非 null；不存在时根字段显式 null；
- existing-CI entry：`source_id/text/content_hash` 都非 null，source ID 唯一；
- Usage：对象始终存在；elapsed/cost 是 finite non-negative JSON number 或 null；四个 token/tool 字段是 non-negative integer 或 null且拒绝 bool；若 input/output/total 三者都存在，则 total 必须等于 input + output；cost amount/currency 必须成对出现。

Clarification Answer 与 Submission exchange 使用相同 action enum，但承担不同职责：Answer 是 Harness 私有脚本，exchange 是 Agent 实际交互 transcript。

| action | matched_answer_id | response | corrected/resolved values |
|---|---|---|---|
| null（未匹配/未回答） | null | null | empty |
| `confirm` | non-null | optional | non-empty resolved values；script corrected values empty |
| `correct` | non-null | non-null | non-empty |
| `reject` | non-null | optional | empty |
| `skip` | non-null | optional | empty |
| `defer` | non-null | optional | empty |

每条 exchange 的 `turn_index/question_id/dimension/question/material_claim` 都非 null；turn index 从 1 连续递增，question ID 唯一。`matched_answer_id` 必须引用 Case Script 中实际消费的 answer；Agent 是否根据 confirm/correct 正确更新最终 Intent 由 Intent Evaluator 判断。

### 7.3 终态与空值不变量

每个终态 Trial 恰好有一个 Submission；`pending/running/incomplete` 是可恢复的非终态，不要求也不得冒充终态 Submission。无法恢复的中断必须由 Runner 最终化为 failed，而不是永久停在 incomplete。

| Submission status | Failure code | Intent / Review | Grader eligibility |
|---|---|---|---|
| `completed` | `failure=null` | 两者都必须非 null | Intent 与 Review 都评分；零 Finding 合法 |
| `failed` | `timeout/non_zero_exit/process_killed/adapter_error/unknown` | 各自可 null；已产生的 canonical 部分可保留 | 对非 null 部分评分，同时计入 failure rate |
| `blocked` + `clarification_required` | `clarification_required` | Intent 必须非 null 且至少有一条未解决 clarification；Review 可 null | Intent/Clarification 评分；非 null Review 可评分 |
| `blocked` + `agent_blocked` | `agent_blocked` | 各自可 null | 对非 null 部分评分，同时计入 failure rate |
| `invalid_output` | `invalid_json/schema_mismatch/output_overflow` | 两者都必须为 null | Outcome ungraded；只计 failure |

timeout 使用 `failed + timeout`，等待用户澄清使用 `blocked + clarification_required`，均不新增 Submission status。

`evidence` 始终是 list，`usage` 始终是对象；没有数据时分别使用空 list 和全部 nullable 字段为 null 的 Usage。`cost_amount` 与 `cost_currency` 同时为 null 或同时存在；amount 必须是有限非负数，currency 是大写 ISO-4217 token。只有 Agent/Provider 实际报告的 cost 才能填写。

- canonical v1 的所有声明字段都必须出现：null 表示未知、未产生或不适用，空集合表示已观察到零项，数字 0 表示真实测得为零；空字符串不能代替 null；
- clarification `turn_index` 从 1 连续递增并保留时序；没有回答时 `matched_answer_id/action/response` 为 null。

Submission Finding/Evidence 的位置字段在 hydration 时只做类型、字符和数值边界检查；以下语义错误必须保留给 Location/Evidence Checker，不得使整份 Submission 消失：

- from/to 只出现一个、逆序或超出真实文件；
- path 为 null/不安全/不存在但仍声明 side 或 line；
- side 缺失、错误，或位置只达到文件级；
- revision 使用 `HEAD`、branch、任意 commit 或其他非 Case binding；
- content hash 形状合法但与 excerpt/replayed source 不符。

Submission 内 Finding ID 和 Evidence object ID 各自唯一；重复 object ID 会造成引用歧义，因此是 schema error。语义重复 Finding 必须保留为不同 ID，不能在 hydration 时去重。悬空或重复 `evidence_refs` 同样保留给 Evidence Checker 分类。

三层边界固定为：

1. 缺/多 key、错误 JSON 类型、超界数据、非有限数字、重复 object ID 或非法 digest 形状：整份 Submission schema invalid；
2. 合法 ref 字符串找不到 Evidence object：该 ref/Evidence 为 `missing`；重复 ref 保留并记 duplicate diagnostic；
3. Evidence object 可安全解析，但 kind-specific 坐标不完整、revision/path 未获授权、source attestation 缺失、excerpt/hash 与真实来源不一致：`invalid`。

Evidence kind 的严格语义：

| kind | 必要字段 | 必须为 null 的字段 | Integrity replay |
|---|---|---|---|
| `repository_file` | single revision、path、from/to、hash、excerpt | command/exit/stream/source_ref | 从精确 base 或 head Git object 重读完整行片段；excerpt 使用 LF，hash 是 exact UTF-8 excerpt SHA-256 |
| `repository_diff` | exact `base..head`、path、hash、excerpt | from/to/command/exit/stream/source_ref | 重放固定 `git diff --no-color --no-ext-diff --unified=3 base..head -- path`；excerpt 必须是完整输出 |
| `command_output` | head revision、non-empty argv、exit code、stream、Harness artifact attestation source_ref、hash、excerpt | path/from/to | source_ref 必须解析到 Trial manifest 中由 Harness 捕获或当前 Adapter 验证的不可变 artifact；hash 覆盖完整输出 |
| `external_record` | head revision、EvalInput existing-CI source_ref、hash、excerpt | path/from/to/command/exit/stream | source_ref 必须解析到 Agent 可见的 `existing_ci_evidence` canonical entry，excerpt/hash 必须一致 |

Repository search/symbol/commit-log 结果可以帮助 Agent 调查，但 v1 若要作为严格可发布 Evidence，Adapter 必须把它规范化为实际 `repository_file`/`repository_diff` 片段；不能仅凭搜索摘要获得 `evidence_integrity=valid`。

### 7.4 必须总有 Submission

每个终态 Trial 都必须产生且只产生一个 `EvalSubmission`。`pending/running/incomplete` 只存在于可恢复执行窗口；它们恢复后必须进入一个终态，或由 Runner 明确最终化为 failed Submission。

即使 Agent：

- 没有发现问题；
- 执行失败；
- 输出无法解析；
- 被阻塞；
- 超时；

也必须写出终态 Submission。

禁止像部分公开示例脚本一样，因为没有评论文件就直接跳过该 PR。没有 Finding 的成功执行应正常计为零 Finding；对含有 ground truth issue 的 Case，其 Recall 为零。

### 7.5 可选 Trace

`trace_ref` 只用于定位失败原因：

```yaml
trace_ref:
  type: local_path | url | opaque_id
  value: ...
```

Grader 不得依赖某种特定 trace 格式才能评分。

## 8. EvalCase v1

### 8.1 结构

```yaml
schema_version: eval_case_v1
task_id: py-auth-admin-check-001
case_version: 1

source:
  suite: core-regression
  origin: hand_authored | aacr_bench | swe_prbench | private
  source_id: ...
  source_version: ...
  source_uri: ...
  license: ...
  content_hash: ...

input:
  repository: ...
  review_request: ...

clarification_script:
  max_rounds: 2
  answers:
    - answer_id: answer-goal-1
      dimension: goal | acceptance_criterion | scope | constraint
      material_claim: ...
      action: confirm | correct | reject | skip | defer
      response: ... | null
      corrected_values: []

intent_truth:
  scorable: true
  authority: explicit_author_metadata | linked_requirement | expert_reconstructed | synthetic
  expected_claims:
    - truth_id: intent-1
      dimension: goal
      text: ...
      required: true
  forbidden_claims:
    - truth_id: forbidden-intent-1
      dimension: goal | acceptance_criterion | scope | constraint
      text: ...
      rationale: ...
  clarification_policy: required | optional | not_required

review_truth:
  completeness: closed_world | expert_augmented | human_observed
  novel_finding_policy: verify | forbid
  expected_findings:
    - truth_id: issue-1
      claim: ...
      severity: high
      category: security
      required: true
      locations:
        - path: app/auth.py
          side: right | left | null
          from_line: 42 | null
          to_line: 45 | null
      evidence_anchors:
        - fact: ...
          locations:
            - path: app/auth.py
              side: right | left | null
              from_line: 42 | null
              to_line: 45 | null
      required_context_level: diff | file | repo
      rationale: ...
  known_invalid_findings:
    - truth_id: invalid-1
      claim: ...
      category: ... | null
      locations:
        - path: app/auth.py
          side: right | left | null
          from_line: 42 | null
          to_line: 45 | null
      rationale: ...
```

### 8.2 Truth 叶子协议与不变量

- EvalCase 根 exact keys 是 `schema_version/task_id/case_version/source/input/clarification_script/intent_truth/review_truth`；case version 是 positive integer且拒绝 bool；
- Case source exact keys 是 `suite/origin/source_id/source_version/source_uri/license/content_hash`；前四项与 content hash 非 null，source URI/license 是 non-empty string 或 null；public origin 必须在 Dataset Loader 阶段提供 URI/license；
- expected Intent claim exact keys 是 `truth_id/dimension/text/required`，全部非 null；required 是 bool；forbidden claim exact keys 是 `truth_id/dimension/text/rationale`，全部非 null；
- IntentTruth exact keys 是 `scorable/authority/expected_claims/forbidden_claims/clarification_policy`；scorable 是 bool，authority/policy 按 scorable 规则 null，其余 list 始终存在；
- ExpectedFinding exact keys 是 `truth_id/claim/severity/category/required/locations/evidence_anchors/required_context_level/rationale`，全部非 null；KnownInvalidFinding exact keys 是 `truth_id/claim/category/locations/rationale`，只有 category 可 null；
- ReviewTruth exact keys 是 `completeness/novel_finding_policy/expected_findings/known_invalid_findings`，全部非 null；两个 Finding collection 始终是 list；
- ClarificationScript exact keys 是 `max_rounds/answers`；Answer exact keys 是 `answer_id/dimension/material_claim/action/response/corrected_values`，只有 response 可 null；
- expected claim、forbidden claim、expected Finding 和 known-invalid Finding 使用各自 typed object，不复用带有无意义字段的结构；
- 所有 truth ID 在所属 Case 内唯一，expected 与 known-invalid Finding ID 集合互斥；
- `intent_truth.scorable=false` 的唯一 canonical 表示是 `authority=null`、两个 claim 集合为空、`clarification_policy=null`；scorable=true 时 authority 和 clarification policy 都必须非 null；
- `novel_finding_policy=forbid` 只允许用于 `closed_world`；`expert_augmented` 和 `human_observed` 必须使用 `verify`；
- `rationale` 记录标注依据，不会进入 Agent-facing EvalInput；
- TruthLocation 的 path 必须是安全 repo-relative POSIX path；from/to 同时为 null或同时存在且 `to >= from >= 1`；path 必填，side 可 null；
- EvidenceAnchor 必须有非空 fact，locations 可为空或包含严格 TruthLocation；
- Finding 是否原子化由 annotation review 保证，dataclass 不使用关键词启发式猜测；
- `evidence_anchors` 仍是可选事实锚点，不是唯一标准 Evidence。

Clarification Script 是 Case/Harness 私有交互数据，不属于 Agent-facing EvalInput：

- `max_rounds` 是 1 到 16 的正整数；answer ID 唯一；
- `correct` 必须提供非空 `corrected_values`；其他 action 的 `corrected_values` 必须为空；
- 问题按 `dimension + material_claim` 做语义匹配，不按精确句子匹配；
- 是否应该澄清只由 `intent_truth.clarification_policy` 表达，不在 Script 中保存第二份可冲突 policy；
- Runner 每次只向 `ClarificationChannel` 返回当前问题匹配到的一个回答，不暴露 truth policy、答案列表、剩余答案或 ground truth；
- 没有匹配答案、答案耗尽或超过最大轮数时保留未解决 transcript，不由 Harness 猜测答案。

### 8.3 Case 只定义输入与答案

Case 不定义：

- 期望的工具调用顺序；
- 期望的 Agent 内部计划；
- 期望的 Reviewer 数量；
- 唯一合法的调查路径；
- 被测模型和 Prompt。

这些都不属于问题本身。

### 8.4 Intent 可以不评分

某些真实 PR 数据集只有 Review Comments，没有可靠 Intent 标注。此时：

```yaml
intent_truth:
  scorable: false
  authority: null
  expected_claims: []
  forbidden_claims: []
  clarification_policy: null
```

Harness 不得凭 PR 标题自动生成参考 Intent，再用它评 Agent。

### 8.5 Canonical JSON、ID、重复项与协议边界

三个根协议只使用 UTF-8 canonical JSON：

```text
ensure_ascii = false
allow_nan = false
separators = (",", ":")
sort_keys = true
digest = SHA-256(canonical UTF-8 bytes)
```

- schema version 使用设计中的字符串 `eval_input_v1`、`eval_submission_v1`、`eval_case_v1`；
- `task_id`、`agent_id`、`trial_id` 和上游 `source_id` 是有界 opaque ID，不因文案变化而自动改写；
- Harness 自行生成且没有权威上游 ID 的对象使用命名空间前缀加完整 64 位小写 SHA-256，不使用截断 digest；
- identity payload 必须包含 schema/version namespace 并排除自身 ID；hydration 对派生 ID 重新计算并比较；
- ID-addressed collection 按 ID 做 canonical 排序，但不转换成 set；语义重复 Finding、重复 evidence ref 和 clarification 时序不能被静默擦除；
- clarification transcript 按连续 `turn_index` 保存；其他有序 transcript 同样不得按文本重排。

唯一 JSON 读取入口必须在 hydration 前：

- 先检查原始字节上限，再严格 UTF-8 decode；
- 使用 recursive duplicate-key rejection 和 non-standard constant rejection；
- 递归拒绝 NaN/Infinity，包括 `1e999` 解析出的非有限 float；
- 每层 object exact-key，未知 key、缺 key、未知 enum/schema/version 全部拒绝；
- integer 使用 `type(value) is int`，拒绝 bool 冒充数字；
- JSON-ready 边界只接受 null/string/int/bool/finite float、字符串 key object、list/tuple 和 enum value，不接受 Path、datetime、bytes、Provider/Runtime 对象。

v1 资源上限是协议的一部分：

| 对象 | 上限 |
|---|---:|
| EvalInput JSON | 2 MiB |
| EvalSubmission JSON | 16 MiB |
| EvalCase JSON | 16 MiB |
| identifier / repo path / URL | 512 / 1024 / 4096 characters |
| title / description | 4096 / 32768 characters |
| claim / rationale / question / answer / uncertainty | 8192 characters |
| 单条 Evidence excerpt | 512 KiB UTF-8 bytes |
| requirements / rules / CI evidence | each 256 items |
| clarification answers / questions | each 64 items |
| Intent claims / Findings / Evidence | 1024 / 2048 / 4096 items |
| evidence refs per Finding | 256 items |
| truth Findings / locations / anchors | 2048 / 64 per Finding / 64 per Finding |

实现先按原始集合长度检查上限，再做任何 canonical 排序，防止大量重复值绕过资源限制。修改这些语义或上限需要新的 schema/rubric version，不能静默改变 v1。

## 9. Ground Truth 完整度

### 9.1 closed_world

用于受控 fixture 或经过完整人工审查的 Case。

- expected findings 可作为完整必要问题集；
- known invalid findings 可以作为 false-positive traps；
- Case 可以明确禁止 novel findings。

### 9.2 expert_augmented

用于 AACR-Bench。

- 多模型提供候选；
- 专家交叉验证形成正负标注；
- 覆盖率较高，但仍不假设穷尽所有可能问题。

### 9.3 human_observed

用于 SWE-PRBench 或原始真实 PR 评论。

- 人类评论是观察到的问题；
- 未匹配人类评论不自动等于误报；
- unmatched Agent Finding 继续判断为 plausible、fabricated 或 unknown。

不同完整度的数据不能混成一个总 Precision 或总排行榜。

## 10. 数据集与 Suite

### 10.1 Core Regression Suite

项目自建、长期维护的回归集，负责同时评 Intent 和 Review。

Case 类型至少包括：

- Intent 明确且无需询问；
- Intent 信息不足且必须询问；
- 合法 inferred Intent；
- unsupported Intent trap；
- 必须发现的 correctness/security/regression 问题；
- diff/file/repo-level 问题；
- clean PR；
- fabricated Finding trap；
- 错误路径、错误行号和无效 Evidence；
- 高严重度问题和低价值噪声的区分。

Capability Case 与 Regression Case 使用同一格式，但 Suite 目标不同：

- Capability：Agent 目前做不到或不稳定，提供提升方向；
- Regression：应接近稳定通过，防止已有能力退化。

### 10.2 AACR-Bench Adapter

AACR-Bench 的公开数据口径：

```text
200 个唯一 PR
1505 条专家确认评论
640 条专家否定评论
10 种语言
Diff / File / Repo Context Level
```

Adapter 映射：

- positive comments -> expected findings；
- negative comments -> known invalid findings；
- category/path/side/line -> Review Truth；
- context level -> 分组统计字段。

AACR-Bench 默认只参与 Review Eval。当前产品只运行 Python eligibility subset，但 Adapter 保留其他语言元数据。

### 10.3 SWE-PRBench Adapter

SWE-PRBench 使用 `human_observed` ground truth，并保留：

- Type1 Direct；
- Type2 Contextual；
- Type3 Latent；
- config A/B/C；
- human review comment；
- language 和 difficulty。

可以运行：

```text
official frozen-context protocol
native repository Agent protocol
```

两种 protocol 分开报告，不声称 native Agent 结果与官方 frozen-context leaderboard 完全同口径。

### 10.4 Private Held-out Suite

来自真实使用中的：

- missed finding；
- fabricated finding；
- Intent 理解错误；
- 不必要或遗漏的澄清；
- 人工严重度修正。

所有 Case 必须经过人工确认和版本化。Private Held-out 不参与日常 Prompt 调参。

## 11. Agent Runner

### 11.1 AACR 风格的运行方式

Runner 的职责保持简单：

```text
加载 Case
-> 准备仓库
-> 调用 Agent Adapter
-> 保存 EvalSubmission
```

Runner 不负责评分。

### 11.2 独立运行与评判

Agent 执行和 Evaluator 必须解耦：

```text
run-agent
evaluate-existing-submissions
```

这样可以：

- 更换 Judge 而不重新运行 Agent；
- 修复 matcher 后重新计分；
- 对同一输出比较不同 rubric；
- 单独审阅 Submission 和 trace。

### 11.3 环境隔离

根据 Anthropic 的原则，每个 Trial 必须从干净环境开始：

- 独立可写工作区；
- 固定 base/head commit；
- 不保留其他 Trial 创建的文件或 commit；
- 被测 Agent 的可变内部状态必须重置或固定；
- 外部数据准备与正式 Trial 分开；
- Trial 默认离线运行，除非 Case 明确授权外部服务。

Eval Harness 不需要知道内部状态是什么，只要求 Adapter 满足隔离协议。

### 11.4 Clarification Script

Intent Eval 不能临时找人回答问题。

Case 预置：

- 是否应该询问；
- 可以回答的 material claims；
- 固定回答；
- 最大澄清轮数。

Runner 记录 Agent 实际问题并返回预置回答。

问题按目标 claim 和 materiality 匹配，不按精确句子匹配。

Clarification Script 只存在于 evaluator-facing Case view。Agent-facing `EvalInput` 不含 policy 或 answers；Adapter 只能通过受控 `ClarificationChannel` 逐问交换。

## 12. Intent Evaluator

### 12.1 评分对象

只评 `EvalSubmission.intent`：

```text
goal
acceptance criteria
scope
constraints
clarification decision
```

### 12.2 Claim Matching

Agent Intent 和 expected Intent 都拆为 claim。

匹配流程：

1. 规范化空白、枚举和明确标识符；
2. 生成同一 dimension 的候选对；
3. 对明显等价内容做确定性匹配；
4. 对剩余候选做语义匹配；
5. 进行一对一分配。

每个 Agent claim 分类为：

```text
supported
partially_supported
unsupported
contradicted
unknown
```

`inferred` 不天然等于错误；只有缺少输入或代码依据的推断才是 unsupported。

### 12.3 Clarification

评测：

- 应问时是否问；
- 不应问时是否继续；
- 问题是否针对会改变 Review 结论的 material claim；
- 得到回答后是否正确更新 Intent。

### 12.4 Intent Metrics

```text
intent_claim_precision
intent_claim_recall
unsupported_intent_claim_rate
contradicted_intent_claim_rate
clarification_decision_accuracy
intent_case_pass_rate
```

不要求 Agent 和 ground truth 使用完全相同的措辞。

## 13. Review Evaluator

### 13.1 与 AACR 一致的主流程

```text
Generated Finding
  -> Location Candidate Matching
  -> Semantic Issue Matching
  -> One-to-one Assignment
  -> Confirmed / Plausible / Fabricated / Unknown
  -> Precision / Recall / Line Metrics
```

### 13.2 Finding Normalization

每条 Finding 规范化为：

```yaml
finding_id: ...
claim: ...
severity: ...
path: ...
side: ...
from_line: ...
to_line: ...
evidence_refs: []
```

缺失字段必须保留为缺失，不能用空值伪装成匹配成功。

### 13.3 Location Matcher

借鉴 AACR：

- path；
- side；
- line range overlap；
- configurable line distance。

但位置只用于生成候选和计算定位指标，不是语义正确性的唯一门槛。

如果 Agent 在调用点报告、ground truth 在定义点标注，只要两者是同一根因，仍可能语义命中。

### 13.4 Evidence Checker

Ground truth 必须标注 expected Finding，但不要求提供唯一标准 Evidence。同一问题可能通过变更行、调用方、测试结果或跨文件约束得到证明，强制 Agent 引用某个固定片段会错误惩罚合法调查路径。

Case 可以提供可选 `evidence_anchors`：

```yaml
evidence_anchors:
  - fact: "head 中 update_admin_user 不再检查 is_admin"
    locations:
      - path: app/auth.py
        from_line: 42
        to_line: 45
```

Anchor 用于描述必须被证明的事实和辅助 Judge，不要求 Agent 引用完全相同的位置。

Evidence Checker 先做不依赖 ground-truth Evidence 的完整性检查：

- evidence ID 是否存在于 Submission；
- kind-specific revision 是否为 Case 的 exact base、head 或 exact base..head；
- repository path/line/diff 是否能从固定 Git object 重放；
- command output 是否有 Trial manifest 中的 Harness/Adapter artifact attestation；
- external record 是否对应 Agent 可见的 existing-CI canonical entry；
- excerpt hash 是否一致且 excerpt 是否是该 kind 的完整 canonical source bytes。

JSON 类型、key 和 digest 形状由 hydration 判断；ref 不存在记 missing；source/revision/path/line/attestation/hash 内容不符记 invalid。这些都是确定性结果，不交给 LLM 修复。

之后单独判断 Evidence Support：Agent 引用的材料是否真正支持 Finding claim。该判断可以结合必要 diff、代码上下文和可选 anchor 交给 Judge。

Evidence 使用两个正交状态：

```text
integrity: valid | invalid | missing
support: supported | weak | unsupported | unknown
```

Integrity 先逐 Evidence item 计算，再聚合到 Finding：

- Finding 没有 evidence ref 或存在 dangling ref：`missing`；
- 所有 ref 都解析，但任一引用 Evidence 为 invalid：`invalid`；
- 至少引用一条 Evidence 且所有引用项都 valid：`valid`。

Support 只基于成功解析的 Evidence 判断：组合材料支持完整 material claim 为 `supported`，只支持部分关键链条为 `weak`，不能支持或与 claim 冲突为 `unsupported`，Judge/上下文不足为 `unknown`。增加一条无关或错误 Evidence 不能提高 Finding 的 publishable 状态。

### 13.5 Semantic Matcher

Judge 输入：

- Agent Finding；
- ground-truth issue；
- 必要 diff；
- 必要代码上下文；
- Agent evidence。

相较 AACR 公开 matcher，本项目不只比较两段评论文本。

### 13.6 全局一对一匹配

禁止按列表顺序贪心计分。

使用全局一对一最大权重匹配，确保：

- 一个 Finding 最多命中一个 truth issue；
- 一个 truth issue 最多被计一次 Recall；
- 重复 Finding 不重复得分；
- 改变输出顺序不会改变结果。

Ground-truth issue 必须尽量保持原子化。Agent 输出协议同样要求一条 Finding 表达一个主要问题。如果一条 Finding 混合多个问题，它最多命中一个 truth issue，不能通过一条评论重复获得多个 Recall；其他 ground-truth issue 仍按 missed 处理。

例如：

```text
Truth T1：权限检查被删除
Truth T2：测试未覆盖非管理员访问

Agent A1：权限检查被删除
Agent A2：非管理员现在可以访问
Agent A3：缺少非管理员测试

Matching：A1 -> T1，A3 -> T2，A2 为 unmatched duplicate
```

候选相似度可以形成矩阵，但最终只选择全局最大权重的一对一分配。

匹配边的权重只衡量两条 Finding 是否指向同一实质问题，包括根因、触发条件、影响和必要的位置关系。Evidence 可以帮助 Judge 理解 Agent 的 claim，但 Evidence 的完整性与支持度不得直接决定 `issue_match`，否则“问题找对但证据引用错误”会被错误地折叠成“问题没有找对”。

### 13.7 Judgement

```text
confirmed
plausible
fabricated
unknown
```

#### confirmed

通过全局一对一分配命中一个 expected issue，并与其指向同一实质问题。这里只判断问题命中，不要求 Evidence 已经通过完整性或支持度检查。

#### plausible

未命中已知 issue，但结合固定 revision 下的仓库事实，该问题仍然可能成立，且合理工程师可能提出。Agent 提交的 Evidence 是否有效、是否支持 claim，继续由独立 Evidence 状态表达。

#### fabricated

包括：

- 命中 known invalid finding；
- claim 所依赖的代码或行为不存在；
- 误解控制流、数据流或 API；
- 把 pre-existing 问题归因于当前变更。

Evidence 路径、行号、hash 或 excerpt 错误本身不自动把 `issue_match` 改为 fabricated；这些错误分别记入 `evidence_integrity` 和 `evidence_support`。只有当 Evidence 暴露出 Finding 的核心事实本身不成立时，才据此把问题判为 fabricated。

#### unknown

现有信息不足以可靠判断。

对 `human_observed` ground truth，未匹配不能单独成为 fabricated 的理由。

问题事实判断之外，Evaluator 单独保存 policy disposition：

```text
matched | duplicate | novel_allowed | novel_disallowed | known_invalid | ungraded
```

判定顺序固定为：

1. exact/semantic known-invalid 命中：`issue_match=fabricated`、`disposition=known_invalid`；
2. 对 expected Findings 做全局一对一分配：选中边为 `confirmed/matched`；
3. 与已匹配 truth 实质重复但未被分配的 Finding：不增加 Recall，记 `plausible/duplicate`；
4. 其他 unmatched 且 policy=`verify`：运行 bounded factuality Judge，得到 plausible/fabricated/unknown，disposition=`novel_allowed`；
5. 其他 unmatched 且 policy=`forbid`：不把“政策禁止”伪装成事实误报，记 `issue_match=unknown`、`disposition=novel_disallowed`，默认不运行 factuality Judge并进入 inspect/人工审查。

因此 `novel_finding_policy=forbid` 只控制 Case 接受什么输出，不改变 fabricated 的事实定义。Regression report 单独展示 novel-disallowed count。

Finding 的问题判断与 Evidence 判断必须分开保存：

```yaml
finding_id: F-1
issue_match: confirmed
matched_truth_id: issue-1
disposition: matched
evidence_integrity: invalid
evidence_support: unsupported
```

“问题找对但引用错了行号”仍可保留为 `issue_match=confirmed`，但不能成为可发布的严格有效 Finding。严格有效要求：

```text
issue_match = confirmed
AND evidence_integrity = valid
AND evidence_support = supported
```

### 13.8 Review Metrics

```text
issue_precision
issue_recall
f1
severity_weighted_recall
critical_high_miss_count
fabricated_findings_per_pr
fabricated_rate
plausible_rate
unknown_rate
line_precision
line_recall
evidence_validity
evidence_support_rate
publishable_finding_precision
```

其中：

```text
issue_precision = confirmed issue matches / all reported findings
issue_recall = matched required truth issues / all required truth issues
publishable_finding_precision = strict valid findings / all reported findings
```

这样既能区分“是否发现真实问题”，也能区分“这条 Finding 是否带着足够可靠的 Evidence 可以交给用户”。F1 是辅助指标，不能掩盖 Recall、fabricated rate 和 Evidence 质量的真实权衡。

## 14. Grader 设计原则

### 14.1 Deterministic First

程序能判断的内容不交给 LLM：

- schema；
- ID；
- path；
- revision；
- line range；
- content hash；
- exact known-invalid match；
- metric calculation。

### 14.2 LLM Where Necessary

LLM 只用于：

- Intent 语义等价；
- Finding 问题等价；
- novel Finding factuality；
- Evidence 是否支持 claim；
- 必要的 severity/actionability 判断。

### 14.3 Structured And Fail-closed

Judge 使用严格结构化输出，支持 `unknown`。

Judge 失败、超时、解析失败或引用非法 ID 时：

```text
judge_failed / ungraded
```

禁止默认判成 plausible、confirmed 或 fabricated。

### 14.4 Blind Judge

Judge 默认看不到：

- 被测模型身份；
- baseline/candidate 标签；
- Prompt 名称；
- 期望哪一方获胜。

### 14.5 Human Calibration

定期人工复核：

- unknown；
- high/critical fabricated；
- Judge 与确定性结果冲突；
- 随机抽样的 confirmed/plausible/fabricated；
- 新 rubric 或新 Judge model 的结果。

保存 Judge 与人工的一致率、Cohen's kappa、rubric 版本和校准集版本。

## 15. Trial 与非确定性

### 15.1 多 Trial

同一 Case 在同一 Agent 配置下可以运行多次。

正式模型能力 Eval 默认至少 3 个 Trial；具体数量由 Run Config 固定。

### 15.2 pass@1 与 pass^k

Code Review 的主要产品指标是：

```text
pass@1
```

因为真实 Review 通常不会运行多次再挑最好结果。

同时记录：

```text
pass^k
```

衡量多次运行都成功的稳定性。

`pass@k` 只作为研究指标，不是默认产品指标。

### 15.3 配对比较

比较模型、Prompt 或 Agent 策略时：

- 使用相同 Case 版本；
- 使用相同 Trial 数量；
- 使用相同 Grader 和 rubric；
- 展示 case-level improved/regressed；
- 报告均值、离散程度和置信区间。

模型、Prompt 和 Agent 内部策略是 Run 配置，不是额外评分维度。

## 16. Metrics 与报告

### 16.1 首页指标

```text
Intent Claim Precision / Recall
Clarification Accuracy
Review Precision / Recall / F1
Severity-weighted Recall
Fabricated Findings Per PR
Critical/High Misses
Line Precision / Recall
Evidence Validity / Support Rate
Publishable Finding Precision
Failure Rate
Average Time / Token / Cost
```

### 16.2 分组

支持按以下字段分组：

- Suite；
- language；
- issue category；
- severity；
- diff/file/repo context level；
- PR size；
- truth completeness；
- Agent configuration。

### 16.3 无单一总分

本项目不依赖一个 Overall Score 做发布决策。

目标是：

```text
在误报受控、Evidence 有效、运行可靠和成本可接受的约束下，
最大化严重问题的 Recall。
```

### 16.4 Regression Gate

回归门禁可以要求：

- Intent case pass rate 不下降；
- 不新增 critical/high 漏报；
- Precision/Recall 不超过允许回退；
- fabricated rate 不超过上限；
- Evidence validity 满足固定门槛；
- Agent failure rate 和成本不超过预算。

阈值属于版本化 Suite Policy，不能在看到 candidate 结果后修改。

## 17. Run Artifact

```text
.eval-runs/<run-id>/
├── run_config.json
├── run_manifest.json
├── cases/
│   └── <task-id>/
│       └── trials/
│           └── <trial-id>/
│               ├── input.json
│               ├── submission.json
│               ├── intent_matches.json
│               ├── review_matches.json
│               ├── judge_input.json
│               ├── judge_output.json
│               ├── score.json
│               └── trace_ref.json
├── summary.json
└── report.md
```

`trace_ref.json` 可不存在。

Run Config 记录：

- Agent 名称和版本；
- Git commit 或发布版本；
- 模型和 Provider；
- Prompt/config digest；
- 模型参数；
- Case/Suite 版本；
- Trial 数量；
- Grader/Judge/rubric 版本。

这些字段只用于回答“这次测的是谁、使用了什么配置”，不形成独立得分。

## 18. CLI

保持与 AACR 工具链相似的简单职责：

```text
review-agent-eval prepare
review-agent-eval run-agent
review-agent-eval evaluate
review-agent-eval compare
review-agent-eval inspect
review-agent-eval calibrate
```

### 18.1 prepare

下载或准备显式指定的数据集和仓库，验证版本、license、commit 和 hash。

### 18.2 run-agent

运行 Agent，只生成 Submission，不评分。

### 18.3 evaluate

读取已有 Submission，运行 Intent/Review Evaluator 并生成 score。

### 18.4 compare

对兼容的两个 Run 做配对比较。

### 18.5 inspect

展示一个 Case 的 input、submission、匹配关系、Judge 结果和可选 trace。

### 18.6 calibrate

在固定人工标注集上测 Judge 一致性。

## 19. 数据与安全

- 外部数据放入 `.eval-data/`，不提交主仓库；
- 小型自建 fixture 可以提交，但不得包含密钥或私有代码；
- 记录 dataset source、version、license 和 hash；
- ground truth 与 Agent workspace 物理隔离；
- repository 内容视为不可信数据；
- repository instruction 不能覆盖 Eval 或 Judge policy；
- Provider credential 不进入 Submission 或 Report；
- Eval 不自动向远端 PR 发布评论；
- 公开 benchmark 不作为训练集或唯一发布门禁；
- Private Held-out 的访问权限单独管理。

## 20. 实现批次

所有批次使用最终的 `EvalInput`、`EvalSubmission`、`EvalCase`、Matcher 和 Metrics 协议，不建设以后废弃的临时版本。

### Eval 1：AACR-style Core Harness

- `EvalInput v1`；
- `EvalSubmission v1`；
- `EvalCase v1`；
- Agent Adapter 接口；
- prepare/run-agent/evaluate 分离；
- Run Artifact Store；
- 确定性 Location Matcher 和基础 Metrics。

### Eval 2：Intent And Review Graders

- Intent claim matcher；
- Clarification evaluator；
- Finding normalizer；
- Evidence checker；
- 全局一对一 Review Matcher；
- Core Regression Cases。

### Eval 3：Semantic Judge And Calibration

- Intent Judge；
- Review Judge；
- structured/fail-closed 输出；
- human calibration；
- inspect 和 re-evaluate 已有 Submission。

### Eval 4：Public Benchmark Adapters

- AACR-Bench Adapter；
- SWE-PRBench Adapter；
- fixed split 和 eligibility；
- source/version/license/hash；
- official/native protocol 分离。

### Eval 5：Repeated Trials And Comparison

- 多 Trial；
- `pass@1`、`pass^k`；
- paired compare；
- 置信区间；
- Regression Gate；
- Private Held-out Case 流程。

## 21. 明确不做

本设计不包含：

- 要求 Eval 理解 Agent 的内部 Runtime；
- 要求 Agent 使用某种 Session；
- 要求 Eval 理解或评测 Memory；
- 给 Risk、Reviewer 数量、Context 策略等内部机制单独打分；
- 使用固定工具调用顺序作为主要成功条件；
- 把没有输出评论的 PR 静默跳过；
- 缺少位置字段时默认位置匹配成功；
- 只比较两段评论而不看必要代码和 Evidence；
- Judge 失败后默认分类；
- 把未匹配 human comment 自动判为 fabricated；
- 用单一 Overall Score 掩盖 Recall 和误报；
- 在 Eval 中自动发布、Approve 或 Merge PR。

## 22. 成功定义

完成后的 Eval System 必须稳定回答：

1. Agent 生成的 Intent 是否符合真实修改意图？
2. Agent 是否在必要时询问、在不必要时避免阻塞？
3. Agent 是否发现了真实问题？
4. Agent 是否产生误报或 fabricated finding？
5. Agent 是否给出了有效、支持 claim 的 Evidence？
6. 新模型、Prompt 或 Agent 版本相对 baseline 改善和退化了哪些 Case？
7. 结果是否在多个 Trial 中稳定？
8. 改善是否值得增加的时间、token 和成本？

最终结论保持简单：

```text
理解对了吗？
Review 对了吗？
结果稳定且成本可接受吗？
```

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
    def compatibility(
        self,
        eval_input: EvalInput,
        config: AgentRunConfig,
    ) -> AdapterCompatibility:
        ...

    def run(
        self,
        eval_input: EvalInput,
        workspace: Path,
        config: AgentRunConfig,
        clarification_channel: ClarificationChannel,
    ) -> EvalSubmission:
        ...
```

`ClarificationChannel` 只暴露“提交一个实际问题并取得至多一个匹配回答”的方法，不把 Case policy、答案列表或剩余答案作为 Adapter authority。Harness 持有并消费 Clarification Script；Adapter 只取得受限 channel，Agent 只取得当前一次回答。

Adapter 是 Harness 内受信任的集成代码，Python facade 是最小权限 API，不冒充同进程安全沙箱。被测 Agent、第三方 CLI 和其他不受信任代码必须位于进程/HTTP/IPC 边界之后，不能取得 channel object；Adapter 只能把本轮单个回答转发给它。若某个“Adapter”本身是不受信任代码，也必须把它放到 Runner 管理的独立进程，不能依靠私有属性或 `__dir__` 隐藏答案。

`compatibility()` 在创建 Trial 之前完成输入能力检查。Adapter 不支持 Case 所需输入能力时，Harness 以稳定 incompatibility reason 拒绝 Run 或生成显式过滤后的新 Suite/Run identity；不得创建 Trial 后再把它计为 `adapter_error`。即使调用方错误地绕过 preflight，Adapter 也必须抛出稳定 `AgentAdapterIncompatibleError`，不能伪造产品失败 Submission。

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
- matcher 没有静默的“精确文本假装语义”fallback：只要产品问题带 claim ID，就必须完整解析全部 ID 后才能选择 versioned canonical matcher，部分/全部失配不得被 proposed values 掩盖；仅在问题本来就没有 claim ID 时，产品显式给出的 canonical proposed values 才可作为 canonical claim。其余路径必须抛出稳定 incompatibility；接收任意自由文本 material claim 的 Run 必须配置真正的 semantic matcher；
- matcher 使用 `EvalRunConfig.clarification_matcher` 中的正式 `ClarificationMatcherSnapshot`，其 digest 独立进入 Run identity；Snapshot 严格绑定 matcher ID/version、implementation digest、可选 model artifact digest、rubric digest、normalization version、threshold 和 bounded parameters；`AgentRunConfig` 只能通过 verified binding 工厂构造，`ClarificationSession` 只接受该具体类型并经 matcher factory 构造 matcher，不接受 duck-typed snapshot/digest 对，修改任一维度必须产生新的 Run identity；
- 每次 material-claim 匹配生成 Harness-private immutable receipt，保存 request/matcher digest、逐候选布尔判断、action eligibility 以及 matched/unmatched/ambiguous/round-limit 结果；Agent-facing transcript 只保存实际问答，不泄漏候选答案表；
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

### 10.1 Suite Manifest v1 与 Run Case Snapshot

所有 Core/public/private 数据集先转换为同一份严格 `suite_manifest_v1`，核心 Loader 不读取数据集专用 sidecar 来决定 Case 身份、split 或分组：

```yaml
schema_version: suite_manifest_v1
suite_id: ...
suite_version: ...
source:
  kind: core | public | private
  source_id: ...
  source_version: ...
  source_uri: ... | null
  license: ... | null
  content_hash: ...
cases:
  - task_id: ...
    case_version: 1
    path: cases/example.json
    split: train | dev | capability | regression | held_out
    protocol_id: native_repository | official_frozen_context | ...
    dimensions:
      - name: language
        value: python
      - name: difficulty
        value: hard
    raw_file_size_bytes: 1234
    raw_file_sha256: ...
    canonical_case_digest: ...
    eval_input_digest: ...
    truth_completeness: closed_world | expert_augmented | human_observed
```

`dimensions` 是通用、evaluator-only 的严格 name/value 列表，用于保存 language、difficulty、benchmark type/config、PR-size bucket 等分组字段；name 使用唯一的小写 ASCII grouping key。`protocol_id` 单独存在，因为 official frozen-context 与 native repository 等执行协议参与兼容性判断，不能只当松散 tag。两者都不进入 Agent-facing EvalInput。

四种内容身份必须分开：

- `raw_file_sha256`：Case 文件 exact bytes 的 SHA-256；
- `canonical_case_digest`：严格 hydration 后 EvalCase canonical JSON 的 SHA-256；
- `CaseSource.content_hash`：上游数据记录的 provenance hash；
- Run snapshot digest：本次固定选择及全部 binding 的 canonical digest。

这些 digest 语义不同，但数值不要求互异；当 Case 文件本身就是 canonical JSON 时，raw SHA 与 canonical Case digest 可以相等。Loader 必须对同一次有界安全读取得到的 bytes 同时执行 raw hash、hydration 和 canonical digest 校验，不能 hash A、hydrate B。

每个 manifest entry 还固定 `eval_input_digest`，使 truth-free Run snapshot 中携带的 EvalInput 可独立验证确实来自该 Case binding。`raw_file_size_bytes` 同时参与单文件和 Suite 累计预算。

运行前生成 `eval_run_case_snapshot_v1`：

```yaml
schema_version: eval_run_case_snapshot_v1
snapshot_id: ...
manifest: ...
cases:
  - manifest_case: ...
    source: ...
    input: ...
```

Snapshot 只保存已验证的 manifest binding、Case provenance 和 EvalInput，不保存 Intent/Review truth、rationale 或 Clarification answers。完整 EvalCase 只留在 evaluator/Runner 私有 CaseBank。源文件后续变化不能改变 snapshot；重新打开私有 Case 时 digest 不符必须失败。

Manifest、Snapshot 每层 exact-key、严格 UTF-8 JSON、重复 key/未知字段/非有限数 fail closed；Case path 使用 suite-relative portable POSIX path并拒绝 symlink/reparse/大小写与 Unicode-normalized collision。Public source 与 public Case 必须有 URI、version、license 和 hash。固定 v1 上限为：Manifest 16 MiB、Snapshot 256 MiB、65,536 Cases、Suite raw Case bytes 累计 512 MiB、每个 Case 64 个 dimensions。

### 10.2 Core Regression Suite

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

### 10.3 AACR-Bench Adapter

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

### 10.4 SWE-PRBench Adapter

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

### 10.5 Private Held-out Suite

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

只评 `EvalSubmission.intent`，并读取 evaluator-facing 的 Intent truth、Clarification Script 以及与本 Trial/Transcript hash-bound 的 Harness-private clarification receipt；不读取 Agent trace、Runtime、Session、Memory、Reviewer 状态或内部规划：

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

### 12.2.1 v1 规范化与 generated claim 投影

Intent matching 使用固定的 `unicode-nfc-whitespace-casefold-v1` normalization policy：先做 Unicode NFC，再将 Unicode 空白折叠为一个 ASCII 空格并去除首尾空格，最后执行 Unicode `casefold`。不删除标点，不做词干化、关键词猜测或代码标识符改写。

`SubmissionIntent` 同时包含结构化字段和 provenance claims，二者不得简单相加造成重复计分，也不得只信任其中一份。Evaluator 按以下规则生成 assignment units：

1. `goal`、`acceptance_criteria`、`scope`、`constraints` 中的非 null/非空文本先按 dimension 生成结构化 claim；
2. `SubmissionIntent.claims` 中与结构化 claim 具有相同 dimension 且 normalized text 相等的 claim，按 canonical claim ID 一对一 overlay，并保留其 `claim_id/source`；
3. 未被 overlay 的结构化 claim 和剩余 provenance claim 都保留为独立 generated claim；
4. 任何语义重复都不在 hydration 或 normalization 阶段去重；多出的重复项必须进入 unmatched/precision 结果。

结构化 claim 的内部 ID 由 versioned evaluator 根据 dimension、规范化前文本和重复序号稳定生成；Agent 提供的 opaque claim ID 只作为 provenance 和审计身份，不被当作语义身份。
按 canonical `SubmissionIntent` 的既有字段上限，v1 generated projection 最多包含 1793 项（1024 provenance claims，加 goal 与三组最多各 256 项的结构化列表）；Result/hydration 必须支持该完整上限，不能让合法 Submission 在 overlay 后越界。

### 12.2.2 v1 候选、Judge boundary 与权重

候选只在相同 Intent dimension 内生成。原文完全相等使用 deterministic `exact`，normalized text 相等使用 deterministic `normalized`；其余候选只生成 typed `IntentSemanticJudgeRequest`，确定性层不得猜测语义结果。每个 request 具有绑定 generated/truth item ID、dimension、文本摘要和稳定 request ID；Task 9 负责把它转换成盲化的 Model Judge 请求，Task 8 只接受严格的 typed decision，不直接调用模型。

Judge decision 的 relation 为 `equivalent`、`partially_equivalent`、`contradicted`、`different` 或 `unknown`，并可带有界整数 `score_ppm`（0–999999）。`different`/`unknown` 不生成 assignment edge。

Assignment edge weight 是固定的正整数：

| edge source/relation | weight |
|---|---:|
| deterministic exact | 4,000,000 |
| deterministic normalized | 3,900,000 |
| semantic full relation | 2,000,000 + `score_ppm` |
| semantic partial relation | 1,000,000 + `score_ppm` |

对 expected truth，`equivalent`/`partially_equivalent`/`contradicted` 分别产生 `supported`/`partially_supported`/`contradicted`；对 forbidden truth，`equivalent` 或 `partially_equivalent` 产生 `contradicted`，其它 relation 不产生 edge。`required`/optional、`explicit`/`inferred` 不改变语义匹配权重。Assignment 先最大化总权重，再使用 Assignment policy v1 的 canonical lexicographic tie-break；每个 generated/truth item 最多出现一次。

单 Case 的 unresolved semantic candidate edge 上限为 65536；deterministic exact/normalized pair 不消耗这项 Judge 预算。为防止大量重复文本形成无界的确定性矩阵或 Judge request 展开，v1 对全部 candidate record 另设 131072 的总数上限和 64 MiB canonical JSON 累计上限；全部 Judge request 设 64 MiB canonical JSON 累计上限，其中重复展开的 generated/truth UTF-8 文本也单独受 64 MiB 累计上限约束；全部 Judge decision 设 16 MiB canonical JSON 累计上限，其中 `reason_refs` 另受 8 MiB 累计上限约束。任一上限超过时均 fail closed；candidate/request 上限返回 Harness-owned `ungraded/limit_exceeded`，非法或过量 decision merge 被拒绝，不得静默裁剪候选、reason refs 或把未调查的 pair 当作 unsupported。

可评分结果的 hydration 必须重新验证完整候选图：每个相同 dimension 的 generated/truth pair 恰好出现一次，semantic candidate 与 Judge request 一一覆盖，然后才允许重算全局 Assignment 和 canonical status。缺失 deterministic edge、semantic candidate 或待处理 request 的截断结果一律拒绝，不能把删边后的局部最优伪装成全局最优，也不能把被删掉的 Judge 工作伪装成已完成。

最终所有 Judge decision 已完整且没有 unknown 时，未被选中的 generated claim（包括重复 claim）归类为 `unsupported`；仍有 pending/failed/unknown 的相关候选时归类为 `unknown`。`inferred` 本身不产生惩罚。

### 12.3 Clarification

评测：

- 应问时是否问；
- 不应问时是否继续；
- 问题是否针对会改变 Review 结论的 material claim；
- 得到回答后是否正确更新 Intent。

Clarification policy 的 v1 真值表如下：

| policy | decision rule | failure/diagnostic |
|---|---|---|
| `required` | 至少一个 material claim 被正确匹配并实际提问；`defer` 虽消费答案但仍保持 unresolved | 未提问、wrong dimension/material claim、未消费答案分别保留 reason code |
| `optional` | 提问或继续都不因 decision 本身扣分；若提问，仍独立检查 materiality、answer consumption 和 update | 各子项单独记录，不合并成一个分数 |
| `not_required` | 不应提出问题；已提出的问题记为 unnecessary，未解决的问题记为 unnecessary-blocking | 不得把不必要的问题伪装成成功澄清 |

`confirm` 的每个 `resolved_value` 必须出现在最终 projected Intent 的同一 dimension；`correct` 必须出现 corrected values 且原 material claim 被替换；`reject` 要求原 material claim 不再存在；`skip/defer` 不要求 update，但 `defer` 必须单独计为 unresolved。Question materiality 和答案消费以 Harness-private matcher receipt 为准，不从 Agent 自报文本重新猜测。

### 12.3.1 Claim judgement 与 truth semantics

Required expected truth 只影响 recall denominator；optional expected truth 不因缺失降低 recall 或 Case pass，但其生成匹配仍进入 claim precision 明细。命中 forbidden claim 始终是 `contradicted`。`partially_supported` 与 `contradicted` 独立计数，不能折算成 `supported`。Judge 未完成、Judge failure 或上下文不足只能产生 `unknown/ungraded`，不能默认正例或负例。

`intent_truth.scorable=false` 时不生成 Judge request，所有 Intent metric denominator 为 null/not-scorable；生成的 claim 仍可作为审计输出保存，但不能被记为零分或满分。由于不可评分 `IntentTruth` 只有 `scorable=false`、null authority/policy 和空 claims 这一种 canonical 表示，`not_scorable` evaluation hydration 必须重算并校验该唯一 truth digest；不得仅凭空候选图或可篡改的 status/reason 接受不可评分结果。

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

### 12.5 Intent Evaluation Result v1

Task 8 输出严格、可 hydration 的 `eval_intent_evaluation_v1`，至少包含：`evaluator_revision`、`submission_intent_digest`、`intent_truth_digest`、`clarification_script_digest`、normalization/assignment policy version、generated claim projection、truth claim references、逐 claim candidate/match/judgement/reason、matched 与 unmatched 集合、typed Judge requests/decisions、clarification decisions、以及带明确分子/分母和 null coverage 的 metric inputs。每条实际 transcript exchange 保存 matcher digest 与完整 canonical receipt digest；receipt 缺失时二者均为 null，materiality/answer-consumption/update 也不得由 transcript 猜测。输出只绑定 immutable Submission、Case truth、Clarification receipt 和 evaluator revision；不得把 Agent/model/provider identity 当作 claim 语义证据。

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

Task 10 的 v1 edge policy 固定如下：原始 `claim` 完全相等时可产生 deterministic exact 等价边；其余候选必须经过 Finding equivalence Judge。`partially_equivalent`、`different` 和 `unknown` 均不产生正权 Assignment edge，也不能单独命中 known-invalid；只有 `equivalent` 才表示同一实质问题。Location 结果始终完整保存为候选/定位审计信号，不得用“不命中位置”删除语义候选，也不得把位置分数、severity、actionability 或 Evidence 状态加入 issue edge weight。若同一 Case 的 expected 与 known-invalid Finding 出现 canonical claim 冲突，Evaluator 必须 fail closed，不能用 known-invalid-first 静默掩盖标注矛盾。

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

`ReviewEvaluationResult` 是受控的 evaluator artifact，不是可由调用方自由拼装的普通 DTO。公开裸构造和 `dataclasses.replace` 必须拒绝；可信结果只能由 `ReviewEvaluator.evaluate()` 在完成候选图、Judge receipt、Assignment、Outcome、coverage 和 metric projection 后内部产生。`from_dict/from_json` 必须接收真实 Submission、ReviewTruth、ReviewEvaluator 与 `JudgeExecutionResult` 集合，重新执行完整确定性评测，并将输入 canonical payload 与重放结果逐字节比较后返回重放结果。这样每条 receipt 的 request/task/request digest、evaluator execution digest 与真实 `JudgeExecutionResult.digest()` 才具有来源绑定；任何伪造 semantic decision、result digest、Submission/Truth/context digest 或删改候选图的 artifact 都必须拒绝。

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

Task 11 的 v1 severity 权重固定为 `low=1`、`medium=2`、`high=4`、`critical=8`，policy version 为 `severity-weight-policy-v1`；完整权重表及其 canonical digest 必须进入 Trial/Case/Aggregate score identity。`severity_weighted_recall` 只对 required expected truth 计算：分子是已匹配 required truth 的权重和，分母是全部 required truth 的权重和。缺权重、未知 severity、版本或 digest 不匹配必须 fail closed，不能临时退回 1/2/3/4。

Task 11 的 v1 Line 指标固定为 `assigned-truth-location-v1`：

- `line_precision = 最终 confirmed Assignment 中定位正确的数量 / 最终 confirmed Assignment 中 truth 至少有一个 location 的数量`；
- `line_recall = 定位正确的 required truth Assignment 数量 / 至少有一个 location 的 required truth 数量`；
- “定位正确”只允许读取该 Finding 最终 Assignment 对应 truth 的 `LocationAuditRecord.match.matched`；其他 expected/known-invalid 候选位置即使匹配，也不能替最终 Assignment 得分；
- Agent 位置缺失时仍进入有 location 的 Assignment 分母并判未定位；truth 自身没有 location 时不进入 Line 分母；
- Line 指标与 issue match、Evidence、severity/actionability 分开保存，不能反向改变 Assignment。

所有 Ratio 使用 aggregate numerator/denominator 的 ratio-of-sums，不平均 Trial 或 Case 百分比。`0/0` 的 value 为 null 并标记 `zero_denominator`；not-scorable、ungraded、missing coverage 与真实 0 必须分开。Agent 的 failed/blocked/invalid-output Trial 始终进入 failure-rate 分母；Outcome 是否按 miss 计入由版本化 `failure_outcome_policy` 显式控制，默认 `count_as_missed-v1`，不能静默从质量指标和 coverage 中消失。Judge failure 只进入独立 Judge/ungraded coverage，不冒充 Agent failure 或产品 Finding 结论。

failed/blocked Submission 已经产生的非 null Intent 或 Review 仍必须由对应 Evaluator 正常评分，同时独立计入 Agent failure；只有缺失的部分才适用 `failure_outcome_policy`。若非 null 部分因 Judge failure/ungraded 而不可评分，它保持 `ungraded`，不能再被 Agent failure policy 覆盖成 miss。缺失输出不会虚构 precision 分母：默认 policy 只把可定义的 recall/case-pass/severity/line-recall outcome 计为 miss，其余 precision/rate 显式标记 `failure_excluded` 或 `missing`。

这里的“缺失”必须区分两种来源：Agent 没有产生该部分输出时，才可按 Agent failure policy 计 miss；Agent 已产生非 null 输出但对应 Evaluator artifact 尚未产生、丢失或无法加载时，必须标记 `missing_evaluation`/`missing`，不能伪装成 Agent 漏报。另一侧已有的 Evaluator result 仍独立评分，不能因为两侧没有成对完成就丢弃。

Case/Aggregate coverage 分开保存 `planned_trial_count`、`terminal_trial_count`、`intent_scored_trial_count`、`review_scored_trial_count` 和 `fully_scored_trial_count`，不得用一个实际只代表 Review 的含糊 `scored_trial_count`。F1 由 aggregate Precision 与 Recall 派生，没有单一独立 Trial coverage；F1 artifact 必须同时引用两侧 coverage，特别是在 failure-as-miss 使 Recall 计 miss、Precision 排除同一失败 Trial 时，不能只复制 Precision coverage。Case/Aggregate hydration 必须逐一验证 source Trial 的 score ID、digest、task 和 trial ID；只比较 trial ID 不足以形成来源绑定。

Metrics compatibility 至少绑定 Run/Suite/Snapshot/Trial count、protocol、truth completeness、novel policy、Agent 与 clarification matcher config、Evaluator execution、Intent/Review evaluator revision，以及包含完整 severity/Line policy snapshot 的 MetricsPolicy。Cost currency 使用集合级 single-currency-or-missing 规则：同一聚合可混合某一真实 currency 与 missing，但两个不同真实 currency 必须拒绝。所有 cost 都缺失时仍保存 `observed=0 / population=N / missing=N`，不能用 `cost=null` 丢掉 coverage。

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

#### 14.3.1 Judge 协议与执行边界

Task 9 的 Judge 不是一个面向所有任务的通用打分 Prompt，而是四个版本化、互相独立的 Judge profile：

```text
intent_equivalence
finding_equivalence
novel_factuality
evidence_support
```

每个 profile 都固定自己的 model/provider/adapter identity、模型参数、system-prompt digest、rubric/version/digest、response schema/version/digest、context builder 和 parser version。它们全部进入 `EvaluatorExecutionConfig`；这些配置用于复现、缓存隔离和分组，不形成额外产品分数。

Judge 只通过项目统一的 `ModelAdapter` 与 factory 发起 `ModelTurnRequest`。Eval business module 不直接拼接 OpenAI、DeepSeek、Claude 等 HTTP。Judge request 强制：

- `tools=[]`、`tool_results=[]`、`tool_choice=none`；
- 有界 request timeout、request deadline、input/output byte/token budgets；
- adapter capability preflight 必须证明 tool choice、timeout 和 response byte limit；第三方 adapter 缺少任一能力时在执行前拒绝；
- 每次 retry 新建 adapter，且每个 attempt 保存 typed status、elapsed、bounded output digest、Provider/model identity 和 failure；
- response 的 Provider/model identity 必须与 profile 完全一致，identity drift、tool call、截断、超限和 schema 解析失败均 fail closed。

`unknown` 是模型成功执行后对语义信息不足的合法分类，因此 Judge execution status 仍是 `graded`；Provider error、timeout、invalid output、capability/identity failure 属于 `judge_failed`；上游数据缺失、策略跳过或不可评分则是 `ungraded`。三者不得互相伪装。

同一个 immutable blind model turn digest 加完整 `EvaluatorExecutionConfig.digest()` 形成 cache key。只有 `graded` 结果可写入 cache；失败结果永不缓存。`judge_input.json` 和 `judge_output.json` 使用 typed aggregate artifact，分别保存完整有界输入和带 attempt 的输出，并通过 evaluator execution digest、input artifact digest 及 request 集合做交叉绑定。

Intent Judge 的 `judge_failed` 必须转换为带 request ID、failure code 和 execution digest 的 `IntentSemanticJudgeFailure`，进入 Intent evaluator 的独立 failure 集合；它会使 Intent evaluation 成为 `ungraded`，但不会被当作 `pending_judge` 或语义 `unknown`。Judge decision 的 reason refs 只能引用该 request 的 generated/truth source ID。

使用 fake/scripted Model Adapter 的单元测试只验证上述协议、blind context、结构化解析、failure taxonomy、retry/cache 与路由；预先把 relation/factuality/support 写入模型输出再解析，不能证明真实 Judge model 作出了正确语义判断。真实模型对同义改写、相邻但不同 issue、错误根因、compound Finding、novel factuality 和 Evidence support 的准确性，必须使用独立人工标注 Case 与 calibration set 测量，并报告与人工的一致率。协议测试通过不得被表述成语义能力已经通过。

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

`summary.json` 是报告层的权威 canonical 数据投影，底层 immutable Submission、Intent/Review evaluation、Judge receipt 与 Score artifact 仍是其可重放根来源。`report.md` 只能由已验证的 summary/inspection 对象纯渲染：不读 repository 或其他文件、不调用模型、不运行 Scorer/Aggregator，也不从 Markdown 反向恢复数据。已持久化 summary/inspection 的 hydration 必须接收真实根来源，完整重放并逐字节比较 canonical payload；顶层报告对象禁止公开裸构造和 `dataclasses.replace`。

同一 Run 中 protocol、truth completeness、novel policy 或其他 `ScoreCompatibilityKey` 不同的 Case 必须形成独立 partition；报告可以同时展示多个 partition，但不能生成跨 partition 的质量 roll-up。Run/Suite/Snapshot/Agent/Evaluator identity 不一致则整份报告拒绝，不能借 partition 掩盖。分组标签只能来自持久化 `CaseDimension` 的共同真实投影，不能由报告层从 Finding 文本猜测。

Inspect 直接复用 canonical EvalInput、Submission、IntentEvaluationResult、ReviewEvaluationResult、TrialScore 和 receipt/ref payload，因而保留 claim/Finding Assignment、Judge receipt、Location audit、Evidence diagnostics、clarification receipt 与 trace ref；不另造第二套匹配或 Evidence 语义。Trace 只展示 ref/capture 元数据，不嵌入 raw trace content、hidden reasoning、credential 或本地绝对路径。

Inspect 对 EvalInput、Submission、Intent/Review evaluation 与 TrialScore 使用明确版本化的 `eval_redacted_artifact_projection_v1` wrapper，绑定原 artifact digest、source ID/schema 和 redaction list；不得把替换过敏感字段的 payload 继续冒充原 canonical artifact。所有 TraceRef（包括 opaque value）、绝对/遍历路径和 URL 都只显示稳定 opaque projection，避免“把本地路径伪装成 opaque ID”或在 URL userinfo/query 中夹带凭据。

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
├── case_snapshot.json
├── run_manifest.json
├── cases/
│   └── <opaque-case-path-id>/
│       └── trials/<trial-id>/
│           ├── trial_manifest.json
│           ├── input.json
│           ├── submission.json
│           ├── trace_ref.json
│           ├── receipts/
│           │   ├── attempt-0001/start.json
│           │   ├── attempt-0001/incomplete.json
│           │   ├── prepare.json
│           │   └── terminal.json
│           └── evaluations/<evaluation-id>/
│               ├── evaluator_execution_config.json
│               ├── intent_matches.json
│               ├── review_matches.json
│               ├── judge_input.json
│               ├── judge_output.json
│               ├── score.json
│               ├── report.md
│               └── receipt.json
└── evaluations/<evaluation-id>/
    ├── summary.json
    └── report.md
```

`case_snapshot.json` 是 10.1 节定义的 truth-free `eval_run_case_snapshot_v1`。它与 `run_config.json` 都由 `run_manifest.json` 的内容 hash/size 引用；Config 同时固定 manifest digest、snapshot ID、snapshot digest 和完整 `SuiteCase` bindings。三者在创建和每次加载时交叉验证，不能只相信其中一份文件。`trace_ref.json`、incomplete receipt 和各 evaluation namespace 可不存在。

Run/Trial manifest 是不可变计划，不保存可变 status。运行状态只从 create-only stage receipts 派生：start 表示一次 attempt 已开始，incomplete 表示该 attempt 可恢复，prepare receipt 最后提交 EvalInput，唯一 terminal receipt 最后提交终态 Submission。`start_trial` 返回的 active attempt 是后续 incomplete/prepare/finalize mutation 的必填 lease；旧 worker 不能省略 attempt，也不能在 retry 已启动后把旧输出提交成当前 attempt。恢复只能采用与 immutable plan 完全一致的 orphan artifact 或补写缺失 receipt，不能重写已有 Submission。

Case 的原始 `task_id` 不进入目录名；`opaque-case-path-id`、run ID、trial ID、evaluation ID 都由稳定 identity payload 派生并做单路径段校验。`canonical_case_digest` 在 Suite Config、Run/Trial manifest 中保持同名，不能重新缩写成语义含糊的 `case_digest`。

Run Config 记录：

- Agent 名称和版本；
- Git commit 或发布版本；
- 模型和 Provider；
- Prompt/config digest；
- 模型参数；
- Clarification Matcher Snapshot 及其独立 config digest；Snapshot exact keys 为 `matcher_id/matcher_version/implementation_digest/model_artifact_digest/rubric_digest/normalization_version/threshold/parameters`，只有 model artifact digest 和 threshold 可为 null；
- Suite manifest digest、Case Snapshot ID/digest、Case bindings 与版本；
- Trial 数量；
- Grader/Judge/rubric 版本。

Run ID 只绑定 Agent-side execution identity：run instance key、Agent config、Clarification Matcher config digest、影响 Agent 执行的 timeout/output/trace/artifact/parallel resource budgets、Case Snapshot/Suite binding 和 Trial 数量；不绑定 Evaluator/Judge 及只影响重评的 evaluator timeout，因此同一 Submission 可以重评。每次重评把 Judge/model/rubric 配置、evaluator timeout 与 execution artifact file/total budgets 冻结为严格 `eval_evaluator_execution_config_v1`；其完整 digest 与显式 revision 派生独立 `evaluation-id`。Evaluator receipt 同时保存 execution digest、revision 与 ID，并在 hydration 时重新验证三者关系。相同 Judge/revision 但 timeout 或执行预算不同，也必须得到不同命名空间，不能覆盖或伪装成同一次评测。

所有 JSON artifact 使用 canonical UTF-8、单文件/累计读取预算、内容 hash 和 create-if-absent 发布。`run_config.json`、`case_snapshot.json`、Run/Trial manifest、receipts 与必须始终落盘的 canonical terminal Submission 属于 control plane，分别受协议上限约束，不受 execution artifact budget 意外截断；v1 Run Config 上限为 32 MiB，Artifact Store 默认单文件/累计读取上限为 256/512 MiB，正好能够持久化协议允许的 Snapshot。Agent 原始 stdout/stderr、trace 与 evaluator 产物另受 `max_execution_artifact_file_bytes` / `max_execution_artifact_total_bytes` 约束，字段名不得再伪装成覆盖所有 control-plane artifact 的全局上限。

文件内容在发布前 fsync；POSIX 额外 fsync 父目录。Python stdlib 在 Windows 不提供等价的目录 fsync，因此 Windows 明确只声明 file flush 与 atomic no-overwrite publication，不声称与 POSIX 父目录持久化语义等价。POSIX 的创建、写入、hard-link publish 与 lock 使用 descriptor-relative 路径；Windows 在写入/锁期间持有拒绝 delete-share 的已验证目录 handle chain，防止父目录被替换成 junction/reparse path。

Evaluator-only 的模块级 Submission loader 以 read-only 模式打开既有 `.eval-runs`；根目录缺失时必须报错，不能以“读取”为名创建目录或 fsync 用户磁盘。Harness 必须显式传入与 Agent workspace 物理隔离的 `.eval-runs` 根，不提供 `for_workspace` 之类容易把 Judge input/truth 写回 Agent 可见工作区的便利入口。

这些配置和 artifact 字段只用于回答“这次测的是谁、使用了什么配置、输入绑定是什么、结果如何复现”，不形成独立得分。

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

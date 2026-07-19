# Core Code Review Eval 标注协议

**协议版本：** `core-annotation-v1`

**适用对象：** `eval_case_v1`、`suite_manifest_v1` 中的 Core Capability / Core Regression Case

**规范词：** “必须”“不得”“应”“可以”均为操作要求；示例使用 YAML 便于阅读，实际 Case 和 Manifest 必须写成 canonical UTF-8 JSON。

## 1. 目标与不可协商原则

本协议定义如何把一个代码审查任务变成独立、可复核、可重复评分的 Case。标注的对象只有两类产品结果：

1. Agent 是否正确理解 Intent，并在必要时正确澄清；
2. Agent 是否发现真实 Finding，避免虚构 Finding，并给出有效且支持 claim 的 Evidence。

标注不得规定工具调用顺序、内部计划、Reviewer 数量、Prompt、模型、Runtime、Session、Memory 或唯一调查路径。

所有 Case 必须遵守以下原则：

- **先定义任务和真值，后运行 Agent。** 当前 Agent、baseline、candidate 或 Judge 的输出不得参与首次 ground truth 的生成。
- **黑盒评分。** 真值只对应 `EvalInput`、clarification transcript 和 `EvalSubmission` 中可观察的结果。
- **独立真值。** Intent 和 Finding 必须能由作者、独立审阅者以及固定的 base/head 仓库事实复核，不能把 head 当前实现直接当成需求。
- **确定性优先。** revision、path、side、line、hash、schema、重复项和 Evidence integrity 由程序校验；人工只判断语义、materiality、严重度和完整度。
- **原子、一对一。** 一个 truth claim / Finding 表达一个可独立判断的命题；不能通过复合描述或重复描述多赚 Recall。
- **问题与证据正交。** Finding 是否命中与 Evidence 是否有效、是否支持 claim 分开标注和评分。
- **失败可见。** 无 Finding、澄清阻塞、Agent failure、Judge failure 和 `unknown` 都是结果，不得通过删除 Case 或改写真值隐藏。
- **真值不因模型表现移动。** 如果 Agent 输出暴露了遗漏，必须重新进行人工审阅并发布新的 Case/Suite 版本；不得只为让某个 Run 得高分或低分而补标。

## 2. Case 准入与来源

### 2.1 准入条件

Core Case 只有同时满足以下条件才可进入 Suite：

- 任务是可由固定 `base_revision` 与 `head_revision` 重放的小型 Python 代码审查任务；
- fixture 根目录严格只有完整的 `base/` 与 `head/` 两棵源码树；
- 两个 revision 不同，均由 `FixtureRepositoryBuilder` 的固定身份、时间、消息和排序规则生成；
- 任务说明是现实用户可能提供的 `ReviewRequest`，不是给 Agent 的解题提示；
- Intent 真值有独立 authority，或按本协议明确标记为不可评分；
- Review 真值可由作者与至少一名独立人工审阅者在固定 revision 上复核；AI 可以协助作者，但不能充当该人工审阅者；
- 测试不依赖网络、当前时间、随机顺序、机器路径、未固定依赖或外部可变服务；
- 不含密钥、私有代码、个人数据、未知许可证材料、symlink/reparse point、special file、submodule、LFS、nested repository 或 VCS metadata；
- Case 足够小，人工能够对 base、head、diff、相关调用方和测试完成一次完整审查；
- Case 不是仅为验证格式、工具顺序或模型措辞而存在；它测量的是 Code Review 结果。

以下情况必须拒绝或暂停标注：

- 预期行为只能从有缺陷的 head 实现倒推；
- 独立审阅者无法判断问题是否由本次变更引入、扩大或暴露；
- 触发条件依赖未固定的生产数据或不可重放环境；
- Finding 只表达审美、无项目规则支持的风格偏好或无实际影响的重构建议；
- 为了把当前 Agent 做不到的任务放进 Capability 而降低真值标准；
- disagreement 尚未裁决；
- fixture 或输入中存在 truth leakage。

### 2.2 Fixture 与 revision

目录必须为：

```text
<fixture-id>/
  base/
  head/
```

`base/` 和 `head/` 都是完整仓库快照，不是 patch。标注流程必须：

1. 用 `FixtureRepositoryBuilder` 构建确定性仓库；
2. 把返回的完整 `base_revision`、`head_revision` 写入 `input.repository`；
3. 保存并复核 builder 返回的 base/head tree digest；
4. 修改任一 fixture byte 后重新构建，禁止沿用旧 revision、tree digest 或 manifest hash；
5. 不手写 Git object ID 或 tree digest。

`Repository.source=fixture` 时，`path` 必须是 Suite 内的安全相对路径，`url` 必须为 `null`。

### 2.3 `CaseSource` provenance

Core Case 使用：

- `origin=hand_authored`；
- `suite` 与装载它的 `SuiteManifest.suite_id` 完全一致；
- `source_id` 是稳定的逻辑来源 ID，不包含答案，例如 `core-py-001-source`；
- `source_version` 与 `SuiteSource.source_version` 完全一致；
- 纯自建 Case 的 `source_uri`、`license` 可以为 `null`；使用外部材料时必须记录真实 URI、版本、许可证和内容 hash；
- `content_hash` 必须由不可变来源记录计算，不得填占位 hash。

手工 Case 的来源记录至少要绑定：任务提案、合法的 Agent-facing 需求文本、fixture logical source digest、base/head tree digest 和本协议版本。`CaseSource.content_hash` 不得循环地定义为“包含自身 hash 的完整 EvalCase hash”。

公开或私有生产材料不得伪装成 `hand_authored`。无法公开的真实材料使用 `origin=private` 和 Private Suite，不得复制进 Core Suite。

### 2.4 可读任务说明

`ReviewRequest.title` / `description` 必须让不知道答案的工程师理解要审查什么。允许的内容是现实输入，例如变更目标、作者声明、验收要求、review focus、项目规则和已有 CI；禁止的内容是 expected Finding、严重度、真值位置、澄清答案或“本 Case 在测什么”的提示。

`review_focus` 是用户希望重点审查的区域，不是 Intent。`user_intent` 才是用户直接声明的修改意图。

## 3. Intent 标注

### 3.1 四个不得混用的概念

| 概念 | canonical 位置 | 合法值 | 含义 |
|---|---|---|---|
| `IntentAuthority` | `intent_truth.authority` | `explicit_author_metadata`, `linked_requirement`, `expert_reconstructed`, `synthetic` | ground truth 为什么可信 |
| `IntentClaimSource` | `submission.intent.claims[*].source` | `explicit`, `inferred` | Agent 最终 claim 的来源/确认状态 |
| `IntentResult` | `submission.intent.status` | `sufficient`, `partial`, `insufficient` | Agent 最终 Intent Packet 是否足以支持审查 |
| scorability | `intent_truth.scorable` | `true`, `false` | 本 Case 是否有可靠 Intent 真值可评分 |

不得在 `IntentTruth` 中增加 `source`、`intent_source`、`status` 或 `intent_status`。当前 Case schema 没有这些字段。Case 作者必须在盲审记录中先判断预期 source/status 行为，golden Submission 再使用 `IntentClaimSource` 和 `IntentResult` 表达；机器评分仍以现有 canonical schema 为准。

### 3.2 `IntentAuthority`

`intent_truth.authority` 是 Case 级 authority，按下列规则选择：

- `explicit_author_metadata`：全部 required truth 由 Agent 可见的 `user_intent`、title、description、project rules 或同等明确作者元数据直接支持；
- `linked_requirement`：全部 required truth 由 `review_request.linked_requirements` 中的固定需求直接支持。该值要求 linked requirements 非空；“非空”本身不证明内容匹配，仍须人工复核；
- `expert_reconstructed`：没有直接作者声明，但专家可根据独立需求事实、base 行为、兼容性契约和变更上下文重建意图。不得只因 head 已这样实现就选择此值；
- `synthetic`：Case 作者在构造 fixture 前定义了人工 oracle，预期行为属于受控合成任务。Core 的隐藏澄清答案通常使用此 authority。

当前 authority 是 Case 级单值。所有 required Intent claims 必须能由同一选定 authority 支持。若 required claims 依赖互不相同且无法忠实归并的 authority，必须拆分 Case、重新设计输入，或将 Intent 设为不可评分；不得随意挑一个“最强”标签。

### 3.3 `IntentTruth.scorable`

设置 `scorable=true` 的必要条件：

- authority 独立于被测 Agent；
- expected / forbidden claims 可以由作者与至少一名独立人工审阅者复核；
- 对 material ambiguity 已有合法的 Clarification Script；
- expected claims 不是从 PR 标题、head 实现或当前测试断言自动生成后未经人工确认的；
- authority 与 `clarification_policy` 均可明确填写。

`scorable=false` 只用于确实没有可靠 Intent oracle 的数据，不是避免低分的手段。其唯一 canonical 形状是：

```yaml
intent_truth:
  scorable: false
  authority: null
  expected_claims: []
  forbidden_claims: []
  clarification_policy: null
```

不得保留“仅供参考”的 expected claim；Evaluator 会把这一组合判为 schema invalid。

Core 中 `scorable=true` 的 Case 必须至少有一个 `required=true` 的 `goal` claim，并标出所有会改变审查结论的 acceptance criterion、scope 和 constraint；无关字段不必为了“填满结构”制造 claim。

### 3.4 Expected / forbidden Intent claims

每个 `ExpectedIntentClaim` 必须是一个原子命题：

- `dimension=goal`：为什么改、要达成的主要行为；
- `dimension=acceptance_criterion`：完成后可验证的行为；
- `dimension=scope`：本次有意修改或明确不修改的边界；
- `dimension=constraint`：必须保持的兼容性、安全性、数据或运行约束。

原子化判据：一个 claim 应能被独立判断为 supported / partially supported / unsupported / contradicted / unknown。如果一句话中的两个部分可以分别为真、分别被确认或分别影响审查结论，必须拆成两个 truth claim。

`required=true` 表示漏掉该 claim 必须降低 Recall 并使 Intent Case 不能通过。只对有助理解但不要求 Agent 必须复述的内容使用 `required=false`。不得为了让当前 Agent 通过而把 material claim 改为 optional。

`ForbiddenIntentClaim` 只标注明确错误且有实际诱惑力的解释，例如把“保持旧 API”误解为“删除旧 API”。它不是所有未列出想法的黑名单。其 `rationale` 必须说明与哪一权威来源冲突，以及错误解释会如何改变审查结论。

所有 truth ID 在一个 Case 内全局唯一，使用稳定、无答案泄漏的形式，例如 `intent-001`、`forbidden-intent-001`、`issue-001`、`invalid-001`。

### 3.5 `IntentClaimSource`

虽然 truth schema 不保存预期 source，作者和审阅者必须按以下口径审核 golden Submission 与真实输出：

- `explicit`：来自用户输入、PR/任务明确元数据、linked requirement、明确 spec/ADR/README/测试说明、项目规则、交互式回答，或由用户确认/纠正后的 inferred claim；
- `inferred`：模型根据 base/head、diff、普通实现、普通测试断言或提交历史推断，但尚未被确定性来源或用户确认。

LLM 从明确文档中提取内容仍是 `explicit`；LLM 基于 head 行为猜测目标仍是 `inferred`。changed files 只能形成 inferred scope 候选，不能自动成为 confirmed scope。

Clarification 后：

- `confirm`：被确认的 proposed value 升级为 `explicit`；
- `correct`：`corrected_values` 成为 `explicit`，原 material claim 必须消失；
- `reject`：原 material claim 必须消失；
- `skip` / `defer`：不能把 claim 升级为 `explicit`，未解决内容应保留在 inferred/uncertainty 语义中。

### 3.6 `IntentResult`

`SubmissionIntent.status` 是最终状态，不是 ground-truth authority：

- `sufficient`：可靠审查所需的关键 goal、acceptance criteria、scope 和 constraints 已由 explicit claim 覆盖，没有未解决的 material question；
- `partial`：审查可以继续，但仍有非阻塞缺失项或关键内容部分依赖未经确认的 inferred claim；
- `insufficient`：关键缺失或冲突会阻止可靠的语义审查。

内容“看起来完整”不等于 `sufficient`。关键值仍为 inferred 时应为 `partial` 或 `insufficient`。`status` 与 claim 真伪分别评分；一个自信的 `sufficient` 不会使 unsupported claim 成真。

### 3.7 Clarification policy 与 Script

`clarification_policy` 使用 canonical `ClarificationPolicy`；Script 的 `action` 使用 canonical `ClarificationAction`。决策只看：缺失信息是否可能改变审查结论。

| policy | 使用条件 | 合格行为 |
|---|---|---|
| `required` | 至少一个 material ambiguity 会改变 Finding、严重度、scope 或验收结论 | Agent 必须提出并正确匹配至少一个 material question，消费答案并按该 action 的语义处理最终 Intent |
| `optional` | 问或不问都可安全继续；提问仍可能提高确定性 | 决策本身不扣分；若提问，materiality、答案消费和更新仍必须正确 |
| `not_required` | Agent-facing 信息已足够，或缺失信息不影响审查结论 | 不提问并继续；提问属于 unnecessary，若因此未达到 `sufficient` 则为 unnecessary-blocking |

Script 规则：

- `max_rounds` 为 1–16；只设满足任务所需的最小值；
- 一个 `ClarificationAnswer` 对应一个独立 material claim；语义等价的重复 answer 会造成 matcher ambiguity，必须合并；
- `material_claim` 写待确认的命题，不写问题句式，也不写完整答案表；
- dimension 必须与最终受影响 claim 的 dimension 一致；
- `required` 至少有一个 answer；`not_required` 默认 answers 为空；
- `confirm` 的 `corrected_values=[]`，实际 resolved values 来自 Agent 提问时提供的 proposed values；
- `correct` 必须有非空 `response` 和非空 `corrected_values`；
- `reject`、`skip`、`defer` 的 `corrected_values` 必须为空；
- `defer` 明确保留 unresolved；`skip` 不要求更新，但不能把未被其他 authority 解决的 claim 升级为 `explicit`，最终 source/status 必须如实反映剩余 uncertainty；
- 没有匹配、超过轮数或答案耗尽时保持 unresolved，不由 Harness 猜答案；
- Script、候选答案和 matcher receipt 都是 Harness-private，不得进入 Agent workspace。

## 4. Review Finding 标注

### 4.1 什么是可标注 Finding

`ExpectedFinding` 必须同时满足：

1. claim 在固定 base/head 上事实成立；
2. 问题由本次变更引入、扩大、重新暴露，或本次新增调用使既有缺陷第一次影响目标行为；
3. 有现实触发条件；
4. 有可描述的用户、数据、安全、兼容性、可用性或可维护性影响；
5. 合理工程师会在合并前提出；
6. 可以给出独立的修复或验收结论。

推荐 claim 句式是“变更事实 + 触发条件 + 影响”，但不要求固定措辞。例如：

> head 删除了 `is_admin` guard，因此任意已登录的非管理员在调用 `update_user` 时都能修改其他账户。

以下内容不得成为 expected Finding：

- 没有项目规则支撑的命名、格式或个人偏好；
- 只说“可能有 bug”“需要更多测试”而没有具体缺口；
- 只在 base 已存在且 head 未扩大、未触发、未重新归因的问题；
- 与当前 change scope 无关的仓库债务；
- 依赖不存在代码、错误控制流或错误 API 语义的猜测；
- 修复建议本身，而没有独立问题 claim。

### 4.2 原子化、compound 与 duplicate

一个 Finding 是一个根因、一个主要触发链和一个可独立处置的问题。按以下测试决定是否拆分：

- 两部分能否分别为真或分别为假？
- 修复其中一部分后，另一部分是否仍存在？
- 两部分是否具有不同根因、触发条件或验收方法？
- 两部分是否会被分配给不同 owner？

任一答案为“是”时通常必须拆分。多个后果若都由同一根因和同一修复消除，可以保留为一个 Finding，并在 claim/rationale 中说明主要影响。

以下情况是 duplicate，必须合并：

- 同一 root cause 在定义点和调用点各写一条；
- “权限检查被删除”和“非管理员可以访问”只是同一问题的原因/后果改写；
- 同一缺陷在多个相邻行重复描述。

合并后可以在一个 `ExpectedFinding.locations` 中保留多个合理位置。不得把 duplicate 保留为多个 truth ID；全局一对一 matcher 只会让一个 Agent Finding 命中一个 truth issue。

缺失测试只有在存在具体、独立、项目要求必须覆盖的行为缺口时才可单独标注，不能作为每个功能 bug 的自动附加 Finding。

### 4.3 Pre-existing 问题

标注前必须逐条比较 base 与 head：

- base 已有且 head 未使其更严重、未新增可达路径、未扩大输入范围：不是 expected Finding；
- head 新增调用使既有危险函数进入本次目标路径：可以是 Finding，rationale 必须写明新增 exposure；
- head 扩大影响范围、改变默认值或移除 mitigation：可以是 Finding，只描述新增 delta；
- “本次变更引入 X”而 X 在 base 已存在：应作为有代表性的 `KnownInvalidFinding` trap，而不是 expected Finding。

不得因为一个问题“值得修”就把 pre-existing debt 归因给当前 PR。

### 4.4 Clean PR

Clean Case 的 `expected_findings=[]`，但它仍是完整 Case，不得跳过。进入 clean Case 前，作者与独立人工审阅者必须确认：

- 变更符合 Intent 和项目规则；
- 没有 material correctness/security/regression 问题；
- 相关 base/head 行为、边界条件和调用方已检查；
- 任何看似可疑但事实错误的高概率 trap 已按需写入 `known_invalid_findings`；
- `completeness=closed_world` 的完整性标准已满足。

“没有 expected Finding”不表示任何 unmatched Agent Finding 自动 fabricated。Evaluator 仍按 `novel_finding_policy` 处理；已知、具体、可证伪的 trap 才写入 `KnownInvalidFinding`。

### 4.5 `KnownInvalidFinding`

只标注满足以下条件的 false-positive trap：

- 是当前 diff 容易诱发的具体错误解读；
- claim 可以由固定 base/head 事实明确否定；
- rationale 说明为什么错误，例如 pre-existing、路径不存在、控制流不可达或 API 语义被误解；
- claim 不与任何 expected Finding canonical 冲突。

不要穷举所有不可能的 Finding，也不要把“真实但低价值”写成 known-invalid。真实但非必报的问题应是 `required=false` 的 `ExpectedFinding`，或在 completeness/policy 下作为 novel Finding 评估。

标注会按固定顺序被 Evaluator 消费，作者不得用 rationale 改变该顺序：

1. 命中 `KnownInvalidFinding`：`IssueJudgement=fabricated`、`FindingDisposition=known_invalid`；
2. 全局一对一命中 `ExpectedFinding`：`confirmed + matched`；
3. 与已匹配 issue 语义重复但未分配：`plausible + duplicate`，不增加 Recall；
4. 其他 unmatched 且 `novel_finding_policy=verify`：经 factuality Judge 得到 `plausible/fabricated/unknown + novel_allowed`；
5. 其他 unmatched 且 policy 为 `forbid`：`unknown + novel_disallowed`，不能冒充 fabricated。

### 4.6 `required`

`ExpectedFinding.required=true` 表示合格 Code Review 必须发现，漏报进入 Recall 和 Case pass。通常：

- 所有 `high` / `critical` Finding 必须 required；
- 会导致错误行为、数据损坏、安全边界破坏或明确兼容性回退的 `medium` Finding应 required；
- `low` 只有在 Case 明确测量该规则且漏掉应算失败时才 required。

`required` 与 severity 独立。不得通过把严重 Finding 改成 optional 来迁入 Regression。

### 4.7 Severity

`FindingSeverity` 只允许 `low`、`medium`、`high`、`critical`。按可合理触发时的影响、blast radius、可恢复性和权限边界判断，不按代码行数、修复难度或当前 Agent 是否能发现判断。

| severity | 操作定义 | 典型例子 |
|---|---|---|
| `low` | 局部、可恢复、影响有限；仍是具体 review issue，不是纯风格 | 边缘诊断信息错误、受限路径上的小型可维护性风险 |
| `medium` | 现实输入下出现错误结果、异常或兼容性问题，影响有限且有合理规避方式 | 特定边界值失败、非核心调用方回退 |
| `high` | 主要功能失效、显著数据完整性/可用性损害、重要安全边界破坏，或影响广且难以规避 | 授权绕过、常见路径数据丢失、公开 API 破坏 |
| `critical` | 可直接造成广泛或不可逆损害、严重可利用安全事件或核心系统普遍失效，且缺少现实 mitigation | 未认证任意账户接管、大范围不可逆数据破坏、默认路径全面停机 |

若两个审阅者相差两级或对 `high/critical` 有分歧，必须由第三名 adjudicator 裁决。严重度不得取平均值。

### 4.8 Category

Core v1 使用以下固定 category 文本：

- `security`：主要影响 confidentiality、integrity、authentication、authorization、injection、secret 或 trust boundary；
- `regression`：主要问题是 head 破坏 base 中受支持且应保持的行为或兼容性；
- `correctness`：其他错误结果、控制流、状态、数据、异常、并发或契约问题。

冲突时按 `security` > `regression` > `correctness` 选择主要 category，并在 rationale 解释。不要写同义变体，例如 `bug`、`auth`、`compat`、`correctness_bug`。公共数据 Adapter 可以保留其固定上游 taxonomy；本规则只约束 Core。

### 4.9 `RequiredContextLevel`

选择能可靠证明完整 claim 的最小上下文层级：

- `diff`：diff 和 hunk 上下文足以确认根因、触发和影响；
- `file`：必须读取同一文件中的非 diff 代码、状态或控制流；
- `repo`：必须跨文件查看调用方、类型、配置、规则、测试映射或数据流。

该字段描述“验证问题最低需要什么”，不是 bug 所在文件数，也不是 Agent 实际用了什么工具。若只看 diff 能看到可疑改动，但必须读调用方才能证明影响，应标 `repo`。

### 4.10 Truth locations

Core 手工 Case 的 location 必须使用：

- repo-relative POSIX `path`；
- `side=left` 表示 base/deleted side，`side=right` 表示 head/added side；
- 同时存在的 `from_line` / `to_line`，且范围尽量窄地覆盖 causal site；
- 文件必须在所选 side 存在，行号必须在范围内。

Schema 为公共数据保真允许 null side/line，但 Core 不得用部分 location 冒充可定位真值。若问题没有稳定、诚实的行位置，使用 `locations=[]`，让它退出 Line metric denominator；不得填一个任意文件或宽泛的整文件范围。

同一 issue 可有多个 location，但只能表示同一根因的等价定位点，不能把多个 issue 塞进一个 Finding。Location 只影响定位审计和 Line metric，不改变 issue semantic match。

### 4.11 Rationale

每个 `ExpectedFinding.rationale` 至少回答：

- base 与 head 的相关事实分别是什么；
- 触发条件和影响是什么；
- 为什么归因于本次变更而非 pre-existing；
- 为什么 severity、category、required 和 context level 如此选择；
- 哪些替代解释已被排除。

每个 `KnownInvalidFinding.rationale` 必须给出可复放的反证。Rationale 是 evaluator-private，不能复制进 Agent-facing request 或 fixture。

## 5. Truth completeness 与 novel policy

### 5.1 `TruthCompleteness`

- `closed_world`：受控 fixture 已被完整人工审查，expected findings 是完整的必要问题集；Core 手工 Case 默认且应使用此值；
- `expert_augmented`：专家和候选生成提高覆盖率，但不声称穷尽；主要用于 AACR-Bench 类数据；
- `human_observed`：只记录真实 PR 中观察到的人类评论；主要用于 SWE-PRBench 类数据。

不同 completeness 必须分开汇总。Capability 不是低质量 truth 的收容区；一个 Core Case 若无法达到 `closed_world`，不得仅因“很难”就降为 Capability。

`SuiteCase.truth_completeness` 必须与对应 `EvalCase.review_truth.completeness` 完全一致；不得只改其中一处。

### 5.2 `NovelFindingPolicy`

- `forbid`：只允许 `closed_world`。用于范围窄、已穷尽审查的 Core Case；unmatched Finding 得到 `unknown + novel_disallowed`，不自动等于 fabricated；
- `verify`：允许对 unmatched Finding 做 bounded factuality Judge；`expert_augmented` 和 `human_observed` 必须使用此值。

`forbid` 不是减少标注工作的捷径。只有作者、独立人工审阅者和 adjudicator（如有）确认 truth 完整后才可使用。

### 5.3 完整性检查

完成 Finding 标注后，审阅者必须独立执行：

1. 从 base 到 head 的行为差异审查；
2. changed symbols 的调用方/被调用方检查；
3. 相关项目规则、输入边界、错误路径和测试检查；
4. 对每个 issue 做 pre-existing comparison；
5. 反向寻找 clean/fabricated traps；
6. 对 truth 列表做 duplicate/compound 检查。

若新发现 issue，先完成原子化和裁决，再冻结 truth；不得把第一次 Agent Run 当作完整性检查步骤。

## 6. Evidence anchor 与 Evidence 有效性

### 6.1 `EvidenceAnchor` 的用途

Case 不保存唯一标准 Evidence。`evidence_anchors` 只描述完整 claim 必须证明的 material fact，帮助 Judge 理解，不要求 Agent 引用同一文件或同一行。

一个 anchor 必须：

- `fact` 原子、可验证且与 Finding 的证明链直接相关；
- location 只在确实帮助定位该事实时提供；
- 不规定工具、命令或唯一调查路线；
- 不重复 Finding claim 本身；
- 不把修复建议当事实。

只有在没有 anchor 会使 Evidence support Judge 难以区分“支持完整 claim”和“只支持部分链条”时才添加。简单 diff-level Finding 通常可以使用空 anchors。

### 6.2 Evidence integrity

`EvidenceIntegrity` 是对 Agent `SubmissionEvidence` 的运行期结果，不是 Case 中预先标注的一组标准 Evidence：

- `valid`：至少一个 ref，全部 ref 可解析且每个 Evidence item 都按 kind 精确重放；
- `invalid`：所有 ref 可解析，但至少一个 item 的 revision/path/field/attestation/hash/excerpt 不合法；
- `missing`：Finding 没有 ref，或存在 dangling ref。

四种 Evidence kind 的 exact 规则：

- `repository_file`：revision 必须是 exact base 或 head；path 和完整 line range 必填；excerpt 是该范围的 canonical UTF-8/LF 文本，hash 对 exact excerpt；
- `repository_diff`：revision 必须是 exact `base..head`；path 必填；line、command、stream、source ref 等不适用字段为 null；excerpt/hash 对该 path 的完整 canonical diff；
- `command_output`：revision 必须是 exact head；必须有完整 argv、exit code、stream 和当前 Trial 的 Harness/Adapter attestation；截断或自报输出无效；
- `external_record`：revision 必须是 exact head；`source_ref` 必须命中 Agent 可见的 `existing_ci_evidence`，excerpt/hash 必须与该记录一致。

不要手工猜 hash。Golden Submission 中的 valid Evidence 必须由固定 replay 生成；bad-Evidence golden 则只改变被测试的一个维度（例如 wrong line 或 hash），避免同时制造多个无法归因的错误。

### 6.3 Evidence support

`EvidenceSupport` 与 integrity 正交：

- `supported`：有效材料组合支持完整 material claim；
- `weak`：只支持关键链条的一部分；
- `unsupported`：材料与 claim 无关、冲突或无法推出 claim；
- `unknown`：Judge 或上下文不足，不能可靠判断。

严格 publishable Finding 只有：

```text
issue_match = confirmed
AND evidence_integrity = valid
AND evidence_support = supported
```

错误 path/line/hash 本身不把正确问题改判为 fabricated；反之，有效文件片段也不能让错误 claim 变成 confirmed。

## 7. 盲化作者—审阅者流程

### 7.1 角色

每个新增或语义修改的 Case 至少有：

- **Author A**：构造 task、fixture、ReviewRequest 和第一版 annotation；
- **Reviewer B**：独立、盲化地重做 Intent/Review 标注；不得是 Author A；
- **Adjudicator C**：仅在 material disagreement 时介入；不得以 Agent/Judge 输出代替裁决。

AI 可以帮助生成候选 fixture 或列出调查问题，但不能作为最终 annotator，也不能满足“至少一次人工审阅”。签署者对每个 truth claim 负责。

### 7.2 冻结顺序

1. Author A 先冻结合法的 Agent-facing `EvalInput` 和 base/head fixture；
2. Author A 在不运行被测 Agent/Judge的情况下写第一版 truth；
3. Reviewer B 只收到与 Agent 相同的 EvalInput、固定 base/head 和本协议，不得看到 Author truth、rationale、Suite placement、golden Submission 或任何 Agent 输出；
4. Reviewer B 独立提交 Intent claims、clarification decision、Finding 列表、severity/category/context、completeness 和 rationale；
5. 同时揭盲，两份标注按原子 claim/Finding 一对一比对；
6. 无 disagreement 时 Reviewer B 签署 checklist；有 disagreement 时进入第 8 节；
7. 完成 truth leakage review、schema/hash/lint 和 golden Submission deterministic regression；
8. 冻结 Case/Suite digest 后，才允许运行当前 Agent 建立 baseline。

模型输出若在冻结后暴露可能漏标：停止使用该 Case 做发布判断；由未看过该输出的 reviewer 先独立检查。确认漏标后按第 10 节升级版本并重新盲审，不能直接把该输出复制进 truth。

### 7.3 审阅记录

人工审阅记录至少包含：

- task ID / Case version / fixture base-head identity；
- Author、Reviewer、可选 Adjudicator 的稳定身份；
- 协议版本；
- 盲审开始/结束时间；
- 独立标注的 digest；
- disagreement 列表及裁决；
- leakage checklist；
- 最终接受/拒绝结论。

不得在被测 Agent workspace 保存 Reviewer B 的私有 work product，也不得伪造一个未知 JSON key 塞入 `EvalCase`。审批 source of truth 是 evaluator-private `eval/human-reviews/records/<task_id>.json`；`annotation.json` 只是 builder 从该 ledger 做出的可读投影，永远不是审批凭证。Release gate 必须打开 ledger，重放其独立 response、Author receipt、可选 adjudication、当前 Case/fixture/protocol binding，再逐字段确认 annotation 等于可信投影。只手改 `annotation.json` 的 `status`、身份、checklist 或 digest 必须失败。

ledger 必须绑定 exact task ID、Case version、canonical Case/EvalInput digest、base/head revision/tree/source digest、fixture byte manifest、协议 digest、完整 canonical batch manifest、packet/batch digest、独立标注 digest 和确定性 comparison digest。hydrate 时必须重算 batch digest，确认当前 task 的 exact packet reference 位于该 manifest，并用这个 manifest 重放 response binding；不得根据 ledger 顶层 ID/digest 拼出伪 batch。最终记录使用原子 no-overwrite 发布；同一 task/version 已有记录时必须拒绝静默覆盖、回滚和 stale binding。truth 或 fixture 变化必须先按第 10 节升级 Case，再重新盲审，不能“更新”旧审批。

稳定 human ID、签署时间和外部引用只是审计数据，机器无法证明现实中的签署者确实是人、确实独立或没有私下看过答案。CLI 只能拒绝明显的 Agent/LLM 身份和自相矛盾的 attestation；真实性必须由受审 PR、CODEOWNERS/组织身份、会议或工单审计记录等仓库外证据证明。AI、子 Agent、Judge 和模型操作者代填的名字都不能充当 Reviewer B 或 Adjudicator C。

pending 记录必须诚实：人工 atomicity、severity/category/context、truth completeness、semantic leakage、anchor 非唯一性和 known-invalid 检查在签署前均为 `false`，disagreement 状态为 `pending_blind_review`。只有 builder 当次已确定性重放的 schema、fixture/VCS 和 base/head binding 检查可以预先记为通过。

### 7.4 Blind packet、response 与 receipt

最终 CLI 是 `eval/authoring/core_human_review.py`：

- `export` 只接受用户明确指定且位于仓库外、尚不存在的目录；输出 canonical EvalInput、固定 base/head bytes、协议和空 response template，不输出 Case、truth/rationale、Suite placement、golden 或 Agent/Judge 结果；
- packet digest 绑定 opaque canonical Case digest、EvalInput digest、fixture manifest、revision/tree/source identity 和协议 bytes；batch digest 再绑定 exact packet 集；
- `verify-response` 严格拒绝 unknown key、重复 JSON key、stale digest、时间倒置、缺失 attestation/checklist、明显机器身份和不能由 `IntentTruth` / `ReviewTruth` hydration 的 truth；
- `import` 只有在 Author receipt 完整、Author A != Reviewer B、外部审计引用存在时才原子写 ledger；有 material disagreement 时还必须有独立 Adjudicator C 的完整逐项 resolution。

Reviewer B 的 `core_independent_human_response_v2` 必须包含稳定身份、开始/结束 UTC 时间、blind/no-output/human/independent attestations、独立 `IntentTruth`、结构化 `clarification_decision`、完整 `ReviewTruth` 和全部 human checklist。`clarification_decision` 包含 `policy`、`max_rounds`、`answers`、`rationale`、`exchanges`：`answers` 必须通过 canonical `ClarificationScript` / `ClarificationAnswer` hydration；Reviewer 使用自己生成的 opaque `answer_id`，不要求与 Author ID 相同；每条 exchange 必须用该 `answer_id` 一对一绑定 Reviewer answer，且 `answered_at` 位于盲审起止时间内。`required` 必须至少有一条 answer 且每条 answer 恰有一条 exchange；`not_required` 的 answers/exchanges 必须都为空。`ReviewTruth` 的 canonical 模型覆盖 Finding 原子 claim、severity、category、required、locations、Evidence anchors、required context、completeness、novel policy 和 rationale。

揭盲 comparison 是严格逐字段 comparison。Clarification 比较忽略 Author/Reviewer 各自 opaque `answer_id`，但必须比较 `policy`、`max_rounds`，以及每个 answer 的 `dimension`、`material_claim`、`action`、`response`、`corrected_values`；其余 truth 的措辞、truth ID、列表结构、location、anchor 或 rationale 不同都会产生 material difference。机器不得把差异自动归为“语义等价”。无 difference 时 Author 可以签署接受/拒绝；有 difference 时 Adjudicator C 必须对每个 `difference_id` 写唯一 resolution 和理由。若 Reviewer/merged truth 胜出，当前 Case 必须拒绝并按第 10 节重做；只有所有 difference 都裁定当前 Author truth 成立时，当前 Case binding 才能被接受。

## 8. Disagreement 与 adjudication

### 8.1 必须比较的项目

揭盲时逐项比较：

- Intent scorability、authority、claim identity/dimension/required、预期 provenance 行为、最终 status 口径；
- clarification policy、material claim、action、corrected values、max rounds；
- expected / known-invalid identity、atomicity、pre-existing attribution；
- severity、category、required、context level、locations、anchors；
- truth completeness、novel policy、clean-PR 结论；
- provenance 和 leakage。

即使看起来“仅措辞不同”，确定性 comparison 也必须先记录 difference，不能自动合并。只有 Adjudicator C 可以在保留原始两份记录的前提下说明 root cause、trigger、impact 和验收边界为何一致并作出 resolution；语义边界不同不能用编辑文字掩盖。

### 8.2 裁决规则

- Author A 和 Reviewer B 先分别提交证据，不得在看见对方结论后改写原始独立记录；
- issue identity 按 root cause、trigger、impact 和独立修复性裁决；
- severity 按第 4.7 节，不投票、不平均；
- intent ambiguity 按“是否会改变审查结论”裁决；
- Adjudicator C 必须查看 exact base/head 和 authority source，并写出选择/拆分/合并理由；
- Agent/Judge 的多数意见、置信度或当前得分不能作为 authority；
- material disagreement 未解决时，Case 不得进入 Capability 或 Regression；
- 不得用 `required=false`、降低 severity、改成 `human_observed` 或标 `unknown` 来隐藏 disagreement。

裁决导致 fixture、input、truth、script、rationale 或 Suite placement 变化时，按第 10 节重新版本化。

## 9. Truth leakage 禁止项

### 9.1 严禁泄漏的内容

下列内容不得进入 fixture tree、commit message、文件/目录名、task ID、ReviewRequest、测试输出、环境变量、working directory、Agent trace 初始数据或 Agent 可访问的持久存储：

- `intent_truth` / `review_truth`、truth IDs、expected/known-invalid Finding；
- severity、required、category、context label 和 annotation rationale；
- Clarification Script、完整答案表、未消费答案和 matcher receipts；
- golden Submission、expected Evidence/hash、Judge rubric/decision；
- `capability` / `regression` placement 或当前 Agent 的历史结果；
- 类似 `auth-bypass-case`、`expected-bug-line-42`、`must-ask-scope` 的答案型命名；
- 当前 Agent、baseline、candidate 或 Judge 生成后被复制回仓库的提示文本。

### 9.2 合法需求不是 leakage

现实用户本来会提供的 `user_intent`、description、linked requirement、project rule 或仓库内既有 spec 可以进入 Agent 输入。它们一旦进入，就必须按真实 authority/source 口径标注，不能一边把答案写给 Agent，一边把对应 claim 称为 inferred。

普通项目测试可以表达可观察契约；禁止的是 eval 专用测试名、注释或失败消息直接说出 hidden Finding。head 中新增测试的当前断言也不能仅因存在就自动成为 ground-truth Intent。

### 9.3 泄漏审计

接受前必须完成：

- 序列化 `AgentCaseView`，确认只有 canonical `EvalInput`；
- 搜索 truth ID、rationale、answer text、expected claim 和答案型别名是否出现在 base/head 与 Agent-facing 文件；
- 检查 task/fixture/file/test 名称是否中性；
- 确认 fixture commit message 由固定 builder 生成；
- 确认 Suite/Case/truth/golden 文件不被复制进 Trial workspace；
- 确认 Agent 进程拿不到 annotator record、Case root 或 evaluator config；
- 人工阅读 ReviewRequest，删除不符合现实输入的解题提示。

自动搜索通过不等于无泄漏；同义提示和结构性提示必须人工检查。

## 10. Capability / Regression 与版本迁移

### 10.1 Suite 含义

- **Capability**：真值已完全审计，但当前产品尚未稳定完成，用来指明能力缺口；
- **Regression**：发布基线应稳定完成，用来阻止已获得能力退化。

两者使用同一个 `EvalCase v1` schema、同一原子化标准和同一 grader。不得为同一任务维护“宽松 Capability truth”和“严格 Regression truth”。

仓库中出现 Case 文件或 Capability/Regression manifest 只表示候选数据已被物化，不表示已经准入。只要完整 Task 13 门禁仍因缺少 source-bound 真人审批、未解决 disagreement 或 Regression 三次真实 baseline 而失败，这些 pending Case/manifest 就不得作为 release gate、发布质量承诺或已批准 Suite 使用。不得通过单独运行排除外部门禁的 pytest 子集，把工程验证通过误写成 Suite 准入通过。

### 10.2 进入 Regression 的最低条件

Case 只有满足以下条件才可从 Capability 提升到 Regression：

- truth、provenance、leakage 和 golden deterministic regression 全部通过；
- 在固定 release baseline、同一 Suite/Case/Evaluator 配置下至少 3 个独立 Trial 均无 Agent failure 或 Judge-ungraded；
- Intent 可评分时，3 个 Trial 的 `intent_case_pass` 均为 true；
- 所有 required Finding 均命中，无 critical/high miss；
- 无 `known_invalid` / fabricated Finding；
- required matched Finding 的 Evidence 达到 `valid + supported`；
- clean Case 无 fabricated/known-invalid 输出；
- 人工抽查确认通过不是由泄漏、过拟合措辞或 Judge 偏差造成。

Task 15 的正式 repeated-trial/gate 实现完成前，可以生成审计记录，但不得声称已由自动 Regression Gate 完成 promotion。

Task 13 不新增平行的 `regression_promotion_receipt` schema。`annotation.json.suite_assignment.promotion_evidence` 只允许保存 `run_id`、`evaluation_id`、`summary_id` 三个 locator；`trial_count`、`passed` 或手填 digest 不是证据。晋级验证必须从 locator 打开原始 `.eval-runs`，通过 `ArtifactStore` 和 `EvaluationOrchestrator.load_run_evaluation` 对 Run Config、Case Snapshot、Run/Trial Manifest、terminal Submission、clarification receipt、Judge input/output、Intent/Review evaluation、TrialScore 和 Run summary 做 source-bound replay。

晋级 Run 还必须满足：固定的 promotion run-instance key；`trial_count >= 3`；完整、未过滤的 strict preflight；`current-agent-cli-v1` adapter；非 `unknown` commit 和非 `working-tree` Agent version；目标 Suite/Case/EvalInput digest 与当前 manifest 完全一致；所选 evaluator execution 等于 Run 初始 evaluator execution。必须使用目标 Case 在该 Run 中的全部 Trial，不能只挑成功的三次。上述本地 artifact 绑定能防止只改 annotation 过门禁，但不能提供密码学签名；可信 CI 签名属于 Task 15。

### 10.3 迁移规则

- `task_id` 保持稳定且永不复用；
- Capability 和 Regression 在同一 Suite snapshot 中不得同时包含同一 task；
- 当前 `CaseSource.suite` 必须等于目标 `SuiteManifest.suite_id`，因此迁移必须修改该字段，而不是把同一 Case 文件同时挂到两个 manifest；
- 迁移会改变 canonical Case identity，必须递增 `case_version`；
- 从原 Suite 删除、向目标 Suite 增加，并同时递增两个受影响的 `suite_version`；
- 更新与目标 Suite 一致的 `source_version`，重算 raw file SHA、canonical Case digest、EvalInput digest 和 manifest/source hash；
- 不复制一份只改目录名的 Case；共享逻辑 task 只有一个当前版本；
- Regression 出现产品退化时保持在 Regression 并报告失败，不得为了让门禁变绿而降回 Capability；只有任务语义、产品承诺或数据资格真实改变并经 adjudication 时才能迁出。

### 10.4 版本何时递增

以下任一变化必须递增 `case_version`：

- fixture 任一 byte、base/head revision 或 tree identity；
- ReviewRequest 或 repository descriptor；
- Intent claims、authority、scorability、clarification policy/script；
- Finding claim、severity/category/required/location/anchor/context/rationale；
- completeness、novel policy、CaseSource 或 Suite placement；
- annotation correction，即使结论看似“只是文字更清楚”但 canonical truth 已改变。

Manifest membership、split、protocol、dimensions、Case binding、Suite source 或 gate policy变化必须递增 `suite_version`。不得修改已发布版本的 bytes 后沿用旧 version/digest。

## 11. 接受前 Checklist

### 11.1 Eligibility / provenance

- [ ] Python task 小型、现实、可在固定 base/head 重放。
- [ ] fixture 只有完整 `base/`、`head/`，无禁止节点或私有数据。
- [ ] builder 生成的 revision/tree identity 已复核，未手写或沿用旧值。
- [ ] `CaseSource` / `SuiteSource` / license / content hash 真实且一致。
- [ ] ReviewRequest 可读、现实，不含解题提示。

### 11.2 Intent

- [ ] `scorable` 有独立依据；false 时使用唯一 canonical 空 truth。
- [ ] authority 使用 exact enum，全部 required claims 由该 authority 支持。
- [ ] claims 原子、dimension 正确、required 决策有依据。
- [ ] forbidden claims 是具体错误解释，并有 rationale。
- [ ] 已区分 truth authority、Submission claim source 和 Submission status。
- [ ] clarification policy 只由 materiality 决定。
- [ ] Script answer 不重复、不歧义，action/response/corrected values 合法。
- [ ] confirm/correct/reject 后的最终 claim/source/status 预期已人工检查。

### 11.3 Review truth

- [ ] 每个 expected Finding 事实成立、归因当前变更、触发和影响明确。
- [ ] compound 已拆分，duplicate 已合并，truth ID 全局唯一。
- [ ] 每个 Finding 已做 base/head pre-existing comparison。
- [ ] severity/category/required/context level 按统一口径标注。
- [ ] Core location 要么完整 path+side+range，要么 locations 为空。
- [ ] rationale 能让第三人独立复核。
- [ ] clean Case 完整审查后才保持空 expected list。
- [ ] known-invalid 只包含具体、可证伪 trap，不与 expected claim 冲突。
- [ ] completeness 与 novel policy 组合合法。

### 11.4 Evidence

- [ ] anchors 只描述 material fact，不规定唯一 Evidence/工具。
- [ ] 每个 anchor 原子且必要；不必要时列表为空。
- [ ] golden valid Evidence 由 replay 生成，revision/path/line/hash/excerpt/attestation exact。
- [ ] bad Evidence golden 每次只故意破坏一个待测维度。
- [ ] issue match 与 Evidence integrity/support 没有混写。

### 11.5 Audit / release

- [ ] Author A 与盲化 Reviewer B 独立完成标注。
- [ ] material disagreement 已由 Adjudicator C 裁决并记录。
- [ ] Agent/Judge 输出未用于首次 truth。
- [ ] truth leakage 自动和人工审计均通过。
- [ ] Case hydration、manifest hash、fixture reproducibility 和 golden tests 通过。
- [ ] Capability/Regression placement 满足第 10 节，版本和所有 digest 已更新。
- [ ] 人工审阅记录可追溯，但不在 Agent workspace。

任一必选项未完成，Case 状态是“未准入”，不能以 `unknown`、Capability placement 或临时 waiver 代替。

## 12. Accepted / rejected 示例

### 12.1 明确 Intent

Agent-facing input：

```yaml
review_request:
  user_intent: "Keep the legacy /v1/users endpoint working while adding /v2/users."
```

Accepted：

```yaml
intent_truth:
  scorable: true
  authority: explicit_author_metadata
  expected_claims:
    - truth_id: intent-001
      dimension: goal
      text: "Add the /v2/users endpoint."
      required: true
    - truth_id: intent-002
      dimension: constraint
      text: "The legacy /v1/users endpoint must remain compatible."
      required: true
  forbidden_claims:
    - truth_id: forbidden-intent-001
      dimension: scope
      text: "Remove the /v1/users endpoint."
      rationale: "The explicit user_intent requires the legacy endpoint to remain."
  clarification_policy: not_required
```

Rejected：一个 claim 写成“新增 v2、保留 v1、更新文档并提高性能”，因为四部分可分别判断；或把相同输入标成 inferred。

### 12.2 Inferred claim 经用户确认

Agent-facing input 未声明重试是否必须幂等，diff 暗示可能重放请求。Accepted Script：

```yaml
intent_truth:
  scorable: true
  authority: synthetic
  expected_claims:
    - truth_id: intent-001
      dimension: constraint
      text: "Retrying the operation must not create a second charge."
      required: true
  forbidden_claims: []
  clarification_policy: required

clarification_script:
  max_rounds: 1
  answers:
    - answer_id: answer-001
      dimension: constraint
      material_claim: "Retries must remain idempotent."
      action: confirm
      response: "Yes."
      corrected_values: []
```

合格 Agent 行为：提问时该候选为 `inferred`，提供 proposed value；确认后最终 claim 为 `explicit`，且 status 在没有其他 material gap 时为 `sufficient`。

Rejected：在未询问时直接把该 claim 标成 `explicit/sufficient`；或在 Case 中增加不存在的 `expected_source: inferred` 字段。

### 12.3 用户纠正

Accepted：

```yaml
- answer_id: answer-scope-001
  dimension: scope
  material_claim: "The migration should rewrite all historical rows."
  action: correct
  response: "Only rows read after deployment should be migrated lazily."
  corrected_values:
    - "Migrate historical rows lazily when they are read after deployment."
```

最终 Intent 必须包含 corrected value，原 material claim 必须消失，corrected claim 的 source 为 `explicit`。

Rejected：同时保留“全量重写”和“懒迁移”为 active claims，或把 correction 只放在 `uncertainties` 而不更新 Intent。

### 12.4 原子 security Finding

Accepted：

```yaml
- truth_id: issue-001
  claim: "The change removes the is_admin guard, so an authenticated non-admin can update another user's account."
  severity: high
  category: security
  required: true
  locations:
    - path: app/auth.py
      side: left
      from_line: 42
      to_line: 43
  evidence_anchors:
    - fact: "The base guard rejects actors whose is_admin flag is false, and head no longer executes an equivalent guard."
      locations:
        - path: app/auth.py
          side: left
          from_line: 42
          to_line: 43
  required_context_level: diff
  rationale: "The deleted guard is the only authorization check on this path; the route remains reachable by any authenticated actor. This is a merge-blocking authorization regression."
```

Rejected：

> 权限坏了、测试不够、日志也不清楚，可能导致各种安全问题。

它混合多个问题，没有具体触发链和可分配修复。

### 12.5 Duplicate 与多个后果

Accepted：把以下两句合成一个 truth Finding，因为根因和修复相同：

- “删除 guard”；
- “非管理员可以访问”。

Rejected：为两句分别创建 `issue-001` 和 `issue-002`，再期待一条 Agent Finding 获得两次 Recall。

Accepted split：

- `issue-001`：权限 guard 被删除；
- `issue-002`：项目明确要求的 non-admin regression test 被移除，且该测试缺口可独立修复和验收。

### 12.6 Pre-existing trap

base 与 head 都已有 `verify_tls=False`，head 只改日志格式。

Accepted：

```yaml
expected_findings: []
known_invalid_findings:
  - truth_id: invalid-001
    claim: "This change disables TLS verification."
    category: security
    locations:
      - path: client.py
        side: right
        from_line: 18
        to_line: 18
    rationale: "TLS verification was already disabled at the exact base revision; head does not change or expand that behavior."
```

Rejected：把它标成 expected high Finding，只因为问题本身严重。

### 12.7 Clean PR

一个只改变局部变量名、AST/输出/异常行为均不变的受控重构，经完整审查后：

```yaml
review_truth:
  completeness: closed_world
  novel_finding_policy: forbid
  expected_findings: []
  known_invalid_findings:
    - truth_id: invalid-001
      claim: "The rename changes the serialized response field name."
      category: regression
      locations: []
      rationale: "The serialized key is a string literal unchanged in base and head; only the local variable was renamed."
```

Rejected：省略该 Case，因为“没有评论可评分”；或把所有 unmatched Agent Finding直接称为 fabricated。

### 12.8 Context level

- Accepted `diff`：删除的 guard 和未受保护调用都在同一 hunk，影响无需额外代码即可证明。
- Accepted `file`：diff 只改默认值，必须读同文件未修改分支才能确认异常路径。
- Accepted `repo`：diff 看似安全，但另一文件的 caller 传入未经验证的用户输入，跨文件后才能证明 injection。
- Rejected `diff`：仅因为 bug 行在 diff 中；完整 claim 实际依赖跨文件 caller。

### 12.9 Evidence anchor

Accepted：

```yaml
evidence_anchors:
  - fact: "head calls write_record before checking whether validation succeeded."
    locations:
      - path: service.py
        side: right
        from_line: 71
        to_line: 76
```

Rejected：

```yaml
evidence_anchors:
  - fact: "The Agent must run rg and quote exactly service.py:73."
```

后者规定了工具和唯一位置，不是必须证明的事实。

### 12.10 Severity

- Accepted `high`：已认证普通用户可修改其他账户；重要授权边界被破坏。
- Accepted `critical`：未认证网络请求可任意接管所有管理员账户，默认部署无 mitigation。
- Rejected `critical`：只有测试环境中的 debug message 拼写错误；不能因修复简单或 Case 想测 high/critical miss 而抬高严重度。

## 13. 当前 v1 schema 的表达边界

以下是标注时必须知道的现状，不得通过给 canonical JSON 添加未知 key 绕过：

1. `IntentTruth.authority` 是 Case 级单值，不能表达每个 expected claim 的混合 authority。
2. `ExpectedIntentClaim` 没有 `source` 或 `rationale`；`IntentClaimSource` 只存在于 Agent Submission。
3. `IntentTruth` 没有预期 `IntentResult/status`；status 只存在于 Agent Submission。
4. `EvalCase` / `SuiteCase` 没有显式 base/head tree digest 字段；repository revision 会绑定 tree，preparer 也会计算 tree identity，但 Task 13 的显式 tree-digest 审计目前只能留在 fixture build/review record。
5. Canonical schema 没有 annotator、blind-review、adjudication 或 annotation-protocol-version 字段；人工审阅记录目前依赖受审 PR 的外部审计证据。
6. `ExpectedFinding.category` 是任意非空 identifier，schema 不会强制 Core 的 `security/regression/correctness` taxonomy。
7. `SuiteManifest` 有 `CaseSplit`，但没有版本化 Suite gate/expected-stability policy 字段；Capability/Regression promotion 的自动 policy binding 要等后续 gate schema。
8. `CaseSource.content_hash` 只校验 SHA-256 形状，当前 Loader 不验证 Core hand-authored provenance packet 的具体组成。
9. `TruthLocation` 为公共数据保真允许 null side/line；schema 不会自动强制本协议对 Core location 的完整性要求。

这些边界不降低本协议的人工准入标准。依赖缺失字段才能可靠评分的 Case必须暂停准入，直到 canonical schema 扩展并版本化；不得把关键信息藏在 `dimensions`、文件名或 rationale 文本中冒充机器约束。

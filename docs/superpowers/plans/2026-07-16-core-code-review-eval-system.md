# Core Code Review Eval System 实施计划

**状态：** 执行中（Task 1 v1 协议已冻结）

**设计来源：** `docs/superpowers/specs/2026-07-16-core-code-review-eval-system-design.md`

**目标：** 按已确认设计实现一套面向最终产品形态的黑盒 Code Review Eval System。系统只评测 Intent 是否理解正确、Review 是否正确，以 AACR-Bench 的数据集—Agent—Matcher—Metrics 工业骨架为主线，落实 Anthropic 关于 task/trial/grader/harness、环境隔离、重复 Trial、失败 transcript 和人类校准的方法，并接入本项目特有的 Intent Eval。

**执行方式：** 建议 Subagent-Driven。按下述 Wave 和文件所有权并行，主线程负责共享协议、跨模块集成、设计一致性审查、全量回归和最终提交。这里的 Wave 是最终架构的依赖拆分，不允许引入之后会被替换的临时 schema、临时 matcher、简化 Runner 或仅服务演示的数据格式。

**技术栈：** Python frozen dataclasses/enums、stdlib JSON/Git/subprocess/hashlib/statistics、现有统一 Model Adapter、pytest；公共 Parquet 数据读取仅放在可选 `eval-public` 依赖中，核心 Harness 不依赖数据科学框架。

**最终协议：** `eval_input_v1`、`eval_submission_v1`、`eval_case_v1`、`eval_run_config_v1`、`eval_run_manifest_v1`、结构化 Judge 输出 v1、Suite Manifest v1 和 Run Artifact v1。所有批次从第一天使用这些最终协议和严格 hydration，不建立以后迁移掉的 v0 格式。

---

## 1. 全局不变量

- Eval Harness 只通过 `EvalInput + bounded ClarificationChannel -> AgentUnderTestAdapter -> EvalSubmission` 观察被测 Agent；除当前产品 Adapter 的输出转换外，Evaluator 不依赖 Agent 的 Runtime、Session、Memory、Reviewer 数量、Context 组装或工具调用顺序。完整 Clarification Script 只在 Harness 侧，不能进入 Agent-facing EvalInput。
- Prompt、模型、Provider、Risk 策略、Reviewer 策略和 Memory 配置只记录为不透明 Run Configuration，用于复现和分组，不形成独立产品得分。
- Ground truth 和完整 Clarification Script 与 Agent workspace 必须物理隔离；Case 路径、环境变量、文件名、提交信息和临时目录都不得泄漏答案。
- 每个 Trial 都从独立、干净、固定 base/head 的工作区开始；一个 Trial 创建的文件、Git 状态和产品持久状态不得影响另一个 Trial。
- `prepare`、`run-agent` 与 `evaluate` 必须解耦。更换 Judge、修复 matcher 或调整 rubric 时可以重评已有 Submission，不重新调用 Agent。
- 每个 Trial 无论成功、失败、超时、阻塞还是输出不可解析，都必须生成一个终态 `EvalSubmission`；没有 Finding 是合法的零 Finding 输出，不得静默跳过 Case。
- Ground-truth Finding 必须原子化标注；不要求唯一标准 Evidence。可选 `evidence_anchors` 只描述必须证明的事实，不要求 Agent 引用相同位置。
- Finding 问题命中和 Evidence 质量正交保存。`issue_match` 不得因为 Evidence 路径、行号或 hash 错误而被自动改判；严格可发布 Finding 才要求三项同时通过。
- Review 匹配使用顺序无关的全局一对一最大权重分配。一个 Agent Finding 最多命中一个 truth issue，一个 truth issue 最多增加一次 Recall；重复和 compound Finding 不能重复赚取 Recall。
- 确定性规则优先。Schema、ID、revision、path、line、hash、exact known-invalid、指标计算和分配算法不交给 LLM。
- LLM Judge 只处理语义等价、novel Finding factuality、Evidence support 和必要的 severity/actionability 判断；所有输出严格结构化、可审计、可重放并 fail closed。
- Judge 失败、超时、格式错误、非法 ID 或不充分信息必须保留为 `judge_failed`、`ungraded` 或 `unknown`，不得默认变成 confirmed、plausible 或 fabricated。
- `closed_world`、`expert_augmented`、`human_observed` 分开汇总和比较；不得把不同 ground-truth 完整度混成一个 Precision 排行榜。
- Case、Suite、数据源、Prompt、Agent、Judge、rubric 和 Run Config 全部记录版本与 digest；比较命令拒绝不可兼容的 Run。
- Trial 默认离线。只有 `prepare` 可在用户明确授权时访问外部数据源；正式 Trial 不下载依赖、不访问 PR、不发布评论、不 Approve、不 Merge。
- `.eval-data/` 和 `.eval-runs/` 不进入 Git；可提交的小型 Core fixture 不包含私有代码、凭证或 Ground Truth 泄漏路径。
- 旧 Eval 设计稿继续保留为历史记录，不在实现中增量修补；已确认的新设计稿是唯一实现来源。
- 不清理、不暂存既有 `.intent-*`、`.p-*`、`.pytest-*`、`.tmp`、`__pycache__` 等用户工作区内容。

## 2. 执行与提交规则

- 每个 Task 先写失败测试，再实现该 Task 的最终行为，再运行定向测试。
- 共享 schema、enum、稳定 ID、状态语义和 artifact 路径只能由 Task 1/3 固定；后续 Task 不自行创建同义字段或旁路 JSON。
- 并行子 Agent 的写入文件必须互斥。需要修改共享文件时，先由主线程提交接口骨架或安排到后续串行 Wave。
- 每个 Task 形成独立提交候选；通过 spec compliance 与 code quality review 后再集成。
- 一个 Wave 的定向测试与接口审计全部通过后，才能开始依赖它的下一 Wave。
- 公共 benchmark Adapter 必须用本地缩小 fixture 测试同一解析路径；不得为测试写另一套 parser。
- Judge 测试使用 scripted/fake Model Adapter，不调用真实远端模型。真实 Judge smoke test 由用户明确提供凭证时单独运行，不作为本地单测前置条件。
- Windows pytest 统一使用 `C:\tmp\rae-*` 短 basetemp；同一 Task 复用固定目录，确认无 pytest 进程后清理本项目目录，禁止为每次重跑创建永久保留的新目录。
- 未经用户明确要求，不 push、不创建 PR、不 merge 到 master。

### 实施开始门禁

- [x] 只暂存并提交已确认设计稿与本计划，排除所有既有临时目录。
- [x] 从包含设计和计划的提交创建 `codex/core-code-review-eval-system` 实现分支。
- [x] 记录基线 HEAD、`git status --short --untracked-files=no`、Git 版本和 Python 版本。
- [x] 使用独立短路径 basetemp 运行全量 pytest；真实失败必须先解释或修复，cleanup warning 不能代替退出码。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest -q -p no:cacheprovider --basetemp 'C:\tmp\rae-b'
```

实施基线：HEAD `ac87db4ab75b34ed73a5849de612e33f598c4742`，分支 `codex/core-code-review-eval-system`，Git `2.50.1.windows.1`，Python `3.9.23`。2026-07-16 使用短路径 `C:\tmp\rb` 完成全量 pytest，退出码为 0。先前长 basetemp 的 7 个 Memory CLI identity 失败已在短路径逐项复现为通过，确认为 Windows 路径环境问题而非代码回归。

## 3. 最终包结构

```text
src/review_agent_eval/
├── __init__.py
├── __main__.py
├── models.py
├── cases.py
├── datasets.py
├── config.py
├── artifacts.py
├── repository.py
├── submission.py
├── clarification.py
├── runner.py
├── evidence_checker.py
├── match_location.py
├── assignment.py
├── intent_evaluator.py
├── review_evaluator.py
├── judge.py
├── metrics.py
├── report.py
├── comparison.py
├── calibration.py
├── gates.py
├── cli.py
└── adapters/
    ├── __init__.py
    ├── base.py
    ├── current_agent.py
    ├── subprocess_agent.py
    ├── aacr_bench.py
    └── swe_prbench.py
```

核心 Eval package 不导入 `review_agent.pipeline`、Session 或 Memory。只有 `adapters/current_agent.py` 可以了解当前产品的 CLI/artifact 形状并把它翻译成公开 `EvalSubmission`；Evaluator、Metrics 和公共数据 Adapter 不得越过这条依赖边界。

## 4. 依赖图

```text
Wave A1
  Task 1 Canonical protocols
        |
        ├──────────────────────┐
Wave A2 v                      v
  Task 2 Case/Suite IO      Task 3 Run config/artifact store
        └──────────┬───────────┘
                   v
Wave A3       Task 4 Repository preparation/isolation
                   |
                   v
Wave B1       Task 5 Agent adapters/submission extraction
                   |
                   v
Wave B2       Task 6 Trial runner/always-submission lifecycle
                   |
        ┌──────────┴───────────┐
Wave C1 v                      v
  Task 7 Location/Evidence   Task 8 Assignment/Intent evaluator
        └──────────┬───────────┘
                   v
Wave C2       Task 9 Structured semantic Judge
                   |
                   v
Wave C3       Task 10 Review evaluator/reconciliation
                   |
        ┌──────────┴───────────┐
Wave D1 v                      v
  Task 11 Metrics/report    Task 12 Harness CLI
        └──────────┬───────────┘
                   v
        ┌──────────┴───────────┐
Wave D2 v                      v
  Task 13 Core suite        Task 14 Public dataset adapters
        └──────────┬───────────┘
                   v
Wave E1       Task 15 Repeated trials/compare/calibrate/gates
                   |
                   v
Wave E2       Task 16 E2E/security/compatibility/docs
```

---

# Batch A：最终协议、Case Bank 与隔离基础设施

## Wave A1：Canonical Eval Protocols

### Task 1：EvalInput、EvalSubmission、EvalCase 与严格 hydration

**依赖：** 无。

**所有权：**

- 新建 `src/review_agent_eval/__init__.py`
- 新建 `src/review_agent_eval/models.py`
- 新建 `tests/eval/test_models.py`
- 新建 `tests/eval/test_schema_hydration.py`

**RED 测试：**

- [ ] `EvalInput v1` 只严格 round-trip repository 和 review request，不包含 clarification policy/answers；EvalCase 单独严格 round-trip 带 `max_rounds`/typed answers 的私有 clarification script。
- [ ] `EvalSubmission v1` 对 completed/failed/blocked/invalid_output 都可构造；每个终态的 intent/review/failure 必填与 null 组合明确，零 Finding 合法。
- [ ] 每个终态 Trial 恰有一个 Submission；pending/running/incomplete 是可恢复非终态；blocked clarification 必须保留可评分 Intent transcript，invalid_output 不伪造部分 Outcome。
- [ ] Intent claim 只接受四个 dimension 与 `explicit|inferred` source；`inferred` 不被 hydration 自动改成 explicit。
- [ ] clarification transcript 保留连续 turn index、material claim、匹配 answer、action、response 和 resolved values；答案耗尽/未匹配不被 Harness 猜测补齐。
- [ ] clarification action/null 组合、matched answer 引用、confirm/correct/reject/skip/defer 的 response/resolved-values 约束严格 round-trip。
- [ ] Finding、typed Evidence、location、uncertainty、usage 与 cost 的 null/缺失语义不同，缺字段不被空字符串伪装。
- [ ] Intent/Review uncertainties 是 bounded non-empty string lists；Usage 的 elapsed/cost 为 finite non-negative number，token/tool fields 为 non-negative int，token total 与 cost currency 组合受跨字段验证。
- [ ] `EvalCase v1` 严格区分 input 与 truth；intent truth 可 `scorable=false`；review completeness 只接受三种设计值。
- [ ] expected/forbidden Intent、expected/known-invalid Finding 使用不同 typed leaf；truth ID 唯一，location/evidence anchor 可多条，known-invalid 不与 expected Finding 使用同一 ID。
- [ ] closed-world 支持 `verify|forbid` novel Finding policy；expert-augmented/human-observed 拒绝 `forbid`。
- [ ] 所有 ID、集合排序、canonical JSON 和完整 SHA-256 digest 稳定；语义重复 Finding、重复 evidence ref 和 clarification 时序不会被 set 去重擦除。
- [ ] 超长 claim、excerpt、Case、Finding 数量和 Evidence 数量 fail closed，防止 benchmark 或 Agent 输出无界占用内存。
- [ ] 未知 schema/version/enum、递归重复 JSON key、NaN/Infinity/`1e999`、bool 冒充 int 全部拒绝。
- [ ] dangling Evidence ref 结构上可 hydration 并留给 Evidence Checker 判 missing；非法 Evidence 对象本身仍拒绝。
- [ ] duplicate Finding/Evidence object ID 是 schema error；重复或 dangling ref 保留；有界但未授权 revision/path/line/hash 内容进入 missing/invalid grader，不把问题 Finding 整体抹掉。

**实现：**

- [ ] 定义 frozen domain models 与 enums：trial/submission status、failure code、clarification action、intent dimension/result、truth completeness、novel policy、issue judgement、Evidence integrity/support、Judge status。
- [ ] 实现唯一 canonical `to_dict/from_dict`、JSON duplicate-key rejection、长度/数量上限和 digest helper。
- [ ] 模型只保存 JSON-ready 基础值，不保存 `Path`、subprocess、Provider response、产品 Session 或 Runtime 类型。
- [ ] 为输入、提交、Case、truth Finding、Evidence、clarification answer 定义稳定 ID 规则。
- [ ] 实现 typed `SubmissionFailure`、`SubmissionClarificationExchange`、`ForbiddenIntentClaim`、`KnownInvalidFinding` 和 annotation rationale；completed 与非 completed 终态做跨字段验证。
- [ ] 集中定义已确认设计中的 v1 字节/字符/数量上限；先检查原始 collection 数量，再 canonicalize。
- [ ] 把设计中的 YAML 作为说明格式；实现与 artifact 使用 canonical UTF-8 JSON，避免新增核心 YAML 依赖。
- [ ] 定义 `repository_file/repository_diff/command_output/external_record` Evidence 的 exact keys；hydration 保留有界但未授权的 revision/path 给 Checker，Checker 才按 kind 验证精确 base/head/range 与 source attestation。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_models.py tests/eval/test_schema_hydration.py -q -p no:cacheprovider --basetemp 'C:\tmp\rae-t1'
```

**提交边界：** `feat(eval): add canonical input submission and case protocols`

## Wave A2：Case 与 Artifact 并行基础

### Task 2：Case Loader、Suite Manifest、版本与 Ground Truth 隔离

**依赖：** Task 1。

**所有权：**

- 新建 `src/review_agent_eval/cases.py`
- 新建 `src/review_agent_eval/datasets.py`
- 新建 `tests/eval/test_cases.py`
- 新建 `tests/eval/test_datasets.py`

**RED 测试：**

- [ ] Suite Manifest 固定 suite ID/version、Case 列表、split、source metadata、license、content hash 和 truth completeness。
- [ ] Loader 在使用 Case 前重新计算文件 hash 与 canonical Case digest，篡改、重复 task ID 和缺失 Case 均失败。
- [ ] Agent-facing loader 只能返回 `EvalInput`；truth 和完整 Clarification Script 只能由 evaluator/Runner-facing API 读取，类型层面不共享一个包含答案的对象。
- [ ] Case path 必须留在 suite root；拒绝 absolute path、`..`、symlink/reparse-point escape 和大小写碰撞。
- [ ] intent authority、required claims、forbidden claims 与 clarification policy 组合合法性受校验。
- [ ] closed-world Case 可以禁止 novel Finding；human-observed Case 不允许把 unmatched 自动标成 fabricated。
- [ ] fixed train/dev/capability/regression/held-out split 不能在一次 Run 内重映射。
- [ ] manifest/source/version/license/hash 缺失时公共数据 Case 不可运行。

**实现：**

- [ ] 实现 immutable `CaseBank`、`SuiteManifest`、`CaseHandle` 和只暴露输入的 `AgentCaseView`。
- [ ] ground truth 和完整 Clarification Script 保留在 Harness 控制目录；Repository Preparer 只接收 repository descriptor，不接收 truth 或答案 payload。
- [ ] 生成运行所需的 Case manifest snapshot，确保后续源文件变化不会静默改变已完成 Run。
- [ ] 支持 Core/public/private suite 元数据，但不在核心 Loader 内写特定数据集分支。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_cases.py tests/eval/test_datasets.py -q -p no:cacheprovider --basetemp 'C:\tmp\rae-t2'
```

**提交边界：** `feat(eval): add versioned case bank and truth isolation`

### Task 3：Run Config、Artifact Store 与可恢复 Manifest

**依赖：** Task 1；可与 Task 2 并行。

**所有权：**

- 新建 `src/review_agent_eval/config.py`
- 新建 `src/review_agent_eval/artifacts.py`
- 新建 `tests/eval/test_config.py`
- 新建 `tests/eval/test_artifacts.py`

**RED 测试：**

- [ ] Run Config 记录 Agent/commit/model/provider/参数/config digest、Suite/Case digest、trial count、Judge/rubric 版本和资源预算。
- [ ] API key 值、认证 URL userinfo、完整环境变量和隐藏 reasoning 不能进入 Config、Manifest、错误或报告。
- [ ] Run ID、trial ID、路径和 manifest digest 稳定且防 traversal。
- [ ] 每个 artifact 使用 UTF-8 canonical JSON、内容 hash、原子写与 fsync；已完成 artifact 不被静默覆盖。
- [ ] interrupted trial 可被识别为 incomplete，resume 只补齐缺失的合法阶段，不重写已有 Submission。
- [ ] incomplete 是非终态且没有 terminal Submission；恢复成功后写 completed，放弃恢复时原子最终化为 failed/process_killed 或其他稳定 failure code。
- [ ] 并行 Trial 写入互不覆盖；同一 run/case/trial 的冲突 writer fail closed。
- [ ] `judge_input/output`、matches、score 和 report 可以在不修改原 submission 的情况下版本化重算。
- [ ] artifact 读取有单文件/总大小上限，拒绝 symlink、special file 和 hash 不匹配。

**实现：**

- [ ] 实现 `.eval-runs/<run-id>/` 的最终目录布局、Run/Trial manifest、stage receipts 和 atomic writer。
- [ ] 将 Agent execution 与 evaluator execution 记为不同阶段与配置 digest。
- [ ] 保留可选不透明 `trace_ref`，但不复制未授权 secret/raw reasoning。
- [ ] 实现只读 `load_existing_submission` 与 re-evaluation output namespace。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_config.py tests/eval/test_artifacts.py -q -p no:cacheprovider --basetemp 'C:\tmp\rae-t3'
```

**提交边界：** `feat(eval): add immutable run artifacts and manifests`

## Wave A3：Repository Preparation 与 Trial Isolation

### Task 4：Repository Preparer、Fixture Builder 与干净工作区

**依赖：** Task 2、Task 3。

**所有权：**

- 新建 `src/review_agent_eval/repository.py`
- 新建 `tests/eval/test_repository.py`
- 新建 `tests/eval/fixtures/repositories/README.md`

**RED 测试：**

- [ ] local fixture、local Git cache 和显式 remote source 都解析到精确 full base/head SHA；不存在、非 commit 或相同错误 binding 失败。
- [ ] fixture source tree 使用固定作者/时间/排序创建可复现 commits，并校验 base/head tree digest。
- [ ] 每个 Trial 获得独立 writable checkout；base object 可读、HEAD 固定，另一个 Trial 的未跟踪文件和产品目录不可见。
- [ ] dirty source repository 不污染 Trial；submodule、LFS、symlink 和 nested repository 使用显式 policy，不隐式访问网络。
- [ ] truth 文件、suite manifest、held-out 标注和 evaluator config 永不复制到 Agent workspace。
- [ ] cleanup 失败形成诊断但不篡改 Trial 成绩；保留 workspace 的 debug policy 明确且默认有界。
- [ ] remote prepare 校验 URL 脱敏、commit allowlist、content hash 和 license；trial phase 断网也可运行。
- [ ] Windows path/case/reparse point 与 Unix symlink 路径均不能逃逸 eval root。

**实现：**

- [ ] 实现 `PreparedRepository`、`TrialWorkspace` 和 `RepositoryPreparer` context manager。
- [ ] 外部 source 只在 `prepare` 阶段进入 `.eval-data/` 内容寻址 cache；Trial 只读取固定对象。
- [ ] Core fixture builder 从提交的 base/head trees 创建确定性 Git 仓库，再产生正常 `EvalInput` full SHA。
- [ ] workspace manifest 记录 source digest、base/head SHA、Git version 和隔离策略，不记录凭证。
- [ ] 显式清理与保留策略都先校验 resolved absolute path 位于 eval workspace root。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_repository.py -q -p no:cacheprovider --basetemp 'C:\tmp\rae-t4'
```

**提交边界：** `feat(eval): isolate reproducible repository trials`

---

# Batch B：黑盒 Agent Runner 与当前产品 Adapter

## Wave B1：Adapter 与 Submission

### Task 5：AgentUnderTestAdapter、Clarification Script 与当前产品输出转换

**依赖：** Task 1-4。

**所有权：**

- 新建 `src/review_agent_eval/adapters/__init__.py`
- 新建 `src/review_agent_eval/adapters/base.py`
- 新建 `src/review_agent_eval/adapters/subprocess_agent.py`
- 新建 `src/review_agent_eval/adapters/current_agent.py`
- 新建 `src/review_agent_eval/submission.py`
- 新建 `src/review_agent_eval/clarification.py`
- 新建 `tests/eval/test_agent_adapter.py`
- 新建 `tests/eval/test_current_agent_adapter.py`
- 新建 `tests/eval/test_clarification_script.py`

**RED 测试：**

- [ ] Adapter 接口只接收 EvalInput、workspace、AgentRunConfig 和能力受限的 `ClarificationChannel`，只返回 EvalSubmission/trace descriptor，不接收 EvalCase truth。
- [ ] `ClarificationChannel` 允许提交一个实际问题并取得至多一个匹配回答，但不能读取 policy、完整答案表或剩余答案。
- [ ] generic subprocess Adapter 使用参数数组而非 shell string，环境变量 allowlist、cwd、timeout、stdout/stderr 大小和终止语义受控。
- [ ] clarification answers 按 material claim/action 消费；多问、少问、问题次序变化、错误 field 和答案耗尽都形成可评分 transcript，而不是 Harness 猜答案。
- [ ] 当前 Agent Adapter 只从最终 `review_brief.json`、受校验 Observation artifacts 和公开终态提取 Intent/Findings/Evidence/Uncertainties。
- [ ] explicit/inferred provenance、clarification history 和 unresolved question 被无损映射；Adapter 不自行把 inferred 改成 explicit。
- [ ] Brief Finding 的 path/line/severity/evidence refs 稳定映射；缺失字段保留 null，不从 claim 文本猜行号。
- [ ] Observation ID、raw artifact hash、revision binding 与 Finding refs 交叉校验；未引用或越权 Observation 不被补成 Agent Evidence。
- [ ] `base@sha`、`head@sha` 和精确 diff-range 规范化为 Eval Evidence binding；无法重放的产品证据保留为可诊断的 invalid/missing，不由 Adapter伪造新的文件引用。
- [ ] completed、awaiting clarification、failed、invalid artifact 和零 Finding 都产生合法 Submission。
- [ ] Adapter 不读取 Memory Store、Session 内部计划或 Reviewer trace 为 Agent 增加它未输出的 Finding；只可为 Agent 已引用的 command artifact 生成可审计 attestation，或把已引用搜索/符号结果规范化为其实际 file/diff source。

**实现：**

- [ ] 定义 `AgentUnderTestAdapter` Protocol、`AgentRunConfig` 和稳定 error taxonomy。
- [ ] generic subprocess Adapter 支持任意外部 Agent 的 JSON 输入/输出约定，具体命令模板由 Run Config 固定。
- [ ] 当前产品 Adapter 黑盒调用正式 `review-agent review` 入口；每个 Trial 使用隔离 run/memory root，并仅把 `ClarificationChannel` 当前返回的单个回答写入 stdin。
- [ ] 解析当前产品最终 artifact，生成 canonical EvalSubmission；保留 `trace_ref` 指向有界 Trial trace。
- [ ] usage 只填写产品真实提供的数据；未知 token/cost 为 null，不估算或捏造。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_agent_adapter.py tests/eval/test_current_agent_adapter.py tests/eval/test_clarification_script.py -q -p no:cacheprovider --basetemp 'C:\tmp\rae-t5'
```

**提交边界：** `feat(eval): adapt black box agents into canonical submissions`

## Wave B2：Trial Lifecycle

### Task 6：Agent Runner、失败分类与 Always-Submission

**依赖：** Task 5。

**所有权：**

- 新建 `src/review_agent_eval/runner.py`
- 新建 `tests/eval/test_runner.py`
- 新建 `tests/eval/test_runner_failures.py`

**RED 测试：**

- [ ] Runner 严格执行 load input -> prepare isolated workspace -> invoke Adapter -> validate Submission -> persist terminal artifacts。
- [ ] 正常零 Finding 输出写 completed Submission；truth 中有 issue 时后续 Recall 为零，但 Runner 不改写状态。
- [ ] timeout、non-zero exit、killed process、output overflow、invalid JSON、schema mismatch、blocked 和 Adapter exception 各有稳定状态/错误码。
- [ ] 所有失败路径都写 submission、stdout/stderr 摘要、timing 和 manifest terminal receipt；不因缺 comments 文件跳过 Case。
- [ ] terminal receipt 与 Submission 一一对应；pending/running/incomplete 不得被 evaluate 命令当成零 Finding 成功结果。
- [ ] resume 不重复运行已有 terminal Trial；只对没有 terminal receipt 且 policy 允许的 Trial 重试。
- [ ] 并行 Case/Trial 的 workspace、run artifact、端口、环境和产品状态相互隔离。
- [ ] 中断和 cancellation 终止完整子进程树，并保留可读诊断。
- [ ] Runner 不读取 truth、不计算 metric、不调用 Judge。

**实现：**

- [ ] 实现单 Trial 与有界 worker pool 调度、timeout/cancellation、terminal Submission factory。
- [ ] 每个 Trial 固定 config digest、Case digest、workspace binding 和 Adapter version。
- [ ] 将原始输出保存为有界 trace；报告只暴露脱敏摘要。
- [ ] 为后续重复 Trial 预留确定性 trial index/seed，但不把 seed 传给不支持它的 Agent 冒充可控随机性。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_runner.py tests/eval/test_runner_failures.py -q -p no:cacheprovider --basetemp 'C:\tmp\rae-t6'
```

**提交边界：** `feat(eval): run isolated trials with terminal submissions`

---

# Batch C：Intent、Finding 与 Evidence Graders

## Wave C1：并行确定性 Graders

### Task 7：Location Matcher 与 Evidence Integrity Checker

**依赖：** Task 1-6。

**所有权：**

- 新建 `src/review_agent_eval/match_location.py`
- 新建 `src/review_agent_eval/evidence_checker.py`
- 新建 `tests/eval/test_location_matcher.py`
- 新建 `tests/eval/test_evidence_checker.py`

**RED 测试：**

- [ ] path 使用规范 POSIX repo-relative 形式；wrong path、case collision、absolute/escape、deleted-side 错误都不会位置命中。
- [ ] side 与 line-range overlap/distance 产生确定性候选分数；缺位置不会自动成功，也不会阻止后续合法 root-cause 语义匹配。
- [ ] Evidence ID 必须存在且被 Finding 引用；missing、duplicate、dangling ref 明确分类。
- [ ] hydration 可安全保留有界 symbolic HEAD/branch 等坏值；Checker 只接受 Case 的 exact base/head/diff range，并把其他值 deterministic invalid。
- [ ] repository-file Evidence 从固定 Git object 重读 exact lines，按 UTF-8/LF 规范计算 excerpt/hash；path/line/hash/excerpt 任一不符为 deterministic invalid。
- [ ] repository-diff Evidence 只可对 exact base..head/path 重放固定完整 diff；不接受 Agent workspace 未提交内容。
- [ ] command-output 必须解析 Harness/Adapter attestation，external-record 必须解析 Agent 可见 existing-CI entry；缺 attestation/source 为 invalid。
- [ ] Evidence anchor 不要求相同位置；它只作为后续 support Judge 的事实提示，不参与 integrity 伪造通过。
- [ ] 问题命中但 Evidence 行号错误时保留独立 `issue_match` 输入，Evidence 结果为 invalid/unsupported。

**实现：**

- [ ] 实现 location normalization、candidate generation 与可配置但版本化的 line-distance policy。
- [ ] 实现 `EvidenceIntegrityResult(valid|invalid|missing)`、稳定 reason code 和仓库重放读取。
- [ ] Integrity Checker 不调用 LLM、不判断 Finding 是否真实，只验证提交材料的身份和内容。
- [ ] 所有 repository read 有字节/行数/path 边界，并绑定 PreparedRepository 的 base/head。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_location_matcher.py tests/eval/test_evidence_checker.py -q -p no:cacheprovider --basetemp 'C:\tmp\rae-t7'
```

**提交边界：** `feat(eval): verify evidence integrity and location candidates`

### Task 8：全局一对一 Assignment 与 Intent Evaluator

**依赖：** Task 1-6；可与 Task 7 并行。

**所有权：**

- 新建 `src/review_agent_eval/assignment.py`
- 新建 `src/review_agent_eval/intent_evaluator.py`
- 新建 `tests/eval/test_assignment.py`
- 新建 `tests/eval/test_intent_evaluator.py`

**RED 测试：**

- [ ] 最大权重二分图分配得到全局最优而不是贪心结果；输入顺序、dict 顺序和同分候选不改变稳定结果。
- [ ] 一个 generated item 和一个 truth item 最多各匹配一次；重复 claim 不增加 Recall，额外重复项留在 unmatched。
- [ ] 不同 Intent dimension 不互相候选；显式 exact/normalized match 在无需 Judge 时确定性完成。
- [ ] supported、partially_supported、unsupported、contradicted、unknown 独立计数；`inferred` 本身不等于 unsupported。
- [ ] required truth、optional truth、forbidden claim 和 unscorable Intent 语义正确。
- [ ] clarification required/optional/not-required、是否提问、问题 materiality、答案消费和更新后 Intent 分开评分。
- [ ] 应问未问、不该问却阻塞、问错维度、得到答案未更新 Intent 都有稳定 reason code。
- [ ] semantic unresolved pair 只生成 Judge request，不由确定性层猜测语义结果。

**实现：**

- [ ] 使用 stdlib 实现精确多项式时间最大权重二分图分配与稳定 lexicographic tie-break，不新增运行时科学计算依赖。
- [ ] 实现 Intent normalization、候选矩阵、deterministic match、Judge request 和最终 claim assignment 合并。
- [ ] Intent evaluator 只读取 Submission.intent 与 intent truth/clarification transcript，不读取 Agent 内部 trace。
- [ ] 输出完整 claim-level matches、unmatched、clarification decisions 和 metric inputs。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_assignment.py tests/eval/test_intent_evaluator.py -q -p no:cacheprovider --basetemp 'C:\tmp\rae-t8'
```

**提交边界：** `feat(eval): score intent claims with global assignment`

## Wave C2：Structured Semantic Judge

### Task 9：统一 Model Adapter Judge、Rubric 与 Fail-closed 解析

**依赖：** Task 7、Task 8。

**所有权：**

- 新建 `src/review_agent_eval/judge.py`
- 新建 `tests/eval/test_judge.py`
- 新建 `tests/eval/test_judge_rubrics.py`

**RED 测试：**

- [ ] Judge 只通过现有 `ModelAdapter`/factory 调用模型，Eval business modules 不直接拼 OpenAI/DeepSeek/Claude HTTP。
- [ ] Intent equivalence、Finding equivalence/novel factuality 和 Evidence support 使用不同 response schema 与 rubric version。
- [ ] Judge 输入包含盲化后的 generated item、truth/anchor、必要 diff/代码上下文和 Evidence，不包含 Agent/model/baseline/candidate 身份。
- [ ] repository 文本被明确包裹为不可信数据，不能覆盖 Judge system/rubric 或请求额外工具权限。
- [ ] structured JSON 拒绝未知 key、非法 ID、越权 classification、缺 reason refs 和非有限 score。
- [ ] timeout、Provider error、invalid output、截断和上下文预算不足返回 judge_failed/ungraded/unknown，不默认分类。
- [ ] 同一 immutable Judge input + config digest 可缓存重用；rubric/model/config 变化必须产生不同 cache key。
- [ ] scripted Judge 覆盖同义改写、相邻但不同 issue、错误根因、compound Finding、真实 novel Finding、Evidence weak/unsupported。

**实现：**

- [ ] 定义 versioned Judge request/result、bounded context builder、严格 parser 和 error taxonomy。
- [ ] Judge 请求使用现有 `ModelTurnRequest` 且 `tool_choice=none`；不向 Judge 暴露 Agent 工具。
- [ ] 保存完整有界 judge_input/output、Provider/model/rubric digest、attempt 与失败状态。
- [ ] 支持固定重试次数，但每次 attempt 都可审计；重试耗尽仍 fail closed。
- [ ] semantic score 只表示 issue/claim 等价；Evidence integrity 不进入 issue matching 权重。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_judge.py tests/eval/test_judge_rubrics.py -q -p no:cacheprovider --basetemp 'C:\tmp\rae-t9'
```

**提交边界：** `feat(eval): add blind structured semantic judges`

## Wave C3：Review Evaluator

### Task 10：Finding Reconciliation、严格一对一匹配与 Evidence Support

**依赖：** Task 7-9。

**所有权：**

- 新建 `src/review_agent_eval/review_evaluator.py`
- 新建 `tests/eval/test_review_evaluator.py`
- 新建 `tests/eval/test_review_truth_completeness.py`

**RED 测试：**

- [ ] location 只生成候选/定位指标；同一根因在定义点和调用点标注时可语义命中。
- [ ] matching edge 只衡量根因、触发条件、影响和必要位置，不把 Evidence integrity/support 混入 issue score。
- [ ] 全局一对一分配优于贪心，重复 Finding 只命中一次，compound Finding 最多命中一个 truth issue。
- [ ] confirmed、plausible、fabricated、unknown 与 matched truth ID 独立保存；unmatched duplicate 不增加 Recall。
- [ ] known-invalid exact/semantic 命中为 fabricated；单纯 Evidence path/hash 错误不自动 fabricated。
- [ ] evaluator 先做 known-invalid、truth assignment、duplicate，再按 novel policy 分流；`forbid` 产生 `unknown + novel_disallowed`，不冒充 fabricated，只有 `verify` unmatched 进入 factuality Judge。
- [ ] closed-world、expert-augmented、human-observed 对 unmatched Finding 使用各自 policy；human-observed 不因未命中人类评论自动误报。
- [ ] Evidence support 在 integrity 之后独立判断 supported/weak/unsupported/unknown；invalid/missing 不可成为严格 publishable Finding。
- [ ] `confirmed + valid + supported` 是唯一 strict publishable 条件；问题正确但证据错误、证据正确但问题错误都不可发布。
- [ ] Judge 失败保留 ungraded，指标 denominator/coverage 显式，不吞掉 Case。

**实现：**

- [ ] 组合 deterministic candidate pairs、semantic Judge scores、global assignment 与 truth completeness policy。
- [ ] 只对 `novel_finding_policy=verify` 的 unmatched Finding 运行 bounded factuality Judge；对 matched/verified-novel Finding分别运行 Evidence support Judge。
- [ ] 输出每条 Finding 的 issue judgement、truth assignment、location、Evidence integrity/support、Judge refs 和 publishable 状态。
- [ ] 保存完整 matching matrix/selected edges/unmatched reasons，支持 `inspect` 解释。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_review_evaluator.py tests/eval/test_review_truth_completeness.py -q -p no:cacheprovider --basetemp 'C:\tmp\rae-t10'
```

**提交边界：** `feat(eval): reconcile review findings with strict evidence scoring`

---

# Batch D：Metrics、CLI 与真实 Case Sources

## Wave D1：Metrics/Report 与 Harness CLI

### Task 11：Metrics Aggregator、Case Report 与可解释 Inspect 数据

**依赖：** Task 8-10。

**所有权：**

- 新建 `src/review_agent_eval/metrics.py`
- 新建 `src/review_agent_eval/report.py`
- 新建 `tests/eval/test_metrics.py`
- 新建 `tests/eval/test_report.py`

**RED 测试：**

- [ ] Intent precision/recall、unsupported/contradicted rate、clarification accuracy 和 case pass 使用明确 denominator；零 denominator 为 null/not-scorable，不伪装 0 或 100%。
- [ ] issue precision/recall/F1、severity-weighted recall、critical/high misses、fabricated/plausible/unknown rate 与 per-PR count 公式正确。
- [ ] line precision/recall、Evidence validity/support rate 和 publishable Finding precision 从独立状态计算。
- [ ] failed/blocked/invalid Submission 进入 failure rate，并按 policy 对 outcome metric 计入；不得从汇总中消失。
- [ ] 不同 truth completeness、Suite、language、category、severity、context level、PR size 和 Agent config 分组，禁止不兼容汇总。
- [ ] report 展示分子/分母、coverage、ungraded/Judge failure、严重漏报和误报 Case，不只显示百分比。
- [ ] 时间/token/cost 只汇总真实提供值，missing coverage 单独报告。
- [ ] 不生成单一 Overall Score。

**实现：**

- [ ] 定义 versioned case/trial score 与 aggregate metric models、公式和 group-by engine。
- [ ] 生成 `summary.json` 与 `report.md`，首页聚焦 Intent、Review、Evidence、Failure 和成本。
- [ ] 生成 inspect 所需的 Case timeline、claim/finding assignments、Judge/Evidence reason refs。
- [ ] 对 incomplete ground truth 使用明确 metric label/annotation，避免与 closed-world leaderboard 混淆。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_metrics.py tests/eval/test_report.py -q -p no:cacheprovider --basetemp 'C:\tmp\rae-t11'
```

**提交边界：** `feat(eval): aggregate outcome metrics and explainable reports`

### Task 12：prepare/run-agent/evaluate/inspect CLI

**依赖：** Task 1-11；可与 Task 11 后半段串行集成。

**所有权：**

- 新建 `src/review_agent_eval/__main__.py`
- 新建 `src/review_agent_eval/cli.py`
- 修改 `pyproject.toml`
- 新建 `tests/eval/test_cli.py`
- 新建 `tests/eval/test_cli_failures.py`

**RED 测试：**

- [ ] 安装脚本为 `review-agent-eval`，不把 Eval 子命令塞进产品 `review-agent` Runtime。
- [ ] `prepare` 只准备/验证指定 Suite 和 repository cache，不运行 Agent 或 Judge。
- [ ] `run-agent` 只产生 Submission/trace，不加载 truth evaluator、不调用 Judge。
- [ ] `evaluate` 只读既有 immutable Submission，写 versioned matches/score/report；可更换 Judge config 重评。
- [ ] `inspect` 展示单 Case/Trial 的 input、submission、matches、Evidence/Judge 状态和可选 trace，不泄漏 secret。
- [ ] 所有命令支持 JSON 输出、稳定 exit code、明确 dry-run/overwrite/resume 语义。
- [ ] 路径、Run ID、Suite ID、并行度、timeout、trial count 和 Provider 参数严格校验。
- [ ] CLI help 清楚区分 Agent provider 与 Judge provider；Judge 配置不能误传给被测 Agent。

**实现：**

- [ ] 在 `pyproject.toml` 增加 `review-agent-eval = "review_agent_eval.cli:main"`。
- [ ] 本 Task 完整实现 `prepare/run-agent/evaluate/inspect` 四个独立命令；`compare/calibrate` 只在 Task 15 的最终能力完成时一起加入，不提前放置占位命令。
- [ ] Provider 构建复用现有统一 Model Adapter factory，stage label 使用 `eval-judge`，凭证只从指定环境变量读取。
- [ ] CLI 只负责编排组件，不复制 parser/evaluator/metric 业务逻辑。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_cli.py tests/eval/test_cli_failures.py -q -p no:cacheprovider --basetemp 'C:\tmp\rae-t12'
```

**提交边界：** `feat(eval): expose separated harness commands`

## Wave D2：Core Case Bank 与公共 Benchmark Adapters

### Task 13：Core Regression/Capability Suite 与人工标注协议

**依赖：** Task 1-12。

**所有权：**

- 新建 `eval/suites/core-regression/manifest.json`
- 新建 `eval/suites/core-capability/manifest.json`
- 新建 `eval/cases/core/`
- 新建 `eval/annotation-guidelines.md`
- 新建 `tests/eval/test_core_suite.py`
- 新建 `tests/eval/test_core_golden_submissions.py`

**RED/数据验收：**

- [ ] 首批至少 18 个小型 Python Case 覆盖 explicit Intent、合法 inferred Intent、必须澄清、不应澄清、unsupported/contradicted Intent 和用户纠正。
- [ ] Review Case 覆盖 correctness/security/regression，diff/file/repo context，clean PR，pre-existing trap，wrong path/line/hash，fabricated Finding、duplicate Finding、compound Finding 和 high/critical miss。
- [ ] 每个 Case 有可读任务说明、固定 base/head tree digest、Intent authority、原子 expected Finding、truth completeness、severity/category/context level 和标注 rationale。
- [ ] Evidence anchor 只在确有帮助时提供，不把唯一文件片段当成唯一合法调查路径。
- [ ] Ground truth 不出现在 repository tree、commit message、文件名、测试输出或 Agent-facing request。
- [ ] capability 与 regression 使用同一 Case schema；区别只在 Suite policy/预期稳定性，不复制 Case 为两种格式。
- [ ] golden Submissions 覆盖 perfect、empty、duplicate、fabricated、bad Evidence、Judge unknown，用于 grader 的确定性回归。
- [ ] 每个新增或修改 Case 通过 annotation checklist、自动 schema/hash 检查和至少一次人工审阅记录。

**实现：**

- [ ] 建立最终 Case 目录与 fixture tree 格式，不提交 nested `.git`。
- [ ] 编写 Intent/Finding/Evidence 原子化、严重度、完整度和 disagreement 的标注指南。
- [ ] 将已稳定通过的 Case放入 regression；当前产品尚不稳定的 Case放入 capability，迁移需显式 Suite version change。
- [ ] 提供 case lint 命令/测试，防 truth leakage、重复 issue 和不稳定 fixture。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_core_suite.py tests/eval/test_core_golden_submissions.py -q -p no:cacheprovider --basetemp 'C:\tmp\rae-t13'
```

**提交边界：** `test(eval): add audited core code review suites`

### Task 14：AACR-Bench 与 SWE-PRBench Adapters

**依赖：** Task 1-12；可与 Task 13 并行。

**所有权：**

- 新建 `src/review_agent_eval/adapters/aacr_bench.py`
- 新建 `src/review_agent_eval/adapters/swe_prbench.py`
- 新建 `tests/eval/test_aacr_adapter.py`
- 新建 `tests/eval/test_swe_prbench_adapter.py`
- 新建 `tests/eval/fixtures/public_datasets/`
- 修改 `pyproject.toml`（仅可选 `eval-public` 依赖）

**RED 测试：**

- [ ] AACR 200 unique PR/positive/negative 元数据口径可由 source manifest 验证，重复 PR/comment、未知语言和缺 source revision 失败。
- [ ] AACR positive comment 映射 expected Finding，negative comment 映射 known-invalid；path/side/line/category/context level 保真。
- [ ] AACR unmatched、缺 comments 和 zero-Finding PR 不被静默跳过；Python eligibility subset 由显式 filter manifest记录。
- [ ] SWE Type1/Type2/Type3、A/B/C config、human comment、language/difficulty 和 repository binding 保真。
- [ ] SWE 使用 `human_observed` completeness，默认 `intent_truth.scorable=false`；Harness 不从标题自动生成参考 Intent。
- [ ] official frozen-context 与 native repository Agent protocol 使用不同 protocol ID/report，不能混为官方排行榜结果。
- [ ] source URI/version/license/hash 全部固定；上游格式漂移 fail closed，不猜字段。
- [ ] 本地缩小 fixture 与真实 parser 共用代码；Parquet 缺可选依赖时给出可操作错误，不影响 Core Harness 安装。

**实现：**

- [ ] 编写两个 Adapter 将上游记录转换为 canonical EvalCase/Suite manifest，不复制 evaluator 逻辑。
- [ ] `prepare` 显式下载或导入用户提供的数据，校验 hash 后写 `.eval-data/`；Trial 不联网。
- [ ] 原始 source ID/comment ID 保留 provenance；转换后 Case digest 可重现。
- [ ] 数据许可和引用信息进入 manifest/report，不将大数据提交主仓库。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_aacr_adapter.py tests/eval/test_swe_prbench_adapter.py -q -p no:cacheprovider --basetemp 'C:\tmp\rae-t14'
```

**提交边界：** `feat(eval): adapt aacr and swe pr review benchmarks`

---

# Batch E：重复 Trial、比较、校准与发布门禁

## Wave E1：Statistics、Comparison 与 Calibration

### Task 15：Repeated Trials、paired compare、Judge calibration 与 Regression Gate

**依赖：** Task 1-14。

**所有权：**

- 新建 `src/review_agent_eval/comparison.py`
- 新建 `src/review_agent_eval/calibration.py`
- 新建 `src/review_agent_eval/gates.py`
- 修改 `src/review_agent_eval/runner.py`
- 修改 `src/review_agent_eval/cli.py`
- 新建 `tests/eval/test_repeated_trials.py`
- 新建 `tests/eval/test_comparison.py`
- 新建 `tests/eval/test_calibration.py`
- 新建 `tests/eval/test_regression_gates.py`

**RED 测试：**

- [ ] 同一 Case 多 Trial 都保留独立 Submission/score，不挑最好一次冒充产品结果。
- [ ] `pass@1` 使用预先版本化 case-pass rubric；`pass^k` 衡量全部 Trial 成功稳定性；研究性 `pass@k` 明确标注。
- [ ] trial count、Case version、Grader/rubric 不一致的 Run 不可直接 paired compare。
- [ ] compare 展示 case-level improved/regressed/unchanged、均值、离散程度与固定算法置信区间；随机重采样 seed/次数写入 artifact。
- [ ] paired comparison 不把 failed/unknown Trial 丢弃，并单独报告 Judge coverage 变化。
- [ ] calibration 导入人工 claim/Finding/Evidence 标签，计算 agreement/confusion matrix/Cohen's kappa，并保存 calibration set/rubric/Judge version。
- [ ] unknown、high/critical fabricated、确定性冲突和随机 confirmed/plausible/fabricated 抽样进入人工复核队列。
- [ ] Regression Gate 在看到 candidate 结果前固定 policy，检查 Intent pass、critical/high miss、precision/recall 回退、fabricated rate、Evidence validity、failure 和成本预算。
- [ ] gate 输出每条失败阈值及对应 Case，不用单一总分决定发布。

**实现：**

- [ ] Runner 支持版本化 trial count、bounded parallelism 和重复 Trial manifest。
- [ ] 使用 stdlib 实现确定性 paired bootstrap/Wilson 等所需统计方法，记录算法版本；不因缺 numpy 改变核心结果。
- [ ] 实现 compatible-run validator、paired compare report 和 case delta artifacts。
- [ ] 实现 human-label import/export、blind calibration package 和复核队列。
- [ ] 完整接通 `compare`、`calibrate` CLI 与 versioned gate policy。
- [ ] Private Held-out 只支持受控 manifest 路径与结果汇总；内容不进入普通 report 或 Git。

**验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval/test_repeated_trials.py tests/eval/test_comparison.py tests/eval/test_calibration.py tests/eval/test_regression_gates.py -q -p no:cacheprovider --basetemp 'C:\tmp\rae-t15'
```

**提交边界：** `feat(eval): compare repeated trials and enforce calibrated gates`

## Wave E2：端到端、安全与文档

### Task 16：E2E、故障注入、边界审计与最终文档

**依赖：** Task 1-15。

**所有权：**

- 新建 `tests/eval/test_e2e_current_agent.py`
- 新建 `tests/eval/test_e2e_re_evaluate.py`
- 新建 `tests/eval/test_security_boundaries.py`
- 新建 `tests/eval/test_architecture_boundaries.py`
- 新建 `docs/eval-system.md`
- 修改 `.gitignore`
- 修改 `README.md`（仅本地 Eval 使用入口）
- 修改 `docs/superpowers/specs/2026-07-16-core-code-review-eval-system-design.md`（仅完成状态与实现链接）
- 修改本计划（仅完成状态、验证证据和必要偏差记录）

**RED/验收测试：**

- [ ] 使用当前 Agent fake provider 在 Core fixture 上完成 prepare -> run-agent -> evaluate -> inspect 全链路。
- [ ] 同一 Submission 更换 scripted Judge/rubric 后可重评，submission hash 不变，旧 score 保留。
- [ ] 两个兼容 Run 可 compare，回归 gate 同时展示 Intent、Review、Evidence、Failure 和成本变化。
- [ ] timeout、invalid Agent output、Judge failure、bad Evidence、truth leakage attempt、path escape、symlink 和 artifact tampering 全部 fail closed。
- [ ] repository prompt injection 不能修改 Harness/Judge policy，不能读取 truth 或调用未授权网络/工具。
- [ ] Core package 不依赖公共数据 optional packages；无 `eval-public` 依赖时 Core suite/CLI 正常。
- [ ] architecture test 保证只有 current Agent Adapter 可以依赖产品 artifact/API；Evaluator 不导入 Runtime/Session/Memory。
- [ ] `.eval-data/`、`.eval-runs/`、API key、private Case 和 raw sensitive trace 不进入 Git/report。
- [ ] Windows/Linux path 与 subprocess 差异有测试或明确兼容策略。
- [ ] 全项目旧功能回归通过，Eval package 不改变正常 `review-agent review/resume/memory` 语义。

**实现：**

- [ ] 编写用户文档：安装、Core suite、当前 Agent、真实 Judge、public prepare、run/evaluate/inspect/compare/calibrate、artifact 与隐私。
- [ ] 文档解释“评的是 Intent/Review outcome，不是 Runtime/Memory 模块”，并给出 Evidence/一对一匹配示例。
- [ ] 记录所有验证命令、退出码、Case/Suite version 和已知外部数据限制。
- [ ] 完成设计—实现 compliance matrix；若实现需要协议变化，先更新设计并重新请求用户确认，不在代码中偷偷扩展 schema。

**定向验证：**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest tests/eval -q -p no:cacheprovider --basetemp 'C:\tmp\rae-eval'
```

**全量回归：**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest -q -p no:cacheprovider --basetemp 'C:\tmp\rae-full'
```

**提交边界：** `docs(eval): complete harness integration and operating guide`

---

## 5. Batch Gates

### Batch A Gate

- [ ] v1 protocols、strict hydration、Case truth isolation、Run Artifact 和 Trial workspace 全部通过。
- [ ] 一个 synthetic Submission 可以在完全不导入产品 Runtime 的进程中 round-trip。
- [ ] truth leakage/path escape/artifact tampering 测试通过。

### Batch B Gate

- [ ] 当前 Agent 与 generic subprocess Agent 都可生成 canonical terminal Submission。
- [ ] clarification required/not-required/answer exhaustion 可在黑盒输出中评分。
- [ ] failure/timeout/zero-Finding 不会被跳过。

### Batch C Gate

- [ ] Intent、Review、Evidence 三类 grader 可单独重跑。
- [ ] global assignment 顺序无关，duplicate/compound 不多赚 Recall。
- [ ] `issue_match` 与 Evidence 状态正交，Judge 失败 fail closed。

### Batch D Gate

- [ ] report 正确展示分子/分母/coverage 和 completeness 分组。
- [ ] Core suite 通过 annotation lint；AACR/SWE 本地 fixture 通过真实 Adapter parser。
- [ ] prepare/run-agent/evaluate/inspect 职责不交叉。

### Batch E Gate

- [ ] repeated trials、paired compare、calibration 和 regression policy 均有 versioned artifact。
- [ ] 端到端与全量回归退出码为 0。
- [ ] 文档、设计状态和实现实际行为一致。

## 6. 设计验收映射

| 已确认设计要求 | 实现 Task | 主要验证 |
|---|---:|---|
| 黑盒 Eval 边界 | 5、6、16 | Adapter dependency test、E2E |
| EvalInput/Submission/Case v1 | 1-3 | strict hydration、artifact round-trip |
| Case Bank 与 truth completeness | 2、13、14 | suite lint、public fixture |
| Intent claim/clarification Eval | 8、9、13 | intent evaluator、Core cases |
| Location/Semantic Matcher | 7、9、10 | matcher/Judge tests |
| Evidence integrity/support 分离 | 7、9、10 | bad hash/wrong line/support tests |
| 全局一对一 Finding 匹配 | 8、10 | non-greedy/duplicate/compound tests |
| Deterministic-first、Judge fail-closed | 7-10 | failure injection |
| Metrics、无单一总分 | 11 | formula/report tests |
| prepare/run/evaluate 解耦 | 3、6、12 | CLI/re-evaluate E2E |
| Core/AACR/SWE suites | 13、14 | adapter and annotation tests |
| 多 Trial、比较、校准、门禁 | 15 | statistical/gate tests |
| 安全、隔离与数据治理 | 2-4、6、16 | security boundary suite |

## 7. 完成定义

只有同时满足以下条件，Eval System 才算完成：

- [ ] 当前 Agent 和至少一个 generic Adapter 使用同一 EvalInput/Submission 协议运行。
- [ ] Core Regression/Capability Suite 能稳定回答 Intent 是否正确、是否该澄清、真实问题是否发现、误报多少、Evidence 是否可发布。
- [ ] Finding 使用全局一对一匹配，问题命中与 Evidence 质量独立评分。
- [ ] AACR-Bench 与 SWE-PRBench 数据可经固定 source/version/hash 转换，且报告清楚标识 completeness/protocol 差异。
- [ ] Agent execution 与 evaluation 可独立重跑，所有失败 Trial 和 Judge failure 可见。
- [ ] 多 Trial 与 paired comparison 可回答新模型/Prompt/Agent 版本在哪些 Case 改善或退化。
- [ ] Judge 经过固定人工校准集评估，结果、rubric 和 agreement 可审计。
- [ ] Regression Gate 不依赖单一总分，能阻止严重漏报、误报、Evidence 退化、failure 和成本失控。
- [ ] `tests/eval` 与全项目 pytest 全部退出码为 0，且未以 cleanup warning 掩盖失败。
- [ ] 旧 Eval 设计稿保持历史原样，新设计稿、实现计划、代码和用户文档相互一致。

最终系统必须能用证据回答：

```text
Agent 理解的修改意图真实吗？
Agent 应该问的时候问了吗，不该问的时候有没有阻塞？
Agent 找到的是真问题还是误报？
它给出的 Evidence 是否真实并足以支持 Finding？
新版本相对基线在哪些 Case 改善、退化，结果是否稳定且成本可接受？
```

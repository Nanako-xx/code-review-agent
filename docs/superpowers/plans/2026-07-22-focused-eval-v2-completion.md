# Focused Eval v2 Completion Implementation Plan

> For agentic workers: use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** 在保留 V1 Repository 评测主干的前提下，完成 Frozen Context 运行、Authority-aware 计分、公共数据本地导入和两条可重放 E2E。

**Architecture:** 以已完成的 v2 ReviewTarget、EvidenceSource、MetricAuthority、WireContract 和 Core 制品为协议基线。Repository 继续使用现有 RepositoryPreparer/replay；Frozen 使用已校验的 SWE bundle；两者通过一个 Target Materialization 行为层进入同一个 Agent/Evidence/Judge 生命周期。现有 Intent、Finding Assignment、Judge、Metrics、Report、resume/rejudge 算法只做边界适配，不重写。

**Tech Stack:** Python 3.11、stdlib dataclasses/json/hashlib/pathlib/subprocess、现有 Repository/Artifact Store、现有 Model Adapter、pytest。测试临时目录使用 D 盘 worktree 外短路径，不使用 C 盘。

**Source of truth:** docs/superpowers/specs/2026-07-22-focused-eval-v2-completion-design.md

**Execution boundary:** 不实现 pinned download、公共 Catalog 服务、HTTP/IPC Adapter、通用 sandbox 平台或大范围新增路径攻击矩阵；不重新实现 V1 主干。

---

## Task 1：Canonical Materialization 行为层与 Repository 投影

**Files**

- Create: src/review_agent_eval/materialization.py
- Create: tests/eval/test_materialization.py
- Modify: src/review_agent_eval/artifacts.py
- Modify: src/review_agent_eval/repository.py
- Modify: src/review_agent_eval/__init__.py
- Modify: tests/eval/test_artifacts.py
- Modify: tests/eval/test_repository.py

**边界：** 只建立统一行为接口和 Repository materializer；不实现 Frozen、Adapter protocol 或 Evaluator。

- [ ] Step 1：锁定现有 v2 Materialization DTO 契约。复用当前 TargetAccess、AgentVisibleFileBinding、TrialMaterializationManifest 的字段、digest 和严格 hydration 规则。新增 round-trip、ID 稳定性、未知字段拒绝测试。若类型从 artifacts.py 移至 materialization.py，只允许一次机械移动；不得保留第二套 DTO/parser，字段和 digest 必须不变。
- [ ] Step 2：实现最小 TargetMaterializer / PreparedTargetMaterialization。接口接收 EvalInput、trial_id、attempt、AdapterCapabilitiesV2，返回 manifest、TargetAccess、Agent-visible file bindings 和同源 replay；不得把宿主绝对路径给 Adapter。
- [ ] Step 3：实现 RepositoryTargetMaterializer。只接收 RepositoryReviewTarget；只调用已有 require_cached/open_replay，禁止 acquisition/network/prepare；复用已有 workspace manifest 的 base/head/repository digest；Target 替换、revision 漂移和 receipt/attempt 不一致必须在返回前失败。
- [ ] Step 4：运行并提交。使用 D:\Anaconda\envs\MINIST\python.exe -m pytest tests/eval/test_materialization.py tests/eval/test_artifacts.py tests/eval/test_repository.py -q -p no:cacheprovider --basetemp D:\tmp\code-review-agent\focused-materialization。预期新契约与既有 Artifact/Repository 测试通过，且没有 acquisition 调用。提交消息：feat(eval): add canonical target materialization boundary。

## Task 2：Frozen Context Runtime、Adapter 能力与 Runner 分派

**Files**

- Create: src/review_agent_eval/frozen_context.py
- Create: tests/eval/test_frozen_context.py
- Create: tests/eval/test_target_runner.py
- Modify: src/review_agent_eval/materialization.py
- Modify: src/review_agent_eval/adapters/base.py
- Modify: src/review_agent_eval/adapters/current_agent.py
- Modify: src/review_agent_eval/adapters/subprocess_agent.py
- Modify: src/review_agent_eval/runner.py
- Modify: src/review_agent_eval/artifacts.py
- Modify: tests/eval/test_agent_adapter.py
- Modify: tests/eval/test_runner.py

**边界：** 只让两种 Target 真正进入 Agent 生命周期；不修改 Finding/Evidence 评分算法。

- [ ] Step 1：写 Frozen bundle 关键边界测试。覆盖 exact rendered UTF-8 bytes、external expected bundle digest、record/context digest、相对 TargetAccess、目标文件替换，以及 current-agent 在 Frozen 上 preflight incompatible。Materialization 失败时 fake Agent 调用计数必须为零。
- [ ] Step 2：实现 FrozenContextTargetMaterializer。复用 V1 Task 14 的 prepare_swe_prbench_frozen_bundle/read_swe_prbench_frozen_bundle；不重新渲染、不下载、不从 Trial workspace 读取正文；每个 record 生成只读 Agent-visible 内容和 Frozen replay。
- [ ] Step 3：升级 Adapter capability 和 subprocess envelope。固定 current-agent-cli-v2=repository-only；subprocess-json-v2 按声明开放 Repository/Frozen。stdin envelope 为 schema_version/eval_input/trial_binding/target_access/materialization_id；输出必须回指相同 task/trial/input/materialization identity。保留 cancellation、output bound 和 failure taxonomy。
- [ ] Step 4：接入 Runner per-attempt materialization。顺序固定为 start_trial -> materialize -> create-only prepare receipt -> Agent -> terminal Submission。Target/capability/materialization 不匹配必须在 Agent 调用前成为 Harness failure；旧 attempt 不能提交新结果。
- [ ] Step 5：运行并提交。使用 D:\Anaconda\envs\MINIST\python.exe -m pytest tests/eval/test_frozen_context.py tests/eval/test_target_runner.py tests/eval/test_agent_adapter.py tests/eval/test_runner.py -q -p no:cacheprovider --basetemp D:\tmp\code-review-agent\focused-target-runtime。提交消息：feat(eval): run repository and frozen targets through one runtime。

## Task 3：Evidence Replay 与 Authority-aware Evaluator 适配

**Files**

- Create: tests/eval/test_frozen_evidence.py
- Create: tests/eval/test_metric_authority.py
- Create: tests/eval/test_evaluator_context.py
- Modify: src/review_agent_eval/evidence_checker.py
- Modify: src/review_agent_eval/review_evaluator.py
- Modify: src/review_agent_eval/metrics.py
- Modify: src/review_agent_eval/report.py
- Modify: src/review_agent_eval/orchestrator.py
- Modify: tests/eval/test_evidence_checker.py
- Modify: tests/eval/test_review_evaluator.py
- Modify: tests/eval/test_metrics.py
- Modify: tests/eval/test_report.py

**边界：** 复用现有 Intent、Finding Assignment、Judge 和 Evidence support；只增加 Frozen dispatch、authority eligibility 和 SWE truth-scoped context。

- [ ] Step 1：写 Frozen Evidence replay 契约测试。覆盖错误 target_materialization_id、context_ref、行范围、rendered bytes/hash 漂移，以及把 Frozen Evidence 当 Repository revision。错误必须确定性地产生 invalid 或 missing，不能交给 Judge 修复。
- [ ] Step 2：将 Evidence Checker 接到 Repository/Frozen replay union。Repository file/diff、command output、existing-CI 沿用已有逻辑；Frozen 使用 target_materialization_id + context_ref + from_line/to_line，验证 Submission、Trial receipt、Materialization 三方 identity。
- [ ] Step 3：实现 Authority 计分。固定 Core severity/location 可评分；AACR severity 不可评分、location 可评分；SWE 两者均不可评分。禁止 placeholder MEDIUM 进入 severity denominator；不可评分 location 只能保留 semantic/diagnostic context，不能进入 Line 分母。
- [ ] Step 4：接入 SWE truth-scoped diff hunk。每条 diff hunk 绑定唯一 truth ID 和 provenance digest，只进入对应 Finding-equivalence Judge request；不进入 Agent Input、Submission Evidence、Location matcher 或其他 Finding。
- [ ] Step 5：扩展 Score/Report compatibility。在现有 partition 基础上加入 authority profile、Target/wire/isolation binding，并保存 eligible/excluded/not-scorable coverage。不同 Target 或 authority profile 不得 roll-up；不修改 Finding assignment edge weight。
- [ ] Step 6：运行并提交。使用 D:\Anaconda\envs\MINIST\python.exe -m pytest tests/eval/test_frozen_evidence.py tests/eval/test_metric_authority.py tests/eval/test_evaluator_context.py tests/eval/test_evidence_checker.py tests/eval/test_review_evaluator.py tests/eval/test_metrics.py tests/eval/test_report.py -q -p no:cacheprovider --basetemp D:\tmp\code-review-agent\focused-evaluator。提交消息：feat(eval): score frozen evidence with metric authority。

## Task 4：AACR/SWE 最终 Adapter 与 prepare-public local-import

**Files**

- Create: tests/eval/test_prepare_public.py
- Modify: src/review_agent_eval/adapters/_public.py
- Modify: src/review_agent_eval/adapters/aacr_bench.py
- Modify: src/review_agent_eval/adapters/swe_prbench.py
- Modify: src/review_agent_eval/cli.py
- Modify: src/review_agent_eval/datasets.py
- Modify: tests/eval/test_public_adapter_common.py
- Modify: tests/eval/test_aacr_adapter.py
- Modify: tests/eval/test_swe_prbench_adapter.py
- Modify: tests/eval/test_cli.py

**边界：** 复用已有严格 Parser、Source Manifest、Record Receipt 和 Frozen Bundle verifier；不实现网络下载器。

- [ ] Step 1：写迁移验收测试。证明 AACR/SWE 输出当前 v2 Case schema；AACR severity 为 null/not-scorable；SWE native/frozen 的 protocol/Suite identity 分离；SWE frozen 能生成 runnable Case；所有 requires_eval_v2 limitation 消失。
- [ ] Step 2：迁移 AACR/SWE case projection。只改 Target、MetricAuthority、EvaluatorContext 和 Evidence/record binding；保留 raw/scorable/isolated statistics、source pointers、license/version/hash 和 filtering 语义；不得重新实现上游解析。
- [ ] Step 3：增加 prepare-public --mode local-import。接收本地 source root、expected manifest/profile digest、filter 和 output root，调用现有 verifier 并 create-only 发布 canonical Suite/Frozen bundle。下载、网络、凭证和宿主绝对路径不得进入 Trial。
- [ ] Step 4：运行并提交。使用 D:\Anaconda\envs\MINIST\python.exe -m pytest tests/eval/test_prepare_public.py tests/eval/test_public_adapter_common.py tests/eval/test_aacr_adapter.py tests/eval/test_swe_prbench_adapter.py tests/eval/test_cli.py -q -p no:cacheprovider --basetemp D:\tmp\code-review-agent\focused-public。提交消息：feat(eval): finalize public target adapters and local import。

## Task 5：Repository/Frozen E2E 与 focused acceptance

**Files**

- Create: tests/eval/test_e2e_repository_v2.py
- Create: tests/eval/test_e2e_frozen_v2.py
- Create: tests/eval/test_protocol_v2_cutover.py
- Modify: src/review_agent_eval/cli.py
- Modify: src/review_agent_eval/orchestrator.py
- Create: docs/eval-system.md

- [ ] Step 1：Repository E2E。使用 Core fixture 完成 prepare -> run-agent -> evaluate -> re-evaluate -> inspect，确认 Repository、Intent、Review、Evidence、Report、resume/rejudge 未退化。
- [ ] Step 2：Frozen E2E。使用最小 SWE frozen fixture 完成 prepare-public -> prepare -> run-agent -> evaluate -> inspect；current-agent 必须 preflight incompatible，frozen-capable subprocess 必须能完成 Trial。
- [ ] Step 3：运行 focused 故障边界。覆盖 Target replacement、wrong identity、capability mismatch、Frozen Evidence drift、authority misuse、cross-partition aggregation 和 stale resume worker；不增加新的安全平台。
- [ ] Step 4：全量回归并提交。使用 D:\Anaconda\envs\MINIST\python.exe -m pytest tests/eval -q -p no:cacheprovider --basetemp D:\tmp\code-review-agent\focused-e2e。预期 Eval 回归通过；平台预期 skip 单独列出；不伪造真人 Reviewer B 或真实模型 baseline。提交消息：test(eval): verify repository and frozen evaluation lifecycles。

## 完成后的下一阶段

Task 1–5 完成后，不再修改 Target/Evaluator 协议，回到原计划 Task 15：Repeated Trials、paired compare、Judge calibration 和 Regression Gate。Task 15 单独制定计划，不在本计划中提前实现。

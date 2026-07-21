# Eval Protocol v2 Full Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前 Eval Harness 一次性切换为完整、可重放、只接受 v2 的 Code Review 评测协议，同时正式支持 Repository 与 Frozen Context 两种 Review Target、权威感知指标、可信公共数据准备和完整本地评测生命周期。

**Architecture:** 保留现有黑盒 `EvalInput -> Agent Adapter -> EvalSubmission -> Evaluators -> Scores/Report` 主干，在输入侧引入严格 tagged Review Target，在执行侧引入双 Materializer 和统一 Materialization/Replay binding，在评判侧引入 Metric Authority 与 truth-scoped evaluator context，在控制面增加独立 `prepare-public`。活动 Loader、Runner、Evaluator、Artifact Store 和 CLI 全量使用 v2，不实现 v1 parser、迁移、resume、rejudge 或混合协议兼容层。

**Tech Stack:** Python 3.11+ frozen dataclasses/enums、stdlib JSON/hashlib/pathlib/subprocess/ssl/zipfile/tarfile/statistics、现有统一 Model Adapter、Git、pytest；`pyarrow` 仅属于可选 `eval-public` 依赖。

---

## 0. Source of Truth、当前基线与执行约束

**设计来源：** `docs/superpowers/specs/2026-07-16-core-code-review-eval-system-design.md`

**当前实现基线：** 分支 `codex/public-benchmark-adapters`，设计提交 `7d05240`。现有 Task 1–13 和 Task 14 公共 Adapter 主体是 v1 实现基础，不是需要并存的兼容版本。

**执行前必须保护的未提交内容：**

- `src/review_agent_eval/adapters/_public.py`
- `src/review_agent_eval/adapters/aacr_bench.py`
- `src/review_agent_eval/adapters/swe_prbench.py`
- `tests/eval/test_public_adapter_common.py`
- `tests/eval/test_aacr_adapter.py`
- `tests/eval/test_swe_prbench_adapter.py`
- `tests/eval/fixtures/public_datasets/`
- `pyproject.toml`

这些文件属于已验证的 Task 14 基线。执行 v2 前先用下列定向测试确认，再选择性提交；不得把旧 Plan 修改、临时目录、`__pycache__` 或 pytest 目录带入提交：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_public_adapter_common.py `
  tests/eval/test_aacr_adapter.py `
  tests/eval/test_swe_prbench_adapter.py `
  tests/eval/test_cases.py `
  tests/eval/test_datasets.py `
  tests/test_architecture_boundaries.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-task14-baseline'
```

Expected: exit code 0；当前基线口径为 158 passed、1 个平台预期 skip。

选择性提交命令：

```powershell
git add pyproject.toml `
  src/review_agent_eval/adapters/_public.py `
  src/review_agent_eval/adapters/aacr_bench.py `
  src/review_agent_eval/adapters/swe_prbench.py `
  tests/eval/test_public_adapter_common.py `
  tests/eval/test_aacr_adapter.py `
  tests/eval/test_swe_prbench_adapter.py `
  tests/eval/fixtures/public_datasets
git diff --cached --name-only
git commit -m "feat(eval): adapt public code review benchmarks"
```

本 Plan 单独选择性提交：

```powershell
git add docs/superpowers/plans/2026-07-21-eval-protocol-v2.md
git diff --cached --name-only
git commit -m "docs(eval): plan protocol v2 cutover"
```

执行约束：

- 所有 scratch、pytest `--basetemp` 和临时 clone 使用 `D:\tmp\code-review-agent\`；不得再向 `C:\tmp` 写入。
- 不清理、不暂存 `.intent-*`、`.p-*`、`.pytest-*`、`.tmp`、`v1/`、`v2/`、`w1/`、`w2/`、`x/`、`y/`、`z*/` 或用户已有未跟踪目录。
- 每个 Wave 可以包含多个中间提交，但整个 Wave 通过 Gate 前不可合并或发布；不得用临时 v1/v2 兼容代码让半条协议看似可运行。
- 测试采用风险驱动策略。只有协议混用、Materialization trust、Evidence replay、Metric Authority、public acquisition、安全边界和实际 bug 修复要求先观察 RED；机械 JSON 迁移、生成制品、样板、文档和纯展示代码使用集中 GREEN 验证。
- fake/scripted Model Adapter 只证明协议、路由、解析和 failure taxonomy；真实 Judge 语义能力仍由独立人工 calibration 数据证明。
- Task 13 的独立真人 Reviewer B 和真实模型每个 Regression Case 三次 baseline 是外部门禁。代码不得伪造这两类记录；v2 切换完成后重新生成对应 blind packet/baseline identity。

## 1. 文件结构与责任边界

### 新建文件

| 文件 | 唯一责任 |
|---|---|
| `src/review_agent_eval/materialization.py` | TargetAccess、TrialMaterializationManifest、双 Materializer 共同协议、dispatch 与 replay union |
| `src/review_agent_eval/frozen_context.py` | Frozen bundle trust、exact rendered bytes materialization、只读 replay |
| `src/review_agent_eval/public_acquisition.py` | 可信 catalog、local-import、pinned-download、Source Object、repository mirror receipt |
| `src/review_agent_eval/comparison.py` | compatible Run 的 paired comparison、case delta、置信区间 |
| `src/review_agent_eval/calibration.py` | blind human labels、Judge agreement/confusion matrix/Cohen's kappa |
| `src/review_agent_eval/gates.py` | 版本化 Regression Gate policy 与逐阈值结果 |
| `tests/eval/test_materialization.py` | Materialization identity、dispatcher、Repository target trust |
| `tests/eval/test_frozen_context.py` | exact bytes、bundle trust、Frozen replay 与隔离 |
| `tests/eval/test_public_acquisition.py` | catalog trust、两种 acquisition、archive/path/size/hash 安全 |
| `tests/eval/test_metric_authority.py` | Severity/Location authority 与 null coverage |
| `tests/eval/test_evaluator_context.py` | truth-scoped diff_hunk、blind Judge context、防泄漏 |
| `tests/eval/test_repeated_trials.py` | 多 Trial、pass@1、pass^k 和失败 coverage |
| `tests/eval/test_comparison.py` | compatibility key、paired delta、统计 artifact |
| `tests/eval/test_calibration.py` | 人工标注导入、agreement、review queue |
| `tests/eval/test_regression_gates.py` | 固定 policy、逐阈值诊断、无 Overall Score |
| `tests/eval/test_e2e_repository_v2.py` | Repository prepare/run/evaluate/re-evaluate/inspect 全链 |
| `tests/eval/test_e2e_frozen_v2.py` | Frozen prepare/run/evaluate/re-evaluate/inspect 全链 |
| `tests/eval/test_protocol_v2_cutover.py` | v1 稳定拒绝、跨版本混用、活动代码无 v1 root |
| `docs/eval-system.md` | 安装、数据准备、运行、重评、比较、校准、安全和口径说明 |

### 主要修改文件

| 文件 | v2 责任 |
|---|---|
| `src/review_agent_eval/models.py` | EvalInputV2、EvalSubmissionV2、EvalCaseV2、ReviewTarget/EvidenceSource tagged unions、MetricAuthority、EvaluatorContext |
| `src/review_agent_eval/cases.py` | WireContractV2、PublicSuitePreparationBindingV2、SuiteManifestV2、RunCaseSnapshotV2 |
| `src/review_agent_eval/datasets.py` | 单次安全读取、v2-only CaseBank、public preparation binding 验证 |
| `src/review_agent_eval/config.py` | EvalRunConfigV2、EvaluatorExecutionConfigV2、wire/capability/materialization/authority binding |
| `src/review_agent_eval/artifacts.py` | v2 Run/Trial/Stage/Preflight/Materialization artifact、create-only recovery |
| `src/review_agent_eval/repository.py` | Repository prepared source/replay 复用与 RepositoryTargetMaterializer |
| `src/review_agent_eval/adapters/base.py` | AdapterCapabilitiesV2、兼容性、TargetAccess boundary |
| `src/review_agent_eval/adapters/current_agent.py` | Repository-only current-agent-cli-v2 输入与 v2 Submission 转换 |
| `src/review_agent_eval/adapters/subprocess_agent.py` | subprocess-json-v2 envelope、relative path 与 identity binding |
| `src/review_agent_eval/runner.py` | kind dispatch、per-attempt materialization、always-submission、preflight |
| `src/review_agent_eval/submission.py` | v2 binding、materialization ID、失败终态 |
| `src/review_agent_eval/evidence_checker.py` | Repository/Frozen/command/CI immutable replay |
| `src/review_agent_eval/intent_evaluator.py` | eval_intent_evaluation_v2 与完整 candidate graph replay |
| `src/review_agent_eval/match_location.py` | authority-aware diagnostic candidate 与 LocationAudit |
| `src/review_agent_eval/review_evaluator.py` | truth-scoped context、ReviewEvaluationV2、受控重放 |
| `src/review_agent_eval/judge.py` | eval_blind_judge_input_v2、四 profile、typed v2 response/artifact |
| `src/review_agent_eval/metrics.py` | authority-sensitive MetricsPolicyV2、ScoreV2、coverage/compatibility |
| `src/review_agent_eval/report.py` | SummaryV2、InspectionV2、partition、redacted projection |
| `src/review_agent_eval/orchestrator.py` | v2 Judge/Evaluator/Score 组合与重评 namespace |
| `src/review_agent_eval/cli.py` | prepare-public/prepare/run-agent/evaluate/compare/inspect/calibrate |
| `src/review_agent_eval/*_exports.py`、`src/review_agent_eval/__init__.py`、`src/review_agent_eval/adapters/__init__.py` | 公开 v2 类型导出，不导出活动 v1 root |
| `eval/authoring/*.py` | Core v2 Case/Golden/Suite 生成、lint、blind packet identity |
| `eval/cases/core/`、`eval/suites/core-*` | 18 个 v2 Case、12 个 v2 Golden、两个 v2 Suite manifest |
| `src/review_agent_eval/adapters/_public.py`、`aacr_bench.py`、`swe_prbench.py` | v2 public preparation、authority、native/frozen cohort |
| `tests/eval/test_*.py` | 既有行为迁移到 v2；删除对活动 v1 root 的正向断言 |
| `pyproject.toml` | 仅保留核心 stdlib 与可选 `eval-public` 依赖 |

`src/review_agent/` 产品 Runtime、Session、Memory、Risk 和 Reviewer 业务代码不进入本次修改范围。只有 `review_agent_eval/adapters/current_agent.py` 了解产品输出形状。

## 2. 锁定的 v2 类型与接口

以下名称和字段在 Task 1–6 固定，后续 Task 不创建同义 DTO、旁路 JSON 或基于 `protocol_id` 的执行猜测。

### 2.1 Review Target 与 Evidence Source

```python
class ReviewTargetKind(str, Enum):
    REPOSITORY = "repository"
    FROZEN_CONTEXT = "frozen_context"

@dataclass(frozen=True)
class RepositoryReviewTarget(_JsonModel):
    kind: ReviewTargetKind
    repository: Repository
    review_request: ReviewRequest

@dataclass(frozen=True)
class FrozenContextReviewTarget(_JsonModel):
    kind: ReviewTargetKind
    bundle_id: str
    record_id: str
    context_format: str
    rendered_sha256: str
    rendered_utf8_bytes: int
    source_binding_digest: str

ReviewTargetV2 = Union[RepositoryReviewTarget, FrozenContextReviewTarget]

@dataclass(frozen=True)
class EvalInput(_JsonModel):
    schema_version: str
    task_id: str
    review_target: ReviewTargetV2

@dataclass(frozen=True)
class RepositoryFileEvidenceSource(_JsonModel):
    kind: EvidenceKind
    target_materialization_id: str
    revision: str
    path: str
    from_line: int
    to_line: int

@dataclass(frozen=True)
class RepositoryDiffEvidenceSource(_JsonModel):
    kind: EvidenceKind
    target_materialization_id: str
    base_revision: str
    head_revision: str
    path: str

@dataclass(frozen=True)
class FrozenContextEvidenceSource(_JsonModel):
    kind: EvidenceKind
    target_materialization_id: str
    context_ref: str
    from_line: int
    to_line: int

@dataclass(frozen=True)
class CommandOutputEvidenceSource(_JsonModel):
    kind: EvidenceKind
    target_materialization_id: str
    command: Tuple[str, ...]
    exit_code: int
    stream: EvidenceStream
    artifact_ref: str

@dataclass(frozen=True)
class ExternalRecordEvidenceSource(_JsonModel):
    kind: EvidenceKind
    target_materialization_id: str
    source_ref: str

EvidenceSourceV2 = Union[
    RepositoryFileEvidenceSource,
    RepositoryDiffEvidenceSource,
    FrozenContextEvidenceSource,
    CommandOutputEvidenceSource,
    ExternalRecordEvidenceSource,
]

@dataclass(frozen=True)
class SubmissionEvidence(_JsonModel):
    evidence_id: str
    source: EvidenceSourceV2
    content_hash: str
    excerpt: str
```

`EvalSubmission` 根字段固定为 `schema_version/task_id/agent_id/trial_id/eval_input_digest/target_materialization_id/status/intent/review/evidence/usage/trace_ref/failure`。`EvalCaseInput` 只含 `review_target`。

### 2.2 Metric Authority 与 evaluator-only context

```python
class MetricAuthoritySource(str, Enum):
    EXPERT_ANNOTATION = "expert_annotation"
    UPSTREAM_ANNOTATION = "upstream_annotation"

@dataclass(frozen=True)
class MetricAuthority(_JsonModel):
    severity_scorable: bool
    severity_authority: Optional[MetricAuthoritySource]
    location_scorable: bool
    location_authority: Optional[MetricAuthoritySource]

class EvaluatorContextTask(str, Enum):
    FINDING_EQUIVALENCE = "finding_equivalence"

class EvaluatorContextSourceKind(str, Enum):
    DIFF_HUNK = "diff_hunk"

@dataclass(frozen=True)
class EvaluatorContextProvenance(_JsonModel):
    source_role: str
    source_file_sha256: str
    record_pointer: str
    record_sha256: str

@dataclass(frozen=True)
class EvaluatorContextSource(_JsonModel):
    kind: EvaluatorContextSourceKind
    content: str
    content_sha256: str
    provenance: EvaluatorContextProvenance

@dataclass(frozen=True)
class TruthEvaluatorContext(_JsonModel):
    truth_id: str
    allowed_tasks: Tuple[EvaluatorContextTask, ...]
    sources: Tuple[EvaluatorContextSource, ...]

@dataclass(frozen=True)
class ReviewEvaluatorContext(_JsonModel):
    truth_contexts: Tuple[TruthEvaluatorContext, ...]
```

Core expected Findings 使用 `expert_annotation` 且 Severity/Location 都可评分；AACR 使用 `location_scorable=true + upstream_annotation`、`severity_scorable=false + severity=null`；SWE 两者都不可评分且 Severity 必须为 null。

### 2.3 Wire Contract、Materialization 与 Adapter

```python
@dataclass(frozen=True)
class WireContractV2(_JsonModel):
    case_schema_version: str
    input_schema_version: str
    submission_schema_version: str
    review_target_kind: ReviewTargetKind
    materializer_protocol: str

@dataclass(frozen=True)
class TargetAccess(_JsonModel):
    target_materialization_id: str
    readable_relative_paths: Tuple[str, ...]

@dataclass(frozen=True)
class AgentVisibleFileBinding(_JsonModel):
    role: str
    relative_path: str
    size_bytes: int
    sha256: str

@dataclass(frozen=True)
class TrialMaterializationManifest(_JsonModel):
    schema_version: str
    run_id: str
    task_id: str
    trial_id: str
    attempt: int
    eval_input_digest: str
    review_target_digest: str
    wire_contract: WireContractV2
    suite_preparation_binding_digest: Optional[str]
    prepared_source_id: str
    adapter_capabilities_digest: str
    target_access: TargetAccess
    files: Tuple[AgentVisibleFileBinding, ...]
    replay_binding_digest: str
    materialization_id: str

@dataclass(frozen=True)
class TrialMaterialization:
    manifest: TrialMaterializationManifest
    target_root: Path
    work_root: Path
    replay: Union[PreparedRepositoryReplay, PreparedFrozenContextReplay]

class TargetMaterializer(Protocol):
    protocol_id: str

    def materialize(
        self,
        *,
        eval_input: EvalInput,
        run_config: EvalRunConfig,
        trial_manifest: TrialManifest,
        attempt: int,
        adapter_capabilities: AdapterCapabilitiesV2,
    ) -> TrialMaterialization:
        ...

@dataclass(frozen=True)
class AdapterCapabilitiesV2(_JsonModel):
    schema_version: str
    adapter_id: str
    adapter_version: str
    input_schema_version: str
    submission_schema_version: str
    target_kinds: Tuple[ReviewTargetKind, ...]
    evidence_kinds: Tuple[EvidenceKind, ...]
    clarification_protocol: str
    trace_protocol: str
    subprocess_wire_version: Optional[str]
    isolation_profile: str

class AgentUnderTestAdapter(Protocol):
    def capabilities(self) -> AdapterCapabilitiesV2:
        ...

    def compatibility(
        self,
        eval_input: EvalInput,
        materialization: TrialMaterialization,
        config: AgentRunConfig,
    ) -> AdapterCompatibility:
        ...

    def run(
        self,
        eval_input: EvalInput,
        target_access: TargetAccess,
        work_root: Path,
        config: AgentRunConfig,
        clarification_channel: ClarificationChannel,
    ) -> EvalSubmission:
        ...
```

静态 preflight 在创建 Trial 前用 `WireContractV2 + AdapterCapabilitiesV2` 拒绝 target/evidence/wire 不兼容；per-attempt Materialization 后再调用 `compatibility()` 做防御性复核。后者若与已通过的 preflight 矛盾，属于 Harness integrity failure，不计为 Agent `adapter_error`。

`TargetAccess` exact keys 只有 `target_materialization_id/readable_relative_paths`，路径全部相对 Runner 控制的 Trial root。Python Adapter 只在 Runner 已验证的 Trial root 下解析它们；`subprocess-json-v2` 以 Trial root 为 cwd，只收到相对路径，不收到宿主绝对路径。可写 `work_root` 通过独立参数或 subprocess `trial_binding` 提供，不混入 Target 的只读授权。

### 2.4 活动 artifact version matrix

必须切到 v2 的根或直接绑定 artifact：

```text
eval_input_v2
eval_submission_v2
eval_case_v2
suite_manifest_v2
eval_run_case_snapshot_v2
eval_run_config_v2
eval_evaluator_execution_config_v2
eval_run_manifest_v2
eval_trial_manifest_v2
eval_stage_receipt_v2
eval_capability_preflight_v2
eval_trial_materialization_v2
eval_intent_evaluation_v2
eval_review_evaluation_v2
eval_blind_judge_input_v2
eval_judge_input_artifact_v2
eval_judge_output_artifact_v2
eval_trial_score_v2
eval_case_score_v2
eval_aggregate_score_v2
eval_run_report_summary_v2
eval_trial_inspection_v2
eval_redacted_artifact_projection_v2
public_dataset_preparation_receipt_v2
public_dataset_preparation_packet_v2
swe_prbench_frozen_context_bundle_v2
```

语义未改变、允许继续保留自身 v1 token 的独立叶子协议：

```text
public_dataset_source_v1
public_dataset_filter_v1
public_dataset_catalog_v1
public_dataset_acquisition_receipt_v1
public_repository_catalog_v1
repository_trust_anchor_v1
local_repository_mirror_receipt_v1
review_agent_ci_evidence_bundle_v1
repository cache/isolation/path/budget internal policies
core annotation/human-review/provenance authoring records
```

保留叶子 token 不允许任何活动 parent 继续输出 `eval_*_v1` root，也不允许 Loader 根据嵌套 v1 root 继续 hydrate。

## 3. 实施依赖顺序

```text
Wave 1  Canonical v2 schemas + artifact graph + regenerated Core assets
   -> Wave 2  Repository/Frozen materialization + Adapter/Runner boundary
      -> Wave 3  Evidence/Intent/Review/Judge + authority-aware Scores/Reports
         -> Wave 4  Trusted public acquisition + AACR/SWE v2 publication + CLI
            -> Wave 5  repeated trials/compare/calibrate/gates + E2E/security/docs
```

---

# Wave 1：Schema、Wire Contract 与制品迁移

### Task 1：切换 EvalInput、EvalSubmission、EvalCase 根协议

**Files:**

- Modify: `src/review_agent_eval/models.py`
- Modify: `src/review_agent_eval/submission.py`
- Modify: `src/review_agent_eval/__init__.py`
- Modify: `src/review_agent_eval/review_exports.py`
- Test: `tests/eval/test_models.py`
- Test: `tests/eval/test_schema_hydration.py`
- Create: `tests/eval/test_protocol_v2_cutover.py`

- [ ] **Step 1: 写协议混用关键 RED 契约**

在 `tests/eval/test_protocol_v2_cutover.py` 固定根版本先验拒绝和 tagged union exact-key：

```python
def test_active_loader_rejects_v1_before_nested_hydration() -> None:
    with pytest.raises(UnsupportedProtocolVersionError) as caught:
        EvalInput.from_json(b'{"schema_version":"eval_input_v1"}')
    assert caught.value.code == "unsupported_protocol_version"
    assert caught.value.expected == "eval_input_v2"
    assert caught.value.actual == "eval_input_v1"

def test_review_target_rejects_mixed_repository_and_frozen_fields() -> None:
    payload = valid_repository_eval_input_dict()
    payload["review_target"]["bundle_id"] = "bundle-x"
    with pytest.raises(SchemaError, match="review_target has unexpected fields"):
        EvalInput.from_dict(payload)
```

- [ ] **Step 2: 运行 RED 并确认失败原因来自未实现 v2**

Run:

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_protocol_v2_cutover.py `
  tests/eval/test_models.py `
  tests/eval/test_schema_hydration.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-w1-t1-red'
```

Expected: FAIL，因为活动 Loader 仍接受 `eval_*_v1` 且 EvalInput 没有 `review_target` tagged union。

- [ ] **Step 3: 实现 v2 根类型、严格 tagged unions 与稳定版本错误**

在 `models.py`：

- 将三个根常量改为 `eval_input_v2/eval_submission_v2/eval_case_v2`；
- 增加 `UnsupportedProtocolVersionError(code, expected, actual)`，根 Loader 只读取并验证 `schema_version` 后再解析其他字段；
- 实现 `RepositoryReviewTarget/FrozenContextReviewTarget` 和唯一 `_review_target_from_dict()` dispatch；
- 把 `EvalInput` 与 `EvalCaseInput` 改为单一 `review_target` 字段；
- 实现五种 `EvidenceSourceV2` 分支，`SubmissionEvidence` 只保存 `evidence_id/source/content_hash/excerpt`；
- 给 `EvalSubmission` 增加 `eval_input_digest` 和 `target_materialization_id`，加入 `HARNESS_MATERIALIZATION_ERROR` failure code及 7.3 节终态矩阵；
- 实现 `MetricAuthority`、`ReviewEvaluatorContext`、provenance leaf，并把它们加入 `ExpectedFinding` 和 `EvalCase`；
- 保留 Submission 中可评分的坏 path/line/revision 形状，kind-specific 合法性留给 Checker；
- 将 identity namespace 改为 `review_agent_eval.identity_v2`，重新派生所有 Harness-owned ID；
- 更新 `submission.failure_submission()` 与 binding 校验，禁止构造缺少 input/materialization binding 的终态 Submission。

根 dispatch 使用下列确定性结构，不做字段存在性猜测：

```python
def _review_target_from_dict(value: Any) -> ReviewTargetV2:
    payload = _object(value, "review_target")
    kind = _enum_value(ReviewTargetKind, payload.get("kind"), "review_target.kind")
    if kind is ReviewTargetKind.REPOSITORY:
        return RepositoryReviewTarget.from_dict(payload)
    if kind is ReviewTargetKind.FROZEN_CONTEXT:
        return FrozenContextReviewTarget.from_dict(payload)
    raise _error("review_target has an unknown kind")
```

- [ ] **Step 4: 迁移模型测试到 v2 最终形状**

更新 `test_models.py` 和 `test_schema_hydration.py`，覆盖：

- Repository/Frozen 两种 round-trip；
- EvalCase exact root keys 和 `review_evaluator_context`；
- Metric Authority nullability；
- 五种 Evidence source exact keys；
- Submission 终态矩阵、materialization binding、duplicate IDs 与 dangling refs；
- v1 root、unknown kind、mixed target/evidence、bool-as-int、NaN/Infinity、资源上限拒绝。

- [ ] **Step 5: 运行 GREEN**

Run: 与 Step 2 相同。

Expected: PASS；三个活动根只接受 v2，Repository/Frozen 与五种 Evidence source 严格 round-trip。

- [ ] **Step 6: 选择性提交**

```powershell
git add src/review_agent_eval/models.py `
  src/review_agent_eval/submission.py `
  src/review_agent_eval/__init__.py `
  src/review_agent_eval/review_exports.py `
  tests/eval/test_models.py `
  tests/eval/test_schema_hydration.py `
  tests/eval/test_protocol_v2_cutover.py
git diff --cached --name-only
git commit -m "feat(eval): define canonical protocol v2 roots"
```

### Task 2：切换 Suite、Snapshot、Run/Trial 与 Artifact graph

**Files:**

- Modify: `src/review_agent_eval/cases.py`
- Modify: `src/review_agent_eval/datasets.py`
- Modify: `src/review_agent_eval/config.py`
- Modify: `src/review_agent_eval/artifacts.py`
- Test: `tests/eval/test_cases.py`
- Test: `tests/eval/test_datasets.py`
- Test: `tests/eval/test_config.py`
- Test: `tests/eval/test_artifacts.py`
- Test: `tests/eval/test_task12_artifact_repository_api.py`
- Test: `tests/eval/test_task12_rejudge.py`

- [ ] **Step 1: 扩展 RED 契约到 parent/child 混用和 content binding**

增加测试：

```python
def test_v2_manifest_rejects_v1_case_child() -> None:
    payload = valid_suite_manifest_v2_dict()
    payload["wire_contract"]["case_schema_version"] = "eval_case_v1"
    with pytest.raises(UnsupportedProtocolVersionError):
        SuiteManifest.from_dict(payload)

def test_snapshot_rejects_repository_and_frozen_case_mix() -> None:
    manifest, cases = mixed_target_suite_fixture()
    with pytest.raises(SchemaError, match="single wire contract"):
        RunCaseSnapshot.build(manifest, cases)
```

- [ ] **Step 2: 运行 RED**

Run:

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_cases.py tests/eval/test_datasets.py `
  tests/eval/test_config.py tests/eval/test_artifacts.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-w1-t2-red'
```

Expected: FAIL，因为 Suite/Run/Trial 仍是 v1 且没有 wire/preparation/materialization binding。

- [ ] **Step 3: 实现 SuiteManifestV2 与 truth-free RunCaseSnapshotV2**

在 `cases.py` 固定：

```python
@dataclass(frozen=True)
class PublicSuitePreparationBindingV2(_JsonModel):
    schema_version: str
    source_catalog_digest: str
    acquisition_receipt_digest: str
    source_manifest_digest: str
    filter_manifest_digest: str
    preparation_packet_digest: str
    repository_catalog_digest: Optional[str]
    frozen_bundle_trust_digest: Optional[str]

@dataclass(frozen=True)
class SuiteCase(_JsonModel):
    task_id: str
    case_version: int
    path: str
    split: CaseSplit
    protocol_id: str
    dimensions: Tuple[CaseDimension, ...]
    raw_file_size_bytes: int
    raw_file_sha256: str
    canonical_case_digest: str
    eval_input_digest: str
    truth_completeness: TruthCompleteness
```

`SuiteManifest` 根加入 `wire_contract`；`SuiteSource` 加 `preparation_binding`。Core/private 固定为 null；public 必须按 Repository/Frozen 分支恰好绑定 repository catalog 或 frozen trust。Snapshot 保存 `manifest/wire_contract/cases(manifest_case/source/input)`，不得保存 truth、clarification answers、authority、evaluator context 或 Frozen 正文。

资源上限固定为 Manifest 16 MiB、Snapshot 256 MiB、65,536 Cases、Suite raw Case bytes 累计 512 MiB、每 Case 64 dimensions；先检查原始集合长度和字节数，再 canonical sort。

- [ ] **Step 4: 实现 v2 Run Config、Manifest、Receipt 与 immutable cross-binding**

在 `config.py` 和 `artifacts.py`：

- 将 Spec 8.6 列出的直接绑定 artifact 全部改为 v2；
- `EvalRunConfig` 保存 wire contract、suite preparation binding digest、AdapterCapabilitiesV2 snapshot/digest、Target kinds、Materializer protocol；
- `TrialManifest` 保存 canonical Case digest、EvalInput digest、wire contract、target kind、preparation binding 和 adapter capability digest；
- `StageReceipt.prepare` 保存 `TrialMaterializationManifest` ref/digest 和 `TargetAccess` projection；
- Run ID 只绑定 Agent-side execution identity；Evaluation ID 绑定完整 EvaluatorExecutionConfigV2；
- v1 `.eval-runs` 在 `load_run_config/load_run_manifest/load_submission/load_evaluation` 第一层稳定拒绝，不能创建目录、迁移或恢复；
- 从同一次有界 read 得到 raw hash、hydration 和 canonical digest；
- ArtifactStore 的 evaluator-only loader 继续 read-only，根不存在时不得创建。

- [ ] **Step 5: 迁移既有 artifact tests 并运行 GREEN**

Run: 与 Step 2 相同，并追加：

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_task12_artifact_repository_api.py `
  tests/eval/test_task12_rejudge.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-w1-t2-api'
```

Expected: PASS；parent/child schema、wire contract、Target kind、Materializer、preparation binding 任一漂移都 fail closed。

- [ ] **Step 6: 选择性提交**

```powershell
git add src/review_agent_eval/cases.py `
  src/review_agent_eval/datasets.py `
  src/review_agent_eval/config.py `
  src/review_agent_eval/artifacts.py `
  tests/eval/test_cases.py `
  tests/eval/test_datasets.py `
  tests/eval/test_config.py `
  tests/eval/test_artifacts.py `
  tests/eval/test_task12_artifact_repository_api.py `
  tests/eval/test_task12_rejudge.py
git diff --cached --name-only
git commit -m "feat(eval): bind protocol v2 run artifacts"
```

### Task 3：重新生成 18 个 Core Case、Golden 与 Suite

**Files:**

- Modify: `eval/authoring/build_core_suites.py`
- Modify: `eval/authoring/verify_core_regression.py`
- Modify: `eval/authoring/core_human_review.py`
- Modify: `eval/cases/core/README.md`
- Modify: `eval/cases/core/golden-index.json`
- Modify: `eval/cases/core/core-py-001/case.json`
- Modify: `eval/cases/core/core-py-002/case.json`
- Modify: `eval/cases/core/core-py-003/case.json`
- Modify: `eval/cases/core/core-py-004/case.json`
- Modify: `eval/cases/core/core-py-005/case.json`
- Modify: `eval/cases/core/core-py-006/case.json`
- Modify: `eval/cases/core/core-py-007/case.json`
- Modify: `eval/cases/core/core-py-008/case.json`
- Modify: `eval/cases/core/core-py-009/case.json`
- Modify: `eval/cases/core/core-py-010/case.json`
- Modify: `eval/cases/core/core-py-011/case.json`
- Modify: `eval/cases/core/core-py-012/case.json`
- Modify: `eval/cases/core/core-py-013/case.json`
- Modify: `eval/cases/core/core-py-014/case.json`
- Modify: `eval/cases/core/core-py-015/case.json`
- Modify: `eval/cases/core/core-py-016/case.json`
- Modify: `eval/cases/core/core-py-017/case.json`
- Modify: `eval/cases/core/core-py-018/case.json`
- Modify: `eval/cases/core/core-py-001/golden/duplicate.json`
- Modify: `eval/cases/core/core-py-001/golden/perfect.json`
- Modify: `eval/cases/core/core-py-001/golden/unsupported-evidence.json`
- Modify: `eval/cases/core/core-py-004/golden/unsupported-intent.json`
- Modify: `eval/cases/core/core-py-011/golden/contradicted-intent.json`
- Modify: `eval/cases/core/core-py-011/golden/empty.json`
- Modify: `eval/cases/core/core-py-012/golden/compound.json`
- Modify: `eval/cases/core/core-py-014/golden/bad-evidence-line.json`
- Modify: `eval/cases/core/core-py-014/golden/bad-evidence-path.json`
- Modify: `eval/cases/core/core-py-014/golden/bad-evidence.json`
- Modify: `eval/cases/core/core-py-015/golden/fabricated.json`
- Modify: `eval/cases/core/core-py-015/golden/judge-unknown.json`
- Modify: `eval/suites/core-regression/manifest.json`
- Modify: `eval/suites/core-capability/manifest.json`
- Modify: `eval/annotation-guidelines.md`
- Test: `tests/eval/test_core_suite.py`
- Test: `tests/eval/test_core_golden_submissions.py`
- Test: `tests/eval/test_core_authoring_security.py`
- Test: `tests/eval/test_core_human_review.py`
- Test: `tests/eval/test_core_promotion.py`

- [ ] **Step 1: 机械迁移生成器到唯一 v2 投影**

生成器必须执行下列确定映射：

- `input.repository + input.review_request` -> `input.review_target={kind:"repository",repository,review_request}`；
- 每个 expected Finding 增加 `metric_authority={severity_scorable:true,severity_authority:"expert_annotation",location_scorable:true,location_authority:"expert_annotation"}`；
- 每个 Case 增加 `review_evaluator_context={truth_contexts:[]}`；
- 18 个 Case 的 `case_version` 统一从 2 增加到 3，Case source/Suite version 固定为 `core-2026-07-21-v3`；
- 每个 Golden 增加 Case 的 `eval_input_digest` 与由固定 golden run/trial/attempt/replay binding 派生的 `target_materialization_id`；
- 每个 Golden Evidence 改成 v2 `source` tagged union并回指同一 materialization ID；
- 两个 Suite 使用 Repository `WireContractV2`、`preparation_binding=null`，重算 raw/canonical/input/Suite digest；
- 重新派生 blind packet、golden index、promotion baseline identity；旧 human approval 内容保留，但其 source Case digest 变化后必须标记为需要重新审阅，不能沿用旧批准身份。

- [ ] **Step 2: 运行生成并检查 diff 只包含预期制品**

Run:

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' eval/authoring/build_core_suites.py --write
git diff -- eval/authoring eval/cases/core eval/suites/core-regression eval/suites/core-capability eval/annotation-guidelines.md
```

Expected: 18 个 Case 全部为 `eval_case_v2`，12 个 Golden 全部为 `eval_submission_v2`，两个 Suite 全部为 `suite_manifest_v2`；Repository fixture 文件内容不变。

- [ ] **Step 3: 运行生成制品集中验证**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' eval/authoring/build_core_suites.py --check
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_core_suite.py `
  tests/eval/test_core_golden_submissions.py `
  tests/eval/test_core_authoring_security.py `
  tests/eval/test_core_human_review.py `
  tests/eval/test_core_promotion.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-w1-t3'
```

Expected: generator check 与可执行 Core 测试通过；两项真实外部门禁明确报告为未满足，不被 fake 数据替代。

- [ ] **Step 4: 选择性提交**

```powershell
git add eval/authoring `
  eval/cases/core `
  eval/suites/core-regression `
  eval/suites/core-capability `
  eval/annotation-guidelines.md `
  tests/eval/test_core_suite.py `
  tests/eval/test_core_golden_submissions.py `
  tests/eval/test_core_authoring_security.py `
  tests/eval/test_core_human_review.py `
  tests/eval/test_core_promotion.py
git diff --cached --name-only
git commit -m "test(eval): regenerate audited core protocol v2 assets"
```

### Wave 1 Gate

- [ ] `rg -n 'eval_(input|submission|case)_v1|suite_manifest_v1|eval_run_case_snapshot_v1' src/review_agent_eval eval/cases/core eval/suites` 只允许命中 v1 拒绝测试或明确历史说明。
- [ ] 三个根协议、Suite/Snapshot/Run/Trial/Receipt 都严格 v2 round-trip。
- [ ] 18 Case、12 Golden、2 Suite 重生成可复现。
- [ ] v1 root 与混合 parent/child 都稳定返回 `unsupported_protocol_version`。

---

# Wave 2：Target Materialization 与 Agent Boundary

### Task 4：建立通用 Materialization 与 RepositoryTargetMaterializer

**Files:**

- Create: `src/review_agent_eval/materialization.py`
- Modify: `src/review_agent_eval/repository.py`
- Modify: `src/review_agent_eval/artifacts.py`
- Modify: `src/review_agent_eval/config.py`
- Create: `tests/eval/test_materialization.py`
- Modify: `tests/eval/test_repository.py`
- Modify: `tests/eval/test_artifacts.py`

- [ ] **Step 1: 写 Materialization trust RED 契约**

覆盖：Target 文件在 manifest 后被替换、prepare receipt 指向另一 Trial/attempt、adapter capability digest 漂移、TargetAccess 绝对路径、Repository/Frozen protocol 混用。测试必须在 fake Adapter 调用计数仍为 0 时失败。

- [ ] **Step 2: 运行 RED**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_materialization.py tests/eval/test_repository.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-w2-t4-red'
```

Expected: FAIL，因为当前 WorkspaceManifest 不能表达通用 TrialMaterializationV2 和 TargetAccess。

- [ ] **Step 3: 实现共同协议、dispatcher 与 identity**

在 `materialization.py` 实现：

```python
MATERIALIZATION_SCHEMA_VERSION = "eval_trial_materialization_v2"
REPOSITORY_MATERIALIZER_PROTOCOL = "repository-materializer-v2"
FROZEN_CONTEXT_MATERIALIZER_PROTOCOL = "frozen-context-materializer-v2"

def materializer_for(
    wire_contract: WireContractV2,
    *,
    repository: RepositoryTargetMaterializer,
    frozen_context: FrozenContextTargetMaterializer,
) -> TargetMaterializer:
    expected = {
        ReviewTargetKind.REPOSITORY: REPOSITORY_MATERIALIZER_PROTOCOL,
        ReviewTargetKind.FROZEN_CONTEXT: FROZEN_CONTEXT_MATERIALIZER_PROTOCOL,
    }
    if expected[wire_contract.review_target_kind] != wire_contract.materializer_protocol:
        raise MaterializationContractError("wire contract target/materializer mismatch")
    return repository if wire_contract.review_target_kind is ReviewTargetKind.REPOSITORY else frozen_context
```

实现 `TargetAccess` 安全相对路径、`AgentVisibleFileBinding`、`TrialMaterializationManifest.create/from_json`、`TrialMaterialization` lease、replay union 和 materialization ID 全量 64 位 digest。

- [ ] **Step 4: 用现有 RepositoryPreparer 实现 RepositoryTargetMaterializer**

保持 Git preparation/replay 的既有安全逻辑，不重写 clone/object verification。Repository Materializer：

1. 验证 EvalInput/Trial/Suite/Wire/attempt/capability binding；
2. 从 cache-only PreparedRepository 创建独立 per-attempt Trial root；
3. 将 Agent-visible Repository Target 放在只读 `target/repository`，可写区固定为 `work`；
4. 计算每个可见文件 size/hash；
5. 打开同一 PreparedRepositoryReplay；
6. 构造 manifest 和 TargetAccess，再原子写 prepare receipt；
7. 返回 lease，close 时按 retention policy 清理。

`RepositoryPreparer.prepare()` 只用于 acquisition/preparation 控制面；正式 `run-agent` 的 Materializer 只能使用 `require_cached/open_replay`。

- [ ] **Step 5: 运行 GREEN 与 tamper regression**

Run: 与 Step 2 相同，并追加 `tests/eval/test_artifacts.py`。

Expected: PASS；Target 替换或 binding 漂移在 Agent 启动前失败，Repository replay 与 Agent Target 来自同一 prepared source。

- [ ] **Step 6: 选择性提交**

```powershell
git add src/review_agent_eval/materialization.py `
  src/review_agent_eval/repository.py `
  src/review_agent_eval/artifacts.py `
  src/review_agent_eval/config.py `
  tests/eval/test_materialization.py `
  tests/eval/test_repository.py `
  tests/eval/test_artifacts.py
git diff --cached --name-only
git commit -m "feat(eval): materialize repository review targets"
```

### Task 5：实现 FrozenContextTargetMaterializer 与只读 replay

**Files:**

- Create: `src/review_agent_eval/frozen_context.py`
- Modify: `src/review_agent_eval/materialization.py`
- Modify: `src/review_agent_eval/adapters/swe_prbench.py`
- Create: `tests/eval/test_frozen_context.py`
- Modify: `tests/eval/test_swe_prbench_adapter.py`

- [ ] **Step 1: 写 exact rendered bytes 与 trust closure RED**

测试以下值必须 fail closed：附加尾换行、BOM、CRLF/NFC 归一化、rendered bytes/hash 漂移、错误 bundle/record/source/filter/preparation digest、Trial 阶段缺 bundle 后尝试下载、TargetAccess 泄漏 bundle manifest 或 annotation。

- [ ] **Step 2: 运行 RED**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_frozen_context.py tests/eval/test_swe_prbench_adapter.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-w2-t5-red'
```

Expected: FAIL，因为现有 frozen bundle 不可作为 runnable Review Target。

- [ ] **Step 3: 实现 Frozen content store、trust 和 replay**

在 `frozen_context.py` 实现：

```python
@dataclass(frozen=True)
class PreparedFrozenContextReplay:
    bundle_id: str
    record_id: str
    context_ref: str
    context_format: str
    rendered_sha256: str
    rendered_utf8_bytes: int
    replay_binding_digest: str

    def read_exact(self) -> bytes: ...
    def read_lines(self, from_line: int, to_line: int) -> bytes: ...

class FrozenContextTargetMaterializer:
    protocol_id = "frozen-context-materializer-v2"

    def materialize(
        self,
        *,
        eval_input: EvalInput,
        run_config: EvalRunConfig,
        trial_manifest: TrialManifest,
        attempt: int,
        adapter_capabilities: AdapterCapabilitiesV2,
    ) -> TrialMaterialization: ...
```

具体行为：从 `.eval-data` 内容寻址对象打开 bundle；验证外部 frozen trust、catalog/acquisition/source/filter/record/preparation 全闭环；将 exact bytes create-only 发布到 `target/context.txt`；不添加尾换行；按 LF 计算 1-based 行；Agent 只看到 `context.txt` 与可写 `work/`；Replay 直接读取已验证 content object，不读 Trial workspace 副本。

- [ ] **Step 4: 升级 SWE frozen bundle 到 v2 trust root**

`swe_prbench_frozen_context_bundle_v2` 必须保存 record binding、content object ref、source/filter/preparation digest 和 bundle trust digest。保留 record 的 exact rendered 语义；删除“只能生成 bundle、不能运行”的 v1 限制分支。

- [ ] **Step 5: 运行 GREEN**

Run: 与 Step 2 相同。

Expected: PASS；相同 rendered bytes 产生相同 content identity，任一 trust/hash/bytes 漂移拒绝，Trial 中没有 acquisition 调用。

- [ ] **Step 6: 选择性提交**

```powershell
git add src/review_agent_eval/frozen_context.py `
  src/review_agent_eval/materialization.py `
  src/review_agent_eval/adapters/swe_prbench.py `
  tests/eval/test_frozen_context.py `
  tests/eval/test_swe_prbench_adapter.py
git diff --cached --name-only
git commit -m "feat(eval): materialize frozen review context"
```

### Task 6：切换 AdapterCapabilitiesV2、subprocess-json-v2 与 Runner

**Files:**

- Modify: `src/review_agent_eval/adapters/base.py`
- Modify: `src/review_agent_eval/adapters/agent_factory.py`
- Modify: `src/review_agent_eval/adapters/current_agent.py`
- Modify: `src/review_agent_eval/adapters/subprocess_agent.py`
- Modify: `src/review_agent_eval/adapters/__init__.py`
- Modify: `src/review_agent_eval/runner.py`
- Modify: `src/review_agent_eval/submission.py`
- Modify: `src/review_agent_eval/artifacts.py`
- Test: `tests/eval/test_agent_adapter.py`
- Test: `tests/eval/test_current_agent_adapter.py`
- Test: `tests/eval/test_runner.py`
- Test: `tests/eval/test_runner_failures.py`
- Test: `tests/eval/test_trace_capture_security.py`
- Test: `tests/eval/test_model_adapter_boundary.py`

- [ ] **Step 1: 写 capability/wire/identity RED 契约**

固定：`current-agent-cli-v2` 只支持 Repository；frozen Case 在创建 Run 前不兼容；subprocess envelope 中所有 path 为 Trial-root relative；输出必须回指 task/trial/input/materialization；capability digest 或 response identity drift 拒绝且不记 Agent failure。

- [ ] **Step 2: 运行 RED**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_agent_adapter.py `
  tests/eval/test_current_agent_adapter.py `
  tests/eval/test_runner.py `
  tests/eval/test_runner_failures.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-w2-t6-red'
```

Expected: FAIL，因为现有 Adapter 没有 capabilities，Runner 总是假设 Repository workspace。

- [ ] **Step 3: 实现 v2 Adapter public boundary**

- `AgentRunConfig.bind()` 加 wire/target/materialization/capability digest；
- `AdapterCapabilitiesV2` strict hydration 并进入 Run identity；
- `current-agent-cli-v2` capabilities 固定 Repository，继续把 ReviewRequest 映射到产品 `--intent/--focus`；
- `subprocess-json-v2` stdin exact keys 为 `schema_version/eval_input/trial_binding/target_access/materialization_id`；
- generic subprocess config 显式声明 target/evidence/clarification/trace 能力，不能按输出字段猜测；
- Adapter output 只接受 `eval_submission_v2`，并校验 input/materialization/trial identity；
- cancellation/timeout 继续由 Runner 管理，但不进入 Eval wire schema。

- [ ] **Step 4: 把 Runner 改为 kind-dispatch 与 per-attempt Materialization**

固定顺序：

```text
preflight capabilities
-> create immutable Run/Trial plan
-> start_trial(active attempt lease)
-> materializer_for(wire_contract).materialize(...)
-> write create-only prepare receipt
-> adapter.compatibility(...)
-> adapter.run(eval_input, target_access, work_root, ...)
-> validate v2 binding
-> terminal receipt
```

Materialization 缺失/漂移生成 Harness-owned `failed + harness_materialization_error`，Intent/Review 均为 null，不调用 Agent，不计 Agent failure。Agent timeout/invalid output/blocked 按 7.3 节 always-submission 终态处理。Resume 只能复用完整验证的同一 attempt materialization 或启动新 attempt；旧 worker 不能提交结果。

- [ ] **Step 5: 运行 GREEN 和边界回归**

Run: 与 Step 2 相同，并追加 trace/model boundary 两个测试文件。

Expected: PASS；Repository current Agent 正常运行，Frozen 只交给声明 frozen capability 的 Adapter，所有不兼容在 preflight 或 Harness integrity 层结束。

- [ ] **Step 6: 选择性提交**

```powershell
git add src/review_agent_eval/adapters/base.py `
  src/review_agent_eval/adapters/agent_factory.py `
  src/review_agent_eval/adapters/current_agent.py `
  src/review_agent_eval/adapters/subprocess_agent.py `
  src/review_agent_eval/adapters/__init__.py `
  src/review_agent_eval/runner.py `
  src/review_agent_eval/submission.py `
  src/review_agent_eval/artifacts.py `
  tests/eval/test_agent_adapter.py `
  tests/eval/test_current_agent_adapter.py `
  tests/eval/test_runner.py `
  tests/eval/test_runner_failures.py `
  tests/eval/test_trace_capture_security.py `
  tests/eval/test_model_adapter_boundary.py
git diff --cached --name-only
git commit -m "feat(eval): enforce protocol v2 agent boundary"
```

### Wave 2 Gate

- [ ] Repository 与 Frozen 都生成 `eval_trial_materialization_v2` 和 create-only prepare receipt。
- [ ] Agent-visible Target、可写 Work、truth/case bank、`.eval-data`、`.eval-runs` 物理分离。
- [ ] current Agent 的 frozen incompatibility 在 Trial 创建前可见；generic subprocess 的声明与实际 envelope 一致。
- [ ] Target replacement、resume stale lease、capability drift、absolute path 和 truth leakage 测试通过。

---

# Wave 3：Metric Authority、Evaluator Context 与 v2 Evaluator graph

### Task 7：升级 Evidence Replay 与 IntentEvaluationV2

**Files:**

- Modify: `src/review_agent_eval/evidence_checker.py`
- Modify: `src/review_agent_eval/intent_evaluator.py`
- Modify: `src/review_agent_eval/clarification.py`
- Modify: `src/review_agent_eval/judge_exports.py`
- Test: `tests/eval/test_evidence_checker.py`
- Test: `tests/eval/test_intent_evaluator.py`
- Test: `tests/eval/test_clarification_script.py`

- [ ] **Step 1: 写 Evidence immutable replay RED**

覆盖 Repository file/diff、Frozen line range、command attestation、existing-CI 五分支；错误 materialization ID、错误 context_ref、Trial workspace 伪造副本、hash/excerpt 漂移分别得到 invalid/missing，不得交给 Judge 修复。

- [ ] **Step 2: 运行 RED**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_evidence_checker.py `
  tests/eval/test_intent_evaluator.py `
  tests/eval/test_clarification_script.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-w3-t7-red'
```

Expected: FAIL，因为 Evidence 仍是扁平 v1 且 Intent evaluation artifact 仍为 v1。

- [ ] **Step 3: 实现 EvidenceSourceV2 replay dispatch**

`EvidenceIntegrityChecker` 构造函数改为绑定 `TrialMaterializationManifest + replay union + immutable command attestations + EvalInput`。每个 kind 只访问其授权 replay；Repository 不接受 frozen context ref，Frozen 不接受 revision/external CI；Judge context 只接收 Checker 返回的 canonical replay bytes。

- [ ] **Step 4: 升级 IntentEvaluationResult 到 v2**

保持现有 generated projection、同 dimension candidate、deterministic exact/normalized、semantic request、全局 assignment 和 clarification receipt 逻辑，完成以下切换：

- schema 为 `eval_intent_evaluation_v2`；
- 结果绑定 `eval_input_digest/target_materialization_id/submission_intent_digest/truth/script/receipt/evaluator_execution`；
- candidate/request/decision 全量预算与完整图重放不变；
- v1 evaluation artifact 第一层拒绝；
- `intent_truth.scorable=false` 重新验证唯一 canonical truth，不生成 Judge request；
- Judge failure 与语义 unknown 分开；
- `inferred` 继续只是 source，不自动扣分。

- [ ] **Step 5: 运行 GREEN**

Run: 与 Step 2 相同。

Expected: PASS；五种 Evidence replay 结果稳定，Intent v2 可从 immutable Submission/Truth/Receipt/Judge results 完整重放。

- [ ] **Step 6: 选择性提交**

```powershell
git add src/review_agent_eval/evidence_checker.py `
  src/review_agent_eval/intent_evaluator.py `
  src/review_agent_eval/clarification.py `
  src/review_agent_eval/judge_exports.py `
  tests/eval/test_evidence_checker.py `
  tests/eval/test_intent_evaluator.py `
  tests/eval/test_clarification_script.py
git diff --cached --name-only
git commit -m "feat(eval): replay v2 evidence and intent outcomes"
```

### Task 8：实现 authority-aware Location、truth-scoped context 与 Review/JudgeV2

**Files:**

- Modify: `src/review_agent_eval/match_location.py`
- Modify: `src/review_agent_eval/review_evaluator.py`
- Modify: `src/review_agent_eval/judge.py`
- Modify: `src/review_agent_eval/judge_exports.py`
- Modify: `src/review_agent_eval/review_exports.py`
- Modify: `src/review_agent_eval/orchestrator.py`
- Create: `tests/eval/test_metric_authority.py`
- Create: `tests/eval/test_evaluator_context.py`
- Modify: `tests/eval/test_location_matcher.py`
- Modify: `tests/eval/test_review_evaluator.py`
- Modify: `tests/eval/test_judge.py`
- Modify: `tests/eval/test_judge_rubrics.py`

- [ ] **Step 1: 写 Metric Authority 与 context 防泄漏 RED**

固定三类核心断言：

```python
def test_unscorable_truth_location_never_creates_location_failure(): ...
def test_unscorable_severity_never_enters_judge_or_weighted_recall(): ...
def test_swe_diff_hunk_is_visible_only_to_its_truth_finding_pair(): ...
```

并验证 diff_hunk 不进入 EvalInput、Submission Evidence、Location matcher、其他 truth pair、novel factuality 或 Evidence support request。

- [ ] **Step 2: 运行 RED**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_metric_authority.py `
  tests/eval/test_evaluator_context.py `
  tests/eval/test_location_matcher.py `
  tests/eval/test_review_evaluator.py `
  tests/eval/test_judge.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-w3-t8-red'
```

Expected: FAIL，因为现有 matcher 会把所有 truth severity/location 当作可评分，context bundle 没有 v2 provenance/allowed-task authority。

- [ ] **Step 3: 实现 authority-aware Location 与 Review candidate graph**

- Location candidates 始终保留 diagnostic；只有 `metric_authority.location_scorable=true` 且坐标满足 policy 才生成可评分 `LocationAuditRecord`；
- 最终 Line 得分只读取 confirmed Assignment 对应 truth 的 audit；
- Severity nullability 与 authority 在 hydration 和 evaluator 二次验证；
- expected/known-invalid canonical claim 冲突 fail closed；
- issue edge weight只包含 finding equivalence，不混入 Location、Severity、Evidence、Actionability；
- ReviewEvaluationResult schema 改为 `eval_review_evaluation_v2`，sealed construction 和 full replay 保持不变并加入 authority/context/materialization binding。
- `issue_match` 与 `evidence_integrity/evidence_support` 继续正交；Finding 找对但 Evidence 错误仍可 confirmed，但只有 confirmed + valid + supported 才是 publishable。

- [ ] **Step 4: 实现四个 v2 Blind Judge profile 与 context builder**

升级 `intent_equivalence/finding_equivalence/novel_factuality/evidence_support` profile、response 和 aggregate artifacts 到 v2。`eval_blind_judge_input_v2` 必须绑定 input/submission/materialization replay/truth/context/evaluator execution digest；tools 始终为空、tool choice 为 none、Provider identity 必须与 profile 一致。

Finding equivalence context builder 只执行：

```python
pair_context = case.review_evaluator_context.for_truth(truth_id)
sources = pair_context.sources_for(EvaluatorContextTask.FINDING_EQUIVALENCE)
blind_sources = immutable_replay_sources + sources
```

不允许按 Case 广播 context。Repository data 与 diff_hunk 都标记为 untrusted data，不能覆盖 system rubric。

- [ ] **Step 5: 运行 GREEN 与完整 Judge rubric 回归**

Run: 与 Step 2 相同，并追加 `tests/eval/test_judge_rubrics.py`、`tests/eval/test_model_adapter_boundary.py`。

Expected: PASS；AACR/SWE authority 行为正确，truth-scoped context 不泄漏，Review/Judge v2 可重放且 Judge failure fail closed。

- [ ] **Step 6: 选择性提交**

```powershell
git add src/review_agent_eval/match_location.py `
  src/review_agent_eval/review_evaluator.py `
  src/review_agent_eval/judge.py `
  src/review_agent_eval/judge_exports.py `
  src/review_agent_eval/review_exports.py `
  src/review_agent_eval/orchestrator.py `
  tests/eval/test_metric_authority.py `
  tests/eval/test_evaluator_context.py `
  tests/eval/test_location_matcher.py `
  tests/eval/test_review_evaluator.py `
  tests/eval/test_judge.py `
  tests/eval/test_judge_rubrics.py `
  tests/eval/test_model_adapter_boundary.py
git diff --cached --name-only
git commit -m "feat(eval): enforce metric authority and scoped judge context"
```

### Task 9：升级 Score、Summary、Inspection 与 partition compatibility

**Files:**

- Modify: `src/review_agent_eval/metrics.py`
- Modify: `src/review_agent_eval/metrics_exports.py`
- Modify: `src/review_agent_eval/report.py`
- Modify: `src/review_agent_eval/report_exports.py`
- Modify: `src/review_agent_eval/orchestrator.py`
- Modify: `src/review_agent_eval/artifacts.py`
- Test: `tests/eval/test_metrics.py`
- Test: `tests/eval/test_report.py`
- Test: `tests/eval/test_report_security_regressions.py`
- Test: `tests/eval/test_task12_rejudge.py`

- [ ] **Step 1: 实现 MetricsPolicyV2 与 authority coverage**

固定 policy token：

```text
severity-weight-policy-v2
assigned-truth-location-v2
metric-authority-policy-v2
count-as-missed-v2
```

`MetricsPolicy` 保存完整 severity weight、Line、Authority、failure outcome snapshot及 digest。Severity/Line contribution 携带 `eligible_truth_count/excluded_truth_count/authority_breakdown`；无 authority 为 `not_scorable`，不能变成 0。失败 Submission 的 non-null Outcome 继续正常评分；缺失 Outcome 才按 versioned failure policy 处理。

- [ ] **Step 2: 升级 ScoreCompatibilityKey 和三层 ScoreV2**

Compatibility 至少绑定 Run/Suite/Snapshot/Trial count、wire contract、protocol、truth completeness、novel policy、agent capability/isolation、clarification matcher、Evaluator execution、Intent/Review revision、Metric Authority profile。Case/Aggregate 验证 source score ID 与 digest，不只比较 trial ID；ratio 使用 ratio-of-sums；F1 引用 Precision/Recall 两侧 coverage。

- [ ] **Step 3: 升级 Summary/InspectionV2 和纯 Markdown render**

同一 Run 按完整 compatibility key partition；不同 protocol/wire/authority/isolation 不 roll-up。Inspection 复用 canonical artifacts，并通过 `eval_redacted_artifact_projection_v2` 隐去绝对路径、Trace value、URL credential/query、Frozen 正文。Markdown renderer 只接收已验证 Summary/Inspection，不读仓库、不调用模型、不重算指标。

- [ ] **Step 4: 运行集中验证**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_metrics.py `
  tests/eval/test_report.py `
  tests/eval/test_report_security_regressions.py `
  tests/eval/test_task12_rejudge.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-w3-t9'
```

Expected: PASS；不可评分与真实 0、missing、ungraded、zero denominator 分开；跨 cohort 只并列展示。

- [ ] **Step 5: 选择性提交**

```powershell
git add src/review_agent_eval/metrics.py `
  src/review_agent_eval/metrics_exports.py `
  src/review_agent_eval/report.py `
  src/review_agent_eval/report_exports.py `
  src/review_agent_eval/orchestrator.py `
  src/review_agent_eval/artifacts.py `
  tests/eval/test_metrics.py `
  tests/eval/test_report.py `
  tests/eval/test_report_security_regressions.py `
  tests/eval/test_task12_rejudge.py
git diff --cached --name-only
git commit -m "feat(eval): score and report authority-aware v2 outcomes"
```

### Wave 3 Gate

- [ ] Intent、Review、Evidence 可单独重放，Judge failure/unknown/ungraded 不互相冒充。
- [ ] AACR Severity 不可评分、Location 可评分；SWE 两者都不可评分；Core 两者可评分。
- [ ] SWE diff_hunk 只进入对应 truth 的 finding-equivalence request。
- [ ] Summary/Inspection/Markdown 不泄漏 truth、raw trace、宿主路径、credential 或完整 Frozen 正文。

---

# Wave 4：Public Acquisition 与最终 Public Adapter 接入

### Task 10：实现 trusted catalog、local-import、pinned-download 与 Source Object

**Files:**

- Create: `src/review_agent_eval/public_acquisition.py`
- Modify: `src/review_agent_eval/adapters/_public.py`
- Create: `tests/eval/test_public_acquisition.py`
- Modify: `tests/eval/test_public_adapter_common.py`
- Create: `tests/eval/fixtures/public_acquisition/dataset-catalog.json`
- Create: `tests/eval/fixtures/public_acquisition/repository-catalog.json`
- Create: `tests/eval/fixtures/public_acquisition/payload/sample.jsonl`

- [ ] **Step 1: 写 acquisition trust 与 archive 安全 RED**

覆盖：缺外部 expected catalog digest、catalog/hash/size/license 漂移、现场扫描自签、URL userinfo、非 HTTPS、host/port 变化、redirect、超预算、zip-slip、tar symlink/hardlink/device、大小写/NFC 路径碰撞、partial publication、resume bytes 漂移。

- [ ] **Step 2: 运行 RED**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_public_acquisition.py tests/eval/test_public_adapter_common.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-w4-t10-red'
```

Expected: FAIL，因为项目还没有独立 acquisition 控制面和 externally pinned catalog。

- [ ] **Step 3: 实现 catalog/receipt/source object 模型**

实现严格模型：

```python
PublicDatasetCatalogV1
PublicDatasetAcquisitionReceiptV1
PublicRepositoryCatalogV1
RepositoryTrustAnchorV1
LocalRepositoryMirrorReceiptV1
PublicDatasetPreparationReceiptV2
PublicDatasetPreparationPacketV2
SourceObjectManifest
PublicAcquisitionResult
VerifiedSourceObject
PinnedHttpsTransport
```

Catalog 固定 URL/origin/revision/license/size/SHA-256/logical extraction map/global quota。Acquisition receipt 保存 catalog digest、Source Object digest、method/result，不保存 credential、DNS 结果或宿主绝对路径。local-import 和 pinned-download 对相同 bytes 产生同一 Source Object identity。Dataset license 与底层 Repository license 分开保存和验证，任何一方都不能替代另一方。

- [ ] **Step 4: 实现两种 acquisition method 与 create-only publication**

公开 API 固定为：

```python
def import_local_source(
    *,
    catalog_bytes: bytes,
    expected_catalog_digest: str,
    transfer_root: Path,
    data_root: Path,
    resume: bool,
) -> PublicAcquisitionResult: ...

def download_pinned_source(
    *,
    catalog_bytes: bytes,
    expected_catalog_digest: str,
    data_root: Path,
    transport: PinnedHttpsTransport,
    resume: bool,
) -> PublicAcquisitionResult: ...
```

先验证外部 digest，再解析 catalog；只提取 map 中列出的 regular files；每个 member 同时验证 path/size/hash；staging 与目标同 volume；全部 fsync/验证后原子 no-overwrite publish。Transport 固定 HTTPS host/port、禁止 redirect/userinfo、限制 response bytes/deadline，并将 Provider egress 与 acquisition client 完全分开。

- [ ] **Step 5: 运行 GREEN**

Run: 与 Step 2 相同。

Expected: PASS；两种方法对相同 bytes 得到相同 Source Object identity，所有 trust/path/quota 失败不留下半成品。

- [ ] **Step 6: 选择性提交**

```powershell
git add src/review_agent_eval/public_acquisition.py `
  src/review_agent_eval/adapters/_public.py `
  tests/eval/test_public_acquisition.py `
  tests/eval/test_public_adapter_common.py `
  tests/eval/fixtures/public_acquisition
git diff --cached --name-only
git commit -m "feat(eval): acquire pinned public benchmark sources"
```

### Task 11：发布 AACR 与 SWE native/frozen v2 Suite

**Files:**

- Modify: `src/review_agent_eval/adapters/_public.py`
- Modify: `src/review_agent_eval/adapters/aacr_bench.py`
- Modify: `src/review_agent_eval/adapters/swe_prbench.py`
- Modify: `tests/eval/test_public_adapter_common.py`
- Modify: `tests/eval/test_aacr_adapter.py`
- Modify: `tests/eval/test_swe_prbench_adapter.py`
- Modify: `tests/eval/fixtures/public_datasets/aacr/valid/source_manifest.json`
- Modify: `tests/eval/fixtures/public_datasets/swe_prbench/source_manifest.json`

- [ ] **Step 1: 把公共 Adapter 输入改为 verified Source Object + preparation packet**

Adapter 不再信任调用方目录或自行发现 sidecar。`prepare_aacr_bench()` 与 `prepare_swe_prbench()` 接收 verified Source Object、filter、catalog/acquisition/repository-or-frozen trust binding和输出 Suite root；PublicPreparationReceiptV2 回指完整 preparation packet。

- [ ] **Step 2: 完成 AACR v2 映射**

- Repository Target + Repository wire contract；
- positive -> expected，negative -> known invalid；
- `location_scorable=true/upstream_annotation`；
- `severity_scorable=false/severity=null`；
- 200 Case / 1502 expected / 639 known-invalid / 4 isolated；
- raw 1505/640 与 scorable/isolated 分母同时进入 receipt/report；
- full 与 Python subset 使用不同 filter/preparation/Suite identity。

- [ ] **Step 3: 完成 SWE native 与 official frozen v2 映射**

- 两种 Suite 都使用 `human_observed + verify + intent_truth.scorable=false`；
- Severity/Location 均不可评分，severity=null，保留上游 path/line 仅供语义/审计；
- native Case 使用真实 canonical remote URL/base/head 和 Repository catalog binding；
- frozen Case 使用 FrozenContextReviewTarget 和 frozen bundle trust binding，无 ReviewRequest；
- diff_hunk 只进入对应 truth 的 `review_evaluator_context`；
- 345 representable / 4 empty truth / 1 oversized isolation；native 1673 expected；
- A/B/C、Type1/2/3、language/difficulty、protocol_id 保留为 Case dimensions/report partition。

- [ ] **Step 4: 运行 local fixture 与固定全量口径验证**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_public_adapter_common.py `
  tests/eval/test_aacr_adapter.py `
  tests/eval/test_swe_prbench_adapter.py `
  tests/eval/test_metric_authority.py `
  tests/eval/test_evaluator_context.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-w4-t11'
```

Expected: PASS；fixture 与正式数据使用同一 parser/mapping path，native/frozen 不混合为一个 Suite。

全量 smoke 使用外部预先固定的 source/catalog roots；若环境未提供这些 roots，本 Task 只能标记代码通过，不能声称全量数据门禁通过。实际运行必须验证上述精确分母和 SWE source manifest digest `75921f6ed330d7406ea966a8915443fc666117ea259ec8e609ae1f2724b907ab`。

- [ ] **Step 5: 选择性提交**

```powershell
git add src/review_agent_eval/adapters/_public.py `
  src/review_agent_eval/adapters/aacr_bench.py `
  src/review_agent_eval/adapters/swe_prbench.py `
  tests/eval/test_public_adapter_common.py `
  tests/eval/test_aacr_adapter.py `
  tests/eval/test_swe_prbench_adapter.py `
  tests/eval/fixtures/public_datasets
git diff --cached --name-only
git commit -m "feat(eval): publish authority-aware public v2 suites"
```

### Task 12：完成 prepare-public 与 v2 CLI lifecycle

**Files:**

- Modify: `src/review_agent_eval/cli.py`
- Modify: `src/review_agent_eval/__main__.py`
- Modify: `src/review_agent_eval/adapters/agent_factory.py`
- Modify: `pyproject.toml`
- Modify: `tests/eval/test_cli.py`
- Modify: `tests/eval/test_cli_failures.py`
- Modify: `tests/eval/test_cli_security.py`

- [ ] **Step 1: 加入最终命令树，不保留占位命令**

```text
review-agent-eval prepare-public local-import
review-agent-eval prepare-public pinned-download
review-agent-eval prepare
review-agent-eval run-agent
review-agent-eval evaluate
review-agent-eval compare
review-agent-eval inspect
review-agent-eval calibrate
```

`compare/calibrate` 的 handler 在 Task 13 接入；本 Task 只在对应实现已存在的同一 Wave 5 commit 中向用户公开，避免不可用占位。Task 12 先完成其 parser/helper 的私有 wiring tests。

- [ ] **Step 2: 固定 stage authority 与 exit taxonomy**

- `prepare-public` 是唯一可构造 acquisition client 的 command；
- `prepare` 只验证 Suite/`.eval-data` cache、做 capability preflight、冻结 Run/Trial plan；
- `run-agent` 只 per-attempt materialize 和生成 Submission；
- `evaluate` 只读既有 Submission并写新 evaluation namespace；
- `inspect` 只读 canonical redacted projection；
- `run-agent/evaluate/compare/inspect/calibrate` 不得 import 或实例化 acquisition client；
- exit code 固定 `2 usage / 10 precondition / 11 conflict / 12 integrity / 13 operational`；
- JSON 输出继续使用稳定 CLI envelope，但内部 artifact schema 全部 v2。

- [ ] **Step 3: 运行 CLI 与安全验证**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_cli.py `
  tests/eval/test_cli_failures.py `
  tests/eval/test_cli_security.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-w4-t12'
```

Expected: PASS；acquisition 权限不进入 Trial/Judge，v1 Run 无法 resume/evaluate/inspect。

- [ ] **Step 4: 选择性提交**

```powershell
git add src/review_agent_eval/cli.py `
  src/review_agent_eval/__main__.py `
  src/review_agent_eval/adapters/agent_factory.py `
  pyproject.toml `
  tests/eval/test_cli.py `
  tests/eval/test_cli_failures.py `
  tests/eval/test_cli_security.py
git diff --cached --name-only
git commit -m "feat(eval): expose trusted protocol v2 lifecycle"
```

### Wave 4 Gate

- [ ] local-import 与 pinned-download 都需要调用方传入外部 expected catalog digest。
- [ ] Trial/evaluate/compare 无 acquisition client、无 dataset/repository 任意网络权限。
- [ ] AACR、SWE native、SWE frozen 使用不同 Suite identity 和不可混合的 report partition。
- [ ] 全量数据 smoke 的 raw/scorable/isolated 分母与固定 digest 已保存到 receipt/report。

---

# Wave 5：重复 Trial、比较、校准、门禁与端到端回归

### Task 13：实现 repeated trials、paired compare、calibration 与 Regression Gate

**Files:**

- Create: `src/review_agent_eval/comparison.py`
- Create: `src/review_agent_eval/calibration.py`
- Create: `src/review_agent_eval/gates.py`
- Modify: `src/review_agent_eval/runner.py`
- Modify: `src/review_agent_eval/metrics.py`
- Modify: `src/review_agent_eval/report.py`
- Modify: `src/review_agent_eval/cli.py`
- Create: `tests/eval/test_repeated_trials.py`
- Create: `tests/eval/test_comparison.py`
- Create: `tests/eval/test_calibration.py`
- Create: `tests/eval/test_regression_gates.py`

- [ ] **Step 1: 实现多 Trial 与稳定性指标**

每个 Trial 保留独立 Submission/Evaluation/Score；不挑最好一次。`pass@1` 使用预先版本化 Case-pass rubric；`pass^k` 表示 k 次全部通过；研究性 pass@k 单独标识。failed/blocked/invalid-output 和 Judge coverage 不能从分母消失。

- [ ] **Step 2: 实现 strict compatible paired comparison**

公开 API：

```python
def compare_runs(
    baseline: VerifiedRunEvaluation,
    candidate: VerifiedRunEvaluation,
    policy: ComparisonPolicyV1,
) -> RunComparison: ...
```

只允许相同 Case/version/trial count、wire contract、protocol、truth completeness、novel policy、authority profile、Evaluator/Judge/rubric 和 isolation profile。输出 case-level improved/regressed/unchanged、均值、离散程度、固定 seed/次数的 paired bootstrap interval 和 Judge coverage delta。不可兼容返回 `not_comparable`，不强行算差值。

- [ ] **Step 3: 实现 calibration 与 blind review queue**

导入有 source digest 的人工 claim/Finding/Evidence label；计算 agreement、confusion matrix、Cohen's kappa；保存 calibration set/rubric/Judge model/version。review queue 包含 unknown、high/critical fabricated、deterministic/Judge conflict 和固定 seed 随机抽样，输出不含被测模型身份。

- [ ] **Step 4: 实现版本化 Regression Gate**

Policy 在 candidate 运行前冻结并进入 digest。逐项检查 Intent pass、critical/high miss、precision/recall allowed regression、fabricated rate、Evidence validity、Agent failure、cost budget；每个失败阈值返回对应 Case/Trial，不生成 Overall Score。

Private Held-out 只通过 canonical private `SuiteManifestV2` 和权限受控 CaseBank 接入；比较/门禁只输出聚合与 opaque Case identity，不把 private Case、truth 或 raw context复制到普通报告或 Git。

- [ ] **Step 5: 接通 compare/calibrate CLI 并运行测试**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_repeated_trials.py `
  tests/eval/test_comparison.py `
  tests/eval/test_calibration.py `
  tests/eval/test_regression_gates.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-w5-t13'
```

Expected: PASS；不可兼容 Run 不比较，失败 Trial 与 missing Judge coverage 均可见，Gate 逐阈值解释结果。

- [ ] **Step 6: 选择性提交**

```powershell
git add src/review_agent_eval/comparison.py `
  src/review_agent_eval/calibration.py `
  src/review_agent_eval/gates.py `
  src/review_agent_eval/runner.py `
  src/review_agent_eval/metrics.py `
  src/review_agent_eval/report.py `
  src/review_agent_eval/cli.py `
  tests/eval/test_repeated_trials.py `
  tests/eval/test_comparison.py `
  tests/eval/test_calibration.py `
  tests/eval/test_regression_gates.py
git diff --cached --name-only
git commit -m "feat(eval): compare calibrated v2 runs and enforce gates"
```

### Task 14：Repository/Frozen E2E、重评、恢复与安全故障注入

**Files:**

- Create: `tests/eval/test_e2e_repository_v2.py`
- Create: `tests/eval/test_e2e_frozen_v2.py`
- Modify: `src/review_agent_eval/materialization.py`
- Modify: `src/review_agent_eval/frozen_context.py`
- Modify: `src/review_agent_eval/public_acquisition.py`
- Modify: `src/review_agent_eval/runner.py`
- Modify: `src/review_agent_eval/artifacts.py`
- Modify: `src/review_agent_eval/orchestrator.py`
- Modify: `src/review_agent_eval/report.py`
- Modify: `src/review_agent_eval/cli.py`
- Modify: `src/review_agent_eval/adapters/subprocess_agent.py`
- Modify: `tests/eval/test_runner_failures.py`
- Modify: `tests/eval/test_cli_security.py`
- Modify: `tests/eval/test_report_security_regressions.py`
- Modify: `tests/test_architecture_boundaries.py`

- [ ] **Step 1: 完成 Repository 全链 E2E**

使用 Core fixture + scripted Agent/Judge 完成 `prepare -> run-agent -> evaluate -> inspect`；用不同 EvaluatorExecutionConfig 再执行一次 evaluate，确认 Submission digest 不变、两个 evaluation namespace 都保留、Report 可重放。

- [ ] **Step 2: 完成 Frozen 全链 E2E**

使用本地 SWE fixture + frozen-capable subprocess Adapter 完成相同链路；确认 target/context exact bytes、Frozen Evidence replay、truth-scoped diff_hunk、Severity/Location not-scorable 和 current-agent incompatibility。

- [ ] **Step 3: 写并观察安全边界 RED**

故障注入：timeout、invalid JSON、output overflow、Judge failure、bad Evidence、Target mutation、truth path probing、prompt injection、symlink/junction/reparse/hardlink、artifact tamper、stale attempt、catalog drift、cross-Run Submission、absolute path和 credential-bearing URL。每个失败必须在正确 taxonomy 层结束且不发布半成品。

- [ ] **Step 4: 修复实际暴露的 bug 并运行 E2E GREEN**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval/test_e2e_repository_v2.py `
  tests/eval/test_e2e_frozen_v2.py `
  tests/eval/test_runner_failures.py `
  tests/eval/test_cli_security.py `
  tests/eval/test_report_security_regressions.py `
  tests/test_architecture_boundaries.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-w5-t14'
```

Expected: PASS；Evaluator 不导入产品 Runtime/Session/Memory/Risk，Agent/Judge/acquisition 三类权限隔离。

- [ ] **Step 5: 选择性提交**

只 stage 本 Task 已列出的边界文件和测试；未变化路径不会进入 index。先用 `git diff --cached --name-only` 核对，不使用 `git add -A`：

```powershell
git add tests/eval/test_e2e_repository_v2.py `
  tests/eval/test_e2e_frozen_v2.py `
  src/review_agent_eval/materialization.py `
  src/review_agent_eval/frozen_context.py `
  src/review_agent_eval/public_acquisition.py `
  src/review_agent_eval/runner.py `
  src/review_agent_eval/artifacts.py `
  src/review_agent_eval/orchestrator.py `
  src/review_agent_eval/report.py `
  src/review_agent_eval/cli.py `
  src/review_agent_eval/adapters/subprocess_agent.py `
  tests/eval/test_runner_failures.py `
  tests/eval/test_cli_security.py `
  tests/eval/test_report_security_regressions.py `
  tests/test_architecture_boundaries.py
git diff --cached --name-only
git commit -m "test(eval): verify protocol v2 end to end"
```

### Task 15：删除活动 v1 分支、完成文档与全量验收

**Files:**

- Modify: `src/review_agent_eval/__init__.py`
- Modify: `src/review_agent_eval/judge_exports.py`
- Modify: `src/review_agent_eval/metrics_exports.py`
- Modify: `src/review_agent_eval/report_exports.py`
- Modify: `src/review_agent_eval/review_exports.py`
- Modify: `src/review_agent_eval/adapters/__init__.py`
- Modify: `tests/eval/test_protocol_v2_cutover.py`
- Create: `docs/eval-system.md`
- Modify: `.gitignore` only if `.eval-data/`、`.eval-runs/`、`.eval-workspaces/` entries are missing
- Modify: `docs/superpowers/specs/2026-07-16-core-code-review-eval-system-design.md` only for implementation status and links
- Modify: `docs/superpowers/plans/2026-07-21-eval-protocol-v2.md` only for completion evidence

- [ ] **Step 1: 做活动 v1 代码审计并删除正向 parser/constructor/export**

```powershell
rg -n 'eval_(input|submission|case|run_config|run_manifest|trial_manifest|intent_evaluation|review_evaluation|trial_score|case_score|aggregate_score|run_report_summary|trial_inspection)_v1|suite_manifest_v1|eval_run_case_snapshot_v1' `
  src/review_agent_eval eval tests/eval
```

允许命中：稳定拒绝测试、迁移说明、Git 历史文档、Spec 明确保留的独立叶子 token。任何活动 factory、Loader、resume、rejudge、CLI、fixture 或 public output 命中都必须删除或改为 v2。

- [ ] **Step 2: 编写最终用户文档**

`docs/eval-system.md` 必须包含：

- Core 与 public optional dependency 安装；
- external catalog digest、local-import、pinned-download；
- Repository/Frozen prepare 与 Adapter capability；
- run-agent/evaluate/re-evaluate/inspect/compare/calibrate；
- Intent/Review/Evidence/Authority 指标解释；
- protocol/authority/isolation partition 与 leaderboard 限制；
- artifact 目录、create-only/resume、安全和隐私；
- fake Judge 与真实 semantic calibration 的区别；
- Task 13 真人审阅和真实模型 baseline 的未满足状态。

- [ ] **Step 3: 运行完整 Eval suite**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/eval `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-final-eval'
```

Expected: exit code 0；只有明确的平台能力或真实外部门禁 skip，不允许协议兼容 skip。

- [ ] **Step 4: 运行产品侧架构与核心回归**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  tests/test_architecture_boundaries.py `
  tests/test_models.py `
  tests/test_context.py `
  tests/test_hydration.py `
  tests/test_risk.py `
  tests/test_brief.py `
  tests/test_pipeline.py `
  tests/test_resume.py `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-final-product'
```

Expected: exit code 0；正常 `review-agent` review/resume/memory 行为未因 Eval v2 改变。

- [ ] **Step 5: 运行全项目测试**

```powershell
& 'D:\Anaconda\envs\MINIST\python.exe' -m pytest `
  -q -p no:cacheprovider --basetemp 'D:\tmp\code-review-agent\v2-final-all'
```

Expected: exit code 0。pytest cleanup warning 需单独记录，但不能替代真实退出码。

- [ ] **Step 6: 更新实现状态与选择性提交**

只在上述验证真实通过后记录命令、退出码、Case/Suite/dataset count、平台 skip 和外部门禁状态：

```powershell
git add docs/eval-system.md `
  docs/superpowers/specs/2026-07-16-core-code-review-eval-system-design.md `
  docs/superpowers/plans/2026-07-21-eval-protocol-v2.md `
  tests/eval/test_protocol_v2_cutover.py `
  src/review_agent_eval/__init__.py `
  src/review_agent_eval/adapters/__init__.py `
  src/review_agent_eval/judge_exports.py `
  src/review_agent_eval/metrics_exports.py `
  src/review_agent_eval/report_exports.py `
  src/review_agent_eval/review_exports.py
git diff --cached --name-only
git commit -m "docs(eval): complete protocol v2 operating guide"
```

### Wave 5 Gate

- [ ] Repository/Frozen 两条 E2E 和同 Submission 多次独立重评通过。
- [ ] repeated trials、compare、calibrate、Regression Gate 均有 source-bound versioned artifact。
- [ ] v1 root 只存在于稳定拒绝测试、历史文档或 Git 历史。
- [ ] Eval 全量、产品核心回归和全项目测试退出码均为 0。
- [ ] 未完成的真人/真实模型门禁明确可见，没有合成记录冒充。

---

## 4. Spec Coverage Map

| Spec 章节 | 实施 Task |
|---|---|
| 3 黑盒边界、Adapter | Task 6、Task 14 |
| 4 总体架构 | Task 1–15 |
| 5 状态术语 | Task 1、Task 6 |
| 6 EvalInput v2 | Task 1、Task 4、Task 5 |
| 7 EvalSubmission v2 | Task 1、Task 6、Task 7 |
| 8 EvalCase v2、canonical JSON、全量切换 | Task 1、Task 2、Task 3、Task 15 |
| 9 Ground Truth 完整度 | Task 1、Task 9、Task 11 |
| 10 Suite/AACR/SWE/Private | Task 2、Task 3、Task 11、Task 13 |
| 11 Runner、隔离、Clarification | Task 4–7、Task 14 |
| 12 Intent Evaluator | Task 7 |
| 13 Review/Evidence/Location/Assignment | Task 7、Task 8 |
| 14 Judge | Task 8、Task 13、Task 14 |
| 15 多 Trial 与配对比较 | Task 13 |
| 16 Metrics/Report/Gate | Task 9、Task 13 |
| 17 Run Artifact | Task 2、Task 4–6、Task 9 |
| 18 CLI | Task 10–13 |
| 19 数据与安全 | Task 4、Task 5、Task 10、Task 12、Task 14 |
| 20 实现批次 | Wave 1–5 |
| 21 明确不做 | 全局执行约束、Task 15 v1 audit |
| 22 成功定义 | Wave 5 Gate 与最终验证 |

## 5. Plan 自检

- [ ] **Spec coverage:** 上表每个设计章节都映射到至少一个具体 Task；Repository/Frozen、Intent/Review、Authority、Public Acquisition、CLI、比较/校准/门禁和安全均有落点。
- [ ] **Placeholder scan:** 运行下列命令，输出必须为空；Protocol 的 `...` 只允许出现在 Python `Protocol` 方法体或公开签名示意中。

```powershell
rg -n 'T[B]D|T[O]DO|implement l[a]ter|fill in d[e]tails|后续处[理]|稍后实[现]|类似 T[a]sk' `
  docs/superpowers/plans/2026-07-21-eval-protocol-v2.md
```

- [ ] **Type consistency:** 全文统一使用 `ReviewTargetKind`、`WireContractV2`、`TargetAccess`、`TrialMaterializationManifest`、`AdapterCapabilitiesV2`、`MetricAuthority`、`ReviewEvaluatorContext`、`EvalSubmission.target_materialization_id`；不存在同义 `workspace_binding_id` 作为 v2 外部协议。
- [ ] **Version consistency:** 活动根全部为 v2；允许保留 v1 的叶子协议只出现在 2.4 白名单。
- [ ] **Test policy consistency:** 每 Wave 至少一个高风险 RED/GREEN；机械迁移、生成制品、展示和文档没有被拆成无价值逐函数 TDD。
- [ ] **Commit safety:** 每个 Task 都使用选择性 `git add` 和 `git diff --cached --name-only`；没有 `git add -A`、临时目录清理或工作区 destructive command。

## 6. 完成定义

只有同时满足下列条件，才能把 Eval Protocol v2 实现标记完成：

1. 活动 Loader/Runner/Evaluator/CLI 只接受完整 v2 graph，v1 稳定拒绝且不迁移；
2. Repository/Frozen Target 都有可信 Materialization、同源 Replay 和 isolated Agent boundary；
3. Intent、Review、Evidence、Severity/Location Authority 和 Judge context 都可完整重放；
4. AACR/SWE 的 public acquisition、native/frozen cohort 和精确分母可审计；
5. repeated trials、paired compare、calibration、Regression Gate 和无 Overall Score 报告可运行；
6. E2E、安全故障注入、Eval suite、产品核心回归和全项目测试退出码为 0；
7. Task 13 真人审阅与真实模型 baseline 若仍未完成，必须继续作为外部门禁报告，不能被代码完成状态掩盖。

# Review Session Memory、Resume 与 Revision Lineage 设计

日期：2026-07-10  
状态：待用户确认  
上位设计：`docs/superpowers/specs/2026-06-22-evidence-driven-multi-agent-code-review-design.md`

## 1. 目标

把当前仅能打印 checkpoint 摘要的 `resume`，升级为真正可恢复、可审计、revision-safe 的本地 Review Runtime。

完成后，系统必须能够：

1. 将一次 Review 的请求、解析后 revision、运行配置、阶段状态、artifact 和 lineage 保存为 Review Session。
2. 进程中断或阶段失败后，从最后一个可信 checkpoint 继续，而不是无条件从头运行。
3. 恢复前验证仓库身份、Base/Head commit、artifact 完整性和 Observation revision。
4. 对已完成 Session 保持审计模式，不重复调用模型或覆盖原结果。
5. Base 或 Head 漂移时不把旧 evidence 混入新 Review，而是创建带 parent lineage 的新 Session。
6. 重复执行同一个 `resume` 保持幂等，不重复创建相同的增量 Session，也不重复运行已经完成且有效的阶段。

## 2. 非目标

本设计不包含：

- GitHub、GitLab 或 Bitbucket PR 接入。
- 云端分布式任务队列。
- 跨机器 Session 同步。
- Durable Project Memory 和 Review Feedback Memory。
- Eval Harness。
- 自动修改被审查仓库。
- 保存 API Key、访问令牌或模型隐藏推理。

## 3. 当前实现缺口

当前 `src/review_agent/cli.py::_run_review()` 是单体顺序流程。它会持续写 `state.json` 和多个 JSON artifact，但存在以下问题：

- `state.json` 保存的是用户输入的 `main`、`HEAD` 等 revision 表达式，不是解析后的 commit SHA。
- `FAILED` 会覆盖原运行阶段，无法知道最后一个可信完成阶段。
- `resume` 只读取并打印 `state.json`、`request.json`，不会恢复执行。
- artifact 没有 hash、schema version、revision binding 或生产阶段信息。
- `ObservationStore` 只能写，不能从已有 `observations.jsonl` 恢复并验证 raw artifact。
- Reviewer、Reconciliation、Completion 等 typed state 缺少统一反序列化入口。
- provider/model/mode/loop 等运行配置没有作为可恢复配置持久化。
- checkpoint 写入不是原子的，进程中断可能留下半个 JSON 文件。

因此，不能在现有 `_run_resume()` 上继续叠加条件分支。必须把运行流程拆成可重入阶段，并引入一个权威 Session Manifest。

## 4. 核心不变量

### 4.1 Session 绑定不可变 commit

每个 Session 在创建时解析并保存：

```text
requested_base_revision -> resolved_base_sha
requested_head_revision -> resolved_head_sha
```

所有 Diff、Repository Intelligence、Observation、Finding 和报告都绑定 resolved SHA，不依赖恢复时工作区的当前状态。

### 4.2 Parent Session 不可变

恢复过程中发现 revision 漂移时：

- 不修改 parent Session 的 request、revision、Observation、Finding 或报告。
- 创建 child Session。
- child 记录 `parent_review_id`、`root_review_id` 和 revision change 类型。

### 4.3 不跨 revision 混用 Evidence

child Session 不把 parent 的 Observation ID、Finding 或 Completion 直接放进当前 evidence 集合。

parent 只用于：

- lineage 与审计链接。
- 计算 `parent_head_sha..new_head_sha` 的增量变更地图。
- 解释为什么创建了 child Session。

child 的最终 Review 仍以新的 resolved base/head 重新生成事实和证据。

### 4.4 Completed Phase 必须可证明

阶段只有同时满足以下条件才算 completed：

- Session Manifest 标记阶段 completed。
- 必需 artifact 存在。
- artifact hash 与 manifest 一致。
- artifact schema 可解析。
- revision binding 与 Session 一致。

任一条件不满足，该阶段和所有下游阶段都失效，恢复从最早失效阶段重新运行。

### 4.5 阶段重跑必须幂等

阶段输出写入临时文件，校验完成后原子替换目标文件。阶段重跑可以覆盖自己尚未提交或已失效的输出，但不能修改上游已验证 artifact。

## 5. 架构

```text
CLI
  -> ReviewSessionService
       -> RevisionResolver
       -> SessionStore
       -> ArtifactValidator
       -> ReviewPipeline
            -> PreflightStage
            -> RepositoryIntelligenceStage
            -> ReviewerStage
            -> ReconciliationStage
            -> CompletionStage
            -> FinalRiskStage
            -> ReportingStage
       -> ResumePlanner
       -> IncrementalReviewPlanner
```

### 5.1 ReviewSessionService

负责创建、加载、恢复和派生 Session。CLI 不直接判断从哪个阶段继续。

公共接口：

```python
class ReviewSessionService:
    def create(self, request: ReviewRequest, config: ReviewExecutionConfig) -> SessionManifest: ...
    def inspect(self, review_id: str) -> SessionInspection: ...
    def resume(self, review_id: str) -> ResumeResult: ...
```

### 5.2 RevisionResolver

只通过 Git 读取 commit：

```python
class RevisionResolver:
    def repository_identity(self, repo: Path) -> RepositoryIdentity: ...
    def resolve_commit(self, repo: Path, revision: str) -> str: ...
    def commit_exists(self, repo: Path, sha: str) -> bool: ...
```

`resolve_commit()` 使用等价于：

```text
git rev-parse --verify <revision>^{commit}
```

不读取未提交工作区内容。

Repository identity 以 canonical Git common directory 为主，而不是当前 worktree 路径。这样同一仓库的合法 linked worktree 可以被识别，其他仓库即使目录名相同也不能冒充当前 Session。`origin_url` 是可选审计信息：本地仓库没有 remote 时允许为 `null`，remote URL 变化本身不改变 identity。

### 5.3 SessionStore

在现有 `CheckpointStore` 基础上负责：

- 原子 JSON 写入。
- manifest 读取和 schema 校验。
- artifact hash 计算与验证。
- Session lineage 查询。
- child Session 去重。

`CheckpointStore` 继续作为低层文件 API，但不再单独决定恢复语义。

### 5.4 ReviewPipeline

把当前 `_run_review()` 拆成明确阶段。每个阶段实现统一接口：

```python
class ReviewStage(Protocol):
    phase: RunPhase

    def required_artifacts(self) -> tuple[str, ...]: ...
    def load(self, context: PipelineContext) -> StageResult: ...
    def run(self, context: PipelineContext) -> StageResult: ...
```

Pipeline 对每个阶段执行：

```text
validate completed checkpoint
  -> valid: load typed result
  -> invalid/pending/failed/running: run stage
  -> atomically write artifacts
  -> update manifest
  -> continue
```

### 5.5 ResumePlanner

输入 Session Manifest、当前仓库 identity、当前解析后的 requested revisions 和 artifact validation 结果，输出：

```python
class ResumeAction(str, Enum):
    AUDIT_COMPLETED = "audit_completed"
    CONTINUE_SESSION = "continue_session"
    CREATE_INCREMENTAL_SESSION = "create_incremental_session"
    BLOCKED = "blocked"
```

ResumePlanner 只产生计划，不直接修改文件。

## 6. Session Manifest

运行目录继续使用：

```text
.review-agent/runs/<review-id>/
```

新增权威文件：

```text
session.json
```

Schema：

```yaml
schema_version: 1
review_id: review-abc123
parent_review_id: null
root_review_id: review-abc123
repository:
  canonical_path: D:/repo
  git_common_dir: D:/repo/.git
  origin_url: https://example/repo.git  # optional
revisions:
  requested_base: main
  requested_head: HEAD
  resolved_base_sha: abc...
  resolved_head_sha: def...
  original_base_sha: abc...
  incremental_from_sha: null
  change_kind: initial
execution:
  reviewer_provider: openai-compatible
  reviewer_model: review-model
  reviewer_base_url: https://provider.example/v1
  reviewer_api_key_env: REVIEW_AGENT_API_KEY
  reviewer_mode: multi
  reviewer_loop: agent-loop
  non_interactive: true
status: running
current_phase: reviewers
last_successful_phase: repository_intelligence
phases:
  preflight:
    status: completed
    attempts: 1
    started_at: ...
    completed_at: ...
    artifacts: [request, intent, risk_packet, risk, assignments, quality_gates]
  repository_intelligence:
    status: completed
    attempts: 1
    artifacts: [repository_intelligence, observations]
  reviewers:
    status: running
    attempts: 1
    artifacts: []
artifacts:
  request:
    path: request.json
    sha256: ...
    schema: review_request_v1
    phase: preflight
    revision_binding: null
  observations:
    path: observations.jsonl
    sha256: ...
    schema: observations_v1
    phase: repository_intelligence
    revision_binding: abc...def...
created_at: ...
updated_at: ...
```

### 6.1 不保存 Secret

Session 只保存：

- provider 名称。
- model 名称。
- base URL。
- API key 环境变量名称。

不保存环境变量值、Authorization Header 或 Provider 原始 secret。

### 6.2 state.json 的定位

`session.json` 成为权威恢复状态。

`state.json` 保留为面向人和旧测试的兼容摘要，字段由 Session Manifest 派生：

```text
review_id
status
phase
repository_path
requested base/head
resolved base/head
message
artifact paths
errors
```

`original_base_sha` 表示 root lineage 创建时的 Base，仅用于 lineage 审计；当前 Session 的实际完整审查范围始终由 `resolved_base_sha..resolved_head_sha` 定义。Base 漂移后的 child 可以保留 root 的 `original_base_sha`，但不会错误地用它替代 child 的 `resolved_base_sha`。

## 7. 阶段状态模型

### 7.1 PhaseStatus

```python
class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALIDATED = "invalidated"
```

### 7.2 阶段顺序

```text
preflight
repository_intelligence
reviewers
reconciliation
completion
final_risk
reporting
completed
```

`quality_gates` 继续属于 preflight 的输出，避免同一批基础输入被拆成无法独立恢复的半阶段。

### 7.3 running 的恢复语义

如果进程在阶段执行中退出，manifest 会保留 `status=running`。恢复时该阶段从头重跑，因为只有完成原子提交的 artifact 才可信。

### 7.4 failed 的恢复语义

可重试失败：

- Provider 暂时不可用。
- API Key 环境变量缺失后被补齐。
- 命令超时。
- 阶段输出解析失败。

恢复时从 failed 阶段重跑。

不可恢复失败：

- 仓库 identity 不匹配。
- resolved commit 不存在。
- Session schema 高于当前 Runtime 支持版本。
- Session lineage 循环或损坏。

返回 `BLOCKED`，不修改原 Session。

### 7.5 Reviewer 子 Checkpoint

Reviewer Stage 不是一个不可分割的黑盒。Manifest 在 `phases.reviewers.tasks` 中按 reviewer index 保存：

```yaml
tasks:
  reviewer-0:
    status: completed
    attempts: 1
    artifacts: [reviewer_0_envelope, reviewer_0_result, reviewer_0_trace]
  reviewer-1:
    status: running
    attempts: 1
    artifacts: []
```

恢复时：

- 已完成且 artifact 有效的 Reviewer 不重复调用模型。
- running、failed、invalidated 的 Reviewer 从该 Reviewer 的初始 Assignment 重跑。
- 所有必需 Reviewer 都完成或明确失败后，才提交 Reviewer Stage 的聚合 artifact。

单个 Reviewer 的 Agent Loop 以 Reviewer 为原子恢复单元。当前设计不尝试从模型的半个 turn 中继续；这样无需持久化 Provider 私有会话状态，也不会误重放一个未确认完成的 tool call。

## 8. Artifact Registry 与验证

### 8.1 ArtifactDescriptor

```python
@dataclass(frozen=True)
class ArtifactDescriptor:
    name: str
    path: str
    sha256: str
    schema: str
    phase: RunPhase
    revision_binding: str | None
```

### 8.2 原子写入

写入流程：

```text
serialize to <name>.tmp
flush and close
parse temporary artifact
calculate sha256
os.replace(temp, final)
update session.json atomically
```

如果进程在 `os.replace()` 前退出，旧 artifact 不受影响；临时文件在下次 inspect/resume 时清理。

### 8.3 下游失效传播

如果 `reviewers` artifact hash 不匹配：

```text
reviewers -> invalidated
reconciliation -> invalidated
completion -> invalidated
final_risk -> invalidated
reporting -> invalidated
```

Preflight 和 Repository Intelligence 仍可复用，只要各自 artifact 通过校验且 revision 未变。

### 8.4 Stage Attempt 隔离

阶段和 Reviewer 子任务执行期间，不直接把半成品写进权威 artifact 路径。每次 attempt 使用隔离目录：

```text
.review-agent/runs/<review-id>/attempts/<phase>/<attempt-number>/
```

Reviewer 子任务进一步使用：

```text
attempts/reviewers/<attempt-number>/reviewer-<index>/
```

本次 attempt 的 Observation JSONL、raw artifact、模型响应和 trace 都先写入该目录。成功后才：

1. 校验 schema、hash 和 revision binding。
2. 原子提升为权威 artifact。
3. 更新 Session Manifest。

失败 attempt 可以保留为审计日志，但不进入 `authorized_observation_ids`，也不能被 Reconciler 当作当前 evidence。重跑不会向旧的半成品 JSONL 追加内容。

## 9. Typed Hydration

真正恢复不能只加载 JSON dict。每个可复用阶段必须提供 typed loader：

```text
intent_from_dict
risk_packet_from_dict
risk_assessment_from_dict
assignments_from_dict
quality_results_from_dict
repository_intelligence_from_dict
reviewer_execution_from_dict
reconciliation_from_dict
completion_from_dict
final_risk_from_dict
review_brief_from_dict
```

Loader 必须：

- 校验必需字段。
- 恢复 Enum 和 dataclass。
- 拒绝未知 schema version。
- 不静默填补会改变审查语义的缺失字段。

Reviewer typed state 以 Assignment + ReviewerResult + trace metadata 为恢复核心；Provider 原始响应只用于审计，不是 Reconciliation 的必需依赖。

## 10. ObservationStore 恢复

`ObservationStore` 新增：

```python
@classmethod
def load(cls, run_dir: Path, expected_revisions: set[str]) -> "ObservationStore": ...
```

加载时验证：

- JSONL 每行可解析。
- Observation ID 可由字段重新计算得到。
- raw artifact 存在。
- raw artifact hash 等于 `content_hash`。
- revision 属于当前 Session 的授权 binding。
- 没有重复 ID 指向不同内容。

失败的 Observation 不进入可授权 evidence 集合，并使生产它的阶段失效。

## 11. Resume 算法

```text
load session.json
validate schema and lineage
validate repository identity
verify stored resolved commits exist
resolve requested base/head in current repository
compare current resolved revisions with stored revisions
  -> drift: plan/create child Session
  -> unchanged: validate phase artifacts
validate observations and revision binding
  -> completed Session + all valid: audit only
  -> otherwise: find earliest incomplete/invalid phase
rebuild provider from persisted non-secret config
run pipeline from earliest phase
print resume result and final state
```

### 11.1 Completed Session

revision 未变化：

```text
Resume action: audit_completed
```

不调用模型、不运行质量门、不覆盖报告。

### 11.2 未完成 Session

从最早 pending、running、failed 或 invalidated 阶段开始。

已经通过 artifact validation 的上游阶段通过 typed loader 恢复。

### 11.3 重复 Resume

第一次 resume 完成后，第二次 resume 进入 audit mode。

如果 revision 已漂移且已经存在相同 parent/base/head 的 child Session，返回该 child，而不是创建新的 review ID。

## 12. Revision 漂移与增量 Session

### 12.1 漂移类型

```python
class RevisionChangeKind(str, Enum):
    INITIAL = "initial"
    HEAD_MOVED = "head_moved"
    BASE_MOVED = "base_moved"
    BASE_AND_HEAD_MOVED = "base_and_head_moved"
```

### 12.2 Head 移动

Parent：

```text
base=A
head=B
```

恢复时 requested head 解析为 C：

```text
child.original_base_sha = A
child.resolved_base_sha = A
child.incremental_from_sha = B
child.resolved_head_sha = C
child.change_kind = head_moved
```

Child 生成两份变更地图：

- Full review map：`A..C`，用于最终审查与报告。
- Incremental priority map：`B..C`，用于优先调查新变化。

Child 不加载 parent 的 Observation 或 Finding 作为 evidence。

### 12.3 Base 移动

如果 requested base 从 A 移到 D：

```text
child.resolved_base_sha = D
child.resolved_head_sha = current head
child.incremental_from_sha = null
child.change_kind = base_moved 或 base_and_head_moved
```

Base 漂移改变了完整 Diff，必须重新运行全部阶段。

### 12.4 Detached SHA

如果 requested revision 本身是 commit SHA，恢复时仍解析到相同 SHA，不产生漂移。

## 13. CLI 行为

### 13.1 review

Preflight 新增显示：

```text
Requested base: main
Resolved base: abc123...
Requested head: HEAD
Resolved head: def456...
Session: review-...
```

### 13.2 resume

未完成且 revision 未变化：

```text
Resume
  Action: continue_session
  Starting phase: reviewers
  Reused phases: preflight, repository_intelligence
```

已完成：

```text
Resume
  Action: audit_completed
  No model or tool execution performed
```

revision 漂移：

```text
Resume
  Action: create_incremental_session
  Parent review: review-old
  New review: review-new
  Change: head_moved
  Full range: A..C
  Incremental priority range: B..C
```

不可恢复：

```text
Resume blocked: repository identity mismatch
```

退出码：

```text
0: audit 或 resume 成功
1: pipeline 执行失败但 Session 已安全保存
2: 参数、Session、仓库 identity 或 schema 不可恢复
```

## 14. Pipeline 重构边界

`cli.py` 只保留：

- 参数解析。
- 创建 request/config。
- 调用 ReviewSessionService。
- 输出人类可读摘要。

新增建议文件：

```text
src/review_agent/session.py
  SessionManifest、PhaseCheckpoint、ArtifactDescriptor、序列化

src/review_agent/session_store.py
  原子写入、hash、lineage、child 去重

src/review_agent/revision.py
  RepositoryIdentity、RevisionResolver、revision drift

src/review_agent/pipeline.py
  ReviewPipeline、PipelineContext、各阶段执行与 hydration

src/review_agent/resume.py
  ResumePlanner、ResumeAction、ResumeResult
```

避免把所有恢复逻辑继续堆进 `cli.py` 或 `run_state.py`。

## 15. 失败与安全

### 15.1 API Key 缺失

Resume 重建 provider 时，如果配置的环境变量不存在：

- 阶段保持 failed 或 pending。
- Session 保存可重试错误。
- 不删除已有 artifact。
- 用户补齐环境变量后可以再次 resume。

### 15.2 Artifact 损坏

- 不信任损坏 artifact。
- 标记生产阶段 invalidated。
- 记录明确原因和文件名。
- 从该阶段重新运行。

### 15.3 Session Schema 不兼容

高于 Runtime 支持版本时 hard block，不猜测迁移。

低版本的显式迁移以后通过独立 migrator 实现，不在读取函数中静默修改。

### 15.4 仓库内容不可信

Session Manifest 和 Runtime Policy 优先级高于仓库内文件。仓库内容不能改变 provider、工具权限、revision binding 或恢复起点。

## 16. 测试策略

### 16.1 单元测试

- revision 表达式解析为 commit SHA。
- repository identity 稳定性。
- Session Manifest round-trip。
- Secret 不进入 manifest。
- Phase 状态转换。
- artifact hash 与 schema 验证。
- 下游 invalidation。
- ResumePlanner 四种 action。
- child Session 去重。
- typed loader 拒绝缺失字段。
- Observation ID、hash 和 revision 校验。

### 16.2 集成测试

- 在 preflight 后模拟中断，resume 从 repository intelligence 继续。
- 在 reviewer 阶段模拟中断，resume 重跑 reviewer，不重跑有效 preflight。
- 多 Reviewer 中一个已完成、一个中断时，只重跑中断 Reviewer。
- reviewer 已完成、reconciliation 未完成时复用 reviewer artifact。
- completed Session resume 不调用模型或质量门。
- 缺失 artifact 使对应阶段和下游失效。
- artifact 内容被篡改后重新运行对应阶段。
- Head 移动创建 child Session，parent 不变。
- child 不继承 parent Observation 作为 evidence。
- 同一 revision drift 重复 resume 返回同一 child。
- Base 移动触发完整重审。
- API Key 缺失后补齐并成功 resume。

### 16.3 回归测试

- 原 `review` CLI 行为和产物继续可用。
- 原 completed resume 摘要继续可读，但新增 audit action。
- fake、openai-compatible、single、multi、single-shot、agent-loop 全路径兼容。
- Context、Evidence、Completion、Final Risk 和 Review Brief 测试全部通过。

## 17. 实现批次

工程量较大，拆成三个连续批次；三个批次共享本设计，不建设临时架构。

### 批次 A：Session Foundation 与 Revision Binding

- Session Manifest、ReviewExecutionConfig、RepositoryIdentity。
- resolved base/head SHA。
- 原子 SessionStore 与 artifact registry。
- state.json 兼容摘要。
- review 创建时写 session.json。

完成标准：新 Review 从创建开始就有不可变 revision binding 和可验证 artifact。

### 批次 B：Resumable Pipeline 与 Typed Hydration

- 拆分 `_run_review()` 为 ReviewPipeline 阶段。
- Phase checkpoint、artifact validation、下游 invalidation。
- typed loaders。
- ObservationStore.load()。
- 从中断/失败阶段继续。

完成标准：revision 未变化时，未完成 Session 能真实恢复且不重复有效阶段。

### 批次 C：Revision Drift 与 Incremental Lineage

- ResumePlanner。
- completed audit mode。
- Base/Head drift 检测。
- child Session、lineage、去重。
- full map + incremental priority map。
- CLI 输出与端到端测试。

完成标准：revision 变化时不会混用旧 evidence，并产生可审计、幂等的 child Review。

## 18. 完成定义

只有同时满足以下条件，才认为本模块完成：

- `resume` 能恢复未完成运行，而不是只打印状态。
- 已完成运行 resume 不重复执行。
- Session 绑定 resolved Base/Head SHA。
- artifact 有 hash、schema、phase 和 revision binding。
- 缺失或损坏 artifact 会使阶段和下游失效。
- Observation 可以从磁盘加载并校验。
- Base/Head 漂移创建 child Session，parent 不变。
- child 不把 parent evidence 当作当前 evidence。
- 相同 drift 的重复 resume 幂等。
- provider secret 不落盘。
- 全量测试通过。

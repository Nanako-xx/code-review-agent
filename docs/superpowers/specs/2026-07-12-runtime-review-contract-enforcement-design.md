# Runtime Review Contract Enforcement 设计

状态：已确认并实施

设计来源：`2026-06-22-evidence-driven-multi-agent-code-review-design.md` 第 8、10.5、12、14、20 节。

## 1. 问题

现有主路径存在三类错误完成：

- `reviewer_provider=none` 时跳过 Reviewer、Reconciliation 和 Completion，Session 仍显示 completed。
- 模型返回 `status=completed` 后 Runtime 只检查 JSON 可解析，不检查 Assignment Contract、Finding 字段和 Evidence authority。
- single Reviewer 的 Finding 不进入 Reconciliation，最终 Brief 的 verified findings 为空。

此外，Tool Gateway 合法产生的 `base@SHA`、`head@SHA` Observation 在持久化回读时只因加载器仅接受 `Base..Head` 而被误拒。

## 2. 核心语义

`Session.status` 表示执行生命周期；`CompletionResult.status` 表示审查结论。二者不能混用：

```text
Session completed + Completion blocked
= 流水线安全结束，但审查责任尚未满足
```

无 Provider、Core Reviewer 未运行或 Core Reviewer 失败时，可以生成完整、可审计的报告，但 Completion 必须为 `blocked`，建议必须为 `manual_review`。

## 3. Observation authority

同一 Session 的合法 Observation revision binding 只有：

```text
<base_sha>..<head_sha>
base@<base_sha>
head@<head_sha>
```

其他 revision 一律拒绝。Artifact descriptor 继续绑定完整 `Base..Head`；单条 Observation 保留产生它的精确 Base 或 Head 来源。

## 4. Finding schema

新模型输出中的每个 confirmed Finding 必须包含：

```yaml
claim: non-empty
severity: blocker | high | medium | low
confidence: high | medium | low
path: repository-relative path
line: positive integer
evidence_refs: [authorized observation id, ...]
impact: non-empty
suggested_action: non-empty
verification_performed: [non-empty description, ...]
```

历史 artifact 缺少新增字段时仍可 typed hydrate，但不能因此绕过新的 completion validation；恢复审计发现不合格 reviewer task 时使该 task 失效并重跑。

## 5. Reviewer completion protocol

模型只能申请完成，Runtime 使用 Assignment 和 Evidence allowlist 检查：

- result、Finding、Contract assessment 的 evidence refs 均已授权。
- Finding 字段完整且至少有一个 evidence ref。
- `completed` result 的 investigation summary 非空。
- 每个 assigned Contract 恰有一个 assessment。
- Contract status 为 `covered` 或 `not_applicable`。

Agent Loop 中校验失败时，Runtime 把稳定的 deficiencies 反馈给下一轮；剩余预算耗尽后仍未修正，则保留原始结果并降级为 `partial`。single-shot 无下一轮，直接降级并记录 uncertainty。`partial`、`blocked`、`failed` 可以缺少 Contract 结论，但不能引用未授权 Evidence。

## 6. 统一后处理

single、multi 和 no-provider 三条路径都必须经过：

```text
Reviewer executions（可为空）
-> deterministic Evidence Reconciliation
-> Global Completion Checker
-> Final Risk
-> Review Brief
```

single Reviewer 的合法 Finding 也生成 CanonicalFinding，并保留 path、line、impact 和 verification information。无 Reviewer 时生成空 reconciliation 和 blocked completion，而不是省略 artifact。

## 7. 兼容与非本批范围

保持现有 Session schema version 和 artifact 名称；Finding 新字段采用 additive hydration 兼容。completed Session 的技术审计语义不变，审查结论由 `completion.json` 和 Review Brief 表达。

本批不实现 Intent Manager、多 Reviewer 并行、动态 Portfolio、语义 Reconciler、补充调查、完整 Quality Gates、Eval 或 GitHub/PR 集成；这些继续使用同一最终架构分批完成。

## 8. 完成标准

- 无 Core Reviewer 时 Completion 明确 blocked。
- 模型不能靠自报 `completed` 绕过 assigned Contract。
- 不完整 Agent Loop completion 会重试，预算耗尽后降级。
- single Reviewer Finding 出现在最终 verified findings。
- `base@SHA` / `head@SHA` Observation 可安全回读，其他 revision 仍拒绝。
- 旧 artifact 可读取，全量回归通过。

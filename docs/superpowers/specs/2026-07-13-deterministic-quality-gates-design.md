# Deterministic Quality Gates 完整设计

**状态：** 已实现并通过回归验证（2026-07-13）

**设计来源：** `2026-06-22-evidence-driven-multi-agent-code-review-design.md` 第 7、14、17、20 节。

## 1. 目标

本批把当前只有 `python_compile` 的占位实现扩展为完整、可审计的 Python Quality Gate Runtime：从固定 Head revision 发现适用门禁，先运行廉价检查，再由风险与 Reviewer 组合触发昂贵检查；任何通过、失败、跳过、超时、不可用或运行错误都形成结构化结果和 Observation，并进入 Risk、Reviewer Context、Completion、Final Risk 与 Review Brief。

Quality Gate 默认是 risk signal，不因普通失败中止 Reviewer。只有仓库显式配置为 blocking 的门禁才成为 Completion blocker。

## 2. 两阶段执行

### 2.1 Pre-risk gates

`QUALITY_GATES` checkpoint 负责：

- 在 `resolved_head_sha` 上发现门禁和仓库配置；
- 运行 Python compile、ruff、mypy/pyright 等廉价门禁；
- 保存 `quality_gate_plan.json`、`quality_gates.json` 和原始日志 Observation；
- 只把这些前置结果作为初始 Risk Assessment 的确定性信号。

### 2.2 Risk-triggered gates

风险和 Reviewer portfolio 形成后，`PLANNING` checkpoint 负责执行 plan 中昂贵门禁：pytest 和仓库显式配置的安全检查。触发条件由本地 Runtime 根据 risk level、suggested focus、Reviewer roles 和 gate policy 决定，不把抽象 risk level交给 Reviewer 模型自行解释。

未触发的昂贵门禁写入 `skipped` 结果及 policy reason；因此 Completion 可以区分“未运行但有明确原因”和“结果丢失”。深度结果与 Observation 作为 Planning artifact 一次提交，恢复时不会重复运行已完成 checkpoint。

## 3. Gate Plan

每个门禁定义包含：

```yaml
name: stable gate id
category: compile | format | type | lint | build | test | security
cost: cheap | expensive
source: builtin | repository_config
command: argv list; never shell text
blocking: bool
timeout_seconds: positive finite number
trigger_risks: [low, medium, high, critical]
```

内建 Python 发现规则：

- 存在 `.py` blob：`python_compile`。
- 存在 ruff 配置：`ruff`。
- 存在 mypy 配置：`mypy`。
- 存在 pyright 配置：`pyright`。
- 存在 pytest 配置或测试文件：`pytest`。

仓库可在 Head revision 的 `pyproject.toml` 中通过 `[[tool.review-agent.quality-gates]]` 显式声明安全门禁。配置必须是严格 argv、合法分类/成本/风险、有限 timeout 和唯一名称；无效配置记录 discovery issue，不能静默执行。

## 4. Revision 与执行隔离

- 发现和执行只绑定 `resolved_head_sha`，不读取脏工作区内容。
- 外部命令在由 Git blob 安全物化的临时快照中运行；拒绝绝对路径、父目录穿越、symlink/submodule 物化和 shell。
- 只允许内建工具或仓库显式声明且通过 argv/module policy 的命令。
- 子进程使用最小环境，不继承 API key/token；工作目录、HOME、TEMP 和 Python import path 均指向隔离目录。
- Python 网络入口通过 Runtime guard 禁用，同时设置 deny-proxy；不安装依赖、不联网补齐工具。
- 每个命令有 wall-clock timeout、输出字节上限和进程树终止；越界形成 `timed_out` 或 `error` 结果。
- 日志进入 Observation 前进行常见 secret pattern 脱敏。

## 5. Result 与 Observation

`QualityGateResult` 在旧字段上 additive 扩展：

```yaml
name: string
status: passed | failed | skipped | unavailable | timed_out | error
command: [string]
summary: string
observation_ref: string | null
category: string
cost: cheap | expensive
source: builtin | repository_config | legacy
blocking: bool
reason: string | null
exit_code: int | null
duration_seconds: number
output_truncated: bool
sandbox: string
```

每个计划内门禁都必须有唯一终态 Result。完整的有界、脱敏 stdout/stderr 保存为 raw Observation；模型上下文只得到结构化摘要和 Observation ref，需要时再通过受控读取获取日志。

## 6. Completion 语义

Completion 对 Gate Plan 与结果做确定性核对：

- discovery issue、缺失/重复/未知结果不能伪装为完成；
- non-blocking failed/unavailable/timed_out/error 进入 uncertainty，并继续审查；按风险策略明确 `skipped` 且记录原因的非阻断门禁满足该深度下的 Completion；
- blocking gate 的非 `passed` 终态进入 blocker；
- 无适用门禁的空 plan 仍证明发现阶段已经完成；
- 旧 artifact 没有新字段或 plan 时继续按 legacy 语义 hydrate，不篡改历史结论。

## 7. 下游传播

- 初始 Risk 只读取 cheap gate 状态；失败提升风险，不可用/超时降低 verification strength。
- 深度结果追加到 Reviewer 的 `quality_gate_summary` 与 authorized Observation refs。
- Final Risk 使用全部结果重新评估。
- JSON/Markdown Brief 披露 category、cost、blocking、status、reason、duration 和 Observation ref。

## 8. 失败与恢复

- 门禁工具返回非零是结构化 `failed`，不是 Pipeline 异常。
- 工具缺失、timeout、输出越界和受控执行错误形成终态结果，Reviewer 继续。
- artifact promotion、hash、Session registry 或 Observation 提交失败仍是控制层失败，整个所属 checkpoint 可恢复重试。
- 已完成 Quality/Planning checkpoint 在 resume 时只 hydrate，不重新执行命令；revision drift child 重新发现并执行新 Head 的门禁。

## 9. 非本批范围

- 模型辅助 Risk Assessor / Portfolio Planner。
- Reviewer 运行中临时提出任意命令；后续有界 supplemental investigation 只能从已发现 Gate Plan 中选择。
- 语义 Reconciler、Durable Memory、Eval Harness、GitHub/PR 集成。

## 10. 完成标准

- compile/ruff/mypy/pyright/pytest 与显式安全门禁可从 Head revision 确定性发现。
- 脏工作区和非 checkout Head 不影响发现/执行。
- cheap/deep policy、blocking/non-blocking、缺失工具、超时和输出上限均有测试。
- 每个结果均有合法 Observation ref，Reviewer 只获得摘要和授权 ref。
- Completion 验证 plan coverage，恢复不重复已提交门禁。
- 旧 quality artifact/Brief 继续 hydrate；定向和全量测试通过。

## 11. 实施结果

- `QUALITY_GATES` checkpoint 产出 `quality_gate_plan.json`、廉价门禁结果和 pre-risk Observation Store。
- `PLANNING` checkpoint 按本地风险与 Reviewer portfolio 产出昂贵门禁的执行或策略跳过结果，并把摘要与授权 Observation refs 写入 Assignment。
- 独立 Quality Gate Runner 在固定 Head snapshot 中执行受允许的 Python 模块，具备命令白名单、文件/字节边界、最小环境、Python 网络 guard、超时、输出上限、进程树终止与日志脱敏。
- Risk、Completion、Final Risk、JSON/Markdown Brief、resume 和旧 artifact hydration 已接入完整 gate metadata 与全部终态。
- 架构边界、发现与脏工作区隔离、真实命令执行、Pipeline、Completion、恢复兼容及完整测试集均已通过。

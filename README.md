# Review Agent

面向 Pull Request 的快照绑定、证据驱动代码审查运行时。
Snapshot-bound, evidence-driven code review runtime for Pull Requests.

- [English](#english)
- [中文](#中文)

---

## English

### Overview

Review Agent is a Python 3.11+ CLI and runtime for reviewing the changes between a Git base revision and head revision.

Each product review is bound to an immutable base_sha..head_sha Snapshot, a PR-scoped PRWorkspace, a complete persisted DiffArtifact, append-only execution records, and a deterministic review-result.json.

The runtime is read-only with respect to the target repository. It can inspect and review code, but it does not modify source files or create Pull Requests.

This repository focuses on the product Code Review Agent runtime. Benchmark runners, AACR/SWE adapters, and Judge implementations are separate concerns and are not required by the product CLI.

### Pipeline

~~~mermaid
flowchart LR
    A["Git base..head"] --> B["Immutable Snapshot"]
    B --> C["Full DiffArtifact"]
    C --> D["Deterministic Preflight"]
    D --> E["Intent Packet"]
    E --> F["Risk Assessment"]
    F --> G["Review Plan / Assignments"]
    G --> H["Reviewer Context"]
    H --> I["Core / Adversarial / Dynamic Reviewers"]
    I --> J["Deterministic Aggregation"]
    J --> K["review-result.json / Markdown"]
~~~

The pre-LLM sequence is:

**DiffArtifact → Quality Gate → Changed Symbols → Intent Packet → Risk Assessment → Review Plan/Assignments → Context Assembly**

### Core capabilities

- **Snapshot-bound PRWorkspace** — facts from different PRs cannot be mixed; a new head commit creates a new immutable Snapshot.
- **Complete DiffArtifact** — the original Git diff is persisted as diff.patch with a mechanically generated index.json; context management never silently truncates the authoritative diff.
- **Deterministic Preflight** — establishes the diff, runs the configured local quality plan, and collects changed symbols before reviewer execution.
- **Separated analysis stages** — Intent, Risk, Planning, and Reviewer execution have explicit contracts and persisted artifacts.
- **Risk-based reviewer slots** — the plan can assign core, adversarial, and dynamic reviewers according to final risk.
- **Developer review rules** — packaged rules are resolved by changed-file path and have higher priority than user project rules.
- **Artifact-backed large results** — large non-reacquirable tool results are stored in the PR workspace and projected through a small preview and paged reads.
- **Per-reviewer durability** — execution journals, tool-call idempotency, context manifests, and compaction state are isolated per reviewer.
- **Resumable execution** — a committed reviewer turn is a recovery boundary; completed tool calls are not blindly repeated.
- **Deterministic aggregation** — reviewer outputs are normalized and merged without an additional semantic model call, then rendered as JSON or Markdown.
- **Read-only tools** — tools are limited to repository inspection, diff/symbol lookup, commit reading, and artifact reading.

### Requirements and installation

Requirements:

- Python 3.11 or newer
- Git available on PATH
- An OpenAI-compatible API and API key for model-backed review
- No mandatory third-party runtime dependency; development dependencies include pytest

~~~powershell
git clone <repository-url>
cd <repository-directory>
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -e ".[dev]"
~~~

On macOS/Linux, use source .venv/bin/activate instead of the PowerShell activation command.

The console entry point is review-agent. During development, python -m review_agent is equivalent.

### Quick start

Offline smoke run with the fake provider:

~~~text
review-agent review --repo . --base origin/main --head HEAD --external-review-id local-demo-1 --reviewer-provider fake --risk-assessor-mode local --format markdown
~~~

Model-backed example with an OpenAI-compatible DeepSeek endpoint:

~~~powershell
$env:REVIEW_AGENT_API_KEY = "your-api-key"
review-agent review --repo . --base origin/main --head HEAD --external-review-id github:owner/repository#123 --reviewer-provider openai-compatible --reviewer-model deepseek-v4-flash --reviewer-base-url https://api.deepseek.com --reviewer-api-key-env REVIEW_AGENT_API_KEY --risk-assessor-mode local --format markdown
~~~

Keep API keys in the environment or a secret manager. Never put them in source files, logs, or committed artifacts.

The product runtime requires external-review-id even though the argument is optional in argparse. Use a stable value for each PR or benchmark instance.

### CLI options

Use the version-specific help for the complete list:

~~~text
review-agent review --help
review-agent resume --help
~~~

Common options:

| Option | Purpose |
| --- | --- |
| --repo | Target repository path; defaults to the current directory |
| --base / --head | Git revisions that define the review Snapshot |
| --external-review-id | Stable PR/review identity; required by the product runtime |
| --workspace-root | PRWorkspace root; defaults to .review-agent/workspaces-v6 |
| --format json/markdown | Output representation |
| --intent / --focus | User-provided review context |
| --project-rule | Additional project rule; cannot override developer rules |
| --reviewer-provider | none, fake, or openai-compatible |
| --reviewer-model | Provider model name |
| --reviewer-base-url | OpenAI-compatible endpoint |
| --reviewer-api-key-env | Environment variable containing the provider key |
| --risk-assessor-mode | local for deterministic risk, or model for model-assisted risk |
| --risk-assessor-provider | Provider selection for model-assisted risk |

### Resume

Resume a product Session with its immutable identifiers:

~~~text
review-agent resume SESSION-<session-sha256> --repo . --pr-id PR-<pr-sha256> --snapshot-id S-<snapshot-sha256> --reviewer-provider openai-compatible --reviewer-model deepseek-v4-flash --reviewer-base-url https://api.deepseek.com --reviewer-api-key-env REVIEW_AGENT_API_KEY --format markdown
~~~

The actual SESSION-, PR-, and S- values are returned by the first run and recorded in workspace manifests. Resume verifies persisted bindings before continuing.

### Output

The default output is JSON. Markdown is a renderer over the same authoritative result.

Illustrative shape, with IDs shortened:

~~~json
{
  "pr_id": "PR-...",
  "snapshot_id": "S-...",
  "status": "completed",
  "risk_level": "medium",
  "findings": [
    {
      "finding_id": "F-...",
      "claim": "The retry path can submit the same operation twice.",
      "severity": "high",
      "path": "src/service.py",
      "line": 42,
      "suggestion": "Make the operation idempotent or persist the request key."
    }
  ],
  "uncertainties": []
}
~~~

completed means that the review protocol completed. It does not mean that the change is safe to merge or that the agent approved the Pull Request.

### PRWorkspace persistence

The default root is .review-agent/workspaces-v6 inside the target repository:

~~~text
PRWorkspace
├─ manifest.json
├─ PR/
│  └─ pr.json
├─ Intent/
│  ├─ current.json
│  └─ history/
├─ Snapshots/
│  └─ <snapshot-id>/
│     ├─ snapshot.json
│     ├─ DiffArtifact/
│     │  ├─ diff.patch
│     │  └─ index.json
│     ├─ QualityGate/
│     ├─ ChangedSymbols/
│     ├─ Risk/
│     ├─ ReviewPlan/
│     │  └─ Assignments/
│     ├─ ToolResults/
│     │  └─ artifacts/
│     └─ Results/
│        ├─ aggregation.json
│        └─ review-result.json
└─ Sessions/
   └─ <session-id>/
      ├─ state.json
      ├─ pipeline-state.json
      └─ Reviewers/
         └─ <reviewer-id>/
            ├─ reviewer.json
            ├─ execution-log.jsonl
            ├─ context-manifest.json
            └─ context-compaction-<generation>.txt
~~~

The workspace is PR-scoped, not repository-global. Immutable Snapshot facts may be shared, while each reviewer keeps its own message history, tool-result projection, journal, last-API timestamp, and compaction generation.

### Runtime policy

| Policy | Default |
| --- | ---: |
| Model context window | 1,000,000 tokens |
| Soft compaction trigger | 700,000 tokens |
| Non-reacquirable artifact threshold | 50,000 characters |
| Per-turn tool-result budget | 200,000 characters |
| Large-result preview | approximately 2,000 characters |
| Idle prompt-cache cleanup | 3,600 seconds; retain the 5 most recent reacquirable results |
| Reviewer active execution time | 1,800 seconds |
| Provider attempts per model turn | 3 |
| Individual tool timeout | 300 seconds |
| Fixed tool-call count limit | none; safety and time controls still apply |

Context compaction changes only the reviewer context projection. It does not delete the authoritative diff, snapshots, artifacts, or execution journal.

Deterministic risk floors are monotonic:

- more than 100 changed files: at least medium;
- inferred Intent: at least medium;
- missing Intent source: high;
- final risk is the maximum of deterministic floors and an optional model decision.

The product runtime no longer uses the old fixed 16,000-character initial-message limit or old reviewer output/total-token stop rules. Providers may still enforce their own API limits.

### Language support and rules

The core diff and review protocol is language-agnostic. The built-in developer rule catalog currently covers:

- C/C++, Go, Java, Kotlin, Rust, Python;
- TypeScript, JavaScript, TSX, JSX;
- JSON/JSON5, YAML, Terraform/HCL/TFVars, GraphQL, Julia;
- ArkTS, Astro, Bicep;
- Gradle, Maven pom.xml, Cargo.toml, package.json;
- GitHub Actions/workflow/configuration;
- properties, FreeMarker, gettext PO/POT, mapper/DAO XML, and a default fallback rule.

Rules are packaged under src/review_agent/developer_rules/. The path map is in system_rules.json and the Markdown documents are in rule_docs/.

The built-in changed-symbol analyzer is currently primarily Python-oriented. Other languages remain reviewable through the complete diff, file/hunk index, repository tools, and path-specific rules, but do not yet receive the same Python AST symbol coverage.

### Current limitations

- The CLI default LocalQualityPlan is empty. The pipeline has a deterministic quality-gate interface, but the CLI does not infer project commands and does not automatically run a compiler, test suite, or linter.
- The runtime does not modify the target repository, install project dependencies, or create a Pull Request.
- Review quality depends on the provider, model, prompt, repository context, and available diff.
- completed is a protocol status, not a merge recommendation.
- There is currently no LICENSE file in the repository. Add one before distributing the project under a specific license.

### Development

~~~powershell
python -m pytest -q
git diff --check
python -m review_agent --help
~~~

Main design specification: [PR Workspace, Deterministic Preflight and Reviewer Runtime Redesign](docs/superpowers/specs/2026-08-10-pr-workspace-preflight-reviewer-runtime-redesign.md).

---

## 中文

### 项目简介

Review Agent 是一个基于 Python 3.11+ 的 CLI 和运行时，用于审查 Git base revision 与 head revision 之间的 Pull Request 改动。

每次产品审查都会绑定到不可变的 base_sha..head_sha Snapshot、PR 级别的 PRWorkspace、完整持久化的 DiffArtifact、追加式执行记录，以及确定性生成的 review-result.json。

对于目标仓库，运行时是只读的：可以读取和分析代码，但不会修改源代码，也不会自动创建 Pull Request。

当前仓库主要包含产品侧 Code Review Agent Runtime。AACR/SWE 等 benchmark 运行器、评测适配器和 Judge 属于独立组件，不是产品 CLI 的运行时依赖。

### 执行流程

上面的流程图对应以下固定主链：

**DiffArtifact → 质量门 → ChangedSymbols → Intent Packet → 风险评级 → Review Plan/Assignments → 上下文组装 → Reviewer → 确定性汇总**

具体过程是：

1. 为 base..head 建立不可变 Snapshot，并保存完整 Diff；
2. 执行本地确定性 Preflight；
3. 生成精简的 Intent Packet；
4. 进行确定性风险评级，必要时再使用模型辅助评级；
5. 根据最终风险分配 Reviewer 和 Assignment；
6. 为每个 Reviewer 建立隔离的上下文、工具结果投影和压缩状态；
7. 汇总 Finding，生成唯一权威结果，再渲染为 JSON 或 Markdown。

### 核心能力

- **Snapshot 绑定的 PRWorkspace**：不同 PR 的事实不会混用；head commit 变化会产生新的不可变 Snapshot。
- **完整 DiffArtifact**：原始 Git diff 保存为 diff.patch，并生成完整 index.json；上下文治理不会静默截断权威 Diff。
- **确定性 Preflight**：先建立 Diff，再执行配置的本地质量计划并收集 Changed Symbols。
- **分析阶段分离**：Intent、Risk、Planning 和 Reviewer 执行都有独立协议和持久化产物。
- **按风险分配 Reviewer**：根据最终风险级别启用 core、adversarial 和 dynamic Reviewer。
- **开发者审查规则**：开发者提供、用户不可覆盖的规则按改动文件路径解析，优先级高于用户项目规则。
- **大工具结果 Artifact 化**：不可重新获取的大工具结果落盘，只向模型提供小预览，并通过分页读取。
- **每个 Reviewer 独立持久化**：Execution Journal、Tool Call 幂等账本、上下文清单和压缩状态彼此隔离。
- **可恢复执行**：最后一个已提交的 Reviewer 轮次是恢复边界，已经完成的工具调用不会被无条件重复执行。
- **确定性汇总**：对 Reviewer 输出进行规范化和合并，不额外调用语义模型；最终结果可输出为 JSON 或 Markdown。
- **只读工具**：工具主要用于读取代码、Diff、符号、提交信息和已保存 Artifact。

### 环境要求与安装

环境要求：

- Python 3.11 或更高版本；
- Git 已加入 PATH；
- 如果要使用模型审查，需要 OpenAI-compatible API 和对应 API Key；
- 运行时没有强制第三方依赖，开发依赖包含 pytest。

~~~powershell
git clone <repository-url>
cd <repository-directory>
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -e ".[dev]"
~~~

macOS/Linux 使用 source .venv/bin/activate 激活虚拟环境。

安装后可以使用 review-agent 命令；开发环境中 python -m review_agent 等价。

### 快速开始

使用 fake Provider 进行离线 Smoke：

~~~text
review-agent review --repo . --base origin/main --head HEAD --external-review-id local-demo-1 --reviewer-provider fake --risk-assessor-mode local --format markdown
~~~

使用 DeepSeek 等 OpenAI-compatible Endpoint：

~~~powershell
$env:REVIEW_AGENT_API_KEY = "your-api-key"
review-agent review --repo . --base origin/main --head HEAD --external-review-id github:owner/repository#123 --reviewer-provider openai-compatible --reviewer-model deepseek-v4-flash --reviewer-base-url https://api.deepseek.com --reviewer-api-key-env REVIEW_AGENT_API_KEY --risk-assessor-mode local --format markdown
~~~

API Key 应保存在环境变量或密钥管理系统中，不要写入命令参数、源代码、日志或提交产物。

产品运行时要求 external-review-id，虽然 argparse 层面将其标记为可选。每个 PR 或 benchmark instance 应使用稳定且唯一的值。

### CLI 主要参数

查看当前版本的完整参数：

~~~text
review-agent review --help
review-agent resume --help
~~~

| 参数 | 作用 |
| --- | --- |
| --repo | 目标仓库路径，默认为当前目录 |
| --base / --head | 定义审查 Snapshot 的 Git revision |
| --external-review-id | 稳定的 PR/审查身份，产品运行时必需 |
| --workspace-root | PRWorkspace 根目录，默认为 .review-agent/workspaces-v6 |
| --format json/markdown | 输出格式 |
| --intent / --focus | 用户提供的审查上下文 |
| --project-rule | 增加项目规则，但不能覆盖开发者规则 |
| --reviewer-provider | none、fake 或 openai-compatible |
| --reviewer-model | Provider 模型名 |
| --reviewer-base-url | OpenAI-compatible Endpoint |
| --reviewer-api-key-env | 保存 Provider Key 的环境变量名 |
| --risk-assessor-mode | local 只使用确定性风险，model 使用模型辅助风险 |
| --risk-assessor-provider | 模型辅助风险时的 Provider 选择 |

### 恢复 Session

使用不可变的 Session、PR 和 Snapshot 标识恢复：

~~~text
review-agent resume SESSION-<session-sha256> --repo . --pr-id PR-<pr-sha256> --snapshot-id S-<snapshot-sha256> --reviewer-provider openai-compatible --reviewer-model deepseek-v4-flash --reviewer-base-url https://api.deepseek.com --reviewer-api-key-env REVIEW_AGENT_API_KEY --format markdown
~~~

实际 ID 会在第一次运行后返回，也会记录在 workspace manifest 中。恢复过程会校验 PR、Snapshot 和持久化 Artifact 的绑定关系。

### 输出结果

默认输出为 JSON，Markdown 是同一权威结果的渲染形式。

下面是示意结构（为便于阅读，ID 已缩短）：

~~~json
{
  "pr_id": "PR-...",
  "snapshot_id": "S-...",
  "status": "completed",
  "risk_level": "medium",
  "findings": [
    {
      "finding_id": "F-...",
      "claim": "重试路径可能重复提交同一个操作。",
      "severity": "high",
      "path": "src/service.py",
      "line": 42,
      "suggestion": "让操作具备幂等性，或持久化请求 Key。"
    }
  ],
  "uncertainties": []
}
~~~

completed 只表示审查协议已经完成，不表示改动一定可以合并，也不表示 Agent 已经批准 Pull Request。

### PRWorkspace 持久化结构

默认 workspace 根目录是目标仓库中的 .review-agent/workspaces-v6：

~~~text
PRWorkspace
├─ manifest.json
├─ PR/
│  └─ pr.json
├─ Intent/
│  ├─ current.json
│  └─ history/
├─ Snapshots/
│  └─ <snapshot-id>/
│     ├─ snapshot.json
│     ├─ DiffArtifact/
│     │  ├─ diff.patch
│     │  └─ index.json
│     ├─ QualityGate/
│     ├─ ChangedSymbols/
│     ├─ Risk/
│     ├─ ReviewPlan/
│     │  └─ Assignments/
│     ├─ ToolResults/
│     │  └─ artifacts/
│     └─ Results/
│        ├─ aggregation.json
│        └─ review-result.json
└─ Sessions/
   └─ <session-id>/
      ├─ state.json
      ├─ pipeline-state.json
      └─ Reviewers/
         └─ <reviewer-id>/
            ├─ reviewer.json
            ├─ execution-log.jsonl
            ├─ context-manifest.json
            └─ context-compaction-<generation>.txt
~~~

Workspace 是 PR 级别共享的，不是整个项目共享的。不可变 Snapshot 事实可以被多个 Reviewer 读取，但每个 Reviewer 的消息历史、工具结果投影、Journal、上次 API 调用时间和压缩代次必须独立。

### 运行时策略

| 策略 | 默认值 |
| --- | ---: |
| 模型上下文窗口 | 1,000,000 tokens |
| 软压缩阈值 | 700,000 tokens |
| 不可重新获取结果的落盘阈值 | 50,000 字符 |
| 单轮工具结果总预算 | 200,000 字符 |
| 大结果预览 | 约 2,000 字符 |
| Prompt Cache 空闲清理 | 3,600 秒；保留最近 5 个可重新获取结果 |
| Reviewer 活跃执行时间 | 1,800 秒 |
| 每轮模型请求 Provider 尝试次数 | 3 |
| 单个工具超时 | 300 秒 |
| 固定工具调用次数上限 | 无；仍受安全和时间控制 |

上下文压缩只改变 Reviewer 发给模型的上下文投影，不会删除权威 Diff、Snapshot、Artifact 或执行日志。

确定性风险下限是单调的：

- 改动文件数大于 100：至少为 medium；
- Intent 为 inferred：至少为 medium；
- Intent source 缺失：为 high；
- 最终风险取确定性下限和可选模型判断中的最高等级。

产品运行时已经取消旧的 16,000 字符初始消息限制，也不再使用旧的 Reviewer 输出/累计 Token 停止规则；Provider 仍可能有自己的 API 限制。

### 语言支持与审查规则

核心 Diff 和审查协议与语言无关。内置开发者规则当前覆盖：

- C/C++、Go、Java、Kotlin、Rust、Python；
- TypeScript、JavaScript、TSX、JSX；
- JSON/JSON5、YAML、Terraform/HCL/TFVars、GraphQL、Julia；
- ArkTS、Astro、Bicep；
- Gradle、Maven pom.xml、Cargo.toml、package.json；
- GitHub Actions/workflow/configuration；
- properties、FreeMarker、gettext PO/POT、mapper/DAO XML，以及默认兜底规则。

规则位于 src/review_agent/developer_rules/：路径映射在 system_rules.json，具体 Markdown 规则在 rule_docs/。这些是开发者规则；用户可以通过 --project-rule 增加项目规则，但不能覆盖开发者规则。

当前内置 Changed Symbols 分析器主要面向 Python AST。其他语言仍然可以通过完整 Diff、文件/hunk 索引、仓库读取工具和按路径规则进行审查，但目前不会获得与 Python 相同的 AST 符号覆盖。

### 当前限制

- CLI 默认的 LocalQualityPlan 为空。系统提供确定性的质量门接口，但 CLI 不会自动推测项目命令，也不会自动运行编译器、测试套件或 Linter。
- 运行时不会修改目标仓库、安装项目依赖或创建 Pull Request。
- 审查质量取决于所选 Provider、模型、Prompt、仓库上下文以及可获得的 Diff。
- completed 是协议状态，不是合并建议。
- 当前仓库没有 LICENSE 文件；如果要以特定许可证发布，请先补充并在 README 中明确说明。

### 开发与测试

~~~powershell
python -m pytest -q
git diff --check
python -m review_agent --help
~~~

主要设计规格见：[PR Workspace、确定性 Preflight 与 Reviewer Runtime 重构](docs/superpowers/specs/2026-08-10-pr-workspace-preflight-reviewer-runtime-redesign.md)。

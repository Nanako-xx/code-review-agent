# Unified Model Adapter Architecture 设计

- 日期：2026-07-03
- 状态：待用户审阅
- 目标分支：`codex/unified-model-adapter-architecture`
- 关联设计：
  - `docs/superpowers/specs/2026-06-22-evidence-driven-multi-agent-code-review-design.md`
  - `docs/superpowers/specs/2026-07-01-model-adapter-agent-loop-design.md`

## 1. 背景

上一版已经实现了两条能力：

1. `ModelAdapter` 协议和 `OpenAICompatibleToolAdapter`。
2. `Reviewer Agent Loop`，Runtime 只认识项目内部的 `ModelTurnRequest` / `ModelTurnResponse`。

但当前代码里仍然存在双接口：

```text
single-shot reviewer
        ↓
ModelProvider

agent-loop reviewer
        ↓
ModelAdapter
```

这不是最终架构。最终方向应该是：所有 reviewer 模型调用都走项目自己的统一 `ModelAdapter` 协议，业务逻辑不直接面对 OpenAI / DeepSeek / Claude / Fake 的 API 差异。

## 2. 最终目标

最终版 B 架构：

```text
CLI / Reviewer / Orchestrator / Agent Runtime
        ↓
ModelAdapter 协议
        ↓
ModelAdapterFactory
        ↓
Fake / OpenAI-compatible / DeepSeek / Claude / ...
```

核心目标：

1. reviewer 业务逻辑只依赖 `ModelAdapter`。
2. CLI 只负责把用户配置交给 `ModelAdapterFactory`，不在业务分支里手写 provider 差异。
3. `single-shot` 和 `agent-loop` 都通过 `ModelAdapter` 调用模型。
4. DeepSeek 先作为 OpenAI-compatible 配置接入。
5. Claude 后续通过新增 `ClaudeToolAdapter` 和 factory 分支接入，不需要改 Runtime。
6. 旧 `ModelProvider` 不再作为 reviewer 业务主接口。

这里允许分多次实现，但每一步都必须直接服务最终架构，不做“先临时支持 fake / 以后再重构”的过渡方案。

## 3. 非目标

本设计不做：

- GitHub / PR 集成。
- Eval harness。
- Claude native adapter 的具体 HTTP 实现。
- 多模型投票策略。
- 动态新增 reviewer assignment。
- LLM Reconciler Agent。
- 长期记忆系统。

Claude adapter 是最终架构的扩展点，但不要求在本次迁移里一次完成。

## 4. 核心设计

### 4.1 ModelAdapter 是唯一 reviewer 模型接口

`ModelAdapter` 保持当前语义：

```python
class ModelAdapter(Protocol):
    provider_name: str

    def complete_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        ...
```

所有 reviewer 执行器，包括 single-shot 和 agent-loop，都只调用这个接口。

`ModelProvider.complete(envelope)` 不再被 reviewer / orchestrator / CLI 的 reviewer 执行路径直接使用。

### 4.2 ModelAdapterFactory

新增一个统一工厂，建议文件为：

```text
src/review_agent/model_adapter_factory.py
```

职责：

- 根据 CLI 配置创建新的 `ModelAdapter`。
- 处理 provider 配置校验。
- 屏蔽 provider-specific 构造细节。
- 对每个 reviewer invocation 返回独立 adapter 实例，避免 stateful fake adapter 在 multi-reviewer 中脚本耗尽。

建议接口：

```python
@dataclass(frozen=True)
class ModelAdapterConfig:
    provider_name: str | None
    model: str | None
    base_url: str | None
    api_key_env: str


class ModelAdapterFactory(Protocol):
    def create(self) -> ModelAdapter:
        ...


def build_model_adapter_factory_from_config(config: ModelAdapterConfig) -> ModelAdapterFactory | None:
    ...
```

返回规则：

```text
provider none
  -> None

provider fake
  -> fake adapter factory

provider openai-compatible
  -> OpenAICompatibleToolAdapter factory

unsupported provider
  -> ProviderConfigError 或新的 AdapterConfigError
```

DeepSeek 不需要单独 provider name；它通过：

```text
--reviewer-provider openai-compatible
--reviewer-base-url <deepseek-compatible-url>
--reviewer-model <deepseek-model>
```

接入。

### 4.3 single-shot 也走 ModelAdapter

现有 `run_single_reviewer(provider, ...)` 应迁移为 adapter-based runner。

建议语义：

```text
run_single_reviewer(adapter, ...)
```

或新增：

```text
run_single_turn_reviewer(adapter, ...)
```

single-shot 的含义是“一次模型调用，不执行工具循环”。因此它应该：

1. 继续使用 `build_reviewer_envelope()` 构造上下文。
2. 转成 `ModelTurnRequest`。
3. 不传可执行工具，或显式设置 `tool_choice = "none"`。
4. 调用 `adapter.complete_turn(request)`。
5. 只接受 `ModelResponseKind.FINAL`。
6. 如果 adapter 返回 `TOOL_CALLS`，返回 failed/partial reviewer result，并提示应使用 `--reviewer-loop agent-loop`。
7. 如果 adapter 返回 `INVALID`，返回 failed reviewer result，记录 adapter error。

这样 single-shot 仍然保持“不执行工具”的产品语义，但底层模型入口已经统一成 `ModelAdapter`。

### 4.4 agent-loop 保持 Runtime 纯净

`run_reviewer_agent_loop()` 已经只依赖 `ModelAdapter`，应保持不变：

```text
ModelAdapter -> ModelTurnResponse -> ToolGateway -> ObservationStore
```

禁止把 OpenAI / Claude 原始字段传进 Runtime。

如果 adapter 返回 provider-specific 异常或无法解析的 tool call，adapter 应转成：

```text
ModelResponseKind.INVALID
```

或结构化的 internal tool call parse error。Runtime 只处理内部协议。

### 4.5 multi-reviewer 统一 adapter builder

multi-reviewer 不应该复用一个可能有状态的 adapter 实例。

建议 orchestrator 接收：

```text
adapter_factory / adapter_builder
```

每个 reviewer execution 调用一次 factory：

```text
adapter = adapter_factory.create()
```

这样：

- fake adapter 每个 reviewer 有自己的脚本。
- openai-compatible adapter 每个 reviewer 有独立配置对象。
- 后续 Claude adapter 也不需要改 orchestrator。

### 4.6 CLI 配置流

CLI 应改成：

```text
parse args
  ↓
build ModelAdapterConfig
  ↓
build ModelAdapterFactory
  ↓
single-shot / agent-loop 只选择 runner
  ↓
runner 使用 adapter_factory.create()
```

CLI 不应该有这种业务约束：

```text
agent-loop currently requires fake
```

迁移后应支持：

```powershell
python -m review_agent review `
  --repo . `
  --base <base> `
  --head <head> `
  --reviewer-provider openai-compatible `
  --reviewer-loop agent-loop `
  --reviewer-model <model> `
  --reviewer-base-url <url>
```

如果 provider 是 `none`：

- `single` 模式：只运行本地 foundation，不跑 reviewer。
- `multi` 模式：仍然报错，因为 multi reviewer 需要模型 reviewer。

## 5. 旧 ModelProvider 的归宿

当前 `provider.py` 中有：

- `ModelProvider`
- `FakeProvider`
- `OpenAICompatibleProvider`
- `OpenAICompatibleConfig`
- HTTP transport helpers

迁移后：

1. reviewer 业务路径不再依赖 `ModelProvider`。
2. `OpenAICompatibleConfig` 和 transport 可以暂时复用，或者移动到 adapter/factory 模块。
3. `OpenAICompatibleProvider` 可以保留为 legacy compatibility，但不得再被 CLI reviewer 主路径调用。
4. 后续清理阶段可以删除 `ModelProvider` / `FakeProvider` / `OpenAICompatibleProvider`，或改成内部兼容 shim。

成功标准不是“文件必须立刻删除”，而是“业务主路径不再通过它调用模型”。

## 6. 错误处理

### 6.1 配置错误

Factory 负责校验：

- openai-compatible 缺少 API key。
- openai-compatible 缺少 model。
- openai-compatible 缺少 base_url。
- unsupported provider name。

CLI 捕获配置错误并返回 exit code 2。

### 6.2 adapter invalid response

adapter 无法解析 provider response 时返回：

```text
ModelResponseKind.INVALID
```

single-shot runner：

- 返回 failed reviewer result。
- raw response artifact 记录 adapter error。

agent-loop runner：

- 记录 trace turn。
- 根据当前 loop 状态返回 failed 或 partial。

### 6.3 provider transport error

OpenAI-compatible 网络错误不应抛穿业务层。

Adapter 应转成：

```text
ModelResponseKind.INVALID
error="provider request failed: ..."
```

然后由 runner 统一处理。

## 7. 测试策略

### 7.1 Factory tests

覆盖：

- `none` 返回 `None`。
- `fake` 返回可创建 `FakeToolCallingAdapter` 的 factory。
- `openai-compatible` 返回可创建 `OpenAICompatibleToolAdapter` 的 factory。
- 缺少 API key / model / base_url 时报配置错误。

### 7.2 single-shot adapter tests

覆盖：

- single-shot runner 使用 adapter 的 FINAL response 并解析 `ReviewerResult`。
- adapter 返回 TOOL_CALLS 时，single-shot 不执行工具，返回明确失败/partial。
- adapter 返回 INVALID 时，返回 failed result。

### 7.3 agent-loop OpenAI-compatible CLI tests

不打真实网络。

可通过 monkeypatch factory 或 transport，使 CLI 走：

```text
--reviewer-provider openai-compatible
--reviewer-loop agent-loop
```

并验证：

- 不再报 “agent-loop only fake”。
- 写出 `reviewer_agent_trace.json`。
- 写出 `reviewer_result.json`。
- trace 中包含 tool call 和 observation id。

### 7.4 regression tests

保留并更新：

- fake single-shot CLI smoke。
- fake agent-loop CLI smoke。
- multi reviewer smoke。
- provider config error smoke。
- full pytest suite。

## 8. 分阶段实现，但不偏离最终架构

可以分阶段提交，但每个阶段都必须是最终架构的一部分。

建议阶段：

1. 新增 `ModelAdapterFactory`，覆盖 fake / openai-compatible / none。
2. 将 single-shot reviewer 从 `ModelProvider` 迁移到 `ModelAdapter`。
3. 将 multi-reviewer orchestration 改为接收 adapter factory / builder。
4. 将 CLI reviewer 主路径改为只构建 adapter factory，不再调用 `build_provider_from_config()`。
5. 允许 `--reviewer-provider openai-compatible --reviewer-loop agent-loop`。
6. 清理旧 provider 测试或标记 legacy，确保业务主路径无直接 `ModelProvider` 依赖。

## 9. 成功标准

完成后应满足：

- `single-shot` 和 `agent-loop` 都通过 `ModelAdapter` 调用模型。
- CLI reviewer 主路径不再调用 `build_provider_from_config()`。
- Runtime / Reviewer / Orchestrator 不依赖 provider-specific schema。
- OpenAI-compatible 可以用于 agent-loop。
- DeepSeek 可以通过 openai-compatible 配置进入同一条路径。
- fake reviewer 仍可稳定用于本地 smoke。
- 全量测试通过。

## 10. 后续扩展

后续新增 Claude 时，只需要：

1. 新增 `ClaudeToolAdapter`。
2. 在 `ModelAdapterFactory` 增加 provider 分支。
3. 增加 Claude adapter tests。

不应修改：

- Agent Runtime。
- Tool Gateway。
- Observation Store。
- Reviewer result parser。
- Evidence Reconciler。

# Model Adapter Reviewer Agent Loop 设计

- 日期：2026-07-01
- 状态：待用户审阅
- 关联主设计：`docs/superpowers/specs/2026-06-22-evidence-driven-multi-agent-code-review-design.md`
- 目标分支：`codex/model-adapter-agent-loop`

## 1. 背景

当前 reviewer 路径已经具备：

- Provider 接口和 OpenAI-compatible single-shot 调用。
- Tool Gateway 与 Observation Store。
- Repository Intelligence。
- Multi-reviewer orchestration。
- Evidence Reconciliation 与 Completion Checker。

但 reviewer 仍然主要是“一次模型调用 + 已预置上下文”。这不符合主设计中 Reviewer Agent Loop 的目标：Reviewer 应该能在预算内按需搜索、读取、比较、观察，然后基于 observation 输出结论。

直接把 OpenAI、DeepSeek、Claude 等 tool calling 协议写进 Agent Runtime 会让业务逻辑被模型 API 绑死。因此本设计引入项目自己的统一模型协议和模型适配器层。

## 2. 目标

实现本地可运行的 Reviewer Agent Loop 第一版：

```text
Reviewer Agent Runtime
  只认识项目内部统一协议
        ↓
Model Adapter
  把统一协议转换为具体模型 API
        ↓
Model API
```

本版目标：

1. Runtime 只处理项目内部的 request、tool call、tool result、final result。
2. Provider/model API 差异只存在于 Adapter 内。
3. Reviewer 可以多轮请求工具。
4. 每次工具调用都必须经过 Tool Gateway，并写入 Observation Store。
5. 每个 reviewer 的 loop trace 可审计、可测试、可复现。
6. 现有 single-shot reviewer 路径保持兼容。

## 3. 非目标

本版不做：

- GitHub / PR 平台集成。
- Eval harness。
- Claude 原生 adapter。
- 多模型异质 reviewer 策略。
- 真并发 reviewer 执行。
- 动态补充 assignment。
- LLM Reconciler Agent。
- 长期记忆系统。

Claude adapter、SWE-PRBench、GitHub PR 输入都可以在后续版本接入，不影响本版架构。

## 4. 核心架构

### 4.1 内部统一模型协议

新增 `model_protocol.py`，定义 Runtime 认识的唯一协议。

核心类型：

```python
ModelToolSpec
ModelToolCall
ModelToolResult
ModelTurnRequest
ModelTurnResponse
ModelResponseKind
```

Runtime 只接收三种模型回合结果：

```text
tool_calls：模型请求一个或多个工具
final：模型返回最终 reviewer result JSON
invalid：adapter 无法把模型输出解析成项目协议
```

这里的 `tool_calls` 是项目内部对象，不等于 OpenAI / Claude / DeepSeek 的原始 tool call schema。

### 4.2 Model Adapter

新增 `model_adapter.py`。

Adapter 职责：

- 把 `ModelTurnRequest` 转成具体 provider 请求。
- 把具体 provider 响应转成 `ModelTurnResponse`。
- 保留 raw response 方便审计。
- 不执行工具。
- 不判断 Review Contract。
- 不校验证据。

第一版 adapter：

1. `FakeToolCallingAdapter`
   - 用于测试。
   - 可以按脚本返回 tool call，再返回 final result。
   - 不访问网络。

2. `OpenAICompatibleToolAdapter`
   - 面向 OpenAI-compatible chat completions tool calling。
   - DeepSeek 如果兼容 OpenAI tool calling，也先走此 adapter。
   - 使用现有 `OpenAICompatibleConfig` 和 HTTP transport 风格。

后续 adapter：

- `ClaudeToolAdapter`
- `JsonTextToolAdapter`，作为不支持原生 tool calling 模型的 fallback。

### 4.3 Reviewer Agent Runtime

新增 `agent_loop.py`。

Runtime 职责：

1. 根据 assignment、intent、diff excerpt、已有 observations 组装第一轮 `ModelTurnRequest`。
2. 调用 adapter。
3. 如果返回 `tool_calls`：
   - 校验预算。
   - 逐个通过 `ToolGateway.execute()` 执行。
   - 把生成的 observation ids 和 summary 包装成 `ModelToolResult`。
   - 追加到下一轮 messages。
4. 如果返回 `final`：
   - 用现有 `parse_reviewer_result()` 解析。
   - 返回 `AgentLoopRun`。
5. 如果超过 `max_turns` 或 `max_tool_calls`：
   - 返回 `ReviewerResultStatus.PARTIAL`。
   - 在 uncertainties 中记录预算耗尽。
6. 如果 adapter 或工具失败：
   - 记录失败回合。
   - 对当前 reviewer 返回 `FAILED` 或 `PARTIAL`，具体取决于是否已有可用 final result。

Runtime 不直接知道 OpenAI/DeepSeek/Claude 原始字段。

### 4.4 Trace 与 Artifact

新增 trace 数据结构：

```text
AgentLoopTrace
  trace_id
  turns
  tool_call_count
  final_status
```

每轮记录：

- turn index
- request message 摘要
- adapter kind
- raw response artifact name 或摘要
- tool calls
- tool results / observation ids
- parse errors 或 adapter errors

CLI multi mode 中写出：

```text
reviewer_<index>_agent_trace.json
```

single reviewer agent-loop mode 写出：

```text
reviewer_agent_trace.json
```

## 5. CLI 设计

新增参数：

```text
--reviewer-loop single-shot|agent-loop
```

默认：

```text
single-shot
```

原因：

- 保持现有行为稳定。
- agent-loop 第一版需要显式启用。

示例：

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

当 `--reviewer-provider fake --reviewer-loop agent-loop` 时，CLI 使用 `FakeToolCallingAdapter`，用于本地 smoke 和测试。

## 6. 与现有模块关系

### 6.1 保留 single-shot path

现有 `run_single_reviewer()` 继续存在。

本版新增：

```python
run_reviewer_agent_loop(...)
```

Orchestrator 根据 CLI 选择 reviewer runner：

```text
single-shot -> run_single_reviewer
agent-loop  -> run_reviewer_agent_loop
```

### 6.2 Tool Gateway

Agent Runtime 只能通过 `ToolGateway` 执行工具。

模型不能直接读文件、执行 git、绕过 Observation Store。

### 6.3 Evidence Reconciliation

Agent Loop 只负责产生 reviewer result 和 observations。

最终证据校验仍由 `evidence.py` 和 `completion.py` 执行。

## 7. Tool Calling 协议映射

### 7.1 项目内部 tool call

```python
ModelToolCall(
    call_id="call-1",
    tool_name="read_range",
    arguments={
        "path": "src/app.py",
        "revision": "head",
        "line_start": 1,
        "line_end": 80,
    },
)
```

### 7.2 OpenAI-compatible 映射

Adapter 把内部 `ModelToolSpec` 转成 OpenAI-compatible `tools`：

```json
{
  "type": "function",
  "function": {
    "name": "read_range",
    "description": "...",
    "parameters": {
      "type": "object",
      "properties": {...},
      "required": [...]
    }
  }
}
```

Adapter 把响应中的 `tool_calls` 转回内部 `ModelToolCall`。

工具结果以 OpenAI-compatible `tool` message 格式回传，但这个细节只在 adapter 内部或 adapter-owned message conversion 内处理，Runtime 不直接拼 provider 原始格式。

## 8. 错误处理

### 8.1 Adapter 输出无效

如果 adapter 无法解析模型输出：

- 当前 turn 记录 `invalid`。
- Runtime 若未到预算，可追加纠错提示再试一次。
- 达到预算后返回 `PARTIAL`，uncertainty 说明模型输出格式无效。

### 8.2 Tool Gateway 失败

工具失败不直接崩掉整个 reviewer：

- Runtime 记录 tool error。
- 把 tool error 作为 tool result 返回给模型。
- 如果模型最终无法完成，则返回 `PARTIAL`。

### 8.3 预算耗尽

预算来自 Assignment：

- `max_turns`
- `max_tool_calls`

耗尽时：

- 不再调用模型。
- 返回 `ReviewerResultStatus.PARTIAL`。
- trace 中记录 `budget_exhausted`。

## 9. 测试策略

第一版测试重点：

1. `model_protocol` 类型可序列化。
2. `FakeToolCallingAdapter` 能按脚本返回 tool call 和 final result。
3. `run_reviewer_agent_loop()` 能：
   - 接收 tool call。
   - 调用 `ToolGateway`。
   - 记录 observation。
   - 把 tool result 传回下一轮。
   - 解析 final reviewer result。
4. 工具预算耗尽返回 partial。
5. CLI `--reviewer-loop agent-loop` 写出 trace artifact。
6. 现有 `single-shot` 默认行为不变。

## 10. 分阶段实现

### 阶段 1：内部协议与 Fake Adapter

实现：

- `model_protocol.py`
- `model_adapter.py` 中的 `FakeToolCallingAdapter`
- 单元测试

### 阶段 2：Reviewer Agent Runtime

实现：

- `agent_loop.py`
- fake adapter + temporary repo integration tests
- trace serialization

### 阶段 3：CLI 集成

实现：

- `--reviewer-loop`
- single 和 multi mode 中的 agent-loop 分支
- artifact 写出

### 阶段 4：OpenAI-compatible Tool Adapter

实现：

- OpenAI-compatible tool request builder
- tool call parser
- transport tests

## 11. 成功标准

本版完成后，项目应满足：

- 本地 fake agent-loop smoke 可稳定运行。
- reviewer 能真正请求工具，而不是只消费预置 observation。
- Runtime 不依赖 OpenAI/DeepSeek/Claude 的原始 tool calling schema。
- OpenAI-compatible adapter 可以作为真实模型入口。
- 现有 single-shot 流程和测试全部保持通过。

## 12. 仍待后续实现

- Claude 原生 adapter。
- JsonText fallback adapter。
- 多 reviewer 并行执行。
- 动态补充 assignment。
- LLM Reconciler Agent。
- Eval Harness。
- GitHub / PR 集成。

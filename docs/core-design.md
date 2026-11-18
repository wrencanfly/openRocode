# rocode 核心设计（v0）

状态：已实现，代码在 `rocode/`。参考了 [pi](https://github.com/badlogic/pi-mono) 的 agent 核心（`packages/agent`），以及 [uni-robot-agent](https://github.com/Open-RUA/uni-robot-agent) 在物理动作上的一些约定。定位和路线见 [design.md](design.md)。

## 0. 目标与非目标

**目标**

- 一个很小、行为完全可预测的 agent 核心，让模型通过看图指点或写程序来操控机器人。
- 同一套核心能接不同的模型（Claude、OpenAI 兼容的开源模型）、不同的身体（MuJoCo、LIBERO、以后的真机）、不同的 policy 组合。
- 每一回合都能完整录制和回放，评测结果可复现。

**非目标（v0 不做）**

- 子 agent、计划模式、技能库。
- 训练好的策略（VLA）作为工具。
- 模型思考和机器人动作并行（流水线）。
- 双臂、灵巧手、移动底盘。
- 交互式终端界面。核心会预留事件流，界面以后作为扩展接上。

## 1. 分层

```
  runner      评测：任务 × 种子 × 模型 × 接口配置 → 结果表
     │
  session     一个回合的生命周期：建身体、装扩展、跑循环、写录制
     │
  core        agent 循环、事件、钩子、控制通道        ← 不认识机器人
     │    ╲
  provider  toolset + extensions                      ← 不认识具体身体
  (模型适配)  指点工具、安全闸门、上下文策略、录制器
                 │
               body       观测和动作的统一接口        ← 不认识模型
                 │
          MuJoCo / RoboDojo HTTP / 真机
```

依赖只能从上往下。`core` 不能导入 `body`，`body` 不能导入 `core`。

## 2. 名词

| 名词 | 含义 |
|---|---|
| episode（回合） | 从任务开始到结束的一次完整运行 |
| turn（轮） | 模型的一次回复，加上它请求的全部工具调用 |
| step（步） | 一次工具调用，是预算计数的单位 |
| observation（观测） | 某一时刻的一组相机图片、深度、标定和本体状态，有唯一 id，不可变 |
| event（事件） | 核心对外发布的一切变化，录制、界面、评测都只靠事件 |

## 3. 消息模型

核心内部使用自己的规范消息格式，由模型适配器负责和各家格式互转。图片在消息里只存引用，字节放在观测仓库里，录制文件因此不会膨胀。

```python
@dataclass(frozen=True)
class ImageRef:
    observation_id: str      # 所属观测
    camera: str              # 相机名
    # 字节从 ObservationStore 取，消息里不存

Content = Text | ImageRef

@dataclass(frozen=True)
class UserMessage:      content: list[Content]
@dataclass(frozen=True)
class AssistantMessage: blocks: list[Any]          # 原样保存模型返回的块，包括思考块，回传时一字不改
@dataclass(frozen=True)
class ToolResultMessage:
    call_id: str
    content: list[Content]   # 给模型看的
    is_error: bool
@dataclass(frozen=True)
class OperatorMessage:  text: str                  # 操作员插话，见 §8
```

两条硬规则：

1. **历史只追加，不修改。** Claude 的思考块绑定了它之前的全部对话，客户端改动或删除任何更早的内容，都会让之后的思考块失效（请求被拒绝，或推理被丢弃），也会让提示缓存失效。所有“缩短上下文”的需求都走 §11 的上下文策略。
2. **助手消息原样回传。** 适配器不能重新拼装模型返回的块。

## 4. 工具契约

```python
class Tool(Protocol):
    name: str
    description: str
    input_schema: dict          # JSON Schema，必须 additionalProperties: false
    kind: Literal["observe", "act", "control"]
    # observe：只读，比如 look
    # act：改变物理世界，必须顺序执行，可被急停打断
    # control：结束回合等，比如 done

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult: ...

@dataclass
class ToolContext:
    body: Body
    observations: ObservationStore
    cancel: CancelToken          # 急停或中止时置位，act 类工具必须在每个控制周期检查

@dataclass
class ToolResult:
    content: list[Content]                 # 给模型看的：文字和标注图
    is_error: bool = False
    outcome: ActionOutcome | None = None   # 只给 act 类工具，不给模型看，供核心和评测使用
    details: dict = field(default_factory=dict)   # 只进录制，不给模型看
    ends_episode: bool = False

@dataclass(frozen=True)
class ActionOutcome:
    executed: Literal["none", "partial", "full", "unknown"]
    stop_reason: Literal["reached", "contact", "blocked", "refused", "cancelled", "fault"]
    stop_confirmed: bool         # 身体确认已经停稳
```

约定（部分借鉴 uni-robot-agent）：

- **物理动作永不自动重试。** 执行结果是 `unknown` 时，核心不会重放这个动作，而是如实告诉模型。
- 参数先按 JSON Schema 校验，不合法直接返回错误，不做猜测性修正。
- 工具返回的文字只描述发生了什么，不含世界坐标（这是论点本身，有测试守护）。

## 5. 循环

```python
def run(ctx: SessionContext) -> EpisodeEnd:
    emit(EpisodeStart)
    append(UserMessage(task + 初始观测))
    while True:
        if 达到任何预算: return end("budget")
        drain(control_queue)                       # 操作员插话、暂停（§8）
        request = hooks.before_request(当前上下文)  # 上下文策略在这里生效（§11）
        emit(TurnStart)
        reply = provider.complete(request, cancel)  # 可被急停中止
        append(reply); emit(AssistantMessage)
        if reply.refused:          return end("refusal")
        if not reply.tool_calls:   append(提示：继续，或调用 done); continue

        stopped_early = False
        for call in reply.tool_calls:              # 按模型给出的顺序，逐个执行
            if stopped_early and tool(call).kind == "act":
                result = skipped("同一轮里前一个动作提前停止了，这个动作没有执行")
            else:
                decision = hooks.before_tool(call)          # 可放行、拒绝、改写参数、终止
                result = decision.result or tool.execute(call.args, tctx)
                result = hooks.after_tool(call, result)
                stopped_early |= result.outcome and result.outcome.stop_reason != "reached"
            emit(ToolEnd(call, result))
        append(全部工具结果，作为一条消息)
        if any(r.ends_episode for r in results): return end("done")
```

要点：

- **单线程、同步。** 模型调用和物理动作不并行。急停通过另一个线程置位 `cancel` 实现（§8）。
- **同一轮里有多个动作时，只要一个提前停止（碰撞、受阻），后面的动作全部跳过**并告知模型。模型基于“上一个动作会成功”写下的后续动作，在世界已经变化后执行是危险的。
- 预算：最大步数、最大轮数、token 或美元上限、墙钟时间，任何一个先到就结束。

## 6. 事件

```
EpisodeStart(task, config)        EpisodeEnd(reason, stats)
TurnStart(turn)                   AssistantMessage(turn, text, tool_calls, usage)
ToolStart(call)                   ToolProgress(call, info)      ToolEnd(call, result, duration)
OperatorInput(text)               Paused / Resumed              EStop(source)
ObservationTaken(id, cameras)     ContextTrimmed(policy, info)  BudgetWarning(kind, used, limit)
```

事件只描述发生了什么，订阅者不能通过事件改变控制流，要改控制流用钩子。录制器、终端界面、实时看板都只是事件订阅者。

## 7. 钩子与扩展

```python
class Extension:
    tools: list[Tool] = []
    prompt: str = ""            # 追加到系统提示词的片段，回合开始时固定，之后不变
    def before_request(self, req: Request) -> Request: ...
    def before_tool(self, call: ToolCall, ctx) -> Decision | None: ...
        # Decision = Allow | Deny(reason) | Rewrite(args) | Terminate(reason)
    def after_tool(self, call: ToolCall, result: ToolResult) -> ToolResult: ...
    def on_event(self, event: Event) -> None: ...
```

- 扩展按注册顺序执行。`before_tool` 里第一个返回 Deny 或 Terminate 的生效，后面的不再调用。
- 系统提示词和工具列表在回合开始时组装完毕，**回合中途不再改变**。这是缓存和思考块的要求；需要中途加指令时，追加一条消息。
- v0 内置的扩展：

| 扩展 | 用的钩子 | 作用 |
|---|---|---|
| `pointing` | tools, prompt | 指点式工具集（就是现在的 `tools.py`，改成通过 body 接口工作） |
| `safety_gate` | before_tool | 工作空间、速度、目标合理性检查；被拒绝的动作返回原因，并确认没有执行 |
| `context_policy` | before_request | 只保留最近 K 次观测的图片（§11） |
| `recorder` | on_event | 写录制文件（§13） |
| `budget` | before_request, on_event | 统计 token 和成本，接近上限时提醒模型 |

消融实验就是换扩展的组合或参数，例如把 `pointing` 换成“输出世界坐标”的工具集作为基线。

## 8. 控制通道

操作员可以在回合进行中从另一个线程（终端界面、HTTP、测试）发指令：

| 指令 | 效果 | 何时生效 |
|---|---|---|
| `say(text)` | 插话，例如“不是这个，是左边那个” | 下一次请求模型之前，作为 OperatorMessage 追加 |
| `pause()` / `resume()` | 暂停在下一个安全点，机器人保持当前位置 | 当前工具执行完之后 |
| `estop()` | 立即停下 | 马上：置位 cancel，调用 `body.stop()`，中止正在进行的模型请求，回合以 `estop` 结束 |

`say` 对应 pi 的插话（steering）。在机器人上，它是语言层面的人工接管，比抢过手柄驾驶更常用。

## 9. 身体接口

> 实现时有调整，以 `rocode/body/base.py` 为准：方法是 `info / observe / move / rotate / grip / stop / reset / ground_truth`，所有动作都返回 `Receipt`。

工具只能通过这个接口碰机器人，不能直接访问仿真内部。

```python
class Body(Protocol):
    def describe(self) -> BodyInfo            # 相机列表、工作空间、夹爪量程、控制频率
    def observe(self, cameras=None) -> Observation
    def move(self, goal: Pose, guard: Guard, cancel: CancelToken) -> MotionReceipt
    def grip(self, opening: float | Literal["close", "open"], cancel) -> GripReceipt
    def stop(self) -> StopReceipt             # 任何线程都可以调用，必须很快返回
    def ground_truth(self) -> dict            # 只给评测用，核心和工具都不能调用

@dataclass(frozen=True)
class Observation:
    id: str
    images: dict[str, np.ndarray]             # 彩色图
    depth: dict[str, np.ndarray] | None       # 没有深度的身体为 None
    cameras: dict[str, CameraModel]           # 内参、外参，属于这一帧
    proprio: Proprio                          # 指尖位置、朝向、夹爪开度、腕部力
    t: float

@dataclass(frozen=True)
class Guard:
    max_force_n: float | None = 4.0           # 相对动作开始时清零的力
    max_speed_mps: float = 0.18
```

**指点引用的是某一张具体的图。** `move_to` 的点击应当用模型当时看到的那张图（那次观测的深度和标定）来换算三维点，而不是用执行时重新拍的图。否则夹爪移动后挡住了那个像素，换算出来的就是夹爪表面。因此 `move_to` 带一个可选的 `observation` 参数，默认是模型最近收到的那组图，所有返回的图上都标出它的 id。现在的原型没有这样做，需要改。

v0 实现：`MujocoBody`（现在的原型）。下一步：`RoboDojoBody`，把上面的调用翻译成 uni-robot-agent 环境服务的 `/captures`、`/steps`、`/finish`。

## 10. 模型适配器

```python
class Provider(Protocol):
    capabilities: Capabilities   # 是否支持工具结果里放图、服务端清理上下文、思考块……
    def complete(self, request: Request, cancel: CancelToken) -> Reply
```

v0 只做两个：

- **Anthropic**：Messages API 加工具调用；思考块原样回传；系统提示词和工具开启缓存；开启拒答时的服务端回退（`fallbacks: "default"`）；用户可以设定推理强度（effort）。
- **OpenAI 兼容**（覆盖 Qwen、DeepSeek、本地 vLLM）：这类接口的工具结果消息一般只能放文字，所以图片挪到紧跟在工具结果之后的一条用户消息里，并注明对应哪个调用。

`Reply` 统一成：文字、工具调用列表、是否拒答、用量（输入、输出、缓存读取的 token 数），以及原始块（回传用）。

## 11. 上下文策略

每一步都带图，几十步后上下文会很大。策略统一定义为“只保留最近 K 次观测的图片，更早的替换成一行文字说明”，但实现方式由适配器根据能力决定：

- **Anthropic**：不能在客户端删历史（§3 规则 1），改用服务端的工具结果清理（context editing，beta）。具体参数在实现前要对照当时的文档确认。
- **OpenAI 兼容**：没有思考块绑定，可以在客户端把旧图片替换成文字。

做不到的策略要在回合开始时报错，而不是悄悄失效。

## 12. 安全分层

安全从来不依赖模型自觉：

1. **模型**：提示词里的建议，不作为任何保证。
2. **外壳**：`safety_gate` 扩展在执行前拒绝越界或不合理的目标，并确认没有执行。
3. **身体**：每个动作都带 `Guard`，力或速度超限就停下后退；`stop()` 随时可用。
4. **真机**（v0 不做）：独立的高频安全回路，跑在机器人旁边，不经过 Python 和网络。

## 13. 录制格式

> 已实现，`requests/` 目录除外（还没做）。

每个回合一个目录：

```
episode.jsonl        事件流，每行一个事件（不含图片字节）
observations/        <观测id>_<相机>.png，以及深度和标定（.npz）
requests/            每次发给模型的完整请求（可选，用于排查缓存和思考块问题）
result.json          结束原因、步数、token、成本、耗时、模型自报的成功
evaluation.json      评测器根据真值判定的成功，与模型看到的内容完全分开
```

有了事件流加观测，可以不调用模型就把整个回合重新画出来，或者从任意一步重放给另一个模型。

## 14. 评测运行器

> 还没实现。目前每次 `rocode run` 跑一个回合并写 `evaluation.json`。

```
rocode eval --tasks mug_to_coaster,lift_block --seeds 0-9 \
            --models claude-opus-5-5,qwen-... --interface pointing,world_coords
```

输出每个组合的成功率和置信区间、平均步数、token、成本、耗时，以及失败原因分布（结束原因加录制里的停止原因）。实现时每个回合是独立进程，避免仿真状态互相影响。

## 15. 目录结构与代码量预算

```
rocode/
  core/        loop.py  events.py  hooks.py  messages.py  control.py      约 500 行
  providers/   anthropic.py  openai_compat.py                              约 300 行
  body/        base.py  mujoco.py  robodojo.py                             约 600 行
  ext/         pointing.py  safety_gate.py  context_policy.py  recorder.py  budget.py
  runner/      session.py  eval.py  tasks.py
```

核心（`core/`）要求不超过约 500 行；超过了说明有东西放错了层。

## 15.5 Code as Policy（已实现，`ext/code.py`）

- `run_code(code)` 是一个普通的 `act` 类工具。程序在独立的 Python 子进程（隔离模式）里运行，通过一对本地 socket 和外壳通信。
- 程序里有一个现成的 `robot` 对象：`look`、`move_to`、`lower`、`nudge`、`turn`、`grip`。每个调用都交回外壳，走和模型直接调用完全相同的路径：参数校验、钩子（安全闸门）、事件、急停。程序做不到模型直接调用做不到的事。
- 每个调用返回结构化结果：`ok`、`text`、`observation`、`height_cm`、`opening_cm`、`holding`、`contact_n`、`stop`，`lower` 还有 `touched`。同样不含世界坐标。
- 程序看不到图片：模型先看图选好像素，再写程序处理循环、重试和数值判断；需要再看图时结束程序。
- 限额：每段程序最多 80 次机器人调用、180 秒，超出即终止并如实报告。急停会立刻中止程序。
- 这不是安全沙箱：程序以用户权限运行。真正的隔离（容器、禁网）留到以后。

**policy 组合**由 `SessionConfig.policy` 决定：

| policy | 模型能直接调用的工具 | 程序能调用的 |
|---|---|---|
| `tools` | look、move_to、lower、nudge、turn、grip、done | —（没有 run_code） |
| `code` | look、run_code、done | 全部运动工具（对模型隐藏，称为内部工具） |
| `both`（默认） | 全部，包括 run_code | 全部运动工具 |

## 15.6 实现中定下的约束

- **线程**：身体的方法（除 `stop` 外）必须在创建它的线程上调用。MuJoCo 的渲染上下文属于那个线程，在 macOS 上跨线程渲染会卡死。所以回合循环跑在主线程，操作员输入（终端插话、暂停、急停）放在后台线程，通过控制通道传递。
- **点击绑定观测**：每张图都印上观测 id（o1、o2……），`move_to` 默认用最近一组图，也可以指定。用的是那一组的深度和标定，所以夹爪之后挡住那个像素也不影响（有测试守护）。
- **代码量**：核心（`core/`）实际约 700 行，超过了原定的 500 行，主要多在事件定义上。

## 16. 已定的问题（原“待决定”）

1. 适配器：v0 只做 Anthropic 和 OpenAI 兼容两个。**已实现。**
2. 同步加急停线程，不用 asyncio。**已实现。**
3. 点击的观测 id 可选，默认最近一组图。**已实现。**
4. 每步返回全部相机、640×480 的图。可以用 `--keep-images N` 只保留最近 N 次结果的图。
5. （原问题 5 是论文对比用的基线，项目不再以发论文为目标，取消。）

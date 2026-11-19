"""The agent loop: ask the model, run the tools it asks for, repeat.

Single-threaded and synchronous. The only thing that crosses threads is the ControlChannel, whose
e-stop cancels motion immediately and is checked at every safe point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..body.base import Body
from ..providers.base import ContextPolicy, Provider, ProviderError, Request, Usage
from .control import ControlChannel
from .events import (AssistantReply, EpisodeEnd, EpisodeStart, EStop, EventBus, Notice, ObservationTaken,
                     OperatorInput, Paused, Resumed, ToolDenied, ToolEnd, ToolStart, TurnStart)
from .extension import Deny, Extension, Rewrite, Terminate
from .messages import (AssistantMessage, Message, OperatorMessage, Text, ToolCall, ToolResultBlock,
                       ToolResultsMessage, UserMessage)
from .observations import ObservationStore
from .tooling import Tool, ToolContext, ToolResult, validate

BASE_PROMPT = """\
You control a robot through tools, the way you would control a computer: look at camera pictures, act, \
then check the result in the new pictures before deciding the next step. Tool results describe what \
actually happened; a call that returns does not mean the physical goal was achieved. Never repeat a \
physical action just to see its result again. If something goes wrong, look again and correct it. When \
the task is complete, or you cannot make further progress, call done exactly once."""

NUDGE = "Continue with the tools, or call done if you are finished."


@dataclass
class Budget:
    max_steps: int = 60          # top-level tool calls made by the model
    max_turns: int = 60          # model replies
    max_seconds: float = 1800.0  # wall clock


@dataclass
class LoopConfig:
    effort: str = "medium"
    max_tokens: int = 16000
    budget: Budget = field(default_factory=Budget)
    context: ContextPolicy = field(default_factory=ContextPolicy)


@dataclass
class EpisodeOutcome:
    reason: str            # done | refusal | budget_steps | budget_turns | budget_time | estop | max_tokens | error
    steps: int
    turns: int
    seconds: float
    usage: Usage
    finish: dict | None    # what the model said when it called done
    error: str = ""


class Agent:
    """One episode's worth of state: transcript, tools, hooks, body."""

    def __init__(self, provider: Provider, body: Body, extensions: list[Extension],
                 config: LoopConfig | None = None, control: ControlChannel | None = None,
                 bus: EventBus | None = None):
        self.provider = provider
        self.body = body
        self.extensions = extensions
        self.config = config or LoopConfig()
        self.control = control or ControlChannel()
        self.control.attach(body)
        self.bus = bus or EventBus()
        for ext in extensions:
            self.bus.subscribe(ext.on_event)
        self.observations = ObservationStore(
            on_put=lambda item: self.bus.emit(ObservationTaken(item.id, sorted(item.views))))
        self.messages: list[Message] = []
        self.tools: dict[str, Tool] = {}        # everything callable
        self.exposed: set[str] = set()          # what the model may call directly
        for ext in extensions:
            for tool in [*ext.tools, *getattr(ext, "internal_tools", [])]:
                if tool.name in self.tools:
                    raise ValueError(f"two extensions both define a tool called {tool.name}")
                self.tools[tool.name] = tool
            self.exposed |= {t.name for t in ext.tools}
        # Fixed for the episode: rebuilding either mid-conversation would break caches and thinking.
        self.system = "\n\n".join([BASE_PROMPT, *(e.prompt for e in extensions if e.prompt)])
        self.tool_specs = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                           for t in self.tools.values() if t.name in self.exposed]
        self.finish: dict | None = None

    # ---- running tools ------------------------------------------------------------------

    def run_tool(self, call: ToolCall, parent_id: str | None = None) -> ToolResult:
        """Validate, apply hooks, execute, publish. Also used by code-as-policy for nested calls."""
        tool = self.tools.get(call.name)
        if tool is not None and parent_id is None and call.name not in self.exposed:
            tool = None  # internal tools are reachable from programs only
        self.bus.emit(ToolStart(call.id, call.name, dict(call.args), parent_id))
        started = time.time()
        if tool is None:
            result = ToolResult.error(f"There is no tool called {call.name}.")
        elif err := validate(tool.input_schema, call.args):
            result = ToolResult.error(f"Invalid arguments for {call.name}: {err}. Nothing was executed.")
        elif self.control.cancel.cancelled and tool.kind == "act":
            result = ToolResult.error("Emergency stop is active. Nothing was executed.")
        else:
            result = self._execute_with_hooks(call, tool, parent_id)
        self.bus.emit(ToolEnd(
            call.id, call.name, result.is_error, "\n".join(c.text for c in result.content if isinstance(c, Text)),
            sorted({c.observation_id for c in result.content if hasattr(c, "observation_id")}),
            vars(result.receipt) if result.receipt else None, round(time.time() - started, 3), parent_id,
            {k: v for k, v in result.data.items() if isinstance(v, (str, int, float, bool, type(None)))}))
        return result

    def _execute_with_hooks(self, call: ToolCall, tool: Tool, parent_id: str | None) -> ToolResult:
        for ext in self.extensions:
            decision = ext.before_tool(call, tool)
            if isinstance(decision, Deny):
                self.bus.emit(ToolDenied(call.id, call.name, decision.reason))
                return ToolResult.error(f"Refused: {decision.reason} Nothing was executed.")
            if isinstance(decision, Terminate):
                self.bus.emit(ToolDenied(call.id, call.name, decision.reason))
                self.control.estop(f"hook {ext.name}: {decision.reason}")
                return ToolResult(content=(Text(f"Stopped: {decision.reason}"),), is_error=True, ends_episode=True)
            if isinstance(decision, Rewrite):
                call = ToolCall(call.id, call.name, decision.args)
        ctx = self.tool_context(call.id)
        try:
            result = tool.execute(dict(call.args), ctx)
        except Exception as err:  # noqa: BLE001 - a tool bug must reach the model, not crash the robot
            self.body.stop()
            result = ToolResult.error(f"{call.name} failed inside the harness: {err!r}. The arm was told to hold.")
        for ext in self.extensions:
            result = ext.after_tool(call, tool, result)
        return result

    # ---- the loop -----------------------------------------------------------------------

    def tool_context(self, call_id: str = "setup") -> ToolContext:
        return ToolContext(self.body, self.observations, self.control.cancel, call_id,
                           run_tool=lambda c: self.run_tool(c, parent_id=call_id))

    def run(self, task: str, first_content: tuple = (), meta: dict | None = None) -> EpisodeOutcome:
        cfg = self.config
        start = time.time()
        usage = Usage()
        steps = turns = 0
        self.bus.emit(EpisodeStart(task, {"provider": self.provider.name, "model": self.provider.model,
                                          "effort": cfg.effort, "tools": sorted(self.exposed),
                                          "budget": vars(cfg.budget), **(meta or {})}))
        if callable(first_content):  # e.g. a first look, taken after EpisodeStart so recordings stay ordered
            first_content = tuple(first_content())
        self.messages.append(UserMessage((Text(f"Task: {task}"), *first_content)))

        def end(reason: str, error: str = "") -> EpisodeOutcome:
            outcome = EpisodeOutcome(reason, steps, turns, round(time.time() - start, 1), usage, self.finish, error)
            self.bus.emit(EpisodeEnd(reason, {"steps": steps, "turns": turns, "seconds": outcome.seconds,
                                              "usage": vars(usage), "finish": self.finish, "error": error}))
            return outcome

        while True:
            if self.control.paused:
                self.bus.emit(Paused())
                self.control.wait_while_paused()
                self.bus.emit(Resumed())
            if self.control.cancel.cancelled:
                self.bus.emit(EStop(self.control.estop_source or "unknown"))
                return end("estop")
            for text in self.control.drain_said():
                self.bus.emit(OperatorInput(text))
                self.messages.append(OperatorMessage(text))
            if steps >= cfg.budget.max_steps:
                return end("budget_steps")
            if turns >= cfg.budget.max_turns:
                return end("budget_turns")
            if time.time() - start > cfg.budget.max_seconds:
                return end("budget_time")

            turns += 1
            self.bus.emit(TurnStart(turns))
            request = Request(self.system, self.tool_specs, list(self.messages), self.observations,
                              cfg.effort, cfg.max_tokens, cfg.context)
            for ext in self.extensions:
                request = ext.before_request(request)
            try:
                reply = self.provider.complete(request)
            except ProviderError as err:
                return end("error", str(err))
            usage.add(reply.usage)
            self.bus.emit(AssistantReply(turns, reply.text, [vars(c) for c in reply.tool_calls], vars(reply.usage),
                                         reply.stop_reason, reply.reasoning))
            if reply.stop_reason == "refusal":
                return end("refusal")
            self.messages.append(AssistantMessage(reply.text, reply.tool_calls, reply.raw, self.provider.name))
            if not reply.tool_calls:
                if reply.stop_reason == "max_tokens":
                    return end("max_tokens")
                self.messages.append(UserMessage((Text(NUDGE),)))
                continue

            blocks, stopped_early, ends = [], False, False
            for call in reply.tool_calls:
                steps += 1
                tool = self.tools.get(call.name)
                if stopped_early and tool is not None and tool.kind == "act":
                    result = ToolResult.error("Skipped: an earlier action in this turn stopped early, so this one "
                                              "was not executed. Look at the latest pictures and decide again.")
                    self.bus.emit(Notice(f"skipped {call.name} ({call.id}) after an early stop"))
                else:
                    result = self.run_tool(call)
                    stopped_early |= result.stopped_early
                if result.ends_episode and tool is not None and tool.kind == "control":
                    self.finish = dict(result.data)
                ends |= result.ends_episode
                blocks.append(ToolResultBlock(call.id, tuple(result.content), result.is_error))
            self.messages.append(ToolResultsMessage(tuple(blocks)))
            if self.control.cancel.cancelled:
                self.bus.emit(EStop(self.control.estop_source or "unknown"))
                return end("estop")
            if ends:
                return end("done")

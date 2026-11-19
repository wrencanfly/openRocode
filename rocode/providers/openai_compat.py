"""Any OpenAI-compatible Chat Completions endpoint: OpenAI, Qwen, DeepSeek, a local vLLM server.

Chat Completions tool messages carry text only, so a turn's images go in one user message placed right
after its tool messages, labelled with the call they belong to. The context policy is applied client-side by
replacing old images with a short note.

With reasoning on, the request asks for `reasoning_effort` and a reasoning summary, which reasoning models and
proxies such as LiteLLM understand. The summary is recorded for people to read. Thinking blocks a proxy returns
for Claude are replayed with their assistant turn, as Anthropic requires when tools are used.
"""

from __future__ import annotations

import base64
import json

from ..core.messages import (AssistantMessage, Content, ImageRef, OperatorMessage, Text, ToolCall,
                             ToolResultsMessage, UserMessage)
from .base import ProviderError, Reply, Request, Usage

FINISH = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}


class OpenAICompatProvider:
    name = "openai"

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 send_reasoning_effort: bool = False, reasoning_summary: bool = False, client=None):
        if client is None:
            import openai

            client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.client = client
        self.model = model
        self.send_reasoning_effort = send_reasoning_effort
        self.reasoning_summary = reasoning_summary

    def complete(self, request: Request) -> Reply:
        params = self.build_params(request)
        try:
            response = self.client.chat.completions.create(**params)
        except Exception as err:
            # Some servers (Claude behind Bedrock, plain vLLM) reject the summary request. It only changes what we
            # record, and no tool has run yet, so ask again without it and stop asking for this episode.
            if "extra_body" not in params or "reasoning" not in str(err):
                raise ProviderError(f"{type(err).__name__}: {err}") from err
            self.reasoning_summary = False
            params.pop("extra_body")
            try:
                response = self.client.chat.completions.create(**params)
            except Exception as err2:
                raise ProviderError(f"{type(err2).__name__}: {err2}") from err2
        choice = response.choices[0]
        msg = choice.message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": tc.function.arguments}
            calls.append(ToolCall(tc.id, tc.function.name, args if isinstance(args, dict) else {"_value": args}))
        u = response.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0 if u else 0
        usage = Usage(getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0, cached, 0)
        blocks = getattr(msg, "thinking_blocks", None)
        reasoning = getattr(msg, "reasoning_content", None) or ""
        return Reply(msg.content or "", tuple(calls), FINISH.get(choice.finish_reason, choice.finish_reason or ""),
                     {"thinking_blocks": blocks} if blocks else None, usage, getattr(response, "model", ""),
                     reasoning.strip() if isinstance(reasoning, str) else "")

    def build_params(self, request: Request) -> dict:
        params = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": self.render(request),
            "tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                        "parameters": t["input_schema"]}} for t in request.tools],
        }
        if self.send_reasoning_effort:
            params["reasoning_effort"] = request.effort
        if self.reasoning_summary:
            params["extra_body"] = {"reasoning": {"summary": "auto"}}
        return params

    def render(self, request: Request) -> list[dict]:
        keep = request.context.keep_image_turns
        result_turns = [i for i, m in enumerate(request.messages) if isinstance(m, ToolResultsMessage)]
        recent = set(result_turns[-keep:]) if keep is not None else set(result_turns)

        out: list[dict] = [{"role": "system", "content": request.system}]
        for i, m in enumerate(request.messages):
            if isinstance(m, UserMessage):
                out.append({"role": "user", "content": self._parts(m.content, request, keep_images=True)})
            elif isinstance(m, OperatorMessage):
                out.append({"role": "user", "content": f"Operator says: {m.text}"})
            elif isinstance(m, AssistantMessage):
                entry: dict = {"role": "assistant", "content": m.text or None}
                if isinstance(m.raw, dict) and m.raw.get("thinking_blocks"):
                    entry["thinking_blocks"] = m.raw["thinking_blocks"]   # replayed as returned
                if m.tool_calls:
                    entry["tool_calls"] = [{"id": c.id, "type": "function",
                                            "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                                           for c in m.tool_calls]
                out.append(entry)
            elif isinstance(m, ToolResultsMessage):
                pictures: list[dict] = []
                for r in m.results:
                    text = "\n".join(c.text for c in r.content if isinstance(c, Text))
                    out.append({"role": "tool", "tool_call_id": r.call_id,
                                "content": ("[error] " if r.is_error else "") + text})
                    images = [c for c in r.content if isinstance(c, ImageRef)]
                    if images:
                        pictures.append({"type": "text", "text": f"Pictures from call {r.call_id}:"})
                        pictures += self._parts(tuple(images), request, keep_images=i in recent)
                if pictures:
                    out.append({"role": "user", "content": pictures})
        return out

    @staticmethod
    def _parts(content: tuple[Content, ...], request: Request, keep_images: bool) -> list[dict]:
        parts: list[dict] = []
        for c in content:
            if isinstance(c, Text):
                parts.append({"type": "text", "text": c.text})
            elif isinstance(c, ImageRef):
                if not keep_images:
                    parts.append({"type": "text", "text": f"[{c.observation_id} {c.camera}: older picture removed]"})
                    continue
                data = base64.standard_b64encode(request.images.png(c.observation_id, c.camera)).decode()
                parts.append({"type": "text", "text": f"[{c.observation_id} {c.camera}]"})
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
        return parts

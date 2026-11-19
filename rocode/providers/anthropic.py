"""Claude via the Anthropic Messages API.

History rules this adapter keeps (see docs/core-design.md §3):
- assistant turns are replayed exactly as the API returned them, thinking blocks included;
- every other message renders to identical bytes on every request (images come from the immutable store);
- old tool results are trimmed only by server-side context editing, never by editing history here.
"""

from __future__ import annotations

import base64

from ..core.messages import (AssistantMessage, Content, ImageRef, OperatorMessage, Text, ToolCall,
                             ToolResultsMessage, UserMessage)
from .base import ProviderError, Reply, Request, Usage

# Models that accept server-side refusal fallbacks (`fallbacks: "default"`).
FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
CONTEXT_BETA = "context-management-2025-06-27"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model

    def complete(self, request: Request) -> Reply:
        params = self.build_params(request)
        try:
            response = self.client.beta.messages.create(**params)
        except Exception as err:  # SDK errors after its own retries; the loop ends the episode
            raise ProviderError(f"{type(err).__name__}: {err}") from err
        calls = tuple(ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use")
        text = "\n".join(b.text for b in response.content if b.type == "text")
        u = response.usage
        usage = Usage(u.input_tokens or 0, u.output_tokens or 0,
                      getattr(u, "cache_read_input_tokens", 0) or 0, getattr(u, "cache_creation_input_tokens", 0) or 0)
        reasoning = "\n".join(getattr(b, "thinking", "") or "" for b in response.content if b.type == "thinking").strip()
        return Reply(text, calls, response.stop_reason or "", response.content, usage, getattr(response, "model", ""),
                     reasoning)

    def build_params(self, request: Request) -> dict:
        betas: list[str] = []
        params: dict = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "tools": request.tools,
            "messages": self.render(request),
            "output_config": {"effort": request.effort},
            "cache_control": {"type": "ephemeral"},
        }
        if self.model in FALLBACK_MODELS:
            betas.append(FALLBACK_BETA)
            params["fallbacks"] = "default"
        keep = request.context.keep_image_turns
        if keep is not None:
            betas.append(CONTEXT_BETA)
            params["context_management"] = {"edits": [{
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 60000},
                "keep": {"type": "tool_uses", "value": max(1, keep)},
                # Clearing re-writes the cache; only do it when it frees a worthwhile amount.
                "clear_at_least": {"type": "input_tokens", "value": 15000},
                "exclude_tools": ["done"],
            }]}
        if betas:
            params["betas"] = betas
        return params

    def render(self, request: Request) -> list[dict]:
        out: list[dict] = []
        for m in request.messages:
            if isinstance(m, UserMessage):
                out.append({"role": "user", "content": self._content(m.content, request)})
            elif isinstance(m, OperatorMessage):
                out.append({"role": "user", "content": [{"type": "text", "text": f"Operator says: {m.text}"}]})
            elif isinstance(m, AssistantMessage):
                if m.provider == self.name and m.raw is not None:
                    out.append({"role": "assistant", "content": m.raw})
                else:  # history from another provider: text and calls only
                    blocks: list[dict] = [{"type": "text", "text": m.text}] if m.text else []
                    blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in m.tool_calls]
                    out.append({"role": "assistant", "content": blocks})
            elif isinstance(m, ToolResultsMessage):
                out.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r.call_id, "content": self._content(r.content, request),
                     "is_error": r.is_error}
                    for r in m.results]})
        return out

    @staticmethod
    def _content(content: tuple[Content, ...], request: Request) -> list[dict]:
        blocks: list[dict] = []
        for c in content:
            if isinstance(c, Text):
                blocks.append({"type": "text", "text": c.text})
            elif isinstance(c, ImageRef):
                data = base64.standard_b64encode(request.images.png(c.observation_id, c.camera)).decode()
                blocks.append({"type": "text", "text": f"[{c.observation_id} {c.camera}]"})
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
        return blocks

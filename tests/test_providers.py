"""Provider adapters render the canonical history correctly. Fake clients; no network."""

import base64
import json
from types import SimpleNamespace

import numpy as np

from rocode.body.base import CameraModel, Observation, Proprio
from rocode.core.messages import (AssistantMessage, ImageRef, OperatorMessage, Text, ToolCall, ToolResultBlock,
                                  ToolResultsMessage, UserMessage)
from rocode.core.observations import ObservationStore
from rocode.providers.anthropic import AnthropicProvider
from rocode.providers.base import ContextPolicy, Request
from rocode.providers.openai_compat import OpenAICompatProvider

TOOLS = [{"name": "look", "description": "Look.", "input_schema": {"type": "object", "properties": {}}}]


def store_with(n: int) -> ObservationStore:
    store = ObservationStore()
    cam = CameraModel("front", 4, 4, 2.0, np.zeros(3), np.eye(3))
    obs = Observation(0.0, {}, None, {"front": cam}, Proprio(np.zeros(3), 0.0, 0.0, 0.0, 0.0))
    for i in range(n):
        store.put(obs, {"front": f"png-{i + 1}".encode()})
    return store


def history(store, provider_name: str, raw=None):
    return [
        UserMessage((Text("Task: x"), ImageRef("o1", "front"))),
        AssistantMessage("thinking aloud", (ToolCall("c1", "look", {}),), raw, provider_name),
        ToolResultsMessage((ToolResultBlock("c1", (Text("Fresh."), ImageRef("o2", "front"))),)),
        OperatorMessage("Use the left one."),
        AssistantMessage("", (ToolCall("c2", "look", {}),), raw, provider_name),
        ToolResultsMessage((ToolResultBlock("c2", (Text("Again."), ImageRef("o3", "front")), is_error=True),)),
    ]


def test_anthropic_request_shape():
    store = store_with(3)
    raw = [SimpleNamespace(type="thinking", thinking="", signature="sig")]   # replayed untouched
    req = Request("SYSTEM", TOOLS, history(store, "anthropic", raw), store, effort="high",
                  context=ContextPolicy(keep_image_turns=2))
    params = AnthropicProvider("claude-opus-5-5", client=object()).build_params(req)
    assert params["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in params["betas"]
    assert params["output_config"] == {"effort": "high"} and params["cache_control"] == {"type": "ephemeral"}
    edit = params["context_management"]["edits"][0]
    assert edit["type"] == "clear_tool_uses_20250919" and edit["keep"] == {"type": "tool_uses", "value": 2}
    msgs = params["messages"]
    assert msgs[1] == {"role": "assistant", "content": raw}
    result = msgs[2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "c1"
    image = [b for b in result["content"] if b["type"] == "image"][0]
    assert base64.b64decode(image["source"]["data"]) == b"png-2"
    assert msgs[3]["content"][0]["text"] == "Operator says: Use the left one."
    assert msgs[5]["content"][0]["is_error"] is True


def test_anthropic_rendering_is_byte_stable_across_requests():
    store = store_with(3)
    msgs = history(store, "anthropic", raw=[{"type": "text", "text": "x"}])
    p = AnthropicProvider("claude-opus-5-5", client=object())
    a = p.render(Request("S", TOOLS, msgs[:3], store))
    b = p.render(Request("S", TOOLS, msgs, store))
    assert json.dumps(a, default=str) == json.dumps(b[:3], default=str)   # append-only: the prefix never changes


def test_anthropic_no_fallback_for_other_models():
    store = store_with(1)
    params = AnthropicProvider("claude-haiku-5-5", client=object()).build_params(
        Request("S", TOOLS, [UserMessage((Text("hi"),))], store))
    assert "fallbacks" not in params and "betas" not in params


def test_anthropic_reply_parsing():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="The mug is left of the block.", signature="s"),
                 SimpleNamespace(type="text", text="Moving."),
                 SimpleNamespace(type="tool_use", id="t1", name="look", input={})],
        stop_reason="tool_use", model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=7,
                              cache_creation_input_tokens=1))
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response)))
    reply = AnthropicProvider("claude-opus-5-5", client=client).complete(
        Request("S", TOOLS, [UserMessage((Text("hi"),))], store_with(1)))
    assert reply.text == "Moving." and reply.tool_calls[0].name == "look"
    assert reply.raw is response.content and reply.usage.cache_read_tokens == 7
    assert reply.reasoning == "The mug is left of the block."


def test_openai_puts_images_after_tool_messages_and_trims_old_ones():
    store = store_with(3)
    req = Request("SYSTEM", TOOLS, history(store, "openai"), store, context=ContextPolicy(keep_image_turns=1))
    msgs = OpenAICompatProvider("qwen", client=object()).render(req)
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "tool", "user", "user", "assistant", "tool", "user"]
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": "Fresh."}
    old_pictures, new_pictures = msgs[4]["content"], msgs[8]["content"]
    assert not any(p["type"] == "image_url" for p in old_pictures)          # trimmed by policy
    assert any("older picture removed" in p.get("text", "") for p in old_pictures)
    assert any(p["type"] == "image_url" for p in new_pictures)
    assert msgs[7]["content"].startswith("[error]")
    assert msgs[2]["tool_calls"][0]["function"] == {"name": "look", "arguments": "{}"}


def test_openai_unparseable_arguments_become_a_validation_error():
    message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
        id="t1", function=SimpleNamespace(name="look", arguments="{not json"))])
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                               usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, prompt_tokens_details=None))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    reply = OpenAICompatProvider("qwen", client=client).complete(
        Request("S", TOOLS, [UserMessage((Text("hi"),))], store_with(1)))
    assert reply.stop_reason == "tool_use"
    assert reply.tool_calls[0].args == {"_unparseable_arguments": "{not json"}


def test_openai_reasoning_is_recorded_and_claude_thinking_blocks_are_replayed():
    blocks = [{"type": "thinking", "thinking": "Above the mug first.", "signature": "sig"}]
    message = SimpleNamespace(content="Moving.", reasoning_content="Above the mug first.", thinking_blocks=blocks,
                              tool_calls=[SimpleNamespace(id="t1", function=SimpleNamespace(name="look", arguments="{}"))])
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                               usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, prompt_tokens_details=None))
    sent = {}
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: sent.update(kw) or response)))
    store = store_with(1)
    p = OpenAICompatProvider("claude-opus-5-5", client=client, send_reasoning_effort=True, reasoning_summary=True)
    reply = p.complete(Request("S", TOOLS, [UserMessage((Text("hi"),))], store, effort="high"))
    assert sent["reasoning_effort"] == "high" and sent["extra_body"] == {"reasoning": {"summary": "auto"}}
    assert reply.reasoning == "Above the mug first." and reply.raw == {"thinking_blocks": blocks}
    msgs = p.render(Request("S", TOOLS, [UserMessage((Text("hi"),)),
                                         AssistantMessage(reply.text, reply.tool_calls, reply.raw, "openai")], store))
    assert msgs[2]["thinking_blocks"] == blocks
    plain = OpenAICompatProvider("qwen", client=object()).build_params(Request("S", TOOLS, [], store))
    assert "reasoning_effort" not in plain and "extra_body" not in plain   # off unless asked for


def test_openai_drops_the_reasoning_summary_when_the_server_rejects_it():
    message = SimpleNamespace(content="ok", tool_calls=[], reasoning_content="Thinking.")
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None))
    calls = []

    def create(**kw):
        calls.append(kw)
        if "extra_body" in kw:
            raise RuntimeError("400 reasoning: Extra inputs are not permitted")
        return response

    p = OpenAICompatProvider("claude-sonnet-5-5", client=SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create))), send_reasoning_effort=True, reasoning_summary=True)
    req = Request("S", TOOLS, [UserMessage((Text("hi"),))], store_with(1))
    assert p.complete(req).reasoning == "Thinking."
    p.complete(req)
    assert ["extra_body" in c for c in calls] == [True, False, False]   # asked once, then never again

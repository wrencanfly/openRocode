"""The tool contract and argument validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ..body.base import Body, CancelToken, Receipt
from .messages import Content, Text

if TYPE_CHECKING:
    from .observations import ObservationStore

ToolKind = Literal["observe", "act", "control"]


@dataclass
class ToolResult:
    content: tuple[Content, ...]
    is_error: bool = False
    receipt: Receipt | None = None   # act tools only; for the harness and recordings, not the model
    data: dict = field(default_factory=dict)  # structured, model-safe; returned to code-as-policy programs
    details: dict = field(default_factory=dict)  # recordings only
    ends_episode: bool = False

    @staticmethod
    def error(message: str) -> "ToolResult":
        return ToolResult((Text(message),), is_error=True)

    @property
    def stopped_early(self) -> bool:
        return self.receipt is not None and self.receipt.stop_reason != "reached"


@dataclass
class ToolContext:
    body: Body
    observations: "ObservationStore"
    cancel: CancelToken
    call_id: str
    # Lets a tool run other tools through the same hooks and events (used by code-as-policy).
    run_tool: Any = None


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict
    kind: ToolKind

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult: ...


def spec(tool: Tool) -> dict:
    return {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}


def validate(schema: dict, value: Any, path: str = "args") -> str | None:
    """Check `value` against the JSON Schema subset our tools use. Returns an error message or None.

    Arguments are never silently repaired: a model that sends bad arguments gets told exactly why.
    """
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return f"{path} must be an object"
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                return f"{path}.{key} is required"
        for key, v in value.items():
            if key not in props:
                if schema.get("additionalProperties") is False:
                    return f"{path}.{key} is not a known argument (known: {', '.join(props) or 'none'})"
                continue
            err = validate(props[key], v, f"{path}.{key}")
            if err:
                return err
        return None
    if t == "integer":
        if isinstance(value, bool) or not (isinstance(value, int) or (isinstance(value, float) and value.is_integer())):
            return f"{path} must be an integer"
    elif t == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{path} must be a number"
    elif t == "string":
        if not isinstance(value, str):
            return f"{path} must be a string"
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return f"{path} is longer than {schema['maxLength']} characters"
    elif t == "boolean":
        if not isinstance(value, bool):
            return f"{path} must be true or false"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of {schema['enum']}"
    if t in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path} must be at least {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path} must be at most {schema['maximum']}"
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            return f"{path} must be greater than {schema['exclusiveMinimum']}"
    return None

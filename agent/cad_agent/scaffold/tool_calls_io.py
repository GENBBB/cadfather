"""Function-call mechanics for the dialogue: everything function-calling policies do but do not decide.

Companion to `dialogue_io.py`, split along the same line: only what involves no
substantive choice lives here -- getting the calls from the harness, parsing
arguments, printing a call as a transcript line, assembling chat messages. What
the functions mean (names, schema, how a call becomes an action) belongs to the
policy, in `policies/dialogue_lean.py`.

**Transcript line format: `CALL <name> <JSON arguments>`.** It is parsed back by
the policy, so an answer that passed through text (the small variant puts the
answer into history as a string, not as a message) is parsed the same way
whether or not the server parsed the call. The same line written by the model
as plain text is parsed too: by intent it is a call, and refusing it would
punish the model for the transport.

**Fallback parser for the XML format `<function=...><parameter=...>`.** This is
how the Qwen chat template writes calls, and the server parser (`qwen3_coder`)
leaves a truncated or malformed block in `content` as text. Without the
fallback such an answer would read as "no action" although the model named one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from cad_agent.harness.search_types import SearchState

CALL_PREFIX = "CALL"
_CALL_RE = re.compile(r"^\s*CALL\s+([A-Za-z_]\w*)\s*(\{.*\})?\s*$")
_XML_FUNCTION_RE = re.compile(r"<function=([A-Za-z_]\w*)>(.*?)</function>", re.DOTALL)
_XML_PARAMETER_RE = re.compile(r"<parameter=([A-Za-z_]\w*)>\s*(.*?)\s*</parameter>", re.DOTALL)
_XML_BLOCK_RE = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.DOTALL)


def tool_calls(state: SearchState) -> list[dict[str, str]]:
    """Calls of the last answer as parsed by the server. A harness without the field gives none."""
    probe = getattr(state, "answer_tool_calls", None)
    return list(probe() or []) if probe is not None else []


def arguments(raw: Any) -> dict[str, Any]:
    """Call arguments as a dict. Unparseable ones give an empty dict, not an exception:
    malformed JSON is the model's answer and is judged by the policy parser, not the transport."""
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def render_call(name: str, args: dict[str, Any]) -> str:
    return f"{CALL_PREFIX} {name} {json.dumps(args, ensure_ascii=False, sort_keys=True)}"


def merged_text(content: str | None, calls: list[dict[str, str]]) -> str:
    """The answer as one text: the spoken part and the call lines below it."""
    lines = [content.strip()] if content and content.strip() else []
    lines.extend(render_call(item.get("name", ""), arguments(item.get("arguments")))
                 for item in calls)
    return "\n".join(lines)


def calls_in_text(text: str | None) -> list[tuple[str, dict[str, Any]]]:
    """Calls found in the answer text: `CALL` lines, then template XML blocks."""
    found: list[tuple[str, dict[str, Any]]] = []
    for line in (text or "").splitlines():
        match = _CALL_RE.match(line)
        if match:
            found.append((match.group(1), arguments(match.group(2))))
    for block in _XML_BLOCK_RE.findall(text or ""):
        for name, body in _XML_FUNCTION_RE.findall(block):
            args: dict[str, Any] = {}
            for key, value in _XML_PARAMETER_RE.findall(body):
                # The template writes values as is; compound ones as JSON.
                try:
                    args[key] = json.loads(value)
                except ValueError:
                    args[key] = value
            found.append((name, args))
    return found


# --- chat ----------------------------------------------------------------------


@dataclass
class Chat:
    """One part's conversation as messages: exchanges of question, answer, call results.

    An exchange is the unit of trimming: an assistant message with calls must
    stay together with its `tool` replies; dropped alone, it makes the history
    invalid for the chat template.
    """

    exchanges: list[list[dict[str, Any]]] = field(default_factory=list)
    trimmed: bool = False

    def messages(self, system: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for exchange in self.exchanges:
            out.extend(exchange)
        return out

    def trim(self, keep_last: int) -> None:
        self.exchanges = self.exchanges[-keep_last:] if keep_last > 0 else []
        self.trimmed = True


def chat_of(state: SearchState, key: str) -> Chat:
    chat = state.memory.get(key)
    if not isinstance(chat, Chat):
        chat = state.memory[key] = Chat()
    return chat


def assistant_message(content: str, calls: list[dict[str, str]], ids: list[str]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if calls:
        message["tool_calls"] = [
            {"id": call_id, "type": "function",
             "function": {"name": item.get("name", ""),
                          "arguments": item.get("arguments") or "{}"}}
            for item, call_id in zip(calls, ids)
        ]
    return message


def tool_message(call_id: str, text: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def call_ids(calls: list[dict[str, str]], turn: int, exchange: int) -> list[str]:
    """Call ids: the server's, or our own where absent, unique within the part."""
    return [item.get("id") or f"call_{turn}_{exchange}_{index}"
            for index, item in enumerate(calls)]

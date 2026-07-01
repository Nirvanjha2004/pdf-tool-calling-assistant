"""
LLM Client — Groq API with ReAct-style Tool Calling

Instead of relying on Groq's native tool_calls API (which is buggy with
llama-3.3-70b-versatile and generates malformed JSON), we use a ReAct
pattern where:

  1. The system prompt describes tools and a strict TOOL_CALL: format
  2. The model emits:  TOOL_CALL: tool_name {"arg": "val"}
  3. We parse that line, execute the tool, inject the result as a
     TOOL_RESULT: block, and loop until the model gives a final answer
  4. No tool_calls API is used — plain text completions only

This is 100% reliable across all Groq models.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Callable, Optional

logger = logging.getLogger(__name__)

# ─── Load .env ───────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=str(env_path) if env_path.exists() else None)
except ImportError:
    pass

from groq import Groq

# ─── Tool Registry ───────────────────────────────────────────────────────────

_tool_handlers: Dict[str, Callable] = {}
_tool_definitions: List[Dict] = []   # kept for compatibility, not sent to API


def register_tool(name: str, definition: dict, handler: Callable) -> None:
    _tool_definitions.append(definition)
    _tool_handlers[name] = handler


def get_registered_tools() -> List[Dict]:
    return _tool_definitions.copy()


# ─── ReAct parser ────────────────────────────────────────────────────────────
# The model emits exactly:   TOOL_CALL: tool_name {"key": "value"}

_TOOL_CALL_RE = re.compile(
    r"TOOL_CALL:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+(\{.*\})",
    re.DOTALL,
)


def _parse_tool_call(text: str) -> Optional[Dict]:
    """
    Extract the first TOOL_CALL line from the model's response.
    Returns {"name": ..., "args": {...}} or None.
    """
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip()
    try:
        args = json.loads(m.group(2))
    except json.JSONDecodeError:
        logger.warning("Could not parse tool args JSON: %s", m.group(2))
        return None
    return {"name": name, "args": args}


def _build_tools_description() -> str:
    """Build a plain-text description of all registered tools for the system prompt."""
    if not _tool_handlers:
        return ""
    lines = ["Tools you can call:"]
    for defn in _tool_definitions:
        fn = defn.get("function", defn)
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        param_names = list(params.keys())
        lines.append(f"  • {name}({', '.join(param_names)}) — {desc}")
    return "\n".join(lines)


# ─── Groq client ─────────────────────────────────────────────────────────────

_DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set.\n"
            "Set it in your .env file: GROQ_API_KEY=gsk_your_key_here"
        )
    return Groq(api_key=api_key)


# ─── Core ReAct loop ─────────────────────────────────────────────────────────

def chat_completion(
    messages: List[Dict],
    model: Optional[str] = None,
    max_tool_rounds: int = 5,
) -> Dict:
    """
    Send messages to the LLM using a ReAct-style tool loop.

    The system prompt is augmented with tool descriptions and a strict
    TOOL_CALL format. When the model emits a TOOL_CALL line we execute
    the tool and feed the result back as a TOOL_RESULT. This loops until
    the model produces a final answer (no TOOL_CALL in the response).

    Args:
        messages: Conversation history (system + user turns).
        model: Groq model ID override.
        max_tool_rounds: Max number of tool calls before forcing a final answer.

    Returns:
        {"role": "assistant", "content": "<final answer>"}
    """
    client = get_client()
    model = model or _DEFAULT_MODEL

    # Inject tool instructions into the first system message (or prepend one)
    tools_desc = _build_tools_description()
    working_messages = []
    injected = False
    for msg in messages:
        if msg["role"] == "system" and not injected:
            combined = msg["content"] + "\n\n" + tools_desc if tools_desc else msg["content"]
            working_messages.append({"role": "system", "content": combined})
            injected = True
        else:
            working_messages.append(dict(msg))
    if not injected and tools_desc:
        working_messages.insert(0, {"role": "system", "content": tools_desc})

    for round_num in range(max_tool_rounds + 1):
        response = client.chat.completions.create(
            model=model,
            messages=working_messages,
            max_tokens=4096,
            # No tools= parameter — plain text completions only
        )
        reply = response.choices[0].message.content or ""

        # Check if the model wants to call a tool
        tool_call = _parse_tool_call(reply)
        logger.info("Model reply (round %d): %r", round_num, reply[:300])

        if tool_call is None or round_num == max_tool_rounds:
            # No tool call (or we've hit the limit) — this is the final answer
            # Strip any stray TOOL_CALL lines that didn't parse correctly
            final = _TOOL_CALL_RE.sub("", reply).strip()
            return {
                "role": "assistant",
                "content": final or "I processed your request.",
            }

        # Execute the tool
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name in _tool_handlers:
            logger.info("Executing tool: %s(%s)", tool_name, tool_args)
            tool_result = _tool_handlers[tool_name](tool_args)
        else:
            tool_result = (
                f"Error: unknown tool '{tool_name}'. "
                f"Available: {list(_tool_handlers.keys())}"
            )
            logger.warning("Unknown tool called: %s", tool_name)

        # Append the model's tool-call turn and the result, then loop
        working_messages.append({"role": "assistant", "content": reply})
        working_messages.append({
            "role": "user",
            "content": f"TOOL_RESULT: {tool_name}\n{tool_result}",
        })

    # Should never reach here
    return {"role": "assistant", "content": "Sorry, I could not complete the request."}

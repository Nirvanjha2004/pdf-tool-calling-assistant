"""
LLM Client — Groq API Wrapper with Tool-Calling Support

This is the heart of the project. It:
1. Sends messages + tool definitions to Groq
2. Receives structured tool call requests from the LLM
3. Executes the corresponding Python functions
4. Feeds results back to the LLM for the final answer

This implements the "Local Tool Calling" pattern:
    LLM → "call this function with these args" (JSON)
    Your code → executes the function → returns result
    LLM → gives final answer using the result
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Callable, Optional

# Load .env from project root (explicit path to avoid CWD issues)
try:
    from dotenv import load_dotenv

    # Look for .env in the project root (parent of the core/ directory)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
    else:
        # Fallback: search CWD
        load_dotenv()
except ImportError:
    pass

from groq import Groq


# ─── Old-Style Function Call Fallback Parser ────────────────────────────────
# Some models (especially smaller ones) sometimes output the old XML-like
# function calling format: `<function=name>{"arg": "val"}</function>`
# instead of the modern JSON tool_calls API.
# This regex + handler parses and executes that format as a fallback.

_OLD_FUNC_RE = re.compile(r"<function=([a-zA-Z_][a-zA-Z0-9_]*)>(\{.*?\}|\[.*?\])</function>")


def _parse_old_style_tool_calls(content: str) -> List[Dict]:
    """
    Parse old-style `<function=name>{args}</function>` calls from text.

    Returns a list of dicts with 'name' and 'arguments' keys,
    or empty list if no old-style calls found.
    """
    matches = _OLD_FUNC_RE.findall(content)
    results = []
    for func_name, args_json in matches:
        try:
            # Validate args are valid JSON
            json.loads(args_json)
            results.append({"name": func_name, "arguments": args_json})
        except json.JSONDecodeError:
            pass  # Invalid JSON in old-style call, skip
    return results


# ─── Tool Registry ────────────────────────────────────────────────────────────
# Tools register themselves here so the chat loop can find them.

_tool_handlers: Dict[str, Callable] = {}
_tool_definitions: List[Dict] = []


# ─── Tool Registry ────────────────────────────────────────────────────────────
# Tools register themselves here so the chat loop can find them.

_tool_handlers: Dict[str, Callable] = {}
_tool_definitions: List[Dict] = []


def register_tool(name: str, definition: dict, handler: Callable) -> None:
    """
    Register a tool with the LLM client.

    Args:
        name: Tool name (must match the 'name' in the definition).
        definition: JSON Schema tool definition dict.
        handler: Function that takes (arguments: dict) and returns str.
    """
    _tool_definitions.append(definition)
    _tool_handlers[name] = handler


def get_registered_tools() -> List[Dict]:
    """Return the list of registered tool definitions."""
    return _tool_definitions.copy()


# ─── Groq Client ─────────────────────────────────────────────────────────────

_DEFAULT_MODEL = "llama-3.1-8b-instant"


def get_client() -> Groq:
    """
    Get or create the Groq client.

    Reads the API key from the GROQ_API_KEY environment variable.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set.\n\n"
            "1. Get your API key: https://console.groq.com/keys\n"
            "2. Set it:\n"
            "   # Windows (Command Prompt):\n"
            "   set GROQ_API_KEY=gsk_your_key_here\n\n"
            "   # Windows (PowerShell):\n"
            "   $env:GROQ_API_KEY='gsk_your_key_here'\n\n"
            "   # macOS/Linux:\n"
            "   export GROQ_API_KEY='gsk_your_key_here'\n\n"
            "   Or create a .env file with:\n"
            "   GROQ_API_KEY=gsk_your_key_here"
        )
    return Groq(api_key=api_key)


# ─── Core Chat Completion with Tool Handling ─────────────────────────────────


def chat_completion(
    messages: List[Dict],
    model: Optional[str] = None,
    max_tool_rounds: int = 5,
) -> Dict:
    """
    Send messages to the LLM and automatically handle tool calls.

    The LLM may respond with:
    - A text response (no tools needed)
    - One or more tool_calls (the LLM wants us to execute functions)

    If tool_calls are received, this function:
    1. Appends the assistant's tool_call message to history
    2. Executes each tool
    3. Appends tool results as "tool" role messages
    4. Sends everything back to the LLM for the final answer

    This loop repeats up to max_tool_rounds times.

    Args:
        messages: Conversation history (system, user, assistant, tool).
        model: Groq model ID. Defaults to a capable tool-use model.
        max_tool_rounds: Maximum number of tool-calling iterations.

    Returns:
        The final assistant response dict with 'role' and 'content'.
    """
    client = get_client()
    model = model or _DEFAULT_MODEL

    # Only pass tools if any are registered
    tools = _tool_definitions if _tool_definitions else None
    tool_choice = "auto" if tools else None

    # First API call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
    )

    message = response.choices[0].message

    # Tool-calling loop
    rounds = 0
    while rounds < max_tool_rounds:
        rounds += 1

        # ── Path A: Modern tool_calls API ──
        if message.tool_calls:
            assistant_msg = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
            messages.append(assistant_msg)

            # Execute each tool call
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                if tool_name in _tool_handlers:
                    result = _tool_handlers[tool_name](tool_args)
                else:
                    result = f"Error: Unknown tool '{tool_name}'. Available tools: {list(_tool_handlers.keys())}"

                messages.append(
                    {
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call.id,
                    }
                )

        # ── Path B: Fallback for old-style <function=name>{...}</function> ──
        elif message.content and _OLD_FUNC_RE.search(message.content):
            old_calls = _parse_old_style_tool_calls(message.content)
            if old_calls:
                # Clean the function call markup from the content
                clean_content = _OLD_FUNC_RE.sub("", message.content).strip()

                # Create properly matched tool_calls in assistant message
                # so the tool_call_id in subsequent tool messages are valid
                assistant_tool_calls = [
                    {
                        "id": f"call_fallback_{fc['name']}_{rounds}_{i}",
                        "type": "function",
                        "function": {
                            "name": fc["name"],
                            "arguments": fc["arguments"],
                        },
                    }
                    for i, fc in enumerate(old_calls)
                ]

                assistant_msg = {
                    "role": "assistant",
                    "content": clean_content or "",
                    "tool_calls": assistant_tool_calls,
                }
                messages.append(assistant_msg)

                for i, fc in enumerate(old_calls):
                    tool_name = fc["name"]
                    tool_args = json.loads(fc["arguments"])

                    if tool_name in _tool_handlers:
                        result = _tool_handlers[tool_name](tool_args)
                    else:
                        result = f"Error: Unknown tool '{tool_name}'. Available tools: {list(_tool_handlers.keys())}"

                    tool_call_id = f"call_fallback_{fc['name']}_{rounds}_{i}"
                    messages.append(
                        {
                            "role": "tool",
                            "content": str(result),
                            "tool_call_id": tool_call_id,
                        }
                    )

        # ── No tool calls ──
        else:
            break

        # Send everything back to the LLM for the final answer
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )

        message = response.choices[0].message

    # Clean any leftover old-style function tags from the final answer
    final_content = message.content or ""
    final_content = _OLD_FUNC_RE.sub("", final_content).strip()

    return {
        "role": "assistant",
        "content": final_content or "I processed your request using the available tools.",
    }

"""
Chat Loop — Main Orchestration Layer

Registers both tools (calculator + document search) with the LLM client,
provides the system prompt, and exposes a clean API for asking questions.

Two interfaces:
    1. ask()             — programmatic: send a question, get an answer
    2. run_interactive()  — interactive terminal chat
"""

import logging
import os
import time
from typing import List, Dict, Optional

from core.llm_client import chat_completion, register_tool
from tools.calculator import TOOL_DEFINITION as CALC_TOOL, handle_tool_call as calc_handler
from tools.pdf_search import (
    TOOL_DEFINITION as SEARCH_TOOL,
    handle_tool_call as search_handler,
    load_document,
    is_document_loaded,
    get_chunk_count,
)

logger = logging.getLogger(__name__)

# ─── Tunables (env-overridable) ──────────────────────────────────────────────

MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "4000"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
RETRY_BACKOFF_SECONDS = float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", "1.5"))

_VALID_ROLES = {"user", "assistant", "system"}

# ─── Register Tools on Import ────────────────────────────────────────────────
# Guarded so re-importing this module (e.g. in tests, or a hot-reloading
# dev server) doesn't attempt to double-register the same tool names.

_tools_registered = False


def _register_tools_once() -> None:
    global _tools_registered
    if _tools_registered:
        return
    try:
        register_tool("calculate", CALC_TOOL, calc_handler)
        register_tool("search_document", SEARCH_TOOL, search_handler)
        _tools_registered = True
    except Exception:
        logger.exception("Tool registration failed — chat loop cannot function without tools.")
        raise


_register_tools_once()

# ─── System Prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful PDF Q&A assistant with access to exactly two tools:

1. calculate       — for any math computation.
2. search_document — for any question about the content of the loaded PDF.

═══ TOOL CALL PROTOCOL — FOLLOW EXACTLY ═══

To call a tool, your entire response must be ONLY this, on one line:
TOOL_CALL: tool_name {"key": "value"}

Hard rules for this line:
- No text before it. No text after it. No markdown, no code fences, no explanation.
- tool_name must be exactly "calculate" or "search_document" — nothing else.
- The JSON after the tool name must be valid, single-line JSON with double-quoted keys.
- Call at most ONE tool per turn. Never chain multiple TOOL_CALL lines in one response.
- After outputting the TOOL_CALL line, STOP immediately. Wait for TOOL_RESULT.
- After you receive a TOOL_RESULT, respond with your final answer in plain text —
  do NOT emit another TOOL_CALL in the same turn unless the result explicitly tells
  you more information is needed and names what to search for next.

If you are not calling a tool, never output the literal text "TOOL_CALL" anywhere
in your response, even as an example or in quotes.

═══ WHEN TO CALL WHICH TOOL ═══

- search_document: ANY question that references the uploaded document, its content,
  questions, tables, figures, or anything the user implies is "in the PDF."
- calculate: ANY arithmetic, algebra, or numeric computation — even a single
  multiplication — rather than computing it yourself. Do not do mental math.
- Neither tool: greetings, clarifying questions, general knowledge unrelated to the
  document, or requests to reformat/summarize something already given to you in this
  conversation (no need to re-search for content you already have in context).

═══ HANDLING THE DOCUMENT STATE ═══

- If no document is loaded and the user asks about "the PDF" or "the document,"
  do NOT call search_document — it will fail. Instead tell the user no document is
  loaded yet and ask them to provide one.
- If search_document returns no relevant results or an empty/error result, say so
  plainly ("I couldn't find that in the document") — do not guess or fabricate an
  answer to fill the gap.

═══ ANSWER FORMATTING ═══

- Format math with LaTeX: inline \\(expr\\), display \\[expr\\].
- For multiple-choice math, one option per line, e.g.:
  (A) \\(\\frac{\\pi}{6}\\)
  (B) \\(\\frac{\\pi}{3}\\)
- If an answer option (A)/(B)/(C)/(D) is blank, garbled, or unreadable in the source
  text, write exactly: (A) *(not readable in source)* — never invent plausible-looking
  content to fill a gap, and never silently drop the option.
- Never claim something is in the document unless it actually appeared in a
  TOOL_RESULT you received this conversation.

═══ EXAMPLE ═══

User: give me questions from the pdf
Assistant: TOOL_CALL: search_document {"query": "questions"}
User: TOOL_RESULT: search_document
[chunk content with questions]
Assistant: Here are the questions from the document: ..."""


# ─── Internal helpers ────────────────────────────────────────────────────────


def _sanitize_history(history: Optional[List[Dict]]) -> List[Dict]:
    """
    Defensively clean caller-supplied history instead of trusting it blindly:
    - drops entries that aren't well-formed {"role", "content"} dicts
    - drops entries with an invalid role or empty/non-string content
    - caps to the most recent MAX_HISTORY_MESSAGES entries
    Never mutates the caller's original list.
    """
    if not history:
        return []

    clean: List[Dict] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if role not in _VALID_ROLES:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        clean.append({"role": role, "content": content})

    if len(clean) > MAX_HISTORY_MESSAGES:
        clean = clean[-MAX_HISTORY_MESSAGES:]

    return clean


def _call_with_retries(messages: List[Dict], model: Optional[str]) -> Dict:
    """
    Call chat_completion with a small retry/backoff loop for transient failures
    (network blips, rate limits, etc). Configuration errors (ValueError, e.g. a
    missing API key) are never retried — retrying won't fix a missing key.
    """
    last_error: Optional[Exception] = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            return chat_completion(messages, model=model)
        except ValueError:
            raise  # config errors are not transient — fail fast
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
                logger.warning(
                    "chat_completion failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, MAX_RETRIES + 1, e, wait,
                )
                time.sleep(wait)
            else:
                logger.error("chat_completion failed after %d attempts: %s", attempt + 1, e)

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES + 1} attempts") from last_error


# ─── Programmatic API ────────────────────────────────────────────────────────


def ask(question: str, history: Optional[List[Dict]] = None, model: Optional[str] = None) -> Dict:
    """
    Ask a question with automatic tool handling.

    Args:
        question: The user's question.
        history: Optional conversation history (list of message dicts). Not mutated.
        model: Optional Groq model override.

    Returns:
        Assistant response dict with 'role' and 'content'.

    Raises:
        ValueError: if the question is empty, or for LLM configuration errors
            (e.g. missing API key) — these are not retried.
        RuntimeError: if the LLM call fails after retries for transient reasons.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    question = question.strip()
    if len(question) > MAX_QUESTION_CHARS:
        logger.warning(
            "question truncated from %d to %d chars", len(question), MAX_QUESTION_CHARS
        )
        question = question[:MAX_QUESTION_CHARS]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        doc_loaded = is_document_loaded()
    except Exception:
        logger.exception("is_document_loaded() failed — assuming no document loaded.")
        doc_loaded = False

    if doc_loaded:
        try:
            chunk_count = get_chunk_count()
        except Exception:
            logger.exception("get_chunk_count() failed.")
            chunk_count = "an unknown number of"
        messages.append({
            "role": "system",
            "content": (
                f"A PDF document is already loaded in memory with {chunk_count} chunk(s). "
                "For any question about the uploaded document, use search_document first."
            ),
        })
    else:
        messages.append({
            "role": "system",
            "content": (
                "No PDF document is currently loaded. If the user asks about a document "
                "or PDF, tell them none is loaded and ask them to provide one instead of "
                "calling search_document."
            ),
        })

    messages.extend(_sanitize_history(history))
    messages.append({"role": "user", "content": question})

    result = _call_with_retries(messages, model)

    if not isinstance(result, dict) or "content" not in result:
        logger.error("Unexpected chat_completion result shape: %r", result)
        return {"role": "assistant", "content": "Sorry, something went wrong generating a response."}

    return result


# ─── Interactive Terminal Chat ───────────────────────────────────────────────


def run_interactive(document_path: Optional[str] = None) -> None:
    """
    Run an interactive Q&A session in the terminal.

    Args:
        document_path: Optional path to a PDF to load at startup.
    """
    print()
    print("=" * 60)
    print("  📄 PDF Q&A Assistant with Tool-Calling")
    print("=" * 60)
    print("  Type your questions — the LLM decides which tool to use.")
    print()

    if document_path:
        if not os.path.exists(document_path):
            print(f"  ❌ File not found: {document_path}")
        else:
            try:
                msg = load_document(document_path)
                print(f"  {msg}")
            except Exception as e:
                print(f"  ❌ Error loading document: {e}")

    print()
    print("  Commands:")
    print("    /load <path>   Load a PDF document")
    print("    /status        Show document load status")
    print("    /clear         Clear conversation history")
    print("    /help          Show this help")
    print("    /quit          Exit")
    print("=" * 60)

    history: List[Dict] = []

    while True:
        try:
            question = input("\n💬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Bye!")
            break

        if not question:
            continue

        cmd = question.lower()

        if cmd == "/quit":
            print("\n👋 Bye!")
            break

        if cmd == "/help":
            print("\n  Commands:")
            print("    /load <path>   Load a PDF document")
            print("    /status        Show document load status")
            print("    /clear         Clear conversation history")
            print("    /help          Show this help")
            print("    /quit          Exit")
            continue

        if cmd == "/status":
            try:
                loaded = is_document_loaded()
            except Exception as e:
                print(f"\n  ❌ Could not check status: {e}")
                continue
            if loaded:
                try:
                    print(f"\n  ✅ Document loaded ({get_chunk_count()} chunks).")
                except Exception:
                    print("\n  ✅ Document is loaded and ready for questions.")
            else:
                print("\n  ❌ No document loaded. Use /load <path> to load one.")
            continue

        if cmd == "/clear":
            history = []
            print("\n  🧹 Conversation history cleared.")
            continue

        if cmd.startswith("/load "):
            path = question[6:].strip()
            if not path:
                print("\n  ❌ Usage: /load <path>")
                continue
            if not os.path.exists(path):
                print(f"\n  ❌ File not found: {path}")
                continue
            try:
                msg = load_document(path)
                print(f"\n  {msg}")
            except Exception as e:
                print(f"\n  ❌ Error: {e}")
            continue

        # Ask the question
        print("\n🤖 Assistant: ", end="", flush=True)

        try:
            result = ask(question, history)
            answer = result.get("content", "") if isinstance(result, dict) else str(result)
            print(answer)

            # Update conversation history (ask() also caps this, but keep the
            # in-memory copy bounded too so it doesn't grow unbounded across
            # a very long interactive session).
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            if len(history) > MAX_HISTORY_MESSAGES:
                history = history[-MAX_HISTORY_MESSAGES:]

        except ValueError as e:
            print(f"\n  ❌ Configuration Error: {e}")
            print("  💡 Set your GROQ_API_KEY environment variable and try again.")
        except RuntimeError as e:
            print(f"\n  ❌ Request failed after retries: {e}")
        except Exception as e:
            logger.exception("Unexpected error handling question.")
            print(f"\n  ❌ Unexpected error: {e}")


# ─── Run Directly ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    doc_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_interactive(doc_path)
"""
Chat Loop — Main Orchestration Layer

Registers both tools (calculator + document search) with the LLM client,
provides the system prompt, and exposes a clean API for asking questions.

Two interfaces:
    1. ask()           — programmatic: send a question, get an answer
    2. run_interactive() — interactive terminal chat
"""

import os
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

# ─── Register Tools on Import ────────────────────────────────────────────────
# This runs once when the module is first imported.

register_tool("calculate", CALC_TOOL, calc_handler)
register_tool("search_document", SEARCH_TOOL, search_handler)

# ─── System Prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful PDF Q&A assistant with access to tools.

## Available Tools

1. **calculate(expression)** — Evaluate math expressions.
   USE WHEN: The user asks anything involving numbers, calculations,
   percentages, statistics, or computations.

2. **search_document(query)** — Search within the uploaded PDF document.
   USE WHEN: The user asks about the content of their document
   (resume, research paper, job description, etc.).

## Your Behavior

- For math questions → ALWAYS use the calculator tool.
- For document questions → ALWAYS search the document first, then summarize or answer based on the returned chunks.
- If the user asks something general like "tell me about the PDF" or "summarize the document", search with a broad query like "main topic content" and summarize what you find.
- NEVER say you couldn't find information if the search tool returned chunk content — use that content to answer.
- For general knowledge → Answer from your own knowledge.
- Be concise, clear, and helpful."""


# ─── Programmatic API ────────────────────────────────────────────────────────


def ask(question: str, history: Optional[List[Dict]] = None, model: Optional[str] = None) -> Dict:
    """
    Ask a question with automatic tool handling.

    Args:
        question: The user's question.
        history: Optional conversation history (list of message dicts).
        model: Optional Groq model override.

    Returns:
        Assistant response dict with 'role' and 'content'.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if is_document_loaded():
        messages.append(
            {
                "role": "system",
                "content": (
                    f"A PDF document is already loaded in memory with {get_chunk_count()} chunk(s). "
                    "For any question about the uploaded document, use search_document first."
                ),
            }
        )

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": question})

    result = chat_completion(messages, model=model)
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
            return
        try:
            msg = load_document(document_path)
            print(f"  {msg}")
        except Exception as e:
            print(f"  ❌ Error loading document: {e}")
            return

    print()
    print("  Commands:")
    print("    /load <path>   Load a PDF document")
    print("    /status        Show document load status")
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

        if question.lower() == "/quit":
            print("\n👋 Bye!")
            break

        if question.lower() == "/help":
            print("\n  Commands:")
            print("    /load <path>   Load a PDF document")
            print("    /status        Show document load status")
            print("    /help          Show this help")
            print("    /quit          Exit")
            continue

        if question.lower() == "/status":
            if is_document_loaded():
                print("\n  ✅ Document is loaded and ready for questions.")
            else:
                print("\n  ❌ No document loaded. Use /load <path> to load one.")
            continue

        if question.lower().startswith("/load "):
            path = question[6:].strip()
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
            answer = result["content"]
            print(answer)

            # Update conversation history (keep last 10 exchanges)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            if len(history) > 20:
                history = history[-20:]

        except ValueError as e:
            print(f"\n  ❌ Configuration Error: {e}")
            print("  💡 Set your GROQ_API_KEY environment variable and try again.")
        except Exception as e:
            print(f"\n  ❌ Error: {e}")


# ─── Run Directly ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    doc_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_interactive(doc_path)

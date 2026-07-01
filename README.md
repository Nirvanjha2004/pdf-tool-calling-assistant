# 📄 PDF Q&A Assistant with Tool-Calling

A full-stack PDF question-answering assistant that demonstrates how LLM tool-calling works — built from scratch with no agent frameworks.

Upload any PDF (text, scanned, or math-heavy), ask natural-language questions, and watch the AI automatically decide which tool to use: search the document, run a calculation, or answer from its own knowledge.

---

## ✨ Features

- **Instant uploads** — file is accepted immediately; extraction runs in the background
- **3-tier PDF extraction** — text PDFs, scanned PDFs (OCR), and math PDFs with embedded equation images all handled automatically
- **LaTeX math rendering** — MathJax renders equations from the LLM's response in real time
- **ReAct-style tool loop** — reliable tool calling without Groq's native tool_calls API
- **Safe math evaluator** — AST-based, no `eval()`, supports `math.*` functions
- **Conversation history** — multi-turn chat with context carried across questions
- **Beautiful UI** — aurora glass design, drag & drop upload, animated progress

---

## 🧠 How Tool-Calling Works

Tool-calling is how LLMs interact with external functions. Instead of guessing an answer, the model decides to call a tool, your code runs it, and the result flows back.

```
User: "Give me questions from the PDF"

  LLM output:   TOOL_CALL: search_document {"query": "questions"}
  Your code:    runs search_document() → returns chunk text
  LLM output:   "Here are the questions from the document: ..."
```

This project uses a **ReAct-style pattern** — the model writes `TOOL_CALL:` as plain text, your code parses it, executes the function, injects the result as `TOOL_RESULT:`, and loops until the model gives a final answer. No JSON schema sent to the API, no native tool_calls — just reliable plain-text completions.

---

## 🏗️ Project Structure

```
pdf-tool-calling-assistant/
├── api/
│   └── main.py             # FastAPI server — /upload, /job/{id}, /ask, /health
├── core/
│   ├── llm_client.py       # Groq API wrapper + ReAct tool-calling loop
│   └── chat_loop.py        # Tool registration, system prompt, ask() API
├── tools/
│   ├── calculator.py       # Safe AST-based math evaluator
│   └── pdf_search.py       # Keyword search + async background PDF loading
├── utils/
│   └── pdf_parser.py       # 3-tier PDF text extraction pipeline
├── static/
│   └── index.html          # Single-file frontend (vanilla JS + MathJax)
├── data/
│   └── sample_pdfs/        # Sample PDFs for testing
├── .env.example
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Get a Groq API Key

- Go to [console.groq.com/keys](https://console.groq.com/keys)
- Create a free API key (starts with `gsk_`)

### 2. Set Up Environment

```bash
cd pdf-tool-calling-assistant

# Copy and fill in your key
copy .env.example .env
# Edit .env and set: GROQ_API_KEY=gsk_your_key_here
```

Or set it directly:

```bash
# Windows (PowerShell)
$env:GROQ_API_KEY='gsk_your_key_here'

# macOS / Linux
export GROQ_API_KEY='gsk_your_key_here'
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `rapidocr-onnxruntime` installs ONNX Runtime (~50MB). The OCR model weights (~5MB) are downloaded automatically on first use.

### 4. Run the Web App

```bash
python -m api.main
```

Open [http://localhost:8000](http://localhost:8000), upload a PDF, and start asking questions.

### 5. Or Use the CLI

```bash
# Without a document
python -m core.chat_loop

# With a PDF loaded at startup
python -m core.chat_loop path/to/your/document.pdf
```

CLI commands: `/load <path>`, `/status`, `/clear`, `/help`, `/quit`

---

## 🔧 Available Tools

### `search_document(query)`
Searches the uploaded PDF using keyword overlap scoring across word-level chunks (300 words, 30-word overlap). Returns the top 3 most relevant chunks, or all chunks for broad/summarization queries. Capped at 6000 chars to stay within LLM context limits.

**Used when:** the user asks about document content — questions, summaries, specific topics.

### `calculate(expression)`
Safely evaluates math expressions using Python's `ast` module. Supports arithmetic, exponentiation, modulo, `math.sqrt()`, `math.sin()`, `math.cos()`, `math.tan()`, `math.log()`, `math.pi`, `math.e`, and more.

**Used when:** the user asks anything involving numbers or computation.

---

## 📄 PDF Extraction Pipeline

Extraction runs in a background thread so uploads return instantly. The frontend polls `/job/{id}` every 600ms until done.

```
Tier 1: pypdfium2      →  fastest, handles most standard PDFs
           ↓ (< 50 chars extracted)
Tier 2: pdfplumber     →  better at complex layouts and tables
           ↓ (< 50 chars extracted)
Tier 3: RapidOCR       →  fully scanned / image-only PDFs

Special case: if Tier 1/2 succeeds but the PDF has low text density
AND blank answer options — it's a math PDF with embedded equation images
(e.g. Word + MathType export). OCR runs anyway and is merged with the
text output so both prose AND equations are captured.
              ↓
         pypdfium2+ocr
```

The extraction method is shown in the UI after upload:
- ⚡ fast extract — pypdfium2
- 📄 text extract — pdfplumber
- 🔍 OCR — RapidOCR
- ⚡+🔍 text & OCR — merged (equation-heavy PDFs)

**OCR performance:** RapidOCR uses PP-OCR mobile models via ONNXRuntime — ~3-5× faster than easyocr on CPU. Pages are rendered at 150 DPI, capped at 1600px wide, converted to grayscale, and processed concurrently across a thread pool.

The OCR engine pre-warms at server startup in a background thread so it's ready before the first image PDF arrives.

---

## 🌐 API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the web frontend |
| `GET` | `/health` | Server status, document loaded state, chunk count |
| `POST` | `/upload` | Upload a PDF. Returns `{job_id, message}` immediately |
| `GET` | `/job/{job_id}` | Poll upload status: `pending \| running \| done \| error` |
| `POST` | `/ask` | Ask a question. Body: `{question, history?}` → `{answer}` |

### Example: curl

```bash
# Upload
curl -X POST -F "file=@notes.pdf" http://localhost:8000/upload
# → {"job_id": "abc123", "message": "Upload received. Processing started."}

# Poll until done
curl http://localhost:8000/job/abc123
# → {"status": "done", "chunks_count": 8, "method": "pypdfium2", ...}

# Ask
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics?"}' \
  http://localhost:8000/ask
# → {"answer": "The document covers..."}
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `PORT` | `8000` | Server port |
| `OCR_DPI` | `150` | Render DPI for OCR pages |
| `OCR_MAX_WIDTH_PX` | `1600` | Max page width before downscaling |
| `OCR_WORKERS` | CPU count | Parallel OCR threads |
| `MAX_QUESTION_CHARS` | `4000` | Max question length |
| `MAX_HISTORY_MESSAGES` | `20` | Max conversation history entries |
| `LLM_MAX_RETRIES` | `2` | LLM call retry attempts |
| `LLM_RETRY_BACKOFF_SECONDS` | `1.5` | Backoff between retries |

---

## 💡 What You'll Learn

- ✅ How tool-calling / function calling works end-to-end
- ✅ Why ReAct-style prompting can be more reliable than native tool APIs
- ✅ How to build a multi-tier PDF extraction pipeline (text → OCR)
- ✅ How to handle async background jobs in FastAPI
- ✅ How to render LaTeX in a browser chat UI with MathJax
- ✅ Safe expression evaluation with Python's `ast` module
- ✅ Conversation history management and context injection

---

## 🔬 Architecture Deep Dive

### Why ReAct instead of Groq's native tool_calls?

`llama-3.3-70b-versatile` consistently generated malformed tool call JSON with Groq's native tool_calls API — specifically missing the `>` separator between function name and arguments, e.g. `<function=search_document {"query": ...}` — which Groq's API rejected with a 400 `tool_use_failed` error.

The ReAct approach bypasses this entirely: the model writes `TOOL_CALL: search_document {"query": "..."}` as plain text, a regex parses it, and the result is injected back as a user message. This is 100% reliable across all Groq models.

### Math PDF handling

Many math exam PDFs are created by exporting Word documents with MathType equations. The equations are embedded as image objects while the surrounding text (question stems, option labels) is selectable text. `pypdfium2` extracts the text but silently skips the image objects, leaving blank answer options.

Detection heuristic: if chars-per-page < 1200 **and** there are 2+ blank answer option lines `(A)\n(B)\n`, the PDF is classified as equation-heavy and OCR is run over every page. The OCR output is merged with the text extraction, capturing both the prose and the equations.

---

## 🚀 Next Steps

1. **Vector search** — replace keyword overlap with sentence embeddings + cosine similarity for semantic search
2. **Multi-document support** — index multiple PDFs and search across all of them
3. **Web search tool** — let the LLM fetch real-time information
4. **Code execution tool** — run Python in a sandbox for data analysis
5. **Persistent storage** — save uploaded documents and conversation history to a database
6. **Streaming responses** — stream the LLM response token-by-token for faster perceived latency

---

## 📝 License

MIT — use it however you want. Built for learning.

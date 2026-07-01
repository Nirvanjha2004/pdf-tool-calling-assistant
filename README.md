# 📄 PDF Q&A Assistant with Tool-Calling

**Learn how LLM tool-calling / function calling works — by building it yourself.**

No agent frameworks. No vector databases. Just pure tool-calling: you define functions, the LLM decides when to call them, your code executes them, and the result flows back to the LLM.

---

## 🧠 What is Tool-Calling?

Tool-calling (also called function calling) is how LLMs interact with the outside world.

### The Flow

```
You: "What's 15% of 2400?"

                ┌─────────────────────────┐
                │    Your Python Code      │
                │                         │
User ──► LLM ──►  "Call calculate() with  │──► Execute function
                │   args: '0.15 * 2400'"  │    → "360.0"
                │                         │
                └─────────┬───────────────┘
                          │  Result fed back
                          ▼
                ┌─────────────────────────┐
                │    LLM says:            │
                │  "15% of 2400 is 360."  │──► You see the answer
                └─────────────────────────┘
```

### Step by Step

1. **You define a "tool"** — a normal Python function + a JSON schema describing it
2. **You send the tool definitions + user message to the LLM**
3. **The LLM decides** if it needs a tool. If yes, it outputs structured JSON:
   ```json
   {
     "name": "calculate",
     "arguments": "{\"expression\": \"0.15 * 2400\"}"
   }
   ```
4. **Your code** receives this JSON, calls the real Python function, gets the result
5. **You send the result back** to the LLM as a "tool" role message
6. **The LLM** uses the result to craft its final response to the user

That's it. No magic. No agents. Just functions + JSON.

---

## 🏗️ Project Structure

```
pdf-tool-calling-assistant/
├── tools/
│   ├── calculator.py       # Tool 1: safe math evaluation
│   └── pdf_search.py       # Tool 2: keyword search in PDF text
├── core/
│   ├── llm_client.py       # Groq API wrapper + tool-calling loop
│   └── chat_loop.py        # Tool registration + interactive CLI
├── utils/
│   └── pdf_parser.py       # Extract text from PDF + chunk it
├── api/
│   └── main.py             # FastAPI REST endpoint
├── data/
│   └── sample_pdfs/        # Put your PDFs here
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Get a Groq API Key

Groq offers free API access with blazing-fast inference.

- Go to [console.groq.com/keys](https://console.groq.com/keys)
- Create an API key (starts with `gsk_`)
- Set it as an environment variable:

```bash
# Windows (Command Prompt)
set GROQ_API_KEY=gsk_your_key_here

# Windows (PowerShell)
$env:GROQ_API_KEY='gsk_your_key_here'

# macOS / Linux
export GROQ_API_KEY='gsk_your_key_here'
```

Or copy `.env.example` to `.env` and fill in your key.

### 2. Install Dependencies

```bash
cd pdf-tool-calling-assistant
pip install -r requirements.txt
```

### 3. Run the Interactive CLI

```bash
# Without a document
python -m core.chat_loop

# With a PDF loaded at startup
python -m core.chat_loop path/to/your/document.pdf
```

### 4. Run the API Server

```bash
python -m api.main
```

Then open http://localhost:8000 in your browser, or use curl:

```bash
# Upload a PDF
curl -X POST -F "file=@data/sample_pdfs/resume.pdf" http://localhost:8000/upload

# Ask a question
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is the candidate's experience with Python?"}' \
  http://localhost:8000/ask
```

---

## 🔧 Available Tools

### 1. Calculator (`calculate`)
- **What it does**: Evaluates math expressions safely
- **When the LLM uses it**: Any question involving numbers, percentages, computations
- **Examples**: "What's 15% of 2400?", "If I have 500 apples and give away 30%, how many remain?", "Calculate the square root of 144"

### 2. Document Search (`search_document`)
- **What it does**: Searches the uploaded PDF using keyword overlap
- **When the LLM uses it**: Questions about the document content
- **Examples**: "What is this paper about?", "Summarize the candidate's skills", "What methodology did they use?"

---

## 🧪 Try These Test Questions

After uploading a PDF like a resume or research paper:

| Question | Expected Tool Use |
|---|---|
| "What is this document about?" | `search_document` |
| "How many years of experience does the candidate have?" | `search_document` + possibly `calculate` |
| "What's 25% of the total pages?" | `calculate` |
| "What's the capital of France?" | No tool (LLM knowledge) |
| "Summarize the key findings" | `search_document` |
| "If the sample size is 1000 and 35% responded, how many is that?" | `calculate` |

---

## 📚 Learning Path

### Phase 0: Understand Tool-Calling (theory)
- Read: [Groq Function Calling Docs](https://console.groq.com/docs/tool-use)
- Read: This README's "What is Tool-Calling" section
- Key insight: The LLM outputs JSON, your code executes it

### Phase 1: Calculator Tool Only (what's built)
- Single tool: `calculate(expression)`
- Test: "What's 15% of 200?" → LLM calls calculate → returns answer
- Understand: The tool-calling loop (LLM → execute → feed back)

### Phase 2: PDF Extraction (what's built)
- `pdfplumber` extracts text from PDFs
- Text is split into overlapping chunks (500 words, 50 overlap)
- No vector DB — just plain text chunks in memory

### Phase 3: Document Search Tool (what's built)
- Second tool: `search_document(query)`
- LLM now chooses between calculate, search, or answer from knowledge
- Keyword scoring finds relevant chunks

### Phase 4: API Wrapper (what's built)
- FastAPI with `/upload` and `/ask` endpoints
- Stateless: document stays in memory until next upload

---

## 💡 What You'll Learn

- ✅ How tool-calling / function calling actually works (JSON in, JSON out)
- ✅ How to define tools with JSON Schema
- ✅ How to implement the tool-calling loop (LLM → execute → feed back)
- ✅ How the LLM decides WHICH tool to use (tool_choice="auto")
- ✅ How to handle multiple tools
- ✅ How to extract text from PDFs and chunk it for search
- ✅ How to build a simple FastAPI wrapper

---

## 🚀 Next Steps (After This Project)

1. **Add a vector search tool** — Replace keyword search with embeddings (sentence-transformers) + cosine similarity
2. **Add a web search tool** — Let the LLM fetch real-time information
3. **Add a code execution tool** — Run Python code in a sandbox
4. **Add memory** — Store conversations so the LLM remembers context
5. **Try multi-agent** — After you're comfortable with tool-calling, explore agent frameworks

---

## 📝 License

MIT — Do whatever you want with this. Built for learning.

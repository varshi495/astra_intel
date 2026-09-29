#  ASTRA INTEL

> **Armed Squad for Tactical Readiness & Awareness**  
> Defence document intelligence — answers grounded strictly in your document.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![SentenceTransformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-orange)](https://sbert.net)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Qwen-6D28D9)](https://groq.com)

---

ASTRA INTEL is a **Retrieval-Augmented Generation (RAG)** web application built with Streamlit. Upload one or more PDF documents, ask questions in plain English, and receive answers **traceable to the exact page and passage** they came from — with zero hallucination risk when the document contains the answer.

---

## ✨ Features

| Feature | Details |
|---|---|
| **PDF Upload** | Upload single or multiple PDFs simultaneously via drag-and-drop sidebar |
| **OCR Fallback** | Scanned/image PDFs are processed via Tesseract OCR automatically |
| **100% Local Embeddings** | Chunks are embedded with `all-MiniLM-L6-v2` on-device — document content never leaves your machine during indexing |
| **Grounded Q&A** | LLM answers using only retrieved excerpts; replies with a clear rejection when the document has no relevant content |
| **Auto-Summarisation** | 5–7 sentence document overview generated on upload |
| **Smart Question Chips** | AI-suggested starter questions appear after upload |
| **Page-Level Citations** | Every answer cites the exact page number and passage it was drawn from |
| **Multi-Turn Chat** | Conversation history maintained across turns (last 3 turns sent for context) |
| **Retrieval Evaluator** | CLI script to benchmark Top-1 and Top-K retrieval accuracy on labelled data |

---

## 🏗️ System Architecture

### High-Level Data Flow

```mermaid
flowchart TD
    User(["👤 User"])

    subgraph Ingestion ["📥 Document Ingestion — runs on upload"]
        direction TB
        UP["PDF Upload\nStreamlit sidebar"]
        EX["extract_pages\nPyMuPDF · Tesseract OCR fallback"]
        CH["chunk_pages\n180-word windows · 40-word overlap"]
        EM["get_embedder\nall-MiniLM-L6-v2 · SentenceTransformer"]
        VI[("DocIndex\nIn-Memory Vector Store\nnumpy ndarray")]

        UP --> EX --> CH --> EM --> VI
    end

    subgraph Query ["💬 Query Pipeline — runs on each question"]
        direction TB
        QE["Embed query\nsame local model"]
        CS["Cosine Similarity\nnp.argsort scores TOP_K"]
        CB["Context Builder\nformat Doc Page N excerpts"]
        LLM["Groq API\nQwen LLM · temp=0"]
        ANS["Answer + Citations\nback to Streamlit UI"]

        QE --> CS --> CB --> LLM --> ANS
    end

    User -->|"Upload PDFs"| UP
    User -->|"Ask question"| QE
    VI --> CS
    ANS --> User
```

### Project Structure

```text
astra_intel/
├── src/
│   ├── app.py          ← Streamlit UI (session state, CSS, chat loop)
│   └── rag.py          ← RAG engine (extraction → chunking → embedding → LLM)
├── tests/
│   ├── eval.py                      ← Top-1 / Top-K retrieval benchmarker
│   ├── eval_questions.example.json  ← Sample evaluation dataset
│   └── example-questions.md         ← Human-readable question guide
├── assets/
│   └── astra_logo.jpeg
├── .env                ← GROQ_API_KEY, LLM_MODEL, MIN_SCORE  (not committed)
├── requirements.txt
└── pyrightconfig.json
```

---

## 🔬 Codebase Deep-Dive

### `src/rag.py` — The RAG Engine

The engine executes a 5-step pipeline on every document upload and query:

#### Step 1 · Extraction (`extract_pages`, `_ocr_page`)
Reads the PDF from bytes using **PyMuPDF (`fitz`)**. Each page's raw text is extracted. If a page yields fewer than 20 characters (indicating a scanned image), **Tesseract OCR** is invoked via `pytesseract` at 200 DPI. Returns a list of `(page_number, text)` tuples where page numbers are 1-indexed.

#### Step 2 · Chunking (`chunk_pages`)
Splits each page's text into **overlapping word windows**:
- Window size: `CHUNK_WORDS = 180` words (~1–2 paragraphs)
- Overlap: `CHUNK_OVERLAP = 40` words (prevents sentences from being severed at chunk boundaries)
- Forward step: `180 − 40 = 140` words per chunk
- Chunks shorter than 15 words are discarded unless they are the only content on a page

Each chunk carries `{"page": int, "text": str, "doc": str}`.

#### Step 3 · Embedding & Indexing (`DocIndex`)
`DocIndex` accepts `List[Tuple[bytes, str]]` (one or more documents). For each document it:
1. Calls `extract_pages()` → `chunk_pages()`
2. Encodes all chunks with `SentenceTransformer("all-MiniLM-L6-v2")`, normalised to unit vectors
3. Stores the result as an `(N, 384)` numpy matrix in `self.emb`

`DocIndex.search(query, k=TOP_K)` encodes the query with the same model and computes **cosine similarity as a matrix dot product** (`self.emb @ q_vec`), then returns the top-`k` chunks sorted by score.

Key attributes:

| Attribute | Type | Description |
|---|---|---|
| `chunks` | `List[Dict]` | All extracted chunks with `page`, `text`, `doc` |
| `emb` | `np.ndarray` | Pre-computed unit-normalised embeddings `(N, 384)` |
| `pages` | `int` | Total page count across all loaded documents |
| `name` | `str` | Comma-separated list of document filenames |
| `names` | `List[str]` | Individual document filenames |

#### Step 4 · LLM Integration (`_chat`, `_get_client`)
Uses the **OpenAI Python SDK** pointed at the **Groq API**. The client is a lazily-initialised singleton to reuse HTTP connection pools. `temperature=0` enforces deterministic, factual responses. Qwen3 `<think>…</think>` reasoning blocks are stripped from the output via regex.

Supports **local LLM servers** (Ollama, vLLM) by setting `LLM_BASE_URL` in `.env`.

#### Step 5 · Grounded Answering (`answer`, `summarize`, `suggest_questions`)
- **`answer()`** — If the top chunk scores below `MIN_SCORE`, the query is rejected without an LLM call. Otherwise, excerpts are injected into a strict system prompt. If the model responds `NOT_FOUND`, a friendly rejection message is returned instead.
- **`summarize()`** — Samples up to 12 evenly-spaced chunks and asks the LLM for a 5–7 sentence overview.
- **`suggest_questions()`** — Samples up to 10 chunks and generates `n` specific questions a reader might ask.

---

### `src/app.py` — The Streamlit Frontend

Streamlit re-executes the entire script on every user interaction. Persistent state lives in `st.session_state`:

| Key | Type | Purpose |
|---|---|---|
| `index` | `DocIndex \| None` | Active vector index for the loaded documents |
| `summary` | `str \| None` | Auto-generated document summary |
| `history` | `List[Dict]` | All chat turns: `{q, a, sources, supported}` |
| `suggested_questions` | `List[str]` | AI-generated starter question chips |
| `pending_question` | `str \| None` | Question queued from a chip click |

UI flow: **Sidebar upload** → `DocIndex` → `summarize` → `suggest_questions` → **Empty state / Chat loop** → `answer` → `st.rerun`.

---

### `tests/eval.py` — The Retrieval Evaluator

CLI tool that measures how reliably the vector search retrieves the correct page:

```bash
python tests/eval.py path/to/document.pdf tests/eval_questions.example.json
```

**Input JSON format:**
```json
[
  { "q": "What is the maximum operational range?", "page": 12 },
  { "q": "Who signed the protocol amendment?",     "page": 47 }
]
```

**Output metrics:**
- **Top-1 Accuracy** — correct page was the single best match
- **Top-K Accuracy** — correct page appeared anywhere in the top-K results

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) *(optional — only needed for scanned PDFs)*
- A free [Groq API key](https://console.groq.com/keys)

### 1 — Clone & create virtual environment

```bash
git clone https://github.com/varshi495/astra_intel.git
cd astra_intel

python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here

# Optional overrides (shown with defaults)
LLM_MODEL=qwen/qwen3-8b
LLM_BASE_URL=https://api.groq.com/openai/v1   # or http://localhost:11434/v1 for Ollama
MIN_SCORE=0.10
```

### 4 — Run the application

```bash
streamlit run src/app.py
```

### 5 — Run retrieval evaluation

```bash
python tests/eval.py path/to/document.pdf tests/eval_questions.example.json
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq cloud API key |
| `LLM_MODEL` | `qwen/qwen3-8b` | Model identifier passed to the API |
| `LLM_BASE_URL` | Groq endpoint | API base URL — swap for Ollama or vLLM |
| `MIN_SCORE` | `0.10` | Cosine similarity floor for "supported" answers |
| `CHUNK_WORDS` | `180` | Words per chunk *(set in `rag.py`)* |
| `CHUNK_OVERLAP` | `40` | Overlapping words between consecutive chunks |
| `TOP_K` | `4` | Number of chunks retrieved per query |

---

## 🙏 Attribution & Acknowledgements

| Component | Library / Service |
|---|---|
| Web framework | [Streamlit](https://streamlit.io/) |
| PDF parsing | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) |
| OCR (optional) | [Tesseract](https://github.com/tesseract-ocr/tesseract) via [pytesseract](https://pypi.org/project/pytesseract/) |
| Local embeddings | [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via [SentenceTransformers](https://sbert.net/) |
| LLM inference | [Qwen](https://huggingface.co/Qwen) open-source model via [Groq API](https://groq.com/) |
| Vector maths | [NumPy](https://numpy.org/) |
| LLM client | [openai-python](https://github.com/openai/openai-python) |
| Fonts | [Space Grotesk](https://fonts.google.com/specimen/Space+Grotesk) · [Inter](https://fonts.google.com/specimen/Inter) via Google Fonts |

All core application logic (`src/app.py`, `src/rag.py`, `tests/eval.py`) is original work.

> **Security note:** No `.env` files or API keys are committed to this repository.

---

*See [`AI_USAGE.md`](AI_USAGE.md) for the full AI tools disclosure.*

#  ASTRA INTEL

> **Armed Squad for Tactical Readiness & Awareness**  
> Defence document intelligence with locally grounded answers.

ASTRA INTEL is a specialized Retrieval-Augmented Generation (RAG) web application built with Streamlit. It allows users to upload PDF documents, processes them securely (using local embeddings), and provides highly accurate answers to user queries grounded *strictly* in the provided text. Every answer cites the exact page and passage it was retrieved from.

##  Features

- **Document Processing**: Upload PDFs or scanned documents. Text extraction falls back to OCR (using Tesseract) if the PDF is purely image-based.
- **Local Embeddings**: Document chunks are embedded locally using `all-MiniLM-L6-v2` via `sentence-transformers`, meaning your sensitive document content doesn't leave your machine during the indexing phase.
- **Grounded Q&A**: Answers are generated strictly from the provided text using large language models (defaulting to Qwen models via the Groq API).
- **Auto-Summarization**: Automatically generates a 5-7 sentence summary of the document upon upload.
- **Smart Suggestions**: Suggests relevant questions based on the document's contents to help users get started.
- **Source Verification**: Every answer includes the exact page number and text snippet used to generate it, ensuring 100% traceability.
- **Usable interface (upload flow + question flow)** with basic error states (corrupt/empty PDF, missing API key, no document uploaded yet).
- **Multi-turn conversation** with visible history.
- **Page-level citations** on answers.
- **Consistent answers** across repeated questions (grounded retrieval, not lucky guessing).

---

##  Project Structure

```text
d:\astra_intel\
├── src/
│   ├── app.py          # The main Streamlit web application interface
│   └── rag.py          # Core RAG logic (extraction, chunking, embedding, LLM calls)
├── tests/
│   ├── eval.py         # Evaluation script to test top-k retrieval accuracy
│   └── eval_questions.example.json 
├── .env                # Environment variables (GROQ_API_KEY, LLM_MODEL)
├── requirements.txt    # Python dependencies
└── pyrightconfig.json  # IDE configuration for module resolution
```

###  Codebase Explanation

#### 1. `src/app.py` (The User Interface)
This is the Streamlit frontend. It features a custom-styled, dark-themed UI with CSS injections to provide a highly polished experience.
- **Sidebar**: Handles PDF file uploads and triggers the document processing pipeline.
- **Session State**: Manages the chat history, current document index, generated summary, and suggested questions.
- **Chat Interface**: A conversational UI where users can ask questions. It displays the AI's response along with expanding tabs that reveal the exact source chunks and similarity scores.

#### 2. `src/rag.py` (The RAG Engine)
This is the backbone of the application. It handles the entire Retrieval-Augmented Generation pipeline:
- **Extraction (`extract_pages`, `_ocr_page`)**: Reads the PDF using `PyMuPDF` (`fitz`). If a page has very little text (e.g., it's a scan), it attempts to extract text using `pytesseract`.
- **Chunking (`chunk_pages`)**: Splits the extracted text into overlapping word windows (180 words per chunk, 40-word overlap) while keeping track of the page numbers.
- **Embedding & Indexing (`DocIndex`)**: Uses a local `SentenceTransformer` model to convert text chunks into vector embeddings. It includes a `search()` method to find the most relevant chunks for a given user query using cosine similarity.
- **LLM Integration (`_chat`, `answer`, `summarize`, `suggest_questions`)**: Communicates with the Groq API (using the `openai` Python client) to synthesize answers, summarize the document, and generate suggested questions based on the retrieved context.

#### 3. `tests/eval.py` (The Evaluator)
A lightweight script used to benchmark the retrieval engine's performance. 
- It takes a PDF and a JSON file containing evaluation questions and the correct page numbers where the answers reside.
- It processes the PDF, runs searches for each question, and reports the Top-1 and Top-k accuracy (i.e., how often the correct page was successfully retrieved in the top results).

---

##  Getting Started

### 1. Prerequisites
- Python 3.10+
-  [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system if you want to support purely scanned PDFs.

### 2. Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=qwen/qwen3.8-27b
MIN_SCORE=0.10
```

### 4. Running the Application

Launch the Streamlit interface:

```bash
streamlit run src/app.py
```

### 5. Running Evaluations

To test the retrieval accuracy on a sample document:

```bash
python tests/eval.py path/to/document.pdf path/to/questions.json
```
"""
ASTRA INTEL - Core Retrieval-Augmented Generation (RAG) Pipeline.
==================================================================

What is RAG (Retrieval-Augmented Generation)?
---------------------------------------------
Standard Large Language Models (LLMs) can sometimes "hallucinate" (make up facts)
or fail to know private/recent documents. RAG solves this in 5 simple steps:

1. Extraction: Read raw text from a PDF file (with OCR fallback for scanned images).
2. Chunking:   Split long text into smaller, overlapping chunks with page tracking.
3. Embedding:  Convert text chunks into vector numbers (embeddings) using a local model.
4. Retrieval:  When a user asks a question, find the most mathematically similar chunks.
5. Synthesis:  Feed the user's question AND the matching document excerpts to an LLM,
               instructing it to answer using ONLY the provided excerpts with page citations.

Why is this safe and private?
-----------------------------
Document chunking and embeddings run 100% locally on your machine via SentenceTransformers.
Only the specific matching excerpts for a query are sent to the LLM to formulate the final answer.
"""

from __future__ import annotations

import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymupdf as fitz
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file (e.g. GROQ_API_KEY, LLM_MODEL)
load_dotenv()

# ==============================================================================
# Configuration & Constants
# ==============================================================================

# Embedding Model:
# 'all-MiniLM-L6-v2' is a lightweight, high-performance model that converts text
# into 384-dimensional vectors. It runs completely locally on CPU with zero API costs.
EMBED_MODEL: str = "all-MiniLM-L6-v2"

# LLM Model used for question answering and document summarization.
# Defaults to Qwen 3.8-27b on Groq for lightning-fast inference.
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")

# Similarity Threshold:
# Cosine similarity ranges from -1.0 to 1.0 (with 1.0 being an exact match).
# If the top retrieved chunk has a score below MIN_SCORE, we know the document
# does not contain relevant information and we tell the user honestly.
MIN_SCORE: float = float(os.getenv("MIN_SCORE", "0.10"))

# Chunking Parameters:
# Documents are split into windows of words so they fit into context windows.
CHUNK_WORDS: int = 180       # Size of each chunk in words (~1-2 paragraphs)
CHUNK_OVERLAP: int = 40      # Overlap between consecutive chunks to avoid breaking sentences in half
TOP_K: int = 4               # Number of best-matching chunks retrieved per query

# Singleton caches for expensive objects to keep execution fast and efficient
_embedder: Optional[SentenceTransformer] = None
_cached_client: Optional[OpenAI] = None


class DocError(Exception):
    """
    Custom exception for friendly user-facing errors.
    Examples: corrupted PDF file, missing API keys, or empty document.
    """
    pass


def get_embedder() -> SentenceTransformer:
    """
    Lazy-loads and caches the SentenceTransformer model.
    
    Why cache?
    Loading a machine learning model from disk takes 1-2 seconds.
    By loading it once and storing it in global `_embedder`, subsequent calls
    take 0 milliseconds.
    """
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# ==============================================================================
# Step 1: Text Extraction (with OCR Fallback)
# ==============================================================================

def _ocr_page(page: fitz.Page) -> str:
    """
    Extracts text from an image-based/scanned PDF page using Optical Character Recognition (OCR).
    
    How it works:
    1. Renders the PDF page into an image (pixmap) at 200 DPI for high clarity.
    2. Passes the image bytes to Tesseract OCR via `pytesseract`.
    3. If Tesseract is not installed on the system, it fails gracefully by returning an empty string.
    """
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        result = pytesseract.image_to_string(img)
        if isinstance(result, dict):
            text = result.get("text", "")
            if isinstance(text, list):
                text = " ".join(str(t) for t in text if str(t).strip())
            return str(text).strip()
        return str(result).strip()
    except Exception:
        # OCR is optional; return empty string if OCR engine or libraries are unavailable
        return ""


def extract_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Reads a PDF file from memory and extracts readable text page-by-page.
    
    Args:
        pdf_bytes: The raw byte content of the uploaded PDF file.
        
    Returns:
        A list of tuples: (page_number, extracted_text).
        Note: Page numbers are 1-indexed (Page 1 is the first page).
        
    Raises:
        DocError: If the PDF is corrupted, password-locked, or contains no readable text.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise DocError("Couldn't open this file. It may be corrupt or not a real PDF.") from e

    pages: List[Tuple[int, str]] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        raw_text = page.get_text()
        text = raw_text.strip() if isinstance(raw_text, str) else ""
        
        # If very little text was found (fewer than 20 characters),
        # the page is likely a scanned photo or image. We attempt OCR.
        if len(text) < 20:
            text = _ocr_page(page)
            
        if text:
            pages.append((i + 1, text))

    if not pages:
        raise DocError("No readable text found (empty PDF, or scanned document without OCR installed).")
    
    return pages


# ==============================================================================
# Step 2: Document Chunking
# ==============================================================================

def chunk_pages(pages: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    """
    Splits page text into small, overlapping windows of words ('chunks').
    
    Why Chunking is Necessary:
    1. Embedding models work best on short, focused passages rather than full 50-page books.
    2. Overlapping (e.g. 40 words) ensures that key thoughts crossing chunk boundaries
       are not severed.
    3. We keep track of the exact `page` number for every chunk so the AI can cite it!
    
    Efficiency Note:
    We slice words directly `words[start:start + CHUNK_WORDS]` and check length
    without redundantly re-splitting strings.
    """
    chunks: List[Dict[str, Any]] = []
    step = CHUNK_WORDS - CHUNK_OVERLAP  # e.g., 180 - 40 = 140 words forward step

    for page_no, text in pages:
        words = text.split()
        total_words = len(words)
        
        if total_words == 0:
            continue

        for start in range(0, total_words, step):
            chunk_slice = words[start:start + CHUNK_WORDS]
            piece = " ".join(chunk_slice)
            
            # Keep chunk if it has at least 15 words, or if it's the only text on that page
            if len(chunk_slice) >= 15 or (start == 0 and piece):
                chunks.append({"page": page_no, "text": piece})
                
            if start + CHUNK_WORDS >= total_words:
                break

    return chunks


# ==============================================================================
# Step 3: Vector Embeddings & Indexing (DocIndex)
# ==============================================================================

class DocIndex:
    """
    In-memory vector database for a single document.
    
    Concepts for Beginners:
    - Embedding: A list of floating-point numbers (e.g., [0.12, -0.45, ...]) that captures
      the semantic meaning of a sentence.
    - Cosine Similarity: A mathematical dot product measuring how close two vectors point in space.
      If two texts mean similar things, their vectors will have a high similarity score (close to 1.0).
    """

    def __init__(self, docs: List[Tuple[bytes, str]]) -> None:
        self.chunks: List[Dict[str, Any]] = []
        self.names: List[str] = [name for _, name in docs]
        self.name: str = ", ".join(self.names)
        self.pages: int = 0
        
        for pdf_bytes, name in docs:
            pages = extract_pages(pdf_bytes)
            self.pages += len(pages)
            doc_chunks = chunk_pages(pages)
            for c in doc_chunks:
                c["doc"] = name
            self.chunks.extend(doc_chunks)
        
        if not self.chunks:
            raise DocError("No text passages could be extracted from these documents.")

        texts = [c["text"] for c in self.chunks]
        
        # Precompute vector embeddings for all chunks in the document.
        # normalize_embeddings=True makes cosine similarity equal to a simple matrix dot product (@).
        self.emb: np.ndarray = get_embedder().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def search(self, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Finds the top `k` most relevant chunks for a given query text.
        
        Args:
            query: The user's question or search phrase.
            k: Maximum number of top chunks to return.
            
        Returns:
            A list of chunk dictionaries sorted from highest to lowest similarity score:
            [{"score": 0.85, "page": 3, "text": "..."}, ...]
        """
        if not query.strip() or len(self.chunks) == 0:
            return []

        # 1. Convert user's question into a normalized vector
        q_vec = get_embedder().encode([query], normalize_embeddings=True)[0]
        
        # 2. Compute similarity against all chunks simultaneously using matrix multiplication
        # Since both vectors are unit normalized, dot product = cosine similarity!
        scores = self.emb @ q_vec
        
        # 3. Efficient Top-K retrieval
        num_results = min(k, len(self.chunks))
        if num_results <= 0:
            return []
            
        # Fast index ranking:
        # np.argsort sorts ascending; [-scores] gives descending order
        top_indices = np.argsort(-scores)[:num_results]
        
        return [
            {"score": float(scores[idx]), **self.chunks[idx]}
            for idx in top_indices
        ]


# ==============================================================================
# Step 4: LLM Integration (OpenAI Client & Groq API)
# ==============================================================================

def _get_client() -> OpenAI:
    """
    Returns a reusable OpenAI client configured for Groq or local LLMs.
    
    Efficiency:
    Reusing a single client instance maintains HTTP connection pooling and avoids
    creating fresh TCP connections on every question.
    """
    global _cached_client
    if _cached_client is not None:
        return _cached_client

    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    key = os.getenv("LLM_API_KEY", os.getenv("GROQ_API_KEY", "local"))

    # If pointing to Groq cloud API but no key was provided in .env
    if base_url == "https://api.groq.com/openai/v1" and key == "local":
        raise DocError(
            "GROQ_API_KEY is missing. Please add your API key to the .env file "
            "or set LLM_BASE_URL if you are running a local LLM server (like Ollama or vLLM)."
        )

    _cached_client = OpenAI(api_key=key, base_url=base_url)
    return _cached_client


def _chat(messages: Any, max_tokens: int = 500) -> str:
    """
    Sends a chat completion request to the LLM and cleans up the response.
    
    Args:
        messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
        max_tokens: Maximum number of tokens the model is allowed to generate.
        
    Returns:
        The clean text response from the model.
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0,  # 0 = greedy decoding for deterministic, factual consistency
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        
        # Remove reasoning tags (like Qwen3 <think>...</think> blocks) if present
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        
        if not content:
            raise DocError("The language model returned an empty response. Please try again.")
            
        return content
    except DocError:
        raise
    except Exception as e:
        raise DocError(f"LLM request failed: {e}") from e


def summarize(index: DocIndex) -> str:
    """
    Generates a concise 5-7 sentence summary of the entire document.
    
    How it works:
    Instead of sending all 100 pages (which would exceed context limits and be slow),
    it samples up to 12 chunks evenly distributed from the beginning, middle, and end
    of the document to provide balanced context to the LLM.
    """
    total_chunks = len(index.chunks)
    if total_chunks == 0:
        return "No text available to summarize."

    # Pick up to 12 evenly spaced chunks across the document
    sample_size = min(total_chunks, 12)
    sample_indices = sorted(set(int(i) for i in np.linspace(0, total_chunks - 1, sample_size)))
    
    context = "\n\n".join(
        f"[{index.chunks[i].get('doc', 'Doc')}, Page {index.chunks[i]['page']}] {index.chunks[i]['text']}"
        for i in sample_indices
    )
    
    return _chat(
        [
            {
                "role": "system",
                "content": (
                    "You summarize documents. Use ONLY the given excerpts. "
                    "Write a concise, informative summary (5-7 sentences)."
                ),
            },
            {"role": "user", "content": context},
        ],
        max_tokens=400,
    )


def suggest_questions(index: DocIndex, n: int = 4) -> List[str]:
    """
    Asks the LLM to inspect the document and brainstorm `n` relevant starter questions.
    
    Why:
    Helps the user immediately see what topics the document covers without guessing.
    """
    total_chunks = len(index.chunks)
    if total_chunks == 0:
        return []

    # Pick up to 10 sample chunks
    sample_size = min(total_chunks, 10)
    sample_indices = sorted(set(int(i) for i in np.linspace(0, total_chunks - 1, sample_size)))
    
    context = "\n\n".join(
        f"[{index.chunks[i].get('doc', 'Doc')}, Page {index.chunks[i]['page']}] {index.chunks[i]['text']}"
        for i in sample_indices
    )
    
    try:
        raw = _chat(
            [
                {
                    "role": "system",
                    "content": (
                        f"Based on the document excerpts below, generate exactly {n} concise, specific questions "
                        "that a reader would likely want to ask. "
                        "Output ONLY the questions, one per line, no numbering, no bullet points, no extra text."
                    ),
                },
                {"role": "user", "content": context},
            ],
            max_tokens=250,
        )
        questions = [q.strip("•-– 1234567890.").strip() for q in raw.strip().splitlines() if q.strip()]
        return questions[:n]
    except DocError:
        return []


# ==============================================================================
# Step 5: Grounded Answering
# ==============================================================================

NOT_FOUND_MSG: str = "The uploaded documents do not contain information regarding this topic."


def answer(index: DocIndex, question: str, history: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
    """
    Answers a question strictly grounded in the uploaded document(s).
    
    Process:
    1. Search vector index for the top matching chunks.
    2. Check if similarity score exceeds MIN_SCORE. If not, reject query.
    3. Format context with document and page tags: "[Doc, Page X] ...".
    4. Provide strict grounded prompt:
       - Cite document and page numbers (Document Name, p. X).
       - Never hallucinate, extrapolate, or invent numbers/statistics.
       - If a specific statistic/claim is not present, state honestly that the documents
         do not contain it, while citing what the documents DO state about the topic.
    5. Maintain multi-turn conversational context (last 3 questions & answers).
    
    Returns:
        A dictionary with:
        - 'answer': The formatted response string.
        - 'supported': True if answered with confidence from the doc, False if not found.
        - 'sources': List of retrieved chunk dictionaries with score, page, and text.
    """
    hits = index.search(question)
    
    # If no results or the best result has poor similarity, reject immediately
    if not hits or hits[0]["score"] < MIN_SCORE:
        return {"answer": NOT_FOUND_MSG, "supported": False, "sources": hits}

    # Format the retrieved excerpts
    context = "\n\n".join(f"[{h.get('doc', 'Doc')}, Page {h['page']}] {h['text']}" for h in hits)
    
    system_prompt = (
        "You are an AI document intelligence assistant adhering strictly to grounded retrieval.\n"
        "Answer the user's question using ONLY the provided excerpts below.\n"
        "Always cite the source document and page numbers using format: (Document Name, p. X).\n\n"
        "CRITICAL GROUNDING & HONESTY RULES:\n"
        "1. Never extrapolate, speculate, or invent facts, numbers, dates, or statistics.\n"
        "2. If the user asks for a specific statistic, percentage, or claim that is NOT present in the excerpts:\n"
        "   - State clearly and directly that the documents do not provide or contain this statistic/claim.\n"
        "   - Describe what the documents DO state about the topic, citing the relevant source document and page numbers.\n"
        "   - Do NOT invent or guess any numbers (e.g. do NOT invent '60% of missions are fully autonomous').\n"
        "3. ONLY if the excerpts have zero relevance or mention of the subject matter at all, reply: "
        "'The uploaded documents do not contain information regarding this topic.' Do NOT use this phrase if the topic itself is discussed in the excerpts."
    )
    
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    
    # Include recent chat history (up to last 3 conversation turns) for multi-turn coherence
    for past_q, past_a in (history or [])[-3:]:
        messages.append({"role": "user", "content": past_q})
        messages.append({"role": "assistant", "content": past_a})
        
    messages.append({"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"})

    out = _chat(messages, max_tokens=800)
    
    # If the model explicitly stated it cannot find the answer or topic in the excerpts
    clean_out = out.strip()
    if clean_out.upper().startswith("NOT_FOUND") or clean_out.startswith("The uploaded documents do not contain information regarding this topic"):
        return {"answer": NOT_FOUND_MSG, "supported": False, "sources": hits}
        
    return {"answer": out, "supported": True, "sources": hits}


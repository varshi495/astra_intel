"""Core RAG pipeline for ASTRA INTEL: PDF -> pages -> chunks -> embeddings -> retrieval -> grounded answer."""
import io
import os

import pymupdf as fitz
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"          # small, fast, runs locally on CPU
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))  # below this -> "not found in document"
CHUNK_WORDS = 180
CHUNK_OVERLAP = 40
TOP_K = 4

_embedder = None


class DocError(Exception):
    """User-facing error (bad PDF, missing key, etc.)."""


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# ---------- 1. extraction ----------
def _ocr_page(page) -> str:
    """OCR fallback for scanned pages. Silently returns '' if OCR libs/binary are missing."""
    try:
        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""


def extract_pages(pdf_bytes: bytes):
    """Returns list of (page_number, text). Page numbers start at 1."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise DocError("Couldn't open this file. It may be corrupt or not a real PDF.") from e

    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text().strip()
        if len(text) < 20:  # probably scanned -> try OCR
            text = _ocr_page(page)
        if text:
            pages.append((i + 1, text))
    if not pages:
        raise DocError("No readable text found (empty PDF, or scanned without OCR available).")
    return pages


# ---------- 2. chunking ----------
def chunk_pages(pages):
    """Overlapping word-window chunks, kept per page so every chunk has a page number."""
    chunks = []
    step = CHUNK_WORDS - CHUNK_OVERLAP
    for page_no, text in pages:
        words = text.split()
        for start in range(0, max(len(words), 1), step):
            piece = " ".join(words[start:start + CHUNK_WORDS])
            if len(piece.split()) >= 15 or (start == 0 and piece):
                chunks.append({"page": page_no, "text": piece})
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks


# ---------- 3. index + retrieval ----------
class DocIndex:
    def __init__(self, pdf_bytes: bytes, name: str = "document"):
        self.name = name
        self.pages = extract_pages(pdf_bytes)
        self.chunks = chunk_pages(self.pages)
        texts = [c["text"] for c in self.chunks]
        self.emb = get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def search(self, query: str, k: int = TOP_K):
        q = get_embedder().encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q  # cosine similarity (vectors are normalized)
        top = np.argsort(-scores)[:k]
        return [{"score": float(scores[i]), **self.chunks[i]} for i in top]


# ---------- 4. LLM ----------
def _client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise DocError("GROQ_API_KEY is missing. Copy .env.example to .env and add your key.")
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")


def _chat(messages, max_tokens=500):
    try:
        r = _client().chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=0, max_tokens=max_tokens
        )
        content = r.choices[0].message.content
        if not content or not content.strip():
            raise DocError("The language model returned an empty response. Please try again.")
        return content.strip()
    except DocError:
        raise
    except Exception as e:
        raise DocError(f"LLM request failed: {e}") from e


def summarize(index: DocIndex) -> str:
    n = len(index.chunks)
    picks = sorted(set(int(i) for i in np.linspace(0, n - 1, min(n, 12))))  # spread across the doc
    context = "\n\n".join(f"[Page {index.chunks[i]['page']}] {index.chunks[i]['text']}" for i in picks)
    return _chat(
        [
            {"role": "system", "content": "You summarize documents. Use ONLY the given excerpts. Write a concise summary (5-7 sentences)."},
            {"role": "user", "content": context},
        ],
        max_tokens=400,
    )


NOT_FOUND_MSG = "I couldn't find this in the uploaded document."


def answer(index: DocIndex, question: str, history=None):
    """Returns dict: answer, supported (bool), sources (list of chunks)."""
    hits = index.search(question)
    if not hits or hits[0]["score"] < MIN_SCORE:
        return {"answer": NOT_FOUND_MSG, "supported": False, "sources": hits}

    context = "\n\n".join(f"[Page {h['page']}] {h['text']}" for h in hits)
    system = (
        "Answer the question using ONLY the excerpts below. Cite pages like (p. 3). "
        "If the excerpts do not contain the answer, reply exactly: NOT_FOUND"
    )
    msgs = [{"role": "system", "content": system}]
    for q, a in (history or [])[-3:]:  # short multi-turn memory
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"})

    out = _chat(msgs)
    if "NOT_FOUND" in out:
        return {"answer": NOT_FOUND_MSG, "supported": False, "sources": hits}
    return {"answer": out, "supported": True, "sources": hits}

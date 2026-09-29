"""
ASTRA INTEL - Web Application Frontend (Streamlit)
=================================================

How Streamlit Works (A Primer for Beginners):
----------------------------------------------
Unlike traditional web frameworks (like Django, Flask, or React) where you manage
client-server state and event loops manually, Streamlit executes your Python script
from TOP to BOTTOM whenever:
  1. The user first opens the web page.
  2. The user interacts with any widget (clicks a button, uploads a file, types text).

Why `st.session_state` is Essential:
------------------------------------
Because the script reruns from scratch on every interaction, normal local variables
would be destroyed and reset! To remember things across reruns (such as:
  - the parsed document index,
  - conversation history,
  - document summary,
  - suggested questions),
we store them in `st.session_state`. This is Streamlit's persistent dictionary.

How to Run:
-----------
Run the following command from the root project folder:
    streamlit run src/app.py
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Import our custom RAG engine functions and classes from src/rag.py
from rag import DocError, DocIndex, answer, summarize, suggest_questions

# Configure page metadata, browser tab title, favicon, and wide layout mode
st.set_page_config(
    page_title="ASTRA INTEL",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Custom CSS / Theme Styling
# ==============================================================================
# We inject custom CSS to give ASTRA INTEL a tactical, military-grade dark aesthetic
# featuring Space Grotesk (for headers) and Inter (for clean body typography),
# accompanied by gold accent highlights (#B8860B).
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
  /* Global typography & background */
  html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
  .stApp { background-color: #000000 !important; }
  header[data-testid="stHeader"] { background-color: transparent !important; }

  /* Headings with tactical military styling */
  h1, h2, h3, h4 {
    font-family: 'Space Grotesk', Arial, sans-serif !important;
    letter-spacing: 0.04em;
  }

  /* Sidebar styling */
  [data-testid="stSidebar"] {
    background-color: #000000 !important;
    border-right: 1px solid rgba(184,134,11,0.2) !important;
  }

  /* Primary action button (Gold gradient) */
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #B8860B 0%, #7A5A08 100%) !important;
    color: #FFFFFF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    letter-spacing: 0.06em;
    transition: all 0.2s ease !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #D4A017 0%, #B8860B 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(184,134,11,0.4) !important;
  }

  /* Secondary buttons (Used for clickable suggestion chips) */
  .stButton > button[kind="secondary"] {
    background-color: #22282E !important;
    color: #B8860B !important;
    border: 1px solid #B8860B66 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button[kind="secondary"]:hover {
    background-color: #B8860B !important;
    color: #1B1F23 !important;
    border-color: #B8860B !important;
  }

  /* Chat message bubble styling */
  [data-testid="stChatMessage"] {
    background-color: #22282E !important;
    border: 1px solid rgba(184,134,11,0.15) !important;
    border-radius: 12px !important;
    padding: 12px !important;
  }

  /* Chat input text box */
  [data-testid="stChatInputTextArea"] {
    background-color: #22282E !important;
    border: 1px solid #B8860B55 !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
  }

  /* Expandable source accordions */
  [data-testid="stExpander"] summary {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: #B8860B !important;
  }

  /* Custom gold scrollbar */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: #1B1F23; }
  ::-webkit-scrollbar-thumb { background: #B8860B66; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #B8860B; }
</style>
""")

# ==============================================================================
# Hero Header
# ==============================================================================
st.html("""
<div style="
  display:flex; align-items:center; gap:16px;
  padding: 0 0 18px 0;
  border-bottom: 1px solid rgba(184,134,11,0.3);
  margin-bottom: 20px;
">
  <div>
    <h1 style="
      font-family: 'Space Grotesk', Arial, sans-serif;
      font-size: 2.2rem; font-weight: 700; margin: 0;
      color: #B8860B;
      text-shadow: 0 0 30px rgba(184,134,11,0.4);
      letter-spacing: 0.12em;
    ">🛰️ ASTRA INTEL</h1>
    <p style="
      font-family: 'Inter', sans-serif; font-size: 0.85rem;
      color: #8A8F96; margin: 4px 0 0 0; letter-spacing: 0.04em;
    ">Armed Squad for Tactical Readiness &amp; Awareness &nbsp;·&nbsp;
       Defence document intelligence &nbsp;·&nbsp;
       Answers grounded strictly in your document
    </p>
  </div>
</div>
""")

# ==============================================================================
# Session State Initialization
# ==============================================================================
# `st.session_state` preserves variables across browser reruns.
# .setdefault(key, default) only sets the value if it doesn't already exist.
ss = st.session_state
ss.setdefault("index", None)                 # Holds the current DocIndex object
ss.setdefault("summary", None)               # Holds the generated document summary string
ss.setdefault("history", [])                 # List of chat turns: [{"q": ..., "a": ..., "sources": ...}]
ss.setdefault("suggested_questions", [])     # List of AI-generated starter questions
ss.setdefault("pending_question", None)      # Question clicked from suggestion chips

# ==============================================================================
# Sidebar: Document Upload & Indexing Flow
# ==============================================================================
with st.sidebar:
    st.image("assets/astra_logo.jpeg", width="stretch")
    st.divider()
    st.header("📂 Document")
    
    uploaded_files = st.file_uploader("Upload or drag & drop PDFs", type=["pdf"], accept_multiple_files=True)
    
    # Trigger processing only when the user clicks 'Process document'
    if uploaded_files and st.button("⚡ Process documents", type="primary", use_container_width=True):
        try:
            # 1. Chunk & embed the PDF locally
            with st.spinner("Reading, chunking and embedding…"):
                docs = [(f.getvalue(), f.name) for f in uploaded_files]
                ss.index = DocIndex(docs)
                # Reset old conversation history when a new document is loaded
                ss.history = []
                ss.summary = None
                ss.suggested_questions = []

            # 2. Automatically generate a concise overview summary
            with st.spinner("Summarizing…"):
                ss.summary = summarize(ss.index)

            # 3. Brainstorm questions to help the user start exploring
            with st.spinner("Generating suggested questions…"):
                ss.suggested_questions = suggest_questions(ss.index)

        except DocError as e:
            st.error(str(e))

    # If a document is currently active, show document statistics and management
    if ss.index:
        st.success(f"**{ss.index.name}**  \n{ss.index.pages} pages · {len(ss.index.chunks)} chunks")
        if st.button("🗑️ Clear chat", use_container_width=True):
            ss.history = []
            st.rerun()

# ==============================================================================
# Empty State: Landing Guide (shown when no document is uploaded yet)
# ==============================================================================
if not ss.index:
    st.html("""
    <div style="max-width:860px; margin:40px auto 0;">
      <p style="font-family:'Space Grotesk',sans-serif; font-size:0.75rem; font-weight:700;
                color:#B8860B; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:20px;">
        WHAT ASTRA INTEL DOES
      </p>
      <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px;">

        <div style="background:#22282E; border:1px solid rgba(184,134,11,0.25);
                    border-radius:12px; padding:20px 18px;">
          <div style="font-size:1.5rem; margin-bottom:10px;">📄</div>
          <div style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                      color:#B8860B; font-size:0.9rem; margin-bottom:6px;">1 · Upload</div>
          <div style="font-family:'Inter',sans-serif; font-size:0.82rem;
                      color:#9AA0A8; line-height:1.55;">
            Upload any PDFs or scanned documents via the sidebar.
          </div>
        </div>

        <div style="background:#22282E; border:1px solid rgba(184,134,11,0.25);
                    border-radius:12px; padding:20px 18px;">
          <div style="font-size:1.5rem; margin-bottom:10px;">⚙️</div>
          <div style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                      color:#B8860B; font-size:0.9rem; margin-bottom:6px;">2 · Extract &amp; Process</div>
          <div style="font-family:'Inter',sans-serif; font-size:0.82rem;
                      color:#9AA0A8; line-height:1.55;">
            Content is extracted, chunked and embedded locally — no data leaves your machine.
          </div>
        </div>

        <div style="background:#22282E; border:1px solid rgba(184,134,11,0.25);
                    border-radius:12px; padding:20px 18px;">
          <div style="font-size:1.5rem; margin-bottom:10px;">📋</div>
          <div style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                      color:#B8860B; font-size:0.9rem; margin-bottom:6px;">3 · Summarise</div>
          <div style="font-family:'Inter',sans-serif; font-size:0.82rem;
                      color:#9AA0A8; line-height:1.55;">
            A concise 5–7 sentence summary is generated automatically from the document.
          </div>
        </div>

        <div style="background:#22282E; border:1px solid rgba(184,134,11,0.25);
                    border-radius:12px; padding:20px 18px;">
          <div style="font-size:1.5rem; margin-bottom:10px;">💬</div>
          <div style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                      color:#B8860B; font-size:0.9rem; margin-bottom:6px;">4 · Ask Questions</div>
          <div style="font-family:'Inter',sans-serif; font-size:0.82rem;
                      color:#9AA0A8; line-height:1.55;">
            Ask anything in natural language using the chat bar at the bottom of the screen.
          </div>
        </div>

        <div style="background:#22282E; border:1px solid rgba(184,134,11,0.25);
                    border-radius:12px; padding:20px 18px;">
          <div style="font-size:1.5rem; margin-bottom:10px;">🎯</div>
          <div style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                      color:#B8860B; font-size:0.9rem; margin-bottom:6px;">5 · Grounded Answers</div>
          <div style="font-family:'Inter',sans-serif; font-size:0.82rem;
                      color:#9AA0A8; line-height:1.55;">
            Every answer is drawn strictly from your document — never hallucinated or invented.
          </div>
        </div>

        <div style="background:#22282E; border:1px solid rgba(184,134,11,0.25);
                    border-radius:12px; padding:20px 18px;">
          <div style="font-size:1.5rem; margin-bottom:10px;">📍</div>
          <div style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                      color:#B8860B; font-size:0.9rem; margin-bottom:6px;">6 · Sources &amp; Pages</div>
          <div style="font-family:'Inter',sans-serif; font-size:0.82rem;
                      color:#9AA0A8; line-height:1.55;">
            Every answer cites the exact page number and passage it was retrieved from.
          </div>
        </div>

      </div>
      <p style="font-family:'Inter',sans-serif; font-size:0.8rem; color:#4A5060;
                margin-top:28px; text-align:center;">
        ⬅️ &nbsp;Upload PDFs in the sidebar and click
        <strong style="color:#B8860B;">⚡ Process document</strong> to begin.
      </p>
    </div>
    """)
    # Stop further execution until the user uploads and processes a PDF
    st.stop()

# ==============================================================================
# Document Summary Section
# ==============================================================================
if ss.summary:
    with st.expander("📄 Document Summary", expanded=True):
        st.write(ss.summary)

# ==============================================================================
# Suggested Question Chips
# ==============================================================================
# Only show suggested question chips if the user hasn't started chatting yet
if ss.suggested_questions and not ss.history:
    st.html("""<p style="
      font-family:'Space Grotesk',sans-serif; font-size:0.8rem; font-weight:600;
      color:#B8860B; letter-spacing:0.08em; text-transform:uppercase; margin:16px 0 8px;
    ">💡 Suggested questions — click to ask</p>""")
    
    cols = st.columns(len(ss.suggested_questions))
    for i, sq in enumerate(ss.suggested_questions):
        if cols[i].button(sq, key=f"sq_{i}", use_container_width=True):
            ss.pending_question = sq
            st.rerun()

# ==============================================================================
# Chat Conversation History
# ==============================================================================
# Renders all past Q&A turns stored in st.session_state.history
for turn in ss.history:
    with st.chat_message("user"):
        st.write(turn["q"])
        
    with st.chat_message("assistant"):
        st.write(turn["a"])
        
        top_score = turn["sources"][0]["score"] if turn["sources"] else 0
        if turn["supported"]:
            st.caption("✅ Supported by document")
        else:
            st.caption(f"⚠️ Not supported by document · top similarity: {top_score:.3f} (threshold 0.10)")
            
        # Expandable inspection showing the exact passages used to answer
        with st.expander("📍 Sources — relevant passages & pages"):
            for s in turn["sources"]:
                # Escape HTML special chars to prevent formatting glitches or XSS
                safe_text = html.escape(s['text'][:420]) + ('&hellip;' if len(s['text']) > 420 else '')
                st.html(f"""
                <div style="display:flex;align-items:flex-start;gap:12px;
                            background:#1B1F23;border:1px solid rgba(184,134,11,0.2);
                            border-radius:8px;padding:12px 14px;margin-bottom:8px;">
                  <div style="background:#B8860B;color:#1B1F23;
                              font-family:'Space Grotesk',sans-serif;font-weight:700;
                              font-size:0.72rem;padding:4px 10px;border-radius:20px;
                              white-space:nowrap;flex-shrink:0;margin-top:2px;">
                    PAGE {s['page']}
                  </div>
                  <div style="font-family:'Inter',sans-serif;font-size:0.82rem;
                              color:#C8CDD4;line-height:1.6;">
                    {safe_text}
                  </div>
                </div>
                """)
                st.caption(f"Similarity: `{s['score']:.3f}`")

# ==============================================================================
# Chat Input & Answer Generation Flow
# ==============================================================================
user_input = st.chat_input("Ask something about the document…")

# If the user clicked a suggestion chip instead of typing, prioritize it
if ss.pending_question:
    user_input = ss.pending_question
    ss.pending_question = None

if user_input:
    try:
        with st.spinner("Searching and answering…"):
            # Provide recent conversation turns (question, answer) for context
            past_turns = [(t["q"], t["a"]) for t in ss.history]
            result = answer(ss.index, user_input, past_turns)

        # Append new exchange to history and rerun so it renders immediately
        ss.history.append({
            "q": user_input,
            "a": result["answer"],
            "supported": result["supported"],
            "sources": result["sources"],
        })
        st.rerun()
    except DocError as e:
        st.error(str(e))


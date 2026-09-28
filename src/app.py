"""ASTRA INTEL - Streamlit UI. Run: streamlit run app.py"""
import streamlit as st

from rag import DocError, DocIndex, answer, summarize, suggest_questions

st.set_page_config(page_title="ASTRA INTEL", page_icon="🛰️", layout="wide")

# ── Custom CSS via st.html() (bypasses Streamlit's sanitiser) ─────────────────
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
  /* Global font & background */
  html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
  .stApp { background-color: #000000 !important; }
  header[data-testid="stHeader"] { background-color: transparent !important; }

  /* Headings */
  h1, h2, h3, h4 {
    font-family: 'Space Grotesk', Arial, sans-serif !important;
    letter-spacing: 0.04em;
  }

  /* Sidebar deeper background */
  [data-testid="stSidebar"] { background-color: #000000 !important; border-right: 1px solid rgba(184,134,11,0.2) !important; }

  /* Primary button gold gradient */
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

  /* Secondary buttons (suggestion chips) */
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

  /* Chat bubbles */
  [data-testid="stChatMessage"] {
    background-color: #22282E !important;
    border: 1px solid rgba(184,134,11,0.15) !important;
    border-radius: 12px !important;
    padding: 12px !important;
  }

  /* Chat input */
  [data-testid="stChatInputTextArea"] {
    background-color: #22282E !important;
    border: 1px solid #B8860B55 !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
  }

  /* Expander header */
  [data-testid="stExpander"] summary {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: #B8860B !important;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: #1B1F23; }
  ::-webkit-scrollbar-thumb { background: #B8860B66; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #B8860B; }
</style>
""")

# ── Branded hero header ───────────────────────────────────────────────────────
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

# ── Session state ─────────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("index", None)
ss.setdefault("summary", None)
ss.setdefault("history", [])
ss.setdefault("suggested_questions", [])
ss.setdefault("pending_question", None)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/astra_logo.jpeg", width="stretch")
    st.divider()
    st.header("📂 Document")
    f = st.file_uploader("Upload a PDF", type=["pdf"])
    if f and st.button("⚡ Process document", type="primary", use_container_width=True):
        try:
            with st.spinner("Reading, chunking and embedding…"):
                ss.index = DocIndex(f.getvalue(), f.name)
                ss.history = []
                ss.summary = None
                ss.suggested_questions = []
            with st.spinner("Summarizing…"):
                ss.summary = summarize(ss.index)
            with st.spinner("Generating suggested questions…"):
                ss.suggested_questions = suggest_questions(ss.index)
        except DocError as e:
            st.error(str(e))

    if ss.index:
        st.success(f"**{ss.index.name}**  \n{len(ss.index.pages)} pages · {len(ss.index.chunks)} chunks")
        if st.button("🗑️ Clear chat", use_container_width=True):
            ss.history = []
            st.rerun()

# ── Guard ─────────────────────────────────────────────────────────────────────
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
            Upload any PDF or scanned document via the sidebar.
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
        ⬅️ &nbsp;Upload a PDF in the sidebar and click
        <strong style="color:#B8860B;">⚡ Process document</strong> to begin.
      </p>
    </div>
    """)
    st.stop()

# ── Summary ───────────────────────────────────────────────────────────────────
if ss.summary:
    with st.expander("📄 Document Summary", expanded=True):
        st.write(ss.summary)

# ── Suggestion chips ──────────────────────────────────────────────────────────
if ss.suggested_questions and not ss.history:
    st.html("""<p style="
      font-family:'Space Grotesk',sans-serif; font-size:0.8rem; font-weight:600;
      color:#B8860B; letter-spacing:0.08em; text-transform:uppercase; margin:16px 0 8px;
    ">💡 Suggested questions — click to ask</p>""")
    cols = st.columns(len(ss.suggested_questions))
    for i, sq in enumerate(ss.suggested_questions):
        if cols[i].button(sq, key=f"sq_{i}"):
            ss.pending_question = sq
            st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
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
        with st.expander("📍 Sources — relevant passages & pages"):
            for s in turn["sources"]:
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
                    {s['text'][:420]}{'&hellip;' if len(s['text']) > 420 else ''}
                  </div>
                </div>
                """)
                st.caption(f"Similarity: `{s['score']:.3f}`")

# ── Chat input ────────────────────────────────────────────────────────────────
q = st.chat_input("Ask something about the document…")

if ss.pending_question:
    q = ss.pending_question
    ss.pending_question = None

if q:
    try:
        with st.spinner("Searching and answering…"):
            past = [(t["q"], t["a"]) for t in ss.history]
            res = answer(ss.index, q, past)
        ss.history.append({"q": q, "a": res["answer"], "supported": res["supported"], "sources": res["sources"]})
        st.rerun()
    except DocError as e:
        st.error(str(e))

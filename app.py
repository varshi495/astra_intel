"""ASTRA INTEL - Streamlit UI. Run: streamlit run app.py"""
import streamlit as st

from rag import DocError, DocIndex, answer, summarize, suggest_questions

st.set_page_config(page_title="ASTRA INTEL", page_icon="🛰️", layout="wide")
st.title(" ASTRA INTEL")
st.caption("Defence document intelligence: upload a PDF, get a summary, ask questions. Answers come only from your document.")

ss = st.session_state
ss.setdefault("index", None)
ss.setdefault("summary", None)
ss.setdefault("history", [])          # list of dicts: q, a, supported, sources
ss.setdefault("suggested_questions", [])  # LLM-generated, never hardcoded
ss.setdefault("pending_question", None)   # set when a suggestion chip is clicked

with st.sidebar:
    st.header("Document")
    f = st.file_uploader("Upload a PDF", type=["pdf"])
    if f and st.button("Process document", type="primary"):
        try:
            with st.spinner("Reading, chunking and embedding..."):
                ss.index = DocIndex(f.getvalue(), f.name)
                ss.history = []
                ss.summary = None
                ss.suggested_questions = []
            with st.spinner("Summarizing..."):
                ss.summary = summarize(ss.index)
            with st.spinner("Generating suggested questions..."):
                ss.suggested_questions = suggest_questions(ss.index)
        except DocError as e:
            st.error(str(e))
    if ss.index:
        st.success(f"{ss.index.name}: {len(ss.index.pages)} pages, {len(ss.index.chunks)} chunks")
        if st.button("Clear chat"):
            ss.history = []

if not ss.index:
    st.info("Upload a PDF in the sidebar and click **Process document** to begin.")
    st.stop()

if ss.summary:
    with st.expander("📄 Summary", expanded=True):
        st.write(ss.summary)

# Dynamically generated suggestion chips (cleared once the user starts chatting)
if ss.suggested_questions and not ss.history:
    st.markdown("**💡 Suggested questions — click to ask:**")
    cols = st.columns(len(ss.suggested_questions))
    for i, sq in enumerate(ss.suggested_questions):
        if cols[i].button(sq, key=f"sq_{i}", use_container_width=True):
            ss.pending_question = sq
            st.rerun()

# chat history
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
        with st.expander("Sources (retrieved passages)"):
            for s in turn["sources"]:
                st.markdown(f"**Page {s['page']}** · similarity `{s['score']:.3f}`")
                st.write(s["text"])

q = st.chat_input("Ask something about the document...")

# A suggestion chip was clicked — treat it exactly like a typed question
if ss.pending_question:
    q = ss.pending_question
    ss.pending_question = None

if q:
    try:
        with st.spinner("Searching and answering..."):
            past = [(t["q"], t["a"]) for t in ss.history]
            res = answer(ss.index, q, past)
        ss.history.append({"q": q, "a": res["answer"], "supported": res["supported"], "sources": res["sources"]})
        st.rerun()
    except DocError as e:
        st.error(str(e))

"""ASTRA INTEL - Streamlit UI. Run: streamlit run app.py"""
import streamlit as st

from rag import DocError, DocIndex, answer, summarize

st.set_page_config(page_title="ASTRA INTEL", page_icon="🛰️", layout="wide")
st.title(" ASTRA INTEL")
st.caption("Defence document intelligence: upload a PDF, get a summary, ask questions. Answers come only from your document.")

ss = st.session_state
ss.setdefault("index", None)
ss.setdefault("summary", None)
ss.setdefault("history", [])  # list of dicts: q, a, supported, sources

with st.sidebar:
    st.header("Document")
    f = st.file_uploader("Upload a PDF", type=["pdf"])
    if f and st.button("Process document", type="primary"):
        try:
            with st.spinner("Reading, chunking and embedding..."):
                ss.index = DocIndex(f.getvalue(), f.name)
                ss.history = []
                ss.summary = None
            with st.spinner("Summarizing..."):
                ss.summary = summarize(ss.index)
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

# chat history
for turn in ss.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        st.caption("✅ Supported by document" if turn["supported"] else "⚠️ Not supported by document")
        with st.expander("Sources (retrieved passages)"):
            for s in turn["sources"]:
                st.markdown(f"**Page {s['page']}** · similarity {s['score']:.2f}")
                st.write(s["text"])

q = st.chat_input("Ask something about the document...")
if q:
    try:
        with st.spinner("Searching and answering..."):
            past = [(t["q"], t["a"]) for t in ss.history]
            res = answer(ss.index, q, past)
        ss.history.append({"q": q, "a": res["answer"], "supported": res["supported"], "sources": res["sources"]})
        st.rerun()
    except DocError as e:
        st.error(str(e))

import os
import tempfile

import streamlit as st

from ITR_project import load_document, answer_question

st.set_page_config(page_title="Product Review Analyzer", page_icon="📄", layout="centered")

st.title("📄 Product Review Analyzer")
st.caption("Upload a .txt document, then ask questions about it. Answers are generated only from the document's content.")

# --- session state ---
if "stored_chunks" not in st.session_state:
    st.session_state.stored_chunks = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- sidebar: upload / load document ---
with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file is not None:
        if st.session_state.doc_name != uploaded_file.name:
            with st.spinner("Loading and embedding document..."):
                # load_document expects a filepath, so write to a temp file first
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".txt", delete=False
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                stored = load_document(tmp_path)
                os.unlink(tmp_path)

            if stored is None:
                st.error("Could not load document (empty or unreadable).")
            else:
                st.session_state.stored_chunks = stored
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []
                st.success(f"Loaded '{uploaded_file.name}' — {len(stored)} chunks indexed.")

    if st.session_state.doc_name:
        st.info(f"Active document: **{st.session_state.doc_name}**")
        if st.button("Clear document"):
            st.session_state.stored_chunks = None
            st.session_state.doc_name = None
            st.session_state.messages = []
            st.rerun()

# --- main: chat interface ---
if st.session_state.stored_chunks is None:
    st.info("Upload a document in the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(question, st.session_state.stored_chunks)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
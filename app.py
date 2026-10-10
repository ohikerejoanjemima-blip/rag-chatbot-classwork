"""Run with: python -m streamlit run app.py"""
import hashlib
import json
import os
from pathlib import Path
import streamlit as st
from conversation import ConversationalPDFRAG
from documents import index_pdf
from resources import ChatModel, EmbeddingResource

st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="wide")
ROOT = Path(__file__).resolve().parent
MODEL_ID = os.environ.get("RAG_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")


@st.cache_resource(show_spinner=False)
def load_chat_model(model_id):
    return ChatModel(model_id)


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return EmbeddingResource()


def clear_document():
    old_index = st.session_state.get("index")
    if old_index is not None:
        try:
            old_index.close()
        except Exception:
            pass
    st.session_state.update(index=None, bot=None, messages=[], document_id=None)


def show_sources(result):
    if not result:
        return
    with st.expander("Sources and search question"):
        st.caption("Search question: " + result["search_question"])
        st.caption("Retrieved excerpts help you verify the answer; citation accuracy is not guaranteed.")
        for source in result["sources"]:
            st.markdown(f'**[{source["id"]}] {source["file"]} · page {source["page"]}**')
            st.text(source["text"])
        if result["invalid_citations"]:
            st.warning("The answer references an unknown source: " + ", ".join(result["invalid_citations"]))


for key, default in {"index": None, "bot": None, "messages": [], "document_id": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("Chat with your PDF")
st.write("Ask a question, then follow up naturally. Your chatbot remembers the recent conversation.")

with st.sidebar:
    st.header("Your document")
    mode = st.radio("Choose a PDF", ["Sample handbook", "Upload my PDF"])
    data = None
    filename = None
    if mode == "Sample handbook":
        sample = ROOT / "sample_data" / "detailed_rag_test_handbook.pdf"
        if sample.is_file():
            data, filename = sample.read_bytes(), sample.name
            st.download_button("Download sample PDF", data, filename, mime="application/pdf")
        else:
            st.info("Select 'Upload my PDF' to add a document.")
    else:
        upload = st.file_uploader("Upload one text-based PDF", type=["pdf"])
        if upload is not None:
            data, filename = upload.getvalue(), upload.name

    selected_id = hashlib.sha256(data + filename.encode()).hexdigest() if data is not None else None
    if st.session_state.document_id is not None and selected_id != st.session_state.document_id:
        clear_document()

    process = st.button("Load document", type="primary", disabled=data is None)
    if process:
        clear_document()
        new_index = None
        try:
            with st.spinner("Reading PDF and building its search index…"):
                new_index = index_pdf(data, filename, load_embeddings())
            with st.spinner("Loading the chat model. First use downloads model files…"):
                generator = load_chat_model(MODEL_ID)
            st.session_state.update(
                index=new_index,
                bot=ConversationalPDFRAG(new_index, generator.generate, generator.tokenizer),
                messages=[], document_id=selected_id,
            )
        except Exception as exc:
            if new_index is not None:
                new_index.close()
            st.error(f"Could not load this document: {exc}")
            st.caption("For a model download error, check your internet connection and available memory.")

    if st.session_state.index is not None:
        index = st.session_state.index
        st.success(f"Ready: {filename}")
        st.caption(f"{index.page_count} pages · {index.chunk_count} searchable passages")
        if st.button("Clear conversation"):
            st.session_state.bot.reset()
            st.session_state.messages = []
            st.rerun()
        retained = len(st.session_state.bot.history) // 2
        st.caption(f"Memory: {retained} recent turns retained (maximum six; token limits also apply).")
    st.caption("Memory lasts for this browser session. Changing PDFs clears it.")

if st.session_state.bot is None:
    st.info("Choose a PDF in the sidebar and click Load document.")
    st.markdown("**Try with the sample:** What is my annual learning budget? → If I spend NGN 120,000, how much remains? → Can I carry that balance into next year?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("result"))

question = st.chat_input("Ask about your PDF…", disabled=st.session_state.bot is None, max_chars=2000)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching your PDF and writing an answer…"):
                result = st.session_state.bot.ask(question)
            st.markdown(result["answer"])
            show_sources(result)
        st.session_state.messages.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": result["answer"], "result": result},
        ])
        st.rerun()
    except Exception as exc:
        st.error(f"Could not answer: {exc}")
        st.caption("The failed turn was not added to conversation memory. Try a shorter question.")

if st.session_state.messages:
    with st.sidebar:
        st.download_button(
            "Download conversation",
            json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
            "conversation.json", mime="application/json",
        )

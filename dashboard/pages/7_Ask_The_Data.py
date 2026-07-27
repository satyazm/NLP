import os

import streamlit as st

from data_loader import data_is_ready, load_conversation_features

st.title("Ask the Data")
st.caption("A RAG assistant over your historical support conversations — retrieves the "
           "most relevant transcripts with FAISS, then asks Gemini to answer grounded in them.")

if not data_is_ready():
    st.warning("Run the pipeline first: `python -m src.pipeline --input data/raw/sample_twcs.csv`")
    st.stop()

if not os.environ.get("GEMINI_API_KEY"):
    st.warning("Set GEMINI_API_KEY in your .env to use this page.")
    st.stop()

index_ready = os.path.exists("vectorstore/conversations.faiss")

if not index_ready:
    st.info("No vector index found yet.")
    if st.button("Build index now"):
        from src.llm.rag_chatbot import build_index
        with st.spinner("Embedding conversations and building the FAISS index..."):
            build_index(load_conversation_features())
        st.success("Index built. You can ask questions now.")
        st.rerun()
    st.stop()

query = st.text_input("Ask a question", placeholder="What are customers complaining about most?")

if query:
    from src.llm.rag_chatbot import ask_question
    with st.spinner("Retrieving conversations and asking Gemini..."):
        answer = ask_question(query)
    st.markdown(answer)

st.divider()
st.caption("Example questions: \"Summarize refund-related conversations\", "
           "\"Which products had the worst sentiment?\", \"Show payment failure complaints.\"")

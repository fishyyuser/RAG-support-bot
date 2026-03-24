import streamlit as st
from rag_pipeline import RAGPipeline

# Initialize RAG once
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

rag = load_pipeline()

st.set_page_config(page_title="FastAPI Docs Chatbot", layout="wide")

st.title("📄 FastAPI Documentation Chatbot")
st.caption("Ask questions about FastAPI docs using RAG")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag.query(prompt)
            st.markdown(response)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")
st.title("🤖 RAG Assistant")
st.caption("Upload documents and chat with them using AI")

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Upload Documents")

    # PDF upload
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file and st.button("Index PDF"):
        with st.spinner("Indexing PDF..."):
            res = requests.post(f"{API_URL}/upload/pdf", files={"file": pdf_file})
            st.success(res.json()["message"])

    # Word upload
    word_file = st.file_uploader("Upload Word Doc", type=["docx"])
    if word_file and st.button("Index Word Doc"):
        with st.spinner("Indexing Word doc..."):
            res = requests.post(f"{API_URL}/upload/word", files={"file": word_file})
            st.success(res.json()["message"])

    # URL input
    url = st.text_input("Or enter a URL")
    if url and st.button("Index URL"):
        with st.spinner("Indexing URL..."):
            res = requests.post(f"{API_URL}/upload/url", data={"url": url})
            st.success(res.json()["message"])

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Build chat history string
    history = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in st.session_state.messages[:-1]
    )

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "chat_history": history}
            )
            response_data = res.json()
            st.write("Debug response:", response_data)  # temporary debug line
            answer = response_data.get("answer", "Error: " + str(response_data))
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

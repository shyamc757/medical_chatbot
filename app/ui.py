import os
import streamlit as st
from tempfile import TemporaryDirectory
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from ingest import ingest_reports
from rag_agent import get_answer, retriever

st.set_page_config(page_title="Medical Report Assistant", layout="wide")
st.title("🩺 Medical Report Assistant")

# ---- Sidebar File Upload ----
st.sidebar.title("📂 Upload Medical Reports")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or image reports", type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# ---- Track Previously Ingested Files ----
if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = set()

if uploaded_files:
    with TemporaryDirectory() as temp_dir:
        new_files = []
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_filenames:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                new_files.append(file.name)

        if new_files:
            st.sidebar.info(f"🔄 Ingesting {len(new_files)} new file(s)...")
            ingest_reports(temp_dir)
            st.session_state.uploaded_filenames.update(new_files)
            st.sidebar.success("✅ New files ingested successfully.")
        else:
            st.sidebar.info("📁 All selected files were already ingested.")

# ---- Chat State Setup ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# ---- Sidebar Clear Chat Option ----
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.memory.clear()
    st.rerun()

st.markdown("### 💬 Chat with Your Medical Reports")

# ---- Chat Input ----
user_input = st.chat_input("Ask a medical question...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.memory.chat_memory.add_user_message(user_input)

    with st.spinner("💬 Thinking..."):
        history = st.session_state.memory.load_memory_variables({}).get("history", "")
        question = f"{history}\n\n{user_input}"
        answer = get_answer(question)
        docs = retriever.get_relevant_documents(user_input)

        # Unique filenames in order of relevance
        filenames = []
        seen = set()
        for doc in docs:
            fname = doc.metadata.get("filename", "Unknown")
            if fname not in seen:
                filenames.append(fname)
                seen.add(fname)

        if filenames:
            source_section = "\n\n📁 **Sources**"
            full_response = f"{answer}{source_section}"
        else:
            full_response = answer

    st.session_state.chat_history.append(AIMessage(content=full_response))
    st.session_state.memory.chat_memory.add_ai_message(full_response)

# ---- Chat Display ----
scroll_placeholder = st.empty()
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").markdown(msg.content, unsafe_allow_html=True)
        if "Sources" in msg.content:
            with st.expander("📂 Click to view sources"):
                for doc in retriever.get_relevant_documents(msg.content):
                    fname = doc.metadata.get("filename", "Unknown")
                    st.markdown(f"- `{fname}`")

with scroll_placeholder.container():
    st.markdown("<div style='height: 1px;'></div>", unsafe_allow_html=True)
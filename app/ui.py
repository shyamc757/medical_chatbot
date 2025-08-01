import os
import streamlit as st
from tempfile import TemporaryDirectory

from ingest import ingest_reports
from rag_agent import get_answer, retriever

# ---- Page Config ----
st.set_page_config(page_title="Medical Report Assistant", layout="centered")
st.title("🩺 Medical Report Assistant")
st.markdown("Upload medical reports and ask questions about them.")

# ---- File Upload ----
uploaded_files = st.file_uploader(
    "📂 Upload PDF or image reports", type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if "uploaded_filenames" not in st.session_state:
    st.session_state["uploaded_filenames"] = set()

if uploaded_files:
    with TemporaryDirectory() as temp_dir:
        new_files = []

        for file in uploaded_files:
            if file.name not in st.session_state["uploaded_filenames"]:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

                new_files.append(file.name)

        if new_files:
            st.info(f"🔄 Processing and ingesting {len(new_files)} new file(s)...")
            ingest_reports(temp_dir)
            st.session_state["uploaded_filenames"].update(new_files)
            st.success("✅ New files ingested successfully.")
        else:
            st.info("📁 All selected files were already ingested.")


st.divider()

# ---- Ask a Question ----
user_query = st.text_input("🔍 Ask a medical question:", placeholder="e.g., What is the patient's glucose level?")

if user_query:
    with st.spinner("💬 Thinking..."):
        answer = get_answer(user_query)

        st.markdown("### ✅ Answer")
        st.success(answer)

        docs = retriever.get_relevant_documents(user_query)

        source_files = sorted({doc.metadata.get("filename", "Unknown") for doc in docs})
        if source_files:
            st.markdown("**📁 Based on files:**")
            for f_name in source_files:
                st.markdown(f"- `{f_name}`")

import streamlit as st
from pathlib import Path

from backend.pdf_loader import save_uploaded_file, load_and_chunk_pdf
from backend.vector_store import build_vector_store
from backend.classifier_agent import classify_document
from backend.summarizer_agent import summarize_document
from backend.qa_agent import answer_question
from backend.quiz_agent import generate_quiz_from_query
from backend.config import CHROMA_DIR

st.set_page_config(
    page_title="InsightPDF – Agentic RAG",
    page_icon="📄",
    layout="wide",
)

COFFEE_BROWN_BG = """
<style>
body {
    background: radial-gradient(circle at top left, #967259 0, #4b3426 35%, #2c1c13 100%);
    color: #f5f0e8;
}
section.main > div {
    background: rgba(20, 13, 8, 0.90);
    border-radius: 18px;
    padding: 20px 24px;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.55);
}
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #3b2619, #1e130d);
}
.stButton>button {
    background-color: #8b5a2b;
    color: #fdf7ee;
    border: 1px solid #1b130e;
    border-radius: 999px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #a96e3d;
    border-color: #f1d3b4;
}
.stTextInput>div>div>input, .stTextArea>div>textarea {
    background-color: #2b1a12;
    color: #f8eee3;
    border-radius: 10px;
}
</style>
"""

st.markdown(COFFEE_BROWN_BG, unsafe_allow_html=True)

if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "doc_type" not in st.session_state:
    st.session_state.doc_type = None
if "summary" not in st.session_state:
    st.session_state.summary = None

def main():
    st.title("InsightPDF – Agentic RAG Document Analyzer")
    st.caption("Upload a PDF → classify → summarize → chat & quiz, all powered by local RAG.")

    if st.session_state.file_path is None:
        render_upload_page()
    else:
        render_summary_and_chat_page()

def render_upload_page():
    st.subheader("Step 1 – Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.write(f"Selected file: **{uploaded_file.name}**")

        if st.button("Process & Summarize"):
            with st.spinner("Reading, chunking, embedding, and summarizing your document..."):
                # Save file
                file_path = save_uploaded_file(uploaded_file)
                st.session_state.file_path = file_path

                chunks = load_and_chunk_pdf(file_path)

                # Build vector store (Chroma)
                build_vector_store(chunks)

                # Get plain text for classification & summarization
                full_text = "\n\n".join([c.page_content for c in chunks])

                # Classify document type
                doc_type = classify_document(full_text)
                st.session_state.doc_type = doc_type

                # Summarize document
                summary = summarize_document(full_text, doc_type)
                st.session_state.summary = summary

            st.success("Document processed! Opening summary & chat…")
            st.rerun()

def render_summary_and_chat_page():
    st.subheader("Step 2 – Structured Summary & Chatbot")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Document Summary")
        st.markdown(f"**Detected type:** {st.session_state.doc_type}")
        st.markdown("---")
        st.markdown(st.session_state.summary)

    with col2:
        st.markdown("### Chat with your PDF")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.chat_message("user").markdown(msg)
            else:
                st.chat_message("assistant").markdown(msg)

        # Chat input
        user_input = st.chat_input("Ask anything about this PDF… or type 'quiz' to generate a quiz.")
        if user_input:
            st.session_state.chat_history.append(("user", user_input))

            with st.chat_message("assistant"):
                with st.spinner("Thinking with RAG..."):
                    if user_input.strip().lower().startswith("quiz"):
                        quiz = generate_quiz_from_query()
                        answer_text = quiz
                    else:
                        answer, sources = answer_question(user_input)
                        answer_text = answer

                    st.markdown(answer_text)
                    st.session_state.chat_history.append(("assistant", answer_text))

    st.markdown("---")
    if st.button("Start over with a new PDF"):
        st.session_state.file_path = None
        st.session_state.doc_type = None
        st.session_state.summary = None
        st.session_state.chat_history = []

if __name__ == "__main__":
    main()

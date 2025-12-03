# InsightPDF – Agentic RAG PDF Analyzer

InsightPDF is a local, agentic RAG-based web app that lets you upload any PDF, automatically classify it, generate a structured summary, chat with it, and even generate quizzes – all powered by open models and a vector database.

- Frontend: Streamlit
- Backend: Python + LangChain-style agents
- Vector DB: ChromaDB
- Models: Local LLM via Ollama + HuggingFace embeddings

---

## Features

- **PDF Upload & Processing**
  - Upload any PDF.
  - Text is extracted, chunked intelligently, and embedded into a local Chroma vector store.

- **Document Type Classification**
  - Classifies PDFs into types such as:
    - Research Paper
    - Novel / Literature
    - Study Material / Notes
    - Technical Documentation
    - Business Report

- **Agentic Summarization**
  - Type-aware, structured summary:
    - Research papers: abstract, methods, dataset, results, limitations
    - Novels: plot, characters, conflicts, themes
    - Notes: key concepts, formulas, definitions
    - etc.
  - Uses an **agentic loop**: draft summary → critic → refined summary.

- **Agentic RAG Chatbot**
  - Ask questions about the document.
  - LLM rewrites the question, retrieves relevant chunks via Chroma, grades the answer, and can refine/retry if the first answer is weak.
  - Minimizes hallucinations by grounding answers in retrieved context.

- **Quiz Generator**
  - Type “quiz” in the chat to generate:
    - MCQs
    - True/False
    - Short-answer questions  
  - All questions are grounded in the document.

---

## Tech Stack

- **Python**
- **Streamlit** – frontend / UI
- **LangChain (core + community)** – LLM orchestration, prompts, RAG patterns
- **ChromaDB** – local vector database for embeddings
- **HuggingFace Embeddings** – sentence-transformer model (e.g. `all-MiniLM-L6-v2`)
- **Ollama** – local LLM runtime (e.g. `llama3`)
- **PyPDF / LangChain PDF loader** – PDF text extraction

---

## 🚀 Local Setup & Run

### 1. Clone the repository

```
git clone https://github.com/Roshh21/InsightPDF.git
cd InsightPDF
```

### 2. Create and activate a virtual environment

```
python -m venv .venv
source .venv/bin/activate    #Mac/Linux
.venv\Scripts\Activate    #Windows (PowerShell)
```

### 3. Install dependencies
```
pip install -r requirements.txt
```
### 4. Configure environment variables
Create a `.env` file in the project root:
```
OLLAMA_LLM_MODEL=llama3
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_COLLECTION=insightpdf_docs
```

You can change model names if you use a different local LLM or embedding model.

### 5. Install and run Ollama (for local LLM)

If you haven’t already:

1. Install Ollama from: https://ollama.com  
2. Pull the model you want (e.g. `llama3`):

```
ollama pull llama3
```
3. Start the Ollama server (in a separate terminal):
```
ollama serve
```
Leave this running; the app will connect to `localhost:11434`.

### 6. Run the Streamlit app
In your venv terminal, from the project root:
```
python -m streamlit run app/app.py
```


Open that URL in your browser.

---

## How to Use the App

1. **Upload a PDF**
   - On the first page, use the file uploader to select a PDF.
   - Click **“Process & Summarize”**.
   - The app will:
     - Save the file.
     - Extract and chunk the text.
     - Build a Chroma vector index (clearing old data for a fresh index).
     - Classify the document type.
     - Generate an agentic, structured summary.

2. **View Summary & Chat**
   - After processing, you’re redirected to the summary + chat page.
   - Left side:
     - Detected document type.
     - Structured summary.
   - Right side:
     - Chat interface to ask questions about the PDF.

3. **Ask Questions**
   - Type any question about the document.
   - The agent will:
     - Rewrite the query.
     - Retrieve relevant chunks from Chroma.
     - Grade the answer and refine the query if needed.
   - Answers are grounded in the document content.

4. **Generate a Quiz**
   - In the chat box, type something like:
     - `quiz`
     - `quiz me on this document`
   - The app will:
     - Use RAG to get context from the PDF.
     - Generate a mixed quiz (MCQs, T/F, short-answer) based on the document.

5. **Start Over with a New PDF**
   - Click **“Start over with a new PDF”**.
   - Streamlit session state is cleared.
   - On next upload, the Chroma collection is also cleared and rebuilt for the new document only.


---


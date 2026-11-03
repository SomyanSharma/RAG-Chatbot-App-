import streamlit as st
import requests
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -------------------------------
# 1. Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Mini RAG Chatbot",
    page_icon="📚"
)

st.title("📚 Mini RAG-Based AI Chatbot")
st.write("Upload a document and ask questions about its content.")


# -------------------------------
# 2. Load Embedding Model
# -------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


model = load_embedding_model()


# -------------------------------
# 3. Extract Document Text
# -------------------------------

def extract_text(uploaded_file):

    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)

        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    elif uploaded_file.name.lower().endswith(".txt"):
        return uploaded_file.getvalue().decode(
            "utf-8",
            errors="replace"
        )

    return ""


# -------------------------------
# 4. Divide Text into Chunks
# -------------------------------

def split_into_chunks(text, chunk_size=180, overlap=30):

    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]

        if chunk:
            chunks.append(" ".join(chunk))

    return chunks


# -------------------------------
# 5. Create FAISS Index
# -------------------------------

def create_index(chunks):

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


# -------------------------------
# 6. Retrieve Relevant Chunks
# -------------------------------

def retrieve_chunks(question, chunks, index, k=3):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    number_to_retrieve = min(k, len(chunks))

    scores, indices = index.search(
        question_embedding,
        number_to_retrieve
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx >= 0:
            results.append({
                "text": chunks[idx],
                "score": float(score)
            })

    return results


# -------------------------------
# 7. Generate Answer with Ollama
# -------------------------------

def generate_answer(question, context):

    prompt = f"""
You are a document-based AI assistant.

Answer the user's question using only the retrieved
document context below.

Rules:
- Do not invent facts or figures.
- Do not use outside knowledge as evidence.
- If the answer is missing, say:
  "I could not find the answer in the uploaded document."
- If the context provides only a partial answer,
  explain the limitation.
- Give a clear and concise answer.

Retrieved document context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["response"]


# -------------------------------
# 8. Upload Document
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT document",
    type=["pdf", "txt"]
)


if uploaded_file is not None:

    try:
        document_text = extract_text(uploaded_file)

        if not document_text.strip():
            st.error(
                "No readable text was found in the document."
            )
            st.stop()

        chunks = split_into_chunks(document_text)

        if not chunks:
            st.error("The document contains no usable text.")
            st.stop()

        index, embeddings = create_index(chunks)

        st.success(
            f"Document processed successfully: "
            f"{len(chunks)} chunks created."
        )

        question = st.text_input(
            "Enter your question"
        )

        if st.button("Get Answer"):

            if not question.strip():
                st.warning("Please enter a question.")

            else:
                with st.spinner("Searching document..."):

                    retrieved = retrieve_chunks(
                        question,
                        chunks,
                        index
                    )

                    context = "\n\n".join(
                        item["text"] for item in retrieved
                    )

                    try:
                        answer = generate_answer(
                            question,
                            context
                        )

                        st.subheader("Answer")
                        st.write(answer)

                        st.subheader("Retrieved Source Text")

                        for i, item in enumerate(
                            retrieved, start=1
                        ):
                            st.markdown(
                                f"**Source {i}** "
                                f"(similarity: "
                                f"{item['score']:.3f})"
                            )
                            st.write(item["text"])

                    except requests.RequestException as error:
                        st.error(
                            "Could not connect to Ollama. "
                            "Check that Ollama is running "
                            "and the model is installed."
                        )
                        st.caption(str(error))

    except Exception as error:
        st.error("An error occurred while processing the document.")
        st.caption(str(error))
from datetime import datetime
from zoneinfo import ZoneInfo   
def get_greeting():
    india_time = datetime.now(ZoneInfo("Asia/Kolkata"))
    hour = india_time.hour

    if 6 <= hour < 10:
        greeting = "Good Morning"
    elif 10 <= hour < 16:
        greeting = "Good Afternoon"
    elif 16 <= hour < 20:
        greeting = "Good Evening"
    else:
        greeting = "Good Night"

    return greeting, india_time.strftime("%I:%M %p")


greeting, current_time = get_greeting()
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Main application */
.stApp {
    background: #0B1120;
    color: #F1F5F9;
    font-family: 'Inter', sans-serif;
}

/* Main content */
section[data-testid="stMain"] {
    background: #0B1120;
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1150px;
}

/* All headings */
h1, h2, h3, h4 {
    color: #F8FAFC !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}

h1 {
    font-size: 40px !important;
}

h2 {
    font-size: 30px !important;
}

h3 {
    font-size: 23px !important;
}

/* Normal text */
p, label, li, .stMarkdown {
    color: #CBD5E1;
    font-size: 17px;
    line-height: 1.7;
    font-family: 'Inter', sans-serif;
}

/* Welcome box */
.welcome-box {
    background: linear-gradient(135deg, #172554, #1D4ED8);
    padding: 35px;
    border-radius: 18px;
    margin: 25px 0;
    border: 1px solid #3B82F6;
}

.welcome-box h1 {
    color: #FFFFFF !important;
    font-size: 38px !important;
}

.welcome-box p {
    color: #DBEAFE !important;
    font-size: 18px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #172033;
    padding: 20px;
    border: 1px dashed #3B82F6;
    border-radius: 14px;
}

[data-testid="stFileUploader"] section {
    background: #172033;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small {
    color: #E2E8F0 !important;
}

/* Input boxes */
.stTextInput input,
.stTextArea textarea {
    background: #172033 !important;
    color: #FFFFFF !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    font-size: 17px !important;
}

/* Buttons */
.stButton > button {
    background: #2563EB;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-size: 16px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #1D4ED8;
    color: white;
    border: 1px solid #60A5FA;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
}

section[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #172033;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 15px;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #172033;
    border: 1px solid #334155;
    border-radius: 12px;
}

[data-testid="stExpander"] summary {
    color: #F8FAFC !important;
}

/* Links */
a {
    color: #60A5FA !important;
}

/* Horizontal separators */
hr {
    border-color: #334155;
}
</style>
""", unsafe_allow_html=True)
st.markdown(f"""
<div class="welcome-box">
    <h1>{greeting}, Somyan! 👋</h1>
    <p>Welcome back! Your personal AI document assistant is ready.</p>
    <p>Indian Standard Time: {current_time}</p>
</div>
""", unsafe_allow_html=True)

st.subheader("Your AI Document Assistant")
st.write(
    "Upload a document, ask questions, and get intelligent answers "
    "based on the information in your document."
)
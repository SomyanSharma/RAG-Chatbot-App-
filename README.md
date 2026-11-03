# Mini RAG-Based AI Chatbot

A document-based question-answering chatbot built with Python and Streamlit. Upload a PDF or TXT document, ask questions about its contents, and view the retrieved source text used to help generate each answer.

## Features

- Upload a PDF or TXT document.
- Extract text from PDFs with PyPDF.
- Split extracted text into smaller chunks.
- Create text embeddings with Sentence Transformers (`all-MiniLM-L6-v2`).
- Retrieve relevant chunks using FAISS.
- Generate answers using a language model served by Ollama.
- Display retrieved source text to support answer checking.
- Use a custom dark theme and a time-based welcome greeting.

## Technology Stack

- **Python** — application logic
- **Streamlit** — web interface
- **PyPDF** — PDF text extraction
- **Sentence Transformers** — text embeddings
- **FAISS** — vector similarity search
- **Ollama** — local language-model runtime
- **Requests** — sends requests to the Ollama API
- **NumPy** — numerical arrays used for embeddings
- **CSS** — interface styling

## Project Structure

```text
RAG-Chatbot-App/
├── app.py
├── requirements.txt
└── README.md
```

Add any sample documents you want to keep in the repository, if appropriate. Do not commit private documents, credentials, or API keys.

## Local Setup

### 1. Install Python

Install a supported Python version and confirm it is available in your terminal:

```bash
python --version
```

### 2. Clone or open the repository

If you have not already cloned the project:

```bash
git clone https://github.com/SomyanSharma/RAG-Chatbot-App.git
cd RAG-Chatbot-App
```

If the project is already open in VS Code, open a terminal in the project folder instead.

### 3. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install and prepare Ollama

Install Ollama from its official website:

https://ollama.com/download

Pull the model used in the application:

```bash
ollama pull llama3.2:3b
```

The model name in `app.py` must match the model you have pulled. If your code uses a different model tag, pull that model or update the code accordingly.

Make sure Ollama is running locally. If required, start its service with:

```bash
ollama serve
```

If the service is already running, you may see a message that the port is in use; in that case, do not start a second instance.

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local address printed by Streamlit, commonly:

http://localhost:8501

The first run may need internet access to download the Sentence Transformers embedding model and the Ollama language model.

## How to Use

1. Start the application.
2. Upload a PDF or TXT file.
3. Wait for the document to be processed and indexed.
4. Enter a question about the document.
5. Review the generated answer and the retrieved source text.
6. Test a question whose answer is not present in the document to check how the application handles missing information.

## How It Works

1. **Document ingestion:** The user uploads a PDF or TXT file.
2. **Text extraction:** Text is extracted from the document.
3. **Chunking:** The text is split into smaller overlapping sections.
4. **Embedding:** Sentence Transformers converts chunks into numerical vectors.
5. **Indexing and retrieval:** FAISS indexes the vectors and retrieves chunks relevant to a question.
6. **Generation:** The question and retrieved context are sent to the Ollama model.
7. **Response:** Streamlit displays the answer and retrieved source text.

This approach is called **Retrieval-Augmented Generation (RAG)** because retrieval provides relevant external context to the language model before it generates a response.

## Deployment Notes

This project uses a locally running Ollama service at `http://localhost:11434` in the starter implementation. When deployed to Streamlit Community Cloud, `localhost` refers to the cloud server—not your personal computer. Therefore, the deployed application cannot reach Ollama on your computer using that address.

To deploy the full chatbot, you must configure an appropriately reachable model-serving endpoint and update the application to use it securely, or choose a hosting setup where the application and Ollama can communicate. Do not expose an unauthenticated Ollama endpoint publicly. Keep credentials and secrets out of source code and GitHub.

The Sentence Transformers model may also need to download during initial startup, depending on the deployment environment and its cache.

## Limitations

- Answers depend on the quality and text extraction of the uploaded document.
- Scanned PDFs containing images may not yield extractable text unless OCR is added.
- Retrieval may return irrelevant chunks, especially for vague questions.
- Prompt instructions can reduce unsupported answers but cannot guarantee that the model will never hallucinate.
- Displayed source chunks should be checked against the generated answer.
- The current implementation is intended as a small educational project, not a production-grade document security system.

## Suggested Test Questions

Use questions that are directly answered by your uploaded document, for example:

1. What is the main topic of the document?
2. What are the key concepts explained?
3. List the steps or stages described in the document.
4. What benefits or limitations does the document mention?
5. Ask for a specific fact that is **not** in the document. The chatbot should indicate that it could not find the information, though this behaviour is not guaranteed by the prompt alone.

## Academic Project

**Project title:** Mini RAG-Based AI Chatbot  
**Purpose:** Demonstrate document-grounded question answering using retrieval, embeddings, vector search, and a locally run language model.

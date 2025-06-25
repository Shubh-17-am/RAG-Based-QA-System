# 📚 RAG-Based QA System

This project implements a **Retrieval-Augmented Generation (RAG)** based Question-Answering system that leverages Large Language Models (LLMs) combined with document retrieval to answer questions based on custom data.

## 🚀 Features

- 📂 **Multi-format Document Ingestion**: Seamlessly supports `.txt`, `.pdf`, and `.docx` files using a unified interface.
- 🧠 **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with generative LLMs to answer context-aware questions.
- 🔍 **Contextual Embedding Search**: Uses vector similarity search (e.g., Chroma) to find the most relevant document chunks.
- 📎 **Persistent Vector Store**: Stores embeddings to avoid reprocessing and enable efficient querying.
- 🌐 **Interactive Web App (Streamlit)**: Allows users to upload documents and interact with the QA system via a clean and intuitive UI.
- ⚙️ **Modular Architecture**: Easily extensible for different embedding models, vector databases, and LLMs.
- 🔐 **Environment Configurable**: API keys and other settings are securely managed via `.env`.

## 🧠 Architecture

The system follows a modular RAG (Retrieval-Augmented Generation) pipeline with the following stages:

1. **Document Ingestion**
   - Accepts `.txt`, `.pdf`, and `.docx` formats.
   - Extracts raw text from documents and splits it into meaningful chunks.

2. **Text Embedding**
   - Converts text chunks into high-dimensional vector embeddings using models like `all-mpnet-base-v2` or any SentenceTransformer.
   - These embeddings capture semantic meaning and are used for similarity search.

3. **Vector Indexing (Retrieval)**
   - Uses Chroma or similar vector stores to index embeddings.
   - Enables fast and accurate similarity search during question answering.

4. **Query Handling**
   - User submits a natural language question.
   - The system embeds the query and retrieves top-k similar document chunks.

5. **Answer Generation**
   - Retrieved context is passed to an LLM (e.g., OpenAI, Anthropic, Groq, etc.).
   - The model generates a context-aware answer.

6. **Web Interface (Optional)**
   - Users interact through a Streamlit UI by uploading files and asking questions.
   - All components run live on user input without needing manual intervention.

## 🌐 Streamlit Web App

A `Streamlit` web interface is available via `app.py`. Users can upload their own documents and interact with the QA system through a user-friendly web interface.

### 📂 Features

- Upload `.txt`, `.pdf`, or `.docx` files
- Enter questions through a text box
- View answers directly below the question
- Real-time document ingestion and QA

### ▶️ To Run the App

```bash
streamlit run app.py
```

Make sure your `.env` file is configured properly and required libraries are installed.

## 🖥️ CLI Interface

The project also supports a command-line interface for offline interaction.

### ▶️ How to Use

After running the script:

```bash
python main.py
```

You will see:

```
📄 RAG-Based QA System is initializing and indexing documents...
🤖 System is ready! You can now ask questions based on your uploaded documents.
```

You can then input your questions in the terminal, and receive context-aware answers generated using your uploaded files. To exit, type `exit` or `quit`.

Example:

```
Your Question (or type 'exit' to quit): What is this document about?
📢 Answer:
This document discusses...
```

## 🛠️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/Shubh-17-am/RAG-Based-QA-System.git
cd RAG-Based-QA-System
```

2. **Create a virtual environment and activate it**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Add your API keys to `.env`**

```
GROQ_API_KEY=your_key_here
```

5. **Add your documents to the `documents/` folder**

6. **Run the application**

```bash
python main.py
```

## 📦 Dependencies

- Python ≥ 3.8
- HuggingFace Transformers
- Chroma
- SentenceTransformers
- Streamlit
- `python-docx`, `PyPDF2`, `dotenv`, etc.

Install all requirements using:

```bash
pip install -r requirements.txt
```

## 📌 TODOs

- [ ] Add upload status progress bar
- [ ] Integrate document re-ingestion toggle
- [ ] Support chat history in Streamlit UI
- [ ] Add evaluation metrics (e.g., accuracy, F1)
- [ ] Extend multi-LLM backend selection

## 🙌 Acknowledgements

Inspired by open-source RAG frameworks and LLM applications from OpenAI, Langchain, HuggingFace, and community-driven tools.

---

**Author:** [Shubham Jadhav](https://github.com/Shubh-17-am)  
Feel free to star ⭐ this repo and contribute!
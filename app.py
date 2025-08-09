import os
import time
import streamlit as st
import tempfile
from datetime import datetime

from rag_system import RAGSystem
from config import Config
from utils import DocumentProcessor, ChatHistoryManager

# Page config
st.set_page_config(page_title="RAG-Based QA System", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# Persistent chat history manager
history_manager = ChatHistoryManager(max_history=100)

def load_custom_css():
    st.markdown("""
    <style>
        .chat-message { padding: 1rem; border-radius: 0.75rem; margin-bottom: 1rem; border: 1px solid #333; }
        .user-message { background-color: #003366; border-left: 4px solid #2196F3; color: #fff; }
        .assistant-message { background-color: #2a2a2a; border-left: 4px solid #4CAF50; color: #e0ffe0; }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        # Convert persistent history to UI format
        raw_history = history_manager.get_history()
        ui_history = []
        for exchange in raw_history:
            ui_history.append({"role": "user", "content": exchange["question"]})
            ui_history.append({"role": "assistant", "content": exchange["answer"]})
        st.session_state.chat_history = ui_history
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = []
    if 'current_llm' not in st.session_state:
        st.session_state.current_llm = "groq"

def process_documents(uploaded_files, chunk_size, chunk_overlap, re_ingest):
    progress_bar = st.progress(0, text="Initializing...")
    cfg = Config()
    rag_system = RAGSystem(cfg)
    rag_system.update_llm_provider(st.session_state.current_llm)

    processed_docs = []
    total_files = len(uploaded_files)
    for i, file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        try:
            dp = DocumentProcessor(cfg)
            chunks = dp.process_document(tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # Verify chunks were created
            if not chunks:
                st.warning(f"No content chunks were extracted from {file.name}. The document might be empty or corrupted.")
                continue
                
            rag_system.add_documents(chunks, file.name, force_reprocess=re_ingest)
            processed_docs.append(file.name)
            progress_bar.progress(int((i+1)/total_files*100), text=f"Processed {file.name}")
        finally:
            os.unlink(tmp_path)
    
    # Verify documents were added to vector store
    if processed_docs:
        doc_count = rag_system.get_document_count()
        chunk_count = rag_system.get_chunk_count()
        st.success(f"Processed {len(processed_docs)} documents. Total documents: {doc_count}, Total chunks: {chunk_count}")
    else:
        st.error("No documents were successfully processed.")
    
    st.session_state.rag_system = rag_system
    st.session_state.documents_processed = processed_docs

def answer_question(question):
    cfg = st.session_state.rag_system.config
    
    # Pass question to RAG system with memory
    result = st.session_state.rag_system.query(
        question,
        k=cfg.rag.retriever_k,
        top_n_after_rerank=cfg.rag.top_n_after_rerank
    )

    # Add to UI memory
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})

    # Save to persistent history
    history_manager.add_exchange(question, result["answer"], result["query_time"], result["sources"])

    # Display sources if available
    if result["sources"]:
        st.markdown(f"**Sources:** {', '.join(result['sources'])}")
    
    # Show conversation memory
    with st.expander("Conversation Memory"):
        memory_content = st.session_state.rag_system.get_conversation_history()
        st.text(memory_content)
    
    with st.expander("View retrieved context"):
        if result["context_used"]:
            for i, ctx in enumerate(result["context_used"]):
                st.write(f"**Chunk {i+1}:**")
                st.write(ctx)
                st.markdown("---")
        else:
            st.write("No context was retrieved for this query")

    st.rerun()

def main():
    load_custom_css()
    initialize_session_state()
    st.title("🤖 RAG-Based QA System (FAISS + LangChain Memory)")

    with st.sidebar:
        st.header("⚙️ Configuration")
        llm_options = ["groq", "openai", "anthropic"]
        selected_llm = st.selectbox("LLM Provider", llm_options, index=llm_options.index(st.session_state.current_llm))
        if selected_llm != st.session_state.current_llm:
            st.session_state.current_llm = selected_llm
            if st.session_state.rag_system:
                st.session_state.rag_system.update_llm_provider(selected_llm)
        chunk_size = st.slider("Chunk Size", 100, 2000, 500, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 100, 50)
        re_ingest = st.checkbox("Force Re-ingestion")
        if st.button("Clear Chat History"):
            history_manager.clear()
            st.session_state.chat_history = []
            if st.session_state.rag_system:
                st.session_state.rag_system.clear_memory()
            st.success("Chat history cleared!")
        if st.button("Clear Vector Store"):
            if st.session_state.rag_system:
                st.session_state.rag_system.clear_documents()
                st.session_state.documents_processed = []
                st.success("Vector store cleared!")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📁 Document Upload")
        uploaded_files = st.file_uploader("Upload Documents", type=['txt','pdf','docx'], accept_multiple_files=True)
        if uploaded_files and st.button("Process Documents"):
            process_documents(uploaded_files, chunk_size, chunk_overlap, re_ingest)
        if st.session_state.documents_processed:
            st.subheader("✅ Processed Documents")
            for doc in st.session_state.documents_processed:
                st.write(f"📄 {doc}")
        
        # Show vector store status
        if st.session_state.rag_system:
            st.subheader("📊 Vector Store Status")
            doc_count = st.session_state.rag_system.get_document_count()
            chunk_count = st.session_state.rag_system.get_chunk_count()
            st.write(f"Documents: {doc_count}")
            st.write(f"Chunks: {chunk_count}")
            st.write("Type: FAISS (In-Memory)")

    with col2:
        st.subheader("💬 Chat Interface")
        for message in st.session_state.chat_history:
            msg_class = "user-message" if message["role"] == "user" else "assistant-message"
            st.markdown(f"<div class='chat-message {msg_class}'>{message['content']}</div>", unsafe_allow_html=True)
        question = st.text_input("Ask a question:")
        if st.button("Ask") and question:
            if st.session_state.rag_system:
                answer_question(question)
            else:
                st.error("Upload and process documents first!")

if __name__ == "__main__":
    main()
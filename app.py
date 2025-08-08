"""
Enhanced RAG-Based QA System - Streamlit Web Interface
Features: Chat history, progress bars, multi-LLM support, document re-ingestion
"""

import os
import time
import streamlit as st
from typing import List, Dict, Optional
import tempfile
from datetime import datetime
import json

# Import our RAG system components
from rag_system import RAGSystem
from config import Config
from utils import DocumentProcessor, ChatHistoryManager

# Page configuration
st.set_page_config(
    page_title="RAG-Based QA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def load_custom_css():
    st.markdown("""
    <style>
        .chat-message {
            padding: 1rem;
            border-radius: 0.75rem;
            margin-bottom: 1rem;
            border: 1px solid #333333;
            background-color: #1e1e1e;
            color: #f1f1f1;
        }

        .user-message {
            background-color: #003366; /* Deep navy blue */
            border-left: 4px solid #2196F3;
            color: #ffffff;
        }

        .assistant-message {
            background-color: #2a2a2a; /* Dark gray */
            border-left: 4px solid #4CAF50;
            color: #e0ffe0;
        }

        .status-text {
            font-size: 0.9rem;
            color: #aaa;
            margin-top: 0.5rem;
        }

        strong {
            color: #cccccc;
        }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = []
    if 'processing_progress' not in st.session_state:
        st.session_state.processing_progress = 0
    if 'current_llm' not in st.session_state:
        st.session_state.current_llm = "groq"

def update_progress(progress_bar, status_text, progress: float, status: str):
    """Update progress bar with correct value format"""
    # Convert percentage to 0-100 integer for Streamlit
    progress_int = int(progress)
    progress_bar.progress(progress_int, text=status)
    status_text.text(f"Status: {status} ({progress_int}%)")

def main():
    load_custom_css()
    initialize_session_state()
    
    st.title("🤖 RAG-Based QA System")
    st.markdown("Upload documents and ask questions based on their content")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM Selection
        llm_options = ["groq", "openai", "anthropic"]
        selected_llm = st.selectbox(
            "Select LLM Provider:",
            llm_options,
            index=llm_options.index(st.session_state.current_llm)
        )
        
        if selected_llm != st.session_state.current_llm:
            st.session_state.current_llm = selected_llm
            if st.session_state.rag_system:
                st.session_state.rag_system.update_llm_provider(selected_llm)
        
        # Document processing options
        st.subheader("📄 Document Settings")
        chunk_size = st.slider("Chunk Size:", 100, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap:", 0, 500, 200, 50)
        
        # Re-ingestion toggle
        re_ingest = st.checkbox("Force Re-ingestion", 
                               help="Reprocess all documents even if they've been processed before")
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
        
        # Clear documents
        if st.button("Clear All Documents"):
            st.session_state.documents_processed = []
            st.session_state.rag_system = None
            st.success("All documents cleared!")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Document Upload")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True,
            help="Supported formats: .txt, .pdf, .docx"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                process_documents(uploaded_files, chunk_size, chunk_overlap, re_ingest)
        
        # Display processed documents
        if st.session_state.documents_processed:
            st.subheader("✅ Processed Documents")
            for doc in st.session_state.documents_processed:
                st.write(f"📄 {doc}")
    
    with col2:
        st.subheader("💬 Chat Interface")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                message_type = "user-message" if message["role"] == "user" else "assistant-message"
                st.markdown(f"""
                <div class="chat-message {message_type}">
                    <strong>{message['role'].title()}:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
        
        # Question input
        question = st.text_input(
            "Ask a question:",
            placeholder="Type your question here...",
            key="question_input"
        )
        
        if st.button("Ask Question", type="primary") and question:
            if st.session_state.rag_system:
                answer_question(question)
            else:
                st.error("Please upload and process documents first!")

def process_documents(uploaded_files, chunk_size, chunk_overlap, re_ingest):
    """Process uploaded documents with progress tracking"""
    # Create progress bar and status text
    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()
    
    try:
        # Initialize RAG system
        update_progress(progress_bar, status_text, 10, "Initializing RAG system...")
        
        config = Config()
        rag_system = RAGSystem(config)
        rag_system.update_llm_provider(st.session_state.current_llm)
        
        # Process documents
        update_progress(progress_bar, status_text, 20, "Processing documents...")
        
        processed_docs = []
        total_files = len(uploaded_files)
        
        for i, file in enumerate(uploaded_files):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Process document
                doc_processor = DocumentProcessor()
                chunks = doc_processor.process_document(
                    tmp_file_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Add to RAG system
                rag_system.add_documents(chunks, file.name, force_reprocess=re_ingest)
                processed_docs.append(file.name)
                
                # Update progress - calculate as percentage (0-100)
                progress = 20 + int((60 * (i + 1) / total_files))
                update_progress(progress_bar, status_text, progress, 
                              f"Processing {file.name} ({i+1}/{total_files})")
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        # Finalize
        update_progress(progress_bar, status_text, 90, "Finalizing...")
        
        st.session_state.rag_system = rag_system
        st.session_state.documents_processed = processed_docs
        
        update_progress(progress_bar, status_text, 100, "✅ Documents processed successfully!")
        
        st.success(f"Successfully processed {len(processed_docs)} documents!")
        
        # Clear progress bar after completion
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def answer_question(question):
    """Answer a question using the RAG system"""
    try:
        # Add question to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now().isoformat()
        })
        
        # Show thinking indicator
        with st.spinner("🤔 Thinking..."):
            # Get answer from RAG system
            response = st.session_state.rag_system.query(question)
            
            # Add answer to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"],
                "timestamp": datetime.now().isoformat(),
                "sources": response.get("sources", [])
            })
        
        # Rerun to update chat display
        st.rerun()
        
    except Exception as e:
        st.error(f"Error answering question: {str(e)}")

if __name__ == "__main__":
    main()
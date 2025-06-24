# app.py
import streamlit as st
import os
import tempfile
from text_processor import process_document
from vector_db import index_documents, get_top_k_chunks
from llm_generator import generate_answer

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = []
if 'processed' not in st.session_state:
    st.session_state.processed = False

st.title("📘 RAG Based QA System")
st.subheader("Upload study materials and ask questions")

# Document upload section
uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT)",
    type=['pdf', 'docx', 'txt'],
    accept_multiple_files=True
)

# Process documents when files are uploaded
if uploaded_files and not st.session_state.processed:
    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                file_path = tmp_file.name
            
            try:
                # Process document
                chunks = process_document(file_path)
                index_documents(chunks)
                st.success(f"Processed: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            finally:
                # Clean up temp file
                os.unlink(file_path)
    
    st.session_state.processed = True
    st.success("All documents processed! Ready for questions.")

# Q&A Section
if st.session_state.processed:
    st.divider()
    st.subheader("Ask a question about your documents")
    
    question = st.text_input("Your question:", placeholder="Type your question here...")
    
    if question:
        with st.spinner("Searching for answers..."):
            try:
                # Retrieve relevant chunks and generate answer
                chunks = get_top_k_chunks(question)
                answer = generate_answer(question, chunks)
                
                st.subheader("Answer:")
                st.write(answer)
                
                with st.expander("See relevant content used"):
                    for i, chunk in enumerate(chunks, 1):
                        st.caption(f"Relevant passage #{i}")
                        st.info(chunk)
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
else:
    st.info("Please upload documents to get started")

# Reset functionality
st.sidebar.header("Configuration")
if st.sidebar.button("Reset Documents"):
    st.session_state.vector_store = []
    st.session_state.processed = False
    st.rerun()
# vector_db.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import Config
import streamlit as st

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer(Config.EMBEDDING_MODEL)

embedder = load_embedder()

# Function to index document chunks
def index_documents(chunks):
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = []
        
    for chunk in chunks:
        vector = embedder.encode(chunk)
        st.session_state.vector_store.append({
            "chunk": chunk,
            "vector": vector
        })

# Function to retrieve top-K most relevant chunks
def get_top_k_chunks(query, k=Config.TOP_K):
    if 'vector_store' not in st.session_state or not st.session_state.vector_store:
        return []
        
    query_vec = embedder.encode(query)
    similarities = [
        cosine_similarity([query_vec], [item["vector"]])[0][0]
        for item in st.session_state.vector_store
    ]
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [st.session_state.vector_store[i]["chunk"] for i in top_indices]
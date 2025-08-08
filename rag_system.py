"""
Core RAG System implementation
"""

import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI, Anthropic
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from groq import Groq
from typing import Optional, List
from pydantic import PrivateAttr

from config import Config
from utils import DocumentProcessor, Logger

# Custom Groq LangChain Wrapper
class GroqLangChainWrapper(LLM):
    """Custom LangChain wrapper for Groq API"""
    
    _client: Groq = PrivateAttr()
    _default_model: str = PrivateAttr()
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", **kwargs):
        super().__init__(**kwargs)
        self._client = Groq(api_key=api_key)
        self._default_model = model
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._default_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000),
                top_p=kwargs.get('top_p', 0.9)
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Error interacting with Groq API: {e}")

@dataclass
class QueryResult:
    """Result of a RAG query"""
    answer: str
    sources: List[str]
    query_time: float
    context_used: List[str]
    confidence_score: float

class RAGSystem:
    """Main RAG System class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger(__name__)
        self.vector_store = None
        self.qa_chain = None
        self.current_llm_provider = config.llm.default_provider
        self.document_hashes = {}  # Track processed documents
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_llm()
        self._initialize_vector_store()
        self._initialize_qa_chain()
    
    def _initialize_embeddings(self):
        """Initialize text embeddings"""
        self.logger.info("Initializing text embeddings...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding.model_name,
            cache_folder=self.config.embedding.cache_folder,
            model_kwargs={'device': self.config.embedding.device}
        )
        
        self.logger.info(f"Embeddings initialized with model: {self.config.embedding.model_name}")
    
    def _initialize_llm(self):
        """Initialize LLM provider"""
        self.logger.info(f"Initializing LLM provider: {self.current_llm_provider}")
        
        api_key = self.config.get_llm_api_key(self.current_llm_provider)
        
        if self.current_llm_provider == "openai":
            self.llm = OpenAI(
                api_key=api_key,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens
            )
        elif self.current_llm_provider == "anthropic":
            self.llm = Anthropic(
                api_key=api_key,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens
            )
        elif self.current_llm_provider == "groq":
            self.llm = GroqLangChainWrapper(
                api_key=api_key,
                model=self.config.llm.groq_model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                top_p=self.config.llm.top_p
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.current_llm_provider}")
        
        self.logger.info(f"LLM initialized: {self.current_llm_provider}")
    
    def _initialize_vector_store(self):
        """Initialize vector store"""
        self.logger.info("Initializing vector store...")
        
        self.vector_store = Chroma(
            persist_directory=self.config.vector_store.persist_directory,
            collection_name=self.config.vector_store.collection_name,
            embedding_function=self.embeddings
        )
        
        self.logger.info("Vector store initialized")
    
    def _initialize_qa_chain(self):
        """Initialize QA chain"""
        self.logger.info("Initializing QA chain...")
        
        # Create prompt template
        prompt_template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        
        Context: {context}
        
        Question: {question}
        
        Helpful Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        self.logger.info("QA chain initialized")
    
    def add_documents(self, chunks: List[str], document_name: str, force_reprocess: bool = False):
        """Add document chunks to the vector store"""
        self.logger.info(f"Adding documents from: {document_name}")
        
        # Calculate document hash
        doc_hash = self._calculate_document_hash(chunks)
        
        # Check if document already processed
        if not force_reprocess and document_name in self.document_hashes:
            if self.document_hashes[document_name] == doc_hash:
                self.logger.info(f"Document {document_name} already processed, skipping...")
                return
        
        # Add documents to vector store
        texts = chunks
        metadatas = [{"source": document_name, "chunk_id": i} for i in range(len(chunks))]
        
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        self.vector_store.persist()
        
        # Update document hash
        self.document_hashes[document_name] = doc_hash
        
        self.logger.info(f"Added {len(chunks)} chunks from {document_name}")
    
    def _calculate_document_hash(self, chunks: List[str]) -> str:
        """Calculate hash of document chunks"""
        content = "".join(chunks)
        return hashlib.md5(content.encode()).hexdigest()
    
    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Query the RAG system"""
        start_time = time.time()
        
        try:
            # Get response from QA chain
            result = self.qa_chain({"query": question})
            
            # Extract answer and sources
            answer = result["result"]
            source_docs = result.get("source_documents", [])
            
            # Process sources
            sources = []
            context_used = []
            
            for doc in source_docs:
                source = doc.metadata.get("source", "Unknown")
                if source not in sources:
                    sources.append(source)
                context_used.append(doc.page_content)
            
            # Calculate confidence score (simple heuristic)
            confidence_score = self._calculate_confidence_score(answer, question, context_used)
            
            query_time = time.time() - start_time
            
            return {
                "answer": answer,
                "sources": sources,
                "query_time": query_time,
                "context_used": context_used,
                "confidence_score": confidence_score
            }
            
        except Exception as e:
            self.logger.error(f"Error querying RAG system: {str(e)}")
            raise
    
    def _calculate_confidence_score(self, answer: str, question: str, context: List[str]) -> float:
        """Calculate confidence score for the answer"""
        # Simple heuristic: longer answers with more context usage get higher scores
        answer_length = len(answer.split())
        context_usage = len(context)
        
        # Normalize scores
        length_score = min(answer_length / 100, 1.0)  # Max 1.0 for 100+ words
        context_score = min(context_usage / 3, 1.0)  # Max 1.0 for 3+ contexts
        
        # Combined score
        confidence = (length_score * 0.4 + context_score * 0.6)
        
        return round(confidence, 2)
    
    def update_llm_provider(self, provider: str):
        """Update LLM provider"""
        if provider == self.current_llm_provider:
            return
        
        self.logger.info(f"Switching LLM provider from {self.current_llm_provider} to {provider}")
        
        self.current_llm_provider = provider
        self._initialize_llm()
        self._initialize_qa_chain()
    
    def get_document_count(self) -> int:
        """Get number of documents in vector store"""
        return len(self.document_hashes)
    
    def get_chunk_count(self) -> int:
        """Get number of chunks in vector store"""
        return self.vector_store._collection.count()
    
    def clear_documents(self):
        """Clear all documents from vector store"""
        self.logger.info("Clearing all documents from vector store")
        
        # Delete and recreate collection
        self.vector_store.delete_collection()
        self._initialize_vector_store()
        self._initialize_qa_chain()
        
        # Clear document hashes
        self.document_hashes.clear()
        
        self.logger.info("All documents cleared")
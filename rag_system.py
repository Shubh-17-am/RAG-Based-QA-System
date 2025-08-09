import os
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema import Document

from groq import Groq
from pydantic import PrivateAttr
from sentence_transformers import CrossEncoder

from config import Config
from utils import Logger, cosine_similarity_list

# --- Groq wrapper left largely unchanged (keeps compatibility) ---
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
                temperature=kwargs.get('temperature', 0.0),
                max_tokens=kwargs.get('max_tokens', 512),
                top_p=kwargs.get('top_p', 0.9)
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Error interacting with Groq API: {e}")


@dataclass
class QueryResult:
    answer: str
    sources: List[str]
    query_time: float
    context_used: List[str]


class RAGSystem:
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger(__name__)
        self.vector_store: Optional[FAISS] = None
        self.current_llm_provider = config.llm.default_provider
        self.document_hashes = {}
        self.embeddings = None
        self.llm = None
        self.cross_encoder = None
        self.max_history_length = config.rag.max_history_length
        
        # Initialize LangChain memory with proper configuration
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True,
            human_prefix="Human",
            ai_prefix="Assistant"
        )
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_llm()
        self._initialize_cross_encoder()
        self._initialize_vector_store()
        self._initialize_prompt_template()
        self._initialize_qa_chain()

    # ------------------------------
    # Initialization helpers
    # ------------------------------
    def _initialize_embeddings(self):
        self.logger.info("Initializing embeddings...")
        model_name = getattr(self.config.embedding, "model_name", None) or "sentence-transformers/all-mpnet-base-v2"
        device = getattr(self.config.embedding, "device", "cpu")
        cache_folder = getattr(self.config.embedding, "cache_folder", None)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_folder,
            model_kwargs={"device": device}
        )
        self.logger.info(f"Embeddings ready: {model_name} on {device}")

    def _initialize_llm(self):
        self.logger.info(f"Initializing LLM provider: {self.current_llm_provider}")
        api_key = self.config.get_llm_api_key(self.current_llm_provider)

        if self.current_llm_provider == "openai":
            from langchain_community.llms import OpenAI  # keep import local to avoid unused import
            self.llm = OpenAI(
                api_key=api_key,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens
            )
        elif self.current_llm_provider == "anthropic":
            from langchain_community.llms import Anthropic
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

    def _initialize_cross_encoder(self):
        self.logger.info("Initializing cross-encoder for re-ranking...")
        model_name = getattr(self.config.rag, "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.cross_encoder = CrossEncoder(model_name)
        self.logger.info(f"Cross-encoder ready: {model_name}")

    def _initialize_vector_store(self):
        self.logger.info("Initializing in-memory FAISS vector store...")
        try:
            # Get embedding dimension
            embedding_dim = len(self.embeddings.embed_query("test"))
            import faiss
            
            # Create an empty FAISS index
            index = faiss.IndexFlatIP(embedding_dim)
            
            # Create empty docstore and mapping
            docstore = InMemoryDocstore({})
            index_to_docstore_id = {}
            
            # Create FAISS vector store
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )
            
            self.logger.info("In-memory FAISS ready")
        except Exception as e:
            self.logger.error(f"Error initializing FAISS: {e}")
            # Fallback: create with a dummy document and clear it
            self.logger.info("Trying fallback initialization...")
            try:
                # Create with one dummy document
                self.vector_store = FAISS.from_texts(
                    texts=["initialization dummy"],
                    embedding=self.embeddings,
                    metadatas=[{"source": "dummy"}]
                )
                
                # Get the current index and create a new empty one
                embedding_dim = len(self.embeddings.embed_query("test"))
                index = faiss.IndexFlatIP(embedding_dim)
                docstore = InMemoryDocstore({})
                index_to_docstore_id = {}
                
                # Replace with empty vector store
                self.vector_store = FAISS(
                    embedding_function=self.embeddings,
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id=index_to_docstore_id
                )
                
                self.logger.info("In-memory FAISS ready (fallback)")
            except Exception as e2:
                self.logger.error(f"Fallback initialization also failed: {e2}")
                raise

    def _initialize_prompt_template(self):
        self.logger.info("Initializing prompt template...")
        template = """
        You are a helpful AI assistant with access to document context and conversation history.
        
        Use the following pieces of context to answer the question at the end. If you don't know the answer, 
        just say that you don't know. Do not try to make up an answer. Keep the answer concise.

        Conversation History:
        {chat_history}

        Context:
        {context}

        Question: {question}

        Helpful Answer:"""
        
        self.prompt_template = PromptTemplate(
            template=template,
            input_variables=["chat_history", "context", "question"]
        )
        self.logger.info("Prompt template initialized")

    def _initialize_qa_chain(self):
        """Initialize the QA chain with memory using the new LangChain syntax"""
        self.logger.info("Initializing QA chain with memory...")
        
        # Create a function to get formatted chat history
        def get_formatted_chat_history(_):
            memory_vars = self.memory.load_memory_variables({})
            messages = memory_vars.get("chat_history", [])
            if not messages:
                return ""
            
            formatted_history = ""
            for i, msg in enumerate(messages):
                if i % 2 == 0:  # Human message
                    formatted_history += f"Human: {msg.content}\n"
                else:  # AI message
                    formatted_history += f"Assistant: {msg.content}\n"
            return formatted_history
        
        # Create the chain using the new pipe syntax
        self.qa_chain = (
            RunnablePassthrough.assign(
                chat_history=get_formatted_chat_history
            )
            | self.prompt_template
            | self.llm
        )
        
        self.logger.info("QA chain initialized")

    # ------------------------------
    # Document ingestion
    # ------------------------------
    def add_documents(self, chunks: List[Tuple[str, Dict]], document_name: str, force_reprocess: bool = False):
        if not chunks:
            self.logger.warning("add_documents called with empty chunks; skipping.")
            return

        self.logger.info(f"Adding document: {document_name} (chunks={len(chunks)})")
        
        # Extract text and metadata from chunks
        texts = [chunk[0] for chunk in chunks]
        metadatas = [chunk[1] for chunk in chunks]
        
        # Calculate hash from the text content
        doc_hash = self._calculate_document_hash(texts)

        if not force_reprocess and document_name in self.document_hashes:
            if self.document_hashes[document_name] == doc_hash:
                self.logger.info(f"{document_name} unchanged — skipping ingestion.")
                return

        try:
            # If this is the first document, create a new FAISS index
            if self.vector_store.index.ntotal == 0:
                self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            else:
                # Add to existing index
                self.vector_store.add_texts(texts, metadatas=metadatas)
            
            self.document_hashes[document_name] = doc_hash
            self.logger.info(f"Added {len(texts)} chunks for {document_name}")
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            raise

    def _calculate_document_hash(self, chunks: List[str]) -> str:
        content = "".join(chunks)
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    # ------------------------------
    # Enhanced querying with proper memory integration
    # ------------------------------
    def query(self, question: str, k: int = None, top_n_after_rerank: int = None) -> Dict[str, Any]:
        """
        Enhanced query pipeline with proper LangChain memory integration:
        1. Use FAISS similarity search to fetch candidates
        2. Re-rank candidates using cross-encoder and embeddings similarity
        3. Build context from retrieved documents
        4. Use LangChain's QA chain with memory to generate answer
        5. Update memory with the conversation
        """
        start_time = time.time()
        k = k or getattr(self.config.rag, "retriever_k", 10)
        top_n_after_rerank = top_n_after_rerank or getattr(self.config.rag, "top_n_after_rerank", 5)

        try:
            # 1) get candidate docs from FAISS similarity search
            candidate_docs = self.vector_store.similarity_search(question, k=k)
                
            # 2) Build context from documents
            if candidate_docs:
                # Re-rank candidates using cross-encoder and embeddings similarity
                query_emb = self.embeddings.embed_query(question)
                texts = [d.page_content for d in candidate_docs]
                doc_embs = self.embeddings.embed_documents(texts)
                sims = cosine_similarity_list(query_emb, doc_embs)

                # Cross-encoder scoring
                query_doc_pairs = [(question, text) for text in texts]
                cross_scores = self.cross_encoder.predict(query_doc_pairs)
                
                # Combine embeddings similarity and cross-encoder scores
                combined_scores = [
                    0.6 * sim + 0.4 * cross_score 
                    for sim, cross_score in zip(sims, cross_scores)
                ]
                
                # attach similarity and choose top N
                scored = list(zip(candidate_docs, combined_scores))
                scored.sort(key=lambda x: x[1], reverse=True)
                top_docs_with_scores = scored[:top_n_after_rerank]
                top_docs = [d for d, s in top_docs_with_scores]
                
                # Build context from documents
                context_text = "\n\n".join([doc.page_content for doc in top_docs])
                
                # Collect sources and context
                sources = []
                context_used = []
                for doc in top_docs:
                    src = getattr(doc, "metadata", {}).get("source", None) or getattr(doc, "source", None)
                    if src and src not in sources:
                        sources.append(src)
                    context_used.append(doc.page_content)
            else:
                context_text = "No relevant documents found."
                sources = []
                context_used = []
            
            # 3) Use LangChain's QA chain with memory to generate answer
            inputs = {"question": question, "context": context_text}
            response = self.qa_chain.invoke(inputs)
            
            # 4) Update memory with the conversation
            self.memory.save_context(
                {"question": question},
                {"answer": response}
            )

            query_time = time.time() - start_time

            return {
                "answer": response,
                "sources": sources,
                "query_time": query_time,
                "context_used": context_used
            }

        except Exception as e:
            self.logger.error(f"Error querying RAG system: {e}")
            fallback_response = "I encountered an issue processing your request. Please try rephrasing your question."
            # Update memory with the fallback response
            try:
                self.memory.save_context(
                    {"question": question},
                    {"answer": fallback_response}
                )
            except:
                pass
            return {
                "answer": fallback_response,
                "sources": [],
                "query_time": time.time() - start_time,
                "context_used": []
            }

    # ------------------------------
    # Memory management
    # ------------------------------
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()
        self.logger.info("Conversation memory cleared")

    def get_conversation_history(self) -> str:
        """Get formatted conversation history"""
        try:
            memory_vars = self.memory.load_memory_variables({})
            messages = memory_vars.get("chat_history", [])
            history_str = ""
            for i, msg in enumerate(messages):
                if i % 2 == 0:
                    history_str += f"Human: {msg.content}\n"
                else:
                    history_str += f"Assistant: {msg.content}\n"
            return history_str.strip()
        except:
            return "No conversation history available"

    def get_memory_variables(self) -> Dict[str, Any]:
        """Get raw memory variables for debugging"""
        return self.memory.load_memory_variables({})

    # ------------------------------
    # Helpers & management
    # ------------------------------
    def update_llm_provider(self, provider: str):
        if provider == self.current_llm_provider:
            return
        self.logger.info(f"Switching LLM provider: {self.current_llm_provider} -> {provider}")
        self.current_llm_provider = provider
        self._initialize_llm()
        self._initialize_qa_chain()  # Reinitialize QA chain with new LLM

    def get_document_count(self) -> int:
        return len(self.document_hashes)

    def get_chunk_count(self) -> int:
        try:
            return self.vector_store.index.ntotal
        except Exception:
            return 0

    def clear_documents(self):
        self.logger.info("Clearing in-memory FAISS vector store...")
        try:
            # Reinitialize an empty FAISS index
            self._initialize_vector_store()
        except Exception as e:
            self.logger.warning(f"Error clearing vector store: {e}")
        self.document_hashes.clear()
        self.clear_memory()
        self.logger.info("Documents and memory cleared.")


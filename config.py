import os
from typing import Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str = "cpu"
    batch_size: int = 32
    cache_folder: str = "./embeddings_cache"

@dataclass
class VectorStoreConfig:
    # FAISS doesn't need persistence config - it's in-memory only
    distance_metric: str = "cosine"

@dataclass
class DocumentConfig:
    chunk_size: int = 500  # Smaller chunks for better precision
    chunk_overlap: int = 100  # Reduced overlap
    max_file_size: int = 50 * 1024 * 1024
    supported_formats: list = field(default_factory=lambda: [".txt", ".pdf", ".docx"])

@dataclass
class LLMConfig:
    default_provider: str = "groq"
    temperature: float = 0.1  # Slightly increased for more flexibility
    max_tokens: int = 800
    top_p: float = 0.9
    openai_model: str = "gpt-3.5-turbo"
    anthropic_model: str = "claude-3-sonnet-20240229"
    groq_model: str = "llama3-8b-8192"

@dataclass
class RAGParams:
    retriever_k: int = 10  # FAISS is efficient, we don't need as many candidates
    top_n_after_rerank: int = 5  # Use top 5 after reranking
    low_conf_threshold: float = 0.2  # Reasonable threshold
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_history_length: int = 5  # Number of conversation exchanges to remember

@dataclass
class AppConfig:
    debug: bool = True  # Enabled by default
    log_level: str = "INFO"
    max_chat_history: int = 100
    enable_evaluation: bool = True
    auto_save: bool = False  # Disabled for session-only

@dataclass
class Config:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    document: DocumentConfig = field(default_factory=DocumentConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGParams = field(default_factory=RAGParams)
    app: AppConfig = field(default_factory=AppConfig)

    def __post_init__(self):
        self._validate_api_keys()
        self._validate_paths()

    def _validate_api_keys(self):
        if self.llm.default_provider == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI API key missing")
        if self.llm.default_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("Anthropic API key missing")
        if self.llm.default_provider == "groq" and not self.groq_api_key:
            raise ValueError("Groq API key missing")

    def _validate_paths(self):
        os.makedirs(self.embedding.cache_folder, exist_ok=True)
        # No vector_store path needed for FAISS

    def get_llm_api_key(self, provider: str) -> str:
        return {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key
        }.get(provider, "")
    
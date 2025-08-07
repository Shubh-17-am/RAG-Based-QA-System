"""
Configuration management for RAG-Based QA System
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class EmbeddingConfig:
    """Configuration for text embeddings"""
    model_name: str = "all-mpnet-base-v2"
    device: str = "cpu"
    batch_size: int = 32
    cache_folder: str = "./embeddings_cache"

@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    persist_directory: str = "./vector_store"
    collection_name: str = "rag_documents"
    distance_metric: str = "cosine"

@dataclass
class DocumentConfig:
    """Configuration for document processing"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    supported_formats: list = field(default_factory=lambda: [".txt", ".pdf", ".docx"])

@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    default_provider: str = "groq"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    
    # Provider-specific configs
    openai_model: str = "gpt-3.5-turbo"
    anthropic_model: str = "claude-3-sonnet-20240229"
    groq_model: str = "llama3-8b-8192"

@dataclass
class AppConfig:
    """Application configuration"""
    debug: bool = False
    log_level: str = "INFO"
    max_chat_history: int = 100
    enable_evaluation: bool = True
    auto_save: bool = True

@dataclass
class Config:
    """Main configuration class"""
    # API Keys
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    
    # Sub-configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    document: DocumentConfig = field(default_factory=DocumentConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    app: AppConfig = field(default_factory=AppConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_api_keys()
        self._validate_paths()
    
    def _validate_api_keys(self):
        """Validate that required API keys are present"""
        if self.llm.default_provider == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI as default provider")
        if self.llm.default_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("Anthropic API key is required when using Anthropic as default provider")
        if self.llm.default_provider == "groq" and not self.groq_api_key:
            raise ValueError("Groq API key is required when using Groq as default provider")
    
    def _validate_paths(self):
        """Validate and create necessary directories"""
        os.makedirs(self.embedding.cache_folder, exist_ok=True)
        os.makedirs(self.vector_store.persist_directory, exist_ok=True)
    
    def get_llm_api_key(self, provider: str) -> str:
        """Get API key for specific LLM provider"""
        key_map = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key
        }
        return key_map.get(provider, "")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "embedding": self.embedding.__dict__,
            "vector_store": self.vector_store.__dict__,
            "document": self.document.__dict__,
            "llm": self.llm.__dict__,
            "app": self.app.__dict__
        }
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file"""
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create config object with loaded data
        config = cls()
        
        # Update sub-configurations
        if "embedding" in config_data:
            config.embedding = EmbeddingConfig(**config_data["embedding"])
        if "vector_store" in config_data:
            config.vector_store = VectorStoreConfig(**config_data["vector_store"])
        if "document" in config_data:
            config.document = DocumentConfig(**config_data["document"])
        if "llm" in config_data:
            config.llm = LLMConfig(**config_data["llm"])
        if "app" in config_data:
            config.app = AppConfig(**config_data["app"])
        
        return config
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        import json
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
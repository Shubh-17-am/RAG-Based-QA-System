import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = "llama3-70b-8192" 
    CHUNK_SIZE = 512 
    CHUNK_OVERLAP = 100
    TOP_K = 3
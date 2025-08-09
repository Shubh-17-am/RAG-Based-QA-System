import os
import re
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import hashlib
import math

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from config import Config

# ---------------------------
# Logging helper
# ---------------------------
class Logger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)

    def warning(self, message: str):
        self.logger.warning(message)

# ---------------------------
# Enhanced Document processing
# ---------------------------
class DocumentProcessor:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = Logger(__name__)

        # default chunker; chunk size / overlap can be overridden at call time
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.document.chunk_size,
            chunk_overlap=self.config.document.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_document(self, file_path: str, chunk_size: int = None, chunk_overlap: int = None) -> List[Tuple[str, Dict]]:
        self.logger.info(f"Processing document: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size > self.config.document.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.config.document.max_file_size})")

        ext = Path(file_path).suffix.lower()
        if ext not in self.config.document.supported_formats:
            raise ValueError(f"Unsupported file extension: {ext}")

        documents = self._load_document(file_path, ext)

        # Apply overrides
        if chunk_size:
            self.text_splitter.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.text_splitter.chunk_overlap = chunk_overlap

        chunks = self.text_splitter.split_documents(documents)
        
        # Enhanced text cleaning and filtering
        text_chunks = []
        for chunk in chunks:
            cleaned_text = self.clean_text(chunk.page_content)
            # Reduced minimum content threshold from 50 to 20 characters
            if cleaned_text and len(cleaned_text.strip()) >= 20:  
                text_chunks.append(cleaned_text)

        # Add metadata to chunks for better source tracking
        chunks_with_metadata = []
        for i, chunk in enumerate(text_chunks):
            metadata = {
                "source": os.path.basename(file_path),
                "chunk_id": i,
                "file_type": ext,
                "char_count": len(chunk)
            }
            chunks_with_metadata.append((chunk, metadata))

        self.logger.info(f"Processed into {len(chunks_with_metadata)} chunks")
        return chunks_with_metadata

    def _load_document(self, file_path: str, file_ext: str):
        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif file_ext == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        return loader.load()

    def clean_text(self, text: str) -> str:
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\^\&\*\+\=\~\`]', '', text)
        # Clean up line breaks
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
        return text.strip()

# ---------------------------
# Simple math / embedding helpers
# ---------------------------
def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def cosine_similarity_list(query_emb: List[float], doc_embs: List[List[float]]) -> List[float]:
    return [_cosine(query_emb, d) for d in doc_embs]

# ---------------------------
# Enhanced ChatHistoryManager
# ---------------------------
class ChatHistoryManager:
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.history_file = "chat_history.json"
        self.history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self):
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving chat history: {str(e)}")

    def add_exchange(self, question: str, answer: str, response_time: float, sources: List[str] = None):
        exchange = {
            "question": question,
            "answer": answer,
            "response_time": response_time,
            "sources": sources or [],
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(exchange)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        self._save_history()

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history.copy()

    def clear(self):
        self.history = []
        self._save_history()

    def search_history(self, query: str) -> List[Dict[str, Any]]:
        q = query.lower()
        return [ex for ex in self.history if q in ex["question"].lower() or q in ex["answer"].lower()]

# -----------
# Evaluator
# -----------
class Evaluator:
    def __init__(self):
        self.logger = Logger(__name__)

    def evaluate(self, rag_system, eval_data: List[Dict[str, Any]]) -> Dict[str, float]:
        self.logger.info(f"Evaluating {len(eval_data)} questions")
        results = []
        total_response_time = 0.0
        for item in eval_data:
            question = item["question"]
            expected_answer = item["expected_answer"]
            try:
                response = rag_system.query(question)
                accuracy = self._calculate_accuracy(response["answer"], expected_answer)
                precision, recall = self._calculate_precision_recall(response["answer"], expected_answer)
                f1_score = self._calculate_f1_score(precision, recall)
                results.append({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "response_time": response.get("query_time", 0.0)
                })
                total_response_time += response.get("query_time", 0.0)
            except Exception as e:
                self.logger.error(f"Error evaluating question '{question}': {e}")
                results.append({"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "response_time": 0.0})

        n = len(results) or 1
        avg_accuracy = sum(r["accuracy"] for r in results) / n
        avg_precision = sum(r["precision"] for r in results) / n
        avg_recall = sum(r["recall"] for r in results) / n
        avg_f1_score = sum(r["f1_score"] for r in results) / n
        avg_response_time = total_response_time / n

        return {
            "accuracy": avg_accuracy,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": avg_f1_score,
            "avg_response_time": avg_response_time,
            "total_questions": len(eval_data)
        }

    def _calculate_accuracy(self, predicted: str, expected: str) -> float:
        pred = set(predicted.lower().split())
        exp = set(expected.lower().split())
        if not exp:
            return 1.0 if not pred else 0.0
        return len(pred.intersection(exp)) / len(exp)

    def _calculate_precision_recall(self, predicted: str, expected: str) -> Tuple[float, float]:
        pred = set(predicted.lower().split())
        exp = set(expected.lower().split())
        if not pred:
            return (0.0, 0.0) if exp else (1.0, 1.0)
        inter = pred.intersection(exp)
        precision = len(inter) / len(pred)
        recall = len(inter) / len(exp) if exp else 0.0
        return precision, recall

    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

# ---------------------------
# Utilities
# ---------------------------
def setup_directories():
    for d in ["documents", "vector_store", "embeddings_cache", "logs", "chat_history"]:
        os.makedirs(d, exist_ok=True)
    print("✅ Directories setup completed")

def validate_environment():
    required_vars = ["GROQ_API_KEY"]  # extend as needed
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        return False
    print("✅ Environment validation passed")
    return True


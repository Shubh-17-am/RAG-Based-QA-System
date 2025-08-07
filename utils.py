"""
Utility functions for RAG-Based QA System
"""

import os
import re
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from config import Config

class Logger:
    """Custom logger class"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def warning(self, message: str):
        self.logger.warning(message)

class DocumentProcessor:
    """Document processing utilities"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = Logger(__name__)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.document.chunk_size,
            chunk_overlap=self.config.document.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_document(self, file_path: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """Process a single document and return chunks"""
        self.logger.info(f"Processing document: {file_path}")
        
        # Validate file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size > self.config.document.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.config.document.max_file_size})")
        
        # Get file extension
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.config.document.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Load document
        try:
            documents = self._load_document(file_path, file_ext)
            
            # Update chunk size and overlap if provided
            if chunk_size:
                self.text_splitter.chunk_size = chunk_size
            if chunk_overlap:
                self.text_splitter.chunk_overlap = chunk_overlap
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Extract text content
            text_chunks = [chunk.page_content for chunk in chunks]
            
            self.logger.info(f"Processed {file_path} into {len(text_chunks)} chunks")
            return text_chunks
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def _load_document(self, file_path: str, file_ext: str):
        """Load document based on file extension"""
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
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Remove empty lines
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
        
        return text.strip()

class ChatHistoryManager:
    """Chat history management"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.history_file = "chat_history.json"
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load chat history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def _save_history(self):
        """Save chat history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving chat history: {str(e)}")
    
    def add_exchange(self, question: str, answer: str, response_time: float, sources: List[str] = None):
        """Add a question-answer exchange to history"""
        exchange = {
            "question": question,
            "answer": answer,
            "response_time": response_time,
            "sources": sources or [],
            "timestamp": datetime.now().isoformat()
        }
        
        self.history.append(exchange)
        
        # Trim history if too long
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        self._save_history()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get chat history"""
        return self.history.copy()
    
    def clear(self):
        """Clear chat history"""
        self.history = []
        self._save_history()
    
    def search_history(self, query: str) -> List[Dict[str, Any]]:
        """Search chat history for relevant exchanges"""
        query_lower = query.lower()
        results = []
        
        for exchange in self.history:
            if (query_lower in exchange["question"].lower() or 
                query_lower in exchange["answer"].lower()):
                results.append(exchange)
        
        return results

class Evaluator:
    """Evaluation utilities for RAG system"""
    
    def __init__(self):
        self.logger = Logger(__name__)
    
    def evaluate(self, rag_system, eval_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate RAG system performance"""
        self.logger.info(f"Evaluating {len(eval_data)} questions")
        
        results = []
        total_response_time = 0
        
        for item in eval_data:
            question = item["question"]
            expected_answer = item["expected_answer"]
            
            try:
                # Get response from RAG system
                response = rag_system.query(question)
                
                # Calculate metrics
                accuracy = self._calculate_accuracy(response["answer"], expected_answer)
                precision, recall = self._calculate_precision_recall(response["answer"], expected_answer)
                f1_score = self._calculate_f1_score(precision, recall)
                
                results.append({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "response_time": response["query_time"]
                })
                
                total_response_time += response["query_time"]
                
            except Exception as e:
                self.logger.error(f"Error evaluating question '{question}': {str(e)}")
                results.append({
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "response_time": 0.0
                })
        
        # Calculate average metrics
        avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
        avg_precision = sum(r["precision"] for r in results) / len(results)
        avg_recall = sum(r["recall"] for r in results) / len(results)
        avg_f1_score = sum(r["f1_score"] for r in results) / len(results)
        avg_response_time = total_response_time / len(results)
        
        return {
            "accuracy": avg_accuracy,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": avg_f1_score,
            "avg_response_time": avg_response_time,
            "total_questions": len(eval_data)
        }
    
    def _calculate_accuracy(self, predicted: str, expected: str) -> float:
        """Calculate accuracy between predicted and expected answers"""
        # Simple word overlap accuracy
        predicted_words = set(predicted.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 1.0 if not predicted_words else 0.0
        
        intersection = predicted_words.intersection(expected_words)
        return len(intersection) / len(expected_words)
    
    def _calculate_precision_recall(self, predicted: str, expected: str) -> Tuple[float, float]:
        """Calculate precision and recall"""
        predicted_words = set(predicted.lower().split())
        expected_words = set(expected.lower().split())
        
        if not predicted_words:
            return 0.0, 0.0 if expected_words else 1.0
        
        intersection = predicted_words.intersection(expected_words)
        
        precision = len(intersection) / len(predicted_words)
        recall = len(intersection) / len(expected_words) if expected_words else 0.0
        
        return precision, recall
    
    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

class ProgressTracker:
    """Progress tracking utilities"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
    
    def update(self, step: int = 1, description: str = ""):
        """Update progress"""
        self.current_step += step
        progress = (self.current_step / self.total_steps) * 100
        elapsed_time = time.time() - self.start_time
        
        if self.current_step < self.total_steps:
            eta = (elapsed_time / self.current_step) * (self.total_steps - self.current_step)
            print(f"\rProgress: {progress:.1f}% | Step: {self.current_step}/{self.total_steps} | "
                  f"ETA: {eta:.1f}s | {description}", end="", flush=True)
        else:
            print(f"\rProgress: 100.0% | Completed in {elapsed_time:.1f}s | {description}")
    
    def finish(self):
        """Mark progress as finished"""
        self.update(0, "Completed!")

def setup_directories():
    """Setup necessary directories"""
    directories = [
        "documents",
        "vector_store",
        "embeddings_cache",
        "logs",
        "chat_history"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directories setup completed")

def validate_environment():
    """Validate environment and dependencies"""
    required_vars = ["GROQ_API_KEY"]  # Add other required variables
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ Environment validation passed")
    return True
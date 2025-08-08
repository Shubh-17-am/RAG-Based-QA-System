"""
Enhanced RAG-Based QA System - CLI Interface
Features: Interactive mode, batch processing, evaluation metrics
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Optional
from datetime import datetime
import json
import readline  # For better CLI input handling

# Import our RAG system components
from rag_system import RAGSystem
from config import Config
from utils import DocumentProcessor, ChatHistoryManager, Evaluator

class RAGCLI:
    def __init__(self, config_path: str = None):
        """Initialize the CLI interface"""
        self.config = Config(config_path)
        self.rag_system = RAGSystem(self.config)
        self.chat_history = ChatHistoryManager()
        self.evaluator = Evaluator()
        self.documents_processed = []
        
    def process_documents(self, document_paths: List[str], force_reprocess: bool = False):
        """Process documents with progress tracking"""
        print("📄 Processing documents...")
        
        for doc_path in document_paths:
            if not os.path.exists(doc_path):
                print(f"❌ Document not found: {doc_path}")
                continue
                
            try:
                print(f"🔄 Processing: {doc_path}")
                
                # Process document
                doc_processor = DocumentProcessor()
                chunks = doc_processor.process_document(doc_path)
                
                # Add to RAG system
                self.rag_system.add_documents(chunks, os.path.basename(doc_path), 
                                            force_reprocess=force_reprocess)
                
                self.documents_processed.append(os.path.basename(doc_path))
                print(f"✅ Processed: {doc_path}")
                
            except Exception as e:
                print(f"❌ Error processing {doc_path}: {str(e)}")
        
        print(f"🎉 Successfully processed {len(self.documents_processed)} documents!")
        
    def interactive_mode(self):
        """Start interactive question-answering mode"""
        if not self.documents_processed:
            print("❌ No documents processed. Please process documents first.")
            return
            
        print("\n🤖 RAG-Based QA System - Interactive Mode")
        print("💬 Type your questions and press Enter")
        print("📝 Type 'history' to see chat history")
        print("📊 Type 'stats' to see system statistics")
        print("🗑️  Type 'clear' to clear chat history")
        print("🚪 Type 'exit' or 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n❓ Your Question: ").strip()
                
                if question.lower() in ['exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                    
                elif question.lower() == 'history':
                    self.show_chat_history()
                    continue
                    
                elif question.lower() == 'stats':
                    self.show_statistics()
                    continue
                    
                elif question.lower() == 'clear':
                    self.chat_history.clear()
                    print("🗑️  Chat history cleared!")
                    continue
                    
                elif not question:
                    continue
                    
                # Process question
                self.process_question(question)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def process_question(self, question: str):
        """Process a single question"""
        print("🤔 Thinking...")
        
        start_time = time.time()
        
        try:
            # Get answer from RAG system
            response = self.rag_system.query(question)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Display answer
            print("\n📢 Answer:")
            print("-" * 30)
            print(response["answer"])
            print("-" * 30)
            
            # Display sources if available
            if response.get("sources"):
                print("\n📚 Sources:")
                for i, source in enumerate(response["sources"], 1):
                    print(f"  {i}. {source}")
            
            # Display metrics
            print(f"\n⏱️  Response time: {response_time:.2f} seconds")
            
            # Add to chat history
            self.chat_history.add_exchange(question, response["answer"], 
                                          response_time, response.get("sources", []))
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    def show_chat_history(self):
        """Display chat history"""
        history = self.chat_history.get_history()
        
        if not history:
            print("📝 No chat history yet.")
            return
            
        print("\n📝 Chat History:")
        print("=" * 50)
        
        for i, exchange in enumerate(history, 1):
            print(f"\n--- Exchange {i} ---")
            print(f"❓ Question: {exchange['question']}")
            print(f"📢 Answer: {exchange['answer']}")
            print(f"⏱️  Response time: {exchange['response_time']:.2f}s")
            if exchange.get('sources'):
                print(f"📚 Sources: {', '.join(exchange['sources'])}")
            print(f"🕐 Timestamp: {exchange['timestamp']}")
        
        print("=" * 50)
    
    def show_statistics(self):
        """Display system statistics"""
        stats = {
            "documents_processed": len(self.documents_processed),
            "chat_exchanges": len(self.chat_history.get_history()),
            "total_response_time": sum(ex["response_time"] for ex in self.chat_history.get_history()),
            "average_response_time": 0
        }
        
        if stats["chat_exchanges"] > 0:
            stats["average_response_time"] = stats["total_response_time"] / stats["chat_exchanges"]
        
        print("\n📊 System Statistics:")
        print("=" * 30)
        print(f"📄 Documents processed: {stats['documents_processed']}")
        print(f"💬 Chat exchanges: {stats['chat_exchanges']}")
        print(f"⏱️  Total response time: {stats['total_response_time']:.2f}s")
        print(f"📈 Average response time: {stats['average_response_time']:.2f}s")
        print("=" * 30)
    
    def batch_mode(self, questions_file: str, output_file: str = None):
        """Process questions in batch mode"""
        if not os.path.exists(questions_file):
            print(f"❌ Questions file not found: {questions_file}")
            return
        
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
            
            print(f"📋 Processing {len(questions)} questions in batch mode...")
            
            results = []
            for i, question in enumerate(questions, 1):
                print(f"🔄 Processing question {i}/{len(questions)}: {question[:50]}...")
                
                try:
                    response = self.rag_system.query(question)
                    results.append({
                        "question": question,
                        "answer": response["answer"],
                        "sources": response.get("sources", [])
                    })
                except Exception as e:
                    results.append({
                        "question": question,
                        "answer": f"Error: {str(e)}",
                        "sources": []
                    })
            
            # Save results
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"✅ Results saved to: {output_file}")
            
            print("🎉 Batch processing completed!")
            
        except Exception as e:
            print(f"❌ Error in batch mode: {str(e)}")
    
    def evaluation_mode(self, eval_file: str):
        """Run evaluation mode with ground truth data"""
        if not os.path.exists(eval_file):
            print(f"❌ Evaluation file not found: {eval_file}")
            return
        
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            print("🔍 Running evaluation...")
            
            metrics = self.evaluator.evaluate(self.rag_system, eval_data)
            
            print("\n📊 Evaluation Results:")
            print("=" * 40)
            print(f"🎯 Accuracy: {metrics['accuracy']:.2%}")
            print(f"📏 Precision: {metrics['precision']:.2%}")
            print(f"📐 Recall: {metrics['recall']:.2%}")
            print(f"📊 F1 Score: {metrics['f1_score']:.2%}")
            print(f"⏱️  Average response time: {metrics['avg_response_time']:.2f}s")
            print("=" * 40)
            
        except Exception as e:
            print(f"❌ Error in evaluation mode: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="RAG-Based QA System CLI")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--documents", "-d", nargs="+", help="Paths to documents to process")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing of documents")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive mode")
    parser.add_argument("--batch", "-b", help="Path to questions file for batch processing")
    parser.add_argument("--output", "-o", help="Output file for batch results")
    parser.add_argument("--evaluate", "-e", help="Path to evaluation file")
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = RAGCLI(args.config)
    
    # Process documents if provided
    if args.documents:
        cli.process_documents(args.documents, args.force_reprocess)
    
    # Start interactive mode
    if args.interactive:
        cli.interactive_mode()
    
    # Batch processing
    if args.batch:
        cli.batch_mode(args.batch, args.output)
    
    # Evaluation mode
    if args.evaluate:
        cli.evaluation_mode(args.evaluate)
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        print("\n💡 Example usage:")
        print("  python main.py -d doc1.pdf doc2.txt -i")
        print("  python main.py --batch questions.txt --output results.json")
        print("  python main.py --evaluate eval_data.json")

if __name__ == "__main__":
    main()
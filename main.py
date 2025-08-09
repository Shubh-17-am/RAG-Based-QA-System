import argparse
import os
from typing import List, Dict, Any
from rag_system import RAGSystem
from config import Config
from utils import DocumentProcessor, ChatHistoryManager

class RAGCLI:
    def __init__(self):
        self.config = Config()
        self.rag_system = RAGSystem(self.config)
        self.history_manager = ChatHistoryManager(max_history=100)
        self.conversation_memory = []

    def process_documents(self, docs: List[str], force_reprocess: bool = False):
        """Process and ingest documents into the RAG system"""
        print(f"📄 Processing {len(docs)} document(s)...")
        dp = DocumentProcessor(self.config)
        
        processed_count = 0
        for doc in docs:
            if os.path.exists(doc):
                try:
                    chunks = dp.process_document(doc)
                    if not chunks:
                        print(f"⚠️  No content extracted from {doc}. Document might be empty or corrupted.")
                        continue
                        
                    self.rag_system.add_documents(chunks, os.path.basename(doc), force_reprocess=force_reprocess)
                    processed_count += 1
                    print(f"✅ Processed {doc} ({len(chunks)} chunks)")
                except Exception as e:
                    print(f"❌ Error processing {doc}: {str(e)}")
            else:
                print(f"❌ File not found: {doc}")
        
        # Verify ingestion
        doc_count = self.rag_system.get_document_count()
        chunk_count = self.rag_system.get_chunk_count()
        print(f"\n📊 Ingestion Summary:")
        print(f"   - Documents processed: {processed_count}/{len(docs)}")
        print(f"   - Total documents in system: {doc_count}")
        print(f"   - Total chunks in system: {chunk_count}")

    def ask(self, question: str):
        """Ask a question to the RAG system with enhanced output"""
        print(f"\n🔍 Query: {question}")
        
        # Query the RAG system
        res = self.rag_system.query(
            question,
            k=self.config.rag.retriever_k,
            top_n_after_rerank=self.config.rag.top_n_after_rerank
        )
        
        # Display answer
        print(f"\n📢 Answer:")
        print(res["answer"])
        
        # Display sources
        if res["sources"]:
            print(f"\n📚 Sources: {', '.join(res['sources'])}")
        else:
            print("\n📚 No sources identified")
        
        # Display query time
        print(f"\n⏱️  Query time: {res['query_time']:.2f} seconds")
        
        # Display retrieved context
        if res["context_used"]:
            print("\n📄 Retrieved Context:")
            for i, ctx in enumerate(res["context_used"]):
                print(f"\n--- Chunk {i+1} ---")
                print(ctx[:300] + "..." if len(ctx) > 300 else ctx)
        else:
            print("\n📄 No context was retrieved for this query")
        
        # Display conversation memory
        print("\n💬 Conversation Memory:")
        memory_content = self.rag_system.get_conversation_history()
        print(memory_content if memory_content else "No conversation history")
        
        # Update conversation memory
        self.conversation_memory.append((question, res["answer"]))
        if len(self.conversation_memory) > 10:
            self.conversation_memory = self.conversation_memory[-10:]
            
        # Save to persistent history (if enabled)
        if self.config.app.auto_save:
            self.history_manager.add_exchange(question, res["answer"], res["query_time"], res["sources"])

    def show_history(self):
        """Display chat history with detailed information"""
        hist = self.history_manager.get_history()
        if not hist:
            print("No conversation history found.")
            return
            
        print(f"\n💬 Chat History ({len(hist)} exchanges):")
        for i, h in enumerate(hist, 1):
            print(f"\n--- Exchange {i} ---")
            print(f"Q: {h['question']}")
            print(f"A: {h['answer']}")
            print(f"Sources: {', '.join(h['sources']) if h['sources'] else 'None'}")
            print(f"Response time: {h['response_time']:.2f}s")
            print(f"Timestamp: {h['timestamp']}")

    def show_status(self):
        """Show current vector store status"""
        doc_count = self.rag_system.get_document_count()
        chunk_count = self.rag_system.get_chunk_count()
        print(f"\n📊 FAISS Vector Store Status:")
        print(f"   - Documents: {doc_count}")
        print(f"   - Chunks: {chunk_count}")
        print(f"   - In-memory: Yes")
        print(f"   - LLM Provider: {self.rag_system.current_llm_provider}")

    def clear_memory(self):
        """Clear conversation memory and history"""
        self.conversation_memory.clear()
        self.rag_system.clear_memory()
        self.history_manager.clear()
        print("✅ Conversation memory and history cleared")

    def clear_vector_store(self):
        """Clear the in-memory vector store"""
        self.rag_system.clear_documents()
        print("✅ In-memory FAISS vector store cleared")

def main():
    parser = argparse.ArgumentParser(description="RAG-Based QA System CLI (FAISS In-Memory)")
    parser.add_argument("-d", "--documents", nargs="+", help="Documents to process")
    parser.add_argument("-q", "--question", help="Question to ask")
    parser.add_argument("--force", action="store_true", help="Force reprocess documents")
    parser.add_argument("--history", action="store_true", help="Show chat history")
    parser.add_argument("--status", action="store_true", help="Show vector store status")
    parser.add_argument("--clear", action="store_true", help="Clear conversation memory")
    parser.add_argument("--clear-store", action="store_true", help="Clear vector store")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    cli = RAGCLI()
    
    if args.debug:
        print("🔧 Debug mode enabled")
        print(f"Configuration: LLM={cli.config.llm.default_provider}, "
              f"Retriever_k={cli.config.rag.retriever_k}, "
              f"Top_n={cli.config.rag.top_n_after_rerank}")
    
    if args.clear:
        cli.clear_memory()
    
    if args.clear_store:
        cli.clear_vector_store()
    
    if args.documents:
        cli.process_documents(args.documents, force_reprocess=args.force)
    
    if args.question:
        cli.ask(args.question)
    
    if args.history:
        cli.show_history()
    
    if args.status:
        cli.show_status()
    
    if args.interactive:
        print("\n🤖 RAG-Based QA System - Interactive Mode (FAISS In-Memory)")
        print("Type 'exit' to quit, 'clear' to clear history, 'status' for status, 'help' for commands")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    cli.clear_memory()
                elif user_input.lower() == 'clear store':
                    cli.clear_vector_store()
                elif user_input.lower() == 'status':
                    cli.show_status()
                elif user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  exit/quit    - Exit the program")
                    print("  clear        - Clear conversation history")
                    print("  clear store  - Clear vector store")
                    print("  status       - Show vector store status")
                    print("  help         - Show this help message")
                    print("  <question>   - Ask a question")
                elif user_input:
                    cli.ask(user_input)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
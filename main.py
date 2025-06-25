from answer import answer_user_question
from ingest import ingest_documents

print("\n📄 RAG-Based QA System is initializing and indexing documents...")
ingest_documents()

print("\n🤖 System is ready! You can now ask questions based on your uploaded documents.")

while True:
    query = input("\nYour Question (or type 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        print("\n👋 Exiting. Thanks for using the RAG-Based QA System!")
        break
    response = answer_user_question(query)
    print("\n📢 Answer:\n", response)

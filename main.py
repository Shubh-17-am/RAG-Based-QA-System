from answer import answer_user_question
from ingest import ingest_documents

print("\n📘 AI Tutor Bot is starting up...")
ingest_documents()

print("\n🤖 Ready to help. Ask your questions below:")

while True:
    query = input("\nYour Question (or 'exit'): ")
    if query.lower() in ["exit", "quit"]:
        break
    response = answer_user_question(query)
    print("\nAnswer:\n", response)

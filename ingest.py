import os
from text_processor import process_document
from vector_db import index_documents

SUPPORTED_FORMATS = (".txt", ".pdf", ".docx")

def ingest_documents(doc_folder="documents"):
    for filename in os.listdir(doc_folder):
        if filename.lower().endswith(SUPPORTED_FORMATS):
            path = os.path.join(doc_folder, filename)
            print(f"📄 Ingesting: {filename}")
            chunks = process_document(path)
            index_documents(chunks)

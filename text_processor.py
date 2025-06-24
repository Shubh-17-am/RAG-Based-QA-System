import os
import fitz  # PyMuPDF
from docx import Document

from typing import List

def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def read_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def split_into_chunks(text: str, chunk_size=500, overlap=100) -> List[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def process_document(file_path: str) -> List[str]:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        raw_text = read_txt(file_path)
    elif ext == ".pdf":
        raw_text = read_pdf(file_path)
    elif ext == ".docx":
        raw_text = read_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Optional preprocessing: strip extra whitespace
    cleaned_text = raw_text.strip().replace("\n", " ")
    chunks = split_into_chunks(cleaned_text)
    return chunks

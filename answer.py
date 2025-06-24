from vector_db import get_top_k_chunks
from llm_generator import generate_answer

def answer_user_question(query):
    chunks = get_top_k_chunks(query)
    return generate_answer(query, chunks)

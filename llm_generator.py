from langchain_groq.chat_models import ChatGroq
from langchain.prompts import PromptTemplate
from config import Config

# Initialize the language model
llm = ChatGroq(
    groq_api_key=Config.GROQ_API_KEY,
    model_name=Config.LLM_MODEL
)

# Define the prompt template
prompt_template = PromptTemplate.from_template(
    template="""
You are an intelligent AI tutor helping students learn any topic from provided study material. 
Use only the context provided below to answer the question.

Context: {context}

Question: {question}

Answer:
"""
)

# Function to generate the answer
def generate_answer(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = prompt_template.format(context=context, question=query)
    return llm.invoke(prompt).content.strip()

import requests
import os
from dotenv import load_dotenv
from src.retriever import retrieve

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"


def ask(question: str, top_k: int = 5) -> tuple[str, list[dict]]:
    chunks = retrieve(question, top_k=top_k)

    context = "\n\n---\n\n".join([c["text"] for c in chunks])

    prompt = f"""You are an expert assistant for machine learning research papers.
Answer the question based on the provided paper excerpts. Cite the source paper title when referencing specific content.
If the answer cannot be found in the excerpts, say so clearly.

Paper excerpts:
{context}

Question: {question}

Answer:"""

    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"], chunks

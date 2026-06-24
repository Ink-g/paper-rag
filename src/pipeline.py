import os

from anthropic import Anthropic
from dotenv import load_dotenv

from src.retriever import retrieve

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

CLAUDE_MODEL = "claude-sonnet-4-6"


def ask(question: str, top_k: int = 5) -> tuple[str, list[dict]]:
    """
    Retrieve relevant paper chunks and generate an answer using Claude.
    """

    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return "No relevant paper excerpts were found.", []

    context = "\n\n---\n\n".join(
        chunk["text"] for chunk in chunks
    )

    system_prompt = """
You are an expert assistant for machine learning research papers.

Rules:
1. Answer ONLY using the provided paper excerpts.
2. Cite the source paper title whenever possible.
3. If the answer is not present in the excerpts, explicitly say so.
4. Do not invent information.
5. Be concise but technically accurate.
"""

    user_prompt = f"""
Paper excerpts:

{context}

Question:
{question}

Answer:
"""

    message = client.messages.create(
        model=CLAUDE_MODEL,
        system=system_prompt,
        max_tokens=4096,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    answer = ""

    for block in message.content:
        if hasattr(block, "text"):
            answer += block.text

    return answer.strip(), chunks
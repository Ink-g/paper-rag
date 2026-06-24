import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

from src.retriever import retrieve

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
CLAUDE_MODEL = "claude-sonnet-4-6"


def ask(question: str, top_k: int = 5) -> tuple[str, list[dict], list[str]]:
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return "No relevant paper excerpts were found.", [], []

    context = "\n\n---\n\n".join(chunk["text"] for chunk in chunks)

    system_prompt = """You are an expert assistant for machine learning research papers.

Rules:
1. Answer ONLY using the provided paper excerpts.
2. Cite the source paper title whenever possible.
3. If the answer is not present in the excerpts, explicitly say so.
4. Do not invent information.
5. Be concise but technically accurate.

After your answer, write exactly this separator and then a JSON array of 3 follow-up questions:
<<<FOLLOWUPS>>>
["follow-up question 1", "follow-up question 2", "follow-up question 3"]

The follow-up questions should be specific, interesting, and naturally extend the current topic."""

    user_prompt = f"""Paper excerpts:

{context}

Question:
{question}"""

    message = client.messages.create(
        model=CLAUDE_MODEL,
        system=system_prompt,
        max_tokens=4096,
        temperature=0,
        messages=[{"role": "user", "content": user_prompt}]
    )

    full_response = message.content[0].text

    if "<<<FOLLOWUPS>>>" in full_response:
        answer_part, followup_part = full_response.split("<<<FOLLOWUPS>>>", 1)
        try:
            follow_ups = json.loads(followup_part.strip())
        except Exception:
            follow_ups = []
    else:
        answer_part = full_response
        follow_ups = []

    return answer_part.strip(), chunks, follow_ups

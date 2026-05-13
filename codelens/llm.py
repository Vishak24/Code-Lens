import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SYSTEM_PROMPT = (
    "You are a code review assistant. Answer questions about the provided code context.\n"
    "- Cite specific file names and line numbers when relevant.\n"
    "- Be concise and precise.\n"
    "- If the answer is not found in the provided context, say \"not found in context\".\n"
    "- Do not reference code that isn't present in the context."
)


@lru_cache(maxsize=1)
def load_groq_client() -> Groq:
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))


def ask(
    query: str,
    context: str,
    model: str = "llama-3.1-8b-instant",
) -> dict:
    client = load_groq_client()
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "answer": response.choices[0].message.content,
        "tokens_used": response.usage.total_tokens,
        "latency_ms": latency_ms,
    }


def generate_followups(query: str, answer: str) -> list[str]:
    client = load_groq_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": (
                    "Generate exactly 3 follow-up questions based on this Q&A. "
                    "One question per line, no bullets or numbers.\n\n"
                    f"Q: {query}\nA: {answer}"
                ),
            }
        ],
    )
    text = response.choices[0].message.content.strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines[:3]

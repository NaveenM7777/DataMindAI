import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("GROQ_API_KEY")

_MODEL_NAME = "llama-3.3-70b-versatile"

_client = None


def _ensure_configured():

    global _client

    if not _API_KEY:

        raise ValueError(
            "GROQ_API_KEY not found. Please add it to your .env file."
        )

    if _client is None:

        _client = Groq(api_key=_API_KEY)


def ask_with_context(question, retrieved_chunks, chat_history=None):
    """
    Sends a question to Groq, grounded in retrieved document chunks.

    Args:
        question: str — the user's question
        retrieved_chunks: list of dicts [{"chunk_id", "page", "text", "distance"}, ...]
        chat_history: optional list of {"role": "user"/"assistant", "content": str}

    Returns:
        str — the model's answer
    """

    _ensure_configured()

    if not retrieved_chunks:

        context_text = "No relevant content was found in the document."

    else:

        context_parts = []

        for chunk in retrieved_chunks:

            context_parts.append(
                f"[Chunk {chunk['chunk_id']}, Page {chunk['page']}]\n{chunk['text']}"
            )

        context_text = "\n\n".join(context_parts)

    system_instruction = f"""You are a document assistant. Answer the user's question using
ONLY the retrieved document excerpts below. If the excerpts don't contain enough
information to answer confidently, say so clearly instead of guessing.

When relevant, mention which chunk/page the information came from.

=== RETRIEVED DOCUMENT EXCERPTS ===
{context_text}
=== END EXCERPTS ===
"""

    messages = [{"role": "system", "content": system_instruction}]

    if chat_history:

        for turn in chat_history:

            role = "user" if turn["role"] == "user" else "assistant"

            messages.append({
                "role": role,
                "content": turn["content"]
            })

    messages.append({
        "role": "user",
        "content": question
    })

    response = _client.chat.completions.create(
        model=_MODEL_NAME,
        messages=messages,
        temperature=0.3,
        max_tokens=1024
    )

    return response.choices[0].message.content
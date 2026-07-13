import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("GROQ_API_KEY")

_MODEL_NAME = "llama-3.1-8b-instant"
_client = None


def _ensure_configured():

    global _client

    if not _API_KEY:

        raise ValueError(
            "GROQ_API_KEY not found. Please add it to your .env file."
        )

    if _client is None:

        _client = Groq(api_key=_API_KEY)


def send_message(system_context, chat_history, new_message):
    """
    system_context: str — dataset/app context to ground the model
    chat_history: list of dicts [{"role": "user"/"assistant", "content": "..."}]
    new_message: str — the user's latest question

    Returns: str (the model's reply) or raises Exception on failure.
    """

    _ensure_configured()

    # Groq's API uses OpenAI-style message format:
    # [{"role": "system"/"user"/"assistant", "content": "..."}]

    messages = [{"role": "system", "content": system_context}]

    for turn in chat_history:

        role = "user" if turn["role"] == "user" else "assistant"

        messages.append({
            "role": role,
            "content": turn["content"]
        })

    messages.append({
        "role": "user",
        "content": new_message
    })

    response = _client.chat.completions.create(
        model=_MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=1024
    )

    return response.choices[0].message.content
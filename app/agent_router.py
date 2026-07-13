import os
from groq import Groq
from dotenv import load_dotenv

from app.agent_tools import TOOL_REGISTRY, execute_tool

load_dotenv()

_API_KEY = os.getenv("GROQ_API_KEY")

_MODEL_NAME = "llama-3.3-70b-versatile"

_client = None


def _ensure_configured():

    global _client

    if not _API_KEY:

        raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")

    if _client is None:

        _client = Groq(api_key=_API_KEY)


def _build_tools_schema():

    tools_schema = []

    for tool_name, tool_info in TOOL_REGISTRY.items():

        tools_schema.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_info["description"],
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        })

    return tools_schema


def _synthesize_final_answer(user_request, tool_results, conversation_history=None):

    combined_context = "\n\n".join([

        f"--- Result from {TOOL_REGISTRY[r['tool']]['description'] if r['tool'] in TOOL_REGISTRY else r['tool']} ---\n{r['result']}"

        for r in tool_results

    ])

    history_text = ""

    if conversation_history:

        history_lines = []

        for turn in conversation_history[-6:]:  # last 6 turns for context, keeps prompt size reasonable

            role = "User" if turn["role"] == "user" else "Assistant"

            history_lines.append(f"{role}: {turn['content']}")

        history_text = "\n".join(history_lines)

    synthesis_prompt = f"""Recent conversation so far:
{history_text}

The user's latest request: "{user_request}"

Multiple tools were used to gather information. Here are their raw results:

{combined_context}

Write one single, clear, well-organized answer for the user that combines the
relevant information from these results, taking into account the conversation
context above (e.g. if the user is following up on something previously discussed).
Do not just repeat both results separately — actually integrate them into a
coherent response that directly addresses what the user asked.
"""

    response = _client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful data science assistant that synthesizes information clearly, using conversation context when relevant."},
            {"role": "user", "content": synthesis_prompt}
        ],
        temperature=0.3,
        max_tokens=1024
    )

    return response.choices[0].message.content


def route_request(df, user_request, target_column=None, conversation_history=None):
    """
    Sends the user's request to Groq with tool definitions AND recent
    conversation history, so follow-up questions are routed with context
    in mind rather than treated as fully independent requests.

    conversation_history: list of {"role": "user"/"assistant", "content": str}
    """

    _ensure_configured()

    tools_schema = _build_tools_schema()

    system_prompt = (
        "You are the routing brain of DataMind AI, a data science platform. "
        "Based on the user's request AND the recent conversation context, choose "
        "the tool(s) needed to fulfill it. Most requests need only ONE tool. Only "
        "choose TWO tools if the request clearly needs information from two "
        "different sources. Never choose more than two tools. If the user's "
        "request is a follow-up (e.g. 'explain that more', 'what about the other one'), "
        "use the conversation history to understand what they're referring to before "
        "picking a tool. Always call at least one tool — do not answer directly "
        "without using a tool."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Include recent conversation history so the router understands follow-ups
    if conversation_history:

        for turn in conversation_history[-6:]:  # last 6 turns, keeps context relevant and prompt small

            role = "user" if turn["role"] == "user" else "assistant"

            messages.append({
                "role": role,
                "content": turn["content"]
            })

    messages.append({
        "role": "user",
        "content": user_request
    })

    try:

        response = _client.chat.completions.create(
            model=_MODEL_NAME,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=512
        )

        message = response.choices[0].message

        if not message.tool_calls:

            result = execute_tool("ai_chat", df, user_request)

            return {"tools_used": ["ai_chat (default fallback)"], "result": result}

        tool_calls = message.tool_calls[:2]

        tool_results = []

        for tool_call in tool_calls:

            tool_name = tool_call.function.name

            result = execute_tool(tool_name, df, user_request, target_column)

            if isinstance(result, dict) and result.get("needs_target_selection"):

                return {
                    "needs_target_selection": True,
                    "available_columns": result["available_columns"],
                    "tools_used": [tool_name]
                }

            tool_results.append({"tool": tool_name, "result": result})

        if len(tool_results) == 1:

            return {
                "tools_used": [tool_results[0]["tool"]],
                "result": tool_results[0]["result"]
            }

        final_answer = _synthesize_final_answer(user_request, tool_results, conversation_history)

        return {
            "tools_used": [r["tool"] for r in tool_results],
            "result": final_answer,
            "raw_tool_results": tool_results
        }

    except Exception as e:

        return {"tools_used": [], "result": f"❌ Agent routing failed: {e}"}
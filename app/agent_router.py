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


def _synthesize_final_answer(user_request, tool_results):
    """
    Combines outputs from multiple tools into one coherent answer.

    tool_results: list of dicts [{"tool": str, "result": str}, ...]
    """

    combined_context = "\n\n".join([

        f"--- Result from {TOOL_REGISTRY[r['tool']]['description'] if r['tool'] in TOOL_REGISTRY else r['tool']} ---\n{r['result']}"

        for r in tool_results

    ])

    synthesis_prompt = f"""The user asked: "{user_request}"

Multiple tools were used to gather information. Here are their raw results:

{combined_context}

Write one single, clear, well-organized answer for the user that combines the
relevant information from these results. Do not just repeat both results
separately — actually integrate them into a coherent response that directly
addresses what the user asked.
"""

    response = _client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful data science assistant that synthesizes information clearly."},
            {"role": "user", "content": synthesis_prompt}
        ],
        temperature=0.3,
        max_tokens=1024
    )

    return response.choices[0].message.content


def route_request(df, user_request, target_column=None):
    """
    Sends the user's request to Groq with tool definitions, lets the model
    pick ONE or TWO tools, executes them, and returns the combined result.

    Returns: dict {"tools_used": list of str, "result": str}
    or, if ML needs a target column: dict with "needs_target_selection"
    """

    _ensure_configured()

    tools_schema = _build_tools_schema()

    system_prompt = (
        "You are the routing brain of DataMind AI, a data science platform. "
        "Based on the user's request, choose the tool(s) needed to fulfill it. "
        "Most requests need only ONE tool. Only choose TWO tools if the request "
        "clearly needs information from two different sources — for example, "
        "explaining ML results using a PDF's requirements, or combining dataset "
        "insights with document content. Never choose more than two tools. "
        "Always call at least one tool — do not answer directly without a tool."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request}
    ]

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

        # Limit to a maximum of 2 tool calls, even if the model suggests more
        tool_calls = message.tool_calls[:2]

        tool_results = []

        for tool_call in tool_calls:

            tool_name = tool_call.function.name

            result = execute_tool(tool_name, df, user_request, target_column)

            # Check if ML tool needs target column — pause immediately if so
            if isinstance(result, dict) and result.get("needs_target_selection"):

                return {
                    "needs_target_selection": True,
                    "available_columns": result["available_columns"],
                    "tools_used": [tool_name]
                }

            tool_results.append({"tool": tool_name, "result": result})

        # ---- Single tool: return directly, no synthesis needed ----

        if len(tool_results) == 1:

            return {
                "tools_used": [tool_results[0]["tool"]],
                "result": tool_results[0]["result"]
            }

        # ---- Two tools: synthesize combined answer ----

        final_answer = _synthesize_final_answer(user_request, tool_results)

        return {
            "tools_used": [r["tool"] for r in tool_results],
            "result": final_answer,
            "raw_tool_results": tool_results
        }

    except Exception as e:

        return {"tools_used": [], "result": f"❌ Agent routing failed: {e}"}
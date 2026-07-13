import streamlit as st

from app.agent_router import route_request
from app.agent_tools import TOOL_REGISTRY


_TOOL_LABELS = {
    "ai_chat": "💬 AI Chat",
    "rag": "📄 RAG (PDF)",
    "ml_quick": "🤖 Machine Learning",
    "eda_summary": "📊 EDA",
    "preprocessing_summary": "🧹 Preprocessing"
}


def _format_tool_labels(tool_names):

    labels = [_TOOL_LABELS.get(name, name) for name in tool_names]

    return " + ".join(labels)


def show_agent_chat(df):

    st.title("🤖 AI Agent")

    st.markdown("---")

    st.caption(
        "Ask a question or give a task. The Agent will automatically pick "
        "one or two tools needed — Machine Learning, EDA, Preprocessing, "
        "AI Chat, or RAG — and combine the results if needed. It also "
        "remembers recent conversation context for follow-up questions."
    )

    with st.expander("🧰 Available Tools"):

        for tool_name, tool_info in TOOL_REGISTRY.items():

            label = _TOOL_LABELS.get(tool_name, tool_name)

            st.markdown(f"**{label}** — {tool_info['description']}")

    st.markdown("---")

    if "agent_history" not in st.session_state:

        st.session_state["agent_history"] = []

    for turn in st.session_state["agent_history"]:

        with st.chat_message(turn["role"]):

            st.markdown(turn["content"])

            if turn["role"] == "assistant" and "tools_used" in turn:

                label = _format_tool_labels(turn["tools_used"])

                st.caption(f"🔧 Tool(s) used: {label}")

    if st.session_state.get("agent_awaiting_target"):

        st.warning("🎯 I need to know which column to predict before training.")

        available_columns = st.session_state["agent_available_columns"]

        selected_target = st.selectbox(
            "Select the target column to predict:",
            available_columns,
            key="agent_target_select"
        )

        if st.button("✅ Confirm Target and Train"):

            pending_question = st.session_state["agent_pending_question"]

            with st.spinner("Training with your selected target column..."):

                response = route_request(
                    df,
                    pending_question,
                    target_column=selected_target,
                    conversation_history=st.session_state["agent_history"]
                )

            with st.chat_message("assistant"):

                st.markdown(response["result"])

                label = _format_tool_labels(response["tools_used"])

                st.caption(f"🔧 Tool(s) used: {label}")

            st.session_state["agent_history"].append({
                "role": "assistant",
                "content": response["result"],
                "tools_used": response["tools_used"]
            })

            st.session_state["agent_awaiting_target"] = False
            st.session_state["agent_pending_question"] = None
            st.session_state["agent_available_columns"] = None

            st.rerun()

        return

    user_request = st.chat_input("Ask the Agent to do something...")

    if user_request:

        st.session_state["agent_history"].append({
            "role": "user",
            "content": user_request
        })

        with st.chat_message("user"):
            st.markdown(user_request)

        with st.chat_message("assistant"):

            with st.spinner("Agent is deciding which tool(s) to use..."):

                # Pass history EXCLUDING the message just added, since it's
                # sent separately as the current request
                history_without_last = st.session_state["agent_history"][:-1]

                response = route_request(
                    df,
                    user_request,
                    conversation_history=history_without_last
                )

            if response.get("needs_target_selection"):

                st.session_state["agent_awaiting_target"] = True
                st.session_state["agent_pending_question"] = user_request
                st.session_state["agent_available_columns"] = response["available_columns"]

                st.info("🎯 Please select the target column below to continue.")

                st.rerun()

            else:

                st.markdown(response["result"])

                if response["tools_used"]:

                    label = _format_tool_labels(response["tools_used"])

                    st.caption(f"🔧 Tool(s) used: {label}")

                    if len(response["tools_used"]) > 1 and "raw_tool_results" in response:

                        with st.expander("🔍 See individual tool outputs"):

                            for r in response["raw_tool_results"]:

                                tool_label = _TOOL_LABELS.get(r["tool"], r["tool"])

                                st.markdown(f"**{tool_label}:**")

                                st.markdown(r["result"])

                                st.markdown("---")

                st.session_state["agent_history"].append({
                    "role": "assistant",
                    "content": response["result"],
                    "tools_used": response["tools_used"]
                })

    st.markdown("---")

    if st.button("🗑️ Clear Agent Chat"):

        st.session_state["agent_history"] = []
        st.session_state["agent_awaiting_target"] = False

        st.rerun()
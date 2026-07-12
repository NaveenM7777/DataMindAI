import streamlit as st

from app.context_builder import build_system_instruction
from app.grok_client import send_message


def show_grok_chat(df):

    st.title("💬 AI Assistent Chat")

    st.markdown("---")

    if df is None:

        st.info("Upload a dataset first to chat about it.")

        return

    if "chat_history" not in st.session_state:

        st.session_state["chat_history"] = []

    # ---- Render existing conversation ----

    for turn in st.session_state["chat_history"]:

        with st.chat_message(turn["role"]):

            st.markdown(turn["content"])

    # ---- Suggested starter questions ----

    st.markdown("##### 💡 Try asking:")

    col1, col2, col3 = st.columns(3)

    suggestion = None

    with col1:

        if st.button("Explain this dataset"):
            suggestion = "Explain this dataset in simple terms."

    with col2:

        if st.button("Suggest best ML model"):
            suggestion = "What is the best ML model for this dataset and why?"

    with col3:

        if st.button("Give business insights"):
            suggestion = "Give me 5 business insights from this dataset."

    # ---- Chat input ----

    user_input = st.chat_input("Ask anything about your dataset, EDA, or ML results...")

    final_input = suggestion or user_input

    if final_input:

        st.session_state["chat_history"].append({
            "role": "user",
            "content": final_input
        })

        with st.chat_message("user"):
            st.markdown(final_input)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    system_instruction = build_system_instruction(df)

                    # Exclude the message we just added from history,
                    # since it's passed separately as new_message
                    history_without_last = st.session_state["chat_history"][:-1]

                    reply = send_message(
                        system_context=system_instruction,
                        chat_history=history_without_last,
                        new_message=final_input
                    )

                except Exception as e:

                    reply = f"❌ Something went wrong: {e}"

                st.markdown(reply)

        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": reply
        })

    # ---- Clear chat button ----

    st.markdown("---")

    if st.button("🗑️ Clear Chat History"):

        st.session_state["chat_history"] = []

        st.rerun()
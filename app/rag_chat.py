import streamlit as st

from app.rag_loader import extract_text_from_pdf, get_pdf_metadata
from app.rag_chunker import chunk_text_with_metadata
from app.rag_embeddings import create_embeddings, create_query_embedding
from app.rag_vectorstore import VectorStore
from app.rag_llm import ask_with_context


def show_rag_chat():

    st.title("📄 Chat With Your PDF")

    st.markdown("---")

    # =====================================================
    # Upload
    # =====================================================

    st.subheader("📤 Upload PDF")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="rag_pdf_uploader"
    )

    if uploaded_file is not None:

        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"

        if st.session_state.get("rag_file_id") != current_file_id:

            with st.spinner("Reading PDF and building search index..."):

                try:

                    metadata = get_pdf_metadata(uploaded_file)

                    full_text, page_texts = extract_text_from_pdf(uploaded_file)

                    if not full_text or not full_text.strip():

                        st.error("❌ No extractable text found in this PDF. It may be scanned/image-based.")

                        return

                    chunks_with_meta = chunk_text_with_metadata(
                        page_texts,
                        chunk_size=500,
                        chunk_overlap=50
                    )

                    chunk_texts_only = [c["text"] for c in chunks_with_meta]

                    embeddings = create_embeddings(chunk_texts_only)

                    if embeddings is None or len(embeddings) == 0:

                        st.error("❌ Failed to create embeddings for this document.")

                        return

                    store = VectorStore(embedding_dim=embeddings.shape[1])

                    store.add(embeddings, chunks_with_meta)

                    st.session_state["rag_file_id"] = current_file_id
                    st.session_state["rag_vectorstore"] = store
                    st.session_state["rag_metadata"] = metadata
                    st.session_state["rag_chunk_count"] = len(chunks_with_meta)
                    st.session_state["rag_chat_history"] = []

                except Exception as e:

                    st.error(f"❌ Failed to process PDF: {e}")

                    return

        st.success("✅ PDF Loaded Successfully")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric("Pages", st.session_state["rag_metadata"]["page_count"])

        with col2:

            st.metric("Chunks Created", st.session_state["rag_chunk_count"])

        with col3:

            st.metric("File Size", f"{st.session_state['rag_metadata']['file_size_kb']} KB")

        st.caption("Embedding Model: **all-MiniLM-L6-v2** | Vector Store: **FAISS**")

    # =====================================================
    # Chat
    # =====================================================

    if "rag_vectorstore" not in st.session_state or st.session_state["rag_vectorstore"].is_empty():

        st.info("Upload a PDF above to start asking questions.")

        return

    st.markdown("---")

    st.subheader("💬 Ask a Question")

    if "rag_chat_history" not in st.session_state:

        st.session_state["rag_chat_history"] = []

    for turn in st.session_state["rag_chat_history"]:

        with st.chat_message(turn["role"]):

            st.markdown(turn["content"])

            if turn["role"] == "assistant" and "sources" in turn:

                with st.expander("📚 Sources Used"):

                    for src in turn["sources"]:

                        st.write(f"Chunk {src['chunk_id']} — Page {src['page']}")

    user_question = st.chat_input("Ask something about this PDF...")

    if user_question:

        st.session_state["rag_chat_history"].append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):

            with st.spinner("Searching document and thinking..."):

                try:

                    query_embedding = create_query_embedding(user_question)

                    retrieved_chunks = st.session_state["rag_vectorstore"].search(
                        query_embedding,
                        top_k=3
                    )

                    history_without_last = st.session_state["rag_chat_history"][:-1]

                    answer = ask_with_context(
                        question=user_question,
                        retrieved_chunks=retrieved_chunks,
                        chat_history=history_without_last
                    )

                except Exception as e:

                    answer = f"❌ Something went wrong: {e}"

                    retrieved_chunks = []

                st.markdown(answer)

                if retrieved_chunks:

                    with st.expander("📚 Sources Used"):

                        for src in retrieved_chunks:

                            st.write(f"Chunk {src['chunk_id']} — Page {src['page']}")

        st.session_state["rag_chat_history"].append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved_chunks
        })

    # =====================================================
    # Clear
    # =====================================================

    st.markdown("---")

    if st.button("🗑️ Clear PDF & Chat"):

        for key in ["rag_file_id", "rag_vectorstore", "rag_metadata", "rag_chunk_count", "rag_chat_history"]:

            if key in st.session_state:

                del st.session_state[key]

        st.rerun()
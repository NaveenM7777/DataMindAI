import streamlit as st
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def _get_model():

    global _model

    if _model is None:

        _model = SentenceTransformer(_MODEL_NAME)

    return _model


@st.cache_resource
def load_embedding_model():
    """
    Cached loader so the model is only loaded once per Streamlit session,
    not on every rerun (loading it repeatedly would be slow).
    """

    return SentenceTransformer(_MODEL_NAME)


def create_embeddings(chunks):
    """
    Creates embeddings for a list of text chunks.

    Args:
        chunks: list of str — the text chunks to embed

    Returns:
        numpy array of shape (num_chunks, embedding_dim)
    """

    if not chunks:

        return None

    model = load_embedding_model()

    embeddings = model.encode(
        chunks,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    return embeddings


def create_query_embedding(query):
    """
    Creates an embedding for a single query string.

    Returns:
        numpy array of shape (embedding_dim,)
    """

    model = load_embedding_model()

    embedding = model.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True
    )

    return embedding[0]
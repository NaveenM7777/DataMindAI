import faiss
import numpy as np


class VectorStore:
    """
    Wraps a FAISS index along with the original chunk metadata,
    so search results can be mapped back to their source text.
    """

    def __init__(self, embedding_dim):

        self.embedding_dim = embedding_dim

        self.index = faiss.IndexFlatL2(embedding_dim)

        self.chunk_metadata = []  # list of dicts: {"chunk_id", "page", "text"}

    def add(self, embeddings, chunk_metadata_list):
        """
        Adds embeddings and their corresponding metadata to the store.

        Args:
            embeddings: numpy array (num_chunks, embedding_dim)
            chunk_metadata_list: list of dicts, same length as embeddings
        """

        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)

        self.chunk_metadata.extend(chunk_metadata_list)

    def search(self, query_embedding, top_k=3):
        """
        Finds the top_k most similar chunks to the query embedding.

        Args:
            query_embedding: numpy array (embedding_dim,)
            top_k: int — number of results to return

        Returns:
            list of dicts: [{"chunk_id", "page", "text", "distance"}, ...]
            sorted by relevance (lowest distance first)
        """

        if self.index.ntotal == 0:

            return []

        query_vec = np.array([query_embedding]).astype("float32")

        top_k = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(query_vec, top_k)

        results = []

        for dist, idx in zip(distances[0], indices[0]):

            if idx == -1:

                continue

            metadata = self.chunk_metadata[idx].copy()

            metadata["distance"] = float(dist)

            results.append(metadata)

        return results

    def is_empty(self):

        return self.index.ntotal == 0
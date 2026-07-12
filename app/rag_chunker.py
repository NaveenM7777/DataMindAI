def chunk_text(full_text, chunk_size=500, chunk_overlap=50):
    """
    Splits text into overlapping chunks based on word count.

    Args:
        full_text: str — the complete extracted text
        chunk_size: int — approximate number of words per chunk
        chunk_overlap: int — number of words to overlap between consecutive chunks

    Returns:
        list of str — text chunks
    """

    if not full_text or not full_text.strip():

        return []

    words = full_text.split()

    if len(words) == 0:

        return []

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)

        if chunk.strip():

            chunks.append(chunk)

        # Move start forward, leaving overlap words for context continuity
        start = end - chunk_overlap

        # Safety guard against infinite loop if overlap >= chunk_size
        if chunk_size <= chunk_overlap:

            break

    return chunks


def chunk_text_with_metadata(page_texts, chunk_size=500, chunk_overlap=50):
    """
    Chunks text page-by-page, tracking which page each chunk came from.
    Useful for citing sources (e.g. "Chunk 17, Page 4").

    Args:
        page_texts: list of str — text per page, from rag_loader
        chunk_size: int
        chunk_overlap: int

    Returns:
        list of dicts: [{"chunk_id": int, "page": int, "text": str}, ...]
    """

    all_chunks = []

    chunk_id = 0

    for page_num, page_text in enumerate(page_texts, start=1):

        page_chunks = chunk_text(page_text, chunk_size, chunk_overlap)

        for chunk in page_chunks:

            all_chunks.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "text": chunk
            })

            chunk_id += 1

    return all_chunks
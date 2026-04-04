"""FAISS helpers for indexing and retrieval."""

from langchain_community.vectorstores import FAISS


def create_faiss_index(chunks, embedding_model):
    """Create a FAISS index from transcript chunks."""
    return FAISS.from_texts(chunks, embedding_model)


def retrieve(query, faiss_index, k=7):
    """
    Retrieve relevant context from the FAISS index based on the user's query.

    Parameters:
        query (str): The user's query string.
        faiss_index (FAISS): The FAISS index containing the embedded documents.
        k (int, optional): The number of most relevant documents to retrieve (default is 3).

    Returns:
        list: A list of the k most relevant documents (or document chunks).
    """
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context


def retrieve_context(query, faiss_index, k=7):
    """Retrieve relevant documents from the vector index."""
    return retrieve(query, faiss_index, k=k)


def perform_similarity_search(faiss_index, query, k=3):
    """Wrapper for quick similarity search use cases."""
    return faiss_index.similarity_search(query, k=k)

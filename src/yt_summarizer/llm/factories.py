"""Factories for LLM and embedding model instances."""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


def llm(
    model: str = "phi3:mini",
    temperature: float = 0.5,
    max_tokens: int = 256,
    timeout_seconds: float = 60.0,
):
    return ChatOllama(
        model=model,
        temperature=temperature,
        num_predict=max_tokens,
        client_kwargs={"timeout": timeout_seconds},
    )


def embedding(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    normalize_embeddings: bool = True,
):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )


def create_llm(
    model="phi3:mini", temperature=0.5, max_tokens=256, timeout_seconds=60.0
):
    """Create the chat model used by the application."""
    return llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )


def create_embedding_model(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
    normalize_embeddings=True,
):
    """Create the embedding model used for vector search."""
    return embedding(
        model_name=model_name,
        device=device,
        normalize_embeddings=normalize_embeddings,
    )

"""Compatibility wrappers for high-level workflows."""

from yt_summarizer.core import VideoRAGService
from yt_summarizer.retrieval.faiss_store import retrieve


def generate_answer(question, faiss_index, qa_chain, k=7):
    """Retrieve relevant context and generate an answer with an existing chain."""
    relevant_context = retrieve(question, faiss_index, k=k)
    return qa_chain.predict(context=relevant_context, question=question)


def summarize_video(video_url):
    """Generate a summary for a YouTube video."""
    try:
        return VideoRAGService().summarize_video(video_url)
    except ValueError as exc:
        return str(exc)


def answer_question(video_url, user_question):
    """Answer a user question about a YouTube video."""
    try:
        return VideoRAGService().answer_question(video_url, user_question)
    except ValueError as exc:
        return str(exc)

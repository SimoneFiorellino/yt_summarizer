"""Core service layer.

VideoRAGService:
    Orchestrates the application pipeline for YouTube transcript summarization
    and question answering.

summarize_video:
    Fetches and preprocesses a transcript, builds the summary chain, and
    generates a final summary.

answer_question:
    Fetches and preprocesses a transcript, chunks it, builds a FAISS index,
    retrieves relevant context, and generates an answer.

_llm:
    Creates the configured chat model used by the summary and QA chains.

_embedding_model:
    Creates the configured embedding model used by vector retrieval.
"""

from yt_summarizer.config import AppConfig, load_config
from yt_summarizer.ingestion import ingest_video
from yt_summarizer.llm.chains import create_qa_chain, create_summary_chain
from yt_summarizer.llm.factories import embedding, llm
from yt_summarizer.prompts.templates import create_qa_prompt_template, create_summary_prompt
from yt_summarizer.retrieval.faiss_store import create_faiss_index, retrieve
from yt_summarizer.transcript.processing import chunk_transcript


class VideoRAGService:
    """Application core for transcript summarization and QA.

    This class contains the service logic of the project: it connects the
    lower-level modules for ingestion, transcript processing, retrieval,
    prompt creation, and LLM calls into complete user-facing workflows.

    The API and UI layers should call this service instead of directly
    orchestrating the RAG pipeline.
    """

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()

    def summarize_video(self, video_url: str) -> str:
        """Fetch a transcript and generate a video summary."""
        ingested = ingest_video(video_url)
        summary_chain = create_summary_chain(self._llm(), create_summary_prompt())
        return summary_chain.run({"transcript": ingested.processed_text})

    def answer_question(self, video_url: str, question: str) -> str:
        """Answer a question using retrieved transcript context."""
        if not question:
            raise ValueError("Please provide a valid question.")

        ingested = ingest_video(video_url)
        chunks = chunk_transcript(
            ingested.processed_text,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        faiss_index = create_faiss_index(chunks, self._embedding_model())
        qa_chain = create_qa_chain(self._llm(), create_qa_prompt_template())
        context = retrieve(question, faiss_index, k=self.config.retrieval_top_k)
        return qa_chain.predict(context=context, question=question)

    def _llm(self):
        return llm(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )

    def _embedding_model(self):
        return embedding(
            model_name=self.config.embedding_model,
            device=self.config.embedding_device,
            normalize_embeddings=self.config.normalize_embeddings,
        )

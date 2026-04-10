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

import logging

from yt_summarizer.config import AppConfig, load_config
from yt_summarizer.errors import InvalidRequestError, ModelError, RetrievalError
from yt_summarizer.ingestion import ingest_video
from yt_summarizer.llm.chains import create_qa_chain, create_summary_chain
from yt_summarizer.llm.factories import embedding, llm
from yt_summarizer.observability import configure_logging, log_step
from yt_summarizer.prompts.templates import create_qa_prompt_template, create_summary_prompt
from yt_summarizer.retrieval.faiss_store import create_faiss_index, retrieve
from yt_summarizer.transcript.processing import chunk_transcript


logger = logging.getLogger(__name__)


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
        configure_logging(self.config)

    def summarize_video(self, video_url: str) -> str:
        """Fetch a transcript and generate a video summary."""
        logger.info("summary_pipeline_started")
        with log_step(logger, "ingestion"):
            ingested = ingest_video(video_url)

        with log_step(logger, "summary_chain_create", video_id=ingested.video_id):
            try:
                summary_chain = create_summary_chain(self._llm(), create_summary_prompt())
            except Exception as exc:
                raise ModelError("Could not initialize the summary model.") from exc

        with log_step(logger, "summary_generation", video_id=ingested.video_id):
            summary = self._run_model_call(
                lambda: summary_chain.run({"transcript": ingested.processed_text}),
                error_message="Could not generate the video summary.",
            )

        logger.info(
            "summary_pipeline_completed",
            extra={"video_id": ingested.video_id, "summary_chars": len(summary)},
        )
        return summary

    def answer_question(self, video_url: str, question: str) -> str:
        """Answer a question using retrieved transcript context."""
        if not question:
            logger.warning("invalid_question")
            raise InvalidRequestError("Please provide a valid question.")

        logger.info("qa_pipeline_started")
        with log_step(logger, "ingestion"):
            ingested = ingest_video(video_url)

        with log_step(logger, "chunking", video_id=ingested.video_id):
            chunks = chunk_transcript(
                ingested.processed_text,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        if not chunks:
            logger.warning("chunking_empty", extra={"video_id": ingested.video_id})
            raise RetrievalError("Transcript chunking produced no searchable content.")

        logger.info(
            "chunks_created",
            extra={"video_id": ingested.video_id, "chunk_count": len(chunks)},
        )

        with log_step(logger, "retrieval", video_id=ingested.video_id):
            try:
                faiss_index = create_faiss_index(chunks, self._embedding_model())
                context = retrieve(question, faiss_index, k=self.config.retrieval_top_k)
            except Exception as exc:
                raise RetrievalError("Could not retrieve relevant transcript context.") from exc

        logger.info(
            "context_retrieved",
            extra={
                "video_id": ingested.video_id,
                "retrieved_chunks": len(context),
                "top_k": self.config.retrieval_top_k,
            },
        )

        with log_step(logger, "qa_chain_create", video_id=ingested.video_id):
            try:
                qa_chain = create_qa_chain(self._llm(), create_qa_prompt_template())
            except Exception as exc:
                raise ModelError("Could not initialize the question-answering model.") from exc

        with log_step(logger, "answer_generation", video_id=ingested.video_id):
            answer = self._run_model_call(
                lambda: qa_chain.predict(context=context, question=question),
                error_message="Could not generate an answer for the question.",
            )

        logger.info(
            "qa_pipeline_completed",
            extra={"video_id": ingested.video_id, "answer_chars": len(answer)},
        )
        return answer

    def _llm(self):
        return llm(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            timeout_seconds=self.config.llm_timeout_seconds,
        )

    def _embedding_model(self):
        return embedding(
            model_name=self.config.embedding_model,
            device=self.config.embedding_device,
            normalize_embeddings=self.config.normalize_embeddings,
        )

    def _run_model_call(self, operation, error_message: str):
        """Run an LLM operation with a small retry budget and stable error output."""
        last_error = None
        attempts = max(1, self.config.llm_retry_attempts)

        for attempt in range(1, attempts + 1):
            try:
                logger.info("model_call_attempt", extra={"attempt": attempt})
                return operation()
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "model_call_failed",
                    extra={"attempt": attempt, "attempts": attempts, "error": str(exc)},
                )

        raise ModelError(error_message) from last_error

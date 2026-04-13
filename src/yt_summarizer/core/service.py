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
from yt_summarizer.core.types import QAResult, SourceAttribution
from yt_summarizer.errors import (
    ArtifactNotFoundError,
    InvalidRequestError,
    ModelError,
    RetrievalError,
)
from yt_summarizer.ingestion import ingest_video
from yt_summarizer.llm.chains import create_qa_chain, create_summary_chain
from yt_summarizer.llm.factories import embedding, llm
from yt_summarizer.observability import configure_logging, log_step
from yt_summarizer.prompts.templates import (
    create_qa_prompt_template,
    create_summary_prompt,
)
from yt_summarizer.retrieval.faiss_store import load_faiss_index, retrieve
from yt_summarizer.storage import LocalArtifactStore
from yt_summarizer.transcript.fetchers import get_video_id


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
        self.artifact_store = LocalArtifactStore(self.config.data_dir)

    def ingest_video(self, video_url: str):
        """Offline step: fetch transcript, build index, and persist artifacts."""
        logger.info("offline_ingestion_pipeline_started")
        ingested = ingest_video(video_url, config=self.config)
        logger.info(
            "offline_ingestion_pipeline_completed",
            extra={"video_id": ingested.video_id, "chunk_count": ingested.chunk_count},
        )
        return ingested

    def summarize_video(
        self, video_url: str | None = None, video_id: str | None = None
    ) -> str:
        """Online step: load processed transcript and generate a summary."""
        logger.info("summary_pipeline_started")
        resolved_video_id = self._resolve_video_id(
            video_url=video_url, video_id=video_id
        )

        with log_step(logger, "artifact_load", video_id=resolved_video_id):
            processed_text = self.artifact_store.load_processed_transcript(
                resolved_video_id
            )

        with log_step(logger, "summary_chain_create", video_id=resolved_video_id):
            try:
                summary_chain = create_summary_chain(
                    self._llm(), create_summary_prompt()
                )
            except Exception as exc:
                raise ModelError("Could not initialize the summary model.") from exc

        with log_step(logger, "summary_generation", video_id=resolved_video_id):
            summary = self._run_model_call(
                lambda: summary_chain.run({"transcript": processed_text}),
                error_message="Could not generate the video summary.",
            )

        logger.info(
            "summary_pipeline_completed",
            extra={"video_id": resolved_video_id, "summary_chars": len(summary)},
        )
        return summary

    def answer_question(
        self,
        video_url: str | None = None,
        question: str | None = None,
        video_id: str | None = None,
    ) -> QAResult:
        """Online step: load a persisted index and answer a question."""
        if not question:
            logger.warning("invalid_question")
            raise InvalidRequestError("Please provide a valid question.")

        logger.info("qa_pipeline_started")
        resolved_video_id = self._resolve_video_id(
            video_url=video_url, video_id=video_id
        )

        with log_step(logger, "artifact_load", video_id=resolved_video_id):
            metadata = self.artifact_store.load_metadata(resolved_video_id)

        with log_step(logger, "retrieval", video_id=resolved_video_id):
            try:
                faiss_index = load_faiss_index(
                    self.artifact_store.video_dir(resolved_video_id) / "faiss_index",
                    self._embedding_model(),
                )
                context = retrieve(question, faiss_index, k=self.config.retrieval_top_k)
            except Exception as exc:
                raise RetrievalError(
                    "Could not retrieve relevant transcript context."
                ) from exc

        logger.info(
            "context_retrieved",
            extra={
                "video_id": resolved_video_id,
                "retrieved_chunks": len(context),
                "top_k": self.config.retrieval_top_k,
                "chunk_count": metadata.get("chunk_count"),
            },
        )

        with log_step(logger, "qa_chain_create", video_id=resolved_video_id):
            try:
                qa_chain = create_qa_chain(self._llm(), create_qa_prompt_template())
            except Exception as exc:
                raise ModelError(
                    "Could not initialize the question-answering model."
                ) from exc

        with log_step(logger, "answer_generation", video_id=resolved_video_id):
            answer = self._run_model_call(
                lambda: qa_chain.predict(context=context, question=question),
                error_message="Could not generate an answer for the question.",
            )

        sources = [
            SourceAttribution(
                video_id=document.metadata.get("video_id", resolved_video_id),
                chunk_id=int(document.metadata.get("chunk_id", -1)),
                source=document.metadata.get("source", "youtube_transcript"),
                language=document.metadata.get(
                    "language", metadata.get("language", "en")
                ),
                start_time=float(document.metadata.get("start_time", 0.0)),
                end_time=float(document.metadata.get("end_time", 0.0)),
                source_attribution=document.metadata.get("source_attribution", ""),
                excerpt=document.page_content,
            )
            for document in context
        ]

        logger.info(
            "qa_pipeline_completed",
            extra={
                "video_id": resolved_video_id,
                "answer_chars": len(answer),
                "source_count": len(sources),
            },
        )
        return QAResult(answer=answer, sources=sources)

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

    def _resolve_video_id(
        self,
        video_url: str | None = None,
        video_id: str | None = None,
    ) -> str:
        """Resolve a request reference into the stored video ID used online."""
        if video_id:
            return video_id

        if video_url:
            resolved_video_id = get_video_id(video_url)
            if resolved_video_id:
                return resolved_video_id

        raise ArtifactNotFoundError(
            "Provide a valid video_id or a YouTube URL that has already been ingested."
        )

"""YouTube video ingestion use cases."""

from dataclasses import dataclass
import logging

from yt_summarizer.config import AppConfig, load_config
from yt_summarizer.errors import InvalidVideoURLError, TranscriptUnavailableError
from yt_summarizer.llm.factories import embedding
from yt_summarizer.observability import log_step
from yt_summarizer.retrieval.faiss_store import create_faiss_index, save_faiss_index
from yt_summarizer.storage import LocalArtifactStore
from yt_summarizer.transcript.fetchers import get_transcript, get_video_id
from yt_summarizer.transcript.processing import chunk_transcript_segments, process


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestedTranscript:
    """Normalized transcript data for a single YouTube video."""

    video_url: str
    video_id: str
    raw_transcript: object
    processed_text: str
    chunk_count: int
    language: str


def ingest_video(
    video_url: str,
    config: AppConfig | None = None,
) -> IngestedTranscript:
    """Fetch, normalize, index, and persist a YouTube transcript."""
    config = config or load_config()
    artifact_store = LocalArtifactStore(config.data_dir)

    with log_step(logger, "video_id_parse"):
        video_id = get_video_id(video_url)
        if not video_id:
            logger.warning("invalid_video_url")
            raise InvalidVideoURLError("Please provide a valid YouTube URL.")

    with log_step(logger, "transcript_fetch", video_id=video_id):
        transcript = get_transcript(video_url)
        if not transcript:
            logger.warning("transcript_missing", extra={"video_id": video_id})
            raise TranscriptUnavailableError(
                "No English transcript is available for this video."
            )

    with log_step(logger, "transcript_process", video_id=video_id):
        processed_text = process(transcript)
        if not processed_text:
            logger.warning(
                "transcript_empty_after_processing",
                extra={"video_id": video_id},
            )
            raise TranscriptUnavailableError(
                "Transcript was fetched but could not be processed."
            )

    with log_step(logger, "chunking", video_id=video_id):
        documents = chunk_transcript_segments(
            transcript,
            video_id=video_id,
            video_url=video_url,
            language="en",
            source="youtube_transcript",
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    with log_step(logger, "index_build", video_id=video_id):
        embedding_model = embedding(
            model_name=config.embedding_model,
            device=config.embedding_device,
            normalize_embeddings=config.normalize_embeddings,
        )
        faiss_index = create_faiss_index(documents, embedding_model)

    with log_step(logger, "artifact_persist", video_id=video_id):
        artifact_store.save_processed_transcript(video_id, processed_text)
        artifact_store.save_metadata(
            video_id,
            {
                "video_id": video_id,
                "video_url": video_url,
                "language": "en",
                "source": "youtube_transcript",
                "transcript_items": len(transcript),
                "processed_chars": len(processed_text),
                "chunk_count": len(documents),
            },
        )
        artifact_store.save_chunk_metadata(
            video_id,
            [
                {
                    "chunk_id": document.metadata["chunk_id"],
                    "video_id": document.metadata["video_id"],
                    "video_url": document.metadata["video_url"],
                    "language": document.metadata["language"],
                    "source": document.metadata["source"],
                    "source_attribution": document.metadata["source_attribution"],
                    "start_time": document.metadata["start_time"],
                    "end_time": document.metadata["end_time"],
                    "text": document.page_content,
                }
                for document in documents
            ],
        )
        save_faiss_index(
            faiss_index, artifact_store.video_dir(video_id) / "faiss_index"
        )

    logger.info(
        "video_ingested",
        extra={
            "video_id": video_id,
            "transcript_items": len(transcript),
            "processed_chars": len(processed_text),
            "chunk_count": len(documents),
            "language": "en",
        },
    )

    return IngestedTranscript(
        video_url=video_url,
        video_id=video_id,
        raw_transcript=transcript,
        processed_text=processed_text,
        chunk_count=len(documents),
        language="en",
    )

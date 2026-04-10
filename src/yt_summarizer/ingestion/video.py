"""YouTube video ingestion use cases."""

from dataclasses import dataclass
import logging

from yt_summarizer.errors import InvalidVideoURLError, TranscriptUnavailableError
from yt_summarizer.observability import log_step
from yt_summarizer.transcript.fetchers import get_transcript, get_video_id
from yt_summarizer.transcript.processing import process


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestedTranscript:
    """Normalized transcript data for a single YouTube video."""

    video_url: str
    video_id: str
    raw_transcript: object
    processed_text: str


def ingest_video(video_url: str) -> IngestedTranscript:
    """Fetch and normalize a YouTube transcript."""
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

    logger.info(
        "video_ingested",
        extra={
            "video_id": video_id,
            "transcript_items": len(transcript),
            "processed_chars": len(processed_text),
        },
    )

    return IngestedTranscript(
        video_url=video_url,
        video_id=video_id,
        raw_transcript=transcript,
        processed_text=processed_text,
    )

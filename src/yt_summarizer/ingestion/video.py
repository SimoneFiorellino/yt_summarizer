"""YouTube video ingestion use cases."""

from dataclasses import dataclass

from yt_summarizer.errors import InvalidVideoURLError, TranscriptUnavailableError
from yt_summarizer.transcript.fetchers import get_transcript, get_video_id
from yt_summarizer.transcript.processing import process


@dataclass(frozen=True)
class IngestedTranscript:
    """Normalized transcript data for a single YouTube video."""

    video_url: str
    video_id: str
    raw_transcript: object
    processed_text: str


def ingest_video(video_url: str) -> IngestedTranscript:
    """Fetch and normalize a YouTube transcript."""
    video_id = get_video_id(video_url)
    if not video_id:
        raise InvalidVideoURLError("Please provide a valid YouTube URL.")

    transcript = get_transcript(video_url)
    if not transcript:
        raise TranscriptUnavailableError(
            "No English transcript is available for this video."
        )

    processed_text = process(transcript)
    if not processed_text:
        raise TranscriptUnavailableError(
            "Transcript was fetched but could not be processed."
        )

    return IngestedTranscript(
        video_url=video_url,
        video_id=video_id,
        raw_transcript=transcript,
        processed_text=processed_text,
    )

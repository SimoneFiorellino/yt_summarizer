"""Utilities for fetching YouTube transcripts."""

import re

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    InvalidVideoId,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApiException,
)

from yt_summarizer.errors import (
    InvalidVideoURLError,
    TranscriptFetchError,
    TranscriptUnavailableError,
)


YOUTUBE_WATCH_URL_PATTERN = r"https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})"


def get_video_id(url):
    # Regex pattern to match YouTube video URLs
    if not url:
        return None

    match = re.search(YOUTUBE_WATCH_URL_PATTERN, url)
    return match.group(1) if match else None


def get_transcript(url):
    # Extracts the video ID from the URL
    video_id = get_video_id(url)
    if not video_id:
        raise InvalidVideoURLError("Please provide a valid YouTube URL.")

    # Create a YouTubeTranscriptApi() object
    ytt_api = YouTubeTranscriptApi()

    # Fetch the list of available transcripts for the given YouTube video
    try:
        transcripts = ytt_api.list(video_id)
    except (InvalidVideoId, VideoUnavailable) as exc:
        raise InvalidVideoURLError(
            "The provided YouTube video is invalid or unavailable."
        ) from exc
    except (NoTranscriptFound, TranscriptsDisabled) as exc:
        raise TranscriptUnavailableError(
            "No transcript is available for this YouTube video."
        ) from exc
    except YouTubeTranscriptApiException as exc:
        raise TranscriptFetchError("Could not fetch the YouTube transcript.") from exc

    transcript = ""
    try:
        for t in transcripts:
            # Check if the transcript's language is English
            if t.language_code == "en":
                if t.is_generated:
                    # If no transcript has been set yet, use the auto-generated one
                    if len(transcript) == 0:
                        transcript = t.fetch()
                else:
                    # If a manually created transcript is found, use it (overrides auto-generated)
                    transcript = t.fetch()
                    break  # Prioritize the manually created transcript, exit the loop
    except YouTubeTranscriptApiException as exc:
        raise TranscriptFetchError("Could not fetch the YouTube transcript.") from exc

    if not transcript:
        raise TranscriptUnavailableError(
            "No English transcript is available for this YouTube video."
        )

    return transcript

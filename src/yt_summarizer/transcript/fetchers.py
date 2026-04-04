"""Utilities for fetching YouTube transcripts."""

import re

from youtube_transcript_api import YouTubeTranscriptApi


YOUTUBE_WATCH_URL_PATTERN = r"https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})"


def get_video_id(url):    
    # Regex pattern to match YouTube video URLs
    match = re.search(YOUTUBE_WATCH_URL_PATTERN, url)
    return match.group(1) if match else None


def get_transcript(url):
    # Extracts the video ID from the URL
    video_id = get_video_id(url)

    # Create a YouTubeTranscriptApi() object
    ytt_api = YouTubeTranscriptApi()

    # Fetch the list of available transcripts for the given YouTube video
    transcripts = ytt_api.list(video_id)

    transcript = ""
    for t in transcripts:
        # Check if the transcript's language is English
        if t.language_code == 'en':
            if t.is_generated:
                # If no transcript has been set yet, use the auto-generated one
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                # If a manually created transcript is found, use it (overrides auto-generated)
                transcript = t.fetch()
                break  # Prioritize the manually created transcript, exit the loop

    return transcript if transcript else None

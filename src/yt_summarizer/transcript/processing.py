"""Helpers for transcript normalization and chunking."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process(transcript):
    # Initialize an empty string to hold the formatted transcript
    txt = ""

    # Loop through each entry in the transcript
    for i in transcript:
        try:
            # Append the text and its start time to the output string
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            # If there is an issue accessing 'text' or 'start', skip this entry
            pass

    # Return the processed transcript as a single string
    return txt


def format_transcript(transcript):
    """Convert transcript items into a plain text block with timestamps."""
    return process(transcript)


def build_transcript_segments(
    transcript, language: str = "en", source: str = "youtube"
):
    """Convert transcript items into structured segments."""
    segments = []

    for index, item in enumerate(transcript):
        text = getattr(item, "text", None)
        start = float(getattr(item, "start", 0.0))
        duration = float(getattr(item, "duration", 0.0))
        if not text:
            continue

        segments.append(
            {
                "segment_id": index,
                "text": text,
                "start_time": start,
                "end_time": start + duration,
                "language": language,
                "source": source,
            }
        )

    return segments


def chunk_transcript(transcript_text, chunk_size=1000, chunk_overlap=200):
    """Split a transcript string into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(transcript_text)


def chunk_transcript_segments(
    transcript,
    *,
    video_id: str,
    video_url: str,
    language: str = "en",
    source: str = "youtube_transcript",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """Split transcript segments into LangChain documents with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    documents = []
    chunk_id = 0

    for segment in build_transcript_segments(
        transcript, language=language, source=source
    ):
        base_document = Document(
            page_content=segment["text"],
            metadata={
                "video_id": video_id,
                "video_url": video_url,
                "segment_id": segment["segment_id"],
                "language": segment["language"],
                "source": segment["source"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
            },
        )
        for split_document in splitter.split_documents([base_document]):
            split_document.metadata.update(
                {
                    "chunk_id": chunk_id,
                    "source_attribution": (
                        f"{video_id}:{split_document.metadata['start_time']:.2f}-"
                        f"{split_document.metadata['end_time']:.2f}"
                    ),
                }
            )
            documents.append(split_document)
            chunk_id += 1

    return documents

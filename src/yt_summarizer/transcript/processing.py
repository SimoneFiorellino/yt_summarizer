"""Helpers for transcript normalization and chunking."""

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


def chunk_transcript(transcript_text, chunk_size=1000, chunk_overlap=200):
    """Split a transcript string into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(transcript_text)

"""API layer.

The FastAPI app lives here so serving code stays separate from the Gradio UI and
core summarization logic.
"""

from yt_summarizer.api.app import create_app

__all__ = ["create_app"]

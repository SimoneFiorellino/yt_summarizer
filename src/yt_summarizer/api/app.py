"""FastAPI application factory."""

from yt_summarizer.core import VideoRAGService
from yt_summarizer.errors import YTSummarizerError


def create_app():
    """Create the API app.

    FastAPI is imported lazily so the package remains usable when only the
    local Gradio UI dependencies are installed.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI API support requires installing the 'fastapi' and 'pydantic' "
            "packages."
        ) from exc
    
    #### Pydantic models for request validation ####

    class VideoRequest(BaseModel):
        """Request model for video summarization."""
        video_url: str

    class QuestionRequest(BaseModel):
        """Request model for question answering."""
        video_url: str
        question: str

    #### RAG logic and API endpoints ####

    service = VideoRAGService() # The API should not manage the RAG service lifecycle;
    app = FastAPI(title="YT Summarizer API") # Create the FastAPI app;

    #### Define API endpoints ####

    @app.get("/health")
    def health(): 
        """Health check endpoint."""
        return {"status": "ok"}

    @app.post("/summarize")
    def summarize(request: VideoRequest): 
        """Summarize the video at the given URL."""
        try:
            return {"summary": service.summarize_video(request.video_url)}
        except YTSummarizerError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/ask")
    def ask(request: QuestionRequest):
        """Answer a question about the video at the given URL."""
        try:
            return {"answer": service.answer_question(request.video_url, request.question)}
        except YTSummarizerError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return app

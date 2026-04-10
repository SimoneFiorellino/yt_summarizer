"""FastAPI application factory."""

import logging
import time

from yt_summarizer.core import VideoRAGService
from yt_summarizer.errors import YTSummarizerError


logger = logging.getLogger(__name__)


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

    @app.middleware("http")
    async def log_requests(request, call_next):
        """Log HTTP request status and latency."""
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.exception(
                "http_request_failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                },
            )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "http_request_completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

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
            logger.warning(
                "api_error",
                extra={
                    "endpoint": "summarize",
                    "error_type": type(exc).__name__,
                    "status_code": exc.status_code,
                    "error": str(exc),
                },
            )
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/ask")
    def ask(request: QuestionRequest):
        """Answer a question about the video at the given URL."""
        try:
            return {"answer": service.answer_question(request.video_url, request.question)}
        except YTSummarizerError as exc:
            logger.warning(
                "api_error",
                extra={
                    "endpoint": "ask",
                    "error_type": type(exc).__name__,
                    "status_code": exc.status_code,
                    "error": str(exc),
                },
            )
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return app

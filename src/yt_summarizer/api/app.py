"""FastAPI application factory."""

import logging
import time

from yt_summarizer.config import load_config
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

        video_url: str | None = None
        video_id: str | None = None

    class HealthResponse(BaseModel):
        """Response model for API health checks."""

        status: str

    class IngestVideoResponse(BaseModel):
        """Response model for transcript ingestion."""

        video_id: str
        language: str
        transcript_items: int
        processed_chars: int
        chunk_count: int

    class QuestionRequest(BaseModel):
        """Request model for question answering."""

        video_url: str | None = None
        video_id: str | None = None
        question: str

    class SummaryResponse(BaseModel):
        """Response model for video summarization."""

        summary: str

    class SourceResponse(BaseModel):
        """Response model for a retrieved supporting chunk."""

        video_id: str
        chunk_id: int
        source: str
        language: str
        start_time: float
        end_time: float
        source_attribution: str
        excerpt: str

    class AnswerResponse(BaseModel):
        """Response model for question answering."""

        answer: str
        sources: list[SourceResponse]

    #### RAG logic and API endpoints ####

    service = VideoRAGService()  # The API should not manage the RAG service lifecycle;
    app = FastAPI(title="YT Summarizer API")  # Create the FastAPI app;

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

    @app.get("/health", response_model=HealthResponse)
    def health():
        """Health check endpoint."""
        return HealthResponse(status="ok")

    @app.post("/ingest_video", response_model=IngestVideoResponse)
    def ingest(request: VideoRequest):
        """Fetch and preprocess the transcript for a YouTube video."""
        try:
            if not request.video_url:
                raise HTTPException(
                    status_code=400,
                    detail="Offline ingestion requires a valid video_url.",
                )
            ingested = service.ingest_video(request.video_url)
            return IngestVideoResponse(
                video_id=ingested.video_id,
                language=ingested.language,
                transcript_items=len(ingested.raw_transcript),
                processed_chars=len(ingested.processed_text),
                chunk_count=ingested.chunk_count,
            )
        except YTSummarizerError as exc:
            logger.warning(
                "api_error",
                extra={
                    "endpoint": "ingest_video",
                    "error_type": type(exc).__name__,
                    "status_code": exc.status_code,
                    "error": str(exc),
                },
            )
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/summarize", response_model=SummaryResponse)
    def summarize(request: VideoRequest):
        """Summarize the video at the given URL."""
        try:
            return SummaryResponse(
                summary=service.summarize_video(
                    video_url=request.video_url,
                    video_id=request.video_id,
                )
            )
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

    @app.post("/ask", response_model=AnswerResponse)
    def ask(request: QuestionRequest):
        """Answer a question about the video at the given URL."""
        try:
            result = service.answer_question(
                video_url=request.video_url,
                video_id=request.video_id,
                question=request.question,
            )
            return AnswerResponse(
                answer=result.answer,
                sources=[
                    SourceResponse(**source.__dict__) for source in result.sources
                ],
            )
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


def main():
    """Run the API server with Uvicorn."""
    import uvicorn

    config = load_config()
    uvicorn.run(
        "yt_summarizer.api.app:create_app",
        factory=True,
        host=config.api_host,
        port=config.api_port,
    )

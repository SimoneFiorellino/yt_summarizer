"""Domain-specific errors for the application."""


class YTSummarizerError(Exception):
    """Base application error with an HTTP-friendly status code."""

    status_code = 500


class InvalidVideoURLError(YTSummarizerError):
    """Raised when a YouTube URL cannot be parsed into a valid video ID."""

    status_code = 400


class InvalidRequestError(YTSummarizerError):
    """Raised when a valid endpoint receives incomplete or invalid input."""

    status_code = 400


class TranscriptUnavailableError(YTSummarizerError):
    """Raised when a transcript cannot be found for a valid video."""

    status_code = 404


class TranscriptFetchError(YTSummarizerError):
    """Raised when transcript retrieval fails for an external reason."""

    status_code = 502


class ArtifactNotFoundError(YTSummarizerError):
    """Raised when online endpoints are called before offline ingestion/indexing."""

    status_code = 404


class RetrievalError(YTSummarizerError):
    """Raised when vector indexing or similarity search fails."""

    status_code = 500


class ModelError(YTSummarizerError):
    """Raised when the LLM or embedding model call fails."""

    status_code = 502

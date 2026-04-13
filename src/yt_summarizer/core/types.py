"""Structured types returned by core services."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceAttribution:
    """Metadata describing a retrieved supporting chunk."""

    video_id: str
    chunk_id: int
    source: str
    language: str
    start_time: float
    end_time: float
    source_attribution: str
    excerpt: str


@dataclass(frozen=True)
class QAResult:
    """Answer plus supporting source attribution."""

    answer: str
    sources: list[SourceAttribution]

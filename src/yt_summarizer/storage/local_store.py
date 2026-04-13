"""Local filesystem storage for processed transcripts and FAISS indexes."""

from __future__ import annotations

import json
from pathlib import Path

from yt_summarizer.errors import ArtifactNotFoundError


class LocalArtifactStore:
    """Persist and load offline artifacts for each ingested video."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def video_dir(self, video_id: str) -> Path:
        return self.root_dir / video_id

    def save_processed_transcript(self, video_id: str, processed_text: str) -> Path:
        video_dir = self.video_dir(video_id)
        video_dir.mkdir(parents=True, exist_ok=True)
        path = video_dir / "transcript.txt"
        path.write_text(processed_text, encoding="utf-8")
        return path

    def load_processed_transcript(self, video_id: str) -> str:
        path = self.video_dir(video_id) / "transcript.txt"
        if not path.exists():
            raise ArtifactNotFoundError(
                "Video artifacts not found. Run offline ingestion first."
            )
        return path.read_text(encoding="utf-8")

    def save_metadata(self, video_id: str, metadata: dict) -> Path:
        video_dir = self.video_dir(video_id)
        video_dir.mkdir(parents=True, exist_ok=True)
        path = video_dir / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return path

    def load_metadata(self, video_id: str) -> dict:
        path = self.video_dir(video_id) / "metadata.json"
        if not path.exists():
            raise ArtifactNotFoundError(
                "Video metadata not found. Run offline ingestion first."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def save_chunk_metadata(self, video_id: str, chunks: list[dict]) -> Path:
        video_dir = self.video_dir(video_id)
        video_dir.mkdir(parents=True, exist_ok=True)
        path = video_dir / "chunks.json"
        path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        return path

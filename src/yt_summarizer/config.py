"""Application configuration.

AppConfig:
    Dataclass that defines the default runtime values used by the app.

load_config:
    Loads a local .env file, reads environment variables, and falls back to
    AppConfig defaults when variables are not defined.

_env_bool:
    Converts boolean-like environment variable strings into Python booleans.
"""

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_ENV_FILE = ".env"


@dataclass(frozen=True)
class AppConfig:
    """Runtime settings shared by UI, API, and core services."""

    llm_model: str = "phi3:mini"
    llm_temperature: float = 0.5
    llm_max_tokens: int = 256
    llm_timeout_seconds: float = 60.0
    llm_retry_attempts: int = 2
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    normalize_embeddings: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 7
    log_level: str = "INFO"
    log_json: bool = True
    gradio_host: str = "0.0.0.0"
    gradio_port: int = 7865


def load_config(env_file: str | Path | None = DEFAULT_ENV_FILE) -> AppConfig:
    """Load runtime configuration from .env, environment variables, and defaults."""
    if env_file:
        _load_env_file(Path(env_file))

    return AppConfig(
        llm_model=os.getenv("YT_LLM_MODEL", AppConfig.llm_model),
        llm_temperature=float(os.getenv("YT_LLM_TEMPERATURE", AppConfig.llm_temperature)),
        llm_max_tokens=int(os.getenv("YT_LLM_MAX_TOKENS", AppConfig.llm_max_tokens)),
        llm_timeout_seconds=float(
            os.getenv("YT_LLM_TIMEOUT_SECONDS", AppConfig.llm_timeout_seconds)
        ),
        llm_retry_attempts=int(
            os.getenv("YT_LLM_RETRY_ATTEMPTS", AppConfig.llm_retry_attempts)
        ),
        embedding_model=os.getenv("YT_EMBEDDING_MODEL", AppConfig.embedding_model),
        embedding_device=os.getenv("YT_EMBEDDING_DEVICE", AppConfig.embedding_device),
        normalize_embeddings=_env_bool(
            "YT_NORMALIZE_EMBEDDINGS",
            AppConfig.normalize_embeddings,
        ),
        chunk_size=int(os.getenv("YT_CHUNK_SIZE", AppConfig.chunk_size)),
        chunk_overlap=int(os.getenv("YT_CHUNK_OVERLAP", AppConfig.chunk_overlap)),
        retrieval_top_k=int(os.getenv("YT_RETRIEVAL_TOP_K", AppConfig.retrieval_top_k)),
        log_level=os.getenv("YT_LOG_LEVEL", AppConfig.log_level).upper(),
        log_json=_env_bool("YT_LOG_JSON", AppConfig.log_json),
        gradio_host=os.getenv("YT_GRADIO_HOST", AppConfig.gradio_host),
        gradio_port=int(os.getenv("YT_GRADIO_PORT", AppConfig.gradio_port)),
    )


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file without overriding existing env vars."""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line.removeprefix("export ").strip()

        key, separator, value = line.partition("=")
        if not separator:
            continue

        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

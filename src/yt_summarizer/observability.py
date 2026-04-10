"""Structured logging helpers."""

from contextlib import contextmanager
import json
import logging
import time

from yt_summarizer.config import AppConfig


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for machine-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _STANDARD_LOG_RECORD_FIELDS:
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(config: AppConfig) -> None:
    """Configure application logging once."""
    app_logger = logging.getLogger("yt_summarizer")
    app_logger.setLevel(config.log_level)
    app_logger.propagate = False

    handler = _get_or_create_handler(app_logger)
    handler.setLevel(config.log_level)
    if config.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )


@contextmanager
def log_step(logger: logging.Logger, step: str, **fields):
    """Log start/end/failure for a pipeline step with elapsed time."""
    start = time.perf_counter()
    logger.info("step_started", extra={"step": step, **fields})
    try:
        yield
    except Exception:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception(
            "step_failed",
            extra={"step": step, "duration_ms": duration_ms, **fields},
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "step_completed",
            extra={"step": step, "duration_ms": duration_ms, **fields},
        )


def _get_or_create_handler(logger: logging.Logger) -> logging.Handler:
    for handler in logger.handlers:
        if getattr(handler, "_yt_summarizer_handler", False):
            return handler

    handler = logging.StreamHandler()
    handler._yt_summarizer_handler = True
    logger.addHandler(handler)
    return handler


_STANDARD_LOG_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}

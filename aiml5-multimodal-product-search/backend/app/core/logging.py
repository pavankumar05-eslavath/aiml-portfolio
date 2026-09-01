"""Structured application logging.

Emits either human-readable console lines (local development) or single-line
JSON (containers, log shippers). Arbitrary structured context is attached with
the ``extra={"context": {...}}`` convention, which keeps the logging call sites
free of string formatting.

Deliberately implemented on the standard library rather than pulling in a
logging framework: the surface we need is small and this keeps the dependency
footprint of the inference container down.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
    | {"asctime", "message", "taskName", "context"}
)


def set_request_id(value: str | None = None) -> str:
    """Bind a request id to the current context, generating one if absent."""
    rid = value or uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def get_request_id() -> str | None:
    """Return the request id bound to the current context, if any."""
    return _request_id.get()


def _collect_context(record: logging.LogRecord) -> dict[str, Any]:
    context: dict[str, Any] = {}
    explicit = getattr(record, "context", None)
    if isinstance(explicit, dict):
        context.update(explicit)
    for key, value in record.__dict__.items():
        if key not in _RESERVED and not key.startswith("_"):
            context[key] = value
    return context


class JsonFormatter(logging.Formatter):
    """Format records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a log record as a single-line JSON object."""
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if rid := get_request_id():
            payload["request_id"] = rid
        if context := _collect_context(record):
            payload["context"] = context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


class ConsoleFormatter(logging.Formatter):
    """Compact, aligned, human-readable output for local development."""

    _COLOURS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    _RESET = "\033[0m"

    def __init__(self, *, use_colour: bool = True) -> None:
        super().__init__()
        self.use_colour = use_colour and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Render a log record as an aligned, optionally coloured console line."""
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = record.levelname
        if self.use_colour:
            level = f"{self._COLOURS.get(record.levelname, '')}{level:<8}{self._RESET}"
        else:
            level = f"{level:<8}"
        name = record.name.removeprefix("app.")
        line = f"{ts} {level} {name:<28} {record.getMessage()}"
        if rid := get_request_id():
            line += f"  [req={rid}]"
        if context := _collect_context(record):
            rendered = " ".join(f"{k}={_render(v)}" for k, v in sorted(context.items()))
            line += f"  {rendered}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _render(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return f'"{text}"' if " " in text else text


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Install the root logging handler. Safe to call more than once."""
    formatter: logging.Formatter = JsonFormatter() if fmt == "json" else ConsoleFormatter()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in root.handlers[:]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # These are chatty at INFO and add nothing over our own request logging.
    for noisy in ("uvicorn.access", "httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)


@contextmanager
def log_duration(
    logger: logging.Logger,
    message: str,
    *,
    level: int = logging.INFO,
    **context: Any,
) -> Iterator[dict[str, Any]]:
    """Log ``message`` with a ``duration_ms`` field once the block completes.

    The yielded dict can be mutated to add fields discovered inside the block::

        with log_duration(log, "indexed batch", batch=3) as ctx:
            ctx["rows"] = len(rows)
    """
    extra: dict[str, Any] = dict(context)
    started = time.perf_counter()
    try:
        yield extra
    except Exception:
        extra["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
        logger.exception(f"{message} failed", extra={"context": extra})
        raise
    extra["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
    logger.log(level, message, extra={"context": extra})

"""Logging setup: ids, counts, timings only — secrets are redacted (I8)."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from typing import IO

from qasystem.config import Settings

REDACTED = "***REDACTED***"


def scrub(secrets: Iterable[str], text: str) -> str:
    """Replace every known secret in ``text``. Public: error messages need it too (I8)."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    return text


class SecretRedactionFilter(logging.Filter):
    """Replaces known secrets in the record message and its args."""

    def __init__(self, secrets: Iterable[str]) -> None:
        super().__init__()
        self._secrets = tuple(secret for secret in secrets if secret)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        record.msg = scrub(self._secrets, str(record.msg))
        if isinstance(record.args, tuple):
            # Scrub only str args: str() on an int arg would break the format spec
            # (httpx logs 'HTTP Request: ... "%s %d %s"' with an int status code).
            record.args = tuple(
                scrub(self._secrets, arg) if isinstance(arg, str) else arg for arg in record.args
            )
        return True


def configure_logging(
    settings: Settings | None = None,
    *,
    level: int = logging.INFO,
    stream: IO[str] | None = None,
) -> logging.Logger:
    """Install one stderr handler whose filter knows the configured token."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    handler.addFilter(SecretRedactionFilter(_secrets_of(settings)))

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    return root


def _secrets_of(settings: Settings | None) -> tuple[str, ...]:
    if settings is None or settings.embedding_api_key is None:
        return ()
    return (settings.embedding_api_key.get_secret_value(),)

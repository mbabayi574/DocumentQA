"""Typed domain errors with stable codes and HTTP mapping (plan.md §4).

Every error crossing an architectural boundary is one of these; the API layer maps
``code``/``http_status`` to the single stable error envelope. Unknown exceptions
become ``INTERNAL_ERROR`` without stack traces or secrets (I8).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar


class QASystemError(Exception):
    """Base class for every error the application raises on purpose."""

    code: ClassVar[str] = "INTERNAL_ERROR"
    http_status: ClassVar[int] = 500

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details or {})


class ConfigError(QASystemError):
    """Invalid or missing configuration (fails fast at startup)."""

    code = "CONFIG_ERROR"
    http_status = 500


class UnsupportedFormatError(QASystemError):
    """Document format is not one of PDF, TXT, Markdown."""

    code = "UNSUPPORTED_FORMAT"
    http_status = 415


class FileTooLargeError(QASystemError):
    """Upload exceeds MAX_UPLOAD_MB."""

    code = "FILE_TOO_LARGE"
    http_status = 413


class EmptyDocumentError(QASystemError):
    """Parsed document has no non-whitespace text."""

    code = "EMPTY_DOCUMENT"
    http_status = 422


class ParseError(QASystemError):
    """Document could not be parsed (corrupt container, bad encoding)."""

    code = "PARSE_ERROR"
    http_status = 422


class NoTextLayerError(QASystemError):
    """PDF has no usable text layer; OCR is out of scope (no OCR, by design)."""

    code = "NO_TEXT_LAYER"
    http_status = 422


class DocumentNotFoundError(QASystemError):
    """Document id is unknown, deleted, or not active."""

    code = "DOCUMENT_NOT_FOUND"
    http_status = 404


class DocumentExistsError(QASystemError):
    """POST with an existing active doc_id and different content; use PUT."""

    code = "DOCUMENT_EXISTS"
    http_status = 409


class EmbeddingUnavailableError(QASystemError):
    """Embedding API failed after the bounded number of retries."""

    code = "EMBEDDING_UNAVAILABLE"
    http_status = 503


class EmbeddingAuthError(QASystemError):
    """Embedding API rejected our credentials (401/403); never retried."""

    code = "EMBEDDING_AUTH"
    http_status = 502


class VectorStoreError(QASystemError):
    """Local Chroma persistent store is unusable."""

    code = "VECTOR_STORE_UNAVAILABLE"
    http_status = 503


class StorageLockedError(QASystemError):
    """Another process already owns this data directory (plan.md L2)."""

    code = "STORAGE_LOCKED"
    http_status = 503

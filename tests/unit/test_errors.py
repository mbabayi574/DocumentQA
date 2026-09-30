"""P0: typed errors carry the plan §4 code/HTTP mapping."""

from __future__ import annotations

import pytest

from qasystem.errors import (
    ConfigError,
    DocumentExistsError,
    DocumentNotFoundError,
    EmbeddingAuthError,
    EmbeddingUnavailableError,
    EmptyDocumentError,
    FileTooLargeError,
    NoTextLayerError,
    ParseError,
    QASystemError,
    UnsupportedFormatError,
    VectorStoreError,
)

CASES = [
    (UnsupportedFormatError, "UNSUPPORTED_FORMAT", 415),
    (FileTooLargeError, "FILE_TOO_LARGE", 413),
    (EmptyDocumentError, "EMPTY_DOCUMENT", 422),
    (ParseError, "PARSE_ERROR", 422),
    (NoTextLayerError, "NO_TEXT_LAYER", 422),
    (DocumentNotFoundError, "DOCUMENT_NOT_FOUND", 404),
    (DocumentExistsError, "DOCUMENT_EXISTS", 409),
    (EmbeddingUnavailableError, "EMBEDDING_UNAVAILABLE", 503),
    (EmbeddingAuthError, "EMBEDDING_AUTH", 502),
    (VectorStoreError, "VECTOR_STORE_UNAVAILABLE", 503),
    (ConfigError, "CONFIG_ERROR", 500),
]


@pytest.mark.parametrize(("error", "code", "status"), CASES)
def test_code_and_status(error: type[QASystemError], code: str, status: int) -> None:
    assert error.code == code
    assert error.http_status == status
    assert issubclass(error, QASystemError)


@pytest.mark.parametrize(("error", "code", "status"), CASES)
def test_message_and_details_are_kept(error: type[QASystemError], code: str, status: int) -> None:
    exc = error("boom", details={"field": "question"})
    assert str(exc) == "boom"
    assert exc.message == "boom"
    assert exc.details == {"field": "question"}
    assert exc.code == code


def test_unknown_error_maps_to_internal_error() -> None:
    assert QASystemError.code == "INTERNAL_ERROR"
    assert QASystemError.http_status == 500

"""P8: the HTTP surface, over the real services.

``TestClient`` against a real SQLite file, a real ``PersistentClient`` and a real service
graph on ``tmp_path`` — only the embedder is fake, because the network is blocked here and
``test_api_live.py`` covers the real one. A test that mocks the service graph would test the
mock.

Every endpoint is exercised on four axes: the happy path, a validation failure, a domain error,
and the response shape. The error envelope and the secret-redaction rule are asserted on the
whole surface, because those are the two things a caller depends on and neither is per-endpoint.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest
from fastapi.testclient import TestClient

from qasystem.api.app import create_app
from qasystem.api.deps import Services, build_services

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "docs"
DIM = 32
FAKE_MODEL = "fake-embedder"

HANDBOOK = """# Handbook

## Install

Run the installer on Linux. It checks the kernel version first.

## Usage

The service returns ERR-404 when the token is invalid.
"""

HANDBOOK_V2 = """# Handbook

## Install

Run the installer on Linux. It checks the kernel version first.

## Usage

The service returns ERR-503 when the upstream cache is unreachable.
"""


@pytest.fixture
async def services(tmp_path: Any, settings_factory: Any) -> Any:
    """A real service graph on tmp_path, fake embeddings."""
    # The shipped thresholds are keyed to Bge-m3 and the loader refuses another model's
    # (D42), which is exactly right -- so these tests supply their own, for the fake model.
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "version": 1,
                "model_id": FAKE_MODEL,
                "calibrated": False,
                "min_dense": 0.05,
                "min_coverage": 0.70,
                "min_coverage_high": 0.90,
                "min_sentence_overlap": 0.15,
            }
        ),
        encoding="utf-8",
    )
    settings = settings_factory(
        app_env="test", embedding_provider="fake", thresholds_path=thresholds_path
    )
    built = await build_services(settings, data_dir=tmp_path)
    yield built
    await built.aclose()


@pytest.fixture
def client(services: Any) -> Any:
    with TestClient(create_app(services=services)) as test_client:
        yield test_client


def _fake_thresholds(tmp_path: Path) -> Path:
    """A thresholds file for the fake model; the shipped one is BGE-M3's (D42)."""
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "model_id": FAKE_MODEL,
                "calibrated": False,
                "min_dense": 0.05,
                "min_coverage": 0.70,
                "min_coverage_high": 0.90,
                "min_sentence_overlap": 0.15,
            }
        ),
        encoding="utf-8",
    )
    return path


def upload(content: bytes = HANDBOOK.encode(), name: str = "handbook.md", **fields: Any) -> Any:
    """A multipart upload. Form fields go in ``data``, which is where httpx documents them."""
    form = {key: value for key, value in fields.items() if value is not None}
    return {"files": {"file": (name, content, "text/markdown")}, "data": form}


def query(client: Any, question: str, **body: Any) -> Any:
    return client.post("/query", json={"question": question, **body})


# ---------------------------------------------------------------- documents: POST


def test_post_a_document_creates_it(client: Any) -> None:
    response = client.post("/documents", **upload())

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["doc_id"] == "handbook"
    assert body["status"] == "created"
    assert body["version"] == 1
    assert body["chunks_added"] > 0
    assert body["embed_requests"] == 1


def test_post_the_same_bytes_again_is_200_unchanged(client: Any) -> None:
    client.post("/documents", **upload())
    response = client.post("/documents", **upload())

    assert response.status_code == 200
    assert response.json()["status"] == "unchanged"
    assert response.json()["version"] == 1


def test_post_different_content_to_an_existing_doc_is_409_and_points_at_put(
    client: Any,
) -> None:
    """A POST that would silently replace a document is ambiguous, so it refuses."""
    client.post("/documents", **upload())
    response = client.post("/documents", **upload(HANDBOOK_V2.encode()))

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "DOCUMENT_EXISTS"
    assert "PUT" in error["message"]
    assert error["request_id"]


def test_an_explicit_doc_id_overrides_the_filename_slug(client: Any) -> None:
    response = client.post("/documents", **upload(name="Hand Book!.md", doc_id="custom"))
    assert response.status_code == 201
    assert response.json()["doc_id"] == "custom"


def test_post_an_empty_file_is_422(client: Any) -> None:
    response = client.post("/documents", **upload(b"", "empty.md"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "EMPTY_DOCUMENT"


def test_post_an_unsupported_extension_is_415(client: Any) -> None:
    response = client.post("/documents", **upload(b"hi", "notes.exe"))
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_post_an_oversized_file_is_413(client: Any) -> None:
    from qasystem.errors import FileTooLargeError

    big = b"x" * (services_limit_bytes := 21 * 1024 * 1024)
    response = client.post("/documents", **upload(big, "big.md"))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == FileTooLargeError.code
    assert services_limit_bytes > 0


def test_post_a_pdf_without_magic_bytes_is_415(client: Any) -> None:
    response = client.post("/documents", **upload(b"not a pdf", "fake.pdf"))
    assert response.status_code == 415


def test_post_the_real_pdf_fixture_works(client: Any) -> None:
    data = (FIXTURES / "fa" / "ai-engineer.pdf").read_bytes()
    response = client.post(
        "/documents", files={"file": ("ai-engineer.pdf", data, "application/pdf")}
    )
    assert response.status_code == 201
    assert response.json()["chunks_added"] > 0


# ---------------------------------------------------------------- documents: PUT


def test_put_replaces_an_active_document(client: Any) -> None:
    client.post("/documents", **upload())
    response = client.put("/documents/handbook", **upload(HANDBOOK_V2.encode()))

    assert response.status_code == 200
    assert response.json()["status"] == "updated"
    assert response.json()["version"] == 2


def test_put_the_same_bytes_is_unchanged(client: Any) -> None:
    client.post("/documents", **upload())
    response = client.put("/documents/handbook", **upload())

    assert response.status_code == 200
    assert response.json()["status"] == "unchanged"
    assert response.json()["embed_requests"] == 0


def test_put_an_unknown_document_is_404(client: Any) -> None:
    response = client.put("/documents/nope", **upload())
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_put_a_deleted_document_is_404_rather_than_resurrecting_it(client: Any) -> None:
    """A tombstone is not an invitation to write to it; re-adding is a POST."""
    client.post("/documents", **upload())
    assert client.delete("/documents/handbook").status_code == 204

    response = client.put("/documents/handbook", **upload(HANDBOOK_V2.encode()))

    assert response.status_code == 404
    assert client.get("/documents").json()["total"] == 0


# ---------------------------------------------------------------- documents: DELETE


def test_delete_returns_204_and_makes_it_unqueryable(client: Any) -> None:
    client.post("/documents", **upload())
    assert query(client, "kernel version").json()["status"] == "answered"

    response = client.delete("/documents/handbook")

    assert response.status_code == 204
    assert response.content == b""
    answer = query(client, "kernel version").json()
    assert answer["status"] == "insufficient_information"
    assert answer["citations"] == []


def test_delete_an_unknown_document_is_404(client: Any) -> None:
    assert client.delete("/documents/nope").status_code == 404


def test_repeat_delete_is_404(client: Any) -> None:
    client.post("/documents", **upload())
    assert client.delete("/documents/handbook").status_code == 204
    assert client.delete("/documents/handbook").status_code == 404


# ---------------------------------------------------------------- documents: GET


def test_list_documents_is_paginated(client: Any) -> None:
    for name in ("alpha", "bravo", "charlie"):
        client.post("/documents", **upload(name=f"{name}.md"))

    page = client.get("/documents", params={"limit": 2}).json()

    assert [d["doc_id"] for d in page["items"]] == ["alpha", "bravo"]
    assert page["total"] == 3
    assert page["limit"] == 2 and page["offset"] == 0

    second = client.get("/documents", params={"limit": 2, "offset": 2}).json()
    assert [d["doc_id"] for d in second["items"]] == ["charlie"]
    assert second["offset"] == 2


def test_listing_excludes_deleted_documents(client: Any) -> None:
    client.post("/documents", **upload(name="alpha.md"))
    client.post("/documents", **upload(name="bravo.md"))
    client.delete("/documents/alpha")

    body = client.get("/documents").json()

    assert [d["doc_id"] for d in body["items"]] == ["bravo"]
    assert body["total"] == 1


def test_listing_bounds_limit_and_offset(client: Any) -> None:
    client.post("/documents", **upload())
    response = client.get("/documents", params={"limit": 0})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert client.get("/documents", params={"limit": -1}).status_code == 422
    assert client.get("/documents", params={"offset": -1}).status_code == 422


def test_get_one_document_returns_metadata_and_a_section_outline(client: Any) -> None:
    client.post("/documents", **upload())
    body = client.get("/documents/handbook").json()

    assert body["doc_id"] == "handbook"
    assert body["title"] == "Handbook"
    assert body["document"] == "handbook.md"
    assert body["format"] == "md"
    assert body["status"] == "active"
    assert body["current_version"] == 1
    assert body["chunk_count"] > 0
    assert body["created_at"] and body["updated_at"]
    # Derived from the version's chunks, so it lists the paths chunks actually carry. There
    # is no chunk for the bare "Handbook" prefix, so no root entry appears -- inventing one
    # would claim a section that no offset range corresponds to.
    assert [section["path"] for section in body["sections"]] == [
        "Handbook > Install",
        "Handbook > Usage",
    ]
    install = body["sections"][0]
    assert install["page"] is None, "Markdown has no pages"
    assert install["lines"] is not None and install["lines"][0] >= 1


def test_get_one_unknown_document_is_404(client: Any) -> None:
    assert client.get("/documents/nope").status_code == 404


def test_get_one_deleted_document_is_404(client: Any) -> None:
    client.post("/documents", **upload())
    client.delete("/documents/handbook")
    assert client.get("/documents/handbook").status_code == 404


def test_a_pdf_section_outline_carries_pages(client: Any) -> None:
    data = (FIXTURES / "fa" / "ai-engineer.pdf").read_bytes()
    client.post("/documents", files={"file": ("ai-engineer.pdf", data, "application/pdf")})
    body = client.get("/documents/ai-engineer").json()

    assert body["format"] == "pdf"
    assert body["sections"]
    assert any(section["page"] is not None for section in body["sections"])


# ---------------------------------------------------------------- query


def test_query_answers_with_citations(client: Any) -> None:
    client.post("/documents", **upload())
    response = query(client, "kernel version")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert "[1]" in body["answer"]
    assert body["citations"]
    citation = body["citations"][0]
    assert citation["doc_id"] == "handbook"
    assert citation["doc_version"] == 1
    assert citation["section"] == "Handbook > Install"
    assert citation["char_span"] == [0, len(citation["excerpt"])] or citation["char_span"][1] > 0
    assert citation["excerpt"]


@pytest.mark.parametrize(
    ("question", "code"),
    [
        ("", "VALIDATION_ERROR"),  # empty string: the field itself is invalid
        ("   ", "INVALID_QUESTION"),  # passes the schema, carries no searchable terms
        ("?!", "INVALID_QUESTION"),  # punctuation only, same
    ],
)
def test_query_a_blank_question_is_422(client: Any, question: str, code: str) -> None:
    response = query(client, question)
    assert response.status_code == 422, question
    assert response.json()["error"]["code"] == code


def test_query_a_missing_field_is_422_in_the_same_envelope(client: Any) -> None:
    response = client.post("/query", json={})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert response.json()["error"]["request_id"]


def test_query_refuses_with_insufficient_information_and_no_citations(client: Any) -> None:
    client.post("/documents", **upload())
    body = query(client, "what is the population of Reykjavik").json()

    assert body["status"] == "insufficient_information"
    assert body["citations"] == []
    assert body["segments"] == []
    assert body["reason"] == "below_threshold"
    assert body["evidence_score"] == 0.0
    assert "Not enough information" in body["answer"]


def test_query_on_an_empty_knowledge_base_says_so(client: Any) -> None:
    body = query(client, "kernel version").json()
    assert body["reason"] == "empty_knowledge_base"


def test_query_honours_doc_ids_filter(client: Any) -> None:
    client.post("/documents", **upload(name="alpha.md"))
    client.post("/documents", **upload(name="bravo.md"))

    only_bravo = query(client, "kernel version", doc_ids=["bravo"]).json()
    only_alpha = query(client, "kernel version", doc_ids=["alpha"]).json()

    assert only_bravo["citations"][0]["doc_id"] == "bravo"
    assert only_alpha["citations"][0]["doc_id"] == "alpha"


def test_query_honours_a_language_filter(client: Any) -> None:
    client.post("/documents", **upload())

    body = query(client, "kernel version", language="fa").json()

    assert body["status"] == "insufficient_information"


def test_query_top_k_and_debug_are_accepted(client: Any) -> None:
    client.post("/documents", **upload())

    quiet = query(client, "kernel version", top_k=1).json()
    loud = query(client, "kernel version", top_k=1, debug=True).json()

    assert quiet["debug"] is None
    assert loud["debug"]["gate"]["passed"] is True
    assert loud["debug"]["gate"]["calibrated"] is False
    assert loud["debug"]["candidates"]


def test_query_never_returns_superseded_content(client: Any) -> None:
    client.post("/documents", **upload())
    client.put("/documents/handbook", **upload(HANDBOOK_V2.encode()))

    body = query(client, "what does the service return when the token is invalid").json()

    assert "ERR-404" not in json.dumps(body)
    assert "ERR-503" in query(client, "upstream cache unreachable").json()["answer"]


# ---------------------------------------------------------------- ops


def test_health_is_always_200(client: Any) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_every_dependency(client: Any) -> None:
    body = client.get("/ready").json()

    assert body["status"] == "ready"
    assert body["sqlite"] == "ok"
    assert body["fts5"] is True
    assert body["chroma"] == "ok"
    assert body["lock_held"] is True
    assert body["embedder"] == "fake"
    assert body["model_id"] == FAKE_MODEL
    assert body["dimension"] == DIM
    assert body["thresholds_calibrated"] is False


def test_ready_makes_no_embedding_call_per_request(client: Any, services: Any) -> None:
    """L9: the model is validated once at startup and cached, not re-probed per request."""
    client.get("/ready")
    before = services.embedder.requests

    for _ in range(3):
        assert client.get("/ready").status_code == 200

    assert services.embedder.requests == before


def test_ready_reports_503_when_a_dependency_is_broken(client: Any, services: Any) -> None:
    services.store.close()

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["sqlite"] != "ok"


# ---------------------------------------------------------------- cross-cutting


def test_every_error_uses_one_envelope(client: Any) -> None:
    calls = [
        (client.get("/documents/nope"), 404, "DOCUMENT_NOT_FOUND"),
        (client.delete("/documents/nope"), 404, "DOCUMENT_NOT_FOUND"),
        (client.put("/documents/nope", **upload()), 404, "DOCUMENT_NOT_FOUND"),
        (query(client, "   "), 422, "INVALID_QUESTION"),
        (client.post("/documents", **upload(b"", "e.md")), 422, "EMPTY_DOCUMENT"),
        (client.post("/documents", **upload(b"x", "a.exe")), 415, "UNSUPPORTED_FORMAT"),
    ]
    for response, status, code in calls:
        assert response.status_code == status, (code, response.text)
        body = response.json()
        assert set(body) == {"error"}, body
        assert set(body["error"]) == {"code", "message", "request_id"}, body["error"]
        assert body["error"]["code"] == code
        assert body["error"]["message"]
        assert body["error"]["request_id"]


async def test_an_unexpected_error_is_500_with_no_stack_trace(services: Any) -> None:
    """The response a real server sends, not the one TestClient re-raises over.

    ``ServerErrorMiddleware`` builds the 500 and then re-raises so the server can log it, and
    ``TestClient`` propagates that by default -- which would test the harness rather than the
    contract. ``ASGITransport`` returns the actual response. The lifespan is not run because it
    only sets ``app.state.services``, which is set directly here.
    """
    from httpx import ASGITransport, AsyncClient

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("internal detail: /srv/app/path.py line 42")

    services.ingestion.ingest = boom  # type: ignore[method-assign]
    app = create_app(services=services)
    app.state.services = services

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.post(
            "/documents", files={"file": ("a.md", b"# T\n\nbody.\n", "text/markdown")}
        )

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert body["error"]["request_id"]
    assert "line 42" not in response.text
    assert "Traceback" not in response.text
    assert "/srv/app" not in response.text


def test_the_middleware_logs_the_path_and_never_the_query_string(
    client: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """I8 in the log, not just the response.

    The plan says the middleware logs ids, counts, timings and error classes only. A URL query
    string is user content -- and this project's own /query endpoint takes a question. Nothing
    else would notice a change from ``url.path`` to ``str(url)``, which is why it needs a test
    of its own rather than riding on the secret-in-response checks.
    """
    secret = "sk-live-in-a-query-string-0123456789"

    with caplog.at_level("INFO", logger="qasystem.api.app"):
        response = client.get(f"/documents?limit=5&token={secret}")

    assert response.status_code == 200, "the extra parameter must be ignored, not rejected"
    # Scoped to this project's logger on purpose: httpx also logs the full URL at INFO, and
    # that is a separate question. httpx here only ever calls the embedding provider on a
    # fixed path with the token in a header, so no user data can reach its log lines.
    logged = "\n".join(
        record.getMessage() for record in caplog.records if record.name == "qasystem.api.app"
    )
    assert "/documents" in logged, "the path should be logged"
    assert secret not in logged, "the query string was logged"
    assert "token=" not in logged


def test_a_request_id_is_returned_and_honoured(client: Any) -> None:
    generated = client.get("/health").headers["x-request-id"]
    assert generated

    echoed = client.get("/health", headers={"X-Request-ID": "caller-supplied-id"})

    assert echoed.headers["x-request-id"] == "caller-supplied-id"


@pytest.mark.invariant  # I8
def test_no_endpoint_echoes_a_secret(client: Any, services: Any, settings_factory: Any) -> None:
    """I8 on the HTTP surface: every response body and header is checked."""
    secret = "sk-live-should-never-appear-anywhere-0123456789"
    app_services = services
    app_services.settings = settings_factory(
        app_env="test", embedding_provider="fake", embedding_api_key=secret
    )
    client.post("/documents", **upload())

    responses = [
        client.get("/health"),
        client.get("/ready"),
        client.get("/documents"),
        client.get("/documents/handbook"),
        query(client, "kernel version", debug=True),
        query(client, "population of Reykjavik"),
        client.get("/documents/nope"),
        client.post("/documents", **upload(b"", "e.md")),
    ]
    for response in responses:
        assert secret not in response.text, response.request.url
        assert secret not in "".join(response.headers.get("x-request-id", ""))


# ---------------------------------------------------------------- the OpenAPI contract


def test_the_openapi_schema_lists_every_documented_endpoint(client: Any) -> None:
    schema = client.get("/openapi.json").json()

    assert set(schema["paths"]) == {
        "/health",
        "/ready",
        "/documents",
        "/documents/{doc_id}",
        "/query",
    }
    operations = {
        (path, method)
        for path, entry in schema["paths"].items()
        for method in entry
        if method in {"get", "post", "put", "delete"}
    }
    assert operations == {
        ("/health", "get"),
        ("/ready", "get"),
        ("/documents", "get"),
        ("/documents", "post"),
        ("/documents/{doc_id}", "get"),
        ("/documents/{doc_id}", "put"),
        ("/documents/{doc_id}", "delete"),
        ("/query", "post"),
    }


def test_the_query_response_schema_is_the_documented_one(client: Any) -> None:
    components = client.get("/openapi.json").json()["components"]["schemas"]
    response = components["QueryResponse"]
    # `reason` and `debug` are nullable rather than required, so they are always *present*
    # in a response but never *required* of a caller. `test_query_answers_with_citations`
    # pins that they are always sent.
    assert set(response["required"]) == {
        "status",
        "answer",
        "segments",
        "evidence_score",
        "citations",
    }
    assert set(response["properties"]) == {
        "status",
        "answer",
        "segments",
        "evidence_score",
        "citations",
        "reason",
        "debug",
    }
    citation = components["Citation"]
    # page and lines are nullable: a Markdown citation has no page and a PDF citation has no
    # line numbers. They are always sent, and honestly null, rather than omitted or invented.
    assert set(citation["required"]) == {
        "id",
        "doc_id",
        "document",
        "doc_version",
        "section",
        "chunk_id",
        "char_span",
        "excerpt",
        "score",
    }
    assert set(citation["properties"]) == set(citation["required"]) | {"page", "lines"}


def test_the_error_envelope_is_declared_in_the_schema(client: Any) -> None:
    components = client.get("/openapi.json").json()["components"]["schemas"]
    assert set(components["ErrorBody"]["required"]) == {"code", "message", "request_id"}
    assert set(components["ErrorEnvelope"]["required"]) == {"error"}


def test_the_schema_never_advertises_an_error_shape_the_api_does_not_return(
    client: Any,
) -> None:
    """The app has exactly one error shape, so the schema must not describe another.

    Found by adding Swagger UI and reading what it advertised: four of the twelve declared
    error responses referenced FastAPI's default ``HTTPValidationError``, while
    `_install_validation_error` converts every one of them into ``VALIDATION_ERROR`` inside an
    ``ErrorEnvelope``. So the documentation described a body the API never returns, on exactly
    the endpoints where a caller most needs the contract to be right.

    A schema is a promise. This is the test that keeps it one.
    """
    schema = client.get("/openapi.json").json()
    # `/ready` is the one legitimate exception: a 503 there is a readiness payload, not an
    # error, and a caller reads it rather than parsing it.
    allowed = {"#/components/schemas/ErrorEnvelope", "#/components/schemas/ReadyResponse"}

    offenders = [
        (method.upper(), path, code)
        for path, entry in schema["paths"].items()
        for method, op in entry.items()
        if method in {"get", "post", "put", "delete"}
        for code, response in op.get("responses", {}).items()
        if code.startswith(("4", "5"))
        and response.get("content", {}).get("application/json", {}).get("schema", {}).get("$ref")
        not in allowed
    ]
    assert not offenders, offenders
    assert "HTTPValidationError" not in json.dumps(schema), (
        "FastAPI's default validation-error shape is still referenced, so some endpoint is "
        "documenting a body the API never returns"
    )


def test_every_documented_operation_can_be_called_from_swagger_ui(client: Any) -> None:
    """Swagger UI is only usable if each operation is described and prefilled.

    `/docs` was already served by the framework, so this is not about the page existing -- it is
    about whether a caller can work out what to send. Every operation needs a summary and a
    description, and every *JSON* body needs an example, because "Try it out" opens a textarea and
    an empty one is a guaranteed 422 on the user's first click.

    File uploads are exempt from the example rule and deliberately so: Swagger renders a file
    picker, and a JSON example for a picker cannot be tried. Their text fields carry field-level
    examples instead, which is what actually prefills the form.
    """
    schema = client.get("/openapi.json").json()
    for path, entry in schema["paths"].items():
        for method, op in entry.items():
            if method not in {"get", "post", "put", "delete"}:
                continue
            where = f"{method.upper()} {path}"
            assert op.get("summary"), f"{where} has no summary in Swagger UI"
            assert op.get("description"), f"{where} has no description in Swagger UI"
            for content_type, media in op.get("requestBody", {}).get("content", {}).items():
                if content_type.startswith("multipart/"):
                    continue
                assert media.get("example") or media.get("examples"), (
                    f"{where} has a JSON request body with no example, so 'Try it out' "
                    "opens empty and the first click is a 422"
                )


def test_the_upload_field_is_described_so_swagger_renders_a_file_picker(client: Any) -> None:
    """The upload field must be recognisable as a file, not a string.

    Probed rather than assumed. FastAPI 0.142 + pydantic 2.13 describe `UploadFile` as
    `{"type": "string", "contentMediaType": "application/octet-stream"}` -- the OpenAPI 3.2 /
    JSON Schema 2020-12 spelling -- instead of the older `format: binary`. Whether Swagger UI
    honours the newer spelling is third-party behaviour, so it was measured in the bundle
    FastAPI pins: `swagger-ui-dist@5`'s `isFileUploadIntendedOAS32` returns true when
    `contentMediaType` is a non-empty string, so a picker is rendered and no override is needed.

    The check exists because the failure is silent and total: if a dependency upgrade changes
    the spelling, Swagger falls back to a text box and a document cannot be uploaded from the
    docs page at all, with nothing in the response to say why.
    """
    schema = client.get("/openapi.json").json()
    body = schema["paths"]["/documents"]["post"]["requestBody"]["content"]["multipart/form-data"][
        "schema"
    ]
    file_field = schema["components"]["schemas"][body["$ref"].rsplit("/", 1)[-1]]["properties"][
        "file"
    ]

    as_30 = file_field.get("format") == "binary"
    as_32 = isinstance(file_field.get("contentMediaType"), str) and file_field["contentMediaType"]
    assert file_field.get("type") == "string"
    assert as_30 or as_32, (
        f"file field is described as {file_field}, which Swagger renders as a text box rather "
        "than a picker, so a document cannot be uploaded from /docs"
    )


def test_the_schema_is_stable_across_runs(client: Any) -> None:
    """A snapshot of the paths and required fields, not of descriptions and prose."""

    def shape() -> list[Any]:
        schema = client.get("/openapi.json").json()
        return [
            (path, sorted(method for method in entry if method in {"get", "post", "put", "delete"}))
            for path, entry in sorted(schema["paths"].items())
        ]

    assert shape() == shape()


# ---------------------------------------------------------------- lifespan


def test_an_injected_graph_is_the_callers_to_close(services: Any) -> None:
    """The lifespan must not close a graph it was handed -- the caller still holds it."""
    with TestClient(create_app(services=services)) as test_client:
        assert test_client.get("/ready").json()["lock_held"] is True
    assert services.lock.is_held is True


# A second *instance* has to be a second *process*: filelock is reentrant within one process,
# so an in-process second graph acquires happily and proves nothing. P5 already learned this.
# The holder must KEEP a reference. `DataLock(path).acquire()` as a bare expression lets the
# object be garbage-collected, the file handle with it, and the OS releases the flock -- so
# the "holder" holds nothing and the second process is admitted. P5 found this; it is why
# `lock = DataLock(...)` is two lines everywhere rather than one.
LOCK_HOLDER = """
import sys, time
from qasystem.storage.lock import DataLock
holder = DataLock(sys.argv[1])
holder.acquire()
print("acquired", flush=True)
time.sleep(60)
"""
SECOND_INSTANCE = """
import sys
from pathlib import Path
from qasystem.api.deps import build_services
from qasystem.config import load_settings
from qasystem.errors import StorageLockedError

data, thresholds = sys.argv[1], sys.argv[2]
settings = load_settings(
    env_file=None, app_env="test", embedding_provider="fake",
    thresholds_path=Path(thresholds),
)
import anyio
try:
    anyio.run(lambda: build_services(settings, data_dir=data))
except StorageLockedError as exc:
    print(exc.code, exc.http_status, flush=True)
else:
    print("ACQUIRED", flush=True)
"""


def _hold_lock(tmp_path: Any) -> Any:
    import subprocess
    import sys

    holder = subprocess.Popen(
        [sys.executable, "-c", LOCK_HOLDER, str(tmp_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "acquired", "the holder never took the lock"
    return holder


def test_a_second_process_on_the_same_data_dir_refuses_to_start(tmp_path: Any) -> None:
    """L2: one process owns data/. A second one fails fast instead of interleaving writes."""
    import subprocess
    import sys

    holder = _hold_lock(tmp_path)
    try:
        second = subprocess.run(
            [sys.executable, "-c", SECOND_INSTANCE, str(tmp_path), str(_fake_thresholds(tmp_path))],
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        holder.kill()
        holder.wait(timeout=30)

    assert "STORAGE_LOCKED 503" in second.stdout, (second.stdout, second.stderr[-500:])


def test_the_data_lock_is_reusable_after_the_holder_exits(
    tmp_path: Any, settings_factory: Any
) -> None:
    """A released lock must be takeable, or restarting the service would be impossible."""
    holder = _hold_lock(tmp_path)
    holder.kill()
    holder.wait(timeout=30)

    settings = settings_factory(
        app_env="test", embedding_provider="fake", thresholds_path=_fake_thresholds(tmp_path)
    )
    services = anyio.run(partial(build_services, settings, data_dir=tmp_path))
    anyio.run(services.aclose)


def test_services_expose_what_the_routes_need(services: Any) -> None:
    """The wiring is explicit, so a missing collaborator fails at build time, not per request."""
    for name in (
        "settings",
        "store",
        "vectors",
        "embedder",
        "thresholds",
        "registry",
        "chunker",
        "ingestion",
        "retrieval",
        "lock",
        "model_id",
        "dimension",
    ):
        assert getattr(services, name) is not None, name
    assert isinstance(services, Services)


def test_the_default_settings_still_build_an_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_app() with no injected services must still work; it is what uvicorn calls."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    app = create_app()
    assert app.title == "Document QA"

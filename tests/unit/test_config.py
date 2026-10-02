"""P0: config validation, secret redaction, provider rules."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from qasystem.config import Settings, load_settings
from qasystem.errors import ConfigError
from qasystem.logging_setup import REDACTED, configure_logging

ROOT = Path(__file__).resolve().parents[2]
SECRET = "sk-super-secret-token-value"
REMOTE = {"embedding_provider": "remote", "embedding_api_key": SECRET, "embedding_model": "bge-m3"}


def test_the_env_example_is_a_valid_config_once_copied() -> None:
    """`cp .env.example .env` is step two of the documented setup. It must not break.

    D31 again, one layer up. A green working tree hid this: the developer's own `.env` carries a
    real `EMBEDDING_DIMENSION=1024`, so the suite passed, while every fresh clone that followed
    the README failed six tests with ``embedding_dimension: Input should be a valid integer,
    unable to parse string as an integer``. The cause was ``EMBEDDING_DIMENSION=`` sitting in
    `.env.example` as an *empty* value: an empty string is a string, and ``int | None`` rejects
    it rather than reading it as unset. Only §0.11's clean-clone check could see it.

    ``extra="ignore"`` means a key the Settings class does not declare would pass silently, so
    this also checks that every key in the example is one the config actually reads.
    """
    example = ROOT / ".env.example"
    values: dict[str, str] = {}
    for line in example.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.split("#", 1)[0].strip()

    # `.env.example` spells them as ENV VARS; `Settings.model_fields` keys are lowercase field
    # names. `pydantic-settings` matches them case-insensitively, so this comparison must too.
    unknown = sorted(set(values) - {name.upper() for name in Settings.model_fields})
    assert not unknown, f".env.example sets keys Settings does not declare: {unknown}"

    # Every non-empty value must survive validation. Empty ones are checked separately below,
    # because an empty value is exactly how this broke.
    populated = {k: v for k, v in values.items() if v}
    settings = load_settings(
        env_file=None,
        app_env="test",
        embedding_provider="fake",
        embedding_api_key=None,
        embedding_model=None,
        **populated,
    )
    assert settings.data_dir == Path("./data")

    for key, value in values.items():
        if value:
            continue
        field = Settings.model_fields[key.lower()]
        # An empty value must mean "unset", not "a string that fails to parse".
        assert type(None) in _optional_types(field.annotation), (
            f".env.example ships {key}= as an empty value, but its type is "
            f'{field.annotation!r}, which cannot read "" as unset. Comment the line out '
            "instead, or the copy of this file fails to load."
        )


def _optional_types(annotation: object) -> tuple[object, ...]:
    """The union members of an annotation, or the annotation itself if it is not a union."""
    if isinstance(annotation, str):  # `from __future__ import annotations` defers evaluation
        return ()
    return getattr(annotation, "__args__", (annotation,))


def test_missing_api_key_is_a_controlled_config_error(settings_factory) -> None:
    with pytest.raises(ConfigError) as exc:
        settings_factory(embedding_provider="remote", embedding_model="bge-m3")
    assert "EMBEDDING_API_KEY" in str(exc.value)
    assert exc.value.code == "CONFIG_ERROR"


def test_missing_model_id_is_a_controlled_config_error(settings_factory) -> None:
    with pytest.raises(ConfigError, match="EMBEDDING_MODEL"):
        settings_factory(**{**REMOTE, "embedding_model": None})


@pytest.mark.invariant  # I8
def test_secret_never_appears_in_repr_or_serialisation(settings_factory) -> None:
    settings = settings_factory(**REMOTE)
    assert SECRET not in repr(settings)
    assert SECRET not in str(settings)
    assert SECRET not in settings.model_dump_json()
    assert SECRET not in json.dumps(settings.model_dump(mode="json"))
    assert settings.embedding_api_key is not None
    assert settings.embedding_api_key.get_secret_value() == SECRET


@pytest.mark.invariant  # I8
def test_secret_is_redacted_in_logs(tmp_path: Path, settings_factory) -> None:
    settings = settings_factory(**REMOTE)
    log_file = tmp_path / "app.log"
    with log_file.open("w") as stream:
        configure_logging(settings, level=logging.DEBUG, stream=stream)
        logging.getLogger("qasystem.test").error("calling %s with token=%s", "api", SECRET)
        logging.getLogger("qasystem.test").error("token is %s", SECRET)
        stream.flush()
    logged = log_file.read_text()
    assert SECRET not in logged
    assert logged.count(REDACTED) == 2


@pytest.mark.invariant  # I8
def test_redaction_preserves_non_string_args(tmp_path: Path, settings_factory) -> None:
    """httpx logs '... "%s %d %s"' with an int status code; str() would break it."""
    log_file = tmp_path / "app.log"
    with log_file.open("w") as stream:
        configure_logging(settings_factory(**REMOTE), stream=stream)
        logging.getLogger("qasystem.test").info(
            'HTTP Request: %s "%s %d %s"', "GET", "HTTP/1.1", 200, "OK"
        )
        stream.flush()
    assert '"HTTP/1.1 200 OK"' in log_file.read_text()


def test_fake_provider_is_rejected_in_prod(settings_factory) -> None:
    with pytest.raises(ConfigError, match="fake"):
        settings_factory(embedding_provider="fake", app_env="prod")


@pytest.mark.parametrize("app_env", ["test", "demo"])
def test_fake_provider_allowed_in_test_and_demo(settings_factory, app_env: str) -> None:
    settings = settings_factory(embedding_provider="fake", app_env=app_env)
    assert settings.embedding_api_key is None


def test_fake_provider_app_env_is_reported(settings_factory) -> None:
    settings = settings_factory(embedding_provider="fake", app_env="test")
    assert settings.is_fake_provider


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("max_chars_per_request", 200_001, "MAX_CHARS_PER_REQUEST"),
        ("rate_limit_per_min", 121, "RATE_LIMIT_PER_MIN"),
        ("max_upload_mb", 0, "MAX_UPLOAD_MB"),
        ("embedding_dimension", 0, "EMBEDDING_DIMENSION"),
        ("chunk_hard_max_tokens", 100, "CHUNK_HARD_MAX_TOKENS"),
    ],
)
def test_out_of_range_limits_are_rejected(
    settings_factory, field: str, value: int, needle: str
) -> None:
    with pytest.raises(ConfigError, match=needle):
        settings_factory(**{**REMOTE, field: value})


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("TOP_K", "7")
    settings = load_settings(env_file=None)
    assert settings.app_env == "test"
    assert settings.top_k == 7


def test_settings_load_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("APP_ENV=test\nEMBEDDING_PROVIDER=fake\nMAX_ANSWER_SENTENCES=3\n")
    assert load_settings(env_file=env_file).max_answer_sentences == 3


def test_secret_accessor_requires_a_key(settings_factory) -> None:
    with pytest.raises(ConfigError, match="EMBEDDING_API_KEY"):
        settings_factory(embedding_provider="fake", app_env="test").api_key()


def test_secret_accessor_returns_the_raw_key(settings_factory) -> None:
    assert settings_factory(**REMOTE).api_key() == SECRET


def test_defaults_match_plan(settings_factory) -> None:
    settings = settings_factory(**REMOTE)
    assert settings.chunk_target_tokens == 350
    assert settings.chunk_overlap_ratio == pytest.approx(0.15)
    assert settings.candidates_n == 30
    assert settings.top_k == 5
    assert settings.rrf_k == 60
    # §9.4 experiment 1 measured these on the eval dev split; the superseded 0.7/0.3 is
    # in docs/eval_report.md and D59 rather than left as a comment that can drift.
    assert settings.dense_weight == pytest.approx(0.9)
    assert settings.lexical_weight == pytest.approx(0.1)
    assert settings.max_answer_sentences == 5
    assert settings.chroma_path == Path("./data/chroma")
    assert settings.sqlite_path == Path("./data/qasystem.db")
    assert settings.max_upload_mb == 20


def test_embedding_client_defaults_match_the_measured_provider_caps(settings_factory) -> None:
    """§2.1: 200 000 chars/request, 120 req/min, and a per-item cap below the 45k failure."""
    settings = settings_factory(**REMOTE)
    assert settings.max_chars_per_request == 160_000
    assert settings.max_chars_per_item == 20_000
    assert settings.max_items_per_batch == 32
    assert settings.rate_limit_per_min == 100
    assert settings.max_retries == 5
    assert settings.request_timeout_s == 30


def test_a_non_positive_per_item_cap_is_rejected(settings_factory) -> None:
    with pytest.raises(ConfigError, match="MAX_CHARS_PER_ITEM"):
        settings_factory(**{**REMOTE, "max_chars_per_item": 0})


def test_the_per_item_cap_may_not_exceed_the_request_cap(settings_factory) -> None:
    with pytest.raises(ConfigError, match="MAX_CHARS_PER_ITEM"):
        settings_factory(**{**REMOTE, "max_chars_per_item": 200_000})

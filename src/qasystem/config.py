"""Application configuration, validated once at startup (plan.md §6).

Every tunable lives here — never inline. The API token is a ``SecretStr`` read
from the environment only; it never reaches logs, repr, or error bodies (I8).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from qasystem.errors import ConfigError

# Provider limits from jobTask.md (C5); config must never exceed them.
PROVIDER_MAX_CHARS_PER_REQUEST = 200_000
PROVIDER_MAX_REQUESTS_PER_MIN = 120
FAKE_PROVIDER_ENVS = frozenset({"test", "demo"})


class Settings(BaseSettings):
    """Validated settings; ``extra`` values in the environment are ignored."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    app_env: Literal["dev", "test", "demo", "prod"] = "dev"
    sentence_rerank: bool = False

    # --- embedding service (embeddings only: there is no chat endpoint, C3) ---
    embedding_base_url: str = "https://models-interview.arvancloudai.ir/v1"
    embedding_api_key: SecretStr | None = None
    embedding_model: str | None = None
    embedding_provider: Literal["remote", "fake"] = "remote"
    embedding_dimension: int | None = None

    # --- storage (all local) ---
    data_dir: Path = Path("./data")
    sqlite_path: Path = Path("./data/qasystem.db")
    chroma_path: Path = Path("./data/chroma")
    max_upload_mb: int = 20

    # --- chunking ---
    chunk_target_tokens: int = 350
    chunk_hard_max_tokens: int = 700
    chunk_overlap_ratio: float = 0.15

    # --- embedding client ---
    max_chars_per_request: int = 160_000
    max_items_per_batch: int = 32
    rate_limit_per_min: int = 100
    request_timeout_s: float = 30
    max_retries: int = 5

    # --- retrieval ---
    candidates_n: int = 30
    overfetch: int = 2
    top_k: int = 5
    rrf_k: int = 60
    dense_weight: float = 0.7
    lexical_weight: float = 0.3

    # --- gate + answer ---
    thresholds_path: Path = Path("./config/thresholds.json")
    max_answer_sentences: int = 5

    @property
    def is_fake_provider(self) -> bool:
        """True when the offline deterministic embedder is in use (shown in /ready)."""
        return self.embedding_provider == "fake"

    @model_validator(mode="after")
    def _validate(self) -> Settings:
        problems: list[str] = []

        if self.embedding_provider == "remote":
            # C4: model ids are never hard-coded; they come from GET /v1/models.
            if not self.embedding_api_key:
                problems.append("EMBEDDING_API_KEY is required when EMBEDDING_PROVIDER=remote")
            if not self.embedding_model:
                problems.append("EMBEDDING_MODEL is required when EMBEDDING_PROVIDER=remote")
        elif self.app_env not in FAKE_PROVIDER_ENVS:
            problems.append(
                f"EMBEDDING_PROVIDER=fake is only allowed when APP_ENV is one of "
                f"{sorted(FAKE_PROVIDER_ENVS)}, got {self.app_env}"
            )

        if not 0 < self.max_chars_per_request <= PROVIDER_MAX_CHARS_PER_REQUEST:
            problems.append(f"MAX_CHARS_PER_REQUEST must be in 1..{PROVIDER_MAX_CHARS_PER_REQUEST}")
        if not 0 < self.rate_limit_per_min <= PROVIDER_MAX_REQUESTS_PER_MIN:
            problems.append(f"RATE_LIMIT_PER_MIN must be in 1..{PROVIDER_MAX_REQUESTS_PER_MIN}")
        if self.max_upload_mb <= 0:
            problems.append("MAX_UPLOAD_MB must be positive")
        if self.max_items_per_batch <= 0:
            problems.append("MAX_ITEMS_PER_BATCH must be positive")
        if self.embedding_dimension is not None and self.embedding_dimension <= 0:
            problems.append("EMBEDDING_DIMENSION must be positive when set")
        if self.chunk_target_tokens <= 0:
            problems.append("CHUNK_TARGET_TOKENS must be positive")
        if self.chunk_hard_max_tokens < self.chunk_target_tokens:
            problems.append("CHUNK_HARD_MAX_TOKENS must be >= CHUNK_TARGET_TOKENS")
        if not 0 <= self.chunk_overlap_ratio < 1:
            problems.append("CHUNK_OVERLAP_RATIO must be in [0, 1)")
        if self.overfetch < 1 or self.top_k < 1 or self.candidates_n < 1:
            problems.append("CANDIDATES_N, OVERFETCH and TOP_K must be >= 1")
        if self.dense_weight + self.lexical_weight <= 0:
            problems.append("DENSE_WEIGHT + LEXICAL_WEIGHT must be > 0")
        if self.max_answer_sentences < 1:
            problems.append("MAX_ANSWER_SENTENCES must be >= 1")

        if problems:
            raise ValueError("; ".join(problems))
        return self

    def api_key(self) -> str:
        """Raw token for the HTTP client. Never log, print, or return in an error."""
        if self.embedding_api_key is None:
            raise ConfigError("EMBEDDING_API_KEY is not set")
        return self.embedding_api_key.get_secret_value()


def load_settings(*, env_file: str | Path | None = ".env", **overrides: Any) -> Settings:
    """Build ``Settings``, turning validation failures into a controlled ConfigError."""
    from pydantic import ValidationError

    try:
        return Settings(_env_file=env_file, **overrides)  # type: ignore[call-arg]
    except ValidationError as exc:
        # include_input=False keeps the raw token out of the message (I8).
        raise ConfigError(_summarize(exc)) from None


def _summarize(exc: Any) -> str:
    parts = []
    for err in exc.errors(include_url=False, include_input=False):
        field = ".".join(str(loc) for loc in err["loc"]) or "config"
        parts.append(f"{field}: {err['msg']}")
    return "; ".join(parts)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (env + .env)."""
    return load_settings()

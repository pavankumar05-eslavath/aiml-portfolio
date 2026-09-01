"""Application configuration.

All tunable behaviour is environment driven. This module is the single source of
truth for configuration; no other module should read ``os.environ`` directly.

Note on embedding dimensionality: it is deliberately *not* configured here. The
vector size is a property of the loaded model and is read from the model config
at runtime (see :class:`app.services.embedding.EmbeddingService`), so changing
``MODEL_NAME`` cannot desynchronise the Qdrant collection from the encoder.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class FusionStrategy(StrEnum):
    """Strategies for combining evidence from multiple retrieval channels."""

    WEIGHTED_SUM = "weighted_sum"
    RRF = "rrf"
    EMBEDDING_FUSION = "embedding_fusion"


class ScoreNormalization(StrEnum):
    """Per-channel score normalisation applied before weighted fusion.

    Motivated by measurement: same-modality cosine similarities (image-image
    0.7305 mean) and cross-modal ones (image-text 0.3124 mean) occupy different
    ranges for CLIP-family models, so combining them raw makes the configured
    weights misleading - the higher-scoring channel contributes more than its
    weight implies.

    The measured effect on ranking quality is real but modest: on the benchmark
    set ``zscore`` reached nDCG@10 0.668 against 0.655 for ``none``. Normalisation
    is kept on by default because it makes the weights mean what they say, not
    because it transforms retrieval quality.
    """

    NONE = "none"
    MINMAX = "minmax"
    ZSCORE = "zscore"


class Settings(BaseSettings):
    """Runtime settings, populated from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "Multimodal Product Search"
    app_env: Literal["local", "docker", "test", "production"] = "local"
    api_prefix: str = "/api"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"
    cors_origins: str = "http://localhost:5173,http://localhost:4173,http://localhost:3000"

    # ------------------------------------------------------------- database
    database_url: str = f"sqlite+aiosqlite:///{REPO_ROOT / 'data' / 'local' / 'catalog.db'}"
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # --------------------------------------------------------------- qdrant
    # If ``qdrant_url`` is set, the client connects to a Qdrant server.
    # Otherwise it runs Qdrant embedded, persisting to ``qdrant_path``.
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: str = str(REPO_ROOT / "data" / "local" / "qdrant")
    qdrant_collection: str = "products"
    qdrant_timeout: float = 30.0
    # HNSW / quantisation knobs kept minimal and documented in docs/architecture.md
    qdrant_hnsw_m: int = 16
    qdrant_hnsw_ef_construct: int = 128

    # ---------------------------------------------------------------- model
    model_name: str = "openai/clip-vit-base-patch32"
    model_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    model_cache_dir: str | None = None
    # Number of torch intra-op threads. 0 leaves torch's default untouched.
    torch_num_threads: int = 0
    embedding_batch_size: int = 32
    # Bounds concurrent forward passes so request bursts cannot thrash the CPU.
    max_concurrent_inferences: int = 2

    # ----------------------------------------------------- ranking / fusion
    # Every default below is the measured optimum from scripts/run_experiments.py
    # on the benchmark query set, not a guess. See evaluation/results/experiments.md.
    fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_SUM
    score_normalization: ScoreNormalization = ScoreNormalization.ZSCORE

    #: Weight of the image side of the query, used only when a query carries BOTH
    #: an image and text. 0.2 maximises mean nDCG@10 across the two multimodal
    #: query styles, which have very different optima:
    #:   contradiction ("this, but in black") peaks at 0.1 -> nDCG@10 0.429
    #:   agreement     (text restates the image) peaks at 0.5 -> nDCG@10 0.831
    #: Fusion beats both single modalities at each group's optimum, but a single
    #: global value cannot be optimal for both, so clients override per request.
    #: Note this default is sensitive to the benchmark's 50/50 mix of the two styles.
    image_weight: float = 0.2
    #: Weight of the text side of the query in multimodal search.
    text_weight: float = 0.8
    #: Weight of the lexical (keyword) channel. 0 disables it.
    #: Measured off by default: combined with z-score normalisation the channel was
    #: neutral-to-negative (nDCG@10 0.677 off vs 0.669 at 0.2) and roughly tripled
    #: median latency, because the SQLite fallback scores candidates in Python.
    #: Worth re-measuring on PostgreSQL, where the channel is real full-text search.
    lexical_weight: float = 0.0
    #: Within one query modality, how much weight goes to the *cross*-modal
    #: channel (e.g. text query against product image vector). The remainder
    #: goes to the same-modality channel.
    cross_modal_weight: float = 0.3
    #: Alpha for FusionStrategy.EMBEDDING_FUSION: alpha * image + (1-alpha) * text.
    embedding_fusion_alpha: float = 0.5
    #: Rank constant for reciprocal rank fusion.
    rrf_k: int = 60
    #: Per-channel candidate pool = top_k * candidate_multiplier (bounded below).
    candidate_multiplier: int = 4
    min_candidate_pool: int = 50
    default_top_k: int = 20
    max_top_k: int = 100

    # ---------------------------------------------------------- image input
    max_upload_bytes: int = 10 * 1024 * 1024
    allowed_image_formats: str = "JPEG,PNG,WEBP"
    #: Longest edge the uploaded image is reduced to before preprocessing.
    max_image_dimension: int = 1024
    min_image_dimension: int = 8

    # ---------------------------------------------------------------- cache
    #: In-process LRU cache of text-query embeddings. 0 disables.
    text_embedding_cache_size: int = 512

    # ------------------------------------------------------------ ingestion
    ingest_batch_size: int = 64
    #: Where ingestion looks for product images referenced by relative path.
    image_root: str = str(REPO_ROOT / "data")

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        level = v.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level: {v}")
        return level

    @field_validator("image_weight", "text_weight", "lexical_weight")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("weights must be non-negative")
        return v

    @field_validator("cross_modal_weight", "embedding_fusion_alpha")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be within [0, 1]")
        return v

    @model_validator(mode="after")
    def _check_weights(self) -> Settings:
        if self.image_weight + self.text_weight + self.lexical_weight <= 0:
            raise ValueError(
                "at least one of IMAGE_WEIGHT, TEXT_WEIGHT, LEXICAL_WEIGHT must be > 0"
            )
        if self.max_top_k < self.default_top_k:
            raise ValueError("MAX_TOP_K must be >= DEFAULT_TOP_K")
        return self

    # ----------------------------------------------------------- accessors
    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origin_list(self) -> list[str]:
        """Allowed CORS origins, parsed from the comma-separated setting."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allowed_image_format_set(self) -> set[str]:
        """Uppercased Pillow format names accepted for uploads."""
        return {f.strip().upper() for f in self.allowed_image_formats.split(",") if f.strip()}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def uses_qdrant_server(self) -> bool:
        """Whether Qdrant runs as a server (rather than embedded in-process)."""
        return bool(self.qdrant_url)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_postgres(self) -> bool:
        """Whether the configured database is PostgreSQL."""
        return self.database_url.startswith(("postgresql", "postgres"))

    def candidate_pool_size(self, top_k: int) -> int:
        """Number of candidates to pull per channel before fusion and filtering."""
        return max(self.min_candidate_pool, top_k * self.candidate_multiplier)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()

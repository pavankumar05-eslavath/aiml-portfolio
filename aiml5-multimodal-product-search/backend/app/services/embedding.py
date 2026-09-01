"""Multimodal embedding service.

Responsibilities
----------------
* Load one encoder per process (never per request) and expose its true output
  dimensionality, so the Qdrant collection can never drift from the model.
* Produce L2-normalised image, text and fused embeddings.
* Keep blocking torch work off the event loop, bounded by a semaphore.

Model compatibility
-------------------
Image/text fusion is only meaningful when both towers project into a *shared*
space. That is a property of the training objective (CLIP/SigLIP-style
contrastive or sigmoid pairing), not something to assume, so
:meth:`EmbeddingService.probe_alignment` measures it and
``scripts/verify_embedding_space.py`` reports it.

Measured on the committed sample catalogue with ``openai/clip-vit-base-patch32``
(48 products): image->text top-1 retrieval **0.812**, matched pairs **0.3124** vs
mismatched **0.1794**. Same-modality cosines sit far above cross-modal ones
(image-image **0.7305**, text-text **0.4499**, image-text **0.3124**) - the
modality gap that :mod:`app.services.fusion` normalises away.

Only architectures exposing ``get_image_features`` / ``get_text_features`` are
supported; anything else is rejected at load time rather than silently producing
embeddings from unrelated spaces.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import anyio
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from app.core.config import Settings, get_settings
from app.core.exceptions import EmbeddingError, ModelNotReadyError
from app.core.logging import get_logger, log_duration
from app.core.text import build_search_document

logger = get_logger(__name__)

#: Float32 embedding array. numpy types arithmetic results as ``Any``, so the
#: annotations below are what keep this module checkable under ``mypy --strict``.
FloatArray = np.ndarray[Any, np.dtype[np.float32]]

#: Architectures verified to expose a shared image/text projection space.
SHARED_SPACE_MODEL_TYPES: Final = frozenset(
    {"clip", "siglip", "siglip2", "altclip", "chinese_clip", "blip", "owlvit"}
)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Metadata describing the loaded encoder."""

    name: str
    model_type: str
    embedding_dim: int
    device: str
    dtype: str
    image_size: int | None = None


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """Empirical evidence that the two towers share a space."""

    pairs: int
    cross_modal_top1: float
    matched_pair_similarity: float
    mismatched_pair_similarity: float
    image_image_similarity: float
    text_text_similarity: float

    @property
    def is_aligned(self) -> bool:
        """True when matched pairs beat mismatched pairs and top-1 is usable."""
        return (
            self.cross_modal_top1 >= 0.5
            and self.matched_pair_similarity > self.mismatched_pair_similarity
        )


def _resolve_device(requested: str) -> str:
    """Pick a torch device, degrading to CPU when accelerators are absent."""
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            return "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but unavailable; falling back to CPU")
            return "cpu"
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def l2_normalise(vectors: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    """L2-normalise along the last axis.

    Normalising once here means cosine similarity reduces to a dot product
    everywhere downstream, and lets Qdrant's COSINE distance and our own fusion
    arithmetic agree exactly.
    """
    array: FloatArray = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    normalised: FloatArray = array / np.maximum(norms, epsilon)
    return normalised


class _LruCache:
    """Small thread-safe LRU cache for text embeddings."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> np.ndarray | None:
        if self.capacity <= 0:
            return None
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key: str, value: np.ndarray) -> None:
        if self.capacity <= 0:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"size": len(self._data), "hits": self.hits, "misses": self.misses}


class EmbeddingService:
    """Encodes images and text into a single shared vector space."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._info: ModelInfo | None = None
        self._load_lock = threading.Lock()
        self._limiter = anyio.CapacityLimiter(max(1, self._settings.max_concurrent_inferences))
        self._text_cache = _LruCache(self._settings.text_embedding_cache_size)

    # ------------------------------------------------------------- lifecycle
    @property
    def is_loaded(self) -> bool:
        """Whether the encoder is resident in memory."""
        return self._model is not None

    @property
    def configured_model_name(self) -> str:
        """Model identifier from configuration, readable before loading."""
        return self._settings.model_name

    @property
    def info(self) -> ModelInfo:
        """Metadata for the loaded encoder."""
        if self._info is None:
            raise ModelNotReadyError("The embedding model has not been loaded yet.")
        return self._info

    @property
    def embedding_dim(self) -> int:
        """Output dimensionality of both towers (single source of truth)."""
        return self.info.embedding_dim

    def load(self) -> ModelInfo:
        """Load the encoder. Idempotent, thread-safe, and blocking.

        Raises:
            ModelNotReadyError: if weights cannot be loaded, or the architecture
                does not expose a shared image/text space.
        """
        info = self._info
        if info is not None:
            return info
        with self._load_lock:
            # Double-checked locking: another thread may have loaded the model
            # while this one waited.
            info = self._info
            if info is not None:
                return info

            settings = self._settings
            if settings.torch_num_threads > 0:
                torch.set_num_threads(settings.torch_num_threads)

            device = _resolve_device(settings.model_device)
            with log_duration(
                logger, "embedding model loaded", model=settings.model_name, device=device
            ) as ctx:
                try:
                    model = AutoModel.from_pretrained(
                        settings.model_name, cache_dir=settings.model_cache_dir
                    )
                    # AutoProcessor.from_pretrained is untyped in transformers'
                    # bundled stubs; AutoModel's equivalent is not.
                    processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                        settings.model_name,
                        cache_dir=settings.model_cache_dir,
                        use_fast=True,
                    )
                except Exception as exc:
                    raise ModelNotReadyError(
                        f"Failed to load model {settings.model_name!r}: {exc}"
                    ) from exc

                for required in ("get_image_features", "get_text_features"):
                    if not hasattr(model, required):
                        raise ModelNotReadyError(
                            f"Model {settings.model_name!r} does not expose {required}(); "
                            "a dual-encoder model with a shared image/text projection "
                            "is required (e.g. CLIP, SigLIP, OpenCLIP)."
                        )

                model_type = getattr(model.config, "model_type", "unknown")
                if model_type not in SHARED_SPACE_MODEL_TYPES:
                    logger.warning(
                        "model architecture is not on the verified shared-space list; "
                        "run scripts/verify_embedding_space.py before trusting fusion",
                        extra={"context": {"model_type": model_type}},
                    )

                model.eval()
                model.to(device)
                dim = self._infer_embedding_dim(model)

                self._model = model
                self._processor = processor
                self._info = ModelInfo(
                    name=settings.model_name,
                    model_type=model_type,
                    embedding_dim=dim,
                    device=device,
                    dtype=str(next(model.parameters()).dtype).removeprefix("torch."),
                    image_size=self._infer_image_size(processor),
                )
                ctx["embedding_dim"] = dim
                ctx["model_type"] = model_type
            return self._info

    @staticmethod
    def _infer_embedding_dim(model: Any) -> int:
        """Determine the projection dimension, preferring a real forward pass.

        Config attribute names differ across architectures, so a one-token probe
        is the only fully reliable answer.
        """
        try:
            with torch.no_grad():
                ids = torch.ones((1, 2), dtype=torch.long, device=model.device)
                features = model.get_text_features(input_ids=ids)
            return int(features.shape[-1])
        except Exception as probe_error:  # fall back to config introspection
            config = model.config
            for attr in ("projection_dim", "hidden_size"):
                value = getattr(config, attr, None)
                if isinstance(value, int) and value > 0:
                    return value
            for sub in ("text_config", "vision_config"):
                sub_config = getattr(config, sub, None)
                value = getattr(sub_config, "hidden_size", None)
                if isinstance(value, int) and value > 0:
                    return value
            raise ModelNotReadyError(
                "Could not determine the model's embedding dimension."
            ) from probe_error

    @staticmethod
    def _infer_image_size(processor: Any) -> int | None:
        image_processor = getattr(processor, "image_processor", processor)
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            for key in ("shortest_edge", "height", "width"):
                if isinstance(size.get(key), int):
                    return int(size[key])
        return size if isinstance(size, int) else None

    def unload(self) -> None:
        """Release the encoder and cached embeddings."""
        self._model = None
        self._processor = None
        self._info = None
        self._text_cache = _LruCache(self._settings.text_embedding_cache_size)

    def _require_loaded(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            raise ModelNotReadyError("The embedding model has not been loaded yet.")
        return self._model, self._processor

    # -------------------------------------------------------- sync encoding
    def embed_texts(self, texts: Sequence[str], *, batch_size: int | None = None) -> np.ndarray:
        """Embed texts into L2-normalised vectors. Blocking.

        Returns:
            Array of shape ``(len(texts), embedding_dim)``.
        """
        model, processor = self._require_loaded()
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        size = batch_size or self._settings.embedding_batch_size
        chunks: list[np.ndarray] = []
        try:
            for start in range(0, len(texts), size):
                batch = [t if t and t.strip() else " " for t in texts[start : start + size]]
                inputs = processor(
                    text=batch, return_tensors="pt", padding=True, truncation=True
                ).to(model.device)
                with torch.no_grad():
                    features = model.get_text_features(**inputs)
                chunks.append(features.detach().float().cpu().numpy())
        except Exception as exc:
            raise EmbeddingError(f"Text embedding failed: {exc}") from exc
        return l2_normalise(np.concatenate(chunks, axis=0))

    def embed_images(
        self, images: Sequence[Image.Image], *, batch_size: int | None = None
    ) -> np.ndarray:
        """Embed PIL images into L2-normalised vectors. Blocking."""
        model, processor = self._require_loaded()
        if not images:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        size = batch_size or self._settings.embedding_batch_size
        chunks: list[np.ndarray] = []
        try:
            for start in range(0, len(images), size):
                batch = [im.convert("RGB") for im in images[start : start + size]]
                inputs = processor(images=batch, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    features = model.get_image_features(**inputs)
                chunks.append(features.detach().float().cpu().numpy())
        except Exception as exc:
            raise EmbeddingError(f"Image embedding failed: {exc}") from exc
        return l2_normalise(np.concatenate(chunks, axis=0))

    # ------------------------------------------------------- async encoding
    async def embed_text(self, text: str) -> np.ndarray:
        """Embed a single query string, using the LRU cache when warm."""
        document = build_search_document(text)
        key = hashlib.sha256(
            f"{self.info.name}\x00{document}".encode(errors="replace")
        ).hexdigest()
        if (cached := self._text_cache.get(key)) is not None:
            return cached
        vectors = await self.embed_texts_async([document])
        vector: FloatArray = vectors[0]
        self._text_cache.put(key, vector)
        return vector

    async def embed_texts_async(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts without blocking the event loop."""
        return await anyio.to_thread.run_sync(
            lambda: self.embed_texts(texts), limiter=self._limiter
        )

    async def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a single image without blocking the event loop."""
        vectors = await self.embed_images_async([image])
        vector: FloatArray = vectors[0]
        return vector

    async def embed_images_async(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Embed images without blocking the event loop."""
        return await anyio.to_thread.run_sync(
            lambda: self.embed_images(images), limiter=self._limiter
        )

    # ----------------------------------------------------------------- fuse
    def fuse(
        self, image_vector: np.ndarray, text_vector: np.ndarray, alpha: float
    ) -> np.ndarray:
        """Blend an image and a text embedding into one query vector.

        Implements ``alpha * image + (1 - alpha) * text`` followed by
        re-normalisation. Re-normalising matters: the convex combination of two
        unit vectors has norm < 1 (as low as 0.5 for opposing vectors at
        alpha=0.5), so skipping it would make the fused query's cosine scores
        incomparable across requests.

        This is one of three supported strategies; see :mod:`app.services.fusion`
        for why score-level fusion is the default.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be within [0, 1]")
        image_vector = l2_normalise(image_vector)[0]
        text_vector = l2_normalise(text_vector)[0]
        if image_vector.shape != text_vector.shape:
            raise EmbeddingError(
                "cannot fuse embeddings of differing shapes: "
                f"{image_vector.shape} vs {text_vector.shape}"
            )
        fused: FloatArray = l2_normalise(alpha * image_vector + (1.0 - alpha) * text_vector)[0]
        return fused

    async def embed_multimodal(
        self, image: Image.Image, text: str, alpha: float | None = None
    ) -> np.ndarray:
        """Embed an image and text, returning the fused query vector."""
        alpha = self._settings.embedding_fusion_alpha if alpha is None else alpha
        image_vector = await self.embed_image(image)
        text_vector = await self.embed_text(text)
        return self.fuse(image_vector, text_vector, alpha)

    # ------------------------------------------------------------ diagnostics
    def probe_alignment(
        self, images: Sequence[Image.Image], texts: Sequence[str]
    ) -> AlignmentReport:
        """Measure whether the two towers actually share a space.

        Args:
            images: Product images.
            texts: Their corresponding documents; ``texts[i]`` must describe
                ``images[i]``.

        Returns:
            An :class:`AlignmentReport`. ``cross_modal_top1`` is the fraction of
            images whose highest-scoring text is their own.
        """
        if len(images) != len(texts):
            raise ValueError("images and texts must be the same length")
        if len(images) < 2:
            raise ValueError("need at least two pairs to measure alignment")

        image_vectors = self.embed_images(images)
        text_vectors = self.embed_texts(texts)
        cross = image_vectors @ text_vectors.T
        n = len(images)
        off_diagonal = ~np.eye(n, dtype=bool)

        return AlignmentReport(
            pairs=n,
            cross_modal_top1=float((cross.argmax(axis=1) == np.arange(n)).mean()),
            matched_pair_similarity=float(cross.diagonal().mean()),
            mismatched_pair_similarity=float(cross[off_diagonal].mean()),
            image_image_similarity=float(
                (image_vectors @ image_vectors.T)[off_diagonal].mean()
            ),
            text_text_similarity=float((text_vectors @ text_vectors.T)[off_diagonal].mean()),
        )

    def cache_stats(self) -> dict[str, int]:
        """Text-embedding cache counters."""
        return self._text_cache.stats()

    def warmup(self) -> float:
        """Run one tiny forward pass per tower so the first request is not slow.

        Returns:
            Seconds spent warming up.
        """
        started = time.perf_counter()
        self.embed_texts(["warmup"])
        self.embed_images([Image.new("RGB", (64, 64), (127, 127, 127))])
        elapsed = time.perf_counter() - started
        logger.info("model warmup complete", extra={"context": {"duration_ms": elapsed * 1000}})
        return elapsed


_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Return the process-wide embedding service (not yet loaded)."""
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service


def reset_embedding_service() -> None:
    """Drop the singleton. Used by tests."""
    global _service
    if _service is not None:
        _service.unload()
    _service = None

"""Catalogue indexing pipeline.

For each batch of products the service:

1. Fingerprints the embedding-relevant fields (``content_hash``).
2. Skips products whose fingerprint is unchanged - this is what makes
   re-indexing an unchanged catalogue nearly free.
3. Resolves and loads the product image (local path or HTTP URL).
4. Builds the canonical text document.
5. Embeds images and texts **in batches**, which is markedly faster per item than
   one forward pass per product.
6. Writes both named vectors to Qdrant and the indexing outcome to the database.

Partial success is a first-class outcome: a product whose image is missing or
corrupt is still indexed on its text vector, so it stays discoverable by text
search. Failures are recorded on the row (``index_error``) as well as logged, so
they can be queried later instead of being lost in a log stream.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import anyio
import httpx
from PIL import Image

from app.core.config import Settings, get_settings
from app.core.exceptions import AppError, IndexingError
from app.core.logging import get_logger
from app.models.product import Product
from app.repositories.product_repository import (
    ProductRepository,
    build_search_document_for,
    compute_content_hash,
)
from app.repositories.vector_repository import (
    IMAGE_VECTOR,
    TEXT_VECTOR,
    VectorRepository,
)
from app.schemas.catalog import IndexFailure, IndexReport
from app.services.embedding import EmbeddingService
from app.services.image_processing import ImageProcessor

logger = get_logger(__name__)

ProgressCallback = Callable[["IndexProgress"], None]


@dataclass(frozen=True, slots=True)
class IndexProgress:
    """Progress snapshot emitted after each batch."""

    processed: int
    total: int
    embedded: int
    skipped: int
    failed: int
    elapsed_seconds: float

    @property
    def percent(self) -> float:
        """Completion percentage."""
        return 100.0 * self.processed / self.total if self.total else 100.0

    @property
    def rate(self) -> float:
        """Products processed per second."""
        return self.processed / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


@dataclass(slots=True)
class _Prepared:
    """A product with its embedding inputs resolved."""

    product: Product
    document: str
    content_hash: str
    image: Image.Image | None = None
    image_path: str | None = None
    note: str | None = None


@dataclass(slots=True)
class _Counters:
    """Mutable tallies for one indexing run."""

    requested: int = 0
    embedded: int = 0
    skipped: int = 0
    failed: int = 0
    image_vectors: int = 0
    text_vectors: int = 0
    failures: list[IndexFailure] = field(default_factory=list)


class IndexingService:
    """Embeds catalogue products and writes them to the vector store."""

    def __init__(
        self,
        *,
        products: ProductRepository,
        vectors: VectorRepository,
        embedder: EmbeddingService,
        images: ImageProcessor,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._products = products
        self._vectors = vectors
        self._embedder = embedder
        self._images = images
        self._settings = settings or get_settings()
        self._http = http_client

    async def index_catalog(
        self,
        *,
        product_ids: Sequence[str] | None = None,
        force: bool = False,
        limit: int | None = None,
        recreate: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IndexReport:
        """Index products that need embedding.

        Args:
            product_ids: Restrict to these products; ``None`` means the catalogue.
            force: Re-embed even when the fingerprint is unchanged.
            limit: Process at most this many products.
            recreate: Drop and rebuild the Qdrant collection first. Also clears
                stored fingerprints, since the vectors they referred to are gone.
            progress: Called after each batch.

        Returns:
            A report tallying embedded, skipped and failed products.
        """
        started = time.perf_counter()
        counters = _Counters()

        # The model owns the dimensionality; the collection is built to match.
        self._embedder.load()
        await self._vectors.ensure_collection(self._embedder.embedding_dim, recreate=recreate)
        if recreate:
            await self._products.clear_index_state()

        pending = await self._products.iter_for_indexing(
            product_ids=product_ids, force=force, limit=limit
        )
        counters.requested = len(pending)
        if not pending:
            logger.info("nothing to index")
            return self._build_report(counters, started)

        logger.info(
            "indexing started",
            extra={
                "context": {
                    "products": len(pending),
                    "force": force,
                    "model": self._embedder.info.name,
                    "dim": self._embedder.embedding_dim,
                    "batch_size": self._settings.ingest_batch_size,
                }
            },
        )

        batch_size = max(1, self._settings.ingest_batch_size)
        processed = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            await self._process_batch(batch, force=force, counters=counters)
            processed += len(batch)
            snapshot = IndexProgress(
                processed=processed,
                total=len(pending),
                embedded=counters.embedded,
                skipped=counters.skipped,
                failed=counters.failed,
                elapsed_seconds=time.perf_counter() - started,
            )
            logger.info(
                "indexing progress",
                extra={
                    "context": {
                        "processed": processed,
                        "total": len(pending),
                        "percent": round(snapshot.percent, 1),
                        "rate_per_s": round(snapshot.rate, 2),
                        "failed": counters.failed,
                    }
                },
            )
            if progress is not None:
                progress(snapshot)

        report = self._build_report(counters, started)
        logger.info(
            "indexing finished",
            extra={
                "context": {
                    "embedded": report.embedded,
                    "skipped": report.skipped_unchanged,
                    "failed": report.failed,
                    "duration_ms": report.duration_ms,
                }
            },
        )
        return report

    # ---------------------------------------------------------------- batches
    async def _process_batch(
        self, batch: Sequence[Product], *, force: bool, counters: _Counters
    ) -> None:
        """Prepare, embed and persist one batch."""
        prepared: list[_Prepared] = []

        for product in batch:
            document = build_search_document_for(product)
            content_hash = compute_content_hash(product, self._embedder.info.name)

            if (
                not force
                and product.content_hash == content_hash
                and product.indexed_at is not None
                and product.has_text_vector
            ):
                counters.skipped += 1
                continue

            if not document.strip():
                counters.failed += 1
                counters.failures.append(
                    IndexFailure(
                        product_id=product.id,
                        name=product.name,
                        reason="Product has no embeddable text.",
                    )
                )
                await self._products.mark_index_failed(product, "no embeddable text")
                continue

            image, image_path, note = await self._load_image(product)
            prepared.append(
                _Prepared(
                    product=product,
                    document=document,
                    content_hash=content_hash,
                    image=image,
                    image_path=image_path,
                    note=note,
                )
            )

        if not prepared:
            return

        try:
            text_vectors = await self._embedder.embed_texts_async(
                [item.document for item in prepared]
            )
            with_images = [item for item in prepared if item.image is not None]
            image_vectors = (
                await self._embedder.embed_images_async([item.image for item in with_images])  # type: ignore[misc]
                if with_images
                else None
            )
        except AppError as exc:
            # An embedding failure is a batch-level fault (bad model state), not a
            # per-product data problem, so surface it rather than mark 64 rows bad.
            raise IndexingError(f"Embedding failed for a batch of products: {exc}") from exc

        image_vector_by_id = {}
        if image_vectors is not None:
            image_vector_by_id = {
                item.product.id: image_vectors[index] for index, item in enumerate(with_images)
            }

        records = []
        for index, item in enumerate(prepared):
            product = item.product
            image_vector = image_vector_by_id.get(product.id)
            vectors: dict[str, list[float]] = {TEXT_VECTOR: text_vectors[index].tolist()}
            if image_vector is not None:
                vectors[IMAGE_VECTOR] = image_vector.tolist()

            product.search_document = item.document
            records.append((product.id, vectors, product.vector_payload()))

        await self._vectors.upsert(records)

        for item in prepared:
            product = item.product
            has_image = product.id in image_vector_by_id
            await self._products.mark_indexed(
                product,
                content_hash=item.content_hash,
                document=item.document,
                has_image_vector=has_image,
                has_text_vector=True,
                image_path=item.image_path,
            )
            if item.note:
                # Text indexing succeeded; record why the image did not.
                product.index_error = item.note
            counters.embedded += 1
            counters.text_vectors += 1
            if has_image:
                counters.image_vectors += 1
            else:
                counters.failures.append(
                    IndexFailure(
                        product_id=product.id,
                        name=product.name,
                        reason=item.note or "no usable image; indexed on text only",
                    )
                )

    # ----------------------------------------------------------------- images
    async def _load_image(
        self, product: Product
    ) -> tuple[Image.Image | None, str | None, str | None]:
        """Resolve and load a product image.

        Returns:
            ``(image, local_path, note)``. ``image`` is ``None`` when no usable
            image exists, in which case ``note`` explains why and the product is
            still indexed on text.
        """
        reference = product.image_path or product.image_url
        if not reference:
            return None, None, "no image_url set; indexed on text only"

        if local := self._images.resolve_catalog_path(reference):
            try:
                loaded = await anyio.to_thread.run_sync(
                    lambda: self._images.load_from_path(local)
                )
            except AppError as exc:
                return None, None, f"image unreadable ({exc.message}); indexed on text only"
            return loaded.image, str(local), None

        if reference.startswith(("http://", "https://")):
            try:
                data = await self._fetch_image(reference)
            except Exception as exc:
                return None, None, f"image download failed ({type(exc).__name__})"
            try:
                loaded = await anyio.to_thread.run_sync(
                    lambda: self._images.load_from_bytes(data, filename=reference)
                )
            except AppError as exc:
                return None, None, f"downloaded image invalid ({exc.message})"
            return loaded.image, None, None

        return None, None, f"image not found at {reference!r}; indexed on text only"

    async def _fetch_image(self, url: str) -> bytes:
        """Download an image, refusing oversized responses.

        The ``Content-Length`` check rejects large images before transfer; the
        streaming cap defends against responses that omit or understate it.
        """
        limit = self._settings.max_upload_bytes
        client = self._http
        owned = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=httpx.Timeout(20.0), follow_redirects=True)
        try:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                declared = response.headers.get("content-length")
                if declared and int(declared) > limit:
                    raise IndexingError(f"remote image exceeds {limit} bytes")
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > limit:
                        raise IndexingError(f"remote image exceeds {limit} bytes")
                return bytes(chunks)
        finally:
            if owned:
                await client.aclose()

    # ---------------------------------------------------------------- reports
    @staticmethod
    def _build_report(counters: _Counters, started: float) -> IndexReport:
        return IndexReport(
            requested=counters.requested,
            embedded=counters.embedded,
            skipped_unchanged=counters.skipped,
            failed=counters.failed,
            image_vectors=counters.image_vectors,
            text_vectors=counters.text_vectors,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            # Bound the response: the full picture stays in the logs and on the rows.
            failures=counters.failures[:50],
        )

    async def remove_products(self, product_ids: Sequence[str]) -> int:
        """Delete vectors for the given products."""
        return await self._vectors.delete(product_ids)

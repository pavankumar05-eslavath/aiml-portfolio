"""Shared test fixtures.

Testing strategy
----------------
The database and the vector store are **real** (SQLite on a temp file, Qdrant in
embedded mode on a temp directory). Only the encoder is replaced, by
:class:`FakeEmbeddingService`, which returns deterministic vectors derived from a
hash of its input.

That split is deliberate. Loading CLIP costs ~600 MB and seconds per session, and
it would make assertions depend on model behaviour rather than on our code. What
these tests need to verify is the part we wrote: that named vectors are written and
queried correctly, that payload filters are pushed into Qdrant, that fusion
arithmetic is right, and that the API contract holds. A fake encoder with stable,
controllable geometry tests all of that *harder* than the real model would,
because exact expectations can be constructed.

The real model is covered separately by ``test_embedding_real.py``, marked ``slow``
and skipped unless ``RUN_SLOW_TESTS=1``.

Configuration is injected through **environment variables**, not only through
FastAPI dependency overrides. Some code paths (the health endpoint, the default
constructors of the repositories) call ``get_settings()`` directly, so overriding
the dependency alone would leave them pointing at the developer's real database.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Sequence
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import create_engine, dispose_engine, get_db_session
from app.main import create_app
from app.models.product import Product, new_product_id
from app.repositories.product_repository import ProductRepository
from app.repositories.vector_repository import VectorRepository, get_vector_repository
from app.services.catalog_service import CatalogService
from app.services.embedding import EmbeddingService, ModelInfo, l2_normalise
from app.services.image_processing import ImageProcessor, get_image_processor
from app.services.indexing_service import IndexingService
from app.services.search_service import SearchService

FAKE_DIM = 32
FAKE_MODEL = "test/fake-clip"


# --------------------------------------------------------------- fake encoder
class FakeEmbeddingService(EmbeddingService):
    """Deterministic stand-in for the real encoder.

    Vectors are derived from a SHA-256 digest of the input, so:

    * the same input always yields the same vector (tests are reproducible);
    * different inputs yield near-orthogonal vectors (ranking is meaningful);
    * a caller can pin a *known* vector via :meth:`register` to construct exact
      similarity expectations.

    Images are hashed from downscaled pixel bytes, so visually identical images
    embed identically while different images do not.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._seeded: dict[str, np.ndarray] = {}
        self.text_calls = 0
        self.image_calls = 0

    def register(self, key: str, vector: Sequence[float]) -> np.ndarray:
        """Pin a specific vector for a given text input."""
        normalised = l2_normalise(np.asarray(vector, dtype=np.float32))[0]
        self._seeded[key] = normalised
        return normalised

    def load(self) -> ModelInfo:
        """Pretend to load a model, populating the same metadata the real one does."""
        self._info = ModelInfo(
            name=FAKE_MODEL,
            model_type="clip",
            embedding_dim=FAKE_DIM,
            device="cpu",
            dtype="float32",
            image_size=224,
        )
        self._model = object()  # marks is_loaded
        self._processor = object()
        return self._info

    @staticmethod
    def _vector_from(payload: bytes) -> np.ndarray:
        """Map arbitrary bytes to a stable unit vector."""
        # Repeat the digest until it covers the dimension, then centre the values
        # so vectors spread over the sphere instead of clustering in one octant.
        raw = bytearray()
        seed = payload
        while len(raw) < FAKE_DIM * 2:
            seed = hashlib.sha256(seed).digest()
            raw.extend(seed)
        values = np.frombuffer(bytes(raw[: FAKE_DIM * 2]), dtype=np.uint16).astype(np.float32)
        return l2_normalise(values / 65535.0 - 0.5)[0]

    def embed_texts(self, texts: Sequence[str], *, batch_size: int | None = None) -> np.ndarray:
        """Embed texts deterministically."""
        self.text_calls += 1
        if not texts:
            return np.zeros((0, FAKE_DIM), dtype=np.float32)
        return np.stack(
            [self._seeded.get(t, self._vector_from(f"text:{t}".encode())) for t in texts]
        )

    def embed_images(
        self, images: Sequence[Image.Image], *, batch_size: int | None = None
    ) -> np.ndarray:
        """Embed images deterministically from their pixel content."""
        self.image_calls += 1
        if not images:
            return np.zeros((0, FAKE_DIM), dtype=np.float32)
        vectors = []
        for image in images:
            small = image.convert("RGB").resize((16, 16), Image.Resampling.NEAREST)
            vectors.append(self._vector_from(b"image:" + small.tobytes()))
        return np.stack(vectors)

    def warmup(self) -> float:
        """No-op warmup."""
        return 0.0


# ------------------------------------------------------------------- settings
@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Isolated settings, applied to the environment so every code path sees them."""
    env = {
        "APP_ENV": "test",
        "LOG_LEVEL": "WARNING",
        "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        "QDRANT_URL": "",
        "QDRANT_PATH": str(tmp_path / "qdrant"),
        "QDRANT_COLLECTION": "test_products",
        "MODEL_NAME": FAKE_MODEL,
        "IMAGE_ROOT": str(tmp_path / "images"),
        "INGEST_BATCH_SIZE": "8",
        "MIN_CANDIDATE_POOL": "20",
        "CANDIDATE_MULTIPLIER": "3",
        # Disabled so the encoder's call counters stay meaningful.
        "TEXT_EMBEDDING_CACHE_SIZE": "0",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    # _env_file=None so a developer's local .env cannot leak into the test run.
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def _isolate_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``get_settings()`` return the test settings everywhere."""
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    for module in (
        "app.db.session",
        "app.repositories.vector_repository",
        "app.services.embedding",
        "app.services.image_processing",
        "app.services.indexing_service",
        "app.services.search_service",
        "app.api.v1.routes.health",
        "app.main",
    ):
        monkeypatch.setattr(f"{module}.get_settings", lambda: settings, raising=False)
    get_settings.cache_clear()


@pytest.fixture
def image_root(settings: Settings) -> Path:
    """Directory that catalogue image paths resolve against."""
    root = Path(settings.image_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


# ------------------------------------------------------------------- database
@pytest.fixture
async def engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    """A fresh database with the schema applied."""
    engine = create_engine(settings)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()
    await dispose_engine()


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory bound to the test engine."""
    return async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


@pytest.fixture
async def session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """A session for repository-level tests."""
    async with session_factory() as session:
        yield session


@pytest.fixture
def products(session: AsyncSession) -> ProductRepository:
    """Product repository bound to the test session."""
    return ProductRepository(session)


# --------------------------------------------------------------- vector store
@pytest.fixture
async def vectors(settings: Settings) -> AsyncIterator[VectorRepository]:
    """Embedded Qdrant with the collection created."""
    repository = VectorRepository(settings)
    await repository.ensure_collection(FAKE_DIM)
    yield repository
    await repository.close()


@pytest.fixture
def embedder(settings: Settings) -> FakeEmbeddingService:
    """A loaded fake encoder."""
    service = FakeEmbeddingService(settings)
    service.load()
    return service


@pytest.fixture
def image_processor(settings: Settings) -> ImageProcessor:
    """Image processor using the test settings."""
    return ImageProcessor(settings)


# ------------------------------------------------------------------- services
@pytest.fixture
def search_service(
    products: ProductRepository,
    vectors: VectorRepository,
    embedder: FakeEmbeddingService,
    image_processor: ImageProcessor,
    settings: Settings,
) -> SearchService:
    """Search service wired to the test collaborators."""
    return SearchService(
        products=products,
        vectors=vectors,
        embedder=embedder,
        images=image_processor,
        settings=settings,
    )


@pytest.fixture
def indexing_service(
    products: ProductRepository,
    vectors: VectorRepository,
    embedder: FakeEmbeddingService,
    image_processor: ImageProcessor,
    settings: Settings,
) -> IndexingService:
    """Indexing service wired to the test collaborators."""
    return IndexingService(
        products=products,
        vectors=vectors,
        embedder=embedder,
        images=image_processor,
        settings=settings,
    )


@pytest.fixture
def catalog_service(
    products: ProductRepository,
    vectors: VectorRepository,
    embedder: FakeEmbeddingService,
) -> CatalogService:
    """Catalogue service wired to the test collaborators."""
    return CatalogService(products=products, vectors=vectors, embedder=embedder)


# ----------------------------------------------------------------- HTTP client
@pytest.fixture
async def client(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    vectors: VectorRepository,
    embedder: FakeEmbeddingService,
    image_processor: ImageProcessor,
) -> AsyncIterator[AsyncClient]:
    """HTTP client against the real ASGI app with test collaborators injected.

    The app's lifespan is deliberately not run: it would load the real model. The
    schema and the Qdrant collection are already created by their fixtures.

    Only four dependencies need overriding. Everything else - the repositories and
    the three services - is constructed by FastAPI from these, so the real wiring
    is still what gets exercised.
    """
    app = create_app()

    async def _session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    from app.api.deps import get_ready_embedding_service

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db_session] = _session
    app.dependency_overrides[get_vector_repository] = lambda: vectors
    app.dependency_overrides[get_ready_embedding_service] = lambda: embedder
    app.dependency_overrides[get_image_processor] = lambda: image_processor

    # Some code paths use the process-wide singletons rather than request-scoped
    # dependencies - notably /health, which must be able to report on the stores
    # without a request context. Point those singletons at the test doubles too.
    #
    # For Qdrant this is not merely tidiness: embedded mode takes an exclusive lock
    # on its storage directory, so a second client on the same path would fail and
    # /health would wrongly report the vector store as unavailable.
    import app.db.session as session_module
    import app.repositories.vector_repository as vector_module
    import app.services.embedding as embedding_module

    saved = (embedding_module._service, vector_module._repository, session_module._engine)
    embedding_module._service = embedder
    vector_module._repository = vectors
    session_module._engine = engine_from(session_factory)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http

    (
        embedding_module._service,
        vector_module._repository,
        session_module._engine,
    ) = saved
    app.dependency_overrides.clear()


def engine_from(factory: async_sessionmaker[AsyncSession]) -> AsyncEngine:
    """Extract the engine a session factory is bound to."""
    bind = factory.kw["bind"]
    assert isinstance(bind, AsyncEngine)
    return bind


# ------------------------------------------------------------------- factories
def make_image(colour: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> Image.Image:
    """Create a solid-colour test image."""
    return Image.new("RGB", size, colour)


def image_bytes(
    colour: tuple[int, int, int] = (200, 30, 30),
    fmt: str = "JPEG",
    size: tuple[int, int] = (64, 64),
) -> bytes:
    """Encode a solid-colour test image."""
    buffer = BytesIO()
    make_image(colour, size).save(buffer, format=fmt)
    return buffer.getvalue()


def make_product(
    name: str = "Test Product",
    *,
    external_id: str | None = None,
    category: str | None = "Footwear",
    subcategory: str | None = "Sports Shoes",
    brand: str | None = "TestBrand",
    colour: str | None = "Black",
    gender: str | None = "Men",
    usage: str | None = "Sports",
    price: float | None = 59.99,
    in_stock: bool = True,
    image_url: str | None = None,
) -> Product:
    """Build an unsaved product row."""
    return Product(
        id=new_product_id(),
        external_id=external_id,
        name=name,
        description=f"{name} description.",
        category=category,
        subcategory=subcategory,
        brand=brand,
        colour=colour,
        gender=gender,
        usage=usage,
        price=price,
        currency="USD",
        in_stock=in_stock,
        image_url=image_url,
    )


@pytest.fixture
def sample_products() -> list[Product]:
    """A small, deliberately varied catalogue.

    Values are chosen so filter assertions have unambiguous answers: exactly two
    black Nike products, one out of stock, one with no price.
    """
    return [
        make_product(
            "Nike Black Sports Shoes",
            external_id="p1",
            brand="Nike",
            colour="Black",
            price=80.0,
        ),
        make_product(
            "Nike Black Running Shoes",
            external_id="p2",
            brand="Nike",
            colour="Black",
            price=120.0,
        ),
        make_product(
            "Puma White Sports Shoes",
            external_id="p3",
            brand="Puma",
            colour="White",
            price=60.0,
        ),
        make_product(
            "Fossil Brown Leather Watch",
            external_id="p4",
            category="Accessories",
            subcategory="Watches",
            brand="Fossil",
            colour="Brown",
            usage="Casual",
            price=210.0,
        ),
        make_product(
            "Levis Blue Denim Jeans",
            external_id="p5",
            category="Apparel",
            subcategory="Jeans",
            brand="Levis",
            colour="Blue",
            usage="Casual",
            price=95.0,
            in_stock=False,
        ),
        make_product(
            "Unpriced Grey Cap",
            external_id="p6",
            category="Accessories",
            subcategory="Caps",
            brand="Generic",
            colour="Grey",
            usage="Casual",
            price=None,
        ),
    ]


@pytest.fixture
async def indexed_catalog(
    session: AsyncSession,
    indexing_service: IndexingService,
    sample_products: list[Product],
    image_root: Path,
) -> list[Product]:
    """Persist and fully index the sample catalogue.

    Each product gets a distinct solid-colour image on disk, so image search has
    real, separable visual content to retrieve.
    """
    palette = [
        (10, 10, 10),
        (30, 30, 30),
        (240, 240, 240),
        (120, 70, 20),
        (40, 60, 180),
        (128, 128, 128),
    ]
    images_dir = image_root / "products"
    images_dir.mkdir(parents=True, exist_ok=True)

    for product, colour in zip(sample_products, palette, strict=True):
        path = images_dir / f"{product.external_id}.jpg"
        make_image(colour).save(path, format="JPEG")
        product.image_url = f"products/{path.name}"
        session.add(product)

    await session.flush()
    await indexing_service.index_catalog()
    await session.commit()
    return sample_products

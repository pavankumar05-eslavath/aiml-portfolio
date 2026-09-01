"""Data access for product metadata.

All SQL lives here; services never build queries. That boundary is what lets the
search service be tested against a fake repository, and keeps dialect-specific
SQL (Postgres full-text search versus the SQLite fallback) in one place.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import Select, case, delete, func, or_, select
from sqlalchemy import text as sql_text
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DatabaseError, DuplicateProductError, NotFoundError
from app.core.logging import get_logger
from app.core.text import build_product_document, lexical_terms
from app.models.product import Product, new_product_id
from app.schemas.product import ProductCreate, ProductUpdate
from app.schemas.search import SearchFilters, SortOption

logger = get_logger(__name__)

#: Fields whose change invalidates a stored embedding.
_EMBEDDING_RELEVANT_FIELDS = (
    "name",
    "description",
    "category",
    "subcategory",
    "brand",
    "colour",
    "gender",
    "usage",
    "image_url",
)


def compute_content_hash(product: Product, model_name: str) -> str:
    """Fingerprint the inputs an embedding depends on.

    The model name participates in the hash, so switching ``MODEL_NAME`` marks
    every product stale automatically instead of leaving a collection of vectors
    from two different encoders silently mixed together.
    """
    parts = [model_name, *(str(getattr(product, f) or "") for f in _EMBEDDING_RELEVANT_FIELDS)]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def build_search_document_for(product: Product) -> str:
    """Build the canonical embedding document for a product row."""
    return build_product_document(
        name=product.name,
        description=product.description,
        category=product.category,
        subcategory=product.subcategory,
        brand=product.brand,
        colour=product.colour,
        gender=product.gender,
        usage=product.usage,
    )


class ProductRepository:
    """Async CRUD, filtering, faceting and lexical search over products."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @property
    def dialect(self) -> str:
        """SQL dialect name of the bound session."""
        return self._session.bind.dialect.name if self._session.bind else "unknown"

    # ------------------------------------------------------------------ reads
    async def get(self, product_id: str) -> Product | None:
        """Fetch one product by id."""
        return await self._session.get(Product, product_id)

    async def get_or_404(self, product_id: str) -> Product:
        """Fetch one product, raising :class:`NotFoundError` when absent."""
        product = await self.get(product_id)
        if product is None:
            raise NotFoundError(f"No product with id {product_id!r}.")
        return product

    async def get_by_external_id(self, external_id: str) -> Product | None:
        """Fetch one product by its source-catalogue identifier."""
        result = await self._session.execute(
            select(Product).where(Product.external_id == external_id)
        )
        return result.scalar_one_or_none()

    async def get_many(self, product_ids: Sequence[str]) -> dict[str, Product]:
        """Batch-fetch products by id.

        Returns a mapping so callers can preserve their own ordering. This exists
        to hydrate search results in a single query rather than N.
        """
        if not product_ids:
            return {}
        result = await self._session.execute(
            select(Product).where(Product.id.in_(list(product_ids)))
        )
        return {product.id: product for product in result.scalars()}

    def _apply_filters(self, stmt: Select[Any], filters: SearchFilters) -> Select[Any]:
        """Attach metadata predicates to a select statement."""
        if filters.categories:
            stmt = stmt.where(Product.category.in_(filters.categories))
        if filters.subcategories:
            stmt = stmt.where(Product.subcategory.in_(filters.subcategories))
        if filters.brands:
            stmt = stmt.where(Product.brand.in_(filters.brands))
        if filters.colours:
            stmt = stmt.where(Product.colour.in_(filters.colours))
        if filters.genders:
            stmt = stmt.where(Product.gender.in_(filters.genders))
        if filters.min_price is not None:
            stmt = stmt.where(Product.price >= filters.min_price)
        if filters.max_price is not None:
            stmt = stmt.where(Product.price <= filters.max_price)
        if filters.in_stock_only:
            stmt = stmt.where(Product.in_stock.is_(True))
        return stmt

    @staticmethod
    def _apply_sort(stmt: Select[Any], sort: SortOption) -> Select[Any]:
        """Attach an ORDER BY clause.

        ``RELEVANCE`` has no meaning without a query, so plain listings fall back
        to newest-first, with id as a tiebreaker for stable pagination.
        """
        match sort:
            case SortOption.PRICE_ASC:
                return stmt.order_by(Product.price.asc().nullslast(), Product.id)
            case SortOption.PRICE_DESC:
                return stmt.order_by(Product.price.desc().nullslast(), Product.id)
            case SortOption.NAME_ASC:
                return stmt.order_by(Product.name.asc(), Product.id)
            case _:
                return stmt.order_by(Product.created_at.desc(), Product.id)

    async def list_products(
        self,
        *,
        filters: SearchFilters | None = None,
        search: str | None = None,
        sort: SortOption = SortOption.RELEVANCE,
        limit: int = 24,
        offset: int = 0,
    ) -> tuple[list[Product], int]:
        """Page through the catalogue.

        Returns:
            ``(rows, total_matching)`` where the total ignores pagination.
        """
        filters = filters or SearchFilters()
        stmt = self._apply_filters(select(Product), filters)
        if search and (needle := search.strip()):
            pattern = f"%{needle.lower()}%"
            stmt = stmt.where(
                or_(
                    func.lower(Product.name).like(pattern),
                    func.lower(Product.brand).like(pattern),
                    func.lower(Product.category).like(pattern),
                )
            )

        count_stmt = select(func.count()).select_from(stmt.subquery())
        try:
            total = int((await self._session.execute(count_stmt)).scalar_one())
            rows = (
                (
                    await self._session.execute(
                        self._apply_sort(stmt, sort).limit(limit).offset(offset)
                    )
                )
                .scalars()
                .all()
            )
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Failed to list products: {type(exc).__name__}") from exc
        return list(rows), total

    async def filter_ids(self, filters: SearchFilters, product_ids: Sequence[str]) -> set[str]:
        """Return the subset of ``product_ids`` satisfying ``filters``."""
        if not product_ids:
            return set()
        stmt = self._apply_filters(
            select(Product.id).where(Product.id.in_(list(product_ids))), filters
        )
        result = await self._session.execute(stmt)
        return set(result.scalars())

    async def lexical_search(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        limit: int = 100,
    ) -> list[tuple[str, float]]:
        """Keyword retrieval channel.

        On PostgreSQL this uses real full-text search: ``plainto_tsquery`` against
        a ``to_tsvector`` of name/brand/category/description, ranked by
        ``ts_rank_cd``. That gives stemming and term weighting rather than
        substring matching.

        On SQLite the fallback scores by how many query terms appear in the
        product's document, normalised to ``[0, 1]``. It is weaker, and the
        difference is documented in ``docs/architecture.md`` rather than hidden.

        Returns:
            ``(product_id, score)`` pairs, best first.
        """
        terms = lexical_terms(query)
        if not terms:
            return []
        filters = filters or SearchFilters()

        if self.dialect == "postgresql":
            return await self._lexical_search_postgres(query, filters, limit)
        return await self._lexical_search_fallback(terms, filters, limit)

    async def _lexical_search_postgres(
        self, query: str, filters: SearchFilters, limit: int
    ) -> list[tuple[str, float]]:
        document = (
            "setweight(to_tsvector('english', coalesce(products.name, '')), 'A') || "
            "setweight(to_tsvector('english', coalesce(products.brand, '')), 'B') || "
            "setweight(to_tsvector('english', coalesce(products.category, '')), 'B') || "
            "setweight(to_tsvector('english', coalesce(products.subcategory, '')), 'B') || "
            "setweight(to_tsvector('english', coalesce(products.description, '')), 'C')"
        )
        rank = func.ts_rank_cd(sql_text(document), func.plainto_tsquery("english", query))
        stmt = self._apply_filters(
            select(Product.id, rank.label("score")).where(
                sql_text(f"({document}) @@ plainto_tsquery('english', :qtext)")
            ),
            filters,
        )
        stmt = stmt.params(qtext=query).order_by(sql_text("score DESC")).limit(limit)
        try:
            rows = (await self._session.execute(stmt)).all()
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Lexical search failed: {type(exc).__name__}") from exc
        return [(row[0], float(row[1] or 0.0)) for row in rows]

    async def _lexical_search_fallback(
        self, terms: Sequence[str], filters: SearchFilters, limit: int
    ) -> list[tuple[str, float]]:
        # Pull a bounded candidate set matching any term, then score in Python.
        # Bounded by ``limit * 5`` so a very common term cannot load the table.
        clauses = [
            or_(
                func.lower(Product.name).like(f"%{term}%"),
                func.lower(func.coalesce(Product.brand, "")).like(f"%{term}%"),
                func.lower(func.coalesce(Product.category, "")).like(f"%{term}%"),
                func.lower(func.coalesce(Product.subcategory, "")).like(f"%{term}%"),
                func.lower(func.coalesce(Product.colour, "")).like(f"%{term}%"),
            )
            for term in terms
        ]
        stmt = self._apply_filters(select(Product).where(or_(*clauses)), filters).limit(
            max(limit * 5, 100)
        )
        try:
            rows = (await self._session.execute(stmt)).scalars().all()
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Lexical search failed: {type(exc).__name__}") from exc

        scored: list[tuple[str, float]] = []
        for product in rows:
            haystack = " ".join(
                filter(
                    None,
                    (
                        product.name,
                        product.brand,
                        product.category,
                        product.subcategory,
                        product.colour,
                        product.usage,
                    ),
                )
            ).casefold()
            # Name matches count double: a term in the title is stronger evidence
            # than the same term buried in a facet.
            hits = sum(1 for term in terms if term in haystack)
            name_hits = sum(1 for term in terms if term in (product.name or "").casefold())
            if not hits:
                continue
            scored.append((product.id, (hits + name_hits) / (2 * len(terms))))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:limit]

    async def facets(self, top_n: int = 40) -> dict[str, Any]:
        """Distinct filter values with counts, plus the observed price range."""

        async def _counts(column: Any) -> list[tuple[str, int]]:
            stmt = (
                select(column, func.count())
                .where(column.is_not(None))
                .group_by(column)
                .order_by(func.count().desc())
                .limit(top_n)
            )
            return [(row[0], int(row[1])) for row in (await self._session.execute(stmt)).all()]

        price_row = (
            await self._session.execute(
                select(func.min(Product.price), func.max(Product.price))
            )
        ).one()
        return {
            "categories": await _counts(Product.category),
            "brands": await _counts(Product.brand),
            "colours": await _counts(Product.colour),
            "price_min": float(price_row[0]) if price_row[0] is not None else None,
            "price_max": float(price_row[1]) if price_row[1] is not None else None,
        }

    async def stats(self) -> dict[str, Any]:
        """Aggregate catalogue and indexing counters.

        Boolean columns are summed via a portable ``CASE`` expression rather than a
        cast: SQLite stores booleans as integers while PostgreSQL has a real
        boolean type, and ``SUM(boolean)`` is invalid on the latter.
        """
        row = (
            await self._session.execute(
                select(
                    func.count(Product.id),
                    func.count(Product.indexed_at),
                    func.sum(case((Product.has_image_vector.is_(True), 1), else_=0)),
                    func.sum(case((Product.has_text_vector.is_(True), 1), else_=0)),
                    func.count(Product.index_error),
                    func.max(Product.indexed_at),
                )
            )
        ).one()
        total = int(row[0] or 0)
        indexed = int(row[1] or 0)
        return {
            "total_products": total,
            "indexed_products": indexed,
            "pending_products": max(0, total - indexed),
            "with_image_vector": int(row[2] or 0),
            "with_text_vector": int(row[3] or 0),
            "failed_products": int(row[4] or 0),
            "last_indexed_at": row[5],
        }

    async def iter_for_indexing(
        self,
        *,
        product_ids: Sequence[str] | None = None,
        force: bool = False,
        limit: int | None = None,
    ) -> list[Product]:
        """Select products that need embedding.

        When ``force`` is false, products whose ``content_hash`` already matches
        and which already carry a vector are excluded at the SQL level, so an
        incremental run over an indexed catalogue does almost no work.
        """
        stmt = select(Product)
        if product_ids:
            stmt = stmt.where(Product.id.in_(list(product_ids)))
        elif not force:
            stmt = stmt.where(
                or_(
                    Product.indexed_at.is_(None),
                    Product.content_hash.is_(None),
                    Product.has_text_vector.is_(False),
                )
            )
        stmt = stmt.order_by(Product.created_at.asc(), Product.id)
        if limit:
            stmt = stmt.limit(limit)
        return list((await self._session.execute(stmt)).scalars().all())

    async def count_all(self) -> int:
        """Total number of products."""
        return int((await self._session.execute(select(func.count(Product.id)))).scalar_one())

    # ----------------------------------------------------------------- writes
    async def create(self, payload: ProductCreate) -> Product:
        """Insert a product.

        Raises:
            DuplicateProductError: when ``external_id`` already exists.
        """
        if payload.external_id:
            existing = await self.get_by_external_id(payload.external_id)
            if existing is not None:
                raise DuplicateProductError(
                    f"A product with external_id {payload.external_id!r} already exists.",
                    details={"product_id": existing.id},
                )

        product = Product(id=new_product_id(), **payload.model_dump())
        product.search_document = build_search_document_for(product)
        self._session.add(product)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            raise DuplicateProductError(
                "A product with the same identity already exists."
            ) from exc
        return product

    async def bulk_upsert(self, payloads: Iterable[ProductCreate]) -> tuple[list[Product], int]:
        """Insert new products and update existing ones, keyed on ``external_id``.

        This is what makes re-running ingestion safe: a second pass over the same
        source data updates rows in place instead of duplicating the catalogue.

        Returns:
            ``(touched_products, updated_count)``.
        """
        payload_list = list(payloads)
        external_ids = [p.external_id for p in payload_list if p.external_id]
        existing: dict[str, Product] = {}
        if external_ids:
            result = await self._session.execute(
                select(Product).where(Product.external_id.in_(external_ids))
            )
            existing = {p.external_id: p for p in result.scalars() if p.external_id}

        touched: list[Product] = []
        updated = 0
        for payload in payload_list:
            current = existing.get(payload.external_id) if payload.external_id else None
            if current is None:
                product = Product(id=new_product_id(), **payload.model_dump())
                product.search_document = build_search_document_for(product)
                self._session.add(product)
                touched.append(product)
            else:
                for field, value in payload.model_dump(exclude={"external_id"}).items():
                    setattr(current, field, value)
                current.search_document = build_search_document_for(current)
                touched.append(current)
                updated += 1

        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            raise DuplicateProductError(
                "Duplicate external_id within the ingestion batch."
            ) from exc
        return touched, updated

    async def update(self, product_id: str, payload: ProductUpdate) -> Product:
        """Apply a partial update and refresh the canonical document."""
        product = await self.get_or_404(product_id)
        changes = payload.model_dump(exclude_unset=True)
        for field, value in changes.items():
            setattr(product, field, value)
        if any(field in _EMBEDDING_RELEVANT_FIELDS for field in changes):
            product.search_document = build_search_document_for(product)
            # Invalidate the fingerprint so the next indexing run re-embeds it.
            product.content_hash = None
        await self._session.flush()
        return product

    async def delete(self, product_id: str) -> None:
        """Delete a product, or raise :class:`NotFoundError`."""
        product = await self.get_or_404(product_id)
        await self._session.delete(product)
        await self._session.flush()

    async def delete_all(self) -> int:
        """Delete every product. Returns the number of rows removed."""
        result = await self._session.execute(delete(Product))
        await self._session.flush()
        # execute() is typed as Result, but a DML statement always yields a
        # CursorResult, which is where rowcount lives.
        return int(cast("CursorResult[Any]", result).rowcount or 0)

    async def mark_indexed(
        self,
        product: Product,
        *,
        content_hash: str,
        document: str,
        has_image_vector: bool,
        has_text_vector: bool,
        image_path: str | None = None,
    ) -> None:
        """Record a successful indexing attempt."""
        product.content_hash = content_hash
        product.search_document = document
        product.has_image_vector = has_image_vector
        product.has_text_vector = has_text_vector
        product.image_path = image_path
        product.indexed_at = datetime.now(UTC)
        product.index_error = None

    async def mark_index_failed(self, product: Product, reason: str) -> None:
        """Record why a product could not be indexed."""
        product.index_error = reason[:1000]
        product.indexed_at = None

    async def clear_index_state(self) -> None:
        """Reset indexing state for every product (used when dropping vectors)."""
        for product in (await self._session.execute(select(Product))).scalars():
            product.content_hash = None
            product.indexed_at = None
            product.has_image_vector = False
            product.has_text_vector = False
            product.index_error = None
        await self._session.flush()

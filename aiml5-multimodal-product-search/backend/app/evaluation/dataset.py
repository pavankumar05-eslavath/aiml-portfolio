"""Evaluation query set and relevance judgements.

Methodology
-----------
Hand-labelling enough query/product pairs to evaluate a 6,000-product catalogue is
not feasible for a project of this size, so relevance is defined **declaratively**
as a predicate over the product attributes the source dataset actually provides::

    query:     "black sports shoes for men"
    relevant:  subcategory == "Sports Shoes" AND colour == "Black"
                              AND gender IN ("Men", "Boys", "Unisex")

The predicate is resolved against the live catalogue at evaluation time, which
makes the ground truth:

* **auditable** - the rule is written down, not a list of ids someone chose;
* **reproducible** - it re-derives correctly for any catalogue size;
* **honest about its limits** - see below.

Known limitations of attribute-derived relevance
------------------------------------------------
1. It measures *attribute agreement*, not human preference. A visually perfect
   match labelled with a different ``baseColour`` counts as a miss.
2. It is generous within a class: every black sports shoe is equally relevant,
   so the metrics cannot distinguish good ranking *within* the relevant set.
3. Queries are written by the same author as the templates, which risks lexical
   overlap flattering the text channel. Queries are therefore phrased in natural
   shopper language ("something for the gym"), not in dataset vocabulary.

These are recorded in ``docs/evaluation.md``. The numbers are meaningful for
*comparing configurations of this system* - which is exactly what the experiments
use them for - and should not be read as absolute retrieval quality.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.core.logging import get_logger
from app.models.product import Product

logger = get_logger(__name__)


class QueryType(StrEnum):
    """Which modality a benchmark query exercises."""

    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


@dataclass(slots=True)
class RelevanceRule:
    """Attribute predicate defining a query's relevant set.

    Fields are ANDed; the values within a field are ORed. All matching is
    case-insensitive. ``exclude_external_ids`` removes specific products, which
    matters for image queries where the query product must not count as its own
    answer.
    """

    categories: list[str] = field(default_factory=list)
    subcategories: list[str] = field(default_factory=list)
    colours: list[str] = field(default_factory=list)
    genders: list[str] = field(default_factory=list)
    usages: list[str] = field(default_factory=list)
    brands: list[str] = field(default_factory=list)
    exclude_external_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RelevanceRule:
        """Build a rule from its JSON form."""
        known = {
            "categories",
            "subcategories",
            "colours",
            "genders",
            "usages",
            "brands",
            "exclude_external_ids",
        }
        if unknown := set(raw) - known:
            raise ValueError(f"unknown relevance field(s): {', '.join(sorted(unknown))}")
        return cls(**{key: list(raw.get(key, [])) for key in known})

    def is_empty(self) -> bool:
        """Whether this rule constrains nothing (which would match everything)."""
        return not any(
            (
                self.categories,
                self.subcategories,
                self.colours,
                self.genders,
                self.usages,
                self.brands,
            )
        )

    def describe(self) -> str:
        """Human-readable form of the predicate, for the report."""
        parts = []
        for label, values in (
            ("category", self.categories),
            ("type", self.subcategories),
            ("colour", self.colours),
            ("gender", self.genders),
            ("usage", self.usages),
            ("brand", self.brands),
        ):
            if values:
                parts.append(f"{label} in {{{', '.join(values)}}}")
        return " AND ".join(parts) or "(unconstrained)"

    def matches(self, product: Product) -> bool:
        """Whether a product satisfies this predicate."""
        if product.external_id and product.external_id in set(self.exclude_external_ids):
            return False

        def _ok(values: Sequence[str], actual: str | None) -> bool:
            if not values:
                return True
            if actual is None:
                return False
            folded = actual.casefold()
            return any(value.casefold() == folded for value in values)

        return (
            _ok(self.categories, product.category)
            and _ok(self.subcategories, product.subcategory)
            and _ok(self.colours, product.colour)
            and _ok(self.genders, product.gender)
            and _ok(self.usages, product.usage)
            and _ok(self.brands, product.brand)
        )


@dataclass(slots=True)
class EvalQuery:
    """One benchmark query.

    ``group`` sub-classifies a query beyond its modality. It exists because
    multimodal queries come in two kinds that want *opposite* weightings:
    ``contradiction`` (text overrides an attribute of the image) and ``agreement``
    (text reinforces the image). Reporting a single multimodal average over both
    would hide that, and would make any recommended weight an artefact of the
    query mix.
    """

    id: str
    type: QueryType
    relevance: RelevanceRule
    text: str | None = None
    image_external_id: str | None = None
    note: str | None = None
    group: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalQuery:
        """Build a query from its JSON form.

        Raises:
            ValueError: if the record is malformed or inconsistent with its type.
        """
        try:
            query_type = QueryType(raw["type"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"query {raw.get('id')!r}: invalid or missing 'type'") from exc

        query = cls(
            id=str(raw["id"]),
            type=query_type,
            relevance=RelevanceRule.from_dict(raw.get("relevance", {})),
            text=raw.get("query"),
            image_external_id=(
                str(raw["image_external_id"]) if raw.get("image_external_id") else None
            ),
            note=raw.get("note"),
            group=raw.get("group"),
        )

        if query_type in (QueryType.TEXT, QueryType.MULTIMODAL) and not query.text:
            raise ValueError(f"query {query.id!r}: '{query_type}' requires 'query' text")
        if (
            query_type in (QueryType.IMAGE, QueryType.MULTIMODAL)
            and not query.image_external_id
        ):
            raise ValueError(f"query {query.id!r}: '{query_type}' requires 'image_external_id'")
        if query.relevance.is_empty():
            raise ValueError(f"query {query.id!r}: relevance rule matches everything")
        return query


@dataclass(slots=True)
class ResolvedQuery:
    """A benchmark query bound to concrete catalogue ids."""

    query: EvalQuery
    relevant_ids: set[str]
    image_product_id: str | None = None
    #: Set when the query cannot be run against this catalogue.
    skip_reason: str | None = None

    @property
    def is_runnable(self) -> bool:
        """Whether this query can be executed and scored."""
        return self.skip_reason is None


def load_queries(path: Path) -> list[EvalQuery]:
    """Load and validate the benchmark query set.

    Raises:
        FileNotFoundError: if the file is missing.
        ValueError: on malformed JSON, duplicate ids, or invalid records.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation query set not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    raw_queries = payload.get("queries") if isinstance(payload, dict) else payload
    if not isinstance(raw_queries, list) or not raw_queries:
        raise ValueError(f"{path} contains no queries")

    queries = [EvalQuery.from_dict(record) for record in raw_queries]
    seen: set[str] = set()
    for query in queries:
        if query.id in seen:
            raise ValueError(f"duplicate query id: {query.id!r}")
        seen.add(query.id)
    return queries


async def resolve_queries(
    queries: Iterable[EvalQuery], session: Any, *, min_relevant: int = 2
) -> list[ResolvedQuery]:
    """Bind each query's predicate to the product ids in the catalogue.

    Loads the catalogue once and evaluates every predicate in memory: with a few
    dozen queries this is far cheaper than one SQL query per rule, and it keeps
    the matching logic identical to :meth:`RelevanceRule.matches`, which the unit
    tests exercise directly.

    Args:
        queries: Benchmark queries.
        session: An ``AsyncSession``.
        min_relevant: Queries with fewer relevant products than this are marked
            skipped, because metrics over one or two items are noise.

    Returns:
        One :class:`ResolvedQuery` per input query, in order.
    """
    products = list((await session.execute(select(Product))).scalars())
    by_external_id = {p.external_id: p for p in products if p.external_id}
    indexed_ids = {p.id for p in products if p.indexed_at is not None}

    resolved: list[ResolvedQuery] = []
    for query in queries:
        image_product_id: str | None = None
        skip: str | None = None

        if query.image_external_id:
            product = by_external_id.get(query.image_external_id)
            if product is None:
                skip = f"query image product {query.image_external_id!r} not in catalogue"
            elif product.id not in indexed_ids:
                skip = f"query image product {query.image_external_id!r} is not indexed"
            else:
                image_product_id = product.id

        relevant = {
            product.id
            for product in products
            if query.relevance.matches(product) and product.id in indexed_ids
        }
        # The query product is never its own answer.
        relevant.discard(image_product_id or "")

        if skip is None and len(relevant) < min_relevant:
            skip = f"only {len(relevant)} relevant product(s) in this catalogue"

        resolved.append(
            ResolvedQuery(
                query=query,
                relevant_ids=relevant,
                image_product_id=image_product_id,
                skip_reason=skip,
            )
        )

    runnable = sum(1 for r in resolved if r.is_runnable)
    logger.info(
        "resolved evaluation queries",
        extra={
            "context": {
                "total": len(resolved),
                "runnable": runnable,
                "skipped": len(resolved) - runnable,
                "catalogue": len(products),
            }
        },
    )
    return resolved

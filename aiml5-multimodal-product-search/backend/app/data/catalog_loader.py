"""Loading canonical catalogue CSV rows into the database.

Validation happens here, not in the CLI script, so that malformed rows are
reported the same way whether ingestion is driven from the command line or from a
test. Rows that fail validation are collected and reported rather than aborting
the run: one bad row in a 40,000-row file should not discard the other 39,999.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from pydantic import ValidationError as PydanticValidationError

from app.core.logging import get_logger
from app.data.fashion_dataset import read_catalog_csv
from app.repositories.product_repository import ProductRepository
from app.schemas.product import ProductCreate

logger = get_logger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "y", "t"}


@dataclass(slots=True)
class RowError:
    """A source row that could not be turned into a product."""

    line: int
    external_id: str | None
    reason: str


@dataclass(slots=True)
class LoadReport:
    """Outcome of loading a catalogue file."""

    total_rows: int = 0
    created: int = 0
    updated: int = 0
    invalid: int = 0
    duplicates_in_file: int = 0
    errors: list[RowError] = field(default_factory=list)

    @property
    def accepted(self) -> int:
        """Rows that became products."""
        return self.created + self.updated


def _optional(value: str | None) -> str | None:
    """Normalise a CSV cell to ``None`` when blank."""
    if value is None:
        return None
    text = value.strip()
    return text or None


def _to_float(value: str | None) -> float | None:
    """Parse a price cell, returning ``None`` when blank or unparsable."""
    text = _optional(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    """Parse an integer cell, tolerating values written as floats."""
    text = _optional(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def row_to_product(row: dict[str, str]) -> ProductCreate:
    """Convert a canonical CSV row into a validated create payload.

    Raises:
        pydantic.ValidationError: if the row violates the product schema.
    """
    in_stock = _optional(row.get("in_stock"))
    return ProductCreate(
        external_id=_optional(row.get("external_id")),
        name=(row.get("name") or "").strip(),
        description=_optional(row.get("description")),
        category=_optional(row.get("category")),
        subcategory=_optional(row.get("subcategory")),
        brand=_optional(row.get("brand")),
        colour=_optional(row.get("colour")),
        gender=_optional(row.get("gender")),
        usage=_optional(row.get("usage")),
        season=_optional(row.get("season")),
        year=_to_int(row.get("year")),
        price=_to_float(row.get("price")),
        currency=(_optional(row.get("currency")) or "USD"),
        image_url=_optional(row.get("image_url")),
        in_stock=True if in_stock is None else in_stock.casefold() in _TRUE_VALUES,
    )


def validate_rows(
    rows: Iterable[dict[str, str]],
) -> tuple[list[ProductCreate], list[RowError], int]:
    """Validate source rows, separating the usable from the broken.

    Duplicate ``external_id`` values *within the file* are dropped, keeping the
    last occurrence: a bulk upsert cannot insert the same key twice in one
    statement, and failing the whole batch over a duplicated source row would be
    unhelpful.

    Returns:
        ``(payloads, errors, duplicates_dropped)``.
    """
    payloads: list[ProductCreate] = []
    errors: list[RowError] = []
    by_external_id: dict[str, int] = {}
    duplicates = 0

    for line, row in enumerate(rows, start=2):  # line 1 is the CSV header
        external_id = _optional(row.get("external_id"))
        try:
            payload = row_to_product(row)
        except PydanticValidationError as exc:
            first = exc.errors()[0]
            field_name = ".".join(str(part) for part in first["loc"]) or "row"
            errors.append(
                RowError(
                    line=line,
                    external_id=external_id,
                    reason=f"{field_name}: {first['msg']}",
                )
            )
            continue

        if payload.external_id and payload.external_id in by_external_id:
            duplicates += 1
            payloads[by_external_id[payload.external_id]] = payload
            continue

        if payload.external_id:
            by_external_id[payload.external_id] = len(payloads)
        payloads.append(payload)

    return payloads, errors, duplicates


class CatalogLoader:
    """Loads canonical catalogue files into the products table."""

    def __init__(self, products: ProductRepository) -> None:
        self._products = products

    async def load_rows(
        self, rows: Sequence[dict[str, str]], *, batch_size: int = 500
    ) -> LoadReport:
        """Upsert rows into the database.

        Existing products are matched on ``external_id`` and updated in place, so
        re-running ingestion is idempotent rather than duplicating the catalogue.
        """
        report = LoadReport(total_rows=len(rows))
        payloads, errors, duplicates = validate_rows(rows)
        report.errors = errors[:100]
        report.invalid = len(errors)
        report.duplicates_in_file = duplicates

        for start in range(0, len(payloads), batch_size):
            batch = payloads[start : start + batch_size]
            touched, updated = await self._products.bulk_upsert(batch)
            report.updated += updated
            report.created += len(touched) - updated

        if report.invalid:
            logger.warning(
                "some rows were rejected",
                extra={
                    "context": {
                        "invalid": report.invalid,
                        "examples": [e.reason for e in report.errors[:3]],
                    }
                },
            )
        logger.info(
            "catalogue rows loaded",
            extra={
                "context": {
                    "created": report.created,
                    "updated": report.updated,
                    "invalid": report.invalid,
                    "duplicates_in_file": report.duplicates_in_file,
                }
            },
        )
        return report

    async def load_csv(self, path: Path, *, batch_size: int = 500) -> LoadReport:
        """Load a canonical catalogue CSV.

        Reading the file is offloaded to a worker thread. A 44,000-row catalogue is
        several megabytes of synchronous I/O and parsing, which would otherwise
        block the event loop for the whole read.

        Raises:
            FileNotFoundError: if the file does not exist.
            ValueError: if required columns are missing.
        """

        def _read() -> list[dict[str, Any]]:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Catalogue file not found: {path}. "
                    "Run 'python scripts/prepare_sample.py' to build the sample catalogue."
                )
            return list(read_catalog_csv(path))

        rows = await anyio.to_thread.run_sync(_read)
        return await self.load_rows(rows, batch_size=batch_size)

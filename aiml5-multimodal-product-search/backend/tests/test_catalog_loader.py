"""Catalogue CSV loading tests.

Ingestion must be robust to messy source files: one malformed row should not
discard the other 39,999.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.catalog_loader import CatalogLoader, row_to_product, validate_rows
from app.data.fashion_dataset import CATALOG_COLUMNS
from app.repositories.product_repository import ProductRepository

GOOD_ROW = {
    "external_id": "1001",
    "name": "Nike Revolution Running Shoe",
    "description": "A light running shoe.",
    "category": "Footwear",
    "subcategory": "Sports Shoes",
    "brand": "Nike",
    "colour": "Black",
    "gender": "Men",
    "usage": "Sports",
    "season": "Fall",
    "year": "2012",
    "price": "74.99",
    "currency": "USD",
    "image_url": "sample/images/1001.jpg",
    "in_stock": "true",
}


def write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write rows using the canonical column order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CATALOG_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CATALOG_COLUMNS})
    return path


class TestRowConversion:
    def test_converts_a_well_formed_row(self):
        payload = row_to_product(GOOD_ROW)
        assert payload.name == "Nike Revolution Running Shoe"
        assert payload.price == 74.99
        assert payload.year == 2012
        assert payload.in_stock is True

    def test_blank_optional_fields_become_none(self):
        payload = row_to_product({**GOOD_ROW, "brand": "", "colour": "   "})
        assert payload.brand is None
        assert payload.colour is None

    def test_unparsable_price_becomes_none_rather_than_failing(self):
        """A bad price should not discard an otherwise usable product."""
        assert row_to_product({**GOOD_ROW, "price": "n/a"}).price is None

    def test_year_written_as_a_float_is_accepted(self):
        assert row_to_product({**GOOD_ROW, "year": "2012.0"}).year == 2012

    def test_missing_currency_defaults_to_usd(self):
        assert row_to_product({**GOOD_ROW, "currency": ""}).currency == "USD"

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("0", False),
            ("no", False),
        ],
    )
    def test_in_stock_parsing(self, value: str, expected: bool):
        assert row_to_product({**GOOD_ROW, "in_stock": value}).in_stock is expected

    def test_missing_in_stock_defaults_to_true(self):
        assert row_to_product({**GOOD_ROW, "in_stock": ""}).in_stock is True


class TestValidateRows:
    def test_accepts_good_rows(self):
        payloads, errors, duplicates = validate_rows([GOOD_ROW])
        assert len(payloads) == 1
        assert errors == []
        assert duplicates == 0

    def test_reports_a_blank_name_with_its_line_number(self):
        payloads, errors, _ = validate_rows([{**GOOD_ROW, "name": ""}])
        assert payloads == []
        assert len(errors) == 1
        assert errors[0].line == 2  # line 1 is the header
        assert "name" in errors[0].reason

    def test_one_bad_row_does_not_discard_the_others(self):
        rows = [
            GOOD_ROW,
            {**GOOD_ROW, "external_id": "2", "name": ""},
            {**GOOD_ROW, "external_id": "3"},
        ]
        payloads, errors, _ = validate_rows(rows)
        assert len(payloads) == 2
        assert len(errors) == 1

    def test_rejects_a_negative_price(self):
        _, errors, _ = validate_rows([{**GOOD_ROW, "price": "-10"}])
        assert len(errors) == 1

    def test_duplicate_external_ids_in_the_file_are_collapsed(self):
        """A bulk upsert cannot insert the same key twice in one statement."""
        rows = [GOOD_ROW, {**GOOD_ROW, "name": "Renamed Later"}]
        payloads, _, duplicates = validate_rows(rows)
        assert duplicates == 1
        assert len(payloads) == 1
        # The last occurrence wins.
        assert payloads[0].name == "Renamed Later"

    def test_rows_without_external_ids_are_all_kept(self):
        rows = [{**GOOD_ROW, "external_id": ""}, {**GOOD_ROW, "external_id": ""}]
        payloads, _, duplicates = validate_rows(rows)
        assert len(payloads) == 2
        assert duplicates == 0


class TestCatalogLoader:
    async def test_loads_a_csv(
        self, session: AsyncSession, products: ProductRepository, tmp_path: Path
    ):
        path = write_csv(
            tmp_path / "catalog.csv",
            [GOOD_ROW, {**GOOD_ROW, "external_id": "1002", "name": "Second Product"}],
        )
        report = await CatalogLoader(products).load_csv(path)
        await session.commit()

        assert report.total_rows == 2
        assert report.created == 2
        assert report.updated == 0
        assert await products.count_all() == 2

    async def test_reloading_updates_instead_of_duplicating(
        self, session: AsyncSession, products: ProductRepository, tmp_path: Path
    ):
        """Re-running ingestion must be idempotent."""
        loader = CatalogLoader(products)
        path = write_csv(tmp_path / "c1.csv", [GOOD_ROW])
        await loader.load_csv(path)
        await session.commit()

        updated_path = write_csv(tmp_path / "c2.csv", [{**GOOD_ROW, "price": "99.99"}])
        report = await loader.load_csv(updated_path)
        await session.commit()

        assert report.created == 0
        assert report.updated == 1
        assert await products.count_all() == 1
        product = await products.get_by_external_id("1001")
        assert product.price == 99.99

    async def test_reports_invalid_rows_without_aborting(
        self, session: AsyncSession, products: ProductRepository, tmp_path: Path
    ):
        path = write_csv(
            tmp_path / "mixed.csv",
            [
                GOOD_ROW,
                {**GOOD_ROW, "external_id": "bad", "name": ""},
                {**GOOD_ROW, "external_id": "1003"},
            ],
        )
        report = await CatalogLoader(products).load_csv(path)
        await session.commit()
        assert report.created == 2
        assert report.invalid == 1
        assert report.accepted == 2

    async def test_missing_file_raises_with_guidance(
        self, products: ProductRepository, tmp_path: Path
    ):
        with pytest.raises(FileNotFoundError, match="prepare_sample"):
            await CatalogLoader(products).load_csv(tmp_path / "absent.csv")

    async def test_missing_required_column_fails_fast(
        self, products: ProductRepository, tmp_path: Path
    ):
        """Better to fail loudly than to create thousands of nameless products."""
        path = tmp_path / "no_name.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["external_id", "price"])
            writer.writeheader()
            writer.writerow({"external_id": "1", "price": "5"})

        with pytest.raises(ValueError, match="missing required column"):
            await CatalogLoader(products).load_csv(path)

    async def test_empty_csv_loads_nothing(
        self, session: AsyncSession, products: ProductRepository, tmp_path: Path
    ):
        path = write_csv(tmp_path / "empty.csv", [])
        report = await CatalogLoader(products).load_csv(path)
        await session.commit()
        assert report.total_rows == 0
        assert report.created == 0

    async def test_loaded_products_get_a_search_document(
        self, session: AsyncSession, products: ProductRepository, tmp_path: Path
    ):
        path = write_csv(tmp_path / "doc.csv", [GOOD_ROW])
        await CatalogLoader(products).load_csv(path)
        await session.commit()
        product = await products.get_by_external_id("1001")
        assert product.search_document
        assert "Nike" in product.search_document


class TestShippedSampleCatalogue:
    """The committed sample CSV must remain loadable."""

    @pytest.fixture
    def sample_csv(self) -> Path:
        path = Path(__file__).resolve().parents[2] / "data" / "sample" / "products.csv"
        if not path.is_file():
            pytest.skip("sample catalogue is not present in this checkout")
        return path

    async def test_every_row_validates(self, sample_csv: Path):
        with sample_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        payloads, errors, duplicates = validate_rows(rows)
        assert errors == [], f"sample catalogue has invalid rows: {errors[:3]}"
        assert duplicates == 0
        assert len(payloads) == len(rows)

    async def test_rows_carry_the_expected_fields(self, sample_csv: Path):
        with sample_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert set(CATALOG_COLUMNS) <= set(reader.fieldnames or [])
            first = next(reader)
        assert first["name"]
        assert first["image_url"].startswith("sample/images/")

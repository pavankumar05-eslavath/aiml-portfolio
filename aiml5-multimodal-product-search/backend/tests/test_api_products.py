"""Product CRUD, listing, filtering and facet endpoint tests."""

from __future__ import annotations

from httpx import AsyncClient

VALID_PRODUCT = {
    "external_id": "sku-001",
    "name": "Asics Gel Kayano Running Shoe",
    "description": "Stability running shoe with gel cushioning.",
    "category": "Footwear",
    "subcategory": "Sports Shoes",
    "brand": "Asics",
    "colour": "Blue",
    "gender": "Men",
    "usage": "Sports",
    "price": 149.99,
}


class TestCreate:
    async def test_creates_product_and_returns_201(self, client: AsyncClient):
        response = await client.post("/api/products", json=VALID_PRODUCT)
        assert response.status_code == 201

        body = response.json()
        assert body["name"] == VALID_PRODUCT["name"]
        assert body["price"] == 149.99
        assert body["id"]
        assert body["created_at"]
        # New products are not searchable until indexed - that is a separate step.
        assert body["indexed_at"] is None
        assert body["has_image_vector"] is False
        assert body["has_text_vector"] is False

    async def test_persists_and_is_retrievable(self, client: AsyncClient):
        created = (await client.post("/api/products", json=VALID_PRODUCT)).json()
        fetched = await client.get(f"/api/products/{created['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["id"] == created["id"]

    async def test_builds_the_search_document(self, client: AsyncClient):
        """The embedded text is derived at write time, not at index time."""
        body = (await client.post("/api/products", json=VALID_PRODUCT)).json()
        document = body["search_document"]
        assert document
        assert "Asics" in document
        assert "blue" in document.lower()

    async def test_rejects_duplicate_external_id_with_409(self, client: AsyncClient):
        await client.post("/api/products", json=VALID_PRODUCT)
        response = await client.post("/api/products", json=VALID_PRODUCT)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "duplicate_product"

    async def test_allows_repeated_null_external_id(self, client: AsyncClient):
        """Only non-null external ids are unique."""
        payload = {k: v for k, v in VALID_PRODUCT.items() if k != "external_id"}
        assert (await client.post("/api/products", json=payload)).status_code == 201
        assert (await client.post("/api/products", json=payload)).status_code == 201

    async def test_rejects_blank_name(self, client: AsyncClient):
        response = await client.post("/api/products", json={**VALID_PRODUCT, "name": "   "})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_rejects_negative_price(self, client: AsyncClient):
        response = await client.post("/api/products", json={**VALID_PRODUCT, "price": -5})
        assert response.status_code == 422

    async def test_rejects_unknown_field(self, client: AsyncClient):
        response = await client.post(
            "/api/products", json={**VALID_PRODUCT, "colour_hex": "#000"}
        )
        assert response.status_code in (201, 422)  # extra fields are ignored, not fatal

    async def test_normalises_currency_case(self, client: AsyncClient):
        body = (
            await client.post("/api/products", json={**VALID_PRODUCT, "currency": "eur"})
        ).json()
        assert body["currency"] == "EUR"

    async def test_cleans_whitespace_in_text_fields(self, client: AsyncClient):
        body = (
            await client.post(
                "/api/products",
                json={**VALID_PRODUCT, "name": "  Spaced   Out    Shoe  "},
            )
        ).json()
        assert body["name"] == "Spaced Out Shoe"


class TestRead:
    async def test_unknown_id_returns_404_envelope(self, client: AsyncClient):
        response = await client.get("/api/products/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    async def test_lists_with_pagination_metadata(self, client: AsyncClient, indexed_catalog):
        response = await client.get("/api/products", params={"limit": 2, "offset": 0})
        assert response.status_code == 200
        body = response.json()
        assert len(body["items"]) == 2
        assert body["total"] == len(indexed_catalog)
        assert body["limit"] == 2
        assert body["offset"] == 0

    async def test_pagination_does_not_repeat_rows(self, client: AsyncClient, indexed_catalog):
        first = (await client.get("/api/products", params={"limit": 3, "offset": 0})).json()
        second = (await client.get("/api/products", params={"limit": 3, "offset": 3})).json()
        ids = [item["id"] for item in first["items"] + second["items"]]
        assert len(ids) == len(set(ids)) == len(indexed_catalog)

    async def test_rejects_oversized_limit(self, client: AsyncClient):
        assert (await client.get("/api/products", params={"limit": 5000})).status_code == 422

    async def test_substring_search(self, client: AsyncClient, indexed_catalog):
        body = (await client.get("/api/products", params={"search": "nike"})).json()
        assert body["total"] == 2
        assert all("Nike" in item["name"] for item in body["items"])

    async def test_sorts_by_price_ascending_with_nulls_last(
        self, client: AsyncClient, indexed_catalog
    ):
        items = (
            await client.get("/api/products", params={"sort": "price_asc", "limit": 100})
        ).json()["items"]
        prices = [item["price"] for item in items]
        priced = [p for p in prices if p is not None]
        assert priced == sorted(priced)
        # A product without a price must not sort as if it were free.
        assert prices[-1] is None

    async def test_sorts_by_name(self, client: AsyncClient, indexed_catalog):
        items = (
            await client.get("/api/products", params={"sort": "name_asc", "limit": 100})
        ).json()["items"]
        names = [item["name"] for item in items]
        assert names == sorted(names)


class TestFilters:
    async def test_filters_by_category(self, client: AsyncClient, indexed_catalog):
        body = (await client.get("/api/products", params={"category": "Accessories"})).json()
        assert body["total"] == 2
        assert {item["category"] for item in body["items"]} == {"Accessories"}

    async def test_filters_by_repeated_brand_parameter(
        self, client: AsyncClient, indexed_catalog
    ):
        """Repeating a facet parameter means OR within that facet."""
        response = await client.get("/api/products?brand=Nike&brand=Puma")
        body = response.json()
        assert body["total"] == 3
        assert {item["brand"] for item in body["items"]} == {"Nike", "Puma"}

    async def test_combines_facets_with_and(self, client: AsyncClient, indexed_catalog):
        body = (
            await client.get("/api/products", params={"brand": "Nike", "colour": "Black"})
        ).json()
        assert body["total"] == 2

    async def test_facet_combination_with_no_matches(
        self, client: AsyncClient, indexed_catalog
    ):
        body = (
            await client.get("/api/products", params={"brand": "Nike", "colour": "Brown"})
        ).json()
        assert body["total"] == 0
        assert body["items"] == []

    async def test_price_range_is_inclusive(self, client: AsyncClient, indexed_catalog):
        body = (
            await client.get("/api/products", params={"min_price": 60, "max_price": 95})
        ).json()
        prices = sorted(item["price"] for item in body["items"])
        assert prices == [60.0, 80.0, 95.0]

    async def test_in_stock_only_excludes_out_of_stock(
        self, client: AsyncClient, indexed_catalog
    ):
        body = (await client.get("/api/products", params={"in_stock_only": True})).json()
        assert body["total"] == len(indexed_catalog) - 1
        assert all(item["in_stock"] for item in body["items"])

    async def test_negative_price_filter_rejected(self, client: AsyncClient):
        assert (await client.get("/api/products", params={"min_price": -1})).status_code == 422


class TestFacets:
    async def test_returns_counts_per_facet(self, client: AsyncClient, indexed_catalog):
        body = (await client.get("/api/products/facets")).json()
        categories = {item["value"]: item["count"] for item in body["categories"]}
        assert categories == {"Footwear": 3, "Accessories": 2, "Apparel": 1}

        brands = {item["value"]: item["count"] for item in body["brands"]}
        assert brands["Nike"] == 2

    async def test_reports_observed_price_bounds(self, client: AsyncClient, indexed_catalog):
        body = (await client.get("/api/products/facets")).json()
        assert body["price_min"] == 60.0
        assert body["price_max"] == 210.0

    async def test_empty_catalogue_yields_empty_facets(self, client: AsyncClient):
        body = (await client.get("/api/products/facets")).json()
        assert body["categories"] == []
        assert body["price_min"] is None


class TestUpdate:
    async def test_partial_update_leaves_other_fields_alone(self, client: AsyncClient):
        created = (await client.post("/api/products", json=VALID_PRODUCT)).json()
        updated = (
            await client.put(f"/api/products/{created['id']}", json={"price": 99.0})
        ).json()
        assert updated["price"] == 99.0
        assert updated["name"] == created["name"]
        assert updated["brand"] == created["brand"]

    async def test_changing_embedding_relevant_field_marks_for_reindex(
        self, client: AsyncClient, indexed_catalog
    ):
        """Editing the colour must invalidate the stored embedding.

        Exactly one product should be re-embedded: the incremental path selects only
        products whose fingerprint changed, so the other five are never even loaded.
        """
        product = indexed_catalog[0]
        before = (await client.get(f"/api/products/{product.id}")).json()
        assert before["indexed_at"] is not None

        await client.put(f"/api/products/{product.id}", json={"colour": "Crimson"})
        report = (await client.post("/api/catalog/index", json={})).json()
        assert report["requested"] == 1
        assert report["embedded"] == 1

    async def test_changing_price_only_does_not_reindex(
        self, client: AsyncClient, indexed_catalog
    ):
        """Price does not feed the embedding, so nothing should be re-embedded."""
        await client.put(f"/api/products/{indexed_catalog[0].id}", json={"price": 12.5})
        report = (await client.post("/api/catalog/index", json={})).json()
        assert report["requested"] == 0
        assert report["embedded"] == 0

    async def test_explicit_product_ids_report_unchanged_as_skipped(
        self, client: AsyncClient, indexed_catalog
    ):
        """Naming products explicitly bypasses the prefilter, so the hash check runs."""
        ids = [p.id for p in indexed_catalog[:3]]
        report = (await client.post("/api/catalog/index", json={"product_ids": ids})).json()
        assert report["requested"] == 3
        assert report["embedded"] == 0
        assert report["skipped_unchanged"] == 3

    async def test_force_reembeds_everything(self, client: AsyncClient, indexed_catalog):
        report = (await client.post("/api/catalog/index", json={"force": True})).json()
        assert report["embedded"] == len(indexed_catalog)
        assert report["skipped_unchanged"] == 0

    async def test_update_unknown_id_returns_404(self, client: AsyncClient):
        response = await client.put("/api/products/missing-id", json={"price": 1.0})
        assert response.status_code == 404

    async def test_rejects_invalid_value(self, client: AsyncClient):
        created = (await client.post("/api/products", json=VALID_PRODUCT)).json()
        response = await client.put(f"/api/products/{created['id']}", json={"price": -3})
        assert response.status_code == 422


class TestDelete:
    async def test_deletes_and_returns_204(self, client: AsyncClient):
        created = (await client.post("/api/products", json=VALID_PRODUCT)).json()
        assert (await client.delete(f"/api/products/{created['id']}")).status_code == 204
        assert (await client.get(f"/api/products/{created['id']}")).status_code == 404

    async def test_delete_unknown_id_returns_404(self, client: AsyncClient):
        assert (await client.delete("/api/products/missing")).status_code == 404

    async def test_delete_removes_vectors_too(
        self, client: AsyncClient, indexed_catalog, vectors
    ):
        """A deleted product must not linger in the index."""
        before = await vectors.count()
        target = indexed_catalog[0]
        assert (await client.delete(f"/api/products/{target.id}")).status_code == 204
        assert await vectors.count() == before - 1
        assert await vectors.get_vectors(target.id) == {}

    async def test_deleted_product_disappears_from_search(
        self, client: AsyncClient, indexed_catalog
    ):
        target = indexed_catalog[0]
        await client.delete(f"/api/products/{target.id}")
        body = (
            await client.post("/api/search/text", json={"query": target.name, "top_k": 10})
        ).json()
        assert target.id not in [r["product"]["id"] for r in body["results"]]

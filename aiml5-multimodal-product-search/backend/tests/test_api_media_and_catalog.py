"""Media serving and catalogue statistics endpoint tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient

from tests.conftest import make_image


class TestMediaEndpoint:
    @pytest.fixture
    def stored_image(self, image_root: Path) -> str:
        directory = image_root / "sample" / "images"
        directory.mkdir(parents=True, exist_ok=True)
        make_image((123, 45, 67)).save(directory / "42.jpg", format="JPEG")
        return "sample/images/42.jpg"

    async def test_serves_a_stored_image(self, client: AsyncClient, stored_image: str):
        response = await client.get(f"/api/media/{stored_image}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 100
        # Product images are immutable once ingested, so caching is safe.
        assert "max-age" in response.headers.get("cache-control", "")

    async def test_missing_image_returns_404(self, client: AsyncClient, image_root: Path):
        response = await client.get("/api/media/sample/images/nope.jpg")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    @pytest.mark.parametrize(
        "path",
        [
            "../../../../etc/passwd",
            "sample/../../../../etc/passwd",
            "sample/images/../../../../etc/hosts",
        ],
    )
    async def test_blocks_path_traversal(
        self, client: AsyncClient, image_root: Path, path: str
    ):
        response = await client.get(f"/api/media/{path}")
        assert response.status_code in (400, 404, 422)
        assert b"root:" not in response.content

    async def test_rejects_non_image_suffixes(self, client: AsyncClient, image_root: Path):
        """Prevents the endpoint from being used to read arbitrary catalogue files."""
        (image_root / "products.csv").write_text("id,name\n1,thing\n")
        response = await client.get("/api/media/products.csv")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_rejects_a_python_file(self, client: AsyncClient, image_root: Path):
        (image_root / "secret.py").write_text("TOKEN = 'x'")
        response = await client.get("/api/media/secret.py")
        assert response.status_code == 422
        assert b"TOKEN" not in response.content

    async def test_refuses_symlink_escaping_the_root(
        self, client: AsyncClient, image_root: Path, tmp_path: Path
    ):
        """A symlink planted inside the root must not be able to escape it.

        Path resolution follows the link, so the containment check catches this
        before the symlink check is reached - hence 422 rather than 404. What
        matters is that the file outside the root is never served.
        """
        outside = tmp_path / "outside_secret.jpg"
        make_image((1, 2, 3)).save(outside, format="JPEG")
        link = image_root / "linked.jpg"
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("symlinks are not supported on this filesystem")

        response = await client.get("/api/media/linked.jpg")
        assert response.status_code in (404, 422)
        assert response.headers["content-type"].startswith("application/json")

    async def test_allows_symlink_that_stays_inside_the_root(
        self, client: AsyncClient, image_root: Path
    ):
        """In-root symlinks are permitted: they reach nothing otherwise unreachable.

        Deployments commonly mount an images directory by symlink, and containment
        already prevents a link from escaping the root.
        """
        real = image_root / "real.jpg"
        make_image((4, 5, 6)).save(real, format="JPEG")
        link = image_root / "alias.jpg"
        try:
            link.symlink_to(real)
        except OSError:
            pytest.skip("symlinks are not supported on this filesystem")

        assert (await client.get("/api/media/real.jpg")).status_code == 200
        assert (await client.get("/api/media/alias.jpg")).status_code == 200


class TestCatalogStats:
    async def test_reports_zero_for_an_empty_catalogue(self, client: AsyncClient):
        body = (await client.get("/api/catalog/stats")).json()
        assert body["total_products"] == 0
        assert body["indexed_products"] == 0
        assert body["vector_points"] == 0
        assert body["collection"] == "test_products"

    async def test_reports_indexed_counts(self, client: AsyncClient, indexed_catalog):
        body = (await client.get("/api/catalog/stats")).json()
        assert body["total_products"] == len(indexed_catalog)
        assert body["indexed_products"] == len(indexed_catalog)
        assert body["pending_products"] == 0
        assert body["with_image_vector"] == len(indexed_catalog)
        assert body["with_text_vector"] == len(indexed_catalog)
        assert body["vector_points"] == len(indexed_catalog)
        assert body["embedding_dim"] == 32
        assert body["last_indexed_at"] is not None

    async def test_reports_pending_products(self, client: AsyncClient):
        await client.post("/api/products", json={"name": "Not Indexed Yet", "price": 5.0})
        body = (await client.get("/api/catalog/stats")).json()
        assert body["total_products"] == 1
        assert body["pending_products"] == 1
        assert body["indexed_products"] == 0


class TestCatalogIndexEndpoint:
    async def test_indexes_pending_products(self, client: AsyncClient, image_root: Path):
        make_image((7, 7, 7)).save(image_root / "one.jpg", format="JPEG")
        await client.post(
            "/api/products",
            json={"name": "Pending Product", "price": 10.0, "image_url": "one.jpg"},
        )
        report = (await client.post("/api/catalog/index", json={})).json()
        assert report["embedded"] == 1
        assert report["image_vectors"] == 1
        assert report["duration_ms"] >= 0

    async def test_report_lists_products_needing_attention(self, client: AsyncClient):
        await client.post("/api/products", json={"name": "No Image Here", "price": 1.0})
        report = (await client.post("/api/catalog/index", json={})).json()
        assert report["image_vectors"] == 0
        assert report["failures"]
        assert report["failures"][0]["name"] == "No Image Here"

    async def test_rejects_an_out_of_range_limit(self, client: AsyncClient):
        response = await client.post("/api/catalog/index", json={"limit": 0})
        assert response.status_code == 422

    async def test_rejects_an_unknown_field(self, client: AsyncClient):
        response = await client.post("/api/catalog/index", json={"forse": True})
        assert response.status_code == 422

    async def test_indexing_makes_a_product_searchable(
        self, client: AsyncClient, image_root: Path
    ):
        """End-to-end: create, index, then find it."""
        make_image((99, 44, 11)).save(image_root / "findme.jpg", format="JPEG")
        created = (
            await client.post(
                "/api/products",
                json={
                    "name": "Distinctive Findable Widget",
                    "category": "Accessories",
                    "price": 20.0,
                    "image_url": "findme.jpg",
                },
            )
        ).json()

        before = (
            await client.post("/api/search/text", json={"query": "widget", "top_k": 5})
        ).json()
        assert created["id"] not in [r["product"]["id"] for r in before["results"]]

        await client.post("/api/catalog/index", json={})

        after = (
            await client.post("/api/search/text", json={"query": "widget", "top_k": 5})
        ).json()
        assert created["id"] in [r["product"]["id"] for r in after["results"]]

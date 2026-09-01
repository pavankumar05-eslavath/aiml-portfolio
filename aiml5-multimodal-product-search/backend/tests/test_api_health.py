"""Health and readiness endpoint tests."""

from __future__ import annotations

from httpx import AsyncClient


async def test_health_reports_ok_when_dependencies_are_up(client: AsyncClient):
    """A reachable stack returns 200 even before anything is indexed.

    The overall status is *degraded* here rather than *ok*, because an empty
    collection cannot serve search. 503 is reserved for a store being unreachable.
    """
    response = await client.get("/api/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is True
    assert body["embedding_dim"] == 32
    assert body["environment"] == "test"

    by_name = {dependency["name"]: dependency for dependency in body["dependencies"]}
    assert {"database", "qdrant", "embedding_model"} <= set(by_name)
    assert by_name["database"]["status"] == "ok"
    assert by_name["embedding_model"]["status"] == "ok"


async def test_health_reports_degraded_when_collection_is_empty(client: AsyncClient):
    """An empty index is degraded, not healthy: search cannot return anything."""
    body = (await client.get("/api/health")).json()
    qdrant = next(d for d in body["dependencies"] if d["name"] == "qdrant")
    assert qdrant["status"] == "degraded"
    assert "run the indexer" in (qdrant["detail"] or "") or "0 points" in (
        qdrant["detail"] or ""
    )


async def test_health_is_ok_once_products_are_indexed(client: AsyncClient, indexed_catalog):
    body = (await client.get("/api/health")).json()
    qdrant = next(d for d in body["dependencies"] if d["name"] == "qdrant")
    assert qdrant["status"] == "ok"
    assert body["status"] == "ok"


async def test_liveness_needs_no_dependencies(client: AsyncClient):
    response = await client.get("/api/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


async def test_readiness_reports_each_check(client: AsyncClient):
    response = await client.get("/api/health/ready")
    body = response.json()
    assert body["ready"] is True
    assert body["checks"] == {"model_loaded": True, "database": True, "qdrant": True}


async def test_root_redirects_to_docs(client: AsyncClient):
    response = await client.get("/")
    assert response.status_code in (307, 308)
    assert response.headers["location"] == "/docs"


async def test_openapi_document_is_generated(client: AsyncClient):
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    assert spec["info"]["title"] == "Multimodal Product Search"
    for path in (
        "/api/search/text",
        "/api/search/image",
        "/api/search/multimodal",
        "/api/products",
        "/api/products/{product_id}",
        "/api/catalog/index",
        "/api/catalog/stats",
        "/api/health",
    ):
        assert path in spec["paths"], f"{path} missing from the OpenAPI document"


async def test_request_id_header_is_returned(client: AsyncClient):
    response = await client.get("/api/health/live")
    assert response.headers.get("X-Request-ID")
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


async def test_supplied_request_id_is_echoed(client: AsyncClient):
    response = await client.get("/api/health/live", headers={"X-Request-ID": "trace-abc-123"})
    assert response.headers["X-Request-ID"] == "trace-abc-123"


async def test_unknown_route_returns_error_envelope(client: AsyncClient):
    response = await client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"

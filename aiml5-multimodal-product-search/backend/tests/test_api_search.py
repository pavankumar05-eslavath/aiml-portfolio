"""Search endpoint tests: all three modes, filtering, ranking contract, errors.

These run against a real (embedded) Qdrant with real named vectors, so they
exercise the actual retrieval path. Only the encoder is faked - see conftest.
"""

from __future__ import annotations

import base64
import json

from httpx import AsyncClient

from tests.conftest import image_bytes


class TestTextSearch:
    async def test_returns_ranked_results(self, client: AsyncClient, indexed_catalog):
        response = await client.post(
            "/api/search/text", json={"query": "black running shoes", "top_k": 5}
        )
        assert response.status_code == 200

        body = response.json()
        assert body["mode"] == "text"
        assert body["has_image_query"] is False
        assert 0 < len(body["results"]) <= 5
        assert body["returned"] == len(body["results"])
        # Ranks are contiguous and start at 1.
        assert [r["rank"] for r in body["results"]] == list(range(1, len(body["results"]) + 1))
        # Scores are non-increasing.
        scores = [r["score"] for r in body["results"]]
        assert scores == sorted(scores, reverse=True)

    async def test_uses_both_text_channels(self, client: AsyncClient, indexed_catalog):
        """A text query is compared against product text *and* product images."""
        body = (
            await client.post("/api/search/text", json={"query": "shoes", "top_k": 5})
        ).json()
        assert set(body["channels_used"]) == {"text_to_text", "text_to_image"}
        assert body["weights"]["text_to_text"] == 0.7
        assert body["weights"]["text_to_image"] == 0.3
        assert sum(body["weights"].values()) == 1.0

    async def test_reports_score_breakdown(self, client: AsyncClient, indexed_catalog):
        body = (
            await client.post("/api/search/text", json={"query": "shoes", "top_k": 3})
        ).json()
        breakdown = body["results"][0]["breakdown"]
        assert breakdown["strategy"] == "weighted_sum"
        assert breakdown["normalization"] == "zscore"
        assert breakdown["text_similarity"] is not None
        # A text-only query has no image side to report.
        assert breakdown["image_similarity"] is None
        assert len(breakdown["channels"]) == 2
        for channel in breakdown["channels"]:
            assert channel["rank"] >= 1
            assert -1.0 <= channel["raw"] <= 1.0

    async def test_explain_false_omits_breakdown(self, client: AsyncClient, indexed_catalog):
        body = (
            await client.post(
                "/api/search/text", json={"query": "shoes", "top_k": 3, "explain": False}
            )
        ).json()
        assert all(result["breakdown"] is None for result in body["results"])

    async def test_reports_timings_and_model(self, client: AsyncClient, indexed_catalog):
        body = (
            await client.post("/api/search/text", json={"query": "shoes", "top_k": 3})
        ).json()
        timings = body["timings"]
        assert timings["total_ms"] > 0
        assert timings["embed_ms"] >= 0
        assert timings["retrieve_ms"] >= 0
        assert body["model_name"] == "test/fake-clip"

    async def test_pagination_returns_distinct_pages(
        self, client: AsyncClient, indexed_catalog
    ):
        first = (
            await client.post(
                "/api/search/text", json={"query": "shoes", "top_k": 2, "offset": 0}
            )
        ).json()
        second = (
            await client.post(
                "/api/search/text", json={"query": "shoes", "top_k": 2, "offset": 2}
            )
        ).json()
        first_ids = {r["product"]["id"] for r in first["results"]}
        second_ids = {r["product"]["id"] for r in second["results"]}
        assert first_ids.isdisjoint(second_ids)
        assert second["results"][0]["rank"] == 3

    async def test_empty_index_returns_no_results_with_warning(self, client: AsyncClient):
        body = (
            await client.post("/api/search/text", json={"query": "anything", "top_k": 5})
        ).json()
        assert body["results"] == []
        assert body["total_candidates"] == 0
        assert body["warnings"]

    async def test_identical_queries_are_deterministic(
        self, client: AsyncClient, indexed_catalog
    ):
        payload = {"query": "black shoes", "top_k": 5}
        first = (await client.post("/api/search/text", json=payload)).json()
        second = (await client.post("/api/search/text", json=payload)).json()
        assert [r["product"]["id"] for r in first["results"]] == [
            r["product"]["id"] for r in second["results"]
        ]


class TestSearchFilters:
    async def test_category_filter_constrains_results(
        self, client: AsyncClient, indexed_catalog
    ):
        body = (
            await client.post(
                "/api/search/text",
                json={
                    "query": "anything",
                    "top_k": 20,
                    "filters": {"categories": ["Accessories"]},
                },
            )
        ).json()
        assert body["results"]
        assert {r["product"]["category"] for r in body["results"]} == {"Accessories"}

    async def test_price_filter_constrains_results(self, client: AsyncClient, indexed_catalog):
        body = (
            await client.post(
                "/api/search/text",
                json={"query": "anything", "top_k": 20, "filters": {"max_price": 90}},
            )
        ).json()
        assert body["results"]
        assert all(r["product"]["price"] <= 90 for r in body["results"])

    async def test_brand_filter_accepts_multiple_values(
        self, client: AsyncClient, indexed_catalog
    ):
        body = (
            await client.post(
                "/api/search/text",
                json={
                    "query": "anything",
                    "top_k": 20,
                    "filters": {"brands": ["Nike", "Fossil"]},
                },
            )
        ).json()
        assert {r["product"]["brand"] for r in body["results"]} <= {"Nike", "Fossil"}
        assert len(body["results"]) == 3

    async def test_in_stock_filter(self, client: AsyncClient, indexed_catalog):
        body = (
            await client.post(
                "/api/search/text",
                json={"query": "anything", "top_k": 20, "filters": {"in_stock_only": True}},
            )
        ).json()
        assert all(r["product"]["in_stock"] for r in body["results"])
        assert len(body["results"]) == len(indexed_catalog) - 1

    async def test_impossible_filter_returns_empty_with_warning(
        self, client: AsyncClient, indexed_catalog
    ):
        body = (
            await client.post(
                "/api/search/text",
                json={
                    "query": "anything",
                    "top_k": 20,
                    "filters": {"brands": ["Nike"], "colours": ["Brown"]},
                },
            )
        ).json()
        assert body["results"] == []
        assert any("filter" in warning.lower() for warning in body["warnings"])

    async def test_inverted_price_range_is_rejected(self, client: AsyncClient):
        response = await client.post(
            "/api/search/text",
            json={"query": "x", "filters": {"min_price": 100, "max_price": 10}},
        )
        assert response.status_code == 422

    async def test_unknown_filter_key_is_rejected(self, client: AsyncClient):
        """Filters use extra='forbid' so a typo fails loudly instead of being ignored."""
        response = await client.post(
            "/api/search/text", json={"query": "x", "filters": {"categorie": ["Footwear"]}}
        )
        assert response.status_code == 422


class TestSortOptions:
    async def test_price_ascending(self, client: AsyncClient, indexed_catalog):
        body = (
            await client.post(
                "/api/search/text",
                json={"query": "anything", "top_k": 20, "sort": "price_asc"},
            )
        ).json()
        prices = [r["product"]["price"] for r in body["results"]]
        priced = [p for p in prices if p is not None]
        assert priced == sorted(priced)

    async def test_price_descending(self, client: AsyncClient, indexed_catalog):
        body = (
            await client.post(
                "/api/search/text",
                json={"query": "anything", "top_k": 20, "sort": "price_desc"},
            )
        ).json()
        priced = [r["product"]["price"] for r in body["results"] if r["product"]["price"]]
        assert priced == sorted(priced, reverse=True)

    async def test_name_ascending(self, client: AsyncClient, indexed_catalog):
        body = (
            await client.post(
                "/api/search/text", json={"query": "anything", "top_k": 20, "sort": "name_asc"}
            )
        ).json()
        names = [r["product"]["name"].casefold() for r in body["results"]]
        assert names == sorted(names)


class TestImageSearch:
    async def test_finds_products_from_an_upload(self, client: AsyncClient, indexed_catalog):
        response = await client.post(
            "/api/search/image",
            files={"file": ("query.jpg", image_bytes((10, 10, 10)), "image/jpeg")},
            data={"options": '{"top_k": 5}'},
        )
        assert response.status_code == 200

        body = response.json()
        assert body["mode"] == "image"
        assert body["has_image_query"] is True
        assert set(body["channels_used"]) == {"image_to_image", "image_to_text"}
        assert body["results"]

    async def test_identical_image_ranks_its_own_product_first(
        self, client: AsyncClient, indexed_catalog
    ):
        """A byte-identical query image must retrieve its own product at rank 1.

        This is the strongest available check that image vectors are stored and
        queried correctly: the fake encoder maps identical pixels to identical
        vectors, so a perfect self-match is expected.
        """
        target = indexed_catalog[3]  # the brown watch, colour (120, 70, 20)
        response = await client.post(
            "/api/search/image",
            files={"file": ("query.jpg", image_bytes((120, 70, 20)), "image/jpeg")},
            data={"options": '{"top_k": 5}'},
        )
        results = response.json()["results"]
        assert results[0]["product"]["id"] == target.id

    async def test_reports_image_similarity_only(self, client: AsyncClient, indexed_catalog):
        body = (
            await client.post(
                "/api/search/image",
                files={"file": ("q.jpg", image_bytes(), "image/jpeg")},
                data={"options": '{"top_k": 3}'},
            )
        ).json()
        breakdown = body["results"][0]["breakdown"]
        assert breakdown["image_similarity"] is not None
        assert breakdown["text_similarity"] is None

    async def test_filters_apply_to_image_search(self, client: AsyncClient, indexed_catalog):
        body = (
            await client.post(
                "/api/search/image",
                files={"file": ("q.jpg", image_bytes(), "image/jpeg")},
                data={"options": '{"top_k": 20, "filters": {"categories": ["Footwear"]}}'},
            )
        ).json()
        assert {r["product"]["category"] for r in body["results"]} == {"Footwear"}

    async def test_accepts_png_and_webp(self, client: AsyncClient, indexed_catalog):
        for fmt, mime in (("PNG", "image/png"), ("WEBP", "image/webp")):
            response = await client.post(
                "/api/search/image",
                files={"file": (f"q.{fmt.lower()}", image_bytes(fmt=fmt), mime)},
                data={"options": '{"top_k": 2}'},
            )
            assert response.status_code == 200, fmt

    async def test_rejects_non_image_upload(self, client: AsyncClient):
        response = await client.post(
            "/api/search/image",
            files={
                "file": ("notes.txt", b"just some text, definitely not an image", "text/plain")
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_image"

    async def test_rejects_image_with_lying_content_type(self, client: AsyncClient):
        """The declared MIME type is a hint; the decoder is the authority."""
        response = await client.post(
            "/api/search/image",
            files={"file": ("evil.jpg", b"\x00\x01\x02not an image at all", "image/jpeg")},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_image"

    async def test_rejects_truncated_image(self, client: AsyncClient):
        truncated = image_bytes()[:40]
        response = await client.post(
            "/api/search/image", files={"file": ("cut.jpg", truncated, "image/jpeg")}
        )
        assert response.status_code == 422

    async def test_rejects_empty_upload(self, client: AsyncClient):
        response = await client.post(
            "/api/search/image", files={"file": ("empty.jpg", b"", "image/jpeg")}
        )
        assert response.status_code == 422

    async def test_missing_file_is_rejected(self, client: AsyncClient):
        response = await client.post("/api/search/image", data={"options": "{}"})
        assert response.status_code == 422

    async def test_malformed_options_json_is_rejected(self, client: AsyncClient):
        response = await client.post(
            "/api/search/image",
            files={"file": ("q.jpg", image_bytes(), "image/jpeg")},
            data={"options": "{not json"},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_options_must_be_an_object(self, client: AsyncClient):
        response = await client.post(
            "/api/search/image",
            files={"file": ("q.jpg", image_bytes(), "image/jpeg")},
            data={"options": "[1, 2, 3]"},
        )
        assert response.status_code == 422


class TestMultimodalSearch:
    async def test_combines_image_and_text(self, client: AsyncClient, indexed_catalog):
        response = await client.post(
            "/api/search/multimodal",
            files={"file": ("q.jpg", image_bytes((10, 10, 10)), "image/jpeg")},
            data={"query": "black shoes", "options": '{"top_k": 5}'},
        )
        assert response.status_code == 200

        body = response.json()
        assert body["mode"] == "multimodal"
        assert set(body["channels_used"]) == {
            "image_to_image",
            "image_to_text",
            "text_to_text",
            "text_to_image",
        }
        assert sum(body["weights"].values()) == 1.0

    async def test_reports_both_modality_similarities(
        self, client: AsyncClient, indexed_catalog
    ):
        """The point of score-level fusion: attribution survives ranking."""
        body = (
            await client.post(
                "/api/search/multimodal",
                files={"file": ("q.jpg", image_bytes((10, 10, 10)), "image/jpeg")},
                data={"query": "black shoes", "options": '{"top_k": 5}'},
            )
        ).json()
        breakdown = body["results"][0]["breakdown"]
        assert breakdown["image_similarity"] is not None
        assert breakdown["text_similarity"] is not None

    async def test_weights_follow_the_configured_split(
        self, client: AsyncClient, indexed_catalog
    ):
        body = (
            await client.post(
                "/api/search/multimodal",
                files={"file": ("q.jpg", image_bytes(), "image/jpeg")},
                data={"query": "shoes", "options": '{"top_k": 5}'},
            )
        ).json()
        weights = body["weights"]
        image_side = weights["image_to_image"] + weights["image_to_text"]
        text_side = weights["text_to_text"] + weights["text_to_image"]
        assert image_side == 0.2  # IMAGE_WEIGHT default
        assert text_side == 0.8

    async def test_per_request_weight_override(self, client: AsyncClient, indexed_catalog):
        body = (
            await client.post(
                "/api/search/multimodal",
                files={"file": ("q.jpg", image_bytes(), "image/jpeg")},
                data={
                    "query": "shoes",
                    "options": '{"top_k": 5, "fusion": {"image_weight": 0.9, "text_weight": 0.1}}',
                },
            )
        ).json()
        weights = body["weights"]
        assert weights["image_to_image"] + weights["image_to_text"] == 0.9

    async def test_weight_override_changes_the_ranking(
        self, client: AsyncClient, indexed_catalog
    ):
        """If weights had no effect, the fusion layer would be decorative."""

        async def ranking(image_weight: float) -> list[str]:
            body = (
                await client.post(
                    "/api/search/multimodal",
                    files={"file": ("q.jpg", image_bytes((120, 70, 20)), "image/jpeg")},
                    data={
                        "query": "black running shoes",
                        "options": json.dumps(
                            {
                                "top_k": 6,
                                "fusion": {
                                    "image_weight": image_weight,
                                    "text_weight": round(1 - image_weight, 2),
                                },
                            }
                        ),
                    },
                )
            ).json()
            return [r["product"]["id"] for r in body["results"]]

        image_heavy = await ranking(1.0)
        text_heavy = await ranking(0.0)
        assert image_heavy != text_heavy

    async def test_text_only_submission_degrades_to_text_mode(
        self, client: AsyncClient, indexed_catalog
    ):
        """One endpoint can serve every mode, which simplifies the frontend."""
        body = (
            await client.post(
                "/api/search/multimodal",
                data={"query": "black shoes", "options": '{"top_k": 3}'},
            )
        ).json()
        assert body["mode"] == "text"
        assert body["has_image_query"] is False

    async def test_rejects_request_with_neither_modality(self, client: AsyncClient):
        response = await client.post("/api/search/multimodal", data={"options": "{}"})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "empty_query"

    async def test_rrf_strategy_produces_valid_ranking(
        self, client: AsyncClient, indexed_catalog
    ):
        body = (
            await client.post(
                "/api/search/multimodal",
                files={"file": ("q.jpg", image_bytes(), "image/jpeg")},
                data={
                    "query": "shoes",
                    "options": '{"top_k": 5, "fusion": {"strategy": "rrf"}}',
                },
            )
        ).json()
        assert body["strategy"] == "rrf"
        scores = [r["score"] for r in body["results"]]
        assert scores == sorted(scores, reverse=True)

    async def test_embedding_fusion_uses_fused_channels(
        self, client: AsyncClient, indexed_catalog
    ):
        """Embedding fusion trades attribution for a single blended query vector."""
        body = (
            await client.post(
                "/api/search/multimodal",
                files={"file": ("q.jpg", image_bytes(), "image/jpeg")},
                data={
                    "query": "shoes",
                    "options": '{"top_k": 5, "fusion": {"strategy": "embedding_fusion"}}',
                },
            )
        ).json()
        assert body["strategy"] == "embedding_fusion"
        assert set(body["channels_used"]) == {"fused_to_image", "fused_to_text"}
        breakdown = body["results"][0]["breakdown"]
        assert breakdown["image_similarity"] is None
        assert breakdown["text_similarity"] is None

    async def test_embedding_fusion_falls_back_for_single_modality(
        self, client: AsyncClient, indexed_catalog
    ):
        body = (
            await client.post(
                "/api/search/text",
                json={
                    "query": "shoes",
                    "top_k": 3,
                    "fusion": {"strategy": "embedding_fusion"},
                },
            )
        ).json()
        assert body["strategy"] == "weighted_sum"
        assert any("embedding_fusion" in warning for warning in body["warnings"])


class TestJsonMultimodalSearch:
    async def test_accepts_base64_image(self, client: AsyncClient, indexed_catalog):
        encoded = base64.b64encode(image_bytes((10, 10, 10))).decode()
        response = await client.post(
            "/api/search/multimodal/json",
            json={"query": "black shoes", "image_base64": encoded, "top_k": 4},
        )
        assert response.status_code == 200
        assert response.json()["mode"] == "multimodal"

    async def test_accepts_data_url_prefix(self, client: AsyncClient, indexed_catalog):
        encoded = base64.b64encode(image_bytes()).decode()
        response = await client.post(
            "/api/search/multimodal/json",
            json={"image_base64": f"data:image/jpeg;base64,{encoded}", "top_k": 3},
        )
        assert response.status_code == 200

    async def test_rejects_invalid_base64(self, client: AsyncClient):
        response = await client.post(
            "/api/search/multimodal/json", json={"image_base64": "!!!not base64!!!"}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_image"

    async def test_rejects_both_image_sources(self, client: AsyncClient):
        response = await client.post(
            "/api/search/multimodal/json",
            json={"image_base64": "abcd", "image_product_id": "some-id"},
        )
        assert response.status_code == 422

    async def test_requires_at_least_one_input(self, client: AsyncClient):
        response = await client.post("/api/search/multimodal/json", json={"top_k": 5})
        assert response.status_code == 422

    async def test_search_by_catalogue_product_image(
        self, client: AsyncClient, indexed_catalog
    ):
        target = indexed_catalog[0]
        body = (
            await client.post(
                "/api/search/multimodal/json",
                json={"image_product_id": target.id, "top_k": 5},
            )
        ).json()
        assert body["mode"] == "image"
        # A product is never its own recommendation.
        assert target.id not in [r["product"]["id"] for r in body["results"]]


class TestSimilarSearch:
    async def test_returns_neighbours_excluding_itself(
        self, client: AsyncClient, indexed_catalog
    ):
        target = indexed_catalog[0]
        response = await client.post(f"/api/search/similar/{target.id}", json={"top_k": 4})
        assert response.status_code == 200

        body = response.json()
        assert body["results"]
        assert target.id not in [r["product"]["id"] for r in body["results"]]

    async def test_unknown_product_returns_404(self, client: AsyncClient, indexed_catalog):
        response = await client.post("/api/search/similar/no-such-product", json={"top_k": 3})
        assert response.status_code == 404

    async def test_unindexed_product_reports_a_warning(self, client: AsyncClient):
        created = (
            await client.post(
                "/api/products", json={"name": "Never Indexed Product", "price": 10.0}
            )
        ).json()
        response = await client.post(f"/api/search/similar/{created['id']}", json={"top_k": 3})
        # Either an explanatory empty-query error or an empty result with a warning
        # is acceptable; silently returning nothing is not.
        if response.status_code == 200:
            assert response.json()["warnings"]
        else:
            assert response.status_code == 422
            assert response.json()["error"]["code"] == "empty_query"

    async def test_text_only_product_falls_back_to_its_description(self, client: AsyncClient):
        created = (
            await client.post(
                "/api/products",
                json={
                    "name": "Text Only Running Shoe",
                    "category": "Footwear",
                    "subcategory": "Sports Shoes",
                    "price": 40.0,
                },
            )
        ).json()
        # Give it a text vector but no image (no image_url was supplied).
        report = (
            await client.post("/api/catalog/index", json={"product_ids": [created["id"]]})
        ).json()
        assert report["text_vectors"] == 1
        assert report["image_vectors"] == 0

        body = (
            await client.post(f"/api/search/similar/{created['id']}", json={"top_k": 3})
        ).json()
        assert body["mode"] == "text"
        assert any("no indexed image" in warning for warning in body["warnings"])


class TestSearchValidation:
    async def test_rejects_empty_query_string(self, client: AsyncClient):
        response = await client.post("/api/search/text", json={"query": ""})
        assert response.status_code == 422

    async def test_rejects_whitespace_only_query(self, client: AsyncClient, indexed_catalog):
        response = await client.post("/api/search/text", json={"query": "   "})
        assert response.status_code == 422
        assert response.json()["error"]["code"] in {"empty_query", "validation_error"}

    async def test_rejects_missing_query_field(self, client: AsyncClient):
        response = await client.post("/api/search/text", json={"top_k": 5})
        assert response.status_code == 422

    async def test_rejects_top_k_above_the_cap(self, client: AsyncClient):
        response = await client.post("/api/search/text", json={"query": "x", "top_k": 9999})
        assert response.status_code == 422
        details = response.json()["error"]["details"]
        assert any("top_k" in field["field"] for field in details["fields"])

    async def test_rejects_zero_top_k(self, client: AsyncClient):
        response = await client.post("/api/search/text", json={"query": "x", "top_k": 0})
        assert response.status_code == 422

    async def test_rejects_negative_offset(self, client: AsyncClient):
        response = await client.post("/api/search/text", json={"query": "x", "offset": -1})
        assert response.status_code == 422

    async def test_rejects_overlong_query(self, client: AsyncClient):
        response = await client.post("/api/search/text", json={"query": "a" * 5000})
        assert response.status_code == 422

    async def test_rejects_out_of_range_fusion_weight(self, client: AsyncClient):
        response = await client.post(
            "/api/search/text", json={"query": "x", "fusion": {"image_weight": 5.0}}
        )
        assert response.status_code == 422

    async def test_rejects_unknown_fusion_key(self, client: AsyncClient):
        response = await client.post(
            "/api/search/text", json={"query": "x", "fusion": {"image_wieght": 0.5}}
        )
        assert response.status_code == 422

    async def test_rejects_unknown_strategy(self, client: AsyncClient):
        response = await client.post(
            "/api/search/text", json={"query": "x", "fusion": {"strategy": "magic"}}
        )
        assert response.status_code == 422

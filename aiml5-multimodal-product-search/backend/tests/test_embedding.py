"""Embedding service tests.

The fake encoder covers the service's *own* logic: normalisation, fusion
arithmetic, batching, caching and lifecycle guards. Behaviour that only the real
model can demonstrate - that image and text land in a shared space - is tested in
``test_embedding_real.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import Settings
from app.core.exceptions import EmbeddingError, ModelNotReadyError
from app.services.embedding import (
    EmbeddingService,
    _resolve_device,
    l2_normalise,
)
from tests.conftest import FAKE_DIM, FakeEmbeddingService, make_image


class TestL2Normalise:
    def test_produces_unit_vectors(self):
        result = l2_normalise(np.array([3.0, 4.0]))
        assert np.linalg.norm(result[0]) == pytest.approx(1.0)
        assert result[0].tolist() == pytest.approx([0.6, 0.8])

    def test_promotes_a_single_vector_to_a_batch(self):
        assert l2_normalise(np.array([1.0, 0.0, 0.0])).shape == (1, 3)

    def test_normalises_each_row_independently(self):
        result = l2_normalise(np.array([[3.0, 4.0], [0.0, 5.0]]))
        assert np.allclose(np.linalg.norm(result, axis=1), 1.0)

    def test_zero_vector_does_not_divide_by_zero(self):
        result = l2_normalise(np.zeros((1, 4)))
        assert np.all(np.isfinite(result))

    def test_output_is_float32(self):
        """Qdrant stores float32; converting once here avoids surprises later."""
        assert l2_normalise(np.array([1.0, 2.0], dtype=np.float64)).dtype == np.float32

    def test_cosine_reduces_to_a_dot_product(self):
        """The reason vectors are normalised once, at the source."""
        a = l2_normalise(np.array([1.0, 2.0, 3.0]))[0]
        b = l2_normalise(np.array([3.0, 2.0, 1.0]))[0]
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert float(np.dot(a, b)) == pytest.approx(cosine)


class TestLifecycle:
    def test_info_before_load_raises(self, settings: Settings):
        service = EmbeddingService(settings)
        assert service.is_loaded is False
        with pytest.raises(ModelNotReadyError):
            _ = service.info

    def test_embedding_before_load_raises(self, settings: Settings):
        service = EmbeddingService(settings)
        with pytest.raises(ModelNotReadyError):
            service.embed_texts(["hello"])

    def test_configured_name_is_readable_before_load(self, settings: Settings):
        service = EmbeddingService(settings)
        assert service.configured_model_name == settings.model_name

    def test_load_populates_metadata(self, embedder: FakeEmbeddingService):
        info = embedder.info
        assert info.embedding_dim == FAKE_DIM
        assert info.model_type == "clip"
        assert embedder.embedding_dim == FAKE_DIM

    def test_unload_resets_state(self, embedder: FakeEmbeddingService):
        embedder.unload()
        assert embedder.is_loaded is False
        with pytest.raises(ModelNotReadyError):
            _ = embedder.info

    def test_rejects_a_model_without_the_required_interface(
        self, settings: Settings, monkeypatch
    ):
        """A model lacking both towers cannot support multimodal search."""

        class Incomplete:
            config = type("C", (), {"model_type": "bert"})()

            def to(self, _device):
                return self

            def eval(self):
                return self

        monkeypatch.setattr(
            "app.services.embedding.AutoModel",
            type("M", (), {"from_pretrained": staticmethod(lambda *a, **k: Incomplete())}),
        )
        monkeypatch.setattr(
            "app.services.embedding.AutoProcessor",
            type("P", (), {"from_pretrained": staticmethod(lambda *a, **k: object())}),
        )
        service = EmbeddingService(settings)
        with pytest.raises(ModelNotReadyError, match="does not expose"):
            service.load()

    def test_load_failure_is_reported_as_not_ready(self, settings: Settings, monkeypatch):
        def _explode(*_args, **_kwargs):
            raise OSError("no such model on the hub")

        monkeypatch.setattr(
            "app.services.embedding.AutoModel",
            type("M", (), {"from_pretrained": staticmethod(_explode)}),
        )
        service = EmbeddingService(settings)
        with pytest.raises(ModelNotReadyError, match="Failed to load model"):
            service.load()


class TestDeviceResolution:
    def test_explicit_cpu_is_respected(self):
        assert _resolve_device("cpu") == "cpu"

    def test_auto_falls_back_to_cpu_without_accelerators(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
        assert _resolve_device("auto") == "cpu"

    def test_auto_prefers_cuda_when_present(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        assert _resolve_device("auto") == "cuda"

    def test_requesting_missing_cuda_degrades_to_cpu(self, monkeypatch):
        """A CPU-only host must still start, not crash."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        assert _resolve_device("cuda") == "cpu"


class TestEncoding:
    def test_text_embeddings_are_normalised(self, embedder: FakeEmbeddingService):
        vectors = embedder.embed_texts(["a", "b", "c"])
        assert vectors.shape == (3, FAKE_DIM)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_image_embeddings_are_normalised(self, embedder: FakeEmbeddingService):
        vectors = embedder.embed_images([make_image((1, 2, 3)), make_image((9, 9, 9))])
        assert vectors.shape == (2, FAKE_DIM)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_empty_input_returns_a_correctly_shaped_array(self, embedder: FakeEmbeddingService):
        assert embedder.embed_texts([]).shape == (0, FAKE_DIM)
        assert embedder.embed_images([]).shape == (0, FAKE_DIM)

    def test_identical_input_gives_identical_output(self, embedder: FakeEmbeddingService):
        first = embedder.embed_texts(["same text"])
        second = embedder.embed_texts(["same text"])
        assert np.allclose(first, second)

    def test_different_input_gives_different_output(self, embedder: FakeEmbeddingService):
        vectors = embedder.embed_texts(["alpha", "beta"])
        assert not np.allclose(vectors[0], vectors[1])

    def test_blank_text_does_not_crash(self, embedder: FakeEmbeddingService):
        assert embedder.embed_texts(["", "   "]).shape == (2, FAKE_DIM)

    async def test_async_wrappers_delegate(self, embedder: FakeEmbeddingService):
        vector = await embedder.embed_text("a query")
        assert vector.shape == (FAKE_DIM,)
        image_vector = await embedder.embed_image(make_image((5, 5, 5)))
        assert image_vector.shape == (FAKE_DIM,)


class TestTextCache:
    async def test_repeated_query_hits_the_cache(self, settings: Settings):
        settings.text_embedding_cache_size = 16
        service = FakeEmbeddingService(settings)
        service.load()

        await service.embed_text("recurring query")
        calls_after_first = service.text_calls
        await service.embed_text("recurring query")

        assert service.text_calls == calls_after_first
        assert service.cache_stats()["hits"] == 1

    async def test_distinct_queries_miss(self, settings: Settings):
        settings.text_embedding_cache_size = 16
        service = FakeEmbeddingService(settings)
        service.load()
        await service.embed_text("one")
        await service.embed_text("two")
        assert service.cache_stats()["misses"] == 2

    async def test_cache_can_be_disabled(self, settings: Settings):
        settings.text_embedding_cache_size = 0
        service = FakeEmbeddingService(settings)
        service.load()
        await service.embed_text("q")
        await service.embed_text("q")
        assert service.text_calls == 2

    async def test_eviction_respects_capacity(self, settings: Settings):
        settings.text_embedding_cache_size = 2
        service = FakeEmbeddingService(settings)
        service.load()
        for query in ("a", "b", "c"):
            await service.embed_text(query)
        assert service.cache_stats()["size"] == 2


class TestFusion:
    def test_alpha_one_returns_the_image_vector(self, embedder: FakeEmbeddingService):
        image = embedder.embed_images([make_image((1, 1, 1))])[0]
        text = embedder.embed_texts(["something"])[0]
        assert np.allclose(embedder.fuse(image, text, 1.0), image, atol=1e-5)

    def test_alpha_zero_returns_the_text_vector(self, embedder: FakeEmbeddingService):
        image = embedder.embed_images([make_image((1, 1, 1))])[0]
        text = embedder.embed_texts(["something"])[0]
        assert np.allclose(embedder.fuse(image, text, 0.0), text, atol=1e-5)

    def test_result_is_renormalised(self, embedder: FakeEmbeddingService):
        """A convex blend of two unit vectors has norm < 1; scores would drift."""
        image = embedder.embed_images([make_image((1, 1, 1))])[0]
        text = embedder.embed_texts(["unrelated"])[0]
        fused = embedder.fuse(image, text, 0.5)
        assert np.linalg.norm(fused) == pytest.approx(1.0, abs=1e-5)

    def test_intermediate_alpha_lies_between_the_inputs(self, embedder: FakeEmbeddingService):
        image = embedder.embed_images([make_image((1, 1, 1))])[0]
        text = embedder.embed_texts(["unrelated"])[0]
        fused = embedder.fuse(image, text, 0.5)
        assert float(np.dot(fused, image)) > 0
        assert float(np.dot(fused, text)) > 0
        # And is closer to each input than the inputs are to each other.
        assert float(np.dot(fused, image)) > float(np.dot(image, text))

    def test_alpha_shifts_similarity_monotonically(self, embedder: FakeEmbeddingService):
        image = embedder.embed_images([make_image((1, 1, 1))])[0]
        text = embedder.embed_texts(["unrelated"])[0]
        similarities = [
            float(np.dot(embedder.fuse(image, text, alpha), image))
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        assert similarities == sorted(similarities)

    def test_rejects_alpha_outside_the_unit_interval(self, embedder: FakeEmbeddingService):
        image = embedder.embed_images([make_image((1, 1, 1))])[0]
        text = embedder.embed_texts(["x"])[0]
        for alpha in (-0.1, 1.1):
            with pytest.raises(ValueError, match="alpha"):
                embedder.fuse(image, text, alpha)

    def test_rejects_mismatched_shapes(self, embedder: FakeEmbeddingService):
        with pytest.raises(EmbeddingError, match="differing shapes"):
            embedder.fuse(np.ones(FAKE_DIM), np.ones(FAKE_DIM + 4), 0.5)

    async def test_embed_multimodal_uses_the_configured_alpha(
        self, embedder: FakeEmbeddingService, settings: Settings
    ):
        settings.embedding_fusion_alpha = 1.0
        fused = await embedder.embed_multimodal(make_image((2, 2, 2)), "text query")
        image_only = embedder.embed_images([make_image((2, 2, 2))])[0]
        assert np.allclose(fused, image_only, atol=1e-5)


class TestAlignmentProbe:
    def test_reports_perfect_alignment_for_matched_pairs(self, embedder: FakeEmbeddingService):
        """With the fake encoder, pinning text vectors to image vectors is exact."""
        images = [make_image((10, 20, 30)), make_image((200, 100, 50))]
        image_vectors = embedder.embed_images(images)
        texts = ["first caption", "second caption"]
        for text, vector in zip(texts, image_vectors, strict=True):
            embedder.register(text, vector)

        report = embedder.probe_alignment(images, texts)
        assert report.pairs == 2
        assert report.cross_modal_top1 == 1.0
        assert report.matched_pair_similarity > report.mismatched_pair_similarity
        assert report.is_aligned

    def test_detects_misalignment(self, embedder: FakeEmbeddingService):
        """An unaligned pair of towers must be reported as such, not assumed fine."""
        images = [make_image((1, 1, 1)), make_image((250, 250, 250))]
        report = embedder.probe_alignment(images, ["unrelated one", "unrelated two"])
        assert report.cross_modal_top1 < 1.0 or not report.is_aligned

    def test_requires_matching_lengths(self, embedder: FakeEmbeddingService):
        with pytest.raises(ValueError, match="same length"):
            embedder.probe_alignment([make_image((1, 1, 1))], ["a", "b"])

    def test_requires_at_least_two_pairs(self, embedder: FakeEmbeddingService):
        with pytest.raises(ValueError, match="at least two"):
            embedder.probe_alignment([make_image((1, 1, 1))], ["a"])

"""Tests that exercise the real embedding model.

Skipped by default: they download ~600 MB of weights on first run and take
seconds per test, which does not belong in a fast feedback loop. Enable with::

    RUN_SLOW_TESTS=1 pytest -m slow

What these cover is the one thing the fake encoder cannot: that the loaded model
genuinely places images and text in a **shared** space. Multimodal fusion is only
meaningful if that holds, and this project's position is that it should be measured
rather than assumed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.core.config import Settings
from app.core.text import build_product_document
from app.services.embedding import EmbeddingService
from tests.conftest import make_image

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RUN_SLOW_TESTS") != "1",
        reason="set RUN_SLOW_TESTS=1 to run tests that load the real model",
    ),
]

MODEL = os.environ.get("TEST_REAL_MODEL", "openai/clip-vit-base-patch32")


@pytest.fixture(scope="module")
def real_service() -> EmbeddingService:
    """Load the real encoder once for the module."""
    settings = Settings(_env_file=None, model_name=MODEL, model_device="cpu")  # type: ignore[call-arg]
    service = EmbeddingService(settings)
    service.load()
    return service


def test_reports_the_true_embedding_dimension(real_service: EmbeddingService):
    """The dimension comes from the model, never from configuration."""
    assert real_service.embedding_dim > 0
    vectors = real_service.embed_texts(["a product"])
    assert vectors.shape == (1, real_service.embedding_dim)


def test_image_and_text_vectors_share_a_dimension(real_service: EmbeddingService):
    """A precondition for cross-modal comparison."""
    text = real_service.embed_texts(["a red shoe"])
    image = real_service.embed_images([make_image((200, 20, 20))])
    assert text.shape[1] == image.shape[1] == real_service.embedding_dim


def test_embeddings_are_unit_norm(real_service: EmbeddingService):
    vectors = real_service.embed_texts(["one", "two", "three"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_semantically_similar_text_scores_higher(real_service: EmbeddingService):
    """Sanity check that the text tower encodes meaning, not surface form."""
    vectors = real_service.embed_texts(
        [
            "a pair of black running shoes",
            "black sports trainers for running",
            "a bottle of strawberry jam",
        ]
    )
    related = float(np.dot(vectors[0], vectors[1]))
    unrelated = float(np.dot(vectors[0], vectors[2]))
    assert related > unrelated


def test_cross_modal_alignment_is_measurable(real_service: EmbeddingService):
    """The central assumption of the whole architecture, verified rather than assumed.

    Coloured swatches are crude stand-ins for products, but CLIP reliably associates
    a colour word with the corresponding colour, which is enough to prove the two
    towers project into one comparable space.
    """
    swatches = {
        "red": (220, 20, 20),
        "green": (20, 180, 20),
        "blue": (20, 20, 220),
        "yellow": (240, 230, 30),
    }
    images = [make_image(rgb, size=(224, 224)) for rgb in swatches.values()]
    texts = [f"a solid {name} colour swatch" for name in swatches]

    report = real_service.probe_alignment(images, texts)

    assert report.pairs == 4
    assert report.matched_pair_similarity > report.mismatched_pair_similarity
    assert report.cross_modal_top1 >= 0.5
    assert report.is_aligned


def test_modality_gap_is_present_as_documented(real_service: EmbeddingService):
    """Same-modality similarities sit well above cross-modal ones.

    This is the measured fact that motivates per-channel score normalisation in the
    fusion layer. If a future model closed the gap, the normalisation default could
    be revisited - so it is worth asserting the gap actually exists.
    """
    swatches = [(220, 20, 20), (20, 180, 20), (20, 20, 220), (240, 230, 30)]
    images = [make_image(rgb, size=(224, 224)) for rgb in swatches]
    texts = ["a red swatch", "a green swatch", "a blue swatch", "a yellow swatch"]

    report = real_service.probe_alignment(images, texts)
    assert report.image_image_similarity > report.matched_pair_similarity
    assert report.text_text_similarity > report.matched_pair_similarity


def test_attribute_template_beats_a_bare_sku_name(real_service: EmbeddingService):
    """Justifies the templated product document.

    Compares cross-modal retrieval using a terse catalogue name against the
    caption-style document the ingestion pipeline builds. The template should be at
    least as good; on the real catalogue it measured 1.000 versus 0.938.
    """
    products = [
        {"name": "AZ-1 Blk Snkr", "colour": "Black", "subcategory": "Sports Shoes"},
        {"name": "RD-9 Red Dress", "colour": "Red", "subcategory": "Dresses"},
        {"name": "BL-3 Blu Jean", "colour": "Blue", "subcategory": "Jeans"},
    ]
    images = [
        make_image((15, 15, 15), size=(224, 224)),
        make_image((200, 20, 40), size=(224, 224)),
        make_image((30, 60, 190), size=(224, 224)),
    ]

    bare = real_service.probe_alignment(images, [p["name"] for p in products])
    templated = real_service.probe_alignment(
        images,
        [
            build_product_document(
                name=p["name"], colour=p["colour"], subcategory=p["subcategory"]
            )
            for p in products
        ],
    )
    assert templated.cross_modal_top1 >= bare.cross_modal_top1


def test_fusion_moves_the_query_between_modalities(real_service: EmbeddingService):
    """Alpha must actually trade off between the image and the text."""
    image = real_service.embed_images([make_image((15, 15, 15), size=(224, 224))])[0]
    text = real_service.embed_texts(["a bright yellow raincoat"])[0]

    similarities = [
        float(np.dot(real_service.fuse(image, text, alpha), image))
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert similarities == sorted(similarities)
    assert similarities[-1] > similarities[0]


def test_warmup_completes(real_service: EmbeddingService):
    assert real_service.warmup() >= 0.0

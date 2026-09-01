"""Tests for the score fusion and ranking layer.

These are the most valuable unit tests in the project: fusion is where ranking is
decided, it is pure, and it is where a silent bug would degrade quality without
raising anything. Expected values are computed by hand in the test bodies rather
than snapshotted, so a change in behaviour has to be justified rather than blessed.
"""

from __future__ import annotations

import math

import pytest

from app.core.config import FusionStrategy, ScoreNormalization
from app.schemas.search import RetrievalChannel, SearchMode
from app.services.fusion import (
    ChannelResult,
    fuse,
    normalise_scores,
    resolve_channel_weights,
)


class TestNormaliseScores:
    def test_minmax_maps_to_unit_interval(self):
        assert normalise_scores([0.2, 0.4, 0.6], ScoreNormalization.MINMAX) == pytest.approx(
            [0.0, 0.5, 1.0]
        )

    def test_minmax_handles_identical_scores(self):
        """A channel with no spread carries no ranking information."""
        assert normalise_scores([0.5, 0.5, 0.5], ScoreNormalization.MINMAX) == [1.0, 1.0, 1.0]

    def test_minmax_handles_single_value(self):
        assert normalise_scores([0.42], ScoreNormalization.MINMAX) == [1.0]

    def test_none_is_passthrough(self):
        raw = [0.73, 0.20, 0.51]
        assert normalise_scores(raw, ScoreNormalization.NONE) == raw

    def test_zscore_is_centred_and_bounded(self):
        result = normalise_scores([0.1, 0.5, 0.9], ScoreNormalization.ZSCORE)
        assert all(0.0 < value < 1.0 for value in result)
        # The middle value sits at the mean, which the logistic maps to exactly 0.5.
        assert result[1] == pytest.approx(0.5)
        assert result[0] < result[1] < result[2]

    def test_zscore_handles_zero_variance(self):
        assert normalise_scores([0.3, 0.3], ScoreNormalization.ZSCORE) == [0.5, 0.5]

    def test_empty_input(self):
        assert normalise_scores([], ScoreNormalization.MINMAX) == []

    def test_normalisation_makes_channels_comparable(self):
        """The reason this layer exists.

        Same-modality cosines (~0.7) and cross-modal cosines (~0.2) occupy
        different ranges. After min-max both channels span [0, 1], so a weight of
        0.5 contributes equally from each.
        """
        same_modality = [0.75, 0.72, 0.70]
        cross_modal = [0.24, 0.21, 0.19]
        assert normalise_scores(same_modality, ScoreNormalization.MINMAX) == pytest.approx(
            normalise_scores(cross_modal, ScoreNormalization.MINMAX)
        )


class TestResolveChannelWeights:
    def _weights(self, **kwargs):
        defaults = {
            "mode": SearchMode.MULTIMODAL,
            "strategy": FusionStrategy.WEIGHTED_SUM,
            "image_weight": 0.5,
            "text_weight": 0.5,
            "lexical_weight": 0.0,
            "cross_modal_weight": 0.3,
            "has_text": True,
            "has_image": True,
        }
        return resolve_channel_weights(**{**defaults, **kwargs})

    def test_weights_always_sum_to_one(self):
        """Otherwise final scores would not be comparable across modes."""
        for kwargs in (
            {},
            {"has_image": False, "mode": SearchMode.TEXT},
            {"has_text": False, "mode": SearchMode.IMAGE},
            {"lexical_weight": 0.3},
            {"cross_modal_weight": 0.0},
            {"cross_modal_weight": 1.0},
        ):
            assert sum(self._weights(**kwargs).values()) == pytest.approx(1.0)

    def test_multimodal_activates_four_channels(self):
        weights = self._weights()
        assert set(weights) == {
            RetrievalChannel.IMAGE_TO_IMAGE,
            RetrievalChannel.IMAGE_TO_TEXT,
            RetrievalChannel.TEXT_TO_TEXT,
            RetrievalChannel.TEXT_TO_IMAGE,
        }

    def test_modality_split_respects_image_and_text_weight(self):
        weights = self._weights(image_weight=0.2, text_weight=0.8, cross_modal_weight=0.25)
        image_side = (
            weights[RetrievalChannel.IMAGE_TO_IMAGE] + weights[RetrievalChannel.IMAGE_TO_TEXT]
        )
        text_side = (
            weights[RetrievalChannel.TEXT_TO_TEXT] + weights[RetrievalChannel.TEXT_TO_IMAGE]
        )
        assert image_side == pytest.approx(0.2)
        assert text_side == pytest.approx(0.8)
        # Within the image side, 25% goes to the cross-modal channel.
        assert weights[RetrievalChannel.IMAGE_TO_TEXT] == pytest.approx(0.2 * 0.25)

    def test_single_modality_gets_full_weight(self):
        """A text-only query must not be scaled down by the image weight."""
        weights = self._weights(
            has_image=False, mode=SearchMode.TEXT, image_weight=0.9, text_weight=0.1
        )
        assert set(weights) == {
            RetrievalChannel.TEXT_TO_TEXT,
            RetrievalChannel.TEXT_TO_IMAGE,
        }
        assert weights[RetrievalChannel.TEXT_TO_TEXT] == pytest.approx(0.7)
        assert weights[RetrievalChannel.TEXT_TO_IMAGE] == pytest.approx(0.3)

    def test_cross_modal_zero_disables_cross_channels(self):
        weights = self._weights(cross_modal_weight=0.0)
        assert RetrievalChannel.IMAGE_TO_TEXT not in weights
        assert RetrievalChannel.TEXT_TO_IMAGE not in weights

    def test_cross_modal_one_disables_same_modality_channels(self):
        weights = self._weights(cross_modal_weight=1.0)
        assert RetrievalChannel.IMAGE_TO_IMAGE not in weights
        assert RetrievalChannel.TEXT_TO_TEXT not in weights

    def test_lexical_requires_text(self):
        assert RetrievalChannel.LEXICAL in self._weights(lexical_weight=0.2)
        assert RetrievalChannel.LEXICAL not in self._weights(
            lexical_weight=0.2, has_text=False, mode=SearchMode.IMAGE
        )

    def test_embedding_fusion_uses_fused_channels(self):
        weights = self._weights(strategy=FusionStrategy.EMBEDDING_FUSION)
        assert set(weights) == {
            RetrievalChannel.FUSED_TO_IMAGE,
            RetrievalChannel.FUSED_TO_TEXT,
        }

    def test_embedding_fusion_falls_back_when_single_modality(self):
        """With one modality there is nothing to blend, so normal channels apply."""
        weights = self._weights(
            strategy=FusionStrategy.EMBEDDING_FUSION, has_image=False, mode=SearchMode.TEXT
        )
        assert RetrievalChannel.FUSED_TO_IMAGE not in weights
        assert RetrievalChannel.TEXT_TO_TEXT in weights

    def test_zero_weights_produce_empty_mapping(self):
        assert self._weights(image_weight=0.0, text_weight=0.0) == {}


class TestFuse:
    def test_weighted_sum_arithmetic_is_exact(self):
        """Verify the fused score against a hand computation."""
        channels = [
            ChannelResult.from_scores(
                RetrievalChannel.TEXT_TO_TEXT, [("a", 0.9), ("b", 0.5), ("c", 0.1)]
            ),
            ChannelResult.from_scores(
                RetrievalChannel.TEXT_TO_IMAGE, [("a", 0.2), ("b", 0.3), ("c", 0.4)]
            ),
        ]
        weights = {
            RetrievalChannel.TEXT_TO_TEXT: 0.7,
            RetrievalChannel.TEXT_TO_IMAGE: 0.3,
        }
        results = fuse(channels, weights, normalization=ScoreNormalization.MINMAX)
        scores = {r.product_id: r.final_score for r in results}

        # text_to_text min-max: a=1.0, b=0.5, c=0.0
        # text_to_image min-max: a=0.0, b=0.5, c=1.0
        assert scores["a"] == pytest.approx(0.7 * 1.0 + 0.3 * 0.0)
        assert scores["b"] == pytest.approx(0.7 * 0.5 + 0.3 * 0.5)
        assert scores["c"] == pytest.approx(0.7 * 0.0 + 0.3 * 1.0)

    def test_results_are_sorted_descending(self):
        channels = [
            ChannelResult.from_scores(
                RetrievalChannel.TEXT_TO_TEXT, [("low", 0.1), ("high", 0.9), ("mid", 0.5)]
            )
        ]
        results = fuse(channels, {RetrievalChannel.TEXT_TO_TEXT: 1.0})
        assert [r.product_id for r in results] == ["high", "mid", "low"]

    def test_ties_break_deterministically(self):
        """Identical scores must produce a stable order across runs."""
        channels = [
            ChannelResult.from_scores(
                RetrievalChannel.TEXT_TO_TEXT, [("b", 0.5), ("a", 0.5), ("c", 0.5)]
            )
        ]
        first = fuse(channels, {RetrievalChannel.TEXT_TO_TEXT: 1.0})
        second = fuse(channels, {RetrievalChannel.TEXT_TO_TEXT: 1.0})
        assert (
            [r.product_id for r in first] == [r.product_id for r in second] == ["a", "b", "c"]
        )

    def test_missing_from_a_channel_contributes_nothing(self):
        """A product only one channel retrieved gets that channel's weight only."""
        channels = [
            ChannelResult.from_scores(
                RetrievalChannel.TEXT_TO_TEXT, [("both", 0.8), ("only_text", 0.8)]
            ),
            ChannelResult.from_scores(RetrievalChannel.TEXT_TO_IMAGE, [("both", 0.5)]),
        ]
        weights = {
            RetrievalChannel.TEXT_TO_TEXT: 0.5,
            RetrievalChannel.TEXT_TO_IMAGE: 0.5,
        }
        results = {r.product_id: r for r in fuse(channels, weights)}
        # Both tie on the text channel (identical scores -> 1.0 each), but only
        # "both" also earns the image channel's contribution.
        assert results["both"].final_score > results["only_text"].final_score

    def test_union_of_all_channels_is_returned(self):
        channels = [
            ChannelResult.from_scores(RetrievalChannel.TEXT_TO_TEXT, [("a", 0.5)]),
            ChannelResult.from_scores(RetrievalChannel.TEXT_TO_IMAGE, [("b", 0.5)]),
        ]
        results = fuse(
            channels,
            {
                RetrievalChannel.TEXT_TO_TEXT: 0.5,
                RetrievalChannel.TEXT_TO_IMAGE: 0.5,
            },
        )
        assert {r.product_id for r in results} == {"a", "b"}

    def test_zero_weight_channel_is_ignored(self):
        channels = [
            ChannelResult.from_scores(RetrievalChannel.TEXT_TO_TEXT, [("a", 0.5)]),
            ChannelResult.from_scores(RetrievalChannel.LEXICAL, [("spam", 0.99)]),
        ]
        results = fuse(
            channels,
            {RetrievalChannel.TEXT_TO_TEXT: 1.0, RetrievalChannel.LEXICAL: 0.0},
        )
        assert {r.product_id for r in results} == {"a"}

    def test_rrf_uses_ranks_not_magnitudes(self):
        """RRF must be invariant to a monotone rescaling of the scores."""
        raw = [("a", 0.99), ("b", 0.98), ("c", 0.01)]
        squashed = [("a", 0.30), ("b", 0.20), ("c", 0.10)]
        weights = {RetrievalChannel.TEXT_TO_TEXT: 1.0}

        first = fuse(
            [ChannelResult.from_scores(RetrievalChannel.TEXT_TO_TEXT, raw)],
            weights,
            strategy=FusionStrategy.RRF,
        )
        second = fuse(
            [ChannelResult.from_scores(RetrievalChannel.TEXT_TO_TEXT, squashed)],
            weights,
            strategy=FusionStrategy.RRF,
        )
        assert [r.product_id for r in first] == [r.product_id for r in second]
        assert [r.final_score for r in first] == [r.final_score for r in second]

    def test_rrf_score_formula(self):
        channels = [
            ChannelResult.from_scores(RetrievalChannel.TEXT_TO_TEXT, [("a", 0.9), ("b", 0.4)])
        ]
        results = fuse(
            channels,
            {RetrievalChannel.TEXT_TO_TEXT: 1.0},
            strategy=FusionStrategy.RRF,
            rrf_k=60,
        )
        scores = {r.product_id: r.final_score for r in results}
        assert scores["a"] == pytest.approx(1.0 / 61)
        assert scores["b"] == pytest.approx(1.0 / 62)

    def test_rrf_k_controls_rank_decay(self):
        """A smaller k sharpens the advantage of the top rank."""
        channels = [
            ChannelResult.from_scores(RetrievalChannel.TEXT_TO_TEXT, [("a", 0.9), ("b", 0.4)])
        ]
        weights = {RetrievalChannel.TEXT_TO_TEXT: 1.0}
        sharp = fuse(channels, weights, strategy=FusionStrategy.RRF, rrf_k=1)
        flat = fuse(channels, weights, strategy=FusionStrategy.RRF, rrf_k=1000)
        sharp_gap = sharp[0].final_score - sharp[1].final_score
        flat_gap = flat[0].final_score - flat[1].final_score
        assert sharp_gap > flat_gap

    def test_modality_similarities_are_reported_on_the_raw_scale(self):
        """image_similarity/text_similarity must stay interpretable as cosines."""
        channels = [
            ChannelResult.from_scores(
                RetrievalChannel.IMAGE_TO_IMAGE, [("a", 0.80), ("z", 0.10)]
            ),
            ChannelResult.from_scores(
                RetrievalChannel.IMAGE_TO_TEXT, [("a", 0.20), ("z", 0.05)]
            ),
            ChannelResult.from_scores(
                RetrievalChannel.TEXT_TO_TEXT, [("a", 0.60), ("z", 0.05)]
            ),
        ]
        weights = {
            RetrievalChannel.IMAGE_TO_IMAGE: 0.35,
            RetrievalChannel.IMAGE_TO_TEXT: 0.15,
            RetrievalChannel.TEXT_TO_TEXT: 0.50,
        }
        result = next(r for r in fuse(channels, weights) if r.product_id == "a")
        # Weighted mean of the image-side raw scores: (0.35*0.8 + 0.15*0.2) / 0.5
        assert result.image_similarity == pytest.approx((0.35 * 0.80 + 0.15 * 0.20) / 0.50)
        assert result.text_similarity == pytest.approx(0.60)

    def test_lexical_score_is_reported_separately(self):
        channels = [
            ChannelResult.from_scores(RetrievalChannel.TEXT_TO_TEXT, [("a", 0.5)]),
            ChannelResult.from_scores(RetrievalChannel.LEXICAL, [("a", 0.75)]),
        ]
        result = fuse(
            channels,
            {RetrievalChannel.TEXT_TO_TEXT: 0.8, RetrievalChannel.LEXICAL: 0.2},
        )[0]
        assert result.lexical_score == pytest.approx(0.75)
        assert result.image_similarity is None

    def test_explain_false_omits_channel_detail(self):
        channels = [
            ChannelResult.from_scores(RetrievalChannel.TEXT_TO_TEXT, [("a", 0.5), ("b", 0.2)])
        ]
        weights = {RetrievalChannel.TEXT_TO_TEXT: 1.0}
        with_detail = fuse(channels, weights, explain=True)
        without = fuse(channels, weights, explain=False)
        assert with_detail[0].channel_scores
        assert not without[0].channel_scores
        # Ranking must be identical either way.
        assert [r.product_id for r in with_detail] == [r.product_id for r in without]

    def test_channel_detail_records_rank_and_weight(self):
        channels = [
            ChannelResult.from_scores(
                RetrievalChannel.TEXT_TO_TEXT, [("a", 0.9), ("b", 0.5), ("c", 0.1)]
            )
        ]
        results = fuse(channels, {RetrievalChannel.TEXT_TO_TEXT: 1.0})
        detail = next(r for r in results if r.product_id == "b").channel_scores[0]
        assert detail.channel is RetrievalChannel.TEXT_TO_TEXT
        assert detail.rank == 2
        assert detail.raw == pytest.approx(0.5)
        assert detail.weight == pytest.approx(1.0)

    def test_empty_channels_produce_no_results(self):
        assert fuse([], {}) == []
        assert fuse([ChannelResult.from_scores(RetrievalChannel.TEXT_TO_TEXT, [])], {}) == []

    def test_final_score_within_unit_interval_for_normalised_weighted_sum(self):
        """With normalised channels and weights summing to 1, scores stay in [0, 1]."""
        channels = [
            ChannelResult.from_scores(
                RetrievalChannel.TEXT_TO_TEXT, [(f"p{i}", 0.9 - i * 0.1) for i in range(5)]
            ),
            ChannelResult.from_scores(
                RetrievalChannel.TEXT_TO_IMAGE, [(f"p{i}", 0.1 + i * 0.05) for i in range(5)]
            ),
        ]
        results = fuse(
            channels,
            {
                RetrievalChannel.TEXT_TO_TEXT: 0.7,
                RetrievalChannel.TEXT_TO_IMAGE: 0.3,
            },
        )
        assert all(0.0 <= r.final_score <= 1.0 for r in results)

    def test_channel_result_from_scores_sorts_and_ranks(self):
        """Ranks must reflect score order even if the backend returned them unsorted."""
        result = ChannelResult.from_scores(
            RetrievalChannel.TEXT_TO_TEXT, [("mid", 0.5), ("top", 0.9), ("low", 0.1)]
        )
        assert [c.product_id for c in result.candidates] == ["top", "mid", "low"]
        assert [c.rank for c in result.candidates] == [1, 2, 3]


class TestRetrievalChannelMetadata:
    def test_product_vector_mapping(self):
        assert RetrievalChannel.TEXT_TO_TEXT.product_vector == "text"
        assert RetrievalChannel.TEXT_TO_IMAGE.product_vector == "image"
        assert RetrievalChannel.IMAGE_TO_IMAGE.product_vector == "image"
        assert RetrievalChannel.IMAGE_TO_TEXT.product_vector == "text"
        assert RetrievalChannel.LEXICAL.product_vector is None

    def test_side_classification(self):
        assert RetrievalChannel.IMAGE_TO_TEXT.is_image_side
        assert not RetrievalChannel.IMAGE_TO_TEXT.is_text_side
        assert RetrievalChannel.TEXT_TO_IMAGE.is_text_side
        assert not RetrievalChannel.TEXT_TO_IMAGE.is_image_side
        # Fused channels belong to neither side, which is why embedding fusion
        # cannot report per-modality similarity.
        assert not RetrievalChannel.FUSED_TO_IMAGE.is_image_side
        assert not RetrievalChannel.FUSED_TO_IMAGE.is_text_side

    def test_every_channel_has_a_vector_mapping(self):
        """Guards against adding a channel and forgetting the mapping."""
        for channel in RetrievalChannel:
            assert channel.product_vector in {"image", "text", None}
            if channel is not RetrievalChannel.LEXICAL:
                assert channel.product_vector is not None


def test_zscore_logistic_is_monotone():
    """Property check: z-score normalisation preserves ordering."""
    raw = [0.05, 0.11, 0.3, 0.31, 0.7, 0.95]
    normalised = normalise_scores(raw, ScoreNormalization.ZSCORE)
    assert normalised == sorted(normalised)
    assert all(math.isfinite(value) for value in normalised)

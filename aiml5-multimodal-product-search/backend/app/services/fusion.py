"""Multi-channel score fusion and ranking.

Why this layer exists
---------------------
Retrieval produces several independent, *incomparable* candidate lists. Measured
on the sample catalogue with ``openai/clip-vit-base-patch32``
(``python scripts/verify_embedding_space.py``):

===================  ================
channel pairing      mean cosine
===================  ================
image -> image        0.7305
text  -> text         0.4499
image -> text         0.3124
===================  ================

A weighted sum of these raw numbers is dominated by whichever channel occupies the
higher range, so the configured weights do not mean what they say: at
``image_weight = text_weight = 0.5`` the same-modality channel effectively decides
the ranking. Two independent defences are provided:

1. **Per-channel normalisation** (:class:`ScoreNormalization`) rescales every
   channel onto a common range over the retrieved candidate pool before weights
   are applied, so the weights behave as documented.
2. **Reciprocal rank fusion** discards magnitudes entirely and combines *ranks*,
   which is scale-free by construction and the usual choice when channel score
   distributions are unknown or unstable.

Both are offered because neither is universally better, and
``scripts/run_experiments.py`` measures which wins rather than asserting it. What
the measurements showed on this catalogue, recorded here so the code is not
oversold:

* Normalisation helps, but modestly - ``zscore`` nDCG@10 0.668 vs 0.655 for
  ``none``. Its main value is making the weights interpretable, not a large
  quality jump.
* Score-level ``weighted_sum`` beats single-vector ``embedding_fusion``
  (nDCG@10 0.597 vs 0.574) **only when compared at a matched image share**. An
  unmatched comparison reversed the ordering, which is why the experiment script
  now pins the image share across strategies.
* ``embedding_fusion`` achieved the best MRR (0.832 vs 0.760), i.e. it more often
  places a relevant item first, while ranking the rest of the list less well.

This module is deliberately pure: it performs no I/O and knows nothing about
Qdrant or the database, which makes the ranking arithmetic directly unit
testable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.core.config import FusionStrategy, ScoreNormalization
from app.schemas.search import (
    ChannelScore,
    RetrievalChannel,
    ScoreBreakdown,
    SearchMode,
)

#: Below this spread a min-max rescale would amplify numerical noise into
#: meaningless ranking differences, so the channel is treated as uninformative
#: and every candidate receives the same normalised score.
_MINMAX_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class ChannelCandidate:
    """One product's score within a single retrieval channel."""

    product_id: str
    score: float
    rank: int


@dataclass(slots=True)
class ChannelResult:
    """A single channel's candidate list, ordered best-first."""

    channel: RetrievalChannel
    candidates: list[ChannelCandidate] = field(default_factory=list)

    @classmethod
    def from_scores(
        cls, channel: RetrievalChannel, scored: Sequence[tuple[str, float]]
    ) -> ChannelResult:
        """Build a channel result from ``(product_id, score)`` pairs.

        Pairs are sorted descending by score so ranks are always consistent,
        regardless of the order the retrieval backend returned them in.
        """
        ordered = sorted(scored, key=lambda pair: pair[1], reverse=True)
        return cls(
            channel=channel,
            candidates=[
                ChannelCandidate(product_id=pid, score=float(score), rank=index)
                for index, (pid, score) in enumerate(ordered, start=1)
            ],
        )


@dataclass(slots=True)
class FusedResult:
    """A product with its fused score and per-channel provenance."""

    product_id: str
    final_score: float
    image_similarity: float | None = None
    text_similarity: float | None = None
    lexical_score: float | None = None
    channel_scores: list[ChannelScore] = field(default_factory=list)

    def to_breakdown(
        self, strategy: FusionStrategy, normalization: ScoreNormalization
    ) -> ScoreBreakdown:
        """Render this result's provenance as an API schema object."""
        return ScoreBreakdown(
            final_score=self.final_score,
            image_similarity=self.image_similarity,
            text_similarity=self.text_similarity,
            lexical_score=self.lexical_score,
            channels=self.channel_scores,
            strategy=strategy,
            normalization=normalization,
        )


# --------------------------------------------------------------------- weights
def resolve_channel_weights(
    *,
    mode: SearchMode,
    strategy: FusionStrategy,
    image_weight: float,
    text_weight: float,
    lexical_weight: float,
    cross_modal_weight: float,
    has_text: bool,
    has_image: bool,
) -> dict[RetrievalChannel, float]:
    """Distribute modality weights across concrete retrieval channels.

    Two-level scheme. The modality-level split (``image_weight`` /
    ``text_weight``) is what a user reasons about. Within each modality,
    ``cross_modal_weight`` decides how much trust goes to the cross-modal channel
    (a text query scored against a product *image* vector) versus the
    same-modality channel.

    Weights are renormalised to sum to 1, so a final score stays on the same
    scale whichever channels happen to be active. Without this, text-only search
    would produce systematically smaller scores than multimodal search purely
    because fewer channels contributed.

    Returns:
        Mapping of active channel to weight, summing to 1.0 (empty only if no
        query modality was supplied).
    """
    weights: dict[RetrievalChannel, float] = {}

    if strategy is FusionStrategy.EMBEDDING_FUSION and has_image and has_text:
        # A single blended query vector is searched against both product
        # vectors; there is no longer a per-modality attribution to make.
        weights[RetrievalChannel.FUSED_TO_IMAGE] = 1.0 - cross_modal_weight
        weights[RetrievalChannel.FUSED_TO_TEXT] = cross_modal_weight
    else:
        if has_image:
            side = image_weight if has_text else 1.0
            if side > 0:
                weights[RetrievalChannel.IMAGE_TO_IMAGE] = side * (1.0 - cross_modal_weight)
                weights[RetrievalChannel.IMAGE_TO_TEXT] = side * cross_modal_weight
        if has_text:
            side = text_weight if has_image else 1.0
            if side > 0:
                weights[RetrievalChannel.TEXT_TO_TEXT] = side * (1.0 - cross_modal_weight)
                weights[RetrievalChannel.TEXT_TO_IMAGE] = side * cross_modal_weight

    if has_text and lexical_weight > 0:
        weights[RetrievalChannel.LEXICAL] = lexical_weight

    weights = {channel: w for channel, w in weights.items() if w > 0}
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {channel: w / total for channel, w in weights.items()}


# --------------------------------------------------------------- normalisation
def normalise_scores(scores: Sequence[float], method: ScoreNormalization) -> list[float]:
    """Rescale one channel's scores onto a comparable range.

    * ``NONE`` - passthrough. Only sensible when every active channel is the same
      modality pairing, or with RRF where magnitudes are ignored.
    * ``MINMAX`` - linear rescale of the retrieved pool onto ``[0, 1]``. Adapts to
      each query, which is what makes cross-modal and same-modality channels
      comparable. The trade-off is that it is *relative*: a pool of uniformly
      poor matches is stretched to fill the range just like a pool of good ones,
      so normalised scores rank well but do not indicate absolute quality. Raw
      cosines are therefore always reported alongside.
    * ``ZSCORE`` - standardise, then squash through a logistic so the result stays
      in ``[0, 1]``. More robust than min-max to a single outlier dominating the
      spread, at the cost of compressing genuinely large gaps.
    """
    if not scores:
        return []
    values = [float(s) for s in scores]

    if method is ScoreNormalization.NONE:
        return values

    if method is ScoreNormalization.MINMAX:
        lowest, highest = min(values), max(values)
        spread = highest - lowest
        if spread < _MINMAX_EPSILON:
            return [1.0] * len(values)
        return [(v - lowest) / spread for v in values]

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    stdev = math.sqrt(variance)
    if stdev < _MINMAX_EPSILON:
        return [0.5] * len(values)
    return [1.0 / (1.0 + math.exp(-((v - mean) / stdev))) for v in values]


# ---------------------------------------------------------------------- fusion
def fuse(
    channel_results: Sequence[ChannelResult],
    weights: Mapping[RetrievalChannel, float],
    *,
    strategy: FusionStrategy = FusionStrategy.WEIGHTED_SUM,
    normalization: ScoreNormalization = ScoreNormalization.MINMAX,
    rrf_k: int = 60,
    explain: bool = True,
) -> list[FusedResult]:
    """Combine per-channel candidate lists into one ranked list.

    Candidates missing from a channel contribute nothing for that channel. With
    min-max normalisation this is equivalent to scoring them at the bottom of the
    retrieved pool, which is the intended pessimistic treatment: a product the
    channel did not retrieve should not be rewarded for its absence.

    Args:
        channel_results: Candidate lists, one per active channel.
        weights: Channel weights, normally summing to 1.
        strategy: ``WEIGHTED_SUM``/``EMBEDDING_FUSION`` combine (normalised)
            scores; ``RRF`` combines ranks and ignores ``normalization``.
        normalization: Per-channel score normalisation.
        rrf_k: RRF rank constant. Larger values flatten the contribution of top
            ranks; 60 is the value from the original Cormack et al. formulation.
        explain: When false, per-channel provenance is not assembled.

    Returns:
        Results ordered by descending fused score, ties broken by product id for
        deterministic output.
    """
    use_ranks = strategy is FusionStrategy.RRF

    totals: dict[str, float] = {}
    per_channel: dict[str, list[ChannelScore]] = {}
    image_numerator: dict[str, float] = {}
    image_denominator: dict[str, float] = {}
    text_numerator: dict[str, float] = {}
    text_denominator: dict[str, float] = {}
    lexical: dict[str, float] = {}

    for result in channel_results:
        weight = float(weights.get(result.channel, 0.0))
        if weight <= 0 or not result.candidates:
            continue

        raw_scores = [candidate.score for candidate in result.candidates]
        normalised = normalise_scores(raw_scores, normalization)

        for candidate, norm in zip(result.candidates, normalised, strict=True):
            contribution = weight / (rrf_k + candidate.rank) if use_ranks else weight * norm
            totals[candidate.product_id] = totals.get(candidate.product_id, 0.0) + contribution

            # Track modality-level similarity on the raw cosine scale, so the
            # reported numbers stay interpretable rather than query-relative.
            if result.channel.is_image_side:
                image_numerator[candidate.product_id] = (
                    image_numerator.get(candidate.product_id, 0.0) + weight * candidate.score
                )
                image_denominator[candidate.product_id] = (
                    image_denominator.get(candidate.product_id, 0.0) + weight
                )
            elif result.channel.is_text_side:
                text_numerator[candidate.product_id] = (
                    text_numerator.get(candidate.product_id, 0.0) + weight * candidate.score
                )
                text_denominator[candidate.product_id] = (
                    text_denominator.get(candidate.product_id, 0.0) + weight
                )
            elif result.channel is RetrievalChannel.LEXICAL:
                lexical[candidate.product_id] = candidate.score

            if explain:
                per_channel.setdefault(candidate.product_id, []).append(
                    ChannelScore(
                        channel=result.channel,
                        raw=round(candidate.score, 6),
                        normalized=round(norm, 6),
                        weight=round(weight, 6),
                        rank=candidate.rank,
                    )
                )

    fused = [
        FusedResult(
            product_id=product_id,
            final_score=score,
            image_similarity=(
                image_numerator[product_id] / image_denominator[product_id]
                if image_denominator.get(product_id)
                else None
            ),
            text_similarity=(
                text_numerator[product_id] / text_denominator[product_id]
                if text_denominator.get(product_id)
                else None
            ),
            lexical_score=lexical.get(product_id),
            channel_scores=per_channel.get(product_id, []),
        )
        for product_id, score in totals.items()
    ]
    fused.sort(key=lambda item: (-item.final_score, item.product_id))
    return fused

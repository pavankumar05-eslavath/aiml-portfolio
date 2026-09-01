# Evaluation methodology

How retrieval quality is measured, what the numbers mean, and — importantly — what
they do **not** mean.

Generated reports live in
[`../evaluation/results/`](../evaluation/results/). Everything below is measured
output from `scripts/evaluate.py`, `scripts/run_experiments.py` and
`scripts/verify_embedding_space.py`. Nothing here is estimated.

## Contents

- [Setup](#setup)
- [How relevance is defined](#how-relevance-is-defined)
- [Metrics](#metrics)
- [Results: search modes](#results-search-modes)
- [Does fusion earn its complexity?](#does-fusion-earn-its-complexity)
- [Experiments](#experiments)
- [How the defaults were chosen](#how-the-defaults-were-chosen)
- [Limitations](#limitations)
- [Reproducing](#reproducing)

## Setup

| | |
|---|---|
| Catalogue | 6,000 products, stratified across 141 article types |
| Model | `openai/clip-vit-base-patch32` (512-d), CPU |
| Queries | 46 (16 text, 10 image, 20 multimodal) |
| Relevant set size | min 10, median 30, max 73 |
| Retrieval depth | `top_k = 20` |
| Cut-offs | 1, 5, 10, 20 |

The harness runs the real `SearchService` in-process — the same code path the HTTP
API uses. That is what allows these numbers to describe the running system rather
than a parallel evaluation-only implementation.

## How relevance is defined

Hand-labelling enough query/product pairs to evaluate 6,000 products is not
feasible for a project this size. Relevance is therefore defined **declaratively**,
as a predicate over the attributes the source dataset actually provides:

```json
{
  "id": "t01",
  "type": "text",
  "query": "white trainers for running",
  "relevance": { "subcategories": ["Sports Shoes"], "colours": ["White"] }
}
```

resolved against the live catalogue at run time. This makes the ground truth:

* **auditable** — the rule is written down, not a list of ids someone chose;
* **reproducible** — it re-derives correctly for any catalogue size or seed;
* **honest about its limits** — see [Limitations](#limitations).

### Relevance rules per query type

| Type | Count | Relevant set |
|---|---|---|
| `text` | 16 | Article type **and** the attribute the query names (colour / gender) |
| `image` | 10 | Same article type as the query product. Colour deliberately *not* required, so this measures category recognition rather than colour fidelity. |
| `multimodal` / `contradiction` | 10 | Article type **and the colour named in the text** — which the query image does *not* have |
| `multimodal` / `agreement` | 10 | Article type **and** the attributes on which image and text agree |

### Why multimodal queries are split into two groups

This split was added after an initial version of the benchmark contained only
contradiction-style queries and consequently recommended a text-heavy weight that
would have been wrong for half of real usage.

* **Contradiction** — *photo of a white shoe* + *"but in black"*. The relevant set
  requires the attribute from the **text**, so a system that quietly ignores the
  text cannot score. This is the discriminating case.
* **Agreement** — *photo of a black handbag* + *"a black handbag"*. The text
  reinforces the image.

They have opposite optimal weightings, so reporting a single multimodal average
over both would make any recommended weight an artefact of the query mix. They are
swept and reported separately.

### Guards against flattering the system

* Queries are phrased in natural shopper language (*"dark glasses to keep the sun
  out of my eyes"*), not in dataset vocabulary, to avoid lexical overlap with the
  templated product documents.
* The query product is always excluded from its own relevant set.
* Products that are not indexed are excluded from relevant sets — ground truth
  cannot contain items search is unable to return.
* Queries with fewer than two relevant products are reported as **skipped** rather
  than scored, since metrics over one or two items are noise.
* An unconstrained relevance rule (one matching everything) is rejected at load
  time; it would report a meaninglessly perfect score.

## Metrics

Binary relevance, macro-averaged (each query weighted equally, so a query with a
large relevant set cannot dominate).

| Metric | Definition | Note |
|---|---|---|
| **P@k** | relevant in top *k* / *k* | Denominator is *k* even if fewer results returned |
| **R@k** | relevant in top *k* / total relevant | **Bounded by `k / |relevant|`** — see below |
| **MRR** | 1 / rank of first relevant result | Sensitive to the top of the list |
| **nDCG@k** | DCG / ideal DCG, log₂ discount | Position-aware; the metric to compare on |
| **MAP** | mean average precision | |

> **Read `R@k` carefully.** Relevant sets here hold 10–73 products, so with
> `k = 10` recall cannot exceed ~0.33 even for perfect retrieval. R@10 = 0.23 is
> therefore closer to its ceiling than it looks. Compare configurations on
> **nDCG@10** and **MRR**; treat recall as relative only.

Metric implementations are unit-tested against hand-worked examples
(`backend/tests/test_evaluation.py`) — if they were wrong, every number in the
reports would be wrong too.

## Results: search modes

| Mode | Queries | nDCG@10 | MRR | P@10 | R@10 | Median latency |
|---|---|---|---|---|---|---|
| Text-only | 16 | 0.677 | 0.891 | 0.637 | 0.233 | ~28 ms |
| Image-only | 10 | **0.878** | **1.000** | **0.860** | 0.118 | ~74 ms |
| Multimodal | 20 | 0.617 | 0.797 | 0.580 | 0.287 | ~100 ms |
| All combined | 46 | 0.694 | 0.874 | 0.661 | 0.231 | — |

**These rows are not comparable to one another.** Each mode is evaluated on the
query set designed for it, and those sets pose different problems — image-only
scores highest partly because "find more of the same article type from a photo" is
an easier task than satisfying a natural-language constraint. Image-only MRR of
1.000 means the first result was relevant for all 10 queries, which is a genuine
strength of visual similarity here but not evidence that image search is "better".

The comparison that actually controls for query difficulty is the ablation.

## Does fusion earn its complexity?

The **same** 20 multimodal queries, run through different channels:

| Channels used | nDCG@10 | MRR | P@10 |
|---|---|---|---|
| Image only (`image_weight=1`) | 0.395 | 0.527 | 0.385 |
| Text only (`text_weight=1`) | 0.595 | 0.804 | 0.550 |
| **Both (fused)** | **0.617** | 0.797 | **0.580** |

Fusion beats either modality alone. Per group, at each group's own optimal
weighting:

| Query group | Image only | Text only | Fused (best) |
|---|---|---|---|
| Contradiction | 0.056 | 0.405 | **0.429** (`image_weight=0.1`) |
| Agreement | 0.735 | 0.786 | **0.831** (`image_weight=0.5`) |

Fusion wins in **both** regimes. The contradiction row also validates the
benchmark: a system ignoring the text scores 0.056, near the floor, confirming
those queries genuinely require both modalities.

## Experiments

Full tables in
[`../evaluation/results/experiments.md`](../evaluation/results/experiments.md).

### 1. Which channels carry the signal?

| Query type | Same-modality | Cross-modal |
|---|---|---|
| Text queries | `text→text` **0.572** | `text→image` 0.514 |
| Image queries | `image→image` 0.796 | `image→text` **0.818** |

The surprise: for image queries the *cross-modal* channel wins. Product text names
the article type explicitly, which is what the ground truth rewards. This is why
`CROSS_MODAL_WEIGHT` defaults to a non-zero value rather than the 0 intuition
suggests.

### 2. Image/text weighting — the key result

| `IMAGE_WEIGHT` | Contradiction nDCG@10 | Agreement nDCG@10 |
|---|---|---|
| 0.0 | 0.405 | 0.786 |
| **0.1** | **0.429** | 0.805 |
| 0.2 | 0.417 | 0.817 |
| 0.3 | 0.347 | 0.830 |
| 0.5 | 0.279 | **0.831** |
| 0.7 | 0.168 | 0.815 |
| 1.0 | 0.056 | 0.735 |

Contradiction degrades monotonically as image weight rises — 0.429 → 0.056, a 7.7×
collapse. Agreement is far flatter and peaks at 0.5. **The optima differ
five-fold**, which is the argument for exposing the weight per request rather than
picking a global constant.

### 3. Cross-modal weight

Swept on single-modality queries, where the cross-modal channel is pure added
evidence rather than a source of conflict. Best: **0.30** (nDCG@10 0.754).

### 4. Score fusion vs embedding fusion — at a *matched* image share

| Strategy | nDCG@10 | MRR |
|---|---|---|
| `weighted_sum` (image share 0.3) | **0.598** | 0.760 |
| `rrf` (image share 0.3) | 0.575 | 0.752 |
| `embedding_fusion` (α = 0.3) | 0.574 | **0.832** |

**This experiment initially produced the wrong conclusion, and the fix is worth
recording.** The first version compared `weighted_sum` at its default 50/50
weighting against `embedding_fusion` at a text-heavy α = 0.3 — and "found" that
embedding fusion won. That was an artefact of the unequal weighting, not a property
of the fusion mechanism. Pinning the image share across all strategies reversed the
ordering. The experiment script now enforces the matched comparison.

Note also that `embedding_fusion` has the **best MRR** — it more often places a
relevant item first while ordering the rest less well. Score-level fusion remains
the default because it wins on overall ranking quality *and* preserves per-modality
attribution, but the trade-off is real rather than one-sided.

### 5. Score normalisation

| Normalisation | nDCG@10 |
|---|---|
| **z-score** | **0.682** |
| none | 0.655–0.677 |
| min-max | 0.646 |

Real but modest. Reported plainly because it would be easy to overstate: the main
value of normalisation is that it makes the configured weights *mean what they
say*, not that it transforms quality.

### 6. Lexical channel

Best at **0.0** (channel off) under z-score normalisation: 0.677 with it off
versus 0.669 at weight 0.2. It also roughly tripled median latency, because the
SQLite fallback scores candidates in Python. Under min-max normalisation it *did*
help slightly (0.639 vs 0.631) — the two knobs interact.

Measured against the SQLite fallback. PostgreSQL full-text search is a genuinely
different (stemmed, term-weighted) implementation and is verified working, so this
is worth re-measuring on PostgreSQL before drawing a general conclusion.

### 7. Retrieval depth

| `top_k` | nDCG@10 | Median latency |
|---|---|---|
| 20 | 0.681 | 45.6 ms |
| 50 | ~0.690 | ~52 ms |
| 100 | ~0.698 | 65.4 ms |

A genuine trade-off rather than a free win. Depths below `max(cut-off)` cannot be
measured — scoring needs at least that much depth — so the sweep starts at 20.

## How the defaults were chosen

Every ranking default in `.env.example` is the measured optimum, not a guess:

| Setting | Default | Evidence |
|---|---|---|
| `FUSION_STRATEGY` | `weighted_sum` | Experiment 4, matched image share |
| `SCORE_NORMALIZATION` | `zscore` | Experiment 5 |
| `IMAGE_WEIGHT` / `TEXT_WEIGHT` | 0.2 / 0.8 | Experiment 2, mean-optimum across both groups |
| `CROSS_MODAL_WEIGHT` | 0.3 | Experiment 3 |
| `LEXICAL_WEIGHT` | 0.0 | Experiment 6 |

Three of these contradicted my initial choices. `SCORE_NORMALIZATION` started as
min-max, `IMAGE_WEIGHT` as 0.5, and `LEXICAL_WEIGHT` was going to be enabled at
0.2 after an early measurement favoured it. Measurement overruled all three.

The `IMAGE_WEIGHT` default deserves a caveat: 0.2 maximises the mean across a
**50/50 mix** of contradiction and agreement queries. That mix is a property of the
benchmark, not of the world. A deployment whose users mostly do agreement-style
searches should raise it.

## Model alignment

Multimodal fusion is only valid if both towers project into a comparable space.
`scripts/verify_embedding_space.py` measures this on the actual catalogue and
exits non-zero on failure, so it can gate CI. On the 1,000-product sample
(48 products probed):

| Measurement | Value | Meaning |
|---|---|---|
| Cross-modal top-1 (templated) | **0.812** | Own document is the best match for own image |
| Cross-modal top-1 (raw name) | 0.771 | The document template is worth +0.042 |
| Matched pair similarity | 0.3124 | |
| Mismatched pair similarity | 0.1794 | Separation +0.133 → shared space confirmed |
| image ↔ image | 0.7305 | |
| text ↔ text | 0.4499 | |
| **Modality gap** | **+0.4181** | Why per-channel normalisation exists |

Verdict: **PASS**.

## Limitations

Stated plainly, because the numbers are easy to over-read:

1. **Attribute-derived relevance measures attribute agreement, not human
   preference.** A visually excellent match labelled with a different `baseColour`
   counts as a miss; a mediocre one with the right label counts as a hit.
2. **It is generous within a class.** Every black sports shoe is equally relevant,
   so the metrics cannot distinguish good ranking *within* the relevant set — which
   is much of what a shopper perceives as quality.
3. **46 queries is a small benchmark.** Differences of one or two points in
   nDCG@10 are not significant, and no confidence intervals are reported. Only the
   large effects (the 7.7× weight collapse, the fusion ablation) should be treated
   as solid.
4. **Queries and product templates share an author.** Natural-language phrasing
   mitigates lexical overlap but does not eliminate the bias.
5. **60 × 80 source images** bound visual precision. Fine detail is simply absent
   from the input.
6. **Synthetic prices.** Price filtering is tested for correctness, never for
   realism, and no query depends on price.
7. **One model, one catalogue, one language.** No cross-model comparison, no
   multilingual evaluation.
8. **`R@k` ceilings** make absolute recall values misleading, as described above.

These numbers are meaningful for **comparing configurations of this system**,
which is exactly what the experiments use them for. They are not comparable to
published retrieval benchmarks.

## Reproducing

```bash
python scripts/download_dataset.py --limit 6000 --out data/eval
python scripts/index_catalog.py --source data/eval/products.csv   # ~5 min, CPU
python scripts/evaluate.py --per-query
python scripts/run_experiments.py
python scripts/verify_embedding_space.py
```

To evaluate against a different model, change `MODEL_NAME`, re-index with
`--force --recreate`, and re-run. The benchmark and harness are model-agnostic.

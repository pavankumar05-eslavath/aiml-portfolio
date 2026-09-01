# AIML-5 — Findings

Multimodal (image + text) product retrieval over a 6,000-product fashion catalogue.
Every number below is produced by committed code; the commands that regenerate each
one are given. Methodology and its limits: [`docs/evaluation.md`](docs/evaluation.md).

---

## 1. The claim

**Fusing image and text evidence beats either modality alone — but the weighting that
makes it work is not a constant. It varies five-fold with the user's intent, so any
single global value is wrong for half of real queries.**

Supporting numbers, all on the same 20 multimodal queries so query difficulty is held
fixed:

| Channels used | nDCG@10 | MRR | P@10 |
|---|---|---|---|
| Image only | 0.395 | 0.527 | 0.385 |
| Text only | 0.595 | 0.804 | 0.550 |
| **Both, fused** | **0.617** | 0.797 | **0.580** |

Fusion wins. But split those queries by intent and the picture changes:

| Query style | Best `IMAGE_WEIGHT` | nDCG@10 at its optimum | at the other's optimum |
|---|---|---|---|
| **Contradiction** — photo of a white shoe + *"but in black"* | **0.1** | 0.429 | 0.279 (at 0.5) |
| **Agreement** — photo of a black bag + *"a black handbag"* | **0.5** | 0.831 | 0.805 (at 0.1) |

Contradiction queries degrade **monotonically and steeply** as image weight rises —
0.429 → 0.056, a **7.7× collapse**. The mechanism is not subtle: when the text asks to
*change* an attribute, weighting the image pulls results back toward the attribute the
user just rejected.

Both groups still beat their single-modality baselines at their own optimum
(contradiction 0.429 vs 0.405 text-only and 0.056 image-only; agreement 0.831 vs 0.786
and 0.735), so fusion is justified in both regimes. What is not justified is a fixed
weight. The system therefore exposes `image_weight` per request, and the shipped
default (0.2) is stated as what it is: the mean-optimum over a 50/50 benchmark mix, not
a universal value.

```bash
make evaluate       # the ablation table
make experiments    # the per-intent sweep
```

## 2. The baseline, stated before the model

Three baselines, all measured, none chosen to lose:

1. **Text-only retrieval** — the obvious thing to do without any of this machinery.
   nDCG@10 **0.595** on the multimodal query set.
2. **Image-only retrieval** — nDCG@10 **0.395**.
3. **Naive single-vector fusion** — average the image and text query embeddings into one
   vector and do a single lookup (`FUSION_STRATEGY=embedding_fusion`). This is what most
   implementations of "multimodal search" actually do.

Against baseline 3 the result is genuinely close, and the comparison had to be repaired
before it meant anything — see §4.

## 3. A negative result worth reporting

**Score normalisation matters much less than the design implies.**

The ranking layer normalises each retrieval channel before combining, because the
channels are on visibly different scales (measured, `make verify-model`):

| Cosine pairing | Mean |
|---|---|
| image ↔ image | 0.7305 |
| text ↔ text | 0.4499 |
| image ↔ text | 0.3124 |

A 0.73-scale channel and a 0.31-scale channel summed raw means the configured weights do
not mean what they say. That reasoning is sound. The measured payoff is small:

| Normalisation | nDCG@10 |
|---|---|
| z-score | **0.682** |
| none | 0.655–0.677 |
| min-max | 0.646 |

**+0.005 to +0.027 over doing nothing.** It is kept on because it makes the weights
*interpretable* — a user moving the slider gets the effect the label promises — not
because it transforms retrieval quality. Reported at its true size rather than as a
headline.

The **lexical (keyword) channel** fared worse: measured best at weight **0.0**, i.e.
switched off (0.677 with it off vs 0.669 at 0.2), while roughly tripling median latency.
It ships disabled. Under min-max normalisation it had helped slightly — the two knobs
interact, which is itself a caution against tuning them independently.

## 4. The bug in my own experiment

The strategy comparison initially concluded that **naive single-vector fusion beat the
score-level fusion this project is built around.** That conclusion was wrong, and the
reason is worth recording.

The first run compared `weighted_sum` at its **default 50/50** image/text weighting
against `embedding_fusion` at a **text-heavy α = 0.3**. Since the benchmark is
half contradiction queries — which strongly prefer text — the second configuration was
handed an advantage that had nothing to do with its fusion mechanism.

Pinning the image share equal across strategies reversed the ordering:

| Strategy | nDCG@10 | MRR |
|---|---|---|
| `weighted_sum` (image share 0.3) | **0.598** | 0.760 |
| `rrf` (image share 0.3) | 0.575 | 0.752 |
| `embedding_fusion` (α = 0.3) | 0.574 | **0.832** |

`scripts/run_experiments.py` now enforces the matched share, so the mistake cannot
recur silently.

Note the split decision: `embedding_fusion` has the **best MRR** (0.832 vs 0.760). It
more often puts a relevant item first, while ordering the rest less well. Score-level
fusion remains the default because it wins on overall ranking quality *and* preserves
per-modality attribution — but this is a trade-off, not a clean win.

## 5. Verified numbers reference

| Quantity | Value | Reproduce with |
|---|---|---|
| Text search, nDCG@10 / MRR | 0.677 / 0.891 | `make evaluate` |
| Image search, nDCG@10 / MRR | 0.878 / 1.000 | `make evaluate` |
| Multimodal, nDCG@10 / MRR | 0.617 / 0.797 | `make evaluate` |
| All 46 queries, nDCG@10 | 0.694 | `make evaluate` |
| Cross-modal top-1, templated text | 0.812 | `make verify-model` |
| Cross-modal top-1, raw SKU name | 0.771 | `make verify-model` |
| Modality gap (image-image − image-text) | +0.4181 | `make verify-model` |
| Indexing throughput (CPU) | 19–21 products/s | `make index` |
| Re-index with nothing changed | 0.1 s (vs 46.8 s cold) | `make index` twice |
| Text search latency, end to end | ~28 ms | `make evaluate` |
| Tests / coverage | 444 / 90% | `make test-all`, `make cov` |

Image-only MRR of 1.000 means the first result was relevant for all 10 image queries.
That is a real strength of visual similarity on this catalogue, **not** evidence that
image search is better than text search — the two are scored on different query sets.
See the caveats.

## 6. Caveats — what this does not establish

1. **Relevance is attribute-derived, not human-judged.** A query's relevant set is a
   predicate over product attributes (article type, colour, gender). This measures
   attribute agreement; a visually excellent match labelled with a different
   `baseColour` counts as a miss, and a mediocre one with the right label counts as a
   hit. No human ever rated these results.
2. **The metrics cannot see ranking quality *within* a class.** Every black sports shoe
   is equally relevant by construction, which is much of what a shopper actually
   perceives as quality.
3. **`R@k` is capped at `k / |relevant|`.** Relevant sets hold 10–73 products, so R@10
   cannot exceed ~0.33 however good retrieval is. R@10 = 0.23 is near its ceiling, not
   poor. Compare on nDCG@10 and MRR.
4. **46 queries is a small benchmark.** One- or two-point nDCG differences are not
   significant and no confidence intervals are reported. Only the large effects — the
   7.7× weight collapse, the fusion ablation — should be treated as solid.
5. **The 0.2 default weight is an artefact of the query mix.** The benchmark is 50/50
   contradiction/agreement. That ratio is a property of the benchmark, not of the world.
6. **Queries and product text share an author.** Natural-language phrasing mitigates
   lexical overlap with the templated documents; it does not eliminate the bias.
7. **60 × 80 source images** bound visual precision — texture, stitching and logos are
   simply absent from the input.
8. **Prices are synthetic.** The source dataset has none. Price filtering is tested for
   *correctness*, never realism, and no evaluation query depends on price.
9. **One model, one catalogue, English only.** No cross-model or multilingual comparison.
10. **The lexical result is SQLite-specific.** It was measured against the fallback
    implementation; PostgreSQL full-text search is a materially different (stemmed,
    term-weighted) path and is verified working but not re-measured.

## 7. What would have to be measured next

- **Click-through data** would replace the attribute-derived proxy and settle caveats 1
  and 2 — the single highest-value addition.
- **An intent classifier** for contradiction vs agreement. The 5× weight gap is the
  largest effect in the project; even a crude negation/modification detector should beat
  any fixed default.
- **A cross-encoder reranker** over the top ~50 candidates, to model query/product
  interaction rather than approximating it with a weighted sum.
- **Re-measuring the lexical channel on PostgreSQL**, where it is real full-text search.

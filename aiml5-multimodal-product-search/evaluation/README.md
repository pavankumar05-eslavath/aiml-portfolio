# Evaluation

Retrieval quality is **measured**, not asserted. This directory holds the
benchmark query set and the generated reports.

```
evaluation/
├── queries/
│   └── benchmark.json          # 46 queries + attribute-predicate relevance rules
└── results/
    ├── evaluation_report.md    # mode comparison + ablation   (scripts/evaluate.py)
    ├── evaluation_report.json  # same, machine-readable
    ├── experiments.md          # 7 parameter sweeps           (scripts/run_experiments.py)
    ├── experiments.json        # same, machine-readable
    └── embedding_space.json    # model alignment measurements (scripts/verify_embedding_space.py)
```

## Running it

```bash
# Build and index the 6,000-product catalogue the reported numbers use (~6 min on CPU)
python scripts/download_dataset.py --limit 6000 --out data/eval
python scripts/index_catalog.py --source data/eval/products.csv

python scripts/evaluate.py --per-query      # quality report
python scripts/run_experiments.py           # parameter sweeps
python scripts/verify_embedding_space.py    # is the model's shared space real?
```

The harness drives the real `SearchService` in-process — the same code path the
HTTP API uses, minus the network. Running the service directly keeps results
reproducible and lets a sweep override ranking parameters per query without
restarting anything.

## Headline results

46 queries against 6,000 products, `openai/clip-vit-base-patch32`, CPU:

| Mode | Queries | nDCG@10 | MRR | P@10 | R@10 |
|---|---|---|---|---|---|
| Text-only | 16 | 0.677 | 0.891 | 0.637 | 0.233 |
| Image-only | 10 | 0.878 | 1.000 | 0.860 | 0.118 |
| Multimodal | 20 | 0.617 | 0.797 | 0.580 | 0.287 |
| All combined | 46 | 0.694 | 0.874 | 0.661 | 0.231 |

**These rows are not comparable to each other** — each mode is scored on the query
set designed for it, and a text query poses a different problem from an image
query. The comparison that controls for this is the ablation, which runs one fixed
query set through different channels.

## Does fusion actually earn its complexity?

The 20 multimodal queries, run three ways:

| Channels used | nDCG@10 | MRR |
|---|---|---|
| Image only | 0.395 | 0.527 |
| Text only | 0.595 | 0.804 |
| **Both (fused)** | **0.617** | 0.797 |

Fusion beats either modality alone. Full detail, including the per-group weight
sweeps that show *why* a single global weight is the wrong abstraction, is in
[`../docs/evaluation.md`](../docs/evaluation.md).

## Reading the numbers honestly

* **`R@k` is capped by `k / |relevant|`.** Relevant sets here hold 10–73 products,
  so R@10 cannot approach 1.0 no matter how good retrieval is. Compare rows on
  **nDCG@10** and **MRR**; treat recall as relative only.
* **Relevance is attribute-derived**, so it measures attribute agreement rather
  than human taste. A visually perfect match labelled with a different
  `baseColour` counts as a miss.
* **Numbers are for comparing configurations of this system**, which is what the
  experiments use them for. They are not comparable to published benchmarks.

The methodology and its limitations are set out in full in
[`../docs/evaluation.md`](../docs/evaluation.md). Read that before quoting any
figure here.

## The query set

`queries/benchmark.json` defines relevance **declaratively**, as a predicate over
product attributes resolved against the live catalogue at run time:

```json
{
  "id": "t01",
  "type": "text",
  "query": "white trainers for running",
  "relevance": { "subcategories": ["Sports Shoes"], "colours": ["White"] }
}
```

| Group | Count | Relevance rule |
|---|---|---|
| `text` | 16 | Article type **and** the attribute the query names |
| `image` | 10 | Same article type as the query product |
| `multimodal` / `contradiction` | 10 | Article type **and the colour named in the text**, which the query image does *not* have |
| `multimodal` / `agreement` | 10 | Article type **and** the attributes image and text agree on |

The contradiction group is the discriminating one: a system that quietly ignores
the text scores near zero on it (measured: nDCG@10 **0.056** using the image
channel alone).

Queries with fewer than two relevant products in the current catalogue are
reported as *skipped* rather than scored, since metrics over one or two items are
noise.

## Adding queries

Append to `queries/benchmark.json`. Validation is strict — an unknown relevance
field or a rule that matches everything is rejected at load time, and
`backend/tests/test_evaluation.py` asserts the shipped file stays loadable and
internally consistent.

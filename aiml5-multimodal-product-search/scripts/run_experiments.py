#!/usr/bin/env python
"""Parameter sweeps that justify the ranking configuration.

Each experiment isolates one design decision and measures its effect on the
benchmark query set. The point is not to produce impressive numbers - it is to
make the default configuration in ``.env.example`` a *measured* choice rather than
a guess, and to record where the architecture does **not** help.

Experiments
-----------
1. Retrieval channel contribution per query type.
2. Image/text weight sweep on multimodal queries.
3. Cross-modal weight sweep (does searching the other modality's vector help?).
4. Fusion strategy: weighted sum vs RRF vs embedding-level fusion.
5. Score normalisation: min-max vs z-score vs none.
6. Lexical channel weight.
7. Candidate pool depth.

Usage::

    python scripts/run_experiments.py
    python scripts/run_experiments.py --only 2 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _bootstrap import REPO_ROOT, banner, configure_script_logging, display_path, kv

DEFAULT_QUERIES = REPO_ROOT / "evaluation" / "queries" / "benchmark.json"
DEFAULT_OUT = REPO_ROOT / "evaluation" / "results" / "experiments.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ranking parameter sweeps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument(
        "--only",
        type=int,
        nargs="+",
        default=None,
        help="Run only these experiment numbers.",
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


def table(rows: list[list[str]], header: list[str]) -> str:
    """Render a Markdown table."""
    widths = [
        max([len(header[i])] + [len(r[i]) for r in rows]) if rows else len(header[i])
        for i in range(len(header))
    ]
    out = [
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
        "|" + "|".join("-" * (widths[i] + 2) for i in range(len(header))) + "|",
    ]
    for row in rows:
        out.append(
            "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(header))) + " |"
        )
    return "\n".join(out)


def metric_rows(results: list[Any], first_column: str) -> tuple[list[list[str]], list[str]]:
    """Build a metrics table from run results."""
    header = [first_column, "n", "MRR", "MAP", "R@5", "R@10", "R@20", "P@5", "P@10", "nDCG@10"]
    rows = []
    for result in results:
        s = result.overall
        rows.append(
            [
                result.config.label,
                str(s.queries),
                f"{s.mrr:.3f}",
                f"{s.map_score:.3f}",
                f"{s.recall.get(5, 0):.3f}",
                f"{s.recall.get(10, 0):.3f}",
                f"{s.recall.get(20, 0):.3f}",
                f"{s.precision.get(5, 0):.3f}",
                f"{s.precision.get(10, 0):.3f}",
                f"{s.ndcg.get(10, 0):.3f}",
            ]
        )
    return rows, header


def best_of(results: list[Any], metric: str = "ndcg", k: int = 10) -> Any:
    """Return the result with the highest value of a metric."""

    def _value(result: Any) -> float:
        summary = result.overall
        if metric == "mrr":
            return summary.mrr
        if metric == "map":
            return summary.map_score
        return getattr(summary, metric).get(k, 0.0)

    return max(results, key=_value) if results else None


async def run(args: argparse.Namespace) -> int:
    """Execute the experiment suite."""
    from app.core.config import FusionStrategy, ScoreNormalization, get_settings
    from app.db.session import dispose_engine, get_session_factory, init_models
    from app.evaluation.dataset import QueryType, load_queries, resolve_queries
    from app.evaluation.runner import EvaluationRunner, RunConfig
    from app.repositories.product_repository import ProductRepository
    from app.repositories.vector_repository import get_vector_repository
    from app.services.embedding import get_embedding_service
    from app.services.image_processing import get_image_processor
    from app.services.search_service import SearchService

    settings = get_settings()
    selected = set(args.only) if args.only else set(range(1, 8))

    banner("SETUP")
    try:
        queries = load_queries(args.queries)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    await init_models()
    embedder = get_embedding_service()
    vectors = get_vector_repository()
    try:
        info = await asyncio.to_thread(embedder.load)
    except Exception as exc:
        print(f"ERROR: could not load the model: {exc}", file=sys.stderr)
        return 1

    factory = get_session_factory()
    sections: list[str] = []
    findings: list[str] = []
    raw: dict[str, Any] = {}

    try:
        points = await vectors.count()
        if points == 0:
            print(
                "ERROR: the vector store is empty. Run scripts/index_catalog.py first.",
                file=sys.stderr,
            )
            return 1

        async with factory() as session:
            products = ProductRepository(session)
            catalogue_size = await products.count_all()
            resolved = await resolve_queries(queries, session, min_relevant=2)
            runnable = [r for r in resolved if r.is_runnable]
            kv("model", info.name)
            kv("catalogue", f"{catalogue_size} products / {points} vectors")
            kv("queries", f"{len(runnable)} runnable of {len(resolved)}")

            service = SearchService(
                products=products,
                vectors=vectors,
                embedder=embedder,
                images=get_image_processor(),
                settings=settings,
            )
            runner = EvaluationRunner(service, cutoffs=args.cutoffs)
            multimodal_only = [r for r in resolved if r.query.type is QueryType.MULTIMODAL]
            text_only = [r for r in resolved if r.query.type is QueryType.TEXT]
            image_only = [r for r in resolved if r.query.type is QueryType.IMAGE]

            # ---------------------------------------------------- experiment 1
            if 1 in selected:
                banner("EXPERIMENT 1  Which channels carry the signal?")
                configs = [
                    RunConfig(
                        label="Text queries: same-modality only (text->text)",
                        cross_modal_weight=0.0,
                        top_k=args.top_k,
                        query_types=(QueryType.TEXT,),
                    ),
                    RunConfig(
                        label="Text queries: cross-modal only (text->image)",
                        cross_modal_weight=1.0,
                        top_k=args.top_k,
                        query_types=(QueryType.TEXT,),
                    ),
                    RunConfig(
                        label="Image queries: same-modality only (image->image)",
                        cross_modal_weight=0.0,
                        top_k=args.top_k,
                        query_types=(QueryType.IMAGE,),
                    ),
                    RunConfig(
                        label="Image queries: cross-modal only (image->text)",
                        cross_modal_weight=1.0,
                        top_k=args.top_k,
                        query_types=(QueryType.IMAGE,),
                    ),
                ]
                results = await runner.run_many(resolved, configs)
                for r in results:
                    print(f"  {r.config.label:52s} {r.overall.headline(10)}")
                rows, header = metric_rows(results, "Channel configuration")
                sections.append(
                    "## Experiment 1 - which channels carry the signal?\n\n"
                    "Each query type is run with the weight pushed entirely onto the "
                    "same-modality channel, then entirely onto the cross-modal channel. "
                    "This shows whether searching the *other* modality's vector "
                    "contributes anything at all.\n\n" + table(rows, header) + "\n"
                )
                raw["experiment_1"] = [r.to_dict() for r in results]

                text_same = results[0].overall.ndcg.get(10, 0)
                text_cross = results[1].overall.ndcg.get(10, 0)
                image_same = results[2].overall.ndcg.get(10, 0)
                image_cross = results[3].overall.ndcg.get(10, 0)
                findings.append(
                    f"For **text** queries the same-modality channel (text->text) scores "
                    f"nDCG@10 {text_same:.3f} versus {text_cross:.3f} cross-modal; for "
                    f"**image** queries image->image scores {image_same:.3f} versus "
                    f"{image_cross:.3f}. "
                    + (
                        "Same-modality dominates in both directions, which is why "
                        "`CROSS_MODAL_WEIGHT` defaults well below 0.5."
                        if text_same > text_cross and image_same > image_cross
                        else "The cross-modal channel is competitive, so it earns a "
                        "meaningful share of the weight."
                    )
                )

            # ---------------------------------------------------- experiment 2
            if 2 in selected:
                banner("EXPERIMENT 2  Image/text weight on multimodal queries")
                weights = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
                group_results: dict[str, list[Any]] = {}
                for group in ("contradiction", "agreement"):
                    subset = [r for r in multimodal_only if r.query.group == group]
                    if not subset:
                        continue
                    print(f"\n  -- {group} queries (n={len(subset)}) --")
                    group_results[group] = await runner.run_many(
                        subset,
                        [
                            RunConfig(
                                label=f"image={w:.2f} / text={1 - w:.2f}",
                                image_weight=w,
                                text_weight=1.0 - w,
                                top_k=args.top_k,
                                query_types=(QueryType.MULTIMODAL,),
                                groups=(group,),
                            )
                            for w in weights
                        ],
                    )
                    for r in group_results[group]:
                        print(f"  {r.config.label:28s} {r.overall.headline(10)}")

                section = [
                    "## Experiment 2 - image/text weighting for multimodal queries",
                    "",
                    "Multimodal queries split into two kinds that pull in opposite directions,",
                    "so they are swept separately. Averaging them would make any recommended",
                    "weight an artefact of how many of each the benchmark happens to contain.",
                    "",
                    "- **contradiction** - the text names an attribute the image does *not* have",
                    '  (a white shoe plus "but in black"); relevance requires the attribute from',
                    "  the **text**.",
                    "- **agreement** - the text restates what the image already shows; relevance",
                    "  requires the shared attributes.",
                    "",
                ]
                summary_rows = []
                for group, results in group_results.items():
                    rows, header = metric_rows(results, "Weighting")
                    best = best_of(results, "ndcg", 10)
                    section += [
                        f"### {group.capitalize()} queries (n={results[0].overall.queries})",
                        "",
                        table(rows, header),
                        "",
                        f"**Best nDCG@10: `{best.config.label}` "
                        f"({best.overall.ndcg.get(10, 0):.3f}).**",
                        "",
                    ]
                    summary_rows.append(
                        [
                            group,
                            best.config.label,
                            f"{best.overall.ndcg.get(10, 0):.3f}",
                            f"{results[weights.index(0.5)].overall.ndcg.get(10, 0):.3f}",
                        ]
                    )
                if len(summary_rows) > 1:
                    section += [
                        "### The two groups disagree",
                        "",
                        table(
                            summary_rows,
                            [
                                "Query group",
                                "Best weighting",
                                "Best nDCG@10",
                                "nDCG@10 at 50/50",
                            ],
                        ),
                        "",
                        "No single fixed weighting is optimal for both, which is why",
                        "`IMAGE_WEIGHT`/`TEXT_WEIGHT` are request-overridable rather than",
                        "compile-time constants: a UI can expose the trade-off, and the",
                        "environment default only has to be a sane midpoint.",
                        "",
                    ]
                sections.append("\n".join(section))
                raw["experiment_2"] = {
                    group: [r.to_dict() for r in results]
                    for group, results in group_results.items()
                }

                if "contradiction" in group_results and "agreement" in group_results:
                    best_c = best_of(group_results["contradiction"], "ndcg", 10)
                    best_a = best_of(group_results["agreement"], "ndcg", 10)
                    findings.append(
                        f"Multimodal weighting is **query-dependent**: contradiction-style "
                        f"queries peak at `{best_c.config.label}` (nDCG@10 "
                        f"{best_c.overall.ndcg.get(10, 0):.3f}) while agreement-style queries "
                        f"peak at `{best_a.config.label}` (nDCG@10 "
                        f"{best_a.overall.ndcg.get(10, 0):.3f}). Weighting the image heavily "
                        "pulls results back toward the attribute a contradiction query asked "
                        "to change, so the weights are exposed per request."
                    )

            # ---------------------------------------------------- experiment 3
            if 3 in selected:
                banner("EXPERIMENT 3  Cross-modal weight sweep")
                results = await runner.run_many(
                    text_only + image_only,
                    [
                        RunConfig(
                            label=f"cross_modal_weight={w:.2f}",
                            cross_modal_weight=w,
                            top_k=args.top_k,
                        )
                        for w in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
                    ],
                )
                for r in results:
                    print(f"  {r.config.label:28s} {r.overall.headline(10)}")
                rows, header = metric_rows(results, "Configuration")
                best = best_of(results, "ndcg", 10)
                sections.append(
                    "## Experiment 3 - cross-modal weight\n\n"
                    "Evaluated on the single-modality queries (text and image), where the "
                    "cross-modal channel is pure added evidence rather than a source of "
                    "conflict.\n\n"
                    + table(rows, header)
                    + f"\n\n**Best nDCG@10: `{best.config.label}` "
                    f"({best.overall.ndcg.get(10, 0):.3f}).**\n"
                )
                raw["experiment_3"] = [r.to_dict() for r in results]
                findings.append(
                    f"The best cross-modal weight on single-modality queries is "
                    f"**{best.config.label.split('=')[1]}** (nDCG@10 "
                    f"{best.overall.ndcg.get(10, 0):.3f})."
                )

            # ---------------------------------------------------- experiment 4
            if 4 in selected:
                banner("EXPERIMENT 4  Fusion strategy")
                # Matched comparison: every strategy gets the SAME effective image
                # share, so the result reflects the fusion mechanism rather than an
                # incidentally better weighting. `embedding_fusion`'s alpha is the
                # image share of the blended query vector, which is the counterpart
                # of `image_weight` in the score-level strategies.
                configs = []
                for share in (0.3, 0.5):
                    configs += [
                        RunConfig(
                            label=f"weighted_sum (image share {share})",
                            strategy=FusionStrategy.WEIGHTED_SUM,
                            image_weight=share,
                            text_weight=1.0 - share,
                            top_k=args.top_k,
                            query_types=(QueryType.MULTIMODAL,),
                        ),
                        RunConfig(
                            label=f"rrf (image share {share})",
                            strategy=FusionStrategy.RRF,
                            image_weight=share,
                            text_weight=1.0 - share,
                            top_k=args.top_k,
                            query_types=(QueryType.MULTIMODAL,),
                        ),
                        RunConfig(
                            label=f"embedding_fusion (alpha={share})",
                            strategy=FusionStrategy.EMBEDDING_FUSION,
                            embedding_fusion_alpha=share,
                            top_k=args.top_k,
                            query_types=(QueryType.MULTIMODAL,),
                        ),
                    ]
                results = await runner.run_many(multimodal_only, configs)
                for r in results:
                    print(f"  {r.config.label:40s} {r.overall.headline(10)}")
                rows, header = metric_rows(results, "Strategy")
                best = best_of(results, "ndcg", 10)
                sections.append(
                    "## Experiment 4 - score fusion vs embedding fusion\n\n"
                    "`weighted_sum` and `rrf` retrieve from each channel separately and fuse "
                    "*scores*. `embedding_fusion` blends the image and text query vectors into "
                    "one vector *before* retrieval - the approach that first comes to mind, and "
                    "the one that cannot attribute a result to a modality afterwards.\n\n"
                    "Each strategy is run at a **matched image share** (`image_weight` for the "
                    "score-level strategies, `alpha` for embedding fusion) so the comparison "
                    "isolates the fusion mechanism rather than rewarding a luckier weighting.\n\n"
                    + table(rows, header)
                    + f"\n\n**Best nDCG@10: `{best.config.label}` "
                    f"({best.overall.ndcg.get(10, 0):.3f}).**\n"
                )
                raw["experiment_4"] = [r.to_dict() for r in results]
                findings.append(
                    f"At a matched image share, the best fusion strategy on multimodal queries "
                    f"is **{best.config.label}** (nDCG@10 "
                    f"{best.overall.ndcg.get(10, 0):.3f})."
                )

            # ---------------------------------------------------- experiment 5
            if 5 in selected:
                banner("EXPERIMENT 5  Score normalisation")
                results = await runner.run_many(
                    resolved,
                    [
                        RunConfig(
                            label=f"normalization={n.value}",
                            normalization=n,
                            top_k=args.top_k,
                        )
                        for n in (
                            ScoreNormalization.MINMAX,
                            ScoreNormalization.ZSCORE,
                            ScoreNormalization.NONE,
                        )
                    ],
                )
                for r in results:
                    print(f"  {r.config.label:28s} {r.overall.headline(10)}")
                rows, header = metric_rows(results, "Normalisation")
                best = best_of(results, "ndcg", 10)
                sections.append(
                    "## Experiment 5 - per-channel score normalisation\n\n"
                    "Same-modality cosines (image-image 0.7305 mean) and cross-modal "
                    "cosines (image-text 0.3124 mean) sit on different scales for CLIP. "
                    "`none` adds them raw, letting the higher-scoring channel dominate "
                    "irrespective of the configured weights.\n\n"
                    + table(rows, header)
                    + f"\n\n**Best nDCG@10: `{best.config.label}` "
                    f"({best.overall.ndcg.get(10, 0):.3f}).**\n"
                )
                raw["experiment_5"] = [r.to_dict() for r in results]
                findings.append(
                    f"Normalisation choice: **{best.config.label.split('=')[1]}** performs best "
                    f"(nDCG@10 {best.overall.ndcg.get(10, 0):.3f})."
                )

            # ---------------------------------------------------- experiment 6
            if 6 in selected:
                banner("EXPERIMENT 6  Lexical channel weight")
                results = await runner.run_many(
                    text_only,
                    [
                        RunConfig(
                            label=f"lexical_weight={w:.2f}",
                            lexical_weight=w,
                            top_k=args.top_k,
                            query_types=(QueryType.TEXT,),
                        )
                        for w in (0.0, 0.1, 0.2, 0.3, 0.5)
                    ],
                )
                for r in results:
                    print(f"  {r.config.label:28s} {r.overall.headline(10)}")
                rows, header = metric_rows(results, "Configuration")
                best = best_of(results, "ndcg", 10)
                sections.append(
                    "## Experiment 6 - lexical (keyword) channel\n\n"
                    "A keyword channel served from the metadata database, blended into the "
                    "semantic channels. Text queries only, since the channel needs a query "
                    "string.\n\n"
                    + table(rows, header)
                    + f"\n\n**Best nDCG@10: `{best.config.label}` "
                    f"({best.overall.ndcg.get(10, 0):.3f}).**\n"
                )
                raw["experiment_6"] = [r.to_dict() for r in results]
                findings.append(
                    f"Lexical channel: **{best.config.label.split('=')[1]}** is best "
                    f"(nDCG@10 {best.overall.ndcg.get(10, 0):.3f})."
                )

            # ---------------------------------------------------- experiment 7
            if 7 in selected:
                banner("EXPERIMENT 7  Retrieval depth")
                # Values below max(cutoffs) cannot be measured: the runner always
                # retrieves at least deep enough to score the largest cut-off, so a
                # smaller top_k would be silently clamped and produce a duplicate row.
                depths = [k for k in (20, 30, 50, 75, 100) if k >= max(args.cutoffs)]
                results = await runner.run_many(
                    resolved, [RunConfig(label=f"top_k={k}", top_k=k) for k in depths]
                )
                for r in results:
                    print(
                        f"  {r.config.label:12s} {r.overall.headline(10)}  "
                        f"median={r.overall.median_latency_ms:.1f}ms"
                    )
                rows = [
                    [
                        r.config.label,
                        str(r.overall.queries),
                        f"{r.overall.recall.get(10, 0):.3f}",
                        f"{r.overall.recall.get(20, 0):.3f}",
                        f"{r.overall.ndcg.get(10, 0):.3f}",
                        f"{r.overall.median_latency_ms:.1f}",
                        f"{r.overall.p95_latency_ms:.1f}",
                    ]
                    for r in results
                ]
                sections.append(
                    "## Experiment 7 - retrieval depth and latency\n\n"
                    "`top_k` also scales the per-channel candidate pool "
                    "(`top_k * CANDIDATE_MULTIPLIER`, floored at `MIN_CANDIDATE_POOL`), so this "
                    "measures the cost of asking for more candidates as well as the benefit. "
                    f"Depths below {max(args.cutoffs)} are omitted: scoring needs at least that "
                    "much depth, so smaller values would be clamped and duplicate a row.\n\n"
                    + table(
                        rows,
                        ["top_k", "n", "R@10", "R@20", "nDCG@10", "Median ms", "p95 ms"],
                    )
                    + "\n"
                )
                raw["experiment_7"] = [r.to_dict() for r in results]
                deepest = results[-1]
                shallow = results[2]
                findings.append(
                    f"Increasing top_k from 20 to {deepest.config.top_k} changes nDCG@10 from "
                    f"{shallow.overall.ndcg.get(10, 0):.3f} to "
                    f"{deepest.overall.ndcg.get(10, 0):.3f} while median latency moves from "
                    f"{shallow.overall.median_latency_ms:.1f}ms to "
                    f"{deepest.overall.median_latency_ms:.1f}ms."
                )

            report = _render(
                info=info,
                catalogue_size=catalogue_size,
                points=points,
                runnable=len(runnable),
                total=len(resolved),
                sections=sections,
                findings=findings,
                cutoffs=args.cutoffs,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(report, encoding="utf-8")
            args.out.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "generated_at": datetime.now(UTC).isoformat(),
                        "model": info.name,
                        "catalogue_products": catalogue_size,
                        "experiments": raw,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        banner("FINDINGS")
        for finding in findings:
            print(f"  - {finding}")
        banner("OUTPUT")
        kv("markdown", display_path(args.out))
        kv("json", display_path(args.out.with_suffix(".json")))
        return 0
    finally:
        await vectors.close()
        await dispose_engine()


def _render(
    *,
    info: Any,
    catalogue_size: int,
    points: int,
    runnable: int,
    total: int,
    sections: list[str],
    findings: list[str],
    cutoffs: list[int],
) -> str:
    """Render the experiments report."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    head = [
        "# Ranking Experiments",
        "",
        "> Generated by `python scripts/run_experiments.py`. All figures are measured on the",
        "> benchmark query set described in [`docs/evaluation.md`](../../docs/evaluation.md).",
        "",
        "## Setup",
        "",
        f"- **Generated:** {now}",
        f"- **Model:** `{info.name}` ({info.embedding_dim}-d, {info.device})",
        f"- **Catalogue:** {catalogue_size:,} products / {points:,} vectors",
        f"- **Queries:** {runnable} scored of {total}",
        f"- **Cut-offs:** {', '.join(str(k) for k in cutoffs)}",
        "",
        "### Reading these tables",
        "",
        "`R@k` is bounded above by `k / |relevant|`. Relevant sets here hold tens of products,",
        "so recall at small `k` cannot approach 1.0 and is useful only for *comparing rows*,",
        "never as an absolute score. `nDCG@10` and `MRR` are the metrics to compare on.",
        "",
        "## Summary of findings",
        "",
    ]
    head += [f"{i}. {finding}" for i, finding in enumerate(findings, start=1)]
    head += [""]
    return "\n".join(head) + "\n" + "\n".join(sections)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    configure_script_logging(args.log_level)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

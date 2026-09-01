#!/usr/bin/env python
"""Measure retrieval quality and write an evaluation report.

Compares the three search modes on the same benchmark query set and writes both
a Markdown report and a JSON record of the run.

Usage::

    python scripts/evaluate.py
    python scripts/evaluate.py --top-k 20 --out evaluation/results/report.md
    python scripts/evaluate.py --per-query        # include the per-query table

Read ``docs/evaluation.md`` before quoting these numbers: relevance is derived
from product attributes, which measures attribute agreement rather than human
preference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from _bootstrap import REPO_ROOT, banner, configure_script_logging, display_path, kv

DEFAULT_QUERIES = REPO_ROOT / "evaluation" / "queries" / "benchmark.json"
DEFAULT_OUT = REPO_ROOT / "evaluation" / "results" / "evaluation_report.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate text, image and multimodal search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="K values for precision/recall/nDCG.",
    )
    parser.add_argument(
        "--min-relevant",
        type=int,
        default=2,
        help="Skip queries with fewer relevant products than this.",
    )
    parser.add_argument(
        "--per-query", action="store_true", help="Include a per-query table in the report."
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


def _markdown_table(rows: list[list[str]], header: list[str]) -> str:
    """Render a Markdown table."""
    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows)) if rows else len(header[i])
        for i in range(len(header))
    ]
    lines = [
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
        "|" + "|".join("-" * (widths[i] + 2) for i in range(len(header))) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(header))) + " |"
        )
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    """Execute the evaluation."""
    from app.core.config import get_settings
    from app.db.session import dispose_engine, get_session_factory, init_models
    from app.evaluation.dataset import QueryType, load_queries, resolve_queries
    from app.evaluation.runner import EvaluationRunner, RunConfig
    from app.repositories.product_repository import ProductRepository
    from app.repositories.vector_repository import get_vector_repository
    from app.services.embedding import get_embedding_service
    from app.services.image_processing import get_image_processor
    from app.services.search_service import SearchService

    settings = get_settings()

    banner("SETUP")
    try:
        queries = load_queries(args.queries)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    kv("query set", display_path(args.queries))
    kv("queries loaded", len(queries))

    await init_models()
    embedder = get_embedding_service()
    vectors = get_vector_repository()
    try:
        info = await asyncio.to_thread(embedder.load)
    except Exception as exc:
        print(f"ERROR: could not load the model: {exc}", file=sys.stderr)
        return 1
    kv("model", info.name)
    kv("embedding dim", info.embedding_dim)
    kv("device", info.device)

    factory = get_session_factory()
    try:
        points = await vectors.count()
        if points == 0:
            print(
                "\nERROR: the vector store is empty. Index a catalogue first:\n"
                "  python scripts/index_catalog.py",
                file=sys.stderr,
            )
            return 1
        kv("indexed vectors", points)

        async with factory() as session:
            products = ProductRepository(session)
            catalogue_size = await products.count_all()
            resolved = await resolve_queries(queries, session, min_relevant=args.min_relevant)
            kv("catalogue products", catalogue_size)

            runnable = [r for r in resolved if r.is_runnable]
            kv("runnable queries", f"{len(runnable)}/{len(resolved)}")
            if skipped := [r for r in resolved if not r.is_runnable]:
                print(f"\n  skipped {len(skipped)} query(ies):")
                for item in skipped[:10]:
                    print(f"    {item.query.id}: {item.skip_reason}")
            if not runnable:
                print(
                    "\nERROR: no queries can be scored against this catalogue.", file=sys.stderr
                )
                return 1

            sizes = [len(r.relevant_ids) for r in runnable]
            kv(
                "relevant set size",
                f"min={min(sizes)} median={sorted(sizes)[len(sizes) // 2]} max={max(sizes)}",
            )

            service = SearchService(
                products=products,
                vectors=vectors,
                embedder=embedder,
                images=get_image_processor(),
                settings=settings,
            )
            runner = EvaluationRunner(service, cutoffs=args.cutoffs)

            banner("EVALUATING SEARCH MODES")
            configs = [
                RunConfig(
                    label="Text-only search",
                    top_k=args.top_k,
                    query_types=(QueryType.TEXT,),
                ),
                RunConfig(
                    label="Image-only search",
                    top_k=args.top_k,
                    query_types=(QueryType.IMAGE,),
                ),
                RunConfig(
                    label="Multimodal search",
                    top_k=args.top_k,
                    query_types=(QueryType.MULTIMODAL,),
                ),
                RunConfig(label="All modes combined", top_k=args.top_k),
            ]
            results = await runner.run_many(resolved, configs)

            for result in results:
                print(f"\n  {result.config.label}")
                print(f"    {result.overall.headline(10)}")
                for query_id, error in result.errors:
                    print(f"    ERROR {query_id}: {error}")

            # A multimodal query whose text is ignored should fail: the target
            # colour never matches the query image's colour. Running the same
            # queries image-only quantifies how much the text actually adds.
            banner("ABLATION: do the multimodal queries really need the text?")
            ablation = await runner.run(
                [r for r in resolved if r.query.type is QueryType.MULTIMODAL],
                RunConfig(
                    label="Multimodal queries, image channel only",
                    image_weight=1.0,
                    text_weight=0.0,
                    top_k=args.top_k,
                    query_types=(QueryType.MULTIMODAL,),
                ),
            )
            text_only_on_mm = await runner.run(
                [r for r in resolved if r.query.type is QueryType.MULTIMODAL],
                RunConfig(
                    label="Multimodal queries, text channel only",
                    image_weight=0.0,
                    text_weight=1.0,
                    top_k=args.top_k,
                    query_types=(QueryType.MULTIMODAL,),
                ),
            )
            multimodal = next(r for r in results if r.config.label == "Multimodal search")
            print(f"    image only : {ablation.overall.headline(10)}")
            print(f"    text only  : {text_only_on_mm.overall.headline(10)}")
            print(f"    both       : {multimodal.overall.headline(10)}")

            report = _build_report(
                args=args,
                settings=settings,
                model=info,
                catalogue_size=catalogue_size,
                points=points,
                resolved=resolved,
                results=results,
                ablation=(ablation, text_only_on_mm, multimodal),
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(report, encoding="utf-8")

            json_path = args.out.with_suffix(".json")
            json_path.write_text(
                json.dumps(
                    {
                        "generated_at": datetime.now(UTC).isoformat(),
                        "model": info.name,
                        "embedding_dim": info.embedding_dim,
                        "catalogue_products": catalogue_size,
                        "indexed_vectors": points,
                        "cutoffs": args.cutoffs,
                        "runs": [r.to_dict(include_queries=args.per_query) for r in results],
                        "ablation": {
                            "image_only": ablation.to_dict(),
                            "text_only": text_only_on_mm.to_dict(),
                            "multimodal": multimodal.to_dict(),
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        banner("OUTPUT")
        kv("markdown report", display_path(args.out))
        kv("json results", display_path(json_path))
        return 0
    finally:
        await vectors.close()
        await dispose_engine()


def _build_report(
    *,
    args: argparse.Namespace,
    settings: object,
    model: object,
    catalogue_size: int,
    points: int,
    resolved: list,
    results: list,
    ablation: tuple,
) -> str:
    """Render the Markdown evaluation report."""
    image_only, text_only_mm, multimodal = ablation
    cutoffs = args.cutoffs
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        "# Evaluation Report",
        "",
        "> Generated by `python scripts/evaluate.py`. Every number here is measured, not estimated.",
        "> Read the methodology and limitations in [`docs/evaluation.md`](../../docs/evaluation.md)",
        "> before quoting these figures.",
        "",
        "## Run configuration",
        "",
        f"- **Generated:** {now}",
        f"- **Model:** `{getattr(model, 'name', '?')}` "
        f"({getattr(model, 'embedding_dim', '?')}-d, {getattr(model, 'device', '?')})",
        f"- **Catalogue:** {catalogue_size:,} products, {points:,} indexed vectors",
        f"- **Fusion strategy:** `{getattr(settings, 'fusion_strategy', '?')}`, "
        f"normalisation `{getattr(settings, 'score_normalization', '?')}`",
        f"- **Weights:** image {getattr(settings, 'image_weight', '?')}, "
        f"text {getattr(settings, 'text_weight', '?')}, "
        f"cross-modal {getattr(settings, 'cross_modal_weight', '?')}, "
        f"lexical {getattr(settings, 'lexical_weight', '?')}",
        f"- **Retrieval depth:** top_k={args.top_k}",
        "",
    ]

    runnable = [r for r in resolved if r.is_runnable]
    by_type_counts: dict[str, int] = {}
    for item in runnable:
        by_type_counts[item.query.type.value] = by_type_counts.get(item.query.type.value, 0) + 1
    sizes = [len(r.relevant_ids) for r in runnable]

    lines += [
        "## Benchmark query set",
        "",
        f"- **Queries scored:** {len(runnable)} of {len(resolved)}"
        + (
            f" ({len(resolved) - len(runnable)} skipped)"
            if len(resolved) != len(runnable)
            else ""
        ),
        "- **By type:** "
        + ", ".join(f"{name} {count}" for name, count in sorted(by_type_counts.items())),
        f"- **Relevant products per query:** min {min(sizes)}, "
        f"median {sorted(sizes)[len(sizes) // 2]}, max {max(sizes)}",
        "",
        "Relevance is defined by an attribute predicate per query, resolved against the",
        "catalogue at run time (see `evaluation/queries/benchmark.json`).",
        "",
    ]

    header = ["Configuration", "Queries", "MRR", "MAP"]
    for k in cutoffs:
        header += [f"R@{k}", f"P@{k}", f"nDCG@{k}"]
    rows = []
    for result in results:
        summary = result.overall
        row = [
            result.config.label,
            str(summary.queries),
            f"{summary.mrr:.3f}",
            f"{summary.map_score:.3f}",
        ]
        for k in cutoffs:
            row += [
                f"{summary.recall.get(k, 0):.3f}",
                f"{summary.precision.get(k, 0):.3f}",
                f"{summary.ndcg.get(k, 0):.3f}",
            ]
        rows.append(row)

    lines += [
        "## Search mode comparison",
        "",
        "Each mode is evaluated on the queries designed for it. The three query sets are",
        "*different*, so the rows are not directly comparable to each other - a text query",
        "and an image query pose different problems. The comparison that controls for this",
        "is the ablation below, which runs one fixed query set through different channels.",
        "",
        _markdown_table(rows, header),
        "",
        "### Latency",
        "",
        _markdown_table(
            [
                [
                    r.config.label,
                    f"{r.overall.median_latency_ms:.1f}",
                    f"{r.overall.p95_latency_ms:.1f}",
                ]
                for r in results
            ],
            ["Configuration", "Median ms", "p95 ms"],
        ),
        "",
        "Measured in-process (no HTTP), CPU only, including query embedding.",
        "",
    ]

    lines += [
        "## Ablation: is the multimodal query actually multimodal?",
        "",
        "The 10 multimodal queries pair an image with text that *contradicts* one of the",
        'image\'s attributes (a white shoe plus "but in black"). The relevant set requires the',
        "attribute named in the **text**, so a system that quietly ignores the text cannot",
        "score well. Running the identical query set through each channel isolates the",
        "contribution of each modality:",
        "",
        _markdown_table(
            [
                [
                    "Image channel only (image_weight=1)",
                    f"{image_only.overall.recall.get(10, 0):.3f}",
                    f"{image_only.overall.precision.get(10, 0):.3f}",
                    f"{image_only.overall.mrr:.3f}",
                    f"{image_only.overall.ndcg.get(10, 0):.3f}",
                ],
                [
                    "Text channel only (text_weight=1)",
                    f"{text_only_mm.overall.recall.get(10, 0):.3f}",
                    f"{text_only_mm.overall.precision.get(10, 0):.3f}",
                    f"{text_only_mm.overall.mrr:.3f}",
                    f"{text_only_mm.overall.ndcg.get(10, 0):.3f}",
                ],
                [
                    "Both (multimodal)",
                    f"{multimodal.overall.recall.get(10, 0):.3f}",
                    f"{multimodal.overall.precision.get(10, 0):.3f}",
                    f"{multimodal.overall.mrr:.3f}",
                    f"{multimodal.overall.ndcg.get(10, 0):.3f}",
                ],
            ],
            ["Channels used", "R@10", "P@10", "MRR", "nDCG@10"],
        ),
        "",
    ]

    for label, result in (
        ("Text", next((r for r in results if r.config.label == "Text-only search"), None)),
        ("Image", next((r for r in results if r.config.label == "Image-only search"), None)),
        ("Multimodal", multimodal),
    ):
        if result and result.errors:
            lines += [f"### Errors ({label})", ""]
            lines += [f"- `{qid}`: {err}" for qid, err in result.errors]
            lines += [""]

    if args.per_query:
        lines += ["## Per-query results", ""]
        for result in results:
            if result.config.label == "All modes combined":
                continue
            lines += [f"### {result.config.label}", ""]
            rows = [
                [
                    score.query_id,
                    score.query_type,
                    str(score.relevant_count),
                    str(score.first_relevant_rank or "-"),
                    f"{score.recall.get(10, 0):.3f}",
                    f"{score.precision.get(10, 0):.3f}",
                    f"{score.reciprocal_rank:.3f}",
                ]
                for score in result.scores
            ]
            lines += [
                _markdown_table(
                    rows, ["Query", "Type", "Relevant", "First hit", "R@10", "P@10", "RR"]
                ),
                "",
            ]

    skipped = [r for r in resolved if not r.is_runnable]
    if skipped:
        lines += ["## Skipped queries", ""]
        lines += [f"- `{r.query.id}`: {r.skip_reason}" for r in skipped]
        lines += [""]

    lines += [
        "## Reproducing this report",
        "",
        "```bash",
        "python scripts/download_dataset.py --limit 6000 --out data/eval",
        "python scripts/index_catalog.py --source data/eval/products.csv",
        "python scripts/evaluate.py --per-query",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    configure_script_logging(args.log_level)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

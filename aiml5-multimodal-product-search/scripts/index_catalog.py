#!/usr/bin/env python
"""Load a catalogue into PostgreSQL/SQLite and index it into Qdrant.

Two stages, both idempotent:

1. **Load** - upsert the canonical CSV into the products table, keyed on
   ``external_id``.
2. **Index** - embed products whose content fingerprint changed and write both
   named vectors to Qdrant.

Re-running after adding rows to the CSV embeds only the new rows. Nothing is
recomputed unnecessarily; that is the point of the fingerprint.

Examples:
--------
Index the sample catalogue::

    python scripts/index_catalog.py

Index a larger catalogue built by ``download_dataset.py``::

    python scripts/index_catalog.py --source data/full/products.csv

Re-embed everything after changing ``MODEL_NAME``::

    python scripts/index_catalog.py --force

Start from a clean collection::

    python scripts/index_catalog.py --recreate
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from _bootstrap import DATA_ROOT, banner, configure_script_logging, kv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load and index a product catalogue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DATA_ROOT / "sample" / "products.csv",
        help="Canonical catalogue CSV to load.",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Index products already in the database; do not read the CSV.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-embed even when the fingerprint matches."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and rebuild the Qdrant collection before indexing.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Index at most N products.")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override INGEST_BATCH_SIZE."
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    """Execute the pipeline."""
    from app.core.config import get_settings
    from app.core.exceptions import AppError
    from app.data.catalog_loader import CatalogLoader
    from app.db.session import dispose_engine, get_session_factory, init_models
    from app.repositories.product_repository import ProductRepository
    from app.repositories.vector_repository import get_vector_repository
    from app.services.embedding import get_embedding_service
    from app.services.image_processing import get_image_processor
    from app.services.indexing_service import IndexingService, IndexProgress

    settings = get_settings()
    if args.batch_size:
        settings.ingest_batch_size = args.batch_size

    banner("CONFIGURATION")
    kv("database", "postgresql" if settings.is_postgres else "sqlite")
    kv(
        "qdrant",
        "server" if settings.uses_qdrant_server else f"embedded ({settings.qdrant_path})",
    )
    kv("collection", settings.qdrant_collection)
    kv("model", settings.model_name)
    kv("batch size", settings.ingest_batch_size)

    await init_models()
    embedder = get_embedding_service()
    vectors = get_vector_repository()
    images = get_image_processor()
    factory = get_session_factory()

    banner("STEP 1/3  Load the embedding model")
    started = time.perf_counter()
    try:
        info = await asyncio.to_thread(embedder.load)
    except AppError as exc:
        print(f"\nERROR: {exc.message}", file=sys.stderr)
        return 1
    kv("model", info.name)
    kv("architecture", info.model_type)
    kv("embedding dim", info.embedding_dim)
    kv("device", f"{info.device} ({info.dtype})")
    kv("load time", f"{time.perf_counter() - started:.1f}s")

    try:
        if not args.skip_load:
            banner("STEP 2/3  Load catalogue metadata")
            async with factory() as session:
                loader = CatalogLoader(ProductRepository(session))
                try:
                    report = await loader.load_csv(args.source)
                except FileNotFoundError as exc:
                    print(f"\nERROR: {exc}", file=sys.stderr)
                    return 1
                await session.commit()
            kv("source", args.source)
            kv("rows read", report.total_rows)
            kv("created", report.created)
            kv("updated", report.updated)
            kv("invalid rows", report.invalid)
            kv("duplicates in file", report.duplicates_in_file)
            for error in report.errors[:5]:
                print(f"    line {error.line}: {error.reason}")
        else:
            banner("STEP 2/3  Load catalogue metadata (skipped)")

        banner("STEP 3/3  Embed and index")
        last_line_length = 0

        def on_progress(progress: IndexProgress) -> None:
            nonlocal last_line_length
            bar_width = 32
            filled = int(bar_width * progress.processed / max(1, progress.total))
            line = (
                f"  [{'#' * filled}{'.' * (bar_width - filled)}] "
                f"{progress.processed}/{progress.total} "
                f"({progress.percent:5.1f}%)  {progress.rate:5.2f}/s  "
                f"embedded={progress.embedded} skipped={progress.skipped} "
                f"failed={progress.failed}"
            )
            padding = " " * max(0, last_line_length - len(line))
            last_line_length = len(line)
            print(f"\r{line}{padding}", end="", flush=True)

        async with factory() as session:
            indexer = IndexingService(
                products=ProductRepository(session),
                vectors=vectors,
                embedder=embedder,
                images=images,
                settings=settings,
            )
            try:
                result = await indexer.index_catalog(
                    force=args.force,
                    limit=args.limit,
                    recreate=args.recreate,
                    progress=on_progress,
                )
            except AppError as exc:
                print(f"\n\nERROR: {exc.message}", file=sys.stderr)
                return 1
            await session.commit()
        print()

        banner("RESULT")
        kv("considered", result.requested)
        kv("embedded", result.embedded)
        kv("skipped (unchanged)", result.skipped_unchanged)
        kv("failed", result.failed)
        kv("with image vector", result.image_vectors)
        kv("with text vector", result.text_vectors)
        kv("duration", f"{result.duration_ms / 1000:.1f}s")
        if result.embedded:
            kv("throughput", f"{result.embedded / (result.duration_ms / 1000):.2f} products/s")
        kv("points in qdrant", await vectors.count())

        if result.failures:
            print(f"\n  {len(result.failures)} product(s) needed attention:")
            for failure in result.failures[:10]:
                print(
                    f"    {failure.product_id[:8]}  {(failure.name or '')[:38]:40s} {failure.reason}"
                )
            if len(result.failures) > 10:
                print(f"    ... and {len(result.failures) - 10} more")

        print("\nNext:")
        print("  uvicorn app.main:app --app-dir backend --reload   # start the API")
        print("  python scripts/evaluate.py                        # measure retrieval quality")
        return 0
    finally:
        await vectors.close()
        await dispose_engine()


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    configure_script_logging(args.log_level)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

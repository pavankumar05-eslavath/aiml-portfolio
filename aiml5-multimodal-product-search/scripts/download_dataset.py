#!/usr/bin/env python
"""Download the fashion product dataset and build a canonical catalogue.

Produces two artefacts under the output directory:

* ``products.csv`` - the canonical catalogue consumed by ``index_catalog.py``
* ``images/<external_id>.jpg`` - one image per product

Examples:
--------
Build a 2,000-product catalogue (stratified across article types)::

    python scripts/download_dataset.py --limit 2000 --out data/full

Build the entire ~44,000-product catalogue::

    python scripts/download_dataset.py --all --out data/full

Only the first shard (~22,000 products, half the download)::

    python scripts/download_dataset.py --limit 5000 --shards 1

Neither the images nor the generated CSV are committed to git; ``data/.gitignore``
excludes them. See ``data/README.md`` for licensing and provenance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import DATA_ROOT, banner, configure_script_logging, display_path, kv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download the fashion dataset and build a canonical catalogue CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_ROOT / "full",
        help="Output directory for products.csv and images/.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int, default=2000, help="Number of products to export.")
    group.add_argument(
        "--all", action="store_true", help="Export every product in the shards read."
    )
    parser.add_argument(
        "--shards",
        type=int,
        choices=(1, 2),
        default=2,
        help="How many Parquet shards to download (1 halves the download).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DATA_ROOT / "raw" / "_hf",
        help="Where to cache the downloaded Parquet shards.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=3,
        help="Minimum occurrences before a mined name is accepted as a brand.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Sampling seed.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Rewrite images that already exist."
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    configure_script_logging(args.log_level)
    # Resolve up front: the rest of the script joins and displays these paths.
    args.out = args.out.resolve()
    args.cache = args.cache.resolve()

    from app.data import fashion_dataset as fd

    banner("STEP 1/5  Download Parquet shards")
    try:
        shards = fd.download_parquet_shards(args.cache, fd.HF_PARQUET_FILES[: args.shards])
    except Exception as exc:
        print(f"\nERROR: could not download the dataset: {exc}", file=sys.stderr)
        print(
            "The dataset is fetched from the Hugging Face Hub and needs network "
            "access. No credentials are required.",
            file=sys.stderr,
        )
        return 1
    for shard in shards:
        kv(shard.name[:34], f"{shard.stat().st_size / 1_048_576:.1f} MB")

    banner("STEP 2/5  Read metadata")
    rows = fd.read_metadata(shards)
    usable = [row for row in rows if fd.is_usable(row)]
    kv("rows read", len(rows))
    kv("usable rows", len(usable))

    banner("STEP 3/5  Mine brand vocabulary")
    # Learned from every display name, not just the exported subset, so the
    # frequency evidence brand extraction relies on is as strong as possible.
    vocabulary = fd.learn_brands(rows, min_count=args.min_count)
    kv("candidate n-grams", len(vocabulary.counts))
    kv("attribute words excluded", len(vocabulary.attribute_words))

    banner("STEP 4/5  Select and export images")
    selected = (
        usable if args.all else fd.select_stratified(usable, limit=args.limit, seed=args.seed)
    )
    kv("products selected", len(selected))

    images_dir = args.out / "images"
    exported = fd.export_images(
        shards,
        {str(row["id"]) for row in selected},
        images_dir,
        overwrite=args.overwrite,
    )
    kv("images exported", len(exported))
    if missing := len(selected) - len(exported):
        kv("images missing", missing)

    banner("STEP 5/5  Write canonical catalogue")
    relative_root = args.out.resolve().relative_to(DATA_ROOT.resolve())
    records = []
    for row in selected:
        external_id = str(row["id"])
        if external_id not in exported:
            continue
        # image_url is stored relative to IMAGE_ROOT (data/) so the catalogue
        # remains portable between the host and the container.
        image_reference = f"{relative_root}/images/{external_id}.jpg"
        records.append(fd.to_record(row, vocabulary, image_reference))

    csv_path = fd.write_catalog_csv(records, args.out / "products.csv")
    summary = fd.summarise(records)

    banner("SUMMARY")
    kv("catalogue csv", display_path(csv_path))
    kv("images directory", display_path(images_dir))
    for key in ("products", "categories", "article_types", "brands", "with_brand", "colours"):
        kv(key, summary[key])
    kv("brand coverage", f"{100 * summary['with_brand'] / max(1, summary['products']):.1f}%")
    kv("price range (SYNTHETIC)", f"${summary['price_min']} - ${summary['price_max']}")
    print("\n  top article types:")
    for name, count in summary["top_article_types"]:
        print(f"    {count:6d}  {name}")

    print("\nNext:")
    print(f"  python scripts/index_catalog.py --source {display_path(csv_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

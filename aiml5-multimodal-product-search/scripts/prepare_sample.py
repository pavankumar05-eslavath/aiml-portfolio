#!/usr/bin/env python
"""Build the small sample catalogue used by the demo and the evaluation set.

The repository commits ``data/sample/products.csv`` (metadata only) but **not** the
product images: the upstream dataset does not declare a licence, so redistributing
its images would be careless. This script fetches exactly the images the committed
CSV references.

Network access is required, which costs nothing extra in practice because the CLIP
weights are also downloaded from Hugging Face on first run.

Usage::

    python scripts/prepare_sample.py              # images for the committed CSV
    python scripts/prepare_sample.py --rebuild    # re-select and rewrite the CSV

``--rebuild`` regenerates the sample from scratch. It is deterministic given
``--seed``, but note that the committed evaluation ground truth in
``evaluation/queries/`` was derived from the committed CSV, so rebuilding with a
different seed or size invalidates it - regenerate it with
``python scripts/build_eval_set.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import DATA_ROOT, banner, configure_script_logging, display_path, kv

SAMPLE_DIR = DATA_ROOT / "sample"
DEFAULT_SIZE = 400


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch images for the sample catalogue (and optionally rebuild it).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-select products and rewrite products.csv instead of only fetching images.",
    )
    parser.add_argument(
        "--size", type=int, default=DEFAULT_SIZE, help="Sample size when rebuilding."
    )
    parser.add_argument("--seed", type=int, default=17, help="Selection seed.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=DATA_ROOT / "raw" / "_hf",
        help="Where to cache the downloaded Parquet shards.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rewrite images that already exist."
    )
    parser.add_argument(
        "--shards",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "Parquet shards to download. The committed catalogue spans both, so 1 "
            "halves the download (~130 MB) but leaves roughly half the products "
            "without an image."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    configure_script_logging(args.log_level)

    from app.data import fashion_dataset as fd

    csv_path = SAMPLE_DIR / "products.csv"
    images_dir = SAMPLE_DIR / "images"

    if not args.rebuild and not csv_path.is_file():
        print(
            f"ERROR: {csv_path} is missing. It should be committed to the repository.\n"
            "Rebuild it from the source dataset with:\n"
            "  python scripts/prepare_sample.py --rebuild",
            file=sys.stderr,
        )
        return 1

    banner("STEP 1/3  Download Parquet shards")
    # Both shards by default. The committed catalogue is stratified across the
    # whole dataset, so its products are split roughly evenly between the two
    # shards - fetching only the first leaves about half the catalogue with no
    # image, which then indexes as text-only.
    try:
        shards = fd.download_parquet_shards(args.cache, fd.HF_PARQUET_FILES[: args.shards])
    except Exception as exc:
        print(f"\nERROR: could not download the dataset: {exc}", file=sys.stderr)
        return 1
    for shard in shards:
        kv(shard.name[:34], f"{shard.stat().st_size / 1_048_576:.1f} MB")

    if args.rebuild:
        banner("STEP 2/3  Select products and write products.csv")
        rows = fd.read_metadata(shards)
        vocabulary = fd.learn_brands(rows)
        usable = [row for row in rows if fd.is_usable(row)]
        selected = fd.select_stratified(usable, limit=args.size, seed=args.seed)
        wanted = {str(row["id"]) for row in selected}
        exported = fd.export_images(shards, wanted, images_dir, overwrite=args.overwrite)
        records = [
            fd.to_record(row, vocabulary, f"sample/images/{row['id']}.jpg")
            for row in selected
            if str(row["id"]) in exported
        ]
        fd.write_catalog_csv(records, csv_path)
        kv("products written", len(records))
    else:
        banner("STEP 2/3  Fetch images for the committed catalogue")
        rows = list(fd.read_catalog_csv(csv_path))
        wanted = {row["external_id"] for row in rows if row.get("external_id")}
        kv("products in csv", len(rows))
        exported = fd.export_images(shards, wanted, images_dir, overwrite=args.overwrite)
        kv("images available", len(exported))
        if missing := wanted - set(exported):
            kv("images missing", len(missing))
            print(
                f"\n  NOTE: {len(missing)} product(s) have no image and will be indexed on\n"
                "  their text only. This is expected when --shards 1 was used, because the\n"
                "  catalogue spans both shards. Re-run without --shards to fetch them all.",
            )

    banner("STEP 3/3  Verify")
    present = sorted(images_dir.glob("*.jpg")) if images_dir.exists() else []
    total_bytes = sum(p.stat().st_size for p in present)
    kv("image files", len(present))
    kv("total image size", f"{total_bytes / 1024:.0f} KB")
    kv("catalogue csv", display_path(csv_path))

    if not present:
        print(
            "\nERROR: no images were written; the catalogue is not usable.",
            file=sys.stderr,
        )
        return 1

    print("\nNext:")
    print("  python scripts/index_catalog.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

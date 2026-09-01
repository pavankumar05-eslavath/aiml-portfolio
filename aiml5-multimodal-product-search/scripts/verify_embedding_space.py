#!/usr/bin/env python
"""Measure whether the configured model really shares one image/text space.

Image + text fusion is only meaningful if both towers project into a *comparable*
space. That is a property of the training objective, not something to take on
trust, so this script measures it on the actual catalogue and prints the numbers
the ranking layer's design depends on:

1. **Cross-modal retrieval accuracy** - for each product, is its own document the
   highest-scoring text for its own image? If this is near chance, image+text
   fusion cannot work and the model is unsuitable.
2. **The modality gap** - how far same-modality similarities (image-image,
   text-text) sit above cross-modal ones. This is what motivates per-channel
   score normalisation in `app.services.fusion`.
3. **Document template vs raw name** - whether the caption-style document the
   ingestion pipeline builds actually beats the bare catalogue name.

Run it after changing ``MODEL_NAME``:

    python scripts/verify_embedding_space.py
    python scripts/verify_embedding_space.py --sample 64 --json out.json

Exit status is non-zero if the model fails the alignment check, so this can be
used as a gate in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from _bootstrap import banner, configure_script_logging, display_path, kv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Measure image/text embedding-space alignment for the configured model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=48,
        help="Number of catalogue products to probe (needs images on disk).",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Also write the measurements as JSON."
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    """Execute the probe."""
    from app.core.text import build_product_document
    from app.db.session import dispose_engine, get_session_factory, init_models
    from app.models.product import Product
    from app.services.embedding import get_embedding_service
    from app.services.image_processing import get_image_processor
    from sqlalchemy import select

    embedder = get_embedding_service()
    images = get_image_processor()

    banner("MODEL")
    try:
        info = await asyncio.to_thread(embedder.load)
    except Exception as exc:
        print(f"ERROR: could not load the model: {exc}", file=sys.stderr)
        return 1
    kv("name", info.name)
    kv("architecture", info.model_type)
    kv("embedding dim", info.embedding_dim)
    kv("device", f"{info.device} ({info.dtype})")
    kv("input resolution", info.image_size or "unknown")

    await init_models()
    factory = get_session_factory()
    try:
        async with factory() as session:
            rows = list(
                (
                    await session.execute(
                        select(Product)
                        .where(Product.image_url.is_not(None))
                        .limit(args.sample * 4)
                    )
                )
                .scalars()
                .all()
            )

        loaded: list[tuple[Product, object]] = []
        for product in rows:
            reference = product.image_path or product.image_url
            if not reference:
                continue
            path = images.resolve_catalog_path(reference)
            if path is None:
                continue
            try:
                loaded.append((product, images.load_from_path(path).image))
            except Exception:  # noqa: S112 - skip unreadable images
                continue
            if len(loaded) >= args.sample:
                break

        if len(loaded) < 4:
            print(
                "\nERROR: need at least 4 catalogue products with readable images.\n"
                "Run 'python scripts/prepare_sample.py' then "
                "'python scripts/index_catalog.py' first.",
                file=sys.stderr,
            )
            return 1

        products = [product for product, _ in loaded]
        pictures = [image for _, image in loaded]
        documents = [
            build_product_document(
                name=p.name,
                description=p.description,
                category=p.category,
                subcategory=p.subcategory,
                brand=p.brand,
                colour=p.colour,
                gender=p.gender,
                usage=p.usage,
            )
            for p in products
        ]
        names = [p.name for p in products]

        banner(f"ALIGNMENT PROBE  ({len(products)} products)")
        templated = await asyncio.to_thread(embedder.probe_alignment, pictures, documents)
        bare = await asyncio.to_thread(embedder.probe_alignment, pictures, names)

        print("  Cross-modal retrieval: is a product's own text the best match for its image?")
        kv("with the attribute template", f"top-1 = {templated.cross_modal_top1:.3f}", indent=4)
        kv("with the raw catalogue name", f"top-1 = {bare.cross_modal_top1:.3f}", indent=4)
        delta = templated.cross_modal_top1 - bare.cross_modal_top1
        kv(
            "template advantage",
            f"{delta:+.3f}"
            + ("  (template is better)" if delta > 0 else "  (no gain from templating)"),
            indent=4,
        )

        print("\n  Matched vs mismatched pairs (a shared space requires matched > mismatched):")
        kv("matched pair similarity", f"{templated.matched_pair_similarity:.4f}", indent=4)
        kv(
            "mismatched pair similarity",
            f"{templated.mismatched_pair_similarity:.4f}",
            indent=4,
        )
        kv(
            "separation",
            f"{templated.matched_pair_similarity - templated.mismatched_pair_similarity:+.4f}",
            indent=4,
        )

        print("\n  The modality gap (this is what per-channel normalisation compensates for):")
        kv("image-image similarity", f"{templated.image_image_similarity:.4f}", indent=4)
        kv("text-text similarity", f"{templated.text_text_similarity:.4f}", indent=4)
        kv("image-text similarity", f"{templated.matched_pair_similarity:.4f}", indent=4)
        gap = templated.image_image_similarity - templated.matched_pair_similarity
        kv("gap (image-image minus image-text)", f"{gap:+.4f}", indent=4)

        banner("VERDICT")
        aligned = templated.is_aligned
        if aligned:
            print(
                "  PASS: the two towers share a comparable space; multimodal fusion is valid."
            )
        else:
            print("  FAIL: image and text embeddings are NOT usefully aligned for this model.")
            print("        Multimodal fusion would be meaningless. Choose a dual-encoder model")
            print("        trained with a contrastive/sigmoid image-text objective (CLIP,")
            print("        OpenCLIP, SigLIP).")

        if gap > 0.2:
            print(
                f"\n  NOTE: a large modality gap ({gap:+.3f}) means raw score fusion would be\n"
                "        dominated by the same-modality channel. Keep SCORE_NORMALIZATION set\n"
                "        to zscore or minmax so the configured weights mean what they say."
            )

        if args.json:
            payload = {
                "model": info.name,
                "embedding_dim": info.embedding_dim,
                "device": info.device,
                "products_probed": len(products),
                "templated": {
                    "cross_modal_top1": templated.cross_modal_top1,
                    "matched_pair_similarity": templated.matched_pair_similarity,
                    "mismatched_pair_similarity": templated.mismatched_pair_similarity,
                    "image_image_similarity": templated.image_image_similarity,
                    "text_text_similarity": templated.text_text_similarity,
                },
                "raw_name": {"cross_modal_top1": bare.cross_modal_top1},
                "modality_gap": gap,
                "aligned": aligned,
            }
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\n  wrote {display_path(args.json)}")

        return 0 if aligned else 2
    finally:
        await dispose_engine()


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    configure_script_logging(args.log_level)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

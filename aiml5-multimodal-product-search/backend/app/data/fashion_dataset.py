"""Ingestion adapter for the fashion product images dataset.

Source
------
``ashraq/fashion-product-images-small`` on the Hugging Face Hub - a public,
no-authentication mirror of the "Fashion Product Images (Small)" catalogue
(~44,000 products, 60x80 JPEGs, ~271 MB as two Parquet shards).

Why this dataset
----------------
* Public and downloadable without credentials, so the project is reproducible.
* Every product has **both** an image and structured attributes, which is what a
  multimodal system needs; image-only or text-only corpora cannot exercise it.
* Its attribute taxonomy (article type, colour, gender, usage) supports
  *attribute-derived relevance judgements*, which is what makes quantitative
  evaluation possible without hand-labelling thousands of pairs.
* Small images keep CPU-only indexing tractable - a deliberate trade-off, since
  60x80 sources limit achievable visual precision. This is recorded as a known
  limitation in the README rather than glossed over.

Memory behaviour
----------------
Image bytes are streamed with :meth:`pyarrow.parquet.ParquetFile.iter_batches`
rather than read as a column, because the image column alone is ~500 MB
decompressed. Metadata is read first as a cheap columns-only pass, which is also
what lets brand mining see the whole catalogue before any image is touched.
"""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from PIL import Image

from app.core.logging import get_logger
from app.data.enrichment import BrandVocabulary, build_description, synthesise_price

logger = get_logger(__name__)

HF_REPO_ID: Final = "ashraq/fashion-product-images-small"
HF_PARQUET_FILES: Final = (
    "data/train-00000-of-00002-6cff4c59f91661c3.parquet",
    "data/train-00001-of-00002-bb459e5ac5f01e71.parquet",
)

METADATA_COLUMNS: Final = (
    "id",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "year",
    "usage",
    "productDisplayName",
)

#: Column order of the canonical catalogue CSV consumed by the indexer.
CATALOG_COLUMNS: Final = (
    "external_id",
    "name",
    "description",
    "category",
    "subcategory",
    "brand",
    "colour",
    "gender",
    "usage",
    "season",
    "year",
    "price",
    "currency",
    "image_url",
    "in_stock",
)


@dataclass(slots=True)
class CatalogRecord:
    """One row of the canonical catalogue CSV.

    Field provenance: ``brand`` is extracted from the display name,
    ``description`` is templated from real attributes, and ``price`` is
    **synthetic**. See :mod:`app.data.enrichment`.
    """

    external_id: str
    name: str
    description: str
    category: str
    subcategory: str
    brand: str
    colour: str
    gender: str
    usage: str
    season: str
    year: str
    price: str
    currency: str
    image_url: str
    in_stock: str = "true"


def download_parquet_shards(cache_dir: Path, shards: Sequence[str] | None = None) -> list[Path]:
    """Download the dataset's Parquet shards from the Hub.

    Args:
        cache_dir: Directory to download into.
        shards: Shard filenames; defaults to both.

    Returns:
        Local paths to the downloaded shards.
    """
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename in shards or HF_PARQUET_FILES:
        logger.info("downloading shard", extra={"context": {"file": filename}})
        local = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
        paths.append(Path(local))
    return paths


def read_metadata(shards: Sequence[Path]) -> list[dict[str, Any]]:
    """Read every row's metadata, without touching image bytes."""
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for shard in shards:
        table = pq.read_table(shard, columns=list(METADATA_COLUMNS))
        rows.extend(table.to_pylist())
    logger.info("read dataset metadata", extra={"context": {"rows": len(rows)}})
    return rows


def learn_brands(rows: Sequence[dict[str, Any]], min_count: int = 3) -> BrandVocabulary:
    """Mine the brand vocabulary from every display name in the dataset.

    Learning from the *whole* catalogue even when only a sample will be ingested
    gives the frequency evidence its accuracy depends on.
    """
    attribute_values = [
        row[column]
        for row in rows
        for column in (
            "gender",
            "masterCategory",
            "subCategory",
            "articleType",
            "baseColour",
            "season",
            "usage",
        )
    ]
    vocabulary = BrandVocabulary.learn(
        (row["productDisplayName"] or "" for row in rows),
        attribute_values=attribute_values,
        min_count=min_count,
    )
    logger.info(
        "learned brand vocabulary",
        extra={"context": {"candidates": len(vocabulary.counts)}},
    )
    return vocabulary


def is_usable(row: dict[str, Any]) -> bool:
    """Whether a source row has the minimum fields needed to be a product."""
    return bool(
        row.get("id") is not None
        and (row.get("productDisplayName") or "").strip()
        and (row.get("articleType") or "").strip()
    )


def select_stratified(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int,
    stratify_by: str = "articleType",
    seed: int = 17,
) -> list[dict[str, Any]]:
    """Pick a diverse subset, spread evenly across a categorical column.

    A head-of-file slice would be dominated by whichever article types happen to
    appear first, which would make both the demo and the evaluation unrepresentative.
    Round-robin sampling across groups keeps every article type present, which
    matters because the evaluation queries target specific article types.

    Args:
        rows: Candidate rows.
        limit: Maximum rows to return.
        stratify_by: Column defining the groups.
        seed: Shuffle seed, for reproducible selection.

    Returns:
        Up to ``limit`` rows, ordered by group so the sample stays balanced.
    """
    import random

    usable = [row for row in rows if is_usable(row)]
    if limit >= len(usable):
        return usable

    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        groups[row.get(stratify_by) or "unknown"].append(row)

    # Reproducible sampling, not security: a seeded PRNG is exactly what is wanted.
    rng = random.Random(seed)  # noqa: S311
    for members in groups.values():
        rng.shuffle(members)

    # Visit the largest groups first so a limit smaller than the group count still
    # yields the most representative article types.
    order = sorted(groups, key=lambda key: len(groups[key]), reverse=True)
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for key in order:
            if depth < len(groups[key]):
                selected.append(groups[key][depth])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        depth += 1

    logger.info(
        "selected stratified sample",
        extra={"context": {"rows": len(selected), "groups": len(groups)}},
    )
    return selected


def to_record(
    row: dict[str, Any], vocabulary: BrandVocabulary, image_path: str
) -> CatalogRecord:
    """Convert one source row into a canonical catalogue record."""
    display_name = (row["productDisplayName"] or "").strip()
    external_id = str(row["id"])
    article_type = (row.get("articleType") or "").strip()
    brand = vocabulary.extract(display_name) or ""
    year_value = row.get("year")
    year = str(int(year_value)) if isinstance(year_value, (int, float)) and year_value else ""

    description = build_description(
        display_name=display_name,
        article_type=article_type,
        subcategory=(row.get("subCategory") or "").strip(),
        master_category=(row.get("masterCategory") or "").strip(),
        colour=(row.get("baseColour") or "").strip(),
        gender=(row.get("gender") or "").strip(),
        usage=(row.get("usage") or "").strip(),
        season=(row.get("season") or "").strip(),
        year=int(year) if year else None,
        brand=brand,
    )
    price = synthesise_price(
        external_id=external_id,
        article_type=article_type,
        usage=(row.get("usage") or "").strip(),
    )

    return CatalogRecord(
        external_id=external_id,
        name=display_name,
        description=description,
        # masterCategory is the coarse filter facet (Apparel, Footwear, ...);
        # articleType is the precise one (Sports Shoes, Watches, ...). The middle
        # subCategory level is folded into the description instead of a column.
        category=(row.get("masterCategory") or "").strip(),
        subcategory=article_type,
        brand=brand,
        colour=(row.get("baseColour") or "").strip(),
        gender=(row.get("gender") or "").strip(),
        usage=(row.get("usage") or "").strip(),
        season=(row.get("season") or "").strip(),
        year=year,
        price=f"{price:.2f}",
        currency="USD",
        image_url=image_path,
        in_stock="true",
    )


def export_images(
    shards: Sequence[Path],
    wanted_ids: set[str],
    images_dir: Path,
    *,
    min_edge: int = 0,
    overwrite: bool = False,
    batch_size: int = 256,
) -> dict[str, str]:
    """Extract the selected products' images to JPEG files.

    Streams row batches so peak memory stays proportional to ``batch_size``
    rather than to the dataset.

    Args:
        shards: Parquet shard paths.
        wanted_ids: ``id`` values (as strings) to export.
        images_dir: Output directory.
        min_edge: If > 0, upscale images whose shortest edge is smaller. Left at 0
            by default: the model's own processor resizes to its input resolution,
            and pre-upscaling would only inflate files on disk.
        overwrite: Re-write files that already exist.
        batch_size: Parquet batch size.

    Returns:
        ``{external_id: relative_path}`` for every image written or already present.
    """
    import pyarrow.parquet as pq

    images_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    remaining = set(wanted_ids)
    failures = 0

    for shard in shards:
        if not remaining:
            break
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["id", "image"]):
            if not remaining:
                break
            records = batch.to_pylist()
            for record in records:
                external_id = str(record["id"])
                if external_id not in remaining:
                    continue
                target = images_dir / f"{external_id}.jpg"
                relative = f"{images_dir.name}/{target.name}"
                if target.exists() and not overwrite:
                    remaining.discard(external_id)
                    written[external_id] = relative
                    continue
                try:
                    payload = record["image"]
                    raw = payload["bytes"] if isinstance(payload, dict) else payload
                    with Image.open(io.BytesIO(raw)) as image:
                        image = image.convert("RGB")
                        if min_edge and min(image.size) < min_edge:
                            scale = min_edge / min(image.size)
                            image = image.resize(
                                (round(image.width * scale), round(image.height * scale)),
                                Image.Resampling.BICUBIC,
                            )
                        image.save(target, format="JPEG", quality=92)
                except Exception:
                    failures += 1
                    logger.warning(
                        "failed to export image",
                        extra={"context": {"external_id": external_id}},
                    )
                    remaining.discard(external_id)
                    continue
                written[external_id] = relative
                remaining.discard(external_id)

    logger.info(
        "exported images",
        extra={
            "context": {
                "written": len(written),
                "missing": len(remaining),
                "failures": failures,
            }
        },
    )
    return written


def write_catalog_csv(records: Sequence[CatalogRecord], destination: Path) -> Path:
    """Write canonical catalogue records to CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CATALOG_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    logger.info(
        "wrote catalogue csv",
        extra={"context": {"path": str(destination), "rows": len(records)}},
    )
    return destination


def summarise(records: Sequence[CatalogRecord]) -> dict[str, Any]:
    """Summarise a catalogue for reporting after a build."""
    prices = [float(r.price) for r in records if r.price]
    return {
        "products": len(records),
        "categories": len({r.category for r in records if r.category}),
        "article_types": len({r.subcategory for r in records if r.subcategory}),
        "brands": len({r.brand for r in records if r.brand}),
        "with_brand": sum(1 for r in records if r.brand),
        "colours": len({r.colour for r in records if r.colour}),
        "price_min": round(min(prices), 2) if prices else None,
        "price_max": round(max(prices), 2) if prices else None,
        "top_article_types": Counter(
            r.subcategory for r in records if r.subcategory
        ).most_common(10),
    }


def read_catalog_csv(path: Path) -> Iterator[dict[str, str]]:
    """Stream rows from a canonical catalogue CSV.

    Raises:
        ValueError: if required columns are missing, so a malformed file fails
            immediately rather than producing thousands of empty products.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"external_id", "name"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {', '.join(sorted(missing))}"
            )
        yield from reader

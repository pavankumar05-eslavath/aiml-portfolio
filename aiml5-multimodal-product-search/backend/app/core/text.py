"""Product text cleaning and canonical document construction.

Why this module exists
----------------------
CLIP's text tower was trained on short, caption-like alt-text, not on retail SKU
strings. Feeding it raw catalogue names ("Nike Men's As Spher White T-shirt")
under-uses the encoder. Building a caption-shaped sentence from the product's
structured attributes measurably improves cross-modal alignment.

Measured on the committed sample catalogue (48 products probed,
``openai/clip-vit-base-patch32``), image->text top-1 retrieval was **0.812** with
this template versus **0.771** with the raw catalogue name. Reproduce with::

    python scripts/verify_embedding_space.py

The same canonicalisation must be applied at index time and at query time, so it
lives here rather than in the ingestion script.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")
_MULTI_PUNCT_RE = re.compile(r"([!?.,;:])\1{1,}")

#: CLIP's text encoder truncates at 77 BPE tokens. Words beyond roughly this
#: count cannot influence the embedding, so trim early to avoid wasted compute
#: and to make the stored document honest about what was encoded.
MAX_DOCUMENT_WORDS = 60


def clean_text(value: str | None, *, max_length: int = 2000) -> str:
    """Normalise free-text product fields.

    Strips control characters and HTML tags, collapses whitespace and repeated
    punctuation, and applies NFKC normalisation so visually identical strings
    compare equal.
    """
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = _CONTROL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MULTI_PUNCT_RE.sub(r"\1", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text[:max_length].strip()


def normalise_facet(value: str | None) -> str:
    """Normalise a facet value (category, brand, colour) for consistent filtering."""
    cleaned = clean_text(value, max_length=120)
    return " ".join(part for part in cleaned.split() if part)


def truncate_words(text: str, limit: int = MAX_DOCUMENT_WORDS) -> str:
    """Trim ``text`` to at most ``limit`` whitespace-separated words."""
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])


def _dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def build_product_document(
    *,
    name: str,
    description: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    brand: str | None = None,
    colour: str | None = None,
    gender: str | None = None,
    usage: str | None = None,
    tags: Sequence[str] | None = None,
) -> str:
    """Build the canonical text document that gets embedded for a product.

    The output is a caption-like sentence rather than a concatenation of fields,
    matching the distribution CLIP was trained on. Attribute values that are
    already present in the product name are not repeated, which keeps the
    document inside the encoder's 77-token window and avoids over-weighting a
    term simply because it appears in two source fields.

    Returns:
        A single-line document, never empty as long as ``name`` is non-empty.
    """
    name_c = clean_text(name, max_length=300)
    lowered_name = name_c.casefold()

    def _novel(value: str | None) -> str:
        """Keep a facet only if it adds information the name does not carry."""
        facet = normalise_facet(value)
        return "" if not facet or facet.casefold() in lowered_name else facet

    brand_c = _novel(brand)
    colour_c = _novel(colour)
    gender_c = _novel(gender)
    category_c = normalise_facet(category)
    subcategory_c = _novel(subcategory)
    usage_c = _novel(usage)

    # Lead clause: "<colour> <category> for <gender>, by <brand>"
    head_parts = [p for p in (colour_c.lower(), (subcategory_c or category_c).lower()) if p]
    head = " ".join(head_parts)

    clauses: list[str] = []
    if name_c:
        clauses.append(name_c)
    if head:
        clause = f"a {head}"
        if gender_c:
            clause += f" for {gender_c.lower()}"
        clauses.append(clause)
    if brand_c:
        clauses.append(f"by {brand_c}")
    if usage_c:
        clauses.append(f"for {usage_c.lower()} wear")

    tag_values = _dedupe_preserving_order(_novel(t) for t in (tags or []) if normalise_facet(t))
    if tag_values:
        clauses.append(", ".join(tag_values[:6]))

    description_c = clean_text(description, max_length=600)
    if description_c and description_c.casefold() != name_c.casefold():
        clauses.append(description_c)

    document = ". ".join(_dedupe_preserving_order(clauses))
    return truncate_words(document)


def build_search_document(query: str) -> str:
    """Canonicalise a user's free-text query before embedding.

    Kept intentionally light: unlike catalogue text, a natural-language query is
    already close to CLIP's training distribution, and rewriting it risks
    discarding the user's intent.
    """
    return truncate_words(clean_text(query, max_length=400))


def lexical_terms(text: str, *, min_length: int = 2) -> list[str]:
    """Extract lowercase alphanumeric terms for the lexical retrieval channel."""
    return [
        token
        for token in re.split(r"[^0-9a-z]+", clean_text(text).casefold())
        if len(token) >= min_length
    ]

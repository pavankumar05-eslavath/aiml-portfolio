"""Deriving catalogue fields the source dataset does not provide.

Provenance, stated plainly
--------------------------
The source dataset (``ashraq/fashion-product-images-small``) provides real
attributes - ``gender``, ``masterCategory``, ``subCategory``, ``articleType``,
``baseColour``, ``season``, ``year``, ``usage``, ``productDisplayName`` - and a
product image. It does **not** provide a brand field, a description, or a price.
This module is the single place where those three are derived, and each is
treated differently according to how well founded it is:

======================  ==========  ==========================================
field                   status      how it is obtained
======================  ==========  ==========================================
``brand``               extracted   Mined from ``productDisplayName``, which in
                                    this dataset begins with the brand
                                    ("Nike Men's As Spher White T-shirt").
``description``         derived     Templated from the product's real
                                    attributes. States no new facts.
``price``               SYNTHETIC   Deterministically generated. Not real data.
======================  ==========  ==========================================

Price exists only so that price filtering and price sorting are demonstrable. It
is seeded and reproducible, and is labelled as synthetic in ``data/README.md``
and in the README. It must never be presented as a measured quantity, and no
evaluation query depends on it.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

from app.core.text import clean_text

#: Tokens that join the words of a multi-word brand ("Gini and Jony").
_CONNECTORS: Final = frozenset({"and", "of", "&", "the", "by", "de", "la"})

#: Leading words that are never part of a brand. Kept deliberately short:
#: attribute words are already excluded via the dataset's own vocabulary, and an
#: over-broad list rejects real brands ("Classic Polo", "New Balance").
_NON_BRAND_LEADING: Final = frozenset(
    {
        "the",
        "a",
        "an",
        "men",
        "mens",
        "men's",
        "women",
        "womens",
        "women's",
        "boys",
        "boy's",
        "girls",
        "girl's",
        "kids",
        "unisex",
        "baby",
        "his",
        "her",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9&'\.\-]+")

#: Maximum tokens considered for a brand name.
MAX_BRAND_TOKENS: Final = 4


def _trim_connectors(brand: str) -> str | None:
    """Drop trailing connector words left by a failed extension."""
    parts = brand.split()
    while parts and parts[-1].casefold() in _CONNECTORS:
        parts.pop()
    return " ".join(parts) or None


@dataclass(slots=True)
class BrandVocabulary:
    """Brand names mined from a catalogue's display names.

    Rather than hardcoding a brand list (which would silently mislabel any other
    dataset), the vocabulary is *learned* from the data: leading n-grams that
    recur across many distinct products are brands, and attribute words the
    dataset itself declares (colours, genders, article types) are excluded.
    """

    counts: Counter[str] = field(default_factory=Counter)
    attribute_words: set[str] = field(default_factory=set)
    min_count: int = 3

    @classmethod
    def learn(
        cls,
        display_names: Iterable[str],
        *,
        attribute_values: Iterable[str] = (),
        min_count: int = 3,
    ) -> BrandVocabulary:
        """Mine candidate brands from display names.

        Args:
            display_names: Raw product display names.
            attribute_values: Every value appearing in the dataset's attribute
                columns (colours, genders, article types, seasons, usages).
                Their individual words are excluded from brand candidates.
            min_count: How many distinct products a candidate must lead before it
                is accepted as a brand.
        """
        attribute_words = {
            word.casefold()
            for value in attribute_values
            for word in _TOKEN_RE.findall(str(value or ""))
            if word
        }
        counts: Counter[str] = Counter()
        for name in display_names:
            tokens = _TOKEN_RE.findall(clean_text(name))
            for size in range(1, MAX_BRAND_TOKENS + 1):
                if len(tokens) < size:
                    break
                counts[" ".join(tokens[:size]).casefold()] += 1
        return cls(counts=counts, attribute_words=attribute_words, min_count=min_count)

    def _is_candidate(self, tokens: Sequence[str]) -> bool:
        """Whether a leading n-gram could be a brand."""
        if not tokens:
            return False
        first = tokens[0].casefold()
        if first in _NON_BRAND_LEADING or first in self.attribute_words:
            return False
        # A single non-alphabetic leading character is never a brand.
        return not (len(first) < 2 and not first.isalpha())

    def extract(self, display_name: str) -> str | None:
        """Extract the brand from one display name.

        Strategy: take the shortest frequent leading n-gram, then extend it while
        the extension is either a connector word or is itself nearly as frequent
        as the base. This keeps genuine multi-word brands together ("Gini and
        Jony", where every "Gini" product is a "Gini and Jony" product) without
        swallowing a descriptive second word ("Turtle Check ...", where "Check"
        follows "Turtle" only occasionally).

        Returns:
            The brand in its original casing, or ``None`` when no candidate
            clears ``min_count``.
        """
        tokens = _TOKEN_RE.findall(clean_text(display_name))
        if not self._is_candidate(tokens):
            return None

        base_count = self.counts.get(tokens[0].casefold(), 0)
        if base_count < self.min_count:
            # Too rare for frequency evidence (a brand appearing once or twice).
            # Fall back to orthography: a leading run of capitalised, non-attribute
            # tokens is almost always a proper noun, i.e. the brand.
            return self._extract_by_capitalisation(tokens)

        size = 1
        while size < min(len(tokens), MAX_BRAND_TOKENS):
            nxt = tokens[size].casefold()
            if nxt in self.attribute_words and nxt not in _CONNECTORS:
                break
            candidate = " ".join(tokens[: size + 1]).casefold()
            candidate_count = self.counts.get(candidate, 0)
            if nxt in _CONNECTORS:
                # A connector alone is not a brand boundary; look one further.
                if size + 1 < len(tokens):
                    extended = " ".join(tokens[: size + 2]).casefold()
                    if self.counts.get(extended, 0) >= self.min_count:
                        size += 2
                        continue
                break
            if candidate_count >= max(self.min_count, 0.5 * base_count):
                size += 1
                continue
            break

        return _trim_connectors(" ".join(tokens[:size]))

    def _extract_by_capitalisation(self, tokens: Sequence[str]) -> str | None:
        """Fallback for brands too rare to have frequency support.

        Takes the leading run of capitalised tokens, stopping at the first
        attribute word. This recovers names such as "Yves Saint Laurent Men Body
        Kouros Perfume" -> "Yves Saint Laurent", which frequency mining misses
        because the brand appears only a handful of times.
        """
        if not tokens or not tokens[0][:1].isupper():
            return None
        taken: list[str] = []
        for token in tokens[:MAX_BRAND_TOKENS]:
            folded = token.casefold()
            if folded in self.attribute_words or folded in _NON_BRAND_LEADING:
                break
            if not token[:1].isupper() and folded not in _CONNECTORS:
                break
            taken.append(token)
        return _trim_connectors(" ".join(taken))


#: Indicative price band per article type, in USD: ``(low, high)``.
#: Hand-set from broad retail intuition purely to keep synthetic prices plausible
#: enough for filter and sort demonstrations. NOT market data.
_PRICE_BANDS: Final[dict[str, tuple[float, float]]] = {
    "watches": (45.0, 320.0),
    "sports shoes": (45.0, 165.0),
    "casual shoes": (35.0, 130.0),
    "formal shoes": (55.0, 210.0),
    "heels": (30.0, 120.0),
    "flats": (22.0, 85.0),
    "sandals": (18.0, 70.0),
    "flip flops": (8.0, 32.0),
    "handbags": (30.0, 190.0),
    "backpacks": (25.0, 110.0),
    "duffel bag": (30.0, 120.0),
    "wallets": (18.0, 95.0),
    "belts": (15.0, 65.0),
    "sunglasses": (25.0, 180.0),
    "jackets": (55.0, 240.0),
    "sweaters": (35.0, 120.0),
    "sweatshirts": (30.0, 95.0),
    "jeans": (35.0, 130.0),
    "trousers": (30.0, 110.0),
    "shorts": (18.0, 60.0),
    "track pants": (25.0, 80.0),
    "capris": (20.0, 65.0),
    "shirts": (25.0, 95.0),
    "tshirts": (12.0, 48.0),
    "tops": (15.0, 58.0),
    "dresses": (28.0, 145.0),
    "kurtas": (20.0, 85.0),
    "kurtis": (18.0, 75.0),
    "sarees": (35.0, 220.0),
    "skirts": (20.0, 75.0),
    "socks": (5.0, 20.0),
    "caps": (10.0, 38.0),
    "scarves": (12.0, 55.0),
    "jewellery set": (25.0, 160.0),
    "earrings": (12.0, 90.0),
    "necklace and chains": (18.0, 130.0),
    "bracelet": (12.0, 80.0),
    "ring": (15.0, 110.0),
    "deodorant": (8.0, 30.0),
    "perfume and body mist": (25.0, 120.0),
    "lipstick": (10.0, 42.0),
    "nail polish": (5.0, 22.0),
    "fragrance gift set": (35.0, 150.0),
}

#: Fallback band when the article type is unknown.
_DEFAULT_BAND: Final[tuple[float, float]] = (15.0, 95.0)

#: Multipliers nudging price by usage, so filters interact with real attributes.
_USAGE_MULTIPLIER: Final[dict[str, float]] = {
    "formal": 1.25,
    "sports": 1.1,
    "party": 1.2,
    "ethnic": 1.15,
    "casual": 1.0,
    "smart casual": 1.1,
    "travel": 1.05,
    "home": 0.9,
}


def synthesise_price(
    *,
    external_id: str,
    article_type: str | None,
    usage: str | None = None,
    seed: str = "mps-v1",
) -> float:
    """Generate a deterministic, plausible price for a product.

    **This is synthetic data.** The source dataset has no price column. The value
    is a hash-derived point inside an article-type band, adjusted by usage, so
    that price filtering and sorting can be demonstrated and evaluated for
    *correctness* (does the filter exclude the right rows?) without ever implying
    the number is real.

    Deterministic by construction: the same product always receives the same
    price, so re-running ingestion does not churn the catalogue and evaluation
    runs stay comparable.

    Args:
        external_id: Stable product identifier; drives the hash.
        article_type: Selects the price band.
        usage: Applies a modest multiplier.
        seed: Change to generate a different but still reproducible assignment.

    Returns:
        Price rounded to a retail-looking ``.99``/``.95``/``.49`` ending.
    """
    low, high = _PRICE_BANDS.get((article_type or "").strip().casefold(), _DEFAULT_BAND)
    digest = hashlib.sha256(f"{seed}\x00{external_id}".encode()).digest()
    # Two bytes give 1/65536 granularity: ample for a price band.
    fraction = int.from_bytes(digest[:2], "big") / 65535.0
    price = low + fraction * (high - low)
    price *= _USAGE_MULTIPLIER.get((usage or "").strip().casefold(), 1.0)

    endings = (0.99, 0.95, 0.49)
    ending = endings[digest[2] % len(endings)]
    return round(max(1.0, int(price)) + ending, 2)


#: Article types that are inherently plural in English and must not be
#: singularised ("a pair of casual shoes", never "a casual shoe").
_PAIR_NOUNS: Final = (
    "shoes",
    "flops",
    "sandals",
    "heels",
    "flats",
    "jeans",
    "shorts",
    "trousers",
    "socks",
    "sunglasses",
    "glasses",
    "pants",
    "capris",
    "leggings",
    "tights",
    "sneakers",
    "loafers",
    "slippers",
    "stockings",
    "clogs",
    "wedges",
    "mules",
    "booties",
    "boots",
    "gloves",
    "earrings",
    "cufflinks",
)


def _singularise(noun: str) -> str:
    """Best-effort singular form of an article-type noun.

    Deliberately a small rule set rather than an inflection library: the article
    types in this dataset are a closed vocabulary of ~140 values, and the goal is
    only a fluent caption, not linguistic completeness.
    """
    lowered = noun.casefold()
    if lowered.endswith(_PAIR_NOUNS):
        return noun
    if lowered.endswith("ies") and len(noun) > 4:
        return noun[:-3] + "y"
    if lowered.endswith(("ses", "xes", "zes", "ches", "shes")):
        return noun[:-2]
    if lowered.endswith("s") and not lowered.endswith(("ss", "us", "is")):
        return noun[:-1]
    return noun


def _indefinite_article(phrase: str) -> str:
    """Return "a" or "an" to suit the following phrase."""
    return "an" if phrase[:1].casefold() in "aeiou" else "a"


def build_description(
    *,
    display_name: str,
    article_type: str | None = None,
    subcategory: str | None = None,
    master_category: str | None = None,
    colour: str | None = None,
    gender: str | None = None,
    usage: str | None = None,
    season: str | None = None,
    year: int | None = None,
    brand: str | None = None,
) -> str:
    """Compose a product description from the dataset's real attributes.

    Every clause restates a value present in the source data - no facts are
    invented. The purpose is to give the text encoder a fluent, caption-like
    passage instead of a terse SKU string, which measurably improves cross-modal
    alignment (see :mod:`app.core.text`).
    """
    article = clean_text(article_type) or clean_text(subcategory) or "product"
    colour_c = clean_text(colour)
    gender_c = clean_text(gender)
    usage_c = clean_text(usage)
    season_c = clean_text(season)
    brand_c = clean_text(brand)
    category_c = clean_text(master_category)

    noun = _singularise(article).lower()
    is_pair = noun.casefold().endswith(_PAIR_NOUNS)
    lead = f"{colour_c.lower()} {noun}".strip() if colour_c else noun
    determiner = "a pair of" if is_pair else _indefinite_article(lead)

    opening = f"{clean_text(display_name)} is {determiner} {lead}"
    if brand_c:
        opening += f" from {brand_c}"
    sentences = [f"{opening}."]

    audience = []
    if gender_c and gender_c.casefold() != "unisex":
        audience.append(f"designed for {gender_c.lower()}")
    elif gender_c:
        audience.append("a unisex design")
    if usage_c:
        audience.append(f"suited to {usage_c.lower()} wear")
    if audience:
        sentences.append(f"It is {' and '.join(audience)}.")

    context = []
    if season_c:
        context.append(f"{season_c.lower()} season")
    if year:
        context.append(str(year))
    if context:
        sentences.append(f"Part of the {' '.join(context)} range.")

    if category_c and category_c.casefold() not in lead.casefold():
        sentences.append(f"Listed under {category_c}.")

    return " ".join(sentences)

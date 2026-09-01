"""Tests for text canonicalisation and derived catalogue fields.

The document builder is worth testing carefully: it decides what the text encoder
actually sees, and a regression there silently degrades retrieval quality without
any error surfacing.
"""

from __future__ import annotations

import pytest

from app.core.text import (
    MAX_DOCUMENT_WORDS,
    build_product_document,
    build_search_document,
    clean_text,
    lexical_terms,
    normalise_facet,
    truncate_words,
)
from app.data.enrichment import (
    BrandVocabulary,
    build_description,
    synthesise_price,
)


class TestCleanText:
    def test_collapses_whitespace(self):
        assert clean_text("  a   b \n\t c  ") == "a b c"

    def test_strips_html_tags(self):
        assert clean_text("<p>Nice <b>shoe</b></p>") == "Nice shoe"

    def test_strips_control_characters(self):
        assert clean_text("a\x00b\x07c") == "a b c"

    def test_collapses_repeated_punctuation(self):
        assert clean_text("Wow!!! Really??") == "Wow! Really?"

    def test_applies_unicode_normalisation(self):
        """Visually identical strings must compare equal."""
        assert clean_text("ﬁt") == "fit"

    def test_handles_none_and_empty(self):
        assert clean_text(None) == ""
        assert clean_text("") == ""

    def test_respects_max_length(self):
        assert len(clean_text("x" * 500, max_length=100)) == 100

    def test_preserves_meaningful_characters(self):
        assert clean_text("U.S. Polo Assn. 32-inch") == "U.S. Polo Assn. 32-inch"


class TestNormaliseFacet:
    def test_trims_and_collapses(self):
        assert normalise_facet("  Sports   Shoes ") == "Sports Shoes"

    def test_empty_becomes_empty(self):
        assert normalise_facet(None) == ""


class TestTruncateWords:
    def test_leaves_short_text_alone(self):
        assert truncate_words("one two three", 5) == "one two three"

    def test_truncates_to_the_limit(self):
        assert truncate_words("a b c d e", 3) == "a b c"


class TestBuildProductDocument:
    def test_includes_the_product_name(self):
        document = build_product_document(name="Nike Air Zoom")
        assert "Nike Air Zoom" in document

    def test_incorporates_attributes(self):
        document = build_product_document(
            name="Air Zoom Pegasus",
            category="Footwear",
            subcategory="Sports Shoes",
            brand="Nike",
            colour="Black",
            gender="Men",
            usage="Sports",
        )
        lowered = document.lower()
        assert "black" in lowered
        assert "sports shoes" in lowered
        assert "nike" in lowered
        assert "men" in lowered
        assert "sports wear" in lowered

    def test_does_not_repeat_attributes_already_in_the_name(self):
        """Repeating a term would over-weight it and waste the 77-token window."""
        document = build_product_document(
            name="Nike Men Black Sports Shoes",
            brand="Nike",
            colour="Black",
            gender="Men",
            subcategory="Sports Shoes",
        )
        assert document.lower().count("nike") == 1
        assert document.lower().count("black") == 1

    def test_stays_within_the_encoder_window(self):
        """CLIP truncates at 77 BPE tokens; the document must not exceed the budget."""
        document = build_product_document(
            name="A very long product name " * 10,
            description="A long description. " * 60,
            category="Apparel",
            brand="Brand",
            colour="Blue",
        )
        assert len(document.split()) <= MAX_DOCUMENT_WORDS

    def test_is_never_empty_for_a_named_product(self):
        assert build_product_document(name="X").strip()

    def test_omits_a_description_identical_to_the_name(self):
        document = build_product_document(name="Blue Shirt", description="Blue Shirt")
        assert document.lower().count("blue shirt") == 1

    def test_deduplicates_tags(self):
        document = build_product_document(name="Shoe", tags=["running", "Running", "sport"])
        assert document.lower().count("running") == 1

    def test_is_deterministic(self):
        kwargs = {"name": "Item", "brand": "Acme", "colour": "Red", "subcategory": "Caps"}
        assert build_product_document(**kwargs) == build_product_document(**kwargs)


class TestBuildSearchDocument:
    def test_preserves_user_intent(self):
        """Query rewriting risks discarding what the user asked for."""
        query = "black running shoes for daily workouts"
        assert build_search_document(query) == query

    def test_cleans_but_does_not_restructure(self):
        assert build_search_document("  black   shoes  ") == "black shoes"

    def test_truncates_absurdly_long_queries(self):
        assert len(build_search_document("word " * 500).split()) <= MAX_DOCUMENT_WORDS


class TestLexicalTerms:
    def test_extracts_lowercase_alphanumeric_terms(self):
        assert lexical_terms("Nike Men's Black Shoes!") == ["nike", "men", "black", "shoes"]

    def test_drops_single_characters(self):
        assert "a" not in lexical_terms("a big shoe")

    def test_empty_input(self):
        assert lexical_terms("") == []


class TestBrandVocabulary:
    @pytest.fixture
    def vocabulary(self) -> BrandVocabulary:
        # A miniature catalogue with realistic repetition patterns.
        names = (
            ["Nike Men Black Sports Shoes"] * 6
            + ["Nike Women White Running Shoes"] * 4
            + ["Gini and Jony Girls Red Skirt"] * 5
            + ["Gini and Jony Boys Blue Shirt"] * 3
            + ["Turtle Check Men Navy Blue Shirt"] * 2
            + ["Turtle Men Green Shirt"] * 5
            + ["United Colors of Benetton Men Blue Jeans"] * 4
            + ["Yves Saint Laurent Men Kouros Perfume"]
        )
        attributes = [
            "Men",
            "Women",
            "Girls",
            "Boys",
            "Black",
            "White",
            "Red",
            "Blue",
            "Navy Blue",
            "Green",
            "Sports Shoes",
            "Shirts",
            "Skirts",
            "Jeans",
            "Perfume",
            "Casual",
        ]
        return BrandVocabulary.learn(names, attribute_values=attributes, min_count=2)

    def test_extracts_a_single_token_brand(self, vocabulary: BrandVocabulary):
        assert vocabulary.extract("Nike Men Black Sports Shoes") == "Nike"

    def test_keeps_a_multi_word_brand_together(self, vocabulary: BrandVocabulary):
        assert vocabulary.extract("Gini and Jony Girls Red Skirt") == "Gini and Jony"

    def test_handles_a_brand_containing_a_preposition(self, vocabulary: BrandVocabulary):
        assert (
            vocabulary.extract("United Colors of Benetton Men Blue Jeans")
            == "United Colors of Benetton"
        )

    def test_does_not_swallow_a_descriptive_second_word(self, vocabulary: BrandVocabulary):
        """In 'Turtle Check ...' the brand is Turtle; 'Check' describes the pattern."""
        assert vocabulary.extract("Turtle Check Men Navy Blue Shirt") == "Turtle"

    def test_capitalisation_fallback_recovers_rare_brands(self, vocabulary: BrandVocabulary):
        """A brand appearing once has no frequency support, so orthography is used."""
        assert (
            vocabulary.extract("Yves Saint Laurent Men Kouros Perfume") == "Yves Saint Laurent"
        )

    def test_returns_none_when_the_name_starts_with_an_attribute(
        self, vocabulary: BrandVocabulary
    ):
        assert vocabulary.extract("Men Black Sports Shoes") is None

    def test_returns_none_for_empty_input(self, vocabulary: BrandVocabulary):
        assert vocabulary.extract("") is None

    def test_is_case_insensitive_in_matching(self, vocabulary: BrandVocabulary):
        assert vocabulary.extract("NIKE Men Black Sports Shoes") == "NIKE"

    def test_learning_from_no_data_is_safe(self):
        empty = BrandVocabulary.learn([], attribute_values=[])
        assert empty.extract("Anything Here") in (None, "Anything Here")


class TestSynthesisePrice:
    def test_is_deterministic(self):
        """Re-running ingestion must not churn prices."""
        args = {"external_id": "12345", "article_type": "Watches", "usage": "Casual"}
        assert synthesise_price(**args) == synthesise_price(**args)

    def test_differs_between_products(self):
        a = synthesise_price(external_id="1", article_type="Watches")
        b = synthesise_price(external_id="2", article_type="Watches")
        assert a != b

    def test_respects_the_article_type_band(self):
        """Watches should generally cost more than flip flops."""
        watches = [
            synthesise_price(external_id=str(i), article_type="Watches") for i in range(40)
        ]
        flip_flops = [
            synthesise_price(external_id=str(i), article_type="Flip Flops") for i in range(40)
        ]
        assert min(watches) > max(flip_flops)

    def test_is_always_positive(self):
        for i in range(200):
            assert synthesise_price(external_id=str(i), article_type="Socks") > 0

    def test_unknown_article_type_uses_the_default_band(self):
        price = synthesise_price(external_id="x", article_type="Nonexistent Category")
        assert 1.0 < price < 200.0

    def test_usage_multiplier_raises_formal_prices(self):
        formal = synthesise_price(external_id="same", article_type="Shirts", usage="Formal")
        casual = synthesise_price(external_id="same", article_type="Shirts", usage="Casual")
        assert formal > casual

    def test_has_retail_style_endings(self):
        for i in range(30):
            price = synthesise_price(external_id=str(i), article_type="Shirts")
            cents = round(price - int(price), 2)
            assert cents in (0.99, 0.95, 0.49)

    def test_seed_changes_the_assignment(self):
        assert synthesise_price(external_id="1", article_type="Shirts") != synthesise_price(
            external_id="1", article_type="Shirts", seed="different"
        )


class TestBuildDescription:
    def test_states_only_the_supplied_attributes(self):
        description = build_description(
            display_name="Puma Brown Casual Shoes",
            article_type="Casual Shoes",
            colour="Brown",
            gender="Men",
            usage="Casual",
            season="Fall",
            year=2011,
            brand="Puma",
            master_category="Footwear",
        )
        assert "Puma Brown Casual Shoes" in description
        assert "brown" in description.lower()
        assert "men" in description.lower()
        assert "Fall".lower() in description.lower()
        assert "2011" in description

    def test_uses_a_pair_construction_for_plural_nouns(self):
        description = build_description(
            display_name="Puma Brown Casual Shoes", article_type="Casual Shoes", colour="Brown"
        )
        assert "a pair of brown casual shoes" in description.lower()

    def test_singularises_countable_nouns(self):
        description = build_description(
            display_name="Doodle Pink Kidswear", article_type="Tshirts", colour="Pink"
        )
        assert "a pink tshirt" in description.lower()
        assert "tshirts" not in description.lower()

    def test_no_duplicated_determiner(self):
        """Regression: an earlier version produced "Part of the the fall season"."""
        description = build_description(
            display_name="X", article_type="Shirts", season="Fall", year=2012
        )
        assert "the the" not in description.lower()

    def test_uses_an_before_a_vowel(self):
        description = build_description(display_name="X", article_type="Innerwear Vests")
        assert " an " in description.lower() or "a pair of" in description.lower()

    def test_handles_unisex_gender(self):
        description = build_description(
            display_name="X", article_type="Backpacks", gender="Unisex"
        )
        assert "unisex" in description.lower()

    def test_survives_missing_attributes(self):
        assert build_description(display_name="Mystery Item").strip()

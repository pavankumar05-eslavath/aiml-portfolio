r"""
Tests on the CLAIMS, not just the code.

Where a conclusion in INSIGHTS.md depends on a property of the data or on a
cleaning step behaving a specific way, a test pins it here, so a future change
that invalidates the conclusion fails loudly.

Runs against the real CSV when present and the defect-preserving stand-in
otherwise; tests needing the real file are skipped explicitly.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.dataset import (
    CONFIRMED_INR,
    INR_PER_USD_AUG_2019,
    clean,
    fix_amount,
    fix_date,
    fix_stage,
    investor_table,
    is_placeholder,
    is_real,
    load_raw,
    money_subset,
    resolve_path,
    unescape,
)

REAL_ONLY = pytest.mark.skipif(not is_real(), reason="needs the real 3,044-row CSV")


@pytest.fixture(scope="module", autouse=True)
def _ensure_data():
    if not resolve_path().exists():
        subprocess.run([sys.executable, "-m", "data.generate_sample"], check=True)


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def rounds(raw):
    return clean(raw)


@pytest.fixture(scope="module")
def investors(rounds):
    return investor_table(rounds)


# ---------------------------------------------------------------------------
# Defect 1: literal escape text
# ---------------------------------------------------------------------------

def test_escape_text_is_literal_not_real_nbsp():
    """The defect is stored as characters, so the obvious grep misses it.

    This is why the audit exists: `grep -P '\xc2\xa0'` reports a clean file.
    """
    blob = resolve_path().read_bytes()
    assert blob.count(rb"\xc2") > 0, "fixture must contain literal escape text"
    assert blob.count("\u00a0".encode()) == 0, "there is no REAL U+00A0 in this file"


def test_unescape_leaves_no_backslash_debris():
    """Matching a single backslash leaves the other behind -- the bug's second form.

    r'\\xc2\\xa0Gurgaon' must become 'Gurgaon', not '\\ Gurgaon'.
    """
    s = pd.Series([r"\\xc2\\xa0Gurgaon", r"Seed\\nFunding", r"\\xc2\\xa0N/A", "Mumbai"],
                  dtype="string")
    out = unescape(s)
    assert out.tolist() == ["Gurgaon", "Seed Funding", "N/A", "Mumbai"]
    assert not out.str.contains(r"\\", regex=True).any()


def test_cleaned_frame_has_no_backslash_debris(rounds):
    debris = rounds.select_dtypes(include="str").apply(
        lambda c: c.astype("string").str.contains(r"\\", regex=True, na=False).sum()).sum()
    assert debris == 0


def test_unescaping_recovers_otherwise_lost_rows(raw, rounds):
    """Without unescaping these cells fail to parse and vanish from every sum."""
    naive = pd.to_numeric(raw["amount_raw"].str.replace(",", "", regex=False), errors="coerce")
    assert rounds["amount_usd"].notna().sum() > naive.notna().sum()


# ---------------------------------------------------------------------------
# Defect 2: Indian digit grouping
# ---------------------------------------------------------------------------

def test_indian_digit_grouping():
    """'20,00,00,000' is 200 million. A 3-digit-group assumption reads 20 million."""
    got = fix_amount(pd.Series(["20,00,00,000", "3,90,00,00,000", "80,48,394", "1,40,000"],
                               dtype="string"))
    assert got.tolist() == [200_000_000, 3_900_000_000, 8_048_394, 140_000]


def test_amounts_violate_international_grouping(tmp_path: Path):
    """The hazard is a three-digit-group assumption, not pandas specifically.

    pandas' thousands=',' does give the right number -- it strips separators
    without validating grouping -- so it is pinned here as SAFE, to stop anyone
    "fixing" it later. What breaks is any validator or formatter that assumes
    international grouping: 20,00,00,000 has three commas but only nine digits,
    where international grouping would imply ten to twelve.
    """
    f = tmp_path / "a.csv"
    f.write_text('amount\n"20,00,00,000"\n', encoding="utf-8")
    assert pd.read_csv(f, thousands=",")["amount"].iloc[0] == 200_000_000

    intl = r"^\d{1,3}(,\d{3})*$"
    assert not pd.Series(["20,00,00,000"]).str.match(intl).iloc[0]
    assert pd.Series(["200,000,000"]).str.match(intl).iloc[0]


# ---------------------------------------------------------------------------
# Defect 3: currency mislabelling
# ---------------------------------------------------------------------------

def test_inr_row_is_flagged_and_converted(rounds):
    row = rounds[rounds["sr_no"].isin(CONFIRMED_INR)]
    assert len(row) == 1
    r = row.iloc[0]
    assert r["amount_flag"] == "confirmed_inr_not_usd"
    assert r["amount_usd"] == 3_900_000_000, "as-published value must be preserved"
    # INR 390 crore at ~71/USD is ~$55M, the round actually reported in Aug 2019.
    assert 50e6 < r["amount_usd_adj"] < 60e6
    assert r["amount_usd_adj"] == pytest.approx(r["amount_usd"] / INR_PER_USD_AUG_2019)


def test_adjustment_touches_only_the_flagged_row(rounds):
    """Regression guard for a nullable-dtype trap.

    `amount_flag != "confirmed_inr_not_usd"` returns NA (not True) on unflagged
    rows, and .where()/.mask() treat NA as False -- which silently divided EVERY
    amount by 71. The bug is invisible per-row and only shows up in totals.
    """
    present = rounds[rounds["amount_usd"].notna()]
    changed = present[present["amount_usd"] != present["amount_usd_adj"]]
    assert len(changed) == len(CONFIRMED_INR)
    assert set(changed["sr_no"]) == CONFIRMED_INR


def test_na_comparison_really_does_yield_na():
    """Documents the language behaviour the guard above defends against."""
    s = pd.Series(["x", None], dtype="string")
    assert (s != "x").tolist()[1] is pd.NA
    assert s.ne("x").fillna(True).tolist() == [False, True]


@REAL_ONLY
def test_single_bad_cell_is_ten_percent_of_the_total(rounds):
    """The headline: one mislabelled cell carries 10% of the dataset's total."""
    published = rounds["amount_usd"].sum()
    corrected = rounds["amount_usd_adj"].sum()
    assert (published - corrected) / published > 0.09
    assert published == pytest.approx(38.14e9, rel=0.01)
    assert corrected == pytest.approx(34.30e9, rel=0.01)


@REAL_ONLY
def test_uncorrected_data_makes_rapido_the_largest_round(rounds):
    """Left alone, a bike-taxi round outranks Flipkart's genuine $2.5B."""
    naive_top = rounds.nlargest(1, "amount_usd").iloc[0]
    assert "Rapido" in naive_top["startup"]
    fixed_top = money_subset(rounds).nlargest(1, "amount_usd_adj").iloc[0]
    assert fixed_top["startup"] == "Flipkart"


# ---------------------------------------------------------------------------
# Entity resolution: must merge, and must NOT over-merge
# ---------------------------------------------------------------------------

def test_split_entities_are_merged(rounds):
    keys = set(rounds["startup_key"])
    assert "olacabs" not in keys and "flipkartcom" not in keys and "oyorooms" not in keys
    assert {"ola", "flipkart", "oyo"} <= keys


def test_near_miss_names_are_not_merged(rounds):
    """'Ola Electric' is not 'Ola'; 'OYOfit'/'FroyoFit' are not 'OYO'.

    Pinned because prefix or fuzzy matching passes the test above while
    silently inflating the leaderboard here.
    """
    keys = set(rounds["startup_key"])
    for distinct in ("olaelectric", "oyofit", "froyofit", "paytmmarketplace"):
        if distinct in keys:
            assert rounds.loc[rounds["startup_key"] == distinct, "startup"].iloc[0] not in (
                "Ola", "OYO", "Paytm")


def test_bangalore_and_bengaluru_are_one_city(rounds):
    cities = set(rounds["city"])
    assert "Bangalore" not in cities and "Gurgaon" not in cities
    assert "Bengaluru" in cities


def test_city_canonicalisation_reduces_the_count(raw, rounds):
    assert rounds["city"].nunique() < raw["city_raw"].nunique()


# ---------------------------------------------------------------------------
# Investors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "Undisclosed Investors", "Undisclosed Investor", "undisclosed", "Unknown",
    "Group of Angel investors", "Individual Investors", "3 undisclosed HNIs",
    "Existing investors", "Angel Investors",
])
def test_placeholders_are_recognised(name):
    assert is_placeholder(name)


@pytest.mark.parametrize("name", [
    "Sequoia Capital India", "Accel", "Ratan Tata", "Tiger Global", "Blume Ventures",
])
def test_real_investors_are_not_placeholders(name):
    assert not is_placeholder(name)


def test_placeholders_never_reach_the_leaderboard(investors):
    """Left in, 'Undisclosed Investors' ranks as the single most active investor."""
    assert not investors["investor"].map(is_placeholder).any()
    top = investors.groupby("investor_key")["sr_no"].nunique().nlargest(5).index
    assert not any("undisclosed" in k for k in top)


def test_investor_lists_split_on_and_as_well_as_comma():
    """Comma-only splitting invents a firm called 'Saama Capital and Sequoia Capital'."""
    rounds = pd.DataFrame({
        "sr_no": [1], "date": [pd.Timestamp("2016-04-22")], "year": [2016],
        "startup": ["Zoomcar"], "city": ["Bengaluru"], "industry": ["Transport"],
        "stage": ["Series B"], "amount_usd": [11e6], "amount_usd_adj": [11e6],
        "amount_flag": pd.Series([None], dtype="string"),
        "investors": pd.Series(["Saama Capital and Sequoia Capital"], dtype="string"),
    })
    got = set(investor_table(rounds)["investor"])
    assert got == {"Saama Capital", "Sequoia Capital India"}


def test_sequoia_spellings_are_merged(investors):
    keys = set(investors["investor_key"])
    assert "sequoiaindia" not in keys and "sequoiacapital" not in keys


@REAL_ONLY
def test_alias_merging_materially_changes_sequoias_rank(investors):
    """73 deals unmerged vs 122 merged -- the difference between 2nd and 1st."""
    deals = investors[investors["investor_key"] == "sequoiacapitalindia"]["sr_no"].nunique()
    assert deals > 100


# ---------------------------------------------------------------------------
# Parsing and label collapse
# ---------------------------------------------------------------------------

def test_malformed_dates_are_repaired():
    got = fix_date(pd.Series(["12/05.2015", "05/072018", "01/07/015", "22/01//2015",
                              "10/7/2015"], dtype="string"))
    assert got.notna().all()
    assert got.iloc[0] == pd.Timestamp("2015-05-12")
    assert got.iloc[1] == pd.Timestamp("2018-07-05")
    assert got.iloc[2] == pd.Timestamp("2015-07-01")


def test_all_dates_parse_after_cleaning(rounds):
    assert rounds["date"].isna().sum() == 0


def test_stage_labels_collapse():
    got = fix_stage(pd.Series(["Seed Funding", "Seed/ Angel Funding", "Seed / Angel Funding",
                               "Seed Funding", "Seed / Angle Funding", "pre-Series A",
                               "Pre-series A", "Series B", "Private Equity"], dtype="string"))
    assert got.iloc[:5].nunique() == 1
    assert got.iloc[5] == got.iloc[6] == "Pre-Series A"
    assert got.iloc[7] == "Series B"


def test_stage_count_is_reduced(raw, rounds):
    assert rounds["stage"].nunique() < raw["stage_raw"].nunique()


def test_bom_is_stripped(raw):
    assert raw.columns[0] == "sr_no"
    assert not any("\ufeff" in c for c in raw.columns)


# ---------------------------------------------------------------------------
# Coverage: the claim that the post-2017 decline is an artefact
# ---------------------------------------------------------------------------

@REAL_ONLY
def test_row_count_matches_the_published_dataset(rounds):
    assert len(rounds) == 3044


@REAL_ONLY
def test_missingness_is_not_random(rounds):
    """Late stages disclose far more often, so disclosed-only averages are biased."""
    seed = rounds[rounds["stage"] == "Seed / Angel"]["amount_usd"].notna().mean()
    pe = rounds[rounds["stage"] == "Private Equity"]["amount_usd"].notna().mean()
    assert seed < 0.65 < pe


@REAL_ONLY
def test_early_years_match_published_totals_and_late_years_do_not(rounds):
    """The core coverage claim, pinned in both directions.

    2015-2017 track reality; from 2018 the source's collection collapses. If a
    future change makes 2019 look complete, this conclusion is void.
    """
    from src.coverage_check import coverage_table
    t = coverage_table(rounds)
    for y in (2015, 2016, 2017):
        assert abs(t.loc[y, "capital_coverage"] - 1) < 0.20, f"{y} should track reality"
    assert t.loc[2019, "capital_coverage"] < 0.50
    assert t.loc[2019, "deal_coverage"] < 0.25


@REAL_ONLY
def test_median_deal_size_rises_as_coverage_falls(rounds):
    """The tell that the decline is selection, not market: survivors are bigger."""
    med = money_subset(rounds).groupby("year")["amount_usd_adj"].median()
    assert med.loc[2019] > 5 * med.loc[2016]


@REAL_ONLY
def test_metro_concentration(rounds):
    metros = ["Bengaluru", "Delhi", "Gurugram", "Noida", "Mumbai"]
    assert rounds["city"].isin(metros).mean() > 0.70


@REAL_ONLY
def test_capital_is_head_heavy(rounds):
    amt = money_subset(rounds)["amount_usd_adj"].sort_values(ascending=False)
    assert amt.head(50).sum() / amt.sum() > 0.45
    assert amt.mean() > 5 * amt.median(), "mean is inflated; quote the median"

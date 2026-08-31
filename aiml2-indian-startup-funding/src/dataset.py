"""
Loading and cleaning for the Kaggle 'Indian Startup Funding' dataset.

Single source of truth for every correction this project makes. The naive
baseline in src/naive_pipeline.py deliberately does NOT use the cleaners here --
that contrast is the project.

Source : https://www.kaggle.com/datasets/sudalairajkumar/indian-startup-funding
License: CC0. Underlying data collected by https://trak.in
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REAL = Path("data/startup_funding.csv")
SAMPLE = Path("data/startup_funding_sample.csv")
EXPECTED_ROWS = 3044

RAW_COLUMNS = [
    "sr_no", "date_raw", "startup", "industry_raw", "subvertical",
    "city_raw", "investors_raw", "stage_raw", "amount_raw", "remarks",
]

# Placeholder text stored where a value is simply absent.
NULL_TOKENS = {
    "", "nan", "n/a", "na", "none", "-", "\u2014", "unknown",
    "undisclosed", "undisclosed amount", "not disclosed",
}


def resolve_path(explicit: str | None = None) -> Path:
    """Prefer the real CSV, fall back to the generated stand-in."""
    if explicit:
        return Path(explicit)
    return REAL if REAL.exists() else SAMPLE


def is_real(explicit: str | None = None) -> bool:
    p = resolve_path(explicit)
    if not p.exists():
        return False
    with p.open(encoding="utf-8-sig") as fh:
        rows = sum(1 for _ in fh) - 1
    return rows >= EXPECTED_ROWS - 1


# ---------------------------------------------------------------------------
# Field-level cleaners
# ---------------------------------------------------------------------------

def unescape(s: pd.Series) -> pd.Series:
    r"""Strip LITERAL escape text that the published CSV stores as characters.

    ``od -c`` on the raw file shows ``\ \ x c 2 \ \ x a 0`` -- the 8 characters
    ``\\xc2\\xa0``, i.e. TWO backslashes, not a real U+00A0 non-breaking space.
    Consequences worth knowing before trusting any parse of this file:

    * ``grep -P '\xc2\xa0'`` reports zero matches while 92 cells are affected,
      so the defect is invisible to the obvious check.
    * ``.str.strip()`` does nothing to them, so ``' Gurgaon'`` survives as a
      city distinct from ``'Gurgaon'``.
    * The ``\\+`` quantifier is load-bearing. Matching a single backslash
      removes ``\xc2`` and leaves the other backslash behind, turning the cell
      into ``'\ Gurgaon'`` -- still distinct, so the bug merely changes shape.
      The final ``\\+`` sweep catches any remaining debris.
    """
    return (
        s.astype("string")
        .str.replace(r"\\+x[0-9a-fA-F]{2}", " ", regex=True)
        .str.replace(r"\\+[nrt]", " ", regex=True)
        .str.replace(r"\\+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def to_na(s: pd.Series) -> pd.Series:
    return s.mask(s.str.lower().isin(NULL_TOKENS))


def clean_name(s: pd.Series) -> pd.Series:
    r"""Normalise display names: curly apostrophes and stray wrapping quotes.

    Row 1 stores ``BYJU'S`` with U+2019 while row 67 stores ``"BYJU\'S"``.
    """
    return (
        s.str.replace(r"[\u2018\u2019\u02bc]", "'", regex=True)
        .str.replace(r'^"+|"+$', "", regex=True)
        .str.strip()
    )


def fix_date(s: pd.Series) -> pd.Series:
    """Repair the 8 malformed dates, then parse day-first.

    Defects present: '12/05.2015' (dot separator), '05/072018' (missing
    slash), '01/07/015' (three-digit year), '22/01//2015' (doubled slash).
    """
    t = (
        s.str.replace(".", "/", regex=False)
        .str.replace("/+", "/", regex=True)
        .str.replace(r"^(\d{2})(\d{2})(\d{4})$", r"\1/\2/\3", regex=True)
        .str.replace(r"^(\d{1,2})/(\d{2})(\d{4})$", r"\1/\2/\3", regex=True)
        .str.replace(r"/0(\d{2})$", r"/20\1", regex=True)
    )
    return pd.to_datetime(t, format="mixed", dayfirst=True, errors="coerce")


def fix_amount(s: pd.Series) -> pd.Series:
    """Parse the amount column.

    Digit grouping is INDIAN (lakh/crore), not international: ``20,00,00,000``
    is 200,000,000 -- not 20,000,000. Removing every comma is the correct parse.

    Note pandas' ``thousands=','`` also happens to give the right number, because
    it strips separators without validating the grouping. The real hazard is
    human and downstream: ``20,00,00,000`` reads as "20,000,000-ish" to an eye
    trained on international grouping, and any code that validates or formats on
    three-digit groups mis-handles it.
    """
    t = (
        s.str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace(r"[^\d.]", "", regex=True)
    )
    # Cast to Float64 explicitly. to_numeric infers nullable Int64 when every
    # surviving value is a whole number, and assigning amount/71 into an Int64
    # array raises "cannot safely cast non-equivalent object to int64". Whether
    # that happens depends on the data, so it surfaces on one file and not
    # another -- money is float here regardless.
    return pd.to_numeric(t.mask(t.eq("")), errors="coerce").astype("Float64")


CITY_MAP = {
    "bangalore": "Bengaluru", "bengaluru": "Bengaluru",
    "gurgaon": "Gurugram", "gurugram": "Gurugram",
    "delhi": "Delhi", "new delhi": "Delhi",
    "noida": "Noida", "mumbai": "Mumbai", "pune": "Pune",
    "hyderabad": "Hyderabad", "chennai": "Chennai",
    "ahmedabad": "Ahmedabad", "kolkata": "Kolkata",
    "jaipur": "Jaipur", "indore": "Indore",
    "trivandrum": "Thiruvananthapuram",
    "thiruvananthapuram": "Thiruvananthapuram",
}


def fix_city(s: pd.Series) -> pd.Series:
    """Canonicalise city; 'Pune / US' keeps the primary (first) location."""
    primary = s.str.split(r"[/&]").str[0].str.strip()
    return primary.str.lower().map(CITY_MAP).fillna(primary.str.title())


# Ordered rules; first match wins.
INDUSTRY_RULES = [
    (r"e-?\s?comm", "eCommerce"),
    (r"fin-?\s?tech|financ|payment|lending|insur|bank", "FinTech / Finance"),
    (r"ed-?\s?tech|educat|e-?learning|e-?tech", "EdTech / Education"),
    (r"health|medic|pharma|wellness|hospital|diagnost", "Healthcare"),
    (r"logistic|transport|mobility|delivery|shipping", "Logistics & Transport"),
    (r"food|beverage|restaurant|grocer|agri|farm", "Food & Agri"),
    (r"real\s?estate|propert|construct", "Real Estate"),
    (r"travel|hospitalit|hotel|tourism", "Travel & Hospitality"),
    (r"consumer internet", "Consumer Internet"),
    (r"technology|^it$|software|saas|ai|analytic|data", "Technology"),
]


def fix_industry(s: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    low = s.str.lower()
    for pattern, label in INDUSTRY_RULES:
        out = out.mask(low.str.contains(pattern, regex=True, na=False) & out.isna(), label)
    return out.fillna(s.str.title())


def fix_stage(s: pd.Series) -> pd.Series:
    """Collapse ~55 spellings of the round label into 16.

    Includes 'Seed\\nFunding', 'Seed/ Angel Funding', 'Seed / Angle Funding'
    (sic) and 'pre-Series A' vs 'Pre-series A'.
    """
    low = s.str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="string")

    def rule(pattern: str, label: str) -> None:
        nonlocal out
        out = out.mask(low.str.contains(pattern, regex=True, na=False) & out.isna(), label)

    rule(r"pre-?\s?series\s?a", "Pre-Series A")
    for letter in "abcdefghij":
        rule(rf"series\s?{letter}\b", f"Series {letter.upper()}")
    rule(r"seed|angel|angle", "Seed / Angel")
    rule(r"debt", "Debt")
    rule(r"private equity", "Private Equity")
    rule(r"venture", "Venture Round")
    rule(r"corporate", "Corporate Round")
    return out.fillna("Other")


# ---------------------------------------------------------------------------
# Entity resolution
#
# Deliberately an EXPLICIT key -> canonical map rather than prefix or fuzzy
# matching. 'Ola Electric' is a different company from 'Ola'; 'OYOfit' and
# 'FroyoFit' are not 'OYO'; 'Paytm Marketplace' (Paytm Mall) is not 'Paytm'.
# Any startswith() rule silently merges those and inflates the leaderboard.
# ---------------------------------------------------------------------------
STARTUP_ALIASES = {
    "olacabs": ("ola", "Ola"),
    "ola": ("ola", "Ola"),
    "flipkartcom": ("flipkart", "Flipkart"),
    "flipkart": ("flipkart", "Flipkart"),
    "oyorooms": ("oyo", "OYO"),
    "oyoroom": ("oyo", "OYO"),
    "oyo": ("oyo", "OYO"),
}

INVESTOR_ALIASES = {
    "sequoia": ("sequoiacapitalindia", "Sequoia Capital India"),
    "sequoiacapital": ("sequoiacapitalindia", "Sequoia Capital India"),
    "sequoiaindia": ("sequoiacapitalindia", "Sequoia Capital India"),
    "sequoiacapitalindia": ("sequoiacapitalindia", "Sequoia Capital India"),
    "sequoiacapitalindiaadvisors": ("sequoiacapitalindia", "Sequoia Capital India"),
    "accelpartners": ("accel", "Accel"),
    "accelpartnersindia": ("accel", "Accel"),
    "accel": ("accel", "Accel"),
    "saifpartners": ("saifpartners", "SAIF Partners"),
    "saif": ("saifpartners", "SAIF Partners"),
    "tigerglobal": ("tigerglobal", "Tiger Global"),
    "tigerglobalmanagement": ("tigerglobal", "Tiger Global"),
    "blumeventures": ("blumeventures", "Blume Ventures"),
    "blumeventure": ("blumeventures", "Blume Ventures"),
    "kalaaricapital": ("kalaaricapital", "Kalaari Capital"),
    "idgventures": ("idgventures", "IDG Ventures"),
    "idgventuresindia": ("idgventures", "IDG Ventures"),
    "nexusventurepartners": ("nexusventures", "Nexus Venture Partners"),
    "nexusventures": ("nexusventures", "Nexus Venture Partners"),
    "matrixpartners": ("matrixpartners", "Matrix Partners"),
    "matrixpartnersindia": ("matrixpartners", "Matrix Partners"),
    "indianangelnetwork": ("indianangelnetwork", "Indian Angel Network"),
}

# Anonymity placeholders, not investors. Left in, 'Undisclosed Investors' (83
# rows) plus 'Undisclosed Investor' (24) ranks as the single most active
# investor in the dataset, ahead of Sequoia.
_PLACEHOLDER = re.compile(
    r"^(\d+\s+)?(other\s+|multiple\s+|several\s+|various\s+|new\s+|existing\s+)?"
    r"(un\s?disclosed|unknown|un\s?named|not\s+disclosed|anonymous|individual|"
    r"private|angel|marquee|hni|consortium|group|investor)",
    re.I,
)
_BARE_COLLECTIVE = {"investors", "investor", "angels", "hnis", "vcs", "others", "n/a"}


def is_placeholder(name: str) -> bool:
    n = str(name).strip().lower().rstrip(".")
    return bool(_PLACEHOLDER.match(n)) or n in _BARE_COLLECTIVE


# ---------------------------------------------------------------------------
# Currency mislabelling
#
# sr_no 61, 'Rapido Bike Taxi', carries 3,90,00,00,000 in the column headed
# 'Amount in USD'. That is INR 390 crore (~$55M) -- the round actually reported
# in Aug 2019. Read as USD it is $3.9B, which makes a bike-taxi company the
# largest round in the dataset, ahead of Flipkart's genuine $2.5B, and accounts
# for 10% of the dataset's entire as-published total.
#   https://inc42.com/buzz/exclusive-rapido-closes-series-b-funding-round-at-55-mn/
#   https://economictimes.indiatimes.com/small-biz/startups/newsbuzz/rapido-enters-fast-lane-with-390-crore/articleshow/70868579.cms
# ---------------------------------------------------------------------------
CONFIRMED_INR = {61}
INR_PER_USD_AUG_2019 = 71.0
OUTLIER_THRESHOLD = 250_000_000

# Indian rounds at or above the threshold that are independently known to be real,
# so they are not flagged for review.
KNOWN_REAL_MEGA = {
    "flipkart", "flipkartcom", "paytm", "ola", "olacabs", "snapdeal", "oyo",
    "oyorooms", "swiggy", "zomato", "udaan", "bigbasket", "byjus",
    "policybazaar", "delhivery",
}


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the published CSV exactly as-is, with only the BOM removed.

    ``encoding='utf-8-sig'`` matters: without it column 0 is named
    ``'\\ufeffSr No'`` and every by-name lookup misses.
    """
    df = pd.read_csv(resolve_path(str(path) if path else None),
                     encoding="utf-8-sig", dtype="string")
    df.columns = RAW_COLUMNS
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Raw frame -> one tidy row per funding round."""
    df = df.copy()
    for c in df.columns:
        df[c] = to_na(unescape(df[c]))
    df["startup"] = clean_name(df["startup"])

    out = pd.DataFrame({
        "sr_no": pd.to_numeric(df["sr_no"]),
        "date": fix_date(df["date_raw"]),
        "startup": df["startup"],
        "startup_key": df["startup"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True),
        "industry": fix_industry(df["industry_raw"]),
        "subvertical": df["subvertical"],
        "city": fix_city(df["city_raw"]),
        "investors": df["investors_raw"],
        "stage": fix_stage(df["stage_raw"]),
        "amount_usd": fix_amount(df["amount_raw"]),
        "remarks": df["remarks"],
    })

    # Merge the split entities (Ola/Ola Cabs, Flipkart/Flipkart.com, OYO/OyoRooms).
    akey = out["startup_key"].map(lambda k: STARTUP_ALIASES.get(k, (None, None))[0])
    aname = out["startup_key"].map(lambda k: STARTUP_ALIASES.get(k, (None, None))[1])
    out["startup_key"] = akey.fillna(out["startup_key"])
    out["startup"] = aname.fillna(out["startup"])

    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.to_period("M").astype("string")

    out["amount_flag"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[out["sr_no"].isin(CONFIRMED_INR), "amount_flag"] = "confirmed_inr_not_usd"
    suspect = (
        out["amount_usd"].ge(OUTLIER_THRESHOLD)
        & ~out["startup_key"].isin(KNOWN_REAL_MEGA)
        & out["amount_flag"].isna()
    )
    out.loc[suspect, "amount_flag"] = "review_outlier"

    # An explicit fillna(False) mask is required here. With the nullable
    # 'string' dtype, `amount_flag != "confirmed_inr_not_usd"` evaluates to NA
    # (not True) on every unflagged row, and .where()/.mask() treat NA as False
    # -- which silently divided all 2,073 amounts by 71. tests/ pins this.
    is_inr = out["amount_flag"].eq("confirmed_inr_not_usd").fillna(False).to_numpy(dtype=bool)
    out["amount_usd_adj"] = out["amount_usd"].mask(is_inr, out["amount_usd"] / INR_PER_USD_AUG_2019)
    return out


def investor_table(rounds: pd.DataFrame) -> pd.DataFrame:
    """One row per investor-round pair.

    Splits on commas AND on ' and ' / ' & '. Comma-only splitting leaves
    'Saama Capital and Sequoia Capital' as a single fictitious investor and
    spreads Sequoia across four spellings (73 deals instead of 122).
    """
    inv = (
        rounds[["sr_no", "date", "year", "startup", "city", "industry", "stage",
                "amount_usd", "amount_usd_adj", "amount_flag", "investors"]]
        .assign(investor=lambda d: d["investors"].str.split(
            r"(?:,|\s+&\s+|\s+and\s+)(?![^(]*\))", regex=True))
        .explode("investor")
        .drop(columns="investors")
    )
    inv["investor"] = (
        inv["investor"].str.strip()
        .str.replace(r"^(and|&)\s+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[.,;]+$", "", regex=True)
    )
    inv = inv[inv["investor"].notna() & inv["investor"].str.len().gt(1)]
    inv = inv[~inv["investor"].map(is_placeholder)]

    inv["investor_key"] = inv["investor"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    ikey = inv["investor_key"].map(lambda k: INVESTOR_ALIASES.get(k, (None, None))[0])
    iname = inv["investor_key"].map(lambda k: INVESTOR_ALIASES.get(k, (None, None))[1])
    inv["investor_key"] = ikey.fillna(inv["investor_key"])
    inv["investor"] = iname.fillna(inv["investor"])
    return inv.reset_index(drop=True)


def money_subset(rounds: pd.DataFrame) -> pd.DataFrame:
    """Rows safe to sum: an amount is present and not flagged for review.

    The .fillna(True) is required and is the same nullable-dtype trap as above:
    `amount_flag.ne("review_outlier")` returns NA on every unflagged row, and
    boolean indexing treats NA as False -- so without it this function returns
    only the handful of FLAGGED rows, the exact inverse of its purpose.
    """
    keep = rounds["amount_flag"].ne("review_outlier").fillna(True).to_numpy(dtype=bool)
    return rounds[rounds["amount_usd_adj"].notna().to_numpy(dtype=bool) & keep]


def load_clean(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rounds = clean(load_raw(path))
    return rounds, investor_table(rounds)


def usd(v: float) -> str:
    if pd.isna(v):
        return "-"
    if abs(v) >= 1e9:
        return f"${v / 1e9:,.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.1f}M"
    return f"${v:,.0f}"

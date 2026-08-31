r"""
Generate a stand-in for the Kaggle CSV that reproduces its DEFECTS.

    python -m data.generate_sample

The point is not to fake the findings -- headline figures in INSIGHTS.md come
from the real 3,044-row file. The point is that the test suite must exercise the
defects rather than trust stored numbers, so every defect the audit corrects is
reproduced here structurally:

  * literal ``\\xc2\\xa0`` escape text (TWO backslashes, not a real U+00A0)
  * Indian digit grouping in the amount column (``20,00,00,000`` = 200M)
  * one INR value sitting in the column headed 'Amount in USD'
  * city spelling variants (Bangalore/Bengaluru, Gurgaon/Gurugram)
  * split entities (Ola/Ola Cabs, Flipkart/Flipkart.com) AND near-miss names
    that must NOT be merged (Ola Electric, OYOfit)
  * 'Undisclosed Investors' placeholders
  * investor lists fused with ' and ' / ' & '
  * malformed dates (dot separator, missing slash, 3-digit year, double slash)
  * ~32% missing amounts, and coverage that decays after 2017
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

DEST = Path("data/startup_funding_sample.csv")
HEADER = ["Sr No", "Date dd/mm/yyyy", "Startup Name", "Industry Vertical", "SubVertical",
          "City  Location", "Investors Name", "InvestmentnType", "Amount in USD", "Remarks"]

NBSP = "\\xc2\\xa0"  # the literal 8-character sequence stored in the real file

CITIES = ["Bangalore", "Bengaluru", "Mumbai", "New Delhi", "Delhi", "Gurgaon",
          "Gurugram", "Pune", "Noida", "Hyderabad", "Chennai", "Jaipur"]
INDUSTRIES = ["Consumer Internet", "Technology", "eCommerce", "ECommerce", "E-Commerce",
              "Healthcare", "Finance", "FinTech", "Ed-Tech", "Logistics", "Food & Beverage"]
STAGES = ["Seed Funding", "Private Equity", "Seed/ Angel Funding", "Seed / Angel Funding",
          "Seed\\nFunding", "Seed / Angle Funding", "Series A", "Series B", "Debt Funding",
          "pre-Series A", "Pre-series A"]
INVESTORS = ["Sequoia Capital", "Sequoia India", "Sequoia Capital India", "Accel Partners",
             "Blume Ventures", "Kalaari Capital", "SAIF Partners", "Tiger Global",
             "Ratan Tata", "IDG Ventures", "Matrix Partners", "Nexus Venture Partners"]
NAMES = ["Zolo", "Cure.fit", "Dunzo", "Meesho", "Khatabook", "Rivigo", "Netmeds",
         "Pepperfry", "ShareChat", "Unacademy", "Lenskart", "Cars24", "Zetwerk",
         "Nykaa", "Bounce", "Milkbasket", "Wakefit", "Vedantu", "Doubtnut", "Toppr"]

# Rows per year, echoing the real file's coverage decay after 2017.
YEAR_ROWS = {2015: 120, 2016: 130, 2017: 90, 2018: 40, 2019: 15, 2020: 2}


def indian_group(n: int) -> str:
    """Format 200000000 as '20,00,00,000' -- last three digits, then pairs."""
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join([*parts, tail])


def main() -> int:
    rng = random.Random(20200113)
    rows: list[list[str]] = []
    sr = 0

    # --- fixed rows carrying the defects the audit must catch --------------
    # sr_no 61 is the confirmed INR-in-a-USD-column row; keep the number stable
    # so tests and src.dataset.CONFIRMED_INR line up.
    fixed = [
        # The curly U+2019 is deliberate: the real file stores this name two
        # ways, and clean_name() must fold them together.
        ("09/01/2020", "BYJU’S", "E-Tech", "Bengaluru", "Tiger Global Management",  # noqa: RUF001
         "Private Equity Round", indian_group(200_000_000)),
        ("13/01/2020", '"BYJU\\\'S"', "E-Tech", "Bengaluru", "Sequoia Capital and Temasek Holdings",
         "Private Equity", indian_group(25_000_000)),
        ("11/08/2017", "Flipkart", "eCommerce", "Bangalore", "SoftBank Vision Fund",
         "Private Equity", indian_group(2_500_000_000)),
        ("28/07/2015", "Flipkart.com", "eCommerce", "Bengaluru", "Undisclosed Investors",
         "Private Equity", indian_group(700_000_000)),
        ("18/11/2015", "Ola", "Transportation", "Bangalore", "Undisclosed Investor",
         "Private Equity", indian_group(500_000_000)),
        ("05/04/2017", "Ola Cabs", "Transportation", "Bengaluru", "Accel Partners & SAIF Partners",
         "Private Equity", indian_group(104_000_000)),
        # near-misses that must NOT be merged into Ola / OYO
        ("02/07/2019", "Ola Electric", "Transportation", "Bengaluru", "Tiger Global",
         "Series B", indian_group(250_000_000)),
        ("15/03/2018", "OyoRooms", "Hospitality", "Gurgaon", "Sequoia Capital",
         "Private Equity", indian_group(250_000_000)),
        ("21/09/2018", "Oyo", "Hospitality", "Gurugram", "Undisclosed Investors",
         "Private Equity", indian_group(100_000_000)),
        ("14/06/2017", "OYOfit", "Health and Wellness", "Gurgaon", "Kalaari Capital",
         "Seed Funding", indian_group(500_000)),
        ("09/09/2016", "FroyoFit", "Food & Beverage", "Mumbai", "Group of Angel investors",
         "Seed Funding", indian_group(150_000)),
        ("17/05/2017", "Paytm", "Finance", "Noida", "SoftBank",
         "Private Equity", indian_group(1_400_000_000)),
        ("03/02/2018", "Paytm Marketplace", "eCommerce", "Noida", "Alibaba",
         "Private Equity", indian_group(200_000_000)),
        # fused investor names and placeholder mixtures
        ("22/04/2016", "Zoomcar", "Transportation", "Bangalore",
         "Saama Capital and Sequoia Capital", "Series B", indian_group(11_000_000)),
        ("30/06/2016", "Practo", "Healthcare", "Bengaluru",
         "Blume Ventures & Other unnamed Investors", "Series A", indian_group(4_000_000)),
        # malformed dates
        ("12/05.2015", "Simplotel", "Technology", "Bangalore", "MakeMyTrip",
         "Private Equity", ""),
        ("05/072018", "Fynd", "eCommerce", "Mumbai", "Google", "Series B",
         indian_group(20_000_000)),
        ("01/07/015", "Zopper", "eCommerce", "Noida", "Nirvana Ventures",
         "Series B", indian_group(20_000_000)),
        ("22/01//2015", "couponmachine.in", "eCommerce", "Bengaluru",
         "UK based Group of Angel Investors", "Seed Funding", indian_group(140_000)),
        # literal escape text in four columns at once
        (f"{NBSP}10/7/2015", f"{NBSP}Infinity Assurance", "Technology", f"{NBSP}Gurgaon",
         f"{NBSP}Undisclosed Investors", "Seed Funding", f"{NBSP}{indian_group(600_000)}"),
        ("27/08/2019", "Vogo Automotive", "Transportation", "Bengaluru", "Matrix Partners",
         "Series B", indian_group(283_000_000)),
    ]
    for date, name, industry, city, investors, stage, amount in fixed:
        sr += 1
        rows.append([str(sr), date, name, industry, "nan", city, investors, stage, amount, "nan"])

    # --- filler rows, padded so the INR row lands on sr_no 61 -------------
    while sr < 60:
        sr += 1
        rows.append([str(sr), f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2016",
                     rng.choice(NAMES), rng.choice(INDUSTRIES), "nan", rng.choice(CITIES),
                     rng.choice(INVESTORS), rng.choice(STAGES),
                     indian_group(rng.randrange(100_000, 5_000_000, 50_000)), "nan"])

    sr += 1
    assert sr == 61, f"INR row must be sr_no 61, got {sr}"
    rows.append([str(sr), "27/08/2019", "Rapido Bike Taxi", "Transportation", "Bike taxi",
                 "Bengaluru", "WestBridge Capital", "Series B",
                 indian_group(3_900_000_000),  # INR 390 crore in a column headed USD
                 "nan"])

    # --- bulk rows, ~32% amounts missing ----------------------------------
    for year, n in YEAR_ROWS.items():
        for _ in range(n):
            sr += 1
            missing = rng.random() < (0.44 if year <= 2016 else 0.12)
            amount = "" if missing else indian_group(
                rng.randrange(100_000, 40_000_000, 50_000) if year <= 2017
                else rng.randrange(5_000_000, 120_000_000, 500_000))
            investors = rng.choice(INVESTORS)
            if rng.random() < 0.18:
                investors = f"{investors} and {rng.choice(INVESTORS)}"
            elif rng.random() < 0.08:
                investors = "Undisclosed Investors"
            rows.append([str(sr), f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{year}",
                         rng.choice(NAMES), rng.choice(INDUSTRIES), "nan",
                         rng.choice(CITIES), investors, rng.choice(STAGES), amount, "nan"])

    DEST.parent.mkdir(parents=True, exist_ok=True)
    with DEST.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)

    print(f"wrote {DEST} ({len(rows)} rows)")
    print("reproduces: literal-escape text, Indian digit grouping, INR-as-USD (sr_no 61),")
    print("            city variants, split entities, placeholder investors, bad dates.")
    print("NOTE: a stand-in. INSIGHTS.md figures come from the real 3,044-row CSV")
    print("      -- run `make download`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

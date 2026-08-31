"""Exploratory analysis.

Each figure answers one business question stated in its title, and each returns
numbers that feed the report. Deliberately six figures rather than thirty: a
correlation heatmap of one-hot encoded columns looks industrious and tells you
nothing actionable.

Run on the TRAINING split only. Exploring the test set is a soft form of
leakage — every decision made after seeing it (which features to build, which
bands to cut) is fitted to it.
"""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import Config
from src.data.loader import make_dataset
from src.data.schema import INTERNET_ADDON_COLUMNS

sns.set_theme(style="whitegrid", palette="deep")

CHURN_COLOR = "#c53030"
STAY_COLOR = "#2b6cb0"

#: Tenure cut points, reused by feature engineering so EDA and features agree.
TENURE_BINS = [-0.1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"]


def _rate(df: pd.DataFrame, by: str, target: str = "churn") -> pd.DataFrame:
    """Churn rate and volume by category, sorted by rate."""
    g = df.groupby(by, observed=True)[target].agg(["mean", "size"])
    return g.rename(columns={"mean": "churn_rate", "size": "customers"}).sort_values(
        "churn_rate", ascending=False)


def _annotate_rates(ax: plt.Axes, rates: pd.Series, fmt: str = "{:.1%}") -> None:
    for i, v in enumerate(rates):
        ax.text(v, i, f" {fmt.format(v)}", va="center", fontsize=9)


def run(cfg: Config, path: str | None = None) -> dict[str, Any]:
    """Produce the EDA figures and return the findings used in the report."""
    cfg.ensure_dirs()
    figdir = cfg.resolve("paths", "figures_dir")
    ds = make_dataset(cfg, path)

    df = ds.X_train.copy()
    df["churn"] = ds.y_train.to_numpy()
    df["tenure_band"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)

    has_net = df["InternetService"].ne("No")
    df["n_addons"] = sum((df[c] == "Yes").astype(int) for c in INTERNET_ADDON_COLUMNS)

    overall = float(df["churn"].mean())
    findings: dict[str, Any] = {"overall_churn_rate": overall, "n_train": len(df)}

    # ---- Q1: how bad is churn, and is the target balanced enough to model? ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    counts = df["churn"].value_counts().sort_index()
    axes[0].bar(["Stayed", "Churned"], counts.to_numpy(), color=[STAY_COLOR, CHURN_COLOR])
    for i, v in enumerate(counts.to_numpy()):
        axes[0].text(i, v, f"{v:,}\n{v / len(df):.1%}", ha="center", va="bottom", fontsize=10)
    axes[0].set_title(f"Q1. How many customers churn?  (rate {overall:.2%})")
    axes[0].set_ylabel("customers")
    axes[0].set_ylim(0, counts.max() * 1.18)

    # A majority-class baseline scores high accuracy while catching nobody.
    axes[1].bar(["Accuracy of\n'nobody churns'", "Churners it\nfinds"],
                [1 - overall, 0.0], color=["#a0aec0", CHURN_COLOR])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Q1b. Why accuracy is the wrong metric")
    for i, v in enumerate([1 - overall, 0.0]):
        axes[1].text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figdir / "01_churn_overview.png", dpi=130)
    plt.close(fig)

    # ---- Q2: when do customers leave? ----
    band = _rate(df, "tenure_band").reindex(TENURE_LABELS)
    monthly = df.groupby("tenure", observed=True)["churn"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].bar(band.index.astype(str), band["churn_rate"], color=CHURN_COLOR)
    axes[0].axhline(overall, ls="--", color="black", lw=1,
                    label=f"overall {overall:.1%}")
    axes[0].set_title("Q2. When do customers leave?")
    axes[0].set_ylabel("churn rate")
    axes[0].legend(fontsize=8)
    for i, v in enumerate(band["churn_rate"]):
        axes[0].text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)

    axes[1].plot(monthly.index, monthly.to_numpy(), color=CHURN_COLOR, lw=1.6)
    axes[1].axhline(overall, ls="--", color="black", lw=1)
    axes[1].set_title("Q2b. Churn rate by exact tenure (months)")
    axes[1].set_xlabel("tenure (months)")
    axes[1].set_ylabel("churn rate")
    fig.tight_layout()
    fig.savefig(figdir / "02_churn_by_tenure.png", dpi=130)
    plt.close(fig)

    findings["churn_by_tenure_band"] = band["churn_rate"].round(4).to_dict()
    findings["churn_first_6m"] = float(band.loc["0-6m", "churn_rate"])
    findings["churn_last_band"] = float(band.loc["49-72m", "churn_rate"])

    # ---- Q3: which commercial terms carry the risk? ----
    contract = _rate(df, "Contract")
    payment = _rate(df, "PaymentMethod")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].barh(contract.index.astype(str), contract["churn_rate"], color=CHURN_COLOR)
    axes[0].set_title("Q3. Which contract type churns?")
    axes[0].set_xlabel("churn rate")
    axes[0].set_xlim(0, max(contract["churn_rate"]) * 1.25)
    _annotate_rates(axes[0], contract["churn_rate"])

    axes[1].barh(payment.index.astype(str), payment["churn_rate"], color="#975a16")
    axes[1].set_title("Q3b. Which payment method churns?")
    axes[1].set_xlabel("churn rate")
    axes[1].set_xlim(0, max(payment["churn_rate"]) * 1.25)
    _annotate_rates(axes[1], payment["churn_rate"])
    fig.tight_layout()
    fig.savefig(figdir / "03_churn_by_contract_payment.png", dpi=130)
    plt.close(fig)

    findings["churn_by_contract"] = contract["churn_rate"].round(4).to_dict()
    findings["churn_by_payment"] = payment["churn_rate"].round(4).to_dict()

    # ---- Q4: does product adoption protect against churn? ----
    net = _rate(df, "InternetService")
    addon_rate = df[has_net].groupby("n_addons", observed=True)["churn"].agg(["mean", "size"])
    support = _rate(df, "TechSupport")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].bar(net.index.astype(str), net["churn_rate"], color=CHURN_COLOR)
    axes[0].set_title("Q4. Does internet type matter?")
    axes[0].set_ylabel("churn rate")
    for i, v in enumerate(net["churn_rate"]):
        axes[0].text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)

    axes[1].bar(addon_rate.index.astype(str), addon_rate["mean"], color="#276749")
    axes[1].set_title("Q4b. Do add-ons make customers stick?\n(internet customers only)")
    axes[1].set_xlabel("number of add-on services")
    axes[1].set_ylabel("churn rate")
    for i, v in enumerate(addon_rate["mean"]):
        axes[1].text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)

    axes[2].bar(support.index.astype(str), support["churn_rate"], color="#553c9a")
    axes[2].set_title("Q4c. Tech support and churn")
    axes[2].set_ylabel("churn rate")
    axes[2].tick_params(axis="x", labelsize=8)
    for i, v in enumerate(support["churn_rate"]):
        axes[2].text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "04_churn_by_services.png", dpi=130)
    plt.close(fig)

    findings["churn_by_internet"] = net["churn_rate"].round(4).to_dict()
    findings["churn_by_n_addons"] = addon_rate["mean"].round(4).to_dict()

    # ---- Q5: what do the money variables look like? ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, col in zip(axes, ("tenure", "MonthlyCharges", "TotalCharges"), strict=True):
        for label, colour, name in ((0, STAY_COLOR, "Stayed"), (1, CHURN_COLOR, "Churned")):
            sns.kdeplot(df.loc[df["churn"] == label, col], ax=ax, fill=True,
                        alpha=0.35, color=colour, label=name, warn_singular=False)
        ax.set_title(f"Q5. {col} by outcome")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "05_numeric_distributions.png", dpi=130)
    plt.close(fig)

    findings["median_monthly_churned"] = float(
        df.loc[df["churn"] == 1, "MonthlyCharges"].median())
    findings["median_monthly_stayed"] = float(
        df.loc[df["churn"] == 0, "MonthlyCharges"].median())

    # ---- Q6: where is risk concentrated, jointly? ----
    pivot = df.pivot_table(index="Contract", columns="tenure_band", values="churn",
                           aggfunc="mean", observed=True)
    # fillna(0).astype(int) is required: an empty Contract x tenure_band cell makes
    # the whole frame float, and seaborn's fmt=",d" then raises. Real data happens
    # to populate every cell, so this only surfaces on smaller inputs.
    vol = df.pivot_table(index="Contract", columns="tenure_band", values="churn",
                         aggfunc="size", observed=True).fillna(0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="Reds", ax=axes[0],
                cbar_kws={"label": "churn rate"})
    axes[0].set_title("Q6. Where is risk concentrated?  (churn rate)")
    sns.heatmap(vol, annot=True, fmt=",d", cmap="Blues", ax=axes[1],
                cbar_kws={"label": "customers"})
    axes[1].set_title("Q6b. ...and how many customers are there?")
    fig.tight_layout()
    fig.savefig(figdir / "06_risk_concentration.png", dpi=130)
    plt.close(fig)

    m2m_new = pivot.loc["Month-to-month", "0-6m"]
    findings["m2m_first6m_churn"] = float(m2m_new)
    findings["m2m_first6m_customers"] = int(vol.loc["Month-to-month", "0-6m"])
    findings["electronic_check_churn"] = float(payment.loc["Electronic check", "churn_rate"])
    findings["fiber_churn"] = float(net.loc["Fiber optic", "churn_rate"])

    # ---- console summary ----
    print(f"training rows            : {len(df):,}")
    print(f"overall churn rate       : {overall:.2%}")
    print(f"churn in first 6 months  : {findings['churn_first_6m']:.2%}")
    print(f"churn at 49-72 months    : {findings['churn_last_band']:.2%}")
    print(f"month-to-month + 0-6m    : {m2m_new:.2%} "
          f"({findings['m2m_first6m_customers']:,} customers)")
    print(f"electronic check churn   : {findings['electronic_check_churn']:.2%}")
    print(f"fiber optic churn        : {findings['fiber_churn']:.2%}")
    print(f"median monthly, churned  : {findings['median_monthly_churned']:.2f}")
    print(f"median monthly, stayed   : {findings['median_monthly_stayed']:.2f}")
    print("\nchurn by contract:")
    print(contract.assign(churn_rate=lambda d: d["churn_rate"].map("{:.2%}".format)).to_string())
    print("\nchurn by add-on count (internet customers):")
    print(addon_rate.assign(mean=lambda d: d["mean"].map("{:.2%}".format))
          .rename(columns={"mean": "churn_rate", "size": "customers"}).to_string())
    print(f"\n[figures] {figdir}")
    return findings

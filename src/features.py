"""Feature engineering for the Spaceship Titanic dataset.

Shared by training and inference so that both see identical columns. Nothing
here touches the target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SUBMISSIONS = ROOT / "submissions"

SEED = 42
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL = ["HomePlanet", "Destination", "Deck", "Side", "CryoSleep", "VIP"]
NUMERIC = [
    "Age", "GroupSize", "GroupPos", "Solo", "CabinNum", "CabinSize",
    "FamilySize", "TotalSpend", "NoSpend", "SpendCount", "IsChild",
    "NaNCount", *SPEND, *[f"log_{c}" for c in [*SPEND, "TotalSpend"]],
]


def load(split: str) -> pd.DataFrame:
    """Read one of the raw competition CSVs from data/."""
    DATA.mkdir(exist_ok=True)
    path = DATA / f"{split}.csv"
    if not path.exists():
        msg = (
            f"{path} not found. Fetch the data first:\n"
            "    import kagglehub; kagglehub.competition_download('spaceship-titanic')\n"
            "then copy train.csv, test.csv and sample_submission.csv into data/."
        )
        raise FileNotFoundError(msg)
    return pd.read_csv(path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features from the raw columns."""
    df = df.copy()

    # PassengerId is gggg_pp: the group travels together, often as a family.
    df["Group"] = df["PassengerId"].str.split("_").str[0].astype(int)
    df["GroupPos"] = df["PassengerId"].str.split("_").str[1].astype(int)
    df["GroupSize"] = df.groupby("Group")["Group"].transform("size")
    df["Solo"] = (df["GroupSize"] == 1).astype(int)

    # Cabin is deck/num/side.
    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2]
    # How crowded the passenger's exact cabin is.
    df["CabinSize"] = df.groupby("Cabin")["Cabin"].transform("size").fillna(0)

    # Surnames group families that a Group id may split across cabins.
    df["Surname"] = df["Name"].str.split().str[-1]
    df["FamilySize"] = df.groupby("Surname")["Surname"].transform("size").fillna(0)

    # Cryosleep passengers are confined to their cabins, so they cannot bill
    # anything. That makes two imputations sound rather than merely convenient.
    cryo = df["CryoSleep"] == True  # noqa: E712 - keep NaN out of the mask
    df.loc[cryo, SPEND] = df.loc[cryo, SPEND].fillna(0.0)
    spent = df[SPEND].sum(axis=1, min_count=1) > 0
    df.loc[spent & df["CryoSleep"].isna(), "CryoSleep"] = False

    df["TotalSpend"] = df[SPEND].sum(axis=1, min_count=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype("float")
    df.loc[df["TotalSpend"].isna(), "NoSpend"] = np.nan
    # Spend is heavily zero-inflated and long-tailed; log1p tames the tail.
    for col in [*SPEND, "TotalSpend"]:
        df[f"log_{col}"] = np.log1p(df[col])
    df["SpendCount"] = (df[SPEND] > 0).sum(axis=1)

    df["Age"] = df["Age"].astype(float)
    df["IsChild"] = (df["Age"] < 13).astype("float")
    df.loc[df["Age"].isna(), "IsChild"] = np.nan

    # ~2% of every column is missing; the pattern itself may carry signal.
    df["NaNCount"] = df[
        ["HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP", *SPEND]
    ].isna().sum(axis=1)

    for col in CATEGORICAL:
        df[col] = df[col].astype("category")

    return df


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model's feature columns, in a fixed order."""
    return df[CATEGORICAL + NUMERIC].copy()


def to_catboost(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost wants categoricals as plain strings with no NaN."""
    X = X.copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype(str).fillna("NA")
    return X


CAT_IDX = [(CATEGORICAL + NUMERIC).index(c) for c in CATEGORICAL]


def align_categories(*matrices: pd.DataFrame) -> None:
    """Give every matrix the same category levels, in place.

    Without this a level seen only in test would be encoded differently from
    training, silently shifting the model's inputs.
    """
    for col in CATEGORICAL:
        levels = matrices[0][col].cat.categories
        for m in matrices[1:]:
            levels = levels.union(m[col].cat.categories)
        for m in matrices:
            m[col] = m[col].cat.set_categories(levels)

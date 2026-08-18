"""Baseline for the Spaceship Titanic competition.

Gradient boosting over engineered features, scored with 5-fold stratified CV
(the competition metric is accuracy), then refit on all of train to write
submission.csv.

Run: python baseline.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

SEED = 42
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL = ["HomePlanet", "Destination", "Deck", "Side", "CryoSleep", "VIP"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features from the raw columns. Nothing here touches the target."""
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
    cryo = df["CryoSleep"] == True  # noqa: E712 — keep NaN out of the mask
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
    numeric = [
        "Age", "GroupSize", "GroupPos", "Solo", "CabinNum", "CabinSize",
        "FamilySize", "TotalSpend", "NoSpend", "SpendCount", "IsChild",
        "NaNCount", *SPEND, *[f"log_{c}" for c in [*SPEND, "TotalSpend"]],
    ]
    return df[CATEGORICAL + numeric].copy()


def main() -> None:
    train = add_features(pd.read_csv("train.csv"))
    test = add_features(pd.read_csv("test.csv"))

    X = build_matrix(train)
    y = train["Transported"].astype(int)
    X_test = build_matrix(test)
    # Keep the categorical level sets aligned between the two matrices.
    for col in CATEGORICAL:
        levels = X[col].cat.categories.union(X_test[col].cat.categories)
        X[col] = X[col].cat.set_categories(levels)
        X_test[col] = X_test[col].cat.set_categories(levels)

    # HistGradientBoosting handles NaN and categoricals natively, so no
    # imputation or one-hot step is needed ahead of it.
    model = HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"{X.shape[1]} features, {len(X)} rows")
    print("CV accuracy per fold:", np.round(scores, 4))
    print(f"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    model.fit(X, y)
    pred = model.predict(X_test).astype(bool)

    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": pred})
    submission.to_csv("submission.csv", index=False)
    print(f"\nWrote submission.csv: {len(submission)} rows, "
          f"{pred.mean():.1%} predicted True")


if __name__ == "__main__":
    main()

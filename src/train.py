"""Train the Spaceship Titanic blend and persist it for inference.

Two boosters over the same features, averaged: CatBoost carries most of the
weight, HistGradientBoosting the rest. Reports out-of-fold accuracy (the
competition metric), refits both on all of train, and writes
model.joblib at the project root.

Run: python src/train.py [--no-cv]
"""

from __future__ import annotations

import sys

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from features import (
    CAT_IDX,
    CATEGORICAL,
    ROOT,
    SEED,
    add_features,
    align_categories,
    build_matrix,
    load,
    to_catboost,
)

MODEL_PATH = ROOT / "model.joblib"
# Chosen in experiment 004. The weight curve is flat between 0.1 and 0.4, so
# this is a plateau rather than a tuned point.
HGB_WEIGHT = 0.3


def build_models() -> tuple[HistGradientBoostingClassifier, CatBoostClassifier]:
    # HistGradientBoosting handles NaN and categoricals natively; its
    # hyperparameters come from the search in experiment 003.
    hgb = HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        max_iter=1000,
        learning_rate=0.1,
        max_leaf_nodes=12,
        min_samples_leaf=120,
        max_features=0.85,
        l2_regularization=0.5,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
    )
    cat = CatBoostClassifier(
        iterations=900,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=4.0,
        cat_features=CAT_IDX,
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False,
    )
    return hgb, cat


def blend(p_hgb: np.ndarray, p_cat: np.ndarray) -> np.ndarray:
    return HGB_WEIGHT * p_hgb + (1 - HGB_WEIGHT) * p_cat


def cross_validate(X, Xc, y) -> None:
    """Out-of-fold accuracy for each model and for the blend."""
    oof = {"hgb": np.zeros(len(y)), "cat": np.zeros(len(y))}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(cv.split(X, y), 1):
        hgb, cat = build_models()
        hgb.fit(X.iloc[tr], y.iloc[tr])
        cat.fit(Xc.iloc[tr], y.iloc[tr])
        oof["hgb"][va] = hgb.predict_proba(X.iloc[va])[:, 1]
        oof["cat"][va] = cat.predict_proba(Xc.iloc[va])[:, 1]
        print(f"  fold {fold}/5 done")

    y_arr = y.to_numpy()
    for name, p in oof.items():
        print(f"OOF accuracy {name:4s}: {((p > 0.5) == y_arr).mean():.4f}")
    p = blend(oof["hgb"], oof["cat"])
    print(f"OOF accuracy blend: {((p > 0.5) == y_arr).mean():.4f}")


def main() -> None:
    train = add_features(load("train"))
    # Test is loaded only to align category levels, never to fit anything.
    test = add_features(load("test"))

    X = build_matrix(train)
    y = train["Transported"].astype(int)
    align_categories(X, build_matrix(test))
    Xc = to_catboost(X)
    print(f"{X.shape[1]} features, {len(X)} rows")

    if "--no-cv" not in sys.argv:
        print("\nCross-validating (5 folds, two models each)...")
        cross_validate(X, Xc, y)

    print("\nFitting on all of train...")
    hgb, cat = build_models()
    hgb.fit(X, y)
    cat.fit(Xc, y)

    # Persist the category levels alongside the models: inference must encode
    # its columns exactly as training did.
    joblib.dump(
        {
            "hgb": hgb,
            "cat": cat,
            "hgb_weight": HGB_WEIGHT,
            "levels": {c: X[c].cat.categories for c in CATEGORICAL},
        },
        MODEL_PATH,
    )
    print(f"Saved {MODEL_PATH.name}")


if __name__ == "__main__":
    main()

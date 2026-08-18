"""Train the Spaceship Titanic model and persist it for inference.

Reports 5-fold stratified CV accuracy (the competition metric), refits on all
of train, and writes models/model.joblib.

Run: python src/train.py
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from features import (
    CATEGORICAL,
    MODELS,
    SEED,
    add_features,
    align_categories,
    build_matrix,
    load,
)

MODEL_PATH = MODELS / "model.joblib"


def build_model() -> HistGradientBoostingClassifier:
    # HistGradientBoosting handles NaN and categoricals natively, so no
    # imputation or one-hot step is needed ahead of it.
    return HistGradientBoostingClassifier(
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


def main() -> None:
    train = add_features(load("train"))
    # Test is loaded only to align category levels, never to fit anything.
    test = add_features(load("test"))

    X = build_matrix(train)
    y = train["Transported"].astype(int)
    align_categories(X, build_matrix(test))

    model = build_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"{X.shape[1]} features, {len(X)} rows")
    print("CV accuracy per fold:", np.round(scores, 4))
    print(f"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    model.fit(X, y)
    MODELS.mkdir(exist_ok=True)
    # Persist the category levels alongside the model: inference must encode
    # its columns exactly as training did.
    joblib.dump(
        {"model": model, "levels": {c: X[c].cat.categories for c in CATEGORICAL}},
        MODEL_PATH,
    )
    print(f"\nSaved {MODEL_PATH.relative_to(MODELS.parent)}")


if __name__ == "__main__":
    main()

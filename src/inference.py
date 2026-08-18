"""Predict the test set and write a Kaggle-ready submission.

Loads the model persisted by train.py, so run that first.

Run: python src/inference.py [nome-do-arquivo.csv]
"""

from __future__ import annotations

import sys

import joblib
import pandas as pd

from features import CATEGORICAL, SUBMISSIONS, add_features, build_matrix, load
from train import MODEL_PATH


def main() -> None:
    if not MODEL_PATH.exists():
        msg = f"{MODEL_PATH} not found. Run `python src/train.py` first."
        raise FileNotFoundError(msg)

    bundle = joblib.load(MODEL_PATH)
    model, levels = bundle["model"], bundle["levels"]

    test = add_features(load("test"))
    X_test = build_matrix(test)
    # Reuse the levels seen at fit time, so the encoding matches exactly.
    for col in CATEGORICAL:
        X_test[col] = X_test[col].cat.set_categories(levels[col])

    pred = model.predict(X_test).astype(bool)
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": pred})

    # The submission must line up row for row with sample_submission.csv.
    sample = load("sample_submission")
    if not (submission["PassengerId"].values == sample["PassengerId"].values).all():
        msg = "PassengerId order does not match sample_submission.csv"
        raise ValueError(msg)

    name = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    SUBMISSIONS.mkdir(exist_ok=True)
    path = SUBMISSIONS / name
    submission.to_csv(path, index=False)
    print(f"Wrote {path.relative_to(SUBMISSIONS.parent)}: {len(submission)} rows, "
          f"{pred.mean():.1%} predicted True")


if __name__ == "__main__":
    main()

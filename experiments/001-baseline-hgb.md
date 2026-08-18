# 001 — Baseline: HistGradientBoosting with 29 features

- **Date:** 2026-08-17
- **Commit:** `6ffa9a1` (code later moved into `src/` with no change in results)
- **Builds on:** from scratch
- **CV:** 0.8110 ± 0.0092 — 5-fold stratified, seed 42
- **Public LB:** 0.80266 (submission 55590027)

## Hypothesis

Establish a reliable floor before optimising anything. `HistGradientBoostingClassifier` was not an
arbitrary pick: it handles NaN and categoricals natively, and since ~2% of **every** column is missing
in both splits, skipping an arbitrary imputation step lets the model learn the direction each missing
value implies instead of receiving an invented median.

## What changed

First experiment. Features (29 in total):

- **Group:** `Group` and `GroupPos` parsed from `PassengerId` (`gggg_pp`), `GroupSize`, `Solo`.
- **Cabin:** `Cabin` split into `Deck` / `CabinNum` / `Side`, plus `CabinSize` (how many people share
  the exact cabin).
- **Family:** `Surname` from `Name`, with `FamilySize`. Catches families that the group id splits.
- **Spend:** `TotalSpend`, `SpendCount`, `NoSpend`, and `log1p` of the five amenity columns and the
  total — they are heavily zero-inflated with a long tail.
- **Missingness:** `NaNCount` per row, testing the missingness pattern itself as signal.
- **Age:** `IsChild` (under 13).

**The domain rule that paid off most:** cryosleep passengers are confined to their cabins and cannot
bill anything. That makes two imputations correct rather than merely convenient:

1. missing spend becomes **zero** for anyone in cryosleep;
2. anyone with spend was **not** in cryosleep.

That recovers ~200 values without guessing.

Hyperparameters: `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=30`,
`l2_regularization=1.0`, early stopping on a 10% validation split.

## Result

```
folds: [0.8125  0.7953  0.8200  0.8199  0.8072]
CV:    0.8110 +/- 0.0092
LB:    0.80266
```

Prediction distribution: 51.7% `True`, close to the training balance (50.4%) — the model is not skewed
towards one class.

**CV 0.8110 vs. LB 0.80266** — a gap of ~0.008. That sits within the standard deviation across folds
(0.0092), and the public leaderboard covers only part of the test set. No sign of meaningful
overfitting.

## Conclusion

Keep as the baseline. The number is already competitive: most of the leaderboard sits at 0.80–0.81 with
the top near 0.82, so the remaining margin is narrow and further gains will be small.

Nothing dropped yet — there is no previous experiment to compare against.

## Next step

In expected order of payoff:

1. **Richer group aggregates** — total group spend, whether the whole group is in cryosleep, the
   group's dominant home planet. These must be computed within each split, since no `gggg` crosses
   train and test; there is no groupmate label to leak.
2. **Hyperparameter search** with `RandomizedSearchCV` over `learning_rate`, `max_leaf_nodes` and
   `min_samples_leaf`.
3. **Ensemble** with LightGBM or CatBoost, averaging probabilities.

Before any of them: confirm the improvement in CV. With 10 submissions a day, it is not worth spending
one on a difference smaller than the noise.

# 003 — Randomised hyperparameter search on HistGradientBoosting

- **Date:** 2026-08-18
- **Commit:** not merged yet — pending the outcome of 004
- **Builds on:** 001 (baseline features, unchanged)
- **CV:** 0.8128 ± 0.0089 vs. 0.8118 ± 0.0112 for 001 — 5-fold × 3 repeats, seed 42
- **Public LB:** not submitted

## Hypothesis

001's hyperparameters were picked by hand and never checked. A search over the main capacity and
regularisation knobs should find something better, or confirm the hand-picked values were reasonable.

## What changed

Features untouched — this is purely the estimator. `RandomizedSearchCV`, 60 candidates, 5-fold,
scoring accuracy, over:

```
learning_rate      0.02 – 0.1
max_leaf_nodes     8 – 48
min_samples_leaf   10 – 120
l2_regularization  0 – 10
max_features       0.5 – 1.0
max_iter           300 / 600 / 1000
```

Best candidate:

```
min_samples_leaf=120   max_leaf_nodes=12   max_iter=1000
max_features=0.85      learning_rate=0.1   l2_regularization=0.5
```

The direction is consistent: **smaller trees, much stronger leaf regularisation, more of them.** Where
001 used 31 leaves with a minimum of 30 samples, the search prefers 12 leaves with a minimum of 120 —
roughly a third the capacity per tree, compensated by more iterations. On 8693 rows with a ~50/50
target, that is a sensible place to land.

## Result

```
search best_score_ (5-fold, the folds it selected on)   0.8139
honest repeated CV (5 x 3, fresh folds)                 0.8128 +/- 0.0089
001 baseline, same repeated CV                          0.8118 +/- 0.0112
```

**+0.0010 over the baseline — inside the noise band.** The variance did drop (0.0089 vs 0.0112), which
is what the extra regularisation was supposed to buy, but accuracy barely moved.

Note the gap between the two CV numbers for the *same* model: 0.8139 in the search, 0.8128 on fresh
folds. That 0.0011 is selection bias, not a real loss — taking the maximum over 60 candidates scored on
identical folds fits the estimate to those folds. Never quote a search's `best_score_` as the model's
accuracy; re-measure the winner on folds it was not selected on.

## Conclusion

Weakly positive, not yet adopted. The tuned parameters are not *worse* and are better behaved, but a
+0.001 gain does not justify a submission on its own.

Decision deferred to 004: if the blend wins, its components should be tuned anyway and these values
carry over; if the blend fails, adopt the tuned parameters in `src/train.py` as a small free
improvement.

## Next step

004 — out-of-fold probabilities for tuned HGB, LightGBM and CatBoost across three 5-fold splits, then
a weight search over the blend.

# 005 — CatBoost hyperparameter search (rejected)

- **Date:** 2026-08-18
- **Commit:** not merged — `src/` unchanged
- **Builds on:** 004
- **CV:** 0.8133 for the tuned CatBoost vs. **0.8167** for the hand-picked one — 3 × 5-fold OOF
- **Public LB:** not submitted

## Hypothesis

004 left CatBoost carrying 70% of the prediction on parameters that were guessed, never searched. That
looked like the clearest remaining headroom in the whole project.

## What changed

`RandomizedSearchCV`, 40 candidates, 5-fold, over eight parameters — including the two regularisation
levers HGB has no equivalent for (`random_strength`, `bagging_temperature`) and `one_hot_max_size`,
which decides whether low-cardinality columns like `Side` go through one-hot or ordered target
statistics.

The winner was a much smaller model than the incumbent: **depth 4** (vs. 6), **600 iterations**
(vs. 900), `learning_rate` 0.03, `l2_leaf_reg` 1.0, `one_hot_max_size` 30, `bagging_temperature` 2.0.

Stage two then re-measured the winner, the incumbent and HGB on **the same** 3 × 5-fold out-of-fold
scheme, and re-swept the blend weight — because a stronger CatBoost should pull the optimal weight
further towards itself.

## Result

```
search best_score_ (its own folds, optimistic)   0.8150

--- OOF accuracy, 3 x 5-fold, identical folds for all three ---
hgb                                              0.8126
cat_old  (hand-picked, in production)            0.8167
cat_new  (searched)                              0.8133     <- worse

--- blend sweep with the tuned CatBoost ---
  hgb 0.2 / cat 0.8   0.8144
  hgb 0.3 / cat 0.7   0.8149     <- best available with cat_new
  hgb 0.4 / cat 0.6   0.8143

current production blend (0.3 hgb / 0.7 cat_old)  0.8184
```

**The search made CatBoost worse by 0.0034**, and the best blend it could support (0.8149) falls well
short of what is already in production (0.8184).

Two things are worth reading here.

**The search never beat the incumbent even on its own folds.** Its optimistic `best_score_` was 0.8150,
below the 0.8167 the hand-picked configuration scores on fresh folds. The incumbent was not among the
40 candidates, so nothing in the run ever compared them directly — the search reported a "best" that
was never good.

**Selection bias again, same size as in 003.** The winner scored 0.8150 on the folds that selected it
and 0.8133 on fresh ones — a 0.0017 drop, consistent with 003's 0.0011. Taking the maximum over 40
candidates on one partition reliably buys about a thousandth of an accuracy point of illusion. Had the
run stopped at `best_score_` and shipped, it would have replaced a 0.8167 model with a 0.8133 one while
reporting an improvement.

## Conclusion

**Rejected.** `src/` is untouched and still runs the 004 blend. No submission was spent.

The likely mechanism: 40 candidates over eight parameters on a single 5-fold partition is a thin search
against a noise floor of ~0.004. The winner is whichever candidate happened to suit those five folds,
and depth 4 with 600 iterations is a genuinely smaller model that the fold-level noise flattered.

This does not prove CatBoost is untunable here — it shows this search was underpowered for the effect
size. Any retry needs repeated folds *inside* the search, not just at the end, which multiplies the
cost by the number of repeats.

## Next step

Cheap and structurally sound, in preference order:

1. **Seed-averaged CatBoost** — fit the current configuration under several seeds and average the
   probabilities. This reduces variance without searching anything, so there is no selection bias to
   pay for.
2. Accept 004 as the stopping point. Four of the five experiments so far have been negative, which is
   the honest signal that the remaining headroom on this dataset is thin.

# 004 — CatBoost + HistGradientBoosting blend (adopted)

- **Date:** 2026-08-18
- **Commit:** `src/` updated
- **Builds on:** 001 features, 003 hyperparameters
- **CV:** 0.8184 blended OOF (3 × 5-fold) vs. 0.8118 for 001
- **Public LB:** 0.80570 (submission 55591122) — up from 0.80266

## Hypothesis

Feature work was exhausted in 002 and tuning gave almost nothing in 003, so the remaining headroom is
in model diversity: different boosters make different errors, and averaging their probabilities should
beat any one of them.

## What changed

Out-of-fold probabilities for three boosters over the same 29 features, each averaged across **three
different 5-fold splits** (seeds 42, 7, 2024) so the comparison does not rest on one lucky partition:

- **HGB** — tuned parameters from 003.
- **LightGBM** — 700 trees, 16 leaves, `lr=0.03`, subsample 0.85, colsample 0.7.
- **CatBoost** — 900 iterations, depth 6, `lr=0.04`, `l2_leaf_reg=4`, native categorical handling.

Then a weight grid over the three sets of probabilities.

## Result

```
hgb   OOF accuracy: 0.8126
lgbm  OOF accuracy: 0.8115
cat   OOF accuracy: 0.8167     <- best single model
equal weights (1/3 each): 0.8146
best weights 0.3 hgb / 0.0 lgbm / 0.7 cat: 0.8184
```

Two things worth reading carefully.

**CatBoost alone beats everything from 001–003.** +0.004 over the tuned HGB, on the same folds. Its
ordered target statistics for categoricals extract more from `Deck`, `HomePlanet` and `Destination`
than one-hot-free histogram splitting does.

**Equal weights are worse than CatBoost alone** (0.8146 vs 0.8167). Dragging in LightGBM, the weakest
model, costs more than the diversity is worth — a blend is not automatically better than its best
component.

The weight grid picked 0.3/0.0/0.7, and that number was chosen on the same predictions it was scored
on, so it is optimistic. The sweep shows why it is still safe to use:

```
hgb 0.0 / cat 1.0   0.8167
hgb 0.1 / cat 0.9   0.8176
hgb 0.2 / cat 0.8   0.8174
hgb 0.3 / cat 0.7   0.8184     <- peak
hgb 0.4 / cat 0.6   0.8180
hgb 0.5 / cat 0.5   0.8165
```

The region from 0.1 to 0.4 is a **plateau**, not a spike. A sharp peak would mean the weight was fitted
to noise; a flat one means any value in the range works, so 0.3 is a safe pick rather than a tuned one.

Standard error on accuracy at n=8693 is ~0.0042, so the +0.0066 OOF gain over 001 is about 1.6 SE —
suggestive rather than conclusive on its own. The leaderboard agreed in direction: **0.80266 → 0.80570,
+0.0030.**

## Conclusion

**Adopted.** `src/train.py` now fits both models and persists them with the blend weight;
`inference.py` averages the probabilities. LightGBM was dropped entirely — it earned weight zero and
adding a dependency that contributes nothing is not worth it.

Both remaining libraries are OSI-licensed with no commercial-use restriction (CatBoost is Apache 2.0),
which rule 8.C requires.

## Next step

The gap between OOF (0.818) and leaderboard (0.806) is larger than in 001, which is what one expects
after selecting a model and a weight on the same folds — some of the OOF gain is selection, not skill.

Remaining ideas, in order of expected payoff:

1. **Tune CatBoost properly.** Its parameters were guessed, not searched, and it is now carrying 70% of
   the prediction. This is the clearest remaining headroom.
2. **Seed-average CatBoost** — fit several seeds and average, which reduces variance without any new
   information.
3. Threshold tuning is unlikely to help: the target is near 50/50 and predictions sit at 51.3% True.

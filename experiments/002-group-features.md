# 002 — Group imputation and group aggregates (rejected)

- **Date:** 2026-08-18
- **Commit:** not merged — the code stayed in scratch
- **Builds on:** 001
- **CV:** 0.8093–0.8105 vs. 0.8118 for the baseline — 5-fold × 3 repeats, seed 42
- **Public LB:** not submitted

## Hypothesis

Two ideas from 001's "next step" list:

1. Passenger groups should let us repair missing categorical values exactly, not approximately.
2. Group-level aggregates (total spend, whether the whole group is in cryosleep, age profile) should
   carry signal that per-passenger columns do not.

The first was measured before building anything, and the measurement was encouraging:

```
HomePlanet   varies within a group in     0 of 9124 groups (0.00%)
Side         varies within a group in     0 of 9119 groups (0.00%)
Destination  varies in 11.72%   |   Deck varies in 7.38%
```

`HomePlanet` and `Side` are **perfectly constant within a group**, so filling them from a groupmate is
exact rather than a guess. That recovers 131 of 288 missing `HomePlanet` and 137 of 299 missing `Side`,
with `Destination` and `Deck` fillable by group mode as a heuristic.

## What changed

Four variants, evaluated on identical folds:

- **A** — baseline from 001, features computed separately per split.
- **B** — features computed on train and test concatenated, which makes `FamilySize` count the whole
  family instead of only the part in one split (surnames do cross the splits, unlike group ids).
- **C** — B plus the group imputation above, plus surname → `HomePlanet` for solo travellers.
- **D** — C plus nine group aggregates: `GroupSpend`, `GroupSpendMean`, `GroupCryoFrac`,
  `GroupAllCryo`, `GroupAgeMean`, `GroupAgeMax`, `GroupHasChild`, `FamilySpend`, `CabinRegion`.

Evaluation moved to **5-fold × 3 repeats** (15 fits). Single 5-fold CV has a standard error near 0.004
on the mean, which is the same size as the effects being chased — the repeats were needed to tell a
real difference from fold noise.

## Result

```
A  baseline (per-split features)           0.8118 +/- 0.0112   (29 feats)
B  combined frame (fuller family sizes)    0.8105 +/- 0.0096   (29 feats)
C  B + group imputation                    0.8103 +/- 0.0101   (29 feats)
D  C + group aggregates                    0.8093 +/- 0.0084   (38 feats)
```

Every variant is **at or below** the baseline, and every gap is well inside the noise. Adding features
monotonically made things slightly worse, which is the usual signature of dilution: more columns to
split on, no new information.

## Conclusion

**Rejected.** All three ideas dropped; `src/` is unchanged.

The instructive part is *why* the exact imputation bought nothing. The information was never missing
from the model's point of view — `HomePlanet` correlates with deck, spend profile and destination, and
the booster was already recovering it through those. Filling the cell explicitly just moved the same
signal to a different column. A repair being *correct* does not make it *informative*.

The same holds for the aggregates: `GroupSize` and `CabinSize` were already in the baseline, and the
richer group statistics are largely functions of them plus the spend columns.

## Next step

Feature engineering looks exhausted at this level. Moving to model capacity: hyperparameter search
(003) and a multi-booster blend (004). Do not retry group features on top of a different model — the
result above is about the information, not the estimator.

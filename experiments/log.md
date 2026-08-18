# Experiment log

A record of what was tried, what worked and what did not. One file per experiment, numbered in
sequence: `001-baseline-hgb.md`, `002-….md`. The blank form is in [`TEMPLATE.md`](TEMPLATE.md). This
file is the index.

The rule that makes these notes worth keeping: **write down the failures too**. Knowing that group
imputation did not help is worth as much as knowing that cryosleep did — it stops the same path being
retried two weeks later.

## Summary

| # | Experiment | CV (accuracy) | Public LB | Outcome | Submission |
|---|---|---|---|---|---|
| [001](001-baseline-hgb.md) | HistGradientBoosting, 29 features | 0.8118 ± 0.0112 | 0.80266 | baseline | 55590027 |
| [002](002-group-features.md) | Group imputation + group aggregates | 0.8093–0.8105 | — | **rejected** | not submitted |
| [003](003-hyperparameter-search.md) | Randomised search on HGB | 0.8128 ± 0.0089 | — | folded into 004 | not submitted |
| [004](004-catboost-blend.md) | CatBoost 0.7 + HGB 0.3 blend | 0.8184 OOF | **0.80570** | **adopted** | 55591122 |

Current best: **004**, the blend now in `src/`.

## What has been ruled out

- **Group-based imputation**, even where it is provably exact (`HomePlanet` and `Side` never vary
  within a group). The information was already reachable through correlated columns — see 002.
- **Group and family aggregates** on top of `GroupSize` / `CabinSize`. Adding them made CV slightly
  worse, the usual signature of dilution.
- **LightGBM in the blend.** It earned weight zero against CatBoost and HGB — see 004.

## Context for reading the numbers

- Local CV is 5-fold stratified with `random_state=42`. Comparisons between experiments use **5 folds ×
  3 repeats**, or out-of-fold probabilities averaged over three different splits. Single 5-fold has a
  standard error near 0.004 on the mean — the same size as the effects being chased, so it cannot
  settle these questions on its own. If you change the scheme, say so in the note.
- Standard error on accuracy at n=8693 is ~0.0042. A difference under ~0.004 between two experiments
  **is not signal** — do not chase it.
- Never quote a search's `best_score_` as the model's accuracy. Selecting the maximum over many
  candidates on identical folds fits the estimate to those folds; re-measure the winner on fresh ones.
  In 003 that bias was 0.0011.
- 10 submissions per day. Only submit what CV says is a real improvement.
- To generate a submission named after an experiment: `python src/inference.py exp005.csv`.

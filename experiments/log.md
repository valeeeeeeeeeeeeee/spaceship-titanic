# Experiment log

A record of what was tried, what worked and what did not. One file per experiment, numbered in
sequence: `001-baseline-hgb.md`, `002-….md`. The blank form is in [`TEMPLATE.md`](TEMPLATE.md). This
file is the index.

The rule that makes these notes worth keeping: **write down the failures too**. Knowing that target
encoding did not help is worth as much as knowing that cryosleep did — it stops the same path being
retried two weeks later.

## Summary

| # | Experiment | CV (accuracy) | Public LB | Δ vs. previous | Commit | Submission |
|---|---|---|---|---|---|---|
| [001](001-baseline-hgb.md) | HistGradientBoosting, 29 features | 0.8110 ± 0.0092 | **0.80266** | — | `6ffa9a1` | 55590027 |

## Context for reading the numbers

- Local CV is 5-fold stratified with `random_state=42`. CV numbers only compare to each other if the
  validation scheme is the same — if you change the fold count or the seed, say so in the note.
- The public leaderboard covers only part of the test set. With ~0.009 standard deviation across folds,
  a 0.002–0.003 difference between two experiments **is not signal** — do not chase it.
- 10 submissions per day. Only submit what CV says is a real improvement.
- To generate a submission named after an experiment: `python src/inference.py exp002.csv`.

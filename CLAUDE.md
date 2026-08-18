# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Kaggle **Spaceship Titanic** working folder (Getting Started competition, sponsored by Google LLC,
https://www.kaggle.com/c/spaceship-titanic). The repo holds a working end-to-end solution in `src/`, with
the competition CSVs untracked in `data/`.

**The task:** predict whether a passenger was transported to an alternate dimension during the
Spaceship Titanic's collision with a spacetime anomaly. Binary classification on `Transported` (bool),
from personal records recovered from the ship's damaged computer system.

## Environment

System Python already has what is needed — no venv or install step is set up:

```
python 3.14  |  pandas 2.3.3  |  scikit-learn 1.9.0  |  numpy 2.3.5  |  kagglehub 1.0.2
```

```bash
python src/train.py       # 5-fold stratified CV, refit, models/model.joblib
python src/inference.py   # submissions/submission.csv (pass a filename to rename)
```

Both take well under a minute. `train.py` and `inference.py` import `features.py` by plain module name,
which works because running a script puts its own directory on `sys.path` — run them as
`python src/train.py`, not as `-m src.train`. There are no tests, linters, or build targets to invoke;
if you add any, document them here.

## Current baseline

`src/` scores **0.8110 +/- 0.0092** CV accuracy (public LB **0.80266**, submission 55590027) with a `HistGradientBoostingClassifier` over 29
engineered features. It handles NaN and categoricals natively, so there is no imputation or one-hot
step ahead of it — keep it that way unless you switch model families.

The one domain rule the feature code leans on: cryosleep passengers are confined to their cabins and
cannot bill anything, so missing spend is zero for them and any passenger with spend was not in
cryosleep. That recovers ~200 values without guessing. Beat the baseline before adding complexity —
target encoding and stacking have historically bought very little here.

## Recording experiments

Every attempt gets a note in `experiments/`, numbered in sequence, with the summary table kept current
in `experiments/log.md` — CV, public LB, commit and submission id. Record the failures too: knowing
that an idea did not pay off is what stops it being retried weeks later. `experiments/TEMPLATE.md` is
the blank form.

### Re-downloading the data

The CSVs are the official competition files, fetched with `kagglehub` and also cached at
`~/.cache/kagglehub/competitions/spaceship-titanic/`:

```python
import kagglehub
path = kagglehub.competition_download('spaceship-titanic')
```

This needs a Kaggle API token in the environment (`KAGGLE_API_TOKEN`, the `KGAT_…` form — no username
required) or a legacy `~/.kaggle/kaggle.json`. The account must have accepted the competition rules
first, or the download 403s. Do not write the token into any file in this folder.

## Data contract

| File | Rows | Cols | Notes |
|---|---|---|---|
| `train.csv` | 8693 | 14 | labelled (~two-thirds of passengers); target `Transported` ~50/50 balanced (50.4% True) |
| `test.csv` | 4277 | 13 | unlabelled (~one-third); identical to train minus `Transported` |
| `sample_submission.csv` | 4277 | 2 | `PassengerId,Transported`, already in the same row order as `test.csv` |

Train and test differ **only** by the target column — every feature available at fit time is available
at predict time.

### Field semantics (official)

- **`PassengerId`** — unique id of the form `gggg_pp`, where `gggg` is the group the passenger travels
  with and `pp` is their number within that group. Group members are often family, but not always.
  9280 groups across 12970 passengers.
- **`HomePlanet`** — planet departed from, typically the passenger's planet of permanent residence.
  Values: Europa, Earth, Mars.
- **`CryoSleep`** — whether the passenger elected suspended animation for the voyage. Cryosleep
  passengers are **confined to their cabins**, so they cannot bill any amenity; this interaction with
  the spend columns is the dataset's strongest structural signal.
- **`Cabin`** — `deck/num/side` (e.g. `B/0/P`), where side is `P` for Port or `S` for Starboard.
  Measured decks: `A`–`G` plus `T`. Split into three features rather than using it raw.
- **`Destination`** — planet the passenger debarks to. Values: TRAPPIST-1e, PSO J318.5-22, 55 Cancri e.
- **`Age`** — passenger age.
- **`VIP`** — whether the passenger paid for special VIP service.
- **`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`** — amount billed at each of the ship's
  luxury amenities. Heavily zero-inflated; a total-spend feature and a spent-anything flag are the
  usual first derivations.
- **`Name`** — first and last names. The surname is the usual route to family grouping, and it survives
  in test, so surname-level aggregates are legitimate features.
- **`Transported`** — whether the passenger was transported to another dimension. The target.

### Two measured facts worth knowing before feature engineering

- **The split is by group — no `gggg` appears in both train and test** (verified: zero overlap of group
  ids). So "look up a groupmate's known label" is not available, and any group feature must be built
  from within-split aggregates (group size, group spend, group home planet), not from leaked targets.
  Same for `PassengerId` itself: no id appears in both files.
- **Missing values are everywhere.** Every column except `PassengerId` (and `Transported`) has ~2% NaN
  in **both** splits — 179–217 per column in train, 80–106 in test. Imputation is mandatory, not
  optional, and the NaN pattern is worth testing as a feature in its own right.

## Submission format

Match `sample_submission.csv` exactly: two columns, `PassengerId` then `Transported`, 4277 rows, in the
same row order as `test.csv` (verified aligned). `Transported` must serialise as `True`/`False`, not
`1`/`0` — write a bool column and let pandas handle it.

## Competition rules that constrain the work

Full official text in `COMPETITION_RULES.md`. The clauses that actually bear on what gets written here:

- **10 submissions per day**, one Kaggle account per participant. Don't churn out throwaway submission
  files expecting to try them all — the budget is real. Final-submission selection does not apply
  (Getting Started competition).
- **No hand labelling.** Submissions may not incorporate hand labelling or human prediction of the
  validation or test records. Never hardcode per-`PassengerId` outcomes or reverse-engineer labels
  from a public source; predictions must come from the model.
- **No private code sharing.** Competition code must not be shared privately outside the team. Sharing
  publicly is allowed *only* on this competition's Kaggle forum or notebooks, and doing so licenses it
  under an OSI-approved licence that does not limit commercial use.
- **Open source only, commercially usable.** Any open-source code used to generate a submission must be
  under an OSI-approved licence with no commercial-use restriction. Check before adding a dependency.
- **External data** is allowed only if it is publicly available and equally accessible to all
  participants at no cost. Anything private, paid, or scraped-and-unshared is out.
- **AutoML tools** (Google AutoML, H2O Driverless AI, etc.) are explicitly permitted, provided the
  licence allows compliance with the rules.
- **Data security.** Do not redistribute the competition CSVs to anyone who has not accepted the rules.
  If this folder ever becomes a git repo, keep the CSVs out of any public remote.
- **Leaderboard decay.** A submission older than two months is invalidated and stops counting; a team
  with no submissions in two months drops off the leaderboard entirely.
- **Timeline:** started Feb 23, 2022. No merger deadline, no entry deadline, no end date.
- **Data use** is licensed for competition use and academic, non-commercial use only.

# Spaceship Titanic — Kaggle

Solution for the [Spaceship Titanic](https://www.kaggle.com/c/spaceship-titanic) competition (Getting
Started, sponsored by Google LLC).

**The task:** predict whether a passenger was transported to an alternate dimension during the
Spaceship Titanic's collision with a spacetime anomaly. Binary classification on `Transported`, from
personal records recovered from the ship's damaged computer system.

> **Status:** CatBoost + HistGradientBoosting blend in `src/`, at **0.8184** out-of-fold accuracy.
> Public leaderboard: **0.80570**.

## Running it

```bash
python src/train.py            # cross-validation, final fit, model.joblib
python src/train.py --no-cv    # skip cross-validation (much faster)
python src/inference.py        # submissions/submission.csv
```

`train.py` reports out-of-fold accuracy for each model and for the blend, then persists both models
with their blend weight; `inference.py` loads them, averages the probabilities, and writes the
submission, checking its row order against `sample_submission.csv`. Both import `features.py`, so
training and prediction see exactly the same columns. Neither model needs imputation or one-hot
encoding — both handle NaN and categoricals natively.

The prediction is **0.7 × CatBoost + 0.3 × HistGradientBoosting**. CatBoost is the stronger of the two
on its own; the weight sits on a plateau rather than a tuned point, so it is not fragile. See
[`experiments/004`](experiments/004-catboost-blend.md).

To name a submission: `python src/inference.py exp005.csv`.

Feature engineering leans on the structure of the problem: `Cabin` split into deck/num/side, group and
surname aggregates, `log1p` spend totals, and the row's NaN count. The central piece is cryosleep —
those passengers are confined to their cabins and cannot bill anything, so missing spend is zero for
them, and any passenger with spend was not in cryosleep.

Richer group features were tried and rejected; read
[`experiments/002`](experiments/002-group-features.md) before reaching for them again.

## Data

The CSVs are **not tracked** — the competition rules forbid redistributing the data to anyone who has
not accepted the terms. Fetch them from Kaggle:

```python
import kagglehub
path = kagglehub.competition_download('spaceship-titanic')
```

Then copy `train.csv`, `test.csv` and `sample_submission.csv` from the returned path
(`~/.cache/kagglehub/competitions/spaceship-titanic/`) into `data/`.

This needs a Kaggle API token in the environment, and the account must have accepted the rules on the
competition page — otherwise the download 403s:

```bash
# current format (KGAT_…), no username needed
export KAGGLE_API_TOKEN="KGAT_…"
# or the legacy file: ~/.kaggle/kaggle.json
```

Never write the token into any file in this repository.

### Files

| File | Rows | Cols | Description |
|---|---|---|---|
| `train.csv` | 8693 | 14 | ~2/3 of the passengers, with the `Transported` target (~50/50, 50.4% True) |
| `test.csv` | 4277 | 13 | ~1/3 of the passengers, identical to train minus the target |
| `sample_submission.csv` | 4277 | 2 | submission format, already in the same row order as `test.csv` |

### Columns

- **`PassengerId`** — unique id of the form `gggg_pp`, where `gggg` is the group the passenger travels
  with and `pp` is their number within it. Group members are often family, but not always.
- **`HomePlanet`** — planet departed from, usually the planet of permanent residence. *Europa, Earth,
  Mars.*
- **`CryoSleep`** — whether the passenger chose suspended animation for the voyage. Passengers in
  cryosleep are **confined to their cabins** and therefore bill nothing.
- **`Cabin`** — `deck/num/side`, where *side* is `P` (Port) or `S` (Starboard). Decks `A` to `G`, plus `T`.
- **`Destination`** — planet of debarkation. *TRAPPIST-1e, PSO J318.5-22, 55 Cancri e.*
- **`Age`** — passenger age.
- **`VIP`** — whether they paid for VIP service on the voyage.
- **`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`** — amount billed at each of the ship's
  luxury amenities. Heavily concentrated at zero.
- **`Name`** — first and last name. Present in test too, so surname aggregates are viable.
- **`Transported`** — whether the passenger was transported to another dimension. **The target.**

### Two things measured in the data

- **The train/test split is by group:** no `gggg` appears in both files. A groupmate's known label is
  not available — group features must be within-split aggregates (size, spend, planet), with no target
  leakage.
- **Missing values everywhere:** every column except `PassengerId` has ~2% NaN in both splits (179–217
  per column in train, 80–106 in test). Imputation is mandatory, and the missingness pattern itself is
  worth testing as a feature.

## Environment

Python 3.14 with:

```
pandas 2.3.3   scikit-learn 1.9.0   numpy 2.3.5   catboost 1.2.10   kagglehub 1.0.2
```

## Submission

The file needs two columns — `PassengerId` and `Transported` — with 4277 rows in the same order as
`test.csv`. `Transported` must serialise as `True`/`False`, not `1`/`0`.

The limit is **10 submissions per day**.

## Competition rules

The full official text is in [`COMPETITION_RULES.md`](COMPETITION_RULES.md). The points that bear on
the code in this repository:

- **Hand labelling** and human prediction of the test records are not permitted.
- Competition code **may not be shared privately** outside the team; public sharing must happen on
  Kaggle's forum or notebooks, under an OSI-approved licence.
- Open source dependencies need an OSI-approved licence with no commercial-use restriction.
- **External data** is allowed only if it is public and equally accessible to all participants, free of
  charge.
- **AutoML** tools are explicitly permitted.
- The data may not be redistributed — hence the `.gitignore`.

## Repository layout

```
├── src/
│   ├── features.py        feature engineering, shared
│   ├── train.py           cross-validation, final fit, saved model
│   └── inference.py       prediction and submission file
├── experiments/
│   ├── log.md             index and summary table
│   ├── TEMPLATE.md        blank form for a new note
│   └── 00N-*.md           one note per experiment
├── data/                  Kaggle CSVs (untracked, created on run)
├── submissions/           submission files (untracked)
├── model.joblib           trained blend (untracked, written by train.py)
├── CLAUDE.md              guidance for Claude Code
├── COMPETITION_RULES.md   full official rules
├── README.md
└── .gitignore
```

## Recording experiments

Every attempt becomes a note in [`experiments/`](experiments/log.md), with the summary table in
`log.md`. Record what did **not** work too — that is what stops the same path being retried later.

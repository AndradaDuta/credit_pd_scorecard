# Credit PD Scorecard — interpretable-first, with ML challengers

A probability-of-default model built for the second-line risk:
an interpretable WoE/logistic **scorecard** as the anchor, with **XGBoost** and a
small **neural-net** challenger used to *measure* — not assume — the cost of
interpretability. Data is staged and feature-engineered in **SQL (DuckDB)** with an
explicit leakage-safe contract, and the model is validated **out-of-time**.

> One-line takeaway (fill after results): *On <dataset>, the scorecard reaches
> Gini <X.XX> vs XGBoost <X.XX> — a <Y> Gini-point gap — while remaining fully
> auditable. The NN challenger did not beat the trees.*

---

## Why this is built this way
- **PD, not generic classification.** Target = default within a **12-month** horizon;
  default defined as `<charged-off / 90+ DPD>` (Basel Art. 178).
- **Interpretable anchor.** WoE binning → logistic regression → point-scaled scorecard,
  the industry-standard auditable form (IFRS 9 / Basel IRB).
- **Honest challengers.** XGBoost + SHAP capture interactions; a small MLP tests whether
  deep learning adds anything on tabular data of this size (it usually doesn't).
- **Validator-grade discipline.** Origination-time features only; out-of-time split;
  discrimination + calibration + stability all reported.

## Headline results  *(fill in)*
| Model            | AUC | Gini | KS | Brier | PSI (train→OOT) |
|------------------|-----|------|----|-------|-----------------|
| WoE scorecard    |     |      |    |       |                 |
| XGBoost + SHAP   |     |      |    |       |                 |
| Small NN         |     |      |    |       |                 |

_One or two sentences interpreting the gap and why the interpretable model is the
production choice._

## Data
- Source: `<Lending Club / Home Credit / Freddie Mac>` — link + licence.
- **Leakage rule:** only fields known at origination enter the modelling frame;
  post-outcome fields (recoveries, last-payment info, total payments) are dropped in SQL.
- Split: train on `<vintages ≤ split_date>`, test out-of-time on `<vintages > split_date>`.

## Method
1. **SQL layer (`src/pipeline.py`)** — DuckDB stages raw CSV → cleans & filters to
   origination-time columns → engineers features → writes an out-of-time train/test parquet.
2. **Binning (`src/binning.py`)** — WoE/IV via `optbinning`, monotonic where sensible;
   IV used for feature selection (flag IV > 0.5 as likely leakage).
3. **Scorecard (`src/scorecard.py`)** — logistic regression on WoE inputs; coefficient
   signs checked; scaled to points (PDO = `<20>`, offset = `<...>`).
4. **Challengers (`src/challengers.py`)** — XGBoost + SHAP; small MLP.
5. **Metrics (`src/metrics.py`)** — AUC/Gini/KS, calibration curve + Brier + Hosmer–Lemeshow,
   PSI. *(These are reused by the companion validation-suite repo.)*

## Reproduce
```bash
pip install -r requirements.txt
python -m src.pipeline          # builds data/processed/*.parquet
jupyter lab                     # run notebooks/01 → 04 in order
```

## Caveats
- Open-data default rate is not through-the-cycle; PDs are calibrated to the sample
  central tendency, noted as a limitation.
- `<other honest limitations>`

## Companion repo
`credit-model-validation-suite` — an independent second-line validation of this exact model
(discrimination, calibration, stability, challenger benchmark, validation report).

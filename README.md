# Credit PD Scorecard — interpretable-first, with ML challengers

A 12-month probability-of-default model for consumer loans, built the way a
second-line validator expects to see it: an interpretable WoE/logistic
**scorecard** as the anchor, with **XGBoost** and a small **neural-net**
challenger used to *measure* the cost of interpretability rather than assume it.
Data is staged in **SQL (DuckDB)** under an explicit leakage contract, and every
model is assessed **out-of-time**.

> **Status (October 2026): in development.** The data pipeline, target
> definition, leakage audit and external benchmark are complete. The scorecard
> and challengers come next (see [Roadmap](#roadmap)). No model results are
> reported until they exist out-of-time.

| Component | Status |
|---|---|
| DuckDB pipeline: raw CSV → leakage-safe frame → out-of-time train/test parquet | Done |
| 12-month default target (charged off / 90+ DPD) | Done |
| Leakage audit: all 151 source columns reviewed, 39 post-origination fields excluded in SQL | Done |
| Out-of-time split at 1 July 2016 | Done |
| External benchmark: Lending Club's own pricing (`int_rate`) | Done |
| WoE/IV binning → scorecard → metrics → challengers | Next |

---

## Data

- **Source:** Lending Club accepted loans, 2007–2018Q4 (Kaggle
  `wordsforthewise/lending-club`, file `accepted_2007_to_2018Q4.csv.gz`).
  Not redistributed here; place the file in `data/raw/`.
- **Population:** originations up to December 2017, so every loan has a fully
  observable 12-month horizon before the data end (December 2018). Loans flagged
  "does not meet the credit policy" are excluded.

| Sample | Originations | Loans | 12-month default rate |
|---|---|---|---|
| Train | 2007 – Jun 2016 | 1,116,432 | 3.90% |
| Out-of-time test | Jul 2016 – Dec 2017 | 646,245 | 4.50% |

## Target definition

A loan is a **12-month default** if it ended *Charged Off* or *Default* **and**
its last payment fell within 9 months of origination, i.e. it would have reached
90 days past due inside the 12-month window (90 DPD ≈ three missed monthly
payments). Loans with no payment at all count as defaults from month 0.

This follows the spirit of the CRR Art. 178 definition (90 DPD / unlikely to
pay), but the delinquency date is *inferred* from payment dates rather than
observed. See [Validator's notes](#validators-notes).

## The leakage contract

The SQL cleaning step in `src/pipeline.py` is the single place where the feature
set is decided:

- Fields recorded after origination (payments, recoveries, settlement and
  hardship flags, last credit pull, …) are dropped in SQL **before any modelling
  code sees the data**. The list was built by auditing all 151 source columns and
  holds 39 fields.
- One documented exception: outcome fields (`loan_status`, `last_pymnt_d`) are
  used once to construct the target, then excluded. Defining the label from
  outcomes is required; predicting from them is leakage.
- **Lending Club's `grade`, `sub_grade` and `int_rate` are deliberately not
  features.** They are known at origination, so this is not leakage, but they
  are the lender's own model output. Using them would turn the scorecard into a
  partial re-estimate of Lending Club's model instead of an independent
  assessment built from applicant and bureau data.
- Backstops: the out-of-time split guards against the time dimension of leakage,
  and any feature with IV > 0.5 will be treated as a leakage alarm.

## External benchmark

Ranking loans by Lending Club's interest rate gives the lender's own view of
risk, measured on this target and this split:

| Lending Club `int_rate` | Gini |
|---|---|
| Train | 0.386 |
| Out-of-time | 0.392 |

This is a **reference point, not a contest.** Lending Club updated its pricing
during the test period, whereas the scorecard will be frozen at the split date,
and its pricing draws on information beyond the 151 public columns. Gini is
computed in DuckDB via the Mann–Whitney rank-sum identity, with average ranks for
ties (`int_rate` has only ~700 distinct values).

## Results

| Model | AUC | Gini | KS | Brier | PSI (train → OOT) |
|---|---|---|---|---|---|
| Benchmark: LC `int_rate` | | 0.392 | | n/a | |
| WoE scorecard | pending | | | | |
| XGBoost + SHAP | pending | | | | |
| Small NN | pending | | | | |

All figures will be out-of-time. The headline is the Gini gap between the
scorecard and XGBoost: added complexity has to justify itself in measurable
performance before it is preferred over an auditable model.

## Validator's notes

Known limitations and open challenges, written the way a second-line review
would raise them: the issue, the expected direction of bias, and how it is or
will be tested.

1. **Default is inferred, not observed.** There is no monthly delinquency
   history, so 90 DPD is approximated as last payment + 3 months. If
   `last_pymnt_d` on charged-off loans reflects later receipts, the inferred
   default date shifts later and some 12-month defaults are missed (downward
   bias). To be tested on a sample of charged-off loans.
2. **Late accounts at the data end are labelled non-default.** Only the youngest
   vintages (late 2017) can hold 12-month defaults that have not yet been charged
   off, so any bias sits in the tail of the out-of-time sample. A per-quarter
   default-rate check showed no drop-off in late-2017 vintages, so the effect is
   judged small, but it is a known downward bias.
3. **Prepayment is a competing risk.** A loan repaid in full in month 4 counts as
   a 12-month non-default although it was only exposed for four months. The
   observed default rate therefore depends on prepayment speeds as well as credit
   quality: if prepayments slow, the same book shows more defaults. A
   competing-risks view (separate hazards for default and prepayment) is the
   natural extension.
4. **Accepted applicants only.** There are no rejected applications and no
   reject inference, so the model describes the population Lending Club
   approved, not the through-the-door population.
5. **Calibration will need adjusting out-of-time.** The out-of-time default rate
   (4.50%) is above train (3.90%), so a model calibrated on train will
   under-predict on average. The implied intercept shift is
   logit(0.045) − logit(0.039) ≈ +0.15, about 4 points at PDO 20. Whether this is
   population drift or economic drift will be separated in S4 using PSI and an
   out-of-time reliability curve. (In-sample calibration is not informative: the
   logistic intercept forces mean predicted PD to equal the observed rate.)
6. **Not through-the-cycle.** Very few loans originate in 2007–2009, so the
   sample under-represents a downturn. PDs reflect the sample's central tendency,
   not a long-run average with a margin of conservatism.
7. **Benchmark is not like-for-like** (see above).

## Reproduce

```bash
pip install -r requirements.txt
python -m src.pipeline   # writes data/processed/{train,test}.parquet,
                         # prints sample sizes, default rates and the benchmark Gini
```

## Roadmap

- **S2** — EDA; WoE/IV binning with `optbinning` (`src/binning.py`), monotonic
  where economically sensible.
- **S3** — Logistic scorecard on WoE inputs (`src/scorecard.py`): coefficient-sign
  diagnostic, points scaling at PDO 20; AUC/Gini/KS on train and out-of-time
  (`src/metrics.py`).
- **S4** — Calibration and stability: reliability curve, Brier, Hosmer–Lemeshow,
  PSI.
- **S5** — XGBoost + SHAP challenger on the same split.
- **Later** — small NN challenger; PD quantification (intercept recalibration,
  margin of conservatism, master-scale grades); link to ECL and IRB RWA; short
  model-development document.
- **Companion repo (planned):** `credit-model-validation-suite`, an independent
  second-line validation of this model.

---

MIT licence (code). Data subject to Lending Club / Kaggle terms.

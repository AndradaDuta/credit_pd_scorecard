"""
DuckDB data pipeline for the credit PD scorecard.

Design intent (this is what a validator reads first):
  1. Raw CSV is registered, never mutated.
  2. The CLEANING query is the *leakage contract*: only origination-time columns
     survive as features. Post-outcome fields are dropped here, before any
     modelling code can ever see them.
     >> Nuance: outcome fields (loan_status, last_pymnt_d) are used ONCE, to
        construct the TARGET, and are then excluded. Defining the label from
        outcomes is required; using outcomes as predictors is leakage.
  3. Feature engineering happens in SQL so the transformation is inspectable
     and re-runnable, not buried in notebook cells.
  4. The split is OUT-OF-TIME (by origination date), not random.

Flavoured for the Kaggle Lending Club accepted-loans file (2007-2018Q4);
adapt column names for Home Credit / Freddie Mac.

Run:  python -m src.pipeline   (or: python pipeline.py)
"""

from pathlib import Path
import duckdb

# ----------------------------------------------------------------------------- #
# Config  (promote to config.yaml once stable)
# ----------------------------------------------------------------------------- #
RAW_CSV      = Path("data/raw/accepted_2007_to_2018Q4.csv.gz")  # DuckDB reads .gz directly
OUT_DIR      = Path("data/processed")
DATA_END     = "2018-12-31"     # last date the data can observe outcomes
HORIZON_MTHS = 12               # PD horizon
DPD_LAG_MTHS = 3                # 90 DPD ~= 3 months after the last payment
SPLIT_DATE   = "2016-07-01"     # out-of-time boundary (train < split <= test)

# Post-outcome columns: must NEVER reach the model as features.
# Audited against the full 151-column schema in S1. When in doubt, exclude.
LEAKAGE_COLS = [
    # loan performance after origination
    "loan_status", "pymnt_plan", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "recoveries", "collection_recovery_fee",
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "out_prncp", "out_prncp_inv",
    # credit bureau pulled AFTER origination (refreshed FICO)
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    # hardship programme (post-origination)
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "hardship_length", "hardship_dpd",
    "hardship_loan_status", "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    # debt settlement (post-default)
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage", "settlement_term",
]
# Default = 90+ DPD / charged off (Basel Art. 178 flavour). 'Late (31-120 days)'
# is deliberately excluded: it mixes pre- and post-90-DPD accounts.
DEFAULT_STATUSES = ["Charged Off", "Default"]


def build(con: duckdb.DuckDBPyConnection) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

      # --- 1. STAGE: register raw CSV, no mutation ------------------------------ #
    # Force the two date strings to VARCHAR so strptime is deterministic.
    # null_padding: the Kaggle file has a few one-field summary lines; pad them
    # with NULLs instead of erroring (they're dropped later: loan_status is NULL).
    con.execute(f"""
        CREATE OR REPLACE VIEW raw AS
        SELECT * FROM read_csv_auto(
            '{RAW_CSV.as_posix()}',
            SAMPLE_SIZE=-1,
            null_padding=true,
            types={{'issue_d': 'VARCHAR', 'last_pymnt_d': 'VARCHAR'}}
        );
    """)

    # --- 2. CLEAN: target construction + the leakage contract ----------------- #
    # 12-month default flag: a loan is a default if it ended Charged Off/Default
    # AND its last payment fell within (HORIZON - DPD_LAG) months of origination,
    # i.e. it would have reached 90 DPD inside the 12-month window. Loans with no
    # payment at all (last_pymnt_d NULL) are defaults from month 0.
    # Cohort restriction: only originations old enough to observe the full horizon.
    default_list = ", ".join(f"'{s}'" for s in DEFAULT_STATUSES)
    drop_list    = ", ".join(f'"{c}"' for c in LEAKAGE_COLS)
    obs_cutoff   = f"DATE '{DATA_END}' - INTERVAL {HORIZON_MTHS} MONTH"

    con.execute(f"""
        CREATE OR REPLACE TABLE clean AS
        WITH base AS (
            SELECT *,
                   strptime(issue_d, '%b-%Y')                          AS orig_date,
                   COALESCE(try_strptime(last_pymnt_d, '%b-%Y'),
                            strptime(issue_d, '%b-%Y'))                AS last_pay_date
            FROM raw
            WHERE loan_status NOT LIKE 'Does not meet%'
        )
        SELECT
            * EXCLUDE ({drop_list}, last_pay_date),
            CASE
                WHEN loan_status IN ({default_list})
                 AND date_diff('month', orig_date, last_pay_date)
                     <= {HORIZON_MTHS - DPD_LAG_MTHS}                  THEN 1
                ELSE 0
            END AS target
        FROM base
        WHERE orig_date <= {obs_cutoff};      -- full 12-month horizon observable
    """)

    # --- 3. FEATURES: origination-time only, engineered in SQL ---------------- #
    # TODO: extend. Every feature must be knowable at loan origination.
    con.execute("""
        CREATE OR REPLACE TABLE features AS
        SELECT
            target,
            orig_date,
            loan_amnt,
            TRY_CAST(REPLACE(CAST(int_rate AS VARCHAR), '%', '') AS DOUBLE)  AS int_rate,
            annual_inc,
            dti,
            CASE WHEN emp_length LIKE '<%' THEN 0          -- '< 1 year' -> 0, not 1
                 ELSE TRY_CAST(regexp_extract(emp_length, '\\d+') AS INTEGER)
            END                                                              AS emp_years,
            CAST(regexp_extract(term, '\\d+') AS INTEGER)                      AS term_months,
            loan_amnt / NULLIF(annual_inc, 0)                                  AS loan_to_income,
            grade,
            home_ownership,
            purpose,
            verification_status
        FROM clean;
    """)
        # --- 4. OUT-OF-TIME SPLIT ------------------------------------------------- #
    # orig_date is kept in the files as a split/monitoring key (EDA by vintage,
    # PSI by period). It is NOT a model feature: modelling code must drop it.
    con.execute(f"""
        COPY (SELECT * FROM features
              WHERE orig_date <  DATE '{SPLIT_DATE}')
        TO '{(OUT_DIR / "train.parquet").as_posix()}' (FORMAT PARQUET);
    """)
    con.execute(f"""
        COPY (SELECT * FROM features
              WHERE orig_date >= DATE '{SPLIT_DATE}')
        TO '{(OUT_DIR / "test.parquet").as_posix()}' (FORMAT PARQUET);
    """)
    # --- 5. SANITY CHECKS ----------------------------------------------------- #
    for name, cond in [("train", f"< DATE '{SPLIT_DATE}'"),
                       ("test",  f">= DATE '{SPLIT_DATE}'")]:
        n, dr = con.execute(f"""
            SELECT COUNT(*), AVG(target) FROM features WHERE orig_date {cond};
        """).fetchone()
        print(f"{name:>5}: n={n:>9,}  default_rate={dr:.4f}")


if __name__ == "__main__":
    con = duckdb.connect()
    build(con)
    print("Wrote train.parquet / test.parquet to", OUT_DIR)

# SBA Loan Default Prediction & Causal Analysis of Guarantee-Rate Thresholds

## Business question

The SBA guarantees 85% of 7(a) loans of $150,000 or less, and 75% of loans above that 
threshold. Lower guarantee coverage means the lender retains more risk on the loan. 
Standard moral hazard theory predicts lenders should underwrite more carefully when 
they bear more of the downside — implying loans just above $150k should default less 
often than loans just below it, holding everything else constant.

This project asks two separate questions on the same dataset:
1. **Causal**: Does crossing the $150k threshold actually change default rates?
2. **Predictive**: Can loan-level characteristics known at origination predict default 
   well enough to be operationally useful?

## Data

SBA 7(a) FOIA loan-level data (data.sba.gov), FY2010–2019 vintage file.
- 545,751 raw loan records
- Filtered to loans with a resolved outcome (`P I F` = paid in full, `CHGOFF` = charged off), 
  excluding still-active (`CURR`), cancelled, and in-process loans, since unresolved 
  loans have no observable default outcome yet: **422,542 resolved loans**
- Overall default rate on resolved loans: **5.65%**

An earlier, more recent vintage (FY2020–present) was evaluated and rejected: 57% of 
those loans were still active/unresolved (too immature to have a reliable outcome), 
and the sample spans a period when COVID-era legislation temporarily raised the 
guarantee rate to a flat 90%, which would have contaminated the $150k discontinuity 
being tested.

## Causal analysis: regression discontinuity design

**Running variable**: gross loan approval amount. **Cutoff**: $150,000.

**Problem found and corrected**: the raw distribution of loan amounts shows severe 
bunching exactly at $150,000 (964 loans in the $2,000 window just below the cutoff vs. 
26,185 in the equivalent window just above it) — lenders and borrowers are clearly 
targeting the round number itself, not distributing smoothly around it. A naive RDD 
on this running variable is invalid. This was diagnosed via density inspection before 
any effect was estimated, not discovered after the fact.

**Fix applied**: a donut RDD, excluding all loans within $500 of the exact cutoff, 
combined with `rdrobust`'s built-in mass-points correction (loan amounts cluster at 
many round numbers, not just $150k, which violates the continuous-density assumption 
underlying standard variance estimation).

**Confound found and corrected**: loan term (`terminmonths`) is also discontinuous at 
$150k (robust p = 0.0099 in a covariate-balance check) — loans just above the cutoff 
have systematically different terms, not just different guarantee rates. This means a 
univariate RDD conflates the guarantee-rate effect with a loan-term effect. The 
reported result below controls for `terminmonths` directly in the RD regression.

**Result**: at the MSE-optimal bandwidth (~$11,267), controlling for loan term, 
loans above the $150k threshold show a lower default rate:
- Conventional estimate: p = 0.035 (significant at 5%)
- Robust bias-corrected estimate: p = 0.067 (borderline at 5%, significant at 10%)
- Direction is consistent with the moral hazard hypothesis: less guarantee coverage → 
  more lender skin in the game → lower default.

**Validity checks performed**:
- Placebo cutoffs at $125,000 and $175,000 (no SBA rule change at either) — used to 
  confirm the $150k result isn't an artifact of general noise in the running variable
- Continuous severity outcome (charge-off amount as % of loan) as a robustness check 
  alongside the binary default flag, since a rare binary outcome (5.6% base rate) is 
  a weak signal for RD estimation on its own

**Honest framing**: this is a real but not overwhelming effect — the robust p-value 
sits right at the conventional significance boundary. It should be read as suggestive 
evidence of moral hazard, not a definitive, high-powered finding.

## Predictive model

**Features**: log loan amount, guarantee percentage, loan term (years), business age 
(new vs. existing), jobs supported, fixed vs. variable rate, initial interest rate, 
top-15 NAICS 2-digit sector dummies.

**Models compared**:
| Model | AUC | PR-AUC |
|---|---|---|
| Logistic regression (baseline, class-weighted) | 0.876 | 0.397 |
| XGBoost (challenger) | 0.969 | 0.843 |

**Leakage checks** (both passed):
1. Removed `initialinterestrate` (a lender-set variable that could partly encode the 
   lender's own risk assessment at origination) — AUC dropped only marginally 
   (0.969 → 0.965), ruling this out as the source of the high performance.
2. Re-split train/test by borrower name rather than randomly by row, so no borrower 
   appears in both sets — AUC held at 0.965, confirming the model isn't simply 
   memorizing repeat borrowers.

The model's strong performance appears to be genuinely driven by loan structure 
variables (loan size, guarantee percentage, term) rather than leakage.

## Known limitation — disclosed, not resolved

The final modeling dataset (32,770 rows) is substantially smaller than the resolved 
dataset (422,542 rows) after feature construction and null-filtering. This drop has 
not yet been fully decomposed by column — it is not yet established which feature(s) 
are driving the loss, and this should be treated as an open item rather than a 
resolved one. Flagging this explicitly rather than presenting the 32,770-row model as 
representative of the full resolved population without qualification.

## Stack

Python (pandas, scikit-learn, XGBoost, SHAP), `rdrobust` (R-derived RDD estimation via 
Python port), PostgreSQL, FastAPI, Streamlit, pytest, GitHub Actions.

## Repository structure

- `ingest.py` — raw file loading with delimiter auto-detection
- `clean.py` — outcome filtering, default flag construction
- `build_features.py` — feature engineering, NAICS grouping
- `train_model.py` — baseline + XGBoost training, model persistence
- `run_pipeline.py` — single entry point running the full pipeline
- `api.py` — FastAPI `/predict` endpoint serving the trained model
- `dashboard.py` — Streamlit interactive scoring tool with SHAP waterfall explanation
- `test_pipeline.py` — pytest suite validating cleaning and feature logic
- `.github/workflows/tests.yml` — CI running the test suite on every push

## Running it
python run_pipeline.py        # ingest, clean, engineer features, train, save model
uvicorn api:app --reload      # serve predictions at /predict
streamlit run dashboard.py    # interactive dashboard
python -m pytest test_pipeline.py -v   # run test suite

## SQL

Loan and outcome data also loaded into PostgreSQL to demonstrate relational query 
work: vintage-cohort default rates with rolling window functions, state-level 
aggregation with `HAVING` filters, CTE-based above-average-risk sector identification, 
CASE-based loan-size tiering, and indexed join performance verified via `EXPLAIN ANALYZE`.
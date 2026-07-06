# SBA Loan Default Prediction & Causal Analysis of Guarantee-Rate Thresholds

## Business question

The SBA guarantees 85% of 7(a) loans of $150,000 or less, and 75% of loans above that 
threshold. Lower guarantee coverage means the lender retains more risk. Standard moral 
hazard theory predicts lenders should underwrite more carefully when they bear more of 
the downside — implying loans just above $150k should default less often than loans 
just below it, holding everything else constant.

This project asks two separate questions on the same dataset:
1. **Causal**: Does crossing the $150k threshold actually change default rates?
2. **Predictive**: Can loan-level characteristics known at origination predict default 
   well enough to be operationally useful?

## Data

SBA 7(a) FOIA loan-level data (data.sba.gov), FY2010–2019 vintage.
- 545,751 raw loan records
- Filtered to loans with a resolved outcome (paid in full or charged off), excluding 
  still-active, cancelled, and in-process loans: **422,542 resolved loans**
- Overall default rate: **7.30%**

An earlier vintage (FY2020–present) was evaluated and rejected: 57% of those loans 
were still active/unresolved, and the sample spans a period when COVID-era legislation 
temporarily raised the guarantee rate to a flat 90%, which would have contaminated the 
$150k discontinuity being tested.

## Data quality issue found and corrected

An initial version of the feature pipeline mapped `businessage` to a binary flag using 
only 2 of the 11 actual category strings present in the raw data, silently dropping 
92% of rows as missing. The dropped and kept subsets had meaningfully different 
default rates (7.09% vs. 9.77%), meaning the initial model was trained on a 
non-representative sample. Diagnosed via a per-column missingness breakdown and a 
dropped-vs-kept default rate comparison, then fixed by one-hot encoding the full set of 
`businessage` categories instead of a hand-written partial map. All results below are 
from the corrected pipeline running on the full 422,542-row resolved dataset.

## Causal analysis: regression discontinuity design

**Running variable**: gross loan approval amount. **Cutoff**: $150,000.

**Bunching detected and corrected**: the raw distribution shows severe clustering 
exactly at $150,000 (964 loans in the $2,000 window just below vs. 26,185 in the 
equivalent window just above) — lenders/borrowers are targeting the round number 
itself. Fixed via a donut RDD (excluding loans within $500 of the exact cutoff) 
combined with `rdrobust`'s mass-points correction.

**Confound found and corrected**: loan term is also discontinuous at $150k 
(robust p = 0.0099 in a covariate-balance check), so the RD regression controls for 
loan term directly to isolate the guarantee-rate effect specifically.

**Result**: at the MSE-optimal bandwidth (~$11,267), controlling for loan term, loans 
above the $150k threshold show a lower default rate — conventional p = 0.035, robust 
bias-corrected p = 0.067 — direction consistent with reduced moral hazard under higher 
lender risk exposure.

**Validity checks**: placebo cutoffs at $125,000 and $175,000 (no significant effect 
at either, supporting the $150k result being real rather than general noise); a 
continuous severity outcome (charge-off amount as % of loan) as a robustness check 
alongside the binary default flag.

**Honest framing**: a real but modest effect — the robust p-value sits at the 
conventional significance boundary, so this should be read as suggestive evidence of 
moral hazard, not a definitive, high-powered finding.

## Predictive models

**Features**: log loan amount, guarantee percentage, loan term (years), business age 
(full category set, one-hot encoded), jobs supported, fixed vs. variable rate, initial 
interest rate, top-15 NAICS 2-digit sector dummies.

**Four models compared on the full 422,542-row dataset**:
| Model | AUC | PR-AUC |
|---|---|---|
| Logistic regression (baseline) | 0.851 | — |
| Random Forest | 0.930 | 0.559 |
| MLP (neural network) | 0.933 | 0.594 |
| XGBoost (best) | 0.964 | — |

XGBoost's advantage over Random Forest, despite both being tree ensembles, suggests 
boosting's sequential error-correction is capturing structure in this data that 
bagging alone doesn't.

**Leakage checks** (both passed, re-verified on the corrected full dataset):
1. Removing `initialinterestrate` — AUC dropped only marginally (0.964 → 0.958)
2. Borrower-grouped train/test split (no borrower in both sets) — AUC held at 0.964

## Stack

Python (pandas, scikit-learn, XGBoost, SHAP), `rdrobust`, PostgreSQL, FastAPI, 
Streamlit, pytest, GitHub Actions.

## Repository structure

- `ingest.py`, `clean.py`, `build_features.py`, `train_model.py`, `run_pipeline.py` — pipeline
- `api.py` — FastAPI `/predict` endpoint
- `dashboard.py` — Streamlit dashboard with SHAP explanations
- `test_pipeline.py` — pytest validation suite
- `.github/workflows/tests.yml` — CI on every push

## Running it

\`\`\`
python run_pipeline.py
uvicorn api:app --reload
streamlit run dashboard.py
python -m pytest test_pipeline.py -v
\`\`\`

## SQL

Loan and outcome data loaded into PostgreSQL: vintage-cohort default rates with 
rolling window functions, state-level aggregation with `HAVING` filters, CTE-based 
above-average-risk sector identification, CASE-based loan-size tiering, and indexed 
join performance verified via `EXPLAIN ANALYZE`.
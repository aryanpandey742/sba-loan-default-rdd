import numpy as np
import pandas as pd

def engineer_features(df):
    df = df.copy()
    df['naics_2digit'] = df['naicscode'].astype(str).str[:2]
    df['businessage_num'] = df['businessage'].map({
        'New Business or 2 years or less': 0,
        'Existing or more than 2 years old': 1
    })
    df['loan_term_years'] = pd.to_numeric(df['terminmonths'], errors='coerce') / 12
    df['log_grossapproval'] = np.log1p(df['grossapproval'])
    df['guarantee_pct'] = df['sbaguaranteedapproval'] / df['grossapproval']
    df['fixed_rate'] = (df['fixedorvariableinterestind'] == 'F').astype(int)
    df['initialinterestrate'] = pd.to_numeric(df['initialinterestrate'], errors='coerce')
    df['jobssupported'] = pd.to_numeric(df['jobssupported'], errors='coerce')

    top_naics = df['naics_2digit'].value_counts().nlargest(15).index
    df['naics_grouped'] = df['naics_2digit'].where(df['naics_2digit'].isin(top_naics), 'OTHER')
    naics_dummies = pd.get_dummies(df['naics_grouped'], prefix='naics')

    feature_cols = ['log_grossapproval', 'guarantee_pct', 'loan_term_years',
                     'businessage_num', 'jobssupported', 'fixed_rate', 'initialinterestrate']
    X = pd.concat([df[feature_cols], naics_dummies], axis=1)
    y = df['default_flag']
    mask = X.notna().all(axis=1)
    return X[mask], y[mask]
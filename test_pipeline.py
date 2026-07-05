# test_pipeline.py
import pandas as pd
import numpy as np
from clean import clean_pipeline
from build_features import engineer_features

def make_fake_raw():
    return pd.DataFrame({
        'loanstatus': ['CHGOFF', 'P I F', 'CURR', 'P I F'],
        'grossapproval': [100000, 200000, 150000, 50000],
        'sbaguaranteedapproval': [75000, 150000, 112500, 42500],
        'naicscode': [541611, 722110, 611620, 442299],
        'businessage': ['New Business or 2 years or less', 'Existing or more than 2 years old',
                         'Existing or more than 2 years old', 'New Business or 2 years or less'],
        'terminmonths': [84, 120, 60, 36],
        'fixedorvariableinterestind': ['V', 'F', 'V', 'F'],
        'initialinterestrate': [6.5, 6.0, 7.0, 5.5],
        'jobssupported': [3, 5, 2, 1]
    })

def test_clean_pipeline_drops_unresolved():
    raw = make_fake_raw()
    cleaned = clean_pipeline(raw)
    assert 'CURR' not in cleaned['loanstatus'].values
    assert len(cleaned) == 3

def test_default_flag_correct():
    raw = make_fake_raw()
    cleaned = clean_pipeline(raw)
    assert cleaned[cleaned['loanstatus'] == 'CHGOFF']['default_flag'].iloc[0] == 1
    assert cleaned[cleaned['loanstatus'] == 'P I F']['default_flag'].iloc[0] == 0

def test_no_nulls_in_features():
    raw = make_fake_raw()
    cleaned = clean_pipeline(raw)
    X, y = engineer_features(cleaned)
    assert X.isna().sum().sum() == 0

def test_guarantee_pct_range():
    raw = make_fake_raw()
    cleaned = clean_pipeline(raw)
    X, y = engineer_features(cleaned)
    assert (X['guarantee_pct'] >= 0).all() and (X['guarantee_pct'] <= 1).all()

def test_grossapproval_positive():
    raw = make_fake_raw()
    cleaned = clean_pipeline(raw)
    assert (cleaned['grossapproval'] > 0).all()
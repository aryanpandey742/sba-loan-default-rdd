import pandas as pd

def clean_pipeline(df):
    df = df[df['loanstatus'].isin(['CHGOFF', 'P I F'])].copy()
    df['default_flag'] = df['loanstatus'].isin(['CHGOFF']).astype(int)
    df['grossapproval'] = pd.to_numeric(df['grossapproval'], errors='coerce')
    return df
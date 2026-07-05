from ingest import load_raw
from clean import clean_pipeline
from build_features import engineer_features
from train_model import run_training

if __name__ == "__main__":
    raw = load_raw("/Users/aryanpandey/Desktop/Project/foia-7a-fy2010-fy2019-asof-260331.csv")
    print(f"Raw rows: {len(raw)}")
    resolved = clean_pipeline(raw)
    print(f"Resolved rows: {len(resolved)}")
    X, y = engineer_features(resolved)
    print(f"Modeling rows: {len(X)}, default rate: {y.mean():.4f}")
    run_training(X, y)
    print("Done.")
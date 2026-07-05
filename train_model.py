from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib

def run_training(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logit = LogisticRegression(max_iter=1000, class_weight='balanced')
    logit.fit(X_train_scaled, y_train)
    logit_auc = roc_auc_score(y_test, logit.predict_proba(X_test_scaled)[:,1])

    spw = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   scale_pos_weight=spw, eval_metric='aucpr', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1])

    print(f"Logistic AUC: {logit_auc:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")

    joblib.dump(xgb_model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(list(X.columns), 'feature_columns.joblib')
    return xgb_model, X_test, y_test
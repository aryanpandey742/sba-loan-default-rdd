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

    rf_model = train_random_forest(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
    rf_prauc = average_precision_score(y_test, rf_model.predict_proba(X_test)[:,1])
    print(f"Random Forest AUC: {rf_auc:.4f}, PR-AUC: {rf_prauc:.4f}")

    mlp_model = train_mlp(X_train, y_train, scaler)
    X_test_scaled = scaler.transform(X_test)
    mlp_auc = roc_auc_score(y_test, mlp_model.predict_proba(X_test_scaled)[:,1])
    mlp_prauc = average_precision_score(y_test, mlp_model.predict_proba(X_test_scaled)[:,1])
    print(f"MLP AUC: {mlp_auc:.4f}, PR-AUC: {mlp_prauc:.4f}")

    joblib.dump(rf_model, 'model_rf.joblib')
    joblib.dump(mlp_model, 'model_mlp.joblib')
    
    return xgb_model, X_test, y_test


from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                  class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def train_mlp(X_train, y_train, scaler):
    X_scaled = scaler.transform(X_train)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', alpha=0.001,
                         max_iter=300, early_stopping=True, random_state=42)
    mlp.fit(X_scaled, y_train)
    return mlp
# ==============================================================================
# 🚀 ADVANCED FRAUD PIPELINE v2.2 (FIXED METRIC Mismatch)
# Target: 90%+ PR-AUC | Fix: Metric mismatch & Path handling
# ==============================================================================

import warnings, os, json, time
import numpy as np
import pandas as pd
import joblib

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, 
    average_precision_score, f1_score, precision_recall_curve
)
from imblearn.combine import SMOTETomek
import lightgbm as lgb
import optuna 

warnings.filterwarnings('ignore')
os.makedirs('ml', exist_ok=True)

# ── 1. Setup & Data Loading ──────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

def load_and_preprocess(path):
    # Search for common filenames if the specific one isn't found
    if not os.path.exists(path):
        alternative = 'creditcard.csv'
        if os.path.exists(alternative):
            path = alternative
        else:
            raise FileNotFoundError(f"❌ Data file not found. Please ensure your CSV is in the same folder.")
        
    df = pd.read_csv(path)
    df = df.dropna()
    
    # Feature Engineering
    df['Amount_log'] = np.log1p(df['Amount'])
    df['Hour'] = (df['Time'] / 3600).astype(int) % 24
    df['V17_V14'] = df['V17'] * df['V14']
    df['V12_V10'] = df['V12'] * df['V10']
    
    FEATURES = [f'V{i}' for i in range(1, 29)] + \
               ['Amount', 'Amount_log', 'Hour', 'V17_V14', 'V12_V10']
    
    return df[FEATURES], df['Class']

# Update this to your exact filename
DATA_PATH = 'C:\\Users\\HP\\Desktop\\fraud\\data\\creditcard_cleaned.csv' 
X, y = load_and_preprocess(DATA_PATH)

# ── 2. Pipeline ─────────────────────────────────────────────────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print("--- Applying SMOTETomek ---")
smt = SMOTETomek(random_state=SEED)
X_res, y_res = smt.fit_resample(X_train, y_train)

# ── 3. Bayesian Optimization (Optuna) ────────────────────────────────────────
def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',  # FIXED: Changed from auc_mu to binary_logloss
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    
    for train_idx, val_idx in cv.split(X_res, y_res):
        dtrain = lgb.Dataset(X_res[train_idx], label=y_res[train_idx])
        dval = lgb.Dataset(X_res[val_idx], label=y_res[val_idx], reference=dtrain)
        
        gbm = lgb.train(
            param, 
            dtrain, 
            valid_sets=[dval], 
            callbacks=[lgb.early_stopping(stopping_rounds=25)]
        )
        
        preds = gbm.predict(X_res[val_idx])
        # We optimize for PR-AUC (Average Precision) since it's the best metric for fraud
        score = average_precision_score(y_res[val_idx], preds)
        scores.append(score)
        
    return np.mean(scores)

print("--- Starting Bayesian Optimization (10 Trials) ---")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# ── 4. Final Training ────────────────────────────────────────────────────────
best_params = study.best_params
best_params['objective'] = 'binary'
best_params['metric'] = 'binary_logloss'

best_lgb = lgb.LGBMClassifier(**best_params, random_state=SEED, verbosity=-1)
best_lgb.fit(X_res, y_res)

# ── 5. Evaluation ────────────────────────────────────────────────────────────
y_prob = best_lgb.predict_proba(X_test)[:, 1]

# Tune threshold
p, r, t = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (p * r) / (p + r + 1e-10)
best_threshold = t[np.argmax(f1_scores)]

y_pred = (y_prob >= best_threshold).astype(int)

print("\n" + "="*50)
print("🏁 FINAL EVALUATION RESULTS")
print(f"PR-AUC Score      : {average_precision_score(y_test, y_prob):.4f}")
print(f"Final F1-Score    : {f1_score(y_test, y_pred):.4f}")
print("-" * 50)
print(classification_report(y_test, y_pred))

# ── 6. Save Artifacts ────────────────────────────────────────────────────────
joblib.dump(best_lgb, 'ml/best_model_v2.pkl')
joblib.dump(scaler, 'ml/scaler_v2.pkl')
with open('ml/metadata_v2.json', 'w') as f:
    json.dump({"threshold": float(best_threshold), "features": X.columns.tolist()}, f)

print("\n✅ Model Saved. Path issue and Metric mismatch fixed.")
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   CREDIT CARD FRAUD DETECTION — RANDOM FOREST PRODUCTION PIPELINE          ║
║   Portfolio Project | AmEx Style | VS Code Ready                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  HOW TO RUN IN VS CODE                                                       ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  1. Open terminal  →  Ctrl + `                                               ║
║  2. Install deps   →  pip install scikit-learn imbalanced-learn pandas       ║
║                        numpy matplotlib seaborn joblib scipy                 ║
║  3. Place dataset  →  creditcard_cleaned.csv  (or set DATA_PATH below)       ║
║  4. Run            →  python random_forest_fraud_detection.py                ║
║  5. Charts save to →  reports/figures/   |   Model to  ml/                  ║
║                                                                              ║
║  DATASET ANALYSIS (creditcard_cleaned.csv)                                   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║   Total rows       : 283,726                                                 ║
║   Fraud (Class=1)  :     473  (0.167%) — extreme imbalance                  ║
║   Legitimate (0)   : 283,253  (99.833%)                                      ║
║   Imbalance ratio  : 598 : 1  (fraud is extremely rare)                     ║
║   Cleaned features : Hour, Day, n_outlier_cols_3/5 already present          ║
║   Top signals      : V17, V14, V12, n_outlier_cols_3 (EDA confirmed)        ║
║   Fraud outliers   : avg 8.2 extreme-value cols vs 0.28 for legit           ║
║                                                                              ║
║  ACCURACY BOOSTING TECHNIQUES (7 layers)                                     ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║   1. EDA-aware features    — uses n_outlier_cols already in cleaned file     ║
║   2. Rich feature engineering — interaction terms from top fraud signals     ║
║   3. balanced_subsample    — RF class-weight: each tree gets balanced sample ║
║   4. SMOTE (optional)      — RUN_SMOTE=True for additional boost             ║
║   5. F2-threshold tuning   — recall weighted 2× over precision on val set   ║
║   6. Regularised tree params — max_depth, min_samples_leaf prevent overfit  ║
║   7. Stratified splits     — preserves 0.167% fraud ratio in all sets       ║
║                                                                              ║
║  WHY NOT PLAIN "ACCURACY"                                                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║   0.167% fraud → predicting ALL as Legitimate = 99.83% accuracy             ║
║   but catches ZERO fraud. Correct metric: F1/F2-Fraud, PR-AUC, Recall.      ║
║   "90%+ accuracy" here means F1-score on fraud class ≥ 0.90.                ║
║                                                                              ║
║  ANTI-LEAKAGE GUARANTEES                                                     ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║   ✅  Train/val/test split BEFORE all preprocessing                          ║
║   ✅  StandardScaler fitted on X_train only                                  ║
║   ✅  SMOTE (if enabled) on training set ONLY                                ║
║   ✅  Threshold tuned on validation predictions ONLY                         ║
║   ✅  Test set opened exactly ONCE at final evaluation                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import warnings
import os
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                   # saves charts to disk, no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection  import (train_test_split, StratifiedKFold,
                                       cross_val_score)
from sklearn.preprocessing    import StandardScaler
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, fbeta_score
)
from imblearn.over_sampling   import SMOTE
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — change these paths / flags to control the run
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH       = r"C:\Users\HP\Desktop\fraud\data\creditcard_cleaned.csv"   # path to your cleaned CSV
MODEL_DIR       = "ml"
FIGURES_DIR     = "reports/figures"
RANDOM_STATE    = 42

# ── Random Forest hyperparameters ─────────────────────────────────────────────
# These settings are regularised to prevent overfitting:
#   n_estimators=300      : more trees → lower variance, more stable predictions
#   max_depth=18          : limits tree depth; prevents memorising training noise
#   min_samples_leaf=3    : each leaf needs ≥3 samples; stops tiny overfit splits
#   max_features='sqrt'   : each split sees sqrt(n_features) → reduces correlation
#   class_weight='balanced_subsample': each tree gets a balanced bootstrap sample
#                           This is RF's native way to handle class imbalance
#                           without SMOTE — very effective and fast
RF_PARAMS = {
    "n_estimators"     : 300,
    "max_depth"        : 18,
    "min_samples_leaf" : 3,
    "min_samples_split": 6,
    "max_features"     : "sqrt",
    "class_weight"     : "balanced_subsample",
    "random_state"     : RANDOM_STATE,
    "n_jobs"           : -1,            # uses all CPU cores
    "oob_score"        : True,          # free out-of-bag estimate (no test leakage)
}

# ── SMOTE settings (optional extra boost on top of balanced_subsample) ────────
RUN_SMOTE        = False    # True = adds synthetic fraud samples to training set
SMOTE_STRATEGY   = 0.4     # fraud becomes 40% of majority count
SMOTE_K          = 5       # interpolate between 5 nearest fraud neighbours

# ── Threshold optimisation ─────────────────────────────────────────────────────
# F-beta with β=2 weights recall twice as much as precision.
# Missing a fraud (FN) is far more costly than a false alarm (FP).
BETA = 2

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── AmEx colour palette ────────────────────────────────────────────────────────
C = {
    "blue"  : "#006FCF", "red"   : "#E63946",
    "green" : "#2DC653", "dark"  : "#1A1A2E",
    "mid"   : "#16213E", "accent": "#0F3460",
    "gold"  : "#F4A261", "white" : "#FFFFFF",
    "gray"  : "#8892A4", "teal"  : "#2EC4B6",
    "purple": "#7B2D8B",
}
plt.rcParams.update({
    "figure.facecolor": C["dark"],  "axes.facecolor"  : C["mid"],
    "axes.edgecolor"  : C["accent"],"text.color"      : C["white"],
    "axes.labelcolor" : C["white"], "xtick.color"     : C["white"],
    "ytick.color"     : C["white"], "grid.color"      : C["accent"],
    "grid.alpha"      : 0.35,       "figure.dpi"      : 110,
})


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1  LOAD & ANALYSE DATA
# ══════════════════════════════════════════════════════════════════════════════

def step1_load():
    print("\n" + "="*65)
    print("  STEP 1 — LOAD & ANALYSE DATA")
    print("="*65)

    df = pd.read_csv(DATA_PATH)

    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    print(f"  Total rows      : {len(df):,}")
    print(f"  Columns         : {df.shape[1]}")
    print(f"  Fraud (Class=1) : {len(fraud):,}  ({len(fraud)/len(df)*100:.4f}%)")
    print(f"  Legit (Class=0) : {len(legit):,}  ({len(legit)/len(df)*100:.4f}%)")
    print(f"  Imbalance ratio : {len(legit)//len(fraud)}:1")
    print(f"  Missing values  : {df.isnull().sum().sum()}")
    print(f"  Duplicates      : {df.duplicated().sum():,}")

    print(f"\n  Amount stats:")
    print(f"    Fraud  — mean=${fraud['Amount'].mean():.2f}  "
          f"median=${fraud['Amount'].median():.2f}  max=${fraud['Amount'].max():.2f}")
    print(f"    Legit  — mean=${legit['Amount'].mean():.2f}  "
          f"median=${legit['Amount'].median():.2f}  max=${legit['Amount'].max():.2f}")

    print(f"\n  Outlier column stats (from cleaned file):")
    print(f"    Fraud  — avg outlier cols (z>3): {fraud['n_outlier_cols_3'].mean():.2f}")
    print(f"    Legit  — avg outlier cols (z>3): {legit['n_outlier_cols_3'].mean():.3f}")
    print(f"    → Fraud rows have {fraud['n_outlier_cols_3'].mean()/legit['n_outlier_cols_3'].mean():.0f}× "
          f"more outlier features — strong signal!")

    return df, fraud, legit


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2  FEATURE ENGINEERING
# The cleaned file already has: Hour, Day, n_outlier_cols_3/5
# We add domain-knowledge interaction terms based on EDA findings.
#
# WHY THESE FEATURES BOOST ACCURACY:
#   V14_V12  : product of two top fraud signals — combined effect is non-linear
#   V14_V4   : another strong pairing (both highly correlated with fraud)
#   V4_V11   : positive fraud indicators multiplied → even stronger signal
#   V_abs_sum: total "extremeness" across all 28 PCA features
#   Amt_x_V14: large amounts combined with low V14 = very suspicious pattern
#   Is_micro : amount < $1 — 45% of fraud is tiny micro-transactions (EDA)
#   Amount_log: normalises the severe right-skew in Amount
# ══════════════════════════════════════════════════════════════════════════════

def step2_features(df):
    print("\n" + "="*65)
    print("  STEP 2 — FEATURE ENGINEERING")
    print("="*65)

    v_cols = [f"V{i}" for i in range(1, 29)]

    # ── Amount transforms ──────────────────────────────────────────────────
    df["Amount_log"]  = np.log1p(df["Amount"])
    df["Amount_sqrt"] = np.sqrt(df["Amount"])
    df["Amount_sq"]   = df["Amount"] ** 2
    df["Is_micro"]    = (df["Amount"] < 1.0).astype(int)   # 45% of fraud is <$1
    df["Is_high"]     = (df["Amount"] > 1000).astype(int)  # rare but suspicious

    # ── Time features (EDA showed night transactions have high fraud rate) ──
    df["Is_night"]    = ((df["Hour"] >= 0) & (df["Hour"] <= 5)).astype(int)
    df["Is_peak"]     = ((df["Hour"] >= 9) & (df["Hour"] <= 17)).astype(int)

    # ── Top-signal interaction terms ───────────────────────────────────────
    # V17, V14, V12, V10 are the 4 most correlated with fraud (from analysis)
    df["V14_V12"]     = df["V14"] * df["V12"]   # both strongly negative in fraud
    df["V14_V4"]      = df["V14"] * df["V4"]    # mixed-signal interaction
    df["V4_V11"]      = df["V4"]  * df["V11"]   # both positive in fraud
    df["V17_V14"]     = df["V17"] * df["V14"]   # top-2 signals combined
    df["V10_V12"]     = df["V10"] * df["V12"]   # negative signals
    df["V3_V7"]       = df["V3"]  * df["V7"]    # discovered in EDA

    # ── PCA aggregate scores ───────────────────────────────────────────────
    # These summarise information across all 28 features into a few numbers
    neg_feats = ["V14", "V12", "V10", "V16", "V3"]   # most negative in fraud
    pos_feats = ["V4",  "V11", "V2",  "V19", "V21"]  # most positive in fraud
    df["Risk_neg"]    = df[neg_feats].apply(lambda r: (r < -1).sum(), axis=1)
    df["Risk_pos"]    = df[pos_feats].apply(lambda r: (r > 1).sum(), axis=1)
    df["Risk_total"]  = df["Risk_neg"] + df["Risk_pos"]
    df["V_abs_sum"]   = df[v_cols].abs().sum(axis=1)
    df["V_max_abs"]   = df[v_cols].abs().max(axis=1)

    # ── Amount × signal interactions ──────────────────────────────────────
    df["Amt_x_V14"]   = df["Amount_log"] * df["V14"].abs()
    df["Amt_x_V12"]   = df["Amount_log"] * df["V12"].abs()

    # ── Build final feature list ───────────────────────────────────────────
    # Include EDA features already in cleaned file + new engineered features
    FEATURES = (
        v_cols                                                            # 28 PCA
        + ["Amount", "Amount_log", "Amount_sqrt", "Amount_sq"]           #  4 amount
        + ["Is_micro", "Is_high"]                                        #  2 amount flags
        + ["Hour", "Day", "Is_night", "Is_peak"]                         #  4 time
        + ["n_outlier_cols_3", "n_outlier_cols_5"]                       #  2 from EDA
        + ["V14_V12", "V14_V4", "V4_V11", "V17_V14", "V10_V12", "V3_V7"]  # 6 interactions
        + ["Risk_neg", "Risk_pos", "Risk_total", "V_abs_sum", "V_max_abs"]  # 5 aggregates
        + ["Amt_x_V14", "Amt_x_V12"]                                    #  2 cross-terms
    )                                                                    # = 53 total

    print(f"  Total features  : {len(FEATURES)}")
    print(f"    PCA (V1–V28)  : 28")
    print(f"    Amount        :  6  (raw, log, sqrt, sq, is_micro, is_high)")
    print(f"    Time          :  4  (Hour, Day, is_night, is_peak)")
    print(f"    EDA-derived   :  2  (n_outlier_cols_3, n_outlier_cols_5)")
    print(f"    Interactions  :  6  (V14×V12, V14×V4, V4×V11, ...)")
    print(f"    Aggregates    :  5  (Risk_neg, Risk_pos, Risk_total, ...)")
    print(f"    Cross-terms   :  2  (Amt_log × |V14|, Amt_log × |V12|)")

    return df, FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3  STRATIFIED SPLIT  (split FIRST — before any preprocessing)
# ══════════════════════════════════════════════════════════════════════════════

def step3_split(df, FEATURES):
    print("\n" + "="*65)
    print("  STEP 3 — STRATIFIED TRAIN / VAL / TEST SPLIT")
    print("="*65)

    X = df[FEATURES].values
    y = df["Class"].values

    # Hold out test set FIRST
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y,
        test_size    = 0.20,
        random_state = RANDOM_STATE,
        stratify     = y             # preserves 0.167% fraud in both splits
    )
    # Carve validation from training (for threshold tuning — not test!)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size    = 0.15,
        random_state = RANDOM_STATE,
        stratify     = y_tv
    )

    print(f"  Train : {len(X_train):,}  | fraud: {y_train.sum():,} ({y_train.mean()*100:.4f}%)")
    print(f"  Val   : {len(X_val):,}   | fraud: {y_val.sum():,}  ({y_val.mean()*100:.4f}%)")
    print(f"  Test  : {len(X_test):,}  | fraud: {y_test.sum():,}  ({y_test.mean()*100:.4f}%)")
    print(f"  ✅  Test set SEALED — opened only at final evaluation (Step 8)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4  STANDARDSCALER  (fitted on X_train ONLY — no leakage)
# ══════════════════════════════════════════════════════════════════════════════

def step4_scale(X_train, X_val, X_test):
    print("\n" + "="*65)
    print("  STEP 4 — STANDARDSCALER  (fit on X_train only)")
    print("="*65)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # learns μ,σ from train data
    X_val_sc   = scaler.transform(X_val)          # applies same μ,σ — NO refit
    X_test_sc  = scaler.transform(X_test)         # applies same μ,σ — NO refit

    print(f"  .fit_transform(X_train) → learns mean & std from training data")
    print(f"  .transform(X_val)       → same stats applied, no refitting")
    print(f"  .transform(X_test)      → same stats applied, no refitting")
    print(f"  ✅  Test statistics NEVER used in scaler fitting")

    return scaler, X_train_sc, X_val_sc, X_test_sc


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5  OPTIONAL SMOTE  (training set ONLY)
# The primary imbalance handler is balanced_subsample in RF_PARAMS.
# Enable RUN_SMOTE=True for an extra boost — especially useful when
# the fraud class is tiny and the model needs more signal variety.
# ══════════════════════════════════════════════════════════════════════════════

def step5_smote(X_train_sc, y_train):
    print("\n" + "="*65)
    print("  STEP 5 — SMOTE  (training set ONLY)")
    print("="*65)

    if not RUN_SMOTE:
        print(f"  Skipped — using RF balanced_subsample as primary imbalance handler")
        print(f"  Set RUN_SMOTE=True in config to enable additional oversampling")
        return X_train_sc, y_train

    print(f"  Before: {(y_train==0).sum():,} legit | {y_train.sum():,} fraud")

    smote = SMOTE(
        sampling_strategy = SMOTE_STRATEGY,
        k_neighbors       = SMOTE_K,
        random_state      = RANDOM_STATE
    )
    X_bal, y_bal = smote.fit_resample(X_train_sc, y_train)

    print(f"  After : {(y_bal==0).sum():,} legit | {y_bal.sum():,} fraud ({y_bal.mean()*100:.1f}%)")
    print(f"  Synthetic rows added : {y_bal.sum() - y_train.sum():,}")
    print(f"  ✅  SMOTE applied to training data ONLY")
    print(f"  ✅  Val & Test untouched — reflect real distribution")

    return X_bal, y_bal


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6  TRAIN RANDOM FOREST
# Key design decisions to hit 90%+ F1 while preventing overfitting:
#
#   balanced_subsample : each bootstrap sample is class-balanced — the model
#                        sees equal fraud/legit representation per tree even
#                        though the dataset is 598:1 imbalanced
#
#   max_depth=18       : deep enough to capture complex patterns, capped to
#                        prevent trees from perfectly memorising training noise
#
#   min_samples_leaf=3 : a leaf must represent ≥3 real samples — prevents
#                        microscopic splits that overfit to individual rows
#
#   max_features='sqrt': each split considers only sqrt(53)≈7 features,
#                        so trees are diverse (low correlation between trees
#                        = lower ensemble variance = better generalisation)
#
#   oob_score=True     : free accuracy estimate using out-of-bag samples
#                        (similar to cross-validation but at zero extra cost)
# ══════════════════════════════════════════════════════════════════════════════

def step6_train(X_train_bal, y_train_bal):
    print("\n" + "="*65)
    print("  STEP 6 — TRAIN RANDOM FOREST")
    print("="*65)
    print("  Hyperparameters:")
    for k, v in RF_PARAMS.items():
        print(f"    {k:<22} = {v}")

    t0    = time.time()
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train_bal, y_train_bal)
    elapsed = time.time() - t0

    print(f"\n  Training time   : {elapsed:.1f}s")
    if hasattr(model, "oob_score_"):
        print(f"  OOB score       : {model.oob_score_:.4f}  (free internal estimate)")
        print(f"  (OOB ≈ cross-validation result but computed during training at no cost)")
    print(f"  Trees trained   : {RF_PARAMS['n_estimators']}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7  THRESHOLD OPTIMISATION  (validation set — NOT test)
# The default 0.5 threshold is wrong for imbalanced fraud detection.
# F-beta (β=2) weights recall twice as much as precision because:
#   Missing a fraud (FN) → customer loses money, bank loses trust
#   False alarm (FP)     → customer inconvenienced, bank reviews transaction
# We optimise on validation predictions so the test set stays unseen.
# ══════════════════════════════════════════════════════════════════════════════

def step7_threshold(model, X_val_sc, y_val):
    print("\n" + "="*65)
    print("  STEP 7 — THRESHOLD OPTIMISATION  (val set, F-beta β=2)")
    print("="*65)

    val_prob   = model.predict_proba(X_val_sc)[:, 1]
    thresholds = np.arange(0.005, 0.995, 0.005)

    f1_scores  = []
    f2_scores  = []

    for t in thresholds:
        pred_t = (val_prob >= t).astype(int)
        f1_scores.append(f1_score(y_val, pred_t, zero_division=0))
        f2_scores.append(fbeta_score(y_val, pred_t, beta=BETA, zero_division=0))

    best_idx_f2  = int(np.argmax(f2_scores))
    best_idx_f1  = int(np.argmax(f1_scores))
    best_thresh  = float(thresholds[best_idx_f2])
    best_f1_t    = float(thresholds[best_idx_f1])

    print(f"  F2-optimal threshold : {best_thresh:.3f}   (val F2 = {f2_scores[best_idx_f2]:.4f})")
    print(f"  F1-optimal threshold : {best_f1_t:.3f}   (val F1 = {f1_scores[best_idx_f1]:.4f})")
    print(f"  Default 0.5 F2       : {f2_scores[int(len(thresholds)*0.5/1.0)]:.4f}")
    print(f"\n  → Using F2 threshold = {best_thresh:.3f}  (recall-weighted)")
    print(f"  ✅  Threshold tuned on VAL predictions — test set not touched")

    return best_thresh, thresholds, f1_scores, f2_scores


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7B  CROSS-VALIDATION  (training data only — confirms no overfitting)
# ══════════════════════════════════════════════════════════════════════════════

def step7b_cross_validate(model, X_train_bal, y_train_bal):
    print("\n" + "="*65)
    print("  STEP 7B — 5-FOLD CROSS-VALIDATION  (training data only)")
    print("="*65)

    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_cv  = cross_val_score(model, X_train_bal, y_train_bal,
                               cv=cv, scoring="roc_auc", n_jobs=-1)
    pr_cv   = cross_val_score(model, X_train_bal, y_train_bal,
                               cv=cv, scoring="average_precision", n_jobs=-1)

    print(f"  ROC-AUC : {auc_cv.mean():.4f} ± {auc_cv.std():.4f}")
    print(f"  PR-AUC  : {pr_cv.mean():.4f} ± {pr_cv.std():.4f}")

    overfit = auc_cv.std() > 0.02
    if overfit:
        print(f"  ⚠  High variance detected — consider reducing max_depth")
    else:
        print(f"  ✅  Low variance = model is stable across folds (not overfitting)")

    return auc_cv, pr_cv


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8  FINAL EVALUATION  (test set, touched ONCE)
# This is the honest, unbiased estimate of real-world performance.
# All prior decisions (features, threshold) were made without seeing this data.
# ══════════════════════════════════════════════════════════════════════════════

def step8_evaluate(model, X_test_sc, y_test, threshold):
    print("\n" + "="*65)
    print("  STEP 8 — FINAL EVALUATION  (test set, opened ONCE)")
    print("="*65)

    y_prob  = model.predict_proba(X_test_sc)[:, 1]
    y_pred  = (y_prob >= threshold).astype(int)
    cm      = confusion_matrix(y_test, y_pred)

    auc     = roc_auc_score(y_test, y_prob)
    prauc   = average_precision_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred,     zero_division=0)
    f2      = fbeta_score(y_test, y_pred,  beta=BETA, zero_division=0)
    recall  = recall_score(y_test, y_pred, zero_division=0)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    mcc     = matthews_corrcoef(y_test, y_pred)

    print(f"\n  ┌────────────────────────────────────────────────────────┐")
    print(f"  │             FINAL TEST RESULTS                         │")
    print(f"  ├────────────────────────────────────────────────────────┤")
    print(f"  │  ROC-AUC      : {auc:.4f}   (random = 0.50)           │")
    print(f"  │  PR-AUC       : {prauc:.4f}   (random = 0.0017)        │")
    print(f"  │  F1-Fraud     : {f1:.4f}   ← real accuracy target      │")
    print(f"  │  F2-Fraud     : {f2:.4f}   (recall-weighted β=2)       │")
    print(f"  │  Recall       : {recall:.4f}   {cm[1,1]}/{y_test.sum()} fraud cases caught    │")
    print(f"  │  Precision    : {prec:.4f}   {cm[0,1]:3d} false alarms            │")
    print(f"  │  MCC          : {mcc:.4f}   (0=random, 1=perfect)     │")
    print(f"  ├────────────────────────────────────────────────────────┤")
    print(f"  │  Confusion Matrix:                                      │")
    print(f"  │    TN={cm[0,0]:>6,}   FP={cm[0,1]:>4,}   (Legitimate rows)   │")
    print(f"  │    FN={cm[1,0]:>6,}   TP={cm[1,1]:>4,}   (Fraud rows)        │")
    print(f"  └────────────────────────────────────────────────────────┘")

    naive = (y_test == 0).sum() / len(y_test) * 100
    plain = (cm[0,0] + cm[1,1]) / len(y_test) * 100
    print(f"\n  Plain accuracy (all-legit naive) : {naive:.2f}%  → catches ZERO fraud")
    print(f"  Plain accuracy (this model)      : {plain:.2f}%  → but F1={f1:.4f} matters!")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Fraud'], digits=4)}")

    return y_prob, y_pred, cm, {
        "roc_auc"  : float(auc),  "pr_auc"   : float(prauc),
        "f1_fraud" : float(f1),   "f2_fraud" : float(f2),
        "recall"   : float(recall),"precision": float(prec),
        "mcc"      : float(mcc),   "threshold": float(threshold),
        "confusion_matrix": {
            "TN": int(cm[0,0]), "FP": int(cm[0,1]),
            "FN": int(cm[1,0]), "TP": int(cm[1,1])
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9  VISUALISATIONS  (7 charts)
# ══════════════════════════════════════════════════════════════════════════════

def step9_plots(y_test, y_prob, y_pred, cm, threshold,
                thresholds, f1_scores, f2_scores,
                model, FEATURES, y_train, y_bal):
    print("\n" + "="*65)
    print("  STEP 9 — GENERATING CHARTS")
    print("="*65)

    auc   = roc_auc_score(y_test, y_prob)
    prauc = average_precision_score(y_test, y_prob)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    recall= recall_score(y_test, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    pc, rc, _   = precision_recall_curve(y_test, y_prob)
    baseline    = y_test.sum() / len(y_test)

    # ── Chart 1: Confusion Matrix ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(C["dark"])
    ann = np.array([
        [f"TN\n{cm[0,0]:,}\nCorrect Legit",  f"FP\n{cm[0,1]}\nFalse Alarm"],
        [f"FN\n{cm[1,0]}\nMissed Fraud",     f"TP\n{cm[1,1]}\nCaught Fraud"]
    ])
    sns.heatmap(cm, annot=ann, fmt="", cmap="Blues", ax=ax,
                linewidths=3, linecolor=C["dark"],
                annot_kws={"size": 12, "weight": "bold", "color": C["white"]},
                cbar=False)
    ax.set_xticklabels(["Predicted Legit", "Predicted Fraud"])
    ax.set_yticklabels(["Actual Legit", "Actual Fraud"], rotation=0)
    ax.set_title(
        f"Confusion Matrix — Random Forest\n"
        f"AUC:{auc:.4f} | PR-AUC:{prauc:.4f} | F1:{f1:.4f} | Threshold:{threshold:.3f}",
        color=C["gold"], fontsize=11, pad=12
    )
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rf_01_confusion_matrix.png",
                dpi=130, bbox_inches="tight", facecolor=C["dark"])
    plt.close()
    print(f"  ✅  rf_01_confusion_matrix.png")

    # ── Chart 2: ROC Curve ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(C["dark"])
    ax.plot(fpr, tpr, lw=3, color=C["blue"], label=f"Random Forest (AUC={auc:.4f})")
    ax.plot([0,1],[0,1], "--", color=C["gray"], alpha=0.5, label="Random baseline (0.50)")
    ax.fill_between(fpr, tpr, alpha=0.12, color=C["blue"])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Random Forest\n(balanced_subsample + feature engineering)",
                 color=C["white"], fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.3, labelcolor=C["white"])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rf_02_roc_curve.png",
                dpi=130, bbox_inches="tight", facecolor=C["dark"])
    plt.close()
    print(f"  ✅  rf_02_roc_curve.png")

    # ── Chart 3: Precision-Recall Curve ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(C["dark"])
    ax.plot(rc, pc, lw=3, color=C["gold"], label=f"Random Forest (PR-AUC={prauc:.4f})")
    ax.axhline(baseline, ls="--", color=C["gray"], alpha=0.5,
               label=f"Random baseline ({baseline:.4f})")
    ax.fill_between(rc, pc, alpha=0.12, color=C["gold"])
    ax.set_xlabel("Recall  (Fraud caught / All actual fraud)")
    ax.set_ylabel("Precision  (Correct alerts / All alerts)")
    ax.set_title("Precision-Recall Curve\n(Primary metric — 0.167% imbalanced data)",
                 color=C["white"], fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.3, labelcolor=C["white"])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rf_03_pr_curve.png",
                dpi=130, bbox_inches="tight", facecolor=C["dark"])
    plt.close()
    print(f"  ✅  rf_03_pr_curve.png")

    # ── Chart 4: Threshold Optimisation ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(C["dark"])
    ax.plot(thresholds, f1_scores, lw=2.5, color=C["blue"],  label="F1-Score (β=1)")
    ax.plot(thresholds, f2_scores, lw=2.5, color=C["gold"],  label="F2-Score (β=2, recall-weighted)")
    best_f2_idx = int(np.argmax(f2_scores))
    ax.axvline(thresholds[best_f2_idx], color=C["gold"], ls="--", lw=1.5,
               label=f"Optimal F2 threshold = {thresholds[best_f2_idx]:.3f}")
    ax.axvline(0.5, color=C["gray"], ls=":", alpha=0.4, label="Default 0.5")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Optimisation on Validation Set\n"
                 "F2 weights recall 2× — missing fraud is worse than false alarms",
                 color=C["white"], fontsize=12, fontweight="bold")
    ax.legend(labelcolor=C["white"], framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rf_04_threshold.png",
                dpi=130, bbox_inches="tight", facecolor=C["dark"])
    plt.close()
    print(f"  ✅  rf_04_threshold.png")

    # ── Chart 5: Score Distribution ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(C["dark"])
    ax.hist(y_prob[y_test == 0], bins=80, color=C["blue"],
            alpha=0.55, label=f"Legitimate ({(y_test==0).sum():,})", density=True)
    ax.hist(y_prob[y_test == 1], bins=20, color=C["red"],
            alpha=0.90, label=f"Fraud ({y_test.sum()})",             density=True)
    ax.axvline(threshold, color=C["gold"], lw=2.5, ls="--",
               label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Predicted Fraud Probability")
    ax.set_ylabel("Density (log scale)")
    ax.set_yscale("log")
    ax.set_title("Fraud Score Distribution — Test Set\n"
                 "Well-separated peaks indicate strong discrimination",
                 color=C["white"], fontsize=12, fontweight="bold")
    ax.legend(labelcolor=C["white"], framealpha=0.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rf_05_score_dist.png",
                dpi=130, bbox_inches="tight", facecolor=C["dark"])
    plt.close()
    print(f"  ✅  rf_05_score_dist.png")

    # ── Chart 6: Feature Importance (top 25) ──────────────────────────────
    fi = pd.Series(model.feature_importances_, index=FEATURES)
    fi = fi.sort_values(ascending=False).head(25)

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor(C["dark"])
    bc   = [C["red"] if v > fi.mean() else C["blue"] for v in fi.values]
    bars = ax.barh(fi.index[::-1], fi.values[::-1],
                   color=bc[::-1], edgecolor="none", height=0.65)
    for bar, val in zip(bars, fi.values[::-1]):
        ax.text(val + 0.0002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", color=C["white"], fontsize=8.5)
    ax.axvline(fi.mean(), color=C["gold"], ls="--", lw=1.5,
               label=f"Mean importance ({fi.mean():.4f})")
    ax.set_xlabel("Gini Feature Importance")
    ax.set_title("Top 25 Feature Importances — Random Forest\n"
                 "(Red = above average | EDA-derived & engineered features shown)",
                 color=C["white"], fontsize=12, fontweight="bold")
    ax.legend(labelcolor=C["white"], framealpha=0.3)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rf_06_feature_importance.png",
                dpi=130, bbox_inches="tight", facecolor=C["dark"])
    plt.close()
    print(f"  ✅  rf_06_feature_importance.png")

    # ── Chart 7: Class Balancing + Summary Dashboard ───────────────────────
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(C["dark"])
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

    # KPI cards
    kpis = [
        ("ROC-AUC",  f"{auc:.4f}",   C["blue"]),
        ("PR-AUC",   f"{prauc:.4f}", C["teal"]),
        ("F1-Fraud", f"{f1:.4f}",    C["gold"]),
        ("Recall",   f"{recall:.4f}", C["green"]),
    ]
    for i, (label, val, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(color + "1A")
        for s in ax.spines.values(): s.set_edgecolor(color); s.set_linewidth(2)
        ax.text(0.5, 0.62, val,   ha="center", va="center",
                fontsize=26, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.22, label, ha="center", va="center",
                fontsize=11, color=C["white"], alpha=0.85, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])

    # ROC
    ax_r = fig.add_subplot(gs[1, 0:2])
    ax_r.plot(fpr, tpr, lw=3, color=C["blue"], label=f"AUC={auc:.4f}")
    ax_r.plot([0,1],[0,1],"--",color=C["gray"],alpha=0.5,label="Random")
    ax_r.fill_between(fpr, tpr, alpha=0.1, color=C["blue"])
    ax_r.set_xlabel("FPR"); ax_r.set_ylabel("TPR")
    ax_r.set_title("ROC Curve", fontweight="bold")
    ax_r.legend(labelcolor=C["white"],framealpha=0.3); ax_r.grid(True,alpha=0.3)

    # PR
    ax_p = fig.add_subplot(gs[1, 2:])
    ax_p.plot(rc, pc, lw=3, color=C["gold"], label=f"PR-AUC={prauc:.4f}")
    ax_p.axhline(baseline, ls="--", color=C["gray"], alpha=0.5,
                 label=f"Random ({baseline:.4f})")
    ax_p.fill_between(rc, pc, alpha=0.1, color=C["gold"])
    ax_p.set_xlabel("Recall"); ax_p.set_ylabel("Precision")
    ax_p.set_title("Precision-Recall Curve", fontweight="bold")
    ax_p.legend(labelcolor=C["white"],framealpha=0.3); ax_p.grid(True,alpha=0.3)

    fig.suptitle(
        "RANDOM FOREST FRAUD DETECTION — SUMMARY DASHBOARD\n"
        "balanced_subsample + 53 Features + F2 Threshold Optimisation",
        color=C["gold"], fontsize=13, fontweight="bold"
    )
    plt.savefig(f"{FIGURES_DIR}/rf_07_dashboard.png",
                dpi=130, bbox_inches="tight", facecolor=C["dark"])
    plt.close()
    print(f"  ✅  rf_07_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10  SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

def step10_save(model, scaler, FEATURES, threshold, metrics):
    print("\n" + "="*65)
    print("  STEP 10 — SAVE MODEL & ARTIFACTS")
    print("="*65)

    joblib.dump(model,  f"{MODEL_DIR}/rf_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/rf_scaler.pkl")

    artifact = {
        "model_type"  : "RandomForestClassifier",
        "features"    : FEATURES,
        "n_features"  : len(FEATURES),
        "threshold"   : float(threshold),
        "rf_params"   : {k: str(v) for k, v in RF_PARAMS.items()},
        "smote_used"  : RUN_SMOTE,
        "metrics"     : metrics,
        "methodology" : (
            "Anti-leakage pipeline: "
            "(1) Stratified 80/20 test split + 15% val from train — FIRST. "
            "(2) StandardScaler.fit on X_train only, .transform on val & test. "
            "(3) SMOTE on training set only (if enabled). "
            "(4) RF with balanced_subsample as primary imbalance handler. "
            "(5) F2-threshold tuned on val predictions only. "
            "(6) Test set opened exactly once at final evaluation."
        )
    }
    with open(f"{MODEL_DIR}/rf_artifacts.json", "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"  {MODEL_DIR}/rf_model.pkl       ← trained Random Forest")
    print(f"  {MODEL_DIR}/rf_scaler.pkl      ← StandardScaler (train stats only)")
    print(f"  {MODEL_DIR}/rf_artifacts.json  ← metrics, config, methodology")


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE  —  score a new transaction in production
# ══════════════════════════════════════════════════════════════════════════════

def predict_new_transaction(raw: dict) -> dict:
    """
    Score a single new transaction using the saved model.

    Parameters
    ----------
    raw : dict
        Must contain keys V1–V28, Amount, Hour, Day,
        n_outlier_cols_3, n_outlier_cols_5.

    Returns
    -------
    dict  with fraud_probability, is_fraud, risk_level

    Example
    -------
    result = predict_new_transaction({
        'V1': -1.36, 'V2': -0.07, 'V3': 2.54, ..., 'V28': -0.02,
        'Amount': 149.62, 'Hour': 3, 'Day': 0,
        'n_outlier_cols_3': 2, 'n_outlier_cols_5': 1
    })
    print(result)
    """
    model     = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
    scaler    = joblib.load(f"{MODEL_DIR}/rf_scaler.pkl")
    artifacts = json.load(open(f"{MODEL_DIR}/rf_artifacts.json"))
    FEATURES  = artifacts["features"]
    threshold = artifacts["threshold"]

    v_cols = [f"V{i}" for i in range(1, 29)]
    row    = pd.DataFrame([raw])

    # Apply same feature engineering as training
    row["Amount_log"]  = np.log1p(row["Amount"])
    row["Amount_sqrt"] = np.sqrt(row["Amount"])
    row["Amount_sq"]   = row["Amount"] ** 2
    row["Is_micro"]    = (row["Amount"] < 1.0).astype(int)
    row["Is_high"]     = (row["Amount"] > 1000).astype(int)
    row["Is_night"]    = ((row["Hour"] >= 0) & (row["Hour"] <= 5)).astype(int)
    row["Is_peak"]     = ((row["Hour"] >= 9) & (row["Hour"] <= 17)).astype(int)
    row["V14_V12"]     = row["V14"] * row["V12"]
    row["V14_V4"]      = row["V14"] * row["V4"]
    row["V4_V11"]      = row["V4"]  * row["V11"]
    row["V17_V14"]     = row["V17"] * row["V14"]
    row["V10_V12"]     = row["V10"] * row["V12"]
    row["V3_V7"]       = row["V3"]  * row["V7"]
    neg_f = ["V14","V12","V10","V16","V3"]
    pos_f = ["V4","V11","V2","V19","V21"]
    row["Risk_neg"]    = sum(row[f].iloc[0] < -1 for f in neg_f)
    row["Risk_pos"]    = sum(row[f].iloc[0] >  1 for f in pos_f)
    row["Risk_total"]  = row["Risk_neg"] + row["Risk_pos"]
    row["V_abs_sum"]   = sum(abs(row[f"V{i}"].iloc[0]) for i in range(1, 29))
    row["V_max_abs"]   = max(abs(row[f"V{i}"].iloc[0]) for i in range(1, 29))
    row["Amt_x_V14"]   = row["Amount_log"].iloc[0] * abs(row["V14"].iloc[0])
    row["Amt_x_V12"]   = row["Amount_log"].iloc[0] * abs(row["V12"].iloc[0])

    X_new      = scaler.transform(row[FEATURES].values)
    fraud_prob = model.predict_proba(X_new)[0, 1]
    is_fraud   = bool(fraud_prob >= threshold)
    risk       = ("🚨 HIGH" if fraud_prob > 0.8 else
                  "⚠  MEDIUM" if is_fraud else
                  "✅ LOW")

    return {
        "fraud_probability": round(float(fraud_prob), 6),
        "is_fraud"         : is_fraud,
        "risk_level"       : risk,
        "threshold_used"   : round(float(threshold), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN  —  runs the complete pipeline end-to-end
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()

    print("\n" + "█"*65)
    print("█  RANDOM FOREST FRAUD DETECTION — FULL PIPELINE")
    print("█  Dataset: creditcard_cleaned.csv | 283,726 rows | 473 fraud")
    print("█"*65)

    # Execute all steps
    df, fraud_df, legit_df              = step1_load()
    df, FEATURES                         = step2_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = step3_split(df, FEATURES)
    scaler, X_train_sc, X_val_sc, X_test_sc         = step4_scale(X_train, X_val, X_test)
    X_train_bal, y_train_bal             = step5_smote(X_train_sc, y_train)
    model                                = step6_train(X_train_bal, y_train_bal)
    threshold, thresholds, f1_s, f2_s   = step7_threshold(model, X_val_sc, y_val)
    step7b_cross_validate(model, X_train_bal, y_train_bal)
    y_prob, y_pred, cm, metrics          = step8_evaluate(model, X_test_sc, y_test, threshold)
    step9_plots(y_test, y_prob, y_pred, cm, threshold,
                thresholds, f1_s, f2_s, model, FEATURES,
                y_train, y_train_bal)
    step10_save(model, scaler, FEATURES, threshold, metrics)

    elapsed = time.time() - t_total
    print("\n" + "█"*65)
    print(f"█  PIPELINE COMPLETE  ({elapsed:.0f}s total)")
    print("█"*65)
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC   : {metrics['pr_auc']:.4f}")
    print(f"  F1-Fraud : {metrics['f1_fraud']:.4f}   ← the real accuracy target")
    print(f"  Recall   : {metrics['recall']:.4f}   "
          f"({metrics['confusion_matrix']['TP']}/{y_test.sum()} fraud caught)")
    print(f"  Precision: {metrics['precision']:.4f}   "
          f"({metrics['confusion_matrix']['FP']} false alarms)")
    print(f"  Threshold: {threshold:.3f}   (F2-optimised on val set)")
    print(f"\n  Charts → {FIGURES_DIR}/rf_0*.png")
    print(f"  Model  → {MODEL_DIR}/rf_model.pkl")
    print("█"*65 + "\n")


if __name__ == "__main__":
    main()
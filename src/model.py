"""
NBA Breakout Season Detector — model training and comparison.

Trains two models on year-over-year player transitions and compares them:
  1. Logistic regression (current best)
  2. Random Forest (tested, underperforms — see note at bottom)

Uses a temporal train/test split so the model never sees future seasons.
"""
from model_config import FEATURE_COLS, eligible_mask
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

feature_cols = FEATURE_COLS

df = pd.read_csv("data/labeled_transitions.csv")

eligibile = eligible_mask(df)
print(f"Eligible pool: {eligibile.sum()} of {len(df)} transitions " f"({df.loc[eligibile, 'BREAKOUT'].sum()} breakouts)")
df = df[eligibile].copy()

train = df[df["SEASON"] <= "2018-19"].copy()
test = df[df["SEASON"] >= "2019-20"].copy()


print(f"Train: {len(train)} transitions, {train['BREAKOUT'].sum()} breakouts "
      f"({train['BREAKOUT'].mean():.1%})")
print(f"Test:  {len(test)} transitions, {test['BREAKOUT'].sum()} breakouts "
      f"({test['BREAKOUT'].mean():.1%})")

X_train, y_train = train[feature_cols], train["BREAKOUT"]
X_test, y_test = test[feature_cols], test["BREAKOUT"]

# ── Logistic regression ─────────────────────────────────────────
# Needs scaling: coefficients are in units of the input features, so
# unscaled inputs make them incomparable to each other.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(class_weight="balanced", max_iter=1000)
lr.fit(X_train_scaled, y_train)

lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
lr_pred = (lr_proba >= 0.5).astype(int)

coefs = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": lr.coef_[0],
}).sort_values("coefficient", ascending=False)

print("\n--- LR Coefficients (positive = predicts breakout) ---")
print(coefs.to_string(index=False))

print("\n--- LR Test Performance ---")
print(classification_report(y_test, lr_pred, target_names=["No Breakout", "Breakout"]))

# ── Random Forest ───────────────────────────────────────────────
# Trees split on raw thresholds, so no scaling needed — unscaled input.
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,              # shallow: only ~124 positives to learn from
    min_samples_leaf=20,      # prevents leaves built from 2-3 specific players
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred = (rf_proba >= 0.5).astype(int)

print("\n--- RF Test Performance ---")
print(classification_report(y_test, rf_pred, target_names=["No Breakout", "Breakout"]))

rf_importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n--- RF Feature Importances ---")
print("(unsigned, sums to 1.0 — correlated features split credit)")
print(rf_importances.to_string(index=False))

# ── Head-to-head ────────────────────────────────────────────────
# AUC is the metric that matters here: it measures ranking quality,
# which is what a candidate shortlist actually needs. Precision/recall
# at threshold 0.5 is close to meaningless at a ~5% base rate.
print("\n--- ROC AUC ---")
print(f"LR: {roc_auc_score(y_test, lr_proba):.3f}")
print(f"RF: {roc_auc_score(y_test, rf_proba):.3f}")

# ── Predictions ─────────────────────────────────────────────────
test_results = test.copy()
test_results["LR_PROB"] = lr_proba
test_results["RF_PROB"] = rf_proba

display_cols = [
    "PLAYER_NAME", "PREV_SEASON", "SEASON",
    "PREV_AGE", "PREV_MIN", "PREV_PTS", "PTS", "BREAKOUT",
]

for label, col in [("LR", "LR_PROB"), ("RF", "RF_PROB")]:
    top = test_results.sort_values(col, ascending=False)[display_cols + [col]].head(25)
    hits = int(top["BREAKOUT"].sum())
    print(f"\n--- Top 25 Predicted Breakouts ({label}) — {hits} real ---")
    print(top.to_string(index=False))

# ── Specific player check ───────────────────────────────────────
print("\n--- Specific player check ---")
players_to_check = [
    "Jalen Brunson", "Shai Gilgeous-Alexander", "Tyrese Maxey",
    "Nickeil Alexander-Walker", "Jordan Poole", "Jaylen Brown",
]
for name in players_to_check:
    rows = test_results[test_results["PLAYER_NAME"] == name][display_cols + ["LR_PROB"]]
    if len(rows) > 0:
        print(rows.to_string(index=False))
        print()


seasons = sorted(df["SEASON"].unique())
test_seasons = [s for s in seasons if s >= "2016-17"]

print("\n--- Walk-forward AUC per fold ---")
fold_results = []
for test_season in test_seasons:
    fold_train = df[df["SEASON"] < test_season]
    fold_test = df[df["SEASON"] == test_season]

    Xtr, ytr = fold_train[feature_cols], fold_train["BREAKOUT"]
    Xte, yte = fold_test[feature_cols], fold_test["BREAKOUT"]

    # Need both classes present in the test fold or AUC is undefined
    if yte.nunique() < 2:
        print(f"{test_season}: skipped (only one class in test)")
        continue

    # LR — scaler is fit on THIS fold's training rows only, then applied
    # to test. Fitting it on all data would leak test info into training.
    fold_scaler = StandardScaler()
    Xtr_s = fold_scaler.fit_transform(Xtr)
    Xte_s = fold_scaler.transform(Xte)
    fold_lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    fold_lr.fit(Xtr_s, ytr)
    lr_auc = roc_auc_score(yte, fold_lr.predict_proba(Xte_s)[:, 1])

    # RF — unscaled, fresh forest each fold
    fold_rf = RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=20,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    fold_rf.fit(Xtr, ytr)
    rf_auc = roc_auc_score(yte, fold_rf.predict_proba(Xte)[:, 1])

    fold_results.append({"season": test_season, "lr_auc": lr_auc, "rf_auc": rf_auc})
    winner = "LR" if lr_auc > rf_auc else "RF"
    print(f"{test_season}: LR={lr_auc:.3f}  RF={rf_auc:.3f}  -> {winner}")


cv = pd.DataFrame(fold_results)
gap = cv["lr_auc"] - cv["rf_auc"]          # paired, fold by fold diff

print("\n--- Walk-forward CV summary ---")
print(f"Folds evaluated:  {len(cv)}")
print(f"LR mean AUC:      {cv['lr_auc'].mean():.3f}  (std {cv['lr_auc'].std():.3f})")
print(f"RF mean AUC:      {cv['rf_auc'].mean():.3f}  (std {cv['rf_auc'].std():.3f})")
print(f"Fold wins — LR: {(gap > 0).sum()}  RF: {(gap < 0).sum()}")
print(f"Mean paired gap (LR - RF): {gap.mean():+.3f}  (std {gap.std():.3f})")
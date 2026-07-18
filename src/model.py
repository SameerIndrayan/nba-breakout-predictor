"""
NBA Breakout Season Detector — model training and comparison.

Trains two models on year-over-year player transitions and compares them:
  1. Logistic regression (current best)
  2. Random Forest (tested, underperforms — see note at bottom)

Uses a temporal train/test split so the model never sees future seasons.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# ── Load ────────────────────────────────────────────────────────
df = pd.read_csv("data/labeled_transitions.csv")

# ── Eligibility filter ──────────────────────────────────────────
# Restrict to players who could plausibly break out under our own
# definition. Without this, the model spends most of its capacity
# rediscovering the age/GP gates baked into the label, which inflates
# AUC without adding real predictive signal.
eligible = (
    (df["PREV_AGE"] <= 26) &
    (df["PREV_GP"] >= 30) &
    (df["PREV_MIN"] >= 12)
)
print(f"Eligible pool: {eligible.sum()} of {len(df)} transitions "
      f"({df.loc[eligible, 'BREAKOUT'].sum()} breakouts)")
df = df[eligible].copy()

# ── Temporal split ──────────────────────────────────────────────
train = df[df["SEASON"] <= "2018-19"].copy()
test = df[df["SEASON"] >= "2019-20"].copy()

print(f"Train: {len(train)} transitions, {train['BREAKOUT'].sum()} breakouts "
      f"({train['BREAKOUT'].mean():.1%})")
print(f"Test:  {len(test)} transitions, {test['BREAKOUT'].sum()} breakouts "
      f"({test['BREAKOUT'].mean():.1%})")

feature_cols = [
    "PREV_AGE", "PREV_GP", "PREV_MIN", "PREV_PTS", "PREV_REB", "PREV_AST",
    "PREV_FG_PCT", "PREV_FG3_PCT", "PREV_FT_PCT", "PREV_TS_PCT",
    "PREV_USG_PCT", "PREV_AST_PCT", "PREV_PIE",
    "PREV_PTS_PER36", "PREV_REB_PER36", "PREV_AST_PER36",
    "TEAM_CHANGED",
]

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


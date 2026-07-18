import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/labeled_transitions.csv")



train = df[df["SEASON"] <= "2018-19"].copy()
test = df[df["SEASON"] >= "2019-20"].copy()

print(f"Train: {len(train)} transitions, {train['BREAKOUT'].sum()} breakouts ({train['BREAKOUT'].mean():.1%})")
print(f"Test:  {len(test)} transitions, {test['BREAKOUT'].sum()} breakouts ({test['BREAKOUT'].mean():.1%})")

feature_cols = [
    "PREV_AGE", "PREV_GP", "PREV_MIN", "PREV_PTS", "PREV_REB", "PREV_AST",
    "PREV_FG_PCT", "PREV_FG3_PCT", "PREV_FT_PCT", "PREV_TS_PCT",
    "PREV_USG_PCT", "PREV_AST_PCT", "PREV_PIE",
    "PREV_PTS_PER36", "PREV_REB_PER36", "PREV_AST_PER36",
    "TEAM_CHANGED"
]

X_train = train[feature_cols]
y_train = train["BREAKOUT"]
X_test = test[feature_cols]
y_test = test["BREAKOUT"]

print(f"Features: {feature_cols}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  log reg
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train_scaled, y_train)

coefs = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": model.coef_[0]
}).sort_values("coefficient", ascending=False)

print("\nFeature importance (positive = predicts breakout):")
print(coefs.to_string(index=False))

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of breakout

print("\n--- Test Set Performance ---")
print(classification_report(y_test, y_pred, target_names=["No Breakout", "Breakout"]))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

test_results = test.copy()
test_results["BREAKOUT_PROB"] = y_proba

# top 25 most confident predictions
print("\n--- Top 25 Predicted Breakouts (Test Set) ---")
top_predictions = test_results.sort_values("BREAKOUT_PROB", ascending=False)[
    ["PLAYER_NAME", "PREV_SEASON", "SEASON", "PREV_AGE", "PREV_PTS", "PTS", "BREAKOUT", "BREAKOUT_PROB"]
].head(25)
print(top_predictions.to_string(index=False))

print("\n--- Specific player check ---")
players_to_check = ["Jalen Brunson", "Shai Gilgeous-Alexander", "Tyrese Maxey",
                    "Nickeil Alexander-Walker", "Jordan Poole", "Jaylen Brown"]
for name in players_to_check:
    player_rows = test_results[test_results["PLAYER_NAME"] == name][
        ["PLAYER_NAME", "PREV_SEASON", "SEASON", "PREV_AGE", "PREV_PTS", "PTS", "BREAKOUT", "BREAKOUT_PROB"]
    ]
    if len(player_rows) > 0:
        print(player_rows.to_string(index=False))
        print()
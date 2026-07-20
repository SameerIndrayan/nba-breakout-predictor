"""
NBA Breakout Season Detector — upcoming-season candidate board.

Trains logistic regression on ALL eligible historical transitions, then
scores every eligible player's most recent season to rank breakout
candidates for the season that hasn't been played yet.
"""


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

raw = pd.read_csv("data/player_stats_all_seasons.csv")

# The most recent season is the launchpad: its stats become the PREV_
# features for a not-yet-played next season.
latest_season = sorted(raw["SEASON"].unique())[-1]
print(f"Predicting breakouts for the season AFTER {latest_season}")

cur = raw[raw["SEASON"] == latest_season].copy()

# Rename this season's raw stats into the PREV_ names the model expects.
rename_map = {
    "AGE": "PREV_AGE", "GP": "PREV_GP", "MIN": "PREV_MIN",
    "PTS": "PREV_PTS", "REB": "PREV_REB", "AST": "PREV_AST",
    "FG_PCT": "PREV_FG_PCT", "FG3_PCT": "PREV_FG3_PCT", "FT_PCT": "PREV_FT_PCT",
    "TS_PCT": "PREV_TS_PCT", "USG_PCT": "PREV_USG_PCT", "AST_PCT": "PREV_AST_PCT",
    "PIE": "PREV_PIE",
}
cur = cur.rename(columns=rename_map)

# per-36 rates — same formula as features.py
prev_min_safe = cur["PREV_MIN"].replace(0, 0.1)
cur["PREV_PTS_PER36"] = cur["PREV_PTS"] / prev_min_safe * 36
cur["PREV_REB_PER36"] = cur["PREV_REB"] / prev_min_safe * 36
cur["PREV_AST_PER36"] = cur["PREV_AST"] / prev_min_safe * 36

# Next season's team is unknown (free agency/trades), so we can't know if
# they'll switch. 0 = "assume stays" — a conservative default.
cur["TEAM_CHANGED"] = 0

# Same eligibility gate we trained under.
eligible = (
    (cur["PREV_AGE"] <= 26) &
    (cur["PREV_GP"] >= 30) &
    (cur["PREV_MIN"] >= 12)
)
cur = cur[eligible].copy()
print(f"Eligible candidates from {latest_season}: {len(cur)}")


feature_cols = [
    "PREV_AGE", "PREV_GP", "PREV_MIN", "PREV_PTS", "PREV_REB", "PREV_AST",
    "PREV_FG_PCT", "PREV_FG3_PCT", "PREV_FT_PCT", "PREV_TS_PCT",
    "PREV_USG_PCT", "PREV_AST_PCT", "PREV_PIE",
    "PREV_PTS_PER36", "PREV_REB_PER36", "PREV_AST_PER36",
    "TEAM_CHANGED",
]

hist = pd.read_csv("data/labeled_transitions.csv")
hist = hist[
    (hist["PREV_AGE"] <= 26) &
    (hist["PREV_GP"] >= 30) &
    (hist["PREV_MIN"] >= 12)
].copy()
print(f"Training on {len(hist)} eligible historical transitions "
      f"({hist['BREAKOUT'].sum()} breakouts)")

X_hist, y_hist = hist[feature_cols], hist["BREAKOUT"]

scaler = StandardScaler()
X_hist_scaled = scaler.fit_transform(X_hist)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_hist_scaled, y_hist)
feature_cols = [
    "PREV_AGE", "PREV_GP", "PREV_MIN", "PREV_PTS", "PREV_REB", "PREV_AST",
    "PREV_FG_PCT", "PREV_FG3_PCT", "PREV_FT_PCT", "PREV_TS_PCT",
    "PREV_USG_PCT", "PREV_AST_PCT", "PREV_PIE",
    "PREV_PTS_PER36", "PREV_REB_PER36", "PREV_AST_PER36",
    "TEAM_CHANGED",
]

hist = pd.read_csv("data/labeled_transitions.csv")
hist = hist[
    (hist["PREV_AGE"] <= 26) &
    (hist["PREV_GP"] >= 30) &
    (hist["PREV_MIN"] >= 12)
].copy()
print(f"Training on {len(hist)} eligible historical transitions "
      f"({hist['BREAKOUT'].sum()} breakouts)")

X_hist, y_hist = hist[feature_cols], hist["BREAKOUT"]

scaler = StandardScaler()
X_hist_scaled = scaler.fit_transform(X_hist)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_hist_scaled, y_hist)

X_cur = cur[feature_cols]
X_cur_scaled = scaler.transform(X_cur)
cur["BREAKOUT_PROB"] = model.predict_proba(X_cur_scaled)[:, 1]

board = cur.sort_values("BREAKOUT_PROB", ascending=False)[
    ["PLAYER_NAME", "PREV_AGE", "PREV_MIN", "PREV_PTS",
     "PREV_USG_PCT", "PREV_TS_PCT", "PREV_PIE", "BREAKOUT_PROB"]
]

print(f"\n--- 2026-27 Breakout Candidate Board (top 25) ---")
print(board.head(25).to_string(index=False))

# Save the full ranked list as the deliverable
board.to_csv("data/breakout_candidates.csv", index=False)
print(f"\nSaved full ranked board ({len(board)} candidates) to data/breakout_candidates.csv")
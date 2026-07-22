from model_config import FEATURE_COLS, eligible_mask
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

raw = pd.read_csv("data/player_stats_all_seasons.csv")

# most recent szn is the PREV_ features for a next season (not played yet)
latest_season = sorted(raw["SEASON"].unique())[-1]
print(f"Predicting breakouts for the season AFTER {latest_season}")

cur = raw[raw["SEASON"] == latest_season].copy()

# rename the szn's raw stats into the PREV_ names 
rename_map = {
    "AGE": "PREV_AGE", "GP": "PREV_GP", "MIN": "PREV_MIN",
    "PTS": "PREV_PTS", "REB": "PREV_REB", "AST": "PREV_AST",
    "FG_PCT": "PREV_FG_PCT", "FG3_PCT": "PREV_FG3_PCT", "FT_PCT": "PREV_FT_PCT",
    "TS_PCT": "PREV_TS_PCT", "USG_PCT": "PREV_USG_PCT", "AST_PCT": "PREV_AST_PCT",
    "PIE": "PREV_PIE",
}
cur = cur.rename(columns=rename_map)

# per 36
prev_min_safe = cur["PREV_MIN"].replace(0, 0.1)
cur["PREV_PTS_PER36"]  = cur["PREV_PTS"] / prev_min_safe * 36
cur["PREV_REB_PER36"]  = cur["PREV_REB"] / prev_min_safe * 36
cur["PREV_AST_PER36"]  = cur["PREV_AST"] / prev_min_safe * 36
cur["PREV_FGA_PER36"]  = cur["FGA"]  / prev_min_safe * 36
cur["PREV_FG3A_PER36"] = cur["FG3A"] / prev_min_safe * 36
cur["PREV_FTA_PER36"]  = cur["FTA"]  / prev_min_safe * 36


cur["TEAM_CHANGED"] = 0

# same eligibility gate as training, from model_config.
cur = cur[eligible_mask(cur)].copy()
print(f"Eligible candidates from {latest_season}: {len(cur)}")

# train log reg on all history
hist = pd.read_csv("data/labeled_transitions.csv")
hist = hist[eligible_mask(hist)].copy()
print(f"Training on {len(hist)} eligible historical transitions "
      f"({hist['BREAKOUT'].sum()} breakouts)")

X_hist, y_hist = hist[FEATURE_COLS], hist["BREAKOUT"]
scaler = StandardScaler()
X_hist_scaled = scaler.fit_transform(X_hist)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_hist_scaled, y_hist)

# give score
cur["BREAKOUT_PROB"] = model.predict_proba(scaler.transform(cur[FEATURE_COLS]))[:, 1]

board = cur.sort_values("BREAKOUT_PROB", ascending=False)[
    ["PLAYER_NAME", "PREV_AGE", "PREV_MIN", "PREV_PTS",
     "PREV_USG_PCT", "PREV_TS_PCT", "PREV_PIE", "BREAKOUT_PROB"]
]

print("\n--- 2026-27 Breakout Candidate Board (top 25) ---")
print(board.head(25).to_string(index=False))

board.to_csv("data/breakout_candidates.csv", index=False)
print(f"\nSaved full ranked board ({len(board)} candidates) to data/breakout_candidates.csv")
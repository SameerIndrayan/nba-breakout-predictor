"""
Build year-over-year player transitions and label breakout seasons.

Reads data/player_stats_all_seasons.csv, pairs each player's consecutive
seasons, computes deltas and per-36 rates, applies the breakout definition,
and writes data/labeled_transitions.csv.
"""

import pandas as pd
from pathlib import Path

df = pd.read_csv("data/player_stats_all_seasons.csv")
df = df.sort_values(["PLAYER_ID", "SEASON"]).reset_index(drop=True)


prev_cols = [
    "SEASON", "AGE", "GP", "MIN", "PTS", "REB", "AST",
    "FG_PCT", "FG3_PCT", "FT_PCT", "FGA", "FG3A", "FTA",
    "USG_PCT", "TS_PCT", "AST_PCT", "PIE",
    "TEAM_ABBREVIATION",
]
for col in prev_cols:
    df[f"PREV_{col}"] = df.groupby("PLAYER_ID")[col].shift(1)

# Drop each player's first season — no prior year to compare against.
df = df.dropna(subset=["PREV_SEASON", "PREV_PTS"]).reset_index(drop=True)



# consecutuve szn filter (if szn missed bc of injury, ex: dejounte murray)

df["PREV_YEAR"] = df["PREV_SEASON"].str[:4].astype(int)
df["YEAR"] = df["SEASON"].str[:4].astype(int)
before = len(df)
df = df[df["YEAR"] - df["PREV_YEAR"] == 1].copy()
print(f"Dropped {before - len(df)} non-consecutive transitions")

df["TEAM_CHANGED"] = (
    df["TEAM_ABBREVIATION"] != df["PREV_TEAM_ABBREVIATION"]
).astype(int)

# protect against divide by zero for players with 0 mins
prev_min_safe = df["PREV_MIN"].replace(0, 0.1)
df["PREV_PTS_PER36"] = df["PREV_PTS"] / prev_min_safe * 36
df["PREV_REB_PER36"] = df["PREV_REB"] / prev_min_safe * 36
df["PREV_AST_PER36"] = df["PREV_AST"] / prev_min_safe * 36

# shot attempt rates: vol of opportunity, dont care about makes.
df["PREV_FGA_PER36"] = df["PREV_FGA"] / prev_min_safe * 36
df["PREV_FG3A_PER36"] = df["PREV_FG3A"] / prev_min_safe * 36
df["PREV_FTA_PER36"] = df["PREV_FTA"] / prev_min_safe * 36

df["PTS_DELTA"] = df["PTS"] - df["PREV_PTS"]
df["PTS_PCT_CHANGE"] = (df["PTS"] - df["PREV_PTS"]) / df["PREV_PTS"].replace(0, 0.1)
df["MIN_DELTA"] = df["MIN"] - df["PREV_MIN"]
df["FG_PCT_DELTA"] = df["FG_PCT"] - df["PREV_FG_PCT"]
df["TS_DELTA"] = df["TS_PCT"] - df["PREV_TS_PCT"]

print(f"Player-transitions: {len(df)}")
print(df[["PLAYER_NAME", "PREV_SEASON", "SEASON", "PREV_PTS", "PTS"]].head(10).to_string())

#breakout def

df["BREAKOUT"] = (
    (df["PREV_AGE"] <= 26) &
    (df["PREV_GP"] >= 30) &
    (df["GP"] >= 30) &
    (df["PTS_DELTA"] >= 5) &
    (df["PTS_PCT_CHANGE"] >= 0.40) &
    (df["MIN_DELTA"] >= 3) &
    (df["TS_DELTA"] >= -0.02)
).astype(int)

print(f"\nTotal transitions: {len(df)}")
print(f"Breakouts found: {df['BREAKOUT'].sum()}")
print(f"Breakout rate: {df['BREAKOUT'].mean():.1%}\n")

print("Sample of breakouts:")
breakouts = df[df["BREAKOUT"] == 1][
    ["PLAYER_NAME", "PREV_SEASON", "SEASON", "PREV_AGE", "PREV_PTS", "PTS", "MIN_DELTA"]
].sort_values("PTS", ascending=False)
print(breakouts.head(20).to_string())

output_cols = [
    "PLAYER_ID", "PLAYER_NAME", "PREV_SEASON", "SEASON",
    "PREV_AGE", "PREV_GP", "PREV_MIN", "PREV_PTS", "PREV_REB", "PREV_AST",
    "PREV_FG_PCT", "PREV_FG3_PCT", "PREV_FT_PCT", "PREV_TS_PCT",
    "PREV_USG_PCT", "PREV_AST_PCT", "PREV_PIE",
    "PREV_PTS_PER36", "PREV_REB_PER36", "PREV_AST_PER36",
    "PREV_FGA_PER36", "PREV_FG3A_PER36", "PREV_FTA_PER36",
    "TEAM_CHANGED",
    "PTS_DELTA", "PTS_PCT_CHANGE", "MIN_DELTA", "FG_PCT_DELTA", "TS_DELTA",
    "PTS", "MIN", "GP", "BREAKOUT",
]

Path("data").mkdir(exist_ok=True)
df[output_cols].to_csv("data/labeled_transitions.csv", index=False)
print("\nSaved labeled dataset to data/labeled_transitions.csv")
import pandas as pd

df = pd.read_csv("data/player_stats_all_seasons.csv")
df = df.sort_values(["PLAYER_ID", "SEASON"]).reset_index(drop=True)

# get last szn
prev_cols = ["AGE", "GP", "MIN", "PTS", "REB", "AST", "FG_PCT", "FG3_PCT", "FT_PCT", "FGA", "FTA"]
for col in prev_cols:
    df[f"PREV_{col}"] = df.groupby("PLAYER_ID")[col].shift(1)

df["PREV_SEASON"] = df.groupby("PLAYER_ID")["SEASON"].shift(1)

df = df.dropna(subset=["PREV_PTS"]).reset_index(drop=True)

print(f"Player-transitions: {len(df)}")
print(df[["PLAYER_NAME", "PREV_SEASON", "SEASON", "PREV_PTS", "PTS"]].head(10).to_string())

df["PTS_DELTA"] = df["PTS"] - df["PREV_PTS"]
df["PTS_PCT_CHANGE"] = (df["PTS"] - df["PREV_PTS"]) / df["PREV_PTS"].replace(0, 0.1)
df["MIN_DELTA"] = df["MIN"] - df["PREV_MIN"]
df["FG_PCT_DELTA"] = df["FG_PCT"] - df["PREV_FG_PCT"]

# TS%
df["TS_PCT"] = df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"])).replace(0, 0.1)
df["PREV_TS_PCT"] = df["PREV_PTS"] / (2 * (df["PREV_FGA"] + 0.44 * df["PREV_FTA"])).replace(0, 0.1)
df["TS_DELTA"] = df["TS_PCT"] - df["PREV_TS_PCT"]

# breakout def
df["BREAKOUT"] = (
    (df["PREV_AGE"] <= 26) &
    (df["PREV_GP"] >= 30) &
    (df["GP"] >= 30) &
    (df["PTS_DELTA"] >= 5) &
    (df["PTS_PCT_CHANGE"] >= 0.40) &
    (df["MIN_DELTA"] >= 3) &
    (df["TS_DELTA"] >= -0.02)
).astype(int)

print(f"Total transitions: {len(df)}")
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
    "PTS_DELTA", "PTS_PCT_CHANGE", "MIN_DELTA", "FG_PCT_DELTA", "TS_DELTA",
    "PTS", "MIN", "GP", "BREAKOUT"
]
df[output_cols].to_csv("data/labeled_transitions.csv", index=False)
print(f"\nSaved labeled dataset to data/labeled_transitions.csv")


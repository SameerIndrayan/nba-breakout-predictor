
import pandas as pd
from model_config import eligible_mask

RENAME_MAP = {
    "AGE": "PREV_AGE", "GP": "PREV_GP", "MIN": "PREV_MIN",
    "PTS": "PREV_PTS", "REB": "PREV_REB", "AST": "PREV_AST",
    "FG_PCT": "PREV_FG_PCT", "FG3_PCT": "PREV_FG3_PCT", "FT_PCT": "PREV_FT_PCT",
    "TS_PCT": "PREV_TS_PCT", "USG_PCT": "PREV_USG_PCT", "AST_PCT": "PREV_AST_PCT",
    "PIE": "PREV_PIE",
}

def load_history():
    # eligible labeled historical transitions 
    hist = pd.read_csv("data/labeled_transitions.csv")
    return hist[eligible_mask(hist)].reset_index(drop=True)

def load_candidates():
    # most recent szn's players as PREV_ features
    raw = pd.read_csv("data/player_stats_all_seasons.csv")
    latest = sorted(raw["SEASON"].unique())[-1]
    cur = raw[raw["SEASON"] == latest].rename(columns=RENAME_MAP).copy()

    m = cur["PREV_MIN"].replace(0, 0.1)
    cur["PREV_PTS_PER36"] = cur["PREV_PTS"] / m * 36
    cur["PREV_REB_PER36"] = cur["PREV_REB"] / m * 36
    cur["PREV_AST_PER36"] = cur["PREV_AST"] / m * 36
    cur["TEAM_CHANGED"] = 0
    return cur[eligible_mask(cur)].reset_index(drop=True)
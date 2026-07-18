from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import time
from pathlib import Path

SEASONS = [
    "2005-06", "2006-07", "2007-08", "2008-09", "2009-10",
    "2010-11", "2011-12", "2012-13", "2013-14", "2014-15",
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
    "2025-26"
]

BASE_COLS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE",
    "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "PLUS_MINUS"
]

# Advanced stats we want. USG_PCT is the key one for our purposes.
ADV_COLS = [
    "PLAYER_ID", "USG_PCT", "TS_PCT", "AST_PCT", "AST_TO", "OREB_PCT", "DREB_PCT", "REB_PCT", "PIE"
]

all_seasons = []

for season in SEASONS:
    print(f"Pulling {season}...")

    # Base per-game stats
    base = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame"
    ).get_data_frames()[0][BASE_COLS]

    time.sleep(1)

    # Advanced stats
    adv = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced"
    ).get_data_frames()[0][ADV_COLS]

    combined_season = base.merge(adv, on="PLAYER_ID", how="left")
    combined_season["SEASON"] = season
    all_seasons.append(combined_season)

    time.sleep(1)

combined = pd.concat(all_seasons, ignore_index=True)
print(f"\nDone! {len(combined)} total player-seasons across {len(SEASONS)} seasons")
print(f"Columns: {list(combined.columns)}")
print(combined.head())

Path("data").mkdir(exist_ok=True)
combined.to_csv("data/player_stats_all_seasons.csv", index=False)
print(f"Saved to data/player_stats_all_seasons.csv")
from model_config import FEATURE_COLS, eligible_mask
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# refs
hist  = pd.read_csv("data/labeled_transitions.csv")
hist = hist[eligible_mask(hist)].reset_index(drop=True)
print(f"Comp pool:  {len(hist)} historical player seasons " f"({hist['BREAKOUT'].sum()} breakouts)")

scaler = StandardScaler()
H = scaler.fit_transform(hist[FEATURE_COLS])


nn = NearestNeighbors(n_neighbors=6, metric = "euclidean")
nn.fit(H)




def comps_for(player_name, prev_season, k = 5):
    #print the k most similar historical player's season to a given one
    match = hist[(hist["PLAYER_NAME"] == player_name) & (hist['PREV_SEASON'] == prev_season)]

    if match.empty:
        print(f"no eligible season found for {player_name} ({prev_season})")
        return
    
    i = match.index[0]
    distances, indicies = nn.kneighbors(H[i:i+1], n_neighbors=k + 1)

    neighbors =hist.iloc[indicies[0][1:]][["PLAYER_NAME", "PREV_SEASON", "SEASON", "PREV_AGE", "PREV_MIN", "PREV_PTS", "PTS", "BREAKOUT"]].copy()
    neighbors['DIST'] = distances[0][1:]

    outcome = int(match["BREAKOUT"].iloc[0])
    print(f"\nComps for {player_name} ({prev_season} → {match['SEASON'].iloc[0]}, " f"broke out: {outcome}):")
    print(neighbors.to_string(index=False))


comps_for("Shai Gilgeous-Alexander", "2018-19")

from prep import load_candidates

cands = load_candidates()
C = scaler.transform(cands[FEATURE_COLS])  

def comps_for_candidate(player_name, k=8):
    match = cands[cands["PLAYER_NAME"] == player_name]
    if match.empty:
        print(f"\n{player_name} not in the eligible 2025-26 candidate pool")
        return
    pos = match.index[0]
    # query a few extra so we can drop the player's own past seasons
    distances, indices = nn.kneighbors(C[pos:pos + 1], n_neighbors=k + 5)
    comps = hist.iloc[indices[0]][
        ["PLAYER_NAME", "PREV_SEASON", "SEASON",
         "PREV_AGE", "PREV_MIN", "PREV_PTS", "PTS", "BREAKOUT"]
    ].copy()
    comps["DIST"] = distances[0]
    comps = comps[comps["PLAYER_NAME"] != player_name].head(k)   # drop his own seasons
    hits = int(comps["BREAKOUT"].sum())
    print(f"\nComps for {player_name} (2025-26 → 2026-27) — "
          f"{hits}/{len(comps)} of his comps broke out ({hits / len(comps):.0%}):")
    print(comps.to_string(index=False))


for name in ["Ronald Holland II", "Jaden Hardy", "Reed Sheppard"]:
    comps_for_candidate(name)
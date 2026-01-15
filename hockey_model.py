# ---------------------------------------------------------------
# hockey_model.py
# Hockey Prop Stop — Simplified SOG Projection Model
# ---------------------------------------------------------------

import pandas as pd
import numpy as np


# ---------------------------------------------------------------
# Step 1️⃣: Basic player form from shots.csv
# ---------------------------------------------------------------
def build_basic_form(shots_df):
    """Compute each player's recent shot form (rolling 5-game avg)."""
    if shots_df.empty:
        print("⚠️ shots.csv empty — cannot compute player form.")
        return pd.DataFrame()

    df = shots_df.rename(columns={"shooterName": "player", "teamCode": "team"}).copy()
    df["shotWasOnGoal"] = (
        df["shotWasOnGoal"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)
    )

    if "game_id" not in df.columns:
        print("⚠️ No game_id found in shots.csv")
        return pd.DataFrame()

    df = df.sort_values(["player", "game_id"])
    grouped = (
        df.groupby(["player", "team", "game_id"], as_index=False)["shotWasOnGoal"].sum()
    )

    grouped["avg_5"] = (
        grouped.groupby("player")["shotWasOnGoal"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    # Only keep each player's most recent game (latest avg_5)
    latest = grouped.groupby(["player", "team"], as_index=False).tail(1)
    print(f"✅ Player form computed for {len(latest)} players.")
    return latest[["player", "team", "avg_5"]]


# ---------------------------------------------------------------
# Step 2️⃣: Simplified projection formula
# ---------------------------------------------------------------
def simple_project_matchup(shots, teams, goalies, team_a, team_b):
    """Build simple matchup-driven SOG projection table."""
    print(f"🔍 Building simple matchup for {team_a} vs {team_b}")

    player_form = build_basic_form(shots)
    if player_form.empty:
        print("❌ No player form available.")
        return pd.DataFrame()

    # --- Prepare team-level stats ---
    team_cols = [c.lower() for c in teams.columns]
    if "xgoalsfor" not in team_cols or "xgoalsagainst" not in team_cols:
        print("⚠️ Missing xGoalsFor/xGoalsAgainst columns in teams.csv.")
        return pd.DataFrame()

    teams_clean = teams.rename(
        columns={c: c.strip() for c in teams.columns}
    )[["team", "xGoalsFor", "xGoalsAgainst"]].copy()

    # --- Prepare goalie suppression ---
    if not goalies.empty and all(c in goalies.columns for c in ["goals", "ongoal", "team"]):
        g = goalies.copy()
        g["goalieSuppression"] = 1 - (g["goals"] / g["ongoal"].replace(0, np.nan))
        goalie_stats = g[["team", "goalieSuppression"]]
    else:
        goalie_stats = pd.DataFrame(columns=["team", "goalieSuppression"])
        print("⚠️ No valid goalie suppression data.")

    # --- Merge ---
    merged = player_form.merge(teams_clean, on="team", how="left")
    merged["opponent"] = np.where(merged["team"] == team_a, team_b, team_a)

    merged = merged.merge(
        teams_clean.add_prefix("opp_"), left_on="opponent", right_on="opp_team", how="left"
    )
    merged = merged.merge(
        goalie_stats.add_prefix("opp_"), left_on="opponent", right_on="opp_team", how="left"
    )

    # --- Apply simple weighted formula ---
    merged["opp_goalieSuppression"] = merged["opp_goalieSuppression"].fillna(0.9)
    merged["xGoalsFor"] = merged["xGoalsFor"].fillna(2.5)
    merged["opp_xGoalsAgainst"] = merged["opp_xGoalsAgainst"].fillna(2.5)

    merged["Projected_SOG"] = (
        0.6 * merged["avg_5"]
        + 0.3 * (merged["xGoalsFor"] - merged["opp_xGoalsAgainst"])
        + 0.1 * (1 - merged["opp_goalieSuppression"])
    ).clip(lower=0).round(2)

    merged["SignalStrength"] = pd.cut(
        merged["Projected_SOG"],
        bins=[-np.inf, 2, 4, np.inf],
        labels=["Weak", "Moderate", "Strong"],
    )

    print(f"✅ Generated {len(merged)} projections.")
    return merged[
        ["player", "team", "opponent", "avg_5", "Projected_SOG", "SignalStrength"]
    ].sort_values("Projected_SOG", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------
# Wrapper for Streamlit
# ---------------------------------------------------------------
def project_matchup(skaters, teams, shots, goalies, lines, team_a, team_b):
    """Compatibility wrapper for Streamlit app."""
    try:
        return simple_project_matchup(shots, teams, goalies, team_a, team_b)
    except Exception as e:
        print(f"❌ Error in project_matchup: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("✅ hockey_model.py (simplified) loaded successfully.")

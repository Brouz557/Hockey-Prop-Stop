# ---------------------------------------------------------------
# hockey_model.py
# Hockey Prop Stop — Calibrated Matchup Model
# ---------------------------------------------------------------

import pandas as pd
import numpy as np


# ---------------------------------------------------------------
# Basic player form
# ---------------------------------------------------------------
def build_basic_form(shots_df):
    """Compute each player's recent shot form (rolling 5-game avg)."""
    if shots_df.empty:
        print("⚠️ shots.csv empty — cannot compute player form.")
        return pd.DataFrame()

    df = shots_df.copy()
    df.columns = df.columns.str.strip()

    team_col = "teamCode" if "teamCode" in df.columns else "team"
    player_col = "shooterName" if "shooterName" in df.columns else "player"

    required = [player_col, team_col, "game_id", "shotWasOnGoal"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        return pd.DataFrame()

    df = df[required].rename(columns={player_col: "player", team_col: "team"})
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["shotWasOnGoal"] = (
        df["shotWasOnGoal"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)
    )

    df = df.sort_values(["player", "game_id"])
    grouped = df.groupby(["player", "team", "game_id"], as_index=False)["shotWasOnGoal"].sum()

    grouped["avg_5"] = (
        grouped.groupby("player")["shotWasOnGoal"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    latest = grouped.groupby(["player", "team"], as_index=False).tail(1)
    print(f"✅ Player form computed for {len(latest)} players.")
    return latest[["player", "team", "avg_5"]]


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def normalize_team_context(teams):
    if teams.empty:
        return pd.DataFrame(columns=["team", "xGF", "xGA"])

    df = teams.copy()
    df.columns = df.columns.str.strip()
    df["team"] = df["team"].astype(str).str.upper().str.strip()

    if not {"xGoalsFor", "xGoalsAgainst"}.issubset(df.columns):
        return pd.DataFrame(columns=["team", "xGF", "xGA"])

    df = df.rename(columns={"xGoalsFor": "xGF", "xGoalsAgainst": "xGA"})
    for c in ["xGF", "xGA"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].mean() > 10:  # scale season totals to per-game
            df[c] = df[c] / 82.0

    df = df.drop_duplicates(subset=["team"])
    return df[["team", "xGF", "xGA"]]


def normalize_goalie_context(goalies):
    if goalies.empty:
        return pd.DataFrame(columns=["team", "goalieSuppression"])

    df = goalies.copy()
    df.columns = df.columns.str.strip()
    df["team"] = df["team"].astype(str).str.upper().str.strip()

    if not {"goals", "ongoal"}.issubset(df.columns):
        return pd.DataFrame(columns=["team", "goalieSuppression"])

    df["goals"] = pd.to_numeric(df["goals"], errors="coerce")
    df["ongoal"] = pd.to_numeric(df["ongoal"], errors="coerce").replace(0, np.nan)
    df["goalieSuppression"] = 1 - (df["goals"] / df["ongoal"])
    df["goalieSuppression"] = df["goalieSuppression"].clip(0.7, 0.97)

    return df.groupby("team", as_index=False)["goalieSuppression"].mean()


# ---------------------------------------------------------------
# Calibrated projection model
# ---------------------------------------------------------------
def simple_project_matchup(shots, teams, goalies, team_a, team_b):
    print(f"🔍 Building calibrated matchup for {team_a} vs {team_b}")

    player_form = build_basic_form(shots)
    if player_form.empty:
        print("❌ No player form available.")
        return pd.DataFrame()

    team_a, team_b = team_a.strip().upper(), team_b.strip().upper()
    player_form["team"] = player_form["team"].astype(str).str.upper()

    # Filter only matchup teams
    player_form = player_form[player_form["team"].isin([team_a, team_b])]
    if player_form.empty:
        print("⚠️ No players found for matchup teams.")
        return pd.DataFrame()

    team_ctx = normalize_team_context(teams)
    goalie_ctx = normalize_goalie_context(goalies)

    merged = player_form.merge(team_ctx, on="team", how="left")
    merged["opponent"] = np.where(merged["team"] == team_a, team_b, team_a)

    opp_ctx = team_ctx.rename(columns={"team": "opponent", "xGF": "opp_xGF", "xGA": "opp_xGA"})
    opp_goalie = goalie_ctx.rename(columns={"team": "opponent", "goalieSuppression": "opp_goalieSuppression"})
    merged = merged.merge(opp_ctx.drop_duplicates(subset=["opponent"]), on="opponent", how="left")
    merged = merged.merge(opp_goalie.drop_duplicates(subset=["opponent"]), on="opponent", how="left")

    merged["xGF"] = merged["xGF"].fillna(2.8)
    merged["opp_xGA"] = merged["opp_xGA"].fillna(2.8)
    merged["opp_goalieSuppression"] = merged["opp_goalieSuppression"].fillna(0.9)

    # --- Calibrated formula ---
    merged["Projected_SOG"] = (
        0.8 * merged["avg_5"]
        + 0.4 * (merged["xGF"] / 3.0)
        - 0.3 * (merged["opp_xGA"] / 3.0)
        + 0.5 * (1 - merged["opp_goalieSuppression"]) * 10
    ).clip(lower=0, upper=7).round(2)

    merged["SignalStrength"] = pd.cut(
        merged["Projected_SOG"],
        bins=[-np.inf, 2, 4, np.inf],
        labels=["Weak", "Moderate", "Strong"]
    )

    result = merged.drop_duplicates(subset=["player", "team"])[
        ["player", "team", "opponent", "avg_5", "Projected_SOG", "SignalStrength"]
    ].sort_values("Projected_SOG", ascending=False).reset_index(drop=True)

    print(f"✅ Generated {len(result)} unique calibrated projections for {team_a} vs {team_b}")
    return result


# ---------------------------------------------------------------
# Wrapper for Streamlit
# ---------------------------------------------------------------
def project_matchup(skaters, teams, shots, goalies, lines, team_a, team_b):
    try:
        return simple_project_matchup(shots, teams, goalies, team_a, team_b)
    except Exception as e:
        print(f"❌ Error in project_matchup: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("✅ hockey_model.py (calibrated) loaded successfully.")

# ---------------------------------------------------------------
# hockey_model.py — Trend-Weighted Version (Column-Safe)
# ---------------------------------------------------------------
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# 1️⃣ Basic Rolling Form (with safe deduplication)
# ---------------------------------------------------------------
def build_basic_form(shots_df):
    """Computes simple 5-game rolling average for SOG."""
    if shots_df.empty:
        print("⚠️ build_basic_form: empty DataFrame.")
        return pd.DataFrame()

    df = shots_df.copy()
    # 🧹 Clean up duplicate / messy columns
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Safe renaming
    if "shooterName" in df.columns:
        df = df.rename(columns={"shooterName": "player"})
    if "teamCode" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"teamCode": "team"})

    # Validate required columns
    if "shotWasOnGoal" not in df.columns or "game_id" not in df.columns:
        print("⚠️ Missing 'shotWasOnGoal' or 'game_id'.")
        return pd.DataFrame()

    df["shotWasOnGoal"] = df["shotWasOnGoal"].astype(int)
    df = df.sort_values(["player", "game_id"])

    grouped = (
        df.groupby(["player", "team", "game_id"], as_index=False)["shotWasOnGoal"]
        .sum()
    )

    grouped["avg_5"] = grouped.groupby("player")["shotWasOnGoal"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    latest = grouped.groupby(["player", "team"]).tail(1).reset_index(drop=True)
    print(f"✅ build_basic_form: {len(latest)} players processed.")
    return latest[["player", "team", "avg_5"]]


# ---------------------------------------------------------------
# 2️⃣ Trend Form (3/5/10/20 Rolling + Safe Dedup)
# ---------------------------------------------------------------
def build_trend_form(shots_df):
    """Adds rolling averages and trend/direction indicators."""
    if shots_df.empty:
        print("⚠️ build_trend_form: shots_df empty.")
        return pd.DataFrame()

    df = shots_df.copy()
    # 🧹 Clean column names and remove duplicates
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    if "shooterName" in df.columns:
        df = df.rename(columns={"shooterName": "player"})
    if "teamCode" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"teamCode": "team"})

    if "shotWasOnGoal" not in df.columns or "game_id" not in df.columns:
        print("⚠️ Missing key columns.")
        return pd.DataFrame()

    df["shotWasOnGoal"] = df["shotWasOnGoal"].astype(int)
    df = df.sort_values(["player", "game_id"])

    grouped = (
        df.groupby(["player", "team", "game_id"], as_index=False)["shotWasOnGoal"]
        .sum()
    )

    for w in [3, 5, 10, 20]:
        grouped[f"avg_{w}"] = grouped.groupby("player")["shotWasOnGoal"].transform(
            lambda x, w=w: x.rolling(w, min_periods=1).mean()
        )

    grouped["trend"] = grouped["avg_3"] - grouped["avg_10"]
    grouped["direction"] = np.where(grouped["trend"] > 0, "Up", "Down")

    latest = grouped.groupby(["player", "team"]).tail(1).reset_index(drop=True)
    print(f"✅ build_trend_form: computed trend data for {len(latest)} players.")
    return latest[
        ["player", "team", "avg_3", "avg_5", "avg_10", "avg_20", "trend", "direction"]
    ]


# ---------------------------------------------------------------
# 3️⃣ Team & Goalie Context
# ---------------------------------------------------------------
def build_team_goalie_context(teams_df, goalies_df):
    """Prepares simplified team and goalie suppression context."""
    team_ctx = pd.DataFrame(columns=["team", "xGoalsFor", "xGoalsAgainst"])
    if not teams_df.empty:
        df = teams_df.copy()
        if "xGoalsFor" in df.columns and "xGoalsAgainst" in df.columns:
            team_ctx = df[["team", "xGoalsFor", "xGoalsAgainst"]]

    goalie_ctx = pd.DataFrame(columns=["team", "goalieSuppression"])
    if not goalies_df.empty:
        g = goalies_df.copy()
        if "goals" in g.columns and "ongoal" in g.columns:
            g["goalieSuppression"] = 1 - (g["goals"] / g["ongoal"].replace(0, np.nan))
            goalie_ctx = g[["team", "goalieSuppression"]]
    return team_ctx, goalie_ctx


# ---------------------------------------------------------------
# 4️⃣ Simple Projection
# ---------------------------------------------------------------
def simple_project_matchup(shots, teams, goalies, team_a, team_b):
    """Simple model using avg_5."""
    pf = build_basic_form(shots)
    if pf.empty:
        print("⚠️ No player form data found.")
        return pd.DataFrame()

    pf = pf[pf["team"].isin([team_a, team_b])]
    if pf.empty:
        print("⚠️ No players found for selected teams.")
        return pd.DataFrame()

    pf["opponent"] = np.where(pf["team"] == team_a, team_b, team_a)
    pf["Projected_SOG"] = pf["avg_5"].round(2)
    pf["SignalStrength"] = pd.cut(
        pf["avg_5"], bins=[-np.inf, 2, 3.5, np.inf],
        labels=["Weak", "Moderate", "Strong"]
    )
    print(f"✅ simple_project_matchup: {len(pf)} player projections created.")
    return pf.sort_values("Projected_SOG", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------
# 5️⃣ Trend-Weighted Projection
# ---------------------------------------------------------------
def project_trend_matchup(shots, teams, goalies, team_a, team_b):
    """Trend-weighted projection using 3/5/10/20 and goalie suppression."""
    print(f"🏒 Running trend-weighted model for {team_a} vs {team_b}")
    form = build_trend_form(shots)
    team_ctx, goalie_ctx = build_team_goalie_context(teams, goalies)

    if form.empty:
        print("⚠️ No player form data available.")
        return pd.DataFrame()

    form = form[form["team"].isin([team_a, team_b])]
    if form.empty:
        print("⚠️ No players found for selected matchup.")
        return pd.DataFrame()

    # Merge opponent goalie context
    form["opponent"] = np.where(form["team"] == team_a, team_b, team_a)
    opp_goalie = goalie_ctx.rename(
        columns={"team": "opponent", "goalieSuppression": "oppGoalieSuppression"}
    )
    merged = form.merge(opp_goalie, on="opponent", how="left")
    merged["oppGoalieSuppression"] = merged["oppGoalieSuppression"].fillna(0.9)

    # Weighted projection formula
    merged["Projected_SOG"] = (
        0.45 * merged["avg_5"]
        + 0.25 * merged["avg_3"]
        + 0.2  * (merged["avg_3"] - merged["avg_10"])
        + 0.1  * (1 - merged["oppGoalieSuppression"])
    ).clip(lower=0).round(2)

    merged["SignalStrength"] = pd.cut(
        merged["trend"], bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=["Weak", "Moderate", "Strong"]
    )

    print(f"✅ project_trend_matchup: {len(merged)} projections generated.")
    return merged[
        [
            "player", "team", "opponent",
            "avg_3", "avg_5", "avg_10", "avg_20",
            "trend", "direction", "oppGoalieSuppression",
            "Projected_SOG", "SignalStrength"
        ]
    ].sort_values("Projected_SOG", ascending=False).reset_index(drop=True)

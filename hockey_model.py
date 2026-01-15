# ---------------------------------------------------------------
# hockey_model.py — Regression & Trend-Weighted Projections
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st  # for debug logging to sidebar


# ---------------------------------------------------------------
# Utility: Clean & prepare SOG data
# ---------------------------------------------------------------
def _prepare_player_shots(shots_df):
    """Standardize shots dataframe and align key columns."""
    shots_df = shots_df.copy()
    shots_df.columns = shots_df.columns.str.lower().str.strip()

    # Smart rename detection
    rename_map = {}
    for col in shots_df.columns:
        if col in ["playername", "shootername", "skater", "name"]:
            rename_map[col] = "player"
        elif col in ["teamcode", "teamname", "team_name", "playerteam", "player_team"]:
            rename_map[col] = "team"
        elif col in [
            "sog", "shots", "shots_on_goal", "shot",
            "shotsongoal", "shotwasongoal"
        ]:
            rename_map[col] = "sog"

    shots_df = shots_df.rename(columns=rename_map)

    # Drop duplicate columns by name
    shots_df = shots_df.loc[:, ~shots_df.columns.duplicated(keep="first")]

    # Explicitly pick one valid team column if multiple still exist
    team_candidates = [c for c in shots_df.columns if "team" in c]
    if len(team_candidates) > 1:
        shots_df["team"] = shots_df[team_candidates[0]]
        shots_df = shots_df.drop(columns=[c for c in team_candidates[1:] if c != "team"])

    # Same for player
    player_candidates = [c for c in shots_df.columns if "player" in c or "shooter" in c]
    if len(player_candidates) > 1:
        shots_df["player"] = shots_df[player_candidates[0]]
        shots_df = shots_df.drop(columns=[c for c in player_candidates[1:] if c != "player"])

    # If no sog yet, derive from shotWasOnGoal if present
    if "sog" not in shots_df.columns:
        if "shotwasongoal" in shots_df.columns:
            shots_df["sog"] = shots_df["shotwasongoal"].astype(float)
        else:
            shots_df["sog"] = 0.0

    # Drop nulls
    shots_df = shots_df.dropna(subset=["player", "team"])
    shots_df = shots_df.reset_index(drop=True)

    # Debug: show what columns remain
    st.sidebar.write("🧾 Shots columns after cleaning:", list(shots_df.columns))

    return shots_df


# ---------------------------------------------------------------
# SIMPLE MODEL: L5 only
# ---------------------------------------------------------------
def simple_project_matchup(shots_df, teams_df, goalies_df, team_a, team_b):
    shots_df = _prepare_player_shots(shots_df)
    shots_df = shots_df[shots_df["team"].isin([team_a, team_b])]

    results = []
    for player, df_p in shots_df.groupby("player"):
        df_p = df_p.sort_values("game_id").tail(5)
        if len(df_p) < 3:
            continue
        avg_sog = df_p["sog"].mean()
        results.append({
            "player": player,
            "team": df_p["team"].iloc[-1],
            "Projected_SOG": round(avg_sog, 2),
            "Signal_Strength": "moderate" if avg_sog >= 2 else "weak",
            "Matchup": f"{team_a} vs {team_b}",
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------
# TREND MODEL: L3/L5/L10/L20 weighted regression
# ---------------------------------------------------------------
def project_trend_matchup(shots_df, teams_df, goalies_df, team_a, team_b):
    shots_df = _prepare_player_shots(shots_df)
    shots_df = shots_df[shots_df["team"].isin([team_a, team_b])]

    results = []
    for player, df_p in shots_df.groupby("player"):
        df_p = df_p.sort_values("game_id")
        if len(df_p) < 5:
            continue

        l3 = df_p.tail(3)["sog"].mean() if len(df_p) >= 3 else np.nan
        l5 = df_p.tail(5)["sog"].mean()
        l10 = df_p.tail(10)["sog"].mean() if len(df_p) >= 10 else np.nan
        l20 = df_p.tail(20)["sog"].mean() if len(df_p) >= 20 else np.nan

        weighted = np.nanmean([
            0.4 * l3 if not np.isnan(l3) else 0,
            0.3 * l5 if not np.isnan(l5) else 0,
            0.2 * l10 if not np.isnan(l10) else 0,
            0.1 * l20 if not np.isnan(l20) else 0,
        ])

        if np.isnan(weighted) or weighted <= 0:
            continue

        opp_team = team_b if df_p["team"].iloc[-1] == team_a else team_a
        if "team" in goalies_df.columns and "sog_allowed" in goalies_df.columns:
            g_mean = goalies_df.loc[goalies_df["team"] == opp_team, "sog_allowed"].mean()
            if pd.notna(g_mean) and g_mean > 0:
                weighted *= (30 / g_mean)

        signal = "strong" if weighted >= 3.5 else ("moderate" if weighted >= 2.0 else "weak")

        results.append({
            "player": player,
            "team": df_p["team"].iloc[-1],
            "Projected_SOG": round(weighted, 2),
            "Signal_Strength": signal,
            "Matchup": f"{team_a} vs {team_b}",
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("Projected_SOG", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------
# BACKTEST: Player-level accuracy check
# ---------------------------------------------------------------
def backtest_sog_accuracy(shots_df, player_name):
    shots_df = _prepare_player_shots(shots_df)
    df_p = shots_df[shots_df["player"] == player_name].copy()
    if df_p.empty:
        return pd.DataFrame()

    df_p = df_p.sort_values("game_id")
    df_p["Projected_SOG"] = df_p["sog"].rolling(5, min_periods=3).mean()
    df_p["Actual_SOG"] = df_p["sog"]
    df_p = df_p.dropna(subset=["Projected_SOG", "Actual_SOG"])
    return df_p[["game_id", "Projected_SOG", "Actual_SOG"]].reset_index(drop=True)

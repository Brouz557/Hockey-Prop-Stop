# ---------------------------------------------------------------
# hockey_model.py — Regression & Trend-Weighted Projections (Foolproof)
# ---------------------------------------------------------------

import pandas as pd
import numpy as np


# ---------------------------------------------------------------
# Utility: Clean & prepare SOG data
# ---------------------------------------------------------------
def _prepare_player_shots(shots_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize shots dataframe and align key columns."""
    shots_df = shots_df.copy()
    shots_df.columns = shots_df.columns.str.lower().str.strip()

    rename_map = {}
    for col in shots_df.columns:
        if any(k in col for k in ["playername", "shootername", "skater", "name", "player"]):
            rename_map[col] = "player"
        elif any(k in col for k in ["teamcode", "teamname", "playerteam", "player_team", "team"]):
            rename_map[col] = "team"
        elif any(k in col for k in ["sog", "shots", "shots_on_goal", "shot", "shotsongoal", "shotwasongoal"]):
            rename_map[col] = "sog"
        elif any(k in col for k in ["gameid", "game_id", "matchid", "match_id", "game", "game id", "date"]):
            rename_map[col] = "game_id"

    shots_df = shots_df.rename(columns=rename_map)

    # Fill missing critical columns
    if "team" not in shots_df.columns:
        team_col = next((c for c in shots_df.columns if "team" in c), None)
        if team_col:
            shots_df["team"] = shots_df[team_col]

    if "player" not in shots_df.columns:
        player_col = next((c for c in shots_df.columns if "player" in c or "skater" in c), None)
        if player_col:
            shots_df["player"] = shots_df[player_col]

    if "sog" not in shots_df.columns:
        sog_col = next((c for c in shots_df.columns if "shot" in c or "sog" in c), None)
        if sog_col:
            shots_df["sog"] = shots_df[sog_col].astype(float)
        else:
            shots_df["sog"] = 0.0

    if "game_id" not in shots_df.columns:
        # Create sequential ID as fallback
        shots_df["game_id"] = np.arange(len(shots_df))

    shots_df = shots_df.dropna(subset=["player", "team"]).reset_index(drop=True)
    return shots_df


# ---------------------------------------------------------------
# Utility: Clean & prepare Goalie data
# ---------------------------------------------------------------
def _prepare_goalie_data(goalies_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize goalie dataframe for opponent SOG suppression."""
    goalies_df = goalies_df.copy()
    goalies_df.columns = goalies_df.columns.str.lower().str.strip()

    rename_map = {}
    for col in goalies_df.columns:
        if any(k in col for k in ["team", "teamname", "teamcode", "franchise"]):
            rename_map[col] = "team"
        elif any(k in col for k in ["sogallowed", "shots_against", "shotsallowed", "sog_allowed", "shots"]):
            rename_map[col] = "sog_allowed"

    goalies_df = goalies_df.rename(columns=rename_map)

    if "team" not in goalies_df.columns:
        team_col = next((c for c in goalies_df.columns if "team" in c), None)
        if team_col:
            goalies_df["team"] = goalies_df[team_col]
    if "sog_allowed" not in goalies_df.columns:
        goalies_df["sog_allowed"] = np.nanmean(
            [goalies_df[c] for c in goalies_df.select_dtypes(include=np.number).columns],
            axis=0,
        )

    return goalies_df.dropna(subset=["team"]).reset_index(drop=True)


# ---------------------------------------------------------------
# SIMPLE MODEL: L5 Only
# ---------------------------------------------------------------
def simple_project_matchup(shots_df, teams_df, goalies_df, team_a, team_b):
    shots_df = _prepare_player_shots(shots_df)
    goalies_df = _prepare_goalie_data(goalies_df)

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
    goalies_df = _prepare_goalie_data(goalies_df)

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

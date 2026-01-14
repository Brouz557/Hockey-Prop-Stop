# ---------------------------------------------------------------
# hockey_model.py
# Hockey Prop Stop — Learning Regression Version (FINAL BUILD)
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------
# 1️⃣ Parse and clean uploaded files
# ---------------------------------------------------------------
def parse_raw_files(file_dfs):
    """Reads uploaded dataframes, cleans, and returns standardized versions."""
    skaters = file_dfs.get("skaters", pd.DataFrame())
    teams = file_dfs.get("teams", pd.DataFrame())
    shots = file_dfs.get("shots", pd.DataFrame())
    goalies = file_dfs.get("goalies", pd.DataFrame())
    lines = file_dfs.get("lines", pd.DataFrame())

    # Basic cleanup
    for df in [skaters, teams, shots, goalies, lines]:
        if not df.empty:
            df.columns = df.columns.str.strip()
            for col in ["team", "teamCode"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()

    # Normalize team names
    def normalize_team(df, col):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(" ", "", regex=False)
                .replace({
                    "MAPLELEAFS": "TOR",
                    "BRUINS": "BOS",
                    "LIGHTNING": "TBL",
                    "PANTHERS": "FLA",
                    "RANGERS": "NYR",
                    "ISLANDERS": "NYI",
                    "DEVILS": "NJD",
                    "CANADIENS": "MTL",
                    "SENATORS": "OTT",
                    "SABRES": "BUF",
                })
            )
        return df

    skaters = normalize_team(skaters, "team")
    teams = normalize_team(teams, "team")
    shots = normalize_team(shots, "teamCode")
    goalies = normalize_team(goalies, "team")
    lines = normalize_team(lines, "team")

    team_list = sorted(skaters["team"].dropna().unique().tolist()) if "team" in skaters else []
    print(f"✅ Parsed data | Teams detected: {team_list}")

    return skaters, teams, shots, goalies, lines, team_list


# ---------------------------------------------------------------
# 2️⃣ Player rolling form
# ---------------------------------------------------------------
def build_player_form(shots_df):
    """Build rolling player form metrics from shots.csv."""
    if shots_df.empty:
        print("⚠️ build_player_form: shots file empty.")
        return pd.DataFrame()

    df = shots_df.copy()

    if "shooterName" in df.columns:
        df.rename(columns={"shooterName": "player"}, inplace=True)
    if "teamCode" in df.columns and "team" not in df.columns:
        df.rename(columns={"teamCode": "team"}, inplace=True)

    # Drop duplicate team columns
    while list(df.columns).count("team") > 1:
        dup = [i for i, c in enumerate(df.columns) if c == "team"][1]
        df.drop(df.columns[dup], axis=1, inplace=True)

    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.strip().str.upper()

    # Detect shotWasOnGoal column
    og_cols = [c for c in df.columns if "shotwason" in c.lower() or "ongoal" in c.lower()]
    if not og_cols:
        print("❌ build_player_form: no shotWasOnGoal column found.")
        return pd.DataFrame()

    on_goal_col = og_cols[0]
    if on_goal_col != "shotWasOnGoal":
        df.rename(columns={on_goal_col: "shotWasOnGoal"}, inplace=True)

    df["shotWasOnGoal"] = df["shotWasOnGoal"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)

    if "game_id" not in df.columns:
        df["game_id"] = pd.factorize(df.index)[0] + 1

    df = df.sort_values(["player", "game_id"])
    grouped = df.groupby(["player", "team", "game_id"], as_index=False)["shotWasOnGoal"].sum()

    for w in [3, 5, 10, 20]:
        grouped[f"avg_{w}"] = grouped.groupby("player")["shotWasOnGoal"].transform(lambda x: x.rolling(w, 1).mean())

    grouped["baseline_20"] = grouped.groupby("player")["avg_20"].transform("mean")
    grouped["std_20"] = grouped.groupby("player")["avg_20"].transform("std").fillna(0.01)
    grouped["z_score"] = (grouped["avg_5"] - grouped["baseline_20"]) / grouped["std_20"]

    latest = grouped.groupby(["player", "team"], as_index=False).tail(1)
    print(f"✅ build_player_form: computed rolling form for {len(latest)} players.")
    return latest[["player", "team", "avg_3", "avg_5", "avg_10", "avg_20", "z_score"]]


# ---------------------------------------------------------------
# 3️⃣ Team & Goalie Context (fixed parentheses)
# ---------------------------------------------------------------
def build_team_goalie_context(teams_df, goalies_df):
    """Prepares team and goalie suppression metrics."""
    if teams_df.empty:
        team_context = pd.DataFrame(columns=["team", "shotSuppression", "xGoalsFor"])
    else:
        team_context = teams_df.copy()
        found = [c for c in team_context.columns if "goal" in c.lower() and "against" in c.lower()]
        col = found[0] if found else None
        team_context["shotSuppression"] = team_context[col] if col else np.nan
        if "xGoalsFor" not in team_context.columns:
            team_context["xGoalsFor"] = np.nan
        team_context = team_context[["team", "shotSuppression", "xGoalsFor"]]

    if not goalies_df.empty:
        g = goalies_df.copy()
        g["savePct"] = 1 - (g["goals"] / g["ongoal"].replace(0, np.nan))
        g["dangerSavePct"] = 1 - (
            (g["lowDangerGoals"] + g["mediumDangerGoals"] + g["highDangerGoals"])
            / (
                g["lowDangerShots"] + g["mediumDangerShots"] + g["highDangerShots"]
            ).replace(0, np.nan)
        )
        g["goalieSuppression"] = g[["savePct", "dangerSavePct"]].mean(axis=1)
        goalie_context = g[["name", "team", "goalieSuppression"]]
    else:
        goalie_context = pd.DataFrame(columns=["name", "team", "goalieSuppression"])

    return team_context, goalie_context


# ---------------------------------------------------------------
# 4️⃣ Line Strength
# ---------------------------------------------------------------
def build_team_line_strength(lines_df):
    if lines_df.empty:
        print("⚠️ No line data provided — skipping line strength.")
        return pd.DataFrame(columns=["team", "lineStrength"])

    df = lines_df.copy()
    if "xGoalsFor" not in df.columns or "xGoalsAgainst" not in df.columns:
        print("⚠️ Missing xGoalsFor/xGoalsAgainst in lines.csv.")
        return pd.DataFrame(columns=["team", "lineStrength"])

    df["raw_strength"] = (df["xGoalsFor"] - df["xGoalsAgainst"]) / (df["xGoalsFor"] + df["xGoalsAgainst"] + 1e-6)
    team_strength = df.groupby("team")["raw_strength"].mean().reset_index()
    team_strength["lineStrength"] = (
        (team_strength["raw_strength"] - team_strength["raw_strength"].mean())
        / team_strength["raw_strength"].std()
    ).clip(-2, 2)

    print(f"🏒 Computed lineStrength for {len(team_strength)} teams.")
    return team_strength[["team", "lineStrength"]]


# ---------------------------------------------------------------
# 5️⃣ Learning Matchup Model
# ---------------------------------------------------------------
def build_matchup_model(skaters, teams, shots, goalies, lines, team_a, team_b, report_metrics=True):
    print(f"🔍 Building learning model for {team_a} vs {team_b}")

    player_form = build_player_form(shots)
    if player_form.empty:
        print("❌ No player form data.")
        return pd.DataFrame()

    team_ctx, goalie_ctx = build_team_goalie_context(teams, goalies)
    line_strength = build_team_line_strength(lines)

    merged = (
        player_form.merge(team_ctx, on="team", how="left")
        .merge(line_strength, on="team", how="left")
    )

    merged["opponent"] = np.where(merged["team"] == team_a, team_b, team_a)
    opp_strength = line_strength.rename(columns={"team": "opponent", "lineStrength": "oppLineStrength"})
    merged = merged.merge(opp_strength, on="opponent", how="left")

    merged["matchupAdj"] = merged["lineStrength"] - merged["oppLineStrength"]

    opp_goalie = goalie_ctx.rename(columns={"team": "opponent", "goalieSuppression": "oppGoalieSuppression"})
    merged = merged.merge(opp_goalie, on="opponent", how="left")

    merged["xGoalsFor"] = merged["xGoalsFor"].fillna(merged["avg_5"])
    merged["oppGoalieSuppression"] = merged["oppGoalieSuppression"].fillna(0.9)
    merged["matchupAdj"] = merged["matchupAdj"].fillna(0)

    # Proxy target
    merged["actual_SOG"] = merged["avg_3"]

    X = merged[["avg_5", "xGoalsFor", "oppGoalieSuppression", "matchupAdj"]]
    y = merged["actual_SOG"]

    model = LinearRegression().fit(X, y)
    merged["Projected_SOG"] = model.predict(X).clip(lower=0).round(2)

    if report_metrics:
        mae = mean_absolute_error(y, merged["Projected_SOG"])
        rmse = np.sqrt(mean_squared_error(y, merged["Projected_SOG"]))
        r2 = r2_score(y, merged["Projected_SOG"])
        print(f"📈 MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

    merged["SignalStrength"] = pd.cut(
        merged["z_score"], bins=[-np.inf, 0, 1, np.inf], labels=["Weak", "Moderate", "Strong"]
    )

    result = merged[merged["team"].isin([team_a, team_b])].copy()
    print(f"✅ Generated {len(result)} player projections.")
    return result.sort_values("Projected_SOG", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------
# 6️⃣ Wrapper
# ---------------------------------------------------------------
def project_matchup(skaters, teams, shots, goalies, lines, team_a, team_b):
    try:
        return build_matchup_model(skaters, teams, shots, goalies, lines, team_a, team_b, report_metrics=True)
    except Exception as e:
        import traceback
        print("❌ project_matchup failed:", e)
        traceback.print_exc()
        return pd.DataFrame()


if __name__ == "__main__":
    print("✅ hockey_model.py loaded successfully (Final Build).")

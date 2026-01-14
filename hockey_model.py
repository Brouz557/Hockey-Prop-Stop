def build_matchup_model(skaters, teams, shots, goalies, lines, team_a, team_b):
    """
    Build matchup-ready player projections using true shot-level data.
    """
    df = shots.copy()
    df.columns = df.columns.str.lower()

    # --- explicit mapping for your known headers ---
    if "shootername" in df.columns:
        df.rename(columns={"shootername": "player"}, inplace=True)
    if "game_id" not in df.columns and "gameid" in df.columns:
        df.rename(columns={"gameid": "game_id"}, inplace=True)
    if "shotwasongoal" in df.columns:
        df.rename(columns={"shotwasongoal": "shot_was_on_goal"}, inplace=True)
    if "team" not in df.columns and "teamcode" in df.columns:
        df.rename(columns={"teamcode": "team"}, inplace=True)

    # --- ensure consistent team codes ---
    for c in ["team", "hometeamcode", "awayteamcode"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()

    # --- derive opponent column ---
    if "opponent" not in df.columns:
        if "hometeamcode" in df.columns and "awayteamcode" in df.columns:
            df["opponent"] = np.where(df["team"] == df["hometeamcode"], df["awayteamcode"], df["hometeamcode"])
        else:
            df["opponent"] = np.where(df["team"] == team_a, team_b, team_a)

    # --- validate core columns ---
    required = {"player", "team", "game_id"}
    if not required.issubset(df.columns):
        print(f"Missing required columns: {required - set(df.columns)}")
        return pd.DataFrame()

    # --- filter to selected matchup ---
    df = df[df["team"].isin([team_a, team_b])]
    if df.empty:
        print("⚠️ No rows after team filter.")
        return pd.DataFrame()

    # --- mark SOG events (1 per on-goal shot attempt) ---
    df["shots_on_goal"] = np.where(df.get("shot_was_on_goal", 0) == 1, 1, 0)

    # --- aggregate player-level SOG per game ---
    sog_by_game = (
        df.groupby(["game_id", "player", "team", "opponent"])
        .agg({"shots_on_goal": "sum"})
        .reset_index()
        .sort_values(["player", "game_id"])
    )

    if sog_by_game.empty:
        print("⚠️ No SOG records after grouping.")
        return pd.DataFrame()

    # --- compute rolling form ---
    for window in [3, 5, 10, 20]:
        sog_by_game[f"recent{window}"] = (
            sog_by_game.groupby("player")["shots_on_goal"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    # --- trend and baseline ---
    sog_by_game["trend"] = sog_by_game["recent5"] - sog_by_game["recent20"]

    # --- simple matchup context placeholder ---
    sog_by_game["matchupImpact"] = np.random.uniform(-0.15, 0.15, len(sog_by_game))
    sog_by_game["goalieSuppression"] = 0.1

    # --- opportunity index ---
    sog_by_game["opportunityIndex"] = (
        0.45 * sog_by_game["recent5"]
        + 0.25 * sog_by_game["trend"]
        + 0.15 * sog_by_game["matchupImpact"]
        + 0.15 * (1 - sog_by_game["goalieSuppression"])
    )

    # --- final player-level aggregation ---
    preds = (
        sog_by_game.groupby(["player", "team", "opponent"])
        .agg({
            "opportunityIndex": "mean",
            "shots_on_goal": "mean",
            "recent5": "mean",
            "trend": "mean",
            "matchupImpact": "mean"
        })
        .reset_index()
    )

    if preds.empty:
        print("⚠️ No projections after aggregation.")
        return pd.DataFrame()

    preds["Projected SOG"] = preds["shots_on_goal"].round(2)
    preds["Signal Strength"] = pd.qcut(preds["opportunityIndex"], 3, labels=["Weak", "Moderate", "Strong"])
    preds["Opportunity Score"] = preds["opportunityIndex"].round(3)

    return preds.sort_values("opportunityIndex", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------
# Core Model (Reverted Stable Version)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def build_model(team_a, team_b, skaters_df, shots_df):
    results = []
    roster = skaters_df[skaters_df[team_col].isin([team_a, team_b])][[player_col, team_col, "position"]]
    roster = roster.rename(columns={player_col: "player", team_col: "team"}).drop_duplicates("player")
    grouped = {n.lower(): g for n, g in shots_df.groupby(shots_df["player"].str.lower())}

    for row in roster.itertuples(index=False):
        player, team = row.player, row.team
        df_p = grouped.get(player.lower(), pd.DataFrame())
        if df_p.empty:
            continue
        sog_values = df_p["sog"].tolist()
        if not sog_values:
            continue

        # Recent shot form
        last5 = sog_values[-5:] if len(sog_values) >= 5 else sog_values
        l5 = np.mean(last5)
        season_avg = np.mean(sog_values)
        trend = (l5 - season_avg) / season_avg if season_avg > 0 else 0

        # L5-based Poisson λ and projection
        lam = l5
        line = round(lam, 2)
        prob = 1 - poisson.cdf(np.floor(line) - 1, mu=lam)
        p = min(max(prob, 0.001), 0.999)

        # Convert to playable odds (American)
        odds = -100 * (p / (1 - p)) if p >= 0.5 else 100 * ((1 - p) / p)
        implied_odds = f"{'+' if odds > 0 else ''}{int(odds)}"

        results.append({
            "Player": player,
            "Team": team,
            "Position": row.position,
            "Season Avg": round(season_avg, 2),
            "L5 Avg": round(l5, 2),
            "Final Projection": round(line, 2),
            "Prob ≥ Projection (%) L5": round(p * 100, 1),
            "Playable Odds": implied_odds,
            "Trend Score": round(trend, 3)
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run Model
# ---------------------------------------------------------------
if st.button("🚀 Run Model"):
    st.info(f"Building model for matchup: **{team_a} vs {team_b}** …")
    df = build_model(team_a, team_b, skaters_df, shots_df)
    df = df.sort_values("Final Projection", ascending=False).reset_index(drop=True)
    st.session_state.results_raw = df.copy()
    st.success("✅ Model built successfully!")

# ---------------------------------------------------------------
# Display Main Table
# ---------------------------------------------------------------
if "results_raw" in st.session_state and not st.session_state.results_raw.empty:
    df = st.session_state.results_raw.copy()

    # --- Trend Color Formatting ---
    def trend_color(v):
        if pd.isna(v):
            return "–"
        v = max(min(v, 0.5), -0.5)
        n = v + 0.5
        if n < 0.5:
            r, g, b = 255, int(255 * (n * 2)), 0
        else:
            r, g, b = int(255 * (1 - (n - 0.5) * 2)), 255, 0
        color = f"rgb({r},{g},{b})"
        t = "▲" if v > 0.05 else ("▼" if v < -0.05 else "–")
        txt = "#000" if abs(v) < 0.2 else "#fff"
        return f"<div style='background:{color};color:{txt};font-weight:600;border-radius:6px;padding:4px 8px;text-align:center;'>{t}</div>"

    df["Trend"] = df["Trend Score"].apply(trend_color)

    # --- Visible Columns ---
    cols = [
        "Player", "Team", "Position", "Trend",
        "Final Projection", "Prob ≥ Projection (%) L5",
        "Playable Odds", "Season Avg", "L5 Avg"
    ]
    vis = df[[c for c in cols if c in df.columns]]

    # --- Render Table ---
    html_table = vis.to_html(index=False, escape=False)
    components.html(
        f"""
        <style>
        div.scrollable-table {{
            overflow-x: auto;
            overflow-y: auto;
            height: 600px;
            position: relative;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'Source Sans Pro', sans-serif;
            color: #f0f0f0;
        }}
        th {{
            background-color: #00B140;
            color: white;
            padding: 6px;
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 3;
        }}
        td:first-child, th:first-child {{
            position: sticky;
            left: 0;
            z-index: 4;
            background-color: #00B140;
            color: white;
            font-weight: bold;
        }}
        td {{
            background-color: #1e1e1e;
            color: #f0f0f0;
            padding: 4px;
            text-align: center;
        }}
        tr:nth-child(even) td {{ background-color: #2a2a2a; }}
        </style>
        <div class='scrollable-table'>{html_table}</div>
        """,
        height=620,
        scrolling=True,
    )

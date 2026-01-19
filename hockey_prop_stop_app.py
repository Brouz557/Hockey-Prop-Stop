# ---------------------------------------------------------------
# Run Model
# ---------------------------------------------------------------
if st.button("🚀 Run Model"):
    st.info(f"Building model for matchup: **{team_a} vs {team_b}** …")
    df = build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df)
    if "Injury" not in df.columns:
        df["Injury"] = ""
    df = df.sort_values("Final Projection", ascending=False).reset_index(drop=True)
    st.session_state.results_raw = df.copy()
    st.success("✅ Model built successfully!")

# ---------------------------------------------------------------
# Display Table + Save/Download
# ---------------------------------------------------------------
if "results_raw" in st.session_state and not st.session_state.results_raw.empty:
    df = st.session_state.results_raw.copy()

    def trend_color(v):
        if pd.isna(v): return "–"
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

    cols = [
        "Player","Team","Injury","Trend","Final Projection",
        "Prob ≥ Projection (%) L5","Playable Odds",
        "Season Avg","Line Adj","Form Indicator",
        "L3 Shots","L5 Shots","L10 Shots"
    ]
    vis = df[[c for c in cols if c in df.columns]]

    html_table = vis.to_html(index=False, escape=False)
    components.html(f"""
        <style>
        div.scrollable-table {{
            overflow-x: auto;
            overflow-y: auto;
            height: 600px;
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
        }}
        td:first-child, th:first-child {{
            position: sticky;
            left: 0;
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
        """, height=620, scrolling=True)

    # ---------------------------------------------------------------
    # 💾 Save Projections Locally + 📥 Download
    # ---------------------------------------------------------------
    st.markdown("---")
    st.subheader("💾 Save or Download Projections")

    selected_date = st.date_input("Select game date:", datetime.date.today())

    if st.button("💾 Save Projections for Selected Date"):
        df_to_save = df.copy()
        df_to_save["Date_Game"] = selected_date.strftime("%Y-%m-%d")
        df_to_save["Matchup"] = f"{team_a} vs {team_b}"

        save_dir = "projections"
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{team_a}_vs_{team_b}_{selected_date.strftime('%Y-%m-%d')}.csv"
        save_path = os.path.join(save_dir, filename)

        df_to_save.to_csv(save_path, index=False)
        st.success(f"✅ Saved projections to **{save_path}**")

        csv = df_to_save.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Projections CSV",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )

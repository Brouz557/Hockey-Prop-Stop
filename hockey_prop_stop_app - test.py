# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Test Mode (Instant Filter + Logos)
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os, requests
from scipy.stats import poisson
import streamlit.components.v1 as components

st.set_page_config(page_title="Puck Shotz Hockey Analytics (Test)", layout="wide", page_icon="🏒")
st.warning("🧪 TEST MODE — Sandbox version. Changes here won’t affect your main app.")

# ---------------------------------------------------------------
# (unchanged setup and model code above)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Display Buttons + Filtered Table (logos + white click box)
# ---------------------------------------------------------------
if "results" in st.session_state:
    df = st.session_state.results.copy()
    games = st.session_state.matchups

    cols = st.columns(3)
    for i, m in enumerate(games):
        team_a, team_b = m["away"], m["home"]
        match_id = f"{team_a}@{team_b}"
        is_selected = st.session_state.get("selected_match") == match_id

        btn_color = "#1E5A99" if is_selected else "#0A3A67"
        border = "2px solid #FF4B4B" if is_selected else "1px solid #1E5A99"
        glow = "0 0 12px #FF4B4B" if is_selected else "none"

        with cols[i % 3]:
            # --- Top blue logo section (unchanged) ---
            st.markdown(f"""
            <div style="
                background-color:{btn_color};
                border:{border};
                border-radius:8px 8px 0 0;
                color:#fff;
                font-weight:600;
                font-size:15px;
                padding:10px 14px;
                width:100%;
                box-shadow:{glow};
                display:flex;
                align-items:center;
                justify-content:center;
                gap:6px;
            ">
                <img src="{m['away_logo']}" height="22">
                <span>{m['away']}</span>
                <span style="color:#D6D6D6;">@</span>
                <span>{m['home']}</span>
                <img src="{m['home_logo']}" height="22">
            </div>
            """, unsafe_allow_html=True)

            # --- White clickable box below (now truly clickable) ---
            form_key = f"form_{i}"
            with st.form(form_key):
                st.markdown(f"""
                <div style="
                    width:100%;
                    background-color:#FFFFFF0A;
                    border:1px solid #1E5A99;
                    border-top:none;
                    border-radius:0 0 8px 8px;
                    color:#D6D6D6;
                    font-size:13px;
                    text-align:center;
                    padding:8px 0;
                    font-weight:500;
                    cursor:pointer;
                ">Click to view</div>
                """, unsafe_allow_html=True)

                clicked = st.form_submit_button("", use_container_width=True)

                if clicked:
                    if is_selected:
                        st.session_state.selected_match = None
                        st.session_state.selected_teams = None
                    else:
                        st.session_state.selected_match = match_id
                        st.session_state.selected_teams = {team_a, team_b}
                    st.rerun()

    # --- Filter + Table (unchanged) ---
    sel_teams = st.session_state.get("selected_teams")
    if sel_teams:
        df = df[df["Team"].isin(sel_teams)]
        st.markdown(f"### Showing results for: **{' vs '.join(sel_teams)}**")
    else:
        st.markdown("### Showing results for: **All Teams**")

    df["Trend"] = df["Trend Score"].apply(lambda v: "▲" if v > 0.05 else ("▼" if v < -0.05 else "–"))
    df = df.sort_values(["Team","Final Projection","Line Adj"],ascending=[True,False,False])

    html_table = df[
        ["Player","Team","Trend","Final Projection","Prob ≥ Projection (%) L5",
         "Playable Odds","Season Avg","Line Adj","Form Indicator",
         "L3 Shots","L5 Shots","L10 Shots"]
    ].to_html(index=False,escape=False)

    components.html(f"""
    <style>
    table {{
        width:100%;border-collapse:collapse;font-family:'Source Sans Pro',sans-serif;color:#D6D6D6;
    }}
    th {{
        background-color:#0A3A67;color:#FFFFFF;padding:6px;text-align:center;position:sticky;top:0;
        border-bottom:2px solid #1E5A99;
    }}
    td:first-child,th:first-child {{
        position:sticky;left:0;background-color:#1E5A99;color:#FFFFFF;font-weight:bold;
    }}
    td {{
        background-color:#0F2743;color:#D6D6D6;padding:4px;text-align:center;
    }}
    tr:nth-child(even) td {{background-color:#142F52;}}
    </style>
    <div style='overflow-x:auto;height:650px;'>{html_table}</div>
    """,height=700,scrolling=True)

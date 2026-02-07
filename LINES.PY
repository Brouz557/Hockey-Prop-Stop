import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="NST Line Data Builder",
    page_icon="🏒",
    layout="centered"
)

st.title("🏒 Natural Stat Trick Line Data Builder")
st.caption("Automated CSV pull → model-ready LINE DATA.xlsx")

# -----------------------------
# NST URL Builder
# -----------------------------
def build_nst_url(season: str):
    return (
        "https://www.naturalstattrick.com/playerteams.php"
        f"?fromseason={season}"
        f"&thruseason={season}"
        "&stype=2"
        "&sit=5v5"
        "&score=all"
        "&rate=n"
        "&team=ALL"
        "&pos=S"
        "&loc=B"
        "&toi=0"
        "&gpfilt=none"
        "&fd=&td="
        "&lines=y"
        "&csv=y"
    )

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Settings")

season = st.sidebar.selectbox(
    "Season",
    ["20242025", "20232024", "20222023"],
    index=0
)

output_file = st.sidebar.text_input(
    "Output filename",
    value="LINE DATA.xlsx"
)

run_button = st.sidebar.button("🚀 Pull & Build Line Data")

# -----------------------------
# Run Pipeline
# -----------------------------
if run_button:
    with st.spinner("Downloading Natural Stat Trick CSV..."):
        url = build_nst_url(season)
        try:
            nst = pd.read_csv(url)
        except Exception as e:
            st.error(f"Failed to download NST data:\n{e}")
            st.stop()

    st.success("NST CSV downloaded")
    st.write("Raw preview:", nst.head())

    # Normalize columns
    nst.columns = nst.columns.str.lower().str.strip()

    # Detect player columns automatically
    player_cols = [c for c in nst.columns if c.startswith("player")]

    if not player_cols:
        st.error("No player columns found in NST CSV")
        st.stop()

    def build_line(row):
        names = []
        for c in player_cols:
            if pd.notna(row[c]):
                names.append(row[c].split()[-1].lower())
        return " ".join(names)

    nst["line pairings"] = nst.apply(build_line, axis=1)

    # Defensive aggregation (column-safe)
    agg_dict = {}
    if "game" in nst.columns:
        agg_dict["games"] = ("game", "nunique")
    else:
        agg_dict["games"] = ("team", "count")

    if "shots against" in nst.columns:
        agg_dict["sog against"] = ("shots against", "sum")
    else:
        agg_dict["sog against"] = ("team", "count")

    out = (
        nst
        .groupby(["team", "line pairings"], as_index=False)
        .agg(**agg_dict)
    )

    out = out.sort_values(["team", "games"], ascending=[True, False])

    # Save
    out.to_excel(output_file, index=False)

    st.success(f"✅ {output_file} saved")
    st.caption(f"Last updated: {datetime.now():%Y-%m-%d %H:%M}")

    st.subheader("Model-Ready Preview")
    st.dataframe(out, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built for deterministic hockey analytics pipelines · NST → LINE DATA")

import time
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from datetime import datetime
from nba_api.stats.static import players as static_players, teams as static_teams
from nba_api.stats.endpoints import (
    playergamelog, playercareerstats, leaguedashteamstats
)

# ---------------------- Streamlit Page Setup ----------------------
st.set_page_config(page_title="NBA Scouting & Projections", layout="wide")
st.title("🏀 NBA Player Scouting & Projections (nba_api)")

# ---------------------- Helpers ----------------------
THIS_SEASON = "2024-25"  # adjust each year or compute dynamically if you prefer
LEAGUE_DEF_FALLBACK = 112.0
LEAGUE_PACE_FALLBACK = 99.0
CACHE_HOURS = 12

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_active_players_df():
    return pd.DataFrame(static_players.get_active_players())  # id, full_name, etc.

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_teams_df():
    return pd.DataFrame(static_teams.get_teams())  # id, full_name, abbreviation, etc.

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
def get_team_defense_table(season: str):
    # league dashboard with team DefRtg & Pace
    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season, measure_type_detailed_defense="Defense"
    ).get_data_frames()[0]
    # normalize key columns
    keep = ["TEAM_ID","TEAM_NAME","PACE","DEF_RATING"]
    return df[keep].copy()

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
def get_player_career(player_id: int):
    df = playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]
    return df

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
def get_player_gamelog(player_id: int, season: str):
    df = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    # ensure proper dtypes
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    return df

def form_score(series: pd.Series, k: int = 5) -> float:
    """Return a simple 0–100 form score based on last k vs season mean/std."""
    series = pd.to_numeric(series, errors="coerce")
    if series.empty:
        return 50.0
    recent = series.head(k)
    mu_r, mu_s, sd_s = recent.mean(), series.mean(), series.std(ddof=1)
    if pd.isna(sd_s) or sd_s == 0:
        return 50.0
    z = (mu_r - mu_s) / sd_s
    return float(np.clip(50 + 15*z, 0, 100))

def blended_projection(r: float, s: float, c: float, weights=(0.55,0.30,0.15)) -> float:
    wr, ws, wc = weights
    total = max(wr + ws + wc, 1e-9)
    wr, ws, wc = wr/total, ws/total, wc/total
    return wr*r + ws*s + wc*c

def opponent_adjustment(def_rating: float, pace: float,
                        league_def=LEAGUE_DEF_FALLBACK, league_pace=LEAGUE_PACE_FALLBACK) -> float:
    """Scale by defensive difficulty and pace (lower DefRtg = harder)."""
    def_scale = league_def / max(def_rating, 1e-9)
    pace_scale = max(pace,1e-9) / league_pace
    return def_scale * pace_scale

# ---------------------- Data Load ----------------------
players_df = get_active_players_df()
teams_df   = get_teams_df()
team_def   = get_team_defense_table(THIS_SEASON)

# ---------------------- Sidebar Controls ----------------------
with st.sidebar:
    st.header("Controls")
    q = st.text_input("Search player", value="Jayson Tatum")
    # match player by case-insensitive substring
    options = players_df[players_df["full_name"].str.contains(q, case=False, na=False)].sort_values("full_name")
    if options.empty:
        st.warning("No active players match your search. Try another name.")
        st.stop()
    player_name = st.selectbox("Pick player", options["full_name"].tolist())
    player_id = int(options.loc[options["full_name"]==player_name, "id"].iloc[0])

    opp_name = st.selectbox("Opponent (for projection)", sorted(team_def["TEAM_NAME"].unique()))
    n_recent = st.slider("Recent window (games)", 3, 15, 5)
    w_recent = st.slider("Weight: Recent", 0.0, 1.0, 0.55, 0.05)
    w_season = st.slider("Weight: Season", 0.0, 1.0, 0.30, 0.05)
    w_career = st.slider("Weight: Career", 0.0, 1.0, 0.15, 0.05)
    weights = (w_recent, w_season, w_career)

# ---------------------- Pull Player Data ----------------------
with st.spinner("Loading player data..."):
    career_raw = get_player_career(player_id)
    logs = get_player_gamelog(player_id, THIS_SEASON)

if logs.empty:
    st.error("No game logs found for this player this season.")
    st.stop()

# Key stats we’ll track (can expand easily)
STAT_COLS = ["PTS","REB","AST","STL","BLK","FGM","FGA","FG_PCT","FG3M","FG3A","FTM","FTA","TOV","PLUS_MINUS","MIN"]
stat_cols_existing = [c for c in STAT_COLS if c in logs.columns]

# Aggregates
season_mean = logs[stat_cols_existing].mean(numeric_only=True)
recent_mean = logs[stat_cols_existing].head(n_recent).mean(numeric_only=True)

# career: take career per-game averages from career_raw (last row or overall)
career_pg_cols = [c for c in ["PTS","REB","AST","STL","BLK","FG_PCT","FG3_PCT","FT_PCT","MIN"] if c in career_raw.columns]
if "SEASON_ID" in career_raw.columns:
    # overall row is often 'Career' aggregated; if not, compute mean over all seasons
    career_overall = career_raw.sort_values("SEASON_ID").tail(1) if "GP" in career_raw.columns else career_raw.tail(1)
    career_mean = career_overall
else:
    career_mean = career_raw.tail(1)
career_means = {}
# Map career columns to our stat names
career_map = {
    "PTS":"PTS","REB":"REB","AST":"AST","STL":"STL","BLK":"BLK",
    "FG_PCT":"FG_PCT","FG3_PCT":"FG3_PCT","FT_PCT":"FT_PCT","MIN":"MIN"
}
for k,v in career_map.items():
    if v in career_raw.columns:
        career_means[k] = float(career_raw[v].mean())
career_means = pd.Series(career_means)

# Opponent numbers
opp_row = team_def.loc[team_def["TEAM_NAME"]==opp_name].iloc[0]
opp_def = float(opp_row.get("DEF_RATING", LEAGUE_DEF_FALLBACK))
opp_pace = float(opp_row.get("PACE", LEAGUE_PACE_FALLBACK))
adj = opponent_adjustment(opp_def, opp_pace)

# ---------------------- Header & Badges ----------------------
left, right = st.columns([2,1])
with left:
    st.subheader(f"{player_name} — {THIS_SEASON}")
with right:
    st.metric("Opponent DEF Rating", f"{opp_def:.1f}")
    st.metric("Opponent Pace", f"{opp_pace:.1f}")

# ---------------------- KPI Row ----------------------
kpi_stats = ["PTS","REB","AST","MIN"]
k_cols = st.columns(len(kpi_stats))
for i, s in enumerate([k for k in kpi_stats if k in stat_cols_existing]):
    fs = form_score(logs[s], k=n_recent)
    delta = recent_mean[s] - season_mean[s]
    k_cols[i].metric(
        label=f"{s} (L{n_recent})",
        value=f"{recent_mean[s]:.1f}",
        delta=f"{delta:+.1f} vs SZN • Form {int(fs)}"
    )

# ---------------------- Trends ----------------------
st.markdown("### Recent Trends")
trend_cols = [c for c in ["GAME_DATE"] + kpi_stats if c in logs.columns]
trend_df = logs[trend_cols].head(15).sort_values("GAME_DATE")
for s in [c for c in kpi_stats if c in trend_df.columns]:
    chart = (
        alt.Chart(trend_df)
        .mark_line(point=True)
        .encode(x="GAME_DATE:T", y=alt.Y(s, title=s))
        .properties(height=150)
    )
    st.altair_chart(chart, use_container_width=True)

# ---------------------- Projection ----------------------
st.markdown("### Matchup-Adjusted Projection")
proj_stats = [s for s in ["PTS","REB","AST"] if s in stat_cols_existing]

rows = []
for s in proj_stats:
    r = float(recent_mean.get(s, np.nan))
    z = float(season_mean.get(s, np.nan))
    c = float(career_means.get(s, np.nan)) if s in career_means.index else np.nan
    base = blended_projection(r, z, c, weights=weights)
    proj = adj * base
    # simple uncertainty: use std of last max(8, n_recent) games
    sd = logs[s].head(max(8, n_recent)).std(ddof=1)
    rows.append({"Stat": s, "Recent": r, "Season": z, "Career": c, "AdjFactor": adj, "Projection": proj, "Std(LastN)": sd})

proj_df = pd.DataFrame(rows)
st.dataframe(
    proj_df.style.format({
        "Recent":"{:.2f}","Season":"{:.2f}","Career":"{:.2f}",
        "AdjFactor":"{:.3f}","Projection":"{:.2f}","Std(LastN)":"{:.2f}"
    }),
    use_container_width=True
)

# ---------------------- Box Score Explorer ----------------------
st.markdown("### Box Score — Last 12 Games")
cols_show = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","STL","BLK","FGM","FGA","FG_PCT","FG3M","FG3A","TOV","PLUS_MINUS"]
cols_show = [c for c in cols_show if c in logs.columns]
st.dataframe(logs[cols_show].head(12), use_container_width=True)

st.caption("Tip: adjust weights and recent window to stress-test different scenarios. Add position-specific opponent allowed tables later for more precise OA.")

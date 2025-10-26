# app.py — NBA Player Scouting Dashboard v2
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import datetime
import re

from nba_api.stats.static import players as static_players, teams as static_teams
from nba_api.stats.endpoints import (
    playergamelog,
    playercareerstats,
    leaguedashteamstats,
    LeagueDashPlayerStats,
    commonplayerinfo,
)

# ----------------------- Streamlit Setup -----------------------
st.set_page_config(page_title="NBA Player Scouting Dashboard", layout="wide")
st.title("NBA Player Scouting Dashboard")

# ----------------------- Config -----------------------
CACHE_HOURS = 12
LEAGUE_DEF_REF = 112.0
DEFAULT_SEASON = "2025-26"

# ----------------------- Utilities -----------------------
def is_nba_team_id(x):
    try:
        return str(int(x)).startswith("161061")
    except Exception:
        return False

def possessions_proxy_row(row):
    FGA  = pd.to_numeric(row.get("FGA", 0), errors="coerce")
    OREB = pd.to_numeric(row.get("OREB", 0), errors="coerce")
    TOV  = pd.to_numeric(row.get("TOV", 0), errors="coerce")
    FTA  = pd.to_numeric(row.get("FTA", 0), errors="coerce")
    return float(FGA - OREB + TOV + 0.44 * FTA)

def pace_proxy_row(row):
    poss = possessions_proxy_row(row)
    MIN  = pd.to_numeric(row.get("MIN", 48), errors="coerce")
    if not np.isfinite(MIN) or MIN <= 0:
        MIN = 48.0
    return float(poss * (48.0 / MIN))

def safe_div(a, b, default=np.nan):
    try:
        if not np.isfinite(b) or b == 0:
            return default
        val = a / b
        return val if np.isfinite(val) else default
    except Exception:
        return default

def _auto_height(df, row_px=34, header_px=38, max_px=900):
    rows = max(len(df), 1)
    return min(max_px, header_px + row_px * rows + 8)

def render_summary_table(df_indexed):
    styler = df_indexed.style.format("{:.2f}")
    h = _auto_height(df_indexed)
    st.dataframe(styler, use_container_width=True, height=h)

def opponent_adjustment(def_rating, pace_val, league_def, league_pace_mean):
    def_factor  = league_def / max(def_rating, 1e-9)
    pace_factor = pace_val / max(league_pace_mean, 1e-9)
    return float(def_factor * pace_factor)

# ----------------------- Season & Data Cache -----------------------
@st.cache_data(ttl=6*3600)
def detect_available_seasons(max_back_years=12):
    now = datetime.datetime.utcnow()
    start_years = [now.year - i for i in range(0, max_back_years)]
    def label(y): return f"{y}-{str((y+1)%100).zfill(2)}"
    candidates = sorted({label(y) for y in start_years}, reverse=True)
    available = []
    for season in candidates:
        try:
            df = LeagueDashPlayerStats(season=season, per_mode_detailed="PerGame").get_data_frames()[0]
            if not df.empty:
                available.append(season)
        except Exception:
            pass
    return available

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_active_players_df():
    # add rookies manually if not present yet in nba_api static data
    df = pd.DataFrame(static_players.get_active_players())
    rookies = [
        {"id": 999901, "full_name": "Cooper Flagg"},
        {"id": 999902, "full_name": "Dylan Harper"},
        {"id": 999903, "full_name": "Hugo Gonzalez"}
    ]
    for r in rookies:
        if not (df["full_name"].str.contains(r["full_name"]).any()):
            df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
    return df

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_teams_static_df():
    return pd.DataFrame(static_teams.get_teams())

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_team_context_advanced(season):
    """Use Advanced team stats for PACE, DEF_RATING, NET_RATING, etc."""
    df_adv = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame"
    ).get_data_frames()[0]
    df_adv = df_adv[df_adv["TEAM_ID"].apply(is_nba_team_id)].copy()
    cols_keep = [
        "TEAM_ID", "TEAM_NAME", "GP", "W_PCT", "PACE", "OFF_RATING", "DEF_RATING",
        "NET_RATING", "E_OFF_RATING", "E_DEF_RATING", "E_NET_RATING"
    ]
    for c in cols_keep:
        if c not in df_adv.columns:
            df_adv[c] = np.nan
    df_adv = df_adv[cols_keep]
    return df_adv, float(df_adv["PACE"].mean()), float(df_adv["DEF_RATING"].mean())

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_player_logs(player_id, season):
    df = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    if df.empty:
        return df
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_player_career(player_id):
    return playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_common_player_info(player_id):
    try:
        return commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

# ----------------------- Sidebar Filters -----------------------
players_df = get_active_players_df().sort_values("full_name")
teams_static_df = get_teams_static_df()
SEASONS = detect_available_seasons()
if DEFAULT_SEASON not in SEASONS:
    SEASONS = [DEFAULT_SEASON] + SEASONS

with st.sidebar:
    st.header("Filters")
    season = st.selectbox("Season", SEASONS, index=0)
    q = st.text_input("Search player", value="Jayson Tatum")
    filtered = players_df[players_df["full_name"].str.contains(q, case=False, na=False)]
    if filtered.empty:
        st.stop()
    player_name = st.selectbox("Player", filtered["full_name"].tolist())
    player_id = int(filtered.loc[filtered["full_name"] == player_name, "id"].iloc[0])
    team_adv, league_pace_mean, league_def_mean = get_team_context_advanced(season)
    opponent = st.selectbox("Opponent", team_adv["TEAM_NAME"].tolist())
    n_recent = st.selectbox("Window", ["Season", 5, 10, 15, 20], index=1)

# ----------------------- Data Fetch -----------------------
logs = get_player_logs(player_id, season)
career_df = get_player_career(player_id)
cpi = get_common_player_info(player_id)
opp_row = team_adv.loc[team_adv["TEAM_NAME"] == opponent].iloc[0]

# ----------------------- Player + Opponent Headers -----------------------
left, right = st.columns([2, 1])
with left:
    st.subheader(f"{player_name} — {season}")
    if not cpi.empty:
        pos = cpi.get("POSITION", pd.Series([None])).iloc[0] if "POSITION" in cpi.columns else None
        exp = cpi.get("SEASON_EXP", pd.Series([None])).iloc[0] if "SEASON_EXP" in cpi.columns else None
        team_name = cpi.get("TEAM_NAME", pd.Series(["—"])).iloc[0] if "TEAM_NAME" in cpi.columns else "—"
        gp = len(logs)
        st.caption(f"**Team:** {team_name} • **Position:** {pos} • **Seasons:** {exp} • **Games Played:** {gp}")

with right:
    st.markdown(f"**Opponent: {opponent}**")
    st.metric("DEF Rating", f"{opp_row['DEF_RATING']:.1f}")
    st.metric("Pace", f"{opp_row['PACE']:.1f}")
    st.metric("Net Rating", f"{opp_row['NET_RATING']:.1f}")

# ----------------------- Opponent Rank Table -----------------------
st.markdown("### Opponent Team Advanced Metrics")
opp_df_disp = team_adv[["TEAM_NAME", "W_PCT", "PACE", "OFF_RATING", "DEF_RATING", "NET_RATING"]].copy()
opp_df_disp["PACE_RANK"] = opp_df_disp["PACE"].rank(ascending=False)
opp_df_disp["DEF_RATING_RANK"] = opp_df_disp["DEF_RATING"].rank(ascending=True)
opp_df_disp["OFF_RATING_RANK"] = opp_df_disp["OFF_RATING"].rank(ascending=False)
st.dataframe(opp_df_disp.sort_values("DEF_RATING_RANK").reset_index(drop=True).style.format("{:.2f}"), use_container_width=True, height=_auto_height(opp_df_disp))

# ----------------------- Player Trends -----------------------
if logs.empty:
    st.error("No game logs found for this player/season.")
    st.stop()

st.markdown(f"### Recent Trends (Last {n_recent if n_recent!='Season' else 'Season'} Games)")
trend_cols = [c for c in ["MIN","PTS","REB","AST","PRA","FG3M"] if c in logs.columns]
trend_df = logs[["GAME_DATE"] + trend_cols].head(int(n_recent) if n_recent != "Season" else len(logs)).sort_values("GAME_DATE")
for s in trend_cols:
    chart = (
        alt.Chart(trend_df)
        .mark_line(point=True)
        .encode(x="GAME_DATE:T", y=alt.Y(s, title=s))
        .properties(height=160)
    )
    st.altair_chart(chart, use_container_width=True)

# ----------------------- Comparison Window Table -----------------------
st.markdown("### Compare Windows (Career / Season / L5 / L15)")
def avg(df, n):
    if n == "Season": return df.mean(numeric_only=True)
    return df.head(int(n)).mean(numeric_only=True)

kpi = ["PTS","REB","AST","MIN"]
vals = {
    "Career": {s: career_df[s].mean() if s in career_df.columns else np.nan for s in kpi},
    "Season": avg(logs[kpi], "Season"),
    "L5": avg(logs[kpi], 5),
    "L15": avg(logs[kpi], 15),
}
cmp_df = pd.DataFrame(vals).round(2)
st.dataframe(cmp_df.style.format("{:.2f}"), use_container_width=True, height=_auto_height(cmp_df))

# ----------------------- Last 5 Games -----------------------
st.markdown("### Last 5 Games")
cols = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","PRA","FG3M"]
last5 = logs[cols].head(5).copy()
st.dataframe(last5.style.format("{:.0f}"), use_container_width=True, height=_auto_height(last5))

# ----------------------- Notes -----------------------
st.caption("Notes: Dashboard combines player, opponent, and advanced team metrics (pace, efficiency, defense). New rookies included manually.")

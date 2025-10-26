# app.py — NBA Player Scouting Dashboard v2 (Stability-First)
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import datetime
import time
import re
from functools import lru_cache

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
st.title("🏀 NBA Player Scouting Dashboard")

# ----------------------- Config -----------------------
CACHE_HOURS = 12
DEFAULT_SEASON = "2025-26"
LEAGUE_DEF_REF = 112.0
REQUEST_TIMEOUT = 15          # seconds per NBA API call
MAX_RETRIES = 2               # simple retry for transient failures

# Static seasons list (stability-first). You can enable auto-detect via a button later.
def _season_labels(start=2015, end=2025):
    def lab(y): return f"{y}-{str((y+1)%100).zfill(2)}"
    return [lab(y) for y in range(end, start-1, -1)]
SEASONS = _season_labels(2015, 2025)

# ----------------------- Helpers -----------------------
def is_nba_team_id(x):
    try:
        s = str(int(x))
        return s.startswith("161061")
    except Exception:
        return False

def safe_div(a, b, default=np.nan):
    try:
        b = float(b)
        if not np.isfinite(b) or b == 0:
            return default
        val = float(a) / b
        return val if np.isfinite(val) else default
    except Exception:
        return default

def _auto_height(df, row_px=34, header_px=38, max_px=900):
    rows = max(len(df), 1)
    return min(max_px, header_px + row_px * rows + 8)

def numeric_format_map(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    return {c: "{:.2f}" for c in num_cols}

def _retry_api(callable_cls, kwargs, timeout=REQUEST_TIMEOUT, retries=MAX_RETRIES, sleep=0.8):
    """Minimal retry wrapper for nba_api endpoint classes."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            obj = callable_cls(timeout=timeout, **kwargs)
            return obj.get_data_frames()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(sleep * (attempt + 1))
            else:
                raise last_err

# ----------------------- Optional: Detect seasons (safe) -----------------------
@st.cache_data(ttl=6*3600, show_spinner=False)
def detect_available_seasons_safe(max_back_years=6):
    """Try a few recent seasons with a per-call timeout. Returns a list or the static fallback."""
    now = datetime.datetime.utcnow()
    start_years = [now.year - i for i in range(0, max_back_years)]
    def label(y): return f"{y}-{str((y+1)%100).zfill(2)}"
    candidates = sorted({label(y) for y in start_years}, reverse=True)
    available = []
    for season in candidates:
        try:
            frames = _retry_api(LeagueDashPlayerStats, {"season": season, "per_mode_detailed": "PerGame"})
            if frames and len(frames[0]) > 0:
                available.append(season)
        except Exception:
            # swallow and try next
            pass
    return available or SEASONS

# ----------------------- Data Caching -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_active_players_df():
    df = pd.DataFrame(static_players.get_active_players())
    # Add notable rookies manually (ids are placeholders for lookup-only)
    rookies = [
        {"id": 999901, "full_name": "Cooper Flagg"},
        {"id": 999902, "full_name": "Dylan Harper"},
        {"id": 999903, "full_name": "Hugo Gonzalez"},
    ]
    for r in rookies:
        if not df["full_name"].str.contains(r["full_name"], case=False, na=False).any():
            df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
    return df

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_teams_static_df():
    return pd.DataFrame(static_teams.get_teams())

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_team_context_advanced(season):
    """Safely fetch team advanced metrics with timeout+retry."""
    try:
        frames = _retry_api(
            leaguedashteamstats.LeagueDashTeamStats,
            {
                "season": season,
                "measure_type_detailed_defense": "Advanced",
                "per_mode_detailed": "PerGame"
            },
        )
        if not frames:
            return pd.DataFrame(), np.nan, np.nan
        df_adv = frames[0]
    except Exception:
        return pd.DataFrame(), np.nan, np.nan

    df_adv = df_adv[df_adv["TEAM_ID"].apply(is_nba_team_id)].copy()
    cols_keep = [
        "TEAM_ID", "TEAM_NAME", "GP", "W_PCT", "PACE", "OFF_RATING", "DEF_RATING", "NET_RATING"
    ]
    for c in cols_keep:
        if c not in df_adv.columns:
            df_adv[c] = np.nan
    league_pace = float(df_adv["PACE"].mean()) if "PACE" in df_adv.columns else np.nan
    league_def = float(df_adv["DEF_RATING"].mean()) if "DEF_RATING" in df_adv.columns else np.nan
    return df_adv[cols_keep], league_pace, league_def

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_player_logs(player_id, season):
    try:
        frames = _retry_api(playergamelog.PlayerGameLog, {"player_id": player_id, "season": season})
        if not frames:
            return pd.DataFrame()
        df = frames[0]
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    for c in ("PTS", "REB", "AST"):
        if c not in df.columns:
            df[c] = 0
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    if "FG3M" not in df.columns:
        df["FG3M"] = np.nan
    return df

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_player_career(player_id):
    try:
        frames = _retry_api(playercareerstats.PlayerCareerStats, {"player_id": player_id})
        return frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_common_player_info(player_id):
    try:
        frames = _retry_api(commonplayerinfo.CommonPlayerInfo, {"player_id": player_id})
        return frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ----------------------- Sidebar -----------------------
players_df = get_active_players_df().sort_values("full_name")
teams_static_df = get_teams_static_df()

with st.sidebar:
    st.header("Filters")
    # Optional: safer season detection
    cols = st.columns([1, 1])
    use_detect = cols[0].checkbox("Auto-detect seasons (slower)", value=False)
    if use_detect:
        detected = detect_available_seasons_safe()
        season = st.selectbox("Season", detected, index=0)
    else:
        season = st.selectbox("Season", SEASONS, index=0)

    q = st.text_input("Search player", value="Jayson Tatum").strip()
    filtered = players_df[players_df["full_name"].str.contains(q, case=False, na=False)] if q else players_df
    if filtered.empty:
        st.info("No players match your search.")
        st.stop()
    player_name = st.selectbox("Player", filtered["full_name"].tolist())
    player_id = int(filtered.loc[filtered["full_name"] == player_name, "id"].iloc[0])

    n_recent = st.selectbox("Recent window", ["Season", 5, 10, 15, 20], index=1)

    st.divider()
    go = st.button("Load Data", type="primary")

# Short-circuit: render UI first, only fetch when user clicks
if not go:
    st.caption("👆 Select your filters and click **Load Data** to fetch from NBA Stats (with timeouts & retries).")
    st.stop()

# ----------------------- Fetch Data (after click) -----------------------
with st.spinner("Fetching team context..."):
    team_adv, league_pace_mean, league_def_mean = get_team_context_advanced(season)

if team_adv.empty:
    st.error("Unable to load team context data (NBA Stats). Try another season or uncheck auto-detect.")
    st.stop()

opponent = st.selectbox("Opponent", team_adv["TEAM_NAME"].tolist())

with st.spinner("Fetching player data..."):
    logs = get_player_logs(player_id, season)
    career_df = get_player_career(player_id)
    cpi = get_common_player_info(player_id)

if logs.empty:
    st.error("No game logs found for this player/season.")
    st.stop()

if opponent not in team_adv["TEAM_NAME"].values:
    st.error("Opponent not found in team context data.")
    st.stop()

opp_row = team_adv.loc[team_adv["TEAM_NAME"] == opponent].iloc[0]

# ----------------------- Header Section -----------------------
left, right = st.columns([2, 1])
with left:
    st.subheader(f"{player_name} — {season}")
    team_name = (cpi["TEAM_NAME"].iloc[0] if ("TEAM_NAME" in cpi.columns and not cpi.empty) else "Unknown")
    pos = (cpi["POSITION"].iloc[0] if ("POSITION" in cpi.columns and not cpi.empty) else "N/A")
    exp = (cpi["SEASON_EXP"].iloc[0] if ("SEASON_EXP" in cpi.columns and not cpi.empty) else "N/A")
    gp = len(logs)
    st.caption(f"**Team:** {team_name} • **Position:** {pos} • **Seasons:** {exp} • **Games Played:** {gp}")

with right:
    st.markdown(f"**Opponent:** {opponent}")
    c1, c2, c3 = st.columns(3)
    def _fmt(v): 
        try: return f"{float(v):.1f}"
        except: return "—"
    c1.metric("DEF Rating", _fmt(opp_row.get("DEF_RATING", np.nan)))
    c2.metric("PACE", _fmt(opp_row.get("PACE", np.nan)))
    c3.metric("NET Rating", _fmt(opp_row.get("NET_RATING", np.nan)))

# ----------------------- Opponent Table -----------------------
st.markdown("### Opponent Team Advanced Metrics")
opp_df_disp = team_adv[["TEAM_NAME", "W_PCT", "PACE", "OFF_RATING", "DEF_RATING", "NET_RATING"]].copy()
for c in ["PACE", "DEF_RATING", "OFF_RATING", "W_PCT", "NET_RATING"]:
    if c not in opp_df_disp.columns:
        opp_df_disp[c] = np.nan

opp_df_disp["PACE_RANK"] = opp_df_disp["PACE"].rank(ascending=False)
opp_df_disp["DEF_RATING_RANK"] = opp_df_disp["DEF_RATING"].rank(ascending=True)
opp_df_disp["OFF_RATING_RANK"] = opp_df_disp["OFF_RATING"].rank(ascending=False)
opp_df_disp = opp_df_disp.sort_values("DEF_RATING_RANK").reset_index(drop=True)

fmt_map = numeric_format_map(opp_df_disp)
st.dataframe(
    opp_df_disp.style.format(fmt_map),
    use_container_width=True,
    height=_auto_height(opp_df_disp)
)

# ----------------------- Recent Trends -----------------------
st.markdown(f"### Recent Trends (Last {n_recent if n_recent!='Season' else 'Season'} Games)")
trend_cols = [c for c in ["MIN","PTS","REB","AST","PRA","FG3M"] if c in logs.columns]
trend_df = logs[["GAME_DATE"] + trend_cols].head(int(n_recent) if n_recent != "Season" else len(logs)).copy()
trend_df = trend_df.sort_values("GAME_DATE")

if "GAME_DATE" in trend_df.columns and len(trend_cols) > 0 and len(trend_df) > 0:
    for s in trend_cols:
        chart = (
            alt.Chart(trend_df)
            .mark_line(point=True)
            .encode(x="GAME_DATE:T", y=alt.Y(s, title=s))
            .properties(height=160)
        )
        st.altair_chart(chart, use_container_width=True)
else:
    st.info("No trend data available to chart.")

# ----------------------- Comparison Windows -----------------------
st.markdown("### Compare Windows (Career / Season / L5 / L15)")
def avg(df, n):
    if df.empty:
        return pd.Series(dtype=float)
    if n == "Season":
        return df.mean(numeric_only=True)
    return df.head(int(n)).mean(numeric_only=True)

kpi = [c for c in ["PTS","REB","AST","MIN"] if c in logs.columns or c in career_df.columns]
vals = {
    "Career": {s: (career_df[s].mean() if s in career_df.columns else np.nan) for s in kpi},
    "Season": avg(logs[kpi], "Season") if set(kpi).issubset(logs.columns) else pd.Series({s: np.nan for s in kpi}),
    "L5": avg(logs[kpi], 5) if set(kpi).issubset(logs.columns) else pd.Series({s: np.nan for s in kpi}),
    "L15": avg(logs[kpi], 15) if set(kpi).issubset(logs.columns) else pd.Series({s: np.nan for s in kpi}),
}
cmp_df = pd.DataFrame(vals).round(2)
st.dataframe(cmp_df.style.format(numeric_format_map(cmp_df)), use_container_width=True, height=_auto_height(cmp_df))

# ----------------------- Last 5 Games -----------------------
st.markdown("### Last 5 Games")
cols = [c for c in ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","PRA","FG3M"] if c in logs.columns]
last5 = logs[cols].head(5).copy()
num_fmt = {c: "{:.0f}" for c in last5.select_dtypes(include=[np.number]).columns}
st.dataframe(last5.style.format(num_fmt), use_container_width=True, height=_auto_height(last5))

# ----------------------- Footer -----------------------
st.caption("Notes: Stability-first build — per-call timeouts, retries, and manual data load. Data from NBA Stats API.")

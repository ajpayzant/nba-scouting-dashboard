# app.py — NBA Player Scouting Dashboard v2 (UX update per request)
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import datetime
import time

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
CACHE_HOURS   = 12
REQUEST_TIMEOUT = 15
MAX_RETRIES     = 2

def _retry_api(endpoint_cls, kwargs, timeout=REQUEST_TIMEOUT, retries=MAX_RETRIES, sleep=0.8):
    last_err = None
    for i in range(retries + 1):
        try:
            obj = endpoint_cls(timeout=timeout, **kwargs)
            return obj.get_data_frames()
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(sleep * (i + 1))
    raise last_err

def _season_labels(start=2010, end=None):
    if end is None:
        end = datetime.datetime.utcnow().year
    def lab(y): return f"{y}-{str((y+1)%100).zfill(2)}"
    return [lab(y) for y in range(end, start-1, -1)]

SEASONS = _season_labels(2015, datetime.datetime.utcnow().year)

# ----------------------- Utils -----------------------
def numeric_format_map(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    return {c: "{:.2f}" for c in num_cols}

def _auto_height(df, row_px=34, header_px=38, max_px=900):
    rows = max(len(df), 1)
    return min(max_px, header_px + row_px * rows + 8)

def is_nba_team_id(x):
    try:
        return str(int(x)).startswith("161061")
    except Exception:
        return False

def _fmt1(v):
    try: return f"{float(v):.1f}"
    except: return "—"

def parse_opponent_abbrev_from_matchup(matchup_str, player_team_abbrev):
    """
    MATCHUP examples: 'BOS vs LAL', 'BOS @ MIA' → returns 'LAL' / 'MIA'.
    """
    if not isinstance(matchup_str, str):
        return None
    parts = matchup_str.split()
    if len(parts) >= 3:
        opp = parts[-1]
        if opp != player_team_abbrev:
            return opp
    return None

def add_shot_breakouts(df):
    """Ensure columns: MIN PTS REB AST PRA 2PM 2PA 3PM 3PA FTM FTA OREB DREB"""
    for col in ["MIN","PTS","REB","AST","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB","GAME_DATE","MATCHUP","WL"]:
        if col not in df.columns:
            df[col] = 0
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df["2PM"] = df["FGM"] - df["FG3M"]
    df["2PA"] = df["FGA"] - df["FG3A"]
    keep_cols = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","PRA","2PM","2PA","FG3M","FG3A","FTM","FTA","OREB","DREB"]
    existing = [c for c in keep_cols if c in df.columns]
    return df[existing]

# ----------------------- Cached data -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_active_players_df():
    df = pd.DataFrame(static_players.get_active_players())
    return df[["id","full_name"]].copy()

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_teams_static_df():
    return pd.DataFrame(static_teams.get_teams())

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_team_context_advanced(season):
    try:
        frames = _retry_api(
            leaguedashteamstats.LeagueDashTeamStats,
            {"season": season, "measure_type_detailed_defense": "Advanced", "per_mode_detailed": "PerGame"}
        )
        if not frames:
            return pd.DataFrame(), np.nan, np.nan
        df_adv = frames[0]
    except Exception:
        return pd.DataFrame(), np.nan, np.nan

    df_adv = df_adv[df_adv["TEAM_ID"].apply(is_nba_team_id)].copy()
    cols_keep = ["TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION","W_PCT","PACE","OFF_RATING","DEF_RATING","NET_RATING"]
    for c in cols_keep:
        if c not in df_adv.columns:
            df_adv[c] = np.nan

    # Compute league ranks (1 = best)
    df_adv["DEF_RANK"] = df_adv["DEF_RATING"].rank(ascending=True, method="min")
    df_adv["PACE_RANK"] = df_adv["PACE"].rank(ascending=False, method="min")
    df_adv["NET_RANK"]  = df_adv["NET_RATING"].rank(ascending=False, method="min")
    for c in ["DEF_RANK","PACE_RANK","NET_RANK"]:
        df_adv[c] = df_adv[c].astype("Int64")

    league_pace = float(df_adv["PACE"].mean()) if "PACE" in df_adv.columns else np.nan
    league_def  = float(df_adv["DEF_RATING"].mean()) if "DEF_RATING" in df_adv.columns else np.nan
    return df_adv[cols_keep + ["DEF_RANK","PACE_RANK","NET_RANK"]], league_pace, league_def

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_season_player_index(season):
    """LeagueDashPlayerStats to map PLAYER_ID → team, enabling Team → Player filter."""
    try:
        frames = _retry_api(LeagueDashPlayerStats, {"season": season, "per_mode_detailed": "PerGame"})
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    keep = ["PLAYER_ID","PLAYER_NAME","TEAM_ID","TEAM_ABBREVIATION","TEAM_NAME","GP","MIN"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].drop_duplicates(subset=["PLAYER_ID"]).reset_index(drop=True)
    return df.sort_values(["TEAM_NAME","PLAYER_NAME"])

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_player_logs(player_id, season):
    try:
        frames = _retry_api(playergamelog.PlayerGameLog, {"player_id": player_id, "season": season})
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

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

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_all_season_logs(player_id, season_ids):
    """Fetch logs for ALL seasons in season_ids and concatenate."""
    frames = []
    for sid in season_ids:
        df = get_player_logs(player_id, sid)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "GAME_DATE" in out.columns:
        out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")
    return out.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

# ----------------------- Sidebar (filters first, Load at bottom) -----------------------
teams_static = get_teams_static_df()
team_name_to_abbrev = dict(zip(teams_static["full_name"], teams_static["abbreviation"]))

with st.sidebar:
    st.header("Filters")
    season = st.selectbox("Season", SEASONS, index=0)

    # Build team list for the selected season (later after team_adv is available we re-use it)
    st.caption("Select Team → (optional) Search → Player → Opponent. Then click **Load Data** below.")

# We need team_adv to fill teams; load small context once, but keep heavy player logs gated by button.
team_adv, league_pace_mean, league_def_mean = get_team_context_advanced(season)
if team_adv.empty:
    st.error("Unable to load league/team context for this season.")
    st.stop()

with st.sidebar:
    team_list = team_adv["TEAM_NAME"].sort_values().tolist()
    sel_team = st.selectbox("Team", ["(All teams)"] + team_list, index=0)

    season_players = get_season_player_index(season)
    q = st.text_input("Search player").strip()

    filtered_players = season_players.copy()
    if sel_team != "(All teams)":
        filtered_players = filtered_players[filtered_players["TEAM_NAME"] == sel_team]
    if q:
        filtered_players = filtered_players[filtered_players["PLAYER_NAME"].str.contains(q, case=False, na=False)]

    if filtered_players.empty:
        st.info("No players match your filters.")
        st.stop()

    player_name = st.selectbox("Player", filtered_players["PLAYER_NAME"].tolist())
    player_row = filtered_players[filtered_players["PLAYER_NAME"] == player_name].iloc[0]
    player_id  = int(player_row["PLAYER_ID"])
    player_team_abbrev = str(player_row.get("TEAM_ABBREVIATION", ""))

    opponent = st.selectbox("Opponent", team_list, index=0)
    n_recent = st.selectbox("Recent window", ["Season", 5, 10, 15, 20], index=1)

    st.divider()
    go = st.button("Load Data", type="primary")  # <-- at bottom as requested

# Basic state to avoid wiping selections unnecessarily
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if go:
    st.session_state.loaded = True
if not st.session_state.loaded:
    st.caption("➡

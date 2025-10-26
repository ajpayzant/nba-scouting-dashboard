# app.py — Updated with rookie support + advanced defense metrics + smarter opponent adjustment

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
LEAGUE_DEF_REF = 112.0  # coarse league baseline for defense scaling
DEFAULT_SEASON = "2025-26"

# ----------------------- Utilities -----------------------
def is_nba_team_id(x) -> bool:
    try:
        return str(int(x)).startswith("161061")
    except Exception:
        return False

def possessions_proxy_row(row: pd.Series) -> float:
    FGA  = pd.to_numeric(row.get("FGA", 0), errors="coerce")
    OREB = pd.to_numeric(row.get("OREB", 0), errors="coerce")
    TOV  = pd.to_numeric(row.get("TOV", 0), errors="coerce")
    FTA  = pd.to_numeric(row.get("FTA", 0), errors="coerce")
    return float(FGA - OREB + TOV + 0.44 * FTA)

def pace_proxy_row(row: pd.Series) -> float:
    poss = possessions_proxy_row(row)
    MIN  = pd.to_numeric(row.get("MIN", 48), errors="coerce")
    if not np.isfinite(MIN) or MIN <= 0:
        MIN = 48.0
    return float(poss * (48.0 / MIN))

def safe_div(a, b, default=np.nan):
    try:
        if b in (0, None, np.nan):
            return default
        val = a / b
        return val if np.isfinite(val) else default
    except Exception:
        return default

def form_score(series: pd.Series, k: int = 5) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 50.0
    recent = s.iloc[:min(k, len(s))]
    mu_r, mu_s, sd = recent.mean(), s.mean(), s.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return 50.0
    z = (mu_r - mu_s) / sd
    return float(np.clip(50 + 15*z, 0, 100))

def opponent_adjustment(def_rating: float, pace_val: float,
                        league_def: float, league_pace_mean: float) -> float:
    def_factor  = league_def / max(def_rating, 1e-9)
    pace_factor = pace_val / max(league_pace_mean, 1e-9)
    return float(def_factor * pace_factor)

def window_avg(df: pd.DataFrame, n: int, cols) -> pd.Series:
    if len(df) < 1:
        return pd.Series({c: np.nan for c in cols})
    return df.iloc[:min(len(df), n)][cols].mean(numeric_only=True)

def per_min(series_num, series_min, n=None):
    num = pd.to_numeric(series_num, errors="coerce")
    den = pd.to_numeric(series_min, errors="coerce")
    if n is not None:
        num, den = num.head(n), den.head(n)
    val = (num / den).replace([np.inf, -np.inf], np.nan)
    return float(val.mean()) if len(val.dropna()) else np.nan

def pm_ratio(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    val = (num / den).replace([np.inf, -np.inf], np.nan)
    return float(val.mean()) if len(val.dropna()) else np.nan

def career_pg_counting_stat(career_df: pd.DataFrame, col: str) -> float:
    if career_df.empty or col not in career_df.columns or "GP" not in career_df.columns:
        return np.nan
    s_tot = pd.to_numeric(career_df[col], errors="coerce")
    s_gp  = pd.to_numeric(career_df["GP"],  errors="coerce")
    mask = s_tot.notna() & s_gp.notna() & (s_gp > 0)
    if not mask.any():
        return np.nan
    return float(s_tot[mask].sum() / s_gp[mask].sum())

def _auto_height(df: pd.DataFrame, row_px: int = 34, header_px: int = 38, max_px: int = 900) -> int:
    rows = max(len(df), 1)
    return min(max_px, header_px + row_px * rows + 8)

def render_summary_table(df_indexed: pd.DataFrame):
    num_cols = df_indexed.select_dtypes(include=[np.number]).columns
    fmt_map = {c: "{:.2f}" for c in num_cols}
    h = _auto_height(df_indexed)
    st.dataframe(df_indexed.style.format(fmt_map), use_container_width=True, height=h)

# ----------------------- Season detection & caching -----------------------
@st.cache_data(ttl=6*3600)
def detect_available_seasons(max_back_years: int = 12) -> list[str]:
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

# ----------------------- Improved: Active players + rookies -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600)
def get_active_players_df():
    df = pd.DataFrame(static_players.get_active_players())
    rookies = [
        {"id": 999901, "full_name": "Cooper Flagg"},
        {"id": 999902, "full_name": "Dylan Harper"},
        {"id": 999903, "full_name": "Hugo Gonzalez"},
    ]
    for r in rookies:
        if not df["full_name"].str.contains(r["full_name"], case=False, na=False).any():
            df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
    return df

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_teams_static_df():
    return pd.DataFrame(static_teams.get_teams())

# ----------------------- Improved: Team metrics with Advanced + Opponent -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600)
def get_team_tables(season: str):
    df_def_all = leaguedashteamstats.LeagueDashTeamStats(
        season=season, per_mode_detailed="PerGame", measure_type_detailed_defense="Defense"
    ).get_data_frames()[0]

    df_base_all = leaguedashteamstats.LeagueDashTeamStats(
        season=season, per_mode_detailed="PerGame"
    ).get_data_frames()[0]

    try:
        df_adv_all = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame"
        ).get_data_frames()[0]
    except Exception:
        df_adv_all = pd.DataFrame()

    try:
        df_opp_all = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed_defense="Opponent", per_mode_detailed="PerGame"
        ).get_data_frames()[0]
    except Exception:
        df_opp_all = pd.DataFrame()

    def _f(df): return df[df["TEAM_ID"].apply(is_nba_team_id)].copy() if not df.empty else df
    df_def, df_base, df_adv, df_opp = map(_f, [df_def_all, df_base_all, df_adv_all, df_opp_all])

    keep_def  = ["TEAM_ID","TEAM_NAME","DEF_RATING","DREB_PCT"]
    keep_base = ["TEAM_ID","TEAM_NAME","MIN","FGA","FTA","OREB","TOV"]
    keep_adv  = ["TEAM_ID","TEAM_NAME","PACE","NET_RATING","OFF_RATING","DEF_RATING"]
    keep_opp  = ["TEAM_ID","TEAM_NAME","OPP_PTS","OPP_REB","OPP_AST","OPP_FG3M","OPP_FG3A"]

    for frame, keep in zip([df_def, df_base, df_adv, df_opp], [keep_def, keep_base, keep_adv, keep_opp]):
        for c in keep:
            if c not in frame.columns:
                frame[c] = np.nan

    df_base["PACE_PROXY"] = df_base.apply(pace_proxy_row, axis=1)
    league_pace_mean = float(df_base["PACE_PROXY"].mean())
    league_dreb_pct_mean = float(df_def["DREB_PCT"].mean())
    league_oreb_mean = float(df_base["OREB"].mean())

    teams_ctx = (
        df_def.merge(df_base[["TEAM_ID","PACE_PROXY","OREB"]], on="TEAM_ID", how="left")
              .merge(df_adv, on=["TEAM_ID","TEAM_NAME"], how="left", suffixes=("","_ADV"))
              .merge(df_opp, on=["TEAM_ID","TEAM_NAME"], how="left", suffixes=("","_OPP"))
              .sort_values("TEAM_NAME")
              .reset_index(drop=True)
    )
    return teams_ctx, league_pace_mean, league_dreb_pct_mean, league_oreb_mean

# ----------------------- Remaining data functions (unchanged) -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600)
def get_player_logs(player_id: int, season: str):
    df = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    if df.empty:
        return df
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    df["FG2M"] = pd.to_numeric(df.get("FGM", 0), errors="coerce") - pd.to_numeric(df.get("FG3M", 0), errors="coerce")
    df["FG2A"] = pd.to_numeric(df.get("FGA", 0), errors="coerce") - pd.to_numeric(df.get("FG3A", 0), errors="coerce")
    df["FG2M"] = df["FG2M"].clip(lower=0)
    df["FG2A"] = df["FG2A"].clip(lower=0)
    for col in ["PTS","REB","AST","FG3M","FG3A","OREB","DREB","MIN"]:
        if col not in df.columns:
            df[col] = np.nan
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_player_career(player_id: int):
    return playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_common_player_info(player_id: int):
    try:
        return commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_HOURS*3600)
def last_n_vs_opponent(player_id: int, opp_abbr: str, seasons_list: list[str], n: int = 5) -> pd.DataFrame:
    frames = []
    for s in seasons_list:
        df = get_player_logs(player_id, s)
        if df.empty:
            continue
        mask = df["MATCHUP"].str.contains(rf"\b{re.escape(opp_abbr)}\b", na=False, regex=True)
        part = df.loc[mask].copy()
        if not part.empty:
            part["SEASON"] = s
            frames.append(part)
        if sum(len(f) for f in frames) >= n:
            break
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("GAME_DATE", ascending=False).head(n)
    for col in ["PTS","REB","AST"]:
        if col not in all_df.columns:
            all_df[col] = np.nan
    all_df["PRA"] = all_df["PTS"] + all_df["REB"] + all_df["AST"]
    return all_df

# ----------------------- Sidebar + core logic (unchanged except adjustment block) -----------------------
players_df = get_active_players_df().sort_values("full_name")
teams_static_df = get_teams_static_df()
SEASONS = detect_available_seasons(max_back_years=12)
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
    teams_ctx, league_pace_mean, league_dreb_pct_mean, league_oreb_mean = get_team_tables(season)
    opponent = st.selectbox("Opponent", teams_ctx["TEAM_NAME"].tolist())
    window_options = ["Season", 5, 10, 15, 20, 25]
    n_recent = st.selectbox("Recent window (games)", window_options, index=1)
    w_recent = st.slider("Weight: Recent", 0.0, 1.0, 0.55, 0.05)
    w_season = st.slider("Weight: Season", 0.0, 1.0, 0.30, 0.05)
    w_career = st.slider("Weight: Career", 0.0, 1.0, 0.15, 0.05)

season_used = season
logs = get_player_logs(player_id, season_used)
if logs.empty:
    st.error("No game logs found for this player/season.")
    st.stop()
career_raw = get_player_career(player_id)
cpi = get_common_player_info(player_id)

# ----------------------- Opponent Context -----------------------
opp_row = teams_ctx.loc[teams_ctx["TEAM_NAME"] == opponent].iloc[0]
opp_def = float(opp_row["DEF_RATING"])
opp_pace = float(opp_row["PACE_PROXY"])
opp_dreb_pct = float(opp_row["DREB_PCT"])
opp_oreb = float(opp_row["OREB"])
opp_pts_allowed = float(opp_row.get("OPP_PTS", np.nan))
opp_reb_allowed = float(opp_row.get("OPP_REB", np.nan))

pts_factor = 1.0
reb_factor = 1.0
if np.isfinite(opp_pts_allowed):
    pts_factor = (112.0 / opp_pts_allowed)
if np.isfinite(opp_reb_allowed):
    reb_factor = (44.0 / opp_reb_allowed)

AdjFactor = opponent_adjustment(opp_def, opp_pace, LEAGUE_DEF_REF, league_pace_mean) * np.sqrt(pts_factor * reb_factor)
ORB_adj = (league_dreb_pct_mean / opp_dreb_pct) if np.isfinite(league_dreb_pct_mean) and np.isfinite(opp_dreb_pct) and opp_dreb_pct > 0 else 1.0
DRB_adj = (league_oreb_mean / opp_oreb)       if np.isfinite(league_oreb_mean)     and np.isfinite(opp_oreb)     and opp_oreb > 0         else 1.0

# (Rest of your code unchanged — projections, charts, tables, metrics, etc.)
# -------------------------------------------------------------------------
# Copy your original logic for projections and Streamlit rendering here.
# -------------------------------------------------------------------------

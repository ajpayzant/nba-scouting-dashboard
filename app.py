# app.py — Original structure preserved with 3 targeted upgrades:
# (1) Rookie inclusion in player list
# (2) Advanced + Opponent team context
# (3) Smarter opponent adjustment (uses OPP_PTS/REB gently)

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
DEFAULT_SEASON = "2025-26"  # requested default

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

# Helper for clean numeric-only styling in tables
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

# ----------------------- (1) Improved: Active players + rookies -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600)
def get_active_players_df():
    df = pd.DataFrame(static_players.get_active_players())
    # Add rookie placeholders if not present in static list yet
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

# ----------------------- (2) Improved: Team metrics with Advanced + Opponent -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600)
def get_team_tables(season: str):
    # Defense slice (DEF_RATING / DREB_PCT – same as your original)
    df_def_all = leaguedashteamstats.LeagueDashTeamStats(
        season=season, per_mode_detailed="PerGame", measure_type_detailed_defense="Defense"
    ).get_data_frames()[0]
    # Base slice for FGA/FTA/etc. to build a Pace proxy and OREB baseline
    df_base_all = leaguedashteamstats.LeagueDashTeamStats(
        season=season, per_mode_detailed="PerGame"
    ).get_data_frames()[0]

    # Advanced slice: PACE, OFF/DEF/NET_RATING (best-effort)
    try:
        df_adv_all = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame"
        ).get_data_frames()[0]
    except Exception:
        df_adv_all = pd.DataFrame()

    # Opponent slice: OPP_* box-counting allowed (best-effort)
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

    # Pace proxy & league means (unchanged)
    df_base["PACE_PROXY"] = df_base.apply(pace_proxy_row, axis=1)
    league_pace_mean = float(df_base["PACE_PROXY"].mean())
    league_dreb_pct_mean = float(df_def["DREB_PCT"].mean()) if "DREB_PCT" in df_def.columns else np.nan
    league_oreb_mean = float(df_base["OREB"].mean())

    teams_ctx = (
        df_def.merge(df_base[["TEAM_ID","PACE_PROXY","OREB"]], on="TEAM_ID", how="left")
              .merge(df_adv, on=["TEAM_ID","TEAM_NAME"], how="left", suffixes=("","_ADV"))
              .merge(df_opp, on=["TEAM_ID","TEAM_NAME"], how="left", suffixes=("","_OPP"))
              .sort_values("TEAM_NAME")
              .reset_index(drop=True)
    )
    return teams_ctx, league_pace_mean, league_dreb_pct_mean, league_oreb_mean

# ----------------------- Data fetchers (unchanged) -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600)
def get_player_logs(player_id: int, season: str):
    df = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    if df.empty:
        return df
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    # derive 2P components
    df["FG2M"] = pd.to_numeric(df.get("FGM", 0), errors="coerce") - pd.to_numeric(df.get("FG3M", 0), errors="coerce")
    df["FG2A"] = pd.to_numeric(df.get("FGA", 0), errors="coerce") - pd.to_numeric(df.get("FG3A", 0), errors="coerce")
    df["FG2M"] = df["FG2M"].clip(lower=0)
    df["FG2A"] = df["FG2A"].clip(lower=0)
    # ensure columns & derive PRA
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

def best_season_for_player(player_id: int, preferred: str, season_pool: list[str]) -> str:
    logs = get_player_logs(player_id, preferred)
    if not logs.empty:
        return preferred
    for s in season_pool:
        df = get_player_logs(player_id, s)
        if not df.empty:
            return s
    return preferred

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

# ----------------------- Sidebar Controls -----------------------
players_df = get_active_players_df().sort_values("full_name")
teams_static_df = get_teams_static_df()

SEASONS = detect_available_seasons(max_back_years=12)
if DEFAULT_SEASON not in SEASONS:
    SEASONS = [DEFAULT_SEASON] + SEASONS

with st.sidebar:
    st.header("Filters")
    season = st.selectbox("Season", SEASONS, index=0)  # default = 2025-26
    q = st.text_input("Search player", value="Jayson Tatum")
    filtered = players_df[players_df["full_name"].str.contains(q, case=False, na=False)]
    if filtered.empty:
        st.stop()
    player_name = st.selectbox("Player", filtered["full_name"].tolist())
    player_id = int(filtered.loc[filtered["full_name"] == player_name, "id"].iloc[0])

    teams_ctx, league_pace_mean, league_dreb_pct_mean, league_oreb_mean = get_team_tables(season)
    opponent = st.selectbox("Opponent", teams_ctx["TEAM_NAME"].tolist())

    # Add "Season" option so charts can scale to full season window
    window_options = ["Season", 5, 10, 15, 20, 25]
    n_recent = st.selectbox("Recent window (games)", window_options, index=1 if 5 in window_options else 0)

    w_recent = st.slider("Weight: Recent", 0.0, 1.0, 0.55, 0.05)
    w_season = st.slider("Weight: Season", 0.0, 1.0, 0.30, 0.05)
    w_career = st.slider("Weight: Career", 0.0, 1.0, 0.15, 0.05)

# ----------------------- Data fetch -----------------------
season_used = best_season_for_player(player_id, season, SEASONS)
logs = get_player_logs(player_id, season_used)
if logs.empty:
    st.error("No game logs found for this player/season.")
    st.stop()
career_raw = get_player_career(player_id)
cpi = get_common_player_info(player_id)

# ----------------------- Opponent Context -----------------------
opp_row = teams_ctx.loc[teams_ctx["TEAM_NAME"] == opponent].iloc[0]
opp_def      = float(opp_row["DEF_RATING"])
opp_pace     = float(opp_row["PACE_PROXY"])
opp_dreb_pct = float(opp_row["DREB_PCT"]) if "DREB_PCT" in teams_ctx.columns else np.nan
opp_oreb     = float(opp_row["OREB"])
# (3) Gentle scaling using opponent allowed points/rebounds if available
opp_pts_allowed = float(opp_row.get("OPP_PTS", np.nan))
opp_reb_allowed = float(opp_row.get("OPP_REB", np.nan))

pts_factor = 1.0
reb_factor = 1.0
if np.isfinite(opp_pts_allowed):
    # scale to a coarse league baseline ~112
    pts_factor = (112.0 / max(opp_pts_allowed, 1e-9))
if np.isfinite(opp_reb_allowed):
    # scale to a coarse baseline for team rebounds allowed ~44
    reb_factor = (44.0 / max(opp_reb_allowed, 1e-9))

AdjFactor = opponent_adjustment(opp_def, opp_pace, LEAGUE_DEF_REF, league_pace_mean) * float(np.sqrt(pts_factor * reb_factor))
ORB_adj = (league_dreb_pct_mean / opp_dreb_pct) if np.isfinite(league_dreb_pct_mean) and np.isfinite(opp_dreb_pct) and opp_dreb_pct > 0 else 1.0
DRB_adj = (league_oreb_mean / opp_oreb)       if np.isfinite(league_oreb_mean)     and np.isfinite(opp_oreb)     and opp_oreb > 0         else 1.0

# ----------------------- Player header (Age • Position • Seasons • GP) -----------------------
left, right = st.columns([2, 1])
with left:
    st.subheader(f"{player_name} — {season_used}")
    age = pos = exp = team_name = None
    if not cpi.empty:
        age = cpi.get("AGE", pd.Series([None])).iloc[0] if "AGE" in cpi.columns else None
        pos = cpi.get("POSITION", pd.Series([None])).iloc[0] if "POSITION" in cpi.columns else None
        exp = cpi.get("SEASON_EXP", pd.Series([None])).iloc[0] if "SEASON_EXP" in cpi.columns else None
        team_name = cpi.get("TEAM_NAME", pd.Series([None])).iloc[0] if "TEAM_NAME" in cpi.columns else None
    gp_this_season = int(len(logs))
    meta = []
    if team_name:
        meta.append(f"Team: {team_name}")
    if age is not None and str(age) != "nan":
        meta.append(f"Age: {int(float(age))}")
    if pos:
        meta.append(f"Position: {pos}")
    if exp is not None and str(exp) != "nan":
        meta.append(f"Seasons: {int(float(exp))}")
    meta.append(f"GP ({season_used}): {gp_this_season}")
    st.caption(" • ".join(meta))

with right:
    st.markdown(f"**Opponent: {opponent}**")
    c1, c2 = st.columns(2)
    c1.metric("DEF Rating", f"{opp_def:.1f}")
    c2.metric("Pace (proxy)", f"{opp_pace:.1f}")

# ----------------------- Baselines & blends (unchanged) -----------------------
def mean_recent(series, n=n_recent):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if n == "Season":
        return float(s.mean()) if len(s) else np.nan
    return float(s.iloc[:min(int(n), len(s))].mean()) if len(s) else np.nan

def season_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan

MIN_recent = mean_recent(logs["MIN"], n_recent)
MIN_season = season_mean(logs["MIN"])
MIN_career_pg = career_pg_counting_stat(career_raw, "MIN")
# Basic minutes projection: prefer recent if available, else season
MIN_proj = float(MIN_recent if np.isfinite(MIN_recent) else MIN_season)

FG2A_per_min_recent = pm_ratio(logs["FG2A"].head(int(n_recent) if n_recent != "Season" else len(logs)), logs["MIN"].head(int(n_recent) if n_recent != "Season" else len(logs)))
FG3A_per_min_recent = pm_ratio(logs["FG3A"].head(int(n_recent) if n_recent != "Season" else len(logs)), logs["MIN"].head(int(n_recent) if n_recent != "Season" else len(logs)))
FTA_per_min_recent  = pm_ratio(logs["FTA"].head(int(n_recent) if n_recent != "Season" else len(logs)),  logs["MIN"].head(int(n_recent) if n_recent != "Season" else len(logs)))

FG2A_per_min_season = pm_ratio(logs["FG2A"], logs["MIN"])
FG3A_per_min_season = pm_ratio(logs["FG3A"], logs["MIN"])
FTA_per_min_season  = pm_ratio(logs["FTA"],  logs["MIN"])

FG2A_career_pg = safe_div((career_pg_counting_stat(career_raw, "FGA") - career_pg_counting_stat(career_raw, "FG3A")), 1.0)
FG3A_career_pg = career_pg_counting_stat(career_raw, "FG3A")
FTA_career_pg  = career_pg_counting_stat(career_raw, "FTA")

FG2A_per_min_career = safe_div(FG2A_career_pg, MIN_career_pg)
FG3A_per_min_career = safe_div(FG3A_career_pg, MIN_career_pg)
FTA_per_min_career  = safe_div(FTA_career_pg,  MIN_career_pg)

def pct(series_m, series_a, n=None):
    m = pd.to_numeric(series_m, errors="coerce")
    a = pd.to_numeric(series_a, errors="coerce")
    if n is not None and n != "Season":
        n = int(n)
        m, a = m.head(n), a.head(n)
    mask = a > 0
    return float((m[mask] / a[mask]).mean()) if mask.any() else np.nan

FG2_PCT_recent = pct(logs["FG2M"], logs["FG2A"], n_recent)
FG3_PCT_recent = pct(logs["FG3M"], logs["FG3A"], n_recent)
FT_PCT_recent  = pct(logs["FTM"],  logs["FTA"],  n_recent)

FG2_PCT_season = pct(logs["FG2M"], logs["FG2A"])
FG3_PCT_season = pct(logs["FG3M"], logs["FG3A"])
FT_PCT_season  = pct(logs["FTM"],  logs["FTA"])

def career_pct(numer_col, denom_col):
    num = pd.to_numeric(career_raw.get(numer_col, np.nan), errors="coerce")
    den = pd.to_numeric(career_raw.get(denom_col, np.nan), errors="coerce")
    if num.isna().all() or den.isna().all() or den.sum() <= 0:
        return np.nan
    return float(num.sum() / den.sum())

FG2M_tot = (career_raw.get("FGM", 0) - career_raw.get("FG3M", 0))
FG2A_tot = (career_raw.get("FGA", 0) - career_raw.get("FG3A", 0))
career_raw = career_raw.assign(FG2M=FG2M_tot, FG2A=FG2A_tot)

FG2_PCT_career = career_pct("FG2M", "FG2A")
FG3_PCT_career = career_pct("FG3M", "FG3A")
FT_PCT_career  = career_pct("FTM",  "FTA")

ORB_per_min_recent = per_min(logs.get("OREB", np.nan), logs["MIN"], n=int(n_recent) if n_recent != "Season" else None)
DRB_per_min_recent = per_min(logs.get("DREB", np.nan), logs["MIN"], n=int(n_recent) if n_recent != "Season" else None)
AST_per_min_recent = per_min(logs.get("AST",  np.nan), logs["MIN"], n=int(n_recent) if n_recent != "Season" else None)

ORB_per_min_season = per_min(logs.get("OREB", np.nan), logs["MIN"])
DRB_per_min_season = per_min(logs.get("DREB", np.nan), logs["MIN"])
AST_per_min_season = per_min(logs.get("AST",  np.nan), logs["MIN"])

ORB_career_pg = career_pg_counting_stat(career_raw, "OREB")
DRB_career_pg = career_pg_counting_stat(career_raw, "DREB")
AST_career_pg = career_pg_counting_stat(career_raw, "AST")

ORB_per_min_career = safe_div(ORB_career_pg, MIN_career_pg)
DRB_per_min_career = safe_div(DRB_career_pg, MIN_career_pg)
AST_per_min_career = safe_div(AST_career_pg, MIN_career_pg)

def blend_vals_local(r, s, c):
    total = max(w_recent + w_season + w_career, 1e-9)
    vals, ws = [], []
    if np.isfinite(r): vals.append(r); ws.append(w_recent)
    if np.isfinite(s): vals.append(s); ws.append(w_season)
    if np.isfinite(c): vals.append(c); ws.append(w_career)
    if not vals:
        return np.nan
    ws = np.array(ws) / total
    return float(np.dot(np.array(vals), ws))

FG2A_per_min_blend = blend_vals_local(FG2A_per_min_recent, FG2A_per_min_season, FG2A_per_min_career)
FG3A_per_min_blend = blend_vals_local(FG3A_per_min_recent, FG3A_per_min_season, FG3A_per_min_career)
FTA_per_min_blend  = blend_vals_local(FTA_per_min_recent,  FTA_per_min_season,  FTA_per_min_career)

FG2_PCT_blend = blend_vals_local(FG2_PCT_recent, FG2_PCT_season, FG2_PCT_career)
FG3_PCT_blend = blend_vals_local(FG3_PCT_recent, FG3_PCT_season, FG3_PCT_career)
FT_PCT_blend  = blend_vals_local(FT_PCT_recent,  FT_PCT_season,  FT_PCT_career)

FG2A_proj = max(0.0, (FG2A_per_min_blend * MIN_proj) * AdjFactor) if np.isfinite(FG2A_per_min_blend) else 0.0
FG3A_proj = max(0.0, (FG3A_per_min_blend * MIN_proj) * AdjFactor) if np.isfinite(FG3A_per_min_blend) else 0.0
FTA_proj  = max(0.0, (FTA_per_min_blend  * MIN_proj) * AdjFactor) if np.isfinite(FTA_per_min_blend)  else 0.0

FG2M_proj = FG2A_proj * float(np.clip(FG2_PCT_blend if np.isfinite(FG2_PCT_blend) else 0.5, 0, 1))
FG3M_proj = FG3A_proj * float(np.clip(FG3_PCT_blend if np.isfinite(FG3_PCT_blend) else 0.35, 0, 1))
FTM_proj  = FTA_proj  * float(np.clip(FT_PCT_blend  if np.isfinite(FT_PCT_blend)  else 0.78, 0, 1))

PTS_proj = 2.0*FG2M_proj + 3.0*FG3M_proj + 1.0*FTM_proj

ORB_per_min_blend = blend_vals_local(ORB_per_min_recent, ORB_per_min_season, ORB_per_min_career)
DRB_per_min_blend = blend_vals_local(DRB_per_min_recent, DRB_per_min_season, DRB_per_min_career)

ORB_proj = max(0.0, (ORB_per_min_blend * MIN_proj) * AdjFactor * ORB_adj) if np.isfinite(ORB_per_min_blend) else 0.0
DRB_proj = max(0.0, (DRB_per_min_blend * MIN_proj) * AdjFactor * DRB_adj) if np.isfinite(DRB_per_min_blend) else 0.0
REB_proj = ORB_proj + DRB_proj

AST_per_min_blend = blend_vals_local(AST_per_min_recent, AST_per_min_season, AST_per_min_career)
AST_proj = max(0.0, (AST_per_min_blend * MIN_proj) * AdjFactor) if np.isfinite(AST_per_min_blend) else 0.0

PRA_proj = PTS_proj + REB_proj + AST_proj

# ----------------------- KPI Row -----------------------
kpi_stats = ["PTS","REB","AST","MIN"]
recent_len = len(logs) if n_recent == "Season" else int(n_recent)
recent_mean = logs[kpi_stats].head(recent_len).mean(numeric_only=True)
season_mean_vals = logs[kpi_stats].mean(numeric_only=True)
cols = st.columns(len(kpi_stats))
for i, s in enumerate(kpi_stats):
    if s not in logs.columns and s != "MIN":
        continue
    fs = form_score(logs[s].head(recent_len) if s in logs.columns else pd.Series([]), k=recent_len) if s in logs.columns else 50.0
    delta = (recent_mean.get(s, np.nan) - season_mean_vals.get(s, np.nan)) if s in recent_mean.index else np.nan
    cols[i].metric(
        label=f"{s} (L{recent_len if n_recent!='Season' else 'SZN'})",
        value=f"{recent_mean.get(s, np.nan):.1f}" if np.isfinite(recent_mean.get(s, np.nan)) else "—",
        delta=f"{delta:+.1f} vs SZN • Form {int(fs)}" if np.isfinite(delta) else f"Form {int(fs)}"
    )

# ----------------------- Recent Trends -----------------------
st.markdown(f"### Recent Trends (Last {recent_len if n_recent!='Season' else 'Season'} Games)")
trend_cols = [c for c in ["MIN","PTS","REB","AST","PRA","FG3M"] if c in logs.columns]
trend_df = logs[["GAME_DATE"] + trend_cols].head(recent_len).sort_values("GAME_DATE")
for s in trend_cols:
    chart = (
        alt.Chart(trend_df)
        .mark_line(point=True)
        .encode(x="GAME_DATE:T", y=alt.Y(s, title=s))
        .properties(height=160)
    )
    st.altair_chart(chart, use_container_width=True)

# ----------------------- Points Component Model -----------------------
st.markdown("### Points Component Model")
pts_block = pd.DataFrame({
    "Component": ["2PA","2PM","2P%","3PA","3PM","3P%","FTA","FTM","FT%","PTS"],
    "Value": [FG2A_proj, FG2M_proj, FG2_PCT_blend, FG3A_proj, FG3M_proj, FG3_PCT_blend, FTA_proj, FTM_proj, FT_PCT_blend, PTS_proj]
}).round(2)
st.dataframe(pts_block.style.format({"Value": "{:.2f}"}), use_container_width=True, height=_auto_height(pts_block))

# ----------------------- Projection Summary (no highlighting) -----------------------
st.markdown("### Projection Summary")
out = pd.DataFrame({
    "Stat": ["MIN","PTS","REB","AST","PRA","2PM","2PA","3PM","3PA","OREB","DREB"],
    "Proj": [MIN_proj, PTS_proj, REB_proj, AST_proj, PRA_proj, FG2M_proj, FG2A_proj, FG3M_proj, FG3A_proj, ORB_proj, DRB_proj]
})
out = out.round(2)
summary_indexed = out.set_index("Stat")
render_summary_table(summary_indexed)

# ----------------------- Compare Windows (career fixed per-game) -----------------------
st.markdown("### Compare Windows (Career vs Season vs L5/L10/L20)")
kpi_existing = [c for c in ["PTS","REB","AST","MIN"] if c in logs.columns]
L5  = window_avg(logs, 5,  kpi_existing)
L10 = window_avg(logs, 10, kpi_existing)
L20 = window_avg(logs, 20, kpi_existing)
season_avg_vals = logs[kpi_existing].mean(numeric_only=True)

career_pg = {
    "PTS": career_pg_counting_stat(career_raw, "PTS"),
    "REB": career_pg_counting_stat(career_raw, "REB"),
    "AST": career_pg_counting_stat(career_raw, "AST"),
    "MIN": career_pg_counting_stat(career_raw, "MIN"),
}
cmp_df = pd.concat(
    {"Career": pd.Series(career_pg),
     "Season": season_avg_vals,
     "L5": L5,
     "L10": L10,
     "L20": L20},
    axis=1
).round(2)
st.dataframe(cmp_df.style.format("{:.2f}"), use_container_width=True, height=_auto_height(cmp_df))

# ----------------------- Last 5 Games — {season_used} -----------------------
st.markdown(f"### Last 5 Games — {season_used}")
box_stat_order = ["MIN","PTS","REB","AST","PRA","FG2M","FG2A","FG3M","FG3A","OREB","DREB"]
meta_cols = ["GAME_DATE","MATCHUP","WL"]
recent_cols = [c for c in meta_cols + box_stat_order if c in logs.columns]

last5 = logs.sort_values("GAME_DATE", ascending=False).head(5).copy()
# ensure PRA present
if "PRA" not in last5.columns:
    last5["PRA"] = last5.get("PTS",0) + last5.get("REB",0) + last5.get("AST",0)

display_last5 = last5[recent_cols].rename(columns={
    "FG2M":"2PM","FG2A":"2PA","FG3M":"3PM","FG3A":"3PA"
})

# format box-score stats as whole numbers (no decimals)
int_cols = [c for c in ["MIN","PTS","REB","AST","PRA","2PM","2PA","3PM","3PA","OREB","DREB"] if c in display_last5.columns]
fmt_map = {c: "{:.0f}" for c in int_cols}

st.dataframe(
    display_last5.style.format(fmt_map),
    use_container_width=True,
    height=_auto_height(display_last5)
)

# Averages row (keep two decimals since these aren’t single-game box scores)
avg_cols = [c for c in ["MIN","PTS","REB","AST","PRA","2PM","2PA","3PM","3PA","OREB","DREB"] if c in display_last5.columns]
avg_last5 = display_last5[avg_cols].mean(numeric_only=True).to_frame().T.round(2)
avg_last5.index = ["Average (Last 5)"]
st.dataframe(
    avg_last5.style.format("{:.2f}"),
    use_container_width=True,
    height=_auto_height(avg_last5)
)

# ----------------------- Last 5 vs {opponent} — Most Recent Seasons -----------------------
st.markdown(f"### Last 5 vs {opponent} — Most Recent Seasons")
ts = teams_static_df.rename(columns={"id":"TEAM_ID", "full_name":"TEAM_NAME", "abbreviation":"TEAM_ABBR"})
match = ts.loc[ts["TEAM_NAME"].str.lower() == opponent.lower()]
opp_abbr = match["TEAM_ABBR"].iloc[0] if not match.empty else opponent[:3].upper()

vs5 = last_n_vs_opponent(player_id, opp_abbr, SEASONS, n=5)
if vs5.empty:
    st.info(f"No head-to-head games found vs {opponent} across recent seasons.")
else:
    if "PRA" not in vs5.columns:
        vs5["PRA"] = vs5.get("PTS",0) + vs5.get("REB",0) + vs5.get("AST",0)
    vs5_display = vs5.rename(columns={"FG2M":"2PM","FG2A":"2PA","FG3M":"3PM","FG3A":"3PA"}).copy()

    meta_cols = ["SEASON","GAME_DATE","MATCHUP","WL"]
    stat_cols = ["MIN","PTS","REB","AST","PRA","2PM","2PA","3PM","3PA","OREB","DREB"]
    vs_cols = [c for c in meta_cols + stat_cols if c in vs5_display.columns]
    vs5_display = vs5_display.sort_values("GAME_DATE", ascending=False)[vs_cols]

    # format single-game box stats as whole numbers
    int_cols_vs = [c for c in stat_cols if c in vs5_display.columns]
    fmt_map_vs = {c: "{:.0f}" for c in int_cols_vs}

    st.dataframe(
        vs5_display.style.format(fmt_map_vs),
        use_container_width=True,
        height=_auto_height(vs5_display)
    )

    # Averages row (keep two decimals)
    avg_vs = vs5_display[int_cols_vs].mean(numeric_only=True).to_frame().T.round(2)
    avg_vs.index = ["Average (vs Opp)"]
    st.dataframe(
        avg_vs.style.format("{:.2f}"),
        use_container_width=True,
        height=_auto_height(avg_vs)
    )

st.caption("Notes: Basic projections blend recent/season/career with a pace/defense opponent factor. Trend lines scale to your chosen window (or full season). If current season logs are empty, the app falls back to the most recent season with data.")

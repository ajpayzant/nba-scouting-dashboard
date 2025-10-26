# app.py
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
LEAGUE_DEF_REF = 112.0  # coarse league baseline for defense scaling (kept for continuity)
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

# ---------- NEW: tiny helpers to render dataframes neatly ----------
def _auto_height(df: pd.DataFrame, row_px: int = 34, header_px: int = 38, max_px: int = 900) -> int:
    """Compute a height that fits all rows to avoid vertical scroll in st.dataframe."""
    rows = max(len(df), 1)
    return min(max_px, header_px + row_px * rows + 8)

def render_summary_table(df_indexed: pd.DataFrame, highlight_rows=None):
    """
    Format to 2 decimals, highlight full rows (e.g., PRA, 3PM), and render with
    a height that avoids vertical scrolling.
    """
    if highlight_rows is None:
        highlight_rows = set()
    styler = df_indexed.style.format("{:.2f}")
    def _row_style(row):
        if row.name in highlight_rows:
            return ["font-weight: 700; background-color: #fff3cd;"] * len(row)
        return [""] * len(row)
    styler = styler.apply(_row_style, axis=1)
    h = _auto_height(df_indexed)
    st.dataframe(styler, use_container_width=True, height=h)

# ---------- Minutes projection (simple, robust, recent-weighted) ----------
def _minutes_cleaned(logs: pd.DataFrame) -> pd.Series:
    """Drop anomaly games where MIN < 10% of the season median minutes."""
    m = pd.to_numeric(logs.get("MIN", np.nan), errors="coerce").dropna()
    if m.empty:
        return m
    season_median = float(m.median())
    thresh = 0.10 * season_median
    m_clean = m[m >= thresh]
    return m_clean if len(m_clean) else m

def _window_mean(m: pd.Series, n: int) -> float:
    if m.empty:
        return np.nan
    return float(pd.to_numeric(m.head(n), errors="coerce").dropna().mean()) if len(m) else np.nan

def _recent_blend_from_windows(m: pd.Series) -> float:
    windows = [(5, 0.50), (10, 0.25), (15, 0.15), (20, 0.10)]
    parts, wts = [], []
    for n, w in windows:
        v = _window_mean(m, n)
        if np.isfinite(v):
            parts.append(w * v); wts.append(w)
    if not parts:
        return float(np.nan)
    return float(sum(parts) / sum(wts))

def project_minutes_simple(logs: pd.DataFrame,
                           career_raw: pd.DataFrame,
                           w_recent: float, w_season: float, w_career: float) -> float:
    m_clean = _minutes_cleaned(logs)
    recent_blend = _recent_blend_from_windows(m_clean)
    season_mean  = float(m_clean.mean()) if len(m_clean) else np.nan
    career_pg    = career_pg_counting_stat(career_raw, "MIN")
    if not np.isfinite(recent_blend): recent_blend = season_mean
    if not np.isfinite(season_mean):  season_mean  = career_pg
    wr, ws, wc = (w_recent or 0.0), (w_season or 0.0), (w_career or 0.0)
    total = wr + ws + wc
    if total <= 0:
        wr, ws, wc = 0.55, 0.30, 0.15; total = 1.0
    wr, ws, wc = wr/total, ws/total, wc/total
    parts = []
    if np.isfinite(recent_blend): parts.append(wr * recent_blend)
    if np.isfinite(season_mean):  parts.append(ws * season_mean)
    if np.isfinite(career_pg):    parts.append(wc * career_pg)
    MIN_proj = float(sum(parts)) if parts else 0.0
    return max(0.0, MIN_proj)

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

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_active_players_df():
    return pd.DataFrame(static_players.get_active_players())

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_teams_static_df():
    return pd.DataFrame(static_teams.get_teams())

# ---------- NEW: Team tables with Opponent profile & league means ----------
@st.cache_data(ttl=CACHE_HOURS*3600)
def get_team_tables(season: str):
    # Advanced: for PACE, DEF_RATING (Regular Season only by default)
    df_adv = leaguedashteamstats.LeagueDashTeamStats(
        season=season, per_mode_detailed="PerGame", measure_type_detailed="Advanced"
    ).get_data_frames()[0]

    # Opponent profile: OPP_* per game
    df_opp = leaguedashteamstats.LeagueDashTeamStats(
        season=season, per_mode_detailed="PerGame", measure_type_detailed="Opponent"
    ).get_data_frames()[0]

    # Base (for OREB and a pace proxy fallback)
    df_base = leaguedashteamstats.LeagueDashTeamStats(
        season=season, per_mode_detailed="PerGame"
    ).get_data_frames()[0]

    # Keep only NBA teams and key cols
    df_adv = df_adv[df_adv["TEAM_ID"].apply(is_nba_team_id)].copy()
    df_opp = df_opp[df_opp["TEAM_ID"].apply(is_nba_team_id)].copy()
    df_base = df_base[df_base["TEAM_ID"].apply(is_nba_team_id)].copy()

    adv_keep = ["TEAM_ID","TEAM_NAME","PACE","DEF_RATING","PACE_RANK","DEF_RATING_RANK"]
    opp_keep = ["TEAM_ID","TEAM_NAME","OPP_FGA","OPP_FG3A","OPP_FTA","OPP_OREB","OPP_DREB","OPP_PTS","OPP_AST"]
    base_keep = ["TEAM_ID","TEAM_NAME","MIN","FGA","FTA","OREB","TOV"]

    df_adv = df_adv[adv_keep].copy()
    df_opp = df_opp[opp_keep].copy()
    df_base = df_base[base_keep].copy()
    df_base["PACE_PROXY"] = df_base.apply(pace_proxy_row, axis=1)

    # League means used for normalization
    league_pace_mean = float(df_adv["PACE"].mean())
    league_opp_fg3a_mean = float(df_opp["OPP_FG3A"].mean())
    league_opp_fta_mean  = float(df_opp["OPP_FTA"].mean())
    league_opp_fga_mean  = float(df_opp["OPP_FGA"].mean())
    league_opp_2pa_mean  = float((df_opp["OPP_FGA"] - df_opp["OPP_FG3A"]).mean())
    league_opp_oreb_mean = float(df_opp["OPP_OREB"].mean())
    league_opp_dreb_mean = float(df_opp["OPP_DREB"].mean())
    league_opp_ast_mean  = float(df_opp["OPP_AST"].mean())

    # Merge to one context table
    teams_ctx = (
        df_adv
        .merge(df_opp, on=["TEAM_ID","TEAM_NAME"], how="left")
        .merge(df_base[["TEAM_ID","PACE_PROXY","OREB"]], on="TEAM_ID", how="left")
        .sort_values("TEAM_NAME").reset_index(drop=True)
    )

    # Return context + league means needed downstream
    league_means = {
        "PACE": league_pace_mean,
        "OPP_FG3A": league_opp_fg3a_mean,
        "OPP_FTA": league_opp_fta_mean,
        "OPP_FGA": league_opp_fga_mean,
        "OPP_2PA": league_opp_2pa_mean,
        "OPP_OREB": league_opp_oreb_mean,
        "OPP_DREB": league_opp_dreb_mean,
        "OPP_AST": league_opp_ast_mean,
    }
    return teams_ctx, league_means

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

    teams_ctx, league_means = get_team_tables(season)
    opponent = st.selectbox("Opponent", teams_ctx["TEAM_NAME"].tolist())

    n_recent = st.radio("Recent window (games)", [5, 10, 15, 20, 25], horizontal=True, index=0)
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

# Baseline opponent numbers
opp_def   = float(opp_row["DEF_RATING"])
opp_pace  = float(opp_row["PACE"])
opp_fga   = float(opp_row.get("OPP_FGA", np.nan))
opp_fg3a  = float(opp_row.get("OPP_FG3A", np.nan))
opp_2pa   = float((opp_fga - opp_fg3a)) if np.isfinite(opp_fga) and np.isfinite(opp_fg3a) else np.nan
opp_fta   = float(opp_row.get("OPP_FTA", np.nan))
opp_oreb  = float(opp_row.get("OPP_OREB", np.nan))
opp_dreb  = float(opp_row.get("OPP_DREB", np.nan))
opp_opp_ast = float(opp_row.get("OPP_AST", np.nan))  # assists allowed

# League means for normalization
league_pace_mean = float(league_means["PACE"])
L_OPP_FG3A = float(league_means["OPP_FG3A"])
L_OPP_FTA  = float(league_means["OPP_FTA"])
L_OPP_2PA  = float(league_means["OPP_2PA"])
L_OPP_OREB = float(league_means["OPP_OREB"])
L_OPP_DREB = float(league_means["OPP_DREB"])
L_OPP_AST  = float(league_means["OPP_AST"])

# Derived multipliers (pace & profile). Keep robust defaults if missing.
pace_factor = (opp_pace / league_pace_mean) if (np.isfinite(opp_pace) and league_pace_mean > 0) else 1.0
m_3pa = (opp_fg3a / L_OPP_FG3A) if (np.isfinite(opp_fg3a) and L_OPP_FG3A > 0) else 1.0
m_2pa = (opp_2pa  / L_OPP_2PA)  if (np.isfinite(opp_2pa)  and L_OPP_2PA  > 0) else 1.0
m_fta = (opp_fta  / L_OPP_FTA)  if (np.isfinite(opp_fta)  and L_OPP_FTA  > 0) else 1.0
# Rebound logic: More OPP_DREB (opponents grabbing DREB vs them) => harder to get OREB → inverse scale
m_oreb = (L_OPP_DREB / opp_dreb) if (np.isfinite(opp_dreb) and opp_dreb > 0) else 1.0
# More OPP_OREB allowed → more DREB chances for us → direct scale
m_dreb = (opp_oreb / L_OPP_OREB) if (np.isfinite(opp_oreb) and L_OPP_OREB > 0) else 1.0
# Assists allowed profile (optional, gentle)
m_ast = (opp_opp_ast / L_OPP_AST) if (np.isfinite(opp_opp_ast) and L_OPP_AST > 0) else 1.0

# ----------------------- Player header (Age • Position • Seasons • GP) -----------------------
left, right = st.columns([2, 1])
with left:
    st.subheader(f"{player_name} — {season_used}")
    age = pos = exp = None
    if not cpi.empty:
        age = cpi.get("AGE", pd.Series([None])).iloc[0] if "AGE" in cpi.columns else None
        pos = cpi.get("POSITION", pd.Series([None])).iloc[0] if "POSITION" in cpi.columns else None
        exp = cpi.get("SEASON_EXP", pd.Series([None])).iloc[0] if "SEASON_EXP" in cpi.columns else None
    gp_this_season = int(len(logs))
    meta = []
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
    c2.metric("Pace", f"{opp_pace:.1f}")

# ----------------------- Baselines & blends -----------------------
def mean_recent(series, n=n_recent):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.iloc[:min(n, len(s))].mean()) if len(s) else np.nan

def season_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan

# Minutes (new simple/robust version)
MIN_proj = project_minutes_simple(logs, career_raw, w_recent, w_season, w_career)

# --- Per-minute rates: recent/season/career blends (unchanged logic) ---
FG2A_per_min_recent = pm_ratio(logs["FG2A"].head(n_recent), logs["MIN"].head(n_recent))
FG3A_per_min_recent = pm_ratio(logs["FG3A"].head(n_recent), logs["MIN"].head(n_recent))
FTA_per_min_recent  = pm_ratio(logs["FTA"].head(n_recent),  logs["MIN"].head(n_recent))

FG2A_per_min_season = pm_ratio(logs["FG2A"], logs["MIN"])
FG3A_per_min_season = pm_ratio(logs["FG3A"], logs["MIN"])
FTA_per_min_season  = pm_ratio(logs["FTA"],  logs["MIN"])

FG2A_career_pg = safe_div((career_pg_counting_stat(career_raw, "FGA") - career_pg_counting_stat(career_raw, "FG3A")), 1.0)
FG3A_career_pg = career_pg_counting_stat(career_raw, "FG3A")
FTA_career_pg  = career_pg_counting_stat(career_raw, "FTA")
MIN_career_pg  = career_pg_counting_stat(career_raw, "MIN")

FG2A_per_min_career = safe_div(FG2A_career_pg, MIN_career_pg)
FG3A_per_min_career = safe_div(FG3A_career_pg, MIN_career_pg)
FTA_per_min_career  = safe_div(FTA_career_pg,  MIN_career_pg)

def pct(series_m, series_a, n=None):
    m = pd.to_numeric(series_m, errors="coerce")
    a = pd.to_numeric(series_a, errors="coerce")
    if n is not None:
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

ORB_per_min_recent = per_min(logs.get("OREB", np.nan), logs["MIN"], n=n_recent)
DRB_per_min_recent = per_min(logs.get("DREB", np.nan), logs["MIN"], n=n_recent)
AST_per_min_recent = per_min(logs.get("AST",  np.nan), logs["MIN"], n=n_recent)

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
    return (w_recent/total)*r + (w_season/total)*s + (w_career/total)*c

FG2A_per_min_blend = blend_vals_local(FG2A_per_min_recent, FG2A_per_min_season, FG2A_per_min_career)
FG3A_per_min_blend = blend_vals_local(FG3A_per_min_recent, FG3A_per_min_season, FG3A_per_min_career)
FTA_per_min_blend  = blend_vals_local(FTA_per_min_recent,  FTA_per_min_season,  FTA_per_min_career)

FG2_PCT_blend = blend_vals_local(FG2_PCT_recent, FG2_PCT_season, FG2_PCT_career)
FG3_PCT_blend = blend_vals_local(FG3_PCT_recent, FG3_PCT_season, FG3_PCT_career)
FT_PCT_blend  = blend_vals_local(FT_PCT_recent,  FT_PCT_season,  FT_PCT_career)

ORB_per_min_blend = blend_vals_local(ORB_per_min_recent, ORB_per_min_season, ORB_per_min_career)
DRB_per_min_blend = blend_vals_local(DRB_per_min_recent, DRB_per_min_season, DRB_per_min_career)
AST_per_min_blend = blend_vals_local(AST_per_min_recent, AST_per_min_season, AST_per_min_career)

# ---------- NEW: Component-level opponent + pace adjustments on PER-MIN rates ----------
FG2A_per_min_adj = (FG2A_per_min_blend or 0.0) * pace_factor * m_2pa
FG3A_per_min_adj = (FG3A_per_min_blend or 0.0) * pace_factor * m_3pa
FTA_per_min_adj  = (FTA_per_min_blend  or 0.0) * pace_factor * m_fta

ORB_per_min_adj  = (ORB_per_min_blend or 0.0) * pace_factor * m_oreb
DRB_per_min_adj  = (DRB_per_min_blend or 0.0) * pace_factor * m_dreb
AST_per_min_adj  = (AST_per_min_blend or 0.0) * pace_factor * m_ast

# ---------- Projected totals ----------
FG2A_proj = max(0.0, FG2A_per_min_adj * MIN_proj) if np.isfinite(FG2A_per_min_adj) else 0.0
FG3A_proj = max(0.0, FG3A_per_min_adj * MIN_proj) if np.isfinite(FG3A_per_min_adj) else 0.0
FTA_proj  = max(0.0, FTA_per_min_adj  * MIN_proj) if np.isfinite(FTA_per_min_adj)  else 0.0

FG2M_proj = FG2A_proj * float(np.clip(FG2_PCT_blend if np.isfinite(FG2_PCT_blend) else 0.5, 0, 1))
FG3M_proj = FG3A_proj * float(np.clip(FG3_PCT_blend if np.isfinite(FG3_PCT_blend) else 0.35, 0, 1))
FTM_proj  = FTA_proj  * float(np.clip(FT_PCT_blend  if np.isfinite(FT_PCT_blend)  else 0.78, 0, 1))
PTS_proj  = 2.0*FG2M_proj + 3.0*FG3M_proj + 1.0*FTM_proj

ORB_proj = max(0.0, ORB_per_min_adj * MIN_proj) if np.isfinite(ORB_per_min_adj) else 0.0
DRB_proj = max(0.0, DRB_per_min_adj * MIN_proj) if np.isfinite(DRB_per_min_adj) else 0.0
REB_proj = ORB_proj + DRB_proj

AST_proj = max(0.0, AST_per_min_adj * MIN_proj) if np.isfinite(AST_per_min_adj) else 0.0
PRA_proj = PTS_proj + REB_proj + AST_proj

# ---------- NEW: Tight central range (45th–55th pct) via light bootstrap ----------
def _bootstrap_ranges(n_sims=400):
    rng = np.random.default_rng(42)
    mins = max(0.0, float(MIN_proj))

    # attempts (Poisson), makes (Binomial), counts (Poisson)
    lam_2pa = max(0.0, FG2A_per_min_adj * mins)
    lam_3pa = max(0.0, FG3A_per_min_adj * mins)
    lam_fta = max(0.0, FTA_per_min_adj  * mins)

    p2 = float(np.clip(FG2_PCT_blend if np.isfinite(FG2_PCT_blend) else 0.5,  0, 1))
    p3 = float(np.clip(FG3_PCT_blend if np.isfinite(FG3_PCT_blend) else 0.35, 0, 1))
    pft= float(np.clip(FT_PCT_blend  if np.isfinite(FT_PCT_blend)  else 0.78, 0, 1))

    lam_oreb = max(0.0, ORB_per_min_adj * mins)
    lam_dreb = max(0.0, DRB_per_min_adj * mins)
    lam_ast  = max(0.0, AST_per_min_adj * mins)

    PTS, REB, AST, PRA = [], [], [], []
    for _ in range(n_sims):
        a2 = rng.poisson(lam_2pa); m2 = rng.binomial(a2, p2)
        a3 = rng.poisson(lam_3pa); m3 = rng.binomial(a3, p3)
        af = rng.poisson(lam_fta); mf = rng.binomial(af, pft)
        orr= rng.poisson(lam_oreb); drr= rng.poisson(lam_dreb)
        ast= rng.poisson(lam_ast)

        pts = 2*m2 + 3*m3 + mf
        reb = orr + drr
        pra = pts + reb + ast

        PTS.append(pts); REB.append(reb); AST.append(ast); PRA.append(pra)

    def band(v):
        lo = float(np.percentile(v, 45))
        hi = float(np.percentile(v, 55))
        return lo, hi

    return band(PTS), band(REB), band(AST), band(PRA)

pts_band, reb_band, ast_band, pra_band = _bootstrap_ranges()
def _fmt_band(b): return f"{b[0]:.1f}–{b[1]:.1f}"

# ----------------------- KPI Row -----------------------
kpi_stats = ["PTS","REB","AST","MIN"]
recent_mean = logs[kpi_stats].head(n_recent).mean(numeric_only=True)
season_mean_vals = logs[kpi_stats].mean(numeric_only=True)
cols = st.columns(len(kpi_stats))
for i, s in enumerate(kpi_stats):
    if s not in logs.columns and s != "MIN":
        continue
    fs = form_score(logs[s], k=n_recent) if s in logs.columns else 50.0
    delta = (recent_mean.get(s, np.nan) - season_mean_vals.get(s, np.nan)) if s in recent_mean.index else np.nan
    cols[i].metric(
        label=f"{s} (L{n_recent})",
        value=f"{recent_mean.get(s, np.nan):.1f}" if np.isfinite(recent_mean.get(s, np.nan)) else "—",
        delta=f"{delta:+.1f} vs SZN • Form {int(fs)}" if np.isfinite(delta) else f"Form {int(fs)}"
    )

# ----------------------- Recent Trends -----------------------
st.markdown("### Recent Trends (Last 20 Games)")
trend_cols = [c for c in ["MIN","PTS","REB","AST","PRA","FG3M"] if c in logs.columns]
trend_df = logs[["GAME_DATE"] + trend_cols].head(20).sort_values("GAME_DATE")
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

# ----------------------- Projection Summary (order + bold PRA & 3PM) -----------------------
st.markdown("### Projection Summary")
out = pd.DataFrame({
    "Stat": ["MIN","PTS","REB","AST","PRA","2PM","2PA","3PM","3PA","OREB","DREB"],
    "Proj": [MIN_proj, PTS_proj, REB_proj, AST_proj, PRA_proj, FG2M_proj, FG2A_proj, FG3M_proj, FG3A_proj, ORB_proj, DRB_proj],
    "Range (45–55%)": ["—", _fmt_band(pts_band), _fmt_band(reb_band), _fmt_band(ast_band), _fmt_band(pra_band), "—","—","—","—","—","—"],
})
out = out.round(2)
summary_indexed = out.set_index("Stat")
render_summary_table(summary_indexed, highlight_rows={"PRA","3PM"})

# ----------------------- Compare Windows (Career vs Season vs L5/L10/L20) -----------------------
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
if "PRA" not in last5.columns:
    last5["PRA"] = last5.get("PTS",0) + last5.get("REB",0) + last5.get("AST",0)

display_last5 = last5[recent_cols].rename(columns={
    "FG2M":"2PM","FG2A":"2PA","FG3M":"3PM","FG3A":"3PA"
})
int_cols = [c for c in ["MIN","PTS","REB","AST","PRA","2PM","2PA","3PM","3PA","OREB","DREB"] if c in display_last5.columns]
fmt_map = {c: "{:.0f}" for c in int_cols}
st.dataframe(display_last5.style.format(fmt_map), use_container_width=True, height=_auto_height(display_last5))

avg_cols = [c for c in ["MIN","PTS","REB","AST","PRA","2PM","2PA","3PM","3PA","OREB","DREB"] if c in display_last5.columns]
avg_last5 = display_last5[avg_cols].mean(numeric_only=True).to_frame().T.round(2)
avg_last5.index = ["Average (Last 5)"]
st.dataframe(avg_last5.style.format("{:.2f}"), use_container_width=True, height=_auto_height(avg_last5))

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

    int_cols_vs = [c for c in stat_cols if c in vs5_display.columns]
    fmt_map_vs = {c: "{:.0f}" for c in int_cols_vs}

    st.dataframe(vs5_display.style.format(fmt_map_vs), use_container_width=True, height=_auto_height(vs5_display))

    avg_vs = vs5_display[int_cols_vs].mean(numeric_only=True).to_frame().T.round(2)
    avg_vs.index = ["Average (vs Opp)"]
    st.dataframe(avg_vs.style.format("{:.2f}"), use_container_width=True, height=_auto_height(avg_vs))

st.caption("Notes: Opponent multipliers (3PA/2PA/FTA/REB/AST) are normalized vs league and applied to per-minute rates, with pace affecting attempts only. Minutes are projected via a robust, recent-weighted method. Ranges show the tight 45–55th percentile from a light bootstrap. Career values use proper per-game aggregation from season totals.")

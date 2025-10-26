# app.py
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import datetime
import re
import time

from nba_api.stats.static import players as static_players, teams as static_teams
from nba_api.stats.endpoints import (
    playergamelog,
    playercareerstats,
    leaguedashteamstats,
    LeagueDashPlayerStats,
    commonplayerinfo,
    leaguegamelog,
)

# ----------------------- Streamlit Setup -----------------------
st.set_page_config(page_title="NBA Player Scouting Dashboard", layout="wide")
st.title("NBA Player Scouting Dashboard")

# ----------------------- Config -----------------------
CACHE_HOURS = 12
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

def pct(series_m, series_a, n=None):
    m = pd.to_numeric(series_m, errors="coerce")
    a = pd.to_numeric(series_a, errors="coerce")
    if n is not None:
        m, a = m.head(n), a.head(n)
    mask = a > 0
    return float((m[mask] / a[mask]).mean()) if mask.any() else np.nan

def career_pct(df, num, den):
    nu = pd.to_numeric(df.get(num,np.nan), errors="coerce")
    de = pd.to_numeric(df.get(den,np.nan), errors="coerce")
    if de.isna().all() or de.sum()<=0: return np.nan
    return float(nu.sum()/de.sum())

# ---------- NEW: tiny helpers to render dataframes neatly ----------
def _auto_height(df: pd.DataFrame, row_px: int = 34, header_px: int = 38, max_px: int = 900) -> int:
    """Compute a height that fits all rows to avoid vertical scroll in st.dataframe."""
    rows = max(len(df), 1)
    return min(max_px, header_px + row_px * rows + 8)

def render_summary_table(df_indexed: pd.DataFrame):
    """Uniform 2-decimal style, no special highlighting."""
    styler = df_indexed.style.format("{:.2f}")
    h = _auto_height(df_indexed)
    st.dataframe(styler, use_container_width=True, height=h)

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

# ----------------------- Robust team tables (Advanced + Opponent) -----------------------
def _call_lds(season: str, measure: str, **extra_kwargs) -> pd.DataFrame:
    """
    Wrapper around LeagueDashTeamStats with retry + longer timeout.
    We pass measure_type_detailed=..., and on older nba_api versions try the *_defense kw.
    """
    base = dict(season=season, per_mode_detailed="PerGame", season_type_all_star="Regular Season")
    base.update(extra_kwargs)
    attempts = 0
    last_err = None
    while attempts < 3:
        attempts += 1
        try:
            # First try with modern kw
            return leaguedashteamstats.LeagueDashTeamStats(
                measure_type_detailed=measure, timeout=60, **base
            ).get_data_frames()[0]
        except TypeError:
            # Fallback kw for older versions
            try:
                return leaguedashteamstats.LeagueDashTeamStats(
                    measure_type_detailed_defense=measure, timeout=60, **base
                ).get_data_frames()[0]
            except Exception as e:
                last_err = e
        except Exception as e:
            last_err = e
        time.sleep(3)
    raise RuntimeError(f"Failed to fetch LeagueDashTeamStats({measure}) after retries: {last_err}")

def _regular_season_start(season: str) -> str:
    try:
        lg = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season").get_data_frames()[0]
        first_dt = pd.to_datetime(lg["GAME_DATE"], errors="coerce").min()
        if pd.notna(first_dt):
            return first_dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    # safe fallback mid-Oct
    return f"{season[:4]}-10-15"

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_team_tables(season: str):
    date_from = _regular_season_start(season)
    df_adv = _call_lds(season, "Advanced", date_from_nullable=date_from)
    df_opp = _call_lds(season, "Opponent", date_from_nullable=date_from)
    df_base = leaguedashteamstats.LeagueDashTeamStats(
        season=season, per_mode_detailed="PerGame", season_type_all_star="Regular Season", timeout=60
    ).get_data_frames()[0]

    df_adv = df_adv[df_adv["TEAM_ID"].apply(is_nba_team_id)].copy()
    df_opp = df_opp[df_opp["TEAM_ID"].apply(is_nba_team_id)].copy()
    df_base = df_base[df_base["TEAM_ID"].apply(is_nba_team_id)].copy()

    # Keep only needed columns
    keep_adv = ["TEAM_ID","TEAM_NAME","PACE","DEF_RATING","DREB_PCT"]
    df_adv = df_adv[keep_adv].copy()

    keep_opp = ["TEAM_ID","TEAM_NAME","OPP_PTS","OPP_FG3A","OPP_FTA","OPP_OREB","OPP_DREB","OPP_AST"]
    df_opp = df_opp[keep_opp].copy().rename(columns={"OPP_PTS":"PTS_ALLOWED"})

    keep_base = ["TEAM_ID","TEAM_NAME","OREB"]
    df_base = df_base[keep_base].copy()

    teams_ctx = (
        df_adv.merge(df_opp, on=["TEAM_ID","TEAM_NAME"], how="left")
              .merge(df_base, on=["TEAM_ID","TEAM_NAME"], how="left")
              .sort_values("TEAM_NAME")
              .reset_index(drop=True)
    )

    # Ranks (1 is best): PACE (higher better), DEF_RATING (lower better), PTS_ALLOWED (lower better)
    teams_ctx["RANK_PACE"] = teams_ctx["PACE"].rank(ascending=False, method="min").astype(int)
    teams_ctx["RANK_DEF"]  = teams_ctx["DEF_RATING"].rank(ascending=True,  method="min").astype(int)
    teams_ctx["RANK_PA"]   = teams_ctx["PTS_ALLOWED"].rank(ascending=True,  method="min").astype(int)

    # League means for adjusters
    league_means = {
        "PACE":      float(teams_ctx["PACE"].mean()),
        "OPP_FG3A":  float(teams_ctx["OPP_FG3A"].mean()),
        "OPP_FTA":   float(teams_ctx["OPP_FTA"].mean()),
        "OPP_OREB":  float(teams_ctx["OPP_OREB"].mean()),
        "OPP_DREB":  float(teams_ctx["OPP_DREB"].mean()),
        "OPP_AST":   float(teams_ctx["OPP_AST"].mean()),
    }
    return teams_ctx, league_means

# ----------------------- Player logs & career -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600)
def get_player_logs(player_id: int, season: str):
    df = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    if df.empty:
        return df
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    # derive 2P components
    df["FG2M"] = pd.to_numeric(df.get("FGM", 0), errors="coerce") - pd.to_numeric(df.get("FG3M", 0), errors="coerce")
    df["FG2A"] = pd.to_numeric(df.get("FGA", 0), errors="coerce") - pd.to_numeric(df.get("FG3A", 0), errors="coerce")
    df["FG2M"] = df["FG2M"].clip(lower=0)
    df["FG2A"] = df["FG2A"].clip(lower=0)
    # ensure columns & derive PRA
    for col in ["PTS","REB","AST","FG3M","FG3A","OREB","DREB","MIN","FTA","FTM"]:
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

# ----------------------- Minutes Projection (robust) -----------------------
def robust_minutes_projection(logs: pd.DataFrame, career_df: pd.DataFrame,
                              n_recent_choice, w_recent=0.55, w_season=0.30, w_career=0.15,
                              outlier_frac=0.10) -> float:
    mins = pd.to_numeric(logs["MIN"], errors="coerce").dropna()
    if mins.empty:
        return float(career_pg_counting_stat(career_df, "MIN"))
    season_med = float(np.nanmedian(mins)) if len(mins) else np.nan
    cutoff = outlier_frac * season_med if np.isfinite(season_med) else 0.0
    clean = mins[mins >= cutoff] if np.isfinite(cutoff) else mins
    if n_recent_choice == "Season":
        recent_series = clean
    else:
        recent_series = clean.head(int(n_recent_choice))
    recent_mean = float(recent_series.mean()) if len(recent_series) else np.nan
    season_mean = float(clean.mean()) if len(clean) else np.nan
    career_min_pg = float(career_pg_counting_stat(career_df, "MIN"))
    tot = max(w_recent + w_season + w_career, 1e-9)
    comps, ws = [], []
    if np.isfinite(recent_mean): comps.append(recent_mean); ws.append(w_recent)
    if np.isfinite(season_mean): comps.append(season_mean); ws.append(w_season)
    if np.isfinite(career_min_pg): comps.append(career_min_pg); ws.append(w_career)
    if not comps:
        return float(season_mean) if np.isfinite(season_mean) else float(mins.mean())
    ws = np.array(ws) / tot
    return float(np.dot(np.array(comps), ws))

# ----------------------- Opponent multipliers -----------------------
def opponent_multipliers(teams_ctx, league_means, opponent_name: str):
    r = teams_ctx.loc[teams_ctx["TEAM_NAME"] == opponent_name]
    if r.empty:
        r = teams_ctx[teams_ctx["TEAM_NAME"].str.contains(opponent_name, case=False, na=False)]
    r = r.iloc[0]

    pace_factor = safe_div(r["PACE"], league_means["PACE"]) or 1.0
    m_3pa = safe_div(r["OPP_FG3A"], league_means["OPP_FG3A"]) or 1.0
    m_fta = safe_div(r["OPP_FTA"],  league_means["OPP_FTA"])  or 1.0
    # Rebound multipliers (see rationale: if OPP_DREB is high, your OREB chances drop)
    m_oreb = safe_div(league_means["OPP_DREB"], r["OPP_DREB"]) or 1.0
    m_dreb = safe_div(r["OPP_OREB"], league_means["OPP_OREB"]) or 1.0
    m_ast  = safe_div(r["OPP_AST"], league_means["OPP_AST"]) or 1.0

    return dict(
        pace_factor=float(pace_factor),
        m_3pa=float(m_3pa),
        m_fta=float(m_fta),
        m_oreb=float(m_oreb),
        m_dreb=float(m_dreb),
        m_ast=float(m_ast),
        row=r  # include the row for header display
    )

def half_width_for_coverage(sd, coverage=0.50):
    z = {0.50: 0.674, 0.68: 1.000, 0.80: 1.282, 0.90: 1.645}.get(coverage, 0.674)
    return float(z * sd)

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

    # Team context & opponent
    teams_ctx, league_means = get_team_tables(season)
    opponent = st.selectbox("Opponent", teams_ctx["TEAM_NAME"].tolist())

    # Windows incl. "Season"
    n_choice = st.radio("Recent window", ["Season", 5, 10, 15, 20, 25], horizontal=True, index=1)
    w_recent = st.slider("Weight: Recent", 0.0, 1.0, 0.55, 0.05)
    w_season = st.slider("Weight: Season", 0.0, 1.0, 0.30, 0.05)
    w_career = st.slider("Weight: Career", 0.0, 1.0, 0.15, 0.05)

# ----------------------- Data fetch -----------------------
season_used = best_season_for_player(player_id, season, SEASONS)
logs = get_player_logs(player_id, season_used)
if logs.empty:
    st.error("No game logs found for this player across available seasons.")
    st.stop()
career_raw = get_player_career(player_id)
cpi = get_common_player_info(player_id)

# ----------------------- Opponent Context & Header -----------------------
adj = opponent_multipliers(teams_ctx, league_means, opponent)
opp_row = adj["row"]
opp_def = float(opp_row["DEF_RATING"])
opp_pace = float(opp_row["PACE"])
opp_pa   = float(opp_row["PTS_ALLOWED"])
rank_def = int(opp_row["RANK_DEF"])
rank_pace = int(opp_row["RANK_PACE"])
rank_pa   = int(opp_row["RANK_PA"])

def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20: suffix = "th"
    else: suffix = {1:"st",2:"nd",3:"rd"}.get(n%10,"th")
    return f"{n}{suffix}"

# ----------------------- Player header (Age • Position • Seasons • GP) -----------------------
left, right = st.columns([2, 1.4])
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
    # Centered headings + centered metric values w/ italics smaller rank
    html = f"""
    <style>
      .oppwrap {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; text-align:center; }}
      .opph {{ font-weight:600; margin-bottom:6px; }}
      .oppv {{ font-size:1.25rem; line-height:1.3; font-weight:700; }}
      .rank {{ font-style:italic; font-size:0.9rem; font-weight:500; margin-left:4px; }}
      @media (max-width: 1200px) {{
         .oppv {{ font-size:1.1rem; }}
         .rank {{ font-size:0.85rem; }}
      }}
    </style>
    <div class="oppwrap">
      <div>
        <div class="opph">DEF Rating</div>
        <div class="oppv">{opp_def:.1f}<span class="rank">({_ordinal(rank_def)})</span></div>
      </div>
      <div>
        <div class="opph">Points Allowed</div>
        <div class="oppv">{opp_pa:.1f}<span class="rank">({_ordinal(rank_pa)})</span></div>
      </div>
      <div>
        <div class="opph">Pace</div>
        <div class="oppv">{opp_pace:.2f}<span class="rank">({_ordinal(rank_pace)})</span></div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ----------------------- Baselines, blends, projections -----------------------
# Determine numeric n_recent from choice
n_recent = len(logs) if n_choice == "Season" else int(n_choice)

# Minutes projection (robust)
MIN_proj = robust_minutes_projection(logs, career_raw, n_choice, w_recent, w_season, w_career, outlier_frac=0.10)

# Per-minute attempt rates (recent/season/career)
FG2A_pm_r = pm_ratio(logs["FG2A"].head(n_recent), logs["MIN"].head(n_recent))
FG3A_pm_r = pm_ratio(logs["FG3A"].head(n_recent), logs["MIN"].head(n_recent))
FTA_pm_r  = pm_ratio(logs["FTA"].head(n_recent),  logs["MIN"].head(n_recent))

FG2A_pm_s = pm_ratio(logs["FG2A"], logs["MIN"])
FG3A_pm_s = pm_ratio(logs["FG3A"], logs["MIN"])
FTA_pm_s  = pm_ratio(logs["FTA"],  logs["MIN"])

MIN_c_pg  = career_pg_counting_stat(career_raw,"MIN")
FG2A_c_pg = (career_pg_counting_stat(career_raw,"FGA") - career_pg_counting_stat(career_raw,"FG3A"))
FG3A_c_pg = career_pg_counting_stat(career_raw,"FG3A")
FTA_c_pg  = career_pg_counting_stat(career_raw,"FTA")

FG2A_pm_c = safe_div(FG2A_c_pg, MIN_c_pg)
FG3A_pm_c = safe_div(FG3A_c_pg, MIN_c_pg)
FTA_pm_c  = safe_div(FTA_c_pg,  MIN_c_pg)

def blend_three(a,b,c, w_r, w_s, w_c):
    tot=max(w_r+w_s+w_c,1e-9)
    arr=[]; ws=[]
    if np.isfinite(a): arr.append(a); ws.append(w_r)
    if np.isfinite(b): arr.append(b); ws.append(w_s)
    if np.isfinite(c): arr.append(c); ws.append(w_c)
    if not arr: return np.nan
    ws=np.array(ws)/tot
    return float(np.dot(np.array(arr), ws))

FG2A_pm_b = blend_three(FG2A_pm_r, FG2A_pm_s, FG2A_pm_c, w_recent, w_season, w_career)
FG3A_pm_b = blend_three(FG3A_pm_r, FG3A_pm_s, FG3A_pm_c, w_recent, w_season, w_career)
FTA_pm_b  = blend_three(FTA_pm_r,  FTA_pm_s,  FTA_pm_c,  w_recent, w_season, w_career)

# Per-minute ORB/DRB/AST blends
ORB_pm_r = per_min(logs.get("OREB",np.nan), logs["MIN"], n=n_recent)
DRB_pm_r = per_min(logs.get("DREB",np.nan), logs["MIN"], n=n_recent)
AST_pm_r = per_min(logs.get("AST", np.nan), logs["MIN"], n=n_recent)

ORB_pm_s = per_min(logs.get("OREB",np.nan), logs["MIN"])
DRB_pm_s = per_min(logs.get("DREB",np.nan), logs["MIN"])
AST_pm_s = per_min(logs.get("AST", np.nan), logs["MIN"])

ORB_c_pg = career_pg_counting_stat(career_raw,"OREB")
DRB_c_pg = career_pg_counting_stat(career_raw,"DREB")
AST_c_pg = career_pg_counting_stat(career_raw,"AST")

ORB_pm_c = safe_div(ORB_c_pg, MIN_c_pg)
DRB_pm_c = safe_div(DRB_c_pg, MIN_c_pg)
AST_pm_c = safe_div(AST_c_pg, MIN_c_pg)

ORB_pm_b = blend_three(ORB_pm_r, ORB_pm_s, ORB_pm_c, w_recent, w_season, w_career)
DRB_pm_b = blend_three(DRB_pm_r, DRB_pm_s, DRB_pm_c, w_recent, w_season, w_career)
AST_pm_b = blend_three(AST_pm_r, AST_pm_s, AST_pm_c, w_recent, w_season, w_career)

# Opponent adjustments (pace to attempts; category-specific multipliers)
FG2A_pm_adj = (FG2A_pm_b or 0.0) * adj["pace_factor"]
FG3A_pm_adj = (FG3A_pm_b or 0.0) * adj["pace_factor"] * adj["m_3pa"]
FTA_pm_adj  = (FTA_pm_b  or 0.0) * adj["pace_factor"] * adj["m_fta"]

ORB_pm_adj  = (ORB_pm_b  or 0.0) * adj["pace_factor"] * adj["m_oreb"]
DRB_pm_adj  = (DRB_pm_b  or 0.0) * adj["pace_factor"] * adj["m_dreb"]
AST_pm_adj  = (AST_pm_b  or 0.0) * adj["pace_factor"] * adj["m_ast"]

# Shooting% blends
FG2_PCT_r = pct(logs["FG2M"], logs["FG2A"], n_recent)
FG3_PCT_r = pct(logs["FG3M"], logs["FG3A"], n_recent)
FT_PCT_r  = pct(logs["FTM"],  logs["FTA"],  n_recent)

FG2_PCT_s = pct(logs["FG2M"], logs["FG2A"])
FG3_PCT_s = pct(logs["FG3M"], logs["FG3A"])
FT_PCT_s  = pct(logs["FTM"],  logs["FTA"])

FG2M_tot = (career_raw.get("FGM", 0) - career_raw.get("FG3M", 0))
FG2A_tot = (career_raw.get("FGA", 0) - career_raw.get("FG3A", 0))
if "FG2M" not in career_raw.columns: career_raw = career_raw.assign(FG2M=FG2M_tot)
if "FG2A" not in career_raw.columns: career_raw = career_raw.assign(FG2A=FG2A_tot)
FG2_PCT_c = career_pct(career_raw,"FG2M","FG2A")
FG3_PCT_c = career_pct(career_raw,"FG3M","FG3A")
FT_PCT_c  = career_pct(career_raw,"FTM","FTA")

FG2_p = blend_three(FG2_PCT_r, FG2_PCT_s, FG2_PCT_c, w_recent, w_season, w_career)
FG3_p = blend_three(FG3_PCT_r, FG3_PCT_s, FG3_PCT_c, w_recent, w_season, w_career)
FT_p  = blend_three(FT_PCT_r,  FT_PCT_s,  FT_PCT_c,  w_recent, w_season, w_career)

# Point-estimate totals
FG2A_proj = max(0.0, FG2A_pm_adj * MIN_proj)
FG3A_proj = max(0.0, FG3A_pm_adj * MIN_proj)
FTA_proj  = max(0.0, FTA_pm_adj  * MIN_proj)

FG2M_proj = FG2A_proj * float(np.clip(FG2_p if np.isfinite(FG2_p) else 0.5, 0, 1))
FG3M_proj = FG3A_proj * float(np.clip(FG3_p if np.isfinite(FG3_p) else 0.35, 0, 1))
FTM_proj  = FTA_proj  * float(np.clip(FT_p  if np.isfinite(FT_p)  else 0.78, 0, 1))

PTS_proj = 2.0*FG2M_proj + 3.0*FG3M_proj + 1.0*FTM_proj

ORB_proj = max(0.0, ORB_pm_adj * MIN_proj)
DRB_proj = max(0.0, DRB_pm_adj * MIN_proj)
REB_proj = ORB_proj + DRB_proj

AST_proj = max(0.0, AST_pm_adj * MIN_proj)

PRA_proj = PTS_proj + REB_proj + AST_proj

# Ranges: empirical per-minute SD from chosen window -> scale by minutes (50% most-likely)
def per_min_sd(series_num):
    s = pd.to_numeric(series_num, errors="coerce")
    m = pd.to_numeric(logs["MIN"], errors="coerce")
    v = (s/m).replace([np.inf,-np.inf], np.nan).head(n_recent).dropna()
    return float(v.std(ddof=1)) if len(v) >= 3 else np.nan

PTS_pm_sd = per_min_sd(logs["PTS"])
REB_pm_sd = per_min_sd(logs["REB"])
AST_pm_sd = per_min_sd(logs["AST"])

def make_range(mu, sd_pm):
    if not np.isfinite(sd_pm):
        return (max(0.0, mu*0.9), mu*1.1)  # ±10% fallback
    sd = sd_pm * MIN_proj
    hw = half_width_for_coverage(sd, coverage=0.50)
    lo, hi = max(0.0, mu - hw), max(0.0, mu + hw)
    return (lo, hi)

PTS_range = make_range(PTS_proj, PTS_pm_sd)
REB_range = make_range(REB_proj, REB_pm_sd)
AST_range = make_range(AST_proj, AST_pm_sd)

# ----------------------- KPI Row -----------------------
kpi_stats = ["PTS","REB","AST","MIN","3PM"]
if "3PM" not in logs.columns:
    logs["3PM"] = logs.get("FG3M", np.nan)

recent_mean = logs[["PTS","REB","AST","MIN","3PM"]].head(n_recent).mean(numeric_only=True)
season_mean_vals = logs[["PTS","REB","AST","MIN","3PM"]].mean(numeric_only=True)

cols = st.columns(5)
labels = ["PTS","REB","AST","MIN","3PM"]
for i, s in enumerate(labels):
    fs = form_score(logs[s], k=n_recent) if s in logs.columns else 50.0
    delta = (recent_mean.get(s, np.nan) - season_mean_vals.get(s, np.nan)) if s in recent_mean.index else np.nan
    cols[i].metric(
        label=f"{s} (L{n_recent if n_choice!='Season' else 'SZN'})",
        value=f"{recent_mean.get(s, np.nan):.1f}" if np.isfinite(recent_mean.get(s, np.nan)) else "—",
        delta=f"{delta:+.1f} vs SZN • Form {int(fs)}" if np.isfinite(delta) else f"Form {int(fs)}"
    )

# ----------------------- Recent Trends (window = user choice) -----------------------
win_len = n_recent
st.markdown(f"### Recent Trends (Last {win_len if n_choice!='Season' else 'Season'} Games)")
trend_cols = [c for c in ["MIN","PTS","REB","AST","PRA","FG3M"] if c in logs.columns]
trend_df = logs[["GAME_DATE"] + trend_cols].head(win_len).sort_values("GAME_DATE")
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
    "Value": [FG2A_proj, FG2M_proj, FG2_p, FG3A_proj, FG3M_proj, FG3_p, FTA_proj, FTM_proj, FT_p, PTS_proj]
}).round(2)
st.dataframe(pts_block.style.format({"Value": "{:.2f}"}), use_container_width=True, height=_auto_height(pts_block))

# ----------------------- Projection Summary -----------------------
st.markdown("### Projection Summary")
out = pd.DataFrame({
    "Stat": ["MIN","PTS","REB","AST","PRA","2PM","2PA","3PM","3PA","OREB","DREB"],
    "Proj": [MIN_proj, PTS_proj, REB_proj, AST_proj, PRA_proj, FG2M_proj, FG2A_proj, FG3M_proj, FG3A_proj, ORB_proj, DRB_proj],
    "Range (50%)": ["—",
                    f"{PTS_range[0]:.1f}–{PTS_range[1]:.1f}",
                    f"{REB_range[0]:.1f}–{REB_range[1]:.1f}",
                    f"{AST_range[0]:.1f}–{AST_range[1]:.1f}",
                    "—","—","—","—","—","—","—"]
})
out = out.round(2)
summary_indexed = out.set_index("Stat")
render_summary_table(summary_indexed)

# ----------------------- Compare Windows (Career vs Season vs L5/L10/L20) -----------------------
st.markdown("### Compare Windows (Career vs Season vs L5/L10/L20)")
# Include 3PM now
if "3PM" not in logs.columns:
    logs["3PM"] = logs.get("FG3M", np.nan)
kpi_existing = [c for c in ["PTS","REB","AST","MIN","3PM"] if c in logs.columns]
L5  = window_avg(logs, 5,  kpi_existing)
L10 = window_avg(logs, 10, kpi_existing)
L20 = window_avg(logs, 20, kpi_existing)
season_avg_vals = logs[kpi_existing].mean(numeric_only=True)

career_pg = {
    "PTS": career_pg_counting_stat(career_raw, "PTS"),
    "REB": career_pg_counting_stat(career_raw, "REB"),
    "AST": career_pg_counting_stat(career_raw, "AST"),
    "MIN": career_pg_counting_stat(career_raw, "MIN"),
    "3PM": career_pg_counting_stat(career_raw, "FG3M"),
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

# Averages row (two decimals)
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

    # Averages row (two decimals)
    avg_vs = vs5_display[int_cols_vs].mean(numeric_only=True).to_frame().T.round(2)
    avg_vs.index = ["Average (vs Opp)"]
    st.dataframe(
        avg_vs.style.format("{:.2f}"),
        use_container_width=True,
        height=_auto_height(avg_vs)
    )

st.caption(
    "Notes: Opponent metrics (DEF Rating, Points Allowed, Pace) come from NBA Advanced/Opponent tables "
    "with preseason filtered out via the real regular-season start date. "
    "Minutes projection uses outlier-trim + recency/season/career blending. "
    "Ranges for PTS/REB/AST are 50% most-likely intervals from empirical per-minute variance."
)

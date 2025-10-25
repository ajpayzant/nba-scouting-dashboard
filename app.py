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
    def_factor  = league_def / max(def_rating, 1e-9)      # lower DEF_RATING => tougher => def_factor > 1
    pace_factor = pace_val / max(league_pace_mean, 1e-9)  # faster pace => more possessions
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

# ---------- helpers for rendering & ordinal ranks ----------
def _auto_height(df: pd.DataFrame, row_px: int = 34, header_px: int = 38, max_px: int = 900) -> int:
    rows = max(len(df), 1)
    return min(max_px, header_px + row_px * rows + 8)

def render_summary_table(df_indexed: pd.DataFrame):
    """Format to 2 decimals and render with a height that avoids vertical scrolling."""
    styler = df_indexed.style.format("{:.2f}")
    h = _auto_height(df_indexed)
    st.dataframe(styler, use_container_width=True, height=h)

def ordinal(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        return "—"
    return "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

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
def get_all_players_df():
    """All NBA players (active + inactive). Prospects not yet in NBA DB won't be present."""
    return pd.DataFrame(static_players.get_players())

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_teams_static_df():
    return pd.DataFrame(static_teams.get_teams())

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_team_tables(season: str):
    """
    Source of truth for opponent context:
      - Advanced team table: PACE, DEF_RATING (Regular Season only)
      - Opponent team table: OPP_PTS (as PTS_ALLOWED)
      - Base table: OREB (for rebound adjustments)
    """
    # Advanced (PACE, DEF_RATING)
    df_adv_all = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]

    # Opponent (Points Allowed)
    df_opp_all = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Opponent",
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]

    # Base (OREB for rebounding context)
    df_base_all = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]

    # Keep NBA teams only
    df_adv  = df_adv_all[df_adv_all["TEAM_ID"].apply(is_nba_team_id)].copy()
    df_opp  = df_opp_all[df_opp_all["TEAM_ID"].apply(is_nba_team_id)].copy()
    df_base = df_base_all[df_base_all["TEAM_ID"].apply(is_nba_team_id)].copy()

    # Select / rename columns
    keep_adv = ["TEAM_ID","TEAM_NAME","PACE","DEF_RATING","DREB_PCT"]
    keep_opp = ["TEAM_ID","OPP_PTS"]
    keep_bas = ["TEAM_ID","OREB"]

    df_adv = df_adv[keep_adv].copy()
    df_opp = df_opp[keep_opp].rename(columns={"OPP_PTS":"PTS_ALLOWED"}).copy()
    df_base = df_base[keep_bas].copy()

    # Merge
    teams_ctx = (
        df_adv.merge(df_opp, on="TEAM_ID", how="left")
              .merge(df_base, on="TEAM_ID", how="left")
              .sort_values("TEAM_NAME")
              .reset_index(drop=True)
    )

    # League means for adjustments
    league_pace_mean = float(df_adv["PACE"].mean())
    league_dreb_pct_mean = float(df_adv["DREB_PCT"].mean()) if "DREB_PCT" in df_adv.columns else np.nan
    league_oreb_mean = float(df_base["OREB"].mean())

    # Ranks: DEF & PTS_ALLOWED asc (lower is better), PACE desc (faster is better)
    teams_ctx["RANK_DEF"]  = teams_ctx["DEF_RATING"].rank(method="min", ascending=True).astype(int)
    teams_ctx["RANK_PA"]   = teams_ctx["PTS_ALLOWED"].rank(method="min", ascending=True).astype(int)
    teams_ctx["RANK_PACE"] = teams_ctx["PACE"].rank(method="min", ascending=False).astype(int)

    return teams_ctx, league_pace_mean, league_dreb_pct_mean, league_oreb_mean

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_player_logs(player_id: int, season: str):
    df = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    if df.empty:
        return df
    # Robust mixed-format parsing
    try:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed", errors="coerce")
    except TypeError:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], infer_datetime_format=True, errors="coerce")
    df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    # derive 2P components
    df["FG3M"] = pd.to_numeric(df.get("FG3M", 0), errors="coerce")
    df["FG3A"] = pd.to_numeric(df.get("FG3A", 0), errors="coerce")
    df["FGM"]  = pd.to_numeric(df.get("FGM",  0), errors="coerce")
    df["FGA"]  = pd.to_numeric(df.get("FGA",  0), errors="coerce")
    df["FG2M"] = (df["FGM"] - df["FG3M"]).clip(lower=0)
    df["FG2A"] = (df["FGA"] - df["FG3A"]).clip(lower=0)
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

def is_rookie_by_cpi(cpi_df: pd.DataFrame) -> bool:
    try:
        if cpi_df.empty:
            return False
        val = cpi_df.get("SEASON_EXP")
        if val is None or len(val) == 0:
            return False
        exp = val.iloc[0]
        return pd.to_numeric(exp, errors="coerce") == 0
    except Exception:
        return False

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
    opp_abbr = (opp_abbr or "").upper()
    for s in seasons_list:
        try:
            df = get_player_logs(player_id, s)
        except Exception:
            continue
        if df.empty or "MATCHUP" not in df.columns:
            continue
        matchup_upper = df["MATCHUP"].astype(str).str.upper()
        mask = matchup_upper.str.contains(rf"\b{re.escape(opp_abbr)}\b", na=False, regex=True)
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
players_df = get_all_players_df()
players_df = players_df[["id","full_name","is_active"]].copy()
players_df = players_df.sort_values("full_name").reset_index(drop=True)

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
        st.info("No matching NBA players found. If you're searching a prospect (e.g., Cooper Flagg), they may not be in the NBA database yet.")
        st.stop()
    player_name = st.selectbox("Player", filtered["full_name"].tolist())
    player_row = filtered.loc[filtered["full_name"] == player_name].iloc[0]
    player_id = int(player_row["id"])
    is_active = bool(player_row.get("is_active", False))

    teams_ctx, league_pace_mean, league_dreb_pct_mean, league_oreb_mean = get_team_tables(season)
    opponent = st.selectbox("Opponent", teams_ctx["TEAM_NAME"].tolist())

    # ---- Window now includes "Season"
    n_recent_choice = st.radio("Recent window", ["Season", 5, 10, 15, 20, 25], horizontal=True, index=1)
    n_recent = n_recent_choice if isinstance(n_recent_choice, int) else None

    w_recent = st.slider("Weight: Recent", 0.0, 1.0, 0.55, 0.05)
    w_season = st.slider("Weight: Season", 0.0, 1.0, 0.30, 0.05)
    w_career = st.slider("Weight: Career", 0.0, 1.0, 0.15, 0.05)

# ----------------------- Data fetch & Rookie/Prospect handling -----------------------
cpi = get_common_player_info(player_id)
rookie = is_rookie_by_cpi(cpi)

if cpi.empty and not is_active:
    st.error(
        f"**{player_name}** is not yet in the NBA stats database. "
        "Prospects (e.g., Cooper Flagg, Dylan Harper, V.J. Edgecombe, Hugo Gonzalez) "
        "become available once the NBA adds them to the official database."
    )
    st.stop()

season_used = best_season_for_player(player_id, season, SEASONS)
if rookie and season_used != season:
    st.warning(f"{player_name} appears to be a rookie. Showing the only season with available logs: **{season_used}**.")
logs = get_player_logs(player_id, season_used)
if logs.empty:
    if rookie or not is_active:
        st.error("No NBA game logs available yet for this player.")
    else:
        st.error("No game logs found for this player/season.")
    st.stop()

career_raw = get_player_career(player_id)

# ----------------------- Opponent Context -----------------------
opp_row = teams_ctx.loc[teams_ctx["TEAM_NAME"] == opponent].iloc[0]
opp_def       = float(opp_row["DEF_RATING"])
opp_pace      = float(opp_row["PACE"])
opp_pts_allow = float(opp_row["PTS_ALLOWED"]) if "PTS_ALLOWED" in opp_row.index and pd.notna(opp_row["PTS_ALLOWED"]) else np.nan

# ranks (precomputed in teams_ctx)
rank_def  = int(opp_row.get("RANK_DEF",  np.nan)) if pd.notna(opp_row.get("RANK_DEF",  np.nan)) else None
rank_pace = int(opp_row.get("RANK_PACE", np.nan)) if pd.notna(opp_row.get("RANK_PACE", np.nan)) else None
rank_pa   = int(opp_row.get("RANK_PA",   np.nan)) if pd.notna(opp_row.get("RANK_PA",   np.nan)) else None

opp_dreb_pct = float(opp_row["DREB_PCT"]) if "DREB_PCT" in teams_ctx.columns else np.nan
opp_oreb     = float(opp_row["OREB"])

AdjFactor = opponent_adjustment(opp_def, opp_pace, LEAGUE_DEF_REF, league_pace_mean)
ORB_adj = (league_dreb_pct_mean / opp_dreb_pct) if np.isfinite(league_dreb_pct_mean) and np.isfinite(opp_dreb_pct) and opp_dreb_pct > 0 else 1.0
DRB_adj = (league_oreb_mean / opp_oreb)       if np.isfinite(league_oreb_mean)     and np.isfinite(opp_oreb)     and opp_oreb > 0         else 1.0

# ----------------------- Player header (Age • Position • Seasons • GP) -----------------------
left, right = st.columns([2, 1.4])  # slightly wider right pane for readability
with left:
    st.subheader(f"{player_name} — {season_used}")
    age = pos = exp = None
    if not cpi.empty:
        if "AGE" in cpi.columns and len(cpi["AGE"]) > 0:
            age = cpi["AGE"].iloc[0]
        if "POSITION" in cpi.columns and len(cpi["POSITION"]) > 0:
            pos = cpi["POSITION"].iloc[0] or None
        if "SEASON_EXP" in cpi.columns and len(cpi["SEASON_EXP"]) > 0:
            exp = cpi["SEASON_EXP"].iloc[0]
    gp_this_season = int(len(logs))
    meta = []
    if age is not None and str(age) != "nan":
        try: meta.append(f"Age: {int(float(age))}")
        except: pass
    meta.append(f"Position: {pos if pos else '—'}")
    if exp is not None and str(exp) != "nan":
        try: meta.append(f"Seasons: {int(float(exp))}")
        except: meta.append(f"Seasons: —")
    else:
        meta.append("Seasons: —")
    meta.append(f"GP ({season_used}): {gp_this_season}")
    st.caption(" • ".join(meta))

with right:
    st.markdown(f"**Opponent: {opponent}**")

    # Centered, responsive stat cards with smaller italic rank
    CARD_CSS = """
    <style>
      .statgrid {display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:10px;}
      @media (max-width: 1100px) { .statgrid {grid-template-columns: 1fr;} }
      .statcard {
        border:1px solid #e6e6e6; border-radius:12px; padding:10px 12px;
        box-sizing:border-box; text-align:center;
      }
      .statcard .title { font-weight:600; font-size:0.95rem; margin-bottom:4px; }
      .statcard .value { display:flex; align-items:baseline; justify-content:center; gap:6px; flex-wrap:wrap; }
      .statcard .value .num { font-size:1.25rem; line-height:1.6rem; }
      .statcard .value .rank { font-size:0.95rem; font-style:italic; opacity:0.85; }
    </style>
    """
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    def stat_card(title: str, value, rank):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            num_html = "—"
            rank_html = ""
        else:
            num_html = f"{value:.1f}"
            rank_html = f"<span class='rank'>({ordinal(rank)})</span>" if rank else ""
        st.markdown(
            f"""
            <div class="statcard">
              <div class="title">{title}</div>
              <div class="value"><span class="num">{num_html}</span>{rank_html}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="statgrid">', unsafe_allow_html=True)
    stat_card("DEF Rating", opp_def, rank_def)
    stat_card("Points Allowed", opp_pts_allow, rank_pa)
    stat_card("Pace", opp_pace, rank_pace)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Baselines & blends -----------------------
def _series_head(s: pd.Series, n_opt):
    return s if n_opt is None else s.head(n_opt)

def mean_recent(series, n_opt=n_recent):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if not len(s):
        return np.nan
    if n_opt is None:
        return float(s.mean())
    return float(s.iloc[:min(n_opt, len(s))].mean())

def season_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan

# Minutes: blend of recent/season/career (improved minute projection)
MIN_recent = mean_recent(logs["MIN"], n_recent)
MIN_season = season_mean(logs["MIN"])
MIN_career_pg = career_pg_counting_stat(career_raw, "MIN")

def blend_vals_local(r, s, c):
    total = max(w_recent + w_season + w_career, 1e-9)
    return (w_recent/total)*(r if np.isfinite(r) else 0.0) + \
           (w_season/total)*(s if np.isfinite(s) else 0.0) + \
           (w_career/total)*(c if np.isfinite(c) else 0.0)

MIN_proj = blend_vals_local(MIN_recent, MIN_season, MIN_career_pg)

FG2A_per_min_recent = pm_ratio(_series_head(logs["FG2A"], n_recent), _series_head(logs["MIN"], n_recent))
FG3A_per_min_recent = pm_ratio(_series_head(logs["FG3A"], n_recent), _series_head(logs["MIN"], n_recent))
FTA_per_min_recent  = pm_ratio(_series_head(logs["FTA"],  n_recent), _series_head(logs["MIN"], n_recent))

FG2A_per_min_season = pm_ratio(logs["FG2A"], logs["MIN"])
FG3A_per_min_season = pm_ratio(logs["FG3A"], logs["MIN"])
FTA_per_min_season  = pm_ratio(logs["FTA"],  logs["MIN"])

FG2A_career_pg = safe_div((career_pg_counting_stat(career_raw, "FGA") - career_pg_counting_stat(career_raw, "FG3A")), 1.0)
FG3A_career_pg = career_pg_counting_stat(career_raw, "FG3A")
FTA_career_pg  = career_pg_counting_stat(career_raw, "FTA")

FG2A_per_min_career = safe_div(FG2A_career_pg, MIN_career_pg)
FG3A_per_min_career = safe_div(FG3A_career_pg, MIN_career_pg)
FTA_per_min_career  = safe_div(FTA_career_pg,  MIN_career_pg)

def pct(series_m, series_a, n_opt=None):
    m = pd.to_numeric(series_m, errors="coerce")
    a = pd.to_numeric(series_a, errors="coerce")
    if n_opt is not None:
        m, a = m.head(n_opt), a.head(n_opt)
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

def blend_vals(r, s, c):
    return blend_vals_local(r, s, c)

FG2A_per_min_blend = blend_vals(FG2A_per_min_recent, FG2A_per_min_season, FG2A_per_min_career)
FG3A_per_min_blend = blend_vals(FG3A_per_min_recent, FG3A_per_min_season, FG3A_per_min_career)
FTA_per_min_blend  = blend_vals(FTA_per_min_recent,  FTA_per_min_season,  FTA_per_min_career)

FG2_PCT_blend = blend_vals(FG2_PCT_recent, FG2_PCT_season, FG2_PCT_career)
FG3_PCT_blend = blend_vals(FG3_PCT_recent, FG3_PCT_season, FG3_PCT_career)
FT_PCT_blend  = blend_vals(FT_PCT_recent,  FT_PCT_season,  FT_PCT_career)

FG2A_proj = max(0.0, (FG2A_per_min_blend * MIN_proj) * AdjFactor) if np.isfinite(FG2A_per_min_blend) else 0.0
FG3A_proj = max(0.0, (FG3A_per_min_blend * MIN_proj) * AdjFactor) if np.isfinite(FG3A_per_min_blend) else 0.0
FTA_proj  = max(0.0, (FTA_per_min_blend  * MIN_proj) * AdjFactor) if np.isfinite(FTA_per_min_blend)  else 0.0

FG2M_proj = FG2A_proj * float(np.clip(FG2_PCT_blend if np.isfinite(FG2_PCT_blend) else 0.5, 0, 1))
FG3M_proj = FG3A_proj * float(np.clip(FG3_PCT_blend if np.isfinite(FG3_PCT_blend) else 0.35, 0, 1))
FTM_proj  = FTA_proj  * float(np.clip(FT_PCT_blend  if np.isfinite(FT_PCT_blend)  else 0.78, 0, 1))

PTS_proj = 2.0*FG2M_proj + 3.0*FG3M_proj + 1.0*FTM_proj

ORB_per_min_blend = blend_vals(ORB_per_min_recent, ORB_per_min_season, ORB_per_min_career)
DRB_per_min_blend = blend_vals(DRB_per_min_recent, DRB_per_min_season, DRB_per_min_career)

ORB_proj = max(0.0, (ORB_per_min_blend * MIN_proj) * AdjFactor * ORB_adj) if np.isfinite(ORB_per_min_blend) else 0.0
DRB_proj = max(0.0, (DRB_per_min_blend * MIN_proj) * AdjFactor * DRB_adj) if np.isfinite(DRB_per_min_blend) else 0.0
REB_proj = ORB_proj + DRB_proj

AST_per_min_blend = blend_vals(AST_per_min_recent, AST_per_min_season, AST_per_min_career)
AST_proj = max(0.0, (AST_per_min_blend * MIN_proj) * AdjFactor) if np.isfinite(AST_per_min_blend) else 0.0

PRA_proj = PTS_proj + REB_proj + AST_proj

# ----------------------- KPI Row -----------------------
kpi_stats = ["PTS","REB","AST","MIN"]
recent_mean = (_series_head(logs[kpi_stats], n_recent)).mean(numeric_only=True)
season_mean_vals = logs[kpi_stats].mean(numeric_only=True)
cols = st.columns(len(kpi_stats))
for i, s in enumerate(kpi_stats):
    if s not in logs.columns and s != "MIN":
        continue
    fs = form_score(_series_head(logs[s], n_recent) if s in logs.columns else logs.get(s, pd.Series(dtype=float)), k=n_recent if n_recent else 5)
    delta = (recent_mean.get(s, np.nan) - season_mean_vals.get(s, np.nan)) if s in recent_mean.index else np.nan
    label = f"{s} ({'Season' if n_recent is None else f'L{n_recent}'})"
    cols[i].metric(
        label=label,
        value=f"{recent_mean.get(s, np.nan):.1f}" if np.isfinite(recent_mean.get(s, np.nan)) else "—",
        delta=f"{delta:+.1f} vs SZN • Form {int(fs)}" if np.isfinite(delta) else f"Form {int(fs)}"
    )

# ----------------------- Recent Trends -----------------------
if n_recent is None:
    st.markdown("### Season Trends")
    trend_n = len(logs)
else:
    st.markdown(f"### Recent Trends (Last {n_recent} Games)")
    trend_n = n_recent

trend_cols = [c for c in ["MIN","PTS","REB","AST","PRA","FG3M"] if c in logs.columns]
trend_df = logs[["GAME_DATE"] + trend_cols].head(trend_n).sort_values("GAME_DATE")
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

# ----------------------- Projection Summary (normal rows; no highlight) -----------------------
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
kpi_existing = [c for c in ["PTS","REB","AST","MIN","FG3M"] if c in logs.columns]
L5  = window_avg(logs, 5,  kpi_existing)
L10 = window_avg(logs, 10, kpi_existing)
L20 = window_avg(logs, 20, kpi_existing)
season_avg_vals = logs[kpi_existing].mean(numeric_only=True)

career_pg = {
    "PTS": career_pg_counting_stat(career_raw, "PTS"),
    "REB": career_pg_counting_stat(career_raw, "REB"),
    "AST": career_pg_counting_stat(career_raw, "AST"),
    "MIN": career_pg_counting_stat(career_raw, "MIN"),
    "FG3M": career_pg_counting_stat(career_raw, "FG3M"),
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

st.dataframe(
    display_last5.style.format(fmt_map),
    use_container_width=True,
    height=_auto_height(display_last5)
)

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
opp_abbr = opp_abbr.upper()

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

    st.dataframe(
        vs5_display.style.format(fmt_map_vs),
        use_container_width=True,
        height=_auto_height(vs5_display)
    )

    avg_vs = vs5_display[int_cols_vs].mean(numeric_only=True).to_frame().T.round(2)
    avg_vs.index = ["Average (vs Opp)"]
    st.dataframe(
        avg_vs.style.format("{:.2f}"),
        use_container_width=True,
        height=_auto_height(avg_vs)
    )

st.caption(
    "Notes: Opponent context now pulls DEF Rating & Pace from Advanced team stats, and Points Allowed from Opponent team stats — all Regular Season only. "
    "Tables are formatted to two decimals and auto-sized to avoid vertical scrolling. "
    "Career values use proper per-game aggregation from season totals (sum totals / sum GP). "
    "If current-season logs are not yet available for a player, the app falls back to the most recent season with data. "
    "Rookies are restricted to their available season. Trend lines follow the selected window (or Season)."
)

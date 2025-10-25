import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from nba_api.stats.static import players as static_players, teams as static_teams
from nba_api.stats.endpoints import (
    playergamelog,
    playercareerstats,
    leaguedashteamstats,
)

# ----------------------- Streamlit Setup -----------------------
st.set_page_config(page_title="NBA Scouting Dashboard", layout="wide")
st.title("🏀 NBA Player Scouting — Component Projections (PTS/REB/AST)")

# ----------------------- Config -----------------------
CACHE_HOURS = 12
LEAGUE_DEF_REF = 112.0   # coarse league baseline for defense scaling

DEFAULT_SEASON = "2024-25"
SEASONS = [
    "2024-25", "2023-24", "2022-23", "2021-22", "2020-21",
    "2019-20", "2018-19", "2017-18"
]

# ----------------------- Utilities -----------------------
def is_nba_team_id(x) -> bool:
    """Filter out non-NBA IDs (e.g., G League / WNBA) that sometimes appear in tables."""
    try:
        return str(int(x)).startswith("161061")
    except Exception:
        return False

def possessions_proxy_row(row: pd.Series) -> float:
    """
    Possessions approximation using per-game team stats:
      Poss ≈ FGA - OREB + TOV + 0.44 * FTA
    """
    FGA  = pd.to_numeric(row.get("FGA", 0), errors="coerce")
    OREB = pd.to_numeric(row.get("OREB", 0), errors="coerce")
    TOV  = pd.to_numeric(row.get("TOV", 0), errors="coerce")
    FTA  = pd.to_numeric(row.get("FTA", 0), errors="coerce")
    return float(FGA - OREB + TOV + 0.44 * FTA)

def pace_proxy_row(row: pd.Series) -> float:
    """Pace proxy per 48 minutes: Possessions * (48 / MIN)."""
    poss = possessions_proxy_row(row)
    MIN  = pd.to_numeric(row.get("MIN", 48), errors="coerce")
    if not np.isfinite(MIN) or MIN <= 0:
        MIN = 48.0
    return float(poss * (48.0 / MIN))

def gp_weighted_mean(df: pd.DataFrame, value_col: str, gp_col: str = "GP") -> float:
    """GP-weighted per-game career mean across seasons."""
    if value_col not in df.columns or gp_col not in df.columns:
        return np.nan
    s_val = pd.to_numeric(df[value_col], errors="coerce")
    s_gp  = pd.to_numeric(df[gp_col], errors="coerce")
    mask = s_val.notna() & s_gp.notna() & (s_gp > 0)
    if not mask.any():
        return np.nan
    return float((s_val[mask] * s_gp[mask]).sum() / s_gp[mask].sum())

def safe_div(num, den, default=np.nan):
    try:
        val = num / den if den not in (0, None, np.nan) else default
        if not np.isfinite(val):
            return default
        return val
    except Exception:
        return default

def form_score(series: pd.Series, k: int = 5) -> float:
    """0–100 'hot/cold' score comparing last k to whole-season mean/std."""
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
    """Combine defense & pace into one scalar factor."""
    def_factor  = league_def / max(def_rating, 1e-9)
    pace_factor = pace_val / max(league_pace_mean, 1e-9)
    return float(def_factor * pace_factor)

def window_avg(df: pd.DataFrame, n: int, cols) -> pd.Series:
    if len(df) < 1:
        return pd.Series({c: np.nan for c in cols})
    return df.iloc[:min(len(df), n)][cols].mean(numeric_only=True)

# ----------------------- Caching Data -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600)
def get_active_players_df():
    return pd.DataFrame(static_players.get_active_players())

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_teams_static_df():
    # id, full_name, abbreviation, etc.
    return pd.DataFrame(static_teams.get_teams())

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_team_tables(season: str):
    """
    Returns:
      teams_ctx: DEF_RATING, DREB_PCT, PACE_PROXY, OREB (NBA-only)
      league_pace_mean, league_dreb_pct_mean, league_oreb_mean
    """
    # Defense table (DEF_RATING, DREB_PCT present)
    df_def_all = leaguedashteamstats.LeagueDashTeamStats(
        season=season, per_mode_detailed="PerGame", measure_type_detailed_defense="Defense"
    ).get_data_frames()[0]

    # Base team per-game (FGA/FTA/OREB/TOV/MIN → pace proxy)
    df_base_all = leaguedashteamstats.LeagueDashTeamStats(
        season=season, per_mode_detailed="PerGame"
    ).get_data_frames()[0]

    # NBA-only
    df_def  = df_def_all[df_def_all["TEAM_ID"].apply(is_nba_team_id)].copy()
    df_base = df_base_all[df_base_all["TEAM_ID"].apply(is_nba_team_id)].copy()

    # select cols
    keep_def  = ["TEAM_ID","TEAM_NAME","DEF_RATING","DREB_PCT"]
    keep_base = ["TEAM_ID","TEAM_NAME","MIN","FGA","FTA","OREB","TOV"]
    df_def  = df_def[keep_def].copy()
    df_base = df_base[keep_base].copy()

    # pace proxy
    df_base["PACE_PROXY"] = df_base.apply(pace_proxy_row, axis=1)
    league_pace_mean = float(df_base["PACE_PROXY"].mean())
    league_dreb_pct_mean = float(df_def["DREB_PCT"].mean()) if "DREB_PCT" in df_def.columns else np.nan
    league_oreb_mean = float(df_base["OREB"].mean())

    teams_ctx = (
        df_def.merge(df_base[["TEAM_ID","PACE_PROXY","OREB"]], on="TEAM_ID", how="left")
              .sort_values("TEAM_NAME")
              .reset_index(drop=True)
    )

    return teams_ctx, league_pace_mean, league_dreb_pct_mean, league_oreb_mean

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
    return df

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_player_career(player_id: int):
    return playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]

# ----------------------- Sidebar Controls -----------------------
players_df = get_active_players_df().sort_values("full_name")
teams_static_df = get_teams_static_df()

with st.sidebar:
    st.header("Controls")
    season = st.selectbox("Season", SEASONS, index=SEASONS.index(DEFAULT_SEASON) if DEFAULT_SEASON in SEASONS else 0)
    q = st.text_input("Search player", value="Jayson Tatum")
    filtered = players_df[players_df["full_name"].str.contains(q, case=False, na=False)]
    if filtered.empty:
        st.stop()
    player_name = st.selectbox("Pick player", filtered["full_name"].tolist())
    player_id = int(filtered.loc[filtered["full_name"] == player_name, "id"].iloc[0])

    # get team context so we can populate opponent list
    teams_ctx, league_pace_mean, league_dreb_pct_mean, league_oreb_mean = get_team_tables(season)
    opponent = st.selectbox("Opponent", teams_ctx["TEAM_NAME"].tolist())

    # Windows & weights
    n_recent = st.radio("Recent window (games)", [5, 10, 20, 25], horizontal=True, index=0)
    w_recent = st.slider("Weight: Recent", 0.0, 1.0, 0.55, 0.05)
    w_season = st.slider("Weight: Season", 0.0, 1.0, 0.30, 0.05)
    w_career = st.slider("Weight: Career", 0.0, 1.0, 0.15, 0.05)

# ----------------------- Pull Player Data -----------------------
logs = get_player_logs(player_id, season)
if logs.empty:
    st.error("No game logs found for this player/season.")
    st.stop()

career_raw = get_player_career(player_id)

# ----------------------- Opponent Context -----------------------
opp_row = teams_ctx.loc[teams_ctx["TEAM_NAME"] == opponent].iloc[0]
opp_def      = float(opp_row["DEF_RATING"])
opp_pace     = float(opp_row["PACE_PROXY"])
opp_dreb_pct = float(opp_row["DREB_PCT"]) if "DREB_PCT" in teams_ctx.columns else np.nan
opp_oreb     = float(opp_row["OREB"])

AdjFactor = opponent_adjustment(
    def_rating=opp_def,
    pace_val=opp_pace,
    league_def=LEAGUE_DEF_REF,
    league_pace_mean=league_pace_mean
)

# Rebound-specific adjusters
ORB_adj = (league_dreb_pct_mean / opp_dreb_pct) if np.isfinite(league_dreb_pct_mean) and np.isfinite(opp_dreb_pct) and opp_dreb_pct > 0 else 1.0
DRB_adj = (league_oreb_mean / opp_oreb)       if np.isfinite(league_oreb_mean)     and np.isfinite(opp_oreb)     and opp_oreb > 0         else 1.0

# ----------------------- Baselines & Blends -----------------------
def mean_recent(series, n=n_recent):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.iloc[:min(n, len(s))].mean()) if len(s) else np.nan

def season_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan

def pm_ratio(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    val = (num / den).replace([np.inf, -np.inf], np.nan)
    return float(val.mean()) if len(val.dropna()) else np.nan

def blend_vals(r, s, c, wr=w_recent, ws=w_season, wc=w_career):
    total = max(wr + ws + wc, 1e-9)
    return (wr/total)*r + (ws/total)*s + (wc/total)*c

# Minutes
MIN_recent = mean_recent(logs["MIN"], n_recent)
MIN_season = season_mean(logs["MIN"])
MIN_career_pg = gp_weighted_mean(career_raw, "MIN")
MIN_proj = float(MIN_recent if np.isfinite(MIN_recent) else MIN_season)

# Attempts per minute (recent/season/career)
FG2A_per_min_recent = pm_ratio(logs["FG2A"].head(n_recent), logs["MIN"].head(n_recent))
FG3A_per_min_recent = pm_ratio(logs["FG3A"].head(n_recent), logs["MIN"].head(n_recent))
FTA_per_min_recent  = pm_ratio(logs["FTA"].head(n_recent),  logs["MIN"].head(n_recent))

FG2A_per_min_season = pm_ratio(logs["FG2A"], logs["MIN"])
FG3A_per_min_season = pm_ratio(logs["FG3A"], logs["MIN"])
FTA_per_min_season  = pm_ratio(logs["FTA"],  logs["MIN"])

FG2A_career_pg = gp_weighted_mean(career_raw, "FGA") - gp_weighted_mean(career_raw, "FG3A")
FG3A_career_pg = gp_weighted_mean(career_raw, "FG3A")
FTA_career_pg  = gp_weighted_mean(career_raw, "FTA")

FG2A_per_min_career = safe_div(FG2A_career_pg, MIN_career_pg)
FG3A_per_min_career = safe_div(FG3A_career_pg, MIN_career_pg)
FTA_per_min_career  = safe_div(FTA_career_pg,  MIN_career_pg)

# Percentages (recent/season/career)
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

# Career percentages via GP-weighted makes/attempts
FG2A_career_pg_full = gp_weighted_mean(career_raw.assign(FG2A=(career_raw["FGA"]-career_raw["FG3A"])), "FG2A")
FG2M_career_pg_full = gp_weighted_mean(career_raw.assign(FG2M=(career_raw["FGM"]-career_raw["FG3M"])), "FG2M")
FG2_PCT_career = safe_div(FG2M_career_pg_full, FG2A_career_pg_full)

FG3_PCT_career = safe_div(gp_weighted_mean(career_raw, "FG3M"), gp_weighted_mean(career_raw, "FG3A"))
FT_PCT_career  = safe_div(gp_weighted_mean(career_raw, "FTM"),  gp_weighted_mean(career_raw, "FTA"))

# Rebounds per minute (ORB/DRB) + assists per minute
def per_min(series_num, series_min, n=None):
    num = pd.to_numeric(series_num, errors="coerce")
    den = pd.to_numeric(series_min, errors="coerce")
    if n is not None:
        num, den = num.head(n), den.head(n)
    val = (num / den).replace([np.inf, -np.inf], np.nan)
    return float(val.mean()) if len(val.dropna()) else np.nan

ORB_per_min_recent = per_min(logs.get("OREB", np.nan), logs["MIN"], n=n_recent)
DRB_per_min_recent = per_min(logs.get("DREB", np.nan), logs["MIN"], n=n_recent)
AST_per_min_recent = per_min(logs.get("AST",  np.nan), logs["MIN"], n=n_recent)

ORB_per_min_season = per_min(logs.get("OREB", np.nan), logs["MIN"])
DRB_per_min_season = per_min(logs.get("DREB", np.nan), logs["MIN"])
AST_per_min_season = per_min(logs.get("AST",  np.nan), logs["MIN"])

ORB_career_pg = gp_weighted_mean(career_raw, "OREB")
DRB_career_pg = gp_weighted_mean(career_raw, "DREB")
AST_career_pg = gp_weighted_mean(career_raw, "AST")

ORB_per_min_career = safe_div(ORB_career_pg, MIN_career_pg)
DRB_per_min_career = safe_div(DRB_career_pg, MIN_career_pg)
AST_per_min_career = safe_div(AST_career_pg, MIN_career_pg)

# ----------------------- Projections -----------------------
# Blended attempts per minute
FG2A_per_min_blend = blend_vals(FG2A_per_min_recent, FG2A_per_min_season, FG2A_per_min_career)
FG3A_per_min_blend = blend_vals(FG3A_per_min_recent, FG3A_per_min_season, FG3A_per_min_career)
FTA_per_min_blend  = blend_vals(FTA_per_min_recent,  FTA_per_min_season,  FTA_per_min_career)

# Blended percentages
FG2_PCT_blend = blend_vals(FG2_PCT_recent, FG2_PCT_season, FG2_PCT_career)
FG3_PCT_blend = blend_vals(FG3_PCT_recent, FG3_PCT_season, FG3_PCT_career)
FT_PCT_blend  = blend_vals(FT_PCT_recent,  FT_PCT_season,  FT_PCT_career)

# Attempts adjusted by (def x pace) and minutes
FG2A_proj = max(0.0, (FG2A_per_min_blend * MIN_proj) * AdjFactor) if np.isfinite(FG2A_per_min_blend) else 0.0
FG3A_proj = max(0.0, (FG3A_per_min_blend * MIN_proj) * AdjFactor) if np.isfinite(FG3A_per_min_blend) else 0.0
FTA_proj  = max(0.0, (FTA_per_min_blend  * MIN_proj) * AdjFactor) if np.isfinite(FTA_per_min_blend)  else 0.0

# Makes via blended accuracy
FG2M_proj = FG2A_proj * float(np.clip(FG2_PCT_blend if np.isfinite(FG2_PCT_blend) else 0.5, 0, 1))
FG3M_proj = FG3A_proj * float(np.clip(FG3_PCT_blend if np.isfinite(FG3_PCT_blend) else 0.35, 0, 1))
FTM_proj  = FTA_proj  * float(np.clip(FT_PCT_blend  if np.isfinite(FT_PCT_blend)  else 0.78, 0, 1))

PTS_proj = 2.0*FG2M_proj + 3.0*FG3M_proj + 1.0*FTM_proj

# Rebounds (ORB/DRB) with specific adjusters
ORB_per_min_blend = blend_vals(ORB_per_min_recent, ORB_per_min_season, ORB_per_min_career)
DRB_per_min_blend = blend_vals(DRB_per_min_recent, DRB_per_min_season, DRB_per_min_career)

ORB_proj = max(0.0, (ORB_per_min_blend * MIN_proj) * AdjFactor * ORB_adj) if np.isfinite(ORB_per_min_blend) else 0.0
DRB_proj = max(0.0, (DRB_per_min_blend * MIN_proj) * AdjFactor * DRB_adj) if np.isfinite(DRB_per_min_blend) else 0.0
REB_proj = ORB_proj + DRB_proj

# Assists
AST_per_min_blend = blend_vals(AST_per_min_recent, AST_per_min_season, AST_per_min_career)
AST_proj = max(0.0, (AST_per_min_blend * MIN_proj) * AdjFactor) if np.isfinite(AST_per_min_blend) else 0.0

# PRA
PRA_proj = PTS_proj + REB_proj + AST_proj

# --------------- Header & Opponent KPIs ---------------
left, right = st.columns([2, 1])
with left:
    st.subheader(f"{player_name} — {season}")
with right:
    st.metric("Opponent DEF Rating", f"{opp_def:.1f}")
    st.metric("Opponent Pace (proxy)", f"{opp_pace:.1f}")

# KPI Row: show MIN and Form + deltas for PTS/REB/AST
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

# --------------- Trends (Last 20) ---------------
st.markdown("### Recent Trends (Last 20 Games)")
trend_cols = [c for c in ["PTS","REB","AST","MIN"] if c in logs.columns]
trend_df = logs[["GAME_DATE"] + trend_cols].head(20).sort_values("GAME_DATE")
for s in trend_cols:
    chart = (
        alt.Chart(trend_df)
        .mark_line(point=True)
        .encode(x="GAME_DATE:T", y=alt.Y(s, title=s))
        .properties(height=160)
    )
    st.altair_chart(chart, use_container_width=True)

# --------------- Component Points Block ---------------
st.markdown("### Points Component Model (2s / 3s / FTs)")
pts_block = pd.DataFrame({
    "Component": ["2PA","2PM","2P%","3PA","3PM","3P%","FTA","FTM","FT%","PTS"],
    "Value": [FG2A_proj, FG2M_proj, FG2_PCT_blend, FG3A_proj, FG3M_proj, FG3_PCT_blend, FTA_proj, FTM_proj, FT_PCT_blend, PTS_proj]
}).round(2)
st.dataframe(pts_block, use_container_width=True)

# Core projections table
st.markdown("### Projection Summary")
out = pd.DataFrame({
    "Stat": ["MIN","PTS","3PM","REB","ORB","DRB","AST","PRA"],
    "Proj": [MIN_proj, PTS_proj, FG3M_proj, REB_proj, ORB_proj, DRB_proj, AST_proj, PRA_proj]
}).round(2)
st.dataframe(out.set_index("Stat"), use_container_width=True)

# Core bar chart
core_metrics = ["PTS","REB","AST","3PM","PRA"]
core_df = out[out["Stat"].isin(core_metrics)].copy()
if not core_df.empty:
    core_chart = (
        alt.Chart(core_df)
        .mark_bar()
        .encode(x=alt.X("Stat:N", title=""), y=alt.Y("Proj:Q", title="Projected"))
        .properties(height=220)
    )
    st.altair_chart(core_chart, use_container_width=True)

# Rebound breakdown chart
reb_df = out[out["Stat"].isin(["ORB","DRB"])].copy()
if not reb_df.empty:
    reb_chart = (
        alt.Chart(reb_df)
        .mark_bar()
        .encode(x=alt.X("Stat:N", title=""), y=alt.Y("Proj:Q", title="Projected"))
        .properties(height=220)
    )
    st.altair_chart(reb_chart, use_container_width=True)

# --------------- Compare Windows ---------------
st.markdown("### Compare Windows (Career vs Season vs L5/L10/L20)")
kpi_existing = [c for c in ["PTS","REB","AST","MIN"] if c in logs.columns]
L5  = window_avg(logs, 5,  kpi_existing)
L10 = window_avg(logs, 10, kpi_existing)
L20 = window_avg(logs, 20, kpi_existing)
season_avg_vals = logs[kpi_existing].mean(numeric_only=True)

# Career = GP-weighted per-game across seasons
career_means = {}
for s in kpi_existing:
    if s in career_raw.columns:
        career_means[s] = gp_weighted_mean(career_raw, s)
cmp_df = pd.concat(
    {"Career": pd.Series(career_means),
     "Season": season_avg_vals,
     "L5": L5,
     "L10": L10,
     "L20": L20},
    axis=1
).round(2)
st.dataframe(cmp_df, use_container_width=True)

# --------------- Past vs Opponent (this season) ---------------
st.markdown(f"### Past Games vs {opponent} — {season}")
# Map opponent to abbreviation (for MATCHUP)
ts = teams_static_df.rename(columns={"id":"TEAM_ID", "full_name":"TEAM_NAME", "abbreviation":"TEAM_ABBR"})
match = ts.loc[ts["TEAM_NAME"].str.lower() == opponent.lower()]
opp_abbr = match["TEAM_ABBR"].iloc[0] if not match.empty else opponent[:3].upper()

mask = logs["MATCHUP"].str.contains(rf"\b{opp_abbr}\b", na=False, regex=True)
past_vs_opp = logs.loc[mask].copy()
cols_show = [c for c in [
    "GAME_DATE","MATCHUP","WL","MIN","PTS","FG3M","REB","AST","OREB","DREB","FTA","FG2M","FG2A","FG3A"
] if c in past_vs_opp.columns]

if past_vs_opp.empty:
    st.info("No games found vs this opponent for the selected season.")
else:
    st.dataframe(past_vs_opp[cols_show].sort_values("GAME_DATE", ascending=False), use_container_width=True)

st.caption("Notes: Pace uses a possession-based proxy from team per-game stats. ORB and DRB use opponent DREB% and OREB adjusters respectively. Career inputs are GP-weighted per-game.")


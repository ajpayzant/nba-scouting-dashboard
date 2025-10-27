# app.py — NBA Player Scouting Dashboard v2.2 (Stabilized)
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import datetime as dt
import math
import re

from nba_api.stats.static import players as static_players, teams as static_teams
from nba_api.stats.endpoints import (
    playergamelog,
    playercareerstats,
    leaguedashteamstats,
    LeagueDashPlayerStats,
    commonplayerinfo,
    leaguedashteamgamelogs,
)

# ----------------------- Streamlit Setup -----------------------
st.set_page_config(page_title="NBA Player Scouting Dashboard", layout="wide")
st.title("🏀 NBA Player Scouting Dashboard")

# ----------------------- Config -----------------------
CACHE_HOURS = 12
DEFAULT_SEASON = "2025-26"
LEAGUE_DEF_REF = 112.0

# ----------------------- Utils -----------------------
UTC = dt.timezone.utc
def _season_labels(start_year: int, end_year: int):
    # inclusive start, inclusive end
    years = range(start_year, end_year + 1)
    return [f"{y}-{str((y+1)%100).zfill(2)}" for y in years]

# Detect seasons from 2015 through current year label (safe default)
SEASONS = _season_labels(2015, dt.datetime.now(UTC).year)

def is_nba_team_id(x):
    try:
        return str(int(x)).startswith("161061")
    except Exception:
        return False

def safe_div(a, b, default=np.nan):
    try:
        if not np.isfinite(b) or b == 0:
            return default
        v = a / b
        return v if np.isfinite(v) else default
    except Exception:
        return default

def _auto_height(df, row_px=34, header_px=38, max_px=900):
    rows = max(len(df), 1)
    return min(max_px, header_px + row_px * rows + 8)

def _rank_str(series, value, ascending):
    if not np.isfinite(value):
        return "—"
    # rank: 1 is best; ascending=False means bigger is better
    ranks = series.rank(ascending=ascending, method="min")
    # find exact rank index for value
    # handle potential duplicates by taking first match
    try:
        idx = (series == value).idxmax()
        r = int(ranks.loc[idx])
    except Exception:
        # fallback approximate rank
        diffs = (series - value).abs()
        r = int(ranks.loc[diffs.idxmin()])
    # add ordinal suffix
    def _ord(n):
        if 10 <= (n % 100) <= 20: return f"{n}th"
        ends = {1: "st", 2: "nd", 3: "rd"}
        return f"{n}{ends.get(n % 10, 'th')}"
    return _ord(r)

def _metric_badge(label, value, rank_text):
    # compact HTML badge for header row
    return f"""
    <div style="display:flex;flex-direction:column;gap:2px;align-items:flex-start;padding:6px 8px;border-radius:8px;background:rgba(0,0,0,0.03);">
      <div style="font-size:12px;color:#666">{label}</div>
      <div style="font-weight:600;font-size:18px;line-height:1.1">{value} <span style="font-size:12px;color:#666">({rank_text})</span></div>
    </div>
    """

# ----------------------- Data Caching -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600)
def get_active_players_df():
    df = pd.DataFrame(static_players.get_active_players())
    # Ensure id is int
    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    return df.sort_values("full_name")

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_teams_static_df():
    df = pd.DataFrame(static_teams.get_teams())
    # Only NBA (no WNBA/G-League)
    df = df[df["id"].apply(is_nba_team_id)].copy()
    return df

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_team_context_advanced(season):
    """
    Pull official NBA Advanced regular-season team metrics (per game),
    strictly NBA teams, and compute league ranks.
    """
    try:
        df_adv = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season"
        ).get_data_frames()[0]
    except Exception:
        return pd.DataFrame(), np.nan, np.nan

    df_adv = df_adv[df_adv["TEAM_ID"].apply(is_nba_team_id)].copy()

    # keep only what we use
    cols_needed = ["TEAM_ID","TEAM_NAME","GP","W","L","PACE","OFF_RATING","DEF_RATING","NET_RATING"]
    for c in cols_needed:
        if c not in df_adv.columns: df_adv[c] = np.nan
    # compute record
    df_adv["RECORD"] = df_adv.apply(lambda r: f"{int(r['W'])}-{int(r['L'])}" if np.isfinite(r["W"]) and np.isfinite(r["L"]) else "—", axis=1)

    # ranks (1 is best). Lower DEF_RATING is better (ascending=True)
    df_adv["DEF_RANK"] = df_adv["DEF_RATING"].rank(ascending=True, method="min").astype("Int64")
    df_adv["PACE_RANK"] = df_adv["PACE"].rank(ascending=False, method="min").astype("Int64")
    df_adv["NET_RANK"]  = df_adv["NET_RATING"].rank(ascending=False, method="min").astype("Int64")

    league_pace = float(df_adv["PACE"].mean()) if "PACE" in df_adv.columns else np.nan
    league_def = float(df_adv["DEF_RATING"].mean()) if "DEF_RATING" in df_adv.columns else np.nan
    return df_adv[cols_needed + ["RECORD","DEF_RANK","PACE_RANK","NET_RANK"]], league_pace, league_def

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_player_logs(player_id, season):
    try:
        df = playergamelog.PlayerGameLog(player_id=int(player_id), season=season, season_type_all_star="Regular Season").get_data_frames()[0]
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    # computed
    for c in ["PRA","FG2M","FG2A"]:
        if c not in df.columns:
            df[c] = np.nan
    if "PTS" in df and "REB" in df and "AST" in df:
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    if "FGA" in df and "FG3A" in df:
        df["FG2A"] = df["FGA"] - df["FG3A"]
    if "FGM" in df and "FG3M" in df:
        df["FG2M"] = df["FGM"] - df["FG3M"]
    return df

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_player_logs_all_seasons(player_id, seasons):
    frames = []
    for s in seasons:
        df = get_player_logs(player_id, s)
        if not df.empty:
            frames.append(df)
    if frames:
        out = pd.concat(frames, ignore_index=True).sort_values("GAME_DATE", ascending=False)
        return out.reset_index(drop=True)
    return pd.DataFrame()

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_player_career_regular_per_game(player_id):
    try:
        df = playercareerstats.PlayerCareerStats(player_id=int(player_id)).get_data_frames()[0]
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    # only regular season, per-game line (the endpoint already provides per-game style columns)
    df = df[df["LEAGUE_ID"] == "00"].copy()  # NBA only
    df = df[df["SEASON_ID"].str.startswith("2") | df["SEASON_ID"].str.startswith("1")]  # guard
    # rename common cols if needed
    return df

@st.cache_data(ttl=CACHE_HOURS*3600)
def get_common_player_info(player_id):
    try:
        return commonplayerinfo.CommonPlayerInfo(player_id=int(player_id)).get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

# ----------------------- Sidebar -----------------------
players_df = get_active_players_df()
teams_static_df = get_teams_static_df()
team_adv, league_pace_mean, league_def_mean = get_team_context_advanced(DEFAULT_SEASON)  # default preload; will refetch by selection

with st.sidebar:
    st.header("Filters")
    season = st.selectbox("Season", options=SEASONS[::-1], index=0)
    # refresh context for selected season
    team_adv, league_pace_mean, league_def_mean = get_team_context_advanced(season)

    # Player search
    q = st.text_input("Search player", value="Jayson Tatum")
    filtered = players_df[players_df["full_name"].str.contains(q, case=False, na=False)]
    if filtered.empty:
        st.stop()
    player_name = st.selectbox("Player", filtered["full_name"].tolist())
    player_id = int(filtered.loc[filtered["full_name"] == player_name, "id"].iloc[0])

    # Opponent (NBA only)
    opp_options = team_adv["TEAM_NAME"].tolist() if not team_adv.empty else []
    opponent = st.selectbox("Opponent", opp_options)

    # Recency window
    n_recent = st.selectbox("Recent window", options=["Season", 5, 10, 15, 20], index=1)

# ----------------------- Fetch Data -----------------------
logs = get_player_logs(player_id, season)
career_df = get_player_career_regular_per_game(player_id)
cpi = get_common_player_info(player_id)

if logs.empty:
    st.error("No game logs found for this player/season.")
    st.stop()

if opponent not in team_adv["TEAM_NAME"].values:
    st.error("Opponent not found in team context data.")
    st.stop()

opp_row = team_adv.loc[team_adv["TEAM_NAME"] == opponent].iloc[0]

# ----------------------- Header Section -----------------------
left, right = st.columns([2, 1.2], gap="large")
with left:
    st.subheader(f"{player_name} — {season}")
    team_name = cpi["TEAM_NAME"].iloc[0] if "TEAM_NAME" in cpi.columns and not cpi.empty else "Unknown"
    pos = cpi["POSITION"].iloc[0] if "POSITION" in cpi.columns and not cpi.empty else "N/A"
    exp = cpi["SEASON_EXP"].iloc[0] if "SEASON_EXP" in cpi.columns and not cpi.empty else "N/A"
    gp = len(logs)
    st.caption(f"**Team:** {team_name} • **Position:** {pos} • **Seasons:** {exp} • **Games Played:** {gp}")

    # Recent Averages strip (MIN, PTS, REB, AST, 3PM)
    if n_recent == "Season":
        recent = logs.copy()
    else:
        recent = logs.head(int(n_recent)).copy()
    cols_ra = [c for c in ["MIN","PTS","REB","AST","FG3M"] if c in recent.columns]
    ra = recent[cols_ra].mean(numeric_only=True).round(1) if not recent.empty else pd.Series(dtype=float)

    m1, m2, m3, m4, m5 = st.columns(5)
    def _metric_box(col, label, val):
        v = "—" if not np.isfinite(val) else f"{val:.1f}"
        col.markdown(
            f"""
            <div style="padding:8px 10px;border-radius:10px;background:rgba(0,0,0,0.03);">
                <div style="font-size:12px;color:#666">{label}</div>
                <div style="font-size:22px;font-weight:700;line-height:1.1">{v}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    _metric_box(m1, "MIN", ra.get("MIN", np.nan))
    _metric_box(m2, "PTS", ra.get("PTS", np.nan))
    _metric_box(m3, "REB", ra.get("REB", np.nan))
    _metric_box(m4, "AST", ra.get("AST", np.nan))
    _metric_box(m5, "3PM", ra.get("FG3M", np.nan))

with right:
    # Opponent header with record
    record = opp_row.get("RECORD", "—")
    st.markdown(f"**Opponent:** {opponent} ({record})")

    # Build compact defensive/pace badges with rank
    def_rank = _rank_str(team_adv["DEF_RATING"], float(opp_row["DEF_RATING"]), ascending=True)
    pace_rank = _rank_str(team_adv["PACE"], float(opp_row["PACE"]), ascending=False)
    net_rank  = _rank_str(team_adv["NET_RATING"], float(opp_row["NET_RATING"]), ascending=False)

    b1, b2, b3 = st.columns(3)
    b1.markdown(_metric_badge("DEF Rating", f"{float(opp_row['DEF_RATING']):.1f}", def_rank), unsafe_allow_html=True)
    b2.markdown(_metric_badge("PACE", f"{float(opp_row['PACE']):.1f}", pace_rank), unsafe_allow_html=True)
    b3.markdown(_metric_badge("NET Rating", f"{float(opp_row['NET_RATING']):.1f}", net_rank), unsafe_allow_html=True)

# ----------------------- Recent Trends -----------------------
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
    st.altair_chart(chart, width="stretch")

# ----------------------- Comparison Windows -----------------------
st.markdown("### Compare Windows (Career / Season / L5 / L15)")
def avg(df, n):
    if n == "Season":
        return df.mean(numeric_only=True)
    return df.head(int(n)).mean(numeric_only=True)

kpi = [c for c in ["PTS","REB","AST","MIN"] if c in logs.columns]
vals = {
    "Career": {s: (career_df[s].mean() if s in career_df.columns else np.nan) for s in kpi},
    "Season": avg(logs[kpi], "Season"),
    "L5": avg(logs[kpi], 5),
    "L15": avg(logs[kpi], 15),
}
cmp_df = pd.DataFrame(vals).round(2)
st.dataframe(cmp_df, width="stretch", height=_auto_height(cmp_df))

# ----------------------- Last 5 Games (season) -----------------------
st.markdown("### Last 5 Games")
cols = [c for c in ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","PRA","FG2M","FG2A","FG3M","FG3A","FTM","FTA","OREB","DREB"] if c in logs.columns]
last5 = logs[cols].head(5).copy()
st.dataframe(last5, width="stretch", height=_auto_height(last5))

# ----------------------- Last 5 vs Specific Opponent (all seasons) -----------------------
st.markdown(f"### Last 5 vs {opponent} (All Seasons)")
logs_all = get_player_logs_all_seasons(player_id, SEASONS)
if not logs_all.empty:
    # detect rows vs opponent by matching TEAM_ABBREVIATION in team_adv with matchup text
    # safer: get opponent city/abbrev mapping from team_adv by joining static_teams
    opp_team_id = int(opp_row["TEAM_ID"])
    opp_abbrev = teams_static_df.loc[teams_static_df["id"] == opp_team_id, "abbreviation"].values
    opp_abbrev = opp_abbrev[0] if len(opp_abbrev) else None

    def _vs_team(df, abbr):
        if not abbr:
            return df.iloc[0:0]
        # MATCHUP strings like "BOS @ LAL" or "LAL vs BOS": check if abbr occurs
        return df[df["MATCHUP"].str.contains(rf"\b{re.escape(abbr)}\b", na=False)]

    vs_df = _vs_team(logs_all, opp_abbrev)
    vs5 = vs_df[cols].head(5).copy() if not vs_df.empty else pd.DataFrame(columns=cols)
    if vs5.empty:
        st.info("No historical games found vs selected opponent.")
    else:
        st.dataframe(vs5, width="stretch", height=_auto_height(vs5))
else:
    st.info("No historical logs available across seasons.")

# ----------------------- Projections (hidden for now) -----------------------
with st.expander("Projections (beta)"):
    st.caption(
        "Early pass at pace- & defense-adjusted projections using opponent "
        "DEF RTG, PACE, and allowed attempt/rebound proxies. This is a work-in-progress."
    )
    try:
        # league references
        lg_pace = league_pace_mean if np.isfinite(league_pace_mean) else np.nan
        lg_def  = league_def_mean if np.isfinite(league_def_mean) else np.nan

        # player baselines
        base = avg(logs, 10)  # rolling last 10 as baseline
        # fallbacks
        for c in ["PTS","REB","AST","FG3A","FG3M","FGA","FTA","OREB","DREB","MIN"]:
            if c not in base.index:
                base[c] = np.nan

        # opponent multipliers
        # def factor (<1 = tough, >1 = soft)
        def_factor = safe_div(lg_def, float(opp_row["DEF_RATING"]), default=1.0)
        pace_factor = safe_div(float(opp_row["PACE"]), lg_pace, default=1.0)

        # Very light-touch attempt mix adjustments via LeagueDashTeamGameLogs (team allowed context)
        # Keep robust to failures.
        allowed = pd.DataFrame()
        try:
            g = leaguedashteamgamelogs.LeagueDashTeamGameLogs(
                season_nullable=season, season_type_nullable="Regular Season"
            ).get_data_frames()[0]
            g = g[g["TEAM_ID"].apply(is_nba_team_id)].copy()
            # Compute opponent allowed per game to approximate shot diet and boards
            # Filter to games involving opponent: their opponents' boxscores are rows where OPPONENT_TEAM_ID == opp_team_id
            opp_team_id = int(opp_row["TEAM_ID"])
            allowed = g[g["OPPONENT_TEAM_ID"] == opp_team_id].copy()
            if not allowed.empty:
                grp = allowed.groupby("OPPONENT_TEAM_ID").agg({
                    "FGA":"mean","FG3A":"mean","FTA":"mean","OREB":"mean","DREB":"mean"
                }).rename(columns={
                    "FGA":"ALLOWED_FGA","FG3A":"ALLOWED_3A","FTA":"ALLOWED_FTA",
                    "OREB":"ALLOWED_OREB","DREB":"ALLOWED_DREB"
                })
                opp_allowed = grp.iloc[0]
            else:
                opp_allowed = pd.Series({
                    "ALLOWED_FGA": np.nan, "ALLOWED_3A": np.nan, "ALLOWED_FTA": np.nan,
                    "ALLOWED_OREB": np.nan, "ALLOWED_DREB": np.nan
                })
        except Exception:
            opp_allowed = pd.Series({
                "ALLOWED_FGA": np.nan, "ALLOWED_3A": np.nan, "ALLOWED_FTA": np.nan,
                "ALLOWED_OREB": np.nan, "ALLOWED_DREB": np.nan
            })

        # simple mix factor (normalize by league medians from same table if available)
        # guard if missing
        mix_factor = 1.0
        if not pd.isna(opp_allowed.get("ALLOWED_FGA", np.nan)):
            # derive a soft factor from allowed FGA + FTA + 3A vs their medians
            numers = []
            denoms = []
            for k in ["ALLOWED_FGA","ALLOWED_3A","ALLOWED_FTA","ALLOWED_OREB","ALLOWED_DREB"]:
                val = opp_allowed.get(k, np.nan)
                if pd.notna(val):
                    denoms.append(val)
            # if we had league medians we could compare; absent, keep neutral at 1
            mix_factor = 1.0

        # final multiplier (cap to avoid wild values)
        mult = np.clip(def_factor * pace_factor * mix_factor, 0.6, 1.4)

        proj_pts = base.get("PTS", np.nan) * mult
        proj_reb = base.get("REB", np.nan) * mult
        proj_ast = base.get("AST", np.nan) * mult
        proj_min = base.get("MIN", np.nan) * pace_factor  # minutes loosely pace-insensitive, keep slight tie

        pr1, pr2, pr3, pr4 = st.columns(4)
        pr1.metric("Proj MIN", "—" if pd.isna(proj_min) else f"{proj_min:.1f}")
        pr2.metric("Proj PTS", "—" if pd.isna(proj_pts) else f"{proj_pts:.1f}")
        pr3.metric("Proj REB", "—" if pd.isna(proj_reb) else f"{proj_reb:.1f}")
        pr4.metric("Proj AST", "—" if pd.isna(proj_ast) else f"{proj_ast:.1f}")

    except Exception as e:
        st.warning(f"Projection engine skipped due to error: {e}")

# ----------------------- Footer -----------------------
st.caption("Stable v2.2 build — syntax/deprecation fixes, NBA-only opponents, accurate DEF/pace ranks, recent averages strip, and historical vs-opponent view.")

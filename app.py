# app.py — NBA Player Scouting Dashboard v2 (Auto-load + Minimal Sidebar)
import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

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
REQUEST_TIMEOUT = 15
MAX_RETRIES = 2

def _retry_api(endpoint_cls, kwargs, timeout=REQUEST_TIMEOUT, retries=MAX_RETRIES, sleep=0.8):
    """Minimal retry wrapper for nba_api endpoint classes."""
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
    try:
        return f"{float(v):.1f}"
    except Exception:
        return "—"

def parse_opp_from_matchup(matchup_str):
    """
    MATCHUP examples: 'BOS vs LAL', 'BOS @ MIA'
    Returns the last token (OPP abbr) robustly.
    """
    if not isinstance(matchup_str, str):
        return None
    parts = matchup_str.split()
    if len(parts) >= 3:
        return parts[-1].strip()
    return None

def add_shot_breakouts(df):
    """
    Ensure columns for: MIN PTS REB AST PRA 2PM 2PA 3PM 3PA FTM FTA OREB DREB
    (and keep GAME_DATE, MATCHUP, WL if present)
    """
    for col in ["MIN","PTS","REB","AST","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB"]:
        if col not in df.columns:
            df[col] = 0
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df["2PM"] = df["FGM"] - df["FG3M"]
    df["2PA"] = df["FGA"] - df["FG3A"]

    keep_order = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","PRA",
                  "2PM","2PA","FG3M","FG3A","FTM","FTA","OREB","DREB"]
    existing = [c for c in keep_order if c in df.columns]
    return df[existing]

# ----------------------- Cached data -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_teams_static_df():
    return pd.DataFrame(static_teams.get_teams())

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_team_context_advanced(season):
    """
    Accurate team metrics from NBA Stats:
      - measure_type_detailed_defense='Advanced', per_mode_detailed='PerGame'
    Adds ranks: DEF_RANK (ascending), PACE_RANK (descending), NET_RANK (descending).
    """
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

    # Ranks (1 = best)
    df_adv["DEF_RANK"] = df_adv["DEF_RATING"].rank(ascending=True,  method="min")
    df_adv["PACE_RANK"] = df_adv["PACE"].rank(ascending=False, method="min")
    df_adv["NET_RANK"]  = df_adv["NET_RATING"].rank(ascending=False, method="min")

    df_adv["DEF_RANK"] = df_adv["DEF_RANK"].astype("Int64")
    df_adv["PACE_RANK"] = df_adv["PACE_RANK"].astype("Int64")
    df_adv["NET_RANK"] = df_adv["NET_RANK"].astype("Int64")

    league_pace = float(df_adv["PACE"].mean()) if "PACE" in df_adv.columns else np.nan
    league_def  = float(df_adv["DEF_RATING"].mean()) if "DEF_RATING" in df_adv.columns else np.nan

    keep = cols_keep + ["DEF_RANK","PACE_RANK","NET_RANK"]
    return df_adv[keep], league_pace, league_def

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_season_player_index(season):
    """
    Season index for Team -> Player filtering (from LeagueDashPlayerStats).
    """
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
    df = df.sort_values(["TEAM_NAME","PLAYER_NAME"])
    return df

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
    df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    return df

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_player_career(player_id):
    """
    SeasonTotalsRegularSeason table (one row per season). We'll compute per-game career from totals.
    """
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
def get_all_player_logs_all_seasons(player_id, season_labels):
    """
    Concatenate logs across provided season labels, newest first.
    """
    frames = []
    for s in season_labels:
        df = get_player_logs(player_id, s)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    return out

# ----------------------- Sidebar (minimal, auto-load) -----------------------
teams_static = get_teams_static_df()
team_name_to_abbrev = dict(zip(teams_static["full_name"], teams_static["abbreviation"]))

with st.sidebar:
    st.header("Filters")
    season = st.selectbox("Season", SEASONS, index=0, key="season_sel")

# Load team & player context for chosen season (auto)
with st.spinner("Loading league/team context..."):
    team_adv, league_pace_mean, league_def_mean = get_team_context_advanced(season)

if team_adv.empty:
    st.error("Unable to load league/team context for this season.")
    st.stop()

team_list = team_adv["TEAM_NAME"].sort_values().tolist()

# Continue sidebar now that we have teams/players for this season
with st.sidebar:
    sel_team = st.selectbox("Team", ["(All teams)"] + team_list, index=0, key="team_sel")

    # Build player index
    with st.spinner("Loading players..."):
        season_players = get_season_player_index(season)

    # Filter by team
    filtered_players = season_players.copy()
    if sel_team != "(All teams)":
        filtered_players = filtered_players[filtered_players["TEAM_NAME"] == sel_team]

    # Search + dropdown
    q = st.text_input("Search player", key="player_search").strip()
    if q:
        filtered_players = filtered_players[
            filtered_players["PLAYER_NAME"].str.contains(q, case=False, na=False)
        ]

    if filtered_players.empty:
        st.info("No players match your filters.")
        st.stop()

    # Select player (Streamlit selectbox is searchable too)
    default_idx = 0
    if "player_sel" in st.session_state:
        if st.session_state["player_sel"] in filtered_players["PLAYER_NAME"].tolist():
            default_idx = filtered_players["PLAYER_NAME"].tolist().index(st.session_state["player_sel"])
    player_name = st.selectbox("Player", filtered_players["PLAYER_NAME"].tolist(), index=default_idx, key="player_sel")
    player_row = filtered_players[filtered_players["PLAYER_NAME"] == player_name].iloc[0]
    player_id  = int(player_row["PLAYER_ID"])

    n_recent = st.selectbox("Recency window", ["Season", 5, 10, 15, 20], index=1, key="recent_sel")

# ----------------------- Fetch Player Data -----------------------
with st.spinner("Fetching player logs & info..."):
    logs = get_player_logs(player_id, season)
    if logs.empty:
        st.error("No game logs for this player/season.")
        st.stop()
    career_df = get_player_career(player_id)
    cpi = get_common_player_info(player_id)

# ----------------------- Header Section (with Opponent selector) -----------------------
left, right = st.columns([2, 1])

with left:
    st.subheader(f"{player_name} — {season}")
    team_name_disp = (cpi["TEAM_NAME"].iloc[0] if ("TEAM_NAME" in cpi.columns and not cpi.empty) else player_row.get("TEAM_NAME","Unknown"))
    pos = (cpi["POSITION"].iloc[0] if ("POSITION" in cpi.columns and not cpi.empty) else "N/A")
    exp = (cpi["SEASON_EXP"].iloc[0] if ("SEASON_EXP" in cpi.columns and not cpi.empty) else "N/A")
    gp = len(logs)
    st.caption(f"**Team:** {team_name_disp} • **Position:** {pos} • **Seasons:** {exp} • **Games Played:** {gp}")

with right:
    # Small Opponent chooser (kept out of the sidebar per your spec)
    opponent = st.selectbox("Opponent", team_list, index=0, key="opponent_sel")
    opp_row = team_adv.loc[team_adv["TEAM_NAME"] == opponent].iloc[0]

    st.markdown(f"**Opponent:** {opponent}")
    c1, c2, c3 = st.columns(3)
    c1.metric("DEF Rating", _fmt1(opp_row.get("DEF_RATING", np.nan)))
    c1.caption(f"Rank: {int(opp_row['DEF_RANK'])}/30" if pd.notna(opp_row.get("DEF_RANK")) else "Rank: —")
    c2.metric("PACE", _fmt1(opp_row.get("PACE", np.nan)))
    c2.caption(f"Rank: {int(opp_row['PACE_RANK'])}/30" if pd.notna(opp_row.get("PACE_RANK")) else "Rank: —")
    c3.metric("NET Rating", _fmt1(opp_row.get("NET_RATING", np.nan)))
    c3.caption(f"Rank: {int(opp_row['NET_RANK'])}/30" if pd.notna(opp_row.get("NET_RANK")) else "Rank: —")

# ----------------------- Trends -----------------------
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
    if df.empty: return pd.Series(dtype=float)
    if n == "Season": return df.mean(numeric_only=True)
    return df.head(int(n)).mean(numeric_only=True)

def career_per_game(career_df, cols=("PTS","REB","AST","MIN")):
    """
    Compute true career per-game from regular-season totals:
      sum(stat) / sum(GP)
    """
    if career_df.empty or "GP" not in career_df.columns:
        return pd.Series({c: np.nan for c in cols}, dtype=float)
    needed = list(set(cols) | {"GP"})
    for c in needed:
        if c not in career_df.columns:
            career_df[c] = 0
    total_gp = career_df["GP"].sum()
    if total_gp == 0:
        return pd.Series({c: np.nan for c in cols}, dtype=float)
    out = {c: career_df[c].sum() / total_gp for c in cols}
    return pd.Series(out).astype(float)

kpi = [c for c in ["PTS","REB","AST","MIN"] if (c in logs.columns) or (c in career_df.columns)]
career_pg = career_per_game(career_df, cols=kpi)
vals = {
    "Career": career_pg,
    "Season": avg(logs[kpi], "Season") if set(kpi).issubset(logs.columns) else pd.Series({s: np.nan for s in kpi}),
    "L5": avg(logs[kpi], 5) if set(kpi).issubset(logs.columns) else pd.Series({s: np.nan for s in kpi}),
    "L15": avg(logs[kpi], 15) if set(kpi).issubset(logs.columns) else pd.Series({s: np.nan for s in kpi}),
}
cmp_df = pd.DataFrame(vals).round(2)
st.dataframe(cmp_df.style.format(numeric_format_map(cmp_df)), use_container_width=True, height=_auto_height(cmp_df))

# ----------------------- Last 5 Games (current season) -----------------------
st.markdown("### Last 5 Games")
cols_base = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB"]
last5 = logs[cols_base].head(5).copy()
last5 = add_shot_breakouts(last5)
num_fmt = {c: "{:.0f}" for c in last5.select_dtypes(include=[np.number]).columns if c != "GAME_DATE"}
st.dataframe(last5.style.format(num_fmt), use_container_width=True, height=_auto_height(last5))

# ----------------------- Last 5 vs Opponent (ALL seasons) -----------------------
st.markdown(f"### Last 5 vs {opponent} (All Seasons)")
opp_abbrev = team_name_to_abbrev.get(opponent)
if not opp_abbrev:
    opp_abbrev = str(opp_row.get("TEAM_ABBREVIATION", "")).strip() or None

# Determine season labels from the player's career
if "SEASON" in career_df.columns and not career_df.empty:
    season_labels = list(career_df["SEASON"].dropna().unique())
    def _yr(s):
        try: return int(s.split("-")[0])
        except: return -1
    season_labels = sorted(season_labels, key=_yr, reverse=True)
else:
    season_labels = SEASONS  # fallback

if opp_abbrev:
    all_logs = get_all_player_logs_all_seasons(player_id, season_labels)
    if all_logs.empty or "MATCHUP" not in all_logs.columns:
        st.info(f"No matchup data available for {opponent}.")
    else:
        all_logs = all_logs.copy()
        all_logs["OPP_ABBR"] = all_logs["MATCHUP"].apply(parse_opp_from_matchup)
        vs_opp_all = all_logs[all_logs["OPP_ABBR"] == opp_abbrev]
        vs_opp5 = vs_opp_all[cols_base].head(5).copy() if not vs_opp_all.empty else pd.DataFrame(columns=cols_base)
        vs_opp5 = add_shot_breakouts(vs_opp5)
        if vs_opp5.empty:
            st.info(f"No historical games vs {opponent}.")
        else:
            num_fmt2 = {c: "{:.0f}" for c in vs_opp5.select_dtypes(include=[np.number]).columns if c != "GAME_DATE"}
            st.dataframe(vs_opp5.style.format(num_fmt2), use_container_width=True, height=_auto_height(vs_opp5))
else:
    st.info("Could not resolve opponent abbreviation; skipping opponent-specific table.")

# ----------------------- Projections (hidden for now) -----------------------
with st.expander("Projection Summary (beta – hidden until finalized)"):
    enable_proj = st.checkbox("Show simple projection using recent vs career and opponent defense", value=False)
    if enable_proj:
        try:
            recent_n = 5 if n_recent == "Season" else int(n_recent)
            base_recent = logs.head(recent_n)[["PTS","REB","AST","MIN","FG3M"]].mean(numeric_only=True)
            base_season = logs[["PTS","REB","AST","MIN","FG3M"]].mean(numeric_only=True)
            blended = 0.6 * base_recent + 0.4 * base_season

            league_def = team_adv["DEF_RATING"].mean()
            opp_def   = opp_row.get("DEF_RATING", league_def)
            def_adj   = (league_def / opp_def) if (pd.notna(league_def) and pd.notna(opp_def) and opp_def != 0) else 1.0

            proj = (blended * def_adj).to_frame("Proj").T
            proj = proj[["PTS","REB","AST","MIN","FG3M"]].round(2)
            st.dataframe(proj, use_container_width=True, height=_auto_height(proj))
        except Exception as e:
            st.info(f"Projection temporarily unavailable: {e}")

# ----------------------- Footer -----------------------
st.caption("Notes: Auto-load filters with per-call timeouts/retries. Defensive metrics are pulled from NBA Stats (Advanced, per-game) and ranks are shown beneath each metric (1 = best).")

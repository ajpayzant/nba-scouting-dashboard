# app.py — NBA Player Scouting Dashboard v2 (NBA-only Opponents + Compact Recent Averages)
import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from nba_api.stats.static import teams as static_teams
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
REQUEST_TIMEOUT = 15
MAX_RETRIES = 2

def _retry_api(endpoint_cls, kwargs, timeout=REQUEST_TIMEOUT, retries=MAX_RETRIES, sleep=0.8):
    """Retry wrapper for nba_api endpoint classes."""
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

def format_record(w, l):
    try:
        return f"{int(w)}–{int(l)}"
    except Exception:
        return "—"

# ----------------------- Cached data -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_teams_static_df():
    return pd.DataFrame(static_teams.get_teams())

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_team_context(season):
    """
    Accurate NBA-only team context for the selected season:
      - Advanced/PerGame: PACE, OFF_RATING, DEF_RATING, NET_RATING
      - Base/PerGame: W, L, GP, W_PCT
    Adds ranks based on these official values.
    Ensures **NBA-only** by filtering TEAM_ID starting with '161061'.
    """
    # Advanced
    try:
        adv_frames = _retry_api(
            leaguedashteamstats.LeagueDashTeamStats,
            {
                "season": season,
                "measure_type_detailed_defense": "Advanced",
                "per_mode_detailed": "PerGame",
            },
        )
        df_adv = adv_frames[0] if adv_frames else pd.DataFrame()
    except Exception:
        df_adv = pd.DataFrame()

    # Base (W/L/GP/W_PCT)
    try:
        base_frames = _retry_api(
            leaguedashteamstats.LeagueDashTeamStats,
            {
                "season": season,
                "measure_type_detailed_defense": "Base",
                "per_mode_detailed": "PerGame",
            },
        )
        df_base = base_frames[0] if base_frames else pd.DataFrame()
    except Exception:
        df_base = pd.DataFrame()

    if df_adv.empty and df_base.empty:
        return pd.DataFrame()

    # --- NBA-only filtering (this was the bug before; do NOT shadow variables) ---
    if not df_adv.empty and "TEAM_ID" in df_adv.columns:
        df_adv = df_adv[df_adv["TEAM_ID"].astype(str).str.startswith("161061")].copy()
    if not df_base.empty and "TEAM_ID" in df_base.columns:
        df_base = df_base[df_base["TEAM_ID"].astype(str).str.startswith("161061")].copy()

    # Select columns and merge
    cols_adv = ["TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION","PACE","OFF_RATING","DEF_RATING","NET_RATING"]
    for c in cols_adv:
        if c not in df_adv.columns:
            df_adv[c] = np.nan
    df_adv = df_adv[cols_adv]

    cols_base = ["TEAM_ID","GP","W","L","W_PCT"]
    for c in cols_base:
        if c not in df_base.columns:
            df_base[c] = np.nan
    df_base = df_base[cols_base]

    df = pd.merge(df_adv, df_base, on="TEAM_ID", how="inner")  # inner to keep NBA-only intersection

    # Ranks (1 = best)
    df["DEF_RANK"] = df["DEF_RATING"].rank(ascending=True,  method="min")   # lower is better
    df["PACE_RANK"] = df["PACE"].rank(ascending=False, method="min")
    df["NET_RANK"]  = df["NET_RATING"].rank(ascending=False, method="min")
    for c in ["DEF_RANK","PACE_RANK","NET_RANK"]:
        df[c] = df[c].astype("Int64")

    return df

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_season_player_index(season):
    """Season index for Player dropdown (with team fields to show current team)."""
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
    """Regular-season totals by season; used for true career per-game."""
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
    """Concat logs across provided seasons, newest first."""
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

# ----------------------- Sidebar (Season, Player, Recency) -----------------------
with st.sidebar:
    st.header("Filters")
    season = st.selectbox("Season", SEASONS, index=0, key="season_sel")

# Load team & player context for chosen season (auto)
with st.spinner("Loading team context..."):
    team_ctx = get_team_context(season)

if team_ctx.empty:
    st.error("Unable to load team context for this season.")
    st.stop()

team_list = team_ctx["TEAM_NAME"].sort_values().tolist()

with st.sidebar:
    with st.spinner("Loading players..."):
        season_players = get_season_player_index(season)

    q = st.text_input("Search player", key="player_search").strip()
    filtered_players = season_players.copy()
    if q:
        filtered_players = filtered_players[
            filtered_players["PLAYER_NAME"].str.contains(q, case=False, na=False)
        ]

    if filtered_players.empty:
        st.info("No players match your search.")
        st.stop()

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

# ----------------------- Header + Opponent Selector -----------------------
left, right = st.columns([2, 1])

with left:
    st.subheader(f"{player_name} — {season}")
    team_name_disp = (cpi["TEAM_NAME"].iloc[0] if ("TEAM_NAME" in cpi.columns and not cpi.empty) else player_row.get("TEAM_NAME","Unknown"))
    pos = (cpi["POSITION"].iloc[0] if ("POSITION" in cpi.columns and not cpi.empty) else "N/A")
    exp = (cpi["SEASON_EXP"].iloc[0] if ("SEASON_EXP" in cpi.columns and not cpi.empty) else "N/A")
    gp = len(logs)
    st.caption(f"**Team:** {team_name_disp} • **Position:** {pos} • **Seasons:** {exp} • **Games Played:** {gp}")

with right:
    opponent = st.selectbox("Opponent", team_list, index=0, key="opponent_sel")

# Opponent row (NBA-only, accurate), plus record
opp_row = team_ctx.loc[team_ctx["TEAM_NAME"] == opponent].iloc[0]
opp_record = format_record(opp_row.get("W", np.nan), opp_row.get("L", np.nan))

# Display opponent with record and accurate metrics
st.markdown(f"### Opponent: **{opponent}** ({opp_record})")
c1, c2, c3 = st.columns(3)
c1.metric("DEF Rating", _fmt1(opp_row.get("DEF_RATING", np.nan)))
c1.caption(f"Rank: {int(opp_row['DEF_RANK'])}/30" if pd.notna(opp_row.get("DEF_RANK")) else "Rank: —")
c2.metric("PACE", _fmt1(opp_row.get("PACE", np.nan)))
c2.caption(f"Rank: {int(opp_row['PACE_RANK'])}/30" if pd.notna(opp_row.get("PACE_RANK")) else "Rank: —")
c3.metric("NET Rating", _fmt1(opp_row.get("NET_RATING", np.nan)))
c3.caption(f"Rank: {int(opp_row['NET_RANK'])}/30" if pd.notna(opp_row.get("NET_RANK")) else "Rank: —")

# ----------------------- Recent Averages (compact, visual metrics) -----------------------
# Calculate recent window stats (MIN, PTS, REB, AST, 3PM)
stats_needed = ["MIN","PTS","REB","AST","FG3M"]
for col in stats_needed:
    if col not in logs.columns:
        logs[col] = 0

if n_recent == "Season":
    window_df = logs
else:
    window_df = logs.head(int(n_recent))

recent_avg = window_df[stats_needed].mean(numeric_only=True)

st.markdown("### Recent Averages")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("MIN", _fmt1(recent_avg.get("MIN", np.nan)))
m2.metric("PTS", _fmt1(recent_avg.get("PTS", np.nan)))
m3.metric("REB", _fmt1(recent_avg.get("REB", np.nan)))
m4.metric("AST", _fmt1(recent_avg.get("AST", np.nan)))
m5.metric("3PM", _fmt1(recent_avg.get("FG3M", np.nan)))

# ----------------------- Trend Lines -----------------------
st.markdown(f"### Trends (Last {n_recent if n_recent!='Season' else 'Season'} Games)")
trend_cols = [c for c in ["MIN","PTS","REB","AST","PRA","FG3M"] if c in logs.columns]
# Build PRA explicitly for trend chart (doesn't modify averages above)
if "PRA" not in logs.columns:
    logs["PRA"] = logs.get("PTS", 0) + logs.get("REB", 0) + logs.get("AST", 0)

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

# ----------------------- Comparison Window -----------------------
st.markdown("### Compare Windows (Career / Season / L5 / L15)")

def avg(df, n):
    if df.empty: return pd.Series(dtype=float)
    if n == "Season": return df.mean(numeric_only=True)
    return df.head(int(n)).mean(numeric_only=True)

def career_per_game(career_df, cols=("PTS","REB","AST","MIN")):
    """
    True career per-game from **regular-season totals**:
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
teams_static = get_teams_static_df()
team_name_to_abbrev = dict(zip(teams_static["full_name"], teams_static["abbreviation"]))
opp_abbrev = team_name_to_abbrev.get(opponent)
if not opp_abbrev:
    opp_abbrev = str(opp_row.get("TEAM_ABBREVIATION", "")).strip() if 'opp_row' in locals() else None

# Determine seasons to search from the player's career
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

            league_def = team_ctx["DEF_RATING"].mean()
            opp_def   = opp_row.get("DEF_RATING", league_def)
            def_adj   = (league_def / opp_def) if (pd.notna(league_def) and pd.notna(opp_def) and opp_def != 0) else 1.0

            proj = (blended * def_adj).to_frame("Proj").T
            proj = proj[["PTS","REB","AST","MIN","FG3M"]].round(2)
            st.dataframe(proj, use_container_width=True, height=_auto_height(proj))
        except Exception as e:
            st.info(f"Projection temporarily unavailable: {e}")

# ----------------------- Footer -----------------------
st.caption("Notes: Sidebar = Season, Player (search/dropdown), Recency. Opponent metrics & record derive directly from NBA Stats (Advanced/Base, PerGame) with NBA-only filtering. Recent averages are compact metric tiles.")

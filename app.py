# app.py — NBA Player Scouting Dashboard (Projections upgraded with opponent allowed rates + pace/def)
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
st.title("🏀 NBA Player Scouting Dashboard")

# ----------------------- Config -----------------------
CACHE_HOURS = 12
REQUEST_TIMEOUT = 15
MAX_RETRIES = 2

BLEND_RECENT = 0.60  # 60% recent, 40% season for baselines
# Defensive elasticity (how much DEF_RTG impacts makes/assists/free throws)
ALPHA_3PT = 0.50
ALPHA_2PT = 0.40
ALPHA_FT  = 0.35
ALPHA_AST = 0.30

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
    # 'BOS vs LAL' or 'BOS @ MIA' -> last token
    if not isinstance(matchup_str, str):
        return None
    parts = matchup_str.split()
    return parts[-1].strip() if len(parts) >= 3 else None

def add_shot_breakouts(df):
    for col in ["MIN","PTS","REB","AST","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB"]:
        if col not in df.columns:
            df[col] = 0
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df["2PM"] = df["FGM"] - df["FG3M"]
    df["2PA"] = df["FGA"] - df["FG3A"]
    keep_order = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","PRA","2PM","2PA","FG3M","FG3A","FTM","FTA","OREB","DREB"]
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

def _nba_only(df):
    if df is None or df.empty or "TEAM_ID" not in df.columns:
        return pd.DataFrame()
    return df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()

COMMON_PARAMS = {
    "season_type_all_star": "Regular Season",
    "per_mode_detailed": "PerGame",
    "league_id_nullable": "00",
    "date_from_nullable": None,
    "date_to_nullable": None,
    "po_round_nullable": None,
}

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_team_context_regular_season_to_date(season: str) -> pd.DataFrame:
    """Advanced + Base (NBA-only)."""
    # Advanced
    try:
        adv_frames = _retry_api(
            leaguedashteamstats.LeagueDashTeamStats,
            dict(COMMON_PARAMS, season=season, measure_type_detailed_defense="Advanced"),
        )
        df_adv = adv_frames[0] if adv_frames else pd.DataFrame()
    except Exception:
        df_adv = pd.DataFrame()
    # Base (W/L/GP/W_PCT)
    try:
        base_frames = _retry_api(
            leaguedashteamstats.LeagueDashTeamStats,
            dict(COMMON_PARAMS, season=season, measure_type_detailed_defense="Base"),
        )
        df_base = base_frames[0] if base_frames else pd.DataFrame()
    except Exception:
        df_base = pd.DataFrame()

    df_adv = _nba_only(df_adv)
    df_base = _nba_only(df_base)
    if df_adv.empty or df_base.empty:
        return pd.DataFrame()

    cols_adv = ["TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION","PACE","OFF_RATING","DEF_RATING","NET_RATING"]
    for c in cols_adv:
        if c not in df_adv.columns: df_adv[c] = np.nan
    df_adv = df_adv[cols_adv].copy()

    cols_base = ["TEAM_ID","GP","W","L","W_PCT","MIN"]
    for c in cols_base:
        if c not in df_base.columns: df_base[c] = np.nan
    df_base = df_base[cols_base].copy()

    df = pd.merge(df_adv, df_base, on="TEAM_ID", how="inner")

    df["DEF_RANK"] = df["DEF_RATING"].rank(ascending=True,  method="min").astype("Int64")
    df["PACE_RANK"] = df["PACE"].rank(ascending=False, method="min").astype("Int64")
    df["NET_RANK"]  = df["NET_RATING"].rank(ascending=False, method="min").astype("Int64")
    return df.sort_values("TEAM_NAME").reset_index(drop=True)

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_team_opponent_allowed(season: str) -> pd.DataFrame:
    """
    Opponent 'allowed' per-game rates (NBA-only).
    Typical columns include: OPP_FGM, OPP_FGA, OPP_FG3M, OPP_FG3A, OPP_FTM, OPP_FTA,
    OPP_OREB, OPP_DREB, OPP_REB, OPP_AST, OPP_TOV, etc.
    """
    try:
        opp_frames = _retry_api(
            leaguedashteamstats.LeagueDashTeamStats,
            dict(COMMON_PARAMS, season=season, measure_type_detailed_defense="Opponent"),
        )
        df_opp = opp_frames[0] if opp_frames else pd.DataFrame()
    except Exception:
        df_opp = pd.DataFrame()
    df_opp = _nba_only(df_opp)
    if df_opp.empty:
        return pd.DataFrame()

    wanted = [
        "TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION",
        "OPP_FGM","OPP_FGA","OPP_FG3M","OPP_FG3A","OPP_FTM","OPP_FTA",
        "OPP_OREB","OPP_DREB","OPP_REB","OPP_AST","OPP_TOV"
    ]
    for c in wanted:
        if c not in df_opp.columns:
            df_opp[c] = np.nan
    return df_opp[wanted].copy().sort_values("TEAM_NAME").reset_index(drop=True)

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_season_player_index(season):
    try:
        frames = _retry_api(LeagueDashPlayerStats, dict(COMMON_PARAMS, season=season))
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    keep = ["PLAYER_ID","PLAYER_NAME","TEAM_ID","TEAM_ABBREVIATION","TEAM_NAME","GP","MIN"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    return df[keep].drop_duplicates(subset=["PLAYER_ID"]).sort_values(["TEAM_NAME","PLAYER_NAME"]).reset_index(drop=True)

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_player_logs(player_id, season):
    try:
        frames = _retry_api(playergamelog.PlayerGameLog, dict(COMMON_PARAMS, player_id=player_id, season=season))
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
def get_all_player_logs_all_seasons(player_id, season_labels):
    frames = []
    for s in season_labels:
        df = get_player_logs(player_id, s)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=0, ignore_index=True)
    return out.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

# ----------------------- Sidebar (Season, Player, Recency) -----------------------
with st.sidebar:
    st.header("Filters")
    season = st.selectbox("Season", SEASONS, index=0, key="season_sel")

# League/team context
with st.spinner("Loading league context..."):
    team_ctx = get_team_context_regular_season_to_date(season)
    opp_allowed_ctx = get_team_opponent_allowed(season)

if team_ctx.empty:
    st.error("Unable to load team context for this season.")
    st.stop()

team_list = team_ctx["TEAM_NAME"].tolist()

with st.sidebar:
    with st.spinner("Loading players..."):
        season_players = get_season_player_index(season)

    q = st.text_input("Search player", key="player_search").strip()
    filtered_players = season_players.copy()
    if q:
        filtered_players = filtered_players[filtered_players["PLAYER_NAME"].str.contains(q, case=False, na=False)]

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

# Opponent row + record/metrics
opp_row = team_ctx.loc[team_ctx["TEAM_NAME"] == opponent].iloc[0]
opp_record = format_record(opp_row.get("W", np.nan), opp_row.get("L", np.nan))

st.markdown(f"### Opponent: **{opponent}** ({opp_record})")
c1, c2, c3 = st.columns(3)
c1.metric("DEF Rating", _fmt1(opp_row.get("DEF_RATING", np.nan)))
c1.caption(f"Rank: {int(opp_row['DEF_RANK'])}/30" if pd.notna(opp_row.get("DEF_RANK")) else "Rank: —")
c2.metric("PACE", _fmt1(opp_row.get("PACE", np.nan)))
c2.caption(f"Rank: {int(opp_row['PACE_RANK'])}/30" if pd.notna(opp_row.get("PACE_RANK")) else "Rank: —")
c3.metric("NET Rating", _fmt1(opp_row.get("NET_RATING", np.nan)))
c3.caption(f"Rank: {int(opp_row['NET_RANK'])}/30" if pd.notna(opp_row.get("NET_RANK")) else "Rank: —")

# ----------------------- Recent Averages (compact tiles) -----------------------
for col in ["MIN","PTS","REB","AST","FG3M","FGM","FGA","FG3A","FTM","FTA","OREB","DREB"]:
    if col not in logs.columns:
        logs[col] = 0

window_df = logs if n_recent == "Season" else logs.head(int(n_recent))
recent_avg_tiles = window_df[["MIN","PTS","REB","AST","FG3M"]].mean(numeric_only=True)

st.markdown("### Recent Averages")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("MIN", _fmt1(recent_avg_tiles.get("MIN", np.nan)))
m2.metric("PTS", _fmt1(recent_avg_tiles.get("PTS", np.nan)))
m3.metric("REB", _fmt1(recent_avg_tiles.get("REB", np.nan)))
m4.metric("AST", _fmt1(recent_avg_tiles.get("AST", np.nan)))
m5.metric("3PM", _fmt1(recent_avg_tiles.get("FG3M", np.nan)))

# ----------------------- Trend Lines -----------------------
st.markdown(f"### Trends (Last {n_recent if n_recent!='Season' else 'Season'} Games)")
if "PRA" not in logs.columns:
    logs["PRA"] = logs.get("PTS", 0) + logs.get("REB", 0) + logs.get("AST", 0)
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

# ----------------------- Last 5 vs Opponent (All Seasons) -----------------------
st.markdown(f"### Last 5 vs {opponent} (All Seasons)")
teams_static = get_teams_static_df()
team_name_to_abbrev = dict(zip(teams_static["full_name"], teams_static["abbreviation"]))
opp_abbrev = team_name_to_abbrev.get(opponent) or str(opp_row.get("TEAM_ABBREVIATION", "")).strip()

if "SEASON" in career_df.columns and not career_df.empty:
    season_labels = list(career_df["SEASON"].dropna().unique())
    def _yr(s):
        try: return int(s.split("-")[0])
        except: return -1
    season_labels = sorted(season_labels, key=_yr, reverse=True)
else:
    season_labels = SEASONS

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

# ----------------------- Projections (upgraded) -----------------------
with st.expander("Projection Summary (pace + defense + opponent allowed rates)"):
    try:
        # 1) Baselines: recent vs season (compute key components for better adjustments)
        recent_n = 5 if st.session_state.get("recent_sel","Season") == "Season" else int(st.session_state["recent_sel"])
        base_recent = logs.head(recent_n)[["MIN","PTS","REB","AST","FG3M","FGM","FGA","FG3A","FTM","FTA"]].mean(numeric_only=True)
        base_season = logs[["MIN","PTS","REB","AST","FG3M","FGM","FGA","FG3A","FTM","FTA"]].mean(numeric_only=True)
        blended = BLEND_RECENT * base_recent + (1 - BLEND_RECENT) * base_season

        # Derive 2PM/2PA from splits (guard against missing columns)
        base_3PM = float(blended.get("FG3M", np.nan))
        base_3PA = float(blended.get("FG3A", np.nan))
        base_FTM = float(blended.get("FTM", np.nan))
        base_FTA = float(blended.get("FTA", np.nan))
        base_FGM = float(blended.get("FGM", np.nan))
        base_FGA = float(blended.get("FGA", np.nan))
        base_2PM = max(0.0, base_FGM - base_3PM) if np.isfinite(base_FGM) and np.isfinite(base_3PM) else np.nan
        base_2PA = max(0.0, base_FGA - base_3PA) if np.isfinite(base_FGA) and np.isfinite(base_3PA) else np.nan

        base_MIN = float(blended.get("MIN", np.nan))
        base_REB = float(blended.get("REB", np.nan))
        base_AST = float(blended.get("AST", np.nan))

        # 2) Multipliers from opponent & league means (PACE, DEF, allowed volumes)
        league_pace = float(team_ctx["PACE"].mean())
        league_def  = float(team_ctx["DEF_RATING"].mean())

        pace_mult = float(opp_row["PACE"]) / league_pace if np.isfinite(opp_row.get("PACE", np.nan)) and league_pace else 1.0
        def_mult  = league_def / float(opp_row["DEF_RATING"]) if np.isfinite(opp_row.get("DEF_RATING", np.nan)) and league_def else 1.0

        # Opponent allowed table join (to get this opponent’s OPP_* row)
        if not opp_allowed_ctx.empty:
            opp_allowed_row = opp_allowed_ctx.loc[opp_allowed_ctx["TEAM_NAME"] == opponent]
            opp_allowed_row = opp_allowed_row.iloc[0] if not opp_allowed_row.empty else pd.Series(dtype=float)

            # League means for OPP_* (exclude NaNs)
            lm = opp_allowed_ctx.mean(numeric_only=True)

            # Helpers to safely compute ratios
            def _ratio(numer, denom):
                return float(numer) / float(denom) if np.isfinite(numer) and np.isfinite(denom) and denom != 0 else 1.0

            three_vol_mult = _ratio(opp_allowed_row.get("OPP_FG3A", np.nan), lm.get("OPP_FG3A", np.nan))
            two_vol_mult   = _ratio(
                (opp_allowed_row.get("OPP_FGA", np.nan) - opp_allowed_row.get("OPP_FG3A", np.nan)),
                (lm.get("OPP_FGA", np.nan) - lm.get("OPP_FG3A", np.nan)),
            )
            fta_vol_mult   = _ratio(opp_allowed_row.get("OPP_FTA", np.nan), lm.get("OPP_FTA", np.nan))
            reb_vol_mult   = _ratio(
                (opp_allowed_row.get("OPP_OREB", np.nan) + opp_allowed_row.get("OPP_DREB", np.nan)),
                (lm.get("OPP_OREB", np.nan) + lm.get("OPP_DREB", np.nan)),
            )
            ast_vol_mult   = _ratio(opp_allowed_row.get("OPP_AST", np.nan), lm.get("OPP_AST", np.nan)) if "OPP_AST" in opp_allowed_ctx.columns else def_mult
        else:
            three_vol_mult = two_vol_mult = fta_vol_mult = reb_vol_mult = ast_vol_mult = 1.0

        # 3) Apply multiplicative adjustments
        # Use DEF as a mild elasticity on makes/assists/FT drawing.
        three_adj = pace_mult * three_vol_mult * (def_mult ** ALPHA_3PT)
        two_adj   = pace_mult * two_vol_mult   * (def_mult ** ALPHA_2PT)
        ft_adj    = pace_mult * fta_vol_mult   * (def_mult ** ALPHA_FT)
        reb_adj   = pace_mult * reb_vol_mult
        ast_adj   = pace_mult * ast_vol_mult   * (def_mult ** ALPHA_AST)

        proj_3PM = base_3PM * three_adj if np.isfinite(base_3PM) else np.nan
        proj_2PM = base_2PM * two_adj     if np.isfinite(base_2PM) else np.nan
        proj_FTM = base_FTM * ft_adj      if np.isfinite(base_FTM) else np.nan

        proj_REB = base_REB * reb_adj if np.isfinite(base_REB) else np.nan
        proj_AST = base_AST * ast_adj if np.isfinite(base_AST) else np.nan
        proj_MIN = base_MIN  # usually rotation-driven; keep baseline

        # Build PTS from components (avoid compounding rounding)
        proj_PTS = (
            (3.0 * proj_3_*

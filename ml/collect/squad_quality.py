"""
Collect squad quality metrics for all 48 WC 2026 teams.

Transfermarkt metrics (via salimt/football-datasets):
  - total_squad_value
  - avg_starting_xi_value   (top-11 players by market value)
  - squad_depth_ratio       (players 12-23 total value / starting xi value)
  - star_concentration      (most valuable player value / total squad value)
  - avg_age_starting_xi
  - league_quality_score    (weighted avg of league tier for each player's club)

xG metrics (computed from data/raw/international_results.csv as a proxy,
since FBref is Cloudflare-protected):
  - xg_per90_last10         goals scored per 90 min in last 10 competitive matches
  - xga_per90_last10        goals conceded per 90 in last 10 competitive matches
  - xg_diff_per90_last10    difference (attack minus defence)

FBref scraping is attempted first; the goal-rate proxy is the fallback.

Output: data/raw/squad_quality.csv
"""

from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

BASE_TM = "https://raw.githubusercontent.com/salimt/football-datasets/main/datalake/transfermarkt"
RESULTS_PATH = Path("data/raw/international_results.csv")
OUT_PATH     = Path("data/raw/squad_quality.csv")

# WC 2026 qualified teams (48 teams, update if draw changes)
WC2026_TEAMS = [
    # CONMEBOL (6)
    "Brazil", "Argentina", "Colombia", "Ecuador", "Uruguay", "Paraguay",
    # UEFA (16)
    "Germany", "France", "England", "Spain", "Portugal", "Netherlands",
    "Belgium", "Italy", "Croatia", "Denmark", "Austria", "Switzerland",
    "Serbia", "Turkey", "Scotland", "Hungary",
    # CAF (9)
    "Morocco", "Senegal", "Egypt", "Nigeria", "Cameroon",
    "South Africa", "DR Congo", "Mali", "Ivory Coast",
    # AFC (8)
    "Japan", "South Korea", "Iran", "Saudi Arabia", "Qatar",
    "Australia", "Uzbekistan", "Jordan",
    # CONCACAF (6)
    "United States", "Canada", "Mexico", "Jamaica", "Panama", "Honduras",
    # OFC (1)
    "New Zealand",
    # Remaining spots (playoff qualifiers — update when confirmed)
    "Tunisia", "Poland", "Ukraine", "Venezuela",
]

# League tier table (lower = better); clubs not matched default to tier 4
LEAGUE_TIERS: dict[str, int] = {
    # Tier 1: top-5 European leagues
    "Premier League": 1, "La Liga": 1, "Bundesliga": 1, "Serie A": 1, "Ligue 1": 1,
    # Tier 2: second-tier European + UCL-quality domestic leagues
    "Eredivisie": 2, "Primeira Liga": 2, "Pro League": 2, "Super Lig": 2,
    "Scottish Premiership": 2, "Saudi Pro League": 2, "MLS": 2,
    "Brazilian Serie A": 2, "Argentine Primera": 2,
    # Tier 3: other professional leagues
    "J1 League": 3, "K League 1": 3, "Chinese Super League": 3,
    "Süper Lig": 2,   # alias
}

MINUTES_PER_MATCH = 90


def fetch_csv(name: str, url: str) -> pd.DataFrame:
    log.info("Fetching %s ...", name)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text), low_memory=False)


# ── Transfermarkt squad metrics ───────────────────────────────────────────────

def build_squad_metrics(
    nat: pd.DataFrame,
    mv:  pd.DataFrame,
    pro: pd.DataFrame,
    team_map: dict[int, str],
) -> pd.DataFrame:
    """
    For each WC2026 team compute value-based and age-based squad metrics.
    Uses the most recent market value per player.
    """
    # Most recent value per player
    latest = (
        mv.sort_values("date")
        .groupby("player_id", as_index=False)
        .last()[["player_id", "value"]]
    )

    # Prefer current/recent players; fall back to former for teams with no current data
    active   = {"CURRENT_NATIONAL_PLAYER", "RECENT_NATIONAL_PLAYER"}
    fallback = {"FORMER_NATIONAL_PLAYER"}

    nat_active = nat[nat["career_state"].isin(active)].copy()
    nat_active["team_name"] = nat_active["team_id"].map(team_map)
    active_teams = set(nat_active["team_name"].dropna().unique())

    # For WC2026 teams with no current players, include former players
    missing_teams = [t for t in WC2026_TEAMS if t not in active_teams]
    if missing_teams:
        nat_former = nat[nat["career_state"].isin(fallback)].copy()
        nat_former["team_name"] = nat_former["team_id"].map(team_map)
        nat_former = nat_former[nat_former["team_name"].isin(missing_teams)]
        nat_active = pd.concat([nat_active, nat_former], ignore_index=True)

    # Keep only WC2026 teams
    nat_wc = nat_active[nat_active["team_name"].isin(WC2026_TEAMS)]

    # Join with values and profiles
    joined = (
        nat_wc[["player_id", "team_name"]]
        .merge(latest, on="player_id", how="inner")
        .merge(pro[["player_id", "date_of_birth", "current_club_name"]], on="player_id", how="left")
    )
    joined["dob"] = pd.to_datetime(joined["date_of_birth"], errors="coerce")
    today = pd.Timestamp.today()
    joined["age"] = ((today - joined["dob"]).dt.days / 365.25).round(1)

    # League quality score from current_club_name (partial match)
    def club_tier(club_name: str) -> int:
        if pd.isna(club_name):
            return 4
        for league, tier in LEAGUE_TIERS.items():
            if league.lower() in str(club_name).lower():
                return tier
        return 4  # unknown

    joined["league_tier"] = joined["current_club_name"].apply(club_tier)

    records = []
    for team, grp in joined.groupby("team_name"):
        grp_sorted = grp.sort_values("value", ascending=False).reset_index(drop=True)

        xi    = grp_sorted.head(11)
        bench = grp_sorted.iloc[11:23]

        xi_val   = xi["value"].sum()
        all_val  = grp_sorted["value"].sum()
        top_val  = grp_sorted["value"].iloc[0] if len(grp_sorted) > 0 else 0
        bench_val = bench["value"].sum()

        records.append({
            "team":                   team,
            "total_squad_value":      int(all_val),
            "avg_starting_xi_value":  int(xi_val / 11) if len(xi) == 11 else int(xi["value"].mean()),
            "squad_depth_ratio":      round(bench_val / xi_val, 3) if xi_val > 0 else 0.0,
            "star_concentration":     round(top_val / all_val, 3) if all_val > 0 else 0.0,
            "avg_age_starting_xi":    round(xi["age"].mean(), 1),
            "league_quality_score":   round(xi["league_tier"].mean(), 2),
            "num_players_tracked":    len(grp_sorted),
        })

    return pd.DataFrame(records)


# ── FBref xG scrape attempt ───────────────────────────────────────────────────

# FBref national team IDs (Transfermarkt → FBref mapping for top teams)
FBREF_IDS: dict[str, str] = {
    "Argentina":    "9a273bde", "Brazil":       "e8ef8c6b",
    "France":       "76f7f8d0", "England":      "8cec06e1",
    "Germany":      "f3608b4d", "Spain":        "cff3d9bb",
    "Portugal":     "ebc8583a", "Netherlands":  "f2b752f1",
    "Belgium":      "fb5e4e5f", "Italy":        "9d139bfb",
    "Croatia":      "cb5b4a42", "Denmark":      "2ccf5ee4",
    "Uruguay":      "3938d466", "Colombia":     "3df36e3f",
    "Morocco":      "d2e81b70", "Senegal":      "3d4c0c2b",
    "Japan":        "1af8e3eb", "South Korea":  "7e6cd3fd",
    "United States":"a2a7c6f8", "Mexico":       "ca5aa04e",
}


def try_fbref_xg(team: str) -> dict | None:
    """Attempt to fetch team xG stats from FBref. Returns None if blocked."""
    fbref_id = FBREF_IDS.get(team)
    if not fbref_id:
        return None
    url = f"https://fbref.com/en/squads/{fbref_id}/{team.replace(' ', '-')}-Stats"
    try:
        time.sleep(4)  # respectful crawl delay
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        # FBref renders tables in HTML comments for Cloudflare bypass
        import re
        html = re.sub(r"<!--(.*?)-->", r"\1", r.text, flags=re.DOTALL)
        tables = pd.read_html(StringIO(html), attrs={"id": "matchlogs_for"})
        if not tables:
            return None
        ml = tables[0]
        ml.columns = [c[1] if isinstance(c, tuple) else c for c in ml.columns]

        # Keep competitive matches only (not friendlies)
        if "Comp" in ml.columns:
            ml = ml[~ml["Comp"].str.lower().str.contains("friendly", na=False)]

        ml = ml.dropna(subset=["xG", "xGA"]).tail(10)
        if ml.empty:
            return None
        return {
            "xg_per90_last10":      round(ml["xG"].astype(float).mean(), 3),
            "xga_per90_last10":     round(ml["xGA"].astype(float).mean(), 3),
            "xg_diff_per90_last10": round(
                (ml["xG"].astype(float) - ml["xGA"].astype(float)).mean(), 3
            ),
            "xg_source": "fbref",
        }
    except Exception:
        return None


# ── Goal-rate xG proxy from results ──────────────────────────────────────────

FRIENDLY_TOURNAMENTS = {"Friendly"}


def goal_rate_xg(results: pd.DataFrame, team: str) -> dict:
    """
    Compute goals-for and goals-against per 90 min in the team's last 10
    competitive (non-friendly) matches as an xG proxy.
    """
    is_home = results["home_team"] == team
    is_away = results["away_team"] == team
    is_comp = ~results["tournament"].isin(FRIENDLY_TOURNAMENTS)

    home_m = results[is_home & is_comp].copy()
    home_m["gf"] = home_m["home_score"]
    home_m["ga"] = home_m["away_score"]

    away_m = results[is_away & is_comp].copy()
    away_m["gf"] = away_m["away_score"]
    away_m["ga"] = away_m["home_score"]

    all_m = pd.concat([home_m[["date", "gf", "ga"]], away_m[["date", "gf", "ga"]]])
    all_m = all_m.sort_values("date").tail(10)

    if all_m.empty:
        return {"xg_per90_last10": np.nan, "xga_per90_last10": np.nan,
                "xg_diff_per90_last10": np.nan, "xg_source": "no_data"}

    xg  = round(all_m["gf"].mean(), 3)
    xga = round(all_m["ga"].mean(), 3)
    return {
        "xg_per90_last10":      xg,
        "xga_per90_last10":     xga,
        "xg_diff_per90_last10": round(xg - xga, 3),
        "xg_source":            "goal_rate_proxy",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def download() -> pd.DataFrame:
    # Load Transfermarkt files
    mv  = fetch_csv("player_market_value",          f"{BASE_TM}/player_market_value/player_market_value.csv")
    nat = fetch_csv("player_national_performances", f"{BASE_TM}/player_national_performances/player_national_performances.csv")
    pro = fetch_csv("player_profiles",              f"{BASE_TM}/player_profiles/player_profiles.csv")

    mv["date"] = pd.to_datetime(mv["date_unix"], errors="coerce")

    # Rebuild team → country mapping (same approach as transfermarkt.py)
    merged = nat[["player_id", "team_id"]].merge(
        pro[["player_id", "citizenship"]], on="player_id", how="left"
    )
    raw_map: dict[int, str] = (
        merged.dropna(subset=["citizenship"])
        .groupby("team_id")["citizenship"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
        .dropna()
        .to_dict()
    )

    # Normalise citizenship strings → WC2026_TEAMS names
    TM_NAME_FIXES: dict[str, str] = {
        "Türkiye":       "Turkey",
        "Korea, South":  "South Korea",
        "Cote d'Ivoire": "Ivory Coast",
        "Morocco  France": "Morocco",
        "Morocco  Spain":  "Morocco",
        "Morocco  Belgium":"Morocco",
        "DR Congo  France":"DR Congo",
        "Jamaica  England":"Jamaica",
        "Canada  Jamaica": "Jamaica",
        "Tunisia  Italy":  "Tunisia",
        "Tunisia  France": "Tunisia",
    }
    team_map: dict[int, str] = {
        tid: TM_NAME_FIXES.get(name, name)
        for tid, name in raw_map.items()
    }

    log.info("Computing Transfermarkt squad metrics ...")
    squad_df = build_squad_metrics(nat, mv, pro, team_map)
    log.info("Got squad metrics for %d teams", len(squad_df))

    # xG metrics
    log.info("Loading results for xG proxy ...")
    results = pd.read_csv(RESULTS_PATH, parse_dates=["date"])

    xg_rows = []
    fbref_success = 0
    for team in squad_df["team"].tolist():
        xg = try_fbref_xg(team)
        if xg:
            fbref_success += 1
        else:
            xg = goal_rate_xg(results, team)
        xg["team"] = team
        xg_rows.append(xg)

    xg_df = pd.DataFrame(xg_rows)
    log.info("xG sources: %d FBref, %d goal-rate proxy",
             fbref_success, len(xg_rows) - fbref_success)

    # Ensure every WC2026 team has a row (fill TM gaps with NaN)
    all_teams_df = pd.DataFrame({"team": WC2026_TEAMS})
    squad_df = all_teams_df.merge(squad_df, on="team", how="left")

    # xG for teams missing from TM (computed from results regardless)
    covered_xg = {r["team"] for r in xg_rows}
    for team in WC2026_TEAMS:
        if team not in covered_xg:
            xg = goal_rate_xg(results, team)
            xg["team"] = team
            xg_rows.append(xg)
    xg_df = pd.DataFrame(xg_rows)

    out = squad_df.merge(xg_df, on="team", how="left")
    out = out.sort_values("total_squad_value", ascending=False).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    log.info("Saved %d rows to %s", len(out), OUT_PATH)
    log.info("\n%s", out[["team", "total_squad_value", "avg_starting_xi_value",
                           "xg_per90_last10", "xg_diff_per90_last10"]].head(10).to_string(index=False))
    return out


if __name__ == "__main__":
    download()

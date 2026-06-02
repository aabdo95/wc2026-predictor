"""
Build squad market value snapshots for national teams.

Source: github.com/salimt/football-datasets (Transfermarkt datalake)

Files used:
  player_market_value.csv        player_id, date_unix, value
  player_national_performances.csv  player_id, team_id, career_state
  player_profiles.csv            player_id, player_name, citizenship

Strategy:
  1. Map team_id → country by taking the modal citizenship of that team's players.
  2. For each WC year (2006-2022) snap each player's value to the closest
     date on or before June 1 of that year.
  3. Aggregate per (team, wc_year): total_squad_value, avg_player_value,
     num_players.
  4. Append a 2026 row using each player's most recent available value.

Output: data/raw/squad_values.csv
        columns: team, wc_year, total_squad_value, avg_player_value, num_players
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE = "https://raw.githubusercontent.com/salimt/football-datasets/main/datalake/transfermarkt"
URLS = {
    "market_value":   f"{BASE}/player_market_value/player_market_value.csv",
    "national_perf":  f"{BASE}/player_national_performances/player_national_performances.csv",
    "profiles":       f"{BASE}/player_profiles/player_profiles.csv",
}

WC_YEARS = [2006, 2010, 2014, 2018, 2022]
WC_SNAP_DATE = "-06-01"   # value snapshot: June 1 of each WC year
OUT_PATH = Path("data/raw/squad_values.csv")


def fetch(name: str, url: str) -> pd.DataFrame:
    log.info("Downloading %s ...", name)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def build_team_country_map(nat: pd.DataFrame, profiles: pd.DataFrame) -> dict[int, str]:
    """Map team_id → country name via modal citizenship of players on that team."""
    merged = nat[["player_id", "team_id"]].merge(
        profiles[["player_id", "citizenship"]], on="player_id", how="left"
    )
    mapping = (
        merged.dropna(subset=["citizenship"])
        .groupby("team_id")["citizenship"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
        .dropna()
        .to_dict()
    )
    return mapping


def snap_values_for_year(mv: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    For each player, return the most recent value on or before June 1 of `year`.
    Only include players who had a value within 3 years of the cutoff — this
    acts as an activity filter so we don't pick up retired players.
    """
    cutoff = pd.Timestamp(f"{year}{WC_SNAP_DATE}")
    window_start = pd.Timestamp(f"{year - 3}-01-01")
    subset = mv[(mv["date"] <= cutoff) & (mv["date"] >= window_start)]
    if subset.empty:
        return pd.DataFrame(columns=["player_id", "value"])
    return (
        subset.sort_values("date")
        .groupby("player_id", as_index=False)
        .last()[["player_id", "value"]]
    )


def aggregate(player_values: pd.DataFrame, nat: pd.DataFrame,
              team_map: dict[int, str], year: int) -> pd.DataFrame:
    """Join player values → national team → country and aggregate."""
    joined = (
        nat[["player_id", "team_id"]]
        .merge(player_values, on="player_id", how="inner")
    )
    joined["team"] = joined["team_id"].map(team_map)
    joined = joined.dropna(subset=["team", "value"])

    agg = (
        joined.groupby("team")["value"]
        .agg(
            total_squad_value="sum",
            avg_player_value="mean",
            num_players="count",
        )
        .reset_index()
    )
    agg["wc_year"] = year
    return agg


def download() -> pd.DataFrame:
    mv  = fetch("player_market_value",          URLS["market_value"])
    nat = fetch("player_national_performances", URLS["national_perf"])
    pro = fetch("player_profiles",              URLS["profiles"])

    # Parse dates
    mv["date"] = pd.to_datetime(mv["date_unix"], errors="coerce")
    mv = mv.dropna(subset=["date", "value"])

    log.info("Building team → country mapping ...")
    team_map = build_team_country_map(nat, pro)
    log.info("Mapped %d national team IDs to countries", len(team_map))

    frames: list[pd.DataFrame] = []

    # Historical WC snapshots — snap_values_for_year already applies a 3-year
    # activity window, so num_players reflects players active around each WC.
    for year in WC_YEARS:
        log.info("Snapping values for WC %d ...", year)
        snapped = snap_values_for_year(mv, year)
        if snapped.empty:
            log.warning("No values found for %d — skipping", year)
            continue
        agg = aggregate(snapped, nat, team_map, year)
        frames.append(agg)
        log.info("  %d teams with value data for %d", len(agg), year)

    # 2026: same 3-year activity window as historical years for consistency
    log.info("Snapping values for WC 2026 ...")
    snapped_2026 = snap_values_for_year(mv, 2026)
    agg_2026 = aggregate(snapped_2026, nat, team_map, 2026)
    frames.append(agg_2026)
    log.info("  %d teams with value data for 2026", len(agg_2026))

    out = pd.concat(frames, ignore_index=True)
    out["total_squad_value"] = out["total_squad_value"].round(0).astype(int)
    out["avg_player_value"]  = out["avg_player_value"].round(0).astype(int)
    out = out[["team", "wc_year", "total_squad_value", "avg_player_value", "num_players"]]
    out = out.sort_values(["wc_year", "total_squad_value"], ascending=[True, False]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    log.info("Saved %d rows to %s", len(out), OUT_PATH)
    log.info("WC years covered: %s", sorted(out["wc_year"].unique().tolist()))
    log.info("Teams covered   : %d unique", out["team"].nunique())
    return out


if __name__ == "__main__":
    download()

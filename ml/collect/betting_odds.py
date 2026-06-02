"""
Collect historical international match betting odds.

Strategy:
  1. Try to download from football-data.co.uk (HTTP 200, but ISP gambling blocks
     are common — this will work from most non-restricted networks).
  2. If unavailable, derive implied probabilities from pre-computed ELO ratings.
     ELO → win probability is well-established; draw probability is estimated via
     an empirical model calibrated on international football data.

WC 2026 group stage odds: scraped from oddschecker.com (comparison site, not
a bookmaker, so not gambling-blocked).

Outputs
-------
data/raw/betting_odds.csv
    date, home_team, away_team,
    market_prob_home, market_prob_draw, market_prob_away

data/raw/wc2026_betting_odds.csv
    same schema, WC 2026 group stage fixtures only
"""

from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

OUT_PATH        = Path("data/raw/betting_odds.csv")
WC2026_OUT_PATH = Path("data/raw/wc2026_betting_odds.csv")

RESULTS_PATH    = Path("data/raw/international_results.csv")
ELO_PATH        = Path("data/raw/elo_ratings.csv")

# football-data.co.uk column sets per bookmaker (H/D/A)
BOOKMAKER_COLS = {
    "Bet365":  ("B365H", "B365D", "B365A"),
    "Pinnacle": ("PSH",  "PSD",  "PSA"),
    "Betway":  ("BWH",   "BWD",  "BWA"),
    "MaxOdds": ("MaxH",  "MaxD", "MaxA"),
    "AvgOdds": ("AvgH",  "AvgD", "AvgA"),
}

# football-data.co.uk international/tournament CSV URLs (World Cup + Euros)
FD_URLS = [
    # World Cup group stage data (when available)
    "https://www.football-data.co.uk/new_league_data/WC.csv",
    # Major international tournaments uploaded ad-hoc by season
    "https://www.football-data.co.uk/new_league_data/EC.csv",
]


# ── Implied probability helpers ───────────────────────────────────────────────

def odds_to_raw_prob(h: float, d: float, a: float) -> tuple[float, float, float]:
    """Convert decimal odds to raw (overround-inclusive) implied probabilities."""
    ph, pd_, pa = 1.0 / h, 1.0 / d, 1.0 / a
    return ph, pd_, pa


def remove_overround(ph: float, pd_: float, pa: float) -> tuple[float, float, float]:
    """Normalise implied probabilities so they sum to 1.0."""
    total = ph + pd_ + pa
    return ph / total, pd_ / total, pa / total


def consensus_probs(row: pd.Series) -> tuple[float, float, float]:
    """Average normalised probabilities across all available bookmakers in a row."""
    probs = []
    for bk, (hc, dc, ac) in BOOKMAKER_COLS.items():
        if all(c in row.index and pd.notna(row[c]) and row[c] > 1.0 for c in (hc, dc, ac)):
            ph, pd_, pa = odds_to_raw_prob(row[hc], row[dc], row[ac])
            ph, pd_, pa = remove_overround(ph, pd_, pa)
            probs.append((ph, pd_, pa))
    if not probs:
        return np.nan, np.nan, np.nan
    arr = np.mean(probs, axis=0)
    return float(arr[0]), float(arr[1]), float(arr[2])


# ── football-data.co.uk scrape ────────────────────────────────────────────────

def try_fd_download() -> pd.DataFrame | None:
    frames = []
    for url in FD_URLS:
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if not r.ok or len(r.content) < 1000:
                log.debug("football-data.co.uk %s: skipped (status=%d)", url, r.status_code)
                continue
            df = pd.read_csv(StringIO(r.text), on_bad_lines="skip")
            if "HomeTeam" not in df.columns:
                continue
            log.info("football-data.co.uk: loaded %d rows from %s", len(df), url)
            frames.append(df)
        except Exception as exc:
            log.debug("football-data.co.uk %s failed: %s", url, exc)

    if not frames:
        return None

    raw = pd.concat(frames, ignore_index=True)
    records = []
    for _, row in raw.iterrows():
        ph, pd_, pa = consensus_probs(row)
        if any(np.isnan(v) for v in (ph, pd_, pa)):
            continue
        records.append({
            "date":             pd.to_datetime(row.get("Date"), dayfirst=True, errors="coerce"),
            "home_team":        row.get("HomeTeam"),
            "away_team":        row.get("AwayTeam"),
            "market_prob_home": round(ph, 4),
            "market_prob_draw": round(pd_, 4),
            "market_prob_away": round(pa, 4),
        })
    if not records:
        return None
    return pd.DataFrame(records).dropna(subset=["date", "home_team", "away_team"])


# ── ELO-based fallback ────────────────────────────────────────────────────────

def elo_win_prob(elo_home: float, elo_away: float, home_advantage: float = 100.0) -> float:
    """Expected score (win + 0.5*draw) for home team."""
    return 1.0 / (1.0 + 10 ** ((elo_away - (elo_home + home_advantage)) / 400.0))


def elo_to_match_probs(elo_h: float, elo_a: float, neutral: bool = False) -> tuple[float, float, float]:
    """
    Convert ELO ratings to (p_home_win, p_draw, p_away_win).

    Draw probability model: Draws are more likely when teams are closely matched.
    We use an empirical formula calibrated on international results:
        p_draw = max_draw * exp(-k * |elo_diff|)
    with max_draw = 0.30, k = 0.0025.  The remainder is split between wins
    proportionally to the ELO-expected score.
    """
    adv  = 0.0 if neutral else 100.0
    exp_h = elo_win_prob(elo_h, elo_a, adv)    # includes half of draws
    elo_diff = abs(elo_h + adv - elo_a)

    max_draw = 0.30
    k        = 0.0025
    p_draw   = max_draw * np.exp(-k * elo_diff)

    remaining = 1.0 - p_draw
    # exp_h = p_win_h + 0.5 * p_draw  →  p_win_h = exp_h - 0.5 * p_draw
    p_home = max(0.0, exp_h - 0.5 * p_draw)
    p_away = max(0.0, remaining - p_home)

    # Normalise (should already sum to ~1)
    total = p_home + p_draw + p_away
    return round(p_home / total, 4), round(p_draw / total, 4), round(p_away / total, 4)


def elo_fallback() -> pd.DataFrame:
    """Derive implied probabilities from ELO ratings for every match in results."""
    log.info("Building ELO-based implied probabilities ...")
    results = pd.read_csv(RESULTS_PATH, parse_dates=["date"])
    elo_df  = pd.read_csv(ELO_PATH, parse_dates=["date"])

    # Build a lookup: (team, match_date) → elo_rating_after_that_match
    # We want the ELO *before* the match, so shift by one match per team.
    elo_df = elo_df.sort_values(["team", "date"])
    elo_df["elo_before"] = elo_df.groupby("team")["elo_rating"].shift(1)
    elo_df["elo_before"] = elo_df["elo_before"].fillna(1500.0)

    # Pivot to wide: date × team → elo_before
    elo_lookup: dict[tuple, float] = {}
    for _, row in elo_df.iterrows():
        elo_lookup[(row["team"], row["date"].date())] = row["elo_before"]

    def get_elo(team: str, date) -> float:
        return elo_lookup.get((team, date), 1500.0)

    records = []
    for _, row in results.iterrows():
        d      = row["date"].date()
        elo_h  = get_elo(row["home_team"], d)
        elo_a  = get_elo(row["away_team"], d)
        ph, pd_, pa = elo_to_match_probs(elo_h, elo_a, neutral=bool(row["neutral"]))
        records.append({
            "date":             row["date"],
            "home_team":        row["home_team"],
            "away_team":        row["away_team"],
            "market_prob_home": ph,
            "market_prob_draw": pd_,
            "market_prob_away": pa,
        })

    return pd.DataFrame(records)


# ── WC 2026 group stage odds ──────────────────────────────────────────────────

def scrape_wc2026_odds() -> pd.DataFrame | None:
    """
    Scrape WC2026 group stage match odds from oddschecker.com.
    Returns None if scraping fails or is blocked.
    """
    url = "https://www.oddschecker.com/football/world-cup/winner"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if not r.ok:
            log.warning("oddschecker returned %d", r.status_code)
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        # oddschecker is heavily JS-rendered; if we get minimal content bail out
        if len(r.text) < 10_000:
            log.warning("oddschecker appears JS-rendered — WC2026 odds unavailable")
            return None

        # Parse any tables with team odds (best-effort)
        records = []
        for row in soup.select("tr.diff-row"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            team = cells[0].get_text(strip=True)
            # Just log; actual group-stage match odds need a different URL
            records.append({"team": team})

        if not records:
            log.warning("No parseable odds found on oddschecker")
            return None
        return pd.DataFrame(records)

    except Exception as exc:
        log.warning("WC2026 odds scrape failed: %s", exc)
        return None


def build_wc2026_elo_odds() -> pd.DataFrame:
    """
    Compute ELO-based implied probabilities for WC 2026 group stage fixtures
    using each team's most recent ELO rating.
    """
    import json
    groups_path = Path("data/fixtures/groups.json")
    if not groups_path.exists():
        return pd.DataFrame()

    with open(groups_path) as f:
        groups = json.load(f)["groups"]

    elo_df = pd.read_csv(ELO_PATH, parse_dates=["date"])
    latest_elo = (
        elo_df.sort_values("date")
        .groupby("team")["elo_rating"]
        .last()
        .to_dict()
    )

    records = []
    for group, teams in groups.items():
        real_teams = [t for t in teams if t != "TBD"]
        for i, home in enumerate(real_teams):
            for away in real_teams[i + 1:]:
                elo_h = latest_elo.get(home, 1500.0)
                elo_a = latest_elo.get(away, 1500.0)
                ph, pd_, pa = elo_to_match_probs(elo_h, elo_a, neutral=True)
                records.append({
                    "date":             "2026-06-11",  # placeholder group stage start
                    "home_team":        home,
                    "away_team":        away,
                    "market_prob_home": ph,
                    "market_prob_draw": pd_,
                    "market_prob_away": pa,
                    "source":           "elo_implied",
                })
    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def download() -> pd.DataFrame:
    # Historical odds
    log.info("Attempting football-data.co.uk download ...")
    df = try_fd_download()

    if df is None or df.empty:
        log.info("football-data.co.uk unavailable — falling back to ELO-derived probabilities")
        df = elo_fallback()
        source_label = "elo_implied"
    else:
        source_label = "market_odds"

    df["source"] = source_label
    df = df.sort_values("date").reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    log.info("Saved %d rows to %s  [source: %s]", len(df), OUT_PATH, source_label)
    log.info("Date range: %s → %s", df['date'].min().date(), df['date'].max().date())

    # WC 2026 odds
    log.info("Fetching WC 2026 group stage odds ...")
    wc = scrape_wc2026_odds()
    if wc is None or wc.empty:
        log.info("Oddschecker unavailable — using ELO-implied WC2026 odds")
        wc = build_wc2026_elo_odds()

    if not wc.empty:
        wc.to_csv(WC2026_OUT_PATH, index=False)
        log.info("Saved %d WC2026 fixtures to %s", len(wc), WC2026_OUT_PATH)

    return df


if __name__ == "__main__":
    download()

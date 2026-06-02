"""
Build historical ELO ratings for all national teams.

Strategy:
  1. Try to scrape eloratings.net (full history endpoint).
  2. If blocked / unavailable, compute from scratch using
     data/raw/international_results.csv with the K-factor and
     home-advantage rules specified below.

Output: data/raw/elo_ratings.csv  (date, team, elo_rating)
        One row per team per match, recorded *after* the match.
"""

from __future__ import annotations

import logging

from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT_PATH = Path("data/raw/elo_ratings.csv")
RESULTS_PATH = Path("data/raw/international_results.csv")

# ── ELO parameters ────────────────────────────────────────────────────────────
STARTING_ELO = 1500
HOME_ADVANTAGE = 100          # added to home team's ELO before expected-score calc

WORLD_CUP_TOURNAMENTS = {"FIFA World Cup"}

CONTINENTAL_TOURNAMENTS = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",                # CONCACAF Gold Cup
    "CONCACAF Championship",
    "OFC Nations Cup",
    "CONCACAF Gold Cup",
}

QUALIFIER_KEYWORDS = ("qualification", "qualifier", "qualifying")

def k_factor(tournament: str) -> int:
    t = tournament.strip()
    if t in WORLD_CUP_TOURNAMENTS:
        return 40
    if t in CONTINENTAL_TOURNAMENTS:
        return 30
    if any(kw in t.lower() for kw in QUALIFIER_KEYWORDS):
        return 20
    if t == "Friendly":
        return 10
    # Nations leagues, regional cups, other competitive → treat as qualifiers
    return 20


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


# ── Scrape attempt ─────────────────────────────────────────────────────────────

def try_scrape() -> pd.DataFrame | None:
    """
    eloratings.net exposes a JSON endpoint used by their frontend.
    Returns a DataFrame(date, team, elo_rating) or None if unavailable.
    """
    url = "https://eloratings.net/World"
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            log.warning("eloratings.net returned HTTP %d — falling back to computation", resp.status_code)
            return None

        # The site is JS-rendered; plain GET won't contain history data.
        # If content is tiny / no table tags it's a SPA shell — bail out.
        if len(resp.text) < 5000 or "<table" not in resp.text.lower():
            log.warning("eloratings.net appears JS-rendered — falling back to computation")
            return None

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        rows = []
        for row in soup.select("table tbody tr"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) >= 3:
                rows.append({"date": pd.Timestamp.today().date(), "team": cells[1], "elo_rating": cells[2]})
        if not rows:
            return None
        return pd.DataFrame(rows)

    except Exception as exc:
        log.warning("Scrape failed (%s) — falling back to computation", exc)
        return None


# ── Compute from scratch ───────────────────────────────────────────────────────

def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replay all matches in chronological order and track each team's ELO
    after every game.  Returns DataFrame(date, team, elo_rating).
    """
    df = df.sort_values("date").reset_index(drop=True)

    ratings: dict[str, float] = {}   # current ELO per team
    records: list[dict] = []

    def get_elo(team: str) -> float:
        return ratings.get(team, STARTING_ELO)

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        h_score, a_score = row["home_score"], row["away_score"]
        neutral = bool(row["neutral"])
        tournament = str(row["tournament"])
        date = row["date"]

        elo_h = get_elo(home)
        elo_a = get_elo(away)

        # Apply home advantage for non-neutral venues
        advantage = 0 if neutral else HOME_ADVANTAGE
        exp_h = expected_score(elo_h + advantage, elo_a)
        exp_a = 1.0 - exp_h

        # Actual scores
        if h_score > a_score:
            act_h, act_a = 1.0, 0.0
        elif h_score < a_score:
            act_h, act_a = 0.0, 1.0
        else:
            act_h = act_a = 0.5

        k = k_factor(tournament)
        new_elo_h = elo_h + k * (act_h - exp_h)
        new_elo_a = elo_a + k * (act_a - exp_a)

        ratings[home] = new_elo_h
        ratings[away] = new_elo_a

        records.append({"date": date, "team": home, "elo_rating": round(new_elo_h, 2)})
        records.append({"date": date, "team": away, "elo_rating": round(new_elo_a, 2)})

    return pd.DataFrame(records)


# ── Main ───────────────────────────────────────────────────────────────────────

def download() -> pd.DataFrame:
    out = try_scrape()

    if out is None:
        log.info("Computing ELO from %s", RESULTS_PATH)
        results = pd.read_csv(RESULTS_PATH, parse_dates=["date"])
        out = compute_elo(results)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    log.info("Rows      : %d", len(out))
    log.info("Teams     : %d unique", out["team"].nunique())
    log.info("Date range: %s → %s", out["date"].min(), out["date"].max())
    return out


if __name__ == "__main__":
    download()

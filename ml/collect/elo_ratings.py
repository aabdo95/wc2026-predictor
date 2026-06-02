"""Scrape current ELO ratings from eloratings.net."""

import logging
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://www.eloratings.net/World"
OUT_PATH = Path("data/raw/elo_ratings.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def scrape() -> pd.DataFrame:
    log.info("Fetching ELO ratings from %s", URL)
    resp = requests.get(URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = []
    for row in soup.select("table tbody tr"):
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) >= 3:
            rows.append({"rank": cells[0], "team": cells[1], "elo": cells[2]})

    df = pd.DataFrame(rows)
    df["elo"] = pd.to_numeric(df["elo"], errors="coerce")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    log.info("Saved %d teams to %s", len(df), OUT_PATH)
    return df


if __name__ == "__main__":
    scrape()

"""Download FIFA world rankings (latest snapshot via FIFA public API)."""

import logging
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://www.fifa.com/en/fifa-world-ranking/men"
OUT_PATH = Path("data/raw/fifa_rankings.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def fetch() -> pd.DataFrame:
    # FIFA rankings JSON endpoint (reverse-engineered from public site)
    api = "https://datasport.fifa.com/api/ranking/men?lang=en&date=&confederation="
    log.info("Fetching FIFA rankings")
    resp = requests.get(api, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    data = resp.json()
    rankings = data.get("rankings", [])
    rows = [
        {
            "rank": r.get("rankingPosition"),
            "team": r.get("teamName", {}).get("longName"),
            "fifa_points": r.get("totalPoints"),
            "confederation": r.get("confederationCode"),
        }
        for r in rankings
    ]

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    log.info("Saved %d teams to %s", len(df), OUT_PATH)
    return df


if __name__ == "__main__":
    fetch()

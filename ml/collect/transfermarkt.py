"""Scrape national team squad market values from Transfermarkt."""

import logging
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.transfermarkt.com/wettbewerbe/nationalmannschaften/plus/1"
OUT_PATH = Path("data/raw/transfermarkt_values.csv")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; wc2026-predictor/1.0)"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def scrape() -> pd.DataFrame:
    log.info("Fetching squad values from Transfermarkt")
    resp = requests.get(BASE_URL, timeout=30, headers=HEADERS)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = []
    for row in soup.select("table.items tbody tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        rows.append(
            {
                "team": cells[1].get_text(strip=True),
                "players": cells[2].get_text(strip=True),
                "avg_age": cells[3].get_text(strip=True),
                "market_value": cells[-1].get_text(strip=True),
            }
        )
        time.sleep(0.1)

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    log.info("Saved %d teams to %s", len(df), OUT_PATH)
    return df


if __name__ == "__main__":
    scrape()

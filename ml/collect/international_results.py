"""Download international football match results from martj42/international_results."""

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

RAW_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
OUT_PATH = Path("data/raw/international_results.csv")
COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "city", "country", "neutral"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def download() -> pd.DataFrame:
    log.info("Fetching %s", RAW_URL)
    resp = requests.get(RAW_URL, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))[COLUMNS]
    df["date"] = pd.to_datetime(df["date"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    log.info("Rows      : %d", len(df))
    log.info("Date range: %s → %s", df["date"].min().date(), df["date"].max().date())
    return df


if __name__ == "__main__":
    download()

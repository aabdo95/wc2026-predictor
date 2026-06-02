"""
Feature engineering pipeline for WC2026 match prediction.

Reads all raw data files and produces data/processed/match_features.csv.

LEAKAGE CONTRACT: for any match on date D, every feature is derived
exclusively from observations with date < D (strict inequality).
A 100-row random assertion validates this at runtime.

Feature groups
--------------
T1  ELO strength          elo_home / elo_away / elo_diff / elo_expected_home
T2  Rolling form          last-5 and last-10 win/draw/goals (all matches)
                          last-10 win/goals competitive matches only
T3  Head-to-head          h2h_count / win_rates / avg goals
T4  Squad value           total_squad_value / avg_xi_value / value_ratio

Context                   tournament_tier / is_neutral / is_wc / is_continental
Target                    outcome  {home_win, draw, away_win}
Weight                    sample_weight  (2x for WC / continental)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW       = Path("data/raw")
PROCESSED = Path("data/processed")

# ── Tournament taxonomy ───────────────────────────────────────────────────────

WC_TOURNAMENTS: set[str] = {"FIFA World Cup"}

CONTINENTAL_TOURNAMENTS: set[str] = {
    "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
    "Gold Cup", "CONCACAF Championship", "OFC Nations Cup", "CONCACAF Gold Cup",
}

QUALIFIER_KEYWORDS = ("qualification", "qualifier", "qualifying")


def _tournament_tier(t: str) -> int:
    """4 = World Cup, 3 = continental, 2 = qualifier / nations league, 1 = friendly."""
    if t in WC_TOURNAMENTS:
        return 4
    if t in CONTINENTAL_TOURNAMENTS:
        return 3
    if any(k in t.lower() for k in QUALIFIER_KEYWORDS):
        return 2
    if t == "Friendly":
        return 1
    return 2  # nations leagues, regional cups → qualifier tier


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(RAW / "international_results.csv", parse_dates=["date"])
    elo     = pd.read_csv(RAW / "elo_ratings.csv",           parse_dates=["date"])
    squad   = pd.read_csv(RAW / "squad_values.csv")
    return results, elo, squad


# ── T1: ELO features ─────────────────────────────────────────────────────────

def add_elo_features(df: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """
    Join ELO as of the last entry STRICTLY BEFORE the match date.
    Implementation: subtract 1 day from match date then merge_asof backward,
    which is equivalent to strict < D while supporting vectorised lookup.
    """
    # Deduplicate: keep last ELO entry per (team, date); sort by date for merge_asof
    elo_dedup = (
        elo.sort_values("date")
        .groupby(["team", "date"], as_index=False)
        .last()
        .sort_values("date")        # merge_asof requires global date sort on right df
    )

    def _join(team_col: str) -> np.ndarray:
        tmp = df[["date", team_col]].copy().reset_index()
        tmp["lookup"] = tmp["date"] - pd.Timedelta(days=1)
        # Drop original date to avoid duplicate column after rename
        tmp = tmp.drop(columns="date").rename(
            columns={team_col: "team", "lookup": "date"}
        )[["index", "date", "team"]].sort_values("date")
        merged = pd.merge_asof(
            tmp,
            elo_dedup[["date", "team", "elo_rating"]],
            on="date",
            by="team",
            direction="backward",
        )
        return merged.sort_values("index")["elo_rating"].to_numpy()

    df = df.copy()
    df["elo_home"] = _join("home_team")
    df["elo_away"] = _join("away_team")
    df["elo_home"] = df["elo_home"].fillna(1500.0)
    df["elo_away"] = df["elo_away"].fillna(1500.0)

    adv = np.where(df["neutral"], 0.0, 100.0)
    df["elo_home_adj"]      = df["elo_home"] + adv
    df["elo_diff"]          = df["elo_home"] - df["elo_away"]
    df["elo_diff_adj"]      = df["elo_home_adj"] - df["elo_away"]
    df["elo_expected_home"] = 1.0 / (1.0 + 10.0 ** (
        (df["elo_away"] - df["elo_home_adj"]) / 400.0
    ))
    return df


# ── T2: Rolling form features ─────────────────────────────────────────────────

def _team_perspective(results: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, from that team's perspective."""
    home = results[["date", "home_team", "away_team",
                    "home_score", "away_score", "tournament"]].copy()
    home = home.rename(columns={"home_team": "team", "away_team": "opponent",
                                 "home_score": "gf",   "away_score": "ga"})
    home["venue"] = "home"

    away = results[["date", "away_team", "home_team",
                    "away_score", "home_score", "tournament"]].copy()
    away = away.rename(columns={"away_team": "team", "home_team": "opponent",
                                 "away_score": "gf",  "home_score": "ga"})
    away["venue"] = "away"

    tp = pd.concat([home, away], ignore_index=True)
    tp["win"]     = (tp["gf"] > tp["ga"]).astype(float)
    tp["draw"]    = (tp["gf"] == tp["ga"]).astype(float)
    tp["loss"]    = (tp["gf"] < tp["ga"]).astype(float)
    tp["gd"]      = tp["gf"] - tp["ga"]
    tp["is_comp"] = (tp["tournament"] != "Friendly").astype(float)
    return tp.sort_values(["team", "date"]).reset_index(drop=True)


def _prior_rolling(series: pd.Series, window: int) -> pd.Series:
    """Mean of the prior `window` entries within a sorted group — excludes current row."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    tp = _team_perspective(df)

    # Compute rolling on UNIQUE (team, date) entries.
    # Teams occasionally play two games in one day (historic tournaments);
    # deduplicating ensures both games receive the same pre-date form value
    # and prevents the second same-day game from leaking the first game's stats.
    tp_sorted = tp.sort_values(["team", "date"])
    tp_uniq   = tp_sorted.drop_duplicates(["team", "date"], keep="first").copy()

    for w in [5, 10]:
        for col in ["win", "draw", "gf", "ga", "gd"]:
            tp_uniq[f"{col}_last{w}"] = (
                tp_uniq.groupby("team")[col]
                .transform(lambda x, _w=w: _prior_rolling(x, _w))
            )

    # Competitive-only rolling on unique competitive entries
    comp_uniq = tp_sorted[tp_sorted["is_comp"] == 1.0].drop_duplicates(
        ["team", "date"], keep="first"
    ).copy()
    for col in ["win", "gf", "ga"]:
        comp_uniq[f"comp_{col}_last10"] = (
            comp_uniq.groupby("team")[col]
            .transform(lambda x: _prior_rolling(x, 10))
        )

    # Merge competitive form back into tp_uniq via merge_asof (both globally date-sorted)
    comp_cols = ["team", "date", "comp_win_last10", "comp_gf_last10", "comp_ga_last10"]
    tp_uniq = pd.merge_asof(
        tp_uniq.sort_values("date"),
        comp_uniq[comp_cols].sort_values("date").rename(columns={"date": "_cdate"}),
        left_on="date",
        right_on="_cdate",
        by="team",
        direction="backward",
    )

    feat_cols = (
        [f"{c}_last{w}" for w in [5, 10] for c in ["win", "draw", "gf", "ga", "gd"]]
        + ["comp_win_last10", "comp_gf_last10", "comp_ga_last10"]
    )

    # Pivot to home / away and merge onto match-level df
    # Use only unique (team, date) rows — no duplicates can arise
    home_f = (
        tp_uniq[tp_uniq["venue"] == "home"][["team", "date"] + feat_cols]
        .drop_duplicates(["team", "date"])
        .rename(columns={"team": "home_team", **{c: f"home_{c}" for c in feat_cols}})
    )
    away_f = (
        tp_uniq[tp_uniq["venue"] == "away"][["team", "date"] + feat_cols]
        .drop_duplicates(["team", "date"])
        .rename(columns={"team": "away_team", **{c: f"away_{c}" for c in feat_cols}})
    )

    df = df.merge(home_f, on=["home_team", "date"], how="left")
    df = df.merge(away_f, on=["away_team", "date"], how="left")
    return df


# ── T3: Head-to-head features ─────────────────────────────────────────────────

def add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative stats from ALL prior meetings between the two teams (strict < D).
    Uses a canonical alphabetical pair key so stats accumulate regardless of
    which team was home/away in each prior meeting.
    """
    w = df[["date", "home_team", "away_team", "home_score", "away_score"]].copy()

    w["team_lo"]   = w.apply(lambda r: min(r["home_team"], r["away_team"]), axis=1)
    w["team_hi"]   = w.apply(lambda r: max(r["home_team"], r["away_team"]), axis=1)
    w["lo_is_home"]= w["home_team"] == w["team_lo"]

    # Stats from team_lo's perspective
    w["lo_win"]  = np.where(w["lo_is_home"],
                            (w["home_score"] > w["away_score"]).astype(float),
                            (w["away_score"] > w["home_score"]).astype(float))
    w["m_draw"]  = (w["home_score"] == w["away_score"]).astype(float)
    w["lo_gf"]   = np.where(w["lo_is_home"], w["home_score"], w["away_score"])
    w["hi_gf"]   = np.where(w["lo_is_home"], w["away_score"], w["home_score"])

    w = w.sort_values(["team_lo", "team_hi", "date"])

    # Aggregate at (pair, date) level so same-day same-pair matches share the
    # same "prior" stats — this ensures strict-before semantics even when two
    # teams play each other twice on the same calendar day (rare in early data).
    pair_date = (
        w.groupby(["team_lo", "team_hi", "date"], as_index=False)
        [["lo_win", "m_draw", "lo_gf", "hi_gf"]]
        .sum()
        .sort_values(["team_lo", "team_hi", "date"])
    )
    pair_date["n_matches"] = (
        w.groupby(["team_lo", "team_hi", "date"])
        .size().reset_index(name="n")["n"].values
    )

    # Cumulative PRIOR stats on the date-level aggregates (shift → cumsum)
    for col in ["lo_win", "m_draw", "lo_gf", "hi_gf", "n_matches"]:
        pair_date[f"cum_{col}"] = (
            pair_date.groupby(["team_lo", "team_hi"])[col]
            .transform(lambda x: x.shift(1).expanding(min_periods=0).sum().fillna(0))
        )

    # h2h_count = cumulative n_matches before this date
    pair_date = pair_date.rename(columns={"cum_n_matches": "h2h_count"})
    pair_date["h2h_count"] = pair_date["h2h_count"].astype(int)

    # Join prior-stats back to individual match rows
    join_cols = ["cum_lo_win", "cum_m_draw", "cum_lo_gf", "cum_hi_gf", "h2h_count"]
    w = w.merge(
        pair_date[["team_lo", "team_hi", "date"] + join_cols],
        on=["team_lo", "team_hi", "date"],
        how="left",
    )

    # Rates — undefined (NaN) for first meeting
    n = w["h2h_count"].replace(0, np.nan)
    lo_wr  = w["cum_lo_win"]  / n
    dr     = w["cum_m_draw"]  / n
    lo_gfa = w["cum_lo_gf"]   / n
    hi_gfa = w["cum_hi_gf"]   / n

    # Map to home / away perspective
    w["h2h_home_win_rate"]  = np.where(w["lo_is_home"], lo_wr,  1 - lo_wr - dr).astype(float)
    w["h2h_away_win_rate"]  = np.where(w["lo_is_home"], 1 - lo_wr - dr, lo_wr).astype(float)
    w["h2h_draw_rate"]      = dr
    w["h2h_avg_goals_home"] = np.where(w["lo_is_home"], lo_gfa, hi_gfa).astype(float)
    w["h2h_avg_goals_away"] = np.where(w["lo_is_home"], hi_gfa, lo_gfa).astype(float)
    w["h2h_gd_avg"]         = w["h2h_avg_goals_home"] - w["h2h_avg_goals_away"]

    # Fill first-meeting NaN with neutral priors
    w["h2h_home_win_rate"]  = w["h2h_home_win_rate"].fillna(1 / 3)
    w["h2h_away_win_rate"]  = w["h2h_away_win_rate"].fillna(1 / 3)
    w["h2h_draw_rate"]      = w["h2h_draw_rate"].fillna(1 / 3)
    w["h2h_avg_goals_home"] = w["h2h_avg_goals_home"].fillna(1.5)
    w["h2h_avg_goals_away"] = w["h2h_avg_goals_away"].fillna(1.5)
    w["h2h_gd_avg"]         = w["h2h_gd_avg"].fillna(0.0)

    h2h_cols = [
        "h2h_count", "h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate",
        "h2h_avg_goals_home", "h2h_avg_goals_away", "h2h_gd_avg",
    ]
    return df.merge(
        w[["date", "home_team", "away_team"] + h2h_cols],
        on=["date", "home_team", "away_team"],
        how="left",
    )


# ── T4: Squad value features ──────────────────────────────────────────────────

def add_squad_features(df: pd.DataFrame, squad: pd.DataFrame) -> pd.DataFrame:
    """
    Joins the most recent WC-cycle squad value snapshot strictly before each
    match.  Snapshots are anchored to Jan 1 of each WC year so intra-year
    matches all receive the same start-of-cycle values.
    """
    squad = squad.copy()
    squad["snapshot_date"] = pd.to_datetime(squad["wc_year"].astype(str) + "-01-01")
    squad = squad.sort_values("snapshot_date")   # merge_asof requires global sort

    def _join(team_col: str, prefix: str) -> pd.DataFrame:
        tmp = df[["date", team_col]].copy().reset_index()
        tmp["lookup"] = tmp["date"] - pd.Timedelta(days=1)
        tmp = (
            tmp.drop(columns="date")
            .rename(columns={team_col: "team", "lookup": "snapshot_date"})[
                ["index", "snapshot_date", "team"]
            ]
            .sort_values("snapshot_date")
        )
        merged = pd.merge_asof(
            tmp,
            squad[["team", "snapshot_date", "total_squad_value", "avg_player_value"]],
            on="snapshot_date",
            by="team",
            direction="backward",
        )
        merged = merged.sort_values("index")
        return pd.DataFrame({
            f"{prefix}_squad_value":      merged["total_squad_value"].to_numpy(),
            f"{prefix}_avg_player_value": merged["avg_player_value"].to_numpy(),
        })

    df = df.copy()
    home_sq = _join("home_team", "home")
    away_sq = _join("away_team", "away")

    df["home_squad_value"]      = home_sq["home_squad_value"].values
    df["away_squad_value"]      = away_sq["away_squad_value"].values
    df["home_avg_player_value"] = home_sq["home_avg_player_value"].values
    df["away_avg_player_value"] = away_sq["away_avg_player_value"].values
    df["squad_value_ratio"]     = np.where(
        df["away_squad_value"].notna() & (df["away_squad_value"] > 0),
        df["home_squad_value"] / df["away_squad_value"],
        np.nan,
    )
    return df


# ── Context & target ──────────────────────────────────────────────────────────

def add_context_and_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tournament_tier"] = df["tournament"].apply(_tournament_tier)
    df["is_neutral"]      = df["neutral"].astype(int)
    df["is_wc"]           = df["tournament"].isin(WC_TOURNAMENTS).astype(int)
    df["is_continental"]  = df["tournament"].isin(CONTINENTAL_TOURNAMENTS).astype(int)
    df["is_qualifier"]    = df["tournament"].str.lower().str.contains(
        "|".join(QUALIFIER_KEYWORDS), na=False
    ).astype(int)
    df["home_advantage"]  = np.where(df["neutral"], 0, 100)

    df["outcome"] = np.where(
        df["home_score"] > df["away_score"], "home_win",
        np.where(df["home_score"] < df["away_score"], "away_win", "draw"),
    )
    df["sample_weight"] = np.where(df["tournament_tier"] >= 3, 2.0, 1.0)
    return df


# ── Leakage validation ────────────────────────────────────────────────────────

def validate_no_leakage(
    features: pd.DataFrame,
    results: pd.DataFrame,
    elo: pd.DataFrame,
    n_samples: int = 100,
    rng_seed: int = 42,
) -> None:
    """
    Re-derives three key features for 100 random rows directly from raw data
    and asserts they match the pipeline output.

    Checks:
      1. elo_home / elo_away   — last raw ELO with date strictly < D
      2. home_win_last5        — mean wins in last-5 home-team matches before D
      3. h2h_count             — count of prior meetings before D
    """
    log.info("Validating leakage for %d random rows ...", n_samples)

    sample = features.sample(n=n_samples, random_state=rng_seed)
    elo_dedup = (
        elo.sort_values("date")
        .groupby(["team", "date"], as_index=False).last()
    )
    failures: list[str] = []

    for _, row in sample.iterrows():
        D         = row["date"]
        home      = row["home_team"]
        away      = row["away_team"]

        # 1. ELO check
        for team, feat in [(home, "elo_home"), (away, "elo_away")]:
            prior = elo_dedup[(elo_dedup["team"] == team) & (elo_dedup["date"] < D)]
            expected = prior.sort_values("date")["elo_rating"].iloc[-1] if len(prior) else 1500.0
            actual   = row[feat]
            if not np.isclose(expected, actual, rtol=1e-4):
                failures.append(
                    f"ELO @ {D.date()} {home} vs {away}: "
                    f"{feat}={actual:.1f} expected={expected:.1f}"
                )

        # 2. home_win_last5 check
        home_matches = results[
            (results["date"] < D) &
            ((results["home_team"] == home) | (results["away_team"] == home))
        ].sort_values("date").tail(5)
        if len(home_matches) >= 1:
            wins = np.where(
                home_matches["home_team"] == home,
                home_matches["home_score"] > home_matches["away_score"],
                home_matches["away_score"] > home_matches["home_score"],
            ).astype(float).mean()
            actual = row.get("home_win_last5", np.nan)
            if not pd.isna(actual) and not np.isclose(wins, actual, atol=1e-3):
                failures.append(
                    f"Form @ {D.date()} {home}: "
                    f"home_win_last5={actual:.3f} expected={wins:.3f}"
                )

        # 3. h2h_count check
        prior_h2h = results[
            (results["date"] < D) &
            (
                ((results["home_team"] == home) & (results["away_team"] == away)) |
                ((results["home_team"] == away) & (results["away_team"] == home))
            )
        ]
        expected_count = len(prior_h2h)
        actual_count   = int(row.get("h2h_count", 0))
        if actual_count != expected_count:
            failures.append(
                f"H2H @ {D.date()} {home} vs {away}: "
                f"h2h_count={actual_count} expected={expected_count}"
            )

    if failures:
        for msg in failures[:10]:
            log.error("  FAIL: %s", msg)
        raise AssertionError(
            f"Leakage validation FAILED: {len(failures)}/{n_samples} rows. "
            f"First failure: {failures[0]}"
        )
    log.info("Leakage validation PASSED (%d / %d rows OK)", n_samples, n_samples)


# ── Column documentation ─────────────────────────────────────────────────────

FEATURE_DOCS: dict[str, str] = {
    "date":                  "Match date",
    "home_team":             "Home team",
    "away_team":             "Away team",
    "tournament":            "Tournament (raw string)",
    "home_score":            "Full-time goals — home",
    "away_score":            "Full-time goals — away",
    "neutral":               "Neutral venue flag",
    # T1
    "elo_home":              "Home ELO before match (strict < D, cold-start = 1500)",
    "elo_away":              "Away ELO before match (strict < D)",
    "elo_diff":              "elo_home − elo_away (no home bonus)",
    "elo_diff_adj":          "elo_home+bonus − elo_away (bonus=100 if not neutral)",
    "elo_home_adj":          "elo_home + home_advantage",
    "elo_expected_home":     "ELO-formula expected score for home team",
    # T2 — all matches
    **{f"home_{c}_last{w}": f"Home team {c.replace('_',' ')} avg, last-{w} matches before D"
       for w in [5, 10] for c in ["win", "draw", "gf", "ga", "gd"]},
    **{f"away_{c}_last{w}": f"Away team {c.replace('_',' ')} avg, last-{w} matches before D"
       for w in [5, 10] for c in ["win", "draw", "gf", "ga", "gd"]},
    # T2 — competitive only
    "home_comp_win_last10":  "Home team win rate, last-10 competitive before D",
    "home_comp_gf_last10":   "Home team avg goals, last-10 competitive",
    "home_comp_ga_last10":   "Home team avg conceded, last-10 competitive",
    "away_comp_win_last10":  "Away team win rate, last-10 competitive before D",
    "away_comp_gf_last10":   "Away team avg goals, last-10 competitive",
    "away_comp_ga_last10":   "Away team avg conceded, last-10 competitive",
    # T3
    "h2h_count":             "Prior meetings between the two teams (strict < D)",
    "h2h_home_win_rate":     "Home team win rate in prior H2H",
    "h2h_draw_rate":         "Draw rate in prior H2H",
    "h2h_away_win_rate":     "Away team win rate in prior H2H",
    "h2h_avg_goals_home":    "Avg home-team goals in prior H2H",
    "h2h_avg_goals_away":    "Avg away-team goals in prior H2H",
    "h2h_gd_avg":            "Avg goal diff (home − away) in prior H2H",
    # T4
    "home_squad_value":      "Home squad total market value (WC-cycle snapshot)",
    "away_squad_value":      "Away squad total market value",
    "home_avg_player_value": "Home avg player value",
    "away_avg_player_value": "Away avg player value",
    "squad_value_ratio":     "home_squad_value / away_squad_value",
    # Context
    "tournament_tier":       "4=WC  3=continental  2=qualifier/nations  1=friendly",
    "is_neutral":            "1 if neutral venue",
    "is_wc":                 "1 if FIFA World Cup match",
    "is_continental":        "1 if continental championship",
    "is_qualifier":          "1 if qualification match",
    "home_advantage":        "100 if home ground, 0 if neutral",
    # Target
    "outcome":               "home_win | draw | away_win",
    "sample_weight":         "2.0 for WC/continental, 1.0 otherwise",
}


# ── Main ──────────────────────────────────────────────────────────────────────

def build() -> pd.DataFrame:
    log.info("Loading raw data ...")
    results, elo, squad = load_data()
    results = results.dropna(subset=["home_score", "away_score"]).copy()
    log.info("Matches with scores: %d", len(results))

    log.info("Adding context & target ...")
    df = add_context_and_target(results)

    log.info("T1: ELO features ...")
    df = add_elo_features(df, elo)

    log.info("T2: Rolling form features ...")
    df = add_form_features(df)

    log.info("T3: Head-to-head features ...")
    df = add_h2h_features(df)

    log.info("T4: Squad value features ...")
    df = add_squad_features(df, squad)

    # Leakage gate — will raise AssertionError if any check fails
    validate_no_leakage(df, results, elo, n_samples=100)

    # Filter to 2000+
    df = df[df["date"].dt.year >= 2000].copy()
    log.info("Rows after year-2000 filter: %d", len(df))

    # Final column order
    id_cols      = ["date", "home_team", "away_team", "tournament",
                    "home_score", "away_score", "neutral"]
    context_cols = ["tournament_tier", "is_neutral", "is_wc", "is_continental",
                    "is_qualifier", "home_advantage"]
    elo_cols     = ["elo_home", "elo_away", "elo_diff", "elo_diff_adj",
                    "elo_home_adj", "elo_expected_home"]
    form_cols    = (
        [f"home_{c}_last{w}" for w in [5, 10] for c in ["win", "draw", "gf", "ga", "gd"]]
        + ["home_comp_win_last10", "home_comp_gf_last10", "home_comp_ga_last10"]
        + [f"away_{c}_last{w}" for w in [5, 10] for c in ["win", "draw", "gf", "ga", "gd"]]
        + ["away_comp_win_last10", "away_comp_gf_last10", "away_comp_ga_last10"]
    )
    h2h_cols     = ["h2h_count", "h2h_home_win_rate", "h2h_draw_rate",
                    "h2h_away_win_rate", "h2h_avg_goals_home",
                    "h2h_avg_goals_away", "h2h_gd_avg"]
    squad_cols   = ["home_squad_value", "away_squad_value",
                    "home_avg_player_value", "away_avg_player_value",
                    "squad_value_ratio"]
    target_cols  = ["outcome", "sample_weight"]

    ordered = (id_cols + context_cols + elo_cols + form_cols
               + h2h_cols + squad_cols + target_cols)
    ordered = [c for c in ordered if c in df.columns]
    df = df[ordered].reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "match_features.csv"
    df.to_csv(out_path, index=False)

    log.info("Saved  %d rows × %d cols  →  %s", len(df), len(df.columns), out_path)
    log.info("Outcome distribution:\n%s",
             df["outcome"].value_counts(normalize=True).mul(100).round(1)
             .rename("pct").to_string())
    log.info("Null rates (top features with missing data):\n%s",
             df.isnull().mean().where(lambda x: x > 0).dropna()
             .sort_values(ascending=False).head(10).round(3).to_string())
    log.info("\n=== Feature documentation ===")
    for col in ordered:
        log.info("  %-30s  %s", col, FEATURE_DOCS.get(col, ""))

    return df


if __name__ == "__main__":
    build()

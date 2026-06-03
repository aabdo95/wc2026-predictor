"""
Dixon-Coles Poisson model for international football match prediction.

Reference: Dixon & Coles (1997) "Modelling Association Football Scores
           and Inefficiencies in the Football Betting Market".

Parameters estimated via MLE:
  α[team]  — attack strength
  β[team]  — defence strength
  γ        — home advantage (multiplicative on λ)
  ρ        — low-score correlation correction

Time-decay weighting: w(t) = exp(-ξ * days_before_reference)
  ξ = 0.003 per day  (≈ half-weight at ~231 days, as in the paper)

Only competitive matches (WC, continental, qualifiers) are used for fitting.

Outputs
-------
ml/models/dixon_coles_params.json
    teams, alpha, beta, gamma, rho, fit_date, num_matches, log_loss
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson
from sklearn.metrics import log_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESULTS_PATH  = Path("data/raw/international_results.csv")
MODELS_DIR    = Path("ml/models")
PARAMS_PATH   = MODELS_DIR / "dixon_coles_params.json"

XI               = 0.003   # time-decay rate (days⁻¹)
MAX_GOALS        = 10      # scoreline matrix dimension
MIN_MATCHES      = 10      # minimum matches to keep a team in the model
SHRINK_PRIOR     = 60      # Bayesian prior strength (pseudo-observations toward global mean)
XG_CAP           = 3.0     # hard cap on expected goals per team per match
DATA_START       = "1990-01-01"  # older matches add noise, not signal


# ── Low-score correction (vectorized) ────────────────────────────────────────

def tau_vectorized(x: np.ndarray, y: np.ndarray, mu: np.ndarray, nu: np.ndarray, rho: float) -> np.ndarray:
    """Vectorized Dixon-Coles correction factor."""
    result = np.ones(len(x))
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    result[m00] = 1.0 - mu[m00] * nu[m00] * rho
    result[m01] = 1.0 + mu[m01] * rho
    result[m10] = 1.0 + nu[m10] * rho
    result[m11] = 1.0 - rho
    return result


def tau_scalar(x: int, y: int, mu: float, nu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - mu * nu * rho
    elif x == 0 and y == 1:
        return 1.0 + mu * rho
    elif x == 1 and y == 0:
        return 1.0 + nu * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


# ── Model class ──────────────────────────────────────────────────────────────

class DixonColesModel:
    def __init__(self, teams: list[str], alpha: dict[str, float],
                 beta: dict[str, float], gamma: float, rho: float) -> None:
        self.teams = teams
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.rho   = rho

    def _expected_goals_raw(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        adv = self.gamma if not neutral else 1.0
        mu  = self.alpha[home] * self.beta[away] * adv
        nu  = self.alpha[away] * self.beta[home]
        return mu, nu

    def _expected_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        mu, nu = self._expected_goals_raw(home, away, neutral)
        return min(mu, XG_CAP), min(nu, XG_CAP)

    def predict_expected_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        return self._expected_goals(home, away, neutral)

    def predict_scoreline_probs(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        mu, nu = self._expected_goals(home, away, neutral)
        mat = np.zeros((MAX_GOALS, MAX_GOALS))
        for i in range(MAX_GOALS):
            for j in range(MAX_GOALS):
                mat[i, j] = (
                    poisson.pmf(i, mu) * poisson.pmf(j, nu)
                    * tau_scalar(i, j, mu, nu, self.rho)
                )
        mat /= mat.sum()
        return mat

    def predict_outcome_probs(self, home: str, away: str, neutral: bool = False) -> tuple[float, float, float]:
        mat = self.predict_scoreline_probs(home, away, neutral)
        p_h = float(np.tril(mat, -1).sum())
        p_d = float(np.trace(mat))
        p_a = float(np.triu(mat, 1).sum())
        total = p_h + p_d + p_a
        return p_h / total, p_d / total, p_a / total

    def to_dict(self) -> dict:
        return {"teams": self.teams, "alpha": self.alpha, "beta": self.beta,
                "gamma": self.gamma, "rho": self.rho}

    @classmethod
    def from_dict(cls, d: dict) -> "DixonColesModel":
        return cls(d["teams"], d["alpha"], d["beta"], d["gamma"], d["rho"])


# ── Data preparation ─────────────────────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Competitive only
    df = df[~df["tournament"].str.lower().str.contains("friendly", na=False)].copy()

    # Post-1990 for relevance (older data is noise given time-decay)
    df = df[df["date"] >= DATA_START].copy()

    log.info("Loaded %d competitive matches (%s → %s)",
             len(df), df["date"].min().date(), df["date"].max().date())
    return df


def filter_teams(df: pd.DataFrame) -> list[str]:
    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    return sorted(counts[counts >= MIN_MATCHES].index.tolist())


def compute_weights(df: pd.DataFrame, ref_date: pd.Timestamp) -> np.ndarray:
    days_before = (ref_date - df["date"]).dt.days.clip(lower=0).values
    return np.exp(-XI * days_before)


# ── Fully vectorized NLL ──────────────────────────────────────────────────────

def build_nll_function(
    home_idx: np.ndarray, away_idx: np.ndarray,
    home_goals: np.ndarray, away_goals: np.ndarray,
    neutral: np.ndarray, weights: np.ndarray, n_teams: int,
):
    """Returns a fast NLL closure with precomputed constants."""
    lgamma_h = gammaln(home_goals + 1)
    lgamma_a = gammaln(away_goals + 1)
    n_matches = len(home_goals)

    def nll(params: np.ndarray) -> float:
        log_alpha = params[:n_teams]
        log_beta  = params[n_teams:2*n_teams]
        log_gamma = params[2*n_teams]
        rho       = np.clip(params[2*n_teams + 1], -0.99, 0.99)

        alpha = np.exp(log_alpha)
        beta  = np.exp(log_beta)
        gamma = np.exp(log_gamma)

        mu = alpha[home_idx] * beta[away_idx] * np.where(neutral == 0, gamma, 1.0)
        nu = alpha[away_idx] * beta[home_idx]

        # Clamp for numerical stability
        mu = np.clip(mu, 1e-8, None)
        nu = np.clip(nu, 1e-8, None)

        log_pmf_h = home_goals * np.log(mu) - mu - lgamma_h
        log_pmf_a = away_goals * np.log(nu) - nu - lgamma_a

        # Vectorized tau
        t = tau_vectorized(home_goals, away_goals, mu, nu, rho)
        log_tau = np.log(np.clip(t, 1e-10, None))

        ll = weights * (log_pmf_h + log_pmf_a + log_tau)
        return -ll.sum()

    return nll


# ── MLE fitting ──────────────────────────────────────────────────────────────

def fit(df: pd.DataFrame, teams: list[str]) -> DixonColesModel:
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    df = df[df["home_team"].isin(idx) & df["away_team"].isin(idx)].copy()
    log.info("Fitting on %d matches, %d teams", len(df), n)

    ref_date = df["date"].max()
    weights  = compute_weights(df, ref_date)

    home_idx   = df["home_team"].map(idx).values.astype(int)
    away_idx   = df["away_team"].map(idx).values.astype(int)
    home_goals = df["home_score"].values.astype(int)
    away_goals = df["away_score"].values.astype(int)
    neutral_arr = df["neutral"].astype(int).values

    nll_fn = build_nll_function(home_idx, away_idx, home_goals, away_goals,
                                neutral_arr, weights, n)

    # Smart initialization: use average goals scored/conceded per team
    avg_goals_per_match = df["home_score"].mean()
    x0 = np.zeros(2 * n + 2)
    # Initialize beta (defence) so that avg attack * avg defence ≈ avg goals
    # With log-parameterisation: log(alpha)=0 means alpha=1, we want alpha*beta ≈ avg_goals
    x0[n:2*n] = np.log(avg_goals_per_match)
    x0[2*n] = np.log(1.3)     # initial home advantage
    x0[2*n + 1] = -0.05       # rho

    # Fix alpha[0] = 1 for identifiability
    bounds = (
        [(0.0, 0.0)] + [(None, None)] * (n - 1)   # log_alpha
        + [(None, None)] * n                        # log_beta
        + [(None, None)]                            # log_gamma
        + [(-0.99, 0.99)]                           # rho
    )

    log.info("Running L-BFGS-B optimisation (%d parameters) ...", 2*n+2)
    result = minimize(
        nll_fn, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 100_000, "maxfun": 500_000, "ftol": 1e-10, "gtol": 1e-7},
    )

    if not result.success:
        log.warning("Optimisation status: %s (nit=%d, fun=%.2f)", result.message, result.nit, result.fun)
    else:
        log.info("Converged in %d iterations, NLL=%.2f", result.nit, result.fun)

    alpha = dict(zip(teams, np.exp(result.x[:n]).tolist()))
    beta  = dict(zip(teams, np.exp(result.x[n:2*n]).tolist()))
    gamma = float(np.exp(result.x[2*n]))
    rho   = float(np.clip(result.x[2*n + 1], -0.99, 0.99))

    log.info("γ (home advantage) = %.4f  |  ρ (low-score correction) = %.4f", gamma, rho)
    return DixonColesModel(teams, alpha, beta, gamma, rho)


# ── Parameter shrinkage ──────────────────────────────────────────────────────

def shrink_parameters(model: DixonColesModel, df: pd.DataFrame) -> DixonColesModel:
    """
    Bayesian shrinkage applied to EVERY team.

    For each team with n competitive matches:
        attack_final  = (n * attack_mle  + K * global_mean_attack)  / (n + K)
        defense_final = (n * defense_mle + K * global_mean_defense) / (n + K)

    Teams with many matches are barely affected; sparse teams are pulled
    strongly toward the league average. This prevents extreme parameters
    for teams like Cape Verde or Haiti with few data points.
    """
    team_set = set(model.teams)
    home_counts = df[df["home_team"].isin(team_set)]["home_team"].value_counts()
    away_counts = df[df["away_team"].isin(team_set)]["away_team"].value_counts()
    match_counts = home_counts.add(away_counts, fill_value=0).astype(int)

    K = SHRINK_PRIOR
    global_alpha = float(np.mean(list(model.alpha.values())))
    global_beta  = float(np.mean(list(model.beta.values())))

    new_alpha = {}
    new_beta  = {}

    for team in model.teams:
        n = int(match_counts.get(team, 0))
        new_alpha[team] = (n * model.alpha[team] + K * global_alpha) / (n + K)
        new_beta[team]  = (n * model.beta[team]  + K * global_beta)  / (n + K)

    log.info(
        "Shrinkage (K=%d): global_α=%.3f  global_β=%.3f  |  "
        "min_n=%d  max_n=%d  median_n=%d",
        K, global_alpha, global_beta,
        match_counts.min(), match_counts.max(), int(match_counts.median()),
    )
    return DixonColesModel(model.teams, new_alpha, new_beta, model.gamma, model.rho)


# ── Sanity check ─────────────────────────────────────────────────────────────

SANITY_MATCHES = [
    ("Brazil", "Haiti"),
    ("Spain", "Cape Verde"),
    ("Argentina", "Algeria"),
    ("Netherlands", "Tunisia"),
    ("France", "Iraq"),
    ("England", "Panama"),
]


def sanity_check(model: DixonColesModel) -> None:
    log.info("\n%s  SANITY CHECK  %s", "=" * 20, "=" * 20)
    log.info("  %-15s  %-15s  %6s  %6s  %6s  %6s  %6s",
             "Home", "Away", "P(HW)", "P(D)", "P(AW)", "xG(H)", "xG(A)")
    log.info("  " + "-" * 70)
    for home, away in SANITY_MATCHES:
        if home not in model.alpha or away not in model.alpha:
            log.warning("  %-15s  %-15s  — not in model", home, away)
            continue
        ph, pd_, pa = model.predict_outcome_probs(home, away, neutral=True)
        xg_h, xg_a = model.predict_expected_goals(home, away, neutral=True)
        log.info("  %-15s  %-15s  %6.3f  %6.3f  %6.3f  %6.2f  %6.2f",
                 home, away, ph, pd_, pa, xg_h, xg_a)


# ── Validation ───────────────────────────────────────────────────────────────

def validate(model: DixonColesModel, df: pd.DataFrame, label: str) -> float:
    records = []
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        if h not in model.alpha or a not in model.alpha:
            continue
        ph, pd_, pa = model.predict_outcome_probs(h, a, neutral=bool(row["neutral"]))
        actual = row["home_score"] - row["away_score"]
        if actual > 0:
            y = 0
        elif actual == 0:
            y = 1
        else:
            y = 2
        records.append({"y": y, "ph": ph, "pd": pd_, "pa": pa})

    if not records:
        log.warning("%s: no rows to validate", label)
        return float("nan")

    res = pd.DataFrame(records)
    y_true = res["y"].values
    y_pred = res[["ph", "pd", "pa"]].values

    ll = log_loss(y_true, y_pred, labels=[0, 1, 2])
    acc = (y_pred.argmax(axis=1) == y_true).mean()
    log.info("%s: n=%d  log-loss=%.4f  accuracy=%.3f", label, len(res), ll, acc)

    for cls, name in ((0, "home_win"), (1, "draw"), (2, "away_win")):
        actual_freq = (y_true == cls).mean()
        pred_mean   = y_pred[:, cls].mean()
        log.info("  %-10s  actual=%.3f  predicted=%.3f  Δ=%+.3f",
                 name, actual_freq, pred_mean, actual_freq - pred_mean)
    return ll


# ── Main ─────────────────────────────────────────────────────────────────────

def train() -> DixonColesModel:
    df    = load_training_data()
    teams = filter_teams(df)
    log.info("%d teams with >= %d competitive matches", len(teams), MIN_MATCHES)

    # Chronological split for validation
    train_df = df[df["date"] < "2018-06-01"].copy()
    wc2018 = df[
        df["date"].between("2018-06-14", "2018-07-15")
        & df["tournament"].str.contains("FIFA World Cup", na=False)
    ].copy()
    wc2022 = df[
        df["date"].between("2022-11-20", "2022-12-18")
        & df["tournament"].str.contains("FIFA World Cup", na=False)
    ].copy()

    log.info("Train: %d  |  WC2018 val: %d  |  WC2022 val: %d",
             len(train_df), len(wc2018), len(wc2022))

    # Fit on train, validate out-of-sample
    model = fit(train_df, teams)
    model = shrink_parameters(model, train_df)
    ll_2018 = validate(model, wc2018, "WC2018 (OOS)")
    ll_2022 = validate(model, wc2022, "WC2022 (OOS)")

    # Re-fit on ALL data for deployment model
    log.info("Re-fitting on full dataset for deployment ...")
    all_teams   = filter_teams(df)
    final_model = fit(df, all_teams)
    final_model = shrink_parameters(final_model, df)

    ll_full = validate(final_model, df[df["date"] >= "2018-01-01"], "Post-2018 (in-sample)")

    sanity_check(final_model)

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        **final_model.to_dict(),
        "fit_date":         str(df["date"].max().date()),
        "num_matches":      len(df),
        "xi":               XI,
        "max_goals":        MAX_GOALS,
        "shrink_prior":     SHRINK_PRIOR,
        "xg_cap":           XG_CAP,
        "data_start":       DATA_START,
        "validation": {
            "wc2018_log_loss": round(ll_2018, 6) if not math.isnan(ll_2018) else None,
            "wc2022_log_loss": round(ll_2022, 6) if not math.isnan(ll_2022) else None,
            "post2018_log_loss": round(ll_full, 6) if not math.isnan(ll_full) else None,
        },
    }
    PARAMS_PATH.write_text(json.dumps(payload, indent=2))
    log.info("Saved → %s", PARAMS_PATH)
    log.info("Teams: %d  |  WC2018: %.4f  |  WC2022: %.4f",
             len(final_model.teams), ll_2018, ll_2022)
    return final_model


def load_model() -> DixonColesModel:
    with open(PARAMS_PATH) as f:
        d = json.load(f)
    return DixonColesModel.from_dict(d)


if __name__ == "__main__":
    train()

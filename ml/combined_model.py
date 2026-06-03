"""
Combined prediction interface for WC2026.

Fuses two trained models into a single, optimally-weighted predictor:

  1. Dixon-Coles Poisson model      (ml/models/dixon_coles_params.json)
  2. Gradient-boosted ensemble       (ml/models/ensemble.joblib)

The blend weight `w` is chosen to minimise log-loss on the held-out
World Cup matches of 2018 and 2022 (out-of-sample for the ensemble):

    combined = w * ensemble_probs + (1 - w) * dixon_coles_probs

`w` is persisted to ml/models/combined_weights.json.

Public API
----------
predict_match(home, away, neutral=True) -> dict
    {
      "home_win_prob", "draw_prob", "away_win_prob",
      "xg_home", "xg_away", "most_likely_score", "confidence"
    }

Run directly to optimise the weight and print the sanity check + the
2022 group-stage backtest:

    python3 ml/combined_model.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

# Allow `python3 ml/combined_model.py` from repo root to import sibling modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dixon_coles import DixonColesModel  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR    = Path("ml/models")
DC_PARAMS     = MODELS_DIR / "dixon_coles_params.json"
ENSEMBLE_PATH = MODELS_DIR / "ensemble.joblib"
WEIGHTS_PATH  = MODELS_DIR / "combined_weights.json"
FEATURES_PATH = Path("data/processed/match_features.csv")

LABEL_NAMES = ["home_win", "draw", "away_win"]   # index 0, 1, 2 — matches DC (ph, pd, pa)
EPS = 1e-15

# Generic (venue-independent) form feature stems.
_FORM_STEMS = (
    [f"{c}_last{w}" for w in (5, 10) for c in ("win", "draw", "gf", "ga", "gd")]
    + ["comp_win_last10", "comp_gf_last10", "comp_ga_last10"]
)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_dixon_coles() -> DixonColesModel:
    with open(DC_PARAMS) as f:
        d = json.load(f)
    return DixonColesModel.from_dict(d)


def load_ensemble() -> dict:
    return joblib.load(ENSEMBLE_PATH)


# ── Ensemble inference (batch) ────────────────────────────────────────────────

def ensemble_predict_proba(ensemble: dict, X_df: pd.DataFrame) -> np.ndarray:
    """Run the saved ensemble on a DataFrame containing its feature columns."""
    feat_cols = ensemble["feature_cols"]
    X_raw = X_df.reindex(columns=feat_cols).to_numpy(dtype=float)
    X = ensemble["imputer"].transform(X_raw)

    models = ensemble["models"]
    scaled = ensemble["uses_scaled"]
    X_list = [ensemble["scaler"].transform(X) if s else X for s in scaled]

    etype = ensemble["ensemble_type"]
    if etype == "stacking":
        meta_X = np.hstack([m.predict_proba(Xi) for m, Xi in zip(models, X_list)])
        return ensemble["meta_lr"].predict_proba(meta_X)
    if etype == "weighted_vote" and ensemble.get("weights"):
        w = ensemble["weights"]
        return np.stack(
            [m.predict_proba(Xi) * wi for m, Xi, wi in zip(models, X_list, w)]
        ).sum(axis=0)
    # soft_vote (default)
    return np.stack([m.predict_proba(Xi) for m, Xi in zip(models, X_list)]).mean(axis=0)


# ── Dixon-Coles inference (batch over rows) ───────────────────────────────────

def dc_predict_proba(dc: DixonColesModel, rows: pd.DataFrame) -> np.ndarray:
    """Per-row (ph, pd, pa) from Dixon-Coles; league-neutral 1/3 if a team is unknown."""
    out = np.empty((len(rows), 3))
    for i, (_, r) in enumerate(rows.iterrows()):
        h, a = r["home_team"], r["away_team"]
        if h in dc.alpha and a in dc.alpha:
            out[i] = dc.predict_outcome_probs(h, a, neutral=bool(r["neutral"]))
        else:
            out[i] = (1 / 3, 1 / 3, 1 / 3)
    return out


def outcome_label(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


# ── Weight optimisation ───────────────────────────────────────────────────────

def _world_cup_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df["tournament"] == "FIFA World Cup")
        & (
            df["date"].between("2018-06-14", "2018-07-15")
            | df["date"].between("2022-11-20", "2022-12-18")
        )
    )
    return df[mask].reset_index(drop=True)


def optimise_weight(
    ens_probs: np.ndarray, dc_probs: np.ndarray, y_true: np.ndarray
) -> float:
    """Find w ∈ [0, 1] minimising log-loss of (w·ens + (1−w)·dc)."""
    def neg(w: float) -> float:
        blend = w * ens_probs + (1.0 - w) * dc_probs
        blend = np.clip(blend, EPS, 1.0)
        blend = blend / blend.sum(axis=1, keepdims=True)
        return log_loss(y_true, blend, labels=[0, 1, 2])

    res = minimize_scalar(neg, bounds=(0.0, 1.0), method="bounded")
    return float(res.x)


def fit_and_save_weight() -> float:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    dc = load_dixon_coles()
    ensemble = load_ensemble()

    wc = _world_cup_rows(df)
    log.info("Calibrating blend weight on %d World Cup matches (2018 + 2022)", len(wc))

    ens_probs = ensemble_predict_proba(ensemble, wc)
    dc_probs  = dc_predict_proba(dc, wc)
    y_true    = np.array([outcome_label(r.home_score, r.away_score) for r in wc.itertuples()])

    w = optimise_weight(ens_probs, dc_probs, y_true)

    # Report the three log-losses for context.
    ll_ens = log_loss(y_true, ens_probs, labels=[0, 1, 2])
    ll_dc  = log_loss(y_true, dc_probs,  labels=[0, 1, 2])
    blend  = np.clip(w * ens_probs + (1 - w) * dc_probs, EPS, 1.0)
    blend /= blend.sum(axis=1, keepdims=True)
    ll_cmb = log_loss(y_true, blend, labels=[0, 1, 2])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps({
        "ensemble_weight":      round(w, 6),
        "dixon_coles_weight":   round(1 - w, 6),
        "calibration_set":      "FIFA World Cup 2018 + 2022",
        "n_matches":            int(len(wc)),
        "log_loss_ensemble":    round(ll_ens, 6),
        "log_loss_dixon_coles": round(ll_dc, 6),
        "log_loss_combined":    round(ll_cmb, 6),
    }, indent=2))

    log.info("Ensemble-only  log-loss = %.4f", ll_ens)
    log.info("Dixon-Coles    log-loss = %.4f", ll_dc)
    log.info("Combined (w=%.3f) log-loss = %.4f  →  %s", w, ll_cmb, WEIGHTS_PATH)
    return w


def load_weight() -> float:
    if WEIGHTS_PATH.exists():
        return float(json.loads(WEIGHTS_PATH.read_text())["ensemble_weight"])
    log.warning("No combined_weights.json found — defaulting ensemble weight to 0.5")
    return 0.5


# ── Per-team feature snapshot (for arbitrary future matchups) ──────────────────

class FeatureBuilder:
    """
    Builds a single-row feature vector for an arbitrary matchup using each
    team's most recent appearance in match_features.csv as their current
    strength snapshot. Used only for forward predictions (no leakage concern).
    """

    def __init__(self, features_csv: Path = FEATURES_PATH) -> None:
        self.df = pd.read_csv(features_csv, parse_dates=["date"]).sort_values("date")
        self._team_cache: dict[str, dict] = {}

    def _team_snapshot(self, team: str) -> dict:
        if team in self._team_cache:
            return self._team_cache[team]

        sub = self.df[(self.df["home_team"] == team) | (self.df["away_team"] == team)]
        if sub.empty:
            snap = {"elo": 1500.0, "form": {s: np.nan for s in _FORM_STEMS},
                    "squad_value": np.nan, "avg_player_value": np.nan}
        else:
            r = sub.iloc[-1]
            side = "home" if r["home_team"] == team else "away"
            snap = {
                "elo": float(r[f"elo_{side}"]),
                "form": {s: r.get(f"{side}_{s}", np.nan) for s in _FORM_STEMS},
                "squad_value":      r.get(f"{side}_squad_value", np.nan),
                "avg_player_value": r.get(f"{side}_avg_player_value", np.nan),
            }
        self._team_cache[team] = snap
        return snap

    def _h2h(self, home: str, away: str) -> dict:
        df = self.df
        meet = df[
            ((df["home_team"] == home) & (df["away_team"] == away))
            | ((df["home_team"] == away) & (df["away_team"] == home))
        ]
        n = len(meet)
        if n == 0:
            return {"h2h_count": 0, "h2h_home_win_rate": 1 / 3, "h2h_draw_rate": 1 / 3,
                    "h2h_away_win_rate": 1 / 3, "h2h_avg_goals_home": 1.5,
                    "h2h_avg_goals_away": 1.5, "h2h_gd_avg": 0.0}

        gf_home = np.where(meet["home_team"] == home, meet["home_score"], meet["away_score"]).astype(float)
        gf_away = np.where(meet["home_team"] == home, meet["away_score"], meet["home_score"]).astype(float)
        home_w = float((gf_home > gf_away).mean())
        draw   = float((gf_home == gf_away).mean())
        return {
            "h2h_count": int(n),
            "h2h_home_win_rate": home_w,
            "h2h_draw_rate": draw,
            "h2h_away_win_rate": 1.0 - home_w - draw,
            "h2h_avg_goals_home": float(gf_home.mean()),
            "h2h_avg_goals_away": float(gf_away.mean()),
            "h2h_gd_avg": float((gf_home - gf_away).mean()),
        }

    def build(self, home: str, away: str, neutral: bool = True) -> pd.DataFrame:
        h, a = self._team_snapshot(home), self._team_snapshot(away)
        adv = 0.0 if neutral else 100.0
        elo_home_adj = h["elo"] + adv

        feat: dict[str, float] = {
            # context (home_advantage is excluded from the feature set)
            "tournament_tier": 4, "is_neutral": int(neutral), "is_wc": 1,
            "is_continental": 0, "is_qualifier": 0,
            # ELO
            "elo_home": h["elo"], "elo_away": a["elo"],
            "elo_diff": h["elo"] - a["elo"],
            "elo_diff_adj": elo_home_adj - a["elo"],
            "elo_home_adj": elo_home_adj,
            "elo_expected_home": 1.0 / (1.0 + 10.0 ** ((a["elo"] - elo_home_adj) / 400.0)),
        }
        for s in _FORM_STEMS:
            feat[f"home_{s}"] = h["form"][s]
            feat[f"away_{s}"] = a["form"][s]
        feat.update(self._h2h(home, away))
        feat["home_squad_value"]      = h["squad_value"]
        feat["away_squad_value"]      = a["squad_value"]
        feat["home_avg_player_value"] = h["avg_player_value"]
        feat["away_avg_player_value"] = a["avg_player_value"]
        feat["squad_value_ratio"] = (
            h["squad_value"] / a["squad_value"]
            if pd.notna(h["squad_value"]) and pd.notna(a["squad_value"]) and a["squad_value"]
            else np.nan
        )
        # carry team/venue so DC and ensemble can both read the row
        feat["home_team"], feat["away_team"], feat["neutral"] = home, away, int(neutral)
        return pd.DataFrame([feat])


# ── Lazy global singletons (so repeated predict_match calls are cheap) ─────────

_DC: DixonColesModel | None = None
_ENS: dict | None = None
_FB: FeatureBuilder | None = None
_W: float | None = None


def _ensure_loaded() -> None:
    global _DC, _ENS, _FB, _W
    if _DC is None:
        _DC = load_dixon_coles()
    if _ENS is None:
        _ENS = load_ensemble()
    if _FB is None:
        _FB = FeatureBuilder()
    if _W is None:
        _W = load_weight()


def _most_likely_score(dc: DixonColesModel, home: str, away: str, neutral: bool) -> list[int]:
    if home not in dc.alpha or away not in dc.alpha:
        return [0, 0]
    mat = dc.predict_scoreline_probs(home, away, neutral=neutral)
    i, j = np.unravel_index(int(mat.argmax()), mat.shape)
    return [int(i), int(j)]


def _confidence(p_max: float) -> str:
    if p_max > 0.60:
        return "high"
    if p_max > 0.45:
        return "medium"
    return "low"


# ── Public prediction interface ───────────────────────────────────────────────

def predict_match(home: str, away: str, neutral: bool = True) -> dict:
    """Blended Dixon-Coles + ensemble prediction for a single matchup."""
    _ensure_loaded()
    assert _DC is not None and _ENS is not None and _FB is not None and _W is not None

    row = _FB.build(home, away, neutral=neutral)
    ens_p = ensemble_predict_proba(_ENS, row)[0]

    if home in _DC.alpha and away in _DC.alpha:
        dc_p = np.array(_DC.predict_outcome_probs(home, away, neutral=neutral))
        xg_h, xg_a = _DC.predict_expected_goals(home, away, neutral=neutral)
    else:
        log.warning("Team(s) missing from Dixon-Coles model — using ensemble only")
        dc_p = ens_p.copy()
        xg_h = xg_a = float("nan")

    blend = _W * ens_p + (1.0 - _W) * dc_p
    blend = np.clip(blend, EPS, 1.0)
    blend = blend / blend.sum()

    return {
        "home_win_prob":     round(float(blend[0]), 4),
        "draw_prob":         round(float(blend[1]), 4),
        "away_win_prob":     round(float(blend[2]), 4),
        "xg_home":           round(float(xg_h), 3) if xg_h == xg_h else None,
        "xg_away":           round(float(xg_a), 3) if xg_a == xg_a else None,
        "most_likely_score": _most_likely_score(_DC, home, away, neutral),
        "confidence":        _confidence(float(blend.max())),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

SANITY_MATCHES = [
    ("Brazil", "Haiti"), ("Spain", "Cape Verde"), ("Argentina", "Algeria"),
    ("Netherlands", "Tunisia"), ("France", "Iraq"), ("England", "Panama"),
]


def print_sanity_check() -> None:
    log.info("\n%s  COMBINED SANITY CHECK (neutral venue)  %s", "=" * 16, "=" * 16)
    hdr = (f"{'Home':<14} {'Away':<14} {'P(HW)':>6} {'P(D)':>6} {'P(AW)':>6} "
           f"{'xG(H)':>6} {'xG(A)':>6}  {'Score':>6}  Conf")
    log.info(hdr)
    log.info("-" * len(hdr))
    for home, away in SANITY_MATCHES:
        r = predict_match(home, away, neutral=True)
        score = f"{r['most_likely_score'][0]}-{r['most_likely_score'][1]}"
        log.info(
            "%-14s %-14s %6.3f %6.3f %6.3f %6.2f %6.2f  %6s  %s",
            home, away, r["home_win_prob"], r["draw_prob"], r["away_win_prob"],
            r["xg_home"] if r["xg_home"] is not None else float("nan"),
            r["xg_away"] if r["xg_away"] is not None else float("nan"),
            score, r["confidence"],
        )


def backtest_wc2022_group_stage(n: int = 16) -> None:
    """Run the first `n` WC2022 group-stage matches through the combined model."""
    _ensure_loaded()
    assert _ENS is not None and _DC is not None and _W is not None

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"]).sort_values("date")
    # Group stage ran 2022-11-20 → 2022-12-02 (knockouts began Dec 3).
    grp = df[
        (df["tournament"] == "FIFA World Cup")
        & df["date"].between("2022-11-20", "2022-12-02")
    ].sort_values("date").head(n).reset_index(drop=True)

    ens_p = ensemble_predict_proba(_ENS, grp)
    dc_p  = dc_predict_proba(_DC, grp)
    blend = _W * ens_p + (1.0 - _W) * dc_p
    blend = np.clip(blend, EPS, 1.0)
    blend = blend / blend.sum(axis=1, keepdims=True)

    log.info("\n%s  2022 WORLD CUP GROUP-STAGE BACKTEST (first %d)  %s", "=" * 12, len(grp), "=" * 12)
    hdr = (f"{'Date':<11} {'Home':<16} {'Away':<16} {'Score':>5} {'Actual':<9} "
           f"{'P(HW)':>6} {'P(D)':>6} {'P(AW)':>6} {'Pred':<9} OK")
    log.info(hdr)
    log.info("-" * len(hdr))

    correct = 0
    y_true = []
    for i, r in grp.iterrows():
        y = outcome_label(r["home_score"], r["away_score"])
        y_true.append(y)
        pred = int(blend[i].argmax())
        ok = pred == y
        correct += ok
        log.info(
            "%-11s %-16s %-16s %2d-%-2d %-9s %6.3f %6.3f %6.3f %-9s %s",
            str(r["date"].date()), r["home_team"][:15], r["away_team"][:15],
            int(r["home_score"]), int(r["away_score"]), LABEL_NAMES[y],
            blend[i, 0], blend[i, 1], blend[i, 2], LABEL_NAMES[pred],
            "✓" if ok else "✗",
        )

    log.info("-" * len(hdr))
    ll = log_loss(np.array(y_true), blend, labels=[0, 1, 2])
    log.info("Group-stage  n=%d  correct=%d (%.1f%%)  log-loss=%.4f",
             len(grp), correct, 100 * correct / len(grp), ll)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    w = fit_and_save_weight()
    # refresh singletons to use the freshly-fit weight
    global _W
    _W = w
    log.info("\nOPTIMAL ENSEMBLE WEIGHT  w = %.4f   (Dixon-Coles weight = %.4f)", w, 1 - w)
    print_sanity_check()
    backtest_wc2022_group_stage(16)


if __name__ == "__main__":
    main()

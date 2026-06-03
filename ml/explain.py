"""
SHAP explanations for the 72 WC2026 group-stage match predictions.

For each match we explain the gradient-boosted ensemble's most interpretable
member — the XGBoost model — with shap.TreeExplainer, surface the top-5 features
driving the predicted outcome class, and render a plain-language summary.

Pipeline
--------
  1. Load ml/models/ensemble.joblib (raw XGBoost pulled from its calibrated wrapper)
  2. Load data/processed/match_predictions.json (carries match_id, home, away, probs)
  3. Build each matchup's 49-feature vector via the same FeatureBuilder used at
     prediction time, impute exactly as in training, and SHAP-explain in one batch
  4. Take the top-5 |SHAP| features for the predicted class → NL explanation
  5. Save data/processed/match_explanations.json keyed by match_id

Run:  python3 ml/explain.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combined_model as cm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ENSEMBLE_PATH = Path("ml/models/ensemble.joblib")
PREDS_PATH    = Path("data/processed/match_predictions.json")
OUT_PATH      = Path("data/processed/match_explanations.json")

LABEL_MAP = {"home_win": 0, "draw": 1, "away_win": 2}
N_CLASSES = 3
TOP_K     = 5


# ── Model + raw XGBoost ───────────────────────────────────────────────────────

def load_raw_xgb(ensemble: dict):
    """The ensemble stores a calibrated XGBoost; pull out the underlying booster."""
    cal = ensemble["models"][ensemble["model_names"].index("xgboost")]
    cc = cal.calibrated_classifiers_[0]
    return getattr(cc, "estimator", None) or getattr(cc, "base_estimator")


# ── Human-readable feature phrasing ───────────────────────────────────────────

def fmt_eur(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    v = float(v)
    if v >= 1e9:
        return f"€{v / 1e9:.1f}B"
    if v >= 1e6:
        return f"€{v / 1e6:.0f}M"
    return f"€{v:,.0f}"


def wdl(win_rate, draw_rate) -> str:
    w = int(round((win_rate or 0.0) * 10))
    d = int(round((draw_rate or 0.0) * 10))
    return f"{w}W {d}D {max(0, 10 - w - d)}L"


def describe_feature(feat: str, d: dict, home: str, away: str) -> str | None:
    """Factual, value-bearing phrase for a feature (None → omit from prose)."""
    eh, ea = d.get("elo_home"), d.get("elo_away")

    if feat.startswith("elo"):
        lead, trail = (home, away) if (eh or 0) >= (ea or 0) else (away, home)
        hi, lo = (eh, ea) if (eh or 0) >= (ea or 0) else (ea, eh)
        return f"{lead}'s higher ELO rating ({lead} {hi:.0f} vs {trail} {lo:.0f}, +{abs((eh or 0)-(ea or 0)):.0f})"

    if "squad_value" in feat or "avg_player_value" in feat:
        return (f"squad market value ({home} {fmt_eur(d.get('home_squad_value'))} "
                f"vs {away} {fmt_eur(d.get('away_squad_value'))})")

    if "win_last10" in feat or "win_last5" in feat:
        if feat.startswith("home"):
            return f"{home}'s recent form ({wdl(d.get('home_win_last10'), d.get('home_draw_last10'))} in last 10)"
        return f"{away}'s recent form ({wdl(d.get('away_win_last10'), d.get('away_draw_last10'))} in last 10)"

    if "gf_last" in feat or "gd_last" in feat:
        team = home if feat.startswith("home") else away
        return f"{team}'s attacking output"
    if "ga_last" in feat:
        team = home if feat.startswith("home") else away
        return f"{team}'s defensive record"

    if feat.startswith("h2h"):
        n = int(d.get("h2h_count") or 0)
        return f"their head-to-head history ({n} prior meeting{'s' if n != 1 else ''})"

    if feat in ("tournament_tier", "is_neutral", "is_wc", "is_continental", "is_qualifier"):
        return None  # context features add little to a human summary

    return feat.replace("_", " ")


def build_explanation(home, away, pred, conf, top_descs) -> str:
    conf_pct = round(conf * 100)
    # Keep the highest-ranked distinct reasons (several ELO/squad variants share
    # a phrase); dedupe so the sentence cites three genuinely different drivers.
    seen, reasons = set(), []
    for x in top_descs:
        if x and x not in seen:
            seen.add(x)
            reasons.append(x)
        if len(reasons) == 3:
            break
    if len(reasons) > 1:
        joined = ", ".join(reasons[:-1]) + f", and {reasons[-1]}"
    elif reasons:
        joined = reasons[0]
    else:
        joined = "the model's combined signals"
    if pred == "draw":
        return (f"{home} and {away} are predicted to draw ({conf_pct}% confidence), "
                f"with the model citing {joined}.")
    winner, loser = (home, away) if pred == "home_win" else (away, home)
    return (f"{winner} are predicted to beat {loser} ({conf_pct}% confidence) "
            f"primarily because of {joined}.")


# ── SHAP extraction ───────────────────────────────────────────────────────────

def class_shap(sv, cls: int) -> np.ndarray:
    """Return (n_samples, n_features) SHAP matrix for one class, across shap versions."""
    if isinstance(sv, list):                      # list of per-class arrays
        return np.asarray(sv[cls])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        if sv.shape[-1] == N_CLASSES:             # (samples, features, classes)
            return sv[:, :, cls]
        if sv.shape[0] == N_CLASSES:              # (classes, samples, features)
            return sv[cls]
    return sv                                     # already 2-D


def cell_value(d: dict, feat: str):
    v = d.get(feat)
    if isinstance(v, (int, float, np.floating)):
        return None if (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ensemble = joblib.load(ENSEMBLE_PATH)
    feat_cols = ensemble["feature_cols"]
    imputer = ensemble["imputer"]
    xgb_raw = load_raw_xgb(ensemble)

    preds = json.loads(PREDS_PATH.read_text())
    log.info("Explaining %d group-stage matches ...", len(preds))

    # Build every feature vector once (same builder as prediction time).
    fb = cm.FeatureBuilder()
    rows, raw_dicts = [], []
    for p in preds:
        row = fb.build(p["home"], p["away"], neutral=True)
        rows.append(row.reindex(columns=feat_cols))
        raw_dicts.append(row.iloc[0].to_dict())

    X = imputer.transform(pd.concat(rows, ignore_index=True)[feat_cols].to_numpy(dtype=float))

    explainer = shap.TreeExplainer(xgb_raw)
    sv = explainer.shap_values(X)

    out = {}
    for i, p in enumerate(preds):
        pred = p["most_likely_result"]
        cls = LABEL_MAP[pred]
        conf = max(p["p_home_win"], p["p_draw"], p["p_away_win"])
        contribs = class_shap(sv, cls)[i]                      # (n_features,)

        order = np.argsort(np.abs(contribs))[::-1][:TOP_K]
        top_features, descs = [], []
        for j in order:
            feat = feat_cols[j]
            desc = describe_feature(feat, raw_dicts[i], p["home"], p["away"])
            top_features.append({
                "feature": feat,
                "shap_value": round(float(contribs[j]), 4),
                "value": cell_value(raw_dicts[i], feat),
                "description": desc,
            })
            descs.append(desc)

        out[p["match_id"]] = {
            "match_id": p["match_id"],
            "group": p["group"], "home": p["home"], "away": p["away"],
            "predicted": pred,
            "confidence": round(float(conf), 4),
            "explanation": build_explanation(p["home"], p["away"], pred, conf, descs),
            "top_features": top_features,
        }

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info("Saved %d explanations → %s", len(out), OUT_PATH)
    for mid in list(out)[:4]:
        log.info("  • %s", out[mid]["explanation"])


if __name__ == "__main__":
    main()

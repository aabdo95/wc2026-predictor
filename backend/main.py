"""FastAPI backend for the WC2026 match-prediction dashboard.

All simulation artefacts (``data/processed/*.json``), the trained models
(``ml/models/``) and the committed fixtures (``data/fixtures/``) are loaded
once at startup into an in-memory :class:`DataStore`. The endpoints only read
from that store — there is no per-request recomputation or model inference.

Run with::

    uvicorn backend.main:app --reload --port 8000

(or ``python3 backend/main.py`` / ``make backend``).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from backend import schemas

# --------------------------------------------------------------------------- #
# Paths & logging
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FIXTURES = ROOT / "data" / "fixtures"
MODELS = ROOT / "ml" / "models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("wc2026.api")

# Rounds in tournament order, used for sorting knockout distributions.
KO_ROUNDS = ["R32", "R16", "QF", "SF", "Final", "Winner"]


# --------------------------------------------------------------------------- #
# In-memory data store (populated once at startup)
# --------------------------------------------------------------------------- #
class DataStore:
    """Holds every artefact the API serves, plus a few derived indexes."""

    def __init__(self) -> None:
        # Raw artefacts -----------------------------------------------------
        self.group_standings: Dict[str, list] = _read_json(PROCESSED / "group_standings.json")
        self.match_predictions: List[dict] = _read_json(PROCESSED / "match_predictions.json")
        self.winner_odds: dict = _read_json(PROCESSED / "tournament_winner_odds.json")
        self.knockout: Dict[str, dict] = _read_json(PROCESSED / "knockout_probabilities.json")
        self.bracket: dict = _read_json(PROCESSED / "bracket.json")
        self.explanations: Dict[str, dict] = _read_json(PROCESSED / "match_explanations.json")
        self.groups_fixture: dict = _read_json(FIXTURES / "groups.json")
        self.dixon_coles: dict = _read_json(MODELS / "dixon_coles_params.json")
        self.combined_weights: dict = _read_json(MODELS / "combined_weights.json")

        # Derived indexes ---------------------------------------------------
        self.groups: Dict[str, List[str]] = self.groups_fixture["groups"]

        # match_id -> prediction, and 1-based ordinal -> match_id
        self.match_by_id: Dict[str, dict] = {
            m["match_id"]: m for m in self.match_predictions
        }
        self.match_order: List[str] = [m["match_id"] for m in self.match_predictions]

        # group -> [predictions] (preserves schedule order)
        self.matches_by_group: Dict[str, List[dict]] = {}
        for m in self.match_predictions:
            self.matches_by_group.setdefault(m["group"], []).append(m)

        # lower-cased team -> (group, standings row)
        self.team_index: Dict[str, dict] = {}
        for grp, rows in self.group_standings.items():
            for row in rows:
                self.team_index[row["team"].lower()] = {"group": grp, "row": row}

        # Ensemble model metadata + averaged tree feature importance --------
        self.ensemble_meta, self.feature_importance = _load_ensemble_meta()

    # -- lookups -----------------------------------------------------------
    def resolve_match(self, match_id: str) -> Optional[dict]:
        """Resolve a match by slug match_id or by 1-based ordinal (e.g. ``1``)."""
        if match_id in self.match_by_id:
            return self.match_by_id[match_id]
        if match_id.isdigit():
            idx = int(match_id) - 1
            if 0 <= idx < len(self.match_order):
                return self.match_by_id[self.match_order[idx]]
        return None


def _read_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_ensemble_meta():
    """Load ensemble metadata and compute averaged tree feature importance.

    Importance is averaged across the tree models (XGBoost, LightGBM,
    CatBoost) that expose ``feature_importances_``; each model's vector is
    L1-normalised first so no single scale dominates. LogReg is excluded as
    it has coefficients rather than impurity-style importances.
    """
    bundle = joblib.load(MODELS / "ensemble.joblib")
    feature_cols: List[str] = bundle["feature_cols"]

    acc = np.zeros(len(feature_cols), dtype=float)
    n_used = 0
    for cal in bundle["models"]:
        est = cal.calibrated_classifiers_[0].estimator
        imp = getattr(est, "feature_importances_", None)
        if imp is None:
            continue
        imp = np.asarray(imp, dtype=float)
        total = imp.sum()
        if total > 0:
            acc += imp / total
            n_used += 1
    if n_used:
        acc /= n_used

    importance = sorted(
        ({"feature": f, "importance": round(float(v), 6)}
         for f, v in zip(feature_cols, acc)),
        key=lambda d: d["importance"],
        reverse=True,
    )

    meta = {
        "ensemble_type": bundle["ensemble_type"],
        "model_names": bundle["model_names"],
        "feature_cols": feature_cols,
        "test_metrics": bundle["test_metrics"],
        "version": bundle.get("version"),
    }
    return meta, importance


# Instantiated in the startup handler so import never blocks on disk I/O.
store: Optional[DataStore] = None


def get_store() -> DataStore:
    if store is None:  # pragma: no cover - guarded by startup event
        raise HTTPException(status_code=503, detail="Data store not initialised")
    return store


# --------------------------------------------------------------------------- #
# App + middleware
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="WC2026 Predictor API",
    version="1.0.0",
    description="Serves Monte Carlo simulation results and model metadata "
    "for the FIFA World Cup 2026 prediction dashboard.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.on_event("startup")
def _startup() -> None:
    global store
    t0 = time.perf_counter()
    store = DataStore()
    logger.info(
        "Loaded artefacts in %.0f ms: %d matches, %d groups, %d teams, %d explanations",
        (time.perf_counter() - t0) * 1000,
        len(store.match_predictions),
        len(store.groups),
        len(store.team_index),
        len(store.explanations),
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/overview", response_model=schemas.OverviewResponse, tags=["overview"])
def overview():
    """Tournament headline: top-10 champion odds + model/run metadata."""
    s = get_store()
    w = s.winner_odds
    tm = s.ensemble_meta["test_metrics"]
    return schemas.OverviewResponse(
        total_matches=len(s.match_predictions),
        n_simulations=w["n_simulations"],
        blend_weight=w["blend_weight"],
        blend_description=w["blend_description"],
        market_source=w["market_source"],
        top_10=w["top_10"],
        model=schemas.ModelSummary(
            ensemble_type=s.ensemble_meta["ensemble_type"],
            log_loss=tm["log_loss"],
            accuracy=tm["accuracy"],
            n_features=len(s.ensemble_meta["feature_cols"]),
            blend_weight=w["blend_weight"],
        ),
    )


@app.get("/api/groups", response_model=List[schemas.GroupSummary], tags=["groups"])
def list_groups():
    """All 12 groups, teams sorted by expected points (descending)."""
    s = get_store()
    out = []
    for grp in sorted(s.group_standings):
        teams = sorted(
            s.group_standings[grp],
            key=lambda r: r["expected_points"],
            reverse=True,
        )
        out.append(schemas.GroupSummary(group=grp, teams=teams))
    return out


@app.get("/api/groups/{group_id}", response_model=schemas.GroupDetail, tags=["groups"])
def group_detail(group_id: str):
    """A single group's standings plus its 6 group-stage match predictions."""
    s = get_store()
    grp = group_id.upper()
    if grp not in s.group_standings:
        raise HTTPException(status_code=404, detail=f"Group '{group_id}' not found")
    teams = sorted(
        s.group_standings[grp], key=lambda r: r["expected_points"], reverse=True
    )
    matches = s.matches_by_group.get(grp, [])
    return schemas.GroupDetail(group=grp, teams=teams, matches=matches)


@app.get("/api/matches", response_model=List[schemas.MatchPrediction], tags=["matches"])
def list_matches(
    group: Optional[str] = Query(None, description="Filter by group letter, e.g. C"),
    team: Optional[str] = Query(None, description="Filter by team (home or away, case-insensitive)"),
    sort: str = Query("group", pattern="^(group|confidence|date)$"),
):
    """All 72 group-stage predictions, filterable and sortable."""
    s = get_store()
    matches = list(s.match_predictions)

    if group:
        grp = group.upper()
        matches = [m for m in matches if m["group"] == grp]
    if team:
        t = team.lower()
        matches = [m for m in matches if t in (m["home"].lower(), m["away"].lower())]

    if sort == "confidence":
        rank = {"high": 0, "medium": 1, "low": 2}
        # within a confidence tier, order by the strongest single-outcome prob
        matches.sort(
            key=lambda m: (
                rank.get(m["confidence"], 3),
                -max(m["p_home_win"], m["p_draw"], m["p_away_win"]),
            )
        )
    elif sort == "group":
        matches.sort(key=lambda m: (m["group"], s.match_order.index(m["match_id"])))
    # sort == "date": schedule.json has no per-match dates, so we preserve the
    # original schedule (insertion) order, which is already date-ordered.

    return matches


@app.get("/api/matches/{match_id}", response_model=schemas.MatchDetail, tags=["matches"])
def match_detail(match_id: str):
    """Full prediction for one match, including SHAP-derived top features.

    ``match_id`` accepts either the slug (``C-brazil-vs-morocco``) or the
    1-based ordinal in schedule order (``1`` .. ``72``).
    """
    s = get_store()
    m = s.resolve_match(match_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"Match '{match_id}' not found")
    exp = s.explanations.get(m["match_id"], {})
    return schemas.MatchDetail(
        match_id=m["match_id"],
        group=m["group"],
        home=m["home"],
        away=m["away"],
        p_home_win=m["p_home_win"],
        p_draw=m["p_draw"],
        p_away_win=m["p_away_win"],
        most_likely_result=m["most_likely_result"],
        most_likely_score=m["expected_scoreline"],
        xg_home=m["xg_home"],
        xg_away=m["xg_away"],
        confidence=m["confidence"],
        predicted=exp.get("predicted"),
        explanation=exp.get("explanation"),
        top_features=exp.get("top_features", []),
    )


@app.get("/api/bracket", response_model=schemas.BracketResponse, tags=["bracket"])
def bracket():
    """Full knockout bracket: per-round reach probabilities for every team."""
    s = get_store()
    return schemas.BracketResponse(method=s.bracket["method"], rounds=s.bracket["rounds"])


@app.get("/api/teams/{team_name}", response_model=schemas.TeamProfile, tags=["teams"])
def team_profile(team_name: str):
    """One team's group standing plus its full knockout-round distribution."""
    s = get_store()
    entry = s.team_index.get(team_name.lower())
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")
    row = entry["row"]
    ko = s.knockout.get(row["team"], {})
    knockout = {r: ko[r] for r in KO_ROUNDS if r in ko}
    return schemas.TeamProfile(
        team=row["team"],
        group=entry["group"],
        expected_points=row["expected_points"],
        expected_gd=row["expected_gd"],
        expected_gf=row["expected_gf"],
        win_group_prob=row["win_group_prob"],
        advance_prob=row["advance_prob"],
        avg_finish=row["avg_finish"],
        knockout=knockout,
    )


@app.get("/api/features", response_model=schemas.FeaturesResponse, tags=["model"])
def features(top: int = Query(15, ge=1, le=49)):
    """Top feature importances, averaged across the ensemble's tree models."""
    s = get_store()
    return schemas.FeaturesResponse(
        source="Mean L1-normalised importance across XGBoost, LightGBM, CatBoost",
        n_features=len(s.ensemble_meta["feature_cols"]),
        top_features=s.feature_importance[:top],
    )


@app.get("/api/model-info", response_model=schemas.ModelInfoResponse, tags=["model"])
def model_info():
    """Architecture, metrics and training metadata for both model families."""
    s = get_store()
    tm = s.ensemble_meta["test_metrics"]
    dc = s.dixon_coles
    val = dc["validation"]
    cw = s.combined_weights
    return schemas.ModelInfoResponse(
        ensemble_type=s.ensemble_meta["ensemble_type"],
        models=s.ensemble_meta["model_names"],
        log_loss=tm["log_loss"],
        accuracy=tm["accuracy"],
        brier=tm["brier"],
        training_date=dc["fit_date"],
        n_features=len(s.ensemble_meta["feature_cols"]),
        feature_names=s.ensemble_meta["feature_cols"],
        blend_weight=s.winner_odds["blend_weight"],
        dixon_coles=schemas.DixonColesInfo(
            fit_date=dc["fit_date"],
            num_matches=dc["num_matches"],
            n_teams=len(dc["teams"]),
            gamma=dc["gamma"],
            rho=dc["rho"],
            elo_c=dc["elo_c"],
            shrink_prior=dc["shrink_prior"],
            wc2018_log_loss=val["wc2018_log_loss"],
            wc2022_log_loss=val["wc2022_log_loss"],
            post2018_log_loss=val["post2018_log_loss"],
        ),
        combined=schemas.CombinedWeights(
            ensemble_weight=cw["ensemble_weight"],
            dixon_coles_weight=cw["dixon_coles_weight"],
            calibration_set=cw["calibration_set"],
            n_matches=cw["n_matches"],
            log_loss_combined=cw["log_loss_combined"],
        ),
    )


@app.get("/api/health", tags=["meta"])
def health():
    return {"status": "ok", "loaded": store is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)

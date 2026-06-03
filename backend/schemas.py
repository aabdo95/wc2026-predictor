"""Pydantic response schemas for the WC2026 prediction API.

Every endpoint in ``backend/main.py`` returns one of these models so the
response shape is documented in the OpenAPI schema and validated on the way
out. Field names mirror the underlying JSON in ``data/processed/`` and
``ml/models/`` so there is a 1:1 mapping with the simulation artefacts.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Tournament winner odds / overview
# --------------------------------------------------------------------------- #
class WinnerOdds(BaseModel):
    team: str
    win_prob: float = Field(..., description="Blended champion probability (0-1)")
    decimal_odds: float
    data_model_prob: float = Field(..., description="Pure-simulation probability")
    market_prob: float = Field(..., description="Bookmaker market prior")


class MarketSource(BaseModel):
    source: str
    source_url: Optional[str] = None
    collected_date: str


class ModelSummary(BaseModel):
    ensemble_type: str
    log_loss: float
    accuracy: float
    n_features: int
    blend_weight: float


class OverviewResponse(BaseModel):
    total_matches: int
    n_simulations: int
    blend_weight: float
    blend_description: str
    market_source: MarketSource
    top_10: List[WinnerOdds]
    model: ModelSummary


# --------------------------------------------------------------------------- #
# Groups
# --------------------------------------------------------------------------- #
class GroupTeam(BaseModel):
    team: str
    expected_points: float
    expected_gd: float
    expected_gf: float
    win_group_prob: float
    advance_prob: float
    avg_finish: float


class GroupSummary(BaseModel):
    group: str
    teams: List[GroupTeam]


# --------------------------------------------------------------------------- #
# Matches
# --------------------------------------------------------------------------- #
class MatchPrediction(BaseModel):
    match_id: str
    group: str
    home: str
    away: str
    p_home_win: float
    p_draw: float
    p_away_win: float
    most_likely_result: str
    expected_scoreline: str
    xg_home: float
    xg_away: float
    confidence: str


class GroupDetail(GroupSummary):
    matches: List[MatchPrediction]


class TopFeature(BaseModel):
    feature: str
    shap_value: float
    value: float
    description: str


class MatchDetail(BaseModel):
    match_id: str
    group: str
    home: str
    away: str
    p_home_win: float
    p_draw: float
    p_away_win: float
    most_likely_result: str
    most_likely_score: str
    xg_home: float
    xg_away: float
    confidence: str
    predicted: Optional[str] = None
    explanation: Optional[str] = None
    top_features: List[TopFeature] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Bracket / knockout
# --------------------------------------------------------------------------- #
class BracketEntry(BaseModel):
    team: str
    prob: float


class BracketResponse(BaseModel):
    method: str
    rounds: Dict[str, List[BracketEntry]]


# --------------------------------------------------------------------------- #
# Team profile
# --------------------------------------------------------------------------- #
class TeamProfile(BaseModel):
    team: str
    group: str
    expected_points: float
    expected_gd: float
    expected_gf: float
    win_group_prob: float
    advance_prob: float
    avg_finish: float
    knockout: Dict[str, float] = Field(
        ..., description="P(reach round) for R32, R16, QF, SF, Final, Winner"
    )


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #
class FeatureImportance(BaseModel):
    feature: str
    importance: float


class FeaturesResponse(BaseModel):
    source: str
    n_features: int
    top_features: List[FeatureImportance]


# --------------------------------------------------------------------------- #
# Model info
# --------------------------------------------------------------------------- #
class DixonColesInfo(BaseModel):
    fit_date: str
    num_matches: int
    n_teams: int
    gamma: float
    rho: float
    elo_c: float
    shrink_prior: int
    wc2018_log_loss: float
    wc2022_log_loss: float
    post2018_log_loss: float


class CombinedWeights(BaseModel):
    ensemble_weight: float
    dixon_coles_weight: float
    calibration_set: str
    n_matches: int
    log_loss_combined: float


class ModelInfoResponse(BaseModel):
    ensemble_type: str
    models: List[str]
    log_loss: float
    accuracy: float
    brier: float
    training_date: str
    n_features: int
    feature_names: List[str]
    blend_weight: float
    dixon_coles: DixonColesInfo
    combined: CombinedWeights

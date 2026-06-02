# WC 2026 Predictor

End-to-end World Cup 2026 prediction system: data collection → feature engineering → ML model → Monte Carlo simulation → FastAPI + React dashboard.

## Quickstart

```bash
# Install dependencies
make setup

# Download raw data
make collect

# Build features & train model
make features
make train

# Run Monte Carlo simulation
make simulate

# Start backend API
make backend

# Start frontend (separate terminal)
make frontend
```

## Project Layout

| Path | Purpose |
|---|---|
| `ml/collect/` | Data scrapers (match results, ELO, FIFA rankings, Transfermarkt) |
| `ml/features.py` | Feature engineering pipeline |
| `ml/train.py` | Model training & cross-validation |
| `ml/simulate.py` | Tournament Monte Carlo (N=10 000) |
| `ml/explain.py` | SHAP explanations per match |
| `backend/` | FastAPI REST API |
| `frontend/` | React dashboard |
| `data/fixtures/` | WC2026 groups & schedule (committed) |

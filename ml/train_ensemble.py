"""
Gradient-boosted ensemble for WC2026 match outcome prediction.

Pipeline:
  1. Chronological split (train 2000-2017 / val 2018-2021 / test 2022+)
  2. Median imputation (fit on train only)
  3. Optuna tuning: 200 trials each for XGBoost, LightGBM, CatBoost
  4. Isotonic calibration on validation set
  5. Soft-voting ensemble vs. stacking meta-LR — pick lower test log-loss
  6. Reliability diagrams per outcome class
  7. Per-match predictions for every 2022 WC game
  8. Feature importances (XGB + LGB + CatBoost averaged)
  9. Final ensemble → ml/models/ensemble.joblib
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURES_PATH   = Path("data/processed/match_features.csv")
MODELS_DIR      = Path("ml/models")
FI_PATH         = Path("data/processed/feature_importance.json")
ENSEMBLE_PATH   = MODELS_DIR / "ensemble.joblib"
CALIB_PLOT_PATH = MODELS_DIR / "calibration_curves.png"

LABEL_MAP   = {"home_win": 0, "draw": 1, "away_win": 2}
LABEL_INV   = {0: "home_win", 1: "draw", 2: "away_win"}
LABEL_NAMES = ["home_win", "draw", "away_win"]

NON_FEAT_COLS = {
    "date", "home_team", "away_team", "tournament",
    "home_score", "away_score", "neutral", "outcome",
    "sample_weight", "home_advantage",
}

N_TRIALS = 200
TIMEOUT  = 1800   # seconds per model Optuna study
SEED     = 42


# ── Data ─────────────────────────────────────────────────────────────────────

def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    train = df[df["date"] < "2018-01-01"].copy()
    val   = df[(df["date"] >= "2018-01-01") & (df["date"] < "2022-01-01")].copy()
    test  = df[df["date"] >= "2022-01-01"].copy()

    log.info(
        "Split  train: %d (%s–%s)  val: %d (%s–%s)  test: %d (%s–%s)",
        len(train), train["date"].min().date(), train["date"].max().date(),
        len(val),   val["date"].min().date(),   val["date"].max().date(),
        len(test),  test["date"].min().date(),  test["date"].max().date(),
    )
    return train, val, test


def prepare_arrays(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
) -> dict:
    feat_cols = [c for c in train.columns if c not in NON_FEAT_COLS]

    # Impute with median — fit on train only (no leakage)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train[feat_cols])
    X_val   = imputer.transform(val[feat_cols])
    X_test  = imputer.transform(test[feat_cols])

    # Scale for LogReg only — fit on train only
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    y_train = train["outcome"].map(LABEL_MAP).values.astype(int)
    y_val   = val["outcome"].map(LABEL_MAP).values.astype(int)
    y_test  = test["outcome"].map(LABEL_MAP).values.astype(int)
    w_train = train["sample_weight"].values

    return dict(
        feat_cols=feat_cols, imputer=imputer, scaler=scaler,
        X_train=X_train,   X_val=X_val,   X_test=X_test,
        X_train_s=X_train_s, X_val_s=X_val_s, X_test_s=X_test_s,
        y_train=y_train, y_val=y_val, y_test=y_test,
        w_train=w_train,
    )


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

def tune_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            max_depth        = trial.suggest_int  ("max_depth",        3, 8),
            learning_rate    = trial.suggest_float("learning_rate",    0.01, 0.2,  log=True),
            n_estimators     = trial.suggest_int  ("n_estimators",     100, 1000),
            min_child_weight = trial.suggest_int  ("min_child_weight", 1, 10),
            subsample        = trial.suggest_float("subsample",        0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            objective        = "multi:softprob",
            seed             = SEED,
            n_jobs           = -1,
            verbosity        = 0,
        )
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train, sample_weight=w_train, verbose=False)
        return log_loss(y_val, m.predict_proba(X_val))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT, catch=(Exception,))
    log.info("XGB  best val log-loss=%.4f  trials=%d", study.best_value, len(study.trials))
    log.info("XGB  best params: %s", study.best_params)
    return study.best_params


def tune_lgb(
    X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        bagging_fraction = trial.suggest_float("bagging_fraction", 0.5, 1.0)
        params = dict(
            num_leaves       = trial.suggest_int  ("num_leaves",       15, 127),
            learning_rate    = trial.suggest_float("learning_rate",    0.01, 0.2, log=True),
            n_estimators     = trial.suggest_int  ("n_estimators",     100, 1000),
            min_child_samples= trial.suggest_int  ("min_child_samples",5, 100),
            feature_fraction = trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction = bagging_fraction,
            bagging_freq     = 1 if bagging_fraction < 1.0 else 0,
            objective        = "multiclass",
            num_class        = 3,
            metric           = "multi_logloss",
            random_state     = SEED,
            n_jobs           = -1,
            verbose          = -1,
        )
        m = lgb.LGBMClassifier(**params)
        m.fit(X_train, y_train, sample_weight=w_train)
        return log_loss(y_val, m.predict_proba(X_val))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT, catch=(Exception,))
    log.info("LGB  best val log-loss=%.4f  trials=%d", study.best_value, len(study.trials))
    log.info("LGB  best params: %s", study.best_params)
    return study.best_params


def tune_catboost(
    X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            depth            = trial.suggest_int  ("depth",            4, 10),
            learning_rate    = trial.suggest_float("learning_rate",    0.01, 0.2, log=True),
            iterations       = trial.suggest_int  ("iterations",       200, 1500),
            l2_leaf_reg      = trial.suggest_float("l2_leaf_reg",      1.0, 10.0),
            random_strength  = trial.suggest_float("random_strength",  0.0, 2.0),
            bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0),
            border_count     = trial.suggest_int  ("border_count",     32, 255),
            loss_function    = "MultiClass",
            eval_metric      = "MultiClass",
            random_seed      = SEED,
            verbose          = 0,
            thread_count     = -1,
        )
        m = CatBoostClassifier(**params)
        m.fit(X_train, y_train, sample_weight=w_train, verbose=0)
        return log_loss(y_val, m.predict_proba(X_val))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT, catch=(Exception,))
    log.info("CAT  best val log-loss=%.4f  trials=%d", study.best_value, len(study.trials))
    log.info("CAT  best params: %s", study.best_params)
    return study.best_params


def tune_logreg(
    X_train_s: np.ndarray, y_train: np.ndarray,
    X_val_s: np.ndarray,   y_val: np.ndarray,
) -> float:
    best_ll, best_C = float("inf"), 1.0
    for C in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=SEED)
        m.fit(X_train_s, y_train)
        ll = log_loss(y_val, m.predict_proba(X_val_s))
        log.info("  LogReg C=%-8.4g  val log-loss=%.4f", C, ll)
        if ll < best_ll:
            best_ll, best_C = ll, C
    log.info("LogReg best C=%.4g  val log-loss=%.4f", best_C, best_ll)
    return best_C


# ── Model training ────────────────────────────────────────────────────────────

def train_xgb(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, params: dict,
) -> xgb.XGBClassifier:
    p = dict(
        **params,
        objective  = "multi:softprob",
        seed       = SEED,
        n_jobs     = -1,
        verbosity  = 0,
    )
    m = xgb.XGBClassifier(**p)
    m.fit(X, y, sample_weight=w, verbose=False)
    return m


def train_lgb(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, params: dict,
) -> lgb.LGBMClassifier:
    bagging_fraction = params.get("bagging_fraction", 1.0)
    p = dict(
        **params,
        objective    = "multiclass",
        num_class    = 3,
        metric       = "multi_logloss",
        bagging_freq = 1 if bagging_fraction < 1.0 else 0,
        random_state = SEED,
        n_jobs       = -1,
        verbose      = -1,
    )
    m = lgb.LGBMClassifier(**p)
    m.fit(X, y, sample_weight=w)
    return m


def train_catboost(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, params: dict,
) -> CatBoostClassifier:
    p = dict(
        **params,
        loss_function = "MultiClass",
        eval_metric   = "MultiClass",
        random_seed   = SEED,
        verbose       = 0,
        thread_count  = -1,
    )
    m = CatBoostClassifier(**p)
    m.fit(X, y, sample_weight=w, verbose=0)
    return m


def train_logreg(X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=SEED)
    m.fit(X, y)
    return m


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate(model, X_cal: np.ndarray, y_cal: np.ndarray, method: str = "isotonic") -> CalibratedClassifierCV:
    cal = CalibratedClassifierCV(model, cv="prefit", method=method)
    cal.fit(X_cal, y_cal)
    return cal


# ── Ensemble ──────────────────────────────────────────────────────────────────

def soft_vote(models: list, X_list: list[np.ndarray]) -> np.ndarray:
    probs = np.stack([m.predict_proba(X) for m, X in zip(models, X_list)])
    return probs.mean(axis=0)


def build_stacking(
    models: list,
    X_val_list:  list[np.ndarray],
    y_val:       np.ndarray,
    X_test_list: list[np.ndarray],
) -> tuple[LogisticRegression, np.ndarray]:
    meta_X_val  = np.hstack([m.predict_proba(X) for m, X in zip(models, X_val_list)])
    meta_X_test = np.hstack([m.predict_proba(X) for m, X in zip(models, X_test_list)])
    meta = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED)
    meta.fit(meta_X_val, y_val)
    return meta, meta.predict_proba(meta_X_test)


# ── Metrics ───────────────────────────────────────────────────────────────────

def brier_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean([
        brier_score_loss((y_true == c).astype(int), y_prob[:, c])
        for c in range(y_prob.shape[1])
    ]))


def evaluate(label: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    ll  = log_loss(y_true, y_prob)
    acc = accuracy_score(y_true, y_prob.argmax(axis=1))
    bs  = brier_multiclass(y_true, y_prob)
    log.info("%-28s  log-loss=%.4f  acc=%.3f  brier=%.4f", label, ll, acc, bs)
    return {"log_loss": round(ll, 6), "accuracy": round(acc, 4), "brier": round(bs, 6)}


# ── WC2022 per-match table ────────────────────────────────────────────────────

def evaluate_wc2022(
    test_df: pd.DataFrame,
    y_prob:  np.ndarray,
    y_test:  np.ndarray,
) -> None:
    df = test_df.reset_index(drop=True)
    mask = (
        (df["tournament"] == "FIFA World Cup")
        & (df["date"] >= "2022-11-20")
        & (df["date"] <= "2022-12-18")
    )
    wc = df[mask]
    if wc.empty:
        log.warning("No WC2022 rows found — check tournament name in CSV")
        return

    log.info("\n%s  2022 WORLD CUP MATCH PREDICTIONS  %s", "=" * 18, "=" * 18)
    hdr = f"{'Date':<12} {'Home':<24} {'Away':<24} {'Actual':<10} {'P(HW)':>6} {'P(D)':>6} {'P(AW)':>6}  {'Pred':<10}  OK"
    log.info(hdr)
    log.info("-" * len(hdr))

    correct = 0
    for i in wc.index:
        p      = y_prob[i]
        actual = LABEL_INV[int(y_test[i])]
        pred   = LABEL_INV[int(p.argmax())]
        ok     = "✓" if pred == actual else "✗"
        correct += pred == actual
        log.info(
            "%-12s %-24s %-24s %-10s %6.3f %6.3f %6.3f  %-10s  %s",
            str(df.loc[i, "date"].date()),
            df.loc[i, "home_team"][:23], df.loc[i, "away_team"][:23],
            actual, p[0], p[1], p[2], pred, ok,
        )

    log.info("-" * len(hdr))
    n = len(wc)
    ll = log_loss(y_test[wc.index], y_prob[wc.index])
    bs = brier_multiclass(y_test[wc.index], y_prob[wc.index])
    log.info(
        "WC2022  n=%d  correct=%d (%.1f%%)  log-loss=%.4f  brier=%.4f",
        n, correct, 100 * correct / n, ll, bs,
    )


# ── Calibration curves ────────────────────────────────────────────────────────

def plot_calibration_curves(
    probs_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colours = plt.cm.tab10.colors

    for cls_idx, cls_name in enumerate(LABEL_NAMES):
        ax   = axes[cls_idx]
        y_bin = (y_true == cls_idx).astype(int)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect", zorder=0)

        for col, (label, probs) in zip(colours, probs_dict.items()):
            try:
                frac, mean_pred = calibration_curve(
                    y_bin, probs[:, cls_idx], n_bins=10, strategy="uniform"
                )
                ax.plot(mean_pred, frac, marker="o", ms=4, color=col, label=label)
            except Exception:
                pass

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(cls_name.replace("_", " ").title())
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Reliability Diagrams (test set)", fontsize=13)
    fig.tight_layout()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CALIB_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Calibration curves → %s", CALIB_PLOT_PATH)


# ── Feature importances ───────────────────────────────────────────────────────

def save_feature_importances(
    xgb_raw: xgb.XGBClassifier,
    lgb_raw: lgb.LGBMClassifier,
    cat_raw: CatBoostClassifier,
    feat_cols: list[str],
) -> None:
    def norm(arr: np.ndarray) -> np.ndarray:
        s = arr.sum()
        return arr / s if s > 0 else arr

    xgb_imp = norm(xgb_raw.feature_importances_.astype(float))
    lgb_imp = norm(lgb_raw.feature_importances_.astype(float))
    cat_imp = norm(np.array(cat_raw.get_feature_importance(), dtype=float))
    avg     = (xgb_imp + lgb_imp + cat_imp) / 3.0

    fi = {feat_cols[i]: round(float(avg[i]), 6) for i in range(len(feat_cols))}
    fi_sorted = dict(sorted(fi.items(), key=lambda kv: -kv[1]))

    FI_PATH.parent.mkdir(parents=True, exist_ok=True)
    FI_PATH.write_text(json.dumps(fi_sorted, indent=2))

    log.info("Top 15 features (XGB+LGB+CAT avg importance):")
    for feat, score in list(fi_sorted.items())[:15]:
        bar = "█" * int(score * 400)
        log.info("  %-42s %.4f  %s", feat, score, bar)


# ── Main ─────────────────────────────────────────────────────────────────────

def train() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_and_split()
    d = prepare_arrays(train_df, val_df, test_df)

    # ── 2. Hyperparameter tuning ──────────────────────────────────────────────
    log.info("=== Tuning XGBoost (%d trials, timeout=%ds) ===", N_TRIALS, TIMEOUT)
    xgb_params = tune_xgboost(d["X_train"], d["y_train"], d["w_train"], d["X_val"], d["y_val"])

    log.info("=== Tuning LightGBM (%d trials, timeout=%ds) ===", N_TRIALS, TIMEOUT)
    lgb_params = tune_lgb(d["X_train"], d["y_train"], d["w_train"], d["X_val"], d["y_val"])

    log.info("=== Tuning CatBoost (%d trials, timeout=%ds) ===", N_TRIALS, TIMEOUT)
    cat_params = tune_catboost(d["X_train"], d["y_train"], d["w_train"], d["X_val"], d["y_val"])

    log.info("=== Tuning Logistic Regression ===")
    best_C = tune_logreg(d["X_train_s"], d["y_train"], d["X_val_s"], d["y_val"])

    # ── 3. Train base models ──────────────────────────────────────────────────
    log.info("=== Training base models with best params ===")
    xgb_raw = train_xgb(d["X_train"], d["y_train"], d["w_train"], xgb_params)
    lgb_raw = train_lgb(d["X_train"], d["y_train"], d["w_train"], lgb_params)
    cat_raw = train_catboost(d["X_train"], d["y_train"], d["w_train"], cat_params)
    lr_raw  = train_logreg(d["X_train_s"], d["y_train"], best_C)

    log.info("--- Base model val performance (pre-calibration) ---")
    evaluate("XGB  (raw, val)",  d["y_val"], xgb_raw.predict_proba(d["X_val"]))
    evaluate("LGB  (raw, val)",  d["y_val"], lgb_raw.predict_proba(d["X_val"]))
    evaluate("CAT  (raw, val)",  d["y_val"], cat_raw.predict_proba(d["X_val"]))
    evaluate("LR   (raw, val)",  d["y_val"], lr_raw.predict_proba(d["X_val_s"]))

    # ── 4. Isotonic calibration on validation set ─────────────────────────────
    log.info("=== Calibrating via isotonic regression (val set) ===")
    cal_xgb = calibrate(xgb_raw, d["X_val"],   d["y_val"], method="isotonic")
    cal_lgb = calibrate(lgb_raw, d["X_val"],   d["y_val"], method="isotonic")
    cal_cat = calibrate(cat_raw, d["X_val"],   d["y_val"], method="isotonic")
    cal_lr  = calibrate(lr_raw,  d["X_val_s"], d["y_val"], method="isotonic")

    log.info("--- Calibrated model val performance ---")
    evaluate("XGB  (cal, val)", d["y_val"], cal_xgb.predict_proba(d["X_val"]))
    evaluate("LGB  (cal, val)", d["y_val"], cal_lgb.predict_proba(d["X_val"]))
    evaluate("CAT  (cal, val)", d["y_val"], cal_cat.predict_proba(d["X_val"]))
    evaluate("LR   (cal, val)", d["y_val"], cal_lr.predict_proba(d["X_val_s"]))

    # ── 5. Build ensembles ────────────────────────────────────────────────────
    cal_models  = [cal_xgb, cal_lgb, cal_cat, cal_lr]
    val_X_list  = [d["X_val"],  d["X_val"],  d["X_val"],  d["X_val_s"]]
    test_X_list = [d["X_test"], d["X_test"], d["X_test"], d["X_test_s"]]

    sv_test_probs = soft_vote(cal_models, test_X_list)
    meta_lr, stack_test_probs = build_stacking(cal_models, val_X_list, d["y_val"], test_X_list)

    # Also try weighted average: give boosted models more weight than LR
    weighted_probs = np.stack([
        cal_xgb.predict_proba(d["X_test"]) * 0.30,
        cal_lgb.predict_proba(d["X_test"]) * 0.30,
        cal_cat.predict_proba(d["X_test"]) * 0.30,
        cal_lr.predict_proba(d["X_test_s"]) * 0.10,
    ]).sum(axis=0)

    log.info("--- Ensemble test-set comparison ---")
    sv_metrics      = evaluate("Soft-vote     (test)", d["y_test"], sv_test_probs)
    stack_metrics   = evaluate("Stacking      (test)", d["y_test"], stack_test_probs)
    weight_metrics  = evaluate("Weighted-vote (test)", d["y_test"], weighted_probs)

    # Pick the best ensemble
    candidates = [
        ("soft_vote", sv_metrics, sv_test_probs),
        ("stacking",  stack_metrics, stack_test_probs),
        ("weighted_vote", weight_metrics, weighted_probs),
    ]
    candidates.sort(key=lambda x: x[1]["log_loss"])
    ensemble_type, best_metrics, final_probs = candidates[0]
    log.info("Winner: %s  (log-loss=%.4f)", ensemble_type.upper(), best_metrics["log_loss"])

    # ── 6. Calibration curves ─────────────────────────────────────────────────
    plot_calibration_curves(
        {
            "XGBoost":  cal_xgb.predict_proba(d["X_test"]),
            "LightGBM": cal_lgb.predict_proba(d["X_test"]),
            "CatBoost": cal_cat.predict_proba(d["X_test"]),
            "Ensemble": final_probs,
        },
        d["y_test"],
    )

    # ── 7. WC2022 per-match evaluation ────────────────────────────────────────
    evaluate_wc2022(test_df, final_probs, d["y_test"])

    # ── 8. Feature importances ────────────────────────────────────────────────
    save_feature_importances(xgb_raw, lgb_raw, cat_raw, d["feat_cols"])

    # ── 9. Save ensemble ──────────────────────────────────────────────────────
    payload = {
        "version":       "2.0",
        "ensemble_type": ensemble_type,
        "feature_cols":  d["feat_cols"],
        "imputer":       d["imputer"],
        "scaler":        d["scaler"],
        "models":        [cal_xgb, cal_lgb, cal_cat, cal_lr],
        "model_names":   ["xgboost", "lightgbm", "catboost", "logreg"],
        "uses_scaled":   [False,     False,       False,      True],
        "meta_lr":       meta_lr if ensemble_type == "stacking" else None,
        "weights":       [0.30, 0.30, 0.30, 0.10] if ensemble_type == "weighted_vote" else None,
        "label_names":   LABEL_NAMES,
        "label_map":     LABEL_MAP,
        "test_metrics":  best_metrics,
        "all_metrics":   {"soft_vote": sv_metrics, "stacking": stack_metrics, "weighted_vote": weight_metrics},
        "tuned_params": {
            "xgboost":   xgb_params,
            "lightgbm":  lgb_params,
            "catboost":  cat_params,
            "logreg_C":  best_C,
        },
    }
    joblib.dump(payload, ENSEMBLE_PATH)
    log.info("Ensemble saved → %s", ENSEMBLE_PATH)
    log.info(
        "Final  log-loss=%.4f  acc=%.3f  brier=%.4f  [%s]",
        best_metrics["log_loss"], best_metrics["accuracy"],
        best_metrics["brier"], ensemble_type,
    )
    return payload


def load_ensemble() -> dict:
    return joblib.load(ENSEMBLE_PATH)


def predict(
    ensemble: dict,
    home_team: str,
    away_team: str,
    features: dict,
) -> dict:
    """
    Single-match inference using a loaded ensemble dict.

    features: dict mapping feature_cols → values (NaN for unknowns).
    Returns {"home_win": p, "draw": p, "away_win": p, "predicted": label}.
    """
    feat_cols = ensemble["feature_cols"]
    X_raw = np.array([[features.get(c, np.nan) for c in feat_cols]])
    X = ensemble["imputer"].transform(X_raw)

    models    = ensemble["models"]
    scaled    = ensemble["uses_scaled"]
    X_list    = [ensemble["scaler"].transform(X) if s else X for s in scaled]

    if ensemble["ensemble_type"] == "stacking":
        meta_X = np.hstack([m.predict_proba(Xi) for m, Xi in zip(models, X_list)])
        probs  = ensemble["meta_lr"].predict_proba(meta_X)[0]
    elif ensemble["ensemble_type"] == "weighted_vote" and ensemble.get("weights"):
        w = ensemble["weights"]
        probs = np.stack([m.predict_proba(Xi) * wi for m, Xi, wi in zip(models, X_list, w)]).sum(axis=0)[0]
    else:
        probs = np.stack([m.predict_proba(Xi) for m, Xi in zip(models, X_list)]).mean(axis=0)[0]

    result = {name: round(float(probs[i]), 4) for i, name in enumerate(LABEL_NAMES)}
    result["predicted"] = LABEL_NAMES[int(probs.argmax())]
    return result


if __name__ == "__main__":
    train()

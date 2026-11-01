"""Training pipeline.  Run:  python -m msme_health.train

Trains, evaluates and saves:
  * Default-probability (PD) model  -> drives risk band, eligibility and limit
  * Financial Health Score model     -> the 0-100 headline score
  * Six pillar-score models          -> the sub-scores on the card
  * A linear baseline for PD and the score, to show what the boosted model adds
  * Anomaly detector (Isolation Forest + rules), tested on planted fraud cases
"""
from __future__ import annotations

import json
import warnings
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, mean_absolute_error, r2_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .anomaly import AnomalyDetector, rule_flags
from .config import (ARTIFACTS, CATEGORICAL, DATA_PATH, PILLARS, RANDOM_STATE, RISK_BANDS,
                     TARGET_PD, TARGET_SCORE)
from .features import DERIVED, MODEL_FEATURES, NUMERIC_RAW, PD_MONOTONE, add_derived, monotone_vector, prepare

LGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=RANDOM_STATE, verbose=-1,
)
EPS = 1e-4
warnings.filterwarnings("ignore", message=".*eval_set.*")


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def band(pd_values) -> np.ndarray:
    out = np.empty(len(pd_values), dtype=object)
    for i, p in enumerate(pd_values):
        out[i] = next(name for cut, name in RISK_BANDS if p < cut)
    return out


def ks_stat(y, score) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


def fit_lgb(X_tr, y_tr, X_va, y_va, monotone=None, **overrides):
    params = {**LGB_PARAMS, **overrides}
    if monotone is not None:
        params["monotone_constraints"] = monotone
        params["monotone_constraints_method"] = "advanced"
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def linear_baseline() -> Pipeline:
    """A scorecard-style linear model: what most banks would start with."""
    nums = NUMERIC_RAW + DERIVED
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ]), nums),
    ])
    return Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])


def pd_metrics(true_pd, pred_pd, y_sim) -> dict:
    true_band, pred_band = band(true_pd), band(pred_pd)
    high_true = (true_band == "High").astype(int)
    return {
        "pd_r2": round(r2_score(true_pd, pred_pd), 4),
        "pd_mae": round(mean_absolute_error(true_pd, pred_pd), 4),
        "logit_r2": round(r2_score(logit(true_pd), logit(pred_pd)), 4),
        "risk_band_accuracy": round(accuracy_score(true_band, pred_band), 4),
        "high_risk_auc": round(roc_auc_score(high_true, pred_pd), 4),
        "eligibility_accuracy": round(accuracy_score(true_band != "High", pred_band != "High"), 4),
        # Credit metrics against defaults simulated from the given PD (no real outcomes exist).
        "sim_default_auc": round(roc_auc_score(y_sim, pred_pd), 4),
        "sim_default_gini": round(2 * roc_auc_score(y_sim, pred_pd) - 1, 4),
        "sim_default_ks": round(ks_stat(y_sim, pred_pd), 4),
    }


def plant_anomalies(raw: pd.DataFrame, rng: np.random.Generator, n_each: int = 300) -> pd.DataFrame:
    """Create fake fraud cases by distorting clean rows in known ways."""
    cases = []
    base = raw.sample(n_each * 4, random_state=RANDOM_STATE).reset_index(drop=True)
    parts = [base.iloc[i * n_each:(i + 1) * n_each] for i in range(4)]

    a = parts[0].copy()  # circular transactions: money goes round in and out
    f = rng.uniform(1.8, 3.0, len(a))
    for c in ["Monthly_Bank_Credits_INR", "Monthly_Bank_Debits_INR", "Monthly_UPI_Inflow_INR"]:
        a[c] = a[c] * f
    a["anomaly_type"] = "Circular transactions"

    b = parts[1].copy()  # GST sales inflated without matching bank inflows
    f = rng.uniform(1.6, 2.5, len(b))
    b["Monthly_GST_Sales_INR"] *= f
    b["Monthly_GST_Purchases_INR"] *= f
    b["anomaly_type"] = "Inflated GST sales"

    c = parts[2].copy()  # ghost employees on payroll
    c["Monthly_Payroll_INR"] *= rng.uniform(4, 8, len(c))
    c["anomaly_type"] = "Ghost payroll"

    d = parts[3].copy()  # hidden debt: large EMI not matched by history
    d["Has_Existing_Loan"] = "Yes"
    d["Monthly_Loan_EMI_INR"] = d["Monthly_Bank_Credits_INR"] * rng.uniform(0.5, 0.8, len(d))
    d["EMI_On_Time_Rate_Pct"] = d["EMI_On_Time_Rate_Pct"].fillna(90)
    d["anomaly_type"] = "Hidden debt burden"
    cases += [a, b, c, d]
    return pd.concat(cases, ignore_index=True)


def main():
    t0 = time.time()
    ARTIFACTS.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    X = prepare(raw)
    strat = raw["Customer_Segment"] + "_" + raw["Credit_Risk_Category"]
    idx_tr, idx_tmp = train_test_split(raw.index, test_size=0.30, stratify=strat, random_state=RANDOM_STATE)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=strat[idx_tmp], random_state=RANDOM_STATE)
    print(f"train {len(idx_tr)}  val {len(idx_va)}  test {len(idx_te)}")

    metrics: dict = {"split": {"train": len(idx_tr), "val": len(idx_va), "test": len(idx_te)}}
    rng = np.random.default_rng(RANDOM_STATE)
    y_sim = (rng.random(len(raw)) < raw[TARGET_PD].to_numpy()).astype(int)  # simulated defaults
    true_pd_te = raw.loc[idx_te, TARGET_PD].to_numpy()

    # ---------- 1. PD model (target: log-odds of the given PD) ----------
    z = pd.Series(logit(raw[TARGET_PD].to_numpy()), index=raw.index)
    mono = monotone_vector(MODEL_FEATURES, PD_MONOTONE)
    pd_model = fit_lgb(X.loc[idx_tr], z[idx_tr], X.loc[idx_va], z[idx_va], monotone=mono)
    pd_free = fit_lgb(X.loc[idx_tr], z[idx_tr], X.loc[idx_va], z[idx_va], monotone=None)
    base_pd = linear_baseline().fit(raw.loc[idx_tr].pipe(add_derived), z[idx_tr])

    pred = {
        "LightGBM (monotonic)": sigmoid(pd_model.predict(X.loc[idx_te])),
        "LightGBM (unconstrained)": sigmoid(pd_free.predict(X.loc[idx_te])),
        "Linear baseline (Ridge)": sigmoid(base_pd.predict(raw.loc[idx_te].pipe(add_derived))),
    }
    metrics["pd_models"] = {k: pd_metrics(true_pd_te, v, y_sim[idx_te]) for k, v in pred.items()}
    metrics["pd_models"]["Given PD (ceiling)"] = {
        "sim_default_auc": round(roc_auc_score(y_sim[idx_te], true_pd_te), 4),
        "sim_default_gini": round(2 * roc_auc_score(y_sim[idx_te], true_pd_te) - 1, 4),
        "sim_default_ks": round(ks_stat(y_sim[idx_te], true_pd_te), 4),
    }
    metrics["pd_best_iteration"] = int(pd_model.best_iteration_)

    # Per-segment check: does the model work as well for NTC businesses?
    p_main = pred["LightGBM (monotonic)"]
    seg = raw.loc[idx_te, "Customer_Segment"].to_numpy()
    metrics["pd_by_segment"] = {
        s: pd_metrics(true_pd_te[seg == s], p_main[seg == s], y_sim[idx_te][seg == s])
        for s in ["NTC", "NTB", "Existing-to-Credit"]
    }

    # Calibration: predicted vs given PD by decile
    cal = pd.DataFrame({"pred": p_main, "true": true_pd_te})
    cal["decile"] = pd.qcut(cal["pred"], 10, labels=False)
    metrics["pd_calibration"] = cal.groupby("decile").agg(pred=("pred", "mean"), true=("true", "mean")).round(4).to_dict("list")

    # ---------- 2. Financial Health Score model ----------
    y_fhs = raw[TARGET_SCORE]
    fhs_mono = [-v for v in mono]  # a feature that raises risk must lower the score
    fhs_model = fit_lgb(X.loc[idx_tr], y_fhs[idx_tr], X.loc[idx_va], y_fhs[idx_va], monotone=fhs_mono)
    base_fhs = linear_baseline().fit(raw.loc[idx_tr].pipe(add_derived), y_fhs[idx_tr])
    f_pred = fhs_model.predict(X.loc[idx_te])
    b_pred = base_fhs.predict(raw.loc[idx_te].pipe(add_derived))
    metrics["score_models"] = {
        "LightGBM (monotonic)": {"r2": round(r2_score(y_fhs[idx_te], f_pred), 4), "mae": round(mean_absolute_error(y_fhs[idx_te], f_pred), 3)},
        "Linear baseline (Ridge)": {"r2": round(r2_score(y_fhs[idx_te], b_pred), 4), "mae": round(mean_absolute_error(y_fhs[idx_te], b_pred), 3)},
    }

    # ---------- 3. Pillar models ----------
    pillar_models, metrics["pillar_models"] = {}, {}
    for col, name in PILLARS.items():
        # Pillars are near-deterministic formulas: a small, fast model is enough (keeps serving memory low).
        m = fit_lgb(X.loc[idx_tr], raw.loc[idx_tr, col], X.loc[idx_va], raw.loc[idx_va, col],
                    n_estimators=500, learning_rate=0.1)
        p = m.predict(X.loc[idx_te])
        pillar_models[col] = m
        metrics["pillar_models"][name] = {
            "r2": round(r2_score(raw.loc[idx_te, col], p), 4),
            "mae": round(mean_absolute_error(raw.loc[idx_te, col], p), 3),
        }

    # ---------- 4. Global explanations (SHAP) ----------
    sample = X.loc[idx_te].sample(3000, random_state=RANDOM_STATE)
    sv = shap.TreeExplainer(pd_model).shap_values(sample)
    imp = pd.Series(np.abs(sv).mean(0), index=MODEL_FEATURES).sort_values(ascending=False)
    metrics["pd_shap_importance"] = imp.round(4).to_dict()
    sv_f = shap.TreeExplainer(fhs_model).shap_values(sample)
    metrics["score_shap_importance"] = pd.Series(np.abs(sv_f).mean(0), index=MODEL_FEATURES).sort_values(ascending=False).round(4).to_dict()

    # ---------- 5. Anomaly detection ----------
    feat_all = add_derived(raw)
    det = AnomalyDetector(contamination=0.02).fit(feat_all.loc[idx_tr])
    clean = feat_all.loc[idx_te]
    planted = add_derived(plant_anomalies(raw.loc[idx_te], rng))
    s_clean, s_plant = det.score(clean), det.score(planted)
    rules_clean = clean.apply(lambda r: len(rule_flags(r)) > 0, axis=1)
    rules_plant = planted.apply(lambda r: len(rule_flags(r)) > 0, axis=1)
    both_plant = s_plant["is_outlier"] | rules_plant
    metrics["anomaly"] = {
        "false_alarm_rate_clean": {
            "isolation_forest": round(float(s_clean["is_outlier"].mean()), 4),
            "rules": round(float(rules_clean.mean()), 4),
            "combined": round(float((s_clean["is_outlier"] | rules_clean).mean()), 4),
        },
        "detection_rate_planted": {
            t: {
                "isolation_forest": round(float(s_plant.loc[g.index, "is_outlier"].mean()), 3),
                "rules": round(float(rules_plant[g.index].mean()), 3),
                "combined": round(float(both_plant[g.index].mean()), 3),
            }
            for t, g in planted.groupby("anomaly_type")
        },
    }

    # ---------- 6. Save ----------
    joblib.dump({
        "pd": pd_model, "fhs": fhs_model, "pillars": pillar_models,
        "baseline_pd": base_pd, "baseline_fhs": base_fhs, "features": MODEL_FEATURES,
    }, ARTIFACTS / "models.joblib")
    joblib.dump(det, ARTIFACTS / "anomaly.joblib")
    pd.DataFrame({
        "MSME_ID": raw.loc[idx_te, "MSME_ID"].to_numpy(),
        "true_pd": true_pd_te, "pred_pd": p_main,
        "pred_pd_baseline": pred["Linear baseline (Ridge)"],
        "true_score": y_fhs[idx_te].to_numpy(), "pred_score": f_pred, "pred_score_baseline": b_pred,
        "sim_default": y_sim[idx_te], "segment": seg,
    }).to_parquet(ARTIFACTS / "test_predictions.parquet", index=False)
    pd.Series("test", index=raw.loc[idx_te, "MSME_ID"]).to_frame("split").to_parquet(ARTIFACTS / "test_ids.parquet")
    metrics["train_seconds"] = round(time.time() - t0, 1)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ["pd_models", "score_models", "pillar_models", "anomaly"]}, indent=2))


if __name__ == "__main__":
    main()

"""
risk_engine.py
Insurance Claim Intelligence — Risk Intelligence Engine

Converts a claim's attributes into a 0-100 Risk Score, a Risk Level
(Critical / High / Medium / Low), a Coverage Ratio, a Review Priority, and
a Recommended Action.

IMPORTANT — precomputed pipeline outputs:
The processed claims data may already contain outputs from the project's
own trained models (Risk_Level, High_Cost_Probability, Predicted_Claim_Amount,
Expected_Loss, Claim_Priority, Recommended_Action, Review_Priority, Scenario).
Those are the authoritative outputs of claim_severity_model.pkl /
high_cost_claim_model.pkl as already applied by the project's own pipeline.
This engine:
  1) preserves them untouched under a "Pipeline_" prefix (nothing is lost),
  2) reuses them wherever available instead of re-guessing (e.g. uses the
     real High_Cost_Probability instead of re-predicting it), and
  3) still computes the app's own 0-100 Risk Score per the fixed weighting
     scheme below, using the real risk-signal columns — adaptively
     normalized to each column's own observed range, since raw scales
     (Incident_Risk_Score, Reporting_Risk_Score, Customer_Claim_Intensity,
     etc.) vary by dataset and are not guaranteed to already be 0-100.

GOVERNANCE NOTE:
This module produces DECISION SUPPORT only. It never approves, rejects,
investigates, or settles a claim. All outputs are advisory signals for
authorized human personnel.
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SEVERITY_MODEL_PATH = os.path.join(MODELS_DIR, "claim_severity_model.pkl")
HIGH_COST_MODEL_PATH = os.path.join(MODELS_DIR, "high_cost_claim_model.pkl")

# Feature set used only as a live-prediction fallback (new/manual claims
# with no precomputed pipeline output yet). Claim_Amount and every
# post-claim / outcome field are intentionally excluded — leakage-safe.
HIGH_COST_FEATURES = [
    "Coverage_Amount",
    "Incident_Risk_Score",
    "Reporting_Delay_Days",
    "Customer_Claim_Intensity",
    "Document_Completeness",
    "Evidence_Count",
]

RISK_LEVELS = ["Critical", "High", "Medium", "Low"]

# Outputs the project's own pipeline may have already computed. Preserved
# under a "Pipeline_" prefix before this engine writes its own columns of
# the same name, so nothing from the real trained models is ever lost.
PIPELINE_OUTPUT_COLUMNS = [
    "Risk_Level", "Recommended_Action", "Review_Priority", "Claim_Priority",
    "Expected_Loss", "Predicted_Claim_Amount", "High_Cost_Probability", "Scenario",
]

# Candidate column names, in preference order, for signals whose exact
# name might vary slightly between pipeline stages.
REPORTING_RISK_CANDIDATES = ["Reporting_Risk_Score", "Reporting_Delay_Days"]
DOC_COMPLETENESS_CANDIDATES = ["Document_Completeness", "Documentation_Completeness"]


def _resolve_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _find_model_path(exact_path: str, keywords: list) -> str:
    """Resolve a model file: prefer the exact expected filename, otherwise
    fall back to the first .pkl in models/ whose name contains any keyword
    (case-insensitive) — so the app still finds the right model if it was
    exported under a slightly different filename.
    """
    if os.path.exists(exact_path):
        return exact_path
    if not os.path.isdir(MODELS_DIR):
        return exact_path
    for fpath in sorted(glob.glob(os.path.join(MODELS_DIR, "*.pkl"))):
        lname = os.path.basename(fpath).lower()
        if any(k in lname for k in keywords):
            return fpath
    return exact_path


def _safe_load_model(path):
    """Load a pickled model, returning None (not raising) if unavailable."""
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_models():
    """Load the severity and high-cost classification models.

    Returns a dict with keys 'severity_model' and 'high_cost_model'.
    Either value may be None if the corresponding .pkl file is missing
    or fails to load — callers must handle that gracefully. These are
    used only as a live-prediction fallback for claims that don't already
    carry precomputed pipeline outputs (see module docstring).
    """
    severity_path = _find_model_path(SEVERITY_MODEL_PATH, ["severity", "regress"])
    high_cost_path = _find_model_path(HIGH_COST_MODEL_PATH, ["high_cost", "highcost", "classif"])
    return {
        "severity_model": _safe_load_model(severity_path),
        "high_cost_model": _safe_load_model(high_cost_path),
    }


def models_available(models: dict) -> dict:
    """Report which models loaded successfully."""
    return {
        "severity_model": models.get("severity_model") is not None,
        "high_cost_model": models.get("high_cost_model") is not None,
    }


def predict_high_cost_probability(models: dict, claim_row: pd.DataFrame) -> float:
    """Predict High-Cost Claim probability for a single-row DataFrame.

    Used only when no precomputed 'High_Cost_Probability' is available
    (e.g. a brand-new manually entered claim). Falls back to a transparent
    heuristic if the trained model is unavailable or its expected features
    don't match this row.
    """
    model = models.get("high_cost_model")
    if model is not None:
        try:
            features = claim_row.reindex(columns=HIGH_COST_FEATURES, fill_value=0)
            if hasattr(model, "predict_proba"):
                return float(model.predict_proba(features)[:, 1][0])
            return float(model.predict(features)[0])
        except Exception:
            pass

    # Heuristic fallback — not a substitute for the trained classifier.
    row = claim_row.iloc[0]
    reporting_signal = row.get("Reporting_Risk_Score", None)
    if reporting_signal is None or pd.isna(reporting_signal):
        reporting_signal = min(float(row.get("Reporting_Delay_Days", 0)) / 30.0 * 100, 100)
    doc_completeness = row.get("Document_Completeness", row.get("Documentation_Completeness", 100))

    proxy = (
        0.35 * min(float(row.get("Incident_Risk_Score", 0)) / 100.0, 1.0)
        + 0.25 * min(float(row.get("Customer_Claim_Intensity", 0)) / 100.0, 1.0)
        + 0.20 * min(float(reporting_signal) / 100.0, 1.0)
        + 0.20 * (1 - min(float(doc_completeness) / 100.0, 1.0))
    )
    return float(np.clip(proxy, 0, 1))


def predict_claim_severity(models: dict, claim_row: pd.DataFrame) -> float:
    """Predict expected claim severity (₹) for a single-row DataFrame.

    Used only when no precomputed 'Predicted_Claim_Amount' is available.
    Falls back to Coverage_Amount if the trained model is unavailable.
    """
    model = models.get("severity_model")
    if model is not None:
        try:
            features = claim_row.reindex(columns=HIGH_COST_FEATURES, fill_value=0)
            return float(model.predict(features)[0])
        except Exception:
            pass
    return float(claim_row.iloc[0].get("Coverage_Amount", 0)) * 0.5


def compute_risk_score(
    high_cost_probability: float,
    claim_amount: float,
    coverage_amount: float,
    incident_risk_0_100: float,
    reporting_risk_0_100: float,
    customer_intensity_0_100: float,
    document_completeness_0_100: float,
    evidence_0_100: float,
) -> dict:
    """Compute the 0-100 Risk Score using the fixed weighting scheme:

        High-cost probability            -> 30 pts
        Claim amount vs coverage         -> 25 pts
        Incident risk                    -> 20 pts
        Reporting risk                   -> 10 pts
        Customer claim intensity         -> 10 pts
        Documentation / evidence adj.    ->  5 pts

    Every *_0_100 argument must already be on a 0-100 scale (higher =
    riskier for incident/reporting/intensity; higher = more complete /
    more evidence for document_completeness / evidence). For bulk
    portfolio scoring, score_claims_dataframe() adaptively normalizes raw
    columns onto this 0-100 scale before calling this function; for a
    manually entered claim in the UI, the sliders already are 0-100.

    Returns a dict with the total score, the risk level, and the
    per-component breakdown (for the "Risk breakdown" UI element).
    """
    coverage_amount = max(float(coverage_amount), 1.0)

    # 1) High-cost probability -> 30 pts
    pts_high_cost = np.clip(high_cost_probability, 0, 1) * 30

    # 2) Claim amount vs coverage -> 25 pts
    coverage_ratio = float(claim_amount) / coverage_amount
    pts_amount_vs_coverage = np.clip(coverage_ratio, 0, 1.25) / 1.25 * 25

    # 3) Incident risk -> 20 pts
    pts_incident_risk = np.clip(incident_risk_0_100, 0, 100) / 100 * 20

    # 4) Reporting risk -> 10 pts
    pts_reporting_risk = np.clip(reporting_risk_0_100, 0, 100) / 100 * 10

    # 5) Customer claim intensity -> 10 pts
    pts_claim_intensity = np.clip(customer_intensity_0_100, 0, 100) / 100 * 10

    # 6) Documentation / evidence adjustment -> 5 pts
    #    Lower completeness / less evidence increases risk.
    doc_gap = 1 - np.clip(document_completeness_0_100, 0, 100) / 100
    evidence_gap = 1 - np.clip(evidence_0_100, 0, 100) / 100
    pts_doc_evidence = ((doc_gap + evidence_gap) / 2) * 5

    total = (
        pts_high_cost
        + pts_amount_vs_coverage
        + pts_incident_risk
        + pts_reporting_risk
        + pts_claim_intensity
        + pts_doc_evidence
    )
    total = float(np.clip(total, 0, 100))

    return {
        "risk_score": round(total, 2),
        "risk_level": classify_risk_level(total),
        "coverage_ratio": round(coverage_ratio, 3),
        "breakdown": {
            "High-cost probability (30 pts)": round(float(pts_high_cost), 2),
            "Claim amount vs coverage (25 pts)": round(float(pts_amount_vs_coverage), 2),
            "Incident risk (20 pts)": round(float(pts_incident_risk), 2),
            "Reporting risk (10 pts)": round(float(pts_reporting_risk), 2),
            "Customer claim intensity (10 pts)": round(float(pts_claim_intensity), 2),
            "Documentation / evidence (5 pts)": round(float(pts_doc_evidence), 2),
        },
    }


def classify_risk_level(risk_score: float) -> str:
    """Bucket a 0-100 risk score into Critical / High / Medium / Low."""
    if risk_score >= 75:
        return "Critical"
    if risk_score >= 50:
        return "High"
    if risk_score >= 25:
        return "Medium"
    return "Low"


def review_priority(risk_level: str) -> str:
    """Human-facing review priority label for a risk level."""
    mapping = {
        "Critical": "Immediate Review",
        "High": "Priority Review",
        "Medium": "Standard Review",
        "Low": "Routine Review",
    }
    return mapping.get(risk_level, "Standard Review")


def recommended_action(risk_level: str) -> str:
    """Advisory recommended action text.

    IMPORTANT: This is guidance for a human reviewer, not an automated
    claim decision. No claim is approved, rejected, or settled here.
    """
    mapping = {
        "Critical": "Escalate immediately to senior claims examiner for manual review",
        "High": "Route to experienced adjuster within 24 hours for detailed review",
        "Medium": "Assign to standard claims queue for review within SLA",
        "Low": "Process through standard workflow; routine documentation check",
    }
    return mapping.get(risk_level, "Assign to standard claims queue for review")


def _adaptive_0_100(series: pd.Series) -> pd.Series:
    """Min-max scale a numeric column onto 0-100 using ITS OWN observed
    1st-99th percentile range (not an assumed fixed scale). This is what
    makes Risk_Score meaningful regardless of whether a raw column happens
    to already run 0-100, 0-1, or some other engineered range — every
    portfolio's own highest-risk claims land near 100, lowest near 0.
    """
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(50.0, index=series.index)  # unknown -> neutral, not zero
    lo, hi = valid.quantile(0.01), valid.quantile(0.99)
    if hi <= lo:
        lo, hi = valid.min(), valid.max()
    if hi <= lo:
        return pd.Series(50.0, index=series.index)  # constant column -> neutral
    scaled = (s - lo) / (hi - lo) * 100
    return scaled.clip(0, 100).fillna(50.0)


def score_claims_dataframe(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Vectorised risk scoring across an entire claims DataFrame.

    Preserves any precomputed pipeline outputs under a "Pipeline_" prefix,
    then adds this engine's own: High_Cost_Probability (reused from the
    pipeline when present), Risk_Score, Risk_Level, Coverage_Ratio,
    Review_Priority, Recommended_Action.
    """
    out = df.copy()

    # Preserve the project's own precomputed outputs before this engine
    # writes columns of the same name.
    for col in PIPELINE_OUTPUT_COLUMNS:
        if col in out.columns:
            out[f"Pipeline_{col}"] = out[col]

    # --- High-cost probability: reuse the pipeline's own value if present ---
    if "Pipeline_High_Cost_Probability" in out.columns:
        hcp = pd.to_numeric(out["Pipeline_High_Cost_Probability"], errors="coerce")
        max_val = hcp.dropna().max() if not hcp.dropna().empty else None
        # Normalize to a 0-1 probability regardless of whether it was
        # stored as 0-1 or 0-100 in the source file.
        if max_val is not None and max_val > 1.5:
            hcp = hcp / 100.0
        out["High_Cost_Probability"] = hcp.clip(0, 1).fillna(0.0)
    else:
        high_cost_model = models.get("high_cost_model")
        if high_cost_model is not None and hasattr(high_cost_model, "predict_proba"):
            try:
                features = out.reindex(columns=HIGH_COST_FEATURES, fill_value=0)
                out["High_Cost_Probability"] = high_cost_model.predict_proba(features)[:, 1]
            except Exception:
                out["High_Cost_Probability"] = out.apply(
                    lambda r: predict_high_cost_probability(models, out.loc[[r.name]]), axis=1
                )
        else:
            out["High_Cost_Probability"] = out.apply(
                lambda r: predict_high_cost_probability(models, out.loc[[r.name]]), axis=1
            )

    # --- Adaptively normalize the other risk signals onto 0-100 ---
    incident_col = _resolve_col(out, ["Incident_Risk_Score"])
    reporting_col = _resolve_col(out, REPORTING_RISK_CANDIDATES)
    intensity_col = _resolve_col(out, ["Customer_Claim_Intensity"])
    doc_col = _resolve_col(out, DOC_COMPLETENESS_CANDIDATES)
    evidence_col = _resolve_col(out, ["Evidence_Count"])

    incident_norm = _adaptive_0_100(out[incident_col]) if incident_col else pd.Series(50.0, index=out.index)
    reporting_norm = _adaptive_0_100(out[reporting_col]) if reporting_col else pd.Series(50.0, index=out.index)
    intensity_norm = _adaptive_0_100(out[intensity_col]) if intensity_col else pd.Series(50.0, index=out.index)
    doc_norm = _adaptive_0_100(out[doc_col]) if doc_col else pd.Series(50.0, index=out.index)
    evidence_norm = _adaptive_0_100(out[evidence_col]) if evidence_col else pd.Series(50.0, index=out.index)

    claim_amt = pd.to_numeric(out.get("Claim_Amount", 0), errors="coerce").fillna(0)
    coverage_amt = pd.to_numeric(out.get("Coverage_Amount", 1), errors="coerce").fillna(1).clip(lower=1)

    scores, levels, ratios, priorities, actions = [], [], [], [], []
    n = len(out)
    for i in range(n):
        result = compute_risk_score(
            high_cost_probability=out["High_Cost_Probability"].iloc[i],
            claim_amount=claim_amt.iloc[i],
            coverage_amount=coverage_amt.iloc[i],
            incident_risk_0_100=incident_norm.iloc[i],
            reporting_risk_0_100=reporting_norm.iloc[i],
            customer_intensity_0_100=intensity_norm.iloc[i],
            document_completeness_0_100=doc_norm.iloc[i],
            evidence_0_100=evidence_norm.iloc[i],
        )
        scores.append(result["risk_score"])
        levels.append(result["risk_level"])
        ratios.append(result["coverage_ratio"])
        priorities.append(review_priority(result["risk_level"]))
        actions.append(recommended_action(result["risk_level"]))

    out["Risk_Score"] = scores
    out["Risk_Level"] = levels
    out["Coverage_Ratio"] = ratios
    out["Review_Priority"] = priorities
    out["Recommended_Action"] = actions
    return out

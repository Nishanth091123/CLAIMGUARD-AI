"""
alerts_engine.py
Insurance Claim Intelligence — Automated Alerts Engine

Scans a scored claims portfolio and raises alerts for claims that cross
defined risk / anomaly / processing thresholds. Alerts are DECISION
SUPPORT signals only — they do not trigger any automatic claim action.

Alert types:
  Critical Risk, High Risk, High Expected Loss, Elevated Expected Loss,
  High-Cost Claim, Unusual Claim, High Anomaly Score, Processing Delay

Alert priority:
  Critical, High, Medium, Review

THRESHOLDS ARE ADAPTIVE, NOT HARD-CODED:
Expected loss and anomaly score don't have a universal "high" cutoff —
what counts as a large expected loss or an unusual anomaly score depends
on the portfolio's own distribution. Rather than an absolute number
guessed in advance, this engine flags the top slice of each portfolio's
own distribution (e.g. the top 10% highest expected-loss claims), so
alert volume adapts correctly to whatever data is loaded. Processing
Delay is the one exception — 30 days is a genuine business SLA, not a
data-scale artifact, so it stays an absolute threshold.
"""

import pandas as pd
import portfolio_analytics as pa

PROCESSING_DELAY_DAYS = 30          # genuine business SLA, not data-scale dependent
HIGH_COST_PROB_THRESHOLD = 0.5      # High_Cost_Probability is a true 0-1 probability

# Percentile cutoffs within the loaded portfolio's own distribution.
HIGH_EXPECTED_LOSS_PERCENTILE = 0.90       # top 10% of expected loss
ELEVATED_EXPECTED_LOSS_PERCENTILE = 0.75   # next slice, 75th-90th percentile
HIGH_ANOMALY_SCORE_PERCENTILE = 0.90       # top 10% of anomaly score

ALERT_PRIORITY_ORDER = ["Critical", "High", "Medium", "Review"]


def _col(df, name, default=0):
    return df[name] if name in df.columns else pd.Series([default] * len(df), index=df.index)


def _resolve_anomalous_mask(status_series: pd.Series) -> pd.Series:
    """Flag rows whose Anomaly_Status differs from the column's own
    majority value.

    Anomaly-status wording varies by pipeline ("Normal"/"Anomalous",
    "Not Flagged"/"Flagged - Review", 0/1, ...) and guessing every
    possible phrasing is fragile. In any real anomaly-detection labeling,
    "not anomalous" is always the large majority class — so the most
    frequent value in the column IS the normal baseline, whatever it's
    called, and every other value is the flagged minority.
    """
    s = status_series.astype(str).str.strip()
    valid = s[(s != "") & (s.str.lower() != "nan")]
    if valid.empty:
        return pd.Series(False, index=status_series.index)
    baseline = valid.mode(dropna=True)
    baseline_value = baseline.iloc[0] if not baseline.empty else None
    return (s != "") & (s.str.lower() != "nan") & (s != baseline_value)


def generate_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (claim, alert) pair that was triggered.

    Columns: Claim_ID, Alert_Type, Alert_Priority, Detail
    """
    rows = []

    # Expected loss: reuse the pipeline's own value when available (see
    # portfolio_analytics._per_claim_expected_loss), so alerts agree with
    # the Dashboard / Portfolio Analytics numbers.
    expected_loss = pa._per_claim_expected_loss(df)
    valid_loss = expected_loss.dropna()
    high_loss_cutoff = (
        valid_loss.quantile(HIGH_EXPECTED_LOSS_PERCENTILE) if not valid_loss.empty else float("inf")
    )
    elevated_loss_cutoff = (
        valid_loss.quantile(ELEVATED_EXPECTED_LOSS_PERCENTILE) if not valid_loss.empty else float("inf")
    )

    anomaly_score_series = None
    high_anomaly_cutoff = None
    if "Anomaly_Score" in df.columns:
        anomaly_score_series = pd.to_numeric(df["Anomaly_Score"], errors="coerce")
        valid_anomaly = anomaly_score_series.dropna()
        if not valid_anomaly.empty and valid_anomaly.nunique() > 1:
            high_anomaly_cutoff = valid_anomaly.quantile(HIGH_ANOMALY_SCORE_PERCENTILE)

    anomalous_status_mask = None
    if "Anomaly_Status" in df.columns:
        anomalous_status_mask = _resolve_anomalous_mask(df["Anomaly_Status"])

    for idx, row in df.iterrows():
        claim_id = row.get("Claim_ID", idx)
        risk_level = row.get("Risk_Level", "Low")
        exp_loss = expected_loss.loc[idx] if idx in expected_loss.index else 0

        if risk_level == "Critical":
            rows.append((claim_id, "Critical Risk", "Critical", f"Risk score {row.get('Risk_Score', 0):.1f}"))
        elif risk_level == "High":
            rows.append((claim_id, "High Risk", "High", f"Risk score {row.get('Risk_Score', 0):.1f}"))

        if pd.notna(exp_loss) and exp_loss >= high_loss_cutoff:
            rows.append((claim_id, "High Expected Loss", "Critical", f"Expected loss ₹{exp_loss:,.0f}"))
        elif pd.notna(exp_loss) and exp_loss >= elevated_loss_cutoff:
            rows.append((claim_id, "Elevated Expected Loss", "High", f"Expected loss ₹{exp_loss:,.0f}"))

        high_cost_prob = row.get("High_Cost_Probability", 0)
        if pd.notna(high_cost_prob) and high_cost_prob >= HIGH_COST_PROB_THRESHOLD:
            rows.append((claim_id, "High-Cost Claim", "High", f"High-cost probability {high_cost_prob:.0%}"))

        if anomalous_status_mask is not None and idx in anomalous_status_mask.index and anomalous_status_mask.loc[idx]:
            rows.append((claim_id, "Unusual Claim", "Medium", f"Anomaly status: {row.get('Anomaly_Status')}"))

        if anomaly_score_series is not None and high_anomaly_cutoff is not None:
            score = anomaly_score_series.loc[idx] if idx in anomaly_score_series.index else None
            if score is not None and pd.notna(score) and score >= high_anomaly_cutoff:
                rows.append((claim_id, "High Anomaly Score", "Medium", f"Anomaly score {float(score):.3f}"))

        processing_days = row.get("Processing_Days", None)
        if processing_days is not None and pd.notna(processing_days) and float(processing_days) >= PROCESSING_DELAY_DAYS:
            rows.append((claim_id, "Processing Delay", "Review", f"{float(processing_days):.0f} days in process"))

    return pd.DataFrame(rows, columns=["Claim_ID", "Alert_Type", "Alert_Priority", "Detail"])


def alert_summary(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """Alert counts by priority — Critical / High / Medium / Review."""
    counts = alerts_df["Alert_Priority"].value_counts().reindex(ALERT_PRIORITY_ORDER, fill_value=0)
    return pd.DataFrame({"Alert_Priority": counts.index, "Count": counts.values})


def alert_type_summary(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """Alert counts by alert type, ordered by priority then volume."""
    if alerts_df.empty:
        return pd.DataFrame(columns=["Alert_Type", "Alert_Priority", "Count"])
    grouped = (
        alerts_df.groupby(["Alert_Type", "Alert_Priority"]).size().reset_index(name="Count")
    )
    priority_rank = {p: i for i, p in enumerate(ALERT_PRIORITY_ORDER)}
    grouped["_rank"] = grouped["Alert_Priority"].map(priority_rank)
    return grouped.sort_values(["_rank", "Count"], ascending=[True, False]).drop(columns="_rank")

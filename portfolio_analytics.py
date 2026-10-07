"""
portfolio_analytics.py
Insurance Claim Intelligence — Portfolio Analytics Engine

Aggregates a scored claims portfolio (output of risk_engine.score_claims_dataframe)
into the analytics views used by the Executive Dashboard and the Portfolio
Analytics module (29D).

Where the project's own pipeline already computed Expected_Loss or
Predicted_Claim_Amount per claim (preserved by risk_engine.py as
Pipeline_Expected_Loss / Pipeline_Predicted_Claim_Amount), those real,
model-derived values are used directly instead of a generic heuristic.
"""

import numpy as np
import pandas as pd

RISK_ORDER = ["Critical", "High", "Medium", "Low"]

# Fallback only — used when the pipeline hasn't already computed a real
# Expected_Loss per claim (Pipeline_Expected_Loss column absent).
EXPECTED_LOSS_FACTOR = {
    "Critical": 0.55,
    "High": 0.35,
    "Medium": 0.15,
    "Low": 0.05,
}

# A "region" column's exact name can vary by pipeline; try these in order.
REGION_CANDIDATES = ["Region", "Customer_Region", "Incident_Location"]


def _col(df, name, default=0):
    return df[name] if name in df.columns else pd.Series([default] * len(df), index=df.index)


def _resolve_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _region_series(df: pd.DataFrame):
    """Return (column_name, series) for whichever region-like column is
    present, or (None, None) if none is available."""
    col = _resolve_col(df, REGION_CANDIDATES)
    if col is None:
        return None, None
    return col, df[col]


def risk_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count and % share of claims by Risk_Level."""
    counts = df["Risk_Level"].value_counts().reindex(RISK_ORDER, fill_value=0)
    pct = (counts / max(len(df), 1) * 100).round(1)
    return pd.DataFrame({"Risk_Level": counts.index, "Claims": counts.values, "Share_%": pct.values})


def _per_claim_expected_loss(df: pd.DataFrame) -> pd.Series:
    """Per-claim expected loss: reuse the pipeline's own computed value
    (Pipeline_Expected_Loss) when available; otherwise fall back to
    Claim_Amount * a risk-tier loss factor.
    """
    if "Pipeline_Expected_Loss" in df.columns:
        return pd.to_numeric(df["Pipeline_Expected_Loss"], errors="coerce").fillna(0.0)
    claim_amt = _col(df, "Claim_Amount")
    factor = df["Risk_Level"].map(EXPECTED_LOSS_FACTOR).fillna(0.1)
    return claim_amt * factor


def expected_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Expected loss summed by Risk_Level (see _per_claim_expected_loss)."""
    exp_loss = _per_claim_expected_loss(df)
    result = pd.DataFrame({"Risk_Level": df["Risk_Level"], "Expected_Loss": exp_loss})
    grouped = result.groupby("Risk_Level")["Expected_Loss"].sum().reindex(RISK_ORDER, fill_value=0)
    return grouped.reset_index()


def total_expected_loss(df: pd.DataFrame) -> float:
    return float(_per_claim_expected_loss(df).sum())


def scenario_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Best / expected / worst-case exposure scenarios.

    Best case: only Critical & High convert to full payout.
    Expected case: the pipeline's own expected loss (or the risk-tier
    heuristic fallback — see _per_claim_expected_loss).
    Worst case: every open claim pays out at full Claim_Amount.
    """
    claim_amt = _col(df, "Claim_Amount")
    total_claim_amount = float(claim_amt.sum())
    best_case = float(
        claim_amt[df["Risk_Level"].isin(["Critical", "High"])].sum() * 0.5
    )
    expected_case = total_expected_loss(df)
    worst_case = total_claim_amount

    return pd.DataFrame(
        {
            "Scenario": ["Best Case", "Expected Case", "Worst Case"],
            "Exposure": [best_case, expected_case, worst_case],
        }
    )


def claim_type_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if "Claim_Type" not in df.columns:
        return pd.DataFrame(columns=["Claim_Type", "Claims", "Total_Claim_Amount", "Avg_Risk_Score"])
    grouped = df.groupby("Claim_Type").agg(
        Claims=("Claim_Type", "count"),
        Total_Claim_Amount=("Claim_Amount", "sum") if "Claim_Amount" in df.columns else ("Claim_Type", "count"),
        Avg_Risk_Score=("Risk_Score", "mean"),
    ).reset_index()
    grouped["Avg_Risk_Score"] = grouped["Avg_Risk_Score"].round(1)
    return grouped.sort_values("Total_Claim_Amount", ascending=False)


def regional_analysis(df: pd.DataFrame) -> pd.DataFrame:
    region_col, region_series = _region_series(df)
    if region_col is None:
        return pd.DataFrame(columns=["Region", "Claims", "Total_Claim_Amount", "Avg_Risk_Score"])
    work = df.copy()
    work["Region"] = region_series
    grouped = work.groupby("Region").agg(
        Claims=("Region", "count"),
        Total_Claim_Amount=("Claim_Amount", "sum") if "Claim_Amount" in work.columns else ("Region", "count"),
        Avg_Risk_Score=("Risk_Score", "mean"),
    ).reset_index()
    grouped["Avg_Risk_Score"] = grouped["Avg_Risk_Score"].round(1)
    return grouped.sort_values("Total_Claim_Amount", ascending=False)


def policy_type_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if "Policy_Type" not in df.columns:
        return pd.DataFrame(columns=["Policy_Type", "Claims", "Total_Claim_Amount", "Avg_Risk_Score"])
    grouped = df.groupby("Policy_Type").agg(
        Claims=("Policy_Type", "count"),
        Total_Claim_Amount=("Claim_Amount", "sum") if "Claim_Amount" in df.columns else ("Policy_Type", "count"),
        Avg_Risk_Score=("Risk_Score", "mean"),
    ).reset_index()
    grouped["Avg_Risk_Score"] = grouped["Avg_Risk_Score"].round(1)
    return grouped.sort_values("Total_Claim_Amount", ascending=False)


def processing_days_summary(df: pd.DataFrame) -> dict:
    if "Processing_Days" not in df.columns:
        return {"avg": None, "median": None, "p90": None}
    pd_series = pd.to_numeric(df["Processing_Days"], errors="coerce").dropna()
    if pd_series.empty:
        return {"avg": None, "median": None, "p90": None}
    return {
        "avg": round(float(pd_series.mean()), 1),
        "median": round(float(pd_series.median()), 1),
        "p90": round(float(pd_series.quantile(0.9)), 1),
    }


def top_priority_claims(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    region_col, _ = _region_series(df)
    cols = [c for c in [
        "Claim_ID", "Claim_Type", region_col, "Claim_Amount", "Risk_Score",
        "Risk_Level", "Review_Priority", "Recommended_Action",
    ] if c and c in df.columns]
    return df.sort_values("Risk_Score", ascending=False).head(n)[cols]


def financial_exposure_summary(df: pd.DataFrame) -> dict:
    claim_amt = _col(df, "Claim_Amount")
    if "Pipeline_Predicted_Claim_Amount" in df.columns:
        predicted_exposure = float(
            pd.to_numeric(df["Pipeline_Predicted_Claim_Amount"], errors="coerce").fillna(0).sum()
        )
    else:
        predicted_exposure = float(claim_amt.sum() * 0.35)
    return {
        "total_claims": int(len(df)),
        "total_claim_amount": float(claim_amt.sum()),
        "average_claim": float(claim_amt.mean()) if len(df) else 0.0,
        "total_expected_loss": total_expected_loss(df),
        "predicted_exposure": predicted_exposure,
    }


def management_insights(df: pd.DataFrame) -> list:
    """Short, human-readable narrative insights derived from the portfolio."""
    insights = []
    if df.empty:
        return ["No claims available for analysis."]

    dist = risk_distribution(df)
    critical_share = float(dist.loc[dist["Risk_Level"] == "Critical", "Share_%"].values[0]) if not dist.empty else 0
    if critical_share > 10:
        insights.append(
            f"Critical-risk claims represent {critical_share}% of the portfolio — above the "
            "10% watch threshold; consider reallocating senior adjuster capacity."
        )
    else:
        insights.append(f"Critical-risk claims represent {critical_share}% of the portfolio — within normal range.")

    if "Claim_Type" in df.columns and "Claim_Amount" in df.columns:
        ct = claim_type_analysis(df)
        if not ct.empty:
            top_type = ct.iloc[0]
            insights.append(
                f"{top_type['Claim_Type']} claims account for the largest financial exposure "
                f"(₹{top_type['Total_Claim_Amount']:,.0f})."
            )

    region_col, _ = _region_series(df)
    if region_col is not None:
        reg = regional_analysis(df)
        if not reg.empty:
            top_region = reg.sort_values("Avg_Risk_Score", ascending=False).iloc[0]
            insights.append(
                f"{top_region['Region']} shows the highest average risk score "
                f"({top_region['Avg_Risk_Score']}) — a candidate for targeted review."
            )

    proc = processing_days_summary(df)
    if proc["p90"] is not None:
        insights.append(f"90th-percentile processing time is {proc['p90']} days — monitor for SLA breaches.")

    return insights

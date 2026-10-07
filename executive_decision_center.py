"""
executive_decision_center.py
Insurance Claim Intelligence — Executive Decision Center

Rolls Risk -> Financial Exposure -> Alerts -> Priority -> Management
Action up into a single executive summary, ending in a Human Decision
handoff. This module never issues a claim decision itself.
"""

import pandas as pd
import portfolio_analytics as pa


def build_executive_summary(claims_df: pd.DataFrame, alerts_df: pd.DataFrame, actions_df: pd.DataFrame) -> dict:
    fin = pa.financial_exposure_summary(claims_df)
    dist = pa.risk_distribution(claims_df)
    dist_map = dict(zip(dist["Risk_Level"], dist["Claims"]))

    action_counts = {"P1": 0, "P2": 0, "P3": 0, "P4": 0}
    if not actions_df.empty:
        vc = actions_df["Action_Priority"].value_counts()
        for p in action_counts:
            action_counts[p] = int(vc.get(p, 0))

    return {
        "total_claims": fin["total_claims"],
        "total_claim_amount": fin["total_claim_amount"],
        "expected_loss": fin["total_expected_loss"],
        "average_claim": fin["average_claim"],
        "critical_count": int(dist_map.get("Critical", 0)),
        "high_count": int(dist_map.get("High", 0)),
        "medium_count": int(dist_map.get("Medium", 0)),
        "low_count": int(dist_map.get("Low", 0)),
        "p1_count": action_counts["P1"],
        "p2_count": action_counts["P2"],
        "p3_count": action_counts["P3"],
        "p4_count": action_counts["P4"],
        "total_alerts": int(len(alerts_df)),
    }


def top_financial_exposure_claims(claims_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    region_col, _ = pa._region_series(claims_df)
    cols = [c for c in [
        "Claim_ID", "Claim_Type", region_col, "Claim_Amount", "Risk_Score", "Risk_Level"
    ] if c and c in claims_df.columns]
    if "Claim_Amount" not in claims_df.columns:
        return pd.DataFrame(columns=cols)
    return claims_df.sort_values("Claim_Amount", ascending=False).head(n)[cols]


def p1_immediate_decisions(actions_df: pd.DataFrame, claims_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """P1 worklist enriched with claim financial context, for the
    'P1 immediate decisions' panel. Human personnel make the final call."""
    if actions_df.empty:
        return actions_df
    p1 = actions_df[actions_df["Action_Priority"] == "P1"].head(n)
    if claims_df is not None and "Claim_ID" in claims_df.columns and "Claim_Amount" in claims_df.columns:
        p1 = p1.merge(
            claims_df[["Claim_ID", "Claim_Amount", "Risk_Score"]], on="Claim_ID", how="left"
        )
    return p1


GOVERNANCE_WARNING = (
    "GOVERNANCE NOTICE: Insurance Claim Intelligence is an AI-assisted decision-support "
    "system. It does NOT automatically approve, reject, investigate, or settle insurance "
    "claims. Every risk score, alert, and recommended action on this screen is advisory. "
    "Final claim decisions remain the responsibility of authorized human personnel, in "
    "accordance with organizational policy and regulatory requirements."
)

"""
management_actions.py
Insurance Claim Intelligence — Management Action Engine

Converts per-claim alerts into a prioritized management-action worklist.
This module recommends actions for human managers; it never executes a
claim decision itself.

Priority levels:
  P1 — Immediate
  P2 — High
  P3 — Monitor
  P4 — Review
"""

import pandas as pd

ALERT_PRIORITY_TO_ACTION_PRIORITY = {
    "Critical": "P1",
    "High": "P2",
    "Medium": "P3",
    "Review": "P4",
}

ACTION_PRIORITY_ORDER = ["P1", "P2", "P3", "P4"]

ACTION_LABEL = {
    "P1": "Immediate",
    "P2": "High",
    "P3": "Monitor",
    "P4": "Review",
}

REVIEW_LEVEL = {
    "P1": "Senior Examiner / Manager Escalation",
    "P2": "Experienced Adjuster Review",
    "P3": "Standard Adjuster Monitoring",
    "P4": "Routine Documentation Review",
}

MANAGEMENT_ACTION_TEXT = {
    "P1": "Escalate to senior claims manager for immediate human review",
    "P2": "Assign to experienced adjuster within 24 hours",
    "P3": "Place on monitoring queue; re-assess at next portfolio review",
    "P4": "Queue for routine review within standard SLA",
}

NEXT_STEP_TEXT = {
    "P1": "Notify management immediately (see Notification & Escalation)",
    "P2": "Adjuster contacts claimant / gathers additional evidence",
    "P3": "Track claim status; re-alert if risk score increases",
    "P4": "Complete standard documentation checklist",
}


def build_management_actions(alerts_df: pd.DataFrame, claims_df: pd.DataFrame) -> pd.DataFrame:
    """Roll per-claim alerts up into one management-action row per claim,
    keyed off the claim's single highest-severity alert.
    """
    if alerts_df.empty:
        return pd.DataFrame(
            columns=[
                "Claim_ID", "Action_Priority", "Priority_Label", "Review_Level",
                "Management_Action", "Reason", "Next_Step",
            ]
        )

    priority_rank = {"Critical": 0, "High": 1, "Medium": 2, "Review": 3}
    alerts_df = alerts_df.copy()
    alerts_df["_rank"] = alerts_df["Alert_Priority"].map(priority_rank)

    # Take the highest-severity alert per claim as the driver for the action.
    top_alert = (
        alerts_df.sort_values("_rank")
        .groupby("Claim_ID", as_index=False)
        .first()
    )
    # Collect all alert reasons per claim for the "Reason" column.
    reasons = (
        alerts_df.groupby("Claim_ID")["Alert_Type"]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .reset_index()
        .rename(columns={"Alert_Type": "Reason"})
    )

    merged = top_alert.merge(reasons, on="Claim_ID")
    merged["Action_Priority"] = merged["Alert_Priority"].map(ALERT_PRIORITY_TO_ACTION_PRIORITY)
    merged["Priority_Label"] = merged["Action_Priority"].map(ACTION_LABEL)
    merged["Review_Level"] = merged["Action_Priority"].map(REVIEW_LEVEL)
    merged["Management_Action"] = merged["Action_Priority"].map(MANAGEMENT_ACTION_TEXT)
    merged["Next_Step"] = merged["Action_Priority"].map(NEXT_STEP_TEXT)

    result = merged[
        ["Claim_ID", "Action_Priority", "Priority_Label", "Review_Level",
         "Management_Action", "Reason", "Next_Step"]
    ]
    priority_order = {p: i for i, p in enumerate(ACTION_PRIORITY_ORDER)}
    result = result.assign(_o=result["Action_Priority"].map(priority_order)).sort_values("_o").drop(columns="_o")
    return result.reset_index(drop=True)


def action_summary(actions_df: pd.DataFrame) -> pd.DataFrame:
    """Counts by P1-P4 priority."""
    counts = actions_df["Action_Priority"].value_counts().reindex(ACTION_PRIORITY_ORDER, fill_value=0)
    labels = [ACTION_LABEL[p] for p in ACTION_PRIORITY_ORDER]
    return pd.DataFrame({"Action_Priority": ACTION_PRIORITY_ORDER, "Label": labels, "Count": counts.values})

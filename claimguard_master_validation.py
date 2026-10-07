"""
CLAIMGUARD AI - MASTER PRODUCTION VALIDATION
=============================================
Single read-only validation suite for the existing CLAIMGUARD AI project.

Checks:
1. File/data/model availability
2. Data quality and target audit
3. Production model schema compatibility
4. Regression diagnostics
5. Classification diagnostics
6. Feature importance / explainability
7. Regression error and segment analysis
8. False-positive / false-negative analysis
9. Probability-threshold analysis
10. Missing-value and numeric outlier audit
11. Leakage-risk heuristic audit
12. Business-rule / risk-engine consistency audit
13. Generalization / overfitting caution checks
14. Consolidated PASS / REVIEW / WARNING summary
15. Production-readiness recommendation

IMPORTANT
- Does NOT retrain any model.
- Does NOT overwrite production .pkl files.
- Does NOT modify app.py or risk_engine.py.
- Full-dataset metrics are diagnostics, not a substitute for the original held-out test.
"""

from pathlib import Path
import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data" / "processed" / "insurance_claims_features.csv"
SEVERITY_FILE = BASE / "models" / "claim_severity_model.pkl"
HIGH_COST_FILE = BASE / "models" / "high_cost_claim_model.pkl"
RISK_ENGINE_FILE = BASE / "risk_engine.py"
APP_FILE = BASE / "app.py"

OUT = BASE / "claimguard_master_validation_results"
OUT.mkdir(exist_ok=True)

# Documented historical held-out references from the completed project.
REF_REG = {
    "MAE": 143067.30,
    "RMSE": 219593.00,
    "R2": 0.7869,
}
REF_CLS = {
    "Accuracy": 0.9350,
    "Precision": 0.6667,
    "Recall": 0.7000,
    "F1": 0.6829,
    "ROC_AUC": 0.9361,
}

EXPECTED_RISK_ENGINE_FEATURES = [
    "Coverage_Amount",
    "Incident_Risk_Score",
    "Reporting_Delay_Days",
    "Customer_Claim_Intensity",
    "Document_Completeness",
    "Evidence_Count",
]

SEGMENTS = [
    "Claim_Type",
    "Incident_Severity",
    "Damage_Level",
    "Incident_Type",
    "Asset_Type",
    "Customer_Region",
    "Incident_Location",
    "Policy_Type",
    "Claim_Channel",
    "Document_Completeness",
]

LEAKAGE_TERMS = [
    "claim_amount",
    "settlement",
    "settled",
    "approved_amount",
    "paid_amount",
    "payment_amount",
    "final_amount",
    "actual_loss",
    "final_decision",
    "claim_outcome",
]

checks = []


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def add_check(area, check, status, detail):
    checks.append({
        "Area": area,
        "Check": check,
        "Status": status,
        "Detail": str(detail),
    })


def numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def load_model(path):
    return joblib.load(path)


def expected_columns(model):
    if hasattr(model, "feature_names_in_"):
        return [str(x) for x in model.feature_names_in_]

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return [str(x) for x in step.feature_names_in_]

        # ColumnTransformer fallback
        for step in model.named_steps.values():
            if hasattr(step, "transformers_"):
                cols = []
                for _, _, selected in step.transformers_:
                    if selected is None or selected == "drop":
                        continue
                    if isinstance(selected, (list, tuple, np.ndarray, pd.Index)):
                        cols.extend([str(x) for x in selected if isinstance(x, str)])
                if cols:
                    # Preserve order, remove duplicates.
                    return list(dict.fromkeys(cols))

    raise RuntimeError("Could not extract fitted model input schema.")


def normalize_binary(values):
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        return (arr.astype(float) >= 0.5).astype(int)

    s = pd.Series(arr).astype("string").str.strip().str.lower()
    positive = {
        "1", "true", "yes", "high", "high-cost", "high cost",
        "high_cost", "high-cost claim", "high cost claim",
    }
    return s.isin(positive).astype(int).to_numpy()


def positive_probability(model, X):
    if not hasattr(model, "predict_proba"):
        return None

    p = np.asarray(model.predict_proba(X))
    if p.ndim != 2 or p.shape[1] < 2:
        return None

    classes = getattr(model, "classes_", None)
    if classes is not None:
        positive = {
            "1", "true", "yes", "high", "high-cost", "high cost",
            "high_cost", "high-cost claim", "high cost claim",
        }
        for i, c in enumerate(classes):
            if str(c).strip().lower() in positive:
                return p[:, i]

    return p[:, 1]


def classification_target(df, y_reg):
    for c in [
        "High_Cost_Claim",
        "High_Cost",
        "Is_High_Cost",
        "High_Cost_Flag",
        "HighCostClaim",
    ]:
        if c in df.columns:
            raw = df[c]
            if pd.api.types.is_numeric_dtype(raw):
                y = (pd.to_numeric(raw, errors="coerce").fillna(0) >= 0.5).astype(int)
            else:
                s = raw.astype("string").str.strip().str.lower()
                y = s.isin({
                    "1", "true", "yes", "high", "high-cost",
                    "high cost", "high_cost",
                }).astype(int)
            return y, c, None

    threshold = float(y_reg.quantile(0.90))
    return (
        (y_reg >= threshold).astype(int),
        "Claim_Amount >= current 90th percentile",
        threshold,
    )


def id_column(df):
    for c in ["Claim_ID", "Claim_Id", "claim_id", "ID", "Id"]:
        if c in df.columns:
            return c
    return None


def get_preprocessor_estimator(pipe):
    if not hasattr(pipe, "named_steps"):
        return None, pipe

    steps = list(pipe.named_steps.items())
    if not steps:
        return None, pipe

    estimator = steps[-1][1]

    if "preprocessor" in pipe.named_steps:
        return pipe.named_steps["preprocessor"], estimator

    if len(steps) >= 2:
        return steps[-2][1], estimator

    return None, estimator


def importance_tables(pipe):
    pre, estimator = get_preprocessor_estimator(pipe)
    source = expected_columns(pipe)

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        method = "feature_importances_"
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coef) if coef.ndim == 1 else np.mean(np.abs(coef), axis=0)
        method = "absolute coefficient magnitude"
    else:
        return None, None, "unavailable"

    names = None
    if pre is not None and hasattr(pre, "get_feature_names_out"):
        try:
            names = [str(x) for x in pre.get_feature_names_out()]
        except Exception:
            names = None

    if names is None or len(names) != len(values):
        names = [f"feature_{i}" for i in range(len(values))]

    def map_source(name):
        clean = name.split("__", 1)[1] if "__" in name else name
        if clean in source:
            return clean
        for s in sorted(source, key=len, reverse=True):
            if (
                clean.startswith(s + "_")
                or clean.startswith(s + "=")
                or clean.startswith(s + "[")
            ):
                return s
        return clean

    transformed = pd.DataFrame({
        "Transformed_Feature": names,
        "Importance": values,
    })
    transformed["Source_Feature"] = [
        map_source(x) for x in transformed["Transformed_Feature"]
    ]
    transformed = transformed.sort_values("Importance", ascending=False).reset_index(drop=True)
    transformed["Rank"] = np.arange(1, len(transformed) + 1)

    grouped = (
        transformed.groupby("Source_Feature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    grouped["Rank"] = np.arange(1, len(grouped) + 1)
    total = grouped["Importance"].sum()
    grouped["Importance_Percent"] = np.where(
        total > 0, grouped["Importance"] / total * 100, 0
    )
    grouped["Cumulative_Importance_Percent"] = grouped["Importance_Percent"].cumsum()

    return transformed, grouped, method


def regression_segment_breakdown(frame):
    all_rows = []
    for segment in [c for c in SEGMENTS if c in frame.columns]:
        for value, g in frame.groupby(segment, dropna=False):
            if len(g) < 2:
                continue
            all_rows.append({
                "Segment": segment,
                "Value": str(value),
                "Rows": len(g),
                "Actual_Mean": g["Actual_Claim_Amount"].mean(),
                "Predicted_Mean": g["Predicted_Claim_Amount"].mean(),
                "MAE": g["Absolute_Error"].mean(),
                "RMSE": np.sqrt(np.mean(np.square(g["Residual"]))),
                "Mean_Residual": g["Residual"].mean(),
                "P90_Absolute_Error": g["Absolute_Error"].quantile(0.90),
            })
    return pd.DataFrame(all_rows)


def classification_segment_breakdown(frame):
    all_rows = []
    for segment in [c for c in SEGMENTS if c in frame.columns]:
        for value, g in frame.groupby(segment, dropna=False):
            if len(g) < 2:
                continue
            y = g["Actual_High_Cost"].to_numpy()
            p = g["Predicted_High_Cost"].to_numpy()
            row = {
                "Segment": segment,
                "Value": str(value),
                "Rows": len(g),
                "Actual_High_Cost": int(y.sum()),
                "Predicted_High_Cost": int(p.sum()),
                "Accuracy": accuracy_score(y, p),
                "Precision": precision_score(y, p, zero_division=0),
                "Recall": recall_score(y, p, zero_division=0),
                "F1": f1_score(y, p, zero_division=0),
                "False_Positives": int(((y == 0) & (p == 1)).sum()),
                "False_Negatives": int(((y == 1) & (p == 0)).sum()),
            }
            if "High_Cost_Probability" in g.columns and len(np.unique(y)) == 2:
                row["ROC_AUC"] = roc_auc_score(y, g["High_Cost_Probability"])
            else:
                row["ROC_AUC"] = np.nan
            all_rows.append(row)
    return pd.DataFrame(all_rows)


def threshold_table(y, prob):
    rows = []
    for t in np.arange(0.20, 0.81, 0.05):
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            "Threshold": round(float(t), 2),
            "Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "Review_Rate": float(pred.mean()),
        })
    return pd.DataFrame(rows)


def numeric_outlier_audit(df, columns):
    rows = []
    for c in columns:
        if c not in df.columns:
            continue
        s = numeric(df[c]).dropna()
        if len(s) < 10:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            low, high = q1, q3
            count = int(((s < low) | (s > high)).sum())
        else:
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            count = int(((s < low) | (s > high)).sum())
        rows.append({
            "Feature": c,
            "Non_Null": len(s),
            "Min": s.min(),
            "Q1": q1,
            "Median": s.median(),
            "Q3": q3,
            "Max": s.max(),
            "IQR_Lower_Bound": low,
            "IQR_Upper_Bound": high,
            "IQR_Outlier_Count": count,
            "IQR_Outlier_Percent": count / len(s) * 100,
        })
    return pd.DataFrame(rows)


def parse_risk_engine_features():
    if not RISK_ENGINE_FILE.exists():
        return None

    text = RISK_ENGINE_FILE.read_text(encoding="utf-8", errors="ignore")
    m = re.search(
        r"HIGH_COST_FEATURES\s*=\s*\[(.*?)\]",
        text,
        flags=re.DOTALL,
    )
    if not m:
        return None

    return re.findall(r'["\']([^"\']+)["\']', m.group(1))


def save_top_importance_chart(grouped, title, filename):
    if grouped is None or grouped.empty:
        return
    top = grouped.head(15).sort_values("Importance", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(top["Source_Feature"], top["Importance"])
    plt.xlabel("Aggregated Importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=170)
    plt.close()


def save_threshold_chart(tab):
    if tab is None or tab.empty:
        return
    plt.figure(figsize=(9, 6))
    plt.plot(tab["Threshold"], tab["Precision"], marker="o", label="Precision")
    plt.plot(tab["Threshold"], tab["Recall"], marker="o", label="Recall")
    plt.plot(tab["Threshold"], tab["F1"], marker="o", label="F1")
    plt.xlabel("Probability Threshold")
    plt.ylabel("Score")
    plt.title("High-Cost Model Threshold Trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "classification_threshold_tradeoff.png", dpi=170)
    plt.close()


def save_regression_error_chart(reg):
    top = reg.nlargest(15, "Absolute_Error").sort_values("Absolute_Error")
    plt.figure(figsize=(10, 7))
    plt.barh(top["Display_ID"].astype(str), top["Absolute_Error"])
    plt.xlabel("Absolute Claim Amount Error")
    plt.ylabel("Claim")
    plt.title("Top 15 Largest Claim Amount Errors")
    plt.tight_layout()
    plt.savefig(OUT / "regression_top15_errors.png", dpi=170)
    plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("\n" + "=" * 82)
    print("CLAIMGUARD AI - MASTER PRODUCTION VALIDATION")
    print("=" * 82)
    print("Single READ-ONLY validation suite. Production files will NOT be changed.\n")

    # ---------------------------------------------------------------
    # 1. File availability
    # ---------------------------------------------------------------
    required_files = {
        "Processed dataset": DATA_FILE,
        "Severity model": SEVERITY_FILE,
        "High-cost model": HIGH_COST_FILE,
    }
    optional_files = {
        "risk_engine.py": RISK_ENGINE_FILE,
        "app.py": APP_FILE,
    }

    fatal = False
    for name, path in required_files.items():
        if path.exists():
            add_check("Files", name, "PASS", str(path.relative_to(BASE)))
        else:
            add_check("Files", name, "FAIL", f"Missing: {path}")
            fatal = True

    for name, path in optional_files.items():
        add_check(
            "Files",
            name,
            "PASS" if path.exists() else "REVIEW",
            str(path.relative_to(BASE)) if path.exists() else "File not found",
        )

    if fatal:
        pd.DataFrame(checks).to_csv(OUT / "master_checklist.csv", index=False)
        print("Required project files are missing. See master_checklist.csv.")
        return

    # ---------------------------------------------------------------
    # 2. Dataset audit
    # ---------------------------------------------------------------
    df0 = pd.read_csv(DATA_FILE)
    add_check("Data", "Dataset loaded", "PASS",
              f"{len(df0):,} rows x {df0.shape[1]} columns")

    duplicate_rows = int(df0.duplicated().sum())
    add_check(
        "Data",
        "Exact duplicate rows",
        "PASS" if duplicate_rows == 0 else "REVIEW",
        f"{duplicate_rows} exact duplicate rows",
    )

    if "Claim_Amount" not in df0.columns:
        add_check("Data", "Claim_Amount target", "FAIL", "Claim_Amount is missing")
        pd.DataFrame(checks).to_csv(OUT / "master_checklist.csv", index=False)
        print("Claim_Amount is missing. Validation stopped safely.")
        return

    y0 = numeric(df0["Claim_Amount"])
    valid = y0.notna()
    invalid_target = int((~valid).sum())
    add_check(
        "Data",
        "Claim_Amount numeric target",
        "PASS" if invalid_target == 0 else "REVIEW",
        f"{valid.sum():,} valid / {len(df0):,}; invalid={invalid_target}",
    )

    df = df0.loc[valid].copy()
    y_reg = y0.loc[valid].astype(float)

    missing = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": [int(df[c].isna().sum()) for c in df.columns],
        "Missing_Percent": [float(df[c].isna().mean() * 100) for c in df.columns],
        "Dtype": [str(df[c].dtype) for c in df.columns],
        "Unique_Count": [int(df[c].nunique(dropna=True)) for c in df.columns],
    }).sort_values(["Missing_Percent", "Column"], ascending=[False, True])
    missing.to_csv(OUT / "data_quality_missingness.csv", index=False)

    high_missing = missing[missing["Missing_Percent"] > 20]
    add_check(
        "Data",
        "High missingness columns",
        "PASS" if high_missing.empty else "REVIEW",
        "No columns >20% missing"
        if high_missing.empty
        else f"{len(high_missing)} columns exceed 20% missing",
    )

    if "Document_Completeness" in df.columns:
        doc_counts = df["Document_Completeness"].astype("string").value_counts(
            dropna=False
        ).to_dict()
        add_check(
            "Data",
            "Document_Completeness",
            "PASS",
            json.dumps({str(k): int(v) for k, v in doc_counts.items()}),
        )

    # ---------------------------------------------------------------
    # 3. Models + schema
    # ---------------------------------------------------------------
    severity = load_model(SEVERITY_FILE)
    classifier = load_model(HIGH_COST_FILE)

    reg_cols = expected_columns(severity)
    cls_cols = expected_columns(classifier)

    reg_missing = [c for c in reg_cols if c not in df.columns]
    cls_missing = [c for c in cls_cols if c not in df.columns]

    add_check(
        "Schema",
        "Severity model schema",
        "PASS" if not reg_missing else "FAIL",
        f"Expected {len(reg_cols)} features; missing={reg_missing}",
    )
    add_check(
        "Schema",
        "High-cost model schema",
        "PASS" if not cls_missing else "FAIL",
        f"Expected {len(cls_cols)} features; missing={cls_missing}",
    )

    pd.DataFrame({
        "Severity_Model_Features": pd.Series(reg_cols),
        "High_Cost_Model_Features": pd.Series(cls_cols),
    }).to_csv(OUT / "production_model_schema.csv", index=False)

    if reg_missing or cls_missing:
        pd.DataFrame(checks).to_csv(OUT / "master_checklist.csv", index=False)
        print("Production schema mismatch. Validation stopped before prediction.")
        return

    X_reg = df[reg_cols].copy()
    X_cls = df[cls_cols].copy()

    # ---------------------------------------------------------------
    # 4. Regression diagnostics
    # ---------------------------------------------------------------
    reg_pred = np.asarray(severity.predict(X_reg), dtype=float)

    reg_mae = mean_absolute_error(y_reg, reg_pred)
    reg_rmse = np.sqrt(mean_squared_error(y_reg, reg_pred))
    reg_r2 = r2_score(y_reg, reg_pred)

    add_check(
        "Regression",
        "Full-dataset diagnostic",
        "PASS",
        f"MAE={reg_mae:,.2f}; RMSE={reg_rmse:,.2f}; R2={reg_r2:.4f}",
    )
    add_check(
        "Regression",
        "Historical held-out reference retained",
        "PASS",
        f"MAE={REF_REG['MAE']:,.2f}; RMSE={REF_REG['RMSE']:,.2f}; "
        f"R2={REF_REG['R2']:.4f}; current full-data metric is not a replacement",
    )

    ids = id_column(df)
    display_id = (
        df[ids].astype(str).to_numpy()
        if ids else np.array([f"ROW-{i}" for i in df.index])
    )

    reg = df.copy()
    reg["Display_ID"] = display_id
    reg["Actual_Claim_Amount"] = np.asarray(y_reg)
    reg["Predicted_Claim_Amount"] = reg_pred
    reg["Residual"] = reg["Actual_Claim_Amount"] - reg["Predicted_Claim_Amount"]
    reg["Absolute_Error"] = reg["Residual"].abs()
    reg["Absolute_Percentage_Error"] = np.where(
        reg["Actual_Claim_Amount"].abs() > 0,
        reg["Absolute_Error"] / reg["Actual_Claim_Amount"].abs() * 100,
        np.nan,
    )

    export_reg = [
        "Display_ID",
        "Actual_Claim_Amount",
        "Predicted_Claim_Amount",
        "Residual",
        "Absolute_Error",
        "Absolute_Percentage_Error",
    ] + [c for c in SEGMENTS if c in reg.columns]

    reg[export_reg].sort_values(
        "Absolute_Error", ascending=False
    ).to_csv(OUT / "regression_all_errors.csv", index=False)

    reg[export_reg].nlargest(
        50, "Absolute_Error"
    ).to_csv(OUT / "regression_top50_errors.csv", index=False)

    reg_break = regression_segment_breakdown(reg)
    if not reg_break.empty:
        reg_break.sort_values(
            ["MAE", "Rows"], ascending=[False, False]
        ).to_csv(OUT / "regression_segment_breakdown.csv", index=False)

    # ---------------------------------------------------------------
    # 5. Classification diagnostics
    # ---------------------------------------------------------------
    y_cls, target_source, high_cost_threshold = classification_target(df, y_reg)
    cls_pred = normalize_binary(classifier.predict(X_cls))
    cls_prob = positive_probability(classifier, X_cls)

    cls_acc = accuracy_score(y_cls, cls_pred)
    cls_precision = precision_score(y_cls, cls_pred, zero_division=0)
    cls_recall = recall_score(y_cls, cls_pred, zero_division=0)
    cls_f1 = f1_score(y_cls, cls_pred, zero_division=0)
    cls_auc = (
        roc_auc_score(y_cls, cls_prob)
        if cls_prob is not None and len(np.unique(y_cls)) == 2
        else np.nan
    )

    tn, fp, fn, tp = confusion_matrix(
        y_cls, cls_pred, labels=[0, 1]
    ).ravel()

    add_check(
        "Classification",
        "Full-dataset diagnostic",
        "PASS",
        f"Accuracy={cls_acc:.2%}; Precision={cls_precision:.2%}; "
        f"Recall={cls_recall:.2%}; F1={cls_f1:.2%}; "
        f"AUC={cls_auc:.2%}" if pd.notna(cls_auc) else
        f"Accuracy={cls_acc:.2%}; Precision={cls_precision:.2%}; "
        f"Recall={cls_recall:.2%}; F1={cls_f1:.2%}; AUC=unavailable",
    )
    add_check(
        "Classification",
        "False negatives",
        "PASS" if fn == 0 else "REVIEW",
        f"FN={fn}; FP={fp}; TP={tp}; TN={tn}",
    )
    add_check(
        "Classification",
        "Historical held-out reference retained",
        "PASS",
        f"Accuracy={REF_CLS['Accuracy']:.2%}; Precision={REF_CLS['Precision']:.2%}; "
        f"Recall={REF_CLS['Recall']:.2%}; F1={REF_CLS['F1']:.2%}; "
        f"AUC={REF_CLS['ROC_AUC']:.2%}",
    )

    cls = df.copy()
    cls["Display_ID"] = display_id
    cls["Actual_High_Cost"] = np.asarray(y_cls)
    cls["Predicted_High_Cost"] = cls_pred
    if cls_prob is not None:
        cls["High_Cost_Probability"] = cls_prob

    cls["Error_Type"] = np.select(
        [
            (cls["Actual_High_Cost"] == 0) & (cls["Predicted_High_Cost"] == 1),
            (cls["Actual_High_Cost"] == 1) & (cls["Predicted_High_Cost"] == 0),
            (cls["Actual_High_Cost"] == 1) & (cls["Predicted_High_Cost"] == 1),
        ],
        ["False Positive", "False Negative", "True Positive"],
        default="True Negative",
    )

    cls_cols_export = [
        "Display_ID",
        "Actual_High_Cost",
        "Predicted_High_Cost",
        "Error_Type",
    ]
    if cls_prob is not None:
        cls_cols_export.append("High_Cost_Probability")
    cls_cols_export.append("Claim_Amount")
    cls_cols_export += [c for c in SEGMENTS if c in cls.columns]

    cls[cls_cols_export].to_csv(
        OUT / "classification_all_predictions.csv", index=False
    )
    cls.loc[
        cls["Error_Type"] == "False Positive", cls_cols_export
    ].to_csv(OUT / "classification_false_positives.csv", index=False)
    cls.loc[
        cls["Error_Type"] == "False Negative", cls_cols_export
    ].to_csv(OUT / "classification_false_negatives.csv", index=False)

    cls_break = classification_segment_breakdown(cls)
    if not cls_break.empty:
        cls_break.to_csv(
            OUT / "classification_segment_breakdown.csv", index=False
        )

    # ---------------------------------------------------------------
    # 6. Explainability
    # ---------------------------------------------------------------
    sev_trans, sev_imp, sev_method = importance_tables(severity)
    cls_trans, cls_imp, cls_method = importance_tables(classifier)

    if sev_imp is not None:
        sev_trans.to_csv(
            OUT / "severity_transformed_feature_importance.csv", index=False
        )
        sev_imp.to_csv(
            OUT / "severity_source_feature_importance.csv", index=False
        )
        top5 = ", ".join(
            f"{r.Source_Feature} ({r.Importance_Percent:.2f}%)"
            for r in sev_imp.head(5).itertuples()
        )
        add_check(
            "Explainability",
            "Severity feature importance",
            "PASS",
            f"{sev_method}; Top 5: {top5}",
        )
    else:
        add_check(
            "Explainability",
            "Severity feature importance",
            "REVIEW",
            "Estimator exposes no supported built-in importance",
        )

    if cls_imp is not None:
        cls_trans.to_csv(
            OUT / "high_cost_transformed_feature_importance.csv", index=False
        )
        cls_imp.to_csv(
            OUT / "high_cost_source_feature_importance.csv", index=False
        )
        top5 = ", ".join(
            f"{r.Source_Feature} ({r.Importance_Percent:.2f}%)"
            for r in cls_imp.head(5).itertuples()
        )
        add_check(
            "Explainability",
            "High-cost feature importance",
            "PASS",
            f"{cls_method}; Top 5: {top5}",
        )
    else:
        add_check(
            "Explainability",
            "High-cost feature importance",
            "REVIEW",
            "Estimator exposes no supported built-in importance",
        )

    # ---------------------------------------------------------------
    # 7. Threshold analysis
    # ---------------------------------------------------------------
    thresholds = None
    if cls_prob is not None:
        thresholds = threshold_table(np.asarray(y_cls), cls_prob)
        thresholds.to_csv(
            OUT / "classification_threshold_analysis.csv", index=False
        )

        best_f1 = thresholds.loc[thresholds["F1"].idxmax()]
        current50 = thresholds.iloc[
            (thresholds["Threshold"] - 0.50).abs().argsort()[:1]
        ].iloc[0]

        safer = thresholds[
            (thresholds["FN"] <= current50["FN"])
            & (thresholds["Recall"] >= current50["Recall"])
        ].sort_values(
            ["FN", "F1", "Precision"],
            ascending=[True, False, False],
        )

        if not safer.empty:
            candidate = safer.iloc[0]
            detail = (
                f"0.50 => FN={int(current50.FN)}, FP={int(current50.FP)}, "
                f"Recall={current50.Recall:.2%}, F1={current50.F1:.2%}. "
                f"Best review candidate in tested range: {candidate.Threshold:.2f} "
                f"=> FN={int(candidate.FN)}, FP={int(candidate.FP)}, "
                f"Recall={candidate.Recall:.2%}, F1={candidate.F1:.2%}. "
                "Diagnostic only; no threshold was changed."
            )
        else:
            detail = (
                f"Best F1 threshold={best_f1.Threshold:.2f}; "
                f"F1={best_f1.F1:.2%}; no production threshold was changed."
            )

        add_check(
            "Threshold",
            "Probability threshold trade-off",
            "REVIEW",
            detail,
        )
    else:
        add_check(
            "Threshold",
            "Probability threshold trade-off",
            "REVIEW",
            "predict_proba unavailable; threshold testing skipped",
        )

    # ---------------------------------------------------------------
    # 8. Outlier audit
    # ---------------------------------------------------------------
    numeric_candidates = list(dict.fromkeys(
        reg_cols + cls_cols + ["Claim_Amount"]
    ))
    outliers = numeric_outlier_audit(df, numeric_candidates)
    if not outliers.empty:
        outliers.to_csv(OUT / "numeric_outlier_audit.csv", index=False)
        severe_outlier_cols = outliers[
            outliers["IQR_Outlier_Percent"] > 10
        ]
        add_check(
            "Robustness",
            "Numeric IQR outlier audit",
            "PASS" if severe_outlier_cols.empty else "REVIEW",
            "No numeric feature >10% IQR outliers"
            if severe_outlier_cols.empty
            else f"{len(severe_outlier_cols)} numeric columns exceed 10% IQR outliers",
        )

    # ---------------------------------------------------------------
    # 9. Leakage heuristic
    # ---------------------------------------------------------------
    all_model_features = sorted(set(reg_cols + cls_cols))
    suspicious = []
    for c in all_model_features:
        low = c.lower()
        if any(term in low for term in LEAKAGE_TERMS):
            suspicious.append(c)

    # Claim_Amount itself must never be an input feature.
    if "Claim_Amount" in all_model_features and "Claim_Amount" not in suspicious:
        suspicious.append("Claim_Amount")

    add_check(
        "Governance",
        "Target leakage heuristic",
        "PASS" if not suspicious else "WARNING",
        "No obvious post-outcome/target columns detected"
        if not suspicious
        else f"Review suspicious model inputs: {suspicious}",
    )

    # ---------------------------------------------------------------
    # 10. risk_engine.py consistency
    # ---------------------------------------------------------------
    risk_features = parse_risk_engine_features()
    if risk_features is None:
        add_check(
            "Architecture",
            "risk_engine HIGH_COST_FEATURES",
            "REVIEW",
            "Could not parse HIGH_COST_FEATURES from risk_engine.py",
        )
    else:
        expected_match = risk_features == EXPECTED_RISK_ENGINE_FEATURES
        subset_of_model = set(risk_features).issubset(set(cls_cols))
        status = "PASS" if expected_match and subset_of_model else "WARNING"
        detail = (
            f"risk_engine features={risk_features}; "
            f"production classifier expects {len(cls_cols)} features. "
            "The risk-engine list is a decision-support subset and must not be "
            "described as the complete production model schema."
        )
        add_check(
            "Architecture",
            "risk_engine vs production schema",
            status,
            detail,
        )

    # ---------------------------------------------------------------
    # 11. Generalization caution
    # ---------------------------------------------------------------
    reg_r2_gap = reg_r2 - REF_REG["R2"]
    cls_f1_gap = cls_f1 - REF_CLS["F1"]

    add_check(
        "Generalization",
        "Full-data vs documented held-out gap",
        "REVIEW",
        f"R2 gap={reg_r2_gap:+.4f}; F1 gap={cls_f1_gap:+.4f}. "
        "Higher full-dataset performance is expected if training observations are included. "
        "Do not present full-data diagnostics as independent test performance.",
    )

    # ---------------------------------------------------------------
    # 12. Charts
    # ---------------------------------------------------------------
    save_top_importance_chart(
        sev_imp,
        "Severity Model - Top 15 Source Features",
        "severity_top15_feature_importance.png",
    )
    save_top_importance_chart(
        cls_imp,
        "High-Cost Model - Top 15 Source Features",
        "high_cost_top15_feature_importance.png",
    )
    save_threshold_chart(thresholds)
    save_regression_error_chart(reg)

    # ---------------------------------------------------------------
    # 13. Consolidated summary
    # ---------------------------------------------------------------
    checklist = pd.DataFrame(checks)
    checklist.to_csv(OUT / "master_checklist.csv", index=False)

    status_counts = checklist["Status"].value_counts().to_dict()
    fail_count = int(status_counts.get("FAIL", 0))
    warning_count = int(status_counts.get("WARNING", 0))
    review_count = int(status_counts.get("REVIEW", 0))
    pass_count = int(status_counts.get("PASS", 0))

    if fail_count > 0:
        readiness = "NOT READY - resolve FAIL items before production use."
    elif warning_count > 0:
        readiness = (
            "CONDITIONALLY READY - production files remain unchanged; "
            "review WARNING items before any model/app implementation change."
        )
    else:
        readiness = (
            "READY TO FREEZE CURRENT MODEL/APPLICATION - REVIEW items are governance/"
            "diagnostic observations, not automatic blockers."
        )

    top_reg_segments = []
    if not reg_break.empty:
        for r in reg_break.sort_values("MAE", ascending=False).head(5).itertuples():
            top_reg_segments.append(
                f"{r.Segment}={r.Value}: MAE {r.MAE:,.2f} ({int(r.Rows)} rows)"
            )

    top_sev = []
    if sev_imp is not None:
        top_sev = [
            f"{r.Source_Feature} {r.Importance_Percent:.2f}%"
            for r in sev_imp.head(5).itertuples()
        ]

    top_cls = []
    if cls_imp is not None:
        top_cls = [
            f"{r.Source_Feature} {r.Importance_Percent:.2f}%"
            for r in cls_imp.head(5).itertuples()
        ]

    report = [
        "CLAIMGUARD AI - MASTER PRODUCTION VALIDATION REPORT",
        "=" * 78,
        "",
        "FINAL STATUS",
        readiness,
        "",
        "CHECK SUMMARY",
        f"PASS: {pass_count}",
        f"REVIEW: {review_count}",
        f"WARNING: {warning_count}",
        f"FAIL: {fail_count}",
        "",
        "DATA",
        f"Rows analyzed: {len(df):,}",
        f"Dataset columns: {df.shape[1]}",
        f"Exact duplicate rows: {duplicate_rows}",
        "",
        "PRODUCTION SCHEMA",
        f"Severity model features: {len(reg_cols)}",
        f"High-cost model features: {len(cls_cols)}",
        "Schema compatibility: PASS",
        "",
        "REGRESSION - CURRENT FULL DATASET DIAGNOSTIC",
        f"MAE: {reg_mae:,.2f}",
        f"RMSE: {reg_rmse:,.2f}",
        f"R2: {reg_r2:.4f}",
        f"Median absolute error: {reg['Absolute_Error'].median():,.2f}",
        f"90th percentile absolute error: {reg['Absolute_Error'].quantile(.90):,.2f}",
        f"95th percentile absolute error: {reg['Absolute_Error'].quantile(.95):,.2f}",
        "",
        "DOCUMENTED REGRESSION HELD-OUT REFERENCE",
        f"MAE: {REF_REG['MAE']:,.2f}",
        f"RMSE: {REF_REG['RMSE']:,.2f}",
        f"R2: {REF_REG['R2']:.4f}",
        "",
        "CLASSIFICATION - CURRENT FULL DATASET DIAGNOSTIC",
        f"Target: {target_source}"
        + (
            f" ({high_cost_threshold:,.2f})"
            if high_cost_threshold is not None else ""
        ),
        f"Accuracy: {cls_acc:.2%}",
        f"Precision: {cls_precision:.2%}",
        f"Recall: {cls_recall:.2%}",
        f"F1: {cls_f1:.2%}",
        f"ROC-AUC: {cls_auc:.2%}" if pd.notna(cls_auc) else "ROC-AUC: unavailable",
        f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}",
        "",
        "DOCUMENTED CLASSIFICATION HELD-OUT REFERENCE",
        f"Accuracy: {REF_CLS['Accuracy']:.2%}",
        f"Precision: {REF_CLS['Precision']:.2%}",
        f"Recall: {REF_CLS['Recall']:.2%}",
        f"F1: {REF_CLS['F1']:.2%}",
        f"ROC-AUC: {REF_CLS['ROC_AUC']:.2%}",
        "",
        "TOP SEVERITY MODEL FEATURES",
        *([f"- {x}" for x in top_sev] if top_sev else ["- unavailable"]),
        "",
        "TOP HIGH-COST MODEL FEATURES",
        *([f"- {x}" for x in top_cls] if top_cls else ["- unavailable"]),
        "",
        "HIGHEST REGRESSION ERROR SEGMENTS",
        *(
            [f"- {x}" for x in top_reg_segments]
            if top_reg_segments else ["- unavailable"]
        ),
        "",
        "GOVERNANCE CONCLUSION",
        "- Keep the documented held-out metrics as the official independent performance reference.",
        "- Treat current full-dataset metrics as diagnostics only.",
        "- Feature importance explains fitted-model reliance; it does not establish causality.",
        "- False negatives should receive special attention in human review workflows.",
        "- Threshold results are analysis only; this script does not alter the production threshold.",
        "- Do not reduce the production 44-feature schema solely from prior small-subset feature-selection tests.",
        "- Final claim decisions remain human-authorized.",
        "",
        "IMPLEMENTATION DECISION",
        "No automatic retraining, model replacement, threshold change, or app.py modification was performed.",
        "Only implement a change after reviewing the consolidated evidence and explicitly approving it.",
        "",
        f"Artifacts saved in: {OUT}",
    ]

    report_text = "\n".join(report)
    (OUT / "MASTER_VALIDATION_REPORT.txt").write_text(
        report_text, encoding="utf-8"
    )

    summary_json = {
        "readiness": readiness,
        "check_counts": {
            "PASS": pass_count,
            "REVIEW": review_count,
            "WARNING": warning_count,
            "FAIL": fail_count,
        },
        "rows": int(len(df)),
        "dataset_columns": int(df.shape[1]),
        "production_features": {
            "severity": len(reg_cols),
            "high_cost": len(cls_cols),
        },
        "regression_full_dataset": {
            "MAE": float(reg_mae),
            "RMSE": float(reg_rmse),
            "R2": float(reg_r2),
        },
        "regression_documented_heldout": REF_REG,
        "classification_full_dataset": {
            "Accuracy": float(cls_acc),
            "Precision": float(cls_precision),
            "Recall": float(cls_recall),
            "F1": float(cls_f1),
            "ROC_AUC": None if pd.isna(cls_auc) else float(cls_auc),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "classification_documented_heldout": REF_CLS,
        "classification_target_source": target_source,
        "classification_threshold": high_cost_threshold,
    }
    (OUT / "master_validation_summary.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8"
    )

    print(report_text)
    print("\n" + "-" * 78)
    print("MASTER CHECKLIST")
    print("-" * 78)
    print(checklist[["Area", "Check", "Status"]].to_string(index=False))

    print("\nGenerated artifacts:")
    for p in sorted(OUT.iterdir()):
        print(f"  - {p.name}")

    print("\nMASTER VALIDATION COMPLETE.")
    print("Production models and application files were NOT changed.")


if __name__ == "__main__":
    main()

# CLAIMGUARD AI

**Intelligent Insurance Claims Decision-Support Platform**

**Author: Nishanth S**

---

## Overview

CLAIMGUARD AI is an AI-assisted insurance claims decision-support platform built with Streamlit and machine learning. It combines claim severity prediction, high-cost claim risk, portfolio analytics, financial exposure, alerts, management actions, routing, escalation, and governance in one application.

The system is designed to help claims teams identify higher-risk cases earlier, prioritize human review, understand portfolio exposure, and route claims to the appropriate department.

> **Governance:** CLAIMGUARD AI does not automatically approve, reject, investigate, or settle insurance claims. Model outputs, risk scores, alerts, routing suggestions, and recommended actions are decision-support signals. Final decisions remain with authorized human personnel.

### End-to-End Flow

```text
Claim Data
    ↓
Machine Learning Prediction
    ↓
Risk Intelligence & Expected Loss
    ↓
Alerts & Priority
    ↓
Management Action
    ↓
Department Routing & Escalation
    ↓
Human Review / Authorization
```

---

## Business Problem

Insurance claim portfolios can contain hundreds or thousands of cases with different levels of financial exposure, incident severity, documentation quality, and operational urgency.

Manual screening alone can make it difficult to:

- identify potentially high-cost claims early,
- prioritize critical and high-risk cases,
- understand portfolio-level exposure,
- consistently route claims to the appropriate team,
- monitor alerts and management actions, and
- maintain a clear human-controlled decision process.

CLAIMGUARD AI brings these activities into a single decision-support workflow.

---

## Project Objectives

The project is designed to:

- predict claim severity,
- estimate high-cost claim probability,
- generate a transparent 0–100 operational risk score,
- classify claims into risk levels,
- estimate expected financial exposure,
- generate risk and operational alerts,
- create a prioritized management-action queue,
- support department routing and escalation,
- provide portfolio and executive analytics, and
- preserve human authorization for final claim decisions.

---

## Final Application Navigation

The production application is organized into five main navigation groups.

### Dashboard

- Executive Dashboard
- Operational Monitor
- Financial Overview
- Risk Analytics

### Claims

- Assessment Workspace
- Claim Review
- Risk Result

### Operations

- Alerts
- Action Queue
- Routing & Escalation

### Intelligence

- Risk Intelligence
- Portfolio Analytics
- Exposure

### Governance

- Model Performance
- Governance
- Policy Rules

---

## Core Application Capabilities

### Executive Dashboard

Provides a consolidated view of portfolio KPIs, claim volumes, risk distribution, financial exposure, expected loss, claim-type analysis, and exposure trends.

Dashboard filters operate on a common filtered dataset so that date, risk-level, and claim-type selections remain synchronized across the page.

### New Claim Assessment

Supports structured assessment using inputs such as:

- Claim Type
- Incident Severity
- Claim Amount
- Coverage Amount
- Reporting Delay
- Previous Claims
- Documents Required
- Documents Submitted
- Evidence Count

The assessment provides decision-support outputs including Risk Score, High-Cost Probability, Priority, Department, and recommended documentation.

Document and evidence recommendations can update according to claim type and incident severity while remaining editable by the reviewer.

### Risk Intelligence

Explains the operational risk framework, risk bands, contributing signals, and portfolio risk distribution.

### Portfolio Analytics

Provides portfolio-level analysis across risk, expected loss, claim type, policy type, region, processing characteristics, exposure, and management priorities.

### Alerts

The alert engine supports the following alert categories:

- Critical Risk
- High Risk
- High Expected Loss
- Elevated Expected Loss
- High-Cost Claim
- Unusual Claim
- High Anomaly Score
- Processing Delay

Alerts are categorized into Critical, High, Medium, or Review priority levels.

### Action Queue

Alerts are converted into a human worklist using:

- **P1** — Critical / immediate attention
- **P2** — High priority
- **P3** — Medium / monitoring
- **P4** — Review

The action queue provides recommended actions and next steps without making the final claim decision.

### Routing & Escalation

Claims can be routed to operational departments including:

- Motor Claims
- Property Claims
- Health Claims
- Travel Claims
- Life Claims
- Marine Claims
- General Claims

Routing supports Department, Priority, and Risk filters together with Claim ID search and claim-level drill-down.

Routing approval represents **administrative routing approval only**. It is not approval of the insurance claim.

After routing approval, the application can prepare an email with editable **To**, **CC**, **Subject**, and **Body** fields. The user can open the message in the computer's default email application using a standard `mailto:` workflow and manually send it.

No SMTP server, API-based automatic email delivery, or hard-coded recipient email address is required by the final workflow.

---

## Machine Learning Models

CLAIMGUARD AI uses two production machine-learning models:

1. **Claim Severity Model** — Random Forest Regressor
2. **High-Cost Claim Model** — Random Forest Classifier

The production model files are:

```text
models/
├── claim_severity_model.pkl
└── high_cost_claim_model.pkl
```

Final production schema validation confirmed that both models expect **44 input features**, and all required features are available in the current processed dataset.

---

## Claim Severity Prediction

Three regression approaches were evaluated.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge Regression | ₹217,904.60 | ₹292,348.90 | 0.6222 |
| Random Forest | ₹143,067.30 | ₹219,593.00 | **0.7869** |
| Gradient Boosting | **₹137,729.20** | ₹222,524.60 | 0.7811 |

**Selected production model: Random Forest**

Random Forest achieved the highest R² and lowest RMSE among the evaluated regression models, providing the strongest overall balance of performance.

### Official Held-Out Regression Reference

- **MAE:** ₹143,067.30
- **RMSE:** ₹219,593.00
- **R²:** 0.7869

These documented held-out results are retained as the official independent performance reference.

---

## High-Cost Claim Classification

The high-cost target is based on the 90th percentile of `Claim_Amount`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 88.00% | 44.12% | 75.00% | 55.56% | 93.97% |
| Decision Tree | 92.50% | 61.90% | 65.00% | 63.41% | 82.39% |
| Random Forest | **93.50%** | **66.67%** | **70.00%** | **68.29%** | **93.61%** |

**Selected production model: Random Forest**

### Official Held-Out Classification Reference

- **Accuracy:** 93.50%
- **Precision:** 66.67%
- **Recall:** 70.00%
- **F1 Score:** 68.29%
- **ROC-AUC:** 93.61%

The held-out metrics above remain the official independent performance reference.

---

## Production Model Schema

Final validation identified **44 production input features** for both production models:

```text
Customer_Age
Gender
Employment_Type
Income_Band
Customer_Region
Customer_Tenure_Years
Previous_Claims_Count
Policy_Type
Policy_Age_Years
Annual_Premium
Coverage_Amount
Deductible
No_Claim_Bonus
Add_On_Coverage
Policy_Channel
Claim_Type
Incident_Type
Incident_Severity
Claim_Channel
Reporting_Delay_Days
Incident_Location
Urban_Rural
Weather_Condition
Road_Condition
Incident_Time
Incident_Cause
Third_Party_Involved
Police_Report
Injury_Reported
Emergency_Service_Required
Asset_Type
Asset_Age_Years
Asset_Value
Asset_Usage
Previous_Accidents
Damage_Level
Document_Completeness
Evidence_Count
Claim_Frequency
Previous_Claim_Ratio
Premium_Coverage_Ratio
Customer_Claim_Intensity
Reporting_Risk_Score
Incident_Risk_Score
```

The production schema should not be reduced solely on the basis of the earlier small-subset feature-selection experiments.

---

## Risk Engine

The application also contains a transparent operational risk-scoring layer.

The six primary decision-support signals used by this layer are:

```text
Coverage_Amount
Incident_Risk_Score
Reporting_Delay_Days
Customer_Claim_Intensity
Document_Completeness
Evidence_Count
```

These six signals are **not the complete production model schema**. The trained production models use the validated 44-feature schema described above.

### 0–100 Risk Score

| Component | Weight |
|---|---:|
| High-Cost Probability | 30 |
| Claim Amount vs Coverage | 25 |
| Incident Risk | 20 |
| Reporting Risk | 10 |
| Customer Claim Intensity | 10 |
| Documentation / Evidence | 5 |

### Risk Levels

| Risk Score | Risk Level |
|---|---|
| 75–100 | Critical |
| 50–74.9 | High |
| 25–49.9 | Medium |
| Below 25 | Low |

This score is an operational prioritization mechanism and does not replace the trained model predictions or human judgment.

---

## Production Explainability

Built-in Random Forest feature importance was analyzed for both production models.

### Claim Severity — Leading Features

| Feature | Importance |
|---|---:|
| Asset_Value | 29.94% |
| Incident_Risk_Score | 28.87% |
| Coverage_Amount | 6.62% |
| Incident_Severity | 5.03% |
| Damage_Level | 4.20% |

The two leading features account for approximately **58.81%** of fitted-model importance, while the top five account for approximately **74.66%**.

### High-Cost Classification — Leading Features

| Feature | Importance |
|---|---:|
| Asset_Value | 16.15% |
| Incident_Risk_Score | 13.88% |
| Incident_Severity | 13.59% |
| Damage_Level | 11.43% |
| Coverage_Amount | 5.41% |

The top four account for approximately **55.05%** of fitted-model importance.

> Feature importance describes how the fitted model relies on available variables. It does not establish causal relationships.

---

## Master Production Validation

A consolidated read-only validation suite is included:

```text
claimguard_master_validation.py
```

The validation checks:

- required production files,
- processed dataset integrity,
- duplicate rows,
- target availability,
- missingness,
- document-completeness data,
- model schema compatibility,
- regression diagnostics,
- classification diagnostics,
- false positives and false negatives,
- feature importance,
- probability-threshold trade-offs,
- numerical outliers,
- target-leakage heuristics,
- risk-engine / production-schema architecture,
- segment-level errors, and
- generalization cautions.

### Final Validation Status

```text
PASS:     21
REVIEW:    3
WARNING:   0
FAIL:      0

FINAL STATUS:
READY TO FREEZE CURRENT MODEL/APPLICATION
```

The three REVIEW items are diagnostic/governance observations:

- False negatives
- Probability threshold trade-off
- Full-data vs documented held-out performance gap

They are not automatic production blockers.

---

## Full-Dataset Production Diagnostics

The master validation also ran the current fitted models across the available processed dataset.

### Regression Diagnostic

- MAE: ₹76,973.14
- RMSE: ₹134,815.22
- R²: 0.9330
- Median Absolute Error: ₹39,531.79
- 90th Percentile Absolute Error: ₹185,416.98
- 95th Percentile Absolute Error: ₹294,487.56

### Classification Diagnostic

- Accuracy: 97.40%
- Precision: 82.46%
- Recall: 94.00%
- F1: 87.85%
- ROC-AUC: 98.97%
- TN: 880
- FP: 20
- FN: 6
- TP: 94

> **Important:** These are current full-dataset diagnostics and are **not** presented as independent held-out performance. The documented held-out metrics remain the official model-performance reference.

False negatives receive particular attention because missing a genuinely high-cost case can be operationally more important than sending an additional case for human review.

---

## Error & Robustness Analysis

The largest regression errors were observed in segments including:

| Segment | Rows | MAE |
|---|---:|---:|
| Damage Level = Severe | 42 | ₹166,733.02 |
| Incident Severity = Critical | 51 | ₹163,799.49 |
| Incident Severity = High | 203 | ₹121,062.52 |
| Damage Level = Major | 197 | ₹112,857.57 |
| Incident Location = Chennai | 108 | ₹94,438.92 |

These are **review signals**, not evidence that a segment causes prediction error.

Critical/severe claims and cases with incomplete documentation deserve additional human attention because of their operational importance and observed error characteristics.

---

## Feature-Selection Decision

Multiple feature-selection experiments were conducted separately from the stable production application.

The reduced-feature experiments did **not** outperform the documented production models under the evaluated conditions. Therefore:

- the production models were not replaced,
- the 44-feature production schema was retained,
- no feature-selection test model was promoted to production, and
- the stable application remained unchanged.

Testing artifacts are retained separately under `archive_testing/` for traceability.

---

## Final Project Structure

```text
Insurance_Claim_Intelligence/
│
├── app.py
├── risk_engine.py
├── alerts_engine.py
├── portfolio_analytics.py
├── management_actions.py
├── executive_decision_center.py
├── notification_engine.py
│
├── claimguard_master_validation.py
├── Insurance_Claim_Intelligence_Complete.ipynb
├── Insurance_Claim_Intelligence_Presentation.pptx
├── README.md
├── requirements.txt
│
├── models/
│   ├── claim_severity_model.pkl
│   └── high_cost_claim_model.pkl
│
├── data/
│   ├── raw/
│   │   └── insurance_claims_raw.csv
│   └── processed/
│       ├── insurance_claims_anomaly.csv
│       ├── insurance_claims_cleaned.csv
│       ├── insurance_claims_communication.csv
│       ├── insurance_claims_expected_loss.csv
│       ├── insurance_claims_features.csv
│       ├── insurance_claims_feedback.csv
│       └── insurance_claims_scenario_actions.csv
│
├── documentation/
│   ├── eda_charts/
│   ├── feedback_analytics/
│   ├── shap_charts/
│   └── model / business-analysis artifacts
│
├── claimguard_master_validation_results/
│   └── consolidated validation artifacts
│
└── archive_testing/
    ├── backups/
    ├── scripts/
    └── results/
```

`archive_testing/` contains historical experiments, test-only models, old backups, and intermediate validation outputs. These files are retained for traceability and are not production model dependencies.

---

## Data

The current master validation analyzed:

- **1,000 claim records**
- **67 dataset columns**
- **0 exact duplicate rows**

Processed datasets are stored under:

```text
data/processed/
```

The main feature dataset used by production validation is:

```text
data/processed/insurance_claims_features.csv
```

---

## Technology Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- joblib
- Jupyter Notebook

---

## Installation

Open Command Prompt and move to the project directory:

```cmd
cd /d C:\Insurance_Claim_Intelligence
```

Create and activate a virtual environment if required:

```cmd
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```cmd
pip install -r requirements.txt
```

---

## Run the Application

From the project directory:

```cmd
python -m streamlit run app.py
```

Using `python -m streamlit` is recommended for environments where direct execution of `streamlit.exe` is restricted.

---

## Run Master Validation

The master validation is read-only and does not retrain or replace production models.

```cmd
cd /d C:\Insurance_Claim_Intelligence
python claimguard_master_validation.py
```

Results are written to:

```text
claimguard_master_validation_results/
```

Important outputs include:

```text
MASTER_VALIDATION_REPORT.txt
master_checklist.csv
master_validation_summary.json
production_model_schema.csv
classification_threshold_analysis.csv
classification_false_negatives.csv
classification_false_positives.csv
regression_segment_breakdown.csv
severity_source_feature_importance.csv
high_cost_source_feature_importance.csv
```

---

## Model Governance

CLAIMGUARD AI follows these project-level governance principles:

**Human authorization**  
Final claim decisions remain with authorized personnel.

**Decision support, not autonomous adjudication**  
Predictions, risk levels, alerts, recommended actions, and routing suggestions are advisory.

**Independent performance reporting**  
Documented held-out model results remain the official performance reference.

**Diagnostic transparency**  
Full-dataset diagnostics are clearly separated from held-out performance.

**False-negative awareness**  
Potentially missed high-cost claims receive special attention in review workflows.

**Explainability caution**  
Feature importance is used to understand fitted-model reliance, not to claim causality.

**Threshold governance**  
Probability-threshold analysis is treated as decision-support analysis. Threshold changes are not automatically applied.

**Model-change control**  
Experimental feature-selection models do not replace production models without evidence and explicit approval.

---

## Policy Rules

Policy Rules provide practical reviewer checkpoints rather than automatic claim decisions. Examples include verifying required documentation, coverage context, incident details, reporting information, supporting evidence, and claim-type-specific requirements.

The rules support consistent review while preserving reviewer authority.

---

## Limitations

- The regression and classification models retain meaningful prediction error and should be used as screening and prioritization aids.
- The 0–100 risk-score weights are predefined operational weights rather than continuously learned weights.
- Full-dataset diagnostics may include observations seen during model fitting and therefore must not be interpreted as independent validation.
- Feature importance does not imply causality.
- High-severity and high-damage cases show larger regression errors in the current diagnostic analysis.
- The final notification workflow prepares an email for human review and manual sending; it is not an automated messaging infrastructure.
- Organizational deployment would require appropriate security, access control, audit logging, privacy controls, and model-monitoring procedures.

---

## Future Enhancements

Potential future enhancements include:

- role-based access control,
- production audit logging,
- model drift monitoring,
- controlled retraining pipelines,
- calibrated threshold governance,
- integration with an authorized claims-management system,
- additional claim-level explainability,
- enterprise notification integration,
- stronger data-quality monitoring, and
- external independent validation on newly collected claims.

These are future enhancements and are not required for the current validated project version.

---

## Interview / Presentation Summary

### What is CLAIMGUARD AI?

CLAIMGUARD AI is an AI-assisted insurance claims decision-support platform that combines machine-learning predictions with risk scoring, expected-loss analysis, alerts, management actions, routing, portfolio intelligence, and human-controlled escalation.

### Which models are used?

Two Random Forest production models are used:

- a Random Forest Regressor for claim severity,
- a Random Forest Classifier for high-cost claim risk.

### What is the official performance?

The documented held-out reference is:

- **Severity:** R² = 0.7869, MAE = ₹143,067.30, RMSE = ₹219,593.00
- **High-Cost Classification:** Accuracy = 93.50%, F1 = 68.29%, ROC-AUC = 93.61%

### Why are the full-dataset scores higher?

The full-dataset results are diagnostic results from the current fitted models on the available processed dataset. They are not treated as an independent test set. Therefore, the project retains the documented held-out results as the official performance reference.

### How many features do the production models use?

Both validated production models expect **44 features**.

The six features highlighted in the risk engine are operational decision-support signals and should not be confused with the complete model input schema.

### Does the AI approve or reject claims?

No.

CLAIMGUARD AI provides predictions, risk signals, alerts, priorities, routing recommendations, and supporting information. Final claim decisions remain human-authorized.

### What did the final validation show?

The consolidated master validation produced:

```text
21 PASS
3 REVIEW
0 WARNING
0 FAIL
```

Final status:

**READY TO FREEZE CURRENT MODEL/APPLICATION**

---

## Project Status

**Model development:** Complete  
**Feature-selection evaluation:** Complete  
**Production schema validation:** Complete  
**Explainability analysis:** Complete  
**Error & robustness analysis:** Complete  
**Master validation:** Complete  
**Application:** Stable / Frozen  
**Production warnings:** 0  
**Production failures:** 0  

---

## Author

**Nishanth S**

*Insurance Claim Intelligence / CLAIMGUARD AI*

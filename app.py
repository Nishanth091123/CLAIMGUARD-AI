"""
app.py
Insurance Claim Intelligence — AI-Assisted Claim Risk & Management Decision Support
Author: Nishanth S

Run locally with:
    streamlit run app.py

GOVERNANCE:
This application provides AI-assisted DECISION SUPPORT only. It does not
automatically approve, reject, investigate, or settle insurance claims.
Final decisions remain with authorized human personnel.
"""

import os
import glob
import pandas as pd
import altair as alt
import streamlit as st

import risk_engine as re
import portfolio_analytics as pa
import alerts_engine as ae
import management_actions as ma
import executive_decision_center as edc
import notification_engine as ne

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="CLAIMGUARD AI",
    page_icon="🛡️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Data / model loading (cached, with graceful error handling)
# --------------------------------------------------------------------------

# Files that hold post-outcome / cleanup variants of the same claims are
# loaded last (and only contribute columns the earlier files don't already
# have), so a raw/duplicate export never silently overwrites a cleaner one.
LOW_PRIORITY_KEYWORDS = ["cleaned", "raw", "backup", "old"]


@st.cache_data(show_spinner=False)
def load_claims_data():
    """Auto-discover and merge every CSV in data/processed/.

    Every *.csv found in data/processed/ is loaded. Files are merged on a
    shared ID column (Claim_ID, or the first column ending in _ID / _Id
    common to two files) so the app works whether the pipeline produced 3
    files or 7 (communication, feedback, anomaly, features, expected_loss,
    scenario_actions, cleaned, ...). Only new columns are added on each
    merge, so duplicate/derived exports never overwrite existing data.
    Returns (df, warnings, files_loaded).
    """
    warnings = []
    csv_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

    if not csv_paths:
        warnings.append(f"No CSV files found in {os.path.relpath(DATA_DIR, BASE_DIR)}\\")
        return pd.DataFrame(), warnings, []

    # Load every file; note which ones actually parsed.
    loaded = []
    for fpath in csv_paths:
        fname = os.path.basename(fpath)
        try:
            frame = pd.read_csv(fpath)
            loaded.append((fname, frame))
        except Exception as exc:
            warnings.append(f"Could not read {fname}: {exc}")

    if not loaded:
        return pd.DataFrame(), warnings, []

    # Prefer a file with "communication" or "feature" in its name as the
    # base (widest claim-level table); otherwise the largest file by column
    # count. Low-priority (cleaned/raw/backup) files merge last.
    def sort_key(item):
        fname, frame = item
        lname = fname.lower()
        is_low_priority = any(k in lname for k in LOW_PRIORITY_KEYWORDS)
        is_preferred_base = ("communication" in lname) or ("feature" in lname)
        return (is_low_priority, not is_preferred_base, -len(frame.columns))

    loaded.sort(key=sort_key)

    base_name, df = loaded[0]
    files_loaded = [base_name]

    def find_id_column(frame):
        for candidate in ["Claim_ID", "ClaimID", "claim_id"]:
            if candidate in frame.columns:
                return candidate
        for col in frame.columns:
            if col.lower().endswith("_id"):
                return col
        return None

    id_col = find_id_column(df)

    for fname, extra in loaded[1:]:
        extra_id_col = find_id_column(extra)
        if id_col and extra_id_col and extra_id_col in extra.columns:
            if extra_id_col != id_col:
                extra = extra.rename(columns={extra_id_col: id_col})
            new_cols = [c for c in extra.columns if c == id_col or c not in df.columns]
            if len(new_cols) > 1:  # more than just the ID column
                df = df.merge(extra[new_cols].drop_duplicates(subset=id_col), on=id_col, how="left")
                files_loaded.append(fname)
            else:
                warnings.append(f"{fname}: no new columns beyond the shared ID — skipped")
        else:
            warnings.append(f"{fname}: no shared ID column with the base table — skipped")

    return df, warnings, files_loaded


@st.cache_resource(show_spinner=False)
def load_models_cached():
    return re.load_models()


@st.cache_data(show_spinner=False)
def score_portfolio(df: pd.DataFrame, _models_key: tuple):
    return re.score_claims_dataframe(df, load_models_cached())


def currency(x) -> str:
    try:
        return f"\u20B9{x:,.0f}"
    except Exception:
        return "\u20B90"


# --------------------------------------------------------------------------
# CLAIMGUARD AI — reference-inspired enterprise shell
# --------------------------------------------------------------------------
st.markdown("""
<style>
:root{--navy:#142640;--blue:#1769d2;--bg:#eef5fb;--line:#dce7f2;--green:#07996b;--amber:#f0ad1e;--red:#c9272c}
[data-testid="stAppViewContainer"]{background:var(--bg)} [data-testid="stSidebar"]{background:#f7fbff;border-right:1px solid var(--line)}
[data-testid="stSidebar"] .block-container{padding-top:1rem}.block-container{padding-top:3.1rem;max-width:1450px} h1,h2,h3{color:var(--navy)}
.cg-brand{font-weight:800;font-size:18px;color:#142640;margin:6px 0}.cg-brand i{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:9px;background:linear-gradient(135deg,#123f73,#1769d2);color:white;font-style:normal;margin-right:8px;font-size:9px;font-weight:900;letter-spacing:.04em;box-shadow:0 2px 7px rgba(23,105,210,.18)}.cg-sub{font-size:11px;color:#7d8ca0;margin:0 0 17px 34px}.cg-sec{font-size:9px;font-weight:800;color:#8997a9;letter-spacing:.12em;margin:14px 0 5px}
.cg-top{background:#fff;border:1px solid var(--line);border-radius:11px;padding:9px 13px;margin-bottom:4px}.cg-bread{font-size:11px;color:#687a91}.cg-bread b{color:#1769d2}.cg-ready{float:right;font-size:10px;font-weight:700;color:#07895f}.cg-tabs{display:flex;gap:10px;border-bottom:1px solid #dfe9f3;margin-bottom:10px}.cg-tab{font-size:11px;font-weight:700;color:#718197;padding:9px 12px;border-radius:8px 8px 0 0}.cg-tab.on{color:#fff;background:#1769d2}
.cg-kpi{background:white;border:1px solid var(--line);border-radius:12px;padding:11px 12px;min-height:78px}.cg-kl{font-size:9px;color:#7b8b9f;font-weight:700;text-transform:uppercase}.cg-kv{font-size:21px;font-weight:800;color:#142640;margin-top:4px}.cg-kn{font-size:9px;color:#8b98a8}.cg-panel{background:white;border:1px solid var(--line);border-radius:13px;padding:13px 14px;margin-bottom:10px}.cg-panel:empty{display:none!important}.cg-pt{font-size:12px;font-weight:800;color:#243850}.cg-ps{font-size:9px;color:#8997a8;margin-bottom:7px}.cg-row{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #edf2f7;padding:8px 0}.cg-row:last-child{border:0}.cg-rt{font-size:11px;font-weight:700;color:#2c405a}.cg-rs{font-size:9px;color:#8492a4}.pill{border-radius:999px;padding:3px 8px;font-size:9px;font-weight:800;background:#edf4ff;color:#1769d2}.red{background:#fff0f1;color:#c9272c}.amber{background:#fff7df;color:#9a6800}.green{background:#eaf8f2;color:#07895f}.cg-foot{text-align:center;font-size:9px;color:#8290a2;padding:8px}.cg-statgrid{display:grid;grid-template-columns:repeat(2,1fr);gap:9px}.cg-stat{background:#f8fbff;border:1px solid #e3edf6;border-radius:10px;padding:10px}.cg-stat b{font-size:17px;color:#17304f}.cg-stat span{display:block;font-size:9px;color:#8090a3;margin-top:2px}.cg-route{background:#fff;border:1px solid #dce7f2;border-radius:10px;padding:11px}.cg-route b{color:#17304f;font-size:12px}.cg-route small{display:block;color:#8290a2;margin-top:3px}.cg-model{background:#fff;border:1px solid #dce7f2;border-radius:12px;padding:13px;min-height:110px}.cg-model.sel{border:2px solid #1769d2;background:#f6faff}.cg-badge{font-size:8px;font-weight:800;background:#e8f2ff;color:#1769d2;padding:3px 7px;border-radius:999px;float:right}
.stRadio>label{display:none}
[data-testid="stMain"] div[data-testid="stRadio"] div[role="radiogroup"]{display:flex;gap:0;border:1px solid #cfe0f1;border-radius:9px;width:max-content;max-width:100%;overflow:hidden;padding:0;margin:7px 0 16px;background:#f4f8fd;flex-wrap:nowrap}
[data-testid="stMain"] div[data-testid="stRadio"] div[role="radiogroup"] label{background:#f4f8fd;border:0;border-right:1px solid #cfe0f1;border-radius:0;padding:8px 18px;margin:0;min-height:38px}
[data-testid="stMain"] div[data-testid="stRadio"] div[role="radiogroup"] label:last-child{border-right:0}
[data-testid="stMain"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover{background:#e9f3ff}
[data-testid="stMain"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked){background:#1769d2!important}
[data-testid="stMain"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p{color:#fff!important}
[data-testid="stMain"] div[data-testid="stRadio"] div[role="radiogroup"] label p{font-size:11px!important;font-weight:750!important;color:#314a68!important;white-space:nowrap}
[data-testid="stMain"] div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child{display:none!important}
[data-testid="stMain"] div[data-testid="stRadio"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"]{display:block!important;visibility:visible!important;opacity:1!important;width:auto!important;height:auto!important}
[data-testid="stMain"] div[data-testid="stRadio"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p{display:block!important;visibility:visible!important;opacity:1!important;margin:0!important;line-height:20px!important}
[data-testid="stMain"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] p{color:#fff!important}
[data-testid="stMain"] div[data-testid="stRadio"] div[role="radiogroup"] label:not(:has(input:checked)) [data-testid="stMarkdownContainer"] p{color:#314a68!important}
[data-testid="stVegaLiteChart"],div[data-testid="stDataFrame"]{background:#fff;border:1px solid var(--line);border-radius:12px;padding:7px;overflow:hidden}
[data-testid="stMain"] .stSelectbox>div>div,[data-testid="stMain"] .stTextInput>div>div,[data-testid="stMain"] .stTextArea textarea{background:#fff;border-color:#d8e5f1} div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px;overflow:hidden}
.mx-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.mx-cell{border-radius:11px;padding:13px 14px;border:1px solid #e1eaf3;min-height:112px}.mx-red{background:#fff0f1;border-color:#ffd2d5}.mx-amber{background:#fff7e4;border-color:#ffe2a0}.mx-green{background:#ecfaf4;border-color:#c7efdf}.mx-blue{background:#edf5ff;border-color:#d0e3ff}.mx-title{font-size:10px;font-weight:800;color:#31465f}.mx-num{font-size:25px;font-weight:850;color:#142640;margin:4px 0}.mx-meta{font-size:9px;color:#6f8095}.mx-action{font-size:9px;font-weight:800;margin-top:7px;color:#40546c;text-transform:uppercase;letter-spacing:.04em}.mx-note{font-size:8px;color:#8a98aa;margin-top:7px}.dept-head,.dept-row{display:grid;grid-template-columns:1.45fr .45fr .7fr 1fr .35fr;gap:8px;align-items:center}.dept-head{font-size:8px;font-weight:800;color:#8795a6;padding:5px 7px;border-bottom:1px solid #e9eff5}.dept-row{font-size:10px;padding:9px 7px;border-bottom:1px solid #edf2f7}.dept-name{font-weight:750;color:#263b55}.dept-count,.dept-exp,.dept-pct{color:#52677f}.dept-track{height:7px;background:#edf2f7;border-radius:99px;overflow:hidden}.dept-track span{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#1769d2,#37b7c8)}</style>""",unsafe_allow_html=True)

def short_money(v):
    v=float(v or 0)
    if abs(v)>=1e7:return f"₹{v/1e7:.1f}Cr"
    if abs(v)>=1e5:return f"₹{v/1e5:.1f}L"
    return f"₹{v:,.0f}"
def card(label,value,note=""):
    st.markdown(f'<div class="cg-kpi"><div class="cg-kl">{label}</div><div class="cg-kv">{value}</div><div class="cg-kn">{note}</div></div>',unsafe_allow_html=True)
def phead(title,sub=""):
    st.markdown(f'<div class="cg-pt">{title}</div><div class="cg-ps">{sub}</div>',unsafe_allow_html=True)



# --------------------------------------------------------------------------
# PHASE 1 — interactive workflow helpers
# --------------------------------------------------------------------------
RISK_COLORS = {"Critical":"#ff3b30", "High":"#2196f3", "Medium":"#ff9f1c", "Low":"#12a66a"}
CLAIM_TYPE_COLORS = ["#1597e5", "#ff9f1c", "#ef3aa8", "#ff4b3e", "#18ad69", "#7a55d9", "#20b8b0", "#e6b422"]
SUPPORTED_DEPARTMENTS = ["Motor Claims","Property Claims","Health Claims","Travel Claims","Life Claims","Marine Claims","General Claims"]
SUPPORTED_PRIORITIES = ["P1","P2","P3","P4"]
SUPPORTED_RISKS = ["Critical","High","Medium","Low"]

def document_profile(claim_type, severity):
    """Business-rule defaults only. Reviewers may edit all values before assessment."""
    ct=str(claim_type).lower(); sev=str(severity).lower()
    base = 5
    docs = ["Claim Form", "ID Proof", "Policy Document"]
    if any(x in ct for x in ["health","medical"]):
        base=6; docs += ["Medical Report","Hospital Bills","Diagnosis / Discharge Summary"]
    elif any(x in ct for x in ["fire","property","natural disaster","flood","building"]):
        base=7; docs += ["Damage Photographs","Property Estimate","Incident / Authority Report","Ownership Proof"]
    elif any(x in ct for x in ["accidental","third party","motor","vehicle","theft"]):
        base=6; docs += ["Vehicle RC","Driving Licence","Damage Photographs"]
        docs += ["Police / FIR Report"] if "theft" in ct or "third party" in ct else ["Repair Estimate"]
    elif "travel" in ct:
        base=5; docs += ["Travel Ticket","Delay / Loss Confirmation"]
    elif "life" in ct:
        base=7; docs += ["Medical / Death Certificate","Nominee Proof","Bank Details","Supporting Certificate"]
    elif "marine" in ct:
        base=7; docs += ["Bill of Lading","Survey Report","Cargo Invoice","Damage Evidence"]
    sev_add={"low":0,"moderate":1,"high":2,"severe":3}.get(sev,1)
    required=min(20,base+sev_add)
    evidence=min(10,max(2,2+sev_add+(1 if required>=8 else 0)))
    # Actual submitted docs remain editable; start slightly below requirement for realistic review.
    submitted=max(0,required-(2 if sev in ["high","severe"] else 1))
    return required, submitted, evidence, list(dict.fromkeys(docs))

def routing_mail_preview(claim_id, dept, priority, risk_level, risk_score, expected_loss):
    subject=f"{priority} Claim Review | {claim_id} | {dept}"
    body=(f"Claim ID: {claim_id}\nDepartment: {dept}\nPriority: {priority}\nRisk Level: {risk_level}\n"
          f"Risk Score: {float(risk_score):.1f}/100\nExpected Loss: {currency(expected_loss)}\n\n"
          "Routing has been reviewed by an authorized user. Please complete the required human claim review.")
    return subject, body
def route_department(row):
    """Derive department from the current claim's real insurance fields.

    Explicit line-of-business / policy fields are checked first.  We then
    inspect claim/incident text.  No synthetic department is assigned merely
    to populate a filter: if the source data does not support a specialist
    line, the claim remains General Claims.
    """
    groups = [
        (["health", "medical", "hospital", "surgery", "illness", "mediclaim"], "Health Claims"),
        (["travel", "trip", "baggage", "flight", "journey"], "Travel Claims"),
        (["life", "death", "term life", "mortality", "nominee"], "Life Claims"),
        (["marine", "cargo", "shipping", "vessel", "transit"], "Marine Claims"),
        (["property", "home", "building", "fire", "flood", "storm", "natural disaster", "earthquake"], "Property Claims"),
        (["motor", "vehicle", "auto", "car", "bike", "collision", "third party", "accidental damage", "vehicle theft"], "Motor Claims"),
    ]

    # Strongest evidence: an explicit insurance/product/LOB field.
    explicit_cols = ["Insurance_Type", "Policy_Type", "Line_of_Business", "Product_Type", "Insurance_Product", "Policy_Category"]
    explicit_text = " | ".join(
        str(row.get(c, "")).strip().lower()
        for c in explicit_cols
        if c in row.index and pd.notna(row.get(c, "")) and str(row.get(c, "")).strip()
    )
    for keys, dept in groups:
        if any(k in explicit_text for k in keys):
            return dept

    # Secondary evidence: claim/incident description.
    detail_cols = ["Claim_Type", "ClaimType", "Incident_Type", "Incident_Description", "Loss_Type", "Cause_of_Loss"]
    detail_text = " | ".join(
        str(row.get(c, "")).strip().lower()
        for c in detail_cols
        if c in row.index and pd.notna(row.get(c, "")) and str(row.get(c, "")).strip()
    )
    for keys, dept in groups:
        if any(k in detail_text for k in keys):
            return dept
    return "General Claims"

def recommend_priority(row):
    risk=float(pd.to_numeric(row.get("Risk_Score",0), errors="coerce") or 0)
    prob=float(pd.to_numeric(row.get("High_Cost_Probability",0), errors="coerce") or 0)
    loss=float(pd.to_numeric(row.get("Expected_Loss",0), errors="coerce") or 0)
    level=str(row.get("Risk_Level","")).lower()
    if level=="critical" or risk>=75 or (prob>=.80 and risk>=60): return "P1"
    if level=="high" or risk>=50 or prob>=.60: return "P2"
    if level=="medium" or risk>=25 or loss>0: return "P3"
    return "P4"

def set_drilldown(title, frame, scope="global"):
    st.session_state["drill_title"] = title
    st.session_state["drill_ids"] = frame["Claim_ID"].astype(str).tolist() if "Claim_ID" in frame.columns else []
    st.session_state["drill_scope"] = scope

def clear_drilldown():
    for k in ["drill_title", "drill_ids", "drill_scope"]:
        st.session_state.pop(k, None)

def drill_button(label, frame, key, scope="global"):
    if st.button(label, key=key, use_container_width=True):
        set_drilldown(label, frame, scope)
        st.rerun()

def show_drilldown(df, key_prefix, scope="global"):
    if st.session_state.get("drill_scope") != scope:
        return
    ids=st.session_state.get("drill_ids", [])
    if not ids or "Claim_ID" not in df.columns: return
    part=df[df["Claim_ID"].astype(str).isin(ids)].copy()
    if part.empty: return
    st.markdown('<div class="cg-panel">', unsafe_allow_html=True)
    phead(st.session_state.get("drill_title","Claim Drill-down"), f"{len(part):,} matching claims — select a claim to inspect")
    useful=[c for c in ["Claim_ID","Claim_Type","Policy_Type","Claim_Amount","Risk_Score","Risk_Level","High_Cost_Probability","Expected_Loss","Recommended_Priority","Routed_Department"] if c in part.columns]
    st.dataframe(part[useful], use_container_width=True, hide_index=True, height=min(360, 42+35*min(len(part),9)))
    drill_search=st.text_input("Search Claim ID",placeholder="Type full or partial Claim ID...",key=f"{key_prefix}_drill_search")
    drill_ids=part["Claim_ID"].astype(str)
    if drill_search.strip(): drill_ids=drill_ids[drill_ids.str.contains(drill_search.strip(),case=False,na=False)]
    if drill_ids.empty:
        st.warning("No matching Claim ID found in this drill-down."); st.markdown('</div>',unsafe_allow_html=True); return
    cid=st.selectbox("Open claim", drill_ids.tolist(), key=f"{key_prefix}_open_claim")
    row=part[part["Claim_ID"].astype(str)==cid].iloc[0]
    a,b,c,d=st.columns(4)
    for col,item in zip([a,b,c,d],[("Risk",f"{float(row.get('Risk_Score',0)):.1f}/100",row.get("Risk_Level","")),("Priority",row.get("Recommended_Priority",""),"Recommended"),("Department",route_department(row),"Current claim routing"),("Expected Loss",short_money(row.get("Expected_Loss",0)),"Risk-adjusted")]):
        with col: card(*item)
    st.markdown('</div>', unsafe_allow_html=True)

def _event_value(event, field):
    try:
        sel = event.selection
        if not sel: return None
        for payload in sel.values():
            if isinstance(payload, list) and payload:
                item = payload[0]
                if isinstance(item, dict): return item.get(field)
            if isinstance(payload, dict):
                if field in payload: return payload.get(field)
                vals = payload.get("values")
                if isinstance(vals, list) and vals and isinstance(vals[0], dict): return vals[0].get(field)
    except Exception:
        return None
    return None

def _sync_chart_choice(state_key, clicked, valid):
    """Chart click becomes the active filter. Selector provides a reliable All/reset path."""
    if state_key not in st.session_state or st.session_state[state_key] not in valid:
        st.session_state[state_key] = "All"
    if clicked is not None and str(clicked) in valid and st.session_state[state_key] != str(clicked):
        st.session_state[state_key] = str(clicked)
        st.rerun()
    return st.session_state[state_key]

def interactive_risk_chart(df, key, scope="global"):
    order=["Critical","High","Medium","Low"]
    state=f"{key}_filter"
    choice=st.radio("Risk filter", ["All"]+order, horizontal=True, key=state, label_visibility="collapsed")
    base=df[df["Risk_Level"].isin(order)].copy()
    view=base if choice=="All" else base[base["Risk_Level"]==choice]
    d=view.groupby("Risk_Level").size().reindex(order if choice=="All" else [choice],fill_value=0).reset_index(name="Claims")
    pick=alt.selection_point(fields=["Risk_Level"], name=f"{key}_pick")
    risk_colors=[RISK_COLORS[x] for x in order]
    chart=alt.Chart(d).mark_arc(innerRadius=52, stroke="white", strokeWidth=2).encode(
        theta=alt.Theta("Claims:Q"),
        color=alt.Color("Risk_Level:N", sort=order, scale=alt.Scale(domain=order, range=risk_colors), legend=alt.Legend(title=None)),
        tooltip=["Risk_Level:N","Claims:Q"]
    ).add_params(pick)
    event=st.altair_chart(chart,use_container_width=True,on_select="rerun",key=f"{key}_chart")
    clicked=_event_value(event,"Risk_Level")
    if clicked and choice != str(clicked):
        st.session_state[state]=str(clicked); set_drilldown(f"{clicked} Risk Claims",base[base["Risk_Level"]==clicked],scope); st.rerun()
    if choice != "All": set_drilldown(f"{choice} Risk Claims",base[base["Risk_Level"]==choice],scope)

def interactive_claim_type_chart(df, key, scope="global"):
    col=next((c for c in ["Claim_Type","ClaimType","Insurance_Type","Product_Type","Line_of_Business"] if c in df.columns),None)
    if not col: st.caption("Claim type unavailable."); return
    d=df.copy(); d[col]=d[col].fillna("Unspecified").astype(str)
    opts=d[col].value_counts().index.astype(str).tolist()
    state=f"{key}_filter"
    choice=st.selectbox("Claim type filter",["All"]+opts,key=state,label_visibility="collapsed")
    view=d if choice=="All" else d[d[col]==choice]
    g=view.groupby(col).agg(Claims=(col,"size"),Exposure=("Claim_Amount","sum") if "Claim_Amount" in d.columns else (col,"size")).reset_index().sort_values("Claims",ascending=False)
    pick=alt.selection_point(fields=[col],name=f"{key}_pick")
    # Keep each claim type's color stable even when only one type is selected.
    color_domain=opts
    color_range=[CLAIM_TYPE_COLORS[i % len(CLAIM_TYPE_COLORS)] for i in range(len(color_domain))]
    chart=alt.Chart(g).mark_bar(cornerRadiusEnd=4).encode(y=alt.Y(f"{col}:N",sort="-x",title=None),x=alt.X("Claims:Q",title="Claims"),color=alt.Color(f"{col}:N",legend=None,scale=alt.Scale(domain=color_domain,range=color_range)),tooltip=[alt.Tooltip(f"{col}:N",title="Claim Type"),"Claims:Q"]).add_params(pick)
    event=st.altair_chart(chart,use_container_width=True,on_select="rerun",key=f"{key}_chart")
    clicked=_event_value(event,col)
    if clicked is not None and choice != str(clicked):
        st.session_state[state]=str(clicked); set_drilldown(f"{clicked} Claims",d[d[col]==str(clicked)],scope); st.rerun()
    if choice!="All": set_drilldown(f"{choice} Claims",d[d[col]==choice],scope)
    return choice, col, view

def interactive_month_chart(df, key, scope="global"):
    if "Claim_Date" not in df.columns or "Claim_Amount" not in df.columns: st.caption("Date/exposure data unavailable."); return
    d=df.copy(); d["Claim_Date"]=pd.to_datetime(d["Claim_Date"],errors="coerce"); d=d.dropna(subset=["Claim_Date"])
    if d.empty: return
    d["Month"]=d["Claim_Date"].dt.to_period("M").astype(str); opts=sorted(d["Month"].unique().tolist())
    state=f"{key}_filter"; choice=st.selectbox("Month filter",["All"]+opts,key=state,label_visibility="collapsed")
    view=d if choice=="All" else d[d["Month"]==choice]
    
    if "Risk_Level" in view.columns:
        g=view.groupby(["Month","Risk_Level"],as_index=False)["Claim_Amount"].sum()
        pick=alt.selection_point(fields=["Month"],name=f"{key}_pick")
        chart=alt.Chart(g).mark_line(point=True).encode(x=alt.X("Month:N",title=None),y=alt.Y("Claim_Amount:Q",title="Exposure"),color=alt.Color("Risk_Level:N",scale=alt.Scale(domain=SUPPORTED_RISKS,range=[RISK_COLORS[x] for x in SUPPORTED_RISKS]),legend=alt.Legend(title=None,orient="top")),tooltip=["Month:N","Risk_Level:N",alt.Tooltip("Claim_Amount:Q",format=",.0f")]).add_params(pick)
    else:
        g=view.groupby("Month",as_index=False)["Claim_Amount"].sum(); pick=alt.selection_point(fields=["Month"],name=f"{key}_pick")
        chart=alt.Chart(g).mark_line(point=True).encode(x=alt.X("Month:N",title=None),y=alt.Y("Claim_Amount:Q",title="Exposure"),tooltip=["Month:N",alt.Tooltip("Claim_Amount:Q",format=",.0f")]).add_params(pick)
    event=st.altair_chart(chart,use_container_width=True,on_select="rerun",key=f"{key}_chart")
    clicked=_event_value(event,"Month")
    if clicked and choice!=str(clicked): st.session_state[state]=str(clicked); set_drilldown(f"Claims in {clicked}",d[d["Month"]==str(clicked)],scope); st.rerun()
    if choice!="All": set_drilldown(f"Claims in {choice}",d[d["Month"]==choice],scope)

def interactive_alert_type_chart(alerts, claims, key, scope="global"):
    if alerts.empty or "Alert_Type" not in alerts.columns: st.caption("No alerts available."); return
    opts=alerts["Alert_Type"].dropna().astype(str).value_counts().index.tolist(); state=f"{key}_filter"
    choice=st.selectbox("Alert filter",["All"]+opts,key=state,label_visibility="collapsed")
    view=alerts if choice=="All" else alerts[alerts["Alert_Type"].astype(str)==choice]
    priority_order=["Critical","High","Medium","Review"]
    if "Alert_Priority" in view.columns:
        g=view.groupby("Alert_Priority").size().reindex(priority_order,fill_value=0).reset_index(name="Alerts")
        chart=alt.Chart(g).mark_bar(cornerRadiusEnd=4).encode(y=alt.Y("Alert_Priority:N",sort=priority_order,title=None),x=alt.X("Alerts:Q",title="Alerts"),color=alt.Color("Alert_Priority:N",scale=alt.Scale(domain=priority_order,range=[RISK_COLORS["Critical"],RISK_COLORS["High"],RISK_COLORS["Medium"],"#7a55d9"]),legend=None),tooltip=["Alert_Priority:N","Alerts:Q"])
        st.altair_chart(chart,use_container_width=True,key=f"{key}_chart")
    else:
        g=view.groupby("Alert_Type").size().reset_index(name="Alerts").sort_values("Alerts",ascending=False)
        alert_domain=opts
        alert_range=[CLAIM_TYPE_COLORS[i % len(CLAIM_TYPE_COLORS)] for i in range(len(alert_domain))]
        chart=alt.Chart(g).mark_bar(cornerRadiusEnd=4).encode(y=alt.Y("Alert_Type:N",sort="-x",title=None),x=alt.X("Alerts:Q"),color=alt.Color("Alert_Type:N",legend=None,scale=alt.Scale(domain=alert_domain,range=alert_range)),tooltip=["Alert_Type:N","Alerts:Q"])
        st.altair_chart(chart,use_container_width=True,key=f"{key}_chart")
    if choice!="All" and "Claim_ID" in alerts.columns:
        ids=alerts.loc[alerts["Alert_Type"].astype(str)==choice,"Claim_ID"].astype(str); set_drilldown(f"{choice} Alerts",claims[claims["Claim_ID"].astype(str).isin(ids)],scope)
    return choice, view

def interactive_risk_loss_chart(df,key,scope="global"):
    if "Risk_Level" not in df.columns or "Expected_Loss" not in df.columns: return
    order=["Critical","High","Medium","Low"]; state=f"{key}_filter"; choice=st.radio("Risk filter",["All"]+order,horizontal=True,key=state,label_visibility="collapsed")
    view=df if choice=="All" else df[df["Risk_Level"]==choice]; g=view.groupby("Risk_Level",as_index=False)["Expected_Loss"].sum(); pick=alt.selection_point(fields=["Risk_Level"],name=f"{key}_pick")
    chart=alt.Chart(g).mark_bar(cornerRadiusEnd=4).encode(x=alt.X("Risk_Level:N",sort=order,title=None),y=alt.Y("Expected_Loss:Q",title="Expected Loss"),color=alt.Color("Risk_Level:N",scale=alt.Scale(domain=order,range=[RISK_COLORS[x] for x in order]),legend=None),tooltip=["Risk_Level:N",alt.Tooltip("Expected_Loss:Q",format=",.0f")]).add_params(pick)
    event=st.altair_chart(chart,use_container_width=True,on_select="rerun",key=f"{key}_chart"); clicked=_event_value(event,"Risk_Level")
    if clicked and choice!=str(clicked): st.session_state[state]=str(clicked); set_drilldown(f"{clicked} Expected-Loss Claims",df[df["Risk_Level"]==str(clicked)],scope); st.rerun()
    if choice!="All": set_drilldown(f"{choice} Expected-Loss Claims",df[df["Risk_Level"]==choice],scope)

def interactive_matrix(df,key,scope="global"):
    risk=pd.to_numeric(df.get("Risk_Score"),errors="coerce"); amt=pd.to_numeric(df.get("Claim_Amount"),errors="coerce"); cut=float(amt.median()) if amt.notna().any() else 0
    groups=[("High Risk · Lower Exposure",(risk>=50)&(amt<cut)),("High Risk · Higher Exposure",(risk>=50)&(amt>=cut)),("Lower Risk · Lower Exposure",(risk<50)&(amt<cut)),("Lower Risk · Higher Exposure",(risk<50)&(amt>=cut))]
    cols=st.columns(2)
    for i,(name,mask) in enumerate(groups):
        part=df[mask.fillna(False)]
        with cols[i%2]:
            if st.button(f"{name}\n\n{len(part):,} claims · {short_money(pd.to_numeric(part.get('Claim_Amount',0),errors='coerce').fillna(0).sum())}",key=f"{key}_{i}",use_container_width=True): set_drilldown(name,part,scope); st.rerun()

def risk_exposure_matrix(df):
    risk = pd.to_numeric(df.get("Risk_Score", pd.Series(index=df.index, dtype=float)), errors="coerce")
    amt = pd.to_numeric(df.get("Claim_Amount", pd.Series(index=df.index, dtype=float)), errors="coerce")
    work = pd.DataFrame({"risk": risk, "amount": amt}).dropna()
    if work.empty:
        st.caption("Risk/exposure data is not available.")
        return
    risk_cut = 50.0
    exposure_cut = float(work["amount"].median())
    total = max(len(work), 1)
    cells = [
        ("High Risk · Lower Exposure", work[(work.risk >= risk_cut) & (work.amount < exposure_cut)], "mx-amber", "Priority Review"),
        ("High Risk · Higher Exposure", work[(work.risk >= risk_cut) & (work.amount >= exposure_cut)], "mx-red", "Immediate Attention"),
        ("Lower Risk · Lower Exposure", work[(work.risk < risk_cut) & (work.amount < exposure_cut)], "mx-green", "Routine Review"),
        ("Lower Risk · Higher Exposure", work[(work.risk < risk_cut) & (work.amount >= exposure_cut)], "mx-blue", "Financial Review"),
    ]
    html='<div class="mx-grid">'
    for title, part, cls, action in cells:
        exposure=float(part["amount"].sum()) if not part.empty else 0.0
        pct=len(part)/total*100
        html += f'<div class="mx-cell {cls}"><div class="mx-title">{title}</div><div class="mx-num">{len(part):,}</div><div class="mx-meta">{short_money(exposure)} exposure · {pct:.1f}% of claims</div><div class="mx-action">{action}</div></div>'
    html += '</div><div class="mx-note">Thresholds: Risk Score 50 · Higher exposure = above portfolio median ('+short_money(exposure_cut)+')</div>'
    st.markdown(html, unsafe_allow_html=True)

def department_portfolio(df):
    candidates=["Claim_Type","ClaimType","Insurance_Type","Policy_Type","Product_Type","Line_of_Business","Department"]
    col=next((c for c in candidates if c in df.columns), None)
    if not col:
        st.caption("Claim-type field is not available in the current portfolio data.")
        return
    work=df[[col]+(["Claim_Amount"] if "Claim_Amount" in df.columns else [])].copy()
    work[col]=work[col].fillna("Unspecified").astype(str).str.strip().replace("","Unspecified")
    if "Claim_Amount" not in work: work["Claim_Amount"]=0.0
    work["Claim_Amount"]=pd.to_numeric(work["Claim_Amount"],errors="coerce").fillna(0)
    g=work.groupby(col,dropna=False).agg(Claims=(col,"size"),Exposure=("Claim_Amount","sum")).sort_values("Claims",ascending=False).head(6)
    total=max(int(g["Claims"].sum()),1)
    rows=''
    for name,row in g.iterrows():
        pct=float(row["Claims"])/total*100
        rows += f'<div class="dept-row"><div class="dept-name">{name}</div><div class="dept-count">{int(row["Claims"]):,}</div><div class="dept-exp">{short_money(row["Exposure"])}</div><div class="dept-track"><span style="width:{pct:.1f}%"></span></div><div class="dept-pct">{pct:.0f}%</div></div>'
    st.markdown('<div class="dept-head"><span>Claim Type</span><span>Claims</span><span>Exposure</span><span></span><span>Share</span></div>'+rows,unsafe_allow_html=True)

def shell(sec, default=None):
    st.markdown(f'<div class="cg-top"><span class="cg-bread">Claims / <b>{sec}</b></span><span class="cg-ready">● SYSTEM READY</span></div>',unsafe_allow_html=True)
    menus={"Dashboard":["Executive Dashboard","Operational Monitor","Financial Overview","Risk Analytics"],"Claims":["Assessment Workspace","Claim Review","Risk Result"],"Operations":["Alerts","Action Queue","Routing & Escalation"],"Intelligence":["Risk Intelligence","Portfolio Analytics","Exposure"],"Governance":["Model Performance","Governance","Policy Rules"]}
    options=menus[sec]
    idx=options.index(default) if default in options else 0
    return st.radio(f"{sec} navigation", options, index=idx, horizontal=True, label_visibility="collapsed", key=f"topnav_{sec}")

st.sidebar.markdown('<div class="cg-brand"><i>📋</i>CLAIMGUARD AI</div><div class="cg-sub">Intelligent Claims Platform</div>',unsafe_allow_html=True)
st.sidebar.markdown('<div class="cg-sec">WORKSPACE</div>',unsafe_allow_html=True)
section=st.sidebar.radio("Workspace",["Dashboard","Claims","Operations","Intelligence","Governance"],label_visibility="collapsed")
st.sidebar.markdown('<div class="cg-sec">CLAIMS INTELLIGENCE</div>',unsafe_allow_html=True)
st.sidebar.caption("AI-assisted risk assessment and routing. Final decisions remain human-authorized.")

try:
    claims_df,data_warnings,files_loaded=load_claims_data(); models=load_models_cached(); model_status=re.models_available(models)
    if claims_df.empty: st.error("No claim data found in data/processed/."); st.stop()
    scored_df=score_portfolio(claims_df,tuple(sorted(model_status.items())))
    scored_df["Routed_Department"]=scored_df.apply(route_department,axis=1)
    scored_df["Recommended_Priority"]=scored_df.apply(recommend_priority,axis=1)
    alerts_df=ae.generate_alerts(scored_df); actions_df=ma.build_management_actions(alerts_df,scored_df)
    if not actions_df.empty and "Claim_ID" in actions_df.columns:
        enrich=scored_df[[c for c in ["Claim_ID","Routed_Department","Recommended_Priority"] if c in scored_df.columns]].drop_duplicates("Claim_ID")
        actions_df=actions_df.merge(enrich,on="Claim_ID",how="left")
except Exception as exc:
    st.error("Unable to load claims workspace.")
    with st.expander("Technical details"): st.exception(exc)
    st.stop()

if section=="Dashboard":
    dview=shell("Dashboard","Executive Dashboard")

    # One synchronized dashboard context: every KPI, chart, queue and drill-down
    # uses the same filtered claim set. This prevents a selected date/risk/type
    # from changing only the lower table while the top still shows all claims.
    dashboard_base = scored_df.copy()
    date_choice = "All"
    if "Claim_Date" in dashboard_base.columns:
        dashboard_base["_DashboardDate"] = pd.to_datetime(dashboard_base["Claim_Date"], errors="coerce").dt.date
        date_options = sorted([d for d in dashboard_base["_DashboardDate"].dropna().unique().tolist()])
        date_labels = ["All"] + [str(d) for d in date_options]
        date_choice = st.selectbox("Dashboard Date", date_labels, key="dashboard_date_filter")

    # Risk and claim-type selections made inside their charts are promoted to
    # dashboard-level filters on the next Streamlit rerun.
    risk_choice = st.session_state.get("dash_risk_donut_filter", "All")
    type_choice = st.session_state.get("dash_claim_type_filter", "All")

    dash_df = dashboard_base.copy()
    if date_choice != "All" and "_DashboardDate" in dash_df.columns:
        selected_date = pd.to_datetime(date_choice, errors="coerce").date()
        dash_df = dash_df[dash_df["_DashboardDate"].eq(selected_date)].copy()
    if risk_choice in SUPPORTED_RISKS and "Risk_Level" in dash_df.columns:
        dash_df = dash_df[dash_df["Risk_Level"].astype(str).eq(risk_choice)].copy()
    dash_pretype_df = dash_df.copy()
    claim_type_col = next((c for c in ["Claim_Type","ClaimType","Insurance_Type","Product_Type","Line_of_Business"] if c in dash_df.columns), None)
    if type_choice != "All" and claim_type_col:
        dash_df = dash_df[dash_df[claim_type_col].fillna("Unspecified").astype(str).eq(str(type_choice))].copy()

    dash_alerts_df = ae.generate_alerts(dash_df) if not dash_df.empty else pd.DataFrame()
    dash_actions_df = ma.build_management_actions(dash_alerts_df,dash_df) if not dash_df.empty else pd.DataFrame()
    if not dash_actions_df.empty and "Claim_ID" in dash_actions_df.columns:
        enrich=dash_df[[c for c in ["Claim_ID","Routed_Department","Recommended_Priority"] if c in dash_df.columns]].drop_duplicates("Claim_ID")
        dash_actions_df=dash_actions_df.merge(enrich,on="Claim_ID",how="left")

    active_parts=[f"Date: {date_choice}", f"Risk: {risk_choice}", f"Claim Type: {type_choice}", f"Claims: {len(dash_df):,}"]
    st.info("Active Filter Summary · " + " · ".join(active_parts))

    if "Claim_ID" in dash_df.columns:
        dash_search=st.text_input("Search portfolio by Claim ID",placeholder="Search Claim ID...",key="dashboard_claim_search")
        if dash_search.strip():
            dash_matches=dash_df[dash_df["Claim_ID"].astype(str).str.contains(dash_search.strip(),case=False,na=False)]
            if not dash_matches.empty:
                set_drilldown(f"Search: {dash_search.strip()}",dash_matches,"dashboard")
            else:
                st.caption("No matching Claim ID found.")
    fin=pa.financial_exposure_summary(dash_df)
    counts={x:int((dash_df["Risk_Level"]==x).sum()) for x in ["Critical","High","Medium","Low"]}
    priority=counts["Critical"]+counts["High"]

    if dview=="Executive Dashboard":
        cols=st.columns(6)
        items=[("Active Claims",f"{fin['total_claims']:,}","Total portfolio"),("Claim Exposure",short_money(fin['total_claim_amount']),"Reported exposure"),("Predicted",short_money(fin['predicted_exposure']),"Model estimate"),("Expected Loss",short_money(fin['total_expected_loss']),"Modeled risk"),("Priority",priority,"Critical + High"),("Open Alerts",len(dash_alerts_df),"Portfolio signals")]
        for c,x in zip(cols,items):
            with c: card(*x)
        # KPI drill-down controls
        bc=st.columns(6)
        all_claims=dash_df
        priority_claims=dash_df[dash_df["Risk_Level"].isin(["Critical","High"])]
        alert_claims=dash_df[dash_df["Claim_ID"].astype(str).isin(dash_alerts_df["Claim_ID"].astype(str).unique())] if (not dash_alerts_df.empty and "Claim_ID" in dash_alerts_df and "Claim_ID" in dash_df) else dash_df.iloc[0:0]
        for col,(lab,frm,key) in zip(bc,[("View claims",all_claims,"k_all"),("View exposure",all_claims,"k_exp"),("View predictions",all_claims,"k_pred"),("View expected loss",all_claims,"k_loss"),("Open priority claims",priority_claims,"k_priority"),("Open alerted claims",alert_claims,"k_alert")]):
            with col: drill_button(lab,frm,key,"dashboard")
        st.write("")
        l,r=st.columns([1.55,1])
        with l:
            st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Claim Exposure Trend","Portfolio exposure over time")
            if "Claim_Date" in dash_df and "Claim_Amount" in dash_df:
                d=dash_df.copy(); d["Claim_Date"]=pd.to_datetime(d["Claim_Date"],errors="coerce"); d=d.dropna(subset=["Claim_Date"])
                if not d.empty: interactive_month_chart(dash_df,"dash_month_chart","dashboard")
            st.markdown('</div>',unsafe_allow_html=True)
        with r:
            st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Risk Severity Distribution","Share of claims by risk level")
            total=max(sum(counts.values()),1)
            interactive_risk_chart(dash_df,"dash_risk_donut","dashboard")
            st.markdown('</div>',unsafe_allow_html=True)
        rb=st.columns(4)
        for col,lvl in zip(rb,["Critical","High","Medium","Low"]):
            with col: drill_button(f"Open {lvl} ({counts[lvl]})", dash_df[dash_df["Risk_Level"]==lvl], f"dash_risk_{lvl}","dashboard")
        l,r=st.columns([1.08,1])
        with l:
            st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Risk vs Financial Exposure","Claims segmented into decision-ready risk/exposure groups")
            interactive_matrix(dash_df,"dash_matrix","dashboard"); st.markdown('</div>',unsafe_allow_html=True)
        with r:
            st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Portfolio by Claim Type","Claim distribution and exposure from actual portfolio fields")
            ct_result=interactive_claim_type_chart(dash_pretype_df,"dash_claim_type","dashboard")
            if ct_result:
                ct_choice, ct_col, ct_view = ct_result
                label = "All Claim Types" if ct_choice=="All" else ct_choice
                critical_ct=int((ct_view.get("Risk_Level",pd.Series(dtype=str)).astype(str)=="Critical").sum()) if "Risk_Level" in ct_view else 0
                high_ct=int((ct_view.get("Risk_Level",pd.Series(dtype=str)).astype(str)=="High").sum()) if "Risk_Level" in ct_view else 0
                if "Claim_Status" in ct_view.columns:
                    pending_ct=int(ct_view["Claim_Status"].astype(str).str.contains("pending|open|review",case=False,regex=True,na=False).sum())
                    pending_label="Pending / Open"
                else:
                    pending_ct=critical_ct+high_ct; pending_label="Priority Review"
                exposure_ct=float(pd.to_numeric(ct_view.get("Claim_Amount",0),errors="coerce").fillna(0).sum()) if len(ct_view) else 0
                st.caption(f"Dynamic summary · {label} · Exposure {short_money(exposure_ct)}")
                k1,k2,k3,k4=st.columns(4)
                for kk,xx in zip([k1,k2,k3,k4],[("Claims",len(ct_view),label),("Critical",critical_ct,"Selected type"),("High Risk",high_ct,"Selected type"),(pending_label,pending_ct,"Selected type")]):
                    with kk: card(*xx)
            st.markdown('</div>',unsafe_allow_html=True)
        p1=int((dash_actions_df["Action_Priority"]=="P1").sum()) if not dash_actions_df.empty else 0; p2=int((dash_actions_df["Action_Priority"]=="P2").sum()) if not dash_actions_df.empty else 0; p3=int((dash_actions_df["Action_Priority"]=="P3").sum()) if not dash_actions_df.empty else 0; p4=int((dash_actions_df["Action_Priority"]=="P4").sum()) if not dash_actions_df.empty else 0
        st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Priority Work Queue","Claims requiring attention based on current action priorities")
        q=st.columns(4)
        qitems=[("P1 · Immediate Review",p1,"Critical / urgent","red"),("P2 · Priority Review",p2,"High-priority","amber"),("P3 · Monitor",p3,"Medium-priority",""),("P4 · Stable",p4,"Routine review","green")]
        for col,(title,val,note,cls) in zip(q,qitems):
            with col:
                st.markdown(f'<div class="cg-route"><div class="cg-rt">{title}</div><div style="display:flex;justify-content:space-between;align-items:end;margin-top:7px"><small>{note}</small><span class="pill {cls}">{val:,}</span></div></div>',unsafe_allow_html=True)
                pri=title.split(" ")[0]
                frame=dash_df[dash_df["Recommended_Priority"]==pri]
                drill_button(f"Open {pri} claims",frame,f"queue_{pri}","dashboard")
        st.markdown('</div>',unsafe_allow_html=True)


    elif dview=="Operational Monitor":
        sm=ae.alert_summary(dash_alerts_df)
        vals={str(r["Alert_Priority"]):int(r["Count"]) for _,r in sm.iterrows()} if not sm.empty else {}
        cols=st.columns(4)
        for c,x in zip(cols,[("Critical Alerts",vals.get("Critical",0),"Immediate"),("High Alerts",vals.get("High",0),"Priority"),("Action Queue",len(dash_actions_df),"Human review"),("Priority Claims",priority,"Critical + High")]):
            with c: card(*x)
        st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Operational Queue","Latest alerts requiring review")
        st.dataframe(dash_alerts_df.head(25),use_container_width=True,hide_index=True,height=390); st.markdown('</div>',unsafe_allow_html=True)

    elif dview=="Financial Overview":
        cols=st.columns(3)
        for c,x in zip(cols,[("Reported Exposure",short_money(fin["total_claim_amount"]),"Current portfolio"),("Predicted Exposure",short_money(fin["predicted_exposure"]),"Model estimate"),("Expected Loss",short_money(fin["total_expected_loss"]),"Risk-adjusted")]):
            with c: card(*x)
        st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Financial Exposure Trend","Monthly claim exposure")
        if "Claim_Date" in dash_df and "Claim_Amount" in dash_df:
            d=dash_df.copy(); d["Claim_Date"]=pd.to_datetime(d["Claim_Date"],errors="coerce"); d=d.dropna(subset=["Claim_Date"])
            if not d.empty: interactive_month_chart(dash_df,"financial_month_chart","dashboard")
        st.markdown('</div>',unsafe_allow_html=True)

    else:
        total=max(sum(counts.values()),1)
        cols=st.columns(4)
        for c,x in zip(cols,[("Critical",counts["Critical"],"Immediate review"),("High",counts["High"],"Priority review"),("Medium",counts["Medium"],"Monitor"),("Low",counts["Low"],"Stable")]):
            with c: card(*x)
        a,b=st.columns([1,1.5])
        with a:
            st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Risk Severity Mix","Select All or one risk level")
            interactive_risk_chart(dash_df,"dash_risk_analytics","dashboard")
            st.markdown('</div>',unsafe_allow_html=True)
        with b:
            st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Risk vs Financial Exposure","Decision matrix using actual portfolio risk and exposure")
            interactive_matrix(dash_df,"intel_matrix","intelligence")
            st.markdown('</div>',unsafe_allow_html=True)


elif section=="Claims":
    cview=shell("Claims","Assessment Workspace")
    if cview=="Assessment Workspace":
        st.markdown("### New Claim Assessment")
        mode=st.radio("Assessment",["Existing Claim","New Claim"],horizontal=True,key="claim_assessment_mode")
        if mode=="Existing Claim" and "Claim_ID" in scored_df:
            search_existing=st.text_input("Search Claim ID",placeholder="Type full or partial Claim ID...",key="existing_claim_search")
            existing_ids=scored_df["Claim_ID"].astype(str)
            if search_existing.strip(): existing_ids=existing_ids[existing_ids.str.contains(search_existing.strip(),case=False,na=False)]
            cid=st.selectbox("Claim ID",existing_ids.tolist(),key="existing_claim_id") if len(existing_ids) else None
            if cid is None:
                st.warning("No matching Claim ID found."); st.stop()
            row=scored_df[scored_df["Claim_ID"].astype(str)==cid].iloc[0]
            cols=st.columns(5)
            items=[("Claim Amount",currency(row.get("Claim_Amount",0)),"Reported"),("Coverage",currency(row.get("Coverage_Amount",0)),"Policy"),("Risk Score",f"{float(row.get('Risk_Score',0)):.1f}","0–100"),("Risk Level",row.get("Risk_Level","Low"),"Review band"),("High-Cost",f"{float(row.get('High_Cost_Probability',0)):.0%}","Probability")]
            for c,x in zip(cols,items):
                with c: card(*x)
            dept=route_department(row); pri=recommend_priority(row)
            st.info(re.recommended_action(row.get("Risk_Level","Low")))
            st.caption(f"Current claim routing: **{dept}** · Current priority: **{pri}**")
        else:
            actual_types=[]
            if "Claim_Type" in scored_df.columns:
                actual_types=scored_df["Claim_Type"].dropna().astype(str).value_counts().index.tolist()
            base_types=["Accidental Damage","Third Party","Fire","Natural Disaster","Theft","Health","Travel","Life","Marine","Motor","Property","General"]
            claim_types=list(dict.fromkeys(actual_types+base_types))

            h1,h2=st.columns(2)
            with h1:
                claim_type=st.selectbox("Claim Type",claim_types,key="new_claim_type")
            with h2:
                severity=st.selectbox("Incident Severity",["Low","Moderate","High","Severe"],index=1,key="new_incident_severity")

            # Claim type + severity create transparent recommended defaults. They remain editable.
            rec_required,rec_submitted,rec_evidence,rec_docs=document_profile(claim_type,severity)
            profile_sig=(claim_type,severity)
            if st.session_state.get("doc_profile_signature") != profile_sig:
                st.session_state["new_docs_required"]=rec_required
                st.session_state["new_docs_submitted"]=rec_submitted
                st.session_state["new_evidence"]=rec_evidence
                st.session_state["doc_profile_signature"]=profile_sig
                st.session_state.pop("new_claim_result",None)

            a,b,c=st.columns(3)
            with a:
                ca=st.number_input("Claim Amount (₹)",0.0,10000000.0,100000.0,1000.0,key="new_claim_amount")
                cov=st.number_input("Coverage Amount (₹)",1.0,30000000.0,300000.0,1000.0,key="new_claim_coverage")
            with b:
                delay_days=st.number_input("Reporting Delay (days)",0,120,2,1,key="new_reporting_delay")
                previous_claims=st.number_input("Previous Claims (last 3 years)",0,10,1,1,key="new_previous_claims")
            with c:
                docs_required=st.number_input("Documents Required",1,20,key="new_docs_required")
                docs_submitted=st.number_input("Documents Submitted",0,20,key="new_docs_submitted")
                ev=st.number_input("Evidence Count",0,10,key="new_evidence")

            ir={"Low":20,"Moderate":40,"High":70,"Severe":90}[severity]
            rr=min(float(delay_days)/30.0*100.0,100.0)
            ci=min(float(previous_claims)/5.0*100.0,100.0)
            dc=min(float(docs_submitted)/max(float(docs_required),1.0)*100.0,100.0)
            signature=(claim_type,ca,cov,severity,delay_days,previous_claims,docs_required,docs_submitted,ev)
            if st.session_state.get("last_assessed_signature") != signature:
                st.session_state.pop("new_claim_result",None)
            st.caption(f"Derived inputs → Incident risk {ir:.0f}/100 · Reporting risk {rr:.0f}/100 · Claim intensity {ci:.0f}/100 · Document completeness {dc:.0f}/100")

            if st.button("Assess Claim",type="primary",use_container_width=True,key="assess_new_claim"):
                m=pd.DataFrame([{"Claim_ID":"NEW-CLAIM","Claim_Type":claim_type,"Claim_Amount":ca,"Coverage_Amount":cov,"Incident_Risk_Score":ir,"Reporting_Risk_Score":rr,"Customer_Claim_Intensity":ci,"Document_Completeness":dc,"Evidence_Count":ev}])
                prob=float(re.predict_high_cost_probability(models,m))
                res=re.compute_risk_score(high_cost_probability=prob,claim_amount=ca,coverage_amount=cov,incident_risk_0_100=ir,reporting_risk_0_100=rr,customer_intensity_0_100=ci,document_completeness_0_100=dc,evidence_0_100=min(ev/10*100,100))
                temp=pd.Series({"Claim_Type":claim_type,"Risk_Score":res["risk_score"],"Risk_Level":res["risk_level"],"High_Cost_Probability":prob,"Expected_Loss":ca*prob})
                st.session_state["new_claim_result"]={"risk_score":res["risk_score"],"risk_level":res["risk_level"],"prob":prob,"priority":recommend_priority(temp),"department":route_department(temp),"expected_loss":ca*prob,"claim_type":claim_type,"recommended_docs":rec_docs}
                st.session_state["last_assessed_signature"]=signature
                clear_drilldown(); st.rerun()

            out=st.session_state.get("new_claim_result")
            if out:
                r1,r2,r3,r4=st.columns(4)
                for cc,xx in zip([r1,r2,r3,r4],[("Risk Score",f"{out['risk_score']:.1f}/100",out['risk_level']),("High-Cost",f"{out['prob']:.0%}","Model probability"),("Priority",out['priority'],"Dynamic recommendation"),("Department",out['department'],"Auto-routed")]):
                    with cc: card(*xx)
                st.success(f"Assessment complete for {out['claim_type']} claim. Routed to {out['department']} with {out['priority']} priority.")

                left,right=st.columns([1,1])
                with left:
                    st.markdown(f'<div class="cg-panel"><div class="cg-pt">This claim will be routed to</div><div class="cg-row"><div><div class="cg-rt">{out["department"]}</div><div class="cg-rs">{out["priority"]} · {out["risk_level"]} risk · human review required</div></div><span class="pill green">ROUTING PREVIEW</span></div></div>',unsafe_allow_html=True)
                    preview_to=st.text_input("Preview To",placeholder=f"Enter authorized {out['department']} recipient",key="new_preview_to")
                    preview_cc=st.text_input("Preview CC (optional)",placeholder="Optional CC recipient(s)",key="new_preview_cc")
                    psub,pbody=routing_mail_preview("NEW-CLAIM",out["department"],out["priority"],out["risk_level"],out["risk_score"],out["expected_loss"])
                    with st.expander("Preview Email",expanded=False):
                        st.text_input("Subject",psub,key="new_preview_subject")
                        st.text_area("Message",pbody,height=180,key="new_preview_body")
                        st.caption("Preview only. Actual email handoff is completed in Operations → Routing & Escalation after authorized review.")
                with right:
                    st.markdown('<div class="cg-panel"><div class="cg-pt">Recommended Documents</div><div class="cg-ps">Claim-type and severity based review checklist</div>',unsafe_allow_html=True)
                    for doc in out.get("recommended_docs",[]):
                        st.markdown(f"✓ {doc}")
                    st.markdown('</div>',unsafe_allow_html=True)
    elif cview=="Claim Review":
        st.markdown("### Claim Review")
        cols=[c for c in ["Claim_ID","Claim_Amount","Coverage_Amount","Risk_Score","Risk_Level","High_Cost_Probability","Expected_Loss"] if c in scored_df.columns]
        st.dataframe(scored_df[cols].sort_values("Risk_Score",ascending=False) if "Risk_Score" in cols else scored_df[cols],use_container_width=True,hide_index=True,height=470)
    else:
        st.markdown("### Risk Result")
        if "Claim_ID" in scored_df:
            risk_search=st.text_input("Search Claim ID",placeholder="Type full or partial Claim ID...",key="risk_result_search")
            risk_ids=scored_df["Claim_ID"].astype(str)
            if risk_search.strip(): risk_ids=risk_ids[risk_ids.str.contains(risk_search.strip(),case=False,na=False)]
            cid=st.selectbox("Claim ID",risk_ids.tolist(),key="risk_result_claim") if len(risk_ids) else None
            if cid is None:
                st.warning("No matching Claim ID found."); st.stop()
            row=scored_df[scored_df["Claim_ID"].astype(str)==cid].iloc[0]
            cols=st.columns(4)
            for c,x in zip(cols,[("Risk Score",f"{float(row.get('Risk_Score',0)):.1f}/100","Composite score"),("Risk Level",row.get("Risk_Level",""),"Review band"),("High-Cost",f"{float(row.get('High_Cost_Probability',0)):.0%}","Probability"),("Expected Loss",currency(row.get("Expected_Loss",0)),"Risk-adjusted")]):
                with c: card(*x)
            st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Recommended Review Action","Decision support only")
            st.info(re.recommended_action(row.get("Risk_Level","Low"))); st.markdown('</div>',unsafe_allow_html=True)

elif section=="Operations":
    op_view=shell("Operations","Alerts")
    if op_view=="Alerts":
        # One source of truth: selected Alert Type filters cards, priority controls and inbox.
        alert_choice=st.session_state.get("ops_alert_type_filter","All")
        selected_alerts=alerts_df if alert_choice in [None,"All"] else alerts_df[alerts_df["Alert_Type"].astype(str)==str(alert_choice)]
        sm=ae.alert_summary(selected_alerts); vals={str(r["Alert_Priority"]):int(r["Count"]) for _,r in sm.iterrows()} if not sm.empty else {}
        st.caption(f"Current alert scope: **{alert_choice if alert_choice not in [None,'All'] else 'All Alert Types'}**")
        cols=st.columns(4)
        for c,x in zip(cols,[("Critical",vals.get("Critical",0),"Immediate attention"),("High",vals.get("High",0),"Priority review"),("Medium",vals.get("Medium",0),"Monitor closely"),("Review",vals.get("Review",0),"Standard review")]):
            with c: card(*x)
        a,b=st.columns([1,1.7])
        with a:
            st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Alert Mix","Select an alert type; all values on this page recalculate")
            interactive_alert_type_chart(alerts_df,scored_df,"ops_alert_type","operations"); st.markdown('</div>',unsafe_allow_html=True)
        # Re-read after widget so current rerun state is consistently reflected.
        alert_choice=st.session_state.get("ops_alert_type_filter","All")
        selected_alerts=alerts_df if alert_choice=="All" else alerts_df[alerts_df["Alert_Type"].astype(str)==str(alert_choice)]
        with b:
            st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Alert Inbox","Filtered operational review queue")
            available=[p for p in ae.ALERT_PRIORITY_ORDER if p in selected_alerts.get("Alert_Priority",pd.Series(dtype=str)).astype(str).unique().tolist()]
            flt=st.multiselect("Priority",ae.ALERT_PRIORITY_ORDER,default=available,key="ops_alert_priority_filter")
            inbox=selected_alerts[selected_alerts["Alert_Priority"].isin(flt)] if (flt and "Alert_Priority" in selected_alerts) else selected_alerts.iloc[0:0]
            st.dataframe(inbox,use_container_width=True,hide_index=True,height=300); st.markdown('</div>',unsafe_allow_html=True)
        prcols=st.columns(4)
        for cc,lvl in zip(prcols,["Critical","High","Medium","Review"]):
            with cc:
                ids=selected_alerts.loc[selected_alerts["Alert_Priority"].astype(str)==lvl,"Claim_ID"].astype(str) if (not selected_alerts.empty and "Alert_Priority" in selected_alerts and "Claim_ID" in selected_alerts) else pd.Series(dtype=str)
                drill_button(f"Open {lvl} ({len(ids)})",scored_df[scored_df["Claim_ID"].astype(str).isin(ids)] if "Claim_ID" in scored_df else scored_df.iloc[0:0],f"alert_pr_{lvl}","operations")
    elif op_view=="Action Queue":
        sm=ma.action_summary(actions_df); cols=st.columns(4)
        for c,(_,r0) in zip(cols,sm.iterrows()):
            with c: card(f'{r0["Action_Priority"]} · {r0["Label"]}',int(r0["Count"]),"Human work queue")
        st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Management Work Queue","Alerts consolidated into one priority per claim"); st.dataframe(actions_df,use_container_width=True,hide_index=True,height=410); st.markdown('</div>',unsafe_allow_html=True)
    else:
        st.markdown("### Department Routing & Admin Review"); st.caption("Filter → select claim → admin routing review → email-app handoff. No email credentials are stored.")
        if scored_df.empty or "Claim_ID" not in scored_df.columns:
            st.info("No claims available for routing.")
        else:
            q=scored_df.copy()
            q["Routing_Department"]=q.apply(route_department,axis=1).astype(str).str.strip()
            q["Routing_Priority"]=q.apply(recommend_priority,axis=1).astype(str).str.strip().str.upper()
            q["Routing_Risk"]=q["Risk_Level"].fillna("").astype(str).str.strip().str.title()
            q["Claim_ID"]=q["Claim_ID"].astype(str).str.strip()
            f1,f2,f3=st.columns(3)
            departments=SUPPORTED_DEPARTMENTS
            priorities=SUPPORTED_PRIORITIES
            risks=SUPPORTED_RISKS
            with f1: dept_filter=st.selectbox("Department",["All Departments"]+departments,key="route_dept_filter")
            with f2: pri_filter=st.selectbox("Priority",["All Priorities"]+priorities,key="route_priority_filter")
            with f3: risk_filter=st.selectbox("Risk Level",["All Risk Levels"]+risks,key="route_risk_filter")
            fq=q.copy()
            if dept_filter!="All Departments": fq=fq[fq["Routing_Department"].eq(dept_filter)]
            if pri_filter!="All Priorities": fq=fq[fq["Routing_Priority"].eq(pri_filter)]
            if risk_filter!="All Risk Levels": fq=fq[fq["Routing_Risk"].eq(risk_filter)]
            search_id=st.text_input("Search Claim ID",placeholder="Type full or partial Claim ID...",key="routing_claim_search")
            if search_id.strip(): fq=fq[fq["Claim_ID"].str.contains(search_id.strip(),case=False,na=False,regex=False)]
            st.info(f"Active Filter Summary · {len(fq):,} matching claims · {dept_filter} · {pri_filter} · {risk_filter}")
            if fq.empty:
                st.warning("No claims match this Department / Priority / Risk combination. Change one or more filters.")
            else:
                cid=st.selectbox("Claim ID",fq["Claim_ID"].astype(str).tolist(),key="routing_claim_id")
                sr=fq[fq["Claim_ID"].astype(str)==cid].iloc[0]
                dept=str(sr["Routing_Department"]); dynamic_priority=str(sr["Routing_Priority"])
                cols=st.columns(4)
                for c,x in zip(cols,[("Claim ID",cid,"Admin review"),("Department",dept,"Auto-routed from current claim"),("Priority",dynamic_priority,"Dynamic engine"),("Risk",sr.get("Risk_Level",""),"Model signal")]):
                    with c: card(*x)
                st.markdown(f'<div class="cg-panel"><div class="cg-pt">Routing Review</div><div class="cg-row"><div><div class="cg-rt">{dept}</div><div class="cg-rs">Recommended destination based on this claim only</div></div><span class="pill">ADMIN REVIEW</span></div></div>',unsafe_allow_html=True)
                track_key=f"routing_status_{cid}"
                if track_key not in st.session_state: st.session_state[track_key]="Pending Admin Review"
                s1,s2=st.columns([1,3])
                with s1:
                    if st.button("Approve Routing",key=f"approve_{cid}",use_container_width=True): st.session_state[track_key]="Routing Approved"
                with s2: st.info(f"Workflow status: {st.session_state[track_key]}")
                # Email handoff is intentionally hidden until an authorized admin approves routing.
                if st.session_state[track_key] in ["Routing Approved", "Email Handoff Complete"]:
                    sub_default,body_default=routing_mail_preview(cid,dept,dynamic_priority,sr.get("Risk_Level",""),sr.get("Risk_Score",0),sr.get("Expected_Loss",0))
                    st.success(f"Admin Status: Approved · Routing destination: {dept}")
                    to_addr=st.text_input("To",placeholder=f"Enter authorized {dept} recipient email",key=f"mail_to_{cid}")
                    cc_addr=st.text_input("CC (optional)",placeholder="Optional CC recipient(s)",key=f"mail_cc_{cid}")
                    sub=st.text_input("Email subject",sub_default,key=f"mail_subject_{cid}")
                    body=st.text_area("Email message",body_default,height=190,key=f"mail_body_{cid}")
                    link=ne.build_mailto_link(to_addr,sub,body)
                    if cc_addr.strip():
                        from urllib.parse import quote
                        sep="&" if "?" in link else "?"
                        link += f"{sep}cc={quote(cc_addr.strip())}"
                    if to_addr.strip():
                        st.markdown(f'<a href="{link}" target="_self"><button style="background:#1769d2;color:#fff;border:0;border-radius:8px;padding:10px 16px;font-weight:700">Open Email App to Send</button></a>',unsafe_allow_html=True)
                        st.caption("Your default mail application opens with To / CC / Subject / Message pre-filled. The authorized user completes the final Send action.")
                    else:
                        st.caption("Enter the authorized recipient email to enable the mail-app handoff.")
                else:
                    st.warning("Admin Status: Pending · Email is hidden until routing approval is completed.")

elif section=="Intelligence":
    iview=shell("Intelligence","Risk Intelligence")
    counts={x:int((scored_df["Risk_Level"]==x).sum()) for x in ["Critical","High","Medium","Low"]}; total=max(sum(counts.values()),1)
    if iview=="Risk Intelligence":
        c1,c2,c3,c4=st.columns(4)
        for c,x in zip([c1,c2,c3,c4],[("Critical",counts["Critical"],"Immediate review"),("High",counts["High"],"Priority review"),("Medium",counts["Medium"],"Monitor"),("Low",counts["Low"],"Stable")]):
            with c: card(*x)
        a,b=st.columns([1,1.4])
        with a:
            st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Risk Severity Mix","Portfolio composition")
            interactive_risk_chart(scored_df,"intel_risk_donut","intelligence"); st.markdown('</div>',unsafe_allow_html=True)
        with b:
            st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Risk vs Financial Exposure","Decision matrix using actual portfolio risk and exposure")
            interactive_matrix(scored_df,"intel_matrix","intelligence")
            st.markdown('</div>',unsafe_allow_html=True)
    elif iview=="Portfolio Analytics":
        a,b=st.columns(2)
        with a:
            st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Expected Loss by Risk","Click a risk band to open matching claims"); interactive_risk_loss_chart(scored_df,"intel_loss_risk","intelligence"); st.markdown('</div>',unsafe_allow_html=True)
        with b:
            st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Claim Mix","Click a claim type to open matching claims"); interactive_claim_type_chart(scored_df,"intel_claim_mix","intelligence"); st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Top Priority Claims","Highest attention items"); st.dataframe(pa.top_priority_claims(scored_df),use_container_width=True,hide_index=True,height=280); st.markdown('</div>',unsafe_allow_html=True)
    else:
        # Exposure is one synchronized filter context: KPIs, charts, scenario and claim data all use the same period.
        exposure_base=scored_df.copy()
        exposure_base["_ClaimDate"]=pd.to_datetime(exposure_base.get("Claim_Date"),errors="coerce") if "Claim_Date" in exposure_base.columns else pd.NaT
        valid_dates=exposure_base.dropna(subset=["_ClaimDate"]).copy()
        if not valid_dates.empty:
            valid_dates["_Month"]=valid_dates["_ClaimDate"].dt.to_period("M").astype(str)
            periods=sorted(valid_dates["_Month"].unique().tolist())
            period_choice=st.selectbox("Exposure Period",["All"]+periods,key="intel_exposure_period")
            exposure_view=exposure_base if period_choice=="All" else valid_dates[valid_dates["_Month"].eq(period_choice)].copy()
        else:
            period_choice="All"; exposure_view=exposure_base
        fin=pa.financial_exposure_summary(exposure_view); cols=st.columns(4)
        priority_count=int(exposure_view.get("Risk_Level",pd.Series(dtype=str)).astype(str).isin(["Critical","High"]).sum())
        for c,x in zip(cols,[("Claims",f"{len(exposure_view):,}","Selected period"),("Reported Exposure",short_money(fin["total_claim_amount"]),"Selected period"),("Predicted Exposure",short_money(fin["predicted_exposure"]),"Model estimate"),("Expected Loss",short_money(fin["total_expected_loss"]),"Risk-adjusted")]):
            with c: card(*x)
        st.info(f"Active Filter Summary · {len(exposure_view):,} claims · Period: {period_choice} · Priority claims: {priority_count:,}")
        a,b=st.columns([1.3,1])
        with a:
            st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Exposure by Risk","Selected-period exposure by risk level")
            if "Risk_Level" in exposure_view.columns and "Claim_Amount" in exposure_view.columns:
                eg=exposure_view.groupby("Risk_Level",as_index=False)["Claim_Amount"].sum()
                ech=alt.Chart(eg).mark_bar(cornerRadiusEnd=4).encode(x=alt.X("Risk_Level:N",sort=SUPPORTED_RISKS,title=None),y=alt.Y("Claim_Amount:Q",title="Exposure"),color=alt.Color("Risk_Level:N",scale=alt.Scale(domain=SUPPORTED_RISKS,range=[RISK_COLORS[x] for x in SUPPORTED_RISKS]),legend=None),tooltip=["Risk_Level:N",alt.Tooltip("Claim_Amount:Q",format=",.0f")])
                st.altair_chart(ech,use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)
        with b:
            st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Risk Mix","Selected-period claim distribution")
            interactive_risk_chart(exposure_view,"intel_exposure_risk","intelligence")
            st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('<div class="cg-panel">',unsafe_allow_html=True); phead("Exposure Claim Details","Claims included in the current period")
        exp_cols=[c for c in ["Claim_ID","Claim_Date","Claim_Type","Claim_Amount","Risk_Level","Expected_Loss","Recommended_Priority","Routed_Department"] if c in exposure_view.columns]
        exp_search=st.text_input("Search Claim ID",placeholder="Type full or partial Claim ID...",key="exposure_claim_search")
        exp_table=exposure_view.copy()
        if exp_search.strip() and "Claim_ID" in exp_table.columns: exp_table=exp_table[exp_table["Claim_ID"].astype(str).str.contains(exp_search.strip(),case=False,na=False,regex=False)]
        st.dataframe(exp_table[exp_cols] if exp_cols else exp_table,use_container_width=True,hide_index=True,height=300)
        st.markdown('</div>',unsafe_allow_html=True)

else:
    gview=shell("Governance","Model Performance")
    if gview=="Model Performance":
        st.markdown("### Claim Severity Prediction")
        a,b,c=st.columns(3)
        models1=[("Ridge Regression","R² 0.6222","RMSE ₹292,349",False),("Random Forest","R² 0.7869","RMSE ₹219,593",True),("Gradient Boosting","R² 0.7811","RMSE ₹222,525",False)]
        for col,(name,m1,m2,sel) in zip([a,b,c],models1):
            with col:
                cls="sel" if sel else ""; badge='<span class="cg-badge">SELECTED</span>' if sel else ""; st.markdown(f'<div class="cg-model {cls}">{badge}<b>{name}</b><div style="font-size:20px;font-weight:800;color:#17304f;margin-top:14px">{m1}</div><small style="color:#8492a4">{m2}</small></div>',unsafe_allow_html=True)
        st.info("Model interpretation: Severity Random Forest explains 78.69% of observed variance (R²). For the imbalanced high-cost target, evaluate Accuracy, F1 and ROC-AUC together rather than Accuracy alone. Scores shown here are the real evaluated results; they are not artificially inflated.")
        st.markdown("### High-Cost Claim Classification")
        a,b,c=st.columns(3); models2=[("Logistic Regression","F1 55.56%","ROC-AUC 93.97%",False),("Decision Tree","F1 63.41%","ROC-AUC 82.39%",False),("Random Forest","F1 68.29%","Accuracy 93.50%",True)]
        for col,(name,m1,m2,sel) in zip([a,b,c],models2):
            with col:
                cls="sel" if sel else ""; badge='<span class="cg-badge">SELECTED</span>' if sel else ""; st.markdown(f'<div class="cg-model {cls}">{badge}<b>{name}</b><div style="font-size:20px;font-weight:800;color:#17304f;margin-top:14px">{m1}</div><small style="color:#8492a4">{m2}</small></div>',unsafe_allow_html=True)
    elif gview=="Governance":
        st.markdown('<div class="cg-panel"><div class="cg-pt">Human-in-the-Loop Governance</div><div class="cg-ps">Decision support boundaries</div><div class="cg-row"><div><div class="cg-rt">AI may score, prioritize and route</div><div class="cg-rs">Supports authorized reviewers with structured signals</div></div><span class="pill green">SUPPORTED</span></div><div class="cg-row"><div><div class="cg-rt">AI does not approve, reject or settle</div><div class="cg-rs">Final claim decisions remain with authorized human personnel</div></div><span class="pill red">HUMAN ONLY</span></div></div>',unsafe_allow_html=True)
    else:
        st.markdown("### Policy & Decision Rules")
        st.caption("Quick reviewer reference. These checkpoints support human review and do not automatically approve or reject a claim.")
        policy_rules={
            "Motor Claims":["Policy must be active on the incident date and the claimed event must fall within covered motor risks.","Verify vehicle registration, driver licence and policy details against the reported vehicle.","Accident claims should include damage evidence and repair estimate; theft / third-party cases may require police or FIR documentation.","Check reporting delay, deductible / excess and applicable policy limits before financial review.","Material inconsistencies or missing mandatory evidence should be escalated for authorized human review."],
            "Property Claims":["Confirm insured property, policy period and covered peril such as fire or natural disaster.","Review ownership / insurable-interest evidence, damage photographs and repair or valuation estimate.","Where relevant, verify fire-service, police, surveyor or local-authority incident documentation.","Apply policy limits, deductible and excluded-peril conditions during reviewer assessment.","Large-loss or unusual-event claims require enhanced human review."],
            "Health Claims":["Confirm member eligibility, active coverage and treatment date within the policy period.","Check waiting-period, hospitalization / treatment eligibility and applicable coverage limits.","Review medical records, diagnosis, bills, prescriptions and discharge documentation as applicable.","Check exclusions, non-payable items, co-pay / deductible and network conditions where relevant.","Clinical or document inconsistencies must be referred to an authorized reviewer."],
            "Travel Claims":["Confirm active travel cover for the trip dates and insured traveller.","Verify tickets / itinerary and evidence for delay, cancellation, baggage loss or medical event.","Check event-specific limits, deductibles, exclusions and notification timelines.","Receipts, carrier confirmation or incident reports should support the claimed loss where applicable.","High-value or conflicting evidence requires human escalation."],
            "Life Claims":["Confirm policy status, insured identity and event date against policy coverage.","Verify nominee / beneficiary details and required identity or KYC documentation.","Review death / medical certificate and supporting records appropriate to the claim event.","Check applicable exclusions, contestability conditions and policy benefits before settlement review.","Final benefit eligibility and settlement remain authorized human decisions."],
            "Marine Claims":["Confirm cargo / marine policy period, voyage and insured interest.","Review bill of lading, invoice / packing list, survey report and damage evidence as applicable.","Check loss circumstances against covered marine perils, exclusions and policy limits.","Verify timely notification and preservation / mitigation of damaged cargo where required.","Complex or high-value losses should be escalated to specialist human review."],
            "General Claims":["Confirm active policy, claimant identity and coverage on the incident date.","Verify claim form, supporting evidence and financial documents relevant to the reported loss.","Check policy limits, deductibles, exclusions and reporting requirements.","Incomplete or inconsistent documentation should be returned for reviewer follow-up.","Final approval, rejection and settlement decisions remain with authorized personnel."]}
        policy_choice=st.selectbox("Insurance / Claim Type",SUPPORTED_DEPARTMENTS,key="governance_policy_type")
        st.markdown(f'<div class="cg-panel"><div class="cg-pt">{policy_choice}</div><div class="cg-ps">Essential policy checkpoints</div>',unsafe_allow_html=True)
        for i,rule in enumerate(policy_rules[policy_choice],1): st.markdown(f"**{i}.** {rule}")
        st.markdown('</div>',unsafe_allow_html=True)

show_drilldown(scored_df,"global",section.lower())

st.markdown('<div class="cg-foot">CLAIMGUARD AI · Portfolio Intelligence · Human authorization required</div>',unsafe_allow_html=True)

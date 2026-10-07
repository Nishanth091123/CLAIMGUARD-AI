"""
notification_engine.py
Insurance Claim Intelligence — Automated Notification & Escalation

Builds P1 / P2 notification queues and email-ready previews from the
management-action worklist. Also supports OPTIONAL live email delivery
via Gmail SMTP (send_email), only when the user has explicitly entered
their own sender credentials and a recipient in the app's sidebar — this
module never sends anything on its own, and credentials are held only in
the running session (never written to disk).
"""

import smtplib
import ssl
import urllib.parse
from email.mime.text import MIMEText

import pandas as pd
from datetime import datetime

GMAIL_SMTP_HOST = "smtp.gmail.com"
GMAIL_SMTP_PORT = 587


def build_mailto_link(recipient_email: str, subject: str, body: str) -> str:
    """Build a 'mailto:' link that opens the user's own default email app
    (Gmail web, Outlook, phone mail app — whatever is already signed in)
    with the subject and body pre-filled. Zero setup, no password or API
    key needed — the person just reviews and clicks Send themselves.

    recipient_email may be blank; the mailto link still opens with the
    subject/body filled in, and the person fills in "To" in their client.
    """
    query = urllib.parse.urlencode({"subject": subject, "body": body}, quote_via=urllib.parse.quote)
    to = urllib.parse.quote(recipient_email) if recipient_email else ""
    return f"mailto:{to}?{query}"


def build_notification_queue(actions_df: pd.DataFrame, priority: str) -> pd.DataFrame:
    """Filter the management-action worklist down to one priority tier."""
    if actions_df.empty:
        return actions_df
    return actions_df[actions_df["Action_Priority"] == priority].reset_index(drop=True)


def render_email_preview(row: pd.Series, claims_df: pd.DataFrame = None) -> dict:
    """Build an email-ready (subject, body) preview for one worklist row.

    If the project's own pipeline already generated real communication
    content for this claim (Email_Template / Customer_Email / SMS_Template
    / Customer_SMS / Manager_CTA columns in the source data), that real,
    pipeline-authored content is used. Otherwise a generic preview is
    generated from the alert/action context. Either way, no email is sent
    — this produces text a human can copy into their email client or
    forward to the live delivery integration later.
    """
    claim_id = row.get("Claim_ID", "N/A")
    priority_label = row.get("Priority_Label", "Review")
    action_priority = row.get("Action_Priority", "P4")

    claim_row = None
    if claims_df is not None and "Claim_ID" in claims_df.columns:
        match = claims_df[claims_df["Claim_ID"] == claim_id]
        if not match.empty:
            claim_row = match.iloc[0]

    # Prefer the pipeline's own generated communication content when present.
    if claim_row is not None:
        real_subject = claim_row.get("Email_Template") or claim_row.get("Manager_CTA")
        real_body = claim_row.get("Customer_Email") or claim_row.get("Email_Template")
        if real_body and pd.notna(real_body) and str(real_body).strip():
            subject = (
                str(real_subject).strip()
                if real_subject and pd.notna(real_subject) and str(real_subject).strip()
                else f"[{action_priority} - {priority_label}] Claim {claim_id} requires attention"
            )
            sms = claim_row.get("Customer_SMS") or claim_row.get("SMS_Template")
            manager_cta = claim_row.get("Manager_CTA")
            extra_lines = []
            if sms and pd.notna(sms) and str(sms).strip():
                extra_lines.append(f"\nSMS Template:\n{sms}")
            if manager_cta and pd.notna(manager_cta) and str(manager_cta).strip():
                extra_lines.append(f"\nManager CTA:\n{manager_cta}")
            body = (
                f"{real_body}\n"
                + "".join(extra_lines)
                + f"\n\n[Source: pipeline-generated communication content, claim {claim_id}, "
                f"priority {action_priority}]"
            )
            return {"subject": subject, "body": body}

    # Fall back to a generated preview from the alert/action context.
    subject = f"[{action_priority} - {priority_label}] Claim {claim_id} requires attention"

    extra = ""
    if claim_row is not None:
        claim_amount = claim_row.get("Claim_Amount", None)
        risk_score = claim_row.get("Risk_Score", None)
        if claim_amount is not None and pd.notna(claim_amount):
            extra += f"Claim Amount: ₹{claim_amount:,.0f}\n"
        if risk_score is not None and pd.notna(risk_score):
            extra += f"Risk Score: {risk_score:.1f}\n"

    body = (
        f"Claim ID: {claim_id}\n"
        f"Priority: {action_priority} ({priority_label})\n"
        f"Review Level: {row.get('Review_Level', '')}\n"
        f"{extra}"
        f"Reason(s): {row.get('Reason', '')}\n"
        f"Recommended Management Action: {row.get('Management_Action', '')}\n"
        f"Next Step: {row.get('Next_Step', '')}\n\n"
        f"Generated by Insurance Claim Intelligence — decision support only.\n"
        f"Final decision remains with authorized human personnel.\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    return {"subject": subject, "body": body}


def build_escalation_workflow_summary(actions_df: pd.DataFrame) -> pd.DataFrame:
    """A simple Queue -> Preview -> Ready-for-Email status table by tier."""
    tiers = ["P1", "P2", "P3", "P4"]
    labels = {"P1": "Immediate", "P2": "High", "P3": "Monitor", "P4": "Review"}
    rows = []
    for tier in tiers:
        n = 0 if actions_df.empty else int((actions_df["Action_Priority"] == tier).sum())
        status = "Ready for Email" if tier in ("P1", "P2") and n > 0 else (
            "Monitoring Queue" if tier == "P3" else "Standard Queue"
        )
        rows.append({"Priority": tier, "Label": labels[tier], "Queued_Claims": n, "Status": status})
    return pd.DataFrame(rows)


def send_email(sender_email: str, app_password: str, recipient_email: str, subject: str, body: str) -> tuple:
    """Send one email via Gmail SMTP (smtp.gmail.com:587, STARTTLS).

    sender_email / app_password are the user's own Gmail address and a
    16-character Gmail "App Password" (not their normal Gmail password —
    see README for how to create one). Neither is ever written to disk by
    this function; the caller (app.py) only holds them in the Streamlit
    session for as long as the browser tab is open.

    Returns (success: bool, message: str) — message is a short, human
    -readable result or error, safe to show directly in the UI.
    """
    if not sender_email or not app_password:
        return False, "Enter your Gmail address and App Password in the sidebar first."
    if not recipient_email:
        return False, "Enter a recipient email address in the sidebar first."

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(GMAIL_SMTP_HOST, GMAIL_SMTP_PORT, timeout=20) as server:
            server.starttls(context=context)
            server.login(sender_email, app_password)
            server.sendmail(sender_email, [recipient_email], msg.as_string())
        return True, f"Email sent to {recipient_email}."
    except smtplib.SMTPAuthenticationError:
        return False, (
            "Gmail rejected the login. Double-check the address and App Password — "
            "make sure 2-Step Verification is on and you're using the 16-character "
            "App Password, not your normal Gmail password."
        )
    except smtplib.SMTPException as exc:
        return False, f"Email could not be sent: {exc}"
    except OSError as exc:
        return False, f"Could not reach Gmail's mail server: {exc}"
    except Exception as exc:
        return False, f"Unexpected error while sending: {exc}"

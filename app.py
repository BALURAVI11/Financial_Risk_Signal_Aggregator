import json
import os
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Import our custom modules
from data_loader import (load_transactions, load_customers, load_alerts,
                         enrich_transactions)
from risk_engine import analyze_transactions, CROSS_SOURCE_DETECTORS
from cases import build_cases, portfolio_stats, cases_to_export
import ai_reasoning as AI
from ai_reasoning import generate_rationale, generate_search_summary, HAS_API, PROVIDER, MODEL
from currency import format_inr

# Load env variables
load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(APP_DIR, "data", "sample_data.csv")

# Streamlit ties st.session_state to a browser session, so a page refresh (F5)
# starts a fresh session and the in-memory dataset would be lost. The active
# dataset is therefore mirrored to disk as Parquet (dtype-preserving, unlike
# CSV) and restored on startup, so a refresh — or even restarting the server —
# keeps the analyst's working set intact until they explicitly clear it.
SESSION_CACHE_DIR = os.path.join(APP_DIR, ".session_cache")
SESSION_META_FILE = os.path.join(SESSION_CACHE_DIR, "active_dataset.json")
# One parquet per source; each is independently optional.
SESSION_FILES = {
    "raw_data": os.path.join(SESSION_CACHE_DIR, "active_dataset.parquet"),
    "customers_data": os.path.join(SESSION_CACHE_DIR, "active_customers.parquet"),
    "alerts_data": os.path.join(SESSION_CACHE_DIR, "active_alerts.parquet"),
}

DEFAULT_CUSTOMERS_CSV = os.path.join(APP_DIR, "data", "sample_customers.csv")
DEFAULT_ALERTS_CSV = os.path.join(APP_DIR, "data", "sample_alerts.csv")

# Page Configuration
st.set_page_config(
    page_title="Financial Risk Signal Aggregator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Dark Theme & Glassmorphism Aesthetics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 255, 255, 0.2);
    }

    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 5px;
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9CA3AF;
        font-weight: 500;
    }

    .badge-critical {
        background-color: #EF4444; color: white; padding: 4px 8px;
        border-radius: 4px; font-weight: bold; font-size: 0.8rem;
    }
    .badge-high {
        background-color: #F97316; color: white; padding: 4px 8px;
        border-radius: 4px; font-weight: bold; font-size: 0.8rem;
    }
    .badge-medium {
        background-color: #F59E0B; color: white; padding: 4px 8px;
        border-radius: 4px; font-weight: bold; font-size: 0.8rem;
    }
    .badge-low {
        background-color: #10B981; color: white; padding: 4px 8px;
        border-radius: 4px; font-weight: bold; font-size: 0.8rem;
    }

    .ai-report {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 24px;
        margin-top: 15px;
        box-shadow: inset 0 0 15px rgba(139, 92, 246, 0.05);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: #9CA3AF;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        border-color: #3B82F6 !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_meta() -> None:
    meta = {"loaded_at": datetime.now().isoformat(timespec="seconds"),
            "labels": st.session_state.get("source_labels", {})}
    with open(SESSION_META_FILE, "w") as fh:
        json.dump(meta, fh)


def persist_source(key: str, df: pd.DataFrame) -> None:
    """Mirror one source to disk so it survives a browser refresh."""
    try:
        os.makedirs(SESSION_CACHE_DIR, exist_ok=True)
        df.to_parquet(SESSION_FILES[key], index=False)
        _write_meta()
    except Exception as exc:
        # Persistence is a convenience, never a hard dependency — if the disk
        # write fails the app must still work for the current session.
        st.warning(f"Could not save '{key}' for reuse after refresh: {exc}")


def restore_all():
    """Reload every source previously written by persist_source()."""
    restored, meta = {}, {}
    if os.path.exists(SESSION_META_FILE):
        try:
            with open(SESSION_META_FILE) as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}
    for key, path in SESSION_FILES.items():
        if os.path.exists(path):
            try:
                restored[key] = pd.read_parquet(path)
            except Exception:
                pass          # a corrupt/unreadable cache is simply ignored
    return restored, meta.get("labels", {})


def clear_persisted_dataset() -> None:
    for path in list(SESSION_FILES.values()) + [SESSION_META_FILE]:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def set_active_source(key: str, df: pd.DataFrame, source_label: str) -> None:
    """Single entry point for making a source current: updates the session and
    mirrors it to disk so the two never drift apart."""
    st.session_state[key] = df
    st.session_state.setdefault("source_labels", {})[key] = source_label
    persist_source(key, df)


# Every stage below is cached because Streamlit re-executes this whole script on
# EVERY widget interaction. Without caching, each filter change would re-run the
# detector suite — and, far worse, re-issue every LLM call and its cost.
@st.cache_data(show_spinner=False)
def run_alert_extraction(alerts_df: pd.DataFrame):
    """LLM pass over the unstructured notes. Cached hard: this is the only
    stage that spends money per run."""
    return AI.extract_alert_signals_batch(alerts_df)


@st.cache_data(show_spinner=False)
def run_analysis(raw_df: pd.DataFrame, customers_df, alert_signals_df) -> pd.DataFrame:
    """Correlate the available sources, then score. Passing the optional frames
    through the cache key means changing/removing a source correctly
    invalidates the analysis."""
    enriched = enrich_transactions(raw_df, customers_df, alert_signals_df)
    return analyze_transactions(enriched)


@st.cache_data(show_spinner=False)
def run_cases(analyzed_df: pd.DataFrame, customers_df, alerts_df) -> pd.DataFrame:
    return build_cases(analyzed_df, customers_df, alerts_df)


@st.cache_data(show_spinner=False)
def cached_case_narrative(case_id: str, _case: dict) -> str:
    """Cache key is the case id alone — the leading underscore tells Streamlit
    not to hash the payload, which contains unhashable lists."""
    return AI.synthesize_case(_case)


@st.cache_data(show_spinner=False)
def cached_executive_summary(signature: str, _stats: dict, _cases) -> str:
    return AI.generate_executive_summary(_stats, _cases)


# An analyst queue has finite capacity. Because several detectors are
# percentile-based, the number of matches grows with the size of the file — so
# the queue is capped and the honest "X matched, showing top N" is surfaced
# rather than rendering an unreviewable (and browser-killing) table.
ALERT_BUDGET_OPTIONS = [50, 100, 200, 500, 1000]
DEFAULT_ALERT_BUDGET = 200


def tier_badge(tier: str) -> str:
    return {
        "CRITICAL": "badge-critical",
        "HIGH": "badge-high",
        "MEDIUM": "badge-medium",
    }.get(tier, "badge-low")


# ---------------------------------------------------------------------------
# Session state / data loading
# ---------------------------------------------------------------------------
# A page refresh gives us a brand-new session, so on first run of a session we
# pick every previously-loaded source back up from disk before falling back to
# the empty state.
if "raw_data" not in st.session_state:
    restored, labels = restore_all()
    st.session_state.raw_data = restored.get("raw_data", pd.DataFrame())
    st.session_state.customers_data = restored.get("customers_data", pd.DataFrame())
    st.session_state.alerts_data = restored.get("alerts_data", pd.DataFrame())
    st.session_state.source_labels = labels or {}

# Remembers which upload each widget last parsed, so a file isn't re-read on
# every unrelated widget interaction.
if "upload_ids" not in st.session_state:
    st.session_state.upload_ids = {}


def _handle_upload(widget_file, key, loader, label_prefix=""):
    """Parse an upload only when the selection actually changes."""
    if widget_file is None:
        return
    upload_id = (widget_file.name, widget_file.size)
    if st.session_state.upload_ids.get(key) == upload_id:
        return
    try:
        set_active_source(key, loader(widget_file), f"{label_prefix}{widget_file.name}")
        st.session_state.upload_ids[key] = upload_id
        st.rerun()
    except Exception as exc:
        st.error(f"Error loading {widget_file.name}: {exc}")


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/shield.png", width=70)
    st.title("Risk Aggregator")
    st.markdown("---")

    st.subheader("System Configuration")
    if HAS_API:
        st.success(f"AI Connected ({PROVIDER} / {MODEL})")
    else:
        st.info("Local Fallback Engine Active")
        st.caption("Add a `GROQ_API_KEY` or `ANTHROPIC_API_KEY` to the workspace `.env` file to unlock live AI-generated rationale.")

    st.markdown("---")
    st.subheader("Data Ingestion")
    st.caption("Source 1 is required. Sources 2 and 3 are optional — each one "
               "unlocks additional cross-source signals.")

    uploaded_file = st.file_uploader(
        "1 · Transaction ledger *", type=["csv", "json", "txt"], key="up_txn")
    _handle_upload(uploaded_file, "raw_data", load_transactions)

    uploaded_customers = st.file_uploader(
        "2 · Customer / KYC file", type=["csv", "json", "txt"], key="up_cust")
    _handle_upload(uploaded_customers, "customers_data", load_customers)

    uploaded_alerts = st.file_uploader(
        "3 · Analyst notes / adverse media", type=["csv", "json", "txt"], key="up_alert")
    _handle_upload(uploaded_alerts, "alerts_data", load_alerts)

    with st.expander("Or paste transaction data (CSV or JSON)"):
        pasted_text = st.text_area(
            "Paste raw CSV or JSON text here",
            height=150,
            placeholder="transaction_id,account_id,date,amount,merchant,country,transaction_type\nTXN0001,ACC1001,2026-06-01 09:15:00,50000,Amazon India,India,UPI",
        )
        if st.button("Load Pasted Data"):
            try:
                set_active_source("raw_data", load_transactions(pasted_text), "pasted text")
                st.session_state.upload_ids.pop("raw_data", None)
                st.rerun()
            except Exception as e:
                st.error(f"Error parsing pasted data: {e}")

    if os.path.exists(DEFAULT_CSV) and st.button("Load Full Demo (all 3 sources)"):
        set_active_source("raw_data", load_transactions(DEFAULT_CSV), "sample_data.csv (demo)")
        if os.path.exists(DEFAULT_CUSTOMERS_CSV):
            set_active_source("customers_data", load_customers(DEFAULT_CUSTOMERS_CSV),
                              "sample_customers.csv (demo)")
        if os.path.exists(DEFAULT_ALERTS_CSV):
            set_active_source("alerts_data", load_alerts(DEFAULT_ALERTS_CSV),
                              "sample_alerts.csv (demo)")
        st.session_state.upload_ids = {}
        st.rerun()

    # Which sources are live right now, and what each is contributing.
    if not st.session_state.raw_data.empty:
        st.markdown("---")
        st.subheader("Active Sources")
        labels = st.session_state.get("source_labels", {})
        for key, icon, name in [("raw_data", "1", "Transactions"),
                                ("customers_data", "2", "Customer / KYC"),
                                ("alerts_data", "3", "Analyst notes")]:
            frame = st.session_state.get(key, pd.DataFrame())
            if frame is not None and not frame.empty:
                st.markdown(f"**{icon}. {name}** — {len(frame):,} rows  \n"
                            f"<span style='color:#9CA3AF;font-size:0.8rem'>{labels.get(key, '—')}</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:#6B7280'>{icon}. {name} — not loaded</span>",
                            unsafe_allow_html=True)

        if st.button("Clear all loaded data"):
            clear_persisted_dataset()
            for k in SESSION_FILES:
                st.session_state[k] = pd.DataFrame()
            st.session_state.source_labels = {}
            st.session_state.upload_ids = {}
            st.rerun()

# Header banner
st.title("Financial Risk Signal Aggregator")
st.markdown("Synthesize fragmented transaction metrics, account behavior anomalies, and jurisdiction markers into actionable insight — amounts in ₹ (INR).")
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Run risk analysis
# ---------------------------------------------------------------------------
if st.session_state.raw_data.empty:
    st.info(
        "**No data loaded yet.** Use the sidebar to upload a transaction ledger "
        "(CSV/JSON/TXT), paste transaction data directly, or load the bundled demo — "
        "which includes all three sources: transactions, a customer/KYC file, and "
        "free-text analyst notes."
    )
    st.stop()

customers_df = st.session_state.get("customers_data", pd.DataFrame())
alerts_df = st.session_state.get("alerts_data", pd.DataFrame())

# Stage 1 — read the unstructured source. This is the only LLM-cost stage that
# scales with input size, so it is cached and reported separately.
alert_signals = None
if alerts_df is not None and not alerts_df.empty:
    try:
        with st.spinner(f"Reading {len(alerts_df)} analyst note(s) with the language model..."):
            alert_signals, alert_detail = run_alert_extraction(alerts_df)
        st.session_state.alert_detail = alert_detail
    except Exception as e:
        st.warning(f"Alert extraction failed, continuing without it: {e}")
        alert_signals = None

# Stage 2 — correlate + score.
try:
    with st.spinner("Correlating sources and scoring risk signals..."):
        analyzed_df = run_analysis(st.session_state.raw_data, customers_df, alert_signals)
except Exception as e:
    st.error(f"Could not analyze this dataset: {e}")
    st.stop()

if analyzed_df.empty:
    st.warning("No valid transactions remained after cleaning (check date/amount formats).")
    st.stop()

# Stage 3 — roll findings up into account-level cases.
cases_df = run_cases(analyzed_df, customers_df, alerts_df)
stats = portfolio_stats(analyzed_df, cases_df)

total_transactions = len(analyzed_df)
avg_risk_score = analyzed_df["risk_score"].mean()
critical_alerts = len(analyzed_df[analyzed_df["risk_tier"] == "CRITICAL"])
high_alerts = len(analyzed_df[analyzed_df["risk_tier"] == "HIGH"])
medium_alerts = len(analyzed_df[analyzed_df["risk_tier"] == "MEDIUM"])
flagged_rate = ((critical_alerts + high_alerts + medium_alerts) / total_transactions) * 100 if total_transactions else 0.0

active_sources = 1 + int(customers_df is not None and not customers_df.empty) \
                   + int(alerts_df is not None and not alerts_df.empty)

tab_summary, tab_metrics, tab_alert_center, tab_ai_query = st.tabs([
    "Consolidated Summary",
    "Executive Dashboard",
    "Risk Alert Center",
    "AI Search Assistant"
])

# ================= TAB 0: CONSOLIDATED SUMMARY (cases) =================
with tab_summary:
    if active_sources == 1:
        st.info(
            "Running on the transaction ledger alone. Add a **customer/KYC file** and "
            "**analyst notes** in the sidebar to unlock cross-source signals "
            "(`income_inconsistency`, `pep_exposure`, `adverse_media_hit`) and the "
            "corroboration scoring below."
        )

    scol1, scol2, scol3, scol4 = st.columns(4)
    for col, title, val in [
        (scol1, "Sources Correlated", f"{active_sources}/3"),
        (scol2, "Cases Raised", f"{stats.get('total_cases', 0)}"),
        (scol3, "Critical Cases", f"{stats.get('critical_cases', 0)}"),
        (scol4, "Multi-Source Cases", f"{stats.get('multi_source_cases', 0)}"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-val">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("AI Executive Briefing")
    st.caption("A single consolidated view of the whole batch — the themes across cases, "
               "not a per-transaction readout.")

    if cases_df.empty:
        st.info("No transaction triggered a risk signal, so no cases were raised.")
    else:
        signature = f"{len(cases_df)}-{cases_df['case_score_raw'].sum():.0f}-{active_sources}"
        with st.spinner("Synthesizing the executive briefing..."):
            briefing = cached_executive_summary(signature, stats, cases_df)
        st.markdown(f"""<div class="ai-report">{briefing}</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Prioritised Cases")
        st.caption(f"{stats['flagged_transactions']} flagged transactions consolidated into "
                   f"{len(cases_df)} account-level cases, ranked by breadth of evidence and "
                   f"then by how many independent sources corroborate them.")

        case_cols = ["case_id", "account_id", "customer_name", "case_score", "case_tier",
                     "distinct_signals", "corroborating_sources", "sources",
                     "flagged_txns", "flagged_amount", "triggered_rules"]
        case_cols = [c for c in case_cols if c in cases_df.columns]
        st.dataframe(cases_df[case_cols].head(50), use_container_width=True, hide_index=True)

        st.download_button(
            "Download case summary (CSV)",
            cases_to_export(cases_df).to_csv(index=False).encode("utf-8"),
            file_name="risk_case_summary.csv", mime="text/csv")

        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        st.subheader("Case File")
        pick = st.selectbox(
            "Open a case",
            options=cases_df["case_id"].tolist(),
            format_func=lambda cid: (
                f"{cid} · {cases_df.loc[cases_df.case_id == cid, 'account_id'].iloc[0]} · "
                f"{cases_df.loc[cases_df.case_id == cid, 'case_tier'].iloc[0]} "
                f"({cases_df.loc[cases_df.case_id == cid, 'corroborating_sources'].iloc[0]} source/s)"))
        case = cases_df[cases_df["case_id"] == pick].iloc[0].to_dict()

        ccol1, ccol2 = st.columns([1, 1.4])
        with ccol1:
            st.markdown("#### Evidence")
            bits = [
                f"**Account:** `{case['account_id']}`",
                f"**Case score:** `{case['case_score']:.0f}/100`"
                + (f" · uncapped `{case['case_score_raw']:.0f}`" if case["case_score_raw"] > 100 else ""),
                f"**Tier:** <span class=\"{tier_badge(case['case_tier'])}\">{case['case_tier']}</span>",
                f"**Signals:** `{case['distinct_signals']}` across `{case['corroborating_sources']}` source(s)",
                f"**Sources:** {case.get('sources', '—')}",
                f"**Flagged:** `{case['flagged_txns']}` of `{case['total_txns']}` txns · "
                f"`{format_inr(case['flagged_amount'])}`",
                f"**Window:** `{case['first_seen']}` → `{case['last_seen']}`",
                f"**Rules:** `{case['triggered_rules']}`",
            ]
            if case.get("customer_name"):
                bits.insert(1, f"**Customer:** {case['customer_name']}"
                               + (f" · {case['occupation']}" if case.get("occupation") else ""))
            if pd.notna(case.get("declared_income_annual")):
                bits.append(f"**Declared income:** `{format_inr(case['declared_income_annual'])}`")
            if case.get("kyc_risk_rating"):
                bits.append(f"**KYC rating at onboarding:** `{case['kyc_risk_rating']}`"
                            + ("  ·  **PEP**" if case.get("pep_flag") is True else ""))
            st.markdown("  \n".join(bits), unsafe_allow_html=True)

            if case.get("analyst_notes"):
                st.markdown("#### Analyst notes (unstructured source)")
                for n in case["analyst_notes"]:
                    st.markdown(f"<div class='ai-report' style='padding:12px;font-size:0.86rem'>{n}</div>",
                                unsafe_allow_html=True)

        with ccol2:
            st.markdown("#### AI Case Assessment")
            with st.spinner("Synthesizing case narrative..."):
                narrative = cached_case_narrative(case["case_id"], case)
            st.markdown(f"""<div class="ai-report">{narrative}</div>""", unsafe_allow_html=True)

# ================= TAB 1: EXECUTIVE DASHBOARD =================
with tab_metrics:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Processed Transactions</div>
            <div class="metric-val">{total_transactions}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Risk Score</div>
            <div class="metric-val">{avg_risk_score:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Critical Threats</div>
            <div class="metric-val" style="background: linear-gradient(135deg, #EF4444 0%, #F87171 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{critical_alerts}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Flagged Rate</div>
            <div class="metric-val">{flagged_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    gcol1, gcol2 = st.columns([1, 1])
    with gcol1:
        st.subheader("Risk Rating Distribution")
        tier_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        tier_counts = analyzed_df["risk_tier"].value_counts().reindex(tier_order, fill_value=0).reset_index()
        tier_counts.columns = ["Risk Tier", "Count"]
        st.bar_chart(tier_counts, x="Risk Tier", y="Count", color="#3B82F6")

    with gcol2:
        st.subheader("Transaction Magnitude vs. Risk Matrix")
        # A scatter plot renders one DOM mark per row, so plotting a large
        # dataset in full will hang the browser. Every flagged transaction is
        # always kept; only the unflagged bulk is sampled to fill in the
        # background distribution.
        SCATTER_MAX_POINTS = 4000
        if len(analyzed_df) > SCATTER_MAX_POINTS:
            flagged = analyzed_df[analyzed_df["risk_score"] > 0]
            unflagged = analyzed_df[analyzed_df["risk_score"] == 0]
            room = max(0, SCATTER_MAX_POINTS - len(flagged))
            if len(unflagged) > room:
                unflagged = unflagged.sample(room, random_state=0)
            plot_df = pd.concat([flagged.head(SCATTER_MAX_POINTS), unflagged])
            st.caption(
                f"Showing {len(plot_df):,} of {len(analyzed_df):,} points "
                f"(all flagged transactions + a random sample of unflagged)."
            )
        else:
            plot_df = analyzed_df
        st.scatter_chart(plot_df, x="date_time", y="amount", color="risk_tier")

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Risk Signal Composition")
    exploded = analyzed_df.explode("flag_reasons").dropna(subset=["flag_reasons"])
    if not exploded.empty:
        rule_counts = exploded["flag_reasons"].value_counts().reset_index()
        rule_counts.columns = ["Rule", "Occurrences"]
        st.bar_chart(rule_counts, x="Rule", y="Occurrences", color="#8B5CF6", horizontal=True)
    else:
        st.info("No detector rules triggered on this dataset.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("High-Risk Accounts Profile")
    grouped = analyzed_df.groupby("account_id").agg(
        total_tx=("transaction_id", "count"),
        avg_risk=("risk_score", "mean"),
        critical_count=("risk_tier", lambda x: sum(x == "CRITICAL")),
        high_count=("risk_tier", lambda x: sum(x == "HIGH")),
    ).reset_index()

    danger_accounts = grouped[grouped["avg_risk"] > 15].sort_values(by="avg_risk", ascending=False)

    if not danger_accounts.empty:
        display_accounts = danger_accounts.rename(columns={
            "account_id": "Account ID",
            "total_tx": "Total Transactions",
            "avg_risk": "Average Risk Rating",
            "critical_count": "Critical Signals",
            "high_count": "High Signals",
        })
        display_accounts["Average Risk Rating"] = display_accounts["Average Risk Rating"].round(1)
        st.dataframe(display_accounts, use_container_width=True, hide_index=True)
    else:
        st.info("No high-risk accounts identified in this dataset.")

# ================= TAB 2: RISK ALERT CENTER =================
with tab_alert_center:
    st.subheader("Heuristic Alerts Requiring Compliance Attention")
    st.markdown("Filter and select any transaction row to generate an automated AI compliance briefing.")

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        tier_filter = st.multiselect("Filter by Severity Tier", options=["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH", "MEDIUM"])
    with fcol2:
        acct_filter = st.text_input("Filter by Account ID (optional)")

    filtered_df = analyzed_df[analyzed_df["risk_tier"].isin(tier_filter)]
    if acct_filter:
        filtered_df = filtered_df[filtered_df["account_id"].str.contains(acct_filter, case=False)]

    if filtered_df.empty:
        st.info("No flagged transactions match the selected filters.")
    else:
        display_cols = ["transaction_id", "account_id", "date_time", "amount", "balance", "merchant", "country",
                         "transaction_type", "risk_score", "risk_tier", "triggered_rules"]
        display_cols = [c for c in display_cols if c in filtered_df.columns]

        # Cap the review queue. analyzed_df is already sorted by risk_score_raw
        # descending, so .head() is genuinely "the highest-priority N", not an
        # arbitrary slice.
        matched_count = len(filtered_df)
        budget = st.selectbox(
            "Alert queue size (highest-priority first)",
            options=ALERT_BUDGET_OPTIONS,
            index=ALERT_BUDGET_OPTIONS.index(DEFAULT_ALERT_BUDGET),
        )
        queue_df = filtered_df.head(budget)
        if matched_count > len(queue_df):
            st.caption(
                f"**{matched_count:,}** transactions matched these filters — showing the "
                f"top **{len(queue_df):,}** by risk score. Narrow the filters, or raise the "
                f"queue size, to see more."
            )
        else:
            st.caption(f"**{matched_count:,}** transactions matched these filters.")

        st.markdown("#### Select a transaction for real-time investigation:")
        tx_options = queue_df["transaction_id"].tolist()
        selected_tx_id = st.selectbox("Choose Transaction ID to Investigate:", options=tx_options)

        selected_row = queue_df[queue_df["transaction_id"] == selected_tx_id].iloc[0].to_dict()

        dcol1, dcol2 = st.columns([1, 1.3])

        with dcol1:
            st.markdown("### Transaction Dossier")
            balance_val = selected_row.get("balance")
            balance_line = f"\n            **Balance After:** `{format_inr(balance_val)}`" if pd.notna(balance_val) else ""
            # Surface the uncapped score when it exceeds the display cap — it's
            # the difference between "just reached CRITICAL" and "far beyond it".
            raw_val = selected_row.get("risk_score_raw", selected_row["risk_score"])
            raw_note = f" · uncapped `{raw_val:.0f}`" if raw_val > 100 else ""
            st.markdown(f"""
            **Transaction ID:** `{selected_row['transaction_id']}`
            **Account ID:** `{selected_row['account_id']}`
            **Risk Tier:** <span class="{tier_badge(selected_row['risk_tier'])}">{selected_row['risk_tier']}</span> (Score: `{selected_row['risk_score']:.0f}/100`{raw_note})
            **Date/Time:** `{selected_row['date_time']}`
            **Amount:** `{format_inr(selected_row['amount'])}`
            **Merchant:** `{selected_row['merchant']}`
            **Country:** `{selected_row['country']}`
            **Transaction Type:** `{selected_row.get('transaction_type', 'N/A')}`{balance_line}
            **Triggered Rules:** `{selected_row['triggered_rules']}`
            """, unsafe_allow_html=True)

        with dcol2:
            st.markdown("### AI Reasoning & Assessment")
            report_placeholder = st.empty()
            report_placeholder.markdown("*Generating intelligent analysis...*")

            explanation = generate_rationale(selected_row)

            report_placeholder.markdown(f"""
            <div class="ai-report">
                {explanation}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        st.markdown("#### Prioritised Alert Queue")
        st.dataframe(queue_df[display_cols], use_container_width=True, hide_index=True)

# ================= TAB 3: AI SEARCH ASSISTANT =================
with tab_ai_query:
    st.subheader("Intuitive Natural Language Search")
    st.markdown("Query your transaction risk database using plain text. "
                "*(e.g., 'Show critical alerts in Russia', 'Show account ACC100005's flagged transactions', 'transactions over 500000')*")

    query = st.text_input("Enter your request:", placeholder="Search transaction risk records...")

    if query:
        keywords = query.lower()
        results = analyzed_df.copy()
        matched = False

        for country in analyzed_df["country"].unique():
            if str(country).lower() in keywords:
                results = results[results["country"].str.lower() == str(country).lower()]
                matched = True

        acct_match = re.search(r'acc\d+', keywords)
        if acct_match:
            results = results[results["account_id"].str.lower() == acct_match.group(0)]
            matched = True

        for tier in ["critical", "high", "medium", "low"]:
            if tier in keywords:
                results = results[results["risk_tier"].str.lower() == tier]
                matched = True

        if "over" in keywords or "greater than" in keywords or "more than" in keywords or ">" in keywords:
            amt_match = re.search(r'\d+', keywords)
            if amt_match:
                results = results[results["amount"] > float(amt_match.group(0))]
                matched = True
        elif "under" in keywords or "less than" in keywords or "<" in keywords:
            amt_match = re.search(r'\d+', keywords)
            if amt_match:
                results = results[results["amount"] < float(amt_match.group(0))]
                matched = True

        if not matched:
            results = results[
                results["account_id"].str.contains(query, case=False) |
                results["country"].str.contains(query, case=False) |
                results["merchant"].str.contains(query, case=False) |
                results["triggered_rules"].str.contains(query, case=False)
            ]

        st.markdown(f"**Found {len(results)} matching records:**")
        if not results.empty:
            search_cols = ["transaction_id", "account_id", "date_time", "amount", "country", "risk_score", "risk_tier", "triggered_rules"]
            search_cols = [c for c in search_cols if c in results.columns]
            st.dataframe(results[search_cols], use_container_width=True, hide_index=True)

            with st.spinner("AI is formulating a summary of the search results..."):
                try:
                    results_summary = results[search_cols].to_dict(orient="records")
                    resp_text = generate_search_summary(query, results_summary)
                    st.markdown(f"""
                    <div class="ai-report" style="border-color: rgba(59, 130, 246, 0.3);">
                        <strong>Search Summary:</strong><br>{resp_text}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.caption(f"(AI summary unavailable: {e})")
        else:
            st.warning("No records matched your search terms. Try refining your keywords (e.g. specify countries, risk tier names, or exact account IDs).")

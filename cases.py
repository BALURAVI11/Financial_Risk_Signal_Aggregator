"""
Case Builder
-------------
Rolls transaction-level findings up into ACCOUNT-level cases.

Why this layer exists: compliance analysts do not work transactions, they work
cases. A single account trickling four different red flags across six
transactions is one investigation, not six alerts — and it is materially more
suspicious than one transaction tripping the same rule four times. Presenting
the transaction list alone pushes that correlation work back onto the analyst,
which is exactly the manual burden the brief describes.

Case scoring therefore sums the weights of the DISTINCT rules triggered
anywhere in the account, rather than taking a single transaction's score:
  - it measures breadth of evidence, not repetition of one signal;
  - it is unaffected by how many transactions the account happens to have;
  - it naturally rewards corroboration across independent sources, since a
    ledger rule, a KYC rule and an adverse-media rule each add their own weight.
"""

import numpy as np
import pandas as pd

from risk_engine import RULE_WEIGHTS, TIER_BANDS, score_to_tier, CROSS_SOURCE_DETECTORS

# Which upstream source each rule depends on — used to report how many
# independent sources corroborate a case.
KYC_RULES = {"income_inconsistency", "pep_exposure"}
ALERT_RULES = {"adverse_media_hit"}


def _distinct_str(series, limit: int = 4) -> str:
    """Distinct non-null values as a display string. Real extracts contain
    blanks (the bundled edge-case file has a row with no country), and NaN is a
    float — sorting it alongside strings raises, so it is dropped first."""
    vals = sorted({str(v) for v in series.dropna() if str(v).strip().lower() != "nan"})
    return ", ".join(vals[:limit])


def _sources_for(rules) -> list:
    src = []
    if any(r not in CROSS_SOURCE_DETECTORS for r in rules):
        src.append("transactions")
    if any(r in KYC_RULES for r in rules):
        src.append("customer/KYC")
    if any(r in ALERT_RULES for r in rules):
        src.append("analyst notes")
    return src


# A "case" must clear the MEDIUM band. Without this floor almost every account
# becomes a case, because ambient signals like odd_hours fire on roughly a
# quarter of all transactions — 46 cases out of 50 accounts is not a triaged
# queue, it is the account list with extra steps.
MIN_CASE_SCORE = next(t for t, name in TIER_BANDS if name == "MEDIUM")


def build_cases(analyzed_df: pd.DataFrame,
                customers_df: pd.DataFrame = None,
                alerts_df: pd.DataFrame = None,
                min_case_score: float = MIN_CASE_SCORE) -> pd.DataFrame:
    """Aggregate flagged transactions into one row per account."""
    if analyzed_df.empty:
        return pd.DataFrame()

    flagged = analyzed_df[analyzed_df["risk_score"] > 0]
    if flagged.empty:
        return pd.DataFrame()

    # Annualised throughput per account — the same basis income_inconsistency
    # uses, so the case file can state the actual ratio instead of leaving the
    # narrative to infer it from raw totals (and get it wrong).
    span_days = max((analyzed_df["date_time"].max() - analyzed_df["date_time"].min()).days, 1)
    annualised_by_acct = analyzed_df.groupby("account_id")["amount"].sum() * 365.0 / span_days

    txn_counts = analyzed_df.groupby("account_id").size()
    records = []

    for acct, group in flagged.groupby("account_id"):
        # Union of every distinct rule this account tripped, anywhere.
        rules = sorted({r for reasons in group["flag_reasons"] for r in reasons})
        raw = float(sum(RULE_WEIGHTS.get(r, 0) for r in rules))
        sources = _sources_for(rules)

        rec = {
            "case_id": "",                     # assigned after ranking
            "account_id": acct,
            "case_score_raw": raw,
            "case_score": min(100.0, raw),
            "case_tier": score_to_tier(min(100.0, raw)),
            "distinct_signals": len(rules),
            "triggered_rules": ", ".join(rules),
            "rules_list": rules,
            "corroborating_sources": len(sources),
            "sources": ", ".join(sources),
            "flagged_txns": int(len(group)),
            "total_txns": int(txn_counts.get(acct, len(group))),
            "flagged_amount": float(group["amount"].sum()),
            "total_amount": float(analyzed_df.loc[analyzed_df["account_id"] == acct, "amount"].sum()),
            "peak_txn_score": float(group["risk_score_raw"].max()),
            "first_seen": group["date_time"].min(),
            "last_seen": group["date_time"].max(),
            "top_countries": _distinct_str(group["country"]),
            "top_merchants": _distinct_str(group["merchant"]),
        }

        rec["annualised_throughput"] = float(annualised_by_acct.get(acct, 0.0))

        # Fold in whatever customer context is available.
        if customers_df is not None and not customers_df.empty:
            match = customers_df[customers_df["account_id"] == acct]
            if not match.empty:
                c = match.iloc[0]
                for field in ("customer_name", "occupation", "kyc_risk_rating",
                              "declared_income_annual", "pep_flag"):
                    if field in match.columns:
                        rec[field] = c[field]
                income = pd.to_numeric(pd.Series([c.get("declared_income_annual")]),
                                       errors="coerce").iloc[0]
                if pd.notna(income) and income > 0:
                    rec["income_ratio"] = rec["annualised_throughput"] / float(income)

        # ...and the raw analyst notes, so the narrative can quote them.
        if alerts_df is not None and not alerts_df.empty:
            notes = alerts_df[alerts_df["account_id"] == acct]["note"].tolist()
            rec["analyst_notes"] = notes
            rec["analyst_note_count"] = len(notes)
        else:
            rec["analyst_notes"] = []
            rec["analyst_note_count"] = 0

        records.append(rec)

    cases = pd.DataFrame(records)
    cases = cases[cases["case_score_raw"] >= min_case_score]
    if cases.empty:
        return cases
    # Rank by breadth of evidence first, then by how many independent sources
    # agree — a case corroborated by three sources outranks an equally-scored
    # one supported only by the ledger.
    cases = cases.sort_values(
        ["case_score_raw", "corroborating_sources", "distinct_signals"],
        ascending=False).reset_index(drop=True)
    cases["case_id"] = [f"CASE-{i + 1:04d}" for i in range(len(cases))]
    return cases


def portfolio_stats(analyzed_df: pd.DataFrame, cases: pd.DataFrame) -> dict:
    """Headline numbers for the consolidated summary."""
    if analyzed_df.empty:
        return {}
    tier_counts = cases["case_tier"].value_counts().to_dict() if not cases.empty else {}
    exploded = analyzed_df.explode("flag_reasons").dropna(subset=["flag_reasons"])
    top_rules = (exploded["flag_reasons"].value_counts().head(5).to_dict()
                 if not exploded.empty else {})
    return {
        "total_transactions": int(len(analyzed_df)),
        "total_accounts": int(analyzed_df["account_id"].nunique()),
        "flagged_transactions": int((analyzed_df["risk_score"] > 0).sum()),
        "total_cases": int(len(cases)),
        "critical_cases": int(tier_counts.get("CRITICAL", 0)),
        "high_cases": int(tier_counts.get("HIGH", 0)),
        "medium_cases": int(tier_counts.get("MEDIUM", 0)),
        "low_cases": int(tier_counts.get("LOW", 0)),
        "multi_source_cases": int((cases["corroborating_sources"] > 1).sum()) if not cases.empty else 0,
        "date_from": analyzed_df["date_time"].min(),
        "date_to": analyzed_df["date_time"].max(),
        "flagged_value": float(analyzed_df.loc[analyzed_df["risk_score"] > 0, "amount"].sum()),
        "top_rules": top_rules,
    }


def cases_to_export(cases: pd.DataFrame) -> pd.DataFrame:
    """Flatten to analyst-friendly columns for CSV download."""
    if cases.empty:
        return cases
    cols = ["case_id", "account_id", "customer_name", "case_score", "case_tier",
            "distinct_signals", "corroborating_sources", "sources", "triggered_rules",
            "flagged_txns", "total_txns", "flagged_amount", "kyc_risk_rating",
            "pep_flag", "declared_income_annual", "first_seen", "last_seen",
            "ai_case_narrative", "recommended_action"]
    return cases[[c for c in cols if c in cases.columns]]

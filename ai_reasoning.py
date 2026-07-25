
"""
AI Reasoning Layer
-------------------
Takes flagged transactions (output of risk_engine.analyze_transactions) and
asks an LLM to generate a short, human-readable risk explanation for each
one, grounded in which rules actually fired.

Usage:
    from ai_reasoning import add_ai_rationale, generate_rationale, generate_search_summary
    flagged_df = add_ai_rationale(flagged_df)

Requires:
    pip install groq anthropic python-dotenv
    A .env file with either: GROQ_API_KEY=your_key_here
                          or: ANTHROPIC_API_KEY=your_key_here

If neither key is present, the module automatically falls back to a
deterministic local explanation template (no network calls, no cost) so the
rest of the app keeps working end-to-end.
"""

import json
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv

from currency import format_inr

load_dotenv()

PROVIDER = os.environ.get("AI_PROVIDER", "groq").lower()   # "groq" or "anthropic"
MODEL = os.environ.get(
    "AI_MODEL",
    "llama-3.3-70b-versatile" if PROVIDER == "groq" else "claude-sonnet-4-6",
)
MAX_RETRIES = 3

GROQ_KEY_PRESENT = bool(os.environ.get("GROQ_API_KEY", "").strip())
ANTHROPIC_KEY_PRESENT = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())

# Auto-detect: use whichever provider actually has a key configured, falling
# back to the local template if neither is set. This means the app "just
# works" the moment a user drops a key into .env, without editing code.
if PROVIDER == "groq" and not GROQ_KEY_PRESENT and ANTHROPIC_KEY_PRESENT:
    PROVIDER = "anthropic"
    MODEL = os.environ.get("AI_MODEL", "claude-sonnet-4-6")
elif PROVIDER == "anthropic" and not ANTHROPIC_KEY_PRESENT and GROQ_KEY_PRESENT:
    PROVIDER = "groq"
    MODEL = os.environ.get("AI_MODEL", "llama-3.3-70b-versatile")

HAS_API = (PROVIDER == "groq" and GROQ_KEY_PRESENT) or (PROVIDER == "anthropic" and ANTHROPIC_KEY_PRESENT)
DRY_RUN = not HAS_API

# Human-readable descriptions of each detector, used to give the AI context
# on what each flag actually means (better than just passing the raw name).
FLAG_DESCRIPTIONS = {
    "card_testing": "multiple very small transactions across different merchants in a short window, consistent with a stolen card being tested",
    "high_velocity": "an unusually high number of transactions in a short time window for this account",
    "rapid_drain": "a large deposit immediately followed by multiple withdrawals, draining most of the balance",
    "structuring": "repeated transfers just under the ₹10,00,000 reporting threshold, spread across a few days",
    "amount_volatility": "an amount far above this account's typical transaction size",
    "round_number": "a suspiciously large, exact round-number transfer",
    "high_risk_country": "involves a country on a high-risk/watchlist jurisdiction list",
    "foreign_transaction": "took place outside the account's home country, which is worth noting even when the country itself isn't on any watchlist",
    "impossible_travel": "two transactions in different countries too close together in time to be physically possible",
    "high_risk_merchant": "a merchant in a high-risk category (e.g. gambling, unregulated crypto, offshore services)",
    "fan_in": "many different accounts sending funds to the same beneficiary in a short window, consistent with a mule account",
    "odd_hours": "a transaction initiated during unusual overnight hours",
    "dormant_awakening": "a large transaction on an account that had been inactive for a while",
    "new_merchant_large_amount": "a large payment to a merchant this account has never paid before",
    "new_account_large_first_txn": "an unusually large amount for what is this account's very first recorded transaction",
    "daily_frequency_spike": "a day with far more transactions than this account's typical daily pattern",
    "circular_transfer": "funds moving back and forth between the same two accounts",
    # Cross-source signals — each requires evidence the transaction ledger alone
    # does not contain, which is why they carry extra weight in a case.
    "income_inconsistency": "the value moved through the account is far out of proportion to the income the customer declared at onboarding, raising an unanswered source-of-funds question",
    "pep_exposure": "the customer is a politically exposed person and is transacting cross-border or at unusually high value",
    "adverse_media_hit": "an analyst note or adverse-media report about this account was assessed as a substantiated, material concern",
}


def _build_prompt(row) -> str:
    reasons = row["flag_reasons"]
    reason_descriptions = "\n".join(
        f"- {r}: {FLAG_DESCRIPTIONS.get(r, 'unusual pattern detected')}" for r in reasons
    )
    return f"""You are a compliance analyst assistant. Write a concise, neutral, professional
risk explanation for the transaction below, based only on the flagged signals provided.
Do not invent facts not present in the data. Keep it to 2 sentences.

Transaction details:
- Amount: {format_inr(row['amount'])}
- Account: {row['account_id']}
- Merchant: {row['merchant']}
- Country: {row['country']}
- Date/time: {row['date_time']}
- Risk tier: {row['risk_tier']} (score {row['risk_score']}/100)

Flagged signals:
{reason_descriptions}

Write the risk explanation now (2 sentences, no preamble):"""


def _call_claude(prompt: str, max_tokens: int = 200) -> str:
    from anthropic import Anthropic

    client = Anthropic()  # reads ANTHROPIC_API_KEY from environment automatically
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return f"[AI rationale unavailable — error: {e}]"
            time.sleep(2 ** attempt)  # exponential backoff


def _call_groq(prompt: str, max_tokens: int = 200) -> str:
    from groq import Groq

    client = Groq()  # reads GROQ_API_KEY from environment automatically
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return f"[AI rationale unavailable — error: {e}]"
            time.sleep(2 ** attempt)  # exponential backoff


def _call_ai(prompt: str, max_tokens: int = 200) -> str:
    """Single entry point for every LLM call. `max_tokens` is per-call because a
    two-sentence transaction rationale and a full case narrative have very
    different budgets — a shared cap silently truncated the longer ones."""
    if PROVIDER == "groq":
        return _call_groq(prompt, max_tokens)
    return _call_claude(prompt, max_tokens)


# ===========================================================================
# UNSTRUCTURED SOURCE — free text -> validated structured signals
# ===========================================================================
# This is the part of the pipeline the rules engine genuinely cannot do. A
# pandas detector can compare numbers; it cannot read "the customer was evasive
# about the purpose and declined to provide invoices" and decide that outranks
# "threshold alert auto-closed, documented festival spending". The LLM converts
# analyst prose into a scored, machine-readable signal that then feeds the same
# weighted scoring model as every other detector.

_EXTRACTION_SCHEMA = """{
  "severity": <integer 0-100>,
  "indicators": [<1-4 short lower_snake_case labels>],
  "summary": "<one neutral sentence>"
}"""


def _build_extraction_prompt(note: str, account_id: str) -> str:
    return f"""You are a financial-crime analyst triaging a free-text case note.

Return ONLY a JSON object — no prose, no markdown — matching exactly:
{_EXTRACTION_SCHEMA}

Severity guidance:
    0-25   routine or administrative; explicitly cleared; acknowledged false positive
   26-55   worth noting but UNCONFIRMED, or already explained/mitigated in the note
   56-75   a substantiated concern that needs analyst review
  76-100   severe, specific AND substantiated (sanctions exposure, mule network,
           confirmed fraud, refusal to evidence source of funds)

Rules:
- Judge only what the note actually states. Do not infer beyond it.
- If the note says a match is NOT confirmed, that an alert was closed as a false
  positive, or that the activity was satisfactorily explained, severity MUST be low.
- "indicators" are short labels such as sanctions_exposure, mule_network,
  source_of_funds_unverified, account_takeover, no_concern.

Account: {account_id}
Note:
\"\"\"{note}\"\"\"

JSON:"""


def _parse_json_object(raw: str) -> dict:
    """Pull a JSON object out of a model response, tolerating markdown fences
    or a stray sentence either side of it."""
    if not raw:
        raise ValueError("empty response")
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object found in response: {raw[:120]}")
        return json.loads(match.group(0))


def _validate_alert_signal(obj: dict) -> dict:
    """Coerce the model's output into the contract the risk engine expects.
    Never trust the shape of an LLM response — a malformed severity silently
    becoming a string would corrupt the downstream score comparison."""
    severity = obj.get("severity", 0)
    try:
        severity = int(round(float(severity)))
    except (TypeError, ValueError):
        severity = 0
    severity = max(0, min(100, severity))

    indicators = obj.get("indicators") or []
    if isinstance(indicators, str):
        indicators = [indicators]
    indicators = [str(i).strip().lower().replace(" ", "_") for i in indicators if str(i).strip()][:4]

    summary = str(obj.get("summary", "")).strip()
    return {"severity": severity, "indicators": indicators, "summary": summary}


# Keyword fallback used when no API key is configured, so the unstructured leg
# still demonstrably works offline. Deliberately ordered: a clearing phrase
# ("false positive", "no concern") wins over an alarming one, because those
# notes describe an alert that was already dispositioned.
_CLEARED_PHRASES = ["false positive", "no concern", "no concerns", "no action required",
                    "auto-closed", "consistent with declared", "identity verified"]
_SEVERE_PHRASES = {"sanction": 88, "mule": 85, "funnel": 82, "evasive": 78, "counterfeit": 85,
                   "cloned": 82, "not physically possible": 84, "account takeover": 80,
                   "unlicensed": 74, "under investigation": 72, "declined to provide": 78}
_MODERATE_PHRASES = {"adverse media": 58, "no source-of-funds": 68, "not confirmed": 45,
                     "stale": 52, "unreachable": 58, "inconsistent": 62,
                     "analyst judgement required": 48, "step-up verification": 60}


def _fallback_alert_signal(note: str) -> dict:
    low = note.lower()
    if any(p in low for p in _CLEARED_PHRASES):
        return {"severity": 10, "indicators": ["no_concern"],
                "summary": "Note indicates the alert was reviewed and cleared."}
    best, label = 0, None
    for phrase, score in {**_SEVERE_PHRASES, **_MODERATE_PHRASES}.items():
        if phrase in low and score > best:
            best, label = score, phrase.replace(" ", "_").replace("-", "_")
    if best == 0:
        return {"severity": 20, "indicators": ["unclassified"],
                "summary": "No recognised risk language found in the note."}
    return {"severity": best, "indicators": [label],
            "summary": f"Keyword fallback matched '{label.replace('_', ' ')}' in the note."}


def extract_alert_signals(note: str, account_id: str = "") -> dict:
    """Turn one free-text note into {severity, indicators, summary}.

    Falls back to the keyword heuristic when no API key is set, and also when
    the model returns something that will not validate — the pipeline must
    always yield a usable signal rather than propagate a parse error."""
    if DRY_RUN:
        result = _fallback_alert_signal(note)
        result["extraction_mode"] = "keyword-fallback"
        return result

    prompt = _build_extraction_prompt(note, account_id or "unknown")
    for attempt in range(2):          # one retry: LLMs occasionally emit prose
        raw = _call_ai(prompt)
        try:
            result = _validate_alert_signal(_parse_json_object(raw))
            result["extraction_mode"] = f"llm:{PROVIDER}"
            return result
        except (ValueError, json.JSONDecodeError):
            if attempt == 0:
                prompt += "\n\nYour previous reply was not valid JSON. Reply with the JSON object only."
                continue
    result = _fallback_alert_signal(note)
    result["extraction_mode"] = "keyword-fallback (LLM output unparseable)"
    return result


def extract_alert_signals_batch(alerts_df, progress_cb=None):
    """Run extraction across an alerts table and roll the results up to one row
    per account, shaped for data_loader.enrich_transactions().

    An account may have several notes; severity takes the MAX (the worst
    substantiated concern drives the case) while indicators and summaries are
    merged so no note's content is lost."""
    import pandas as pd

    if alerts_df is None or alerts_df.empty:
        return pd.DataFrame(columns=["account_id", "adverse_media_severity",
                                     "adverse_media_indicators", "adverse_media_summary"])

    records = []
    total = len(alerts_df)
    for i, (_, row) in enumerate(alerts_df.iterrows()):
        signal = extract_alert_signals(row["note"], row.get("account_id", ""))
        records.append({
            "account_id": row["account_id"],
            "severity": signal["severity"],
            "indicators": signal["indicators"],
            "summary": signal["summary"] or row["note"][:160],
            "extraction_mode": signal.get("extraction_mode", "unknown"),
        })
        if progress_cb:
            progress_cb(i + 1, total)

    per_note = pd.DataFrame(records)
    grouped = per_note.groupby("account_id").agg(
        adverse_media_severity=("severity", "max"),
        adverse_media_indicators=("indicators", lambda s: ", ".join(
            dict.fromkeys(i for lst in s for i in lst))),
        adverse_media_summary=("summary", lambda s: " | ".join(s)),
        adverse_media_note_count=("summary", "size"),
    ).reset_index()
    return grouped, per_note


# ===========================================================================
# CASE SYNTHESIS — the brief's "synthesize fragmented signals into a coherent,
# prioritised risk view", executed. Everything above produces individual
# signals; this reasons over ALL of an account's evidence, across ALL sources,
# and states what it collectively means and what should happen next.
# ===========================================================================
def _build_case_prompt(case: dict) -> str:
    ctx = []
    if case.get("customer_name"):
        ctx.append(f"- Customer: {case['customer_name']}")
    if case.get("occupation"):
        ctx.append(f"- Stated occupation: {case['occupation']}")
    if case.get("declared_income_annual") and pd.notna(case.get("declared_income_annual")):
        ctx.append(f"- Declared annual income: {format_inr(case['declared_income_annual'])}")
        # State the comparison explicitly. Given only a raw flagged total and an
        # income figure, the model was comparing those two directly and drawing
        # the wrong conclusion — the detector actually compares ANNUALISED
        # throughput to income, which is a different (and much larger) number.
        if case.get("income_ratio") and pd.notna(case.get("income_ratio")):
            ctx.append(
                f"- Annualised throughput through the account: "
                f"{format_inr(case['annualised_throughput'])}, which is "
                f"{case['income_ratio']:.1f}x the declared annual income "
                f"(this ratio, not the raw flagged total, is what "
                f"income_inconsistency measures)")
    if case.get("kyc_risk_rating"):
        ctx.append(f"- KYC risk rating at onboarding: {case['kyc_risk_rating']}")
    if case.get("pep_flag") is True:
        ctx.append("- Politically exposed person: YES")
    customer_block = "\n".join(ctx) if ctx else "- No customer/KYC record available for this account."

    rules = case.get("rules_list") or []
    signal_block = "\n".join(
        f"- {r}: {FLAG_DESCRIPTIONS.get(r, 'unusual pattern detected')}" for r in rules)

    notes = case.get("analyst_notes") or []
    notes_block = ("\n".join(f'- "{n}"' for n in notes[:3])
                   if notes else "- No analyst notes on file for this account.")

    return f"""You are a senior financial-crime analyst writing the summary section of a case file.

ACCOUNT: {case['account_id']}
Case score {case['case_score']:.0f}/100 ({case['case_tier']}), built from
{case['distinct_signals']} distinct signals corroborated across
{case['corroborating_sources']} independent source(s): {case.get('sources', 'transactions')}.

CUSTOMER CONTEXT
{customer_block}

ACTIVITY
- {case['flagged_txns']} of {case['total_txns']} transactions flagged
- Flagged value: {format_inr(case['flagged_amount'])}
- Window: {case['first_seen']} to {case['last_seen']}
- Counterparties: {case.get('top_merchants', 'n/a')}
- Jurisdictions: {case.get('top_countries', 'n/a')}

AUTOMATED SIGNALS
{signal_block}

ANALYST NOTES ON FILE
{notes_block}

Write exactly two short paragraphs, no headings, no bullet points:
1. What the combined evidence indicates — explicitly connect signals from
   DIFFERENT sources where they reinforce each other. Say plainly if the
   sources corroborate one another or merely coexist.
2. The recommended next step for the compliance team, and any material
   caveat or exculpatory point present in the evidence.

Ground every statement in the evidence above. Do not invent transactions,
names, dates or allegations. If the evidence is thin or ambiguous, say so
rather than overstating it."""


def _fallback_case_narrative(case: dict) -> str:
    return (
        f"**{case['case_tier']}** case ({case['case_score']:.0f}/100) on account "
        f"{case['account_id']} — {case['distinct_signals']} distinct signals across "
        f"{case['corroborating_sources']} source(s): {case.get('sources', 'transactions')}. "
        f"{case['flagged_txns']} of {case['total_txns']} transactions flagged, totalling "
        f"{format_inr(case['flagged_amount'])}. Rules triggered: {case.get('triggered_rules', '')}. "
        f"Add a GROQ_API_KEY or ANTHROPIC_API_KEY to `.env` for an AI-written case narrative."
    )


def synthesize_case(case: dict) -> str:
    """Produce the narrative for one case from all of its evidence."""
    if DRY_RUN:
        return _fallback_case_narrative(case)
    return _call_ai(_build_case_prompt(case), max_tokens=600)


def _fallback_executive_summary(stats: dict, cases) -> str:
    top = "; ".join(
        f"{c['account_id']} ({c['case_tier']}, {c['distinct_signals']} signals)"
        for _, c in cases.head(5).iterrows()) if len(cases) else "none"
    return (
        f"**Portfolio summary** — {stats['flagged_transactions']} of "
        f"{stats['total_transactions']} transactions across {stats['total_accounts']} accounts "
        f"were flagged, consolidating into {stats['total_cases']} cases "
        f"({stats['critical_cases']} critical, {stats['high_cases']} high). "
        f"{stats['multi_source_cases']} case(s) are corroborated by more than one source. "
        f"Highest priority: {top}. "
        f"Add an API key to `.env` for an AI-written executive briefing."
    )


def generate_executive_summary(stats: dict, cases, top_n: int = 8) -> str:
    """The brief's named deliverable: one consolidated risk summary over the
    whole dataset, rather than a per-transaction rationale the analyst has to
    assemble themselves."""
    if cases is None or len(cases) == 0:
        return "No cases were raised for this dataset — no transaction triggered a risk signal."
    if DRY_RUN:
        return _fallback_executive_summary(stats, cases)

    lines = []
    for _, c in cases.head(top_n).iterrows():
        who = f" [{c['customer_name']}]" if c.get("customer_name") else ""
        lines.append(
            f"- {c['case_id']} {c['account_id']}{who}: {c['case_tier']} "
            f"{c['case_score']:.0f}/100, {c['distinct_signals']} signals across "
            f"{c['corroborating_sources']} source(s) — {c['triggered_rules']} "
            f"({c['flagged_txns']} txns, {format_inr(c['flagged_amount'])})")
    case_block = "\n".join(lines)

    rules_block = ", ".join(f"{k} ({v})" for k, v in (stats.get("top_rules") or {}).items())

    return _call_ai(f"""You are the head of a financial-crime unit briefing the compliance
committee on one batch of monitoring output. Be specific and concise.

PORTFOLIO
- {stats['total_transactions']} transactions, {stats['total_accounts']} accounts
- Period {stats['date_from']} to {stats['date_to']}
- {stats['flagged_transactions']} transactions flagged, worth {format_inr(stats['flagged_value'])}
- {stats['total_cases']} cases raised: {stats['critical_cases']} critical, {stats['high_cases']} high, {stats['medium_cases']} medium
- {stats['multi_source_cases']} case(s) corroborated by more than one independent source
- Most frequent signals: {rules_block}

HIGHEST-PRIORITY CASES
{case_block}

Write a briefing of at most 200 words:
1. Open with one sentence stating the overall picture.
2. Then identify the 2-3 distinct THEMES or typologies present (for example a
   coordinated funnel network, structuring below a reporting threshold,
   sanctions exposure). Group the related cases under each theme by account ID
   rather than listing cases one by one.
3. Close with the single action you want the committee to authorise first, and why.

Use only the evidence above. Do not invent accounts, amounts or allegations.""",
                    max_tokens=700)


def _dry_run_rationale(row) -> str:
    """Deterministic local fallback used when no API key is configured."""
    reasons = ", ".join(row["flag_reasons"]) if row["flag_reasons"] else "no rule triggers"
    return (
        f"Flagged **{row['risk_tier']}** (score {row['risk_score']}/100) — this {format_inr(row['amount'])} "
        f"transaction on account {row['account_id']} triggered: {reasons}. "
        f"Add a GROQ_API_KEY or ANTHROPIC_API_KEY to `.env` to generate a live AI-written explanation instead of this template."
    )


def generate_rationale(row) -> str:
    if DRY_RUN:
        return _dry_run_rationale(row)
    prompt = _build_prompt(row)
    return _call_ai(prompt)


def add_ai_rationale(flagged_df, limit: int = None):
    """Adds an 'ai_rationale' column to the flagged transactions DataFrame.
    `limit` caps how many rows get a live rationale (useful to bound API
    cost/latency); remaining rows get the local template instead."""
    flagged_df = flagged_df.copy()
    rationales = []
    for i, (_, row) in enumerate(flagged_df.iterrows()):
        if limit is not None and i >= limit:
            rationales.append(_dry_run_rationale(row))
            continue
        rationale = generate_rationale(row)
        rationales.append(rationale)
    flagged_df["ai_rationale"] = rationales
    return flagged_df


def generate_search_summary(query: str, results: list) -> str:
    """Given a natural-language query and the matching transaction records
    (list of dicts), asks the LLM for a short narrative summary. Falls back
    to a deterministic local summary if no API key is configured."""
    if not results:
        return "No matching transactions to summarize."

    if DRY_RUN:
        tiers = {}
        for r in results:
            t = r.get("risk_tier", "LOW")
            tiers[t] = tiers.get(t, 0) + 1
        tier_str = ", ".join(f"{v} {k}" for k, v in tiers.items())
        return (
            f"[Local summary] {len(results)} transaction(s) matched \"{query}\" — {tier_str}. "
            f"Add a GROQ_API_KEY or ANTHROPIC_API_KEY to `.env` for an AI-generated narrative summary."
        )

    rows_text = "\n".join(
        f"- {r.get('transaction_id')}: ₹{r.get('amount', 0):,.2f} | {r.get('account_id')} | "
        f"{r.get('country')} | {r.get('risk_tier')} | rules: {r.get('triggered_rules', '—')}"
        for r in results[:30]
    )
    prompt = f"""You are a compliance analyst assistant. A colleague searched the transaction
risk database for: "{query}"

Matching records ({len(results)} total, showing up to 30):
{rows_text}

Write a short (2-4 sentence) plain-English summary of what this result set shows,
highlighting the most important risk patterns. Do not invent facts not present above."""
    return _call_ai(prompt)


if __name__ == "__main__":
    import pandas as pd
    from risk_engine import run_risk_engine

    df = pd.read_csv("data/sample_data.csv")
    flagged = run_risk_engine(df)

    print(f"Generating AI rationale for {len(flagged)} flagged transactions "
          f"({'DRY RUN — no API key found' if DRY_RUN else f'LIVE API CALLS via {PROVIDER}/{MODEL}'})...\n")

    result = add_ai_rationale(flagged)

    print("\n--- Sample output ---")
    for _, row in result.head(5).iterrows():
        print(f"\n{row['transaction_id']} | {row['risk_tier']} | Score: {row['risk_score']}")
        print(f"Reasons: {row['flag_reasons']}")
        print(f"AI Rationale: {row['ai_rationale']}")

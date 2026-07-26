# 🛡️ Financial Risk Signal Aggregator

A working prototype that ingests **three independent financial data sources** —
a structured transaction ledger, a structured customer/KYC master, and
**unstructured** free-text analyst notes — correlates them, and produces a
prioritised, AI-synthesised **case** view for a compliance analyst. Amounts in
**₹ (INR)**.

The premise: no single source is decisive on its own. A transaction just under a
reporting threshold is ambiguous; the same transaction from a customer whose
declared income cannot support it, on an account an analyst has already noted as
*"evasive about the purpose, declined to provide invoices"*, is not. Aggregating
those fragmented signals is the whole point.

| Source | Type | Unlocks |
|---|---|---|
| 1. Transaction ledger *(required)* | Structured | 18 behavioural/statistical detectors |
| 2. Customer / KYC master *(optional)* | Structured | `income_inconsistency`, `pep_exposure` |
| 3. Analyst notes / adverse media *(optional)* | **Unstructured** | `adverse_media_hit` (via LLM extraction) |

Sources 2 and 3 are genuinely optional — with only the ledger the app behaves
exactly as before, and the cross-source detectors report nothing rather than
guessing.

## Approach

1. **Ingest** — `data_loader.load_transactions()` accepts a CSV/JSON/TXT
   upload, pasted CSV/JSON text, or a file path, and standardizes column
   names via an alias map (e.g. `txn_id`/`id` → `transaction_id`,
   `timestamp`/`date` → `date_time`). Columns beyond the required schema are
   dropped (e.g. `ip_address`, `status`, internal notes) *except* two
   recognized-but-optional ones — `transaction_type` and `balance` — which
   are kept and used opportunistically if present, never required. The app
   also falls back to the bundled sample dataset (`data/sample_data.csv`,
   300 synthetic Indian retail transactions across 50 accounts) on demand.
2. **Clean** — `risk_engine.clean_data()` standardizes types (parses dates,
   strips currency symbols from amounts and balance, coerces non-numeric
   junk to NaN instead of crashing, drops zero/negative amounts, dedupes)
   and reports what it dropped and why.
3. **Detect** — `risk_engine.analyze_transactions()` runs 21 independent rule
   detectors (velocity, structuring, geography, merchant risk, behavioral
   deviation — see below) over every transaction and combines the results into
   a single weighted **risk_score (0–100)** and **risk_tier**
   (LOW / MEDIUM / HIGH / CRITICAL).
4. **Read the unstructured source** — `ai_reasoning.extract_alert_signals()`
   is where the LLM does work rules cannot. Each free-text note is converted
   into validated JSON `{severity 0-100, indicators[], summary}`, which then
   feeds the same weighted scoring model as every other detector. Output is
   schema-validated and retried on malformed JSON; a keyword heuristic backs it
   up when no API key is set.

   Measured on the bundled notes, it discriminates rather than pattern-matches:
   sanctions/mule/counterfeit notes score 85–90, a note about an unlicensed
   gambling counterparty *with no direct allegation against the customer*
   scores 50, a note whose identity match is explicitly **"NOT confirmed"**
   scores 40 (below the action threshold, despite containing the alarming
   phrase "financial-crime probe"), and all four routine/cleared notes score 0.

5. **Build cases** — `cases.py` rolls transaction findings up to the ACCOUNT
   level. Analysts investigate customers, not rows. Case score sums the weights
   of the **distinct** rules triggered anywhere in the account, so it measures
   breadth of evidence rather than repetition of one signal, and naturally
   rewards corroboration across independent sources.

6. **Synthesise** — two further LLM stages: a per-case narrative reasoning over
   *all* of an account's evidence across *all* sources, and a single
   consolidated executive briefing that groups the batch into themes. Every
   prompt is grounded in supplied evidence with explicit instructions not to
   invent facts. If no API key is configured, deterministic templates are used
   so the app still works end-to-end offline.
7. **Present** — `app.py` (Streamlit) shows a **Consolidated Summary** (the AI
   executive briefing, the ranked case queue, a drill-down case file, and CSV
   export), an Executive Dashboard (KPIs, distributions, rule composition), a
   Risk Alert Center (transaction-level triage), and an AI Search Assistant.

**Session durability.** Streamlit scopes `st.session_state` to a browser
session, so a page refresh would ordinarily discard the analyst's uploaded
dataset. The active dataset is therefore mirrored to `.session_cache/` as
Parquet (dtype-preserving, unlike CSV) and restored on startup — it survives a
refresh *and* a server restart, and is only discarded via the explicit
"Clear loaded data" control. Analysis results are cached separately with
`@st.cache_data`, so a restored dataset re-renders instantly rather than
re-running the detector suite.

## Tools Used

Python · pandas · NumPy · PyArrow · Streamlit · Groq API (Llama 3.3 70B) /
Anthropic API (Claude) / Gemini API (Gemini 2.5 Flash), with automatic
cross-provider failover · python-dotenv

## Risk Detection Logic (`risk_engine.py`)

| Category | Rules |
|---|---|
| Velocity/frequency | card_testing, high_velocity, rapid_drain (balance-aware, see below) |
| Amount/volume | structuring (near ₹10,00,000 CTR threshold), amount_volatility (robust log-scale outlier test), round_number (dataset-relative) |
| Geographic | **high_risk_country** (sanctioned / FATF call-for-action, wt 65), **monitored_jurisdiction** (FATF grey list, wt 35), **offshore_centre** (secrecy jurisdictions, wt 20), foreign_transaction (any non-domestic, wt 12), impossible_travel |
| Merchant/beneficiary | high_risk_merchant, fan_in (mule-account pattern) |
| **Cross-source** | **income_inconsistency** (annualised throughput vs declared income), **pep_exposure** (PEP + cross-border/high-value), **adverse_media_hit** (LLM-scored analyst note) |
| Time/behavioral | odd_hours, dormant_awakening (adaptive gap), new_merchant_large_amount, new_account_large_first_txn, daily_frequency_spike (leave-one-out z-score) |

`rapid_drain` uses a precise **balance-delta** method when the optional
`balance` column is present — a genuine inflow (balance rose) followed by
outflows draining 80%+ of that specific inflow within 30 minutes (balance
fell). Without `balance`, it falls back to guessing direction from
`transaction_type`, restricted to unambiguous rail names only (see
Limitations below).

`fan_in` does **not** use a fixed sender count. A real merchant has thousands
of distinct daily payers, so any absolute threshold flags all ordinary retail
once the dataset grows (measured: 0.8% fire rate at 1k rows → **56% at 16k
rows** before the fix). Instead a window must hold both ≥5 distinct senders
*and* ≥80% of that merchant's **entire** distinct-sender population — the
burst signature of a collection/mule account. An established merchant's senders
are spread across its whole history, so it can never satisfy the second
condition. Being a ratio, this behaves identically on 300 rows and 7M rows.

`amount_volatility` scores a **robust modified z-score on log amounts**, not a
plain z-score on rupees. Transaction amounts are lognormal — measured skew
**12.6 raw vs 0.04 after a log transform** — so a z-score on raw rupees is
applied to the wrong distribution and flags the healthy right tail as
anomalous (4.7% of a *clean* synthetic dataset containing no anomalies at all).
The baseline uses median/MAD rather than mean/σ because mean and σ are
themselves distorted by the outliers being searched for: an account with
several large transactions has them mask one another, which a leave-one-out
mean cannot fix since it only ever removes one. Benchmarked against 300 known
injected spikes, this roughly **halved alert volume (974 → 335) while nearly
doubling precision (30% → 58%)**.

`dormant_awakening` likewise derives its threshold rather than hard-coding it:
`max(span-derived floor, 4 × that account's own median gap)`, with the
multiplier applying only once an account has ≥3 gaps to form a baseline. This
means it transfers between a one-month extract and a multi-year one without
retuning — and correctly stays *silent* on very short files, where no gap can
legitimately qualify as dormancy.

### Scoring & prioritisation

Each rule has a severity weight (e.g. `high_risk_country` = 65,
`structuring` = 70, `odd_hours` = 10). Two scores are produced:

- **`risk_score_raw`** — uncapped sum of triggered weights, used for **ordering**.
- **`risk_score`** — the same value capped to 0–100, used for **display** and tiers.

The split matters: with a capped score alone, every transaction tripping enough
rules lands on exactly 100 and the most severe alerts become unorderable among
themselves. On the bundled test file, 6 rows sit at the cap but span raw scores
of **107–167**, so the queue still ranks them correctly.

Bands: **0–29 LOW · 30–54 MEDIUM · 55–79 HIGH · 80–100 CRITICAL**. Weights are
chosen so a single strong signal (sanctioned jurisdiction, structuring) alone
already reaches HIGH, while weak contextual signals stay LOW unless
corroborated by something else.

Because several detectors are percentile-based, the *number* of matches grows
with file size even though the *rate* stays stable. The Alert Center therefore
caps the review queue (default top 200 by score) and states plainly how many
matched versus how many are shown, rather than rendering an unreviewable table.

## Data Assumptions & Limitations

- Expected columns: `transaction_id, account_id, date, amount, merchant,
  country` (required). `transaction_type` and `balance` are optional —
  recognized and used opportunistically when present, never required. Any
  other extra columns in an upload (`ip_address`, `status`, internal notes,
  etc.) are silently dropped by `data_loader.py`; nothing crashes, that
  information just isn't part of the analysis.
- We deliberately tested what happens with columns beyond the required
  schema and found one worth treating specially: `balance`. Without it,
  `rapid_drain` can only guess a transaction's direction from
  `transaction_type` — unreliable for ambiguous Indian rails like UPI/NEFT
  (see below). With a `balance` column, direction becomes an observed fact
  (balance rose = inflow, balance fell = outflow) rather than a guess, so
  the detector activates automatically and precisely, no configuration
  needed. Its one inherent limitation: an account's very first observed
  transaction has no prior balance to diff against, so direction is
  genuinely unknowable for that one row from balance data alone — this
  mostly overlaps with `new_account_large_first_txn` already covering that
  case separately.
- No IP/device columns exist in a standard transaction export, so
  device/session-based fraud patterns aren't modelled.
- No beneficiary-account column exists, so fan-in/mule detection is
  approximated using `merchant` as a stand-in for the receiving party.
- No geolocation (lat/long), so "impossible travel" is approximated from
  country changes between consecutive transactions within a short window,
  not true distance/time physics.
- `dormant_awakening`'s gap threshold is set to 14 days (vs. a typical 90+
  days in production) because the bundled sample dataset only spans ~1 month.
- Standard transaction schemas often don't include an explicit credit/debit
  direction field — payment-rail names like NEFT/UPI/RTGS don't indicate
  direction on their own. We identified this during testing and excluded
  ambiguous-direction rails from direction-dependent detectors (`rapid_drain`'s
  fallback path) rather than guessing, to avoid silently misclassifying
  transactions. As a result, the fallback path will legitimately return
  few/no flags on datasets that don't carry unambiguous inflow/outflow
  labels and also have no `balance` column — that's the honest outcome when
  the data doesn't support the detection, not a bug.
- `round_number` is calibrated relative to each dataset's own median
  transaction size (not a fixed rupee amount), so it adapts to whatever
  scale the uploaded data actually uses instead of guessing a number that
  could be too high or too low depending on the dataset.
- The bundled sample data is synthetic/randomly generated (not hand-crafted
  fraud scenarios) — it's meant to exercise the pipeline, not "prove" any one
  detector.

## Setup & Installation

```bash
cd risk-aggregator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Configure `.env` (all optional — the app runs fully offline without any key,
using local fallback rationale):
```env
AI_PROVIDER=groq                # primary: "groq" (default), "anthropic", or "gemini"
GROQ_API_KEY=                   # https://console.groq.com/keys
ANTHROPIC_API_KEY=
GEMINI_API_KEY=                 # https://aistudio.google.com/apikey
```

**Automatic failover.** If more than one key is configured, a rate limit or
outage on the primary provider is not visible to the user — the app tries
the next configured provider before giving up, and only shows an error once
every configured provider has failed. This is what stops a single provider's
free-tier daily cap from ever showing up on screen mid-session.

Run it:
```bash
streamlit run app.py
```

### Deploying

`.env` is gitignored — secrets must never be committed — so a deployed app has
no `.env` to read and would otherwise fall back to the offline templates.
Credentials are therefore resolved from **the environment first, then
`st.secrets`**, which covers both cases without a code change:

| Where | How to supply the key |
|---|---|
| Local | `risk-aggregator/.env` (as above) |
| Streamlit Community Cloud | App → **Settings → Secrets**, then reboot |
| Any container/PaaS (Render, Fly, HF Spaces…) | Standard environment variables |

On Streamlit Cloud the Secrets panel takes TOML:
```toml
AI_PROVIDER    = "groq"
GROQ_API_KEY   = "your_key_here"
GEMINI_API_KEY = "your_key_here"   # optional — enables failover if Groq is rate-limited
```

Keys are also passed explicitly into the Groq/Anthropic/Gemini clients rather
than relying on their implicit environment lookup — a key arriving via
`st.secrets` never reaches the environment, so the implicit path would
silently fail in exactly the deployed case it is needed for.

The sidebar always states which mode is active, so a missing key is visible
rather than silent.

## Example Input → Output

**Input row** (`data/sample_data.csv`):
```csv
transaction_id,account_id,date,amount,merchant,country,transaction_type
TXN000285,ACC100031,2026-06-07 15:35:00,190916.08,Apollo Pharmacy,Russia,UPI
```

**Output** (Risk Alert Center):
- **Risk Tier:** CRITICAL (score 87/100)
- **Triggered Rules:** `high_risk_country`, `new_merchant_large_amount`
- **AI Rationale:** *"This ₹1,90,916.08 transaction on account ACC100031 was
  flagged because it involves Russia, a jurisdiction on the high-risk/watchlist
  list, and was paid to Apollo Pharmacy — a merchant this account has never
  paid before — for an unusually large amount. Recommend compliance review of
  account ACC100031 before further transactions are authorized."*
  (local-fallback template shown when no API key is set; live LLM calls
  produce an equivalent narrative grounded in the same two signals.)

## Validation

The engine was tested against a purpose-built dataset (`test_dataset.csv`, 222
rows) containing **15 deliberately planted fraud scenarios** — one per detector
— plus a block of data-quality edge cases (missing amounts, `"₹12,500.00"`
currency strings, `N/A`, negative and zero amounts, blank country, duplicate
`transaction_id`, an unparseable date, and whitespace padding).

Results: **16/16 planted scenarios detected**, 7 malformed rows correctly
quarantined with a per-reason breakdown, and **0 false positives** on the
edge-case rows. Scores were verified by hand against the weight table (e.g.
`rapid_drain` 60 + `round_number` 15 + `new_merchant_large_amount` 22 +
`new_account_large_first_txn` 32 = 129 → capped 100 ✓).

One genuine defect was found this way and fixed: four unrelated edge-case rows
sharing a placeholder merchant name at an identical timestamp were being flagged
as `fan_in`, which led to the concentration-based rewrite described above.

**Multi-source correlation is verified both ways.** Running the same ledger with
and without the optional sources confirms that (a) with the ledger alone all 16
planted scenarios still pass and the three cross-source detectors report exactly
**zero** — they stay silent rather than guessing — and (b) with all three sources
they fire on precisely the accounts the fixtures plant them on:

| Detector | Accounts flagged | Planted on |
|---|---|---|
| `income_inconsistency` | ACC2004, ACC2014 | ACC2004, ACC2014 ✓ |
| `pep_exposure` | ACC2007, ACC2009 | ACC2007, ACC2009 ✓ |
| `adverse_media_hit` | ACC2004, ACC2007, ACC2009, ACC2012, ACC300 | the 5 substantiated notes ✓ |

The corroboration effect is visible in the ranking: ACC2009 scores 272 raw
(ledger `impossible_travel` + KYC `pep_exposure` + an analyst note describing a
cloned card), ahead of accounts with an equally severe single-source signal.

A second defect was caught here too. Given only a raw flagged total and an income
figure, the case narrative was comparing those two numbers directly and drawing a
false conclusion (`₹60.7L` "inconsistent with" `₹65.7L`). The detector actually
compares *annualised throughput* to income, so the computed ratio is now passed
into the prompt explicitly rather than left for the model to infer — the same
case now correctly reads "12.5x the declared annual income". Grounding a model
means supplying the derived figure, not just its inputs.

## Scalability — measured, and deliberately scoped

The prototype targets interactive analyst use, not batch ETL. Rather than
assume it scales, the engine was benchmarked at increasing volumes; the
findings drove the fixes above.

**What the benchmarks found (and what changed):**

| Issue | Before | After |
|---|---|---|
| `fan_in` false-positive rate at 16k rows | 56.4% | **0.0%** |
| `fan_in` runtime at 16k rows | 3.21 s | **0.042 s** (76×) |
| `high_velocity` runtime at 16k rows | 4.18 s | **0.12 s** (34×, output byte-identical) |
| `amount_volatility` false positives on clean data | 4.74% | **0.36%** |
| `amount_volatility` precision vs known spikes | 30% | **58%** |
| Row-wise scoring at 200k rows | 2.37 s | **0.008 s** (284×, output byte-identical) |
| End-to-end `analyze_transactions` at 16k rows | 10.88 s | **3.02 s** (3.6×) |

Sliding-window detectors were rewritten from per-row rescans into single
`searchsorted` / two-pointer passes, and scoring from a row-wise `apply` into a
matrix multiply. Streamlit results are now cached, so the engine runs once per
dataset instead of re-running on every filter click.

**Known ceiling (documented, not engineered around):** at ~69 bytes/row, a
100 MB file is ~1.5M rows and a 500 MB file ~7.6M rows. Extrapolating the
measured rate, those land at roughly 8 and 40 minutes respectively, with peak
memory several times file size (whole-file load, plus a `flag_reasons` Python
list per row). This is beyond what an in-memory pandas prototype should be
asked to do.

**Migration path if this became a production service:**
1. Swap the pandas backend for **DuckDB or Polars** — out-of-core execution and
   native SQL window functions replace the remaining per-group Python loops.
2. Two-pass chunked processing: pass 1 computes global statistics (percentiles,
   per-account aggregates), pass 2 applies detectors per chunk.
3. Replace the `flag_reasons` list column with the packed boolean matrix,
   materialising rule names only for rows actually displayed.
4. Segment percentile baselines by customer type — a single global percentile is
   meaningless once retail (₹500) and corporate (₹50L) populations are mixed.
5. Persist alerts to a store with pagination, instead of rendering DataFrames.

## Known Gaps / Next Steps

- Optional enhancements implemented: natural-language query interface, visual
  dashboard.
- Not yet implemented: multi-source correlation across *separate* datasets
  (e.g. joining transaction data with a KYC/watchlist feed) — currently all
  signals come from the single transaction file.
- Detector thresholds are tuned against synthetic and semi-synthetic data. On
  real production traffic they would need recalibration against labelled
  outcomes (confirmed fraud / cleared alerts) rather than injected anomalies.

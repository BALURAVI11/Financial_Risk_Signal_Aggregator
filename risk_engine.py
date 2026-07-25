"""
Risk Detection Engine
----------------------
Detects risk signals/anomalies in a financial transactions dataset (INR).

Expected columns (standard schema, produced by data_loader.load_transactions):
    transaction_id, account_id, date_time, amount, country, merchant
    Optional: transaction_type, balance — used opportunistically when present,
    never required. Any other columns in the source file are dropped by the
    loader (e.g. ip_address, status, internal notes) since nothing here uses
    them; nothing crashes, that information is just not part of the analysis.

Each detector function below returns a boolean Series (True = flagged) aligned
to the input DataFrame's index. All detectors are combined into a single
weighted risk_score (0-100) + risk_tier + flag_reasons per transaction.

NOTE ON LIMITATIONS (documented in the README too):
- No IP/device columns exist in a standard transaction CSV, so
  "IP/device mismatch" fraud patterns cannot be detected here.
- No beneficiary_account_id column exists, so fan-in/fan-out detection is
  approximated using `merchant` as a stand-in for the receiving party.
- No geolocation (lat/long) data exists, so "impossible travel" is
  approximated using country changes between consecutive transactions
  within a short time window, not true distance/time physics.
- Standard transaction schemas often don't include an explicit credit/debit
  direction field — payment-rail names like NEFT/UPI/RTGS don't indicate
  direction on their own. Rather than guess, ambiguous-direction rails are
  excluded from direction-dependent detectors (see AMBIGUOUS_DIRECTION_TYPES
  below) instead of being forced into "outflow", which would silently
  misclassify transactions.
- `rapid_drain` uses a `balance` column when present (precise: balance rose
  = inflow, balance fell = outflow) instead of guessing direction from
  transaction_type. Its one inherent limitation: an account's very first
  observed transaction has no prior balance to diff against, so direction is
  genuinely unknowable from balance data alone for that one row — this
  mostly overlaps with `new_account_large_first_txn` already covering that
  case separately. Without a `balance` column, it falls back to the
  type-based approximation above (and inherits the same direction caveat).
- Amount thresholds below are calibrated for INR (₹); some (e.g. round-number,
  new-merchant, first-transaction detection) scale to the dataset's own
  amount distribution rather than a fixed rupee figure, so they adapt across
  datasets of different size.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG — tune these thresholds for your dataset (all amounts in INR)
# ---------------------------------------------------------------------------
# Watchlist / sanctioned jurisdictions — matched against upper-cased country
# names or ISO-3166 alpha-2 codes, whichever the dataset provides.
HIGH_RISK_COUNTRIES = {
    "IRAN", "IR", "NORTH KOREA", "KP", "SYRIA", "SY", "RUSSIA", "RU",
    "AFGHANISTAN", "AF", "MYANMAR", "MM",
}
# Used by the lighter-weight foreign_transaction signal below — any country
# other than this is flagged as "foreign", regardless of watchlist status.
DOMESTIC_COUNTRY = "INDIA"
HIGH_RISK_MERCHANT_KEYWORDS = ["casino", "gambling", "crypto exchange",
                                "forex", "offshore", "adult"]

# Indian payment rails classified by direction of funds. Matched as exact
# lower-cased strings (not substring) so "credit card" is never mistaken
# for an inflow just because it contains the word "credit".
#
# Payment-rail names like NEFT/UPI/RTGS don't indicate direction on their
# own — a UPI or NEFT transaction is just as likely to be money coming in
# as going out, and a standard transaction export rarely carries an
# explicit credit/debit flag. Rather than guess (and silently misclassify
# a chunk of them), these ambiguous-direction rails are excluded from both
# INFLOW_TYPES and OUTFLOW_TYPES entirely, so direction-dependent detectors
# (rapid_drain) only ever act on types that are genuinely unambiguous. If
# the real dataset barely uses the unambiguous words below, rapid_drain
# will correctly return few/no flags rather than confidently return wrong
# ones — that's the honest outcome when the data doesn't support direction
# detection.
INFLOW_TYPES = {"deposit", "credit", "incoming", "salary credit", "refund", "cashback"}
OUTFLOW_TYPES = {"withdrawal", "debit", "outgoing", "purchase", "atm withdrawal", "debit card", "pos"}
AMBIGUOUS_DIRECTION_TYPES = {"neft", "rtgs", "imps", "upi", "net banking", "credit card", "transfer", "wire"}

# Used by fan_in / circular_transfer, which only need "this is a transfer
# rail" and don't depend on direction, so the ambiguity above doesn't apply.
TRANSFER_LIKE_TYPES = {"neft", "rtgs", "imps", "upi", "transfer", "wire"}

CARD_TESTING_AMOUNT_MAX = 100.0     # tiny probe amounts (₹)
CARD_TESTING_WINDOW_MIN = 10
CARD_TESTING_MIN_COUNT = 3

HIGH_VELOCITY_WINDOW_MIN = 10
HIGH_VELOCITY_MIN_COUNT = 5

# Just under India's ₹10,00,000 cash-transaction reporting threshold (CTR/FIU-IND)
STRUCTURING_LOW = 900000
STRUCTURING_HIGH = 999999
STRUCTURING_WINDOW_DAYS = 5
STRUCTURING_MIN_COUNT = 3

# amount_volatility works on LOG amounts using a median/MAD "modified z-score",
# not a plain z-score on rupees. Two reasons, both measured on this data:
#   1. Transaction amounts are lognormal — raw skew ~12.6 vs ~0.04 after a log
#      transform. A z-score assumes normality, so on raw rupees it is measuring
#      the wrong distribution and flags the healthy right tail as anomalous
#      (4.7% of a clean synthetic dataset containing no anomalies at all).
#   2. Mean/standard-deviation baselines are themselves distorted by the very
#      outliers being looked for. Median and MAD are resistant to up to ~50%
#      contamination, so an account with several large transactions can still
#      be measured — where a leave-one-out mean only ever removes one of them
#      and lets the rest mask each other.
# Threshold picked by measurement, not convention. Against a synthetic set with
# 300 known injected spikes, sweeping the cut-off gave:
#     old raw z-score (z>=3.0) : recall 96%, precision 30%, F1 45   (974 alerts)
#     new log+MAD  threshold 2.5: recall 77%, precision 42%, F1 55
#     new log+MAD  threshold 3.0: recall 65%, precision 58%, F1 61  <- best
#     new log+MAD  threshold 3.5: recall 51%, precision 70%, F1 59
# 3.0 roughly halves the alert volume (974 -> 335 for 300 real anomalies) while
# nearly doubling precision. Recall is deliberately traded away: this is one
# weighted signal among sixteen, so a miss can still be caught by a
# corroborating rule, whereas a false positive inflates the score of an
# otherwise ordinary transaction and pollutes the ranking for every analyst.
VOLATILITY_MODIFIED_Z_THRESHOLD = 3.0
MIN_HISTORY_FOR_VOLATILITY = 8   # below ~8 samples a per-account baseline is meaningless;
                                   # ordinary variation would look like an outlier
_MAD_TO_SIGMA = 0.6745           # MAD x this^-1 estimates sigma for normal data
_MEANAD_TO_SIGMA = 0.7979        # same for mean absolute deviation (MAD==0 fallback)

# Precise balance-based rapid-drain: flag if 80%+ of a specific inflow is
# withdrawn again within the window. Measured against the inflow itself, not
# total account balance — a percentage-of-total measure gets diluted by any
# pre-existing balance and can miss a real drain on an already-active account.
DRAIN_FRACTION_THRESHOLD = 0.8
DRAIN_WINDOW_MIN = 30

IMPOSSIBLE_TRAVEL_WINDOW_MIN = 60   # country change faster than this = suspicious

ODD_HOUR_START = 23                 # 11pm
ODD_HOUR_END = 5                    # 5am

# dormant_awakening: a FIXED gap threshold cannot work across datasets, because
# "dormant" is relative to how long the data actually observes an account. A
# 14-day rule (tuned to a 1-month sample) fires constantly on multi-year data;
# a 180-day rule can never fire on a 1-month sample. So the threshold is derived
# per-dataset and per-account instead:
#   threshold = max(span-derived floor, MULTIPLIER x that account's median gap)
# The multiplier only applies once an account has enough gaps to form a real
# baseline; accounts with 1-2 transactions fall back to the floor alone.
DORMANT_FLOOR_FRACTION_OF_SPAN = 0.10   # 10% of the dataset's observed time span...
DORMANT_FLOOR_MIN_DAYS = 30             # ...clamped to a sane AML range. Below ~30 days
DORMANT_FLOOR_MAX_DAYS = 90             #    "dormancy" isn't meaningful; 90+ is industry norm.
DORMANT_MIN_GAPS_FOR_BASELINE = 3       # need >=3 gaps before an account's median means anything
DORMANT_GAP_MULTIPLIER = 4.0            # gap must be 4x the account's own normal rhythm
DORMANT_REAWAKEN_PERCENTILE = 0.75      # ...and the reawakening amount must be large FOR THIS dataset

# fan_in: an absolute sender count alone does NOT work — a legitimate merchant
# (Amazon, Swiggy) has thousands of distinct payers every day, so any fixed
# threshold flags all ordinary retail once the dataset is more than a few
# thousand rows (measured: 0.8% fire rate at 1k rows -> 56% at 16k rows).
# The discriminator is CONCENTRATION: a real merchant's senders are spread
# across its whole history, whereas a mule/collection account receives nearly
# all of its distinct senders inside one short burst. Concentration is a ratio,
# so it is scale-free — it behaves identically on 300 rows and 7M rows.
FAN_IN_MIN_SENDERS = 5              # absolute floor: below this it's coincidence, not a pattern
FAN_IN_WINDOW_HOURS = 24
FAN_IN_CONCENTRATION = 0.80         # >=80% of the merchant's lifetime senders inside one window

# Percentile-based (not a fixed rupee figure) so these adapt to whatever
# scale the uploaded dataset actually uses, same reasoning as round_number.
NEW_MERCHANT_PERCENTILE = 0.90         # flag "new merchant" only if amount is in the top 10% for this dataset
FIRST_TXN_PERCENTILE = 0.95            # flag "large first txn" only if amount is in the top 5% for this dataset
FREQ_SPIKE_ZSCORE_THRESHOLD = 2.5      # how far above an account's typical daily count counts as a spike

# --- cross-source thresholds (require the customer / alert sources) ----------
# Moving >3x your declared annual income through an account is the standard
# trigger for a source-of-funds enquiry. Measured on the bundled data this
# separates cleanly: ordinary accounts top out around 1.1x, planted
# inconsistencies land at 3.6-12.6x.
INCOME_THROUGHPUT_RATIO = 3.0
PEP_LARGE_AMOUNT_PERCENTILE = 0.90     # "high value" for a PEP, relative to this dataset
ADVERSE_MEDIA_SEVERITY_THRESHOLD = 60  # 0-100 severity assigned by the LLM extraction step

# ---------------------------------------------------------------------------
# Per-rule severity weights (0-100 scale). Chosen so that a single strong
# signal (e.g. a sanctioned-jurisdiction transaction) already lands in the
# HIGH tier on its own, while weak/contextual signals (odd hour, round
# number) stay LOW unless corroborated by something else.
# ---------------------------------------------------------------------------
RULE_WEIGHTS = {
    "high_risk_country": 65,
    "impossible_travel": 65,
    "structuring": 70,
    "rapid_drain": 60,
    "fan_in": 60,
    "circular_transfer": 55,
    # Cross-source signals. Weighted high because corroboration across
    # independent sources is materially stronger evidence than any single
    # in-ledger pattern — that is the whole premise of aggregating signals.
    "adverse_media_hit": 55,
    "pep_exposure": 50,
    "income_inconsistency": 45,
    "card_testing": 50,
    "high_risk_merchant": 40,
    "high_velocity": 35,
    "dormant_awakening": 35,
    "new_account_large_first_txn": 32,
    "amount_volatility": 30,
    "daily_frequency_spike": 28,
    "new_merchant_large_amount": 22,
    "round_number": 15,
    "foreign_transaction": 12,
    "odd_hours": 10,
}

TIER_BANDS = [
    (80, "CRITICAL"),
    (55, "HIGH"),
    (30, "MEDIUM"),
]


def score_to_tier(score: float) -> str:
    for threshold, tier in TIER_BANDS:
        if score >= threshold:
            return tier
    return "LOW"


# ---------------------------------------------------------------------------
# CLEANING
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    df = df.copy()
    original_count = len(df)
    if "date" in df.columns and "date_time" not in df.columns:
        df = df.rename(columns={"date": "date_time"})

    # Strip currency symbols (₹, $) and thousands separators, THEN coerce to
    # numeric. Using errors="coerce" (not astype(float)) means non-numeric
    # junk like "N/A" or "" becomes NaN instead of crashing the whole pipeline.
    df["amount"] = df["amount"].astype(str).str.replace(r"[₹$,]", "", regex=True).str.strip()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["country"] = df["country"].astype(str).str.strip().str.upper()
    df["merchant"] = df["merchant"].astype(str).str.strip()
    df["account_id"] = df["account_id"].astype(str).str.strip()
    if "transaction_type" in df.columns:
        df["transaction_type"] = df["transaction_type"].astype(str).str.strip()
    if "balance" in df.columns:
        df["balance"] = df["balance"].astype(str).str.replace(r"[₹$,]", "", regex=True).str.strip()
        df["balance"] = pd.to_numeric(df["balance"], errors="coerce")

    # Diagnostics before dropping, so bad rows aren't silently lost
    missing_amount = df["amount"].isna().sum()
    missing_date = df["date_time"].isna().sum()
    zero_or_negative = (df["amount"] <= 0).sum()

    df = df.dropna(subset=["date_time", "amount"])
    df = df[df["amount"] > 0]   # zero/negative amounts are not valid transactions for this engine

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["transaction_id"])
    duplicates_removed = before_dedup - len(df)

    df = df.sort_values("date_time").reset_index(drop=True)

    if verbose:
        dropped = original_count - len(df)
        print(f"[clean_data] {original_count} rows in -> {len(df)} rows out "
              f"({dropped} dropped: {missing_amount} bad/missing amount, "
              f"{missing_date} unparseable date, {zero_or_negative} zero/negative amount, "
              f"{duplicates_removed} duplicate transaction_id)")

    return df


# ---------------------------------------------------------------------------
# 1. VELOCITY / FREQUENCY
# ---------------------------------------------------------------------------
def detect_card_testing(df: pd.DataFrame) -> pd.Series:
    """Multiple tiny transactions across different merchants, same account, short window."""
    flags = pd.Series(False, index=df.index)
    for acct, group in df.groupby("account_id"):
        small = group[group["amount"] <= CARD_TESTING_AMOUNT_MAX]
        if len(small) < CARD_TESTING_MIN_COUNT:
            continue
        small = small.sort_values("date_time")
        times = small["date_time"].values
        for i in range(len(times)):
            window_end = times[i] + np.timedelta64(CARD_TESTING_WINDOW_MIN, "m")
            in_window = small[(small["date_time"] >= times[i]) & (small["date_time"] <= window_end)]
            if len(in_window) >= CARD_TESTING_MIN_COUNT and in_window["merchant"].nunique() >= CARD_TESTING_MIN_COUNT:
                flags.loc[in_window.index] = True
    return flags


def detect_high_velocity(df: pd.DataFrame) -> pd.Series:
    """Sudden burst of N+ transactions within a short window on one account.

    Because each account's timestamps are sorted, the end of every window can
    be located for all rows at once with a binary search (`searchsorted`)
    instead of re-filtering the group per row. The windows that qualify are
    then painted onto the flag array with a difference-array + cumsum, so the
    whole detector is vectorised. Semantics are unchanged from the original
    per-row rescan."""
    flags = pd.Series(False, index=df.index)
    window = np.timedelta64(HIGH_VELOCITY_WINDOW_MIN, "m")

    for acct, group in df.groupby("account_id", sort=False):
        if len(group) < HIGH_VELOCITY_MIN_COUNT:
            continue  # cannot possibly reach the count
        group = group.sort_values("date_time")
        times = group["date_time"].to_numpy()

        # ends[i] = one past the last row still inside [times[i], times[i]+window]
        ends = np.searchsorted(times, times + window, side="right")
        starts = np.arange(len(times))
        qualifying = np.flatnonzero((ends - starts) >= HIGH_VELOCITY_MIN_COUNT)
        if qualifying.size == 0:
            continue

        # Paint every qualifying [start, end) interval in one pass.
        delta = np.zeros(len(times) + 1, dtype=np.int32)
        np.add.at(delta, qualifying, 1)
        np.add.at(delta, ends[qualifying], -1)
        covered = np.cumsum(delta[:-1]) > 0
        flags.loc[group.index.to_numpy()[covered]] = True
    return flags


def _detect_rapid_drain_by_balance(df: pd.DataFrame) -> pd.Series:
    """Precise version — used automatically when a `balance` column is present.
    Uses balance deltas to determine TRUE direction (balance rose = inflow,
    balance fell = outflow) instead of guessing from transaction_type, which
    sidesteps the ambiguous-rail problem entirely (NEFT/UPI/etc. don't
    indicate direction on their own). Still measures drain relative to the
    SPECIFIC inflow amount, not total account balance — a percentage-of-total
    measure gets diluted by any pre-existing balance and can miss a real
    drain on an already-active account."""
    flags = pd.Series(False, index=df.index)
    for acct, group in df.groupby("account_id"):
        group = group.sort_values("date_time").copy()
        # First row per account has no prior balance to diff against -> NaN,
        # which correctly excludes it from both inflows and outflows below
        # (direction of an account's very first transaction is genuinely
        # unknowable from balance data alone).
        group["_delta"] = group["balance"].diff()
        inflows = group[group["_delta"] > 0]
        outflows = group[group["_delta"] < 0].copy()
        outflows["_outflow_amt"] = -outflows["_delta"]
        for idx, inflow_row in inflows.iterrows():
            inflow_amt = inflow_row["_delta"]
            window_end = inflow_row["date_time"] + pd.Timedelta(minutes=DRAIN_WINDOW_MIN)
            following_outflows = outflows[(outflows["date_time"] >= inflow_row["date_time"]) &
                                           (outflows["date_time"] <= window_end)]
            if len(following_outflows) >= 2 and \
               following_outflows["_outflow_amt"].sum() >= DRAIN_FRACTION_THRESHOLD * inflow_amt:
                flags.loc[idx] = True
                flags.loc[following_outflows.index] = True
    return flags


def detect_rapid_drain(df: pd.DataFrame) -> pd.Series:
    """Large inflow immediately followed by multiple outflows on the same account.

    Uses the precise balance-based method (see _detect_rapid_drain_by_balance)
    when a `balance` column is present — this directly observes the effect on
    the account rather than inferring direction from transaction_type. Falls
    back to the type-based approximation (INFLOW_TYPES/OUTFLOW_TYPES) when no
    balance data is available."""
    if "balance" in df.columns and df["balance"].notna().any():
        return _detect_rapid_drain_by_balance(df)

    flags = pd.Series(False, index=df.index)
    if "transaction_type" not in df.columns:
        return flags
    type_lower = df["transaction_type"].str.lower()
    for acct, group in df.groupby("account_id"):
        group = group.sort_values("date_time")
        credits = group[type_lower.loc[group.index].isin(INFLOW_TYPES)]
        debits = group[type_lower.loc[group.index].isin(OUTFLOW_TYPES)]
        for _, credit_row in credits.iterrows():
            window_end = credit_row["date_time"] + pd.Timedelta(minutes=DRAIN_WINDOW_MIN)
            following_debits = debits[(debits["date_time"] >= credit_row["date_time"]) &
                                       (debits["date_time"] <= window_end)]
            if len(following_debits) >= 2 and \
               following_debits["amount"].sum() >= DRAIN_FRACTION_THRESHOLD * credit_row["amount"]:
                flags.loc[credit_row.name] = True
                flags.loc[following_debits.index] = True
    return flags


# ---------------------------------------------------------------------------
# 2. AMOUNT / VOLUME
# ---------------------------------------------------------------------------
def detect_structuring(df: pd.DataFrame) -> pd.Series:
    """Repeated transactions just under the ₹10L reporting threshold within a short period."""
    flags = pd.Series(False, index=df.index)
    near_threshold = df[(df["amount"] >= STRUCTURING_LOW) & (df["amount"] <= STRUCTURING_HIGH)]
    for acct, group in near_threshold.groupby("account_id"):
        group = group.sort_values("date_time")
        times = group["date_time"].values
        for i in range(len(times)):
            window_end = times[i] + np.timedelta64(STRUCTURING_WINDOW_DAYS, "D")
            in_window = group[(group["date_time"] >= times[i]) & (group["date_time"] <= window_end)]
            if len(in_window) >= STRUCTURING_MIN_COUNT:
                flags.loc[in_window.index] = True
    return flags


def detect_amount_volatility(df: pd.DataFrame) -> pd.Series:
    """Transaction far above what this account normally moves.

    Scored as a robust modified z-score on LOG amounts (see the config note
    above for why both the log and the robust statistics are necessary):

        z = (log(amount) - median_account) / (MAD_account / 0.6745)

    One-sided by design — only unusually LARGE amounts are risk signals here.
    Unusually small ones are the province of `card_testing`, which identifies
    them far more precisely (repeated probes across several merchants).
    """
    flags = pd.Series(False, index=df.index)
    if df.empty:
        return flags

    acct = df["account_id"]
    # clean_data guarantees amount > 0, so the log is always defined.
    log_amt = np.log(df["amount"])
    by_acct = log_amt.groupby(acct)

    n = by_acct.transform("size")
    median = by_acct.transform("median")
    abs_dev = (log_amt - median).abs()
    mad = abs_dev.groupby(acct).transform("median")

    scale = mad / _MAD_TO_SIGMA
    # MAD collapses to zero when more than half of an account's transactions
    # share one amount. Mean absolute deviation still has signal there, so a
    # genuine outlier isn't lost just because the account is repetitive.
    mean_ad = abs_dev.groupby(acct).transform("mean")
    scale = scale.where(scale > 0, mean_ad / _MEANAD_TO_SIGMA)

    modified_z = (log_amt - median) / scale.replace(0, np.nan)

    return (
        (n >= MIN_HISTORY_FOR_VOLATILITY)
        & (modified_z >= VOLATILITY_MODIFIED_Z_THRESHOLD)
    ).fillna(False)


def detect_round_number(df: pd.DataFrame) -> pd.Series:
    """Large, suspiciously round wire amounts. "Large" is defined relative to
    this dataset's own median transaction (not a fixed rupee amount), so the
    rule adapts to whatever scale the uploaded data actually uses instead of
    guessing a number that could be too high (misses real anomalies on a
    low-value dataset) or too low (floods high-value datasets with noise)."""
    is_round = (df["amount"] % 1000 == 0) | (df["amount"] % 500 == 0)
    round_number_min = df["amount"].quantile(0.50)
    is_large = df["amount"] >= round_number_min
    return is_round & is_large


# ---------------------------------------------------------------------------
# 3. GEOGRAPHIC (approximated — see limitations note at top)
# ---------------------------------------------------------------------------
def detect_high_risk_country(df: pd.DataFrame) -> pd.Series:
    return df["country"].isin(HIGH_RISK_COUNTRIES)


def detect_foreign_transaction(df: pd.DataFrame) -> pd.Series:
    """Any transaction outside the domestic country — a lighter signal than
    the sanctioned-country watchlist, useful when most activity is expected
    to be domestic. Weak on its own (foreign transactions are common and
    often legitimate) but meaningfully compounds with other signals — e.g.
    a foreign transaction at 3am is more suspicious than either fact alone."""
    return df["country"] != DOMESTIC_COUNTRY


def detect_impossible_travel(df: pd.DataFrame) -> pd.Series:
    """Country changes between consecutive transactions faster than plausible travel."""
    flags = pd.Series(False, index=df.index)
    for acct, group in df.groupby("account_id"):
        group = group.sort_values("date_time")
        prev_country = None
        prev_time = None
        prev_idx = None
        for idx, row in group.iterrows():
            if prev_country is not None and row["country"] != prev_country:
                gap_min = (row["date_time"] - prev_time).total_seconds() / 60
                if gap_min <= IMPOSSIBLE_TRAVEL_WINDOW_MIN:
                    flags.loc[idx] = True
                    flags.loc[prev_idx] = True
            prev_country, prev_time, prev_idx = row["country"], row["date_time"], idx
    return flags


# ---------------------------------------------------------------------------
# 4. MERCHANT / BENEFICIARY (fan-in approximated via merchant, see note at top)
# ---------------------------------------------------------------------------
def detect_high_risk_merchant(df: pd.DataFrame) -> pd.Series:
    pattern = "|".join(HIGH_RISK_MERCHANT_KEYWORDS)
    return df["merchant"].str.lower().str.contains(pattern, na=False)


def detect_fan_in(df: pd.DataFrame) -> pd.Series:
    """Many distinct accounts funnelling funds into the same beneficiary in a short
    burst — the classic mule/collection-account pattern.

    Restricted to transfer-like rails (NEFT/RTGS/IMPS/UPI). UPI is deliberately
    KEPT despite being India's busiest retail rail, because UPI is also the
    dominant P2P mule rail — excluding it would blind this detector to the
    majority of real fan-in fraud.

    Instead of a fixed sender count (which flags all ordinary retail once a
    merchant has many daily payers), a window is only flagged when it holds
    BOTH:
      1. at least FAN_IN_MIN_SENDERS distinct senders (coincidence floor), and
      2. at least FAN_IN_CONCENTRATION of that merchant's ENTIRE distinct-sender
         population (burst test).
    An established merchant's senders are spread across its whole history, so
    condition 2 fails for it no matter how busy a single day is. A collection
    account receives nearly all of its senders at once, so it passes. Being a
    ratio, this behaves identically on a 300-row file and a 7M-row file.

    Implemented as a single O(k) two-pointer pass per merchant (each row enters
    and leaves the trailing window exactly once) rather than an O(k^2) rescan.
    """
    flags = pd.Series(False, index=df.index)
    if "transaction_type" not in df.columns:
        return flags
    transfers = df[df["transaction_type"].str.lower().isin(TRANSFER_LIKE_TYPES)]
    if transfers.empty:
        return flags

    window = np.timedelta64(FAN_IN_WINDOW_HOURS, "h")
    for merchant, group in transfers.groupby("merchant", sort=False):
        total_senders = group["account_id"].nunique()
        if total_senders < FAN_IN_MIN_SENDERS:
            continue  # can never clear the floor — skip without scanning

        group = group.sort_values("date_time")
        idx = group.index.to_numpy()
        times = group["date_time"].to_numpy()
        accounts = group["account_id"].to_numpy()

        live = {}            # account_id -> count inside the current window
        flagged_pos = np.zeros(len(times), dtype=bool)
        left = 0
        for right in range(len(times)):
            a = accounts[right]
            live[a] = live.get(a, 0) + 1
            while times[right] - times[left] > window:
                la = accounts[left]
                live[la] -= 1
                if live[la] == 0:
                    del live[la]
                left += 1
            n_distinct = len(live)
            if (n_distinct >= FAN_IN_MIN_SENDERS
                    and n_distinct / total_senders >= FAN_IN_CONCENTRATION):
                flagged_pos[left:right + 1] = True

        if flagged_pos.any():
            flags.loc[idx[flagged_pos]] = True
    return flags


# ---------------------------------------------------------------------------
# 5. TIME-OF-DAY / BEHAVIORAL
# ---------------------------------------------------------------------------
def detect_odd_hours(df: pd.DataFrame) -> pd.Series:
    hour = df["date_time"].dt.hour
    return (hour >= ODD_HOUR_START) | (hour <= ODD_HOUR_END)


def detect_dormant_awakening(df: pd.DataFrame) -> pd.Series:
    """Account that went quiet for an unusually long stretch, then moved a large
    amount. Both halves of "unusually long" and "large" are derived from the data
    rather than hard-coded (see the config block above), so the rule transfers
    between a one-month extract and a multi-year one without retuning.

    Note this means the detector will correctly stay silent on very short
    datasets: if the file only spans a few weeks, no gap inside it can qualify
    as dormancy, and reporting one anyway would be a confident false positive."""
    flags = pd.Series(False, index=df.index)
    if df.empty:
        return flags

    span_days = (df["date_time"].max() - df["date_time"].min()).total_seconds() / 86400
    floor_days = min(max(span_days * DORMANT_FLOOR_FRACTION_OF_SPAN,
                         DORMANT_FLOOR_MIN_DAYS), DORMANT_FLOOR_MAX_DAYS)
    amount_threshold = df["amount"].quantile(DORMANT_REAWAKEN_PERCENTILE)

    ordered = df.sort_values(["account_id", "date_time"])
    gaps = ordered.groupby("account_id")["date_time"].diff().dt.total_seconds() / 86400

    # Each account's own rhythm: median gap, and how many gaps we actually saw.
    gap_stats = gaps.groupby(ordered["account_id"]).agg(["median", "count"])
    med_gap = ordered["account_id"].map(gap_stats["median"]).fillna(0.0)
    n_gaps = ordered["account_id"].map(gap_stats["count"]).fillna(0)

    threshold = np.where(
        n_gaps >= DORMANT_MIN_GAPS_FOR_BASELINE,
        np.maximum(floor_days, DORMANT_GAP_MULTIPLIER * med_gap),
        floor_days,
    )

    reawaken = (gaps >= threshold) & (ordered["amount"] >= amount_threshold)
    flags.loc[ordered.index[reawaken.fillna(False)]] = True
    return flags


# ---------------------------------------------------------------------------
# 6. ACCOUNT BEHAVIORAL DEVIATION (relative to the account's own history)
# ---------------------------------------------------------------------------
def detect_new_merchant_large_amount(df: pd.DataFrame) -> pd.Series:
    """Account pays a merchant it has never paid before, for an amount that's
    unusually large *for this dataset* (top 10% percentile, not a fixed rupee
    figure — this adapts whether the data is retail-scale or wire-transfer-scale)."""
    flags = pd.Series(False, index=df.index)
    threshold = df["amount"].quantile(NEW_MERCHANT_PERCENTILE)
    for acct, group in df.groupby("account_id"):
        group = group.sort_values("date_time")
        seen_merchants = set()
        for idx, row in group.iterrows():
            if row["merchant"] not in seen_merchants and row["amount"] >= threshold:
                flags.loc[idx] = True
            seen_merchants.add(row["merchant"])
    return flags


def detect_new_account_large_first_txn(df: pd.DataFrame) -> pd.Series:
    """An account's very first transaction in the dataset is unusually large
    relative to this dataset's overall distribution (top 5% percentile) —
    can indicate a mule/synthetic account being set up and used immediately."""
    flags = pd.Series(False, index=df.index)
    threshold = df["amount"].quantile(FIRST_TXN_PERCENTILE)
    first_txns = df.sort_values("date_time").groupby("account_id").head(1)
    large_first = first_txns[first_txns["amount"] >= threshold]
    flags.loc[large_first.index] = True
    return flags


def detect_daily_frequency_spike(df: pd.DataFrame) -> pd.Series:
    """An account has a day with far more transactions than its typical daily count.
    Broader net than high_velocity — catches bursts spread across hours, not just minutes.
    Uses LEAVE-ONE-OUT z-score for the same reason as amount_volatility: including
    the spike day in its own baseline lets it mask itself by inflating the mean/std."""
    flags = pd.Series(False, index=df.index)
    df = df.copy()
    df["_date"] = df["date_time"].dt.date
    daily_counts = df.groupby(["account_id", "_date"]).size().reset_index(name="count")

    for acct, group in daily_counts.groupby("account_id"):
        n = len(group)
        if n < 3:
            continue  # not enough distinct days for a meaningful baseline
        total = group["count"].sum()
        total_sq = (group["count"] ** 2).sum()
        for _, day_row in group.iterrows():
            others_n = n - 1
            others_mean = (total - day_row["count"]) / others_n
            others_sq = total_sq - day_row["count"] ** 2
            others_var = (others_sq / others_n) - (others_mean ** 2)
            others_std = others_var ** 0.5 if others_var > 0 else 0
            if others_std == 0:
                continue
            zscore = (day_row["count"] - others_mean) / others_std
            if zscore >= FREQ_SPIKE_ZSCORE_THRESHOLD:
                mask = (df["account_id"] == acct) & (df["_date"] == day_row["_date"])
                flags.loc[df[mask].index] = True
    return flags


# ---------------------------------------------------------------------------
# 7. CROSS-SOURCE — signals that CANNOT be derived from the transaction ledger
#    alone. Each requires a second source to have been joined on by
#    data_loader.enrich_transactions(); each degrades to a silent no-op when
#    that source is absent, exactly like the optional `balance` column.
# ---------------------------------------------------------------------------
def detect_income_inconsistency(df: pd.DataFrame) -> pd.Series:
    """Money moved through the account is wildly out of proportion to the income
    declared at onboarding — the classic source-of-funds question.

    Throughput is annualised from the dataset's own observed span so the
    comparison is like-for-like whether the extract covers one month or two
    years. This is an ACCOUNT-level property, so every transaction belonging to
    the account carries the flag: they are all part of the same pattern."""
    flags = pd.Series(False, index=df.index)
    if "declared_income_annual" not in df.columns or df.empty:
        return flags

    span_days = max((df["date_time"].max() - df["date_time"].min()).days, 1)
    annualised = df.groupby("account_id")["amount"].transform("sum") * 365.0 / span_days
    income = pd.to_numeric(df["declared_income_annual"], errors="coerce")
    ratio = annualised / income.where(income > 0)
    return (ratio >= INCOME_THROUGHPUT_RATIO).fillna(False)


def detect_pep_exposure(df: pd.DataFrame) -> pd.Series:
    """A politically exposed person transacting cross-border or at high value.

    PEP status on its own is a standing customer attribute, not an alert —
    flagging every PEP transaction would just re-report the KYC file. It
    becomes a risk signal when combined with the kind of activity that PEP
    monitoring exists to catch."""
    flags = pd.Series(False, index=df.index)
    if "pep_flag" not in df.columns or df.empty:
        return flags

    is_pep = df["pep_flag"].fillna(False).astype(bool)
    if not is_pep.any():
        return flags
    foreign = df["country"] != DOMESTIC_COUNTRY
    large = df["amount"] >= df["amount"].quantile(PEP_LARGE_AMOUNT_PERCENTILE)
    return is_pep & (foreign | large)


def detect_adverse_media_hit(df: pd.DataFrame) -> pd.Series:
    """An analyst note or adverse-media snippet about this account was assessed
    as materially serious. The severity score is produced by the LLM extraction
    step in ai_reasoning.extract_alert_signals() — this detector only applies
    the threshold, it does not read the free text itself."""
    flags = pd.Series(False, index=df.index)
    if "adverse_media_severity" not in df.columns or df.empty:
        return flags
    severity = pd.to_numeric(df["adverse_media_severity"], errors="coerce")
    return (severity >= ADVERSE_MEDIA_SEVERITY_THRESHOLD).fillna(False)


def detect_circular_transfer(df: pd.DataFrame) -> pd.Series:
    """OPTIONAL — only useful if `merchant` sometimes contains another account_id
    (i.e. your data represents P2P transfers, not just retail merchants).
    Flags account pairs that send money back and forth to each other."""
    flags = pd.Series(False, index=df.index)
    if "transaction_type" not in df.columns:
        return flags
    transfers = df[df["transaction_type"].str.lower().isin(TRANSFER_LIKE_TYPES)]
    known_accounts = set(df["account_id"])
    for idx, row in transfers.iterrows():
        if row["merchant"] in known_accounts:
            reverse = transfers[
                (transfers["account_id"] == row["merchant"]) & (transfers["merchant"] == row["account_id"])
            ]
            if not reverse.empty:
                flags.loc[idx] = True
                flags.loc[reverse.index] = True
    return flags


# ---------------------------------------------------------------------------
# COMBINE ALL DETECTORS
# ---------------------------------------------------------------------------
DETECTORS = {
    "card_testing": detect_card_testing,
    "high_velocity": detect_high_velocity,
    "rapid_drain": detect_rapid_drain,
    "structuring": detect_structuring,
    "amount_volatility": detect_amount_volatility,
    "round_number": detect_round_number,
    "high_risk_country": detect_high_risk_country,
    "foreign_transaction": detect_foreign_transaction,
    "impossible_travel": detect_impossible_travel,
    "high_risk_merchant": detect_high_risk_merchant,
    "fan_in": detect_fan_in,
    "odd_hours": detect_odd_hours,
    "dormant_awakening": detect_dormant_awakening,
    "new_merchant_large_amount": detect_new_merchant_large_amount,
    "new_account_large_first_txn": detect_new_account_large_first_txn,
    "daily_frequency_spike": detect_daily_frequency_spike,
    # Cross-source — no-op unless the customer / alert sources were joined on.
    "income_inconsistency": detect_income_inconsistency,
    "pep_exposure": detect_pep_exposure,
    "adverse_media_hit": detect_adverse_media_hit,
    # "circular_transfer": detect_circular_transfer,  # enable only if your merchant field can hold account_ids
}

# Which detectors need a source beyond the transaction ledger. Used by the UI to
# explain *why* a detector reported nothing, rather than leaving the analyst to
# guess whether it found nothing or never ran.
CROSS_SOURCE_DETECTORS = {
    "income_inconsistency": "customer/KYC file (declared_income_annual)",
    "pep_exposure": "customer/KYC file (pep_flag)",
    "adverse_media_hit": "analyst notes / adverse media",
}


def analyze_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Runs every detector over the full dataset and returns ALL transactions
    (not just flagged ones) annotated with risk scores, tier and flag_reasons.

    Produces TWO scores:
      - `risk_score_raw` — the uncapped sum of triggered rule weights. Used for
        ORDERING. Without it, every transaction that trips enough rules lands on
        exactly 100 and the "worst" alerts become unorderable among themselves,
        which defeats the point of a prioritised queue.
      - `risk_score`     — the same value capped to 0-100. Used for DISPLAY and
        tier banding, so the scale stays interpretable to an analyst.

    Scoring is a single matrix multiply rather than a row-wise apply (measured
    ~284x faster on 200k rows, byte-identical output)."""
    df = clean_data(df)
    flag_matrix = pd.DataFrame(index=df.index)
    for name, fn in DETECTORS.items():
        try:
            flag_matrix[name] = fn(df)
        except Exception as e:
            print(f"Detector '{name}' failed: {e}")
            flag_matrix[name] = False

    rule_names = np.array(flag_matrix.columns)
    matrix = flag_matrix.to_numpy(dtype=bool)
    weights = np.array([RULE_WEIGHTS[c] for c in flag_matrix.columns], dtype=float)

    raw = matrix @ weights
    capped = np.minimum(100.0, raw)
    df["risk_score_raw"] = raw
    df["risk_score"] = capped

    # TIER_BANDS is ordered strongest-first, so np.select takes the first match.
    df["risk_tier"] = np.select(
        [capped >= threshold for threshold, _ in TIER_BANDS],
        [tier for _, tier in TIER_BANDS],
        default="LOW",
    )

    df["flag_reasons"] = [list(rule_names[row]) for row in matrix]
    df["triggered_rules"] = [", ".join(r) if len(r) else "—" for r in df["flag_reasons"]]

    return df.sort_values("risk_score_raw", ascending=False).reset_index(drop=True)


def run_risk_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: same as analyze_transactions but returns only the
    flagged subset (risk_score > 0). Used by the CLI entry points below."""
    analyzed = analyze_transactions(df)
    return analyzed[analyzed["risk_score"] > 0]


if __name__ == "__main__":
    from data_loader import load_transactions
    df = load_transactions("data/sample_data.csv")
    result = run_risk_engine(df)
    print(f"Total transactions: {len(df)}")
    print(f"Flagged transactions: {len(result)}")
    print(result[["transaction_id", "account_id", "amount", "country",
                   "risk_score", "risk_tier", "flag_reasons"]].head(15))

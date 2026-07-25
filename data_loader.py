"""
Data Loader
------------
Accepts transaction data as CSV, JSON, or plain delimited text and returns
a standardized pandas DataFrame with consistent column names, regardless
of which format or column-name variant the input uses.

Usage:
    from data_loader import load_transactions
    df = load_transactions("data/sample_data.csv")
    df = load_transactions(raw_text)                # pasted CSV or JSON text
    df = load_transactions(uploaded_file_object)     # e.g. Streamlit's file_uploader
"""

import io
import json
import pandas as pd

# Maps common column-name variants to the standard name the risk engine expects.
COLUMN_ALIASES = {
    "date": "date_time",
    "datetime": "date_time",
    "date_time": "date_time",
    "timestamp": "date_time",
    "transaction_date": "date_time",
    "amount": "amount",
    "txn_amount": "amount",
    "value": "amount",
    "account_id": "account_id",
    "account": "account_id",
    "acct_id": "account_id",
    "transaction_id": "transaction_id",
    "txn_id": "transaction_id",
    "id": "transaction_id",
    "merchant": "merchant",
    "beneficiary": "merchant",
    "payee": "merchant",
    "country": "country",
    "region": "country",
    "location": "country",
    "transaction_type": "transaction_type",
    "type": "transaction_type",
    "txn_type": "transaction_type",
    "mode": "transaction_type",
    "balance": "balance",
    "account_balance": "balance",
    "running_balance": "balance",
    "closing_balance": "balance",
}

REQUIRED_COLUMNS = ["transaction_id", "account_id", "date_time", "amount", "merchant", "country"]
# Used opportunistically by the risk engine if present, but never required —
# a dataset without them still gets a full analysis, just with a couple of
# detectors (rapid_drain's precise mode, fan_in, etc.) falling back or no-op'ing.
OPTIONAL_KNOWN_COLUMNS = ["transaction_type", "balance"]


def _standardize_columns(df: pd.DataFrame,
                         aliases: dict = None,
                         required: list = None,
                         optional: list = None) -> pd.DataFrame:
    """Map incoming column-name variants onto the canonical schema for one
    source type, then verify the mandatory columns survived. Defaults to the
    transaction schema; the customer and alert loaders pass their own."""
    aliases = COLUMN_ALIASES if aliases is None else aliases
    required = REQUIRED_COLUMNS if required is None else required
    optional = OPTIONAL_KNOWN_COLUMNS if optional is None else optional

    rename_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in aliases:
            rename_map[col] = aliases[key]
    df = df.rename(columns=rename_map)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input data is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Keep required columns + any recognized optional ones if present. Truly
    # unrecognized extra columns (ip_address, status, internal notes, etc.) are
    # dropped — nothing downstream uses them, and carrying unused columns risks
    # them being silently misinterpreted later.
    keep_cols = required + [c for c in optional if c in df.columns]
    return df[keep_cols]


def _parse_text(raw_text: str) -> pd.DataFrame:
    """Best-effort parser for plain delimited text (comma or tab separated)."""
    delimiter = "\t" if "\t" in raw_text.splitlines()[0] else ","
    return pd.read_csv(io.StringIO(raw_text), sep=delimiter)


def _read_any(source, json_root: str = "transactions") -> pd.DataFrame:
    """Read a file path, raw string, or file-like object into a raw DataFrame.
    Format is auto-detected from the extension, or from the content when a bare
    string is pasted in."""
    # Case 1: file path with a recognised extension
    if isinstance(source, str) and source.endswith((".csv", ".json", ".txt")):
        if source.endswith(".json"):
            with open(source, "r") as f:
                data = json.load(f)
            return pd.DataFrame(data if isinstance(data, list) else data.get(json_root, data))
        if source.endswith(".csv"):
            return pd.read_csv(source)
        with open(source, "r") as f:            # .txt
            return _parse_text(f.read())

    # Case 2: raw string content (pasted text, or a JSON document)
    if isinstance(source, str):
        stripped = source.strip()
        if not stripped:
            raise ValueError("No data provided.")
        if stripped.startswith(("[", "{")):
            data = json.loads(stripped)
            return pd.DataFrame(data if isinstance(data, list) else data.get(json_root, data))
        return _parse_text(stripped)

    # Case 3: file-like object (e.g. Streamlit's uploaded_file)
    if hasattr(source, "read"):
        name = getattr(source, "name", "").lower()
        content = source.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        if name.endswith(".json"):
            data = json.loads(content)
            return pd.DataFrame(data if isinstance(data, list) else data.get(json_root, data))
        if name.endswith(".csv"):
            return pd.read_csv(io.StringIO(content))
        return _parse_text(content)

    raise ValueError(f"Unsupported input type: {type(source).__name__}")


def load_transactions(source) -> pd.DataFrame:
    """Load the primary transaction ledger (CSV, JSON, or delimited text)."""
    return _standardize_columns(_read_any(source, json_root="transactions"))


# ---------------------------------------------------------------------------
# SOURCE 2 — customer / KYC master (optional)
# ---------------------------------------------------------------------------
CUSTOMER_ALIASES = {
    "account_id": "account_id", "account": "account_id", "acct_id": "account_id",
    "customer_name": "customer_name", "name": "customer_name", "customer": "customer_name",
    "declared_income_annual": "declared_income_annual", "declared_income": "declared_income_annual",
    "annual_income": "declared_income_annual", "income": "declared_income_annual",
    "occupation": "occupation", "profession": "occupation",
    "pep_flag": "pep_flag", "pep": "pep_flag", "is_pep": "pep_flag",
    "kyc_risk_rating": "kyc_risk_rating", "kyc_rating": "kyc_risk_rating",
    "risk_rating": "kyc_risk_rating",
    "onboarding_date": "onboarding_date", "account_open_date": "onboarding_date",
}
CUSTOMER_REQUIRED = ["account_id"]
CUSTOMER_OPTIONAL = ["customer_name", "declared_income_annual", "occupation",
                     "pep_flag", "kyc_risk_rating", "onboarding_date"]


def load_customers(source) -> pd.DataFrame:
    """Load the customer/KYC master. Only `account_id` is mandatory — every
    other attribute enriches the analysis if present and is simply skipped if
    not, so a partial KYC extract is still useful."""
    df = _read_any(source, json_root="customers")
    df = _standardize_columns(df, CUSTOMER_ALIASES, CUSTOMER_REQUIRED, CUSTOMER_OPTIONAL)
    df["account_id"] = df["account_id"].astype(str).str.strip()

    if "declared_income_annual" in df.columns:
        df["declared_income_annual"] = pd.to_numeric(
            df["declared_income_annual"].astype(str).str.replace(r"[₹$,]", "", regex=True).str.strip(),
            errors="coerce")
    if "pep_flag" in df.columns:
        # Accept Y/N, true/false, 1/0 — all collapse to a real boolean.
        df["pep_flag"] = (df["pep_flag"].astype(str).str.strip().str.lower()
                          .isin(["y", "yes", "true", "1", "t"]))
    if "kyc_risk_rating" in df.columns:
        df["kyc_risk_rating"] = df["kyc_risk_rating"].astype(str).str.strip().str.upper()

    return df.drop_duplicates(subset=["account_id"])


# ---------------------------------------------------------------------------
# SOURCE 3 — unstructured analyst notes / adverse media (optional)
# ---------------------------------------------------------------------------
ALERT_ALIASES = {
    "account_id": "account_id", "account": "account_id", "acct_id": "account_id",
    "note": "note", "text": "note", "narrative": "note", "body": "note",
    "comment": "note", "description": "note", "alert_text": "note",
    "alert_date": "alert_date", "date": "alert_date", "created_at": "alert_date",
    "source": "source", "origin": "source", "system": "source",
}
ALERT_REQUIRED = ["account_id", "note"]
ALERT_OPTIONAL = ["alert_date", "source"]


def load_alerts(source) -> pd.DataFrame:
    """Load free-text analyst notes / adverse-media snippets keyed to an account.
    This is the unstructured leg of the pipeline — the text is not parsed here,
    it is handed to the LLM extraction step in `ai_reasoning.py`."""
    df = _read_any(source, json_root="alerts")
    df = _standardize_columns(df, ALERT_ALIASES, ALERT_REQUIRED, ALERT_OPTIONAL)
    df["account_id"] = df["account_id"].astype(str).str.strip()
    df["note"] = df["note"].astype(str).str.strip()
    return df[df["note"].str.len() > 0]


# ---------------------------------------------------------------------------
# CORRELATION — fold the optional sources onto the transaction ledger
# ---------------------------------------------------------------------------
def enrich_transactions(txn_df: pd.DataFrame,
                        customers_df: pd.DataFrame = None,
                        alert_signals_df: pd.DataFrame = None) -> pd.DataFrame:
    """Left-join customer attributes and extracted alert signals onto every
    transaction row, keyed on account_id.

    A left join keeps the transaction ledger authoritative: accounts with no
    KYC record or no analyst note simply carry NaN in those columns. Detectors
    then see them as ordinary optional columns — exactly how `balance` is
    already handled — so nothing needs a new signature and nothing breaks when
    a source is missing entirely.
    """
    enriched = txn_df.copy()

    if customers_df is not None and not customers_df.empty:
        cols = [c for c in customers_df.columns if c != "account_id"]
        enriched = enriched.merge(
            customers_df[["account_id"] + cols], on="account_id", how="left")

    if alert_signals_df is not None and not alert_signals_df.empty:
        cols = [c for c in alert_signals_df.columns if c != "account_id"]
        enriched = enriched.merge(
            alert_signals_df[["account_id"] + cols], on="account_id", how="left")

    return enriched


if __name__ == "__main__":
    df = load_transactions("data/sample_data.csv")
    print(f"transactions : {len(df)} rows, columns {list(df.columns)}")
    cust = load_customers("data/sample_customers.csv")
    print(f"customers    : {len(cust)} rows, columns {list(cust.columns)}")
    alerts = load_alerts("data/sample_alerts.csv")
    print(f"alerts       : {len(alerts)} notes, columns {list(alerts.columns)}")
    merged = enrich_transactions(df, cust)
    print(f"enriched     : {len(merged)} rows, columns {list(merged.columns)}")

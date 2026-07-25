from data_loader import load_transactions
from risk_engine import clean_data, DETECTORS

df = load_transactions("data/sample_data.csv")
df = clean_data(df, verbose=True)
print(f"\nTotal transactions: {len(df)}")
print(f"'balance' column present: {'balance' in df.columns}\n")
for name, fn in DETECTORS.items():
    flags = fn(df)
    print(f"{name}: {flags.sum()} flagged ({flags.sum()/len(df)*100:.1f}%)")
 
print("\n--- Amount stats ---")
print(df['amount'].describe())
 
print("\n--- Transactions per account ---")
print(df.groupby('account_id').size().describe())

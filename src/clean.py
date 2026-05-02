import pandas as pd
from pathlib import Path

RAW_DATA_PATH = 'data/raw/telco_customer_churn.csv'
PROCESSED_DATA_PATH = Path('data/processed/telco_customer_churn_clean.csv')

def clean_telco_data(path):
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Convert totalcharges from object/string to numeric
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")

    # Remove rows where totalcharges was blank or invalid
    df = df.dropna(subset=["totalcharges"])

    # Convert churn target from Yes/No to 1/0
    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

    # Save cleaned data
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Cleaning complete.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Saved to: {PROCESSED_DATA_PATH}")

def main():
    clean_telco_data(RAW_DATA_PATH)

if __name__ == "__main__":
    main()
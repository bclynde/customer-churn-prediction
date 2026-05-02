import pandas as pd

RAW_PATH = 'data/raw/telco_customer_churn.csv'

def acquire_telco_data(path):
    df = pd.read_csv(path)
    return df

def summarize_telco_data(df):
    print(df.info())
    print(df.describe())
    print(df.head())

def main():
    df = acquire_telco_data(RAW_PATH)
    summarize_telco_data(df)

if __name__ == "__main__":
    main()
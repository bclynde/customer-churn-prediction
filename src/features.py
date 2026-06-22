"""
Feature engineering for Telco Customer Churn project.

Creates business-motivated features based on EDA findings.
Input:  data/processed/telco_customer_churn_clean.csv
Output: data/processed/telco_customer_churn_features.csv
"""

from pathlib import Path
import pandas as pd


CLEAN_DATA_PATH = Path("data/processed/telco_customer_churn_clean.csv")
FEATURED_DATA_PATH = Path("data/processed/telco_customer_churn_features.csv")


def load_clean_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded clean data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Month-to-month contract flag
    # EDA showed month-to-month customers churn much more often.
    df["is_month_to_month"] = (df["contract"] == "Month-to-month").astype(int)

    # 2. Fiber optic customer flag
    # Fiber optic customers had much higher churn than DSL/no internet.
    df["is_fiber_customer"] = (df["internetservice"] == "Fiber optic").astype(int)

    # 3. Electronic check payment flag
    # Electronic check customers had the highest churn rate by payment method.
    df["uses_electronic_check"] = (
        df["paymentmethod"] == "Electronic check"
    ).astype(int)

    # 4. New customer flag
    # Lower-tenure customers are usually more likely to churn.
    df["is_new_customer"] = (df["tenure"] < 12).astype(int)

    # 5. High monthly charge flag
    # Uses the median so the threshold is data-driven.
    monthly_charge_median = df["monthlycharges"].median()
    df["high_monthly_charge"] = (
        df["monthlycharges"] > monthly_charge_median
    ).astype(int)

    # 6. Service count
    # Counts how many optional services a customer has.
    service_cols = [
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies",
    ]

    df["service_count"] = df[service_cols].apply(
        lambda row: sum(value == "Yes" for value in row),
        axis=1,
    )

    # 7. Has support/security services
    # These services appeared negatively associated with churn.
    df["has_security_or_support"] = (
        (df["onlinesecurity"] == "Yes") | (df["techsupport"] == "Yes")
    ).astype(int)

    # 8. Estimated average charge per tenure month
    # Avoid division by zero, although cleaned Telco data should have tenure > 0.
    df["avg_charge_per_tenure_month"] = df["totalcharges"] / df["tenure"]

    return df


def save_featured_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Featured data saved to: {path}")
    print(f"New shape: {df.shape[0]} rows, {df.shape[1]} columns")


def main():
    df = load_clean_data(CLEAN_DATA_PATH)
    df_features = create_features(df)
    save_featured_data(df_features, FEATURED_DATA_PATH)


if __name__ == "__main__":
    main()
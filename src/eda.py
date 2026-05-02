from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_DATA_PATH = Path("data/processed/telco_customer_churn_clean.csv")
FIGURES_DIR = Path("outputs/figures")
HISTOGRAM_DIR = FIGURES_DIR / "histograms"
BAR_DIR = FIGURES_DIR / "bar"


def header():
    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)


def ensure_output_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    HISTOGRAM_DIR.mkdir(parents=True, exist_ok=True)
    BAR_DIR.mkdir(parents=True, exist_ok=True)


def data_overview(df):
    # Dataset Overview
    print("\n--- DATASET OVERVIEW ---")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

def column_overview(df):
    # Column Information
    print("\n--- COLUMN INFORMATION ---")
    df.info()


def statistical_summary(df):
    # Statistical Summary
    print("\n--- STATISTICAL SUMMARY ---")
    print(df.describe())

def churn_analysis(df):
    # Churn analysis
    print("\n--- CHURN ANALYSIS ---")
    churn_counts = df["churn"].value_counts().sort_index()
    churn_percentage = df["churn"].value_counts(normalize=True).sort_index() * 100

    print("\nChurn Distribution:")
    print(churn_counts)
    print("\nChurn Percentage:")
    print(churn_percentage.round(2))


def data_quality(df):
    # Data Quality
    print("\n--- DATA QUALITY ---")
    print(f"Missing Values:\n{df.isnull().sum()}")

def categorical_overview(df):
    # Categorical Columns
    print("\n--- CATEGORICAL COLUMNS ---")
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Values: {df[col].unique()}")


def plot_numeric_distributions(df):
    numeric_cols = [col for col in df.select_dtypes(include="number").columns if col != "churn"]
    if not numeric_cols:
        return

    for col in numeric_cols:
        values = df[col].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].hist(values, bins=30, color="#2a6f97", edgecolor="white")
        axes[0].set_title(f"{col} Distribution")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Count")

        axes[1].boxplot(values, vert=False)
        axes[1].set_title(f"{col} Box Plot")
        axes[1].set_xlabel(col)

        fig.tight_layout()
        fig.savefig(HISTOGRAM_DIR / f"{col}_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include="number")
    correlation_matrix = numeric_df.corr()

    if correlation_matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(correlation_matrix.columns)
    ax.set_title("Numeric Feature Correlations")

    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            ax.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black")

    fig.colorbar(heatmap, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_categorical_churn_relationships(df):
    categorical_cols = [col for col in df.select_dtypes(include="object").columns if col != "customerid"]

    for col in categorical_cols:
        churn_rate = df.groupby(col)["churn"].mean().sort_values(ascending=False) * 100

        fig, ax = plt.subplots(figsize=(10, 5))
        churn_rate.plot(kind="bar", ax=ax, color="#f28e2b")
        ax.set_title(f"Churn Rate by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Churn Rate (%)")
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        safe_name = col.lower().replace(" ", "_")
        fig.savefig(BAR_DIR / f"{safe_name}_churn_rate.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    header()
    ensure_output_dir()
    df = pd.read_csv(PROCESSED_DATA_PATH)

    data_overview(df)
    column_overview(df)
    statistical_summary(df)
    churn_analysis(df)
    data_quality(df)
    categorical_overview(df)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(df)
    plot_categorical_churn_relationships(df)

    print(f"\nEDA figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

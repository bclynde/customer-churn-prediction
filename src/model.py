"""
Baseline model comparison for Telco Customer Churn prediction.

Models:
1. Logistic Regression
2. Random Forest

No feature engineering yet — this establishes baseline model performance.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/processed/telco_customer_churn_clean.csv")

METRICS_DIR = Path("outputs/metrics")
MODELS_DIR = Path("outputs/models")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def prepare_features(df: pd.DataFrame):
    df = df.drop(columns=["customerid"])

    X = df.drop(columns=["churn"])
    y = df["churn"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def build_model_pipeline(model, numeric_cols: list, categorical_cols: list) -> Pipeline:
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def evaluate_model(y_test, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
    }


def print_metrics(model_name: str, metrics: dict) -> None:
    print("\n" + "=" * 50)
    print(f"{model_name} Results")
    print("=" * 50)

    for name, value in metrics.items():
        print(f"{name:<12}: {value:.4f}")

    print("=" * 50)


def save_metrics(model_name: str, metrics: dict, report: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(f"{model_name} Classification Metrics\n")
        f.write("=" * 50 + "\n")

        for name, value in metrics.items():
            f.write(f"{name:<12}: {value:.4f}\n")

        f.write("\nClassification Report\n")
        f.write("=" * 50 + "\n")
        f.write(report)

    print(f"Metrics saved to {path}")


def plot_confusion_matrix(model_name: str, y_test, y_pred, path: Path) -> None:
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed (0)", "Churned (1)"],
        yticklabels=["Stayed (0)", "Churned (1)"],
        ax=ax,
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Confusion matrix saved to {path}")


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def train_and_evaluate_model(
    model_name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    numeric_cols: list,
    categorical_cols: list,
) -> dict:
    print(f"\nTraining {model_name}...")

    pipeline = build_model_pipeline(model, numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print_metrics(model_name, metrics)
    print("\nClassification Report:")
    print(report)

    safe_name = model_name.lower().replace(" ", "_")

    plot_confusion_matrix(
        model_name,
        y_test,
        y_pred,
        METRICS_DIR / f"{safe_name}_confusion_matrix.png",
    )

    save_metrics(
        model_name,
        metrics,
        report,
        METRICS_DIR / f"{safe_name}_metrics.txt",
    )

    save_model(
        pipeline,
        MODELS_DIR / f"{safe_name}.joblib",
    )

    return metrics


def main():
    df = load_data(DATA_PATH)
    X, y, numeric_cols, categorical_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Churn rate — train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

    logistic_metrics = train_and_evaluate_model(
        "Logistic Regression",
        LogisticRegression(max_iter=1000),
        X_train,
        X_test,
        y_train,
        y_test,
        numeric_cols,
        categorical_cols,
    )

    random_forest_metrics = train_and_evaluate_model(
        "Random Forest",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42        
            ),
        X_train,
        X_test,
        y_train,
        y_test,
        numeric_cols,
        categorical_cols,
    )

    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    print(f"{'Metric':<12} {'Logistic':<12} {'Random Forest':<15}")
    print("-" * 50)

    for metric in logistic_metrics:
        print(
            f"{metric:<12} "
            f"{logistic_metrics[metric]:<12.4f} "
            f"{random_forest_metrics[metric]:<15.4f}"
        )


if __name__ == "__main__":
    main()
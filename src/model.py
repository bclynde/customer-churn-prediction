"""
Model training and evaluation for Telco Customer Churn.

Models:
1. Logistic Regression baseline
2. Random Forest baseline
3. Tuned Random Forest with GridSearchCV

Metrics:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
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
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
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

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def build_pipeline(model, numeric_cols, categorical_cols):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
            ("classifier", model),
        ]
    )


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    report = classification_report(y_test, y_pred)

    return metrics, report, y_pred


def print_metrics(model_name, metrics):
    print("\n" + "=" * 55)
    print(f"{model_name} Results")
    print("=" * 55)

    for metric, value in metrics.items():
        print(f"{metric:<12}: {value:.4f}")

    print("=" * 55)


def save_metrics(model_name, metrics, report):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_")
    path = METRICS_DIR / f"{safe_name}_metrics.txt"

    with open(path, "w") as f:
        f.write(f"{model_name} Metrics\n")
        f.write("=" * 50 + "\n")

        for metric, value in metrics.items():
            f.write(f"{metric:<12}: {value:.4f}\n")

        f.write("\nClassification Report\n")
        f.write("=" * 50 + "\n")
        f.write(report)

    print(f"Metrics saved to {path}")


def plot_confusion_matrix(model_name, y_test, y_pred):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_")
    path = METRICS_DIR / f"{safe_name}_confusion_matrix.png"

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

    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Confusion matrix saved to {path}")


def save_model(model_name, model):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_")
    path = MODELS_DIR / f"{safe_name}.joblib"

    joblib.dump(model, path)
    print(f"Model saved to {path}")


def train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test, numeric_cols, categorical_cols):
    print(f"\nTraining {model_name}...")

    pipeline = build_pipeline(model, numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    metrics, report, y_pred = evaluate_model(pipeline, X_test, y_test)

    print_metrics(model_name, metrics)
    print("\nClassification Report:")
    print(report)

    save_metrics(model_name, metrics, report)
    plot_confusion_matrix(model_name, y_test, y_pred)
    save_model(model_name, pipeline)

    return metrics


def train_tuned_random_forest(X_train, X_test, y_train, y_test, numeric_cols, categorical_cols):
    print("\nTraining Tuned Random Forest with GridSearchCV...")

    rf_pipeline = build_pipeline(
        RandomForestClassifier(random_state=42),
        numeric_cols,
        categorical_cols,
    )

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_leaf": [1, 3, 5],
    }

    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Random Forest Parameters:")
    print(grid_search.best_params_)
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    metrics, report, y_pred = evaluate_model(best_model, X_test, y_test)

    model_name = "Tuned Random Forest"

    print_metrics(model_name, metrics)
    print("\nClassification Report:")
    print(report)

    save_metrics(model_name, metrics, report)
    plot_confusion_matrix(model_name, y_test, y_pred)
    save_model(model_name, best_model)

    return metrics


def print_model_comparison(results):
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)

    metrics = list(next(iter(results.values())).keys())

    header = f"{'Metric':<12}" + "".join(f"{model:<22}" for model in results)
    print(header)
    print("-" * 70)

    for metric in metrics:
        row = f"{metric:<12}"
        for model_name in results:
            row += f"{results[model_name][metric]:<22.4f}"
        print(row)


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

    results = {}

    results["Logistic Regression"] = train_and_evaluate(
        "Logistic Regression",
        LogisticRegression(max_iter=1000),
        X_train,
        X_test,
        y_train,
        y_test,
        numeric_cols,
        categorical_cols,
    )

    results["Random Forest"] = train_and_evaluate(
        "Random Forest",
        RandomForestClassifier(random_state=42),
        X_train,
        X_test,
        y_train,
        y_test,
        numeric_cols,
        categorical_cols,
    )

    results["Tuned Random Forest"] = train_tuned_random_forest(
        X_train,
        X_test,
        y_train,
        y_test,
        numeric_cols,
        categorical_cols,
    )

    print_model_comparison(results)


if __name__ == "__main__":
    main()
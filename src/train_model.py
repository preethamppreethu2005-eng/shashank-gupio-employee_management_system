"""Train and compare classification models for term-deposit subscription."""
from __future__ import annotations

from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_preprocessing import build_preprocessor


def train_and_select(
    X: pd.DataFrame,
    y: pd.Series,
    model_dir: str | Path,
    output_dir: str | Path,
    random_state: int = 42,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Fit two baselines, save the ROC-AUC winner, and return test split data."""
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    rows: list[dict[str, float | str]] = []
    fitted: dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("classifier", estimator)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
        )
        fitted[name] = pipeline

    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["model"])
    best_model = fitted[best_name]
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "target": y.name or "y",
        "feature_columns": list(X.columns),
        "positive_class": 1,
    }
    with (model_dir / "final_model.pkl").open("wb") as file:
        pickle.dump(bundle, file)
    return best_model, comparison, X_test, y_test

#!/usr/bin/env python3
"""AI Analyst: Train a machine-learning model on tabular CSV data and print insights."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TaskType = Literal["classification", "regression"]


@dataclass
class AnalysisResult:
    task_type: TaskType
    metric_summary: str
    top_features: list[tuple[str, float]]


def infer_task_type(target: pd.Series) -> TaskType:
    """Infer whether this is classification or regression."""
    if pd.api.types.is_numeric_dtype(target):
        unique_ratio = target.nunique(dropna=True) / max(len(target), 1)
        if target.nunique(dropna=True) <= 20 or unique_ratio < 0.05:
            return "classification"
        return "regression"
    return "classification"


def build_pipeline(X: pd.DataFrame, task_type: TaskType) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42)

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_top_features(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> list[tuple[str, float]]:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    importances = result.importances_mean

    ranked = sorted(
        zip(feature_names, importances, strict=False),
        key=lambda item: item[1],
        reverse=True,
    )
    return [(name, float(score)) for name, score in ranked[:10]]


def analyze_dataset(
    csv_path: Path,
    target_column: str,
    test_size: float = 0.2,
) -> AnalysisResult:
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")

    df = df.dropna(axis=0, how="all")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    task_type = infer_task_type(y)

    stratify = y if task_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    pipeline = build_pipeline(X, task_type)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if task_type == "classification":
        acc = accuracy_score(y_test, y_pred)
        metric_summary = (
            f"Task: Classification\\n"
            f"Accuracy: {acc:.4f}\\n"
            f"\\nDetailed report:\\n{classification_report(y_test, y_pred, zero_division=0)}"
        )
    else:
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metric_summary = (
            f"Task: Regression\\n"
            f"MAE: {mae:.4f}\\n"
            f"R²: {r2:.4f}"
        )

    top_features = get_top_features(pipeline, X_test, y_test)
    return AnalysisResult(task_type=task_type, metric_summary=metric_summary, top_features=top_features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Analyst for CSV datasets (Python + Machine Learning)")
    parser.add_argument("csv", type=Path, help="Path to input CSV file")
    parser.add_argument("target", help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_dataset(args.csv, args.target, args.test_size)

    print("=" * 70)
    print("AI ANALYST REPORT")
    print("=" * 70)
    print(result.metric_summary)
    print("\nTop influential features:")
    for rank, (name, score) in enumerate(result.top_features, start=1):
        print(f"{rank:>2}. {name:<40} importance={score:.6f}")


if __name__ == "__main__":
    main()

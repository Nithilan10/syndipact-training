# train.py
# Train a task-commitment detector on the combined dataset.
# Output: trained model artifacts + metrics
#
# Install:
#   pip install pandas scikit-learn joblib matplotlib
#
# Run:
#   python train.py --data data/processed/all_sources.parquet --out artifacts

from __future__ import annotations
import os
import json
import argparse
import joblib
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score
)

# -----------------------
# Utilities
# -----------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_df(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Unsupported file format. Use .parquet or .csv")

def bootstrap_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    If you don't have a real labeled dataset yet, we can create WEAK labels:
    - label=1 if is_strong_task_candidate True OR has_commitment_phrase True
    - else label=0
    This is NOT perfect, but it’s great for a first baseline.
    """
    if "label" in df.columns:
        return df

    # safest fallback if columns missing
    strong = df["is_strong_task_candidate"] if "is_strong_task_candidate" in df.columns else False
    commit = df["has_commitment_phrase"] if "has_commitment_phrase" in df.columns else False
    df["label"] = (strong.astype(bool) | commit.astype(bool)).astype(int)
    return df

def stratified_split(df: pd.DataFrame, label_col: str, seed: int = 42):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df[label_col]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df[label_col]
    )
    return train_df, val_df, test_df

def eval_model(pipe: Pipeline, X, y) -> Dict[str, Any]:
    preds = pipe.predict(X)
    probs = None
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(X)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    report = classification_report(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds).tolist()

    out = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": report,
    }

    if probs is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y, probs))
        except Exception:
            out["roc_auc"] = None

    return out

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to combined dataset (.parquet or .csv)")
    ap.add_argument("--out", default="artifacts", help="Output directory for model + metrics")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_len", type=int, default=5, help="Minimum cleaned text length")
    args = ap.parse_args()

    ensure_dir(args.out)

    df = load_df(args.data)

    # Basic cleaning assumptions
    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column")

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() >= args.min_len].reset_index(drop=True)

    # Labels
    df = bootstrap_labels(df)
    if "label" not in df.columns:
        raise ValueError("No labels found or created. Need a 'label' column or heuristics columns.")

    # Optional: remove duplicates
    if "source" in df.columns and "message_id" in df.columns:
        df = df.drop_duplicates(subset=["source", "message_id"], keep="first").reset_index(drop=True)

    # Split
    train_df, val_df, test_df = stratified_split(df, "label", seed=args.seed)

    # Save splits (useful for reproducibility)
    train_df.to_parquet(os.path.join(args.out, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(args.out, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.out, "test.parquet"), index=False)

    # Model: TF-IDF + Logistic Regression baseline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"  # important for imbalance
        ))
    ])

    X_train, y_train = train_df["text"], train_df["label"]
    X_val, y_val = val_df["text"], val_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]

    pipe.fit(X_train, y_train)

    # Evaluate
    val_metrics = eval_model(pipe, X_val, y_val)
    test_metrics = eval_model(pipe, X_test, y_test)

    metrics = {
        "dataset": {
            "total": int(len(df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
            "positive_rate_total": float(df["label"].mean()),
            "positive_rate_train": float(train_df["label"].mean()),
        },
        "val": val_metrics,
        "test": test_metrics,
        "model": {
            "type": "tfidf + logistic_regression",
            "features": "text",
        }
    }

    # Save artifacts
    model_path = os.path.join(args.out, "commitment_detector.joblib")
    joblib.dump(pipe, model_path)

    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Training complete")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("\n--- Validation summary ---")
    print(metrics["val"]["classification_report"])
    print("\n--- Test summary ---")
    print(metrics["test"]["classification_report"])


if __name__ == "__main__":
    main()

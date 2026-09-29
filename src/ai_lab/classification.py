"""Train and evaluate a small text classifier from labeled CSV data."""

import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def train_and_evaluate(csv_path: str | Path) -> dict:
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    if len(set(labels)) < 2 or min(labels.count(label) for label in set(labels)) < 4:
        raise ValueError("Need at least two classes with four examples each")
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {"train_count": len(x_train), "test_count": len(x_test),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "report": classification_report(y_test, predictions, zero_division=0)}

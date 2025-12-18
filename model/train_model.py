from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import json

# Paths
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "catboost_model.joblib"
EVAL_METRICS_PATH = ARTIFACT_DIR / "evaluation_metrics.json"
STATIC_DIR = Path("static")  # where plots will be saved
STATIC_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str  # "number" | "integer" | "category"
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    default: Optional[Any] = None


# ---------- New Feature Specification ----------
FEATURE_SPEC: List[Feature] = [
    Feature("website_domain_age_days", "integer", min=0, max=5000, step=1, default=365),
    Feature("has_https", "category", options=["yes", "no"], default="yes"),
    Feature("domain_reputation_score", "number", min=0, max=1, step=0.01, default=0.8),
    Feature("product_category", "category",
            options=["electronics", "fashion", "gift_card", "luxury", "digital", "other"],
            default="other"),
    Feature("product_price", "number", min=1, max=200000, step=1, default=1000),
    Feature("price_deviation_score", "number", min=0, max=1, step=0.01, default=0.1),
]


def get_feature_spec() -> List[Dict[str, Any]]:
    return [feature.__dict__ for feature in FEATURE_SPEC]


def _feature_names() -> List[str]:
    return [f.name for f in FEATURE_SPEC]


def _categorical_feature_names() -> List[str]:
    return [f.name for f in FEATURE_SPEC if f.kind == "category"]


# ---------- Synthetic dataset generation ----------
def _random_choice(options: List[str], p: Optional[List[float]] = None, size: int = 1,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    idx = rng.choice(len(options), size=size, replace=True, p=p)
    return np.array(options, dtype=object)[idx]


def generate_synthetic_dataset(num_rows: int = 15000, fraud_rate: float = 0.15,
                               random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    df = pd.DataFrame()

    # Website features
    df["website_domain_age_days"] = rng.integers(0, 5000, size=num_rows)
    df["has_https"] = _random_choice(["yes", "no"], p=[0.85, 0.15], size=num_rows, rng=rng)
    df["domain_reputation_score"] = np.clip(rng.beta(5, 2, size=num_rows), 0, 1)

    # Product features
    categories = ["electronics", "fashion", "gift_card", "luxury", "digital", "other"]
    probs = [0.25, 0.25, 0.1, 0.15, 0.1, 0.15]
    df["product_category"] = _random_choice(categories, p=probs, size=num_rows, rng=rng)

    base_prices = {
        "electronics": 40000,
        "fashion": 2000,
        "gift_card": 5000,
        "luxury": 80000,
        "digital": 1000,
        "other": 3000,
    }

    df["product_price"] = [
        max(1, rng.normal(base_prices[cat], base_prices[cat] * 0.3))
        for cat in df["product_category"]
    ]

    # Price deviation score
    df["price_deviation_score"] = np.clip(
        np.abs((df["product_price"] - df["product_category"].map(base_prices)) /
               df["product_category"].map(base_prices)),
        0, 1
    )

    # ---------- Risk Scoring ----------
    risk = np.zeros(num_rows, dtype=float)

    # New websites = riskier
    risk += np.clip((365 - df["website_domain_age_days"]) / 365, 0, 1) * 0.7
    # No HTTPS = risky
    risk += (df["has_https"] == "no").astype(float) * 0.6
    # Low reputation score
    risk += (1 - df["domain_reputation_score"]) * 0.8
    # High-risk product categories
    risky_cats = {"electronics": 0.4, "gift_card": 0.6, "luxury": 0.5}
    risk += df["product_category"].map(risky_cats).fillna(0)
    # Weird prices
    risk += df["price_deviation_score"] * 0.9

    # Normalize risk
    risk = (risk - risk.min()) / (risk.max() - risk.min() + 1e-8)

    # Label fraud if risk above threshold
    threshold = np.quantile(risk, 1.0 - fraud_rate)
    df["is_fraud"] = (risk >= threshold).astype(int)

    df = df[_feature_names() + ["is_fraud"]]
    return df


# ---------- Training ----------
def train_and_save_model(num_rows: int = 15000, fraud_rate: float = 0.15,
                         random_state: int = 42) -> Dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_dataset(num_rows=num_rows, fraud_rate=fraud_rate,
                                    random_state=random_state)

    feature_names = _feature_names()
    categorical_feature_names = _categorical_feature_names()

    # Train/test split
    train_df = df.sample(frac=0.8, random_state=random_state)
    test_df = df.drop(train_df.index)

    X_train, y_train = train_df[feature_names], train_df["is_fraud"]
    X_test, y_test = test_df[feature_names], test_df["is_fraud"]

    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.1,
        iterations=200,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False,
    )

    model.fit(X_train, y_train,
              cat_features=[feature_names.index(n) for n in categorical_feature_names])

    # Predictions
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
    with open(EVAL_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(STATIC_DIR / "confusion_matrix.png")
    plt.close()

    # Save artifacts
    artifacts = {
        "model": model,
        "feature_names": feature_names,
        "categorical_feature_names": categorical_feature_names,
    }
    joblib.dump(artifacts, ARTIFACT_PATH)
    return artifacts


def ensure_model_artifacts_exist() -> None:
    if not ARTIFACT_PATH.exists():
        train_and_save_model()


def load_artifacts() -> Dict[str, Any]:
    if not ARTIFACT_PATH.exists():
        ensure_model_artifacts_exist()
    return joblib.load(ARTIFACT_PATH)


# ---------- Inference ----------
def preprocess_input_row(raw: Dict[str, Any], artifacts: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[str]]:
    feature_names: List[str] = artifacts["feature_names"]

    values: Dict[str, Any] = {}
    spec_map: Dict[str, Feature] = {f.name: f for f in FEATURE_SPEC}

    for name in feature_names:
        feature = spec_map[name]
        raw_val = raw.get(name, None)

        try:
            if feature.kind == "number":
                val = float(raw_val if raw_val not in [None, ""] else feature.default)
                if feature.min is not None:
                    val = max(val, feature.min)
                if feature.max is not None:
                    val = min(val, feature.max)
                values[name] = val

            elif feature.kind == "integer":
                val = int(float(raw_val if raw_val not in [None, ""] else feature.default))
                if feature.min is not None:
                    val = max(val, feature.min)
                if feature.max is not None:
                    val = min(val, feature.max)
                values[name] = val

            elif feature.kind == "category":
                options = feature.options or []
                val = str(raw_val).strip() if raw_val else feature.default
                if options and val not in options:
                    if "other" in options:
                        val = "other"
                    else:
                        return pd.DataFrame(), f"Invalid value '{val}' for {name}"
                values[name] = val

        except Exception as exc:
            return pd.DataFrame(), f"Invalid value for {name}: {exc}"

    df = pd.DataFrame([values], columns=feature_names)
    return df, None


if __name__ == "__main__":
    print("[train_model] Training model for Website + Product fraud detection...")
    train_and_save_model()
    print(f"[train_model] Artifacts at: {ARTIFACT_PATH}")
    print(f"[train_model] Metrics saved at: {EVAL_METRICS_PATH}")

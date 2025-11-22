from __future__ import annotations
import io
from pathlib import Path
from typing import Any, Dict
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from flask import Flask, jsonify, render_template, request, send_file

from model.train_model import (
    ensure_model_artifacts_exist,
    get_feature_spec,
    load_artifacts,
    preprocess_input_row,
)

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

# Ensure model exists at startup
ensure_model_artifacts_exist()

# Load artifacts
ARTIFACTS = load_artifacts()
FEATURE_SPEC = get_feature_spec()

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/")
def index():
    return render_template(
        "index.html",
        feature_spec=FEATURE_SPEC,
        prediction=None,
        probability=None,
        batch_results=None,
        error=None,
        confusion_matrix="confusion_matrix.png" if (STATIC_DIR / "confusion_matrix.png").exists() else None,
    )


# ---------- NEW ENDPOINT ----------
@app.post("/predict_product")
def predict_product():
    """
    Predict fraud risk based on Website + Product details
    """
    try:
        payload: Dict[str, Any] = request.get_json(force=True)

        website_url = payload.get("website_url", "").strip()
        product_name = payload.get("product_name", "").strip()
        product_category = payload.get("product_category", "other")
        product_price = float(payload.get("product_price", 0.0))

        # --- Derive extra features ---
        # Fake logic for demo: in real app, call WHOIS, SSL checks, reputation APIs
        website_domain_age_days = 30 if "cheap" in website_url else 1000
        has_https = "yes" if website_url.startswith("https") else "no"
        domain_reputation_score = 0.3 if "scam" in website_url else 0.8

        # Price deviation heuristic (compare to average category price)
        base_prices = {
            "electronics": 40000,
            "fashion": 2000,
            "gift_card": 5000,
            "luxury": 80000,
            "digital": 1000,
            "other": 3000,
        }
        base_price = base_prices.get(product_category, 3000)
        price_deviation_score = min(abs(product_price - base_price) / base_price, 1.0)

        # Final model input
        row = {
            "website_domain_age_days": website_domain_age_days,
            "has_https": has_https,
            "domain_reputation_score": domain_reputation_score,
            "product_category": product_category,
            "product_price": product_price,
            "price_deviation_score": price_deviation_score,
        }

        row_df, error = preprocess_input_row(row, artifacts=ARTIFACTS)
        if error:
            return jsonify({"ok": False, "error": error}), 400

        model = ARTIFACTS["model"]
        proba = float(model.predict_proba(row_df)[:, 1][0])
        label = "Fraud" if proba >= 0.5 else "Legit"

        # Reasons for explainability
        reasons = []
        if website_domain_age_days < 90:
            reasons.append("Website domain is very new")
        if has_https == "no":
            reasons.append("Website does not use HTTPS")
        if domain_reputation_score < 0.5:
            reasons.append("Low website reputation score")
        if price_deviation_score > 0.5:
            reasons.append("Product price is unusual compared to market")
        if product_category in ["electronics", "gift_card", "luxury"]:
            reasons.append(f"High-risk product category: {product_category}")

        return jsonify({
            "ok": True,
            "label": label,
            "probability": proba,
            "reasons": reasons,
        })

    except Exception as exc:
        app.logger.exception("Error in /predict_product")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/upload")
def upload_csv():
    try:
        if "csv_file" not in request.files:
            return jsonify({"ok": False, "error": "No file uploaded"}), 400

        file = request.files["csv_file"]
        if file.filename == "":
            return jsonify({"ok": False, "error": "No file selected"}), 400

        if not file.filename.endswith(".csv"):
            return jsonify({"ok": False, "error": "Please upload a CSV file"}), 400

        content = file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))

        required_features = [f["name"] for f in FEATURE_SPEC]
        missing_features = [col for col in required_features if col not in df.columns]
        if missing_features:
            return jsonify({"ok": False, "error": f"Missing columns: {', '.join(missing_features)}"}), 400

        if len(df) > 10000:
            return jsonify({"ok": False, "error": "Maximum 10,000 rows allowed"}), 400

        model = ARTIFACTS.get("model")
        if model is None:
            return jsonify({"ok": False, "error": "Model not loaded"}), 500

        # Run predictions
        X = df[required_features]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        df["prediction"] = ["Fraud" if p == 1 else "Legit" for p in predictions]
        df["fraud_probability"] = probabilities

        total = len(df)
        fraud_count = int(predictions.sum())
        legit_count = total - fraud_count
        fraud_rate = round(fraud_count / total, 4)

        # Save results for download
        results_dir = Path("temp_results")
        results_dir.mkdir(exist_ok=True)
        results_file = results_dir / f"fraud_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(results_file, index=False)

        # Confusion Matrix (simulated ground truth if not provided)
        threshold = pd.Series(probabilities).quantile(0.9)
        y_true = (probabilities >= threshold).astype(int)
        y_pred = predictions

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.savefig(STATIC_DIR / "confusion_matrix.png")
        plt.close()

        preview_df = df.head(10).copy()
        preview_records = preview_df.to_dict(orient="records")

        return jsonify({
            "ok": True,
            "total": total,
            "fraud_count": fraud_count,
            "legit_count": legit_count,
            "fraud_rate": fraud_rate,
            "download_link": f"/download/{results_file.name}",
            "preview": preview_records,
            "confusion_matrix": "confusion_matrix.png",
        })

    except Exception as exc:
        app.logger.exception("Error processing uploaded file")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/download/<filename>")
def download_results(filename):
    file_path = Path("temp_results") / filename
    if not file_path.exists():
        return "File not found", 404
    return send_file(file_path, as_attachment=True, download_name=filename)


if __name__ == "__main__":
    app.run(debug=True)

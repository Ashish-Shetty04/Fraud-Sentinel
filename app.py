#!/usr/bin/env python3
import os
import re
import logging
from urllib.parse import urlparse
from typing import Optional, Any

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup

# -----------------------
# App setup
# -----------------------
logging.basicConfig(level=logging.INFO)

app = Flask(
    __name__,
    static_folder="static",
    static_url_path="/static"
)
CORS(app)

INDEX_PATH = "templates/index.html"

USER_AGENT = "Mozilla/5.0"
REQUEST_TIMEOUT = 10

# -----------------------
# Utility functions
# -----------------------
def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except:
        return ""


def is_suspicious_domain(url: str) -> bool:
    domain = domain_from_url(url)
    suspicious_tlds = (".shop", ".xyz", ".store", ".top", ".live")
    suspicious_words = ("offer", "discount", "sale", "deal", "cheap")

    if any(domain.endswith(tld) for tld in suspicious_tlds):
        return True
    if any(word in domain for word in suspicious_words):
        return True
    return False


def safe_get(url: str) -> Optional[str]:
    try:
        r = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        return r.text
    except:
        return None


def parse_product_info(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    data = {}

    if soup.title:
        data["title"] = soup.title.string.strip()

    price_match = soup.find(text=re.compile(r"₹|\d"))
    if price_match:
        m = re.search(r"[\d,]+", price_match)
        if m:
            data["price"] = m.group(0)

    return data


# -----------------------
# FRAUD SCORING LOGIC
# -----------------------
def simple_security_score(title: Optional[str], price: Any, url: str) -> int:
    score = 0

    if title:
        score += 1
    if price:
        score += 1
    if url.startswith("https://"):
        score += 1

    trusted_sites = (
        "amazon.in",
        "flipkart.com",
        "myntra.com",
        "snapdeal.com",
        "paytm.com"
    )

    domain = domain_from_url(url)
    if any(t in domain for t in trusted_sites):
        score += 2

    if is_suspicious_domain(url):
        score -= 3

    suspicious_keywords = [
        "replica", "copy", "duplicate", "fake",
        "cheap", "wholesale", "bulk",
        "original", "100%", "best price",
        "offer", "discount", "sale"
    ]

    title_low = (title or "").lower()
    if any(word in title_low for word in suspicious_keywords):
        score -= 2

    try:
        p = float(re.sub(r"[^\d.]", "", str(price)))
        if p < 1500:
            score -= 2
    except:
        pass

    return max(0, min(5, score))


def simple_prediction(score: int) -> dict:
    if score <= 1:
        return {"label": "suspicious", "probability": 0.95}
    elif score == 2:
        return {"label": "uncertain", "probability": 0.65}
    else:
        return {"label": "likely_safe", "probability": round(0.7 + 0.1 * score, 2)}


# -----------------------
# FALLBACK RECOMMENDATIONS
# -----------------------
def fallback_recommendations(product_name: str):
    return [
        {
            "title": f"{product_name} – Amazon",
            "price": "₹6495",
            "url": "https://www.amazon.in",
            "site": "amazon.in"
        },
        {
            "title": f"{product_name} – Flipkart",
            "price": "₹6599",
            "url": "https://www.flipkart.com",
            "site": "flipkart.com"
        },
        {
            "title": f"{product_name} – Myntra",
            "price": "₹6999",
            "url": "https://www.myntra.com",
            "site": "myntra.com"
        },
        {
            "title": f"{product_name} – Official Store",
            "price": "₹6495",
            "url": "https://www.casio.com/in",
            "site": "casio.com"
        }
    ]


# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET"])
def index():
    if os.path.exists(INDEX_PATH):
        return send_file(INDEX_PATH)
    return "index.html not found", 404


@app.route("/predict_product", methods=["POST"])
def predict_product():
    payload = request.get_json(force=True) or {}

    website_url = payload.get("website_url", "").strip()
    product_name = payload.get("product_name", "").strip()
    product_price = payload.get("product_price")

    product_details = {
        "title": product_name,
        "price": product_price,
        "url": website_url
    }

    # Try to enrich data by scraping
    if website_url:
        html = safe_get(website_url)
        if html:
            scraped = parse_product_info(html)
            product_details.update({k: v for k, v in scraped.items() if v})

    # Fraud score
    score = simple_security_score(
        product_details.get("title"),
        product_details.get("price"),
        website_url
    )

    prediction = simple_prediction(score)

    # ✅ FIX probability (convert to percentage for UI)
    prediction["probability"] = round(prediction["probability"] * 100, 2)

    # ✅ Recommendations logic
    if prediction["label"] == "likely_safe":
        related_products = fallback_recommendations(
            product_details.get("title", "Product")
        )
    else:
        related_products = []  # Do not recommend for suspicious products

    return jsonify({
        "product_details": product_details,
        "security_score": score * 20,  # show as percentage (0–100)
        "prediction": prediction,
        "related_products": related_products
    })
@app.route("/related_products", methods=["POST"])
def related_products():
    payload = request.get_json(force=True) or {}
    product_name = payload.get("product_name", "Product")

    return jsonify({
        "related_products": fallback_recommendations(product_name)
    })





# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server running at http://127.0.0.1:{port}")
    app.run(debug=True)

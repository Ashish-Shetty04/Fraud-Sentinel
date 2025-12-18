<<<<<<< HEAD
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
=======
#!/usr/bin/env python3
import os
import re
import json
import logging
from urllib.parse import urlparse, quote_plus, parse_qs, unquote_plus
from typing import Optional, Any

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file")
except ImportError:
    print("python-dotenv not installed. Install it with: pip install python-dotenv")
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.DEBUG)
app = Flask(
    __name__,
    static_folder=r"C:\Projects\E-commercce\static",
    static_url_path="/static"
)
CORS(app)

# ←----- SET TO YOUR LOCAL index.html PATH -----
INDEX_PATH = r"C:\Projects\E-commercce\templates\index.html"
# ---------------------------------------------
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
if SERPAPI_KEY:
    # Mask the key for security (show first 8 and last 4 chars)
    masked_key = SERPAPI_KEY[:8] + "..." + SERPAPI_KEY[-4:] if len(SERPAPI_KEY) > 12 else "***"
    print(f"✓ SERPAPI_KEY loaded successfully (length: {len(SERPAPI_KEY)}, key: {masked_key})")
    app.logger.info("SERPAPI_KEY loaded successfully (length: %d)", len(SERPAPI_KEY))
else:
    print("✗ SERPAPI_KEY not found in environment variables. Check .env file or environment setup.")
    app.logger.warning("SERPAPI_KEY not found in environment variables. Check .env file or environment setup.")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
REQUEST_TIMEOUT = 12
MAX_RELATED_RESULTS = 8


# -----------------------
# Utilities
# -----------------------
def domain_of(url: str) -> str:
    if not url:
        return ""
    try:
        p = urlparse(url)
        return p.netloc.lower().replace("www.", "")
    except Exception:
        try:
            return url.lower().split("/")[0]
        except Exception:
            return ""


def is_same_website(input_domain: str, product_url: str, product_site: str) -> bool:
    """
    Check if a product is from the same website as the input domain.
    Returns True if they match (should be filtered out), False otherwise.
    """
    if not input_domain:
        return False
    
    input_domain_lower = input_domain.lower().strip()
    input_domain_key = input_domain_lower.split(".")[0] if "." in input_domain_lower else input_domain_lower
    
    product_url_lower = (product_url or "").strip().lower()
    product_site_lower = (product_site or "").strip().lower()
    
    product_domain = domain_of(product_url) if product_url else ""
    product_domain_lower = product_domain.lower() if product_domain else ""
    product_domain_key = product_domain_lower.split(".")[0] if product_domain_lower and "." in product_domain_lower else product_domain_lower
    
    # Check 1: Exact domain match
    if input_domain_lower and product_domain_lower and input_domain_lower == product_domain_lower:
        return True
    
    # Check 2: Domain key match (e.g., "flipkart" == "flipkart")
    if input_domain_key and product_domain_key and input_domain_key == product_domain_key:
        return True
    
    # Check 3: Input domain key appears in product domain/URL/site
    if input_domain_key:
        if (input_domain_key in product_domain_lower or 
            input_domain_key in product_url_lower or 
            input_domain_key in product_site_lower):
            return True
    
    # Check 4: Product domain key appears in input domain
    if product_domain_key and product_domain_key in input_domain_lower:
        return True
    
    # Check 5: Input domain appears in product URL/site
    if input_domain_lower:
        if (input_domain_lower in product_url_lower or 
            input_domain_lower in product_site_lower):
            return True
    
    return False


def safe_get(url: str, headers: Optional[dict] = None, timeout: int = REQUEST_TIMEOUT) -> Optional[str]:
    headers = headers or {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r.text
    except Exception as e:
        app.logger.debug("safe_get failed for %s: %s", url, e)
        return None


def parse_product_info(html_text: str) -> dict:
    if not html_text:
        return {}
    soup = BeautifulSoup(html_text, "html.parser")
    title = None
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og.get("content").strip()
    if not title:
        meta_title = soup.find("meta", attrs={"name": "title"})
        if meta_title and meta_title.get("content"):
            title = meta_title.get("content").strip()
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    price = None
    meta_price = soup.find("meta", property="product:price:amount") or soup.find("meta", attrs={"name": "price"})
    if meta_price and meta_price.get("content"):
        price = meta_price.get("content").strip()
    if not price:
        selectors = ['[id*="price"]', '[class*="price"]', '[itemprop="price"]', '[data-test*="price"]']
        for sel in selectors:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(" ", strip=True)
                if text:
                    m = re.search(r"[\d\.,]+", text.replace("\u20b9", ""))
                    if m:
                        price = m.group(0)
                        break
    out = {}
    if title:
        out["title"] = title
    if price:
        out["price"] = price
    return out


# -----------------------
# SerpAPI helper (filter aggregator/search pages)
# -----------------------
def serpapi_google_shopping_search(query: str, num: int = 10) -> list:
    """
    SerpAPI shopping wrapper with aggregator filtering.

    - Extracts link from a variety of fields
    - Normalizes/decodes links
    - Filters out aggregator/search/listing domains (google, bing, pinterest, etc.)
      so the caller can fall back to the DDG scraper when SerpAPI only returns
      aggregator pages instead of merchant product pages.
    """
    if not SERPAPI_KEY:
        app.logger.warning("SERPAPI_KEY not set, skipping SerpAPI search")
        return []
    try:
        params = {"engine": "google_shopping", "q": query, "api_key": SERPAPI_KEY, "num": num, "gl": "in", "hl": "en"}
        app.logger.info("SerpAPI request: query='%s' (length=%d)", query, len(query))
        app.logger.debug("SerpAPI params: engine=google_shopping, gl=in, num=%d", num)
        
        try:
            resp = requests.get("https://serpapi.com/search.json", params=params, timeout=REQUEST_TIMEOUT)
            app.logger.info("SerpAPI response status: %s", resp.status_code)
            
            if not resp.ok:
                error_text = resp.text[:500] if resp.text else "No error text"
                app.logger.error("SerpAPI returned status %s: %s", resp.status_code, error_text)
                try:
                    error_data = resp.json()
                    if "error" in error_data:
                        error_msg = error_data.get("error")
                        app.logger.error("SerpAPI error message: %s", error_msg)
                        # Check for common errors
                        if "Invalid API key" in str(error_msg) or "invalid_api_key" in str(error_msg).lower():
                            app.logger.error("INVALID API KEY - Check your SERPAPI_KEY in .env file")
                        elif "rate limit" in str(error_msg).lower() or "quota" in str(error_msg).lower():
                            app.logger.error("RATE LIMIT EXCEEDED - You may have exceeded your SerpAPI quota")
                except Exception as e:
                    app.logger.error("Could not parse error response: %s", str(e))
                return []
        except requests.exceptions.Timeout:
            app.logger.error("SerpAPI request timed out after %d seconds", REQUEST_TIMEOUT)
            return []
        except requests.exceptions.RequestException as e:
            app.logger.error("SerpAPI request exception: %s", str(e))
            return []
        data = resp.json()
        
        # Check for errors in response first
        if "error" in data:
            error_msg = data.get("error", "Unknown error")
            app.logger.error("SerpAPI error in response: %s", error_msg)
            return []
        
        raw_results = data.get("shopping_results", [])[:num]
        app.logger.info("SerpAPI returned %d shopping_results (raw) for query: %s", len(raw_results), query)
        
        # Also check for organic_results as fallback
        if not raw_results:
            organic = data.get("organic_results", [])[:num * 2]  # Get more to filter
            app.logger.info("No shopping_results, checking organic_results: %d found", len(organic))
            if organic:
                # Convert organic results to shopping format, prioritizing e-commerce sites
                for org in organic:
                    link = org.get("link") or org.get("url") or ""
                    if link:
                        # Prioritize e-commerce sites but don't exclude others
                        is_ecommerce = any(domain in link.lower() for domain in ["amazon", "flipkart", "myntra", "snapdeal", "paytm", "ebay", "shopclues"])
                        if is_ecommerce or len(raw_results) < num:  # Always add e-commerce, add others if we need more
                            raw_results.append({
                                "title": org.get("title", ""),
                                "link": link,
                                "price": None,
                                "source": {"name": domain_of(link)}
                            })
                app.logger.info("Converted %d organic results to shopping format", len(raw_results))
        
        # If still no results, log response structure for debugging
        if not raw_results:
            app.logger.warning("No shopping_results or suitable organic_results found")
            app.logger.debug("Response keys: %s", list(data.keys())[:10])
            # Try to find any product-like results
            for key in ["products", "items", "results"]:
                if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                    app.logger.info("Found alternative results key '%s' with %d items", key, len(data[key]))
        results = []

        def try_extract_link(it: dict) -> str:
            # Google Shopping API returns product_link (not link)
            # Priority order: product_link > link > other fields
            candidates = [
                it.get("product_link"),  # PRIMARY: Google Shopping uses this
                it.get("link"),
                it.get("url"),
                it.get("href"),
                it.get("product_id_link"),
                # Check nested structures
                it.get("source", {}).get("url") if isinstance(it.get("source"), dict) else None,
                it.get("source", {}).get("link") if isinstance(it.get("source"), dict) else None,
                it.get("merchant", {}).get("url") if isinstance(it.get("merchant"), dict) else None,
                it.get("merchant", {}).get("link") if isinstance(it.get("merchant"), dict) else None,
                it.get("offers", {}).get("url") if isinstance(it.get("offers"), dict) else None,
            ]
            
            # Filter out None values
            candidates = [c for c in candidates if c]
            
            for c in candidates:
                if not c:
                    continue
                if isinstance(c, dict):
                    c = c.get("link") or c.get("url") or c.get("href") or ""
                c = str(c).strip()
                if not c:
                    continue
                # decode common wrappers: /url?q=... or encoded https%3A%2F%2F...
                if c.startswith("/url?") or c.startswith("/translate?") or ("?q=" in c and c.startswith("/")):
                    parsed = urlparse(c)
                    qs = parse_qs(parsed.query)
                    for key in ("q", "url", "uddg", "u"):
                        if key in qs and qs[key]:
                            return unquote_plus(qs[key][0])
                if re.search(r"https?%3A%2F%2F", c):
                    try:
                        return unquote_plus(c)
                    except Exception:
                        pass
                # if not absolute, try to prefix with https:// if it looks url-like
                if c and not c.lower().startswith("http"):
                    if "." in c:
                        return "https://" + c
                    continue
                return c
            return ""

        for idx, it in enumerate(raw_results):
            title = it.get("title") or it.get("product_title") or it.get("name") or ""
            raw_price = it.get("extracted_price") or it.get("price") or it.get("price_raw") or it.get("formatted_price") or None
            link = try_extract_link(it) or ""
            link = str(link)
            
            # Log if link is still empty for debugging (only for first result to avoid spam)
            if not link and idx == 0:
                app.logger.warning("No link extracted from first result. Available keys: %s", list(it.keys()))
                app.logger.warning("First result data (first 5 fields): %s", {k: str(v)[:100] for k, v in list(it.items())[:5]})

            site = domain_of(link)
            if not site:
                # fallback to merchant/source fields
                src = it.get("source") or it.get("merchant") or ""
                if isinstance(src, dict):
                    site = (src.get("domain") or src.get("name") or "").lower()
                else:
                    site = str(src).lower()
                    # Extract domain from source name (e.g., "Acer Store - India" -> "acer")
                    # Or if source contains merchant name, use that
                    if "amazon" in site.lower():
                        site = "amazon.in"
                    elif "flipkart" in site.lower():
                        site = "flipkart.com"
                    elif "myntra" in site.lower():
                        site = "myntra.com"
                    elif "snapdeal" in site.lower():
                        site = "snapdeal.com"
                    elif "paytm" in site.lower():
                        site = "paytm.com"
            
            # For Google Shopping links, try to extract merchant from source
            if "google.com" in site and link:
                src = it.get("source") or ""
                if isinstance(src, str):
                    if "amazon" in src.lower():
                        site = "amazon.in"
                    elif "flipkart" in src.lower():
                        site = "flipkart.com"
                    elif "myntra" in src.lower():
                        site = "myntra.com"
                    elif "snapdeal" in src.lower():
                        site = "snapdeal.com"
                    elif "paytm" in src.lower():
                        site = "paytm.com"
                    # Extract domain from source name if it contains a domain
                    if site and ("amazon" in site or "flipkart" in site or "myntra" in site or "snapdeal" in site or "paytm" in site):
                        # Source might be "Amazon.in" or "Flipkart.com" - extract domain
                        if "amazon" in site.lower():
                            site = "amazon.in"
                        elif "flipkart" in site.lower():
                            site = "flipkart.com"
                        elif "myntra" in site.lower():
                            site = "myntra.com"
                        elif "snapdeal" in site.lower():
                            site = "snapdeal.com"
                        elif "paytm" in site.lower():
                            site = "paytm.com"

            # parse price number
            price_val = None
            if raw_price is not None:
                try:
                    if isinstance(raw_price, (int, float)):
                        price_val = float(raw_price)
                    else:
                        p = re.sub(r"[^\d.]", "", str(raw_price))
                        if p:
                            price_val = float(p)
                except Exception:
                    price_val = None

            # ensure link is http(s) absolute, else empty (so caller can fallback)
            if link and not link.lower().startswith("http"):
                app.logger.debug("serpapi: skipping non-absolute link: %s", link)
                link = ""

            results.append({"title": title, "price": price_val, "url": link, "site": site})

        # Filter out aggregator / search/listing domains
        # IMPORTANT: Google Shopping product links (google.com/search?ibp=oshop) are VALID - they link to real products!
        aggregator_domains = ("bing.", "pinterest.", "facebook.", "twitter.", "reddit.", "t.co", "linkedin.")
        filtered = []
        removed = 0
        removed_no_url = 0
        removed_agg = 0
        for r in results:
            s = (r.get("site") or "").lower()
            urlstr = (r.get("url") or "").lower()
            
            # Google Shopping product pages are valid product links - keep them!
            is_google_shopping_product = "google.com/search" in urlstr and ("ibp=oshop" in urlstr or "udm=28" in urlstr)
            
            is_agg = False
            # Only filter out aggregators if it's NOT a Google Shopping product page
            if not is_google_shopping_product:
                for ad in aggregator_domains:
                    if ad in s or ad in urlstr:
                        is_agg = True
                        break
            
            if is_agg:
                removed += 1
                removed_agg += 1
                continue
            
            # Accept results with URLs (including Google Shopping product links)
            if r.get("url"):
                filtered.append(r)
            else:
                removed += 1
                removed_no_url += 1
                if removed_no_url == 1:
                    app.logger.warning("Removing result with no URL. Title: %s, Site: %s", r.get("title", "")[:50], r.get("site", ""))
        
        if removed_no_url > 0:
            app.logger.warning("Removed %d results with no URL, %d aggregator results. Kept %d results.", removed_no_url, removed_agg, len(filtered))

        app.logger.debug(
            "serpapi_google_shopping_search parsed %d results, removed %d aggregator/empty entries, kept %d",
            len(results), removed, len(filtered)
        )

        # return filtered merchant links, or empty list to trigger fallback
        return filtered if filtered else []
    except requests.exceptions.Timeout:
        app.logger.error("SerpAPI request timed out for query: %s", query)
        return []
    except requests.exceptions.RequestException as e:
        app.logger.error("SerpAPI request exception for query %s: %s", query, str(e))
        return []
    except Exception as e:
        app.logger.exception("serpapi_google_shopping_search error for query %s: %s", query, e)
        return []


# -----------------------
# DuckDuckGo scraping fallback (robust)
# -----------------------
def find_related_products_by_scrape(query: str, max_results: int = 6) -> list:
    if not query:
        app.logger.warning("DDG scrape: empty query")
        return []
    try:
        # Try a shopping-specific search query
        shopping_query = f"{query} buy online india"
        q = quote_plus(shopping_query)
        ddg_url = f"https://html.duckduckgo.com/html/?q={q}"
        app.logger.debug("DDG scrape: query=%s, url=%s", shopping_query, ddg_url)
        html = safe_get(ddg_url)
        if not html:
            app.logger.warning("DDG fetch failed for query: %s (no HTML returned)", query)
            return []
        if len(html) < 1000:
            app.logger.warning("DDG returned very short HTML (%d chars), might be blocked or error page", len(html))
        soup = BeautifulSoup(html, "html.parser")
        results = []

        anchors = []
        # Try multiple selectors to find result links
        anchors += soup.select("a.result__a")
        anchors += soup.select("a.result--url__link")
        anchors += soup.select("a.result__url")
        anchors += soup.select("a[href^='http']")
        anchors += soup.select("a[href^='/l/']")
        anchors += soup.select("a[href^='/link?']")
        anchors += soup.select("a.web-result")
        anchors += soup.select("a[class*='result']")
        
        app.logger.debug("DDG found %d anchor elements", len(anchors))

        seen_urls = set()
        for a in anchors:
            href = a.get("href") or ""
            if not href:
                continue
            title = a.get_text(" ", strip=True) or ""
            real_url = href
            if real_url.startswith("/l/") or real_url.startswith("/link"):
                parsed = urlparse(real_url)
                qs = parsed.query or ""
                m = re.search(r"(uddg|u|url)=([^&]+)", qs)
                if m:
                    try:
                        real_url = unquote_plus(m.group(2))
                    except Exception:
                        real_url = real_url
                else:
                    uu = re.search(r"(https?%3A%2F%2F[^&]+)", real_url)
                    if uu:
                        try:
                            real_url = unquote_plus(uu.group(1))
                        except Exception:
                            real_url = real_url
                    else:
                        app.logger.debug("could not decode ddg wrapper href=%s", href)
                        continue
            if real_url.startswith("/"):
                real_url = "https://html.duckduckgo.com" + real_url
            if a.has_attr("data-href"):
                real_url = a["data-href"]
            if not real_url.lower().startswith("http"):
                app.logger.debug("skipping non-http ddg link: %s", real_url)
                continue
            norm = re.sub(r"(\?.*)$", "", real_url).rstrip("/")
            if not norm or norm in seen_urls:
                continue
            seen_urls.add(norm)
            site = domain_of(real_url)
            
            # Skip if it's clearly not a product page (aggregators, search pages, etc.)
            skip_domains = ("google.", "bing.", "pinterest.", "facebook.", "twitter.", "reddit.", "youtube.", "wikipedia.")
            if any(skip in site.lower() for skip in skip_domains):
                continue
            
            # Only fetch price for e-commerce sites to speed things up
            price = None
            ecommerce_domains = ("amazon.", "flipkart.", "myntra.", "snapdeal.", "paytm.", "ebay.", "shopclues.")
            if any(ec in site.lower() for ec in ecommerce_domains):
                try:
                    candidate_html = safe_get(real_url, timeout=5)  # Shorter timeout
                    if candidate_html:
                        info = parse_product_info(candidate_html)
                        if info.get("price"):
                            price = info.get("price")
                        if not title and info.get("title"):
                            title = info.get("title")
                except Exception as e:
                    app.logger.debug("candidate fetch failed for %s: %s", real_url[:80], str(e)[:50])
            
            if not title:
                title = norm.split("/")[-1].replace("-", " ").replace("_", " ")[:200]  # Use URL as fallback title
            
            results.append({"title": title, "price": price, "url": real_url, "site": site})
            if len(results) >= max_results:
                break
        
        app.logger.info("DDG fallback found %d candidates for query=%s", len(results), query)
        if len(results) == 0:
            app.logger.warning("DDG found 0 results. HTML length: %d, anchors found: %d", 
                             len(html) if html else 0, len(anchors))
        return results[:max_results]
    except requests.exceptions.Timeout:
        app.logger.error("DDG scrape timed out for query: %s", query)
        return []
    except requests.exceptions.RequestException as e:
        app.logger.error("DDG scrape request exception for query %s: %s", query, str(e))
        return []
    except Exception as e:
        app.logger.exception("find_related_products_by_scrape error for query %s: %s", query, e)
        return []


# -----------------------
# Helpers to create consistent output
# -----------------------
def domain_from_url(u: str) -> str:
    try:
        if not u:
            return ""
        p = urlparse(u)
        return p.netloc.lower().replace("www.", "")
    except Exception:
        return ""


def clean_site_value(raw_site: Any, url: str) -> str:
    if not raw_site:
        return domain_from_url(url) or ""
    s = str(raw_site).strip()
    if s.isdigit() or (len(s) > 30 and s.replace("-", "").replace("_", "").isalnum()):
        return domain_from_url(url) or ""
    return s.lower()


def format_price(raw_price: Any) -> Optional[str]:
    if raw_price is None:
        return None
    try:
        if isinstance(raw_price, (int, float)):
            return f"₹{raw_price:.2f}"
        s = str(raw_price).strip()
        if any(c in s for c in ["₹", "$", "£", "Rs"]):
            return s
        num = re.sub(r"[^\d\.]", "", s)
        if num:
            return f"₹{float(num):.2f}"
        return s
    except Exception:
        return str(raw_price)


# -----------------------
# Basic security scoring + prediction
# -----------------------
def simple_security_score(title: Optional[str], price: Any, url: str) -> int:
    score = 0
    try:
        if title and str(title).strip():
            score += 1
        if price is not None and str(price).strip():
            pstr = str(price)
            if re.search(r"\d", pstr):
                score += 1
        if isinstance(url, str) and url.lower().startswith("https://"):
            score += 1
        trusted = ("amazon.in", "flipkart.com", "myntra.com", "paytm.com", "snapdeal.com", "ebay.in")
        domain = domain_from_url(url)
        if domain and any(t in domain for t in trusted):
            score += 1
        suspicious_keywords = [
            "replica", "copy", "duplicate", "fake", "knockoff",
            "cheap", "wholesale", "bulk", "discount code", "free", "call now",
            "cash on delivery only", "urgent", "limited stock", "best price"
        ]
        title_low = (title or "").lower()
        for kw in suspicious_keywords:
            if kw in title_low:
                score = max(0, score - 1)
                break
    except Exception:
        return 0
    score = max(0, min(5, int(score)))
    return score


def simple_prediction(score: int) -> dict:
    try:
        s = int(score)
    except Exception:
        s = 0
    if s <= 1:
        label = "suspicious"
        probability = 0.95 if s == 0 else 0.80
    elif s == 2:
        label = "uncertain"
        probability = 0.60
    else:
        label = "likely_safe"
        probability = 0.7 + 0.1 * (s - 3)
        probability = min(0.95, probability)
    return {"label": label, "probability": round(float(probability), 2)}


# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET"])
def serve_index():
    if os.path.exists(INDEX_PATH):
        return send_file(INDEX_PATH)
    return "<h3>index.html not found at {}</h3>".format(INDEX_PATH), 404


@app.route("/predict_product", methods=["POST"])
def predict_product():
    try:
        payload = request.get_json(force=True) or {}
        website_url = (payload.get("website_url") or "").strip()
        product_name = (payload.get("product_name") or "").strip()
        product_price = payload.get("product_price") or payload.get("price") or None

        product_details = {"title": product_name or "", "price": product_price, "url": website_url}
        if website_url:
            html = safe_get(website_url)
            if html:
                info = parse_product_info(html)
                if info.get("title") and not product_name:
                    product_details["title"] = info.get("title")
                if info.get("price") and not product_price:
                    product_details["price"] = info.get("price")

        security_score = simple_security_score(product_details.get("title"), product_details.get("price"), website_url)
        pred = simple_prediction(security_score)

        related = []
        candidate_title = product_details.get("title") or product_name or ""
        if candidate_title:
            if SERPAPI_KEY:
                app.logger.debug("Using SerpAPI for related products for: %s", candidate_title)
                related = serpapi_google_shopping_search(candidate_title, num=MAX_RELATED_RESULTS)
            if not related:
                app.logger.debug("Falling back to DDG scrape for: %s", candidate_title)
                related = find_related_products_by_scrape(candidate_title, max_results=MAX_RELATED_RESULTS)

        # normalize related entries
        normalized = []
        for r in (related or [])[:MAX_RELATED_RESULTS]:
            url_val = r.get("url") or ""
            raw_site = r.get("site") or r.get("source") or r.get("merchant") or ""
            site_val = clean_site_value(raw_site, url_val)
            price_val = format_price(r.get("price") or r.get("extracted_price") or None)
            app.logger.debug("normalizing related result url=%s title=%s", url_val, r.get("title"))
            normalized.append({
                "title": (r.get("title") or r.get("name") or "").strip()[:300],
                "price": price_val,
                "url": url_val,
                "site": site_val
            })

        # Filter out products from the same website domain
        filtered_normalized = []
        input_domain = domain_of(website_url)
        if input_domain:
            app.logger.info("PREDICT FILTERING: input_domain=%s, total products=%d", input_domain, len(normalized))
            for it in normalized:
                product_url = it.get("url") or ""
                product_site = it.get("site") or ""
                if is_same_website(input_domain, product_url, product_site):
                    app.logger.debug("PREDICT FILTERED OUT: site=%s, url=%s", product_site, product_url[:80])
                    continue
                filtered_normalized.append(it)
            app.logger.info("PREDICT FILTERING RESULT: kept %d out of %d", len(filtered_normalized), len(normalized))
        else:
            filtered_normalized = normalized

        resp = {
            "product_details": product_details,
            "security_score": security_score,
            "prediction": {"label": pred["label"], "probability": pred["probability"]},
            "related_products": filtered_normalized
        }
        return jsonify(resp)
    except Exception as e:
        app.logger.exception("predict_product error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/related_products", methods=["POST"])
def related_products():
    try:
        payload = request.get_json(force=True) or {}
        product_name = (payload.get("product_name") or "").strip()
        website_url = (payload.get("website_url") or "").strip()

        candidate_title = product_name or ""
        if website_url and not candidate_title:
            page_html = safe_get(website_url)
            if page_html:
                info = parse_product_info(page_html)
                candidate_title = info.get("title") or candidate_title

        app.logger.info("related_products called for title: %s, website_url: %s", candidate_title, website_url)
        app.logger.info("SERPAPI_KEY is %s", "SET" if SERPAPI_KEY else "NOT SET")

        related = []
        if candidate_title:
            # Try multiple search query strategies
            search_queries = []
            
            # Strategy 1: Full title (if not too long)
            if len(candidate_title) <= 150:
                search_queries.append(candidate_title)
            
            # Strategy 2: Extract key terms - brand + model number
            words = candidate_title.split()
            model_match = re.search(r'\b[A-Z0-9]+[-]?[A-Z0-9]+\b', candidate_title)
            # Try to find brand (first capitalized word, or common brands)
            brand_match = re.search(r'\b([A-Z][a-z]+)\b', candidate_title)  # First capitalized word (usually brand)
            # Also check for common brands
            common_brands = ["acer", "dell", "hp", "lenovo", "asus", "msi", "samsung", "apple", "lg"]
            brand_name = None
            for brand in common_brands:
                if brand.lower() in candidate_title.lower():
                    brand_name = brand.capitalize()
                    break
            if not brand_name and brand_match:
                brand_name = brand_match.group(1)
            
            if brand_name and model_match:
                search_queries.append(f"{brand_name} {model_match.group()}")
            
            # Strategy 3: Just brand name (if found)
            if brand_name:
                if brand_name not in search_queries:
                    search_queries.append(brand_name)
            
            # Strategy 4: First 5-7 words (usually contains brand and key specs)
            first_words = ' '.join(words[:7])
            if first_words and first_words not in search_queries:
                search_queries.append(first_words)
            
            # Strategy 5: First 3 words (minimal)
            if len(words) >= 3:
                minimal = ' '.join(words[:3])
                if minimal not in search_queries:
                    search_queries.append(minimal)
            
            # Strategy 6: Brand + "laptop" or "gaming laptop" if it's a laptop
            if brand_name:
                if "laptop" in candidate_title.lower() or "notebook" in candidate_title.lower():
                    laptop_query = f"{brand_name} laptop"
                    if laptop_query not in search_queries:
                        search_queries.append(laptop_query)
                if "gaming" in candidate_title.lower():
                    gaming_query = f"{brand_name} gaming laptop"
                    if gaming_query not in search_queries:
                        search_queries.append(gaming_query)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in search_queries:
                q_lower = q.lower()
                if q_lower not in seen:
                    seen.add(q_lower)
                    unique_queries.append(q)
            
            app.logger.info("Trying %d search query strategies: %s", len(unique_queries), unique_queries)
            
            # Try each query until we get results
            last_error = None
            for idx, search_query in enumerate(unique_queries):
                if related and len(related) > 0:
                    app.logger.info("Found results with query %d, stopping search", idx + 1)
                    break  # Stop if we already have results
                    
                app.logger.info("Trying search query %d/%d: %s", idx + 1, len(unique_queries), search_query)
                
                if SERPAPI_KEY:
                    app.logger.info("Using SerpAPI with query=%s", search_query)
                    try:
                        related = serpapi_google_shopping_search(search_query, num=MAX_RELATED_RESULTS)
                        app.logger.info("SerpAPI returned %d results", len(related))
                    except Exception as e:
                        app.logger.error("SerpAPI exception: %s", str(e))
                        last_error = f"SerpAPI error: {str(e)}"
                        related = []
                else:
                    app.logger.info("SERPAPI_KEY not set, skipping SerpAPI")
                
                if not related or len(related) == 0:
                    app.logger.info("Using DDG fallback with query=%s", search_query)
                    try:
                        ddg_results = find_related_products_by_scrape(search_query, max_results=MAX_RELATED_RESULTS)
                        app.logger.info("DDG returned %d results", len(ddg_results))
                        if ddg_results and len(ddg_results) > 0:
                            related = ddg_results
                    except Exception as e:
                        app.logger.error("DDG exception: %s", str(e))
                        last_error = f"DDG error: {str(e)}"
                        related = []
            
            if not related or len(related) == 0:
                error_msg = f"All search strategies failed. Tried {len(unique_queries)} queries."
                if last_error:
                    error_msg += f" Last error: {last_error}"
                app.logger.error(error_msg)
                app.logger.error("Queries tried: %s", unique_queries[:5])
                
                # Last resort: Try a very simple generic query to test if APIs work at all
                if brand_name:
                    test_query = f"{brand_name} laptop"
                    app.logger.warning("Trying last resort test query: %s", test_query)
                    try:
                        if SERPAPI_KEY:
                            test_results = serpapi_google_shopping_search(test_query, num=3)
                            app.logger.info("Test query SerpAPI returned %d results", len(test_results))
                            if test_results:
                                related = test_results
                        if not related:
                            test_results = find_related_products_by_scrape(test_query, max_results=3)
                            app.logger.info("Test query DDG returned %d results", len(test_results))
                            if test_results:
                                related = test_results
                    except Exception as e:
                        app.logger.error("Test query also failed: %s", str(e))
                        last_error = f"Test query failed: {str(e)}"
        else:
            app.logger.warning("No candidate_title available for related products search")

        # Store whether search returned results (for debug info)
        search_returned_results = len(related) > 0 if related else False
        search_error_message = last_error if 'last_error' in locals() and last_error else None
        
        # normalize list
        normalized = []
        for r in (related or [])[:MAX_RELATED_RESULTS]:
            url_val = r.get("url") or ""
            raw_site = r.get("site") or r.get("source") or r.get("merchant") or ""
            site_val = clean_site_value(raw_site, url_val)
            price_val = format_price(r.get("price") or None)
            normalized.append({
                "title": (r.get("title") or "").strip(),
                "price": price_val,
                "url": url_val,
                "site": site_val
            })

        input_domain = domain_of(website_url)
        if "flipkart" in input_domain:
            competitors = ["amazon.in", "myntra.com", "snapdeal.com", "paytm.com"]
        elif "amazon." in input_domain:
            competitors = ["flipkart.com", "myntra.com", "snapdeal.com", "paytm.com"]
        else:
            competitors = ["amazon.in", "flipkart.com", "myntra.com", "snapdeal.com", "paytm.com"]

        # Filter out products from the same website domain
        filtered_normalized = []
        if input_domain:
            app.logger.info("FILTERING: input_domain=%s, total products=%d", input_domain, len(normalized))
            if len(normalized) == 0:
                app.logger.warning("No products to filter - search returned empty results")
            for idx, it in enumerate(normalized):
                product_url = it.get("url") or ""
                product_site = it.get("site") or ""
                if is_same_website(input_domain, product_url, product_site):
                    app.logger.info("FILTERED OUT[%d]: site=%s, url=%s", idx, product_site, product_url[:80])
                    continue
                filtered_normalized.append(it)
                app.logger.debug("KEPT[%d]: site=%s, url=%s", idx, product_site, product_url[:80])
            app.logger.info("FILTERING RESULT: kept %d out of %d products (input domain: %s)", 
                           len(filtered_normalized), len(normalized), input_domain)
            
            # If all products were filtered out, log a warning but still return empty list
            if len(filtered_normalized) == 0 and len(normalized) > 0:
                app.logger.warning("ALL products filtered out! This might indicate all search results were from the same domain.")
                app.logger.info("Sample filtered products: %s", 
                              [(it.get("site"), it.get("url")[:50]) for it in normalized[:3]])
        else:
            # If no input domain, return all normalized products
            filtered_normalized = normalized
            app.logger.warning("No input domain provided, returning all %d products", len(normalized))

        targeted_by_site = {}
        for comp in competitors:
            found = []
            comp_key = comp.split(".")[0]
            for it in filtered_normalized:
                site = (it.get("site") or it.get("url") or "").lower()
                if comp_key in site or comp in site:
                    found.append(it)
            if found:
                targeted_by_site[comp] = found

        # Include debug info in response if no products found
        response_data = {
            "related_products": filtered_normalized, 
            "targeted_by_site": targeted_by_site
        }
        
        # Add debug info if no products
        if len(filtered_normalized) == 0:
            response_data["debug"] = {
                "input_domain": input_domain,
                "total_before_filter": len(normalized),
                "total_after_filter": len(filtered_normalized),
                "search_returned_results": search_returned_results,
                "candidate_title": candidate_title[:100] if candidate_title else None,
                "website_url": website_url[:100] if website_url else None,
                "serpapi_configured": bool(SERPAPI_KEY),
                "message": "No products found. Check server logs for details.",
                "search_error": search_error_message
            }
            if not SERPAPI_KEY:
                response_data["debug"]["message"] = "SERPAPI_KEY not set. Using DDG fallback only."
            elif search_error_message:
                response_data["debug"]["message"] = f"Search error: {search_error_message}"
            elif not search_returned_results:
                response_data["debug"]["message"] = "Search APIs returned no results. Try a simpler product name or check API configuration."
            elif len(normalized) > 0:
                response_data["debug"]["message"] = f"All {len(normalized)} results were filtered out (same domain as input)."
            app.logger.warning("Returning empty related_products. Debug: %s", response_data["debug"])
        
        return jsonify(response_data)
    except Exception as e:
        app.logger.exception("related_products error: %s", e)
        return jsonify({"error": str(e)}), 500


# -----------------------
# Test endpoint to verify SerpAPI
# -----------------------
@app.route("/test_serpapi", methods=["GET"])
def test_serpapi():
    """Test endpoint to verify SerpAPI is working"""
    if not SERPAPI_KEY:
        return jsonify({"error": "SERPAPI_KEY not set", "status": "failed"}), 500
    
    try:
        test_query = "acer laptop"
        params = {"engine": "google_shopping", "q": test_query, "api_key": SERPAPI_KEY, "num": 3, "gl": "in"}
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=10)
        
        result = {
            "status_code": resp.status_code,
            "serpapi_key_set": bool(SERPAPI_KEY),
            "key_length": len(SERPAPI_KEY) if SERPAPI_KEY else 0,
            "test_query": test_query
        }
        
        if resp.ok:
            data = resp.json()
            if "error" in data:
                result["error"] = data.get("error")
                result["status"] = "error"
            else:
                shopping_results = data.get("shopping_results", [])
                result["shopping_results_count"] = len(shopping_results)
                result["status"] = "success" if shopping_results else "no_results"
                # Show sample with all available fields
                if shopping_results:
                    sample = shopping_results[0]
                    result["sample_result_keys"] = list(sample.keys())
                    result["sample_result"] = {k: str(v)[:100] for k, v in list(sample.items())[:10]}
                result["sample_results"] = [{"title": r.get("title", ""), "link": r.get("link", ""), "product_link": r.get("product_link", "")} for r in shopping_results[:2]]
        else:
            result["status"] = "http_error"
            result["response_text"] = resp.text[:200]
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "status": "exception"}), 500


# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0"
    print(f"Serving index from: {INDEX_PATH}")
    print(f"Server starting on http://{host}:{port}")
    print(f"Test SerpAPI at: http://{host}:{port}/test_serpapi")
    app.run(host=host, port=port, debug=True)
>>>>>>> e9a88ed6 (Initial clean commit)

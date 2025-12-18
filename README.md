<<<<<<< HEAD

# 🛡️ Fraud Sentinel
#### AI-Powered Online Product Fraud Detection System

Fraud Sentinel is a web-based application that helps users identify fraudulent or suspicious online product listings using machine learning and trust signal analysis.
It analyzes website and product details such as domain age, HTTPS security, price deviation, and product category, and predicts the fraud risk of a product listing in real time.


## 🚀 Features

- 🔍 Fraud Risk Prediction using Machine Learning (CatBoost)

- 🌐 Website Trust Analysis

- HTTPS / SSL verification

 - Domain age evaluation

- 💰 Price Deviation Detection to identify unrealistic offers

- 🔁 Related Product Comparison from trusted marketplaces

- 📊 Clear Risk Classification

  - Safe

   - Uncertain

  - Suspicious

- 🖥️ User-Friendly Web Interface

- ⚡ Fast real-time prediction

## 🧠 How It Works

- User enters product details (URL, name, category, price).

- Backend validates inputs and extracts trust indicators.

- Features are passed to a trained CatBoost ML model.

- Model predicts fraud probability.

- System fetches similar products for comparison.

- Results are displayed with clear risk indicators.


## ⚙️ Technology Stack

| Layer      | Technologies                     |
|------------|----------------------------------|
| Frontend   | HTML, CSS, JavaScript             |
| Backend    | Python, Flask                     |
| ML Model   | CatBoost                          |
| Scraping   | BeautifulSoup, Requests           |
| Tools      | Git, VS Code, Postman             |

## 🔐 API Keys & Configuration

This project uses external services and configuration values that must be set before running the application.

### Environment Variables
Create a `.env` file in the project root and add the following:

```env
SERPAPI_KEY=your_serpapi_key_here
FLASK_ENV=development
FLASK_DEBUG=True
```


## 📦 Installation & Setup
1️⃣ Clone the Repository
```
git clone https://github.com/your-username/fraud-sentinel.git
cd fraud-sentinel
```

2️⃣ Create Virtual Environment
```
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/Mac
```

3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

4️⃣ Run the Application
```
python app.py
```

5️⃣ Open in Browser
```
http://127.0.0.1:5000
```
## 📜 License

This project is developed for academic and educational purposes.
=======
# Fraud-Sentinel
>>>>>>> e9a88ed6 (Initial clean commit)

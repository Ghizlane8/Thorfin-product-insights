# 🚀 Thorfin Product Insights — AI-Powered Product Analytics Dashboard

**Thorfin Product Insights** is an interactive **Data & AI dashboard** built with **Streamlit**, designed to help product teams analyze customer reviews, product performance, and sentiment trends across **multiple languages**.

The application combines **data analytics**, **business rules**, and **state-of-the-art NLP (Transformer models)** to support **data-driven product decisions**.

---

## 🎯 Project Objectives

- Identify **best and worst performing products**
- Analyze a product’s performance (ratings, prices, reviews)
- Extract insights from **unstructured customer reviews**
- Perform **multilingual sentiment analysis** using AI
- Support decision-making with **recommendations & risk assessment**
- Analyze performance **by country / language**

---

## 🧠 AI & Machine Learning

### Sentiment Analysis Model

- **Model**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Architecture**: Transformer (BERT)
- **Framework**: Hugging Face Transformers
- **Languages supported**:  
  English, French, Spanish, German, Italian, Portuguese, Arabic, Chinese, and more
- **Output**: 1–5 star rating per review

### Business Interpretation

| Stars | Sentiment |
|------|----------|
| ⭐ 1–2 | Negative |
| ⭐ 3 | Neutral |
| ⭐ 4–5 | Positive |

### Why this model?

- Understands **context and negation**
- Handles **long and nuanced reviews**
- Native **multilingual support**
- Significantly more reliable than classical TF-IDF models on real-world text

⚠️ The model is **pre-trained** and used in **inference mode only** (no local retraining).

---

## ⚡ Performance & Scalability Strategy

Transformer models are computationally expensive.  
To keep the application **responsive and production-ready**, the following optimizations are implemented:

- Sentiment inference limited to a **representative sample (max 25 reviews per product)**
- Predictions **cached** using Streamlit caching
- No AI inference on static pages
- Lightweight similarity engine using **TF-IDF + cosine similarity**

> This reflects **real production constraints** and best practices in applied AI systems.

---

## 📊 Key Features

### 🔹 Global Overview
- Number of products & reviews
- Average rating
- % of high-rating reviews (≥ 4⭐)
- Global leaders & laggards

### 🔹 Product Analysis
- Average rating, price, number of reviews
- Rating distribution per product
- AI-based sentiment distribution
- Sample customer reviews
- Automated AI sentiment summary

### 🔹 Multilingual Sentiment Demo
- Free-text input
- Real-time sentiment prediction
- Star rating + confidence score
- Designed for **live demos**

### 🔹 Recommendations & Decisions
- Product verdicts:
  - ✅ Highly recommended
  - 🟡 Acceptable
  - ⚠️ At risk
- Rule-based logic combining ratings, volume, and AI sentiment
- Similar product recommendations (TF-IDF similarity)

### 🔹 Country & Language Analysis
- Review volume per country
- Best & worst products by country
- Sentiment distribution by country

---

## 🛠️ Tech Stack

- **Python 3.10**
- **Streamlit** — interactive dashboard
- **Pandas / NumPy** — data processing
- **Plotly** — interactive visualizations
- **Scikit-learn** — TF-IDF & similarity engine
- **Hugging Face Transformers** — multilingual sentiment analysis
- **PyTorch** — inference backend

---

## ▶️ How to Run the App

### 1️⃣ Create environment

```bash
conda create -n cpu python=3.10
conda activate cpu
```

2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

3️⃣ Run the application
```bash
streamlit run thorfin_app.py
```

## 📂 Project Structure
```text
thorfin-product-insights/
│
├── thorfin_app.py       
├── thorfin_reviews_electronics_home_appliance.csv
├── requirements.txt
└── README.md
```

## 🧪 Example Use Case

- A product manager wants to quickly identify products with high customer dissatisfaction across different countries and languages — without manually reading hundreds of reviews.
- Thorfin Product Insights provides:
- AI-powered sentiment breakdown
- Clear visual indicators
- Automated summaries
- Actionable recommendations

## 🎓 What This Project Demonstrates

- End-to-end Data + AI pipeline
- Practical application of Transformer models
- Performance-aware AI integration
- Strong focus on business value
- Clean, modular, and maintainable code


## 👤 Author

**Ghizlane Baali**
**baali.ghizlane2@gmail.com**

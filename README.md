# 🧠 Customer Intelligence AI

**Intelligent Customer Segmentation & Predictive Analytics Dashboard**
*BICS Global Internship · Data Analyst Intern — Data Science Domain*

---

## 📋 Overview

Customer Intelligence AI is an interactive analytics application that:
- Segments customers into meaningful groups using K-Means clustering
- Predicts customer spending scores using a Random Forest regression model
- Provides an AI-powered assistant that explains results in plain English
- Works entirely offline with a built-in synthetic dataset — no API key required

---

## 🚀 Quick Start

### 1. Create and activate a virtual environment (recommended)

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure OpenAI API key

```bash
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux
```

Then edit `.env` and replace `your_api_key_here` with your actual key.
The application works fully without an API key using the built-in assistant.

### 4. Launch the application

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📁 Project Structure

```
customer-intelligence-ai/
├── app.py                    # Main Streamlit application (all 7 pages)
├── requirements.txt          # Python dependencies
├── README.md
├── .env.example              # API key template
├── .gitignore
├── data/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── data_loader.py        # Sample data generation & CSV validation
    ├── analytics.py          # KPI calculations & dataset summaries
    ├── clustering.py         # K-Means segmentation & cluster labeling
    ├── prediction.py         # RandomForest regression model
    ├── ai_agent.py           # AI assistant (built-in + OpenAI)
    └── visualizations.py     # Plotly chart builders
```

---

## 🎯 Features

| Page | Features |
|---|---|
| Dashboard Overview | KPI cards, scatter plots, histograms, correlation heatmap, searchable table |
| Customer Explorer | Filter by segment, gender, age, income; single-customer deep-dive |
| Customer Segmentation | K-Means (K=2–8), silhouette score, cluster profiles, export CSV |
| Predictive Analytics | RandomForest, MAE, R², actual vs predicted chart, feature importance, prediction form |
| AI Analytics Assistant | 7 suggested questions, chat history, built-in + optional LLM mode |
| Dataset Management | CSV upload, column mapping, data quality report, reset to sample |
| About the Project | Tech stack, demo script, limitations |

---

## 🎬 5-Minute Demo Script

1. **Dashboard** → Point out KPI cards (total customers, avg income, spending score)
2. **Segmentation** → Set K=4, click Run Segmentation, explain cluster profiles
3. **Predictive Analytics** → Train model, show MAE/R², use prediction form
4. **AI Assistant** → Ask: *"Summarize the customer segments and recommend strategies for improving retention."*
5. **Dataset Management** → Show data quality report
6. **Export** → Download segmented_customers.csv

### Sample Questions for the AI Assistant
- "Summarize the customer dataset."
- "Explain the customer segments."
- "Which customer segment has the highest spending?"
- "What strategies could improve customer retention?"
- "Give me a summary for my internship presentation."

---

## 🛠️ Technology Stack

- **Python 3.11+** / Python 3.13 compatible
- **Streamlit** — web interface
- **Pandas + NumPy** — data processing
- **Scikit-learn** — K-Means clustering, RandomForest regression
- **Plotly** — interactive charts
- **OpenAI SDK** — optional LLM integration (gpt-4o-mini)
- **python-dotenv** — environment variable management

---

## ⚠️ Limitations

- Results are based on synthetic or uploaded data — for demonstration only.
- The built-in assistant uses keyword routing; configure an OpenAI key for richer answers.
- No authentication or multi-user support.
- Prediction requires a column named `SpendingScore` (or a mapped equivalent).

---

*Built for internship presentation — BICS Global, 2025*

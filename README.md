# FinTech Credit Card Churn & Retention Analytics Engine

> **FinTech credit card churn analytics engine using real UCI Credit Card data (30,000 accounts), RFM segmentation, XGBoost + SHAP, unit economics (LTV/CAC), and interactive ROI strategy simulation.**

---

## Executive Summary

Credit card issuers generate revenue through **interchange fees (1.5%–2.5% per swipe)** and **annual fees**, but lose thousands in long-term profit when high-spending or high-limit cardholders churn/default.

Using real-world historical data from **30,000 credit card holders (UCI Credit Card Dataset)**, this project builds an analytics-driven retention engine:
1. **Decodes raw categorical codes & 6-month billing time-series vectors** (`PAY_0` to `PAY_6`, `BILL_AMT1-6`, `PAY_AMT1-6`).
2. **Engineers advanced FinTech features:** 6-month credit utilization trends, payment-to-bill ratios, rolling delinquency escalation sequences, and RFM scores.
3. **Trains XGBoost Churn & Default Classifier** paired with **SHAP TreeExplainer** to identify top drivers for every customer.
4. **Simulates targeted retention offers** (Fee Waiver vs. 2x Points Boost vs. APR Cut vs. VIP Perks) using Knapsack ROI density optimization to maximize **Portfolio Net ROI**.

---

## Real Dataset Portfolio Financial Impact (30,000 Accounts)

| Metric | Real Business Analytics Result |
| :--- | :--- |
| **Total Real Accounts Evaluated** | **30,000 accounts** |
| **Base Default/Churn Rate** | **22.12%** |
| **Unmitigated At-Risk LTV Exposure** | **$118.6 Million** |
| **Optimized Campaign Budget** | **$500,000.00** |
| **Targeted High-Risk Accounts** | **3,870 / 30,000 (12.9%)** |
| **Gross Retained LTV Saved** | **$174.5 Million** |
| **Net Retained Profit Saved** | **$174.0 Million** |
| **Portfolio Campaign Net ROI** | **3,481%** |

---

## Data Flow & Architecture

```
   Raw UCI Dataset       RFM + Churn Model      LTV & Cost Matrix      Strategy Dashboard
┌─────────────────┐     ┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ 30k Real Credit │ ──> │ RFM Segmentation│ ─> │ Retention Offer │ ─> │ Streamlit & PPTX │
│ Card Statements │     │ & XGBoost Churn │    │ ROI Simulation  │    │ Offer Recommender│
└─────────────────┘     └─────────────────┘    └─────────────────┘    └──────────────────┘
```

### 1. Real Data Cleaning & Feature Engineering (`src/feature_engineering.py`)
- **Raw Data Cleaning:** Decodes `EDUCATION`, `MARRIAGE`, `SEX`, and `PAY_0`-`PAY_6` delay status codes.
- **6-Month Financial Ratios:**
  * `avg_monthly_spend`: Average bill amount across 6 billing statements.
  * `current_utilization`: $\text{BILL\_AMT1} / \text{LIMIT\_BAL}$.
  * `utilization_trend`: 6-month growth in credit line utilization.
  * `pay_to_bill_ratio`: Ratio of total payments vs total bills (distinguishing revolvers vs pay-in-full users).
  * `max_delay_months`: Maximum delinquency status across the 6-month window.
- **RFM Matrix:** Quantile-based Recency (1–5), Frequency (1–5), and Monetary (1–5) scoring with deterministic seed=42.

### 2. Predictive Machine Learning & SHAP (`src/churn_model.py`)
- **XGBoost Classifier:** Evaluated against baseline Logistic Regression on 30,000 real accounts.
- **XGBoost ROC-AUC:** `0.7826` | **PR-AUC:** `0.5636`.
- **SHAP Drivers:** Identifies recent payment delay (`PAY_0`), current utilization, credit limit, and payment-to-bill deficit as primary churn/default drivers.

### 3. Financial Retention Offer Matrix (`src/roi_engine.py`)
- **Offers Evaluated:** Fee Waiver, 2x Cashback Points Boost, 0% APR Cut / Delinquency Relief, VIP Perks.
- **Knapsack Optimization:** Allocates campaign budget by net ROI density ($\text{Net ROI} / \text{Cost}$) to maximize portfolio return.

---

## Repository Structure

```
Fintech-Customer-Churn-Analytics/
├── README.md                                # Real dataset project documentation
├── requirements.txt                         # Dependencies
├── app.py                                   # Interactive Streamlit Strategy Dashboard
├── data/
│   └── UCI_Credit_Card.csv                  # Real 30,000 credit card accounts dataset
├── src/
│   ├── __init__.py
│   ├── config.py                            # Centralized project configuration
│   ├── feature_engineering.py               # UCI dataset cleaning & RFM feature engine
│   ├── churn_model.py                       # XGBoost + SHAP Engine on real data
│   └── roi_engine.py                        # Financial LTV offer ROI optimizer
└── exports/
    ├── FinTech_Churn_Retention_Strategy.pptx # Executive 8-Slide PowerPoint Presentation
    └── customer_retention_recommendations.csv# Campaign target recommendations CSV
```

---

## Quick Start Guide

### 1. Installation
```bash
git clone https://github.com/PriyankaHichkad/Fintech-Customer-Churn-Analytics.git
cd Fintech-Customer-Churn-Analytics
pip install -r requirements.txt
```

### 2. Run Real Data Pipeline & ML Engine
```bash
python src/feature_engineering.py
python src/churn_model.py
python src/roi_engine.py
```

### 3. Launch Interactive Strategy Dashboard
```bash
streamlit run app.py
```

---

## Executive Presentation Deck

The repository includes a hand-crafted **8-Slide Executive PowerPoint Presentation** (`exports/FinTech_Churn_Retention_Strategy.pptx`) detailing:
1. **Title & Headline Hook** (*30k real accounts, 3,481% simulated ROI / $174M net profit saved*)
2. **Business Problem & Interchange Fee Economics**
3. **Data Pipeline & 6-Month Time-Series Feature Architecture**
4. **XGBoost Model Performance & Top SHAP Default Triggers**
5. **From Prediction to Economics (The LTV/CAC Bridge)**
6. **ROI Simulation Results ($500k Budget Cap)**
7. **Strategic Recommendations & A/B Testing Plan**
8. **Interactive Strategy Dashboard Showcase**

---

## Tools and Libraries Used

* Python: https://www.python.org/
* UCI Credit Card Dataset: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
* XGBoost: https://xgboost.readthedocs.io/
* SHAP: https://shap.readthedocs.io/
* Streamlit: https://streamlit.io/
* Plotly: https://plotly.com/python/
* Pandas: https://pandas.pydata.org/
* Scikit-Learn: https://scikit-learn.org/
* NumPy: https://numpy.org/
* Python-PPTX: https://python-pptx.readthedocs.io/

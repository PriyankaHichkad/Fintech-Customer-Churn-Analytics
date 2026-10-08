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
| **Optimized Campaign Budget Cap** | **$100,000.00** |
| **Targeted High-Risk Accounts** | **752 / 30,000 (2.5%)** |
| **Gross Retained LTV Saved** | **$692,133.00** |
| **Net Retained Profit Saved (Base)** | **$592,313.00** |
| **Net Retained Profit Saved (50% Sensitivity)** | **$246,246.00** |
| **Portfolio Campaign Net ROI (Base)** | **593.4%** |
| **Portfolio Campaign Net ROI (Conservative)** | **246.7%** |

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
- **Currency Standardization:** Converts raw UCI credit limits and bill amounts from NTD to USD ($1\text{ USD} = 30\text{ NTD}$).
- **6-Month Financial Ratios:**
  * `avg_monthly_spend`: Average bill amount across 6 billing statements.
  * `current_utilization`: $\text{BILL\_AMT1} / \text{LIMIT\_BAL}$.
  * `utilization_trend`: 6-month growth in credit line utilization.
  * `pay_to_bill_ratio`: Ratio of total payments vs total bills (distinguishing revolvers vs pay-in-full users).
  * `max_delay_months`: Maximum delinquency status across the 6-month window.
- **RFM Matrix:** Quantile-based Recency (1–5), Frequency (1–5), and Monetary (1–5) scoring with deterministic `seed=42`.

---

## Financial Unit Economics & Mathematical Formulations

To ensure rigorous financial modeling, the engine uses explicit unit economic equations across all 30,000 accounts:

### 1. Annual Net Margin Equation
$$\text{Annual Net Margin} = (\text{Interchange Fees} + \text{Annual Fee} + \text{Interest Income}) - (\text{Rewards Cost} + \text{Servicing Cost})$$

Where:
- $\text{Interchange Fees} = \text{Annual Spend} \times \text{Interchange Rate } (1.5\% \text{ to } 2.5\% \text{ by card tier})$
- $\text{Annual Fee} = \$0 \text{ (Standard)}, \$50 \text{ (Gold)}, \$95 \text{ (Platinum)}, \$250 \text{ (Black)}$
- $\text{Interest Income} = \text{LIMIT\_BAL\_USD} \times \text{Utilization} \times 18\% \times \mathbb{I}(\text{pay\_to\_bill\_ratio} < 0.90)$
- $\text{Rewards Cost} = \text{Annual Spend} \times 1.5\%$
- $\text{Servicing Cost} = \$50.00 + (\text{delinquency\_count} \times \$20.00)$

### 2. Baseline Customer Lifetime Value (LTV)
Customer horizon is conservatively capped at a **3.0-year ceiling** to avoid infinite horizon compounding:
$$\text{Effective LTV Horizon (Years)} = \min\left(3.0, \frac{1}{\text{Churn Prob} + \text{Discount Rate (10\%)}}\right)$$
$$\text{Baseline LTV} = \text{Annual Net Margin} \times \text{Effective LTV Horizon}$$

### 3. Retention Offer ROI & Knapsack Density Optimization
For each candidate retention offer $k \in \{\text{Fee Waiver}, \text{2x Points Boost}, \text{APR Cut}, \text{VIP Perks}\}$:
$$\text{Gross Retained LTV Saved}_{i,k} = \text{Annual Net Margin}_i \times \left( \text{Horizon}_{\text{New}} - \text{Horizon}_{\text{Baseline}} \right)$$
$$\text{Expected Net ROI}_{i,k} = \text{Gross Retained LTV Saved}_{i,k} - \text{Offer Cost}_k$$
$$\text{ROI Density Score}_{i,k} = \frac{\text{Expected Net ROI}_{i,k}}{\text{Offer Cost}_k}$$

The campaign budget ($B = \$100,000$) is allocated by selecting accounts in descending order of $\text{ROI Density Score}$ until total offer cost reaches $B$.

---

### 2. Predictive Machine Learning & SHAP (`src/churn_model.py`)
- **ECOA Fair-Lending Compliance:** Explicitly excludes `AGE`, `SEX`, `MARRIAGE`, and `EDUCATION` from model features.
- **XGBoost Classifier:** Evaluated against baseline Logistic Regression on test set.
- **Test Set Metrics:** XGBoost Test ROC-AUC: `0.7789` | Test PR-AUC: `0.5618` | Top 20% Decile Capture: `51.66%`.
- **SHAP Drivers:** Identifies recent payment delay (`PAY_0`), current utilization, credit limit, and payment-to-bill deficit as primary churn/default drivers.

### 3. Financial Retention Offer Matrix (`src/roi_engine.py`)
- **Offers Evaluated:** Fee Waiver, 2x Cashback Points Boost, APR Cut / Delinquency Relief, VIP Perks.
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
1. **Title & Headline Hook** (*30k real accounts, 593.4% simulated ROI / $592k net profit saved*)
2. **Business Problem & Interchange Fee Economics**
3. **Data Pipeline & 6-Month Time-Series Feature Architecture**
4. **XGBoost Model Performance & Top SHAP Default Triggers**
5. **From Prediction to Economics (The LTV/CAC Bridge)**
6. **ROI Simulation Results ($100k Budget Cap)**
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

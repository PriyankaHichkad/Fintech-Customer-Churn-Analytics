# 💳 FinTech Credit Card Churn & Retention Analytics Engine

> **FinTech credit card churn analytics engine using RFM segmentation, XGBoost + SHAP, unit economics (LTV/CAC), and interactive ROI strategy simulation.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%2B%20SHAP-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary

Credit card issuers make money through **interchange fees (1.5%–2.5% per transaction)** and **annual charges**, but lose thousands in long-term profit when high-spending cardholders churn silently to competitors.

Rather than offering generic blanket discounts (which waste retention budget on low-risk cardholders), this project builds an **analytics-driven retention engine**:
1. **Predicts high-value cardholder churn** before it happens using XGBoost.
2. **Explains root-cause churn triggers** for every single customer using **SHAP values**.
3. **Simulates targeted retention offers** (Fee Waiver vs. 2x Points Boost vs. APR Reduction vs. VIP Perks) using unit economics to maximize **Portfolio Net ROI**.

---

## 📈 Verified Portfolio Financial Impact (5,000 Accounts)

| Metric | Business Analytics Result |
| :--- | :--- |
| **Unmitigated At-Risk LTV Exposure** | **$368,866.38** |
| **Optimized Retention Campaign Budget** | **$138,893.36** |
| **Targeted High-Risk Accounts** | **787 / 5,000 (15.7%)** |
| **Gross Retained LTV Saved** | **$678,156.75** |
| **Net Retained Profit Saved** | **$539,263.39** |
| **Portfolio Campaign Net ROI** | **388.3%** |

---

## ⚙️ Project Architecture & Data Flow

```
  Customer Logs        RFM + Churn Model      LTV & Cost Matrix      Strategy Dashboard
┌──────────────┐      ┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Transactions │ ───> │ RFM Segmentation│ ─> │ Retention Offer │ ─> │ Streamlit & PPTX │
│ & Support    │      │ & XGBoost Churn │    │ ROI Simulation  │    │ Offer Recommender│
└──────────────┘      └─────────────────┘    └─────────────────┘    └──────────────────┘
```

### 1. Data Pipeline & Feature Engineering (`src/data_generator.py`, `src/feature_engineering.py`)
- **RFM Segmentation Matrix:** Quantile-based Recency (1–5), Frequency (1–5), and Monetary (1–5) scoring.
- **Customer Cohorts:** *Champions / Whales*, *Loyal High Spenders*, *At-Risk High Value*, *Dormant / Lost*, and *Need Attention*.
- **FinTech Risk Signals:** Credit limit utilization spikes (>75%), spend velocity drops ($3\text{m vs } 12\text{m ratio} < 0.60$), support friction scores, and annual fee friction flags.

### 2. Predictive Modeling & SHAP Diagnostics (`src/churn_model.py`)
- **Model Evaluation:** XGBoost Classifier evaluated against baseline Logistic Regression.
- **Top Decile Capture:** Captures **61.57%** of true churners within the top 20% highest risk deciles.
- **SHAP Explainability:** Pinpoints individual customer friction (e.g., fee dissatisfaction vs. support ticket friction vs. competitor balance transfer offers).

### 3. Financial LTV & Retention Offer Matrix (`src/roi_engine.py`)
- **Customer Lifetime Value (LTV):** $\text{LTV} = \text{Annual Net Margin} \times \frac{1}{\text{Churn Prob} + 0.10}$
- **4-Offer Strategy Catalog:**
  - **Offer 1: Annual Fee Waiver** (Cost: $95–$495; target: fee-dissatisfied high spenders).
  - **Offer 2: 2x Cashback / Points Boost** (Cost: 1.5% of spend; target: active spenders).
  - **Offer 3: 0% Balance Transfer / APR Discount** (Cost: $120; target: revolvers paying interest).
  - **Offer 4: VIP Perks & Concierge** (Cost: $150; target: Platinum & Black tier holders).
- **Optimization Formula:** Selects $\max \Delta \text{Net ROI} = (\text{Retained LTV}) - (\text{Baseline LTV}) - (\text{Offer Cost})$ under budget caps.

---

## 📂 Repository Structure

```
Fintech-Customer-Churn-Analytics/
├── README.md                                # Project documentation
├── requirements.txt                         # Dependencies
├── app.py                                   # Interactive Streamlit Strategy Dashboard
├── data/
│   ├── raw_credit_card_data.csv             # Synthetic customer transactions & support logs
│   ├── processed_rfm_data.csv               # RFM scores & behavioral features
│   ├── churn_predictions.csv                # XGBoost predicted probabilities & risk deciles
│   └── retention_roi_simulated.csv          # Retention offer simulation results
├── src/
│   ├── __init__.py
│   ├── data_generator.py                    # 5,000 credit card account generator
│   ├── feature_engineering.py               # RFM Segmentation & LTV calculator
│   ├── churn_model.py                       # XGBoost + SHAP Engine
│   └── roi_engine.py                        # Financial LTV offer ROI optimizer
├── scripts/
│   └── generate_deck.py                     # Programmatic PPTX slide deck generator
└── exports/
    ├── FinTech_Churn_Retention_Strategy.pptx # Executive 11-Slide PowerPoint Presentation
    └── customer_retention_recommendations.csv# Campaign target recommendations CSV
```

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
git clone https://github.com/PriyankaHichkad/Fintech-Customer-Churn-Analytics.git
cd Fintech-Customer-Churn-Analytics
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Model Training
```bash
python src/data_generator.py
python src/feature_engineering.py
python src/churn_model.py
python src/roi_engine.py
```

### 3. Generate Executive PowerPoint Presentation
```bash
python scripts/generate_deck.py
```
*Outputs slide deck to `exports/FinTech_Churn_Retention_Strategy.pptx`.*

### 4. Launch Interactive Strategy Dashboard
```bash
streamlit run app.py
```

---

## 📊 Executive Presentation Deck

The project includes an automatically generated **11-Slide PowerPoint Presentation** (`exports/FinTech_Churn_Retention_Strategy.pptx`) designed for FinTech leaders (VP of Product, Chief Risk Officer, Growth Leads):
1. **Title & Executive Summary**
2. **Business Problem & Interchange Fee Economics**
3. **Data Pipeline & RFM Framework**
4. **XGBoost Model Diagnostics & Decile Lift**
5. **SHAP Root-Cause Diagnostics**
6. **LTV & Unit Economics Equation**
7. **Retention Offer Catalog & Matrix**
8. **ROI Simulation Results & $539k Net Profit Saved**
9. **Interactive Dashboard Overview**
10. **A/B Testing & Operational Rollout Plan**
11. **Strategic Recommendations**

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).

import pandas as pd
import numpy as np
import os

try:
    from . import config
except ImportError:
    import config

def clean_and_engineer_uci_data(data_path: str = None) -> pd.DataFrame:
    """
    Cleans raw UCI Credit Card dataset (30,000 accounts), converts New Taiwan Dollars (NT$) to USD,
    engineers 6-month time-series financial metrics, RFM scores, and 3-year capped baseline LTV.
    
    ECOA Compliance Note:
    - Excludes protected demographic attributes (AGE, SEX, MARRIAGE, EDUCATION) from model features.
    - Currency conversion applied: 1 USD = 30 NTD.
    """
    np.random.seed(config.RANDOM_SEED)
    
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), '../data/UCI_Credit_Card.csv')
        
    df = pd.read_csv(data_path)
    
    # 1. Clean Categorical Descriptions
    edu_map = {1: 'Graduate School', 2: 'University', 3: 'High School', 4: 'Others', 5: 'Others', 6: 'Others', 0: 'Others'}
    df['education_clean'] = df['EDUCATION'].map(edu_map).fillna('Others')
    
    marr_map = {1: 'Married', 2: 'Single', 3: 'Others', 0: 'Others'}
    df['marriage_clean'] = df['MARRIAGE'].map(marr_map).fillna('Others')
    df['gender'] = df['SEX'].map({1: 'Male', 2: 'Female'}).fillna('Unknown')
    
    # Target: default.payment.next.month (Credit Default Risk Trigger)
    df['churn_label'] = df['default.payment.next.month']
    df['customer_id'] = [f"CC-{id_val}" for id_val in df['ID']]
    
    # 2. Currency Conversion (NT$ to USD)
    df['LIMIT_BAL_USD'] = np.round(df['LIMIT_BAL'] * config.NTD_TO_USD, 2)
    
    bill_cols_raw = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_amt_cols_raw = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    
    for col in bill_cols_raw:
        df[f"{col}_USD"] = np.round(df[col] * config.NTD_TO_USD, 2)
    for col in pay_amt_cols_raw:
        df[f"{col}_USD"] = np.round(df[col] * config.NTD_TO_USD, 2)
        
    bill_cols_usd = [f"{col}_USD" for col in bill_cols_raw]
    pay_amt_cols_usd = [f"{col}_USD" for col in pay_amt_cols_raw]
    
    # 3. Time-Series Bill & Payment Aggregations (USD)
    df['avg_monthly_spend'] = np.round(df[bill_cols_usd].clip(lower=0).mean(axis=1), 2)
    df['annual_spend'] = df['avg_monthly_spend'] * 12
    df['avg_monthly_payment'] = np.round(df[pay_amt_cols_usd].clip(lower=0).mean(axis=1), 2)
    
    # Utilization Ratios
    df['current_utilization'] = np.round(np.clip(df['BILL_AMT1_USD'] / (df['LIMIT_BAL_USD'] + 1e-5), 0.0, 2.0), 4)
    df['utilization_6m_avg'] = np.round(np.clip(df[bill_cols_usd].mean(axis=1) / (df['LIMIT_BAL_USD'] + 1e-5), 0.0, 2.0), 4)
    df['utilization_trend'] = np.round(df['current_utilization'] - df['utilization_6m_avg'], 4)
    
    # Pay-to-Bill Ratio
    total_bills = df[bill_cols_usd].clip(lower=0).sum(axis=1)
    total_pays = df[pay_amt_cols_usd].clip(lower=0).sum(axis=1)
    df['pay_to_bill_ratio'] = np.round(np.clip(total_pays / (total_bills + 1.0), 0.0, 2.0), 4)
    
    # Delinquency Metrics
    df['max_delay_months'] = df[pay_status_cols].max(axis=1)
    df['delinquency_count'] = (df[pay_status_cols] > 0).sum(axis=1)
    
    # Deterministic Recency and Frequency Proxies (Seed=42)
    df['days_since_last_txn'] = np.where(df['PAY_0'] <= 0, np.random.randint(1, 15, len(df)), df['PAY_0'] * 30)
    df['transaction_count_monthly'] = np.clip(np.round(df['avg_monthly_spend'] / 80 + 5).astype(int), 1, 120)
    
    # Card Tier Derivation based on USD Credit Limit
    def assign_card_tier(limit_usd):
        if limit_usd >= 10000:
            return 'Black'
        elif limit_usd >= 5000:
            return 'Platinum'
        elif limit_usd >= 2500:
            return 'Gold'
        else:
            return 'Standard'
            
    df['card_tier'] = df['LIMIT_BAL_USD'].apply(assign_card_tier)
    
    # 4. RFM Scoring Matrix
    df['R_score'] = pd.qcut(df['days_since_last_txn'].rank(method='first'), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['F_score'] = pd.qcut(df['transaction_count_monthly'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['M_score'] = pd.qcut(df['avg_monthly_spend'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    df['RFM_Score'] = df['R_score'] + df['F_score'] + df['M_score']
    
    def assign_segment(row):
        score = row['RFM_Score']
        r = row['R_score']
        m = row['M_score']
        if score >= 13:
            return "Champions / Whales"
        elif score >= 10 and r >= 3:
            return "Loyal High Spenders"
        elif r <= 2 and m >= 4:
            return "At-Risk High Value"
        elif r <= 2:
            return "Dormant / Lost"
        else:
            return "Need Attention"
            
    df['rfm_segment'] = df.apply(assign_segment, axis=1)
    
    # 5. Financial Unit Economics (USD)
    df['utilization_risk_flag'] = ((df['current_utilization'] > 0.75) | (df['utilization_trend'] > 0.15)).astype(int)
    df['support_friction_score'] = np.round(df['delinquency_count'] * 1.5 + np.maximum(0, df['max_delay_months']), 2)
    
    df['interchange_revenue_annual'] = np.round(df['annual_spend'] * df['card_tier'].map(config.INTERCHANGE_RATES), 2)
    df['annual_fee'] = df['card_tier'].map(config.ANNUAL_FEE_MAP)
    
    df['interest_paid_12m'] = np.round(df['LIMIT_BAL_USD'] * df['current_utilization'] * 0.18 * (df['pay_to_bill_ratio'] < 0.90).astype(int), 2)
    df['reward_cost_annual'] = np.round(df['annual_spend'] * 0.015, 2)
    df['servicing_cost_annual'] = np.round(50.0 + df['delinquency_count'] * 20.0, 2)
    
    df['annual_net_margin'] = (df['interchange_revenue_annual'] + df['annual_fee'] + df['interest_paid_12m']) - (df['reward_cost_annual'] + df['servicing_cost_annual'])
    
    # Baseline expected LTV (Capped at 3-Year Horizon)
    approx_churn = np.clip(0.10 + 0.15 * (df['max_delay_months'] > 0).astype(int) + 0.10 * (df['current_utilization'] > 0.80).astype(int), 0.05, 0.85)
    effective_years = np.minimum(config.LTV_MAX_HORIZON_YEARS, 1.0 / (approx_churn + config.DISCOUNT_RATE))
    df['baseline_ltv'] = np.round(df['annual_net_margin'] * effective_years, 2)
    
    return df

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), '../data/UCI_Credit_Card.csv')
    df_processed = clean_and_engineer_uci_data(data_path)
    out_path = os.path.join(os.path.dirname(__file__), '../data/processed_rfm_data.csv')
    df_processed.to_csv(out_path, index=False)
    print(f"[Feature Engineering] Successfully processed {len(df_processed):,} records into USD (Seed={config.RANDOM_SEED})")

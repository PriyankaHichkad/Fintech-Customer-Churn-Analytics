import pandas as pd
import numpy as np
import os

def compute_rfm_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes RFM Scores, Behavioral Flags, Financial Ratios, and Baseline LTV.
    """
    df = df.copy()
    
    # 1. RFM Score Calculation (1 to 5 scale using quantiles or fixed cutoffs)
    # Recency (Lower days = higher score 5)
    df['R_score'] = pd.qcut(df['days_since_last_txn'].rank(method='first'), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    
    # Frequency (Higher transactions = higher score 5)
    df['F_score'] = pd.qcut(df['transaction_count_monthly'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    # Monetary (Higher spend = higher score 5)
    df['M_score'] = pd.qcut(df['avg_monthly_spend'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    df['RFM_Score'] = df['R_score'] + df['F_score'] + df['M_score']
    
    # RFM Segment Categorization
    def assign_segment(row):
        score = row['RFM_Score']
        r = row['R_score']
        f = row['F_score']
        m = row['M_score']
        
        if score >= 13:
            return "Champions / Whales"
        elif score >= 10 and r >= 3:
            return "Loyal High Spenders"
        elif r >= 4 and f <= 2:
            return "New / Promising"
        elif r <= 2 and m >= 4:
            return "At-Risk High Value"
        elif r <= 2 and f <= 2:
            return "Dormant / Lost"
        else:
            return "Need Attention"
            
    df['rfm_segment'] = df.apply(assign_segment, axis=1)
    
    # 2. Advanced FinTech Behavioral Flags & Signals
    df['utilization_risk_flag'] = ((df['current_utilization'] > 0.75) | (df['utilization_3m_change'] > 0.15)).astype(int)
    df['inactivity_trend_flag'] = (df['spend_3m_vs_12m'] < 0.60).astype(int)
    df['support_friction_score'] = np.round(df['support_tickets_12m'] + 2.5 * df['complaint_tickets_12m'], 2)
    df['high_fee_friction_flag'] = ((df['annual_fee'] > 0) & (df['spend_3m_vs_12m'] < 0.70)).astype(int)
    
    # 3. Baseline Customer Lifetime Value (LTV)
    # Baseline expected remaining lifespan (years) based on churn probability
    discount_rate = 0.10
    # Expected tenure multiplier = 1 / (annual_churn_rate + discount_rate)
    annual_churn_est = np.clip(df['churn_prob_true'], 0.05, 0.90)
    df['expected_lifespan_years'] = np.round(1.0 / (annual_churn_est + discount_rate), 2)
    df['baseline_ltv'] = np.round(df['annual_net_margin'] * df['expected_lifespan_years'], 2)
    
    return df

if __name__ == '__main__':
    raw_path = os.path.join(os.path.dirname(__file__), '../data/raw_credit_card_data.csv')
    if not os.path.exists(raw_path):
        from data_generator import generate_credit_card_data
        df_raw = generate_credit_card_data()
        df_raw.to_csv(raw_path, index=False)
    else:
        df_raw = pd.read_csv(raw_path)
        
    df_processed = compute_rfm_and_features(df_raw)
    out_path = os.path.join(os.path.dirname(__file__), '../data/processed_rfm_data.csv')
    df_processed.to_csv(out_path, index=False)
    print(f"[Feature Engineering] Successfully processed {len(df_processed)} records into {out_path}")
    print(f"[Feature Engineering] RFM Segment Distribution:\n{df_processed['rfm_segment'].value_counts()}")

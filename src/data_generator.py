import pandas as pd
import numpy as np
import os

def generate_credit_card_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates realistic credit card customer dataset with transaction patterns,
    RFM metrics, support logs, unit economics, and churn labels.
    """
    np.random.seed(random_state)
    
    customer_ids = [f"CC-{10000 + i}" for i in range(n_samples)]
    
    # Card Tiers & Demographics
    card_tiers = np.random.choice(['Standard', 'Gold', 'Platinum', 'Black'], size=n_samples, p=[0.45, 0.30, 0.18, 0.07])
    ages = np.random.randint(21, 72, size=n_samples)
    tenure_months = np.random.randint(3, 120, size=n_samples)
    
    # Income & Credit Limits by Tier
    income_base = {'Standard': 45000, 'Gold': 85000, 'Platinum': 140000, 'Black': 280000}
    incomes = np.array([int(np.random.normal(income_base[t], income_base[t] * 0.25)) for t in card_tiers])
    incomes = np.clip(incomes, 25000, 500000)
    
    limit_ratios = {'Standard': 0.15, 'Gold': 0.25, 'Platinum': 0.35, 'Black': 0.45}
    credit_limits = np.array([int(inc * limit_ratios[t] * np.random.uniform(0.8, 1.2)) for inc, t in zip(incomes, card_tiers)])
    credit_limits = np.clip(credit_limits, 1000, 100000)
    
    # Spend & Utilization Patterns
    spend_ratios = {'Standard': 0.08, 'Gold': 0.12, 'Platinum': 0.18, 'Black': 0.25}
    avg_monthly_spend = np.array([int(inc * spend_ratios[t] / 12 * np.random.uniform(0.5, 1.5)) for inc, t in zip(incomes, card_tiers)])
    avg_monthly_spend = np.clip(avg_monthly_spend, 100, 25000)
    
    current_utilization = np.round(np.clip((avg_monthly_spend * np.random.uniform(0.8, 2.5)) / credit_limits, 0.02, 0.95), 4)
    utilization_3m_change = np.round(np.random.normal(-0.02, 0.15, size=n_samples), 4) # positive means utilization spiked
    
    spend_3m_vs_12m = np.round(np.random.normal(0.95, 0.30, size=n_samples), 4) # < 1 means spending is dropping (inactivity trend)
    spend_3m_vs_12m = np.clip(spend_3m_vs_12m, 0.05, 2.50)
    
    # RFM Raw Metrics
    days_since_last_txn = np.random.choice([
        np.random.randint(1, 15),
        np.random.randint(15, 45),
        np.random.randint(45, 120)
    ], size=n_samples, p=[0.70, 0.20, 0.10])
    
    transaction_count_monthly = np.clip(np.random.poisson(lam=avg_monthly_spend / 80 + 3), 1, 150)
    
    # Support Logs & Credit Risk
    late_payment_count_12m = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.75, 0.15, 0.06, 0.03, 0.01])
    support_tickets_12m = np.random.poisson(lam=1.5, size=n_samples)
    complaint_tickets_12m = np.clip(np.random.poisson(lam=0.4, size=n_samples), 0, 5)
    app_session_frequency_monthly = np.clip(np.random.poisson(lam=12, size=n_samples), 0, 60)
    competitor_offer_received = np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30])
    
    # Unit Economics (Annualized)
    annual_spend = avg_monthly_spend * 12
    interchange_rates = {'Standard': 0.0175, 'Gold': 0.020, 'Platinum': 0.0225, 'Black': 0.025}
    interchange_revenue_annual = np.round([annual_spend[i] * interchange_rates[card_tiers[i]] for i in range(n_samples)], 2)
    
    annual_fee_map = {'Standard': 0, 'Gold': 95, 'Platinum': 295, 'Black': 495}
    annual_fees = np.array([annual_fee_map[t] for t in card_tiers])
    
    interest_paid_12m = np.round(credit_limits * current_utilization * np.random.uniform(0.12, 0.24) * (late_payment_count_12m > 0).astype(int), 2)
    
    reward_cost_rates = {'Standard': 0.01, 'Gold': 0.015, 'Platinum': 0.022, 'Black': 0.03}
    reward_cost_annual = np.round([annual_spend[i] * reward_cost_rates[card_tiers[i]] for i in range(n_samples)], 2)
    
    servicing_cost_annual = np.round(np.random.uniform(35, 95, size=n_samples) + support_tickets_12m * 15, 2)
    
    # Net Financial Margin (Annual)
    annual_net_margin = (interchange_revenue_annual + annual_fees + interest_paid_12m) - (reward_cost_annual + servicing_cost_annual)
    
    # Churn Probability Ground Truth Formula (Domain Driven + Non-linear effects)
    logits = (
        - 2.80
        + 0.035 * days_since_last_txn
        - 1.60 * (spend_3m_vs_12m - 1.0)
        + 1.80 * (current_utilization > 0.80).astype(int)
        + 0.85 * complaint_tickets_12m
        + 0.40 * late_payment_count_12m
        + 0.75 * competitor_offer_received
        - 0.02 * (app_session_frequency_monthly)
        + 0.35 * (annual_fees > 0).astype(int) * (spend_3m_vs_12m < 0.7)
        - 0.01 * (tenure_months / 12)
    )
    
    churn_probs = 1 / (1 + np.exp(-logits))
    churn_label = (np.random.uniform(0, 1, size=n_samples) < churn_probs).astype(int)
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'card_tier': card_tiers,
        'tenure_months': tenure_months,
        'income_annual': incomes,
        'credit_limit': credit_limits,
        'current_utilization': current_utilization,
        'utilization_3m_change': utilization_3m_change,
        'avg_monthly_spend': avg_monthly_spend,
        'annual_spend': annual_spend,
        'spend_3m_vs_12m': spend_3m_vs_12m,
        'days_since_last_txn': days_since_last_txn,
        'transaction_count_monthly': transaction_count_monthly,
        'late_payment_count_12m': late_payment_count_12m,
        'interest_paid_12m': interest_paid_12m,
        'support_tickets_12m': support_tickets_12m,
        'complaint_tickets_12m': complaint_tickets_12m,
        'app_session_frequency_monthly': app_session_frequency_monthly,
        'competitor_offer_received': competitor_offer_received,
        'interchange_revenue_annual': interchange_revenue_annual,
        'annual_fee': annual_fees,
        'reward_cost_annual': reward_cost_annual,
        'servicing_cost_annual': servicing_cost_annual,
        'annual_net_margin': annual_net_margin,
        'churn_prob_true': np.round(churn_probs, 4),
        'churn_label': churn_label
    })
    
    return df

if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'raw_credit_card_data.csv')
    df = generate_credit_card_data(n_samples=5000)
    df.to_csv(file_path, index=False)
    print(f"[Data Generator] Successfully generated {len(df)} records at {file_path}")
    print(f"[Data Generator] Churn Rate: {df['churn_label'].mean():.2%}")

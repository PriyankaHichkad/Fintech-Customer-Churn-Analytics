import pandas as pd
import numpy as np
import os

def simulate_retention_offers(df: pd.DataFrame, budget_cap: float = 250000.0) -> pd.DataFrame:
    """
    Simulates ROI for 4 retention offer strategies per customer, selecting optimal
    offer to maximize portfolio Net ROI under budget constraints.
    """
    df = df.copy()
    discount_rate = 0.10
    
    # 1. Offer Definitions & Costs
    df['cost_fee_waiver'] = df['annual_fee']
    df['cost_points_boost'] = np.round(df['avg_monthly_spend'] * 12 * 0.015, 2)
    df['cost_apr_cut'] = 120.0
    df['cost_vip_perks'] = 150.0
    
    p_baseline = np.clip(df['predicted_churn_prob'], 0.01, 0.95)
    
    # 2. Probability Reductions (Effectiveness per offer)
    # Fee Waiver: highly effective for fee-paying cards with spend drop
    eff_fee_waiver = np.where((df['annual_fee'] > 0), np.minimum(0.45, p_baseline * 0.65), 0.0)
    
    # Points Boost: effective for high monthly spenders
    eff_points_boost = np.where(df['avg_monthly_spend'] >= 500, np.minimum(0.38, p_baseline * 0.52), 0.0)
    
    # APR Cut: effective for revolvers paying interest
    eff_apr_cut = np.where((df['interest_paid_12m'] > 100) | (df['current_utilization'] > 0.50), np.minimum(0.42, p_baseline * 0.58), 0.0)
    
    # VIP Perks: effective for Platinum and Black tier cards
    eff_vip_perks = np.where(df['card_tier'].isin(['Platinum', 'Black']), np.minimum(0.35, p_baseline * 0.48), 0.0)
    
    # 3. Calculate LTV gain for each offer
    def calc_ltv(margin, churn_p):
        return margin * (1.0 / (churn_p + discount_rate))
        
    baseline_ltv = calc_ltv(df['annual_net_margin'], p_baseline)
    df['baseline_ltv'] = np.round(baseline_ltv, 2)
    
    # Net ROI per offer = (New LTV - Baseline LTV) - Cost of Offer
    offers = ['fee_waiver', 'points_boost', 'apr_cut', 'vip_perks']
    eff_dict = {
        'fee_waiver': eff_fee_waiver,
        'points_boost': eff_points_boost,
        'apr_cut': eff_apr_cut,
        'vip_perks': eff_vip_perks
    }
    
    for offer in offers:
        p_new = np.clip(p_baseline - eff_dict[offer], 0.01, 0.95)
        ltv_new = calc_ltv(df['annual_net_margin'], p_new)
        delta_ltv = ltv_new - baseline_ltv
        cost = df[f'cost_{offer}']
        net_value = delta_ltv - cost
        
        df[f'delta_ltv_{offer}'] = np.round(delta_ltv, 2)
        df[f'net_roi_{offer}'] = np.round(net_value, 2)
        
    # 4. Select Optimal Offer per Customer (Greedy Maximum Net ROI)
    def pick_best_offer(row):
        # Only consider offers if churn prob >= 0.20 (target high/medium risk)
        if row['predicted_churn_prob'] < 0.20:
            return 'No Offer (Low Risk)', 0.0, 0.0, 0.0
            
        best_off = 'No Offer (Low Risk)'
        best_net = 0.0
        best_cost = 0.0
        best_delta = 0.0
        
        for offer in offers:
            net_val = row[f'net_roi_{offer}']
            if net_val > best_net:
                best_net = net_val
                best_off = offer.replace('_', ' ').title()
                best_cost = row[f'cost_{offer}']
                best_delta = row[f'delta_ltv_{offer}']
                
        return best_off, best_net, best_cost, best_delta
        
    res = df.apply(pick_best_offer, axis=1)
    df['recommended_offer'] = [r[0] for r in res]
    df['expected_net_roi'] = [r[1] for r in res]
    df['offer_cost'] = [r[2] for r in res]
    df['retained_ltv_gain'] = [r[3] for r in res]
    
    # 5. Apply Budget Constraint Prioritization (Sort by Net ROI / Cost efficiency)
    eligible = df[df['expected_net_roi'] > 0].copy()
    eligible = eligible.sort_values(by='expected_net_roi', ascending=False)
    
    eligible['cum_cost'] = eligible['offer_cost'].cumsum()
    eligible['within_budget'] = eligible['cum_cost'] <= budget_cap
    
    df['campaign_target'] = False
    df.loc[eligible[eligible['within_budget']].index, 'campaign_target'] = True
    
    # Update recommended offer to 'No Offer (Budget Exceeded)' for those cut by budget
    budget_exceeded_idx = eligible[~eligible['within_budget']].index
    df.loc[budget_exceeded_idx, 'recommended_offer'] = 'No Offer (Budget Exceeded)'
    
    print("[ROI Simulation Engine] Simulation Complete!")
    print(f"  Total Portfolio At-Risk LTV: ${df[df['predicted_churn_prob'] >= 0.30]['baseline_ltv'].sum():,.2f}")
    print(f"  Campaign Targeted Customers: {df['campaign_target'].sum():,} / {len(df):,}")
    print(f"  Total Campaign Cost:         ${df[df['campaign_target']]['offer_cost'].sum():,.2f}")
    print(f"  Total Gross LTV Saved:       ${df[df['campaign_target']]['retained_ltv_gain'].sum():,.2f}")
    print(f"  Total Net Profit Saved:      ${df[df['campaign_target']]['expected_net_roi'].sum():,.2f}")
    
    ret_roi = (df[df['campaign_target']]['expected_net_roi'].sum() / (df[df['campaign_target']]['offer_cost'].sum() + 1e-5)) * 100
    print(f"  Portfolio Campaign ROI:      {ret_roi:.1f}%")
    
    return df

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), '../data/churn_predictions.csv')
    if not os.path.exists(data_path):
        from churn_model import train_and_evaluate_churn_models
        _, _, df_pred, _ = train_and_evaluate_churn_models()
    else:
        df_pred = pd.read_csv(data_path)
        
    df_sim = simulate_retention_offers(df_pred, budget_cap=250000.0)
    out_path = os.path.join(os.path.dirname(__file__), '../data/retention_roi_simulated.csv')
    df_sim.to_csv(out_path, index=False)
    
    export_path = os.path.join(os.path.dirname(__file__), '../exports/customer_retention_recommendations.csv')
    df_sim[['customer_id', 'card_tier', 'avg_monthly_spend', 'predicted_churn_prob', 'rfm_segment', 'baseline_ltv', 'recommended_offer', 'offer_cost', 'expected_net_roi']].to_csv(export_path, index=False)
    print(f"[ROI Simulation Engine] Exported recommendations to {export_path}")

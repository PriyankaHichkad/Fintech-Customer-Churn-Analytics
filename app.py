import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle

# Page Configuration
st.set_page_config(
    page_title="FinTech Credit Card Churn & Retention Analytics Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Executive UI CSS Styling
st.markdown("""
<style>
    /* Global Page Styling */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 18px 20px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
    }
    div[data-testid="stMetricLabel"] > div {
        color: #64748B !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] > div {
        color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 1.75rem !important;
    }
    div[data-testid="stMetricDelta"] > div {
        color: #059669 !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
    }
    
    /* Takeaway Callout Boxes */
    .takeaway-box {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-top: 12px;
        margin-bottom: 20px;
    }
    .takeaway-box strong {
        color: #1E40AF;
    }
    .takeaway-box p {
        color: #1E3A8A;
        margin: 4px 0 0 0;
        font-size: 0.92rem;
    }
    
    /* Offer Badges */
    .badge-offer {
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    sim_path = os.path.join(base_dir, 'data/retention_roi_simulated.csv')
    
    if not os.path.exists(sim_path):
        from src.feature_engineering import clean_and_engineer_uci_data
        from src.churn_model import train_and_evaluate_churn_models
        from src.roi_engine import simulate_retention_offers
        clean_and_engineer_uci_data()
        _, _, df_pred, _ = train_and_evaluate_churn_models()
        df = simulate_retention_offers(df_pred)
    else:
        df = pd.read_csv(sim_path)
        
    metrics_path = os.path.join(base_dir, 'data/model_metrics.pkl')
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
        
    shap_path = os.path.join(base_dir, 'data/shap_data.pkl')
    with open(shap_path, 'rb') as f:
        shap_data = pickle.load(f)
        
    return df, metrics, shap_data

df, metrics, shap_data = load_data()

# Clean Native Header
st.title("FinTech Credit Card Churn & Retention Analytics Engine")
st.caption("Executive Strategy & Decision-Support Dashboard | Portfolio Analysis of 30,000 Real UCI Accounts")
st.divider()

# Sidebar Navigation & Filters
st.sidebar.header("Portfolio Filter Controls")
selected_tiers = st.sidebar.multiselect("Card Tier (Credit Limit)", options=df['card_tier'].unique(), default=df['card_tier'].unique())
selected_segments = st.sidebar.multiselect("RFM Segment", options=df['rfm_segment'].unique(), default=df['rfm_segment'].unique())
min_risk, max_risk = st.sidebar.slider("Predicted Risk Cutoff Range", 0.0, 1.0, (0.0, 1.0), 0.05)

filtered_df = df[
    (df['card_tier'].isin(selected_tiers)) &
    (df['rfm_segment'].isin(selected_segments)) &
    (df['predicted_churn_prob'] >= min_risk) &
    (df['predicted_churn_prob'] <= max_risk)
]

# Dashboard Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Portfolio Summary",
    "Churn & Default Drivers (SHAP Insights)",
    "Customer Risk & Offer Lookup",
    "Campaign Budget & ROI Simulator"
])

# TAB 1: EXECUTIVE PORTFOLIO SUMMARY
with tab1:
    st.markdown("### Executive Portfolio Health")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_acc = len(filtered_df)
    avg_churn = filtered_df['predicted_churn_prob'].mean()
    high_risk_cnt = (filtered_df['predicted_churn_prob'] >= 0.30).sum()
    at_risk_ltv = filtered_df[filtered_df['predicted_churn_prob'] >= 0.30]['baseline_ltv'].sum()
    total_net_roi = filtered_df[filtered_df['campaign_target']]['expected_net_roi'].sum()
    
    col1.metric("Total Accounts", f"{total_acc:,}")
    col2.metric("Avg Portfolio Risk", f"{avg_churn:.1%}")
    col3.metric("High-Risk Accounts", f"{high_risk_cnt:,}")
    col4.metric("At-Risk LTV Exposure", f"${at_risk_ltv/1e6:.2f}M" if at_risk_ltv >= 1e6 else f"${at_risk_ltv:,.0f}")
    col5.metric("Simulated Net Saved ROI", f"${total_net_roi/1e6:.2f}M" if total_net_roi >= 1e6 else f"${total_net_roi:,.0f}", delta=f"{total_net_roi/(at_risk_ltv+1e-5):.1%} of exposure")
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Default Risk Distribution by Card Tier")
        fig_tier = px.box(
            filtered_df, x='card_tier', y='predicted_churn_prob', color='card_tier',
            labels={'predicted_churn_prob': 'Risk Probability', 'card_tier': 'Card Tier'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_tier.update_layout(template="plotly_white", height=380, showlegend=False)
        st.plotly_chart(fig_tier, use_container_width=True)
        
        st.markdown("""
        <div class="takeaway-box">
            <strong>Stakeholder Takeaway:</strong> Standard and Gold cardholders exhibit higher median default risk variance. Targeting high-limit Platinum and Black cardholders yields significantly higher LTV margin recovery per targeted dollar.
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("#### RFM Customer Segment Distribution")
        rfm_counts = filtered_df['rfm_segment'].value_counts().reset_index()
        rfm_counts.columns = ['rfm_segment', 'count']
        fig_rfm = px.pie(
            rfm_counts, names='rfm_segment', values='count', hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_rfm.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig_rfm, use_container_width=True)
        
        st.markdown("""
        <div class="takeaway-box">
            <strong>Stakeholder Takeaway:</strong> 'Loyal High Spenders' and 'Champions' represent top-margin segments. Prioritizing retention offers for 'At-Risk High Value' accounts prevents catastrophic revenue drop-offs.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("#### Credit Line Utilization vs. Repayment Delinquency")
    fig_scat = px.scatter(
        filtered_df, x='current_utilization', y='max_delay_months',
        color='predicted_churn_prob', size='LIMIT_BAL',
        hover_data=['customer_id', 'card_tier', 'baseline_ltv'],
        labels={'current_utilization': 'Credit Utilization Ratio (0.0 to 1.0+)', 'max_delay_months': 'Max Delay Months (PAY_0 to PAY_6)'},
        color_continuous_scale="Viridis"
    )
    fig_scat.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig_scat, use_container_width=True)

# TAB 2: CHURN & DEFAULT DRIVERS (SHAP INSIGHTS)
with tab2:
    st.markdown("### Machine Learning Model Diagnostics & Risk Triggers")
    
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("XGBoost Model ROC-AUC", f"{metrics['xgb_auc']:.4f}")
    mc2.metric("XGBoost PR-AUC", f"{metrics['xgb_pr_auc']:.4f}")
    mc3.metric("Baseline Logistic Reg AUC", f"{metrics['lr_auc']:.4f}")
    mc4.metric("Top 20% Decile Capture", f"{metrics['top_20_capture_rate']:.1%}")
    
    st.markdown("---")
    shap_title = "Top Empirical Drivers of Credit Risk (SHAP Feature Attribution)" if metrics.get('is_real_shap', False) else "Top Empirical Drivers of Credit Risk (XGBoost Feature Importance Gain)"
    st.markdown(f"#### {shap_title}")
    
    shap_vals = shap_data['shap_values']
    feat_names = shap_data['feature_names']
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    
    shap_df = pd.DataFrame({'feature': feat_names, 'importance': mean_abs_shap})
    shap_df = shap_df.sort_values(by='importance', ascending=True).tail(10)
    
    fig_shap = px.bar(
        shap_df, x='importance', y='feature', orientation='h',
        labels={'importance': 'Mean Feature Attribution Impact Score', 'feature': 'Feature Name'},
        color='importance', color_continuous_scale='Blues'
    )
    fig_shap.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig_shap, use_container_width=True)
    
    st.markdown("""
    <div class="takeaway-box">
        <strong>Stakeholder Takeaway:</strong> Recent Payment Delay (PAY_0), 6-Month Utilization, Credit Limit, and Payment-to-Bill Deficits are the primary signals driving account risk. Tailored offers addressing interest and fee friction directly lower these risk drivers.
    </div>
    """, unsafe_allow_html=True)

# TAB 3: CUSTOMER RISK & OFFER LOOKUP
with tab3:
    st.markdown("### Individual Account Lookup & Recommendation Engine")
    
    cust_id = st.selectbox("Select Customer ID", options=filtered_df['customer_id'].unique())
    cust_data = filtered_df[filtered_df['customer_id'] == cust_id].iloc[0]
    
    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.metric("Card Tier", cust_data['card_tier'])
    ic2.metric("Predicted Churn Risk", f"{cust_data['predicted_churn_prob']:.1%}")
    ic3.metric("Baseline LTV", f"${cust_data['baseline_ltv']:,.2f}")
    ic4.metric("Recommended Retention Offer", cust_data['recommended_offer'])
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Account Financial & Behavioral Metrics")
        st.write(f"• **Credit Limit:** ${cust_data['LIMIT_BAL']:,}")
        st.write(f"• **Current Utilization Ratio:** {cust_data['current_utilization']:.1%}")
        st.write(f"• **Avg Monthly Bill / Payment:** ${cust_data['avg_monthly_spend']:,} bill | ${cust_data['avg_monthly_payment']:,} pay")
        st.write(f"• **Pay-to-Bill Ratio:** {cust_data['pay_to_bill_ratio']:.2f}x")
        st.write(f"• **Recent Repayment Status (PAY_0):** {cust_data['PAY_0']} month(s) late")
        st.write(f"• **Max Delay Months (6-Mon):** {cust_data['max_delay_months']} month(s)")
        
    with col_b:
        st.markdown("#### Retention Offer Economics")
        st.write(f"• **Recommended Strategy:** <span class='badge-offer'>{cust_data['recommended_offer']}</span>", unsafe_allow_html=True)
        st.write(f"• **Estimated Offer Cost:** ${cust_data['offer_cost']:,.2f}")
        st.write(f"• **Gross Retained LTV Saved:** ${cust_data['retained_ltv_gain']:,.2f}")
        st.write(f"• **Expected Net ROI Value:** ${cust_data['expected_net_roi']:,.2f}")
        st.write(f"• **Knapsack ROI Density Score:** {cust_data['roi_density']:.2f}x")
        
        if cust_data['expected_net_roi'] > 0:
            st.success("Positive ROI Recommendation: Offer generates net profit above offer cost.")
        else:
            st.warning("Low/Negative ROI: Retention offer cost exceeds expected LTV recovery.")

# TAB 4: CAMPAIGN BUDGET & ROI SIMULATOR
with tab4:
    st.markdown("### Interactive Campaign Budget & ROI Simulator")
    st.markdown("Adjust campaign retention budget and risk targeting thresholds to simulate portfolio net ROI under Knapsack density optimization.")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        campaign_budget = st.slider("Campaign Retention Budget ($)", 10000, 500000, 100000, 10000)
    with sim_col2:
        risk_cutoff = st.slider("Target Minimum Risk Probability Cutoff", 0.10, 0.50, 0.20, 0.05)
        
    eligible_sim = df[(df['predicted_churn_prob'] >= risk_cutoff) & (df['expected_net_roi'] > 0)].copy()
    eligible_sim = eligible_sim.sort_values(by='roi_density', ascending=False)
    eligible_sim['cum_cost'] = eligible_sim['offer_cost'].cumsum()
    budget_sim = eligible_sim[eligible_sim['cum_cost'] <= campaign_budget]
    
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Targeted Accounts", f"{len(budget_sim):,}")
    sc2.metric("Total Budget Spent", f"${budget_sim['offer_cost'].sum():,.0f}")
    sc3.metric("Gross LTV Saved", f"${budget_sim['retained_ltv_gain'].sum():,.0f}")
    sc4.metric("Net Profit Saved", f"${budget_sim['expected_net_roi'].sum():,.0f}")
    
    roi_pct = (budget_sim['expected_net_roi'].sum() / (budget_sim['offer_cost'].sum() + 1e-5)) * 100
    sc5.metric("Net Campaign ROI", f"{roi_pct:.1f}%")
    
    st.markdown("---")
    st.markdown("#### Targeted Account Recommendations List")
    st.dataframe(
        budget_sim[['customer_id', 'card_tier', 'LIMIT_BAL', 'predicted_churn_prob', 'avg_monthly_spend', 'recommended_offer', 'offer_cost', 'expected_net_roi', 'roi_density']],
        use_container_width=True
    )
    
    csv_data = budget_sim[['customer_id', 'card_tier', 'LIMIT_BAL', 'predicted_churn_prob', 'avg_monthly_spend', 'recommended_offer', 'offer_cost', 'expected_net_roi', 'roi_density']].to_csv(index=False).encode('utf-8')
    st.download_button("Download Real Account Recommendations CSV", data=csv_data, file_name="uci_retention_campaign_targets.csv", mime="text/csv")

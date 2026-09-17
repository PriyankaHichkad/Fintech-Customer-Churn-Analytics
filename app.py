import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle

# Page Config
st.set_page_config(
    page_title="FinTech Credit Card Churn & Retention ROI Engine (Real UCI Data)",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0F172A; }
    .stMetric { background-color: #1E293B; border-radius: 10px; padding: 15px; border: 1px solid #334155; }
    .stCard { background-color: #1E293B; border-radius: 10px; padding: 20px; border: 1px solid #334155; margin-bottom: 15px; }
    h1, h2, h3 { color: #F8FAFC; }
    .highlight { color: #0EA5E9; font-weight: bold; }
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

# Header Banner
st.title("💳 FinTech Credit Card Churn & Retention Analytics Engine")
st.markdown("**Real UCI Dataset Strategy Dashboard (30,000 Accounts)** | Target Roles: *Data Analyst • Business Analyst • FinTech Product Manager*")
st.markdown("---")

# Sidebar Filters
st.sidebar.header("🔍 Real Account Filters")
selected_tiers = st.sidebar.multiselect("Card Tier (by Limit)", options=df['card_tier'].unique(), default=df['card_tier'].unique())
selected_segments = st.sidebar.multiselect("RFM Segment", options=df['rfm_segment'].unique(), default=df['rfm_segment'].unique())
selected_education = st.sidebar.multiselect("Education Tier", options=df['education_clean'].unique(), default=df['education_clean'].unique())
min_risk, max_risk = st.sidebar.slider("Predicted Default/Churn Risk Range", 0.0, 1.0, (0.0, 1.0), 0.05)

filtered_df = df[
    (df['card_tier'].isin(selected_tiers)) &
    (df['rfm_segment'].isin(selected_segments)) &
    (df['education_clean'].isin(selected_education)) &
    (df['predicted_churn_prob'] >= min_risk) &
    (df['predicted_churn_prob'] <= max_risk)
]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Portfolio Overview",
    "🤖 Machine Learning & SHAP Diagnostics",
    "👤 Real Account Deep-Dive",
    "💰 What-If Campaign ROI Simulator"
])

# TAB 1: EXECUTIVE PORTFOLIO OVERVIEW
with tab1:
    st.subheader("Key Portfolio Health Metrics (Real UCI Credit Card Data)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_acc = len(filtered_df)
    avg_churn = filtered_df['predicted_churn_prob'].mean()
    high_risk_cnt = (filtered_df['predicted_churn_prob'] >= 0.30).sum()
    at_risk_ltv = filtered_df[filtered_df['predicted_churn_prob'] >= 0.30]['baseline_ltv'].sum()
    total_net_roi = filtered_df['expected_net_roi'].sum()
    
    col1.metric("Filtered Accounts", f"{total_acc:,}")
    col2.metric("Avg Default/Churn Risk", f"{avg_churn:.1%}")
    col3.metric("High-Risk Accounts (≥30%)", f"{high_risk_cnt:,}")
    col4.metric("At-Risk LTV Exposure", f"${at_risk_ltv:,.0f}")
    col5.metric("Simulated Net Saved ROI", f"${total_net_roi:,.0f}", delta=f"{total_net_roi/(at_risk_ltv+1e-5):.1%} of exposure")
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Predicted Risk by Card Tier (Credit Limit Cohorts)")
        fig_tier = px.box(
            filtered_df, x='card_tier', y='predicted_churn_prob', color='card_tier',
            title="Predicted Default/Churn Probability by Card Tier",
            labels={'predicted_churn_prob': 'Churn Probability', 'card_tier': 'Card Tier'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_tier.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_tier, use_container_width=True)
        
    with c2:
        st.subheader("RFM Customer Segment Breakdown")
        rfm_counts = filtered_df['rfm_segment'].value_counts().reset_index()
        rfm_counts.columns = ['rfm_segment', 'count']
        fig_rfm = px.pie(
            rfm_counts, names='rfm_segment', values='count', hole=0.4,
            title="30,000 UCI Customer Distribution across RFM Segments",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_rfm.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_rfm, use_container_width=True)
        
    st.subheader("Utilization Ratio vs. Maximum Delay Months by Risk")
    fig_scat = px.scatter(
        filtered_df, x='current_utilization', y='max_delay_months',
        color='predicted_churn_prob', size='LIMIT_BAL',
        hover_data=['customer_id', 'card_tier', 'education_clean', 'baseline_ltv'],
        title="Credit Utilization vs. Maximum Repayment Delay Status (PAY_0 to PAY_6)",
        color_continuous_scale="Plasma"
    )
    fig_scat.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_scat, use_container_width=True)

# TAB 2: ML & SHAP DIAGNOSTICS
with tab2:
    st.subheader("Model Performance Evaluation (Real 30,000 Accounts)")
    
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("XGBoost ROC-AUC", f"{metrics['xgb_auc']:.4f}")
    mc2.metric("XGBoost PR-AUC", f"{metrics['xgb_pr_auc']:.4f}")
    mc3.metric("Baseline Logistic Reg AUC", f"{metrics['lr_auc']:.4f}")
    mc4.metric("Top 20% Decile Capture Rate", f"{metrics['top_20_capture_rate']:.1%}")
    
    st.markdown("---")
    st.subheader("Global Feature Importance & SHAP Drivers")
    
    shap_vals = shap_data['shap_values']
    feat_names = shap_data['feature_names']
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    
    shap_df = pd.DataFrame({'feature': feat_names, 'importance': mean_abs_shap})
    shap_df = shap_df.sort_values(by='importance', ascending=True).tail(12)
    
    fig_shap = px.bar(
        shap_df, x='importance', y='feature', orientation='h',
        title="Top 12 Most Influential Real Features Driving Default/Churn (Mean |SHAP Value|)",
        labels={'importance': 'Mean |SHAP Value| (Impact on Log-Odds)', 'feature': 'Feature Name'},
        color='importance', color_continuous_scale='Tealgrn'
    )
    fig_shap.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_shap, use_container_width=True)
    
    st.info("💡 **Real Insight:** `PAY_0` (Recent Repayment Delay), `PAY_2`, `current_utilization`, `LIMIT_BAL`, and 6-month Payment-to-Bill Ratios are the strongest empirical drivers of credit default/churn in the UCI dataset.")

# TAB 3: REAL CUSTOMER DEEP-DIVE
with tab3:
    st.subheader("Individual Real Account Lookup")
    
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
        st.markdown("### 📋 Account Demographics & Real Financial Metrics")
        st.write(f"• **Age / Gender:** {cust_data['AGE']} yrs | {cust_data['gender']}")
        st.write(f"• **Education / Marital Status:** {cust_data['education_clean']} | {cust_data['marriage_clean']}")
        st.write(f"• **Credit Limit:** ${cust_data['LIMIT_BAL']:,}")
        st.write(f"• **Current Utilization Ratio:** {cust_data['current_utilization']:.1%}")
        st.write(f"• **Avg Monthly Bill / Payment:** ${cust_data['avg_monthly_spend']:,} bill | ${cust_data['avg_monthly_payment']:,} pay")
        st.write(f"• **Pay-to-Bill Ratio:** {cust_data['pay_to_bill_ratio']:.2f}x")
        st.write(f"• **Recent Delay Status (PAY_0):** {cust_data['PAY_0']} month(s) late")
        st.write(f"• **Max Repayment Delay (6-Mon):** {cust_data['max_delay_months']} month(s)")
        
    with col_b:
        st.markdown("### 🎯 Retention Offer Economics")
        st.write(f"• **Offer Name:** <span class='highlight'>{cust_data['recommended_offer']}</span>", unsafe_allow_html=True)
        st.write(f"• **Estimated Cost of Offer:** ${cust_data['offer_cost']:,.2f}")
        st.write(f"• **Gross Retained LTV Gain:** ${cust_data['retained_ltv_gain']:,.2f}")
        st.write(f"• **Expected Net ROI Value:** ${cust_data['expected_net_roi']:,.2f}")
        
        if cust_data['expected_net_roi'] > 0:
            st.success("✅ **Positive ROI Recommendation:** Offer generates net profit above cost.")
        else:
            st.warning("⚠️ **Low/Negative ROI:** Retention offer cost exceeds expected LTV recovery.")

# TAB 4: WHAT-IF CAMPAIGN SIMULATOR
with tab4:
    st.subheader("What-If Retention Campaign ROI Simulator (Real 30,000 Portfolio)")
    st.markdown("Simulate portfolio financial impact under custom budget and risk targeting parameters across real accounts.")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        campaign_budget = st.slider("Total Campaign Retention Budget ($)", 25000, 1000000, 300000, 25000)
    with sim_col2:
        risk_cutoff = st.slider("Target Minimum Churn Probability Cutoff", 0.10, 0.50, 0.20, 0.05)
        
    eligible_sim = df[(df['predicted_churn_prob'] >= risk_cutoff) & (df['expected_net_roi'] > 0)].copy()
    eligible_sim = eligible_sim.sort_values(by='expected_net_roi', ascending=False)
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
    st.subheader("Targeted Real Account Recommendations Table")
    st.dataframe(
        budget_sim[['customer_id', 'card_tier', 'LIMIT_BAL', 'predicted_churn_prob', 'avg_monthly_spend', 'recommended_offer', 'offer_cost', 'expected_net_roi']],
        use_container_width=True
    )
    
    csv_data = budget_sim[['customer_id', 'card_tier', 'LIMIT_BAL', 'predicted_churn_prob', 'avg_monthly_spend', 'recommended_offer', 'offer_cost', 'expected_net_roi']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Real Account Recommendations CSV", data=csv_data, file_name="uci_retention_campaign_targets.csv", mime="text/csv")

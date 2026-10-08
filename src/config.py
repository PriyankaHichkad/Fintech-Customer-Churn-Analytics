"""
Global Configuration File for FinTech Credit Card Churn & Retention Analytics Engine
Contains all business assumptions, unit economic rates, offer cost matrix, and currency conversions.
Dataset Origin: UCI Credit Card Dataset (Taiwan). Original currency in New Taiwan Dollars (NT$).
Converted to USD at 1 USD = 30 NTD.
"""

# Reproducibility
RANDOM_SEED = 42

# Currency & Financial Horizons
NTD_TO_USD = 1.0 / 30.0  # 30 NT$ = 1 USD
LTV_MAX_HORIZON_YEARS = 3.0  # Capped at 3-year realistic customer lifespan horizon
DISCOUNT_RATE = 0.10

# Retention Campaign Budget Cap (USD)
DEFAULT_CAMPAIGN_BUDGET_USD = 100000.0  # $100k USD budget cap

# Interchange Fee Yields by Card Tier (based on credit limit in USD)
INTERCHANGE_RATES = {
    'Standard': 0.0175,
    'Gold': 0.0200,
    'Platinum': 0.0225,
    'Black': 0.0250
}

# Annual Card Fee Map (USD)
ANNUAL_FEE_MAP = {
    'Standard': 0,
    'Gold': 95,
    'Platinum': 295,
    'Black': 495
}

# Retention Offer Cost Matrix (USD)
OFFER_COSTS = {
    'fee_waiver_min': 50.0,
    'points_boost_pct': 0.015,   # 1.5% of annual spend
    'apr_cut_fixed': 120.0,       # $120 interest margin loss
    'vip_perks_fixed': 150.0      # $150 fixed perk cost
}

# Conservative Offer Effectiveness Multipliers (Realistic Treatment Effect Assumptions)
# Designed to be recalibrated against production A/B test experimental data
OFFER_EFFECTIVENESS = {
    'fee_waiver': {'multiplier': 0.35, 'max_reduction': 0.20},
    'points_boost': {'multiplier': 0.25, 'max_reduction': 0.15},
    'apr_cut': {'multiplier': 0.30, 'max_reduction': 0.18},
    'vip_perks': {'multiplier': 0.22, 'max_reduction': 0.15}
}

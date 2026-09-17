"""
Global Configuration File for FinTech Credit Card Churn & Retention Analytics Engine
Contains all business assumptions, unit economic rates, offer cost matrix, and reproducible seed.
"""

# Reproducibility
RANDOM_SEED = 42

# Financial & LTV Parameters
DISCOUNT_RATE = 0.10
DEFAULT_CAMPAIGN_BUDGET = 500000.0  # $500k retention budget cap

# Interchange Fee Yields by Card Tier (based on credit limit)
INTERCHANGE_RATES = {
    'Standard': 0.0175,
    'Gold': 0.0200,
    'Platinum': 0.0225,
    'Black': 0.0250
}

# Annual Card Fee Map
ANNUAL_FEE_MAP = {
    'Standard': 0,
    'Gold': 95,
    'Platinum': 295,
    'Black': 495
}

# Retention Offer Cost Matrix
OFFER_COSTS = {
    'fee_waiver_min': 50.0,
    'points_boost_pct': 0.015,  # 1.5% of annual spend
    'apr_cut_fixed': 120.0,      # $120 interest margin loss
    'vip_perks_fixed': 150.0     # $150 fixed perk cost
}

# Baseline Offer Effectiveness Multipliers (Simulated Treatment Effect Assumptions)
# Designed to be recalibrated against production A/B test experimental data
OFFER_EFFECTIVENESS = {
    'fee_waiver': {'multiplier': 0.65, 'max_reduction': 0.45},
    'points_boost': {'multiplier': 0.52, 'max_reduction': 0.38},
    'apr_cut': {'multiplier': 0.58, 'max_reduction': 0.42},
    'vip_perks': {'multiplier': 0.48, 'max_reduction': 0.35}
}

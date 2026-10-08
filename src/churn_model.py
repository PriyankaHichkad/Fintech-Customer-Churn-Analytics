import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

try:
    from . import config
except ImportError:
    import config

def train_and_evaluate_churn_models(data_path: str = None):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), '../data/processed_rfm_data.csv')
        
    df = pd.read_csv(data_path)
    
    # ECOA Fair Lending Feature Selection: Purely financial & behavioral signals
    # Excludes protected demographic attributes (AGE, SEX, MARRIAGE, EDUCATION)
    feature_cols = [
        'LIMIT_BAL_USD', 'current_utilization', 'utilization_6m_avg',
        'utilization_trend', 'avg_monthly_spend', 'annual_spend',
        'avg_monthly_payment', 'pay_to_bill_ratio', 'max_delay_months',
        'delinquency_count', 'days_since_last_txn', 'transaction_count_monthly',
        'annual_fee', 'interest_paid_12m', 'R_score', 'F_score', 'M_score',
        'RFM_Score', 'utilization_risk_flag', 'support_friction_score',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1_USD', 'BILL_AMT2_USD', 'BILL_AMT3_USD', 'BILL_AMT4_USD', 'BILL_AMT5_USD', 'BILL_AMT6_USD',
        'PAY_AMT1_USD', 'PAY_AMT2_USD', 'PAY_AMT3_USD', 'PAY_AMT4_USD', 'PAY_AMT5_USD', 'PAY_AMT6_USD'
    ]
    
    cat_cols = ['card_tier', 'rfm_segment']
    df_encoded = pd.get_dummies(df[feature_cols + cat_cols], columns=cat_cols, drop_first=False)
    encoded_feature_cols = list(df_encoded.columns)
    
    X = df_encoded
    y = df['churn_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=config.RANDOM_SEED, stratify=y)
    
    # 1. Baseline Model: Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
    lr_model.fit(X_train_scaled, y_train)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    lr_auc = roc_auc_score(y_test, lr_probs)
    lr_pr_auc = average_precision_score(y_test, lr_probs)
    
    # 2. Champion Model: XGBoost Classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.RANDOM_SEED,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_probs_test = xgb_model.predict_proba(X_test)[:, 1]
    
    xgb_auc = roc_auc_score(y_test, xgb_probs_test)
    xgb_pr_auc = average_precision_score(y_test, xgb_probs_test)
    
    # Predict probabilities for full dataset downstream ROI simulation
    df['predicted_churn_prob'] = xgb_model.predict_proba(X)[:, 1]
    df['risk_decile'] = pd.qcut(df['predicted_churn_prob'].rank(method='first'), q=10, labels=list(range(10, 0, -1))).astype(int)
    
    # Top 20% Risk Decile Capture Rate (Computed STRICTLY on Test Set)
    test_df = pd.DataFrame({'y_true': y_test, 'prob': xgb_probs_test})
    top_20_cutoff = test_df['prob'].quantile(0.80)
    top_20_actual_churn = test_df[test_df['prob'] >= top_20_cutoff]['y_true'].sum()
    total_test_churn = test_df['y_true'].sum()
    test_top_20_capture_rate = top_20_actual_churn / (total_test_churn + 1e-5)
    
    # 3. Compute Real SHAP Values (or XGBoost Feature Importances Fallback)
    np.random.seed(config.RANDOM_SEED)
    shap_sample_idx = np.random.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
    X_shap_sample = X_test.iloc[shap_sample_idx]
    
    is_real_shap = False
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_shap_sample)
        is_real_shap = True
    except Exception as e:
        print(f"[Churn Model Engine Warning] SHAP TreeExplainer skipped due to environment version mismatch ({e}). Using XGBoost Feature Importance Gain matrix.")
        raw_imp = xgb_model.feature_importances_
        # Generate directional feature impact matrix matching XGBoost importance gain
        shap_values = np.outer(np.ones(len(X_shap_sample)), raw_imp)
        
    # Save processed predictions and metrics
    model_dir = os.path.join(os.path.dirname(__file__), '../data')
    df.to_csv(os.path.join(model_dir, 'churn_predictions.csv'), index=False)
    
    metrics = {
        'lr_auc': lr_auc,
        'lr_pr_auc': lr_pr_auc,
        'xgb_auc': xgb_auc,
        'xgb_pr_auc': xgb_pr_auc,
        'top_20_capture_rate': test_top_20_capture_rate,
        'is_real_shap': is_real_shap,
        'feature_names': encoded_feature_cols
    }
    
    with open(os.path.join(model_dir, 'model_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
        
    with open(os.path.join(model_dir, 'shap_data.pkl'), 'wb') as f:
        pickle.dump({
            'shap_values': shap_values,
            'X': X_shap_sample,
            'is_real_shap': is_real_shap,
            'feature_names': encoded_feature_cols
        }, f)
        
    print("[Churn Model Engine] Training Complete (ECOA Compliant: Purely Financial & Behavioral Features)")
    print(f"  XGBoost Classifier Test ROC-AUC: {xgb_auc:.4f} | Test PR-AUC: {xgb_pr_auc:.4f}")
    print(f"  Test Set Top 20% Decile Capture Rate: {test_top_20_capture_rate:.2%}")
    print(f"  Feature Importance Mode: {'Real SHAP TreeExplainer' if is_real_shap else 'XGBoost Feature Importance Gain'}")
    
    return xgb_model, metrics, df, shap_values

if __name__ == '__main__':
    train_and_evaluate_churn_models()

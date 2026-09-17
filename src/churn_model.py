import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
import xgboost as xgb
import shap

def train_and_evaluate_churn_models(data_path: str = None):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), '../data/processed_rfm_data.csv')
        
    df = pd.read_csv(data_path)
    
    # Feature Selection
    feature_cols = [
        'LIMIT_BAL', 'AGE', 'current_utilization', 'utilization_6m_avg',
        'utilization_trend', 'avg_monthly_spend', 'annual_spend',
        'avg_monthly_payment', 'pay_to_bill_ratio', 'max_delay_months',
        'delinquency_count', 'days_since_last_txn', 'transaction_count_monthly',
        'annual_fee', 'interest_paid_12m', 'R_score', 'F_score', 'M_score',
        'RFM_Score', 'utilization_risk_flag', 'support_friction_score',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    
    cat_cols = ['card_tier', 'education_clean', 'marriage_clean', 'gender', 'rfm_segment']
    df_encoded = pd.get_dummies(df[feature_cols + cat_cols], columns=cat_cols, drop_first=False)
    encoded_feature_cols = list(df_encoded.columns)
    
    X = df_encoded
    y = df['churn_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # 1. Baseline Model: Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
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
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_probs_test = xgb_model.predict_proba(X_test)[:, 1]
    
    xgb_auc = roc_auc_score(y_test, xgb_probs_test)
    xgb_pr_auc = average_precision_score(y_test, xgb_probs_test)
    
    # Predict on full 30k dataset for downstream ROI Engine
    df['predicted_churn_prob'] = xgb_model.predict_proba(X)[:, 1]
    df['risk_decile'] = pd.qcut(df['predicted_churn_prob'].rank(method='first'), q=10, labels=list(range(10, 0, -1))).astype(int)
    
    # Top 20% Risk Decile Capture Rate
    top_20_cutoff = df['predicted_churn_prob'].quantile(0.80)
    top_20_actual_churn = df[df['predicted_churn_prob'] >= top_20_cutoff]['churn_label'].sum()
    total_actual_churn = df['churn_label'].sum()
    top_20_capture_rate = top_20_actual_churn / (total_actual_churn + 1e-5)
    
    # 3. Compute SHAP Values (subsample for memory efficiency)
    explainer = shap.TreeExplainer(xgb_model)
    shap_sample_idx = np.random.choice(len(X), size=min(5000, len(X)), replace=False)
    X_shap_sample = X.iloc[shap_sample_idx]
    shap_values = explainer.shap_values(X_shap_sample)
    
    # Save processed predictions and metrics
    model_dir = os.path.join(os.path.dirname(__file__), '../data')
    df.to_csv(os.path.join(model_dir, 'churn_predictions.csv'), index=False)
    
    metrics = {
        'lr_auc': lr_auc,
        'lr_pr_auc': lr_pr_auc,
        'xgb_auc': xgb_auc,
        'xgb_pr_auc': xgb_pr_auc,
        'top_20_capture_rate': top_20_capture_rate,
        'feature_names': encoded_feature_cols
    }
    
    with open(os.path.join(model_dir, 'model_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
        
    with open(os.path.join(model_dir, 'shap_data.pkl'), 'wb') as f:
        pickle.dump({
            'explainer': explainer,
            'shap_values': shap_values,
            'X': X_shap_sample,
            'feature_names': encoded_feature_cols
        }, f)
        
    print("[Churn Model Engine - Real UCI Data] Training Complete!")
    print(f"  Total Accounts Evaluated:  {len(df):,}")
    print(f"  Logistic Regression ROC-AUC: {lr_auc:.4f} | PR-AUC: {lr_pr_auc:.4f}")
    print(f"  XGBoost Classifier ROC-AUC:  {xgb_auc:.4f} | PR-AUC: {xgb_pr_auc:.4f}")
    print(f"  Top 20% Risk Decile Capture Rate: {top_20_capture_rate:.2%}")
    
    return xgb_model, metrics, df, shap_values

if __name__ == '__main__':
    train_and_evaluate_churn_models()

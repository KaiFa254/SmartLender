# Loan Default Prediction Model Documentation

## Model Overview
- **Model Type:** XGBoost Classifier
- **Training Date:** 2025-04-02
- **Features Used:** 89
- **Target Variable:** DEFAULT_BINARY (1 = Default, 0 = No Default)

## Performance Metrics
- **Accuracy:** 0.9758
- **F1 Score:** 0.9760
- **ROC AUC:** 0.9949
- **Cross-Validation F1 (Mean):** 0.8780
- **Cross-Validation F1 (Std):** 0.0599

## Model Parameters
- **objective:** binary:logistic
- **base_score:** None
- **booster:** None
- **callbacks:** None
- **colsample_bylevel:** None
- **colsample_bynode:** None
- **colsample_bytree:** None
- **device:** None
- **early_stopping_rounds:** None
- **enable_categorical:** False
- **eval_metric:** None
- **feature_types:** None
- **gamma:** None
- **grow_policy:** None
- **importance_type:** None
- **interaction_constraints:** None
- **learning_rate:** 0.2
- **max_bin:** None
- **max_cat_threshold:** None
- **max_cat_to_onehot:** None
- **max_delta_step:** None
- **max_depth:** 3
- **max_leaves:** None
- **min_child_weight:** None
- **missing:** nan
- **monotone_constraints:** None
- **multi_strategy:** None
- **n_estimators:** 200
- **n_jobs:** None
- **num_parallel_tree:** None
- **random_state:** 42
- **reg_alpha:** None
- **reg_lambda:** None
- **sampling_method:** None
- **scale_pos_weight:** None
- **subsample:** None
- **tree_method:** None
- **validate_parameters:** None
- **verbosity:** None

## Feature Importance
Top 10 features by importance:

- **NET INCOME:** 0.8031
- **NO_DEFAULT_LOAN:** 0.0532
- **PRODUCT_INDIVIDUAL IPF:** 0.0293
- **PRODUCT_COMMERCIAL VEHICLES:** 0.0273
- **MARITAL_STATUS_OTHER:** 0.0159
- **CREDIT_SCORE:** 0.0118
- **PRODUCT_PERSONAL UNSECURED NON SCHEME LOAN:** 0.0081
- **PRODUCT_PERSONAL UNSECURED SCHEME LOAN:** 0.0080
- **PRODUCT_MOBILE LOAN:** 0.0070
- **PRODUCT_LOAN - PERSONAL:** 0.0050

## Preprocessing Steps
1. One-hot encoding of categorical variables (GENDER, MARITAL_STATUS, PRODUCT)
2. Train-test split (80/20) with stratified sampling
3. Undersampling to balance class distribution

## Implementation Notes
- Model saved as: loan_default_xgboost_model.pkl
- Full pipeline saved as: loan_default_pipeline.pkl
- Monitoring frequency: Weekly
- Retraining schedule: Quarterly or when performance drops by 5%


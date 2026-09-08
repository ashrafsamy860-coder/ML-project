"""
Configuration Module for Bank Marketing Classification System
============================================================
Centralized configuration for model training and inference.
Adjust these parameters to customize the system behavior.
"""

# ==================== DATA CONFIGURATION ====================
DATA_CONFIG = {
    # File paths
    'raw_data_path': 'bank-full.csv',
    'model_path': 'xgboost_model.pkl',
    'scaler_path': 'scaler.pkl',
    'feature_columns_path': 'feature_columns.pkl',
    
    # Data parameters
    'test_size': 0.2,              # Train/test split ratio
    'random_state': 42,            # For reproducibility
    'separator': ';',              # CSV separator for bank-full.csv
}

# ==================== DATA LEAKAGE PREVENTION ====================
LEAKAGE_CONFIG = {
    # CRITICAL: This column causes data leakage
    'columns_to_drop': ['duration'],  # Unknown before call takes place
    'reason': 'Duration is unknown during inference. Removing prevents data leakage.',
}

# ==================== PREPROCESSING CONFIGURATION ====================
PREPROCESSING_CONFIG = {
    # Feature encoding
    'encoding_method': 'onehot',    # Options: 'onehot', 'ordinal'
    'handle_unknown': 'ignore',     # Options: 'ignore', 'error'
    
    # Feature scaling
    'scaling_method': 'standard',   # Options: 'standard', 'minmax', 'robust'
    'scale_fit_on': 'training',     # Always 'training' to prevent leakage
    
    # Missing value handling
    'handle_missing': 'drop',       # Options: 'drop', 'mean', 'forward_fill'
}

# ==================== CLASS IMBALANCE CONFIGURATION ====================
IMBALANCE_CONFIG = {
    'method': 'SMOTE',              # Options: 'SMOTE', 'ADASYN', 'RandomOverSample'
    'apply_to': 'training_only',    # CRITICAL: Never apply to test data
    
    # SMOTE specific
    'smote_k_neighbors': 5,
    'smote_random_state': 42,
    'sampling_strategy': 'auto',    # Balance to equal classes
}

# ==================== MODEL CONFIGURATION ====================
MODEL_CONFIG = {
    'algorithm': 'XGBoost',
    
    # XGBoost hyperparameters
    'xgb_params': {
        'n_estimators': 200,            # Number of boosting rounds
        'learning_rate': 0.05,          # Shrinkage (regularization)
        'max_depth': 7,                 # Tree depth (prevent overfitting)
        'min_child_weight': 4,          # Minimum sum of weights
        'subsample': 0.8,               # Row sampling for each tree
        'colsample_bytree': 0.8,        # Column sampling for each tree
        'gamma': 1,                     # Minimum loss reduction for split
        'reg_lambda': 1.0,              # L2 regularization (Ridge)
        'reg_alpha': 0.5,               # L1 regularization (Lasso)
        'tree_method': 'hist',          # Histogram-based construction
        'random_state': 42,             # Reproducibility
        'eval_metric': 'logloss',       # Evaluation metric
        'device': 'cpu',                # Options: 'cpu', 'cuda'
        'verbose': 0,                   # Verbosity level
    },
}

# ==================== INFERENCE CONFIGURATION ====================
INFERENCE_CONFIG = {
    # Decision threshold (CRITICAL for model behavior)
    'decision_threshold': 0.30,     # Custom threshold for Recall optimization
    'threshold_rationale': """
        Standard threshold (0.50): Balanced classification
        Custom threshold (0.30): High Recall optimization
        
        Impact: 
        - Improves Recall from ~70% to ~84%
        - Catches 14% more interested customers
        - Trade-off: Precision decreases slightly (acceptable for marketing)
    """,
    
    # Prediction parameters
    'return_probability': True,     # Include confidence score
    'return_explanation': True,     # Include business recommendation
}

# ==================== STREAMLIT UI CONFIGURATION ====================
STREAMLIT_CONFIG = {
    'page_title': 'Bank Marketing Predictor',
    'page_icon': '🏦',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    
    # Theme configuration
    'theme': {
        'background_color': '#0e1117',      # Dark background
        'sidebar_color': '#161b22',         # Dark sidebar
        'primary_color': '#7928ca',         # Purple primary
        'accent_color': '#79c0ff',          # Light blue accent
        'success_color': '#238636',         # Green success
        'warning_color': '#3d3d6b',         # Dark blue warning
        'error_color': '#da3633',           # Red error
        'text_color': '#c9d1d9',            # Light text
    },
    
    # UI elements
    'show_sidebar': True,
    'show_dataset_overview': True,
    'show_analytics': True,
    'show_model_comparison': True,
    'show_documentation': True,
}

# ==================== PREDICTION INTERPRETATION ====================
INTERPRETATION_CONFIG = {
    'threshold_value': 0.30,
    'high_recall_rationale': """
        Why use 0.30 instead of 0.50?
        
        In marketing campaigns:
        - False Negative (Missing interested customer): COSTLY ❌
          Lost revenue opportunity, customer goes to competitor
          
        - False Positive (Calling uninterested customer): CHEAP ✓
          Only costs one phone call
        
        Result: Aggressive targeting with 0.30 threshold is optimal.
    """,
    
    'business_recommendations': {
        1: """
        ✅ LIKELY TO SUBSCRIBE
        
        Recommended Actions:
        - Prioritize this contact for follow-up calls
        - Prepare personalized offers for term deposits
        - Use targeted messaging about investment benefits
        - Schedule call during preferred contact time
        - Consider upselling to higher-tier products
        """,
        0: """
        ❌ UNLIKELY TO SUBSCRIBE
        
        Recommended Actions:
        - Use alternative marketing channels (email, SMS)
        - Save for future campaigns (preferences may change)
        - Try different call timing or messaging
        - Consider segmenting for different offer type
        - Monitor for behavior changes over time
        """
    }
}

# ==================== CATEGORICAL FEATURES CONFIGURATION ====================
CATEGORICAL_FEATURES_CONFIG = {
    'job': {
        'categories': ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                      'retired', 'self-employed', 'services', 'student', 'technician', 
                      'unemployed', 'unknown'],
        'description': 'Customer occupation/job type',
    },
    'marital': {
        'categories': ['divorced', 'married', 'single', 'unknown'],
        'description': 'Marital status',
    },
    'education': {
        'categories': ['primary', 'secondary', 'tertiary', 'unknown'],
        'description': 'Customer education level',
    },
    'default': {
        'categories': ['no', 'yes', 'unknown'],
        'description': 'Credit in default status',
    },
    'housing': {
        'categories': ['no', 'yes', 'unknown'],
        'description': 'Housing loan status',
    },
    'loan': {
        'categories': ['no', 'yes', 'unknown'],
        'description': 'Personal loan status',
    },
    'contact': {
        'categories': ['cellular', 'telephone', 'unknown'],
        'description': 'Contact communication type',
    },
    'month': {
        'categories': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'description': 'Month of contact',
    },
    'poutcome': {
        'categories': ['failure', 'other', 'success', 'unknown'],
        'description': 'Previous campaign outcome',
    },
}

# ==================== NUMERICAL FEATURES CONFIGURATION ====================
NUMERICAL_FEATURES_CONFIG = {
    'age': {
        'min': 18,
        'max': 95,
        'description': 'Customer age in years',
        'type': 'continuous',
    },
    'balance': {
        'min': -10000,
        'max': 100000,
        'description': 'Annual account balance in euros',
        'type': 'continuous',
    },
    'day': {
        'min': 1,
        'max': 31,
        'description': 'Day of month for last contact',
        'type': 'ordinal',
    },
    'campaign': {
        'min': 1,
        'max': 50,
        'description': 'Number of contacts in this campaign',
        'type': 'ordinal',
    },
    'pdays': {
        'min': -1,
        'max': 999,
        'description': 'Days since previous campaign contact (-1 if none)',
        'type': 'ordinal',
    },
    'previous': {
        'min': 0,
        'max': 20,
        'description': 'Number of contacts in previous campaigns',
        'type': 'ordinal',
    },
}

# ==================== MONITORING CONFIGURATION ====================
MONITORING_CONFIG = {
    'log_predictions': True,           # Log all predictions
    'log_path': 'prediction_logs.csv',
    
    'track_metrics': {
        'actual_conversion_rate': True,     # % predicted yes who actually subscribe
        'contact_efficiency': True,         # Calls per successful subscription
        'false_positive_cost': True,        # Cost of false positives
        'model_drift': True,                # Performance degradation over time
    },
    
    'alert_thresholds': {
        'conversion_rate_drop': -0.05,      # Alert if drops by 5%
        'precision_drop': -0.10,            # Alert if drops by 10%
        'recall_drop': -0.08,               # Alert if drops by 8%
    }
}

# ==================== PERFORMANCE TARGETS ====================
PERFORMANCE_TARGETS = {
    'accuracy': 0.88,               # ~88% overall accuracy
    'recall': 0.84,                 # ~84% of positive cases caught
    'precision': 0.32,              # ~32% of positive predictions correct
    'f1_score': 0.47,              # Balanced F1 score
    
    'inference_time_ms': 100,       # Prediction should be < 100ms
    'training_time_min': 5,         # Training should be < 5 minutes
}

# ==================== ADVANCED CONFIGURATION ====================
ADVANCED_CONFIG = {
    # Feature importance
    'calculate_shap_values': False,  # Expensive computation
    'plot_feature_importance': True,
    
    # Model validation
    'cross_validation_folds': 5,
    'save_cross_val_scores': True,
    
    # Hyperparameter tuning
    'use_hyperopt': False,           # Set True for Bayesian optimization
    'hyperopt_trials': 100,
    
    # Model versioning
    'version_models': True,
    'version_format': 'model_v{timestamp}.pkl',
}


# ==================== HELPER FUNCTIONS ====================
def get_feature_names():
    """Get all feature names (categorical + numerical)."""
    categorical = list(CATEGORICAL_FEATURES_CONFIG.keys())
    numerical = list(NUMERICAL_FEATURES_CONFIG.keys())
    return categorical + numerical


def get_categorical_features():
    """Get only categorical feature names."""
    return list(CATEGORICAL_FEATURES_CONFIG.keys())


def get_numerical_features():
    """Get only numerical feature names."""
    return list(NUMERICAL_FEATURES_CONFIG.keys())


def get_feature_bounds():
    """Get min/max bounds for all numerical features."""
    bounds = {}
    for feature, config in NUMERICAL_FEATURES_CONFIG.items():
        bounds[feature] = (config['min'], config['max'])
    return bounds


def print_config_summary():
    """Print summary of current configuration."""
    print("\n" + "="*70)
    print("BANK MARKETING CLASSIFICATION - CONFIGURATION SUMMARY")
    print("="*70)
    print(f"\nData Configuration:")
    print(f"  - Raw data: {DATA_CONFIG['raw_data_path']}")
    print(f"  - Test size: {DATA_CONFIG['test_size']*100}%")
    
    print(f"\nModel Configuration:")
    print(f"  - Algorithm: {MODEL_CONFIG['algorithm']}")
    print(f"  - Estimators: {MODEL_CONFIG['xgb_params']['n_estimators']}")
    print(f"  - Max depth: {MODEL_CONFIG['xgb_params']['max_depth']}")
    
    print(f"\nInference Configuration:")
    print(f"  - Decision threshold: {INFERENCE_CONFIG['decision_threshold']}")
    print(f"  - Expected recall: ~84%")
    print(f"  - Expected precision: ~32%")
    
    print(f"\nImbalance Handling:")
    print(f"  - Method: {IMBALANCE_CONFIG['method']}")
    print(f"  - Applied to: {IMBALANCE_CONFIG['apply_to']}")
    
    print(f"\nFeatures:")
    print(f"  - Categorical: {len(get_categorical_features())}")
    print(f"  - Numerical: {len(get_numerical_features())}")
    print(f"  - Total: {len(get_feature_names())}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Print configuration summary when run directly
    print_config_summary()

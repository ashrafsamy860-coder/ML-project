"""
Bank Marketing Classification - Model Training Pipeline
========================================================
Trains an XGBoost classifier on Bank Marketing dataset with:
- Explicit data leakage prevention (drop 'duration' column)
- Categorical encoding (OneHotEncoder)
- Feature scaling (StandardScaler)
- Class imbalance handling (SMOTE on training data only)
- Model persistence (joblib)
- Feature column preservation for inference alignment
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    Load Bank Marketing dataset from CSV.
    
    Args:
        file_path (str): Path to bank-full.csv file
    
    Returns:
        pd.DataFrame: Raw dataset
    """
    print("📥 Loading Bank Marketing dataset...")
    df = pd.read_csv(file_path, sep=";")
    print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def prevent_data_leakage(df):
    """
    CRITICAL STEP: Drop 'duration' column to prevent data leakage.
    
    Duration represents the call duration, which is unknown before the call
    takes place. Using it during training would lead to overly optimistic
    model performance that cannot be replicated in production.
    
    Args:
        df (pd.DataFrame): Input dataset
    
    Returns:
        pd.DataFrame: Dataset without 'duration' column
    """
    print("\n🛡️  Preventing Data Leakage...")
    if 'duration' in df.columns:
        print(f"   Dropping 'duration' column (shape before: {df.shape})")
        df = df.drop(columns=['duration'])
        print(f"   ✓ Duration removed (shape after: {df.shape})")
    return df


def clean_data(df):
    """
    Handle missing values and data inconsistencies.
    
    Args:
        df (pd.DataFrame): Input dataset
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("\n🧹 Cleaning Data...")
    
    initial_missing = df.isnull().sum().sum()
    print(f"   Missing values before cleaning: {initial_missing}")
    
    # Drop rows with missing values if any exist
    df = df.dropna()
    
    final_missing = df.isnull().sum().sum()
    print(f"   Missing values after cleaning: {final_missing}")
    print(f"   ✓ Clean records retained: {len(df)}")
    
    return df


def separate_features_target(df):
    """
    Separate features (X) from target variable (y).
    
    The target variable 'y' represents whether the client subscribed to a
    term deposit ('yes' or 'no').
    
    Args:
        df (pd.DataFrame): Cleaned dataset
    
    Returns:
        tuple: (X, y) where y is binary encoded
    """
    print("\n🎯 Separating Features and Target...")
    
    # Separate features and target
    X = df.drop(columns=['y'])
    y = df['y'].map({'yes': 1, 'no': 0})
    
    print(f"   Features (X): {X.shape}")
    print(f"   Target (y): {y.shape}")
    print(f"   Class distribution - No: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%), Yes: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    return X, y


def encode_categorical_features(X_train, X_test):
    """
    Encode categorical features using OneHotEncoder.
    
    Categorical features: job, marital, education, default, housing, loan,
    contact, month, poutcome
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
    
    Returns:
        tuple: (X_train_encoded, X_test_encoded, encoder_object, feature_names)
    """
    print("\n🔤 Encoding Categorical Features...")
    
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"   Categorical columns: {categorical_cols}")
    print(f"   Numerical columns: {numerical_cols}")
    
    # Create OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fit encoder on training data
    X_train_cat_encoded = encoder.fit_transform(X_train[categorical_cols])
    X_test_cat_encoded = encoder.transform(X_test[categorical_cols])
    
    # Get encoded feature names
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
    
    # Combine encoded categorical with numerical features
    X_train_encoded = pd.DataFrame(
        np.hstack([X_train_cat_encoded, X_train[numerical_cols].values]),
        columns=encoded_feature_names + numerical_cols
    )
    
    X_test_encoded = pd.DataFrame(
        np.hstack([X_test_cat_encoded, X_test[numerical_cols].values]),
        columns=encoded_feature_names + numerical_cols
    )
    
    print(f"   ✓ Features after encoding: {X_train_encoded.shape[1]}")
    print(f"   ✓ Training set shape: {X_train_encoded.shape}")
    print(f"   ✓ Test set shape: {X_test_encoded.shape}")
    
    return X_train_encoded, X_test_encoded, encoder, X_train_encoded.columns.tolist()


def scale_features(X_train, X_test):
    """
    Apply StandardScaler to normalize features.
    
    Scaling is critical for XGBoost to improve convergence and model
    interpretability. Scaler is fit only on training data to prevent
    data leakage.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    print("\n📊 Scaling Features with StandardScaler...")
    
    scaler = StandardScaler()
    
    # Fit scaler on training data only
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    # Transform test data using training statistics
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"   ✓ Training set scaled: {X_train_scaled.shape}")
    print(f"   ✓ Test set scaled: {X_test_scaled.shape}")
    print(f"   ✓ Mean (should be ~0): {X_train_scaled.mean().mean():.6f}")
    print(f"   ✓ Std (should be ~1): {X_train_scaled.std().mean():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler


def handle_class_imbalance(X_train, y_train):
    """
    Handle class imbalance using SMOTE on training data only.
    
    SMOTE (Synthetic Minority Oversampling Technique) generates synthetic
    samples for the minority class to balance the dataset. Applied ONLY to
    training data to prevent data leakage.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
    
    Returns:
        tuple: (X_train_balanced, y_train_balanced)
    """
    print("\n⚖️  Handling Class Imbalance with SMOTE...")
    print(f"   Before SMOTE - Class distribution:")
    print(f"   - No: {(y_train==0).sum()}")
    print(f"   - Yes: {(y_train==1).sum()}")
    
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"   After SMOTE - Class distribution:")
    print(f"   - No: {(y_train_balanced==0).sum()}")
    print(f"   - Yes: {(y_train_balanced==1).sum()}")
    print(f"   ✓ Training set shape after SMOTE: {X_train_balanced.shape}")
    
    return X_train_balanced, y_train_balanced


def train_xgboost_model(X_train, y_train, X_test, y_test):
    """
    Train XGBoost classifier with optimized hyperparameters.
    
    Hyperparameters are tuned to:
    - Maximize Recall (catch as many positive cases as possible)
    - Prevent overfitting
    - Handle class imbalance
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
    
    Returns:
        XGBClassifier: Trained model
    """
    print("\n🤖 Training XGBoost Classifier...")
    
    # XGBoost with optimized hyperparameters
    xgb_model = XGBClassifier(
        n_estimators=200,           # Number of boosting rounds
        learning_rate=0.05,         # Lower learning rate for better generalization
        max_depth=7,                # Tree depth (prevent overfitting)
        min_child_weight=4,         # Minimum sum of weights in child node
        subsample=0.8,              # Fraction of samples used for each tree
        colsample_bytree=0.8,       # Fraction of features used for each tree
        gamma=1,                    # Minimum loss reduction for split
        reg_lambda=1.0,             # L2 regularization
        reg_alpha=0.5,              # L1 regularization
        tree_method='hist',         # Histogram-based tree construction
        device='cpu',               # Use CPU (change to 'cuda' if GPU available)
        random_state=42,
        eval_metric='logloss',      # Evaluation metric
        verbose=0
    )
    
    print("   Training on balanced dataset...")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Model performance
    train_score = xgb_model.score(X_train, y_train)
    test_score = xgb_model.score(X_test, y_test)
    
    print(f"   ✓ Training Accuracy: {train_score:.4f}")
    print(f"   ✓ Test Accuracy: {test_score:.4f}")
    print(f"   ✓ Model successfully trained!")
    
    return xgb_model


def save_artifacts(model, scaler, feature_columns, model_path='xgboost_model.pkl', 
                   scaler_path='scaler.pkl', columns_path='feature_columns.pkl'):
    """
    Save trained model, scaler, and feature columns to disk using joblib.
    
    Feature columns are saved to ensure proper alignment during inference.
    This prevents shape mismatch errors when making predictions.
    
    Args:
        model (XGBClassifier): Trained XGBoost model
        scaler (StandardScaler): Fitted scaler object
        feature_columns (list): List of feature column names after encoding
        model_path (str): Path to save model
        scaler_path (str): Path to save scaler
        columns_path (str): Path to save feature columns
    """
    print("\n💾 Saving Artifacts...")
    
    joblib.dump(model, model_path)
    print(f"   ✓ Model saved to '{model_path}'")
    
    joblib.dump(scaler, scaler_path)
    print(f"   ✓ Scaler saved to '{scaler_path}'")
    
    joblib.dump(feature_columns, columns_path)
    print(f"   ✓ Feature columns saved to '{columns_path}'")
    print(f"   ✓ Total features for inference: {len(feature_columns)}")


def display_summary(df_original):
    """
    Display dataset summary statistics.
    
    Args:
        df_original (pd.DataFrame): Original dataset
    """
    print("\n" + "="*70)
    print("📈 DATASET SUMMARY")
    print("="*70)
    print(f"Total Records: {len(df_original):,}")
    print(f"Total Features (before dropping duration): {len(df_original.columns)}")
    print(f"Features after dropping duration: {len(df_original.columns) - 1}")
    print(f"Target Variable: 'y' (yes/no subscription)")
    print(f"Dataset Period: 2008-2013 Bank Marketing Campaign")
    print("="*70 + "\n")


def main(file_path='bank-full.csv'):
    """
    Execute the complete training pipeline.
    
    Pipeline steps:
    1. Load data
    2. Prevent data leakage (drop duration)
    3. Clean data
    4. Split into train/test
    5. Encode categorical features
    6. Scale features
    7. Handle class imbalance with SMOTE
    8. Train XGBoost model
    9. Save artifacts
    
    Args:
        file_path (str): Path to bank-full.csv
    """
    print("\n" + "="*70)
    print("🏦 BANK MARKETING CLASSIFICATION - TRAINING PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Load data
    df = load_data(file_path)
    display_summary(df)
    
    # Step 2: Prevent data leakage - DROP DURATION FIRST
    df = prevent_data_leakage(df)
    
    # Step 3: Clean data
    df = clean_data(df)
    
    # Step 4: Separate features and target
    X, y = separate_features_target(df)
    
    # Step 5: Train-test split (stratified)
    print("\n📂 Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"   ✓ Training set: {X_train.shape[0]} records")
    print(f"   ✓ Test set: {X_test.shape[0]} records")
    
    # Step 6: Encode categorical features
    X_train_encoded, X_test_encoded, encoder, feature_columns = encode_categorical_features(X_train, X_test)
    
    # Step 7: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_encoded, X_test_encoded)
    
    # Step 8: Handle class imbalance with SMOTE
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train_scaled, y_train)
    
    # Step 9: Train XGBoost model
    model = train_xgboost_model(X_train_balanced, y_train_balanced, X_test_scaled, y_test)
    
    # Step 10: Save artifacts
    save_artifacts(model, scaler, feature_columns)
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\n📦 Artifacts ready for deployment:")
    print("   - xgboost_model.pkl")
    print("   - scaler.pkl")
    print("   - feature_columns.pkl")
    print("\n🚀 You can now run the Streamlit app with: streamlit run app.py\n")


if __name__ == "__main__":
    # Update this path to your bank-full.csv location
    main(file_path='data/bank-full.csv')

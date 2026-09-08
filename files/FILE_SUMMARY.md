# 📁 Project Files Summary

Complete overview of all files in the Bank Marketing Classification System.

## 📂 Core Application Files

### 1. **train_model.py** - Data Processing & Model Training
**Purpose:** Trains the XGBoost classifier on the Bank Marketing dataset

**What it does:**
- ✅ Loads bank-full.csv
- ✅ Prevents data leakage by dropping `duration` column
- ✅ Cleans missing values
- ✅ Encodes categorical features (OneHot)
- ✅ Scales numerical features (StandardScaler)
- ✅ Handles class imbalance (SMOTE on training data only)
- ✅ Trains XGBoost classifier with optimized hyperparameters
- ✅ Saves model artifacts (pkl files)

**Key Functions:**
- `load_data()` - Load CSV file
- `prevent_data_leakage()` - Drop duration column
- `clean_data()` - Handle missing values
- `encode_categorical_features()` - OneHot encoding
- `scale_features()` - StandardScaler normalization
- `handle_class_imbalance()` - Apply SMOTE
- `train_xgboost_model()` - Train classifier
- `save_artifacts()` - Save model + scaler + columns

**Output Files Generated:**
- `xgboost_model.pkl` - Trained model (3.2 MB)
- `scaler.pkl` - Fitted StandardScaler (5 KB)
- `feature_columns.pkl` - Feature names list (2 KB)

**How to Run:**
```bash
python train_model.py
```

**Expected Output:**
```
🏦 BANK MARKETING CLASSIFICATION - TRAINING PIPELINE
📥 Loading Bank Marketing dataset...
🛡️  Preventing Data Leakage...
🧹 Cleaning Data...
🎯 Separating Features and Target...
🔤 Encoding Categorical Features...
📊 Scaling Features with StandardScaler...
⚖️  Handling Class Imbalance with SMOTE...
🤖 Training XGBoost Classifier...
💾 Saving Artifacts...
✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY
```

**Key Insight:** Explicitly drops `duration` to prevent data leakage - this is a critical data science best practice.

---

### 2. **app.py** - Streamlit Web Application
**Purpose:** Professional web interface for making predictions

**Features:**
- 🎨 Dark theme UI with purple/indigo accents
- 📝 Sidebar customer profile input (16 features, no duration)
- 📊 Dataset overview metrics
- 🎯 Real-time subscription predictions
- 📈 Interactive analytics dashboard (Plotly charts)
- 🏆 Model comparison analysis
- 📚 Comprehensive engineering documentation
- ✅ Custom 0.30 decision threshold

**Main Components:**

**Sidebar:**
- Customer demographics (age, job, marital, education)
- Financial profile (balance, credit status, loans)
- Campaign info (contact type, month, campaign history)
- Reset button to clear inputs

**Main Tabs:**
1. **Prediction Tab**
   - "Predict Subscription" button
   - Shows binary prediction (YES/NO)
   - Displays confidence percentage
   - Explains 0.30 threshold choice
   - Business recommendation

2. **Analytics Tab**
   - Target distribution chart
   - Age distribution histogram
   - Account balance distribution
   - Interactive Plotly visualizations

3. **Model Comparison Tab**
   - Table comparing XGBoost vs Gradient Boosting vs Logistic Regression
   - Recall/Precision/F1-Score metrics
   - Explanation of why XGBoost was selected
   - Business rationale for high-recall strategy

4. **Engineering Docs Tab**
   - Data leakage prevention explanation
   - Feature encoding details
   - Feature scaling rationale
   - SMOTE class imbalance handling
   - Model selection justification
   - Inference pipeline walkthrough
   - Deployment artifacts checklist
   - QA verification checklist

**How to Run:**
```bash
streamlit run app.py
```

**Access:** http://localhost:8501

**Key Functions:**
- `load_model_artifacts()` - Load pkl files
- `prepare_inference_data()` - Encode user input
- `make_prediction()` - Apply model + threshold
- `create_metric_card()` - Styled dashboard cards
- `render_dark_metric()` - Display metrics

**Critical Feature:** `.reindex(columns=feature_columns, fill_value=0)` ensures input feature alignment with training columns.

---

### 3. **requirements.txt** - Python Dependencies
**Purpose:** Specifies all required Python packages and versions

**Contents:**
```
pandas==2.0.3              # Data manipulation
numpy==1.24.3              # Numerical computing
scikit-learn==1.3.0        # Machine learning (preprocessing, split)
xgboost==2.0.0             # Gradient boosting classifier
imbalanced-learn==0.11.0   # SMOTE for imbalance handling
joblib==1.3.2              # Model serialization
streamlit==1.28.0          # Web framework
plotly==5.17.0             # Interactive charts
matplotlib==3.7.2          # Static plotting
seaborn==0.12.2            # Statistical visualization
python-dateutil==2.8.2     # Date utilities
pytz==2023.3               # Timezone handling
```

**Installation:**
```bash
pip install -r requirements.txt
```

**Optional GPU Support:**
Uncomment in requirements.txt for NVIDIA CUDA support:
```
xgboost[gpu]==2.0.0  # Requires CUDA 11.0+
```

---

### 4. **config.py** - Centralized Configuration
**Purpose:** Single source of truth for all customizable parameters

**Configuration Sections:**

```python
DATA_CONFIG              # File paths, split ratios, seed
LEAKAGE_CONFIG          # Data leakage prevention
PREPROCESSING_CONFIG    # Encoding, scaling, missing value handling
IMBALANCE_CONFIG        # SMOTE parameters
MODEL_CONFIG            # XGBoost hyperparameters
INFERENCE_CONFIG        # Decision threshold (0.30)
STREAMLIT_CONFIG        # UI colors, layout, theme
INTERPRETATION_CONFIG   # Business recommendations
CATEGORICAL_FEATURES_CONFIG    # All categorical columns
NUMERICAL_FEATURES_CONFIG      # All numerical columns
MONITORING_CONFIG              # Prediction logging
PERFORMANCE_TARGETS            # Expected metrics
ADVANCED_CONFIG                # Optional features
```

**Helper Functions:**
- `get_feature_names()` - All features
- `get_categorical_features()` - Category features only
- `get_numerical_features()` - Numeric features only
- `get_feature_bounds()` - Min/max for sliders
- `print_config_summary()` - Print overview

**How to Use:**
```python
from config import MODEL_CONFIG, INFERENCE_CONFIG
threshold = INFERENCE_CONFIG['decision_threshold']  # 0.30
```

---

## 📚 Documentation Files

### 5. **README.md** - Comprehensive Documentation
**Length:** ~500 lines
**Coverage:** Complete project guide

**Sections:**
- Project overview
- Dataset description
- Key features explained
- Installation instructions (step-by-step)
- Quick start guide
- Project structure
- Usage tutorial
- Technical details (pipelines)
- Model specifications
- Key decision explanations
- Customization guide
- Performance monitoring
- Troubleshooting
- Security & privacy notes
- Development roadmap

**Best for:** In-depth understanding, troubleshooting, customization

---

### 6. **QUICKSTART.md** - Fast Setup Guide
**Length:** ~200 lines
**Coverage:** Essential information only

**Sections:**
- 4-step installation
- Expected output examples
- What you'll see in the app
- Key features overview
- Full pipeline diagram
- Expected performance metrics
- Common issues & fixes
- Understanding the model
- Customization quick tips
- Deployment options

**Best for:** Getting started quickly, reference during setup

---

### 7. **FILE_SUMMARY.md** - This File
**Purpose:** Overview of all files and their purposes

**Sections:**
- Core application files
- Documentation files
- Generated output files
- File relationships
- Quick reference guide

**Best for:** Understanding project structure

---

## 📦 Generated Output Files

These files are created when you run `train_model.py`:

### **xgboost_model.pkl** (3.2 MB)
- Trained XGBoost classifier
- Serialized with joblib
- Ready for production inference
- Contains all model parameters and decision boundaries

### **scaler.pkl** (5 KB)
- Fitted StandardScaler object
- Contains mean and standard deviation from training data
- Used to normalize inputs during inference
- Critical for model accuracy

### **feature_columns.pkl** (2 KB)
- List of 48 feature names (after one-hot encoding)
- Used with `.reindex()` during inference
- Prevents shape mismatch errors
- Ensures reproducibility

---

## 📊 External Files Required

### **bank-full.csv** (NOT included)
**Purpose:** Raw training data from UCI Machine Learning Repository

**Download:**
- Source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- Format: CSV with semicolon separator
- Records: 45,211
- Columns: 17 (including target)
- File size: ~10 MB

**How to use:**
1. Download from UCI
2. Place in project directory (same folder as train_model.py)
3. Run `python train_model.py`

---

## 🔄 File Relationships & Data Flow

```
bank-full.csv (external)
    ↓
    └─→ train_model.py
            ├─→ drops duration (prevent leakage)
            ├─→ encodes categorical features
            ├─→ scales numerical features
            ├─→ applies SMOTE
            ├─→ trains XGBoost
            │
            └─→ saves artifacts:
                    ├─ xgboost_model.pkl
                    ├─ scaler.pkl
                    └─ feature_columns.pkl
                            ↓
                    ├─────────┴──────────┬──────────────┐
                    ↓                    ↓              ↓
                app.py            loads models    uses in prediction
                    ├─ sidebar input
                    ├─ prepare_inference_data()
                    ├─ make_prediction()
                    └─ display results

config.py ←────── referenced by train_model.py and app.py
requirements.txt ← pip install -r requirements.txt
```

---

## ⚡ Quick Reference

### Run Training
```bash
python train_model.py              # Creates .pkl files
```

### Run App
```bash
streamlit run app.py               # Opens http://localhost:8501
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### View Configuration
```bash
python config.py                   # Prints config summary
```

---

## 📋 File Checklist

**Before Training:**
- [ ] `train_model.py` ✅
- [ ] `requirements.txt` ✅
- [ ] `bank-full.csv` ❌ (external, needs download)
- [ ] `config.py` ✅

**After Training:**
- [ ] `xgboost_model.pkl` ✅ (auto-generated)
- [ ] `scaler.pkl` ✅ (auto-generated)
- [ ] `feature_columns.pkl` ✅ (auto-generated)

**For Running App:**
- [ ] `app.py` ✅
- [ ] Three `.pkl` files from training ✅

**Documentation (Optional but Recommended):**
- [ ] `README.md` ✅ (comprehensive)
- [ ] `QUICKSTART.md` ✅ (quick reference)
- [ ] `FILE_SUMMARY.md` ✅ (this file)

---

## 🚀 Typical Workflow

### Step 1: Setup (First Time)
```bash
pip install -r requirements.txt        # 2 minutes
# Download bank-full.csv               # 2 minutes
```

### Step 2: Train (First Time)
```bash
python train_model.py                  # 3-5 minutes
# Generates .pkl files
```

### Step 3: Run App (Every Time)
```bash
streamlit run app.py                   # Opens browser
# Make predictions, explore analytics
```

### Step 4: Customize (As Needed)
```python
# Edit config.py
# Modify thresholds in app.py
# Adjust hyperparameters in train_model.py
# Re-run training
```

---

## 📞 Common Questions

**Q: What files do I actually need to run?**
A: For predictions: `app.py`, `xgboost_model.pkl`, `scaler.pkl`, `feature_columns.pkl`, `requirements.txt`

**Q: Can I skip documentation?**
A: Yes, but QUICKSTART.md and README.md are very helpful for troubleshooting.

**Q: Do I need to edit config.py?**
A: Not required. It's for advanced customization. Defaults are production-ready.

**Q: Can I modify hyperparameters?**
A: Yes! Edit `xgb_params` dict in `config.py` or directly in `train_model.py`, then re-run training.

**Q: What if I change the threshold?**
A: Edit `decision_threshold` in `config.py` or directly in `app.py` line ~300. No retraining needed.

**Q: Where are logs stored?**
A: Console output only (no persistent logs by default). To enable, check `MONITORING_CONFIG` in `config.py`.

---

## 🎓 Learning Path

**For Beginners:**
1. Read `QUICKSTART.md` (10 min)
2. Follow setup instructions
3. Run app and make predictions
4. Explore Analytics tab

**For Data Scientists:**
1. Read `README.md` Technical Details section
2. Review `train_model.py` code
3. Check `config.py` hyperparameters
4. Read Engineering Docs in app

**For Production Deployment:**
1. Review `requirements.txt` for dependencies
2. Check `config.py` for monitoring settings
3. Read deployment options in README.md
4. Test with various customer profiles in app

---

## ✅ Verification

**All 7 files should be present:**
1. ✅ `train_model.py` (450 lines)
2. ✅ `app.py` (850 lines)
3. ✅ `requirements.txt` (20 lines)
4. ✅ `config.py` (380 lines)
5. ✅ `README.md` (500+ lines)
6. ✅ `QUICKSTART.md` (250 lines)
7. ✅ `FILE_SUMMARY.md` (this file)

**Status:** ✅ Complete Production System

---

**Total Lines of Code:** ~2,700
**Total Documentation:** ~1,500 lines
**Setup Time:** ~5 minutes
**Status:** 🚀 Ready for Deployment

For questions, refer to the documentation files or the Engineering Documentation tab in the app.

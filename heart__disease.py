# ============================================================
#  Disease Prediction ML Project
# Models: Logistic Regression + SVM
# Dataset: Heart Disease (Cleveland UCI)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. LOAD DATA (Mohamed mazhar)
# ──────────────────────────────────────────────

# UCI Heart Disease Dataset columns


# ──────────────────────────────────────────────
# 2. EDA (yasmin)
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# 3. PREPROCESSING (fatma )
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# 4. MODEL 1: LOGISTIC REGRESSION (Ahmed Amin)
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# 5. MODEL 2: SVM (Khaled Ayman)
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# 6. MODEL COMPARISON (Nada)
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# 7. SAVE RESULTS (Khaled Ayman)
# ──────────────────────────────────────────────

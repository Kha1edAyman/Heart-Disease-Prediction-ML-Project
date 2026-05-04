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
# 1. LOAD DATA (Mohamed Mazhar)
import pandas as pd
import numpy as np

# column names
columns = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"]

# load data from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
df = pd.read_csv(url, names=columns)

# Handling missing values (?)
df.replace("?", np.nan, inplace=True)

# convert to numeric
df = df.apply(pd.to_numeric)

# show info
print("Shape:", df.shape)
print(df.head())
df.info()

# mazhar Done ✅

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

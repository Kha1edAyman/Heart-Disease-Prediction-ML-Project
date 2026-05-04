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
<<<<<<< HEAD
# ──────────────────────────────────────────────
# UCI Heart Disease Dataset columns
=======

# mazhar Done ✅

>>>>>>> 326968237b2b2e543473094d6e409661882dc32d
# ──────────────────────────────────────────────
# 2. EDA (yasmin)
# ──────────────────────────────────────────────

<<<<<<< HEAD
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn theme
sns.set_theme(style="whitegrid")

# 1. Print general information about the data
print("=== 1. General Information About the Data (Info & Shape) ===")
print(df.info())
print("\nShape:", df.shape)
print("-" * 50)

# 2. Print descriptive statistics
print("\n=== 2. Descriptive Statistics (Describe) ===")
print(df.describe())
print("-" * 50)

# 3. Check for missing values
print("\n=== 3. Missing values in each column ===")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])
print("-" * 50)

# 4. Plot the distribution for each column
print("\n=== 4. Plotting feature distributions (Histograms) ===")
df.hist(figsize=(15, 12), bins=15, color='skyblue', edgecolor='black')
plt.suptitle('Features Distribution', fontsize=16)
plt.tight_layout()
plt.show()

# 5. Plot the Correlation Heatmap
print("\n=== 5. Plotting Correlation Heatmap ===")
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.show()

# 6. Distribution of the target variable (healthy vs disease)
print("\n=== 6. Target Variable Distribution ===")
target_counts = df['target'].value_counts()
print("Target Counts:\n", target_counts)

plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette='Set2')
plt.title('Target Variable Distribution', fontsize=14)
plt.xlabel('Target (0 = Healthy, 1 = Disease)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

=======
>>>>>>> 326968237b2b2e543473094d6e409661882dc32d
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

# ============================================================
#  Disease Prediction ML Project
# Models: Logistic Regression + SVM + Random Forest
# Dataset: Heart Disease (Cleveland UCI)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────

import os

base_dir = os.path.dirname(os.path.abspath(__file__))

local_path = os.path.join(base_dir, "data", "heart.csv")

github_urls = [
    "https://raw.githubusercontent.com/Kha1edAyman/Heart-Disease-Prediction-ML-Project/main/data/heart.csv",
    "https://raw.githubusercontent.com/Kha1edAyman/Heart-Disease-Prediction-ML-Project/refs/heads/main/data/heart.csv"
]

df = None

for url in github_urls:
    try:
        df = pd.read_csv(url)
        print(f"✅ Data loaded from GitHub: {url}")
        break
    except:
        continue

if df is None:
    try:
        df = pd.read_csv(local_path)
        print(f"🏠 Loaded from local path: {local_path}")
    except Exception as e:
        print(f"❌ Error: Could not find data anywhere! {e}")

# ──────────────────────────────────────────────
# 2. EDA
# ──────────────────────────────────────────────
print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Target Distribution ===")
print(df['target'].value_counts())

# Histograms
df.hist(figsize=(15, 12), bins=15, color='skyblue', edgecolor='black')
plt.suptitle('Features Distribution', fontsize=16)
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.show()

# Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette='Set2')
plt.title('Target Variable Distribution', fontsize=14)
plt.xlabel('Target (0 = Healthy, 1 = Disease)')
plt.ylabel('Count')
plt.show()

# ──────────────────────────────────────────────
# 3. PREPROCESSING
# ──────────────────────────────────────────────
df = df.dropna()
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train size: {X_train.shape}")
print(f"Test size:  {X_test.shape}")

# ──────────────────────────────────────────────
# 4. MODEL 1: LOGISTIC REGRESSION
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*60)

lr_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear']
}
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    lr_params,
    cv=5,
    scoring='accuracy'
)
lr_grid.fit(X_train_sc, y_train)

lr_best   = lr_grid.best_estimator_
y_pred_lr = lr_best.predict(X_test_sc)
y_prob_lr = lr_best.predict_proba(X_test_sc)[:, 1]

lr_acc = accuracy_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_prob_lr)
lr_cv  = cross_val_score(lr_best, X_train_sc, y_train, cv=5).mean()

print(f"Best Params : {lr_grid.best_params_}")
print(f"Accuracy    : {lr_acc:.4f} ({lr_acc*100:.1f}%)")
print(f"ROC-AUC     : {lr_auc:.4f}")
print(f"CV Score    : {lr_cv:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Healthy', 'Disease']))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_lr),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Disease'],
            yticklabels=['Healthy', 'Disease'])
plt.title('Logistic Regression - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ──────────────────────────────────────────────
# 5. MODEL 2: SVM
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL 2: SUPPORT VECTOR MACHINE (SVM)")
print("="*60)

svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm_grid = GridSearchCV(
    SVC(probability=True),
    svm_params,
    cv=5,
    scoring='accuracy'
)
svm_grid.fit(X_train_sc, y_train)

svm_best   = svm_grid.best_estimator_
y_pred_svm = svm_best.predict(X_test_sc)
y_prob_svm = svm_best.predict_proba(X_test_sc)[:, 1]

svm_acc = accuracy_score(y_test, y_pred_svm)
svm_auc = roc_auc_score(y_test, y_prob_svm)
svm_cv  = cross_val_score(svm_best, X_train_sc, y_train, cv=5).mean()

print(f"Best Params : {svm_grid.best_params_}")
print(f"Accuracy    : {svm_acc:.4f} ({svm_acc*100:.1f}%)")
print(f"ROC-AUC     : {svm_auc:.4f}")
print(f"CV Score    : {svm_cv:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm, target_names=['Healthy', 'Disease']))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_svm),
            annot=True, fmt='d', cmap='Greens',
            xticklabels=['Healthy', 'Disease'],
            yticklabels=['Healthy', 'Disease'])
plt.title('SVM - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ──────────────────────────────────────────────
# 6. MODEL 3: RANDOM FOREST
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL 3: RANDOM FOREST")
print("="*60)

rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)  # Random Forest مش محتاج scaling

rf_best   = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)
y_prob_rf = rf_best.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_prob_rf)
rf_cv  = cross_val_score(rf_best, X_train, y_train, cv=5).mean()

print(f"Best Params : {rf_grid.best_params_}")
print(f"Accuracy    : {rf_acc:.4f} ({rf_acc*100:.1f}%)")
print(f"ROC-AUC     : {rf_auc:.4f}")
print(f"CV Score    : {rf_cv:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Healthy', 'Disease']))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf),
            annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Healthy', 'Disease'],
            yticklabels=['Healthy', 'Disease'])
plt.title('Random Forest - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature Importance
feat_imp = pd.Series(
    rf_best.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title('Random Forest - Feature Importance', fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print("\nTop 5 Important Features:")
print(feat_imp.head())


# ──────────────────────────────────────────────
# 7. MODEL COMPARISON (Nada)
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL MODEL COMPARISON & EVALUATION")
print("="*60)

 
results_dict = {
    'Model': ['Logistic Regression', 'SVM', 'Random Forest'],
    'Accuracy': [lr_acc, svm_acc, rf_acc],
    'ROC-AUC': [lr_auc, svm_auc, rf_auc],
    'CV Score': [lr_cv, svm_cv, rf_cv]
}

results_df = pd.DataFrame(results_dict)

#1. Table
print("\n📊 Model Comparison Table:")
print(results_df.to_string(index=False))

# 2. Bar Chart 
results_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 6))
sns.barplot(x='Metric', y='Score', hue='Model', data=results_melted, palette='viridis')
plt.title('Comparison: Logistic Regression vs SVM vs Random Forest', fontsize=14, fontweight='bold')
plt.ylim(0.7, 1.0)  
plt.ylabel('Score Value')
plt.xlabel('Evaluation Metrics')
plt.legend(title='Machine Learning Models', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3.Winner based on ROC-AUC 
winner = results_df.loc[results_df['ROC-AUC'].idxmax()]
print("\n" + "⭐"*30)
print(f"🏆 THE WINNER MODEL IS: {winner['Model']} 🏆")
print(f"Highest ROC-AUC Score: {winner['ROC-AUC']:.4f}")
print("⭐"*30)

# 4.Dictionary of results
final_results_summary = results_dict


# ──────────────────────────────────────────────
# 8. SAVE RESULTS
# ──────────────────────────────────────────────


print("\n" + "="*60)
print("SAVING RESULTS TO CSV")
print("="*60)

results_df.to_csv("results.csv", index=False)

print("✅ Results saved successfully as 'results.csv'")
print("PROJECT EXECUTION COMPLETED!")
print("="*60)
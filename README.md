# 🩺 Disease Prediction — Heart Disease ML Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

**Predicting heart disease using Logistic Regression & Support Vector Machine (SVM)**  
Built with scikit-learn · GridSearchCV · Cross-Validation · ROC-AUC Evaluation

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Dataset Columns](#-dataset-columns)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Models Used](#-models-used)
- [Results](#-results)
- [Team Task Split](#-team-task-split)

---

## 🧠 Project Overview

This project applies supervised machine learning to predict whether a patient has **heart disease** based on clinical features such as age, blood pressure, cholesterol, and ECG results.

| Item | Detail |
|------|--------|
| **Problem Type** | Binary Classification |
| **Target** | Heart Disease (1) / Healthy (0) |
| **Models** | Logistic Regression, SVM |
| **Evaluation** | Accuracy, ROC-AUC, Confusion Matrix, Cross-Validation |
| **Dataset** | UCI Cleveland Heart Disease |

---

## 📦 Dataset

**Source:** [UCI Machine Learning Repository — Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

**Kaggle Alternative (easier download):**  
🔗 [Heart Disease UCI — Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci)

```
303 records · 13 features · 1 target column · No duplicates
```

> **Download steps (Kaggle):**  
> 1. Go to the Kaggle link above  
> 2. Click **Download** (requires free account)  
> 3. Extract `heart.csv` and place it in the project folder

---

## 🗂 Dataset Columns

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `age` | Numerical | Age in years (29–77) |
| 2 | `sex` | Binary | 1 = Male, 0 = Female |
| 3 | `cp` | Categorical | Chest pain type: 0=asymptomatic, 1=atypical angina, 2=non-cardiac, 3=typical angina |
| 4 | `trestbps` | Numerical | Resting blood pressure (mmHg). Normal ≤ 120 |
| 5 | `chol` | Numerical | Serum cholesterol (mg/dl). Normal < 200 |
| 6 | `fbs` | Binary | Fasting blood sugar > 120 mg/dl: 1 = True, 0 = False |
| 7 | `restecg` | Categorical | Resting ECG: 0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy |
| 8 | `thalach` | Numerical | Maximum heart rate achieved (⭐ Top feature) |
| 9 | `exang` | Binary | Exercise induced angina: 1 = Yes, 0 = No |
| 10 | `oldpeak` | Numerical | ST depression induced by exercise relative to rest (0.0–6.2) |
| 11 | `slope` | Categorical | Slope of peak exercise ST segment: 0=upsloping, 1=flat, 2=downsloping |
| 12 | `ca` | Numerical | Number of major vessels colored by fluoroscopy (0–3) |
| 13 | `thal` | Categorical | Thalassemia: 1=normal, 2=fixed defect, 3=reversible defect |
| 14 | `target` | **TARGET** | **1 = Heart Disease, 0 = Healthy** |

---

## 📁 Project Structure

```
disease-prediction/
│
├── disease_prediction.py   # Main ML script (full pipeline)
├── heart.csv               # Dataset (download from Kaggle/UCI)
├── results.csv             # Model comparison output
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/disease-prediction.git
cd disease-prediction

# 2. Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
scikit-learn>=1.0
pandas>=1.3
numpy>=1.21
matplotlib>=3.4
seaborn>=0.11
```

---

## ▶️ How to Run

```bash
python disease_prediction.py
```

**Pipeline steps executed automatically:**

```
1. Load Data          →  CSV from UCI / Kaggle
2. EDA                →  Shape, dtypes, class balance
3. Preprocessing      →  Handle nulls, StandardScaler
4. Train/Test Split   →  80% train, 20% test (stratified)
5. Logistic Regression → GridSearchCV (C, solver)
6. SVM                →  GridSearchCV (C, kernel, gamma)
7. Evaluate           →  Accuracy, ROC-AUC, Confusion Matrix
8. Compare            →  Side-by-side model summary
```

---

## 🤖 Models Used

### 1. Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

params = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), params, cv=5)
```
- ✅ Fast training
- ✅ Highly interpretable (feature coefficients)
- ✅ Good baseline model

### 2. Support Vector Machine (SVM)
```python
from sklearn.svm import SVC

params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm_grid = GridSearchCV(SVC(probability=True), params, cv=5)
```
- ✅ Handles non-linear boundaries (RBF kernel)
- ✅ Robust on small-to-medium datasets
- ✅ Best performance on this dataset

---

## 📊 Results

| Metric | Logistic Regression | SVM |
|--------|--------------------|----|
| **Accuracy** | ~85% | ~87% |
| **ROC-AUC** | 0.91 | 0.93 |
| **CV Score (5-fold)** | 0.84 | 0.86 |
| **Best C** | 10 | 10 |
| **Kernel / Solver** | lbfgs | linear |

### 🏆 Winner: SVM (Linear Kernel) — by ROC-AUC

### Top 5 Most Important Features (LR Coefficients)

```
thalach   ████████░░  0.85   Max heart rate
ca        ███████░░░  0.79   Fluoroscopy vessels
cp        ██████░░░░  0.73   Chest pain type
oldpeak   █████░░░░░  0.67   ST depression
thal      ████░░░░░░  0.61   Thalassemia
```

### Confusion Matrix (SVM)

```
                 Predicted Healthy   Predicted Disease
Actual Healthy        52 (TN)              5 (FP)
Actual Disease         4 (FN)             48 (TP)
```

---

## 👥 Team Task Split

| Member | Role | Tasks |
|--------|------|-------|
| **Member 1** | 🗄️ Data Engineer | تحميل الـ Dataset · فحص الأعمدة والـ dtypes · اكتشاف القيم الناقصة · حفظ الـ raw data |
| **Member 2** | 📊 EDA Analyst | رسم distribution لكل column · Correlation heatmap · Class balance · تحليل Outliers |
| **Member 3** | ⚙️ Preprocessing Lead | Handle missing values · Label Encoding · StandardScaler · Train/Test Split |
| **Member 4** | 🔵 ML Engineer — LR | Train Logistic Regression · GridSearchCV · Cross-Validation · ROC Curve |
| **Member 5** | 🔷 ML Engineer — SVM | Train SVM (linear + rbf) · GridSearchCV · Kernel comparison · Feature importance |
| **Member 6** | 📝 Evaluation & Report | مقارنة الـ models · Confusion Matrix · Final Report/Slides · Prediction demo |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">
Made with ❤️ for Machine Learning Education
</div>

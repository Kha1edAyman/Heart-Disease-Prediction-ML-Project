# 👥 Task Assignments — Heart Disease Prediction Project

> **Project:** Disease Prediction · Logistic Regression + SVM  
> **Dataset:** Heart Disease (Cleveland UCI)  
> **Team Size:** 6 Members

---

## 🗄️ Mohamed Mazhar — Section 1: LOAD DATA

يا محمد، مهمتك إنك تكمل الـ Section 1 في الكود اللي اسمك عليه.

**المطلوب منك:**

**1. عرّف قائمة الـ columns:**
```python
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]
```

**2. حمّل الداتا:**
```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
df = pd.read_csv(url, header=None, names=columns, na_values='?')
```

**3. اطبع shape وأول 5 صفوف:**
```python
print(df.shape)
print(df.head())
```

**4. سمّي الـ DataFrame اسمه `df`** عشان باقي الفريق يكمل عليه.

---

## 📊 Yasmin — Section 2: EDA

يا ياسمين، مهمتك الـ Section 2 — EDA.

**شغلتك إنك تحللي الداتا قبل ما نبدأ نشتغل عليها:**

**1. معلومات عامة:**
```python
print(df.info())
print(df.describe())
```

**2. القيم الناقصة:**
```python
print(df.isnull().sum())
```

**3. Distribution لكل column:**
```python
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()
```

**4. Correlation Heatmap:**
```python
import seaborn as sns
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

**5. توزيع الـ Target:**
```python
print(df['target'].value_counts())
df['target'].value_counts().plot(kind='bar')
plt.title('Target Distribution')
plt.show()
```

---

## ⚙️ Fatma — Section 3: PREPROCESSING

يا فاطمة، مهمتك الـ Section 3.

**المطلوب منك إنك تجهزي الداتا للموديلات:**

**1. تعاملي مع القيم الناقصة:**
```python
df.dropna(inplace=True)
```

**2. حوّلي الـ Target:**
```python
df['target'] = (df['target'] > 0).astype(int)
# أي قيمة > 0 تبقى 1 (مريض)، 0 تفضل 0 (سليم)
```

**3. افصلي Features عن Target:**
```python
X = df.drop('target', axis=1)
y = df['target']
```

**4. Train/Test Split:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**5. StandardScaler — fit على Train بس:**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

---

## 🔵 Ahmed Amin — Section 4: LOGISTIC REGRESSION

يا أحمد، مهمتك Section 4 — Logistic Regression.

**المطلوب:**

**1. GridSearchCV:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear']
}
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    params, cv=5, scoring='accuracy'
)
lr_grid.fit(X_train_scaled, y_train)
```

**2. Best Params:**
```python
print("Best Params:", lr_grid.best_params_)
print("Best CV Score:", lr_grid.best_score_)
```

**3. Predict وتقييم:**
```python
lr_best   = lr_grid.best_estimator_
y_pred_lr = lr_best.predict(X_test_scaled)

print(accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
```

**4. ROC-AUC + ROC Curve:**
```python
y_prob_lr = lr_best.predict_proba(X_test_scaled)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))

fpr, tpr, _ = roc_curve(y_test, y_prob_lr)
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curve — Logistic Regression')
plt.legend(); plt.show()
```

---

## 🔷 Khaled Ayman — Sections 5 & 7: SVM + SAVE RESULTS

يا خالد، عليك Section 5 (SVM) و Section 7 (Save Results).

### Section 5 — SVM:

**1. GridSearchCV:**
```python
from sklearn.svm import SVC

params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm_grid = GridSearchCV(
    SVC(probability=True),
    params, cv=5, scoring='accuracy'
)
svm_grid.fit(X_train_scaled, y_train)
```

**2. Best Params وتقييم:**
```python
svm_best   = svm_grid.best_estimator_
y_pred_svm = svm_best.predict(X_test_scaled)

print("Best Params:", svm_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
```

**3. Confusion Matrix:**
```python
cm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy','Disease'],
            yticklabels=['Healthy','Disease'])
plt.title('Confusion Matrix — SVM')
plt.show()
```

### Section 7 — Save Results:

بعد ما Nada تخلص المقارنة، احفظ النتائج:
```python
pd.DataFrame(results).to_csv('results.csv', index=False)
print("✅ Results saved to results.csv")
```

---

## 📝 Nada — Section 6: MODEL COMPARISON

يا ندى، مهمتك Section 6 — المقارنة النهائية.

**المطلوب منك:**

**1. جدول مقارنة:**
```python
results = {
    'Model':    ['Logistic Regression', 'SVM'],
    'Accuracy': [accuracy_score(y_test, y_pred_lr),
                 accuracy_score(y_test, y_pred_svm)],
    'ROC-AUC':  [roc_auc_score(y_test, y_prob_lr),
                 roc_auc_score(y_test, y_prob_svm)],
    'CV Score': [cross_val_score(lr_best, X_train_scaled, y_train, cv=5).mean(),
                 cross_val_score(svm_best, X_train_scaled, y_train, cv=5).mean()]
}
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
```

**2. Bar Chart مقارنة:**
```python
df_results.set_index('Model')[['Accuracy', 'ROC-AUC']].plot(
    kind='bar', figsize=(8, 5), colormap='Set2'
)
plt.title('Model Comparison')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.ylim(0.7, 1.0)
plt.tight_layout()
plt.show()
```

**3. إعلان الـ Winner:**
```python
winner = df_results.loc[df_results['ROC-AUC'].idxmax(), 'Model']
print(f"🏆 Best Model by ROC-AUC: {winner}")
```

**4. ⭐ إنتي المسؤولة عن الـ Final Presentation**  
اعملي Slides أو Report تلخص النتائج وتحطي فيه:
- جدول المقارنة
- الرسومات (ROC Curve, Confusion Matrix, Bar Chart)
- الـ Winner وسبب اختياره

---

## ✅ Checklist للـ Leader

| Task | Member | Status |
|------|--------|--------|
| Load Data | Mohamed Mazhar | ⬜ |
| EDA | Yasmin | ⬜ |
| Preprocessing | Fatma | ⬜ |
| Logistic Regression | Ahmed Amin | ⬜ |
| SVM + Save Results | Khaled Ayman | ⬜ |
| Model Comparison + Presentation | Nada | ⬜ |

> بعد ما كل واحد يخلص الجزء بتاعه، يعمل **commit** باسمه على الـ branch بتاعه ثم يعمل **Pull Request** للـ main branch.

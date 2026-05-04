
# 👥 Task Assignments — Heart Disease Prediction Project

> **Project:** Disease Prediction · Logistic Regression + SVM
> **Dataset:** Heart Disease (Cleveland UCI)
> **Team Size:** 6 Members

---

## 🗄️ Mohamed Mazhar — Section 1: LOAD DATA

يا محمد، مهمتك إنك تكمل الـ Section 1 في الكود.

المطلوب منك:

- تعرّف قائمة بأسماء الـ 14 column بالترتيب الصح
- تحمّل الداتا من رابط UCI على الإنترنت مباشرة وتحطها في DataFrame اسمه `df`
- تتعامل مع القيم الناقصة اللي بتيجي على شكل علامة `?` في الملف
- تطبع شكل الداتا وأول 5 صفوف عشان تتأكد إن التحميل اتعمل صح
- **مهم:** الـ DataFrame لازم اسمه `df` عشان باقي الفريق يكمل عليه

---

## 📊 Yasmin — Section 2: EDA

يا ياسمين، مهمتك الـ Section 2 — التحليل الاستكشافي للداتا.

المطلوب منك:

- تطبعي معلومات عامة عن الداتا زي أنواع الأعمدة وعدد الصفوف
- تطبعي الإحصائيات الوصفية لكل column مثل mean و min و max
- تشوفي فين القيم الناقصة وكام عددها في كل column
- ترسمي distribution لكل column عشان نشوف شكل البيانات
- ترسمي Correlation Heatmap عشان نشوف العلاقة بين الـ features ببعض
- تشوفي توزيع الـ target — كام سليم وكام مريض — وترسميه

---

## ⚙️ Fatma — Section 3: PREPROCESSING

يا فاطمة، مهمتك الـ Section 3 — تجهيز الداتا.

المطلوب منك:

- تحذفي الصفوف اللي فيها قيم ناقصة
- تحوّلي الـ target لـ 0 و 1 بس — أي قيمة أكبر من صفر تبقى 1 (يعني مريض)
- تفصلي الـ features عن الـ target في متغيرين `X` و `y`
- تقسمي الداتا 80% للتدريب و 20% للاختبار مع ضمان إن نسبة الـ target متوازنة في الاتنين
- تطبّقي StandardScaler على الداتا — **مهم:** الـ fit بيتعمل على Train بس، مش على Test

---

## 🔵 Ahmed Amin — Section 4: LOGISTIC REGRESSION

يا أحمد، مهمتك Section 4 — بناء موديل Logistic Regression.

المطلوب منك:

- تجرب أكتر من قيمة لـ C وأكتر من solver عشان تلاقي أفضل إعدادات للموديل باستخدام GridSearchCV
- تستخدم 5-Fold Cross Validation في الـ GridSearchCV
- تطبع أفضل باراميترز اتلاقت وأفضل score
- تعمل predict على بيانات الـ Test وتطبع الـ Accuracy وتقرير التصنيف الكامل
- تحسب الـ ROC-AUC Score وترسم الـ ROC Curve
- **مهم:** احتفظ بمتغيرات الـ predictions عشان Nada تستخدمهم في المقارنة

---

## 🔷 Khaled Ayman — Sections 5 & 7: SVM + SAVE RESULTS

يا خالد، عليك Section 5 وكمان Section 7.

**Section 5 — SVM:**

- تجرب أكتر من قيمة لـ C وأكتر من kernel يعني linear و rbf وأكتر من gamma باستخدام GridSearchCV
- تفعّل الـ probability عشان نقدر نحسب الـ ROC-AUC بعدين
- تطبع أفضل باراميترز والـ Accuracy على الـ Test
- ترسم Confusion Matrix واضحة
- **مهم:** احتفظ بمتغيرات الـ predictions عشان Nada تستخدمهم

**Section 7 — Save Results:**

- بعد ما Nada تخلص وتجهز النتائج، احفظها في ملف CSV اسمه `results.csv`

---

## 📝 Nada — Section 6: MODEL COMPARISON + FINAL PRESENTATION

يا ندى، مهمتك Section 6 وإنتي المسؤولة عن الـ Presentation الكاملة.

المطلوب منك:

- تجمعي نتائج الموديلين في جدول واحد فيه Accuracy و ROC-AUC و CV Score لكل موديل
- ترسمي Bar Chart يقارن الموديلين على المقاييس دي جنب بعض
- تحددي الـ Winner بناءً على الـ ROC-AUC وتطبعيه بوضوح
- تجهزي النتائج في dictionary وتدّيه لـ خالد عشان يحفظه في الـ CSV
- **⭐ إنتي المسؤولة عن الـ Final Presentation:** اعملي Slides أو Report تلخص فيه كل النتائج والرسومات وسبب اختيار الـ Winner

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

> بعد ما كل واحد يخلص الجزء بتاعه، يعمل **commit** باسمه على branch بتاعه ثم يعمل **Pull Request** للـ main branch.

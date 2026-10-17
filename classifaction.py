
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. تحميل البيانات
data = load_breast_cancer(as_frame=True)
df = data.frame

# 2. فصل الميزات عن الهدف
X = df.drop(columns="target")
y = df["target"]

# 3. تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. إنشاء المودل
model = RandomForestClassifier(random_state=42)

# 5. تحديد الإعدادات
params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

# 6. إعداد Stratified K-Fold
kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 7. البحث عن أفضل الإعدادات باستخدام K-Fold
grid = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=kfold,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

# 8. عرض أفضل النتائج
best_model = grid.best_estimator_

print("Best parameters:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

# 9. التقييم النهائي على بيانات الاختبار
y_pred = best_model.predict(X_test)

print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(
    y_test, y_pred, target_names=data.target_names
))

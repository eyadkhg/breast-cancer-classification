import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------- 1) تحميل وفحص ----------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()

print("Shape قبل التنظيف:", df.shape)
print("Dtypes غير رقمية:", df.select_dtypes(exclude="number").columns.tolist())

# ---------- 2) تنظيف على مستوى الصفوف (آمن قبل التقسيم) ----------
# حذف الصفوف المكررة
n_dup = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print("صفوف مكررة محذوفة:", n_dup)

# حذف الصفوف اللي target فيها مفقود (ما نقدر نتدرب عليها)
df = df.dropna(subset=["target"])

# حذف الأعمدة الثابتة (ما فيها معلومة)
const_cols = [c for c in df.columns if df[c].nunique() <= 1]
df = df.drop(columns=const_cols)
print("أعمدة ثابتة محذوفة:", const_cols)

# القيم المستحيلة: كل خصائص هذي الداتا لازم تكون >= 0
feature_cols = df.columns.drop("target")
df[feature_cols] = df[feature_cols].mask(df[feature_cols] < 0)  # تتحول NaN ويعالجها الـ imputer

print("Shape بعد التنظيف:", df.shape)
print("Missing values:", df.isna().sum().sum())

# تقرير القيم الشاذة (للمعاينة فقط — ما نحذفها)
q1, q3 = df[feature_cols].quantile(0.25), df[feature_cols].quantile(0.75)
iqr = q3 - q1
outliers = ((df[feature_cols] < q1 - 3 * iqr) | (df[feature_cols] > q3 + 3 * iqr)).sum()
print("أكثر الأعمدة فيها قيم شاذة:\n", outliers.nlargest(5))

X = df.drop(columns="target")
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 3) تنظيف يتعلم من التدريب فقط (داخل الـ Pipeline) ----------
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """يحذف الأعمدة اللي ترابطها مع عمود سابق أعلى من threshold."""
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.drop_idx_ = [i for i, col in enumerate(upper.columns)
                          if (upper[col] > self.threshold).any()]
        self.keep_idx_ = [i for i in range(X.shape[1]) if i not in self.drop_idx_]
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.keep_idx_]

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("corr_filter", CorrelationFilter(threshold=0.95)),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42)),
])

params = {
    "corr_filter__threshold": [0.90, 0.95, 1.0],   # 1.0 = ما يحذف شي
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_split": [2, 5],
    "clf__class_weight": [None, "balanced"],
}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(pipe, params, cv=kfold, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print("\nBest parameters:", grid.best_params_)
print("Best CV ROC-AUC:", round(grid.best_score_, 4))
print("عدد الأعمدة المحتفظ بها:", len(best_model.named_steps["corr_filter"].keep_idx_), "من", X.shape[1])

# ---------- 4) التقييم (malignant = 0 هي الفئة المهمة) ----------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nRecall (malignant)   :", round(recall_score(y_test, y_pred, pos_label=0), 4))
print("Precision (malignant):", round(precision_score(y_test, y_pred, pos_label=0), 4))
print("ROC-AUC              :", round(roc_auc_score(y_test, y_proba), 4))
print(classification_report(y_test, y_pred, target_names=data.target_names))
# ===============================
# 📌 FINAL: REGRESSOR + CLASSIFIER FOR ANOMALY
# ===============================

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, mean_squared_error, r2_score
from xgboost import XGBRegressor, XGBClassifier

# -------------------------------
# 1️⃣ Load Data
# -------------------------------
df = pd.read_csv('insurance.csv')

# -------------------------------
# 2️⃣ Encode Categorical
# -------------------------------
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

# -------------------------------
# 3️⃣ Add Contextual Feature
# -------------------------------
df['bmi_age_interaction'] = df['bmi'] * df['age']

features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'bmi_age_interaction']
X = df[features]
y = df['charges']

# -------------------------------
# 4️⃣ Log transform target
# -------------------------------
y_log = np.log1p(y)

# -------------------------------
# 5️⃣ Scale features
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 6️⃣ Train regressor
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_log, test_size=0.2, random_state=42)

xgb_reg = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
xgb_reg.fit(X_train, y_train)

y_pred_log = xgb_reg.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

# -------------------------------
# 7️⃣ Add regressor output back
# -------------------------------
df['predicted'] = np.expm1(xgb_reg.predict(scaler.transform(X)))
df['residual'] = df['charges'] - df['predicted']
df['ratio'] = df['charges'] / (df['predicted'] + 1)

# Label: top 5% = anomaly
threshold = np.percentile(df['charges'], 95)
df['anomaly'] = df['charges'].apply(lambda x: 1 if x >= threshold else 0)

# Final classifier features
features_cls = features + ['predicted', 'residual', 'ratio']
X_cls = df[features_cls]
y_cls = df['anomaly']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

xgb_cls = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, scale_pos_weight=10, use_label_encoder=False, eval_metric='logloss')
xgb_cls.fit(X_train_c, y_train_c)

# -------------------------------
# 🔟 Evaluate classifier
# -------------------------------
y_pred_cls = xgb_cls.predict(X_test_c)
y_proba_cls = xgb_cls.predict_proba(X_test_c)[:,1]

cm = confusion_matrix(y_test_c, y_pred_cls)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal','Anomaly'])
fig, ax = plt.subplots(figsize=(5,5))
disp.plot(ax=ax, cmap='Reds', values_format='d')
plt.title('Confusion Matrix - Final Classifier')
plt.show()

print("Classification Report:\n")
print(classification_report(y_test_c, y_pred_cls, digits=4))

fpr, tpr, thresholds = roc_curve(y_test_c, y_proba_cls)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'--', color='navy')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Final Classifier')
plt.legend(loc="lower right")
plt.show()

# -------------------------------
# ✅ Save regressor + classifier + scaler
# -------------------------------
joblib.dump(xgb_reg, 'model_reg.pkl')
joblib.dump(xgb_cls, 'model_cls.pkl')
joblib.dump(scaler, 'scaler_reg.pkl')

print("✅ Final regressor, classifier & scaler saved: model_reg.pkl, model_cls.pkl, scaler_reg.pkl")

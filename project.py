# 1) IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
import os


# 2) LOAD DATA

df = pd.read_csv("/content/high_accuracy_credit_data.csv")
print("Loaded data:", df.shape)
print(df.head())

target = "target_default"
X = df.drop(columns=[target])
y = df[target]


# 3) TRAIN–TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

numeric_features = X.columns.tolist()

# 4) PREPROCESSING PIPELINE
numeric_transformer = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features)]
)

# 5) MODEL
model = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# 6) TRAIN MODEL
pipeline.fit(X_train, y_train)
print("Model training completed!")

# 7) PREDICT
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# 8) METRICS
print("\n--- MODEL PERFORMANCE ---")
print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"AUPRC: {average_precision_score(y_test, y_proba):.4f}")
print(f"Brier Score: {brier_score_loss(y_test, y_proba):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9) PLOTS (ROC + PR + CALIBRATION)
# ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label="Model ROC")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.grid()
plt.show()

# Precision–Recall
prec, rec, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()

# 10) SAVE MODEL
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/credit_risk_pipeline.joblib")
print("\nModel saved to models/credit_risk_pipeline.joblib")

# 11) SAVE PREDICTION CSV
pred_df = X_test.copy()
pred_df["y_true"] = y_test.values
pred_df["y_pred"] = y_pred
pred_df["y_proba"] = y_proba

pred_df.to_csv("credit_risk_predictions.csv", index=False)
print("Predictions saved as credit_risk_predictions.csv")

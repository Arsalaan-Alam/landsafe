import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, auc
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data.csv', low_memory=False)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Select features from previous 15 days
columns = []
for i in range(15, -1, -1):
    columns += [f'precip{i}', f'temp{i}', f'air{i}', f'humidity{i}', f'wind{i}']

# Add ARI0 to ARI9 (since only these exist)
for i in range(9, -1, -1):
    columns.append(f'ARI{i}')

# Add static features
columns += ['forest', 'slope', 'osm', 'lithology']

# Prepare feature matrix X and target vector y
X = df[columns].copy()
y = df['landslide']

# Encode categorical variable
X['lithology'] = OrdinalEncoder().fit_transform(X[['lithology']])

# Handle missing or infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'alpha': 0.1,
    'lambda': 1.0,
    'seed': 42
}

# Train model
evals_result = {}
watchlist = [(dtrain, 'train'), (dtest, 'eval')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=watchlist,
    early_stopping_rounds=20,
    evals_result=evals_result,
    verbose_eval=10
)

# Predictions
y_pred_prob = model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)

# Training metrics
train_preds = (model.predict(dtrain) > 0.5).astype(int)
train_accuracy = accuracy_score(y_train, train_preds)
test_accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = auc(*roc_curve(y_test, y_pred_prob)[:2])

# Print evaluation results
print("\n========================= MODEL EVALUATION =========================")
print(f"Train Accuracy : {train_accuracy:.4f}")
print(f"Test Accuracy  : {test_accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"ROC AUC        : {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"[[{cm[0,0]} {cm[0,1]}] <- [True Negatives, False Positives]]")
print(f"[[{cm[1,0]} {cm[1,1]}] <- [False Negatives, True Positives]]")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Training Log Loss
plt.figure(figsize=(8, 6))
epochs = len(evals_result['train']['logloss'])
plt.plot(range(epochs), evals_result['train']['logloss'], label='Train Log Loss')
plt.plot(range(epochs), evals_result['eval']['logloss'], label='Validation Log Loss')
plt.xlabel("Boosting Rounds")
plt.ylabel("Log Loss")
plt.title("XGBoost Training Performance")
plt.legend()
plt.grid()
plt.show()

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

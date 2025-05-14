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
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization, LeakyReLU

# Load data
df = pd.read_csv('data.csv', low_memory=False)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Feature selection: Use past 7 days (0 to 6) and static features
columns = []
for i in range(6, -1, -1):
    columns += [f'precip{i}', f'temp{i}', f'air{i}', f'humidity{i}', f'wind{i}']
for i in range(6, -1, -1):
    columns.append(f'ARI{i}')
columns += ['forest', 'slope', 'osm', 'lithology']

X = df[columns].copy()
y = df['landslide']

# Handle categorical feature
X['lithology'] = OrdinalEncoder().fit_transform(X[['lithology']])

# Handle any potential NaNs or infs
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Prepare data for LSTM
# We'll use only the time-series features for LSTM input (7 timesteps, 2 features: precip and temp)
# You can adjust the features as needed
def get_lstm_features(X):
    # Time-series features
    features = []
    for i in range(6, -1, -1):
        features.extend([f'precip{i}', f'temp{i}', f'air{i}', f'humidity{i}', f'wind{i}', f'ARI{i}'])
    
    # Static features
    static_features = ['forest', 'slope', 'osm', 'lithology']
    
    # Get time-series data
    time_series_data = X[features].values
    # Get static data and repeat for each timestep
    static_data = np.repeat(X[static_features].values[:, np.newaxis, :], 7, axis=1)
    
    # Combine time-series and static features
    combined_data = np.concatenate([
        time_series_data.reshape(len(X), 7, 6),  # 6 time-series features
        static_data  # 4 static features
    ], axis=2)
    
    return combined_data

X_train_lstm = get_lstm_features(pd.DataFrame(X_train_scaled, columns=X_train.columns))
X_test_lstm = get_lstm_features(pd.DataFrame(X_test_scaled, columns=X_test.columns))

# Now each timestep has 10 features (6 time-series + 4 static)
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(7, 10)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Bidirectional(LSTM(32)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(32))
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Dense(16))
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

# Train model
history = model.fit(X_train_lstm, y_train, epochs=25, batch_size=64, verbose=2, validation_data=(X_test_lstm, y_test))

# Predict
y_pred_prob = model.predict(X_test_lstm).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

# Training metrics
train_preds = (model.predict(X_train_lstm).flatten() > 0.5).astype(int)
train_accuracy = accuracy_score(y_train, train_preds)
test_accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = auc(*roc_curve(y_test, y_pred_prob)[:2])

# Output
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

# Training Loss Plot
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Binary Crossentropy Loss")
plt.title("LSTM Training Performance")
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


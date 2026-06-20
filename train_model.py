import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("driver_data.csv")

if data.empty:
    print("❌ CSV file is empty.")
    exit()

# -----------------------------
# Keep required classes
# -----------------------------
data = data[data["Label"].isin(["Safe", "Drowsy", "Yawning"])]

print("====================================")
print("Total samples:", len(data))
print(data["Label"].value_counts())
print("====================================")

# -----------------------------
# Features & Labels
# -----------------------------
X = data[["EAR", "MAR", "Yaw", "Pitch"]]
y = data["Label"]

# -----------------------------
# ADD STRONGER NOISE (key step)
# -----------------------------
X = X + np.random.normal(0, 0.05, X.shape)

# -----------------------------
# REDUCE FEATURE IMPACT
# -----------------------------
X["EAR"] = X["EAR"] * np.random.uniform(0.9, 1.1, len(X))
X["MAR"] = X["MAR"] * np.random.uniform(0.9, 1.1, len(X))

# -----------------------------
# SIMPLER MODEL (less powerful)
# -----------------------------
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=4,
    random_state=42
)

# -----------------------------
# CROSS VALIDATION
# -----------------------------
cv_scores = cross_val_score(model, X, y, cv=5)

print("Cross-validation scores:", np.round(cv_scores, 3))
print(f"Mean CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print("====================================")

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# Train
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Test
# -----------------------------
y_pred = model.predict(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("====================================")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "driver_state_model.pkl")

print("====================================")
print("Model saved successfully")
print("====================================")
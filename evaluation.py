import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("driver_data.csv")  # Make sure file is in same folder

X = data[["EAR", "MAR", "Pitch", "Yaw"]]
y = data["Label"]

# Class labels (for display)
labels = ["Alert", "Drowsy", "Yawning"]

# -----------------------------
# 2. Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Train Model
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 4. Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 5. Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")

# -----------------------------
# 6. Confusion Matrix (Improved)
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='viridis', values_format='d')

plt.title("Confusion Matrix - Driver Drowsiness Detection")
plt.xlabel("Predicted Driver State")
plt.ylabel("Actual Driver State")

plt.show()

# -----------------------------
# 7. Classification Report
# -----------------------------
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=labels))

# -----------------------------
# 8. Learning Curve
# -----------------------------
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure()

plt.plot(train_sizes, train_mean, marker='o', linewidth=2, label="Training Accuracy")
plt.plot(train_sizes, test_mean, marker='s', linewidth=2, label="Validation Accuracy")

plt.xlabel("Training Data Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve - Model Performance")
plt.legend()
plt.grid()

plt.show()
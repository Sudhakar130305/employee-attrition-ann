import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from neural_network import NeuralNetwork
from losses import binary_cross_entropy
from visualize import plot_loss


# =====================================================
# Start Timer
# =====================================================

start_time = time.time()


# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv(
    "../data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

print(f"Dataset Shape: {df.shape}")


# =====================================================
# Drop Unnecessary Columns
# =====================================================

df.drop(
    [
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours"
    ],
    axis=1,
    inplace=True
)


# =====================================================
# Encode Target Variable
# =====================================================

df["Attrition"] = df["Attrition"].map(
    {
        "Yes": 1,
        "No": 0
    }
)


# =====================================================
# One Hot Encoding
# =====================================================

df = pd.get_dummies(
    df,
    drop_first=True
)


# =====================================================
# Features and Labels
# =====================================================

X = df.drop(
    "Attrition",
    axis=1
).values

y = df["Attrition"].values.reshape(
    -1,
    1
)


# =====================================================
# Train Validation Test Split
# =====================================================

# 70% Train
# 15% Validation
# 15% Test

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)
# =====================================================
# Feature Scaling
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)

X_test = scaler.transform(X_test)


# =====================================================
# Create Neural Network
# =====================================================

nn = NeuralNetwork(
    input_size=X_train.shape[1],
    hidden_size=16,
    output_size=1
)


# =====================================================
# Training Parameters
# =====================================================

epochs = 1000
learning_rate = 0.01

loss_history = []
val_loss_history = []

best_val_loss = float("inf")

patience = 50
counter = 0


# =====================================================
# Training Loop
# =====================================================

for epoch in range(epochs):

    train_output = nn.forward(X_train)

    train_loss = binary_cross_entropy(
        y_train,
        train_output
    )

    nn.backward(
        X_train,
        y_train,
        learning_rate
    )

    val_output = nn.forward(X_val)

    val_loss = binary_cross_entropy(
        y_val,
        val_output
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        counter = 0
        nn.save_weights()

    else:

        counter += 1

    if counter >= patience:

        print(
            f"\nEarly stopping triggered at epoch {epoch}"
        )

        break

    if epoch % 100 == 0:

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )

# =====================================================
# Load Best Model
# =====================================================

nn.load_weights()


# =====================================================
# Prediction
# =====================================================

probabilities = nn.forward(X_test)

predictions = (
    probabilities > 0.5
).astype(int)


# =====================================================
# Evaluation Metrics
# =====================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

auc = roc_auc_score(
    y_test,
    probabilities
)

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n==========================")
print("MODEL PERFORMANCE")
print("==========================")

print(
    f"Accuracy : {accuracy*100:.2f}%"
)

print(
    f"ROC AUC  : {auc:.4f}"
)

print("\nConfusion Matrix:\n")
print(cm)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)


# =====================================================
# Training Time
# =====================================================

end_time = time.time()

print(
    f"\nTraining Time: "
    f"{end_time-start_time:.2f} seconds"
)


# =====================================================
# Plot Training Loss
# =====================================================

plot_loss(
    loss_history,
    val_loss_history
)
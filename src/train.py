import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from neural_network import NeuralNetwork
from losses import binary_cross_entropy
from visualize import plot_loss


# ==========================
# Load Dataset
# ==========================

df = pd.read_csv(
    "../data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

print("Dataset Shape:", df.shape)


# ==========================
# Drop unnecessary columns
# ==========================

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


# ==========================
# Convert Target Variable
# ==========================

df["Attrition"] = df["Attrition"].map(
    {
        "Yes": 1,
        "No": 0
    }
)


# ==========================
# One Hot Encoding
# ==========================

df = pd.get_dummies(
    df,
    drop_first=True
)


# ==========================
# Features and Labels
# ==========================

X = df.drop(
    "Attrition",
    axis=1
).values

y = df["Attrition"].values.reshape(
    -1,
    1
)


# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================
# Create Neural Network
# ==========================

nn = NeuralNetwork(
    input_size=X_train.shape[1],
    hidden_size=16,
    output_size=1
)


# ==========================
# Training Parameters
# ==========================

epochs = 1000
learning_rate = 0.01

loss_history = []


# ==========================
# Training Loop
# ==========================

for epoch in range(epochs):

    # Forward Propagation
    output = nn.forward(X_train)

    # Calculate Loss
    loss = binary_cross_entropy(
        y_train,
        output
    )

    loss_history.append(loss)

    # Backpropagation + Gradient Descent
    nn.backward(
        X_train,
        y_train,
        learning_rate
    )

    # Print Progress
    if epoch % 100 == 0:
        print(
            f"Epoch {epoch} | Loss: {loss:.4f}"
        )


# ==========================
# Prediction
# ==========================

predictions = nn.forward(X_test)

predictions = (
    predictions > 0.5
).astype(int)


# ==========================
# Evaluation Metrics
# ==========================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==========================")
print("MODEL PERFORMANCE")
print("==========================")

print(
    f"Accuracy: {accuracy*100:.2f}%"
)

print("\nConfusion Matrix:\n")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        predictions
    )
)


# ==========================
# Plot Loss Curve
# ==========================

plot_loss(loss_history)

    
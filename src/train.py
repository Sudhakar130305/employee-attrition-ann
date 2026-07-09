import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from losses import binary_cross_entropy
from neural_network import NeuralNetwork

# Load Dataset

df = pd.read_csv("../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Remove unnecessary columns

df.drop(["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"], axis=1, inplace=True)

# Convert target

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

# One Hot Encoding

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Attrition", axis=1).values
y = df["Attrition"].values.reshape(-1, 1)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Model

nn = NeuralNetwork(

    input_size=X_train.shape[1],

    hidden_size=16,

    output_size=1

)

epochs = 1000
learning_rate = 0.01

for epoch in range(epochs):

    # Forward Pass
    output = nn.forward(X_train)

    # Compute Loss
    loss = binary_cross_entropy(
        y_train,
        output
    )

    # Backward Pass + Weight Update
    nn.backward(
        X_train,
        y_train,
        learning_rate
    )

    # Print progress
    if epoch % 100 == 0:
        print(
            f"Epoch {epoch} Loss: {loss:.4f}"
        )
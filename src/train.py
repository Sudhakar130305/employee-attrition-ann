import pandas as pd
from neural_network import NeuralNetwork

# Load preprocessed data
X = pd.read_csv("../data/X_train.csv").values
y = pd.read_csv("../data/y_train.csv").values

print("X Shape:", X.shape)
print("y Shape:", y.shape)
print("Data Type:", X.dtype)

network = NeuralNetwork(X.shape[1])

predictions = network.forward(X)

print("\nFirst 10 Predictions:\n")
print(predictions[:10])
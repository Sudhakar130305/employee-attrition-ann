import numpy as np
from activations import Activation


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):

        np.random.seed(42)

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):

        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.sigmoid(self.z2)

        return self.output
    
    def backward(self, X, y, learning_rate):

        m = X.shape[0]

        # Output layer error
        dz2 = self.output - y

        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer error
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)

        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Gradient Descent Update
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    def save_weights(self):
        np.save("W1.npy", self.W1)
        np.save("b1.npy", self.b1)
        np.save("W2.npy", self.W2)
        np.save("b2.npy", self.b2)


    def load_weights(self):
        self.W1 = np.load("W1.npy")
        self.b1 = np.load("b1.npy")
        self.W2 = np.load("W2.npy")
        self.b2 = np.load("b2.npy")
import numpy as np
from activations import sigmoid

class NeuralNetwork:

    def __init__(self, input_size):

        np.random.seed(42)

        self.weights = np.random.randn(input_size, 1) * 0.01
        self.bias = np.zeros((1, 1))

    def forward(self, X):

        self.z = np.dot(X, self.weights) + self.bias

        self.output = sigmoid(self.z)

        return self.output
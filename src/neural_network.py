import numpy as np
from activations import Activation


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):

        np.random.seed(42)

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):

        # Input -> Hidden

        self.Z1 = np.dot(X, self.W1) + self.b1

        self.A1 = Activation.relu(self.Z1)

        # Hidden -> Output

        self.Z2 = np.dot(self.A1, self.W2) + self.b2

        self.A2 = Activation.sigmoid(self.Z2)

        return self.A2
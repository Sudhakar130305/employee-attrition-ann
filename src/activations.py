import numpy as np

class Activation:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid_derivative(x):
        s = Activation.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

        
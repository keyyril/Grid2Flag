"""Activation functions."""

import numpy as np


class ReLU:
    """Rectified Linear Unit activation: max(0, x)"""
    
    def forward(self, Z):
        """ReLU forward: max(0, Z)"""
        self.Z = Z
        return np.maximum(0, Z)
    
    def backward(self, dA):
        """ReLU backward: gradient is 1 if Z > 0, else 0"""
        return dA * (self.Z > 0)


class Sigmoid:
    """Sigmoid activation: 1 / (1 + e^-x)"""
    
    def forward(self, Z):
        """Sigmoid forward with numerical stability"""
        self.A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        return self.A
    
    def backward(self, dA):
        """Sigmoid backward: dZ = dA * A * (1 - A)"""
        return dA * self.A * (1 - self.A)
"""Dense layer implementation."""

import numpy as np


class Dense:
    """Fully connected (dense) layer."""
    
    def __init__(self, input_size, output_size, learning_rate=0.01):
        """
        Initialize weights and biases.
        
        Args:
            input_size: Number of input features
            output_size: Number of neurons in this layer
            learning_rate: Step size for gradient descent
        """
        # He initialization for weights
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        
        # Cache for backward pass
        self.X = None
    
    def forward(self, X):
        """
        Forward pass: Z = X @ W + b
        
        Args:
            X: Input array, shape (batch_size, input_size)
            
        Returns:
            Z: Pre-activation output, shape (batch_size, output_size)
        """
        self.X = X
        Z = X @ self.W + self.b
        return Z
    
    def backward(self, dZ):
        """
        Backward pass: compute gradients and update weights.
        
        Args:
            dZ: Gradient from next layer, shape (batch_size, output_size)
            
        Returns:
            dX: Gradient w.r.t. input, shape (batch_size, input_size)
        """
        m = self.X.shape[0]  # batch size
        
        # Compute gradients
        dW = (self.X.T @ dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = dZ @ self.W.T
        
        # Update weights using gradient descent
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db
        
        return dX
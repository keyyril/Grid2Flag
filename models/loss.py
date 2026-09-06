"""Loss function for binary classification."""

import numpy as np


class BinaryCrossEntropy:
    """Binary cross-entropy loss function."""
    
    @staticmethod
    def forward(y_pred, y_true):
        """
        Compute BCE loss.
        Loss = -1/m * sum(y * log(y_pred) + (1-y) * log(1-y_pred))
        
        Args:
            y_pred: Predicted probabilities, shape (m, 1), values in [0, 1]
            y_true: True labels, shape (m, 1), values in {0, 1}
            
        Returns:
            Scalar loss value
        """
        m = y_true.shape[0]
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    @staticmethod
    def backward(y_pred, y_true):
        """
        Compute gradient of BCE w.r.t. predictions.
        dL/dy_pred = -(y/y_pred - (1-y)/(1-y_pred)) / m
        
        Args:
            y_pred: Predicted probabilities, shape (m, 1)
            y_true: True labels, shape (m, 1)
            
        Returns:
            Gradient, shape (m, 1)
        """
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        dL = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / m
        return dL
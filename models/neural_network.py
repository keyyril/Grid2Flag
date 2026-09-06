"""Main neural network class."""

import numpy as np
from .layers import Dense
from .activations import ReLU, Sigmoid
from .loss import BinaryCrossEntropy


class NeuralNetwork:
    """Simple feed-forward neural network for binary classification."""
    
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Initialize neural network.
        
        Args:
            layer_sizes: List of layer sizes. E.g., [6, 16, 8, 1] means:
                         input: 6 features -> hidden: 16 neurons -> hidden: 8 -> output: 1
            learning_rate: Learning rate for all layers
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.layers = []
        self.activations = []
        
        # Build layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i + 1], learning_rate))
            
            # Hidden layers use ReLU, output layer uses Sigmoid
            if i < len(layer_sizes) - 2:
                self.activations.append(ReLU())
            else:
                self.activations.append(Sigmoid())
        
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, X):
        """
        Forward pass through all layers.
        
        Args:
            X: Input features, shape (batch_size, input_size)
            
        Returns:
            Output predictions, shape (batch_size, 1)
        """
        self.layer_outputs = []  # Store Z values for backward pass
        self.activation_outputs = [X]  # Store A values for backward pass
        
        A = X
        for i, layer in enumerate(self.layers):
            Z = layer.forward(A)
            self.layer_outputs.append(Z)
            A = self.activations[i].forward(Z)
            self.activation_outputs.append(A)
        
        return A
    
    def backward(self, y_true):
        """
        Backward pass through all layers (backpropagation).
        
        Args:
            y_true: True labels, shape (batch_size, 1)
        """
        # Start with gradient from loss function
        dA = BinaryCrossEntropy.backward(self.activation_outputs[-1], y_true)
        
        # Backpropagate through each layer
        for i in reversed(range(len(self.layers))):
            # Gradient through activation function
            dZ = self.activations[i].backward(dA)
            
            # Gradient through dense layer (also updates weights)
            dA = self.layers[i].backward(dZ)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=True):
        """
        Train the neural network.
        
        Args:
            X_train: Training features, shape (n_train, n_features)
            y_train: Training labels, shape (n_train, 1)
            X_val: Validation features, shape (n_val, n_features)
            y_val: Validation labels, shape (n_val, 1)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            verbose: Print loss every 10 epochs
            
        Returns:
            Dictionary with training history
        """
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(epochs):
            # Mini-batch training
            n_batches = int(np.ceil(X_train.shape[0] / batch_size))
            
            for batch in range(n_batches):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, X_train.shape[0])
                
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]
                
                # Forward and backward pass
                y_pred = self.forward(X_batch)
                self.backward(y_batch)
            
            # Compute losses on full datasets
            y_train_pred = self.forward(X_train)
            train_loss = BinaryCrossEntropy.forward(y_train_pred, y_train)
            
            y_val_pred = self.forward(X_val)
            val_loss = BinaryCrossEntropy.forward(y_val_pred, y_val)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1:3d}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def predict(self, X):
        """
        Make probability predictions.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities, shape (n_samples, 1)
        """
        return self.forward(X)
    
    def predict_binary(self, X, threshold=0.5):
        """
        Make binary predictions (0 or 1).
        
        Args:
            X: Features, shape (n_samples, n_features)
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Binary predictions, shape (n_samples, 1)
        """
        probas = self.predict(X)
        return (probas >= threshold).astype(int)
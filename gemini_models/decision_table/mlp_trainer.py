#!/usr/bin/env python3

"""
mlp_trainer.py

A Multi-Layer Perceptron (MLP) implementation in Python using NumPy.
This script handles data preprocessing, training, and saving/loading of the model.
It is designed to train a large MLP with approximately 2 million parameters.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class MLP:
    """A simple Multi-Layer Perceptron for binary classification."""

    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # Weights: (neurons_in_current_layer, neurons_in_previous_layer)
            self.weights.append(np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.01)
            # Biases: (neurons_in_current_layer, 1)
            self.biases.append(np.zeros((layer_sizes[i+1], 1)))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        s = self._sigmoid(x)
        return s * (1 - s)

    def forward(self, X):
        activations = [X.T] # Input layer activation (transposed to be column vector)
        zs = [] # Weighted sums before activation

        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            zs.append(z)
            a = self._sigmoid(z)
            activations.append(a)
        return activations, zs

    def backward(self, X, y, activations, zs):
        # y needs to be a column vector
        y = y.reshape(-1, 1)

        # Output layer error
        delta = (activations[-1] - y) * self._sigmoid_derivative(zs[-1])
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Backpropagate through hidden layers
        for l in range(2, len(self.layer_sizes)):
            z = zs[-l]
            sp = self._sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)

        return nabla_w, nabla_b

    def train(self, X_train, y_train, epochs):
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Mini-batch or full-batch gradient descent (here, full-batch for simplicity)
            activations, zs = self.forward(X_train)
            nabla_w, nabla_b = self.backward(X_train, y_train, activations, zs)

            for i in range(len(self.weights)):
                self.weights[i] -= (self.learning_rate / num_samples) * nabla_w[i]
                self.biases[i] -= (self.learning_rate / num_samples) * nabla_b[i]

    def predict_proba(self, X):
        activations, _ = self.forward(X)
        return activations[-1].T # Transpose back to (num_samples, num_outputs)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

def preprocess_diabetic_data(df):
    # Drop columns with too many missing values or irrelevant for this task
    df = df.drop(columns=['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr'], errors='ignore')

    # Handle missing values by filling with 'UNKNOWN' for categorical or median for numerical
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace('?', 'UNKNOWN')
        else:
            df[col] = df[col].fillna(df[col].median())

    # Define categorical and numerical features
    categorical_features = df.select_dtypes(include='object').columns.tolist()
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()

    # Remove the target variable from features
    if 'readmitted' in categorical_features:
        categorical_features.remove('readmitted')
    if 'readmitted' in numerical_features:
        numerical_features.remove('readmitted')

    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Target variable transformation
    # 'NO' -> 0, '>30' or '<30' -> 1
    df['readmitted'] = df['readmitted'].map({'NO': 0, '>30': 1, '<30': 1})

    X = df.drop('readmitted', axis=1)
    y = df['readmitted']

    return X, y, preprocessor

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        print("Usage: python3 mlp_trainer.py <data_file.csv> <model_output.pkl> <learning_rate> <epochs>")
        sys.exit(1)

    data_file = sys.argv[1]
    model_file = sys.argv[2]
    learning_rate = float(sys.argv[3])
    epochs = int(sys.argv[4])

    print("Loading and preprocessing data...")
    df = pd.read_csv(data_file)
    X, y, preprocessor = preprocess_diabetic_data(df.copy())

    # Fit preprocessor on the entire dataset before splitting to avoid data leakage
    X_processed = preprocessor.fit_transform(X)

    # Get the actual number of input features after one-hot encoding
    input_features_count = X_processed.shape[1]

    # Define layer sizes based on the calculated input features and target 2M parameters
    # Input: input_features_count
    # Hidden1: 2000
    # Hidden2: 1000
    # Output: 2
    layer_sizes = [input_features_count, 2000, 1000, 2]

    print(f"Training MLP with architecture: {layer_sizes} and {epochs} epochs...")
    mlp = MLP(layer_sizes, learning_rate=learning_rate)
    mlp.train(X_processed, y, epochs)

    print(f"Saving model to {model_file}...")
    mlp.save_model(model_file)

    total_params = 0
    for i in range(len(layer_sizes) - 1):
        total_params += (layer_sizes[i+1] * layer_sizes[i]) + layer_sizes[i+1]
    print(f"Model training complete. Total parameters: {total_params}")


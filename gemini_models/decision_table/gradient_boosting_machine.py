#!/usr/bin/env python3

"""
_gradient_boosting_machine.py_

A from-scratch implementation of a Gradient Boosting Machine (GBM) for regression.
This implementation uses Decision Trees as the base learners.

This script is designed to handle tabular data with both numerical and categorical features.
It includes data preprocessing, model training, and the ability to save/load the model.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class GradientBoostingMachine:
    """A Gradient Boosting Machine for regression tasks."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """Initialize the GBM.

        Args:
            n_estimators (int): The number of boosting stages to perform.
            learning_rate (float): Shrinks the contribution of each tree.
            max_depth (int): The maximum depth of the individual regression estimators.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        """Fit the GBM model.

        Args:
            X (pd.DataFrame): The training input samples.
            y (pd.Series): The target values.
        """
        # Start with an initial prediction (the mean of the target values).
        self.initial_prediction = np.mean(y)
        residuals = y - self.initial_prediction

        for _ in range(self.n_estimators):
            # Create and fit a new decision tree on the residuals.
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update the residuals with the predictions from the new tree.
            prediction = tree.predict(X)
            residuals -= self.learning_rate * prediction

    def predict(self, X):
        """Predict regression target for X.

        Args:
            X (pd.DataFrame): The input samples.

        Returns:
            np.array: The predicted values.
        """
        # Start with the initial prediction.
        predictions = np.full(X.shape[0], self.initial_prediction)

        # Add the predictions from each tree.
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return predictions

    def save_model(self, filename):
        """Save the trained model to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        """Load a model from a file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

def preprocess_data(df):
    """Preprocess the enterprise survey data."""
    # For simplicity, we'll focus on a subset of variables.
    df = df[df['variable'].isin(['Salaries and wages paid', 'Total income', 'Total expenditure', 'Operating profit before tax', 'Total assets'])]

    # Convert categorical columns to numerical using Label Encoding.
    categorical_cols = ['industry_code_ANZSIC', 'industry_name_ANZSIC', 'rme_size_grp', 'variable', 'unit']
    encoders = {col: LabelEncoder() for col in categorical_cols}
    for col, encoder in encoders.items():
        df[col] = encoder.fit_transform(df[col])

    # The 'value' column is our target.
    X = df.drop('value', axis=1)
    y = df['value']

    # Remove columns that are not useful for training.
    X = X.drop(['year'], axis=1)
    # The original dataset has many empty columns at the end, remove them.
    X = X.loc[:, ~X.columns.str.contains('^', na=False)]

    return X, y, encoders

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python3 gradient_boosting_machine.py <data_file.csv> <model_output.pkl> <n_estimators>")
        sys.exit(1)

    data_file = sys.argv[1]
    model_file = sys.argv[2]
    n_estimators = int(sys.argv[3])

    print("Loading and preprocessing data...")
    df = pd.read_csv(data_file)
    X, y, _ = preprocess_data(df.copy())

    # Split data for training and validation.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training Gradient Boosting Machine with {n_estimators} estimators...")
    gbm = GradientBoostingMachine(n_estimators=n_estimators, learning_rate=0.1, max_depth=5)
    gbm.fit(X_train, y_train)

    print(f"Saving model to {model_file}...")
    gbm.save_model(model_file)

    # The number of parameters in a tree-based model is less direct to calculate.
    # We can estimate it by the total number of nodes in all trees.
    total_nodes = sum(tree.tree_.node_count for tree in gbm.trees)
    print(f"Model training complete. Approximate parameter count (total nodes): {total_nodes}")


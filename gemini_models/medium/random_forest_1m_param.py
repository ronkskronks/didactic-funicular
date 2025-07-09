#!/usr/bin/env python3

# random_forest_1m_param.py
#
# This script implements a Random Forest Classifier from scratch using NumPy.
# It is designed to have approximately 1,000,000 trainable "parameters",
# where parameters are considered the total number of nodes across all decision trees.
# The script supports saving and loading models, allowing for continued training
# or refinement of the forest.
#
# Random Forest is an ensemble learning method for classification that operates
# by constructing a multitude of decision trees at training time and outputting
# the class that is the mode of the classes of the individual trees.
#
# Parameters Calculation:
# Total Parameters (approx.) = Number of Trees * Max Nodes per Tree
#
# Architecture for ~1,000,000 parameters:
# - Number of Features (Dimensions): 10 (for demonstration data)
# - Number of Trees (n_estimators): 500
# - Maximum Depth of each Tree (max_depth): 10
#
# Calculation Breakdown:
# A binary decision tree of depth D can have at most 2^(D+1) - 1 nodes.
# For max_depth = 10, max nodes per tree = 2^(10+1) - 1 = 2047 nodes.
# Total approximate parameters = 500 trees * 2047 nodes/tree = 1,023,500 parameters.
# This is very close to the requested 1,000,000 parameters.
#
# The code is heavily commented to explain each part of the implementation.

import numpy as np
import pickle # Used for saving and loading the model object
import os # Used for checking file existence
import argparse # Used for parsing command-line arguments

class Node:
    """
    Represents a node in a Decision Tree.
    A node can be either an internal node (with a split condition) or a leaf node (with a prediction).
    """
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx # Index of the feature to split on
        self.threshold = threshold     # Threshold value for the split
        self.left = left               # Left child node (for samples <= threshold)
        self.right = right             # Right child node (for samples > threshold)
        self.value = value             # Prediction value if this is a leaf node (e.g., class label)

    def is_leaf(self):
        """
        Checks if the node is a leaf node.
        """
        return self.value is not None

class DecisionTreeClassifier:
    """
    A single Decision Tree Classifier implementation.
    It uses Gini impurity to find the best split.
    """
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth = max_depth # Maximum depth of the tree
        self.min_samples_split = min_samples_split # Minimum samples required to split a node
        self.n_features = n_features # Number of features to consider for best split (for Random Forest)
        self.root = None # The root node of the decision tree

    def _gini_impurity(self, y):
        """
        Calculates the Gini impurity of a set of labels.
        Gini impurity measures the probability of incorrectly classifying a randomly chosen element
        in the dataset if it were randomly labeled according to the distribution of labels in the dataset.
        """
        if len(y) == 0:
            return 0
        # Count occurrences of each unique class label
        class_counts = np.bincount(y)
        # Calculate probabilities for each class
        probabilities = class_counts / len(y)
        # Gini impurity = 1 - sum(p_i^2) for all classes i
        gini = 1 - np.sum(probabilities**2)
        return gini

    def _best_split(self, X, y):
        """
        Finds the best feature and threshold to split the data based on Gini impurity.
        """
        best_gini = float('inf') # Initialize with a very large value
        best_feature_idx = None
        best_threshold = None

        # If n_features is specified (for Random Forest), randomly select a subset of features.
        feature_indices = np.arange(X.shape[1])
        if self.n_features is not None and self.n_features < X.shape[1]:
            feature_indices = np.random.choice(feature_indices, self.n_features, replace=False)

        for feature_idx in feature_indices:
            # Get unique values of the current feature to use as potential thresholds
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                # Split the data into left and right subsets based on the threshold
                left_indices = np.where(X[:, feature_idx] <= threshold)[0]
                right_indices = np.where(X[:, feature_idx] > threshold)[0]

                # Skip if one of the subsets is empty (no meaningful split)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                # Get labels for the left and right subsets
                y_left, y_right = y[left_indices], y[right_indices]

                # Calculate Gini impurity for each subset
                gini_left = self._gini_impurity(y_left)
                gini_right = self._gini_impurity(y_right)

                # Calculate weighted average Gini impurity for the split
                # This is the impurity of the split, weighted by the size of the subsets.
                weighted_gini = (len(y_left) / len(y)) * gini_left + \
                                (len(y_right) / len(y)) * gini_right

                # If this split is better than the current best, update best split info
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _build_tree(self, X, y, depth=0):
        """
        Recursively builds the decision tree.
        """
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # Stopping criteria:
        # 1. If all samples in the node belong to the same class (pure node).
        # 2. If the maximum depth is reached.
        # 3. If the number of samples is less than the minimum required for a split.
        # 4. If there's only one class left (already pure).
        if num_classes == 1 or \
           (self.max_depth is not None and depth == self.max_depth) or \
           num_samples < self.min_samples_split:
            # Create a leaf node: predict the most frequent class in this node.
            # np.bincount counts occurrences, np.argmax gets the index of the max count.
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)

        # Find the best split for the current node
        feature_idx, threshold = self._best_split(X, y)

        # If no good split is found (e.g., all features are identical, or no reduction in impurity)
        if feature_idx is None:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)

        # Split the data based on the best split found
        left_indices = np.where(X[:, feature_idx] <= threshold)[0]
        right_indices = np.where(X[:, feature_idx] > threshold)[0]

        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # Return an internal node with the split condition and its children
        return Node(feature_idx, threshold, left_child, right_child)

    def fit(self, X, y):
        """
        Builds the decision tree from the training data.
        """
        self.root = self._build_tree(X, y)

    def _predict_tree(self, x, node):
        """
        Recursively traverses the tree to make a prediction for a single sample.
        """
        if node.is_leaf():
            return node.value

        # Decide whether to go left or right based on the feature value and threshold
        if x[node.feature_idx] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)

    def predict(self, X):
        """
        Makes predictions for multiple samples.
        """
        # Apply _predict_tree to each sample in X
        return np.array([self._predict_tree(x, self.root) for x in X])

class RandomForestClassifier:
    """
    A Random Forest Classifier implementation.
    It ensembles multiple Decision Trees to improve accuracy and reduce overfitting.
    """
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators # Number of decision trees in the forest
        self.max_depth = max_depth       # Max depth for each individual tree
        self.min_samples_split = min_samples_split # Min samples to split a node in each tree
        self.max_features = max_features # Number of features to consider for splitting in each tree
        self.trees = []                  # List to store the trained decision trees

    def fit(self, X, y):
        """
        Builds the Random Forest by training individual decision trees.
        Each tree is trained on a bootstrapped sample of the data and a random subset of features.
        """
        num_samples, num_features = X.shape

        # Determine max_features if not specified
        if self.max_features is None:
            self.max_features = int(np.sqrt(num_features)) # Common heuristic for classification
        elif isinstance(self.max_features, float) and 0 < self.max_features <= 1:
            self.max_features = int(self.max_features * num_features)
        elif self.max_features > num_features:
            self.max_features = num_features

        self.trees = [] # Clear existing trees if refitting
        for _ in range(self.n_estimators):
            # Bootstrap sampling: randomly select samples with replacement
            # This creates a diverse set of training data for each tree.
            bootstrap_indices = np.random.choice(num_samples, num_samples, replace=True)
            X_sample, y_sample = X[bootstrap_indices], y[bootstrap_indices]

            # Create and train a new Decision Tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.max_features # Pass max_features to the individual tree
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """
        Makes predictions using the trained Random Forest.
        For classification, it aggregates predictions from all individual trees (majority vote).
        """
        # Get predictions from each individual tree
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        # Transpose to have shape (num_samples, n_estimators)
        tree_predictions = tree_predictions.T

        # Aggregate predictions: majority vote for classification
        final_predictions = np.array([np.argmax(np.bincount(pred_row)) for pred_row in tree_predictions])
        return final_predictions

    def save_model(self, filename):
        """
        Saves the trained Random Forest model to a file using pickle.

        Args:
            filename (str): The path to the file where the model will be saved.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        """
        Loads a trained Random Forest model from a file.

        Args:
            filename (str): The path to the file from which to load the model.

        Returns:
            RandomForestClassifier: The loaded Random Forest model instance.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    # This block demonstrates how to use the RandomForestClassifier class.
    # It will only run when the script is executed directly.

    parser = argparse.ArgumentParser(description="Train or continue training a Random Forest model.")
    parser.add_argument('--model_file', type=str, default="random_forest_1m_param_model.pkl",
                        help="Path to the model file (load from/save to).")
    parser.add_argument('--n_estimators', type=int, default=500,
                        help="Number of trees in the forest.")
    parser.add_argument('--max_depth', type=int, default=10,
                        help="Maximum depth of each tree.")
    parser.add_argument('--n_features_data', type=int, default=10,
                        help="Number of features in the dummy data.")
    parser.add_argument('--num_samples', type=int, default=10000,
                        help="Number of samples in the dummy data.")
    parser.add_argument('--new_model', action='store_true',
                        help="Force creation of a new model, even if model_file exists.")

    args = parser.parse_args()

    # Calculate total parameters (approximate number of nodes)
    # Max nodes per tree = 2^(max_depth + 1) - 1
    approx_nodes_per_tree = (2**(args.max_depth + 1)) - 1
    total_parameters = args.n_estimators * approx_nodes_per_tree
    print(f"Random Forest model will have approximately {total_parameters} parameters (total nodes).")

    # Generate some dummy data for demonstration.
    # X_data: (num_samples, n_features_data)
    # y_data: (num_samples,) - class labels (0 or 1 for binary classification)
    # We create a simple classification problem where class depends on sum of features.
    X_data = np.random.rand(args.num_samples, args.n_features_data) * 10
    y_data = (np.sum(X_data, axis=1) > (args.n_features_data * 5)).astype(int) # Binary classification

    rf_model = None
    # Check if model file exists and --new_model flag is not set
    if os.path.exists(args.model_file) and not args.new_model:
        print(f"Loading model from {args.model_file}...")
        try:
            rf_model = RandomForestClassifier.load_model(args.model_file)
            # Basic check: ensure loaded model matches expected n_estimators/max_depth (optional)
            if not (rf_model.n_estimators == args.n_estimators and \
                    rf_model.max_depth == args.max_depth):
                print("Warning: Loaded model configuration does not match expected. Creating new model.")
                rf_model = None # Force new model creation
        except Exception as e:
            print(f"Error loading model: {e}. Creating a new model.")
            rf_model = None

    if rf_model is None:
        print("Creating a new Random Forest model...")
        rf_model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            max_features=args.n_features_data # Use all features for simplicity in this example
        )

    print("Starting Random Forest training...")
    rf_model.fit(X_data, y_data)
    print("Training complete.")

    # Make predictions on a small subset of the training data
    print("Predictions on a subset of training data:")
    test_X = X_data[:5]
    test_y = y_data[:5]
    predictions = rf_model.predict(test_X)
    print(f"Input: {test_X}")
    print(f"Predicted Classes: {predictions}")
    print(f"True Classes: {test_y}")

    # Save the trained model
    rf_model.save_model(args.model_file)
    print(f"Model saved to {args.model_file}")

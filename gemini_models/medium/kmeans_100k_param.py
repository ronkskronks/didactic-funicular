#!/usr/bin/env python3

# kmeans_100k_param.py
#
# This script implements the K-Means clustering algorithm from scratch using NumPy.
# It is designed to have approximately 100,000 "parameters", where parameters
# are defined as the coordinates of the learned centroids.
# The script supports saving and loading models, allowing for continued training
# or refinement of clusters.
#
# K-Means is an unsupervised learning algorithm used for grouping data points
# into K distinct clusters, where each data point belongs to the cluster
# with the nearest mean (centroid).
#
# Parameters Calculation:
# Total Parameters = Number of Clusters (K) * Number of Features (Dimensions)
#
# Architecture for ~100,000 parameters:
# - Number of Features (Dimensions): 10 (e.g., for a 10-dimensional input feature vector)
# - Number of Clusters (K): 10,000
#
# Calculation Breakdown:
# Each centroid is a point in the 10-dimensional space. So, each centroid has 10 coordinates.
# Total Parameters = 10,000 clusters * 10 dimensions/coordinates per cluster = 100,000 parameters.
#
# The code is heavily commented to explain each part of the implementation.

import numpy as np
import pickle # Used for saving and loading the model object
import os # Used for checking file existence
import argparse # Used for parsing command-line arguments

class KMeans:
    """
    A simple K-Means clustering algorithm implementation.
    """

    def __init__(self, n_clusters, max_iterations=300, tolerance=1e-4):
        """
        Initializes the K-Means model.

        Args:
            n_clusters (int): The number of clusters (K) to form.
            max_iterations (int): The maximum number of iterations to run the K-Means algorithm.
            tolerance (float): The threshold for convergence. If the change in centroids
                               between iterations is less than this, the algorithm stops.
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None # Will store the learned centroids (the model parameters)

    def _initialize_centroids(self, X, initial_centroids=None):
        """
        Initializes the centroids for the clustering process.
        If initial_centroids are provided (e.g., from a loaded model), use them.
        Otherwise, randomly select n_clusters data points from X as initial centroids.

        Args:
            X (np.array): The input data, shape (num_samples, num_features).
            initial_centroids (np.array, optional): Pre-existing centroids to start with.
        """
        if initial_centroids is not None:
            self.centroids = initial_centroids
        else:
            # Randomly select n_clusters data points as initial centroids.
            # np.random.choice selects unique indices from the data.
            random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            self.centroids = X[random_indices]

    def _assign_clusters(self, X):
        """
        Assigns each data point to the closest centroid.

        Args:
            X (np.array): The input data, shape (num_samples, num_features).

        Returns:
            np.array: An array of cluster assignments for each data point,
                      shape (num_samples,).
        """
        # Calculate Euclidean distance from each data point to each centroid.
        # X[:, np.newaxis, :] makes X (num_samples, 1, num_features)
        # self.centroids makes it (n_clusters, num_features)
        # The subtraction results in (num_samples, n_clusters, num_features)
        # np.linalg.norm calculates the Euclidean norm (distance) along the last axis.
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
        # np.argmin finds the index of the minimum distance along the cluster axis (axis=1).
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, assignments):
        """
        Recalculates the centroids based on the mean of the data points assigned to each cluster.

        Args:
            X (np.array): The input data, shape (num_samples, num_features).
            assignments (np.array): The cluster assignments for each data point.

        Returns:
            np.array: The new centroids, shape (n_clusters, num_features).
        """
        new_centroids = np.zeros_like(self.centroids) # Initialize new centroids with zeros
        for i in range(self.n_clusters):
            # Select all data points assigned to the current cluster i.
            points_in_cluster = X[assignments == i]
            if len(points_in_cluster) > 0:
                # Calculate the mean of these points to get the new centroid.
                new_centroids[i] = np.mean(points_in_cluster, axis=0)
            else:
                # If a cluster becomes empty, re-initialize its centroid randomly
                # from the data points to prevent issues. This is a common strategy.
                new_centroids[i] = X[np.random.choice(X.shape[0])]
        return new_centroids

    def fit(self, X, initial_centroids=None):
        """
        Runs the K-Means algorithm to find the optimal centroids.

        Args:
            X (np.array): The input data, shape (num_samples, num_features).
            initial_centroids (np.array, optional): Pre-existing centroids to start with.
                                                  Used for continuing training.
        """
        self._initialize_centroids(X, initial_centroids) # Initialize or load centroids

        for i in range(self.max_iterations):
            old_centroids = self.centroids.copy() # Keep a copy to check for convergence

            # Step 1: Assign data points to the closest centroids
            assignments = self._assign_clusters(X)

            # Step 2: Update centroids based on the new assignments
            self.centroids = self._update_centroids(X, assignments)

            # Check for convergence: if centroids haven't moved much, stop.
            # np.linalg.norm calculates the Frobenius norm (distance) between old and new centroids.
            if np.linalg.norm(self.centroids - old_centroids) < self.tolerance:
                # print(f"K-Means converged after {i+1} iterations.") # For debugging
                break

    def predict(self, X):
        """
        Predicts the cluster assignment for new data points.

        Args:
            X (np.array): New input data, shape (num_samples, num_features).

        Returns:
            np.array: An array of cluster assignments for each data point,
                      shape (num_samples,).
        """
        if self.centroids is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._assign_clusters(X)

    def save_model(self, filename):
        """
        Saves the learned centroids (the model parameters) to a file using pickle.

        Args:
            filename (str): The path to the file where the model will be saved.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.centroids, f)

    @staticmethod
    def load_model(filename):
        """
        Loads centroids from a file.

        Args:
            filename (str): The path to the file from which to load the centroids.

        Returns:
            np.array: The loaded centroids.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    # This block demonstrates how to use the KMeans class.
    # It will only run when the script is executed directly.

    parser = argparse.ArgumentParser(description="Train or continue training a K-Means model.")
    parser.add_argument('--model_file', type=str, default="kmeans_100k_param_model.pkl",
                        help="Path to the model file (load from/save to).")
    parser.add_argument('--n_clusters', type=int, default=10000,
                        help="Number of clusters (K).")
    parser.add_argument('--n_features', type=int, default=10,
                        help="Number of features (dimensions) in the data.")
    parser.add_argument('--max_iterations', type=int, default=100,
                        help="Maximum number of K-Means iterations.")
    parser.add_argument('--new_model', action='store_true',
                        help="Force creation of a new model, even if model_file exists.")

    args = parser.parse_args()

    # Calculate total parameters based on chosen K and dimensions.
    total_parameters = args.n_clusters * args.n_features
    print(f"K-Means model will have approximately {total_parameters} parameters (centroid coordinates).")

    # Generate some dummy data for demonstration.
    # For K-Means, data should be numerical.
    # We generate data with args.n_features dimensions and a reasonable number of samples.
    # The number of samples should be significantly larger than n_clusters for meaningful clustering.
    num_samples = args.n_clusters * 5 # 5 times the number of clusters
    X_data = np.random.rand(num_samples, args.n_features) * 100 # Random data between 0 and 100

    kmeans_model = KMeans(args.n_clusters, args.max_iterations)
    initial_centroids = None

    # Check if model file exists and --new_model flag is not set
    if os.path.exists(args.model_file) and not args.new_model:
        print(f"Loading existing centroids from {args.model_file}...")
        try:
            initial_centroids = KMeans.load_model(args.model_file)
            # Basic check: ensure loaded centroids match expected dimensions
            if initial_centroids.shape[1] != args.n_features or \
               initial_centroids.shape[0] != args.n_clusters:
                print("Warning: Loaded centroids do not match expected dimensions or cluster count. Starting new model.")
                initial_centroids = None
        except Exception as e:
            print(f"Error loading model: {e}. Starting a new model.")
            initial_centroids = None

    if initial_centroids is None:
        print("Initializing new centroids...")

    print(f"Starting K-Means training for {args.max_iterations} iterations...")
    kmeans_model.fit(X_data, initial_centroids=initial_centroids)
    print("Training complete.")

    # Save the trained model (centroids)
    kmeans_model.save_model(args.model_file)
    print(f"Model (centroids) saved to {args.model_file}")

    # Example prediction (optional)
    # new_data_point = np.random.rand(1, args.n_features) * 100
    # cluster_assignment = kmeans_model.predict(new_data_point)
    # print(f"New data point {new_data_point} assigned to cluster {cluster_assignment[0]}")

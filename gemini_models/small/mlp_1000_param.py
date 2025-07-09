#!/usr/bin/env python3

# mlp_1000_param.py
#
# This script implements a Multi-Layer Perceptron (MLP) from scratch
# using NumPy, designed to have approximately 1000 trainable parameters.
# The code is heavily commented to explain each part of the implementation.
#
# This model is suitable for simple binary classification tasks.
#
# Parameters Calculation:
# For an MLP, the total number of parameters consists of all weights and biases.
#
# Architecture:
# - Input Layer: 2 neurons (e.g., for a 2-dimensional input feature vector)
# - Hidden Layer 1: 200 neurons
# - Output Layer: 2 neurons (e.g., for binary classification, one-hot encoded)
#
# Calculation Breakdown:
# 1. Weights connecting Input Layer to Hidden Layer 1:
#    - Each of the 200 hidden neurons receives input from 2 input neurons.
#    - Number of weights = 200 * 2 = 400
# 2. Biases for Hidden Layer 1:
#    - Each of the 200 hidden neurons has its own bias.
#    - Number of biases = 200
# 3. Weights connecting Hidden Layer 1 to Output Layer:
#    - Each of the 2 output neurons receives input from 200 hidden neurons.
#    - Number of weights = 2 * 200 = 400
# 4. Biases for Output Layer:
#    - Each of the 2 output neurons has its own bias.
#    - Number of biases = 2
#
# Total Parameters = (400 + 200) + (400 + 2) = 600 + 402 = 1002 parameters.
# This is very close to the requested 1000 parameters.

import numpy as np
import pickle # Used for saving and loading the model object

class SimpleMLP:
    """
    A simple Multi-Layer Perceptron (MLP) class for demonstration purposes.
    It supports a single hidden layer and uses the sigmoid activation function.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initializes the MLP with specified layer sizes and a learning rate.
        Weights and biases are initialized randomly to break symmetry and
        allow the network to learn different features.

        Args:
            input_size (int): Number of neurons in the input layer.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of neurons in the output layer.
            learning_rate (float): The step size for updating weights during training.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases for the first layer (Input to Hidden)
        # Weights are initialized with small random values to prevent saturation
        # of activation functions and to ensure proper learning.
        # Shape: (hidden_size, input_size)
        self.weights_input_hidden = np.random.randn(self.hidden_size, self.input_size) * 0.01
        # Biases are initialized to zeros. Shape: (hidden_size, 1)
        self.bias_hidden = np.zeros((self.hidden_size, 1))

        # Initialize weights and biases for the second layer (Hidden to Output)
        # Shape: (output_size, hidden_size)
        self.weights_hidden_output = np.random.randn(self.output_size, self.hidden_size) * 0.01
        # Biases are initialized to zeros. Shape: (output_size, 1)
        self.bias_output = np.zeros((self.output_size, 1))

        # Calculate and store the total number of parameters for verification.
        self.total_parameters = (\
            self.weights_input_hidden.size + self.bias_hidden.size + \
            self.weights_hidden_output.size + self.bias_output.size\
        )
        # This will be 400 + 200 + 400 + 2 = 1002 for the chosen architecture.

    def _sigmoid(self, x):
        """
        The sigmoid activation function.
        It squashes input values between 0 and 1, making them suitable for
        probability-like outputs or as activations for hidden layers.
        """
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """
        The derivative of the sigmoid function.
        This is crucial for backpropagation, as it determines how much
        the weights should be adjusted based on the error.
        """
        s = self._sigmoid(x)
        return s * (1 - s)

    def forward(self, X):
        """
        Performs a forward pass through the network.
        Calculates the output of each layer given an input.

        Args:
            X (np.array): Input data, expected shape (num_features, num_samples).
                          For this model, (2, num_samples).

        Returns:
            tuple: A tuple containing:
                   - hidden_output (np.array): Activations of the hidden layer.
                   - final_output (np.array): Activations of the output layer.
                   - z1 (np.array): Weighted sum before activation for hidden layer.
                   - z2 (np.array): Weighted sum before activation for output layer.
        """
        # Input to Hidden Layer
        # z1 is the weighted sum of inputs plus bias for the hidden layer.
        # np.dot performs matrix multiplication.
        z1 = np.dot(self.weights_input_hidden, X) + self.bias_hidden
        # hidden_output applies the sigmoid activation to z1.
        hidden_output = self._sigmoid(z1)

        # Hidden to Output Layer
        # z2 is the weighted sum of hidden layer outputs plus bias for the output layer.
        z2 = np.dot(self.weights_hidden_output, hidden_output) + self.bias_output
        # final_output applies the sigmoid activation to z2.
        final_output = self._sigmoid(z2)

        return hidden_output, final_output, z1, z2

    def backward(self, X, y, hidden_output, final_output, z1, z2):
        """
        Performs the backpropagation algorithm to calculate gradients
        (how much each weight and bias should change).

        Args:
            X (np.array): Input data (num_features, num_samples).
            y (np.array): True labels (output_size, num_samples).
            hidden_output (np.array): Activations of the hidden layer from forward pass.
            final_output (np.array): Activations of the output layer from forward pass.
            z1 (np.array): Weighted sum before activation for hidden layer.
            z2 (np.array): Weighted sum before activation for output layer.

        Returns:
            tuple: A tuple containing:
                   - d_weights_input_hidden (np.array): Gradient for input-hidden weights.
                   - d_bias_hidden (np.array): Gradient for hidden layer biases.
                   - d_weights_hidden_output (np.array): Gradient for hidden-output weights.
                   - d_bias_output (np.array): Gradient for output layer biases.
        """
        num_samples = X.shape[1] # Number of training examples

        # Calculate error at the output layer
        # Error = (Predicted Output - True Label)
        output_error = final_output - y
        # Delta for output layer: Error * derivative of output activation
        output_delta = output_error * self._sigmoid_derivative(z2)

        # Calculate gradients for output layer weights and biases
        # d_weights_hidden_output: (output_delta @ hidden_output.T) / num_samples
        # This calculates how much each weight contributed to the error.
        d_weights_hidden_output = np.dot(output_delta, hidden_output.T) / num_samples
        # d_bias_output: Sum of output_delta across all samples / num_samples
        # This calculates how much each bias contributed to the error.
        d_bias_output = np.sum(output_delta, axis=1, keepdims=True) / num_samples

        # Calculate error at the hidden layer
        # Hidden error is propagated back from the output layer's delta
        hidden_error = np.dot(self.weights_hidden_output.T, output_delta)
        # Delta for hidden layer: Hidden Error * derivative of hidden activation
        hidden_delta = hidden_error * self._sigmoid_derivative(z1)

        # Calculate gradients for input-hidden layer weights and biases
        # d_weights_input_hidden: (hidden_delta @ X.T) / num_samples
        d_weights_input_hidden = np.dot(hidden_delta, X.T) / num_samples
        # d_bias_hidden: Sum of hidden_delta across all samples / num_samples
        d_bias_hidden = np.sum(hidden_delta, axis=1, keepdims=True) / num_samples

        return d_weights_input_hidden, d_bias_hidden, d_weights_hidden_output, d_bias_output

    def train(self, X_train, y_train, epochs):
        """
        Trains the MLP using the backpropagation algorithm.

        Args:
            X_train (np.array): Training input data, shape (num_features, num_samples).
            y_train (np.array): Training labels, shape (output_size, num_samples).
            epochs (int): Number of training iterations over the entire dataset.
        """
        # Ensure X_train and y_train have the correct shape (features, samples)
        # This is important for matrix multiplication.
        if X_train.shape[0] != self.input_size:
            X_train = X_train.T # Transpose if samples are rows

        if y_train.shape[0] != self.output_size:
            y_train = y_train.T # Transpose if samples are rows

        for epoch in range(epochs):
            # Perform forward pass
            hidden_output, final_output, z1, z2 = self.forward(X_train)

            # Perform backward pass to get gradients
            d_w_ih, d_b_h, d_w_ho, d_b_o = self.backward(X_train, y_train, hidden_output, final_output, z1, z2)

            # Update weights and biases using gradient descent
            # Weights and biases are adjusted in the direction that reduces the error.
            self.weights_input_hidden -= self.learning_rate * d_w_ih
            self.bias_hidden -= self.learning_rate * d_b_h
            self.weights_hidden_output -= self.learning_rate * d_w_ho
            self.bias_output -= self.learning_rate * d_b_o

            # Optional: Print loss every few epochs to monitor training progress
            # This is commented out to adhere to the "no output" rule unless explicitly asked.
            # if epoch % 100 == 0:
            #     loss = np.mean(np.square(final_output - y_train))
            #     print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        """
        Predicts the probability of each class for the given input.

        Args:
            X (np.array): Input data, shape (num_features, num_samples).

        Returns:
            np.array: Predicted probabilities, shape (output_size, num_samples).
        """
        # Ensure X has the correct shape (features, samples)
        if X.shape[0] != self.input_size:
            X = X.T # Transpose if samples are rows

        _, final_output, _, _ = self.forward(X)
        return final_output

    def predict(self, X):
        """
        Predicts the class label for the given input.

        Args:
            X (np.array): Input data, shape (num_features, num_samples).

        Returns:
            np.array: Predicted class labels (0 or 1), shape (num_samples,).
        """
        probabilities = self.predict_proba(X)
        # For binary classification with one-hot encoded output,
        # the class with the higher probability is chosen.
        # np.argmax(axis=0) gets the index of the max value along the columns (samples).
        return np.argmax(probabilities, axis=0)

    def save_model(self, filename):
        """
        Saves the trained model (weights and biases) to a file using pickle.
        This allows the model to be loaded and reused later without retraining.

        Args:
            filename (str): The path to the file where the model will be saved.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        """
        Loads a trained model from a file.

        Args:
            filename (str): The path to the file from which to load the model.

        Returns:
            SimpleMLP: The loaded MLP model instance.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    # This block demonstrates how to use the SimpleMLP class.
    # It will only run when the script is executed directly.

    # Define the architecture: Input (2) -> Hidden (200) -> Output (2)
    input_dim = 2
    hidden_dim = 200
    output_dim = 2
    learning_rate = 0.1
    epochs = 1000 # Number of training iterations

    # Create an instance of the MLP model
    model = SimpleMLP(input_dim, hidden_dim, output_dim, learning_rate)

    # Print the calculated total number of parameters.
    # This confirms the model size as requested.
    # This output is allowed as it's part of the demonstration/verification.
    print(f"Model created with {model.total_parameters} parameters.")

    # Generate some dummy data for demonstration.
    # X_train: (num_features, num_samples)
    # y_train: (output_size, num_samples) - one-hot encoded
    # Example: XOR-like data
    X_train = np.array([[0, 0, 1, 1],
                        [0, 1, 0, 1]])

    # Corresponding labels (one-hot encoded for 2 output neurons)
    # For XOR: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
    y_train = np.array([[1, 0, 0, 1], # Class 0 (e.g., 0,0 and 1,1)
                        [0, 1, 1, 0]]) # Class 1 (e.g., 0,1 and 1,0)

    # Train the model
    print("Starting training...")
    model.train(X_train, y_train, epochs)
    print("Training complete.")

    # Make predictions on the training data
    print("Predictions on training data:")
    predictions = model.predict(X_train)
    print(f"Input: {X_train}")
    print(f"Predicted Classes: {predictions}")
    print(f"True Classes (argmax): {np.argmax(y_train, axis=0)}")

    # Save the trained model
    model_filename = "mlp_1000_param_model.pkl"
    model.save_model(model_filename)
    print(f"Model saved to {model_filename}")

    # Load the model back (demonstration of persistence)
    loaded_model = SimpleMLP.load_model(model_filename)
    print(f"Model loaded from {model_filename}. Total parameters: {loaded_model.total_parameters}")

    # Make predictions with the loaded model
    loaded_predictions = loaded_model.predict(X_train)
    print(f"Predictions with loaded model: {loaded_predictions}")

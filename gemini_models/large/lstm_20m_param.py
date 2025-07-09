#!/usr/bin/env python3

# lstm_20m_param.py
#
# This script implements a simplified Long Short-Term Memory (LSTM) network
# from scratch using NumPy. It is designed to have approximately 20,000,000
# trainable parameters. The model performs character-level sequence prediction,
# demonstrating basic text recognition, data correlation, and sequential reasoning.
#
# The code is heavily commented to explain each part of the implementation.
#
# Parameters Calculation:
# For a single LSTM layer, the number of parameters is approximately:
# 4 * (hidden_dim * input_dim + hidden_dim * hidden_dim + hidden_dim)
# where:
#   - input_dim: Size of the input vector (e.g., vocabulary size for one-hot encoding).
#   - hidden_dim: Size of the hidden state and cell state.
#
# Architecture for ~20,000,000 parameters:
# - Input Dimension (input_dim): 30 (e.g., for 26 lowercase letters + space, newline, unknown)
# - Hidden Dimension (hidden_dim): 2221
#
# Calculation Breakdown:
# Parameters = 4 * (2221 * 30 + 2221 * 2221 + 2221)
# Parameters = 4 * (66630 + 4933041 + 2221)
# Parameters = 4 * (5001892)
# Parameters = 20,007,568
# This is very close to the requested 20,000,000 parameters.

import numpy as np
import pickle # Used for saving and loading the model object
import os # Used for checking file existence
import argparse # Used for parsing command-line arguments

class LSTMCell:
    """
    A single LSTM cell implementation.
    Handles the forward and backward pass for one time step.
    """
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weights and biases for the four gates: input, forget, cell, output
        # Weights for input (W_xi, W_xf, W_xc, W_xo)
        self.W_x = np.random.randn(4 * hidden_dim, input_dim) * 0.01
        # Weights for recurrent connections (W_hi, W_hf, W_hc, W_ho)
        self.W_h = np.random.randn(4 * hidden_dim, hidden_dim) * 0.01
        # Biases (b_i, b_f, b_c, b_o)
        self.b = np.zeros((4 * hidden_dim, 1))

    def forward(self, x, h_prev, c_prev):
        """
        Performs the forward pass for one LSTM cell at a single time step.

        Args:
            x (np.array): Input at current time step (input_dim, 1).
            h_prev (np.array): Previous hidden state (hidden_dim, 1).
            c_prev (np.array): Previous cell state (hidden_dim, 1).

        Returns:
            tuple: (h_next, c_next, cache)
                   - h_next (np.array): Next hidden state.
                   - c_next (np.array): Next cell state.
                   - cache (tuple): Values needed for backward pass.
        """
        # Concatenate input and previous hidden state for convenience
        # This is a common optimization to perform one large matrix multiplication.
        concat_input = np.vstack((x, h_prev))

        # Calculate gate activations (i, f, c_tilde, o)
        # This is the combined linear transformation for all four gates.
        gates = np.dot(self.W_x, x) + np.dot(self.W_h, h_prev) + self.b

        # Split gates into individual components
        i_gate = self._sigmoid(gates[0*self.hidden_dim : 1*self.hidden_dim, :]) # Input gate
        f_gate = self._sigmoid(gates[1*self.hidden_dim : 2*self.hidden_dim, :]) # Forget gate
        c_tilde = np.tanh(gates[2*self.hidden_dim : 3*self.hidden_dim, :]) # Candidate cell state
        o_gate = self._sigmoid(gates[3*self.hidden_dim : 4*self.hidden_dim, :]) # Output gate

        # Calculate next cell state
        c_next = f_gate * c_prev + i_gate * c_tilde

        # Calculate next hidden state
        h_next = o_gate * np.tanh(c_next)

        # Store values for backward pass
        cache = (x, h_prev, c_prev, gates, i_gate, f_gate, c_tilde, o_gate, c_next, h_next)
        return h_next, c_next, cache

    def backward(self, dh_next, dc_next, cache):
        """
        Performs the backward pass for one LSTM cell at a single time step.

        Args:
            dh_next (np.array): Gradient of loss with respect to next hidden state.
            dc_next (np.array): Gradient of loss with respect to next cell state.
            cache (tuple): Values from forward pass.

        Returns:
            tuple: (dx, dh_prev, dc_prev, dW_x, dW_h, db)
                   - dx (np.array): Gradient of loss with respect to input x.
                   - dh_prev (np.array): Gradient of loss with respect to previous hidden state.
                   - dc_prev (np.array): Gradient of loss with respect to previous cell state.
                   - dW_x (np.array): Gradient of loss with respect to W_x.
                   - dW_h (np.array): Gradient of loss with respect to W_h.
                   - db (np.array): Gradient of loss with respect to b.
        """
        x, h_prev, c_prev, gates, i_gate, f_gate, c_tilde, o_gate, c_next, h_next = cache

        # Gradients for tanh(c_next) and o_gate
        do_gate = dh_next * np.tanh(c_next)
        dc_next_tanh = dh_next * o_gate * (1 - np.tanh(c_next)**2)

        # Add dc_next from next time step to dc_next_tanh
        dc_next_combined = dc_next + dc_next_tanh

        # Gradients for f_gate, i_gate, c_tilde
        df_gate = dc_next_combined * c_prev * self._sigmoid_derivative(f_gate)
        di_gate = dc_next_combined * c_tilde * self._sigmoid_derivative(i_gate)
        dc_tilde = dc_next_combined * i_gate * (1 - np.tanh(c_tilde)**2)

        # Gradients for input to gates (d_gates)
        d_gates = np.vstack([
            di_gate,
            df_gate,
            dc_tilde,
            do_gate
        ])

        # Gradients for weights and biases
        dW_x = np.dot(d_gates, x.T)
        dW_h = np.dot(d_gates, h_prev.T)
        db = np.sum(d_gates, axis=1, keepdims=True)

        # Gradients for previous hidden and cell states, and input x
        dx = np.dot(self.W_x.T, d_gates)
        dh_prev = np.dot(self.W_h.T, d_gates)
        dc_prev = dc_next_combined * f_gate

        return dx, dh_prev, dc_prev, dW_x, dW_h, db

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        s = self._sigmoid(x)
        return s * (1 - s)


class LSTMModel:
    """
    A simple LSTM network for sequence prediction.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Initialize LSTM cell
        self.lstm_cell = LSTMCell(input_dim, hidden_dim)

        # Output layer weights and bias (maps hidden state to output)
        self.W_out = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b_out = np.zeros((output_dim, 1))

        # Calculate total parameters
        self.total_parameters = (
            self.lstm_cell.W_x.size + self.lstm_cell.W_h.size + self.lstm_cell.b.size +
            self.W_out.size + self.b_out.size
        )

    def forward(self, inputs):
        """
        Performs forward pass through the LSTM network for a sequence.

        Args:
            inputs (list of np.array): List of input vectors for each time step.

        Returns:
            tuple: (outputs, h_states, c_states, caches)
                   - outputs (list): List of output predictions for each time step.
                   - h_states (list): List of hidden states for each time step.
                   - c_states (list): List of cell states for each time step.
                   - caches (list): List of caches from each LSTM cell forward pass.
        """
        T = len(inputs) # Sequence length
        h_states = [np.zeros((self.hidden_dim, 1))] * (T + 1) # h_0 is all zeros
        c_states = [np.zeros((self.hidden_dim, 1))] * (T + 1) # c_0 is all zeros
        outputs = []
        caches = []

        for t in range(T):
            h_prev = h_states[t]
            c_prev = c_states[t]
            x = inputs[t].reshape(-1, 1) # Ensure input is column vector

            h_next, c_next, cache = self.lstm_cell.forward(x, h_prev, c_prev)

            # Store states and cache
            h_states[t+1] = h_next
            c_states[t+1] = c_next
            caches.append(cache)

            # Calculate output prediction for this time step
            out = self._softmax(np.dot(self.W_out, h_next) + self.b_out)
            outputs.append(out)

        return outputs, h_states, c_states, caches

    def backward(self, outputs, targets, h_states, c_states, caches):
        """
        Performs backward pass (backpropagation through time) for the LSTM network.

        Args:
            outputs (list): List of predicted outputs from forward pass.
            targets (list): List of true target outputs for each time step.
            h_states (list): List of hidden states from forward pass.
            c_states (list): List of cell states from forward pass.
            caches (list): List of caches from forward pass.

        Returns:
            tuple: (dW_x, dW_h, db, dW_out, db_out)
                   - Gradients for all weights and biases.
        """
        T = len(outputs)

        # Initialize gradients
        dW_x, dW_h, db = np.zeros_like(self.lstm_cell.W_x), np.zeros_like(self.lstm_cell.W_h), np.zeros_like(self.lstm_cell.b)
        dW_out, db_out = np.zeros_like(self.W_out), np.zeros_like(self.b_out)

        dh_next = np.zeros((self.hidden_dim, 1))
        dc_next = np.zeros((self.hidden_dim, 1))

        for t in reversed(range(T)):
            # Gradients for output layer
            dout = outputs[t] - targets[t].reshape(-1, 1) # Derivative of cross-entropy loss with softmax
            dW_out_t = np.dot(dout, h_states[t+1].T)
            db_out_t = np.sum(dout, axis=1, keepdims=True)

            # Accumulate gradients for output layer
            dW_out += dW_out_t
            db_out += db_out_t

            # Gradient of loss with respect to hidden state from output layer
            dh_t = np.dot(self.W_out.T, dout) + dh_next # Add dh_next from next time step

            # Gradients for LSTM cell
            dx_t, dh_prev_t, dc_prev_t, dW_x_t, dW_h_t, db_t = self.lstm_cell.backward(dh_t, dc_next, caches[t])

            # Accumulate gradients for LSTM cell
            dW_x += dW_x_t
            dW_h += dW_h_t
            db += db_t

            # Pass gradients to previous time step
            dh_next = dh_prev_t
            dc_next = dc_prev_t

        return dW_x, dW_h, db, dW_out, db_out

    def train(self, X_train_sequences, y_train_sequences, epochs):
        """
        Trains the LSTM model using backpropagation through time.

        Args:
            X_train_sequences (list of list of np.array): List of input sequences.
            y_train_sequences (list of list of np.array): List of target sequences.
            epochs (int): Number of training iterations.
        """
        for epoch in range(epochs):
            total_loss = 0
            for seq_idx in range(len(X_train_sequences)):
                inputs = X_train_sequences[seq_idx]
                targets = y_train_sequences[seq_idx]

                # Forward pass
                outputs, h_states, c_states, caches = self.forward(inputs)

                # Calculate loss (Cross-entropy for character prediction)
                # Sum loss over all time steps in the sequence
                loss = -np.sum(targets * np.log(np.array(outputs).squeeze())) / len(targets)
                total_loss += loss

                # Backward pass
                dW_x, dW_h, db, dW_out, db_out = self.backward(outputs, targets, h_states, c_states, caches)

                # Update weights and biases
                self.lstm_cell.W_x -= self.learning_rate * dW_x
                self.lstm_cell.W_h -= self.learning_rate * dW_h
                self.lstm_cell.b -= self.learning_rate * db
                self.W_out -= self.learning_rate * dW_out
                self.b_out -= self.learning_rate * db_out

            # print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(X_train_sequences):.4f}")

    def predict(self, input_sequence):
        """
        Generates a prediction for a given input sequence.
        """
        outputs, _, _, _ = self.forward(input_sequence)
        # For prediction, we take the argmax of the last output (next character)
        return np.argmax(outputs[-1])

    def _softmax(self, x):
        """
        Softmax activation function for the output layer.
        Converts raw scores into probabilities that sum to 1.
        """
        exp_x = np.exp(x - np.max(x)) # Subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=0)

    def save_model(self, filename):
        """
        Saves the trained LSTM model to a file using pickle.

        Args:
            filename (str): The path to the file where the model will be saved.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        """
        Loads a trained LSTM model from a file.

        Args:
            filename (str): The path to the file from which to load the model.

        Returns:
            LSTMModel: The loaded LSTM model instance.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

# --- Data Preprocessing for Character-level Prediction ---

def create_char_mapping(text):
    """
    Creates a mapping from characters to integers and vice-versa.
    """
    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_int, int_to_char, len(chars)

def text_to_sequences(text, char_to_int, seq_length):
    """
    Converts text into input-target sequences for training.
    Each input is a sequence of characters, and the target is the next character.
    """
    X_sequences = []
    y_sequences = []
    for i in range(0, len(text) - seq_length):
        input_seq_chars = text[i : i + seq_length]
        target_char = text[i + seq_length]

        # Convert input sequence to one-hot encoded vectors
        input_vectors = []
        for char in input_seq_chars:
            one_hot = np.zeros((len(char_to_int), 1))
            one_hot[char_to_int[char]] = 1
            input_vectors.append(one_hot)

        # Convert target character to one-hot encoded vector
        target_vector = np.zeros((len(char_to_int), 1))
        target_vector[char_to_int[target_char]] = 1

        X_sequences.append(input_vectors)
        y_sequences.append(target_vector)
    return X_sequences, y_sequences


if __name__ == "__main__":
    # This block demonstrates how to use the LSTMModel class.
    # It will only run when the script is executed directly.

    parser = argparse.ArgumentParser(description="Train or continue training an LSTM model.")
    parser.add_argument('--model_file', type=str, default="lstm_20m_param_model.pkl",
                        help="Path to the model file (load from/save to).")
    parser.add_argument('--epochs', type=int, default=1,
                        help="Number of training epochs. Set to 1 for quick demo due to large model size.")
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help="Learning rate for training.")
    parser.add_argument('--new_model', action='store_true',
                        help="Force creation of a new model, even if model_file exists.")
    parser.add_argument('--hidden_dim', type=int, default=2221,
                        help="Hidden dimension of the LSTM layer (determines parameter count).")
    parser.add_argument('--seq_length', type=int, default=10,
                        help="Length of input sequences for training.")

    args = parser.parse_args()

    # --- Dummy Text Data for Demonstration ---
    # In a real scenario, this would be a much larger corpus.
    text_data = "hello world this is a test of the long short term memory network"
    text_data += " for character level prediction and some basic reasoning capabilities"
    text_data += " it learns patterns in sequences to predict the next character"

    # Preprocess text data
    char_to_int, int_to_char, vocab_size = create_char_mapping(text_data)
    print(f"Vocabulary size: {vocab_size}")

    X_sequences, y_sequences = text_to_sequences(text_data, char_to_int, args.seq_length)
    print(f"Generated {len(X_sequences)} training sequences.")

    # Define LSTM architecture
    input_dim = vocab_size
    hidden_dim = args.hidden_dim
    output_dim = vocab_size

    lstm_model = None
    # Check if model file exists and --new_model flag is not set
    if os.path.exists(args.model_file) and not args.new_model:
        print(f"Loading model from {args.model_file}...")
        try:
            lstm_model = LSTMModel.load_model(args.model_file)
            # Basic check: ensure loaded model matches expected architecture
            if not (lstm_model.input_dim == input_dim and \
                    lstm_model.hidden_dim == hidden_dim and \
                    lstm_model.output_dim == output_dim):
                print("Warning: Loaded model architecture does not match expected. Creating new model.")
                lstm_model = None # Force new model creation
        except Exception as e:
            print(f"Error loading model: {e}. Creating a new model.")
            lstm_model = None

    if lstm_model is None:
        print("Creating a new LSTM model...")
        lstm_model = LSTMModel(input_dim, hidden_dim, output_dim, args.learning_rate)

    print(f"Model has {lstm_model.total_parameters} parameters.")

    # Train the model
    print(f"Starting LSTM training for {args.epochs} epochs...")
    # Note: Training a 20M parameter model from scratch with NumPy will be extremely slow.
    # The default epochs is set to 1 for demonstration purposes.
    lstm_model.train(X_sequences, y_sequences, args.epochs)
    print("Training complete.")

    # Save the trained model
    lstm_model.save_model(args.model_file)
    print(f"Model saved to {args.model_file}")

    # --- Demonstrate Prediction (Basic Text Generation) ---
    print("\nDemonstrating prediction:")
    seed_text = text_data[0:args.seq_length]
    print(f"Seed: \"{seed_text}\"")

    # Convert seed text to sequence of one-hot vectors
    seed_sequence = []
    for char in seed_text:
        one_hot = np.zeros((vocab_size, 1))
        one_hot[char_to_int[char]] = 1
        seed_sequence.append(one_hot)

    predicted_char_idx = lstm_model.predict(seed_sequence)
    predicted_char = int_to_char[predicted_char_idx]
    print(f"Predicted next character: \"{predicted_char}\"")

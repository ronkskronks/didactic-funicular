#!/bin/bash

# neural_network.sh
#
# A feed-forward neural network with a single hidden layer, implemented in Bash.
# This script trains a binary classifier using backpropagation.
# It relies heavily on `bc -l` for floating-point arithmetic, including the
# exponential function required for the sigmoid activation.
#
# WARNING: This is an educational example. It is extremely slow due to the
# overhead of calling `bc` for every single mathematical operation.
#
# Usage:
# ./neural_network.sh <path/to/data.csv> <hidden_neurons> <learning_rate> <epochs>
#
# Example:
# ./neural_network.sh nn_data.csv 5 0.1 1000
#
# The input CSV file should not have headers and must contain only numerical data.
# - The initial columns are the input features (x1, x2, ...).
# - The very last column is the class label, which must be either 0 or 1.
#
# The script does not produce any output to standard out.
# The final trained weights and biases are stored in variables but not displayed.

# --- Argument and File Validation ---

if [ "$#" -ne 4 ]; then
    exit 1 # Exit silently
fi

input_file="$1"
num_hidden_neurons="$2"
learning_rate="$3"
epochs="$4"

if [ ! -f "$input_file" ] || [ ! -r "$input_file" ]; then
    exit 1 # Exit silently
fi

# --- Math and Helper Functions ---

# Define the sigmoid activation function using bc.
# s(x) = 1 / (1 + e(-x))
function sigmoid() {
    echo "scale=10; 1 / (1 + e(-1 * $1))" | bc -l
}

# Define the derivative of the sigmoid function.
# s'(x) = s(x) * (1 - s(x))
function sigmoid_derivative() {
    local sig_val=$(sigmoid $1)
    echo "scale=10; $sig_val * (1 - $sig_val)" | bc -l
}

# Generate a random float between -0.5 and 0.5 for weight initialization.
function random_float() {
    # Using awk for a simple random number generation.
    awk 'BEGIN {srand(); print rand() - 0.5}'
}

# --- Initialization ---

# Get the number of input features from the data file.
num_input_features=$(head -n 1 "$input_file" | awk -F, '{print NF-1}')

# Initialize weights and biases with small random numbers.
# declare -A is used to create associative arrays for easier indexing.
declare -A hidden_weights hidden_bias output_weights

# Hidden layer weights and biases.
for ((h=0; h<num_hidden_neurons; h++)); do
    hidden_bias[$h]=$(random_float)
    for ((i=0; i<num_input_features; i++)); do
        hidden_weights[$h,$i]=$(random_float)
    done
done

# Output layer weights and bias.
# There is one output neuron for binary classification.
output_bias=$(random_float)
for ((h=0; h<num_hidden_neurons; h++)); do
    output_weights[$h]=$(random_float)
done

# --- Training Loop ---

for ((e=1; e<=epochs; e++)); do
    while IFS=, read -r -a fields; do
        # --- Data Preparation ---
        features=("${fields[@]:0:$num_input_features}")
        label=$(echo "${fields[-1]}" | tr -d '\r')

        # --- Forward Pass ---

        # 1. Hidden Layer Calculation
        declare -a hidden_layer_input hidden_layer_output
        for ((h=0; h<num_hidden_neurons; h++)); do
            # Start with the bias for this neuron.
            current_sum=${hidden_bias[$h]}
            for ((i=0; i<num_input_features; i++)); do
                weight=${hidden_weights[$h,$i]}
                feature=${features[i]}
                term=$(echo "scale=10; $weight * $feature" | bc -l)
                current_sum=$(echo "scale=10; $current_sum + $term" | bc -l)
            done
            hidden_layer_input[$h]=$current_sum
            hidden_layer_output[$h]=$(sigmoid $current_sum)
        done

        # 2. Output Layer Calculation
        output_layer_input=${output_bias}
        for ((h=0; h<num_hidden_neurons; h++)); do
            weight=${output_weights[$h]}
            h_output=${hidden_layer_output[$h]}
            term=$(echo "scale=10; $weight * $h_output" | bc -l)
            output_layer_input=$(echo "scale=10; $output_layer_input + $term" | bc -l)
        done
        final_output=$(sigmoid $output_layer_input)

        # --- Backward Pass (Backpropagation) ---

        # 1. Calculate Output Error and Delta
        error=$(echo "scale=10; $label - $final_output" | bc -l)
        output_delta=$(echo "scale=10; $error * $(sigmoid_derivative $output_layer_input)" | bc -l)

        # 2. Calculate Hidden Layer Errors and Deltas
        declare -a hidden_layer_error hidden_layer_delta
        for ((h=0; h<num_hidden_neurons; h++)); do
            h_error=$(echo "scale=10; $output_delta * ${output_weights[$h]}" | bc -l)
            hidden_layer_error[$h]=$h_error
            h_input=${hidden_layer_input[$h]}
            hidden_layer_delta[$h]=$(echo "scale=10; $h_error * $(sigmoid_derivative $h_input)" | bc -l)
        done

        # --- Weight and Bias Updates ---

        # 1. Update Output Layer
        output_bias=$(echo "scale=10; $output_bias + $learning_rate * $output_delta" | bc -l)
        for ((h=0; h<num_hidden_neurons; h++)); do
            h_output=${hidden_layer_output[$h]}
            update=$(echo "scale=10; $learning_rate * $output_delta * $h_output" | bc -l)
            output_weights[$h]=$(echo "scale=10; ${output_weights[$h]} + $update" | bc -l)
        done

        # 2. Update Hidden Layer
        for ((h=0; h<num_hidden_neurons; h++)); do
            h_delta=${hidden_layer_delta[$h]}
            hidden_bias[$h]=$(echo "scale=10; ${hidden_bias[$h]} + $learning_rate * $h_delta" | bc -l)
            for ((i=0; i<num_input_features; i++)); do
                feature=${features[i]}
                update=$(echo "scale=10; $learning_rate * $h_delta * $feature" | bc -l)
                hidden_weights[$h,$i]=$(echo "scale=10; ${hidden_weights[$h,$i]} + $update" | bc -l)
            done
        done

    done < "$input_file"
done

# --- Completion ---
# Training is complete. The final weights and biases are stored in the respective arrays.
# As per instructions, no output is produced.

exit 0

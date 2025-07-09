#!/bin/bash

# perceptron.sh
#
# A script to implement the Perceptron learning algorithm from scratch in Bash.
# It trains a linear binary classifier on a given dataset.
# The script uses `bc` for floating-point arithmetic.
#
# Usage:
# ./perceptron.sh <path/to/data.csv> <learning_rate> <epochs>
#
# Example:
# ./perceptron.sh data.csv 0.1 100
#
# The input CSV file should not have headers and must contain only numerical data.
# - The initial columns are the input features (x1, x2, ...).
# - The very last column is the class label, which must be either 1 or -1.
#
# The script does not produce any output to standard out.
# The final trained weights are stored in a variable but are not displayed.

# --- Argument and File Validation ---

# Silently exit if the wrong number of arguments is provided.
if [ "$#" -ne 3 ]; then
    exit 1
fi

input_file="$1"
learning_rate="$2"
epochs="$3"

# Silently exit if the input file is not readable.
if [ ! -f "$input_file" ] || [ ! -r "$input_file" ]; then
    exit 1
fi

# --- Initialization ---

# Determine the number of features from the first line of the data file.
# `awk` is used to count the fields (columns) and subtract one for the label.
num_features=$(head -n 1 "$input_file" | awk -F, '{print NF-1}')

# Initialize the weights array. There is one weight for each feature, plus one for the bias.
# The bias weight is stored at index 0.
# All weights are initialized to 0.0.
weights=()
for (( i=0; i<=num_features; i++ )); do
    weights[i]="0.0"
done

# --- Training Loop ---

# Loop for the specified number of epochs.
# An epoch is one full pass through the entire training dataset.
for (( e=1; e<=epochs; e++ )); do

    # Read the input file line by line for each epoch.
    # IFS (Internal Field Separator) is set to a comma to correctly parse the CSV.
    # `read -r` prevents backslash interpretation.
    # `read -a fields` reads the comma-separated values into an array named `fields`.
    while IFS=, read -r -a fields; do

        # Extract the label from the last element of the array.
        # `tr -d '\r'` is used to remove carriage returns for cross-platform compatibility.
        label=$(echo "${fields[-1]}" | tr -d '\r')

        # Extract the features from the beginning of the array.
        features=("${fields[@]:0:$num_features}")

        # --- Activation Calculation ---

        # Start the activation sum with the bias term (the first weight).
        activation="${weights[0]}"

        # Loop through each feature to calculate the weighted sum.
        for (( i=0; i<num_features; i++ )); do
            feature_val=${features[i]}
            weight_val=${weights[i+1]} # Weight index is i+1 because weights[0] is the bias.

            # Use `bc` to perform floating-point multiplication.
            term=$(echo "$weight_val * $feature_val" | bc)

            # Add the result to the activation sum.
            activation=$(echo "$activation + $term" | bc)
        done

        # --- Prediction ---

        # Apply a step function to the activation to get the prediction.
        # If activation is >= 0, predict class 1. Otherwise, predict class -1.
        prediction=0
        is_positive=$(echo "$activation >= 0" | bc) # `bc` returns 1 for true, 0 for false.
        if [ "$is_positive" -eq 1 ]; then
            prediction=1
        else
            prediction=-1
        fi

        # --- Weight Update ---

        # Update the weights only if the prediction was incorrect.
        if [ "$prediction" -ne "$label" ]; then

            # The update rule is: W_new = W_old + learning_rate * (label) * X
            # We calculate the update amount for the bias first.
            # For the bias, X is implicitly 1.
            update_val=$(echo "$learning_rate * $label" | bc)
            weights[0]=$(echo "${weights[0]} + $update_val" | bc)

            # Loop through the feature weights to update them.
            for (( i=0; i<num_features; i++ )); do
                feature_val=${features[i]}
                # Calculate the update amount for this specific weight.
                update_val=$(echo "$learning_rate * $label * $feature_val" | bc)
                # Apply the update.
                weights[i+1]=$(echo "${weights[i+1]} + $update_val" | bc)
            done
        fi
    done < "$input_file"
done

# --- Completion ---

# The training is complete. The final weights are stored in the `weights` array.
# For example, to see them, you could uncomment the following line:
# echo "Training complete. Final weights: ${weights[*]}"
#
# As per instructions, no output is produced.

exit 0

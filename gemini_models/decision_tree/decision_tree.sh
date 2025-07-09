#!/bin/bash

# decision_tree.sh
#
# A script to find the best initial split for a decision tree.
# It processes a CSV file with numerical features and a class label in the last column.
# The script calculates the Gini impurity for every possible split point in the data
# and identifies the feature and value that provide the best split (lowest Gini impurity).
#
# This represents the core logic of a single-level decision tree, or a "decision stump."
#
# Usage:
# ./decision_tree.sh <path/to/data.csv>
#
# The input CSV file should not have headers and should contain only numerical data.
# The last column is assumed to be the class label.
#
# The script does not produce any output to standard out.
# The best split feature and value are stored in variables but not displayed.

# Silently exit if the input file is not provided.
if [ "$#" -ne 1 ]; then
    exit 1
fi

input_file="$1"

# Silently exit if the input file is not readable.
if [ ! -f "$input_file" ] || [ ! -r "$input_file" ]; then
    exit 1
fi

# Use awk to find the best split. This is a complex task, so the awk script is substantial.
# The script is broken down into functions for clarity:
#   - gini(counts, total): Calculates the Gini impurity for a set of class counts.
#   - process_split(): Calculates the weighted Gini for the current split and updates the best split if necessary.
#
# The main part of the script iterates through each feature and each unique value in that feature
# to test it as a potential split point.

best_split_info=$(awk -F, '
# Function to calculate Gini impurity for a given distribution of classes.
# It takes an array of class counts and the total number of items.
function gini(counts, total,    i, sum_sq, p) {
    if (total == 0) {
        return 0;
    }
    sum_sq = 0;
    # for (i in counts) iterates over the keys (class labels) in the counts array.
    for (i in counts) {
        p = counts[i] / total;
        sum_sq += p * p;
    }
    return 1 - sum_sq;
}

# This block is executed for every line of the input file.
# It populates the `data` array and `labels` array.
# It also keeps track of the number of features and rows.
{
    # Store the label (last field) for the current row.
    labels[NR] = $NF;
    # Store the feature data for the current row.
    for (i = 1; i < NF; i++) {
        data[NR, i] = $i;
    }
    # Keep track of the number of rows.
    num_rows = NR;
    # Determine the number of features from the first row.
    if (NR == 1) {
        num_features = NF - 1;
    }
}

# The END block is executed after the entire file has been read.
# This is where the main logic for finding the best split resides.
END {
    # Initialize best_gini to a value higher than any possible Gini impurity.
    best_gini = 2;
    best_feature = -1;
    best_value = -1;

    # Iterate over each feature column.
    for (f = 1; f <= num_features; f++) {
        # Create a set of unique values for the current feature to test as split points.
        delete unique_values;
        for (r = 1; r <= num_rows; r++) {
            unique_values[data[r, f]] = 1;
        }

        # Iterate over each unique value as a potential split threshold.
        for (val in unique_values) {
            # Reset counts for the left and right sides of the split.
            delete left_counts;
            delete right_counts;
            left_total = 0;
            right_total = 0;

            # For the current split value, divide the data into two groups (left and right).
            for (r = 1; r <= num_rows; r++) {
                # If the feature value is less than or equal to the split value, it goes to the left group.
                if (data[r, f] <= val) {
                    left_counts[labels[r]]++;
                    left_total++;
                } else { # Otherwise, it goes to the right group.
                    right_counts[labels[r]]++;
                    right_total++;
                }
            }

            # Calculate the Gini impurity for both the left and right groups.
            gini_left = gini(left_counts, left_total);
            gini_right = gini(right_counts, right_total);

            # Calculate the weighted average of the Gini impurities.
            # This is the overall quality of the split.
            weighted_gini = (left_total / num_rows) * gini_left + (right_total / num_rows) * gini_right;

            # If the current split is better than the best one found so far, update the best split information.
            if (weighted_gini < best_gini) {
                best_gini = weighted_gini;
                best_feature = f;
                best_value = val;
            }
        }
    }

    # After checking all features and all possible splits, print the best one found.
    # This output will be captured by the shell variable `best_split_info`.
    print best_feature, best_value, best_gini;
}' "$input_file")

# Read the results from awk into shell variables.
read best_feature best_value best_gini <<< "$best_split_info"

# The script has now found the best feature and value to split the data on.
# These are stored in the variables $best_feature and $best_value.
# As per the instructions, no output is displayed.

# To use these results, one might, for example, print them:
#
# echo "Best split is on feature $best_feature at value $best_value with Gini impurity $best_gini"
#
# This line is commented out to ensure silent operation.

exit 0

#!/bin/bash

# linear_regression.sh
#
# A script to perform simple linear regression on a two-column CSV file.
# This script calculates the slope (m) and y-intercept (b) of the line
# that best fits the data, using the least squares method.
#
# Usage:
# ./linear_regression.sh <path/to/data.csv>
#
# The input CSV file should not have headers and contain two columns:
# Column 1: Independent variable (x)
# Column 2: Dependent variable (y)
#
# The script does not produce any output to standard out.
# The calculated values for slope and intercept are stored in variables
# but are not displayed, adhering to the operational constraints.

# Check if an input file is provided.
# If not, the script will silently exit.
if [ "$#" -ne 1 ]; then
    # No error message is printed to stderr to maintain silence.
    exit 1
fi

# Assign the input file path to a variable for clarity.
input_file="$1"

# Check if the input file exists and is readable.
# If not, the script will silently exit.
if [ ! -f "$input_file" ] || [ ! -r "$input_file" ]; then
    # No error message is printed.
    exit 1
fi

# Use awk to calculate the necessary sums for linear regression.
# awk is used for its robust floating-point arithmetic capabilities.
#
# The script calculates:
#   - N: The total number of data points.
#   - sum_x: The sum of all x values.
#   - sum_y: The sum of all y values.
#   - sum_xy: The sum of the product of each x and y pair.
#   - sum_x_sq: The sum of the squares of all x values.
#
# FS="," sets the field separator to a comma for CSV parsing.
# The END block is executed after all lines of the file have been processed.
sums=$(awk '
BEGIN {
    FS=",";
    # Initialize all sum variables and the count to zero.
    N = 0;
    sum_x = 0;
    sum_y = 0;
    sum_xy = 0;
    sum_x_sq = 0;
}
{
    # For each line in the file:
    # Increment the data point counter.
    N++;
    # Add the first field ($1) to sum_x.
    sum_x += $1;
    # Add the second field ($2) to sum_y.
    sum_y += $2;
    # Add the product of the fields to sum_xy.
    sum_xy += $1 * $2;
    # Add the square of the first field to sum_x_sq.
    sum_x_sq += $1 * $1;
}
END {
    # After processing all lines, print the calculated values.
    # These will be captured by the shell variable "sums".
    print N, sum_x, sum_y, sum_xy, sum_x_sq;
}' "$input_file")

# Read the calculated sums from the awk output into separate shell variables.
read N sum_x sum_y sum_xy sum_x_sq <<< "$sums"

# Check if N is zero to prevent division by zero errors.
# This would happen if the input file was empty.
if [ "$N" -eq 0 ]; then
    # Silently exit if there is no data.
    exit 1
fi

# Calculate the slope (m) and y-intercept (b) using the standard formulas.
# Another awk command is used for this calculation to ensure floating-point precision.
#
# Formula for slope (m):
# m = (N * Σ(xy) - Σx * Σy) / (N * Σ(x^2) - (Σx)^2)
#
# Formula for y-intercept (b):
# b = (Σy - m * Σx) / N
#
# The -v options pass shell variables into the awk script.
results=$(awk -v N="$N" -v sum_x="$sum_x" -v sum_y="$sum_y" -v sum_xy="$sum_xy" -v sum_x_sq="$sum_x_sq" '
BEGIN {
    # Calculate the denominator for the slope formula first.
    denominator = (N * sum_x_sq) - (sum_x * sum_x);

    # Avoid division by zero if all x values are the same.
    if (denominator == 0) {
        # In this case, the line is vertical, and the slope is undefined.
        # We will not print anything and let the script exit silently.
        exit;
    }

    # Calculate slope (m).
    m = ((N * sum_xy) - (sum_x * sum_y)) / denominator;

    # Calculate y-intercept (b).
    b = (sum_y - (m * sum_x)) / N;

    # Print the final results.
    print m, b;
}')

# Read the final results into shell variables.
# If the calculation failed (e.g., division by zero), these will be empty.
read slope intercept <<< "$results"

# The script has now calculated the slope and intercept.
# The values are stored in the "slope" and "intercept" variables.
# Per instructions, no output is displayed on the screen.

# To use the results, one would access the variables $slope and $intercept
# in a calling script or later in this script. For example:
#
# echo "Slope (m): $slope"
# echo "Intercept (b): $intercept"
#
# But these lines are commented out to ensure no output is produced.

# The script finishes execution here, silently.
exit 0

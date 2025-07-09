#!/bin/bash

# decision_table_trainer_v2.sh
#
# This script generates a decision table model by extracting a large set of features
# from 'diabetic_data.csv' and counting the number of unique combinations of these features.
# Each unique combination represents a 'parameter' in this model.
#
# Usage:
# ./decision_table_trainer_v2.sh <path/to/diabetic_data.csv> <output_model_unique_features.csv>
#
# The script will output a CSV file where each row is a unique feature combination.
# No output is printed to stdout during execution.

# --- Configuration ---

# Column indices for a broad set of features to maximize unique combinations.
# These are 1-based indices from the diabetic_data.csv file.
COL_FEATURES="3,4,5,6,7,8,11,12,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,48,49"

# --- Argument Validation ---

if [ "$#" -ne 2 ]; then
    exit 1 # Silent exit
fi

INPUT_FILE="$1"
OUTPUT_MODEL="$2"

if [ ! -f "$INPUT_FILE" ] || [ ! -r "$INPUT_FILE" ]; then
    exit 1 # Silent exit
fi

# --- Data Processing and Unique Combination Generation ---

# 1. Extract relevant columns and remove the header.
# 2. Clean '?' values to 'UNKNOWN'.
# 3. Remove carriage returns (\r) to ensure consistent line endings for sorting.
# 4. Replace commas with a consistent delimiter '|' for easier processing.
# 5. Sort the lines to group identical feature combinations together.
# 6. Use 'uniq' to filter out duplicate lines, leaving only unique feature combinations.

cat "$INPUT_FILE" \
| tail -n +2 \
| cut -d, -f${COL_FEATURES} \
| sed 's/\?/UNKNOWN/g' \
| tr -d '\r' \
| tr ',' '|' \
| sort \
| uniq > "$OUTPUT_MODEL"

exit 0

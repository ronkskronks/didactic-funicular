#!/bin/bash

# decision_table_trainer.sh
#
# Implements a Decision Table Classifier using Bash utilities.
# This script processes the 'diabetic_data.csv' to create a lookup table
# (the 'model') where unique combinations of selected features are mapped
# to a predicted outcome (readmission status).
#
# The 'parameters' of this model are the number of unique rules (rows)
# in the generated decision table.
#
# Usage:
# ./decision_table_trainer.sh <path/to/diabetic_data.csv> <output_model.csv>
#
# The script will output the decision table to the specified output file.
# No output is printed to stdout during execution.

# --- Configuration ---

# Column indices for selected features and the target variable.
# These are 1-based indices from the diabetic_data.csv file.
# Increased number of features to generate more unique rules.
COL_RACE=3
COL_GENDER=4
COL_AGE=5
COL_ADM_TYPE=6
COL_DIS_DISP=7
COL_ADM_SRC=8
COL_PAYER_CODE=11
COL_MEDICAL_SPECIALTY=12
COL_DIAG1=19
COL_DIAG2=20
COL_DIAG3=21

# Drug columns (from metformin to metformin-pioglitazone)
COL_DRUGS="24-47"

COL_CHANGE=48
COL_DIABETES_MED=49
COL_READMITTED=50

# --- Argument Validation ---

if [ "$#" -ne 2 ]; then
    exit 1 # Silent exit
fi

INPUT_FILE="$1"
OUTPUT_MODEL="$2"

if [ ! -f "$INPUT_FILE" ] || [ ! -r "$INPUT_FILE" ]; then
    exit 1 # Silent exit
fi

# --- Data Processing and Rule Generation ---

# 1. Extract relevant columns and remove the header.
# 2. Clean '?' values to 'UNKNOWN'.
# 3. Convert 'readmitted' column to binary: 'NO' -> 0, '<30' or '>30' -> 1.
# 4. Concatenate features and the binary label with a unique delimiter '|'.
# 5. Sort the concatenated strings and count unique occurrences.
# 6. Use awk to find the majority class for each unique feature combination.
#    The awk script groups by feature combination and counts occurrences of each label.
#    Finally, it prints the feature combination and the label with the highest count.

# The entire processing pipeline is enclosed in a single command to minimize intermediate files.
cat "$INPUT_FILE" \
| tail -n +2 \
| cut -d, -f${COL_RACE},${COL_GENDER},${COL_AGE},${COL_ADM_TYPE},${COL_DIS_DISP},${COL_ADM_SRC},${COL_PAYER_CODE},${COL_MEDICAL_SPECIALTY},${COL_DIAG1},${COL_DIAG2},${COL_DIAG3},${COL_DRUGS},${COL_CHANGE},${COL_DIABETES_MED},${COL_READMITTED} \
| sed 's/\?/UNKNOWN/g' \
| sed 's/NO$/0/' \
| sed 's/>30$/1/' \
| sed 's/<30$/1/' \
| tr ',' '|' \
| sort \
| uniq -c \
| awk -F'|' '{
    count = $1;
    # Remove the count from the first field to get the feature string.
    sub(/^[0-9]+ /, "", $1);
    feature_str = substr($1, 1, length($1)-1); # Remove the last char which is the label
    label = substr($1, length($1)); # Get the last char which is the label

    # Store counts for each label for a given feature_str
    counts[feature_str, label] += count;
    # Keep track of all labels seen for this feature_str
    if (!(feature_str in labels_seen)) {
        labels_seen[feature_str] = label;
    } else {
        labels_seen[feature_str] = labels_seen[feature_str] " " label;
    }
}
END {
    # Iterate through each unique feature string
    for (fs in labels_seen) {
        max_count = -1;
        best_label = "";
        # Split the space-separated labels seen for this feature string
        split(labels_seen[fs], current_labels, " ");
        # Find the label with the maximum count
        for (i in current_labels) {
            l = current_labels[i];
            if (counts[fs, l] > max_count) {
                max_count = counts[fs, l];
                best_label = l;
            }
        }
        # Print the feature string and its majority label
        print fs "," best_label;
    }
}' > "$OUTPUT_MODEL"

exit 0
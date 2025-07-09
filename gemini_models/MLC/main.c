/*
 * main.c
 *
 * Main entry point for the C-based neural network.
 * This version is specifically tailored to process 'diabetic_data.csv'.
 * It handles data loading, preprocessing (basic categorical encoding, missing values),
 * and trains a large MLP with a fixed architecture to achieve ~1M parameters.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> // For access() to check file existence.
#include <ctype.h> // For isdigit
#include "neural_network.h"

#define MAX_LINE_LENGTH 1024 * 10 // Increased for longer lines in diabetic_data.csv
#define MAX_SAMPLES 101766 // Max samples in diabetic_data.csv
#define MAX_FEATURES 100 // Max possible features after preprocessing
#define MAX_UNIQUE_VAL_LEN 128 // Max length for a unique categorical value string

// --- Helper Functions ---

// Function to get the index of a column by its name
int get_column_index(const char* header_line, const char* col_name) {
    char header_copy[MAX_LINE_LENGTH];
    strcpy(header_copy, header_line);
    char* token = strtok(header_copy, ",");
    int index = 0;
    while (token != NULL) {
        // Remove potential carriage return from token
        token[strcspn(token, "\r\n")] = 0;
        if (strcmp(token, col_name) == 0) {
            return index;
        }
        token = strtok(NULL, ",");
        index++;
    }
    return -1; // Not found
}

// Load the dataset from a CSV file, specifically for diabetic_data.csv
int load_diabetic_data(
    const char* filename,
    double** data,
    double** labels,
    int* actual_num_inputs // Output: actual number of input features after preprocessing
) {
    FILE* file = fopen(filename, "r");
    if (!file) { perror("Error opening data file"); exit(EXIT_FAILURE); }

    char header_line[MAX_LINE_LENGTH];
    fgets(header_line, sizeof(header_line), file); // Read header

    // Get column indices for relevant columns
    int col_readmitted = get_column_index(header_line, "readmitted");
    int col_encounter_id = get_column_index(header_line, "encounter_id");
    int col_patient_nbr = get_column_index(header_line, "patient_nbr");

    // Map for categorical features to integer IDs
    char unique_values[MAX_FEATURES][MAX_SAMPLES][MAX_UNIQUE_VAL_LEN]; // Store unique strings for each column
    int unique_counts[MAX_FEATURES]; // Count of unique strings for each column
    for(int i=0; i<MAX_FEATURES; ++i) unique_counts[i] = 0;

    // First pass: collect unique categorical values to assign IDs
    fseek(file, 0, SEEK_SET); // Reset file pointer
    fgets(header_line, sizeof(header_line), file); // Skip header again

    char line[MAX_LINE_LENGTH];
    int current_sample_idx = 0;
    while (fgets(line, sizeof(line), file) && current_sample_idx < MAX_SAMPLES) {
        char line_copy[MAX_LINE_LENGTH];
        strcpy(line_copy, line);
        char* token = strtok(line_copy, ",");
        int col_idx = 0;
        while (token != NULL) {
            token[strcspn(token, "\r\n")] = 0; // Remove newline

            // Skip encounter_id, patient_nbr, and readmitted for feature processing
            if (col_idx == col_encounter_id || col_idx == col_patient_nbr || col_idx == col_readmitted) {
                token = strtok(NULL, ",");
                col_idx++;
                continue;
            }

            // Check if numerical (simplified: if it contains only digits and a dot)
            int is_numerical = 1;
            if (strcmp(token, "?") == 0) { // Treat '?' as non-numerical for unique value collection
                is_numerical = 0;
            } else {
                for(int i=0; token[i] != '\0'; ++i) {
                    if (!isdigit(token[i]) && token[i] != '.' && token[i] != '-') {
                        is_numerical = 0;
                        break;
                    }
                }
            }

            if (!is_numerical) {
                // Store unique categorical values
                int found = 0;
                for(int i=0; i<unique_counts[col_idx]; ++i) {
                    if (strcmp(unique_values[col_idx][i], token) == 0) {
                        found = 1;
                        break;
                    }
                }
                if (!found && unique_counts[col_idx] < MAX_SAMPLES) { // Prevent overflow
                    strncpy(unique_values[col_idx][unique_counts[col_idx]], token, MAX_UNIQUE_VAL_LEN - 1);
                    unique_values[col_idx][unique_counts[col_idx]][MAX_UNIQUE_VAL_LEN - 1] = '\0';
                    unique_counts[col_idx]++;
                }
            }
            token = strtok(NULL, ",");
            col_idx++;
        }
        current_sample_idx++;
    }

    // Second pass: load data and convert categorical to numerical IDs
    fseek(file, 0, SEEK_SET); // Reset file pointer
    fgets(header_line, sizeof(header_line), file); // Skip header again

    int sample_count = 0;
    int current_num_inputs = 0; // To count actual input features for the first sample

    while (fgets(line, sizeof(line), file) && sample_count < MAX_SAMPLES) {
        char line_copy[MAX_LINE_LENGTH];
        strcpy(line_copy, line);
        char* token = strtok(line_copy, ",");
        int col_idx = 0;
        int feature_idx = 0;
        char readmitted_val[MAX_UNIQUE_VAL_LEN];
        readmitted_val[0] = '\0';

        while (token != NULL) {
            token[strcspn(token, "\r\n")] = 0; // Remove newline

            if (col_idx == col_encounter_id || col_idx == col_patient_nbr) {
                // Skip these columns
            } else if (col_idx == col_readmitted) {
                strncpy(readmitted_val, token, MAX_UNIQUE_VAL_LEN - 1);
                readmitted_val[MAX_UNIQUE_VAL_LEN - 1] = '\0';
            } else { // This is a feature column
                // Check if numerical
                int is_numerical = 1;
                if (strcmp(token, "?") == 0) {
                    is_numerical = 0;
                } else {
                    for(int i=0; token[i] != '\0'; ++i) {
                        if (!isdigit(token[i]) && token[i] != '.' && token[i] != '-') {
                            is_numerical = 0;
                            break;
                        }
                    }
                }

                if (is_numerical) {
                    data[sample_count][feature_idx] = atof(token);
                } else { // Categorical or '?'
                    // Assign integer ID based on unique_values mapping
                    int id = -1;
                    for(int i=0; i<unique_counts[col_idx]; ++i) {
                        if (strcmp(unique_values[col_idx][i], token) == 0) {
                            id = i;
                            break;
                        }
                    }
                    data[sample_count][feature_idx] = (double)id; // Use ID as feature value
                }
                feature_idx++;
            }
            token = strtok(NULL, ",");
            col_idx++;
        }

        // Set actual_num_inputs only once after processing the first sample
        if (sample_count == 0) {
            *actual_num_inputs = feature_idx;
        }

        // Process readmitted label
        if (readmitted_val[0] != '\0') {
            if (strcmp(readmitted_val, "NO") == 0) {
                labels[sample_count][0] = 1.0; // Class 0
                labels[sample_count][1] = 0.0;
            } else {
                labels[sample_count][0] = 0.0;
                labels[sample_count][1] = 1.0; // Class 1
            }
        }
        sample_count++;
    }

    fclose(file);

    return sample_count;
}

// --- Main Function ---

int main(int argc, char* argv[]) {
    // Usage: ./trainer <model_file> <data_file> <learning_rate> <epochs>
    if (argc != 5) {
        return 1; // Silent exit
    }

    const char* model_filename = argv[1];
    const char* data_filename = argv[2];
    double learning_rate = atof(argv[3]);
    int epochs = atoi(argv[4]);

    // --- Data Loading and Preprocessing ---
    double** training_data = (double**)malloc(MAX_SAMPLES * sizeof(double*));
    double** training_labels = (double**)malloc(MAX_SAMPLES * sizeof(double*));
    for (int i = 0; i < MAX_SAMPLES; i++) {
        training_data[i] = (double*)malloc(MAX_FEATURES * sizeof(double)); // Max possible features
        training_labels[i] = (double*)malloc(2 * sizeof(double)); // 2 output classes
    }

    int num_inputs_actual = 0;
    int num_samples = load_diabetic_data(data_filename, training_data, training_labels, &num_inputs_actual);
    int num_outputs = 2; // Fixed for binary classification

    // --- Network Architecture Setup ---
    // Hardcode for ~1M parameters: Input -> 20000 Hidden -> Output
    int num_total_layers = 3; // Input + Hidden + Output
    int* layer_sizes = (int*)malloc(num_total_layers * sizeof(int));
    layer_sizes[0] = num_inputs_actual;
    layer_sizes[1] = 20000; // Hidden layer size for ~1M parameters
    layer_sizes[2] = num_outputs;

    // --- Network Loading or Creation ---
    NeuralNetwork* network;
    if (access(model_filename, F_OK) != -1) {
        network = load_network(model_filename);
        if (!network) { return 1; }
    } else {
        network = create_network(num_total_layers, layer_sizes);
    }

    // --- Training ---
    train_network(network, training_data, training_labels, num_samples, epochs, learning_rate);

    // --- Saving and Cleanup ---
    save_network(network, model_filename);

    free_network(network);
    free(layer_sizes);
    for (int i = 0; i < MAX_SAMPLES; i++) {
        free(training_data[i]);
        free(training_labels[i]);
    }
    free(training_data);
    free(training_labels);

    return 0;
}
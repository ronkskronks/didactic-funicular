/*
 * neural_network.c
 *
 * Core implementation of a multi-layer perceptron in C.
 * Contains the logic for forward propagation, backpropagation, weight updates,
 * and saving/loading the model to/from a file.
 */

#include "neural_network.h"
#include <math.h>
#include <time.h>
#include <string.h>

// --- Private Helper Functions ---

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

double random_weight() {
    return ((double)rand() / RAND_MAX) - 0.5;
}

// --- Public Function Implementations ---

NeuralNetwork* create_network(int num_layers, const int* layer_sizes) {
    srand(time(NULL));
    NeuralNetwork* network = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    network->num_layers = num_layers;
    network->layers = (Layer*)malloc(num_layers * sizeof(Layer));

    for (int i = 0; i < num_layers; i++) {
        Layer* layer = &network->layers[i];
        layer->num_neurons = layer_sizes[i];
        layer->input_size = (i == 0) ? 0 : layer_sizes[i - 1];
        layer->outputs = (double*)malloc(layer->num_neurons * sizeof(double));
        layer->biases = (double*)malloc(layer->num_neurons * sizeof(double));
        layer->deltas = (double*)malloc(layer->num_neurons * sizeof(double));

        if (i == 0) {
            layer->weights = NULL;
            continue;
        }

        layer->weights = (double*)malloc(layer->num_neurons * layer->input_size * sizeof(double));
        for (int j = 0; j < layer->num_neurons; j++) {
            layer->biases[j] = random_weight();
            for (int k = 0; k < layer->input_size; k++) {
                layer->weights[j * layer->input_size + k] = random_weight();
            }
        }
    }
    return network;
}

void free_network(NeuralNetwork* network) {
    for (int i = 0; i < network->num_layers; i++) {
        free(network->layers[i].outputs);
        free(network->layers[i].biases);
        free(network->layers[i].deltas);
        if (network->layers[i].weights != NULL) {
            free(network->layers[i].weights);
        }
    }
    free(network->layers);
    free(network);
}

void forward_pass(NeuralNetwork* network, const double* inputs) {
    double* current_inputs = (double*)inputs;
    for (int i = 0; i < network->num_layers; i++) {
        Layer* layer = &network->layers[i];
        if (i == 0) {
            memcpy(layer->outputs, current_inputs, layer->num_neurons * sizeof(double));
            continue;
        }
        for (int j = 0; j < layer->num_neurons; j++) {
            double activation = layer->biases[j];
            for (int k = 0; k < layer->input_size; k++) {
                activation += current_inputs[k] * layer->weights[j * layer->input_size + k];
            }
            layer->outputs[j] = sigmoid(activation);
        }
        current_inputs = layer->outputs;
    }
}

void backpropagate(NeuralNetwork* network, const double* expected) {
    for (int i = network->num_layers - 1; i > 0; i--) {
        Layer* layer = &network->layers[i];
        double* errors = (double*)malloc(layer->num_neurons * sizeof(double));
        if (i == network->num_layers - 1) {
            for (int j = 0; j < layer->num_neurons; j++) {
                errors[j] = expected[j] - layer->outputs[j];
            }
        } else {
            Layer* next_layer = &network->layers[i + 1];
            for (int j = 0; j < layer->num_neurons; j++) {
                errors[j] = 0.0;
                for (int k = 0; k < next_layer->num_neurons; k++) {
                    errors[j] += next_layer->weights[k * next_layer->input_size + j] * next_layer->deltas[k];
                }
            }
        }
        for (int j = 0; j < layer->num_neurons; j++) {
            double output = layer->outputs[j];
            output = fmax(0.000001, fmin(0.999999, output));
            double logit_val = log(output / (1.0 - output));
            layer->deltas[j] = errors[j] * sigmoid_derivative(logit_val);
        }
        free(errors);
    }
}

void update_weights(NeuralNetwork* network, const double* inputs, double learning_rate) {
    for (int i = 1; i < network->num_layers; i++) {
        Layer* layer = &network->layers[i];
        const double* prev_layer_outputs = (i == 1) ? inputs : network->layers[i - 1].outputs;
        for (int j = 0; j < layer->num_neurons; j++) {
            layer->biases[j] += learning_rate * layer->deltas[j];
            for (int k = 0; k < layer->input_size; k++) {
                layer->weights[j * layer->input_size + k] += learning_rate * layer->deltas[j] * prev_layer_outputs[k];
            }
        }
    }
}

void train_network(NeuralNetwork* network, double** training_data, double** training_labels, int num_samples, int epochs, double learning_rate) {
    for (int e = 0; e < epochs; e++) {
        for (int s = 0; s < num_samples; s++) {
            forward_pass(network, training_data[s]);
            backpropagate(network, training_labels[s]);
            update_weights(network, training_data[s], learning_rate);
        }
    }
}

// Saves the network's architecture and weights to a file.
int save_network(const NeuralNetwork* network, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) return -1; // Failure

    // Write the number of layers.
    fwrite(&network->num_layers, sizeof(int), 1, file);

    // Write the size of each layer.
    for (int i = 0; i < network->num_layers; i++) {
        fwrite(&network->layers[i].num_neurons, sizeof(int), 1, file);
    }

    // Write the weights and biases for each layer (except input layer).
    for (int i = 1; i < network->num_layers; i++) {
        Layer* layer = &network->layers[i];
        fwrite(layer->biases, sizeof(double), layer->num_neurons, file);
        fwrite(layer->weights, sizeof(double), layer->num_neurons * layer->input_size, file);
    }

    fclose(file);
    return 0; // Success
}

// Loads a network from a file.
NeuralNetwork* load_network(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL; // Failure

    // Read the number of layers.
    int num_layers;
    fread(&num_layers, sizeof(int), 1, file);

    // Read the layer sizes and create a new network.
    int* layer_sizes = (int*)malloc(num_layers * sizeof(int));
    for (int i = 0; i < num_layers; i++) {
        fread(&layer_sizes[i], sizeof(int), 1, file);
    }
    NeuralNetwork* network = create_network(num_layers, layer_sizes);
    free(layer_sizes);

    // Read the weights and biases.
    for (int i = 1; i < num_layers; i++) {
        Layer* layer = &network->layers[i];
        fread(layer->biases, sizeof(double), layer->num_neurons, file);
        fread(layer->weights, sizeof(double), layer->num_neurons * layer->input_size, file);
    }

    fclose(file);
    return network;
}
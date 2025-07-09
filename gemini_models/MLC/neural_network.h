/*
 * neural_network.h
 *
 * Header file for a multi-layer perceptron (MLP) in C.
 * Defines the structures for the network, layers, and the function prototypes
 * for creating, training, and managing the neural network.
 */

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include <stdio.h>

// --- Data Structures ---

// Represents a single layer in the neural network.
typedef struct {
    double* weights;    // 2D matrix (rows=num_neurons, cols=input_size) flattened into a 1D array.
    double* biases;     // Array of biases for each neuron in this layer.
    double* outputs;    // The output of each neuron after activation (the "activations").
    double* deltas;     // The error signal (delta) for each neuron, used in backpropagation.
    int num_neurons;    // Number of neurons in this layer.
    int input_size;     // The number of inputs to this layer (i.e., neurons in the previous layer).
} Layer;

// Represents the entire neural network.
typedef struct {
    Layer* layers;      // An array of all layers in the network.
    int num_layers;     // The total number of layers (including input, hidden, and output).
} NeuralNetwork;

// --- Function Prototypes ---

/**
 * @brief Creates and initializes a new neural network.
 *
 * @param num_layers The total number of layers.
 * @param layer_sizes An array containing the number of neurons in each layer.
 * @return A pointer to the newly created NeuralNetwork.
 */
NeuralNetwork* create_network(int num_layers, const int* layer_sizes);

/**
 * @brief Frees all memory allocated for the neural network.
 *
 * @param network A pointer to the NeuralNetwork to be freed.
 */
void free_network(NeuralNetwork* network);

/**
 * @brief Performs a forward pass through the entire network.
 *
 * @param network The neural network.
 * @param inputs The input data for the first layer.
 */
void forward_pass(NeuralNetwork* network, const double* inputs);

/**
 * @brief Performs backpropagation to calculate error gradients.
 *
 * @param network The neural network.
 * @param expected The expected output values (the ground truth).
 */
void backpropagate(NeuralNetwork* network, const double* expected);

/**
 * @brief Updates the weights and biases of the network using the calculated gradients.
 *
 * @param network The neural network.
 * @param inputs The original input data for this training sample.
 * @param learning_rate The learning rate to apply to the updates.
 */
void update_weights(NeuralNetwork* network, const double* inputs, double learning_rate);

/**
 * @brief Trains the neural network on a given dataset for a specified number of epochs.
 *
 * @param network The neural network to train.
 * @param training_data A 2D array of training samples.
 * @param training_labels A 2D array of corresponding labels.
 * @param num_samples The total number of training samples.
 * @param epochs The number of training iterations.
 * @param learning_rate The learning rate.
 */
void train_network(NeuralNetwork* network, double** training_data, double** training_labels, int num_samples, int epochs, double learning_rate);

/**
 * @brief Saves the entire network architecture and weights to a binary file.
 *
 * @param network The neural network to save.
 * @param filename The path to the file where the model will be saved.
 * @return 0 on success, -1 on failure.
 */
int save_network(const NeuralNetwork* network, const char* filename);

/**
 * @brief Loads a neural network from a binary file.
 *
 * This function allocates a new network and populates it with the data from the file.
 *
 * @param filename The path to the file from which to load the model.
 * @return A pointer to the newly loaded NeuralNetwork, or NULL on failure.
 */
NeuralNetwork* load_network(const char* filename);

#endif // NEURAL_NETWORK_H
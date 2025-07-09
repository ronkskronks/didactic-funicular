#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    double weights[4];  // 2 inputs + 1 hidden + 1 bias = tiny network
    double bias;
} TinyNN;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double forward(TinyNN* nn, double x1, double x2) {
    double hidden = sigmoid(nn->weights[0] * x1 + nn->weights[1] * x2 + nn->bias);
    double output = sigmoid(nn->weights[2] * hidden + nn->weights[3]);
    return output;
}

int main() {
    TinyNN nn = {{0.5, -0.3, 0.8, 0.2}, 0.1};
    
    printf("🤖 Tiny Neural Network (5 parameters total)\n");
    printf("Testing XOR-like function:\n");
    printf("Input (0,0): %.3f\n", forward(&nn, 0.0, 0.0));
    printf("Input (0,1): %.3f\n", forward(&nn, 0.0, 1.0));
    printf("Input (1,0): %.3f\n", forward(&nn, 1.0, 0.0));
    printf("Input (1,1): %.3f\n", forward(&nn, 1.1, 1.0));
    
    return 0;
}
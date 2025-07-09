#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Unified Trainable GPT Implementation
#define MAX_VOCAB 100
#define CONTEXT_LEN 8
#define D_MODEL 16
#define N_HEADS 2
#define N_LAYERS 1
#define D_FF 32
#define LEARNING_RATE 0.01

// Tokenizer
typedef struct {
    char words[MAX_VOCAB][50];
    int word_count;
} Tokenizer;

// GPT Model
typedef struct {
    double token_embed[MAX_VOCAB][D_MODEL];
    double pos_embed[CONTEXT_LEN][D_MODEL];
    double wq[D_MODEL][D_MODEL];
    double wk[D_MODEL][D_MODEL];
    double wv[D_MODEL][D_MODEL];
    double wo[D_MODEL][D_MODEL];
    double w1[D_MODEL][D_FF];
    double w2[D_FF][D_MODEL];
    double output_proj[D_MODEL][MAX_VOCAB];
} TrainableGPT;

// Training example
typedef struct {
    int* tokens;
    int length;
} TrainingExample;

// Initialize tokenizer
void init_tokenizer(Tokenizer* tok) {
    tok->word_count = 0;
}

// Add word to vocabulary
int add_word(Tokenizer* tok, char* word) {
    for (int i = 0; i < tok->word_count; i++) {
        if (strcmp(tok->words[i], word) == 0) {
            return i;
        }
    }
    
    if (tok->word_count < MAX_VOCAB) {
        strcpy(tok->words[tok->word_count], word);
        return tok->word_count++;
    }
    return -1;
}

// Tokenize text
int tokenize_text(Tokenizer* tok, char* text, int* tokens) {
    char* text_copy = malloc(strlen(text) + 1);
    strcpy(text_copy, text);
    
    int token_count = 0;
    char* word = strtok(text_copy, " \n\t.,!?;");
    
    while (word != NULL && token_count < CONTEXT_LEN) {
        // Convert to lowercase
        for (int i = 0; word[i]; i++) {
            if (word[i] >= 'A' && word[i] <= 'Z') {
                word[i] = word[i] + 32;
            }
        }
        
        int token_id = add_word(tok, word);
        if (token_id >= 0) {
            tokens[token_count++] = token_id;
        }
        
        word = strtok(NULL, " \n\t.,!?;");
    }
    
    free(text_copy);
    return token_count;
}

// Initialize GPT model
void init_gpt(TrainableGPT* gpt) {
    srand(time(NULL));
    
    // Initialize all weights with small random values
    for (int i = 0; i < MAX_VOCAB; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            gpt->token_embed[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
    
    for (int i = 0; i < CONTEXT_LEN; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            gpt->pos_embed[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
    
    for (int i = 0; i < D_MODEL; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            gpt->wq[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            gpt->wk[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            gpt->wv[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            gpt->wo[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
    
    for (int i = 0; i < D_MODEL; i++) {
        for (int j = 0; j < D_FF; j++) {
            gpt->w1[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
    
    for (int i = 0; i < D_FF; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            gpt->w2[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
    
    for (int i = 0; i < D_MODEL; i++) {
        for (int j = 0; j < MAX_VOCAB; j++) {
            gpt->output_proj[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
    
    printf("GPT model initialized with %d parameters\n", 
           MAX_VOCAB * D_MODEL + CONTEXT_LEN * D_MODEL + 
           4 * D_MODEL * D_MODEL + D_MODEL * D_FF + D_FF * D_MODEL + 
           D_MODEL * MAX_VOCAB);
}

// Simple ReLU activation
double relu(double x) {
    return x > 0 ? x : 0;
}

// Forward pass through model
void forward_pass(TrainableGPT* gpt, int* tokens, int len, double logits[MAX_VOCAB]) {
    double embeddings[CONTEXT_LEN][D_MODEL];
    double attn_out[CONTEXT_LEN][D_MODEL];
    double ff_out[CONTEXT_LEN][D_MODEL];
    
    // Get embeddings
    for (int pos = 0; pos < len && pos < CONTEXT_LEN; pos++) {
        for (int d = 0; d < D_MODEL; d++) {
            embeddings[pos][d] = gpt->token_embed[tokens[pos]][d] + gpt->pos_embed[pos][d];
        }
    }
    
    // Simple attention (just use values for now)
    for (int pos = 0; pos < len; pos++) {
        for (int d = 0; d < D_MODEL; d++) {
            double val = 0;
            for (int k = 0; k < D_MODEL; k++) {
                val += embeddings[pos][k] * gpt->wv[k][d];
            }
            attn_out[pos][d] = val + embeddings[pos][d]; // residual
        }
    }
    
    // Feed forward
    for (int pos = 0; pos < len; pos++) {
        double hidden[D_FF];
        
        // First layer
        for (int h = 0; h < D_FF; h++) {
            hidden[h] = 0;
            for (int d = 0; d < D_MODEL; d++) {
                hidden[h] += attn_out[pos][d] * gpt->w1[d][h];
            }
            hidden[h] = relu(hidden[h]);
        }
        
        // Second layer
        for (int d = 0; d < D_MODEL; d++) {
            ff_out[pos][d] = 0;
            for (int h = 0; h < D_FF; h++) {
                ff_out[pos][d] += hidden[h] * gpt->w2[h][d];
            }
            ff_out[pos][d] += attn_out[pos][d]; // residual
        }
    }
    
    // Output projection (use last token)
    int last_pos = len - 1;
    for (int v = 0; v < MAX_VOCAB; v++) {
        logits[v] = 0;
        for (int d = 0; d < D_MODEL; d++) {
            logits[v] += ff_out[last_pos][d] * gpt->output_proj[d][v];
        }
    }
}

// Calculate cross-entropy loss
double calculate_loss(double* logits, int target, int vocab_size) {
    double max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    
    double sum = 0;
    for (int i = 0; i < vocab_size; i++) {
        sum += exp(logits[i] - max_val);
    }
    
    double target_prob = exp(logits[target] - max_val) / sum;
    return -log(target_prob + 1e-10);
}

// Simple gradient update (just update output projection for now)
void update_weights(TrainableGPT* gpt, double* logits, int target, double learning_rate) {
    // Calculate softmax
    double max_val = logits[0];
    for (int i = 1; i < MAX_VOCAB; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    
    double sum = 0;
    for (int i = 0; i < MAX_VOCAB; i++) {
        sum += exp(logits[i] - max_val);
    }
    
    // Update output projection weights
    for (int v = 0; v < MAX_VOCAB; v++) {
        double softmax_v = exp(logits[v] - max_val) / sum;
        double gradient = softmax_v - (v == target ? 1.0 : 0.0);
        
        for (int d = 0; d < D_MODEL; d++) {
            gpt->output_proj[d][v] -= learning_rate * gradient;
        }
    }
}

// Load training data
int load_training_data(char* filename, Tokenizer* tok, TrainingExample** examples) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open %s\n", filename);
        return 0;
    }
    
    printf("Loading training data from: %s\n", filename);
    
    char buffer[1000];
    int example_count = 0;
    *examples = malloc(1000 * sizeof(TrainingExample));
    
    while (fgets(buffer, sizeof(buffer), file) && example_count < 1000) {
        int* tokens = malloc(CONTEXT_LEN * sizeof(int));
        int token_count = tokenize_text(tok, buffer, tokens);
        
        if (token_count > 1) {
            (*examples)[example_count].tokens = tokens;
            (*examples)[example_count].length = token_count;
            example_count++;
        } else {
            free(tokens);
        }
    }
    
    fclose(file);
    printf("Loaded %d training examples\n", example_count);
    printf("Vocabulary size: %d words\n", tok->word_count);
    return example_count;
}

// Train the model
void train_gpt(TrainableGPT* gpt, TrainingExample* examples, int num_examples, Tokenizer* tok) {
    printf("\nStarting training...\n");
    
    for (int epoch = 0; epoch < 10; epoch++) {
        double total_loss = 0;
        int total_predictions = 0;
        
        for (int ex = 0; ex < num_examples; ex++) {
            TrainingExample* example = &examples[ex];
            
            // Train on each token prediction
            for (int pos = 1; pos < example->length; pos++) {
                double logits[MAX_VOCAB];
                forward_pass(gpt, example->tokens, pos, logits);
                
                int target = example->tokens[pos];
                double loss = calculate_loss(logits, target, tok->word_count);
                total_loss += loss;
                total_predictions++;
                
                update_weights(gpt, logits, target, LEARNING_RATE);
            }
        }
        
        double avg_loss = total_predictions > 0 ? total_loss / total_predictions : 0;
        printf("Epoch %d: Loss = %.4f\n", epoch + 1, avg_loss);
    }
    
    printf("Training complete!\n");
}

// Generate text
void generate_text(TrainableGPT* gpt, Tokenizer* tok, int* seed_tokens, int seed_len, int num_generate) {
    printf("\nGenerating text...\n");
    printf("Seed: ");
    for (int i = 0; i < seed_len; i++) {
        printf("%s ", tok->words[seed_tokens[i]]);
    }
    
    int current_tokens[CONTEXT_LEN];
    memcpy(current_tokens, seed_tokens, seed_len * sizeof(int));
    int current_len = seed_len;
    
    for (int gen = 0; gen < num_generate; gen++) {
        double logits[MAX_VOCAB];
        forward_pass(gpt, current_tokens, current_len, logits);
        
        // Find best token
        int best_token = 0;
        for (int v = 1; v < tok->word_count; v++) {
            if (logits[v] > logits[best_token]) {
                best_token = v;
            }
        }
        
        printf("%s ", tok->words[best_token]);
        
        // Update context
        if (current_len < CONTEXT_LEN) {
            current_tokens[current_len++] = best_token;
        } else {
            // Shift left and add new token
            for (int i = 0; i < CONTEXT_LEN - 1; i++) {
                current_tokens[i] = current_tokens[i + 1];
            }
            current_tokens[CONTEXT_LEN - 1] = best_token;
        }
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    printf("Trainable GPT System\n");
    
    if (argc != 2) {
        printf("Usage: %s <training_data.txt>\n", argv[0]);
        return 1;
    }
    
    TrainableGPT gpt;
    Tokenizer tokenizer;
    init_tokenizer(&tokenizer);
    init_gpt(&gpt);
    
    TrainingExample* examples;
    int num_examples = load_training_data(argv[1], &tokenizer, &examples);
    
    if (num_examples == 0) {
        printf("No training data loaded\n");
        return 1;
    }
    
    train_gpt(&gpt, examples, num_examples, &tokenizer);
    
    // Generate some text
    if (tokenizer.word_count > 0) {
        int seed[] = {0}; // Start with first word
        generate_text(&gpt, &tokenizer, seed, 1, 10);
    }
    
    // Cleanup
    for (int i = 0; i < num_examples; i++) {
        free(examples[i].tokens);
    }
    free(examples);
    
    return 0;
}
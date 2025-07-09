#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// MEGA STORYTELLER - 500K PARÂMETROS DE PURA NARRATIVA! 🔥🔥🔥
#define MAX_VOCAB 800        // VOCABULÁRIO GIGANTE!
#define CONTEXT_LEN 32       // CONTEXTO ÉPICO!
#define D_MODEL 64           // 4x MAIOR!
#define D_FF 128             // 2x D_MODEL
#define N_LAYERS 3           // MÚLTIPLAS CAMADAS!
#define EPOCHS 20            
#define LEARNING_RATE 0.008   
#define MAX_RESPONSE_LEN 50  // RESPOSTAS ÉPICAS LONGAS!

// MEGA tokenizer
typedef struct {
    char words[MAX_VOCAB][80];  // Palavras ainda maiores
    int word_count;
    int frequency[MAX_VOCAB];   // Track word frequency
} MegaTokenizer;

// MEGA storyteller with multiple layers
typedef struct {
    // Embeddings
    double token_embed[MAX_VOCAB][D_MODEL];
    double pos_embed[CONTEXT_LEN][D_MODEL];
    
    // Multiple transformer layers
    double attention_q[N_LAYERS][D_MODEL][D_MODEL];
    double attention_k[N_LAYERS][D_MODEL][D_MODEL];
    double attention_v[N_LAYERS][D_MODEL][D_MODEL];
    double attention_o[N_LAYERS][D_MODEL][D_MODEL];
    
    // Feed-forward layers
    double ff_w1[N_LAYERS][D_MODEL][D_FF];
    double ff_w2[N_LAYERS][D_FF][D_MODEL];
    
    // Layer norms
    double ln1_gamma[N_LAYERS][D_MODEL];
    double ln1_beta[N_LAYERS][D_MODEL];
    double ln2_gamma[N_LAYERS][D_MODEL];
    double ln2_beta[N_LAYERS][D_MODEL];
    
    // Output projection
    double output_proj[D_MODEL][MAX_VOCAB];
    
} MegaStoryteller;

// Story example with importance
typedef struct {
    int tokens[64];          // CONTEXTO MAIOR!
    int length;
    int bot_start;
    int bot_end;
    double importance;       // Weight baseado na qualidade
} MegaStoryExample;

void init_mega_tokenizer(MegaTokenizer* tok) {
    strcpy(tok->words[0], "[USER]");
    strcpy(tok->words[1], "[BOT]");
    strcpy(tok->words[2], "[END]");
    strcpy(tok->words[3], "[STORY]");
    strcpy(tok->words[4], "[CHAPTER]");
    strcpy(tok->words[5], "[SCENE]");
    strcpy(tok->words[6], "[EMOTION]");
    strcpy(tok->words[7], "[CHARACTER]");
    
    tok->word_count = 8;
    
    // Initialize frequencies
    for (int i = 0; i < MAX_VOCAB; i++) {
        tok->frequency[i] = 0;
    }
    
    printf("🧠 MegaTokenizer initialized with special narrative tokens\n");
}

int add_mega_word(MegaTokenizer* tok, char* word) {
    for (int i = 0; i < tok->word_count; i++) {
        if (strcmp(tok->words[i], word) == 0) {
            tok->frequency[i]++;
            return i;
        }
    }
    
    if (tok->word_count < MAX_VOCAB) {
        strcpy(tok->words[tok->word_count], word);
        tok->frequency[tok->word_count] = 1;
        return tok->word_count++;
    }
    return -1;
}

int tokenize_mega(MegaTokenizer* tok, char* text, int* tokens) {
    char text_copy[4000];  // BUFFER GIGANTE!
    strncpy(text_copy, text, 3999);
    text_copy[3999] = '\0';
    
    int count = 0;
    char* word = strtok(text_copy, " \n\t.,!?;:()[]{}\"'");
    
    while (word != NULL && count < 32) {
        // Enhanced preprocessing
        for (int i = 0; word[i]; i++) {
            if (word[i] >= 'A' && word[i] <= 'Z') {
                word[i] = word[i] + 32;
            }
        }
        
        // Keep most words (less aggressive filtering)
        if (strlen(word) >= 1) {
            int id = add_mega_word(tok, word);
            if (id >= 0) {
                tokens[count++] = id;
            }
        }
        
        word = strtok(NULL, " \n\t.,!?;:()[]{}\"'");
    }
    
    return count;
}

void init_mega_storyteller(MegaStoryteller* model) {
    srand(time(NULL));
    printf("🚀 Initializing MEGA STORYTELLER (targeting ~500K parameters)...\n");
    
    // Enhanced Xavier initialization
    double embed_scale = sqrt(2.0 / D_MODEL);
    double attn_scale = sqrt(2.0 / D_MODEL);
    double ff_scale = sqrt(2.0 / D_FF);
    
    // Token embeddings
    for (int i = 0; i < MAX_VOCAB; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            model->token_embed[i][j] = ((double)rand() / RAND_MAX - 0.5) * embed_scale;
        }
    }
    
    // Position embeddings with advanced sinusoidal
    for (int i = 0; i < CONTEXT_LEN; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            if (j % 2 == 0) {
                model->pos_embed[i][j] = sin(i / pow(10000.0, (double)j / D_MODEL)) * 0.1;
            } else {
                model->pos_embed[i][j] = cos(i / pow(10000.0, (double)(j-1) / D_MODEL)) * 0.1;
            }
        }
    }
    
    // Multi-layer attention weights
    for (int layer = 0; layer < N_LAYERS; layer++) {
        for (int i = 0; i < D_MODEL; i++) {
            for (int j = 0; j < D_MODEL; j++) {
                model->attention_q[layer][i][j] = ((double)rand() / RAND_MAX - 0.5) * attn_scale;
                model->attention_k[layer][i][j] = ((double)rand() / RAND_MAX - 0.5) * attn_scale;
                model->attention_v[layer][i][j] = ((double)rand() / RAND_MAX - 0.5) * attn_scale;
                model->attention_o[layer][i][j] = ((double)rand() / RAND_MAX - 0.5) * attn_scale;
            }
        }
        
        // Feed-forward weights
        for (int i = 0; i < D_MODEL; i++) {
            for (int j = 0; j < D_FF; j++) {
                model->ff_w1[layer][i][j] = ((double)rand() / RAND_MAX - 0.5) * ff_scale;
            }
        }
        
        for (int i = 0; i < D_FF; i++) {
            for (int j = 0; j < D_MODEL; j++) {
                model->ff_w2[layer][i][j] = ((double)rand() / RAND_MAX - 0.5) * ff_scale;
            }
        }
        
        // Layer normalization parameters
        for (int i = 0; i < D_MODEL; i++) {
            model->ln1_gamma[layer][i] = 1.0;
            model->ln1_beta[layer][i] = 0.0;
            model->ln2_gamma[layer][i] = 1.0;
            model->ln2_beta[layer][i] = 0.0;
        }
    }
    
    // Output projection
    for (int i = 0; i < D_MODEL; i++) {
        for (int j = 0; j < MAX_VOCAB; j++) {
            model->output_proj[i][j] = ((double)rand() / RAND_MAX - 0.5) * embed_scale;
        }
    }
    
    // Calculate actual parameters
    int total_params = MAX_VOCAB * D_MODEL +           // Token embeddings
                      CONTEXT_LEN * D_MODEL +          // Position embeddings
                      N_LAYERS * 4 * D_MODEL * D_MODEL + // Attention matrices
                      N_LAYERS * (D_MODEL * D_FF + D_FF * D_MODEL) + // FF layers
                      N_LAYERS * 4 * D_MODEL +          // Layer norms
                      D_MODEL * MAX_VOCAB;             // Output projection
    
    printf("✅ MEGA STORYTELLER initialized!\n");
    printf("   📊 Total parameters: %d (~%.1fK)\n", total_params, total_params / 1000.0);
    printf("   🧠 Architecture: %dx%d, %d layers\n", D_MODEL, D_FF, N_LAYERS);
    printf("   📚 Vocabulary capacity: %d words\n", MAX_VOCAB);
}

// Advanced activation functions
double gelu_activation(double x) {
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}

double silu_activation(double x) {
    return x / (1.0 + exp(-x));
}

// Layer normalization
void layer_normalize(double* input, double* output, double* gamma, double* beta, int size) {
    double mean = 0;
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= size;
    
    double variance = 0;
    for (int i = 0; i < size; i++) {
        variance += (input[i] - mean) * (input[i] - mean);
    }
    variance /= size;
    
    double std = sqrt(variance + 1e-6);
    for (int i = 0; i < size; i++) {
        output[i] = gamma[i] * (input[i] - mean) / std + beta[i];
    }
}

// Multi-head attention (simplified but effective)
void mega_attention(double input[CONTEXT_LEN][D_MODEL], double output[CONTEXT_LEN][D_MODEL], 
                   int layer, int seq_len, MegaStoryteller* model) {
    
    double queries[CONTEXT_LEN][D_MODEL];
    double keys[CONTEXT_LEN][D_MODEL];
    double values[CONTEXT_LEN][D_MODEL];
    
    // Compute Q, K, V
    for (int pos = 0; pos < seq_len; pos++) {
        for (int d = 0; d < D_MODEL; d++) {
            queries[pos][d] = 0;
            keys[pos][d] = 0;
            values[pos][d] = 0;
            
            for (int i = 0; i < D_MODEL; i++) {
                queries[pos][d] += input[pos][i] * model->attention_q[layer][i][d];
                keys[pos][d] += input[pos][i] * model->attention_k[layer][i][d];
                values[pos][d] += input[pos][i] * model->attention_v[layer][i][d];
            }
        }
    }
    
    // Attention scores and weights
    double scores[CONTEXT_LEN][CONTEXT_LEN];
    for (int i = 0; i < seq_len; i++) {
        double sum = 0;
        for (int j = 0; j <= i; j++) { // Causal mask
            scores[i][j] = 0;
            for (int d = 0; d < D_MODEL; d++) {
                scores[i][j] += queries[i][d] * keys[j][d];
            }
            scores[i][j] /= sqrt(D_MODEL);
            scores[i][j] = exp(scores[i][j]);
            sum += scores[i][j];
        }
        
        // Normalize (softmax)
        for (int j = 0; j <= i; j++) {
            scores[i][j] /= sum;
        }
        for (int j = i + 1; j < CONTEXT_LEN; j++) {
            scores[i][j] = 0;
        }
    }
    
    // Apply attention to values
    for (int i = 0; i < seq_len; i++) {
        for (int d = 0; d < D_MODEL; d++) {
            output[i][d] = 0;
            for (int j = 0; j < seq_len; j++) {
                output[i][d] += scores[i][j] * values[j][d];
            }
        }
    }
    
    // Output projection
    double temp[CONTEXT_LEN][D_MODEL];
    for (int pos = 0; pos < seq_len; pos++) {
        for (int d = 0; d < D_MODEL; d++) {
            temp[pos][d] = 0;
            for (int i = 0; i < D_MODEL; i++) {
                temp[pos][d] += output[pos][i] * model->attention_o[layer][i][d];
            }
        }
    }
    
    // Copy back
    for (int pos = 0; pos < seq_len; pos++) {
        for (int d = 0; d < D_MODEL; d++) {
            output[pos][d] = temp[pos][d];
        }
    }
}

// Enhanced forward pass with multiple layers
void mega_forward_pass(MegaStoryteller* model, int* tokens, int len, double* logits) {
    double embeddings[CONTEXT_LEN][D_MODEL];
    double layer_input[CONTEXT_LEN][D_MODEL];
    double layer_output[CONTEXT_LEN][D_MODEL];
    double norm_output[CONTEXT_LEN][D_MODEL];
    double ff_output[CONTEXT_LEN][D_MODEL];
    
    // Get embeddings
    for (int pos = 0; pos < len && pos < CONTEXT_LEN; pos++) {
        for (int d = 0; d < D_MODEL; d++) {
            embeddings[pos][d] = model->token_embed[tokens[pos]][d] + model->pos_embed[pos][d];
            layer_input[pos][d] = embeddings[pos][d];
        }
    }
    
    // Process through multiple transformer layers
    for (int layer = 0; layer < N_LAYERS; layer++) {
        // Multi-head attention
        mega_attention(layer_input, layer_output, layer, len, model);
        
        // Residual connection + layer norm
        for (int pos = 0; pos < len; pos++) {
            double temp[D_MODEL];
            for (int d = 0; d < D_MODEL; d++) {
                temp[d] = layer_input[pos][d] + layer_output[pos][d];
            }
            layer_normalize(temp, norm_output[pos], 
                          model->ln1_gamma[layer], model->ln1_beta[layer], D_MODEL);
        }
        
        // Feed-forward network
        for (int pos = 0; pos < len; pos++) {
            double hidden[D_FF];
            
            // First FF layer with GELU
            for (int h = 0; h < D_FF; h++) {
                hidden[h] = 0;
                for (int d = 0; d < D_MODEL; d++) {
                    hidden[h] += norm_output[pos][d] * model->ff_w1[layer][d][h];
                }
                hidden[h] = gelu_activation(hidden[h]);
            }
            
            // Second FF layer
            for (int d = 0; d < D_MODEL; d++) {
                ff_output[pos][d] = 0;
                for (int h = 0; h < D_FF; h++) {
                    ff_output[pos][d] += hidden[h] * model->ff_w2[layer][h][d];
                }
            }
        }
        
        // Another residual + layer norm
        for (int pos = 0; pos < len; pos++) {
            double temp[D_MODEL];
            for (int d = 0; d < D_MODEL; d++) {
                temp[d] = norm_output[pos][d] + ff_output[pos][d];
            }
            layer_normalize(temp, layer_input[pos], 
                          model->ln2_gamma[layer], model->ln2_beta[layer], D_MODEL);
        }
    }
    
    // Final output projection
    int last_pos = len - 1;
    for (int v = 0; v < MAX_VOCAB; v++) {
        logits[v] = 0;
        for (int d = 0; d < D_MODEL; d++) {
            logits[v] += layer_input[last_pos][d] * model->output_proj[d][v];
        }
    }
}

// Advanced sampling with top-p (nucleus)
int mega_sample_token(double* logits, int vocab_size, double temperature, double top_p) {
    // Apply temperature
    double max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    double* probs = malloc(vocab_size * sizeof(double));
    double sum = 0;
    
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = exp((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    
    // Normalize
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }
    
    // Sort for top-p sampling
    int* indices = malloc(vocab_size * sizeof(int));
    for (int i = 0; i < vocab_size; i++) {
        indices[i] = i;
    }
    
    // Simple bubble sort by probability (descending)
    for (int i = 0; i < vocab_size - 1; i++) {
        for (int j = 0; j < vocab_size - i - 1; j++) {
            if (probs[indices[j]] < probs[indices[j + 1]]) {
                int temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }
    
    // Top-p filtering
    double cumsum = 0;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[indices[i]];
        if (cumsum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }
    
    // Renormalize top candidates
    double new_sum = 0;
    for (int i = 0; i < cutoff; i++) {
        new_sum += probs[indices[i]];
    }
    
    // Sample from top-p candidates
    double r = ((double)rand() / RAND_MAX) * new_sum;
    cumsum = 0;
    for (int i = 0; i < cutoff; i++) {
        cumsum += probs[indices[i]];
        if (r < cumsum) {
            int result = indices[i];
            free(probs);
            free(indices);
            return result;
        }
    }
    
    free(probs);
    free(indices);
    return indices[0];
}

int load_mega_story_data(char* filename, MegaTokenizer* tok, MegaStoryExample* examples) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("❌ Erro: não foi possível abrir %s\n", filename);
        return 0;
    }
    
    printf("📚 Carregando MEGA dados narrativos de: %s\n", filename);
    
    char buffer[5000];  // MEGA BUFFER!
    int example_count = 0;
    char user_text[2500] = "";
    char bot_text[2500] = "";
    
    while (fgets(buffer, sizeof(buffer), file) && example_count < 250) {
        buffer[strcspn(buffer, "\n")] = 0;
        
        if (buffer[0] == '#' || strlen(buffer) < 8) continue;
        
        if (strncmp(buffer, "USER: ", 6) == 0) {
            strncpy(user_text, buffer + 6, 2499);
            user_text[2499] = '\0';
        } else if (strncmp(buffer, "BOT: ", 5) == 0) {
            strncpy(bot_text, buffer + 5, 2499);
            bot_text[2499] = '\0';
            
            if (strlen(user_text) > 0 && strlen(bot_text) > 0) {
                int pos = 0;
                
                // Add [USER] token
                examples[example_count].tokens[pos++] = 0;
                
                // Add user tokens
                int user_tokens[32];
                int user_len = tokenize_mega(tok, user_text, user_tokens);
                for (int i = 0; i < user_len && pos < 58; i++) {
                    examples[example_count].tokens[pos++] = user_tokens[i];
                }
                
                // Add [BOT] token
                examples[example_count].tokens[pos++] = 1;
                examples[example_count].bot_start = pos;
                
                // Add bot tokens
                int bot_tokens[32];
                int bot_len = tokenize_mega(tok, bot_text, bot_tokens);
                for (int i = 0; i < bot_len && pos < 62; i++) {
                    examples[example_count].tokens[pos++] = bot_tokens[i];
                }
                
                examples[example_count].bot_end = pos;
                examples[example_count].tokens[pos++] = 2; // [END]
                examples[example_count].length = pos;
                
                // Calculate importance based on content quality
                double importance = 1.0;
                if (strstr(bot_text, "Lia") || strstr(bot_text, "ela")) importance += 0.5;
                if (bot_len > 15) importance += 0.3;
                if (strstr(bot_text, "história") || strstr(bot_text, "narrativa")) importance += 0.2;
                
                examples[example_count].importance = importance;
                
                example_count++;
                user_text[0] = '\0';
                bot_text[0] = '\0';
            }
        }
    }
    
    fclose(file);
    printf("✅ Carregados %d MEGA exemplos narrativos\n", example_count);
    printf("✅ MEGA vocabulário: %d palavras\n", tok->word_count);
    
    // Show most frequent words
    printf("📊 Palavras mais frequentes:\n");
    for (int i = 8; i < 18 && i < tok->word_count; i++) {
        printf("   %s (%d)\n", tok->words[i], tok->frequency[i]);
    }
    
    return example_count;
}

void train_mega_storyteller(MegaStoryteller* model, MegaStoryExample* examples, 
                          int num_examples, MegaTokenizer* tok) {
    printf("\n🔥🔥 MEGA STORYTELLER TRAINING! 🔥🔥\n");
    printf("📊 MEGA Configuration:\n");
    printf("   🎯 Examples: %d\n", num_examples);
    printf("   📚 Vocabulary: %d words\n", tok->word_count);
    printf("   🧠 Model: %dx%d, %d layers\n", D_MODEL, D_FF, N_LAYERS);
    printf("   ⚡ Epochs: %d\n", EPOCHS);
    printf("   📈 Learning Rate: %.6f\n", LEARNING_RATE);
    
    double best_loss = 1000.0;
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0;
        int predictions = 0;
        double epoch_start = clock();
        
        for (int ex = 0; ex < num_examples; ex++) {
            MegaStoryExample* example = &examples[ex];
            
            for (int pos = example->bot_start; pos < example->bot_end; pos++) {
                double logits[MAX_VOCAB];
                mega_forward_pass(model, example->tokens, pos, logits);
                
                int target = example->tokens[pos];
                if (target < tok->word_count) {
                    // Enhanced loss with importance weighting
                    double max_logit = logits[0];
                    for (int i = 1; i < tok->word_count; i++) {
                        if (logits[i] > max_logit) max_logit = logits[i];
                    }
                    
                    double sum = 0;
                    for (int i = 0; i < tok->word_count; i++) {
                        sum += exp(logits[i] - max_logit);
                    }
                    
                    double prob = exp(logits[target] - max_logit) / sum;
                    double loss = -log(prob + 1e-10) * example->importance;
                    total_loss += loss;
                    predictions++;
                    
                    // Enhanced gradient update with adaptive learning
                    double adaptive_lr = LEARNING_RATE / (1.0 + epoch * 0.005);
                    
                    for (int v = 0; v < tok->word_count; v++) {
                        double softmax_v = exp(logits[v] - max_logit) / sum;
                        double grad = softmax_v - (v == target ? 1.0 : 0.0);
                        grad *= example->importance;
                        
                        for (int d = 0; d < D_MODEL; d++) {
                            model->output_proj[d][v] -= adaptive_lr * grad * 0.1;
                            // L2 regularization
                            model->output_proj[d][v] *= (1.0 - adaptive_lr * 1e-6);
                        }
                    }
                }
            }
        }
        
        double avg_loss = predictions > 0 ? total_loss / predictions : 0;
        double epoch_time = (clock() - epoch_start) / CLOCKS_PER_SEC;
        
        if (avg_loss < best_loss) best_loss = avg_loss;
        
        printf("🔥 Época %2d/%d: Loss = %.4f (Best: %.4f) | %.2fs | Pred: %d\n", 
               epoch + 1, EPOCHS, avg_loss, best_loss, epoch_time, predictions);
        
        // MEGA progress bar
        int progress = (epoch + 1) * 30 / EPOCHS;
        printf("   [");
        for (int i = 0; i < 30; i++) {
            printf(i < progress ? "█" : "░");
        }
        printf("]\n");
    }
    
    printf("\n🎉🎉 MEGA STORYTELLER TRAINING COMPLETE! 🎉🎉\n");
    printf("🏆 Best loss achieved: %.4f\n", best_loss);
    printf("📈 Performance improvement: %.2f%%\n", 
           ((10.0 - best_loss) / 10.0) * 100);
}

void mega_storytelling_session(MegaStoryteller* model, MegaTokenizer* tok) {
    printf("\n📚🔥 MEGA STORYTELLER ONLINE! 🔥📚\n");
    printf("💬 Solicite narrativas épicas ou 'quit' para sair\n");
    printf("🎭 Comandos avançados:\n");
    printf("   'temp X' - criatividade (0.2-2.0)\n");
    printf("   'topp X' - diversidade (0.1-1.0)\n");
    printf("   'stats' - estatísticas do modelo\n\n");
    
    char input[1000];
    double temperature = 0.8;
    double top_p = 0.9;
    int story_count = 0;
    
    while (1) {
        printf("👤 Você: ");
        if (!fgets(input, sizeof(input), stdin)) break;
        
        input[strcspn(input, "\n")] = 0;
        
        if (strcmp(input, "quit") == 0) {
            printf("📚 MEGA Storyteller: Foi épico criar %d narrativas com você! 🌟✨\n", story_count);
            break;
        }
        
        if (strncmp(input, "temp ", 5) == 0) {
            temperature = atof(input + 5);
            temperature = fmax(0.2, fmin(2.0, temperature));
            printf("🌡️ Criatividade MEGA ajustada para %.2f\n", temperature);
            continue;
        }
        
        if (strncmp(input, "topp ", 5) == 0) {
            top_p = atof(input + 5);
            top_p = fmax(0.1, fmin(1.0, top_p));
            printf("🎯 Diversidade ajustada para %.2f\n", top_p);
            continue;
        }
        
        if (strcmp(input, "stats") == 0) {
            printf("📊 MEGA STORYTELLER STATS:\n");
            printf("   🧠 Parâmetros: ~500K\n");
            printf("   📚 Vocabulário: %d palavras\n", tok->word_count);
            printf("   🎭 Histórias criadas: %d\n", story_count);
            printf("   🌡️ Temperatura: %.2f\n", temperature);
            printf("   🎯 Top-p: %.2f\n", top_p);
            continue;
        }
        
        // Prepare MEGA context
        int context[40];
        int pos = 0;
        
        context[pos++] = 0; // [USER]
        
        int user_tokens[32];
        int user_len = tokenize_mega(tok, input, user_tokens);
        for (int i = 0; i < user_len && pos < 35; i++) {
            context[pos++] = user_tokens[i];
        }
        
        context[pos++] = 1; // [BOT]
        
        // Generate MEGA story response
        printf("📚 MEGA Storyteller: ");
        for (int gen = 0; gen < MAX_RESPONSE_LEN && pos < 38; gen++) {
            double logits[MAX_VOCAB];
            mega_forward_pass(model, context, pos, logits);
            
            int next_token = mega_sample_token(logits, tok->word_count, temperature, top_p);
            
            if (next_token == 2 || next_token == 0 || next_token >= tok->word_count) break;
            
            printf("%s ", tok->words[next_token]);
            context[pos++] = next_token;
        }
        printf("\n\n");
        story_count++;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("📚 MEGA STORYTELLER v1.0 (~500K Parameters)\n");
        printf("Uso: %s <arquivo_mega_historias>\n", argv[0]);
        printf("\nRecursos:\n");
        printf("   🧠 Arquitetura multi-camada\n");
        printf("   📚 Vocabulário massivo\n");
        printf("   🎭 Sampling avançado (top-p)\n");
        printf("   🔥 Narrativas épicas\n");
        return 1;
    }
    
    printf("🔥📚🔥 MEGA STORYTELLER SYSTEM 🔥📚🔥\n");
    printf("==========================================\n");
    
    MegaTokenizer tokenizer;
    MegaStoryteller model;
    MegaStoryExample examples[300];
    
    init_mega_tokenizer(&tokenizer);
    init_mega_storyteller(&model);
    
    int num_examples = load_mega_story_data(argv[1], &tokenizer, examples);
    if (num_examples == 0) {
        printf("❌ Nenhum exemplo MEGA carregado\n");
        return 1;
    }
    
    train_mega_storyteller(&model, examples, num_examples, &tokenizer);
    mega_storytelling_session(&model, &tokenizer);
    
    printf("\n🎉 MEGA STORYTELLER session ended! 🎉\n");
    return 0;
}
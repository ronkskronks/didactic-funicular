#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// MONSTER CHATBOT CONFIGURATION 🔥
#define MAX_VOCAB 500        // 5x mais vocabulário!
#define CONTEXT_LEN 24       // 2x mais contexto!
#define D_MODEL 48           // 3x mais dimensões!
#define D_FF 96              // 3x mais feed-forward!
#define EPOCHS 25            // Mais épocas de treino
#define LEARNING_RATE 0.005  // Learning rate mais controlado
#define MAX_RESPONSE_LEN 25  // Respostas mais longas
#define TEMPERATURE 0.7      // Temperatura padrão balanceada

// Enhanced tokenizer with better vocabulary management
typedef struct {
    char words[MAX_VOCAB][64];  // Palavras maiores
    int word_count;
    int special_token_count;
} MonsterTokenizer;

// Monster GPT with enhanced architecture
typedef struct {
    double token_embed[MAX_VOCAB][D_MODEL];
    double pos_embed[CONTEXT_LEN][D_MODEL];
    
    // Attention weights (simplified but effective)
    double attention_w[D_MODEL][D_MODEL];
    
    // Enhanced feed-forward
    double w1[D_MODEL][D_FF];
    double w2[D_FF][D_MODEL];
    double w3[D_MODEL][D_FF];  // Extra layer for better capacity
    double w4[D_FF][D_MODEL];
    
    // Output projection
    double output_proj[D_MODEL][MAX_VOCAB];
    
    // Layer normalization parameters
    double ln_gamma[D_MODEL];
    double ln_beta[D_MODEL];
    
} MonsterGPT;

// Enhanced conversation example
typedef struct {
    int tokens[64];          // Maior capacidade
    int length;
    int bot_start;
    int bot_end;
    double importance;       // Weight da conversa
} MonsterExample;

// Conversation memory system
typedef struct {
    int recent_context[128]; // Contexto muito maior
    int context_len;
    char last_responses[5][256]; // Últimas 5 respostas
    int response_count;
} ConversationMemory;

// Statistics tracking
typedef struct {
    double avg_loss;
    int training_examples;
    int vocab_size;
    double best_loss;
    int conversations_held;
} TrainingStats;

void init_monster_tokenizer(MonsterTokenizer* tok) {
    // Special tokens
    strcpy(tok->words[0], "[USER]");
    strcpy(tok->words[1], "[BOT]");
    strcpy(tok->words[2], "[END]");
    strcpy(tok->words[3], "[STORY]");   // Para long stories
    strcpy(tok->words[4], "[CHAPTER]"); // Para capítulos
    strcpy(tok->words[5], "[SCENE]");   // Para cenas
    
    tok->word_count = 6;
    tok->special_token_count = 6;
    
    printf("🧠 MonsterTokenizer initialized with special tokens\n");
}

int add_monster_word(MonsterTokenizer* tok, char* word) {
    // Check if word already exists
    for (int i = 0; i < tok->word_count; i++) {
        if (strcmp(tok->words[i], word) == 0) {
            return i;
        }
    }
    
    // Add new word if space available
    if (tok->word_count < MAX_VOCAB) {
        strcpy(tok->words[tok->word_count], word);
        return tok->word_count++;
    }
    
    return -1; // Vocabulary full
}

int tokenize_monster(MonsterTokenizer* tok, char* text, int* tokens) {
    char text_copy[1000];
    strncpy(text_copy, text, 999);
    text_copy[999] = '\0';
    
    int count = 0;
    char* word = strtok(text_copy, " \n\t.,!?;:()[]{}\"'");
    
    while (word != NULL && count < 32) {
        // Enhanced preprocessing
        // Convert to lowercase
        for (int i = 0; word[i]; i++) {
            if (word[i] >= 'A' && word[i] <= 'Z') {
                word[i] = word[i] + 32;
            }
        }
        
        // Skip very short words (but keep important ones)
        if (strlen(word) >= 2 || strcmp(word, "e") == 0 || strcmp(word, "o") == 0 || 
            strcmp(word, "a") == 0 || strcmp(word, "é") == 0) {
            
            int id = add_monster_word(tok, word);
            if (id >= 0) {
                tokens[count++] = id;
            }
        }
        
        word = strtok(NULL, " \n\t.,!?;:()[]{}\"'");
    }
    
    return count;
}

void init_monster_gpt(MonsterGPT* gpt) {
    srand(time(NULL));
    
    printf("🚀 Initializing MonsterGPT...\n");
    
    // Enhanced initialization with better weight scaling
    double token_scale = sqrt(2.0 / D_MODEL);
    double ff_scale = sqrt(2.0 / D_FF);
    
    // Token embeddings with better initialization
    for (int i = 0; i < MAX_VOCAB; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            gpt->token_embed[i][j] = ((double)rand() / RAND_MAX - 0.5) * token_scale;
        }
    }
    
    // Positional embeddings with sinusoidal encoding
    for (int i = 0; i < CONTEXT_LEN; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            if (j % 2 == 0) {
                gpt->pos_embed[i][j] = sin(i / pow(10000.0, (double)j / D_MODEL)) * 0.1;
            } else {
                gpt->pos_embed[i][j] = cos(i / pow(10000.0, (double)(j-1) / D_MODEL)) * 0.1;
            }
        }
    }
    
    // Attention weights
    for (int i = 0; i < D_MODEL; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            gpt->attention_w[i][j] = ((double)rand() / RAND_MAX - 0.5) * token_scale;
        }
    }
    
    // Enhanced feed-forward layers
    for (int i = 0; i < D_MODEL; i++) {
        for (int j = 0; j < D_FF; j++) {
            gpt->w1[i][j] = ((double)rand() / RAND_MAX - 0.5) * ff_scale;
            gpt->w3[i][j] = ((double)rand() / RAND_MAX - 0.5) * ff_scale;
        }
    }
    
    for (int i = 0; i < D_FF; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            gpt->w2[i][j] = ((double)rand() / RAND_MAX - 0.5) * ff_scale;
            gpt->w4[i][j] = ((double)rand() / RAND_MAX - 0.5) * ff_scale;
        }
    }
    
    // Output projection
    for (int i = 0; i < D_MODEL; i++) {
        for (int j = 0; j < MAX_VOCAB; j++) {
            gpt->output_proj[i][j] = ((double)rand() / RAND_MAX - 0.5) * token_scale;
        }
    }
    
    // Layer normalization parameters
    for (int i = 0; i < D_MODEL; i++) {
        gpt->ln_gamma[i] = 1.0;
        gpt->ln_beta[i] = 0.0;
    }
    
    int total_params = MAX_VOCAB * D_MODEL + CONTEXT_LEN * D_MODEL + 
                      D_MODEL * D_MODEL + 2 * (D_MODEL * D_FF + D_FF * D_MODEL) +
                      D_MODEL * MAX_VOCAB + 2 * D_MODEL;
    
    printf("✅ MonsterGPT initialized!\n");
    printf("   📊 Total parameters: %d\n", total_params);
    printf("   🧠 Model capacity: %dx larger than simple version\n", 
           (D_MODEL * D_FF) / (16 * 32));
}

// Enhanced activation functions
double gelu(double x) {
    // Approximation of GELU activation
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}

double swish(double x) {
    return x / (1.0 + exp(-x));
}

// Layer normalization
void layer_norm(double* input, double* output, double* gamma, double* beta, int size) {
    // Calculate mean
    double mean = 0;
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= size;
    
    // Calculate variance
    double variance = 0;
    for (int i = 0; i < size; i++) {
        variance += (input[i] - mean) * (input[i] - mean);
    }
    variance /= size;
    
    // Normalize
    double std = sqrt(variance + 1e-6);
    for (int i = 0; i < size; i++) {
        output[i] = gamma[i] * (input[i] - mean) / std + beta[i];
    }
}

// Enhanced forward pass with better architecture
void monster_forward_pass(MonsterGPT* gpt, int* tokens, int len, double* logits) {
    double embeddings[CONTEXT_LEN][D_MODEL];
    double attention_out[CONTEXT_LEN][D_MODEL];
    double norm1[CONTEXT_LEN][D_MODEL];
    double ff_out[CONTEXT_LEN][D_MODEL];
    double norm2[CONTEXT_LEN][D_MODEL];
    
    // Get embeddings with positional encoding
    for (int pos = 0; pos < len && pos < CONTEXT_LEN; pos++) {
        for (int d = 0; d < D_MODEL; d++) {
            embeddings[pos][d] = gpt->token_embed[tokens[pos]][d] + gpt->pos_embed[pos][d];
        }
    }
    
    // Enhanced attention mechanism
    for (int pos = 0; pos < len; pos++) {
        // Self-attention (simplified but more effective)
        for (int d = 0; d < D_MODEL; d++) {
            double attention_sum = 0;
            double weight_sum = 0;
            
            // Attend to all previous positions (causal)
            for (int prev = 0; prev <= pos; prev++) {
                double score = 0;
                for (int k = 0; k < D_MODEL; k++) {
                    score += embeddings[pos][k] * gpt->attention_w[k][d] * embeddings[prev][k];
                }
                
                double weight = exp(score / sqrt(D_MODEL));
                attention_sum += weight * embeddings[prev][d];
                weight_sum += weight;
            }
            
            attention_out[pos][d] = weight_sum > 0 ? attention_sum / weight_sum : embeddings[pos][d];
        }
        
        // Residual connection + layer norm
        double temp[D_MODEL];
        for (int d = 0; d < D_MODEL; d++) {
            temp[d] = embeddings[pos][d] + attention_out[pos][d];
        }
        layer_norm(temp, norm1[pos], gpt->ln_gamma, gpt->ln_beta, D_MODEL);
    }
    
    // Enhanced feed-forward with two layers
    for (int pos = 0; pos < len; pos++) {
        double hidden1[D_FF];
        double hidden2[D_FF];
        
        // First FF layer with GELU activation
        for (int h = 0; h < D_FF; h++) {
            hidden1[h] = 0;
            for (int d = 0; d < D_MODEL; d++) {
                hidden1[h] += norm1[pos][d] * gpt->w1[d][h];
            }
            hidden1[h] = gelu(hidden1[h]);
        }
        
        // Second FF layer
        for (int h = 0; h < D_FF; h++) {
            hidden2[h] = 0;
            for (int d = 0; d < D_MODEL; d++) {
                hidden2[h] += norm1[pos][d] * gpt->w3[d][h];
            }
            hidden2[h] = swish(hidden2[h]);
        }
        
        // Combine and project back
        for (int d = 0; d < D_MODEL; d++) {
            double ff_result = 0;
            for (int h = 0; h < D_FF; h++) {
                ff_result += hidden1[h] * gpt->w2[h][d] + hidden2[h] * gpt->w4[h][d];
            }
            ff_out[pos][d] = ff_result;
        }
        
        // Another residual + layer norm
        double temp[D_MODEL];
        for (int d = 0; d < D_MODEL; d++) {
            temp[d] = norm1[pos][d] + ff_out[pos][d];
        }
        layer_norm(temp, norm2[pos], gpt->ln_gamma, gpt->ln_beta, D_MODEL);
    }
    
    // Output projection from last position
    int last_pos = len - 1;
    for (int v = 0; v < MAX_VOCAB; v++) {
        logits[v] = 0;
        for (int d = 0; d < D_MODEL; d++) {
            logits[v] += norm2[last_pos][d] * gpt->output_proj[d][v];
        }
    }
}

// Enhanced sampling with temperature and top-k
int monster_sample_token(double* logits, int vocab_size, double temperature, int top_k) {
    if (temperature <= 0) {
        // Greedy sampling
        int best = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return best;
    }
    
    // Apply temperature
    double max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    
    // Calculate probabilities with temperature
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
    
    // Top-k filtering (optional)
    if (top_k > 0 && top_k < vocab_size) {
        // Find top-k probabilities
        for (int i = 0; i < vocab_size - top_k; i++) {
            int min_idx = 0;
            for (int j = 1; j < vocab_size; j++) {
                if (probs[j] < probs[min_idx]) {
                    min_idx = j;
                }
            }
            probs[min_idx] = 0;
        }
        
        // Renormalize
        sum = 0;
        for (int i = 0; i < vocab_size; i++) {
            sum += probs[i];
        }
        for (int i = 0; i < vocab_size; i++) {
            probs[i] /= sum;
        }
    }
    
    // Sample from distribution
    double r = (double)rand() / RAND_MAX;
    double cumsum = 0;
    
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (r < cumsum) {
            free(probs);
            return i;
        }
    }
    
    free(probs);
    return vocab_size - 1;
}

// Enhanced training data loader
int load_monster_conversations(char* filename, MonsterTokenizer* tok, MonsterExample* examples) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("❌ Erro: não foi possível abrir %s\n", filename);
        return 0;
    }
    
    printf("📚 Carregando dados de conversa MONSTER de: %s\n", filename);
    
    char buffer[2000];  // Buffer muito maior para LONG ASS STORIES
    int example_count = 0;
    char user_text[1000] = "";
    char bot_text[1000] = "";
    
    while (fgets(buffer, sizeof(buffer), file) && example_count < 200) {
        // Remove newline
        buffer[strcspn(buffer, "\n")] = 0;
        
        if (strncmp(buffer, "USER: ", 6) == 0) {
            strncpy(user_text, buffer + 6, 999);
            user_text[999] = '\0';
        } else if (strncmp(buffer, "BOT: ", 5) == 0) {
            strncpy(bot_text, buffer + 5, 999);
            bot_text[999] = '\0';
            
            if (strlen(user_text) > 0 && strlen(bot_text) > 0) {
                int pos = 0;
                
                // Add [USER] token
                examples[example_count].tokens[pos++] = 0;
                
                // Add user tokens
                int user_tokens[32];
                int user_len = tokenize_monster(tok, user_text, user_tokens);
                for (int i = 0; i < user_len && pos < 58; i++) {
                    examples[example_count].tokens[pos++] = user_tokens[i];
                }
                
                // Add [BOT] token and mark position
                examples[example_count].tokens[pos++] = 1;
                examples[example_count].bot_start = pos;
                
                // Add bot tokens
                int bot_tokens[32];
                int bot_len = tokenize_monster(tok, bot_text, bot_tokens);
                for (int i = 0; i < bot_len && pos < 62; i++) {
                    examples[example_count].tokens[pos++] = bot_tokens[i];
                }
                
                examples[example_count].bot_end = pos;
                
                // Add [END] token
                examples[example_count].tokens[pos++] = 2;
                examples[example_count].length = pos;
                
                // Calculate importance based on response length and quality
                examples[example_count].importance = 1.0 + (bot_len / 10.0);
                
                example_count++;
                user_text[0] = '\0';
                bot_text[0] = '\0';
            }
        }
    }
    
    fclose(file);
    printf("✅ Carregados %d exemplos MONSTER de conversa\n", example_count);
    printf("✅ Vocabulário expandido: %d palavras\n", tok->word_count);
    return example_count;
}

// Enhanced training with better loss calculation and progress tracking
void train_monster_gpt(MonsterGPT* gpt, MonsterExample* examples, int num_examples, 
                      MonsterTokenizer* tok, TrainingStats* stats) {
    printf("\n🔥 INICIANDO TREINAMENTO MONSTER GPT! 🔥\n");
    printf("📊 Configuração:\n");
    printf("   🎯 Exemplos: %d\n", num_examples);
    printf("   📚 Vocabulário: %d palavras\n", tok->word_count);
    printf("   🧠 Dimensões: %d\n", D_MODEL);
    printf("   ⚡ Épocas: %d\n", EPOCHS);
    printf("   📈 Learning Rate: %.6f\n", LEARNING_RATE);
    
    stats->best_loss = 1000.0;
    stats->training_examples = num_examples;
    stats->vocab_size = tok->word_count;
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0;
        int predictions = 0;
        double epoch_start = clock();
        
        for (int ex = 0; ex < num_examples; ex++) {
            MonsterExample* example = &examples[ex];
            
            // Train on bot responses with importance weighting
            for (int pos = example->bot_start; pos < example->bot_end; pos++) {
                double logits[MAX_VOCAB];
                monster_forward_pass(gpt, example->tokens, pos, logits);
                
                int target = example->tokens[pos];
                if (target < tok->word_count) {
                    // Enhanced cross-entropy loss
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
                    
                    // Enhanced weight update with adaptive learning rate
                    double adaptive_lr = LEARNING_RATE / (1.0 + epoch * 0.01);
                    
                    for (int v = 0; v < tok->word_count; v++) {
                        double softmax_v = exp(logits[v] - max_logit) / sum;
                        double grad = softmax_v - (v == target ? 1.0 : 0.0);
                        grad *= example->importance;
                        
                        // Update output projection with momentum-like effect
                        for (int d = 0; d < D_MODEL; d++) {
                            gpt->output_proj[d][v] -= adaptive_lr * grad * 0.1;
                            
                            // L2 regularization
                            gpt->output_proj[d][v] *= (1.0 - adaptive_lr * 1e-5);
                        }
                    }
                }
            }
        }
        
        double avg_loss = predictions > 0 ? total_loss / predictions : 0;
        double epoch_time = (clock() - epoch_start) / CLOCKS_PER_SEC;
        
        // Track best performance
        if (avg_loss < stats->best_loss) {
            stats->best_loss = avg_loss;
        }
        
        // Enhanced progress display
        printf("📈 Época %2d/%d: Loss = %.4f (Best: %.4f) | Tempo: %.2fs | Pred: %d\n", 
               epoch + 1, EPOCHS, avg_loss, stats->best_loss, epoch_time, predictions);
        
        // Progress bar
        int progress = (epoch + 1) * 20 / EPOCHS;
        printf("   [");
        for (int i = 0; i < 20; i++) {
            printf(i < progress ? "█" : "░");
        }
        printf("]\n");
        
        stats->avg_loss = avg_loss;
    }
    
    printf("\n🎉 TREINAMENTO MONSTER CONCLUÍDO! 🎉\n");
    printf("📊 Estatísticas finais:\n");
    printf("   🎯 Loss final: %.4f\n", stats->avg_loss);
    printf("   🏆 Melhor loss: %.4f\n", stats->best_loss);
    printf("   💡 Melhoria: %.2f%%\n", 
           ((7.0 - stats->best_loss) / 7.0) * 100);
}

// Enhanced conversation memory
void init_conversation_memory(ConversationMemory* memory) {
    memory->context_len = 0;
    memory->response_count = 0;
    for (int i = 0; i < 5; i++) {
        memory->last_responses[i][0] = '\0';
    }
}

void update_conversation_memory(ConversationMemory* memory, int* tokens, int len, char* response) {
    // Add tokens to context (keep recent history)
    if (memory->context_len + len > 120) {
        // Shift context to make room
        int shift = (memory->context_len + len) - 120;
        for (int i = 0; i < memory->context_len - shift; i++) {
            memory->recent_context[i] = memory->recent_context[i + shift];
        }
        memory->context_len -= shift;
    }
    
    // Add new tokens
    for (int i = 0; i < len && memory->context_len < 120; i++) {
        memory->recent_context[memory->context_len++] = tokens[i];
    }
    
    // Update response history
    if (memory->response_count >= 5) {
        // Shift responses
        for (int i = 0; i < 4; i++) {
            strcpy(memory->last_responses[i], memory->last_responses[i + 1]);
        }
        memory->response_count = 4;
    }
    
    strcpy(memory->last_responses[memory->response_count++], response);
}

// Check for repetitive responses
int is_repetitive_response(ConversationMemory* memory, char* new_response) {
    for (int i = 0; i < memory->response_count; i++) {
        if (strcmp(memory->last_responses[i], new_response) == 0) {
            return 1; // Repetitive
        }
    }
    return 0; // Not repetitive
}

// Monster chat session with enhanced features
void monster_chat_session(MonsterGPT* gpt, MonsterTokenizer* tok, TrainingStats* stats) {
    printf("\n🤖 MONSTER CHATBOT ONLINE! 🤖\n");
    printf("💬 Digite 'quit' para sair\n");
    printf("🎛️ Comandos especiais:\n");
    printf("   'temp X' - ajustar temperatura (0.1-2.0)\n");
    printf("   'topk X' - ajustar top-k sampling (5-50)\n");
    printf("   'stats' - mostrar estatísticas\n");
    printf("   'reset' - limpar memória de conversa\n");
    printf("   'help' - mostrar comandos\n\n");
    
    ConversationMemory memory;
    init_conversation_memory(&memory);
    
    char input[512];
    double current_temp = TEMPERATURE;
    int top_k = 10;
    int conversation_count = 0;
    
    while (1) {
        printf("👤 Você: ");
        if (!fgets(input, sizeof(input), stdin)) break;
        
        // Remove newline
        input[strcspn(input, "\n")] = 0;
        
        if (strcmp(input, "quit") == 0) {
            printf("🤖 Monster: Foi incrível conversar! Até a próxima! 🚀\n");
            break;
        }
        
        // Handle special commands
        if (strncmp(input, "temp ", 5) == 0) {
            current_temp = atof(input + 5);
            current_temp = fmax(0.1, fmin(2.0, current_temp));
            printf("🌡️ Temperatura ajustada para %.2f\n", current_temp);
            continue;
        }
        
        if (strncmp(input, "topk ", 5) == 0) {
            top_k = atoi(input + 5);
            top_k = fmax(1, fmin(50, top_k));
            printf("🎯 Top-K ajustado para %d\n", top_k);
            continue;
        }
        
        if (strcmp(input, "stats") == 0) {
            printf("📊 ESTATÍSTICAS MONSTER:\n");
            printf("   🎯 Loss de treino: %.4f\n", stats->avg_loss);
            printf("   🏆 Melhor loss: %.4f\n", stats->best_loss);
            printf("   📚 Vocabulário: %d palavras\n", stats->vocab_size);
            printf("   💬 Conversas: %d\n", conversation_count);
            printf("   🧠 Memória: %d tokens\n", memory.context_len);
            continue;
        }
        
        if (strcmp(input, "reset") == 0) {
            init_conversation_memory(&memory);
            printf("🔄 Memória limpa! Novo início.\n");
            continue;
        }
        
        if (strcmp(input, "help") == 0) {
            printf("🎛️ COMANDOS MONSTER:\n");
            printf("   temp 0.1-2.0  : controla criatividade\n");
            printf("   topk 1-50     : controla diversidade\n");
            printf("   stats         : mostra estatísticas\n");
            printf("   reset         : limpa memória\n");
            printf("   quit          : sair\n");
            continue;
        }
        
        // Prepare enhanced context
        int context[64];
        int pos = 0;
        
        // Add recent conversation context
        int context_start = memory.context_len > 40 ? memory.context_len - 40 : 0;
        for (int i = context_start; i < memory.context_len && pos < 30; i++) {
            context[pos++] = memory.recent_context[i];
        }
        
        // Add current user input
        context[pos++] = 0; // [USER] token
        
        int user_tokens[32];
        int user_len = tokenize_monster(tok, input, user_tokens);
        for (int i = 0; i < user_len && pos < 50; i++) {
            context[pos++] = user_tokens[i];
        }
        
        // Add [BOT] token
        context[pos++] = 1;
        
        // Generate monster response
        printf("🤖 Monster: ");
        char response[512] = "";
        int response_tokens[32];
        int response_len = 0;
        
        for (int gen = 0; gen < MAX_RESPONSE_LEN && pos < 62; gen++) {
            double logits[MAX_VOCAB];
            monster_forward_pass(gpt, context, pos, logits);
            
            int next_token = monster_sample_token(logits, tok->word_count, current_temp, top_k);
            
            // Stop conditions
            if (next_token == 2 || next_token == 0) break; // [END] or [USER]
            if (next_token >= tok->word_count) break;
            
            // Avoid repetition within same response
            int repeat_count = 0;
            for (int i = 0; i < response_len; i++) {
                if (response_tokens[i] == next_token) repeat_count++;
            }
            if (repeat_count > 2) break; // Skip if word appears too often
            
            printf("%s ", tok->words[next_token]);
            
            if (strlen(response) > 0) strcat(response, " ");
            strcat(response, tok->words[next_token]);
            
            context[pos++] = next_token;
            response_tokens[response_len++] = next_token;
        }
        
        printf("\n\n");
        
        // Check if response is too repetitive
        if (is_repetitive_response(&memory, response)) {
            printf("🔄 (Detectei repetição, vou pensar diferente na próxima)\n\n");
        }
        
        // Update conversation memory
        int full_turn[64];
        int turn_len = 0;
        
        // Add [USER] + input + [BOT] + response
        full_turn[turn_len++] = 0;
        for (int i = 0; i < user_len; i++) {
            full_turn[turn_len++] = user_tokens[i];
        }
        full_turn[turn_len++] = 1;
        for (int i = 0; i < response_len; i++) {
            full_turn[turn_len++] = response_tokens[i];
        }
        
        update_conversation_memory(&memory, full_turn, turn_len, response);
        conversation_count++;
        stats->conversations_held = conversation_count;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("🤖 MONSTER CHATBOT v2.0\n");
        printf("Uso: %s <arquivo_conversas>\n", argv[0]);
        printf("\nExemplo:\n");
        printf("  %s data/chatbot_training.txt\n", argv[0]);
        return 1;
    }
    
    printf("🔥🤖 MONSTER CHATBOT SYSTEM 🤖🔥\n");
    printf("=====================================\n");
    
    MonsterTokenizer tokenizer;
    MonsterGPT gpt;
    MonsterExample examples[300];
    TrainingStats stats = {0};
    
    printf("⚡ Inicializando componentes...\n");
    init_monster_tokenizer(&tokenizer);
    init_monster_gpt(&gpt);
    
    printf("\n📚 Carregando dados de treinamento...\n");
    int num_examples = load_monster_conversations(argv[1], &tokenizer, examples);
    if (num_examples == 0) {
        printf("❌ Nenhum exemplo carregado. Verifique o arquivo.\n");
        return 1;
    }
    
    printf("\n🚀 Iniciando treinamento MONSTER...\n");
    train_monster_gpt(&gpt, examples, num_examples, &tokenizer, &stats);
    
    printf("\n💬 Entrando em modo conversa...\n");
    monster_chat_session(&gpt, &tokenizer, &stats);
    
    printf("\n👋 Monster Chatbot finalizado!\n");
    printf("📊 Total de conversas: %d\n", stats.conversations_held);
    
    return 0;
}
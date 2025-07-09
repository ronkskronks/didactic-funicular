#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// TURBO STORYTELLER - Versão otimizada para LONG ASS STORIES 🚀
#define MAX_VOCAB 250        // Mais compacto mas ainda poderoso
#define CONTEXT_LEN 20       // Contexto bom para narrativas
#define D_MODEL 24           // Balanceado para performance
#define D_FF 48              // 2x D_MODEL
#define EPOCHS 15            // Menos épocas para ser mais rápido
#define LEARNING_RATE 0.01   // LR mais alto para convergência rápida
#define MAX_RESPONSE_LEN 30  // Respostas épicas

// Storyteller tokenizer
typedef struct {
    char words[MAX_VOCAB][64];
    int word_count;
} StoryTokenizer;

// Turbo storyteller model
typedef struct {
    double token_embed[MAX_VOCAB][D_MODEL];
    double pos_embed[CONTEXT_LEN][D_MODEL];
    double w1[D_MODEL][D_FF];
    double w2[D_FF][D_MODEL];
    double output_proj[D_MODEL][MAX_VOCAB];
} TurboStoryteller;

// Story example
typedef struct {
    int tokens[48];
    int length;
    int bot_start;
    int bot_end;
} StoryExample;

void init_story_tokenizer(StoryTokenizer* tok) {
    strcpy(tok->words[0], "[USER]");
    strcpy(tok->words[1], "[BOT]");
    strcpy(tok->words[2], "[END]");
    strcpy(tok->words[3], "[STORY]");
    tok->word_count = 4;
    printf("📚 StoryTokenizer initialized\n");
}

int add_story_word(StoryTokenizer* tok, char* word) {
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

int tokenize_story(StoryTokenizer* tok, char* text, int* tokens) {
    char text_copy[2000];
    strncpy(text_copy, text, 1999);
    text_copy[1999] = '\0';
    
    int count = 0;
    char* word = strtok(text_copy, " \n\t.,!?;:()[]{}\"'");
    
    while (word != NULL && count < 24) {
        // Convert to lowercase
        for (int i = 0; word[i]; i++) {
            if (word[i] >= 'A' && word[i] <= 'Z') {
                word[i] = word[i] + 32;
            }
        }
        
        // Skip muito short words mas manter importantes
        if (strlen(word) >= 2 || 
            strcmp(word, "e") == 0 || strcmp(word, "o") == 0 || 
            strcmp(word, "a") == 0 || strcmp(word, "é") == 0 ||
            strcmp(word, "é") == 0 || strcmp(word, "me") == 0) {
            
            int id = add_story_word(tok, word);
            if (id >= 0) {
                tokens[count++] = id;
            }
        }
        
        word = strtok(NULL, " \n\t.,!?;:()[]{}\"'");
    }
    
    return count;
}

void init_turbo_storyteller(TurboStoryteller* model) {
    srand(time(NULL));
    printf("🚀 Initializing TurboStoryteller...\n");
    
    // Xavier initialization for better convergence
    double token_scale = sqrt(1.0 / D_MODEL);
    double ff_scale = sqrt(2.0 / D_FF);
    
    // Token embeddings
    for (int i = 0; i < MAX_VOCAB; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            model->token_embed[i][j] = ((double)rand() / RAND_MAX - 0.5) * token_scale;
        }
    }
    
    // Position embeddings
    for (int i = 0; i < CONTEXT_LEN; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            model->pos_embed[i][j] = sin(i / pow(10000.0, (double)j / D_MODEL)) * 0.1;
        }
    }
    
    // Feed forward
    for (int i = 0; i < D_MODEL; i++) {
        for (int j = 0; j < D_FF; j++) {
            model->w1[i][j] = ((double)rand() / RAND_MAX - 0.5) * ff_scale;
        }
    }
    
    for (int i = 0; i < D_FF; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            model->w2[i][j] = ((double)rand() / RAND_MAX - 0.5) * ff_scale;
        }
    }
    
    // Output projection
    for (int i = 0; i < D_MODEL; i++) {
        for (int j = 0; j < MAX_VOCAB; j++) {
            model->output_proj[i][j] = ((double)rand() / RAND_MAX - 0.5) * token_scale;
        }
    }
    
    printf("✅ TurboStoryteller initialized!\n");
    printf("   📊 Parameters: %d\n", 
           MAX_VOCAB * D_MODEL + CONTEXT_LEN * D_MODEL + 
           D_MODEL * D_FF + D_FF * D_MODEL + D_MODEL * MAX_VOCAB);
}

double swish_activation(double x) {
    return x / (1.0 + exp(-x));
}

void turbo_forward_pass(TurboStoryteller* model, int* tokens, int len, double* logits) {
    double embeddings[CONTEXT_LEN][D_MODEL];
    double ff_out[CONTEXT_LEN][D_MODEL];
    
    // Get embeddings with positional encoding
    for (int pos = 0; pos < len && pos < CONTEXT_LEN; pos++) {
        for (int d = 0; d < D_MODEL; d++) {
            embeddings[pos][d] = model->token_embed[tokens[pos]][d] + model->pos_embed[pos][d];
        }
    }
    
    // Enhanced feed-forward
    for (int pos = 0; pos < len; pos++) {
        double hidden[D_FF];
        
        // First layer with Swish activation
        for (int h = 0; h < D_FF; h++) {
            hidden[h] = 0;
            for (int d = 0; d < D_MODEL; d++) {
                hidden[h] += embeddings[pos][d] * model->w1[d][h];
            }
            hidden[h] = swish_activation(hidden[h]);
        }
        
        // Second layer with residual
        for (int d = 0; d < D_MODEL; d++) {
            ff_out[pos][d] = 0;
            for (int h = 0; h < D_FF; h++) {
                ff_out[pos][d] += hidden[h] * model->w2[h][d];
            }
            ff_out[pos][d] += embeddings[pos][d]; // Residual connection
        }
    }
    
    // Output projection
    int last_pos = len - 1;
    for (int v = 0; v < MAX_VOCAB; v++) {
        logits[v] = 0;
        for (int d = 0; d < D_MODEL; d++) {
            logits[v] += ff_out[last_pos][d] * model->output_proj[d][v];
        }
    }
}

int sample_with_temperature(double* logits, int vocab_size, double temperature) {
    // Temperature sampling for creativity
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
    
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }
    
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

int load_story_data(char* filename, StoryTokenizer* tok, StoryExample* examples) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("❌ Erro: não foi possível abrir %s\n", filename);
        return 0;
    }
    
    printf("📚 Carregando dados narrativos de: %s\n", filename);
    
    char buffer[3000];  // Buffer grande para histórias longas
    int example_count = 0;
    char user_text[1500] = "";
    char bot_text[1500] = "";
    
    while (fgets(buffer, sizeof(buffer), file) && example_count < 150) {
        buffer[strcspn(buffer, "\n")] = 0;
        
        // Skip headers e comments
        if (buffer[0] == '#' || strlen(buffer) < 5) continue;
        
        if (strncmp(buffer, "USER: ", 6) == 0) {
            strncpy(user_text, buffer + 6, 1499);
            user_text[1499] = '\0';
        } else if (strncmp(buffer, "BOT: ", 5) == 0) {
            strncpy(bot_text, buffer + 5, 1499);
            bot_text[1499] = '\0';
            
            if (strlen(user_text) > 0 && strlen(bot_text) > 0) {
                int pos = 0;
                
                // Add [USER] token
                examples[example_count].tokens[pos++] = 0;
                
                // Add user tokens
                int user_tokens[24];
                int user_len = tokenize_story(tok, user_text, user_tokens);
                for (int i = 0; i < user_len && pos < 42; i++) {
                    examples[example_count].tokens[pos++] = user_tokens[i];
                }
                
                // Add [BOT] token
                examples[example_count].tokens[pos++] = 1;
                examples[example_count].bot_start = pos;
                
                // Add bot tokens
                int bot_tokens[24];
                int bot_len = tokenize_story(tok, bot_text, bot_tokens);
                for (int i = 0; i < bot_len && pos < 46; i++) {
                    examples[example_count].tokens[pos++] = bot_tokens[i];
                }
                
                examples[example_count].bot_end = pos;
                
                // Add [END] token
                examples[example_count].tokens[pos++] = 2;
                examples[example_count].length = pos;
                
                example_count++;
                user_text[0] = '\0';
                bot_text[0] = '\0';
            }
        }
    }
    
    fclose(file);
    printf("✅ Carregados %d exemplos narrativos\n", example_count);
    printf("✅ Vocabulário: %d palavras\n", tok->word_count);
    return example_count;
}

void train_turbo_storyteller(TurboStoryteller* model, StoryExample* examples, 
                           int num_examples, StoryTokenizer* tok) {
    printf("\n🔥 TREINAMENTO TURBO STORYTELLER! 🔥\n");
    printf("📊 Configuração OTIMIZADA:\n");
    printf("   🎯 Exemplos: %d\n", num_examples);
    printf("   📚 Vocabulário: %d\n", tok->word_count);
    printf("   🧠 Modelo: %dx%d\n", D_MODEL, D_FF);
    printf("   ⚡ Épocas: %d\n", EPOCHS);
    
    double best_loss = 1000.0;
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0;
        int predictions = 0;
        double epoch_start = clock();
        
        for (int ex = 0; ex < num_examples; ex++) {
            StoryExample* example = &examples[ex];
            
            // Train on bot responses
            for (int pos = example->bot_start; pos < example->bot_end; pos++) {
                double logits[MAX_VOCAB];
                turbo_forward_pass(model, example->tokens, pos, logits);
                
                int target = example->tokens[pos];
                if (target < tok->word_count) {
                    // Cross-entropy loss
                    double max_logit = logits[0];
                    for (int i = 1; i < tok->word_count; i++) {
                        if (logits[i] > max_logit) max_logit = logits[i];
                    }
                    
                    double sum = 0;
                    for (int i = 0; i < tok->word_count; i++) {
                        sum += exp(logits[i] - max_logit);
                    }
                    
                    double prob = exp(logits[target] - max_logit) / sum;
                    total_loss += -log(prob + 1e-10);
                    predictions++;
                    
                    // Weight update with momentum-like effect
                    for (int v = 0; v < tok->word_count; v++) {
                        double softmax_v = exp(logits[v] - max_logit) / sum;
                        double grad = softmax_v - (v == target ? 1.0 : 0.0);
                        
                        for (int d = 0; d < D_MODEL; d++) {
                            model->output_proj[d][v] -= LEARNING_RATE * grad * 0.2;
                        }
                    }
                }
            }
        }
        
        double avg_loss = predictions > 0 ? total_loss / predictions : 0;
        double epoch_time = (clock() - epoch_start) / CLOCKS_PER_SEC;
        
        if (avg_loss < best_loss) best_loss = avg_loss;
        
        // Progress display
        printf("📈 Época %2d/%d: Loss = %.4f (Best: %.4f) | %.2fs\n", 
               epoch + 1, EPOCHS, avg_loss, best_loss, epoch_time);
        
        // Progress bar
        int progress = (epoch + 1) * 20 / EPOCHS;
        printf("   [");
        for (int i = 0; i < 20; i++) {
            printf(i < progress ? "█" : "░");
        }
        printf("]\n");
    }
    
    printf("\n🎉 TREINAMENTO TURBO CONCLUÍDO! 🎉\n");
    printf("🏆 Melhor loss: %.4f\n", best_loss);
}

void storytelling_session(TurboStoryteller* model, StoryTokenizer* tok) {
    printf("\n📚 TURBO STORYTELLER ONLINE! 📚\n");
    printf("💬 Digite sua solicitação ou 'quit' para sair\n");
    printf("🎭 Comandos: 'temp X' para ajustar criatividade\n\n");
    
    char input[512];
    double temperature = 0.8;
    
    while (1) {
        printf("👤 Você: ");
        if (!fgets(input, sizeof(input), stdin)) break;
        
        input[strcspn(input, "\n")] = 0;
        
        if (strcmp(input, "quit") == 0) {
            printf("📚 Storyteller: Foi um prazer criar histórias com você! 🌟\n");
            break;
        }
        
        if (strncmp(input, "temp ", 5) == 0) {
            temperature = atof(input + 5);
            temperature = fmax(0.2, fmin(2.0, temperature));
            printf("🌡️ Criatividade ajustada para %.2f\n", temperature);
            continue;
        }
        
        // Prepare context
        int context[32];
        int pos = 0;
        
        // Add [USER] token
        context[pos++] = 0;
        
        // Add user input
        int user_tokens[20];
        int user_len = tokenize_story(tok, input, user_tokens);
        for (int i = 0; i < user_len && pos < 25; i++) {
            context[pos++] = user_tokens[i];
        }
        
        // Add [BOT] token
        context[pos++] = 1;
        
        // Generate story response
        printf("📚 Storyteller: ");
        for (int gen = 0; gen < MAX_RESPONSE_LEN && pos < 30; gen++) {
            double logits[MAX_VOCAB];
            turbo_forward_pass(model, context, pos, logits);
            
            int next_token = sample_with_temperature(logits, tok->word_count, temperature);
            
            // Stop conditions
            if (next_token == 2 || next_token == 0 || next_token >= tok->word_count) break;
            
            printf("%s ", tok->words[next_token]);
            context[pos++] = next_token;
        }
        printf("\n\n");
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("📚 TURBO STORYTELLER v1.0\n");
        printf("Uso: %s <arquivo_historias>\n", argv[0]);
        return 1;
    }
    
    printf("🔥📚 TURBO STORYTELLER SYSTEM 📚🔥\n");
    printf("====================================\n");
    
    StoryTokenizer tokenizer;
    TurboStoryteller model;
    StoryExample examples[200];
    
    init_story_tokenizer(&tokenizer);
    init_turbo_storyteller(&model);
    
    int num_examples = load_story_data(argv[1], &tokenizer, examples);
    if (num_examples == 0) {
        printf("❌ Nenhum exemplo carregado\n");
        return 1;
    }
    
    train_turbo_storyteller(&model, examples, num_examples, &tokenizer);
    storytelling_session(&model, &tokenizer);
    
    return 0;
}
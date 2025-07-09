#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_VOCAB 100
#define CONTEXT_LEN 12
#define D_MODEL 16
#define D_FF 32
#define EPOCHS 10
#define LEARNING_RATE 0.01

// Simple tokenizer
typedef struct {
    char words[MAX_VOCAB][50];
    int word_count;
} SimpleTokenizer;

// Simple GPT
typedef struct {
    double token_embed[MAX_VOCAB][D_MODEL];
    double pos_embed[CONTEXT_LEN][D_MODEL];
    double w1[D_MODEL][D_FF];
    double w2[D_FF][D_MODEL];
    double output_proj[D_MODEL][MAX_VOCAB];
} SimpleChatGPT;

// Conversation example
typedef struct {
    int tokens[32];
    int length;
    int bot_start;
} ConversationExample;

void init_tokenizer(SimpleTokenizer* tok) {
    // Special tokens
    strcpy(tok->words[0], "[USER]");
    strcpy(tok->words[1], "[BOT]");
    strcpy(tok->words[2], "[END]");
    tok->word_count = 3;
}

int add_word(SimpleTokenizer* tok, char* word) {
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

int tokenize_simple(SimpleTokenizer* tok, char* text, int* tokens) {
    char text_copy[500];
    strcpy(text_copy, text);
    
    int count = 0;
    char* word = strtok(text_copy, " \n\t.,!?;");
    
    while (word != NULL && count < 16) {
        // Convert to lowercase
        for (int i = 0; word[i]; i++) {
            if (word[i] >= 'A' && word[i] <= 'Z') {
                word[i] = word[i] + 32;
            }
        }
        
        int id = add_word(tok, word);
        if (id >= 0) {
            tokens[count++] = id;
        }
        word = strtok(NULL, " \n\t.,!?;");
    }
    
    return count;
}

void init_gpt(SimpleChatGPT* gpt) {
    srand(time(NULL));
    
    // Initialize embeddings
    for (int i = 0; i < MAX_VOCAB; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            gpt->token_embed[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
    
    for (int i = 0; i < CONTEXT_LEN; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            gpt->pos_embed[i][j] = sin(i / 100.0) * 0.1;
        }
    }
    
    // Initialize weights
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
    
    printf("✅ SimpleChatGPT initialized!\n");
}

double relu(double x) {
    return x > 0 ? x : 0;
}

void forward_pass(SimpleChatGPT* gpt, int* tokens, int len, double* logits) {
    double embeddings[CONTEXT_LEN][D_MODEL];
    double ff_out[CONTEXT_LEN][D_MODEL];
    
    // Get embeddings
    for (int pos = 0; pos < len && pos < CONTEXT_LEN; pos++) {
        for (int d = 0; d < D_MODEL; d++) {
            embeddings[pos][d] = gpt->token_embed[tokens[pos]][d] + gpt->pos_embed[pos][d];
        }
    }
    
    // Simple feed forward (skip attention for now)
    for (int pos = 0; pos < len; pos++) {
        double hidden[D_FF];
        
        // First layer
        for (int h = 0; h < D_FF; h++) {
            hidden[h] = 0;
            for (int d = 0; d < D_MODEL; d++) {
                hidden[h] += embeddings[pos][d] * gpt->w1[d][h];
            }
            hidden[h] = relu(hidden[h]);
        }
        
        // Second layer
        for (int d = 0; d < D_MODEL; d++) {
            ff_out[pos][d] = 0;
            for (int h = 0; h < D_FF; h++) {
                ff_out[pos][d] += hidden[h] * gpt->w2[h][d];
            }
            ff_out[pos][d] += embeddings[pos][d]; // residual
        }
    }
    
    // Output projection
    int last_pos = len - 1;
    for (int v = 0; v < MAX_VOCAB; v++) {
        logits[v] = 0;
        for (int d = 0; d < D_MODEL; d++) {
            logits[v] += ff_out[last_pos][d] * gpt->output_proj[d][v];
        }
    }
}

int sample_greedy(double* logits, int vocab_size) {
    int best = 0;
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

int load_conversations(char* filename, SimpleTokenizer* tok, ConversationExample* examples) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Erro: não foi possível abrir %s\n", filename);
        return 0;
    }
    
    char buffer[500];
    int example_count = 0;
    char user_text[256] = "";
    char bot_text[256] = "";
    
    while (fgets(buffer, sizeof(buffer), file) && example_count < 50) {
        buffer[strcspn(buffer, "\n")] = 0;
        
        if (strncmp(buffer, "USER: ", 6) == 0) {
            strcpy(user_text, buffer + 6);
        } else if (strncmp(buffer, "BOT: ", 5) == 0) {
            strcpy(bot_text, buffer + 5);
            
            if (strlen(user_text) > 0 && strlen(bot_text) > 0) {
                int pos = 0;
                
                // Add [USER] token
                examples[example_count].tokens[pos++] = 0;
                
                // Add user tokens
                int user_tokens[16];
                int user_len = tokenize_simple(tok, user_text, user_tokens);
                for (int i = 0; i < user_len && pos < 30; i++) {
                    examples[example_count].tokens[pos++] = user_tokens[i];
                }
                
                // Add [BOT] token and mark position
                examples[example_count].tokens[pos++] = 1;
                examples[example_count].bot_start = pos;
                
                // Add bot tokens
                int bot_tokens[16];
                int bot_len = tokenize_simple(tok, bot_text, bot_tokens);
                for (int i = 0; i < bot_len && pos < 30; i++) {
                    examples[example_count].tokens[pos++] = bot_tokens[i];
                }
                
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
    printf("✅ Carregados %d exemplos de conversa\n", example_count);
    return example_count;
}

void train_simple(SimpleChatGPT* gpt, ConversationExample* examples, int num_examples, SimpleTokenizer* tok) {
    printf("\n🚀 Treinando SimpleChatGPT...\n");
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0;
        int predictions = 0;
        
        for (int ex = 0; ex < num_examples; ex++) {
            ConversationExample* example = &examples[ex];
            
            // Train on bot responses
            for (int pos = example->bot_start; pos < example->length - 1; pos++) {
                double logits[MAX_VOCAB];
                forward_pass(gpt, example->tokens, pos, logits);
                
                int target = example->tokens[pos];
                if (target < tok->word_count) {
                    // Simple cross-entropy loss
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
                    
                    // Update weights (output layer only)
                    for (int v = 0; v < tok->word_count; v++) {
                        double softmax_v = exp(logits[v] - max_logit) / sum;
                        double grad = softmax_v - (v == target ? 1.0 : 0.0);
                        
                        for (int d = 0; d < D_MODEL; d++) {
                            gpt->output_proj[d][v] -= LEARNING_RATE * grad * 0.1;
                        }
                    }
                }
            }
        }
        
        double avg_loss = predictions > 0 ? total_loss / predictions : 0;
        printf("Época %d/%d: Loss = %.4f\n", epoch + 1, EPOCHS, avg_loss);
    }
    
    printf("✅ Treinamento concluído!\n");
}

void chat_with_bot(SimpleChatGPT* gpt, SimpleTokenizer* tok) {
    printf("\n💬 Modo Chat Ativo! Digite 'quit' para sair\n\n");
    
    char input[256];
    while (1) {
        printf("Você: ");
        if (!fgets(input, sizeof(input), stdin)) break;
        
        input[strcspn(input, "\n")] = 0;
        
        if (strcmp(input, "quit") == 0) {
            printf("🤖 Bot: Tchau! Foi ótimo conversar!\n");
            break;
        }
        
        // Prepare context
        int context[32];
        int pos = 0;
        
        // Add [USER] token
        context[pos++] = 0;
        
        // Add user input
        int user_tokens[16];
        int user_len = tokenize_simple(tok, input, user_tokens);
        for (int i = 0; i < user_len && pos < 20; i++) {
            context[pos++] = user_tokens[i];
        }
        
        // Add [BOT] token
        context[pos++] = 1;
        
        // Generate response
        printf("🤖 Bot: ");
        for (int gen = 0; gen < 10 && pos < 30; gen++) {
            double logits[MAX_VOCAB];
            forward_pass(gpt, context, pos, logits);
            
            int next_token = sample_greedy(logits, tok->word_count);
            
            // Stop if end token or special token
            if (next_token == 2 || next_token == 0) break;
            
            printf("%s ", tok->words[next_token]);
            context[pos++] = next_token;
        }
        printf("\n\n");
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Uso: %s <arquivo_conversas>\n", argv[0]);
        return 1;
    }
    
    printf("🤖 SimpleChatGPT - Chatbot em C\n");
    
    SimpleTokenizer tokenizer;
    SimpleChatGPT gpt;
    ConversationExample examples[100];
    
    init_tokenizer(&tokenizer);
    init_gpt(&gpt);
    
    int num_examples = load_conversations(argv[1], &tokenizer, examples);
    if (num_examples == 0) {
        printf("Nenhum exemplo carregado\n");
        return 1;
    }
    
    printf("Vocabulário: %d palavras\n", tokenizer.word_count);
    
    train_simple(&gpt, examples, num_examples, &tokenizer);
    chat_with_bot(&gpt, &tokenizer);
    
    return 0;
}
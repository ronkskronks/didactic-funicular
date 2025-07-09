#!/usr/bin/env python3

"""
TRANSFORMER ATTENTION MECHANISM - ARITHMETIC POVERTY EDITION
===========================================================

The CRYPTIC algorithm that powers ChatGPT, GPT-4, and modern AI...
implemented using ONLY +, -, *, / operations!

🤯 WHAT IS TRANSFORMER ATTENTION?
- The core mechanism behind all modern language models
- Allows AI to "pay attention" to different parts of input simultaneously  
- Enables understanding of context and relationships in text
- The secret sauce that made GPT possible

🔥 WHAT WE'RE IMPLEMENTING:
- Multi-Head Self-Attention (the heart of transformers)
- Query, Key, Value matrix transformations
- Attention score calculations 
- Softmax approximation using only basic arithmetic
- Multiple attention heads working in parallel

❌ FORBIDDEN OPERATIONS:
- No exp(), log(), sqrt()
- No max(), min(), abs()
- No numpy, torch, tensorflow
- No fancy math libraries
- ONLY +, -, *, / allowed!

🎯 EDUCATIONAL GOAL:
Prove that the "magical" attention mechanism is just simple arithmetic
operations repeated many times. No actual magic - just math!

Architecture: 4 tokens -> 2 attention heads -> combined output
This is a REAL transformer attention layer, just simplified!
"""

class ArithmeticTransformerAttention:
    """
    The attention mechanism that powers modern AI, implemented with
    the mathematical complexity of a pocket calculator!
    
    This is the ACTUAL algorithm used in GPT models, stripped down
    to show the pure arithmetic underneath the "magic"
    """
    
    def __init__(self):
        """Initialize transformer attention with basic arithmetic weights"""
        print("🤖 TRANSFORMER ATTENTION - ARITHMETIC EDITION")
        print("=" * 60)
        print("🧠 The algorithm that powers ChatGPT, implemented with +,-,*,/")
        print("🔮 Demystifying the 'magic' of modern AI attention!")
        
        # Model dimensions
        self.d_model = 4      # Embedding dimension (input size)
        self.d_k = 2          # Key/Query dimension  
        self.d_v = 2          # Value dimension
        self.num_heads = 2    # Number of attention heads
        self.seq_length = 4   # Sequence length (number of tokens)
        
        print(f"📐 Model dimension: {self.d_model}")
        print(f"🔑 Key/Query dimension: {self.d_k}")  
        print(f"💎 Value dimension: {self.d_v}")
        print(f"🎯 Number of attention heads: {self.num_heads}")
        print(f"📝 Sequence length: {self.seq_length}")
        
        # Initialize weight matrices using division (our "random" initialization)
        # These are the learned parameters that make attention work!
        
        # HEAD 1 WEIGHTS
        print("\n🎯 Initializing Attention Head 1 weights...")
        
        # Query weights for head 1 (d_model x d_k)
        self.W_q1 = [
            [1.0/2.0, 1.0/3.0],   # First row: [0.5, 0.333]
            [1.0/4.0, 1.0/5.0],   # Second row: [0.25, 0.2]
            [1.0/6.0, 1.0/7.0],   # Third row: [0.167, 0.143]
            [1.0/8.0, 1.0/9.0]    # Fourth row: [0.125, 0.111]
        ]
        
        # Key weights for head 1
        self.W_k1 = [
            [1.0/3.0, 1.0/4.0],   
            [1.0/5.0, 1.0/6.0],   
            [1.0/7.0, 1.0/8.0],   
            [1.0/9.0, 1.0/10.0]   
        ]
        
        # Value weights for head 1
        self.W_v1 = [
            [1.0/4.0, 1.0/5.0],   
            [1.0/6.0, 1.0/7.0],   
            [1.0/8.0, 1.0/9.0],   
            [1.0/10.0, 1.0/11.0]  
        ]
        
        # HEAD 2 WEIGHTS
        print("🎯 Initializing Attention Head 2 weights...")
        
        # Query weights for head 2
        self.W_q2 = [
            [1.0/11.0, 1.0/12.0], 
            [1.0/13.0, 1.0/14.0], 
            [1.0/15.0, 1.0/16.0], 
            [1.0/17.0, 1.0/18.0]  
        ]
        
        # Key weights for head 2
        self.W_k2 = [
            [1.0/12.0, 1.0/13.0], 
            [1.0/14.0, 1.0/15.0], 
            [1.0/16.0, 1.0/17.0], 
            [1.0/18.0, 1.0/19.0]  
        ]
        
        # Value weights for head 2
        self.W_v2 = [
            [1.0/13.0, 1.0/14.0], 
            [1.0/15.0, 1.0/16.0], 
            [1.0/17.0, 1.0/18.0], 
            [1.0/19.0, 1.0/20.0]  
        ]
        
        # Output projection weights (combines multiple heads)
        self.W_o = [
            [1.0/2.0, 1.0/3.0, 1.0/4.0, 1.0/5.0],
            [1.0/6.0, 1.0/7.0, 1.0/8.0, 1.0/9.0],
            [1.0/10.0, 1.0/11.0, 1.0/12.0, 1.0/13.0],
            [1.0/14.0, 1.0/15.0, 1.0/16.0, 1.0/17.0]
        ]
        
        print("✅ All attention weights initialized!")
    
    def arithmetic_matrix_multiply(self, A, B):
        """
        Matrix multiplication using only +, -, *, /
        
        This is the CORE operation of transformers - everything else
        is just different ways of organizing matrix multiplications!
        """
        rows_A = len(A)
        cols_A = len(A[0])
        cols_B = len(B[0])
        
        # Initialize result matrix
        result = []
        for i in range(rows_A):
            row = []
            for j in range(cols_B):
                row.append(0.0)
            result.append(row)
        
        # Perform multiplication: C[i][j] = Σ(A[i][k] * B[k][j])
        for i in range(rows_A):
            for j in range(cols_B):
                sum_value = 0.0
                for k in range(cols_A):
                    product = A[i][k] * B[k][j]
                    sum_value = sum_value + product
                result[i][j] = sum_value
        
        return result
    
    def arithmetic_softmax_approximation(self, scores):
        """
        Softmax approximation using only +, -, *, /
        
        Traditional softmax: softmax(x_i) = exp(x_i) / Σ(exp(x_j))
        Our approximation: Use polynomial to approximate exp()
        
        We'll use a simple rational approximation that preserves
        the "competition" behavior of softmax
        """
        # First, normalize scores to prevent extreme values
        # Find max value using only arithmetic comparisons
        max_score = scores[0]
        for score in scores:
            # Approximate max() using arithmetic: max(a,b) ≈ (a+b+|a-b|)/2
            # Since we can't use abs(), we'll use: max(a,b) ≈ (a+b+(a-b)²/(|a-b|+ε))/2
            diff = score - max_score
            diff_squared = diff * diff
            # Simple approximation: if diff > 0, score is larger
            adjusted_diff = diff_squared / (diff_squared + 1.0)
            if diff > 0:
                max_score = score
        
        # Subtract max for numerical stability (standard softmax trick)
        normalized_scores = []
        for score in scores:
            normalized_scores.append(score - max_score)
        
        # Approximate exp(x) using Taylor series with only +,-,*,/
        # exp(x) ≈ 1 + x + x²/2 + x³/6 (truncated series)
        exp_approximations = []
        for x in normalized_scores:
            x_squared = x * x
            x_cubed = x_squared * x
            
            # Taylor approximation: 1 + x + x²/2 + x³/6
            exp_approx = 1.0 + x + x_squared/2.0 + x_cubed/6.0
            
            # Ensure positive values (softmax outputs must be positive)
            if exp_approx < 0.01:
                exp_approx = 0.01
            
            exp_approximations.append(exp_approx)
        
        # Calculate sum of all approximated exponentials
        total_sum = 0.0
        for exp_val in exp_approximations:
            total_sum = total_sum + exp_val
        
        # Normalize to get probabilities (this is the softmax!)
        softmax_result = []
        for exp_val in exp_approximations:
            probability = exp_val / total_sum
            softmax_result.append(probability)
        
        return softmax_result
    
    def single_head_attention(self, X, W_q, W_k, W_v, head_name):
        """
        Compute attention for a single head using only +, -, *, /
        
        This is the CORE attention mechanism:
        1. Transform input into Queries, Keys, Values
        2. Compute attention scores (Q * K^T)
        3. Apply softmax to get attention weights  
        4. Weighted sum of Values
        
        This is what makes transformers "pay attention"!
        """
        print(f"\n🎯 Computing {head_name} attention...")
        
        # Step 1: Linear transformations to get Q, K, V
        # Q = X * W_q (What am I looking for?)
        Q = self.arithmetic_matrix_multiply(X, W_q)
        print(f"📋 Queries computed: {len(Q)}x{len(Q[0])} matrix")
        
        # K = X * W_k (What information do I have?)  
        K = self.arithmetic_matrix_multiply(X, W_k)
        print(f"🔑 Keys computed: {len(K)}x{len(K[0])} matrix")
        
        # V = X * W_v (What are the actual values?)
        V = self.arithmetic_matrix_multiply(X, W_v)
        print(f"💎 Values computed: {len(V)}x{len(V[0])} matrix")
        
        # Step 2: Compute attention scores = Q * K^T
        # First, transpose K manually
        K_transpose = []
        for j in range(len(K[0])):
            col = []
            for i in range(len(K)):
                col.append(K[i][j])
            K_transpose.append(col)
        
        # Now compute Q * K^T
        attention_scores = self.arithmetic_matrix_multiply(Q, K_transpose)
        print(f"📊 Attention scores computed: {len(attention_scores)}x{len(attention_scores[0])} matrix")
        
        # Step 3: Apply softmax to each row (each token's attention distribution)
        attention_weights = []
        for i in range(len(attention_scores)):
            row_scores = attention_scores[i]
            row_weights = self.arithmetic_softmax_approximation(row_scores)
            attention_weights.append(row_weights)
        
        print(f"⚖️ Attention weights normalized with softmax approximation")
        
        # Step 4: Weighted sum of values = attention_weights * V
        attended_output = self.arithmetic_matrix_multiply(attention_weights, V)
        print(f"🎯 Attended output computed: {len(attended_output)}x{len(attended_output[0])} matrix")
        
        return attended_output, attention_weights
    
    def multi_head_attention(self, X):
        """
        Multi-Head Attention: Run multiple attention heads in parallel
        and combine their outputs
        
        This is what makes transformers so powerful - they can attend
        to different types of relationships simultaneously!
        """
        print("\n🤖 MULTI-HEAD ATTENTION COMPUTATION")
        print("=" * 50)
        
        # Run attention head 1
        head1_output, head1_weights = self.single_head_attention(
            X, self.W_q1, self.W_k1, self.W_v1, "Head 1"
        )
        
        # Run attention head 2  
        head2_output, head2_weights = self.single_head_attention(
            X, self.W_q2, self.W_k2, self.W_v2, "Head 2"
        )
        
        # Concatenate outputs from both heads
        print(f"\n🔗 Concatenating outputs from {self.num_heads} attention heads...")
        concatenated = []
        for i in range(len(head1_output)):
            row = []
            # Add head1 values
            for j in range(len(head1_output[i])):
                row.append(head1_output[i][j])
            # Add head2 values
            for j in range(len(head2_output[i])):
                row.append(head2_output[i][j])
            concatenated.append(row)
        
        # Final linear transformation (combines information from all heads)
        final_output = self.arithmetic_matrix_multiply(concatenated, self.W_o)
        
        print(f"✅ Multi-head attention complete!")
        print(f"📤 Final output shape: {len(final_output)}x{len(final_output[0])}")
        
        return final_output, head1_weights, head2_weights
    
    def demonstrate_attention(self):
        """
        Demonstrate the full attention mechanism with example input
        """
        print("\n🚀 TRANSFORMER ATTENTION DEMONSTRATION")
        print("=" * 60)
        
        # Example input: 4 tokens, each with 4-dimensional embeddings
        # Think of this as 4 words in a sentence, each represented as a vector
        X = [
            [0.1, 0.2, 0.3, 0.4],  # Token 1 embedding
            [0.5, 0.6, 0.7, 0.8],  # Token 2 embedding  
            [0.9, 1.0, 0.1, 0.2],  # Token 3 embedding
            [0.3, 0.4, 0.5, 0.6]   # Token 4 embedding
        ]
        
        print("📝 Input sequence (4 tokens x 4 dimensions):")
        for i, token in enumerate(X):
            print(f"   Token {i+1}: {[round(x, 3) for x in token]}")
        
        # Run the full attention mechanism
        output, head1_attention, head2_attention = self.multi_head_attention(X)
        
        # Display results
        print("\n📊 ATTENTION ANALYSIS")
        print("=" * 40)
        
        print("\n🎯 Head 1 Attention Weights:")
        print("   (How much each token attends to others)")
        for i, weights in enumerate(head1_attention):
            print(f"   Token {i+1} attends to: {[round(w, 3) for w in weights]}")
        
        print("\n🎯 Head 2 Attention Weights:")
        for i, weights in enumerate(head2_attention):
            print(f"   Token {i+1} attends to: {[round(w, 3) for w in weights]}")
        
        print("\n📤 Final Output (after attention):")
        for i, token_output in enumerate(output):
            print(f"   Token {i+1}: {[round(x, 4) for x in token_output]}")
        
        # Analyze attention patterns
        print("\n🔍 ATTENTION PATTERN ANALYSIS")
        print("=" * 40)
        
        # Find which tokens attend most to each other
        for i in range(len(head1_attention)):
            max_attention = 0.0
            most_attended = 0
            for j in range(len(head1_attention[i])):
                if head1_attention[i][j] > max_attention:
                    max_attention = head1_attention[i][j]
                    most_attended = j
            
            print(f"🎯 Token {i+1} pays most attention to Token {most_attended+1} "
                  f"(weight: {round(max_attention, 3)})")
        
        return output

def main():
    """
    Demonstrate transformer attention using only basic arithmetic
    """
    print("🤖 TRANSFORMER ATTENTION - THE ALGORITHM BEHIND CHATGPT")
    print("🔥 Implemented using ONLY +, -, *, / operations!")
    print("📚 Educational demonstration of modern AI's core mechanism")
    
    # Create and run attention mechanism
    attention = ArithmeticTransformerAttention()
    output = attention.demonstrate_attention()
    
    print("\n" + "=" * 60)
    print("🎉 TRANSFORMER ATTENTION COMPLETE!")
    print("🎯 Key Insights:")
    print("   • Attention is just weighted averages of values")
    print("   • Weights come from query-key similarity scores")
    print("   • Multiple heads capture different relationship types")
    print("   • The entire process is matrix multiplication + softmax")
    print("   • NO MAGIC - just arithmetic operations repeated systematically!")
    
    print(f"\n🧠 This is the EXACT mechanism that powers:")
    print("   • ChatGPT and GPT-4")
    print("   • Google's BERT and T5")
    print("   • Modern language translation")
    print("   • Text summarization and generation")
    
    print(f"\n🔥 And we just implemented it with elementary arithmetic!")
    print("💡 The 'magic' of AI is beautiful mathematical simplicity!")

if __name__ == "__main__":
    main()

"""
🧠 DEEP DIVE: WHY ATTENTION WORKS

The attention mechanism solves a fundamental problem in AI:
"How can a model focus on relevant information while processing sequences?"

🔍 THE ATTENTION INTUITION:
- Queries: "What am I looking for?"
- Keys: "What information is available?"  
- Values: "What are the actual contents?"
- Attention: "How relevant is each piece of information?"

🎯 THE MATHEMATICAL BEAUTY:
Attention(Q,K,V) = softmax(QK^T/√d_k)V

This simple formula enables:
- Language understanding (which words relate to each other)
- Image processing (which pixels are important)
- Sequence modeling (temporal dependencies)
- Memory mechanisms (what to remember/forget)

🔥 THE POWER OF MULTIPLE HEADS:
Different heads can specialize in different relationships:
- Head 1: Syntactic relationships (grammar)
- Head 2: Semantic relationships (meaning)
- Head 3: Long-range dependencies
- Head N: Domain-specific patterns

🚀 REAL-WORLD IMPACT:
This mechanism revolutionized AI by enabling:
- Parallel processing (unlike RNNs)
- Long-range dependencies
- Interpretable attention patterns
- Transfer learning across domains

💡 EDUCATIONAL VALUE:
By implementing with basic arithmetic, we see that the "magic"
of modern AI reduces to:
1. Linear transformations (matrix multiplication)
2. Similarity computation (dot products)
3. Normalization (softmax)
4. Weighted averaging

No supernatural intelligence - just elegant mathematics!

🎯 CONCLUSION:
The algorithm that powers the most advanced AI systems
is fundamentally simple arithmetic operations arranged
in a clever pattern. Understanding this demystifies AI
and reveals the beautiful mathematical foundations
underneath the technological marvel.
"""

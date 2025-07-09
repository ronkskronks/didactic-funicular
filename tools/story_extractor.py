#!/usr/bin/env python3
"""
🔥 STORY EXTRACTOR - Transformador de LONG ASS STORIES em dados de treino
Converte as obras literárias em formato conversacional para o Monster Chatbot
"""

import os
import re
import random

def extract_story_content(file_path):
    """Extrai conteúdo limpo dos arquivos markdown"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove headers e metadata
    lines = content.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        # Pula linhas vazias, headers, e metadata
        if (line and 
            not line.startswith('#') and 
            not line.startswith('LIVRO') and
            not 'frases' in line.lower() and
            len(line) > 20):  # Apenas frases substanciais
            clean_lines.append(line)
    
    return clean_lines

def create_conversation_prompts():
    """Cria prompts para converter narrativa em conversa"""
    prompts = [
        "Me conte uma história",
        "Narre uma cena dramática",
        "Descreva um momento emocional",
        "Como essa história continua?",
        "Conte-me sobre as emoções da personagem",
        "O que aconteceu em seguida?",
        "Desenvolva essa narrativa",
        "Elabore sobre esse sentimento",
        "Continue a história",
        "Descreva a próxima cena",
        "Me fale sobre essa situação",
        "Narre esse momento",
        "Como a protagonista se sentia?",
        "O que se passou pela mente dela?",
        "Descreva o ambiente da cena",
        "Conte sobre esse episódio",
        "Elabore essa passagem",
        "Desenvolva esse momento",
        "Como essa cena se desenrolou?",
        "Fale sobre essa experiência emocional"
    ]
    return prompts

def group_sentences_into_chunks(sentences, chunk_size=3):
    """Agrupa frases em chunks para respostas mais longas"""
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i+chunk_size]
        # Junta as frases do chunk
        combined = ' '.join(chunk)
        chunks.append(combined)
    return chunks

def create_dialogue_dataset(story_files, output_file):
    """Cria dataset conversacional a partir das histórias"""
    print("📚 Extraindo dados das LONG ASS STORIES...")
    
    all_sentences = []
    prompts = create_conversation_prompts()
    
    # Extrai todas as frases de todos os livros
    for file_path in story_files:
        print(f"   📖 Processando: {os.path.basename(file_path)}")
        sentences = extract_story_content(file_path)
        all_sentences.extend(sentences)
        print(f"      ✓ {len(sentences)} frases extraídas")
    
    print(f"\n📊 Total de frases coletadas: {len(all_sentences)}")
    
    # Agrupa frases em chunks para respostas mais elaboradas
    story_chunks = group_sentences_into_chunks(all_sentences, chunk_size=3)
    print(f"📦 Criados {len(story_chunks)} chunks narrativos")
    
    # Cria pares pergunta-resposta
    dialogue_pairs = []
    
    # 1. Conversas sobre histórias
    for i, chunk in enumerate(story_chunks[:500]):  # Limita para não explodir
        prompt = random.choice(prompts)
        dialogue_pairs.append((prompt, chunk))
    
    # 2. Conversas sequenciais (continuação de narrativa)
    for i in range(min(200, len(story_chunks) - 1)):
        prompt = f"Continue essa história: {story_chunks[i][:100]}..."
        response = story_chunks[i + 1]
        dialogue_pairs.append((prompt, response))
    
    # 3. Análise de emoções (extrai emoções das frases)
    emotions = ['tristeza', 'alegria', 'raiva', 'medo', 'amor', 'esperança', 
              'ansiedade', 'tédio', 'luto', 'ambição', 'desejo', 'estagnação']
    
    for emotion in emotions:
        emotion_sentences = [s for s in all_sentences if emotion in s.lower()]
        if emotion_sentences:
            sample_sentences = random.sample(emotion_sentences, min(10, len(emotion_sentences)))
            for sentence in sample_sentences:
                prompt = f"Me conte sobre {emotion}"
                dialogue_pairs.append((prompt, sentence))
    
    # 4. Personagem (Lia)
    lia_sentences = [s for s in all_sentences if 'Lia' in s or 'ela' in s]
    for i in range(min(100, len(lia_sentences))):
        prompts_lia = [
            "Fale sobre a protagonista Lia",
            "O que aconteceu com Lia?",
            "Descreva Lia nessa situação",
            "Como Lia se sentia?",
            "Continue a história de Lia"
        ]
        prompt = random.choice(prompts_lia)
        dialogue_pairs.append((prompt, lia_sentences[i]))
    
    print(f"💬 Criados {len(dialogue_pairs)} pares de diálogo")
    
    # Salva no formato USER/BOT
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt, response in dialogue_pairs:
            f.write(f"USER: {prompt}\n")
            f.write(f"BOT: {response}\n")
            f.write("\n")
    
    print(f"✅ Dataset salvo em: {output_file}")
    return len(dialogue_pairs)

def enhance_existing_dataset(original_file, story_file, output_file):
    """Combina dataset original com dados das histórias"""
    print("🔗 Combinando datasets...")
    
    # Lê dataset original
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = f.read()
    
    # Lê dados das histórias
    with open(story_file, 'r', encoding='utf-8') as f:
        story_data = f.read()
    
    # Combina
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# MEGA DATASET - Original + LONG ASS STORIES\n\n")
        f.write("# === CONVERSAS ORIGINAIS ===\n")
        f.write(original_data)
        f.write("\n\n# === NARRATIVAS ÉPICAS ===\n")
        f.write(story_data)
    
    print(f"✅ Mega dataset criado: {output_file}")

def main():
    print("🔥🤖 STORY EXTRACTOR - ALIMENTANDO O MONSTER! 🤖🔥")
    print("=" * 55)
    
    # Localiza arquivos das histórias
    story_dir = "FRESHLY_MILKED_DATA"
    story_files = []
    
    for filename in os.listdir(story_dir):
        if filename.endswith('.md') and 'trilogia' in filename:
            story_files.append(os.path.join(story_dir, filename))
    
    print(f"📚 Encontrados {len(story_files)} livros da trilogia:")
    for file in story_files:
        print(f"   📖 {os.path.basename(file)}")
    
    # Cria dataset de histórias
    story_dataset = "data/story_conversations.txt"
    num_conversations = create_dialogue_dataset(story_files, story_dataset)
    
    # Combina com dataset original
    original_dataset = "data/chatbot_training.txt"
    mega_dataset = "data/mega_training_dataset.txt"
    enhance_existing_dataset(original_dataset, story_dataset, mega_dataset)
    
    print("\n🎉 EXTRAÇÃO CONCLUÍDA!")
    print(f"📊 Estatísticas:")
    print(f"   💬 {num_conversations} conversas sobre histórias")
    print(f"   📚 {len(story_files)} livros processados")
    print(f"   🚀 Mega dataset pronto para o Monster!")
    print("\n🔥 Agora é só treinar o Monster com esse BANQUETE!")

if __name__ == "__main__":
    main()
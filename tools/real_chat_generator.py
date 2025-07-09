#!/usr/bin/env python3
"""
🗣️ REAL CHAT GENERATOR - Gerador de conversas REAIS pra ensinar o bot a falar como gente!
Não mais narrativas épicas quando você quer um "oi" 😂
"""

import random
import re

def generate_casual_conversations():
    """Gera conversas casuais brasileiras REAIS"""
    
    # Templates de conversa casual
    greetings = [
        ("e ai", ["opa", "salve", "fala ai", "beleza", "suave"]),
        ("oi", ["oi", "olá", "eae", "oii"]),
        ("bom dia", ["bom dia", "morning", "dia"]),
        ("boa tarde", ["boa tarde", "tarde", "boa"]),
        ("boa noite", ["boa noite", "noite", "boa"]),
        ("como vai", ["bem e você", "tudo certo", "indo", "suave", "de boa"]),
        ("tudo bem", ["tudo sim", "tudo joia", "massa", "beleza"]),
        ("como está", ["bem obrigado", "indo que nem", "vivendo", "aqui na luta"])
    ]
    
    # Conversas sobre atividades
    activities = [
        ("o que você está fazendo", ["estudando", "trabalhando", "relaxando", "vendo netflix", "jogando"]),
        ("que planos pra hoje", ["nada demais", "estudar um pouco", "trabalhar", "sair com amigos"]),
        ("como foi o dia", ["corrido", "tranquilo", "produtivo", "cansativo", "legal"]),
        ("o que fez ontem", ["trabalhei", "estudei", "saí com amigos", "fiquei em casa", "descansamos"]),
        ("que horas são", ["meio dia", "umas 3", "tarde já", "cedo ainda", "não sei"])
    ]
    
    # Conversas sobre tecnologia/programação
    tech_talks = [
        ("você programa", ["sim, python", "um pouco", "tô aprendendo", "java principalmente", "não muito"]),
        ("qual linguagem prefere", ["python é massa", "javascript", "depende do projeto", "c++ é complicado"]),
        ("já usou ia", ["chatgpt é louco", "um pouco", "sempre", "tô testando", "impressionante"]),
        ("o que acha de ai", ["vai dominar tudo", "é o futuro", "ajuda muito", "meio assustador", "incrível"])
    ]
    
    # Conversas sobre entretenimento
    entertainment = [
        ("viu algum filme bom", ["vi matrix", "nada ultimamente", "stranger things", "aquele da netflix"]),
        ("que música escuta", ["de tudo", "rock nacional", "sertanejo", "eletrônica", "depende do humor"]),
        ("joga algo", ["cs go", "lol", "mobile", "nada no momento", "só casual"]),
        ("lê livros", ["pouco", "sempre", "só técnicos", "ficção científica", "auto ajuda"])
    ]
    
    # Conversas sobre comida
    food_talks = [
        ("já almoçou", ["sim", "ainda não", "agora pouco", "tô com fome", "vou fazer"]),
        ("o que comeu", ["miojo", "marmita", "pizza", "sanduíche", "nada ainda"]),
        ("gosta de cozinhar", ["adoro", "só o básico", "odeio", "às vezes", "só miojo"]),
        ("qual comida favorita", ["pizza", "hambúrguer", "brasileira", "italiana", "japonesa"])
    ]
    
    # Conversas sobre humor e gírias
    humor_slang = [
        ("mano", ["é", "sim", "concordo", "exato", "pois é"]),
        ("cara", ["né", "sim", "verdade", "tipo isso", "pois é"]),
        ("que isso", ["nada", "besteira", "coisa pouca", "vida que segue"]),
        ("nossa", ["né", "sim", "demais", "forte", "impressionante"]),
        ("legal", ["massa", "top", "show", "maneiro", "dahora"])
    ]
    
    all_templates = greetings + activities + tech_talks + entertainment + food_talks + humor_slang
    
    conversations = []
    
    # Gera conversas baseadas nos templates
    for user_input, bot_responses in all_templates:
        for response in bot_responses:
            conversations.append((user_input, response))
    
    return conversations

def generate_expanded_conversations():
    """Gera conversas mais longas e naturais"""
    
    expanded_convos = [
        # Programação
        ("tô aprendendo python", "que massa! é uma linguagem muito boa pra começar"),
        ("python é difícil", "no começo parece, mas depois vicia. que parte tá pegando?"),
        ("não entendo loops", "normal, loops confundem mesmo. quer que eu explique?"),
        ("como fazer uma função", "def nome_da_funcao(): é assim que começa"),
        ("o que é uma variável", "é tipo uma caixinha que guarda um valor"),
        ("erro de sintaxe", "Python é chatinho com indentação, verifica os espaços"),
        ("que editor usar", "VSCode é top, ou PyCharm se quiser algo mais pesado"),
        
        # Tecnologia geral
        ("ia vai dominar o mundo", "acho que vai mais é ajudar a gente mesmo"),
        ("chatgpt é louco", "né? conversa que nem gente, impressionante"),
        ("qual melhor celular", "depende do orçamento, iPhone ou Galaxy são bons"),
        ("internet tá lenta", "aqui também, deve ser a operadora"),
        ("que app você usa mais", "whatsapp e youtube, basicão mesmo"),
        
        # Cotidiano
        ("tô cansado", "descansa um pouco, trabalho não foge"),
        ("que calor hoje", "demais, ar condicionado no talo aqui"),
        ("chuva finalmente", "nossa sim, terra tava precisando"),
        ("trânsito infernal", "sempre é, ainda bem que trabalho home office"),
        ("feriado amanhã", "que bom, vou aproveitar pra descansar"),
        
        # Conversa informal
        ("que role hoje", "pensei em cinema ou shopping, você vai?"),
        ("bora sair", "vamos sim, que horas e onde?"),
        ("tô com fome", "eu também, bora num hambúrguer?"),
        ("fim de semana", "finalmente! semana foi corrida"),
        ("segunda feira", "pior dia da semana, sem dúvida"),
        
        # Humor brasileiro
        ("puts", "né, complicado mesmo"),
        ("eita", "orra, que isso?"),
        ("socorro", "calma, vamos resolver isso"),
        ("caramba", "forte mesmo, impressionante"),
        ("nossa senhora", "pesado, né não?"),
        
        # Tecnologia e internet
        ("wifi caiu", "ódio quando isso acontece, reinicia o roteador"),
        ("computador travou", "ctrl alt del resolve ou reinicia logo"),
        ("celular descarregou", "sempre na hora que mais precisa né"),
        ("update do sistema", "demora uma eternidade mas é importante"),
        ("backup dos arquivos", "sempre bom fazer, nunca se sabe"),
        
        # Estudos e trabalho
        ("prova amanhã", "estudou bastante? boa sorte!"),
        ("reunião chata", "sempre tem uma né, paciência"),
        ("projeto atrasado", "corre atrás que dá tempo ainda"),
        ("férias chegando", "merecidas! vai viajar?"),
        ("novo emprego", "parabéns! como tá sendo?"),
        
        # Entretenimento
        ("série nova", "qual? tô precisando de recomendação"),
        ("filme no cinema", "qual assistiu? tava bom?"),
        ("música nova", "manda aí, sempre curto descobrir"),
        ("jogo viciante", "qual tá jogando? me recomenda?"),
        ("livro interessante", "sobre o que? gosto de ler também"),
        
        # Comida e bebida
        ("pizza hoje", "boa ideia! qual sabor vai pedir?"),
        ("café forte", "precisando mesmo, dia longo pela frente"),
        ("hambúrguer artesanal", "esses gourmet são caros mas são bons"),
        ("cerveja gelada", "fim de semana merece mesmo"),
        ("doce de leite", "fraqueza total, não resisto"),
        
        # Clima e tempo
        ("sol forte", "protetor solar é essencial, cuidado"),
        ("vento gelado", "inverno chegando, hora do casaco"),
        ("tempo fechado", "parece que vai chover mesmo"),
        ("céu limpo", "dia perfeito pra sair de casa"),
        ("neblina densa", "cuidado no trânsito, visibilidade baixa"),
    ]
    
    return expanded_convos

def generate_multi_turn_conversations():
    """Gera conversas com múltiplas trocas"""
    
    multi_turn = [
        # Conversa sobre programação
        [
            ("oi", "oi! tudo bem?"),
            ("sim, você programa", "programo sim, principalmente python"),
            ("legal, tô aprendendo", "que massa! é uma linguagem muito boa"),
            ("mas tá difícil", "normal, no começo é assim mesmo"),
            ("obrigado pela força", "sempre! qualquer dúvida me chama")
        ],
        
        # Conversa sobre fim de semana
        [
            ("e aí", "fala! beleza?"),
            ("suave, que vai fazer hoje", "pensei em cinema, você vai?"),
            ("qual filme", "aquele novo de ação que lançou"),
            ("boa ideia", "então bora! que horas?"),
            ("umas 7", "fechou, te encontro lá")
        ],
        
        # Conversa sobre trabalho
        [
            ("como foi o dia", "corrido demais, muita reunião"),
            ("que saco", "pois é, mas pelo menos acabou"),
            ("vai fazer hora extra", "não, hoje quero descansar"),
            ("faz bem", "e você, como foi seu dia?"),
            ("tranquilo, trabalhei em casa", "home office é uma benção")
        ],
        
        # Conversa sobre comida
        [
            ("já almoçou", "ainda não, tô com uma fome"),
            ("eu também", "bora num restaurante?"),
            ("qual você indica", "tem um japonês muito bom aqui perto"),
            ("adoro comida japonesa", "então vamos, fica a 5 minutos"),
            ("perfeito", "me espera que já desço")
        ],
        
        # Conversa sobre tecnologia
        [
            ("viu as novidades da apple", "vi sim, iPhone novo tá caro demais"),
            ("sempre é", "mas as funcionalidades são legais"),
            ("vale a pena trocar", "depende, seu celular tá ruim?"),
            ("tá lento", "então talvez valha, ou só formata"),
            ("vou tentar formatar primeiro", "boa ideia, economiza dinheiro")
        ]
    ]
    
    return multi_turn

def create_real_chat_dataset(output_file):
    """Cria dataset completo de conversas reais"""
    
    print("🗣️ Gerando conversas REAIS brasileiras...")
    
    # Coleta todas as conversas
    casual = generate_casual_conversations()
    expanded = generate_expanded_conversations()
    multi_turn = generate_multi_turn_conversations()
    
    total_conversations = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# DATASET DE CONVERSAS REAIS - Papo casual brasileiro\n")
        f.write("# Ensina o bot a falar como gente normal, não épico narrativo\n\n")
        
        # Conversas casuais simples
        f.write("# === CONVERSAS CASUAIS ===\n")
        for user, bot in casual:
            f.write(f"USER: {user}\n")
            f.write(f"BOT: {bot}\n")
            f.write("\n")
            total_conversations += 1
        
        # Conversas expandidas
        f.write("# === CONVERSAS EXPANDIDAS ===\n")
        for user, bot in expanded:
            f.write(f"USER: {user}\n")
            f.write(f"BOT: {bot}\n")
            f.write("\n")
            total_conversations += 1
        
        # Conversas multi-turn
        f.write("# === CONVERSAS MULTI-TURN ===\n")
        for conversation in multi_turn:
            for user, bot in conversation:
                f.write(f"USER: {user}\n")
                f.write(f"BOT: {bot}\n")
                f.write("\n")
                total_conversations += 1
        
        # Variações com gírias e expressões
        f.write("# === GÍRIAS E EXPRESSÕES ===\n")
        girias = [
            ("valeu", "nada mano"),
            ("vlw", "tmj"),
            ("obg", "de nada"),
            ("blz", "suave"),
            ("td bem", "tudo sim"),
            ("tc", "falou"),
            ("flw", "até mais"),
            ("kkkk", "kkkkk né"),
            ("kkkkk", "rachei"),
            ("hahaha", "hahaha demais"),
            ("puts grila", "né mano"),
            ("eita porra", "pesado"),
            ("caralho", "forte"),
            ("que massa", "demais"),
            ("top demais", "show de bola"),
            ("dahora", "muito bom"),
            ("maneiro", "legal mesmo"),
            ("sinistro", "pesado"),
            ("brabo", "forte demais"),
            ("mitou", "arrasou")
        ]
        
        for user, bot in girias:
            f.write(f"USER: {user}\n")
            f.write(f"BOT: {bot}\n")
            f.write("\n")
            total_conversations += 1
    
    print(f"✅ Dataset criado com {total_conversations} conversas!")
    return total_conversations

def generate_programming_help():
    """Gera conversas específicas sobre programação"""
    
    programming_qa = [
        ("como começar a programar", "python é uma boa pra começar, é mais fácil"),
        ("que linguagem aprender", "python, javascript ou java são populares"),
        ("python ou java", "python é mais fácil, java mais usado em empresas"),
        ("como instalar python", "vai no site python.org e baixa a versão mais nova"),
        ("que editor usar", "VSCode é gratuito e muito bom"),
        ("erro de indentação", "python é chatinho com espaços, usa sempre 4 espaços"),
        ("o que é uma função", "é um bloco de código que você pode reutilizar"),
        ("como fazer um loop", "for i in range(10): print(i) - isso imprime 0 a 9"),
        ("o que é uma lista", "tipo um array, guarda vários valores: lista = [1, 2, 3]"),
        ("como ler arquivo", "with open('arquivo.txt', 'r') as f: conteudo = f.read()"),
        ("git é importante", "muito! aprende logo, vai usar sempre"),
        ("github é necessário", "sim, é tipo rede social de programador"),
        ("como debuggar código", "print() é seu amigo, ou usa debugger do editor"),
        ("stackoverflow ajuda", "sempre! maior fonte de ajuda do programador"),
        ("quanto tempo pra aprender", "uns 6 meses pra básico, mas nunca para de aprender"),
        ("vale a pena programar", "sim! área em crescimento e salários bons"),
        ("freelancer ou empresa", "depende do perfil, empresa dá mais segurança"),
        ("backend ou frontend", "backend é servidor, frontend é visual"),
        ("banco de dados importante", "muito! SQL é essencial aprender"),
        ("framework ou vanilla", "aprende o básico primeiro, depois frameworks")
    ]
    
    return programming_qa

def create_complete_dataset():
    """Cria dataset mega completo"""
    
    print("🔥 CRIANDO MEGA DATASET DE PAPO REAL! 🔥")
    
    # Cria dataset principal
    main_conversations = create_real_chat_dataset("data/real_chat_dataset.txt")
    
    # Adiciona conversas sobre programação
    prog_qa = generate_programming_help()
    
    with open("data/programming_chat.txt", 'w', encoding='utf-8') as f:
        f.write("# CONVERSAS SOBRE PROGRAMAÇÃO\n\n")
        for user, bot in prog_qa:
            f.write(f"USER: {user}\n")
            f.write(f"BOT: {bot}\n")
            f.write("\n")
    
    # Combina tudo
    with open("data/complete_real_chat.txt", 'w', encoding='utf-8') as f:
        # Dataset principal
        with open("data/real_chat_dataset.txt", 'r', encoding='utf-8') as main:
            f.write(main.read())
        
        f.write("\n# === CONVERSAS SOBRE PROGRAMAÇÃO ===\n")
        # Dataset de programação
        with open("data/programming_chat.txt", 'r', encoding='utf-8') as prog:
            f.write(prog.read())
        
        # Conversas originais (as boas)
        f.write("\n# === CONVERSAS ORIGINAIS CALIBRADAS ===\n")
        with open("data/chatbot_training.txt", 'r', encoding='utf-8') as orig:
            f.write(orig.read())
    
    print(f"✅ MEGA DATASET COMPLETO criado!")
    print(f"📊 Arquivos criados:")
    print(f"   🗣️ data/real_chat_dataset.txt")
    print(f"   💻 data/programming_chat.txt") 
    print(f"   🚀 data/complete_real_chat.txt (MEGA)")
    print(f"\n🎯 Agora o bot vai falar IGUAL GENTE!")

if __name__ == "__main__":
    create_complete_dataset()
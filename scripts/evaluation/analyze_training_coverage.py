import json
import os
import re
from collections import Counter
from supabase import create_client
from dotenv import load_dotenv

def analyze_coverage():
    print("--- Iniciando Analise de Cobertura de Treino ---")
    
    # Carrega .env
    load_dotenv()
    
    # 1. Carrega Girias do Supabase
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_SERVICE_KEY') # Usar service key para garantir acesso
    if not url or not key:
        print("Erro: SUPABASE_URL ou SUPABASE_KEY nao configurados.")
        return

    supabase = create_client(url, key)
    response = supabase.table('slang_dictionary').select('word').execute()
    db_slangs = [item['word'].lower() for item in response.data]
    print(f"Total de girias no dicionario (DB): {len(db_slangs)}")

    # 2. Carrega Datasets de Treino
    datasets = [
        "sentences_train.json",
        "advanced_sentences_train.json",
        "neutral_sentences_train.json"
    ]
    
    all_training_text = ""
    total_examples = 0
    
    for filename in datasets:
        path = os.path.join("data", filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                total_examples += len(data)
                # Extrai apenas o texto informal de cada exemplo
                for item in data:
                    all_training_text += " " + item.get('informal', '')
    
    print(f"Total de exemplos de treino carregados: {total_examples}")
    
    # 3. Analisa Cobertura
    missing_slangs = []
    low_coverage = []
    coverage_counts = Counter()
    
    # Regex para variaveis linguísticas comuns
    def check_slang_in_text(slang, text):
        # Procura por base + possíveis sufixos
        pattern = r'\b' + re.escape(slang) + r'(?:ing|ed|es|s|er)?\b'
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        return len(matches)

    print("\nVerificando cada giria no dataset...")
    for slang in db_slangs:
        count = check_slang_in_text(slang, all_training_text)
        coverage_counts[slang] = count
        
        if count == 0:
            missing_slangs.append(slang)
        elif count < 5:
            low_coverage.append((slang, count))
            
    # 4. Relatorio Final
    print("\n" + "="*40)
    print("RELATORIO DE COBERTURA")
    print("="*40)
    
    print(f"\n[!] Girias SEM NENHUM exemplo no treino ({len(missing_slangs)}):")
    for s in sorted(missing_slangs):
        print(f" - {s}")
        
    print(f"\n[?] Girias com COBERTURA BAIXA (menos de 5 exemplos):")
    for s, c in sorted(low_coverage, key=lambda x: x[1]):
        print(f" - {s}: {c} ocorrencias")
        
    print("\n[+] Top 10 girias mais frequentes no treino:")
    for s, c in coverage_counts.most_common(10):
        print(f" - {s}: {c} ocorrencias")

    # 5. Sugestao de Acão
    print("\n" + "="*40)
    print("PROXIMOS PASSOS SUGERIDOS")
    print("="*40)
    if missing_slangs:
        print(f"1. Gerar frases para as {len(missing_slangs)} girias faltantes.")
    print("2. Gerar variacoes (gerundio/passado) para as girias de baixa cobertura.")
    print("3. Criar um dataset sintético focado apenas nestes gaps.")

if __name__ == "__main__":
    analyze_coverage()

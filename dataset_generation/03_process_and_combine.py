"""
ESTRATEGIA 3: Processar e combinar todos os datasets
Junta HuggingFace + OpenAI + seu dicionario atual
"""

import os
import pandas as pd
import re
from typing import List, Tuple

RAW_DIR = "data/raw"
GENERATED_DIR = "data/generated"
OUTPUT_DIR = "data/final"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    """Limpa texto basico"""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove mencoes (@user)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove espacos extras
    text = ' '.join(text.split())
    
    return text.strip()


def process_urban_dictionary() -> pd.DataFrame:
    """
    Processa Urban Dictionary para:
    1. Deteccao de girias (todas sao girias)
    2. Normalizacao (word -> definition resumida)
    """
    
    path = f"{RAW_DIR}/urban_dictionary.csv"
    if not os.path.exists(path):
        print(f"Arquivo nao encontrado: {path}")
        return pd.DataFrame()
    
    print("Processando Urban Dictionary...")
    df = pd.read_csv(path)
    
    # Para DETECCAO: os exemplos sao girias (label=1)
    detection_data = []
    for _, row in df.iterrows():
        example = clean_text(str(row.get("example", "")))
        if 10 < len(example) < 200:
            detection_data.append({"text": example, "label": 1})
    
    # Para NORMALIZACAO: word -> primeira frase da definicao
    normalization_data = []
    for _, row in df.iterrows():
        word = str(row.get("word", "")).lower().strip()
        definition = str(row.get("definition", ""))
        
        # Pega primeira frase da definicao
        first_sentence = definition.split(".")[0].strip()
        if word and first_sentence and len(first_sentence) < 100:
            normalization_data.append({
                "slang_text": word,
                "standard_text": first_sentence
            })
    
    print(f"  Deteccao: {len(detection_data)} exemplos")
    print(f"  Normalizacao: {len(normalization_data)} pares")
    
    return pd.DataFrame(detection_data), pd.DataFrame(normalization_data)


def process_tweets() -> pd.DataFrame:
    """
    Processa tweets como exemplos de texto informal
    """
    
    path = f"{RAW_DIR}/tweets.csv"
    if not os.path.exists(path):
        print(f"Arquivo nao encontrado: {path}")
        return pd.DataFrame()
    
    print("Processando Tweets...")
    df = pd.read_csv(path)
    
    detection_data = []
    for _, row in df.iterrows():
        text = clean_text(str(row.get("text", "")))
        if 10 < len(text) < 200:
            # Tweets sao geralmente informais (label=1)
            # mas nao 100% giria, entao usamos como "provavelmente informal"
            detection_data.append({"text": text, "label": 1})
    
    print(f"  Deteccao: {len(detection_data)} exemplos")
    return pd.DataFrame(detection_data)


def process_formal_texts() -> pd.DataFrame:
    """
    Processa textos formais (Wikipedia)
    """
    
    path = f"{RAW_DIR}/formal_texts.csv"
    if not os.path.exists(path):
        print(f"Arquivo nao encontrado: {path}")
        return pd.DataFrame()
    
    print("Processando textos formais...")
    df = pd.read_csv(path)
    
    # Ja vem com label=0
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 10]
    
    print(f"  Deteccao: {len(df)} exemplos formais")
    return df


def process_openai_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Processa dados gerados pelo OpenAI
    """
    
    detection_df = pd.DataFrame()
    normalization_df = pd.DataFrame()
    dictionary_df = pd.DataFrame()
    
    # Deteccao
    path = f"{GENERATED_DIR}/slang_detection_openai.csv"
    if os.path.exists(path):
        print("Carregando deteccao OpenAI...")
        detection_df = pd.read_csv(path)
        print(f"  {len(detection_df)} exemplos")
    
    # Normalizacao
    path = f"{GENERATED_DIR}/slang_normalization_openai.csv"
    if os.path.exists(path):
        print("Carregando normalizacao OpenAI...")
        normalization_df = pd.read_csv(path)
        print(f"  {len(normalization_df)} pares")
    
    # Dicionario
    path = f"{GENERATED_DIR}/slang_dictionary_openai.csv"
    if os.path.exists(path):
        print("Carregando dicionario OpenAI...")
        dictionary_df = pd.read_csv(path)
        print(f"  {len(dictionary_df)} girias")
    
    return detection_df, normalization_df, dictionary_df


def create_detection_dataset():
    """
    Combina todos os datasets para DETECCAO de girias
    Garante balanceamento 50/50
    """
    
    print("\n" + "="*60)
    print("CRIANDO DATASET DE DETECCAO")
    print("="*60)
    
    all_data = []
    
    # 1. Urban Dictionary (girias)
    urban_det, _ = process_urban_dictionary()
    if not urban_det.empty:
        all_data.append(urban_det)
    
    # 2. Tweets (informal)
    tweets = process_tweets()
    if not tweets.empty:
        all_data.append(tweets)
    
    # 3. Formal (Wikipedia)
    formal = process_formal_texts()
    if not formal.empty:
        all_data.append(formal)
    
    # 4. OpenAI gerado
    openai_det, _, _ = process_openai_data()
    if not openai_det.empty:
        all_data.append(openai_det)
    
    if not all_data:
        print("ERRO: Nenhum dado encontrado!")
        return
    
    # Combina
    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=["text"])
    
    # Balanceia
    slang_df = df[df["label"] == 1]
    formal_df = df[df["label"] == 0]
    
    min_count = min(len(slang_df), len(formal_df))
    
    slang_sample = slang_df.sample(n=min_count, random_state=42)
    formal_sample = formal_df.sample(n=min_count, random_state=42)
    
    balanced_df = pd.concat([slang_sample, formal_sample])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Salva
    output_path = f"{OUTPUT_DIR}/slang_detection.csv"
    balanced_df.to_csv(output_path, index=False)
    
    print(f"\nDataset final: {len(balanced_df)} exemplos")
    print(f"  Girias (1): {len(balanced_df[balanced_df['label']==1])}")
    print(f"  Formal (0): {len(balanced_df[balanced_df['label']==0])}")
    print(f"Salvo em: {output_path}")


def create_normalization_dataset():
    """
    Combina todos os datasets para NORMALIZACAO
    """
    
    print("\n" + "="*60)
    print("CRIANDO DATASET DE NORMALIZACAO")
    print("="*60)
    
    all_data = []
    
    # 1. Urban Dictionary
    _, urban_norm = process_urban_dictionary()
    if not urban_norm.empty:
        all_data.append(urban_norm)
    
    # 2. OpenAI gerado
    _, openai_norm, _ = process_openai_data()
    if not openai_norm.empty:
        all_data.append(openai_norm)
    
    # 3. Pares manuais (seu dicionario atual)
    manual_pairs = create_manual_pairs()
    if not manual_pairs.empty:
        all_data.append(manual_pairs)
    
    if not all_data:
        print("ERRO: Nenhum dado encontrado!")
        return
    
    # Combina
    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=["slang_text"])
    
    # Filtra pares onde input != output (senao nao aprende nada)
    df = df[df["slang_text"].str.lower().str.strip() != df["standard_text"].str.lower().str.strip()]
    
    # Salva
    output_path = f"{OUTPUT_DIR}/slang_normalization.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset final: {len(df)} pares")
    print(f"Salvo em: {output_path}")


def create_manual_pairs() -> pd.DataFrame:
    """
    Cria pares manuais das girias mais comuns
    Isso GARANTE que o modelo aprenda as basicas
    """
    
    # Girias essenciais com variações de contexto
    pairs = [
        # Contracoes basicas
        ("gonna", "going to"),
        ("wanna", "want to"),
        ("gotta", "got to"),
        ("kinda", "kind of"),
        ("sorta", "sort of"),
        ("dunno", "don't know"),
        ("lemme", "let me"),
        ("gimme", "give me"),
        ("coulda", "could have"),
        ("shoulda", "should have"),
        ("woulda", "would have"),
        
        # Com contexto
        ("I'm gonna go", "I'm going to go"),
        ("I wanna eat", "I want to eat"),
        ("I gotta run", "I have to run"),
        ("it's kinda cold", "it's kind of cold"),
        ("I dunno man", "I don't know"),
        ("lemme see", "let me see"),
        ("gimme that", "give me that"),
        
        # Internet/texting
        ("u", "you"),
        ("r", "are"),
        ("ur", "your"),
        ("2", "to"),
        ("4", "for"),
        ("b4", "before"),
        ("cuz", "because"),
        ("bc", "because"),
        ("tho", "though"),
        ("thru", "through"),
        
        # Com contexto
        ("how r u", "how are you"),
        ("r u coming", "are you coming"),
        ("thats 4 u", "that's for you"),
        
        # Girias populares
        ("lit", "exciting"),
        ("fire", "excellent"),
        ("dope", "cool"),
        ("sick", "awesome"),
        ("lowkey", "somewhat"),
        ("highkey", "very"),
        ("salty", "upset"),
        ("shook", "shocked"),
        ("slay", "do something excellently"),
        ("bet", "okay, agreed"),
        ("cap", "lie"),
        ("no cap", "no lie, seriously"),
        ("bussin", "very good"),
        ("mid", "mediocre"),
        ("sus", "suspicious"),
        ("vibe", "atmosphere"),
        ("mood", "relatable feeling"),
        ("stan", "be a big fan of"),
        ("ghost", "ignore someone"),
        ("flex", "show off"),
        ("simp", "someone who does too much"),
        ("yeet", "throw"),
        ("bruh", "expression of disbelief"),
        ("fam", "close friends"),
        ("squad", "friend group"),
        
        # Com contexto
        ("that's lit", "that's exciting"),
        ("this food is fire", "this food is excellent"),
        ("she lowkey likes him", "she somewhat likes him"),
        ("I'm highkey tired", "I'm very tired"),
        ("no cap fr", "seriously, for real"),
        ("that movie was mid", "that movie was mediocre"),
        ("he acting sus", "he's acting suspicious"),
        ("she ghosted me", "she ignored me"),
        ("stop flexing", "stop showing off"),
        
        # Saudacoes
        ("wassup", "what's up"),
        ("sup", "what's up"),
        ("yo", "hey"),
        ("hey y'all", "hey everyone"),
        ("what's good", "how are you"),
        
        # Despedidas
        ("gotta bounce", "have to leave"),
        ("peace out", "goodbye"),
        ("catch ya later", "see you later"),
        ("imma head out", "I'm going to leave"),
        
        # Expressoes
        ("ngl", "not gonna lie"),
        ("tbh", "to be honest"),
        ("imo", "in my opinion"),
        ("idk", "I don't know"),
        ("idc", "I don't care"),
        ("brb", "be right back"),
        ("btw", "by the way"),
        ("omg", "oh my god"),
        ("lol", "laughing out loud"),
        ("lmao", "laughing my ass off"),
        ("rofl", "rolling on the floor laughing"),
        ("smh", "shaking my head"),
        ("fomo", "fear of missing out"),
        ("goat", "greatest of all time"),
        ("af", "as fuck"),
        ("fr", "for real"),
        ("rn", "right now"),
        ("asap", "as soon as possible"),
        
        # Com contexto
        ("ngl that was funny", "not gonna lie, that was funny"),
        ("tbh I don't care", "to be honest, I don't care"),
        ("idk what to do", "I don't know what to do"),
        ("brb gonna eat", "be right back, going to eat"),
        ("he's the goat fr", "he's the greatest of all time, for real"),
    ]
    
    df = pd.DataFrame(pairs, columns=["slang_text", "standard_text"])
    print(f"  Pares manuais: {len(df)}")
    
    return df


def print_statistics():
    """Mostra estatisticas dos datasets finais"""
    
    print("\n" + "="*60)
    print("ESTATISTICAS DOS DATASETS FINAIS")
    print("="*60)
    
    # Deteccao
    det_path = f"{OUTPUT_DIR}/slang_detection.csv"
    if os.path.exists(det_path):
        df = pd.read_csv(det_path)
        print(f"\nDETECCAO ({det_path}):")
        print(f"  Total: {len(df)}")
        print(f"  Girias (1): {len(df[df['label']==1])}")
        print(f"  Formal (0): {len(df[df['label']==0])}")
        print(f"  Tamanho medio: {df['text'].str.len().mean():.0f} chars")
    
    # Normalizacao
    norm_path = f"{OUTPUT_DIR}/slang_normalization.csv"
    if os.path.exists(norm_path):
        df = pd.read_csv(norm_path)
        print(f"\nNORMALIZACAO ({norm_path}):")
        print(f"  Total: {len(df)} pares")
        print(f"  Input medio: {df['slang_text'].str.len().mean():.0f} chars")
        print(f"  Output medio: {df['standard_text'].str.len().mean():.0f} chars")


if __name__ == "__main__":
    print("="*60)
    print("PROCESSAMENTO E COMBINACAO DE DATASETS")
    print("="*60)
    
    # 1. Cria dataset de deteccao
    create_detection_dataset()
    
    # 2. Cria dataset de normalizacao
    create_normalization_dataset()
    
    # 3. Mostra estatisticas
    print_statistics()
    
    print("\n" + "="*60)
    print("CONCLUIDO!")
    print("Datasets prontos em data/final/")
    print("="*60)

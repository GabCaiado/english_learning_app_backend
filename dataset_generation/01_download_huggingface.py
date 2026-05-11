"""
ESTRATEGIA 1: Baixar datasets prontos do HuggingFace
Custo: GRATIS
Qualidade: Media (precisa filtrar)
"""

from datasets import load_dataset
import pandas as pd
import os

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_urban_dictionary():
    """
    Urban Dictionary - Milhares de girias com definicoes
    Muito bom para: Slang Detector + exemplos de uso
    """
    print("Baixando Urban Dictionary...")
    
    ds = load_dataset("daspartho/urban_dictionary", split="train")
    
    # Estrutura: word, definition, example, author, thumbs_up, thumbs_down
    df = pd.DataFrame({
        "word": ds["word"],
        "definition": ds["definition"],
        "example": ds["example"],
        "thumbs_up": ds["thumbs_up"],
        "thumbs_down": ds["thumbs_down"]
    })
    
    # Filtra apenas os mais populares (thumbs_up > 100)
    df_quality = df[df["thumbs_up"] > 100].copy()
    
    print(f"Total: {len(df)} | Qualidade: {len(df_quality)}")
    
    df_quality.to_csv(f"{OUTPUT_DIR}/urban_dictionary.csv", index=False)
    print(f"Salvo em {OUTPUT_DIR}/urban_dictionary.csv")
    
    return df_quality


def download_twitter_sentiment():
    """
    Tweet Eval - Tweets reais com linguagem informal
    Muito bom para: Textos informais vs formais
    """
    print("\nBaixando TweetEval...")
    
    ds = load_dataset("tweet_eval", "sentiment", split="train")
    
    df = pd.DataFrame({
        "text": ds["text"],
        "label": ds["label"]  # 0=negative, 1=neutral, 2=positive
    })
    
    print(f"Total: {len(df)} tweets")
    df.to_csv(f"{OUTPUT_DIR}/tweets.csv", index=False)
    print(f"Salvo em {OUTPUT_DIR}/tweets.csv")
    
    return df


def download_informal_formal_pairs():
    """
    GYAFC - Girias/Informal para Formal (Yahoo Answers)
    PERFEITO para: Slang Normalizer (T5)
    """
    print("\nBaixando GYAFC (Informal -> Formal)...")
    
    try:
        # Este dataset tem pares informal -> formal
        ds = load_dataset("grammarly/coedit", split="train")
        
        # Filtra apenas tarefas de formalizacao
        df = pd.DataFrame({
            "src": ds["src"],
            "tgt": ds["tgt"],
            "task": ds["task"]
        })
        
        # Pega apenas "formality" tasks
        df_formal = df[df["task"].str.contains("formal", case=False, na=False)]
        
        print(f"Total: {len(df)} | Formalizacao: {len(df_formal)}")
        df_formal.to_csv(f"{OUTPUT_DIR}/informal_formal_pairs.csv", index=False)
        
        return df_formal
        
    except Exception as e:
        print(f"Erro ao baixar GYAFC: {e}")
        print("Tentando dataset alternativo...")
        
        # Alternativa: JFLEG (correcao gramatical)
        ds = load_dataset("jfleg", split="validation")
        df = pd.DataFrame({
            "sentence": ds["sentence"],
            "corrections": [c[0] if c else "" for c in ds["corrections"]]
        })
        df.to_csv(f"{OUTPUT_DIR}/grammar_corrections.csv", index=False)
        return df


def download_slang_corpus():
    """
    Slang específico - varios datasets menores
    """
    print("\nBaixando corpus de girias adicionais...")
    
    datasets_to_try = [
        ("lmsys/lmsys-chat-1m", "Conversas reais"),
        ("allenai/real-toxicity-prompts", "Linguagem informal/toxica"),
    ]
    
    for ds_name, description in datasets_to_try:
        try:
            print(f"  Tentando {ds_name}...")
            ds = load_dataset(ds_name, split="train[:10000]")  # Apenas 10k
            print(f"  {description}: {len(ds)} exemplos")
        except Exception as e:
            print(f"  Erro: {e}")


def create_formal_examples():
    """
    Cria exemplos de texto FORMAL (para balancear o detector)
    Usa Wikipedia/News
    """
    print("\nBaixando textos formais (Wikipedia)...")
    
    ds = load_dataset("wikipedia", "20220301.simple", split="train[:50000]")
    
    # Pega apenas primeiras frases (mais limpas)
    texts = []
    for item in ds:
        first_sentence = item["text"].split(".")[0] + "."
        if 20 < len(first_sentence) < 200:  # Filtra tamanho razoavel
            texts.append(first_sentence)
        if len(texts) >= 10000:
            break
    
    df = pd.DataFrame({"text": texts, "label": 0})  # label=0 = formal
    df.to_csv(f"{OUTPUT_DIR}/formal_texts.csv", index=False)
    print(f"Salvo {len(texts)} textos formais")
    
    return df


if __name__ == "__main__":
    print("="*60)
    print("DOWNLOAD DE DATASETS DO HUGGINGFACE")
    print("="*60)
    
    # 1. Urban Dictionary (girias)
    urban_df = download_urban_dictionary()
    
    # 2. Tweets (informal)
    tweets_df = download_twitter_sentiment()
    
    # 3. Pares informal -> formal (para T5)
    formal_pairs = download_informal_formal_pairs()
    
    # 4. Textos formais (para balancear detector)
    formal_texts = create_formal_examples()
    
    print("\n" + "="*60)
    print("DOWNLOADS CONCLUIDOS!")
    print("Arquivos em data/raw/")
    print("="*60)

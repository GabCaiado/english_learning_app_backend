from app.database import get_supabase
from app.ml.pipeline import get_pipeline
import os

# Adiciona o diretorio atual ao PYTHONPATH para encontrar o modulo 'app'
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("INICIANDO TESTE DO PIPELINE ML")
    print("Nota: Se for a primeira vez, isso pode demorar alguns minutos")
    print("para baixar os modelos (Helsinki-NLP e Sentence-Transformers).")
    print("=" * 60)

    try:
        # Conecta ao Supabase
        supabase = get_supabase()
        
        # Cria pipeline
        pipeline = get_pipeline(supabase)
        
        # Testa palavras
        test_words = [
            "wassup",      # giria
            "lit",         # giria
            "beautiful",   # normal
            "gonna",       # contracao
            "hello",       # normal
        ]
        
        print("\n" + "-" * 30)
        print("TESTE DE PALAVRAS")
        print("-" * 30)
        
        for word in test_words:
            result = pipeline.analyze_word(word)
            print(f"\nPalavra: {word}")
            print(f"  E giria: {result.is_slang}")
            print(f"  Normalizado: {result.normalized}")
            print(f"  Traducao: {result.translation_pt}")
            print(f"  Formalidade: {result.formality}")
        
        # Testa frases
        test_sentences = [
            "Wassup bro, that party was lit!",
            "I'm gonna go to the store.",
            "The weather is beautiful today.",
        ]
        
        print("\n" + "-" * 30)
        print("TESTE DE FRASES")
        print("-" * 30)
        
        for sentence in test_sentences:
            result = pipeline.translate_sentence(sentence)
            print(f"\nOriginal: {sentence}")
            print(f"Girias encontradas: {result['slangs_detected']}")
            print(f"Normalizado: {result['normalized_english']}")
            print(f"Traducao: {result['translation_pt']}")

        print("\n" + "=" * 60)
        print("TESTE CONCLUIDO COM SUCESSO!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERRO] Ocorreu um problema no teste: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

"""
Pipeline completo de traducao.
Combina: Dicionario + Tradutor + Embeddings
"""

from dataclasses import dataclass
from typing import Optional

from app.ml.translator import get_translator
from app.ml.embeddings import get_embedding_generator
from app.ml.slang_dictionary import get_slang_dictionary, SlangInfo


@dataclass
class WordAnalysis:
    """Resultado da analise de uma palavra"""
    original: str
    is_slang: bool
    normalized: str
    translation_pt: str
    meaning_en: Optional[str]
    meaning_pt: Optional[str]
    formality: str
    category: str
    examples: list[str]
    embedding: list[float]
    similar_words: list[str]


class TranslationPipeline:
    """
    Pipeline completo de processamento de palavras.
    
    Fluxo:
    1. Verifica se é giria no dicionario
    2. Normaliza (se for giria)
    3. Traduz para portugues
    4. Gera embedding
    """
    
    def __init__(self, supabase_client):
        self.translator = get_translator()
        self.embeddings = get_embedding_generator()
        self.dictionary = get_slang_dictionary()
        
        # Carrega dicionario do banco de dados
        self.dictionary.load_from_supabase(supabase_client)
    
    def analyze_word(self, word: str) -> WordAnalysis:
        """
        Analisa uma palavra completamente.
        
        Args:
            word: Palavra em ingles
        Returns:
            WordAnalysis com todas as informacoes
        """
        word_lower = word.lower().strip()
        
        # 1- Verifica no dicionario de girias
        slang_info = self.dictionary.lookup(word_lower)
        
        if slang_info:
            # É uma giria conhecida
            normalized = slang_info.normalized
            is_slang = True
            meaning_en = slang_info.meaning_en
            meaning_pt = slang_info.meaning_pt
            formality = slang_info.formality
            category = slang_info.category
            examples = slang_info.examples
            
            # Traduz a forma normalizada
            translation_pt = meaning_pt if meaning_pt else self.translator.translate(normalized)
        else:
            # Palavra normal
            normalized = word_lower
            is_slang = False
            meaning_en = None
            meaning_pt = None
            formality = "neutral"
            category = ""
            examples = []
            
            # Traduz direto
            translation_pt = self.translator.translate(word_lower)
        
        # 2- Gera embedding (sempre)
        embedding = self.embeddings.generate(word_lower)
        
        # 3- Busca palavras similares (TODO: implementar com pgvector)
        similar_words = []
        
        return WordAnalysis(
            original=word,
            is_slang=is_slang,
            normalized=normalized,
            translation_pt=translation_pt,
            meaning_en=meaning_en,
            meaning_pt=meaning_pt,
            formality=formality,
            category=category,
            examples=examples,
            embedding=embedding,
            similar_words=similar_words
        )
    
    def translate_sentence(self, sentence: str) -> dict:
        """
        Traduz uma frase completa.
        Detecta girias, normaliza, e traduz.
        """
        words = sentence.split()
        slangs_found = []
        normalized_words = []
        
        for word in words:
            # Remove pontuacao para lookup
            clean_word = word.strip(".,!?;:'\"")
            
            if self.dictionary.is_slang(clean_word):
                slang_info = self.dictionary.lookup(clean_word)
                normalized_text = slang_info.normalized if slang_info.normalized else clean_word
                slangs_found.append({
                    "original": clean_word,
                    "normalized": normalized_text,
                    "meaning_pt": slang_info.meaning_pt
                })
                # Substitui pela forma normalizada
                normalized_word = word.replace(clean_word, normalized_text)
                normalized_words.append(normalized_word)
            else:
                normalized_words.append(word)
        
        # Junta e traduz
        normalized_sentence = " ".join(normalized_words)
        translation = self.translator.translate(normalized_sentence)
        
        return {
            "original": sentence,
            "slangs_detected": slangs_found,
            "normalized_english": normalized_sentence,
            "translation_pt": translation
        }


# Pipeline global
_pipeline: Optional[TranslationPipeline] = None


def get_pipeline(supabase_client) -> TranslationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TranslationPipeline(supabase_client)
    return _pipeline
"""
NOVA ARQUITETURA: IA-FIRST PIPELINE
====================================
A IA processa TUDO primeiro, banco e apenas complemento.

Fluxo:
1. IA analisa a frase inteira (contexto)
2. IA classifica cada palavra (slang/formal)
3. IA + Banco buscam significados
4. IA traduz com contexto
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer
)


@dataclass
class WordAnalysis:
    """Analise de uma palavra individual"""
    word: str
    is_slang: bool
    slang_confidence: float
    normalized_form: Optional[str]
    translation_pt: Optional[str]
    source: str  # "ai_model", "database", "ai_fallback"


@dataclass
class SentenceAnalysis:
    """Analise de uma frase completa"""
    original: str
    formality_score: float  # 0 = muito informal, 1 = muito formal
    words_analysis: list[WordAnalysis]
    normalized_sentence: str
    translation_pt: str
    embedding: list[float]


class AIFirstPipeline:
    """
    Pipeline onde a IA e protagonista.
    Banco de dados e usado apenas para complementar.
    """
    
    def __init__(self, supabase_client=None):
        self.supabase = supabase_client
        self._load_models()
    
    def _load_models(self):
        """Carrega todos os modelos de IA"""
        print("[AI Pipeline] Carregando modelos...")
        
        # 1. Modelo de embeddings (analise semantica)
        print("  - Carregando Sentence Transformers...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Classificador de formalidade (frase inteira)
        print("  - Carregando classificador de formalidade...")
        self.formality_classifier = pipeline(
            "text-classification",
            model="s-nlp/roberta-base-formality-ranker",
            top_k=None
        )
        
        # 3. Detector de girias (palavra por palavra)
        print("  - Carregando detector de girias...")
        try:
            # Tenta carregar modelo fine-tuned
            self.slang_detector = pipeline(
                "text-classification",
                model="./models/slang_detector",
                top_k=None
            )
        except:
            # Fallback: usa modelo de sentimento como proxy
            self.slang_detector = None
            print("    [AVISO] Modelo de girias nao encontrado, usando heuristicas")
        
        # 4. Normalizador de girias (T5)
        print("  - Carregando normalizador T5...")
        try:
            self.normalizer_tokenizer = AutoTokenizer.from_pretrained("./models/slang_normalizer")
            self.normalizer_model = T5ForConditionalGeneration.from_pretrained("./models/slang_normalizer")
        except:
            # Fallback: T5-small base
            self.normalizer_tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.normalizer_model = T5ForConditionalGeneration.from_pretrained("t5-small")
            print("    [AVISO] Usando T5-small base")
        
        # 5. Tradutor EN -> PT
        print("  - Carregando tradutor Helsinki-NLP...")
        self.translator_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-pt")
        self.translator_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-pt")
        
        print("[AI Pipeline] Todos os modelos carregados!")
    
    # =========================================================================
    # CAMADA 1: ANALISE DE CONTEXTO (FRASE INTEIRA)
    # =========================================================================
    
    def analyze_formality(self, sentence: str) -> float:
        """
        Analisa o nivel de formalidade da frase inteira.
        Retorna: 0.0 (muito informal) a 1.0 (muito formal)
        """
        try:
            result = self.formality_classifier(sentence)
            # Extrai score de formalidade
            for item in result[0]:
                if item['label'] == 'formal':
                    return item['score']
            return 0.5
        except:
            # Fallback: heuristica simples
            informal_markers = ['lol', 'lmao', 'omg', 'wtf', 'gonna', 'wanna', 'u ', ' r ', 'bro', 'dude']
            text_lower = sentence.lower()
            informal_count = sum(1 for marker in informal_markers if marker in text_lower)
            return max(0.0, 1.0 - (informal_count * 0.15))
    
    def generate_embedding(self, text: str) -> list[float]:
        """Gera embedding semantico do texto"""
        embedding = self.embedder.encode(text)
        return embedding.tolist()
    
    # =========================================================================
    # CAMADA 2: CLASSIFICACAO PALAVRA POR PALAVRA
    # =========================================================================
    
    def detect_slang_word(self, word: str, sentence_context: str = "") -> tuple[bool, float]:
        """
        Detecta se uma palavra e giria.
        Usa contexto da frase para melhor precisao.
        
        Retorna: (is_slang, confidence)
        """
        word_lower = word.lower().strip()
        
        # Lista de girias conhecidas (heuristica rapida)
        known_slangs = {
            'lit', 'fire', 'dope', 'sick', 'cool', 'awesome', 'bro', 'dude',
            'homie', 'fam', 'gonna', 'wanna', 'gotta', 'kinda', 'sorta',
            'wassup', 'sup', 'yo', 'hey', 'yep', 'nope', 'yeah', 'nah',
            'lol', 'lmao', 'omg', 'wtf', 'btw', 'tbh', 'idk', 'imo',
            'chill', 'vibe', 'mood', 'flex', 'slay', 'lowkey', 'highkey',
            'goat', 'goated', 'based', 'cringe', 'sus', 'cap', 'nocap',
            'bet', 'facts', 'deadass', 'periodt', 'snatched', 'tea',
            'simp', 'stan', 'ghosting', 'salty', 'shook', 'woke',
            'bougie', 'extra', 'basic', 'savage', 'legit', 'hype',
            'clout', 'drip', 'bussin', 'slaps', 'hits', 'banger',
            'mid', 'rent-free', 'main-character', 'understood-the-assignment'
        }
        
        # Contracoes informais
        contractions = {
            'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'lemme', 'gimme',
            'dunno', 'aint', "ain't", 'ima', "i'ma", 'tryna', 'finna',
            'shoulda', 'coulda', 'woulda', 'musta', 'oughta'
        }
        
        # Verifica lista conhecida primeiro
        if word_lower in known_slangs or word_lower in contractions:
            return (True, 0.95)
        
        # Usa modelo se disponivel
        if self.slang_detector:
            try:
                # Usa contexto para melhor classificacao
                text_to_classify = f"{sentence_context} [SEP] {word}" if sentence_context else word
                result = self.slang_detector(text_to_classify)
                
                for item in result[0]:
                    if item['label'].upper() in ['SLANG', 'INFORMAL', '1', 'LABEL_1']:
                        if item['score'] > 0.6:
                            return (True, item['score'])
                
                return (False, 1.0 - result[0][0]['score'])
            except:
                pass
        
        # Heuristicas de fallback
        # Palavras muito curtas com letras repetidas: "yooo", "brooo"
        if len(word_lower) > 3 and word_lower[-1] == word_lower[-2]:
            return (True, 0.7)
        
        # Palavras todas em maiusculo (exceto acronimos comuns)
        if word.isupper() and len(word) > 1 and word not in ['I', 'OK', 'US', 'UK']:
            return (True, 0.6)
        
        return (False, 0.8)
    
    # =========================================================================
    # CAMADA 3: NORMALIZACAO E BUSCA DE SIGNIFICADOS
    # =========================================================================
    
    def normalize_with_ai(self, word: str) -> Optional[str]:
        """
        Usa T5 para normalizar giria -> ingles padrao.
        Esta e a IA trabalhando, nao o banco.
        """
        try:
            prompt = f"normalize slang to formal English: {word}"
            inputs = self.normalizer_tokenizer(prompt, return_tensors="pt", max_length=64, truncation=True)
            outputs = self.normalizer_model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
            normalized = self.normalizer_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Se retornou igual ou vazio, nao conseguiu
            if normalized.lower().strip() == word.lower().strip() or not normalized:
                return None
            
            return normalized
        except:
            return None
    
    def search_in_database(self, word: str) -> Optional[dict]:
        """
        Busca no banco de dados (complemento, nao protagonista).
        """
        if not self.supabase:
            return None
        
        try:
            result = self.supabase.table("slang_dictionary")\
                .select("*")\
                .ilike("word", word)\
                .limit(1)\
                .execute()
            
            if result.data:
                return result.data[0]
        except:
            pass
        
        return None
    
    def search_similar_by_embedding(self, embedding: list[float], limit: int = 5) -> list[dict]:
        """
        Busca palavras semanticamente similares usando embeddings.
        Esta e IA + Banco trabalhando juntos.
        """
        if not self.supabase:
            return []
        
        try:
            # Usa funcao RPC do Supabase para busca vetorial
            result = self.supabase.rpc(
                "search_similar_words",
                {
                    "query_embedding": embedding,
                    "match_count": limit,
                    "match_threshold": 0.5
                }
            ).execute()
            
            return result.data if result.data else []
        except:
            return []
    
    def get_word_meaning(self, word: str, sentence_context: str = "") -> WordAnalysis:
        """
        Obtem significado de uma palavra.
        
        Ordem de prioridade:
        1. IA classifica se e giria
        2. IA tenta normalizar
        3. Banco complementa com dados extras
        4. IA faz busca semantica se nao encontrou
        """
        word_lower = word.lower().strip()
        
        # PASSO 1: IA classifica
        is_slang, confidence = self.detect_slang_word(word, sentence_context)
        
        # Se nao e giria, retorna direto
        if not is_slang:
            translation = self.translate_text(word)
            return WordAnalysis(
                word=word,
                is_slang=False,
                slang_confidence=1.0 - confidence,
                normalized_form=word,
                translation_pt=translation,
                source="ai_model"
            )
        
        # PASSO 2: IA tenta normalizar
        normalized = self.normalize_with_ai(word)
        
        if normalized:
            translation = self.translate_text(normalized)
            return WordAnalysis(
                word=word,
                is_slang=True,
                slang_confidence=confidence,
                normalized_form=normalized,
                translation_pt=translation,
                source="ai_model"
            )
        
        # PASSO 3: Banco complementa
        db_result = self.search_in_database(word)
        
        if db_result:
            return WordAnalysis(
                word=word,
                is_slang=True,
                slang_confidence=confidence,
                normalized_form=db_result.get('normalized_form', word),
                translation_pt=db_result.get('translation_pt'),
                source="database"
            )
        
        # PASSO 4: Busca semantica (IA + Banco)
        word_embedding = self.generate_embedding(word)
        similar_words = self.search_similar_by_embedding(word_embedding, limit=3)
        
        if similar_words:
            # Usa o mais similar como referencia
            best_match = similar_words[0]
            return WordAnalysis(
                word=word,
                is_slang=True,
                slang_confidence=confidence,
                normalized_form=best_match.get('normalized_form'),
                translation_pt=best_match.get('translation_pt'),
                source="ai_semantic_search"
            )
        
        # FALLBACK: Traduz direto
        translation = self.translate_text(word)
        return WordAnalysis(
            word=word,
            is_slang=True,
            slang_confidence=confidence,
            normalized_form=None,
            translation_pt=translation,
            source="ai_fallback"
        )
    
    # =========================================================================
    # CAMADA 4: TRADUCAO CONTEXTUAL
    # =========================================================================
    
    def translate_text(self, text: str) -> str:
        """Traduz texto EN -> PT"""
        try:
            inputs = self.translator_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.translator_model.generate(**inputs)
            return self.translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        except:
            return text
    
    # =========================================================================
    # PIPELINE PRINCIPAL
    # =========================================================================
    
    def analyze_sentence(self, sentence: str) -> SentenceAnalysis:
        """
        Analisa uma frase completa.
        A IA e protagonista em todo o processo.
        """
        # 1. Gera embedding da frase (contexto semantico)
        sentence_embedding = self.generate_embedding(sentence)
        
        # 2. Analisa formalidade
        formality_score = self.analyze_formality(sentence)
        
        # 3. Analisa cada palavra
        words = sentence.split()
        words_analysis = []
        normalized_words = []
        
        for word in words:
            # Remove pontuacao para analise
            clean_word = ''.join(c for c in word if c.isalnum())
            if not clean_word:
                normalized_words.append(word)
                continue
            
            # IA analisa a palavra com contexto da frase
            analysis = self.get_word_meaning(clean_word, sentence)
            words_analysis.append(analysis)
            
            # Usa forma normalizada se disponivel
            if analysis.normalized_form and analysis.is_slang:
                # Preserva pontuacao original
                punctuation = ''.join(c for c in word if not c.isalnum())
                normalized_words.append(analysis.normalized_form + punctuation)
            else:
                normalized_words.append(word)
        
        # 4. Monta frase normalizada
        normalized_sentence = ' '.join(normalized_words)
        
        # 5. Traduz frase completa (contexto!)
        translation = self.translate_text(normalized_sentence)
        
        return SentenceAnalysis(
            original=sentence,
            formality_score=formality_score,
            words_analysis=words_analysis,
            normalized_sentence=normalized_sentence,
            translation_pt=translation,
            embedding=sentence_embedding
        )
    
    def analyze_word(self, word: str) -> WordAnalysis:
        """
        Analisa uma palavra individual.
        """
        return self.get_word_meaning(word)


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    # Inicializa pipeline (sem Supabase para teste)
    pipeline = AIFirstPipeline(supabase_client=None)
    
    # Testa frases
    test_sentences = [
        "That party was lit bro",
        "I'm gonna go to the store",
        "The meeting was very productive",
        "No cap, this food is bussin",
        "She's lowkey the goat at this",
    ]
    
    print("\n" + "="*60)
    print("TESTANDO AI-FIRST PIPELINE")
    print("="*60)
    
    for sentence in test_sentences:
        print(f"\n[INPUT] {sentence}")
        result = pipeline.analyze_sentence(sentence)
        print(f"  Formalidade: {result.formality_score:.2f}")
        print(f"  Normalizado: {result.normalized_sentence}")
        print(f"  Traducao: {result.translation_pt}")
        
        slangs = [w for w in result.words_analysis if w.is_slang]
        if slangs:
            print(f"  Girias detectadas:")
            for s in slangs:
                print(f"    - {s.word} -> {s.normalized_form} ({s.source})")

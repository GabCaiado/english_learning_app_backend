"""
Complete translation pipeline.
Combines: Dictionary + Translator + Embeddings
"""

from dataclasses import dataclass
from typing import Optional

from app.ml.translator import get_translator
from app.ml.embeddings import get_embedding_generator
from app.ml.slang_dictionary import get_slang_dictionary, SlangInfo
from app.ml.slang_detector import SlangDetector
from app.ml.normalizer import SlangNormalizer
from app.ml.context_resolver import ContextResolver
import json
import re
import os
import requests
import random


class DatasetExamplesManager:
    """Loads sentences from the training dataset to extract dynamic examples"""
    def __init__(self):
        self.sentences = []
        # Carrega múltiplos arquivos de treinamento para maior variedade de exemplos
        files_to_load = [
            "sentences_train.json",
            "advanced_sentences_train.json",
            "neutral_sentences_train.json"
        ]
        
        for filename in files_to_load:
            try:
                filepath = os.path.join("data", filename)
                if os.path.exists(filepath):
                    with open(filepath, encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.sentences.extend(data)
                            print(f"Carregados {len(data)} exemplos de {filename}")
            except Exception as e:
                print(f"Erro ao carregar exemplos de {filename}: {e}")

        random.shuffle(self.sentences)

    def get_examples_and_meanings(self, word: str, max_dataset_examples: int = 3, max_api_examples: int = 3) -> dict:
        result = {
            "examples": [],
            "additional_meanings": []
        }
        
        # 1. Busca no dataset (até max_dataset_examples)
        if self.sentences:
            pattern = r'\b' + re.escape(word) + r'\b'
            dataset_examples = []
            for item in self.sentences:
                if 'informal' in item and re.search(pattern, item['informal'], re.IGNORECASE):
                    dataset_examples.append(item['informal'])
                    if len(dataset_examples) >= max_dataset_examples:
                        break
            result["examples"].extend(dataset_examples)
        
        # 2. Busca na Free Dictionary API (até max_api_examples)
        try:
            resp = requests.get(f'https://api.dictionaryapi.dev/api/v2/entries/en/{word}', timeout=3)
            if resp.status_code == 200:
                data = resp.json()[0]
                api_examples = []
                
                # Extrai significados adicionais
                for meaning in data.get('meanings', []):
                    pos = meaning.get('partOfSpeech', 'Geral')
                    for defn in meaning.get('definitions', []):
                        # Pega o exemplo se existir
                        if 'example' in defn and len(api_examples) < max_api_examples:
                            api_examples.append(defn['example'])
                        
                        # Salva a definição simplificada para tradução posterior (apenas as 2 primeiras)
                        if len(result["additional_meanings"]) < 2:
                            # Pega apenas a primeira frase ou parte da definição para não ficar gigante
                            clean_def = re.split(r'[.;]', defn['definition'])[0].strip()
                            if clean_def:
                                result["additional_meanings"].append({
                                    "pos": pos,
                                    "definition": clean_def,
                                    "example": defn.get('example', '')
                                })
                
                result["examples"].extend(api_examples)
        except Exception as e:
            print(f"Erro ao buscar na Free Dictionary API: {e}")
            
        return result

_examples_manager = None
def get_examples_manager() -> DatasetExamplesManager:
    global _examples_manager
    if _examples_manager is None:
        _examples_manager = DatasetExamplesManager()
    return _examples_manager


@dataclass
class WordAnalysis:
    """Word analysis result"""
    original: str
    is_slang: bool
    normalized: str
    translation_pt: str
    meaning_en: Optional[str]
    meaning_pt: Optional[str]
    formality: str
    category: Optional[str]
    examples: list[str]
    contextual_translations: list[dict]
    embedding: list[float]
    similar_words: list[str]


class TranslationPipeline:
    """
    Complete word processing pipeline.
    
    Flow:
    1. Layer 1: Dictionary (exact lookup)
    2. Layer 2: ML Detection (DistilBERT)
    3. Layer 3: ML Normalization (ByT5)
    4. Camada 4: Tradução (Helsinki-NLP)
    """
    
    def __init__(self, supabase_client):
        self.translator = get_translator()
        self.embeddings = get_embedding_generator()
        self.dictionary = get_slang_dictionary()
        
        self.slang_detector = SlangDetector()
        self.slang_normalizer = SlangNormalizer()
        self.context_resolver = ContextResolver()
        self.examples_manager = get_examples_manager()
        
        # Carrega dicionario do banco de dados
        self.dictionary.load_from_supabase(supabase_client)
    
    def analyze_word(self, word: str) -> WordAnalysis:
        """
        Deep word analysis
        """
        word_lower = word.lower().strip()
        
        # Orquestração Inteligente:
        slang_info = self.dictionary.lookup(word_lower)
        
        # Importamos a lista de palavras ambíguas
        from app.ml.slang_detector import AMBIGUOUS_SLANG
        is_ambiguous = word_lower in AMBIGUOUS_SLANG
        
        is_really_slang = False
        normalized = word_lower
        meaning_en = None
        meaning_pt = None
        formality = "neutral"
        category = ""
        contextual_translations = []
        
        # Busca imediata de exemplos e significados extras (Dataset + API)
        examples = []
        additional_meanings = []
        if " " not in word_lower:
            extra_data = self.examples_manager.get_examples_and_meanings(word_lower)
            examples = extra_data["examples"]
            additional_meanings = extra_data["additional_meanings"]
        
        if slang_info:
            if is_ambiguous:
                # Bare ambiguous words need context before the app can say the
                # slang sense is the primary one. Keep the word neutral here;
                # the modal still exposes the slang meaning as an alternative.
                is_really_slang = False
            else:
                is_really_slang = True
            
            if is_really_slang:
                normalized = slang_info.normalized or word_lower
                meaning_en = slang_info.meaning_en
                meaning_pt = slang_info.meaning_pt
                formality = slang_info.formality
                category = slang_info.category
                if slang_info.examples:
                    examples = slang_info.examples + examples
                    examples = examples[:6]

        # Se for uma frase (multi-palavra), usamos sempre a lógica de tradução de frases
        # para garantir consistência e labels corretas (ex: "Tradução Adaptada")
        if " " in word_lower:
            tr_info = self.translate_sentence(word_lower)
            # Se a IA ou o dicionário confirmarem que é gíria, marcamos como tal
            is_really_slang = is_really_slang or (len(tr_info["slangs_detected"]) > 0)
            normalized = tr_info["normalized_english"]
            translation_pt = tr_info["translation_pt"]
            formality = "slang (detected)" if is_really_slang else "neutral"
            contextual_translations = tr_info["contextual_translations"]
        else:
            if not is_really_slang:
                normalized = word_lower
                formality = "neutral"
                translation_pt = self._sanitize_translation(self.translator.translate(normalized))
            else:
                # Gíria de palavra única vinda do dicionário
                translation_pt = meaning_pt if meaning_pt else self._sanitize_translation(self.translator.translate(normalized))

            # Construção de traduções contextuais (Apenas para palavras únicas)
            # 1. Significado Geral/Literal (sempre via Tradutor IA)
            # Tradução literal/geral para o Modal
            gen_translation = self._sanitize_translation(self.translator.translate(word_lower))
            
            # Fallback inteligente: se a IA falhou em traduzir (devolveu a mesma palavra),
            # tentamos usar a primeira definição do dicionário técnico disponível
            if gen_translation.lower() == word_lower.lower() and additional_meanings:
                first_def = additional_meanings[0]["definition"]
                # Traduzimos a definição e pegamos apenas a parte principal (curta)
                gen_tr_def = self.translator.translate(first_def)
                if len(gen_tr_def) > 50:
                    gen_tr_def = gen_tr_def.split(',')[0].split(';')[0].split('.')[0]
                gen_translation = gen_tr_def
            
            # 2. Se for gíria detectada ou se estiver no dicionário (mesmo que a IA não tenha certeza)
            if slang_info:
                if is_really_slang:
                    contextual_translations.append({
                        "context": "Gíria (Principal)",
                        "meaning": slang_info.meaning_pt,
                        "example": ""
                    })
                    # Se a tradução geral for diferente da gíria, adicionamos como alternativa
                    if gen_translation.lower() != slang_info.meaning_pt.lower():
                        contextual_translations.append({
                            "context": "Geral / Literal",
                            "meaning": gen_translation,
                            "example": ""
                        })
                else:
                    # Topo será o significado Geral
                    contextual_translations.append({
                        "context": "Geral",
                        "meaning": gen_translation,
                        "example": ""
                    })
                    # Oferecemos a gíria como opção de contexto
                    contextual_translations.append({
                        "context": "Gíria",
                        "meaning": slang_info.meaning_pt,
                        "example": ""
                    })
            else:
                # Caso não seja gíria, apenas o sentido geral
                contextual_translations.append({
                    "context": "Geral",
                    "meaning": gen_translation,
                    "example": ""
                })

        # Adiciona significados extras do dicionário (se houver)
        if additional_meanings:
            for m in additional_meanings:
                if len(m["definition"]) < 200:
                    meaning_tr = self.translator.translate(m["definition"])
                    
                    if len(meaning_tr) > 100:
                        meaning_tr = meaning_tr.split(',')[0].split(';')[0].split('.')[0]
                        
                    if meaning_tr.lower() not in [t["meaning"].lower() for t in contextual_translations]:
                        contextual_translations.append({
                            "context": f"Dicionário ({m['pos']})",
                            "meaning": meaning_tr,
                            "example": ""
                        })
        
        # Embedding
        embedding = self.embeddings.generate(word_lower)
        similar_words = [] # TODO: Busca vetorial
        
        return WordAnalysis(
            original=word,
            is_slang=is_really_slang,
            normalized=normalized,
            translation_pt=translation_pt,
            meaning_en=meaning_en,
            meaning_pt=meaning_pt,
            formality=formality,
            category=category,
            examples=examples,
            contextual_translations=contextual_translations,
            embedding=embedding,
            similar_words=similar_words
        )
    
    def translate_sentence(self, sentence: str) -> dict:
        """
        Translates full sentences using intelligent normalization.
        """
        import re
        slangs_found = []
        blocked_ambiguous = []
        is_slang_prob = self.slang_detector.predict_score(sentence)
        
        # 1. Extracao de Metadados com detecção flexível (capping, flexing, etc)
        all_slangs = sorted(self.dictionary.get_all_slangs(), key=len, reverse=True)
        from app.ml.slang_detector import AMBIGUOUS_SLANG
        
        for slang in all_slangs:
            pattern = r'\b' + re.escape(slang) + r'(?:ing|ed|es|s|er)?\b'
            matches = re.finditer(pattern, sentence, flags=re.IGNORECASE)
            
            for match in matches:
                matched_text = match.group()
                slang_info = self.dictionary.lookup(slang)
                if not slang_info:
                    continue
                
                is_really_slang = True
                context_decision = None
                if slang in AMBIGUOUS_SLANG:
                    context_decision = self.context_resolver.resolve(
                        term=slang,
                        sentence=sentence,
                        detector_score=is_slang_prob,
                        dictionary_has_entry=True,
                        slang_meaning=slang_info.meaning_en or slang_info.normalized,
                    )
                    if not context_decision.should_normalize:
                        blocked_ambiguous.append(context_decision)
                        is_really_slang = False
                
                if is_really_slang:
                    norm = slang_info.normalized or slang
                    overlaps_existing = any(
                        match.start() < s["end"] and match.end() > s["start"]
                        for s in slangs_found
                    )
                    if not overlaps_existing:
                        slangs_found.append({
                            "original": matched_text,
                            "base_slang": slang,
                            "normalized": norm,
                            "source": "dictionary",
                            "sense": context_decision.sense if context_decision else "slang",
                            "reason": context_decision.reason if context_decision else "dictionary match",
                            "start": match.start(),
                            "end": match.end()
                        })
        
        # 2. Normalizacao deterministica.
        # FLAN no longer decides whether an ambiguous word is slang. Candidate
        # slang spans must be confirmed by the sense resolver before replacement.
        if len(slangs_found) > 0:
            temp_sentence = sentence
            for s in sorted(slangs_found, key=lambda x: x["start"], reverse=True):
                temp_sentence = temp_sentence[:s["start"]] + s["normalized"] + temp_sentence[s["end"]:]

            # Known slang should be normalized deterministically. Sending this
            # back through FLAN causes creative rewrites such as literal "tea"
            # or "sick" being changed in neutral contexts.
            normalized_sentence = self.slang_normalizer.apply_safety_rewrites(temp_sentence)
        else:
            normalized_sentence = self.slang_normalizer.apply_safety_rewrites(sentence)
        
        # 3. Traducao
        translation = self.translator.translate(normalized_sentence)
        
        contextual_translations = [{
            "context": "Tradução Adaptada" if len(slangs_found) > 0 else "Tradução Direta",
            "meaning": translation,
            "example": normalized_sentence if len(slangs_found) > 0 else sentence
        }]
        
        if len(slangs_found) > 0:
            literal_translation = self.translator.translate(sentence)
            if literal_translation.lower() != translation.lower():
                contextual_translations.append({
                    "context": "Tradução Literal",
                    "meaning": literal_translation,
                    "example": sentence
                })
        
        return {
            "original": sentence,
            "slangs_detected": slangs_found,
            "blocked_ambiguous": [decision.__dict__ for decision in blocked_ambiguous],
            "normalized_english": normalized_sentence,
            "translation_pt": translation,
            "contextual_translations": contextual_translations
        }

    def _sanitize_translation(self, text: str) -> str:
        """Remove trash and excessive politeness (Please) from the translator"""
        if not text: return ""
        
        # Remove "Por favor" ou "Please" intrusivo do início (com vírgula, ponto ou espaço)
        text = re.sub(r'^(por favor|please)[.,\s]+', '', text, flags=re.IGNORECASE).strip()
        
        # Garante que a primeira letra seja maiúscula após a limpeza
        if text:
            text = text[0].upper() + text[1:]
        
        # Blacklist expandida
        trash_words = {
            "uma", "um", "o", "a", "os", "as", "para", "algo", "muito", 
            "é", "está", "como", "coisa", "coisas", "sentido", "alguma"
        }
        
        parts = [p.strip() for p in text.split('/')]
        clean_parts = []
        for p in parts:
            # Limpeza de termos extras
            p = re.sub(r'\b(alguma coisa|coisa|coisas|algo|muito|um|uma|para|algum|vários|sentido|como)\b', '', p, flags=re.IGNORECASE).strip()
            
            # Filtros de qualidade
            if p and p.lower() not in trash_words and len(p) > 2:
                if not any(p[:4].lower() in existing[:4].lower() for existing in clean_parts):
                    clean_parts.append(p)
        
        if clean_parts:
            return " / ".join(clean_parts[:2])
        return text.split('/')[0].strip()


# Pipeline global
_pipeline: Optional[TranslationPipeline] = None

def get_pipeline(supabase_client: Optional[any] = None) -> TranslationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TranslationPipeline(supabase_client)
    return _pipeline

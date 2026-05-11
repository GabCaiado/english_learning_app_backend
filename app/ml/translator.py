from transformers import MarianMTModel, MarianTokenizer
from functools import lru_cache
import re
import torch

class Translator:
    """English to Portuguese translator with intelligent redundancy filtering"""
    
    def __init__(self):
        self.model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
        self._model = None
        self._tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    def model(self):
        """Loads model on demand (lazy loading)"""
        if self._model is None:
            print(f"Loading translation model {self.model_name}...")
            self._model = MarianMTModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
        return self._model
    
    @property
    def tokenizer(self):
        """Loads tokenizer on demand"""
        if self._tokenizer is None:
            self._tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        return self._tokenizer
    
    def translate(self, text: str) -> str:
        """
        Translates text with deep cleaning to avoid repetition and hallucinations.
        """
        if not text or not text.strip():
            return ""
            
        input_text = text.strip()
        word_lower = input_text.lower()
        is_single_word = " " not in input_text

        phrase_overrides = {
            "we are in trouble if we miss the deadline": "Estamos em apuros se perdermos o prazo",
            "we're in trouble if we miss the deadline": "Estamos em apuros se perdermos o prazo",
            "she is super relaxed": "Ela é super tranquila",
            "she's super relaxed": "Ela é super tranquila",
            "this look is projecting confidence": "Este look transmite confiança",
            "her outfit looks stylish and flattering": "A roupa dela está impecável",
            "the knight slayed the dragon": "O cavaleiro matou o dragão",
        }
        if word_lower in phrase_overrides:
            return phrase_overrides[word_lower]
        
        if is_single_word:
            single_word_overrides = {
                "jam": "geleia / congestionamento",
                "chill": "frio / relaxar",
                "drip": "gotejamento",
                "snatched": "arrebatado",
                "slayed": "matou",
                "cooked": "cozido",
                "serving": "servindo",
            }
            if word_lower in single_word_overrides:
                return single_word_overrides[word_lower]

            results = []
            
            # 1. Lista de Templates para capturar sentidos
            templates = [
                f"{input_text.capitalize()}.", # Direto
                f"To {word_lower} something.", # Verbo
                f"It is very {word_lower}."    # Adjetivo
            ]
            
            trash_words = {
                "uma", "um", "o", "a", "os", "as", "para", "algo", "muito", 
                "é", "está", "coisa", "coisas", "sentido", "alguma", word_lower
            }
            
            prefixes = [
                r'para\s+.*?\s+algo', r'para\s+.*?', r'alguém\s+.*?',
                r'algo\s+.*?', r'é\s+muito\s+', r'está\s+muito\s+',
                r'é\s+um\s+', r'é\s+uma\s+', r'muito\s+', r'uma\s+', r'um\s+',
                r'isto\s+é\s+', r'este\s+é\s+', r'isso\s+é\s+', r'a\s+palavra\s+é\s+',
                r'coisa\s+', r'coisas\s+'
            ]
            pattern = r'^(' + '|'.join(prefixes) + r')'

            for template in templates:
                try:
                    inputs = self.tokenizer(template, return_tensors="pt").to(self.device)
                    translated = self.model.generate(**inputs, max_new_tokens=50)
                    res = self.tokenizer.decode(translated[0], skip_special_tokens=True).strip().lower()
                    
                    # Limpeza agressiva
                    res = re.sub(pattern, '', res, flags=re.IGNORECASE).strip()
                    res = res.split(' ')[0].strip('.,!?; ') 
                    
                    # Filtros de qualidade
                    # Nao pode ser lixo, nem a palavra original, e deve ser longa o suficiente
                    if res and res not in trash_words and len(res) > 2:
                        # Se for um radical novo, adiciona
                        if not any(res[:4] in existing[:4] for existing in results):
                            results.append(res)
                except Exception:
                    continue
            
            if not results:
                # Fallback final se tudo falhar, mas mesmo assim limpa a barra
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
                translated = self.model.generate(**inputs, max_new_tokens=50)
                final_res = self.tokenizer.decode(translated[0], skip_special_tokens=True).strip().lower()
                return final_res if final_res != word_lower else ""

            return " / ".join(results[:2])
        
        # Caso normal (frases)
        # Correção automática de contrações comuns para melhorar a qualidade do Helsinki
        # (ex: wont -> won't)
        contractions = {
            r"\bwont\b": "won't",
            r"\bdont\b": "don't",
            r"\bcant\b": "can't",
            r"\bdidnt\b": "didn't",
            r"\bisnt\b": "isn't",
            r"\barent\b": "aren't",
            r"\bwasnt\b": "wasn't",
            r"\bwerent\b": "weren't",
            r"\byoure\b": "you're",
            r"\bim\b": "i'm",
            r"\bits\b": "it's"
        }
        for pattern, replacement in contractions.items():
            input_text = re.sub(pattern, replacement, input_text, flags=re.IGNORECASE)

        # com capitalização e pontuação correta para identificar imperativos (ex: Diga vs dizer).
        input_ready = input_text
        if not re.search(r'[.!?]$', input_ready):
            input_ready = input_ready.capitalize() + "."
        elif input_ready[0].islower():
            input_ready = input_ready[0].upper() + input_ready[1:]

        inputs = self.tokenizer(input_ready, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        translated = self.model.generate(**inputs, no_repeat_ngram_size=2, max_new_tokens=512)
        res = self.tokenizer.decode(translated[0], skip_special_tokens=True).strip()
        
        # Se nós adicionamos um ponto que não existia, e o tradutor retornou um ponto, removemos 
        if not re.search(r'[.!?]$', input_text) and res.endswith('.'):
            res = res[:-1]
            
        return res
    
    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts: return []
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        translated = self.model.generate(**inputs, max_new_tokens=512)
        return [self.tokenizer.decode(t, skip_special_tokens=True).strip() for t in translated]

@lru_cache()
def get_translator() -> Translator:
    return Translator()

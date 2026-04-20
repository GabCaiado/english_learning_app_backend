from transformers import MarianMTModel, MarianTokenizer
from functools import lru_cache


class Translator:
    """Tradutor ingles para portugues"""
    
    def __init__(self):
        self.model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
        self._model = None
        self._tokenizer = None
    
    @property
    def model(self):
        """Carrega modelo sob demanda (lazy loading)"""
        if self._model is None:
            print("Carregando modelo de traducao...")
            self._model = MarianMTModel.from_pretrained(self.model_name)
        return self._model
    
    @property
    def tokenizer(self):
        """Carrega tokenizer sob demanda"""
        if self._tokenizer is None:
            self._tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        return self._tokenizer
    
    def translate(self, text: str) -> str:
        """
        Traduz texto de ingles para portugues.
        
        Args:
            text: Texto em ingles
            
        Returns:
            Texto traduzido em portugues
        """
        if not text or not text.strip():
            return ""
        
        # Tokeniza
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Gera traducao
        translated = self.model.generate(**inputs)
        
        # Decodifica
        result = self.tokenizer.decode(
            translated[0], 
            skip_special_tokens=True
        )
        
        return result
    
    def translate_batch(self, texts: list[str]) -> list[str]:
        """Traduz multiplos textos de uma vez pra ser mais eficiente"""
        if not texts:
            return []
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        translated = self.model.generate(**inputs)
        
        results = [
            self.tokenizer.decode(t, skip_special_tokens=True)
            for t in translated
        ]
        
        return results


# Instancia global (singleton)
@lru_cache()
def get_translator() -> Translator:
    return Translator()
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class SlangInfo:
    """Informacoes de uma giria"""
    slang: str
    normalized: str
    meaning_en: str
    meaning_pt: str
    formality: str
    region: str
    category: str
    examples: list[str]


class SlangDictionary:
    """
    Dicionario de girias.
    Carrega do banco de dados de forma local e instantanea.
    """
    
    def __init__(self):
        self._cache: dict[str, SlangInfo] = {}
        self._loaded = False
    
    def load_from_supabase(self, supabase_client):
        if self._loaded:
            return
        
        print("Carregando dicionario de girias...")
        
        response = supabase_client.table("slang_dictionary").select("*").execute()
        
        for row in response.data:
            slang = row["word"].lower()
            normalized = row.get("normalized_form")
            self._cache[slang] = SlangInfo(
                slang=slang,
                normalized=normalized if normalized else slang,
                meaning_en=row.get("meaning_en", ""),
                meaning_pt=row.get("translation_pt", ""),
                formality=row.get("formality_level", "informal"),
                region=row.get("region", "universal"),
                category=row.get("category", ""),
                examples=row.get("example_sentences", []) or []
            )
        
        self._loaded = True
        print(f"Carregadas {len(self._cache)} girias")
    
    def lookup(self, word: str) -> Optional[SlangInfo]:
        return self._cache.get(word.lower())
    
    def normalize(self, word: str) -> str:
        """
        Retorna forma normalizada da giria.
        Se nao for giria, retorna a propria palavra.
        """
        info = self.lookup(word)
        if info:
            return info.normalized
        return word
    
    def is_slang(self, word: str) -> bool:
        """Verifica se palavra e uma giria conhecida"""
        return word.lower() in self._cache
    
    def get_all_slangs(self) -> list[str]:
        """Retorna lista de todas as girias"""
        return list(self._cache.keys())


# Instancia global
_dictionary: Optional[SlangDictionary] = None


def get_slang_dictionary() -> SlangDictionary:
    global _dictionary
    if _dictionary is None:
        _dictionary = SlangDictionary()
    return _dictionary
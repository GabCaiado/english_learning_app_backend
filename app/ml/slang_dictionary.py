import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class SlangInfo:
    """slang information"""
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
    Slang dictionary.
    Loads from database in a local and instant way.
    """
    
    def __init__(self):
        # Fallbacks estáticos para gírias essenciais (evita alucinações da IA se o banco estiver vazio)
        self._cache: dict[str, SlangInfo] = {
            "beef": SlangInfo("beef", "conflict", "A disagreement or argument", "conflito / desentendimento", "informal", "universal", "social", []),
            "cap": SlangInfo("cap", "lie", "To lie or exaggerate", "mentira / mentir", "informal", "universal", "social", []),
            "ghost": SlangInfo("ghost", "ignore", "To suddenly stop all communication", "ignorar / sumir", "informal", "universal", "social", []),
            "tea": SlangInfo("tea", "gossip", "Gossip or interesting news", "fofoca / babado", "informal", "universal", "social", []),
            "fire": SlangInfo("fire", "excellent", "Something very good or impressive", "excelente / incrível", "informal", "universal", "social", []),
            "lit": SlangInfo("lit", "amazing", "Very exciting or excellent", "incrível / animado", "informal", "universal", "social", []),
            "goat": SlangInfo("goat", "greatest of all time", "The greatest of all time", "melhor de todos os tempos", "informal", "universal", "social", []),
        }
        self._cache.update({
            "no cap": SlangInfo("no cap", "honestly", "Honestly or without exaggeration", "sem mentira / honestamente", "informal", "universal", "social", []),
            "ghosted": SlangInfo("ghosted", "stopped responding", "Stopped communicating", "parou de responder / sumiu", "informal", "universal", "social", []),
            "ghosted me": SlangInfo("ghosted me", "stopped responding to me", "Stopped communicating with me", "parou de me responder / sumiu", "informal", "universal", "social", []),
            "spill the tea": SlangInfo("spill the tea", "share the gossip", "Share gossip or details", "contar a fofoca / contar o babado", "informal", "universal", "social", []),
            "ate and left no crumbs": SlangInfo("ate and left no crumbs", "did very well", "Performed extremely well", "arrasou / foi muito bem", "informal", "universal", "praise", []),
            "understood the assignment": SlangInfo("understood the assignment", "did exactly what was needed", "Did exactly what was needed", "entendeu a proposta / fez exatamente o necessario", "informal", "universal", "praise", []),
            "lit": SlangInfo("lit", "great", "Very exciting or excellent", "incrivel / animado", "informal", "universal", "social", []),
            "salty": SlangInfo("salty", "upset", "Bitter or upset", "irritado / ressentido", "informal", "universal", "social", []),
            "mid": SlangInfo("mid", "mediocre", "Average or unimpressive", "mediano / sem graca", "informal", "universal", "social", []),
            "facts": SlangInfo("facts", "that is true", "That is true", "verdade / isso e verdade", "informal", "universal", "agreement", []),
            "deadass": SlangInfo("deadass", "seriously", "Seriously or genuinely", "serio / de verdade", "informal", "universal", "emphasis", []),
            "chill": SlangInfo("chill", "relaxed", "Relaxed, calm, or easygoing", "relaxado / tranquilo", "informal", "universal", "personality", []),
            "piece of cake": SlangInfo("piece of cake", "easy", "Something easy to do", "muito facil", "informal", "universal", "social", []),
            "sick": SlangInfo("sick", "great", "Very good or impressive", "muito bom / incrivel", "informal", "universal", "social", []),
            "bummer": SlangInfo("bummer", "disappointment", "Something disappointing or unfortunate", "chatice / decepcao", "informal", "universal", "reaction", []),
            "flopped": SlangInfo("flopped", "failed", "Failed badly", "fracassou", "informal", "universal", "reaction", []),
            "slayed": SlangInfo("slayed", "did very well", "Did something very well", "arrasou", "informal", "universal", "praise", []),
            "lowkey": SlangInfo("lowkey", "somewhat", "Somewhat or secretly", "meio que / discretamente", "informal", "universal", "degree", []),
            "sus": SlangInfo("sus", "suspicious", "Suspicious", "suspeito", "informal", "universal", "judgment", []),
            "legit": SlangInfo("legit", "excellent", "Excellent, real, or credible depending on context", "excelente / legitimo", "informal", "universal", "judgment", []),
            "shady": SlangInfo("shady", "suspicious", "Suspicious or dishonest depending on context", "suspeito / desonesto", "informal", "universal", "judgment", []),
            "hard": SlangInfo("hard", "impressive", "Impressive or intense depending on context", "impressionante / intenso", "informal", "universal", "praise", []),
            "flex": SlangInfo("flex", "show off", "Show off or brag", "ostentar / se exibir", "informal", "universal", "social", []),
            "drip": SlangInfo("drip", "style", "Stylish clothing or appearance", "estilo / roupa estilosa", "informal", "universal", "fashion", []),
            "fit": SlangInfo("fit", "outfit", "Outfit or clothing look", "look / roupa", "informal", "universal", "fashion", []),
            "extra": SlangInfo("extra", "excessive", "Over the top or excessive", "exagerado / demais", "informal", "universal", "judgment", []),
            "wanna": SlangInfo("wanna", "want to", "Want to", "querer", "informal", "universal", "contraction", []),
            "dunno": SlangInfo("dunno", "do not know", "Do not know", "nao sei", "informal", "universal", "contraction", []),
            "lemme": SlangInfo("lemme", "let me", "Let me", "deixe-me", "informal", "universal", "contraction", []),
            "gotta": SlangInfo("gotta", "have to", "Have to", "tenho que / preciso", "informal", "universal", "contraction", []),
            "gonna": SlangInfo("gonna", "going to", "Going to", "vou / vai", "informal", "universal", "contraction", []),
            "my jam": SlangInfo("my jam", "my favorite", "A favorite song, thing, or activity", "meu favorito / minha praia", "informal", "universal", "preference", []),
            "crash at": SlangInfo("crash at", "sleep at", "Sleep somewhere temporarily", "dormir / ficar na casa de alguem", "informal", "universal", "social", []),
            "crashed at": SlangInfo("crashed at", "slept at", "Slept somewhere temporarily", "dormiu / ficou na casa de alguem", "informal", "universal", "social", []),
            "have beef with": SlangInfo("have beef with", "have a conflict with", "Have a disagreement or conflict with someone", "ter conflito com", "informal", "universal", "social", []),
            "has beef with": SlangInfo("has beef with", "has a conflict with", "Has a disagreement or conflict with someone", "tem conflito com", "informal", "universal", "social", []),
            "nasty": SlangInfo("nasty", "amazing", "Very impressive or excellent in a performance context", "incrivel / impressionante", "informal", "universal", "praise", []),
            "ate": SlangInfo("ate", "did very well in", "Performed very well", "arrasou / foi muito bem", "informal", "universal", "praise", []),
            "dipped": SlangInfo("dipped", "left", "Left a place or situation", "saiu / foi embora", "informal", "universal", "movement", []),
            "tripping": SlangInfo("tripping", "overreacting", "Overreacting or acting irrationally", "exagerando / viajando", "informal", "universal", "reaction", []),
            "cooked": SlangInfo("cooked", "in trouble", "In serious trouble or likely to fail", "em apuros / ferrado", "informal", "universal", "reaction", []),
            "serving": SlangInfo("serving", "projecting", "Projecting or giving off a strong vibe", "transmitindo / passando uma imagem de", "informal", "universal", "fashion", []),
            "snatched": SlangInfo("snatched", "stylish and flattering", "Very stylish, flattering, or well put together", "impecavel / muito estilosa", "informal", "universal", "fashion", []),
        })
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
        Returns the normalized form of the slang.
        If it is not a slang, returns the word itself.
        """
        info = self.lookup(word)
        if info:
            return info.normalized
        return word
    
    def is_slang(self, word: str) -> bool:
        """Checks if the word is a known slang"""
        return word.lower() in self._cache
    
    def get_all_slangs(self) -> list[str]:
        """Returns a list of all slangs"""
        return list(self._cache.keys())


# Instancia global
_dictionary: Optional[SlangDictionary] = None


def get_slang_dictionary() -> SlangDictionary:
    global _dictionary
    if _dictionary is None:
        _dictionary = SlangDictionary()
    return _dictionary

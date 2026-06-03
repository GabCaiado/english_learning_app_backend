from typing import Optional
from sqlalchemy.orm import Session

from app.ml.pipeline import get_pipeline, WordAnalysis

class TranslationService:
    def __init__(self, db: Session):
        self.db = db
        self.pipeline = get_pipeline(db)

    def analyze_word(self, word: str) -> WordAnalysis:
        """Deep word analysis (detects slang, translates, generates embedding)"""
        return self.pipeline.analyze_word(word)

    def translate_sentence(self, sentence: str) -> dict:
        """Translates a full sentence using intelligent slang normalization"""
        return self.pipeline.translate_sentence(sentence)

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# Common English words that are NEVER slang — skip ML for these
NEVER_SLANG = {
    # Pronouns & contractions
    "i", "i'm", "i'll", "i've", "i'd", "you", "you're", "you'll", "you've",
    "he", "she", "it", "we", "they", "he's", "she's", "it's", "we're",
    "they're", "they've", "they'll", "that's", "there's", "here's",
    "don't", "can't", "won't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "shouldn't", "wouldn't", "couldn't",
    "dont", "cant", "wont", "didnt", "isnt", "arent", "wasnt", "werent",
    # Auxiliaries & common verbs
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could",
    # Articles & prepositions
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "through", "during", "out", "off",
    # Conjunctions & common words
    "and", "but", "or", "so", "yet", "nor", "not", "no", "yes",
    "this", "that", "these", "those", "my", "your", "his", "her",
    "its", "our", "their", "what", "which", "who", "how", "when",
    "where", "why", "if", "then", "than", "as", "just", "very",
    "go", "get", "got", "let", "put", "set", "see", "say", "said",
    "know", "think", "come", "came", "make", "made", "take", "took",
    "want", "wanted", "wanting", "going", "gone", "back", "again", "too",
}

# Words that can be slang OR literal depending on context
AMBIGUOUS_SLANG = {
    "fire", "sick", "lit", "cool", "dope", "goat", "salty", "tea", "beef", 
    "ghost", "cap", "capping", "bread", "clout", "sauce", "extra", "fit", "later",
    "flex", "drop", "catch", "hard", "soft", "legit", "shady", "drip",
    "jam", "my jam", "crash", "crash at", "crashed at", "nasty", "ate", "dip", "dipped", "trip", "tripping",
    "chill", "cooked", "serving", "snatched", "slayed",
}

class SlangDetector:
    def __init__(self, model_path="models/slang_detector"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if os.path.exists(model_path):
            print(f"Carregando Slang Detector de {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
        else:
            print("Modelo de detecção não encontrado. Usando fallback (Dicionário).")
            self.model = None

    def predict_score(self, text: str) -> float:
        """Returns the probability (0.0 to 1.0) that the text contains slang"""
        if self.model is None: return 0.0
        
        # Heuristic for Neutral Sentence:
        # If the sentence consists entirely of words in NEVER_SLANG, we ignore the ML.
        # This avoids hallucinations in standard phrases like "say you won't let go".
        import re
        words = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
        if words and all(w in NEVER_SLANG for w in words):
            return 0.0

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            return probs[0][1].item()

    def is_slang(self, word: str, context: str = None, threshold: float = 0.75) -> bool:
        """
        Detect if a word is being used as slang.

        Args:
            word: The word to check.
            context: Optional full sentence containing the word. When provided,
                     the full sentence is passed to the model instead of the word
                     in isolation — this lets DistilBERT use surrounding words to
                     correctly resolve ambiguous terms like 'fire', 'sick', 'dead'.
            threshold: Minimum slang probability to classify as slang.
        """
        if self.model is None:
            return False

        # Skip ML for known common words — avoids false positives
        if word.lower().strip() in NEVER_SLANG:
            return False

        # Use full sentence context when available — much more accurate for
        # ambiguous words like 'fire' (literal vs. slang), 'sick', 'dead', etc.
        text_to_classify = context.strip() if context else word.strip()

        inputs = self.tokenizer(text_to_classify, return_tensors="pt", truncation=True, padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            slang_prob = probs[0][1].item()

        return slang_prob >= threshold

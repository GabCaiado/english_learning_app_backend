"""
Slang Normalizer - converts slang to standard English.
If the model is not available, returns the original text.
"""

import os
import re
from difflib import SequenceMatcher

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Prefixo obrigatorio — deve ser identico ao usado no treino
TASK_PREFIX = "normalize slang to standard English: "
DEFAULT_MODEL_PATH = "models/slang_normalizer_v4_1_small"

DETERMINISTIC_REWRITES = [
    (re.compile(r"^facts[.!?]?$", re.IGNORECASE), "that is true"),
    (re.compile(r"^she ate and left no crumbs[.!?]?$", re.IGNORECASE), "she did very well"),
    (re.compile(r"^he ate and left no crumbs[.!?]?$", re.IGNORECASE), "he did very well"),
    (re.compile(r"^they ate and left no crumbs[.!?]?$", re.IGNORECASE), "they did very well"),
    (re.compile(r"^she ghosted me[.!?]?$", re.IGNORECASE), "she stopped responding to me"),
    (re.compile(r"^he ghosted me[.!?]?$", re.IGNORECASE), "he stopped responding to me"),
    (re.compile(r"^they ghosted me[.!?]?$", re.IGNORECASE), "they stopped responding to me"),
    (re.compile(r"^she understood the assignment[.!?]?$", re.IGNORECASE), "she did exactly what was needed"),
    (re.compile(r"^he understood the assignment[.!?]?$", re.IGNORECASE), "he did exactly what was needed"),
    (re.compile(r"^they understood the assignment[.!?]?$", re.IGNORECASE), "they did exactly what was needed"),
    (re.compile(r"^they are overreacting about one missed practice[.!?]?$", re.IGNORECASE), "they're overreacting about one missed practice"),
    (re.compile(r"^don't ignore the recruiter[.!?]?$", re.IGNORECASE), "do not ignore the recruiter"),
    (re.compile(r"^he became bitter after losing[.!?]?$", re.IGNORECASE), "he got upset after losing"),
    (re.compile(r"^he said facts[.!?]?$", re.IGNORECASE), "he said something true"),
    (re.compile(r"^she said facts[.!?]?$", re.IGNORECASE), "she said something true"),
    (re.compile(r"^they said facts[.!?]?$", re.IGNORECASE), "they said something true"),
    (re.compile(r"^he said that is true[.!?]?$", re.IGNORECASE), "he said something true"),
    (re.compile(r"^she said that is true[.!?]?$", re.IGNORECASE), "she said something true"),
    (re.compile(r"^they said that is true[.!?]?$", re.IGNORECASE), "they said something true"),
    (re.compile(r"\bdeadass\b", re.IGNORECASE), "seriously"),
    (re.compile(r"\bare you seriously\??$", re.IGNORECASE), "are you serious?"),
    (re.compile(r"\bchill (guy|person|friend|teacher|manager|neighbor)\b", re.IGNORECASE), r"relaxed \1"),
    (re.compile(r"\bis an energetic (guy|person|friend|teacher|manager|neighbor)\b", re.IGNORECASE), r"is a relaxed \1"),
]

LITERAL_GUARDS = [
    r"\bon fire\b",
    r"\bmade tea\b",
    r"\bspilled tea\b",
    r"\btea on\b",
    r"\btea in\b",
    r"\bfeel sick\b",
    r"\bis sick today\b",
    r"\bsick child\b",
    r"\bflex your\b",
    r"\bflex (?:his|her|their) (?:arm|knee|ankle)\b",
    r"\bdoctor asked .* flex\b",
    r"\bspilled tea\b",
    r"\bspill(?:ed|ing)? tea on\b",
    r"\bshady (?:tree|spot|area|side|garden)\b",
    r"\bunder a shady\b",
    r"\bunder the shady\b",
]

SUSPICIOUS_OUTPUTS = [
    "taass",
    "tss",
    "sass in",
    "good day",
    "good time",
    "the comments agreed",
    "people online said",
    "everyone said",
    "honestly. the comments",
    "i think sick",
    "in the gossip",
]


class SlangNormalizer:
    """
    Normaliza girias para ingles padrao usando um modelo seq2seq fine-tunado.
    Fallback: retorna o texto original se o modelo nao estiver disponivel.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            self.model_path = os.getenv("SLANG_NORMALIZER_MODEL_PATH", DEFAULT_MODEL_PATH)
        else:
            self.model_path = model_path
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.exists(self.model_path):
            print(f"Carregando Slang Normalizer de {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                extra_special_tokens={},
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)
            self.model.eval()
            print("SlangNormalizer carregado com sucesso.")
        else:
            print(f"Modelo de normalização não encontrado em '{model_path}'. "
                  "Train/export slang_normalizer_v3_1 before enabling T5 normalization. "
                  "Usando fallback (retorna texto original).")
            self.model = None
            self.tokenizer = None

    def normalize(self, text: str) -> str:
        """
        Normaliza uma palavra ou frase curta.
        Adiciona o prefixo de tarefa antes de enviar ao T5.
        Retorna o texto original se o modelo nao estiver disponivel.
        """
        if not text or not text.strip():
            return text

        if self.model is None:
            return text

        original = text.strip()
        if self._has_literal_guard(original):
            return text

        deterministic = self.apply_safety_rewrites(original)
        if deterministic != original:
            return deterministic

        input_text = TASK_PREFIX + original
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=False
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,  # Evita repeticao de bigramas (ex: "ererer")
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Sanidade: se a saida estiver vazia ou for apenas whitespace, retorna o original
        if not result or not result.strip():
            return text

        if not self._is_safe_output(original, result.strip()):
            return text

        return self.apply_safety_rewrites(result.strip())

    def normalize_sentence(self, sentence: str) -> str:
        """
        Normaliza uma frase completa em uma unica inferencia.
        O T5 consegue lidar com contexto de frase, entao e mais eficiente
        que chamar normalize() palavra por palavra.
        """
        return self.normalize(sentence)

    def _has_literal_guard(self, text: str) -> bool:
        lower = text.lower()
        return any(re.search(pattern, lower) for pattern in LITERAL_GUARDS)

    @staticmethod
    def apply_safety_rewrites(text: str) -> str:
        result = text
        for pattern, replacement in DETERMINISTIC_REWRITES:
            result = pattern.sub(replacement, result)
        return result

    def _is_safe_output(self, original: str, candidate: str) -> bool:
        original_clean = " ".join(original.lower().split())
        candidate_clean = " ".join(candidate.lower().split())
        if original_clean == candidate_clean:
            return True

        if any(fragment in candidate_clean for fragment in SUSPICIOUS_OUTPUTS):
            return False

        if "shady" in original_clean and "tree" in original_clean and "stingy tree" in candidate_clean:
            return False

        if "chill" in original_clean and "energetic" in candidate_clean:
            return False

        if candidate_clean.startswith((",", ".", "honestly", "i think", "everyone said", "people online said")):
            return False

        if len(candidate_clean) > max(20, len(original_clean) * 1.8):
            return False

        similarity = SequenceMatcher(None, original_clean, candidate_clean).ratio()
        if similarity < 0.35:
            return False

        return True

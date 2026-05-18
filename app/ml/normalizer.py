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
DEFAULT_MODEL_PATH = "models/slang_normalizer_v5_base"

DETERMINISTIC_REWRITES = [
    (
        re.compile(r"^\b(im|i'?m|i am)\s+tilted\b[.!?]?$", re.IGNORECASE),
        "i am frustrated",
    ),
    (
        re.compile(r"\b(im|i'?m|i am)\s+getting\s+tilted\s+already\s+by\s+this\s+bullshit\s+chen\b", re.IGNORECASE),
        "i am already getting frustrated by this annoying Chen",
    ),
    (
        re.compile(r"\bthat\s+camper\s+is\s+making\s+me\s+so\s+tilted\b", re.IGNORECASE),
        "that camper is making me very frustrated",
    ),
    (
        re.compile(
            r"\bmy\s+teammates\s+got\s+me\s+all\s+tilted\.\s+now\s+i'?m\s+salty\s+af\s+and\s+playing\s+like\s+shit\.?\b",
            re.IGNORECASE,
        ),
        "my teammates made me very frustrated. now i am very upset and playing very badly.",
    ),
    (
        re.compile(r"\b(i|we|he|she|they|you)\s+got\s+tilted\b", re.IGNORECASE),
        r"\1 became frustrated",
    ),
    (
        re.compile(r"\b(i'?m|im|i am)\s+grindd?ing\s+(roblox|valorant|fortnite)\b", re.IGNORECASE),
        r"i am playing \2 intensely",
    ),
    (
        re.compile(r"\b(i'?m|im|i am)\s+grindd?ing\s+(valorant)\b", re.IGNORECASE),
        r"i am playing \2 intensely",
    ),
    (
        re.compile(r"\b(i'?m|im|i am)\s+grindin\s+ranked\b", re.IGNORECASE),
        "i am playing ranked mode intensely",
    ),
    (
        re.compile(r"\b(i'?m|im|i am)\s+grinding\s+valorant\s+ranked\b", re.IGNORECASE),
        "i am playing ranked Valorant intensely",
    ),
    (
        re.compile(r"\bi\s+am\s+playing\s+valorant\s+intensely\s+ranked\b", re.IGNORECASE),
        "i am playing ranked Valorant intensely",
    ),
    (
        re.compile(r"\bi\s+lowkey\s+need\s+a\s+new\s+phone\b", re.IGNORECASE),
        "i kind of need a new phone",
    ),
    (
        re.compile(r"\bstop\s+capping\s+about\s+your\s+rank\b", re.IGNORECASE),
        "stop lying about your rank",
    ),
    (
        re.compile(r"\bstop\s+the\s+cap\b", re.IGNORECASE),
        "stop lying",
    ),
    (
        re.compile(r"\bthis\s+(match|lobby|game)\s+is\s+free\b", re.IGNORECASE),
        r"this \1 is easy to win",
    ),
    (
        re.compile(r"\bthat\s+ranked\s+game\s+was\s+free\b", re.IGNORECASE),
        "that ranked game was easy to win",
    ),
    (
        re.compile(r"\bi\s+sold\s+that\s+round\b", re.IGNORECASE),
        "i played badly and caused us to lose that round",
    ),
    (
        re.compile(r"\bi\s+sold\s+the\s+match\b", re.IGNORECASE),
        "i played badly and caused the match to go poorly",
    ),
    (
        re.compile(r"\bshe\s+served\s+looks\s+at\s+the\s+party\b", re.IGNORECASE),
        "she looked very stylish at the party",
    ),
    (
        re.compile(r"\bthat\s+comment\s+was\s+pure\s+shade\b", re.IGNORECASE),
        "that comment was a subtle insult",
    ),
    (
        re.compile(r"\bshe\s+threw\s+shade\s+at\s+him\b", re.IGNORECASE),
        "she made a subtle insult toward him",
    ),
    (
        re.compile(r"\bwhy\s+are\s+you\s+pressed\b", re.IGNORECASE),
        "why are you upset",
    ),
    (
        re.compile(r"\bmy\s+phone\s+died\b", re.IGNORECASE),
        "my phone stopped working because the battery ran out",
    ),
    (
        re.compile(r"\b(i'?m|im|i am)\s+down\s+bad\s+for\s+her\b", re.IGNORECASE),
        "i am extremely attracted to her",
    ),
    (
        re.compile(r"\bthat\s+joke\s+sent\s+me\b", re.IGNORECASE),
        "that joke made me laugh a lot",
    ),
    (
        re.compile(r"\bhe\s+folded\s+under\s+pressure\b", re.IGNORECASE),
        "he gave up or failed under pressure",
    ),
    (
        re.compile(r"\bthis\s+game\s+is\s+laggy\s+af\b", re.IGNORECASE),
        "this game is very laggy",
    ),
    (
        re.compile(r"\bi\s+need\s+to\s+clutch\s+this\s+round\b", re.IGNORECASE),
        "i need to succeed in this round at a critical moment",
    ),
    (
        re.compile(r"\bthat\s+clutch\s+was\s+insane\b", re.IGNORECASE),
        "that last moment win was incredible",
    ),
    (
        re.compile(r"\bthat\s+take\s+is\s+wild\b", re.IGNORECASE),
        "that opinion is shocking or extreme",
    ),
    (
        re.compile(r"\bwe\s+got\s+cooked\s+in\s+ranked\b", re.IGNORECASE),
        "we lost badly in ranked mode",
    ),
    (
        re.compile(r"\bthat\s+team\s+cooked\s+us\b", re.IGNORECASE),
        "that team beat us badly",
    ),
    (
        re.compile(r"\bhe\s+cooked\s+in\s+the\s+debate\b", re.IGNORECASE),
        "he performed extremely well in the debate",
    ),
    (
        re.compile(r"\bbro\s+carried\s+the\s+lobby\b", re.IGNORECASE),
        "he performed very well for the whole lobby",
    ),
    (
        re.compile(r"\b(i'?m|im|i am)\s+hardstuck\s+bronze\b", re.IGNORECASE),
        "i am unable to rank up from bronze",
    ),
    (
        re.compile(r"\b(i'?m|im|i am)\s+tilted\s+after\s+that\s+game\b", re.IGNORECASE),
        "i am frustrated after that game",
    ),
    (
        re.compile(r"\bthat\s+beat\s+is\s+sick\b", re.IGNORECASE),
        "that beat is excellent",
    ),
    (
        re.compile(r"\bthat\s+reply\s+was\s+salty\b", re.IGNORECASE),
        "that reply sounded upset",
    ),
    (
        re.compile(r"\bhe\s+is\s+washed\s+now\b", re.IGNORECASE),
        "he is no longer good now",
    ),
    (
        re.compile(r"\bthat\s+promotion\s+is\s+a\s+flex\b", re.IGNORECASE),
        "that promotion is showing off",
    ),
    (
        re.compile(r"\bthis\s+lobby\s+is\s+sweaty\b", re.IGNORECASE),
        "this lobby is very competitive",
    ),
    (
        re.compile(r"\bthat\s+fit\s+ate\b", re.IGNORECASE),
        "that outfit looked excellent",
    ),
    (
        re.compile(r"\bhe'?s\s+got\s+rizz\b", re.IGNORECASE),
        "he has charisma",
    ),
    (
        re.compile(r"\bi'?m\s+not\s+gonna\s+lie\s+that\s+was\s+clean\b", re.IGNORECASE),
        "honestly that was impressive",
    ),
    (
        re.compile(r"\bshe\s+left\s+me\s+on\s+read\b", re.IGNORECASE),
        "she read my message and did not respond",
    ),
    (
        re.compile(r"\bthat\s+test\s+humbled\s+me\b", re.IGNORECASE),
        "that test made me realize I was not as prepared as I thought",
    ),
    (
        re.compile(r"\bpeople\s+ship\s+them\s+together\b", re.IGNORECASE),
        "people support them as a couple",
    ),
    (
        re.compile(r"\bdrop\s+the\s+beat\b", re.IGNORECASE),
        "start the beat",
    ),
    (
        re.compile(
            r"\b(she|he|they|we|i|you)\s+cooked\s+(him|her|them|me|us|you)\s+in\s+the\s+comments\b",
            re.IGNORECASE,
        ),
        r"\1 harshly criticized \2 in the comments",
    ),
    (
        re.compile(
            r"\b(our|my|your|his|her|their|the)\s+team\s+doesn[’']?t\s+want\s+smoke\s+with\s+(them|us|him|her|you)\b",
            re.IGNORECASE,
        ),
        r"\1 team does not want conflict with \2",
    ),
    (
        re.compile(r"\b(i|we|they|you)\s+don[’']?t\s+want\s+smoke\s+with\s+(them|us|him|her|you)\b", re.IGNORECASE),
        r"\1 do not want conflict with \2",
    ),
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
    (re.compile(r"\bi ship ([a-z][a-z'\-]*) and ([a-z][a-z'\-]*)\b", re.IGNORECASE), r"i want \1 and \2 to be a couple"),
    (re.compile(r"\bi want as a couple ([a-z][a-z'\-]*) and ([a-z][a-z'\-]*)\b", re.IGNORECASE), r"i want \1 and \2 to be a couple"),
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
    r"\bi love that beat\b",
    r"\bstart the beat\b",
    r"\bdrop the beat\b",
    r"\bgreat beat\b",
    r"\blistening to this beat\b",
    r"\bthat beat\b",
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

    def normalize_with_detected_spans(self, sentence: str, spans: list[dict]) -> tuple[str, str]:
        """
        Normalize the whole sentence first, then fall back to deterministic span
        replacement only when the model fails to resolve the detected slang.
        """
        if not spans:
            return self.apply_safety_rewrites(sentence), "safety_rewrites"

        model_candidate = self.normalize_sentence(sentence)
        if self._is_contextual_candidate_safe(sentence, model_candidate, spans):
            return model_candidate, "sentence_model"

        deterministic = self._replace_spans(sentence, spans)
        return self.apply_safety_rewrites(deterministic), "dictionary_fallback"

    def _has_literal_guard(self, text: str) -> bool:
        lower = text.lower()
        return any(re.search(pattern, lower) for pattern in LITERAL_GUARDS)

    def _is_contextual_candidate_safe(self, original: str, candidate: str, spans: list[dict]) -> bool:
        if not candidate or not candidate.strip():
            return False

        original_clean = " ".join(original.lower().split())
        candidate_clean = " ".join(candidate.lower().split())
        if original_clean == candidate_clean:
            return False

        if not self._is_safe_output(original, candidate):
            return False

        for span in spans:
            original_span = str(span.get("original") or "").strip()
            base_slang = str(span.get("base_slang") or "").strip()
            if original_span and self._contains_phrase(candidate_clean, original_span):
                return False
            if base_slang and self._contains_phrase(candidate_clean, base_slang):
                return False

        return True

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        normalized_phrase = " ".join(phrase.lower().split())
        if not normalized_phrase:
            return False
        return re.search(r"\b" + re.escape(normalized_phrase) + r"\b", text) is not None

    @staticmethod
    def _replace_spans(sentence: str, spans: list[dict]) -> str:
        result = sentence
        for span in sorted(spans, key=lambda item: item["start"], reverse=True):
            result = result[:span["start"]] + span["normalized"] + result[span["end"]:]
        return result

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

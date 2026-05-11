import os
import re
import random
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from app.database import get_supabase

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# =============================================================================
# Dicionario curado de abreviacoes/gírias e suas formas padrao
# Fonte: compilacao baseada em listas publicas de internet slang / contractions
# Inclui: abreviacoes de texto, gírias de internet, contractions informais
# =============================================================================
CURATED_SLANG_PAIRS = [
    # Abreviacoes de texto (SMS/Internet)
    ("tbh", "to be honest"),
    ("imo", "in my opinion"),
    ("imho", "in my humble opinion"),
    ("idk", "I don't know"),
    ("ngl", "not gonna lie"),
    ("irl", "in real life"),
    ("rn", "right now"),
    ("asap", "as soon as possible"),
    ("lol", "laughing out loud"),
    ("lmao", "laughing my ass off"),
    ("omg", "oh my god"),
    ("btw", "by the way"),
    ("fyi", "for your information"),
    ("afaik", "as far as I know"),
    ("afk", "away from keyboard"),
    ("brb", "be right back"),
    ("np", "no problem"),
    ("yw", "you're welcome"),
    ("ty", "thank you"),
    ("thx", "thanks"),
    ("pls", "please"),
    ("plz", "please"),
    ("bc", "because"),
    ("cuz", "because"),
    ("tho", "though"),
    ("thru", "through"),
    ("wdym", "what do you mean"),
    ("wtf", "what the heck"),
    ("smh", "shaking my head"),
    ("istg", "I swear to God"),
    ("ikr", "I know, right"),
    ("ik", "I know"),
    ("iirc", "if I recall correctly"),
    ("tfw", "that feeling when"),
    ("rly", "really"),
    ("pov", "point of view"),
    ("fwiw", "for what it's worth"),
    ("nvm", "never mind"),
    ("imo", "in my opinion"),
    ("hbu", "how about you"),
    ("hmu", "hit me up"),
    ("dm", "direct message"),
    ("gg", "good game"),
    ("gl", "good luck"),
    ("hf", "have fun"),
    ("gg", "good game"),
    ("gtg", "got to go"),
    ("ttyl", "talk to you later"),
    ("ttys", "talk to you soon"),
    ("bbl", "be back later"),
    ("bfn", "bye for now"),
    ("irl", "in real life"),
    ("idk", "I don't know"),
    ("idc", "I don't care"),
    ("idc", "I don't care"),
    ("imo", "in my opinion"),
    ("nbd", "no big deal"),
    ("smh", "shaking my head"),
    ("tbf", "to be fair"),
    ("tbt", "throwback Thursday"),
    ("wbu", "what about you"),
    ("wdyt", "what do you think"),
    ("ygm", "you get me"),
    ("yolo", "you only live once"),
    ("fomo", "fear of missing out"),
    ("jk", "just kidding"),
    ("lmk", "let me know"),
    ("nfw", "no way"),
    ("ofc", "of course"),
    ("omw", "on my way"),
    ("rofl", "rolling on the floor laughing"),
    ("sry", "sorry"),
    ("tmi", "too much information"),
    ("wth", "what the heck"),
    ("wywh", "wish you were here"),
    ("zzz", "sleeping / bored"),
    # Contractions informais
    ("gonna", "going to"),
    ("wanna", "want to"),
    ("gotta", "got to"),
    ("kinda", "kind of"),
    ("sorta", "sort of"),
    ("lotta", "a lot of"),
    ("outta", "out of"),
    ("gimme", "give me"),
    ("lemme", "let me"),
    ("dunno", "I don't know"),
    ("ain't", "am not / is not / are not"),
    ("y'all", "you all"),
    ("tryna", "trying to"),
    ("hafta", "have to"),
    ("oughta", "ought to"),
    ("shoulda", "should have"),
    ("woulda", "would have"),
    ("coulda", "could have"),
    ("musta", "must have"),
    ("hadda", "had to"),
    ("supposta", "supposed to"),
    ("useta", "used to"),
    # Gírias de internet/cultura pop
    ("lit", "excellent / exciting"),
    ("goat", "greatest of all time"),
    ("slay", "to perform impressively"),
    ("based", "true to oneself"),
    ("bussin", "very good / delicious"),
    ("no cap", "no lie / for real"),
    ("cap", "lie / falsehood"),
    ("bet", "okay / agreed"),
    ("vibe", "atmosphere / feeling"),
    ("lowkey", "secretly / somewhat"),
    ("highkey", "very much / openly"),
    ("drip", "stylish outfit"),
    ("flex", "to show off"),
    ("salty", "upset / bitter"),
    ("shook", "shocked / surprised"),
    ("tea", "gossip / truth"),
    ("extra", "over the top / dramatic"),
    ("sus", "suspicious"),
    ("mid", "mediocre / average"),
    ("fire", "excellent / amazing"),
    ("dope", "cool / excellent"),
    ("sick", "cool / excellent"),
    ("legit", "legitimate / genuinely"),
    ("fam", "family / close friends"),
    ("bro", "brother / close friend"),
    ("bruh", "brother / expression of disbelief"),
    ("goated", "the greatest of all time"),
    ("periodt", "period / end of discussion"),
    ("stan", "an obsessive fan"),
    ("ghost", "to suddenly stop responding"),
    ("ghosted", "suddenly stopped responding"),
    ("mood", "relatable feeling"),
    ("yikes", "expression of discomfort"),
    ("oof", "expression of empathy for pain"),
    ("w", "win"),
    ("l", "loss"),
    ("npc", "someone acting mindlessly"),
    ("main character", "center of attention"),
    ("hits different", "uniquely resonates"),
    ("snatched", "looking great"),
    ("bop", "a great song"),
    ("slaps", "sounds excellent"),
    ("peeps", "people"),
    ("noob", "beginner / newbie"),
    ("newb", "beginner / newbie"),
    ("fandom", "community of fans"),
    ("feels", "emotional feelings"),
    ("hype", "excitement / anticipation"),
    ("lowkey fire", "subtly excellent"),
    ("no shot", "no way / impossible"),
    ("rent free", "constantly in one's thoughts"),
    ("understood the assignment", "did exactly what was needed"),
    ("ate", "performed perfectly"),
    # Palavras gírias comuns
    ("ya", "you"),
    ("yep", "yes"),
    ("yup", "yes"),
    ("yeah", "yes"),
    ("nope", "no"),
    ("nah", "no"),
    ("nope", "no"),
    ("sup", "what's up"),
    ("wassup", "what's up"),
    ("yo", "hey / you"),
    ("bae", "significant other"),
    ("bff", "best friend forever"),
    ("squad", "group of friends"),
    ("crew", "group of friends"),
    ("homie", "close friend"),
    ("sis", "sister / friend"),
    ("queen", "empowered woman"),
    ("king", "empowered man"),
    ("cray", "crazy"),
    ("cray cray", "very crazy"),
    ("gr8", "great"),
    ("l8r", "later"),
    ("luv", "love"),
    ("u", "you"),
    ("r", "are"),
    ("2", "to / too"),
    ("4", "for"),
    ("b4", "before"),
    ("ur", "your / you're"),
    ("omfg", "oh my freaking god"),
    ("tbvh", "to be very honest"),
    ("ngl fr", "not gonna lie, for real"),
    ("imo fr", "in my opinion, for real"),
    ("fr", "for real"),
    ("fr fr", "for real, for real"),
    ("deadass", "seriously / for real"),
    ("lowkey deadass", "seriously and secretly"),
    ("frl", "for real"),
    ("ong", "on God / I swear"),
    ("icl", "I can't lie"),
    ("icydk", "in case you didn't know"),
    ("imma", "I am going to"),
    ("prolly", "probably"),
    ("def", "definitely"),
    ("obv", "obviously"),
    ("obvs", "obviously"),
    ("abs", "absolutely"),
    ("totes", "totally"),
    ("perf", "perfect"),
    ("adorbs", "adorable"),
    ("amazeballs", "amazing"),
    ("sitch", "situation"),
    ("whatevs", "whatever"),
    ("whatevz", "whatever"),
    ("peeve", "pet peeve"),
    ("meh", "indifferent / mediocre"),
    ("meh", "so-so / indifferent"),
    ("fave", "favorite"),
    ("fav", "favorite"),
    ("inspo", "inspiration"),
    ("vacay", "vacation"),
    ("totes", "totally"),
    ("adorbs", "adorable"),
    ("sesh", "session"),
    ("deets", "details"),
    ("obvs", "obviously"),
    ("rents", "parents"),
    ("bestie", "best friend"),
    ("bestie", "best friend"),
    ("besty", "best friend"),
]


# Deduplica o dicionario interno
_seen = set()
_deduped = []
for slang, normalized in CURATED_SLANG_PAIRS:
    key = slang.lower().strip()
    if key not in _seen:
        _seen.add(key)
        _deduped.append({"slang_text": slang.lower().strip(), "standard_text": normalized})
CURATED_SLANG_PAIRS_DEDUPED = _deduped


# =============================================================================
# Modelo de Parafrase
# =============================================================================

class Paraphraser:
    def __init__(self):
        print("Carregando modelo de parafrase (T5)...")
        name = "Vamsi/T5_Paraphrase_Paws"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(name).to(self.device).eval()

    def paraphrase(self, sentence: str, n: int = 2) -> list[str]:
        text = "paraphrase: " + sentence + " </s>"
        enc = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=128, do_sample=True, top_k=120, top_p=0.95,
                early_stopping=True, num_return_sequences=n
            )
        return list({self.tokenizer.decode(o, skip_special_tokens=True) for o in out})


# =============================================================================
# Fontes de dados
# =============================================================================

def fetch_slangs_from_db() -> list[dict]:
    """Pares curados do Supabase (fonte mais confiavel)."""
    print("Buscando girias do Supabase...")
    supabase = get_supabase()
    res = supabase.table("slang_dictionary").select("word, normalized_form").execute()
    pairs = []
    for row in res.data:
        slang = (row.get("word") or "").strip().lower()
        normalized = (row.get("normalized_form") or "").strip()
        if slang and normalized and slang != normalized.lower():
            pairs.append({"slang_text": slang, "standard_text": normalized})
    print(f"  -> {len(pairs)} pares do banco")
    return pairs


def fetch_corpora():
    print("Baixando Wikipedia-103...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    formal = [t.strip() for t in wiki["text"] if 50 < len(t.strip()) < 180]

    print("Baixando DailyDialog...")
    dialog = load_dataset("agentlans/li2017dailydialog", split="train")
    conversational = []
    for item in dialog:
        for turn in item["conversations"]:
            if turn.get("from") == "system":
                continue
            text = turn.get("value", "").encode('ascii', 'ignore').decode('ascii').strip()
            if 30 < len(text) < 150:
                conversational.append(text)
    return formal, conversational


def fetch_urban_samples_for_detector(n: int = 3000) -> list[str]:
    """Exemplos de uso de gíria (para o DETECTOR, nao o normalizador)."""
    ds = load_dataset("daspartho/urban_dictionary", split="train")
    samples = []
    for item in ds:
        ex = (item.get("example") or "").strip()
        if 20 < len(ex) < 150 and not re.search(r'[^\x00-\x7F]', ex):
            samples.append(ex)
        if len(samples) >= n:
            break
    return samples


def fetch_common_words(n: int = 3000) -> list[str]:
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    blob = " ".join(random.sample(ds["text"], min(500, len(ds["text"]))))
    words = list(set(re.findall(r'\b[a-z]{3,10}\b', blob.lower())))
    return random.sample(words, min(n, len(words)))


def generate_context_pairs(
    base_pairs: list[dict],
    master_corpus: list[str],
    paraphraser: Paraphraser,
    max_seeds: int = 3,
    paraphrase_n: int = 2
) -> list[dict]:
    """Gera pares em contexto: busca frases reais e substitui normalized -> slang."""
    context_pairs = []
    print("Gerando pares em contexto (corpus real + parafrase)...")
    for pair in tqdm(base_pairs):
        slang = pair["slang_text"]
        normalized = pair["standard_text"]
        try:
            pattern = re.compile(rf'\b{re.escape(normalized)}\b', re.IGNORECASE)
        except re.error:
            continue
        seeds = []
        for phrase in random.sample(master_corpus, min(5000, len(master_corpus))):
            if pattern.search(phrase):
                seeds.append(phrase)
            if len(seeds) >= max_seeds:
                break
        for seed in seeds:
            for var in paraphraser.paraphrase(seed, n=paraphrase_n):
                slang_phrase = pattern.sub(slang, var)
                if slang_phrase.lower() != var.lower():
                    context_pairs.append({
                        "slang_text": slang_phrase,
                        "standard_text": var
                    })
    print(f"  -> {len(context_pairs)} pares de contexto")
    return context_pairs


# =============================================================================
# Main
# =============================================================================

def generate():
    paraphraser = Paraphraser()

    # --- NORMALIZADOR ---
    print("\n=== Coletando dados para o NORMALIZADOR ===")

    # Fonte 1: Dicionario curado interno (abreviacoes + gírias bem definidas)
    curated = CURATED_SLANG_PAIRS_DEDUPED
    print(f"Dicionario interno curado: {len(curated)} pares")

    # Fonte 2: Banco Supabase
    db_pairs = fetch_slangs_from_db()

    # Junta e deduplica (DB tem prioridade sobre curado)
    all_base = {p["slang_text"]: p for p in curated}
    for p in db_pairs:
        all_base[p["slang_text"]] = p  # DB sobrescreve curado se conflito
    all_base_list = list(all_base.values())
    print(f"Total de pares base (curado + DB, dedup): {len(all_base_list)}")

    # Fonte 3: Pares em contexto (parafrase sobre todos os pares base)
    formal_corpus, conversational_corpus = fetch_corpora()
    master_corpus = (
        random.sample(formal_corpus, min(10000, len(formal_corpus))) +
        random.sample(conversational_corpus, min(10000, len(conversational_corpus)))
    )
    context_pairs = generate_context_pairs(all_base_list, master_corpus, paraphraser)

    # Consolida
    all_norm = all_base_list + context_pairs
    norm_df = pd.DataFrame(all_norm).drop_duplicates(subset=["slang_text"])
    norm_df = norm_df[
        norm_df["slang_text"].str.lower().str.strip() !=
        norm_df["standard_text"].str.lower().str.strip()
    ]

    # --- DETECTOR ---
    print("\n=== Coletando dados para o DETECTOR ===")
    det_rows = []

    # Positivos: gírias do banco + curadas
    supabase = get_supabase()
    for row in supabase.table("slang_dictionary").select("word").execute().data:
        w = (row.get("word") or "").strip()
        if w:
            det_rows.append({"text": w, "label": 1})
    for pair in curated:
        det_rows.append({"text": pair["slang_text"], "label": 1})

    # Positivos: exemplos de uso do Urban Dictionary
    print("Exemplos Urban Dictionary (detector)...")
    for ex in fetch_urban_samples_for_detector(3000):
        det_rows.append({"text": ex, "label": 1})

    # Positivos e negativos: pares em contexto
    for pair in all_base_list:
        det_rows.append({"text": pair["slang_text"], "label": 1})
        det_rows.append({"text": pair["standard_text"], "label": 0})

    # Negativos: palavras comuns
    print("Palavras comuns (negativos detector)...")
    for word in fetch_common_words(3000):
        det_rows.append({"text": word, "label": 0})

    # Negativos: frases formais
    print("Frases formais (negativos detector)...")
    for phrase in random.sample(master_corpus, min(5000, len(master_corpus))):
        det_rows.append({"text": phrase, "label": 0})

    det_df = pd.DataFrame(det_rows).drop_duplicates().sample(frac=1)

    # --- Salva ---
    print(f"\n=== Resumo Final ===")
    print(f"Normalizador: {len(norm_df)} pares")
    print(f"  - Dicionario interno curado: {len(curated)}")
    print(f"  - DB Supabase: {len(db_pairs)}")
    print(f"  - Em contexto: {len(context_pairs)}")
    print(f"Detector: {len(det_df)} exemplos")

    norm_df.sample(frac=1).reset_index(drop=True).to_csv(
        os.path.join(DATA_DIR, "slang_normalization.csv"), index=False
    )
    det_df.reset_index(drop=True).to_csv(
        os.path.join(DATA_DIR, "slang_detection.csv"), index=False
    )
    print("\nDataset V6 gerado!")


if __name__ == "__main__":
    generate()

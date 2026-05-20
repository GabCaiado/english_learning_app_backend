# English Learning App — Backend

The API and ML pipeline powering an AI-driven English learning tool for Brazilian Portuguese speakers. Handles translation, slang detection, semantic search, vocabulary storage, and user feedback — with all ML models running locally.

> **Frontend repo:** [english_learning_app_frontend](https://github.com/GabCaiado/english_learning_app_frontend)

---

## Overview

This FastAPI service exposes REST endpoints consumed by the Next.js frontend. Its core is a multi-step ML pipeline that detects slang, normalizes informal English, translates to Brazilian Portuguese, and stores sentence embeddings for semantic search. All models run offline — no external AI API calls.

---

## Features

- **EN → PT-BR translation** — Helsinki-NLP `opus-mt-en-pt` transformer model
- **Slang detection** — fine-tuned DistilBERT classifier with hard bypass and ambiguous-word sets
- **Slang normalization** — regex rewrites for known patterns, ByT5-small seq2seq for unknowns
- **Semantic search** — `all-MiniLM-L6-v2` embeddings stored in Supabase pgvector
- **Context resolution** — disambiguates multi-meaning words before translation
- **Vocabulary CRUD** — save and retrieve user word lists
- **Translation feedback** — users submit corrections; stored for future fine-tuning
- **JWT auth** — Supabase token validation on all user-scoped endpoints

---

## Tech Stack

| Category | Technology |
|---|---|
| Framework | FastAPI |
| Language | Python 3.11 |
| Schemas | Pydantic v2 |
| ML — Slang Detection | DistilBERT (fine-tuned classifier) |
| ML — Normalization | ByT5-small (seq2seq) |
| ML — Translation | Helsinki-NLP `opus-mt-en-pt` |
| ML — Embeddings | `all-MiniLM-L6-v2` via Sentence Transformers |
| Database | Supabase (PostgreSQL + pgvector) |
| Auth | Supabase JWT validation |
| Fallback | Slang dictionary + Free Dictionary API (3 s timeout) |

---

## Project Structure

```
english_learning_app_backend/
├── app/
│   ├── main.py                  # FastAPI app factory, CORS, router registration
│   ├── auth.py                  # JWT validation (Supabase)
│   ├── config.py                # Pydantic BaseSettings — reads .env
│   ├── database.py              # Supabase client factory (Depends pattern)
│   ├── routers/
│   │   └── translate.py         # Main API endpoints
│   ├── schemas/                 # Pydantic request/response models
│   ├── ml/
│   │   ├── pipeline.py          # Orchestrates the full ML flow
│   │   ├── slang_detector.py    # DistilBERT classifier
│   │   ├── normalizer.py        # Regex + ByT5-small normalization
│   │   ├── translator.py        # opus-mt-en-pt translation
│   │   ├── embeddings.py        # MiniLM embeddings + pgvector storage
│   │   └── context_resolver.py  # Multi-meaning disambiguation
│   └── services/                # Stub layer (business logic lives in routers for now)
├── data/                        # Training data
├── models/                      # Fine-tuned model checkpoints
├── tests/
│   └── test_sense_classifier_integration.py
└── requirements.txt
```

---

## ML Pipeline

`app/ml/pipeline.py` orchestrates every request end-to-end:

```
Input (word or sentence)
        │
        ▼
1. Slang Detection  ──── DistilBERT classifier (threshold 0.75)
                         NEVER_SLANG bypass (pronouns, auxiliaries)
                         AMBIGUOUS_SLANG always gets full sentence context
        │
        ▼
2. Normalization  ─────  Deterministic regex for high-confidence patterns
                         ByT5-small seq2seq for unknown slang
        │
        ▼
3. Context Resolution ── Disambiguates multi-meaning words
        │
        ▼
4. Translation  ───────  Helsinki-NLP opus-mt-en-pt  (EN → PT-BR)
        │
        ▼
5. Embeddings  ────────  all-MiniLM-L6-v2 → stored in Supabase pgvector
        │
        ▼
Output (translation, slang flag, formality, embedding ID)
```

**Fallback:** if fine-tuned checkpoints are missing, the pipeline falls back to the slang dictionary + Free Dictionary API (3 s timeout).

All models **lazy-load on first request** and are cached for subsequent calls.

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/translate/word/{word}` | Optional | Full word analysis |
| `POST` | `/translate/sentence` | Optional | Full sentence translation |
| `GET` | `/words` | Required | User's saved vocabulary |
| `POST` | `/words` | Required | Save a word |
| `POST` | `/translation_feedback` | Required | Submit a translation correction |

---

## Getting Started

### Prerequisites

- Python 3.11+
- A [Supabase](https://supabase.com) project with the **pgvector** extension enabled

### Installation

```bash
git clone https://github.com/GabCaiado/english_learning_app_backend.git
cd english_learning_app_backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file at the root:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

### Running Locally

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

> On first request, Hugging Face models download automatically (~1 GB total). Subsequent starts use the local cache.

CORS is configured to allow `localhost:3000` and `localhost:5173`.

---

## Tests

```bash
pytest tests/test_sense_classifier_integration.py
```

The integration test runs the full sense classifier against real inputs (requires the model checkpoint to be present).

---

## Project Status

| Feature | Status |
|---|---|
| Translation pipeline | Done |
| Slang detection + normalization | Done |
| Embeddings + semantic search | Done |
| Vocabulary CRUD endpoints | Done |
| Translation feedback storage | Done |
| Docker / deployment config | Done |
| Spaced repetition logic | Planned |

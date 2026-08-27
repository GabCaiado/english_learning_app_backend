import uuid
import math
import json
import random
from typing import List, Optional, Tuple
from datetime import date, datetime, timedelta

from app.models.vocabulary import UserWord
from app.models.study import StudySession, ReviewAttempt
from app.repositories.word_repository import WordRepository
from app.repositories.user_repository import UserRepository
from app.services.auth_service import redis_client

# Curated intermediate-level vocabulary for "word of the day" (avoids obscure/archaic
# words that generic random-word APIs return, which translate poorly for learners).
COMMON_WORDS = [
    "ambitious", "generous", "curious", "reliable", "convenient", "flexible", "grateful",
    "enthusiastic", "thoughtful", "efficient", "considerate", "resilient", "spontaneous",
    "meticulous", "versatile", "compassionate", "diligent", "optimistic", "pragmatic",
    "vivid", "subtle", "genuine", "profound", "inevitable", "reluctant", "ambiguous",
    "coherent", "eloquent", "candid", "sincere", "tedious", "plausible", "arbitrary",
    "innovative", "substantial", "prevalent", "tangible", "intriguing", "elusive",
    "mundane", "resourceful", "tenacious", "whimsical", "articulate", "benevolent",
    "cordial", "empathetic", "frugal", "gregarious", "humble", "impartial", "jubilant",
    "keen", "lucid", "nostalgic", "obstinate", "perceptive", "quaint", "serene",
    "tactful", "unwavering", "vibrant", "witty", "adequate", "brief", "cautious",
    "deliberate", "eager", "faint", "graceful", "harsh", "immense", "joyful", "keenly",
    "lively", "modest", "notable", "obvious", "peculiar", "quiet", "rapid", "steady",
    "thorough", "urgent", "valid", "wary", "abundant", "brilliant", "clumsy", "durable",
    "elegant", "fragile", "gentle", "humid", "inevitable", "jagged", "lavish", "mellow",
    "narrow", "outdated", "polite", "rigid", "shallow", "sturdy", "timid", "vague",
    "wealthy", "accomplish", "achieve", "acquire", "adapt", "anticipate", "assume",
    "attain", "clarify", "collaborate", "commit", "compensate", "confront", "consult",
    "convince", "cooperate", "demonstrate", "determine", "emphasize", "enhance",
    "establish", "estimate", "evaluate", "examine", "exceed", "expand", "facilitate",
    "generate", "identify", "implement", "improve", "indicate", "influence", "inspire",
    "investigate", "justify", "maintain", "negotiate", "obtain", "overcome", "persuade",
    "predict", "prioritize", "recognize", "reduce", "reinforce", "resolve", "sustain",
    "utilize", "verify",
]


class WordService:
    XP_PER_CORRECT_REVIEW = 10

    def __init__(self, db):
        self.db = db
        self.word_repo = WordRepository(db)
        self.user_repo = UserRepository(db)

    @staticmethod
    def _update_streak(user, today: date) -> None:
        last = user.last_activity_date
        if last == today:
            if user.current_streak == 0:
                user.current_streak = 1
        elif last == today - timedelta(days=1):
            user.current_streak += 1
        else:
            user.current_streak = 1
        user.longest_streak = max(user.longest_streak, user.current_streak)
        user.last_activity_date = today

    def get_user_words(self, user_id: uuid.UUID) -> List[UserWord]:
        return self.word_repo.get_all_user_words(user_id)

    def add_user_word(
        self,
        user_id: uuid.UUID,
        word: str,
        translation: str,
        normalized_form: Optional[str] = None,
        is_slang: bool = False,
        formality_level: Optional[str] = None,
        category: Optional[str] = None,
        personal_note: Optional[str] = None,
        context_sentence: Optional[str] = None,
        source: Optional[str] = None,
        embedding: Optional[List[float]] = None
    ) -> UserWord:
        existing = self.word_repo.get_user_word_by_name(user_id, word)
        if existing:
            raise ValueError("Palavra ja adicionada ao vocabulario")

        slang_entry = self.word_repo.get_slang_entry_by_word(word)
        slang_id = slang_entry.id if slang_entry else None

        user_word = self.word_repo.create_user_word(
            user_id=user_id,
            word=word,
            translation=translation,
            normalized_form=normalized_form,
            is_slang=is_slang,
            formality_level=formality_level,
            category=category,
            personal_note=personal_note,
            context_sentence=context_sentence,
            source=source,
            embedding=embedding
        )
        if slang_id:
            user_word.slang_id = slang_id
            self.db.add(user_word)
            self.db.commit()
            self.db.refresh(user_word)

        return user_word

    def update_user_word(self, user_id: uuid.UUID, word_id: uuid.UUID, update_data: dict) -> UserWord:
        user_word = self.word_repo.get_word_by_id(word_id)
        if not user_word or user_word.user_id != user_id:
            raise ValueError("Palavra nao encontrada ou sem permissao")

        return self.word_repo.update_user_word(user_word, update_data)

    def delete_user_word(self, user_id: uuid.UUID, word_id: uuid.UUID) -> None:
        user_word = self.word_repo.get_word_by_id(word_id)
        if not user_word or user_word.user_id != user_id:
            raise ValueError("Palavra nao encontrada ou sem permissao")

        self.word_repo.delete_user_word(user_word)

    def get_due_reviews(self, user_id: uuid.UUID) -> List[UserWord]:
        return self.word_repo.get_due_reviews(user_id)

    def get_weekly_progress(self, user_id: uuid.UUID, days: int = 7) -> List[dict]:
        """Words added per day for the last `days` days (rolling window ending today)."""
        words = self.word_repo.get_words_added_last_n_days(user_id, days)
        counts: dict = {}
        for w in words:
            day_key = w.created_at.date().isoformat()
            counts[day_key] = counts.get(day_key, 0) + 1

        today = date.today()
        result = []
        for i in range(days - 1, -1, -1):
            day = today - timedelta(days=i)
            key = day.isoformat()
            result.append({"date": key, "count": counts.get(key, 0)})
        return result

    async def get_word_of_the_day(self, user_id: uuid.UUID, translation_service) -> Optional[dict]:
        """Returns the same curated word for this user for the rest of the day (cached in Redis),
        picking a new one the user hasn't added yet and enriching it via the translation pipeline."""
        cache_key = f"word_of_day:{user_id}:{date.today().isoformat()}"
        cached = await redis_client.get(cache_key)
        if cached:
            return json.loads(cached)

        candidates = random.sample(COMMON_WORDS, k=len(COMMON_WORDS))
        for word in candidates:
            if self.word_repo.get_user_word_by_name(user_id, word):
                continue

            try:
                analysis = translation_service.analyze_word(word)
                if not analysis.translation_pt:
                    continue
                example = analysis.examples[0] if analysis.examples else None
                example_pt = None
                if example:
                    try:
                        example_pt = translation_service.translate_sentence(example).get("translation_pt")
                    except Exception:
                        example_pt = None
            except Exception:
                continue

            result = {
                "word": word,
                "translation_pt": analysis.translation_pt,
                "meaning_pt": analysis.meaning_pt,
                "example": example,
                "example_pt": example_pt,
            }
            await redis_client.set(cache_key, json.dumps(result), ex=60 * 60 * 26)
            return result
        return None

    # --- Spaced Repetition (SRS) SuperMemo-2 (SM2) Algorithm ---
    def log_review_attempt(
        self,
        user_id: uuid.UUID,
        session_id: Optional[uuid.UUID],
        word_id: uuid.UUID,
        quality_rating: int, # Quality score from 0 to 5
        user_answer: Optional[str] = None,
        response_time_ms: Optional[int] = None
    ) -> Tuple[ReviewAttempt, UserWord]:
        user_word = self.word_repo.get_word_by_id(word_id)
        if not user_word or user_word.user_id != user_id:
            raise ValueError("Palavra nao encontrada ou sem permissao")

        is_correct = quality_rating >= 3

        # Update stats
        if is_correct:
            user_word.times_correct += 1
        else:
            user_word.times_incorrect += 1

        # SuperMemo 2 logic:
        # Calculate new Easiness Factor (EF)
        ef = user_word.easiness_factor + (0.1 - (5 - quality_rating) * (0.08 + (5 - quality_rating) * 0.02))
        user_word.easiness_factor = max(1.3, ef)

        # Calculate repetitions and interval
        if is_correct:
            if user_word.repetitions == 0:
                user_word.interval_days = 1
            elif user_word.repetitions == 1:
                user_word.interval_days = 6
            else:
                user_word.interval_days = math.ceil(user_word.interval_days * user_word.easiness_factor)
            user_word.repetitions += 1
        else:
            user_word.repetitions = 0
            user_word.interval_days = 1

        # Save date updates
        today = date.today()
        user_word.last_review_date = today
        user_word.next_review_date = today + timedelta(days=user_word.interval_days)
        user_word.next_review_at = datetime.utcnow() + timedelta(days=user_word.interval_days)

        # Update mastery status
        if user_word.repetitions >= 5:
            user_word.is_mastered = True

        self.db.add(user_word)
        self.db.commit()
        self.db.refresh(user_word)

        # Award XP and update the daily streak for every review
        user = self.user_repo.get_by_id(user_id)
        if user:
            if is_correct:
                user.total_xp += self.XP_PER_CORRECT_REVIEW
            self._update_streak(user, today)
            self.db.add(user)
            self.db.commit()

        # Update user XP and daily streak if session exists
        if session_id:
            session = self.word_repo.get_study_session_by_id(session_id)
            if session and not session.is_completed:
                session.words_studied += 1
                if is_correct:
                    session.words_correct += 1
                    session.xp_earned += 2 # 2 XP per correct SRS word
                else:
                    session.words_incorrect += 1
                self.db.add(session)
                self.db.commit()

        # Record Review Attempt
        attempt = self.word_repo.create_review_attempt(
            session_id=session_id,
            user_word_id=word_id,
            is_correct=is_correct,
            user_answer=user_answer,
            quality_rating=quality_rating,
            response_time_ms=response_time_ms
        )

        return attempt, user_word

    # --- Study Sessions ---
    def start_study_session(self, user_id: uuid.UUID, session_type: str) -> StudySession:
        return self.word_repo.create_study_session(user_id, session_type)

    def end_study_session(
        self,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        duration_seconds: int
    ) -> StudySession:
        session = self.word_repo.get_study_session_by_id(session_id)
        if not session or session.user_id != user_id:
            raise ValueError("Sessao de estudo nao encontrada")

        session.duration_seconds = duration_seconds
        session.is_completed = True
        session.ended_at = datetime.utcnow()

        self.db.add(session)
        
        # Credit session XP to the user
        user = self.user_repo.get_by_id(user_id)
        if user:
            user.total_xp += session.xp_earned
            self.db.add(user)

        self.db.commit()
        self.db.refresh(session)
        return session

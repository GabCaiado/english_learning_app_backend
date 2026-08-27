import uuid
from typing import List, Optional
from datetime import date
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session

from app.models.vocabulary import UserWord, SlangDictionaryEntry, WordExample, WordVideo
from app.models.study import StudySession, ReviewAttempt

class WordRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_word_by_id(self, word_id: uuid.UUID) -> Optional[UserWord]:
        stmt = select(UserWord).where(UserWord.id == word_id)
        return self.db.scalars(stmt).first()

    def get_user_word_by_name(self, user_id: uuid.UUID, word: str) -> Optional[UserWord]:
        stmt = select(UserWord).where(
            UserWord.user_id == user_id,
            func.lower(UserWord.word) == word.strip().lower()
        )
        return self.db.scalars(stmt).first()

    def get_all_user_words(self, user_id: uuid.UUID) -> List[UserWord]:
        stmt = select(UserWord).where(UserWord.user_id == user_id).order_by(UserWord.created_at.desc())
        return list(self.db.scalars(stmt).all())

    def get_user_words_page(self, user_id: uuid.UUID, limit: int, offset: int) -> tuple[List[UserWord], int]:
        base = select(UserWord).where(UserWord.user_id == user_id)
        total = self.db.scalar(select(func.count()).select_from(base.subquery())) or 0
        stmt = base.order_by(UserWord.created_at.desc()).limit(limit).offset(offset)
        items = list(self.db.scalars(stmt).all())
        return items, total

    def create_user_word(
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
        user_word = UserWord(
            user_id=user_id,
            word=word.strip(),
            translation=translation.strip(),
            normalized_form=normalized_form,
            is_slang=is_slang,
            formality_level=formality_level,
            category=category,
            personal_note=personal_note,
            context_sentence=context_sentence,
            source=source,
            embedding=embedding
        )
        self.db.add(user_word)
        self.db.commit()
        self.db.refresh(user_word)
        return user_word

    def update_user_word(self, user_word: UserWord, update_data: dict) -> UserWord:
        for key, value in update_data.items():
            if hasattr(user_word, key):
                setattr(user_word, key, value)
        self.db.add(user_word)
        self.db.commit()
        self.db.refresh(user_word)
        return user_word

    def delete_user_word(self, user_word: UserWord) -> None:
        self.db.delete(user_word)
        self.db.commit()

    # Spaced Repetition (SRS)
    def get_due_reviews(self, user_id: uuid.UUID) -> List[UserWord]:
        stmt = select(UserWord).where(
            UserWord.user_id == user_id,
            UserWord.is_archived.is_(False),
            UserWord.next_review_date <= func.current_date()
        ).order_by(UserWord.next_review_date.asc())
        return list(self.db.scalars(stmt).all())

    # Slang Dictionary
    def get_slang_entry_by_word(self, word: str) -> Optional[SlangDictionaryEntry]:
        stmt = select(SlangDictionaryEntry).where(func.lower(SlangDictionaryEntry.word) == word.strip().lower())
        return self.db.scalars(stmt).first()

    def get_all_slang_entries(self) -> List[SlangDictionaryEntry]:
        stmt = select(SlangDictionaryEntry)
        return list(self.db.scalars(stmt).all())

    def create_slang_entry(
        self,
        word: str,
        translation_pt: str,
        normalized_form: Optional[str] = None,
        meaning_en: Optional[str] = None,
        meaning_pt: Optional[str] = None,
        is_slang: bool = True,
        formality_level: str = "informal",
        category: Optional[str] = None,
        region: str = "universal",
        example_sentences: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        created_by_user_id: Optional[uuid.UUID] = None
    ) -> SlangDictionaryEntry:
        entry = SlangDictionaryEntry(
            word=word.strip().lower(),
            translation_pt=translation_pt,
            normalized_form=normalized_form,
            meaning_en=meaning_en,
            meaning_pt=meaning_pt,
            is_slang=is_slang,
            formality_level=formality_level,
            category=category,
            region=region,
            embedding=embedding,
            created_by_user_id=created_by_user_id
        )
        self.db.add(entry)
        
        # Se vieram exemplos estáticos no setup, criamos na tabela de exemplos
        if example_sentences:
            for ex in example_sentences:
                entry.examples.append(WordExample(example_en=ex))
                
        self.db.commit()
        self.db.refresh(entry)
        return entry

    # Progress stats
    def get_words_added_last_n_days(self, user_id: uuid.UUID, days: int) -> List[UserWord]:
        from datetime import datetime, timedelta
        since = datetime.utcnow() - timedelta(days=days - 1)
        stmt = select(UserWord).where(
            UserWord.user_id == user_id,
            UserWord.created_at >= since
        )
        return list(self.db.scalars(stmt).all())

    # Study Sessions
    def create_study_session(self, user_id: uuid.UUID, session_type: str) -> StudySession:
        session = StudySession(
            user_id=user_id,
            session_type=session_type
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def get_study_session_by_id(self, session_id: uuid.UUID) -> Optional[StudySession]:
        stmt = select(StudySession).where(StudySession.id == session_id)
        return self.db.scalars(stmt).first()

    def update_study_session(self, session: StudySession, update_data: dict) -> StudySession:
        for key, value in update_data.items():
            if hasattr(session, key):
                setattr(session, key, value)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    # Review Attempts
    def create_review_attempt(
        self,
        session_id: Optional[uuid.UUID],
        user_word_id: uuid.UUID,
        is_correct: bool,
        user_answer: Optional[str] = None,
        quality_rating: Optional[int] = None,
        response_time_ms: Optional[int] = None
    ) -> ReviewAttempt:
        attempt = ReviewAttempt(
            session_id=session_id,
            user_word_id=user_word_id,
            is_correct=is_correct,
            user_answer=user_answer,
            quality_rating=quality_rating,
            response_time_ms=response_time_ms
        )
        self.db.add(attempt)
        self.db.commit()
        self.db.refresh(attempt)
        return attempt

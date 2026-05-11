"""
Load and validate slang context guardrail rules.

These rules are not a mock model. They are a deterministic safety policy used
before and after the sense classifier to avoid high-cost literal mistakes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RULES_PATH = Path(__file__).with_name("slang_context_rules.json")


@dataclass(frozen=True)
class ContextRuleSet:
    literal_patterns: dict[str, list[str]]
    slang_contexts: dict[str, list[str]]
    literal_regexes: dict[str, list[str]]
    slang_regexes: dict[str, list[str]]

    @property
    def ambiguous_terms(self) -> set[str]:
        return set(self.literal_patterns.keys())


def _read_string_list_map(payload: dict[str, Any], key: str) -> dict[str, list[str]]:
    raw_map = payload.get(key)
    if not isinstance(raw_map, dict):
        raise ValueError(f"{key} must be an object mapping terms to string lists.")

    result: dict[str, list[str]] = {}
    for term, values in raw_map.items():
        if not isinstance(term, str) or not term.strip():
            raise ValueError(f"{key} contains an invalid term: {term!r}")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"{key}.{term} must be a list of strings.")
        result[term.strip().lower()] = values
    return result


def load_context_rules(path: str | Path | None = None) -> ContextRuleSet:
    rules_path = Path(path) if path is not None else DEFAULT_RULES_PATH
    with rules_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"{rules_path} must contain a JSON object.")

    return ContextRuleSet(
        literal_patterns=_read_string_list_map(payload, "literal_patterns"),
        slang_contexts=_read_string_list_map(payload, "slang_contexts"),
        literal_regexes=_read_string_list_map(payload, "literal_regexes"),
        slang_regexes=_read_string_list_map(payload, "slang_regexes"),
    )

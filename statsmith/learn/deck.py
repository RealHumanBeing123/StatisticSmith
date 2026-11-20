"""Flashcard model: decks built from ``content.py`` + the live analysis registry, quiz generation, saved progress."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .content import CONCEPTS, PITFALLS, TESTS

DECK_CONCEPTS = "Key ideas (p-value, z-score, SD …)"
DECK_TESTS = "Which test do I use?"
DECK_PITFALLS = "Pitfalls & good habits"


@dataclass
class Card:
    id: str
    deck: str
    front: str                  # question / title
    back: str                   # answer (plain text, may contain newlines)
    analysis_id: str = ""       # set for 'Which test?' cards -> can open the dialog
    group: str = ""             # category (used to pick sensible quiz distractors)
    title: str = ""             # short name (test name / concept title)


def _concept_cards() -> list[Card]:
    out = []
    for cid, title, text, example, where in CONCEPTS:
        out.append(Card(f"concept:{cid}", DECK_CONCEPTS, f"What is: {title}?",
                        f"{text}\n\nExample: {example}\n\nIn Statistic Smith: {where}", title=title))
    return out


def _test_cards() -> list[Card]:
    from ..stats import REGISTRY
    out = []
    for aid, situation, when, alt in TESTS:
        spec = REGISTRY[aid]
        back = (f"Use: {spec.name}\n\nWhy / assumptions: {when}\n\nAlternatives: {alt}\n\n"
                f"Run it: Analysis ▸ {spec.category} ▸ {spec.name}")
        out.append(Card(f"test:{aid}", DECK_TESTS, situation, back, analysis_id=aid, group=spec.category, title=spec.name))
    return out


def _pitfall_cards() -> list[Card]:
    return [Card(f"pitfall:{pid}", DECK_PITFALLS, title, text, title=title) for pid, title, text in PITFALLS]


def all_cards() -> list[Card]:
    return _concept_cards() + _test_cards() + _pitfall_cards()


def decks() -> dict[str, list[Card]]:
    out: dict[str, list[Card]] = {DECK_TESTS: [], DECK_CONCEPTS: [], DECK_PITFALLS: []}
    for c in all_cards():
        out[c.deck].append(c)
    return out


def search(cards: list[Card], text: str) -> list[Card]:
    t = text.lower().strip()
    return [c for c in cards if not t or t in c.front.lower() or t in c.back.lower() or t in c.title.lower()]


# ----------------------------------------------------------------------------- quiz
@dataclass
class Question:
    card: Card
    options: list[str]          # test names, shuffled
    answer: int                 # index of the correct option


def make_quiz(n: int = 10, seed: int | None = None, deck: list[Card] | None = None) -> list[Question]:
    """'Which test fits this situation?' multiple-choice questions with 4 options. Distractors prefer the same category."""
    rnd = random.Random(seed)
    cards = [c for c in (deck or decks()[DECK_TESTS]) if c.analysis_id]
    by_group: dict[str, list[Card]] = {}
    for c in cards:
        by_group.setdefault(c.group, []).append(c)
    picks = rnd.sample(cards, min(n, len(cards)))
    out = []
    for c in picks:
        pool = [x for x in by_group[c.group] if x.title != c.title]
        rnd.shuffle(pool)
        distract = pool[:3]
        if len(distract) < 3:
            rest = [x for x in cards if x.title != c.title and x not in distract]
            rnd.shuffle(rest)
            distract += rest[: 3 - len(distract)]
        opts = [c.title] + [d.title for d in distract]
        rnd.shuffle(opts)
        out.append(Question(c, opts, opts.index(c.title)))
    return out


# ----------------------------------------------------------------------------- progress
@dataclass
class Progress:
    """Which cards the user marked 'I know this'. Stored as JSON so it survives restarts."""
    known: set[str] = field(default_factory=set)
    quiz_best: int = 0
    path: Path | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> "Progress":
        if path is None:
            from ..settings import config_dir
            path = config_dir() / "flashcards.json"
        p = cls(path=path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            valid = {c.id for c in all_cards()}
            p.known = set(raw.get("known", [])) & valid
            p.quiz_best = int(raw.get("quiz_best", 0))
        except Exception:
            pass
        return p

    def save(self):
        if self.path:
            try:
                self.path.write_text(json.dumps({"known": sorted(self.known), "quiz_best": self.quiz_best}), encoding="utf-8")
            except OSError:
                pass

    def mark(self, card_id: str, known: bool):
        (self.known.add if known else self.known.discard)(card_id)
        self.save()

    def count(self, cards: list[Card]) -> tuple[int, int]:
        return sum(1 for c in cards if c.id in self.known), len(cards)

    def reset(self, cards: list[Card] | None = None):
        ids = {c.id for c in cards} if cards else None
        self.known = set() if ids is None else self.known - ids
        self.save()

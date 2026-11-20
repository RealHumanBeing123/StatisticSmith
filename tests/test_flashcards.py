"""Stats Flashcards: content integrity, quiz logic, saved progress and the dialog."""
import pytest

from statsmith.learn import Progress, all_cards, decks, make_quiz
from statsmith.learn.content import CONCEPTS, PITFALLS, TESTS
from statsmith.stats import REGISTRY


def test_every_analysis_has_exactly_one_which_test_card():
    ids = [t[0] for t in TESTS]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(REGISTRY), (set(REGISTRY) - set(ids), set(ids) - set(REGISTRY))


def test_card_ids_unique_and_fields_nonempty():
    cards = all_cards()
    assert len({c.id for c in cards}) == len(cards)
    for c in cards:
        assert c.front.strip() and len(c.back) > 40 and c.title


def test_basic_concepts_present():
    titles = " ".join(c[1].lower() for c in CONCEPTS)
    for word in ("p-value", "z-score", "standard deviation", "confidence interval", "standard error", "degrees of freedom",
                 "type i", "power", "correlation", "normal distribution", "effect size", "null"):
        assert word in titles, word
    assert len(PITFALLS) >= 8


def test_test_cards_reference_real_menu_path():
    for c in decks()["Which test do I use?"]:
        spec = REGISTRY[c.analysis_id]
        assert f"Analysis ▸ {spec.category} ▸ {spec.name}" in c.back


def test_quiz_shape_and_determinism():
    q1, q2 = make_quiz(10, seed=3), make_quiz(10, seed=3)
    assert [x.card.id for x in q1] == [x.card.id for x in q2] and len(q1) == 10
    for q in q1:
        assert len(q.options) == 4 and len(set(q.options)) == 4
        assert q.options[q.answer] == q.card.title
    # distractors mostly come from the same category
    same = sum(1 for q in make_quiz(40, seed=1) for o in q.options if o != q.card.title and
               any(c.title == o and c.group == q.card.group for c in all_cards()))
    assert same > 60


def test_progress_persists_and_ignores_unknown_ids(tmp_path):
    p = Progress.load(tmp_path / "f.json")
    first = all_cards()[0].id
    p.mark(first, True)
    p.quiz_best = 7; p.save()
    (tmp_path / "f.json").write_text('{"known": ["%s", "gone:card"], "quiz_best": 7}' % first)
    q = Progress.load(tmp_path / "f.json")
    assert q.known == {first} and q.quiz_best == 7
    q.mark(first, False)
    assert Progress.load(tmp_path / "f.json").known == set()


# ------------------------------------------------------------------ dialog
@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_dialog_flip_navigate_mark_quiz(app, tmp_path):
    from statsmith.ui.flashcards import FlashcardDialog
    opened = []
    d = FlashcardDialog(None, on_open_analysis=opened.append, progress=Progress.load(tmp_path / "p.json"))
    d.show()
    n = len(d.queue)
    assert n == 77 and "WHICH TEST" in d.card.toHtml()
    d.flip()
    assert "Run it:" in d.card.toPlainText()
    d.mark(True)
    assert d.pos == 1 and d.progress.count(d.current_deck_cards())[0] == 1
    d.b_open.click() if d.b_open.isVisible() else None
    d.open_current()
    assert opened and opened[0] in REGISTRY
    d.deck_box.setCurrentText("Key ideas (p-value, z-score, SD …)")
    assert len(d.queue) == len(CONCEPTS)
    d.search_box.setText("z-score")
    assert any("Z-score" in d.listw.item(i).text() for i in range(d.listw.count()))
    d.show_card_by_id("test:t_welch", flipped=True)
    assert "Welch" in d.card.toPlainText()
    d.start_quiz(seed=2)
    for _ in range(10):
        d.answer(d.quiz[d.qpos].answer)
        d.next_question()
    assert d.score == 10 and d.progress.quiz_best == 10 and "10 / 10" in d.q_text.toPlainText()


def test_help_menu_entry_and_learn_button(app, tmp_path, monkeypatch):
    from statsmith.data import Engine
    from statsmith.settings import Settings
    from statsmith.ui.main_window import MainWindow
    monkeypatch.setattr(Settings, "path", classmethod(lambda cls: tmp_path / "s.json"))
    monkeypatch.setattr(Progress, "load", classmethod(lambda cls, path=None: Progress(path=tmp_path / "f.json")))
    w = MainWindow(Settings(), Engine(1, 2, workdir=str(tmp_path / "w")))
    help_menu = [a.menu() for a in w.menuBar().actions() if a.text().replace("&", "") == "Help"][0]
    acts = [a.text() for a in help_menu.actions()]
    assert "Stats Flashcards…" in acts
    w.show_flashcards("test:mann_whitney", True)
    assert "Mann" in w.flash_win.card.toPlainText()
    w.flash_win.close(); w.close()

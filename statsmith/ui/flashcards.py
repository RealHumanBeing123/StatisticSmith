"""Stats Flashcards (Help menu): learn which test to use and what p-values, z-scores, SDs ... mean."""
from __future__ import annotations

import html
import random

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QProgressBar, QPushButton, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from ..learn import Card, Progress, all_cards, decks, make_quiz
from ..learn.deck import DECK_TESTS, search

LABELS = ("Use:", "Why / assumptions:", "Alternatives:", "Run it:", "Example:", "In Statistic Smith:")


def card_html(c: Card, side: str) -> str:
    """Front = the question (big). Back = formatted answer."""
    if side == "front":
        kind = {"concept": "KEY IDEA", "test": "WHICH TEST?", "pitfall": "GOOD HABIT"}[c.id.split(":")[0]]
        return (f"<p style='font-size:9pt;color:gray;letter-spacing:2px'>{kind}</p>"
                f"<p style='font-size:17pt'><b>{html.escape(c.front)}</b></p>"
                "<p style='color:gray'>Think of your answer, then press <b>Show answer</b> (Space).</p>")
    parts = []
    for para in c.back.split("\n\n"):
        esc = html.escape(para).replace("\n", "<br>")
        for lab in LABELS:
            if esc.startswith(html.escape(lab)):
                esc = f"<b>{html.escape(lab)}</b>{esc[len(html.escape(lab)):]}"
                break
        size = "15pt" if para.startswith("Use:") else "11.5pt"
        parts.append(f"<p style='font-size:{size}'>{esc}</p>")
    return "".join(parts)


class FlashcardDialog(QDialog):
    def __init__(self, parent=None, on_open_analysis=None, progress: Progress | None = None):
        super().__init__(parent)
        self.setWindowTitle("Stats Flashcards - learn which test to use")
        self.resize(820, 640)
        self.on_open_analysis = on_open_analysis
        self.progress = progress or Progress.load()
        self.decks = decks()
        self.cards_all = all_cards()
        self.queue: list[Card] = []
        self.pos = 0
        self.flipped = False
        self.quiz: list = []
        self.qpos = 0
        self.score = 0
        self.answered = False

        self.setStyleSheet("QTextBrowser { font-family: 'Segoe UI', 'DejaVu Sans', Arial, sans-serif; font-size: 11pt; }")
        lay = QVBoxLayout(self)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)
        self._build_cards_tab()
        self._build_quiz_tab()
        self._build_browse_tab()
        self.reload_queue()

    # ======================================================================= flashcards tab
    def _build_cards_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        top = QHBoxLayout()
        top.addWidget(QLabel("Deck:"))
        self.deck_box = QComboBox()
        self.deck_box.addItem("All cards")
        self.deck_box.addItems(list(self.decks))
        self.deck_box.setCurrentText(DECK_TESTS)
        self.deck_box.currentIndexChanged.connect(self.reload_queue)
        top.addWidget(self.deck_box, 1)
        self.only_new = QCheckBox("Hide cards I know")
        self.only_new.toggled.connect(self.reload_queue)
        top.addWidget(self.only_new)
        self.shuffle = QCheckBox("Shuffle")
        self.shuffle.toggled.connect(self.reload_queue)
        top.addWidget(self.shuffle)
        reset = QPushButton("Reset progress")
        reset.clicked.connect(self._reset_progress)
        top.addWidget(reset)
        v.addLayout(top)
        self.bar = QProgressBar()
        self.bar.setFormat("learned %v of %m cards in this deck")
        v.addWidget(self.bar)
        self.card = QTextBrowser()
        self.card.setOpenLinks(False)
        self.card.setFrameShape(QTextBrowser.StyledPanel)
        v.addWidget(self.card, 1)
        self.pos_label = QLabel("")
        self.pos_label.setAlignment(Qt.AlignCenter)
        v.addWidget(self.pos_label)
        row = QHBoxLayout()
        self.b_prev = QPushButton("◀ Previous")
        self.b_prev.clicked.connect(lambda: self.step(-1))
        self.b_flip = QPushButton("Show answer")
        self.b_flip.clicked.connect(self.flip)
        self.b_next = QPushButton("Next ▶")
        self.b_next.clicked.connect(lambda: self.step(1))
        for b in (self.b_prev, self.b_flip, self.b_next):
            row.addWidget(b)
        v.addLayout(row)
        row2 = QHBoxLayout()
        self.b_know = QPushButton("✓ I know this")
        self.b_know.clicked.connect(lambda: self.mark(True))
        self.b_learn = QPushButton("↻ Still learning")
        self.b_learn.clicked.connect(lambda: self.mark(False))
        self.b_open = QPushButton("Open this test in the Analysis menu…")
        self.b_open.clicked.connect(self.open_current)
        row2.addWidget(self.b_know)
        row2.addWidget(self.b_learn)
        row2.addStretch(1)
        row2.addWidget(self.b_open)
        v.addLayout(row2)
        hint = QLabel("Keys: Space = flip · ← → = previous / next · K = I know this · L = still learning")
        hint.setStyleSheet("color: gray;")
        v.addWidget(hint)
        self.tabs.addTab(w, "Flashcards")

    def current_deck_cards(self) -> list[Card]:
        t = self.deck_box.currentText()
        return list(self.cards_all) if t == "All cards" else list(self.decks[t])

    def reload_queue(self, *_, keep: str | None = None):
        cards = self.current_deck_cards()
        n_known, n = self.progress.count(cards)
        self.bar.setMaximum(max(n, 1))
        self.bar.setValue(n_known)
        if self.only_new.isChecked():
            cards = [c for c in cards if c.id not in self.progress.known]
        if self.shuffle.isChecked():
            random.shuffle(cards)
        self.queue = cards
        self.pos = 0
        if keep:
            for i, c in enumerate(cards):
                if c.id == keep:
                    self.pos = i
        self.flipped = False
        self.show_card()

    def cur(self) -> Card | None:
        return self.queue[self.pos] if self.queue else None

    def show_card(self):
        c = self.cur()
        has = c is not None
        for b in (self.b_prev, self.b_flip, self.b_next, self.b_know, self.b_learn):
            b.setEnabled(has)
        if not has:
            self.card.setHtml("<h3>🎉 Nothing left here</h3><p>You have marked every card in this deck as known. "
                              "Untick <b>Hide cards I know</b> to review them again.</p>")
            self.pos_label.setText("")
            self.b_open.setVisible(False)
            return
        self.card.setHtml(card_html(c, "back" if self.flipped else "front"))
        self.b_flip.setText("Show question" if self.flipped else "Show answer")
        known = c.id in self.progress.known
        self.pos_label.setText(f"Card {self.pos + 1} of {len(self.queue)}" + ("   ·   ✓ known" if known else ""))
        self.b_open.setVisible(bool(c.analysis_id and self.on_open_analysis))

    def flip(self):
        if self.cur():
            self.flipped = not self.flipped
            self.show_card()

    def step(self, d: int):
        if self.queue:
            self.pos = (self.pos + d) % len(self.queue)
            self.flipped = False
            self.show_card()

    def mark(self, known: bool):
        c = self.cur()
        if not c:
            return
        self.progress.mark(c.id, known)
        cards = self.current_deck_cards()
        self.bar.setValue(self.progress.count(cards)[0])
        if self.only_new.isChecked() and known:
            self.reload_queue()
        else:
            self.step(1)

    def _reset_progress(self):
        self.progress.reset(self.current_deck_cards() if self.deck_box.currentText() != "All cards" else None)
        self.reload_queue()

    def open_current(self):
        c = self.cur()
        if c and c.analysis_id and self.on_open_analysis:
            self.on_open_analysis(c.analysis_id)

    def show_card_by_id(self, card_id: str, flipped: bool = False):
        """Jump to a card (used by the 'Learn when to use this test' button in the Analysis dialogs)."""
        self.tabs.setCurrentIndex(0)
        self.deck_box.blockSignals(True)
        self.deck_box.setCurrentText("All cards")
        self.deck_box.blockSignals(False)
        self.only_new.setChecked(False)
        self.reload_queue(keep=card_id)
        self.flipped = flipped
        self.show_card()

    def keyPressEvent(self, e):
        if self.tabs.currentIndex() == 0 and not isinstance(self.focusWidget(), QLineEdit):
            k = e.key()
            if k == Qt.Key_Space:
                self.flip(); return
            if k == Qt.Key_Right:
                self.step(1); return
            if k == Qt.Key_Left:
                self.step(-1); return
            if k == Qt.Key_K:
                self.mark(True); return
            if k == Qt.Key_L:
                self.mark(False); return
        super().keyPressEvent(e)

    # ======================================================================= quiz tab
    def _build_quiz_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.q_title = QLabel("<b>Quiz: which test fits this situation?</b>")
        v.addWidget(self.q_title)
        self.q_text = QTextBrowser()
        v.addWidget(self.q_text, 1)
        self.q_buttons = []
        for i in range(4):
            b = QPushButton("")
            b.setStyleSheet("text-align:left; padding:8px;")
            b.clicked.connect(lambda _=False, i=i: self.answer(i))
            v.addWidget(b)
            self.q_buttons.append(b)
        self.q_feedback = QTextBrowser()
        self.q_feedback.setMaximumHeight(170)
        v.addWidget(self.q_feedback)
        row = QHBoxLayout()
        self.q_start = QPushButton("Start a new 10-question quiz")
        self.q_start.clicked.connect(self.start_quiz)
        self.q_next = QPushButton("Next question ▶")
        self.q_next.clicked.connect(self.next_question)
        self.q_score = QLabel("")
        row.addWidget(self.q_start)
        row.addWidget(self.q_next)
        row.addStretch(1)
        row.addWidget(self.q_score)
        v.addLayout(row)
        self.tabs.addTab(w, "Quiz")
        self._quiz_idle()

    def _quiz_idle(self):
        self.q_text.setHtml("<p style='font-size:13pt'>Test yourself: you will see a data situation and pick the right test from four options.</p>"
                            f"<p>Best score so far: <b>{self.progress.quiz_best}/10</b></p>")
        for b in self.q_buttons:
            b.setVisible(False)
        self.q_feedback.clear()
        self.q_next.setVisible(False)
        self.q_score.setText("")

    def start_quiz(self, seed: int | None = None):
        self.quiz = make_quiz(10, seed=seed if isinstance(seed, int) else None)
        self.qpos, self.score = 0, 0
        self.show_question()

    def show_question(self):
        q = self.quiz[self.qpos]
        self.answered = False
        self.q_text.setHtml(f"<p style='color:gray'>Question {self.qpos + 1} of {len(self.quiz)}</p>"
                            f"<p style='font-size:14pt'><b>{html.escape(q.card.front)}</b></p>")
        for b, opt in zip(self.q_buttons, q.options):
            b.setText(opt)
            b.setVisible(True)
            b.setEnabled(True)
            b.setStyleSheet("text-align:left; padding:8px;")
        self.q_feedback.clear()
        self.q_next.setVisible(False)
        self.q_score.setText(f"Score: {self.score}")

    def answer(self, i: int):
        if self.answered:
            return
        self.answered = True
        q = self.quiz[self.qpos]
        ok = i == q.answer
        self.score += ok
        for j, b in enumerate(self.q_buttons):
            b.setEnabled(False)
            if j == q.answer:
                b.setStyleSheet("text-align:left; padding:8px; background:#2e8b57; color:white;")
            elif j == i:
                b.setStyleSheet("text-align:left; padding:8px; background:#b03a2e; color:white;")
        head = "✅ Correct!" if ok else f"❌ Not quite - the answer is <b>{html.escape(q.options[q.answer])}</b>."
        self.q_feedback.setHtml(f"<p><b>{head}</b></p>{card_html(q.card, 'back')}")
        self.q_score.setText(f"Score: {self.score}")
        last = self.qpos == len(self.quiz) - 1
        self.q_next.setText("See result" if last else "Next question ▶")
        self.q_next.setVisible(True)
        if ok:
            self.progress.known.add(q.card.id)
        self.progress.save()

    def next_question(self):
        if self.qpos >= len(self.quiz) - 1:
            n = len(self.quiz)
            new_best = self.score > self.progress.quiz_best
            if new_best:
                self.progress.quiz_best = self.score
                self.progress.save()
            self.q_text.setHtml(f"<h2>Quiz finished: {self.score} / {n}</h2>"
                                + ("<p>🏆 New best score!</p>" if new_best else "")
                                + "<p>Cards you answered correctly are now marked as known in the flashcards. "
                                  "Press <b>Start a new quiz</b> to try different questions.</p>")
            for b in self.q_buttons:
                b.setVisible(False)
            self.q_feedback.clear()
            self.q_next.setVisible(False)
            self.reload_queue()
            return
        self.qpos += 1
        self.show_question()

    # ======================================================================= browse tab
    def _build_browse_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search all cards: 'p-value', 'paired', 'normality', 'proportion' …")
        self.search_box.textChanged.connect(self._fill_list)
        v.addWidget(self.search_box)
        row = QHBoxLayout()
        self.listw = QListWidget()
        self.listw.setMaximumWidth(300)
        self.listw.currentItemChanged.connect(self._browse_select)
        row.addWidget(self.listw)
        self.detail = QTextBrowser()
        row.addWidget(self.detail, 1)
        v.addLayout(row, 1)
        self.tabs.addTab(w, "Browse && search")
        self._fill_list()

    def _fill_list(self):
        self.listw.clear()
        for c in search(self.cards_all, self.search_box.text()):
            it = QListWidgetItem(("✓ " if c.id in self.progress.known else "") + (c.title or c.front)[:60])
            it.setData(Qt.UserRole, c.id)
            self.listw.addItem(it)
        if self.listw.count():
            self.listw.setCurrentRow(0)
        else:
            self.detail.setHtml("<p>No cards match.</p>")

    def _browse_select(self, item):
        if not item:
            return
        c = next(c for c in self.cards_all if c.id == item.data(Qt.UserRole))
        self.detail.setHtml(f"<h3>{html.escape(c.front)}</h3>" + card_html(c, "back"))

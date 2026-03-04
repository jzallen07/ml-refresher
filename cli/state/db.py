from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_DIR = Path.home() / ".ml-refresher"
DB_PATH = DB_DIR / "state.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    category TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'novice',
    total_interactions INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS learning_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    score REAL,
    timestamp TEXT NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS fsrs_cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id TEXT NOT NULL,
    concept TEXT NOT NULL,
    card_json TEXT NOT NULL,
    due_date TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'new',
    FOREIGN KEY (topic_id) REFERENCES topics(id),
    UNIQUE(topic_id, concept)
);
"""


class StateDB:
    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # -- Topics --

    def ensure_topic(self, topic_id: str, display_name: str, category: str):
        self.conn.execute(
            "INSERT OR IGNORE INTO topics (id, display_name, category) VALUES (?, ?, ?)",
            (topic_id, display_name, category),
        )
        self.conn.commit()

    def get_topic(self, topic_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
        return dict(row) if row else None

    def get_all_topics(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM topics ORDER BY category, id").fetchall()
        return [dict(r) for r in rows]

    def update_topic_level(self, topic_id: str, level: str):
        self.conn.execute("UPDATE topics SET level = ? WHERE id = ?", (level, topic_id))
        self.conn.commit()

    def increment_interactions(self, topic_id: str):
        self.conn.execute(
            "UPDATE topics SET total_interactions = total_interactions + 1 WHERE id = ?",
            (topic_id,),
        )
        self.conn.commit()

    # -- Learning Events --

    def insert_event(
        self,
        topic_id: str,
        event_type: str,
        score: float | None = None,
        metadata: dict | None = None,
    ):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO learning_events (topic_id, event_type, score, timestamp, metadata_json) VALUES (?, ?, ?, ?, ?)",
            (topic_id, event_type, score, now, json.dumps(metadata) if metadata else None),
        )
        self.conn.commit()

    def get_events(self, topic_id: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM learning_events WHERE topic_id = ? ORDER BY timestamp DESC LIMIT ?",
            (topic_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- FSRS Cards --

    def upsert_card(self, topic_id: str, concept: str, card_json: str, due_date: str, state: str):
        self.conn.execute(
            """INSERT INTO fsrs_cards (topic_id, concept, card_json, due_date, state)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(topic_id, concept)
               DO UPDATE SET card_json = excluded.card_json,
                             due_date = excluded.due_date,
                             state = excluded.state""",
            (topic_id, concept, card_json, due_date, state),
        )
        self.conn.commit()

    def get_due_cards(self, topic_id: str | None = None, limit: int = 20) -> list[dict]:
        now = datetime.now(timezone.utc).isoformat()
        if topic_id:
            rows = self.conn.execute(
                "SELECT * FROM fsrs_cards WHERE topic_id = ? AND due_date <= ? ORDER BY due_date ASC LIMIT ?",
                (topic_id, now, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM fsrs_cards WHERE due_date <= ? ORDER BY due_date ASC LIMIT ?",
                (now, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_cards(self, topic_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM fsrs_cards WHERE topic_id = ? ORDER BY due_date ASC",
            (topic_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_topic_mastery(self, topic_id: str) -> dict:
        """Aggregate mastery for a topic: avg stability, % mastered, weakest concepts."""
        cards = self.get_all_cards(topic_id)
        if not cards:
            return {"topic_id": topic_id, "card_count": 0, "avg_stability": 0.0, "pct_mastered": 0.0, "weakest": []}

        stabilities = []
        mastered = 0
        weakest = []
        for card in cards:
            card_data = json.loads(card["card_json"])
            stability = card_data.get("stability")
            if stability is not None:
                stabilities.append(stability)
                if stability >= 2.0:
                    mastered += 1
                else:
                    weakest.append({"concept": card["concept"], "stability": stability})
            else:
                weakest.append({"concept": card["concept"], "stability": 0.0})

        avg_stability = sum(stabilities) / len(stabilities) if stabilities else 0.0
        pct_mastered = mastered / len(cards) if cards else 0.0
        weakest.sort(key=lambda x: x["stability"])

        return {
            "topic_id": topic_id,
            "card_count": len(cards),
            "avg_stability": round(avg_stability, 2),
            "pct_mastered": round(pct_mastered, 2),
            "weakest": weakest[:3],
        }

    def get_card(self, topic_id: str, concept: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM fsrs_cards WHERE topic_id = ? AND concept = ?",
            (topic_id, concept),
        ).fetchone()
        return dict(row) if row else None

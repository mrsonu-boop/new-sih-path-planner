"""SQLite persistence for obstacle encounter events (backend/events.db)."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Dict, List

DEFAULT_DB = Path(__file__).resolve().parent / "events.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS obstacle_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id TEXT UNIQUE NOT NULL,
    label TEXT,
    name TEXT,
    entered_at REAL,
    cleared_at REAL,
    duration_s REAL,
    min_distance_m REAL,
    steering_deg REAL,
    speed_kmh REAL,
    action TEXT,
    status TEXT
);
"""

FIELDS = ("track_id", "label", "name", "entered_at", "cleared_at", "duration_s",
          "min_distance_m", "steering_deg", "speed_kmh", "action", "status")


class EventLogger:
    """Thread-safe minimal writer/reader for obstacle encounter records."""

    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def log_event(self, rec: Dict) -> bool:
        missing = [f for f in FIELDS if f not in rec]
        if missing:
            raise ValueError(f"missing fields: {missing}")
        with self._lock, self._connect() as con:
            cur = con.execute(
                f"INSERT OR REPLACE INTO obstacle_events ({', '.join(FIELDS)}) "
                f"VALUES ({', '.join(':' + f for f in FIELDS)})",
                {f: rec[f] for f in FIELDS},
            )
            return cur.rowcount > 0

    def recent(self, limit: int = 200) -> List[Dict]:
        limit = max(1, min(int(limit), 1000))
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT * FROM obstacle_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._lock, self._connect() as con:
            (n,) = con.execute("SELECT COUNT(*) FROM obstacle_events").fetchone()
        return int(n)


if __name__ == "__main__":
    logger = EventLogger()
    print("db:", logger.db_path, "| events:", logger.count())

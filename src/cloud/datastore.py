"""
datastore.py
------------
SQLite-backed store that MIRRORS AWS DynamoDB (and Timestream for history).

AWS mapping:
  - put_event(...)       ~  DynamoDB PutItem   (anomaly events table)
  - upsert_health(...)   ~  DynamoDB PutItem   (device health / shadow)
  - recent_events(...)   ~  DynamoDB Query / Timestream query

Why SQLite: zero setup, file-based, and the method surface (put / query by
device) maps cleanly onto DynamoDB operations. Swapping to boto3 + DynamoDB
later means reimplementing these few methods — callers don't change.

Tables:
  anomaly_events(device_id, t, vibration, temperature, anomaly_score,
                 severity, received_at)
  device_health(device_id PK, total_anomalies, last_severity, last_seen)
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any

DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "pipeline.db",
)


class DataStore:
    def __init__(self, db_path: str = DEFAULT_DB) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS anomaly_events (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id     TEXT NOT NULL,
                t             REAL NOT NULL,
                vibration     REAL,
                temperature   REAL,
                anomaly_score REAL,
                severity      TEXT,
                received_at   REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS device_health (
                device_id       TEXT PRIMARY KEY,
                total_anomalies INTEGER NOT NULL DEFAULT 0,
                last_severity   TEXT,
                last_seen       REAL
            );
            """
        )
        self._conn.commit()

    def put_event(self, device_id: str, event: dict[str, Any]) -> None:
        """Insert an anomaly event (mirrors DynamoDB PutItem)."""
        self._conn.execute(
            """INSERT INTO anomaly_events
               (device_id, t, vibration, temperature, anomaly_score, severity, received_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                device_id,
                event.get("t"),
                event.get("vibration"),
                event.get("temperature"),
                event.get("anomaly_score"),
                event.get("severity"),
                time.time(),
            ),
        )
        self._conn.commit()

    def upsert_health(self, device_id: str, severity: str) -> None:
        """Update rolling device-health summary (mirrors a DynamoDB shadow item)."""
        self._conn.execute(
            """INSERT INTO device_health (device_id, total_anomalies, last_severity, last_seen)
               VALUES (?, 1, ?, ?)
               ON CONFLICT(device_id) DO UPDATE SET
                 total_anomalies = total_anomalies + 1,
                 last_severity   = excluded.last_severity,
                 last_seen       = excluded.last_seen""",
            (device_id, severity, time.time()),
        )
        self._conn.commit()

    def recent_events(self, device_id: str | None = None, limit: int = 10) -> list[dict]:
        """Query most recent anomaly events (mirrors DynamoDB Query / Timestream)."""
        if device_id:
            rows = self._conn.execute(
                "SELECT * FROM anomaly_events WHERE device_id = ? ORDER BY id DESC LIMIT ?",
                (device_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM anomaly_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def health(self, device_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM device_health WHERE device_id = ?", (device_id,)
        ).fetchone()
        return dict(row) if row else None

    def count_events(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS c FROM anomaly_events").fetchone()["c"]

    def close(self) -> None:
        self._conn.close()

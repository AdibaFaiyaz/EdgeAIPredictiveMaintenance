"""
telemetry.py
------------
Thread-safe, in-memory telemetry buffer that the dashboard reads from.

The pipeline runs in a background thread and pushes updates here; the HTTP
server reads a consistent snapshot to serve the browser. A lock guards all
access so the reader never sees a half-written state.

This is a dashboard-side concern only — it does not change the edge/cloud
pipeline semantics. Conceptually it mirrors querying DynamoDB/Timestream plus
a live device shadow, but kept in memory for a responsive live view.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class Telemetry:
    def __init__(self, max_points: int = 120, max_anomalies: int = 50) -> None:
        self._lock = threading.Lock()
        # Rolling window of recent raw readings (for the live charts).
        self._readings: deque[dict] = deque(maxlen=max_points)
        # Rolling window of recent anomaly events (for the alert feed).
        self._anomalies: deque[dict] = deque(maxlen=max_anomalies)
        # Pipeline stage counters (mirror AWS stages).
        self.counters = {
            "samples": 0,       # edge: samples processed on-device
            "published": 0,     # IoT Core: anomalies published
            "processed": 0,     # Lambda: messages processed
            "stored": 0,        # DynamoDB: events stored
            "alerts": 0,        # SNS: alerts sent
        }
        self.device_id = "motor-01"
        self.status = "healthy"      # healthy | anomaly
        self.started_at = time.time()
        self.running = True

    # ---- writers (called from the pipeline thread) ----

    def add_reading(self, reading: dict, is_anomaly: bool) -> None:
        with self._lock:
            self._readings.append({
                "t": reading["t"],
                "vibration": reading["vibration"],
                "temperature": reading["temperature"],
                "anomaly": is_anomaly,
            })
            self.counters["samples"] += 1
            self.status = "anomaly" if is_anomaly else "healthy"

    def add_anomaly(self, event: dict) -> None:
        with self._lock:
            self._anomalies.appendleft(event)
            self.counters["published"] += 1

    def mark_processed(self) -> None:
        with self._lock:
            self.counters["processed"] += 1

    def mark_stored(self) -> None:
        with self._lock:
            self.counters["stored"] += 1

    def mark_alert(self) -> None:
        with self._lock:
            self.counters["alerts"] += 1

    def stop(self) -> None:
        with self._lock:
            self.running = False

    # ---- reader (called from the HTTP thread) ----

    def snapshot(self) -> dict:
        """Return a consistent, JSON-serializable copy of current state."""
        with self._lock:
            samples = self.counters["samples"]
            published = self.counters["published"]
            reduction = 1.0 - (published / samples) if samples else 1.0
            return {
                "device_id": self.device_id,
                "status": self.status,
                "running": self.running,
                "uptime_s": round(time.time() - self.started_at, 1),
                "counters": dict(self.counters),
                "bandwidth_reduction": round(reduction, 4),
                "readings": list(self._readings),
                "anomalies": list(self._anomalies)[:20],
            }

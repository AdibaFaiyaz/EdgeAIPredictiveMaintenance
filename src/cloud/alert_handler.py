"""
alert_handler.py
----------------
Alert dispatcher that MIRRORS AWS SNS.

AWS mapping:
  - publish_alert(subject, message)  ~  SNS Publish to a topic
  - subscribers / log sink           ~  SNS subscriptions (email/SMS/HTTP)

Phase 2 (local): alerts are printed/logged. The interface mirrors SNS so that
migrating later means calling boto3 sns.publish() inside publish_alert()
instead of printing — callers are unchanged.
"""

from __future__ import annotations

import time


class AlertHandler:
    def __init__(self, min_severity: str = "high") -> None:
        # Only fire notifications at/above this severity (mirrors SNS filtering).
        self._rank = {"medium": 1, "high": 2}
        self.min_rank = self._rank.get(min_severity, 2)
        self.alerts_sent = 0

    def publish_alert(self, device_id: str, event: dict) -> bool:
        """Send an alert if the event is severe enough (mirrors SNS Publish)."""
        severity = event.get("severity", "medium")
        if self._rank.get(severity, 0) < self.min_rank:
            return False

        self.alerts_sent += 1
        ts = time.strftime("%H:%M:%S")
        subject = f"[ALERT] {device_id}: {severity.upper()} anomaly"
        message = (
            f"vibration={event.get('vibration')} "
            f"temperature={event.get('temperature')} "
            f"score={event.get('anomaly_score')} "
            f"(device time t={event.get('t')}s)"
        )
        # Phase 2: print/log. Phase 3+: boto3 sns.publish(TopicArn=..., ...)
        print(f"  🔔 {ts}  {subject} | {message}")
        return True

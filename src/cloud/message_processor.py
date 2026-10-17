"""
message_processor.py
--------------------
Message processor that MIRRORS AWS Lambda.

AWS mapping:
  - handle(topic, payload)   ~  Lambda handler(event, context) triggered by
                                an IoT Rule on each incoming MQTT message
  - DataStore writes         ~  Lambda writing to DynamoDB
  - AlertHandler.publish     ~  Lambda publishing to SNS

The handler is intentionally shaped like a Lambda entrypoint: it receives one
event, does its work (validate -> store -> alert), and returns a small result
dict. Migrating later means pasting this logic into a Lambda function.
"""

from __future__ import annotations

from .datastore import DataStore
from .alert_handler import AlertHandler


class MessageProcessor:
    def __init__(self, store: DataStore, alerts: AlertHandler) -> None:
        self.store = store
        self.alerts = alerts
        self.processed = 0

    def handle(self, topic: str, payload: dict) -> dict:
        """
        Process one anomaly message (mirrors a Lambda invocation).

        Expected topic: devices/<device_id>/anomalies
        Expected payload: the anomaly event dict from the edge agent.
        """
        self.processed += 1

        # Derive device_id from the topic (mirrors parsing an IoT topic).
        parts = topic.split("/")
        device_id = parts[1] if len(parts) >= 2 else "unknown"

        # Basic validation (a Lambda would guard against malformed events).
        required = {"t", "vibration", "temperature", "severity"}
        if not required.issubset(payload):
            return {"status": "rejected", "reason": "missing fields", "topic": topic}

        # 1. Persist (DynamoDB PutItem).
        self.store.put_event(device_id, payload)
        # 2. Update rolling device health (DynamoDB shadow item).
        self.store.upsert_health(device_id, payload["severity"])
        # 3. Fire alert if severe (SNS Publish).
        alerted = self.alerts.publish_alert(device_id, payload)

        return {"status": "ok", "device_id": device_id, "alerted": alerted}

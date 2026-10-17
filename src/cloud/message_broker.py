"""
message_broker.py
-----------------
In-process pub/sub broker that MIRRORS AWS IoT Core.

AWS mapping:
  - publish(topic, payload)      ~  IoT Core MQTT publish
  - subscribe(topic, callback)   ~  IoT Rule routing a topic to a target
  - topic strings                ~  MQTT topics (e.g. "devices/motor-01/anomalies")

Why in-process: it lets us build and test the ENTIRE pipeline with zero
infrastructure. The public surface (publish / subscribe + topic strings) is
deliberately MQTT-shaped, so migrating to real AWS IoT Core later means
swapping this class for an MQTT client — the rest of the pipeline is unchanged.

Supports simple single-level '+' and multi-level '#' wildcard subscriptions,
like MQTT, so routing behaves similarly to IoT Core topic filters.
"""

from __future__ import annotations

from typing import Callable


Callback = Callable[[str, dict], None]


def _topic_matches(filter_: str, topic: str) -> bool:
    """MQTT-style topic matching with '+' (single level) and '#' (multi level)."""
    f_parts = filter_.split("/")
    t_parts = topic.split("/")
    for i, fp in enumerate(f_parts):
        if fp == "#":
            return True
        if i >= len(t_parts):
            return False
        if fp == "+":
            continue
        if fp != t_parts[i]:
            return False
    return len(f_parts) == len(t_parts)


class MessageBroker:
    """A minimal synchronous pub/sub broker mirroring AWS IoT Core."""

    def __init__(self) -> None:
        self._subscriptions: list[tuple[str, Callback]] = []
        self.published_count = 0

    def subscribe(self, topic_filter: str, callback: Callback) -> None:
        """Register a callback for a topic filter (mirrors an IoT Rule)."""
        self._subscriptions.append((topic_filter, callback))

    def publish(self, topic: str, payload: dict) -> int:
        """
        Publish a message to a topic (mirrors IoT Core MQTT publish).
        Returns the number of subscribers that received it.
        Delivery is synchronous and in-process.
        """
        self.published_count += 1
        delivered = 0
        for topic_filter, callback in self._subscriptions:
            if _topic_matches(topic_filter, topic):
                callback(topic, payload)
                delivered += 1
        return delivered

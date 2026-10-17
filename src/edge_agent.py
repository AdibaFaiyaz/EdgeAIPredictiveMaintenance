"""
edge_agent.py
-------------
Simulates the EDGE DEVICE in the predictive-maintenance pipeline.

Responsibilities (same as it would have on a real MCU/Raspberry Pi):
  1. Read the sensor stream (here: the simulator).
  2. Maintain a sliding window and run the anomaly model locally ("on-device").
  3. Decide what to do:
       - healthy  -> keep local, do NOT send to cloud (saves bandwidth)
       - anomaly  -> this is what WOULD be published to AWS IoT Core (Phase 2)
  4. Print a live view and a bandwidth-savings summary.

This is the heart of the "edge AI" value: inference happens at the edge, and
only meaningful events leave the device. In Phase 2, the `_publish()` method
becomes a real MQTT publish to AWS IoT Core.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sensor_simulator import MotorSensorSimulator  # noqa: E402
from train_model import extract_features  # noqa: E402

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "anomaly_model.pkl"
)


class EdgeAgent:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        broker=None,
        device_id: str = "motor-01",
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run: python src/train_model.py"
            )
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.window_size = bundle["window_size"]
        self._window: deque[list[float]] = deque(maxlen=self.window_size)

        # Phase 2: optional message broker (mirrors AWS IoT Core).
        self.broker = broker
        self.device_id = device_id
        self.topic = f"devices/{device_id}/anomalies"

        # Metrics for the bandwidth-savings story.
        self.total_samples = 0
        self.sent_events = 0

    def _publish(self, event: dict) -> None:
        """
        Publish an anomaly event to the cloud.

        Phase 1 (no broker): just records that this event WOULD be sent.
        Phase 2 (broker set): publishes to the in-process broker that mirrors
          AWS IoT Core. In the real-AWS migration this becomes an MQTT publish.
        """
        self.sent_events += 1
        if self.broker is not None:
            self.broker.publish(self.topic, event)

    def process(self, reading) -> dict | None:
        """Feed one reading through the model. Returns an event if anomalous."""
        self.total_samples += 1
        self._window.append([reading.vibration, reading.temperature])

        # Need a full window before we can score.
        if len(self._window) < self.window_size:
            return None

        feats = extract_features(np.array(self._window)).reshape(1, -1)
        feats_scaled = self.scaler.transform(feats)
        pred = self.model.predict(feats_scaled)[0]          # -1 anomaly, 1 normal
        score = float(self.model.score_samples(feats_scaled)[0])  # lower = more anomalous

        if pred == -1:
            event = {
                "t": reading.t,
                "vibration": reading.vibration,
                "temperature": reading.temperature,
                "anomaly_score": round(score, 4),
                "severity": "high" if score < -0.15 else "medium",
            }
            self._publish(event)
            return event
        return None

    def summary(self) -> dict:
        sent = self.sent_events
        total = max(1, self.total_samples)
        reduction = 1.0 - (sent / total)
        return {
            "total_samples": self.total_samples,
            "events_sent_to_cloud": sent,
            "bandwidth_reduction": f"{reduction:.1%}",
        }


def run(fault_mode: str, severity: float, n: int, delay: float) -> None:
    agent = EdgeAgent()
    sim = MotorSensorSimulator(fault_mode=fault_mode, fault_severity=severity, seed=7)

    print(f"Edge agent started | fault_mode={fault_mode} severity={severity}")
    print("-" * 64)
    first_anomaly_t = None

    for reading in sim.stream(n):
        event = agent.process(reading)
        if event:
            if first_anomaly_t is None:
                first_anomaly_t = event["t"]
            print(f"  [ANOMALY] t={event['t']:>6}s  vib={event['vibration']:>6}  "
                  f"temp={event['temperature']:>6}  score={event['anomaly_score']:>7}  "
                  f"severity={event['severity']}  --> would publish to AWS")
        if delay:
            time.sleep(delay)

    print("-" * 64)
    s = agent.summary()
    print(f"Samples processed on-device : {s['total_samples']}")
    print(f"Events sent to cloud        : {s['events_sent_to_cloud']}")
    print(f"Bandwidth reduction         : {s['bandwidth_reduction']}")
    if first_anomaly_t is not None:
        print(f"First anomaly detected at   : t={first_anomaly_t}s")
    else:
        print("No anomalies detected (healthy run).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Edge AI predictive-maintenance agent")
    p.add_argument("--fault", default="combined",
                   choices=["none", "bearing", "overheat", "spike", "combined"],
                   help="fault to inject into the simulated motor")
    p.add_argument("--severity", type=float, default=0.8, help="fault severity 0..1")
    p.add_argument("--samples", type=int, default=600, help="number of samples to stream")
    p.add_argument("--delay", type=float, default=0.0,
                   help="seconds between samples (use 0.05 for a live feel)")
    args = p.parse_args()
    run(args.fault, args.severity, args.samples, args.delay)

"""
run_pipeline.py
---------------
Phase 2 end-to-end runner: wires the edge agent to the in-process cloud
pipeline that mirrors AWS.

Flow (each stage maps 1:1 to an AWS service):

  MotorSensorSimulator            (simulated sensor / hardware later)
        │
        ▼
  EdgeAgent  --_publish-->  MessageBroker        ~ AWS IoT Core
                                 │ (IoT Rule)
                                 ▼
                           MessageProcessor       ~ AWS Lambda
                              │          │
                              ▼          ▼
                         DataStore   AlertHandler  ~ DynamoDB + SNS

Run:
  python src/run_pipeline.py --fault combined --severity 0.8 --samples 400
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sensor_simulator import MotorSensorSimulator      # noqa: E402
from edge_agent import EdgeAgent                        # noqa: E402
from cloud.message_broker import MessageBroker          # noqa: E402
from cloud.message_processor import MessageProcessor    # noqa: E402
from cloud.datastore import DataStore                   # noqa: E402
from cloud.alert_handler import AlertHandler            # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 end-to-end pipeline")
    p.add_argument("--fault", default="combined",
                   choices=["none", "bearing", "overheat", "spike", "combined"])
    p.add_argument("--severity", type=float, default=0.8)
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--device-id", default="motor-01")
    p.add_argument("--db", default=None, help="override SQLite path (default data/pipeline.db)")
    args = p.parse_args()

    # --- Build the cloud side (mirrors AWS resources) ---
    store = DataStore(args.db) if args.db else DataStore()
    alerts = AlertHandler(min_severity="high")
    processor = MessageProcessor(store, alerts)
    broker = MessageBroker()
    # Wire the IoT Rule: route anomaly topics to the Lambda-equivalent processor.
    broker.subscribe("devices/+/anomalies", processor.handle)

    # --- Build the edge side, pointed at the broker ---
    agent = EdgeAgent(broker=broker, device_id=args.device_id)
    sim = MotorSensorSimulator(fault_mode=args.fault, fault_severity=args.severity, seed=7)

    print(f"Pipeline started | device={args.device_id} fault={args.fault} "
          f"severity={args.severity}")
    print(f"Topic: {agent.topic}")
    print("-" * 70)

    for reading in sim.stream(args.samples):
        agent.process(reading)  # anomalies auto-publish through the broker

    print("-" * 70)
    s = agent.summary()
    print("EDGE  | samples:", s["total_samples"],
          "| published:", s["events_sent_to_cloud"],
          "| bandwidth reduction:", s["bandwidth_reduction"])
    print("IOT   | broker messages published:", broker.published_count)
    print("LAMBDA| messages processed:", processor.processed)
    print("DDB   | anomaly events stored:", store.count_events())
    print("SNS   | alerts sent:", alerts.alerts_sent)

    health = store.health(args.device_id)
    if health:
        print(f"HEALTH| {args.device_id}: total_anomalies={health['total_anomalies']} "
              f"last_severity={health['last_severity']}")

    print("\nMost recent stored events:")
    for ev in store.recent_events(limit=3):
        print(f"  - t={ev['t']}s vib={ev['vibration']} temp={ev['temperature']} "
              f"severity={ev['severity']} score={ev['anomaly_score']}")

    store.close()


if __name__ == "__main__":
    main()

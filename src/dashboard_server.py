"""
dashboard_server.py
-------------------
Phase 3: a live web dashboard for the edge->cloud pipeline.

Standard-library only (http.server + threading) — no extra dependencies, in
keeping with the lightweight edge mindset.

Architecture:
  - A background thread runs the pipeline continuously (sensor -> edge ->
    broker -> processor -> store -> alert), pushing updates into a shared
    Telemetry buffer.
  - An HTTP server serves:
      GET /            -> the single-page dashboard (static HTML/CSS/JS)
      GET /api/state   -> JSON snapshot the page polls for live updates
      GET /api/health  -> simple liveness probe

Run:
  python src/dashboard_server.py            # then open http://localhost:8000
  python src/dashboard_server.py --port 8080 --cycle-faults
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sensor_simulator import MotorSensorSimulator      # noqa: E402
from edge_agent import EdgeAgent                        # noqa: E402
from cloud.message_broker import MessageBroker          # noqa: E402
from cloud.message_processor import MessageProcessor    # noqa: E402
from cloud.datastore import DataStore                   # noqa: E402
from cloud.alert_handler import AlertHandler            # noqa: E402
from cloud.telemetry import Telemetry                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "cloud", "static")

# Shared telemetry read by the HTTP handler, written by the pipeline thread.
TELEMETRY = Telemetry()


def pipeline_loop(cycle_faults: bool, sample_delay: float) -> None:
    """
    Continuously stream sensor data through the full pipeline, updating
    telemetry. Runs in a daemon thread for the lifetime of the server.
    """
    store = DataStore(os.path.join(HERE, "..", "data", "dashboard.db"))
    alerts = AlertHandler(min_severity="high")
    processor = MessageProcessor(store, alerts)
    broker = MessageBroker()

    # Wrap the processor so we can mirror each stage into telemetry.
    def routed(topic: str, payload: dict) -> None:
        TELEMETRY.mark_processed()
        result = processor.handle(topic, payload)
        if result.get("status") == "ok":
            TELEMETRY.mark_stored()
            if result.get("alerted"):
                TELEMETRY.mark_alert()

    broker.subscribe("devices/+/anomalies", routed)
    agent = EdgeAgent(broker=broker, device_id=TELEMETRY.device_id)

    # Fault schedule: alternate healthy and faulty phases so the dashboard
    # shows the system reacting over time (great for a live demo).
    schedule = (
        [("none", 0.0)] if not cycle_faults
        else [("none", 0.0), ("bearing", 0.7), ("none", 0.0),
              ("overheat", 0.8), ("combined", 0.9), ("none", 0.0)]
    )
    phase_len = 60  # samples per phase when cycling

    phase_idx = 0
    count_in_phase = 0
    fault_mode, severity = schedule[0]
    sim = MotorSensorSimulator(fault_mode=fault_mode, fault_severity=severity, seed=7)

    while TELEMETRY.running:
        reading = sim.read_sample()
        event = agent.process(reading)
        TELEMETRY.add_reading(reading.as_dict(), is_anomaly=event is not None)
        if event is not None:
            # agent._publish already pushed to the broker; record for the feed.
            event_with_dev = {**event, "device_id": TELEMETRY.device_id}
            TELEMETRY.add_anomaly(event_with_dev)

        # Advance the fault schedule when cycling.
        if cycle_faults:
            count_in_phase += 1
            if count_in_phase >= phase_len:
                count_in_phase = 0
                phase_idx = (phase_idx + 1) % len(schedule)
                fault_mode, severity = schedule[phase_idx]
                sim = MotorSensorSimulator(
                    fault_mode=fault_mode, fault_severity=severity,
                    seed=7 + phase_idx,
                )

        time.sleep(sample_delay)

    store.close()


class DashboardHandler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path == "/" or self.path == "/index.html":
            self._serve_static("index.html", "text/html; charset=utf-8")
        elif self.path == "/api/state":
            body = json.dumps(TELEMETRY.snapshot()).encode("utf-8")
            self._send(200, body, "application/json")
        elif self.path == "/api/health":
            self._send(200, b'{"ok":true}', "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def _serve_static(self, name: str, content_type: str) -> None:
        path = os.path.join(STATIC_DIR, name)
        if not os.path.isfile(path):
            self._send(404, b"missing static asset", "text/plain")
            return
        with open(path, "rb") as f:
            self._send(200, f.read(), content_type)

    def log_message(self, *args) -> None:
        # Silence per-request console spam; keep startup/alerts readable.
        pass


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3 live dashboard server")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cycle-faults", action="store_true",
                   help="alternate healthy/faulty phases for a lively demo")
    p.add_argument("--delay", type=float, default=0.1,
                   help="seconds between samples (controls live speed)")
    args = p.parse_args()

    worker = threading.Thread(
        target=pipeline_loop, args=(args.cycle_faults, args.delay), daemon=True
    )
    worker.start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    url = f"http://localhost:{args.port}"
    print(f"Dashboard running at {url}")
    print(f"  pipeline: {'cycling faults' if args.cycle_faults else 'healthy baseline'} "
          f"| sample delay {args.delay}s")
    print("  Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        TELEMETRY.stop()
        server.shutdown()


if __name__ == "__main__":
    main()

# Edge-AI Predictive Maintenance

> An edge-AI predictive maintenance system that runs a TinyML anomaly-detection
> model on a (simulated) sensor device to detect machine faults on-device in
> real time, sending only anomalies to a serverless AWS cloud pipeline for
> storage, dashboards, and instant alerts.

Built end-to-end: on-device ML inference + a serverless AWS IoT pipeline —
designed to cut cloud data transfer dramatically while catching equipment
faults in real time.

> **Hardware note:** The physical sensor is currently simulated in software.
> The entire ML + edge + cloud architecture is real. When an MCU/Raspberry Pi
> is available, only `read_sample()` in `src/sensor_simulator.py` is replaced —
> nothing else changes. This is a standard way to prototype edge-AI systems.

---

## Why this project

It sits at the intersection of **embedded**, **CSE/ML**, and **AWS AI** —
a combination that maps directly to the in-demand **Edge AI / AIoT** space:

- **On-device ML (TinyML):** lightweight statistical features + an anomaly
  detector small enough to run on constrained hardware.
- **Edge intelligence:** inference happens at the edge; only meaningful events
  leave the device → large bandwidth savings.
- **Cloud (planned Phase 2):** secure device→cloud messaging and serverless
  processing on AWS.

---

## Architecture

```
          ┌─────────────────────────────────────────────┐
          │                EDGE DEVICE                   │
          │        (laptop now; MCU/Pi later)            │
          │                                              │
  [Sensor]│→ sensor_simulator.py  (swap point: real HW)  │
          │        │                                     │
          │        ▼                                     │
          │  edge_agent.py                               │
          │   • sliding window                           │
          │   • on-device inference (IsolationForest)    │
          │   • healthy → stay local                     │
          │   • anomaly → publish                        │
          └────────────────────┬─────────────────────────┘
                               │ (only anomalies)
                               ▼
                    ┌──────────────────────┐
                    │  AWS IoT Core (MQTT)  │   ◀── Phase 2
                    └──────────┬────────────┘
                               ▼
                    [Lambda] → [DynamoDB / Timestream]
                               ▼
                    [Dashboard + SNS alerts]       ◀── Phase 3
```

---

## Project structure

```
edge-ai-predictive-maintenance/
├── src/
│   ├── sensor_simulator.py   # simulated motor sensor (hardware swap point)
│   ├── train_model.py        # trains the anomaly detector on healthy data
│   ├── edge_agent.py         # live on-device inference + bandwidth metrics
│   ├── run_pipeline.py       # Phase 2: end-to-end edge -> cloud runner
│   └── cloud/                # in-process pipeline mirroring AWS (Phase 2)
│       ├── message_broker.py # mirrors AWS IoT Core (pub/sub, MQTT topics)
│       ├── message_processor.py  # mirrors AWS Lambda (event handler)
│       ├── datastore.py      # mirrors DynamoDB (SQLite-backed)
│       └── alert_handler.py  # mirrors AWS SNS (alerts)
├── models/                   # trained model artifacts (generated)
├── data/                     # datasets + pipeline.db (generated)
├── requirements.txt
└── README.md
```

---

## How it works

1. **Simulator** generates realistic vibration + temperature readings for a
   motor. It can inject faults: `bearing`, `overheat`, `spike`, `combined`.
2. **Training** (`train_model.py`) streams *healthy* data, slices it into short
   sliding windows, extracts 8 cheap statistical features (mean/std/min/max/
   peak-to-peak of vibration, plus temperature stats), and fits an unsupervised
   **IsolationForest**. Training only on healthy data mirrors real predictive
   maintenance, where failures are rare and mostly unlabeled.
3. **Edge agent** (`edge_agent.py`) streams live data through the model with a
   sliding window. Healthy windows stay on-device; anomalies are the events
   that *would* be published to AWS IoT Core (Phase 2). It reports a
   bandwidth-reduction metric.

---

## Quickstart

Requires Python 3.10+ (tested on 3.13).

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (writes models/anomaly_model.pkl)
python src/train_model.py

# 3. Run the edge agent on a faulty motor
python src/edge_agent.py --fault combined --severity 0.8

# ...or a healthy motor (should report no anomalies)
python src/edge_agent.py --fault none
```

### Edge agent options

| Flag         | Description                                   | Default    |
|--------------|-----------------------------------------------|------------|
| `--fault`    | `none`/`bearing`/`overheat`/`spike`/`combined`| `combined` |
| `--severity` | fault severity `0..1`                         | `0.8`      |
| `--samples`  | number of samples to stream                   | `600`      |
| `--delay`    | seconds between samples (`0.05` = live feel)  | `0.0`      |

---

## Phase 2 — End-to-end pipeline (mirrors AWS, runs locally)

Phase 2 adds the cloud side as an **in-process pipeline that mirrors the AWS
architecture 1:1**, so the whole system runs on your laptop with no AWS account
required. Each component is written against an AWS-shaped interface, so
migrating to real AWS is a localized change — callers don't move.

| Local component (`src/cloud/`) | Mirrors AWS service | Role |
|--------------------------------|---------------------|------|
| `message_broker.py`            | **IoT Core**        | MQTT-style pub/sub with `+`/`#` topic wildcards |
| `message_processor.py`         | **Lambda**          | Event handler: validate → store → alert |
| `datastore.py`                 | **DynamoDB**        | Persists anomaly events + rolling device health |
| `alert_handler.py`             | **SNS**             | Publishes severity-filtered alerts |

Run the full pipeline (sensor → edge → broker → processor → store → alert):

```bash
# faulty motor — events flow through the whole chain and fire alerts
python src/run_pipeline.py --fault combined --severity 0.8 --samples 400

# healthy motor — nothing crosses the "network", 100% bandwidth reduction
python src/run_pipeline.py --fault none --samples 400
```

The edge→cloud boundary is `EdgeAgent._publish()`, which publishes anomalies to
the broker on topic `devices/<device_id>/anomalies`. For the real-AWS migration,
that single call becomes an MQTT publish to AWS IoT Core.

### Phase 2 results (measured locally)

- Faulty run: edge published **281** events → IoT **281** → Lambda **281** →
  DynamoDB **281** → SNS **281** (counts flow consistently end-to-end).
- Healthy run: **0** events through the entire chain (**100%** bandwidth reduction).
- Broker topic-wildcard matching (`+`, `#`) verified against MQTT semantics.

---

## Phase 1 results (measured locally)

- Healthy windows flagged as anomalies: **~1%** (low false-alarm rate)
- Faulty windows flagged as anomalies: **100%**
- On a combined-fault run, first anomaly detected at **t ≈ 1.9 s**
- On a healthy run: **0 events sent → 100% bandwidth reduction**

> These numbers come from the simulated environment and will be recalibrated
> against real sensor data once hardware is attached.

---

## Roadmap

- **Phase 1 — Edge simulator + ML (local)** ✅ *done*
- **Phase 2 — Edge → cloud pipeline** ✅ *done (local AWS-mirror)* —
  anomalies published to a broker (IoT Core), processed by a Lambda-equivalent,
  stored in a DynamoDB-equivalent, and alerted via an SNS-equivalent.
  Real-AWS swap: point `_publish()` at AWS IoT Core and move the processor into
  a Lambda. *(Pending AWS account activation.)*
- **Phase 3 — Dashboard + polish:** live web dashboard, anomaly history,
  device health; quantify bandwidth saved, detection lead time, inference
  latency.
- **Phase 4 — Productionize:** Infrastructure-as-code (Terraform/CDK),
  CI/CD, architecture diagram, measured results.
- **Hardware:** replace `read_sample()` with real accelerometer/temperature
  reads on an MCU/Raspberry Pi.

---

## Tech

Python · NumPy · pandas · scikit-learn (IsolationForest) · joblib
AWS IoT Core · Lambda · DynamoDB/Timestream · SNS *(Phase 2+)*

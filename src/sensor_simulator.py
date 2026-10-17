"""
sensor_simulator.py
-------------------
Simulates a vibration + temperature sensor attached to an industrial motor.

This stands in for real hardware (e.g., an accelerometer + thermistor on an
ESP32/Raspberry Pi). The output format is intentionally simple so that, when
real hardware is available, only read_sample() needs to be swapped — the rest
of the pipeline (edge agent, ML model, cloud) stays identical.

A "healthy" motor shows:
  - vibration oscillating around a stable baseline with small noise
  - temperature near a nominal operating point

A "faulty" motor shows one or more of:
  - rising vibration amplitude (bearing wear / imbalance)
  - temperature climbing (friction / overheating)
  - sudden spikes (mechanical shock)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Iterator


# Nominal operating characteristics of a healthy motor.
BASELINE_VIBRATION = 1.0      # arbitrary "g" units
BASELINE_TEMPERATURE = 45.0   # degrees Celsius
VIBRATION_NOISE = 0.08
TEMPERATURE_NOISE = 0.4
ROTATION_HZ = 0.5             # slow oscillation to mimic rotating machinery


@dataclass
class SensorReading:
    """A single timestamped multi-channel sensor sample."""
    t: float            # seconds since stream start
    vibration: float    # vibration amplitude
    temperature: float  # temperature in Celsius

    def as_dict(self) -> dict:
        return asdict(self)


class MotorSensorSimulator:
    """
    Produces a stream of SensorReadings for a simulated motor.

    Fault injection is controlled by `fault_mode`:
      - "none"        : healthy motor
      - "bearing"     : gradually rising vibration (degradation over time)
      - "overheat"    : gradually rising temperature
      - "spike"       : intermittent sudden vibration spikes
      - "combined"    : bearing wear + overheating together

    `fault_severity` (0..1) scales how aggressive the fault is.
    """

    def __init__(
        self,
        fault_mode: str = "none",
        fault_severity: float = 0.0,
        sample_rate_hz: float = 10.0,
        seed: int | None = None,
    ) -> None:
        self.fault_mode = fault_mode
        self.fault_severity = max(0.0, min(1.0, fault_severity))
        self.dt = 1.0 / sample_rate_hz
        self._t = 0.0
        self._step = 0
        self._rng = random.Random(seed)

    def _healthy_sample(self) -> tuple[float, float]:
        """Baseline vibration + temperature with sinusoidal motion and noise."""
        osc = math.sin(2 * math.pi * ROTATION_HZ * self._t)
        vibration = BASELINE_VIBRATION + 0.15 * osc
        vibration += self._rng.gauss(0, VIBRATION_NOISE)

        temperature = BASELINE_TEMPERATURE + 0.5 * osc
        temperature += self._rng.gauss(0, TEMPERATURE_NOISE)
        return vibration, temperature

    def _apply_fault(self, vibration: float, temperature: float) -> tuple[float, float]:
        """Mutate a healthy sample according to the configured fault mode."""
        s = self.fault_severity
        # Progress grows over time so degradation faults worsen as they run.
        progress = min(1.0, self._step * self.dt / 30.0)  # ramps over ~30s

        if self.fault_mode == "bearing":
            vibration += s * (0.6 + 1.8 * progress)
            vibration += self._rng.gauss(0, VIBRATION_NOISE * (1 + 2 * s))

        elif self.fault_mode == "overheat":
            temperature += s * (5.0 + 25.0 * progress)

        elif self.fault_mode == "spike":
            if self._rng.random() < 0.05 * (0.5 + s):
                vibration += self._rng.uniform(1.5, 4.0) * (0.5 + s)

        elif self.fault_mode == "combined":
            vibration += s * (0.6 + 1.8 * progress)
            temperature += s * (5.0 + 25.0 * progress)
            if self._rng.random() < 0.05 * (0.5 + s):
                vibration += self._rng.uniform(1.5, 4.0) * (0.5 + s)

        return vibration, temperature

    def read_sample(self) -> SensorReading:
        """
        Return the next sensor reading.

        >>> HARDWARE SWAP POINT <<<
        On real hardware, replace the body of this method with an actual
        read from the accelerometer / temperature sensor. Everything
        downstream stays the same.
        """
        vibration, temperature = self._healthy_sample()
        if self.fault_mode != "none":
            vibration, temperature = self._apply_fault(vibration, temperature)

        reading = SensorReading(
            t=round(self._t, 3),
            vibration=round(vibration, 4),
            temperature=round(temperature, 4),
        )
        self._t += self.dt
        self._step += 1
        return reading

    def stream(self, n: int) -> Iterator[SensorReading]:
        """Yield `n` sequential readings."""
        for _ in range(n):
            yield self.read_sample()


if __name__ == "__main__":
    # Quick manual check: print a few healthy and a few faulty samples.
    print("Healthy motor:")
    healthy = MotorSensorSimulator(fault_mode="none", seed=1)
    for r in healthy.stream(3):
        print(" ", r.as_dict())

    print("\nFaulty motor (bearing wear):")
    faulty = MotorSensorSimulator(fault_mode="bearing", fault_severity=0.8, seed=1)
    for r in faulty.stream(3):
        print(" ", r.as_dict())

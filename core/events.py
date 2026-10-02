"""Plain-dict integration boundary; no dependency on another person's CV."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping


class RiskLevel(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class RiskEvent:
    object: str
    direction: str
    risk: RiskLevel | str
    timestamp: float
    distance_m: float | None = None
    ttc_s: float | None = None

    def __post_init__(self):
        level = RiskLevel[self.risk.upper()] if isinstance(self.risk, str) else RiskLevel(self.risk)
        object.__setattr__(self, "risk", level)
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")

    @classmethod
    def from_dict(cls, value: Mapping) -> RiskEvent:
        return cls(object=value["object"], direction=value["direction"], risk=value["risk"],
                   timestamp=float(value["timestamp"]), distance_m=value.get("distance_m"),
                   ttc_s=value.get("ttc_s"))


@dataclass(frozen=True)
class BusEvent:
    route: str
    confidence: float
    bbox: tuple[float, float, float, float] | None
    timestamp: float

    def __post_init__(self):
        if not math.isfinite(self.timestamp) or not 0 <= self.confidence <= 1:
            raise ValueError("Bus timestamp must be finite and confidence must be 0–1")

    @classmethod
    def from_dict(cls, value: Mapping) -> BusEvent:
        bbox = value.get("bbox")
        if bbox is not None and len(bbox) != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2]")
        return cls(route=str(value["route"]), confidence=float(value.get("confidence", 1)),
                   bbox=tuple(float(v) for v in bbox) if bbox is not None else None,
                   timestamp=float(value["timestamp"]))


@dataclass(frozen=True)
class SystemEvent:
    message: str
    timestamp: float
    kind: str = "info"

    def __post_init__(self):
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")

    @classmethod
    def from_dict(cls, value: Mapping) -> SystemEvent:
        return cls(message=str(value.get("message", value.get("text", ""))),
                   timestamp=float(value["timestamp"]), kind=value.get("kind", "info"))


def from_dict(value: Mapping) -> RiskEvent | BusEvent | SystemEvent:
    event_type = value.get("type")
    factories = {"risk": RiskEvent, "bus": BusEvent, "system": SystemEvent}
    if event_type not in factories:
        raise ValueError(f"Unknown integration event type: {event_type!r}")
    return factories[event_type].from_dict(value)


def risk_to_phrase(event: RiskEvent) -> str:
    direction = {"front": "ahead", "back": "behind"}.get(event.direction.lower(), event.direction.lower())
    if direction not in {"right", "left", "ahead", "behind"}:
        direction = "ahead"
    name = {"car": "Car", "motorcycle": "Motorcycle", "motorbike": "Motorcycle",
            "bus": "Bus", "truck": "Truck", "bicycle": "Bicycle", "person": "Person"}.get(event.object.lower(), "Obstacle")
    if event.risk == RiskLevel.CRITICAL:
        return f"{name} {direction}!" if name == "Motorcycle" else f"Stop. {name} {direction}."
    if event.risk == RiskLevel.HIGH:
        subject = "Person" if name == "Person" else "Vehicle" if name != "Obstacle" else "Obstacle"
        return f"{subject} approaching {direction}."
    return f"{name} {direction}."

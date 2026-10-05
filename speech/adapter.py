"""Translate Person 1 risk observations and confirmed buses into speech."""
from __future__ import annotations

import re

from common.events import RiskLevel
from speech.scheduler import Priority


def bus_phrase(current_bus):
    if not current_bus:
        return "No bus identified."
    number = re.sub(r"(\d)([A-Z])", r"\1 \2", current_bus["route_number"])
    route = current_bus["route_name"].replace("→", "to")
    return f"Bus {number}, {route}." if route else f"Bus {number}."


def submit_bus(scheduler, current_bus):
    text = bus_phrase(current_bus)
    return scheduler.submit(text, priority=Priority.BUS, identity=f"bus:{text}")


def submit_risk(scheduler, event):
    if event.risk_level not in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        return None
    direction = {"CENTER": "ahead", "LEFT": "left", "RIGHT": "right"}.get(event.direction.upper(), "ahead")
    object_name = "Motorcycle" if event.object_class.lower() in {"motorcycle", "motorbike"} else event.object_class.title()
    text = (f"Stop. {object_name} {direction}." if event.risk_level == RiskLevel.CRITICAL
            else f"{object_name} {direction}.")
    priority = Priority.CRITICAL if event.risk_level == RiskLevel.CRITICAL else Priority.HIGH
    return scheduler.submit(text, priority=priority,
                            identity=f"risk:{event.object_class}:{direction}")

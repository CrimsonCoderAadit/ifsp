"""Per-track risk hysteresis: quick escalation, slower de-escalation."""

from dataclasses import replace

from common.events import RiskEvent, RiskLevel


ORDER = (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)


class RiskStabilizer:
    def __init__(self, escalation_persistence: int = 1,
                 deescalation_persistence: int = 3, cooldown_s: float = 0.15,
                 stale_timeout_s: float = 2.0) -> None:
        if escalation_persistence < 1 or deescalation_persistence < 1 or cooldown_s < 0:
            raise ValueError("invalid risk hysteresis settings")
        self.escalation_persistence = escalation_persistence
        self.deescalation_persistence = deescalation_persistence
        self.cooldown_s = cooldown_s
        self.stale_timeout_s = stale_timeout_s
        self.state: dict[int, tuple[RiskLevel, RiskLevel, int, float, float]] = {}

    def apply(self, event: RiskEvent, thresholds: dict) -> RiskEvent:
        for track_id, state in list(self.state.items()):
            if event.timestamp - state[4] > self.stale_timeout_s:
                del self.state[track_id]
        old = self.state.get(event.track_id)
        if old is None:
            self.state[event.track_id] = (event.risk_level, event.risk_level, 0, event.timestamp, event.timestamp)
            return event
        level, candidate, count, changed_at, _last_seen = old
        if event.risk_level == level:
            candidate, count = level, 0
        else:
            candidate, count = (candidate, count + 1) if candidate == event.risk_level else (event.risk_level, 1)
            needed = (self.escalation_persistence if ORDER.index(candidate) > ORDER.index(level)
                      else self.deescalation_persistence)
            escalating = ORDER.index(candidate) > ORDER.index(level)
            if count >= needed and (escalating or event.timestamp - changed_at >= self.cooldown_s):
                level, changed_at, count = candidate, event.timestamp, 0
        self.state[event.track_id] = (level, candidate, count, changed_at, event.timestamp)
        if level == event.risk_level:
            return event
        floor = {RiskLevel.LOW: 0.0, RiskLevel.MEDIUM: thresholds["medium_threshold"],
                 RiskLevel.HIGH: thresholds["high_threshold"],
                 RiskLevel.CRITICAL: thresholds["critical_threshold"]}[level]
        return replace(event, risk_level=level, risk_score=max(event.risk_score, floor),
                       risk_reason=event.risk_reason + "+HYSTERESIS_HOLD")

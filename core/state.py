"""Shared state and per-track temporal route votes; owned by Assistant's lock."""
from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TargetBus:
    route: str
    set_at: float
    long_names: tuple[str, ...] = ()


@dataclass
class AssistantState:
    target_bus: TargetBus | None = None
    last_announcement: str | None = None
    mode: str = "idle"
    votes: dict = field(default_factory=dict)
    confirmed: dict = field(default_factory=dict)

    def vote(self, track_id, route, confidence, method, timestamp, *, k=2, n=3):
        if not 1 <= k <= n:
            raise ValueError("Voting requires 1 <= K <= N")
        buffer = self.votes.setdefault(track_id, deque(maxlen=n))
        buffer.append(route)
        if route is None:
            return None
        counts = Counter(buffer)
        confident = confidence > .9 and method == "exact"
        # A tie cannot confirm a new route. Current reads must support it.
        maximum = max(counts.values())
        winners = [candidate for candidate, count in counts.items() if count == maximum and candidate is not None]
        agreed = counts[route] >= k and winners == [route]
        if confident or agreed:
            self.confirmed[track_id] = {"route": route, "confidence": confidence,
                                        "timestamp": timestamp, "method": method}
            return route
        return None

    def recent_routes(self, now, ttl=3):
        return sorted({value["route"] for value in self.confirmed.values() if now-value["timestamp"] < ttl})

    def forget_route(self, route):
        """Drop votes and confirmations that involve route, so it must re-confirm."""
        for track_id in [key for key, value in self.votes.items() if route in value]:
            del self.votes[track_id]
        for track_id in [key for key, value in self.confirmed.items() if value["route"] == route]:
            del self.confirmed[track_id]

    def prune(self, active_ids):
        for mapping in (self.votes, self.confirmed):
            for track_id in list(mapping):
                if track_id not in active_ids:
                    del mapping[track_id]

    def payload(self):
        return {"target_bus": ({"route": self.target_bus.route, "set_at": self.target_bus.set_at,
                                "long_names": list(self.target_bus.long_names)}
                               if self.target_bus else None),
                "last_announcement": self.last_announcement, "mode": self.mode,
                "votes": {str(key): list(value) for key, value in self.votes.items()},
                "confirmed_routes": {str(key): dict(value) for key, value in self.confirmed.items()}}

"""Audible Review 1 demo: HIGH car-right risk interrupts Bus 21G speech.

Run: .venv/bin/python -m cli.mock_risk
--null exercises the same scheduler without speakers; it is not the audible demo.
"""
import argparse
import time
from pathlib import Path

from core.config import ROOT
from core.events import BusEvent, RiskEvent, risk_to_phrase
from speech.scheduler import AudioScheduler, LocalSink, NullSink, SchedulerConfig
from speech.tts import SpeechSynthesizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--null", action="store_true")
    parser.add_argument("--log", type=Path, default=ROOT / "logs/mock_risk.csv")
    args = parser.parse_args()
    tts = SpeechSynthesizer()
    bus_text = "Bus 21G detected."
    risk = RiskEvent("car", "right", "HIGH", time.time())
    # Warm BOTH phrases on the main thread, including an OS fallback voice.
    # Risk playback then incurs only cache lookup, no cold synthesis delay.
    bus_pcm, sr = tts.synthesize(bus_text)
    tts.synthesize(risk_to_phrase(risk))
    duration = len(bus_pcm)/sr
    sink = NullSink(duration_s=None) if args.null else LocalSink()
    config = SchedulerConfig(log_path=args.log)
    interrupted = False
    with AudioScheduler(tts, sink, config) as scheduler:
        bus_id = scheduler.submit(BusEvent("21G", 1.0, None, time.time()))
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            if any(row["decision"] == "played" and row["item_id"] == bus_id for row in list(scheduler.decisions)):
                break
            if any(row["decision"] in {"sink_error", "synthesis_error", "scheduler_error"} for row in list(scheduler.decisions)):
                raise RuntimeError("Audio failed; see scheduler decision log")
            time.sleep(.005)
        else:
            raise TimeoutError("Bus speech did not start")
        time.sleep(duration/2)
        submitted_at = time.monotonic()
        risk_id = scheduler.submit({"type": "risk", "object": "car", "direction": "right",
                                    "risk": "HIGH", "timestamp": time.time(), "extra": "ignored"})
        if not scheduler.wait_idle(10):
            raise TimeoutError("Demo did not finish")
        rows = list(scheduler.decisions)
        for row in rows:
            print(f"{row['monotonic']-submitted_at:+.4f}s {row['decision']:20s} "
                  f"p={row['priority']} {row['identity']} {row['text']} {row['detail']}")
        stops = [row for row in rows if row["decision"] == "pre-empted" and row["item_id"] == bus_id]
        plays = [row for row in rows if row["decision"] == "played" and row["item_id"] == risk_id]
        interrupted = bool(stops and plays)
        if interrupted:
            print(f"Bus interrupted in {(stops[0]['monotonic']-submitted_at)*1000:.3f} ms; "
                  f"warning playback requested in {(plays[0]['monotonic']-submitted_at)*1000:.3f} ms")
            if isinstance(sink, LocalSink):
                print(f"Device output latency: {sink.output_latency_ms:.3f} ms + at most one {sink.block_duration_ms:g} ms block; "
                      f"underflows: {sink.underflows}")
            if (stops[0]["monotonic"]-submitted_at)*1000 >= 50:
                raise RuntimeError("Interruption exceeded the 50 ms target")
    print(f"Decision CSV: {args.log}")
    if not interrupted:
        raise RuntimeError("Demo failed to interrupt the bus")
    print("NULL demo passed" if args.null else "LOCAL AUDIO demo passed")


if __name__ == "__main__":
    main()

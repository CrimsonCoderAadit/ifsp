# Review 1 demo

1. Start in a fresh Python 3.10–3.12 environment using the README commands. Have `yolo26n.pt`, the GTFS ZIP, Vosk/Piper and RapidOCR models ready locally.
2. Open the Rider page. Upload `data/real/commons/MTC_Bus_21G.JPG` and show BUS DETECTED, OCR `21G`, NORMALIZED `21G`, and the GTFS route name. The current feed selects **Kilambakkam → Island Ground** by trip count; do not use the example Tambaram → Broadway line.
3. Hold the mic button and say “bus”. The expected caption is “Bus 21 G, Kilambakkam to Island Ground.” Show the spoken audio.
4. Start `python main.py --source data/real/own/<video> --bus-reader --listen --web --save --csv --no-display` when the team's real video is available. The page and video loop share the detector and current-bus state in one process. A HIGH/CRITICAL risk event should interrupt bus speech and show the red banner.
5. Show an unknown-route case such as `data/real/commons/Chennai_bus.jpg`, where the visible route is 576 but the current GTFS feed has no 576. Explain that the app must not announce a different route.

## Current evidence

- A 150-frame run of the bundled synthetic clip completed with one yolo26n YOLO instance and saved `outputs/test_video_ego_motion_20261005_112007.mp4`; it contained no detected bus tracks.
- A real Commons 21G photo produced OCR `21G` and a GTFS match. A prerecorded Vosk utterance returned the caption “Bus 21 G, Kilambakkam to Island Ground.” and Piper WAV audio over WebSocket in 1,123 ms including first model loads. This is one observation, not a test-set accuracy or live-mic latency.
- The 45-second spoken screen capture, 15-second risk-interruption clip, and real own-video run require the team's footage and live microphone/audio session.

A fresh Python 3.12 venv on macOS arm64 installed the pinned dependencies and passed 47 unittests plus 189 pytest tests (10 skipped, including browser tests without a running server). It processed the 150-frame bundled clip with yolo26n in one process at 21.72 ms mean detector time and 24.52 ms mean total frame time. Eight Playwright cases on four real Commons inputs passed at both 375 px and 1280 px; screenshots are in `docs/rider-*.png`.

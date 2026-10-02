# Known failures and unmeasured conditions

Measured on the laptop (Apple Silicon, macOS) on 2026-10-02 with
`python -m bench.benchmark --device-label laptop`. Raw per-case rows are in
`logs/latency.csv`; the scheduler's raw per-event decisions are in
`logs/scheduler_decisions.csv` and `logs/web_latency.csv`.

> **There are no real bus photos.** `data/test_images/real/` contains only a
> README and an empty `labels.csv`. Every route-accuracy number below comes
> from 24 generated board close-ups. Synthetic accuracy is not evidence of
> accuracy on Chennai streets.

## Route-number reading

| Dataset | Images | Exact | Fuzzy-correct | Wrong | No route |
| --- | --- | --- | --- | --- | --- |
| Synthetic close-ups (board mode) | 24 | 23 | 0 | 1 | 0 |
| Real Chennai photos | 0 | not measured | not measured | not measured | not measured |

The single wrong case was `019_M70_led.jpg` (LED style, JPEG artefacts):
expected **M70**, OCR read `6D`, which is itself a valid GTFS route, so it
was accepted as an exact match after one attempt (346 ms). Catalogue
correction cannot catch an error that lands on another real route.

Across the 210 individual crop and preprocessing combinations, 183 were
exact, 5 wrong and 22 returned no route. Split by crop, the top 35% crop was
89/96 exact with 0 wrong (median 108 ms per read); the full crop was 94/114
exact with 5 wrong (median 344 ms per read).

## Conditions

| Condition | What this run shows | Mitigation in place | Next step |
| --- | --- | --- | --- |
| Board mode reads the full crop first | Requested change, now in place. On the 24 synthetic boards it made reads **slower**: median 344 ms versus 110 ms when the top crop was tried first, because the route sits in the top 35% and the full crop is about three times larger. All five wrong variant reads came from full crops. It helps when the route is *not* near the top of the frame, which was the case on the live frame that motivated it. | Pipeline accuracy unchanged at 23/24. | Measure on real close-ups; if the route is usually at the top, try top then full again, or downscale the full crop before OCR. |
| Glare on LED boards | 6/6 synthetic glare images exact; real glare untested. | CLAHE, threshold and inverted variants; OCR across frames. | Collect sun and night LED footage. |
| Motion blur | 11/11 synthetic motion images exact; moving real buses untested. | Repeated reads and 2-of-3 voting. | Rank frames by sharpness before OCR. |
| JPEG artefacts | 6/7 exact; the one failure is the M70→6D case above. | None specific. | Require stronger multi-frame agreement when a conflicting exact route exists. |
| Tamil-only boards | Not tested. Generated boards have Latin route tokens; PP-OCRv4 here is Latin/Chinese. | Optional PaddleOCR Tamil adapter (simulated-engine test only). | Provision Tamil weights and labelled Tamil-only signs. |
| Partial occlusion, oblique angles, CUT/EXP labels | No labelled cases. 7 synthetic perspective images were exact. | Perspective rectification, track voting. | Label real examples of each. |
| Bus detection on real buses | The detector ran on 30 synthetic-video frames in median 29.3 ms (p95 33.3 ms) and found **0 buses**, as expected for board-only frames. This is latency only; detection recall is unmeasured. | YOLO11n ONNX; board mode bypasses it for close-ups. | Label real street frames. |
| Synthetic commands misheard | Piper's output differs on every run, so the round-trip test is not deterministic. Over 40 fresh syntheses per command (streamed into Vosk), the intent was right for "repeat" 40/40, "stop" 40/40, "what is ahead" 40/40 after the parser fix (only 33/40 transcripts were exact; others were "is ahead", "for is ahead", "where is ahead"), and "find bus twenty one g" 38/40 ("twenty what gee" twice). Before the fix the "what is ahead" test failed about one full-suite run in five; afterwards 10 consecutive full-suite runs passed, but a "repeat" failure was seen once in about 50 runs of the round-trip file and could not be reproduced to capture its transcript. | The intent parser accepts any short phrase ending in "ahead"; the four saved Piper clips score 4/4 in the benchmark. | Record real speakers in quiet and roadside conditions; report intent accuracy by condition. |
| Real speech, accents, traffic noise | `data/test_audio/real/` has no recordings. STT finalisation was a median 6.1 ms after buffered feed, not live capture latency. | Indian-English Vosk model, command grammar, typed fallback. | Record consenting local speakers. |
| Repeated announcements | At 30 submissions per second for 5 s: HIGH warning played 2/150 with cooldown versus 150/150 without; a confirmed target bus played 1/150 with cooldown versus 150/150 without (149 repeats). | Per-identity cooldowns; collapsed log rows in the UI. | Tune BUS cooldown and TTL against timed real bus arrivals. |
| New search inherited old cooldown (fixed) | Before the fix, "cancel bus" then "find bus 21G" stayed silent for up to 6 s. A regression test and the live web check now show the second announcement 0.9 s after the first. | FIND_BUS and CANCEL_BUS clear `bus:` cooldowns and the old target's votes. | None. Side effect: after "cancel bus", "what is ahead" says "No confirmed buses yet." until the board is read again. |
| ONNX Runtime telemetry (fixed) | onnxruntime 1.30.0 on macOS starts a telemetry thread that posts to `mobile.events.data.microsoft.com`. In this run it aborted Python at exit (exit code 134) in about one test run in three. `disable_telemetry_events()` did not stop it. | `ORT_DISABLE_TELEMETRY=1` is set in `core/config.py` before onnxruntime loads, and in the Dockerfile. 10/10 later suite runs exited cleanly. | The `socket.connect` block in the offline tests cannot see native network calls; audit with a firewall rule if offline operation must be proven. |
| Missed bus during warning suppression | Not tested with a bus passing during suppression. BUS cooldown is 6 s and TTL 3 s. | Interrupted BUS retries once within TTL; relevance rechecked before playback. | Timed bus-arrival video with overlapping warnings. |

Other measured limits: Piper model load 696.5 ms, first synthesis 121.2 ms,
cached synthesis 0.005 ms; LocalSink event to software `play()` 1.8 ms.
Audio driver buffering and acoustic onset were not measured. Nothing here was
measured on a Raspberry Pi.

The `--approaching-only` cue requires three growing bounding-box areas and
12% total growth. It passed a simulated test only; camera shake and box
jitter on real video are untested.

## Web UI limitations

Measured with `python -m pytest tests/ui` (Chromium 153 headless, Playwright
1.63, axe-core 4.13.0) on 2026-10-02: 12 passed, three consecutive runs.

| Limitation | Evidence | Next step |
| --- | --- | --- |
| Only Chromium is tested. | No Firefox, Safari or Raspberry Pi browser run. The Pi is the target device. | Run `tests/ui` with `p.firefox`/`p.webkit` and on the Pi's Chromium. |
| Microphone and camera are fakes in automation. | Space-to-talk used Chromium's fake device playing the Piper WAV (transcript “find bus twenty one gee”); the webcam path was checked with a canvas stream. No physical mic, phone camera or permission-denied path was tested. | Manual run on a phone over HTTPS, including denying permission. |
| Screen readers are not tested. | axe found no WCAG A/AA violations of any severity in all three themes, but automated rules cover only part of WCAG. Live-region wording and announcement order are unverified with VoiceOver, NVDA or TalkBack. | One session with each screen reader, ideally with a visually impaired tester. |
| The demo video is silent. | Playwright records frames only; `logs/ui/demo.webm` has no TTS audio. | Record the live demo with a system-audio screen recorder. |
| Risk direction is parsed from the spoken text. | `speak` messages carry no direction field, so the arrow comes from “left/right/behind” in the phrase. A rephrased warning would show the wrong arrow. | An additive `direction` field on `speak`. |
| The timeline is reconstructed from the 2 Hz state log. | It uses the last 40 collapsed scheduler rows; under heavy churn older rows drop out, and server/client clock skew is estimated from the first played item. The requeue line disappears once the replay leaves the 30 s window. | A dedicated timeline message or a larger log window. |
| The TTS card measures event → audio start for every item. | It includes time queued behind other speech, so an acknowledgement ahead of the bus message pushes it over the 300 ms budget (observed 363–801 ms). | Separate cards for queueing and synthesis. |
| Canvas performance on weak GPUs. | Resizing canvases every 250 ms dropped Chromium to one frame per second; canvases now resize only on size change (back to ~33 ms per frame). Pi frame rate is unmeasured. | Profile on the Pi with the timeline visible. |
| The start screen is a modal without a real `<dialog>`. | Background is `inert` and Tab is trapped on the Start button (a test caught focus escaping before this fix). Browsers without `inert` would let focus reach the page behind. | Use `<dialog>.showModal()` for the start screen. |

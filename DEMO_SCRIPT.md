# Browser demo shot list (about 60 seconds)

A 46-second scripted recording of this flow is at `logs/ui/demo.webm`
(regenerate with `UI_RECORD_DEMO=1 .venv/bin/python -m pytest tests/ui -k record_demo`).
It is silent: Playwright records the screen only, so the spoken TTS is not in
the file. For the submission, record the live run below with a screen
recorder that captures system audio, or narrate over the silent capture.

Start `python -m app.server`, open `http://localhost:8000` and press
**Start assistant**; you should hear “Ready.” and the pill should read
Connected. Use **Engineer** mode for the panel. In the camera card choose
**Samples**; the 21G board samples switch on close-up board mode for you.
A real bus clip can replace the samples once one has been labelled and tested.

## Demo 1: voice → bus → spoken result (25–30 seconds)

| Time | Shot / action | Expected evidence |
| --- | --- | --- |
| 0–5 s | Show the LED board (empty), the talk button and the pipeline tracer. | Connected; board says “Say find bus 21G”. |
| 5–12 s | Hold **Space** (or the talk button), say “find bus twenty one G”, release. | Level meter moves; partial then final transcript; tracer lights Mic → STT → Intent with ms; LED board shows **21G** and its GTFS destinations; acknowledgement is spoken. |
| 12–20 s | Click the **21G printed board** sample. | Status strip “Reading board…” then “21G confirmed — your bus”; green ✓ chip with votes, blue OCR box on “21 G”; tracer Detect → OCR “21 G” → Normalise 21G → GTFS match exact. |
| 20–28 s | Let the bus announcement play. | “Bus 21G detected ahead.” in Last spoken; green flash on the LED board; BUS bar on the speech timeline. |

If microphone permission or recognition fails, open **Type a command instead**,
send `find bus 21G`, and say on camera that this is the fallback.

## Demo 2: urgent risk interrupts bus information (25–30 seconds)

| Time | Shot / action | Expected evidence |
| --- | --- | --- |
| 0–8 s | Press **C** then type `find bus 21G` with the 21G sample in view. | A fresh “Bus 21G detected ahead.” starts (the new search is not blocked by the old cooldown). |
| 8–12 s | While it is audible, press **2** (HIGH car right) — a shortcut does not scroll the page — or use the risk pad and **Fire**. | Bus audio stops; red banner “Vehicle approaching right.” with a right arrow and right-edge glow. |
| 12–20 s | Scroll to “Last spoken”, then to the speech timeline. | “Bus 21G detected ahead.” plays again; the BUS bar ends with a hatched cut and red ✕, a HIGH bar, and a dashed line to the BUS replay. |
| 20–28 s | Press **Spam HIGH · 30 fps for 5 s**. | One or two HIGH bars and a “×148 suppressed” badge instead of 150 warnings; Event log shows `suppressed_cooldown ×N` rows. |

## Optional extra take: explaining silence

Switch close-up board mode off with the board sample still shown: the status
strip reads “No bus in view”. Load the **23C** sample with target 21G: a grey
chip and “23C confirmed” without an announcement. Press **H** to show the
high-contrast theme and **M** for Rider mode.

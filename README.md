# # AI Reading/Study Focus Tracker

A real-time webcam application that tracks eye gaze, blinks, and attention during study or reading sessions, then generates a short, encouraging end-of-session summary using a local Gemma model via Ollama — no images ever leave your machine.

## Features

- **Gaze Tracking** — Uses MediaPipe Face Mesh iris landmarks to detect whether you're looking at your screen or looking away.
- **Blink Detection** — Calculates eye aspect ratio (EAR) to count deliberate blinks, separate from sustained eye closure (drowsiness).
- **Focused vs. Distracted State Machine** — Debounces brief glances so momentary look-aways don't flip your status; tracks how long you spend focused vs. distracted, and counts distraction episodes.
- **Live Session Overlay** — On-screen display of current status, focused/distracted time, and blink count.
- **Local AI Session Summary** — At the end of a session, sends only aggregated stats (no images) — total time, focus percentage, distraction count, blink rate — to a local [Ollama](https://ollama.ai/) instance running `gemma2:2b`, which returns a short, encouraging summary and a practical tip for next time.
- **Session History Logging** — Saves each session's stats and AI summary to a local JSON log, viewable from the CLI menu.

## Requirements

- Python 3.8+
- A webcam
- [Ollama](https://ollama.ai/) installed and running locally with the `gemma2:2b` model pulled (optional — the app runs without it, just without the AI summary)

### Python dependencies

```bash
pip install opencv-python mediapipe requests
```

## Project Structure

```
.
├── main.py              # (this script)
├── logs/
│   └── focus_history.json   # created automatically after your first session
```

## Usage

Run the script and choose an option from the menu:

```bash
python main.py
```

```
1. Start a focus session
2. View past session history
```

1. **Start a focus session** — Opens the webcam feed and tracks your focus in real time. Press `q` or `ESC` to end the session; you'll then get a printed summary and an AI-generated recap.
2. **View past session history** — Lists your 15 most recent sessions with their stats and saved AI summaries.

## Configuration

Key parameters can be adjusted near the top of the script:

| Variable | Description |
|---|---|
| `OLLAMA_URL` | Ollama API endpoint |
| `MODEL_NAME` | Ollama model used for the session summary (default: `gemma2:2b`) |
| `GAZE_H_MIN` / `GAZE_H_MAX` | Horizontal iris-position band considered "looking at screen" |
| `GAZE_V_MIN` / `GAZE_V_MAX` | Vertical iris-position band considered "looking at screen" |
| `EAR_BLINK_THRESHOLD` | Eye aspect ratio below which eyes are considered closed |
| `EAR_CONSEC_FRAMES_BLINK` | Consecutive closed frames required to count as a deliberate blink |
| `EYES_CLOSED_DISTRACTION_SEC` | Eye-closure duration beyond a blink that counts as drowsy/distracted |
| `DISTRACTION_CONFIRM_SEC` | How long gaze must be away before counting as "distracted" |
| `FOCUS_CONFIRM_SEC` | How long gaze must return before counting as "focused" again |

## How It Works

1. MediaPipe Face Mesh (with iris refinement) tracks eye and iris landmarks each frame.
2. **Blinks**: eye aspect ratio (EAR) is computed per eye; a dip below `EAR_BLINK_THRESHOLD` for a few frames counts as a blink, while a longer closure is treated as drowsiness/distraction instead.
3. **Gaze**: the iris position within each eye's bounding box gives a horizontal/vertical ratio; values outside the centered band mean you're looking away from the screen.
4. A debounced state machine confirms a status change ("Focused" ↔ "Distracted") only after it holds for a short threshold, so quick glances don't count as distractions.
5. At session end, aggregated stats (never raw video or images) are sent to Gemma for a natural-language summary, which is printed and saved to your local session history.

## Privacy Note

No video, images, or gaze/blink frame-by-frame data are ever saved to disk or sent externally. Only the final aggregated session statistics (durations, percentages, counts) are sent to your **local** Ollama instance, and only those stats plus the generated summary are stored in `logs/focus_history.json`.

## License

Add a license of your choice (e.g. MIT) here.

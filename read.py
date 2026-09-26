import os
import json
import time
import math
import cv2
import mediapipe as mp
import requests
from datetime import datetime

# --- Config ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"

LOG_DIR = "logs"
SESSION_HISTORY_PATH = os.path.join(LOG_DIR, "focus_history.json")

# Gaze thresholds: ratio of iris position within the eye (0 = one corner, 1 = other corner).
# ~0.5 is centered/looking at screen; outside this band = looking away.
GAZE_H_MIN, GAZE_H_MAX = 0.35, 0.65
GAZE_V_MIN, GAZE_V_MAX = 0.35, 0.70

EAR_BLINK_THRESHOLD = 0.21       # eye aspect ratio below this = eyes closed
EAR_CONSEC_FRAMES_BLINK = 2      # frames closed to count as a deliberate blink
EYES_CLOSED_DISTRACTION_SEC = 2.0  # eyes closed longer than this = distracted (drowsy), not a blink

DISTRACTION_CONFIRM_SEC = 1.5    # gaze must be away for this long before counting as "distracted"
FOCUS_CONFIRM_SEC = 0.5          # gaze must return for this long before counting as "focused" again


# --- Gemma / Ollama ---
def ask_gemma_summary(stats):
    """
    Sends only the aggregated session statistics (no images) to Gemma
    for a short, friendly end-of-session summary.
    """
    prompt = (
        f"A person just finished a {stats['total_minutes']:.1f}-minute study/reading session. "
        f"They were focused for {stats['focused_minutes']:.1f} minutes "
        f"({stats['focus_percent']:.0f}%) and distracted for {stats['distracted_minutes']:.1f} minutes. "
        f"They had {stats['distraction_events']} separate distraction episodes "
        f"and an average blink rate of {stats['blinks_per_minute']:.1f} blinks per minute. "
        "In 2-3 short, encouraging sentences, summarize how the session went and give one "
        "practical suggestion for their next session. Do not mention that you are an AI or "
        "reference any image or camera."
    )

    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=15)
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        return text if text else "(No response from Gemma.)"
    except requests.exceptions.ConnectionError:
        return "[Ollama not reachable - is 'ollama serve' running?]"
    except requests.exceptions.Timeout:
        return "[Ollama request timed out.]"
    except Exception as e:
        return f"[Gemma error: {e}]"


# --- Geometry helpers ---
def _px(landmark, w, h):
    return landmark.x * w, landmark.y * h


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(landmarks, indices, w, h):
    """
    Standard 6-point EAR: indices = [outer_corner, top1, top2, inner_corner, bottom2, bottom1]
    EAR = (||top1-bottom1|| + ||top2-bottom2||) / (2 * ||outer_corner-inner_corner||)
    Lower EAR = eye more closed.
    """
    p = [_px(landmarks[i], w, h) for i in indices]
    vertical_1 = _dist(p[1], p[5])
    vertical_2 = _dist(p[2], p[4])
    horizontal = _dist(p[0], p[3])
    if horizontal == 0:
        return 0.3
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def gaze_ratios(landmarks, iris_center_idx, eye_left_idx, eye_right_idx, eye_top_idx, eye_bottom_idx, w, h):
    """
    Returns (horizontal_ratio, vertical_ratio) describing where the iris sits
    within the eye's bounding box. 0.5/0.5 = centered (looking straight ahead).
    """
    iris = _px(landmarks[iris_center_idx], w, h)
    left = _px(landmarks[eye_left_idx], w, h)
    right = _px(landmarks[eye_right_idx], w, h)
    top = _px(landmarks[eye_top_idx], w, h)
    bottom = _px(landmarks[eye_bottom_idx], w, h)

    eye_width = right[0] - left[0]
    eye_height = bottom[1] - top[1]

    h_ratio = (iris[0] - left[0]) / eye_width if eye_width != 0 else 0.5
    v_ratio = (iris[1] - top[1]) / eye_height if eye_height != 0 else 0.5

    return h_ratio, v_ratio


# MediaPipe Face Mesh landmark indices (with refine_landmarks=True for iris)
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

LEFT_IRIS_CENTER = 468
LEFT_EYE_LEFT, LEFT_EYE_RIGHT = 33, 133
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145

RIGHT_IRIS_CENTER = 473
RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT = 362, 263
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374


# --- Session History Storage ---
def save_session(stats, gemma_summary):
    os.makedirs(LOG_DIR, exist_ok=True)

    history = []
    if os.path.exists(SESSION_HISTORY_PATH):
        try:
            with open(SESSION_HISTORY_PATH, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            history = []

    history.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_minutes": round(stats["total_minutes"], 2),
        "focused_minutes": round(stats["focused_minutes"], 2),
        "distracted_minutes": round(stats["distracted_minutes"], 2),
        "focus_percent": round(stats["focus_percent"], 1),
        "distraction_events": stats["distraction_events"],
        "blinks_per_minute": round(stats["blinks_per_minute"], 1),
        "summary": gemma_summary
    })

    with open(SESSION_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def view_session_history():
    if not os.path.exists(SESSION_HISTORY_PATH):
        print("No focus sessions recorded yet.")
        return

    with open(SESSION_HISTORY_PATH, "r") as f:
        history = json.load(f)

    print(f"\n--- Past Focus Sessions ({len(history)}) ---")
    for s in history[-15:]:
        print(f"  {s['timestamp']}  |  {s['total_minutes']}min total  |  "
              f"{s['focus_percent']}% focused  |  {s['distraction_events']} distractions")
        print(f"      \"{s['summary']}\"")


# --- Overlay ---
def draw_overlay(frame, status, elapsed_focused, elapsed_distracted, blink_count):
    color = (0, 200, 0) if status == "Focused" else (0, 140, 255)
    cv2.rectangle(frame, (0, 0), (330, 110), (245, 117, 16), -1)
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, color, 2)
    cv2.putText(frame, f"Focused: {elapsed_focused:.0f}s", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Distracted: {elapsed_distracted:.0f}s", (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def run_focus_tracker():
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam. Check the camera connection/permissions.")
        return

    session_start = time.time()
    current_status = "Focused"
    status_since = session_start

    focused_seconds = 0.0
    distracted_seconds = 0.0
    last_tick = session_start

    distraction_events = 0
    candidate_status = None
    candidate_since = None

    # Blink tracking
    blink_count = 0
    consec_closed_frames = 0
    eyes_closed_since = None

    print("Starting Reading/Study Focus Tracker.")
    print("Look at your screen naturally. Press 'q' or ESC to end the session and get a summary.")

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                min_detection_confidence=0.6,
                                min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_frame)

            now = time.time()
            frame_status = current_status  # default: carry over if no face

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark

                # --- Blink / eye-closure detection ---
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_EAR, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_EAR, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                eyes_closed = avg_ear < EAR_BLINK_THRESHOLD

                if eyes_closed:
                    consec_closed_frames += 1
                    if eyes_closed_since is None:
                        eyes_closed_since = now
                else:
                    if EAR_CONSEC_FRAMES_BLINK <= consec_closed_frames:
                        blink_count += 1
                    consec_closed_frames = 0
                    eyes_closed_since = None

                eyes_closed_duration = (now - eyes_closed_since) if eyes_closed_since else 0.0

                # --- Gaze direction ---
                left_h, left_v = gaze_ratios(
                    landmarks, LEFT_IRIS_CENTER, LEFT_EYE_LEFT, LEFT_EYE_RIGHT,
                    LEFT_EYE_TOP, LEFT_EYE_BOTTOM, w, h
                )
                right_h, right_v = gaze_ratios(
                    landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT,
                    RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, w, h
                )
                gaze_h = (left_h + right_h) / 2.0
                gaze_v = (left_v + right_v) / 2.0

                gaze_centered = (GAZE_H_MIN <= gaze_h <= GAZE_H_MAX) and (GAZE_V_MIN <= gaze_v <= GAZE_V_MAX)

                # Sustained eye closure (not a normal blink) also counts as distracted/drowsy
                is_looking_away_or_drowsy = (not gaze_centered) or (eyes_closed_duration >= EYES_CLOSED_DISTRACTION_SEC)

                frame_status = "Distracted" if is_looking_away_or_drowsy else "Focused"

            else:
                frame_status = "Distracted"  # no face in frame = not reading/studying

            # --- Debounce status changes so brief glances don't flip the state ---
            if frame_status != current_status:
                if candidate_status != frame_status:
                    candidate_status = frame_status
                    candidate_since = now

                required_hold = DISTRACTION_CONFIRM_SEC if frame_status == "Distracted" else FOCUS_CONFIRM_SEC
                if now - candidate_since >= required_hold:
                    if frame_status == "Distracted":
                        distraction_events += 1
                    current_status = frame_status
                    status_since = now
                    candidate_status = None
            else:
                candidate_status = None

            # --- Accumulate elapsed time per status ---
            dt = now - last_tick
            if current_status == "Focused":
                focused_seconds += dt
            else:
                distracted_seconds += dt
            last_tick = now

            draw_overlay(frame, current_status, focused_seconds, distracted_seconds, blink_count)

            cv2.imshow("Reading/Study Focus Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # --- End-of-session summary ---
    total_seconds = focused_seconds + distracted_seconds
    if total_seconds <= 0:
        print("Session too short to summarize.")
        return

    total_minutes = total_seconds / 60
    focused_minutes = focused_seconds / 60
    distracted_minutes = distracted_seconds / 60
    focus_percent = (focused_seconds / total_seconds) * 100
    blinks_per_minute = blink_count / total_minutes if total_minutes > 0 else 0

    stats = {
        "total_minutes": total_minutes,
        "focused_minutes": focused_minutes,
        "distracted_minutes": distracted_minutes,
        "focus_percent": focus_percent,
        "distraction_events": distraction_events,
        "blinks_per_minute": blinks_per_minute
    }

    print(f"\nSession ended: {total_minutes:.1f} min total, {focus_percent:.0f}% focused, "
          f"{distraction_events} distraction episodes, {blinks_per_minute:.1f} blinks/min.")
    print("Asking Gemma for a session summary...")

    summary = ask_gemma_summary(stats)
    print(f"\nGemma's summary: {summary}")

    save_session(stats, summary)


# --- Entry Point ---
if __name__ == "__main__":
    while True:
        print("1. Start a focus session")
        print("2. View past session history")
        choice = input("Choose an option (1/2): ").strip()

        if choice == "1":
            run_focus_tracker()
            break
        elif choice == "2":
            view_session_history()
            break
        else:
            print(f"Invalid choice: '{choice}'. Please type 1 or 2.\n")

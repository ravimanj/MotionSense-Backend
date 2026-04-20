# ============================================================
#  MotionSense AI — FastAPI + WebSocket Backend
#  Receives camera frames from the React Native app,
#  runs MediaPipe pose detection, returns landmarks +
#  joint angles + rep count + form feedback as JSON.
# ============================================================

import asyncio
import base64
import json
import logging
from typing import Dict

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ─── Logging ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App setup ──────────────────────────────────────────────
app = FastAPI(title="MotionSense AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MediaPipe ──────────────────────────────────────────────
mp_pose    = mp.solutions.pose
PoseLandmark = mp_pose.PoseLandmark

# ─── Angle thresholds per exercise ──────────────────────────
EXERCISE_CONFIG = {
    "bicep_curl": {
        "name":           "Bicep Curl",
        "down_threshold": 160,   # arm fully extended
        "up_threshold":   40,    # arm fully curled
        "start_stage":    "down",
    },
    "push_up": {
        "name":           "Push Up",
        "down_threshold": 160,   # arms straight (top)
        "up_threshold":   90,    # arms bent (bottom)
        "start_stage":    "up",
    },
    "squat": {
        "name":           "Squat",
        "down_threshold": 80,    # squat depth
        "up_threshold":   170,   # standing
        "start_stage":    "up",
    },
    "shoulder_press": {
        "name":           "Shoulder Press",
        "down_threshold": 160,   # arms overhead
        "up_threshold":   80,    # arms at shoulder level
        "start_stage":    "down",
    },
}

# ─── Per-connection session state ────────────────────────────
class SessionState:
    """Holds all rep-counting state for one connected client."""

    def __init__(self, exercise: str, reps_target: int, weight: float):
        cfg               = EXERCISE_CONFIG.get(exercise, EXERCISE_CONFIG["bicep_curl"])
        self.exercise     = exercise
        self.cfg          = cfg
        self.reps_target  = reps_target
        self.weight       = weight

        self.counter        = 0
        self.correct_reps   = 0
        self.incorrect_reps = 0
        self.form_errors    = set()

        # Full-cycle gate flags
        self.stage          = cfg["start_stage"]
        self.mid_reached    = False          # curl_reached / depth_reached / press_reached
        self.current_rep_error = False
        self.bad_form_frames   = 0
        BAD_FORM_THRESHOLD     = 10
        self.BAD_FORM_THRESHOLD = BAD_FORM_THRESHOLD

        # MediaPipe pose (one instance per connection → thread safe)
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )


# ─── Helper: angle between 3 landmarks ──────────────────────
def calculate_angle(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = float(np.abs(radians * 180.0 / np.pi))
    if angle > 180.0:
        angle = 360.0 - angle
    return round(angle, 1)


# ─── Helper: get landmark as [x, y] ─────────────────────────
def lm_xy(landmarks, idx):
    l = landmarks[idx]
    return [l.x, l.y]


# ─── Core pose analysis ──────────────────────────────────────
def analyse_frame(frame: np.ndarray, state: SessionState) -> dict:
    """
    Runs MediaPipe on one BGR frame.
    Updates session state in-place.
    Returns a JSON-serialisable result dict.
    """
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = state.pose.process(rgb)

    if not results.pose_landmarks:
        return {"detected": False}

    lm = results.pose_landmarks.landmark

    # ── Compute all joint angles ────────────────────────────
    left_shoulder  = lm_xy(lm, PoseLandmark.LEFT_SHOULDER)
    left_elbow     = lm_xy(lm, PoseLandmark.LEFT_ELBOW)
    left_wrist     = lm_xy(lm, PoseLandmark.LEFT_WRIST)

    right_shoulder = lm_xy(lm, PoseLandmark.RIGHT_SHOULDER)
    right_elbow    = lm_xy(lm, PoseLandmark.RIGHT_ELBOW)
    right_wrist    = lm_xy(lm, PoseLandmark.RIGHT_WRIST)

    right_hip      = lm_xy(lm, PoseLandmark.RIGHT_HIP)
    right_knee     = lm_xy(lm, PoseLandmark.RIGHT_KNEE)
    right_ankle    = lm_xy(lm, PoseLandmark.RIGHT_ANKLE)

    left_hip       = lm_xy(lm, PoseLandmark.LEFT_HIP)

    angles = {
        "left_elbow":   calculate_angle(left_shoulder,  left_elbow,  left_wrist),
        "right_elbow":  calculate_angle(right_shoulder, right_elbow, right_wrist),
        "right_knee":   calculate_angle(right_hip,      right_knee,  right_ankle),
        "right_hip":    calculate_angle(right_shoulder, right_hip,   right_knee),
        "left_hip":     calculate_angle(left_shoulder,  left_hip,    lm_xy(lm, PoseLandmark.LEFT_KNEE)),
    }

    # ── Select primary angle per exercise ───────────────────
    ex = state.exercise
    if ex == "bicep_curl":
        primary = angles["left_elbow"]
    elif ex == "push_up":
        primary = angles["right_elbow"]
    elif ex == "squat":
        primary = angles["right_knee"]
    elif ex == "shoulder_press":
        primary = angles["right_elbow"]
    else:
        primary = angles["right_elbow"]

    # ── Form checking ────────────────────────────────────────
    feedback      = "Good form! Keep going."
    feedback_type = "good"         # good | warning | error
    form_error    = None

    if ex == "bicep_curl":
        elbow_shift = abs(left_elbow[0] - left_shoulder[0])
        if elbow_shift > 0.06:
            form_error    = "arm swing"
            feedback      = "ARM SWING: Keep elbow fixed!"
            feedback_type = "error"
        else:
            feedback = "Good form! Keep going."

    elif ex == "push_up":
        hip_angle = angles["right_hip"]
        if hip_angle < 160:
            form_error    = "hip sag"
            feedback      = "HIP SAG: Keep your back straight!"
            feedback_type = "error"

    elif ex == "squat":
        if angles["right_hip"] < 95:
            form_error    = "forward lean"
            feedback      = "LEAN: Keep chest up!"
            feedback_type = "warning"

    elif ex == "shoulder_press":
        lean_dx = abs(lm[PoseLandmark.RIGHT_HIP].x - lm[PoseLandmark.RIGHT_SHOULDER].x)
        if lean_dx > 0.06:
            form_error    = "leaning"
            feedback      = "LEANING: Keep core tight!"
            feedback_type = "error"

    # Form error smoothing
    if form_error:
        state.bad_form_frames += 1
    else:
        state.bad_form_frames = 0

    if state.bad_form_frames >= state.BAD_FORM_THRESHOLD:
        state.current_rep_error = True
        state.form_errors.add(form_error)

    # ── Full-cycle rep counting ──────────────────────────────
    cfg           = state.cfg
    down_th       = cfg["down_threshold"]
    up_th         = cfg["up_threshold"]
    rep_completed = False

    if ex == "bicep_curl":
        # Cycle: extended (down) → curled (up) → extended (down)
        if primary > down_th and state.stage == "down":
            state.mid_reached = False
        if primary < up_th and state.stage == "down" and not state.mid_reached:
            state.stage       = "up"
            state.mid_reached = True
        if primary > down_th and state.stage == "up" and state.mid_reached:
            rep_completed = True
            state.stage   = "down"

    elif ex == "push_up":
        # Cycle: arms straight (up) → arms bent (down) → arms straight (up)
        if primary > down_th and state.stage == "up":
            state.mid_reached = False
        if primary < up_th and state.stage == "up" and not state.mid_reached:
            state.stage       = "down"
            state.mid_reached = True
        if primary > down_th and state.stage == "down" and state.mid_reached:
            rep_completed = True
            state.stage   = "up"

    elif ex == "squat":
        # Cycle: standing (up) → squat depth (down) → standing (up)
        if primary > up_th and state.stage == "up":
            state.mid_reached     = False
            state.current_rep_error = False
        if primary < down_th and state.stage == "up" and not state.mid_reached:
            state.stage       = "down"
            state.mid_reached = True
        if primary > up_th and state.stage == "down" and state.mid_reached:
            rep_completed = True
            state.stage   = "up"

    elif ex == "shoulder_press":
        # Cycle: arms low (down) → overhead (up) → arms low (down)
        if primary < up_th and state.stage == "down":
            state.mid_reached = False
        if primary > down_th and state.stage == "down" and not state.mid_reached:
            state.stage       = "up"
            state.mid_reached = True
        if primary < up_th and state.stage == "up" and state.mid_reached:
            rep_completed = True
            state.stage   = "down"

    if rep_completed:
        state.counter += 1
        if state.current_rep_error:
            state.incorrect_reps += 1
        else:
            state.correct_reps += 1
        state.current_rep_error = False
        state.bad_form_frames   = 0
        state.mid_reached       = False

    # ── Serialize landmarks for the app to draw ──────────────
    landmark_list = [
        {"x": round(l.x, 4), "y": round(l.y, 4), "z": round(l.z, 4),
         "visibility": round(l.visibility, 3)}
        for l in lm
    ]

    # ── Accuracy ─────────────────────────────────────────────
    total    = state.correct_reps + state.incorrect_reps
    accuracy = round((state.correct_reps / total) * 100) if total > 0 else 0

    return {
        "detected":       True,
        "landmarks":      landmark_list,
        "angles":         angles,
        "primary_angle":  primary,
        "stage":          state.stage,
        "counter":        state.counter,
        "correct_reps":   state.correct_reps,
        "incorrect_reps": state.incorrect_reps,
        "accuracy":       accuracy,
        "reps_target":    state.reps_target,
        "rep_completed":  rep_completed,
        "target_reached": state.counter >= state.reps_target,
        "feedback":       feedback,
        "feedback_type":  feedback_type,
        "form_errors":    list(state.form_errors),
        "exercise_name":  cfg["name"],
        "weight":         state.weight,
    }


# ─── REST: health check ──────────────────────────────────────
@app.get("/")
async def health():
    return {"status": "ok", "service": "MotionSense AI API"}


@app.get("/exercises")
async def list_exercises():
    return {
        key: {
            "name":           cfg["name"],
            "down_threshold": cfg["down_threshold"],
            "up_threshold":   cfg["up_threshold"],
        }
        for key, cfg in EXERCISE_CONFIG.items()
    }


# ─── WebSocket: main workout session ────────────────────────
@app.websocket("/ws/{exercise}")
async def websocket_endpoint(
    websocket: WebSocket,
    exercise:  str,
    reps:      int   = 10,
    weight:    float = 0.0,
):
    """
    WebSocket endpoint.
    URL:  ws://your-server/ws/bicep_curl?reps=10&weight=5
    Client sends: base64-encoded JPEG frames as text messages.
    Server sends: JSON result objects.
    """
    if exercise not in EXERCISE_CONFIG:
        await websocket.close(code=4000, reason=f"Unknown exercise: {exercise}")
        return

    await websocket.accept()
    logger.info(f"Client connected — exercise={exercise}, reps={reps}, weight={weight}")

    state = SessionState(exercise, reps, weight)

    try:
        while True:
            # Receive base64 frame from React Native
            raw = await websocket.receive_text()

            # Strip data-URI prefix if present: "data:image/jpeg;base64,..."
            if "," in raw:
                raw = raw.split(",", 1)[1]

            # Decode → numpy array → BGR image
            try:
                img_bytes = base64.b64decode(raw)
                nparr     = np.frombuffer(img_bytes, np.uint8)
                frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("imdecode returned None")
            except Exception as e:
                await websocket.send_text(
                    json.dumps({"error": f"Frame decode failed: {str(e)}"})
                )
                continue

            # Run pose analysis
            result = analyse_frame(frame, state)

            # Send result back to app
            await websocket.send_text(json.dumps(result))

            # Session complete
            if result.get("target_reached"):
                logger.info(f"Target reached — {state.counter} reps completed")
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    finally:
        state.pose.close()
        logger.info(f"Session ended — counter={state.counter}")


# ─── Run locally ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

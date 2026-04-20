# ============================================================
#  MotionSense AI — FastAPI + WebSocket Backend
#  Fixed for Render.com deployment
#  Uses mediapipe.tasks API (works on mediapipe >= 0.10.x)
# ============================================================

import base64
import json
import logging

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ─── MediaPipe Tasks API (works on all Python 3.8–3.12) ─────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import landmark as mp_landmark

# Landmark index constants (same as before)
# We define them manually since mp.solutions may not exist
# on newer mediapipe builds
class PL:
    NOSE           = 0
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW     = 13
    RIGHT_ELBOW    = 14
    LEFT_WRIST     = 15
    RIGHT_WRIST    = 16
    LEFT_HIP       = 23
    RIGHT_HIP      = 24
    LEFT_KNEE      = 25
    RIGHT_KNEE     = 26
    LEFT_ANKLE     = 27
    RIGHT_ANKLE    = 28

# MediaPipe POSE_CONNECTIONS for drawing skeleton (index pairs)
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),
]

# ─── Logging ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App ────────────────────────────────────────────────────
app = FastAPI(title="MotionSense AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Exercise config ─────────────────────────────────────────
EXERCISE_CONFIG = {
    "bicep_curl": {
        "name":           "Bicep Curl",
        "down_threshold": 160,
        "up_threshold":   40,
        "start_stage":    "down",
    },
    "push_up": {
        "name":           "Push Up",
        "down_threshold": 160,
        "up_threshold":   90,
        "start_stage":    "up",
    },
    "squat": {
        "name":           "Squat",
        "down_threshold": 80,
        "up_threshold":   170,
        "start_stage":    "up",
    },
    "shoulder_press": {
        "name":           "Shoulder Press",
        "down_threshold": 160,
        "up_threshold":   80,
        "start_stage":    "down",
    },
}

# ─── Build PoseLandmarker options ────────────────────────────
def _make_landmarker():
    """
    Creates a MediaPipe PoseLandmarker using the Tasks API.
    Downloads the model from the official MediaPipe CDN.
    """
    import urllib.request, os, tempfile

    model_url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task"
    )
    model_path = os.path.join(tempfile.gettempdir(), "pose_landmarker_lite.task")

    if not os.path.exists(model_path):
        logger.info("Downloading MediaPipe pose model...")
        urllib.request.urlretrieve(model_url, model_path)
        logger.info("Model downloaded.")

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        num_poses=1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


# ─── Session state ───────────────────────────────────────────
class SessionState:
    def __init__(self, exercise: str, reps_target: int, weight: float):
        cfg = EXERCISE_CONFIG.get(exercise, EXERCISE_CONFIG["bicep_curl"])
        self.exercise           = exercise
        self.cfg                = cfg
        self.reps_target        = reps_target
        self.weight             = weight
        self.counter            = 0
        self.correct_reps       = 0
        self.incorrect_reps     = 0
        self.form_errors        = set()
        self.stage              = cfg["start_stage"]
        self.mid_reached        = False
        self.current_rep_error  = False
        self.bad_form_frames    = 0
        self.BAD_FORM_THRESHOLD = 10
        # Each session gets its own landmarker instance
        self.landmarker = _make_landmarker()

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


# ─── Helpers ─────────────────────────────────────────────────
def calculate_angle(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = (np.arctan2(c[1]-b[1], c[0]-b[0]) -
               np.arctan2(a[1]-b[1], a[0]-b[0]))
    angle = float(np.abs(radians * 180.0 / np.pi))
    if angle > 180.0:
        angle = 360.0 - angle
    return round(angle, 1)


def lm_xy(lm_list, idx):
    l = lm_list[idx]
    return [l.x, l.y]


# ─── Core analysis ───────────────────────────────────────────
def analyse_frame(frame: np.ndarray, state: SessionState) -> dict:
    # Convert BGR → RGB → MediaPipe Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = state.landmarker.detect(mp_image)

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return {"detected": False}

    lm = result.pose_landmarks[0]   # first (only) person

    # ── Angles ───────────────────────────────────────────────
    left_shoulder  = lm_xy(lm, PL.LEFT_SHOULDER)
    left_elbow     = lm_xy(lm, PL.LEFT_ELBOW)
    left_wrist     = lm_xy(lm, PL.LEFT_WRIST)
    right_shoulder = lm_xy(lm, PL.RIGHT_SHOULDER)
    right_elbow    = lm_xy(lm, PL.RIGHT_ELBOW)
    right_wrist    = lm_xy(lm, PL.RIGHT_WRIST)
    right_hip      = lm_xy(lm, PL.RIGHT_HIP)
    right_knee     = lm_xy(lm, PL.RIGHT_KNEE)
    right_ankle    = lm_xy(lm, PL.RIGHT_ANKLE)
    left_hip       = lm_xy(lm, PL.LEFT_HIP)
    left_knee      = lm_xy(lm, PL.LEFT_KNEE)

    angles = {
        "left_elbow":  calculate_angle(left_shoulder,  left_elbow,  left_wrist),
        "right_elbow": calculate_angle(right_shoulder, right_elbow, right_wrist),
        "right_knee":  calculate_angle(right_hip,      right_knee,  right_ankle),
        "right_hip":   calculate_angle(right_shoulder, right_hip,   right_knee),
        "left_hip":    calculate_angle(left_shoulder,  left_hip,    left_knee),
    }

    ex = state.exercise
    primary = {
        "bicep_curl":     angles["left_elbow"],
        "push_up":        angles["right_elbow"],
        "squat":          angles["right_knee"],
        "shoulder_press": angles["right_elbow"],
    }.get(ex, angles["right_elbow"])

    # ── Form checking ────────────────────────────────────────
    feedback      = "Good form! Keep going."
    feedback_type = "good"
    form_error    = None

    if ex == "bicep_curl":
        elbow_shift = abs(lm[PL.LEFT_ELBOW].x - lm[PL.LEFT_SHOULDER].x)
        if elbow_shift > 0.06:
            form_error, feedback, feedback_type = "arm swing", "ARM SWING: Keep elbow fixed!", "error"

    elif ex == "push_up":
        if angles["right_hip"] < 160:
            form_error, feedback, feedback_type = "hip sag", "HIP SAG: Keep your back straight!", "error"

    elif ex == "squat":
        if angles["right_hip"] < 95:
            form_error, feedback, feedback_type = "forward lean", "LEAN: Keep chest up!", "warning"

    elif ex == "shoulder_press":
        lean_dx = abs(lm[PL.RIGHT_HIP].x - lm[PL.RIGHT_SHOULDER].x)
        if lean_dx > 0.06:
            form_error, feedback, feedback_type = "leaning", "LEANING: Keep core tight!", "error"

    # Form smoothing
    if form_error:
        state.bad_form_frames += 1
    else:
        state.bad_form_frames = 0
    if state.bad_form_frames >= state.BAD_FORM_THRESHOLD:
        state.current_rep_error = True
        state.form_errors.add(form_error)

    # ── Full-cycle rep counting ───────────────────────────────
    cfg    = state.cfg
    down_th, up_th = cfg["down_threshold"], cfg["up_threshold"]
    rep_completed = False

    if ex == "bicep_curl":
        if primary > down_th and state.stage == "down":
            state.mid_reached = False
        if primary < up_th and state.stage == "down" and not state.mid_reached:
            state.stage, state.mid_reached = "up", True
        if primary > down_th and state.stage == "up" and state.mid_reached:
            rep_completed, state.stage = True, "down"

    elif ex == "push_up":
        if primary > down_th and state.stage == "up":
            state.mid_reached = False
        if primary < up_th and state.stage == "up" and not state.mid_reached:
            state.stage, state.mid_reached = "down", True
        if primary > down_th and state.stage == "down" and state.mid_reached:
            rep_completed, state.stage = True, "up"

    elif ex == "squat":
        if primary > up_th and state.stage == "up":
            state.mid_reached, state.current_rep_error = False, False
        if primary < down_th and state.stage == "up" and not state.mid_reached:
            state.stage, state.mid_reached = "down", True
        if primary > up_th and state.stage == "down" and state.mid_reached:
            rep_completed, state.stage = True, "up"

    elif ex == "shoulder_press":
        if primary < up_th and state.stage == "down":
            state.mid_reached = False
        if primary > down_th and state.stage == "down" and not state.mid_reached:
            state.stage, state.mid_reached = "up", True
        if primary < up_th and state.stage == "up" and state.mid_reached:
            rep_completed, state.stage = True, "down"

    if rep_completed:
        state.counter += 1
        if state.current_rep_error:
            state.incorrect_reps += 1
        else:
            state.correct_reps += 1
        state.current_rep_error = False
        state.bad_form_frames   = 0
        state.mid_reached       = False

    # ── Serialize landmarks ───────────────────────────────────
    landmark_list = [
        {
            "x": round(l.x, 4),
            "y": round(l.y, 4),
            "z": round(l.z, 4),
            "visibility": round(l.visibility if hasattr(l, "visibility") else 1.0, 3),
        }
        for l in lm
    ]

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


# ─── REST endpoints ──────────────────────────────────────────
@app.get("/")
async def health():
    return {"status": "ok", "service": "MotionSense AI API", "version": "2.0.0"}


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


# ─── WebSocket endpoint ──────────────────────────────────────
@app.websocket("/ws/{exercise}")
async def websocket_endpoint(
    websocket: WebSocket,
    exercise:  str,
    reps:      int   = 10,
    weight:    float = 0.0,
):
    if exercise not in EXERCISE_CONFIG:
        await websocket.close(code=4000, reason=f"Unknown exercise: {exercise}")
        return

    await websocket.accept()
    logger.info(f"Client connected — exercise={exercise}, reps={reps}, weight={weight}")

    state = SessionState(exercise, reps, weight)

    try:
        while True:
            raw = await websocket.receive_text()

            # Strip data-URI prefix if present
            if "," in raw:
                raw = raw.split(",", 1)[1]

            try:
                img_bytes = base64.b64decode(raw)
                nparr     = np.frombuffer(img_bytes, np.uint8)
                frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("imdecode returned None")
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"Frame decode failed: {e}"}))
                continue

            result = analyse_frame(frame, state)
            await websocket.send_text(json.dumps(result))

            if result.get("target_reached"):
                logger.info(f"Target reached — {state.counter} reps")
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    finally:
        state.close()
        logger.info(f"Session ended — counter={state.counter}")


# ─── Run locally ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

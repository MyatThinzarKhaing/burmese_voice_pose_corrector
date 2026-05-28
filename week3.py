import asyncio
import math
import os
import queue
import sys
import tempfile
import threading
import time
import types

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import edge_tts
import numpy as np
import pygame
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from google.protobuf import descriptor as protobuf_descriptor
from google.protobuf import message_factory
from google.protobuf import symbol_database

if not hasattr(protobuf_descriptor.FieldDescriptor, "label"):
    protobuf_descriptor.FieldDescriptor.label = property(
        lambda self: self._label,
        lambda self, value: setattr(self, "_label", value),
    )

if not hasattr(message_factory.MessageFactory, "GetPrototype"):
    message_factory.MessageFactory.GetPrototype = staticmethod(message_factory.GetMessageClass)

if not hasattr(symbol_database.SymbolDatabase, "GetPrototype"):
    symbol_database.SymbolDatabase.GetPrototype = lambda self, descriptor: message_factory.GetMessageClass(descriptor)


def _doc_control_passthrough(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(obj):
        return obj

    return decorator


tensorflow_stub = types.ModuleType("tensorflow")
tensorflow_tools_stub = types.ModuleType("tensorflow.tools")
tensorflow_docs_stub = types.ModuleType("tensorflow.tools.docs")
doc_controls_stub = types.ModuleType("tensorflow.tools.docs.doc_controls")
doc_controls_stub.do_not_generate_docs = _doc_control_passthrough
doc_controls_stub.do_not_doc_inheritable = _doc_control_passthrough
doc_controls_stub.for_subclass_implementers = _doc_control_passthrough
doc_controls_stub.__getattr__ = lambda name: _doc_control_passthrough
tensorflow_stub.tools = tensorflow_tools_stub
tensorflow_tools_stub.docs = tensorflow_docs_stub
tensorflow_docs_stub.doc_controls = doc_controls_stub
sys.modules.setdefault("tensorflow", tensorflow_stub)
sys.modules.setdefault("tensorflow.tools", tensorflow_tools_stub)
sys.modules.setdefault("tensorflow.tools.docs", tensorflow_docs_stub)
sys.modules.setdefault("tensorflow.tools.docs.doc_controls", doc_controls_stub)

import mediapipe as mp


app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

state_lock = threading.RLock()
pose_lock = threading.Lock()
audio_queue = queue.Queue(maxsize=2)

counter = 0
stage = "paused"
angle = 0
is_paused = True
last_spoken_at = {}

BURMESE_VOICE = "my-MM-NilarNeural"
VOICE_TEXT = {
    "Good Rep!": "အနေအထား ကောင်းမွန်သည်",
    "Keep your body straight": "ကိုယ်ကိုတည့်တည့်ထားပါ",
    "Lower your body": "ပိုနိမ့်ပါ",
    "Too low": "အရမ်းမနိမ့်ပါနဲ့",
}


def make_payload(feedback=None):
    with state_lock:
        return {
            "counter": int(counter),
            "angle": int(angle),
            "stage": stage,
            "feedback": feedback or [],
            "paused": bool(is_paused),
        }


def calculate_angle(a, b, c):
    x1, y1 = a[:2]
    x2, y2 = b[:2]
    x3, y3 = c[:2]
    raw_angle = math.degrees(
        math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
    )

    raw_angle = abs(raw_angle)
    if raw_angle > 180:
        raw_angle = 360 - raw_angle

    return raw_angle


def init_audio():
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        return True
    except pygame.error as exc:
        print(f"Audio disabled: {exc}")
        return False


AUDIO_READY = init_audio()


async def speak_async(text):
    if not AUDIO_READY:
        return

    fd, filename = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)

    try:
        communicate = edge_tts.Communicate(text=text, voice=BURMESE_VOICE)
        await communicate.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.05)

        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        await asyncio.sleep(0.1)
    except Exception as exc:
        print(f"TTS skipped: {exc}")
    finally:
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except OSError:
            pass


def audio_worker():
    while True:
        text = audio_queue.get()
        try:
            asyncio.run(speak_async(text))
        finally:
            audio_queue.task_done()


def queue_voice(feedback_key, cooldown=3.0):
    now = time.time()
    with state_lock:
        previous = last_spoken_at.get(feedback_key, 0)
        if now - previous < cooldown:
            return
        last_spoken_at[feedback_key] = now

    try:
        audio_queue.put_nowait(VOICE_TEXT[feedback_key])
    except queue.Full:
        pass


def extract_landmarks(results, width, height):
    if not results.pose_landmarks:
        return []

    return [
        (int(landmark.x * width), int(landmark.y * height), landmark.z)
        for landmark in results.pose_landmarks.landmark
    ]


def evaluate_pushup(landmarks):
    global angle, counter, stage

    feedback = []
    voice_events = []

    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

    current_angle = int(calculate_angle(shoulder, elbow, wrist))
    body_alignment = abs(((shoulder[1] + ankle[1]) / 2) - hip[1])

    with state_lock:
        angle = current_angle

        if current_angle < 95:
            stage = "down"

        if current_angle > 155 and stage == "down":
            stage = "up"
            counter += 1
            feedback.append("Good Rep!")
            voice_events.append(("Good Rep!", 1.8))

        if body_alignment > 44:
            feedback.append("Keep your body straight")
            voice_events.append(("Keep your body straight", 3.5))

        if current_angle > 168 and stage == "up":
            feedback.append("Lower your body")
            voice_events.append(("Lower your body", 3.0))
        elif current_angle < 55:
            feedback.append("Too low")
            voice_events.append(("Too low", 3.0))

        payload = make_payload(feedback)

    for key, cooldown in voice_events:
        queue_voice(key, cooldown=cooldown)

    return payload


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    global angle, counter, is_paused, stage

    with state_lock:
        counter = 0
        angle = 0
        stage = "up"
        is_paused = False
        last_spoken_at.clear()

    return jsonify({"status": "started"})


@app.route("/pause", methods=["POST"])
def pause():
    global is_paused, stage

    with state_lock:
        is_paused = True
        stage = "paused"

    return jsonify({"status": "paused"})


@app.route("/resume", methods=["POST"])
def resume():
    global is_paused, stage

    with state_lock:
        is_paused = False
        if stage == "paused":
            stage = "up"

    return jsonify({"status": "resumed"})


@app.route("/analyze", methods=["POST"])
def analyze():
    with state_lock:
        paused_now = is_paused

    if paused_now:
        return jsonify(make_payload())

    uploaded_frame = request.files.get("frame")
    if uploaded_frame is None:
        return jsonify({"error": "Missing form-data file field: frame", **make_payload()}), 400

    frame_data = np.frombuffer(uploaded_frame.read(), dtype=np.uint8)
    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Unable to decode image frame", **make_payload()}), 400

    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False

    try:
        with pose_lock:
            results = pose.process(rgb_frame)
    except Exception as exc:
        print(f"MediaPipe processing error: {exc}")
        return jsonify({"error": "Pose processing failed", **make_payload()}), 500

    landmarks = extract_landmarks(results, width, height)
    if not landmarks:
        return jsonify(make_payload())

    return jsonify(evaluate_pushup(landmarks))


if __name__ == "__main__":
    threading.Thread(target=audio_worker, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True, use_reloader=False)

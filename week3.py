import asyncio
import base64
import binascii
import math
import os
import re
import sys
import threading
import types

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import edge_tts
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
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
pose_lock = threading.Lock()
state_lock = threading.Lock()

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

BURMESE_VOICE = "my-MM-NilarNeural"

DOWN_ANGLE_THRESHOLD = 115
UP_ANGLE_THRESHOLD = 140
TOO_LOW_ANGLE_THRESHOLD = 45

counter = 0
stage = "up"
last_angle = 0
tracking_active = False
last_feedback_spoken = {}


def calculate_angle(a, b, c):
    ax, ay = a[:2]
    bx, by = b[:2]
    cx, cy = c[:2]

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)
    ba_length = math.hypot(*ba)
    bc_length = math.hypot(*bc)

    if ba_length == 0 or bc_length == 0:
        return 0.0

    cosine = ((ba[0] * bc[0]) + (ba[1] * bc[1])) / (ba_length * bc_length)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def point_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_from_line(point, line_start, line_end):
    x0, y0 = point[:2]
    x1, y1 = line_start[:2]
    x2, y2 = line_end[:2]

    line_length = math.hypot(x2 - x1, y2 - y1)
    if line_length == 0:
        return 0.0

    return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / line_length


def current_snapshot(feedback=None):
    return {
        "counter": int(counter),
        "angle": int(last_angle),
        "stage": stage,
        "feedback": feedback or [],
    }


def landmarks_from_results(results, width, height):
    if not results.pose_landmarks:
        return []

    return [
        (int(lm.x * width), int(lm.y * height), lm.z, lm.visibility)
        for lm in results.pose_landmarks.landmark
    ]


def is_arm_visible(landmarks):
    required = (
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
    )
    return all(landmarks[index][3] >= 0.45 for index in required)


def is_body_visible(landmarks):
    required = (
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
    )
    return all(landmarks[index][3] >= 0.45 for index in required)


def is_horizontal_pushup_position(shoulder, hip, ankle):
    body_length = max(point_distance(shoulder, ankle), 1.0)
    shoulder_hip_gap = abs(shoulder[1] - hip[1])
    hip_ankle_gap = abs(hip[1] - ankle[1])

    shoulder_hip_aligned = shoulder_hip_gap <= body_length * 0.22
    hip_ankle_not_standing = hip_ankle_gap <= body_length * 0.32
    body_has_horizontal_span = abs(shoulder[0] - ankle[0]) >= body_length * 0.45

    return shoulder_hip_aligned and hip_ankle_not_standing and body_has_horizontal_span


def body_alignment_feedback(shoulder, hip, ankle):
    body_length = max(point_distance(shoulder, ankle), 1.0)
    hip_line_distance = distance_from_line(hip, shoulder, ankle)
    return hip_line_distance > body_length * 0.12


def detect_pushup(landmarks):
    global counter, stage, last_angle

    with state_lock:
        if not is_arm_visible(landmarks):
            return current_snapshot([])

        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        angle = calculate_angle(shoulder, elbow, wrist)
        last_angle = int(angle)

        feedback = []

        if is_body_visible(landmarks):
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            horizontal_position = is_horizontal_pushup_position(shoulder, hip, ankle)
            alignment_problem = body_alignment_feedback(shoulder, hip, ankle)

            if not horizontal_position or alignment_problem:
                feedback.append("Keep your body straight")

        if angle < TOO_LOW_ANGLE_THRESHOLD:
            feedback.append("Too low")

        if angle < DOWN_ANGLE_THRESHOLD:
            stage = "down"

        if angle > UP_ANGLE_THRESHOLD and stage == "down":
            stage = "up"
            counter += 1
            feedback.append("Good Rep!")

        return current_snapshot(feedback)


def decode_frame_from_request():
    uploaded_frame = request.files.get("frame")
    if uploaded_frame is not None:
        return uploaded_frame.read()

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        encoded = payload.get("frame") or payload.get("image") or payload.get("data")
        if not encoded:
            return None

        if isinstance(encoded, str):
            encoded = re.sub(r"^data:image/[^;]+;base64,", "", encoded)
            try:
                return base64.b64decode(encoded, validate=True)
            except binascii.Error:
                return None

        return None

    if request.data:
        return request.data

    return None


def decode_jpeg_frame(frame_bytes):
    if not frame_bytes:
        return None

    frame_array = np.frombuffer(frame_bytes, np.uint8)
    return cv2.imdecode(frame_array, cv2.IMREAD_COLOR)


async def tts_chunk_generator(text):
    communicate = edge_tts.Communicate(text=text, voice=BURMESE_VOICE)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


def stream_tts_audio(text):
    loop = asyncio.new_event_loop()
    generator = tts_chunk_generator(text)

    try:
        while True:
            try:
                yield loop.run_until_complete(generator.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start_tracking():
    global counter, stage, last_angle, tracking_active, last_feedback_spoken

    with state_lock:
        counter = 0
        stage = "up"
        last_angle = 0
        tracking_active = True
        last_feedback_spoken = {}

    return jsonify(current_snapshot())


@app.route("/pause", methods=["POST"])
def pause_tracking():
    global tracking_active

    with state_lock:
        tracking_active = False
        snapshot = current_snapshot()

    return jsonify(snapshot)


@app.route("/resume", methods=["POST"])
def resume_tracking():
    global tracking_active

    with state_lock:
        tracking_active = True
        snapshot = current_snapshot()

    return jsonify(snapshot)


@app.route("/tts", methods=["GET"])
def tts():
    text = request.args.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing text query parameter"}), 400

    headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    return Response(
        stream_with_context(stream_tts_audio(text)),
        headers=headers,
        mimetype="audio/mpeg",
        direct_passthrough=True,
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    frame_bytes = decode_frame_from_request()
    frame = decode_jpeg_frame(frame_bytes)

    if frame is None:
        return jsonify({"error": "Unable to decode image frame"}), 400

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with pose_lock:
        results = pose.process(rgb_frame)

    landmarks = landmarks_from_results(results, width, height)

    if not landmarks:
        with state_lock:
            return jsonify(current_snapshot())

    return jsonify(detect_pushup(landmarks))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True, use_reloader=False)
